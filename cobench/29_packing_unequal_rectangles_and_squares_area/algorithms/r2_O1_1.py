"""Separation-graph coupled pushing and min-cut recombination packing
Generates layouts by contact construction and removal-refill, propagates the separation graph to shift rectangles jointly so new items can be inserted, recombines via min-cut, and keeps the best feasible solution after item-by-item checks.
"""
import math
import random
import time


def _recombine(left, right, values, gap, stop):
    """Weighted independent-set recombination of two internally feasible layouts.

    Overlapping and duplicated items form a bipartite conflict graph; a min-cut yields the maximum-area compatible subset.
    Uses a non-recursive flow algorithm; on timeout keeps the parent layout with larger area.
    """
    lv = math.fsum(values[p[0]] for p in left)
    rv = math.fsum(values[p[0]] for p in right)
    fallback = left if lv >= rv else right
    if not left or not right or time.perf_counter() >= stop:
        return fallback
    left_ids = {p[0] for p in left}
    upper = lv + math.fsum(values[p[0]] for p in right if p[0] not in left_ids)
    if upper <= max(lv, rv):
        return fallback

    # Pre-fix positions identical in both parent layouts to shrink the flow network for the local changes.
    lookup = {p[0]: p for p in right}
    common = [p for p in left if lookup.get(p[0]) == p]
    shared = {p[0] for p in common}
    aa = [p for p in left if p[0] not in shared]
    bb = [p for p in right if p[0] not in shared]
    na, nb = len(aa), len(bb)
    if not na or not nb:
        return fallback
    conflicts = [set() for _ in aa]
    right_ids = {p[0]: j for j, p in enumerate(bb)}
    for i, p in enumerate(aa):
        j = right_ids.get(p[0])
        if j is not None:
            conflicts[i].add(j)

    # Scan rectangle boundaries to build cross-layout conflicts; the conservative inflation is smaller than the gap reserved during construction.
    pad = gap * 0.125
    events = []
    for side, records in enumerate((aa, bb)):
        for i, p in enumerate(records):
            if i % 64 == 0 and time.perf_counter() >= stop:
                return fallback
            events.append((p[1] - p[3] - pad, 1, side, i))
            events.append((p[1] + p[3] + pad, 0, side, i))
    events.sort()
    active = ({}, {})
    for event_no, (_, entering, side, i) in enumerate(events):
        if event_no % 64 == 0 and time.perf_counter() >= stop:
            return fallback
        if not entering:
            active[side].pop(i, None)
            continue
        p = (aa, bb)[side][i]
        for k, (j, q) in enumerate(active[1 - side].items()):
            if k % 64 == 0 and time.perf_counter() >= stop:
                return fallback
            if (abs(p[2] - q[2]) < p[4] + q[4] + 2.0 * pad and
                    abs(p[1] - q[1]) < p[3] + q[3] + 2.0 * pad):
                if side:
                    conflicts[j].add(i)
                else:
                    conflicts[i].add(j)
        active[side][i] = p

    source, sink = na + nb, na + nb + 1
    count = sink + 1
    adjacency = [[] for _ in range(count)]
    targets, capacity = [], []

    def edge(u, v, cap):
        e = len(targets)
        adjacency[u].append(e)
        adjacency[v].append(e + 1)
        targets.extend((v, u))
        capacity.extend((cap, 0.0))

    infinite = lv + rv + 1.0
    for i, p in enumerate(aa):
        if time.perf_counter() >= stop:
            return fallback
        edge(source, i, values[p[0]])
        for k, j in enumerate(conflicts[i]):
            if k % 64 == 0 and time.perf_counter() >= stop:
                return fallback
            edge(i, na + j, infinite)
    for j, p in enumerate(bb):
        edge(na + j, sink, values[p[0]])

    # Non-recursive layered augmentation, independent of the interpreter's recursion depth.
    while time.perf_counter() < stop:
        level = [-1] * count
        level[source] = 0
        queue = [source]
        for k, u in enumerate(queue):
            if k % 64 == 0 and time.perf_counter() >= stop:
                return fallback
            for j, e in enumerate(adjacency[u]):
                if j % 128 == 0 and time.perf_counter() >= stop:
                    return fallback
                v = targets[e]
                if capacity[e] > 0.0 and level[v] < 0:
                    level[v] = level[u] + 1
                    queue.append(v)
        if level[sink] < 0:
            # Recover the compatible rectangles from the complement of the minimum vertex cover.
            result = common + [p for i, p in enumerate(aa) if level[i] >= 0]
            result += [p for j, p in enumerate(bb) if level[na + j] < 0]
            score = math.fsum(values[p[0]] for p in result)
            return result if score > max(lv, rv) else fallback
        cursor = [0] * count
        nodes, path = [source], []
        operations = 0
        while nodes:
            operations += 1
            if operations % 64 == 0 and time.perf_counter() >= stop:
                return fallback
            u = nodes[-1]
            if u == sink:
                flow = min(capacity[e] for e in path)
                blocked = None
                for k, e in enumerate(path):
                    if k % 128 == 0 and time.perf_counter() >= stop:
                        return fallback
                    capacity[e] -= flow
                    capacity[e ^ 1] += flow
                    if capacity[e] <= 0.0 and blocked is None:
                        blocked = k
                nodes = nodes[:blocked + 1]
                path = path[:blocked]
                cursor[nodes[-1]] += 1
            elif cursor[u] >= len(adjacency[u]):
                level[u] = -1
                nodes.pop()
                if path:
                    path.pop()
                    cursor[nodes[-1]] += 1
            else:
                e = adjacency[u][cursor[u]]
                v = targets[e]
                if capacity[e] > 0.0 and level[v] == level[u] + 1:
                    path.append(e)
                    nodes.append(v)
                else:
                    cursor[u] += 1
    return fallback


def _frontier(packed, direction, stop):
    sx, sy, _ = direction
    horizontal, vertical = [], []
    for k, (idx, x, y, a, b, _) in enumerate(packed):
        if k % 64 == 0 and time.perf_counter() >= stop:
            return None
        x, y = sx * x, sy * y
        horizontal.append((x - a, x + a, y, b, idx))
        vertical.append((y - b, y + b, x, a, idx))
    horizontal.sort()
    vertical.sort()
    return horizontal, vertical


def _events(a, b, frontier, slope, rho, gap, stop):
    """Circle-boundary and rectangle contact events are used for both direct insertion and minimum-loss removal."""
    if math.hypot(a, b) > rho:
        return []
    xmax = max(0.0, math.sqrt(max(0.0, (rho - b) * (rho + b))) - a)
    ymax = max(0.0, math.sqrt(max(0.0, (rho - a) * (rho + a))) - b)
    xs, ys = {0.0, -xmax, xmax}, {0.0, -ymax, ymax}
    norm = math.hypot(slope, 1.0)
    tx, ty = rho * slope / norm - a, rho / norm - b
    if tx >= 0.0 and ty >= 0.0:
        xs.add(-tx)
        ys.add(-ty)
    for k, (lo, hi, y, pb, _) in enumerate(frontier[0]):
        if k % 128 == 0 and time.perf_counter() >= stop:
            return []
        for x in (lo - a - gap, hi + a + gap):
            if -xmax <= x <= xmax:
                xs.add(x)
        for yy in (y - pb - b - gap, y + pb + b + gap):
            if -ymax <= yy <= ymax:
                ys.add(yy)
    events = []
    for axis, levels in ((0, xs), (1, ys)):
        for k, fixed in enumerate(levels):
            if k % 128 == 0 and time.perf_counter() >= stop:
                return []
            far = abs(fixed) + (a if axis == 0 else b)
            if far > rho:
                continue
            extent = math.sqrt(max(0.0, (rho - far) * (rho + far))) - (b if axis == 0 else a)
            if extent >= 0.0:
                bound = slope * fixed - extent if axis == 0 else fixed - slope * extent
                events.append((bound, axis, fixed, extent))
    events.sort()
    return events


def _locate(item, frontier, direction, rotate, rho, gap, bias, stop,
            weights=None):
    """Without weights, find a feasible gravity position; otherwise find the position with minimum blocking area.

    After fixing one contact coordinate, direct insertion skips forbidden intervals; weighted insertion scans interval endpoints,
    accumulates blocking area, and keeps candidate positions where several intervals touch simultaneously.
    """
    if frontier is None:
        return None
    idx, a0, b0, _ = item
    sx, sy, slope = direction
    options = [(a0, b0, 0)]
    if rotate and a0 != b0:
        options.append((b0, a0, 1))
    result, best_score, best_loss = None, float('inf'), float('inf')
    for a, b, rot in options:
        if time.perf_counter() >= stop:
            break
        offset = bias * (b + slope * a)
        events = _events(a, b, frontier, slope, rho, gap, stop)
        for bound, axis, fixed, extent in events:
            if time.perf_counter() >= stop:
                return result
            if (weights is None or best_loss == 0.0) and bound + offset >= best_score:
                break
            blockers = frontier[1 - axis]
            along, across = (b, a) if axis == 0 else (a, b)
            if weights is None:
                pos = -extent
                for j, (lo, hi, center, half, _) in enumerate(blockers):
                    if j % 128 == 127 and time.perf_counter() >= stop:
                        return result
                    if lo - along - gap >= pos:
                        break
                    if hi + along + gap > pos and abs(fixed - center) < across + half + gap:
                        pos = hi + along + gap
                        if pos > extent:
                            break
                if pos <= extent:
                    x, y = (fixed, pos) if axis == 0 else (pos, fixed)
                    score = y + slope * x + offset
                    if score < best_score:
                        best_score = score
                        result = (idx, sx * x, sy * y, a, b, rot)
                continue

            changes, initial = [(-extent, 0, 0.0), (extent, 0, 0.0)], []
            for j, (lo, hi, center, half, other_id) in enumerate(blockers):
                if j % 128 == 127 and time.perf_counter() >= stop:
                    return result
                if abs(fixed - center) >= across + half + gap:
                    continue
                low, high = lo - along - gap, hi + along + gap
                if high <= -extent or low >= extent:
                    continue
                w = weights[other_id]
                if low < -extent:
                    initial.append(w)
                else:
                    changes.append((low, 1, w))
                if high <= extent:
                    changes.append((high, -1, w))
            changes.sort()
            loss, j = math.fsum(initial), 0
            while j < len(changes):
                if j % 64 == 0 and time.perf_counter() >= stop:
                    return result
                pos, entered, exited = changes[j][0], 0.0, 0.0
                while j < len(changes) and changes[j][0] == pos:
                    _, typ, w = changes[j]
                    if typ < 0:
                        exited += w
                    elif typ > 0:
                        entered += w
                    j += 1
                loss -= exited
                cost = max(0.0, loss)
                x, y = (fixed, pos) if axis == 0 else (pos, fixed)
                score = y + slope * x + offset
                if cost < best_loss or (cost == best_loss and score < best_score):
                    best_loss, best_score = cost, score
                    result = (idx, sx * x, sy * y, a, b, rot)
                    if cost == 0.0:
                        return result
                loss += entered
    return result


def _shelves(sequence, transpose, rotate, rho, gap, stop):
    rows, packed = [], []
    low, high = 0.0, 0.0
    for idx, a0, b0, _ in sequence:
        if time.perf_counter() >= stop:
            break
        options = [(a0, b0, 0)]
        if rotate and a0 != b0:
            options.append((b0, a0, 1))
        if transpose:
            options = [(b, a, r) for a, b, r in options]
        choice, best = None, float('inf')
        for a, b, r in options:
            for j, (bottom, top, cursor, right) in enumerate(rows):
                if j % 128 == 127 and time.perf_counter() >= stop:
                    return packed
                if 2.0 * b <= top - bottom and cursor + 2.0 * a <= right:
                    score = top - bottom - 2.0 * b + 0.03 * (right - cursor - 2.0 * a)
                    if score < best:
                        choice, best = (j, a, b, r), score
        if choice is None:
            band, best = None, float('inf')
            for a, b, r in options:
                bands = [(-b, b)] if not rows else [(high + gap, high + gap + 2.0 * b),
                                                   (low - gap - 2.0 * b, low - gap)]
                for bottom, top in bands:
                    far = max(abs(bottom), abs(top))
                    if far >= rho:
                        continue
                    width = math.sqrt(max(0.0, (rho - far) * (rho + far))) - gap
                    if a > width:
                        continue
                    score = 2.0 * b / max(width, 1e-300) + 0.02 * far
                    if score < best:
                        band, best = (bottom, top, width, a, b, r), score
            if band is None:
                continue
            bottom, top, width, a, b, r = band
            rows.append([bottom, top, -width, width])
            low, high = min(low, bottom), max(high, top)
            choice = len(rows) - 1, a, b, r
        j, a, b, r = choice
        row = rows[j]
        x, y = row[2] + a, row[0] + b
        row[2] += 2.0 * a + gap
        packed.append((idx, y, x, b, a, r) if transpose else (idx, x, y, a, b, r))
    return packed


def _compact(packed, direction, rho, gap, stop):
    sx, sy, slope = direction
    work = [(i, sx * x, sy * y, a, b, r) for i, x, y, a, b, r in packed]
    # Iterate over the two physical coordinate axes; does not limit the number of rounds of the outer time-limited search.
    for axis in ((1, 0) if slope < 1.0 else (0, 1)):
        work.sort(key=lambda p: p[axis + 1])
        for j, (idx, x, y, a, b, rot) in enumerate(work):
            if time.perf_counter() >= stop:
                return [(i, sx * u, sy * v, aa, bb, r) for i, u, v, aa, bb, r in work]
            pos, cross, half, other = (x, y, a, b) if axis == 0 else (y, x, b, a)
            far = abs(cross) + other
            if far > rho:
                continue
            extent = math.sqrt(max(0.0, (rho - far) * (rho + far))) - half
            if extent < 0.0:
                continue
            target = -extent
            for k, (_, xx, yy, aa, bb, _) in enumerate(work):
                if k % 128 == 127 and time.perf_counter() >= stop:
                    break
                if j == k:
                    continue
                p, c, h, o = (xx, yy, aa, bb) if axis == 0 else (yy, xx, bb, aa)
                if p <= pos and abs(cross - c) < other + o + gap:
                    target = max(target, p + h + half + gap)
            else:
                if target < pos:
                    work[j] = (idx, target, y, a, b, rot) if axis == 0 else (idx, x, target, a, b, rot)
    return [(i, sx * x, sy * y, a, b, r) for i, x, y, a, b, r in work]


def _axis_context(packed, axis, rho, clearance, stop):
    """With the transverse coordinate fixed, compute earliest and latest feasible positions in the separation DAG."""
    work = sorted(packed, key=lambda p: p[axis + 1])
    size = len(work)
    centers, cross, half, thick, reach = [], [], [], [], []
    for p in work:
        if time.perf_counter() >= stop:
            return None
        q, h, t = p[2 - axis], p[3 + axis], p[4 - axis]
        far = abs(q) + t
        if far > rho:
            return None
        bound = math.sqrt(max(0.0, (rho - far) * (rho + far))) - h
        if bound < 0.0:
            return None
        centers.append(p[axis + 1])
        cross.append(q)
        half.append(h)
        thick.append(t)
        reach.append(bound)
    successors = [[] for _ in work]
    early = [-v for v in reach]
    late = list(reach)
    for i in range(size):
        if time.perf_counter() >= stop or early[i] > reach[i]:
            return None
        qi, ti, hi = cross[i], thick[i], half[i]
        for j in range(i + 1, size):
            if j % 128 == 0 and time.perf_counter() >= stop:
                return None
            if abs(qi - cross[j]) < ti + thick[j] + clearance:
                successors[i].append(j)
                early[j] = max(early[j], early[i] + hi + half[j] + clearance)
    for i in range(size - 1, -1, -1):
        if time.perf_counter() >= stop:
            return None
        for k, j in enumerate(successors[i]):
            if k % 128 == 0 and time.perf_counter() >= stop:
                return None
            late[i] = min(late[i], late[j] - half[i] - half[j] - clearance)
        if early[i] > late[i]:
            return None
    return work, centers, cross, half, thick, early, late, successors


def _axis_record(p, pos, axis):
    if axis == 0:
        return p[0], pos, p[2], p[3], p[4], p[5]
    return p[0], p[1], pos, p[3], p[4], p[5]


def _push_insert(item, context, axis, rotate, rho, clearance, stop):
    """Enumerate transverse contact positions and insertion orders; propagate the whole separation graph to shift old items jointly."""
    if context is None:
        return None
    work, centers, cross, half, thick, early, late, successors = context
    idx, a0, b0, _ = item
    options = [(a0, b0, 0)]
    if rotate and a0 != b0:
        options.append((b0, a0, 1))
    for a, b, rot in options:
        h, t = (a, b) if axis == 0 else (b, a)
        if math.hypot(h, t) > rho:
            continue
        transverse_bound = math.sqrt(max(0.0, (rho - h) * (rho + h))) - t
        levels = {0.0, -transverse_bound, transverse_bound}
        for j, q in enumerate(cross):
            if j % 128 == 0 and time.perf_counter() >= stop:
                return None
            delta = thick[j] + t + clearance
            # Take the outward adjacent float so that contact positions are not mistaken for interior intersections.
            left = math.nextafter(q - delta, -math.inf)
            right = math.nextafter(q + delta, math.inf)
            if abs(left) <= transverse_bound:
                levels.add(left)
            if abs(right) <= transverse_bound:
                levels.add(right)
        # The blocking set is constant between contact events; the position closest to zero has the largest axial range of motion.
        for fixed in sorted(levels, key=abs):
            if time.perf_counter() >= stop:
                return None
            far = abs(fixed) + t
            reach = math.sqrt(max(0.0, (rho - far) * (rho + far))) - h
            if reach < 0.0:
                continue
            crossed = []
            for j, q in enumerate(cross):
                if j % 128 == 0 and time.perf_counter() >= stop:
                    return None
                if abs(fixed - q) < t + thick[j] + clearance:
                    crossed.append(j)
            count = len(crossed)
            suffix = [reach] * (count + 1)
            for k in range(count - 1, -1, -1):
                if k % 128 == 0 and time.perf_counter() >= stop:
                    return None
                j = crossed[k]
                suffix[k] = min(suffix[k + 1], late[j] - half[j] - h - clearance)
            lower, choice, quality = -reach, None, float('inf')
            for cut in range(count + 1):
                if cut % 128 == 0 and time.perf_counter() >= stop:
                    return None
                upper = suffix[cut]
                if lower <= upper:
                    left = (centers[crossed[cut - 1]] + half[crossed[cut - 1]]
                            + h + clearance) if cut else -reach
                    right = (centers[crossed[cut]] - half[crossed[cut]]
                             - h - clearance) if cut < count else reach
                    pos = min(upper, max(lower, (left + right) * 0.5))
                    motion = max(0.0, left - pos) + max(0.0, pos - right)
                    if motion < quality:
                        quality, choice = motion, (cut, pos)
                if cut < count:
                    j = crossed[cut]
                    lower = max(lower, early[j] + half[j] + h + clearance)
            if choice is None:
                continue
            cut, pos = choice
            lo, hi = list(early), list(late)
            for k, j in enumerate(crossed):
                if k < cut:
                    hi[j] = min(hi[j], pos - h - half[j] - clearance)
                else:
                    lo[j] = max(lo[j], pos + h + half[j] + clearance)
            # First back-propagate new upper bounds, then forward-propagate new lower bounds; never commit a half-finished state.
            for i in range(len(work) - 1, -1, -1):
                if time.perf_counter() >= stop:
                    return None
                for k, j in enumerate(successors[i]):
                    if k % 128 == 0 and time.perf_counter() >= stop:
                        return None
                    hi[i] = min(hi[i], hi[j] - half[i] - half[j] - clearance)
            result = []
            for i, p in enumerate(work):
                if time.perf_counter() >= stop:
                    return None
                if lo[i] > hi[i]:
                    break
                moved = min(hi[i], max(lo[i], centers[i]))
                result.append(_axis_record(p, moved, axis))
                for k, j in enumerate(successors[i]):
                    if k % 128 == 0 and time.perf_counter() >= stop:
                        return None
                    lo[j] = max(lo[j], moved + half[i] + half[j] + clearance)
            else:
                x, y = (pos, fixed) if axis == 0 else (fixed, pos)
                result.append((idx, x, y, a, b, rot))
                return result
    return None


def solve(instance: dict, time_limit_s: float) -> dict:
    """Return only alpha, x, y, r; always keep the verified complete feasible solution."""
    start = time.perf_counter()
    answer = {'alpha': [], 'x': [], 'y': [], 'r': []}
    try:
        n = max(0, int(instance['n']))
        answer = {'alpha': [0] * n, 'x': [-1.0] * n,
                  'y': [-1.0] * n, 'r': [0] * n}
        budget = max(0.0, float(time_limit_s))
        guard = min(0.04 * budget, 0.008 + 0.000003 * n)
        finish = start + budget - guard
        reserve = min(0.12 * budget, 0.022 + 0.000007 * n)
        deadline = finish - reserve
        if n == 0 or time.perf_counter() >= deadline:
            return answer
        cx, cy, radius = float(instance['cx']), float(instance['cy']), float(instance['R'])
        rotate = int(instance['rotation_enabled']) == 1
        strength = 1.0 + math.log1p(budget)
        rng = random.Random(717149)
        precision = max(math.ulp(cx) / radius, math.ulp(cy) / radius,
                        math.ulp(radius) / radius)
        gap = max(4e-12, 8.0 * precision)
        rho = 1.0 - max(8e-12, 8.0 * precision)
        original, candidates, scanned = {}, [], True
        largest_log = -float('inf')
        for idx, pair in enumerate(instance['items']):
            if time.perf_counter() >= deadline:
                scanned = False
                break
            ha, hb = float(pair[0]) * 0.5, float(pair[1]) * 0.5
            a, b = ha / radius, hb / radius
            if a > 0.0 and b > 0.0 and math.hypot(a, b) <= 1.0 + 2e-15:
                logv = math.log(a) + math.log(b)
                largest_log = max(largest_log, logv)
                candidates.append((idx, a, b, logv))
                original[idx] = (ha, hb)
        if not candidates:
            return answer
        # Uniformly scale areas by a positive factor to reduce overflow and underflow in physical area computations.
        items = [(i, a, b, math.exp(v - largest_log)) for i, a, b, v in candidates]
        values = {p[0]: p[3] for p in items}
        by_id = {p[0]: p for p in items}
        metrics = {i: (v - largest_log, math.log(max(a, b)), math.log(min(a, b)),
                       abs(math.log(a) - math.log(b))) for i, a, b, v in candidates}
        best_layout, best_area = [], 0.0

        def area(packed):
            return math.fsum(values[p[0]] for p in packed)

        def offer(packed):
            """Independent check and atomic commit; timed-out or invalid candidates cannot corrupt the current answer."""
            nonlocal answer, best_layout, best_area
            if time.perf_counter() >= finish or area(packed) <= best_area:
                return
            accepted, physical, grid, seen = [], [], {}, set()
            cell = 2.0 / math.sqrt(max(1, len(packed)))
            pad = min(2.0, max(2e-11, 8.0 * precision))
            radius2 = radius * radius
            for rec in sorted(packed, key=lambda p: values[p[0]], reverse=True):
                if time.perf_counter() >= finish:
                    break
                idx, u, v, a, b, rot = rec
                if idx in seen or (rot and not rotate):
                    continue
                x, y = cx + radius * u, cy + radius * v
                if not (math.isfinite(x) and math.isfinite(y)):
                    continue
                ha, hb = original[idx]
                if rot:
                    ha, hb = hb, ha
                a, b = ha / radius, hb / radius
                u, v = (x - cx) / radius, (y - cy) / radius
                farx, fary = abs(u) + a, abs(v) + b
                if math.hypot(farx, fary) > 1.0 or farx * farx + fary * fary > 1.0:
                    continue
                valid = True
                for sx in (-1, 1):
                    for sy in (-1, 1):
                        dx, dy = x + sx * ha - cx, y + sy * hb - cy
                        if (math.hypot(dx, dy) > radius or
                                dx * dx + dy * dy > radius2):
                            valid = False
                if not valid:
                    continue
                # Enlarge spatial buckets so normalisation rounding does not miss rectangle pairs that need checking.
                lx = math.floor((u - a - pad + 1.0) / cell)
                ux = math.floor((u + a + pad + 1.0) / cell)
                ly = math.floor((v - b - pad + 1.0) / cell)
                uy = math.floor((v + b + pad + 1.0) / cell)
                neighbors = set()
                for gx in range(lx, ux + 1):
                    if time.perf_counter() >= finish:
                        return
                    for gy in range(ly, uy + 1):
                        neighbors.update(grid.get((gx, gy), ()))
                for k in neighbors:
                    qx, qy, qa, qb = physical[k]
                    q = accepted[k]
                    if ((abs(x - qx) < ha + qa and abs(y - qy) < hb + qb) or
                            (abs(u - q[1]) < a + q[3] and abs(v - q[2]) < b + q[4])):
                        valid = False
                        break
                if not valid:
                    continue
                k = len(accepted)
                seen.add(idx)
                accepted.append((idx, u, v, a, b, rot))
                physical.append((x, y, ha, hb))
                for gx in range(lx, ux + 1):
                    for gy in range(ly, uy + 1):
                        grid.setdefault((gx, gy), []).append(k)
            value = area(accepted)
            if value > best_area:
                out = {'alpha': [0] * n, 'x': [-1.0] * n,
                       'y': [-1.0] * n, 'r': [0] * n}
                for p, q in zip(accepted, physical):
                    idx = p[0]
                    out['alpha'][idx], out['r'][idx] = 1, int(p[5])
                    out['x'][idx], out['y'][idx] = q[0], q[1]
                answer, best_layout, best_area = out, accepted, value

        def fill(sequence, retained, direction, bias, stop):
            packed = list(retained)
            present = {p[0] for p in packed}
            failures, frontier = [], None
            for item in sequence:
                if time.perf_counter() >= stop:
                    break
                idx, a, b, _ = item
                if idx in present:
                    continue
                da, db = (min(a, b), max(a, b)) if rotate else (a, b)
                if any(fa <= da and fb <= db for fa, fb in failures):
                    continue
                if frontier is None:
                    frontier = _frontier(packed, direction, stop)
                rec = _locate(item, frontier, direction, rotate, rho, gap, bias, stop)
                if rec is not None:
                    packed.append(rec)
                    present.add(idx)
                    frontier = None
                elif time.perf_counter() < stop:
                    failures = [(fa, fb) for fa, fb in failures if not (da <= fa and db <= fb)]
                    failures.append((da, db))
            return packed

        def joint_fill(packed, sequence, stop, balance=False):
            """First try insertion without removal; may first redistribute gaps along one axis, then insert along the other."""
            work = list(packed)
            if not work or time.perf_counter() >= stop:
                return work
            clearance = gap * 0.375
            axes = (0, 1) if rng.random() < 0.5 else (1, 0)
            contexts = {}
            if balance:
                axis = axes[0]
                context = _axis_context(work, axis, rho, clearance, stop)
                if context is not None:
                    records, _, _, _, _, early, late, _ = context
                    fraction = rng.uniform(0.15, 0.85)
                    work = [_axis_record(p, (1.0 - fraction) * early[j]
                                         + fraction * late[j], axis)
                            for j, p in enumerate(records)]
                    axes = axes[::-1]
            present = {p[0] for p in work}
            for item in sequence:
                if time.perf_counter() >= stop:
                    break
                if item[0] in present:
                    continue
                for axis in axes:
                    if time.perf_counter() >= stop:
                        break
                    if axis not in contexts:
                        contexts[axis] = _axis_context(work, axis, rho, clearance, stop)
                    proposal = _push_insert(item, contexts[axis], axis, rotate,
                                            rho, clearance, stop)
                    if proposal is not None:
                        work = proposal
                        present.add(item[0])
                        contexts.clear()
                        offer(work)
                        break
            return work

        # Keep layouts with different item subsets as recombination parents, without relying on external files or state.
        pool = []
        pool_size = max(2, math.ceil(strength * math.sqrt(math.log1p(len(items)))))

        def remember(packed):
            if not packed or time.perf_counter() >= deadline:
                return
            chosen = frozenset(p[0] for p in packed)
            score = area(packed)
            entry = (score, chosen, list(packed))
            for k, old in enumerate(pool):
                if old[1] == chosen:
                    # Same item set: only update the geometric layout, without changing the area ordering.
                    pool[k] = entry
                    return
            pool.append(entry)
            if len(pool) > pool_size:
                loser = min(range(len(pool)), key=lambda k: pool[k][0])
                pool.pop(loser)

        def combine(packed, parent, stop):
            """Min-cut multi-item exchange between complementary layouts within the time slice."""
            if time.perf_counter() >= stop:
                return packed
            chosen = {p[0] for p in packed}
            score = area(packed)
            donors = []
            if parent:
                donors.append(parent)
            ranked = []
            for val, ids, old in pool:
                if time.perf_counter() >= stop:
                    return packed
                novel = math.fsum(values[i] for i in ids if i not in chosen)
                if score + novel > best_area:
                    ranked.append((novel * rng.uniform(0.8, 1.2), old))
            ranked.sort(key=lambda p: p[0], reverse=True)
            donors.extend(p[1] for p in ranked)
            for donor in donors:
                now = time.perf_counter()
                if now >= stop:
                    break
                # An area upper bound skips flow computations that cannot improve the current answer.
                chosen = {p[0] for p in packed}
                upper = score + math.fsum(values[p[0]] for p in donor if p[0] not in chosen)
                if upper <= best_area:
                    continue
                attempt = (stop - now) / (1.0 + math.log1p(len(donor)) / strength)
                mixed = _recombine(packed, donor, values, gap, min(stop, now + attempt))
                mixed_area = area(mixed)
                if mixed_area > score:
                    packed, score = mixed, mixed_area
                    offer(packed)
            return packed

        order = sorted(items, key=lambda p: p[3], reverse=True)
        # The centred single-item fallback also covers items that exactly touch the circle with no interior slack.
        for idx, a, b, val in order:
            if val <= best_area or time.perf_counter() >= deadline:
                break
            offer([(idx, 0.0, 0.0, a, b, 0)])
            if rotate and val > best_area:
                offer([(idx, 0.0, 0.0, b, a, 1)])
        if (scanned and len(best_layout) == len(items)) or rho <= 0.0:
            return answer
        seed_stop = min(deadline, time.perf_counter() + budget / (9.0 + strength))
        seed = _shelves(order, False, rotate, rho, gap, seed_stop)
        offer(seed)
        remember(seed)

        # Two items must be separated along some axis; the axial range is largest when the transverse centre is zero.
        # Also enumerate exact-contact cases that the construction gap excludes.
        pair_stop = min(deadline, time.perf_counter() + budget / (12.0 + strength))
        for j, first_item in enumerate(order):
            if time.perf_counter() >= pair_stop or j + 1 == len(order):
                break
            if first_item[3] + order[j + 1][3] <= best_area:
                break
            opts1 = [(first_item[1], first_item[2], 0)]
            if rotate and first_item[1] != first_item[2]:
                opts1.append((first_item[2], first_item[1], 1))
            for second in order[j + 1:]:
                if time.perf_counter() >= pair_stop or first_item[3] + second[3] <= best_area:
                    break
                opts2 = [(second[1], second[2], 0)]
                if rotate and second[1] != second[2]:
                    opts2.append((second[2], second[1], 1))
                for a, b, r in opts1:
                    for c, d, rr in opts2:
                        for axis in (0, 1):
                            h1, v1 = (a, b) if axis == 0 else (b, a)
                            h2, v2 = (c, d) if axis == 0 else (d, c)
                            reach1 = math.sqrt(max(0.0, (1.0 - v1) * (1.0 + v1))) - h1
                            reach2 = math.sqrt(max(0.0, (1.0 - v2) * (1.0 + v2))) - h2
                            reach = reach1 + reach2
                            if reach <= 0.0 or reach < h1 + h2:
                                continue
                            distance = min(reach, h1 + h2 + gap)
                            p1 = -distance * reach1 / reach
                            p2 = distance * reach2 / reach
                            x1, y1 = (p1, 0.0) if axis == 0 else (0.0, p1)
                            x2, y2 = (p2, 0.0) if axis == 0 else (0.0, p2)
                            offer([(first_item[0], x1, y1, a, b, r),
                                   (second[0], x2, y2, c, d, rr)])
        seed_stop = min(deadline, time.perf_counter() + budget / (7.0 + strength))
        seed = fill(order, best_layout, (1, 1, 0.08), 1.0, seed_stop)
        offer(seed)
        remember(seed)
        remember(best_layout)
        push_stop = min(deadline, time.perf_counter()
                        + budget / (16.0 + math.sqrt(len(items)) / strength))
        pushed = joint_fill(best_layout, order, push_stop)
        offer(pushed)
        remember(pushed)
        current, current_area = list(best_layout), best_area
        last_gain, first = time.perf_counter(), True
        mean_area = math.fsum(values.values()) / len(items)

        while time.perf_counter() < deadline:
            if scanned and len(best_layout) == len(items):
                break  # All individually fitting items are packed; the area upper bound is reached.
            now = time.perf_counter()
            stale = min(1.0, (now - last_gain) / max(budget * 0.3, 1e-12))
            progress = min(1.0, (now - start) / max(budget, 1e-12))
            duration = budget / (4.0 + math.sqrt(len(items)) / strength)
            cycle_end = min(deadline, now + duration)
            # Scale time slices with the total limit, reserving time for coupled pushing and min-cut recombination.
            fusion_share = strength / (strength + 12.0)
            use_push = rng.random() < 0.3 + 0.35 * stale
            joint_share = (1.0 - fusion_share) * strength / (strength + 18.0) if use_push else 0.0
            stop = cycle_end - (cycle_end - now) * (fusion_share + joint_share)
            direction = (rng.choice((-1, 1)), rng.choice((-1, 1)),
                         math.exp(rng.uniform(-3.0, 1.3) * (0.65 + 0.1 * strength)))
            bias = rng.uniform(0.0, 1.4)
            noise = 0.3 + strength * (0.12 + 0.4 * stale)
            mode = rng.random()
            if first:
                sequence, direction, bias = order, (1, 1, 0.08), 1.0
            else:
                exponent, aspect = rng.uniform(0.4, 1.3), rng.uniform(-0.35, 0.35)
                def priority(p):
                    la, lmax, lmin, shape = metrics[p[0]]
                    score = lmax if mode < 0.17 else lmin if mode < 0.32 else exponent * la + aspect * shape
                    return score + noise * rng.random()
                sequence = sorted(items, key=priority, reverse=True)
            retained, parent = [], []
            restart = first or rng.random() < (1.0 + stale) / (4.0 + strength)
            if restart:
                if not first and rng.random() < 1.0 / (2.0 + strength):
                    retained = _shelves(sequence, rng.random() < 0.5, rotate, rho, gap,
                                        min(stop, now + duration * 0.25))
            else:
                base = best_layout if rng.random() < 0.62 else current
                sx, sy = rng.choice((-1, 1)), rng.choice((-1, 1))
                swap = rotate and rng.random() < 0.5
                transformed = []
                for idx, x, y, a, b, r in base:
                    if swap:
                        x, y, a, b, r = y, x, b, a, 1 - r
                    transformed.append((idx, sx * x, sy * y, a, b, r))
                parent = list(transformed)
                present = {p[0] for p in transformed}
                absent = [p for p in sequence if p[0] not in present]
                cavity = None
                if absent and rng.random() < 0.48 + 0.12 * stale:
                    width = min(len(absent), max(1, math.ceil(strength * math.sqrt(math.log1p(len(items))))))
                    target = rng.choice(absent[:width]) if rng.random() < 0.75 else rng.choice(absent)
                    repair_stop = min(stop, now + duration * (0.18 + 0.08 * stale))
                    front = _frontier(transformed, direction, repair_stop)
                    rec = _locate(target, front, direction, rotate, rho, gap, bias,
                                  repair_stop, values)
                    if rec is not None:
                        _, x, y, a, b, _ = rec
                        # Recompute all blocking items; do not substitute a partial weighted scan for the separation check.
                        cavity = [p for p in transformed if
                                  abs(x - p[1]) >= a + p[3] + gap or
                                  abs(y - p[2]) >= b + p[4] + gap]
                        cavity.append(rec)
                if cavity is not None:
                    retained = cavity
                elif transformed:
                    k = len(transformed)
                    count = min(k, max(1, math.ceil(math.sqrt(k) * strength * rng.uniform(0.12, 0.5))))
                    if rng.random() < (strength + 3.0 * stale) / (strength + 12.0):
                        count = min(k, max(count, math.ceil(k * rng.uniform(0.2, 0.55 + 0.3 * stale))))
                    anchor = rng.choice(transformed)
                    kind = rng.random()
                    if kind < 0.68:
                        ax, ay, aa, ab = anchor[1:5]
                        stretch = math.exp(rng.uniform(-1.0, 1.0))
                        transformed.sort(key=lambda p: stretch * max(0.0, abs(p[1] - ax) - p[3] - aa) ** 2
                                         + max(0.0, abs(p[2] - ay) - p[4] - ab) ** 2 / stretch
                                         + rng.random() * 0.012)
                    elif kind < 0.87:
                        dx, dy, slope = direction
                        transformed.sort(key=lambda p: dy * p[2] + slope * dx * p[1], reverse=True)
                    else:
                        rng.shuffle(transformed)
                    retained = transformed[count:]
                    if absent and rng.random() < 0.45:
                        target = rng.choice(absent[:max(1, math.ceil(strength))])
                        sequence = [target] + [p for p in sequence if p[0] != target[0]]
                if retained and rng.random() < strength / (strength + 2.0):
                    move_stop = min(stop, now + duration * 0.38)
                    retained = _compact(retained, direction, rho, gap, move_stop)
                    if rng.random() < 0.45:
                        movers = list(retained)
                        rng.shuffle(movers)
                        depth = min(len(movers), max(1, math.ceil(math.sqrt(len(movers)) * strength)))
                        for old in movers[:depth]:
                            if time.perf_counter() >= move_stop:
                                break
                            rest = [p for p in retained if p[0] != old[0]]
                            front = _frontier(rest, direction, move_stop)
                            rec = _locate(by_id[old[0]], front, direction, rotate, rho, gap, bias, move_stop)
                            if rec is not None:
                                retained = rest + [rec]
            trial = fill(sequence, retained, direction, bias, stop)
            before = best_area
            offer(trial)
            remember(trial)
            if use_push and time.perf_counter() < cycle_end:
                push_stop = min(cycle_end - (cycle_end - now) * fusion_share,
                                time.perf_counter() + duration * joint_share)
                base = trial if area(trial) >= best_area or rng.random() < 0.5 else best_layout
                pushed = joint_fill(base, sequence, push_stop, rng.random() < 0.5)
                if area(pushed) >= area(trial):
                    trial = pushed
                    offer(trial)
                    remember(trial)
            if time.perf_counter() < cycle_end:
                cut_stop = min(cycle_end, time.perf_counter() + duration * fusion_share)
                hybrid = combine(trial, parent, cut_stop)
                if area(hybrid) > area(trial):
                    trial = fill(sequence, hybrid, direction, bias, cycle_end)
                    offer(trial)
                    remember(trial)
                elif time.perf_counter() >= stop and time.perf_counter() < cycle_end:
                    # When recombination yields no gain, return the remaining time to direct insertion.
                    trial = fill(sequence, trial, direction, bias, cycle_end)
                    offer(trial)
                    remember(trial)
            value = area(trial)
            if best_area > before:
                last_gain = time.perf_counter()
            temperature = max(1e-300, mean_area * (0.08 + 0.06 * strength)
                              * (1.0 - 0.8 * progress) * (1.0 + stale))
            if value >= current_area or rng.random() < math.exp(max(-700.0, (value - current_area) / temperature)):
                current, current_area = trial, value
            first = False
    except Exception:
        # On exceptions, return only the fallback answer that passed full checking and was committed.
        pass
    return answer
