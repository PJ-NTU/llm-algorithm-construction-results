"""Cooperative projection and min-cut recombination packing
Combines contact construction and eviction-repacking with joint multi-rectangle projection repair and min-cut fusion, checking all hard constraints in the original coordinates.
"""
import math
import random
import time


def _recombine(left, right, values, gap, stop):
    """Build a bipartite conflict graph for two internally feasible layouts and take the maximum-area compatible subset via min-cut; on interruption keep the parent layout."""
    lv = math.fsum(values[p[0]] for p in left)
    rv = math.fsum(values[p[0]] for p in right)
    fallback = left if lv >= rv else right
    if not left or not right or time.perf_counter() >= stop:
        return fallback
    left_ids = {p[0] for p in left}
    upper = lv + math.fsum(values[p[0]] for p in right if p[0] not in left_ids)
    if upper <= max(lv, rv):
        return fallback

    # Positions identical in both parent layouts can be fixed in advance to shrink the flow network.
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

    # Scan rectangle boundaries to build conflicts; the conservative inflation is smaller than the gap reserved during construction.
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

    # Non-recursive Dinic, avoiding the recursion stack depth limit.
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
            # Take the complement of the minimum vertex cover to obtain the maximum-weight independent set.
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
    """Plain insertion and priced insertion share circular-arc and rectangle contact events."""
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
    """Plain insertion skips forbidden zones; priced insertion scans interval endpoints for the position with minimum total blocking area."""
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
    # Iterate over the two physical coordinate axes; this does not limit the number of search iterations.
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


def _joint_project(packed, mobile_ids, rho, gap, rng, strength, stop):
    """Choose separating half-spaces and jointly adjust movable centres by corrected cyclic projection."""
    started = time.perf_counter()
    if started >= stop:
        return None
    count = len(packed)
    mobile = [p[0] in mobile_ids for p in packed]
    positions = [[p[1] for p in packed], [p[2] for p in packed]]
    halves = [[p[3] for p in packed], [p[4] for p in packed]]
    inner = rho - 16.0 * gap
    if inner <= 0.0:
        return None
    reaches = [[], []]
    for j, p in enumerate(packed):
        if time.perf_counter() >= stop:
            return None
        a, b = p[3:5]
        if math.hypot(a, b) > (inner if mobile[j] else rho):
            return None
        rad = inner if mobile[j] else rho
        reaches[0].append(math.sqrt(max(0.0, (rad - b) * (rad + b))) - a)
        reaches[1].append(math.sqrt(max(0.0, (rad - a) * (rad + a))) - b)
    constraints = []
    noise = rng.random() / (1.0 + math.sqrt(strength))
    for i in (k for k in range(count) if mobile[k]):
        for j in range(count):
            if i == j or (mobile[j] and j < i):
                continue
            if j % 32 == 0 and time.perf_counter() >= stop:
                return None
            free = int(mobile[i]) + int(mobile[j])
            if not free:
                continue
            choices = []
            for axis in (0, 1):
                sep = halves[axis][i] + halves[axis][j] + 16.0 * gap
                for sign in (-1, 1):
                    bound = (reaches[axis][i] if mobile[i] else -sign * positions[axis][i])
                    bound += reaches[axis][j] if mobile[j] else sign * positions[axis][j]
                    if bound < sep:
                        continue
                    slack = sign * (positions[axis][j] - positions[axis][i]) - sep
                    # Already separated old items keep one existing relation; overlapping or new items explore alternative relations.
                    score = -slack
                    if i == count - 1 or j == count - 1 or slack < 0.0:
                        score += noise * (halves[axis][i] + halves[axis][j]) * rng.random()
                    choices.append((score, axis, sign, sep))
            if not choices:
                return None
            _, axis, sign, sep = min(choices)
            constraints.append((i, j, axis, sign, sep, 1.0 / free))
    rng.shuffle(constraints)
    corrections = [0.0] * len(constraints)
    circle_x, circle_y = [0.0] * count, [0.0] * count
    tolerance = 2.0 * gap
    best_error, last_progress = float('inf'), time.perf_counter()
    stall_span = (stop - started) / (2.0 + math.sqrt(strength))
    xs, ys = positions
    while time.perf_counter() < stop:
        # Each item's centre domain is the intersection of four disks; its projection can be computed directly in the corresponding quadrant.
        for j, movable in enumerate(mobile):
            if not movable:
                continue
            if j % 32 == 0 and time.perf_counter() >= stop:
                return None
            x, y = xs[j] + circle_x[j], ys[j] + circle_y[j]
            a, b = halves[0][j], halves[1][j]
            ax, ay = abs(x), abs(y)
            length = math.hypot(ax + a, ay + b)
            if length > inner:
                scale = inner / length
                u, v = (ax + a) * scale - a, (ay + b) * scale - b
                if u < 0.0:
                    u, v = 0.0, math.sqrt(max(0.0, (inner - a) * (inner + a))) - b
                elif v < 0.0:
                    u, v = math.sqrt(max(0.0, (inner - b) * (inner + b))) - a, 0.0
                px, py = math.copysign(max(0.0, u), x), math.copysign(max(0.0, v), y)
            else:
                px, py = x, y
            circle_x[j], circle_y[j] = x - px, y - py
            xs[j], ys[j] = px, py
        for k, (i, j, axis, sign, sep, inverse) in enumerate(constraints):
            if k % 64 == 0 and time.perf_counter() >= stop:
                return None
            coords = positions[axis]
            old = corrections[k]
            if mobile[i]:
                coords[i] += sign * old
            if mobile[j]:
                coords[j] -= sign * old
            amount = max(0.0, (sep - sign * (coords[j] - coords[i])) * inverse)
            if mobile[i]:
                coords[i] -= sign * amount
            if mobile[j]:
                coords[j] += sign * amount
            corrections[k] = amount
        error = 0.0
        for j, movable in enumerate(mobile):
            if j % 64 == 0 and time.perf_counter() >= stop:
                return None
            if movable:
                error = max(error, math.hypot(abs(xs[j]) + halves[0][j],
                                              abs(ys[j]) + halves[1][j]) - inner)
        for k, (i, j, axis, sign, sep, _) in enumerate(constraints):
            if k % 64 == 0 and time.perf_counter() >= stop:
                return None
            error = max(error, sep - sign * (positions[axis][j] - positions[axis][i]))
        if not math.isfinite(error):
            return None
        if error <= tolerance:
            return [(p[0], xs[j], ys[j], p[3], p[4], p[5])
                    for j, p in enumerate(packed)]
        now = time.perf_counter()
        if error < best_error * 0.995:
            best_error, last_progress = error, now
        elif now - last_progress >= stall_span:
            return None
    return None


def solve(instance: dict, time_limit_s: float) -> dict:
    """Return only alpha, x, y, r, and always keep the complete feasible current best solution."""
    start = time.perf_counter()
    answer = {'alpha': [], 'x': [], 'y': [], 'r': []}
    try:
        n = max(0, int(instance['n']))
        answer = {'alpha': [0] * n, 'x': [-1.0] * n,
                  'y': [-1.0] * n, 'r': [0] * n}
        budget = max(0.0, float(time_limit_s))
        if not math.isfinite(budget) or budget <= 0.0:
            return answer
        guard = min(0.04 * budget, 0.008 + 0.000003 * n)
        finish = start + budget - guard
        reserve = min(0.12 * budget, 0.022 + 0.000007 * n)
        deadline = finish - reserve
        if n == 0 or time.perf_counter() >= deadline:
            return answer
        cx, cy, radius = float(instance['cx']), float(instance['cy']), float(instance['R'])
        if radius <= 0.0 or not all(map(math.isfinite, (cx, cy, radius))):
            return answer
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
        # Uniformly scale areas by a positive factor to reduce the risk of physical area overflow or underflow.
        items = [(i, a, b, math.exp(v - largest_log)) for i, a, b, v in candidates]
        values = {p[0]: p[3] for p in items}
        by_id = {p[0]: p for p in items}
        metrics = {i: (v - largest_log, math.log(max(a, b)), math.log(min(a, b)),
                       abs(math.log(a) - math.log(b))) for i, a, b, v in candidates}
        best_layout, best_area = [], 0.0

        def area(packed):
            return math.fsum(values[p[0]] for p in packed)

        def offer(packed):
            """Transactional check; candidate layouts interrupted by the time limit cannot overwrite the committed answer."""
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
                # Extend grid bounds so normalisation rounding does not miss adjacent rectangles.
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
                for number, k in enumerate(neighbors):
                    if number % 128 == 0 and time.perf_counter() >= finish:
                        return
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
                    if time.perf_counter() >= finish:
                        return
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

        # Keep different selected subsets as complementary recombination sources, without relying on external state or files.
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
                    # When the item set is the same, update geometric positions; the area ordering stays unchanged.
                    pool[k] = entry
                    return
            pool.append(entry)
            if len(pool) > pool_size:
                loser = min(range(len(pool)), key=lambda k: pool[k][0])
                pool.pop(loser)

        def combine(packed, parent, stop):
            """Group min-cut exchange using complementary parent layouts within the time limit."""
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
                # An area upper bound skips flow networks that cannot improve the current answer.
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

        joint_rng = random.Random(2718281)
        joint_spent = 0.0
        joint_quantum = budget / ((6.0 + math.sqrt(len(items)) / strength) *
                                  (1.0 + strength))

        def cooperative(stop):
            """For unpacked items, try cooperative shifts, rotations and small removals with positive area gain."""
            first_attempt = True
            while time.perf_counter() < stop:
                base = list(best_layout)
                present = {p[0] for p in base}
                missing = [p for p in order if p[0] not in present]
                if not missing:
                    return
                width = min(len(missing), max(1, math.ceil(
                    strength * math.sqrt(math.log1p(len(items))))))
                target = (missing[0] if first_attempt else
                          joint_rng.choice(missing[:width]) if joint_rng.random() < 0.8 else
                          joint_rng.choice(missing))
                first_attempt = False
                now = time.perf_counter()
                attempt_end = min(stop, now + joint_quantum)
                direction = (joint_rng.choice((-1, 1)), joint_rng.choice((-1, 1)),
                             math.exp(joint_rng.uniform(-2.5, 1.0)))
                locate_end = now + (attempt_end - now) / (4.0 + strength)
                front = _frontier(base, direction, locate_end)
                rec = _locate(target, front, direction, rotate, rho, gap,
                              joint_rng.random(), locate_end, values)
                if rec is None:
                    idx, a, b, _ = target
                    rot = int(rotate and joint_rng.random() < 0.5)
                    if rot:
                        a, b = b, a
                    rec = (idx, 0.0, 0.0, a, b, rot)
                elif rotate and joint_rng.random() < 1.0 / (1.0 + strength):
                    rec = (rec[0], rec[1], rec[2], rec[4], rec[3], 1 - rec[5])
                idx, x, y, a, b, _ = rec
                def distance(p):
                    return (max(0.0, abs(p[1] - x) - p[3] - a) ** 2 +
                            max(0.0, abs(p[2] - y) - p[4] - b) ** 2)
                nearby = sorted(base, key=distance)
                # When total area exceeds the circle area not everything can be kept; only explore removals that can still raise the objective.
                removal = []
                physical_area = math.fsum(4.0 * p[3] * p[4] for p in base) + 4.0 * a * b
                potential = area(base) + values[idx]
                local_size = min(len(base), max(1, math.ceil(math.sqrt(len(base) + 1) * strength)))
                cheap = sorted(nearby[:local_size],
                               key=lambda p: values[p[0]] * joint_rng.uniform(0.65, 1.35))
                extra = (joint_rng.random() < 1.0 / (1.0 + math.sqrt(strength)))
                for p in cheap:
                    if time.perf_counter() >= attempt_end:
                        break
                    if physical_area <= math.pi * rho * rho and not extra:
                        break
                    if potential - values[p[0]] <= best_area:
                        continue
                    removal.append(p[0])
                    physical_area -= 4.0 * p[3] * p[4]
                    potential -= values[p[0]]
                    extra = extra and joint_rng.random() < 1.0 / (1.0 + strength)
                if physical_area > math.pi * rho * rho or potential <= best_area:
                    continue
                removed = set(removal)
                work = [p for p in base if p[0] not in removed]
                movable_count = min(len(work), max(1, math.ceil(
                    math.sqrt(len(work) + 1) * strength * joint_rng.uniform(0.75, 1.5))))
                mobile_ids = {p[0] for p in nearby if p[0] not in removed}
                if movable_count < len(work):
                    mobile_ids = {p[0] for p in sorted(work, key=distance)[:movable_count]}
                mobile_ids.add(idx)
                if rotate and work and joint_rng.random() < 1.0 / (2.0 + strength):
                    candidates_to_turn = [j for j, p in enumerate(work) if p[0] in mobile_ids]
                    if candidates_to_turn:
                        j = joint_rng.choice(candidates_to_turn)
                        p = work[j]
                        work[j] = (p[0], p[1], p[2], p[4], p[3], 1 - p[5])
                work.append(rec)
                repaired = _joint_project(work, mobile_ids, rho, gap, joint_rng,
                                          strength, attempt_end)
                if repaired is not None:
                    offer(repaired)
                    remember(repaired)
                    if time.perf_counter() < stop:
                        completed = fill(order, repaired, direction, 0.5, stop)
                        offer(completed)
                        remember(completed)

        order = sorted(items, key=lambda p: p[3], reverse=True)
        # The centred single-item construction also covers boundary items with no interior slack.
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

        # Two items must be separated along some axis; setting the other axis coordinate to zero gives the maximum axial spacing.
        # Also try exact-contact cases excluded by the interior safety margin.
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
        current, current_area = list(best_layout), best_area
        last_gain, first = time.perf_counter(), True
        mean_area = math.fsum(values.values()) / len(items)

        while time.perf_counter() < deadline:
            if scanned and len(best_layout) == len(items):
                break  # All individually packable items are packed; the global area upper bound is reached.
            now = time.perf_counter()
            stale = min(1.0, (now - last_gain) / max(budget * 0.3, 1e-12))
            progress = min(1.0, (now - start) / max(budget, 1e-12))
            duration = budget / (4.0 + math.sqrt(len(items)) / strength)
            cycle_end = min(deadline, now + duration)
            # Reserve recombination time according to the budget, fusing complete item groups after plain insertion.
            fusion_share = strength / (strength + 12.0)
            stop = cycle_end - (cycle_end - now) * fusion_share
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
                        # Recompute the full blocking set; do not substitute an interrupted or rounded price scan for the separation check.
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
            if time.perf_counter() < cycle_end:
                cut_stop = min(cycle_end, time.perf_counter() + duration * fusion_share)
                hybrid = combine(trial, parent, cut_stop)
                if area(hybrid) > area(trial):
                    trial = fill(sequence, hybrid, direction, bias, cycle_end)
                    offer(trial)
                    remember(trial)
                elif time.perf_counter() >= stop and time.perf_counter() < cycle_end:
                    # When recombination does not improve, spend the remaining time on further filling.
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
            now = time.perf_counter()
            joint_share = strength / (strength + 8.0 + math.sqrt(len(best_layout)))
            allowance = joint_share * (now - start) - joint_spent
            if allowance >= joint_quantum / (4.0 + strength) and now < deadline:
                prior_area = best_area
                cooperative(min(deadline, now + min(allowance, duration)))
                joint_spent += time.perf_counter() - now
                if best_area > prior_area:
                    last_gain = time.perf_counter()
                    current, current_area = list(best_layout), best_area
    except Exception:
        # Keep only answers that have been fully checked and atomically committed.
        pass
    return answer
