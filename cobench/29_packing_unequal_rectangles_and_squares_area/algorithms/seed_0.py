"""Anytime area maximization by shared contact and minimum-loss exchange sweeps.

Shelf seeds, directional compaction, ordinary insertion and cavity exchanges all
modify the same layouts.  In a cavity exchange the contact sweep minimizes the
area of its blockers, rather than treating every blocker as forbidden.  Every
published incumbent is checked again after conversion to the input coordinates.
"""
import math
import random
import time


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
    """The same circle/contact events serve both hard and priced insertion."""
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
    """weights=None: feasible gravity placement; otherwise minimum-area cavity.

    At a fixed contact coordinate, ordinary insertion skips forbidden intervals.
    Priced insertion instead sweeps their entry/exit events, finding the least
    total blocking area, including positions where several intervals touch.
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
    # These are the two physical axes, not a limit on the anytime search.
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


def solve(instance: dict, time_limit_s: float) -> dict:
    """Return exactly alpha, x, y, r; retain a feasible incumbent until timeout."""
    start = time.perf_counter()
    answer = {'alpha': [], 'x': [], 'y': [], 'r': []}
    try:
        n = max(0, int(instance['n']))
        answer = {'alpha': [0] * n, 'x': [-1.0] * n,
                  'y': [-1.0] * n, 'r': [0] * n}
        budget = max(0.0, float(time_limit_s))
        guard = min(0.02 * budget, 0.004 + 0.000001 * n)
        finish = start + budget - guard
        reserve = min(0.12 * budget, 0.018 + 0.000005 * n)
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
        # Common positive scaling avoids overflow/underflow of physical areas.
        items = [(i, a, b, math.exp(v - largest_log)) for i, a, b, v in candidates]
        values = {p[0]: p[3] for p in items}
        by_id = {p[0]: p for p in items}
        metrics = {i: (v - largest_log, math.log(max(a, b)), math.log(min(a, b)),
                       abs(math.log(a) - math.log(b))) for i, a, b, v in candidates}
        best_layout, best_area = [], 0.0

        def area(packed):
            return math.fsum(values[p[0]] for p in packed)

        def offer(packed):
            """Transactional validation; a timed-out candidate cannot hurt best."""
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
                # Expanded bins prevent rounded normalized edges hiding a pair.
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

        order = sorted(items, key=lambda p: p[3], reverse=True)
        # Exact centered fallback also handles items with no interior slack.
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
        seed_stop = min(deadline, time.perf_counter() + budget / (7.0 + strength))
        offer(fill(order, best_layout, (1, 1, 0.08), 1.0, seed_stop))
        current, current_area = list(best_layout), best_area
        last_gain, first = time.perf_counter(), True
        mean_area = math.fsum(values.values()) / len(items)

        while time.perf_counter() < deadline:
            if scanned and len(best_layout) == len(items):
                break  # Every individually admissible item is loaded: optimal.
            now = time.perf_counter()
            stale = min(1.0, (now - last_gain) / max(budget * 0.3, 1e-12))
            progress = min(1.0, (now - start) / max(budget, 1e-12))
            duration = budget / (4.0 + math.sqrt(len(items)) / strength)
            stop = min(deadline, now + duration)
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
            retained = []
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
                        # Recompute the entire blocker set; never trust a partial
                        # or rounded weighted sweep to certify separation.
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
            value, before = area(trial), best_area
            offer(trial)
            if best_area > before:
                last_gain = time.perf_counter()
            temperature = max(1e-300, mean_area * (0.08 + 0.06 * strength)
                              * (1.0 - 0.8 * progress) * (1.0 + stale))
            if value >= current_area or rng.random() < math.exp(max(-700.0, (value - current_area) / temperature)):
                current, current_area = trial, value
            first = False
    except Exception:
        # Only fully validated, atomically committed solutions reach this point.
        pass
    return answer
