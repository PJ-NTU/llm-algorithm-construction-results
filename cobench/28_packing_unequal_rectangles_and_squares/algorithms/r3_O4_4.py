"""Two-axis contact search with blocker-eviction packing
Uses two-axis gap scanning with lower-bound pruning, rotation-relocation compaction and blocker-eviction rebuilding, keeping the geometrically certified layout with the most items within the wall-clock limit.
"""
import math
import random
import time


def solve(instance: dict, time_limit_s: float) -> dict:
    started = time.perf_counter()
    answer = {"alpha": [], "x": [], "y": [], "r": []}
    best = []
    try:
        n = instance["num_rectangles"]
        answer = {"alpha": [0] * n, "x": [0.0] * n,
                  "y": [0.0] * n, "r": [0] * n}
        budget = float(time_limit_s)
        if not math.isfinite(budget) or budget <= 0.0 or n == 0:
            return answer
        widths, heights = instance["widths"], instance["heights"]
        radius = float(instance["container_radius"])
        now = time.perf_counter
        deadline = started + budget - min(budget * 0.08, 0.005 + n * 0.000004)
        effort = math.log1p(budget)
        rng = random.Random(173921)
        gap = 4.0e-13
        tol = gap * 0.125
        inner = 1.0 - gap
        rr = inner * inner
        dims = {}
        items = []
        scanned = True
        for i in range(n):
            if now() >= deadline:
                scanned = False
                break
            w, h = float(widths[i]), float(heights[i])
            a, b = (w / radius) * 0.5, (h / radius) * 0.5
            if not math.isfinite(a + b) or math.hypot(a, b) > 1.0:
                continue
            if math.hypot(w * 0.5, h * 0.5) > radius:
                continue
            dims[i] = (a, b, w * 0.5, h * 0.5)
            la, lb = math.log(max(a, 1e-300)), math.log(max(b, 1e-300))
            items.append((i, la + lb, math.log(max(a + b, 1e-300)), max(la, lb)))
            if not best:
                best = [(i, 0.0, 0.0, a + gap, b + gap, 0,
                         0.0, 0.0, a, b, w * 0.5, h * 0.5)]
        if not items:
            return answer

        def realize(i, x, y, rot, placed, end):
            """Certifies using both the normalised dimensions and the final output values in original units."""
            a, b, w, h = dims[i]
            if rot:
                a, b, w, h = b, a, h, w
            ox, oy = x * radius, y * radius
            if not (math.isfinite(ox) and math.isfinite(oy)):
                return None
            x, y = ox / radius, oy / radius
            if (math.hypot(abs(x) + a, abs(y) + b) > 1.0 or
                    math.hypot(abs(ox) + w, abs(oy) + h) > radius):
                return None
            for j, q in enumerate(placed):
                if j % 64 == 0 and now() >= end:
                    return None
                if abs(x - q[1]) < a + q[8] and abs(y - q[2]) < b + q[9]:
                    return None
                if abs(ox - q[6]) < w + q[10] and abs(oy - q[7]) < h + q[11]:
                    return None
            return (i, x, y, a + gap, b + gap, rot, ox, oy, a, b, w, h)

        def lines(a, b, placed, beta, end):
            """Orders contact lines by a lower bound on the position score in the obstacle-free case."""
            if math.hypot(a, b) > inner:
                return []
            xmax = math.sqrt(max(0.0, rr - b * b)) - a
            ymax = math.sqrt(max(0.0, rr - a * a)) - b
            sign = 1.0 if beta >= 0.0 else -1.0
            norm = math.hypot(1.0, beta)
            xs = [0.0, -xmax, xmax,
                  -sign * max(0.0, inner * abs(beta) / norm - a)]
            ys = [0.0, -ymax, ymax, -max(0.0, inner / norm - b)]
            cap = max(8, int((24.0 + 64.0 * effort) /
                            (1.0 + len(placed) / (32.0 + 32.0 * effort)) ** 0.2))
            # Small layouts cover all contact lines; for large layouts the coverage width grows with the budget.
            contacts = placed if 2 * len(placed) <= cap else rng.sample(placed, cap // 2)
            for j, q in enumerate(contacts):
                if j % 64 == 0 and now() >= end:
                    return []
                xs.extend((q[1] - q[3] - a, q[1] + q[3] + a))
                ys.extend((q[2] - q[4] - b, q[2] + q[4] + b))
            candidates = []
            for x in set(xs):
                if abs(x) <= xmax + tol:
                    x = max(-xmax, min(xmax, x))
                    extent = math.sqrt(max(0.0, rr - (abs(x) + a) ** 2)) - b
                    if extent >= -tol:
                        extent = max(0.0, extent)
                        candidates.append((-extent + beta * x, 1, x, extent))
            for y in set(ys):
                if abs(y) <= ymax + tol:
                    y = max(-ymax, min(ymax, y))
                    extent = math.sqrt(max(0.0, rr - (abs(y) + b) ** 2)) - a
                    if extent >= -tol:
                        extent = max(0.0, extent)
                        candidates.append((y - abs(beta) * extent, 2, y, extent))
            candidates.sort()
            return candidates

        def locate(i, placed, by_y, by_x, beta, theta, end, ceiling=math.inf):
            """Fixes one axis and exactly scans forbidden intervals to find the earliest gap on the other axis."""
            winner, best_score = None, ceiling
            a0, b0 = dims[i][:2]
            sign = 1.0 if beta >= 0.0 else -1.0
            for rot in ((0,) if a0 == b0 else (0, 1)):
                if now() >= end:
                    break
                a, b = (b0 + gap, a0 + gap) if rot else (a0 + gap, b0 + gap)
                constant = theta * (b + abs(beta) * a)
                for lower, axis, fixed, extent in lines(a, b, placed, beta, end):
                    if now() >= end or lower + constant >= best_score:
                        break
                    value = -extent
                    if axis == 1:
                        ordered, half, cross, direction = by_y, b, a, 1.0
                    else:
                        ordered, half, cross, direction = by_x, a, b, sign
                    other = 3 - axis
                    valid = True
                    for j, q in enumerate(ordered):
                        if j % 64 == 0 and now() >= end:
                            valid = False
                            break
                        if abs(fixed - q[axis]) >= cross + q[axis + 2] - tol:
                            continue
                        low = direction * q[other] - q[other + 2] - half
                        if value <= low + tol:
                            break
                        value = max(value, direction * q[other] + q[other + 2] + half)
                        if value > extent + tol:
                            valid = False
                            break
                    if not valid or value > extent + tol:
                        continue
                    x, y = (fixed, value) if axis == 1 else (sign * value, fixed)
                    score = y + beta * x + constant
                    if score < best_score:
                        candidate = realize(i, x, y, rot, placed, end)
                        if candidate is not None:
                            winner, best_score = candidate, score
            return winner

        def ordered_layout(placed, beta):
            sign = 1.0 if beta >= 0.0 else -1.0
            return (sorted(placed, key=lambda p: p[2] - p[4]),
                    sorted(placed, key=lambda p: sign * p[1] - p[3]))

        def compact(placed, beta, theta, end):
            """Repeated compaction; after stagnation, allows item-by-item relocation across gaps and with orientation changes."""
            sign = 1.0 if beta >= 0.0 else -1.0
            changed = True
            while changed and now() < end:
                changed = False
                for axis in (2, 1):
                    direction = 1.0 if axis == 2 else sign
                    other = 3 - axis
                    order = sorted(range(len(placed)), key=lambda j: direction * placed[j][axis])
                    for j in order:
                        if now() >= end:
                            return placed
                        p = placed[j]
                        half, cross = p[axis + 2], p[other + 2]
                        corner = abs(p[other]) + cross
                        if math.hypot(corner, half) > inner:
                            continue
                        value = -math.sqrt(max(0.0, rr - corner * corner)) + half
                        old = direction * p[axis]
                        for k, q in enumerate(placed):
                            if k % 64 == 0 and now() >= end:
                                return placed
                            if k == j or abs(p[other] - q[other]) >= cross + q[other + 2] - tol:
                                continue
                            if direction * q[axis] < old:
                                value = max(value, direction * q[axis] + q[axis + 2] + half)
                        if value < old - 8.0 * gap:
                            x, y = (direction * value, p[2]) if axis == 1 else (p[1], value)
                            q = realize(p[0], x, y, p[5], placed[:j] + placed[j + 1:], end)
                            if q is not None:
                                placed[j], changed = q, True
                if changed:
                    continue
                # Relocations that rotate or jump across obstacles can change adjacency relations that sliding alone cannot.
                order = sorted(range(len(placed)), key=lambda j:
                               -(placed[j][2] + beta * placed[j][1]))
                for j in order:
                    if now() >= end:
                        return placed
                    p = placed[j]
                    others = placed[:j] + placed[j + 1:]
                    by_y, by_x = ordered_layout(others, beta)
                    old = p[2] + beta * p[1] + theta * (p[4] + abs(beta) * p[3])
                    q = locate(p[0], others, by_y, by_x, beta, theta, end, old - 8.0 * gap)
                    if q is not None:
                        placed[j], changed = q, True
            return placed

        def eject(i, placed, beta, theta, loss_limit, end):
            """Sweeps intervals along a contact line to find forced insertion positions with few blocking items."""
            winner = None
            best_key = (loss_limit + 1, math.inf, math.inf)
            a0, b0 = dims[i][:2]
            areas = [p[8] * p[9] for p in placed]
            for rot in ((0,) if a0 == b0 else (0, 1)):
                if now() >= end:
                    break
                a, b = (b0 + gap, a0 + gap) if rot else (a0 + gap, b0 + gap)
                for _, axis, fixed, extent in lines(a, b, placed, beta, end):
                    if now() >= end:
                        return winner
                    other = 3 - axis
                    cross, half = (a, b) if axis == 1 else (b, a)
                    events, active = [], set()
                    active_area = 0.0
                    for j, q in enumerate(placed):
                        if j % 64 == 0 and now() >= end:
                            return winner
                        if abs(fixed - q[axis]) >= cross + q[axis + 2] - tol:
                            continue
                        low, high = q[other] - q[other + 2] - half, q[other] + q[other + 2] + half
                        if low < -extent < high:
                            active.add(j)
                            active_area += areas[j]
                        if -extent <= low < extent:
                            events.append((low, 1, j))
                        if -extent < high <= extent:
                            events.append((high, 0, j))
                    events.sort()
                    pos, value = 0, -extent
                    while now() < end:
                        count = len(active)
                        x, y = (fixed, value) if axis == 1 else (value, fixed)
                        key = (count, -active_area, y + beta * x + theta * (b + abs(beta) * a))
                        if count <= loss_limit and key < best_key:
                            keep = [p for j, p in enumerate(placed) if j not in active]
                            candidate = realize(i, x, y, rot, keep, end)
                            if candidate is not None:
                                winner, best_key = keep + [candidate], key
                                if count == 0:
                                    return winner
                        # At equal coordinates, remove ending intervals first and evaluate the contact point before adding starting intervals.
                        while pos < len(events) and events[pos][0] == value:
                            _, kind, j = events[pos]
                            if kind == 1:
                                active.add(j)
                                active_area += areas[j]
                            pos += 1
                        if pos >= len(events):
                            break
                        value = events[pos][0]
                        while pos < len(events) and events[pos][0] == value and events[pos][1] == 0:
                            j = events[pos][2]
                            if j in active:
                                active.remove(j)
                                active_area -= areas[j]
                            pos += 1
            return winner

        def reflect(placed):
            swap = rng.randrange(2)
            sx, sy = rng.choice((-1.0, 1.0)), rng.choice((-1.0, 1.0))
            if swap:
                return [(p[0], sx * p[2], sy * p[1], p[4], p[3], p[5] ^ 1,
                         sx * p[7], sy * p[6], p[9], p[8], p[11], p[10]) for p in placed]
            return [(p[0], sx * p[1], sy * p[2], p[3], p[4], p[5],
                     sx * p[6], sy * p[7], p[8], p[9], p[10], p[11]) for p in placed]

        population = [best[:]]
        population_limit = max(1, min(len(items), int(1.0 + 2.0 * effort)))
        stagnant, turn = 0, 0
        while now() < deadline:
            if scanned and len(best) == len(items):
                break  # Every rectangle that fits on its own is already packed; the count has reached its upper bound.
            old_best = len(best)
            job_span = budget / (2.0 + effort)
            job_end = min(deadline, now() + job_span)
            beta = 0.12 if turn == 0 else rng.uniform(-1.0, 1.0)
            theta = 1.0 if turn == 0 else rng.random()
            fresh = turn < math.ceil(math.sqrt(1.0 + effort)) or rng.random() < 1.0 / (3.0 + effort)
            noise = 0.0 if turn == 0 else rng.random() * (0.15 + 0.4 * effort / (1.0 + effort))
            mode = turn % 4
            mix = rng.random()
            if mode == 0:
                priority = lambda t: t[1] + noise * rng.gauss(0.0, 1.0)
            elif mode == 1:
                priority = lambda t: 2.0 * t[2] + noise * rng.gauss(0.0, 1.0)
            elif mode == 2:
                priority = lambda t: 0.5 * t[1] + t[3] + noise * rng.gauss(0.0, 1.0)
            else:
                priority = lambda t: (1.0 - mix) * t[1] + 2.0 * mix * t[3] + noise * rng.gauss(0.0, 1.0)
            order = sorted(items, key=priority)
            if fresh:
                placed = []
            else:
                parent = best if rng.random() < 0.5 else rng.choice(population)
                placed = reflect(parent)
                selected = {p[0] for p in placed}
                remaining = [t for t in order if t[0] not in selected]
                forced = None
                if remaining and rng.random() < effort / (1.5 + effort):
                    pool = min(len(remaining), max(1, int((1.0 + effort) * math.sqrt(len(remaining)))))
                    target = remaining[min(pool - 1, int(pool * rng.random() ** 2))][0]
                    loss_limit = min(len(placed), max(1, int(math.sqrt(effort) + math.log1p(stagnant))))
                    forced = eject(target, placed, beta, theta, loss_limit,
                                   min(job_end, now() + job_span / (2.0 + effort)))
                if forced is not None:
                    placed = forced
                    if len(placed) > len(best):
                        best = placed[:]
                else:
                    k = len(placed)
                    depth = min(k, max(1, int(math.sqrt(k) * (0.6 + 0.4 * effort) *
                                             (1.0 + math.log1p(stagnant) / (2.0 + effort)))))
                    count = rng.randint(1, depth) if depth else 0
                    method = rng.randrange(4)
                    if method == 0 and placed:
                        anchor = rng.choice(placed)
                        stretch = math.exp(rng.uniform(-1.0, 1.0))
                        ranking = sorted(placed, key=lambda p:
                                         stretch * (p[1] - anchor[1]) ** 2 +
                                         (p[2] - anchor[2]) ** 2 / stretch)
                    elif method == 1:
                        ranking = sorted(placed, key=lambda p: -(p[2] + beta * p[1]))
                    elif method == 2:
                        ranking = sorted(placed, key=lambda p: -p[3] * p[4] * (0.5 + rng.random()))
                    else:
                        ranking = rng.sample(placed, k)
                    removed = {p[0] for p in ranking[:count]}
                    placed = [p for p in placed if p[0] not in removed]
                if placed and rng.random() < effort / (1.0 + effort):
                    compact_end = min(job_end, now() + job_span / (3.0 + math.sqrt(len(placed))))
                    placed = compact(placed, beta, theta, compact_end)
            selected = {p[0] for p in placed}
            by_y, by_x = ordered_layout(placed, beta)
            sign = 1.0 if beta >= 0.0 else -1.0
            for t in order:
                if now() >= job_end:
                    break
                if t[0] in selected:
                    continue
                p = locate(t[0], placed, by_y, by_x, beta, theta, job_end)
                if p is None:
                    continue
                placed.append(p)
                if len(placed) > len(best):
                    best = placed[:]
                by_y.append(p)
                by_x.append(p)
                by_y.sort(key=lambda q: q[2] - q[4])
                by_x.sort(key=lambda q: sign * q[1] - q[3])
                if scanned and len(best) == len(items):
                    break
            if placed:
                if len(population) < population_limit:
                    population.append(placed[:])
                else:
                    worst = min(range(len(population)), key=lambda j: len(population[j]))
                    if len(placed) >= len(population[worst]):
                        population[worst] = placed[:]
            stagnant = 0 if len(best) > old_best else stagnant + 1
            turn += 1
    except Exception:
        # A failed optional search does not destroy the already certified feasible layout.
        pass
    try:
        for p in best:
            i = p[0]
            answer["x"][i], answer["y"][i] = p[6], p[7]
            answer["r"][i], answer["alpha"][i] = p[5], 1
    except Exception:
        pass
    return answer
