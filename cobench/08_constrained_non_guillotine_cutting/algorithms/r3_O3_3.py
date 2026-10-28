"""Value-geometry cooperative repair with conflict-graph recombination
Generates candidates with quota backtracking, backfilling by combined value and fit, count re-sequencing and blocker removal, and improves layouts through conflict-graph branch and bound and adaptive spatial search.
"""

import math
import random
import time


def solve(instance: dict, time_limit_s: float) -> dict:
    clock = time.perf_counter
    start = clock()
    K = 0
    best = []
    scale = 1.0
    inactive = (0.0, 0.0)
    try:
        K = max(0, int(instance.get("max_slots", 0)))
        budget = max(0.0, float(time_limit_s))
        if not math.isfinite(budget):
            budget = 0.0
        reserve = min(budget * 0.12, 0.003 + K * 0.000004) + budget * 0.015
        end = start + max(0.0, budget - reserve)
        verification_end = start + max(0.0, budget - reserve * 0.65)
        N = int(instance["num_piece_types"])
        stock_l = float(instance["stock_length"])
        stock_w = float(instance["stock_width"])
        scale = max(stock_l, stock_w)
        if K <= 0 or N <= 0 or scale <= 0.0:
            return {"placements": [_unused() for _ in range(K)]}
        L, W = stock_l / scale, stock_w / scale
        M = float(instance["big_m"]) / scale
        sizes = [(float(instance["piece_lengths"][i]) / scale,
                  float(instance["piece_widths"][i]) / scale) for i in range(N)]
        value = [float(v) for v in instance["piece_values"]]
        lower = [max(0, int(v)) for v in instance["min_usage"]]
        upper = [min(K, max(0, int(v))) for v in instance["max_usage"]]
        area = [a * b for a, b in sizes]
        if (len(value) != N or len(lower) != N or len(upper) != N
                or not all(math.isfinite(z) for z in [L, W, M] + value
                           + [z for pair in sizes for z in pair])
                or M < 0 or L <= 0 or W <= 0
                or any(a <= 0 or b <= 0 for a, b in sizes)):
            return {"placements": [_unused() for _ in range(K)]}
        eps = 2e-12
        board = L * W
        orientations = []
        for i, (a, b) in enumerate(sizes):
            choices = []
            if a <= L + eps and b <= W + eps:
                choices.append((a, b, 0))
            if a != b and b <= L + eps and a <= W + eps:
                choices.append((b, a, 1))
            orientations.append(choices)
            upper[i] = min(upper[i], int((board + eps) / area[i])) if choices else 0
        possible = [i for i in range(N) if upper[i] > 0]
        minimum_area = min((area[i] for i in possible), default=board + 1.0)
        minimum_side = min((min(sizes[i]) for i in possible), default=2.0)
        required_count = sum(lower)
        tight_m = M < max(L, W) - eps
        effort = math.log1p(budget)
        pool_limit = max(2, 2 + int(2 * effort))
        rng = random.Random(190731 + 1009 * N + 65537 * K)
        best_value = 0.0 if required_count == 0 else -math.inf
        last_gain = start
        pool = []
        root_area = [math.sqrt(a) for a in area]
        log_area = [math.log(a) for a in area]
        log_value = [math.log(v) if v > 0 else -math.inf for v in value]

        def counts_of(placed):
            counts = [0] * N
            for p in placed:
                counts[p[0]] += 1
            return counts

        def compatible(p, placed):
            if not tight_m:
                return True
            _, x, y, w, h, _ = p
            for j, q in enumerate(placed):
                if j & 127 == 0 and clock() >= verification_end:
                    return False
                if (x + w - q[1] > M + eps or q[1] + q[3] - x > M + eps
                        or y + h - q[2] > M + eps or q[2] + q[4] - y > M + eps):
                    return False
            return True

        def unused_position(placed):
            if not tight_m or not placed or len(placed) == K:
                return (0.0, 0.0)
            lx = max(0.0, max(p[1] + p[3] - 2 * M for p in placed))
            hx = min(L, min(p[1] + 2 * M for p in placed))
            ly = max(0.0, max(p[2] + p[4] - 2 * M for p in placed))
            hy = min(W, min(p[2] + 2 * M for p in placed))
            if lx > hx + eps or ly > hy + eps:
                return None
            for x, y in ((lx, ly), (hx, ly), (lx, hy), (hx, hy)):
                if all(min(p[1] + p[3] - x, x - p[1],
                           p[2] + p[4] - y, y - p[2]) <= M + eps for p in placed):
                    return (max(0.0, x), max(0.0, y))
            return None

        def remember(placed, counts=None):
            nonlocal best, best_value, last_gain, inactive
            if placed is None or len(placed) > K:
                return None
            counts = counts_of(placed) if counts is None else counts
            if any(counts[i] < lower[i] or counts[i] > upper[i] for i in range(N)):
                return None
            score = math.fsum(value[p[0]] for p in placed)
            if score > best_value:
                active = []
                for p in sorted(placed, key=lambda q: q[1]):
                    if clock() >= verification_end:
                        return None
                    if (p[1] < -eps or p[2] < -eps or p[1] + p[3] > L + eps
                            or p[2] + p[4] > W + eps):
                        return None
                    next_active = []
                    for j, q in enumerate(active):
                        if j & 127 == 0 and clock() >= verification_end:
                            return None
                        if q[1] + q[3] <= p[1] + eps:
                            continue
                        if q[2] + q[4] > p[2] + eps and p[2] + p[4] > q[2] + eps:
                            return None
                        next_active.append(q)
                    next_active.append(p)
                    active = next_active
                if tight_m and any(not compatible(p, placed[:j])
                                   for j, p in enumerate(placed)):
                    return None
                point = unused_position(placed)
                if point is None:
                    return None
                best, best_value = placed[:], score
                inactive = point
                last_gain = clock()
            return score

        def carve(free, p, until=None):
            stop = end if until is None else min(end, until)
            x, y, right, top = p[1], p[2], p[1] + p[3], p[2] + p[4]
            old, fragments = [], []
            for index, (a, b, c, d) in enumerate(free):
                if index & 127 == 0 and clock() >= stop:
                    return None
                if right <= a + eps or c <= x + eps or top <= b + eps or d <= y + eps:
                    old.append((a, b, c, d))
                    continue
                if x > a + eps:
                    fragments.append((a, b, x, d))
                if right < c - eps:
                    fragments.append((right, b, c, d))
                if y > b + eps:
                    fragments.append((a, b, c, y))
                if top < d - eps:
                    fragments.append((a, top, c, d))
            # Original free regions do not contain each other; only new fragments are checked for containment, and overlapping maximal free regions are kept.
            fragments.sort(key=lambda z: -(z[2] - z[0]) * (z[3] - z[1]))
            checks = 0
            for a, b, c, d in fragments:
                if (c - a + eps < minimum_side or d - b + eps < minimum_side
                        or (c - a) * (d - b) + eps < minimum_area):
                    continue
                contained = False
                for q in old:
                    checks += 1
                    if checks & 255 == 0 and clock() >= stop:
                        return None
                    if a >= q[0] - eps and b >= q[1] - eps and c <= q[2] + eps and d <= q[3] + eps:
                        contained = True
                        break
                if not contained:
                    old.append((a, b, c, d))
            return old

        def spaces(placed, until=None):
            stop = end if until is None else min(end, until)
            free = [(0.0, 0.0, L, W)]
            for p in placed:
                if clock() >= stop:
                    return None
                free = carve(free, p, stop)
                if free is None:
                    return None
            return free

        def parameters(first=False):
            if first:
                return (0.0, 0.0, 0, [1.0] * N, 0)
            # Continuously blend piece value and geometric fit; zero weight also uses the same repair and recombination flow.
            alpha = rng.choice((0.0, 1.0, 0.45, 0.72, 0.2, 1.25))
            if rng.random() < 0.5:
                alpha = min(1.45, max(0.0, rng.gauss(0.8, 0.3)))
            weight = 0.0 if rng.random() < 0.23 else rng.uniform(0.08, 0.7 + 0.09 * effort)
            noise = min(0.6, 0.07 + 0.06 * effort)
            return (alpha, weight, rng.randrange(3),
                    [math.exp(rng.uniform(-noise, noise)) for _ in range(N)], rng.randrange(5))

        def priorities(params):
            alpha, _, _, noise, _ = params
            logs = [(log_value[i] - alpha * log_area[i] + math.log(noise[i]), i)
                    for i in possible if value[i] > 0]
            peak = max((z[0] for z in logs), default=0.0)
            return sorted(((math.exp(max(-700.0, v - peak)), i) for v, i in logs), reverse=True)

        def selected_counts(params):
            target = lower[:]
            slots = K - required_count
            room = max(0.0, board * rng.uniform(0.90, 1.0)
                       - math.fsum(area[i] * lower[i] for i in possible))
            for _, i in priorities(params):
                if clock() >= end or slots <= 0:
                    break
                take = min(upper[i] - target[i], slots, max(0, int((room + eps) / area[i])))
                target[i] += take
                slots -= take
                room = max(0.0, room - take * area[i])
            return target

        def mandatory_options(placed, free, counts, noise, until):
            # Handle required types with fewer usable free regions first, then backtrack over positions.
            selected = None
            selected_key = None
            selected_options = None
            for i in possible:
                need = lower[i] - counts[i]
                if need <= 0:
                    continue
                options = []
                seen = set()
                capacity = 0
                for a, b, c, d in free:
                    fw, fh = c - a, d - b
                    for w, h, rot in orientations[i]:
                        if w > fw + eps or h > fh + eps:
                            continue
                        capacity += int((fw + eps) / w) * int((fh + eps) / h)
                        left, bottom, right, top = a, b, c - w, d - h
                        if tight_m and placed:
                            left = max(left, max(q[1] + q[3] - M for q in placed))
                            right = min(right, min(q[1] + M - w for q in placed))
                            bottom = max(bottom, max(q[2] + q[4] - M for q in placed))
                            top = min(top, min(q[2] + M - h for q in placed))
                            if left > right + eps or bottom > top + eps:
                                continue
                        for x, y in ((left, bottom), (right, bottom),
                                     (left, top), (right, top)):
                            x, y = max(0.0, x), max(0.0, y)
                            key = (round(x, 13), round(y, 13), rot)
                            if key in seen:
                                continue
                            seen.add(key)
                            p = (i, x, y, w, h, rot)
                            if not compatible(p, placed):
                                continue
                            dx, dy = max(0.0, fw - w), max(0.0, fh - h)
                            cost = (min(dx, dy) / math.sqrt(area[i])
                                    + 0.12 * (fw * fh / area[i] - 1)
                                    + 0.035 * (x / L + y / W))
                            options.append((cost * noise[i], p))
                    if clock() >= until:
                        return None
                if not options:
                    return []
                key = (capacity / need * noise[i], -area[i] * noise[i])
                if selected_key is None or key < selected_key:
                    selected_key, selected, selected_options = key, i, options
            if selected is None:
                return []
            selected_options.sort(key=lambda item: item[0])
            return [p for _, p in selected_options]

        def satisfy(placed, free, counts, noise, until):
            if all(counts[i] >= lower[i] for i in range(N)):
                return placed, free, counts
            if (sum(max(0, lower[i] - counts[i]) for i in range(N)) > K - len(placed)
                    or sum(max(0, lower[i] - counts[i]) * area[i] for i in range(N))
                    > board - sum(area[p[0]] for p in placed) + eps):
                return None
            options = mandatory_options(placed, free, counts, noise, until)
            if options is None:
                return None
            stack = [[placed, free, counts, options, 0]]
            while stack and clock() < until:
                frame = stack[-1]
                ps, fs, cs, opts, at = frame
                if at >= len(opts):
                    stack.pop()
                    continue
                p = opts[at]
                frame[4] += 1
                next_placed = ps + [p]
                next_counts = cs[:]
                next_counts[p[0]] += 1
                next_free = carve(fs, p, until)
                if next_free is None:
                    return None
                if all(next_counts[i] >= lower[i] for i in range(N)):
                    return next_placed, next_free, next_counts
                opts = mandatory_options(next_placed, next_free, next_counts, noise, until)
                if opts:
                    stack.append([next_placed, next_free, next_counts, opts, 0])
            return None

        def refill(kept, params, until=None, corner=0, target=None, free_seed=None):
            stop = end if until is None else min(end, until)
            counts = counts_of(kept)
            if any(counts[i] > upper[i] for i in range(N)) or len(kept) > K:
                return None
            if clock() >= stop:
                return kept if all(c >= q for c, q in zip(counts, lower)) else None
            free = list(free_seed) if free_seed is not None else spaces(kept, stop)
            if free is None:
                return None
            alpha, weight, mode, noise, rule = params
            missing = sum(max(0, lower[i] - counts[i]) for i in range(N))
            slice_size = budget * (0.18 if best_value == -math.inf else 0.009)
            slice_size *= 1.0 + missing / max(1, required_count)
            repaired = satisfy(list(kept), free, counts, noise,
                               min(stop, clock() + slice_size))
            if repaired is None:
                return None
            placed, free, counts = repaired
            remember(placed, counts)
            ranked = priorities(params)
            scan = 0
            max_right = max((p[1] + p[3] for p in placed), default=0.0)
            max_top = max((p[2] + p[4] for p in placed), default=0.0)
            blocked = set()

            def choose(candidates, geometric=False):
                nonlocal scan
                chosen, best_key = None, None
                if tight_m and placed:
                    low_x = max(q[1] + q[3] - M for q in placed)
                    high_x = min(q[1] + M for q in placed)
                    low_y = max(q[2] + q[4] - M for q in placed)
                    high_y = min(q[2] + M for q in placed)
                else:
                    low_x = low_y = 0.0
                    high_x, high_y = L, W
                for priority, i in candidates:
                    if counts[i] >= upper[i] or i in blocked:
                        continue
                    if best_key is not None and not geometric and priority * (1 + weight) < best_key[0]:
                        break
                    fitted = False
                    for a, b, c, d in free:
                        fw, fh = c - a, d - b
                        for w, h, rot in orientations[i]:
                            scan += 1
                            if scan & 127 == 0 and clock() >= stop:
                                return None
                            if w > fw + eps or h > fh + eps:
                                continue
                            left, bottom = max(a, low_x), max(b, low_y)
                            right, top = min(c, high_x) - w, min(d, high_y) - h
                            if left > right + eps or bottom > top + eps:
                                continue
                            x = left if corner % 2 == 0 else max(left, right)
                            y = bottom if corner < 2 else max(bottom, top)
                            anchors = ((x, y),)
                            if rule == 3:
                                anchors = ((left, bottom), (max(left, right), bottom),
                                           (left, max(bottom, top)), (max(left, right), max(bottom, top)))
                            dx, dy = max(0.0, fw - w), max(0.0, fh - h)
                            short, long_side = min(dx, dy), max(dx, dy)
                            fraction = min(1.0, area[i] / (fw * fh))
                            snug = 1.0 / (1.0 + short / root_area[i])
                            alignment = 0.5 * ((w / fw) ** 4 + (h / fh) ** 4)
                            fitted = True
                            for x, y in anchors:
                                if mode == 0:
                                    fit = 0.5 * snug + 0.3 * alignment + 0.2 * fraction
                                elif mode == 1:
                                    fit = 0.3 * snug + 0.7 * fraction
                                else:
                                    fit = 0.3 * snug + 0.4 * alignment + 0.3 * (1 - (y + h) / W)
                                merit = priority * (1.0 + weight * fit)
                                if best_key is not None and not geometric and merit < best_key[0]:
                                    continue
                                waste = fw * fh - area[i]
                                if rule == 0:
                                    shape = (-short, -long_side, -y, -x)
                                elif rule == 1:
                                    shape = (-waste, -short, -long_side, -y, -x)
                                elif rule == 2:
                                    shape = (-(y + h), -(x + w), -short, -long_side)
                                elif rule == 3:
                                    contact = (h if x <= eps or abs(x + w - L) <= eps else 0.0)
                                    contact += w if y <= eps or abs(y + h - W) <= eps else 0.0
                                    for j, q in enumerate(placed):
                                        if j & 127 == 0 and clock() >= stop:
                                            return None
                                        if abs(x + w - q[1]) <= eps or abs(q[1] + q[3] - x) <= eps:
                                            contact += max(0.0, min(y + h, q[2] + q[4]) - max(y, q[2]))
                                        if abs(y + h - q[2]) <= eps or abs(q[2] + q[4] - y) <= eps:
                                            contact += max(0.0, min(x + w, q[1] + q[3]) - max(x, q[1]))
                                    shape = (contact, -waste, -(y + h), -(x + w))
                                else:
                                    nr, nt = max(max_right, x + w), max(max_top, y + h)
                                    shape = (-nr * nt, -max(nr, nt), -waste, -y, -x)
                                key = shape if geometric else (merit,) + shape
                                if best_key is None or key > best_key:
                                    best_key, chosen = key, (i, max(0.0, x), max(0.0, y), w, h, rot)
                    if not fitted:
                        blocked.add(i)
                return chosen

            def append_piece(p):
                nonlocal free, max_right, max_top
                placed.append(p)
                counts[p[0]] += 1
                max_right = max(max_right, p[1] + p[3])
                max_top = max(max_top, p[2] + p[4])
                free = carve(free, p, stop)

            if target is not None:
                style = rng.randrange(4)
                order = []
                for i in possible:
                    a, b = sizes[i]
                    measure = (area[i], max(a, b), min(a, b), a + b)[style]
                    for _ in range(max(0, min(target[i], upper[i]) - counts[i])):
                        if clock() >= stop:
                            return placed
                        order.append((-measure * noise[i] * rng.uniform(0.9, 1.1), i))
                order.sort()
                for _, i in order:
                    if clock() >= stop or len(placed) >= K or free is None:
                        return placed
                    p = choose(((1.0, i),), True)
                    if p is not None:
                        append_piece(p)
            while free and len(placed) < K and clock() < stop:
                chosen = choose(ranked)
                if chosen is None:
                    break
                append_piece(chosen)
            return placed

        def compact(placed):
            if tight_m or len(placed) < 2:
                return placed
            stop = min(end, clock() + budget / (50.0 + math.sqrt(len(placed))))
            ps = sorted(placed, key=lambda p: (p[1] + p[2], p[2], p[1]))
            changed = True
            while changed and clock() < stop:
                changed = False
                for k, p in enumerate(ps):
                    if clock() >= stop:
                        return ps
                    i, x, y, w, h, rot = p
                    nx = 0.0
                    for j, q in enumerate(ps):
                        if j & 127 == 0 and clock() >= stop:
                            return ps
                        if (j != k and q[2] < y + h - eps and y < q[2] + q[4] - eps
                                and q[1] + q[3] <= x + eps):
                            nx = max(nx, q[1] + q[3])
                    nx = min(x, nx)
                    ny = 0.0
                    for j, q in enumerate(ps):
                        if j & 127 == 0 and clock() >= stop:
                            return ps
                        if (j != k and q[1] < nx + w - eps and nx < q[1] + q[3] - eps
                                and q[2] + q[4] <= y + eps):
                            ny = max(ny, q[2] + q[4])
                    ny = min(y, ny)
                    if x - nx > eps or y - ny > eps:
                        changed = True
                        ps[k] = (i, nx, ny, w, h, rot)
            return ps

        def reflect(placed):
            flip_x, flip_y = rng.randrange(2), rng.randrange(2)
            return [(i, max(0.0, L - x - w) if flip_x else x,
                     max(0.0, W - y - h) if flip_y else y, w, h, r)
                    for i, x, y, w, h, r in placed]

        def eject(placed, size, params, until=None, protected=None):
            stop = min(end if until is None else until,
                       clock() + budget / (42.0 + 3.0 * math.sqrt(len(placed) + N + 1)))
            protected = set() if protected is None else protected
            ranked = [i for _, i in priorities(params)]
            if not ranked or clock() >= stop:
                return None
            i = ranked[min(len(ranked) - 1, int(rng.random() ** 1.8 * len(ranked)))]
            counts = counts_of(placed)
            batch = 1
            if rng.random() < min(0.5, 0.15 + 0.035 * effort):
                batch += int(rng.random() ** 2 * min(size, upper[i] - 1))
            anchors = [p for p in placed if p not in protected]
            choices = []
            for w, h, rot in orientations[i]:
                across = min(int((L + eps) / w), batch,
                             max(1, int(math.sqrt(batch * h / w))))
                down = min(int((W + eps) / h), max(1, batch // across))
                if rng.random() < 0.5:
                    down = min(int((W + eps) / h), batch,
                               max(1, int(math.sqrt(batch * w / h))))
                    across = min(int((L + eps) / w), max(1, batch // down))
                bw, bh = across * w, down * h
                points = [(0.0, 0.0), (L - bw, 0.0), (0.0, W - bh), (L - bw, W - bh)]
                for p in anchors:
                    if clock() >= stop:
                        break
                    x, y, right, top = p[1], p[2], p[1] + p[3], p[2] + p[4]
                    points.extend(((x, y), (right - bw, y), (x, top - bh),
                                   (right - bw, top - bh), (right, y), (x, top),
                                   (x - bw, y), (x, y - bh)))
                seen = set()
                for x, y in points:
                    x, y = max(0.0, min(L - bw, x)), max(0.0, min(W - bh, y))
                    if (x, y) not in seen:
                        seen.add((x, y))
                        choices.append((x, y, w, h, rot, across, down))
            rng.shuffle(choices)
            packing_density = (math.fsum(value[p[0]] for p in placed)
                               / max(eps, math.fsum(area[p[0]] for p in placed)))
            best_move, best_merit = None, -math.inf
            for x, y, w, h, rot, across, down in choices:
                if clock() >= stop:
                    break
                number = across * down
                bw, bh = across * w, down * h
                victims = set()
                left = counts[:]
                blocked = False
                for j, p in enumerate(placed):
                    if j & 127 == 0 and clock() >= stop:
                        blocked = True
                        break
                    if (p[1] + p[3] > x + eps and x + bw > p[1] + eps
                            and p[2] + p[4] > y + eps and y + bh > p[2] + eps):
                        if p in protected:
                            blocked = True
                            break
                        victims.add(j)
                        left[p[0]] -= 1
                if blocked:
                    continue
                excess = left[i] + number - upper[i]
                if excess > 0:
                    same = sorted((j for j, p in enumerate(placed)
                                   if j not in victims and p[0] == i and p not in protected),
                                  key=lambda j: abs(placed[j][1] - x) + abs(placed[j][2] - y))
                    if len(same) < excess:
                        continue
                    for j in same[:excess]:
                        victims.add(j)
                        left[i] -= 1
                left[i] += number
                overflow = len(placed) - len(victims) + number - K
                if overflow > 0:
                    removable = sorted((j for j, p in enumerate(placed)
                                        if j not in victims and p not in protected),
                                       key=lambda j: value[placed[j][0]])
                    for j in removable:
                        tid = placed[j][0]
                        if left[tid] > lower[tid]:
                            victims.add(j)
                            left[tid] -= 1
                            overflow -= 1
                            if overflow <= 0:
                                break
                    if overflow > 0:
                        continue
                deficit = sum(max(0, lower[t] - left[t]) for t in possible)
                if (len(placed) - len(victims) + number + deficit > K
                        or len(victims) > max(size, number)):
                    continue
                loss = math.fsum(value[placed[j][0]] for j in victims)
                freed = math.fsum(area[placed[j][0]] for j in victims)
                merit = (number * value[i] - loss
                         + 0.4 * packing_density * max(0.0, freed - number * area[i])
                         - 0.03 * abs(packing_density) * board * deficit / max(1, len(placed)))
                if merit <= best_merit:
                    continue
                kept = [p for j, p in enumerate(placed) if j not in victims]
                patch = [(i, x + col * w, y + row * h, w, h, rot)
                         for row in range(down) for col in range(across)]
                if tight_m:
                    trial = kept[:]
                    for q in patch:
                        if not compatible(q, trial):
                            blocked = True
                            break
                        trial.append(q)
                    if blocked:
                        continue
                best_merit, best_move = merit, kept + patch
            return best_move

        def neighborhood(base, op, size, params, stall):
            placed = reflect(base)
            n = len(placed)
            if op == 5:
                return eject(placed, size, params)
            if not placed:
                return []
            anchor = rng.choice(placed)
            cx, cy = anchor[1] + anchor[3] / 2, anchor[2] + anchor[4] / 2
            if op == 0 or op == 4:
                aspect = math.exp(rng.uniform(-1.3, 1.3))
                rank = sorted(range(n), key=lambda k:
                              abs(placed[k][1] + placed[k][3] / 2 - cx) / L * aspect
                              + abs(placed[k][2] + placed[k][4] / 2 - cy) / W / aspect)
                if op == 4:
                    size = min(n, max(size, int(n * rng.uniform(0.22, min(0.95, 0.45 + stall)))))
                    if rng.random() < 0.12:
                        size = n
            elif op == 1:
                axis = rng.randrange(2)
                center = cx if axis == 0 else cy
                rank = sorted(range(n), key=lambda k:
                              abs(placed[k][1 + axis] + placed[k][3 + axis] / 2 - center))
            elif op == 2:
                power = rng.uniform(0.3, 1.3)
                rank = sorted(range(n), key=lambda k:
                              value[placed[k][0]] / area[placed[k][0]] ** power
                              * rng.uniform(0.6, 1.4))
            else:
                target = anchor[0]
                rank = sorted(range(n), key=lambda k:
                              (placed[k][0] != target, rng.random()))
            remove = set(rank[:max(1, size)])
            kept = [p for k, p in enumerate(placed) if k not in remove]
            if rng.random() < min(0.45, 0.10 + 0.06 * effort):
                kept = compact(kept)
            return kept

        def recombine(source, params, stall):
            # External pieces are fixed and the candidate pool contains the original layout, so candidate optimisation never loses the original feasible solution.
            begun = clock()
            local_slice = budget / (8.0 + math.sqrt(len(source) + N + 1))
            stop = min(end, begun + local_slice * (0.75 + 0.5 * stall))
            if not source or stop <= begun:
                return source
            parent = reflect(source)
            n = len(parent)
            radius = max(2, int((1.0 + math.sqrt(budget)) * math.sqrt(math.log1p(n))))
            amount = min(n, max(2, int(radius * rng.uniform(0.45, 1.0))))
            pivot = rng.choice(parent)
            cx, cy = pivot[1] + pivot[3] / 2, pivot[2] + pivot[4] / 2
            aspect = math.exp(rng.uniform(-1.2, 1.2))
            order = sorted(range(n), key=lambda j:
                           abs(parent[j][1] + parent[j][3] / 2 - cx) / L * aspect
                           + abs(parent[j][2] + parent[j][4] / 2 - cy) / W / aspect)
            removed = set(order[:amount])
            kept = [p for j, p in enumerate(parent) if j not in removed]
            fixed = set(kept)
            fixed_counts = counts_of(kept)
            fixed_value = math.fsum(value[p[0]] for p in kept)
            local_best = parent
            local_value = math.fsum(value[p[0]] for p in parent) - fixed_value
            catalogue = {}
            candidate_limit = max(amount, int((10.0 + 5.0 * effort) * math.sqrt(amount + 1)))

            def collect(layout):
                for p in layout:
                    if p in fixed:
                        continue
                    key = (p[0], round(p[1], 13), round(p[2], 13), p[5])
                    if key not in catalogue and len(catalogue) < candidate_limit:
                        catalogue[key] = p

            collect(parent)
            generate_stop = begun + (stop - begun) * 0.43
            shared_free = spaces(kept, generate_stop)
            if shared_free is None:
                return local_best
            trial_params = params
            corner = 0
            target_counts = counts_of(parent)
            while clock() < generate_stop and len(catalogue) < candidate_limit:
                if rng.random() < 0.24:
                    # Keep the external layout and only remove blockers locally; candidates with incomplete quotas are also passed to the graph search for repair.
                    trial = eject(parent, amount, trial_params, generate_stop, fixed)
                else:
                    target = target_counts if rng.random() < 0.35 else None
                    trial = refill(kept, trial_params, until=generate_stop, corner=corner,
                                   target=target, free_seed=shared_free)
                if trial is not None:
                    collect(trial)
                    score = remember(trial)
                    if score is not None and score - fixed_value > local_value:
                        local_best, local_value = trial, score - fixed_value
                trial_params = parameters()
                corner = rng.randrange(4)
            candidates = sorted(catalogue.values(), key=lambda p: (-value[p[0]], p))
            m = len(candidates)
            if m <= amount or clock() >= stop:
                return local_best
            tids = sorted(set(p[0] for p in candidates))
            t_index = {tid: j for j, tid in enumerate(tids)}
            cap = tuple(upper[i] - fixed_counts[i] for i in tids)
            need = tuple(max(0, lower[i] - fixed_counts[i]) for i in tids)
            kinds = [t_index[p[0]] for p in candidates]
            group = [0] * len(tids)
            weights = [value[p[0]] for p in candidates]
            item_area = [area[p[0]] for p in candidates]
            type_value = [value[i] for i in tids]
            type_area = [area[i] for i in tids]
            density_order = sorted(range(len(tids)),
                                   key=lambda j: type_value[j] / type_area[j], reverse=True)
            value_order = sorted(range(len(tids)), key=lambda j: type_value[j], reverse=True)
            conflict = [1 << j for j in range(m)]
            full_mask = (1 << m) - 1
            for j, p in enumerate(candidates):
                if clock() >= stop:
                    return local_best
                group[kinds[j]] |= 1 << j
                for k in range(j):
                    q = candidates[k]
                    overlap = (p[1] + p[3] > q[1] + eps and q[1] + q[3] > p[1] + eps
                               and p[2] + p[4] > q[2] + eps and q[2] + q[4] > p[2] + eps)
                    bad_m = tight_m and max(p[1] + p[3] - q[1], q[1] + q[3] - p[1],
                                            p[2] + p[4] - q[2], q[2] + q[4] - p[2]) > M + eps
                    if overlap or bad_m or (kinds[j] == kinds[k] and cap[kinds[j]] == 1):
                        conflict[j] |= 1 << k
                        conflict[k] |= 1 << j
                    if k % 256 == 0 and clock() >= stop:
                        return local_best
            for j in range(len(tids)):
                if cap[j] <= 0 or (type_value[j] <= 0 and need[j] == 0):
                    full_mask &= ~group[j]
            free_area = max(0.0, board - math.fsum(area[p[0]] for p in kept))
            slots = K - len(kept)
            # The state is only the local candidate set; recursion is avoided so that larger candidate pools do not trigger recursion-depth errors.
            stack = [(full_mask, 0, 0.0, slots, cap, need, free_area)]
            while stack and clock() < stop:
                mask, selected, gain, left_slots, caps, needs, room = stack.pop()
                required = sum(needs)
                if required == 0 and gain > local_value + 1e-9:
                    picked = []
                    bits = selected
                    while bits:
                        bit = bits & -bits
                        picked.append(candidates[bit.bit_length() - 1])
                        bits ^= bit
                    trial = kept + picked
                    score = remember(trial)
                    if score is not None and unused_position(trial) is not None:
                        local_best, local_value = trial, gain
                if not mask or left_slots <= 0 or required > left_slots:
                    continue
                available = []
                feasible = True
                mandatory_area = mandatory_value = 0.0
                simple_bound = 0.0
                branch_mask = mask
                scarcity = math.inf
                for j in range(len(tids)):
                    options = mask & group[j]
                    number = min(caps[j], options.bit_count())
                    available.append(number)
                    if number < needs[j]:
                        feasible = False
                        break
                    mandatory_area += needs[j] * type_area[j]
                    mandatory_value += needs[j] * type_value[j]
                    simple_bound += number * max(0.0, type_value[j])
                    if needs[j] and number / needs[j] < scarcity:
                        scarcity = number / needs[j]
                        branch_mask = options
                if (not feasible or mandatory_area > room + eps
                        or gain + simple_bound <= local_value + 1e-9):
                    continue
                # Two independent fractional knapsack upper bounds relax area and slots respectively; lower-bound pieces are reserved first.
                remaining_area = max(0.0, room - mandatory_area)
                area_bound = gain + mandatory_value
                for j in density_order:
                    if type_value[j] <= 0:
                        continue
                    take = min(available[j] - needs[j], remaining_area / type_area[j])
                    area_bound += take * type_value[j]
                    remaining_area -= take * type_area[j]
                if area_bound <= local_value + 1e-9:
                    continue
                remaining_slots = left_slots - required
                slot_bound = gain + mandatory_value
                for j in value_order:
                    if type_value[j] <= 0:
                        continue
                    take = min(available[j] - needs[j], remaining_slots)
                    slot_bound += take * type_value[j]
                    remaining_slots -= take
                    if remaining_slots == 0:
                        break
                if slot_bound <= local_value + 1e-9:
                    continue
                # At most one candidate per conflict clique; covering in descending weight order gives a safe value upper bound.
                todo = mask
                clique_bound = gain
                while todo:
                    bit = todo & -todo
                    vertex = bit.bit_length() - 1
                    clique_bound += max(0.0, weights[vertex])
                    if clique_bound > local_value + 1e-9:
                        break
                    todo ^= bit
                    common = todo & conflict[vertex]
                    while common:
                        next_bit = common & -common
                        other = next_bit.bit_length() - 1
                        todo ^= next_bit
                        common &= conflict[other] & todo
                if clique_bound <= local_value + 1e-9:
                    continue
                # Branch on scarce types first; otherwise use a high-conflict-degree candidate to shrink both subproblems.
                choices = branch_mask
                chosen = -1
                branch_key = None
                while choices:
                    bit = choices & -choices
                    j = bit.bit_length() - 1
                    key = ((conflict[j] & mask).bit_count(), weights[j])
                    if branch_key is None or key > branch_key:
                        chosen, branch_key = j, key
                    choices ^= bit
                bit = 1 << chosen
                reduced = mask ^ bit
                stack.append((reduced, selected, gain, left_slots, caps, needs, room))
                kind = kinds[chosen]
                if item_area[chosen] > room + eps:
                    continue
                next_caps, next_needs = list(caps), list(needs)
                next_caps[kind] -= 1
                next_needs[kind] = max(0, next_needs[kind] - 1)
                next_mask = reduced & ~conflict[chosen]
                if next_caps[kind] == 0 or (type_value[kind] <= 0 and next_needs[kind] == 0):
                    next_mask &= ~group[kind]
                stack.append((next_mask, selected | bit, gain + weights[chosen],
                              left_slots - 1, tuple(next_caps), tuple(next_needs),
                              max(0.0, room - item_area[chosen])))
            return local_best

        def add_pool(placed, score):
            if score is None:
                return
            counts = counts_of(placed)
            fingerprint = (tuple(counts), round(sum(p[1] * p[3] * p[4] for p in placed), 7),
                           round(sum(p[2] * p[3] * p[4] for p in placed), 7))
            if any(entry[2] == fingerprint for entry in pool):
                return
            pool.append((score, placed[:], fingerprint))
            pool.sort(key=lambda entry: entry[0], reverse=True)
            if len(pool) > pool_limit:
                worst = pool[-1][0]
                tied = [j for j in range(1, len(pool)) if pool[j][0] <= worst + 1e-9]
                pool.pop(rng.choice(tied))

        if (required_count > K or any(lower[i] > upper[i] for i in range(N))
                or math.fsum(lower[i] * area[i] for i in possible) > board + eps):
            return {"placements": [_unused() for _ in range(K)]}
        initial = refill([], parameters(True))
        initial_score = remember(initial)
        if initial_score is not None:
            current, current_value = initial, initial_score
            add_pool(initial, initial_score)
        else:
            current, current_value = best[:], best_value
        weights = [1.0] * 8
        weights[6] = 1.0 + 0.10 * effort
        joint_spent = 0.0
        sequence = 0
        alpha_cycle = (0.0, 1.0, 0.45, 0.72, 0.2, 1.25)
        while clock() < end:
            now = clock()
            old_best = best_value
            stall = min(1.0, (now - last_gain) / max(0.001, budget * 0.3))
            params = parameters()
            pick = rng.random() * sum(weights)
            op = 0
            while op < len(weights) - 1 and pick > weights[op]:
                pick -= weights[op]
                op += 1
            joint_share = effort / (9.0 + 2.0 * effort) + 0.06 * stall
            if op == 6 and joint_spent >= joint_share * (now - start):
                op = rng.randrange(6)
            target = None
            if best_value == -math.inf:
                kept = []
            else:
                draw = rng.random()
                if draw < 0.46:
                    base = best
                elif draw < 0.79 or not pool:
                    base = current
                else:
                    base = rng.choice(pool)[1]
                fraction = min(0.8, (0.025 + 0.027 * effort) * (0.35 + rng.random()) * (1 + 2 * stall))
                size = max(1, int(1 + len(base) * fraction))
                if now - start < budget * 0.16:
                    # Layouts from multiple starts and the subsequent spatial repair share the elite pool and candidate graph rather than being solved separately.
                    op = 7
                    kept = []
                    params = (alpha_cycle[sequence % len(alpha_cycle)],
                              params[1] if sequence % 2 else 0.0, params[2],
                              params[3], sequence % 5)
                elif op == 7:
                    kept = []
                    draw = rng.random()
                    if draw < 0.4:
                        target = counts_of(base)
                    elif draw < 0.8:
                        target = selected_counts(params)
                else:
                    kept = base if op == 6 else neighborhood(base, op, size, params, stall)
            sequence += 1
            if kept is None or clock() >= end:
                continue
            if op == 6 and best_value != -math.inf:
                joint_start = clock()
                candidate = recombine(kept, params, stall)
                joint_spent += clock() - joint_start
                # After graph recombination changes the selected pieces, continue backfilling; freed space immediately returns to the same repair flow.
                if candidate is not None and clock() < end:
                    candidate = refill(candidate, params)
            else:
                candidate = refill(kept, params, target=target)
            score = remember(candidate)
            reward = 0.2
            if score is not None:
                remaining = max(0.0, (end - clock()) / max(0.001, budget))
                temperature = max(1e-10, abs(best_value) * 0.008 * remaining * (0.2 + stall))
                accept = score >= current_value
                if not accept:
                    accept = rng.random() < math.exp(max(-700.0, (score - current_value) / temperature))
                if accept:
                    current, current_value = candidate, score
                    reward = 1.0
                if best_value > old_best:
                    reward = 5.0
                add_pool(candidate, score)
            weights[op] = 0.93 * weights[op] + 0.07 * reward
    except Exception:
        pass
    result = []
    for i, x, y, w, h, rot in best:
        result.append({"is_used": 1, "piece_type_id": int(i),
                       "x": float(max(0.0, x * scale)), "y": float(max(0.0, y * scale)),
                       "is_rotated": int(rot)})
    while len(result) < K:
        result.append({"is_used": 0, "piece_type_id": -1,
                       "x": float(inactive[0] * scale), "y": float(inactive[1] * scale),
                       "is_rotated": 0})
    return {"placements": result}


def _unused():
    return {"is_used": 0, "piece_type_id": -1, "x": 0.0, "y": 0.0, "is_rotated": 0}
