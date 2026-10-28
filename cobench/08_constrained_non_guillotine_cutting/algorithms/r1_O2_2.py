"""Quota repair with spatial large-neighbourhood packing
Keeps high-quality layouts and performs time-limited search through local removal, targeted space clearing, coordinate compaction and backfilling by combined value and free-region fit.
"""

import math
import random
import time


def solve(instance: dict, time_limit_s: float) -> dict:
    start = time.perf_counter()
    K = 0
    best = []
    scale = 1.0
    inactive = (0.0, 0.0)
    try:
        K = max(0, int(instance.get("max_slots", 0)))
        budget = max(0.0, float(time_limit_s))
        reserve = min(0.12, budget * 0.025) + min(budget * 0.015, K * 0.000003)
        end = start + max(0.0, budget - reserve)
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
        if L <= 0 or W <= 0 or any(a <= 0 or b <= 0 for a, b in sizes):
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

        def counts_of(placed):
            counts = [0] * N
            for p in placed:
                counts[p[0]] += 1
            return counts

        def compatible(p, placed):
            if not tight_m:
                return True
            _, x, y, w, h, _ = p
            for q in placed:
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
                point = unused_position(placed)
                if point is None:
                    return None
                best, best_value = placed[:], score
                inactive = point
                last_gain = time.perf_counter()
            return score

        def carve(free, p):
            # Free regions may overlap each other; only containment redundancy is removed, and they are not treated as additive capacity.
            x, y, right, top = p[1], p[2], p[1] + p[3], p[2] + p[4]
            fragments = []
            for a, b, c, d in free:
                if right <= a + eps or c <= x + eps or top <= b + eps or d <= y + eps:
                    fragments.append((a, b, c, d))
                    continue
                if x > a + eps:
                    fragments.append((a, b, x, d))
                if right < c - eps:
                    fragments.append((right, b, c, d))
                if y > b + eps:
                    fragments.append((a, b, c, y))
                if top < d - eps:
                    fragments.append((a, top, c, d))
            unique = set()
            candidates = []
            for rect in fragments:
                a, b, c, d = rect
                if (c - a + eps < minimum_side or d - b + eps < minimum_side
                        or (c - a) * (d - b) + eps < minimum_area or rect in unique):
                    continue
                unique.add(rect)
                candidates.append(rect)
            candidates.sort(key=lambda z: -(z[2] - z[0]) * (z[3] - z[1]))
            kept = []
            for index, rect in enumerate(candidates):
                if index % 32 == 0 and time.perf_counter() >= end:
                    return None
                a, b, c, d = rect
                if not any(a >= q[0] - eps and b >= q[1] - eps
                           and c <= q[2] + eps and d <= q[3] + eps for q in kept):
                    kept.append(rect)
            return kept

        def spaces(placed):
            free = [(0.0, 0.0, L, W)]
            for p in placed:
                if time.perf_counter() >= end:
                    return None
                free = carve(free, p)
                if free is None:
                    return None
            return free

        def parameters(first=False):
            if first:
                return (1.0, 0.45, 0, [1.0] * N)
            alpha = min(1.45, max(0.0, rng.gauss(0.88, 0.29)))
            if rng.random() < 1.0 / (4.0 + effort):
                alpha = 0.0
            weight = rng.uniform(0.12, 0.75 + 0.12 * effort)
            noise = min(0.6, 0.08 + 0.055 * effort)
            return (alpha, weight, rng.randrange(3),
                    [math.exp(rng.uniform(-noise, noise)) for _ in range(N)])

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
                        for x, y in ((a, b), (c - w, b), (a, d - h), (c - w, d - h)):
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
                    if time.perf_counter() >= until:
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
            while stack and time.perf_counter() < until:
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
                next_free = carve(fs, p)
                if next_free is None:
                    return None
                if all(next_counts[i] >= lower[i] for i in range(N)):
                    return next_placed, next_free, next_counts
                opts = mandatory_options(next_placed, next_free, next_counts, noise, until)
                if opts:
                    stack.append([next_placed, next_free, next_counts, opts, 0])
            return None

        def refill(kept, params, first=False):
            if time.perf_counter() >= end:
                return kept if all(c >= q for c, q in zip(counts_of(kept), lower)) else None
            free = spaces(kept)
            if free is None:
                return None
            counts = counts_of(kept)
            alpha, weight, mode, noise = params
            missing = sum(max(0, lower[i] - counts[i]) for i in range(N))
            slice_size = budget * (0.13 if best_value == -math.inf else 0.008)
            slice_size *= 1.0 + missing / max(1, required_count)
            repaired = satisfy(kept, free, counts, noise,
                               min(end, time.perf_counter() + max(0.001, slice_size)))
            if repaired is None:
                return None
            placed, free, counts = repaired
            remember(placed, counts)
            ranked = sorted(((value[i] / area[i] ** alpha * noise[i], i)
                             for i in possible if value[i] > 0), reverse=True)
            scan = 0
            while free and len(placed) < K and time.perf_counter() < end:
                chosen, best_key = None, None
                for priority, i in ranked:
                    if counts[i] >= upper[i]:
                        continue
                    if best_key is not None and priority * (1 + weight) < best_key[0]:
                        break
                    root_area = math.sqrt(area[i])
                    for a, b, c, d in free:
                        fw, fh = c - a, d - b
                        for w, h, rot in orientations[i]:
                            scan += 1
                            if scan % 128 == 0 and time.perf_counter() >= end:
                                return placed
                            if w > fw + eps or h > fh + eps:
                                continue
                            dx, dy = max(0.0, fw - w), max(0.0, fh - h)
                            fraction = min(1.0, area[i] / (fw * fh))
                            snug = 1.0 / (1.0 + min(dx, dy) / root_area)
                            alignment = 0.5 * ((w / fw) ** 4 + (h / fh) ** 4)
                            if mode == 0:
                                fit = 0.5 * snug + 0.3 * alignment + 0.2 * fraction
                            elif mode == 1:
                                fit = 0.3 * snug + 0.7 * fraction
                            else:
                                fit = 0.3 * snug + 0.4 * alignment + 0.3 * (1 - (b + h) / W)
                            score = priority * (1.0 + weight * fit)
                            key = (score, -min(dx, dy), -fw * fh, -(b + h), -a)
                            if best_key is None or key > best_key:
                                p = (i, a, b, w, h, rot)
                                if compatible(p, placed):
                                    best_key, chosen = key, p
                if chosen is None:
                    break
                placed = placed + [chosen]
                counts[chosen[0]] += 1
                next_free = carve(free, chosen)
                if next_free is None:
                    return placed
                free = next_free
            return placed

        def compact(placed):
            if tight_m or len(placed) < 2:
                return placed
            ps = sorted(placed, key=lambda p: (p[1] + p[2], p[2], p[1]))
            changed = True
            while changed and time.perf_counter() < end:
                changed = False
                for k, p in enumerate(ps):
                    if time.perf_counter() >= end:
                        return ps
                    i, x, y, w, h, rot = p
                    nx = 0.0
                    for j, q in enumerate(ps):
                        if (j != k and q[2] < y + h - eps and y < q[2] + q[4] - eps
                                and q[1] + q[3] <= x + eps):
                            nx = max(nx, q[1] + q[3])
                    nx = min(x, nx)
                    ny = 0.0
                    for j, q in enumerate(ps):
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

        def eject(placed, size, params):
            eligible = [i for i in possible if value[i] > 0 or lower[i] > 0]
            if not eligible:
                return None
            alpha, _, _, noise = params
            eligible.sort(key=lambda i: value[i] * noise[i] / area[i] ** alpha, reverse=True)
            i = eligible[min(len(eligible) - 1, int(rng.random() ** 1.7 * len(eligible)))]
            w, h, rot = rng.choice(orientations[i])
            batch = 1
            if rng.random() < 0.15 + 0.03 * effort:
                batch += int(rng.random() ** 2 * min(size, upper[i] - 1))
            across = min(int((L + eps) / w), batch,
                         max(1, int(math.sqrt(batch * h / w))))
            down = min(int((W + eps) / h), max(1, batch // across))
            if rng.random() < 0.5:
                down = min(int((W + eps) / h), batch,
                           max(1, int(math.sqrt(batch * w / h))))
                across = min(int((L + eps) / w), max(1, batch // down))
            number = across * down
            bw, bh = across * w, down * h
            if placed:
                anchor = rng.choice(placed)
                x = rng.choice((anchor[1], anchor[1] + anchor[3],
                                anchor[1] - bw, anchor[1] + anchor[3] - bw))
                y = rng.choice((anchor[2], anchor[2] + anchor[4],
                                anchor[2] - bh, anchor[2] + anchor[4] - bh))
            else:
                x = y = 0.0
            x, y = max(0.0, min(L - bw, x)), max(0.0, min(W - bh, y))
            kept = [p for p in placed if (p[1] + p[3] <= x + eps or x + bw <= p[1] + eps
                                         or p[2] + p[4] <= y + eps or y + bh <= p[2] + eps)]
            excess = sum(p[0] == i for p in kept) + number - upper[i]
            if excess > 0:
                same = sorted((k for k, p in enumerate(kept) if p[0] == i),
                              key=lambda k: abs(kept[k][1] - x) + abs(kept[k][2] - y))
                remove = set(same[:excess])
                kept = [p for k, p in enumerate(kept) if k not in remove]
            if len(kept) + number > K:
                kept.sort(key=lambda p: value[p[0]] / area[p[0]], reverse=True)
                kept = kept[:K - number]
            for row in range(down):
                for col in range(across):
                    p = (i, x + col * w, y + row * h, w, h, rot)
                    if not compatible(p, kept):
                        return None
                    kept.append(p)
            return kept

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
                pool.pop()

        if required_count > K or any(lower[i] > upper[i] for i in range(N)):
            return {"placements": [_unused() for _ in range(K)]}
        initial = refill([], parameters(True), True)
        initial_score = remember(initial)
        if initial_score is not None:
            current, current_value = initial, initial_score
            add_pool(initial, initial_score)
        else:
            current, current_value = best[:], best_value
        weights = [1.0] * 6
        while time.perf_counter() < end:
            now = time.perf_counter()
            stall = min(1.0, (now - last_gain) / max(0.001, budget * 0.3))
            params = parameters()
            pick = rng.random() * sum(weights)
            op = 0
            while op < len(weights) - 1 and pick > weights[op]:
                pick -= weights[op]
                op += 1
            if best_value == -math.inf:
                kept = []
            else:
                draw = rng.random()
                if draw < 0.45:
                    base = best
                elif draw < 0.78 or not pool:
                    base = current
                else:
                    base = rng.choice(pool)[1]
                fraction = min(0.8, (0.02 + 0.025 * effort) * (0.3 + rng.random()) * (1 + 2 * stall))
                size = max(1, int(1 + len(base) * fraction))
                kept = neighborhood(base, op, size, params, stall)
            if kept is None or time.perf_counter() >= end:
                continue
            old_best = best_value
            candidate = refill(kept, params)
            score = remember(candidate)
            reward = 0.2
            if score is not None:
                remaining = max(0.0, (end - time.perf_counter()) / max(0.001, budget))
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
