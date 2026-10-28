"""Dynamic geometric beam search with elite layout recombination
Combines quota backtracking, free-region union upper bounds, dynamic piece-selection placement beam search and an elite candidate conflict graph to improve layouts within the wall-clock limit.
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
        if not math.isfinite(budget):
            budget = 0.0
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
                active = []
                for p in sorted(placed, key=lambda q: q[1]):
                    if (p[1] < -eps or p[2] < -eps or p[1] + p[3] > L + eps
                            or p[2] + p[4] > W + eps):
                        return None
                    active = [q for q in active if q[1] + q[3] > p[1] + eps]
                    if any(q[2] + q[4] > p[2] + eps and
                           p[2] + p[4] > q[2] + eps for q in active):
                        return None
                    active.append(p)
                if tight_m and any(not compatible(p, placed[:j])
                                   for j, p in enumerate(placed)):
                    return None
                point = unused_position(placed)
                if point is None:
                    return None
                best, best_value = placed[:], score
                inactive = point
                last_gain = time.perf_counter()
            return score

        def carve(free, p, until=None):
            stop = end if until is None else min(end, until)
            # Free regions may overlap each other; only containment redundancy is removed, and they are not treated as additive capacity.
            x, y, right, top = p[1], p[2], p[1] + p[3], p[2] + p[4]
            fragments = []
            for index, (a, b, c, d) in enumerate(free):
                if index % 128 == 0 and time.perf_counter() >= stop:
                    return None
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
                if index % 32 == 0 and time.perf_counter() >= stop:
                    return None
                a, b, c, d = rect
                if not any(a >= q[0] - eps and b >= q[1] - eps
                           and c <= q[2] + eps and d <= q[3] + eps for q in kept):
                    kept.append(rect)
            return kept

        def spaces(placed, until=None):
            stop = end if until is None else min(end, until)
            free = [(0.0, 0.0, L, W)]
            for p in placed:
                if time.perf_counter() >= stop:
                    return None
                free = carve(free, p, stop)
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
                next_free = carve(fs, p, until)
                if next_free is None:
                    return None
                if all(next_counts[i] >= lower[i] for i in range(N)):
                    return next_placed, next_free, next_counts
                opts = mandatory_options(next_placed, next_free, next_counts, noise, until)
                if opts:
                    stack.append([next_placed, next_free, next_counts, opts, 0])
            return None

        def refill(kept, params, first=False, until=None, corner=0, prepared=None):
            stop = end if until is None else min(end, until)
            if time.perf_counter() >= stop:
                return kept if all(c >= q for c, q in zip(counts_of(kept), lower)) else None
            if prepared is None:
                free = spaces(kept, stop)
                counts = counts_of(kept)
            else:
                free, saved_counts = prepared
                counts = saved_counts[:]
            if free is None:
                return None
            alpha, weight, mode, noise = params
            missing = sum(max(0, lower[i] - counts[i]) for i in range(N))
            slice_size = budget * (0.13 if best_value == -math.inf else 0.008)
            slice_size *= 1.0 + missing / max(1, required_count)
            repaired = satisfy(kept, free, counts, noise,
                               min(stop, time.perf_counter() + max(0.001, slice_size)))
            if repaired is None:
                return None
            placed, free, counts = repaired
            remember(placed, counts)
            ranked = sorted(((value[i] / area[i] ** alpha * noise[i], i)
                             for i in possible if value[i] > 0), reverse=True)
            scan = 0
            while free and len(placed) < K and time.perf_counter() < stop:
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
                            if scan % 128 == 0 and time.perf_counter() >= stop:
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
                                x = a if corner % 2 == 0 else max(a, c - w)
                                y = b if corner < 2 else max(b, d - h)
                                if tight_m and placed:
                                    x = max(x, max(q[1] + q[3] - M for q in placed))
                                    y = max(y, max(q[2] + q[4] - M for q in placed))
                                    if x + w > c + eps or y + h > d + eps:
                                        continue
                                p = (i, x, y, w, h, rot)
                                if compatible(p, placed):
                                    best_key, chosen = key, p
                if chosen is None:
                    break
                placed = placed + [chosen]
                counts[chosen[0]] += 1
                next_free = carve(free, chosen, stop)
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

        def geometric_search(source, params, stall):
            # Candidate positions are regenerated after every placement, not limited to complete greedy layouts collected in advance.
            begun = time.perf_counter()
            stop = min(end, begun + budget * (0.8 + 0.4 * stall)
                       / (7.0 + math.sqrt(len(source) + N + 1)))
            if not source or begun >= stop:
                return source
            parent = reflect(source)
            n = len(parent)
            amount = min(n, max(2, int((1.0 + math.sqrt(budget))
                                      * math.sqrt(math.log1p(n)) * (0.65 + 0.7 * stall))))
            pivot = rng.choice(parent)
            cx, cy = pivot[1] + pivot[3] / 2, pivot[2] + pivot[4] / 2
            aspect = math.exp(rng.uniform(-1.4, 1.4))
            order = sorted(range(n), key=lambda j:
                           abs(parent[j][1] + parent[j][3] / 2 - cx) / L * aspect
                           + abs(parent[j][2] + parent[j][4] / 2 - cy) / W / aspect)
            removed = set(order[:amount])
            kept = [p for j, p in enumerate(parent) if j not in removed]
            free = spaces(kept, stop)
            if free is None:
                return source
            counts = counts_of(kept)
            local_best = parent
            local_value = math.fsum(value[p[0]] for p in parent)
            fixed_value = math.fsum(value[p[0]] for p in kept)
            initial_room = max(0.0, board - math.fsum(area[p[0]] for p in kept))
            width = max(2, int((1.0 + effort) * math.sqrt(amount + N + 1)))
            branches = max(3, int((2.0 + effort) * math.sqrt(len(possible) + 1)))
            same_counts = max(2, int(1.0 + math.sqrt(effort)))
            density_order = sorted(possible, key=lambda i: value[i] / area[i], reverse=True)
            value_order = sorted(possible, key=lambda i: value[i], reverse=True)
            alpha, weight, mode, noise = params
            optimism = rng.uniform(0.82, 0.98)
            expand_stop = begun + (stop - begun) * 0.85

            def union_area(rects, fallback):
                # Maximal empty rectangles overlap each other; the sweep line computes only the union, excluding fragments that can no longer be used.
                xs = sorted({z for a, b, c, d in rects for z in (a, c)})
                total = 0.0
                for left, right in zip(xs, xs[1:]):
                    if time.perf_counter() >= expand_stop:
                        return fallback
                    mid = (left + right) * 0.5
                    intervals = sorted((b, d) for a, b, c, d in rects if a < mid < c)
                    bottom = top = length = 0.0
                    for low, high in intervals:
                        if low > top:
                            length += top - bottom
                            bottom, top = low, high
                        else:
                            top = max(top, high)
                    total += (right - left) * (length + top - bottom)
                return min(fallback, total + eps * (1 + len(rects)))

            def bound(rects, cs, score, slots, room):
                available = [0] * N
                needs = [max(0, lower[i] - cs[i]) for i in range(N)]
                required = sum(needs)
                if required > slots:
                    return -math.inf
                for i in possible:
                    if time.perf_counter() >= expand_stop:
                        return math.inf
                    if cs[i] >= upper[i] or (value[i] <= 0 and needs[i] == 0):
                        continue
                    fits = any(w <= c - a + eps and h <= d - b + eps
                               for a, b, c, d in rects for w, h, r in orientations[i])
                    if fits:
                        available[i] = min(upper[i] - cs[i], slots)
                    if available[i] < needs[i]:
                        return -math.inf
                room = union_area(rects, room)
                mandatory_area = math.fsum(needs[i] * area[i] for i in possible)
                if mandatory_area > room + eps:
                    return -math.inf
                mandatory_value = math.fsum(needs[i] * value[i] for i in possible)
                remaining_area = max(0.0, room - mandatory_area)
                area_bound = score + mandatory_value
                for i in density_order:
                    if value[i] <= 0:
                        continue
                    take = min(available[i] - needs[i], remaining_area / area[i])
                    area_bound += take * value[i]
                    remaining_area = max(0.0, remaining_area - take * area[i])
                remaining_slots = slots - required
                slot_bound = score + mandatory_value
                for i in value_order:
                    if value[i] <= 0:
                        continue
                    take = min(available[i] - needs[i], remaining_slots)
                    slot_bound += take * value[i]
                    remaining_slots -= take
                    if remaining_slots == 0:
                        break
                return min(area_bound, slot_bound)

            def options(ps, rects, cs):
                needed = [i for i in possible if cs[i] < lower[i]]
                ids = needed if needed else [i for i in possible
                                            if cs[i] < upper[i] and value[i] > 0]
                by_type = []
                constrained = None
                selected = []
                if tight_m and ps:
                    min_x = max(p[1] + p[3] - M for p in ps)
                    max_x = min(p[1] + M for p in ps)
                    min_y = max(p[2] + p[4] - M for p in ps)
                    max_y = min(p[2] + M for p in ps)
                else:
                    min_x, max_x, min_y, max_y = 0.0, L, 0.0, W
                for i in ids:
                    choices = {}
                    priority = max(0.0, value[i]) / area[i] ** alpha * noise[i]
                    root_area = math.sqrt(area[i])
                    for a, b, c, d in rects:
                        if time.perf_counter() >= expand_stop:
                            return []
                        fw, fh = c - a, d - b
                        for w, h, rot in orientations[i]:
                            if w > fw + eps or h > fh + eps:
                                continue
                            left, right = max(a, min_x), min(c, max_x) - w
                            bottom, top = max(b, min_y), min(d, max_y) - h
                            if left > right + eps or bottom > top + eps:
                                continue
                            dx, dy = max(0.0, fw - w), max(0.0, fh - h)
                            snug = 1.0 / (1.0 + min(dx, dy) / root_area)
                            alignment = 0.5 * ((w / fw) ** 4 + (h / fh) ** 4)
                            fraction = min(1.0, area[i] / (fw * fh))
                            fit = 0.5 * snug + 0.3 * alignment + 0.2 * fraction
                            merit = priority * (1.0 + weight * fit)
                            if needed:
                                merit = snug + fraction + alignment
                            for x, y in ((left, bottom), (right, bottom),
                                         (left, top), (right, top)):
                                x, y = max(a, x), max(b, y)
                                p = (i, x, y, w, h, rot)
                                if not compatible(p, ps):
                                    continue
                                key = (round(x, 13), round(y, 13), rot)
                                rank = (merit, -min(dx, dy), -fw * fh,
                                        -((x / L + y / W) if mode != 2 else (x / L - y / W)))
                                if key not in choices or rank > choices[key][0]:
                                    choices[key] = (rank, p)
                    ranked = sorted(choices.values(), reverse=True)
                    if needed:
                        if not ranked:
                            return []
                        scarcity = (len(ranked) / (lower[i] - cs[i]), -area[i])
                        if constrained is None or scarcity < constrained:
                            constrained, selected = scarcity, ranked
                    elif ranked:
                        by_type.append(ranked)
                if needed:
                    return [p for rank, p in selected[:branches]]
                # Branch on different types first, then fill alternative positions of the same type, preventing a single density from monopolising the beam.
                heads = sorted((items[0] for items in by_type), reverse=True)
                tails = sorted((item for items in by_type for item in items[1:]), reverse=True)
                return [p for rank, p in (heads[:branches] + tails[:max(0, branches - len(heads))])]

            root_bound = bound(free, counts, fixed_value, K - len(kept), initial_room)
            beam = [(kept, free, counts, fixed_value, initial_room, root_bound)]

            def rollout(state, until):
                nonlocal local_best, local_value
                ps, fs, cs, score, room, ub = state
                trial = refill(ps, params, until=until, corner=rng.randrange(4),
                               prepared=(fs, cs))
                trial_value = remember(trial)
                if trial_value is not None and trial_value > local_value:
                    local_best, local_value = trial, trial_value

            while beam and time.perf_counter() < expand_stop:
                rollout(beam[0], min(expand_stop, time.perf_counter()
                                    + (stop - begun) / (3.0 + width)))
                children = []
                seen = set()
                for ps, fs, cs, score, room, ub in beam:
                    if time.perf_counter() >= expand_stop:
                        break
                    if ub <= local_value + 1e-9 or len(ps) >= K:
                        continue
                    for p in options(ps, fs, cs):
                        if time.perf_counter() >= expand_stop:
                            break
                        next_cs = cs[:]
                        next_cs[p[0]] += 1
                        next_ps = ps + [p]
                        next_score = score + value[p[0]]
                        next_room = max(0.0, room - area[p[0]])
                        if (next_score > local_value + 1e-9
                                and all(next_cs[i] >= lower[i] for i in possible)):
                            valid_score = remember(next_ps, next_cs)
                            if valid_score is not None:
                                local_best, local_value = next_ps, valid_score
                        next_fs = carve(fs, p, expand_stop)
                        if next_fs is None:
                            break
                        signature = (tuple(next_cs), tuple(sorted(
                            tuple(round(z, 13) for z in rect) for rect in next_fs)))
                        if signature in seen:
                            continue
                        seen.add(signature)
                        next_bound = bound(next_fs, next_cs, next_score,
                                           K - len(next_ps), next_room)
                        if next_bound <= local_value + 1e-9:
                            continue
                        rank = next_score + optimism * (next_bound - next_score)
                        state = (next_ps, next_fs, next_cs, next_score, next_room, next_bound)
                        children.append((rank, next_score, -len(next_fs), state))
                children.sort(key=lambda item: item[:3], reverse=True)
                beam, postponed = [], []
                multiplicity = {}
                for entry in children:
                    state = entry[3]
                    key = tuple(state[2])
                    if multiplicity.get(key, 0) < same_counts:
                        multiplicity[key] = multiplicity.get(key, 0) + 1
                        beam.append(state)
                    else:
                        postponed.append(state)
                    if len(beam) >= width:
                        break
                if len(beam) < width:
                    beam.extend(postponed[:width - len(beam)])
            for state in beam:
                if time.perf_counter() >= stop:
                    break
                rollout(state, stop)
            return local_best

        def recombine(source, params, stall):
            # External pieces are fixed and the candidate pool contains the original layout, so candidate optimisation never loses the original feasible solution.
            begun = time.perf_counter()
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
            # Add spatial fragments of elite layouts directly as candidates, rather than only collecting the current greedy repair.
            elite_limit = min(candidate_limit, len(catalogue)
                              + max(1, int((candidate_limit - len(catalogue)) * 0.35)))
            elite_stop = min(stop, begun + (stop - begun) * 0.12)
            donors = pool[:]
            rng.shuffle(donors)
            for _, donor, _ in donors:
                if len(catalogue) >= elite_limit or time.perf_counter() >= elite_stop:
                    break
                flips = [(0, 0), (1, 0), (0, 1), (1, 1)]
                rng.shuffle(flips)
                for fx, fy in flips:
                    if len(catalogue) >= elite_limit or time.perf_counter() >= elite_stop:
                        break
                    for i, x, y, w, h, r in donor:
                        if len(catalogue) >= elite_limit or time.perf_counter() >= elite_stop:
                            break
                        if fixed_counts[i] >= upper[i]:
                            continue
                        p = (i, max(0.0, L - x - w) if fx else x,
                             max(0.0, W - y - h) if fy else y, w, h, r)
                        if p in fixed or not compatible(p, kept):
                            continue
                        if all(p[1] + w <= q[1] + eps or q[1] + q[3] <= p[1] + eps
                               or p[2] + h <= q[2] + eps or q[2] + q[4] <= p[2] + eps
                               for q in kept):
                            collect([p])
            generate_stop = begun + (stop - begun) * 0.42
            trial_params = params
            corner = 0
            while time.perf_counter() < generate_stop and len(catalogue) < candidate_limit:
                trial = refill(kept, trial_params, until=generate_stop, corner=corner)
                if trial is not None:
                    score = remember(trial)
                    if score is not None:
                        collect(trial)
                        if score - fixed_value > local_value:
                            local_best, local_value = trial, score - fixed_value
                trial_params = parameters()
                # Local recombination needs different density preferences and edge directions, not just a repeat of the same greedy run.
                if rng.random() < 0.5:
                    trial_params = (rng.uniform(0.0, 1.4), trial_params[1],
                                    trial_params[2], trial_params[3])
                corner = rng.randrange(4)
            candidates = sorted(catalogue.values(), key=lambda p: (-value[p[0]], p))
            m = len(candidates)
            if m <= amount or time.perf_counter() >= stop:
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
                if time.perf_counter() >= stop:
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
                    if k % 256 == 0 and time.perf_counter() >= stop:
                        return local_best
            for j in range(len(tids)):
                if cap[j] <= 0 or (type_value[j] <= 0 and need[j] == 0):
                    full_mask &= ~group[j]
            free_area = max(0.0, board - math.fsum(area[p[0]] for p in kept))
            slots = K - len(kept)
            # The state is only the local candidate set; recursion is avoided so that larger candidate pools do not trigger recursion-depth errors.
            stack = [(full_mask, 0, 0.0, slots, cap, need, free_area)]
            while stack and time.perf_counter() < stop:
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
        weights = [1.0] * 8
        weights[6] = 1.0 + 0.15 * effort
        weights[7] = 1.0 + 0.3 * effort
        joint_spent = 0.0
        geometric_spent = 0.0
        while time.perf_counter() < end:
            now = time.perf_counter()
            stall = min(1.0, (now - last_gain) / max(0.001, budget * 0.3))
            params = parameters()
            pick = rng.random() * sum(weights)
            op = 0
            while op < len(weights) - 1 and pick > weights[op]:
                pick -= weights[op]
                op += 1
            joint_share = effort / (6.0 + 2.0 * effort) + 0.10 * stall
            if op == 6 and joint_spent >= joint_share * (now - start):
                op = rng.randrange(6)
            geometric_share = effort / (7.0 + 3.0 * effort) + 0.08 * stall
            if op == 7 and geometric_spent >= geometric_share * (now - start):
                op = rng.randrange(6)
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
                kept = base if op >= 6 else neighborhood(base, op, size, params, stall)
            if kept is None or time.perf_counter() >= end:
                continue
            old_best = best_value
            if op == 6 and best_value != -math.inf:
                joint_start = time.perf_counter()
                candidate = recombine(kept, params, stall)
                joint_spent += time.perf_counter() - joint_start
            elif op == 7 and best_value != -math.inf:
                geometric_start = time.perf_counter()
                candidate = geometric_search(kept, params, stall)
                geometric_spent += time.perf_counter() - geometric_start
            else:
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
