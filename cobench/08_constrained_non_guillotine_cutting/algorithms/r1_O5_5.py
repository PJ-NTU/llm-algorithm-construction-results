"""Adaptive spatial destroy-repair packing
Uses multi-start maximal-empty-rectangle construction combined with region re-sequencing, blocker removal and insertion, axial compaction and elite layout reuse, keeping the highest total-value feasible solution within the wall-clock limit.
"""

import heapq
import math
import random
import time


def solve(instance: dict, time_limit_s: float) -> dict:
    start = time.perf_counter()
    try:
        K = max(0, int(instance.get("max_slots", 0)))
    except Exception:
        K = 0
    blank = {"is_used": 0, "piece_type_id": -1,
             "x": 0.0, "y": 0.0, "is_rotated": 0}
    best_result = {"placements": [dict(blank) for _ in range(K)]}

    try:
        budget = max(0.0, float(time_limit_s))
        deadline = start + budget * 0.975 - min(0.02, budget * 0.01)
        clock = time.perf_counter
        L0 = float(instance["stock_length"])
        W0 = float(instance["stock_width"])
        N = int(instance["num_piece_types"])
        scale = max(L0, W0, 1.0)
        L, W = L0 / scale, W0 / scale
        lengths = [float(z) / scale for z in instance["piece_lengths"]]
        widths = [float(z) / scale for z in instance["piece_widths"]]
        values = [float(z) for z in instance["piece_values"]]
        minimum = [int(z) for z in instance["min_usage"]]
        maximum = [min(K, int(z)) for z in instance["max_usage"]]
        M = float(instance["big_m"]) / scale
        if N < 0 or L <= 0.0 or W <= 0.0 or M < 0.0:
            return best_result
        if not all(len(a) == N for a in
                   (lengths, widths, values, minimum, maximum)):
            return best_result
        if any(minimum[i] < 0 or minimum[i] > maximum[i] or
               lengths[i] < 0.0 or widths[i] < 0.0 for i in range(N)):
            return best_result
        if not all(math.isfinite(z) for a in (lengths, widths, values)
                   for z in a):
            return best_result
        required_size = sum(minimum)
        if required_size > K or K == 0 or N == 0:
            return best_result

        eps = 1e-12
        board_area = L * W
        areas = [lengths[i] * widths[i] for i in range(N)]
        log_areas = [math.log(max(a, 1e-300)) for a in areas]
        log_values = [math.log(v) if v > 0.0 else -math.inf for v in values]
        orientations = []
        for i in range(N):
            a, b = lengths[i], widths[i]
            options = []
            if a <= L + eps and b <= W + eps:
                options.append((a, b, 0))
            if a != b and b <= L + eps and a <= W + eps:
                options.append((b, a, 1))
            orientations.append(options)
            if not options:
                if minimum[i]:
                    return best_result
                maximum[i] = 0
        if sum(areas[i] * minimum[i] for i in range(N)) > board_area + eps:
            return best_result

        tight_m = M + eps < max(L, W)
        effort = math.log1p(budget)
        archive_size = max(2, int(2.0 + 2.0 * math.sqrt(effort)))
        max_backtrack_depth = max(1, int(3.0 * math.sqrt(budget + 1.0)))
        seed = K * 1009 + N * 9176 + int(L0 * 17.0 + W0 * 31.0)
        for i in range(N):
            seed += (i + 11) * int(values[i] * 101.0 +
                                  lengths[i] * 100003.0 + widths[i] * 31337.0)
        rng = random.Random(seed & 0xFFFFFFFF)
        archive = []
        best_pack = [] if required_size == 0 else None
        best_value = 0.0 if best_pack is not None else -math.inf
        last_gain = start
        alphas = (0.0, 1.0, 0.45, 0.72, 0.2, 1.25)

        def counts_of(packed):
            counts = [0] * N
            for p in packed:
                counts[p[0]] += 1
            return counts

        def complete(packed):
            rows = [{"is_used": 1, "piece_type_id": int(p[0]),
                     "x": float(max(0.0, p[1] * scale)),
                     "y": float(max(0.0, p[2] * scale)),
                     "is_rotated": int(p[3])} for p in packed]
            rows.extend(dict(blank) for _ in range(K - len(rows)))
            return {"placements": rows}

        def valid(packed, counts):
            if len(packed) > K:
                return False
            if any(counts[i] < minimum[i] or counts[i] > maximum[i]
                   for i in range(N)):
                return False
            checks = 0
            for j, p in enumerate(packed):
                _, x, y, _, w, h = p
                if x < -eps or y < -eps or x + w > L + eps or y + h > W + eps:
                    return False
                # Unused slots have zero coordinates; this also satisfies the big-M relaxation of the original problem.
                if tight_m and len(packed) < K and (x + w > 2 * M + eps or
                                                   y + h > 2 * M + eps):
                    return False
                for q in packed[:j]:
                    checks += 1
                    if checks & 255 == 0 and clock() >= deadline:
                        return False
                    if not (x + w <= q[1] + eps or q[1] + q[4] <= x + eps or
                            y + h <= q[2] + eps or q[2] + q[5] <= y + eps):
                        return False
                    if tight_m and (x + w - q[1] > M + eps or
                                    q[1] + q[4] - x > M + eps or
                                    y + h - q[2] > M + eps or
                                    q[2] + q[5] - y > M + eps):
                        return False
            return True

        def remember(packed, alpha, rule):
            nonlocal best_pack, best_value, best_result, last_gain
            if packed is None or clock() >= deadline:
                return
            counts = counts_of(packed)
            if any(counts[i] < minimum[i] or counts[i] > maximum[i]
                   for i in range(N)):
                return
            value = sum(values[p[0]] for p in packed)
            improved = best_pack is None or value > best_value + 1e-9
            if (improved or tight_m) and not valid(packed, counts):
                return
            if improved:
                best_pack = list(packed)
                best_value = value
                best_result = complete(packed)
                last_gain = clock()
            if not packed or any(row[1] == packed for row in archive):
                return
            entry = (value, list(packed), alpha, rule)
            if len(archive) < archive_size:
                archive.append(entry)
            else:
                # Different layouts with the same value can still replace lower-ranked members, preventing the spatial search from freezing.
                replace = [j for j in range(1, len(archive))
                           if value >= archive[j][0] - 1e-9]
                if not replace:
                    return
                archive[rng.choice(replace)] = entry
            archive.sort(key=lambda row: row[0], reverse=True)

        def split_free(free, p):
            x, y, right, top = p[1], p[2], p[1] + p[4], p[2] + p[5]
            old, new = [], []
            for j, (a, b, c, d) in enumerate(free):
                if j & 63 == 0 and clock() >= deadline:
                    return None
                if right <= a + eps or x >= c - eps or top <= b + eps or y >= d - eps:
                    old.append((a, b, c, d))
                    continue
                if x > a + eps:
                    new.append((a, b, x, d))
                if right < c - eps:
                    new.append((right, b, c, d))
                if y > b + eps:
                    new.append((a, b, c, y))
                if top < d - eps:
                    new.append((a, top, c, d))
            # Existing non-intersected rectangles already do not contain each other; only newly generated rectangles need checking.
            new.sort(key=lambda r: (r[2] - r[0]) * (r[3] - r[1]), reverse=True)
            checks = 0
            for a, b, c, d in new:
                if c - a <= eps or d - b <= eps:
                    continue
                contained = False
                for oa, ob, oc, od in old:
                    checks += 1
                    if checks & 255 == 0 and clock() >= deadline:
                        return None
                    if a >= oa - eps and b >= ob - eps and c <= oc + eps and d <= od + eps:
                        contained = True
                        break
                if not contained:
                    old.append((a, b, c, d))
            return old

        def free_of(packed):
            free = [(0.0, 0.0, L, W)]
            for p in packed:
                free = split_free(free, p)
                if free is None:
                    return None
            return free

        def allowed_m(p, packed):
            if not tight_m:
                return True
            x, y, w, h = p[1], p[2], p[4], p[5]
            for q in packed:
                if (x + w - q[1] > M + eps or q[1] + q[4] - x > M + eps or
                        y + h - q[2] > M + eps or q[2] + q[5] - y > M + eps):
                    return False
            return True

        def position(tid, free, packed, rule, noise):
            chosen, best_key = None, None
            max_right = max((p[1] + p[4] for p in packed), default=0.0)
            max_top = max((p[2] + p[5] for p in packed), default=0.0)
            checks = 0
            for a, b, c, d in free:
                fw, fh = c - a, d - b
                for w, h, rot in orientations[tid]:
                    if w > fw + eps or h > fh + eps:
                        continue
                    dw, dh = max(0.0, fw - w), max(0.0, fh - h)
                    short, long_side = min(dw, dh), max(dw, dh)
                    waste = fw * fh - w * h
                    anchors = [(a, b)]
                    if rule == 3 or noise > 0.0:
                        if dw > eps:
                            anchors.append((c - w, b))
                        if dh > eps:
                            anchors.append((a, d - h))
                        if dw > eps and dh > eps:
                            anchors.append((c - w, d - h))
                    for x, y in anchors:
                        checks += 1
                        if checks & 63 == 0 and clock() >= deadline:
                            return chosen
                        p = (tid, x, y, rot, w, h)
                        if not allowed_m(p, packed):
                            continue
                        jitter = noise * rng.random() if noise else 0.0
                        if rule == 0:
                            key = (short + jitter, long_side, y, x)
                        elif rule == 1:
                            key = (waste + jitter * board_area, short, long_side, y, x)
                        elif rule == 2:
                            key = (y + h + jitter, x + w, short, long_side)
                        elif rule == 3:
                            contact = 0.0
                            if abs(x) <= eps or abs(x + w - L) <= eps:
                                contact += h
                            if abs(y) <= eps or abs(y + h - W) <= eps:
                                contact += w
                            for q in packed:
                                if abs(x + w - q[1]) <= eps or abs(q[1] + q[4] - x) <= eps:
                                    contact += max(0.0, min(y + h, q[2] + q[5]) - max(y, q[2]))
                                if abs(y + h - q[2]) <= eps or abs(q[2] + q[5] - y) <= eps:
                                    contact += max(0.0, min(x + w, q[1] + q[4]) - max(x, q[1]))
                            key = (-contact + jitter, waste, y + h, x + w)
                        else:
                            nr, nt = max(max_right, x + w), max(max_top, y + h)
                            key = (nr * nt + jitter * board_area, max(nr, nt), waste, y, x)
                        if best_key is None or key < best_key:
                            best_key, chosen = key, p
            return chosen

        def ordered(counts, style, spread):
            rows = []
            for tid, number in enumerate(counts):
                a, b = lengths[tid], widths[tid]
                if style == 0:
                    base = areas[tid]
                elif style == 1:
                    base = max(a, b)
                elif style == 2:
                    base = min(a, b)
                else:
                    base = a + b
                for _ in range(number):
                    weight = base * math.exp(rng.uniform(-spread, spread)) if spread else base
                    rows.append((-weight, -areas[tid], -values[tid], tid))
            rows.sort()
            return [row[3] for row in rows]

        def priority(tid, alpha, spread):
            return log_values[tid] - alpha * log_areas[tid] + rng.uniform(-spread, spread)

        def build(kept, alpha, rule, style, spread, noise, target=None, free=None):
            packed = list(kept)
            counts = counts_of(packed)
            if free is None:
                free = free_of(packed)
            if free is None:
                return None
            target = minimum if target is None else target
            missing = [max(0, target[i] - counts[i]) for i in range(N)]
            for tid in ordered(missing, style, spread):
                if clock() >= deadline or len(packed) >= K:
                    break
                p = position(tid, free, packed, rule, noise)
                if p is None:
                    if counts[tid] < minimum[tid]:
                        return None
                    continue
                packed.append(p)
                counts[tid] += 1
                free = split_free(free, p)
                if free is None:
                    break
            if any(counts[i] < minimum[i] for i in range(N)):
                return None
            # Before backfilling, save the prefix that already satisfies all lower bounds, so that a timeout in the last construction does not lose the solution.
            remember(packed, alpha, rule)
            if free is None:
                return packed
            heap = []
            for tid in range(N):
                if values[tid] > 0.0 and counts[tid] < maximum[tid]:
                    heap.append((-priority(tid, alpha, spread), rng.random(), tid))
            heapq.heapify(heap)
            while heap and len(packed) < K and clock() < deadline:
                _, _, tid = heapq.heappop(heap)
                p = position(tid, free, packed, rule, noise)
                if p is None:
                    continue
                packed.append(p)
                counts[tid] += 1
                free = split_free(free, p)
                if free is None:
                    break
                if counts[tid] < maximum[tid]:
                    heapq.heappush(heap, (-priority(tid, alpha, spread), rng.random(), tid))
            return packed

        def selected_counts(alpha, spread):
            counts = list(minimum)
            room = K - required_size
            reserve = rng.uniform(0.88, 1.0)
            area_left = max(0.0, board_area * reserve -
                            sum(areas[i] * counts[i] for i in range(N)))
            heap = [(-priority(i, alpha, spread), rng.random(), i)
                    for i in range(N) if values[i] > 0.0 and counts[i] < maximum[i]]
            heapq.heapify(heap)
            while room and heap and clock() < deadline:
                _, _, i = heapq.heappop(heap)
                if areas[i] > area_left + eps:
                    continue
                counts[i] += 1
                room -= 1
                area_left -= areas[i]
                if counts[i] < maximum[i]:
                    heapq.heappush(heap, (-priority(i, alpha, spread), rng.random(), i))
            return counts

        def compact(packed):
            work = list(packed)
            axes = [0, 1]
            rng.shuffle(axes)
            for axis in axes:
                reverse = rng.random() < 0.5
                co, dim = (1, 4) if axis == 0 else (2, 5)
                other, other_dim = (2, 5) if axis == 0 else (1, 4)
                limit = L if axis == 0 else W
                indices = sorted(range(len(work)), key=lambda j: work[j][co], reverse=reverse)
                for j in indices:
                    if clock() >= deadline:
                        return None
                    p = work[j]
                    new_coordinate = limit - p[dim] if reverse else 0.0
                    for k, q in enumerate(work):
                        if k == j or p[other] + p[other_dim] <= q[other] + eps or \
                                q[other] + q[other_dim] <= p[other] + eps:
                            continue
                        if reverse and q[co] >= p[co] + p[dim] - eps:
                            new_coordinate = min(new_coordinate, q[co] - p[dim])
                        elif not reverse and q[co] + q[dim] <= p[co] + eps:
                            new_coordinate = max(new_coordinate, q[co] + q[dim])
                    row = list(p)
                    row[co] = max(0.0, new_coordinate)
                    work[j] = tuple(row)
            return work

        def ruin(parent, amount):
            n = len(parent)
            mode = rng.randrange(4)
            if mode == 0:
                p = rng.choice(parent)
                cx = p[1] + p[4] * rng.random()
                cy = p[2] + p[5] * rng.random()
                def proximity(j):
                    q = parent[j]
                    dx = max(q[1] - cx, cx - q[1] - q[4], 0.0) / L
                    dy = max(q[2] - cy, cy - q[2] - q[5], 0.0) / W
                    return dx * dx + dy * dy + rng.random() * 0.0001
                order = sorted(range(n), key=proximity)
            elif mode == 1:
                axis = rng.choice((1, 2))
                order = sorted(range(n), key=lambda j: parent[j][axis], reverse=rng.random() < 0.5)
            elif mode == 2:
                alpha = rng.random()
                order = sorted(range(n), key=lambda j:
                               values[parent[j][0]] / max(areas[parent[j][0]], eps) ** alpha *
                               rng.uniform(0.6, 1.4))
            else:
                order = rng.sample(range(n), n)
            removed = set(order[:amount])
            return [p for j, p in enumerate(parent) if j not in removed]

        def eject(parent, alpha, amount):
            now = clock()
            stop = min(deadline, now + budget / (40.0 + 4.0 * math.sqrt(len(parent) + N)))
            counts = counts_of(parent)
            candidates = [i for i in range(N) if values[i] > 0.0 and
                          maximum[i] > counts[i] and orientations[i]]
            if not candidates:
                return None
            tid = max(candidates, key=lambda i: priority(i, alpha, 0.8))
            best_move, best_merit = None, -math.inf
            density = sum(values[p[0]] for p in parent) / max(sum(areas[p[0]] for p in parent), eps)
            choices = list(orientations[tid])
            rng.shuffle(choices)
            for w, h, rot in choices:
                xs, ys = {0.0, max(0.0, L - w)}, {0.0, max(0.0, W - h)}
                points = [(0.0, 0.0), (L - w, 0.0), (0.0, W - h), (L - w, W - h)]
                for p in parent:
                    px, py, right, top = p[1], p[2], p[1] + p[4], p[2] + p[5]
                    xs.update((px, right, px - w, right - w))
                    ys.update((py, top, py - h, top - h))
                    points.extend(((px, py), (right - w, py), (px, top - h),
                                   (right - w, top - h), (right, py), (px, top)))
                rng.shuffle(points)
                xlist = [x for x in xs if -eps <= x <= L - w + eps]
                ylist = [y for y in ys if -eps <= y <= W - h + eps]
                rng.shuffle(xlist)
                rng.shuffle(ylist)

                def locations():
                    yield from points
                    for y in ylist:
                        for x in xlist:
                            yield x, y

                for x, y in locations():
                    if clock() >= stop:
                        break
                    if x < -eps or y < -eps or x + w > L + eps or y + h > W + eps:
                        continue
                    victims = []
                    loss, freed = 0.0, 0.0
                    for j, p in enumerate(parent):
                        if not (x + w <= p[1] + eps or p[1] + p[4] <= x + eps or
                                y + h <= p[2] + eps or p[2] + p[5] <= y + eps):
                            victims.append(j)
                            loss += values[p[0]]
                            freed += areas[p[0]]
                    if not victims and len(parent) >= K:
                        removable = [j for j, p in enumerate(parent)
                                     if counts[p[0]] > minimum[p[0]]]
                        if not removable:
                            continue
                        j = min(removable, key=lambda j: values[parent[j][0]])
                        victims.append(j)
                        loss += values[parent[j][0]]
                        freed += areas[parent[j][0]]
                    if len(victims) > amount:
                        continue
                    left = list(counts)
                    for j in victims:
                        left[parent[j][0]] -= 1
                    left[tid] += 1
                    deficit = sum(max(0, minimum[i] - left[i]) for i in range(N))
                    if len(parent) - len(victims) + 1 + deficit > K:
                        continue
                    merit = values[tid] - loss + 0.45 * density * max(0.0, freed - areas[tid])
                    merit -= 0.02 * abs(density) * board_area * deficit / max(1, len(parent))
                    if merit > best_merit:
                        best_merit = merit
                        best_move = (set(victims), (tid, max(0.0, x), max(0.0, y), rot, w, h))
                if clock() >= stop:
                    break
            if best_move is None:
                return None
            removed, p = best_move
            kept = [q for j, q in enumerate(parent) if j not in removed]
            if not allowed_m(p, kept):
                return None
            kept.append(p)
            return kept

        def mandatory_backtrack(style):
            stop = min(deadline, clock() + budget / (8.0 + math.sqrt(required_size + 1.0)))
            sequence = ordered(minimum, style, 0.25)
            packed = []

            def visit(depth, free):
                if clock() >= stop:
                    return None
                if depth == len(sequence):
                    return list(packed)
                tid = sequence[depth]
                choices = {}
                for a, b, c, d in free:
                    for w, h, rot in orientations[tid]:
                        if w > c - a + eps or h > d - b + eps:
                            continue
                        for x, y in ((a, b), (c - w, b), (a, d - h), (c - w, d - h)):
                            p = (tid, x, y, rot, w, h)
                            if allowed_m(p, packed):
                                key = (min(c - a - w, d - b - h),
                                       (c - a) * (d - b) - w * h, y, x)
                                if p not in choices or key < choices[p]:
                                    choices[p] = key
                for p in sorted(choices, key=choices.get):
                    if clock() >= stop:
                        return None
                    next_free = split_free(free, p)
                    if next_free is None:
                        return None
                    packed.append(p)
                    result = visit(depth + 1, next_free)
                    if result is not None:
                        return result
                    packed.pop()
                return None

            return visit(0, [(0.0, 0.0, L, W)])

        # The first round secures the lower bounds first, then backfills by value; the total length of the later search is determined only by the wall clock.
        candidate = build([], 0.0, 0, 0, 0.0, 0.0)
        remember(candidate, 0.0, 0)
        iteration = 0
        while clock() < deadline:
            now = clock()
            alpha = alphas[iteration % len(alphas)]
            rule = iteration % 5
            style = iteration % 4
            spread = rng.uniform(0.06, 0.38)
            noise = 0.0 if iteration % 5 == 0 else rng.uniform(0.001, 0.025)
            candidate = None
            if best_pack is None:
                if required_size <= max_backtrack_depth and now - start > budget * 0.12 and rng.random() < 0.3:
                    candidate = mandatory_backtrack(style)
                    remember(candidate, alpha, rule)
                    if candidate is not None:
                        candidate = build(candidate, alpha, rule, style, spread, noise)
                else:
                    candidate = build([], alpha, rule, style, spread, noise)
            elif now < start + budget * 0.20 or not best_pack or rng.random() < 0.26:
                candidate = build([], alpha, rule, style, spread, noise)
            else:
                if archive and rng.random() < 0.35:
                    _, parent, old_alpha, old_rule = rng.choice(archive)
                else:
                    parent = best_pack
                    old_alpha, old_rule = (archive[0][2], archive[0][3]) if archive else (alpha, rule)
                if rng.random() < 0.6:
                    alpha = max(0.0, min(1.4, old_alpha + rng.uniform(-0.2, 0.2)))
                if rng.random() < 0.5:
                    rule = old_rule
                stagnant = min(1.0, (now - last_gain) / max(budget * 0.25, 1e-9))
                fraction = min(0.75, 0.09 + 0.055 * effort + 0.30 * stagnant)
                depth = max(1, min(len(parent), int(math.ceil(len(parent) * fraction))))
                amount = rng.randint(1, depth)
                branch = rng.random()
                if branch < 0.68:
                    kept = ruin(parent, amount)
                    if kept and not tight_m and rng.random() < 0.25:
                        kept = compact(kept)
                    if kept is not None:
                        candidate = build(kept, alpha, rule, style, spread, noise)
                elif branch < 0.85:
                    if rng.random() < 0.5:
                        target = counts_of(parent)
                    else:
                        target = selected_counts(alpha, spread)
                    candidate = build([], alpha, rule, style, spread, noise, target=target)
                else:
                    kept = eject(parent, alpha, depth)
                    if kept is not None:
                        candidate = build(kept, alpha, rule, style, spread, noise)
            remember(candidate, alpha, rule)
            iteration += 1
        return best_result
    except Exception:
        return best_result
