"""Fixed-bin-count multi-item tabu bin packing
Builds a feasible solution by decreasing-order packing, then repairs a layout with one bin fewer through one- and two-item swaps, overload memory penalties and time-limited restarts.
"""

import bisect
import math
import random
import time


def solve(instance: dict, time_limit_s: float) -> dict:
    clock = time.perf_counter
    started = clock()
    safe = {"bin_activation": [], "item_assignments": []}
    try:
        n = max(0, int(instance["n"]))
        safe = {"bin_activation": [1] * n, "item_assignments": list(range(n))}
        if n == 0:
            return safe
        capacity = int(instance["capacity"])
        weights = [int(w) for w in instance["weights"]]
        if len(weights) != n or capacity < 0 or any(w < 0 or w > capacity for w in weights):
            return safe
        items = [i for i, w in enumerate(weights) if w > 0]
        if not items:
            return {"bin_activation": [0] * n, "item_assignments": [0] * n}
        budget = float(time_limit_s)
        if not math.isfinite(budget) or budget <= 0:
            return safe
        deadline = started + budget * 0.955
        strength = math.log1p(budget)
        size = len(items)
        rng = random.Random(811 * n + 104729 * capacity + sum(weights))
        order = sorted(items, key=weights.__getitem__, reverse=True)

        def encode(groups):
            assignment = [0] * n
            count = 0
            for group in groups:
                if group:
                    for i in group:
                        assignment[i] = count
                    count += 1
            return {"bin_activation": [1] * count + [0] * (n - count),
                    "item_assignments": assignment}

        def construct(sequence, cutoff, first_fit=False):
            groups = []
            spaces = []
            if first_fit:
                leaves = 1 << max(0, (len(sequence) - 1).bit_length())
                tree = [0] * (2 * leaves)
                loads = []
            for p, i in enumerate(sequence):
                if p % 32 == 0 and clock() >= cutoff:
                    groups.extend([j] for j in sequence[p:])
                    break
                w = weights[i]
                if first_fit:
                    if tree[1] < w:
                        b = len(groups)
                        groups.append([])
                        loads.append(0)
                    else:
                        node = 1
                        while node < leaves:
                            node *= 2
                            if tree[node] < w:
                                node += 1
                        b = node - leaves
                    groups[b].append(i)
                    loads[b] += w
                    node = leaves + b
                    tree[node] = capacity - loads[b]
                    while node > 1:
                        node //= 2
                        tree[node] = max(tree[node * 2], tree[node * 2 + 1])
                else:
                    pspace = bisect.bisect_left(spaces, (w, -1))
                    if pspace == len(spaces):
                        b = len(groups)
                        groups.append([i])
                        bisect.insort(spaces, (capacity - w, b))
                    else:
                        room, b = spaces.pop(pspace)
                        groups[b].append(i)
                        bisect.insort(spaces, (room - w, b))
            return groups

        best = construct(order, deadline)
        safe = encode(best)
        if clock() >= deadline:
            return safe
        total = sum(weights)
        lower = max((total + capacity - 1) // capacity,
                    sum(2 * weights[i] > capacity for i in items))
        if len(best) == lower:
            return safe

        # With at most two items per bin, two-pointer matching of the largest item with the smallest compatible item is exact.
        if size < 3 or sum(weights[i] for i in order[-3:]) > capacity:
            lo, hi = 0, size - 1
            groups = []
            while lo <= hi:
                group = [order[lo]]
                if lo < hi and weights[order[lo]] + weights[order[hi]] <= capacity:
                    group.append(order[hi])
                    hi -= 1
                groups.append(group)
                lo += 1
            return encode(groups)

        # Threshold-weight lower bound: large items each take a bin, and medium items that cannot enter a large item's bin are counted separately by capacity.
        ascending = sorted(weights[i] for i in items)
        prefix = [0]
        for w in ascending:
            prefix.append(prefix[-1] + w)
        half = bisect.bisect_right(ascending, capacity // 2)
        large = size - half
        previous = -1
        for p, a in enumerate(ascending):
            if p % 64 == 0 and clock() >= deadline:
                return safe
            if a == previous:
                continue
            previous = a
            per_bin = capacity // a
            lower = max(lower, (size - p + per_bin - 1) // per_bin)
            if p < half:
                right = bisect.bisect_right(ascending, capacity - a)
                free = (right - half) * capacity - (prefix[right] - prefix[half])
                uncovered = max(0, prefix[half] - prefix[p] - free)
                lower = max(lower, large + (uncovered + capacity - 1) // capacity)
        if len(best) <= lower or clock() >= deadline:
            return safe
        alternative = construct(order, deadline, True)
        if len(alternative) < len(best):
            best = alternative
            safe = encode(best)
        if len(best) <= lower or clock() >= deadline:
            return safe

        # The best feasible solution is saved throughout; the fixed-bin-count state below is only an internal search workspace.
        target = len(best) - 1
        checkpoint = None
        checkpoint_key = None
        round_id = 0
        pulse = budget / (5.0 + math.log1p(size))
        avg_size = size / max(1, target)
        pair_limit = max(1, int((1.0 + strength) ** 2 * math.sqrt(avg_size)))
        source_limit = max(2, int((2.0 + strength) * math.sqrt(size)))
        tenure_base = max(1, int((1.0 + strength) * math.sqrt(avg_size)))

        while clock() < deadline and len(best) > lower:
            k = len(best) - 1
            if target != k:
                target = k
                checkpoint = None
                checkpoint_key = None
                round_id = 0
                avg_size = size / max(1, k)
                pair_limit = max(1, int((1.0 + strength) ** 2 * math.sqrt(avg_size)))
                tenure_base = max(1, int((1.0 + strength) * math.sqrt(avg_size)))
            if k <= 0:
                break
            now = clock()
            cutoff = min(deadline, now + pulse * (1.0 + math.log1p(round_id) / (1.0 + strength)))
            from_checkpoint = checkpoint is not None and rng.random() < 0.7
            if from_checkpoint:
                groups = [g[:] for g in checkpoint]
            else:
                base_loads = [sum(weights[i] for i in g) for g in best]
                ranked = sorted(range(len(best)), key=base_loads.__getitem__)
                width = min(len(ranked), max(1, int(1 + strength * math.sqrt(round_id))))
                removed = ranked[0] if round_id == 0 else ranked[rng.randrange(width)]
                groups = [g[:] for b, g in enumerate(best) if b != removed]
                loose = best[removed][:]
                loads = [sum(weights[i] for i in g) for g in groups]
                loose.sort(key=weights.__getitem__, reverse=True)
                for i in loose:
                    if clock() >= cutoff:
                        break
                    w = weights[i]
                    b = min(range(k), key=lambda b: (
                        max(0, loads[b] + w - capacity) - max(0, loads[b] - capacity),
                        abs(capacity - loads[b] - w), rng.random()))
                    groups[b].append(i)
                    loads[b] += w
                else:
                    loose = []
                if loose:
                    # If initialisation has not finished, keep the previous complete feasible solution directly.
                    break

            loads = [sum(weights[i] for i in g) for g in groups]
            if round_id:
                # Break up a small group of bins around an overloaded bin and reassign them within the same number of bins.
                over = [b for b in range(k) if loads[b] > capacity]
                extent = 1.0 - math.exp(-round_id / (1.0 + strength))
                width = min(k, max(len(over), int(2 + strength * math.sqrt(k) * extent)))
                selected = set(over)
                candidates = [b for b in range(k) if b not in selected]
                rng.shuffle(candidates)
                selected.update(candidates[:max(0, width - len(selected))])
                selected = list(selected)
                loose = [i for b in selected for i in groups[b]]
                if from_checkpoint:
                    noise = capacity * (0.04 + 0.3 * extent) / (1.0 + strength)
                else:
                    noise = capacity * rng.random()
                loose.sort(key=lambda i: weights[i] + noise * rng.random(), reverse=True)
                for b in selected:
                    groups[b] = []
                    loads[b] = 0
                for p, i in enumerate(loose):
                    if p % 16 == 0 and clock() >= cutoff:
                        break
                    w = weights[i]
                    b = min(selected, key=lambda b: (
                        max(0, loads[b] + w - capacity) - max(0, loads[b] - capacity),
                        abs(capacity - loads[b] - w), rng.random()))
                    groups[b].append(i)
                    loads[b] += w
                else:
                    loose = []
                if loose:
                    round_id += 1
                    continue

            location = [-1] * n
            position = [-1] * n
            for b, group in enumerate(groups):
                for p, i in enumerate(group):
                    location[i] = b
                    position[i] = p
            overflow = sum(max(0, load - capacity) for load in loads)
            overloaded = {b for b in range(k) if loads[b] > capacity}
            tabu_bin = [-1] * n
            tabu_until = [0] * n
            heat = [0] * k
            catalogs = [None] * k
            step = 0
            run_best = overflow
            progress = clock()

            def save_checkpoint():
                nonlocal checkpoint, checkpoint_key
                key = (overflow, len(overloaded),
                       -sum(max(0, capacity - load) ** 2 for load in loads))
                if checkpoint_key is None or key < checkpoint_key:
                    checkpoint_key = key
                    checkpoint = [g[:] for g in groups]

            def packets(b):
                cached = catalogs[b]
                if cached is not None:
                    return cached
                group = groups[b]
                count = len(group)
                packets_list = [(0, -1, -1)]
                packets_list.extend((weights[i], i, -1) for i in group)
                combinations = count * (count - 1) // 2
                if combinations <= pair_limit:
                    for p in range(count):
                        i = group[p]
                        for j in group[p + 1:]:
                            packets_list.append((weights[i] + weights[j], i, j))
                else:
                    seen = set()
                    while len(seen) < pair_limit and clock() < cutoff:
                        p = rng.randrange(count)
                        q = rng.randrange(count - 1)
                        if q >= p:
                            q += 1
                        i, j = group[p], group[q]
                        if i > j:
                            i, j = j, i
                        if (i, j) not in seen:
                            seen.add((i, j))
                            packets_list.append((weights[i] + weights[j], i, j))
                packets_list.sort()
                cached = (packets_list, [packet[0] for packet in packets_list])
                catalogs[b] = cached
                return cached

            def relocate(i, b):
                if i < 0:
                    return
                a = location[i]
                p = position[i]
                last = groups[a][-1]
                groups[a][p] = last
                position[last] = p
                groups[a].pop()
                position[i] = len(groups[b])
                groups[b].append(i)
                location[i] = b
                loads[a] -= weights[i]
                loads[b] += weights[i]
                tabu_bin[i] = a
                tabu_until[i] = step + tenure_base + rng.randrange(1 + tenure_base)

            save_checkpoint()
            while clock() < cutoff:
                if overflow == 0:
                    candidate = [g[:] for g in groups if g]
                    if len(candidate) < len(best):
                        best = candidate
                        safe = encode(best)
                    break
                step += 1
                best_rank = None
                chosen = None
                ties = 0
                sources = list(overloaded)
                rng.shuffle(sources)
                for a in sources:
                    if clock() >= cutoff:
                        break
                    load_a = loads[a]
                    excess_a = load_a - capacity
                    source_packets = packets(a)[0][1:]
                    if len(source_packets) > source_limit:
                        source_packets = rng.sample(source_packets, source_limit)
                    for packet_a in source_packets:
                        if clock() >= cutoff:
                            break
                        mass_a, i, j = packet_a
                        for b in range(k):
                            if b == a:
                                continue
                            if b % 32 == 0 and clock() >= cutoff:
                                break
                            load_b = loads[b]
                            excess_b = max(0, load_b - capacity)
                            free_b = capacity - load_b
                            target_packets, masses = packets(b)
                            # Both ends of the interval of optimal continuous swap amounts, plus candidates inside it.
                            left = bisect.bisect_left(masses, mass_a - max(excess_a, free_b))
                            right = bisect.bisect_left(masses, mass_a - min(excess_a, free_b))
                            indices = {0, left - 1, left, right - 1, right}
                            if right > left + 1:
                                indices.add(rng.randrange(left, right))
                            old_count = 1 + int(excess_b > 0)
                            old_slack_square = max(0, free_b) ** 2
                            for p in indices:
                                if p < 0 or p >= len(target_packets):
                                    continue
                                mass_b, u, v = target_packets[p]
                                delta_mass = mass_a - mass_b
                                if delta_mass == 0:
                                    continue
                                new_a = max(0, excess_a - delta_mass)
                                new_b = max(0, load_b + delta_mass - capacity)
                                change = new_a + new_b - excess_a - excess_b
                                trial_overflow = overflow + change
                                forbidden = ((tabu_until[i] > step and tabu_bin[i] == b)
                                             or (j >= 0 and tabu_until[j] > step and tabu_bin[j] == b)
                                             or (u >= 0 and tabu_until[u] > step and tabu_bin[u] == a)
                                             or (v >= 0 and tabu_until[v] > step and tabu_bin[v] == a))
                                if forbidden and trial_overflow >= run_best:
                                    continue
                                count_change = int(new_a > 0) + int(new_b > 0) - old_count
                                memory_change = heat[a] * (new_a - excess_a) + heat[b] * (new_b - excess_b)
                                new_slack_square = (max(0, delta_mass - excess_a) ** 2
                                                    + max(0, free_b - delta_mass) ** 2)
                                rank = (change, count_change, memory_change,
                                        old_slack_square - new_slack_square)
                                if best_rank is None or rank < best_rank:
                                    best_rank = rank
                                    chosen = (a, b, i, j, u, v, change)
                                    ties = 1
                                elif rank == best_rank:
                                    ties += 1
                                    if rng.randrange(ties) == 0:
                                        chosen = (a, b, i, j, u, v, change)
                if chosen is None:
                    break
                a, b, i, j, u, v, change = chosen
                relocate(i, b)
                relocate(j, b)
                relocate(u, a)
                relocate(v, a)
                overflow += change
                for b0 in (a, b):
                    catalogs[b0] = None
                    if loads[b0] > capacity:
                        overloaded.add(b0)
                    else:
                        overloaded.discard(b0)
                for b0 in overloaded:
                    heat[b0] += 1
                if overflow < run_best:
                    run_best = overflow
                    progress = clock()
                    # Attempts with substantial progress get more time, but are always bounded by the total wall clock.
                    cutoff = min(deadline, max(cutoff, progress + pulse * 0.5))
                save_checkpoint()
                if overflow == 0:
                    candidate = [g[:] for g in groups if g]
                    if len(candidate) < len(best):
                        best = candidate
                        safe = encode(best)
                    break
                if clock() - progress > pulse * (1.0 + strength):
                    break
            round_id += 1
        return safe
    except Exception:
        return safe
