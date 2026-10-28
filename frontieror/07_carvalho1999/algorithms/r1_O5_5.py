"""Column-generation bin-pattern combination with adaptive-neighbourhood bin packing
Generates candidates by bitset bin filling and bin-pattern crossover, then reduces the bin count with dual knapsack pricing, time-limited integer covering and exact neighbourhood repacking.
"""

import bisect
import math
import random
import time


def solve(instance: dict, time_limit_s: float) -> dict:
    started = time.perf_counter()
    n = 0
    safe = {"bin_activation": [], "item_assignments": []}
    try:
        n = max(0, int(instance["n"]))
        safe = {"bin_activation": [1] * n, "item_assignments": list(range(n))}
        if not n:
            return safe
        capacity = int(instance["capacity"])
        weights = [int(w) for w in instance["weights"]]
        if len(weights) != n or capacity < 0 or any(w < 0 or w > capacity for w in weights):
            return safe
        if not any(weights):
            return {"bin_activation": [0] * n, "item_assignments": [0] * n}
        budget = float(time_limit_s)
        if not math.isfinite(budget) or budget <= 0:
            return safe
        deadline = started + budget * 0.955
        clock = time.perf_counter
        positive = [i for i in range(n) if weights[i]]
        descending = sorted(positive, key=lambda i: (-weights[i], i))
        size = len(positive)
        strength = math.log1p(budget)
        quantum = budget / (18.0 + 1.5 * math.sqrt(size))
        archive_limit = max(2, int(4 + 3 * strength))
        pool_limit = max(16, int(size * (3 + 3 * strength)))
        cache_limit = max(32, int(size * (20 + 80 * strength)))
        bit_limit = max(1024, int(40000 * (1 + strength)))
        seed = n * 1000003 + capacity * 9176
        for i, w in enumerate(weights):
            seed = (seed * 1000003 + (i + 1) * (w + 17)) & 0xFFFFFFFFFFFFFFFF
        rng = random.Random(seed)

        def output(groups):
            # Zero-weight items are assigned with the first bin and do not increase the bin count.
            assignment = [0] * n
            for b, group in enumerate(groups):
                for item in group:
                    assignment[item] = b
            return {"bin_activation": [1] * len(groups) + [0] * (n - len(groups)),
                    "item_assignments": assignment}

        def loads_of(groups):
            return [sum(weights[i] for i in group) for group in groups]

        def quality(groups):
            return (len(groups), -sum(v * v for v in loads_of(groups)))

        def best_fit(order, cutoff, initial=None):
            groups = [] if initial is None else [list(g) for g in initial]
            spaces = sorted((capacity - v, b) for b, v in enumerate(loads_of(groups)))
            for p, item in enumerate(order):
                if (p & 31) == 0 and clock() >= cutoff:
                    groups.extend([i] for i in order[p:])
                    break
                w = weights[item]
                j = bisect.bisect_left(spaces, (w, -1))
                if j == len(spaces):
                    b = len(groups)
                    groups.append([item])
                    bisect.insort(spaces, (capacity - w, b))
                else:
                    space, b = spaces.pop(j)
                    groups[b].append(item)
                    bisect.insort(spaces, (space - w, b))
            return groups

        best = best_fit(descending, deadline)
        best_quality = quality(best)
        safe = output(best)
        lower_bound = max((sum(weights) + capacity - 1) // capacity,
                          sum(2 * weights[i] > capacity for i in positive))
        if len(best) <= lower_bound or clock() >= deadline:
            return safe

        # When no three positive-weight items can share a bin, two-pointer pairing is an exact algorithm.
        if size < 3 or sum(weights[i] for i in descending[-3:]) > capacity:
            ordered = list(reversed(descending))
            paired = []
            lo, hi = 0, size - 1
            while lo <= hi:
                group = [ordered[hi]]
                if lo < hi and weights[ordered[lo]] + weights[ordered[hi]] <= capacity:
                    group.append(ordered[lo])
                    lo += 1
                paired.append(group)
                hi -= 1
            return output(paired)

        type_weights = sorted({weights[i] for i in positive}, reverse=True)
        type_index = {w: t for t, w in enumerate(type_weights)}
        item_type = [type_index.get(w, -1) for w in weights]
        type_items = [[] for _ in type_weights]
        for i in positive:
            type_items[item_type[i]].append(i)
        demand = [len(items) for items in type_items]
        common = capacity
        for w in type_weights:
            common = math.gcd(common, w)
        scaled_capacity = capacity // common
        scaled_weights = [w // common for w in type_weights]
        types = len(type_weights)
        patterns = {}
        archive = []
        archive_seen = set()
        protected_patterns = set()
        serial = 0

        def pattern_of(group):
            counts = {}
            for i in group:
                t = item_type[i]
                counts[t] = counts.get(t, 0) + 1
            return tuple(sorted(counts.items()))

        def add_pattern(pattern):
            if pattern:
                patterns[pattern] = serial

        def prune_pool():
            if len(patterns) <= pool_limit:
                return
            keep = {pattern_of(g) for g in best} | protected_patterns
            optional = [p for p in patterns if p not in keep]
            optional.sort(key=lambda p: (sum(type_weights[t] * c for t, c in p), patterns[p]),
                          reverse=True)
            keep.update(optional[:max(0, pool_limit - len(keep))])
            for p in list(patterns):
                if p not in keep:
                    del patterns[p]

        def consider(candidate):
            nonlocal best, best_quality, safe, serial
            serial += 1
            candidate = [g for g in candidate if g]
            value = quality(candidate)
            signature = tuple(sorted(pattern_of(g) for g in candidate))
            for p in signature:
                add_pattern(p)
            if value < best_quality:
                best = [list(g) for g in candidate]
                best_quality = value
                safe = output(best)
            if len(candidate) <= len(best) + 1 and signature not in archive_seen:
                archive_seen.add(signature)
                archive.append((value, signature, [list(g) for g in candidate]))
                archive.sort(key=lambda entry: entry[0])
                if len(archive) > archive_limit:
                    archive_seen.discard(archive.pop()[1])
            if len(patterns) > pool_limit * 1.2:
                prune_pool()

        consider(best)

        # A max-residual-capacity segment tree implements first fit, avoiding bin-by-bin scanning.
        leaves = 1 << (size - 1).bit_length()
        tree = [0] * (leaves * 2)
        first = []
        first_loads = []
        for p, item in enumerate(descending):
            if (p & 31) == 0 and clock() >= deadline:
                first.extend([i] for i in descending[p:])
                break
            w = weights[item]
            if tree[1] < w:
                b = len(first)
                first.append([])
                first_loads.append(0)
            else:
                node = 1
                while node < leaves:
                    node *= 2
                    if tree[node] < w:
                        node += 1
                b = node - leaves
            first[b].append(item)
            first_loads[b] += w
            node = leaves + b
            tree[node] = capacity - first_loads[b]
            while node > 1:
                node //= 2
                tree[node] = max(tree[2 * node], tree[2 * node + 1])
        consider(first)

        def varied(items, amplitude=None):
            order = list(items)
            if amplitude is None:
                amplitude = capacity * rng.random() ** 2
            if rng.random() < 0.12:
                rng.shuffle(order)
            else:
                order.sort(key=lambda i: weights[i] + rng.random() * amplitude, reverse=True)
            return order

        def subset(candidates, room, cutoff, randomized):
            if room <= 0 or not candidates:
                return []
            ordered = list(candidates)
            if randomized:
                rng.shuffle(ordered)
            units = room // common
            # Limit the space of a single bitset history; for oversized instances fall back to conservative greedy filling.
            use_bits = units <= bit_limit and units * len(ordered) <= 128000000
            if use_bits:
                bits = 1
                mask = (1 << (units + 1)) - 1
                history = []
                for item in ordered:
                    if clock() >= cutoff:
                        break
                    value = weights[item] // common
                    if value <= units:
                        before = bits
                        bits = (bits | (bits << value)) & mask
                        if bits != before:
                            history.append((item, value, before))
                        if bits >> units:
                            break
                target = bits.bit_length() - 1
                chosen = []
                for item, value, before in reversed(history):
                    if not ((before >> target) & 1):
                        chosen.append(item)
                        target -= value
                return chosen
            chosen = []
            for p, item in enumerate(ordered):
                if (p & 31) == 0 and clock() >= cutoff:
                    break
                if weights[item] <= room:
                    chosen.append(item)
                    room -= weights[item]
            return chosen

        def completion(items, cutoff, randomized):
            remaining = sorted(items, key=lambda i: (-weights[i], i))
            result = []
            while remaining and clock() < cutoff:
                anchor_pos = 0
                if randomized and rng.random() < 0.15:
                    width = min(len(remaining), max(1, int(1 + strength)))
                    anchor_pos = rng.randrange(width)
                anchor = remaining[anchor_pos]
                room = capacity - weights[anchor]
                candidates = [i for i in remaining if i != anchor and weights[i] <= room]
                chosen = subset(candidates, room, cutoff, randomized)
                result.append([anchor] + chosen)
                taken = set(chosen)
                taken.add(anchor)
                remaining = [i for i in remaining if i not in taken]
            if remaining:
                result = best_fit(remaining, min(deadline, cutoff + quantum * 0.15), result)
            return result

        def exact_repack(items, count, cutoff):
            if count <= 0 or sum(weights[i] for i in items) > count * capacity:
                return None
            order = sorted(items, key=lambda i: (-weights[i], i))
            trial = best_fit(order, cutoff)
            if len(trial) <= count:
                return trial
            spaces = [capacity] * count
            path = [-1] * len(order)
            frames = [None] * len(order)
            failed = set()
            pos = 0
            # An explicit stack avoids recursion-depth limits when there are many items; at the deadline the whole unfinished neighbourhood is discarded.
            while clock() < cutoff:
                if pos == len(order):
                    result = [[] for _ in spaces]
                    for i, b in zip(order, path):
                        result[b].append(i)
                    return [g for g in result if g]
                if frames[pos] is None:
                    state = (pos, tuple(sorted(spaces)))
                    choices = []
                    if state not in failed:
                        seen = set()
                        for b in sorted(range(count), key=lambda b: spaces[b], reverse=True):
                            space = spaces[b]
                            if space >= weights[order[pos]] and space not in seen:
                                seen.add(space)
                                choices.append(b)
                    frames[pos] = (state, choices)
                state, choices = frames[pos]
                if choices:
                    b = choices.pop()
                    path[pos] = b
                    spaces[b] -= weights[order[pos]]
                    pos += 1
                    if pos < len(order):
                        frames[pos] = None
                else:
                    if len(failed) >= cache_limit:
                        failed.clear()
                    failed.add(state)
                    frames[pos] = None
                    if not pos:
                        return None
                    pos -= 1
                    spaces[path[pos]] += weights[order[pos]]
            return None

        def neighborhood(parent, cutoff):
            loads = loads_of(parent)
            ranked = sorted(range(len(parent)),
                            key=lambda b: loads[b] + rng.random() * capacity * 0.18)
            selected = []
            slack = 0
            for b in ranked:
                selected.append(b)
                slack += capacity - loads[b]
                if slack >= capacity:
                    break
            if slack < capacity:
                return None
            chosen = set(selected)
            donors = [b for b in range(len(parent)) if b not in chosen]
            gap = max(capacity - loads[b] for b in selected)
            donors.sort(key=lambda b: (sum(weights[i] > gap for i in parent[b]),
                                      -sum(weights[i] for i in parent[b] if weights[i] <= gap)
                                      + rng.random() * capacity * 0.16))
            extra = min(len(donors), rng.randrange(1 + max(1, int(2 * strength))))
            selected.extend(donors[:extra])
            chosen.update(donors[:extra])
            items = [i for b in selected for i in parent[b]]
            repaired = exact_repack(items, len(selected) - 1,
                                    min(cutoff, clock() + (cutoff - clock()) * 0.8))
            if repaired is None and clock() < cutoff:
                repaired = best_fit(varied(items), cutoff)
                if len(repaired) > len(selected):
                    return None
            if repaired is None:
                return None
            return [list(parent[b]) for b in range(len(parent)) if b not in chosen] + repaired

        def crossover(a, b, cutoff):
            choices = [list(g) for g in a] + [list(g) for g in b]
            choices.sort(key=lambda g: sum(weights[i] for i in g)
                         + rng.random() * capacity * 0.2, reverse=True)
            used = set()
            locked = []
            target = size * rng.uniform(0.3, 0.75)
            for group in choices:
                if len(used) >= target or clock() >= cutoff:
                    break
                if all(i not in used for i in group):
                    used.update(group)
                    locked.append(group)
            missing = [i for i in positive if i not in used]
            return best_fit(varied(missing, capacity * rng.random() * 0.25), cutoff, locked)

        # Scientific libraries are imported on demand only after a complete feasible solution exists; failure does not affect the pure-Python route.
        native = None
        native_disabled = False
        dual_previous = None

        def unpack_patterns(keys, multiplicities, cutoff, fractional=False):
            available = [list(items) for items in type_items]
            result = []
            indices = list(range(len(keys)))
            if fractional:
                indices.sort(key=lambda j: multiplicities[j] - math.floor(multiplicities[j]),
                             reverse=True)
            for j in indices:
                if clock() >= cutoff:
                    break
                value = float(multiplicities[j])
                if not math.isfinite(value) or value < 0:
                    continue
                copies = int(math.floor(value + 1e-7)) if fractional else int(round(value))
                copies = min(copies, size)
                pattern = keys[j]
                for _ in range(copies):
                    group = []
                    for t, count in pattern:
                        for _ in range(min(count, len(available[t]))):
                            group.append(available[t].pop())
                    if group:
                        result.append(group)
                    else:
                        break
            if fractional:
                # After the integer part, greedily pick mutually non-conflicting complete bin patterns by fractional support.
                for j in indices:
                    if clock() >= cutoff:
                        break
                    fraction = float(multiplicities[j]) % 1.0
                    if fraction <= 1e-7:
                        continue
                    pattern = keys[j]
                    if all(len(available[t]) >= count for t, count in pattern):
                        group = []
                        for t, count in pattern:
                            for _ in range(count):
                                group.append(available[t].pop())
                        result.append(group)
            remaining = [i for items in available for i in items]
            remaining.sort(key=lambda i: weights[i], reverse=True)
            return best_fit(remaining, cutoff, result)

        def pricing(duals, cutoff, np):
            # Binary grouping for the bounded knapsack, so items of equal weight no longer get separate pricing states.
            chunks = []
            for t, weight in enumerate(scaled_weights):
                left = min(demand[t], scaled_capacity // weight)
                step = 1
                while left:
                    count = min(step, left)
                    chunks.append((t, count, weight * count, float(duals[t]) * count))
                    left -= count
                    step *= 2
            operations = scaled_capacity * len(chunks)
            allowance = int(1500000 * (1 + strength))
            if scaled_capacity > bit_limit or operations > allowance:
                # With large integer capacities only heuristic bin patterns are produced, and no lower bound is claimed from them.
                order = sorted(range(types), key=lambda t: float(duals[t]) / type_weights[t],
                               reverse=True)
                counts = {}
                room = capacity
                for t in order:
                    if clock() >= cutoff:
                        return None
                    count = min(demand[t], room // type_weights[t])
                    if count:
                        counts[t] = count
                        room -= count * type_weights[t]
                add_pattern(tuple(sorted(counts.items())))
                return None
            dp = np.zeros(scaled_capacity + 1, dtype=float)
            history = []
            for t, count, weight, profit in chunks:
                if clock() >= cutoff:
                    return None
                candidate = dp[:-weight] + profit
                take = candidate > dp[weight:]
                dp[weight:] = np.maximum(dp[weight:], candidate)
                history.append((t, count, weight, take))
            maximum = float(dp[-1])
            positions = np.flatnonzero(dp[1:] > dp[:-1] + 1e-10) + 1
            positions = [int(p) for p in positions if dp[p] > 1.0000001]
            rng.shuffle(positions)
            width = max(1, int((1 + strength) * math.sqrt(types)))
            targets = [scaled_capacity] + positions[:width]
            for target in targets:
                if clock() >= cutoff:
                    break
                counts = {}
                for t, count, weight, take in reversed(history):
                    if target >= weight and take[target - weight]:
                        counts[t] = counts.get(t, 0) + count
                        target -= weight
                room = capacity - sum(type_weights[t] * count for t, count in counts.items())
                # Extend patterns backtracked from lower capacities into maximal patterns.
                for t in range(types):
                    count = min(demand[t] - counts.get(t, 0), room // type_weights[t])
                    if count:
                        counts[t] = counts.get(t, 0) + count
                        room -= type_weights[t] * count
                add_pattern(tuple(sorted(counts.items())))
            return maximum

        def pattern_search(cutoff):
            nonlocal native, native_disabled, lower_bound, dual_previous, protected_patterns
            if native_disabled or clock() >= cutoff:
                return
            try:
                if native is None:
                    if cutoff - clock() < 0.25:
                        return
                    import numpy as np
                    from scipy.optimize import Bounds, LinearConstraint, linprog, milp
                    from scipy.sparse import csc_matrix
                    native = (np, Bounds, LinearConstraint, linprog, milp, csc_matrix)
                np, Bounds, LinearConstraint, linprog, milp, csc_matrix = native
                prune_pool()
                generation_end = clock() + max(0.0, cutoff - clock()) * 0.52
                demand_array = np.asarray(demand, dtype=float)

                def matrix():
                    keys = list(patterns)
                    rows, cols, values = [], [], []
                    for j, pattern in enumerate(keys):
                        for t, count in pattern:
                            rows.append(t)
                            cols.append(j)
                            values.append(float(count))
                    return keys, csc_matrix((values, (rows, cols)), shape=(types, len(keys)))

                while clock() < generation_end and len(best) > lower_bound:
                    keys, a = matrix()
                    remaining = generation_end - clock()
                    if remaining <= 0:
                        break
                    lp = linprog(np.ones(len(keys)), A_ub=-a, b_ub=-demand_array,
                                 bounds=(0, None), method="highs",
                                 options={"time_limit": remaining * 0.45, "presolve": True})
                    if not lp.success or lp.x is None or clock() >= generation_end:
                        break
                    protected_patterns = {keys[j] for j, x in enumerate(lp.x) if x > 1e-7}
                    consider(unpack_patterns(keys, lp.x, generation_end, True))
                    duals = np.maximum(0.0, -lp.ineqlin.marginals)
                    maximum = pricing(duals, generation_end, np)
                    if maximum is not None:
                        # Dividing the dual vector by the maximum of exact pricing gives a feasible dual over the full pattern space.
                        bound = float(duals @ demand_array) / max(1.0, maximum)
                        valid_bound = int(math.ceil(bound - 1e-6 * (1 + abs(bound))))
                        lower_bound = max(lower_bound, min(valid_bound, len(best)))
                        if maximum <= 1.0000001:
                            break
                    if dual_previous is not None and clock() < generation_end:
                        pricing((duals + dual_previous) * 0.5, generation_end, np)
                    dual_previous = duals
                    prune_pool()
                if len(best) <= lower_bound or clock() >= cutoff:
                    return
                keys, a = matrix()
                remaining = cutoff - clock()
                if remaining <= 0:
                    return
                result = milp(np.ones(len(keys)), integrality=np.ones(len(keys)),
                              bounds=Bounds(0, len(best) - 1),
                              constraints=[LinearConstraint(a, demand_array, np.inf),
                                           LinearConstraint(np.ones((1, len(keys))),
                                                            -np.inf, len(best) - 1)],
                              options={"time_limit": remaining * 0.83,
                                       "mip_rel_gap": 0.0, "presolve": True})
                if result.x is not None:
                    candidate = unpack_patterns(keys, result.x, deadline)
                    consider(candidate)
            except Exception:
                native_disabled = True

        opening_end = min(deadline, started + budget * 0.16)
        opening_turn = 0
        while clock() < opening_end and len(best) > lower_bound:
            cutoff = min(opening_end, clock() + quantum)
            consider(completion(positive, cutoff, opening_turn != 0))
            opening_turn += 1

        next_master = clock()
        turn = 0
        while clock() < deadline and len(best) > lower_bound:
            now = clock()
            remaining = deadline - now
            if not native_disabled and now >= next_master and remaining > budget * 0.035:
                cutoff = min(deadline - budget * 0.015, now + min(budget * 0.24, remaining * 0.65))
                pattern_search(cutoff)
                next_master = clock() + budget * 0.13
                continue
            cutoff = min(deadline, now + min(quantum * (1 + turn % 3), remaining * 0.25))
            if turn % 4 == 0:
                consider(completion(positive, cutoff, True))
            elif turn % 4 == 1 and len(archive) > 1:
                a = archive[rng.randrange(len(archive))][2]
                b = archive[rng.randrange(len(archive))][2]
                consider(crossover(a, b, cutoff))
            else:
                parent = best if turn % 2 else archive[rng.randrange(len(archive))][2]
                child = neighborhood(parent, cutoff)
                if child is not None:
                    consider(child)
            turn += 1
        return safe
    except Exception:
        return safe
