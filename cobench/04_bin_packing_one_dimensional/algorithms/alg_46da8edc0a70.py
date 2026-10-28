# Library slot 5/10  id=alg_46da8edc0a70
# score=143.4  feasible_rate=1.0
# score_vec=[49.0, 213.0, 137.0, 156.0, 145.0, 198.0, 127.0, 97.0, 170.0, 142.0]

import math
import random
import time


def solve(instance, tools, time_limit_s):
    """Anytime bin packing with MBS construction and fixed-cardinality LNS."""
    started = time.time()
    n = 0
    safe = {"bin_activation": [], "item_assignments": []}

    try:
        n = int(instance.get("n", 0))
        capacity = int(instance.get("capacity", 0))
        weights = list(instance.get("weights", []))
        if n <= 0:
            return safe

        # This is feasible for every valid instance and is installed before search.
        safe = {
            "bin_activation": [1] * n,
            "item_assignments": list(range(n)),
        }
        if capacity <= 0 or len(weights) != n:
            return safe

        budget = max(0.0, float(time_limit_s))
        margin = min(0.06, max(0.004, budget * 0.012))
        deadline = started + max(0.0, budget - margin)

        def make_output(bins):
            live = [b for b in bins if b]
            assignments = [0] * n
            for b_idx, items in enumerate(live):
                for item in items:
                    assignments[item] = b_idx
            return {
                "bin_activation": [1] * len(live) + [0] * (n - len(live)),
                "item_assignments": assignments,
            }

        def best_fit(order, initial=None):
            bins = [] if initial is None else [list(b) for b in initial]
            loads = [sum(weights[i] for i in b) for b in bins]
            for item in order:
                w = weights[item]
                chosen = -1
                least_left = capacity + 1
                for b_idx, load in enumerate(loads):
                    left = capacity - load - w
                    if 0 <= left < least_left:
                        chosen = b_idx
                        least_left = left
                if chosen < 0:
                    bins.append([item])
                    loads.append(w)
                else:
                    bins[chosen].append(item)
                    loads[chosen] += w
            return bins

        descending = sorted(range(n), key=lambda i: (-weights[i], i))
        best_bins = best_fit(descending)
        safe = make_output(best_bins)
        lower_bound = (sum(weights) + capacity - 1) // capacity

        # If every bin can contain at most two items, maximum threshold matching is
        # the whole problem.  The two-pointer construction is optimal in this case.
        if min(weights) * 3 > capacity:
            ordered = sorted(range(n), key=lambda i: (weights[i], i))
            paired = []
            lo = 0
            hi = n - 1
            while lo < hi:
                if weights[ordered[lo]] + weights[ordered[hi]] <= capacity:
                    paired.append([ordered[lo], ordered[hi]])
                    lo += 1
                    hi -= 1
                else:
                    paired.append([ordered[hi]])
                    hi -= 1
            if lo == hi:
                paired.append([ordered[lo]])
            return make_output(paired)

        scale = capacity
        for w in weights:
            scale = math.gcd(scale, int(w))
        scaled_capacity = capacity // max(1, scale)

        def mbs_construct(shuffle_candidates):
            """Fill each bin by a residual subset-sum, forcing its largest item."""
            if scaled_capacity > 24000:
                return None
            remaining = list(descending)
            result = []

            while remaining:
                if time.time() >= deadline:
                    return best_fit(remaining, result)
                seed = remaining[0]
                room = (capacity - weights[seed]) // scale
                candidates = remaining[1:]
                if shuffle_candidates:
                    random.shuffle(candidates)

                reachable = bytearray(room + 1)
                parent_item = [-1] * (room + 1)
                parent_sum = [-1] * (room + 1)
                reachable[0] = 1

                for pos, item in enumerate(candidates):
                    sw = weights[item] // scale
                    if sw <= room:
                        for total in range(room, sw - 1, -1):
                            if not reachable[total] and reachable[total - sw]:
                                reachable[total] = 1
                                parent_item[total] = item
                                parent_sum[total] = total - sw
                    if reachable[room] or (pos & 15) == 15 and time.time() >= deadline:
                        break

                fill = room
                while fill > 0 and not reachable[fill]:
                    fill -= 1
                chosen = {seed}
                while fill > 0:
                    item = parent_item[fill]
                    if item < 0:
                        break
                    chosen.add(item)
                    fill = parent_sum[fill]
                result.append(list(chosen))
                remaining = [item for item in remaining if item not in chosen]

            return result

        if time.time() < deadline:
            candidate = mbs_construct(False)
            if candidate is not None and len(candidate) < len(best_bins):
                best_bins = candidate
                safe = make_output(best_bins)
        if len(best_bins) <= lower_bound:
            return safe

        def fullness(bins):
            return sum(sum(weights[i] for i in b) ** 2 for b in bins)

        def fixed_pack(items, bin_count, cutoff):
            """Try to repack a selected neighborhood into exactly bin_count bins."""
            if bin_count <= 0:
                return [] if not items else None
            if sum(weights[i] for i in items) > bin_count * capacity:
                return None

            order = sorted(items, key=lambda i: (-weights[i], random.random()))
            quick = best_fit(order)
            if len(quick) <= bin_count:
                return quick + [[] for _ in range(bin_count - len(quick))]

            loads = [0] * bin_count
            packed = [[] for _ in range(bin_count)]
            failed = set()

            def place(pos):
                if time.time() >= cutoff:
                    return False
                if pos == len(order):
                    return True
                key = (pos, tuple(sorted(loads)))
                if key in failed:
                    return False

                item = order[pos]
                w = weights[item]
                choices = sorted(
                    range(bin_count),
                    key=lambda b: capacity - loads[b] - w
                    if loads[b] + w <= capacity
                    else capacity + loads[b],
                )
                seen_loads = set()
                for b_idx in choices:
                    load = loads[b_idx]
                    if load in seen_loads or load + w > capacity:
                        continue
                    seen_loads.add(load)
                    loads[b_idx] += w
                    packed[b_idx].append(item)
                    if place(pos + 1):
                        return True
                    packed[b_idx].pop()
                    loads[b_idx] -= w
                    if load == 0:
                        break
                failed.add(key)
                return False

            if place(0):
                return packed
            return None

        def choose_group(bins, randomized):
            loads = [sum(weights[i] for i in b) for b in bins]
            low = sorted(range(len(bins)), key=lambda b: loads[b])
            if randomized:
                width = max(1, len(low) // 4)
                target = low[random.randrange(width)]
            else:
                target = low[0]

            others = [b for b in range(len(bins)) if b != target]
            if randomized:
                others.sort(
                    key=lambda b: random.random() ** (1.0 / (capacity - loads[b] + 1)),
                    reverse=True,
                )
            else:
                others.sort(key=lambda b: capacity - loads[b], reverse=True)

            group = [target]
            total_slack = capacity - loads[target]
            for b_idx in others:
                if total_slack >= capacity:
                    break
                group.append(b_idx)
                total_slack += capacity - loads[b_idx]
            if total_slack < capacity:
                return None
            if randomized and len(group) < len(bins) and random.random() < 0.45:
                unused = [b for b in others if b not in group]
                if unused:
                    group.append(random.choice(unused))
            return group

        current = [list(b) for b in best_bins]
        current_fullness = fullness(current)
        turn = 0

        while time.time() < deadline:
            turn += 1
            changed = False

            if turn % 4 == 0:
                candidate = mbs_construct(True)
                if candidate is not None:
                    value = fullness(candidate)
                    if len(candidate) < len(best_bins):
                        best_bins = candidate
                        safe = make_output(best_bins)
                        current = [list(b) for b in candidate]
                        current_fullness = value
                        changed = True
                    elif len(candidate) == len(current) and value > current_fullness:
                        current = [list(b) for b in candidate]
                        current_fullness = value
                        changed = True
            else:
                group = choose_group(current, turn % 3 != 1)
                if group is not None:
                    pool = []
                    for b_idx in group:
                        pool.extend(current[b_idx])
                    remaining_time = deadline - time.time()
                    cutoff = min(deadline, time.time() + max(0.015, remaining_time * 0.16))
                    replacement = fixed_pack(pool, len(group) - 1, cutoff)
                    if replacement is not None:
                        selected = set(group)
                        candidate = [b for b_idx, b in enumerate(current) if b_idx not in selected]
                        candidate.extend([b for b in replacement if b])
                        current = candidate
                        current_fullness = fullness(current)
                        changed = True
                        if len(current) < len(best_bins):
                            best_bins = [list(b) for b in current]
                            safe = make_output(best_bins)

            if len(best_bins) <= lower_bound:
                break
            if not changed and time.time() < deadline:
                # A randomized BFD restart cheaply changes the LNS basin.
                noisy = sorted(
                    range(n),
                    key=lambda i: -(weights[i] * (0.92 + 0.16 * random.random())),
                )
                candidate = best_fit(noisy)
                value = fullness(candidate)
                if len(candidate) < len(best_bins):
                    best_bins = candidate
                    safe = make_output(best_bins)
                    current = [list(b) for b in candidate]
                    current_fullness = value
                elif len(candidate) == len(current) and value > current_fullness:
                    current = candidate
                    current_fullness = value

        return safe
    except Exception:
        return safe
