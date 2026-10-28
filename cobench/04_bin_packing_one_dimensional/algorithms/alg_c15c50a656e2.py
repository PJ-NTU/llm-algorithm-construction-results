# Library slot 9/10  id=alg_c15c50a656e2
# score=143.7  feasible_rate=1.0
# score_vec=[49.0, 213.0, 137.0, 156.0, 146.0, 199.0, 127.0, 97.0, 171.0, 142.0]

import math
import random
import time


def solve(instance: dict, time_limit_s: float) -> dict:
    t0 = time.time()

    try:
        n = int(instance["n"])
        capacity = int(instance["capacity"])
        weights = [int(w) for w in instance["weights"]]
        if n != len(weights):
            n = len(weights)
        budget = max(0.0, float(time_limit_s))
    except Exception:
        n = 0
        capacity = 1
        weights = []
        budget = 0.0

    deadline = t0 + budget * 0.85

    def format_solution(bins):
        assignments = [0] * n
        activation = [0] * n
        for b_idx, items in enumerate(bins[:n]):
            activation[b_idx] = 1
            for item in items:
                if 0 <= item < n:
                    assignments[item] = b_idx
        return {
            "bin_activation": activation,
            "item_assignments": assignments,
        }

    if n == 0:
        return format_solution([])

    # A complete solution is established before doing any search.
    def best_fit(order, initial_bins=None):
        bins = [] if initial_bins is None else [list(b) for b in initial_bins]
        loads = [sum(weights[i] for i in b) for b in bins]
        for item in order:
            w = weights[item]
            chosen = -1
            least_space = capacity + 1
            for b_idx, load in enumerate(loads):
                space = capacity - load - w
                if 0 <= space < least_space:
                    chosen = b_idx
                    least_space = space
            if chosen < 0:
                bins.append([item])
                loads.append(w)
            else:
                bins[chosen].append(item)
                loads[chosen] += w
        return bins

    descending = sorted(range(n), key=lambda i: (-weights[i], i))
    best = best_fit(descending)

    if capacity <= 0 or any(w < 0 or w > capacity for w in weights):
        return format_solution(best)

    total_weight = sum(weights)
    volume_lb = (total_weight + capacity - 1) // capacity

    if total_weight == 0:
        return {
            "bin_activation": [0] * n,
            "item_assignments": [0] * n,
        }

    lower_bound = max(volume_lb, sum(1 for w in weights if 2 * w > capacity))

    # When no three items fit together, bin packing is exactly maximum matching
    # in a threshold graph.  The two-pointer greedy is optimal for that graph.
    positive = [w for w in weights if w > 0]
    if positive and 3 * min(positive) > capacity and len(positive) == n:
        ordered = sorted(range(n), key=lambda i: (weights[i], i))
        paired_bins = []
        lo = 0
        hi = n - 1
        while lo <= hi:
            if lo == hi:
                paired_bins.append([ordered[hi]])
                break
            if weights[ordered[lo]] + weights[ordered[hi]] <= capacity:
                paired_bins.append([ordered[hi], ordered[lo]])
                lo += 1
                hi -= 1
            else:
                paired_bins.append([ordered[hi]])
                hi -= 1
        return format_solution(paired_bins)

    if len(best) == lower_bound:
        return format_solution(best)

    seed = (n * 1000003 + capacity * 9176 + total_weight * 37) & 0xFFFFFFFF
    rng = random.Random(seed)

    common_divisor = capacity
    for w in weights:
        common_divisor = math.gcd(common_divisor, w)
    common_divisor = max(1, common_divisor)
    scaled_capacity = capacity // common_divisor
    scaled_weights = [w // common_divisor for w in weights]

    def subset_for_anchor(candidates, room, stop_at, randomized):
        if room <= 0 or not candidates:
            return []

        candidate_order = list(candidates)
        if randomized:
            rng.shuffle(candidate_order)
        else:
            candidate_order.sort(key=lambda i: (-weights[i], i))

        scaled_room = room // common_divisor
        # The sparse exact DP is extremely effective on the small integer
        # capacities common in benchmark bin-packing instances.  On a huge
        # numeric range, use a bounded-memory best-fill fallback instead.
        if scaled_capacity > 50000:
            chosen = []
            left = room
            for item in candidate_order:
                if weights[item] <= left:
                    chosen.append(item)
                    left -= weights[item]
            return chosen

        states = {0: None}
        best_sum = 0
        for item in candidate_order:
            if time.time() >= stop_at:
                break
            iw = scaled_weights[item]
            additions = {}
            for old_sum, old_node in list(states.items()):
                new_sum = old_sum + iw
                if new_sum <= scaled_room and new_sum not in states and new_sum not in additions:
                    additions[new_sum] = (item, old_node)
                    if new_sum > best_sum:
                        best_sum = new_sum
            states.update(additions)
            if best_sum == scaled_room:
                break

        node = states.get(best_sum)
        chosen = []
        while node is not None:
            chosen.append(node[0])
            node = node[1]
        return chosen

    def minimum_bin_slack(stop_at, randomized=False):
        remaining = set(range(n))
        bins = []
        while remaining and time.time() < stop_at:
            anchor = max(remaining, key=lambda i: (weights[i], -i))
            remaining.remove(anchor)
            chosen = subset_for_anchor(
                remaining,
                capacity - weights[anchor],
                stop_at,
                randomized,
            )
            current = [anchor]
            for item in chosen:
                if item in remaining:
                    remaining.remove(item)
                    current.append(item)
            bins.append(current)

        if remaining:
            tail = sorted(remaining, key=lambda i: (-weights[i], i))
            return best_fit(tail, bins)
        return bins

    # Spend a small opening slice diversifying high-quality constructions.
    opening_end = min(deadline, t0 + budget * 0.16)
    construction_round = 0
    while time.time() < opening_end:
        candidate = minimum_bin_slack(opening_end, construction_round > 0)
        if len(candidate) < len(best):
            best = candidate
            if len(best) == lower_bound:
                return format_solution(best)
        construction_round += 1

    def target_feasibility(target_bins, stop_at, randomized):
        if target_bins < lower_bound or target_bins <= 0 or n > 700:
            return None

        order = list(descending)
        if randomized:
            # Only ties are shuffled: largest-first is the crucial pruning rule.
            start = 0
            while start < n:
                end = start + 1
                while end < n and weights[order[end]] == weights[order[start]]:
                    end += 1
                group = order[start:end]
                rng.shuffle(group)
                order[start:end] = group
                start = end

        residual = [capacity] * target_bins
        contents = [[] for _ in range(target_bins)]
        timed_out = [False]
        failed = set()
        use_memo = n <= 70 and target_bins <= 25

        def place(pos):
            if time.time() >= stop_at:
                timed_out[0] = True
                return False
            if pos == n:
                return True

            item = order[pos]
            w = weights[item]
            if w > max(residual):
                return False

            memo_key = None
            if use_memo:
                memo_key = (pos, tuple(sorted(residual)))
                if memo_key in failed:
                    return False

            choices = []
            seen_space = set()
            for b_idx, space in enumerate(residual):
                if space >= w and space not in seen_space:
                    seen_space.add(space)
                    choices.append((space - w, b_idx))
            choices.sort(reverse=randomized)

            for _, b_idx in choices:
                residual[b_idx] -= w
                contents[b_idx].append(item)
                if place(pos + 1):
                    return True
                contents[b_idx].pop()
                residual[b_idx] += w
                if timed_out[0]:
                    return False

            if use_memo and memo_key is not None:
                failed.add(memo_key)
            return False

        if place(0):
            return [b for b in contents if b]
        return None

    def destroy_and_repack(source, selected, stop_at, randomized):
        selected_set = set(selected)
        pool = []
        fixed = []
        for b_idx, items in enumerate(source):
            if b_idx in selected_set:
                pool.extend(items)
            else:
                fixed.append(list(items))

        new_slots = max(0, len(selected) - 1)
        slots = fixed + [[] for _ in range(new_slots)]
        residual = [capacity - sum(weights[i] for i in b) for b in fixed]
        residual.extend([capacity] * new_slots)
        if sum(weights[i] for i in pool) > sum(residual):
            return None

        pool.sort(key=lambda i: (-weights[i], i))
        if len(pool) > 700:
            return None
        if randomized:
            start = 0
            while start < len(pool):
                end = start + 1
                while end < len(pool) and weights[pool[end]] == weights[pool[start]]:
                    end += 1
                group = pool[start:end]
                rng.shuffle(group)
                pool[start:end] = group
                start = end

        timed_out = [False]

        def place(pos):
            if time.time() >= stop_at:
                timed_out[0] = True
                return False
            if pos == len(pool):
                return True

            item = pool[pos]
            w = weights[item]
            choices = []
            seen_space = set()
            for slot_idx, space in enumerate(residual):
                if space >= w and space not in seen_space:
                    seen_space.add(space)
                    choices.append((space - w, slot_idx))
            choices.sort(reverse=randomized)

            for _, slot_idx in choices:
                residual[slot_idx] -= w
                slots[slot_idx].append(item)
                if place(pos + 1):
                    return True
                slots[slot_idx].pop()
                residual[slot_idx] += w
                if timed_out[0]:
                    return False
            return False

        if place(0):
            return [b for b in slots if b]
        return None

    search_round = 0
    failed_selections = set()
    while time.time() < deadline:
        now = time.time()
        remaining_time = deadline - now
        if remaining_time <= 0:
            break

        # Occasionally attack the complete k-1-bin feasibility problem.  Each
        # attempt gets a wall-clock slice so a hard branch cannot consume the
        # whole anytime budget.
        if search_round % 9 == 0:
            slice_end = min(deadline, now + max(0.01, min(0.22, remaining_time * 0.12)))
            candidate = target_feasibility(len(best) - 1, slice_end, search_round > 0)
            if candidate is not None and len(candidate) < len(best):
                best = candidate
                failed_selections.clear()
                if len(best) == lower_bound:
                    return format_solution(best)
                search_round += 1
                continue

        k = len(best)
        if k <= 1:
            break
        max_destroy = min(8, k)
        destroy_count = 1 + (search_round % max_destroy)
        loads = [sum(weights[i] for i in b) for b in best]
        by_load = sorted(range(k), key=lambda b: (loads[b], b))

        mode = search_round % 4
        if mode == 0:
            offset_range = max(1, min(k - destroy_count + 1, 12))
            offset = (search_round // 4) % offset_range
            selected = by_load[offset:offset + destroy_count]
        elif mode == 1:
            pool_size = min(k, max(destroy_count, 3 * destroy_count))
            selected = rng.sample(by_load[:pool_size], destroy_count)
        elif mode == 2:
            selected = [by_load[0]]
            choices = [b for b in range(k) if b != by_load[0]]
            if destroy_count > 1:
                selected.extend(rng.sample(choices, destroy_count - 1))
        else:
            selected = rng.sample(range(k), destroy_count)

        selection_key = tuple(sorted(selected))
        if selection_key not in failed_selections:
            slice_end = min(
                deadline,
                time.time() + max(0.008, min(0.12, remaining_time * 0.08)),
            )
            candidate = destroy_and_repack(
                best,
                selected,
                slice_end,
                mode in (1, 3),
            )
            if candidate is not None and len(candidate) < len(best):
                best = candidate
                failed_selections.clear()
                if len(best) == lower_bound:
                    return format_solution(best)
            else:
                failed_selections.add(selection_key)

        # Fresh subset-sum constructions explore a different basin from LNS.
        if search_round % 13 == 12 and time.time() < deadline:
            slice_end = min(deadline, time.time() + min(0.16, remaining_time * 0.08))
            candidate = minimum_bin_slack(slice_end, True)
            if len(candidate) < len(best):
                best = candidate
                failed_selections.clear()
                if len(best) == lower_bound:
                    return format_solution(best)

        search_round += 1

    return format_solution(best)
