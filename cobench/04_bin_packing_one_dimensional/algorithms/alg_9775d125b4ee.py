# Library slot 4/10  id=alg_9775d125b4ee
# score=143.3  feasible_rate=1.0
# score_vec=[49.0, 213.0, 137.0, 155.0, 146.0, 198.0, 127.0, 97.0, 170.0, 141.0]

import random
import time
from math import gcd


def solve(instance: dict, tools, time_limit_s: float) -> dict:
    """Anytime bin packing using exact pairing, bin completion, and ejection chains."""
    started = time.time()

    try:
        n = instance["n"]
        capacity = instance["capacity"]
        weights = instance["weights"]

        if n <= 0:
            return {"bin_activation": [], "item_assignments": []}

        budget = max(0.0, float(time_limit_s))
        margin = max(0.015, min(0.20, budget * 0.035))
        deadline = started + max(0.0, budget - margin)
        seed = n * 1000003 + capacity * 9176
        for i, w in enumerate(weights):
            seed = (seed * 1000003 + (i + 1) * w) & 0xFFFFFFFFFFFFFFFF
        rng = random.Random(seed)
        descending = sorted(range(n), key=lambda i: (weights[i], -i), reverse=True)

        def insert_best_fit(bins, loads, items, mode=0):
            order = list(items)
            if mode == 0:
                order.sort(key=lambda i: weights[i], reverse=True)
            elif mode == 1:
                order.sort(
                    key=lambda i: weights[i] + rng.random() * max(1.0, capacity * 0.035),
                    reverse=True,
                )
            else:
                rng.shuffle(order)
                order.sort(key=lambda i: weights[i] // max(1, capacity // 20), reverse=True)

            for item in order:
                w = weights[item]
                chosen = -1
                smallest_space = capacity + 1
                for b, load in enumerate(loads):
                    space = capacity - load - w
                    if 0 <= space < smallest_space:
                        smallest_space = space
                        chosen = b
                if chosen < 0:
                    bins.append([item])
                    loads.append(w)
                else:
                    bins[chosen].append(item)
                    loads[chosen] += w

        def best_fit_decreasing():
            bins = []
            loads = []
            insert_best_fit(bins, loads, descending)
            return bins

        def exact_pairing():
            """Optimal when every feasible bin contains at most two items."""
            ordered = sorted(range(n), key=lambda i: (weights[i], i))
            bins = []
            left = 0
            right = n - 1
            while left <= right:
                if left == right:
                    bins.append([ordered[right]])
                    break
                if weights[ordered[left]] + weights[ordered[right]] <= capacity:
                    bins.append([ordered[right], ordered[left]])
                    left += 1
                    right -= 1
                else:
                    bins.append([ordered[right]])
                    right -= 1
            return bins

        common = capacity
        for w in weights:
            common = gcd(common, w)
        common = max(1, common)
        scaled = [w // common for w in weights]
        dp_limit = 24000

        def complete_bins(randomized):
            """MBS-style construction: anchor a largest item and fill its residual exactly."""
            alive = [True] * n
            left_count = n
            bins = []
            loads = []

            while left_count and time.time() < deadline:
                anchor = -1
                for item in descending:
                    if alive[item]:
                        anchor = item
                        break
                if anchor < 0:
                    break
                alive[anchor] = False
                left_count -= 1
                chosen_items = [anchor]
                residual = capacity - weights[anchor]

                candidates = [
                    i for i in range(n)
                    if alive[i] and weights[i] <= residual
                ]
                if randomized:
                    rng.shuffle(candidates)
                else:
                    candidates.sort(key=lambda i: (weights[i], -i), reverse=True)

                residual_units = residual // common
                if candidates and residual_units <= dp_limit and time.time() < deadline:
                    mask = (1 << (residual_units + 1)) - 1
                    bits = 1
                    history = []
                    for item in candidates:
                        if time.time() >= deadline:
                            break
                        before = bits
                        bits = (bits | (bits << scaled[item])) & mask
                        history.append((item, before))
                        if (bits >> residual_units) & 1:
                            break
                    target = bits.bit_length() - 1
                    for item, before in reversed(history):
                        if ((before >> target) & 1) == 0:
                            chosen_items.append(item)
                            target -= scaled[item]
                else:
                    used = 0
                    for item in candidates:
                        if used + weights[item] <= residual:
                            chosen_items.append(item)
                            used += weights[item]

                load = 0
                packed = []
                for item in chosen_items:
                    if alive[item] or item == anchor:
                        if item != anchor:
                            alive[item] = False
                            left_count -= 1
                        packed.append(item)
                        load += weights[item]
                bins.append(packed)
                loads.append(load)

            if left_count:
                remainder = [i for i in range(n) if alive[i]]
                insert_best_fit(bins, loads, remainder)
            return bins

        def normalized(candidate):
            return [list(b) for b in candidate if b]

        def loads_of(candidate):
            return [sum(weights[i] for i in b) for b in candidate]

        def ruin_and_recreate(source, turn):
            count = len(source)
            if count < 2:
                return [list(b) for b in source]
            loads = loads_of(source)
            width = min(count, 2 + rng.randrange(max(1, min(14, count) - 1)))
            ranked = sorted(range(count), key=lambda b: loads[b])
            selected = set(ranked[: max(1, width // 2)])
            pool = [b for b in range(count) if b not in selected]
            rng.shuffle(pool)
            for b in pool:
                if len(selected) >= width:
                    break
                selected.add(b)

            kept = [list(source[b]) for b in range(count) if b not in selected]
            kept_loads = [loads[b] for b in range(count) if b not in selected]
            unpacked = []
            for b in selected:
                unpacked.extend(source[b])
            insert_best_fit(kept, kept_loads, unpacked, turn % 3)
            return normalized(kept)

        def try_empty_bin(source, target, local_deadline):
            bins = [list(source[b]) for b in range(len(source)) if b != target]
            loads = loads_of(bins)
            pending = list(source[target])
            protected = [False] * n

            def place(items):
                if not items:
                    return True
                if time.time() >= local_deadline:
                    return False

                pick_pos = max(range(len(items)), key=lambda p: weights[items[p]])
                item = items[pick_pos]
                rest = items[:pick_pos] + items[pick_pos + 1:]
                w = weights[item]

                direct = []
                seen_spaces = set()
                for b, load in enumerate(loads):
                    space = capacity - load
                    if w <= space and space not in seen_spaces:
                        seen_spaces.add(space)
                        direct.append((space - w, b))
                direct.sort()
                for unused, b in direct:
                    bins[b].append(item)
                    loads[b] += w
                    protected[item] = True
                    if place(rest):
                        return True
                    protected[item] = False
                    loads[b] -= w
                    bins[b].pop()
                    if time.time() >= local_deadline:
                        return False

                swaps = []
                seen_swaps = set()
                for b, load in enumerate(loads):
                    for pos, displaced in enumerate(bins[b]):
                        dw = weights[displaced]
                        signature = (load, dw)
                        if (
                            not protected[displaced]
                            and dw < w
                            and load - dw + w <= capacity
                            and signature not in seen_swaps
                        ):
                            seen_swaps.add(signature)
                            swaps.append((capacity - load + dw - w, -dw, b, pos, displaced))
                swaps.sort()
                for unused, neg_weight, b, pos, displaced in swaps:
                    bins[b][pos] = item
                    loads[b] += w - weights[displaced]
                    protected[item] = True
                    if place(rest + [displaced]):
                        return True
                    protected[item] = False
                    loads[b] -= w - weights[displaced]
                    bins[b][pos] = displaced
                    if time.time() >= local_deadline:
                        return False
                return False

            if place(pending):
                return normalized(bins)
            return None

        initial = best_fit_decreasing()
        pair_solution = exact_pairing()
        best = initial if len(initial) <= len(pair_solution) else pair_solution

        if time.time() < deadline:
            completion = complete_bins(False)
            if len(completion) < len(best):
                best = completion

        working = [list(b) for b in best]
        total_weight = sum(weights)
        volume_bound = (total_weight + capacity - 1) // capacity
        pair_proven = min(weights) * 3 > capacity
        turn = 0

        while time.time() < deadline:
            turn += 1
            candidate = None
            if turn % 5 == 0 and not pair_proven:
                candidate = complete_bins(True)
            elif turn % 2 == 0:
                candidate = ruin_and_recreate(working, turn)
            else:
                current_loads = loads_of(working)
                if current_loads:
                    if turn % 3 == 0:
                        target = rng.randrange(len(working))
                    else:
                        target = min(range(len(working)), key=lambda b: current_loads[b])
                    slice_end = min(deadline, time.time() + 0.035)
                    candidate = try_empty_bin(working, target, slice_end)

            if candidate is not None:
                if len(candidate) < len(best):
                    best = [list(b) for b in candidate]
                if len(candidate) <= len(working):
                    working = [list(b) for b in candidate]
                elif rng.random() < 0.02:
                    working = [list(b) for b in best]

            if len(best) <= volume_bound:
                working = [list(b) for b in best]
            if pair_proven and len(best) == len(pair_solution):
                working = ruin_and_recreate(best, turn)

        item_assignments = [0] * n
        bin_activation = [0] * n
        for b, items in enumerate(best):
            bin_activation[b] = 1
            for item in items:
                item_assignments[item] = b
        return {
            "bin_activation": bin_activation,
            "item_assignments": item_assignments,
        }

    except Exception:
        try:
            n = instance["n"]
            capacity = instance["capacity"]
            weights = instance["weights"]
            order = sorted(range(n), key=lambda i: weights[i], reverse=True)
            fallback_bins = []
            fallback_loads = []
            for item in order:
                placed = False
                for b in range(len(fallback_bins)):
                    if fallback_loads[b] + weights[item] <= capacity:
                        fallback_bins[b].append(item)
                        fallback_loads[b] += weights[item]
                        placed = True
                        break
                if not placed:
                    fallback_bins.append([item])
                    fallback_loads.append(weights[item])
            assignments = [0] * n
            activation = [0] * n
            for b, items in enumerate(fallback_bins):
                activation[b] = 1
                for item in items:
                    assignments[item] = b
            return {"bin_activation": activation, "item_assignments": assignments}
        except Exception:
            return {"bin_activation": [], "item_assignments": []}
