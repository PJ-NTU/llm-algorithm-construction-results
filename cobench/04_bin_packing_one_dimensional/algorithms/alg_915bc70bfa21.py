# Library slot 8/10  id=alg_915bc70bfa21
# score=143.7  feasible_rate=1.0
# score_vec=[49.0, 213.0, 137.0, 156.0, 146.0, 199.0, 127.0, 97.0, 171.0, 142.0]

"""Anytime heuristic for one-dimensional bin packing."""

import bisect
import math
import random
import time


def _as_solution(bins, n):
    bins = [members for members in bins if members]
    assignment = [0] * n
    for bin_index, members in enumerate(bins):
        for item_index in members:
            assignment[item_index] = bin_index
    return {
        "bin_activation": [1] * len(bins) + [0] * (n - len(bins)),
        "item_assignments": assignment,
    }


def _best_fit(order, weights, capacity):
    bins = []
    spaces = []
    for item_index in order:
        weight = weights[item_index]
        position = bisect.bisect_left(spaces, (weight, -1))
        if position == len(spaces):
            bin_index = len(bins)
            bins.append([item_index])
            bisect.insort(spaces, (capacity - weight, bin_index))
        else:
            space, bin_index = spaces.pop(position)
            bins[bin_index].append(item_index)
            bisect.insort(spaces, (space - weight, bin_index))
    return bins


def _optimal_pairs(weights, capacity):
    """Optimal packing when no bin can contain three items."""
    ordered = sorted(range(len(weights)), key=lambda i: (weights[i], i))
    bins = []
    low = 0
    high = len(ordered) - 1
    while low < high:
        small = ordered[low]
        large = ordered[high]
        if weights[small] + weights[large] <= capacity:
            bins.append([large, small])
            low += 1
            high -= 1
        else:
            bins.append([large])
            high -= 1
    if low == high:
        bins.append([ordered[low]])
    return bins


def _small_exact(weights, capacity, incumbent, deadline):
    n = len(weights)
    order = sorted(range(n), key=lambda i: (-weights[i], i))
    suffix = [0] * (n + 1)
    for position in range(n - 1, -1, -1):
        suffix[position] = suffix[position + 1] + weights[order[position]]

    best = [members[:] for members in incumbent]
    lower_bound = max(
        int(math.ceil(sum(weights) / float(capacity))),
        sum(1 for weight in weights if 2 * weight > capacity),
    )
    if len(best) == lower_bound:
        return best

    loads = []
    bins = []

    def search(position):
        nonlocal best
        if time.time() >= deadline:
            return
        if position == n:
            best = [members[:] for members in bins]
            return
        if len(bins) >= len(best):
            return

        free_space = sum(capacity - load for load in loads)
        overflow = max(0, suffix[position] - free_space)
        extra_bins = int(math.ceil(overflow / float(capacity)))
        if len(bins) + extra_bins >= len(best):
            return

        item_index = order[position]
        weight = weights[item_index]
        choices = []
        seen_loads = set()
        for bin_index, load in enumerate(loads):
            if load + weight <= capacity and load not in seen_loads:
                seen_loads.add(load)
                choices.append((capacity - load - weight, bin_index))
        choices.sort()

        for _, bin_index in choices:
            loads[bin_index] += weight
            bins[bin_index].append(item_index)
            search(position + 1)
            bins[bin_index].pop()
            loads[bin_index] -= weight
            if len(best) == lower_bound or time.time() >= deadline:
                return

        if len(bins) + 1 < len(best):
            loads.append(weight)
            bins.append([item_index])
            search(position + 1)
            bins.pop()
            loads.pop()

    search(0)
    return best


def _minimum_slack(weights, capacity, deadline):
    """Build full patterns by solving a small knapsack for each new bin."""
    remaining = set(range(len(weights)))
    bins = []
    while remaining:
        if time.time() >= deadline:
            return None
        anchor = max(remaining, key=lambda i: (weights[i], -i))
        remaining.remove(anchor)
        room = capacity - weights[anchor]
        if room <= 0 or not remaining:
            bins.append([anchor])
            continue

        states = {0: ()}
        candidates = sorted(remaining, key=lambda i: (-weights[i], i))
        for item_index in candidates:
            weight = weights[item_index]
            additions = {}
            for packed_weight, chosen in tuple(states.items()):
                new_weight = packed_weight + weight
                if new_weight <= room and new_weight not in states:
                    additions[new_weight] = chosen + (item_index,)
            states.update(additions)
            if room in states or time.time() >= deadline:
                break
            if len(states) > 5000:
                keep = sorted(states, reverse=True)[:3500]
                keep.append(0)
                states = {value: states[value] for value in set(keep)}

        chosen = states[max(states)]
        for item_index in chosen:
            remaining.remove(item_index)
        bins.append([anchor] + list(chosen))
    return bins


def _remove_one(bins, weights, capacity, target, deadline):
    moving = sorted(bins[target], key=lambda i: (-weights[i], i))
    kept = [members[:] for index, members in enumerate(bins) if index != target]
    loads = [sum(weights[i] for i in members) for members in kept]

    def place(position):
        if time.time() >= deadline:
            return False
        if position == len(moving):
            return True
        item_index = moving[position]
        weight = weights[item_index]
        choices = []
        seen_loads = set()
        for bin_index, load in enumerate(loads):
            if load + weight <= capacity and load not in seen_loads:
                seen_loads.add(load)
                choices.append((capacity - load - weight, bin_index))
        choices.sort()
        for _, bin_index in choices:
            loads[bin_index] += weight
            kept[bin_index].append(item_index)
            if place(position + 1):
                return True
            kept[bin_index].pop()
            loads[bin_index] -= weight
        return False

    return kept if place(0) else None


def _pack_fixed(item_indices, weights, capacity, bin_count, deadline, rng):
    if bin_count <= 0:
        return [] if not item_indices else None
    if sum(weights[i] for i in item_indices) > bin_count * capacity:
        return None

    tie_keys = {item_index: rng.random() for item_index in item_indices}
    order = sorted(item_indices, key=lambda i: (-weights[i], tie_keys[i]))
    loads = [0] * bin_count
    bins = [[] for _ in range(bin_count)]

    def place(position):
        if time.time() >= deadline:
            return False
        if position == len(order):
            return True
        item_index = order[position]
        weight = weights[item_index]
        choices = []
        seen_loads = set()
        for bin_index, load in enumerate(loads):
            if load + weight <= capacity and load not in seen_loads:
                seen_loads.add(load)
                choices.append((capacity - load - weight, rng.random(), bin_index))
        choices.sort()
        for _, _, bin_index in choices:
            loads[bin_index] += weight
            bins[bin_index].append(item_index)
            if place(position + 1):
                return True
            bins[bin_index].pop()
            loads[bin_index] -= weight
        return False

    if place(0):
        return [members for members in bins if members]
    return None


def _joint_repack(bins, weights, capacity, selected, deadline, rng):
    selected_set = set(selected)
    untouched = [
        members[:] for index, members in enumerate(bins)
        if index not in selected_set
    ]
    items = []
    for index in selected:
        items.extend(bins[index])
    replacement = _pack_fixed(
        items, weights, capacity, len(selected) - 1, deadline, rng
    )
    if replacement is None:
        return None
    return untouched + replacement


def solve(instance: dict, tools, time_limit_s: float) -> dict:
    """Return a complete feasible assignment before the wall-clock deadline."""
    started = time.time()
    safe_n = 0
    fallback = {"bin_activation": [], "item_assignments": []}
    try:
        safe_n = int(instance.get("n", 0))
        capacity = int(instance.get("capacity", 0))
        weights = [int(value) for value in instance.get("weights", [])]
        if safe_n < 0 or len(weights) != safe_n or capacity <= 0:
            return fallback

        fallback = {
            "bin_activation": [1] * safe_n,
            "item_assignments": list(range(safe_n)),
        }
        if safe_n == 0:
            return fallback
        if any(weight < 0 or weight > capacity for weight in weights):
            return fallback

        try:
            budget = max(0.0, float(time_limit_s))
        except Exception:
            budget = 0.0
        margin = min(0.05, max(0.002, 0.03 * budget))
        deadline = started + max(0.0, budget - margin)

        ascending = sorted(weights)
        if safe_n < 3 or sum(ascending[:3]) > capacity:
            return _as_solution(_optimal_pairs(weights, capacity), safe_n)

        descending_order = sorted(
            range(safe_n), key=lambda i: (-weights[i], i)
        )
        best = _best_fit(descending_order, weights, capacity)
        total_lower_bound = int(math.ceil(sum(weights) / float(capacity)))
        if len(best) == total_lower_bound:
            return _as_solution(best, safe_n)

        if safe_n <= 32 and time.time() < deadline:
            best = _small_exact(weights, capacity, best, deadline)
            if len(best) == total_lower_bound:
                return _as_solution(best, safe_n)

        if time.time() < deadline:
            construction_deadline = min(
                deadline, started + max(0.0, budget * 0.18)
            )
            candidate = _minimum_slack(weights, capacity, construction_deadline)
            if candidate is not None and len(candidate) < len(best):
                best = candidate
            if len(best) == total_lower_bound:
                return _as_solution(best, safe_n)

        seed = 1469598103934665603
        for item_index, weight in enumerate(weights):
            seed ^= (item_index + 1) * (weight + 1000003)
            seed = (seed * 1099511628211) & ((1 << 64) - 1)
        rng = random.Random(seed)
        attempt = 0

        while time.time() < deadline:
            attempt += 1
            loads = [sum(weights[i] for i in members) for members in best]
            if attempt % 3 == 1:
                target = min(range(len(best)), key=lambda b: (loads[b], b))
            else:
                target = rng.randrange(len(best))

            slice_end = min(deadline, time.time() + 0.018)
            improved = _remove_one(
                best, weights, capacity, target, slice_end
            )
            if improved is not None:
                best = improved
                if len(best) == total_lower_bound:
                    break
                continue

            if len(best) < 3 or time.time() >= deadline:
                continue
            group_size = min(len(best), 3 + rng.randrange(min(7, len(best) - 2)))
            if attempt % 4 == 0:
                selected = sorted(
                    range(len(best)), key=lambda b: (loads[b], b)
                )[:group_size]
            else:
                selected = rng.sample(range(len(best)), group_size)
            slice_end = min(deadline, time.time() + 0.035)
            improved = _joint_repack(
                best, weights, capacity, selected, slice_end, rng
            )
            if improved is not None:
                best = improved
                if len(best) == total_lower_bound:
                    break

        return _as_solution(best, safe_n)
    except Exception:
        return fallback
