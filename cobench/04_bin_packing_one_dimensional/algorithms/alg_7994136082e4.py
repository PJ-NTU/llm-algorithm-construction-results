# Library slot 6/10  id=alg_7994136082e4
# score=143.4  feasible_rate=1.0
# score_vec=[49.0, 213.0, 136.0, 156.0, 146.0, 199.0, 127.0, 96.0, 170.0, 142.0]

"""Anytime solver for one-dimensional bin packing."""

import bisect
import math
import random
import time


def _best_fit_decreasing(weights, capacity):
    """Return bins of item indices using best-fit decreasing."""
    bins = []
    spaces = []
    for item in sorted(range(len(weights)), key=lambda i: (-weights[i], i)):
        weight = weights[item]
        pos = bisect.bisect_left(spaces, (weight, -1))
        if pos == len(spaces):
            bin_id = len(bins)
            bins.append([item])
            bisect.insort(spaces, (capacity - weight, bin_id))
        else:
            remaining, bin_id = spaces.pop(pos)
            bins[bin_id].append(item)
            bisect.insort(spaces, (remaining - weight, bin_id))
    return bins


def _best_fit_order(weights, capacity, order):
    bins = []
    spaces = []
    for item in order:
        weight = weights[item]
        pos = bisect.bisect_left(spaces, (weight, -1))
        if pos == len(spaces):
            bin_id = len(bins)
            bins.append([item])
            bisect.insort(spaces, (capacity - weight, bin_id))
        else:
            remaining, bin_id = spaces.pop(pos)
            bins[bin_id].append(item)
            bisect.insort(spaces, (remaining - weight, bin_id))
    return bins


def _subset_fill(weights, capacity, deadline):
    """Repeatedly fill a bin exactly or nearly exactly by subset-sum DP."""
    remaining = sorted(range(len(weights)), key=lambda i: (-weights[i], i))
    bins = []
    while remaining and time.time() < deadline:
        anchor = remaining.pop(0)
        room = capacity - weights[anchor]
        if room <= 0:
            bins.append([anchor])
            continue

        # The states are actual reachable sums.  Keeping parents makes the
        # selected subset recoverable without a second pass.
        parents = {0: None}
        for item in remaining:
            weight = weights[item]
            if weight > room:
                continue
            for old_sum in sorted(tuple(parents), reverse=True):
                new_sum = old_sum + weight
                if new_sum <= room and new_sum not in parents:
                    parents[new_sum] = (old_sum, item)
            if room in parents or time.time() >= deadline:
                break

        fill = max(parents)
        chosen = []
        while fill:
            previous, item = parents[fill]
            chosen.append(item)
            fill = previous
        if chosen:
            chosen_set = set(chosen)
            remaining = [item for item in remaining if item not in chosen_set]
        bins.append([anchor] + chosen)

    # A deadline can interrupt the DP between bins; best-fit completes the
    # partial construction while preserving every already packed item.
    if remaining:
        spaces = sorted(
            (capacity - sum(weights[item] for item in items), bin_id)
            for bin_id, items in enumerate(bins)
        )
        for item in remaining:
            weight = weights[item]
            pos = bisect.bisect_left(spaces, (weight, -1))
            if pos == len(spaces):
                bin_id = len(bins)
                bins.append([item])
                bisect.insort(spaces, (capacity - weight, bin_id))
            else:
                space, bin_id = spaces.pop(pos)
                bins[bin_id].append(item)
                bisect.insort(spaces, (space - weight, bin_id))
    return bins


def _pairing_optimum(weights, capacity):
    """Exact packing when no three items can share a bin."""
    ordered = sorted(range(len(weights)), key=lambda i: (weights[i], i))
    left = 0
    right = len(ordered) - 1
    bins = []
    while left <= right:
        heavy = ordered[right]
        right -= 1
        if left <= right and weights[ordered[left]] + weights[heavy] <= capacity:
            bins.append([heavy, ordered[left]])
            left += 1
        else:
            bins.append([heavy])
    return bins


def _lower_bound(weights, capacity):
    if not weights:
        return 0
    if capacity <= 0:
        return len(weights)
    bound = int(math.ceil(sum(weights) / float(capacity)))

    # For a threshold a, an item heavier than C-a cannot share a bin with
    # any item of weight at least a.  Those two groups therefore give a
    # valid additive lower bound.  This is much sharper than volume alone
    # on instances dominated by large items.
    ordered = sorted(weights)
    prefix = [0]
    for weight in ordered:
        prefix.append(prefix[-1] + weight)
    thresholds = {weight for weight in ordered if 0 < weight <= capacity // 2}
    for threshold in thresholds:
        upper = capacity - threshold
        middle_start = bisect.bisect_left(ordered, threshold)
        middle_end = bisect.bisect_right(ordered, upper)
        very_large = len(ordered) - middle_end
        middle_weight = prefix[middle_end] - prefix[middle_start]
        candidate = very_large + int(math.ceil(middle_weight / float(capacity)))
        if candidate > bound:
            bound = candidate
    return bound


def _balanced_greedy(weights, bin_count, order):
    """LPT-style construction for a prescribed number of bins."""
    if bin_count <= 0:
        return [] if not order else None
    bins = [[] for _ in range(bin_count)]
    heap = [(0, bin_id) for bin_id in range(bin_count)]
    for item in order:
        load, bin_id = heap[0]
        bins[bin_id].append(item)
        heap[0] = (load + weights[item], bin_id)
        # Replacing the root and restoring the heap is quicker than finding
        # the least-loaded bin by a full scan.
        _sift_down(heap, 0)
    return bins


def _sift_down(heap, pos):
    size = len(heap)
    value = heap[pos]
    child = pos * 2 + 1
    while child < size:
        right = child + 1
        if right < size and heap[right] < heap[child]:
            child = right
        if value <= heap[child]:
            break
        heap[pos] = heap[child]
        pos = child
        child = pos * 2 + 1
    heap[pos] = value


def _loads(bins, weights):
    return [sum(weights[item] for item in items) for items in bins]


def _pack_to_k(items, bin_count, weights, capacity, deadline, rng):
    """Find a packing into exactly bin_count bins by bounded DFS."""
    if bin_count < 0 or sum(weights[item] for item in items) > bin_count * capacity:
        return None
    ordered = sorted(items, key=lambda i: (-weights[i], rng.random()))
    greedy = _balanced_greedy(weights, bin_count, ordered)
    if greedy is not None and max(_loads(greedy, weights), default=0) <= capacity:
        return [items_in_bin for items_in_bin in greedy if items_in_bin]
    # Python's call stack is deliberately left untouched (the sandbox bans
    # sys imports).  Large neighborhoods use the iterative overload search.
    if len(ordered) > 800:
        return None

    spaces = [capacity] * bin_count
    placement = [-1] * len(ordered)
    failed = set()
    memo_limit = max(2000, min(100000, 5000000 // max(1, bin_count)))

    def visit(pos):
        if time.time() >= deadline:
            return False
        if pos == len(ordered):
            return True
        state = (pos, tuple(sorted(spaces)))
        if state in failed:
            return False
        item = ordered[pos]
        weight = weights[item]
        choices = sorted(
            (space - weight, bin_id)
            for bin_id, space in enumerate(spaces)
            if space >= weight
        )
        seen_spaces = set()
        for unused, bin_id in choices:
            old_space = spaces[bin_id]
            if old_space in seen_spaces:
                continue
            seen_spaces.add(old_space)
            spaces[bin_id] = unused
            placement[pos] = bin_id
            if visit(pos + 1):
                return True
            spaces[bin_id] = old_space
            if old_space == capacity:
                break
        if len(failed) < memo_limit:
            failed.add(state)
        return False

    if not visit(0):
        return None
    bins = [[] for _ in range(bin_count)]
    for pos, item in enumerate(ordered):
        bins[placement[pos]].append(item)
    return [items_in_bin for items_in_bin in bins if items_in_bin]


def _overload_search(weights, capacity, bin_count, deadline, rng):
    """Minimize capacity overflow for a fixed bin count by moves/swaps."""
    item_ids = list(range(len(weights)))
    while time.time() < deadline:
        noise = rng.random() * 0.35 * capacity
        order = sorted(item_ids, key=lambda i: (-(weights[i] + rng.random() * noise), i))
        bins = _balanced_greedy(weights, bin_count, order)
        loads = _loads(bins, weights)
        if max(loads, default=0) <= capacity:
            return [items for items in bins if items]

        # Descend in total squared overflow.  Random restarts and perturbing
        # moves supply diversification when all improving moves are blocked.
        stagnation = 0
        while time.time() < deadline:
            overloaded = [b for b in range(bin_count) if loads[b] > capacity]
            if not overloaded:
                return [items for items in bins if items]
            old_cost = sum(max(0, load - capacity) ** 2 for load in loads)
            best_delta = 0
            best_move = None

            for source in overloaded:
                for item in tuple(bins[source]):
                    weight = weights[item]
                    for target in range(bin_count):
                        if target == source:
                            continue
                        new_source = loads[source] - weight
                        new_target = loads[target] + weight
                        delta = (
                            max(0, new_source - capacity) ** 2
                            + max(0, new_target - capacity) ** 2
                            - max(0, loads[source] - capacity) ** 2
                            - max(0, loads[target] - capacity) ** 2
                        )
                        if delta < best_delta:
                            best_delta = delta
                            best_move = (source, target, item, None)
                        # A smaller item can be ejected from the target,
                        # transferring only the useful weight difference.
                        for other in bins[target]:
                            other_weight = weights[other]
                            if other_weight >= weight:
                                continue
                            swapped_source = new_source + other_weight
                            swapped_target = new_target - other_weight
                            swap_delta = (
                                max(0, swapped_source - capacity) ** 2
                                + max(0, swapped_target - capacity) ** 2
                                - max(0, loads[source] - capacity) ** 2
                                - max(0, loads[target] - capacity) ** 2
                            )
                            if swap_delta < best_delta:
                                best_delta = swap_delta
                                best_move = (source, target, item, other)
                        if time.time() >= deadline:
                            return None

            if best_move is None:
                stagnation += 1
                source = max(overloaded, key=lambda b: loads[b])
                item = rng.choice(bins[source])
                targets = [b for b in range(bin_count) if b != source]
                if not targets:
                    break
                target = rng.choice(targets)
                bins[source].remove(item)
                bins[target].append(item)
                loads[source] -= weights[item]
                loads[target] += weights[item]
                if stagnation >= 4:
                    break
            else:
                stagnation = 0
                source, target, item, other = best_move
                bins[source].remove(item)
                bins[target].append(item)
                loads[source] -= weights[item]
                loads[target] += weights[item]
                if other is not None:
                    bins[target].remove(other)
                    bins[source].append(other)
                    loads[target] -= weights[other]
                    loads[source] += weights[other]
                if old_cost + best_delta <= 0:
                    return [items for items in bins if items]
    return None


def _random_order(weights, capacity, rng):
    scale = rng.random() * 0.8 * capacity
    return sorted(
        range(len(weights)),
        key=lambda i: (-(weights[i] + rng.random() * scale), rng.random()),
    )


def _local_repack(bins, weights, capacity, deadline, rng):
    if len(bins) < 2:
        return None
    loads = _loads(bins, weights)
    ranked = sorted(range(len(bins)), key=lambda b: (loads[b], b))
    max_group = min(len(bins), 14)
    group_size = rng.randint(2, max_group)
    if rng.random() < 0.65:
        selected = ranked[:group_size]
    else:
        pool = ranked[:min(len(ranked), max(group_size * 3, group_size))]
        selected = [ranked[0]] + rng.sample(pool[1:], group_size - 1)
    selected_items = [item for b in selected for item in bins[b]]
    if sum(weights[item] for item in selected_items) > (group_size - 1) * capacity:
        return None
    repacked = _pack_to_k(
        selected_items,
        group_size - 1,
        weights,
        capacity,
        deadline,
        rng,
    )
    if repacked is None:
        return None
    selected_set = set(selected)
    untouched = [items for b, items in enumerate(bins) if b not in selected_set]
    return untouched + repacked


def _as_solution(n, bins):
    assignments = [0] * n
    for bin_id, items in enumerate(bins):
        for item in items:
            assignments[item] = bin_id
    return {
        "bin_activation": [1] * len(bins) + [0] * (n - len(bins)),
        "item_assignments": assignments,
    }


def solve(instance: dict, time_limit_s: float) -> dict:
    t0 = time.time()
    try:
        n = max(0, int(instance.get("n", 0)))
        capacity = int(instance.get("capacity", 0))
        raw_weights = instance.get("weights", [])
        weights = [int(raw_weights[i]) for i in range(n)]
    except Exception:
        return {"bin_activation": [], "item_assignments": []}

    best = _best_fit_decreasing(weights, capacity)
    if n <= 1:
        return _as_solution(n, best)

    budget = max(0.0, float(time_limit_s))
    deadline = t0 + budget * 0.85
    if time.time() >= deadline:
        return _as_solution(n, best)

    # This condition proves that every bin contains at most two items.  The
    # two-pointer maximum matching is then globally optimal.
    three_smallest = sum(sorted(weights)[:3]) if n >= 3 else capacity + 1
    if three_smallest > capacity:
        return _as_solution(n, _pairing_optimum(weights, capacity))

    # Exact subset fill is particularly effective for the small integral
    # capacities common in benchmark families, but its state space is kept
    # bounded for unusually large capacities.
    if capacity <= 10000:
        candidate = _subset_fill(weights, capacity, min(deadline, time.time() + budget * 0.12))
        if len(candidate) < len(best):
            best = candidate

    lower_bound = _lower_bound(weights, capacity)
    if len(best) <= lower_bound or time.time() >= deadline:
        return _as_solution(n, best)

    seed = n * 1000003 + capacity * 9176
    for index, weight in enumerate(weights):
        seed = (seed * 1000003 + (index + 1) * weight) & 0xFFFFFFFFFFFFFFFF
    rng = random.Random(seed)

    # A short exact attempt is valuable on small and tightly constrained
    # instances.  The separate deadline prevents a hard proof from consuming
    # the full anytime budget before diversified search gets a chance.
    exact_slice = min(deadline, time.time() + max(0.01, budget * 0.16))
    exact = _pack_to_k(list(range(n)), len(best) - 1, weights, capacity, exact_slice, rng)
    if exact is not None and len(exact) < len(best):
        best = exact
        if len(best) <= lower_bound:
            return _as_solution(n, best)

    # All remaining loops are stopped only by wall clock.  They combine
    # order-space GRASP, fixed-k overload repair, and exact local repacking.
    while time.time() < deadline:
        choice = rng.random()
        if choice < 0.34:
            candidate = _best_fit_order(weights, capacity, _random_order(weights, capacity, rng))
        elif choice < 0.72:
            slice_deadline = min(deadline, time.time() + max(0.01, budget * 0.035))
            candidate = _overload_search(
                weights, capacity, len(best) - 1, slice_deadline, rng
            )
        else:
            slice_deadline = min(deadline, time.time() + max(0.01, budget * 0.025))
            candidate = _local_repack(best, weights, capacity, slice_deadline, rng)
        if candidate is not None and len(candidate) < len(best):
            best = candidate
            if len(best) <= lower_bound:
                break

    return _as_solution(n, best)
