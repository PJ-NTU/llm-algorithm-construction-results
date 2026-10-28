# Library slot 7/10  id=alg_2b202e9faf04
# score=143.6  feasible_rate=1.0
# score_vec=[49.0, 213.0, 136.0, 157.0, 146.0, 198.0, 127.0, 96.0, 170.0, 144.0]

import bisect
import random
import time


def solve(instance: dict, tools, time_limit_s: float) -> dict:
    """Anytime grouping search with exact pairing on the dominant size regime."""
    started = time.time()

    try:
        n = int(instance["n"])
    except Exception:
        n = 0
    if n < 0:
        n = 0

    fallback = {
        "bin_activation": [1] * n,
        "item_assignments": list(range(n)),
    }

    try:
        capacity = instance["capacity"]
        weights = list(instance["weights"])
        if len(weights) != n or capacity <= 0:
            return fallback
        for weight in weights:
            if weight < 0 or weight > capacity:
                return fallback

        try:
            budget = float(time_limit_s)
        except Exception:
            budget = 0.0
        if budget < 0.0:
            budget = 0.0
        margin = min(0.04, max(0.001, budget * 0.01))
        deadline = started + max(0.0, budget - margin)

        seed = n * 1000003 + int(capacity * 97)
        for i, weight in enumerate(weights):
            seed = (seed * 1000003 + (i + 1) * int(weight * 101)) & 0xFFFFFFFF
        rng = random.Random(seed)

        def format_bins(bins):
            clean = [list(group) for group in bins if group]
            assignment = [-1] * n
            for bin_index, group in enumerate(clean):
                load = 0
                for item in group:
                    if item < 0 or item >= n or assignment[item] != -1:
                        return fallback
                    assignment[item] = bin_index
                    load += weights[item]
                if load > capacity:
                    return fallback
            if any(bin_index < 0 for bin_index in assignment):
                return fallback
            return {
                "bin_activation": [1] * len(clean) + [0] * (n - len(clean)),
                "item_assignments": assignment,
            }

        def exact_pairing():
            ordered = sorted(range(n), key=lambda item: (weights[item], item))
            left = 0
            right = n - 1
            result = []
            while left <= right:
                if left == right:
                    result.append([ordered[right]])
                    break
                largest = ordered[right]
                if weights[ordered[left]] + weights[largest] <= capacity:
                    result.append([ordered[left], largest])
                    left += 1
                else:
                    result.append([largest])
                right -= 1
            return result

        def pack_order(order, randomized=False):
            bins = []
            loads = []
            spaces = []
            for item in order:
                weight = weights[item]
                position = bisect.bisect_left(spaces, (weight, -1))
                if position == len(spaces):
                    bin_index = len(bins)
                    bins.append([item])
                    loads.append(weight)
                    bisect.insort(spaces, (capacity - weight, bin_index))
                    continue

                if randomized:
                    stop = min(len(spaces), position + 4)
                    position = rng.randrange(position, stop)
                old_space, bin_index = spaces.pop(position)
                bins[bin_index].append(item)
                loads[bin_index] += weight
                bisect.insort(spaces, (old_space - weight, bin_index))
            return [group for group in bins if group]

        def signature(bins):
            return tuple(sorted(tuple(sorted(group)) for group in bins if group))

        def tie_quality(bins):
            waste_square = 0
            for group in bins:
                load = sum(weights[item] for item in group)
                waste = capacity - load
                waste_square += waste * waste
            return (len(bins), -waste_square)

        def direct_empty(bins, source_index):
            source = sorted(bins[source_index], key=lambda item: weights[item], reverse=True)
            targets = [i for i in range(len(bins)) if i != source_index]
            remaining = [capacity - sum(weights[item] for item in bins[i]) for i in targets]
            chosen = [-1] * len(source)

            def place(position):
                if time.time() >= deadline:
                    return False
                if position == len(source):
                    return True
                item = source[position]
                weight = weights[item]
                candidates = [j for j in range(len(targets)) if remaining[j] >= weight]
                candidates.sort(key=lambda j: remaining[j] - weight)
                seen_spaces = set()
                for candidate in candidates:
                    old_space = remaining[candidate]
                    if old_space in seen_spaces:
                        continue
                    seen_spaces.add(old_space)
                    remaining[candidate] -= weight
                    chosen[position] = candidate
                    if place(position + 1):
                        return True
                    remaining[candidate] = old_space
                return False

            if not place(0):
                return None
            result = [list(group) for group in bins]
            for position, item in enumerate(source):
                result[targets[chosen[position]]].append(item)
            result.pop(source_index)
            return result

        def repartition(items, bin_count):
            ordered = sorted(items, key=lambda item: weights[item], reverse=True)
            loads = [0] * bin_count
            groups = [[] for _ in range(bin_count)]

            suffix = [0] * (len(ordered) + 1)
            for position in range(len(ordered) - 1, -1, -1):
                suffix[position] = suffix[position + 1] + weights[ordered[position]]

            def assign(position):
                if time.time() >= deadline:
                    return False
                if position == len(ordered):
                    return True
                free_total = bin_count * capacity - sum(loads)
                if suffix[position] > free_total:
                    return False
                item = ordered[position]
                weight = weights[item]
                choices = sorted(range(bin_count), key=lambda j: capacity - loads[j] - weight)
                seen_loads = set()
                for target in choices:
                    old_load = loads[target]
                    if old_load in seen_loads or old_load + weight > capacity:
                        continue
                    seen_loads.add(old_load)
                    loads[target] += weight
                    groups[target].append(item)
                    if assign(position + 1):
                        return True
                    groups[target].pop()
                    loads[target] = old_load
                    if old_load == 0:
                        break
                return False

            if assign(0):
                return [group for group in groups if group]
            return None

        def joint_empty(bins, source_index):
            if len(bins) < 3:
                return None
            others = [i for i in range(len(bins)) if i != source_index]
            source_load = sum(weights[item] for item in bins[source_index])
            rng.shuffle(others)
            others.sort(key=lambda i: sum(weights[item] for item in bins[i]))

            # Three old groups are rebuilt into two. This permits ejection chains
            # that one-item migration cannot see.
            for first_pos in range(min(len(others), 12)):
                first = others[first_pos]
                for second_pos in range(first_pos + 1, min(len(others), 14)):
                    if time.time() >= deadline:
                        return None
                    second = others[second_pos]
                    selected = [source_index, first, second]
                    total = source_load
                    total += sum(weights[item] for item in bins[first])
                    total += sum(weights[item] for item in bins[second])
                    if total > 2 * capacity:
                        continue
                    items = []
                    for index in selected:
                        items.extend(bins[index])
                    rebuilt = repartition(items, 2)
                    if rebuilt is not None:
                        selected_set = set(selected)
                        result = [list(group) for i, group in enumerate(bins) if i not in selected_set]
                        result.extend(rebuilt)
                        return result
            return None

        def improve(bins):
            current = [list(group) for group in bins]
            while time.time() < deadline and len(current) > 1:
                ranked = sorted(
                    range(len(current)),
                    key=lambda i: sum(weights[item] for item in current[i]),
                )
                changed = False
                for source in ranked:
                    if time.time() >= deadline:
                        break
                    candidate = direct_empty(current, source)
                    if candidate is None:
                        candidate = joint_empty(current, source)
                    if candidate is not None and len(candidate) < len(current):
                        current = candidate
                        changed = True
                        break
                if not changed:
                    break
            return current

        def crossover(first, second):
            child = []
            used = set()
            sources = [first, second]
            start = rng.randrange(2)
            target_fraction = 0.45 + 0.35 * rng.random()

            # Alternate donors so intact high-quality groups from both parents
            # coexist in one child before the uncovered items are repaired.
            turn = 0
            while len(used) < n * target_fraction:
                donor = sources[(start + turn) % 2]
                available = [
                    group for group in donor
                    if group and all(item not in used for item in group)
                ]
                if not available:
                    other = sources[(start + turn + 1) % 2]
                    available = [
                        group for group in other
                        if group and all(item not in used for item in group)
                    ]
                    if not available:
                        break
                available.sort(
                    key=lambda group: sum(weights[item] for item in group)
                    + rng.random() * capacity * 0.08,
                    reverse=True,
                )
                selected = available[0]
                child.append(list(selected))
                used.update(selected)
                turn += 1

            remaining_items = [item for item in range(n) if item not in used]
            remaining_items.sort(
                key=lambda item: weights[item] * (0.92 + 0.16 * rng.random()),
                reverse=True,
            )
            loads = [sum(weights[item] for item in group) for group in child]
            for item in remaining_items:
                weight = weights[item]
                feasible = [i for i in range(len(child)) if loads[i] + weight <= capacity]
                if feasible:
                    target = min(feasible, key=lambda i: capacity - loads[i] - weight)
                    child[target].append(item)
                    loads[target] += weight
                else:
                    child.append([item])
                    loads.append(weight)
            return child

        def perturb(bins):
            if len(bins) < 2:
                return [list(group) for group in bins]
            result = [list(group) for group in bins]
            destroy_count = 1 + rng.randrange(min(3, len(result)))
            chosen = sorted(rng.sample(range(len(result)), destroy_count), reverse=True)
            loose = []
            for index in chosen:
                loose.extend(result.pop(index))
            loose.sort(
                key=lambda item: weights[item] * (0.85 + 0.30 * rng.random()),
                reverse=True,
            )
            loads = [sum(weights[item] for item in group) for group in result]
            for item in loose:
                feasible = [i for i in range(len(result)) if loads[i] + weights[item] <= capacity]
                if feasible:
                    feasible.sort(key=lambda i: capacity - loads[i] - weights[item])
                    target = feasible[rng.randrange(min(3, len(feasible)))]
                    result[target].append(item)
                    loads[target] += weights[item]
                else:
                    result.append([item])
                    loads.append(weights[item])
            return result

        descending = sorted(range(n), key=lambda item: (weights[item], item), reverse=True)
        initial = pack_order(descending)

        # When every bin can contain at most two items, this greedy matching is
        # exact: pair the largest feasible item with the smallest remaining one.
        if n == 0:
            best = []
        elif min(weights) * 3 > capacity:
            best = exact_pairing()
        else:
            best = improve(initial)

        population = [[list(group) for group in best], [list(group) for group in initial]]
        seen = {signature(best), signature(initial)}

        while time.time() < deadline:
            jittered = sorted(
                range(n),
                key=lambda item: weights[item] * (0.82 + 0.36 * rng.random()),
                reverse=True,
            )
            restart = pack_order(jittered, randomized=True)
            parent_one = population[rng.randrange(len(population))]
            parent_two = population[rng.randrange(len(population))]
            child = crossover(parent_one, parent_two)
            if rng.random() < 0.75:
                child = perturb(child)
            if time.time() < deadline:
                child = improve(child)

            candidates = [restart, child]
            for candidate in candidates:
                candidate_signature = signature(candidate)
                if candidate_signature not in seen:
                    seen.add(candidate_signature)
                    population.append([list(group) for group in candidate])
                if tie_quality(candidate) < tie_quality(best):
                    best = [list(group) for group in candidate]

            population.sort(key=tie_quality)
            if len(population) > 10:
                population = population[:10]

        return format_bins(best)
    except Exception:
        return fallback
