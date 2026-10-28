# Library slot 3/10  id=alg_1ef670a6d999
# score=142.7  feasible_rate=1.0
# score_vec=[49.0, 211.0, 135.0, 155.0, 145.0, 199.0, 126.0, 97.0, 169.0, 141.0]

"""Anytime heuristic for one-dimensional bin packing.

The search combines minimum-bin-slack constructions with exact, bounded-time
repacking of small groups of bins.  A separate cardinality-three construction
is useful on instances just outside the easy pairing regime.
"""

import bisect
import math
import random
import time


def solve(instance: dict, time_limit_s: float) -> dict:
    t0 = time.time()

    try:
        n = int(instance["n"])
    except Exception:
        n = 0
    n = max(0, n)
    empty_answer = {
        "bin_activation": [0] * n,
        "item_assignments": [0] * n,
    }
    if n == 0:
        return empty_answer

    # A singleton packing is installed before any nontrivial work.  For every
    # valid instance it is a complete feasible answer.
    fallback = {
        "bin_activation": [1] * n,
        "item_assignments": list(range(n)),
    }
    try:
        capacity = int(instance["capacity"])
        raw_weights = instance["weights"]
        weights = [int(raw_weights[i]) for i in range(n)]
        budget = max(0.0, float(time_limit_s))
    except Exception:
        return fallback
    if not math.isfinite(budget):
        budget = 0.0
    if capacity <= 0 or len(raw_weights) != n:
        return fallback
    if any(weight < 0 or weight > capacity for weight in weights):
        return fallback

    deadline = t0 + budget * 0.85
    positive_items = [i for i in range(n) if weights[i] > 0]
    zero_items = [i for i in range(n) if weights[i] == 0]
    if not positive_items:
        return empty_answer

    seed = (n * 1000003 + capacity * 9176 + sum(weights) * 37) & 0xFFFFFFFFFFFFFFFF
    for i, weight in enumerate(weights):
        seed = (seed * 6364136223846793005 + (i + 1) * (weight + 17)) & 0xFFFFFFFFFFFFFFFF
    rng = random.Random(seed)

    def format_bins(groups):
        clean = [list(group) for group in groups if group]
        if not clean:
            return fallback
        seen = [False] * n
        assignments = [0] * n
        for bin_index, group in enumerate(clean):
            load = 0
            for item in group:
                if item < 0 or item >= n or seen[item]:
                    return fallback
                seen[item] = True
                assignments[item] = bin_index
                load += weights[item]
            if load > capacity:
                return fallback
        if not all(seen):
            return fallback
        return {
            "bin_activation": [1] * len(clean) + [0] * (n - len(clean)),
            "item_assignments": assignments,
        }

    def attach_zeros(groups):
        result = [list(group) for group in groups if group]
        if zero_items:
            if result:
                result[0].extend(zero_items)
            else:
                result.append(list(zero_items))
        return result

    descending = sorted(positive_items, key=lambda i: (-weights[i], i))

    def best_fit(order):
        """Best fit in O(n log n), retaining the actual bin identities."""
        groups = []
        spaces = []
        for item in order:
            weight = weights[item]
            position = bisect.bisect_left(spaces, (weight, -1))
            if position == len(spaces):
                bin_index = len(groups)
                groups.append([item])
                bisect.insort(spaces, (capacity - weight, bin_index))
            else:
                old_space, bin_index = spaces.pop(position)
                groups[bin_index].append(item)
                bisect.insort(spaces, (old_space - weight, bin_index))
        return [group for group in groups if group]

    def loads_of(groups):
        return [sum(weights[item] for item in group) for group in groups]

    def quality(groups):
        # With the bin count tied, concentrating the unavoidable waste makes a
        # later whole-bin elimination substantially easier.
        return sum(load * load for load in loads_of(groups))

    best = best_fit(descending)
    best_quality = quality(best)

    total_weight = sum(weights)
    lower_bound = max(
        (total_weight + capacity - 1) // capacity,
        sum(1 for weight in weights if 2 * weight > capacity),
    )

    # If no bin can contain three positive items, bin packing is maximum
    # threshold matching.  The two-pointer rule is exact.
    minimum_positive = min(weights[item] for item in positive_items)
    if 3 * minimum_positive > capacity:
        ordered = sorted(positive_items, key=lambda i: (weights[i], i))
        paired = []
        left = 0
        right = len(ordered) - 1
        while left <= right:
            if left == right:
                paired.append([ordered[right]])
                break
            heavy = ordered[right]
            if weights[ordered[left]] + weights[heavy] <= capacity:
                paired.append([heavy, ordered[left]])
                left += 1
            else:
                paired.append([heavy])
            right -= 1
        return format_bins(attach_zeros(paired))

    common = capacity
    for item in positive_items:
        common = math.gcd(common, weights[item])
    common = max(1, common)

    def subset_for_room(candidates, room, cutoff, randomized):
        """Return a high-fill subset, exactly when the scaled DP is affordable."""
        if room <= 0 or not candidates:
            return []
        ordered = list(candidates)
        if randomized:
            rng.shuffle(ordered)
        else:
            ordered.sort(key=lambda i: (-weights[i], i))

        scaled_room = room // common
        if scaled_room <= 50000:
            mask = (1 << (scaled_room + 1)) - 1
            bits = 1
            history = []
            for position, item in enumerate(ordered):
                if (position & 15) == 0 and time.time() >= cutoff:
                    break
                value = weights[item] // common
                if value <= scaled_room:
                    before = bits
                    bits = (bits | (bits << value)) & mask
                    history.append((item, value, before))
                    if (bits >> scaled_room) & 1:
                        break
            target = bits.bit_length() - 1
            chosen = []
            for item, value, before in reversed(history):
                if target >= value and ((before >> target) & 1) == 0:
                    chosen.append(item)
                    target -= value
            return chosen

        # Huge, poorly scaled capacities make dense DP unattractive.  A small
        # Pareto frontier still beats a plain greedy fill without risking RAM.
        states = {0: ()}
        for position, item in enumerate(ordered):
            if (position & 7) == 0 and time.time() >= cutoff:
                break
            value = weights[item]
            additions = {}
            for used, chosen in tuple(states.items()):
                new_used = used + value
                if new_used <= room and new_used not in states and new_used not in additions:
                    additions[new_used] = chosen + (item,)
            states.update(additions)
            if room in states:
                break
            if len(states) > 6000:
                keep = sorted(states, reverse=True)[:4500]
                keep.append(0)
                states = {value: states[value] for value in set(keep)}
        return list(states[max(states)])

    def completion_pack(cutoff, randomized):
        """MBS': force an anchor, then solve its residual subset sum."""
        remaining = list(descending)
        result = []
        turn = 0
        while remaining:
            if time.time() >= cutoff:
                result.extend(best_fit(remaining))
                break
            anchor_position = 0
            if randomized and len(remaining) > 3 and turn % 5 == 4:
                anchor_position = rng.randrange(min(4, len(remaining)))
            anchor = remaining[anchor_position]
            room = capacity - weights[anchor]
            candidates = [item for item in remaining if item != anchor and weights[item] <= room]
            chosen = subset_for_room(candidates, room, cutoff, randomized)
            selected = set(chosen)
            selected.add(anchor)
            result.append([anchor] + chosen)
            remaining = [item for item in remaining if item not in selected]
            turn += 1
        return result

    def optimal_pairs(items):
        ordered = sorted(items, key=lambda i: (weights[i], i))
        result = []
        left = 0
        right = len(ordered) - 1
        while left <= right:
            if left == right:
                result.append([ordered[right]])
                break
            heavy = ordered[right]
            if weights[ordered[left]] + weights[heavy] <= capacity:
                result.append([heavy, ordered[left]])
                left += 1
            else:
                result.append([heavy])
            right -= 1
        return result

    def cardinality_three_pack(cutoff):
        """Build triples explicitly, then solve the residual pairing optimally."""
        # When every pair fits, the objective is driven entirely by how many
        # triples can be formed.  Reserve the lightest 3t items for triples and
        # balance each largest reserved item with a small and a tight-fitting
        # companion.  This avoids MBS' tendency to spend two scarce small items
        # on one very heavy anchor.
        if 2 * max(weights[item] for item in positive_items) <= capacity:
            ranked_all = sorted(positive_items, key=lambda i: (weights[i], i))
            prefix = [0]
            for item in ranked_all:
                prefix.append(prefix[-1] + weights[item])
            target = len(ranked_all) // 3
            while target > 0 and prefix[3 * target] > target * capacity:
                target -= 1
            while target > 0 and time.time() < cutoff:
                target_cutoff = min(cutoff, time.time() + max(0.001, (cutoff - time.time()) * 0.20))
                attempt = 0
                while time.time() < target_cutoff:
                    trial_rng = random.Random(attempt)
                    selected = [(weights[item], item) for item in ranked_all[:3 * target]]
                    groups = []
                    possible = True
                    while selected and time.time() < target_cutoff:
                        heavy_weight, heavy = selected.pop()
                        small_position = 0
                        if attempt and len(selected) > 6 and trial_rng.random() < 0.70:
                            small_position = trial_rng.randrange(min(12, len(selected)))
                        small_weight, small = selected.pop(small_position)
                        room = capacity - heavy_weight - small_weight
                        companion_position = bisect.bisect_right(selected, (room, n)) - 1
                        if companion_position < 0:
                            possible = False
                            break
                        if attempt and companion_position > 0 and trial_rng.random() < 0.50:
                            companion_position -= trial_rng.randrange(min(4, companion_position + 1))
                        _, companion = selected.pop(companion_position)
                        groups.append([heavy, small, companion])
                    if possible and not selected:
                        chosen = {item for group in groups for item in group}
                        groups.extend(optimal_pairs(item for item in ranked_all if item not in chosen))
                        return groups
                    attempt += 1
                target -= 1

        remaining = set(positive_items)
        triples = []
        while len(remaining) >= 3 and time.time() < cutoff:
            ranked = sorted(remaining, key=lambda i: (-weights[i], rng.random()))
            choices = []
            smallest = sorted(remaining, key=lambda i: (weights[i], i))[:3]
            feasible_anchors = []
            for anchor in ranked:
                if time.time() >= cutoff:
                    break
                companions = [item for item in smallest if item != anchor]
                if len(companions) < 2:
                    for item in sorted(remaining, key=lambda i: (weights[i], i)):
                        if item != anchor and item not in companions:
                            companions.append(item)
                        if len(companions) == 2:
                            break
                if len(companions) == 2 and (
                    weights[anchor] + weights[companions[0]] + weights[companions[1]] <= capacity
                ):
                    feasible_anchors.append(anchor)
                    if len(feasible_anchors) == 10:
                        break
            for anchor in feasible_anchors:
                room = capacity - weights[anchor]
                others = sorted(
                    (item for item in remaining if item != anchor),
                    key=lambda i: (weights[i], i),
                )
                lo = 0
                hi = len(others) - 1
                best_pair = None
                best_fill = -1
                while lo < hi:
                    pair_weight = weights[others[lo]] + weights[others[hi]]
                    if pair_weight <= room:
                        if pair_weight > best_fill:
                            best_fill = pair_weight
                            best_pair = (others[lo], others[hi])
                        lo += 1
                    else:
                        hi -= 1
                if best_pair is not None:
                    # Small noise changes which equally promising triples are
                    # consumed; this is where most cardinality-three diversity lives.
                    score = weights[anchor] + best_fill + rng.random() * max(1.0, capacity * 0.01)
                    choices.append((score, anchor, best_pair))
            if not choices:
                break
            _, anchor, pair = max(choices)
            group = [anchor, pair[0], pair[1]]
            triples.append(group)
            for item in group:
                remaining.remove(item)
        triples.extend(optimal_pairs(remaining))
        return triples

    def consider(candidate):
        nonlocal best, best_quality
        candidate = [list(group) for group in candidate if group]
        candidate_quality = quality(candidate)
        if len(candidate) < len(best) or (
            len(candidate) == len(best) and candidate_quality > best_quality
        ):
            best = candidate
            best_quality = candidate_quality
            return True
        return False

    # Spend the opening portion on structurally different constructions.  This
    # loop is clock-driven: faster instances automatically receive more starts.
    construction_end = min(deadline, t0 + budget * 0.24)
    construction_turn = 0
    while time.time() < construction_end and len(best) > lower_bound:
        remaining_time = construction_end - time.time()
        cutoff = min(construction_end, time.time() + max(0.004, remaining_time * 0.22))
        if 4 * minimum_positive > capacity and construction_turn % 3 == 1:
            candidate = cardinality_three_pack(cutoff)
        else:
            candidate = completion_pack(cutoff, construction_turn != 0)
        consider(candidate)
        construction_turn += 1

    if len(best) <= lower_bound:
        return format_bins(attach_zeros(best))

    def direct_empty(groups, source_index, cutoff):
        """Try to place a whole source bin into all other residual spaces."""
        moving = sorted(groups[source_index], key=lambda i: (-weights[i], i))
        kept = [list(group) for index, group in enumerate(groups) if index != source_index]
        spaces = [capacity - sum(weights[item] for item in group) for group in kept]
        if sum(weights[item] for item in moving) > sum(spaces):
            return None
        placement = [-1] * len(moving)
        failed = set()

        def place(position):
            if time.time() >= cutoff:
                return False
            if position == len(moving):
                return True
            state = (position, tuple(sorted(spaces)))
            if state in failed:
                return False
            item = moving[position]
            weight = weights[item]
            choices = sorted(
                (space - weight, index)
                for index, space in enumerate(spaces)
                if space >= weight
            )
            seen_spaces = set()
            for _, target in choices:
                old_space = spaces[target]
                if old_space in seen_spaces:
                    continue
                seen_spaces.add(old_space)
                spaces[target] -= weight
                placement[position] = target
                if place(position + 1):
                    return True
                spaces[target] = old_space
            failed.add(state)
            return False

        if not place(0):
            return None
        for position, item in enumerate(moving):
            kept[placement[position]].append(item)
        return kept

    def exact_repack(items, target_count, cutoff):
        """Pack a neighborhood into target_count bins by exact DFS."""
        if target_count <= 0:
            return [] if not items else None
        if sum(weights[item] for item in items) > target_count * capacity:
            return None

        tie = {item: rng.random() for item in items}
        ordered = sorted(items, key=lambda i: (-weights[i], tie[i]))
        quick = best_fit(ordered)
        if len(quick) <= target_count:
            return quick

        spaces = [capacity] * target_count
        result = [[] for _ in range(target_count)]
        suffix = [0] * (len(ordered) + 1)
        for position in range(len(ordered) - 1, -1, -1):
            suffix[position] = suffix[position + 1] + weights[ordered[position]]
        failed = set()

        def place(position):
            if time.time() >= cutoff:
                return False
            if position == len(ordered):
                return True
            if suffix[position] > sum(spaces):
                return False
            state = (position, tuple(sorted(spaces, reverse=True)))
            if state in failed:
                return False
            item = ordered[position]
            weight = weights[item]
            choices = sorted(
                (space - weight, index)
                for index, space in enumerate(spaces)
                if space >= weight
            )
            seen_spaces = set()
            for _, target in choices:
                old_space = spaces[target]
                if old_space in seen_spaces:
                    continue
                seen_spaces.add(old_space)
                spaces[target] -= weight
                result[target].append(item)
                if place(position + 1):
                    return True
                result[target].pop()
                spaces[target] = old_space
                if old_space == capacity:
                    break
            failed.add(state)
            return False

        if place(0):
            return [group for group in result if group]
        return None

    def choose_cluster(groups, width, turn):
        loads = loads_of(groups)
        count = len(groups)
        if width >= count:
            return list(range(count))
        if turn % 5 == 0:
            return sorted(range(count), key=lambda index: loads[index])[:width]

        # Always include a lightly loaded source.  The noise ranges from nearly
        # deterministic to broad LNS, producing many different exact subproblems.
        light = sorted(range(count), key=lambda index: loads[index])
        source_pool = light[:max(1, min(count, count // 3 + 1))]
        source = source_pool[turn % len(source_pool)] if turn % 2 == 0 else rng.choice(source_pool)
        temperature = capacity * (0.08 + 0.12 * (turn % 4))
        ranked = sorted(
            (index for index in range(count) if index != source),
            key=lambda index: loads[index] + rng.random() * temperature,
        )
        selected = [source] + ranked[:width - 1]
        return sorted(selected)

    def randomized_repartition(items, target_count):
        order = list(items)
        noise = max(1.0, capacity * (0.03 + rng.random() * 0.20))
        order.sort(key=lambda i: weights[i] + rng.random() * noise, reverse=True)
        candidate = best_fit(order)
        return candidate if len(candidate) <= target_count else None

    working = [list(group) for group in best]
    turn = 0

    # The main LNS has no iteration limit.  Every operation, including each DFS
    # node, observes a wall-clock cutoff derived from the global deadline.
    while time.time() < deadline and len(best) > lower_bound:
        now = time.time()
        remaining_time = deadline - now
        if remaining_time <= 0:
            break

        if len(working) != len(best) or turn % 23 == 0:
            working = [list(group) for group in best]

        loads = loads_of(working)
        light_order = sorted(range(len(working)), key=lambda index: loads[index])
        source_pool = light_order[:max(1, min(len(light_order), 2 + len(light_order) // 8))]
        source = source_pool[turn % len(source_pool)] if turn % 3 else rng.choice(source_pool)
        direct_cutoff = min(deadline, now + min(0.035, max(0.004, remaining_time * 0.025)))
        candidate = direct_empty(working, source, direct_cutoff)
        if candidate is not None:
            working = candidate
            consider(candidate)
            turn += 1
            continue

        count = len(working)
        maximum_width = min(14, count)
        width = 3 + (turn % max(1, maximum_width - 2))
        width = min(width, maximum_width)
        selected = choose_cluster(working, width, turn)
        selected_set = set(selected)
        items = [item for index in selected for item in working[index]]
        target_count = len(selected) - 1

        if sum(weights[item] for item in items) <= target_count * capacity:
            now = time.time()
            exact_cutoff = min(
                deadline,
                now + min(0.16, max(0.012, (deadline - now) * 0.07)),
            )
            repacked = exact_repack(items, target_count, exact_cutoff)
            if repacked is not None:
                kept = [
                    list(group)
                    for index, group in enumerate(working)
                    if index not in selected_set
                ]
                working = kept + repacked
                consider(working)
                turn += 1
                continue

        # Repartitioning the same neighborhood does not score immediately, but
        # changes the boundary presented to the next exact elimination.
        repacked = randomized_repartition(items, len(selected))
        if repacked is not None and len(repacked) == len(selected):
            kept = [
                list(group)
                for index, group in enumerate(working)
                if index not in selected_set
            ]
            perturbed = kept + repacked
            if len(perturbed) == len(best):
                working = perturbed
                consider(perturbed)

        # Periodic global target feasibility is expensive but can cross a
        # barrier that no 14-bin neighborhood contains.
        if turn % 19 == 18 and len(positive_items) <= 180:
            now = time.time()
            global_cutoff = min(
                deadline,
                now + min(0.24, max(0.025, (deadline - now) * 0.10)),
            )
            global_candidate = exact_repack(positive_items, len(best) - 1, global_cutoff)
            if global_candidate is not None:
                working = global_candidate
                consider(global_candidate)

        # Fresh minimum-slack starts remain useful after LNS has exhausted one
        # basin; they are interleaved by time, never by a terminating count.
        if turn % 13 == 12 and time.time() < deadline:
            now = time.time()
            restart_cutoff = min(
                deadline,
                now + min(0.09, max(0.008, (deadline - now) * 0.04)),
            )
            restart = completion_pack(restart_cutoff, True)
            if len(restart) <= len(best):
                working = restart
                consider(restart)

        turn += 1

    return format_bins(attach_zeros(best))
