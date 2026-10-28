# Library slot 1/10  id=alg_277ff42af2bd
# score=141.5  feasible_rate=1.0
# score_vec=[49.0, 209.0, 134.0, 155.0, 144.0, 195.0, 125.0, 95.0, 168.0, 141.0]

import bisect
import math
import random
import time


def _solve(instance: dict, time_limit_s: float) -> dict:
    """Hybrid MBS/evolutionary large-neighborhood search for one-dimensional BPP."""
    t0 = time.time()

    try:
        n = int(instance["n"])
        capacity = int(instance["capacity"])
        weights = [int(w) for w in instance["weights"]]
    except Exception:
        return {"bin_activation": [], "item_assignments": []}

    if n <= 0:
        return {"bin_activation": [], "item_assignments": []}

    # Install a complete answer before doing any search.  Valid challenge inputs have
    # positive capacities, n weights, and no item heavier than a bin.
    best_bins = [[i] for i in range(n)]

    def output(bins):
        live = [b for b in bins if b]
        assignment = [0] * n
        for b_idx, items in enumerate(live):
            for item in items:
                if 0 <= item < n:
                    assignment[item] = b_idx
        return {
            "bin_activation": [1] * len(live) + [0] * (n - len(live)),
            "item_assignments": assignment,
        }

    if capacity <= 0 or len(weights) != n or any(w < 0 or w > capacity for w in weights):
        return output(best_bins)

    try:
        budget = max(0.0, float(time_limit_s))
    except Exception:
        budget = 0.0
    deadline = t0 + budget * 0.85
    if time.time() >= deadline:
        return output(best_bins)

    seed = n * 1000003 + capacity * 9176
    for i, w in enumerate(weights):
        seed = (seed * 1000003 + (i + 1) * (w + 17)) & ((1 << 63) - 1)
    rng = random.Random(seed)

    all_items = list(range(n))
    descending = sorted(all_items, key=lambda i: (-weights[i], i))
    total_weight = sum(weights)
    volume_lb = (total_weight + capacity - 1) // capacity

    def loads_of(bins):
        return [sum(weights[i] for i in b) for b in bins]

    def quality(bins):
        loads = loads_of(bins)
        # Concentrated load is useful: it leaves a light bin for the next elimination.
        return (len(bins), -sum(load * load for load in loads))

    def construct(order, mode="best", initial=None, cutoff=None):
        """Permutation decoder.  Best-fit supplies selection pressure to the GA."""
        bins = [] if initial is None else [list(b) for b in initial]
        loads = loads_of(bins)
        available = sorted((capacity - load, b) for b, load in enumerate(loads))

        for pos, item in enumerate(order):
            if cutoff is not None and (pos & 31) == 0 and time.time() >= cutoff:
                for rest in order[pos:]:
                    bins.append([rest])
                return bins

            w = weights[item]
            chosen = -1
            if mode == "first":
                for b, load in enumerate(loads):
                    if load + w <= capacity:
                        chosen = b
                        break
                if chosen >= 0:
                    old = (capacity - loads[chosen], chosen)
                    old_pos = bisect.bisect_left(available, old)
                    if old_pos < len(available) and available[old_pos] == old:
                        available.pop(old_pos)
            else:
                fit_pos = bisect.bisect_left(available, (w, -1))
                if fit_pos < len(available):
                    _, chosen = available.pop(fit_pos)

            if chosen < 0:
                chosen = len(bins)
                bins.append([item])
                loads.append(w)
            else:
                bins[chosen].append(item)
                loads[chosen] += w
            bisect.insort(available, (capacity - loads[chosen], chosen))
        return bins

    def mbs_pack(items, cutoff, randomized=False):
        """MBS construction: force the largest item, then solve its residual knapsack."""
        if not items:
            return []
        scale = capacity
        for item in items:
            scale = math.gcd(scale, weights[item])
        scale = max(1, scale)
        scaled_capacity = capacity // scale
        if scaled_capacity > 30000:
            return None

        remaining = sorted(items, key=lambda i: (-weights[i], i))
        result = []
        while remaining:
            if time.time() >= cutoff:
                return construct(remaining, "best", result, cutoff)
            seed_pos = 0
            if randomized and len(remaining) > 2:
                large_count = 0
                for item in remaining:
                    if weights[item] * 2 > capacity:
                        large_count += 1
                    else:
                        break
                if large_count:
                    seed_pos = rng.randrange(large_count)
                else:
                    seed_pos = rng.randrange(min(8, len(remaining)))
            seed_item = remaining[seed_pos]
            room = (capacity - weights[seed_item]) // scale
            candidates = remaining[:seed_pos] + remaining[seed_pos + 1:]
            if randomized:
                # Different DP discovery order chooses different exact-fill patterns.
                rng.shuffle(candidates)

            reachable = bytearray(room + 1)
            parent_item = [-1] * (room + 1)
            parent_sum = [-1] * (room + 1)
            reachable[0] = 1
            for pos, item in enumerate(candidates):
                sw = weights[item] // scale
                if 0 < sw <= room:
                    for value in range(room, sw - 1, -1):
                        if not reachable[value] and reachable[value - sw]:
                            reachable[value] = 1
                            parent_item[value] = item
                            parent_sum[value] = value - sw
                if reachable[room]:
                    break
                if (pos & 15) == 15 and time.time() >= cutoff:
                    return construct(remaining, "best", result, cutoff)

            fill = room
            while fill and not reachable[fill]:
                fill -= 1
            chosen = {seed_item}
            while fill:
                item = parent_item[fill]
                if item < 0:
                    break
                chosen.add(item)
                fill = parent_sum[fill]
            result.append(sorted(chosen))
            remaining = [item for item in remaining if item not in chosen]
        return result

    def varied_order(items, strength):
        """Weight-aware mutation: from gentle rank jitter to a full restart."""
        order = list(items)
        if strength >= 0.85:
            rng.shuffle(order)
            return order
        amplitude = capacity * (0.025 + 0.32 * strength)
        order.sort(key=lambda i: (-(weights[i] + rng.uniform(-amplitude, amplitude)), rng.random()))
        if strength > 0.45 and len(order) > 3:
            # A few large/small alternations create patterns unreachable by tie shuffles.
            ordered = sorted(order, key=lambda i: weights[i], reverse=True)
            lo, hi = 0, len(ordered) - 1
            mixed = []
            while lo <= hi:
                if lo <= hi:
                    mixed.append(ordered[lo])
                    lo += 1
                if lo <= hi:
                    mixed.append(ordered[hi])
                    hi -= 1
            if rng.random() < strength - 0.30:
                order = mixed
        return order

    # First-fit and best-fit respond differently to the same chromosome, so both
    # interpretations seed the population.
    ffd = construct(descending, "first", cutoff=deadline)
    bfd = construct(descending, "best", cutoff=deadline)
    best_bins = min((best_bins, ffd, bfd), key=quality)

    # If no three items can coexist, bin packing is exactly maximum threshold
    # matching.  The two-pointer packing is optimal, not merely heuristic.
    if n < 3 or sum(sorted(weights)[:3]) > capacity:
        ascending = sorted(all_items, key=lambda i: (weights[i], i))
        paired = []
        lo, hi = 0, n - 1
        while lo < hi:
            if weights[ascending[lo]] + weights[ascending[hi]] <= capacity:
                paired.append([ascending[hi], ascending[lo]])
                lo += 1
                hi -= 1
            else:
                paired.append([ascending[hi]])
                hi -= 1
        if lo == hi:
            paired.append([ascending[lo]])
        return output(paired)

    if len(best_bins) <= volume_lb or time.time() >= deadline:
        return output(best_bins)

    def fingerprint(bins):
        return tuple(sorted(tuple(sorted(b)) for b in bins))

    # A deterministic MBS parent supplies high-quality bin patterns; randomized
    # permutation parents supply the complementary diversity.
    mbs_cutoff = min(deadline, time.time() + max(0.002, (deadline - time.time()) * 0.16))
    mbs = mbs_pack(all_items, mbs_cutoff, False)
    initial_candidates = [ffd, bfd]
    if mbs is not None:
        initial_candidates.append(mbs)
        if quality(mbs) < quality(best_bins):
            best_bins = mbs

    initial_seen = {fingerprint(candidate) for candidate in initial_candidates}

    while time.time() < min(deadline, mbs_cutoff):
        order = varied_order(all_items, rng.random())
        candidate = construct(order, "best", cutoff=mbs_cutoff)
        mark = fingerprint(candidate)
        if mark not in initial_seen:
            initial_seen.add(mark)
            initial_candidates.append(candidate)
            initial_candidates.sort(key=quality)
            if len(initial_candidates) > 14:
                removed = initial_candidates.pop()
                initial_seen.discard(fingerprint(removed))
        if quality(candidate) < quality(best_bins):
            best_bins = candidate
        if len(best_bins) <= volume_lb:
            return output(best_bins)

    population = []
    population_seen = set()

    def admit(candidate):
        nonlocal population, best_bins
        candidate = [list(b) for b in candidate if b]
        if not candidate:
            return
        mark = fingerprint(candidate)
        if mark in population_seen:
            return
        population_seen.add(mark)
        population.append(candidate)
        population.sort(key=quality)
        # A small elite archive keeps crossover cheap while retaining distinct bins.
        if len(population) > 14:
            removed = population.pop()
            population_seen.discard(fingerprint(removed))
        if quality(candidate) < quality(best_bins):
            best_bins = [list(b) for b in candidate]

    for candidate in initial_candidates:
        admit(candidate)
    admit(best_bins)

    def fixed_pack(items, bin_count, cutoff):
        """Try to repack a neighborhood into at most bin_count bins."""
        if not items:
            return []
        if bin_count <= 0 or sum(weights[i] for i in items) > bin_count * capacity:
            return None

        base = sorted(items, key=lambda i: (-weights[i], i))
        for mode in ("best", "first"):
            candidate = construct(base, mode, cutoff=cutoff)
            if len(candidate) <= bin_count:
                return candidate
        if time.time() >= cutoff:
            return None

        # Spend the first part of the slice evolving permutations.  Unlike a plain
        # GA, survivors are immediately handed to exact fixed-cardinality repair.
        random_cutoff = min(cutoff, time.time() + (cutoff - time.time()) * 0.30)
        while time.time() < random_cutoff:
            order = varied_order(items, rng.random())
            candidate = construct(order, "best", cutoff=random_cutoff)
            if len(candidate) <= bin_count:
                return candidate

        if time.time() >= cutoff:
            return None

        # Symmetry-reduced exact assignment.  It is especially effective after MBS
        # has reduced the problem to a handful of mutually constraining bins.
        order = sorted(items, key=lambda i: (-weights[i], i))
        if len(order) > 800:
            return None
        bin_loads = [0] * bin_count
        packed = [[] for _ in range(bin_count)]
        failed = set()

        suffix = [0] * (len(order) + 1)
        for pos in range(len(order) - 1, -1, -1):
            suffix[pos] = suffix[pos + 1] + weights[order[pos]]

        def place(pos):
            if time.time() >= cutoff:
                return False
            if pos == len(order):
                return True
            if suffix[pos] > bin_count * capacity - sum(bin_loads):
                return False

            state = (pos, tuple(sorted(bin_loads)))
            if state in failed:
                return False

            item = order[pos]
            w = weights[item]
            choices = sorted(
                range(bin_count),
                key=lambda b: (capacity - bin_loads[b] - w) if bin_loads[b] + w <= capacity else capacity + bin_loads[b],
            )
            seen_loads = set()
            for b in choices:
                load = bin_loads[b]
                if load in seen_loads or load + w > capacity:
                    continue
                seen_loads.add(load)
                bin_loads[b] += w
                packed[b].append(item)
                if place(pos + 1):
                    return True
                packed[b].pop()
                bin_loads[b] -= w
                # An item failing in a previously empty bin cannot benefit from
                # trying another equivalent empty bin.
                if load == 0:
                    break
            failed.add(state)
            return False

        if place(0):
            return [b for b in packed if b]
        return None

    def choose_neighborhood(bins, phase):
        """Select bins whose combined slack can pay for eliminating one bin."""
        loads = loads_of(bins)
        ranked = sorted(range(len(bins)), key=lambda b: (loads[b], rng.random()))
        if phase % 4 == 0:
            candidates = ranked
        else:
            # Bias toward slack, but let evolutionary restarts change which item
            # interactions the exact repair sees.
            candidates = list(range(len(bins)))
            rng.shuffle(candidates)
            candidates.sort(key=lambda b: loads[b] + rng.random() * capacity * (0.18 + 0.08 * (phase % 4)))

        selected = []
        slack = 0
        for b in candidates:
            selected.append(b)
            slack += capacity - loads[b]
            if slack >= capacity and len(selected) >= 2:
                break
        if slack < capacity:
            return []

        # A low-load bin is often a stranded item just above C/2.  Adding more
        # stranded bins cannot free it; add full donor bins containing compatible
        # small items instead.  This is the ejection-chain bridge between the
        # permutation parents and fixed-cardinality repair.
        extra = 1 + phase % 4
        selected_set = set(selected)
        max_gap = max(capacity - loads[b] for b in selected)
        donors = [b for b in range(len(bins)) if b not in selected_set]

        def donor_key(b):
            compatible = sum(weights[i] for i in bins[b] if weights[i] <= max_gap)
            incompatible = sum(1 for i in bins[b] if weights[i] > max_gap)
            noise = rng.random() * capacity * (0.04 if phase % 4 == 0 else 0.18)
            return (incompatible, -compatible + noise, capacity - loads[b])

        donors.sort(key=donor_key)
        for b in donors:
            if extra <= 0:
                break
            selected.append(b)
            selected_set.add(b)
            extra -= 1
        return selected

    def crossover(parent_a, parent_b, cutoff):
        """Bin-pattern crossover followed by best-fit completion."""
        used = set()
        locked = []
        target_items = max(1, int(n * rng.uniform(0.30, 0.68)))
        sources = [parent_a, parent_b]
        turn = rng.randrange(2)
        misses = 0
        while len(used) < target_items and time.time() < cutoff:
            source = sources[turn]
            turn = 1 - turn
            ranked = sorted(source, key=lambda b: (-sum(weights[i] for i in b), rng.random()))
            if not ranked:
                break
            window = min(len(ranked), 1 + max(1, len(ranked) // 3))
            chosen = ranked[rng.randrange(window)]
            if all(i not in used for i in chosen):
                locked.append(list(chosen))
                used.update(chosen)
                misses = 0
            else:
                misses += 1
                if misses > len(parent_a) + len(parent_b):
                    break
        leftovers = [i for i in all_items if i not in used]
        order = varied_order(leftovers, rng.random() * 0.75)
        return construct(order, "best", locked, cutoff)

    phase = 0
    while time.time() < deadline:
        phase += 1
        now = time.time()
        remaining_time = deadline - now
        if remaining_time <= 0:
            break
        slice_s = min(0.22, max(0.006, remaining_time * 0.11))
        cutoff = min(deadline, now + slice_s)

        if phase % 5 == 0:
            # Re-running MBS is meaningful rather than a restart clone: shuffled DP
            # discovery selects a different exact residual subset for every seed.
            candidate = mbs_pack(all_items, cutoff, True)
            if candidate is not None:
                admit(candidate)
            if len(best_bins) <= volume_lb:
                break
            continue

        if phase % 3 == 0 and len(population) >= 2:
            a = population[rng.randrange(len(population))]
            b = population[rng.randrange(len(population))]
            child = crossover(a, b, cutoff)
            admit(child)
            if len(best_bins) <= volume_lb:
                break
            continue

        parent = best_bins if phase % 2 else population[rng.randrange(len(population))]
        selected = choose_neighborhood(parent, phase)
        if not selected:
            continue
        selected_set = set(selected)
        neighborhood_items = [item for b in selected for item in parent[b]]
        target_count = len(selected) - 1
        repaired = fixed_pack(neighborhood_items, target_count, cutoff)
        if repaired is not None:
            child = [list(parent[b]) for b in range(len(parent)) if b not in selected_set]
            child.extend(repaired)
            admit(child)
            if len(child) < len(best_bins):
                best_bins = [list(b) for b in child]
            if len(best_bins) <= volume_lb:
                break
        elif time.time() < cutoff:
            # Failed elimination still mutates the chromosome: randomized same-size
            # rebuilding changes which high-fill patterns crossover can inherit.
            order = varied_order(neighborhood_items, 0.45 + 0.5 * rng.random())
            rebuilt = construct(order, "best", cutoff=cutoff)
            if len(rebuilt) <= len(selected):
                child = [list(parent[b]) for b in range(len(parent)) if b not in selected_set]
                child.extend(rebuilt)
                admit(child)

    return output(best_bins)


def solve(instance: dict, time_limit_s: float) -> dict:
    """Never let a failed improvement phase discard the installed singleton packing."""
    try:
        return _solve(instance, time_limit_s)
    except Exception:
        try:
            n = max(0, int(instance.get("n", 0)))
        except Exception:
            n = 0
        return {
            "bin_activation": [1] * n,
            "item_assignments": list(range(n)),
        }
