"""Subset-sum repacking with tabu bin-reduction search
Combines best-fit-decreasing construction, bitset subset-sum repacking, one- or two-item swap tabu search and time-limited backtracking, continuously saving the best feasible packing.
"""

import bisect
import math
import random
import time


def _best_fit(items, weights, capacity, deadline):
    bins = []
    spaces = []
    for p, i in enumerate(items):
        if p % 64 == 0 and time.perf_counter() >= deadline:
            bins.extend([j] for j in items[p:])
            break
        w = weights[i]
        q = bisect.bisect_left(spaces, (w, -1))
        if q == len(spaces):
            b = len(bins)
            bins.append([i])
            bisect.insort(spaces, (capacity - w, b))
        else:
            r, b = spaces.pop(q)
            bins[b].append(i)
            bisect.insort(spaces, (r - w, b))
    return bins


def _subset_pack(items, weights, capacity, deadline, rng, grid, noise):
    remaining = sorted(items, key=weights.__getitem__, reverse=True)
    bins = []
    while remaining:
        if time.perf_counter() >= deadline:
            bins.extend(_best_fit(remaining, weights, capacity, deadline))
            break
        anchor = remaining[0]
        rest = capacity - weights[anchor]
        selected = {anchor}
        candidates = [i for i in remaining[1:] if weights[i] <= rest]
        if candidates and rest:
            if noise:
                candidates.sort(
                    key=lambda i: weights[i] * (1.0 + noise * rng.random()),
                    reverse=True,
                )
            scale = max(1, (rest + grid - 1) // grid)
            target = rest // scale
            mask = (1 << (target + 1)) - 1
            reachable = 1
            history = []
            for p, i in enumerate(candidates):
                if p % 16 == 0 and time.perf_counter() >= deadline:
                    break
                v = (weights[i] + scale - 1) // scale
                if v > target:
                    continue
                history.append((i, v, reachable))
                reachable = (reachable | (reachable << v)) & mask
                if (reachable >> target) & 1:
                    break
            value = reachable.bit_length() - 1
            for i, v, old in reversed(history):
                if not ((old >> value) & 1):
                    selected.add(i)
                    rest -= weights[i]
                    value -= v
            # Rounding up is only used to build feasible subsets; the true residual capacity continues to be filled greedily.
            for i in candidates:
                if i not in selected and weights[i] <= rest:
                    selected.add(i)
                    rest -= weights[i]
        bins.append(list(selected))
        remaining = [i for i in remaining if i not in selected]
    return bins


def _exact_pack(items, weights, capacity, count, deadline, memo_limit):
    """Returns the packing and whether the search was exhaustive; a timeout does not treat unknown states as infeasible."""
    if count <= 0 or sum(weights[i] for i in items) > count * capacity:
        return None, True
    ordered = sorted(items, key=weights.__getitem__, reverse=True)
    spaces = [capacity] * count
    bins = [[] for _ in range(count)]
    failed = set()
    expired = False

    def visit(p):
        nonlocal expired
        if time.perf_counter() >= deadline:
            expired = True
            return False
        if p == len(ordered):
            return True
        signature = (p, tuple(sorted(spaces)))
        if signature in failed:
            return False
        i = ordered[p]
        w = weights[i]
        seen = set()
        for b in sorted(range(count), key=spaces.__getitem__):
            room = spaces[b]
            if room < w or room in seen:
                continue
            seen.add(room)
            spaces[b] -= w
            bins[b].append(i)
            if visit(p + 1):
                return True
            bins[b].pop()
            spaces[b] += w
            if expired:
                return False
            if room == capacity or room == w:
                break
        if len(failed) < memo_limit:
            failed.add(signature)
        return False

    if visit(0):
        return [b for b in bins if b], True
    return None, not expired


def _lower_bound(items, weights, capacity):
    values = sorted(weights[i] for i in items)
    total = sum(values)
    bound = (total + capacity - 1) // capacity
    size = len(values)
    prefix = [0]
    for w in values:
        prefix.append(prefix[-1] + w)
    half = bisect.bisect_right(values, capacity // 2)
    large = size - half
    bound = max(bound, large)
    for p, w in enumerate(values):
        if p and w == values[p - 1]:
            continue
        fit = capacity // w
        bound = max(bound, (size - p + fit - 1) // fit)
        if 2 * w <= capacity:
            end = bisect.bisect_right(values, capacity - w)
            spare = (end - half) * capacity - (prefix[end] - prefix[half])
            extra = prefix[half] - prefix[p] - spare
            if extra > 0:
                bound = max(bound, large + (extra + capacity - 1) // capacity)
    # Items larger than one third of the capacity fit at most two per bin; two pointers give an exact bound for this subset.
    left = bisect.bisect_right(values, capacity // 3)
    right = size - 1
    paired_bins = 0
    while left <= right:
        if left < right and values[left] + values[right] <= capacity:
            left += 1
        right -= 1
        paired_bins += 1
    return max(bound, paired_bins)


def _reduce_bins(base, weights, capacity, deadline, rng, budget, attempt):
    """Fixes one bin fewer and guides swaps by total overload; only fully feasible results are output."""
    count = len(base) - 1
    if count < 1:
        return None
    started = time.perf_counter()
    span = max(1e-6, deadline - started)
    loads0 = [sum(weights[i] for i in b) for b in base]
    victims = sorted(range(len(base)), key=loads0.__getitem__)
    breadth = min(len(victims), max(1, int(1 + math.sqrt(budget))))
    victim = victims[0] if attempt == 0 else rng.choice(victims[:breadth])
    bins = [list(b) for p, b in enumerate(base) if p != victim]
    loads = [sum(weights[i] for i in b) for b in bins]
    for i in sorted(base[victim], key=weights.__getitem__, reverse=True):
        if time.perf_counter() >= deadline:
            return None
        w = weights[i]
        b = min(range(count), key=lambda j: (
            max(0, loads[j] + w - capacity) - max(0, loads[j] - capacity),
            abs(capacity - loads[j] - w),
        ))
        bins[b].append(i)
        loads[b] += w
    excess = sum(max(0, v - capacity) for v in loads)
    if excess == 0:
        return [b for b in bins if b]
    record = excess
    saved_bins = [b[:] for b in bins]
    saved_loads = loads[:]
    last_record = time.perf_counter()
    last_shake = last_record
    cache = [None] * count
    tabu = {}
    step = 0
    pair_width = max(4, int(3 + 2 * math.sqrt(budget)))
    source_width = max(5, int(4 + math.sqrt(budget * math.log2(len(weights) + 2))))
    destination_width = min(count, max(8, int(
        12 + 5 * math.sqrt(budget * math.log2(len(weights) + 2))
    )))
    tenure_base = max(2, int(math.sqrt(len(weights)) * (0.5 + math.log1p(budget) / 3)))
    stagnant_span = max(0.005, span / (2 + math.log1p(budget)))

    def options(b):
        if cache[b] is not None:
            return cache[b]
        group = bins[b]
        choices = {0: ()}
        for i in group:
            choices[weights[i]] = (i,)
        if len(group) <= pair_width:
            sampled = group
        else:
            ranked = sorted(group, key=weights.__getitem__)
            edge = max(1, pair_width // 4)
            chosen = set(ranked[:edge] + ranked[-edge:])
            chosen.update(rng.sample(group, min(len(group), pair_width - len(chosen))))
            sampled = list(chosen)
        for p, i in enumerate(sampled):
            for j in sampled[p + 1:]:
                value = weights[i] + weights[j]
                if value not in choices:
                    choices[value] = (i, j)
        sums = sorted(choices)
        result = (sums, [choices[v] for v in sums])
        cache[b] = result
        return result

    while time.perf_counter() < deadline:
        step += 1
        now = time.perf_counter()
        overloaded = [b for b in range(count) if loads[b] > capacity]
        if not overloaded:
            return [b for b in bins if b]
        if now - max(last_record, last_shake) > stagnant_span:
            # Perturb from this run's minimum-overload state, with a perturbation strength that grows with the budget and stagnation time.
            bins = [b[:] for b in saved_bins]
            loads = saved_loads[:]
            cache = [None] * count
            tabu.clear()
            strength = max(1, int(1 + math.sqrt(budget) * (now - started) / span))
            for _ in range(strength):
                if time.perf_counter() >= deadline:
                    return None
                occupied = [b for b in range(count) if bins[b]]
                if count < 2 or not occupied:
                    break
                a = rng.choice(occupied)
                b = rng.randrange(count - 1)
                if b >= a:
                    b += 1
                i = rng.choice(bins[a])
                j = rng.choice(bins[b]) if bins[b] else None
                bins[a].remove(i)
                bins[b].append(i)
                loads[a] -= weights[i]
                loads[b] += weights[i]
                if j is not None:
                    bins[b].remove(j)
                    bins[a].append(j)
                    loads[b] -= weights[j]
                    loads[a] += weights[j]
            excess = sum(max(0, v - capacity) for v in loads)
            overloaded = [b for b in range(count) if loads[b] > capacity]
            if not overloaded:
                return [b for b in bins if b]
            last_shake = now
        a = (max(overloaded, key=loads.__getitem__)
             if rng.random() < 0.65 else rng.choice(overloaded))
        over = loads[a] - capacity
        sums_a, groups_a = options(a)
        sources = list(range(1, len(sums_a)))
        if len(sources) > source_width:
            sources = rng.sample(sources, source_width)
        if destination_width < count:
            ranked = sorted(range(count), key=loads.__getitem__)
            destinations = set(ranked[:max(1, destination_width // 3)])
            destinations.update(rng.sample(range(count), destination_width - len(destinations)))
        else:
            destinations = range(count)
        chosen_move = None
        chosen_score = None
        for sa in sources:
            if time.perf_counter() >= deadline:
                return None
            value_a = sums_a[sa]
            moved_a = groups_a[sa]
            for b in destinations:
                if b == a:
                    continue
                if time.perf_counter() >= deadline:
                    return None
                spare = capacity - loads[b]
                sums_b, groups_b = options(b)
                lower = value_a - max(over, spare)
                upper = value_a - min(over, spare)
                lo = bisect.bisect_left(sums_b, lower)
                hi = bisect.bisect_right(sums_b, upper)
                positions = {0, lo - 1, lo, hi - 1, hi}
                if hi > lo:
                    positions.add(rng.randrange(lo, hi))
                old_over = over + max(0, -spare)
                old_square = max(0, spare) ** 2
                for sb in positions:
                    if sb < 0 or sb >= len(sums_b):
                        continue
                    moved_b = groups_b[sb]
                    transfer = value_a - sums_b[sb]
                    if transfer == 0:
                        continue
                    new_a = loads[a] - transfer
                    new_b = loads[b] + transfer
                    delta = max(0, new_a - capacity) + max(0, new_b - capacity) - old_over
                    new_excess = excess + delta
                    if new_excess >= record:
                        if any(tabu.get((i, b), 0) > step for i in moved_a):
                            continue
                        if any(tabu.get((i, a), 0) > step for i in moved_b):
                            continue
                    square_gain = (
                        max(0, capacity - new_a) ** 2
                        + max(0, capacity - new_b) ** 2 - old_square
                    ) / (capacity * capacity)
                    secondary = (
                        0.08 * (int(new_a > capacity) + int(new_b > capacity)
                                - 1 - int(spare < 0))
                        - 0.15 * square_gain + 0.025 * rng.random()
                    )
                    score = (new_excess, secondary)
                    if chosen_score is None or score < chosen_score:
                        chosen_score = score
                        chosen_move = (b, moved_a, moved_b, new_a, new_b)
        if chosen_move is None:
            tabu.clear()
            last_shake = started - span
            last_record = started - span
            continue
        b, moved_a, moved_b, new_a, new_b = chosen_move
        expiration = step + tenure_base + rng.randrange(tenure_base + 1)
        for i in moved_a:
            bins[a].remove(i)
            bins[b].append(i)
            tabu[i, a] = expiration
        for i in moved_b:
            bins[b].remove(i)
            bins[a].append(i)
            tabu[i, b] = expiration
        loads[a], loads[b] = new_a, new_b
        cache[a] = cache[b] = None
        excess = chosen_score[0]
        if excess == 0:
            return [group for group in bins if group]
        if excess < record:
            record = excess
            saved_bins = [group[:] for group in bins]
            saved_loads = loads[:]
            last_record = time.perf_counter()
        if len(tabu) > max(32, 4 * len(weights)):
            tabu = {key: value for key, value in tabu.items() if value > step}
    return None


def solve(instance: dict, time_limit_s: float) -> dict:
    started = time.perf_counter()
    n = int(instance['n'])
    original = [int(w) for w in instance['weights']]
    capacity = int(instance['capacity'])
    fallback = {'bin_activation': [1] * n, 'item_assignments': list(range(n))}
    if n == 0:
        return fallback
    best = [[i] for i in range(n)]
    try:
        budget = max(0.0, float(time_limit_s))
        reserve = min(0.2, 0.02 * budget) + min(0.05, n * 0.000002)
        deadline = started + max(0.0, budget - reserve)
        if time.perf_counter() >= deadline:
            return fallback
        positive = [i for i, w in enumerate(original) if w > 0]
        zeros = [i for i, w in enumerate(original) if w == 0]
        if not positive:
            return {'bin_activation': [1] + [0] * (n - 1), 'item_assignments': [0] * n}
        # Non-positive capacity, over-capacity items or negative weights are not standard feasible bin-packing inputs.
        if capacity <= 0 or len(original) != n or min(original) < 0 or max(original) > capacity:
            return fallback
        divisor = capacity
        for w in original:
            divisor = math.gcd(divisor, w)
        weights = [w // divisor for w in original]
        capacity //= divisor
        rng = random.Random((n * 1000003 + capacity * 9176 + sum(weights)) & ((1 << 64) - 1))
        ordered = sorted(positive, key=weights.__getitem__, reverse=True)
        best = _best_fit(ordered, weights, capacity, deadline)
        bound = _lower_bound(positive, weights, capacity)
        current = [b[:] for b in best]
        grid = max(512, int(4096 * (1 + math.sqrt(budget)) / max(1, math.sqrt(n / 600))))
        grid = min(grid, max(512, 24000000 // max(1, n)))
        exact_size = min(240, max(12, int(12 + 12 * math.sqrt(budget))))
        memo_limit = max(1024, int(2000 * (1 + math.sqrt(budget))))

        def accept(candidate):
            nonlocal best, current
            if candidate is None:
                return False
            candidate = [b for b in candidate if b]
            if len(candidate) > len(best):
                return False
            current = [b[:] for b in candidate]
            if len(candidate) < len(best):
                best = [b[:] for b in candidate]
                return True
            return False

        if len(best) > bound and time.perf_counter() < deadline:
            candidate = _subset_pack(
                ordered, weights, capacity,
                min(deadline, time.perf_counter() + budget * 0.09),
                rng, grid, 0.0,
            )
            accept(candidate)
        if len(best) > bound and len(positive) <= exact_size and time.perf_counter() < deadline:
            candidate, complete = _exact_pack(
                positive, weights, capacity, len(best) - 1,
                min(deadline, time.perf_counter() + budget * 0.09), memo_limit,
            )
            if candidate is None and complete:
                bound = len(best)
            else:
                accept(candidate)
        attempt = 0
        stalled = 0
        span = max(0.002, min(
            budget * 0.20,
            0.04 * (1 + math.sqrt(budget)) * (1 + math.sqrt(n / 100)),
        ))
        while len(best) > bound and time.perf_counter() < deadline:
            stop = min(deadline, time.perf_counter() + span)
            candidate = _reduce_bins(current, weights, capacity, stop, rng, budget, attempt)
            attempt += 1
            if accept(candidate):
                stalled = 0
            else:
                stalled += 1
            if len(best) <= bound or time.perf_counter() >= deadline:
                break
            count = len(current)
            width = min(count, max(2, int(2 + math.sqrt(budget * (1 + stalled)))))
            width = rng.randint(2, width) if width > 2 else width
            loads = [sum(weights[i] for i in b) for b in current]
            ranked = sorted(range(count), key=loads.__getitem__)
            selected = {ranked[0]}
            if rng.random() < 0.5:
                selected.update(ranked[:max(1, width // 3)])
            available = [b for b in range(count) if b not in selected]
            selected.update(rng.sample(available, min(len(available), width - len(selected))))
            pool = [i for b in selected for i in current[b]]
            rest = [b[:] for b in range(count) if b not in selected]
            stop = min(deadline, time.perf_counter() + span * 0.28)
            packed = _subset_pack(pool, weights, capacity, stop, rng, grid, rng.random() * 0.8)
            improved = accept(rest + packed)
            if improved:
                stalled = 0
            if len(best) <= bound or time.perf_counter() >= deadline:
                break
            if (not improved and len(pool) <= exact_size
                    and sum(weights[i] for i in pool) <= (len(selected) - 1) * capacity):
                packed, _ = _exact_pack(
                    pool, weights, capacity, len(selected) - 1,
                    min(deadline, time.perf_counter() + span * 0.20), memo_limit,
                )
                if packed is not None and accept(rest + packed):
                    stalled = 0
        # Recover the compact representation of the full x and y using consecutive bin indices; zero-weight items go into the first bin.
        assignments = [0] * n
        for b, group in enumerate(best):
            for i in group:
                assignments[i] = b
        for i in zeros:
            assignments[i] = 0
        return {
            'bin_activation': [1] * len(best) + [0] * (n - len(best)),
            'item_assignments': assignments,
        }
    except Exception:
        # A search exception must not destroy the independently saved best feasible solution.
        assignments = [0] * n
        for b, group in enumerate(best):
            for i in group:
                assignments[i] = b
        return {
            'bin_activation': [1] * len(best) + [0] * (n - len(best)),
            'item_assignments': assignments,
        }
