"""Lagrangian-guided adaptive neighbourhood set covering
Combines multiplier updates, random perturbation, incremental greedy repair, redundancy removal and set swaps, improving local coverage with time-limited branch search.
"""

import heapq
import math
import random
import time


def solve(instance: dict, time_limit_s: float) -> dict:
    start = time.perf_counter()
    try:
        n = max(0, int(instance['n']))
    except Exception:
        return {'selection_status': []}
    answer = [1] * n
    try:
        budget = max(0.0, float(time_limit_s))
        reserve = min(0.2, budget * 0.04) + min(budget * 0.04, n * 0.000002)
        deadline = start + budget - reserve
        if time.perf_counter() < deadline:
            _search(instance, budget, deadline, answer)
    except Exception:
        # For feasible input, select-all or the saved improved solution is always a complete fallback.
        pass
    return {'selection_status': answer}


def _search(instance, budget, deadline, answer):
    clock = time.perf_counter
    n, m = instance['n'], instance['m']
    original_cost = instance['costs']
    fixed = {j for j in range(n) if original_cost[j] <= 0}
    canonical = []
    for i in range(m):
        if clock() >= deadline:
            return
        row = tuple(dict.fromkeys(instance['covers'][i]))
        if not row:
            return
        if len(row) == 1:
            fixed.add(row[0])
        canonical.append(row)

    residual = []
    columns = [[] for _ in range(n)]
    for row in canonical:
        if clock() >= deadline:
            return
        if any(j in fixed for j in row):
            continue
        i = len(residual)
        residual.append(row)
        for j in row:
            columns[j].append(i)
    if not residual:
        answer[:] = [int(j in fixed) for j in range(n)]
        return

    # Among identical residual coverages keep only the cheapest set; all non-positive-cost sets are always kept.
    representatives = {}
    for j, col in enumerate(columns):
        if clock() >= deadline:
            return
        if col:
            key = tuple(col)
            old = representatives.get(key)
            if old is None or original_cost[j] < original_cost[old]:
                representatives[key] = j
    originals = list(representatives.values())
    index = {j: k for k, j in enumerate(originals)}
    elems = [columns[j] for j in originals]
    costs = [original_cost[j] for j in originals]
    rows = []
    for row in residual:
        if clock() >= deadline:
            return
        rows.append(sorted((index[j] for j in row if j in index), key=costs.__getitem__))
    k, r = len(elems), len(rows)
    del canonical, residual, columns, representatives, index
    rng = random.Random(19790317)
    strength = math.log1p(budget)
    epoch = max(0.001, budget / (6.0 + math.log2(k + r + 2)))
    forced = {row[0] for row in rows if len(row) == 1}
    lookup = {}
    best = set(range(k))
    best_counts = [len(row) for row in rows]
    best_cost = sum(costs)
    last_improvement = clock()

    def save(selected, counts):
        nonlocal best, best_counts, best_cost, last_improvement
        value = sum(costs[j] for j in selected)
        if value < best_cost:
            result = [0] * n
            for j in fixed:
                result[j] = 1
            for j in selected:
                result[originals[j]] = 1
            answer[:] = result
            best, best_counts, best_cost = selected.copy(), counts.copy(), value
            last_improvement = clock()
        return value

    # Save the complete feasible solution after preprocessing before any search.
    result = [0] * n
    for j in fixed:
        result[j] = 1
    for j in originals:
        result[j] = 1
    answer[:] = result

    def add(selected, counts, j):
        selected.add(j)
        for i in elems[j]:
            counts[i] += 1

    def remove(selected, counts, j):
        selected.remove(j)
        for i in elems[j]:
            counts[i] -= 1

    def prune(selected, counts, end):
        for j in sorted(selected, key=costs.__getitem__, reverse=True):
            if clock() >= end:
                return
            if j not in forced and all(counts[i] > 1 for i in elems[j]):
                remove(selected, counts, j)

    def repair(selected, counts, weights, adjusted, noise, end):
        missing = [i for i in range(r) if counts[i] == 0]
        if not missing:
            return True
        gains = {}
        for i in missing:
            if clock() >= end:
                return False
            w = weights[i]
            for j in rows[i]:
                gains[j] = gains.get(j, 0.0) + w
        multipliers = {}
        heap = []
        for j, gain in gains.items():
            if clock() >= end:
                return False
            coefficient = adjusted[j] * math.exp(noise * (rng.random() - 0.5))
            multipliers[j] = coefficient
            heap.append((coefficient / gain, j, gain))
        heapq.heapify(heap)
        remaining = len(missing)
        while remaining and heap:
            if clock() >= end:
                return False
            _, j, old_gain = heapq.heappop(heap)
            gain = gains[j]
            if gain <= 1e-10 or j in selected:
                continue
            if gain != old_gain:
                heapq.heappush(heap, (multipliers[j] / gain, j, gain))
                continue
            selected.add(j)
            for i in elems[j]:
                if counts[i] == 0:
                    remaining -= 1
                    w = weights[i]
                    for other in rows[i]:
                        gains[other] -= w
                counts[i] += 1
                if clock() >= end:
                    return False
        return remaining == 0

    def exchange(selected, counts, end):
        order = list(selected)
        rng.shuffle(order)
        for old in order:
            if clock() >= end:
                return
            if old not in selected or old in forced:
                continue
            unique = {i for i in elems[old] if counts[i] == 1}
            if not unique:
                remove(selected, counts, old)
                continue
            target = min(unique, key=lambda i: len(rows[i]))
            for new in rows[target]:
                if clock() >= end or costs[new] >= costs[old]:
                    break
                if new in selected:
                    continue
                membership = lookup.get(new)
                if membership is None:
                    membership = frozenset(elems[new])
                    lookup[new] = membership
                if unique.issubset(membership):
                    add(selected, counts, new)
                    remove(selected, counts, old)
                    break
        prune(selected, counts, end)

    def exact_repair(missing, initial, lambdas, end):
        # Solve only the covering subproblem exposed by the removed sets, always keeping a feasible complement.
        incumbent = list(initial)
        upper = sum(costs[j] for j in incumbent)
        if not missing or clock() >= end:
            return incumbent
        masks, prices = {}, {}
        for p, i in enumerate(missing):
            if clock() >= end:
                return incumbent
            bit = 1 << p
            for j in rows[i]:
                if costs[j] < upper:
                    masks[j] = masks.get(j, 0) | bit
                    prices[j] = prices.get(j, 0.0) + lambdas[i]
        cheapest = {}
        scale = 1.0
        for j, mask in masks.items():
            if clock() >= end:
                return incumbent
            old = cheapest.get(mask)
            if old is None or costs[j] < costs[old]:
                cheapest[mask] = j
            if prices[j] > 0.0:
                scale = min(scale, costs[j] / prices[j])
        candidates = list(cheapest.values())
        local_rows = [[] for _ in missing]
        for j in candidates:
            if clock() >= end:
                return incumbent
            mask = masks[j]
            while mask:
                bit = mask & -mask
                local_rows[bit.bit_length() - 1].append(j)
                mask ^= bit
        if any(not row for row in local_rows):
            return incumbent
        dual = [max(0.0, lambdas[i] * scale * (1.0 - 1e-10)) for i in missing]
        full = (1 << len(missing)) - 1
        stack = [(full, 0, ())]
        seen = {}
        while stack and clock() < end:
            mask, value, chosen = stack.pop()
            if value >= upper:
                continue
            if mask == 0:
                upper, incumbent = value, list(chosen)
                continue
            previous = seen.get(mask)
            if previous is not None and previous <= value:
                continue
            seen[mask] = value
            remaining = mask
            bound, target, smallest = 0.0, -1, math.inf
            while remaining:
                bit = remaining & -remaining
                p = bit.bit_length() - 1
                bound += dual[p]
                degree = len(local_rows[p])
                if degree < smallest:
                    target, smallest = p, degree
                remaining ^= bit
            if value + bound >= upper - 1e-8 * max(1.0, abs(upper)):
                continue
            options = []
            for j in local_rows[target]:
                if clock() >= end:
                    return incumbent
                if value + costs[j] < upper:
                    gain = (masks[j] & mask).bit_count()
                    options.append((costs[j] / gain, j))
            options.sort(reverse=True)
            for _, j in options:
                next_mask = mask & ~masks[j]
                next_cost = value + costs[j]
                if not next_mask:
                    if next_cost < upper:
                        upper, incumbent = next_cost, list(chosen) + [j]
                elif next_cost < upper:
                    stack.append((next_mask, next_cost, chosen + (j,)))
        return incumbent

    selected, counts = best.copy(), best_counts.copy()
    prune(selected, counts, deadline)
    save(selected, counts)
    if clock() >= deadline:
        return
    selected, counts = set(forced), [0] * r
    for j in selected:
        for i in elems[j]:
            counts[i] += 1
    if repair(selected, counts, [1.0] * r, costs, 0.0, deadline):
        prune(selected, counts, deadline)
        save(selected, counts)
        exchange(selected, counts, min(deadline, clock() + epoch))
        save(selected, counts)
    if clock() >= deadline:
        return

    lambdas = []
    for row in rows:
        if clock() >= deadline:
            return
        lambdas.append(min(costs[j] / len(elems[j]) for j in row))
    best_dual = lambdas.copy()
    lower_bound = -math.inf
    last_dual = clock()
    current, current_counts, current_cost = best.copy(), best_counts.copy(), best_cost

    while clock() < deadline:
        # The relaxation solution, lower bound and repair scores use the same multipliers.
        reduced, relaxed_counts, relaxed = [], [0] * r, []
        lagrangian = sum(lambdas)
        for j, col in enumerate(elems):
            if clock() >= deadline:
                return
            rc = costs[j] - sum(lambdas[i] for i in col)
            reduced.append(rc)
            if rc < 0.0:
                relaxed.append(j)
                lagrangian += rc
                for i in col:
                    relaxed_counts[i] += 1
        now = clock()
        if lagrangian > lower_bound + 1e-8:
            lower_bound, best_dual, last_dual = lagrangian, lambdas.copy(), now
        stale = min(2.0, (now - last_improvement) / max(epoch, budget * 0.15))
        mean = max(1e-12, sum(lambdas) / r)
        influence = (0.15 + 0.5 * rng.random()) * strength / (1.0 + strength)
        weights = [1.0 - influence + influence * min(6.0, max(0.2, x / mean)) for x in lambdas]
        adjusted = [costs[j] * (1.0 - influence + influence * max(0.08, min(1.0, reduced[j] / costs[j]))) for j in range(k)]

        restart_probability = min(0.3, (0.04 + 0.06 * strength) * (1.0 + stale))
        if rng.random() < restart_probability:
            kept, kept_counts = set(forced), [0] * r
            for j in kept:
                for i in elems[j]:
                    kept_counts[i] += 1
            # Embed part of the relaxation solution into the reconstruction rather than running a separate solver.
            relaxed.sort(key=lambda j: reduced[j] / costs[j])
            seed_size = min(len(relaxed), int(len(best) * influence))
            for j in relaxed[:seed_size]:
                if j not in kept:
                    add(kept, kept_counts, j)
            original_additions = best - kept
        else:
            if rng.random() < 0.65:
                source, source_counts = best, best_counts
            else:
                source, source_counts = current, current_counts
            kept, kept_counts = source.copy(), source_counts.copy()
            removable = [j for j in source if j not in forced]
            limit = min(len(removable), max(1, int((1.0 + strength) * math.sqrt(len(removable) + 1) * (1.0 + stale))))
            quantity = min(len(removable), 1 + int(rng.random() ** 2 * limit))
            ranking = []
            for j in removable:
                if clock() >= deadline:
                    return
                unique_price = sum(lambdas[i] for i in elems[j] if kept_counts[i] == 1)
                inefficiency = costs[j] / max(mean, unique_price + 0.1 * costs[j])
                ranking.append((-math.log(max(1e-12, rng.random())) / max(0.05, inefficiency), j))
            ranking.sort()
            for _, j in ranking[:quantity]:
                remove(kept, kept_counts, j)
            original_additions = source - kept
        missing = [i for i in range(r) if kept_counts[i] == 0]
        selected, counts = kept.copy(), kept_counts.copy()
        noise = min(1.2, (0.05 + 0.15 * strength) * (1.0 + stale) * rng.random())
        if not repair(selected, counts, weights, adjusted, noise, deadline):
            return
        additions = selected - kept
        if sum(costs[j] for j in original_additions) < sum(costs[j] for j in additions):
            additions = original_additions
        prune(selected, counts, deadline)
        value = save(selected, counts)
        exchange(selected, counts, min(deadline, clock() + epoch * (0.3 + influence)))
        value = save(selected, counts)
        candidate, candidate_counts, candidate_cost = selected, counts, value

        # Neighbourhood size and allotted wall-clock time are both determined by the available time limit and instance size.
        local_limit = (1.0 + strength) * (1.0 + math.sqrt(r))
        if missing and len(missing) <= local_limit and clock() < deadline:
            end = min(deadline, clock() + epoch * (0.2 + 0.35 * strength) / (1.0 + stale))
            improved_additions = exact_repair(missing, additions, lambdas, end)
            selected, counts = kept.copy(), kept_counts.copy()
            for j in improved_additions:
                if j not in selected:
                    add(selected, counts, j)
            prune(selected, counts, deadline)
            value = save(selected, counts)
            if value < candidate_cost:
                candidate, candidate_counts, candidate_cost = selected, counts, value

        temperature = max(1e-12, best_cost / max(1, len(best)) * 0.12 * strength / (1.0 + strength))
        if candidate_cost <= current_cost or rng.random() < math.exp(-min(700.0, (candidate_cost - current_cost) / temperature)):
            current, current_counts, current_cost = candidate.copy(), candidate_counts.copy(), candidate_cost
        if current_cost > best_cost * (1.0 + 0.02 * strength):
            current, current_counts, current_cost = best.copy(), best_counts.copy(), best_cost

        norm = sum((1 - count) ** 2 for count in relaxed_counts)
        dual_age = max(0.0, clock() - last_dual)
        alpha = 1.8 / (1.0 + dual_age / epoch)
        if norm:
            step = alpha * max(0.0, best_cost - lagrangian) / norm
            lambdas = [max(0.0, x + step * (1 - count)) for x, count in zip(lambdas, relaxed_counts)]
        else:
            lambdas = [x * (0.9 + 0.2 * rng.random()) for x in lambdas]
        if dual_age > epoch * (2.0 + strength):
            lambdas = [max(0.0, x * (0.85 + 0.3 * rng.random())) for x in best_dual]
            last_dual = clock()
