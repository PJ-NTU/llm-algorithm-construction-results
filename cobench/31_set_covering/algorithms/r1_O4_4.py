"""Dual pricing and exact-neighbourhood set covering
Uses adaptive Lagrangian subgradient, incremental weighted greedy repair, redundancy removal and time-limited branch-and-bound neighbourhood search.
"""

import array
import heapq
import math
import random
import time

try:
    import numpy as np
except Exception:
    np = None


def solve(instance: dict, time_limit_s: float) -> dict:
    start = time.perf_counter()
    answer = {"selection_status": [1] * instance["n"]}
    try:
        _search(instance, float(time_limit_s), start, answer)
    except Exception:
        # The committed solution is always complete and feasible; failure of an optional acceleration module does not discard the current upper bound.
        pass
    return answer


def _search(instance, budget, start, answer):
    clock = time.perf_counter
    budget = max(0.0, budget)
    deadline = start + budget - min(0.1, 0.025 * budget)
    if clock() >= deadline:
        return
    n, m = instance["n"], instance["m"]
    original_costs = instance["costs"]
    fixed = [v <= 0 for v in original_costs]
    original_rows = []
    for i in range(m):
        row = instance["covers"][i]
        if clock() >= deadline:
            return
        row = list(dict.fromkeys(row))
        if not row:
            return  # For infeasible input no return value satisfies all hard constraints.
        if len(row) == 1:
            fixed[row[0]] = True
        original_rows.append(row)
    residual_rows = []
    active = set()
    for row in original_rows:
        if clock() >= deadline:
            return
        if not any(fixed[j] for j in row):
            residual_rows.append(row)
            active.update(row)
    fixed_x = [int(v) for v in fixed]
    if not residual_rows:
        answer["selection_status"] = fixed_x
        return
    ids = sorted(active)
    index = {j: k for k, j in enumerate(ids)}
    costs = [original_costs[j] for j in ids]
    N, M = len(ids), len(residual_rows)
    rows = []
    elements = [[] for _ in ids]
    cheapest = set()
    for i, row in enumerate(residual_rows):
        if clock() >= deadline:
            return
        r = [index[j] for j in row]
        rows.append(r)
        cheapest.add(min(r, key=costs.__getitem__))
        for j in r:
            elements[j].append(i)
    del original_rows, residual_rows, index, active

    best = set(range(N))
    best_cost = sum(costs)
    published = False
    last_improvement = clock()
    rng = random.Random(9173)
    strength = math.log1p(budget)
    intensity = strength / (1.0 + strength)

    def publish(chosen):
        nonlocal best, best_cost, last_improvement, published
        value = sum(costs[j] for j in chosen)
        if value < best_cost or not published:
            x = fixed_x[:]
            for j in chosen:
                x[ids[j]] = 1
            answer["selection_status"] = x
            published = True
            if value < best_cost:
                last_improvement = clock()
            best, best_cost = set(chosen), value
        return value

    publish(cheapest)

    def counts_for(chosen, stop):
        count = [0] * M
        for j in chosen:
            if clock() >= stop:
                return None
            for i in elements[j]:
                count[i] += 1
        return count

    def prune(chosen, count, stop, noise=0.0):
        if noise:
            order = sorted(chosen, key=lambda j: costs[j] *
                           (1.0 + noise * rng.random()), reverse=True)
        else:
            order = sorted(chosen, key=costs.__getitem__, reverse=True)
        for j in order:
            if clock() >= stop:
                break
            if all(count[i] > 1 for i in elements[j]):
                chosen.remove(j)
                for i in elements[j]:
                    count[i] -= 1
        return chosen, count

    def repair(seed, weights, prices, stop, noise=0.0):
        chosen = set(seed)
        count = counts_for(chosen, stop)
        if count is None:
            return None
        uncovered = [i for i in range(M) if count[i] == 0]
        left = len(uncovered)
        if left:
            gain = [0.0] * N
            for i in uncovered:
                if clock() >= stop:
                    return None
                w = weights[i]
                for j in rows[i]:
                    gain[j] += w
            heap = [(-gain[j] / prices[j], j) for j in range(N)
                    if gain[j] > 0.0 and j not in chosen]
            heapq.heapify(heap)
            while left and heap:
                if clock() >= stop:
                    return None
                old, j = heapq.heappop(heap)
                if j in chosen or gain[j] <= 1e-12:
                    continue
                key = -gain[j] / prices[j]
                if key > old + 1e-12 * max(1.0, abs(old)):
                    heapq.heappush(heap, (key, j))
                    continue
                chosen.add(j)
                for i in elements[j]:
                    if clock() >= stop:
                        return None
                    if count[i] == 0:
                        left -= 1
                        w = weights[i]
                        for k in rows[i]:
                            gain[k] -= w
                    count[i] += 1
            if left:
                return None
        return prune(chosen, count, stop, noise)

    count = counts_for(cheapest, deadline)
    if count is None:
        return
    chosen, count = prune(set(cheapest), count, deadline)
    publish(chosen)
    result = repair(set(), [1.0] * M, costs, deadline)
    if result is None:
        return
    publish(result[0])
    if clock() >= deadline:
        return

    # Non-negative and single-element mandatory sets are already fixed; below only the remaining positive-cost variables are optimised.
    initial = []
    for row in rows:
        if clock() >= deadline:
            return
        initial.append(min(costs[j] / len(elements[j]) for j in row))
    slack = []
    for j in range(N):
        if clock() >= deadline:
            return
        slack.append(max(0.0, costs[j] - sum(initial[i] for i in elements[j])))
    order = sorted(range(M), key=lambda i: (len(rows[i]), -initial[i]))
    for i in order:
        if clock() >= deadline:
            return
        delta = max(0.0, min(slack[j] for j in rows[i]))
        initial[i] += delta
        for j in rows[i]:
            slack[j] = max(0.0, slack[j] - delta)

    use_numpy = np is not None
    if use_numpy:
        row_edges, col_edges = array.array('q'), array.array('q')
        for i, row in enumerate(rows):
            if clock() >= deadline:
                return
            row_edges.extend([i] * len(row))
            col_edges.extend(row)
        er = np.frombuffer(row_edges, dtype=np.int64)
        ec = np.frombuffer(col_edges, dtype=np.int64)
        cvector = np.asarray(costs, dtype=float)
        multipliers = np.asarray(initial, dtype=float)
    else:
        multipliers = initial[:]
    best_lambda = multipliers.copy()
    best_lb = -math.inf
    alpha = 1.8
    last_dual_improvement = clock()
    last_adapt = clock()
    last_relaxed = None
    dual_spent = 0.0
    local_spent = 0.0
    current = set(best)
    current_cost = best_cost

    def exact_neighborhood(kept, removed, repaired, rc, stop):
        """Fix the sets outside the neighbourhood; branch-and-bound with valid dual bounds inside the screened neighbourhood."""
        count = counts_for(kept, stop)
        if count is None:
            return None
        uncovered = [i for i in range(M) if count[i] == 0]
        if not uncovered:
            return set(kept)
        R = len(uncovered)
        row_limit = max(1, int((1.0 + strength) * math.sqrt(M + N)))
        if R > row_limit:
            return None
        extra = set(removed)
        upper = sum(costs[j] for j in extra)
        if repaired is not None:
            candidate = repaired - kept
            value = sum(costs[j] for j in candidate)
            if value < upper:
                upper, extra = value, candidate
        best_extra = set(extra)
        masks = {}
        for k, i in enumerate(uncovered):
            if clock() >= stop:
                return None
            bit = 1 << k
            for j in rows[i]:
                if j not in kept and costs[j] < upper:
                    masks[j] = masks.get(j, 0) | bit
        if not masks:
            return None
        # Keep low-cost / low-reduced-cost candidates for every row, plus the existing feasible solution of the neighbourhood.
        per_row = max(1, int(math.ceil(1.0 + strength)))
        rank = {j: (0.5 * costs[j] + 0.5 * max(0.0, rc[j])) /
                mask.bit_count() for j, mask in masks.items()}
        pool = set(removed) | best_extra
        for i in uncovered:
            if clock() >= stop:
                return None
            eligible = [j for j in rows[i] if j in masks]
            pool.update(heapq.nsmallest(per_row, eligible, key=rank.__getitem__))
        by_mask = {}
        for j in pool:
            mask = masks.get(j, 0)
            if mask and (mask not in by_mask or costs[j] < costs[by_mask[mask]]):
                by_mask[mask] = j
        candidates = sorted(by_mask.values(), key=lambda j:
                            (costs[j], -masks[j].bit_count()))
        local_ids, local_masks = [], []
        for j in candidates:
            if clock() >= stop:
                return None
            mask = masks[j]
            if any(mask & old == mask for old in local_masks):
                continue
            local_ids.append(j)
            local_masks.append(mask)
        local_costs = [costs[j] for j in local_ids]
        row_cols = [[] for _ in uncovered]
        for a, mask in enumerate(local_masks):
            if clock() >= stop:
                return None
            while mask:
                bit = mask & -mask
                row_cols[bit.bit_length() - 1].append(a)
                mask ^= bit
        if any(not row for row in row_cols):
            return None
        # Build a feasible covering dual on the screened columns, so no branch lower bound overestimates.
        slack_local = local_costs[:]
        dual = [0] * R
        row_order = sorted(range(R), key=lambda k:
                           (len(row_cols[k]), -min(local_costs[a] for a in row_cols[k])))
        for k in row_order:
            if clock() >= stop:
                return None
            delta = min(slack_local[a] for a in row_cols[k])
            dual[k] = delta
            for a in row_cols[k]:
                slack_local[a] -= delta
        positive_dual = [(1 << k, v) for k, v in enumerate(dual) if v > 0]
        all_mask = (1 << R) - 1
        stack = [(all_mask, 0, ())]
        visited = {}
        improved = False
        while stack and clock() < stop:
            missing, value, path = stack.pop()
            if value >= upper:
                continue
            if not missing:
                upper = value
                best_extra = {local_ids[a] for a in path}
                improved = True
                publish(kept | best_extra)
                continue
            if value + sum(v for bit, v in positive_dual if missing & bit) >= upper:
                continue
            old = visited.get(missing)
            if old is not None and old <= value:
                continue
            visited[missing] = value
            target = next(k for k in row_order if missing & (1 << k))
            options = []
            for a in row_cols[target]:
                new_value = value + local_costs[a]
                if new_value >= upper:
                    continue
                new_mask = missing & ~local_masks[a]
                if not new_mask:
                    upper = new_value
                    best_extra = {local_ids[b] for b in path}
                    best_extra.add(local_ids[a])
                    improved = True
                    publish(kept | best_extra)
                    continue
                bound = sum(v for bit, v in positive_dual if new_mask & bit)
                if new_value + bound >= upper:
                    continue
                gain = (missing ^ new_mask).bit_count()
                options.append((local_costs[a] / gain, new_mask, new_value, a))
            options.sort(reverse=True)
            for _, new_mask, new_value, a in options:
                stack.append((new_mask, new_value, path + (a,)))
        return kept | best_extra if improved else None

    # The main loop is controlled only by the wall-clock deadline; neighbourhood size, screening width and restart pace scale with the budget.
    while clock() < deadline:
        round_start = clock()
        if use_numpy:
            rc = cvector - np.bincount(ec, weights=multipliers[er], minlength=N)
            relaxed_bits = rc < 0.0
            relaxed_counts = np.bincount(er, weights=relaxed_bits[ec], minlength=M)
            lb = float(multipliers.sum() + np.minimum(rc, 0.0).sum())
            gradient = 1.0 - relaxed_counts
            norm = float(np.dot(gradient, gradient))
            changed = last_relaxed is None or not np.array_equal(relaxed_bits, last_relaxed)
            relaxed = set(map(int, np.flatnonzero(relaxed_bits))) if changed or norm == 0 else None
            last_relaxed = relaxed_bits
        else:
            rc = [costs[j] - sum(multipliers[i] for i in elements[j]) for j in range(N)]
            relaxed = {j for j in range(N) if rc[j] < 0.0}
            relaxed_counts = counts_for(relaxed, deadline)
            if relaxed_counts is None:
                break
            lb = sum(multipliers) + sum(min(v, 0.0) for v in rc)
            gradient = [1 - v for v in relaxed_counts]
            norm = sum(v * v for v in gradient)
            changed = relaxed != last_relaxed
            last_relaxed = relaxed.copy()
        if clock() >= deadline:
            break
        if lb > best_lb + 1e-8:
            best_lb = lb
            best_lambda = multipliers.copy()
            last_dual_improvement = clock()
        if norm == 0:
            publish(relaxed)
            return
        if best_lb >= best_cost - 1e-7:
            return
        dual_spent += clock() - round_start
        mean_lambda = max(1e-12, float(sum(multipliers)) / M)
        beta = intensity * rng.random()
        eta = intensity * rng.random()
        weights = [(1.0 - beta) + beta * float(v) / mean_lambda for v in multipliers]
        prices = [(1.0 - eta) * costs[j] + eta * max(0.0, float(rc[j])) for j in range(N)]
        noise = 0.3 * intensity * rng.random()
        prices = [v * (1.0 + noise * rng.random()) for v in prices]
        stage_start = clock()
        if local_spent >= dual_spent:
            if changed:
                result = repair(relaxed, weights, prices, deadline, noise)
                if result is not None:
                    publish(result[0])
            dual_spent += clock() - stage_start
        else:
            staleness = min(1.0, (clock() - last_improvement) / max(1e-6, budget * 0.15))
            if current_cost > best_cost * (1.0 + 0.025 * intensity) or rng.random() < 0.7:
                current, current_cost = set(best), best_cost
            selected = list(current)
            width = min(len(selected), max(1, int(math.ceil(
                math.sqrt(len(selected) * max(strength, 1e-6)) * (0.35 + staleness)))))
            remove_count = rng.randint(1, width)
            if rng.random() < 0.5:
                removed = set(rng.sample(selected, remove_count))
            else:
                selected.sort(key=lambda j: costs[j] * (0.2 + rng.random()), reverse=True)
                removed = set(selected[:remove_count])
            kept = current - removed
            result = repair(kept, weights, prices, deadline, noise)
            repaired = None if result is None else result[0]
            if repaired is not None:
                value = publish(repaired)
                if value <= best_cost * (1.0 + 0.025 * intensity * (0.25 + staleness)):
                    current, current_cost = set(repaired), value
            stop = min(deadline, clock() + budget * 0.012 * (1.0 + staleness))
            if clock() < stop:
                exact = exact_neighborhood(kept, removed, repaired, rc, stop)
                if exact is not None:
                    value = publish(exact)
                    count = counts_for(exact, deadline)
                    if count is not None:
                        exact, count = prune(exact, count, deadline)
                        value = publish(exact)
                    if value <= current_cost:
                        current, current_cost = set(exact), value
            local_spent += clock() - stage_start
        if clock() >= deadline:
            break
        step = alpha * max(0.0, best_cost - lb) / norm
        if use_numpy:
            multipliers = np.maximum(0.0, multipliers + step * gradient)
        else:
            multipliers = [max(0.0, v + step * g) for v, g in zip(multipliers, gradient)]
        now = clock()
        if now - last_adapt >= max(1e-5, budget * 0.025):
            if now - last_dual_improvement > budget * 0.025:
                alpha *= 0.65
            else:
                alpha = min(1.95, alpha * 1.05)
            if alpha < 0.075:
                perturb = [1.0 + 0.2 * intensity * (rng.random() - 0.5) for _ in range(M)]
                if use_numpy:
                    multipliers = best_lambda * np.asarray(perturb)
                else:
                    multipliers = [v * p for v, p in zip(best_lambda, perturb)]
                alpha = 1.8
            last_adapt = now
