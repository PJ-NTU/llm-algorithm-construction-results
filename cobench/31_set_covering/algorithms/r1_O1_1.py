"""Lagrangian-guided adaptive large neighbourhood set covering algorithm
Combines bitset greedy, redundancy removal, set swaps and time-limited branch-and-bound repair, continually saving the best feasible solution.
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
        margin = min(0.12, max(0.001, budget * 0.025))
        end = start + max(0.0, budget - margin)
        clock = time.perf_counter
        costs0 = instance['costs']
        covers = instance['covers']
        m = instance['m']
        if m == 0:
            return {'selection_status': [int(c <= 0) for c in costs0]}
        if clock() >= end:
            return {'selection_status': answer}

        # Sets with non-positive cost are always kept; a set that is the sole cover of some row is mandatory.
        fixed = {j for j in range(n) if costs0[j] <= 0}
        raw_masks = [0] * n
        raw_cols = [[] for _ in range(n)]
        for i, row in enumerate(covers):
            choices = set(row)
            if len(choices) == 1:
                fixed.update(choices)
            bit = 1 << i
            for j in choices:
                raw_masks[j] |= bit
                raw_cols[j].append(i)
            if (i & 127) == 0 and clock() >= end:
                return {'selection_status': answer}
        fixed_mask = 0
        for j in fixed:
            fixed_mask |= raw_masks[j]
        full = ((1 << m) - 1) & ~fixed_mask
        if not full:
            return {'selection_status': [int(j in fixed) for j in range(n)]}

        # Among identical residual coverages keep only the cheapest column; the optimal value is unchanged.
        representatives = {}
        for j in range(n):
            mask = raw_masks[j] & full
            if mask and j not in fixed:
                old = representatives.get(mask)
                if old is None or costs0[j] < costs0[old]:
                    representatives[mask] = j
            if (j & 255) == 0 and clock() >= end:
                return {'selection_status': answer}
        originals = list(representatives.values())
        masks = [raw_masks[j] & full for j in originals]
        costs = [costs0[j] for j in originals]
        count = len(originals)
        rows = [[] for _ in range(m)]
        cols = []
        for j, original in enumerate(originals):
            col = [i for i in raw_cols[original] if not (fixed_mask >> i) & 1]
            cols.append(col)
            for i in col:
                rows[i].append(j)
            if (j & 127) == 0 and clock() >= end:
                return {'selection_status': answer}
        row_ids = [i for i in range(m) if rows[i]]
        if len(row_ids) != full.bit_count():
            # For infeasible input no return value satisfies the covering constraints; keep the select-all fallback.
            return {'selection_status': answer}
        del raw_masks, raw_cols, representatives
        rng = random.Random(20261002 + 17 * n + m)
        edges = sum(map(len, cols))
        mean_degree = edges / max(1, len(row_ids))
        strength = math.log1p(budget)
        pool_limit = max(2, int(2 + strength))
        best = list(range(count))
        best_cost = sum(costs)
        elites = []
        last_gain = clock()

        def publish(selection):
            nonlocal best, best_cost, answer, last_gain
            value = sum(costs[j] for j in selection)
            if value < best_cost or not elites:
                best = list(selection)
                best_cost = value
                result = [0] * n
                for j in fixed:
                    result[j] = 1
                for j in best:
                    result[originals[j]] = 1
                answer = result
                last_gain = clock()
            if value <= best_cost * 1.08:
                key = frozenset(selection)
                if all(key != e[2] for e in elites):
                    elites.append((value, list(selection), key))
                    elites.sort(key=lambda e: e[0])
                    if len(elites) > pool_limit:
                        elites.pop()
            return value

        def union(selection):
            result = 0
            for j in selection:
                result |= masks[j]
            return result

        def prune(selection, stop, extra=0):
            order = sorted(selection, key=lambda j: costs[j], reverse=True)
            suffix = [0] * (len(order) + 1)
            for p in range(len(order) - 1, -1, -1):
                suffix[p] = suffix[p + 1] | masks[order[p]]
            kept = []
            covered = extra
            for p, j in enumerate(order):
                if (p & 63) == 0 and clock() >= stop:
                    return kept + order[p:]
                if masks[j] & ~(covered | suffix[p + 1]):
                    kept.append(j)
                    covered |= masks[j]
            return kept

        def repair(missing, stop, noise=0.0, guide=None, cap=math.inf):
            if not missing:
                return []
            if missing.bit_count() * mean_degree < count:
                candidates = set()
                scan = missing
                while scan:
                    bit = scan & -scan
                    scan ^= bit
                    candidates.update(rows[bit.bit_length() - 1])
                    if clock() >= stop:
                        return None
            else:
                candidates = range(count)
            heap = []
            for p, j in enumerate(candidates):
                if (p & 127) == 0 and clock() >= stop:
                    return None
                if costs[j] >= cap:
                    continue
                gain = (masks[j] & missing).bit_count()
                if gain:
                    weight = costs[j] if guide is None else guide[j]
                    if noise:
                        weight *= math.exp(noise * (rng.random() - 0.5))
                    heap.append((weight / gain, costs[j], j, weight))
            heapq.heapify(heap)
            chosen = []
            total = 0
            while missing and heap:
                if clock() >= stop:
                    return None
                score, cost, j, weight = heapq.heappop(heap)
                gain = (masks[j] & missing).bit_count()
                if not gain:
                    continue
                new_score = weight / gain
                if heap and new_score > heap[0][0] + 1e-12:
                    heapq.heappush(heap, (new_score, cost, j, weight))
                    continue
                total += cost
                if total >= cap:
                    return None
                chosen.append(j)
                missing &= ~masks[j]
            return chosen if not missing else None

        # First obtain a compact feasible fallback, then construct by global marginal coverage ratio.
        initial = set()
        for i in row_ids:
            initial.add(min(rows[i], key=lambda j: costs[j]))
            if clock() >= end:
                return {'selection_status': answer}
        publish(prune(initial, end))
        greedy = repair(full, end)
        if greedy is not None:
            publish(prune(greedy, end))
        if clock() >= end:
            return {'selection_status': answer}

        def exchange(selection, stop):
            current = list(selection)
            value = sum(costs[j] for j in current)
            while clock() < stop:
                suffix = [0] * (len(current) + 1)
                for p in range(len(current) - 1, -1, -1):
                    suffix[p] = suffix[p + 1] | masks[current[p]]
                prefix = 0
                savings = {}
                selected = set(current)
                for p, j in enumerate(current):
                    unique = masks[j] & ~(prefix | suffix[p + 1])
                    prefix |= masks[j]
                    if not unique:
                        continue
                    scan = unique
                    rare = -1
                    degree = count + 1
                    while scan:
                        bit = scan & -scan
                        scan ^= bit
                        i = bit.bit_length() - 1
                        if len(rows[i]) < degree:
                            rare, degree = i, len(rows[i])
                    for q, k in enumerate(rows[rare]):
                        if k not in selected and not unique & ~masks[k]:
                            savings[k] = savings.get(k, 0) + costs[j]
                        if (q & 127) == 0 and clock() >= stop:
                            return current
                proposals = [j for j, saving in savings.items() if saving > costs[j]]
                proposals.sort(key=lambda j: savings[j] - costs[j], reverse=True)
                improved = False
                for j in proposals:
                    if clock() >= stop:
                        return current
                    trial = prune(current, stop, masks[j]) + [j]
                    new_value = sum(costs[k] for k in trial)
                    if new_value < value:
                        current, value = trial, new_value
                        publish(current)
                        improved = True
                        break
                if not improved:
                    break
            return current

        def exact_repair(missing, incumbent, stop):
            """Fix the sets not removed and run a time-limited search with dual lower bounds on the residual coverage."""
            incumbent = list(incumbent)
            bound = sum(costs[j] for j in incumbent)
            if not missing or clock() >= stop:
                return incumbent, False
            projected = {}
            if missing.bit_count() * mean_degree < count:
                candidate_ids = set()
                scan = missing
                while scan:
                    bit = scan & -scan
                    scan ^= bit
                    candidate_ids.update(rows[bit.bit_length() - 1])
                    if clock() >= stop:
                        return incumbent, False
            else:
                candidate_ids = range(count)
            for p, j in enumerate(candidate_ids):
                if (p & 63) == 0 and clock() >= stop:
                    return incumbent, False
                if costs[j] >= bound:
                    continue
                mask = masks[j] & missing
                if mask:
                    old = projected.get(mask)
                    if old is None or costs[j] < costs[old]:
                        projected[mask] = j
            cm = list(projected)
            ids = list(projected.values())
            cc = [costs[j] for j in ids]
            local_rows = {}
            for j, mask in enumerate(cm):
                scan = mask
                while scan:
                    bit = scan & -scan
                    scan ^= bit
                    local_rows.setdefault(bit, []).append(j)
                if (j & 31) == 0 and clock() >= stop:
                    return incumbent, False
            coverable = 0
            for bit in local_rows:
                coverable |= bit
            if coverable != missing:
                return incumbent, True
            ordered_rows = sorted(local_rows, key=lambda bit: len(local_rows[bit]))

            # Build a feasible dual of the covering linear relaxation, then use the remaining slack for dual ascent.
            price = {}
            ratio = [cc[j] / cm[j].bit_count() for j in range(len(cm))]
            for bit in ordered_rows:
                price[bit] = min(ratio[j] for j in local_rows[bit])
            slack = []
            for j, mask in enumerate(cm):
                scan = mask
                value = 0.0
                while scan:
                    bit = scan & -scan
                    scan ^= bit
                    value += price[bit]
                slack.append(max(0.0, cc[j] - value))
                if (j & 31) == 0 and clock() >= stop:
                    return incumbent, False
            for bit in ordered_rows:
                adjacent = local_rows[bit]
                delta = min(slack[j] for j in adjacent)
                if delta > 0:
                    price[bit] += delta
                    for j in adjacent:
                        slack[j] = max(0.0, slack[j] - delta)
                if clock() >= stop:
                    return incumbent, False

            def dual_weight(mask):
                value = 0.0
                while mask:
                    bit = mask & -mask
                    mask ^= bit
                    value += price[bit]
                return value

            # Scale down slightly, so floating-point rounding does not misjudge valid branches as non-improving.
            for bit in price:
                price[bit] *= 1.0 - 1e-10
            root_lb = sum(price.values())
            stack = [(missing, 0, (), root_lb)]
            seen = {missing: 0}
            while stack and clock() < stop:
                todo, spent, path, lower = stack.pop()
                if spent > seen.get(todo, math.inf) or spent + lower >= bound - 1e-8:
                    continue
                gap = bound - spent
                branches = None
                for bit in ordered_rows:
                    if todo & bit:
                        available = [j for j in local_rows[bit] if cc[j] < gap]
                        if not available:
                            branches = []
                            break
                        if branches is None or len(available) < len(branches):
                            branches = available
                            if len(branches) == 1:
                                break
                    if clock() >= stop:
                        return incumbent, False
                if not branches:
                    continue
                children = []
                for j in branches:
                    if clock() >= stop:
                        return incumbent, False
                    left = todo & ~cm[j]
                    new_spent = spent + cc[j]
                    if not left:
                        if new_spent < bound:
                            bound = new_spent
                            incumbent = [ids[k] for k in path] + [ids[j]]
                        continue
                    if new_spent >= seen.get(left, math.inf):
                        continue
                    gain = todo ^ left
                    removed_price = dual_weight(gain)
                    new_lower = max(0.0, lower - removed_price)
                    if new_spent + new_lower >= bound - 1e-8:
                        continue
                    seen[left] = new_spent
                    priority = (cc[j] - removed_price) / gain.bit_count()
                    children.append((priority, left, new_spent, path + (j,), new_lower))
                children.sort(key=lambda item: item[0], reverse=True)
                for _, left, new_spent, new_path, new_lower in children:
                    stack.append((left, new_spent, new_path, new_lower))
            return incumbent, not stack

        exchange_end = min(end, clock() + budget * 0.025)
        exchange(best, exchange_end)
        # Small search trees directly attempt a full-problem proof; the size criterion varies with the time budget.
        if len(best) * math.log1p(mean_degree) < math.log1p(budget * 100000):
            exact, proven = exact_repair(full, best, min(end, clock() + budget * 0.06))
            publish(prune(exact, end))
            if proven:
                return {'selection_status': answer}

        # Initial Lagrangian multipliers satisfy dual feasibility and are then updated by projected subgradient.
        multipliers = [0.0] * m
        for i in row_ids:
            multipliers[i] = min(costs[j] / len(cols[j]) for j in rows[i])
        initial_multipliers = multipliers[:]
        best_multipliers = multipliers[:]
        lower_bound = -math.inf
        last_dual_gain = clock()
        guide = costs[:]
        phase_start = clock()
        dual_time = 0.0
        dual_started = False
        current = list(best)
        current_cost = best_cost
        last_exchange_cost = math.inf

        def lagrangian(stop):
            nonlocal lower_bound, best_multipliers, last_dual_gain, guide
            reduced = [0.0] * count
            counts = [0] * m
            selected = []
            covered = 0
            lb = sum(multipliers)
            for j in range(count):
                if (j & 63) == 0 and clock() >= stop:
                    return
                rc = costs[j] - sum(multipliers[i] for i in cols[j])
                reduced[j] = rc
                if rc < 0:
                    lb += rc
                    selected.append(j)
                    covered |= masks[j]
                    for i in cols[j]:
                        counts[i] += 1
            now = clock()
            if lb > lower_bound + 1e-7:
                lower_bound = lb
                best_multipliers = multipliers[:]
                last_dual_gain = now
            guide = [max(costs[j] * 0.12, costs[j] + reduced[j] * 0.75)
                     for j in range(count)]
            added = repair(full & ~covered, stop)
            if added is not None:
                publish(prune(selected + added, stop))
            norm = sum((1 - counts[i]) ** 2 for i in row_ids)
            stale = (clock() - last_dual_gain) / max(0.001, budget * 0.15)
            if norm:
                alpha = 1.9 / (1.0 + 3.0 * stale)
                step = alpha * max(0.0, best_cost - lb) / norm
                for i in row_ids:
                    multipliers[i] = max(0.0, multipliers[i] + step * (1 - counts[i]))
            else:
                for i in row_ids:
                    multipliers[i] *= 0.85 + 0.3 * rng.random()
            if stale > 1.0:
                for i in row_ids:
                    multipliers[i] = max(0.0, best_multipliers[i] * (0.8 + 0.4 * rng.random())
                                         + initial_multipliers[i] * rng.random() * 0.1)
                last_dual_gain = clock()

        # The number of searches is unlimited; neighbourhood size and each exact-repair budget are controlled by time, size and stagnation.
        while clock() < end:
            now = clock()
            if not dual_started or dual_time < 0.28 * (now - phase_start):
                dual_started = True
                before = clock()
                lagrangian(end)
                dual_time += clock() - before
                if lower_bound >= best_cost - 1e-7:
                    break
                continue
            if best_cost < last_exchange_cost:
                last_exchange_cost = best_cost
                exchange(best, min(end, now + budget * 0.015))
                current, current_cost = list(best), best_cost
                continue
            stagnation = min(1.0, (now - last_gain) / max(0.001, budget * 0.16))
            if rng.random() < 0.35:
                source = list(best)
            elif elites and rng.random() < 0.25:
                source = list(rng.choice(elites)[1])
            else:
                source = list(current)
            if not source:
                break
            k = len(source)
            max_destroy = min(k, max(1, int(1 + math.sqrt(k) * (1 + strength))))
            destroy = 1 + int((max_destroy - 1) * rng.random() ** 1.5
                              * (0.25 + 0.75 * stagnation))
            mode = rng.random()
            if mode < 0.15 * stagnation:
                removed = set(source)
            elif mode < 0.45:
                removed = set(rng.sample(source, min(destroy, k)))
            elif mode < 0.70:
                order = sorted(source, key=lambda j: costs[j] * (0.25 + rng.random()), reverse=True)
                removed = set(order[:destroy])
            else:
                anchor = rng.choice(source)
                order = sorted(source, key=lambda j: (masks[j] & masks[anchor]).bit_count()
                               / max(1, len(cols[j])) + 0.35 * rng.random(), reverse=True)
                removed = set(order[:destroy])
            retained = [j for j in source if j not in removed]
            missing = full & ~union(retained)
            seed = list(removed)
            seed_cost = sum(costs[j] for j in seed)
            noise = (0.1 + 0.9 * stagnation) * rng.random()
            use_guide = guide if rng.random() < 0.45 else None
            # Greedy first gives a feasible repair; a few uphill moves are allowed, but the global best never regresses.
            candidate = repair(missing, end, noise, use_guide,
                               seed_cost + best_cost * (0.01 + 0.04 * stagnation))
            trial = None
            if candidate is not None:
                trial = prune(retained + candidate, end)
                trial_cost = publish(trial)
                candidate_cost = sum(costs[j] for j in candidate)
                if candidate_cost < seed_cost:
                    seed, seed_cost = candidate, candidate_cost
            remaining = end - clock()
            if remaining <= 0:
                break
            slice_size = (0.0015 + 0.0025 * strength) * (1.0 + math.sqrt(destroy))
            slice_size *= 1.0 + 2.0 * stagnation
            exact_stop = min(end, clock() + min(remaining * 0.15, slice_size))
            improved, proven = exact_repair(missing, seed, exact_stop)
            improved_trial = prune(retained + improved, end)
            improved_cost = publish(improved_trial)
            if trial is None or improved_cost <= trial_cost:
                trial, trial_cost = improved_trial, improved_cost
            if not retained and proven:
                break
            temperature = max(1e-12, best_cost * (0.001 + 0.012 * stagnation))
            if trial_cost <= current_cost or rng.random() < math.exp(
                    -min(700.0, (trial_cost - current_cost) / temperature)):
                current, current_cost = trial, trial_cost
            if current_cost > best_cost * (1.03 + 0.04 * stagnation):
                current, current_cost = list(best), best_cost
    except Exception:
        # No recoverable failure during search can damage the complete feasible solution already saved.
        pass
    return {'selection_status': answer}
