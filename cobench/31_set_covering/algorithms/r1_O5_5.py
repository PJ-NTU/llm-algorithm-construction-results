"""Hybrid bitset-greedy and integer-programming neighbourhood search
Improves coverage through global marginal greedy, time-limited integer programming, adaptive destroy-repair and residual branch-and-bound, keeping the best verified feasible solution.
"""

import heapq
import math
import random
import time


def solve(instance: dict, time_limit_s: float) -> dict:
    started = time.perf_counter()
    answer = {"selection_status": []}
    try:
        n = instance["n"]
        answer["selection_status"] = [1] * n
        m = instance["m"]
        costs = instance["costs"]
        budget = max(0.0, float(time_limit_s))
        deadline = started + budget * 0.97
        clock = time.perf_counter
        if m == 0:
            answer["selection_status"] = [int(c < 0) for c in costs]
            return answer
        if clock() >= deadline:
            return answer

        # First save the select-all fallback solution, then perform interruptible simplification.
        full = (1 << m) - 1
        masks = [0] * n
        locked = {j for j in range(n) if costs[j] <= 0}
        rows = []
        for i, raw in enumerate(instance["covers"]):
            if clock() >= deadline:
                return answer
            row = tuple(dict.fromkeys(raw))
            if not row:
                return answer
            rows.append(row)
            if len(row) == 1:
                locked.add(row[0])
            bit = 1 << i
            for p, j in enumerate(row):
                if p % 512 == 0 and clock() >= deadline:
                    return answer
                masks[j] |= bit

        locked = sorted(locked)
        base_mask = 0
        base_cost = 0
        for j in locked:
            base_mask |= masks[j]
            base_cost += costs[j]
        if base_mask == full:
            result = [0] * n
            for j in locked:
                result[j] = 1
            answer["selection_status"] = result
            return answer

        residual = full ^ base_mask
        representatives = {}
        for j in range(n):
            if clock() >= deadline:
                return answer
            if costs[j] <= 0:
                continue
            mask = masks[j] & residual
            if mask:
                old = representatives.get(mask)
                if old is None or costs[j] < costs[old]:
                    representatives[mask] = j
        active = list(representatives.values())
        if not active:
            return answer
        best_cost = sum(costs)
        best_free = active[:]
        last_improved = clock()
        rng = random.Random(0)
        strength = math.log1p(budget)

        def publish(free):
            nonlocal best_cost, best_free, last_improved
            value = base_cost
            covered = base_mask
            for j in free:
                value += costs[j]
                covered |= masks[j]
            if covered != full or value > best_cost:
                return False
            if value == best_cost and len(free) > len(best_free):
                return False
            improved = value < best_cost
            result = [0] * n
            for j in locked:
                result[j] = 1
            for j in free:
                result[j] = 1
            answer["selection_status"] = result
            best_free = free[:]
            best_cost = value
            if improved:
                last_improved = clock()
            return improved

        def prune(free):
            # Suffix unions and the kept prefix support reverse removal with a linear number of bit operations.
            original = free
            order = sorted(free, key=lambda j: costs[j], reverse=True)
            suffix = [0] * (len(order) + 1)
            for p in range(len(order) - 1, -1, -1):
                if clock() >= deadline:
                    return original
                suffix[p] = suffix[p + 1] | masks[order[p]]
            kept = []
            prefix = base_mask
            for p, j in enumerate(order):
                if clock() >= deadline:
                    return original
                if (prefix | suffix[p + 1]) != full:
                    kept.append(j)
                    prefix |= masks[j]
            return kept

        def greedy(kept, noise=0.0, power=1.0):
            covered = base_mask
            chosen = kept[:]
            for j in kept:
                covered |= masks[j]
            missing = full ^ covered
            if not missing:
                return prune(chosen)
            heap = []
            for j in active:
                if clock() >= deadline:
                    return None
                gain = (masks[j] & missing).bit_count()
                if gain:
                    price = costs[j]
                    if noise:
                        price *= math.exp(noise * (rng.random() - 0.5))
                    heap.append((price / gain ** power, costs[j], j, gain, price))
            heapq.heapify(heap)
            while missing and heap:
                if clock() >= deadline:
                    return None
                score, cost, j, old_gain, price = heapq.heappop(heap)
                gain = (masks[j] & missing).bit_count()
                if not gain:
                    continue
                if gain != old_gain:
                    heapq.heappush(heap, (price / gain ** power, cost, j, gain, price))
                    continue
                chosen.append(j)
                missing &= ~masks[j]
            return prune(chosen) if not missing else None

        publish(active)
        initial = greedy([])
        if initial is not None:
            publish(initial)
        if clock() >= deadline:
            return answer

        # Eliminate fixed variables; every remaining column still corresponds to an original x[j].
        # Search only strict improvements; integer costs make best_cost - 1 a valid objective bound.
        def integer_search():
            try:
                import numpy as np
                from scipy.optimize import Bounds, LinearConstraint, milp
                from scipy.sparse import csr_matrix

                if clock() >= deadline:
                    return
                mapping = {j: p for p, j in enumerate(active)}
                indices = []
                indptr = [0]
                for i, row in enumerate(rows):
                    if clock() >= deadline:
                        return
                    if (base_mask >> i) & 1:
                        continue
                    indices.extend(mapping[j] for j in row if j in mapping)
                    indptr.append(len(indices))
                row_count = len(indptr) - 1
                coefficients = np.asarray([costs[j] for j in active], dtype=float)
                nonzeros = len(indices)
                indices.extend(range(len(active)))
                indptr.append(len(indices))
                data = np.concatenate((np.ones(nonzeros), coefficients))
                matrix = csr_matrix((data, np.asarray(indices), np.asarray(indptr)),
                                    shape=(row_count + 1, len(active)))
                lower = np.ones(row_count + 1)
                upper = np.full(row_count + 1, np.inf)
                lower[-1] = -np.inf
                upper[-1] = best_cost - base_cost - 1
                remaining = deadline - clock()
                if remaining <= 0:
                    return
                result = milp(
                    coefficients,
                    integrality=np.ones(len(active), dtype=np.uint8),
                    bounds=Bounds(np.zeros(len(active)), np.ones(len(active))),
                    constraints=LinearConstraint(matrix, lower, upper),
                    options={"time_limit": remaining * 0.45,
                             "mip_rel_gap": 0.0, "presolve": True},
                )
                if result.x is not None:
                    candidate = [j for p, j in enumerate(active) if result.x[p] > 0.5]
                    # Any result from a floating-point backend must pass re-checks of the original coverage and integer cost.
                    if publish(candidate):
                        publish(prune(candidate))
            except Exception:
                # A missing backend, numerical failure or no feasible return does not affect the saved solution.
                return

        if deadline - clock() > budget / (2.0 + strength):
            integer_search()

        def exact_repair(kept, stop):
            """Wall-clock-limited branch-and-bound on the residual coverage outside the kept sets."""
            covered = base_mask
            fixed_cost = base_cost
            for j in kept:
                covered |= masks[j]
                fixed_cost += costs[j]
            missing = full ^ covered
            bound = best_cost - fixed_cost
            if not missing:
                publish(prune(kept))
                return
            if bound <= 0:
                return
            patterns = {}
            for j in active:
                if clock() >= stop:
                    return
                if costs[j] >= bound:
                    continue
                mask = masks[j] & missing
                if mask:
                    old = patterns.get(mask)
                    if old is None or costs[j] < costs[old]:
                        patterns[mask] = j
            options = {}
            conflicts = {}
            cheapest = {}
            for mask, j in patterns.items():
                todo = mask
                while todo:
                    if clock() >= stop:
                        return
                    bit = todo & -todo
                    todo ^= bit
                    options.setdefault(bit, []).append((j, mask))
                    conflicts[bit] = conflicts.get(bit, 0) | mask
                    cheapest[bit] = min(cheapest.get(bit, costs[j]), costs[j])
            todo = missing
            while todo:
                bit = todo & -todo
                todo ^= bit
                if bit not in options:
                    return
            order = sorted(options, key=lambda bit: (len(options[bit]), -cheapest[bit]))
            stack = [(missing, 0, ())]
            seen = {}
            while stack and clock() < stop:
                uncovered, spent, path = stack.pop()
                if spent >= bound:
                    continue
                if not uncovered:
                    candidate = kept + list(path)
                    if publish(candidate):
                        bound = best_cost - fixed_cost
                        if clock() < stop:
                            publish(prune(candidate))
                            bound = best_cost - fixed_cost
                    continue
                previous = seen.get(uncovered)
                if previous is not None and previous <= spent:
                    continue
                seen[uncovered] = spent
                # Mutually disjoint elements of the candidate set family must each pay their minimum covering cost.
                rest = uncovered
                dual_bound = 0
                target = None
                interrupted = False
                for bit in order:
                    if clock() >= stop:
                        interrupted = True
                        break
                    if target is None and (uncovered & bit):
                        target = bit
                    if rest & bit:
                        dual_bound += cheapest[bit]
                        if spent + dual_bound >= bound:
                            break
                        rest &= ~conflicts[bit]
                    if not rest:
                        break
                if interrupted:
                    return
                if spent + dual_bound >= bound:
                    continue
                branches = []
                for j, mask in options[target]:
                    if clock() >= stop:
                        return
                    if spent + costs[j] < bound:
                        gain = (mask & uncovered).bit_count()
                        branches.append((costs[j] / gain, j, mask))
                branches.sort(reverse=True)
                for _, j, mask in branches:
                    if clock() >= stop:
                        return
                    stack.append((uncovered & ~mask, spent + costs[j], path + (j,)))

        current = best_free[:]
        current_cost = base_cost + sum(costs[j] for j in current)
        epoch = budget / (2.0 + strength)
        next_restart = clock() + epoch
        while clock() < deadline:
            now = clock()
            stagnant = min(1.0, (now - last_improved) / max(epoch, 1e-9))
            if now >= next_restart:
                fresh = greedy([], 0.4 + stagnant, 0.8 + 0.4 * rng.random())
                if fresh is not None:
                    publish(fresh)
                current = best_free[:]
                current_cost = best_cost
                next_restart = clock() + epoch
            if clock() >= deadline or not current:
                break
            if rng.random() < 0.5:
                current = best_free[:]
                current_cost = best_cost
            count = len(current)
            # The longer the time limit or stagnation, the larger the reachable joint replacement size.
            reach = min(count, max(1, math.ceil(
                math.sqrt(count) * (1.0 + stagnant * strength))))
            destroy = rng.randint(1, reach)
            if rng.random() < 0.5:
                removed = set(rng.sample(current, destroy))
            else:
                anchor = rng.choice(current)
                related = sorted(current, key=lambda j: (
                    -(masks[j] & masks[anchor]).bit_count() * (0.5 + rng.random()),
                    -costs[j]))
                removed = set(related[:destroy])
            kept = [j for j in current if j not in removed]
            candidate = greedy(kept, 0.15 + stagnant * rng.random(),
                               0.9 + 0.2 * rng.random())
            if candidate is None:
                break
            candidate_cost = base_cost + sum(costs[j] for j in candidate)
            publish(candidate)
            temperature = (max(1, best_cost - base_cost) / max(1, len(best_free)))
            temperature *= (0.05 + 0.2 * stagnant) * max(
                0.0, (deadline - clock()) / max(budget, 1e-9))
            delta = candidate_cost - current_cost
            if delta <= 0 or (temperature > 0 and rng.random() < math.exp(
                    -min(700.0, delta / temperature))):
                current = candidate
                current_cost = candidate_cost
            if clock() < deadline:
                remaining = deadline - clock()
                fraction = (0.02 + 0.08 * strength / (1.0 + strength))
                fraction *= math.sqrt(destroy / count)
                exact_repair(kept, min(deadline, clock() + remaining * fraction))
        return answer
    except Exception:
        return answer
