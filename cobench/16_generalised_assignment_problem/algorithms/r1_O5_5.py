"""Relaxation-guided integer large neighbourhood search
Uses linear-relaxation capacity prices and reduced costs to guide global integer search and adaptive neighbourhood re-optimisation, together with feasible construction, move/swap and verification on the original data.
"""

import math
import random
import time

try:
    import numpy as np
    from scipy.optimize import Bounds, LinearConstraint, linprog, milp
    from scipy.sparse import csc_matrix, vstack
except Exception:
    np = None


def solve(instance: dict, time_limit_s: float) -> dict:
    start = time.monotonic()
    best = None
    best_value = math.inf
    emergency = []
    n = 0
    m = 0

    try:
        budget = max(0.0, float(time_limit_s))
        if not math.isfinite(budget):
            budget = 0.0
        deadline = start + 0.94 * budget
        m = int(instance["num_machines"])
        n = int(instance["num_jobs"])
        cost = instance["cost_matrix"]
        resource = instance["resource_matrix"]
        capacity = instance["capacities"]
        if n == 0:
            return {"assignments": {}}
        if m <= 0:
            return {"assignments": {str(j): 0 for j in range(n)}}

        clock = time.monotonic
        rng = random.Random(1000003 * m + 9176 * n + sum(capacity))
        nonnegative = all(min(row, default=0) >= 0 for row in resource)
        domains = [
            [i for i in range(m)
             if not nonnegative or resource[i][j] <= capacity[i]]
            for j in range(n)
        ]
        if not all(domains):
            domains = [list(range(m)) for _ in range(n)]
        ordered = [sorted(domains[j], key=lambda i: cost[i][j]) for j in range(n)]
        cheapest = [ordered[j][0] for j in range(n)]
        minimum = [cost[cheapest[j]][j] for j in range(n)]
        trivial_bound = sum(minimum)
        scale = max(1.0, sum(max(cost[i][j] for i in domains[j]) - minimum[j]
                             for j in range(n)) / n)
        emergency = [min(domains[j], key=lambda i: resource[i][j] / max(1, capacity[i]))
                     for j in range(n)]
        least_violation = math.inf
        prices = [0.0] * m

        def remember(assignment):
            nonlocal best, best_value, emergency, least_violation
            if assignment is None or len(assignment) != n:
                return False
            a = [int(i) for i in assignment]
            if any(i < 0 or i >= m for i in a):
                return False
            loads = [0] * m
            value = 0
            for j, i in enumerate(a):
                loads[i] += resource[i][j]
                value += cost[i][j]
            violation = sum(max(0, loads[i] - capacity[i]) for i in range(m))
            if violation < least_violation:
                least_violation = violation
                emergency = a
            if violation != 0:
                return False
            if value < best_value:
                best, best_value = a, value
                return True
            return False

        def construct(weight, noise, stop):
            remaining = list(capacity)
            a = [-1] * n
            pending = list(range(n))
            dynamic = n * n * m <= max(budget, 0.01) * 90000
            if not dynamic:
                ranking = []
                for j in pending:
                    vals = sorted(cost[i][j] + prices[i] * resource[i][j]
                                  for i in domains[j])
                    regret = vals[1] - vals[0] if len(vals) > 1 else scale * (m + 1)
                    hard = min(resource[i][j] / max(1, capacity[i]) for i in domains[j])
                    ranking.append((len(vals), -regret - scale * hard
                                    - noise * scale * rng.random(), j))
                pending = [item[2] for item in sorted(ranking)]
            while pending and clock() < stop:
                selected = None
                for pos, j in enumerate(pending if dynamic else pending[:1]):
                    if (pos & 63) == 0 and clock() >= stop:
                        return None
                    opts = []
                    for i in domains[j]:
                        r = resource[i][j]
                        if r <= remaining[i]:
                            pressure = r / max(1, remaining[i])
                            score = (cost[i][j] + prices[i] * r
                                     + weight * scale * pressure
                                     + noise * scale * rng.random())
                            opts.append((score, i))
                    if not opts:
                        return None
                    opts.sort()
                    regret = opts[1][0] - opts[0][0] if len(opts) > 1 else scale * (m + 1)
                    key = (len(opts), -regret)
                    if selected is None or key < selected[0]:
                        selected = (key, pos, j, opts[0][1])
                _, pos, j, i = selected
                a[j] = i
                remaining[i] -= resource[i][j]
                pending.pop(pos)
            return a if not pending else None

        def descent(base, stop):
            a = base[:]
            loads = [0] * m
            for j, i in enumerate(a):
                loads[i] += resource[i][j]
            while clock() < stop:
                moved = False
                for j in range(n):
                    if clock() >= stop:
                        remember(a)
                        return
                    old = a[j]
                    for i in ordered[j]:
                        if cost[i][j] >= cost[old][j]:
                            break
                        if (loads[i] + resource[i][j] <= capacity[i]
                                and loads[old] - resource[old][j] <= capacity[old]):
                            loads[old] -= resource[old][j]
                            loads[i] += resource[i][j]
                            a[j] = i
                            moved = True
                            break
                if moved:
                    remember(a)
                    continue
                for j in range(n):
                    if clock() >= stop:
                        remember(a)
                        return
                    i = a[j]
                    for k in range(j + 1, n):
                        if (k & 255) == 0 and clock() >= stop:
                            remember(a)
                            return
                        h = a[k]
                        if i == h:
                            continue
                        delta = cost[h][j] + cost[i][k] - cost[i][j] - cost[h][k]
                        if delta >= 0:
                            continue
                        li = loads[i] - resource[i][j] + resource[i][k]
                        lh = loads[h] - resource[h][k] + resource[h][j]
                        if li <= capacity[i] and lh <= capacity[h]:
                            a[j], a[k] = h, i
                            loads[i], loads[h] = li, lh
                            moved = True
                            break
                    if moved:
                        break
                remember(a)
                if not moved:
                    return

        def enumerate_subset(base, jobs, stop):
            # An explicit stack avoids recursion-depth errors when there are many jobs.
            if not jobs or clock() >= stop:
                return False
            jobs_set = set(jobs)
            remaining = list(capacity)
            if base is not None:
                for j, i in enumerate(base):
                    if j not in jobs_set:
                        remaining[i] -= resource[i][j]
            opts = {}
            for j in jobs:
                opts[j] = [i for i in ordered[j] if resource[i][j] <= remaining[i]]
                if not opts[j]:
                    return False
            jobs = sorted(jobs, key=lambda j: (
                len(opts[j]),
                -(cost[opts[j][1]][j] - cost[opts[j][0]][j])
                if len(opts[j]) > 1 else -scale * (m + 1)))
            size = len(jobs)
            lower = [0] * (size + 1)
            for p in range(size - 1, -1, -1):
                lower[p] = lower[p + 1] + cost[opts[jobs[p]][0]][jobs[p]]
            incumbent = sum(cost[base[j]][j] for j in jobs) if base is not None else best_value
            pointers = [0] * size
            choice = [-1] * size
            pos, value = 0, 0
            while pos >= 0 and clock() < stop:
                if pos == size:
                    incumbent = value
                    trial = base[:] if base is not None else [-1] * n
                    for p, j in enumerate(jobs):
                        trial[j] = choice[p]
                    remember(trial)
                    backtrack = True
                else:
                    backtrack = (value + lower[pos] >= incumbent
                                 or pointers[pos] >= len(opts[jobs[pos]]))
                if backtrack:
                    if pos < size:
                        pointers[pos] = 0
                    pos -= 1
                    if pos >= 0:
                        j, i = jobs[pos], choice[pos]
                        remaining[i] += resource[i][j]
                        value -= cost[i][j]
                    continue
                j = jobs[pos]
                i = opts[j][pointers[pos]]
                pointers[pos] += 1
                if (resource[i][j] <= remaining[i]
                        and value + cost[i][j] + lower[pos + 1] < incumbent):
                    choice[pos] = i
                    remaining[i] -= resource[i][j]
                    value += cost[i][j]
                    pos += 1
                    if pos < size:
                        pointers[pos] = 0
            return pos < 0

        remember(emergency)
        remember(cheapest)
        if best_value == trivial_bound:
            return {"assignments": {str(j): best[j] for j in range(n)}}
        opening_end = min(deadline, start + budget * 0.055)
        opening = 0
        while clock() < opening_end:
            weight = 0.25 if opening == 0 else 10.0 ** rng.uniform(-1.2, 1.5)
            remember(construct(weight, 0.0 if opening == 0 else 0.07, opening_end))
            opening += 1
            if best is not None:
                break
        if best is not None:
            descent(best, min(deadline, start + budget * 0.08))

        native = np is not None
        dual_bound = -math.inf
        reduced = None
        fractional = None
        global_proved = False
        if native and clock() < deadline:
            try:
                C = np.asarray(cost, dtype=float).T.copy()
                R = np.asarray(resource, dtype=float).T.copy()
                caps = np.asarray(capacity, dtype=float)
                eligible = np.ones((n, m), dtype=bool)
                if nonnegative:
                    eligible = R <= caps[None, :]
                rows, machines = np.nonzero(eligible)
                indices = np.arange(len(rows))
                aj = csc_matrix((np.ones(len(rows)), (rows, indices)), shape=(n, len(rows)))
                ar = csc_matrix((R[rows, machines], (machines, indices)), shape=(m, len(rows)))
                lp_time = min(budget * 0.13, (deadline - clock()) * 0.2)
                if lp_time > 0:
                    lp = linprog(C[rows, machines], A_ub=ar, b_ub=caps,
                                 A_eq=aj, b_eq=np.ones(n), bounds=(0.0, None),
                                 method="highs", options={"time_limit": lp_time,
                                                         "presolve": True})
                    if lp.status == 0 and lp.x is not None:
                        dual_bound = float(lp.fun)
                        prices = [max(0.0, -float(x)) for x in lp.ineqlin.marginals]
                        reduced = (C - np.asarray(lp.eqlin.marginals)[:, None]
                                   + R * np.asarray(prices)[None, :])
                        reduced = np.maximum(reduced, 0.0)
                        fractional = np.zeros((n, m))
                        fractional[rows, machines] = lp.x
                        remember(np.argmax(fractional, axis=1).tolist())
                del aj, ar
            except Exception:
                native = False

        priced_end = min(deadline, start + budget * 0.23)
        variant = 0
        while clock() < priced_end:
            weight = 0.0 if variant == 0 else 10.0 ** rng.uniform(-1.8, 0.4)
            improved = remember(construct(weight, 0.0 if variant == 0 else 0.06, priced_end))
            variant += 1
            if best is not None and (improved or variant > 1):
                descent(best, min(priced_end, clock() + budget * 0.018))
                break
        if best is not None and best_value < dual_bound + 1 - 1e-6:
            return {"assignments": {str(j): best[j] for j in range(n)}}

        def native_neighborhood(base, jobs, stop, width=None):
            # The subproblem contains only unfixed x[i][j]; the remaining assignments are substituted into residual capacities.
            if clock() >= stop or not jobs:
                return False, False
            old_best = best_value
            job_ids = np.asarray(jobs, dtype=int)
            size = len(jobs)
            residual = list(capacity)
            old_part = 0
            if base is not None:
                selected = set(jobs)
                for j, i in enumerate(base):
                    if j in selected:
                        old_part += cost[i][j]
                    else:
                        residual[i] -= resource[i][j]
            mask = eligible[job_ids].copy()
            if reduced is not None and best is not None:
                gap = max(0.0, best_value - dual_bound)
                tolerance = 1e-6 * max(1.0, abs(dual_bound), abs(best_value))
                mask &= reduced[job_ids] <= gap + tolerance
            if width is not None and width < m:
                key = reduced[job_ids].copy() if reduced is not None else C[job_ids].copy()
                key[~mask] = np.inf
                candidates = np.argpartition(key, width - 1, axis=1)[:, :width]
                narrow = np.zeros_like(mask)
                narrow[np.arange(size)[:, None], candidates] = True
                if fractional is not None:
                    narrow |= fractional[job_ids] > 1e-8
                mask &= narrow
            if base is not None:
                mask[np.arange(size), np.asarray(base, dtype=int)[job_ids]] = True
            local_jobs, machine_ids = np.nonzero(mask)
            if len(local_jobs) == 0 or np.any(mask.sum(axis=1) == 0):
                return False, False
            global_jobs = job_ids[local_jobs]
            variables = len(local_jobs)
            columns = np.arange(variables)
            objective = C[global_jobs, machine_ids]
            matrix = csc_matrix(
                (np.concatenate((np.ones(variables), R[global_jobs, machine_ids])),
                 (np.concatenate((local_jobs, size + machine_ids)),
                  np.concatenate((columns, columns)))), shape=(size + m, variables))
            lower = np.concatenate((np.ones(size), np.full(m, -np.inf)))
            upper = np.concatenate((np.ones(size), np.asarray(residual, dtype=float)))
            if base is not None:
                matrix = vstack((matrix, csc_matrix(objective.reshape(1, -1))), format="csc")
                lower = np.append(lower, -np.inf)
                upper = np.append(upper, float(old_part - 1))
            remaining_time = stop - clock()
            if remaining_time <= 0:
                return False, False
            try:
                result = milp(
                    objective, integrality=np.ones(variables, dtype=np.uint8),
                    bounds=Bounds(np.zeros(variables), np.ones(variables)),
                    constraints=LinearConstraint(matrix, lower, upper),
                    options={"time_limit": remaining_time * 0.88,
                             "presolve": True, "mip_rel_gap": 0.0})
            except Exception:
                return False, False
            valid = False
            if result.x is not None:
                a = base[:] if base is not None else [-1] * n
                picked = np.full(size, -1, dtype=int)
                selected_arcs = np.flatnonzero(result.x > 0.5)
                picked[local_jobs[selected_arcs]] = machine_ids[selected_arcs]
                if np.all(picked >= 0):
                    for p, j in enumerate(jobs):
                        a[j] = int(picked[p])
                    remember(a)
                    loads = [0] * m
                    for j, i in enumerate(a):
                        loads[i] += resource[i][j]
                    valid = all(loads[i] <= capacity[i] for i in range(m))
            exhausted = result.status == 2 or (result.status == 0 and valid)
            return best_value < old_best, exhausted

        if native and clock() < deadline:
            global_slice = budget * (0.31 if best is not None else 0.60)
            _, global_proved = native_neighborhood(
                best, list(range(n)), min(deadline, clock() + global_slice))
        if global_proved and best is not None:
            return {"assignments": {str(j): best[j] for j in range(n)}}
        if best is None:
            if native and clock() < deadline:
                native_neighborhood(None, list(range(n)),
                                    min(deadline, clock() + (deadline - clock()) * 0.75))
            if best is None and clock() < deadline:
                enumerate_subset(None, list(range(n)), deadline)
        if best is not None and clock() < deadline:
            descent(best, min(deadline, clock() + budget * 0.025))

        strength = math.log1p(budget)
        size = min(n, max(2, int((math.sqrt(n) + math.sqrt(m)) * (1 + strength)
                                 / math.sqrt(max(1, math.log2(m + 1))))))
        base_width = min(m, max(2, int(math.ceil(1 + strength))))
        stale = 0
        cycle = 0
        last_improvement = clock()
        while best is not None and clock() < deadline:
            if best_value == trivial_bound or best_value < dual_bound + 1 - 1e-6:
                break
            before = best_value
            base = best[:]
            mode = cycle % 4
            jobs = []
            seen = set()

            def include(j):
                if len(jobs) < size and j not in seen:
                    jobs.append(j)
                    seen.add(j)

            if mode == 0 and fractional is not None:
                ranked = sorted(range(n), key=lambda j: (
                    scale * (1 - fractional[j, base[j]])
                    + (float(reduced[j, base[j]]) if reduced is not None else 0.0)
                    + scale * 0.20 * rng.random()), reverse=True)
                for j in ranked[:max(1, 3 * size // 4)]:
                    include(j)
            elif mode == 1:
                ranked = sorted(range(n), key=lambda j:
                                cost[base[j]][j] - minimum[j] + scale * rng.random(),
                                reverse=True)
                pivot = ranked[0]
                source = base[pivot]
                alternatives = [i for i in ordered[pivot] if i != source]
                target = alternatives[int(rng.random() ** 2 * len(alternatives))] if alternatives else source
                include(pivot)
                related = [j for j in range(n) if base[j] in (source, target)]
                rng.shuffle(related)
                for j in related:
                    include(j)
            elif mode == 2:
                ranked = sorted(range(n), key=lambda j:
                                cost[base[j]][j] - minimum[j] + scale * rng.random(),
                                reverse=True)
                for j in ranked[:max(1, size // 2)]:
                    include(j)
                targets = set(ordered[j][0] for j in jobs)
                related = [j for j in range(n) if base[j] in targets]
                rng.shuffle(related)
                for j in related:
                    include(j)
            rest = [j for j in range(n) if j not in seen]
            rng.shuffle(rest)
            for j in rest:
                include(j)
            width = min(m, base_width + int(strength * (clock() - last_improvement)
                                             / max(budget * 0.12, 1e-9)))
            chunk = budget * (0.022 + 0.013 * math.log1p(stale))
            stop = min(deadline, clock() + chunk)
            if native:
                changed, exhausted = native_neighborhood(base, jobs, stop, width)
                if exhausted and len(jobs) == n and width == m:
                    break
                if exhausted:
                    size = min(n, max(size + 1, int(math.ceil(size * 1.20))))
                elif not changed:
                    size = max(2, min(n, int(size * 0.88)))
            else:
                exhausted = enumerate_subset(base, jobs, stop)
                if exhausted and len(jobs) == n:
                    break
                if exhausted:
                    size = min(n, size + max(1, int(strength)))
                else:
                    size = min(n, max(2, int(size * 0.85)))
            if best_value < before:
                stale = 0
                last_improvement = clock()
                descent(best, min(deadline, clock() + budget * 0.015))
            else:
                stale += 1
            cycle += 1
    except Exception:
        # The verified incumbent is unaffected by later numerical or solver exceptions.
        pass

    answer = best if best is not None else emergency
    return {"assignments": {str(j): int(answer[j]) if j < len(answer) else 0
                             for j in range(max(0, n))}}
