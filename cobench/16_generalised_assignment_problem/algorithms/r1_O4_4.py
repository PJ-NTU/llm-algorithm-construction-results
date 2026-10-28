"""Relaxation-guided capacity-linked large neighbourhood search
Combines regret construction, migration-swap and two-job ejection chains, guiding time-limited integer-programming re-optimisation with linear-relaxation pricing, and keeps the verified best feasible assignment.
"""

import math
import random
import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp
from scipy.sparse import csc_matrix


def solve(instance: dict, time_limit_s: float) -> dict:
    started = time.perf_counter()
    best = None
    backup = None
    best_cost = math.inf
    backup_key = (math.inf, math.inf)
    n = 0
    m = 0

    try:
        budget = max(0.0, float(time_limit_s))
        deadline = started + 0.94 * budget
        m = int(instance["num_machines"])
        n = int(instance["num_jobs"])
        costs = instance["cost_matrix"]
        resources = instance["resource_matrix"]
        capacities = [int(v) for v in instance["capacities"]]
        if n == 0:
            return {"assignments": {}}
        if m <= 0:
            return {"assignments": {str(j): 0 for j in range(n)}}

        c = np.asarray(costs, dtype=np.float64).T.copy()
        r = np.asarray(resources, dtype=np.float64).T.copy()
        cap = np.asarray(capacities, dtype=np.float64)
        jobs = np.arange(n)
        nonnegative = bool(np.all(r >= 0) and np.all(cap >= 0))
        eligible = (r <= cap[None, :]) if nonnegative else np.ones((n, m), dtype=bool)
        safe_cap = np.maximum(1.0, cap)
        seed = n * 9176 + m * 1000003 + sum(capacities)
        rng = random.Random(seed)
        strength = 1.0 + math.log1p(budget)
        cost_span = max(1.0, float(np.max(c) - np.min(c)))
        prices = np.zeros(m)
        lp_choice = None
        fractionality = np.zeros(n)
        best_loads = None
        last_improvement = started
        raw_min = np.min(np.where(eligible, c, np.inf), axis=1)
        lower_bound = float(np.sum(raw_min))
        base_size = min(n, max(2, int((4.0 + math.sqrt(n)) * strength)))
        width = min(m, max(2, int(strength)))

        def pack(a):
            return {"assignments": {str(j): int(a[j]) for j in range(n)}}

        def loads_of(a):
            loads = [0] * m
            for j, i in enumerate(a):
                loads[i] += int(resources[i][j])
            return loads

        def consider(a):
            nonlocal best, best_loads, best_cost, backup, backup_key, last_improvement
            if a is None or len(a) != n or any(i < 0 or i >= m for i in a):
                return False
            a = [int(i) for i in a]
            loads = loads_of(a)
            value = sum(int(costs[i][j]) for j, i in enumerate(a))
            excess = sum(max(0, loads[i] - capacities[i]) for i in range(m))
            if (excess, value) < backup_key:
                backup, backup_key = a[:], (excess, value)
            if excess == 0 and value < best_cost:
                best, best_loads, best_cost = a[:], loads, value
                last_improvement = time.perf_counter()
                return True
            return False

        def construct(stop, weight, noisy=False):
            """Dynamic fewest-choices-first, breaking ties by regret and capacity pressure."""
            a = [-1] * n
            remaining = np.arange(n)
            free = cap.copy()
            score0 = c + r * prices
            if noisy:
                noise = np.random.default_rng(rng.getrandbits(63))
                score0 = score0 + noise.random((n, m)) * cost_span / strength
            while len(remaining) and time.perf_counter() < stop:
                rr = r[remaining]
                fits = eligible[remaining] & (rr <= free[None, :])
                counts = np.sum(fits, axis=1)
                if np.any(counts == 0):
                    return None
                score = score0[remaining] + weight * cost_span * (
                    rr / safe_cap[None, :] + rr / np.maximum(1.0, free)[None, :]
                )
                score = np.where(fits, score, np.inf)
                first = np.argmin(score, axis=1)
                minimum = score[np.arange(len(remaining)), first]
                if m > 1:
                    second = np.partition(score, 1, axis=1)[:, 1]
                    regret = np.where(counts == 1, cost_span * (n + 1), second - minimum)
                else:
                    regret = np.full(len(remaining), cost_span)
                difficulty = np.min(np.where(fits, rr / np.maximum(1.0, free), np.inf), axis=1)
                priority = regret + cost_span * difficulty
                allowed = counts == np.min(counts)
                pos = int(np.argmax(np.where(allowed, priority, -np.inf)))
                j = int(remaining[pos])
                i = int(first[pos])
                a[j] = i
                free[i] -= r[j, i]
                remaining = np.delete(remaining, pos)
            return a if not len(remaining) else None

        def descend(a, stop):
            """Vectorised migration, swaps and two-job ejection chains, keeping each feasible improvement in turn."""
            a = np.asarray(a, dtype=np.int64).copy()
            loads = np.asarray(loads_of(a), dtype=np.float64)
            while time.perf_counter() < stop:
                current = c[jobs, a]
                slack = cap - loads
                delta = c - current[:, None]
                fits = eligible & (r <= slack[None, :])
                fits[jobs, a] = False
                fits &= ((loads[a] - r[jobs, a]) <= cap[a])[:, None]
                move_cost = np.where(fits, delta, np.inf)
                flat = int(np.argmin(move_cost))
                j, b = divmod(flat, m)
                if move_cost[j, b] < -1e-8:
                    old = int(a[j])
                    loads[old] -= r[j, old]
                    loads[b] += r[j, b]
                    a[j] = b
                    consider(a.tolist())
                    continue

                chosen = None
                gain = -1e-8
                for j in range(n):
                    if time.perf_counter() >= stop:
                        return
                    old = int(a[j])
                    ds = c[j, a] - current[j] + c[:, old] - current
                    la = loads[old] - r[j, old] + r[:, old]
                    lb = loads[a] - r[jobs, a] + r[j, a]
                    valid = (jobs > j) & (a != old) & (la <= cap[old]) & (lb <= cap[a])
                    values = np.where(valid, ds, np.inf)
                    k = int(np.argmin(values))
                    if values[k] < gain:
                        gain = float(values[k])
                        chosen = (j, k)
                if chosen is not None:
                    j, k = chosen
                    a[j], a[k] = a[k], a[j]
                    loads = np.asarray(loads_of(a), dtype=np.float64)
                    consider(a.tolist())
                    continue

                # First cache the cheapest feasible exit of each ejected job, avoiding triple enumeration.
                exits = np.argmin(move_cost, axis=1)
                exit_delta = move_cost[jobs, exits]
                chosen = None
                gain = -1e-8
                for j in range(n):
                    if time.perf_counter() >= stop:
                        return
                    old = int(a[j])
                    lb = loads[a] - r[jobs, a] + r[j, a]
                    la = loads[old] - r[j, old] + np.where(exits == old, r[:, old], 0.0)
                    ds = c[j, a] - current[j] + exit_delta
                    valid = (a != old) & (lb <= cap[a]) & (la <= cap[old])
                    values = np.where(valid, ds, np.inf)
                    k = int(np.argmin(values))
                    if values[k] < gain:
                        gain = float(values[k])
                        chosen = (j, k, int(a[k]), int(exits[k]))
                if chosen is None:
                    return
                j, k, target, exit_machine = chosen
                a[j], a[k] = target, exit_machine
                loads = np.asarray(loads_of(a), dtype=np.float64)
                consider(a.tolist())

        def model(subset, base):
            """Release only the original binary assignment variables in the subset; all other variables are fixed."""
            subset = np.asarray(subset, dtype=np.int64)
            q = len(subset)
            remaining = cap.copy()
            old_value = None
            if base is not None:
                fixed = loads_of(base)
                if all(fixed[i] <= capacities[i] for i in range(m)):
                    old_value = sum(int(costs[base[j]][j]) for j in subset)
                for j in subset:
                    fixed[base[j]] -= int(resources[base[j]][j])
                remaining -= np.asarray(fixed, dtype=np.float64)
            possible = eligible[subset].copy()
            if nonnegative:
                possible &= r[subset] <= remaining[None, :]
            if np.any(np.sum(possible, axis=1) == 0):
                return None
            local_job, machine = np.nonzero(possible)
            orig_job = subset[local_job]
            count = len(machine)
            columns = np.arange(count, dtype=np.int32)
            rows = np.concatenate((local_job, q + machine)).astype(np.int32)
            cols = np.concatenate((columns, columns))
            data = np.concatenate((np.ones(count), r[orig_job, machine]))
            matrix = csc_matrix((data, (rows, cols)), shape=(q + m, count))
            shift = np.min(np.where(possible, c[subset], np.inf), axis=1)
            objective = c[orig_job, machine] - shift[local_job]
            return (subset, local_job, machine, matrix, objective, remaining,
                    float(np.sum(shift)), old_value)

        def integer_search(subset, base, stop, full=False):
            """Time-limited exact neighbourhood; even on reaching the time limit only complete feasible solutions are accepted."""
            if time.perf_counter() >= stop:
                return False
            built = model(subset, base)
            if built is None:
                return False
            subset, local_job, machine, matrix, objective, remaining, shift, old_value = built
            q = len(subset)
            lower = np.concatenate((np.ones(q), np.full(m, -np.inf)))
            upper = np.concatenate((np.ones(q), remaining))
            constraints = [LinearConstraint(matrix, lower, upper)]
            if old_value is not None:
                constraints.append(LinearConstraint(
                    csc_matrix(objective.reshape(1, -1)), -np.inf, old_value - shift + 1e-7
                ))
            allowance = stop - time.perf_counter()
            if allowance <= 0:
                return False
            try:
                result = milp(
                    objective, integrality=np.ones(len(objective), dtype=np.uint8),
                    bounds=Bounds(0.0, 1.0), constraints=constraints,
                    options={"time_limit": allowance, "mip_rel_gap": 0.0, "presolve": True},
                )
            except Exception:
                return False
            accepted_result = False
            if result.x is not None:
                a = list(base) if base is not None else [-1] * n
                selected = np.flatnonzero(result.x > 0.5)
                covered = set()
                for p in selected:
                    j = int(subset[local_job[p]])
                    a[j] = int(machine[p])
                    covered.add(j)
                if len(covered) == q:
                    consider(a)
                    # Checked independently; a rounded result that fails the check cannot be treated as a proof of optimality.
                    accepted_result = all(v >= 0 for v in a) and all(
                        x <= capacities[i] for i, x in enumerate(loads_of(a))
                    )
            return bool(full and result.status == 0 and accepted_result)

        def feasibility_dfs(stop):
            """Iterative backtracking rescue, avoiding dependence on recursion depth."""
            if not nonnegative:
                return
            domains = [np.flatnonzero(eligible[j]).tolist() for j in range(n)]
            if any(not d for d in domains):
                return
            order = sorted(range(n), key=lambda j: (
                len(domains[j]), -min(r[j, i] / safe_cap[i] for i in domains[j])
            ))
            a = [-1] * n
            load = [0] * m
            stack = []
            pos = 0
            while time.perf_counter() < stop:
                if pos == n:
                    consider(a)
                    return
                j = order[pos]
                if len(stack) == pos:
                    options = [i for i in domains[j] if load[i] + int(resources[i][j]) <= capacities[i]]
                    options.sort(key=lambda i: (
                        (load[i] + int(resources[i][j])) / max(1, capacities[i]), costs[i][j]
                    ))
                    stack.append([options, 0])
                options, at = stack[pos]
                if at < len(options):
                    i = options[at]
                    stack[pos][1] += 1
                    a[j] = i
                    load[i] += int(resources[i][j])
                    pos += 1
                else:
                    stack.pop()
                    if pos == 0:
                        return
                    pos -= 1
                    old_job = order[pos]
                    i = a[old_job]
                    load[i] -= int(resources[i][old_job])
                    a[old_job] = -1

        # Keep two complete candidates first; only a candidate that truly satisfies capacity can become the best solution.
        consider(np.argmin(np.where(eligible, c, np.inf), axis=1).tolist())
        if best is not None and best_cost <= lower_bound + 1e-7:
            return pack(best)
        consider(np.argmin(r / safe_cap[None, :], axis=1).tolist())
        startup_stop = min(deadline, started + 0.08 * budget)
        attempt = 0
        while time.perf_counter() < startup_stop:
            weight = 0.0 if attempt == 0 else 10.0 ** rng.uniform(-1.0, 1.5)
            candidate = construct(startup_stop, weight, attempt > 1)
            if candidate is not None:
                consider(candidate)
                if best is not None:
                    break
            attempt += 1

        # Linear relaxation of the original problem: capacity dual prices are used for ordering and do not change the actual objective.
        lp_stop = min(deadline, time.perf_counter() + 0.10 * budget)
        built = model(jobs, None) if time.perf_counter() < lp_stop else None
        if built is not None and time.perf_counter() < lp_stop:
            _, lj, mi, matrix, objective, remaining, shift, _ = built
            try:
                relaxation = linprog(
                    objective, A_ub=matrix[n:], b_ub=cap,
                    A_eq=matrix[:n], b_eq=np.ones(n), bounds=(0.0, 1.0),
                    method="highs", options={"time_limit": max(1e-6, lp_stop - time.perf_counter())},
                )
                if relaxation.success:
                    fractional = np.zeros((n, m))
                    fractional[lj, mi] = relaxation.x
                    lp_choice = np.argmax(fractional, axis=1).tolist()
                    fractionality = 1.0 - np.max(fractional, axis=1)
                    prices = np.maximum(0.0, -relaxation.ineqlin.marginals)
                    priced = np.where(eligible, c + r * prices, np.inf)
                    dual = float(np.sum(np.min(priced, axis=1)) - np.dot(prices, cap))
                    if math.isfinite(dual):
                        lower_bound = max(lower_bound, dual)
                    consider(lp_choice)
            except Exception:
                pass

        if best is not None and best_cost <= lower_bound + 1e-7:
            return pack(best)

        # Fractional jobs are released together with the machines they occupy, keeping the stable part of the relaxation.
        if best is None and lp_choice is not None and time.perf_counter() < deadline:
            fractional_jobs = set(np.flatnonzero(fractionality > 1e-7).tolist())
            involved = {lp_choice[j] for j in fractional_jobs}
            selected = list(fractional_jobs)
            candidates = [j for j in range(n) if j not in fractional_jobs]
            candidates.sort(key=lambda j: (
                lp_choice[j] not in involved,
                float(np.min(np.where(
                    eligible[j] & (np.arange(m) != lp_choice[j]),
                    c[j] + prices * r[j] - c[j, lp_choice[j]] - prices[lp_choice[j]] * r[j, lp_choice[j]],
                    np.inf,
                ))),
            ))
            selected.extend(candidates[:max(0, base_size - len(selected))])
            if selected:
                integer_search(selected, lp_choice, min(deadline, time.perf_counter() + 0.12 * budget))

        if best is not None:
            descend(best, min(deadline, time.perf_counter() + 0.055 * budget))
        if time.perf_counter() < deadline:
            share = 0.24 if best is not None else 0.70
            stop = min(started + 0.88 * budget, time.perf_counter() + share * budget)
            if integer_search(jobs, best, stop, full=True):
                return pack(best)

        # If the initial solution fails, prioritise feasibility rather than spending budget on cost improvement of infeasible solutions.
        while best is None and time.perf_counter() < deadline:
            now = time.perf_counter()
            if n * math.log(max(2, m)) <= 6.0 * strength:
                feasibility_dfs(min(deadline, now + 0.06 * budget))
            else:
                integer_search(jobs, None, min(deadline, now + 0.12 * budget), full=True)
            if best is None and time.perf_counter() < deadline:
                candidate = construct(min(deadline, time.perf_counter() + 0.025 * budget),
                                      10.0 ** rng.uniform(-0.5, 2.0), True)
                if candidate is not None:
                    consider(candidate)

        if best is None:
            return pack(backup if backup is not None else [0] * n)

        cycle = 0
        next_global = started + 0.72 * budget
        while time.perf_counter() < deadline:
            now = time.perf_counter()
            if best_cost <= lower_bound + 1e-7:
                break
            if now >= next_global:
                next_global = now + 0.24 * budget
                if integer_search(jobs, best, min(deadline, now + 0.12 * budget), full=True):
                    break
                continue

            base = best[:]
            old_best = best_cost
            aa = np.asarray(base, dtype=np.int64)
            priced = np.where(eligible, c + r * prices, np.inf)
            reduced = priced - np.min(priced, axis=1)[:, None]
            premium = c[jobs, aa] - raw_min
            tension = reduced[jobs, aa]
            if lp_choice is not None:
                disagreement = aa != np.asarray(lp_choice)
            else:
                disagreement = np.zeros(n, dtype=bool)
            stale = min(2.0, max(0.0, (now - last_improvement) / max(1e-9, 0.16 * budget)))
            q = min(n, max(2, int(base_size * (0.70 + 0.55 * stale + 0.45 * rng.random()))))
            scale = max(1.0, float(np.mean(premium + tension)))
            priorities = premium + tension + scale * (fractionality + disagreement)
            ranks = sorted(range(n), key=lambda j: float(priorities[j]) + scale * rng.expovariate(1.0),
                           reverse=True)
            pivot = ranks[0]
            by_machine = [[] for _ in range(m)]
            for j, i in enumerate(base):
                by_machine[i].append(j)
            alternatives = np.argsort(priced, axis=1)
            selected = []
            chosen = set()

            def add(j):
                j = int(j)
                if j not in chosen and len(selected) < q:
                    selected.append(j)
                    chosen.add(j)

            mode = cycle % 4
            cycle += 1
            if mode == 0:
                add(pivot)
            elif mode == 1:
                machines = {base[pivot]}
                machines.update(int(i) for i in alternatives[pivot, :width])
                for j in ranks:
                    if base[j] in machines:
                        add(j)
            elif mode == 2:
                for j in rng.sample(range(n), max(1, q // 2)):
                    add(j)
            else:
                for j in ranks:
                    if disagreement[j] or fractionality[j] > 1e-7:
                        add(j)
            if not selected:
                add(pivot)

            # Extend along low-cost transfer directions, releasing blocking jobs on the target machine that are easy to move.
            cursor = 0
            related_limit = max(1, int(q * (0.6 + 0.25 * rng.random())))
            while cursor < len(selected) and len(selected) < related_limit and time.perf_counter() < deadline:
                j = selected[cursor]
                cursor += 1
                targets = [int(i) for i in alternatives[j, :width] if i != base[j]]
                for target in targets:
                    blockers = [k for k in by_machine[target] if k not in chosen]
                    blockers.sort(key=lambda k: (
                        float(np.min(np.where(np.arange(m) != target, priced[k], np.inf))) - priced[k, target]
                    ) / max(1.0, r[k, target]))
                    number = max(1, int(math.sqrt(q) * (1.0 + stale) / max(1, len(targets))))
                    for k in blockers[:number]:
                        add(k)
                    if len(selected) >= related_limit:
                        break
            for j in ranks:
                if len(selected) >= q:
                    break
                add(j)

            now = time.perf_counter()
            slice_s = budget * (0.025 + 0.025 * stale) * (0.75 + 0.5 * q / max(1, n))
            if integer_search(selected, base, min(deadline, now + slice_s), full=(q == n)):
                break
            if best_cost < old_best and time.perf_counter() < deadline:
                descend(best, min(deadline, time.perf_counter() + 0.035 * budget))
        return pack(best)

    except Exception:
        # Feasible solutions already found are unaffected by later search or numerical library exceptions.
        if best is not None:
            return {"assignments": {str(j): int(best[j]) for j in range(len(best))}}
        if backup is not None:
            return {"assignments": {str(j): int(backup[j]) for j in range(len(backup))}}
        try:
            n = max(0, int(instance.get("num_jobs", 0)))
            m = max(0, int(instance.get("num_machines", 0)))
            rr = instance.get("resource_matrix", [])
            cc = instance.get("capacities", [])
            loads = [0] * m
            answer = {}
            for j in range(n):
                choices = [i for i in range(m) if loads[i] + rr[i][j] <= cc[i]]
                if not choices:
                    choices = list(range(m))
                i = min(choices, key=lambda h: (max(0, loads[h] + rr[h][j] - cc[h]), rr[h][j])) if choices else 0
                answer[str(j)] = int(i)
                if m:
                    loads[i] += rr[i][j]
            return {"assignments": answer}
        except Exception:
            return {"assignments": {str(j): 0 for j in range(max(0, n))}}
