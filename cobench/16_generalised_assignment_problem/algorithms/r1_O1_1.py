"""Dual-guided integer-programming large neighbourhood search
Combines regret construction, capacity repair, linear-relaxation pricing, global integer programming and machine-block neighbourhood re-optimisation, keeping the best feasible solution.
"""

import math
import random
import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp
from scipy.sparse import coo_matrix, csc_matrix, vstack


def solve(instance: dict, time_limit_s: float) -> dict:
    start = time.monotonic()
    budget = max(0.0, float(time_limit_s))
    deadline = start + budget - min(budget * 0.04, 0.20)
    best = None
    best_cost = math.inf
    fallback = []
    n = 0
    m = 0

    try:
        n = int(instance["num_jobs"])
        m = int(instance["num_machines"])
        if n == 0:
            return {"assignments": {}}
        costs = instance["cost_matrix"]
        resources = instance["resource_matrix"]
        capacities = instance["capacities"]
        fallback = [0] * n
        if m <= 0:
            return {"assignments": {str(j): 0 for j in range(n)}}

        C = np.asarray(costs, dtype=float).T.copy()
        R = np.asarray(resources, dtype=float).T.copy()
        B = np.asarray(capacities, dtype=float)
        rows = np.arange(n)
        scale = np.maximum(1.0, np.abs(B))
        nonnegative = bool(np.all(R >= 0))
        allowed = R <= B[None, :] if nonnegative else np.ones((n, m), dtype=bool)
        shifted = C - np.min(C, axis=1)[:, None]
        span = max(1.0, float(np.max(shifted)))
        rng = random.Random(1000003 * m + 9176 * n)
        lam = np.zeros(m)
        lp_x = None
        best_loads = None
        least_bad = math.inf
        least_bad_cost = math.inf
        fallback = np.argmin(R / scale[None, :], axis=1).tolist()
        strength = math.log1p(budget)
        backend_ok = True
        build_time = 0.0
        last_solver_time = 0.0

        def loads_of(a):
            return np.bincount(a, weights=R[rows, a], minlength=m)

        def accept(a):
            nonlocal best, best_cost, best_loads, fallback, least_bad, least_bad_cost
            if a is None or len(a) != n:
                return False
            a = [int(i) for i in a]
            if any(i < 0 or i >= m for i in a):
                return False
            loads = [0] * m
            value = 0
            for j, i in enumerate(a):
                loads[i] += resources[i][j]
                value += costs[i][j]
            bad = sum(max(0, loads[i] - capacities[i]) / scale[i] for i in range(m))
            if (bad, value) < (least_bad, least_bad_cost):
                least_bad, least_bad_cost = bad, value
                fallback = a[:]
            if any(loads[i] > capacities[i] for i in range(m)):
                return False
            if value < best_cost:
                best = np.asarray(a, dtype=int)
                best_loads = np.asarray(loads, dtype=float)
                best_cost = value
                return True
            return False

        accept(fallback)
        cheapest = np.argmin(C, axis=1)
        accept(cheapest)
        if best is not None and best_cost == sum(min(costs[i][j] for i in range(m)) for j in range(n)):
            return {"assignments": {str(j): int(best[j]) for j in range(n)}}
        if m == 1:
            return {"assignments": {str(j): 0 for j in range(n)}}

        def construct(weight, noisy, stop):
            prices = shifted + R * lam[None, :]
            scores = prices + weight * span * R / scale[None, :]
            scores = np.where(allowed, scores, np.inf)
            if noisy:
                scores = scores + np.asarray([rng.random() for _ in range(n)])[:, None] * span * 0.02
            two = np.partition(scores, 1, axis=1)[:, :2]
            regret = np.max(two, axis=1) - np.min(two, axis=1)
            regret = np.nan_to_num(regret, nan=0.0, posinf=span * (n + 1))
            if noisy:
                regret *= np.asarray([0.5 + rng.random() for _ in range(n)])
            size = np.min(np.where(allowed, R / scale[None, :], np.inf), axis=1)
            order = np.lexsort((-size, -regret, allowed.sum(axis=1)))
            remaining = B.copy()
            assignment = np.empty(n, dtype=int)
            for j in order:
                if time.monotonic() >= stop:
                    return
                fits = allowed[j] & (R[j] <= remaining)
                if not np.any(fits):
                    return
                pressure = R[j] / np.maximum(1.0, remaining)
                values = prices[j] + weight * span * pressure
                if noisy:
                    values = values + np.asarray([rng.random() for _ in range(m)]) * span / (2.0 + strength)
                i = int(np.argmin(np.where(fits, values, np.inf)))
                assignment[j] = i
                remaining[i] -= R[j, i]
            accept(assignment)

        def repair(stop):
            """Tabu migration allowing temporary overload; feasibility is confirmed on the original integer data."""
            a = np.asarray(fallback, dtype=int).copy()
            loads = loads_of(a)
            tabu = np.zeros((n, m), dtype=int)
            step = 0
            tenure = max(1, int(math.sqrt(n) * (1.0 + strength / 4.0)))
            while time.monotonic() < stop and best is None:
                bad = np.maximum(loads - B, 0.0) / scale
                if not np.any(bad > 0):
                    accept(a)
                    return
                gain = bad[a] - np.maximum(loads[a] - R[rows, a] - B[a], 0.0) / scale[a]
                change = (np.maximum(loads[None, :] + R - B[None, :], 0.0) / scale[None, :]
                          - bad[None, :] - gain[:, None])
                delta_cost = shifted - shifted[rows, a][:, None]
                mask = allowed & ((tabu <= step) | (bad.sum() + change < least_bad - 1e-10))
                change = np.where(mask, change, np.inf)
                change[rows, a] = np.inf
                ranking = change + delta_cost * (1e-5 / span)
                flat = int(np.argmin(ranking))
                j, i = divmod(flat, m)
                if not np.isfinite(ranking[j, i]):
                    tabu.fill(0)
                    step += 1
                    continue
                old = int(a[j])
                loads[old] -= R[j, old]
                loads[i] += R[j, i]
                a[j] = i
                tabu[j, old] = step + tenure + rng.randrange(tenure + 1)
                step += 1
                accept(a)

        construct_stop = min(deadline, start + budget * 0.055)
        attempt = 0
        while time.monotonic() < construct_stop:
            weight = (0.0, 0.3, 1.5, 6.0)[attempt % 4] if attempt < 4 else 10.0 ** rng.uniform(-1.2, 1.4)
            construct(weight, attempt >= 4, construct_stop)
            attempt += 1
            if best is not None:
                break
        if best is None:
            repair(min(deadline, start + budget * 0.10))

        # Each column is exactly one x[i][j]; there are no extra decision variables.
        setup = time.monotonic()
        all_columns = np.arange(n * m, dtype=np.int32)
        job_rows = np.repeat(np.arange(n, dtype=np.int32), m)
        machine_rows = np.tile(np.arange(m, dtype=np.int32), n)
        Aeq = coo_matrix((np.ones(n * m), (job_rows, all_columns)), shape=(n, n * m)).tocsc()
        Acap = coo_matrix((R.ravel(), (machine_rows, all_columns)), shape=(m, n * m)).tocsc()
        Afull = vstack((Aeq, Acap), format="csc")
        build_time = time.monotonic() - setup

        # Dual multipliers are used only for pricing; later lower bounds are recomputed from non-negative multipliers, without relying on floating-point optimality claims.
        lp_time = min(budget * 0.10, max(0.0, (deadline - time.monotonic()) * 0.18))
        if lp_time > 0.002:
            try:
                lp = linprog(shifted.ravel(), A_ub=Acap, b_ub=B, A_eq=Aeq,
                             b_eq=np.ones(n), bounds=np.column_stack((np.zeros(n * m), allowed.ravel().astype(float))),
                             method="highs", options={"time_limit": lp_time, "presolve": True})
                if lp.x is not None and np.all(np.isfinite(lp.x)):
                    lp_x = np.asarray(lp.x).reshape(n, m)
                    accept(np.argmax(lp_x, axis=1))
                if lp.success and lp.ineqlin.marginals is not None:
                    candidate_lam = np.maximum(0.0, -np.asarray(lp.ineqlin.marginals))
                    if np.all(np.isfinite(candidate_lam)):
                        lam = candidate_lam
            except Exception:
                pass

        if best is None and time.monotonic() < deadline:
            priced = shifted + R * lam[None, :]
            accept(np.argmin(np.where(allowed, priced, np.inf), axis=1))
            stop = min(deadline, time.monotonic() + budget * 0.035)
            while time.monotonic() < stop and best is None:
                construct(10.0 ** rng.uniform(-1.5, 0.8), True, stop)

        def integer_neighborhood(subset, allowance, prove_global=False):
            """Fix x outside the neighbourhood and jointly optimise the assignment of all jobs and machines inside it."""
            nonlocal backend_ok, build_time, last_solver_time
            if not backend_ok or time.monotonic() >= deadline:
                return False, False
            before = time.monotonic()
            subset = np.asarray(sorted(set(int(j) for j in subset)), dtype=int)
            k = len(subset)
            if k == 0:
                return False, False
            full = k == n
            old = best.copy() if best is not None else None
            if old is None and not full:
                return False, False
            residual = B.copy()
            if old is not None and not full:
                residual -= best_loads
                residual += np.bincount(old[subset], weights=R[subset, old[subset]], minlength=m)
            c = shifted[subset].ravel()
            r = R[subset]
            upper = allowed[subset].copy()
            if nonnegative:
                upper &= r <= residual[None, :] + 1e-9
            if np.any(~np.any(upper, axis=1)):
                return False, False
            incumbent_part = None
            if old is not None:
                incumbent_part = float(np.sum(shifted[subset, old[subset]]))
                priced = shifted[subset] + r * lam[None, :]
                minima = np.min(np.where(upper, priced, np.inf), axis=1)
                dual_bound = float(np.sum(minima) - np.dot(lam, residual))
                tolerance = 1e-7 * (1.0 + abs(dual_bound) + abs(incumbent_part))
                gap = incumbent_part - 1.0 - dual_bound
                if gap < -tolerance:
                    return False, bool(full and prove_global)
                # For any feasible solution, the objective is at least the lower bound plus the reduced costs of the selected arcs.
                upper &= priced - minima[:, None] <= gap + tolerance
                if np.any(~np.any(upper, axis=1)):
                    return False, bool(full and prove_global)
            if full:
                matrix = Afull
            else:
                cols = np.arange(k * m, dtype=np.int32)
                rr = np.concatenate((np.repeat(np.arange(k, dtype=np.int32), m),
                                     k + np.tile(np.arange(m, dtype=np.int32), k)))
                matrix = coo_matrix((np.concatenate((np.ones(k * m), r.ravel())),
                                     (rr, np.concatenate((cols, cols)))), shape=(k + m, k * m)).tocsc()
            lower_rows = np.concatenate((np.ones(k), np.full(m, -np.inf)))
            upper_rows = np.concatenate((np.ones(k), residual))
            if incumbent_part is not None:
                matrix = vstack((matrix, csc_matrix(c.reshape(1, -1))), format="csc")
                lower_rows = np.append(lower_rows, -np.inf)
                upper_rows = np.append(upper_rows, incumbent_part - 1.0)
            build_time = max(build_time * 0.8, time.monotonic() - before)
            remaining = deadline - time.monotonic()
            usable = min(allowance - (time.monotonic() - before), remaining * 0.88)
            if usable <= max(0.001, build_time * 0.1):
                return False, False
            try:
                solve_start = time.monotonic()
                result = milp(c, integrality=np.ones(k * m, dtype=np.uint8),
                              bounds=Bounds(np.zeros(k * m), upper.ravel().astype(float)),
                              constraints=LinearConstraint(matrix, lower_rows, upper_rows),
                              options={"time_limit": usable, "mip_rel_gap": 0.0, "presolve": True})
                last_solver_time = time.monotonic() - solve_start
                improved = False
                valid_result = False
                if result.x is not None and np.all(np.isfinite(result.x)):
                    values = np.asarray(result.x).reshape(k, m)
                    selected = np.argmax(values, axis=1)
                    if np.all(values[np.arange(k), selected] >= 1.0 - 1e-5):
                        candidate = old.copy() if old is not None else np.zeros(n, dtype=int)
                        candidate[subset] = selected
                        previous = best_cost
                        improved = accept(candidate)
                        valid_result = improved or (best is not None and np.array_equal(candidate, best))
                        if best_cost > previous:
                            valid_result = False
                proven = (full and prove_global and
                          ((result.status == 0 and valid_result) or (result.status == 2 and old is not None)))
                return improved, bool(proven)
            except Exception:
                backend_ok = False
                return False, False

        def descent(stop):
            if best is None:
                return
            a = best.copy()
            loads = best_loads.copy()
            while time.monotonic() < stop:
                differences = C - C[rows, a][:, None]
                fits = loads[None, :] + R <= B[None, :] + 1e-9
                gains = np.where(fits & (differences < -1e-9), differences, np.inf)
                gains[rows, a] = np.inf
                flat = int(np.argmin(gains))
                j, i = divmod(flat, m)
                if np.isfinite(gains[j, i]):
                    old = int(a[j])
                    loads[old] -= R[j, old]
                    loads[i] += R[j, i]
                    a[j] = i
                    accept(a)
                    continue
                changed = False
                order = list(range(n))
                rng.shuffle(order)
                for j in order:
                    if time.monotonic() >= stop:
                        return
                    i = int(a[j])
                    new_i = loads[i] - R[j, i] + R[:, i]
                    new_other = loads[a] - R[rows, a] + R[j, a]
                    deltas = C[j, a] + C[:, i] - C[j, i] - C[rows, a]
                    possible = (a != i) & (new_i <= B[i] + 1e-9) & (new_other <= B[a] + 1e-9)
                    deltas = np.where(possible & (deltas < -1e-9), deltas, np.inf)
                    k = int(np.argmin(deltas))
                    if np.isfinite(deltas[k]):
                        other = int(a[k])
                        loads[i], loads[other] = new_i[k], new_other[k]
                        a[j], a[k] = other, i
                        accept(a)
                        changed = True
                        break
                if not changed:
                    return

        if best is not None:
            descent(min(deadline, time.monotonic() + budget * 0.025))
        initial_slice = max(0.0, deadline - time.monotonic()) * (0.55 if best is None else 0.43)
        _, proven = integer_neighborhood(range(n), initial_slice, True)
        if proven:
            return {"assignments": {str(j): int(best[j]) for j in range(n)}}

        failures = 0
        iteration = 0
        while time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            if best is None:
                stop = min(deadline, time.monotonic() + remaining / (3.0 + strength))
                repair(stop)
                if best is None:
                    _, proven = integer_neighborhood(range(n), max(0.0, deadline - time.monotonic()) * 0.65, True)
                    if not backend_ok:
                        construct(10.0 ** rng.uniform(-1.0, 2.0), True, deadline)
                    continue
            if remaining <= max(0.002, build_time * 2.0):
                descent(deadline)
                break
            if not backend_ok:
                construct(10.0 ** rng.uniform(-1.5, 1.0), True,
                          min(deadline, time.monotonic() + remaining / (2.0 + strength)))
                descent(min(deadline, time.monotonic() + remaining / (2.0 + strength)))
                continue

            base = best.copy()
            prices = shifted + R * lam[None, :]
            prices = np.where(allowed, prices, np.inf)
            preferred = np.argmin(prices, axis=1)
            opportunity = np.maximum(0.0, prices[rows, base] - prices[rows, preferred])
            raw_saving = shifted[rows, base] - np.min(shifted, axis=1)
            priorities = opportunity + raw_saving / (2.0 + strength)
            if lp_x is not None:
                priorities += span * (1.0 - lp_x[rows, base]) / (2.0 + strength)
            priorities = np.nan_to_num(priorities, nan=0.0, posinf=span)
            priorities += span / (1.0 + math.sqrt(n))
            probabilities = priorities / max(1e-30, float(np.sum(priorities)))
            # Scale grows with the time budget and stagnation; neither the number of restarts nor the maximum iterations is limited.
            target_size = min(n, max(2, int(math.sqrt(n) * (2.0 + strength) *
                                          (1.0 + math.log1p(failures) / (2.0 + strength)))))
            phase = iteration % 4
            all_jobs = phase == 3 and failures > 1.0 + strength
            if all_jobs or target_size >= n:
                subset = list(range(n))
            else:
                pivot = rng.choices(range(n), weights=probabilities, k=1)[0]
                subset_set = {pivot}
                if phase == 0:
                    machines = {int(base[pivot]), int(preferred[pivot])}
                    order_machines = list(range(m))
                    rng.shuffle(order_machines)
                    desired = min(m, max(2, int(m * target_size / max(1, n))))
                    for i in order_machines:
                        if len(machines) >= desired:
                            break
                        machines.add(i)
                    subset_set.update(j for j in range(n) if int(base[j]) in machines)
                elif phase == 1 and lp_x is not None:
                    difference = 1.0 - lp_x[rows, base]
                    keys = difference + np.asarray([rng.random() for _ in range(n)]) / (2.0 + strength)
                    subset_set.update(int(j) for j in np.argsort(-keys)[:target_size])
                else:
                    # Release blocking jobs along the target machine, then bring in jobs whose cheap destinations lie there.
                    frontier = [pivot]
                    while frontier and len(subset_set) < target_size and time.monotonic() < deadline:
                        j = frontier.pop(0)
                        destination = int(preferred[j])
                        blockers = [k for k in range(n) if base[k] == destination and k not in subset_set]
                        rng.shuffle(blockers)
                        blockers.sort(key=lambda k: (prices[k, base[k]] - np.min(prices[k])) /
                                      max(1.0, R[k, base[k]]), reverse=True)
                        for k in blockers:
                            subset_set.add(k)
                            frontier.append(k)
                            if len(subset_set) >= target_size:
                                break
                if len(subset_set) < target_size:
                    # An exponential race implements priority sampling without replacement.
                    keys = [(-math.log(max(1e-300, rng.random())) / priorities[j], j)
                            for j in range(n) if j not in subset_set]
                    keys.sort()
                    subset_set.update(j for _, j in keys[:target_size - len(subset_set)])
                subset = list(subset_set)
            remaining = deadline - time.monotonic()
            slice_s = budget / (5.0 + 2.0 * strength) * (1.0 + math.log1p(failures) / (2.0 + strength))
            if len(subset) == n:
                slice_s *= 1.0 + strength / 2.0
            slice_s = min(slice_s, remaining * 0.75)
            before_cost = best_cost
            improved, proven = integer_neighborhood(subset, slice_s, len(subset) == n)
            if proven:
                break
            if improved:
                failures = 0
                descent(min(deadline, time.monotonic() + min(budget * 0.025, (deadline - time.monotonic()) * 0.15)))
            else:
                failures += 1
                # When the solver very quickly finds no improvement in a neighbourhood, interleave cheap local descent.
                if last_solver_time < slice_s / (2.0 + strength):
                    descent(min(deadline, time.monotonic() + min(budget * 0.01, (deadline - time.monotonic()) * 0.1)))
            if best_cost < before_cost:
                failures = 0
            iteration += 1

    except Exception:
        # A confirmed feasible solution is not lost due to later numerical computation or solver failure.
        pass

    answer = best if best is not None else fallback
    return {"assignments": {str(j): int(answer[j]) if j < len(answer) else 0 for j in range(n)}}
