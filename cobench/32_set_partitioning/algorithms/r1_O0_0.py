"""Relaxation-guided warm-start large-neighbourhood set partitioning
Merges duplicate covers and propagates forced selections; combines exact-cover construction, linear relaxation, warm-started integer programming and adaptive destroy-and-repair, keeping the best verified complete partition.
"""

import math
import random
import time
from collections import deque


def solve(instance: dict, time_limit_s: float) -> dict:
    started = time.perf_counter()
    answer = {"feasible": False, "x": None, "total_cost": None}
    try:
        budget = max(0.0, float(time_limit_s))
        if not math.isfinite(budget):
            budget = 0.0
        deadline = started + budget - min(0.12, 0.015 * budget)
        n, m = instance["n"], instance["m"]
        if (type(n) is not int or type(m) is not int or n < 0 or m < 0
                or len(instance["costs"]) != n or len(instance["covers"]) != n):
            return dict(answer, error="Input error: invalid dimensions or array lengths")
        original_cost = []
        cheapest = {}
        for j, (value, rows) in enumerate(zip(instance["costs"], instance["covers"])):
            if isinstance(value, (str, bool)):
                return dict(answer, error="Input error: costs must be non-negative finite reals")
            cost = float(value)
            if not math.isfinite(cost) or cost < 0:
                return dict(answer, error="Input error: costs must be non-negative finite reals")
            original_cost.append(cost)
            if any(type(r) is not int or r < 0 or r >= m for r in rows):
                return dict(answer, error="Input error: invalid task index")
            key = tuple(sorted(rows))
            if len(set(key)) != len(key):
                return dict(answer, error="Input error: duplicate index in a cover list")
            if key and (key not in cheapest or cost < original_cost[cheapest[key]]):
                cheapest[key] = j
        if m == 0:
            return {"feasible": True, "x": [0] * n, "total_cost": 0.0}

        covers = list(cheapest)
        representatives = list(cheapest.values())
        costs = [original_cost[j] for j in representatives]
        row_cols = [[] for _ in range(m)]
        for j, rows in enumerate(covers):
            for r in rows:
                row_cols[r].append(j)
        degree = [len(v) for v in row_cols]
        if not all(degree):
            return answer
        active = bytearray([1]) * len(covers)
        remaining = bytearray([1]) * m
        queue = deque(r for r, d in enumerate(degree) if d == 1)
        forced = []
        while queue:
            r = queue.popleft()
            if not remaining[r]:
                continue
            if degree[r] == 0:
                return answer
            j = next((j for j in row_cols[r] if active[j]), -1)
            if j < 0:
                return answer
            forced.append(representatives[j])
            for s in covers[j]:
                remaining[s] = 0
            for s in covers[j]:
                for k in row_cols[s]:
                    if active[k]:
                        active[k] = 0
                        for u in covers[k]:
                            if remaining[u]:
                                degree[u] -= 1
                                if degree[u] == 0:
                                    return answer
                                if degree[u] == 1:
                                    queue.append(u)
        rows_left = [r for r in range(m) if remaining[r]]
        row_map = {r: i for i, r in enumerate(rows_left)}
        keep = [j for j in range(len(covers)) if active[j]]
        covers = [tuple(row_map[r] for r in covers[j]) for j in keep]
        representatives = [representatives[j] for j in keep]
        costs = [costs[j] for j in keep]
        M, N = len(rows_left), len(keep)
        row_cols = [[] for _ in range(M)]
        for j, rows in enumerate(covers):
            for r in rows:
                row_cols[r].append(j)
        best = None
        best_value = math.inf
        rng = random.Random(1729 + 1009 * M + N)
        sizes = [len(rows) for rows in covers]

        def accept(selected):
            nonlocal best, best_value, answer
            seen = bytearray(M)
            total_rows = 0
            for j in selected:
                if not 0 <= j < N:
                    return False
                for r in covers[j]:
                    if seen[r]:
                        return False
                    seen[r] = 1
                    total_rows += 1
            if total_rows != M:
                return False
            value = math.fsum(costs[j] for j in selected)
            if value >= best_value:
                return False
            ids = forced + [representatives[j] for j in selected]
            x = [0] * n
            for j in ids:
                x[j] = 1
            answer = {"feasible": True, "x": x,
                      "total_cost": math.fsum(original_cost[j] for j in ids)}
            best = list(selected)
            best_value = value
            return True

        if M == 0:
            accept([])
            return answer
        singles = [-1] * M
        for j, rows in enumerate(covers):
            if len(rows) == 1:
                singles[rows[0]] = j
        if all(j >= 0 for j in singles):
            accept(singles)
            savings = [sum(costs[singles[r]] for r in rows) - costs[j]
                       for j, rows in enumerate(covers)]
            used = bytearray(M)
            chosen = []
            for j in sorted(range(N), key=lambda j: -savings[j] / math.sqrt(sizes[j])):
                if savings[j] > 0 and not any(used[r] for r in covers[j]):
                    chosen.append(j)
                    for r in covers[j]:
                        used[r] = 1
            chosen.extend(singles[r] for r in range(M) if not used[r])
            accept(chosen)
        if best_value == 0 or time.perf_counter() >= deadline:
            return answer

        # Exact-cover search with an explicit stack, free of recursion-depth limits; every search stops at the wall-clock deadline.
        ranks = [costs[j] / sizes[j] for j in range(N)]
        reduced = [0.0] * N
        lower_bound = 0.0
        ordered_rows = sorted(range(M), key=lambda r: len(row_cols[r]))

        def exact_search(stop, first_only=False):
            occupied = bytearray(M)
            path, frames = [], []
            path_cost = 0.0
            path_reduced = 0.0
            covered = 0
            descend = True
            while time.perf_counter() < stop:
                if descend:
                    options = []
                    if covered == M:
                        accept(path)
                        if first_only:
                            return False
                    elif (path_cost < best_value and
                          lower_bound + path_reduced < best_value - 1e-9):
                        min_degree = N + 1
                        for r in ordered_rows:
                            if occupied[r]:
                                continue
                            available = []
                            for j in row_cols[r]:
                                if (costs[j] + path_cost < best_value and
                                        not any(occupied[s] for s in covers[j])):
                                    available.append(j)
                                if len(available) >= min_degree:
                                    break
                            if len(available) < min_degree:
                                min_degree = len(available)
                                options = available
                                if min_degree <= 1:
                                    break
                            if time.perf_counter() >= stop:
                                return False
                        if options:
                            options.sort(key=lambda j: ranks[j], reverse=True)
                    frames.append(options)
                while frames and not frames[-1]:
                    frames.pop()
                    if path:
                        j = path.pop()
                        path_cost -= costs[j]
                        path_reduced -= reduced[j]
                        covered -= sizes[j]
                        for r in covers[j]:
                            occupied[r] = 0
                if not frames:
                    return True
                j = frames[-1].pop()
                path.append(j)
                path_cost += costs[j]
                path_reduced += reduced[j]
                covered += sizes[j]
                for r in covers[j]:
                    occupied[r] = 1
                descend = True
            return False

        if best is None:
            finished = exact_search(min(deadline, time.perf_counter() + budget * 0.045), True)
            if finished:
                return answer
        if time.perf_counter() >= deadline:
            return answer
        try:
            import numpy as np
            from scipy.sparse import csc_matrix
            from scipy.optimize import Bounds, LinearConstraint, linprog, milp
        except Exception:
            exact_search(deadline)
            return answer
        try:
            from scipy.optimize._highspy import _core
        except Exception:
            _core = None

        indices, starts = [], [0]
        for rows in covers:
            indices.extend(rows)
            starts.append(len(indices))
        A = csc_matrix((np.ones(len(indices)), np.asarray(indices, dtype=np.int32),
                        np.asarray(starts, dtype=np.int32)), shape=(M, N))
        positive = sorted(c for c in costs if c > 0)
        scale = max(1e-100, min(1e100, positive[len(positive) // 2])) if positive else 1.0
        c = np.asarray(costs, dtype=float) / scale
        size_array = np.asarray(sizes)

        def optimize(matrix, objective, seconds, integer=True, hint=None):
            nonlocal _core
            end = min(deadline, time.perf_counter() + max(0.0, seconds))
            nr, nc = matrix.shape
            if nc == 0 or time.perf_counter() >= end:
                return None, None, False, False
            if _core is not None:
                try:
                    h = _core._Highs()
                    h.setOptionValue("output_flag", False)
                    h.setOptionValue("threads", 1)
                    h.setOptionValue("random_seed", rng.randrange(2147483647))
                    h.setOptionValue("mip_rel_gap", 0.0)
                    h.setOptionValue("mip_abs_gap", 1e-7 / scale)
                    if integer:
                        h.setOptionValue("mip_heuristic_effort", 0.2 if hint is None else 0.1)
                    lp = _core.HighsLp()
                    lp.num_col_, lp.num_row_ = nc, nr
                    lp.col_cost_ = objective
                    lp.col_lower_ = np.zeros(nc)
                    lp.col_upper_ = np.ones(nc) if integer else np.full(nc, np.inf)
                    lp.row_lower_, lp.row_upper_ = np.ones(nr), np.ones(nr)
                    lp.a_matrix_.format_ = _core.MatrixFormat.kColwise
                    lp.a_matrix_.start_ = matrix.indptr
                    lp.a_matrix_.index_ = matrix.indices
                    lp.a_matrix_.value_ = matrix.data
                    if integer:
                        lp.integrality_ = [_core.HighsVarType.kInteger] * nc
                    h.passModel(lp)
                    if hint is not None:
                        h.setSolution(nc, np.arange(nc, dtype=np.int32), hint)

                    def interrupt(kind, message, data_out, data_in, user_data):
                        if time.perf_counter() >= end:
                            data_in.user_interrupt = True

                    h.setCallback(interrupt, None)
                    for event in (_core.cb.kCallbackMipInterrupt,
                                  _core.cb.kCallbackSimplexInterrupt,
                                  _core.cb.kCallbackIpmInterrupt):
                        h.startCallback(event)
                    h.setOptionValue("time_limit", max(0.0, end - time.perf_counter()))
                    h.run()
                    status = h.getModelStatus()
                    sol = h.getSolution()
                    values = np.asarray(sol.col_value) if sol.value_valid else None
                    dual = np.asarray(sol.row_dual) if sol.dual_valid else None
                    return (values, dual, status == _core.HighsModelStatus.kOptimal,
                            status == _core.HighsModelStatus.kInfeasible)
                except Exception:
                    _core = None
            seconds = end - time.perf_counter()
            if seconds <= 0:
                return None, None, False, False
            try:
                if integer:
                    result = milp(objective, integrality=np.ones(nc),
                                  bounds=Bounds(np.zeros(nc), np.ones(nc)),
                                  constraints=LinearConstraint(matrix, np.ones(nr), np.ones(nr)),
                                  options={"time_limit": seconds, "mip_rel_gap": 0.0,
                                           "disp": False})
                    return result.x, None, result.status == 0, result.status == 2
                result = linprog(objective, A_eq=matrix, b_eq=np.ones(nr),
                                 bounds=(0, None), method="highs",
                                 options={"time_limit": seconds, "disp": False})
                dual = result.eqlin.marginals if result.status == 0 else None
                return result.x, dual, result.status == 0, result.status == 2
            except Exception:
                return None, None, False, False

        def take_solution(values, columns, fixed=()):
            if values is None or len(values) != len(columns):
                return False
            if not np.all(np.isfinite(values)) or np.max(np.abs(values - np.rint(values))) > 1e-5:
                return False
            chosen = list(fixed) + [int(columns[i]) for i in np.flatnonzero(values > 0.5)]
            return accept(chosen)

        all_columns = np.arange(N)
        values, dual, optimal, infeasible = optimize(
            A, c, min(budget * 0.13, max(0.0, deadline - time.perf_counter()) * 0.22), False)
        if infeasible:
            return answer
        if values is not None:
            take_solution(values, all_columns)
            if best is not None and optimal and np.max(np.abs(values - np.rint(values))) < 1e-7:
                return answer
        if dual is not None and optimal and np.all(np.isfinite(dual)):
            prices = dual * scale
            residual = np.asarray(costs) - A.T.dot(prices)
            # Lower the row prices so every column has non-negative reduced cost, then apply safe variable elimination.
            repair = max(0.0, float(np.max(-residual / size_array)))
            repair += 1e-10 * max(1.0, float(np.max(np.abs(prices))))
            prices -= repair
            residual = np.asarray(costs) - A.T.dot(prices)
            reduced = np.maximum(0.0, residual).tolist()
            lower_bound = max(0.0, float(np.sum(prices)))
        rc = np.asarray(reduced)
        if values is not None:
            ranks = [(-float(values[j]), reduced[j] / sizes[j], costs[j] / sizes[j])
                     for j in range(N)]
        if best is None:
            if exact_search(min(deadline, time.perf_counter() + budget * 0.04), True):
                return answer
        if best_value <= lower_bound + 1e-7:
            return answer

        def eligible_columns():
            if best is None:
                return all_columns
            tolerance = 1e-7 * max(1.0, abs(best_value), abs(lower_bound))
            eligible = rc <= best_value - lower_bound + tolerance
            eligible[best] = True
            return np.flatnonzero(eligible)

        columns = eligible_columns()
        matrix = A if len(columns) == N else A[:, columns].tocsc()
        hint = None if best is None else np.isin(columns, best).astype(float)
        left = max(0.0, deadline - time.perf_counter())
        values, _, optimal, infeasible = optimize(matrix, c[columns],
                                                  left * (0.68 if best is None else 0.53), True, hint)
        take_solution(values, columns)
        if optimal or (infeasible and best is None) or best_value <= lower_bound + 1e-7:
            return answer

        strength = 1.0 + math.log1p(budget)
        stall = 0
        while time.perf_counter() < deadline:
            left = deadline - time.perf_counter()
            if left <= 0:
                break
            columns = eligible_columns()
            fixed = []
            free_rows = None
            if best is not None:
                count = len(best)
                target = min(count, max(2, math.ceil(math.sqrt(count) * strength
                             * (1.0 + stall / (2.0 + math.log1p(M))))))
                owner = [-1] * M
                for j in best:
                    for r in covers[j]:
                        owner[r] = j
                weights = [max(reduced[j], costs[j] * 0.02, scale * 1e-8) for j in best]
                release = {rng.choices(best, weights=weights, k=1)[0]}
                pool = list(best)
                rng.shuffle(pool)
                while len(release) < target and time.perf_counter() < deadline:
                    seed = rng.choice(tuple(release))
                    r = rng.choice(covers[seed])
                    candidate = rng.choice(row_cols[r])
                    owners = {owner[s] for s in covers[candidate]}
                    if len(release | owners) <= target:
                        release.update(owners)
                    while pool and pool[-1] in release:
                        pool.pop()
                    if pool and len(release) < target:
                        release.add(pool.pop())
                if len(release) < count:
                    fixed = [j for j in best if j not in release]
                    blocked_rows = np.ones(M)
                    for j in release:
                        blocked_rows[list(covers[j])] = 0.0
                    free_rows = np.flatnonzero(blocked_rows == 0)
                    conflicts = A.T.dot(blocked_rows)
                    columns = columns[conflicts[columns] == 0]
            if time.perf_counter() >= deadline:
                break
            matrix = A[:, columns]
            if free_rows is not None:
                matrix = matrix[free_rows, :]
            matrix = matrix.tocsc()
            hint = None if best is None else np.isin(columns, best).astype(float)
            allowance = math.sqrt(max(budget, 1e-6)) * (0.085 + 0.025 * stall)
            if not fixed:
                allowance *= strength
            allowance = min(deadline - time.perf_counter(), allowance)
            if allowance <= 0:
                break
            old = best_value
            values, _, optimal, infeasible = optimize(matrix, c[columns], allowance, True, hint)
            take_solution(values, columns, fixed)
            if (optimal and not fixed) or (infeasible and best is None):
                break
            if best_value <= lower_bound + 1e-7:
                break
            if best_value < old - 1e-8:
                stall = 0
            else:
                stall += 1
            if values is None and best is None:
                # With no native solver, or no integer point yet, continue the explicit exact-cover search.
                if exact_search(min(deadline, time.perf_counter() + allowance), True):
                    break
        return answer
    except Exception:
        # A verified complete partition is never lost to a later solver exception.
        return answer
