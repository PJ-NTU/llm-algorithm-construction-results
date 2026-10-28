"""Adaptive facility location search strengthened by capacity covers
First constructs a feasible assignment, then keeps improving by combining transportation linear programming, capacity cover inequalities and time-limited mixed-integer neighbourhood search.
"""

import math
import time
import numpy as np


def solve(instance: dict, time_limit_s: float) -> dict:
    started = time.perf_counter()
    budget = max(0.0, float(time_limit_s))
    n = int(instance["num_customers"])
    m = int(instance["num_plants"])
    demand = np.asarray(instance["customers_demands"], dtype=float)
    capacity = np.asarray(instance["plants_capacities"], dtype=float)
    fixed = np.asarray(instance["plants_fixed_costs"], dtype=float)
    shipping = np.asarray(instance["shipping_costs"], dtype=float).reshape(n, m)
    positive = np.flatnonzero(demand > 0.0)
    zero = np.flatnonzero(demand == 0.0)
    d = demand[positive]
    c = shipping[positive]
    p = len(positive)
    total = float(d.sum())
    reserve = min(0.15 * budget, max(0.015, 0.04 * budget + n * m * 2e-7))
    deadline = started + budget - reserve
    negative = fixed < 0.0

    def output(y, x):
        result = np.zeros((n, m), dtype=float)
        result[positive] = x
        if len(zero):
            result[zero, np.argmin(shipping[zero], axis=1)] = 1.0
        keys = [str(j) for j in range(m)]
        return {
            "open_plants": [int(v) for v in y],
            "assignments": {
                str(i): dict(zip(keys, map(float, result[i]))) for i in range(n)
            },
        }

    if p == 0:
        return output(negative.astype(int), np.zeros((0, m)))

    # In a complete bipartite transportation network, a feasible solution can be constructed whenever total capacity is sufficient.
    def construct(chosen):
        ids = np.flatnonzero(chosen & (capacity > 0.0))
        available = capacity.copy()
        x = np.zeros((p, m), dtype=float)
        if len(ids) > 1:
            two = np.partition(c[:, ids], 1, axis=1)[:, :2]
            regret = np.abs(two[:, 1] - two[:, 0]) / d
            order = np.argsort(-regret, kind="stable")
        else:
            order = np.arange(p)
        for i in order:
            remaining = float(d[i])
            last = -1
            for j in ids[np.argsort(c[i, ids], kind="stable")]:
                if available[j] <= 0.0:
                    continue
                amount = min(remaining, float(available[j]))
                x[i, j] = amount / d[i]
                available[j] -= amount
                remaining -= amount
                last = j
                if remaining <= 0.0:
                    break
            # Only absorb residual errors from float addition and subtraction; actual assignment fractions are not truncated.
            if last >= 0:
                x[i, last] += 1.0 - float(x[i].sum())
        y = ((np.sum(x, axis=0) > 0.0) | negative).astype(int)
        return y, x

    best_y, best_x = construct(np.ones(m, dtype=bool))
    best_value = float(fixed @ best_y + np.sum(c * best_x))
    last_improvement = time.perf_counter()
    if last_improvement >= deadline:
        return output(best_y, best_x)

    try:
        from scipy.optimize import Bounds, LinearConstraint, linprog, milp
        from scipy.sparse import coo_matrix, csc_matrix, vstack

        rng = np.random.default_rng(20261002)
        scale_d = max(1.0, float(np.max(d)), min(total, float(np.max(capacity))))
        scale_c = max(1.0, float(np.max(np.abs(fixed))), float(np.max(np.abs(c))))
        density = np.maximum(fixed, 0.0) / np.maximum(np.minimum(capacity, total), 1e-30)
        seed_score = density + np.sum(c, axis=0) / total
        chosen = negative.copy()
        selected_capacity = float(capacity[chosen].sum())
        for j in np.argsort(seed_score, kind="stable"):
            if selected_capacity >= total:
                break
            if not chosen[j]:
                chosen[j] = True
                selected_capacity += capacity[j]
        # Only remove selected facilities with redundant capacity, then rebuild the transportation assignment.
        for j in np.argsort(-fixed, kind="stable"):
            if chosen[j] and fixed[j] >= 0.0 and selected_capacity - capacity[j] >= total:
                chosen[j] = False
                selected_capacity -= capacity[j]
        if time.perf_counter() < deadline:
            sy, sx = construct(chosen)
            sv = float(fixed @ sy + np.sum(c * sx))
            if sv < best_value:
                best_y, best_x, best_value = sy, sx, sv

        # Model size and time budget jointly determine the degree of sparsification; all edges of the current solution are always kept.
        edge_budget = max(2048, int(6000 * math.sqrt(max(budget, 0.001)) *
                                    (1.0 + math.log1p(m) / 10.0)))
        mandatory = negative | (total - (float(capacity.sum()) - capacity) > 1e-8)
        duals = None

        def edges(allowed):
            ids = np.flatnonzero(allowed & (capacity > 0.0))
            if p * len(ids) <= edge_budget:
                return np.repeat(np.arange(p), len(ids)), np.tile(ids, p), True
            width = min(len(ids), max(1, edge_budget // max(2 * p, 1)))
            nearest = np.argpartition(c[:, ids], width - 1, axis=1)[:, :width]
            amortized = c[:, ids] + d[:, None] * density[ids]
            cheap = np.argpartition(amortized, width - 1, axis=1)[:, :width]
            rows = np.repeat(np.arange(p), width)
            bi, bj = np.nonzero(best_x)
            keep = allowed[bj]
            packed = np.concatenate((rows * m + ids[nearest.ravel()],
                                     rows * m + ids[cheap.ravel()],
                                     bi[keep] * m + bj[keep]))
            packed = np.unique(packed)
            return packed // m, packed % m, False

        def accept(y, x):
            nonlocal best_y, best_x, best_value, last_improvement, duals
            y = np.asarray(y, dtype=int).copy()
            if float(capacity @ y) < total - 1e-8:
                return False
            x = np.clip(np.asarray(x, dtype=float), 0.0, 1.0)
            x[:, y == 0] = 0.0
            sums = x.sum(axis=1)
            if not np.all(np.isfinite(sums)) or np.any(sums < 0.9):
                return False
            x /= sums[:, None]
            loads = d @ x
            slack = capacity * y - loads
            # Small capacity errors after integer rounding and row normalisation are repaired by row-sum-preserving transportation transfers.
            for j in np.flatnonzero(slack < -1e-9):
                for i in np.argsort(-x[:, j]):
                    if slack[j] >= -1e-9:
                        break
                    if time.perf_counter() >= deadline:
                        return False
                    if x[i, j] <= 0.0:
                        continue
                    destinations = np.flatnonzero((slack > 0.0) & (y > 0))
                    for k in destinations[np.argsort(c[i, destinations])]:
                        amount = min(-slack[j], float(x[i, j] * d[i]), float(slack[k]))
                        if amount <= 0.0:
                            continue
                        fraction = amount / d[i]
                        x[i, j] -= fraction
                        x[i, k] += fraction
                        slack[j] += amount
                        slack[k] -= amount
                        if slack[j] >= -1e-9 or x[i, j] <= 0.0:
                            break
            if np.min(slack) < -1e-7 or np.max(np.abs(x.sum(axis=1) - 1.0)) > 1e-9:
                return False
            y = ((np.sum(x, axis=0) > 0.0) | negative).astype(int)
            value = float(fixed @ y + np.sum(c * x))
            if value < best_value - 1e-10 * max(1.0, abs(best_value)):
                best_y, best_x, best_value = y, x, value
                last_improvement = time.perf_counter()
                duals = None
                return True
            return False

        def polish(duration):
            nonlocal duals
            if duration <= 0.002 or time.perf_counter() >= deadline:
                return
            try:
                y = best_y.copy()
                ii, jj, _ = edges(y.astype(bool))
                e = len(ii)
                k = np.arange(e)
                ae = coo_matrix((np.ones(e), (ii, k)), shape=(p, e)).tocsc()
                au = coo_matrix((d[ii] / scale_d, (jj, k)), shape=(m, e)).tocsc()
                remaining = min(duration, deadline - time.perf_counter())
                if remaining <= 0.002:
                    return
                result = linprog(c[ii, jj] / scale_c, A_ub=au,
                                 b_ub=capacity * y / scale_d, A_eq=ae,
                                 b_eq=np.ones(p), bounds=(0.0, 1.0), method="highs-ds",
                                 options={"time_limit": remaining, "presolve": True,
                                          "primal_feasibility_tolerance": 1e-9,
                                          "dual_feasibility_tolerance": 1e-9})
                if result.x is not None and result.success:
                    x = np.zeros((p, m))
                    x[ii, jj] = result.x
                    accept(y, x)
                    if np.array_equal(y, best_y):
                        duals = (np.asarray(result.eqlin.marginals) * scale_c,
                                 np.asarray(result.ineqlin.marginals) * scale_c / scale_d)
            except Exception:
                return

        polish(min(0.15 * budget, 0.15 * max(0.0, deadline - time.perf_counter())))

        def priorities():
            if duals is None:
                prices = np.sum(c * best_x, axis=1)
                closing = fixed.copy()
            else:
                prices, plant_prices = duals
                closing = fixed + capacity * plant_prices
            opening = seed_score.copy()
            # Compute the fractional-knapsack gain of a single candidate plant using the current transportation dual prices.
            for j in np.flatnonzero((best_y == 0) & (capacity > 0.0)):
                if time.perf_counter() >= deadline:
                    break
                benefit = prices - c[:, j]
                ids = np.flatnonzero(benefit > 0.0)
                if len(ids):
                    ids = ids[np.argsort(-(benefit[ids] / d[ids]))]
                    before = np.cumsum(d[ids]) - d[ids]
                    amount = np.maximum(0.0, np.minimum(d[ids], capacity[j] - before))
                    opening[j] = fixed[j] - float(np.sum(amount / d[ids] * benefit[ids]))
                else:
                    opening[j] = fixed[j]
            return closing, opening

        def build_model(ii, jj, lower_y, upper_y, free, radius):
            e = len(ii)
            arc = np.arange(e)
            max_fraction = np.minimum(1.0, capacity[jj] / d[ii])
            rows = np.concatenate((ii, p + jj, p + np.arange(m),
                                   p + m + arc, p + m + arc))
            cols = np.concatenate((m + arc, m + arc, np.arange(m), m + arc, jj))
            values = np.concatenate((np.ones(e), d[ii] / scale_d, -capacity / scale_d,
                                     np.ones(e), -max_fraction))
            a = coo_matrix((values, (rows, cols)), shape=(p + m + e, m + e)).tocsc()
            low = np.concatenate((np.ones(p), np.full(m + e, -np.inf)))
            high = np.concatenate((np.ones(p), np.zeros(m + e)))
            objective = np.concatenate((fixed, c[ii, jj])) / scale_c
            extra_rows, extra_low, extra_high = [], [], []

            def add(coefficients, lo, hi):
                extra_rows.append(csc_matrix(coefficients.reshape(1, -1)))
                extra_low.append(lo)
                extra_high.append(hi)

            row = np.zeros(m + e)
            row[:m] = capacity / total
            add(row, 1.0, np.inf)
            count = int(np.searchsorted(np.cumsum(np.sort(capacity)[::-1]), total)) + 1
            row = np.zeros(m + e)
            row[:m] = 1.0
            add(row, float(min(count, m)), np.inf)
            # For a plant subset S with insufficient capacity, truncate the capacity of its complement to obtain a valid cover inequality.
            for order in (np.argsort(seed_score), np.argsort(-(best_y * capacity))):
                used = np.zeros(m, dtype=bool)
                residual = total
                for j in order:
                    if 0.0 < capacity[j] < residual - 1e-8:
                        used[j] = True
                        residual -= capacity[j]
                if np.any(used) and residual > 1e-8:
                    row = np.zeros(m + e)
                    row[:m] = np.where(used, 0.0, np.minimum(capacity, residual) / residual)
                    add(row, 1.0, np.inf)
            add(objective.copy(), -np.inf, best_value / scale_c + 1e-8)
            if free is not None and radius < len(free):
                row = np.zeros(m + e)
                row[free] = 1.0 - 2.0 * best_y[free]
                add(row, -np.inf, float(radius - np.sum(best_y[free])))
            a = vstack([a] + extra_rows, format="csc")
            constraints = LinearConstraint(a, np.concatenate((low, extra_low)),
                                           np.concatenate((high, extra_high)))
            bounds = Bounds(np.concatenate((lower_y, np.zeros(e))),
                            np.concatenate((upper_y, max_fraction)))
            integrality = np.concatenate((np.ones(m, dtype=np.uint8), np.zeros(e, dtype=np.uint8)))
            return objective, integrality, bounds, constraints

        base_width = min(m, max(4, int(math.sqrt(m) * (1.0 + math.log1p(budget) / 3.0))))
        base_slice = max(0.02, 0.20 * math.sqrt(max(budget, 0.001)))
        base_radius = max(2, int(math.log1p(budget)) + 1)
        first = True
        closing, opening = priorities()
        # No iteration cap; on stagnation enlarge the release set, swap radius and per-solve time.
        while time.perf_counter() < deadline:
            remaining = deadline - time.perf_counter()
            if remaining <= 0.005:
                break
            heat = max(0.0, (time.perf_counter() - last_improvement) / base_slice)
            width = min(m, max(base_width, int(base_width * math.sqrt(1.0 + heat))))
            radius = min(width, max(base_radius, int(base_radius + math.sqrt(heat))))
            lower_y = mandatory.astype(float)
            upper_y = np.ones(m)
            free = None
            global_search = first or width >= m
            if not global_search:
                opened = np.flatnonzero((best_y > 0) & ~mandatory)
                closed = np.flatnonzero((best_y == 0) & (capacity > 0.0))
                opened = opened[np.argsort(-closing[opened])]
                closed = closed[np.argsort(opening[closed])]
                no = min(len(opened), max(1, width // 2))
                nc = min(len(closed), width - no)
                no = min(len(opened), width - nc)

                def sample_ranked(ids, size):
                    head = min(size, max(1, int(size / (1.4 + 0.1 * heat))))
                    if size <= head:
                        return ids[:size]
                    tail = rng.choice(ids[head:], size=size - head, replace=False)
                    return np.concatenate((ids[:head], tail))

                free = np.concatenate((sample_ranked(opened, no), sample_ranked(closed, nc)))
                lower_y = best_y.astype(float).copy()
                upper_y = lower_y.copy()
                lower_y[free] = 0.0
                upper_y[free] = 1.0
                lower_y[mandatory] = 1.0
                upper_y[mandatory] = 1.0
            ii, jj, full_edges = edges(upper_y > 0.0)
            model = build_model(ii, jj, lower_y, upper_y, free, radius)
            remaining = deadline - time.perf_counter()
            duration = min(0.96 * remaining,
                           0.58 * remaining if first else base_slice * (1.0 + 0.25 * heat))
            if duration <= 0.002:
                break
            was_first = first
            first = False
            try:
                objective, integrality, bounds, constraints = model
                result = milp(objective, integrality=integrality, bounds=bounds,
                              constraints=constraints,
                              options={"time_limit": duration, "mip_rel_gap": 0.0,
                                       "presolve": True, "disp": False})
                improved = False
                if result.x is not None:
                    y = (result.x[:m] > 0.5).astype(int)
                    x = np.zeros((p, m))
                    x[ii, jj] = result.x[m:]
                    improved = accept(y, x)
                # When the full model already has a proof of global optimality, stop useless search.
                if global_search and full_edges and result.status == 0 and result.x is not None:
                    break
                if improved or was_first:
                    polish(min(0.12 * max(0.0, deadline - time.perf_counter()), base_slice / 2.0))
                    closing, opening = priorities()
            except Exception:
                # Solver failure does not affect the feasible solution already kept; remaining time can still try other neighbourhoods.
                if time.perf_counter() < deadline:
                    polish(min(base_slice, 0.25 * (deadline - time.perf_counter())))
                closing, opening = priorities()
    except Exception:
        # When the optional optimisation phase is unavailable, return the independently constructed and stored feasible solution.
        pass

    return output(best_y, best_x)
