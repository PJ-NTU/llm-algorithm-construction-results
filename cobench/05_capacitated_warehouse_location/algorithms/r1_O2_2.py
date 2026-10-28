"""Relaxation-guided strengthened integer programming
Uses a feasible construction as a safeguard, strengthens the model with linking constraints, and alternately searches the relaxation support set, joint open/close neighbourhoods and the full model.
"""

import math
import random
import time


def _construct(demand, capacity, fixed, shipping):
    n, m = len(demand), len(capacity)
    x = [[0.0] * m for _ in range(n)]
    y = [int(v <= 0.0) for v in fixed]
    if not m:
        return y, x
    remaining = capacity[:]
    density = [max(0.0, fixed[j]) / max(capacity[j], 1e-30)
               for j in range(m)]

    def urgency(i):
        costs = sorted(shipping[i][j] / demand[i] + density[j]
                       for j in range(m) if capacity[j] > 0.0)
        return costs[1] - costs[0] if len(costs) > 1 else float('inf')

    order = sorted((i for i in range(n) if demand[i] > 0.0),
                   key=urgency, reverse=True)
    for i in order:
        left = demand[i]
        while left > 0.0:
            available = [j for j in range(m) if remaining[j] > 0.0]
            if not available:
                break
            j = min(available, key=lambda j: shipping[i][j] / demand[i]
                    + (0.0 if y[j] else density[j]))
            amount = min(left, remaining[j])
            x[i][j] += amount / demand[i]
            remaining[j] -= amount
            left -= amount
            y[j] = 1
    for i in range(n):
        if demand[i] == 0.0:
            x[i][min(range(m), key=lambda j: shipping[i][j])] = 1.0
    return y, x


def _search(demand, capacity, fixed, shipping, best, deadline, budget):
    import numpy as np
    from scipy.optimize import Bounds, LinearConstraint, linprog, milp
    from scipy.sparse import coo_matrix, csc_matrix, vstack

    clock = time.perf_counter
    if clock() >= deadline:
        return
    d = np.asarray(demand, dtype=float)
    cap = np.asarray(capacity, dtype=float)
    f = np.asarray(fixed, dtype=float)
    c = np.asarray(shipping, dtype=float)
    n, m = c.shape
    total = float(d.sum())
    if total <= 0.0:
        return
    count = n * m
    size = m + count
    positive = np.flatnonzero(d > 0.0)
    zero = np.flatnonzero(d == 0.0)
    effective_cap = np.minimum(cap, total)
    scale_cap = np.maximum(effective_cap, float(d.max()))
    variable_ids = np.arange(count, dtype=np.int32).reshape(n, m)
    plant_ids = np.tile(np.arange(m, dtype=np.int32), n)
    customer_ids = np.repeat(np.arange(n, dtype=np.int32), m)
    eq = coo_matrix((np.ones(count), (customer_ids, variable_ids.ravel())),
                    shape=(n, count)).tocsc()
    transportation = coo_matrix(
        ((d[:, None] / scale_cap).ravel(),
         (plant_ids, variable_ids.ravel())), shape=(m, count)).tocsc()
    transportation.eliminate_zeros()
    customer_rows = coo_matrix(
        (np.ones(count), (customer_ids, m + variable_ids.ravel())),
        shape=(n, size)).tocsc()
    capacity_rows = coo_matrix(
        (np.concatenate(((d[:, None] / scale_cap).ravel(),
                         -effective_cap / scale_cap)),
         (np.concatenate((plant_ids, np.arange(m))),
          np.concatenate((m + variable_ids.ravel(), np.arange(m))))),
        shape=(m, size)).tocsc()
    capacity_rows.eliminate_zeros()

    # With positive demand, this linking is a valid strengthening of the original binary model; zero-demand rows must not get the linking.
    pcount = len(positive) * m
    link_ids = np.arange(pcount, dtype=np.int32)
    links = coo_matrix(
        (np.concatenate((np.ones(pcount),
                         -np.minimum(1.0, effective_cap / d[positive, None]).ravel())),
         (np.concatenate((link_ids, link_ids)),
          np.concatenate((m + variable_ids[positive].ravel(),
                          np.tile(np.arange(m), len(positive)))))),
        shape=(pcount, size)).tocsc()

    lower = np.zeros(size)
    upper = np.ones(size)
    lower[:m][f <= 0.0] = 1.0
    upper[:m][(cap == 0.0) & (f > 0.0)] = 0.0
    necessary = effective_cap.sum() - effective_cap < total - 1e-9
    lower[:m][necessary] = 1.0
    upper[m:] = np.where(d[:, None] > 0.0,
                        np.minimum(1.0, effective_cap / np.maximum(d[:, None], 1e-30)),
                        1.0).ravel()
    # Zero-demand customers can be fixed directly to the lowest-freight column, independent of plant open status.
    for i in zero:
        upper[m + variable_ids[i]] = 0.0
        j = int(np.argmin(c[i]))
        lower[m + variable_ids[i, j]] = 1.0
        upper[m + variable_ids[i, j]] = 1.0

    rows = [customer_rows, capacity_rows, links]
    row_lower = [np.ones(n), np.full(m + pcount, -np.inf)]
    row_upper = [np.ones(n), np.zeros(m + pcount)]
    residual = total - float(effective_cap[lower[:m] > 0.5].sum())
    mutable = np.flatnonzero((lower[:m] < 0.5) & (upper[:m] > 0.5))
    if residual > 1e-9:
        cover = np.zeros(size)
        cover[mutable] = np.minimum(effective_cap[mutable], residual) / residual
        descending = np.sort(effective_cap[mutable])[::-1]
        minimum_count = int(np.searchsorted(np.cumsum(descending), residual - 1e-9)) + 1
        cardinality = np.zeros(size)
        cardinality[mutable] = 1.0
        rows.extend((csc_matrix(cover[None, :]), csc_matrix(cardinality[None, :])))
        row_lower.append(np.array([1.0, float(minimum_count)]))
        row_upper.append(np.array([np.inf, np.inf]))
    matrix = vstack(rows, format='csc')
    row_lower = np.concatenate(row_lower)
    row_upper = np.concatenate(row_upper)
    constraints = LinearConstraint(matrix, row_lower, row_upper)
    raw_objective = np.concatenate((f, c.ravel()))
    cost_scale = max(1.0, float(np.mean(np.abs(raw_objective))))
    objective = raw_objective / cost_scale
    integer = np.zeros(size, dtype=np.uint8)
    integer[:m] = 1
    rng = random.Random(73129)
    guide = np.asarray(best['y'], dtype=float)
    mutable_list = mutable.tolist()

    def accept(y_value, x_value):
        if x_value is None:
            return False
        y_value = np.asarray(y_value, dtype=float)
        if not np.all(np.isfinite(y_value)):
            return False
        if np.max(np.abs(y_value - np.rint(y_value))) > 1e-5:
            return False
        y = np.rint(y_value).astype(int)
        y[f <= 0.0] = 1
        x = np.asarray(x_value, dtype=float).reshape(n, m).copy()
        if not np.all(np.isfinite(x)) or np.min(x) < -1e-5 or np.max(x) > 1.00001:
            return False
        np.clip(x, 0.0, 1.0, out=x)
        x[np.ix_(positive, np.flatnonzero(y == 0))] = 0.0
        sums = x.sum(axis=1)
        if np.max(np.abs(sums - 1.0)) > 1e-5:
            return False
        x /= sums[:, None]
        loads = d @ x
        slack = cap * y - loads
        # Only repair numerical errors: transfer supply in pairs, preserving each customer's fraction sum.
        for j in np.flatnonzero(slack < 0.0):
            excess = -float(slack[j])
            if excess > 1e-5 * max(1.0, cap[j]):
                return False
            for i in np.flatnonzero((d > 0.0) & (x[:, j] > 0.0)):
                if excess <= 1e-10:
                    break
                for k in np.flatnonzero(slack > 1e-12):
                    amount = min(excess, float(x[i, j] * d[i]), float(slack[k]))
                    if amount <= 0.0:
                        continue
                    fraction = amount / d[i]
                    x[i, j] -= fraction
                    x[i, k] += fraction
                    slack[k] -= amount
                    slack[j] += amount
                    excess -= amount
                    if excess <= 1e-10 or x[i, j] <= 0.0:
                        break
            if excess > 1e-7:
                return False
        loads = d @ x
        if np.max(loads - cap * y) > 1e-7:
            return False
        y[(loads == 0.0) & (f > 0.0)] = 0
        value = float(f @ y + np.sum(c * x))
        if value < best['cost'] - 1e-9:
            best.update(y=y.tolist(), x=x, cost=value)
            return True
        return False

    def price(y, allowance):
        remaining = deadline - clock()
        if remaining <= 0.0 or allowance <= 0.0:
            return
        y = np.asarray(y, dtype=int)
        if float(effective_cap @ y) < total - 1e-8:
            return
        ub = upper[m:].reshape(n, m).copy()
        ub[np.ix_(positive, np.flatnonzero(y == 0))] = 0.0
        result = linprog(
            objective[m:], A_ub=transportation,
            b_ub=effective_cap * y / scale_cap,
            A_eq=eq, b_eq=np.ones(n),
            bounds=np.column_stack((lower[m:], ub.ravel())), method='highs',
            options={'time_limit': max(1e-4, min(allowance, remaining) * 0.9),
                     'primal_feasibility_tolerance': 1e-8,
                     'dual_feasibility_tolerance': 1e-8})
        if result.x is not None:
            accept(y, result.x)

    def integer_search(lb, ub, allowance):
        remaining = deadline - clock()
        if remaining <= 0.0 or allowance <= 0.0:
            return None
        incumbent = best['cost'] / cost_scale
        cutoff = LinearConstraint(objective[None, :], -np.inf,
                                  incumbent + 1e-7 * max(1.0, abs(incumbent)))
        result = milp(
            objective, integrality=integer, bounds=Bounds(lb, ub),
            constraints=[constraints, cutoff],
            options={'time_limit': max(1e-4, min(allowance, remaining) * 0.9),
                     'mip_rel_gap': 0.0, 'presolve': True})
        if result.x is not None:
            accept(result.x[:m], result.x[m:])
        return result

    if clock() >= deadline:
        return
    # The native linear program provides plant-selection signals that jointly consider fixed costs and split-flow costs.
    relaxation = milp(
        objective, integrality=np.zeros(size, dtype=np.uint8),
        bounds=Bounds(lower, upper), constraints=constraints,
        options={'time_limit': max(1e-4, (deadline - clock()) * 0.15),
                 'presolve': True})
    if relaxation.x is not None:
        guide = np.clip(relaxation.x[:m], 0.0, 1.0)
        accept(relaxation.x[:m], relaxation.x[m:])
        rounded = np.maximum(lower[:m], (guide >= 0.5).astype(float)).astype(int)
        covered = float(effective_cap @ rounded)
        for j in sorted(mutable_list, key=lambda j: (-guide[j], f[j] / max(cap[j], 1e-30))):
            if covered >= total - 1e-9:
                break
            if not rounded[j]:
                rounded[j] = 1
                covered += effective_cap[j]
        price(rounded, (deadline - clock()) * 0.09)
    else:
        price(best['y'], (deadline - clock()) * 0.09)

    # First let plants in the relaxation support set and the safeguard solution compete jointly, rather than flipping plant by plant.
    if clock() < deadline:
        supported = (guide > 1e-7) | (np.asarray(best['y']) > 0)
        breadth = min(len(mutable), max(1, int(math.ceil(
            math.sqrt(max(1, len(mutable))) * math.log1p(max(0.0, budget))))))
        outsiders = [j for j in mutable_list if not supported[j]]
        rng.shuffle(outsiders)
        for j in outsiders[:breadth]:
            supported[j] = True
        ub = upper.copy()
        ub[:m][~supported] = 0.0
        integer_search(lower, ub, (deadline - clock()) * 0.25)

    turn = 0
    stagnation = 0
    # No restart cap; the time slice and release size of each solve are determined by the budget and remaining time.
    while clock() < deadline:
        remaining = deadline - clock()
        before = best['cost']
        full = turn % 3 == 0 or len(mutable) == 0
        if full:
            lb, ub = lower, upper
        else:
            incumbent = np.asarray(best['y'], dtype=float)
            width = min(len(mutable), max(2, int(math.ceil(
                math.sqrt(max(1, len(mutable)))
                * (1.0 + math.log1p(max(0.0, budget)))
                * (1.0 + math.sqrt(stagnation) / (1.0 + math.sqrt(m)))))))
            priority = [(abs(guide[j] - incumbent[j]) + rng.random(), j)
                        for j in mutable_list]
            priority.sort(reverse=True)
            free = [j for _, j in priority[:width]]
            lb, ub = lower.copy(), upper.copy()
            lb[:m] = incumbent
            ub[:m] = incumbent
            lb[free] = lower[free]
            ub[free] = upper[free]
        allowance = min(remaining, max(budget * 0.05, remaining * 0.68))
        result = integer_search(lb, ub, allowance)
        turn += 1
        stagnation = 0 if best['cost'] < before - 1e-9 else stagnation + 1
        # When the full model has proven optimality, there is no need to waste the remaining time.
        if full and result is not None and result.status == 0 and result.x is not None:
            return


def solve(instance: dict, time_limit_s: float) -> dict:
    started = time.perf_counter()
    n = m = 0
    best = {'y': [], 'x': [], 'cost': float('inf')}
    try:
        n = int(instance['num_customers'])
        m = int(instance['num_plants'])
        demand = [float(v) for v in instance['customers_demands']]
        capacity = [float(v) for v in instance['plants_capacities']]
        fixed = [float(v) for v in instance['plants_fixed_costs']]
        shipping = [[float(v) for v in row] for row in instance['shipping_costs']]
        y, x = _construct(demand, capacity, fixed, shipping)
        value = sum(fixed[j] * y[j] for j in range(m))
        value += sum(shipping[i][j] * x[i][j] for i in range(n) for j in range(m))
        best.update(y=y, x=x, cost=value)
        budget = max(0.0, float(time_limit_s))
        deadline = started + 0.93 * budget
        if n and m and sum(demand) > 0.0 and time.perf_counter() < deadline:
            try:
                _search(demand, capacity, fixed, shipping, best, deadline, budget)
            except Exception:
                pass
    except Exception:
        # Valid feasible inputs take the construction above; on exceptions all agreed keys and dimensions are still kept.
        if len(best['y']) != m:
            best['y'] = [1] * m
        if len(best['x']) != n:
            best['x'] = [[1.0 if j == 0 else 0.0 for j in range(m)] for _ in range(n)]
    return {
        'open_plants': [int(v) for v in best['y']],
        'assignments': {str(i): {str(j): float(best['x'][i][j]) for j in range(m)}
                        for i in range(n)},
    }
