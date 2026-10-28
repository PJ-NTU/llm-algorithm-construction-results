"""Location neighbourhood algorithm with dual screening and dynamic arc addition
First stores a capacity-feasible assignment, then keeps improving within the wall clock via exact transportation subproblems, dual lower-bound screening, dynamic arc addition and integer-programming neighbourhoods.
"""

import math
import random
import time

try:
    import numpy as np
    from scipy.optimize import Bounds, LinearConstraint, linprog, milp
    from scipy.sparse import coo_matrix, csr_matrix
    _SCIPY = True
except Exception:
    _SCIPY = False


def _fallback_search(d, cap, fixed, cost, state, end, budget):
    """Without a numerical solver, still keep feasible joint location-assignment perturbations."""
    n, m = len(d), len(cap)
    rng = random.Random(1729)
    total = sum(d)
    positive = [i for i in range(n) if d[i] > 0]
    rankings = [sorted(range(m), key=lambda j: cost[i][j]) for i in range(n)]
    mutable = [j for j in range(m) if cap[j] > 0 and fixed[j] > 0]
    amplitude = max(1, min(m, int(math.sqrt(m) * math.log1p(budget) + 1)))
    while time.perf_counter() < end:
        y = list(state[1])
        if mutable:
            for j in rng.sample(mutable, rng.randint(1, min(len(mutable), amplitude))):
                y[j] = 1 - y[j]
        capacity = sum(cap[j] for j in range(m) if y[j])
        closed = [j for j in mutable if not y[j]]
        rng.shuffle(closed)
        for j in closed:
            if capacity >= total:
                break
            y[j] = 1
            capacity += cap[j]
        rem = [cap[j] if y[j] else 0.0 for j in range(m)]
        order = list(positive)
        rng.shuffle(order)
        x = [[0.0] * m for _ in range(n)]
        feasible = True
        for i in order:
            if time.perf_counter() >= end:
                return
            left = d[i]
            for j in rankings[i]:
                q = min(left, rem[j])
                if q > 0:
                    x[i][j] = q / d[i]
                    rem[j] -= q
                    left -= q
                if left <= 0:
                    break
            if left > 1e-8 * max(1.0, d[i]):
                feasible = False
                break
        if not feasible:
            continue
        for i in range(n):
            if d[i] == 0:
                x[i][rankings[i][0]] = 1.0
        y = [int(fixed[j] <= 0 or any(x[i][j] > 0 for i in positive)) for j in range(m)]
        value = sum(fixed[j] * y[j] for j in range(m))
        value += sum(cost[i][j] * x[i][j] for i in range(n) for j in range(m))
        if value < state[0]:
            state[:] = [value, y, x]


def _search(d0, cap0, fixed0, cost0, state, start, budget):
    clock = time.perf_counter
    n, m = len(d0), len(cap0)
    reserve = min(0.2 * budget, max(0.045 * budget, n * m * 6e-7))
    end = start + max(0.0, budget - reserve)
    if clock() >= end or not n or not m:
        return
    if not _SCIPY:
        _fallback_search(d0, cap0, fixed0, cost0, state, end, budget)
        return
    pos = np.flatnonzero(np.asarray(d0) > 0)
    p = len(pos)
    if not p:
        return
    d = np.asarray(d0, dtype=float)[pos]
    cap = np.asarray(cap0, dtype=float)
    fixed = np.asarray(fixed0, dtype=float)
    costs = np.asarray(cost0, dtype=float)[pos]
    total = float(np.sum(d))
    if float(np.sum(cap)) < total - 1e-8:
        return
    active = np.flatnonzero(cap > 0)
    effective = np.minimum(cap, total)
    mandatory = (fixed <= 0) | (np.sum(cap) - cap < total - 1e-8)
    density = np.full(m, np.inf)
    density[active] = np.maximum(fixed[active], 0) / effective[active]
    row_min = np.min(costs, axis=1)
    shifted = costs - row_min[:, None]
    zeros = [i for i in range(n) if d0[i] == 0]
    zero_ids = {i: min(range(m), key=lambda j: cost0[i][j]) for i in zeros}
    constant = float(np.sum(row_min)) + sum(cost0[i][j] for i, j in zero_ids.items())
    best_x = np.asarray(state[2], dtype=float)[pos].copy()
    best_y = np.asarray(state[1], dtype=np.uint8)
    rng = random.Random(1729)
    floor = max(0.002, budget * 0.0006)
    lp_time = max(floor, budget * 0.001)
    values_cache = {}
    dual_cache = {}
    cuts = []
    seen = set()
    relax_y = best_y.astype(float)
    arc_mask = None
    model = None
    phase_span = max(0.0, end - start)
    economy = density + np.mean(shifted / d[:, None], axis=0)

    def key(y):
        return np.asarray(y, dtype=np.uint8).tobytes()

    def repair(y):
        y = np.asarray(y, dtype=np.uint8).copy()
        y[mandatory] = 1
        y[(cap <= 0) & ~mandatory] = 0
        available = float(cap @ y)
        if available < total:
            for j in np.argsort(economy):
                if cap[j] > 0 and not y[j]:
                    y[j] = 1
                    available += cap[j]
                    if available >= total:
                        break
        return y

    def install(y, x):
        """Only after capacity scaling and shortfall flow completion may the stored complete solution be overwritten."""
        nonlocal best_x, best_y
        y = repair(y)
        x = np.asarray(x, dtype=float).copy()
        if x.shape != (p, m) or not np.all(np.isfinite(x)):
            return False
        x = np.clip(x, 0.0, 1.0)
        x[:, y == 0] = 0.0
        x[x < 1e-12] = 0.0
        sums = x.sum(axis=1)
        x /= np.where(sums > 0, sums, 1.0)[:, None]
        load = d @ x
        factors = np.ones(m)
        over = load > cap * y
        factors[over] = (cap * y)[over] / load[over]
        x *= factors[None, :]
        remaining = np.maximum(0.0, cap * y - d @ x)
        deficits = np.maximum(0.0, 1.0 - x.sum(axis=1)) * d
        ids = np.flatnonzero(y & (cap > 0))
        for i in np.flatnonzero(deficits > 0):
            left = deficits[i]
            for j in ids[np.argsort(costs[i, ids])]:
                q = min(left, remaining[j])
                if q > 0:
                    x[i, j] += q / d[i]
                    remaining[j] -= q
                    left -= q
                if left <= 0:
                    break
            if left > 1e-8 * max(1.0, d[i]):
                return False
        sums = x.sum(axis=1)
        if np.any(np.abs(sums - 1) > 1e-8):
            return False
        largest = np.argmax(x, axis=1)
        x[np.arange(p), largest] += 1.0 - sums
        if np.any(d @ x > cap * y + 1e-8 * np.maximum(1.0, cap)):
            return False
        y = ((np.any(x > 0, axis=0)) | mandatory).astype(np.uint8)
        value = constant + float(np.sum(shifted * x)) + float(fixed @ y)
        if value < state[0] - 1e-8:
            full = np.zeros((n, m), dtype=float)
            full[pos] = x
            for i, j in zero_ids.items():
                full[i, j] = 1.0
            best_x, best_y = x, y
            state[:] = [value, y.tolist(), full]
            return True
        return False

    def greedy(y, randomized=False):
        y = repair(y)
        ids = np.flatnonzero(y & (cap > 0))
        order_cost = np.argsort(costs[:, ids], axis=1)
        if len(ids) > 1:
            regret = (costs[np.arange(p), ids[order_cost[:, 1]]] -
                      costs[np.arange(p), ids[order_cost[:, 0]]]) / d
        else:
            regret = d.copy()
        if randomized:
            regret *= np.asarray([0.25 + 1.5 * rng.random() for _ in range(p)])
        order = np.argsort(-regret)
        rem = cap * y
        x = np.zeros((p, m))
        for i in order:
            if clock() >= end:
                return
            left = d[i]
            for j in ids[order_cost[i]]:
                q = min(left, rem[j])
                if q > 0:
                    x[i, j] = q / d[i]
                    rem[j] -= q
                    left -= q
                if left <= 0:
                    break
        install(y, x)

    def lower_bound(y):
        bound = -math.inf
        for intercept, coef in cuts:
            bound = max(bound, intercept + float(coef @ y))
        return bound

    def transport(y, stop, force=False):
        nonlocal lp_time
        y = np.asarray(y, dtype=np.uint8)
        k0 = key(y)
        if not force and k0 in values_cache:
            return values_cache[k0]
        if cap @ y < total - 1e-8 or stop - clock() <= floor:
            return math.inf
        if not force and cuts and lower_bound(y) >= state[0] - 1e-7:
            return math.inf
        ids = np.flatnonzero(y & (cap > 0))
        k = len(ids)
        e = p * k
        cols = np.arange(e)
        rr = np.repeat(np.arange(p), k)
        jj = np.tile(np.arange(k), p)
        eq = coo_matrix((np.ones(e), (rr, cols)), shape=(p, e)).tocsr()
        ub = coo_matrix((d[rr] / effective[ids[jj]], (jj, cols)), shape=(k, e)).tocsr()
        scale = max(1.0, float(np.max(shifted[:, ids])))
        available = min(stop - clock(), max(4 * lp_time, budget * 0.026))
        if available <= floor:
            return math.inf
        began = clock()
        try:
            result = linprog(shifted[:, ids].ravel() / scale, A_ub=ub, b_ub=np.ones(k),
                             A_eq=eq, b_eq=np.ones(p), bounds=(0.0, None),
                             method="highs-ds", options={"time_limit": available * 0.8,
                                                         "presolve": True})
            lp_time = max(floor, 0.65 * lp_time + 0.35 * (clock() - began))
            if result.x is None:
                return math.inf
            x = np.zeros((p, m))
            x[:, ids] = result.x.reshape(p, k)
            install(y, x)
            value = constant + float(result.fun) * scale + float(fixed @ y)
            if result.status == 0:
                values_cache[k0] = value
                u = np.asarray(result.eqlin.marginals) * scale
                v = np.zeros(m)
                v[ids] = np.asarray(result.ineqlin.marginals) * scale
                dual_cache[k0] = (u, v)
                # Independent per-plant transportation relaxations give a valid lower bound for any set of open plants.
                profit = np.maximum(u[:, None] - shifted, 0.0)
                gain = np.minimum(np.sum(profit, axis=0),
                                  effective * np.max(profit / d[:, None], axis=0))
                gain[ids] = np.maximum(0.0, -v[ids])
                cuts.append((constant + float(np.sum(u)), fixed - gain))
            return value
        except Exception:
            lp_time = max(lp_time, clock() - began)
            return math.inf

    def neighborhood(stop):
        """Screen single opens, single closes and swaps by exact transportation duals, recentring immediately after an improvement."""
        before_key = key(best_y)
        base = best_y.copy()
        if before_key not in dual_cache:
            transport(base, stop, force=True)
        if clock() >= stop:
            return
        base = best_y.copy()
        base_key = key(base)
        opened = np.flatnonzero((base == 1) & ~mandatory & (cap > 0))
        closed = np.flatnonzero((base == 0) & (cap > 0))
        info = dual_cache.get(base_key)
        close_scores = -fixed.copy()
        open_scores = fixed.copy()
        if info is not None:
            u, v = info
            profit = np.maximum(u[:, None] - shifted, 0.0)
            gain = np.minimum(profit.sum(axis=0), effective * np.max(profit / d[:, None], axis=0))
            open_scores = fixed - gain
            close_scores = -fixed - v
            ids = np.flatnonzero(base & (cap > 0))
            if len(ids) > 1:
                adjusted = shifted[:, ids] - d[:, None] * (v[ids] / effective[ids])[None, :]
                order = np.argpartition(adjusted, 1, axis=1)[:, :2]
                vals = np.take_along_axis(adjusted, order, axis=1)
                swap = vals[:, 0] > vals[:, 1]
                first = np.where(swap, order[:, 1], order[:, 0])
                gap = np.abs(vals[:, 1] - vals[:, 0])
                close_scores += np.bincount(ids[first], weights=gap, minlength=m)
            # Solve an independent fractional knapsack for open-plant candidates, sharpening the ranking when capacity is scarce.
            for j in closed[np.argsort(open_scores[closed])]:
                if clock() >= stop:
                    return
                q = profit[:, j]
                order = np.argsort(-(q / d))
                remaining = effective[j]
                gain_j = 0.0
                for i in order:
                    if q[i] <= 0 or remaining <= 0:
                        break
                    amount = min(remaining, d[i])
                    gain_j += q[i] * amount / d[i]
                    remaining -= amount
                open_scores[j] = fixed[j] - gain_j
        opened = opened[np.argsort(close_scores[opened])]
        closed = closed[np.argsort(open_scores[closed])]
        single = [(float(close_scores[j]), int(j)) for j in opened]
        single += [(float(open_scores[j]), int(j)) for j in closed]
        single.sort()
        for _, j in single:
            if stop - clock() <= max(floor, lp_time * 1.25):
                return
            trial = base.copy()
            trial[j] ^= 1
            trial_key = key(trial)
            if trial_key in seen or cap @ trial < total - 1e-8:
                continue
            seen.add(trial_key)
            transport(trial, stop)
            if key(best_y) != base_key:
                return
        # Swaps are enumerated lazily by the sum of the two marginal scores, without creating a quadratic-size candidate table.
        import heapq
        heap = []
        if len(closed):
            for a in opened:
                heapq.heappush(heap, (float(close_scores[a] + open_scores[closed[0]]), int(a), 0))
        while heap and stop - clock() > max(floor, lp_time * 1.25):
            _, a, bidx = heapq.heappop(heap)
            b = closed[bidx]
            trial = base.copy()
            trial[a], trial[b] = 0, 1
            trial_key = key(trial)
            if trial_key not in seen and cap @ trial >= total - 1e-8:
                seen.add(trial_key)
                transport(trial, stop)
                if key(best_y) != base_key:
                    return
            bidx += 1
            if bidx < len(closed):
                heapq.heappush(heap, (float(close_scores[a] + open_scores[closed[bidx]]), a, bidx))

    # Capacity economics and transportation economics produce complementary initial facility sets.
    seed_stop = min(end, start + 0.14 * phase_span)
    transport(best_y, seed_stop, force=True)
    for scores in (economy, density, np.mean(shifted, axis=0)):
        if clock() >= seed_stop:
            break
        y = mandatory.astype(np.uint8)
        available = float(cap @ y)
        for j in np.argsort(scores):
            if available >= total:
                break
            if cap[j] > 0 and not y[j]:
                y[j] = 1
                available += cap[j]
        greedy(y)
        transport(y, seed_stop)
    if clock() < seed_stop:
        transport(np.ones(m, dtype=np.uint8), seed_stop)
    while clock() < seed_stop:
        before = state[0]
        neighborhood(seed_stop)
        if state[0] >= before - 1e-8:
            break
    if clock() >= end:
        return

    # Large models start only from necessary arcs; relaxation pricing and later arc expansion keep the candidate set from being permanently closed.
    edge_budget = max(2 * p, int(9500 * max(budget, 0.01)))
    width = min(len(active), max(2, int(math.sqrt(m) * math.log1p(budget))))
    if p * len(active) <= edge_budget:
        width = len(active)
    else:
        width = min(width, max(2, edge_budget // max(1, 3 * p)))
    arc_mask = np.zeros((p, m), dtype=bool)
    if width >= len(active):
        arc_mask[:, active] = True
    else:
        adjusted = shifted[:, active] / d[:, None] + density[active][None, :]
        nearest = np.argpartition(shifted[:, active], width - 1, axis=1)[:, :width]
        cheap = np.argpartition(adjusted, width - 1, axis=1)[:, :width]
        rows = np.arange(p)[:, None]
        arc_mask[rows, active[nearest]] = True
        arc_mask[rows, active[cheap]] = True
    arc_mask |= best_x > 0
    model_dirty = True

    def build():
        nonlocal model, model_dirty
        missing = (best_x > 0) & ~arc_mask
        if np.any(missing):
            arc_mask[:] |= missing
            model_dirty = True
        if model is not None and not model_dirty:
            return model
        a, b = np.nonzero(arc_mask)
        e = len(a)
        z = np.arange(e)
        upper = np.minimum(1.0, effective[b] / d[a])
        eq = coo_matrix((np.ones(e), (a, m + z)), shape=(p, m + e)).tocsr()
        rr = np.concatenate((b, np.arange(m), m + z, m + z,
                             np.full(m, m + e), np.full(m, m + e + 1)))
        jj = np.concatenate((m + z, np.arange(m), m + z, b, np.arange(m), np.arange(m)))
        vv = np.concatenate((d[a] / effective[b], -np.ones(m), np.ones(e), -upper,
                             -effective / total, -np.ones(m)))
        ub = coo_matrix((vv, (rr, jj)), shape=(m + e + 2, m + e)).tocsr()
        rhs = np.zeros(m + e + 2)
        rhs[-2] = -1.0
        sums = np.cumsum(np.sort(effective)[::-1])
        rhs[-1] = -min(m, int(np.searchsorted(sums, total - 1e-9)) + 1)
        c = np.concatenate((fixed, shifted[a, b]))
        scale = max(1.0, float(np.max(np.abs(c))))
        c /= scale
        lo = np.zeros(m + e)
        lo[:m] = mandatory
        hi = np.concatenate((np.ones(m), upper))
        hi[:m][(cap <= 0) & ~mandatory] = 0.0
        integer = np.zeros(m + e, dtype=np.uint8)
        integer[:m] = 1
        model = (a, b, eq, ub, rhs, c, scale, lo, hi, integer)
        model_dirty = False
        return model

    def relaxation(stop):
        nonlocal relax_y, model_dirty
        if stop - clock() <= floor:
            return False
        a, b, eq, ub, rhs, c, scale, lo, hi, _ = build()
        available = stop - clock()
        if available <= floor:
            return False
        try:
            result = linprog(c, A_ub=ub, b_ub=rhs, A_eq=eq, b_eq=np.ones(p),
                             bounds=np.column_stack((lo, hi)), method="highs",
                             options={"time_limit": 0.8 * available, "presolve": True})
            if result.x is None:
                return False
            relax_y = np.clip(result.x[:m], 0.0, 1.0)
            if result.status != 0 or np.all(arc_mask[:, active]):
                return False
            alpha = np.asarray(result.eqlin.marginals) * scale
            beta = np.asarray(result.ineqlin.marginals[:m]) * scale
            reduced = shifted[:, active] - alpha[:, None] - d[:, None] * (beta[active] / effective[active])[None, :]
            reduced[arc_mask[:, active]] = np.inf
            add = max(1, min(len(active), int(math.sqrt(m) * math.log1p(max(0.0, stop - clock())))))
            indices = np.argpartition(reduced, add - 1, axis=1)[:, :add]
            rows = np.repeat(np.arange(p), add)
            cols = active[indices.ravel()]
            good = reduced[np.arange(p)[:, None], indices].ravel() < -1e-7 * scale
            if not np.any(good):
                return False
            arc_mask[rows[good], cols[good]] = True
            model_dirty = True
            return True
        except Exception:
            return False

    def integer_search(stop, mode, stalled):
        if stop - clock() <= floor:
            return False
        a, b, eq, ub, rhs, c, scale, lo0, hi0, integer = build()
        lo, hi = lo0.copy(), hi0.copy()
        constraints = [LinearConstraint(eq, np.ones(p), np.ones(p)),
                       LinearConstraint(ub, -np.inf, rhs)]
        cutoff = (state[0] - constant) / scale
        constraints.append(LinearConstraint(csr_matrix(c[None, :]), -np.inf,
                                            cutoff + 1e-9 * max(1.0, abs(cutoff))))
        strength = math.log1p(budget) + math.log1p(stalled)
        if mode == 1:
            # Relaxation disagreement and current customer service areas jointly determine the free facilities, rather than random fixing.
            size = min(m, max(2, int(math.sqrt(m) * (1.0 + strength))))
            paid = np.sum(shifted * best_x, axis=1)
            benefit = np.maximum(paid[:, None] - shifted, 0).sum(axis=0)
            importance = np.abs(relax_y - best_y)
            denominator = max(1.0, float(np.max(np.abs(fixed))), float(np.max(benefit)))
            importance += (benefit - fixed) / denominator
            importance += np.asarray([rng.random() for _ in range(m)]) * (0.15 + 0.1 * strength)
            opened = np.flatnonzero(best_y & ~mandatory)
            closed = np.flatnonzero((best_y == 0) & (cap > 0))
            first = opened[np.argsort(-importance[opened])][:max(1, size // 2)]
            second = closed[np.argsort(-importance[closed])][:max(1, size - len(first))]
            free = np.concatenate((first, second))
            locked = np.ones(m, dtype=bool)
            locked[free] = False
            ids = np.flatnonzero(locked)
            lo[ids], hi[ids] = best_y[ids], best_y[ids]
        elif mode == 2:
            radius = min(m, max(2, int(strength + math.sqrt(1.0 + stalled))))
            coefficients = np.where(best_y, -1.0, 1.0)
            row = coo_matrix((coefficients, (np.zeros(m, dtype=int), np.arange(m))),
                             shape=(1, len(c))).tocsr()
            constraints.append(LinearConstraint(row, -np.inf, radius - int(best_y.sum())))
        available = stop - clock()
        if available <= floor:
            return False
        before = state[0]
        try:
            result = milp(c, integrality=integer, bounds=Bounds(lo, hi), constraints=constraints,
                          options={"time_limit": 0.82 * available, "mip_rel_gap": 0.0,
                                   "presolve": True, "disp": False})
            if result.x is not None:
                y = (result.x[:m] >= 0.5).astype(np.uint8)
                x = np.zeros((p, m))
                x[a, b] = np.maximum(0.0, result.x[m:])
                install(y, x)
            return state[0] < before - 1e-8
        except Exception:
            return False

    root_stop = min(end, max(clock(), start + 0.25 * phase_span))
    while root_stop - clock() > floor:
        if not relaxation(root_stop):
            break
    round_stop = min(end, clock() + 0.045 * budget)
    y = mandatory.astype(np.uint8)
    available = float(cap @ y)
    rank = np.lexsort((economy, -relax_y))
    for j in rank:
        if available >= total and relax_y[j] < 0.5:
            break
        if cap[j] > 0 and not y[j]:
            y[j] = 1
            available += cap[j]
    y = repair(y)
    if clock() < round_stop:
        greedy(y)
        transport(y, round_stop)
        transport(best_y, round_stop)
    if clock() < end:
        integer_search(min(end, clock() + 0.44 * (end - clock())), 0, 0)

    turn, stalled = 0, 0
    while clock() < end:
        remaining = end - clock()
        if remaining <= max(3 * floor, 1.3 * lp_time):
            # In the final stretch, do not start native searches that cannot finish in time; just keep the solution already obtained.
            y = best_y.copy()
            mutable = np.flatnonzero(~mandatory & (cap > 0))
            if not len(mutable):
                return
            count = min(len(mutable), max(1, int(math.sqrt(m) * math.log1p(budget))))
            for j in rng.sample(mutable.tolist(), rng.randint(1, count)):
                y[j] ^= 1
            greedy(repair(y), True)
            continue
        before = state[0]
        chunk = min(remaining, max(5 * lp_time, budget / (3 + math.sqrt(m)) *
                                  (1.0 + 0.25 * math.log1p(stalled))))
        stop = min(end, clock() + chunk)
        mode = turn % 4
        if mode == 0:
            neighborhood(stop)
            if clock() < stop and state[0] >= before - 1e-8:
                # Open one competing facility and randomly close several facilities, then reassign transportation as a whole.
                y = best_y.copy()
                mutable = np.flatnonzero(~mandatory & (cap > 0)).tolist()
                count = min(len(mutable), max(1, int(math.sqrt(m) * math.log1p(budget + stalled))))
                if count:
                    for j in rng.sample(mutable, rng.randint(1, count)):
                        y[j] ^= 1
                y = repair(y)
                greedy(y, True)
                transport(y, stop)
        elif mode == 3:
            if not np.all(arc_mask[:, active]):
                if not relaxation(min(stop, clock() + 0.28 * chunk)):
                    width = min(len(active), max(width + 1, int(width * (1.0 + 1.0 / math.sqrt(1 + budget)))))
                    near = np.argpartition(shifted[:, active], width - 1, axis=1)[:, :width]
                    arc_mask[np.arange(p)[:, None], active[near]] = True
                    model_dirty = True
            integer_search(stop, 0, stalled)
        else:
            integer_search(stop, mode, stalled)
        if state[0] < before - 1e-8:
            stalled = 0
            if end - clock() > max(3 * floor, lp_time * 2):
                transport(best_y, min(end, clock() + max(3 * lp_time, budget * 0.018)))
        else:
            stalled += 1
        turn += 1


def solve(instance: dict, time_limit_s: float) -> dict:
    """Return the best known feasible solution covering all customers and plants."""
    start = time.perf_counter()
    n = m = 0
    state = None
    try:
        n = max(0, int(instance["num_customers"]))
        m = max(0, int(instance["num_plants"]))
        d = [float(v) for v in instance["customers_demands"]]
        cap = [max(0.0, float(v)) for v in instance["plants_capacities"]]
        fixed = [float(v) for v in instance["plants_fixed_costs"]]
        cost = instance["shipping_costs"]
        x = [[0.0] * m for _ in range(n)]
        remaining = list(cap)
        used = [False] * m
        cursor = 0
        for i in range(n):
            if d[i] <= 0:
                if m:
                    x[i][min(range(m), key=lambda j: cost[i][j])] = 1.0
                continue
            left = d[i]
            last = -1
            while left > 0 and cursor < m:
                q = min(left, remaining[cursor])
                if q > 0:
                    x[i][cursor] = q / d[i]
                    remaining[cursor] -= q
                    left -= q
                    last = cursor
                    used[cursor] = True
                if remaining[cursor] <= 0:
                    cursor += 1
            if last >= 0:
                x[i][last] += 1.0 - sum(x[i])
            elif m:
                x[i][0] = 1.0
                used[0] = True
        y = [int(used[j] or fixed[j] <= 0) for j in range(m)]
        value = sum(fixed[j] * y[j] for j in range(m))
        value += sum(float(cost[i][j]) * x[i][j] for i in range(n) for j in range(m))
        state = [value, y, x]
        budget = max(0.0, float(time_limit_s))
        if math.isfinite(budget) and budget > 0:
            _search(d, cap, fixed, cost, state, start, budget)
    except Exception:
        # Exceptions in the optimisation phase must not corrupt the stored capacity-feasible solution.
        pass
    if state is None:
        return {"open_plants": [1] * m,
                "assignments": {str(i): {str(j): float(j == 0) for j in range(m)} for i in range(n)}}
    return {"open_plants": [int(v) for v in state[1]],
            "assignments": {str(i): {str(j): float(state[2][i][j]) for j in range(m)} for i in range(n)}}
