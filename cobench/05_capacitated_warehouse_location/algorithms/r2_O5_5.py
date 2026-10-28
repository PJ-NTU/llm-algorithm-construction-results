"""Dual pricing and joint customer-plant neighbourhood search
First stores a feasible assignment, then improves via transportation-dual arc addition, capacity-cover strengthening and time-limited integer neighbourhoods that release customers and plants simultaneously.
"""

import math
import random
import time

try:
    import numpy as np
except Exception:
    np = None

try:
    from scipy.optimize import Bounds, LinearConstraint, linprog, milp
    from scipy.sparse import coo_matrix, vstack
    _OPT = True
except Exception:
    _OPT = False


def _search(demand, capacity, fixed, shipping, state, started, budget):
    clock = time.perf_counter
    n, m = len(demand), len(capacity)
    elapsed = clock() - started
    reserve = min(0.35 * budget,
                  max(0.18 * budget, 4.0 * elapsed, n * m * 2.5e-7))
    end = started + max(0.0, budget - reserve)
    if np is None or clock() >= end or not n or not m:
        return
    pos = np.flatnonzero(np.asarray(demand) > 0.0)
    p = len(pos)
    if not p:
        return
    d = np.asarray(demand, dtype=float)[pos]
    cap = np.asarray(capacity, dtype=float)
    f = np.asarray(fixed, dtype=float)
    raw = np.asarray(shipping, dtype=float)[pos]
    c = raw - np.min(raw, axis=1)[:, None]
    total = float(d.sum())
    if float(cap.sum()) < total - 1e-8:
        return
    active = np.flatnonzero(cap > 0.0)
    neg = f <= 0.0
    effective = np.minimum(cap, total)
    mandatory = neg | (effective.sum() - effective < total - 1e-8)
    best_y = np.asarray(state[0], dtype=int)
    best_x = np.asarray(state[1], dtype=float)[pos].copy()
    best_val = float(f @ best_y + np.sum(c * best_x))
    rng = random.Random(682173 + 1009 * n + 9176 * m)
    density = np.maximum(f, 0.0) / np.maximum(effective, 1e-30)
    seed_score = density + np.sum(c, axis=0) / total
    last_gain = clock()
    duals = None
    lp_time = budget / (100.0 + p + m)
    min_call = max(1e-4, budget * 0.0005)
    edge_base = max(2 * p, int(1800.0 * math.sqrt(max(budget, 1e-6))
                              * (1.0 + math.log1p(m) / 3.0)))
    initial_x = state[1]

    def save(y, x):
        nonlocal best_y, best_x, best_val, last_gain, duals
        if clock() >= end:
            return False
        y = np.asarray(y, dtype=int).copy()
        y[mandatory] = 1
        if float(effective @ y) < total - 1e-8:
            return False
        x = np.asarray(x, dtype=float).copy()
        if x.shape != (p, m) or not np.all(np.isfinite(x)):
            return False
        np.clip(x, 0.0, 1.0, out=x)
        x[:, y == 0] = 0.0
        sums = x.sum(axis=1)
        if np.any(sums <= 0.0):
            return False
        x /= sums[:, None]
        slack = cap * y - d @ x
        # Preserve customer row sums and transfer over-capacity flow to remaining capacity within the same open set.
        for j in np.flatnonzero(slack < -1e-10):
            donors = np.flatnonzero(x[:, j] > 0.0)
            for i in donors:
                if slack[j] >= -1e-10:
                    break
                if clock() >= end:
                    return False
                targets = np.flatnonzero((slack > 1e-12) & (y > 0))
                targets = targets[np.argsort(c[i, targets], kind='stable')]
                for k in targets:
                    q = min(-float(slack[j]), float(x[i, j] * d[i]), float(slack[k]))
                    if q > 0.0:
                        x[i, j] = max(0.0, x[i, j] - q / d[i])
                        x[i, k] += q / d[i]
                        slack[j] += q
                        slack[k] -= q
                    if slack[j] >= -1e-10 or x[i, j] <= 0.0:
                        break
        # The final check is against the original capacities, not just the solver's scaled constraints.
        if np.any(d @ x - cap * y > 1e-7):
            return False
        if np.max(np.abs(x.sum(axis=1) - 1.0)) > 1e-9:
            return False
        y = ((x.sum(axis=0) > 0.0) | mandatory).astype(int)
        val = float(f @ y + np.sum(c * x))
        if val >= best_val - 1e-9:
            return False
        best_y, best_x, best_val = y, x, val
        last_gain = clock()
        duals = None
        # State is updated only after the complete solution passes the check; exceptions do not corrupt the safeguard solution.
        complete = list(initial_x)
        for a, i in enumerate(pos):
            complete[int(i)] = x[a].tolist()
        state[:] = [y.tolist(), complete]
        return True

    def construct(chosen, balance=0.0):
        ids = np.flatnonzero(chosen & (cap > 0.0))
        if not len(ids) or float(cap[ids].sum()) < total - 1e-8:
            return
        if clock() >= end:
            return
        score = c[:, ids] + balance * d[:, None] * density[ids]
        pref = np.argsort(score, axis=1, kind='stable')
        if len(ids) > 1:
            r = (score[np.arange(p), pref[:, 1]] - score[np.arange(p), pref[:, 0]]) / d
            order = np.argsort(-r, kind='stable')
        else:
            order = np.arange(p)
        x = np.zeros((p, m))
        room = cap.copy()
        for i in order:
            if clock() >= end:
                return
            left = float(d[i])
            last = -1
            for j in ids[pref[i]]:
                q = min(left, float(room[j]))
                if q > 0.0:
                    x[i, j] = q / d[i]
                    left -= q
                    room[j] -= q
                    last = j
                if left <= 0.0:
                    break
            if last < 0 or left > 1e-7:
                return
            x[i, last] += 1.0 - float(x[i].sum())
        save(chosen.astype(int), x)

    construct(np.ones(m, dtype=bool))
    seed_end = min(end, started + budget * 0.075)
    seen = set()
    seed_turn = 0
    while clock() < seed_end:
        balance = 1.0 if seed_turn == 0 else math.exp(rng.uniform(-math.log1p(budget),
                                                                                math.log1p(budget)))
        scores = density + balance * np.sum(c, axis=0) / total
        if seed_turn > 1:
            scores *= np.asarray([rng.uniform(0.55, 1.45) for _ in range(m)])
        chosen = mandatory.copy()
        available = float(effective[chosen].sum())
        for j in np.argsort(scores, kind='stable'):
            if available >= total:
                break
            if not chosen[j] and cap[j] > 0.0:
                chosen[j] = True
                available += effective[j]
        key = chosen.tobytes()
        if key not in seen:
            seen.add(key)
            construct(chosen, 0.0 if seed_turn % 2 == 0 else balance)
        seed_turn += 1
    if not _OPT:
        while clock() < end:
            scores = seed_score * np.asarray([rng.uniform(0.2, 2.0) for _ in range(m)])
            chosen = mandatory.copy()
            available = float(effective[chosen].sum())
            for j in np.argsort(scores):
                if available >= total:
                    break
                if not chosen[j] and cap[j] > 0:
                    chosen[j] = True
                    available += effective[j]
            construct(chosen, rng.random())
        return

    def arcs(rows, plants, edge_limit, extra=None, economic=True):
        nr, nk = len(rows), len(plants)
        if nr * nk <= edge_limit:
            return np.repeat(np.arange(nr), nk), np.tile(np.arange(nk), nr), True
        width = min(nk, max(1, edge_limit // max(2 * nr, 1)))
        sub = c[np.ix_(rows, plants)]
        nearest = np.argpartition(sub, width - 1, axis=1)[:, :width]
        rid = np.repeat(np.arange(nr), width)
        pieces = [rid * nk + nearest.ravel()]
        if economic:
            adjusted = sub + d[rows, None] * density[plants]
            cheap = np.argpartition(adjusted, width - 1, axis=1)[:, :width]
            pieces.append(rid * nk + cheap.ravel())
        ai, aj = np.nonzero(best_x[np.ix_(rows, plants)] > 0.0)
        pieces.append(ai * nk + aj)
        if extra is not None:
            pieces.append(extra)
        packed = np.unique(np.concatenate(pieces))
        return packed // nk, packed % nk, False

    def polish(stop):
        nonlocal duals, lp_time
        if stop - clock() <= max(min_call, lp_time * 0.7):
            return
        chosen = best_y.copy()
        plants = np.flatnonzero((chosen > 0) & (cap > 0.0))
        k = len(plants)
        rows = np.arange(p)
        if not k:
            return
        limit = max(edge_base, int(edge_base * (1.0 + math.log1p(budget) / 4.0)))
        ai, aj, full = arcs(rows, plants, limit, economic=False)
        scale = max(1.0, float(np.max(c[:, plants])))
        capscale = np.maximum(effective[plants], float(d.max()))
        while clock() < stop:
            e = len(ai)
            z = np.arange(e)
            eq = coo_matrix((np.ones(e), (ai, z)), shape=(p, e)).tocsc()
            ub = coo_matrix((d[ai] / capscale[aj], (aj, z)), shape=(k, e)).tocsc()
            remaining = min(end, stop) - clock()
            if remaining <= max(min_call, 0.6 * lp_time):
                return
            began = clock()
            try:
                result = linprog(c[ai, plants[aj]] / scale,
                                 A_ub=ub, b_ub=cap[plants] / capscale,
                                 A_eq=eq, b_eq=np.ones(p), bounds=(0.0, None),
                                 method='highs-ds',
                                 options={'time_limit': remaining * 0.72,
                                          'presolve': False,
                                          'primal_feasibility_tolerance': 1e-8,
                                          'dual_feasibility_tolerance': 1e-8})
            except Exception:
                return
            lp_time = max(min_call, 0.5 * lp_time + 0.5 * (clock() - began))
            if result.x is None or clock() >= end:
                return
            x = np.zeros((p, m))
            x[ai, plants[aj]] = result.x
            save(chosen, x)
            if result.status != 0:
                return
            alpha = np.asarray(result.eqlin.marginals) * scale
            beta = np.zeros(m)
            beta[plants] = np.asarray(result.ineqlin.marginals) * scale / capscale
            if np.array_equal(chosen, best_y):
                duals = (alpha, beta)
            if full or stop - clock() <= max(min_call, 1.5 * lp_time):
                return
            # The fixed-location transportation problem can add arcs by full reduced cost, rather than substituting nearby arcs for pricing.
            reduced = c[:, plants] - alpha[:, None] - d[:, None] * beta[plants]
            width = min(k, max(1, int(math.log1p(budget) + math.sqrt(k))))
            add = np.argpartition(reduced, width - 1, axis=1)[:, :width]
            ar = np.repeat(np.arange(p), width)
            ac = add.ravel()
            good = reduced[ar, ac] < -1e-8 * scale
            if not np.any(good):
                return
            old = ai * k + aj
            packed = np.unique(np.concatenate((old, ar[good] * k + ac[good])))
            if len(packed) == len(old):
                return
            ai, aj = packed // k, packed % k
            full = len(packed) == p * k

    polish(min(end, clock() + 0.11 * budget))

    def guidance():
        if duals is None:
            prices = np.sum(c * best_x, axis=1)
            loads = d @ best_x
            closing = f * (1.0 - 0.5 * loads / np.maximum(cap, 1e-30))
        else:
            prices, beta = duals
            closing = f + effective * beta
        opening = f.copy()
        for j in np.flatnonzero((best_y == 0) & (cap > 0.0)):
            if clock() >= end:
                break
            benefit = prices - c[:, j]
            ids = np.flatnonzero(benefit > 0.0)
            if len(ids):
                ids = ids[np.argsort(-benefit[ids] / d[ids], kind='stable')]
                before = np.cumsum(d[ids]) - d[ids]
                take = np.maximum(0.0, np.minimum(d[ids], cap[j] - before))
                opening[j] -= float(np.sum(benefit[ids] * take / d[ids]))
        return closing, opening

    def joint(rows, free, stop, edge_limit, global_search):
        if stop - clock() <= min_call:
            return False, False
        center = best_y.copy()
        if global_search:
            plants = active
            lo_y = mandatory[plants].astype(float)
            hi_y = np.ones(len(plants))
        else:
            allowed = center.astype(bool)
            allowed[free] = True
            plants = np.flatnonzero(allowed & (cap > 0.0))
            lo_y = center[plants].astype(float)
            hi_y = lo_y.copy()
            release = np.isin(plants, free)
            lo_y[release] = 0.0
            hi_y[release] = 1.0
            lo_y[mandatory[plants]] = 1.0
        k, r = len(plants), len(rows)
        if not k or not r:
            return False, False
        ai, aj, complete = arcs(rows, plants, edge_limit)
        e = len(ai)
        if clock() >= stop:
            return False, False
        xx = best_x[rows]
        frozen_load = np.maximum(0.0, d @ best_x - d[rows] @ xx)
        if r == p:
            frozen_load[:] = 0.0
        capscale = np.maximum(effective[plants], float(d.max()))
        arc = np.arange(e)
        ca, pl = rows[ai], plants[aj]
        max_x = np.minimum(1.0, effective[pl] / d[ca])
        rr = np.concatenate((ai, r + aj, r + np.arange(k),
                             r + k + arc, r + k + arc))
        jj = np.concatenate((k + arc, k + arc, np.arange(k), k + arc, aj))
        vv = np.concatenate((np.ones(e), d[ca] / capscale[aj],
                             -effective[plants] / capscale, np.ones(e), -max_x))
        mat = coo_matrix((vv, (rr, jj)), shape=(r + k + e, k + e)).tocsc()
        low = np.concatenate((np.ones(r), np.full(k + e, -np.inf)))
        high = np.concatenate((np.ones(r), -frozen_load[plants] / capscale, np.zeros(e)))
        scale = max(1.0, float(np.max(np.abs(f[plants]))), float(np.max(c[ca, pl])))
        obj = np.concatenate((f[plants], c[ca, pl])) / scale
        extra, elow, ehigh = [], [], []

        def add(coeff, lower, upper):
            nz = np.flatnonzero(coeff)
            extra.append(coo_matrix((coeff[nz], (np.zeros(len(nz), dtype=int), nz)),
                                    shape=(1, k + e)).tocsc())
            elow.append(lower)
            ehigh.append(upper)

        row = np.zeros(k + e)
        row[:k] = effective[plants] / total
        add(row, 1.0, np.inf)
        row = np.zeros(k + e)
        row[:k] = 1.0
        need = int(np.searchsorted(np.cumsum(np.sort(effective[plants])[::-1]), total - 1e-8)) + 1
        add(row, float(min(need, k)), np.inf)
        # The complement of any capacity-deficient set S must provide the remaining capacity.
        for order in (np.argsort(seed_score[plants]),
                      np.argsort(-(center[plants] * effective[plants]))):
            inside = np.zeros(k, dtype=bool)
            residual = total
            for j in order:
                if 0.0 < effective[plants[j]] < residual - 1e-8:
                    inside[j] = True
                    residual -= effective[plants[j]]
            if np.any(inside) and residual > 1e-8:
                row = np.zeros(k + e)
                row[:k] = np.where(inside, 0.0, np.minimum(effective[plants], residual) / residual)
                add(row, 1.0, np.inf)
        cutoff = (float(f[plants] @ center[plants]) + float(np.sum(c[rows] * xx))) / scale
        add(obj, -np.inf, cutoff + 1e-9 * max(1.0, abs(cutoff)))
        mat = vstack([mat] + extra, format='csc')
        con = LinearConstraint(mat, np.concatenate((low, elow)), np.concatenate((high, ehigh)))
        bounds = Bounds(np.concatenate((lo_y, np.zeros(e))),
                        np.concatenate((hi_y, max_x)))
        integer = np.concatenate((np.ones(k, dtype=np.uint8), np.zeros(e, dtype=np.uint8)))
        remaining = min(end, stop) - clock()
        if remaining <= min_call:
            return False, False
        try:
            # For large models, disable presolve that is not under fine-grained wall-clock control, keeping an independent exit margin.
            presolve = (k + e <= max(64, int(350.0 * math.sqrt(max(budget, 1e-6)))))
            result = milp(obj, integrality=integer, bounds=bounds, constraints=con,
                          options={'time_limit': remaining * 0.74, 'mip_rel_gap': 0.0,
                                   'presolve': presolve, 'disp': False})
        except Exception:
            return False, False
        improved = False
        if result.x is not None and clock() < end:
            y = center.copy()
            y[plants] = np.rint(result.x[:k]).astype(int)
            x = best_x.copy()
            x[rows] = 0.0
            x[ca, pl] = result.x[k:]
            improved = save(y, x)
        optimal = (global_search and complete and result.status == 0
                   and result.x is not None and clock() < end)
        return improved, optimal

    # First try full plant combinations, then spend most of the time on smaller joint neighbourhoods.
    stop = min(end, clock() + 0.49 * max(0.0, end - clock()))
    improved, optimal = joint(np.arange(p), None, stop, edge_base, True)
    if optimal:
        return
    if improved:
        polish(min(end, clock() + min(0.08 * budget, max(2.0 * lp_time, 0.025 * budget))))
    turn = 0
    width_base = max(2, int(math.sqrt(m) * (1.0 + math.log1p(budget) / 3.0)))
    quantum = budget / (3.0 + math.log1p(p + m))
    mutable = np.flatnonzero((~mandatory) & (cap > 0.0))
    while clock() < end:
        remaining = end - clock()
        if remaining <= min_call:
            break
        close, opening = guidance()
        heat = max(0.0, (clock() - last_gain) / max(quantum, min_call))
        width = min(len(mutable), max(2, int(width_base * math.sqrt(1.0 + heat))))
        edge_limit = max(edge_base, int(edge_base * (1.0 + 0.25 * math.sqrt(heat))))
        # Enlarge the release set each time; periodically lift all local fixings.
        global_search = (turn % 4 == 3) or len(mutable) == 0
        rows = np.arange(p)
        free = None
        if not global_search:
            opened = [int(j) for j in mutable if best_y[j]]
            closed = [int(j) for j in mutable if not best_y[j]]
            opened.sort(key=lambda j: -close[j])
            closed.sort(key=lambda j: opening[j])
            # Use one competing facility as an anchor and release the existing facilities that compete with it for the same customers.
            if closed and turn % 2 == 0:
                head = min(len(closed), max(1, int(math.sqrt(width) * (1.0 + heat))))
                anchor = closed[rng.randrange(head)]
                paid = np.sum(c * best_x, axis=1)
                attraction = np.maximum(0.0, paid - c[:, anchor])
                shared = best_x.T @ attraction
                opened.sort(key=lambda j: -(shared[j] + 0.25 * max(0.0, close[j])))
            no = min(len(opened), max(1, width // 2))
            nc = min(len(closed), width - no)
            no = min(len(opened), width - nc)

            def select(ids, count):
                head = min(count, max(1, int(count / (1.4 + 0.1 * heat))))
                return ids[:head] + rng.sample(ids[head:], max(0, count - head))

            free = np.asarray(select(opened, no) + select(closed, nc), dtype=int)
            free_open = free[best_y[free] > 0]
            free_closed = free[best_y[free] == 0]
            affected = np.zeros(p, dtype=bool)
            if len(free_open):
                affected |= np.any(best_x[:, free_open] > 0.0, axis=1)
            target = min(p, max(int(affected.sum()),
                                int(math.sqrt(p) * (1.0 + math.log1p(budget)) * math.sqrt(1.0 + heat))))
            if len(free_closed):
                gain = np.sum(c * best_x, axis=1) - np.min(c[:, free_closed], axis=1)
            else:
                gain = np.sum(c * best_x, axis=1)
            order = np.argsort(-gain, kind='stable')
            candidates = order[~affected[order]]
            amount = max(0, target - int(affected.sum()))
            chosen = candidates[:amount]
            if turn % 2 and len(candidates) > amount and amount > 1:
                half = amount // 2
                chosen = np.concatenate((candidates[:half],
                                         np.asarray(rng.sample(candidates[half:].tolist(), amount - half))))
            affected[chosen] = True
            rows = np.flatnonzero(affected)
        slice_time = min(remaining, quantum * (1.0 + 0.35 * math.sqrt(heat)))
        if global_search:
            slice_time = min(remaining, max(slice_time, 0.55 * remaining))
        stop = min(end, clock() + slice_time)
        improved, optimal = joint(rows, free, stop, edge_limit, global_search)
        if optimal:
            return
        if improved or turn % 3 == 2:
            polish(min(end, clock() + min(0.15 * max(0.0, end - clock()),
                                         max(2.0 * lp_time, quantum / 3.0))))
        turn += 1


def solve(instance: dict, time_limit_s: float) -> dict:
    """Return only the complete open vector and an assignment dict covering all customers and plants."""
    started = time.perf_counter()
    n = m = 0
    state = None
    try:
        n = max(0, int(instance['num_customers']))
        m = max(0, int(instance['num_plants']))
        demand = [float(v) for v in instance['customers_demands']]
        capacity = [max(0.0, float(v)) for v in instance['plants_capacities']]
        fixed = [float(v) for v in instance['plants_fixed_costs']]
        shipping = [[float(v) for v in row] for row in instance['shipping_costs']]
        x = [[0.0] * m for _ in range(n)]
        y = [int(fixed[j] <= 0.0) for j in range(m)]
        room = capacity[:]
        cursor = 0
        for i in range(n):
            if demand[i] <= 0.0:
                if m:
                    x[i][min(range(m), key=lambda j: shipping[i][j])] = 1.0
                continue
            left = demand[i]
            last = -1
            while cursor < m and left > 0.0:
                amount = min(left, room[cursor])
                if amount > 0.0:
                    x[i][cursor] = amount / demand[i]
                    room[cursor] -= amount
                    left -= amount
                    last = cursor
                    y[cursor] = 1
                if room[cursor] <= 0.0:
                    cursor += 1
            if last >= 0:
                x[i][last] += 1.0 - sum(x[i])
        state = [y, x]
        budget = max(0.0, float(time_limit_s))
        if math.isfinite(budget) and time.perf_counter() < started + budget:
            _search(demand, capacity, fixed, shipping, state, started, budget)
    except Exception:
        # If the numerical backend or the optimisation phase fails, still keep the earlier complete feasible assignment.
        pass
    if state is None:
        state = [[1] * m, [[float(j == 0) for j in range(m)] for _ in range(n)]]
    keys = [str(j) for j in range(m)]
    return {'open_plants': [int(v) for v in state[0]],
            'assignments': {str(i): {keys[j]: float(state[1][i][j]) for j in range(m)}
                            for i in range(n)}}
