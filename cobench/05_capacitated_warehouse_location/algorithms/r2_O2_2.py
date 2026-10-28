"""Location algorithm with transportation warm start and dual-cut decomposition
Reuses the transportation linear program, accumulates dual cuts and customer-cost cuts into the facility master problem, and interleaves location neighbourhood search driven by true transportation costs.
"""

import math
import random
import time

try:
    import numpy as np
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix
    _NUMERIC = True
except Exception:
    _NUMERIC = False

try:
    from scipy.optimize._highspy import _core as _hs
except Exception:
    _hs = None


def _native():
    h = _hs._Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("threads", 1)
    h.setOptionValue("parallel", "off")
    h.setOptionValue("random_seed", 731)
    return h


def _interruptible(h, deadline):
    # Native solves accumulate timing; callbacks additionally use this call's wall-clock deadline.
    def callback(kind, message, output, inputs, data):
        if inputs is not None and time.perf_counter() >= deadline[0]:
            inputs.user_interrupt = True
    try:
        h.setCallback(callback, None)
        h.startCallback(_hs.cb.kCallbackSimplexInterrupt)
        h.startCallback(_hs.cb.kCallbackMipInterrupt)
        h.startCallback(_hs.cb.kCallbackIpmInterrupt)
    except Exception:
        pass
    return callback


class _Transport:
    def __init__(self, demand, capacity, cost):
        self.d, self.cap, self.c = demand, capacity, cost
        self.p, self.m = cost.shape
        self.den = np.where(capacity > 0, capacity, 1.0)
        p, m = self.p, self.m
        e = p * m
        cols = np.arange(e)
        customers = np.repeat(np.arange(p), m)
        plants = np.tile(np.arange(m), p)
        self.eq = coo_matrix((np.ones(e), (customers, cols)), shape=(p, e)).tocsr()
        self.ub = coo_matrix((demand[customers] / self.den[plants], (plants, cols)),
                             shape=(m, e)).tocsr()
        self.h = None
        self.limit = [math.inf]
        if _hs is not None:
            try:
                h = _native()
                h.setOptionValue("solver", "simplex")
                h.setOptionValue("simplex_strategy", 1)
                lp = _hs.HighsLp()
                lp.num_col_, lp.num_row_ = e, p + m
                lp.col_cost_ = cost.ravel()
                lp.col_lower_ = np.zeros(e)
                lp.col_upper_ = np.full(e, np.inf)
                lp.row_lower_ = np.r_[np.ones(p), np.full(m, -np.inf)]
                lp.row_upper_ = np.r_[np.ones(p), (capacity > 0).astype(float)]
                mat = lp.a_matrix_
                mat.format_ = _hs.MatrixFormat.kColwise
                mat.num_col_, mat.num_row_ = e, p + m
                mat.start_ = np.arange(0, 2 * e + 1, 2, dtype=np.int32)
                indices = np.empty(2 * e, dtype=np.int32)
                indices[::2], indices[1::2] = customers, p + plants
                values = np.empty(2 * e)
                values[::2], values[1::2] = 1.0, demand[customers] / self.den[plants]
                mat.index_, mat.value_ = indices, values
                h.passModel(lp)
                self.callback = _interruptible(h, self.limit)
                self.h = h
                self.previous = (capacity > 0).astype(float)
            except Exception:
                self.h = None

    def run(self, y, stop):
        left = stop - time.perf_counter()
        if left <= 0:
            return None
        rhs = y * (self.cap > 0)
        if self.h is not None:
            h = self.h
            for j in np.flatnonzero(rhs != self.previous):
                h.changeRowBounds(self.p + int(j), -np.inf, float(rhs[j]))
            self.previous = rhs.copy()
            self.limit[0] = stop
            h.setOptionValue("time_limit", h.getRunTime() + max(0.0, stop - time.perf_counter()))
            h.run()
            s = h.getSolution()
            if not s.value_valid:
                return None
            x = np.asarray(s.col_value).reshape(self.p, self.m)
            beta = np.asarray(s.row_dual[self.p:]) if s.dual_valid else None
            exact = h.getModelStatus() == _hs.HighsModelStatus.kOptimal
        else:
            r = linprog(self.c.ravel(), A_ub=self.ub, b_ub=rhs,
                        A_eq=self.eq, b_eq=np.ones(self.p), bounds=(0, None),
                        method="highs-ds", options={"time_limit": left})
            if r.x is None:
                return None
            x = r.x.reshape(self.p, self.m)
            beta = np.asarray(r.ineqlin.marginals) if r.status == 0 else None
            exact = r.status == 0
        cut = None
        if beta is not None and np.all(np.isfinite(beta)):
            beta = np.minimum(beta, 0.0)
            # Reconstruct a feasible dual so that numerical errors do not make the master problem cut off feasible solutions.
            alpha = np.min(self.c - self.d[:, None] * (beta / self.den), axis=1)
            b = beta * (self.cap > 0)
            a = float(alpha.sum())
            a -= 1e-9 * (1.0 + abs(a) + float(np.abs(b).sum()))
            cut = a, b
        return x, cut, exact


class _Master:
    def __init__(self, cap, fixed, cost, forced, total):
        self.m, self.p = len(cap), len(cost)
        self.c, self.f = cost, fixed
        self.limit = [math.inf]
        self.h = _native()
        h = self.h
        h.setOptionValue("mip_rel_gap", 1e-7)
        h.setOptionValue("mip_abs_gap", 1e-8)
        lp = _hs.HighsLp()
        q = self.m + self.p
        lp.num_col_, lp.num_row_ = q, 1
        lp.col_cost_ = np.r_[fixed, np.ones(self.p)]
        lp.col_lower_ = np.r_[forced.astype(float), np.zeros(self.p)]
        upper = np.where((cap > 0) | forced, 1.0, 0.0)
        lp.col_upper_ = np.r_[upper, np.max(cost, axis=1)]
        lp.integrality_ = ([_hs.HighsVarType.kInteger] * self.m +
                           [_hs.HighsVarType.kContinuous] * self.p)
        lp.row_lower_, lp.row_upper_ = np.array([1.0]), np.array([np.inf])
        mat = lp.a_matrix_
        mat.format_ = _hs.MatrixFormat.kRowwise
        mat.num_col_, mat.num_row_ = q, 1
        mat.start_ = np.array([0, self.m], dtype=np.int32)
        mat.index_ = np.arange(self.m, dtype=np.int32)
        mat.value_ = cap / total
        h.passModel(lp)
        self.callback = _interruptible(h, self.limit)
        self.levels = [set() for _ in range(self.p)]
        self.columns = np.arange(q, dtype=np.int32)

    def add(self, y, cut, stop):
        if cut is not None:
            a, b = cut
            vals = np.r_[-b, np.ones(self.p)]
            self.h.addRow(float(a), np.inf, len(vals), self.columns, vals)
        # Customer-cost lower envelope: z_i >= r - sum_j max(r-c_ij, 0) y_j.
        opened = np.flatnonzero(y > 0.5)
        if not len(opened):
            return
        nearest = np.min(self.c[:, opened], axis=1)
        for i, r in enumerate(nearest):
            if time.perf_counter() >= stop:
                break
            r = float(r)
            if r <= 1e-12 or r in self.levels[i]:
                continue
            self.levels[i].add(r)
            idx = np.flatnonzero(self.c[i] < r)
            columns = np.r_[idx, self.m + i].astype(np.int32)
            values = np.r_[r - self.c[i, idx], 1.0]
            self.h.addRow(r - 1e-10 * (1 + abs(r)), np.inf, len(columns), columns, values)

    def run(self, y, x, stop):
        if time.perf_counter() >= stop:
            return None, -math.inf
        h = self.h
        candidate = np.r_[y, np.sum(self.c * x, axis=1)]
        h.setSolution(len(candidate), self.columns, candidate)
        self.limit[0] = stop
        h.setOptionValue("time_limit", h.getRunTime() + max(0.0, stop - time.perf_counter()))
        h.run()
        info, sol = h.getInfo(), h.getSolution()
        lower = float(info.mip_dual_bound)
        if not sol.value_valid:
            return None, lower
        return (np.asarray(sol.col_value[:self.m]) > 0.5).astype(float), lower


def _search(d, cap, fixed, cost, state, start, budget):
    clock = time.perf_counter
    n, m = len(d), len(cap)
    reserve = min(0.3 * budget, max(0.07 * budget, n * m * 7e-7))
    end = start + budget - reserve
    if not _NUMERIC or clock() >= end or not n or not m:
        return
    positive = np.flatnonzero(np.asarray(d) > 0)
    if not len(positive):
        return
    dd = np.asarray(d, dtype=float)[positive]
    total = float(dd.sum())
    caps = np.asarray(cap, dtype=float)
    if caps.sum() < total - 1e-9:
        return
    caps = np.minimum(caps, total)
    original = np.asarray(cost, dtype=float)
    ff0 = np.asarray(fixed, dtype=float)
    raw = original[positive]
    scale = max(1.0, float(np.mean(np.abs(ff0))), float(np.mean(np.abs(raw))))
    minima = np.min(raw, axis=1)
    c = (raw - minima[:, None]) / scale
    f = ff0 / scale
    zero = np.flatnonzero(np.asarray(d) <= 0)
    zero_j = np.argmin(original[zero], axis=1) if len(zero) else np.empty(0, dtype=int)
    constant = float(minima.sum()) + float(original[zero, zero_j].sum())
    forced = (ff0 <= 0) | (caps.sum() - caps < total - 1e-9)
    active = np.flatnonzero(caps > 0)
    state[1] = np.asarray(state[1], dtype=int)
    state[2] = np.asarray(state[2], dtype=float)
    rng = random.Random(37019)
    p = len(dd)
    best_y = state[1].astype(float)
    best_x = state[2][positive].copy()
    best = (state[0] - constant) / scale

    def keep(y, x):
        nonlocal best, best_y, best_x
        if x.shape != (p, m) or not np.all(np.isfinite(x)):
            return False
        y = np.maximum(y > 0.5, forced)
        if float(caps @ y) < total - 1e-9:
            return False
        q = np.maximum(x, 0.0) * dd[:, None]
        q[:, ~y] = 0.0
        row = q.sum(axis=1)
        q *= np.minimum(1.0, dd / np.maximum(row, 1e-300))[:, None]
        load = q.sum(axis=0)
        q *= np.minimum(1.0, caps * y / np.maximum(load, 1e-300))[None, :]
        remaining = np.maximum(0.0, caps * y - q.sum(axis=0))
        missing = np.maximum(0.0, dd - q.sum(axis=1))
        ids = np.flatnonzero(y & (caps > 0))
        for i in np.flatnonzero(missing > 0):
            left = float(missing[i])
            for j in ids[np.argsort(c[i, ids])]:
                take = min(left, float(remaining[j]))
                if take > 0:
                    q[i, j] += take
                    remaining[j] -= take
                    left -= take
                if left <= 0:
                    break
            if left > 1e-8 * max(1.0, dd[i]):
                return False
        clean = q / dd[:, None]
        for i in range(p):
            j = int(np.argmax(clean[i]))
            clean[i, j] += 1.0 - float(clean[i].sum())
        used = np.any(clean > 0, axis=0) | (f <= 0)
        value = float(f @ used + np.sum(c * clean))
        if value >= best - 1e-10 * max(1.0, abs(best)):
            return False
        complete = np.zeros((n, m))
        complete[positive] = clean
        complete[zero, zero_j] = 1.0
        best, best_y, best_x = value, used.astype(float), clean
        state[:] = [constant + scale * value, used.astype(int), complete]
        return True

    # Build a feasible solution prioritising high-regret customers, using marginal freight costs and amortised capacity costs of unopened facilities.
    available = caps.copy()
    y = forced.copy()
    x = np.zeros((p, m))
    adjusted = c[:, active] + dd[:, None] * np.maximum(f[active], 0) / caps[active]
    if len(active) > 1:
        pair = np.partition(adjusted, 1, axis=1)[:, :2]
        regret = np.max(pair, axis=1) - np.min(pair, axis=1)
    else:
        regret = dd
    for i in np.argsort(-regret):
        left = float(dd[i])
        while left > 0:
            ids = np.flatnonzero(available > 0)
            if not len(ids):
                break
            charge = c[i, ids] / dd[i] + (~y[ids]) * np.maximum(f[ids], 0) / caps[ids]
            j = int(ids[np.argmin(charge)])
            take = min(left, float(available[j]))
            x[i, j] += take / dd[i]
            available[j] -= take
            left -= take
            y[j] = True
    keep(y, x)
    if clock() >= end:
        return
    transport = _Transport(dd, caps, c)
    master = None
    if _hs is not None and clock() < end:
        try:
            master = _Master(caps, f, c, forced, total)
        except Exception:
            pass
    evaluated = set()
    dual_a, dual_b = [], []
    average_lp = max(budget * 0.001, 1e-4)
    stalled = 0

    def key(y):
        return np.packbits(y > 0.5).tobytes()

    def repair(y):
        y = np.maximum(y > 0.5, forced).astype(float)
        short = total - float(caps @ y)
        if short > 0:
            ids = np.flatnonzero((y == 0) & (caps > 0))
            rank = f[ids] / caps[ids]
            for j in ids[np.argsort(rank)]:
                y[j] = 1.0
                short -= caps[j]
                if short <= 0:
                    break
        return y

    def evaluate(y, stop):
        nonlocal average_lp, master
        y = repair(y)
        k = key(y)
        if k in evaluated or clock() >= stop:
            return False
        began = clock()
        try:
            result = transport.run(y, stop)
        except Exception:
            transport.h = None
            result = None
        elapsed = clock() - began
        average_lp = max(budget * 0.0001, 0.7 * average_lp + 0.3 * elapsed)
        if result is None:
            return False
        x, cut, exact = result
        improved = keep(y, x)
        if exact:
            evaluated.add(k)
            evaluated.add(key(best_y)) if improved else None
        if cut is not None:
            dual_a.append(cut[0])
            dual_b.append(cut[1])
        if master is not None and clock() < stop:
            try:
                master.add(y, cut, stop)
            except Exception:
                master = None
        return improved

    evaluate(best_y.copy(), min(end, clock() + 0.15 * budget))
    if clock() < end:
        evaluate(np.ones(m), min(end, clock() + max(4 * average_lp, 0.09 * budget)))

    mutable = np.flatnonzero(~forced & (caps > 0))
    span = max(1, min(m, int(math.sqrt(m) * math.sqrt(math.log1p(max(0.0, budget))))))
    queue = []
    center_key = None
    local_until = min(end, start + 0.28 * budget)
    global_due = local_until

    def neighbors():
        base = best_y.copy()
        opened = np.flatnonzero((base > 0.5) & ~forced)
        closed = np.flatnonzero((base < 0.5) & (caps > 0))
        spare = float(caps @ base) - total
        paid = np.sum(c * best_x, axis=1)
        improvement = np.maximum(paid[:, None] - c, 0.0)
        gain = improvement.sum(axis=0)
        mass = ((improvement > 0) * dd[:, None]).sum(axis=0)
        opening = f - gain * np.minimum(1.0, caps / np.maximum(mass, 1e-300))
        present = np.flatnonzero(base > 0.5)
        nearest = np.min(c[:, present], axis=1)
        if len(present) > 1:
            second = np.partition(c[:, present], 1, axis=1)[:, 1]
        else:
            second = np.full(p, np.max(c) + np.max(np.abs(f)) + 1.0)
        closing = np.full(m, np.inf)
        for j in opened:
            alt = np.where(c[:, j] <= nearest + 1e-12, second, nearest)
            closing[j] = float(np.sum(best_x[:, j] * (alt - c[:, j])) - f[j])
        a = np.asarray(dual_a)
        b = np.asarray(dual_b)
        lower_base = a + b @ base if len(a) else None
        fixed_base = float(f @ base)
        proposals = []
        seen = set()

        def put(remove, add, estimate):
            trial = base.copy()
            if remove >= 0:
                trial[remove] = 0
            if add >= 0:
                trial[add] = 1
            k = key(trial)
            if k in evaluated or k in seen:
                return
            seen.add(k)
            delta = (f[add] if add >= 0 else 0.0) - (f[remove] if remove >= 0 else 0.0)
            if lower_base is not None:
                bounds = lower_base.copy()
                if remove >= 0:
                    bounds -= b[:, remove]
                if add >= 0:
                    bounds += b[:, add]
                if fixed_base + delta + float(np.max(bounds)) >= best - 1e-9:
                    return
            noise = rng.random() * (1.0 + abs(estimate)) / (2 + math.log1p(budget))
            proposals.append((estimate + noise, trial))

        for j in opened:
            if caps[j] <= spare + 1e-9:
                put(int(j), -1, float(closing[j]))
        for j in closed:
            put(-1, int(j), float(opening[j]))
        removals = opened[np.argsort(closing[opened])[:span]]
        additions = closed[np.argsort(opening[closed])[:span]]
        for j in removals:
            for k in additions:
                if clock() >= end:
                    break
                if caps[j] - caps[k] <= spare + 1e-9:
                    put(int(j), int(k), float(closing[j] + opening[k]))
        proposals.sort(key=lambda item: item[0], reverse=True)
        return proposals

    while clock() < end:
        remaining = end - clock()
        if remaining <= max(budget * 0.001, 1.2 * average_lp):
            break
        # The master problem contains only location variables and customer-cost variables; no full joint integer transportation model is built.
        if master is not None and clock() >= global_due:
            chunk = min(remaining * 0.38,
                        budget * (0.035 + 0.025 * math.log1p(1 + stalled)))
            try:
                candidate, lower = master.run(best_y, best_x, min(end, clock() + chunk))
                if math.isfinite(lower) and lower >= best - 1e-7 * max(1.0, abs(best)):
                    break
                if candidate is not None and clock() < end:
                    improved = evaluate(candidate, min(end, clock() +
                                         max(4 * average_lp, 0.065 * budget)))
                    stalled = 0 if improved else stalled + 1
            except Exception:
                master = None
            global_due = min(end, clock() + max(5 * average_lp, budget * 0.025))
        if clock() >= end:
            break
        if center_key != key(best_y) or not queue:
            center_key = key(best_y)
            queue = neighbors()
        if queue:
            _, trial = queue.pop()
        elif len(mutable):
            # On stagnation enlarge multi-facility perturbations; the magnitude is determined jointly by size, time limit and degree of stagnation.
            trial = best_y.copy()
            depth = min(len(mutable), max(1, int(math.sqrt(math.log1p(budget)) *
                        (1.0 + math.log1p(stalled)))))
            count = rng.randint(1, depth)
            for j in rng.sample(list(mutable), count):
                trial[j] = 1.0 - trial[j]
            trial = repair(trial)
            stalled += 1
            if key(trial) in evaluated:
                global_due = min(global_due, clock())
                continue
        else:
            break
        improved = evaluate(trial, min(end, clock() + max(4 * average_lp, 0.06 * budget)))
        stalled = 0 if improved else stalled + 1


def solve(instance: dict, time_limit_s: float) -> dict:
    """First store a capacity-feasible assignment, then monotonically update the best solution within the wall-clock budget."""
    start = time.perf_counter()
    n = m = 0
    state = None
    try:
        n = int(instance["num_customers"])
        m = int(instance["num_plants"])
        d = [float(v) for v in instance["customers_demands"]]
        cap = [float(v) for v in instance["plants_capacities"]]
        fixed = [float(v) for v in instance["plants_fixed_costs"]]
        cost = [[float(v) for v in row] for row in instance["shipping_costs"]]
        x = [[0.0] * m for _ in range(n)]
        y = [int(v <= 0) for v in fixed]
        remaining = cap.copy()
        j = 0
        for i in range(n):
            if d[i] <= 0:
                if m:
                    x[i][min(range(m), key=lambda k: cost[i][k])] = 1.0
                continue
            left = d[i]
            last = -1
            while left > 0 and j < m:
                take = min(left, max(0.0, remaining[j]))
                if take > 0:
                    x[i][j] = take / d[i]
                    remaining[j] -= take
                    left -= take
                    last = j
                    y[j] = 1
                if remaining[j] <= 0:
                    j += 1
            if last >= 0:
                x[i][last] += 1.0 - sum(x[i])
        value = sum(fixed[j] * y[j] for j in range(m))
        value += sum(cost[i][j] * x[i][j] for i in range(n) for j in range(m))
        state = [value, y, x]
        budget = max(0.0, float(time_limit_s))
        if math.isfinite(budget) and budget > 0:
            _search(d, cap, fixed, cost, state, start, budget)
    except Exception:
        pass
    if state is None:
        return {"open_plants": [1] * m,
                "assignments": {str(i): {str(j): float(j == 0) for j in range(m)}
                                for i in range(n)}}
    keys = [str(j) for j in range(m)]
    return {"open_plants": [int(v) for v in state[1]],
            "assignments": {str(i): {keys[j]: float(state[2][i][j]) for j in range(m)}
                            for i in range(n)}}
