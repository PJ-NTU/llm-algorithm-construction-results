"""Cardinality-layered reduced-cost cooperative neighbourhood search
Fuses layered linear relaxations, reduced-cost constraints, persistent warm-started integer programming and adaptive tabu search, sharing a pool of feasible solutions and allocating search time by wall clock.
"""
import math
import time

import numpy as np

try:
    from scipy.optimize import linprog, milp, LinearConstraint, Bounds
except Exception:
    linprog = milp = None
try:
    from scipy.optimize._highspy import _core as _HC
    _HC._Highs
    _HC.cb.HighsCallbackType
except Exception:
    _HC = None


_clock = time.perf_counter


class _State:
    def __init__(self, c, G, h, T):
        self.n = len(c)
        # Large integers are converted to floats only inside heuristics and the solver; feasibility and value are always verified exactly.
        mx = max([abs(v) for v in c] + [abs(v) for v in h] +
                 [max((abs(v) for v in row), default=0) for row in G], default=0)
        dtype = object if mx * (self.n + 1) >= 2 ** 61 else np.int64
        self.c = np.asarray(c, dtype=dtype)
        self.G = np.asarray(G, dtype=dtype).reshape(len(h), self.n)
        self.h = np.asarray(h, dtype=dtype)
        self.cf = np.asarray(c, dtype=float)
        raw = np.asarray(G, dtype=float).reshape(len(h), self.n)
        rhs = np.asarray(h, dtype=float)
        scales = np.maximum(1.0, np.max(np.abs(raw), axis=1)) if len(h) else np.ones(0)
        self.Gf = raw / scales[:, None]
        self.hf = rhs / scales
        self.R = len(h)
        self.best_x = None
        self.best_obj = None
        self.elite = []
        self.kbest = {}
        self.cap = min(self.n + 1, max(2, int(2 + math.sqrt(self.n) * math.log1p(T) / 4)))
        self.last_improve = _clock()
        self.rng = np.random.default_rng(20261001)
        self.T = T

    def exact(self, x):
        if np.any(self.G @ x > self.h):
            return None
        return int(self.c @ x)

    def offer(self, x):
        try:
            x = np.asarray(x).reshape(-1)
            if x.size != self.n or not np.all(np.isfinite(x.astype(float))):
                return False
            x = (x > 0.5).astype(np.int64)
            obj = self.exact(x)
            if obj is None:
                return False
            k = int(x.sum())
            if k not in self.kbest or obj > self.kbest[k][0]:
                self.kbest[k] = (obj, x.copy())
            key = x.astype(np.int8).tobytes()
            if (len(self.elite) < self.cap or obj > self.elite[-1][0]) and not any(
                    e[1] == key for e in self.elite):
                self.elite.append((obj, key, x.copy()))
                self.elite.sort(key=lambda e: -e[0])
                self.elite = self.elite[:self.cap]
            if self.best_obj is None or obj > self.best_obj:
                self.best_obj = obj
                self.best_x = x.copy()
                self.last_improve = _clock()
                return True
        except Exception:
            pass
        return False


class _Arm:
    def __init__(self, k, ub, xl, rc, tol, st):
        self.k, self.ub, self.xl, self.rc, self.tol = k, ub, xl, rc, tol
        self.arc = np.abs(rc)
        self.base = (xl > 0.5).astype(np.int64)
        self.base[rc > tol / max(1, st.n)] = 1
        self.base[rc < -tol / max(1, st.n)] = 0
        self.frac = (xl > 1e-6) & (xl < 1 - 1e-6)
        self.base[self.frac] = (xl[self.frac] > 0.5).astype(np.int64)
        self.scale = max(1e-12, float(np.mean(self.arc[self.arc > 1e-9]))) if np.any(self.arc > 1e-9) else 1.0
        effort = math.log1p(st.T)
        self.minimum = min(st.n, max(2, int(self.frac.sum()) + 2))
        self.core = float(min(st.n, max(self.minimum, 1.5 * (st.R + 1),
                                               math.sqrt(st.n) * (1 + effort / 3))))
        self.size = float(min(st.n, max(self.core, st.n * (0.07 + 0.045 * effort))))
        self.radius = float(min(st.n, max(2, effort * math.sqrt(st.n) / 3)))
        self.spent = 0.0
        self.reward = 1.0
        self.closed = False
        self.weights = np.ones(7)
        self.slice = st.T / (18 + 2 * effort)

    def closed_by_bound(self, st):
        return st.best_obj is not None and self.ub + self.tol < st.best_obj + 1


def _lp(st, k, deadline):
    if linprog is None or _clock() >= deadline:
        return None
    kw = {}
    if k is not None:
        kw['A_eq'] = np.ones((1, st.n))
        kw['b_eq'] = [k]
    try:
        r = linprog(-st.cf, A_ub=st.Gf if st.R else None,
                    b_ub=st.hf if st.R else None, bounds=(0.0, 1.0),
                    method='highs', options={'time_limit': max(1e-6, deadline - _clock())}, **kw)
        if r.status == 2:
            return False
        if r.status != 0 or r.x is None:
            return None
        xl = np.clip(np.asarray(r.x), 0, 1)
        # Non-negative multipliers give a Lagrangian upper bound over the box constraints, independent of the rounded LP objective.
        y = np.maximum(0, -np.asarray(r.ineqlin.marginals)) if st.R else np.zeros(0)
        alpha = -float(r.eqlin.marginals[0]) if k is not None else 0.0
        rc = st.cf - st.Gf.T @ y - alpha
        dual = float(st.hf @ y) + (alpha * k if k is not None else 0.0)
        ub = dual + float(np.maximum(rc, 0).sum())
        tol = 1e-6 + 2e-9 * (abs(dual) + float(np.abs(rc).sum()) + abs(ub))
        if not math.isfinite(ub) or not np.all(np.isfinite(rc)):
            return None
        arm = _Arm(k, ub, xl, rc, tol, st)
        st.offer(xl)
        st.offer(xl > 1e-7)
        st.offer(xl > 1 - 1e-7)
        return arm
    except Exception:
        return None


class _Engine:
    def __init__(self, st):
        self.st = st
        self.h = None
        self.end = 0.0
        self.stall = None
        self.last_imp = 0.0
        self.run_start = 0.0
        self.min_end = 0.0
        self.run_best = -math.inf
        self.idx = np.arange(st.n, dtype=np.int32)
        if _HC is not None:
            try:
                h = _HC._Highs()
                h.setOptionValue('output_flag', False)
                h.setOptionValue('mip_rel_gap', 0.0)
                h.setOptionValue('mip_abs_gap', 0.0)
                inf = h.getInfinity()
                mask = st.Gf != 0
                start = np.r_[0, np.cumsum(mask.sum(axis=1))].astype(np.int32)
                cols = np.nonzero(mask)[1].astype(np.int32)
                vals = st.Gf[mask].astype(float)
                ret = h.passModel(st.n, st.R, int(mask.sum()), 2, -1, 0.0, st.cf,
                                  np.zeros(st.n), np.ones(st.n), np.full(st.R, -inf),
                                  st.hf, start, cols, vals, np.ones(st.n, dtype=np.int32))
                if 'Error' in str(ret):
                    raise ValueError('model initialisation failed')
                self.inf = inf
                self.ct_imp = int(_HC.cb.HighsCallbackType.kCallbackMipImprovingSolution)
                self.ct_int = int(_HC.cb.HighsCallbackType.kCallbackMipInterrupt)
                h.setCallback(self._callback, None)
                h.startCallback(_HC.cb.HighsCallbackType.kCallbackMipImprovingSolution)
                h.startCallback(_HC.cb.HighsCallbackType.kCallbackMipInterrupt)
                self.h = h
            except Exception:
                self.h = None

    def _callback(self, ct, msg, out, inp, data):
        try:
            now = _clock()
            if int(ct) == self.ct_int:
                stop = now >= self.end
                if self.stall is not None and now >= self.min_end:
                    stop = stop or now - self.last_imp > max(self.stall, self.last_imp - self.run_start)
                inp.user_interrupt = bool(stop)
            elif int(ct) == self.ct_imp:
                val = float(out.objective_function_value)
                if val > self.run_best + 1e-7:
                    self.run_best = val
                    self.last_imp = now
        except Exception:
            pass

    def run(self, lb, ub, rows, warm, deadline, stall=None, min_time=0.0):
        if _clock() >= deadline or np.any(lb > ub):
            return 1, None
        if self.h is None:
            return self._scipy(lb, ub, rows, deadline)
        h, st = self.h, self.st
        self.end = deadline
        self.stall = stall
        self.run_start = self.last_imp = _clock()
        self.min_end = self.run_start + min_time
        self.run_best = -math.inf
        added = 0
        code, bound = -1, None
        try:
            h.changeColsBounds(st.n, self.idx, lb.astype(float), ub.astype(float))
            for coef, low, high in rows:
                nz = np.flatnonzero(coef).astype(np.int32)
                h.addRow(float(low) if math.isfinite(low) else -self.inf,
                         float(high) if math.isfinite(high) else self.inf,
                         nz.size, nz, np.asarray(coef[nz], dtype=float))
                added += 1
            valid_warm = warm is not None and np.all(warm >= lb) and np.all(warm <= ub)
            if valid_warm:
                for coef, low, high in rows:
                    v = float(coef @ warm)
                    if v < low - 1e-7 or v > high + 1e-7:
                        valid_warm = False
                        break
            if valid_warm:
                try:
                    h.setSolution(st.n, self.idx, warm.astype(float))
                except Exception:
                    sol = _HC.HighsSolution()
                    sol.col_value = warm.astype(float).tolist()
                    h.setSolution(sol)
            else:
                fixed = np.flatnonzero(lb == ub).astype(np.int32)
                if fixed.size:
                    h.setSolution(fixed.size, fixed, lb[fixed].astype(float))
            runtime = float(h.getRunTime())
            h.setOptionValue('time_limit', runtime + max(1e-6, deadline - _clock()))
            if _clock() < deadline:
                h.run()
                status = h.getModelStatus()
                sol = h.getSolution()
                if sol.value_valid:
                    st.offer(np.asarray(sol.col_value, dtype=float)[:st.n])
                v = float(h.getInfo().mip_dual_bound)
                if math.isfinite(v):
                    bound = v
                if status == _HC.HighsModelStatus.kOptimal:
                    code = 0
                elif status == _HC.HighsModelStatus.kInfeasible:
                    code = 2
                else:
                    code = 1
        except Exception:
            code = -1
        finally:
            if added:
                try:
                    ret = h.deleteRows(added, np.arange(st.R, st.R + added, dtype=np.int32))
                    if 'Error' in str(ret):
                        self.h = None
                except Exception:
                    self.h = None
        if code == -1:
            self.h = None
            return self._scipy(lb, ub, rows, deadline)
        return code, bound

    def _scipy(self, lb, ub, rows, deadline):
        if milp is None or _clock() >= deadline:
            return -1, None
        st = self.st
        try:
            cons = [LinearConstraint(st.Gf, -np.inf, st.hf)] if st.R else []
            for coef, low, high in rows:
                cons.append(LinearConstraint(coef.reshape(1, -1), low, high))
            r = milp(-st.cf, integrality=np.ones(st.n), bounds=Bounds(lb, ub),
                     constraints=cons or None,
                     options={'time_limit': max(1e-6, deadline - _clock()),
                              'disp': False, 'mip_rel_gap': 0.0})
            if r.x is not None:
                st.offer(r.x)
            bound = None
            if getattr(r, 'mip_dual_bound', None) is not None:
                v = -float(r.mip_dual_bound)
                if math.isfinite(v):
                    bound = v
            return int(r.status), bound
        except Exception:
            return -1, None


def _tabu(st, start, arm, deadline):
    if _clock() >= deadline or not st.R:
        return
    n, G, h, rng = st.n, st.Gf, st.hf, st.rng
    x = np.asarray(start, dtype=np.int64).copy()
    load = G @ x
    cscale = max(1.0, float(np.mean(np.abs(st.cf))))
    c = st.cf / cscale
    row_scale = np.maximum(np.mean(np.abs(G), axis=1), 1e-9)
    G = G / row_scale[:, None]
    h = h / row_scale
    load = load / row_scale
    weight = np.ones(st.R)
    lam = 1.0
    tenure = max(2, int(math.sqrt(n) * (0.3 + 0.1 * math.log1p(st.T))))
    tabu = np.zeros(n, dtype=np.int64)
    it = 0
    begun = last_gain = _clock()
    window = max(1e-6, (deadline - begun) / (2 + math.log1p(st.T)))
    # Bound the matrix size of each step; subsets keep refreshing and the total number of search steps is not limited.
    batch = min(n, max(8, int(math.sqrt(n) * (2 + math.log1p(st.T)))))
    if n * st.R <= 4096 * (1 + math.log1p(st.T)):
        batch = n
    probability = np.ones(n)
    if arm is not None:
        probability += np.exp(-np.minimum(arm.arc / arm.scale, 30)) + arm.frac
    probability /= probability.sum()
    while _clock() < deadline:
        it += 1
        cand = np.arange(n) if batch == n else rng.choice(n, batch, replace=False, p=probability)
        d = 1 - 2 * x[cand]
        P = float(weight @ np.maximum(load - h, 0))
        V = np.maximum(load[:, None] + G[:, cand] * d - h[:, None], 0)
        penalty = weight @ V
        gains = c[cand] * d
        score = gains - lam * (penalty - P)
        feasible = np.all(V <= 1e-9, axis=0)
        aspiration = feasible & (gains > 0)
        allowed = (tabu[cand] <= it) | aspiration
        if not np.any(allowed):
            allowed[:] = True
        score = np.where(allowed, score + rng.random(cand.size) * 1e-10, -np.inf)
        pos = int(np.argmax(score))
        changes = [(int(cand[pos]), int(d[pos]))]
        # Combine add/drop/swap moves among single-point candidates, temporarily violating hard constraints but never accepting infeasible solutions.
        selected = np.flatnonzero((x[cand] == 1) & allowed)
        unselected = np.flatnonzero((x[cand] == 0) & allowed)
        width = max(2, int(math.log1p(st.T) + n ** 0.125))
        if selected.size and unselected.size and _clock() < deadline:
            a = unselected[np.argsort(-score[unselected])[:width]]
            b = selected[np.argsort(-score[selected])[:width]]
            ai, bi = cand[a], cand[b]
            v = np.maximum(load[:, None, None] + G[:, ai, None] - G[:, None, bi] - h[:, None, None], 0)
            pairs = c[ai, None] - c[None, bi] - lam * (np.tensordot(weight, v, axes=(0, 0)) - P)
            p = int(np.argmax(pairs))
            ii, jj = divmod(p, bi.size)
            if pairs[ii, jj] > score[pos] + 1e-10:
                changes = [(int(ai[ii]), 1), (int(bi[jj]), -1)]
        for j, change in changes:
            x[j] += change
            load += change * G[:, j]
            tabu[j] = it + int(rng.integers(1, tenure + 1))
        violation = load > h + 1e-8
        if not np.any(violation):
            if st.offer(x):
                last_gain = _clock()
            lam = max(1e-5, lam / 1.025)
        else:
            weight[violation] += 0.03
            weight /= weight.mean()
            lam = min(1e6, lam * 1.025)
        now = _clock()
        if now - last_gain > window:
            last_gain = now
            if st.elite:
                x = st.elite[int(rng.integers(len(st.elite)))][2].copy()
            elif arm is not None:
                x = (rng.random(n) < arm.xl).astype(np.int64)
            strength = min(n, max(1, int(math.sqrt(n) * (1 + math.log1p(st.T)) *
                                                   (now - begun) / max(1e-6, deadline - begun))))
            flip = rng.choice(n, size=strength, replace=False, p=probability)
            x[flip] = 1 - x[flip]
            load = G @ x
            tabu[:] = 0
            weight[:] = 1
            lam = 1.0


def _constraints(st, full, arm):
    lb, ub = np.zeros(st.n), np.ones(st.n)
    rows = []
    if arm.k is not None:
        rows.append((np.ones(st.n), float(arm.k), float(arm.k)))
    if st.best_obj is not None:
        for a in ([full] if arm is full else [full, arm]):
            gap = a.ub - float(st.best_obj) + a.tol
            forced = a.arc > gap + a.tol
            lb[forced & (a.rc > 0)] = 1
            ub[forced & (a.rc < 0)] = 0
            # Keep the current objective value so that the incumbent of the same layer remains a valid warm start.
            if np.any(a.arc > 0):
                rhs = a.ub - float(st.best_obj) - float(np.maximum(a.rc, 0).sum()) + a.tol
                rows.append((-a.rc, -np.inf, rhs))
    return lb, ub, rows


def _neighborhood(st, full, arm, strategy):
    lb, ub, rows = _constraints(st, full, arm)
    available = np.flatnonzero(lb < ub)
    n, rng = st.n, st.rng
    if arm.k in st.kbest:
        warm = st.kbest[arm.k][1]
    else:
        warm = st.best_x
    noise = rng.random(n)
    rcs = np.minimum(arm.arc / arm.scale, 30)
    base = warm.copy() if warm is not None else arm.base.copy()
    if strategy == 0:
        base = arm.base.copy()
        size = min(available.size, max(arm.minimum, int(arm.core)))
        priority = -rcs + 0.35 * noise + arm.frac
        if warm is not None:
            priority += 0.6 * (warm != base)
    else:
        size = min(available.size, max(arm.minimum, int(arm.size)))
        disagreement = np.zeros(n)
        if len(st.elite) > 1:
            mean = np.mean([e[2] for e in st.elite], axis=0)
            disagreement = np.abs(mean - base)
        if strategy == 1:
            priority = -rcs + 1.5 * disagreement + abs(arm.xl - base) + noise
        elif strategy == 2:
            priority = abs(arm.xl - base) + 0.5 * disagreement + 0.45 * noise
        elif strategy == 3:
            priority = disagreement + 0.35 * noise
        elif strategy == 4 and st.R:
            slack = (st.hf - st.Gf @ base) / (np.abs(st.Gf).sum(axis=1) + 1e-9)
            tight = np.argsort(slack)[:max(1, int(math.sqrt(st.R)))]
            row = np.abs(st.Gf[int(rng.choice(tight))])
            priority = row / (row.max() + 1e-9) + noise
        elif strategy == 6:
            size = available.size
            coef = 1.0 - 2.0 * base
            rows.append((coef, -np.inf, float(int(arm.radius) - int(base.sum()))))
            priority = noise
        else:
            priority = noise
    base = np.maximum(lb, np.minimum(ub, base)).astype(np.int64)
    if size >= available.size:
        free = available
    elif size:
        free = available[np.argpartition(-priority[available], size - 1)[:size]]
    else:
        free = available
    free_mask = np.zeros(n, dtype=bool)
    free_mask[free] = True
    # A cross-layer neighbourhood must be able to reach at least the specified cardinality.
    if arm.k is not None:
        fixed_count = int(base[~free_mask].sum())
        need_drop = max(0, fixed_count - arm.k)
        need_add = max(0, arm.k - fixed_count - free.size)
        for bit, need in ((1, need_drop), (0, need_add)):
            if need:
                pool = available[(~free_mask[available]) & (base[available] == bit)]
                if pool.size < need:
                    return lb, ub, rows, warm, False, False
                picks = pool[np.argsort(-priority[pool])[:need]]
                free_mask[picks] = True
    fixed = ~free_mask
    lb[fixed] = base[fixed]
    ub[fixed] = base[fixed]
    complete = np.all(free_mask[available]) and (strategy != 6 or arm.radius >= n)
    return lb, ub, rows, warm, complete, True


def _search(st, T, end):
    n, rng = st.n, st.rng
    st.offer(np.zeros(n, dtype=np.int64))
    st.offer(st.c > 0)
    st.offer(np.ones(n, dtype=np.int64))
    if not st.R:
        return
    full = _lp(st, None, min(end, _clock() + 0.07 * T))
    if full is False:
        return
    if full is None:
        ub = float(np.maximum(st.cf, 0).sum())
        full = _Arm(None, ub, np.full(n, 0.5), np.zeros(n), 1e-6 + abs(ub) * 1e-8, st)
    if full.closed_by_bound(st):
        return
    start = st.best_x if st.best_x is not None else (full.xl > 0.5).astype(np.int64)
    _tabu(st, start, full, min(end, _clock() + 0.035 * T))
    eng = _Engine(st)
    if _clock() >= end:
        return
    lb, ub, rows = _constraints(st, full, full)
    before = _clock()
    deadline = min(end, before + 0.42 * (end - before))
    code, bound = eng.run(lb, ub, rows, st.best_x, deadline,
                         stall=0.12 * T, min_time=0.18 * T)
    if code in (0, 2):
        return
    if bound is not None:
        # Solver bounds are used for termination; the LP reduced-cost constraint always uses the LP bound of the original pairing.
        if st.best_obj is not None and bound + full.tol < st.best_obj + 1:
            return
    while st.best_x is None and _clock() < end:
        _tabu(st, full.xl > 0.5, full, min(end, _clock() + 0.04 * T))
        if st.best_x is not None:
            break
        deadline = min(end, _clock() + 0.2 * T)
        code, bound = eng.run(np.zeros(n), np.ones(n), [], None, deadline)
        if code in (0, 2):
            return
        if code == -1:
            _tabu(st, rng.random(n) < full.xl, full, min(end, _clock() + 0.1 * T))
    if st.best_x is None:
        return
    arms = [full]
    visited = set()
    k_order = sorted(range(n + 1), key=lambda k: abs(k - float(full.xl.sum())))
    cursor = 0
    next_lp = _clock()
    last_full = _clock()
    total_spent = 0.0
    unit = T / (15 + math.log1p(T))
    while _clock() < end:
        if full.closed or full.closed_by_bound(st):
            return
        now = _clock()
        rem = end - now
        if now >= next_lp and rem > 2 * unit:
            kb = int(st.best_x.sum())
            if kb not in visited and abs(kb - float(full.xl.sum())) <= 1:
                k = kb
            else:
                while cursor < len(k_order) and k_order[cursor] in visited:
                    cursor += 1
                k = k_order[cursor] if cursor < len(k_order) else None
            if k is not None:
                visited.add(k)
                a = _lp(st, k, min(end, now + 0.04 * T))
                if a is not None and a is not False and not a.closed_by_bound(st):
                    arms.append(a)
            next_lp = _clock() + T / (3 + math.log1p(T))
            continue
        # After stagnation, restart the global model from a better starting point, keeping the valid reduced-cost cuts.
        if now - st.last_improve > 0.24 * T and now - last_full > 0.24 * T and rem > 0.17 * T:
            last_full = now
            lb, ub, rows = _constraints(st, full, full)
            code, bound = eng.run(lb, ub, rows, st.best_x, min(end, now + 0.16 * T),
                                 stall=0.065 * T, min_time=0.045 * T)
            if code in (0, 2) or (bound is not None and bound + full.tol < st.best_obj + 1):
                return
            continue
        active = []
        for a in arms:
            if a.closed or a.closed_by_bound(st):
                a.closed = True
            else:
                active.append(a)
        if not active:
            return
        # The global model gets at least half of the neighbourhood time; the rest is scheduled by layer bounds, gains and elapsed time.
        if full in active and full.spent < 0.5 * total_spent:
            arm = full
        else:
            arm = max(active, key=lambda a: (1.3 if a is full else 1.0) *
                      max(1.0, a.ub - st.best_obj) * a.reward / (1 + a.spent / max(unit, 1e-9)))
        if arm.k is not None and arm.k not in st.kbest:
            strategy = 0
        else:
            strategy = int(rng.choice(len(arm.weights), p=arm.weights / arm.weights.sum()))
        lb, ub, rows, warm, complete, possible = _neighborhood(st, full, arm, strategy)
        if not possible or np.any(lb > ub):
            arm.core = min(n, arm.core * 1.4 + 1)
            arm.size = min(n, arm.size * 1.4 + 1)
            arm.spent += unit * 0.05
            continue
        now = _clock()
        before = st.best_obj
        code, bound = eng.run(lb, ub, rows, warm, min(end, now + arm.slice))
        used = _clock() - now
        arm.spent += used
        total_spent += used
        improved = st.best_obj > before
        if complete and code in (0, 2):
            arm.closed = True
        if code in (0, 2):
            if strategy == 0:
                arm.core = min(n, arm.core * 1.3 + 1)
            elif strategy == 6:
                arm.radius = min(n, arm.radius * 1.25 + 1)
            else:
                arm.size = min(n, arm.size * 1.2 + 1)
        elif not improved:
            if strategy == 0:
                arm.core = max(arm.minimum, arm.core * 0.85)
            elif strategy == 6:
                arm.radius = max(2, arm.radius * 0.8)
            else:
                arm.size = max(arm.minimum, arm.size * 0.85)
            arm.slice = min(0.15 * T, arm.slice * 1.15)
        arm.weights = 0.96 * arm.weights + 0.04
        arm.reward = 0.85 * arm.reward + 0.15
        if improved:
            arm.weights[strategy] += 1
            arm.reward += min(2.0, (st.best_obj - before) / max(1.0, arm.ub - before))
        if _clock() < end and not full.closed:
            burst = min(0.02 * T, (0.18 if improved else 0.045) * max(used, unit * 0.1))
            _tabu(st, st.best_x, arm, min(end, _clock() + burst))


def solve(instance: dict, time_limit_s: float) -> dict:
    start = _clock()
    st = None
    indices = []
    N = 0
    try:
        T = float(time_limit_s)
        if not math.isfinite(T) or T < 0:
            T = 0.0
        # All search phases share the same wall-clock deadline, keeping a proportional margin for output and solver shutdown.
        end = start + 0.955 * T
        N, m, q = int(instance['n']), int(instance['m']), int(instance['q'])
        c = [int(v) for v in instance['cost_vector']]
        AL = [[int(v) for v in row] for row in instance['A_leq']]
        AG = [[int(v) for v in row] for row in instance['A_geq']]
        bL = [int(v) for v in instance['b_leq']]
        bG = [int(v) for v in instance['b_geq']]
        if N < 1 or len(c) != N or len(AL) != m or len(bL) != m or len(AG) != q or len(bG) != q:
            return {'x': None}
        if any(len(row) != N for row in AL + AG):
            return {'x': None}
        if not m and not q:
            return {'x': [int(v > 0) for v in c]}
        indices = [j for j in range(N) if all(AL[i][j] <= bL[i] for i in range(m))]
        if any(sum(row[j] for j in indices) < b for row, b in zip(AG, bG)):
            return {'x': None}
        if not any(bG):
            indices = [j for j in indices if c[j] > 0]
        if not indices:
            return {'x': [0] * N if not any(bG) else None}
        G, h = [], []
        for row, b in zip(AL, bL):
            reduced = [row[j] for j in indices]
            if sum(reduced) > b:
                G.append(reduced)
                h.append(b)
        for row, b in zip(AG, bG):
            if b > 0:
                G.append([-row[j] for j in indices])
                h.append(-b)
        reduced_c = [c[j] for j in indices]
        divisor = 0
        for v in reduced_c:
            divisor = math.gcd(divisor, abs(v))
        if divisor > 1:
            reduced_c = [v // divisor for v in reduced_c]
        st = _State(reduced_c, G, h, T)
        _search(st, T, end)
    except Exception:
        # Error handling never discards an incumbent that has already been verified exactly.
        pass
    if st is not None and st.best_x is not None:
        answer = [0] * N
        for p, j in enumerate(indices):
            answer[j] = int(st.best_x[p])
        return {'x': answer}
    return {'x': None}
