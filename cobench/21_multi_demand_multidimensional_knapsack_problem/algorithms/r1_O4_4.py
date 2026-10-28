"""Reduced-cost-guided persistent adaptive neighbourhood search
Combines integer bound propagation, LP core warm-up, warm-started MIP and gain-driven large neighbourhood search, adding candidates with tabu search and verifying feasibility exactly.
"""
# diagnosis
# The stronger algorithm links construction and improvement through a persistent model, warm starts and complementary neighbourhoods; it leads the weaker one on eight of the nine given scores.
# The weaker algorithm spreads its budget over cardinality slices and cold-started submodels, and fixing variables around LP rounding can miss cross-cardinality improvements.
# This round strengthens core warm-up, reduced-cost screening and gain-per-time scheduling on the same model; neighbourhood size is set jointly by the time limit and solver feedback.
import time
import math

import numpy as np

try:
    from scipy.optimize._highspy import _core as _HC
    _ = _HC._Highs
    _ = _HC.cb.HighsCallbackType
except Exception:
    _HC = None

try:
    from scipy.optimize import milp as _milp, LinearConstraint as _LC, Bounds as _Bounds, linprog as _linprog
except Exception:
    _milp = None
    _linprog = None


class _State:
    def __init__(self, c, AL, bL, AG, bG):
        self.c = c
        self.AL, self.bL, self.AG, self.bG = AL, bL, AG, bG
        self.best_x = None
        self.best_obj = None
        self.elite = []
        self.elite_keys = set()
        self.elite_cap = 2
        self.last_improve_time = time.perf_counter()
        mx = 0
        for arr in (c, AL, bL, AG, bG):
            if arr.size:
                mx = max(mx, int(np.abs(arr).max()))
        self.use_py = mx * (c.shape[0] + 1) >= 2 ** 61
        if self.use_py:
            self.cl = [int(v) for v in c]
            self.ALl = [[int(v) for v in r] for r in AL]
            self.AGl = [[int(v) for v in r] for r in AG]
            self.bLl = [int(v) for v in bL]
            self.bGl = [int(v) for v in bG]

    def exact(self, xi):
        if self.use_py:
            sel = [j for j in range(len(xi)) if xi[j]]
            for i, row in enumerate(self.ALl):
                if sum(row[j] for j in sel) > self.bLl[i]:
                    return None
            for k, row in enumerate(self.AGl):
                if sum(row[j] for j in sel) < self.bGl[k]:
                    return None
            return sum(self.cl[j] for j in sel)
        if self.bL.shape[0] > 0 and np.any(self.AL @ xi > self.bL):
            return None
        if self.bG.shape[0] > 0 and np.any(self.AG @ xi < self.bG):
            return None
        return int(self.c @ xi)

    def add_elite(self, xi, obj):
        if len(self.elite) >= self.elite_cap and obj <= self.elite[-1][0]:
            return
        key = xi.astype(np.int8).tobytes()
        if key in self.elite_keys:
            return
        self.elite.append((obj, key, xi.copy()))
        self.elite_keys.add(key)
        self.elite.sort(key=lambda t: -t[0])
        while len(self.elite) > self.elite_cap:
            _o, k, _x = self.elite.pop()
            self.elite_keys.discard(k)

    def offer(self, x):
        try:
            xi = np.asarray(x).reshape(-1)
            if xi.shape[0] != self.c.shape[0]:
                return False
            xi = (xi > 0.5).astype(np.int64)
            obj = self.exact(xi)
            if obj is None:
                return False
            self.add_elite(xi, obj)
            if self.best_obj is None or obj > self.best_obj:
                self.best_obj = obj
                self.best_x = xi.copy()
                self.last_improve_time = time.perf_counter()
                return True
        except Exception:
            pass
        return False


class _LS:
    def __init__(self, st, G, h, movable, pert_w, rng):
        self.st = st
        self.c = st.c.astype(np.float64)
        self.ci = st.c.astype(np.int64)
        self.Gi = G.astype(np.int64)
        self.G = G.astype(np.float64)
        self.hi = h.astype(np.int64)
        self.h = h.astype(np.float64)
        self.row_span = np.max(np.abs(self.G), axis=1) if G.shape[0] else np.zeros(0)
        self.n = len(st.c)
        self.R = G.shape[0]
        self.movable = movable
        self.fixed = st.fixed.copy()
        self.T = st.T
        self.rng = rng
        s = np.ones(self.R)
        for i in range(self.R):
            nz = np.abs(self.G[i])
            nz = nz[nz > 0]
            if nz.size:
                s[i] = 1.0 / max(1.0, float(nz.mean()))
        self.s = s
        nzc = np.abs(self.c[self.c != 0])
        self.base_lam = float(nzc.mean()) if nzc.size else 1.0
        p = np.where(movable, np.asarray(pert_w, dtype=np.float64), 0.0)
        if p.sum() <= 0:
            p = movable.astype(np.float64)
        if p.sum() <= 0:
            p = np.ones(self.n)
        self.pert_p = p / p.sum()

    def run(self, x0, deadline):
        st = self.st
        n, R = self.n, self.R
        if R == 0 or n == 0:
            return
        c, G, h, Gi, hi = self.c, self.G, self.h, self.Gi, self.hi
        rng = self.rng
        movable = self.movable
        x = np.array(x0, dtype=np.int64)
        x[~movable] = self.fixed[~movable]
        L = Gi @ x
        obj = int(self.ci @ x)
        w = np.ones(R)
        lam = self.base_lam
        tabu = np.zeros(n, dtype=np.int64)
        t_lo = max(1, min(n // 3, 3 + n // 50))
        t_hi = max(t_lo + 1, min(max(2, n // 2), 10 + n // 15))
        restart_wait = max(self.T * 0.012, (deadline - time.perf_counter()) * 0.3)
        K = max(1, min(n, int(math.sqrt(n) * (0.35 + 0.15 * math.log1p(self.T))) + 1))
        it = 0
        last_imp = time.perf_counter()
        pert_k = max(1, min(n, int(math.sqrt(n) * (0.5 + 0.2 * math.log1p(self.T)))))
        eps = 0.02
        Lf = L.astype(np.float64)
        while True:
            if time.perf_counter() >= deadline:
                break
            it += 1
            d = 1 - 2 * x
            df = d.astype(np.float64)
            Lf = L.astype(np.float64)
            active = (h - Lf) < self.row_span
            ws = (w * self.s)[active]
            Ga, La, ha = G[active], Lf[active], h[active]
            P = float(ws @ np.maximum(La - ha, 0.0))
            NL = La[:, None] + Ga * df[None, :]
            NV = np.maximum(NL - ha[:, None], 0.0)
            newP = ws @ NV
            feas_after = ~(NV > 0).any(axis=0)
            delta = c * df - lam * (newP - P)
            newobj = obj + self.ci * d
            bo = st.best_obj
            asp = feas_after if bo is None else (feas_after & (newobj > bo))
            canmove = movable
            nontabu = (tabu <= it) & canmove
            allowed = nontabu | (asp & canmove)
            if not allowed.any():
                allowed = canmove.copy()
                if not allowed.any():
                    break
            noise = rng.random(n) * 1e-9 * (abs(lam) + 1.0)
            score = np.where(allowed, delta + noise, -np.inf)
            j = int(np.argmax(score))
            best_val = score[j]
            move_a = -1
            move_d = -1
            ones = np.nonzero((x == 1) & nontabu)[0]
            zeros = np.nonzero((x == 0) & nontabu & movable)[0]
            if ones.size > 0 and zeros.size > 0:
                dsel = ones[np.argpartition(-delta[ones], K - 1)[:K]] if ones.size > K else ones
                asel = zeros[np.argpartition(-delta[zeros], K - 1)[:K]] if zeros.size > K else zeros
                NL2 = La[:, None, None] + Ga[:, asel][:, :, None] - Ga[:, dsel][:, None, :]
                NV2 = np.maximum(NL2 - ha[:, None, None], 0.0)
                pen = np.tensordot(ws, NV2, axes=(0, 0))
                ps = (c[asel][:, None] - c[dsel][None, :]) - lam * (pen - P)
                ij = int(np.argmax(ps))
                ai, di = divmod(ij, dsel.size)
                if ps[ai, di] > best_val + 1e-12:
                    move_a = int(asel[ai])
                    move_d = int(dsel[di])
            if move_a >= 0:
                x[move_a] = 1
                x[move_d] = 0
                L = L + Gi[:, move_a] - Gi[:, move_d]
                obj += int(self.ci[move_a]) - int(self.ci[move_d])
                tabu[move_a] = it + int(rng.integers(t_lo, t_hi + 1))
                tabu[move_d] = it + int(rng.integers(t_lo, t_hi + 1))
            else:
                dj = int(d[j])
                x[j] += dj
                L = L + dj * Gi[:, j]
                obj += int(self.ci[j]) * dj
                tabu[j] = it + int(rng.integers(t_lo, t_hi + 1))
            vmask = L > hi
            if not vmask.any():
                bo = st.best_obj
                if bo is None or obj > bo:
                    if st.offer(x):
                        last_imp = time.perf_counter()
                elif len(st.elite) < st.elite_cap or obj > st.elite[-1][0]:
                    st.add_elite(x, obj)
                lam = max(lam / (1.0 + eps), 1e-6 * self.base_lam)
            else:
                lam = min(lam * (1.0 + eps), self.base_lam * 1e8)
                w[vmask] += 1.0
                if w.max() > 1000.0:
                    w /= w.mean()
            if time.perf_counter() - last_imp > restart_wait:
                last_imp = time.perf_counter()
                if st.best_x is not None and rng.random() < 0.8:
                    x = st.best_x.copy()
                kk = min(n, int(rng.integers(1, pert_k + 1)) + 1)
                if np.count_nonzero(self.pert_p) >= kk:
                    idx = rng.choice(n, size=kk, replace=False, p=self.pert_p)
                else:
                    idx = rng.choice(n, size=kk, replace=False)
                x[idx] = 1 - x[idx]
                x[~movable] = self.fixed[~movable]
                L = Gi @ x
                obj = int(self.ci @ x)
                w[:] = 1.0
                lam = self.base_lam
                tabu[:] = 0
                tabu[idx] = it + t_hi
                restart_wait = max(self.T * 0.012, (deadline - time.perf_counter()) * 0.3) * (0.7 + rng.random())


_OPT, _INF, _LIM, _ERR = 0, 2, 1, -1


class _HighsEngine:
    """Reuse the model, update variable bounds and warm start; the callback interrupts on wall clock and stagnation time."""

    def __init__(self, st, Gf, rl, ru, n, integer=True):
        self.st = st
        self.n = n
        h = _HC._Highs()
        h.setOptionValue("output_flag", False)
        inf = h.getInfinity()
        R = Gf.shape[0]
        self.R = R
        rl2 = np.where(np.isfinite(rl), rl, -inf).astype(np.float64)
        ru2 = np.where(np.isfinite(ru), ru, inf).astype(np.float64)
        mask = Gf != 0
        nnz = int(mask.sum())
        rows_nnz = mask.sum(axis=1)
        start = np.concatenate([[0], np.cumsum(rows_nnz)]).astype(np.int32)
        cols = np.nonzero(mask)[1].astype(np.int32)
        vals = Gf[mask].astype(np.float64)
        integ = np.ones(n, dtype=np.int32) if integer else np.zeros(n, dtype=np.int32)
        stt = h.passModel(int(n), int(R), int(nnz), 2, -1, 0.0, st.c.astype(np.float64),
                          np.zeros(n), np.ones(n), rl2, ru2, start, cols, vals, integ)
        if stt != _HC.HighsStatus.kOk and int(stt) != 0:
            if "Error" in str(stt):
                raise RuntimeError("model initialisation failed")
        self.h = h
        self.inf = inf
        self.idx = np.arange(n, dtype=np.int32)
        self.run_end = 0.0
        self.stall = None
        self.min_end = 0.0
        self.last_imp = 0.0
        self.run_start = 0.0
        self.run_best = None
        if integer:
            try:
                h.setOptionValue("mip_rel_gap", 0.0)
            except Exception:
                pass
            cbt = _HC.cb.HighsCallbackType
            self.k_imp = int(cbt.kCallbackMipImprovingSolution)
            self.k_int = int(cbt.kCallbackMipInterrupt)
            h.setCallback(self._cb, None)
            h.startCallback(cbt.kCallbackMipImprovingSolution)
            h.startCallback(cbt.kCallbackMipInterrupt)

    def _cb(self, ct, msg, dout, din, ud):
        try:
            ct = int(ct)
            if ct == self.k_int:
                now = time.perf_counter()
                stop = now >= self.run_end
                if (not stop) and self.stall is not None and now >= self.min_end:
                    wait = max(self.stall, self.last_imp - self.run_start)
                    if now - self.last_imp > wait:
                        stop = True
                din.user_interrupt = bool(stop)
            elif ct == self.k_imp:
                v = float(dout.objective_function_value)
                if self.run_best is None or v > self.run_best + 0.5:
                    self.run_best = v
                    self.last_imp = time.perf_counter()
        except Exception:
            pass

    def solve_lp(self, tl):
        h = self.h
        h.setOptionValue("time_limit", float(max(0.0001, min(tl, self.st.end - time.perf_counter()))))
        h.run()
        ms = h.getModelStatus()
        if ms == _HC.HighsModelStatus.kInfeasible:
            return _INF, None, None
        if ms != _HC.HighsModelStatus.kOptimal:
            return _LIM, None, None
        sol = h.getSolution()
        x = np.array(sol.col_value, dtype=np.float64)[:self.n]
        rc = None
        try:
            if sol.dual_valid:
                rc = np.array(sol.col_dual, dtype=np.float64)[:self.n]
        except Exception:
            rc = None
        return _OPT, x, rc

    def run(self, lb, ub, x_start, tl, stall=None, min_time=0.0, lb_row=None):
        h = self.h
        n = self.n
        now = time.perf_counter()
        tl = min(tl, self.st.end - now)
        if tl <= 0:
            return _LIM, None
        self.run_end = now + tl
        self.stall = stall
        self.min_end = now + min_time
        self.last_imp = now
        self.run_start = now
        self.run_best = None
        added = False
        try:
            h.changeColsBounds(n, self.idx, np.asarray(lb, dtype=np.float64), np.asarray(ub, dtype=np.float64))
            if lb_row is not None:
                coef, lo, up = lb_row
                nzi = np.nonzero(coef)[0].astype(np.int32)
                h.addRow(float(lo), float(up), int(nzi.size), nzi, coef[nzi].astype(np.float64))
                added = True
            if x_start is not None:
                try:
                    h.setSolution(n, self.idx, np.asarray(x_start, dtype=np.float64))
                except Exception:
                    try:
                        s = _HC.HighsSolution()
                        s.col_value = [float(v) for v in x_start]
                        h.setSolution(s)
                    except Exception:
                        pass
            try:
                rt = float(h.getRunTime())
            except Exception:
                rt = 0.0
            h.setOptionValue("time_limit", float(rt + max(0.0001, self.run_end - time.perf_counter())))
            h.run()
            ms = h.getModelStatus()
            try:
                sol = h.getSolution()
                if sol.value_valid:
                    x = np.array(sol.col_value, dtype=np.float64)[:n]
                    self.st.offer(x)
            except Exception:
                pass
            db = None
            try:
                v = float(h.getInfo().mip_dual_bound)
                if math.isfinite(v):
                    db = v
            except Exception:
                db = None
            if ms == _HC.HighsModelStatus.kOptimal:
                code = _OPT
            elif ms == _HC.HighsModelStatus.kInfeasible:
                code = _INF
            else:
                code = _LIM
        except Exception:
            code, db = _ERR, None
        if added:
            try:
                h.deleteRows(1, np.array([self.R], dtype=np.int32))
            except Exception:
                try:
                    h.changeRowBounds(self.R, -self.inf, self.inf)
                except Exception:
                    pass
        return code, db


class _ScipyEngine:
    def __init__(self, st, Gf, rl, ru, n):
        self.st = st
        self.n = n
        self.Gf, self.rl, self.ru = Gf, rl, ru
        self.cf = st.c.astype(np.float64)

    def run(self, lb, ub, x_start, tl, stall=None, min_time=0.0, lb_row=None):
        n = self.n
        cons = [_LC(self.Gf, self.rl, self.ru)]
        if lb_row is not None:
            coef, lo, up = lb_row
            cons.append(_LC(coef.reshape(1, n), np.array([lo]), np.array([up])))
        try:
            res = _milp(-self.cf, constraints=cons, integrality=np.ones(n),
                        bounds=_Bounds(np.asarray(lb, float), np.asarray(ub, float)),
                        options={"time_limit": float(max(0.0001, min(tl, self.st.end - time.perf_counter()))), "disp": False, "mip_rel_gap": 0.0})
        except Exception:
            return _ERR, None
        if res.x is not None:
            self.st.offer(res.x)
        db = None
        try:
            v = getattr(res, "mip_dual_bound", None)
            if v is not None and math.isfinite(v):
                db = -float(v)
        except Exception:
            db = None
        if res.status == 0:
            return _OPT, db
        if res.status == 2:
            return _INF, db
        return _LIM, db



def solve(instance: dict, time_limit_s: float) -> dict:
    started = time.perf_counter()
    st = None
    try:
        T = float(time_limit_s)
        if not math.isfinite(T) or T <= 0:
            T = 0.001
        end = started + T * 0.965 - min(0.01, 0.01 * T)
        n, m, q = int(instance['n']), int(instance['m']), int(instance['q'])
        cr = instance['cost_vector']
        al, bl = instance['A_leq'], instance['b_leq']
        ag, bg = instance['A_geq'], instance['b_geq']
        if n < 1 or m < 0 or q < 0 or len(cr) != n or len(al) != m or len(bl) != m \
                or len(ag) != q or len(bg) != q or any(len(r) != n for r in al) \
                or any(len(r) != n for r in ag):
            return {'x': None}
        # Large integers are accumulated as Python integers to avoid accepting infeasible solutions after int64 overflow.
        mx = max([abs(int(v)) for v in cr] + [int(v) for v in bl] + [int(v) for v in bg])
        for row in al + ag:
            mx = max(mx, max((int(v) for v in row), default=0))
        dtype = object if mx * (n + 1) >= 2 ** 61 else np.int64
        c = np.asarray(cr, dtype=dtype)
        AL = np.asarray(al, dtype=dtype).reshape(m, n)
        AG = np.asarray(ag, dtype=dtype).reshape(q, n)
        bL = np.asarray(bl, dtype=dtype)
        bG = np.asarray(bg, dtype=dtype)
        st = _State(c, AL, bL, AG, bG)
        st.end, st.T = end, T
        st.elite_cap = max(2, min(n + 1, int(2 + math.sqrt(n) * math.log1p(T) / 3)))
        _run(st, n, m, q)
    except Exception:
        pass
    if st is not None and st.best_x is not None:
        return {'x': [int(v) for v in st.best_x]}
    return {'x': None}


def _run(st, n, m, q):
    c, AL, bL, AG, bG = st.c, st.AL, st.bL, st.AG, st.bG
    end, T = st.end, st.T
    rng = np.random.default_rng(20261001)
    if m == 0 and q == 0:
        st.offer(c > 0)
        return
    st.offer(np.zeros(n, dtype=np.int64))
    st.offer(c > 0)
    lo = np.zeros(n, dtype=np.int64)
    hi = np.ones(n, dtype=np.int64)
    # Forced take/leave propagation for non-negative coefficients; every decision uses integers.
    while time.perf_counter() < end:
        old_lo, old_hi = lo.copy(), hi.copy()
        if m:
            residual = bL - AL @ lo
            if np.any(residual < 0):
                return
            excluded = np.any(AL > residual[:, None], axis=0) & (lo == 0)
            hi[excluded] = 0
        if q:
            covered = AG @ hi
            if np.any(covered < bG):
                return
            forced = np.any(covered[:, None] - AG < bG[:, None], axis=0) & (hi == 1)
            lo[forced] = 1
        if np.any(lo > hi):
            return
        if np.array_equal(lo, old_lo) and np.array_equal(hi, old_hi):
            break
    st.offer(lo)
    st.offer(hi)
    st.offer(np.maximum(lo, np.minimum(hi, (c > 0).astype(np.int64))))
    movable = lo != hi
    st.fixed = lo.copy()
    mov_idx = np.flatnonzero(movable)
    n_mov = mov_idx.size
    if not n_mov or time.perf_counter() >= end:
        return
    G = np.vstack((AL, -AG))
    h = np.concatenate((bL, -bG))
    Gf = G.astype(np.float64)
    hf = h.astype(np.float64)
    cf = c.astype(np.float64)
    if not np.all(np.isfinite(Gf)) or not np.all(np.isfinite(cf)):
        return
    # Remove rows that always hold within the propagated variable bounds.
    upper_activity = np.maximum(G, 0) @ hi + np.minimum(G, 0) @ lo
    keep = upper_activity > h
    G, h, Gf, hf = G[keep], h[keep], Gf[keep], hf[keep]
    if not h.size:
        st.offer(np.maximum(lo, np.minimum(hi, (c > 0).astype(np.int64))))
        return
    rl = np.full(h.size, -np.inf)
    xl, rc, lp_bound = None, None, None
    dual_valid = False
    lp_time = min(end - time.perf_counter(), 0.075 * T)
    if lp_time > 0 and _HC is not None:
        try:
            lp = _HighsEngine(st, Gf, rl, hf, n, integer=False)
            lp.h.changeColsBounds(n, lp.idx, lo.astype(float), hi.astype(float))
            code, xlp, rcv = lp.solve_lp(lp_time)
            if code == _INF and st.best_x is None:
                return
            if code == _OPT and xlp is not None:
                xl = np.clip(xlp, lo, hi)
                rc = rcv
                dual_valid = rc is not None and np.all(np.isfinite(rc))
                lp_bound = float(cf @ xl)
            del lp
        except Exception:
            pass
    if xl is None and _linprog is not None and time.perf_counter() < end:
        try:
            res = _linprog(-cf, A_ub=Gf, b_ub=hf, bounds=list(zip(lo, hi)),
                           method='highs', options={'time_limit': min(0.05 * T, end - time.perf_counter())})
            if res.status == 2 and st.best_x is None:
                return
            if res.status == 0 and res.x is not None:
                xl = np.clip(res.x, lo, hi)
                rc = -(np.asarray(res.lower.marginals) + np.asarray(res.upper.marginals))
                dual_valid = np.all(np.isfinite(rc))
                lp_bound = float(-res.fun)
        except Exception:
            pass
    if xl is not None:
        for threshold in (0.5, 1e-7, 1 - 1e-7):
            st.offer(np.maximum(lo, np.minimum(hi, (xl > threshold).astype(np.int64))))
    if rc is None or not np.all(np.isfinite(rc)):
        rc = cf.copy()
        dual_valid = False
    arc = np.abs(rc)
    positive_rc = arc[movable & (arc > 1e-9)]
    rc_scale = float(np.median(positive_rc)) if positive_rc.size else 1.0
    rc_scale = max(rc_scale, 1e-9)
    frac = (np.minimum(xl, 1 - xl) > 1e-7) & movable if xl is not None else np.zeros(n, dtype=bool)
    frac_count = int(frac.sum())
    pert_w = np.exp(-np.minimum(arc / rc_scale, 25)) + 0.05 + frac
    if lp_bound is not None and st.best_obj is not None and st.best_obj >= math.floor(
            lp_bound + 1e-6 * (1.0 + float(np.abs(cf).sum()))):
        return
    ls = None if st.use_py else _LS(st, G, h, movable, pert_w, rng)
    x0 = np.maximum(lo, np.minimum(hi, (xl > 0.5).astype(np.int64))) if xl is not None else lo.copy()
    if ls is not None and time.perf_counter() < end:
        ls.run(st.best_x if st.best_x is not None else x0,
               min(end, time.perf_counter() + T * (0.025 if st.best_x is not None else 0.055)))
    eng = None
    if _HC is not None and time.perf_counter() < end:
        try:
            eng = _HighsEngine(st, Gf, rl, hf, n)
        except Exception:
            pass
    if eng is None and _milp is not None:
        eng = _ScipyEngine(st, Gf, rl, hf, n)
    if eng is None:
        if ls is not None:
            ls.run(st.best_x if st.best_x is not None else x0, end)
        return

    tolerance = 1e-6 * (1.0 + float(np.abs(cf).sum()))
    global_bound = lp_bound + tolerance if lp_bound is not None else None

    def proven():
        return st.best_obj is not None and global_bound is not None \
            and st.best_obj >= math.floor(global_bound)

    def bounds():
        lb, ub = lo.astype(float), hi.astype(float)
        if dual_valid and xl is not None and lp_bound is not None and st.best_x is not None:
            gap = max(0.0, lp_bound - st.best_obj) + tolerance
            at_zero = movable & (xl < 1e-7) & (rc < -gap) & (st.best_x == 0)
            at_one = movable & (xl > 1 - 1e-7) & (rc > gap) & (st.best_x == 1)
            ub[at_zero] = 0.0
            lb[at_one] = 1.0
        return lb, ub

    def run_mip(lb, ub, start, seconds, row=None, stall=None):
        remaining = end - time.perf_counter()
        if remaining <= 0:
            return _LIM, None, 0.0
        begin = time.perf_counter()
        status, db = eng.run(lb, ub, start, min(seconds, remaining),
                             stall=stall, min_time=min(seconds * 0.4, 0.12 * T), lb_row=row)
        return status, db, time.perf_counter() - begin

    def record_bound(db):
        nonlocal global_bound
        if db is not None and math.isfinite(db):
            u = db + tolerance
            global_bound = u if global_bound is None else min(global_bound, u)

    if proven():
        return
    # Solve the LP core first: keep all disagreements between the current solution and LP rounding so the warm start stays inside the neighbourhood.
    strength = math.log1p(T)
    core_fraction = min(0.6, 0.16 + 0.035 * strength)
    k_min = min(n_mov, max(2, int(math.sqrt(n_mov) * (0.7 + 0.2 * strength))))
    k_initial = min(n_mov, max(k_min, frac_count + int(math.sqrt(n_mov)), int(core_fraction * n_mov)))
    if xl is not None and time.perf_counter() < end:
        anchor = x0.copy()
        order = mov_idx[np.argsort(arc[mov_idx] - frac[mov_idx] * (rc_scale + 1))]
        free = order[:k_initial]
        if st.best_x is not None:
            free = np.union1d(free, np.flatnonzero(st.best_x != anchor))
        lb, ub = anchor.astype(float), anchor.astype(float)
        lb[free], ub[free] = lo[free], hi[free]
        code, db, used = run_mip(lb, ub, st.best_x, T * 0.08)
        if free.size == n_mov:
            record_bound(db)
            if code == _OPT and st.best_x is not None:
                return
            if code == _INF and st.best_x is None:
                return
    if proven():
        return
    # Give the global search a good initial solution while reserving enough budget to improve neighbourhoods repeatedly.
    if time.perf_counter() < end:
        lb, ub = bounds()
        remaining = end - time.perf_counter()
        cap = remaining * (0.36 if st.best_x is not None else 0.65)
        code, db, used = run_mip(lb, ub, st.best_x, cap, stall=T * 0.12)
        record_bound(db)
        if code == _OPT and st.best_x is not None:
            return
        if code == _INF and st.best_x is None:
            return
    if proven():
        return
    while st.best_x is None and time.perf_counter() < end:
        if ls is not None:
            ls.run(x0, min(end, time.perf_counter() + T * 0.045))
        if st.best_x is not None:
            break
        code, db, used = run_mip(lo, hi, None, end - time.perf_counter())
        if code in (_OPT, _INF):
            return
        if code == _ERR and ls is None:
            return
    if st.best_x is None:
        return

    families = ['core', 'rins', 'elite', 'balanced', 'branch']
    weights = np.ones(len(families))
    if xl is None:
        weights[1] = 0.0
    sizes = np.full(len(families), float(k_initial))
    base_slice = T / (12.0 + 2.0 * math.log1p(n_mov))
    slices = np.full(len(families), base_slice)
    exposure = np.zeros(n)
    radius = max(1.0, min(float(n_mov), math.sqrt(n_mov) * (0.7 + 0.3 * strength)))
    last_full = time.perf_counter()
    total_scale = np.maximum(np.abs(Gf).sum(axis=1), 1.0)
    # No cap on restarts or iterations; neighbourhoods scale with solver feedback and the search continues until the wall-clock deadline.
    while time.perf_counter() < end:
        if proven():
            return
        now = time.perf_counter()
        rem = end - now
        gl, gu = bounds()
        eligible = np.flatnonzero(gl != gu)
        count = eligible.size
        if not count:
            return
        if now - st.last_improve_time > 0.24 * T and now - last_full > 0.24 * T \
                and rem > 0.12 * T:
            last_full = now
            code, db, used = run_mip(gl, gu, st.best_x, min(rem * 0.45, 0.2 * T), stall=0.07 * T)
            record_bound(db)
            if code == _OPT:
                return
            continue
        f = int(rng.choice(len(families), p=weights / weights.sum()))
        family = families[f]
        bx = st.best_x.copy()
        k = min(count, max(min(count, k_min), int(round(sizes[f]))))
        noise = -np.log(-np.log(np.clip(rng.random(n), 1e-12, 1 - 1e-12)))
        row = None
        if family == 'branch':
            free = eligible
            coef = np.zeros(n)
            coef[eligible] = 1.0 - 2.0 * bx[eligible]
            row = (coef, -np.inf, float(max(1, int(radius)) - int(bx[eligible].sum())))
        else:
            lp_dis = np.abs(xl - bx) if xl is not None else np.zeros(n)
            loss = np.maximum((2.0 * bx - 1.0) * rc, 0.0) / rc_scale
            dis = np.zeros(n)
            if len(st.elite) > 1:
                # Prefer elite differences, avoiding releasing the same low reduced-cost variables every time.
                elite = st.elite[int(rng.integers(len(st.elite)))][2]
                dis = (elite != bx).astype(float)
            novelty = 1.0 / np.sqrt(1.0 + exposure)
            if family == 'core':
                priority = -np.log1p(loss) + 1.4 * lp_dis + dis + 0.55 * noise + 0.8 * novelty
            elif family == 'rins':
                priority = 2.0 * lp_dis - 0.2 * np.log1p(loss) + 0.6 * noise + novelty
            elif family == 'elite':
                priority = 2.0 * dis + 0.4 * lp_dis + noise + novelty
            else:
                slack = (hf - Gf @ bx.astype(float)) / total_scale
                tight = np.argsort(slack)[:max(1, min(h.size, int(math.sqrt(h.size))))]
                rr = int(rng.choice(tight))
                a = np.abs(Gf[rr])
                priority = a / max(1.0, float(a.max())) - 0.35 * np.log1p(loss) + noise + novelty
            if k == count:
                free = eligible
            elif family == 'balanced':
                ones = eligible[bx[eligible] == 1]
                zeros = eligible[bx[eligible] == 0]
                if ones.size and zeros.size:
                    take_one = min(ones.size, max(k - zeros.size, int(k * np.clip(ones.size / count, 0.25, 0.75))))
                else:
                    take_one = min(k, ones.size)
                oo = ones[np.argsort(-priority[ones])[:take_one]]
                zz = zeros[np.argsort(-priority[zeros])[:k - take_one]]
                free = np.concatenate((oo, zz))
            else:
                free = eligible[np.argpartition(-priority[eligible], k - 1)[:k]]
        lb, ub = bx.astype(float), bx.astype(float)
        lb[free], ub[free] = gl[free], gu[free]
        exposure[free] += 1.0
        before = st.best_obj
        old_gap = max(1.0, global_bound - before) if global_bound is not None else max(1.0, abs(before) * 0.05)
        code, db, used = run_mip(lb, ub, bx, min(rem, slices[f]), row=row)
        gain = max(0, st.best_obj - before)
        is_full = row is None and free.size == count
        if is_full:
            record_bound(db)
            if code == _OPT:
                return
        reward = min(20.0, (gain / old_gap) * T / max(used, T * 1e-5))
        weights[f] = 0.8 * weights[f] + 0.2 * (0.3 + reward)
        if code in (_OPT, _INF):
            if family == 'branch':
                radius = min(count, radius * (1.25 if gain else 1.5) + 1)
            else:
                sizes[f] = min(count, sizes[f] * (1.15 if gain else 1.35) + 1)
            slices[f] = max(base_slice * 0.6, slices[f] * 0.9)
        elif code == _LIM and not gain:
            if family == 'branch':
                radius = max(1.0, radius * 0.8)
            else:
                sizes[f] = max(min(k_min, count), sizes[f] * 0.8)
            slices[f] = min(T * 0.16, slices[f] * 1.2)
        if ls is not None and time.perf_counter() < end:
            burst = min(0.02 * T, used * (0.18 if gain else 0.045))
            ls.run(st.best_x, min(end, time.perf_counter() + burst))
