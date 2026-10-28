"""MDMKP solver (maximise c.x  s.t.  A_leq x <= b_leq,  A_geq x >= b_geq,  x binary).

Pipeline (wall-clock driven, every stage scaled by time_limit_s):
  1. parse / trivial cases / quick infeasibility proofs, items that can never fit are fixed to 0
  2. LP relaxation (HiGHS) -> x_lp, reduced costs (LP infeasible => IP infeasible)
  3. short penalty tabu search (adaptive row weights, lambda oscillation, swap moves) -> incumbent
  4. ONE persistent HiGHS MIP model (scipy's bundled HiGHS low-level API):
       * full MIP, warm-started with the incumbent, every improving solution captured by callback,
         interrupted by callback when it stagnates (so long MIP runs continue only while useful)
       * MIP-LNS on the same model: only column bounds change, each sub-MIP warm-started with the
         incumbent (HiGHS only searches for strictly better solutions); neighbourhoods chosen by an
         adaptive roulette over reduced-cost core / RINS / elite disagreement / random / tight-row /
         local-branching (soft Hamming ball row); neighbourhood size adapted to sub-MIP outcome
       * tabu bursts polish every improvement and refill the elite pool
       * after long LNS stagnation the full MIP is re-run warm-started from the better incumbent
  Fallback to scipy.optimize.milp (rebuilt per call) if the low-level API is missing.
All candidate solutions are verified with exact integer arithmetic before being accepted.
"""
import time
import math

import numpy as np

try:
    from scipy.optimize._highspy import _core as _HC
    _ = _HC._Highs
    _ = _HC.cb.HighsCallbackType
except Exception:  # pragma: no cover
    _HC = None

try:
    from scipy.optimize import milp as _milp, LinearConstraint as _LC, Bounds as _Bounds, linprog as _linprog
except Exception:  # pragma: no cover
    _milp = None
    _linprog = None


# ----------------------------------------------------------------------------- state
class _State:
    def __init__(self, c, AL, bL, AG, bG):
        self.c = c
        self.AL, self.bL, self.AG, self.bG = AL, bL, AG, bG
        self.best_x = None
        self.best_obj = None
        self.elite = []
        self.elite_keys = set()
        self.elite_cap = 12
        self.last_improve_time = time.time()
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
                self.last_improve_time = time.time()
                return True
        except Exception:
            pass
        return False


# ----------------------------------------------------------------------------- tabu LS
class _LS:
    def __init__(self, st, G, h, movable, pert_w, rng):
        self.st = st
        self.c = st.c.astype(np.float64)
        self.ci = st.c.astype(np.int64)
        self.Gi = G.astype(np.int64)
        self.G = G.astype(np.float64)
        self.hi = h.astype(np.int64)
        self.h = h.astype(np.float64)
        self.n = len(st.c)
        self.R = G.shape[0]
        self.movable = movable
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
        x[~movable] = 0
        L = Gi @ x
        obj = int(self.ci @ x)
        w = np.ones(R)
        lam = self.base_lam
        tabu = np.zeros(n, dtype=np.int64)
        t_lo = max(1, min(n // 3, 3 + n // 50))
        t_hi = max(t_lo + 1, min(max(2, n // 2), 10 + n // 15))
        stall_limit = max(200, 10 * n)
        K = max(2, min(8, int(math.sqrt(n)) + 1))
        it = 0
        last_imp = 0
        pert_k = max(2, n // 25)
        eps = 0.02
        Lf = L.astype(np.float64)
        while True:
            if (it & 7) == 0 and time.time() >= deadline:
                break
            it += 1
            d = 1 - 2 * x
            df = d.astype(np.float64)
            ws = w * self.s
            Lf = L.astype(np.float64)
            P = float(ws @ np.maximum(Lf - h, 0.0))
            NL = Lf[:, None] + G * df[None, :]
            NV = np.maximum(NL - h[:, None], 0.0)
            newP = ws @ NV
            feas_after = ~(NV > 0).any(axis=0)
            delta = c * df - lam * (newP - P)
            newobj = obj + self.ci * d
            bo = st.best_obj
            asp = feas_after if bo is None else (feas_after & (newobj > bo))
            canmove = movable | (x == 1)
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
                NL2 = Lf[:, None, None] + G[:, asel][:, :, None] - G[:, dsel][:, None, :]
                NV2 = np.maximum(NL2 - h[:, None, None], 0.0)
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
                        last_imp = it
                elif len(st.elite) < st.elite_cap or obj > st.elite[-1][0]:
                    st.add_elite(x, obj)
                lam = max(lam / (1.0 + eps), 1e-6 * self.base_lam)
            else:
                lam = lam * (1.0 + eps)
                w[vmask] += 1.0
                if it % 200 == 0:
                    w /= w.mean()
            if it - last_imp > stall_limit:
                last_imp = it
                if st.best_x is not None and rng.random() < 0.8:
                    x = st.best_x.copy()
                kk = min(n, int(rng.integers(1, pert_k + 1)) + 1)
                if np.count_nonzero(self.pert_p) >= kk:
                    idx = rng.choice(n, size=kk, replace=False, p=self.pert_p)
                else:
                    idx = rng.choice(n, size=kk, replace=False)
                x[idx] = 1 - x[idx]
                x[~movable] = 0
                L = Gi @ x
                obj = int(self.ci @ x)
                w[:] = 1.0
                lam = self.base_lam
                tabu[:] = 0
                tabu[idx] = it + t_hi
                stall_limit = max(200, 10 * n) + int(rng.integers(0, 2 * n + 1))


# ----------------------------------------------------------------------------- MIP engines
_OPT, _INF, _LIM, _ERR = 0, 2, 1, -1


class _HighsEngine:
    """Persistent HiGHS model; bounds changed per call, warm start, callback capture/interrupt."""

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
                raise RuntimeError("passModel failed")
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
                now = time.time()
                stop = now >= self.run_end
                if (not stop) and self.stall is not None and now >= self.min_end:
                    # stagnation: no own improvement for max(stall, time needed to reach the last one)
                    wait = max(self.stall, self.last_imp - self.run_start)
                    if now - self.last_imp > wait:
                        stop = True
                din.user_interrupt = bool(stop)
            elif ct == self.k_imp:
                # NOTE: the solution vector exposed to callbacks is unreliable -> only use the value
                v = float(dout.objective_function_value)
                if self.run_best is None or v > self.run_best + 0.5:
                    self.run_best = v
                    self.last_imp = time.time()
        except Exception:
            pass

    def solve_lp(self, tl):
        h = self.h
        h.setOptionValue("time_limit", float(max(0.02, tl)))
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
        now = time.time()
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
            h.setOptionValue("time_limit", float(rt + tl + 0.05))
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
                # cannot remove: rebuild impossible here, neutralise the row instead
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
                        options={"time_limit": float(max(0.02, tl)), "disp": False, "mip_rel_gap": 0.0})
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


# ----------------------------------------------------------------------------- main
def solve(instance: dict, time_limit_s: float) -> dict:
    t0 = time.time()
    try:
        T = float(time_limit_s)
        if not (T > 0) or math.isnan(T) or math.isinf(T):
            T = 10.0
    except Exception:
        T = 10.0
    end = t0 + max(0.05, T * 0.94 - 0.06)
    box = {"st": None}
    try:
        c = np.array(instance.get("cost_vector", []), dtype=np.int64).reshape(-1)
        n = int(c.shape[0])
        if n == 0:
            return {"x": []}
        Al_raw = instance.get("A_leq") or []
        bl_raw = instance.get("b_leq") or []
        Ag_raw = instance.get("A_geq") or []
        bg_raw = instance.get("b_geq") or []
        m = len(bl_raw)
        q = len(bg_raw)
        AL = np.array(Al_raw, dtype=np.int64).reshape(m, n) if m > 0 else np.zeros((0, n), dtype=np.int64)
        bL = np.array(bl_raw, dtype=np.int64).reshape(m) if m > 0 else np.zeros(0, dtype=np.int64)
        AG = np.array(Ag_raw, dtype=np.int64).reshape(q, n) if q > 0 else np.zeros((0, n), dtype=np.int64)
        bG = np.array(bg_raw, dtype=np.int64).reshape(q) if q > 0 else np.zeros(0, dtype=np.int64)
    except Exception:
        return {"x": None}
    try:
        st = _State(c, AL, bL, AG, bG)
        box["st"] = st
        _run(st, c, AL, bL, AG, bG, n, m, q, T, end)
    except Exception:
        pass
    st = box["st"]
    try:
        if st is not None and st.best_x is not None:
            return {"x": [int(v) for v in st.best_x]}
    except Exception:
        pass
    return {"x": None}


def _run(st, c, AL, bL, AG, bG, n, m, q, T, end):
    rng = np.random.default_rng(20240917)
    if m == 0 and q == 0:
        st.offer((c > 0).astype(np.int64))
        return
    if q > 0 and np.any(AG.sum(axis=1) < bG):
        return  # infeasible
    st.offer(np.zeros(n, dtype=np.int64))
    st.offer((c > 0).astype(np.int64))

    movable = np.ones(n, dtype=bool)
    if m > 0:
        movable &= ~np.any(AL > bL[:, None], axis=0)
    if q > 0 and np.any((AG * movable[None, :]).sum(axis=1) < bG):
        return  # infeasible
    ub0 = movable.astype(np.float64)
    n_mov = int(movable.sum())
    mov_idx = np.nonzero(movable)[0]
    if n_mov == 0:
        return

    Gls = np.vstack([AL, -AG]).astype(np.int64)
    hls = np.concatenate([bL, -bG]).astype(np.int64)
    Gf = np.vstack([AL, AG]).astype(np.float64)
    rl = np.concatenate([np.full(m, -np.inf), bG.astype(np.float64)])
    ru = np.concatenate([bL.astype(np.float64), np.full(q, np.inf)])
    cf = c.astype(np.float64)
    total = max(1e-3, end - time.time())

    # ---------------------------------------------------------------- LP relaxation
    x_lp = None
    rc = None
    lp_done = False
    if _HC is not None:
        try:
            lpe = _HighsEngine(st, Gf, rl, ru, n, integer=False)
            lpe.h.changeColsBounds(n, lpe.idx, np.zeros(n), ub0)
            code, xl, rcv = lpe.solve_lp(max(0.05, 0.08 * total))
            if code == _INF:
                return
            if code == _OPT and xl is not None:
                x_lp = np.clip(xl, 0.0, 1.0)
                if rcv is not None and np.all(np.isfinite(rcv)):
                    rc = rcv
                lp_done = True
            del lpe
        except Exception:
            lp_done = False
    if not lp_done and _linprog is not None:
        try:
            Gs = np.vstack([AL, -AG]).astype(np.float64)
            r = _linprog(-cf, A_ub=Gs, b_ub=hls.astype(np.float64), bounds=list(zip([0.0] * n, ub0.tolist())),
                         method="highs", options={"time_limit": max(0.05, 0.08 * total)})
            if r is not None and r.status == 2:
                return
            if r is not None and r.x is not None:
                x_lp = np.clip(np.asarray(r.x, dtype=np.float64), 0.0, 1.0)
                try:
                    y = np.maximum(-np.asarray(r.ineqlin.marginals, dtype=np.float64), 0.0)
                    rc = cf - Gs.T @ y
                except Exception:
                    rc = None
        except Exception:
            pass
    if rc is None:
        rc = cf.copy()
    if x_lp is not None:
        st.offer((x_lp > 0.5).astype(np.int64))
        st.offer((x_lp > 1e-6).astype(np.int64))
        st.offer((x_lp > 1 - 1e-6).astype(np.int64))
    arc = np.abs(rc)
    rc_scale = float(arc[arc > 0].mean()) if np.any(arc > 0) else 1.0
    rcs = arc / rc_scale
    pert_w = np.exp(-np.minimum(rcs, 30.0)) + 0.05
    if x_lp is not None:
        pert_w = pert_w + (np.minimum(x_lp, 1.0 - x_lp) > 1e-6) * 1.0

    ls = _LS(st, Gls, hls, movable, pert_w, rng)

    # ---------------------------------------------------------------- quick incumbent by LS
    if x_lp is not None:
        x0 = (x_lp > 0.5).astype(np.int64)
    elif st.best_x is not None:
        x0 = st.best_x.copy()
    else:
        x0 = np.zeros(n, dtype=np.int64)
    x0[~movable] = 0
    ls.run(x0, min(end, time.time() + 0.05 * total))

    # ---------------------------------------------------------------- MIP engine
    eng = None
    if _HC is not None:
        try:
            eng = _HighsEngine(st, Gf, rl, ru, n, integer=True)
        except Exception:
            eng = None
    if eng is None and _milp is not None:
        eng = _ScipyEngine(st, Gf, rl, ru, n)
    if eng is None:
        while time.time() < end:
            ls.run(st.best_x if st.best_x is not None else x0, end)
        return
    persistent = isinstance(eng, _HighsEngine)
    zeros_n = np.zeros(n)
    global_ub = None

    def upd_ub(db):
        nonlocal global_ub
        if db is None:
            return
        u = int(math.floor(db + 1e-6))
        global_ub = u if global_ub is None else min(global_ub, u)

    def proven():
        return global_ub is not None and st.best_obj is not None and st.best_obj >= global_ub

    # ---------------------------------------------------------------- full MIP (stagnation-interrupted)
    rem = end - time.time()
    if rem > 0.05:
        cap = 0.60 * rem if persistent else 0.38 * rem
        code, db = eng.run(zeros_n, ub0, st.best_x, cap, stall=max(0.3, 0.15 * T),
                           min_time=0.3 * rem)
        if code == _OPT:
            return
        if code == _INF:
            return  # no feasible solution at all (a warm start would have been feasible)
        if code != _ERR:
            upd_ub(db)
    if proven():
        return

    while st.best_x is None and time.time() < end - 0.02:
        ls.run(x0, min(end, time.time() + max(0.05, 0.1 * total)))
        if st.best_x is not None:
            break
        code, db = eng.run(zeros_n, ub0, None, end - time.time())
        if code in (_OPT, _INF):
            return
    if st.best_x is None:
        return

    # ---------------------------------------------------------------- LNS
    strategies = ["core_lo", "core_mid", "core_hi", "rins", "elite", "random", "row", "lb"]
    sw = {s_: 1.0 for s_ in strategies}
    if x_lp is None:
        sw["rins"] = 0.0
    k_min = min(n_mov, max(8, int(0.05 * n_mov)))
    k0 = float(min(n_mov, max(k_min, int(0.15 * n_mov))))
    fam_of = {"core_lo": "core", "core_mid": "core", "core_hi": "core", "rins": "rins",
              "elite": "elite", "random": "random", "row": "row", "lb": "lb"}
    kfs = {"core": k0, "rins": k0, "elite": k0, "random": k0, "row": k0}
    lb_k = float(max(4, min(n_mov, 10 + n_mov // 50)))
    sub_tl = max(0.1, 0.06 * T)
    last_full = time.time()
    Gabs_rowsum = np.abs(Gf).sum(axis=1) + 1e-9
    while time.time() < end - 0.02:
        if proven():
            return
        now = time.time()
        rem = end - now
        # long stagnation: re-run the full MIP warm-started from the (better) incumbent
        if persistent and now - st.last_improve_time > max(0.25 * T, 6 * sub_tl) \
                and now - last_full > max(0.25 * T, 6 * sub_tl) and rem > 0.2 * T:
            last_full = now
            code, db = eng.run(zeros_n, ub0, st.best_x, 0.5 * rem, stall=max(0.3, 0.1 * T),
                               min_time=0.08 * T)
            if code == _OPT:
                return
            if code != _ERR:
                upd_ub(db)
            continue
        bx = st.best_x
        tot = sum(sw.values())
        u = rng.random() * tot
        strat = strategies[-1]
        for s_ in strategies:
            if u <= sw[s_]:
                strat = s_
                break
            u -= sw[s_]
        fam = fam_of[strat]
        kk = int(min(n_mov, max(k_min, round(kfs.get(fam, k0)))))
        noise = rng.random(n)
        lb_row = None
        if strat == "lb":
            free = mov_idx
            coef = np.where(movable, 1.0 - 2.0 * bx, 0.0)
            kl = int(round(lb_k))
            lb_row = (coef, -np.inf if not persistent else -eng.inf, float(kl - int(bx[movable].sum())))
        else:
            if kk >= n_mov:
                pr = None
            elif strat.startswith("core"):
                sig = {"core_lo": 0.3, "core_mid": 1.0, "core_hi": 3.0}[strat]
                if len(st.elite) >= 2:
                    E = np.array([e[2] for e in st.elite], dtype=np.float64)
                    dis = np.abs(E - bx[None, :]).mean(axis=0)
                else:
                    dis = np.zeros(n)
                lpd = np.abs(x_lp - bx) if x_lp is not None else np.zeros(n)
                pr = -np.minimum(rcs, 50.0) + 2.0 * dis + lpd + sig * noise
            elif strat == "rins" and x_lp is not None:
                pr = np.abs(x_lp - bx) + 0.6 * noise
            elif strat == "elite" and len(st.elite) >= 2:
                E = np.array([e[2] for e in st.elite], dtype=np.float64)
                p = E.mean(axis=0)
                pr = p * (1 - p) + (E[0] != bx) * 0.25 + 0.3 * noise
            elif strat == "row":
                L = Gf @ bx.astype(np.float64)
                slack = np.minimum(ru - L, L - rl) / Gabs_rowsum
                slack = np.where(np.isfinite(slack), slack, np.inf)
                tight = np.argsort(slack)[:max(1, min(Gf.shape[0], 3))]
                i = int(tight[int(rng.integers(tight.size))])
                row = np.abs(Gf[i])
                pr = row / (row.max() + 1e-9) + noise
            else:
                pr = noise
            if pr is None:
                free = mov_idx
            else:
                pm = pr[mov_idx]
                free = mov_idx[np.argpartition(-pm, kk - 1)[:kk]]
        lb = bx.astype(np.float64).copy()
        ub = bx.astype(np.float64).copy()
        lb[free] = 0.0
        ub[free] = ub0[free]
        tl = min(end - time.time() - 0.01, sub_tl)
        if tl < 0.03:
            break
        ts = time.time()
        before = st.best_obj
        code, db = eng.run(lb, ub, bx, tl, lb_row=lb_row)
        used = time.time() - ts
        improved = st.best_obj is not None and before is not None and st.best_obj > before
        full_space = (lb_row is None and free.size >= n_mov)
        if code == _OPT or code == _INF:
            if full_space:
                return  # whole problem solved
            if strat == "lb":
                if not improved:
                    lb_k = min(float(n_mov), lb_k * 1.3 + 1.0)
            elif not improved:
                kfs[fam] = min(float(n_mov), kfs[fam] * 1.2 + 1.0)
        elif code == _LIM:
            if strat == "lb":
                if not improved:
                    lb_k = max(3.0, lb_k * 0.75)
            elif not improved:
                kfs[fam] = max(float(k_min), kfs[fam] * 0.75)
        if full_space and code != _ERR:
            upd_ub(db)
        if improved:
            sw[strat] = min(20.0, sw[strat] + 1.0)
        for s_ in strategies:
            if sw[s_] > 0:
                sw[s_] = 0.95 * sw[s_] + 0.05
        if proven():
            return
        burst = max(0.02, 0.35 * used) if improved else max(0.01, 0.12 * used)
        ls_end = min(end, time.time() + burst)
        if time.time() < ls_end:
            ls.run(st.best_x, ls_end)
