"""
MDMKP solver: cardinality-stratified LP bounds + reduced-cost kernel search (HiGHS sub-MIPs).

diagnosis:
1. The champion loses on instances 1/3/6/8 (gaps 122/113/510/224 to the best rival); these are the
   cases where its 38%-of-T full-model MIP does not close and the few remaining 0.06T LNS calls repeat.
2. Its LNS frees 15% of variables picked mostly by the same full-LP reduced costs, so neighbourhoods
   overlap heavily, and nothing bounds the search: time is not concentrated where improvement is possible.
3. Its dense numpy tabu recomputes R x n matrices per move (few thousand moves in 10 s), so with
   mixed-sign costs / tight demands (inst. 6) it brings little diversity to the MIP phase.

New design (not LS+full MIP+LNS):
* Every feasible x lies on a hyperplane sum(x)=k.  For each k an LP (with the equality) gives a bound
  UB_k (concave in k) and reduced costs rc_k.  Hyperplanes with UB_k < z*+1 are closed.
* Any improving solution on hyperplane k satisfies  sum_j |rc_kj| [x_j != xbar_j] <= UB_k - z* - 1,
  so variables with |rc| above that gap are provably fixed; the constraint itself is added to sub-MIPs.
* A scheduler picks the hyperplane (or the unconstrained model) with the largest
  (UB - z*) / (1 + spent/unit); it solves a core sub-MIP (smallest |rc| + randomised fringe),
  grows the core after optimal solves, shrinks it / extends the slice after timeouts, and closes the
  arm once the core covers every non-fixable variable.  All arms closed => proven optimal.
* Each new incumbent is polished by a vectorised add/drop/swap descent.
"""
import time
import math
import random

import numpy as np

try:  # import at load time so it does not eat into the solve budget
    from scipy.optimize import linprog as _linprog, milp as _milp  # noqa: F401
except Exception:
    pass


class _Solver:
    def __init__(self, c, G, h, end, T):
        self.c = c                    # int64, reduced (movable) space
        self.G = G                    # int64 R x n
        self.h = h                    # int64 R
        self.cf = c.astype(np.float64)
        self.Gf = G.astype(np.float64)
        self.hf = h.astype(np.float64)
        self.n = c.shape[0]
        self.R = G.shape[0]
        self.end = end
        self.T = T
        self.best_x = None
        self.best_obj = None
        self.kbest = {}
        self.rng = random.Random(12345)

    # ---------------------------------------------------------------- incumbents
    def feasible(self, x):
        if self.R == 0:
            return True
        return bool(np.all(self.G @ x <= self.h))

    def offer(self, x, polish=True):
        try:
            x = (np.asarray(x).reshape(-1) != 0).astype(np.int64)
            if x.shape[0] != self.n or not self.feasible(x):
                return False
            if polish:
                x = self.polish(x)
            obj = int(self.c @ x)
            k = int(x.sum())
            kb = self.kbest.get(k)
            if kb is None or obj > kb[0]:
                self.kbest[k] = (obj, x.copy())
            if self.best_obj is None or obj > self.best_obj:
                self.best_obj = obj
                self.best_x = x.copy()
                return True
        except Exception:
            pass
        return False

    def polish(self, x):
        c, G, h = self.c, self.G, self.h
        n, R = self.n, self.R
        x = x.copy()
        L = G @ x if R > 0 else np.zeros(0, dtype=np.int64)
        cap = 400
        while time.time() < self.end:
            slack = h - L
            ones = np.flatnonzero(x == 1)
            zeros = np.flatnonzero(x == 0)
            best_gain = 0
            move = None
            if zeros.size:
                if R > 0:
                    ok = np.all(G[:, zeros] <= slack[:, None], axis=0)
                else:
                    ok = np.ones(zeros.size, dtype=bool)
                g = np.where(ok, c[zeros], 0)
                j = int(np.argmax(g))
                if g[j] > best_gain:
                    best_gain = int(g[j])
                    move = (int(zeros[j]), -1)
            if ones.size:
                if R > 0:
                    ok = np.all(-G[:, ones] <= slack[:, None], axis=0)
                else:
                    ok = np.ones(ones.size, dtype=bool)
                g = np.where(ok, -c[ones], 0)
                j = int(np.argmax(g))
                if g[j] > best_gain:
                    best_gain = int(g[j])
                    move = (-1, int(ones[j]))
            if ones.size and zeros.size and R > 0:
                O = ones[np.argsort(c[ones], kind="stable")][:cap]
                Z = zeros[np.argsort(-c[zeros], kind="stable")][:cap]
                gain = c[Z][None, :] - c[O][:, None]
                mask = gain > best_gain
                if mask.any():
                    GO = G[:, O]
                    GZ = G[:, Z]
                    for r in range(R):
                        mask &= (GZ[r][None, :] - GO[r][:, None]) <= slack[r]
                        if not mask.any():
                            break
                    if mask.any():
                        gm = np.where(mask, gain, -1)
                        idx = int(np.argmax(gm))
                        oi, zi = divmod(idx, Z.size)
                        if gm[oi, zi] > best_gain:
                            best_gain = int(gm[oi, zi])
                            move = (int(Z[zi]), int(O[oi]))
            if move is None:
                break
            a, d = move
            if a >= 0:
                x[a] = 1
                if R > 0:
                    L = L + G[:, a]
            if d >= 0:
                x[d] = 0
                if R > 0:
                    L = L - G[:, d]
        return x

    # ---------------------------------------------------------------- LP
    def lp(self, k):
        from scipy.optimize import linprog
        n, R = self.n, self.R
        rem = self.end - time.time()
        if rem < 0.02:
            return None
        kw = {}
        if R > 0:
            kw["A_ub"] = self.Gf
            kw["b_ub"] = self.hf
        if k is not None:
            kw["A_eq"] = np.ones((1, n))
            kw["b_eq"] = np.array([float(k)])
        try:
            res = linprog(-self.cf, bounds=(0.0, 1.0), method="highs",
                          options={"time_limit": float(max(0.02, rem))}, **kw)
        except Exception:
            return None
        if res.status == 2:
            return "infeasible"
        if res.status != 0 or res.x is None:
            return None
        xl = np.clip(np.asarray(res.x, dtype=np.float64), 0.0, 1.0)
        ub = float(-res.fun)
        rc = None
        try:
            d = np.asarray(res.lower.marginals, dtype=np.float64) + \
                np.asarray(res.upper.marginals, dtype=np.float64)
            rc = -d
        except Exception:
            rc = None
        if rc is None or rc.shape[0] != n or not np.all(np.isfinite(rc)):
            try:
                rc = self.cf.copy()
                if R > 0:
                    y = np.maximum(-np.asarray(res.ineqlin.marginals, dtype=np.float64), 0.0)
                    rc -= self.Gf.T @ y
                if k is not None:
                    rc -= float(-np.asarray(res.eqlin.marginals, dtype=np.float64)[0])
            except Exception:
                rc = np.zeros(n)
        rc = np.where(np.abs(rc) < 1e-9, 0.0, rc)
        return ub, xl, rc

    # ---------------------------------------------------------------- sub-MIP
    def sub_mip(self, free, xfix, k, rc_coef, rc_rhs, tl):
        """free: index array; xfix: 0/1 vector for non-free vars. Returns (status, x_full or None)."""
        from scipy.optimize import milp, LinearConstraint, Bounds
        n, R = self.n, self.R
        fmask = np.zeros(n, dtype=bool)
        fmask[free] = True
        nf = int(fmask.sum())
        fixed_x = np.where(fmask, 0, xfix).astype(np.int64)
        if nf == 0:
            x = fixed_x
            ok = self.feasible(x) and (k is None or int(x.sum()) == k)
            return (0 if ok else 2), (x if ok else None)
        idx = np.flatnonzero(fmask)
        cons = []
        if R > 0:
            rhs = self.hf - (self.G @ fixed_x).astype(np.float64)
            cons.append(LinearConstraint(self.Gf[:, idx], -np.inf, rhs))
        if k is not None:
            kk = k - int(fixed_x.sum())
            if kk < 0 or kk > nf:
                return 2, None
            cons.append(LinearConstraint(np.ones((1, nf)), kk, kk))
        if rc_coef is not None:
            coef = rc_coef[idx]
            if np.any(coef != 0):
                cons.append(LinearConstraint(coef.reshape(1, -1), -np.inf, rc_rhs))
            elif rc_rhs < 0:
                return 2, None
        try:
            res = milp(-self.cf[idx], constraints=cons if cons else None,
                       integrality=np.ones(nf), bounds=Bounds(np.zeros(nf), np.ones(nf)),
                       options={"time_limit": float(max(0.01, tl)), "disp": False,
                                "mip_rel_gap": 0.0})
        except Exception:
            return 4, None
        x = None
        if res.x is not None:
            x = fixed_x.copy()
            x[idx] = (np.asarray(res.x) > 0.5).astype(np.int64)
        return int(res.status), x


class _Arm:
    def __init__(self, k):
        self.k = k
        self.computed = False
        self.closed = False
        self.ub = None
        self.parent_ub = None
        self.rc = None
        self.arc = None
        self.xbar = None
        self.order = None
        self.s = 0
        self.s_min = 0
        self.tl = 0.0
        self.spent = 0.0
        self.tries = 0


def solve(instance: dict, time_limit_s: float) -> dict:
    t0 = time.time()
    try:
        T = float(time_limit_s)
        if not (T > 0) or math.isnan(T) or math.isinf(T):
            T = 10.0
    except Exception:
        T = 10.0
    end = t0 + max(0.05, T * 0.92 - 0.05)

    try:
        c_full = np.array(instance.get("cost_vector", []), dtype=np.int64).reshape(-1)
        N = int(c_full.shape[0])
        if N == 0:
            return {"x": []}
        bl_raw = instance.get("b_leq") or []
        bg_raw = instance.get("b_geq") or []
        m = len(bl_raw)
        q = len(bg_raw)
        AL = np.array(instance.get("A_leq") or [], dtype=np.int64).reshape(m, N) if m > 0 \
            else np.zeros((0, N), dtype=np.int64)
        bL = np.array(bl_raw, dtype=np.int64).reshape(m) if m > 0 else np.zeros(0, dtype=np.int64)
        AG = np.array(instance.get("A_geq") or [], dtype=np.int64).reshape(q, N) if q > 0 \
            else np.zeros((0, N), dtype=np.int64)
        bG = np.array(bg_raw, dtype=np.int64).reshape(q) if q > 0 else np.zeros(0, dtype=np.int64)
    except Exception:
        return {"x": None}

    if m == 0 and q == 0:
        return {"x": [1 if v > 0 else 0 for v in c_full.tolist()]}

    movable = np.ones(N, dtype=bool)
    if m > 0:
        movable &= ~np.any(AL > bL[:, None], axis=0)
    if q > 0 and np.any((AG * movable[None, :]).sum(axis=1) < bG):
        return {"x": None}
    mov_idx = np.flatnonzero(movable)
    if mov_idx.size == 0:
        if q == 0 or np.all(bG <= 0):
            return {"x": [0] * N}
        return {"x": None}

    G_full = np.vstack([AL, -AG]).astype(np.int64)
    h = np.concatenate([bL, -bG]).astype(np.int64)
    G = G_full[:, mov_idx]
    c = c_full[mov_idx]
    S = _Solver(c, G, h, end, T)
    # drop rows that can never be violated
    if S.R > 0:
        keep = ~(np.maximum(G, 0).sum(axis=1) <= h)
        if not keep.all():
            S = _Solver(c, G[keep], h[keep], end, T)

    def result():
        if S.best_x is None:
            return {"x": None}
        x = [0] * N
        for p, j in enumerate(mov_idx.tolist()):
            if S.best_x[p]:
                x[j] = 1
        return {"x": x}

    if S.R == 0:
        S.offer((c > 0).astype(np.int64), polish=False)
        return result()

    try:
        out = _main(S, T, end, result)
    except Exception:
        out = result()
    if out.get("x", None) is None and S.best_x is None:
        # last resort (time ran out before any incumbent): short penalty-flip repair
        try:
            _repair(S, time.time() + max(0.02, min(0.2, 0.05 * T)))
        except Exception:
            pass
        if S.best_x is not None:
            out = result()
    return out


def _repair(S, stop):
    n, G, h, c = S.n, S.G, S.h, S.c
    for start in ((c > 0).astype(np.int64), np.zeros(n, dtype=np.int64), np.ones(n, dtype=np.int64)):
        y = start.copy()
        for _ in range(4 * n + 4):
            if time.time() > stop:
                return
            L = G @ y
            if not np.any(L > h):
                S.offer(y, polish=False)
                return
            d = 1 - 2 * y
            nv = np.maximum(L[:, None] + G * d[None, :] - h[:, None], 0).sum(axis=0)
            j = int(np.argmin(nv - 1e-6 * c * d))
            y[j] = 1 - y[j]


def _main(S, T, end, result):
    n = S.n
    R = S.R
    rng = S.rng
    try:
        from scipy.optimize import milp  # noqa: F401
        milp_ok = True
    except Exception:
        milp_ok = False

    S.offer(np.zeros(n, dtype=np.int64))
    S.offer((S.c > 0).astype(np.int64))

    unit = max(0.05, 0.03 * T)
    tl0 = max(0.05, 0.03 * T)

    def setup(arm, lpres):
        ub, xl, rc = lpres
        arm.computed = True
        arm.ub = ub
        arm.rc = rc
        arm.arc = np.abs(rc)
        frac = (xl > 1e-6) & (xl < 1 - 1e-6)
        xbar = np.where(rc > 0, 1, np.where(rc < 0, 0, (xl > 0.5).astype(np.int64))).astype(np.int64)
        xbar = np.where(frac, (xl > 0.5).astype(np.int64), xbar)
        arm.xbar = xbar
        key = arm.arc - 1e-3 * frac
        arm.order = np.argsort(key, kind="stable")
        nb = int(frac.sum())
        arm.s_min = min(n, max(10, nb + 4))
        arm.s = min(n, max(arm.s_min, nb + 10, int(1.5 * (R + 1)), 20))
        arm.tl = tl0
        # quick rounding incumbents
        S.offer((xl > 1 - 1e-6).astype(np.int64))
        S.offer((xl > 0.5).astype(np.int64))

    def ub_eff(u):
        return u + 1e-3 + 1e-9 * abs(u)

    def arm_closed_by_bound(arm):
        if S.best_obj is None:
            return False
        u = arm.ub if arm.computed else arm.parent_ub
        return u is not None and math.floor(ub_eff(u)) < S.best_obj + 1

    # ---------------------------------------------------------------- full LP
    full = _Arm(None)
    r0 = S.lp(None)
    if r0 == "infeasible":
        return {"x": None}
    if r0 is None:
        r0 = (float(np.maximum(S.c, 0).sum()), np.full(n, 0.5), np.zeros(n))
    setup(full, r0)
    if S.best_x is not None:
        S.offer(S.best_x)

    def attempt(arm):
        """One kernel sub-MIP on the arm. Returns True if the arm got closed."""
        now = time.time()
        rem = end - now
        if rem < 0.03:
            return False
        exact_n = n
        rc_coef = None
        rc_rhs = None
        if S.best_obj is not None:
            g = arm.ub - (S.best_obj + 1) + 0.01 + 1e-9 * abs(arm.ub)
            if g < 0:
                arm.closed = True
                return True
            exact_n = int(np.count_nonzero(arm.arc <= g))
            rc_coef = np.where(arm.xbar == 1, -arm.arc, arm.arc)
        exact = exact_n <= arm.s
        if exact:
            free = arm.order[:exact_n]
        else:
            s = arm.s
            r = max(2, s // 5)
            base = arm.order[:s - r]
            pool = arm.order[s - r:min(n, 3 * s + 10)]
            if pool.size > r:
                pick = np.array(rng.sample(range(pool.size), r), dtype=np.int64)
                extra = pool[pick]
            else:
                extra = pool
            free = np.concatenate([base, extra])
            kb = S.kbest.get(arm.k) if arm.k is not None else None
            src = kb[1] if kb is not None else (S.best_x if arm.k is None else None)
            if src is not None:
                diff = np.flatnonzero(src != arm.xbar)
                if diff.size <= max(4, s // 2):
                    free = np.union1d(free, diff)
        if rc_coef is not None:
            fm = np.zeros(n, dtype=bool)
            fm[free] = True
            rc_rhs = float(arm.ub - (S.best_obj + 1) + 0.01 + 1e-9 * abs(arm.ub)
                           - arm.arc[fm & (arm.xbar == 1)].sum())
        tl = min(arm.tl, rem - 0.02)
        ts = time.time()
        status, x = S.sub_mip(free, arm.xbar, arm.k, rc_coef, rc_rhs, tl)
        used = time.time() - ts
        arm.spent += used
        arm.tries += 1
        if x is not None:
            S.offer(x)
        if status in (0, 2):
            if exact:
                arm.closed = True
                return True
            arm.s = min(n, int(math.ceil(arm.s * 1.35)) + 2)
        elif status == 1:
            arm.s = max(arm.s_min, int(arm.s * 0.85))
            arm.tl = min(max(tl0, 0.3 * T), arm.tl * 1.4)
        else:
            arm.s = max(arm.s_min, int(arm.s * 0.85))
        return False

    # ---------------------------------------------------------------- first incumbent
    if milp_ok:
        full.tl = max(tl0, 0.06 * T)
        if attempt(full) and S.best_obj is not None:
            return result()
        full.tl = tl0
        if S.best_x is None:
            from scipy.optimize import milp, LinearConstraint, Bounds
            first = True
            while S.best_x is None and time.time() < end - 0.03:
                rem = end - time.time()
                tl = min(rem - 0.02, 0.3 * T) if first else rem - 0.02
                first = False
                try:
                    res = milp(-S.cf, constraints=[LinearConstraint(S.Gf, -np.inf, S.hf)],
                               integrality=np.ones(n), bounds=Bounds(np.zeros(n), np.ones(n)),
                               options={"time_limit": float(max(0.02, tl)), "disp": False,
                                        "mip_rel_gap": 0.0})
                    if res.x is not None:
                        S.offer(np.asarray(res.x) > 0.5)
                    if res.status == 2 and S.best_x is None:
                        return {"x": None}
                    if res.status == 0 and S.best_x is not None:
                        return result()
                except Exception:
                    break
            if S.best_x is None:
                return result()
    else:
        _fallback(S, end)
        return result()

    # ---------------------------------------------------------------- hyperplane arms
    arms = {}
    sx = float(r0[1].sum())
    kf = int(math.floor(sx + 1e-9))
    kc = int(math.ceil(sx - 1e-9))
    for k in sorted(set([min(n, max(0, kf)), min(n, max(0, kc))])):
        a = _Arm(k)
        a.parent_ub = full.ub
        arms[k] = a

    def compute(arm):
        res = S.lp(arm.k)
        if res is None:
            return False
        if res == "infeasible":
            arm.computed = True
            arm.closed = True
            return True
        setup(arm, res)
        if arm_closed_by_bound(arm):
            arm.closed = True
            return True
        ks = [kk for kk, aa in arms.items()]
        if arm.k == min(ks) and arm.k - 1 >= 0:
            b = _Arm(arm.k - 1)
            b.parent_ub = arm.ub
            arms[arm.k - 1] = b
        if arm.k == max(ks) and arm.k + 1 <= n:
            b = _Arm(arm.k + 1)
            b.parent_ub = arm.ub
            arms[arm.k + 1] = b
        return True

    lp_fail = 0
    while time.time() < end - 0.03:
        if full.closed:
            return result()
        zb = S.best_obj
        best_arm = None
        best_p = -1.0
        open_cnt = 0
        for arm in list(arms.values()) + [full]:
            if arm.closed:
                continue
            if arm_closed_by_bound(arm):
                arm.closed = True
                continue
            open_cnt += 1
            u = arm.ub if arm.computed else arm.parent_ub
            gap = max(1.0, u - zb)
            if not arm.computed:
                p = 4.0 * gap
            else:
                p = gap / (1.0 + arm.spent / unit)
            if p > best_p:
                best_p = p
                best_arm = arm
        k_open = sum(1 for a in arms.values() if not a.closed)
        if k_open == 0 and all(a.computed or a.closed for a in arms.values()):
            # every hyperplane closed => incumbent optimal
            return result()
        if best_arm is None:
            return result()
        if not best_arm.computed:
            ok = compute(best_arm)
            if not ok:
                lp_fail += 1
                if lp_fail > 3:
                    best_arm.computed = True
                    best_arm.ub = best_arm.parent_ub
                    best_arm.rc = np.zeros(n)
                    best_arm.arc = np.zeros(n)
                    best_arm.xbar = S.best_x.copy()
                    best_arm.order = np.array(rng.sample(range(n), n), dtype=np.int64)
                    best_arm.s_min = min(n, 10)
                    best_arm.s = min(n, max(20, R + 1))
                    best_arm.tl = tl0
            continue
        attempt(best_arm)
    return result()


def _fallback(S, end):
    """No MILP solver: randomized destroy/repair around the incumbent with polishing."""
    n, R = S.n, S.R
    G, h, c = S.G, S.h, S.c
    rng = np.random.default_rng(7)
    x = (S.best_x.copy() if S.best_x is not None else np.zeros(n, dtype=np.int64))
    while time.time() < end:
        y = x.copy()
        k = max(1, n // 20)
        idx = rng.choice(n, size=min(n, k), replace=False)
        y[idx] = 1 - y[idx]
        for _ in range(4 * n):
            L = G @ y
            v = np.maximum(L - h, 0)
            if not v.any():
                break
            d = 1 - 2 * y
            nv = np.maximum(L[:, None] + G * d[None, :] - h[:, None], 0).sum(axis=0)
            j = int(np.argmin(nv - 1e-6 * c * d))
            y[j] = 1 - y[j]
        if S.feasible(y):
            S.offer(y)
            if S.best_x is not None and rng.random() < 0.7:
                x = S.best_x.copy()
            else:
                x = y
