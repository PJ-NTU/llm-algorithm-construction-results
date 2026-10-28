"""Set Partitioning solver (anytime, wall-clock driven).

Pipeline:
  1. validation + exact presolve (duplicate/empty columns, singleton rows)
  2. fail-first exact-cover search (DLX style) -> first feasible solution,
     exhaustion proves infeasibility
  3. LP relaxation (HiGHS interior point) -> lower bound + reduced costs
  4. short exact MIP attempt on reduced-cost-filtered columns (easy instances end here)
  5. LNS: free rows of related incumbent columns, re-optimise the sub-problem
     exactly with HiGHS MIP; neighbourhood size adapts; runs until the deadline
"""
import math
import random
import time

import numpy as np

try:
    from scipy.optimize import milp, linprog, LinearConstraint, Bounds
    from scipy.sparse import csc_matrix
    _HAVE_SCIPY = True
except Exception:  # pragma: no cover
    _HAVE_SCIPY = False

_TRACE = None


def _tr(*a):
    if _TRACE is not None:
        _TRACE.append(a)


def _infeasible(msg=None):
    d = {"feasible": False, "x": None, "total_cost": None}
    if msg:
        d["error"] = msg
    return d


def _validate(instance):
    if not isinstance(instance, dict):
        return None, "instance must be a dict"
    try:
        n = instance.get("n")
        m = instance.get("m")
        costs = instance.get("costs")
        covers = instance.get("covers")
        if isinstance(n, bool) or isinstance(m, bool):
            return None, "n/m must be integers"
        if isinstance(n, float) and n.is_integer():
            n = int(n)
        if isinstance(m, float) and m.is_integer():
            m = int(m)
        if isinstance(n, np.integer):
            n = int(n)
        if isinstance(m, np.integer):
            m = int(m)
        if not isinstance(n, int) or not isinstance(m, int) or n < 0 or m < 0:
            return None, "n/m must be non-negative integers"
        if costs is None or covers is None:
            return None, "costs/covers missing"
        costs = list(costs)
        covers = list(covers)
        if len(costs) != n or len(covers) != n:
            return None, "length of costs/covers differs from n"
        cc = []
        for c in costs:
            if isinstance(c, bool):
                return None, "cost must be a number"
            v = float(c)
            if not math.isfinite(v) or v < 0:
                return None, "cost negative or not finite"
            cc.append(v)
        cv = []
        for lst in covers:
            s = []
            seen = set()
            for e in lst:
                if isinstance(e, bool):
                    return None, "task id must be integer"
                if isinstance(e, float):
                    if not e.is_integer():
                        return None, "task id must be integer"
                    e = int(e)
                if not isinstance(e, (int, np.integer)):
                    return None, "task id must be integer"
                e = int(e)
                if e < 0 or e >= m:
                    return None, "task id out of range"
                if e in seen:
                    return None, "duplicate task id in a cover"
                seen.add(e)
                s.append(e)
            cv.append(s)
        return (n, m, cc, cv), None
    except Exception as ex:
        return None, "invalid input: %r" % (ex,)


def _check(n, m, covers, x):
    cnt = [0] * m
    for j in range(n):
        if x[j]:
            for r in covers[j]:
                cnt[r] += 1
    return all(c == 1 for c in cnt)


def _make_result(n, m, costs, covers, cols):
    x = [0] * n
    for j in cols:
        x[j] = 1
    if not _check(n, m, covers, x):
        return None
    tc = math.fsum(costs[j] for j in range(n) if x[j])
    return {"feasible": True, "x": x, "total_cost": float(tc)}


def _presolve(n, m, costs, covers):
    """Exact reductions. Returns (status, fixed_cols, active_cols, active_rows)."""
    best_by_key = {}
    for j in range(n):
        if not covers[j]:
            continue
        key = tuple(sorted(covers[j]))
        b = best_by_key.get(key)
        if b is None or costs[j] < costs[b]:
            best_by_key[key] = j
    alive_col = [False] * n
    for j in best_by_key.values():
        alive_col[j] = True
    row_cols = [set() for _ in range(m)]
    for j in range(n):
        if alive_col[j]:
            for r in covers[j]:
                row_cols[r].add(j)
    alive_row = [True] * m
    fixed = []
    stack = [r for r in range(m) if len(row_cols[r]) <= 1]
    while stack:
        r = stack.pop()
        if not alive_row[r]:
            continue
        if len(row_cols[r]) == 0:
            return "infeasible", None, None, None
        if len(row_cols[r]) > 1:
            continue
        j = next(iter(row_cols[r]))
        fixed.append(j)
        rows_j = covers[j]
        for rr in rows_j:
            alive_row[rr] = False
        kill = set()
        for rr in rows_j:
            kill |= row_cols[rr]
        for k in kill:
            alive_col[k] = False
            for rr in covers[k]:
                row_cols[rr].discard(k)
                if alive_row[rr] and len(row_cols[rr]) <= 1:
                    stack.append(rr)
    act_rows = [r for r in range(m) if alive_row[r]]
    for r in act_rows:
        if not row_cols[r]:
            return "infeasible", None, None, None
    act_cols = [j for j in range(n) if alive_col[j]]
    return "ok", fixed, act_cols, act_rows


def _dlx_gen(act_rows, cols, covers, key, n, rcv=None, lim=None):
    """Fail-first exact cover search (DLX style).
    Yields None periodically (time checks), yields a list of columns when a cover
    is found; returns on exhaustion.  If rcv/lim are given, a branch is pruned when
    the accumulated rcv of chosen columns reaches lim[0] (valid bound pruning)."""
    row_cols = {r: [] for r in act_rows}
    for j in cols:
        for r in covers[j]:
            row_cols[r].append(j)
    for r in act_rows:
        row_cols[r].sort(key=key.__getitem__)
    dead = [0] * n
    count = {r: len(row_cols[r]) for r in act_rows}
    uncovered = set(act_rows)
    sel = []

    def select(j):
        rem = []
        for r in covers[j]:
            uncovered.discard(r)
            for k in row_cols[r]:
                if dead[k] == 0:
                    for rr in covers[k]:
                        count[rr] -= 1
                    rem.append(k)
                dead[k] += 1
        return rem

    def unselect(j, rem):
        for r in covers[j]:
            uncovered.add(r)
            for k in row_cols[r]:
                dead[k] -= 1
        for k in rem:
            for rr in covers[k]:
                count[rr] += 1

    def pick():
        best = None
        bc = 1 << 60
        for r in uncovered:
            c = count[r]
            if c < bc:
                bc = c
                best = r
                if c <= 1:
                    break
        return best, bc

    if not uncovered:
        yield []
        return
    r, bc = pick()
    if bc == 0:
        return
    stack = [[r, 0, None, None, 0.0]]
    nodes = 0
    prune = rcv is not None and lim is not None
    while stack:
        nodes += 1
        if (nodes & 255) == 0:
            yield None
        fr = stack[-1]
        if fr[2] is not None:
            unselect(fr[2], fr[3])
            sel.pop()
            fr[2] = None
        lst = row_cols[fr[0]]
        i = fr[1]
        base = fr[4]
        j = None
        L = len(lst)
        while i < L:
            cand = lst[i]
            i += 1
            if dead[cand]:
                continue
            if prune and base + rcv[cand] >= lim[0]:
                continue
            j = cand
            break
        fr[1] = i
        if j is None:
            stack.pop()
            continue
        rem = select(j)
        sel.append(j)
        fr[2] = j
        fr[3] = rem
        nb = base + (rcv[j] if prune else 0.0)
        if not uncovered:
            yield list(sel)
            continue
        r, bc = pick()
        if bc == 0:
            continue
        stack.append([r, 0, None, None, nb])


def _run_gen(gen, deadline):
    try:
        while True:
            v = next(gen)
            if v is not None:
                return "sol", v
            if time.time() >= deadline:
                return "timeout", None
    except StopIteration:
        return "exhausted", None


def _run_mip(rows, cols, costs, covers, tlim):
    """Exact MIP on rows x cols. Returns (status, selected cols or None, optimal)."""
    nc = len(cols)
    rpos = {r: i for i, r in enumerate(rows)}
    indptr = [0]
    indices = []
    for j in cols:
        for r in covers[j]:
            indices.append(rpos[r])
        indptr.append(len(indices))
    A = csc_matrix((np.ones(len(indices)), np.array(indices, dtype=np.int64),
                    np.array(indptr, dtype=np.int64)), shape=(len(rows), nc))
    c = np.array([costs[j] for j in cols], dtype=float)
    ones = np.ones(len(rows))
    res = milp(c, constraints=LinearConstraint(A, ones, ones),
               integrality=np.ones(nc), bounds=Bounds(0, 1),
               options={"time_limit": max(0.01, float(tlim)), "mip_rel_gap": 0.0,
                        "presolve": True, "disp": False})
    sel = None
    if res.x is not None:
        xs = np.round(res.x).astype(int)
        sel = [cols[k] for k in range(nc) if xs[k] == 1]
    return res.status, sel, (res.status == 0 and sel is not None)


def solve(instance: dict, time_limit_s: float) -> dict:
    t0 = time.time()
    try:
        T = float(time_limit_s)
        if not math.isfinite(T) or T <= 0:
            T = 1.0
    except Exception:
        T = 1.0
    deadline = t0 + T * 0.93 - 0.08
    try:
        parsed, err = _validate(instance)
        if parsed is None:
            return _infeasible(err)
        n, m, costs, covers = parsed
        if m == 0:
            return {"feasible": True, "x": [0] * n, "total_cost": 0.0}
        if n == 0:
            return _infeasible()
        return _solve_core(n, m, costs, covers, t0, T, deadline)
    except Exception:
        return _infeasible()


def _solve_core(n, m, costs, covers, t0, T, deadline):
    st, fixed, act_cols, act_rows = _presolve(n, m, costs, covers)
    if st == "infeasible":
        return _infeasible()
    if not act_rows:
        return _make_result(n, m, costs, covers, fixed) or _infeasible()
    R = len(act_rows)

    best = {"cols": None, "val": float("inf"), "ver": 0}

    def offer(cols):
        if cols is None:
            return False
        v = math.fsum(costs[j] for j in cols)
        if best["cols"] is not None and v >= best["val"] - 1e-9:
            return False
        if _make_result(n, m, costs, covers, fixed + list(cols)) is None:
            return False
        best["val"] = v
        best["cols"] = list(cols)
        best["ver"] += 1
        return True

    def finish():
        if best["cols"] is None:
            return _infeasible()
        return _make_result(n, m, costs, covers, fixed + best["cols"]) or _infeasible()

    def left():
        return deadline - time.time()

    # ---- Stage 1: fail-first exact cover search (cost-ratio order) ----
    ratio = [0.0] * n
    for j in act_cols:
        ratio[j] = costs[j] / len(covers[j])
    gen = None
    try:
        gen = _dlx_gen(act_rows, act_cols, covers, ratio, n)
        stt, sol = _run_gen(gen, min(deadline, time.time() + 0.1 * T))
        if stt == "exhausted":
            return _infeasible()
        if stt == "sol":
            offer(sol)
        _tr("dlx", time.time() - t0, stt, best["val"])
    except Exception:
        gen = None

    if not _HAVE_SCIPY:
        if best["cols"] is None and gen is not None:
            try:
                stt, sol = _run_gen(gen, deadline)
                if stt == "sol":
                    offer(sol)
            except Exception:
                pass
        return finish()

    # ---- Stage 2: LP relaxation (interior point) -> bound + reduced costs ----
    lp = {"ok": False, "rc": None, "ysum": 0.0, "neg": 0.0}
    try:
        if left() > 0.1:
            rpos = {r: i for i, r in enumerate(act_rows)}
            indptr = [0]
            indices = []
            for j in act_cols:
                for r in covers[j]:
                    indices.append(rpos[r])
                indptr.append(len(indices))
            A = csc_matrix((np.ones(len(indices)), np.array(indices, dtype=np.int64),
                            np.array(indptr, dtype=np.int64)), shape=(R, len(act_cols)))
            cvec = np.array([costs[j] for j in act_cols], dtype=float)
            res = linprog(cvec, A_eq=A, b_eq=np.ones(R), bounds=(0, 1), method="highs-ipm",
                          options={"time_limit": max(0.05, min(0.25 * T, 0.5 * left()))})
            _tr("lp", time.time() - t0, res.status, res.fun)
            if res.status == 2 and best["cols"] is None:
                return _infeasible()
            if res.status == 0 and res.x is not None and res.eqlin is not None:
                y = np.asarray(res.eqlin.marginals, dtype=float)
                rcv = cvec - A.T.dot(y)
                if np.all(np.isfinite(rcv)):
                    rc = [0.0] * n
                    rcl = rcv.tolist()
                    for k, j in enumerate(act_cols):
                        rc[j] = rcl[k] if rcl[k] > 0.0 else 0.0
                    lp["rc"] = rc
                    lp["neg"] = float(np.minimum(rcv, 0.0).sum())
                    lp["ysum"] = float(y.sum())
                    lp["ok"] = True
    except Exception:
        lp["ok"] = False

    def rc_filter():
        if not lp["ok"] or best["cols"] is None:
            return list(act_cols)
        lim = best["val"] + 1e-6 + 1e-9 * abs(best["val"]) - lp["ysum"] - lp["neg"]
        rc = lp["rc"]
        keep = [j for j in act_cols if rc[j] <= lim]
        ks = set(keep)
        for j in best["cols"]:
            if j not in ks:
                keep.append(j)
        return keep

    rdlx = _RestartDLX(n, act_rows, act_cols, covers, costs, ratio, lp, best, offer, rc_filter, T)

    # still no feasible solution: reduced-cost ordered search, then resume first search
    if best["cols"] is None:
        try:
            if lp["ok"]:
                rdlx.first(min(deadline, time.time() + 0.5 * max(0.0, left())))
            if best["cols"] is None and gen is not None:
                stt, sol = _run_gen(gen, min(deadline, time.time() + 0.5 * max(0.0, left())))
                if stt == "exhausted":
                    return _infeasible()
                if stt == "sol":
                    offer(sol)
            _tr("first2", time.time() - t0, best["val"])
        except Exception:
            pass

    # ---- Stage 3: reduced-cost restarts, short exact MIP attempt ----
    try:
        if best["cols"] is not None:
            rdlx.run(min(deadline, time.time() + 0.15 * T))
            _tr("rdlx", time.time() - t0, best["val"], rdlx.k)
            if rdlx.proven:
                return finish()
    except Exception:
        pass
    try:
        if left() > 0.05:
            cand = rc_filter()
            tl = left() if best["cols"] is None else min(left(), 0.15 * T)
            stt, sel, opt = _run_mip(act_rows, cand, costs, covers, tl)
            if stt == 2 and len(cand) == len(act_cols) and best["cols"] is None:
                return _infeasible()
            offer(sel)
            _tr("mip", time.time() - t0, len(cand), stt, best["val"])
            if opt:
                return finish()
    except Exception:
        pass

    if best["cols"] is None:
        if gen is not None and left() > 0.02:
            try:
                stt, sol = _run_gen(gen, deadline)
                if stt == "sol":
                    offer(sol)
            except Exception:
                pass
        return finish()

    # ---- Stage 4: adaptive time slices between restart-DLX and LNS ----
    lns = _LNS(n, m, costs, covers, act_rows, best, offer, rc_filter, lp, T)
    ops = [rdlx, lns]
    rate = [1.0, 1.0]
    rng = random.Random(7 + R)
    slice_len = max(0.05, 0.06 * T)
    turn = 0
    while left() > 0.03:
        turn += 1
        if turn <= 2:
            oi = turn - 1
        elif rng.random() < 0.2:
            oi = rng.randrange(2)
        else:
            oi = 0 if rate[0] >= rate[1] else 1
        before = best["val"]
        ts = time.time()
        try:
            ops[oi].run(min(deadline, ts + slice_len))
        except Exception as ex:
            _tr("op-exc", oi, repr(ex))
            rate[oi] *= 0.1
        el = max(1e-3, time.time() - ts)
        gain = (before - best["val"]) / el
        rate[oi] = 0.5 * rate[oi] + 0.5 * (gain + 1e-9)
        if getattr(ops[oi], "proven", False):
            break
    _tr("end", time.time() - t0, best["val"], rdlx.k, lns.it, rate)
    return finish()


class _RestartDLX:
    """Randomised restarts of the fail-first exact-cover search, ordered by LP reduced
    costs (+ noise).  Every other restart uses valid bound pruning (UB - LB gap);
    if such a pruned search is exhausted, the incumbent is proven optimal."""

    def __init__(self, n, act_rows, act_cols, covers, costs, ratio, lp, best, offer, rc_filter, T):
        self.n = n
        self.act_rows = act_rows
        self.act_cols = act_cols
        self.covers = covers
        self.costs = costs
        self.ratio = ratio
        self.lp = lp
        self.best = best
        self.offer = offer
        self.rc_filter = rc_filter
        self.T = T
        self.rng = random.Random(12345 + len(act_rows))
        self.k = 0
        self.avg = None
        self.proven = False

    def _base(self):
        return self.lp["rc"] if self.lp["ok"] else self.ratio

    def first(self, until):
        base = self._base()
        gen = _dlx_gen(self.act_rows, self.act_cols, self.covers, base, self.n)
        stt, sol = _run_gen(gen, until)
        if stt == "sol":
            self.offer(sol)

    def run(self, until):
        while time.time() < until - 1e-3 and not self.proven:
            base = self._base()
            k = self.k
            self.k += 1
            rng = self.rng
            noise = 0.0 if k == 0 else (0.25 + 0.5 * rng.random())
            if noise > 0:
                key = [0.0] * self.n
                for j in self.act_cols:
                    b = base[j]
                    key[j] = b + noise * rng.random() * (1.0 + b)
            else:
                key = base
            use_prune = self.lp["ok"] and self.best["cols"] is not None and (k % 2 == 1)
            if use_prune:
                cols = self.rc_filter()
                lim = [self.best["val"] - 1e-9 - self.lp["ysum"] - self.lp["neg"]]
                gen = _dlx_gen(self.act_rows, cols, self.covers, key, self.n, self.lp["rc"], lim)
            else:
                cols = self.rc_filter()
                gen = _dlx_gen(self.act_rows, cols, self.covers, key, self.n)
            cap = max(0.02 * self.T, 5.0 * self.avg) if self.avg is not None else 0.1 * self.T
            ts = time.time()
            stt, sol = _run_gen(gen, min(until, ts + cap))
            el = time.time() - ts
            if stt == "sol":
                self.avg = el if self.avg is None else 0.8 * self.avg + 0.2 * el
                self.offer(sol)
            elif stt == "exhausted" and use_prune:
                self.proven = True


class _LNS:
    """Destroy rows of related incumbent columns, re-optimise exactly with HiGHS."""

    def __init__(self, n, m, costs, covers, act_rows, best, offer, rc_filter, lp, T):
        self.n, self.m, self.costs, self.covers = n, m, costs, covers
        self.act_rows = act_rows
        self.R = len(act_rows)
        self.best, self.offer, self.rc_filter, self.lp, self.T = best, offer, rc_filter, lp, T
        self.rng = random.Random(20240915 + self.R)
        self.K = min(self.R, max(6, int(round(min(40.0, 0.25 * self.R)))))
        self.sub_tl = max(0.05, 0.02 * T)
        self.stamp = [0] * n
        self.tick = 0
        self.inF = [False] * m
        self.owner = [-1] * m
        self.it = 0
        self.filter_val = None
        self.row_cols = None
        self.full_tried_at = None
        self.proven = False

    def _refresh(self):
        if self.filter_val is None or self.best["val"] < self.filter_val - 1e-9:
            cols = self.rc_filter()
            rcx = [[] for _ in range(self.m)]
            for j in cols:
                for r in self.covers[j]:
                    rcx[r].append(j)
            self.row_cols = rcx
            self.filter_val = self.best["val"]
            return cols
        return None

    def _set_owner(self):
        covers = self.covers
        owner = self.owner
        for j in self.best["cols"]:
            for r in covers[j]:
                owner[r] = j

    def run(self, until):
        covers, costs = self.covers, self.costs
        rng = self.rng
        inF = self.inF
        owner = self.owner
        self._set_owner()
        lp = self.lp
        while time.time() < until - 0.01 and not self.proven:
            self.it += 1
            cols = self._refresh()
            if cols is not None and lp["ok"] and len(cols) <= 4 * self.R and self.full_tried_at != len(cols):
                self.full_tried_at = len(cols)
                stt, sel, opt = _run_mip(self.act_rows, cols, costs, covers, max(0.01, until - time.time()))
                self.offer(sel)
                self._set_owner()
                if opt:
                    self.proven = True
                continue
            row_cols = self.row_cols
            inc = self.best["cols"]
            rc = lp["rc"]
            if lp["ok"] and rng.random() < 0.5:
                a = inc[rng.randrange(len(inc))]
                b = inc[rng.randrange(len(inc))]
                seed = a if rc[a] >= rc[b] else b
            else:
                seed = inc[rng.randrange(len(inc))]
            Flist = list(covers[seed])
            for r in Flist:
                inF[r] = True
            target = self.K
            tries = 0
            while len(Flist) < target and tries < 6 * target:
                tries += 1
                r = Flist[rng.randrange(len(Flist))]
                lst = row_cols[r]
                if not lst:
                    continue
                k = lst[rng.randrange(len(lst))]
                if lp["ok"]:
                    k2 = lst[rng.randrange(len(lst))]
                    if rc[k2] < rc[k]:
                        k = k2
                for rr in covers[k]:
                    if not inF[rr]:
                        o = owner[rr]
                        for r3 in covers[o]:
                            if not inF[r3]:
                                inF[r3] = True
                                Flist.append(r3)
            self.tick += 1
            tick = self.tick
            stamp = self.stamp
            subcols = []
            for r in Flist:
                for j in row_cols[r]:
                    if stamp[j] != tick:
                        stamp[j] = tick
                        ok = True
                        for rr in covers[j]:
                            if not inF[rr]:
                                ok = False
                                break
                        if ok:
                            subcols.append(j)
            in_inc = [j for j in inc if inF[covers[j][0]]]
            keep_inc = [j for j in inc if not inF[covers[j][0]]]
            cur = math.fsum(costs[j] for j in in_inc)
            sset = set(subcols)
            for j in in_inc:
                if j not in sset:
                    subcols.append(j)
            whole = len(Flist) >= self.R
            remaining = until - time.time()
            tl = remaining if whole else min(remaining, self.sub_tl)
            ts = time.time()
            opt = True
            if tl > 0.01 and len(subcols) > len(in_inc):
                stt, sel, opt = _run_mip(Flist, subcols, costs, covers, tl)
                if sel is not None:
                    v = math.fsum(costs[j] for j in sel)
                    if v < cur - 1e-9:
                        if self.offer(keep_inc + sel):
                            self._set_owner()
                    elif v <= cur + 1e-9 and set(sel) != set(in_inc):
                        cand = keep_inc + sel
                        if True:
                            cnt_ok = True
                            seen = set()
                            for j in sel:
                                for rr in covers[j]:
                                    if rr in seen or not inF[rr]:
                                        cnt_ok = False
                                    seen.add(rr)
                            if cnt_ok and len(seen) == len(Flist):
                                self.best["cols"] = cand
                                self._set_owner()
            elif tl <= 0.01:
                opt = False
            el = time.time() - ts
            for r in Flist:
                inF[r] = False
            if whole and opt:
                self.proven = True
                break
            if opt and el < 0.3 * self.sub_tl:
                self.K = min(self.R, int(self.K * 1.15) + 1)
            elif not opt:
                self.K = max(4, int(self.K * 0.8))
