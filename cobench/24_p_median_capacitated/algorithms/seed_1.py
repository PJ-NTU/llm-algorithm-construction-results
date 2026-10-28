import time
import math
import random
import numpy as np

try:
    from scipy.optimize import linear_sum_assignment as _lsa
except Exception:  # pragma: no cover
    _lsa = None
try:
    from scipy.optimize import milp as _milp, LinearConstraint as _LC, Bounds as _Bounds
    from scipy.sparse import csr_matrix as _csr
except Exception:  # pragma: no cover
    _milp = None
    _csr = None

_INF = float("inf")
_EPS = 1e-9


def _infeasible():
    return {"status": "infeasible", "medians": [], "assignments": []}


def solve(instance: dict, time_limit_s: float) -> dict:
    t0 = time.time()
    holder = {"sol": None}
    try:
        res = _solve(instance, time_limit_s, t0, holder)
        if isinstance(res, dict):
            return res
    except Exception:
        pass
    if holder.get("sol") is not None:
        return holder["sol"]
    try:
        r = _fallback(instance)
        if r is not None:
            return r
    except Exception:
        pass
    return _infeasible()


def _fallback(inst):
    n = int(inst["n"]); p = int(inst["p"]); cap = int(inst["capacity"])
    dem = [int(x) for x in inst["demands"]]
    if n < 1 or p < 1 or p > n or max(dem) > cap or sum(dem) > p * cap:
        return _infeasible()
    order = sorted(range(n), key=lambda i: -dem[i])
    loads = [0] * p
    lab = [0] * n
    for i in order:
        bk = -1
        for k in range(p):
            if loads[k] + dem[i] <= cap and (bk < 0 or loads[k] > loads[bk]):
                bk = k
        if bk < 0:
            return None
        loads[bk] += dem[i]
        lab[i] = bk
    return {"status": "feasible", "medians": list(range(p)), "assignments": [lab[i] for i in range(n)]}


def _solve(inst, tl, t0, holder):
    n = int(inst["n"]); p = int(inst["p"]); cap = int(inst["capacity"])
    dem = [int(x) for x in inst["demands"]]
    if n < 1 or p < 1 or p > n or len(dem) != n:
        return _infeasible()
    try:
        tl = float(tl)
        if not (tl > 0):
            tl = 1.0
    except Exception:
        tl = 10.0
    deadline = t0 + max(0.03, min(tl * 0.93, tl - 0.25))
    if max(dem) > cap or sum(dem) > p * cap:
        return _infeasible()

    D = np.asarray(inst["distance_matrix"], dtype=np.float64).reshape(n, n)
    d = np.asarray(dem, dtype=np.int64)
    ar = np.arange(n)
    rng = random.Random(12345)
    nrng = np.random.default_rng(12345)

    def tleft():
        return deadline - time.time()

    # ------------------------------------------------------------------ output
    best = {"cost": _INF, "S": None, "a": None}

    def record(S, a):
        S = np.asarray(S); a = np.asarray(a)
        if len(set(int(x) for x in S)) != p:
            return False
        load = np.bincount(a, weights=d, minlength=p)
        if load.max() > cap + 1e-9:
            return False
        cost = float(D[ar, S[a]].sum())
        if cost < best["cost"] - _EPS:
            best["cost"] = cost; best["S"] = S.copy(); best["a"] = a.copy()
            assign = [int(S[a[i]]) for i in range(n)]
            holder["sol"] = {"status": "feasible", "medians": sorted(int(x) for x in S),
                             "assignments": assign}
            return True
        return False

    # --------------------------------------------------------- assignment ops
    def assign_subset(Ds, a, load, U, mode=0):
        """greedy regret assignment of customers U; failures -> max residual."""
        if len(U) == 0:
            return True
        sub = Ds[U]
        if p >= 2:
            part = np.partition(sub, 1, axis=1)
            reg = part[:, 1] - part[:, 0]
        else:
            reg = np.zeros(len(U))
        dU = d[U]
        if mode == 0:
            order = np.lexsort((-dU, -reg))
        elif mode == 1:
            order = np.lexsort((-reg, -dU))
        else:
            sc = reg * (0.5 + nrng.random(len(U))) + dU * (reg.mean() + 1.0) / (dU.mean() + 1.0) * nrng.random()
            order = np.argsort(-sc)
        pref = np.argsort(sub, axis=1)
        ll = load.tolist()
        ok = True
        Ul = U.tolist() if hasattr(U, "tolist") else list(U)
        for idx in order.tolist():
            i = Ul[idx]; di = dem[i]
            placed = False
            for k in pref[idx].tolist():
                if ll[k] + di <= cap:
                    ll[k] += di; a[i] = k; placed = True
                    break
            if not placed:
                k = int(np.argmin(np.asarray(ll)))
                ll[k] += di; a[i] = k; ok = False
        load[:] = np.asarray(ll, dtype=np.int64)
        return ok

    def repair(Ds, a, load):
        while True:
            over = np.nonzero(load > cap)[0]
            if over.size == 0:
                return True
            if time.time() > deadline:
                return False
            k = int(over[np.argmax(load[over])])
            excess = int(load[k] - cap)
            mem = np.nonzero((a == k) & (d > 0))[0]
            if mem.size == 0:
                return False
            resid = cap - load
            dm = d[mem]
            feas = dm[:, None] <= resid[None, :]
            feas[:, k] = False
            if feas.any():
                delta = Ds[mem] - Ds[mem, k][:, None]
                full = (dm >= excess)[:, None] & feas
                if full.any():
                    cm = np.where(full, delta, _INF)
                else:
                    rows = feas.any(axis=1)
                    mx = dm[rows].max()
                    sel = (dm == mx)[:, None] & feas
                    cm = np.where(sel, delta, _INF)
                flat = int(np.argmin(cm))
                ii, m = divmod(flat, p)
                i = int(mem[ii])
                a[i] = m; load[k] -= d[i]; load[m] += d[i]
                continue
            found = False
            cur = Ds[ar, a]
            for i in mem[np.argsort(-dm)].tolist():
                di = dem[i]
                cond = (a != k) & (d < di) & (load[a] - d + di <= cap)
                if cond.any():
                    idx = np.nonzero(cond)[0]
                    red = np.minimum(di - d[idx], excess)
                    delta = Ds[i, a[idx]] - Ds[i, k] + Ds[idx, k] - cur[idx]
                    j = int(idx[np.lexsort((delta, -red))[0]])
                    m = int(a[j])
                    a[i] = m; a[j] = k
                    load[k] += dem[j] - di; load[m] += di - dem[j]
                    found = True
                    break
            if not found:
                return False

    def ls_assign(Ds, a, load):
        while time.time() < deadline:
            moved = False
            # --- shifts
            while time.time() < deadline:
                cur = Ds[ar, a]
                resid = cap - load
                feas = d[:, None] <= resid[None, :]
                delta = np.where(feas, Ds - cur[:, None], _INF)
                bm = delta.argmin(axis=1)
                bd = delta[ar, bm]
                cand = np.nonzero(bd < -_EPS)[0]
                if cand.size == 0:
                    break
                cand = cand[np.argsort(bd[cand])]
                sm = False
                for i in cand.tolist():
                    m = int(bm[i]); k = int(a[i])
                    if load[m] + dem[i] <= cap and Ds[i, m] < Ds[i, k] - _EPS:
                        a[i] = m; load[k] -= dem[i]; load[m] += dem[i]
                        sm = True
                if not sm:
                    break
                moved = True
            # --- swaps
            cur = Ds[ar, a]
            rmin = Ds.min(axis=1)
            cand = np.nonzero(cur > rmin + _EPS)[0]
            if cand.size:
                cand = cand[np.argsort(-(cur[cand] - rmin[cand]))]
                cnt = 0
                for i in cand.tolist():
                    cnt += 1
                    if (cnt & 31) == 0 and time.time() > deadline:
                        break
                    k = int(a[i])
                    ci = Ds[i, k]
                    if ci <= rmin[i] + _EPS:
                        continue
                    di = dem[i]
                    delta = Ds[i, a] - ci + Ds[:, k] - cur
                    feas = (a != k) & (load[a] - d + di <= cap) & (d - di + load[k] <= cap)
                    delta = np.where(feas, delta, _INF)
                    j = int(np.argmin(delta))
                    if delta[j] < -_EPS:
                        m = int(a[j])
                        a[i] = m; a[j] = k
                        load[k] += dem[j] - di; load[m] += di - dem[j]
                        cur[i] = Ds[i, m]; cur[j] = Ds[j, k]
                        moved = True
            if not moved:
                break

    # ----------------------------------------------------------- recentering
    use_lsa = _lsa is not None and p * n <= 1500000

    def cluster_cost_matrix(a):
        if _csr is not None:
            M = _csr((np.ones(n), (a, ar)), shape=(p, n))
            return np.asarray(M @ D)
        C = np.zeros((p, n))
        for k in range(p):
            mk = a == k
            if mk.any():
                C[k] = D[mk].sum(axis=0)
        return C

    def recenter(S, a, cur_cost):
        C = cluster_cost_matrix(a)
        if use_lsa:
            try:
                r, c = _lsa(C)
                S2 = np.empty(p, dtype=np.int64)
                S2[r] = c
                c2 = float(C[np.arange(p), S2].sum())
                if c2 < cur_cost - _EPS:
                    return S2, c2
                return None, cur_cost
            except Exception:
                pass
        S2 = np.array(S, dtype=np.int64)
        taken = np.zeros(n, dtype=bool)
        taken[S2] = True
        ch = False
        for k in nrng.permutation(p).tolist():
            taken[S2[k]] = False
            row = np.where(taken, _INF, C[k])
            j = int(np.argmin(row))
            if row[j] < C[k, S2[k]] - _EPS:
                S2[k] = j; ch = True
            taken[S2[k]] = True
        if ch:
            c2 = float(C[np.arange(p), S2].sum())
            if c2 < cur_cost - _EPS:
                return S2, c2
        return None, cur_cost

    def centers_from_labels(lab):
        C = cluster_cost_matrix(lab)
        if use_lsa:
            try:
                r, c = _lsa(C)
                S2 = np.empty(p, dtype=np.int64); S2[r] = c
                return S2
            except Exception:
                pass
        S2 = np.full(p, -1, dtype=np.int64)
        taken = np.zeros(n, dtype=bool)
        sizes = np.bincount(lab, minlength=p)
        for k in np.argsort(-sizes).tolist():
            row = np.where(taken, _INF, C[k])
            j = int(np.argmin(row)); S2[k] = j; taken[j] = True
        return S2

    def improve(S, a, load):
        Ds = D[:, S]
        ls_assign(Ds, a, load)
        cost = float(Ds[ar, a].sum())
        while time.time() < deadline:
            S2, c2 = recenter(S, a, cost)
            if S2 is None:
                break
            S = S2; Ds = D[:, S]; cost = c2
            ls_assign(Ds, a, load)
            cost = float(Ds[ar, a].sum())
        return S, a, load, Ds, cost

    # ------------------------------------------------------------ MIP helpers
    def gap_milp(S, tlim):
        if _milp is None or tlim < 0.05:
            return None
        Ds = D[:, S]
        nv = n * p
        c = Ds.ravel()
        A_eq = _csr((np.ones(nv), (np.repeat(ar, p), np.arange(nv))), shape=(n, nv))
        pos = np.repeat(d > 0, p)
        vidx = np.arange(nv)[pos]
        A_cap = _csr((np.repeat(d, p)[pos].astype(float), (np.tile(np.arange(p), n)[pos], vidx)), shape=(p, nv))
        cons = [_LC(A_eq, 1, 1), _LC(A_cap, -np.inf, cap)]
        res = _milp(c, constraints=cons, integrality=np.ones(nv), bounds=_Bounds(0, 1),
                    options={"time_limit": float(tlim), "disp": False})
        if res.x is None:
            return None
        X = res.x.reshape(n, p)
        a = X.argmax(axis=1).astype(np.int64)
        load = np.bincount(a, weights=d, minlength=p)
        if load.max() > cap + 1e-9:
            return None
        return a

    def binpack_milp(tlim):
        """returns (status, labels): status 'feas', 'infeas', 'unknown'"""
        if _milp is None or tlim < 0.05:
            return "unknown", None
        items = [i for i in np.argsort(-d, kind="stable").tolist() if dem[i] > 0]
        m = len(items)
        if m == 0:
            return "feas", np.zeros(n, dtype=np.int64)
        var_i = []; var_b = []
        for r, i in enumerate(items):
            for b in range(min(r, p - 1) + 1):
                var_i.append(r); var_b.append(b)
        var_i = np.asarray(var_i); var_b = np.asarray(var_b)
        nv = len(var_i)
        A_eq = _csr((np.ones(nv), (var_i, np.arange(nv))), shape=(m, nv))
        dd = np.asarray([dem[i] for i in items], dtype=float)
        A_cap = _csr((dd[var_i], (var_b, np.arange(nv))), shape=(p, nv))
        res = _milp(np.zeros(nv), constraints=[_LC(A_eq, 1, 1), _LC(A_cap, -np.inf, cap)],
                    integrality=np.ones(nv), bounds=_Bounds(0, 1),
                    options={"time_limit": float(tlim), "disp": False})
        if res.x is not None:
            lab = np.zeros(n, dtype=np.int64)
            x = res.x
            for v in np.nonzero(x > 0.5)[0].tolist():
                lab[items[var_i[v]]] = var_b[v]
            load = np.bincount(lab, weights=d, minlength=p)
            if load.max() <= cap:
                return "feas", lab
        if res.status == 2:
            return "infeas", None
        return "unknown", None

    def pack_heur(order):
        loads = np.zeros(p, dtype=np.int64)
        lab = np.zeros(n, dtype=np.int64)
        for i in order:
            di = dem[i]
            if di == 0:
                continue
            r = cap - loads - di
            r = np.where(r >= 0, r, cap + 1)
            k = int(np.argmin(r))
            if r[k] > cap:
                return None
            loads[k] += di; lab[i] = k
        return lab

    # ------------------------------------------------------------ construction
    def construct_kpp():
        first = int(nrng.integers(n))
        S = [first]
        chosen = np.zeros(n, dtype=bool); chosen[first] = True
        mind = D[:, first].copy()
        w = d.astype(float) + max(1.0, float(d.mean())) * 0.2
        for _ in range(1, p):
            pr = (mind ** 2) * w
            pr[chosen] = 0.0
            s = pr.sum()
            if not (s > 0):
                free = np.nonzero(~chosen)[0]
                j = int(free[nrng.integers(len(free))])
            else:
                j = int(nrng.choice(n, p=pr / s))
            S.append(j); chosen[j] = True
            mind = np.minimum(mind, D[:, j])
        return np.asarray(S, dtype=np.int64)

    def from_S(S, mode=0):
        Ds = D[:, S]
        a = np.zeros(n, dtype=np.int64)
        load = np.zeros(p, dtype=np.int64)
        ok = assign_subset(Ds, a, load, ar, mode)
        if not ok and not repair(Ds, a, load):
            return None
        return improve(S, a, load)

    sol = None
    cons_budget = max(0.02, 0.06 * (deadline - t0))
    tries = 0
    while True:
        tries += 1
        S0 = construct_kpp()
        r = from_S(S0, mode=(tries - 1) % 3)
        if r is not None:
            record(r[0], r[1])
            if sol is None or r[4] < sol[4]:
                sol = r
        if time.time() - t0 > cons_budget or time.time() > deadline:
            break
        if tries >= 2 and sol is not None and time.time() - t0 > cons_budget * 0.5:
            break

    if sol is None:
        # capacity-tight: bin packing route
        lab = None
        orders = [np.argsort(-d, kind="stable").tolist()]
        lab = pack_heur(orders[0])
        pk_end = t0 + 0.15 * (deadline - t0)
        while lab is None and time.time() < pk_end:
            noise = d.astype(float) * (1.0 + 0.3 * nrng.random(n))
            lab = pack_heur(np.argsort(-noise).tolist())
        if lab is None:
            st, lab = binpack_milp(max(0.05, 0.6 * tleft()))
            if st == "infeas":
                return _infeasible()
            while lab is None and time.time() < deadline:
                noise = d.astype(float) * (1.0 + 0.6 * nrng.random(n))
                lab = pack_heur(np.argsort(-noise).tolist())
            if lab is None:
                if holder["sol"] is not None:
                    return holder["sol"]
                return _infeasible()
        S0 = centers_from_labels(lab)
        a0 = lab.copy()
        load0 = np.bincount(a0, weights=d, minlength=p).astype(np.int64)
        record(S0, a0)
        sol = improve(S0, a0, load0)
        record(sol[0], sol[1])

    S, a, load, Ds, cost = sol
    record(S, a)

    # ------------------------------------------------------------ ILS
    polish_ok = _milp is not None and n * p <= 60000
    polish_used = 0.0
    polish_cap = 0.25 * (deadline - t0)
    polished = set()

    def try_polish():
        nonlocal polish_used
        if not polish_ok:
            return
        key = tuple(sorted(int(x) for x in best["S"]))
        if key in polished:
            return
        rem = tleft()
        budget = min(rem * 0.3, polish_cap - polish_used)
        if budget < 0.1:
            return
        polished.add(key)
        ts = time.time()
        try:
            aa = gap_milp(best["S"], budget)
        except Exception:
            aa = None
        polish_used += time.time() - ts
        if aa is not None:
            Sb = best["S"].copy()
            loadb = np.bincount(aa, weights=d, minlength=p).astype(np.int64)
            record(Sb, aa)
            r2 = improve(Sb, aa, loadb)
            record(r2[0], r2[1])

    if n == p:
        try_polish()
        return holder["sol"]

    cur = (S.copy(), a.copy(), load.copy(), Ds.copy(), cost)
    t_start = time.time()
    last_imp = time.time()
    total = max(1e-3, deadline - t_start)
    n_it = 0
    kmax = max(1, min(p, n - p, 2 + p // 8))
    stagn = 0

    while time.time() < deadline:
        n_it += 1
        if best["cost"] < _INF and n_it % 3 == 1:
            try_polish()
        S, a, load, Ds, cost = cur
        inS = np.zeros(n, dtype=bool); inS[S] = True
        # choose number of swaps (grows with stagnation)
        u = rng.random()
        kk = 1 if u < 0.55 else (2 if u < 0.8 else rng.randint(1, kmax))
        if stagn > 20 and rng.random() < 0.3:
            kk = rng.randint(1, kmax)
        kk = max(1, min(kk, p, n - p))
        curc = Ds[ar, a]
        newS = S.copy()
        closed = []
        opened_mask = inS.copy()
        loss = None
        for _t in range(kk):
            # open candidate
            v = rng.random()
            free = ~opened_mask
            if not free.any():
                break
            c = -1
            if v < 0.45:
                R = ar if n <= 600 else nrng.choice(n, 600, replace=False)
                g = np.maximum(curc[R][:, None] - D[R], 0.0).sum(axis=0)
                g[opened_mask] = 0.0
                g = g * g
                s = g.sum()
                if s > 0:
                    c = int(nrng.choice(n, p=g / s))
            elif v < 0.7:
                w = curc * (d + 1.0)
                s = w.sum()
                if s > 0:
                    i = int(nrng.choice(n, p=w / s))
                    row = np.where(opened_mask, _INF, D[i])
                    c = int(np.argmin(row))
            if c < 0 or opened_mask[c]:
                fi = np.nonzero(free)[0]
                c = int(fi[rng.randrange(len(fi))])
            # close candidate
            v = rng.random()
            avail = [k for k in range(p) if k not in closed]
            if not avail:
                break
            if v < 0.35:
                r_ = avail[rng.randrange(len(avail))]
            elif v < 0.7:
                dd = D[c, S[avail]] + D[S[avail], c]
                r_ = avail[int(np.argmin(dd))]
            else:
                if loss is None:
                    tmp = Ds.copy(); tmp[ar, a] = _INF
                    sec = tmp.min(axis=1) if p > 1 else np.zeros(n)
                    loss = np.bincount(a, weights=sec - curc, minlength=p)
                la = loss[avail]
                wts = 1.0 / (la - la.min() + 1.0 + 1e-9)
                wts = wts / wts.sum()
                r_ = avail[int(nrng.choice(len(avail), p=wts))]
            closed.append(r_)
            opened_mask[c] = True
            newS[r_] = c
        if not closed:
            break
        Dn = Ds.copy()
        for r_ in closed:
            Dn[:, r_] = D[:, newS[r_]]
        a2 = a.copy(); load2 = load.copy()
        cl = np.asarray(closed)
        U = np.nonzero(np.isin(a2, cl))[0]
        load2[cl] = 0
        ok = assign_subset(Dn, a2, load2, U, mode=rng.randrange(3))
        if not ok and not repair(Dn, a2, load2):
            stagn += 1
            continue
        if time.time() > deadline:
            break
        S2, a2, load2, D2, c2 = improve(newS, a2, load2)
        if record(S2, a2):
            last_imp = time.time(); stagn = 0
        else:
            stagn += 1
        frac = min(1.0, (time.time() - t_start) / total)
        T = 0.003 * max(cost, 1.0) / max(1.0, math.sqrt(p)) * (1.0 - frac)
        if c2 < cost - _EPS:
            cur = (S2, a2, load2, D2, c2)
        elif T > 0 and c2 - cost < 50 * T and rng.random() < math.exp(-(c2 - cost + 1e-12) / T):
            cur = (S2, a2, load2, D2, c2)
        if time.time() - last_imp > max(0.08 * total, 0.05) and best["S"] is not None:
            bS = best["S"].copy(); ba = best["a"].copy()
            bl = np.bincount(ba, weights=d, minlength=p).astype(np.int64)
            cur = (bS, ba, bl, D[:, bS], best["cost"])
            last_imp = time.time()

    return holder["sol"] if holder["sol"] is not None else _infeasible()
