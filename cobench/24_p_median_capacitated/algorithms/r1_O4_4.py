"""Capacity-slot matching and joint location neighbourhood search
Alternates capacity-feasible assignment descent with joint median matching, and keeps improving via adaptive integer-programming neighbourhoods and facility perturbations.
"""
import time
import math
import random
import numpy as np

try:
    from scipy.optimize import linear_sum_assignment as _lsa
except Exception:
    _lsa = None
try:
    from scipy.optimize import milp as _milp, LinearConstraint as _LC, Bounds as _Bounds
    from scipy.sparse import csr_matrix as _csr
except Exception:
    _milp = None
    _csr = None

_INF = float("inf")
_EPS = 1e-9


def _infeasible():
    return {"status": "infeasible", "medians": [], "assignments": []}


def solve(instance: dict, time_limit_s: float) -> dict:
    t0 = time.perf_counter()
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
    if not math.isfinite(tl):
        tl = 10.0
    deadline = t0 + max(0.001, tl - min(tl * 0.06, 0.02 + 0.035 * tl))
    strength = math.log1p(tl)
    work_size = max(256, int(850 * math.sqrt(tl)))
    local_deadline = deadline
    if cap < 1 or min(dem) < 0 or max(dem) > cap or sum(dem) > p * cap:
        return _infeasible()

    D = np.asarray(inst["distance_matrix"], dtype=np.float64).reshape(n, n)
    d = np.asarray(dem, dtype=np.int64)
    ar = np.arange(n)
    rng = random.Random(12345)
    nrng = np.random.default_rng(12345)

    def tleft():
        return deadline - time.perf_counter()

    if p == 1:
        j = min(range(n), key=lambda j: sum(inst["distance_matrix"][i][j] for i in range(n)))
        return {"status": "feasible", "medians": [j], "assignments": [j] * n}
    if sum(x > cap // 2 for x in dem) > p:
        return _infeasible()

    # Store the best complete, capacity-checked solution
    best = {"cost": _INF, "S": None, "a": None}

    def record(S, a):
        S = np.asarray(S, dtype=np.int64)
        a = np.asarray(a, dtype=np.int64)
        if S.shape != (p,) or a.shape != (n,):
            return False
        if S.min() < 0 or S.max() >= n or len(set(S.tolist())) != p:
            return False
        if a.min() < 0 or a.max() >= p:
            return False
        load = np.bincount(a, weights=d, minlength=p)
        if load.max() > cap:
            return False
        cost = float(D[ar, S[a]].sum())
        if cost >= best["cost"] - _EPS:
            return False
        aa = a.tolist()
        loads = [0] * p
        for i, k in enumerate(aa):
            loads[k] += dem[i]
        if max(loads) > cap:
            return False
        assign = S[a].tolist()
        exact = sum(inst["distance_matrix"][i][assign[i]] for i in range(n))
        if holder.get("cost") is not None and exact >= holder["cost"]:
            return False
        best["cost"] = cost
        best["S"] = S.copy()
        best["a"] = a.copy()
        holder["cost"] = exact
        holder["sol"] = {"status": "feasible", "medians": sorted(S.tolist()),
                         "assignments": assign}
        return True

    def assign_subset(Ds, a, load, U, mode=0):
        """Insert by regret value and demand, then repair capacity violations."""
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
            if time.perf_counter() > local_deadline:
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
        while time.perf_counter() < local_deadline:
            moved = False
            while time.perf_counter() < local_deadline:
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
            cur = Ds[ar, a]
            rmin = Ds.min(axis=1)
            cand = np.nonzero(cur > rmin + _EPS)[0]
            if cand.size:
                cand = cand[np.argsort(-(cur[cand] - rmin[cand]))]
                cnt = 0
                for i in cand.tolist():
                    cnt += 1
                    if (cnt & 31) == 0 and time.perf_counter() > local_deadline:
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

    use_lsa = _lsa is not None and p * n <= max(200000, int(500000 * math.sqrt(tl)))

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

    positive = np.flatnonzero(d > 0)
    equal_demand = positive.size == 0 or np.all(d[positive] == d[positive[0]])
    slot_limit = max(10000, int(450000 * math.sqrt(tl)))
    slot_time = 0.0

    def exact_slots(Ds):
        nonlocal slot_time
        if sum(dem) <= cap:
            return Ds.argmin(axis=1).astype(np.int64)
        if not equal_demand or _lsa is None or positive.size == 0:
            return None
        slots_per = min(len(positive), cap // dem[int(positive[0])])
        if slots_per * p < len(positive):
            return None
        if len(positive) * slots_per * p > slot_limit:
            return None
        if time.perf_counter() + max(slot_time * 1.5, 0.001) >= local_deadline:
            return None
        ts = time.perf_counter()
        slot = np.repeat(np.arange(p), slots_per)
        rr, cc = _lsa(Ds[np.ix_(positive, slot)])
        aa = Ds.argmin(axis=1).astype(np.int64)
        aa[positive[rr]] = slot[cc]
        slot_time = max(slot_time, time.perf_counter() - ts)
        return aa

    def improve(S, a, load, budget=None):
        nonlocal local_deadline
        if budget is None:
            budget = max(tl * 0.04, math.sqrt(tl) * 0.045)
        local_deadline = min(deadline, time.perf_counter() + budget)
        Ds = D[:, S]
        record(S, a)
        while time.perf_counter() < local_deadline:
            aa = exact_slots(Ds)
            if aa is not None:
                a = aa
                load = np.bincount(a, weights=d, minlength=p).astype(np.int64)
            else:
                ls_assign(Ds, a, load)
            cost = float(Ds[ar, a].sum())
            record(S, a)
            if time.perf_counter() >= local_deadline:
                break
            S2, c2 = recenter(S, a, cost)
            if S2 is None:
                break
            S = S2
            Ds = D[:, S]
            record(S, a)
        return S, a, load, Ds, float(Ds[ar, a].sum())

    def binpack_milp(tlim):
        if _milp is None or tlim <= 0:
            return "unknown", None
        end = min(deadline, time.perf_counter() + tlim)
        groups = {}
        for i, q in enumerate(dem):
            if q:
                groups.setdefault(q, []).append(i)
        kinds = sorted(groups, reverse=True)
        m = len(kinds)
        if not m:
            return "feas", np.zeros(n, dtype=np.int64)
        q = min(p, len(positive))
        nv = m * q
        ix = np.arange(nv)
        gi = np.repeat(np.arange(m), q)
        bi = np.tile(np.arange(q), m)
        counts = np.array([len(groups[v]) for v in kinds], dtype=float)
        sizes = np.array(kinds, dtype=float)
        Aeq = _csr((np.ones(nv), (gi, ix)), shape=(m, nv))
        Ac = _csr((sizes[gi], (bi, ix)), shape=(q, nv))
        cons = [_LC(Aeq, counts, counts), _LC(Ac, -np.inf, cap)]
        if q > 1:
            cons.append(_LC(Ac[:-1] - Ac[1:], 0, np.inf))
        rem = end - time.perf_counter()
        if rem <= 0:
            return "unknown", None
        res = _milp(np.zeros(nv), constraints=cons, integrality=np.ones(nv),
                    bounds=_Bounds(np.zeros(nv), counts[gi]),
                    options={"time_limit": float(rem), "disp": False})
        if res.x is not None:
            x = np.rint(res.x).reshape(m, q).astype(np.int64)
            if np.all(x >= 0) and np.array_equal(x.sum(axis=1), counts):
                lab = np.zeros(n, dtype=np.int64)
                for g, size in enumerate(kinds):
                    off = 0
                    for k in range(q):
                        cnt = int(x[g, k])
                        lab[groups[size][off:off + cnt]] = k
                        off += cnt
                load = np.bincount(lab, weights=d, minlength=p)
                if load.max() <= cap:
                    return "feas", lab
        return ("infeas" if res.status == 2 else "unknown"), None

    def pack_dfs(end):
        # Explicit stack avoids recursion-depth errors; equal remaining capacities are probed only once.
        items = sorted((i for i in range(n) if dem[i] > 0), key=lambda i: -dem[i])
        m = len(items)
        remain = [cap] * p
        lab = np.zeros(n, dtype=np.int64)
        frames = [None] * (m + 1)
        chosen = [-1] * m
        depth = 0
        while time.perf_counter() < end:
            if depth == m:
                return "feas", lab
            if frames[depth] is None:
                val = dem[items[depth]]
                seen = set()
                choices = []
                for k in sorted(range(p), key=lambda k: remain[k]):
                    r = remain[k]
                    if r >= val and r not in seen:
                        choices.append(k)
                        seen.add(r)
                frames[depth] = choices
            if frames[depth]:
                k = frames[depth].pop(0)
                chosen[depth] = k
                lab[items[depth]] = k
                remain[k] -= dem[items[depth]]
                depth += 1
                frames[depth] = None
            else:
                frames[depth] = None
                if depth == 0:
                    return "infeas", None
                depth -= 1
                remain[chosen[depth]] += dem[items[depth]]
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

    def construct_kpp():
        chosen = np.zeros(n, dtype=bool)
        S = []
        cur = None
        sample = min(n, max(p, int(work_size / max(1, n) + strength * math.sqrt(n))))
        for k in range(p):
            free = np.flatnonzero(~chosen)
            if time.perf_counter() >= local_deadline:
                S.extend(nrng.choice(free, p - k, replace=False).tolist())
                break
            cand = free if len(free) <= sample else nrng.choice(free, sample, replace=False)
            if cur is None:
                values = D[:, cand].sum(axis=0)
            else:
                values = np.minimum(cur[:, None], D[:, cand]).sum(axis=0)
            width = min(len(cand), max(1, int(1 + strength * nrng.random())))
            top = np.argpartition(values, width - 1)[:width]
            j = int(cand[int(nrng.choice(top))])
            S.append(j)
            chosen[j] = True
            cur = D[:, j].copy() if cur is None else np.minimum(cur, D[:, j])
        return np.asarray(S, dtype=np.int64)

    def from_S(S, mode=0):
        Ds = D[:, S]
        aa = exact_slots(Ds)
        if aa is not None:
            load = np.bincount(aa, weights=d, minlength=p).astype(np.int64)
            record(S, aa)
            return improve(S, aa, load, max(0.0, local_deadline - time.perf_counter()))
        a = np.zeros(n, dtype=np.int64)
        load = np.zeros(p, dtype=np.int64)
        ok = assign_subset(Ds, a, load, ar, mode)
        if not ok and not repair(Ds, a, load):
            return None
        record(S, a)
        return improve(S, a, load, max(0.0, local_deadline - time.perf_counter()))

    # First store a distance-independent feasible packing as insurance, then construct a better assignment.
    lab = pack_heur(np.argsort(-d, kind="stable").tolist())
    sol = None
    if lab is not None:
        S0 = np.arange(p, dtype=np.int64)
        record(S0, lab)
        if time.perf_counter() < deadline:
            S0 = centers_from_labels(lab)
            record(S0, lab)
        load0 = np.bincount(lab, weights=d, minlength=p).astype(np.int64)
        sol = (S0, lab, load0, D[:, S0], float(D[ar, S0[lab]].sum()))
    cons_end = min(deadline, t0 + tl * 0.09)
    tries = 0
    while p < n and time.perf_counter() < cons_end:
        local_deadline = cons_end
        S0 = construct_kpp()
        r = from_S(S0, mode=tries % 3)
        tries += 1
        if r is not None and (sol is None or r[4] < sol[4]):
            sol = r

    if sol is None:
        lab = None
        orders = [np.argsort(-d, kind="stable").tolist()]
        lab = pack_heur(orders[0])
        pk_end = t0 + 0.15 * (deadline - t0)
        while lab is None and time.perf_counter() < pk_end:
            noise = d.astype(float) * (1.0 + 0.3 * nrng.random(n))
            lab = pack_heur(np.argsort(-noise).tolist())
        if lab is None:
            st, lab = binpack_milp(max(0.05, 0.6 * tleft()))
            if st == "infeas":
                return _infeasible()
            if lab is None:
                st, lab = pack_dfs(deadline)
                if st == "infeas":
                    return _infeasible()
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

    def joint_neighborhood(S, a, load, joint, limit, budget, depth):
        """Fix the plan outside the region and jointly optimise the region's facilities and the complete assignment."""
        end = min(deadline, time.perf_counter() + budget)
        if _milp is None or end <= time.perf_counter():
            return None, False, False
        full = n * (n + 1 if joint else p) <= limit
        if full:
            R = np.arange(p, dtype=np.int64)
            U = ar
        else:
            seed = int(nrng.integers(p))
            ids = np.flatnonzero(a == seed)
            if ids.size:
                # Choose neighbouring regions from actual cross-cluster customer assignment costs, without assuming a symmetric or metric matrix.
                forward = D[np.ix_(ids, S)].mean(axis=0)
                back = np.bincount(a, weights=D[:, S[seed]], minlength=p)
                back /= np.maximum(1, np.bincount(a, minlength=p))
                related = forward + back
            else:
                related = nrng.random(p)
            related[seed] = -_INF
            R = np.argsort(related)[:min(p, depth)].astype(np.int64)
            U = np.flatnonzero(np.isin(a, R))
            if joint:
                while len(R) > 1 and len(U) * (len(R) + 1) > limit:
                    R = R[:-1]
                    U = np.flatnonzero(np.isin(a, R))
            elif len(U) * len(R) > limit:
                count = max(1, int(limit // len(R)))
                curv = D[U, S[a[U]]]
                pressure = curv - D[np.ix_(U, S[R])].min(axis=1)
                keys = pressure * (0.5 + nrng.random(len(U)))
                U = U[np.argpartition(-keys, count - 1)[:count]]
        if len(U) == 0:
            return None, False, False
        r = len(R)
        m = len(U)
        global_model = full and (joint or p == n)
        if joint:
            forbidden = np.zeros(n, dtype=bool)
            forbidden[S] = True
            forbidden[S[R]] = False
            free = np.flatnonzero(~forbidden)
            maxq = min(len(free), int(limit // m))
            if maxq < r:
                return None, False, False
            if full or m * len(free) <= limit:
                cand = free
            else:
                target = min(maxq, max(r + 1, int(r * (2 + strength))))
                pool = list(S[R])
                used = set(int(j) for j in pool)
                per = max(1, (target - r) // r)
                for k in R.tolist():
                    clients = np.flatnonzero(a == k)
                    if len(clients) == 0:
                        continue
                    values = D[clients].sum(axis=0)
                    values[forbidden] = _INF
                    cnt = min(per, len(free))
                    jj = np.argpartition(values, cnt - 1)[:cnt]
                    for j in jj.tolist():
                        if j not in used and len(pool) < target:
                            pool.append(j)
                            used.add(j)
                if len(pool) < target:
                    old = D[U, S[a[U]]]
                    gain = np.maximum(old[:, None] - D[U], 0).sum(axis=0)
                    gain[forbidden] = -_INF
                    gain[list(used)] = -_INF
                    cnt = min(target - len(pool), len(free) - len(pool))
                    if cnt:
                        pool.extend(np.argpartition(-gain, cnt - 1)[:cnt].tolist())
                cand = np.array(pool, dtype=np.int64)
        else:
            cand = S[R]
        q = len(cand)
        nv = m * q
        totalvars = nv + (q if joint else 0)
        ii = np.repeat(np.arange(m), q)
        jj = np.tile(np.arange(q), m)
        xx = np.arange(nv)
        costs = D[np.ix_(U, cand)].copy()
        rowmin = costs.min(axis=1)
        costs -= rowmin[:, None]
        c = np.zeros(totalvars)
        c[:nv] = costs.ravel()
        cutoff = float((D[U, S[a[U]]] - rowmin).sum()) - 0.5
        if cutoff < 0:
            return None, global_model, True
        Aeq = _csr((np.ones(nv), (ii, xx)), shape=(m, totalvars))
        vals = d[U][ii].astype(float) / cap
        if joint:
            Acap = _csr((np.concatenate((vals, -np.ones(q))),
                         (np.concatenate((jj, np.arange(q))),
                          np.concatenate((xx, nv + np.arange(q))))), shape=(q, totalvars))
            Alink = _csr((np.concatenate((np.ones(nv), -np.ones(nv))),
                          (np.concatenate((xx, xx)), np.concatenate((xx, nv + jj)))),
                         shape=(nv, totalvars))
            Ay = _csr((np.ones(q), (np.zeros(q, dtype=int), nv + np.arange(q))),
                      shape=(1, totalvars))
            cons = [_LC(Aeq, 1, 1), _LC(Acap, -np.inf, 0),
                    _LC(Alink, -np.inf, 0), _LC(Ay, r, r)]
        else:
            outside = load.copy()
            outside -= np.bincount(a[U], weights=d[U], minlength=p).astype(np.int64)
            Acap = _csr((vals, (jj, xx)), shape=(q, totalvars))
            cons = [_LC(Aeq, 1, 1), _LC(Acap, -np.inf, (cap - outside[R]) / cap)]
        cons.append(_LC(_csr(c.reshape(1, -1)), -np.inf, cutoff))
        remain = end - time.perf_counter()
        if remain <= max(0.001, tl * 0.001):
            return None, False, False
        res = _milp(c, integrality=np.ones(totalvars), bounds=_Bounds(0, 1),
                    constraints=cons, options={"time_limit": float(remain),
                                               "mip_rel_gap": 0.0, "disp": False})
        optimal = res.status in (0, 2)
        certified = global_model and optimal
        if res.x is None:
            return None, certified, optimal
        x = res.x[:nv].reshape(m, q)
        ak = x.argmax(axis=1)
        if np.any(x[np.arange(m), ak] < 0.5):
            return None, False, False
        S2 = S.copy()
        if joint:
            opened = np.flatnonzero(res.x[nv:] > 0.5)
            if len(opened) != r:
                return None, False, False
            S2[R] = cand[opened]
        inverse = {int(j): k for k, j in enumerate(S2)}
        mapped = [inverse.get(int(cand[k]), -1) for k in ak]
        if min(mapped) < 0:
            return None, False, False
        a2 = a.copy()
        a2[U] = mapped
        load2 = np.bincount(a2, weights=d, minlength=p).astype(np.int64)
        if load2.max() > cap:
            return None, False, False
        cost2 = float(D[ar, S2[a2]].sum())
        record(S2, a2)
        return (S2, a2, load2, D[:, S2], cost2), certified, optimal

    # Every improvement phase starts from a verified feasible solution.
    S = best["S"].copy()
    a = best["a"].copy()
    load = np.bincount(a, weights=d, minlength=p).astype(np.int64)
    cur = improve(S, a, load)
    if n == p and time.perf_counter() < deadline:
        local_deadline = deadline
        aa = exact_slots(D[:, S])
        if aa is not None:
            record(S, aa)
            return holder["sol"]

    start_search = time.perf_counter()
    last_imp = start_search
    last_reset = start_search
    mip_elapsed = 0.0
    next_mip = start_search
    mip_turn = 0
    neighborhood_size = work_size
    mip_share = min(0.38, 0.16 + 0.035 * strength)
    max_depth = max(1, min(p, int(1 + strength * math.sqrt(p) / 2)))
    gain_sample = min(n, max(1, int(work_size / max(1, math.sqrt(p)))))
    proven_assignments = set()

    while time.perf_counter() < deadline:
        now = time.perf_counter()
        oldbest = best["cost"]
        stagnation = (now - last_imp) / max(tl * 0.055, 0.001)
        depth = min(p, max(2, int(1 + math.sqrt(strength) + math.log1p(stagnation))))
        available_share = mip_share * (now - t0) + tl * 0.025
        if (_milp is not None and now >= next_mip and mip_elapsed < available_share
                and tleft() > max(tl * 0.012, 0.01)):
            ts = time.perf_counter()
            Sb = best["S"].copy()
            ab = best["a"].copy()
            lb = np.bincount(ab, weights=d, minlength=p).astype(np.int64)
            key = tuple(sorted(Sb.tolist()))
            joint = (mip_turn % 3 != 0 or key in proven_assignments or equal_demand) and p < n
            mip_turn += 1
            budget = min(tleft() * 0.3, tl * 0.1,
                         max(tl * 0.025, math.sqrt(tl) * 0.11) * (1 + 0.15 * math.log1p(stagnation)))
            r = None
            certified = False
            optimal = False
            try:
                r, certified, optimal = joint_neighborhood(
                    Sb, ab, lb, joint, neighborhood_size, budget, depth)
            except Exception:
                pass
            elapsed = time.perf_counter() - ts
            mip_elapsed += elapsed
            next_mip = time.perf_counter() + max(tl * 0.01, elapsed * 0.5)
            if optimal and elapsed < budget * 0.75:
                neighborhood_size = min(n * (n + 1), max(neighborhood_size + 1, int(neighborhood_size * 1.25)))
            elif not optimal:
                neighborhood_size = max(work_size // 2, int(neighborhood_size * 0.85))
            if not joint and n * p <= neighborhood_size and optimal:
                proven_assignments.add(key)
            if r is not None:
                r = improve(r[0], r[1], r[2])
                if r[4] < cur[4]:
                    cur = r
            if best["cost"] < oldbest - _EPS:
                last_imp = time.perf_counter()
            if certified:
                break
            if time.perf_counter() >= deadline:
                break

        S, a, load, Ds, cost = cur
        newS = S.copy()
        current_cost = Ds[ar, a]
        local_deadline = min(deadline, time.perf_counter() + max(tl * 0.018, math.sqrt(tl) * 0.018))
        if n > p:
            depth_now = min(max_depth, max(1, int(1 + math.log1p(stagnation))))
            kk = 1 if rng.random() < 0.6 else rng.randint(1, depth_now)
            kk = min(kk, p, n - p)
            opened = np.zeros(n, dtype=bool)
            opened[S] = True
            closed = []
            alt = Ds.copy()
            alt[ar, a] = _INF
            second = alt.min(axis=1)
            for _ in range(kk):
                free = np.flatnonzero(~opened)
                if len(free) == 0:
                    break
                if rng.random() < 0.72:
                    rows = ar if gain_sample == n else nrng.choice(n, gain_sample, replace=False)
                    gain = np.maximum(current_cost[rows, None] - D[rows], 0).sum(axis=0)
                    gain[opened] = 0
                    scale = gain.max()
                    if scale > 0:
                        gain = (gain / scale) ** 2
                        j = int(nrng.choice(n, p=gain / gain.sum()))
                    else:
                        j = int(nrng.choice(free))
                else:
                    j = int(nrng.choice(free))
                avail = np.array([k for k in range(p) if k not in closed], dtype=np.int64)
                if rng.random() < 0.25:
                    k = int(nrng.choice(avail))
                else:
                    base = np.minimum(current_cost, D[:, j])
                    loss = np.bincount(a, weights=np.minimum(second, D[:, j]) - base, minlength=p)
                    rank = np.argsort(loss[avail])
                    breadth = min(len(avail), max(1, int(1 + strength * rng.random())))
                    k = int(avail[int(nrng.choice(rank[:breadth]))])
                newS[k] = j
                opened[j] = True
                closed.append(k)
            if not closed:
                continue
            Dn = D[:, newS]
            changed = np.array(closed, dtype=np.int64)
            # Release both the customers of the closed facility and boundary customers that can benefit from the new facility.
            mask = np.isin(a, changed)
            mask |= Dn[:, changed].min(axis=1) < current_cost
            U = np.flatnonzero(mask)
        else:
            Dn = Ds
            count = min(p, max(2, depth))
            changed = nrng.choice(p, count, replace=False)
            U = np.flatnonzero(np.isin(a, changed))
        a2 = a.copy()
        load2 = load - np.bincount(a[U], weights=d[U], minlength=p).astype(np.int64)
        ok = assign_subset(Dn, a2, load2, U, mode=rng.randrange(3))
        if not ok and not repair(Dn, a2, load2):
            # Facility replacement does not change cluster demand; the original labels are always a feasible fallback for the new facility set.
            a2 = a.copy()
            load2 = load.copy()
        r = improve(newS, a2, load2)
        if best["cost"] < oldbest - _EPS:
            last_imp = time.perf_counter()
        fraction = min(1.0, (time.perf_counter() - t0) / tl)
        temperature = max(cost, 1) * 0.003 / math.sqrt(p) * (1 - fraction)
        delta = r[4] - cost
        if delta <= 0 or (temperature > 0 and delta / temperature < 50
                          and rng.random() < math.exp(-delta / temperature)):
            cur = r
        now = time.perf_counter()
        if now - last_reset > max(tl * 0.09, 0.001):
            Sb = best["S"].copy()
            ab = best["a"].copy()
            lb = np.bincount(ab, weights=d, minlength=p).astype(np.int64)
            cur = (Sb, ab, lb, D[:, Sb], best["cost"])
            last_reset = now

    return holder["sol"] if holder["sol"] is not None else _infeasible()
