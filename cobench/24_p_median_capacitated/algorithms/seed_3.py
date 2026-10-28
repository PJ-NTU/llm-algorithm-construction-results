import time
import math
import random
import numpy as np

try:
    import mip as _mip
    _HAS_MIP = True
except Exception:  # pragma: no cover
    _mip = None
    _HAS_MIP = False

try:
    from scipy.optimize import linear_sum_assignment as _lsa
except Exception:  # pragma: no cover
    _lsa = None

_BIG = 4 * 10 ** 18


def _infeasible():
    return {"status": "infeasible", "medians": [], "assignments": []}


class _Sol(object):
    __slots__ = ("M", "a", "load", "mem", "cost")

    def copy(self):
        s = _Sol()
        s.M = list(self.M)
        s.a = list(self.a)
        s.load = list(self.load)
        s.mem = {k: set(v) for k, v in self.mem.items()}
        s.cost = self.cost
        return s


def _verify(n, p, C, dem, D, M, a):
    try:
        if len(M) != p or len(set(M)) != p:
            return None
        Ms = set(M)
        for m in M:
            if not (0 <= m < n):
                return None
        if len(a) != n:
            return None
        load = {}
        cost = 0
        for i in range(n):
            j = a[i]
            if j not in Ms:
                return None
            load[j] = load.get(j, 0) + dem[i]
            cost += D[i][j]
        for v in load.values():
            if v > C:
                return None
        return cost
    except Exception:
        return None


def solve(instance, time_limit_s):
    t0 = time.time()
    holder = {}
    try:
        res = _solve(instance, time_limit_s, t0, holder)
        if isinstance(res, dict):
            return res
    except Exception:
        pass
    try:
        if "best" in holder:
            n, p, C, dem, D = holder["data"]
            M, a = holder["best"]
            if _verify(n, p, C, dem, D, M, a) is not None:
                return {"status": "feasible", "medians": sorted(int(x) for x in M),
                        "assignments": [int(x) for x in a]}
    except Exception:
        pass
    return _infeasible()


# ---------------------------------------------------------------- packing
def _l2_bound(dem, C):
    items = [d for d in dem if d > 0]
    if not items:
        return 0
    best = int(math.ceil(sum(items) / float(C)))
    alphas = set([0])
    for d in items:
        if 2 * d <= C:
            alphas.add(d)
    for al in alphas:
        n1 = 0
        n2 = 0
        s2 = 0
        s3 = 0
        for d in items:
            if d > C - al:
                n1 += 1
            elif 2 * d > C:
                n2 += 1
                s2 += d
            elif d >= al:
                s3 += d
        extra = s3 - (n2 * C - s2)
        L = n1 + n2 + (int(math.ceil(extra / float(C))) if extra > 0 else 0)
        if L > best:
            best = L
    return best


def _pack_fit(order, dem, p, C, best_fit):
    loads = [0] * p
    bins = [[] for _ in range(p)]
    for i in order:
        d = dem[i]
        bb = -1
        if best_fit:
            bl = -1
            for b in range(p):
                l = loads[b]
                if l + d <= C and l > bl:
                    bl = l
                    bb = b
        else:
            for b in range(p):
                if loads[b] + d <= C:
                    bb = b
                    break
        if bb < 0:
            return None
        loads[bb] += d
        bins[bb].append(i)
    return bins


def _find_packing(n, p, C, dem, rng, deadline, budget):
    """returns ('ok', bins) / ('infeasible', None) / ('unknown', None)"""
    if max(dem) > C or sum(dem) > p * C:
        return "infeasible", None
    if _l2_bound(dem, C) > p:
        return "infeasible", None
    order = sorted(range(n), key=lambda i: -dem[i])
    for bf in (False, True):
        b = _pack_fit(order, dem, p, C, bf)
        if b is not None:
            return "ok", b
    tend = min(deadline, time.time() + budget * 0.3)
    while time.time() < tend:
        keys = [(-dem[i] * (1.0 + 0.3 * rng.random()), i) for i in range(n)]
        keys.sort()
        b = _pack_fit([i for _, i in keys], dem, p, C, rng.random() < 0.5)
        if b is not None:
            return "ok", b
    if _HAS_MIP:
        try:
            left = min(deadline - time.time(), budget * 0.7)
            if left > 0.05:
                pos = [i for i in order if dem[i] > 0]
                mdl = _mip.Model(sense=_mip.MINIMIZE, solver_name=_mip.CBC)
                mdl.verbose = 0
                x = {}
                for k, i in enumerate(pos):
                    for bb in range(min(k + 1, p)):
                        x[k, bb] = mdl.add_var(var_type=_mip.BINARY)
                for k, i in enumerate(pos):
                    mdl += _mip.xsum(x[k, bb] for bb in range(min(k + 1, p))) == 1
                for bb in range(p):
                    terms = [dem[i] * x[k, bb] for k, i in enumerate(pos) if bb <= k]
                    if terms:
                        mdl += _mip.xsum(terms) <= C
                mdl.objective = _mip.xsum([])
                st = mdl.optimize(max_seconds=left)
                if st == _mip.OptimizationStatus.INFEASIBLE:
                    return "infeasible", None
                if mdl.num_solutions > 0:
                    bins = [[] for _ in range(p)]
                    for k, i in enumerate(pos):
                        for bb in range(min(k + 1, p)):
                            v = x[k, bb].x
                            if v is not None and v > 0.5:
                                bins[bb].append(i)
                                break
                    for i in range(n):
                        if dem[i] == 0:
                            bins[0].append(i)
                    ok = sum(len(b) for b in bins) == n and all(sum(dem[i] for i in b) <= C for b in bins)
                    if ok:
                        return "ok", bins
        except Exception:
            pass
    return "unknown", None


# ---------------------------------------------------------------- main
def _solve(instance, T, t0, holder):
    n = int(instance["n"])
    p = int(instance["p"])
    C = int(instance["capacity"])
    dem = [int(x) for x in instance["demands"]]
    if n < 1 or p < 1 or p > n or len(dem) != n:
        return _infeasible()
    Draw = instance["distance_matrix"]
    Dn = np.asarray(Draw, dtype=np.int64).reshape(n, n)
    if isinstance(Draw, list) and all(isinstance(r, list) for r in Draw):
        D = Draw
    else:
        D = Dn.tolist()
    demn = np.asarray(dem, dtype=np.int64)
    holder["data"] = (n, p, C, dem, D)
    try:
        T = float(T)
    except Exception:
        T = 10.0
    if not (T > 0):
        T = 1.0
    safety = min(0.5, 0.06 * T + 0.02)
    deadline = t0 + T - safety

    def left():
        return deadline - time.time()

    rng = random.Random(20240915 + n * 7 + p)
    ar_n = np.arange(n)

    status, bins = _find_packing(n, p, C, dem, rng, deadline, max(0.05, 0.35 * T))
    if status == "infeasible":
        return _infeasible()

    # ------------------------------------------------ helpers
    def build_sol(M, a):
        s = _Sol()
        s.M = list(M)
        s.a = list(a)
        s.load = [0] * n
        s.mem = {m: set() for m in M}
        c = 0
        for i in range(n):
            j = a[i]
            s.load[j] += dem[i]
            s.mem[j].add(i)
            c += D[i][j]
        s.cost = c
        return s

    best_holder = {"sol": None}

    def record(s):
        b = best_holder["sol"]
        if b is None or s.cost < b.cost:
            best_holder["sol"] = s.copy()
            holder["best"] = (list(s.M), list(s.a))
            return True
        return False

    def reinsert(A, Mnew, nl, modes=(0, 1)):
        sub = Dn[np.ix_(A, Mnew)]
        k = len(Mnew)
        if k >= 2:
            part = np.partition(sub, 1, axis=1)
            regret = part[:, 1] - part[:, 0]
        else:
            regret = np.zeros(len(A), dtype=np.int64)
        ordr = np.argsort(sub, axis=1, kind="stable").tolist()
        Adem = demn[A]
        for mode in modes:
            if mode == 0:
                order = np.lexsort((-Adem, -regret)).tolist()
            elif mode == 1:
                order = np.lexsort((-regret, -Adem)).tolist()
            else:
                noise = np.array([rng.random() for _ in range(len(A))])
                order = np.argsort(-(regret + 1.0) * (0.5 + noise) - 1e-9 * Adem).tolist()
            nl2 = dict(nl)
            asg = {}
            cost = 0
            ok = True
            for t in order:
                i = A[t]
                di = dem[i]
                Di = D[i]
                placed = False
                for idx in ordr[t]:
                    mm = Mnew[idx]
                    if nl2[mm] + di <= C:
                        nl2[mm] += di
                        asg[i] = mm
                        cost += Di[mm]
                        placed = True
                        break
                if not placed:
                    # single ejection repair
                    bestv = None
                    for mm in Mnew:
                        need = nl2[mm] + di - C
                        for u in [q for q, v in asg.items() if v == mm]:
                            du = dem[u]
                            if du < need:
                                continue
                            Du = D[u]
                            for l in Mnew:
                                if l != mm and nl2[l] + du <= C:
                                    v = Di[mm] + Du[l] - Du[mm]
                                    if bestv is None or v < bestv[0]:
                                        bestv = (v, mm, u, l)
                    if bestv is None:
                        ok = False
                        break
                    v, mm, u, l = bestv
                    nl2[mm] += di - dem[u]
                    nl2[l] += dem[u]
                    asg[i] = mm
                    asg[u] = l
                    cost += v
            if ok:
                return cost, asg, nl2
        return None

    def greedy_solution(M, modes=(0, 1)):
        nl = {m: 0 for m in M}
        r = reinsert(list(range(n)), list(M), nl, modes)
        if r is None:
            return None
        cost, asg, _ = r
        a = [asg[i] for i in range(n)]
        return build_sol(M, a)

    def packing_solution(M):
        if bins is None:
            return None
        Mi = list(M)
        cm = np.zeros((p, p), dtype=np.int64)
        for b in range(p):
            if bins[b]:
                cm[b] = Dn[np.ix_(bins[b], Mi)].sum(axis=0)
        if _lsa is not None:
            rows, cols = _lsa(cm)
            mp = {int(r): Mi[int(c)] for r, c in zip(rows, cols)}
        else:
            mp = {b: Mi[b] for b in range(p)}
        a = [0] * n
        for b in range(p):
            for i in bins[b]:
                a[i] = mp[b]
        return build_sol(M, a)

    def assign_ls(s):
        a = s.a
        load = s.load
        mem = s.mem
        M = s.M
        cnt = 0
        while True:
            moved = False
            for i in range(n):
                ai = a[i]
                di = dem[i]
                Di = D[i]
                bestd = Di[ai]
                bk = -1
                for k in M:
                    dk = Di[k]
                    if dk < bestd and load[k] + di <= C:
                        bestd = dk
                        bk = k
                if bk >= 0:
                    s.cost += bestd - Di[ai]
                    load[ai] -= di
                    load[bk] += di
                    mem[ai].discard(i)
                    mem[bk].add(i)
                    a[i] = bk
                    moved = True
            for i in range(n):
                if (i & 63) == 0 and left() <= 0:
                    return
                ai = a[i]
                di = dem[i]
                Di = D[i]
                dai = Di[ai]
                bestg = 0
                bi2 = -1
                bk = -1
                hi = C - load[ai] + di
                for k in M:
                    g = dai - Di[k]
                    if g <= 0:
                        continue
                    lo = load[k] + di - C
                    for i2 in mem[k]:
                        d2 = dem[i2]
                        if d2 < lo or d2 > hi:
                            continue
                        D2 = D[i2]
                        gain = g - (D2[ai] - D2[k])
                        if gain > bestg:
                            bestg = gain
                            bi2 = i2
                            bk = k
                if bi2 >= 0:
                    d2 = dem[bi2]
                    mem[ai].discard(i)
                    mem[bk].discard(bi2)
                    mem[bk].add(i)
                    mem[ai].add(bi2)
                    load[ai] += d2 - di
                    load[bk] += di - d2
                    a[i] = bk
                    a[bi2] = ai
                    s.cost -= bestg
                    moved = True
            cnt += 1
            if not moved or left() <= 0:
                break

    def recenter(s):
        changed = False
        for idx in range(p):
            m = s.M[idx]
            S = s.mem[m]
            if not S:
                continue
            Sl = list(S)
            col = Dn[Sl].sum(axis=0)
            cur = int(col[m])
            others = [x for x in s.M if x != m]
            if others:
                col[others] = _BIG
            j = int(np.argmin(col))
            v = int(col[j])
            if v < cur and j != m:
                s.M[idx] = j
                s.mem[j] = s.mem.pop(m)
                s.load[j] = s.load[m]
                s.load[m] = 0
                for i in Sl:
                    s.a[i] = j
                s.cost += v - cur
                changed = True
        return changed

    def curd_of(s):
        return Dn[ar_n, np.asarray(s.a, dtype=np.int64)]

    def eval_swap(s, m, j, curd, modes=(0, 1)):
        a = s.a
        load = s.load
        Aset = set(np.flatnonzero(Dn[:, j] < curd).tolist())
        Aset |= s.mem[m]
        Mnew = [x for x in s.M if x != m]
        Mnew.append(j)
        nl = {x: load[x] for x in Mnew}
        nl[j] = 0
        if not Aset:
            return 0, m, j, {}, nl
        A = list(Aset)
        old = 0
        for i in A:
            ai = a[i]
            old += D[i][ai]
            if ai != m:
                nl[ai] -= dem[i]
        r = reinsert(A, Mnew, nl, modes)
        if r is None:
            return None
        cost, asg, nl2 = r
        return cost - old, m, j, asg, nl2

    def apply_swap(s, res):
        delta, m, j, asg, nl = res
        idx = s.M.index(m)
        s.M[idx] = j
        mem = s.mem
        mem.pop(m, None)
        mem[j] = set()
        for i, mm in asg.items():
            old = s.a[i]
            if old != m:
                mem[old].discard(i)
            mem[mm].add(i)
            s.a[i] = mm
        s.load[m] = 0
        for x, v in nl.items():
            s.load[x] = v
        s.cost += delta

    def cand_for(s, m, K, is_med):
        S = s.mem[m]
        nm = n - p
        if nm <= 0:
            return []
        if S:
            col = Dn[list(S)].sum(axis=0).astype(np.float64)
        else:
            col = Dn[:, m].astype(np.float64) + Dn[m, :].astype(np.float64)
        col[is_med] = np.inf
        K = min(K, nm)
        if K >= nm:
            c = np.flatnonzero(~is_med)
        else:
            c = np.argpartition(col, K - 1)[:K]
        return [int(x) for x in c if not is_med[x]]

    def median_descent(s, K):
        if n == p:
            return False
        improved = False
        is_med = np.zeros(n, dtype=bool)
        is_med[s.M] = True
        curd = curd_of(s)
        gain_c = []
        if n <= 2500:
            gv = np.maximum(curd[:, None] - Dn, 0).sum(axis=0).astype(np.float64)
            gv[is_med] = -1.0
            kk = min(max(2, K // 2), n - p)
            gain_c = [int(x) for x in np.argpartition(-gv, kk - 1)[:kk]]
        order = list(s.M)
        rng.shuffle(order)
        for m in order:
            if left() <= 0:
                break
            if m not in s.mem:
                continue
            cands = cand_for(s, m, K, is_med)
            cs = set(cands)
            for g in gain_c:
                if not is_med[g] and g not in cs:
                    cands.append(g)
                    cs.add(g)
            bestr = None
            for j in cands:
                r = eval_swap(s, m, j, curd)
                if r is not None and r[0] < 0 and (bestr is None or r[0] < bestr[0]):
                    bestr = r
            if bestr is not None:
                apply_swap(s, bestr)
                is_med[m] = False
                is_med[bestr[2]] = True
                curd = curd_of(s)
                improved = True
        return improved

    def local_search(s, K):
        while left() > 0:
            c0 = s.cost
            assign_ls(s)
            if recenter(s):
                assign_ls(s)
            if left() <= 0:
                break
            median_descent(s, K)
            if s.cost >= c0:
                break

    def perturb(s, k, K):
        if n == p:
            # only reassignments possible: random shifts
            for _ in range(k * 3):
                i = rng.randrange(n)
                m = rng.choice(s.M)
                if m != s.a[i] and s.load[m] + dem[i] <= C:
                    ai = s.a[i]
                    s.cost += D[i][m] - D[i][ai]
                    s.load[ai] -= dem[i]
                    s.load[m] += dem[i]
                    s.mem[ai].discard(i)
                    s.mem[m].add(i)
                    s.a[i] = m
            return
        for _ in range(k):
            is_med = np.zeros(n, dtype=bool)
            is_med[s.M] = True
            curd = curd_of(s)
            for _t in range(6):
                m = rng.choice(s.M)
                if rng.random() < 0.7:
                    c = cand_for(s, m, 3 * K, is_med)
                    if not c:
                        continue
                    j = rng.choice(c)
                else:
                    j = rng.randrange(n)
                    if is_med[j]:
                        continue
                r = eval_swap(s, m, j, curd, modes=(2, 0, 1))
                if r is not None:
                    apply_swap(s, r)
                    break

    # ------------------------------------------------ MIP neighbourhoods
    def mip_sub(s, R, U, cand, budget, fixed_y=False):
        """solve sub CPMP: choose |R| medians from cand, assign U. returns (improved, optimal)"""
        r = len(R)
        Ul = list(U)
        subcost = 0
        for i in Ul:
            subcost += D[i][s.a[i]]
        mdl = _mip.Model(sense=_mip.MINIMIZE, solver_name=_mip.CBC)
        mdl.verbose = 0
        y = {}
        if not fixed_y:
            for j in cand:
                y[j] = mdl.add_var(var_type=_mip.BINARY)
        x = {}
        inv = {j: [] for j in cand}
        obj = []
        for i in Ul:
            row = []
            Di = D[i]
            for j in cand:
                v = mdl.add_var(var_type=_mip.BINARY)
                x[i, j] = v
                inv[j].append(i)
                row.append(v)
                obj.append(Di[j] * v)
            mdl += _mip.xsum(row) == 1
        if not fixed_y:
            mdl += _mip.xsum(y.values()) == r
            strong = len(x) <= 40000
            for j in cand:
                if inv[j]:
                    mdl += _mip.xsum(dem[i] * x[i, j] for i in inv[j]) <= C * y[j]
                    for i in inv[j]:
                        if strong or dem[i] == 0:
                            mdl += x[i, j] <= y[j]
        else:
            for j in cand:
                if inv[j]:
                    mdl += _mip.xsum(dem[i] * x[i, j] for i in inv[j]) <= C
        mdl.objective = _mip.xsum(obj)
        start = []
        if not fixed_y:
            for j in R:
                start.append((y[j], 1.0))
        for i in Ul:
            start.append((x[i, s.a[i]], 1.0))
        try:
            mdl.start = start
        except Exception:
            pass
        rem = left() - 0.05
        if rem <= 0.05:
            return False, False
        st = mdl.optimize(max_seconds=max(0.05, min(budget, rem)))
        optimal = (st == _mip.OptimizationStatus.OPTIMAL)
        if mdl.num_solutions <= 0 or mdl.objective_value is None:
            return False, optimal
        if mdl.objective_value > subcost - 0.5:
            return False, optimal
        newa = {}
        for i in Ul:
            for j in cand:
                v = x[i, j].x
                if v is not None and v > 0.5:
                    newa[i] = j
                    break
            if i not in newa:
                return False, optimal
        if fixed_y:
            newR = list(R)
        else:
            newR = [j for j in cand if y[j].x is not None and y[j].x > 0.5]
            if len(newR) != r:
                return False, optimal
        Rs = set(R)
        newM = [m for m in s.M if m not in Rs] + newR
        a2 = list(s.a)
        for i, j in newa.items():
            a2[i] = j
        c2 = _verify(n, p, C, dem, D, newM, a2)
        if c2 is None or c2 >= s.cost:
            return False, optimal
        ns = build_sol(newM, a2)
        s.M, s.a, s.load, s.mem, s.cost = ns.M, ns.a, ns.load, ns.mem, ns.cost
        return True, optimal

    def lns_region(s, V, budget):
        M = s.M
        seed = rng.choice(M)
        dm = sorted((D[seed][x] + D[x][seed] + rng.random() * 1e-6, x) for x in M)
        R = []
        Ucnt = 0
        for _, x in dm:
            sz = Ucnt + len(s.mem[x])
            if R and len(R) >= 2 and sz * (sz + len(R) + 1) > V:
                break
            R.append(x)
            Ucnt = sz
        if len(R) == 1 and p >= 2:
            R.append(dm[1][1])
        U = set()
        for x in R:
            U |= s.mem[x]
        if not U:
            return False, False, False
        Rs = set(R)
        fixed = set(M) - Rs
        cand = [j for j in (U | Rs) if j not in fixed]
        full = (len(R) == p)
        imp, opt = mip_sub(s, R, U, cand, budget)
        return imp, opt, full

    def gap_polish(s, budget):
        U = set(range(n))
        imp, opt = mip_sub(s, list(s.M), U, list(s.M), budget, fixed_y=True)
        return imp, opt

    # ------------------------------------------------ initial medians
    def greedy_medians():
        cur = np.full(n, _BIG // 4, dtype=np.int64)
        chosen = []
        is_med = np.zeros(n, dtype=bool)
        for _ in range(p):
            if n * n <= 4000000:
                cands = np.flatnonzero(~is_med)
            else:
                ssz = max(20, int(4000000 // n))
                cands = np.flatnonzero(~is_med)
                if len(cands) > ssz:
                    cands = np.array(rng.sample(cands.tolist(), ssz))
            tot = np.minimum(cur[:, None], Dn[:, cands]).sum(axis=0)
            j = int(cands[int(np.argmin(tot))])
            chosen.append(j)
            is_med[j] = True
            cur = np.minimum(cur, Dn[:, j])
        return chosen

    init = None
    try:
        M0 = greedy_medians()
    except Exception:
        M0 = list(range(p))
    init = greedy_solution(M0)
    if init is None:
        init = packing_solution(M0)
    if init is None:
        # try random median sets with greedy assignment until time
        while left() > 0 and init is None:
            Mr = rng.sample(range(n), p)
            init = greedy_solution(Mr, modes=(1, 0, 2))
    if init is None:
        return _infeasible()
    record(init)

    K = max(4, min(n - p, int(round(math.sqrt(max(1, n - p)))) + 2)) if n > p else 0

    cur = init.copy()
    local_search(cur, K)
    record(cur)

    proven = False
    polished = set()
    V = 1200.0
    Vmax = float(n * n + n)
    t_ils = 1e-9
    t_mip = 1e-9
    mip_share = 0.5 if _HAS_MIP else 0.0
    stagn = 0
    it = 0
    mip_fail = 0
    while left() > 0 and not proven:
        it += 1
        use_mip = _HAS_MIP and mip_fail < 8 and (t_mip / (t_mip + t_ils)) < mip_share and left() > 0.15
        if use_mip:
            ts = time.time()
            try:
                best = best_holder["sol"]
                s = best.copy()
                key = tuple(sorted(s.M))
                budget = max(0.2, 0.12 * T)
                if key not in polished:
                    polished.add(key)
                    imp, opt = gap_polish(s, budget)
                    if imp and p == n and opt:
                        record(s)
                        proven = True
                else:
                    t1 = time.time()
                    imp, opt, full = lns_region(s, V, budget)
                    el = time.time() - t1
                    if opt and full and n == len(set().union(*[best.mem[m] for m in best.M])):
                        # full model solved to optimality
                        if imp:
                            record(s)
                        proven = True
                    if opt and el < 0.5 * budget:
                        V = min(Vmax, V * 1.25)
                    elif not opt:
                        V = max(150.0, V * 0.7)
                if imp:
                    local_search(s, K)
                    if record(s):
                        stagn = 0
                    if s.cost < cur.cost:
                        cur = s.copy()
                mip_fail = 0
            except Exception:
                mip_fail += 1
            t_mip += time.time() - ts
        else:
            ts = time.time()
            s = cur.copy()
            kmax = max(1, min(p, 1 + stagn // 8, max(2, p // 2)))
            perturb(s, 1 + rng.randrange(kmax), K)
            local_search(s, K)
            if s.cost < cur.cost:
                cur = s
                if record(s):
                    stagn = 0
                else:
                    stagn += 1
            else:
                stagn += 1
                if s.cost == cur.cost:
                    cur = s
                if stagn % 25 == 0:
                    cur = best_holder["sol"].copy()
            t_ils += time.time() - ts

    best = best_holder["sol"]
    c = _verify(n, p, C, dem, D, best.M, best.a)
    if c is None:
        return None
    return {"status": "feasible", "medians": sorted(int(x) for x in best.M),
            "assignments": [int(x) for x in best.a]}
