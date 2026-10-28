"""Feasibility-guarded rebuild and double-swap search
Guarantees coverage via sink strongly connected components and combines incremental single swaps, double-swap probes, group removal-rebuild and elite candidate recombination, keeping the best solution throughout the wall-clock budget.
"""

import time
import math
import random
import heapq
import itertools
import decimal

import numpy as np

_INF = float("inf")


def _now():
    return time.perf_counter()


class _Ctx:
    def __init__(self):
        self.best_S = None
        self.best_cost = _INF


def _raw_feasible(instance):
    # In the shortest-path closure, choosing one vertex per sink strongly connected component covers all vertices.
    n, p, rows = int(instance["n"]), int(instance["p"]), instance["dist"]
    unseen = set(range(n))
    chosen = []
    while unseen:
        v = next(iter(unseen))
        reachable = [j for j in range(n) if rows[v][j] is not None]
        component = [j for j in reachable if rows[j][v] is not None]
        unseen.difference_update(component)
        if len(component) == len(reachable):
            chosen.append(v)
    used = set(chosen)
    for v in range(n):
        if len(chosen) == p:
            break
        if v not in used:
            chosen.append(v)
    return chosen


def _matrix(rows):
    try:
        D = np.array(rows, dtype=np.float64)
        if np.isinf(D).any():
            raise OverflowError
    except (OverflowError, ValueError):
        # Finite integers beyond double-precision range are still reachable distances and are not treated as unreachable.
        biggest = max(v for row in rows for v in row if v is not None)
        with decimal.localcontext() as context:
            context.prec = 32
            scale = decimal.Decimal(biggest) if biggest else decimal.Decimal(1)
            D = np.array([[np.nan if v is None else float(decimal.Decimal(v) / scale)
                           for v in row] for row in rows], dtype=np.float64)
    fin = ~np.isnan(D)
    scale = float(np.nanmax(D))
    if scale > 0:
        D /= scale
    D[~fin] = 2.0
    return D, fin


def solve(instance: dict, time_limit_s: float) -> dict:
    start = _now()
    ctx = _Ctx()
    try:
        n, p = int(instance["n"]), int(instance["p"])
        if p == n:
            return {"medians": list(range(n))}
        budget = float(time_limit_s)
        if not math.isfinite(budget):
            budget = 0.0
        budget = max(0.0, budget)
        deadline = start + budget * 0.965
        D, fin = _matrix(instance["dist"])
        _run(D, fin, p, budget, deadline, ctx)
    except Exception:
        # If an optimisation step raises, do not discard the feasible solution already verified.
        pass
    if ctx.best_S is None:
        try:
            ctx.best_S = _raw_feasible(instance)
        except Exception:
            # Valid input never enters this branch.
            try:
                ctx.best_S = list(range(int(instance["p"])))
            except Exception:
                ctx.best_S = []
    return {"medians": [int(v) for v in ctx.best_S]}


def _sink_classes(fin, n):
    """Return sink-component labels and the vertices of each sink component."""
    nonsink = np.zeros(n, dtype=bool)
    B = max(1, 2000000 // max(1, n))
    for a in range(0, n, B):
        b = min(n, a + B)
        nonsink[a:b] = (fin[a:b] & ~(fin[:, a:b].T)).any(axis=1)
    sink_idx = np.flatnonzero(~nonsink)
    lab = np.full(n, -1, dtype=np.int64)
    if sink_idx.size == 0:
        return lab, []
    first = np.empty(sink_idx.size, dtype=np.int64)
    for a in range(0, sink_idx.size, B):
        b = min(sink_idx.size, a + B)
        first[a:b] = np.argmax(fin[sink_idx[a:b]], axis=1)
    uniq, inv = np.unique(first, return_inverse=True)
    inv = inv.reshape(-1).astype(np.int64)
    lab[sink_idx] = inv
    order = np.argsort(inv, kind="stable")
    sl = inv[order]
    bounds = np.flatnonzero(sl[1:] != sl[:-1]) + 1
    members = np.split(sink_idx[order], bounds)
    return lab, members


def _lazy_greedy(D, seeds, p, CAP, deadline, tol, chunk):
    n = D.shape[0]
    S = [int(v) for v in seeds]
    inS = np.zeros(n, dtype=bool)
    if S:
        inS[S] = True
        d1 = np.empty(n)
        sa = np.array(S, dtype=np.int64)
        for a in range(0, n, chunk):
            b = min(n, a + chunk)
            d1[a:b] = D[a:b, sa].min(axis=1)
    else:
        d1 = np.full(n, CAP)
    g = np.zeros(n)
    for a in range(0, n, chunk):
        if _now() >= deadline:
            S.extend(int(v) for v in np.flatnonzero(~inS)[:p - len(S)])
            return S
        b = min(n, a + chunk)
        T = d1[a:b, None] - D[a:b]
        np.maximum(T, 0.0, out=T)
        g += T.sum(axis=0)
    gl = g.tolist()
    heap = [(-gl[u], u) for u in range(n) if not inS[u]]
    heapq.heapify(heap)
    while len(S) < p and heap:
        if _now() >= deadline:
            rest = sorted(heap)
            for _, u in rest:
                if len(S) >= p:
                    break
                if not inS[u]:
                    S.append(u)
                    inS[u] = True
            break
        negg, u = heapq.heappop(heap)
        col = D[:, u]
        cur = float(np.maximum(d1 - col, 0.0).sum())
        if (not heap) or cur >= -heap[0][0] - tol:
            S.append(u)
            inS[u] = True
            np.minimum(d1, col, out=d1)
        else:
            heapq.heappush(heap, (-cur, u))
    if len(S) < p:
        for u in range(n):
            if len(S) >= p:
                break
            if not inS[u]:
                S.append(u)
                inS[u] = True
    return S


def _elite_add(elite, c, S, E, eps):
    s = np.sort(S)
    for i in range(len(elite)):
        if abs(elite[i][0] - c) <= eps and np.array_equal(elite[i][1], s):
            return
    if len(elite) < E:
        elite.append((c, s))
        return
    wi = 0
    for i in range(1, len(elite)):
        if elite[i][0] > elite[wi][0]:
            wi = i
    if c < elite[wi][0] - eps:
        elite[wi] = (c, s)


class _PM(object):
    def __init__(self, D, p, CAP, lab, members, eps, ctx):
        n = D.shape[0]
        self.ctx = ctx
        self.swap_time = 0.0
        self.rebuild_time = 0.0
        self.D = D
        self.n = n
        self.p = p
        self.CAP = CAP
        self.eps = eps
        self.has_con = (lab is not None) and (members is not None) and len(members) > 0
        self.lab = lab
        self.members = members
        self.chunk = max(1, 2000000 // max(1, n))
        self.S = np.zeros(p, dtype=np.int64)
        self.inS = np.zeros(n, dtype=bool)
        self.c1 = np.zeros(n, dtype=np.int64)
        self.c2 = np.full(n, -1, dtype=np.int64)
        self.d1 = np.zeros(n)
        self.d2 = np.zeros(n)
        self.gain = np.zeros(n)
        self.loss = np.zeros(p)
        self.extra = np.zeros((p, n))
        self.buf = np.empty((p, n))
        self.cnt = np.zeros(len(members) if self.has_con else 1, dtype=np.int64)
        self.nswaps = 0

    def set_solution(self, S):
        started = _now()
        S = np.asarray(S, dtype=np.int64).copy()
        self.S[:] = S
        self.inS[:] = False
        self.inS[S] = True
        if self.has_con:
            ls = self.lab[S]
            ls = ls[ls >= 0]
            self.cnt[:] = np.bincount(ls, minlength=len(self.members))[: len(self.members)]
        self.gain.fill(0.0)
        self.loss.fill(0.0)
        self.extra.fill(0.0)
        n = self.n
        B = self.chunk
        for a in range(0, n, B):
            b = min(n, a + B)
            idx = np.arange(a, b)
            M = self.D[a:b]
            self._top2(idx, M)
            self._contrib(idx, M, True)
        self.nswaps = 0
        self.rebuild_time = max(1e-6, _now() - started)
        if not self.swap_time:
            self.swap_time = self.rebuild_time
        self.record()

    def _top2(self, idx, M):
        DS = M[:, self.S]
        if self.p == 1:
            self.c1[idx] = 0
            self.d1[idx] = DS[:, 0]
            self.c2[idx] = -1
            self.d2[idx] = self.CAP
            return
        part = np.argpartition(DS, 1, axis=1)[:, :2]
        v = np.take_along_axis(DS, part, axis=1)
        sw = v[:, 0] > v[:, 1]
        if sw.any():
            part[sw] = part[sw][:, ::-1]
            v[sw] = v[sw][:, ::-1]
        self.c1[idx] = part[:, 0]
        self.c2[idx] = part[:, 1]
        self.d1[idx] = v[:, 0]
        self.d2[idx] = v[:, 1]

    def _contrib(self, idx, M, add):
        d1 = self.d1[idx]
        d2 = self.d2[idx]
        c1 = self.c1[idx]
        d1c = d1[:, None]
        G = d1c - M
        np.maximum(G, 0.0, out=G)
        gs = G.sum(axis=0)
        E = np.maximum(M, d1c)
        np.subtract(d2[:, None], E, out=E)
        np.maximum(E, 0.0, out=E)
        k = idx.shape[0]
        if k == 1:
            if add:
                self.extra[c1[0]] += E[0]
            else:
                self.extra[c1[0]] -= E[0]
        else:
            order = np.argsort(c1, kind="stable")
            cs = c1[order]
            starts = np.flatnonzero(np.concatenate(([True], cs[1:] != cs[:-1])))
            sums = np.add.reduceat(E[order], starts, axis=0)
            rows = cs[starts]
            if add:
                self.extra[rows] += sums
            else:
                self.extra[rows] -= sums
        lw = np.bincount(c1, weights=d2 - d1, minlength=self.p)
        if add:
            self.gain += gs
            self.loss += lw
        else:
            self.gain -= gs
            self.loss -= lw

    def swap(self, u, r):
        started = _now()
        u = int(u)
        r = int(r)
        old = int(self.S[r])
        Du = self.D[:, u]
        aff = self.c1 == r
        aff |= self.c2 == r
        aff |= Du < self.d2
        A = np.flatnonzero(aff)
        self.S[r] = u
        self.inS[old] = False
        self.inS[u] = True
        if self.has_con:
            lo = int(self.lab[old])
            if lo >= 0:
                self.cnt[lo] -= 1
            lu = int(self.lab[u])
            if lu >= 0:
                self.cnt[lu] += 1
        B = self.chunk
        for a in range(0, A.shape[0], B):
            idx = A[a:a + B]
            M = self.D[idx]
            self._contrib(idx, M, False)
            self._top2(idx, M)
            self._contrib(idx, M, True)
        self.nswaps += 1
        elapsed = _now() - started
        self.swap_time = max(elapsed, 0.8 * self.swap_time + 0.2 * elapsed)
        self.record()

    def feasible_swap(self, u, r):
        if self.inS[u]:
            return False
        if not self.has_con:
            return True
        lo = int(self.lab[self.S[r]])
        if lo >= 0 and self.cnt[lo] == 1 and int(self.lab[u]) != lo:
            return False
        return True

    def cost(self):
        return float(self.d1.sum())

    def feasible(self):
        return not self.has_con or bool((self.cnt > 0).all())

    def record(self):
        cost = self.cost()
        if cost < self.ctx.best_cost and self.feasible():
            self.ctx.best_cost = cost
            self.ctx.best_S = self.S.copy()
        return cost

    def best_move(self):
        buf = self.buf
        np.subtract(self.loss[:, None], self.extra, out=buf)
        col = np.where(self.inS, -np.inf, self.gain)
        np.subtract(buf, col, out=buf)
        if self.has_con:
            labS = self.lab[self.S]
            cm = labS >= 0
            if cm.any():
                cm &= self.cnt[np.where(cm, labS, 0)] == 1
                crit = np.flatnonzero(cm)
                if crit.size:
                    forbid = self.lab[None, :] != labS[crit][:, None]
                    sub = buf[crit]
                    sub[forbid] = np.inf
                    buf[crit] = sub
        flat = int(np.argmin(buf))
        r, u = divmod(flat, self.n)
        return float(buf[r, u]), u, r

    def local_search(self, deadline):
        eps = self.eps
        while True:
            if _now() + self.swap_time >= deadline:
                return False
            val, u, r = self.best_move()
            if not (val < -eps):
                return True
            before = self.cost()
            self.swap(u, r)
            after = self.cost()
            if after >= before or abs((after - before) - val) > 1e-7 * (abs(val) + abs(before) * 1e-3) + eps:
                self.set_solution(self.S.copy())

    def goto(self, target, deadline):
        target = np.asarray(target, dtype=np.int64)
        mask = np.zeros(self.n, dtype=bool)
        mask[target] = True
        R = np.flatnonzero(~mask[self.S]).tolist()
        U = target[~self.inS[target]].tolist()
        if not R:
            return True
        if self.rebuild_time < len(R) * self.swap_time:
            if _now() + self.rebuild_time >= deadline:
                return False
            self.set_solution(target)
            return True
        while R:
            if _now() + self.swap_time >= deadline:
                return False
            selected = None
            for i, r in enumerate(R):
                for j, u in enumerate(U):
                    if self.feasible_swap(u, r):
                        selected = (i, j, r, u)
                        break
                if selected is not None:
                    break
            if selected is None:
                return False
            i, j, r, u = selected
            self.swap(u, r)
            del R[i]
            del U[j]
        return True


def _cost(D, S):
    total = 0.0
    block = max(1, 1000000 // max(1, len(S)))
    for a in range(0, D.shape[0], block):
        total += float(D[a:a + block, S].min(axis=1).sum())
    return total


def _record_candidate(st, S):
    # Candidate solutions come from a construction process that preserves sink-component coverage.
    value = _cost(st.D, S)
    if value < st.ctx.best_cost:
        st.ctx.best_cost = value
        st.ctx.best_S = list(S)
    return value


def _movable(st):
    if not st.has_con:
        return np.arange(st.p)
    result = []
    for r, v in enumerate(st.S):
        label = int(st.lab[v])
        if label < 0 or st.cnt[label] > 1 or len(st.members[label]) > 1:
            result.append(r)
    return np.asarray(result, dtype=np.int64)


def _shake(st, k, rng, deadline):
    applied = 0
    while applied < k and _now() + st.swap_time < deadline:
        movable = _movable(st)
        if movable.size == 0:
            return applied
        r = int(movable[rng.randrange(movable.size)])
        possible = ~st.inS
        if st.has_con:
            label = int(st.lab[st.S[r]])
            if label >= 0 and st.cnt[label] == 1:
                possible = possible & (st.lab == label)
        local = possible & ((st.c1 == r) | (st.c2 == r))
        if local.any() and rng.random() < 0.7:
            choices = np.flatnonzero(local)
        else:
            choices = np.flatnonzero(possible)
        if choices.size == 0:
            return applied
        u = int(choices[rng.randrange(choices.size)])
        st.swap(u, r)
        applied += 1
    return applied


def _recreate(st, k, rng, deadline, guide=None):
    original = st.S.copy()
    movable = _movable(st)
    if movable.size == 0:
        return original.tolist()
    if guide is not None:
        pool = np.union1d(original, guide)
        differences = movable[~np.isin(original[movable], guide)]
        if differences.size:
            movable = differences
    else:
        pool = np.arange(st.n)
    if pool.size <= st.p:
        return original.tolist()
    k = min(k, movable.size)
    r0 = int(movable[rng.randrange(movable.size)])
    if rng.random() < 0.75:
        v = int(original[r0])
        proximity = st.D[v, original[movable]] + st.D[original[movable], v]
        order = np.argsort(proximity, kind="stable")
        removed = movable[order[:k]]
    else:
        removed = np.array(rng.sample(movable.tolist(), k), dtype=np.int64)
    keep = np.ones(st.p, dtype=bool)
    keep[removed] = False
    S = original[keep].tolist()
    used = np.zeros(st.n, dtype=bool)
    used[S] = True
    d = st.D[:, S].min(axis=1) if S else np.full(st.n, st.CAP)
    allowed = np.zeros(st.n, dtype=bool)
    allowed[pool] = True
    allowed[used] = False

    # Forbid re-selecting a removed vertex so that greedy rebuild does not simply return the original.
    for r in removed:
        old = int(original[r])
        label = int(st.lab[old]) if st.has_con else -1
        if label >= 0 and not np.any(st.lab[np.asarray(S, dtype=np.int64)] == label):
            if np.count_nonzero(allowed & (st.lab == label)) <= 1:
                continue
        allowed[old] = False
        break

    # First fill in all missing sink components: coverage is a hard constraint, not compensated by a cost penalty.
    if st.has_con:
        present = np.zeros(len(st.members), dtype=bool)
        if S:
            labels = st.lab[np.asarray(S, dtype=np.int64)]
            present[labels[labels >= 0]] = True
        for label in np.flatnonzero(~present):
            if _now() >= deadline:
                return original.tolist()
            candidates = np.flatnonzero(allowed & (st.lab == label))
            if candidates.size == 0:
                return original.tolist()
            best_gain, best_u = -_INF, int(candidates[0])
            block = max(1, 500000 // st.n)
            for a in range(0, candidates.size, block):
                if _now() >= deadline:
                    return original.tolist()
                cols = candidates[a:a + block]
                gains = np.maximum(d[:, None] - st.D[:, cols], 0.0).sum(axis=0)
                j = int(np.argmax(gains))
                if gains[j] > best_gain:
                    best_gain, best_u = float(gains[j]), int(cols[j])
            S.append(best_u)
            used[best_u] = True
            allowed[best_u] = False
            np.minimum(d, st.D[:, best_u], out=d)

    if len(S) == st.p:
        return S
    candidates = np.flatnonzero(allowed)
    if candidates.size < st.p - len(S):
        return original.tolist()
    # Gain cache of the original state plus deltas of the affected clients, avoiding a full recomputation each time.
    gains = st.gain[candidates].copy()
    changed = np.flatnonzero(d != st.d1)
    block = max(1, 500000 // max(1, candidates.size))
    for a in range(0, changed.size, block):
        if _now() >= deadline:
            return original.tolist()
        rows = changed[a:a + block]
        matrix = st.D[np.ix_(rows, candidates)]
        gains += np.maximum(d[rows, None] - matrix, 0.0).sum(axis=0)
        gains -= np.maximum(st.d1[rows, None] - matrix, 0.0).sum(axis=0)
    heap = [(-float(g), int(u)) for g, u in zip(gains, candidates)]
    heapq.heapify(heap)
    while len(S) < st.p and heap:
        if _now() >= deadline:
            return original.tolist()
        _, u = heapq.heappop(heap)
        gain = float(np.maximum(d - st.D[:, u], 0.0).sum())
        if not heap or gain >= -heap[0][0] - st.eps:
            S.append(u)
            np.minimum(d, st.D[:, u], out=d)
        else:
            heapq.heappush(heap, (-gain, u))
    return S if len(S) == st.p else original.tolist()


def _try_double(st, deadline, cursor):
    # The first step may temporarily worsen, but the second step must make the whole double swap strictly improving.
    st.best_move()
    rows = np.argmin(st.buf, axis=0)
    cols = np.argmin(st.buf, axis=1)
    proposals = {(int(rows[u]), u) for u in range(st.n) if not st.inS[u]}
    proposals.update((r, int(cols[r])) for r in range(st.p))
    proposals = [(float(st.buf[r, u]), r, u) for r, u in proposals
                 if np.isfinite(st.buf[r, u])]
    proposals.sort()
    if not proposals or cursor >= len(proposals):
        return False, cursor
    base_cost = st.cost()
    while cursor < len(proposals) and _now() + 4 * st.swap_time < deadline:
        _, r, u = proposals[cursor]
        cursor += 1
        old = int(st.S[r])
        st.swap(u, r)
        st.best_move()
        # The two outgoing and two incoming vertices must all be distinct; undoing the first step must not pass as an improvement.
        st.buf[r, :] = np.inf
        st.buf[:, old] = np.inf
        flat = int(np.argmin(st.buf))
        r2, u2 = divmod(flat, st.n)
        if st.cost() + float(st.buf[r2, u2]) < base_cost - st.eps:
            old2 = int(st.S[r2])
            st.swap(u2, r2)
            if st.cost() < base_cost - st.eps:
                return True, 0
            st.swap(old2, r2)
        st.swap(old, r)
    return False, cursor


def _enumerate_timed(D, fin, p, deadline, ctx):
    n = D.shape[0]
    combinations = itertools.combinations(range(n), p)
    # This cap only bounds per-batch memory; whether enumeration continues depends solely on the wall clock and on candidates running out.
    block = max(1, 250000 // max(1, n * p))
    seconds_per = 0.0
    while _now() < deadline:
        left = deadline - _now()
        count = block if seconds_per == 0 else min(block, max(1, int(left / (2 * seconds_per))))
        started = _now()
        batch = list(itertools.islice(combinations, count))
        if not batch:
            return True
        choices = np.asarray(batch, dtype=np.int64)
        values = D[:, choices].min(axis=2).sum(axis=0)
        feasible = fin[:, choices].any(axis=2).all(axis=0)
        values[~feasible] = np.inf
        j = int(np.argmin(values))
        if values[j] < ctx.best_cost:
            ctx.best_cost = float(values[j])
            ctx.best_S = choices[j].copy()
        seconds_per = max(1e-9, (_now() - started) / len(batch))
    return False


def _run(D, fin, p, budget, deadline, ctx):
    n = D.shape[0]
    if fin.all():
        lab, members = None, []
    else:
        lab, members = _sink_classes(fin, n)
    colsum = D.sum(axis=0)
    seeds = [int(mem[np.argmin(colsum[mem])]) for mem in members]
    used = np.zeros(n, dtype=bool)
    used[seeds] = True
    fallback = list(seeds)
    for v in np.argsort(colsum, kind="stable"):
        if len(fallback) == p:
            break
        if not used[v]:
            fallback.append(int(v))
            used[v] = True
    ctx.best_S = fallback
    ctx.best_cost = _cost(D, fallback)
    if p == 1:
        eligible = fin.all(axis=0)
        v = int(np.argmin(np.where(eligible, colsum, np.inf)))
        ctx.best_S = [v]
        ctx.best_cost = float(colsum[v])
        return
    if ctx.best_cost == 0.0 or _now() >= deadline:
        return
    count = math.comb(n, p)
    remaining = max(0.0, deadline - _now())
    if count * n * p <= 300000 * remaining:
        done = _enumerate_timed(D, fin, p, deadline, ctx)
        if done or _now() >= deadline:
            return
    mean = float(D[fin].mean())
    eps = max(1e-300, 16 * np.finfo(np.float64).eps * n * mean)
    chunk = max(1, 1000000 // n)
    S = _lazy_greedy(D, seeds, p, 2.0, deadline, eps, chunk)
    value = _cost(D, S)
    if value < ctx.best_cost:
        ctx.best_S, ctx.best_cost = S, value
    if _now() >= deadline:
        return
    st = _PM(D, p, 2.0, lab, members, eps, ctx)
    st.set_solution(ctx.best_S)
    _search(st, budget, deadline, random.Random(918273645))


def _search(st, budget, deadline, rng):
    started = _now()
    complete = st.local_search(deadline)
    epoch_time = max(st.swap_time, _now() - started, 1e-6)
    current = st.S.copy()
    current_cost = st.record()
    if not complete or current_cost <= 0:
        return
    delta, _, _ = st.best_move()
    if not math.isfinite(delta):
        return
    elite = [(current_cost, np.sort(current))]
    stagnation, cursor = 0, 0
    k = min(2, st.p, st.n - st.p)
    rebuild_at = _now() + math.sqrt(max(budget * st.rebuild_time, 1e-12))

    while _now() + max(st.swap_time, st.rebuild_time * 0.1) < deadline:
        iteration_start = _now()
        expected = max(1.0, budget / epoch_time)
        capacity = max(2, int(math.ceil(1 + math.log1p(expected))))
        restart_after = max(2, int(math.ceil(math.sqrt(expected))))
        span = min(st.p, st.n - st.p)
        kmax = min(span, max(1, int(math.ceil(math.sqrt(span * math.log1p(expected))))))
        k = min(k, kmax)

        if _now() >= rebuild_at and _now() + st.rebuild_time < deadline:
            st.set_solution(st.S.copy())
            current_cost = st.cost()
            rebuild_at = _now() + math.sqrt(max(budget * st.rebuild_time, 1e-12))

        if stagnation >= restart_after:
            parent = elite[rng.randrange(len(elite))][1]
            if not st.goto(parent, deadline):
                break
            current, current_cost = st.S.copy(), st.cost()
            stagnation, cursor = 0, 0
            k = kmax

        improved_pair = False
        if span >= 2 and (stagnation == 0 or rng.random() < 0.25):
            allowance = min((deadline - _now()) * 0.15,
                            math.sqrt(max(budget * epoch_time, 1e-12)) * 0.35)
            improved_pair, cursor = _try_double(st, min(deadline, _now() + allowance), cursor)

        if not improved_pair:
            guides = [entry[1] for entry in elite
                      if not np.array_equal(entry[1], np.sort(st.S))]
            guide = None
            if guides and rng.random() < 0.35:
                guide = guides[rng.randrange(len(guides))]
            if guide is not None or rng.random() < 0.7:
                target = _recreate(st, k, rng, deadline - st.swap_time, guide)
                _record_candidate(st, target)
                if not st.goto(target, deadline):
                    break
            else:
                _shake(st, k, rng, deadline)

        complete = st.local_search(deadline)
        cost = st.record()
        _elite_add(elite, cost, st.S, capacity, st.eps)
        if not complete or st.ctx.best_cost <= 0:
            break
        if cost < current_cost - st.eps:
            current, current_cost = st.S.copy(), cost
            stagnation, cursor = 0, 0
            k = min(2, kmax)
        elif cost <= current_cost + st.eps:
            changed = not np.array_equal(np.sort(st.S), np.sort(current))
            current, current_cost = st.S.copy(), cost
            if changed:
                cursor = 0
            stagnation += 1
            k = 1 if k >= kmax else k + 1
        else:
            if not st.goto(current, deadline):
                break
            stagnation += 1
            k = 1 if k >= kmax else k + 1
        elapsed = _now() - iteration_start
        epoch_time = max(st.swap_time, 0.8 * epoch_time + 0.2 * elapsed, 1e-6)
