"""Coverage-preserving union recombination and path-relinking median search
Improves the location set with lazy greedy and incremental fast swaps, fusing coverage-preserving union pruning, path relinking, an elite pool and adaptive local perturbation in a wall-clock-driven search.
"""
import heapq
import itertools
import math
import random
import time

import numpy as np


class _Expired(Exception):
    pass


def _clock():
    return time.perf_counter()


def _check(deadline):
    if _clock() >= deadline:
        raise _Expired()


class _Best:
    def __init__(self):
        self.S = None
        self.cost = math.inf

    def record(self, S, cost):
        if cost < self.cost:
            self.S = [int(v) for v in S]
            self.cost = float(cost)


def _fallback(instance):
    # In the shortest-path closure, the reachable set of a sink strongly connected component is the component itself.
    n, p, D = instance['n'], instance['p'], instance['dist']
    marked = [False] * n
    S = []
    for i in range(n):
        if marked[i]:
            continue
        reach = [j for j, x in enumerate(D[i])
                 if x is not None and x != math.inf]
        if all(D[j][i] is not None and D[j][i] != math.inf for j in reach):
            S.append(i)
            for j in reach:
                marked[j] = True
    used = set(S)
    for j in range(n):
        if len(S) == p:
            break
        if j not in used:
            S.append(j)
    return S


def solve(instance: dict, time_limit_s: float) -> dict:
    start = _clock()
    best = _Best()
    try:
        n, p = int(instance['n']), int(instance['p'])
        if p == n:
            return {'medians': list(range(n))}
        budget = float(time_limit_s)
        if not math.isfinite(budget):
            budget = 1.0
        budget = max(0.0, budget)
        deadline = start + budget * 0.96 - min(0.02, budget * 0.01)
        _run(instance, p, deadline, budget, best)
    except Exception:
        # If the search is aborted or a numerical operation fails, keep the most recent complete feasible solution.
        pass
    if best.S is not None:
        return {'medians': best.S}
    # For valid input this branch depends only on reachability, not on float conversion or search success.
    return {'medians': _fallback(instance)}


def _sinks(finite):
    n = len(finite)
    labels = np.full(n, -1, dtype=np.int64)
    members = []
    if bool(finite.all()):
        return labels, members
    nonsink = np.empty(n, dtype=bool)
    block = max(1, 1000000 // n)
    for a in range(0, n, block):
        b = min(n, a + block)
        nonsink[a:b] = (finite[a:b] & ~finite[:, a:b].T).any(axis=1)
    for v in np.flatnonzero(~nonsink):
        if labels[v] < 0:
            group = np.flatnonzero(finite[v])
            labels[group] = len(members)
            members.append(group)
    return labels, members


def _cost(D, S, deadline):
    total = 0.0
    block = max(1, 1000000 // len(S))
    for a in range(0, len(D), block):
        _check(deadline)
        total += float(D[a:a + block, S].min(axis=1).sum())
    return total


def _greedy(D, seeds, p, colsum, deadline):
    n = len(D)
    S = list(seeds)
    used = np.zeros(n, dtype=bool)
    used[S] = True
    block = max(1, 1000000 // n)
    if S:
        d = D[:, S].min(axis=1)
        gain = np.zeros(n)
        for a in range(0, n, block):
            _check(deadline)
            gain += np.maximum(d[a:a + block, None] - D[a:a + block], 0).sum(axis=0)
    else:
        d = np.full(n, 2.0)
        gain = 2.0 * n - colsum
    heap = [(-float(gain[v]), v) for v in range(n) if not used[v]]
    heapq.heapify(heap)
    while len(S) < p:
        _check(deadline)
        old, v = heapq.heappop(heap)
        g = float(np.maximum(d - D[:, v], 0.0).sum())
        if not heap or g >= -heap[0][0]:
            S.append(v)
            used[v] = True
            np.minimum(d, D[:, v], out=d)
        else:
            heapq.heappush(heap, (-g, v))
    return S, float(d.sum())


def _enumerate(D, p, deadline, best):
    n = len(D)
    combinations = itertools.combinations(range(n), p)
    block = max(1, 1000000 // (n * p))
    while _clock() < deadline:
        batch = list(itertools.islice(combinations, block))
        if not batch:
            return True
        C = np.asarray(batch, dtype=np.int64)
        mins = D[:, C].min(axis=2)
        costs = mins.sum(axis=0)
        costs[(mins >= 2.0).any(axis=0)] = np.inf
        q = int(np.argmin(costs))
        best.record(C[q], costs[q])
        if best.cost == 0.0:
            return True
    return False


class _Search:
    def __init__(self, D, p, labels, members, deadline, best, eps):
        self.D, self.p = D, p
        self.n = len(D)
        self.lab, self.members = labels, members
        self.deadline, self.best, self.eps = deadline, best, eps
        self.block = max(1, 1000000 // self.n)
        self.S = np.zeros(p, dtype=np.int64)
        self.used = np.zeros(self.n, dtype=bool)
        self.c1 = np.zeros(self.n, dtype=np.int64)
        self.c2 = np.zeros(self.n, dtype=np.int64)
        self.d1 = np.zeros(self.n)
        self.d2 = np.zeros(self.n)
        self.gain = np.zeros(self.n)
        self.loss = np.zeros(p)
        self.extra = np.zeros((p, self.n))
        self.buf = np.empty((p, self.n))
        self.cnt = np.zeros(len(members), dtype=np.int64)
        self.updates = 0
        self.build_time = 0.0
        self.cost = math.inf

    def nearest(self, rows, M):
        DS = M[:, self.S]
        part = np.argpartition(DS, 1, axis=1)[:, :2]
        val = np.take_along_axis(DS, part, axis=1)
        rev = val[:, 0] > val[:, 1]
        if rev.any():
            part[rev] = part[rev, ::-1]
            val[rev] = val[rev, ::-1]
        self.c1[rows], self.c2[rows] = part[:, 0], part[:, 1]
        self.d1[rows], self.d2[rows] = val[:, 0], val[:, 1]

    def contribute(self, rows, M, sign):
        d1, d2, c1 = self.d1[rows], self.d2[rows], self.c1[rows]
        G = d1[:, None] - M
        np.maximum(G, 0.0, out=G)
        self.gain += sign * G.sum(axis=0)
        del G
        E = np.maximum(M, d1[:, None])
        np.subtract(d2[:, None], E, out=E)
        np.maximum(E, 0.0, out=E)
        order = np.argsort(c1, kind='stable')
        groups = c1[order]
        starts = np.flatnonzero(np.r_[True, groups[1:] != groups[:-1]])
        self.extra[groups[starts]] += sign * np.add.reduceat(E[order], starts, axis=0)
        self.loss += sign * np.bincount(c1, weights=d2 - d1, minlength=self.p)

    def set_solution(self, S):
        _check(self.deadline)
        start = _clock()
        self.S[:] = S
        self.used[:] = False
        self.used[self.S] = True
        sid = self.lab[self.S]
        self.cnt[:] = np.bincount(sid[sid >= 0], minlength=len(self.members))
        for a in range(0, self.n, self.block):
            _check(self.deadline)
            b = min(self.n, a + self.block)
            self.nearest(np.arange(a, b), self.D[a:b])
        self.cost = float(self.d1.sum())
        self.best.record(self.S, self.cost)
        self.gain.fill(0.0)
        self.loss.fill(0.0)
        self.extra.fill(0.0)
        for a in range(0, self.n, self.block):
            _check(self.deadline)
            b = min(self.n, a + self.block)
            self.contribute(np.arange(a, b), self.D[a:b], 1.0)
        self.updates = 0
        self.build_time = _clock() - start

    def mask(self, buf, rows=None, columns=None):
        if not self.members:
            return
        if rows is None:
            rows = np.arange(self.p)
        if columns is None:
            columns = np.arange(self.n)
        labs = self.lab[self.S[rows]]
        critical = labs >= 0
        critical &= self.cnt[np.maximum(labs, 0)] == 1
        forbid = critical[:, None] & (labs[:, None] != self.lab[columns][None, :])
        buf[forbid] = np.inf

    def swap(self, u, r):
        _check(self.deadline)
        old = int(self.S[r])
        Du = self.D[:, u]
        affected = np.flatnonzero((self.c1 == r) | (self.c2 == r) | (Du < self.d2))
        newd = np.minimum(np.where(self.c1 == r, self.d2, self.d1), Du)
        self.S[r] = u
        self.used[old], self.used[u] = False, True
        lo, lu = int(self.lab[old]), int(self.lab[u])
        if lo >= 0:
            self.cnt[lo] -= 1
        if lu >= 0:
            self.cnt[lu] += 1
        # Record the complete location set first; even if the incremental cache is cut off by the time limit, the returned solution is unaffected.
        self.best.record(self.S, float(newd.sum()))
        for a in range(0, len(affected), self.block):
            _check(self.deadline)
            rows = affected[a:a + self.block]
            M = self.D[rows]
            self.contribute(rows, M, -1.0)
            self.nearest(rows, M)
            self.contribute(rows, M, 1.0)
        self.cost = float(self.d1.sum())
        self.updates += 1

    def local(self):
        rebuilt = False
        while _clock() < self.deadline:
            np.subtract(self.loss[:, None], self.extra, out=self.buf)
            self.buf -= self.gain
            self.buf[:, self.S] = np.inf
            self.mask(self.buf)
            flat = int(np.argmin(self.buf))
            r, u = divmod(flat, self.n)
            delta = float(self.buf[r, u])
            if delta >= -self.eps:
                if self.updates > self.n + self.p and not rebuilt:
                    self.set_solution(self.S.copy())
                    rebuilt = True
                    continue
                return
            # Verify the incremental evaluation with true nearest distances, preventing wrong acceptance from accumulated floating-point error.
            nd = np.minimum(np.where(self.c1 == r, self.d2, self.d1), self.D[:, u])
            c = float(nd.sum())
            if c < self.cost - self.eps:
                self.swap(u, r)
                rebuilt = False
                if self.best.cost == 0.0:
                    return
            elif not rebuilt:
                self.set_solution(self.S.copy())
                rebuilt = True
            else:
                return

    def goto(self, target):
        _check(self.deadline)
        target = np.asarray(target, dtype=np.int64)
        mark = np.zeros(self.n, dtype=bool)
        mark[target] = True
        R = list(np.flatnonzero(~mark[self.S]))
        U = list(target[~self.used[target]])
        if not R:
            return
        if len(R) > max(1, math.isqrt(self.p)):
            self.set_solution(target)
            return
        while R:
            _check(self.deadline)
            chosen = None
            for i, r in enumerate(R):
                lab = self.lab[self.S[r]]
                for j, u in enumerate(U):
                    if lab < 0 or self.cnt[lab] > 1 or self.lab[u] == lab:
                        chosen = (i, j, r, u)
                        break
                if chosen is not None:
                    break
            if chosen is None:
                self.set_solution(target)
                return
            i, j, r, u = chosen
            self.swap(int(u), int(r))
            R.pop(i)
            U.pop(j)

    def shake(self, depth, rng):
        anchor = int(self.S[rng.randrange(self.p)])
        proximity = self.D[anchor, self.S] + self.D[self.S, anchor]
        width = min(self.p, max(1, 2 * depth))
        near = np.argpartition(proximity, width - 1)[:width]
        done = 0
        while done < depth and _clock() < self.deadline:
            slots = np.ones(self.p, dtype=bool)
            if self.members:
                for r, v in enumerate(self.S):
                    q = int(self.lab[v])
                    if q >= 0 and self.cnt[q] == 1 and len(self.members[q]) == 1:
                        slots[r] = False
            eligible = np.flatnonzero(slots)
            if eligible.size == 0:
                return done
            local = near[slots[near]]
            pool = local if local.size and rng.random() < 0.7 else eligible
            r = int(pool[rng.randrange(len(pool))])
            q = int(self.lab[self.S[r]])
            cand = np.flatnonzero(~self.used)
            if q >= 0 and self.cnt[q] == 1:
                cand = cand[self.lab[cand] == q]
            roll = rng.random()
            if roll < 0.5:
                cluster = cand[(self.c1[cand] == r) | (self.c2[cand] == r)]
                if cluster.size:
                    cand = cluster
            if roll < 0.8:
                weights = np.cumsum(self.d1[cand])
                if weights[-1] > 0:
                    j = min(len(cand) - 1, int(np.searchsorted(weights, rng.random() * weights[-1])))
                else:
                    j = rng.randrange(len(cand))
            else:
                j = rng.randrange(len(cand))
            self.swap(int(cand[j]), r)
            done += 1
        return done

    def union_drop(self, guide, strength, rng):
        # Union recombination also injects a few unselected, poorly served candidates to avoid elite-pool homogenisation.
        U = np.union1d(self.S, guide).tolist()
        available = np.ones(self.n, dtype=bool)
        available[U] = False
        added = 0
        while added < strength and available.any():
            _check(self.deadline)
            candidates = np.flatnonzero(available)
            weights = np.cumsum(self.d1[candidates])
            if rng.random() < 0.5 and weights[-1] > 0.0:
                k = min(len(candidates) - 1, int(np.searchsorted(weights, rng.random() * weights[-1])))
            else:
                k = rng.randrange(len(candidates))
            v = int(candidates[k])
            U.append(v)
            available[v] = False
            added += 1
        U = np.asarray(U, dtype=np.int64)
        q = len(U)
        if q == self.p:
            return U
        DU = self.D[:, U]
        order = np.argpartition(DU, 1, axis=1)[:, :2]
        ar = np.arange(self.n)
        c1, c2 = order[:, 0].copy(), order[:, 1].copy()
        d1, d2 = DU[ar, c1], DU[ar, c2]
        loss = np.bincount(c1, weights=d2 - d1, minlength=q)
        alive = np.ones(q, dtype=bool)
        lab = self.lab[U]
        cnt = np.bincount(lab[lab >= 0], minlength=len(self.members))
        count = q
        rowblock = max(1, 1000000 // q)
        while count > self.p:
            _check(self.deadline)
            score = np.where(alive, loss, np.inf)
            if self.members:
                locked = (lab >= 0) & (cnt[np.maximum(lab, 0)] == 1)
                score[locked] = np.inf
            j = int(np.argmin(score))
            if rng.random() < 0.2 and count > self.p + 1:
                options = np.flatnonzero(np.isfinite(score))
                width = min(len(options), 1 + math.isqrt(strength))
                top = options[np.argpartition(score[options], width - 1)[:width]]
                j = int(top[rng.randrange(width)])
            alive[j] = False
            if lab[j] >= 0:
                cnt[lab[j]] -= 1
            count -= 1
            affected = np.flatnonzero((c1 == j) | (c2 == j))
            for a in range(0, len(affected), rowblock):
                _check(self.deadline)
                rows = affected[a:a + rowblock]
                loss -= np.bincount(c1[rows], weights=d2[rows] - d1[rows], minlength=q)
                M = DU[rows]
                M[:, ~alive] = np.inf
                part = np.argpartition(M, 1, axis=1)[:, :2]
                idx = np.arange(len(rows))
                c1[rows], c2[rows] = part[:, 0], part[:, 1]
                d1[rows], d2[rows] = M[idx, part[:, 0]], M[idx, part[:, 1]]
                loss += np.bincount(c1[rows], weights=d2[rows] - d1[rows], minlength=q)
        child = U[alive]
        self.best.record(child, float(d1.sum()))
        return child

    def relink(self, guide):
        # The offspring obtained by pruning becomes the path target; all intermediate swaps keep reusing the fast-swap cache.
        mark = np.zeros(self.n, dtype=bool)
        mark[guide] = True
        R = list(np.flatnonzero(~mark[self.S]))
        U = list(guide[~self.used[guide]])
        best_path, best_cost = None, math.inf
        while R and _clock() < self.deadline:
            Ra, Ua = np.asarray(R, dtype=np.int64), np.asarray(U, dtype=np.int64)
            delta = self.loss[Ra, None] - self.extra[np.ix_(Ra, Ua)] - self.gain[Ua]
            self.mask(delta, Ra, Ua)
            k = int(np.argmin(delta))
            i, j = divmod(k, len(U))
            if not np.isfinite(delta[i, j]):
                break
            self.swap(int(U[j]), int(R[i]))
            R.pop(i)
            U.pop(j)
            if self.cost < best_cost:
                best_cost, best_path = self.cost, self.S.copy()
        if best_path is not None:
            self.goto(best_path)
        return best_path is not None


def _pool_add(pool, S, cost, capacity):
    key = tuple(sorted(int(v) for v in S))
    if any(key == item[1] for item in pool):
        return
    pool.append((float(cost), key))
    while len(pool) > capacity:
        pool.sort(key=lambda item: item[0])
        # Keep the best solution; the remaining entries balance quality ranking with nearest-neighbour similarity.
        values = []
        for i in range(1, len(pool)):
            A = set(pool[i][1])
            distance = min(len(A.difference(item[1])) for j, item in enumerate(pool) if j != i)
            values.append((i / (len(pool) - 1) + 1.0 / (1 + distance), i))
        pool.pop(max(values)[1])


def _improve(st, budget, rng):
    st.local()
    if st.best.cost == 0.0:
        return
    current, current_cost = st.S.copy(), st.cost
    pool = []
    step_time = max(st.build_time, 1e-6)
    depth, stalled = 1, 0
    while _clock() < st.deadline:
        start = _clock()
        remaining = st.deadline - start
        work = max(1.0, budget / max(step_time, 1e-6))
        depth_max = min(st.p, st.n - st.p, 1 + math.ceil(math.sqrt(math.log2(2.0 + work))))
        capacity = max(2, math.ceil(work ** 0.25))
        stall_limit = max(2, math.ceil(math.sqrt(work)))
        _pool_add(pool, st.S, st.cost, capacity)
        if stalled >= stall_limit:
            # One fusion chain: elite union -> coverage-preserving pruning -> target-guided relinking -> fast descent.
            distances = [len(set(current).difference(item[1])) for item in pool]
            weights = [1.0 + d for d in distances]
            guide = np.asarray(rng.choices(pool, weights=weights, k=1)[0][1], dtype=np.int64)
            child = st.union_drop(guide, depth_max, rng)
            moved = st.relink(child)
            if not moved and not st.shake(depth_max, rng):
                return
            st.local()
            current, current_cost = st.S.copy(), st.cost
            depth, stalled = 1, 0
            _pool_add(pool, st.S, st.cost, capacity)
        else:
            if not st.shake(min(depth, depth_max), rng):
                return
            st.local()
            _pool_add(pool, st.S, st.cost, capacity)
            if st.cost < current_cost - st.eps:
                current, current_cost = st.S.copy(), st.cost
                depth, stalled = 1, 0
            else:
                if st.cost <= current_cost + st.eps:
                    current, current_cost = st.S.copy(), st.cost
                else:
                    st.goto(current)
                stalled += 1
                depth = depth + 1 if depth < depth_max else 1
        if st.best.cost == 0.0:
            return
        elapsed = _clock() - start
        step_time = 0.8 * step_time + 0.2 * elapsed


def _run(instance, p, deadline, budget, best):
    D = np.array(instance['dist'], dtype=np.float64)
    D[np.isnan(D)] = np.inf
    finite = np.isfinite(D)
    n = len(D)
    lab, members = _sinks(finite)
    # First build a purely structural feasible fallback; no later interruption can return an uncovered solution.
    seeds = [int(group[0]) for group in members]
    fallback = list(seeds)
    taken = set(fallback)
    for v in range(n):
        if len(fallback) == p:
            break
        if v not in taken:
            fallback.append(v)
    best.S = fallback
    _check(deadline)
    max_value = float(D[finite].max())
    if max_value == 0.0:
        best.cost = 0.0
        return
    # Scaling is only for floating-point arithmetic; reachability always comes from the original finite flags, never guessed by a numerical threshold.
    D /= max_value
    D[~finite] = 2.0
    colsum = D.sum(axis=0)
    if p == 1:
        allowed = finite.all(axis=0)
        colsum[~allowed] = np.inf
        best.record([int(np.argmin(colsum))], float(colsum.min()))
        return
    seeds = [int(group[np.argmin(colsum[group])]) for group in members]
    fallback = list(seeds)
    taken = set(fallback)
    for v in np.argsort(colsum):
        if len(fallback) == p:
            break
        if int(v) not in taken:
            fallback.append(int(v))
    best.S = fallback
    best.cost = _cost(D, fallback, deadline)
    remaining = max(0.0, deadline - _clock())
    count = math.comb(n, p)
    if count <= int(remaining * 200000.0 / max(1, n * p)):
        if _enumerate(D, p, deadline, best):
            return
        _check(deadline)
    S, cost = _greedy(D, seeds, p, colsum, deadline)
    best.record(S, cost)
    _check(deadline)
    # Tolerance scales with distance magnitude; unreachable placeholder values are not treated as the objective scale.
    mean = float(D[finite].mean())
    eps = max(1e-300, 32.0 * np.finfo(np.float64).eps * mean)
    st = _Search(D, p, lab, members, deadline, best, eps)
    st.set_solution(best.S)
    rng = random.Random(918273645 + 7919 * n + 104729 * p)
    _improve(st, budget, rng)
