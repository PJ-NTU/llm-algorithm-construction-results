"""Hard-coverage incremental swap and elite relinking search
Guarantees coverage via sink strongly connected components, searching with direct-loss incremental swaps, local multi-point perturbation, destroy-repair and elite path relinking, and enumerating small search spaces exhaustively.
"""
import heapq
import itertools
import math
import random
import time

import numpy as np


_clock = time.perf_counter


def _fallback(instance):
    """In the shortest-path closure, taking one vertex per sink strongly connected component guarantees coverage."""
    n, p, rows = instance["n"], instance["p"], instance["dist"]
    marked = [False] * n
    selected = []
    for i in range(n):
        if marked[i]:
            continue
        reach = [j for j, v in enumerate(rows[i]) if v is not None]
        mutual = [j for j in reach if rows[j][i] is not None]
        for j in mutual:
            marked[j] = True
        if len(mutual) == len(reach):
            selected.append(i)
    used = set(selected)
    for j in range(n):
        if len(selected) == p:
            break
        if j not in used:
            selected.append(j)
    return selected


def solve(instance: dict, time_limit_s: float) -> dict:
    start = _clock()
    saved = {"S": None, "cost": math.inf}
    try:
        n, p = int(instance["n"]), int(instance["p"])
        if p == n:
            return {"medians": list(range(n))}
        budget = max(0.0, float(time_limit_s))
        if not math.isfinite(budget):
            budget = 1.0
        _run(instance, start + 0.97 * budget, budget, saved)
    except Exception:
        # If the optimiser raises, fall back only to the saved feasible solution, never to the first p vertices without a coverage check.
        pass
    if saved["S"] is None:
        saved["S"] = _fallback(instance)
    return {"medians": [int(v) for v in saved["S"]]}


def _sinks(finite):
    n = len(finite)
    labels = np.full(n, -1, dtype=np.int64)
    if finite.all():
        return labels, []
    step = max(1, 524288 // n)
    nonsink = np.empty(n, dtype=bool)
    for a in range(0, n, step):
        b = min(n, a + step)
        nonsink[a:b] = (finite[a:b] & ~finite[:, a:b].T).any(axis=1)
    members = []
    for i in np.flatnonzero(~nonsink):
        if labels[i] < 0:
            group = np.flatnonzero(finite[i])
            labels[group] = len(members)
            members.append(group)
    return labels, members


def _cost(D, S):
    return float(D[:, S].min(axis=1).sum())


def _greedy(D, seeds, p, labels, members, forbidden, deadline):
    """First fill in the required sink components, then add vertices by lazy marginal gain. On timeout, no partial result is committed."""
    n = len(D)
    S = list(map(int, seeds))
    chosen = np.zeros(n, dtype=bool)
    chosen[S] = True
    nearest = D[:, S].min(axis=1) if S else np.full(n, 2.0)
    blocked = np.zeros(n, dtype=bool)
    if forbidden is not None:
        blocked[forbidden] = True
    for group in members:
        if chosen[group].any():
            continue
        if _clock() >= deadline:
            return None
        cand = group[~blocked[group]]
        if cand.size == 0:
            cand = group
        values = np.minimum(nearest[:, None], D[:, cand]).sum(axis=0)
        u = int(cand[int(np.argmin(values))])
        S.append(u)
        chosen[u] = True
        np.minimum(nearest, D[:, u], out=nearest)
    if len(S) == p:
        return S
    # Forbidding a return to the original vertex is only a perturbation device; it is lifted when insufficient and does not affect the cardinality or coverage constraints.
    allowed = ~chosen & ~blocked
    if int(allowed.sum()) < p - len(S):
        allowed = ~chosen
    gain = np.zeros(n)
    step = max(1, 524288 // n)
    for a in range(0, n, step):
        if _clock() >= deadline:
            return None
        gain += np.maximum(nearest[a:a + step, None] - D[a:a + step], 0).sum(axis=0)
    heap = [(-float(gain[u]), int(u)) for u in np.flatnonzero(allowed)]
    heapq.heapify(heap)
    while len(S) < p:
        if _clock() >= deadline:
            return None
        _, u = heapq.heappop(heap)
        g = float(np.maximum(nearest - D[:, u], 0).sum())
        if not heap or g >= -heap[0][0] - 1e-14 * max(g, 1e-300):
            S.append(u)
            np.minimum(nearest, D[:, u], out=nearest)
        else:
            heapq.heappush(heap, (-g, u))
    return S


class _Search:
    def __init__(self, D, p, labels, members, deadline, saved):
        self.D, self.n, self.p = D, len(D), p
        self.labels, self.members = labels, members
        self.deadline, self.saved = deadline, saved
        self.step = max(1, 524288 // self.n)
        self.S = None
        self.open = np.zeros(self.n, dtype=bool)
        self.counts = np.zeros(len(members), dtype=np.int64)
        self.c1 = np.zeros(self.n, dtype=np.int64)
        self.c2 = np.zeros(self.n, dtype=np.int64)
        self.d1 = np.zeros(self.n)
        self.d2 = np.zeros(self.n)
        self.gain = np.zeros(self.n)
        self.penalty = np.zeros((p, self.n))
        self.buffer = np.empty((p, self.n))
        self.ready = False
        self.cost = math.inf
        self.work_time = 0.0

    def save(self, S, d):
        value = float(d.sum())
        if value < self.saved["cost"]:
            self.saved["cost"] = value
            self.saved["S"] = S.copy()
        return value

    def metadata(self, S):
        self.S = np.asarray(S, dtype=np.int64).copy()
        self.open[:] = False
        self.open[self.S] = True
        if self.members:
            lab = self.labels[self.S]
            self.counts[:] = np.bincount(lab[lab >= 0], minlength=len(self.members))

    def nearest(self, S, indices=None):
        if indices is None:
            indices = np.arange(self.n)
        m = len(indices)
        c1, c2 = np.empty(m, dtype=np.int64), np.empty(m, dtype=np.int64)
        d1, d2 = np.empty(m), np.empty(m)
        step = max(1, 524288 // self.p)
        for a in range(0, m, step):
            if _clock() >= self.deadline:
                return None
            ids = indices[a:a + step]
            matrix = self.D[np.ix_(ids, S)]
            part = np.argpartition(matrix, 1, axis=1)[:, :2]
            vals = np.take_along_axis(matrix, part, axis=1)
            rev = vals[:, 0] > vals[:, 1]
            part[rev] = part[rev, ::-1]
            vals[rev] = vals[rev, ::-1]
            c1[a:a + step], c2[a:a + step] = part[:, 0], part[:, 1]
            d1[a:a + step], d2[a:a + step] = vals[:, 0], vals[:, 1]
        return c1, c2, d1, d2

    def contribute(self, indices, sign):
        for a in range(0, len(indices), self.step):
            if _clock() >= self.deadline:
                return False
            ids = indices[a:a + self.step]
            M = self.D[ids]
            first, second = self.d1[ids, None], self.d2[ids, None]
            G = np.maximum(first - M, 0.0)
            gs = G.sum(axis=0)
            del G
            # Compute the removal loss directly, avoiding computing a huge loss and then subtracting a nearly equal extra.
            np.minimum(M, second, out=M)
            M -= first
            np.maximum(M, 0.0, out=M)
            owner = self.c1[ids]
            order = np.argsort(owner, kind="stable")
            own = owner[order]
            starts = np.flatnonzero(np.r_[True, own[1:] != own[:-1]])
            rows = own[starts]
            sums = np.add.reduceat(M[order], starts, axis=0)
            if sign > 0:
                self.gain += gs
                self.penalty[rows] += sums
            else:
                self.gain -= gs
                self.penalty[rows] -= sums
        return True

    def build(self, S):
        begin = _clock()
        data = self.nearest(np.asarray(S, dtype=np.int64))
        if data is None:
            self.ready = False
            return False
        self.metadata(S)
        self.c1, self.c2, self.d1, self.d2 = data
        self.cost = self.save(self.S, self.d1)
        self.gain.fill(0)
        self.penalty.fill(0)
        self.ready = self.contribute(np.arange(self.n), 1)
        self.work_time = max(self.work_time, _clock() - begin)
        return self.ready

    def apply(self, S, data):
        """A whole batch swap updates the affected clients' contributions only once."""
        begin = _clock()
        c1, c2, d1, d2 = data
        affected = np.flatnonzero((self.c1 != c1) | (self.d1 != d1) | (self.d2 != d2))
        self.ready = False
        self.metadata(S)
        self.cost = self.save(self.S, d1)
        if not self.contribute(affected, -1):
            return False
        self.c1, self.c2, self.d1, self.d2 = data
        self.ready = self.contribute(affected, 1)
        self.work_time = max(0.9 * self.work_time, _clock() - begin)
        return self.ready

    def move(self, u, r):
        S = self.S.copy()
        S[r] = u
        c1, c2 = self.c1.copy(), self.c2.copy()
        d1, d2 = self.d1.copy(), self.d2.copy()
        ruined = (c1 == r) | (c2 == r)
        ids = np.flatnonzero(ruined)
        if ids.size:
            sub = self.nearest(S, ids)
            if sub is None:
                self.ready = False
                return False
            c1[ids], c2[ids], d1[ids], d2[ids] = sub
        col = self.D[:, u]
        improve = ~ruined & (col < d1)
        c2[improve], d2[improve] = c1[improve], d1[improve]
        c1[improve], d1[improve] = r, col[improve]
        second = ~ruined & ~improve & (col < d2)
        c2[second], d2[second] = r, col[second]
        return self.apply(S, (c1, c2, d1, d2))

    def goto(self, target):
        target = np.asarray(target, dtype=np.int64)
        target_mask = np.zeros(self.n, dtype=bool)
        target_mask[target] = True
        slots = np.flatnonzero(~target_mask[self.S])
        if slots.size == 0:
            return True
        S = self.S.copy()
        S[slots] = target[~self.open[target]]
        data = self.nearest(S)
        if data is None:
            self.ready = False
            return False
        return self.apply(S, data)

    def scores(self):
        np.subtract(self.penalty, self.gain[None, :], out=self.buffer)
        self.buffer[:, self.S] = np.inf
        if self.members:
            labs = self.labels[self.S]
            for r in np.flatnonzero(labs >= 0):
                if self.counts[labs[r]] == 1:
                    self.buffer[r, self.labels != labs[r]] = np.inf
        return self.buffer

    def local(self):
        rebuilt = False
        while self.ready and _clock() < self.deadline:
            scores = self.scores()
            r, u = divmod(int(np.argmin(scores)), self.n)
            tolerance = 1e-12 * max(self.cost, 1e-300)
            if scores[r, u] >= -tolerance:
                return True
            # Each descent is checked with true nearest distances, rejecting false improvements caused by accumulated error.
            nd = np.minimum(self.d1, self.D[:, u])
            ids = self.c1 == r
            nd[ids] = np.minimum(self.d2[ids], self.D[ids, u])
            if float(nd.sum()) >= self.cost - tolerance:
                if rebuilt or not self.build(self.S):
                    return True
                rebuilt = True
                continue
            if not self.move(u, r):
                return False
            rebuilt = False
            if self.cost == 0.0:
                return True
        return False

    def shake(self, k, rng):
        """Concentrated perturbation of related positions: first produce a complete feasible set, then merge the updates."""
        S = self.S.copy()
        counts = self.counts.copy()
        available = ~self.open.copy()
        slots = list(range(self.p))
        root = rng.randrange(self.p)
        near = self.D[S[root], S] + self.D[S, S[root]]
        order = np.argsort(near, kind="stable").tolist()
        locality = rng.random() < 0.8
        changed = 0
        while slots and changed < k and _clock() < self.deadline:
            if locality:
                width = min(len(order), max(1, 2 * k))
                r = order.pop(rng.randrange(width))
                slots.remove(r)
            else:
                r = slots.pop(rng.randrange(len(slots)))
            oldlab = int(self.labels[S[r]])
            allowed = available.copy()
            if oldlab >= 0 and counts[oldlab] == 1:
                allowed &= self.labels == oldlab
            cand = np.flatnonzero(allowed)
            if not cand.size:
                continue
            cluster = cand[self.c1[cand] == r]
            choice = rng.random()
            if locality and cluster.size and choice < 0.65:
                u = int(cluster[rng.randrange(cluster.size)])
            elif choice < 0.85:
                values = self.penalty[r, cand] - self.gain[cand]
                width = min(cand.size, max(2, k + 1))
                top = np.argpartition(values, width - 1)[:width]
                u = int(cand[top[rng.randrange(width)]])
            else:
                u = int(cand[rng.randrange(cand.size)])
            S[r] = u
            available[u] = False
            if oldlab >= 0:
                counts[oldlab] -= 1
            newlab = int(self.labels[u])
            if newlab >= 0:
                counts[newlab] += 1
            changed += 1
        if not changed:
            return False
        data = self.nearest(S)
        if data is None:
            self.ready = False
            return False
        return self.apply(S, data)

    def ruin(self, k, rng, deadline):
        root = rng.randrange(self.p)
        ds = self.D[self.S[root], self.S] + self.D[self.S, self.S[root]]
        near = np.argsort(ds, kind="stable")
        width = min(self.p, max(k, 2 * k))
        slots = rng.sample(near[:width].tolist(), k)
        keep = np.ones(self.p, dtype=bool)
        keep[slots] = False
        S = _greedy(self.D, self.S[keep], self.p, self.labels, self.members,
                    self.S[~keep], deadline)
        return S is not None and self.goto(S)

    def relink(self, guide, deadline):
        mask = np.zeros(self.n, dtype=bool)
        mask[guide] = True
        remove = np.flatnonzero(~mask[self.S]).tolist()
        insert = np.asarray(guide)[~self.open[guide]].tolist()
        if len(remove) < 2:
            return False
        best, best_cost = self.S.copy(), self.cost
        while remove and self.ready and _clock() < deadline:
            R, U = np.asarray(remove), np.asarray(insert)
            scores = self.penalty[np.ix_(R, U)] - self.gain[U]
            if self.members:
                labs = self.labels[self.S[R]]
                for a in np.flatnonzero(labs >= 0):
                    if self.counts[labs[a]] == 1:
                        scores[a, self.labels[U] != labs[a]] = np.inf
            a, b = divmod(int(np.argmin(scores)), len(U))
            if not math.isfinite(float(scores[a, b])):
                break
            r, u = remove.pop(a), insert.pop(b)
            if not self.move(u, r):
                return True
            if self.cost < best_cost:
                best, best_cost = self.S.copy(), self.cost
        if self.ready and _clock() < self.deadline:
            self.goto(best)
        return True


def _pool_add(pool, S, cost, capacity):
    key = tuple(sorted(map(int, S)))
    if any(key == e[1] for e in pool):
        return
    item = (cost, key, np.asarray(key, dtype=np.int64))
    if len(pool) < capacity:
        pool.append(item)
        return
    eligible = [i for i, e in enumerate(pool) if e[0] >= cost]
    if eligible:
        points = set(key)
        # Prefer replacing members that are close and not better, so the elite pool is not filled by one basin.
        i = min(eligible, key=lambda j: (len(points.difference(pool[j][1])), -pool[j][0]))
        pool[i] = item


def _enumerate(D, p, labels, members, deadline, saved):
    n = len(D)
    iterator = itertools.combinations(range(n), p)
    batch = max(1, 262144 // (n * p))
    while _clock() < deadline:
        tuples = list(itertools.islice(iterator, batch))
        if not tuples:
            return True
        choices = np.asarray(tuples, dtype=np.int64)
        values = D[:, choices].min(axis=2).sum(axis=0)
        for group_id in range(len(members)):
            values[~(labels[choices] == group_id).any(axis=1)] = np.inf
        b = int(np.argmin(values))
        if float(values[b]) < saved["cost"]:
            saved["cost"], saved["S"] = float(values[b]), choices[b].copy()
    return False


def _search(st, budget, rng):
    st.local()
    if not st.ready or st.saved["cost"] == 0:
        return
    accepted, accepted_cost = st.S.copy(), st.cost
    pool = []
    _pool_add(pool, st.S, st.cost, 2)
    stagnation = 0
    step_time = max(st.work_time, budget * 1e-5, 1e-6)
    max_changes = min(st.p, st.n - st.p)
    while st.ready and _clock() < st.deadline:
        begin = _clock()
        remaining = st.deadline - begin
        if remaining <= 1.2 * st.work_time:
            break
        estimated = max(1.0, budget / step_time)
        depth = min(max_changes, max(1, int(math.sqrt(st.p) * math.log1p(estimated) / 3)))
        stall_limit = max(3, int(math.sqrt(estimated)))
        capacity = min(st.n, max(2, int(2 + estimated ** 0.25)))
        if stagnation >= stall_limit:
            stagnation = 0
            choices = [e for e in pool if len(set(e[1]).difference(map(int, st.S))) >= 2]
            did = False
            if choices and rng.random() < 0.65:
                # Either direction can be chosen as the starting point; guidance favours more dissimilar elites.
                current = st.S.copy()
                base_set = set(map(int, current))
                weights = [len(base_set.difference(e[1])) for e in choices]
                guide = rng.choices(choices, weights=weights, k=1)[0][2]
                if rng.random() < 0.5:
                    if not st.goto(guide):
                        break
                    guide = current
                did = st.relink(guide, min(st.deadline, _clock() + 0.4 * remaining))
            if not did and st.ready:
                k = min(st.p, max(2, 2 * depth), max_changes)
                did = st.ruin(k, rng, min(st.deadline, _clock() + 0.35 * remaining))
            if not did and st.ready:
                did = st.shake(depth, rng)
            if not st.ready:
                break
            if did:
                st.local()
                if not st.ready:
                    break
                accepted, accepted_cost = st.S.copy(), st.cost
                _pool_add(pool, st.S, st.cost, capacity)
        else:
            k = min(depth, max(1, 1 + int(math.sqrt(stagnation + 1))))
            if not st.shake(k, rng):
                if not st.ready or not np.isfinite(st.scores()).any():
                    break
                stagnation += 1
                continue
            st.local()
            if not st.ready:
                break
            _pool_add(pool, st.S, st.cost, capacity)
            tol = 1e-12 * max(accepted_cost, 1e-300)
            if st.cost < accepted_cost - tol:
                accepted, accepted_cost = st.S.copy(), st.cost
                stagnation = 0
            else:
                stagnation += 1
                if st.cost <= accepted_cost + tol:
                    accepted, accepted_cost = st.S.copy(), st.cost
                elif not st.goto(accepted):
                    break
        if st.saved["cost"] == 0:
            return
        elapsed = _clock() - begin
        step_time = 0.85 * step_time + 0.15 * elapsed


def _run(instance, deadline, budget, saved):
    n, p = int(instance["n"]), int(instance["p"])
    D = np.array(instance["dist"], dtype=np.float64)
    finite = np.isfinite(D)
    labels, members = _sinks(finite)
    # Coverage is judged only by true reachability; numerical scaling and internal placeholder values play no part in feasibility.
    maximum = float(np.max(D, where=finite, initial=0.0))
    if maximum > 0:
        np.divide(D, maximum, out=D, where=finite)
    D[~finite] = 2.0
    colsum = D.sum(axis=0)
    seeds = [int(group[int(np.argmin(colsum[group]))]) for group in members]
    used = np.zeros(n, dtype=bool)
    used[seeds] = True
    fill = np.argsort(colsum, kind="stable")
    fallback = seeds + fill[~used[fill]][:p - len(seeds)].tolist()
    saved["S"], saved["cost"] = fallback, _cost(D, fallback)
    if p == 1:
        values = np.where(finite.all(axis=0), colsum, np.inf)
        saved["S"] = [int(np.argmin(values))]
        return
    if p == n - 1:
        np.fill_diagonal(D, np.inf)
        removal = D.min(axis=1)
        removal[removal >= 2.0] = np.inf
        excluded = int(np.argmin(removal))
        saved["S"] = [j for j in range(n) if j != excluded]
        return
    del finite
    if saved["cost"] == 0 or _clock() >= deadline:
        return
    remaining = deadline - _clock()
    S = _greedy(D, seeds, p, labels, members, None, _clock() + 0.22 * remaining)
    if S is not None:
        value = _cost(D, S)
        if value < saved["cost"]:
            saved["cost"], saved["S"] = value, S
    if saved["cost"] == 0 or _clock() >= deadline:
        return
    # Exact enumeration starts only when the combination space is expected to be traversable within this budget.
    count = math.comb(n, p)
    remaining = deadline - _clock()
    affordable = 0.28 * remaining / (n * p * 7e-9 + p * 3e-7)
    if count < affordable:
        if _enumerate(D, p, labels, members, deadline, saved):
            return
    if _clock() >= deadline:
        return
    st = _Search(D, p, labels, members, deadline, saved)
    if st.build(saved["S"]):
        rng = random.Random(918273645 + 7919 * n + 104729 * p)
        _search(st, max(deadline - _clock(), budget * 1e-3), rng)
