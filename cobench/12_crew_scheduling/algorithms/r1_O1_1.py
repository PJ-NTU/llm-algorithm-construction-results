"""Duty-feasible path column generation with large-neighbourhood recombination
Combines cost matching, dual pricing, full-path integer recombination and duty-constrained subproblem search, saving better feasible solutions within the wall-clock budget.
"""

import math
import random
import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linear_sum_assignment, linprog, milp
from scipy.sparse import coo_matrix


_EPS = 1e-9


def _shape(routes, k):
    out = [list(r) for r in routes if r]
    out.extend([] for _ in range(max(0, k - len(out))))
    return {"crew_routes": out,
            "start_tasks": [r[0] if r else -1 for r in out],
            "end_tasks": [r[-1] if r else -1 for r in out]}


class _Search:
    def __init__(self, instance, budget, started):
        self.n = int(instance["num_tasks"])
        self.k = int(instance["max_crews"])
        self.l = float(instance["max_duty_time"])
        self.s = list(map(float, instance["start_times"]))
        self.f = list(map(float, instance["finish_times"]))
        self.c = instance["transition_costs"]
        self.v = instance["valid_transitions"]
        self.budget = max(0.0, budget)
        self.started = started
        self.end = started + self.budget - min(self.budget * 0.04, 0.20)
        self.rng = random.Random(9856)
        self.best = None
        self.best_cost = float("inf")
        self.near = [[i] for i in range(self.n)]
        self.near_key = (self.n + 1, float("inf"))
        self.pool = {}
        self.cuts = set()
        self.adj = [set() for _ in range(self.n)]
        self.ins = [[] for _ in range(self.n)]
        self.outs = [[] for _ in range(self.n)]
        self.arcs = []
        self.order = []
        self.dual = np.zeros(self.n)
        self.mu = 0.0
        self.cursor = 0
        self.price_speed = 5.0e6
        self.stall = 0
        self.lp_valid = False
        self.scale = 1.0
        self.maxcost = 1.0
        self.pool_limit = max(self.n + 1, int(self.n * (10 + 12 * math.log1p(self.budget))))

    def now(self):
        return time.perf_counter()

    def cap(self, span):
        return min(self.end, self.now() + max(0.0, span))

    def cost(self, r):
        return sum(float(self.c[a][b]) for a, b in zip(r, r[1:]))

    def route_ok(self, r):
        return (bool(r) and self.f[r[-1]] - self.s[r[0]] <= self.l + _EPS
                and all(b in self.adj[a] for a, b in zip(r, r[1:])))

    def valid(self, routes, crew_cap=None):
        k = self.k if crew_cap is None else crew_cap
        if len(routes) > k:
            return False
        seen = [0] * self.n
        for r in routes:
            if not self.route_ok(r):
                return False
            for i in r:
                if i < 0 or i >= self.n or seen[i]:
                    return False
                seen[i] = 1
        return all(seen)

    def add(self, r):
        if r:
            key = tuple(r)
            if key not in self.pool:
                self.pool[key] = self.cost(r)

    def record(self, routes):
        routes = [list(r) for r in routes if r]
        if not self.valid(routes, self.n):
            return False
        # Spare crews are only used to cut positive-cost edges; negative-cost edges are not forcibly cut because of the crew-count objective.
        room = self.k - len(routes)
        if room > 0:
            cuts = sorted(((float(self.c[r[p]][r[p + 1]]), q, p)
                           for q, r in enumerate(routes) for p in range(len(r) - 1)
                           if float(self.c[r[p]][r[p + 1]]) > _EPS), reverse=True)
            chosen = {(q, p) for _, q, p in cuts[:room]}
            new = []
            for q, r in enumerate(routes):
                start = 0
                for p in range(len(r) - 1):
                    if (q, p) in chosen:
                        new.append(r[start:p + 1])
                        start = p + 1
                new.append(r[start:])
            routes = new
        value = sum(self.cost(r) for r in routes)
        key = (max(0, len(routes) - self.k), value)
        if key < self.near_key:
            self.near_key = key
            self.near = [r[:] for r in routes]
        for r in routes:
            self.add(r)
        if len(routes) <= self.k and value < self.best_cost - _EPS:
            self.best = [r[:] for r in routes]
            self.best_cost = value
            self.stall = 0
            return True
        return False

    def prepare(self):
        for i in range(self.n):
            for j in range(self.n):
                if (i != j and self.v[i][j] == 1 and self.f[i] <= self.s[j] + _EPS
                        and self.f[j] - self.s[i] <= self.l + _EPS):
                    w = float(self.c[i][j])
                    self.adj[i].add(j)
                    self.outs[i].append((w, j))
                    self.ins[j].append((w, i))
                    self.arcs.append((w, i, j))
            if self.f[i] - self.s[i] <= self.l + _EPS:
                self.add([i])
        self.maxcost = max([1.0] + [abs(w) for w, _, _ in self.arcs])
        self.scale = max(1e-8, sum(abs(w) for w, _, _ in self.arcs) / max(1, len(self.arcs)))
        degree = [len(p) for p in self.ins]
        ready = [i for i, d in enumerate(degree) if not d]
        while ready:
            u = ready.pop()
            self.order.append(u)
            for _, v in self.outs[u]:
                degree[v] -= 1
                if degree[v] == 0:
                    ready.append(v)
        self.dag = len(self.order) == self.n
        if not self.dag:
            self.order = sorted(range(self.n), key=lambda i: (self.s[i], self.f[i], i))
        for i in range(self.n):
            self.ins[i].sort()
            self.outs[i].sort()
        self.record([[i] for i in range(self.n)])

    def sweep(self, mode, noise):
        routes = []
        for t in self.order:
            candidates = []
            for q, r in enumerate(routes):
                u = r[-1]
                if t not in self.adj[u] or self.f[t] - self.s[r[0]] > self.l + _EPS:
                    continue
                if mode == 0:
                    score = float(self.c[u][t]) / self.scale
                elif mode == 1:
                    score = (self.s[t] - self.f[u]) / max(self.l, 1e-9)
                elif mode == 2:
                    score = (self.s[r[0]] - self.s[t]) / max(self.l, 1e-9)
                else:
                    score = (self.s[t] - self.s[r[0]]) / max(self.l, 1e-9)
                candidates.append((score + noise * self.rng.random(), q))
            if candidates:
                routes[min(candidates)[1]].append(t)
            else:
                routes.append([t])
        return routes

    def merge_seed(self, noise):
        routes = {i: [i] for i in range(self.n)}
        owner = list(range(self.n))
        edges = sorted((w / self.scale + noise * self.rng.random(), w, i, j)
                       for w, i, j in self.arcs)
        for _, w, i, j in edges:
            a, b = owner[i], owner[j]
            if a == b or (len(routes) <= self.k and w >= 0):
                continue
            ra, rb = routes[a], routes[b]
            if ra[-1] != i or rb[0] != j or self.f[rb[-1]] - self.s[ra[0]] > self.l + _EPS:
                continue
            ra.extend(rb)
            for t in rb:
                owner[t] = a
            del routes[b]
        return list(routes.values())

    def insertion(self, t, routes):
        ans = []
        for q, r in enumerate(routes):
            for p in range(len(r) + 1):
                if p and t not in self.adj[r[p - 1]]:
                    continue
                if p < len(r) and r[p] not in self.adj[t]:
                    continue
                h, e = (t if p == 0 else r[0]), (t if p == len(r) else r[-1])
                if self.f[e] - self.s[h] > self.l + _EPS:
                    continue
                delta = (float(self.c[r[p - 1]][t]) if p else 0.0)
                if p < len(r):
                    delta += float(self.c[t][r[p]])
                if p and p < len(r):
                    delta -= float(self.c[r[p - 1]][r[p]])
                ans.append((delta, q, p))
        return ans

    def repair(self, routes, until):
        routes = [r[:] for r in routes]
        while len(routes) > self.k and self.now() < until:
            donors = sorted(range(len(routes)), key=lambda q: (len(routes[q]), self.rng.random()))
            success = False
            for donor in donors:
                if self.now() >= until:
                    break
                work = [r[:] for q, r in enumerate(routes) if q != donor]
                pool = routes[donor][:]
                taboo = set()
                attempt_end = min(until, self.now() + (until - self.now()) / max(1, len(donors)))
                while pool and self.now() < attempt_end:
                    choices = []
                    for t in pool:
                        ins = self.insertion(t, work)
                        if ins:
                            ins.sort()
                            regret = ins[1][0] - ins[0][0] if len(ins) > 1 else self.maxcost * self.n
                            choices.append((len(ins), -regret, ins[0], t))
                    if choices:
                        _, _, (_, q, p), t = min(choices)
                        work[q].insert(p, t)
                        pool.remove(t)
                        continue
                    t = self.rng.choice(pool)
                    replacements = []
                    for q, r in enumerate(work):
                        for p, old in enumerate(r):
                            if (t, old) in taboo:
                                continue
                            rr = r[:p] + [t] + r[p + 1:]
                            if self.route_ok(rr):
                                replacements.append((self.cost(rr) - self.cost(r)
                                                     + self.rng.random() * self.scale, q, p, old))
                    if not replacements:
                        break
                    _, q, p, old = min(replacements)
                    work[q][p] = t
                    pool.remove(t)
                    pool.append(old)
                    taboo.add((t, old))
                    taboo.add((old, t))
                for r in work:
                    self.add(r)
                if not pool:
                    routes = work
                    success = True
                    self.record(routes)
                    break
            if not success:
                break
        return routes

    def decode(self, tasks, succ):
        pred = set(succ.values())
        seen = set()
        routes, cycles = [], []
        for h in list(i for i in tasks if i not in pred) + list(tasks):
            if h in seen:
                continue
            r, local = [], {}
            u = h
            while u not in seen:
                local[u] = len(r)
                r.append(u)
                seen.add(u)
                if u not in succ:
                    break
                u = succ[u]
                if u in local:
                    cycle = r[local[u]:]
                    cycles.append(tuple((a, succ[a]) for a in cycle))
                    q = max(range(len(cycle)), key=lambda p: float(self.c[cycle[p]][succ[cycle[p]]]))
                    r = cycle[q + 1:] + cycle[:q + 1]
                    break
            routes.append(r)
        return routes, cycles

    def split(self, routes):
        out = []
        for r in routes:
            # Every shortest overtime sub-chain yields a valid cut forbidding selecting all of its arcs.
            q = 0
            for p in range(len(r)):
                q = max(q, p)
                while q < len(r) and self.f[r[q]] - self.s[r[p]] <= self.l + _EPS:
                    q += 1
                if q < len(r):
                    self.cuts.add(tuple(zip(r[p:q], r[p + 1:q + 1])))
            start = 0
            for p in range(len(r)):
                if self.f[r[p]] - self.s[r[start]] > self.l + _EPS:
                    if p > start:
                        out.append(r[start:p])
                    start = p
            out.append(r[start:])
        return out

    def matching(self, tasks, k):
        n = len(tasks)
        k = min(k, n)
        if not n or k <= 0:
            return None, False
        # The dummy node allows using at most k paths, rather than wrongly forcing exactly k.
        a = np.full((n + k, n + k), np.inf)
        a[:n, n:] = 0.0
        a[n:, :] = 0.0
        index = {i: p for p, i in enumerate(tasks)}
        for p, i in enumerate(tasks):
            for w, j in self.outs[i]:
                q = index.get(j)
                if q is not None:
                    a[p, q] = w
        try:
            rows, cols = linear_sum_assignment(a)
        except (ValueError, MemoryError):
            return None, False
        succ = {tasks[int(p)]: tasks[int(q)] for p, q in zip(rows, cols) if p < n and q < n}
        routes, cycles = self.decode(tasks, succ)
        self.cuts.update(cycles)
        feasible = not cycles and len(routes) <= k and all(self.route_ok(r) for r in routes)
        return (routes if feasible else self.split(routes)), feasible

    def polish(self, routes, until, perturb=False):
        routes = [r[:] for r in routes]
        while self.now() < until:
            loc = {i: (q, p) for q, r in enumerate(routes) for p, i in enumerate(r)}
            improved = False
            vertices = list(loc)
            self.rng.shuffle(vertices)
            for a in vertices:
                if self.now() >= until:
                    return routes
                qa, pa = loc[a]
                ra = routes[qa]
                for w, y in self.outs[a]:
                    qb, py = loc[y]
                    if qa == qb:
                        continue
                    rb = routes[qb]
                    b = ra[pa + 1] if pa + 1 < len(ra) else None
                    x = rb[py - 1] if py else None
                    if b is not None and x is not None and b not in self.adj[x]:
                        continue
                    aa, bb = ra[:pa + 1] + rb[py:], rb[:py] + ra[pa + 1:]
                    if not self.route_ok(aa) or (bb and not self.route_ok(bb)):
                        continue
                    delta = w - (float(self.c[a][b]) if b is not None else 0.0)
                    delta -= float(self.c[x][y]) if x is not None else 0.0
                    if b is not None and x is not None:
                        delta += float(self.c[x][b])
                    if delta >= -_EPS and not (perturb and self.rng.random() < 1.0 / max(2, self.n)):
                        continue
                    routes[qa], routes[qb] = aa, bb
                    routes = [r for r in routes if r]
                    self.add(aa)
                    self.add(bb)
                    improved = True
                    perturb = False
                    break
                if improved:
                    break
            if improved:
                self.record(routes)
                continue
            # Single-task relocation may change connections inside two paths, not only tail exchanges.
            for qa in self.rng.sample(range(len(routes)), len(routes)):
                if self.now() >= until:
                    return routes
                ra = routes[qa]
                for p, t in enumerate(ra):
                    remaining = ra[:p] + ra[p + 1:]
                    if remaining and not self.route_ok(remaining):
                        continue
                    work = [r[:] for q, r in enumerate(routes) if q != qa]
                    options = self.insertion(t, work)
                    if not options:
                        continue
                    delta, qb, pos = min(options)
                    delta += self.cost(remaining) - self.cost(ra)
                    if delta < -_EPS:
                        work[qb].insert(pos, t)
                        if remaining:
                            work.append(remaining)
                        routes = work
                        self.record(routes)
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
        return routes

    def trim_pool(self):
        if len(self.pool) <= self.pool_limit:
            return
        keep = {(i,) for i in range(self.n)}
        for r in (self.best or self.near):
            keep.add(tuple(r))
        others = [r for r in self.pool if r not in keep]
        self.rng.shuffle(others)
        others.sort(key=lambda r: (self.pool[r] - sum(self.dual[i] for i in r) - self.mu)
                    / math.sqrt(len(r)))
        room = max(0, self.pool_limit - len(keep))
        main = int(room * 0.85)
        keep.update(others[:main])
        rest = others[main:]
        self.rng.shuffle(rest)
        keep.update(rest[:room - main])
        self.pool = {r: w for r, w in self.pool.items() if r in keep}

    def master(self, until, integer):
        self.trim_pool()
        columns = list(self.pool)
        m = len(columns)
        if not m or self.now() >= until:
            return False
        rows, cols, data = [], [], []
        for j, r in enumerate(columns):
            rows.extend(r)
            cols.extend([j] * len(r))
            data.extend([1.0] * len(r))
            rows.append(self.n)
            cols.append(j)
            data.append(1.0)
        rows.append(self.n)
        cols.append(m)
        data.append(-1.0)
        mat = coo_matrix((data, (rows, cols)), shape=(self.n + 1, m + 1)).tocsc()
        penalty = (2 * self.n * self.maxcost + 1.0) / self.scale
        c = np.array([self.pool[r] / self.scale for r in columns] + [penalty])
        remaining = until - self.now()
        if remaining <= 0:
            return False
        try:
            if not integer:
                res = linprog(c, A_ub=mat[-1:, :], b_ub=[self.k],
                              A_eq=mat[:-1, :], b_eq=np.ones(self.n), bounds=(0, None),
                              method="highs", options={"time_limit": remaining * 0.88})
                if res.success:
                    self.dual = np.asarray(res.eqlin.marginals) * self.scale
                    self.mu = float(res.ineqlin.marginals[0]) * self.scale
                    self.lp_valid = True
                    return True
                return False
            ub = np.ones(m + 1)
            ub[-1] = 0.0 if self.best is not None else self.n
            lower = np.ones(self.n + 1)
            upper = np.ones(self.n + 1)
            lower[-1], upper[-1] = -np.inf, self.k
            res = milp(c, integrality=np.ones(m + 1), bounds=Bounds(np.zeros(m + 1), ub),
                       constraints=LinearConstraint(mat, lower, upper),
                       options={"time_limit": remaining * 0.88, "mip_rel_gap": 0.0})
            if res.x is not None:
                routes = [list(columns[j]) for j in range(m) if res.x[j] > 0.5]
                return self.record(routes)
        except Exception:
            return False
        return False

    def price(self, until, noise):
        if self.now() >= until:
            return
        pi = self.dual.copy()
        if noise:
            pi += np.array([self.rng.gauss(0, noise * self.scale) for _ in range(self.n)])
        if not self.dag:
            self.price_cyclic(pi, until)
            return
        start_time = self.now()
        batch = min(self.n, max(1, int((until - start_time) * self.price_speed
                                       / max(1, len(self.arcs)))))
        starts = [(self.cursor + q) % self.n for q in range(batch)]
        self.cursor = (self.cursor + batch) % self.n
        dist = np.full((batch, self.n), np.inf)
        back = np.full((batch, self.n), -1, dtype=np.int32)
        start_pos = {h: q for q, h in enumerate(starts)}
        start_times = np.array([self.s[h] for h in starts])
        ix = np.arange(batch)
        processed = 0
        for j in self.order:
            if self.now() >= until:
                break
            if self.ins[j]:
                predecessors = np.array([i for _, i in self.ins[j]], dtype=np.int32)
                costs = np.array([w for w, _ in self.ins[j]])
                values = dist[:, predecessors] + costs
                arg = np.argmin(values, axis=1)
                dist[:, j] = values[ix, arg] - pi[j]
                back[:, j] = predecessors[arg]
                invalid = self.f[j] - start_times > self.l + _EPS
                dist[invalid, j] = np.inf
                back[invalid, j] = -1
            q = start_pos.get(j)
            if q is not None and self.f[j] - self.s[j] <= self.l + _EPS:
                dist[q, j] = -pi[j]
                back[q, j] = -1
            processed += 1
        elapsed = self.now() - start_time
        if elapsed > 0 and processed:
            speed = batch * max(1, len(self.arcs)) * processed / max(1, self.n) / elapsed
            self.price_speed = 0.6 * self.price_speed + 0.4 * speed
        reduced = dist - self.mu
        candidates = np.flatnonzero(np.isfinite(reduced.ravel()) &
                                    (reduced.ravel() < self.scale * (noise + 1e-8)))
        count = min(len(candidates), max(batch, int(batch * (3 + math.log1p(self.budget)))))
        if count < len(candidates):
            values = reduced.ravel()[candidates]
            candidates = candidates[np.argpartition(values, count - 1)[:count]]
        for flat in candidates:
            q, end = divmod(int(flat), self.n)
            route = [end]
            u = end
            while back[q, u] >= 0:
                u = int(back[q, u])
                route.append(u)
            route.reverse()
            if route[0] == starts[q] and self.route_ok(route):
                self.add(route)

    def price_cyclic(self, pi, until):
        # Zero-duration tasks at the same time may form cycles; simple paths are then enumerated without arbitrarily deleting valid reverse edges.
        while self.now() < until:
            h = self.cursor % self.n
            self.cursor += 1
            stack = [([h], {h}, -float(pi[h]))]
            local_end = min(until, self.now() + (until - self.now()) / max(1, self.n))
            while stack and self.now() < local_end:
                route, seen, value = stack.pop()
                if value - self.mu < self.scale:
                    self.add(route)
                choices = self.outs[route[-1]][:]
                self.rng.shuffle(choices)
                choices.sort(key=lambda x: x[0] - pi[x[1]], reverse=True)
                for w, j in choices:
                    if j not in seen and self.f[j] - self.s[h] <= self.l + _EPS:
                        stack.append((route + [j], seen | {j}, value + w - pi[j]))

    def exact(self, tasks, k, incumbent, until):
        if self.now() >= until or not tasks or k <= 0:
            return None
        n = len(tasks)
        index = {i: p for p, i in enumerate(tasks)}
        edges = [(w, i, j) for i in tasks for w, j in self.outs[i] if j in index]
        m = len(edges)
        if not m:
            return [[i] for i in tasks] if n <= k else None
        rank_needed = not self.dag
        size = m + n * (2 if rank_needed else 1)
        rows, cols, data, lb, ub = [], [], [], [], []

        def row(terms, lo, hi):
            p = len(lb)
            for j, a in terms:
                rows.append(p)
                cols.append(j)
                data.append(a)
            lb.append(lo)
            ub.append(hi)

        outgoing, incoming = [[] for _ in tasks], [[] for _ in tasks]
        edge_index = {}
        for e, (_, i, j) in enumerate(edges):
            outgoing[index[i]].append((e, 1.0))
            incoming[index[j]].append((e, 1.0))
            edge_index[i, j] = e
        for terms in outgoing + incoming:
            row(terms, 0.0, 1.0)
        target = max(0, n - k)
        nonnegative = all(w >= 0 for w, _, _ in edges)
        row([(e, 1.0) for e in range(m)], target, target if nonnegative else n - 1)
        unit = max(1.0, abs(self.l), max(self.f[i] for i in tasks) - min(self.s[i] for i in tasks))
        base = min(self.s[i] for i in tasks)
        lo = np.zeros(size)
        hi = np.ones(size)
        for i, p in index.items():
            lo[m + p] = (self.f[i] - self.l - base) / unit
            hi[m + p] = (self.s[i] - base) / unit
        for e, (_, i, j) in enumerate(edges):
            p, q = index[i], index[j]
            big = max(0.0, hi[m + q] - lo[m + p])
            # A selected connection makes the successor duty start no later than the predecessor; the endpoint bound guarantees the whole path is duty-feasible.
            row([(e, big), (m + q, 1.0), (m + p, -1.0)], -np.inf, big)
            if rank_needed:
                row([(e, float(n)), (m + n + p, 1.0), (m + n + q, -1.0)], -np.inf, n - 1.0)
        if rank_needed:
            hi[m + n:] = n - 1
        for cut in self.cuts:
            if cut and all(arc in edge_index for arc in cut):
                row([(edge_index[arc], 1.0) for arc in cut], -np.inf, len(cut) - 1)
        obj = np.zeros(size)
        obj[:m] = [w / self.scale for w, _, _ in edges]
        if incumbent is not None:
            cutoff = sum(self.cost(r) for r in incumbent) / self.scale
            row([(e, obj[e]) for e in range(m)], -np.inf, cutoff + 1e-8)
        mat = coo_matrix((data, (rows, cols)), shape=(len(lb), size)).tocsc()
        remaining = until - self.now()
        if remaining <= 0:
            return None
        integral = np.zeros(size)
        integral[:m] = 1
        try:
            res = milp(obj, integrality=integral, bounds=Bounds(lo, hi),
                       constraints=LinearConstraint(mat, np.array(lb), np.array(ub)),
                       options={"time_limit": remaining * 0.85, "mip_rel_gap": 0.0})
            if res.x is None:
                return None
            succ = {i: j for e, (_, i, j) in enumerate(edges) if res.x[e] > 0.5}
            routes, cycles = self.decode(tasks, succ)
            if (not cycles and len(routes) <= k and sum(map(len, routes)) == n
                    and all(self.route_ok(r) for r in routes)):
                return routes
        except Exception:
            return None
        return None

    def neighborhood(self, until):
        base = self.best or self.near
        if self.best is None:
            tasks, fixed, selected, k = list(range(self.n)), [], None, self.k
        else:
            count = len(base)
            if count <= 1:
                selected_ids = list(range(count))
            else:
                strength = 2 + int(math.log1p(self.budget) + math.sqrt(self.stall + 1))
                amount = min(count, self.rng.randint(2, max(2, strength)))
                first = self.rng.randrange(count)
                r = base[first]
                ranked = []
                for q, rr in enumerate(base):
                    overlap = max(0.0, min(self.f[r[-1]], self.f[rr[-1]]) - max(self.s[r[0]], self.s[rr[0]]))
                    affinity = overlap / max(1.0, self.l) + self.rng.random()
                    ranked.append((-affinity, q))
                selected_ids = [first] + [q for _, q in sorted(ranked) if q != first][:amount - 1]
            chosen = set(selected_ids)
            selected = [r for q, r in enumerate(base) if q in chosen]
            fixed = [r for q, r in enumerate(base) if q not in chosen]
            tasks = [i for r in selected for i in r]
            k = self.k - len(fixed)
        routes, optimal = self.matching(tasks, k)
        if routes is not None:
            self.record(fixed + routes)
        if optimal:
            return
        routes = self.exact(tasks, k, selected, until)
        if routes is not None:
            self.record(fixed + routes)

    def run(self):
        self.prepare()
        if self.n == 0:
            return _shape([], self.k)
        if not self.arcs:
            return _shape(self.best or self.near, self.k)
        seed_end = min(self.end, self.started + self.budget * 0.10)
        for mode in (2, 1, 0, 3):
            if self.now() >= seed_end:
                break
            self.record(self.sweep(mode, 0.0))
        # Cost matching keeps the exact solving ability of the original algorithm when the duty limit is inactive.
        if self.now() < self.end:
            routes, optimal = self.matching(list(range(self.n)), self.k)
            if routes is not None:
                self.record(routes)
                if optimal and self.best is not None:
                    return _shape(self.best, self.k)
        if self.now() < seed_end:
            self.record(self.merge_seed(0.0))
        if self.best is None and self.now() < self.end:
            repair_end = self.cap(self.budget * 0.05)
            self.record(self.repair(self.near, repair_end))
        if self.best is None and self.now() < self.end:
            self.neighborhood(self.cap(self.budget * 0.22))
        if self.best is not None:
            self.record(self.polish(self.best, self.cap(self.budget * 0.04)))
        turn = 0
        while self.now() < self.end:
            before = self.best_cost
            remaining = self.end - self.now()
            quantum = min(remaining, self.budget / (7.0 + math.log1p(self.budget)))
            if turn % 3 == 0:
                self.master(self.cap(quantum * 0.24), False)
                noise = 0.0 if turn == 0 else min(1.0, 0.04 * math.sqrt(self.stall + 1))
                self.price(self.cap(quantum * 0.30), noise)
                self.master(self.cap(quantum * 0.70), True)
            elif turn % 3 == 1:
                if self.best is not None:
                    route = self.polish(self.best, self.cap(quantum * 0.15), True)
                    self.record(route)
                self.neighborhood(self.cap(quantum))
            else:
                routes = (self.sweep(self.rng.randrange(4), self.rng.random())
                          if self.rng.random() < 0.7 else self.merge_seed(self.rng.random()))
                self.record(routes)
                if len(routes) > self.k:
                    routes = self.repair(routes, self.cap(quantum * 0.20))
                if len(routes) <= self.k:
                    self.record(self.polish(routes, self.cap(quantum * 0.20)))
                self.master(self.cap(quantum * 0.20), False)
                self.price(self.cap(quantum * 0.35), 0.10 + 0.1 * self.rng.random())
                self.master(self.cap(quantum * 0.60), True)
            self.stall = 0 if self.best_cost < before - _EPS else self.stall + 1
            turn += 1
        return _shape(self.best if self.best is not None else self.near, self.k)


def solve(instance: dict, time_limit_s: float) -> dict:
    started = time.perf_counter()
    search = None
    try:
        search = _Search(instance, float(time_limit_s), started)
        return search.run()
    except Exception:
        # If an exception occurs during search, keep the fully verified historical best solution and do not overwrite it with the new state.
        if search is not None:
            return _shape(search.best if search.best is not None else search.near, search.k)
        k = max(0, int(instance.get("max_crews", 0)))
        return _shape([], k)
