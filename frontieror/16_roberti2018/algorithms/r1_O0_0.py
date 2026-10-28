"""Group-cooperative destroy-and-repair with route recombination
Uses capacity allocation, two-echelon route construction, time-limited neighbourhood search and integer-programming route recombination, keeping the feasible solution evaluated by the original total cost.
"""
import math
import random
import time
import heapq
from collections import Counter

import numpy as np
try:
    from scipy.optimize import milp, Bounds, LinearConstraint
    from scipy.sparse import coo_matrix
except ImportError:
    milp = None


class _Rows:
    def __init__(self, x, y, budget):
        self.x, self.y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        self.cache = {}
        self.limit = max(2, int(budget // max(1, len(x))))

    def __getitem__(self, i):
        row = self.cache.get(i)
        if row is None:
            row = np.hypot(self.x - self.x[i], self.y - self.y[i]).tolist()
            if len(self.cache) >= self.limit:
                self.cache.pop(next(iter(self.cache)))
            self.cache[i] = row
        return row


class _Model:
    def __init__(self, c, lb, ub, integrality):
        self.c, self.lb, self.ub, self.integrality = c, lb, ub, integrality
        self.ri, self.ci, self.val, self.lo, self.hi = [], [], [], [], []

    def row(self, terms, lo=-np.inf, hi=np.inf):
        k = len(self.lo)
        for j, v in terms:
            if v:
                self.ri.append(k)
                self.ci.append(j)
                self.val.append(float(v))
        self.lo.append(lo)
        self.hi.append(hi)

    def run(self, deadline):
        remaining = deadline - time.perf_counter()
        if milp is None or remaining <= 0.008:
            return None
        try:
            mat = coo_matrix((self.val, (self.ri, self.ci)),
                             shape=(len(self.lo), len(self.c))).tocsc()
            remaining = deadline - time.perf_counter()
            if remaining <= 0.008:
                return None
            result = milp(np.asarray(self.c, dtype=float),
                          integrality=np.asarray(self.integrality, dtype=np.int32),
                          bounds=Bounds(self.lb, self.ub),
                          constraints=LinearConstraint(mat, self.lo, self.hi),
                          options={'time_limit': remaining, 'mip_rel_gap': 0.0,
                                   'presolve': True, 'disp': False})
            if result.x is not None and np.all(np.isfinite(result.x)):
                return result.x
        except Exception:
            pass
        return None


class _Solver:
    def __init__(self, data, limit, started):
        self.data = data
        self.start = started
        self.limit = max(0.0, float(limit))
        self.strength = math.log1p(self.limit)
        self.end = started + max(0.0, self.limit - min(0.08, 0.012 * self.limit))
        self.rng = random.Random(1658)
        self.ids = [int(v) for v in data['customers_id']]
        self.sids = [int(v) for v in data['satellites_id']]
        self.gids = [int(v) for v in data['groups_group_id']]
        self.n, self.m, self.H = len(self.ids), len(self.sids), len(self.gids)
        self.q = [int(v) for v in data['customers_demand']]
        self.Q1, self.Q2 = float(data['Q1']), int(data['Q2'])
        self.T1, self.T2 = int(data['T1']), int(data['T2'])
        self.cap = [float(v) for v in data['satellites_capacity']]
        self.maxv = [int(v) for v in data['satellites_max_vehicles']]
        self.hand = [float(v) for v in data['satellites_handling_cost']]
        self.depot_id = int(data['depot_id'])
        index = {v: i for i, v in enumerate(self.ids)}
        self.groups = [[index[int(v)] for v in members]
                       for members in data['groups_customer_ids']]
        self.gh = [0] * self.n
        for h, members in enumerate(self.groups):
            for i in members:
                self.gh[i] = h
        self.gd = [sum(self.q[i] for i in g) for g in self.groups]
        self.x = list(data['customers_x']) + list(data['satellites_x']) + [data['depot_x']]
        self.y = list(data['customers_y']) + list(data['satellites_y']) + [data['depot_y']]
        self.depot = self.n + self.m
        budget = int(220000 * (1 + self.strength))
        if len(self.x) ** 2 <= budget:
            xx, yy = np.asarray(self.x, dtype=float), np.asarray(self.y, dtype=float)
            self.d = np.hypot(xx[:, None] - xx, yy[:, None] - yy).tolist()
        else:
            self.d = _Rows(self.x, self.y, budget)
        self.width = min(max(0, self.n - 1), int(5 + 5 * self.strength + math.sqrt(self.n)))
        self.near = {}
        self.fc = {}
        self.pool = {}
        self.pool_limit = max(self.T2, int(self.n * (3 + 2 * self.strength)))
        self.best = None
        self.group_cost = []
        avgq = sum(self.q) / max(1, self.n)
        for h, members in enumerate(self.groups):
            self.group_cost.append([
                sum(2.0 * self.d[self.n + s][i] * (self.q[i] + 0.18 * avgq)
                    / max(1, self.Q2) for i in members)
                + self.gd[h] * (self.hand[s] + 2 * self.d[self.depot][self.n + s]
                                / max(1.0, self.Q1))
                for s in range(self.m)])

    def route_cost(self, s, seq):
        prev, value = self.n + s, 0.0
        for i in seq:
            value += self.d[prev][i]
            prev = i
        return value + self.d[prev][self.n + s]

    def travel(self, routes):
        return sum(self.route_cost(s, seq) for s, seq, load in routes)

    def nearest(self, i):
        if i not in self.near:
            row = self.d[i]
            self.near[i] = heapq.nsmallest(self.width, (j for j in range(self.n) if j != i),
                                          key=row.__getitem__)
        return self.near[i]

    def tour(self, s, nodes, deadline):
        if len(nodes) < 2:
            return nodes[:]
        remaining = set(nodes)
        seq, prev = [], self.n + s
        while remaining:
            row = self.d[prev]
            v = min(remaining, key=row.__getitem__)
            seq.append(v)
            remaining.remove(v)
            prev = v
        self.two_opt(s, seq, deadline)
        return seq

    def two_opt(self, s, seq, deadline):
        changed = True
        while changed and time.perf_counter() < deadline:
            changed = False
            for i in range(len(seq) - 1):
                if time.perf_counter() >= deadline:
                    return
                a, b = (self.n + s if i == 0 else seq[i - 1]), seq[i]
                da, db = self.d[a], self.d[b]
                for j in range(i + 1, len(seq)):
                    c, e = seq[j], (self.n + s if j + 1 == len(seq) else seq[j + 1])
                    if da[c] + db[e] < da[b] + self.d[c][e] - 1e-9:
                        seq[i:j + 1] = reversed(seq[i:j + 1])
                        changed = True
                        break
                if changed:
                    break

    def bfd(self, nodes):
        bins, loads = [], []
        for i in sorted(nodes, key=lambda v: -self.q[v]):
            q = self.q[i]
            choices = [b for b in range(len(bins)) if loads[b] + q <= self.Q2]
            if choices:
                b = max(choices, key=lambda k: loads[k])
                bins[b].append(i)
                loads[b] += q
            else:
                bins.append([i])
                loads.append(q)
        return bins

    def packing_lb(self, nodes):
        if not nodes:
            return 0
        lb = max(1, math.ceil(sum(self.q[i] for i in nodes) / max(1, self.Q2)))
        for k in range(1, max(2, int(2 + self.strength))):
            count = sum((k + 1) * self.q[i] > self.Q2 for i in nodes)
            lb = max(lb, (count + k - 1) // k)
        return lb

    def exact_pack(self, nodes, B, deadline):
        if not nodes:
            return []
        if B <= 0 or self.packing_lb(nodes) > B:
            return None
        counts = Counter(self.q[i] for i in nodes)
        sizes = sorted(counts, reverse=True)
        J = len(sizes)
        model = _Model([0.] * (J * B), [0.] * (J * B),
                       [min(counts[q], self.Q2 // q if q else counts[q])
                        for b in range(B) for q in sizes], [1] * (J * B))
        for j, q in enumerate(sizes):
            model.row(((b * J + j, 1) for b in range(B)), counts[q], counts[q])
        for b in range(B):
            model.row(((b * J + j, q) for j, q in enumerate(sizes)), hi=self.Q2)
            if b:
                model.row([(b * J + j, q) for j, q in enumerate(sizes)] +
                          [((b - 1) * J + j, -q) for j, q in enumerate(sizes)], hi=0)
        result = model.run(deadline)
        if result is None:
            return None
        bysize = {q: [i for i in nodes if self.q[i] == q] for q in sizes}
        bins = [[] for _ in range(B)]
        try:
            for b in range(B):
                for j, q in enumerate(sizes):
                    for _ in range(max(0, int(round(result[b * J + j])))):
                        bins[b].append(bysize[q].pop())
            if any(bysize[q] for q in sizes) or any(sum(self.q[i] for i in r) > self.Q2 for r in bins):
                return None
        except (IndexError, ValueError):
            return None
        return [r for r in bins if r]

    def savings(self, s, nodes, deadline):
        if len(nodes) < 2:
            return [nodes[:]] if nodes else []
        if time.perf_counter() >= deadline:
            return None
        base = self.n + s
        ds = self.d[base]
        pairs = []
        for p, i in enumerate(nodes):
            if time.perf_counter() >= deadline:
                return None
            di = self.d[i]
            js = nodes[p + 1:] if len(nodes) <= self.width * 3 else self.nearest(i)
            allowed = None if len(nodes) <= self.width * 3 else set(nodes)
            for j in js:
                if allowed is not None and (j not in allowed or j <= i):
                    continue
                pairs.append((ds[i] + ds[j] - di[j], i, j))
        pairs.sort(reverse=True)
        routes = {i: [i] for i in nodes}
        owner = {i: i for i in nodes}
        loads = {i: self.q[i] for i in nodes}
        for saving, i, j in pairs:
            if time.perf_counter() >= deadline:
                break
            a, b = owner[i], owner[j]
            if a == b or loads[a] + loads[b] > self.Q2:
                continue
            ra, rb = routes[a], routes[b]
            if i not in (ra[0], ra[-1]) or j not in (rb[0], rb[-1]):
                continue
            if ra[0] == i:
                ra.reverse()
            if rb[-1] == j:
                rb.reverse()
            ra.extend(rb)
            loads[a] += loads[b]
            for v in rb:
                owner[v] = a
            del routes[b]
        return list(routes.values())

    def assignment_mip(self, deadline, excluded=()):
        H, m = self.H, self.m
        P = H * m
        c = [v for row in self.group_cost for v in row] + [1e-6] * m
        upper = [1 if self.gd[h] <= self.cap[s] and (not self.groups[h] or self.maxv[s] > 0)
                 else 0 for h in range(H) for s in range(m)] + self.maxv[:]
        model = _Model(c, [0] * len(c), upper, [1] * len(c))
        for h in range(H):
            model.row(((h * m + s, 1) for s in range(m)), 1, 1)
        for s in range(m):
            terms = [(h * m + s, self.gd[h]) for h in range(H)]
            model.row(terms, hi=self.cap[s])
            model.row(terms + [(P + s, -self.Q2)], hi=0)
            for h in range(H):
                if self.groups[h]:
                    model.row([(h * m + s, 1), (P + s, -1)], hi=0)
            for k in range(1, max(2, int(2 + self.strength))):
                terms = [(h * m + s, sum((k + 1) * self.q[i] > self.Q2 for i in self.groups[h]))
                         for h in range(H)]
                model.row(terms + [(P + s, -k)], hi=0)
        model.row(((P + s, 1) for s in range(m)), hi=self.T2)
        for a in excluded:
            model.row(((h * m + a[h], 1) for h in range(H)), hi=H - 1)
        result = model.run(deadline)
        if result is None:
            return None
        a = [int(np.argmax(result[h * m:(h + 1) * m])) for h in range(H)]
        if any(result[h * m + a[h]] < 0.99 for h in range(H)):
            return None
        return a

    def greedy_assignment(self, randomized=False):
        a = [-1] * self.H
        nodes = [[] for _ in range(self.m)]
        loads, counts = [0] * self.m, [0] * self.m
        order = list(range(self.H))
        def priority(h):
            costs = sorted(self.group_cost[h])
            regret = costs[1] - costs[0] if len(costs) > 1 else 0
            return (self.gd[h] + regret) * (self.rng.uniform(0.6, 1.5) if randomized else 1)
        order.sort(key=priority, reverse=True)
        for h in order:
            best = None
            for s in range(self.m):
                if loads[s] + self.gd[h] > self.cap[s]:
                    continue
                b = len(self.bfd(nodes[s] + self.groups[h]))
                if b > self.maxv[s] or sum(counts) - counts[s] + b > self.T2:
                    continue
                cost = self.group_cost[h][s]
                if randomized:
                    cost *= self.rng.uniform(0.85, 1.15)
                if best is None or cost < best[0]:
                    best = (cost, s, b)
            if best is None:
                return None
            _, s, b = best
            a[h] = s
            nodes[s].extend(self.groups[h])
            loads[s] += self.gd[h]
            counts[s] = b
        return a

    def routes_for_assignment(self, a, deadline):
        if a is None:
            return None
        nodes = [[] for _ in range(self.m)]
        for h, s in enumerate(a):
            nodes[s].extend(self.groups[h])
        if any(sum(self.q[i] for i in nodes[s]) > self.cap[s] for s in range(self.m)):
            return None
        bins = [self.bfd(v) for v in nodes]
        counts = [len(b) for b in bins]
        lower = [self.packing_lb(v) for v in nodes]
        order = sorted(range(self.m), key=lambda s: (counts[s] > self.maxv[s], counts[s] - lower[s]), reverse=True)
        for s in order:
            excess = max(counts[s] - self.maxv[s], sum(counts) - self.T2, 0)
            if not excess or counts[s] <= lower[s]:
                continue
            target = max(lower[s], counts[s] - excess)
            while target < counts[s] and time.perf_counter() < deadline:
                part_end = min(deadline, time.perf_counter() + max(0.01, (deadline - time.perf_counter()) / 2))
                packed = self.exact_pack(nodes[s], target, part_end)
                if packed is not None:
                    bins[s], counts[s] = packed, len(packed)
                    break
                target += 1
        if sum(counts) > self.T2 or any(counts[s] > self.maxv[s] for s in range(self.m)):
            return None
        result = []
        for s in range(self.m):
            seqs = [self.tour(s, r, deadline) for r in bins[s]]
            saved = self.savings(s, nodes[s], deadline)
            if saved is not None and len(saved) <= self.maxv[s] and sum(counts) - counts[s] + len(saved) <= self.T2:
                for r in saved:
                    self.two_opt(s, r, deadline)
                if sum(self.route_cost(s, r) for r in saved) < sum(self.route_cost(s, r) for r in seqs):
                    seqs = saved
                    counts[s] = len(saved)
            result.extend([s, r, sum(self.q[i] for i in r)] for r in seqs)
        return result

    def integrated_construct(self, deadline):
        if time.perf_counter() >= deadline or milp is None:
            return None
        m, H = self.m, self.H
        sizes = sorted(set(self.q), reverse=True)
        J, P = len(sizes), H * m
        gc = [Counter(self.q[i] for i in members) for members in self.groups]
        B = [min(self.maxv[s], self.T2, self.n) for s in range(m)]
        c = [v for row in self.group_cost for v in row]
        lb, ub, it = [0] * P, [1] * P, [1] * P
        ys, zs = [], []
        for s in range(m):
            sy, sz = [], []
            for b in range(B[s]):
                sy.append(len(c))
                c.append(1e-5)
                lb.append(0); ub.append(1); it.append(1)
                z = []
                for q in sizes:
                    z.append(len(c))
                    c.append(0)
                    lb.append(0)
                    ub.append(min(self.n, self.Q2 // q) if q else self.n)
                    it.append(1)
                sz.append(z)
            ys.append(sy); zs.append(sz)
        model = _Model(c, lb, ub, it)
        for h in range(H):
            model.row(((h * m + s, 1) for s in range(m)), 1, 1)
        for s in range(m):
            model.row(((h * m + s, self.gd[h]) for h in range(H)), hi=self.cap[s])
            for j, q in enumerate(sizes):
                model.row([(zs[s][b][j], 1) for b in range(B[s])] +
                          [(h * m + s, -gc[h][q]) for h in range(H)], 0, 0)
            for b in range(B[s]):
                terms = [(zs[s][b][j], q) for j, q in enumerate(sizes)]
                model.row(terms + [(ys[s][b], -self.Q2)], hi=0)
                model.row([(v, 1) for v in zs[s][b]] + [(ys[s][b], -self.n)], hi=0)
                if b:
                    model.row([(ys[s][b], 1), (ys[s][b - 1], -1)], hi=0)
                    model.row(terms + [(zs[s][b - 1][j], -q) for j, q in enumerate(sizes)], hi=0)
        model.row(((v, 1) for sy in ys for v in sy), hi=self.T2)
        result = model.run(deadline)
        if result is None:
            return None
        a = [int(np.argmax(result[h * m:(h + 1) * m])) for h in range(H)]
        routes = []
        try:
            for s in range(m):
                bysize = {q: [] for q in sizes}
                for h in range(H):
                    if a[h] == s:
                        for i in self.groups[h]:
                            bysize[self.q[i]].append(i)
                for b in range(B[s]):
                    seq = []
                    for j, q in enumerate(sizes):
                        for _ in range(max(0, int(round(result[zs[s][b][j]])))):
                            seq.append(bysize[q].pop())
                    if seq:
                        seq = self.tour(s, seq, deadline)
                        routes.append([s, seq, sum(self.q[i] for i in seq)])
                if any(bysize.values()):
                    return None
        except (IndexError, ValueError):
            return None
        return (routes, a) if self.valid(routes, a) else None

    def first(self, demands, randomize=False):
        key = tuple(demands)
        if key in self.fc and not randomize:
            return self.fc[key]
        active = [s for s in range(self.m) if demands[s] > 1e-9]
        order = sorted(active, key=lambda s: math.atan2(self.y[self.n + s] - self.y[self.depot],
                                                      self.x[self.n + s] - self.x[self.depot]))
        if randomize:
            self.rng.shuffle(order)
        candidates = []
        # Sequential splitting can always build a feasible first echelon when the total transport capacity is sufficient.
        cargo, load = [], 0.0
        for s in order:
            rest = float(demands[s])
            while rest > 1e-8:
                if not cargo or load >= self.Q1 - 1e-8:
                    cargo.append({})
                    load = 0.0
                take = min(rest, self.Q1 - load)
                cargo[-1][s] = cargo[-1].get(s, 0.0) + take
                load += take
                rest -= take
        if len(cargo) <= self.T1:
            candidates.append(cargo)
        # Keep full-truckload direct shipments first, then merge the less-than-truckload remainders.
        chunks, residual = [], []
        for s in order:
            full = int(demands[s] // self.Q1)
            chunks.extend({s: self.Q1} for _ in range(full))
            rem = demands[s] - full * self.Q1
            if rem > 1e-8:
                residual.append((rem, s))
        residual.sort(reverse=True)
        partial, pload = [], []
        for rem, s in residual:
            fits = [b for b in range(len(partial)) if pload[b] + rem <= self.Q1 + 1e-8]
            if fits:
                def extra(b):
                    return min(self.d[self.n + s][self.n + t] + self.d[self.depot][self.n + s]
                               - self.d[self.depot][self.n + t] for t in partial[b])
                b = min(fits, key=extra)
                partial[b][s] = rem
                pload[b] += rem
            else:
                partial.append({s: rem}); pload.append(rem)
        if len(chunks) + len(partial) <= self.T1:
            candidates.append(chunks + partial)
        record = self.fc.get(key)
        for cargo in candidates:
            truck_routes, cost = [], 0.0
            for delivery in cargo:
                left = set(delivery)
                seq, prev = [], self.depot
                while left:
                    s = min(left, key=lambda a: self.d[prev][self.n + a])
                    seq.append(s); left.remove(s); prev = self.n + s
                # Full 2-exchange on satellite routes, without changing delivered quantities.
                changed = True
                while changed and time.perf_counter() < self.end:
                    changed = False
                    for i in range(len(seq) - 1):
                        a = self.depot if i == 0 else self.n + seq[i - 1]
                        b = self.n + seq[i]
                        for j in range(i + 1, len(seq)):
                            c = self.n + seq[j]
                            e = self.depot if j + 1 == len(seq) else self.n + seq[j + 1]
                            if self.d[a][c] + self.d[b][e] < self.d[a][b] + self.d[c][e] - 1e-9:
                                seq[i:j + 1] = reversed(seq[i:j + 1])
                                changed = True
                                break
                        if changed:
                            break
                prev = self.depot
                for s in seq:
                    cost += self.d[prev][self.n + s]
                    prev = self.n + s
                cost += self.d[prev][self.depot]
                truck_routes.append(seq)
            if record is None or cost < record[0] - 1e-9:
                record = (cost, truck_routes, cargo)
        self.fc[key] = record
        return record

    def valid(self, routes, a):
        if len(a) != self.H or len(routes) > self.T2:
            return False
        seen, counts, loads = [0] * self.n, [0] * self.m, [0] * self.m
        for s, seq, load in routes:
            if not seq or s < 0 or s >= self.m:
                return False
            total = 0
            for i in seq:
                if i < 0 or i >= self.n or a[self.gh[i]] != s:
                    return False
                seen[i] += 1
                total += self.q[i]
            if total > self.Q2 or abs(total - load) > 1e-6:
                return False
            counts[s] += 1; loads[s] += total
        return (all(v == 1 for v in seen) and
                all(counts[s] <= self.maxv[s] and loads[s] <= self.cap[s] + 1e-8 for s in range(self.m)) and
                sum(loads) <= self.Q1 * self.T1 + 1e-8)

    def state(self, routes, a):
        loads = [0] * self.m
        for s, seq, q in routes:
            loads[s] += q
        first = self.first(loads)
        if first is None:
            return None
        score = self.travel(routes) + sum(loads[s] * self.hand[s] for s in range(self.m)) + first[0]
        return [routes, a[:], loads, score]

    def refresh(self, state):
        state[3] = self.travel(state[0]) + sum(state[2][s] * self.hand[s] for s in range(self.m)) + self.first(state[2])[0]

    def register(self, routes):
        for s, seq, load in routes:
            key = (s, tuple(sorted(seq)))
            cost = self.route_cost(s, seq)
            old = self.pool.get(key)
            if old is None or cost < old[3] - 1e-9:
                self.pool[key] = (s, seq[:], load, cost)
        if len(self.pool) > self.pool_limit:
            protected = {(s, tuple(sorted(seq))) for s, seq, load in self.best[0]} if self.best else set()
            for key in list(self.pool):
                if len(self.pool) <= self.pool_limit:
                    break
                if key not in protected:
                    del self.pool[key]

    def consider(self, routes, a):
        if not self.valid(routes, a):
            return None
        st = self.state(routes, a)
        if st is None:
            return None
        if self.best is not None:
            self.refresh(self.best)
        if self.best is None or st[3] < self.best[3] - 1e-8:
            self.best = [[s, seq[:], load] for s, seq, load in routes], a[:], st[2][:], st[3]
            self.best = list(self.best)
        self.register(routes)
        return st

    def remove(self, routes, removed):
        result = []
        for s, seq, load in routes:
            r = [i for i in seq if i not in removed]
            if r:
                result.append([s, r, sum(self.q[i] for i in r)])
        return result

    def insertion(self, s, seq, i):
        ds, di = self.d[self.n + s], self.d[i]
        if not seq:
            return 2 * ds[i], 0
        best, pos = ds[i] + di[seq[0]] - ds[seq[0]], 0
        for p in range(1, len(seq)):
            value = di[seq[p - 1]] + di[seq[p]] - self.d[seq[p - 1]][seq[p]]
            if value < best:
                best, pos = value, p
        value = di[seq[-1]] + ds[i] - ds[seq[-1]]
        if value < best:
            best, pos = value, len(seq)
        return best, pos

    def repair_fixed(self, routes, removed, a, deadline):
        pending = list(removed)
        self.rng.shuffle(pending)
        counts = [0] * self.m
        for s, seq, load in routes:
            counts[s] += 1
        while pending and time.perf_counter() < deadline:
            chosen = None
            for i in pending:
                if time.perf_counter() >= deadline:
                    return None
                s, options = a[self.gh[i]], []
                for b, (t, seq, load) in enumerate(routes):
                    if t == s and load + self.q[i] <= self.Q2:
                        delta, p = self.insertion(s, seq, i)
                        options.append((delta, b, p))
                if counts[s] < self.maxv[s] and len(routes) < self.T2:
                    options.append((2 * self.d[self.n + s][i], len(routes), 0))
                if not options:
                    return None
                options.sort()
                regret = options[1][0] - options[0][0] if len(options) > 1 else 1e12
                priority = (regret, self.q[i], -options[0][0])
                if chosen is None or priority > chosen[0]:
                    chosen = (priority, i, s, options[0])
            _, i, s, (_, b, p) = chosen
            if b == len(routes):
                routes.append([s, [i], self.q[i]])
                counts[s] += 1
            else:
                routes[b][1].insert(p, i)
                routes[b][2] += self.q[i]
            pending.remove(i)
        return routes if not pending else None

    def place_group(self, existing, members, s, slots, deadline):
        local = [[s, seq[:], load] for _, seq, load in existing]
        members = sorted(members, key=lambda i: (-self.q[i], -self.d[self.n + s][i]))
        for i in members:
            if time.perf_counter() >= deadline:
                return None
            choice = None
            for b, (_, seq, load) in enumerate(local):
                if load + self.q[i] <= self.Q2:
                    delta, p = self.insertion(s, seq, i)
                    if choice is None or delta < choice[0]:
                        choice = (delta, b, p)
            if choice is None:
                if len(local) >= slots:
                    allnodes = [v for _, seq, _ in existing for v in seq] + members
                    bins = self.bfd(allnodes)
                    if len(bins) > slots:
                        return None
                    return [[s, self.tour(s, r, deadline), sum(self.q[v] for v in r)] for r in bins]
                local.append([s, [i], self.q[i]])
            else:
                _, b, p = choice
                local[b][1].insert(p, i)
                local[b][2] += self.q[i]
        return local

    def repair_groups(self, routes, a, selected, deadline, forced=None):
        loads = [0] * self.m
        for s, seq, q in routes:
            loads[s] += q
        order = list(selected)
        order.sort(key=lambda h: self.gd[h] * self.rng.uniform(0.65, 1.35), reverse=True)
        for h in order:
            if time.perf_counter() >= deadline:
                return None
            best = None
            sats = [forced[h]] if forced and h in forced else list(range(self.m))
            self.rng.shuffle(sats)
            for s in sats:
                if loads[s] + self.gd[h] > self.cap[s]:
                    continue
                existing = [r for r in routes if r[0] == s]
                slots = min(self.maxv[s], self.T2 - len(routes) + len(existing))
                local = self.place_group(existing, self.groups[h], s, slots, deadline)
                if local is None:
                    continue
                gain = self.travel(local) - self.travel(existing) + self.hand[s] * self.gd[h]
                old, new = loads[s], loads[s] + self.gd[h]
                gain += self.d[self.depot][self.n + s] * ((new - old) / max(1, self.Q1) +
                         math.ceil(new / max(1, self.Q1)) - math.ceil(old / max(1, self.Q1)))
                if best is None or gain < best[0]:
                    best = (gain, s, local)
            if best is None:
                return None
            _, s, local = best
            routes = [r for r in routes if r[0] != s] + local
            a[h] = s
            loads[s] += self.gd[h]
        return routes, a

    def local_search(self, routes, deadline):
        owner, pos = [-1] * self.n, [-1] * self.n
        def update(b):
            for p, i in enumerate(routes[b][1]):
                owner[i], pos[i] = b, p
        for b in range(len(routes)):
            update(b)
        changed = True
        while changed and time.perf_counter() < deadline:
            changed = False
            order = list(range(self.n))
            self.rng.shuffle(order)
            for i in order:
                if time.perf_counter() >= deadline:
                    break
                b = owner[i]
                if b < 0:
                    continue
                s, ra, la = routes[b]
                p = pos[i]
                depot = self.n + s
                for j in self.nearest(i):
                    t, q = owner[j], pos[j]
                    if t < 0 or routes[t][0] != s:
                        continue
                    rb, lb = routes[t][1], routes[t][2]
                    ap = depot if p == 0 else ra[p - 1]
                    an = depot if p + 1 == len(ra) else ra[p + 1]
                    bp = depot if q == 0 else rb[q - 1]
                    bn = depot if q + 1 == len(rb) else rb[q + 1]
                    if b == t:
                        l, r = min(p, q), max(p, q)
                        if r == l:
                            continue
                        a0 = depot if l == 0 else ra[l - 1]
                        a1, b0 = ra[l], ra[r]
                        b1 = depot if r + 1 == len(ra) else ra[r + 1]
                        if self.d[a0][b0] + self.d[a1][b1] < self.d[a0][a1] + self.d[b0][b1] - 1e-8:
                            ra[l:r + 1] = reversed(ra[l:r + 1])
                            update(b); changed = True
                            break
                        if r > l + 1:
                            a0, a1 = ra[l], ra[l + 1]
                            if self.d[a0][b0] + self.d[a1][b1] < self.d[a0][a1] + self.d[b0][b1] - 1e-8:
                                ra[l + 1:r + 1] = reversed(ra[l + 1:r + 1])
                                update(b); changed = True
                                break
                        continue
                    removal = self.d[ap][an] - self.d[ap][i] - self.d[i][an]
                    if lb + self.q[i] <= self.Q2:
                        before = self.d[bp][i] + self.d[i][j] - self.d[bp][j]
                        after = self.d[j][i] + self.d[i][bn] - self.d[j][bn]
                        if removal + min(before, after) < -1e-8:
                            ra.pop(p); rb.insert(q if before <= after else q + 1, i)
                            routes[b][2] -= self.q[i]; routes[t][2] += self.q[i]
                            update(b); update(t); changed = True
                            break
                    if la - self.q[i] + self.q[j] <= self.Q2 and lb - self.q[j] + self.q[i] <= self.Q2:
                        delta = (self.d[ap][j] + self.d[j][an] + self.d[bp][i] + self.d[i][bn]
                                 - self.d[ap][i] - self.d[i][an] - self.d[bp][j] - self.d[j][bn])
                        if delta < -1e-8:
                            ra[p], rb[q] = j, i
                            routes[b][2] += self.q[j] - self.q[i]
                            routes[t][2] += self.q[i] - self.q[j]
                            update(b); update(t); changed = True
                            break
                    delta = self.d[i][j] + self.d[bp][an] - self.d[i][an] - self.d[bp][j]
                    if delta < -1e-8:
                        pa = sum(self.q[v] for v in ra[:p + 1])
                        pb = sum(self.q[v] for v in rb[:q])
                        na, nb = pa + lb - pb, pb + la - pa
                        if na <= self.Q2 and nb <= self.Q2:
                            routes[b][1], routes[t][1] = ra[:p + 1] + rb[q:], rb[:q] + ra[p + 1:]
                            routes[b][2], routes[t][2] = na, nb
                            update(b); update(t); changed = True
                            break
        routes[:] = [r for r in routes if r[1]]

    def recombine(self, deadline):
        columns = list(self.pool.values())
        R, m, H = len(columns), self.m, self.H
        if not R or time.perf_counter() >= deadline:
            return None
        c = [cost + load * (self.hand[s] + 2 * self.d[self.depot][self.n + s] / max(1, self.Q1))
             for s, seq, load, cost in columns] + [0.] * (H * m)
        model = _Model(c, [0] * len(c), [1] * len(c), [1] * len(c))
        coverage = [[] for _ in range(self.n)]
        hg = [[[] for _ in range(m)] for _ in range(H)]
        satcols = [[] for _ in range(m)]
        for r, (s, seq, load, cost) in enumerate(columns):
            satcols[s].append(r)
            for i in seq:
                coverage[i].append((r, 1))
            for h, count in Counter(self.gh[i] for i in seq).items():
                hg[h][s].append((r, count))
        for row in coverage:
            model.row(row, 1, 1)
        for h in range(H):
            model.row(((R + h * m + s, 1) for s in range(m)), 1, 1)
            for s in range(m):
                model.row(hg[h][s] + [(R + h * m + s, -len(self.groups[h]))], 0, 0)
        for s in range(m):
            model.row(((r, 1) for r in satcols[s]), hi=self.maxv[s])
            model.row(((r, columns[r][2]) for r in satcols[s]), hi=self.cap[s])
        model.row(((r, 1) for r in range(R)), hi=self.T2)
        x = model.run(deadline)
        if x is None:
            return None
        routes = [[s, seq[:], load] for r, (s, seq, load, cost) in enumerate(columns) if x[r] > 0.5]
        a = [int(np.argmax(x[R + h * m:R + (h + 1) * m])) for h in range(H)]
        return self.consider(routes, a)

    def run(self):
        if not self.n:
            self.consider([], [0] * self.H)
            return self.output()
        initial_end = min(self.end, self.start + self.limit * 0.24)
        a = self.greedy_assignment()
        if a is not None:
            routes = self.routes_for_assignment(a, initial_end)
            if routes is not None:
                self.consider(routes, a)
        excluded = []
        if time.perf_counter() < initial_end:
            a = self.assignment_mip(min(initial_end, time.perf_counter() + self.limit * 0.10))
            if a is not None:
                routes = self.routes_for_assignment(a, initial_end)
                if routes is not None:
                    self.consider(routes, a)
                else:
                    excluded.append(a)
        if self.best is None:
            # When capacity is tight, decide group assignment and per-vehicle load quantities jointly and directly.
            feasible = self.integrated_construct(min(self.end, self.start + self.limit * 0.78))
            if feasible is not None:
                self.consider(*feasible)
        while self.best is None and time.perf_counter() < self.end:
            a = self.greedy_assignment(True)
            if a is None and milp is not None:
                a = self.assignment_mip(min(self.end, time.perf_counter() + (self.end - time.perf_counter()) * 0.45), excluded)
            if a is not None:
                routes = self.routes_for_assignment(a, self.end)
                if routes is not None:
                    self.consider(routes, a)
                else:
                    excluded.append(a)
        if self.best is None:
            return self.emergency()
        current = self.best
        weights = [1.0, 1.0, 0.7, 0.6]
        last_improve = time.perf_counter()
        interval = self.limit / (2.5 + self.strength)
        next_master = time.perf_counter() + interval
        max_fraction = min(0.48, 0.10 + 0.055 * self.strength)
        while time.perf_counter() < self.end:
            now = time.perf_counter()
            remaining = self.end - now
            if now >= next_master and remaining > self.limit * 0.08:
                recombined = self.recombine(min(self.end, now + min(remaining * 0.22, self.limit * 0.07)))
                if recombined is not None and recombined[3] < current[3]:
                    current = recombined
                next_master = time.perf_counter() + interval
                continue
            if self.rng.random() < min(0.16, self.m / max(1, self.n)):
                self.first(current[2], True)
                self.first(self.best[2], True)
                self.refresh(current); self.refresh(self.best)
            if now - last_improve > self.limit / (2 + self.strength):
                current = self.best
            op = self.rng.choices(range(4), weights=weights)[0]
            stagnation = min(1., (now - last_improve) / max(0.001, self.limit * 0.25))
            fraction = 0.025 + (max_fraction - 0.025) * (self.rng.random() ** (2 - stagnation))
            amount = min(self.n, max(1, int(math.ceil(self.n * fraction))))
            routes, a = current[0], current[1][:]
            candidate = None
            repair_end = min(self.end, now + remaining * min(0.24, 0.025 + 0.006 * self.strength))
            if op == 0 or op == 2:
                if op == 2:
                    selected_route = self.rng.choice(routes)
                    removed = set(selected_route[1])
                    if len(removed) < amount:
                        seed = self.rng.choice(selected_route[1])
                        related = sorted(range(self.n), key=lambda i: self.d[seed][i] * self.rng.uniform(0.7, 1.3))
                        removed.update(related[:amount])
                elif self.rng.random() < 0.35:
                    removed = set(self.rng.sample(range(self.n), amount))
                else:
                    seed = self.rng.randrange(self.n)
                    row = self.d[seed]
                    related = sorted(range(self.n), key=lambda i: row[i] * self.rng.uniform(0.7, 1.3))
                    removed = set(related[:amount])
                partial = self.remove(routes, removed)
                repaired = self.repair_fixed(partial, removed, a, repair_end)
                if repaired is not None:
                    candidate = (repaired, a)
            else:
                nonempty = [h for h in range(self.H) if self.groups[h]]
                seed = self.rng.choice(nonempty)
                forced = None
                if op == 3:
                    other = [h for h in nonempty if a[h] != a[seed]]
                    if other:
                        h = self.rng.choice(other)
                        selected = [seed, h]
                        forced = {seed: a[h], h: a[seed]}
                    else:
                        selected = [seed]
                else:
                    c = self.rng.choice(self.groups[seed])
                    row = self.d[c]
                    related = sorted(nonempty, key=lambda h: min(row[i] for i in self.groups[h]) * self.rng.uniform(0.65, 1.4))
                    selected, count = [], 0
                    for h in related:
                        selected.append(h); count += len(self.groups[h])
                        if count >= amount:
                            break
                removed = {i for h in selected for i in self.groups[h]}
                partial = self.remove(routes, removed)
                candidate = self.repair_groups(partial, a, selected, repair_end, forced)
            if candidate is None:
                weights[op] = 0.99 * weights[op] + 0.01 * 0.2
                continue
            routes, a = candidate
            local_end = min(self.end, time.perf_counter() + (self.end - time.perf_counter()) *
                            min(0.06, 0.005 + 0.004 * self.strength))
            self.local_search(routes, local_end)
            oldbest = self.best[3]
            st = self.consider(routes, a)
            if st is None:
                continue
            self.refresh(current)
            delta = st[3] - current[3]
            progress = min(1., max(0., (time.perf_counter() - self.start) / max(0.001, self.limit)))
            temperature = max(1e-9, abs(self.best[3]) / max(1, self.n) *
                              (0.15 + 0.10 * self.strength) * (1 - progress) ** 2)
            accepted = delta < 0 or self.rng.random() < math.exp(-min(700., max(0., delta) / temperature))
            reward = 0.2
            if accepted:
                current = st
                reward = 1.0
            if self.best[3] < oldbest - 1e-8:
                last_improve = time.perf_counter()
                reward = 5.0
            weights[op] = 0.97 * weights[op] + 0.03 * reward
        return self.output()

    def emergency(self):
        # Used only when no integer-programming solution has been obtained; the complete construction result is still serialised by the common interface.
        a = [min(range(self.m), key=lambda s: self.group_cost[h][s]) for h in range(self.H)]
        nodes = [[] for _ in range(self.m)]
        for h, s in enumerate(a):
            nodes[s].extend(self.groups[h])
        routes = [[s, r, sum(self.q[i] for i in r)] for s in range(self.m) for r in self.bfd(nodes[s])]
        self.best = self.state(routes, a)
        return self.output()

    def output(self):
        routes, a, loads, score = self.best
        first = self.first(loads)
        fr = [[self.depot_id] + [self.sids[s] for s in seq] + [self.depot_id] for seq in first[1]]
        fr += [[self.depot_id, self.depot_id] for _ in range(self.T1 - len(fr))]
        deliveries = []
        for k in range(self.T1):
            cargo = first[2][k] if k < len(first[2]) else {}
            for s in range(self.m):
                deliveries.append({'satellite_id': self.sids[s], 'vehicle_id': k,
                                   'quantity': float(cargo.get(s, 0.0))})
        return {'first_echelon_routes': fr,
                'first_echelon_deliveries': deliveries,
                'second_echelon_routes': [[self.sids[s]] + [self.ids[i] for i in seq] + [self.sids[s]]
                                          for s, seq, load in routes],
                'group_assignments': {'group_' + str(self.gids[h]): self.sids[a[h]] for h in range(self.H)},
                'customer_satellite_assignments': {'customer_' + str(self.ids[i]): self.sids[a[self.gh[i]]]
                                                   for i in range(self.n)}}


def solve(instance: dict, time_limit_s: float) -> dict:
    started = time.perf_counter()
    solver = _Solver(instance, time_limit_s, started)
    try:
        return solver.run()
    except Exception:
        if solver.best is not None:
            return solver.output()
        return solver.emergency()
