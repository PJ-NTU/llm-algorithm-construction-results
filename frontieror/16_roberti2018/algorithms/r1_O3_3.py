"""Group-cooperative two-echelon adaptive neighbourhood search
Fuses random-start construction, the savings method and sequence splitting to jointly improve group assignment, customer routes and splittable replenishment, continuously saving good feasible solutions.
"""
import math
import random
import time
import itertools
from collections import deque

try:
    from scipy.optimize import milp, Bounds, LinearConstraint
    from scipy.sparse import coo_matrix
    import numpy as np
except Exception:
    milp = None


class _Search:
    def __init__(self, data, seconds):
        self.t0 = time.perf_counter()
        self.budget = max(0.001, float(seconds))
        self.end = self.t0 + self.budget - min(self.budget * 0.025, 0.08)
        self.q1 = float(data['Q1'])
        self.q2 = float(data['Q2'])
        self.t1 = int(data['T1'])
        self.t2 = int(data['T2'])
        self.n = int(data['num_customers'])
        self.m = int(data['num_satellites'])
        self.g = int(data['num_groups'])
        self.did = data['depot_id']
        self.sid = list(data['satellites_id'])
        self.cid = list(data['customers_id'])
        self.dem = [float(v) for v in data['customers_demand']]
        self.cap = [float(v) for v in data['satellites_capacity']]
        self.vmax = [min(int(v), self.t2, self.n) for v in data['satellites_max_vehicles']]
        self.h = list(data['satellites_handling_cost'])
        self.effective = [min(self.cap[s], self.q2 * self.vmax[s]) for s in range(self.m)]
        ci = {v: i for i, v in enumerate(self.cid)}
        self.groups = [[ci[c] for c in ids] for ids in data['groups_customer_ids']]
        self.cg = [0] * self.n
        for g, cs in enumerate(self.groups):
            for c in cs:
                self.cg[c] = g
        self.gd = [sum(self.dem[c] for c in cs) for cs in self.groups]
        self.xy = list(zip(data['customers_x'], data['customers_y']))
        self.xy += list(zip(data['satellites_x'], data['satellites_y']))
        self.xy.append((data['depot_x'], data['depot_y']))
        self.depot = self.n + self.m
        self.d = [[math.hypot(x - xx, y - yy) for xx, yy in self.xy] for x, y in self.xy]
        self.center = []
        for cs in self.groups:
            self.center.append((sum(self.xy[c][0] for c in cs) / max(1, len(cs)),
                                sum(self.xy[c][1] for c in cs) / max(1, len(cs))))
        self.proxy = []
        for g, cs in enumerate(self.groups):
            row = []
            for s in range(self.m):
                dist = self.d[self.n + s]
                radial = sum(self.dem[c] * dist[c] for c in cs)
                access = min((dist[c] for c in cs), default=0.0)
                row.append(self.h[s] * self.gd[g] + 2.0 * radial / max(self.q2, 1e-9)
                           + 2.0 * dist[self.depot] * self.gd[g] / max(self.q1, 1e-9)
                           + 0.45 * access)
            self.proxy.append(row)
        self.prefer = [sorted(range(self.m), key=lambda s: self.proxy[g][s]) for g in range(self.g)]
        self.rng = random.Random(1729 + self.n * 10007 + self.g * 101 + self.m)
        self.pool = {}
        self.exact_done = set()
        self.first_cache = {}
        self.tour_cache = {(): (0.0, [])}
        self.pattern_cache = {}
        self.best = None
        self.strength = math.log1p(self.budget)
        self.exact_size = max(3, min(10, int(6 + self.strength)))
        self.cache_limit = max(32, int((self.m + 1) * 24 * (1 + self.strength)))
        self.quantum = self.budget / (40.0 + 3.0 * self.g + math.sqrt(self.n + 1))

    def now(self):
        return time.perf_counter()

    def cost(self, s, route):
        if not route:
            return 0.0
        d = self.d
        hub = self.n + s
        return d[hub][route[0]] + d[route[-1]][hub] + sum(d[a][b] for a, b in zip(route, route[1:]))

    def route_load(self, r):
        return sum(self.dem[c] for c in r)

    def ffd(self, cs):
        routes, loads = [], []
        for c in sorted(cs, key=lambda c: (-self.dem[c], c)):
            b = min((j for j in range(len(routes)) if loads[j] + self.dem[c] <= self.q2 + 1e-8),
                    key=lambda j: self.q2 - loads[j] - self.dem[c], default=-1)
            if b < 0:
                routes.append([c])
                loads.append(self.dem[c])
            else:
                routes[b].append(c)
                loads[b] += self.dem[c]
        return routes

    def pack_exact(self, cs, limit, stop):
        routes = self.ffd(cs)
        if len(routes) <= limit:
            return routes
        if limit <= 0 or sum(self.dem[c] for c in cs) > self.q2 * limit + 1e-8:
            return None
        ordered = sorted(cs, key=lambda c: -self.dem[c])
        bins = [[] for _ in range(limit)]
        loads = [0.0] * limit
        # An explicit stack avoids hitting the recursion-depth limit when there are many customers.
        stack = [(0, None, None)]
        while stack and self.now() < stop:
            p, choices, placed = stack[-1]
            if placed is not None:
                c = ordered[p]
                bins[placed].pop()
                loads[placed] -= self.dem[c]
                stack[-1] = (p, choices, None)
            if p == len(ordered):
                return [r[:] for r in bins if r]
            if choices is None:
                unique = set()
                choices = []
                c = ordered[p]
                for b in sorted(range(limit), key=lambda b: -loads[b]):
                    if loads[b] not in unique and loads[b] + self.dem[c] <= self.q2 + 1e-8:
                        unique.add(loads[b])
                        choices.append(b)
            if not choices:
                stack.pop()
                continue
            b = choices.pop(0)
            bins[b].append(ordered[p])
            loads[b] += self.dem[ordered[p]]
            stack[-1] = (p, choices, b)
            stack.append((p + 1, None, None))
        return None

    def two_opt(self, s, route, stop):
        r = list(route)
        d = self.d
        hub = self.n + s
        changed = True
        while changed and self.now() < stop:
            changed = False
            for i in range(len(r) - 1):
                if self.now() >= stop:
                    break
                a = r[i - 1] if i else hub
                b = r[i]
                for j in range(i + 1, len(r)):
                    c = r[j]
                    e = r[j + 1] if j + 1 < len(r) else hub
                    if d[a][c] + d[b][e] < d[a][b] + d[c][e] - 1e-8:
                        r[i:j + 1] = reversed(r[i:j + 1])
                        changed = True
                        break
                if changed:
                    break
        return r

    def improve(self, s, routes, stop):
        rs = [self.two_opt(s, r, stop) for r in routes if r]
        d, q, dem = self.d, self.q2, self.dem
        hub = self.n + s
        while self.now() < stop:
            loads = [self.route_load(r) for r in rs]
            move = None
            # Single-customer relocation, including shifts within the same route.
            for a, ra in enumerate(rs):
                if self.now() >= stop:
                    return rs
                for p, c in enumerate(ra):
                    left = ra[p - 1] if p else hub
                    right = ra[p + 1] if p + 1 < len(ra) else hub
                    rem = d[left][right] - d[left][c] - d[c][right]
                    for b, rb in enumerate(rs):
                        if a != b and loads[b] + dem[c] > q + 1e-8:
                            continue
                        for j in range(len(rb) + 1):
                            if a == b and (j == p or j == p + 1):
                                continue
                            x = rb[j - 1] if j else hub
                            y = rb[j] if j < len(rb) else hub
                            delta = rem + d[x][c] + d[c][y] - d[x][y]
                            if delta < -1e-8:
                                move = (a, p, b, j)
                                break
                        if move:
                            break
                    if move:
                        break
                if move:
                    break
            if move:
                a, p, b, j = move
                c = rs[a].pop(p)
                if a == b and j > p:
                    j -= 1
                rs[b].insert(j, c)
                rs = [r for r in rs if r]
                continue
            # Capacity-constrained customer swaps.
            for a in range(len(rs)):
                ra = rs[a]
                if self.now() >= stop:
                    return rs
                for b in range(a + 1, len(rs)):
                    rb = rs[b]
                    for p, c in enumerate(ra):
                        x = ra[p - 1] if p else hub
                        y = ra[p + 1] if p + 1 < len(ra) else hub
                        for j, e in enumerate(rb):
                            if loads[a] - dem[c] + dem[e] > q + 1e-8 or loads[b] - dem[e] + dem[c] > q + 1e-8:
                                continue
                            v = rb[j - 1] if j else hub
                            w = rb[j + 1] if j + 1 < len(rb) else hub
                            delta = (d[x][e] + d[e][y] - d[x][c] - d[c][y]
                                     + d[v][c] + d[c][w] - d[v][e] - d[e][w])
                            if delta < -1e-8:
                                move = (a, p, b, j)
                                break
                        if move:
                            break
                    if move:
                        break
                if move:
                    break
            if move:
                a, p, b, j = move
                rs[a][p], rs[b][j] = rs[b][j], rs[a][p]
                rs[a] = self.two_opt(s, rs[a], stop)
                rs[b] = self.two_opt(s, rs[b], stop)
                continue
            # 2-opt* exchanges the suffixes of two routes and also allows routes to be merged.
            prefix = []
            for r in rs:
                pr = [0.0]
                for c in r:
                    pr.append(pr[-1] + dem[c])
                prefix.append(pr)
            for a in range(len(rs)):
                if self.now() >= stop:
                    return rs
                ra = rs[a]
                for b in range(a + 1, len(rs)):
                    rb = rs[b]
                    for i in range(len(ra) + 1):
                        x = ra[i - 1] if i else hub
                        y = ra[i] if i < len(ra) else hub
                        for j in range(len(rb) + 1):
                            if prefix[a][i] + loads[b] - prefix[b][j] > q + 1e-8:
                                continue
                            if prefix[b][j] + loads[a] - prefix[a][i] > q + 1e-8:
                                continue
                            v = rb[j - 1] if j else hub
                            w = rb[j] if j < len(rb) else hub
                            if d[x][w] + d[v][y] < d[x][y] + d[v][w] - 1e-8:
                                move = (a, i, b, j)
                                break
                        if move:
                            break
                    if move:
                        break
                if move:
                    break
            if not move:
                break
            a, i, b, j = move
            rs[a], rs[b] = rs[a][:i] + rs[b][j:], rs[b][:j] + rs[a][i:]
            rs = [self.two_opt(s, r, stop) for r in rs if r]
        return rs

    def nearest(self, s, cs, randomize):
        left = set(cs)
        routes = []
        hub = self.n + s
        while left:
            if randomize:
                c = self.rng.choice(sorted(left))
            else:
                c = max(left, key=lambda c: self.d[hub][c])
            left.remove(c)
            r, load = [c], self.dem[c]
            while left:
                candidates = [j for j in left if load + self.dem[j] <= self.q2 + 1e-8]
                if not candidates:
                    break
                nxt = min(candidates, key=lambda j: self.d[c][j])
                r.append(nxt)
                load += self.dem[nxt]
                left.remove(nxt)
                c = nxt
            routes.append(r)
        return routes

    def savings(self, s, cs, stop, randomize):
        routes = {c: [c] for c in cs}
        loads = {c: self.dem[c] for c in cs}
        owner = {c: c for c in cs}
        dist = self.d[self.n + s]
        lam = self.rng.uniform(0.7, 1.6) if randomize else 1.0
        noise = self.rng.uniform(0.0, 0.18) if randomize else 0.0
        pairs = []
        for p, c in enumerate(cs):
            if self.now() >= stop:
                return list(routes.values())
            for e in cs[p + 1:]:
                if self.dem[c] + self.dem[e] <= self.q2 + 1e-8:
                    saving = dist[c] + dist[e] - lam * self.d[c][e]
                    if noise:
                        saving += noise * (dist[c] + dist[e]) * self.rng.uniform(-1, 1)
                    pairs.append((saving, c, e))
        pairs.sort(reverse=True)
        for _, c, e in pairs:
            a, b = owner[c], owner[e]
            if a == b or loads[a] + loads[b] > self.q2 + 1e-8:
                continue
            ra, rb = routes[a], routes[b]
            if c not in (ra[0], ra[-1]) or e not in (rb[0], rb[-1]):
                continue
            if ra[-1] != c:
                ra = ra[::-1]
            if rb[0] != e:
                rb = rb[::-1]
            routes[a] = ra + rb
            loads[a] += loads.pop(b)
            for j in rb:
                owner[j] = a
            del routes[b]
        return list(routes.values())

    def insert(self, s, routes, missing, limit, stop):
        rs = [list(r) for r in routes if r]
        loads = [self.route_load(r) for r in rs]
        left = set(missing)
        hub, d = self.n + s, self.d
        while left and self.now() < stop:
            chosen = None
            for c in left:
                choices = []
                for b, r in enumerate(rs):
                    if loads[b] + self.dem[c] > self.q2 + 1e-8:
                        continue
                    delta, pos = min((d[r[j - 1] if j else hub][c]
                                      + d[c][r[j] if j < len(r) else hub]
                                      - d[r[j - 1] if j else hub][r[j] if j < len(r) else hub], j)
                                     for j in range(len(r) + 1))
                    choices.append((delta, b, pos))
                if len(rs) < limit:
                    choices.append((2 * d[hub][c], len(rs), 0))
                if not choices:
                    return None
                choices.sort()
                regret = choices[1][0] - choices[0][0] if len(choices) > 1 else (1 + choices[0][0]) * 1000
                priority = regret + 1e-6 * self.dem[c]
                if chosen is None or priority > chosen[0]:
                    chosen = (priority, c, choices[0])
            _, c, (_, b, pos) = chosen
            if b == len(rs):
                rs.append([c])
                loads.append(self.dem[c])
            else:
                rs[b].insert(pos, c)
                loads[b] += self.dem[c]
            left.remove(c)
        return rs if not left else None

    def split(self, s, seq, limit, stop):
        n = len(seq)
        if not n:
            return {0: (0.0, [])}
        incoming = [[] for _ in range(n + 1)]
        hub, d = self.n + s, self.d
        for i in range(n):
            if self.now() >= stop:
                return {}
            load, value = 0.0, d[hub][seq[i]]
            for j in range(i, n):
                load += self.dem[seq[j]]
                if load > self.q2 + 1e-8:
                    break
                if j > i:
                    value += d[seq[j - 1]][seq[j]]
                incoming[j + 1].append((i, value + d[seq[j]][hub]))
        prev = [float('inf')] * (n + 1)
        prev[0] = 0.0
        trace = []
        result = {}
        for k in range(1, min(limit, n) + 1):
            if self.now() >= stop:
                break
            cur = [float('inf')] * (n + 1)
            tr = [-1] * (n + 1)
            for j in range(k, n + 1):
                for i, cost in incoming[j]:
                    val = prev[i] + cost
                    if val < cur[j]:
                        cur[j], tr[j] = val, i
            trace.append(tr)
            if cur[n] < float('inf'):
                j, routes = n, []
                for kk in range(k - 1, -1, -1):
                    i = trace[kk][j]
                    routes.append(seq[i:j])
                    j = i
                result[k] = (cur[n], routes[::-1])
            prev = cur
        return result

    def small_exact(self, s, cs, limit, stop):
        n = len(cs)
        count = 1 << n
        loads = [0.0] * count
        dp, parent = {}, {}
        hub, d = self.n + s, self.d
        for j in range(n):
            dp[(1 << j, j)] = d[hub][cs[j]]
            parent[(1 << j, j)] = -1
        one, last = {}, {}
        for mask in range(1, count):
            if self.now() >= stop:
                return None
            bit = mask & -mask
            loads[mask] = loads[mask ^ bit] + self.dem[cs[bit.bit_length() - 1]]
            if loads[mask] > self.q2 + 1e-8:
                continue
            bits = [j for j in range(n) if mask & (1 << j)]
            for j in bits:
                old = mask ^ (1 << j)
                if old:
                    value, p = min((dp[(old, i)] + d[cs[i]][cs[j]], i) for i in bits if i != j)
                    dp[(mask, j)], parent[(mask, j)] = value, p
            value, j = min((dp[(mask, j)] + d[cs[j]][hub], j) for j in bits)
            one[mask], last[mask] = value, j
        table = {0: 0.0}
        traces = []
        result = {}
        for k in range(1, min(n, limit) + 1):
            nxt, choices = {}, {}
            for mask in range(1, count):
                if self.now() >= stop:
                    return None
                if mask.bit_count() < k:
                    continue
                anchor = mask & -mask
                sub = mask
                best, chosen = float('inf'), 0
                while sub:
                    if sub & anchor and sub in one and (mask ^ sub) in table:
                        val = one[sub] + table[mask ^ sub]
                        if val < best:
                            best, chosen = val, sub
                    sub = (sub - 1) & mask
                if chosen:
                    nxt[mask], choices[mask] = best, chosen
            traces.append(choices)
            full = count - 1
            if full in nxt:
                routes, mask = [], full
                for kk in range(k - 1, -1, -1):
                    sub = traces[kk][mask]
                    rem, j, r = sub, last[sub], []
                    while j >= 0:
                        r.append(cs[j])
                        p = parent[(rem, j)]
                        rem ^= 1 << j
                        j = p
                    routes.append(r[::-1])
                    mask ^= sub
                result[k] = (nxt[full], routes)
            table = nxt
        return result

    def options(self, s, cs, seed, stop, refresh):
        key = (s, tuple(sorted(cs)))
        if not cs:
            return {0: (0.0, [])}
        opts = dict(self.pool.get(key, {}))
        limit = self.vmax[s]
        if not limit:
            return {}

        def offer(routes):
            if routes is None:
                return
            rs = [list(r) for r in routes if r]
            if len(rs) > limit or any(self.route_load(r) > self.q2 + 1e-8 for r in rs):
                return
            if sorted(c for r in rs for c in r) != sorted(cs):
                return
            value = sum(self.cost(s, r) for r in rs)
            k = len(rs)
            if k not in opts or value < opts[k][0] - 1e-8:
                opts[k] = (value, rs)

        seed = seed or []
        offer(seed)
        if key in self.exact_done:
            return opts
        if not opts:
            packed = self.ffd(cs)
            offer([self.two_opt(s, r, stop) for r in packed])
        if len(cs) <= self.exact_size and self.now() < stop:
            exact = self.small_exact(s, cs, limit, stop)
            if exact is not None:
                opts = exact
                self.exact_done.add(key)
        if key not in self.exact_done and self.now() < stop:
            keep = set(cs)
            partial = [[c for c in r if c in keep] for r in seed]
            present = {c for r in partial for c in r}
            if present != keep:
                offer(self.insert(s, partial, keep - present, limit, stop))
            offer(self.nearest(s, cs, refresh))
            if self.now() < stop:
                offer(self.savings(s, cs, stop, refresh))
            if self.now() < stop:
                hub = self.xy[self.n + s]
                seq = sorted(cs, key=lambda c: math.atan2(self.xy[c][1] - hub[1], self.xy[c][0] - hub[0]))
                if refresh:
                    offset = self.rng.randrange(len(seq))
                    seq = seq[offset:] + seq[:offset]
                    if self.rng.random() < 0.5:
                        seq.reverse()
                for _, routes in self.split(s, seq, limit, stop).values():
                    offer(routes)
            if opts and self.now() < stop:
                selected = min(opts.values(), key=lambda v: v[0])[1]
                offer(self.improve(s, selected, stop))
            if not opts and self.now() < stop:
                packed = self.pack_exact(cs, limit, stop)
                offer(packed)
        self.pool[key] = opts
        if len(self.pool) > self.cache_limit:
            old = next(iter(self.pool))
            del self.pool[old]
            self.exact_done.discard(old)
        return opts

    def sat_tour(self, visits, stop):
        key = tuple(sorted(visits))
        if key in self.tour_cache:
            return self.tour_cache[key]
        hub, d = self.depot, self.d
        remaining = set(key)
        route = []
        last = hub
        while remaining:
            s = min(remaining, key=lambda s: d[last][self.n + s])
            route.append(s)
            remaining.remove(s)
            last = self.n + s
        def cost(r):
            if not r:
                return 0.0
            return d[hub][self.n + r[0]] + d[self.n + r[-1]][hub] + sum(d[self.n + a][self.n + b] for a, b in zip(r, r[1:]))
        best = (cost(route), route)
        if len(key) <= self.exact_size and len(key) > 2 and self.now() < stop:
            table = {}
            parent = {}
            count = len(key)
            for j in range(count):
                table[(1 << j, j)] = d[hub][self.n + key[j]]
                parent[(1 << j, j)] = -1
            complete = True
            for mask in range(1, 1 << count):
                if self.now() >= stop:
                    complete = False
                    break
                for j in range(count):
                    if not mask & (1 << j):
                        continue
                    old = mask ^ (1 << j)
                    if old:
                        val, p = min((table[(old, i)] + d[self.n + key[i]][self.n + key[j]], i)
                                     for i in range(count) if old & (1 << i))
                        table[(mask, j)], parent[(mask, j)] = val, p
            if complete:
                mask = (1 << count) - 1
                val, j = min((table[(mask, j)] + d[self.n + key[j]][hub], j) for j in range(count))
                r = []
                while j >= 0:
                    r.append(key[j])
                    p = parent[(mask, j)]
                    mask ^= 1 << j
                    j = p
                best = (val, r[::-1])
        else:
            change = True
            while change and self.now() < stop:
                change = False
                for i in range(len(route) - 1):
                    a = self.n + route[i - 1] if i else hub
                    b = self.n + route[i]
                    for j in range(i + 1, len(route)):
                        c = self.n + route[j]
                        e = self.n + route[j + 1] if j + 1 < len(route) else hub
                        if d[a][c] + d[b][e] < d[a][b] + d[c][e] - 1e-8:
                            route[i:j + 1] = reversed(route[i:j + 1])
                            change = True
                            break
                    if change:
                        break
            best = (cost(route), route)
        self.tour_cache[key] = best
        return best

    def flow(self, visits, loads):
        kcount = len(visits)
        source, sink = kcount + self.m, kcount + self.m + 1
        graph = [[] for _ in range(sink + 1)]
        def edge(a, b, cap):
            graph[a].append([b, float(cap), len(graph[b])])
            graph[b].append([a, 0.0, len(graph[a]) - 1])
        tracked = []
        for k, ss in enumerate(visits):
            edge(source, k, self.q1)
            for s in ss:
                tracked.append((k, s, len(graph[k])))
                edge(k, kcount + s, self.q1)
        for s in range(self.m):
            edge(kcount + s, sink, loads[s])
        total, value = sum(loads), 0.0
        while value < total - 1e-7:
            parent = [None] * len(graph)
            parent[source] = (-1, -1)
            queue = deque([source])
            while queue and parent[sink] is None:
                a = queue.popleft()
                for j, (b, cap, _) in enumerate(graph[a]):
                    if cap > 1e-8 and parent[b] is None:
                        parent[b] = (a, j)
                        queue.append(b)
            if parent[sink] is None:
                return None
            inc, a = total - value, sink
            while a != source:
                b, j = parent[a]
                inc = min(inc, graph[b][j][1])
                a = b
            a = sink
            while a != source:
                b, j = parent[a]
                e = graph[b][j]
                e[1] -= inc
                graph[a][e[2]][1] += inc
                a = b
            value += inc
        quantities = [[0.0] * self.m for _ in range(self.t1)]
        for k, s, j in tracked:
            e = graph[k][j]
            quantities[k][s] = graph[e[0]][e[2]][1]
        return quantities

    def first(self, loads, stop, refresh=False):
        key = tuple(loads)
        old = self.first_cache.get(key)
        if old is not None and not refresh:
            return old
        active = [s for s in range(self.m) if loads[s] > 1e-8]
        if not active:
            return (0.0, [[] for _ in range(self.t1)], [[0.0] * self.m for _ in range(self.t1)])
        if sum(loads) > self.q1 * self.t1 + 1e-7:
            return None
        best = old
        remaining = list(loads)
        visits = [[] for _ in range(self.t1)]
        quantities = [[0.0] * self.m for _ in range(self.t1)]
        for k in range(self.t1):
            cap, node = self.q1, self.depot
            while cap > 1e-8:
                options = [s for s in active if remaining[s] > 1e-8]
                if not options:
                    break
                s = min(options, key=lambda s: self.d[node][self.n + s])
                take = min(cap, remaining[s])
                visits[k].append(s)
                quantities[k][s] += take
                remaining[s] -= take
                cap -= take
                node = self.n + s
        def offer(vs, qs):
            nonlocal best
            tours = [self.sat_tour(v, stop) for v in vs]
            val = sum(v[0] for v in tours)
            if best is None or val < best[0] - 1e-8:
                best = (val, [v[1] for v in tours], qs)
        offer(visits, quantities)
        orders = [sorted(active, key=lambda s: -loads[s]),
                  sorted(active, key=lambda s: math.atan2(self.xy[self.n + s][1] - self.xy[self.depot][1],
                                                         self.xy[self.n + s][0] - self.xy[self.depot][0]))]
        if refresh:
            order = active[:]
            self.rng.shuffle(order)
            orders.insert(0, order)
        for order in orders:
            if self.now() >= stop:
                break
            qs = [[0.0] * self.m for _ in range(self.t1)]
            vs = [[] for _ in range(self.t1)]
            space = [self.q1] * self.t1
            for s in order:
                amount = loads[s]
                while amount > 1e-8:
                    fit = [k for k in range(self.t1) if space[k] >= amount - 1e-8]
                    if fit:
                        k = min(fit, key=lambda k: (not vs[k], space[k] - amount))
                    else:
                        k = max(range(self.t1), key=lambda k: space[k])
                    take = min(space[k], amount)
                    if take <= 1e-8:
                        break
                    qs[k][s] += take
                    vs[k].append(s)
                    space[k] -= take
                    amount -= take
            offer(vs, qs)
        # For small satellite networks, enumerate route patterns and check splittable replenishment with maximum flow.
        mask_count = (1 << len(active)) - 1 if len(active) < 16 else 0
        predict = 0
        if mask_count:
            for used in range(1, self.t1 + 1):
                predict += math.comb(mask_count + used - 1, used)
                if predict > self.budget * 2500:
                    break
        if mask_count and predict <= self.budget * 2500 and self.now() < stop:
            pkey = tuple(active)
            if pkey not in self.pattern_cache:
                subset = {}
                complete = True
                for mask in range(1, mask_count + 1):
                    if self.now() >= stop:
                        complete = False
                        break
                    ss = [s for p, s in enumerate(active) if mask & (1 << p)]
                    subset[mask] = self.sat_tour(ss, stop)
                patterns = []
                if complete:
                    for used in range(1, self.t1 + 1):
                        for masks in itertools.combinations_with_replacement(range(1, mask_count + 1), used):
                            if self.now() >= stop:
                                complete = False
                                break
                            cover = 0
                            for mask in masks:
                                cover |= mask
                            if cover == mask_count:
                                patterns.append((sum(subset[mask][0] for mask in masks), masks))
                        if not complete:
                            break
                if complete:
                    patterns.sort()
                    self.pattern_cache[pkey] = (patterns, subset)
            if pkey in self.pattern_cache:
                patterns, subset = self.pattern_cache[pkey]
                for val, masks in patterns:
                    if val >= best[0] - 1e-8 or self.now() >= stop:
                        break
                    if len(masks) * self.q1 < sum(loads) - 1e-8:
                        continue
                    if any(sum(bool(mask & (1 << p)) for mask in masks) * self.q1 < loads[s] - 1e-8
                           for p, s in enumerate(active)):
                        continue
                    vs = [subset[mask][1] for mask in masks]
                    qs = self.flow(vs, loads)
                    if qs is not None:
                        best = (val, vs + [[] for _ in range(self.t1 - len(vs))], qs)
                        break
        # After removing redundant visits, reassign the flow instead of fixing the existing delivered quantities.
        if self.now() < stop:
            vs = [v[:] for v in best[1]]
            removals = []
            for k, r in enumerate(vs):
                oldcost = self.sat_tour(r, stop)[0]
                for s in r:
                    nr = [j for j in r if j != s]
                    saving = oldcost - self.sat_tour(nr, stop)[0]
                    if saving > 1e-8:
                        removals.append((saving, k, s))
            removals.sort(reverse=True)
            for _, k, s in removals:
                if self.now() >= stop:
                    break
                trial = [r[:] for r in vs]
                trial[k].remove(s)
                qs = self.flow(trial, loads)
                if qs is not None:
                    vs = trial
                    offer(vs, qs)
        self.first_cache[key] = best
        if len(self.first_cache) > self.cache_limit * 2:
            del self.first_cache[next(iter(self.first_cache))]
        return best

    def evaluate(self, assignment, base, stop, focus=None, seed_routes=None, fast=False):
        loads, customers = [0.0] * self.m, [[] for _ in range(self.m)]
        for g, s in enumerate(assignment):
            loads[s] += self.gd[g]
            customers[s].extend(self.groups[g])
        if any(loads[s] > self.effective[s] + 1e-8 for s in range(self.m)):
            return None
        low = [max(bool(customers[s]), int(math.ceil((loads[s] - 1e-8) / self.q2))) for s in range(self.m)]
        if sum(low) > self.t2:
            return None
        states = {0: (0.0, [])}
        options = []
        for s in range(self.m):
            old = base['routes'][s] if base else []
            unchanged = bool(base) and set(c for r in old for c in r) == set(customers[s])
            key = (s, tuple(sorted(customers[s])))
            if unchanged and focus != s:
                opts = dict(self.pool.get(key, {}))
                val = sum(self.cost(s, r) for r in old)
                if len(old) not in opts or val < opts[len(old)][0]:
                    opts[len(old)] = (val, old)
            elif fast:
                rs = self.ffd(customers[s])
                opts = {len(rs): (sum(self.cost(s, r) for r in rs), rs)} if len(rs) <= self.vmax[s] else {}
            else:
                seed = seed_routes if focus == s and seed_routes is not None else old
                allowance = min(stop, self.now() + max(self.quantum, (stop - self.now()) / max(1, self.m - s)))
                opts = self.options(s, customers[s], seed, allowance, base is not None)
            if not opts:
                return None
            options.append(opts)
        for s, opts in enumerate(options):
            nxt = {}
            for used, (value, choices) in states.items():
                for k, (cost, rs) in opts.items():
                    if used + k > self.t2 or k > self.vmax[s]:
                        continue
                    val = value + cost
                    if used + k not in nxt or val < nxt[used + k][0]:
                        nxt[used + k] = (val, choices + [rs])
            states = nxt
            if not states:
                return None
        value, rs = min(states.values(), key=lambda x: x[0])
        e1 = self.first(loads, stop)
        if e1 is None:
            return None
        return {'a': list(assignment), 'loads': loads, 'routes': rs, 'e1': e1,
                'cost': value + e1[0] + sum(loads[s] * self.h[s] for s in range(self.m))}

    def greedy(self, noise, order_kind):
        if order_kind == 0:
            order = sorted(range(self.g), key=lambda g: -self.gd[g])
        elif order_kind == 1:
            order = sorted(range(self.g), key=lambda g: -(self.proxy[g][self.prefer[g][1]] - self.proxy[g][self.prefer[g][0]]) if self.m > 1 else -self.gd[g])
        else:
            order = list(range(self.g))
            self.rng.shuffle(order)
        a, loads, cs, counts = [-1] * self.g, [0.0] * self.m, [[] for _ in range(self.m)], [0] * self.m
        for g in order:
            choices = []
            for s in range(self.m):
                if loads[s] + self.gd[g] > self.effective[s] + 1e-8:
                    continue
                bins = self.ffd(cs[s] + self.groups[g])
                count = len(bins)
                if count > self.vmax[s] or sum(counts) - counts[s] + count > self.t2:
                    continue
                opening = 0.3 * self.d[self.n + s][self.depot] if not cs[s] else 0.0
                val = self.proxy[g][s] * (1.0 + noise * self.rng.uniform(-1.0, 1.0)) + opening
                val += 0.015 * abs(val) * (loads[s] + self.gd[g]) / max(1.0, self.effective[s])
                choices.append((val, s, count))
            if not choices:
                return None
            _, s, count = min(choices)
            a[g] = s
            loads[s] += self.gd[g]
            cs[s].extend(self.groups[g])
            counts[s] = count
        return a

    def feasibility_mip(self, stop):
        if milp is None or self.now() >= stop:
            return None
        # This auxiliary model only finds an initial solution satisfying the grouping and bin-packing constraints; routes are still generated by the common search.
        try:
            bins = [(s, k) for s in range(self.m) for k in range(self.vmax[s])]
            bcount = len(bins)
            allowed = [s for s in range(self.m) if self.vmax[s] > 0]
            v = {(g, s): j for j, (g, s) in enumerate(itertools.product(range(self.g), allowed))}
            index = len(v)
            p = {}
            for c in range(self.n):
                for b in range(bcount):
                    p[c, b] = index
                    index += 1
            use = list(range(index, index + bcount))
            index += bcount
            rows, cols, vals, lower, upper = [], [], [], [], []
            def row(items, lo, hi):
                r = len(lower)
                for j, val in items:
                    if val:
                        rows.append(r)
                        cols.append(j)
                        vals.append(val)
                lower.append(lo)
                upper.append(hi)
            for g in range(self.g):
                row([(v[g, s], 1) for s in allowed], 1, 1)
            for s in allowed:
                row([(v[g, s], self.gd[g]) for g in range(self.g)], 0, self.cap[s])
            for c in range(self.n):
                for s in allowed:
                    row([(p[c, b], 1) for b, (ss, _) in enumerate(bins) if ss == s] + [(v[self.cg[c], s], -1)], 0, 0)
                if self.now() >= stop:
                    return None
            for b, (s, k) in enumerate(bins):
                row([(p[c, b], self.dem[c]) for c in range(self.n)] + [(use[b], -self.q2)], -float('inf'), 0)
                row([(p[c, b], 1) for c in range(self.n)] + [(use[b], -self.n)], -float('inf'), 0)
                if k:
                    row([(use[b], 1), (use[b - 1], -1)], -float('inf'), 0)
            row([(u, 1) for u in use], 0, self.t2)
            obj = np.zeros(index)
            for (g, s), j in v.items():
                obj[j] = self.proxy[g][s]
            for u in use:
                obj[u] = 1e-4
            mat = coo_matrix((vals, (rows, cols)), shape=(len(lower), index)).tocsc()
            result = milp(obj, integrality=np.ones(index), bounds=Bounds(np.zeros(index), np.ones(index)),
                          constraints=LinearConstraint(mat, lower, upper),
                          options={'time_limit': max(0.001, stop - self.now()), 'mip_rel_gap': 0.1, 'presolve': True})
            if result.x is None:
                return None
            x = result.x
            a = [max(allowed, key=lambda s: x[v[g, s]]) for g in range(self.g)]
            rs = [[] for _ in range(self.m)]
            for b, (s, _) in enumerate(bins):
                route = [c for c in range(self.n) if x[p[c, b]] > 0.5]
                if route:
                    rs[s].append(route)
            loads = [sum(self.gd[g] for g in range(self.g) if a[g] == s) for s in range(self.m)]
            e1 = self.first(loads, stop)
            if e1 is None:
                return None
            plan = {'a': a, 'loads': loads, 'routes': rs, 'e1': e1,
                    'cost': e1[0] + sum(self.h[s] * loads[s] + sum(self.cost(s, r) for r in rs[s]) for s in range(self.m))}
            return plan if self.valid(plan) else None
        except Exception:
            return None

    def valid(self, plan):
        if plan is None:
            return False
        seen, counts, load = [], [], []
        for s, rs in enumerate(plan['routes']):
            counts.append(len(rs))
            total = 0.0
            for r in rs:
                q = self.route_load(r)
                if q > self.q2 + 1e-6 or any(plan['a'][self.cg[c]] != s for c in r):
                    return False
                total += q
                seen.extend(r)
            load.append(total)
        if sorted(seen) != list(range(self.n)) or sum(counts) > self.t2:
            return False
        if any(counts[s] > self.vmax[s] or load[s] > self.cap[s] + 1e-6 for s in range(self.m)):
            return False
        _, visits, qs = plan['e1']
        if len(visits) != self.t1 or len(qs) != self.t1:
            return False
        for k in range(self.t1):
            if len(visits[k]) != len(set(visits[k])) or sum(qs[k]) > self.q1 + 1e-6:
                return False
            if any(qs[k][s] < -1e-7 or (qs[k][s] > 1e-7 and s not in visits[k]) for s in range(self.m)):
                return False
        return all(abs(sum(qs[k][s] for k in range(self.t1)) - load[s]) <= 1e-5 for s in range(self.m))

    def mutate(self, base, op, age):
        a = list(base['a'])
        if not self.g or self.m < 2:
            return a
        g = self.rng.randrange(self.g)
        if op == 1:
            choices = self.prefer[g]
            if self.rng.random() < 0.8:
                width = min(self.m, max(2, int(1 + self.strength)))
                choices = choices[:width]
            a[g] = self.rng.choice([s for s in choices if s != a[g]] or list(range(self.m)))
        elif op == 2:
            others = [j for j in range(self.g) if a[j] != a[g]]
            if others:
                h = self.rng.choice(others)
                a[g], a[h] = a[h], a[g]
        else:
            upper = min(self.g, max(1, int(math.sqrt(self.g) * (0.4 + self.strength * (0.15 + age)))))
            count = self.rng.randint(1, upper)
            if self.rng.random() < 0.7:
                x, y = self.center[g]
                order = sorted(range(self.g), key=lambda j: math.hypot(self.center[j][0] - x, self.center[j][1] - y)
                               * self.rng.uniform(0.7, 1.3))
                removed = order[:count]
            else:
                removed = self.rng.sample(range(self.g), count)
            loads = list(base['loads'])
            for j in removed:
                loads[a[j]] -= self.gd[j]
                a[j] = -1
            removed.sort(key=lambda j: -self.gd[j] * self.rng.uniform(0.7, 1.3))
            for j in removed:
                feasible = [s for s in range(self.m) if loads[s] + self.gd[j] <= self.effective[s] + 1e-8]
                if not feasible:
                    return None
                s = min(feasible, key=lambda s: self.proxy[j][s] * self.rng.uniform(0.45, 1.55)
                        + (0.3 * self.d[self.n + s][self.depot] if loads[s] < 1e-8 else 0))
                a[j] = s
                loads[s] += self.gd[j]
        return a

    def route_seed(self, base, s, age, stop):
        routes = base['routes'][s]
        cs = [c for r in routes for c in r]
        if not cs:
            return routes
        root = self.rng.choice(cs)
        upper = min(len(cs), max(2, int(math.sqrt(len(cs)) * (0.5 + self.strength * (0.15 + age)))))
        count = self.rng.randint(1, upper)
        if self.rng.random() < 0.7:
            removed = set(sorted(cs, key=lambda c: self.d[root][c] * self.rng.uniform(0.6, 1.4))[:count])
        else:
            removed = set(self.rng.sample(cs, count))
        partial = [[c for c in r if c not in removed] for r in routes]
        limit = min(self.vmax[s], self.t2 - sum(len(base['routes'][j]) for j in range(self.m) if j != s))
        inserted = self.insert(s, partial, removed, limit, stop)
        if inserted is None:
            return routes
        return self.improve(s, inserted, stop)

    def run(self):
        # Save a complete feasible solution first, so no later time-consuming step can overwrite it.
        a = self.greedy(0.0, 0)
        if a is not None:
            plan = self.evaluate(a, None, min(self.end, self.now() + self.quantum), fast=True)
            if self.valid(plan):
                self.best = plan
        init_stop = min(self.end, self.t0 + self.budget * 0.15)
        phase = 1
        while self.best is None and self.now() < init_stop:
            a = self.greedy(0.25 * self.rng.random(), phase % 3)
            phase += 1
            if a is not None:
                plan = self.evaluate(a, None, min(init_stop, self.now() + self.quantum), fast=True)
                if self.valid(plan):
                    self.best = plan
        if self.best is None and self.now() < self.end:
            plan = self.feasibility_mip(min(self.end, self.t0 + self.budget * 0.75))
            if plan is not None:
                self.best = plan
        while self.best is None and self.now() < self.end:
            a = self.greedy(self.rng.random(), 2)
            if a is not None:
                plan = self.evaluate(a, None, self.end)
                if self.valid(plan):
                    self.best = plan
        if self.best is None:
            return
        current = self.best
        # For small instances, traverse group assignments in an interleaved order; the main loop is still controlled by the wall clock.
        exhaustive = None
        if self.n <= self.exact_size and self.g * math.log(max(1, self.m)) <= math.log(max(1, self.budget * 10000)):
            exhaustive = iter(itertools.product(range(self.m), repeat=self.g))
        weights = [2.5, 2.0, 1.4, 1.5, 0.5]
        last_improvement, last_restart = self.now(), self.now()
        while self.now() < self.end:
            remaining = self.end - self.now()
            age = min(1.0, (self.now() - last_improvement) / max(self.budget * 0.25, 1e-9))
            op = self.rng.choices(range(5), weights=weights, k=1)[0]
            allowance = self.quantum * (1 + self.strength * (0.25 + age))
            stop = min(self.end, self.now() + allowance)
            plan = None
            if exhaustive is not None and self.rng.random() < 0.45:
                candidate = next(exhaustive, None)
                if candidate is None:
                    exhaustive = None
                else:
                    plan = self.evaluate(candidate, current, stop)
            elif op == 0 or self.m == 1 or not self.g:
                active = [s for s in range(self.m) if current['routes'][s]]
                if active:
                    s = self.rng.choices(active, weights=[1 + len(current['routes'][s]) for s in active], k=1)[0]
                    seed_stop = min(stop, self.now() + allowance * 0.45)
                    seed = self.route_seed(current, s, age, seed_stop)
                    plan = self.evaluate(current['a'], current, stop, focus=s, seed_routes=seed)
            elif op == 4:
                e1 = self.first(current['loads'], stop, refresh=True)
                if e1 is not None:
                    plan = dict(current)
                    plan['e1'] = e1
                    plan['cost'] = current['cost'] - current['e1'][0] + e1[0]
            else:
                a = self.mutate(current, op, age)
                if a is not None:
                    plan = self.evaluate(a, current, stop)
            reward = 0.1
            if plan is not None and self.valid(plan):
                delta = plan['cost'] - current['cost']
                fraction = max(0.0, remaining / self.budget)
                temp = max(1e-9, self.best['cost'] / max(10, self.n + self.g) * 0.12 * fraction * fraction)
                if delta < -1e-8 or self.rng.random() < math.exp(-max(0.0, delta) / temp):
                    current = plan
                    reward = 0.5 if delta >= -1e-8 else 2.0
                if plan['cost'] < self.best['cost'] - 1e-8:
                    self.best = plan
                    last_improvement = self.now()
                    reward = 6.0
            weights[op] = max(0.2, 0.92 * weights[op] + 0.08 * reward)
            restart_span = self.budget / (3 + self.strength)
            if self.now() - max(last_improvement, last_restart) > restart_span:
                current = self.best
                last_restart = self.now()
                weights = [2.5, 2.0, 1.4, 1.5, 0.5]

    def output(self):
        p = self.best
        if p is None:
            # The interface fallback is used only when there is no feasible solution or the runtime environment fails; it is never treated as a feasible solution during the search.
            a = [self.prefer[g][0] for g in range(self.g)]
            rs = [[] for _ in range(self.m)]
            for g, s in enumerate(a):
                rs[s].extend([[c] for c in self.groups[g]])
            loads = [sum(self.route_load(r) for r in rs[s]) for s in range(self.m)]
            e1 = self.first(loads, self.now())
            if e1 is None:
                e1 = (0.0, [[] for _ in range(self.t1)], [[0.0] * self.m for _ in range(self.t1)])
            p = {'a': a, 'routes': rs, 'e1': e1}
        a, rs = p['a'], p['routes']
        _, visits, quantities = p['e1']
        return {
            'first_echelon_routes': [[self.did] + [self.sid[s] for s in r] + [self.did] for r in visits],
            'first_echelon_deliveries': [{'satellite_id': self.sid[s], 'vehicle_id': k, 'quantity': float(quantities[k][s])}
                                        for k in range(self.t1) for s in range(self.m)],
            'second_echelon_routes': [[self.sid[s]] + [self.cid[c] for c in r] + [self.sid[s]]
                                      for s in range(self.m) for r in rs[s] if r],
            'group_assignments': {'group_%d' % g: self.sid[a[g]] for g in range(self.g)},
            'customer_satellite_assignments': {'customer_%s' % self.cid[c]: self.sid[a[self.cg[c]]] for c in range(self.n)},
        }


def solve(instance: dict, time_limit_s: float) -> dict:
    search = _Search(instance, time_limit_s)
    try:
        search.run()
    except Exception:
        # The saved good feasible solution is unaffected by unfinished candidate construction.
        pass
    return search.output()
