"""Joint inventory-timing and routing neighbourhood search
Combines inventory integer programming, single-retailer dynamic programming, pairwise timing enumeration and route local search, keeping the best feasible solution within the wall-clock limit.
"""
import math
import random
import time

import numpy as np
try:
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import coo_matrix
except Exception:
    milp = None


class _Search:
    def __init__(self, data, seconds, start):
        self.n = int(data['num_retailers'])
        self.h = int(data['time_horizon'])
        self.budget = max(0.0, float(seconds))
        self.start = start
        reserve = min(self.budget * .04, .025 + self.n * self.h * .000003)
        self.end = start + max(0.0, self.budget - reserve)
        self.rng = random.Random(int(data.get('seed', 0)) + 71821)
        self.r = np.asarray(data['retailers_demand'], dtype=float)
        self.u = np.asarray(data['retailers_max_inventory'], dtype=float)
        self.i0 = np.asarray(data['retailers_starting_inventory'], dtype=float)
        self.h0 = float(data['supplier_inventory_cost'])
        self.hs = np.asarray(data['retailers_inventory_cost'], dtype=float)
        self.b0 = float(data['supplier_starting_inventory'])
        self.prod = float(data['supplier_production_rate'])
        self.cap = float(data['vehicle_capacity'])
        raw = data['distance_matrix']
        self.d = [[float(raw[max(i, j)][min(i, j)]) if i != j else 0.0
                   for j in range(self.n + 1)] for i in range(self.n + 1)]
        self.weights = (self.hs - self.h0)[:, None] * np.arange(self.h, 0, -1)[None, :]
        self.supply = self.b0 + self.prod * np.arange(self.h)
        self.z = np.zeros((self.n, self.h), dtype=np.int8)
        self.x = np.zeros((self.n, self.h))
        self.routes = [[] for _ in range(self.h)]
        self.best = None
        self.best_cost = math.inf
        self.cache = {}
        self.patterns = [None] * self.n
        self.near = [sorted((j for j in range(self.n) if j != s),
                            key=lambda j: self.d[s + 1][j + 1]) for s in range(self.n)]
        self.exact_size = max(3, min(13, int(math.log2(2 + 18000 * max(.01, self.budget)
                                                       / max(1, self.n * self.h)))))
        self.loads = np.zeros(self.h)
        self.value = math.inf

    def remaining(self):
        return self.end - time.perf_counter()

    def route_cost(self, route):
        if not route:
            return 0.0
        d = self.d
        return d[0][route[0]] + sum(d[a][b] for a, b in zip(route, route[1:])) + d[route[-1]][0]

    def amount(self, s, visits):
        q = np.zeros(self.h)
        inv = self.i0[s]
        for t in range(self.h):
            if visits[t]:
                q[t] = self.u[s] - inv
                inv = self.u[s]
            inv -= self.r[s]
        return q

    def feasible(self, z, x):
        tol = 1e-6
        if np.any(x < -tol):
            return False
        inv = self.i0.copy()
        b = self.b0
        if np.any(inv < -tol) or np.any(inv > self.u + tol):
            return False
        for t in range(self.h):
            q = x[:, t]
            wanted = np.where(z[:, t] != 0, self.u - inv, 0.0)
            if np.max(np.abs(q - wanted), initial=0.0) > tol:
                return False
            load = float(q.sum())
            if load > self.cap + tol or load > b + tol:
                return False
            inv += q - self.r
            b += self.prod - load
            if b < -tol or np.any(inv < -tol) or np.any(inv > self.u + tol):
                return False
        return True

    def remember(self):
        self.loads = self.x.sum(axis=0)
        self.value = float(np.sum(self.x * self.weights)) + sum(self.route_cost(p) for p in self.routes)
        if self.value < self.best_cost - 1e-7 and self.feasible(self.z, self.x):
            self.best_cost = self.value
            self.best = (self.z.copy(), self.x.copy(), [p[:] for p in self.routes])
            return True
        return False

    def restore(self):
        if self.best is not None:
            self.z = self.best[0].copy()
            self.x = self.best[1].copy()
            self.routes = [p[:] for p in self.best[2]]
            self.loads = self.x.sum(axis=0)
            self.value = self.best_cost

    def insert(self, base, node):
        d = self.d
        best = math.inf
        pos = 0
        for k in range(len(base) + 1):
            a = base[k - 1] if k else 0
            b = base[k] if k < len(base) else 0
            delta = d[a][node] + d[node][b] - d[a][b]
            if delta < best:
                best, pos = delta, k
        return best, base[:pos] + [node] + base[pos:]

    def improve_route(self, route, stop, exact=False):
        if len(route) < 3:
            return route[:]
        key = tuple(sorted(route))
        old = self.cache.get(key)
        p = route[:]
        if old is not None and old[0] <= self.route_cost(p) + 1e-8:
            p = list(old[1])
            if old[2]:
                return p
        d = self.d
        changed = True
        while changed and time.perf_counter() < stop:
            changed = False
            k = len(p)
            for i in range(k - 1):
                if time.perf_counter() >= stop:
                    break
                a, b = (p[i - 1] if i else 0), p[i]
                for j in range(i + 1, k):
                    c, e = p[j], (p[j + 1] if j + 1 < k else 0)
                    if d[a][c] + d[b][e] < d[a][b] + d[c][e] - 1e-8:
                        p[i:j + 1] = reversed(p[i:j + 1])
                        changed = True
                        break
                if changed:
                    break
            if changed:
                continue
            # Single-node relocation complements 2-opt.
            for i in range(k):
                if time.perf_counter() >= stop:
                    break
                v = p[i]
                a = p[i - 1] if i else 0
                b = p[i + 1] if i + 1 < k else 0
                saving = d[a][v] + d[v][b] - d[a][b]
                q = p[:i] + p[i + 1:]
                delta, candidate = self.insert(q, v)
                if delta < saving - 1e-8:
                    p = candidate
                    changed = True
                    break
        optimal = bool(old and old[2])
        if exact and len(p) <= self.exact_size and time.perf_counter() < stop:
            result = self.held_karp(p, stop)
            if result is not None:
                p = result
                optimal = True
        cost = self.route_cost(p)
        if old is None or cost < old[0] - 1e-8 or optimal:
            self.cache[key] = (cost, tuple(p), optimal)
        return p

    def held_karp(self, p, stop):
        k = len(p)
        size = 1 << k
        dp = np.full((size, k), np.inf)
        prev = np.full((size, k), -1, dtype=np.int16)
        for j in range(k):
            dp[1 << j, j] = self.d[0][p[j]]
        for mask in range(1, size):
            if mask & 31 == 0 and time.perf_counter() >= stop:
                return None
            bits = mask
            while bits:
                bit = bits & -bits
                j = bit.bit_length() - 1
                rem = mask ^ bit
                if rem:
                    bb = rem
                    best = math.inf
                    source = -1
                    while bb:
                        ib = bb & -bb
                        i = ib.bit_length() - 1
                        value = dp[rem, i] + self.d[p[i]][p[j]]
                        if value < best:
                            best, source = value, i
                        bb ^= ib
                    dp[mask, j] = best
                    prev[mask, j] = source
                bits ^= bit
        mask = size - 1
        j = min(range(k), key=lambda a: dp[mask, a] + self.d[p[a]][0])
        ans = []
        while j >= 0:
            ans.append(p[j])
            nj = int(prev[mask, j])
            mask ^= 1 << j
            j = nj
        return ans[::-1]

    def rebuild_routes(self):
        for t in range(self.h):
            nodes = [s + 1 for s in range(self.n) if self.z[s, t]]
            key = tuple(nodes)
            if key in self.cache:
                self.routes[t] = list(self.cache[key][1])
                continue
            nodes.sort(key=lambda s: self.d[0][s], reverse=True)
            p = []
            for node in nodes:
                p = self.insert(p, node)[1]
            stop = min(self.end, time.perf_counter() + max(.001, self.budget / max(20, 8 * self.h)))
            self.routes[t] = self.improve_route(p, stop)

    def schedule_dp(self, s, costs, cap, available, strict=True, penalty=1.0):
        h = self.h
        r, u, initial = self.r[s], self.u[s], self.i0[s]
        gap = u - initial
        first = h - 1 if r <= 0 else min(h - 1, int(math.floor((initial + 1e-8) / r)))
        length = h + 1 if r <= 0 else int(math.floor((u + 1e-8) / r))
        if length < 1:
            return None
        val = [math.inf] * h
        parent = [-2] * h

        def resource_cost(a, b, shipped):
            if b <= a:
                return 0.0
            short = shipped - available[a:b]
            if strict:
                return math.inf if np.any(short > 1e-6) else 0.0
            return penalty * float(np.maximum(short, 0.0) @ np.maximum(short, 0.0))

        for t in range(h):
            for p in [-1] + list(range(max(0, t - length), t)):
                if p < 0:
                    if t > first:
                        continue
                    q = gap + r * t
                    value = resource_cost(0, t, 0.0)
                else:
                    if not math.isfinite(val[p]):
                        continue
                    q = r * (t - p)
                    value = val[p] + resource_cost(p, t, gap + r * p)
                if q > cap[t] + 1e-6:
                    if strict:
                        continue
                    value += penalty * ((q - cap[t]) ** 2 - max(0.0, -cap[t]) ** 2)
                value += costs[t] + q * self.weights[s, t]
                if value < val[t]:
                    val[t], parent[t] = value, p
        best = math.inf
        last = -2
        if initial + 1e-8 >= h * r:
            best, last = resource_cost(0, h, 0.0), -1
        for t in range(h):
            if (h - t) * r > u + 1e-8:
                continue
            value = val[t] + resource_cost(t, h, gap + r * t)
            if value < best:
                best, last = value, t
        if last == -2 or not math.isfinite(best):
            return None
        visits = np.zeros(h, dtype=np.int8)
        while last >= 0:
            visits[last] = 1
            last = parent[last]
        return visits

    def single_move(self, s, noise=0.0):
        node = s + 1
        base = [[v for v in p if v != node] for p in self.routes]
        inserted = [self.insert(p, node) for p in base]
        costs = np.array([v[0] for v in inserted])
        if noise:
            costs = costs * np.array([self.rng.uniform(1 - noise, 1 + noise) for _ in range(self.h)])
        others = self.loads - self.x[s]
        z = self.schedule_dp(s, costs, self.cap - others,
                             self.supply - np.cumsum(others))
        if z is None:
            return False
        x = self.amount(s, z)
        routes = [inserted[t][1] if z[t] else base[t] for t in range(self.h)]
        delta = float((x - self.x[s]) @ self.weights[s])
        delta += sum(self.route_cost(routes[t]) - self.route_cost(self.routes[t]) for t in range(self.h))
        if delta < -1e-7 or (noise and delta < noise * max(1.0, sum(costs)) * .1):
            self.z[s], self.x[s] = z, x
            self.routes = routes
            self.loads = self.x.sum(axis=0)
            self.remember()
            return True
        return False

    def inventory_mip(self, group, duration, varied=False, feasibility=False):
        if milp is None or duration <= .001 or self.remaining() <= .001:
            return False
        h, k = self.h, len(group)
        if not k or not h:
            return False
        size = 3 * k * h
        c = np.zeros(size)
        ub = np.empty(size)
        integrality = np.zeros(size, dtype=np.int8)
        rows, cols, values, low, high = [], [], [], [], []
        def row(entries, lo=-np.inf, hi=np.inf):
            a = len(low)
            for b, v in entries:
                if v:
                    rows.append(a)
                    cols.append(b)
                    values.append(v)
            low.append(lo)
            high.append(hi)
        removed = {s + 1 for s in group}
        base = [[v for v in p if v not in removed] for p in self.routes]
        outside = self.loads - self.x[group].sum(axis=0)
        limits = self.supply - np.cumsum(outside)
        for a, s in enumerate(group):
            off = 3 * a * h
            u, r, initial = self.u[s], self.r[s], self.i0[s]
            for t in range(h):
                zz, xx, ii = off + t, off + h + t, off + 2 * h + t
                integrality[zz] = 1
                ub[zz], ub[xx], ub[ii] = 1, min(u, self.cap), u
                insert_cost = self.insert(base[t], s + 1)[0]
                if varied:
                    insert_cost *= self.rng.uniform(.35, 1.65)
                if not feasibility:
                    c[zz] = insert_cost
                    c[xx] = self.weights[s, t]
                balance = [(ii, 1), (xx, -1)]
                fill = [(xx, 1)]
                rhs = -r
                fill_upper = u
                fill_lower = 0
                if t:
                    prev = ii - 1
                    balance.append((prev, -1))
                    fill.append((prev, 1))
                else:
                    rhs += initial
                    fill_upper -= initial
                    fill_lower -= initial
                row(balance, rhs, rhs)
                row(fill, hi=fill_upper)
                row(fill + [(zz, -u)], lo=fill_lower)
                row([(xx, 1), (zz, -u)], hi=0)
            if r > 0:
                first = int(math.floor((initial + 1e-8) / r))
                if first < h:
                    row([(off + t, 1) for t in range(first + 1)], lo=1)
                length = int(math.floor((u + 1e-8) / r))
                if length > 0:
                    for t in range(1, h - length + 1):
                        row([(off + j, 1) for j in range(t, t + length)], lo=1)
        for t in range(h):
            row([(3 * a * h + h + t, 1) for a in range(k)], hi=self.cap - outside[t])
            row([(3 * a * h + h + j, 1) for a in range(k) for j in range(t + 1)], hi=limits[t])
        remaining = min(duration, self.remaining() * .93)
        if remaining <= .001:
            return False
        try:
            matrix = coo_matrix((values, (rows, cols)), shape=(len(low), size)).tocsc()
            result = milp(c, integrality=integrality, bounds=Bounds(np.zeros(size), ub),
                          constraints=LinearConstraint(matrix, np.asarray(low), np.asarray(high)),
                          options={'time_limit': remaining, 'mip_rel_gap': .001, 'presolve': True})
            if result.x is None:
                return False
            zz = self.z.copy()
            xx = self.x.copy()
            for a, s in enumerate(group):
                zz[s] = (result.x[3 * a * h:3 * a * h + h] > .5).astype(np.int8)
                xx[s] = self.amount(s, zz[s])
            if not self.feasible(zz, xx):
                return False
            old = (self.z, self.x, self.routes, self.loads, self.value)
            self.z, self.x = zz, xx
            self.routes = [p[:] for p in old[2]]
            self.rebuild_routes()
            self.loads = xx.sum(axis=0)
            value = float(np.sum(xx * self.weights)) + sum(self.route_cost(p) for p in self.routes)
            scale = max(1.0, sum(self.route_cost(p) for p in self.routes))
            temperature = scale * .015 * max(.05, self.remaining() / max(.001, self.budget))
            accept = (self.best is None or value < old[4] - 1e-7 or
                      (varied and self.rng.random() < math.exp(min(0.0, (old[4] - value) / temperature))))
            if accept:
                self.remember()
                return True
            self.z, self.x, self.routes, self.loads, self.value = old
        except Exception:
            return False
        return False

    def repair(self, stop):
        scale = max(1.0, self.cap, float(np.sum(self.r)))
        penalty = (1.0 + max((max(row) for row in self.d), default=1.0)
                   + float(np.max(np.abs(self.weights), initial=0.0)) * scale) * 100.0
        best_violation = math.inf
        saved = self.z.copy(), self.x.copy()
        while time.perf_counter() < stop:
            order = list(range(self.n))
            self.rng.shuffle(order)
            changed = False
            for s in order:
                if time.perf_counter() >= stop:
                    break
                others = self.x.sum(axis=0) - self.x[s]
                costs = np.array([self.rng.random() * penalty * scale * .001 for _ in range(self.h)])
                z = self.schedule_dp(s, costs, self.cap - others,
                                     self.supply - np.cumsum(others), False, penalty)
                if z is not None and np.any(z != self.z[s]):
                    self.z[s], self.x[s] = z, self.amount(s, z)
                    changed = True
                if self.feasible(self.z, self.x):
                    self.loads = self.x.sum(axis=0)
                    self.rebuild_routes()
                    self.remember()
                    return True
            loads = self.x.sum(axis=0)
            violation = float(np.sum(np.maximum(loads - self.cap, 0) ** 2)
                              + np.sum(np.maximum(np.cumsum(loads) - self.supply, 0) ** 2))
            if violation < best_violation:
                best_violation = violation
                saved = self.z.copy(), self.x.copy()
            if not changed:
                self.z, self.x = saved[0].copy(), saved[1].copy()
                if self.n:
                    s = self.rng.randrange(self.n)
                    # Random timing perturbation that keeps inventory valid throughout.
                    z = self.schedule_dp(s, np.array([self.rng.uniform(-penalty, penalty) for _ in range(self.h)]),
                                         np.full(self.h, self.cap), np.full(self.h, math.inf))
                    if z is not None:
                        self.z[s], self.x[s] = z, self.amount(s, z)
        self.loads = self.x.sum(axis=0)
        return False

    def make_patterns(self):
        cap = max(16, int(2600 * math.sqrt(max(.001, self.budget) / max(1, self.n * self.h))))
        if self.h > int(math.log2(cap)):
            return
        stop = min(self.end, time.perf_counter() + self.budget * .035)
        h = self.h
        masks = np.arange(1 << h, dtype=np.int64)
        z = ((masks[:, None] >> np.arange(h)[None, :]) & 1).astype(np.int8)
        for s in range(self.n):
            if time.perf_counter() >= stop:
                break
            inv = np.full(len(z), self.i0[s])
            x = np.zeros(z.shape)
            valid = np.ones(len(z), dtype=bool)
            for t in range(h):
                x[:, t] = np.where(z[:, t] != 0, self.u[s] - inv, 0.0)
                valid &= (x[:, t] >= -1e-8) & (x[:, t] <= self.cap + 1e-8)
                inv += x[:, t] - self.r[s]
                valid &= (inv >= -1e-8) & (inv <= self.u[s] + 1e-8)
            xx, zz = x[valid], z[valid]
            self.patterns[s] = (zz, xx, np.cumsum(xx, axis=1), xx @ self.weights[s])

    def two_insertions(self, base, a, b):
        d = self.d
        m = len(base) + 1
        ca, cb = [], []
        together = math.inf
        same_pos, same_order = 0, [a, b]
        for e in range(m):
            u = base[e - 1] if e else 0
            v = base[e] if e < len(base) else 0
            ca.append(d[u][a] + d[a][v] - d[u][v])
            cb.append(d[u][b] + d[b][v] - d[u][v])
            va = d[u][a] + d[a][b] + d[b][v] - d[u][v]
            vb = d[u][b] + d[b][a] + d[a][v] - d[u][v]
            if min(va, vb) < together:
                together = min(va, vb)
                same_pos = e
                same_order = [a, b] if va <= vb else [b, a]
        ia = min(range(m), key=ca.__getitem__)
        ib = min(range(m), key=cb.__getitem__)
        r1, r2 = base[:ia] + [a] + base[ia:], base[:ib] + [b] + base[ib:]
        r3 = base[:same_pos] + same_order + base[same_pos:]
        if m > 1:
            other_b = min((j for j in range(m) if j != ib), key=cb.__getitem__)
            ea = min(range(m), key=lambda j: ca[j] + cb[ib if j != ib else other_b])
            eb = ib if ea != ib else other_b
            separate = ca[ea] + cb[eb]
            if separate < together:
                together = separate
                r3 = []
                for e in range(m):
                    if e == ea:
                        r3.append(a)
                    if e == eb:
                        r3.append(b)
                    if e < len(base):
                        r3.append(base[e])
        return (0.0, ca[ia], cb[ib], together), (base, r1, r2, r3)

    def pair_move(self, a, b, shake=False):
        pa, pb = self.patterns[a], self.patterns[b]
        if pa is None or pb is None or not len(pa[0]) or not len(pb[0]):
            return False
        za, xa, qa, ha = pa
        zb, xb, qb, hb = pb
        others = self.loads - self.x[a] - self.x[b]
        cap = self.cap - others
        available = self.supply - np.cumsum(others)
        costs = ha[:, None] + hb[None, :]
        valid = np.ones(costs.shape, dtype=bool)
        choices = []
        base_cost = 0.0
        for t in range(self.h):
            base = [v for v in self.routes[t] if v != a + 1 and v != b + 1]
            delta, routes = self.two_insertions(base, a + 1, b + 1)
            choices.append(routes)
            base_cost += self.route_cost(base)
            ia, ib = za[:, t, None], zb[None, :, t]
            costs += delta[1] * ia + delta[2] * ib + (delta[3] - delta[1] - delta[2]) * ia * ib
            valid &= xa[:, t, None] + xb[None, :, t] <= cap[t] + 1e-7
            valid &= qa[:, t, None] + qb[None, :, t] <= available[t] + 1e-7
        costs[~valid] = np.inf
        oldcost = float(self.x[a] @ self.weights[a] + self.x[b] @ self.weights[b])
        oldcost += sum(self.route_cost(p) for p in self.routes) - base_cost
        ranking = costs
        if shake:
            noise = max(1.0, sum(self.route_cost(p) for p in self.routes)) * .015
            # Row/column perturbation, without a candidate-count cap unrelated to problem size.
            ranking = costs + np.array([self.rng.uniform(-noise, noise) for _ in range(len(za))])[:, None]
            ranking = ranking + np.array([self.rng.uniform(-noise, noise) for _ in range(len(zb))])[None, :]
        ia, ib = np.unravel_index(np.argmin(ranking), ranking.shape)
        newcost = float(costs[ia, ib])
        if not math.isfinite(newcost) or (not shake and newcost >= oldcost - 1e-7):
            return False
        self.z[a], self.x[a] = za[ia], xa[ia]
        self.z[b], self.x[b] = zb[ib], xb[ib]
        self.routes = [list(choices[t][int(za[ia, t]) + 2 * int(zb[ib, t])]) for t in range(self.h)]
        self.remember()
        return newcost < oldcost - 1e-7

    def polish(self, exact=False, perturb=False):
        changed = False
        order = list(range(self.h))
        self.rng.shuffle(order)
        for t in order:
            if self.remaining() <= .001:
                break
            old = self.routes[t]
            p = old[:]
            if perturb and len(p) >= 4:
                a, b = sorted(self.rng.sample(range(len(p)), 2))
                p[a:b + 1] = reversed(p[a:b + 1])
                # Bypass the cached starting point, but still accept only true-cost improvements.
                key = tuple(sorted(p))
                saved = self.cache.pop(key, None)
            else:
                saved = None
            stop = min(self.end, time.perf_counter() + max(.002, self.budget / max(50, self.h * 20)))
            p = self.improve_route(p, stop, exact)
            if saved is not None:
                key = tuple(sorted(p))
                if key not in self.cache or self.cache[key][0] > saved[0]:
                    self.cache[key] = saved
            if self.route_cost(p) < self.route_cost(old) - 1e-7:
                self.routes[t] = p
                changed = True
        if changed:
            self.remember()
        return changed

    def run(self):
        if self.n == 0 or self.h == 0:
            self.remember()
            return
        # An independent latest-replenishment schedule provides a cheap starting point.
        for s in range(self.n):
            inv = self.i0[s]
            for t in range(self.h):
                if inv < self.r[s] - 1e-8:
                    self.z[s, t] = 1
                    inv = self.u[s]
                inv -= self.r[s]
            self.x[s] = self.amount(s, self.z[s])
        self.loads = self.x.sum(axis=0)
        if self.feasible(self.z, self.x):
            self.rebuild_routes()
            self.remember()
        else:
            self.repair(min(self.end, time.perf_counter() + self.budget * .035))
        if self.best is None:
            # Here all retailers are released and external delivery amounts are set strictly to zero.
            self.inventory_mip(list(range(self.n)), max(.001, self.remaining() * .55), feasibility=True)
        if self.best is None and self.remaining() > .001:
            self.repair(min(self.end, time.perf_counter() + self.remaining() * .3))
            if self.best is None:
                self.inventory_mip(list(range(self.n)), self.remaining() * .90, feasibility=True)
        if self.best is None:
            self.repair(self.end)
            return
        self.make_patterns()
        self.polish(exact=True)
        order = list(range(self.n))
        self.rng.shuffle(order)
        cursor = 0
        stagnant = 0
        last_mip = time.perf_counter()
        strength = max(2, int(math.sqrt(max(1.0, self.budget) * max(1, self.n) / max(1, self.h))))
        while self.remaining() > .001:
            before = self.best_cost
            if cursor < len(order):
                self.single_move(order[cursor])
                cursor += 1
            else:
                if self.n > 1:
                    a = self.rng.randrange(self.n)
                    width = min(self.n - 1, max(2, strength))
                    if self.rng.random() < .65:
                        b = self.rng.choice(self.near[a][:width])
                    else:
                        b = self.rng.randrange(self.n - 1)
                        b += b >= a
                    if self.patterns[a] is not None and self.patterns[b] is not None:
                        self.pair_move(a, b)
                    else:
                        self.single_move(a)
                        self.single_move(b)
                else:
                    self.single_move(0)
                stagnant += 1
                if stagnant >= max(2, self.n // 2):
                    self.polish(exact=True, perturb=True)
                    cursor = 0
                    self.rng.shuffle(order)
            if self.best_cost < before - 1e-7:
                stagnant = 0
            now = time.perf_counter()
            spacing = max(.025, self.budget / (8 + math.sqrt(max(1, self.n))))
            if stagnant >= max(2, self.n) and now - last_mip >= spacing:
                self.restore()
                if self.n > 1:
                    a = self.rng.randrange(self.n)
                    width = min(self.n, 2 + self.rng.randrange(max(1, min(self.n - 1, strength))))
                    pool = self.near[a][:min(self.n - 1, width * 3)]
                    group = [a] + self.rng.sample(pool, min(width - 1, len(pool)))
                else:
                    group = [0]
                duration = min(self.remaining() * .15, self.budget * .035 * math.sqrt(len(group)))
                changed = self.inventory_mip(group, duration, varied=True)
                if not changed:
                    if len(group) > 1 and self.patterns[group[0]] is not None and self.patterns[group[1]] is not None:
                        self.pair_move(group[0], group[1], shake=True)
                    else:
                        self.single_move(group[0], noise=.9)
                last_mip = time.perf_counter()
                stagnant = 0
                cursor = 0
                self.rng.shuffle(order)
        self.restore()

    def export(self):
        if self.best is not None:
            self.restore()
        result = {'routes': {}, 'retailer_visit_status': {}, 'deliveries': {},
                  'retailer_inventory': {}, 'supplier_inventory': {}}
        inv = self.i0.copy()
        stock = self.b0
        for t in range(self.h + 1):
            key = str(t + 1)
            result['retailer_inventory'][key] = {str(s + 1): float(inv[s]) for s in range(self.n)}
            result['supplier_inventory'][key] = float(stock)
            if t == self.h:
                break
            z = [int(self.z[s, t]) for s in range(self.n)]
            q = np.where(self.z[:, t] != 0, self.u - inv, 0.0)
            result['retailer_visit_status'][key] = z
            result['deliveries'][key] = {str(s + 1): float(q[s]) for s in range(self.n)}
            p = self.routes[t]
            required = {s + 1 for s in range(self.n) if z[s]}
            if set(p) != required or len(p) != len(required):
                p = sorted(required)
            if len(p) == 1:
                arcs = [[p[0], 0, 2]]
            elif p:
                path = [0] + p + [0]
                arcs = [[max(a, b), min(a, b), 1] for a, b in zip(path, path[1:])]
            else:
                arcs = []
            result['routes'][key] = arcs
            inv += q - self.r
            stock += self.prod - float(q.sum())
        return result


def solve(instance: dict, time_limit_s: float) -> dict:
    start = time.perf_counter()
    search = _Search(instance, time_limit_s, start)
    try:
        search.run()
    except Exception:
        # The saved feasible solution is unaffected by unfinished neighbourhoods or numerical solver failures.
        if search.best is None and search.remaining() > .001:
            try:
                search.inventory_mip(list(range(search.n)), search.remaining() * .9, feasibility=True)
            except Exception:
                pass
    return search.export()
