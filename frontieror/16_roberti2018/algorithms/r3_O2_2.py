"""Sequence crossover splitting with swap* neighbourhood search
Generates routes by capacity-based construction, customer-sequence crossover and dynamic-programming splitting, combined with swap* local search and first-echelon delivery optimisation, keeping the feasible solution with the lowest true total cost.
"""
import math
import random
import time
from array import array
from collections import Counter, deque, OrderedDict

import numpy as np
try:
    from scipy.optimize import milp, Bounds, LinearConstraint
    from scipy.sparse import coo_matrix
except ImportError:
    milp = None


class _Distances:
    def __init__(self, xy, budget):
        self.xy = np.asarray(xy, dtype=float)
        self.rows = OrderedDict()
        self.keep = max(2, budget // max(1, len(xy)))
        self.full = None
        if len(xy) ** 2 <= budget:
            self.full = np.hypot(self.xy[:, 0, None] - self.xy[:, 0],
                                 self.xy[:, 1, None] - self.xy[:, 1]).tolist()

    def __getitem__(self, i):
        if self.full is not None:
            return self.full[i]
        row = self.rows.get(i)
        if row is None:
            row = np.hypot(self.xy[:, 0] - self.xy[i, 0],
                           self.xy[:, 1] - self.xy[i, 1]).tolist()
            self.rows[i] = row
            if len(self.rows) > self.keep:
                self.rows.popitem(last=False)
        return row


class _Linear:
    def __init__(self):
        self.c, self.ub, self.integer = [], [], []
        self.ri, self.ci, self.v, self.lo, self.hi = [], [], [], [], []

    def var(self, cost=0., upper=1., integer=True):
        j = len(self.c)
        self.c.append(float(cost))
        self.ub.append(float(upper))
        self.integer.append(int(integer))
        return j

    def row(self, terms, lower=-math.inf, upper=math.inf):
        r = len(self.lo)
        for j, a in terms:
            if a:
                self.ri.append(r); self.ci.append(j); self.v.append(float(a))
        self.lo.append(lower); self.hi.append(upper)

    def run(self, until):
        if milp is None or time.perf_counter() >= until:
            return None
        try:
            a = coo_matrix((self.v, (self.ri, self.ci)),
                           shape=(len(self.lo), len(self.c))).tocsc()
            left = until - time.perf_counter()
            if left <= 0:
                return None
            result = milp(np.asarray(self.c), integrality=np.asarray(self.integer),
                          bounds=Bounds(np.zeros(len(self.c)), self.ub),
                          constraints=LinearConstraint(a, self.lo, self.hi),
                          options={'time_limit': left, 'mip_rel_gap': 0., 'disp': False})
            if result.x is not None and np.all(np.isfinite(result.x)):
                return result.x
        except Exception:
            pass
        return None


class _Engine:
    def __init__(self, data, seconds, start):
        self.data = data
        self.start = start
        self.seconds = max(0., float(seconds))
        self.strength = math.log1p(self.seconds)
        self.end = start + self.seconds * .975
        self.rng = random.Random(739391)
        self.cid = list(map(int, data['customers_id']))
        self.sid = list(map(int, data['satellites_id']))
        self.gid = list(map(int, data['groups_group_id']))
        self.n, self.m = len(self.cid), len(self.sid)
        self.H = len(self.gid)
        self.depot_id = int(data['depot_id'])
        self.q = list(map(int, data['customers_demand']))
        self.Q = int(data['Q2'])
        self.Q1 = float(data['Q1'])
        self.T1, self.T2 = int(data['T1']), int(data['T2'])
        self.cap = list(map(float, data['satellites_capacity']))
        self.maxv = list(map(int, data['satellites_max_vehicles']))
        self.hand = list(map(float, data['satellites_handling_cost']))
        ix = {v: i for i, v in enumerate(self.cid)}
        self.groups = [[ix[int(v)] for v in row] for row in data['groups_customer_ids']]
        self.group_of = [0] * self.n
        for h, nodes in enumerate(self.groups):
            for i in nodes:
                self.group_of[i] = h
        self.gd = [sum(self.q[i] for i in row) for row in self.groups]
        self.xy = list(zip(data['customers_x'], data['customers_y']))
        self.xy += list(zip(data['satellites_x'], data['satellites_y']))
        self.xy.append((data['depot_x'], data['depot_y']))
        self.depot = self.n + self.m
        self.d = _Distances(self.xy, int(450000 * (1 + self.strength)))
        self.width = min(max(0, self.n - 1), max(1, int(math.sqrt(self.n) *
                         (1 + .5 * self.strength) + self.strength)))
        self.neighbors = {}
        self.first_cache = {}
        self.tour_cache = {0: (0., [])}
        self.population = []
        self.best = None
        self.last_routes = None
        self.poolsize = max(2, int(2 + math.sqrt(max(1, self.n)) ** .5 * self.strength))
        avg = sum(self.q) / max(1, self.n)
        self.proxy = [[self.gd[h] * (self.hand[s] +
                       2 * self.d[self.depot][self.n + s] / max(1., self.Q1)) +
                       sum(2 * self.d[self.n + s][i] * (self.q[i] + .12 * avg) /
                           max(1, self.Q) for i in nodes)
                       for s in range(self.m)] for h, nodes in enumerate(self.groups)]
        self.exact_first = self.m * self.m * (1 << min(self.m, 25)) <= \
                           16000 * (1 + self.strength)
        self.first_due = start

    def route_cost(self, s, seq):
        base = self.n + s
        old, value = base, 0.
        for i in seq:
            value += self.d[old][i]
            old = i
        return value + self.d[old][base]

    def nearest(self, i):
        if i not in self.neighbors:
            if not self.width:
                self.neighbors[i] = []
            else:
                values = np.asarray(self.d[i][:self.n])
                k = min(self.n, self.width + 1)
                part = np.argpartition(values, k - 1)[:k]
                self.neighbors[i] = [int(j) for j in part[np.argsort(values[part])]
                                     if int(j) != i][:self.width]
        return self.neighbors[i]

    def pack(self, nodes, s):
        bins, loads = [], []
        ds = self.d[self.n + s]
        # Load comes first; ties in residual capacity are broken by geometric distance.
        for i in sorted(nodes, key=lambda v: (-self.q[v], -ds[v])):
            best = None
            for b, load in enumerate(loads):
                if load + self.q[i] <= self.Q:
                    key = (self.Q - load - self.q[i], self.d[bins[b][-1]][i])
                    if best is None or key < best[0]:
                        best = (key, b)
            if best is None:
                bins.append([i]); loads.append(self.q[i])
            else:
                b = best[1]
                bins[b].append(i); loads[b] += self.q[i]
        return bins

    def order_route(self, s, nodes, until):
        if len(nodes) < 3:
            return nodes[:]
        base = self.n + s
        left, out, prev = set(nodes), [], base
        while left:
            row = self.d[prev]
            v = min(left, key=row.__getitem__)
            out.append(v); left.remove(v); prev = v
        changed = True
        while changed and time.perf_counter() < until:
            changed = False
            for p in range(len(out) - 1):
                if time.perf_counter() >= until:
                    return out
                a, b = base if p == 0 else out[p - 1], out[p]
                for q in range(p + 1, len(out)):
                    c, e = out[q], base if q + 1 == len(out) else out[q + 1]
                    if self.d[a][c] + self.d[b][e] < self.d[a][b] + self.d[c][e] - 1e-9:
                        out[p:q + 1] = reversed(out[p:q + 1])
                        changed = True
                        break
                if changed:
                    break
        return out

    def construct(self, until, noisy=False):
        assigned = [-1] * self.H
        nodes = [[] for _ in range(self.m)]
        bins = [[] for _ in range(self.m)]
        loads, count = [0] * self.m, [0] * self.m
        order = list(range(self.H))
        def priority(h):
            eligible = sum(self.gd[h] <= self.cap[s] and
                           (not self.groups[h] or self.maxv[s] > 0) for s in range(self.m))
            scores = sorted(self.proxy[h])
            regret = scores[1] - scores[0] if len(scores) > 1 else 0.
            noise = self.rng.uniform(.65, 1.5) if noisy else 1.
            return (-eligible, noise * (self.gd[h] + regret))
        order.sort(key=priority, reverse=True)
        for h in order:
            choice = None
            for s in range(self.m):
                if loads[s] + self.gd[h] > self.cap[s]:
                    continue
                packed = self.pack(nodes[s] + self.groups[h], s)
                k = len(packed)
                if k > self.maxv[s] or sum(count) - count[s] + k > self.T2:
                    continue
                value = self.proxy[h][s]
                if noisy:
                    value *= self.rng.uniform(.75, 1.25)
                if choice is None or value < choice[0]:
                    choice = (value, s, packed)
            if choice is None:
                return None
            _, s, bins[s] = choice
            assigned[h] = s
            nodes[s].extend(self.groups[h])
            loads[s] += self.gd[h]; count[s] = len(bins[s])
        routes = []
        for s in range(self.m):
            for row in bins[s]:
                seq = self.order_route(s, row, until)
                routes.append([s, seq, sum(self.q[i] for i in row)])
        return routes, assigned

    def capacity_model(self, until):
        # Aggregate load counts by demand type; a customer group's satellite choice remains indivisible.
        if milp is None or time.perf_counter() >= until:
            return None
        model = _Linear()
        choice = [[model.var(self.proxy[h][s],
                   int(self.gd[h] <= self.cap[s] and
                       (not self.groups[h] or self.maxv[s] > 0)))
                   for s in range(self.m)] for h in range(self.H)]
        sizes = sorted(set(self.q), reverse=True)
        gc = [Counter(self.q[i] for i in nodes) for nodes in self.groups]
        bins, used = [], []
        for s in range(self.m):
            slots = min(self.maxv[s], self.T2, self.n)
            counts, on = [], []
            for b in range(slots):
                on.append(model.var(1e-7))
                counts.append([model.var(0., min(self.n, self.Q // q) if q else self.n)
                               for q in sizes])
            bins.append(counts); used.append(on)
        for h in range(self.H):
            model.row([(j, 1) for j in choice[h]], 1, 1)
        for s in range(self.m):
            model.row([(choice[h][s], self.gd[h]) for h in range(self.H)],
                      upper=self.cap[s])
            for t, q in enumerate(sizes):
                model.row([(row[t], 1) for row in bins[s]] +
                          [(choice[h][s], -gc[h][q]) for h in range(self.H)], 0, 0)
            for b, row in enumerate(bins[s]):
                terms = [(j, q) for j, q in zip(row, sizes)]
                model.row(terms + [(used[s][b], -self.Q)], upper=0)
                model.row([(j, 1) for j in row] + [(used[s][b], -self.n)], upper=0)
                if b:
                    model.row([(used[s][b], 1), (used[s][b - 1], -1)], upper=0)
                    model.row(terms + [(j, -q) for j, q in zip(bins[s][b - 1], sizes)], upper=0)
        model.row([(j, 1) for row in used for j in row], upper=self.T2)
        values = model.run(until)
        if values is None:
            return None
        a = [max(range(self.m), key=lambda s: values[choice[h][s]]) for h in range(self.H)]
        if any(values[choice[h][a[h]]] < .99 for h in range(self.H)):
            return None
        routes = []
        for s in range(self.m):
            stock = {q: [] for q in sizes}
            for h in range(self.H):
                if a[h] == s:
                    for i in self.groups[h]:
                        stock[self.q[i]].append(i)
            for b, row in enumerate(bins[s]):
                seq = []
                for t, q in enumerate(sizes):
                    count = int(round(values[row[t]]))
                    if count < 0 or count > len(stock[q]):
                        return None
                    seq.extend(stock[q][-count:] if count else [])
                    if count:
                        del stock[q][-count:]
                if seq:
                    seq = self.order_route(s, seq, until)
                    routes.append([s, seq, sum(self.q[i] for i in seq)])
            if any(stock.values()):
                return None
        return (routes, a) if self.valid(routes, a) else None

    def sat_tour(self, mask, until):
        cached = self.tour_cache.get(mask)
        if cached is not None:
            return cached
        nodes = [s for s in range(self.m) if mask & (1 << s)]
        if not nodes:
            return (0., [])
        if self.exact_first and time.perf_counter() < until:
            k = len(nodes)
            dp = {(1 << j, j): (self.d[self.depot][self.n + nodes[j]], -1)
                  for j in range(k)}
            completed = True
            for sub in range(1, 1 << k):
                if time.perf_counter() >= until:
                    completed = False
                    break
                for j in range(k):
                    if not (sub & (1 << j)) or sub == 1 << j:
                        continue
                    prev = sub ^ (1 << j)
                    cost, p = min((dp[prev, i][0] + self.d[self.n + nodes[i]][self.n + nodes[j]], i)
                                  for i in range(k) if prev & (1 << i))
                    dp[sub, j] = (cost, p)
            if completed:
                sub = (1 << k) - 1
                value, j = min((dp[sub, i][0] + self.d[self.n + nodes[i]][self.depot], i)
                               for i in range(k))
                route = []
                while j >= 0:
                    route.append(nodes[j])
                    p = dp[sub, j][1]; sub ^= 1 << j; j = p
                route.reverse()
                result = (value, route)
                self.tour_cache[mask] = result
                return result
        remaining, route, prev = set(nodes), [], self.depot
        while remaining:
            s = min(remaining, key=lambda v: self.d[prev][self.n + v])
            route.append(s); remaining.remove(s); prev = self.n + s
        value = 0.; prev = self.depot
        for s in route:
            value += self.d[prev][self.n + s]; prev = self.n + s
        result = (value + self.d[prev][self.depot], route)
        self.tour_cache[mask] = result
        return result

    def cargo_plan(self, cargo, until):
        routes, cost = [], 0.
        for delivery in cargo:
            mask = sum(1 << s for s, q in delivery.items() if q > 1e-8)
            value, path = self.sat_tour(mask, until)
            routes.append(path); cost += value
        return (cost, routes, cargo)

    def first(self, demands, until):
        key = tuple(demands)
        if key in self.first_cache:
            return self.first_cache[key]
        if sum(demands) > self.Q1 * self.T1 + 1e-7:
            return None
        active = [s for s in range(self.m) if demands[s] > 1e-8]
        dx, dy = self.xy[self.depot]
        polar = sorted(active, key=lambda s: math.atan2(self.xy[self.n + s][1] - dy,
                                                       self.xy[self.n + s][0] - dx))
        candidates = [polar]
        candidates.append(sorted(active, key=lambda s: -demands[s]))
        if polar:
            # Use the largest angular gap as the boundary for sequential splitting, avoiding forcibly chaining distant satellites together.
            cut = max(range(len(polar)), key=lambda j: self.d[self.n + polar[j - 1]][self.n + polar[j]])
            candidates.append(polar[cut:] + polar[:cut])
        record = None
        for order in candidates:
            cargo, load = [], 0.
            for s in order:
                rest = float(demands[s])
                while rest > 1e-8:
                    if not cargo or load >= self.Q1 - 1e-8:
                        cargo.append({}); load = 0.
                    amount = min(rest, self.Q1 - load)
                    cargo[-1][s] = cargo[-1].get(s, 0.) + amount
                    load += amount; rest -= amount
            if len(cargo) <= self.T1:
                plan = self.cargo_plan(cargo, until)
                if record is None or plan[0] < record[0]:
                    record = plan
        # Vehicles need not depart fully loaded; extra idle vehicles can avoid expensive inter-satellite connections.
        cargo = []
        for s in active:
            rest = float(demands[s])
            while rest > 1e-8:
                amount = min(rest, self.Q1)
                cargo.append({s: amount}); rest -= amount
        while len(cargo) > self.T1 and time.perf_counter() < until:
            move = None
            for i in range(len(cargo)):
                for j in range(i):
                    if sum(cargo[i].values()) + sum(cargo[j].values()) > self.Q1 + 1e-8:
                        continue
                    joined = cargo[j].copy()
                    for s, q in cargo[i].items():
                        joined[s] = joined.get(s, 0.) + q
                    delta = (self.cargo_plan([joined], until)[0] -
                             self.cargo_plan([cargo[i], cargo[j]], until)[0])
                    if move is None or delta < move[0]:
                        move = (delta, i, j, joined)
            if move is None:
                break
            _, i, j, joined = move
            cargo[j] = joined; cargo.pop(i)
        if len(cargo) <= self.T1:
            plan = self.cargo_plan(cargo, until)
            if record is None or plan[0] < record[0]:
                record = plan
        self.first_cache[key] = record
        return record

    def optimize_first(self, state, until):
        if not self.exact_first or milp is None or time.perf_counter() >= until:
            return
        demand = state[2]
        active = [s for s in range(self.m) if demand[s] > 1e-8]
        if len(active) < 2:
            return
        model = _Linear()
        masks, shipment, num = [], [], []
        for sub in range(1, 1 << len(active)):
            if time.perf_counter() >= until:
                return
            mask = sum(1 << active[j] for j in range(len(active)) if sub & (1 << j))
            value, _ = self.sat_tour(mask, until)
            k = model.var(value, self.T1)
            w = {s: model.var(0., demand[s], False) for s in active if mask & (1 << s)}
            model.row([(j, 1) for j in w.values()] + [(k, -self.Q1)], upper=0)
            masks.append(mask); shipment.append(w); num.append(k)
        for s in active:
            model.row([(w[s], 1) for w in shipment if s in w], demand[s], demand[s])
        model.row([(k, 1) for k in num], upper=self.T1)
        values = model.run(until)
        if values is None:
            return
        cargo = []
        for w, k in zip(shipment, num):
            count = int(round(values[k]))
            if count < 0:
                return
            deliveries = [{} for _ in range(count)]
            b, filled = 0, 0.
            for s, j in w.items():
                rest = max(0., float(values[j]))
                while rest > 1e-7:
                    if b >= count:
                        return
                    amount = min(rest, self.Q1 - filled)
                    deliveries[b][s] = deliveries[b].get(s, 0.) + amount
                    filled += amount; rest -= amount
                    if filled >= self.Q1 - 1e-7:
                        b += 1; filled = 0.
            cargo.extend(row for row in deliveries if row)
        if len(cargo) > self.T1:
            return
        if any(abs(sum(row.get(s, 0.) for row in cargo) - demand[s]) > 1e-5 for s in active):
            return
        plan = self.cargo_plan(cargo, until)
        old = self.first_cache[tuple(demand)]
        if plan[0] < old[0] - 1e-8:
            self.first_cache[tuple(demand)] = plan

    def valid(self, routes, assignment):
        if len(assignment) != self.H or len(routes) > self.T2:
            return False
        if any(s < 0 or s >= self.m for s in assignment):
            return False
        seen = [0] * self.n
        count, loads = [0] * self.m, [0] * self.m
        for s, seq, load in routes:
            if not seq or s < 0 or s >= self.m:
                return False
            total = 0
            for i in seq:
                if i < 0 or i >= self.n or assignment[self.group_of[i]] != s:
                    return False
                seen[i] += 1; total += self.q[i]
            if total > self.Q or abs(total - load) > 1e-6:
                return False
            count[s] += 1; loads[s] += total
        return (all(v == 1 for v in seen) and sum(loads) <= self.T1 * self.Q1 + 1e-8
                and all(count[s] <= self.maxv[s] and loads[s] <= self.cap[s] + 1e-8
                        for s in range(self.m)))

    def state(self, routes, assignment, until):
        if not self.valid(routes, assignment):
            return None
        demand = [0] * self.m
        base = 0.
        for s, seq, load in routes:
            demand[s] += load
            base += self.route_cost(s, seq) + load * self.hand[s]
        if self.first(demand, until) is None:
            return None
        return [routes, assignment[:], demand, base]

    def score(self, state):
        return state[3] + self.first_cache[tuple(state[2])][0]

    def remember(self, state):
        if state is None:
            return
        if self.best is None or self.score(state) < self.score(self.best) - 1e-8:
            self.best = state
        # Distinguish individuals by route edges and group assignments, rather than keeping only copies with approximately the same cost.
        edge = frozenset((min(i, j), max(i, j)) for s, seq, _ in state[0]
                         for i, j in zip([self.n + s] + seq, seq + [self.n + s]))
        if any(edge == item[1] and state[1] == item[0][1] for item in self.population):
            return
        self.population.append((state, edge))
        self.population.sort(key=lambda item: self.score(item[0]))
        if len(self.population) > self.poolsize:
            costs = [self.score(item[0]) for item in self.population]
            spread = max(1e-8, costs[-1] - costs[0])
            worst, penalty = None, -math.inf
            for j in range(1, len(self.population)):
                if self.population[j][0] is self.best:
                    continue
                similarity = max(len(self.population[j][1] & self.population[k][1]) /
                                 max(1, len(self.population[j][1] | self.population[k][1]))
                                 for k in range(len(self.population)) if k != j)
                value = (costs[j] - costs[0]) / spread + similarity
                if value > penalty:
                    penalty, worst = value, j
            if worst is not None:
                self.population.pop(worst)

    def split_one(self, s, seq, layers, until):
        # Sliding-window minimum: capacity limits give a lower bound on the left endpoint, and each layer is an exact vehicle count.
        n = len(seq)
        if not n:
            return {0: (0., [])}
        ds = self.d[self.n + s]
        pref, edge = [0], [0.] * n
        for j, i in enumerate(seq):
            pref.append(pref[-1] + self.q[i])
            if j:
                edge[j] = edge[j - 1] + self.d[seq[j - 1]][i]
        left = [0] * n
        p = 0
        for j in range(n):
            while p <= j and pref[j + 1] - pref[p] > self.Q:
                p += 1
            left[j] = p
        add = [ds[i] - edge[j] for j, i in enumerate(seq)]
        tail = [ds[i] + edge[j] for j, i in enumerate(seq)]
        if layers is None:
            cost = [math.inf] * (n + 1); cost[0] = 0.
            count = [0] * (n + 1)
            pred = [-1] * (n + 1)
            queue = deque()
            for j in range(n):
                value = cost[j] + add[j]
                rank = (value, count[j])
                while queue and queue[-1][0] >= rank:
                    queue.pop()
                queue.append((rank, j))
                while queue and queue[0][1] < left[j]:
                    queue.popleft()
                if queue:
                    rank, k = queue[0]
                    cost[j + 1] = rank[0] + tail[j]
                    count[j + 1] = count[k] + 1; pred[j + 1] = k
            if not math.isfinite(cost[n]):
                return {}
            out, j = [], n
            while j:
                k = pred[j]
                out.append(seq[k:j]); j = k
            out.reverse()
            return {len(out): (cost[n], out)}
        previous = [math.inf] * (n + 1); previous[0] = 0.
        predecessors, options = [], {}
        for b in range(1, min(n, layers) + 1):
            if time.perf_counter() >= until:
                return options
            cost = [math.inf] * (n + 1)
            pred = array('i', [-1]) * (n + 1)
            queue = deque()
            for j in range(n):
                value = previous[j] + add[j]
                while queue and queue[-1][0] >= value:
                    queue.pop()
                if math.isfinite(value):
                    queue.append((value, j))
                while queue and queue[0][1] < left[j]:
                    queue.popleft()
                if queue:
                    cost[j + 1] = queue[0][0] + tail[j]
                    pred[j + 1] = queue[0][1]
            predecessors.append(pred)
            if math.isfinite(cost[n]):
                out, j, r = [], n, b - 1
                while j:
                    k = predecessors[r][j]
                    out.append(seq[k:j]); j = k; r -= 1
                out.reverse(); options[b] = (cost[n], out)
            previous = cost
        return options

    def decode(self, assignment, order, until):
        nodes = [[] for _ in range(self.m)]
        demand = [0] * self.m
        for i in order:
            s = assignment[self.group_of[i]]
            nodes[s].append(i); demand[s] += self.q[i]
        if any(demand[s] > self.cap[s] for s in range(self.m)):
            return None
        choices = []
        for s in range(self.m):
            if time.perf_counter() >= until:
                return None
            options = self.split_one(s, nodes[s], None, until)
            if not options:
                return None
            choices.append(options)
        ks = [next(iter(v)) for v in choices]
        if sum(ks) <= self.T2 and all(ks[s] <= self.maxv[s] for s in range(self.m)):
            return [[s, row, sum(self.q[i] for i in row)] for s in range(self.m)
                    for row in choices[s][ks[s]][1]]
        # First choose vehicle counts globally along the given sequence; the bin-packing fallback prevents the sequence itself from making splitting infeasible.
        for s in range(self.m):
            if time.perf_counter() >= until:
                return None
            upper = min(self.maxv[s], self.T2, len(nodes[s]))
            old = choices[s]
            options = self.split_one(s, nodes[s], upper, until)
            for k, option in old.items():
                if k <= upper:
                    options[k] = option
            packed = self.pack(nodes[s], s)
            k = len(packed)
            if k <= upper:
                packed = [self.order_route(s, row, until) for row in packed]
                value = sum(self.route_cost(s, row) for row in packed)
                if k not in options or value < options[k][0]:
                    options[k] = (value, packed)
            if not options:
                return None
            choices[s] = options
        dp = {0: (0., [])}
        for options in choices:
            new = {}
            for used, (cost, counts) in dp.items():
                for k, (value, rows) in options.items():
                    t = used + k
                    if t <= self.T2 and (t not in new or cost + value < new[t][0]):
                        new[t] = (cost + value, counts + [k])
            dp = new
            if not dp:
                return None
        _, ks = min(dp.values(), key=lambda v: v[0])
        return [[s, row, sum(self.q[i] for i in row)] for s in range(self.m)
                for row in choices[s][ks[s]][1]]

    def feasibility_search(self, until):
        # The explicit-stack search does not depend on recursion depth; only one branch is kept among vehicles with equal residual capacity.
        groups = sorted((h for h in range(self.H) if self.groups[h]),
                        key=lambda h: (-self.gd[h], -max(self.q[i] for i in self.groups[h])))
        items = [i for h in groups for i in sorted(self.groups[h], key=lambda i: -self.q[i])]
        assigned = [min(range(self.m), key=lambda s: self.proxy[h][s]) for h in range(self.H)]
        starts = set()
        p = 0
        for h in groups:
            starts.add(p); p += len(self.groups[h]); assigned[h] = -1
        bins = [[] for _ in range(self.m)]
        bl = [[] for _ in range(self.m)]
        sl = [0] * self.m
        frames, chosen = [None] * len(items), [None] * len(items)
        depth, vehicles = 0, 0
        while depth >= 0 and time.perf_counter() < until:
            if depth == len(items):
                routes = [[s, self.order_route(s, row, until), sum(self.q[i] for i in row)]
                          for s in range(self.m) for row in bins[s]]
                return routes, assigned
            i = items[depth]; h = self.group_of[i]; q = self.q[i]
            if frames[depth] is None:
                options = []
                sats = (sorted(range(self.m), key=lambda s: self.proxy[h][s])
                        if depth in starts else [assigned[h]])
                for s in sats:
                    if sl[s] + (self.gd[h] if depth in starts else q) > self.cap[s] + 1e-8:
                        continue
                    unique = set()
                    for b in sorted(range(len(bins[s])), key=lambda b: -bl[s][b]):
                        if bl[s][b] + q <= self.Q and bl[s][b] not in unique:
                            unique.add(bl[s][b]); options.append((s, b, False))
                    if q <= self.Q and len(bins[s]) < self.maxv[s] and vehicles < self.T2:
                        options.append((s, len(bins[s]), True))
                frames[depth] = options[::-1]
            if frames[depth]:
                s, b, new = frames[depth].pop()
                if new:
                    bins[s].append([]); bl[s].append(0); vehicles += 1
                bins[s][b].append(i); bl[s][b] += q; sl[s] += q
                if depth in starts:
                    assigned[h] = s
                chosen[depth] = (s, b, new)
                depth += 1
                if depth < len(items):
                    frames[depth] = None
            else:
                frames[depth] = None
                depth -= 1
                if depth >= 0:
                    i = items[depth]; h = self.group_of[i]; q = self.q[i]
                    s, b, new = chosen[depth]
                    bins[s][b].pop(); bl[s][b] -= q; sl[s] -= q
                    if new:
                        bins[s].pop(); bl[s].pop(); vehicles -= 1
                    if depth in starts:
                        assigned[h] = -1
        return None

    def swap_star(self, routes, b, t, until):
        s, aa, la = routes[b]
        _, bb, lb = routes[t]
        if not aa or not bb:
            return False
        d, q, base = self.d, self.q, self.n + s
        a, c = [base] + aa + [base], [base] + bb + [base]
        # Removing a customer invalidates at most two old insertion edges, so keeping the three best edges per customer suffices.
        def cheapest(nodes, dest):
            result = {}
            for i in nodes:
                if time.perf_counter() >= until:
                    return None
                best = []
                for p in range(len(dest) - 1):
                    value = d[dest[p]][i] + d[i][dest[p + 1]] - d[dest[p]][dest[p + 1]]
                    best.append((value, p))
                best.sort()
                result[i] = best[:3]
            return result
        into_b, into_a = cheapest(aa, c), cheapest(bb, a)
        if into_b is None or into_a is None:
            return False
        remove_a = [d[a[p - 1]][a[p + 1]] - d[a[p - 1]][i] - d[i][a[p + 1]]
                    for p, i in enumerate(a[1:-1], 1)]
        remove_b = [d[c[p - 1]][c[p + 1]] - d[c[p - 1]][i] - d[i][c[p + 1]]
                    for p, i in enumerate(c[1:-1], 1)]
        best = None
        for p, i in enumerate(aa, 1):
            if time.perf_counter() >= until:
                break
            for r, j in enumerate(bb, 1):
                if la - q[i] + q[j] > self.Q or lb - q[j] + q[i] > self.Q:
                    continue
                vb = d[c[r - 1]][i] + d[i][c[r + 1]] - d[c[r - 1]][c[r + 1]]
                pb = r - 1
                for value, pos in into_b[i]:
                    if pos != r - 1 and pos != r:
                        if value < vb:
                            vb, pb = value, pos if pos < r else pos - 1
                        break
                va = d[a[p - 1]][j] + d[j][a[p + 1]] - d[a[p - 1]][a[p + 1]]
                pa = p - 1
                for value, pos in into_a[j]:
                    if pos != p - 1 and pos != p:
                        if value < va:
                            va, pa = value, pos if pos < p else pos - 1
                        break
                delta = remove_a[p - 1] + remove_b[r - 1] + va + vb
                if delta < -1e-8 and (best is None or delta < best[0]):
                    best = (delta, p - 1, r - 1, pa, pb)
        if best is None:
            return False
        _, p, r, pa, pb = best
        i, j = aa.pop(p), bb.pop(r)
        aa.insert(pa, j); bb.insert(pb, i)
        routes[b][2] += q[j] - q[i]; routes[t][2] += q[i] - q[j]
        return True

    def polish(self, routes, until):
        d, demand, Q = self.d, self.q, self.Q
        owner, position = [-1] * self.n, [-1] * self.n
        prefix = [None] * len(routes)
        def refresh(b):
            row = routes[b][1]
            acc = [0]
            for p, i in enumerate(row):
                owner[i], position[i] = b, p
                acc.append(acc[-1] + demand[i])
            prefix[b] = acc
            routes[b][2] = acc[-1]
        for b in range(len(routes)):
            refresh(b)
        changed = True
        while changed and time.perf_counter() < until:
            changed = False
            order = list(range(self.n)); self.rng.shuffle(order)
            pairs = set()
            for i in order:
                if time.perf_counter() >= until:
                    break
                b, p = owner[i], position[i]
                if b < 0:
                    continue
                s, ra, la = routes[b]
                base = self.n + s
                for j in self.nearest(i):
                    t, r = owner[j], position[j]
                    if t < 0 or routes[t][0] != s:
                        continue
                    rb, lb = routes[t][1], routes[t][2]
                    ap = base if p == 0 else ra[p - 1]
                    an = base if p + 1 == len(ra) else ra[p + 1]
                    bp = base if r == 0 else rb[r - 1]
                    bn = base if r + 1 == len(rb) else rb[r + 1]
                    if b == t:
                        lo, hi = min(p, r), max(p, r)
                        if lo == hi:
                            continue
                        v0 = base if lo == 0 else ra[lo - 1]
                        v1, v2 = ra[lo], ra[hi]
                        v3 = base if hi + 1 == len(ra) else ra[hi + 1]
                        if d[v0][v2] + d[v1][v3] < d[v0][v1] + d[v2][v3] - 1e-8:
                            ra[lo:hi + 1] = reversed(ra[lo:hi + 1])
                            refresh(b); changed = True; break
                        if r != p - 1:
                            delta = (d[ap][an] - d[ap][i] - d[i][an] +
                                     d[j][i] + d[i][bn] - d[j][bn])
                            if delta < -1e-8:
                                ra.pop(p); ra.insert(r if r > p else r + 1, i)
                                refresh(b); changed = True; break
                        continue
                    pairs.add((min(b, t), max(b, t)))
                    removal = d[ap][an] - d[ap][i] - d[i][an]
                    if lb + demand[i] <= Q:
                        v1 = d[bp][i] + d[i][j] - d[bp][j]
                        v2 = d[j][i] + d[i][bn] - d[j][bn]
                        if removal + min(v1, v2) < -1e-8:
                            ra.pop(p); rb.insert(r if v1 <= v2 else r + 1, i)
                            refresh(b); refresh(t); changed = True; break
                    if la - demand[i] + demand[j] <= Q and lb - demand[j] + demand[i] <= Q:
                        delta = (d[ap][j] + d[j][an] + d[bp][i] + d[i][bn] -
                                 d[ap][i] - d[i][an] - d[bp][j] - d[j][bn])
                        if delta < -1e-8:
                            ra[p], rb[r] = j, i
                            refresh(b); refresh(t); changed = True; break
                    ca, cb = prefix[b][p + 1], prefix[t][r]
                    delta = d[i][j] + d[bp][an] - d[i][an] - d[bp][j]
                    if delta < -1e-8 and ca + lb - cb <= Q and cb + la - ca <= Q:
                        routes[b][1], routes[t][1] = ra[:p + 1] + rb[r:], rb[:r] + ra[p + 1:]
                        refresh(b); refresh(t); changed = True; break
                    delta = d[i][bp] + d[an][j] - d[i][an] - d[bp][j]
                    if delta < -1e-8 and ca + cb <= Q and la + lb - ca - cb <= Q:
                        routes[b][1] = ra[:p + 1] + rb[:r][::-1]
                        routes[t][1] = ra[p + 1:][::-1] + rb[r:]
                        refresh(b); refresh(t); changed = True; break
                    if p + 1 < len(ra):
                        k = ra[p + 1]
                        if lb + demand[i] + demand[k] <= Q:
                            after = base if p + 2 == len(ra) else ra[p + 2]
                            rem = d[ap][after] - d[ap][i] - d[k][after]
                            options = [(d[bp][i] + d[k][j] - d[bp][j], r, False),
                                       (d[j][i] + d[k][bn] - d[j][bn], r + 1, False),
                                       (d[bp][k] + d[i][j] - d[bp][j], r, True),
                                       (d[j][k] + d[i][bn] - d[j][bn], r + 1, True)]
                            value, at, reverse = min(options)
                            if rem + value < -1e-8:
                                del ra[p:p + 2]
                                rb[at:at] = [k, i] if reverse else [i, k]
                                refresh(b); refresh(t); changed = True; break
            if not changed and time.perf_counter() < until:
                ordered = list(pairs); self.rng.shuffle(ordered)
                for b, t in ordered:
                    if time.perf_counter() >= until:
                        break
                    if self.swap_star(routes, b, t, until):
                        refresh(b); refresh(t); changed = True
                        break
        routes[:] = [row for row in routes if row[1]]

    def genome(self, state, randomized):
        order = []
        for s in range(self.m):
            rows = [row[1][:] for row in state[0] if row[0] == s]
            x, y = self.xy[self.n + s]
            rows.sort(key=lambda seq: math.atan2(sum(self.xy[i][1] - y for i in seq),
                                                 sum(self.xy[i][0] - x for i in seq)))
            if rows and randomized:
                cut = self.rng.randrange(len(rows)); rows = rows[cut:] + rows[:cut]
            for seq in rows:
                if randomized and self.rng.random() < .5:
                    seq.reverse()
                order.extend(seq)
        return order

    def mutate_assignment(self, a):
        groups = [h for h in range(self.H) if self.groups[h]]
        if self.m < 2 or not groups:
            return a
        h = self.rng.choice(groups); old = a[h]
        target = min((s for s in range(self.m) if s != old),
                     key=lambda s: self.proxy[h][s] * self.rng.uniform(.7, 1.3))
        load = [0] * self.m
        for g, s in enumerate(a):
            load[s] += self.gd[g]
        if load[target] + self.gd[h] <= self.cap[target] and self.rng.random() < .7:
            a[h] = target
            return a
        other = [g for g in groups if a[g] == target and
                 load[target] + self.gd[h] - self.gd[g] <= self.cap[target] and
                 load[old] + self.gd[g] - self.gd[h] <= self.cap[old]]
        if other:
            g = min(other, key=lambda g: (self.proxy[g][old] - self.proxy[g][target] +
                                         abs(self.gd[g] - self.gd[h])) * self.rng.uniform(.5, 1.5))
            a[h], a[g] = target, old
        return a

    def offspring(self, first, second, fraction, until):
        a = first[1][:]
        if first is not second and self.rng.random() < .5:
            a = [second[1][h] if self.rng.random() < fraction else a[h] for h in range(self.H)]
            loads = [0] * self.m
            for h, s in enumerate(a):
                loads[s] += self.gd[h]
            if any(loads[s] > self.cap[s] for s in range(self.m)):
                a = first[1][:]
        if self.rng.random() < .6:
            a = self.mutate_assignment(a)
        x = self.genome(first, True)
        if self.n > 1:
            if first is not second and self.rng.random() < .65:
                y = self.genome(second, True)
                left = self.rng.randrange(self.n)
                length = max(1, int(self.n * self.rng.uniform(.1, max(.11, fraction))))
                right = min(self.n, left + length)
                segment = x[left:right]; keep = set(segment)
                rest = [i for i in y if i not in keep]
                x = rest[:left] + segment + rest[left:]
            else:
                left = self.rng.randrange(self.n - 1)
                width = max(2, int(math.sqrt(self.n) * (1 + self.strength) * fraction))
                right = min(self.n, left + self.rng.randint(2, max(2, width)))
                if self.rng.random() < .5:
                    x[left:right] = reversed(x[left:right])
                else:
                    chunk = x[left:right]; del x[left:right]
                    pos = self.rng.randrange(len(x) + 1); x[pos:pos] = chunk
        routes = self.decode(a, x, until)
        return (routes, a) if routes is not None else None

    def shake(self, state, until):
        routes = [[s, row[:], q] for s, row, q in state[0]]
        bysat = [[] for _ in range(self.m)]
        for b, (s, row, q) in enumerate(routes):
            bysat[s].append(b)
        eligible = [s for s in range(self.m) if bysat[s]]
        if not eligible:
            return routes
        # Perturbation uses a wall-clock quota; the number of perturbations grows with the budget and the actual processing speed.
        while time.perf_counter() < until:
            s = self.rng.choice(eligible)
            b, t = self.rng.choice(bysat[s]), self.rng.choice(bysat[s])
            aa, bb = routes[b][1], routes[t][1]
            p, r = self.rng.randrange(len(aa)), self.rng.randrange(len(bb))
            if b == t:
                lo, hi = min(p, r), max(p, r)
                aa[lo:hi + 1] = reversed(aa[lo:hi + 1])
            else:
                i, j = aa[p], bb[r]
                na, nb = routes[b][2] - self.q[i] + self.q[j], routes[t][2] - self.q[j] + self.q[i]
                if na <= self.Q and nb <= self.Q:
                    aa[p], bb[r] = j, i
                    routes[b][2], routes[t][2] = na, nb
        return routes

    def run(self):
        if not self.n:
            a = [0] * self.H
            self.remember(self.state([], a, self.end))
            return self.output()
        initial_end = min(self.end, self.start + .12 * self.seconds)
        result = self.construct(initial_end)
        if result is not None:
            self.remember(self.state(*result, initial_end))
        if self.best is None:
            until = min(self.end, time.perf_counter() + .07 * self.seconds)
            result = self.feasibility_search(until)
            if result is not None:
                self.remember(self.state(*result, self.end))
        if self.best is None:
            result = self.capacity_model(min(self.end, self.start + .8 * self.seconds))
            if result is not None:
                self.remember(self.state(*result, self.end))
        while self.best is None and time.perf_counter() < self.end:
            until = min(self.end, time.perf_counter() + .04 * self.seconds)
            result = self.construct(until, True)
            if result is None:
                result = self.feasibility_search(until)
            if result is not None:
                self.remember(self.state(*result, self.end))
        if self.best is None:
            return self.output()
        # The initial refinement only modifies a copy, so any mid-way exit does not destroy the saved feasible solution.
        routes = [[s, row[:], q] for s, row, q in self.best[0]]
        until = min(self.end, time.perf_counter() + .17 * self.seconds)
        self.polish(routes, until)
        self.remember(self.state(routes, self.best[1], self.end))
        last_gain = time.perf_counter()
        self.first_due = time.perf_counter()
        while time.perf_counter() < self.end:
            now = time.perf_counter()
            remaining = self.end - now
            before = self.score(self.best)
            if now >= self.first_due and remaining > .04 * self.seconds:
                self.optimize_first(self.best, min(self.end, now + .055 * self.seconds))
                if self.population:
                    candidate = min((v[0] for v in self.population), key=self.score)
                    if self.score(candidate) < self.score(self.best):
                        self.best = candidate
                self.first_due = time.perf_counter() + self.seconds / (1 + self.strength)
                if self.score(self.best) < before - 1e-8:
                    last_gain = time.perf_counter()
                continue
            self.population.sort(key=lambda v: self.score(v[0]))
            pool = [item[0] for item in self.population]
            parent = min(self.rng.choices(pool, k=2), key=self.score)
            donor = self.rng.choice(pool)
            stagnation = min(1., (now - last_gain) / max(1e-6, self.seconds / (1 + self.strength)))
            fraction = min(.8, .12 + .08 * self.strength + .25 * stagnation)
            child_until = min(self.end, now + remaining * .10)
            child = self.offspring(parent, donor, fraction, child_until)
            if child is None:
                shake_end = min(self.end, time.perf_counter() + self.seconds * .00003 * (1 + stagnation))
                routes = self.shake(parent, shake_end)
                a = parent[1][:]
            else:
                routes, a = child
            if time.perf_counter() >= self.end:
                self.remember(self.state(routes, a, self.end))
                break
            local_share = self.seconds / (5 + math.sqrt(max(1, self.n)) / (1 + self.strength))
            until = min(self.end, time.perf_counter() + min(remaining * .18, local_share))
            self.polish(routes, until)
            self.remember(self.state(routes, a, self.end))
            if self.score(self.best) < before - 1e-8:
                last_gain = time.perf_counter()
        return self.output()

    def output(self):
        if self.best is None:
            # For valid feasible inputs the normal path has already saved a solution during construction; this branch only preserves the interface shape.
            return {'first_echelon_routes': [[self.depot_id, self.depot_id] for _ in range(self.T1)],
                    'first_echelon_deliveries': [{'satellite_id': self.sid[s], 'vehicle_id': k, 'quantity': 0.}
                                               for k in range(self.T1) for s in range(self.m)],
                    'second_echelon_routes': [],
                    'group_assignments': {'group_' + str(g): self.sid[0] for g in self.gid},
                    'customer_satellite_assignments': {'customer_' + str(i): self.sid[0] for i in self.cid}}
        routes, a, demand, _ = self.best
        _, paths, cargo = self.first_cache[tuple(demand)]
        first = [[self.depot_id] + [self.sid[s] for s in row] + [self.depot_id] for row in paths]
        first.extend([[self.depot_id, self.depot_id] for _ in range(self.T1 - len(first))])
        deliveries = [{'satellite_id': self.sid[s], 'vehicle_id': k,
                       'quantity': float(cargo[k].get(s, 0.)) if k < len(cargo) else 0.}
                      for k in range(self.T1) for s in range(self.m)]
        return {'first_echelon_routes': first,
                'first_echelon_deliveries': deliveries,
                'second_echelon_routes': [[self.sid[s]] + [self.cid[i] for i in row] + [self.sid[s]]
                                          for s, row, q in routes],
                'group_assignments': {'group_' + str(self.gid[h]): self.sid[a[h]] for h in range(self.H)},
                'customer_satellite_assignments': {'customer_' + str(self.cid[i]): self.sid[a[self.group_of[i]]]
                                                   for i in range(self.n)}}


def solve(instance: dict, time_limit_s: float) -> dict:
    started = time.perf_counter()
    engine = _Engine(instance, time_limit_s, started)
    try:
        return engine.run()
    except Exception:
        if engine.best is not None:
            return engine.output()
        result = engine.feasibility_search(engine.end)
        if result is not None:
            engine.remember(engine.state(*result, engine.end))
        return engine.output()
