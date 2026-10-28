"""Hybrid dual pricing and route-elimination search
Maintains feasible solutions with exact integer checks, regret repair and ejection chains, and reduces vehicles via dual-guided label pricing and set-covering recombination.
"""
from array import array
from decimal import Decimal
from fractions import Fraction
import math
import random
import time

def _empty(status):
    return {"status": status, "routes": [], "route_departure_times": [],
            "customer_departure_times": [], "route_return_times": [],
            "customer_loads": []}

def _integers(values, minimum_digits=0):
    
    
    dec = [Decimal(str(x)) for x in values]
    digits = max([minimum_digits] + [-x.as_tuple().exponent for x in dec])
    scale = 10 ** digits
    result = []
    for x in dec:
        num, den = x.as_integer_ratio()
        result.append(num * scale // den)
    return result, scale

def _tenths(x):
    if x < 100000000000000:
        return round(x * 10)
    num, den = Decimal(str(x)).as_integer_ratio()
    return num * 10 // den

class _Stop(Exception):
    pass

class _Route:
    __slots__ = ("nodes", "path", "early", "late", "load", "cost", "mask")

    def __init__(self, nodes, early, late, load, cost):
        self.nodes = tuple(nodes)
        self.path = (0,) + self.nodes + (0,)
        self.early = early
        self.late = late
        self.load = load
        self.cost = cost
        mask = 0
        for u in nodes:
            mask |= 1 << (u - 1)
        self.mask = mask

class _Search:
    def __init__(self, instance, budget, started):
        self.n = len(instance["customer_demands"])
        self.base = instance["base_travel_times"]
        loads, self.qscale = _integers(
            [instance["vehicle_capacity"]] + instance["customer_demands"])
        self.capacity = loads[0]
        self.q = [0] + loads[1:]
        values = list(instance["depot_time_window"])
        for window in instance["customer_time_windows"]:
            values.extend(window)
        values.extend(instance["customer_service_times"])
        times, self.tscale = _integers(values, 1)
        self.a = [times[0]] + [times[2 + 2 * k] for k in range(self.n)]
        self.b = [times[1]] + [times[3 + 2 * k] for k in range(self.n)]
        self.s = [0] + times[2 + 2 * self.n:]
        self.factor = self.tscale // 10
        self.budget = budget
        self.end = started + budget - min(budget * .08, .03 + budget * .008 + self.n * .000035)
        self.rng = random.Random(917623)
        self.strength = math.log1p(budget)
        self.rows = [None] * (self.n + 1)
        self.small_ints = max(abs(x) for x in times) < (1 << 58)
        self.singletons = []
        self.best = []
        self.pool = {}
        self.pool_limit = max(self.n, int(self.n * (2 + self.strength) ** 2))
        self.penalty = [1.0] * (self.n + 1)
        self.unit = 1
        self.lower_bound = 0
        self.elites = []
        self.elite_limit = max(2, int(2 + self.strength))
        self.dual = [0.0] + [1.0] * self.n
        self.adjacency = [None] * (self.n + 1)
        self.backend = None
        self.enumeration = None
        self.exact_routes = {}
        self.exact_complete = False
        self.exact_size = max(2, int(7 + 2 * self.strength))
        self.pricing_width = max(self.n, int(math.sqrt(self.n) * (2 + self.strength) ** 2))
        self.neighbor_width = max(2, int(math.sqrt(self.n) * (1 + self.strength)))

    def check(self, stop=None):
        if time.perf_counter() >= (self.end if stop is None else min(stop, self.end)):
            raise _Stop()

    def row(self, u):
        row = self.rows[u]
        if row is None:
            self.check()
            
            values = []
            for x in self.base[u]:
                self.check()
                values.append(_tenths(x) * self.factor)
            if self.small_ints:
                try:
                    row = array('q', values)
                except OverflowError:
                    row = values
            else:
                row = values
            self.rows[u] = row
        return row

    def initialize(self):
        total = 0
        for u in range(1, self.n + 1):
            outward = _tenths(self.base[0][u]) * self.factor
            inward = _tenths(self.base[u][0]) * self.factor + self.s[u]
            earliest = max(self.a[u], self.a[0] + outward)
            returned = max(self.a[0], earliest + inward)
            if self.q[u] > self.capacity or earliest > self.b[u] or returned > self.b[0]:
                
                
                return False
            latest = min(self.b[u], self.b[0] - inward)
            self.singletons.append(_Route(
                (u,), (self.a[0], earliest, returned),
                (min(self.b[0], latest - outward), latest, self.b[0]),
                self.q[u], outward + inward - self.s[u]))
            total += self.q[u]
        self.best = self.singletons[:]
        self.lower_bound = (total + self.capacity - 1) // self.capacity
        self.lower_bound = max(self.lower_bound,
                               sum(2 * q > self.capacity for q in self.q))
        if self.n:
            self.unit = max(1, sum(r.cost for r in self.singletons) // self.n)
        return True

    def make(self, nodes):
        nodes = tuple(nodes)
        path = (0,) + nodes + (0,)
        early = [self.a[0]]
        load = 0
        cost = 0
        previous = 0
        mask = 0
        for u in path[1:]:
            self.check()
            if u:
                bit = 1 << (u - 1)
                if mask & bit:
                    return None
                mask |= bit
            distance = self.row(previous)[u]
            t = max(self.a[u], early[-1] + distance + self.s[previous])
            load += self.q[u]
            if t > self.b[u] or load > self.capacity:
                return None
            early.append(t)
            cost += distance
            previous = u
        late = [0] * len(path)
        late[-1] = self.b[0]
        for p in range(len(path) - 2, -1, -1):
            u, v = path[p], path[p + 1]
            late[p] = min(self.b[u], late[p + 1] - self.row(u)[v] - self.s[u])
        return _Route(nodes, early, late, load, cost)

    def insertion(self, u, route, delay_weight, fill_weight):
        self.check()
        if route.load + self.q[u] > self.capacity:
            return None
        du = self.row(u)
        best = math.inf
        best_pos = -1
        
        
        
        for p, (v, w) in enumerate(zip(route.path, route.path[1:])):
            self.check()
            dv = self.row(v)
            arrival = max(self.a[u], route.early[p] + dv[u] + self.s[v])
            if arrival > self.b[u]:
                continue
            following = arrival + du[w] + self.s[u]
            if following > route.late[p + 1]:
                continue
            delta = dv[u] + du[w] - dv[w]
            delay = max(0, following - route.early[p + 1])
            score = delta / self.unit + delay_weight * (delay / self.unit)
            if score < best:
                best, best_pos = score, p
        if best_pos < 0:
            return None
        return best - fill_weight * (route.load / self.capacity), best_pos

    def insert_at(self, u, route, position):
        return self.make(route.nodes[:position] + (u,) + route.nodes[position:])

    def construct(self, stop, first=False):
        order = list(range(1, self.n + 1))
        rng = self.rng
        mode = 0 if first else rng.randrange(5)
        if mode == 0:
            order.sort(key=lambda u: (self.b[u], -self.q[u]))
        elif mode == 1:
            order.sort(key=lambda u: (-self.q[u], self.b[u]))
        elif mode == 2:
            order.sort(key=lambda u: (-self.singletons[u - 1].cost, self.b[u]))
        elif mode == 3:
            order.sort(key=lambda u: (self.a[u], self.b[u]))
        else:
            rng.shuffle(order)
        routes = []
        delay_weight = .15 if first else rng.random() * .6
        fill_weight = .2 if first else rng.random()
        at = 0
        try:
            while at < self.n:
                self.check(stop)
                u = order[at]
                best_option = None
                for k, r in enumerate(routes):
                    self.check(stop)
                    option = self.insertion(u, r, delay_weight, fill_weight)
                    if option is not None:
                        score, pos = option
                        if not first:
                            score += rng.uniform(-.12, .12)
                        if best_option is None or score < best_option[0]:
                            best_option = score, k, pos
                if best_option is None:
                    routes.append(self.singletons[u - 1])
                else:
                    _, k, pos = best_option
                    added = self.insert_at(u, routes[k], pos)
                    if added is None:
                        routes.append(self.singletons[u - 1])
                    else:
                        routes[k] = added
                at += 1
        except _Stop:
            pass
        
        routes.extend(self.singletons[u - 1] for u in order[at:])
        self.remember(routes)
        self.publish(routes)
        return routes

    def repair(self, routes, bank, stop):
        routes = routes[:]
        bank = set(bank)
        cache = {u: {} for u in bank}
        rng = self.rng
        delay = rng.random() * .55
        fill = rng.random() * .6
        noise = rng.random() * .16
        regret_mode = rng.random() < .85
        regret_depth = 2 + int(rng.random() * min(len(routes), 1 + self.strength / 2))
        try:
            while bank:
                self.check(stop)
                selected = None
                priority = -math.inf
                for u in bank:
                    self.check(stop)
                    options = []
                    entries = cache[u]
                    for k, route in enumerate(routes):
                        old = entries.get(k)
                        if old is None or old[0] is not route:
                            value = self.insertion(u, route, delay, fill)
                            if value is not None:
                                value = (value[0] + rng.uniform(-noise, noise), value[1])
                            entries[k] = (route, value)
                        else:
                            value = old[1]
                        if value is not None:
                            options.append((value[0], k, value[1]))
                    if not options:
                        continue
                    options.sort()
                    score, k, pos = options[0]
                    if regret_mode:
                        regret = sum(x[0] - score for x in options[1:regret_depth])
                        regret += (regret_depth - min(regret_depth, len(options))) * 3.0
                        value = regret + .25 * self.penalty[u] + .15 * self.q[u] / self.capacity
                        value -= .025 * score
                    else:
                        value = -score + .25 * self.penalty[u]
                    if value > priority:
                        priority = value
                        selected = u, k, pos
                if selected is None:
                    break
                u, k, pos = selected
                added = self.insert_at(u, routes[k], pos)
                if added is None:
                    
                    break
                routes[k] = added
                bank.remove(u)
                del cache[u]
        except _Stop:
            pass
        return routes, bank

    def eliminate(self, routes):
        
        ranked = sorted(range(len(routes)), key=lambda k:
                        len(routes[k].nodes) + routes[k].load / self.capacity)
        rank = int((self.rng.random() ** (1.3 + .1 * self.strength)) * len(ranked))
        victim = ranked[rank]
        return routes[:victim] + routes[victim + 1:], set(routes[victim].nodes)

    def ruin(self, routes, bank, count, operator):
        rng = self.rng
        assigned = [u for r in routes for u in r.nodes]
        count = min(count, len(assigned))
        if not count:
            return routes[:], set(bank)
        removed = set()
        if operator == 0:
            removed.update(rng.sample(assigned, count))
        elif operator in (1, 4):
            seed = rng.choice(tuple(bank)) if bank and operator == 4 else rng.choice(assigned)
            ds = self.row(seed)
            span = max(1, self.b[0] - self.a[0])
            scored = [(ds[u] / self.unit + .6 * abs(self.a[u] - self.a[seed]) / span
                       + .25 * abs(self.q[u] - self.q[seed]) / self.capacity
                       + rng.random() * .35, u) for u in assigned]
            scored.sort()
            removed.update(u for _, u in scored[:count])
        elif operator == 2:
            scored = []
            for r in routes:
                for p, u in enumerate(r.nodes):
                    v, w = r.path[p], r.path[p + 2]
                    saving = self.row(v)[u] + self.row(u)[w] - self.row(v)[w]
                    scored.append((saving / self.unit + rng.random(), u))
            scored.sort(reverse=True)
            removed.update(u for _, u in scored[:count])
        else:
            indices = list(range(len(routes)))
            rng.shuffle(indices)
            indices.sort(key=lambda k: len(routes[k].nodes) * rng.uniform(.5, 1.5))
            for k in indices:
                nodes = list(routes[k].nodes)
                if len(nodes) > count - len(removed):
                    length = count - len(removed)
                    start = rng.randrange(len(nodes) - length + 1)
                    nodes = nodes[start:start + length]
                removed.update(nodes)
                if len(removed) >= count:
                    break
        new_routes = []
        for r in routes:
            if any(u in removed for u in r.nodes):
                shorter = self.make(u for u in r.nodes if u not in removed)
                if shorter is None:
                    return routes[:], set(bank)
                new_routes.append(shorter)
            else:
                new_routes.append(r)
        return new_routes, set(bank) | removed

    def eject(self, routes, bank, stop):






        routes, bank = routes[:], set(bank)
        rng = self.rng
        duration = max(0.000001, stop - time.perf_counter())
        tabu = {}
        anchor, left = routes[:], set(bank)
        depth = max(2, int(math.log2(2 + self.budget)))
        near_count = max(2, int(math.sqrt(len(routes)) * (1 + self.strength / 2)))
        try:
            while bank:
                self.check(stop)
                u = max(bank, key=lambda v: self.penalty[v] * rng.uniform(.7, 1.3))
                direct = None
                for k, r in enumerate(routes):
                    self.check(stop)
                    option = self.insertion(u, r, .15, .15)
                    if option is not None and (direct is None or option[0] < direct[0]):
                        direct = option[0], k, option[1]
                if direct is not None:
                    _, k, pos = direct
                    added = self.insert_at(u, routes[k], pos)
                    if added is not None:
                        routes[k] = added
                        bank.remove(u)
                        if len(bank) < len(left):
                            anchor, left = routes[:], set(bank)
                        continue
                self.penalty[u] += 1.0
                now = time.perf_counter()
                du = self.row(u)
                candidates = sorted(range(len(routes)), key=lambda k:
                    min((du[v] for v in routes[k].nodes), default=0) / self.unit
                    + max(0, routes[k].load + self.q[u] - self.capacity) / self.capacity)
                if len(candidates) > near_count:
                    candidates = candidates[:near_count - 1] + [rng.choice(candidates[near_count - 1:])]
                best, value = None, math.inf
                for k in candidates:
                    self.check(stop)
                    r = routes[k]
                    weights = {v: self.penalty[v] + (2 + self.strength if tabu.get(v, 0) > now else 0)
                               for v in r.nodes}
                    ranked = sorted(r.nodes, key=lambda v: weights[v] + rng.random() * .2)
                    for v in ranked:
                        self.check(stop)
                        if weights[v] - .04 >= value or r.load - self.q[v] + self.q[u] > self.capacity:
                            continue
                        reduced = self.make(x for x in r.nodes if x != v)
                        if reduced is None:
                            continue
                        option = self.insertion(u, reduced, .15, 0)
                        if option is not None:
                            score = weights[v] + .03 * math.tanh(option[0])
                            if score < value:
                                added = self.insert_at(u, reduced, option[1])
                                if added is not None:
                                    value, best = score, (k, (v,), added)
                    width = min(len(ranked), max(2, int(math.sqrt(len(ranked)) * (1 + self.strength))))
                    short = ranked[:width]
                    if len(ranked) > width:
                        short[-1] = rng.choice(ranked[width:])
                    for p, v in enumerate(short):
                        for w in short[p + 1:]:
                            self.check(stop)
                            weight = weights[v] + weights[w]
                            if weight - .04 >= value or r.load - self.q[v] - self.q[w] + self.q[u] > self.capacity:
                                continue
                            reduced = self.make(x for x in r.nodes if x != v and x != w)
                            if reduced is None:
                                continue
                            option = self.insertion(u, reduced, .15, 0)
                            if option is not None:
                                score = weight + .03 * math.tanh(option[0])
                                if score < value:
                                    added = self.insert_at(u, reduced, option[1])
                                    if added is not None:
                                        value, best = score, (k, (v, w), added)
                    if best is None and len(ranked) > 2:
                        ranked.sort(key=lambda v: weights[v] / max(self.q[v] / self.capacity, 1 / self.n)
                                    * rng.uniform(.7, 1.3))
                        removed, weight = set(), 0.0
                        for v in ranked[:depth]:
                            self.check(stop)
                            removed.add(v)
                            weight += weights[v]
                            if r.load - sum(self.q[x] for x in removed) + self.q[u] > self.capacity:
                                continue
                            reduced = self.make(x for x in r.nodes if x not in removed)
                            if reduced is None:
                                continue
                            option = self.insertion(u, reduced, .15, 0)
                            if option is not None:
                                added = self.insert_at(u, reduced, option[1])
                                if added is not None:
                                    value, best = weight, (k, tuple(removed), added)
                                    break
                if best is None:
                    break
                k, victims, added = best
                routes[k] = added
                bank.remove(u)
                bank.update(victims)
                tabu[u] = time.perf_counter() + duration / (4 + self.strength)
                if len(bank) < len(left):
                    anchor, left = routes[:], set(bank)
        except _Stop:
            pass
        self.remember(routes)
        if len(left) < len(bank):
            return anchor, left
        return routes, bank

    def exchange(self, source, stop):

        routes = source[:]
        if not routes:
            return routes
        rng = self.rng
        j = rng.randrange(len(routes))
        r = routes[j]
        try:
            if len(routes) > 1 and rng.random() < .8:
                k = rng.randrange(len(routes) - 1)
                if k >= j:
                    k += 1
                s = routes[k]
                prefix_r, prefix_s = [0], [0]
                for v in r.nodes:
                    prefix_r.append(prefix_r[-1] + self.q[v])
                for v in s.nodes:
                    prefix_s.append(prefix_s[-1] + self.q[v])
                first, second = list(range(len(r.path) - 1)), list(range(len(s.path) - 1))
                rng.shuffle(first)
                rng.shuffle(second)
                for p in first:
                    for z in second:
                        self.check(stop)
                        if prefix_r[p] + s.load - prefix_s[z] > self.capacity:
                            continue
                        if prefix_s[z] + r.load - prefix_r[p] > self.capacity:
                            continue
                        rr = self.make(r.nodes[:p] + s.nodes[z:])
                        if rr is None:
                            continue
                        ss = self.make(s.nodes[:z] + r.nodes[p:])
                        if ss is None:
                            continue
                        delta = rr.cost + ss.cost - r.cost - s.cost
                        old_empty = int(not r.nodes) + int(not s.nodes)
                        new_empty = int(not rr.nodes) + int(not ss.nodes)
                        if new_empty > old_empty or delta < 0 or rng.random() < .025:
                            routes[j], routes[k] = rr, ss
                            self.remember((rr, ss))
                            return routes
            elif len(r.nodes) > 2:
                positions = list(range(len(r.nodes) - 1))
                rng.shuffle(positions)
                for p in positions:
                    self.check(stop)
                    z = rng.randrange(p + 2, len(r.nodes) + 1)
                    rr = self.make(r.nodes[:p] + tuple(reversed(r.nodes[p:z])) + r.nodes[z:])
                    if rr is not None and rr.cost < r.cost:
                        routes[j] = rr
                        self.remember((rr,))
                        break
        except _Stop:
            pass
        return routes

    def answer(self):
        result = _empty("feasible")
        result["customer_departure_times"] = [0] * self.n
        result["customer_loads"] = [0] * self.n
        for r in self.best:
            result["routes"].append(list(r.path))
            result["route_departure_times"].append(self.a[0] / self.tscale)
            result["route_return_times"].append(r.early[-1] / self.tscale)
            load = 0
            for p, u in enumerate(r.nodes, 1):
                load += self.q[u]
                result["customer_loads"][u - 1] = load / self.qscale
                result["customer_departure_times"][u - 1] = r.early[p] / self.tscale
        return result

    def remember(self, routes):
        for r in routes:
            if len(r.nodes) > 1:
                old = self.pool.get(r.mask)
                if old is None or r.cost < old.cost:
                    self.pool[r.mask] = r
        if len(self.pool) > 2 * self.pool_limit:
            protected = {r.mask for r in self.best}
            for _, rs in self.elites:
                protected.update(r.mask for r in rs)
            ranked = sorted(self.pool.values(), key=lambda r: (
                r.mask in protected,
                len(r.nodes) + sum(self.dual[u] for u in r.nodes)
                + self.rng.random()), reverse=True)
            keep, covered = {}, 0
            for r in ranked:
                if r.mask in protected or r.mask & ~covered:
                    keep[r.mask] = r
                    covered |= r.mask
            for r in ranked:
                if len(keep) >= self.pool_limit:
                    break
                keep[r.mask] = r
            self.pool = keep

    def publish(self, routes):
        routes = [r for r in routes if r.nodes]
        mask = 0
        for r in routes:
            if mask & r.mask:
                return False
            mask |= r.mask
        if mask != (1 << self.n) - 1:
            return False
        improved = len(routes) < len(self.best)
        if improved:
            self.best = routes[:]
        self.remember(routes)
        if len(routes) == len(self.best):
            signature = tuple(sorted(r.mask for r in routes))
            self.elites = [(sig, rs) for sig, rs in self.elites
                           if len(rs) == len(self.best)]
            if not any(sig == signature for sig, _ in self.elites):
                self.elites.append((signature, routes[:]))
                if len(self.elites) > self.elite_limit:
                    del self.elites[self.rng.randrange(len(self.elites) - 1)]
        return improved

    def source(self):
        peers = [rs for _, rs in self.elites if len(rs) == len(self.best)]
        return self.rng.choice(peers) if peers and self.rng.random() < .65 else self.best

    def successors(self, u, stop):
        cached = self.adjacency[u]
        if cached is None:
            row = self.row(u)
            earliest = self.singletons[u - 1].early[1]
            cached = []
            for v in range(1, self.n + 1):
                self.check(stop)
                if v != u and self.q[u] + self.q[v] <= self.capacity:
                    t = max(self.a[v], earliest + row[v] + self.s[u])
                    if t <= self.singletons[v - 1].late[1]:
                        cached.append(v)
            self.adjacency[u] = cached
        return cached

    def enumerate_routes(self, stop):
        """For small problems keep the earliest label per subset and last customer; enumeration can continue across time slices."""
        if self.exact_complete or self.n > self.exact_size:
            return
        if self.enumeration is None:
            states, loads = {}, {}
            for u in range(1, self.n + 1):
                bit = 1 << (u - 1)
                states[bit] = {u: (self.singletons[u - 1].early[1], 0)}
                loads[bit] = self.q[u]
                self.exact_routes[bit] = self.singletons[u - 1]
            self.enumeration = [states, loads, 1]
        states, loads, _ = self.enumeration
        full = (1 << self.n) - 1
        while self.enumeration[2] <= full:
            self.check(stop)
            mask = self.enumeration[2]
            entries = states.get(mask)
            if entries is None:
                self.enumeration[2] += 1
                continue
            load = loads[mask]
            if mask not in self.exact_routes:
                last = min(entries, key=lambda u: entries[u][0] + self.row(u)[0] + self.s[u])
                seq, left = [], mask
                while last:
                    seq.append(last)
                    previous = states[left][last][1]
                    left ^= 1 << (last - 1)
                    last = previous
                r = self.make(reversed(seq))
                if r is not None:
                    self.exact_routes[mask] = r
                    self.remember((r,))
            for u, (arrival, _) in entries.items():
                self.check(stop)
                row = self.row(u)
                for v in self.successors(u, stop):
                    self.check(stop)
                    bit = 1 << (v - 1)
                    if mask & bit or load + self.q[v] > self.capacity:
                        continue
                    t = max(self.a[v], arrival + row[v] + self.s[u])
                    if t > self.singletons[v - 1].late[1]:
                        continue
                    nxt = mask | bit
                    labels = states.get(nxt)
                    if labels is None:
                        labels = {}
                        states[nxt] = labels
                        loads[nxt] = load + self.q[v]
                    old = labels.get(v)
                    if old is None or t < old[0]:
                        labels[v] = (t, u)
            self.enumeration[2] += 1
        self.exact_complete = True
        self.enumeration = None

    def price_routes(self, stop):
        """Dual-profit-driven elementary path beam search; time, load and visited set are all retained."""
        if self.exact_complete:
            return
        rng = self.rng
        span = max(1, self.b[0] - self.a[0])
        per_last = max(2, int(1 + self.strength))
        while time.perf_counter() < stop:
            self.check(stop)
            prices = self.dual[:]
            if rng.random() < .45:
                for u in range(1, self.n + 1):
                    prices[u] = prices[u] * rng.uniform(.8, 1.2) + .02 * self.q[u] / self.capacity
            weight_q, weight_t = rng.random() * .5, rng.random() * .7
            ordered = {}
            labels = [(r.mask, r.nodes[0], r.early[1], r.load,
                       prices[r.nodes[0]], r.nodes) for r in self.singletons]

            def rank(label):
                _, u, t, load, reward, _ = label
                travel = t - self.a[0] + self.row(u)[0] + self.s[u]
                return reward - weight_q * load / self.capacity - weight_t * travel / span

            while labels:
                self.check(stop)
                future = {}
                for mask, u, arrival, load, reward, seq in labels:
                    self.check(stop)
                    row = self.row(u)
                    if u not in ordered:
                        neighbors = self.successors(u, stop)[:]
                        neighbors.sort(key=lambda v: (
                            prices[v] / (.15 + row[v] / self.unit + .3 * self.q[v] / self.capacity)),
                            reverse=True)
                        if len(neighbors) > self.neighbor_width:
                            width = self.neighbor_width
                            extra = max(1, width // (2 + int(self.strength)))
                            neighbors = neighbors[:width - extra] + rng.sample(neighbors[width - extra:], extra)
                        ordered[u] = neighbors
                    for v in ordered[u]:
                        self.check(stop)
                        bit = 1 << (v - 1)
                        if mask & bit or load + self.q[v] > self.capacity:
                            continue
                        t = max(self.a[v], arrival + row[v] + self.s[u])
                        if t > self.singletons[v - 1].late[1]:
                            continue
                        new_mask = mask | bit
                        key = (new_mask, v)
                        old = future.get(key)
                        if old is None or t < old[2]:
                            future[key] = (new_mask, v, t, load + self.q[v], reward + prices[v], seq + (v,))
                if not future:
                    break
                labels, counts = [], {}
                for label in sorted(future.values(), key=rank, reverse=True):
                    self.check(stop)
                    v = label[1]
                    if counts.get(v, 0) < per_last:
                        counts[v] = counts.get(v, 0) + 1
                        labels.append(label)
                        if len(labels) >= self.pricing_width:
                            break
                for mask, u, t, load, reward, seq in labels:
                    self.check(stop)
                    if reward >= 1 - 1e-9 and mask not in self.pool:
                        r = self.make(seq)
                        if r is not None:
                            self.remember((r,))

    def cover_source(self):
        source = self.exact_routes if self.exact_complete else self.pool
        unique = dict(source)
        for r in self.best + self.singletons:
            unique[r.mask] = r
        return list(unique.values())

    def accept_cover(self, columns, chosen):
        used, routes = 0, []
        for k in chosen:
            r = columns[k]
            if r.mask & used:
                r = self.make(u for u in r.nodes if not used & (1 << (u - 1)))
                if r is None:
                    return False
            if r.nodes:
                routes.append(r)
                used |= r.mask
        if used != (1 << self.n) - 1:
            return False
        return self.publish(routes)

    def cover_dfs(self, stop):
        """Covering search over the route pool with no branch-count limit; no global optimality is claimed when the pool is incomplete."""
        columns = self.cover_source()
        by = [[] for _ in range(self.n + 1)]
        for k, r in enumerate(columns):
            self.check(stop)
            for u in r.nodes:
                by[u].append(k)
        order = sorted(range(1, self.n + 1), key=lambda u: len(by[u]))
        full = (1 << self.n) - 1
        target = len(self.best) - 1
        stack = [(full, sum(self.q), ())]
        seen = {}
        max_size = max(len(r.nodes) for r in columns)
        while stack:
            self.check(stop)
            left, demand, chosen = stack.pop()
            if not left:
                self.accept_cover(columns, chosen)
                return
            slots = target - len(chosen)
            if slots <= 0 or demand > slots * self.capacity or left.bit_count() > slots * max_size:
                continue
            if seen.get(left, target + 1) <= len(chosen):
                continue
            seen[left] = len(chosen)
            if len(seen) > self.pool_limit * (2 + self.strength):
                seen.clear()
            pivot = next(u for u in order if left & (1 << (u - 1)))
            options = {}
            required = demand - (slots - 1) * self.capacity
            for k in by[pivot]:
                self.check(stop)
                r = columns[k]
                covered = left & r.mask
                if covered in options or (slots == 1 and covered != left):
                    continue
                saved = sum(self.q[u] for u in r.nodes if left & (1 << (u - 1)))
                if saved >= required:
                    options[covered] = (covered.bit_count(), saved, k, covered)
            for _, saved, k, covered in sorted(options.values()):
                stack.append((left ^ covered, demand - saved, chosen + (k,)))
        if self.exact_complete:
            self.lower_bound = len(self.best)

    def load_backend(self, stop):
        if self.backend is not None:
            return bool(self.backend)
        if min(stop, self.end) - time.perf_counter() < .9:
            return False
        try:
            import numpy as np
            from scipy.optimize import linprog, milp, Bounds, LinearConstraint
            from scipy.sparse import csc_matrix
            self.backend = (np, linprog, milp, Bounds, LinearConstraint, csc_matrix)
        except Exception:
            self.backend = False
        return bool(self.backend)

    def cover_matrix(self, columns, stop):
        np, _, _, _, _, csc_matrix = self.backend
        indices, indptr, values = [], [0], []
        for r in columns:
            self.check(stop)
            indices.extend(u - 1 for u in r.nodes)
            indices.append(self.n)
            values.extend([1.] * (len(r.nodes) + 1))
            indptr.append(len(indices))
        return csc_matrix((np.asarray(values), np.asarray(indices, dtype=np.int32),
                           np.asarray(indptr, dtype=np.int32)),
                          shape=(self.n + 1, len(columns)))

    def master_duals(self, stop):
        self.check(stop)
        np, linprog, _, _, _, _ = self.backend
        columns = self.cover_source()
        matrix = self.cover_matrix(columns, stop)
        available = min(stop, self.end) - time.perf_counter()
        if available < .015:
            return
        result = linprog(np.ones(len(columns)), A_ub=-matrix[:self.n],
                         b_ub=-np.ones(self.n), bounds=(0, None), method='highs',
                         options={'time_limit': available * .85, 'presolve': True})
        if result.success:
            values = result.ineqlin.marginals
            self.dual = [0.] + [max(0., -float(x)) for x in values]

    def integer_master(self, stop):
        self.check(stop)
        np, _, milp, Bounds, LinearConstraint, _ = self.backend
        columns = self.cover_source()
        matrix = self.cover_matrix(columns, stop)
        lower = np.ones(self.n + 1)
        lower[-1] = 0
        upper = np.full(self.n + 1, np.inf)
        upper[-1] = len(self.best) - 1
        available = min(stop, self.end) - time.perf_counter()
        if available < .02:
            return
        result = milp(np.ones(len(columns)), integrality=np.ones(len(columns), dtype=np.int32),
                      bounds=Bounds(0, 1), constraints=LinearConstraint(matrix, lower, upper),
                      options={'time_limit': available * .85, 'mip_rel_gap': 0., 'presolve': True})
        if result.x is not None:
            chosen = [k for k, value in enumerate(result.x) if value > .5]
            improved = self.accept_cover(columns, chosen)
            if self.exact_complete and improved and result.status == 0 and len(chosen) == len(self.best):
                self.lower_bound = len(self.best)
        elif self.exact_complete and result.status == 2:
            self.lower_bound = len(self.best)

    def column_phase(self, stop):
        start = time.perf_counter()
        available = min(stop, self.end) - start
        if available <= 0:
            return
        if not self.exact_complete and self.n <= self.exact_size:
            try:
                self.enumerate_routes(start + available * .4)
            except _Stop:
                pass
        if not self.load_backend(stop):
            if not self.exact_complete:
                try:
                    self.price_routes(time.perf_counter() + max(0., stop - time.perf_counter()) * .45)
                except _Stop:
                    pass
            try:
                self.cover_dfs(stop)
            except _Stop:
                pass
            return
        if not self.exact_complete:
            try:
                now = time.perf_counter()
                self.master_duals(now + max(0., stop - now) * .18)
            except _Stop:
                pass
            except Exception:
                self.backend = False
                return
            try:
                now = time.perf_counter()
                self.price_routes(now + max(0., stop - now) * .52)
            except _Stop:
                pass
        try:
            self.integer_master(stop)
        except _Stop:
            pass
        except Exception:
            self.backend = False
            try:
                self.cover_dfs(stop)
            except _Stop:
                pass

    def strengthen_bound(self, stop):
        """Cliques of customers that pairwise cannot share a vehicle give a safe vehicle lower bound."""
        clique = []
        order = sorted(range(1, self.n + 1), key=lambda u: (self.b[u] - self.a[u], -self.q[u]))
        for u in order:
            self.check(stop)
            compatible = False
            for v in clique:
                self.check(stop)
                if self.q[u] + self.q[v] > self.capacity:
                    continue
                uv = max(self.a[v], self.singletons[u - 1].early[1] + self.row(u)[v] + self.s[u])
                vu = max(self.a[u], self.singletons[v - 1].early[1] + self.row(v)[u] + self.s[v])
                if uv <= self.singletons[v - 1].late[1] or vu <= self.singletons[u - 1].late[1]:
                    compatible = True
                    break
            if not compatible:
                clique.append(u)
                self.lower_bound = max(self.lower_bound, len(clique))

    def search(self):
        if self.n < 2 or len(self.best) <= self.lower_bound:
            return
        start = time.perf_counter()
        construction_end = min(self.end, start + (self.end - start) / (9 + self.strength))
        first = True
        while time.perf_counter() < construction_end:
            self.construct(construction_end, first)
            first = False
            if len(self.best) <= self.lower_bound:
                return
        try:
            self.strengthen_bound(min(self.end, time.perf_counter() + self.budget / (30 + self.n)))
        except _Stop:
            pass
        if len(self.best) <= self.lower_bound:
            return
        now = time.perf_counter()
        span = max(.000001, self.budget / (2 + self.strength))
        self.column_phase(min(self.end, now + span * .5))
        if len(self.best) <= self.lower_bound:
            return
        self.check()
        routes, bank = self.eliminate(self.source())
        routes, bank = self.repair(routes, bank, min(self.end, time.perf_counter() + span * .1))
        anchor, anchor_bank = routes[:], set(bank)
        weights = [1.] * 5
        last_progress = epoch = time.perf_counter()
        next_columns = last_progress + span
        min_remove = max(1, int(math.sqrt(self.n / max(1, len(self.best)))))
        max_remove = min(self.n, max(min_remove, int(math.sqrt(self.n) * (1 + self.strength))))
        rng = self.rng
        # The search stops only on the wall clock or a proven lower bound; there is no limit on iterations or restarts.
        while time.perf_counter() < self.end:
            if len(self.best) <= self.lower_bound:
                return
            if not bank:
                self.publish(routes)
                if len(self.best) <= self.lower_bound:
                    return
                routes, bank = self.eliminate(self.source())
                routes, bank = self.repair(routes, bank, min(self.end, time.perf_counter() + span * .1))
                anchor, anchor_bank = routes[:], set(bank)
                last_progress = epoch = time.perf_counter()
                continue
            now = time.perf_counter()
            if now >= next_columns:
                before = len(self.best)
                self.column_phase(min(self.end, now + span * .30))
                next_columns = time.perf_counter() + span
                if len(self.best) < before:
                    routes, bank = self.eliminate(self.source())
                    anchor, anchor_bank = routes[:], set(bank)
                    last_progress = epoch = time.perf_counter()
                continue
            stagnation = min(1., (now - last_progress) / span)
            if now - epoch >= span * (1 + self.strength / 5):
                if rng.random() < .3:
                    self.construct(min(self.end, now + span * .1))
                routes, bank = self.eliminate(self.source())
                routes, bank = self.repair(routes, bank, min(self.end, time.perf_counter() + span * .1))
                anchor, anchor_bank = routes[:], set(bank)
                last_progress = epoch = time.perf_counter()
                for u in range(1, self.n + 1):
                    self.penalty[u] = 1 + .7 * (self.penalty[u] - 1)
                continue
            if rng.random() < .12:
                routes = self.exchange(routes, min(self.end, now + span * .04))
            operator = rng.choices(range(5), weights=weights)[0]
            upper = max(min_remove, int(max_remove * (.2 + .8 * stagnation)))
            count = rng.randint(min_remove, upper)
            base_routes, base_bank = (anchor, anchor_bank) if stagnation > .5 and rng.random() < .35 else (routes, bank)
            trial, waiting = self.ruin(base_routes, base_bank, count, operator)
            slice_end = min(self.end, time.perf_counter() + span * (.10 + .17 * stagnation))
            trial, waiting = self.repair(trial, waiting, slice_end)
            if waiting and time.perf_counter() < slice_end and (len(waiting) <= len(bank) or rng.random() < .25):
                trial, waiting = self.eject(trial, waiting, slice_end)
            self.remember(trial)
            if not waiting:
                self.publish(trial)
                routes, bank = trial, waiting
                weights[operator] += 2
                continue
            old_missing = sum(self.penalty[u] * (1 + self.q[u] / self.capacity) for u in bank)
            new_missing = sum(self.penalty[u] * (1 + self.q[u] / self.capacity) for u in waiting)
            delta_cost = (sum(r.cost for r in trial) - sum(r.cost for r in routes)) / self.unit
            change = new_missing - old_missing + .02 * delta_cost / max(1, len(routes))
            temperature = (.08 + .6 * stagnation) * (1 + self.strength / 8)
            improved = len(waiting) < len(bank)
            if improved or change <= 0 or rng.random() < math.exp(-min(700, change / temperature)):
                routes, bank = trial, waiting
                weights[operator] = .95 * weights[operator] + (.2 if improved else .06)
            else:
                weights[operator] = .95 * weights[operator] + .05
            if len(bank) < len(anchor_bank):
                anchor, anchor_bank = routes[:], set(bank)
                last_progress = time.perf_counter()
            for u in bank:
                self.penalty[u] += .04
        if not bank:
            self.publish(routes)


def _baseline(instance):





    result = _empty("feasible")
    exact = lambda x: Fraction(str(x))
    h0, h1 = map(exact, instance["depot_time_window"])
    cap = exact(instance["vehicle_capacity"])
    base = instance["base_travel_times"]
    for k, q in enumerate(instance["customer_demands"]):
        u = k + 1
        ready, due = map(exact, instance["customer_time_windows"][k])
        service = exact(instance["customer_service_times"][k])
        t = max(ready, h0 + exact(base[0][u]))
        returned = t + max(exact(base[u][0]) + service, h0 - due)
        if exact(q) > cap or t > due or returned > h1:
            return _empty("infeasible")
        result["routes"].append([0, u, 0])
        result["route_departure_times"].append(float(h0))
        result["customer_departure_times"].append(float(t))
        result["route_return_times"].append(float(returned))
        result["customer_loads"].append(q)
    return result

def solve(instance: dict, time_limit_s: float) -> dict:
    """Always keep a complete solution on valid input; infeasibility is reported only when a single-customer feasibility proof fails."""
    started = time.perf_counter()
    fallback = _baseline(instance)
    if fallback['status'] == 'infeasible' or not fallback['routes']:
        return fallback
    search = None
    try:
        budget = max(0., float(time_limit_s))
        if not math.isfinite(budget) or time.perf_counter() - started >= budget:
            return fallback
        search = _Search(instance, budget, started)
        if not search.initialize():
            return fallback
        search.search()
    except Exception:
        # Failure of an optional backend or an interrupted search does not change the status of the complete feasible solution.
        pass
    if search is not None and search.best:
        try:
            return search.answer()
        except Exception:
            pass
    return fallback
