"""Fused vehicle-reduction search driven by exact feasibility
Verifies routes with integer scaling and combines persistent ejection chains, regret repair, local exchanges, elite route-pool recombination and small-neighbourhood subset dynamic programming.
"""

from decimal import Decimal
from fractions import Fraction
from itertools import combinations
import math
import random
import time


def _empty(status):
    return {"status": status, "routes": [], "route_departure_times": [],
            "customer_departure_times": [], "route_return_times": [],
            "customer_loads": []}


def _number(x):
    return x.numerator if x.denominator == 1 else float(x)


def _backup(data):
    """With an unlimited fleet and the effective-time triangle inequality, the single-customer check is a proof of infeasibility."""
    result = _empty("feasible")
    a0, b0 = (Fraction(str(x)) for x in data["depot_time_window"])
    capacity = Fraction(str(data["vehicle_capacity"]))
    matrix = data["base_travel_times"]
    for k, demand in enumerate(data["customer_demands"]):
        u = k + 1
        a, b = (Fraction(str(x)) for x in data["customer_time_windows"][k])
        t = max(a, a0 + Fraction(str(matrix[0][u])))
        inward = max(Fraction(str(matrix[u][0])) +
                     Fraction(str(data["customer_service_times"][k])), a0 - b)
        returned = max(a0, t + inward)
        if Fraction(str(demand)) > capacity or t > b or returned > b0:
            return _empty("infeasible")
        result["routes"].append([0, u, 0])
        result["route_departure_times"].append(_number(a0))
        result["customer_departure_times"].append(_number(t))
        result["route_return_times"].append(_number(returned))
        result["customer_loads"].append(demand)
    return result


def _integers(values, digits=0):
    values = [Decimal(str(x)) for x in values]
    scale = 10 ** max([digits] + [-x.as_tuple().exponent for x in values])
    result = []
    for x in values:
        p, q = x.as_integer_ratio()
        result.append(p * scale // q)
    return result, scale


def _tenths(x):
    if x < 100000000000000:
        return round(x * 10)
    p, q = Decimal(str(x)).as_integer_ratio()
    return p * 10 // q


class _Stop(Exception):
    pass


class _Route:
    __slots__ = ("nodes", "path", "early", "late", "load", "cost", "mask")

    def __init__(self, nodes, early, late, load, cost):
        self.nodes = tuple(nodes)
        self.path = (0,) + self.nodes + (0,)
        self.early, self.late = early, late
        self.load, self.cost = load, cost
        self.mask = 0
        for u in self.nodes:
            self.mask |= 1 << (u - 1)


class _Search:
    def __init__(self, data, budget, start):
        self.n = len(data["customer_demands"])
        self.base = data["base_travel_times"]
        q, self.qscale = _integers([data["vehicle_capacity"]] + data["customer_demands"])
        self.capacity, self.q = q[0], [0] + q[1:]
        values = list(data["depot_time_window"])
        for pair in data["customer_time_windows"]:
            values.extend(pair)
        values.extend(data["customer_service_times"])
        t, self.tscale = _integers(values, 1)
        self.a = [t[0]] + [t[2 + 2 * k] for k in range(self.n)]
        self.b = [t[1]] + [t[3 + 2 * k] for k in range(self.n)]
        self.s = [0] + t[2 + 2 * self.n:]
        self.factor = self.tscale // 10
        self.rows = [None] * (self.n + 1)
        self.budget = budget
        self.end = start + budget - min(.07 * budget, .025 + .00003 * self.n)
        self.strength = math.log1p(budget)
        self.rng = random.Random(917623 + 131 * self.n)
        self.full = (1 << self.n) - 1
        self.penalty = [1.0] * (self.n + 1)
        self.pool = {}
        self.pool_limit = max(self.n, int(self.n * (5 + 2 * self.strength) ** 1.5))
        self.elites = []
        self.elite_limit = max(2, int(2 + self.strength))
        self.exact_size = min(18, max(2, int(7 + 2 * self.strength)))
        self.singletons = []
        for u in range(1, self.n + 1):
            out = _tenths(self.base[0][u]) * self.factor
            back = _tenths(self.base[u][0]) * self.factor
            earliest = max(self.a[u], self.a[0] + out)
            latest = min(self.b[u], self.b[0] - back - self.s[u])
            self.singletons.append(_Route((u,),
                (self.a[0], earliest, max(self.a[0], earliest + back + self.s[u])),
                (min(self.b[0], latest - out), latest, self.b[0]), self.q[u], out + back))
        self.best = self.singletons[:]
        self.unit = max(1, (sum(r.cost for r in self.best) + sum(self.s)) // max(1, self.n))
        self.total = sum(self.q)
        self.lower_bound = (self.total + self.capacity - 1) // self.capacity
        # For customers above any demand threshold, the number per vehicle is strictly limited by capacity.
        for k, demand in enumerate(sorted(self.q[1:], reverse=True), 1):
            per_route = self.capacity // demand
            if per_route:
                self.lower_bound = max(self.lower_bound, (k + per_route - 1) // per_route)
        self.hard = [0.0] + [self.q[u] / self.capacity +
                    self.singletons[u - 1].cost / self.unit +
                    self.unit / (self.unit + self.b[u] - self.a[u])
                    for u in range(1, self.n + 1)]
        self.local_slice = min(budget * .0015,
                               .002 * (1 + self.strength) * math.sqrt(1 + self.n))
        self.last_gain = time.perf_counter()

    def check(self, stop=None):
        if time.perf_counter() >= (self.end if stop is None else min(self.end, stop)):
            raise _Stop()

    def row(self, u):
        row = self.rows[u]
        if row is None:
            self.check()
            row = []
            for k, x in enumerate(self.base[u]):
                if not k % 128:
                    self.check()
                row.append(_tenths(x) * self.factor)
            self.rows[u] = row
        return row

    def make(self, nodes):
        self.check()
        nodes = tuple(nodes)
        path = (0,) + nodes + (0,)
        early = [self.a[0]]
        load = cost = previous = 0
        for p, u in enumerate(path[1:]):
            if not p % 32:
                self.check()
            distance = self.row(previous)[u]
            # When t_i<=b_i, max(a_j,t_i+max(tau+s_i,a_j-b_i))
            # is exactly max(a_j,t_i+tau+s_i); the correction is not omitted and service is not counted twice.
            arrival = max(self.a[u], early[-1] + distance + self.s[previous])
            load += self.q[u]
            if arrival > self.b[u] or load > self.capacity:
                return None
            early.append(arrival)
            cost += distance
            previous = u
        late = [0] * len(path)
        late[-1] = self.b[0]
        for p in range(len(path) - 2, -1, -1):
            u, v = path[p:p + 2]
            late[p] = min(self.b[u], late[p + 1] - self.row(u)[v] - self.s[u])
        # The full-route time recursion and positive-demand prefix sums together imply the pre-forbidden arc conditions.
        return _Route(nodes, early, late, load, cost)

    def insertion(self, u, route, delay=.15, fill=.15):
        self.check()
        if route.load + self.q[u] > self.capacity:
            return None
        du = self.row(u)
        best, position = math.inf, -1
        for p in range(len(route.path) - 1):
            if not p % 32:
                self.check()
            v, w = route.path[p:p + 2]
            dv = self.row(v)
            arrival = max(self.a[u], route.early[p] + dv[u] + self.s[v])
            if arrival > self.b[u]:
                continue
            following = arrival + du[w] + self.s[u]
            if following > route.late[p + 1]:
                continue
            added = (dv[u] + du[w] - dv[w]) / self.unit
            score = added + delay * max(0, following - route.early[p + 1]) / self.unit
            if score < best:
                best, position = score, p
        if position < 0:
            return None
        return best - fill * route.load / self.capacity, position

    def insert_at(self, u, route, position):
        return self.make(route.nodes[:position] + (u,) + route.nodes[position:])

    def remember(self, routes):
        for r in routes:
            if r is None or len(r.nodes) < 2:
                continue
            old = self.pool.get(r.mask)
            if old is None or r.cost < old.cost:
                self.pool[r.mask] = r
        if len(self.pool) > self.pool_limit:
            protected = {r.mask for r in self.best}
            for sol in self.elites:
                protected.update(r.mask for r in sol)
            target = max(len(protected), int(.8 * self.pool_limit))
            for mask in list(self.pool):
                if len(self.pool) <= target:
                    break
                if mask not in protected:
                    del self.pool[mask]

    def consider(self, routes):
        routes = [r for r in routes if r is not None and r.nodes]
        mask = count = 0
        for r in routes:
            if mask & r.mask:
                return False
            mask |= r.mask
            count += len(r.nodes)
        if mask != self.full or count != self.n:
            return False
        self.remember(routes)
        gain = len(routes) < len(self.best)
        if gain:
            self.best = routes[:]
            self.elites = []
            self.last_gain = time.perf_counter()
        if len(routes) == len(self.best):
            signature = frozenset(r.mask for r in routes)
            if all(signature != frozenset(r.mask for r in sol) for sol in self.elites):
                self.elites.append(routes[:])
                if len(self.elites) > self.elite_limit:
                    del self.elites[self.rng.randrange(len(self.elites) - 1)]
        # The published best solution is updated only by vehicle count; distance only affects the search process.
        return gain

    def construct(self, stop, first=False):
        order = list(range(1, self.n + 1))
        mode = 0 if first else self.rng.randrange(5)
        if mode == 0:
            order.sort(key=lambda u: (self.b[u], -self.q[u]))
        elif mode == 1:
            order.sort(key=lambda u: (-self.q[u], self.b[u]))
        elif mode == 2:
            order.sort(key=lambda u: (-self.singletons[u - 1].cost, self.b[u]))
        elif mode == 3:
            order.sort(key=lambda u: (self.a[u], self.b[u]))
        else:
            self.rng.shuffle(order)
        delay = .15 if first else self.rng.random() * .6
        fill = .2 if first else self.rng.random()
        routes, at = [], 0
        try:
            while at < self.n:
                self.check(stop)
                u, best = order[at], None
                for k, r in enumerate(routes):
                    self.check(stop)
                    option = self.insertion(u, r, delay, fill)
                    if option is not None:
                        score, pos = option
                        score += 0 if first else self.rng.uniform(-.15, .15)
                        if best is None or score < best[0]:
                            best = score, k, pos
                if best is None:
                    routes.append(self.singletons[u - 1])
                else:
                    _, k, pos = best
                    added = self.insert_at(u, routes[k], pos)
                    if added is None:
                        routes.append(self.singletons[u - 1])
                    else:
                        routes[k] = added
                at += 1
        except _Stop:
            pass
        routes.extend(self.singletons[u - 1] for u in order[at:])
        self.consider(routes)
        return routes

    def repair(self, routes, bank, stop, opening=False):
        routes, bank = routes[:], set(bank)
        cache = {u: {} for u in bank}
        rng = self.rng
        delay, fill, noise = rng.random() * .55, rng.random() * .7, rng.random() * .2
        depth = 2 + int(rng.random() * min(len(routes), 1 + self.strength / 2))
        use_regret = rng.random() < .9
        try:
            while bank:
                self.check(stop)
                selected, priority = None, -math.inf
                for u in bank:
                    self.check(stop)
                    options, entries = [], cache[u]
                    for k, route in enumerate(routes):
                        self.check(stop)
                        old = entries.get(k)
                        if old is None or old[0] is not route:
                            value = self.insertion(u, route, delay, fill)
                            if value is not None:
                                value = value[0] + rng.uniform(-noise, noise), value[1]
                            entries[k] = route, value
                        else:
                            value = old[1]
                        if value is not None:
                            options.append((value[0], k, value[1]))
                    if not options:
                        continue
                    options.sort()
                    score, k, pos = options[0]
                    value = -score
                    if use_regret:
                        value = sum(x[0] - score for x in options[1:depth])
                        value += 3 * (depth - min(depth, len(options))) - .025 * score
                    value += .25 * self.penalty[u] + .06 * self.hard[u]
                    if value > priority:
                        selected, priority = (u, k, pos), value
                if selected is None:
                    if not opening:
                        break
                    u = max(bank, key=lambda x: self.hard[x] * rng.uniform(.8, 1.2))
                    routes.append(self.singletons[u - 1])
                    bank.remove(u)
                    del cache[u]
                    continue
                u, k, pos = selected
                added = self.insert_at(u, routes[k], pos)
                if added is None:
                    break
                routes[k] = added
                bank.remove(u)
                del cache[u]
        except _Stop:
            pass
        if opening:
            routes.extend(self.singletons[u - 1] for u in bank)
            bank.clear()
        return routes, bank

    def eliminate(self, routes):
        routes = [r for r in routes if r.nodes]
        ranked = sorted(range(len(routes)), key=lambda k:
                        len(routes[k].nodes) + routes[k].load / self.capacity)
        rank = int(self.rng.random() ** (1.3 + .1 * self.strength) * len(ranked))
        victim = ranked[rank]
        return routes[:victim] + routes[victim + 1:], set(routes[victim].nodes)

    def ruin(self, routes, bank, count, mode):
        rng = self.rng
        assigned = [u for r in routes for u in r.nodes]
        count = min(count, len(assigned))
        if not count:
            return routes[:], set(bank)
        removed = set()
        if mode == 0:
            removed.update(rng.sample(assigned, count))
        elif mode in (1, 4):
            seed = rng.choice(tuple(bank)) if bank and mode == 4 else rng.choice(assigned)
            distances = self.row(seed)
            span = max(1, self.b[0] - self.a[0])
            ranked = sorted(assigned, key=lambda u: distances[u] / self.unit +
                    .6 * abs(self.a[u] - self.a[seed]) / span +
                    .3 * abs(self.b[u] - self.b[seed]) / span +
                    .2 * abs(self.q[u] - self.q[seed]) / self.capacity + .3 * rng.random())
            removed.update(ranked[:count])
        elif mode == 2:
            scored = []
            for r in routes:
                for p, u in enumerate(r.nodes):
                    v, w = r.path[p], r.path[p + 2]
                    saving = self.row(v)[u] + self.row(u)[w] - self.row(v)[w]
                    scored.append((saving / self.unit + rng.random(), u))
            scored.sort(reverse=True)
            removed.update(u for _, u in scored[:count])
        else:
            order = list(routes)
            rng.shuffle(order)
            if mode == 3:
                order.sort(key=lambda r: len(r.nodes) * rng.uniform(.5, 1.5))
            for r in order:
                length = min(len(r.nodes), count - len(removed))
                if mode == 5:
                    length = min(length, max(1, int(math.sqrt(count) * rng.uniform(.5, 1.5))))
                if length:
                    start = rng.randrange(len(r.nodes))
                    removed.update((r.nodes + r.nodes)[start:start + length])
                if len(removed) == count:
                    break
            if len(removed) < count:
                rest = [u for u in assigned if u not in removed]
                removed.update(rng.sample(rest, count - len(removed)))
        work = []
        for r in routes:
            if any(u in removed for u in r.nodes):
                r = self.make(u for u in r.nodes if u not in removed)
                if r is None:
                    return routes[:], set(bank)
            work.append(r)
        return work, set(bank) | removed

    def local(self, routes, stop, shake=False):
        """Relocation, swap, reversal and two-route tail exchange on the same feasible route representation."""
        work = routes[:]
        rng = self.rng
        try:
            while work:
                self.check(stop)
                i, j = rng.randrange(len(work)), rng.randrange(len(work))
                x, y = work[i], work[j]
                mode = rng.random()
                if not x.nodes:
                    continue
                if i == j:
                    if len(x.nodes) < 2:
                        continue
                    k, h = sorted(rng.sample(range(len(x.nodes)), 2))
                    if mode < .45:
                        seq = x.nodes[:k] + x.nodes[k:h + 1][::-1] + x.nodes[h + 1:]
                        z = self.make(seq)
                    else:
                        u = x.nodes[k]
                        short = self.make(x.nodes[:k] + x.nodes[k + 1:])
                        ins = self.insertion(u, short) if short is not None else None
                        z = self.insert_at(u, short, ins[1]) if ins is not None else None
                    if z is not None and (z.cost < x.cost or
                            (shake and rng.random() < .1)):
                        work[i] = z
                        self.remember([z])
                    continue
                if mode < .4:
                    k = rng.randrange(len(x.nodes))
                    u = x.nodes[k]
                    ins = self.insertion(u, y, .1, .2)
                    if ins is None:
                        continue
                    vx = x.nodes[:k] + x.nodes[k + 1:]
                    vy = y.nodes[:ins[1]] + (u,) + y.nodes[ins[1]:]
                elif mode < .7:
                    if not y.nodes:
                        continue
                    k, h = rng.randrange(len(x.nodes)), rng.randrange(len(y.nodes))
                    u, v = x.nodes[k], y.nodes[h]
                    if (x.load - self.q[u] + self.q[v] > self.capacity or
                            y.load - self.q[v] + self.q[u] > self.capacity):
                        continue
                    vx = x.nodes[:k] + (v,) + x.nodes[k + 1:]
                    vy = y.nodes[:h] + (u,) + y.nodes[h + 1:]
                else:
                    k, h = rng.randrange(len(x.nodes) + 1), rng.randrange(len(y.nodes) + 1)
                    vx, vy = x.nodes[:k] + y.nodes[h:], y.nodes[:h] + x.nodes[k:]
                rx = self.make(vx)
                if rx is None:
                    continue
                ry = self.make(vy)
                if ry is None:
                    continue
                delta = (rx.cost + ry.cost - x.cost - y.cost) / self.unit
                concentration = ((rx.load / self.capacity) ** 2 + (ry.load / self.capacity) ** 2 -
                                 (x.load / self.capacity) ** 2 - (y.load / self.capacity) ** 2)
                fewer = bool(rx.nodes) + bool(ry.nodes) < bool(x.nodes) + bool(y.nodes)
                if fewer or delta - .25 * concentration < -1e-12 or (shake and rng.random() < .12):
                    work[i], work[j] = rx, ry
                    self.remember([rx, ry])
        except _Stop:
            pass
        return work

    def eject(self, routes, bank, stop):
        """Penalty-guided variable-depth ejection; local exchanges directly free insertion positions for the same customer pool."""
        work, bank = routes[:], set(bank)
        rng, tabu = self.rng, {}
        width = max(2, int(math.sqrt(max(1, self.n)) * (1 + self.strength / 5)))
        depth_limit = max(1, int(1 + self.strength / 1.8))
        try:
            while bank:
                self.check(stop)
                u = max(bank, key=lambda v: self.penalty[v] * rng.uniform(.8, 1.2))
                direct = []
                for k, r in enumerate(work):
                    self.check(stop)
                    option = self.insertion(u, r, .1, .1)
                    if option is not None:
                        direct.append((option[0], k, option[1]))
                if direct:
                    _, k, pos = min(direct)
                    added = self.insert_at(u, work[k], pos)
                    if added is not None:
                        work[k] = added
                        bank.remove(u)
                        continue
                self.penalty[u] += .5
                now = time.perf_counter()
                order = list(range(len(work)))
                rng.shuffle(order)
                du = self.row(u)
                order.sort(key=lambda k: min((du[v] for v in work[k].nodes), default=0))
                best, best_penalty = None, math.inf
                for k in order:
                    self.check(stop)
                    r = work[k]
                    available = [v for v in r.nodes if tabu.get(v, 0) <= now]
                    near = sorted(available, key=lambda v: self.penalty[v] + .12 * du[v] / self.unit)
                    heavy = sorted(available, key=lambda v: -self.q[v])
                    choices = list(dict.fromkeys(near[:width] + heavy[:depth_limit]))
                    for depth in range(1, min(depth_limit, len(available)) + 1):
                        candidates = available if depth == 1 else choices
                        for victims in combinations(candidates, depth):
                            self.check(stop)
                            penalty = sum(self.penalty[v] for v in victims)
                            if penalty > best_penalty + .1:
                                continue
                            if r.load + self.q[u] - sum(self.q[v] for v in victims) > self.capacity:
                                continue
                            removed = set(victims)
                            short = self.make(v for v in r.nodes if v not in removed)
                            if short is None:
                                continue
                            ins = self.insertion(u, short, .1, 0)
                            if ins is None:
                                continue
                            added = self.insert_at(u, short, ins[1])
                            score = penalty + .04 * ins[0] + rng.random() * .08
                            if added is not None and (best is None or score < best[0]):
                                best = score, k, victims, added
                                best_penalty = penalty
                if best is None:
                    work = self.local(work, min(stop, time.perf_counter() + self.local_slice), True)
                    continue
                _, k, victims, added = best
                work[k] = added
                bank.remove(u)
                bank.update(victims)
                tabu[u] = time.perf_counter() + max(0, stop - time.perf_counter()) / (3 + self.strength)
                self.remember([added])
        except _Stop:
            pass
        return work, bank

    def recombine(self, stop):
        """Set covering over the route pool; after shortcutting overlapping customers, routes must pass the full feasibility check again."""
        self.remember(self.best)
        pool = list(self.pool.values()) + self.singletons
        by_customer = [[] for _ in range(self.n)]
        for k, r in enumerate(pool):
            self.check(stop)
            for u in r.nodes:
                by_customer[u - 1].append(k)
        order = sorted(range(self.n), key=lambda u: len(by_customer[u]))
        target = len(self.best) - 1
        stack, seen = [(self.full, self.total, ())], {}
        width = max(2, int((1 + self.strength) * math.sqrt(len(pool))))
        while stack:
            self.check(stop)
            left, demand, chosen = stack.pop()
            if not left:
                used, solution = 0, []
                for k in chosen:
                    r = self.make(u for u in pool[k].nodes if not (used >> (u - 1)) & 1)
                    if r is None:
                        break
                    if r.nodes:
                        solution.append(r)
                        used |= r.mask
                if used == self.full and self.consider(solution):
                    return
                continue
            slots = target - len(chosen)
            if slots <= 0 or demand > slots * self.capacity:
                continue
            if seen.get(left, target + 1) <= len(chosen):
                continue
            seen[left] = len(chosen)
            if len(seen) > self.pool_limit * (2 + self.strength):
                seen.clear()
            pivot = next(u for u in order if (left >> u) & 1)
            options = {}
            for k in by_customer[pivot]:
                self.check(stop)
                covered = left & pool[k].mask
                if covered in options:
                    continue
                saved = sum(self.q[u] for u in pool[k].nodes if (left >> (u - 1)) & 1)
                if demand - saved <= (slots - 1) * self.capacity:
                    options[covered] = (covered.bit_count(), saved, k, covered)
            ranked = sorted(options.values(), reverse=True)
            for _, saved, k, covered in reversed(ranked[:width]):
                stack.append((left ^ covered, demand - saved, chosen + (k,)))

    def partition(self, nodes, target, stop):
        """Share earliest-arrival labels on a small customer subset and find a partition into at most target routes."""
        nodes = tuple(nodes)
        n, size = len(nodes), 1 << len(nodes)
        if not n:
            return [], True
        if target < 1 or sum(self.q[u] for u in nodes) > target * self.capacity:
            return None, True
        labels, loads, feasible = [None] * size, [0] * size, {}
        for i, u in enumerate(nodes):
            labels[1 << i] = {i: (self.singletons[u - 1].early[1], -1)}
        complete = True
        enum_end = time.perf_counter() + max(0, stop - time.perf_counter()) * .72
        try:
            for mask in range(1, size):
                self.check(enum_end)
                low = mask & -mask
                loads[mask] = loads[mask ^ low] + self.q[nodes[low.bit_length() - 1]]
                row = labels[mask]
                if not row:
                    continue
                last = min(row, key=lambda i: row[i][0] + self.row(nodes[i])[0] + self.s[nodes[i]])
                feasible[mask] = last
                free = (size - 1) ^ mask
                for i, (t, _) in row.items():
                    self.check(enum_end)
                    u = nodes[i]
                    distances = self.row(u)
                    rest = free
                    while rest:
                        low = rest & -rest
                        rest ^= low
                        j, v = low.bit_length() - 1, nodes[low.bit_length() - 1]
                        if loads[mask] + self.q[v] > self.capacity:
                            continue
                        arrival = max(self.a[v], t + distances[v] + self.s[u])
                        if arrival > self.singletons[v - 1].late[1]:
                            continue
                        nxt = mask | low
                        if labels[nxt] is None:
                            labels[nxt] = {}
                        old = labels[nxt].get(j)
                        if old is None or arrival < old[0]:
                            labels[nxt][j] = arrival, i
        except _Stop:
            complete = False

        def unpack(mask):
            i, path = feasible[mask], []
            while mask:
                path.append(nodes[i])
                previous = labels[mask][i][1]
                mask ^= 1 << i
                i = previous
            return self.make(reversed(path))

        def demand(mask):
            if mask and not loads[mask]:
                bits, value = mask, 0
                while bits:
                    bit = bits & -bits
                    bits ^= bit
                    value += self.q[nodes[bit.bit_length() - 1]]
                loads[mask] = value
            return loads[mask]

        by_node = [[] for _ in nodes]
        failed = set()

        def dfs(rem, slots):
            self.check(stop)
            if not rem:
                return []
            if slots <= 0 or demand(rem) > slots * self.capacity:
                return None
            if rem in feasible:
                return [rem]
            if slots == 1 or (rem, slots) in failed:
                return None
            bits, options = rem, None
            while bits:
                self.check(stop)
                bit = bits & -bits
                bits ^= bit
                choices = [m for m in by_node[bit.bit_length() - 1] if m & rem == m]
                if options is None or len(choices) < len(options):
                    options = choices
            for mask in options:
                tail = dfs(rem ^ mask, slots - 1)
                if tail is not None:
                    return [mask] + tail
            failed.add((rem, slots))
            return None

        try:
            for mask in sorted(feasible, key=int.bit_count, reverse=True):
                self.check(stop)
                bits = mask
                while bits:
                    bit = bits & -bits
                    bits ^= bit
                    by_node[bit.bit_length() - 1].append(mask)
            selected = dfs(size - 1, target)
            if selected is not None:
                solution = [unpack(mask) for mask in selected]
                if all(r is not None for r in solution):
                    self.remember(solution)
                    return solution, complete
            # Even when no partition is found, feed the large covering routes back into the global route pool.
            for mask in sorted(feasible, key=int.bit_count, reverse=True):
                self.check(stop)
                if mask.bit_count() < 2:
                    break
                r = unpack(mask)
                if r is not None:
                    self.remember([r])
        except _Stop:
            return None, False
        return None, complete

    def exact_neighborhood(self, routes, bank, stop):
        if not bank or len(bank) >= self.exact_size:
            return routes, bank
        seed = self.rng.choice(tuple(bank))
        distances = self.row(seed)
        order = sorted(range(len(routes)), key=lambda k:
                min((distances[u] for u in routes[k].nodes), default=0))
        nodes, chosen = list(bank), []
        for k in order:
            if len(nodes) + len(routes[k].nodes) <= self.exact_size:
                nodes.extend(routes[k].nodes)
                chosen.append(k)
        if not chosen:
            return routes, bank
        result, _ = self.partition(nodes, len(chosen), stop)
        if result is None:
            return routes, bank
        removed = set(chosen)
        return [r for k, r in enumerate(routes) if k not in removed] + result, set()

    def search(self):
        if self.n < 2 or len(self.best) <= self.lower_bound:
            return
        self.check()
        first_end = min(self.end, time.perf_counter() + self.budget * .055)
        self.construct(first_end, True)
        while time.perf_counter() < first_end and len(self.best) > self.lower_bound:
            if self.rng.random() < .5:
                self.construct(first_end)
            else:
                trial, _ = self.repair([], range(1, self.n + 1), first_end, True)
                self.consider(trial)
        if len(self.best) <= self.lower_bound:
            return
        if self.n <= self.exact_size:
            result, proven = self.partition(range(1, self.n + 1), len(self.best) - 1,
                            min(self.end, time.perf_counter() + self.budget * .24))
            if result is not None:
                self.consider(result)
            elif proven:
                self.lower_bound = len(self.best)
                return
        routes, bank = self.eliminate(self.best)
        weights = [1.0] * 6
        restart_span = max(.002, self.budget / (3 + self.strength))
        progress = time.perf_counter()
        next_restart = progress + restart_span
        pool_span = self.budget / (4 + 2 * self.strength)
        next_pool = progress + pool_span
        minimum = max(1, int(math.sqrt(self.n / max(1, len(self.best)))))
        maximum = min(self.n, max(minimum, int(math.sqrt(self.n) * (1 + self.strength))))
        # No limit on iterations or restarts; only the wall clock and a proven lower bound can stop the search.
        while time.perf_counter() < self.end:
            if len(self.best) <= self.lower_bound:
                return
            now = time.perf_counter()
            if now >= next_pool:
                old = len(self.best)
                try:
                    self.recombine(min(self.end, now + pool_span * .16))
                except _Stop:
                    pass
                next_pool = time.perf_counter() + pool_span
                if len(self.best) < old:
                    routes, bank = self.eliminate(self.best)
                    progress = time.perf_counter()
            now = time.perf_counter()
            if now >= next_restart:
                # First repair the persistent vehicle-reduction state into an elite candidate, then continue reducing from the shared elite.
                completed, _ = self.repair(routes, bank,
                    min(self.end, now + restart_span * .055), True)
                completed = self.local(completed,
                    min(self.end, time.perf_counter() + self.local_slice), True)
                self.consider(completed)
                if self.rng.random() < .3:
                    self.construct(min(self.end, time.perf_counter() + restart_span * .08))
                base = self.rng.choice(self.elites) if self.elites else self.best
                routes, bank = self.eliminate(base)
                for u in range(1, self.n + 1):
                    self.penalty[u] = 1 + .65 * (self.penalty[u] - 1)
                progress = time.perf_counter()
                next_restart = progress + restart_span
            now = time.perf_counter()
            stagnation = min(1.0, (now - progress) / restart_span)
            allowance = self.budget * (.004 + .01 * stagnation) * math.sqrt(1 + self.n / 100)
            stop = min(self.end, now + min((self.end - now) * .28, allowance))
            mode = self.rng.choices(range(6), weights=weights)[0]
            # Ejection and local exchanges act directly on the retained pool of unassigned customers; no separate algorithm is started.
            if self.rng.random() < .24:
                trial = self.local(routes, min(stop, now + self.local_slice), stagnation > .3)
                waiting = set(bank)
            else:
                upper = max(minimum, int(maximum * (.2 + .8 * stagnation)))
                trial, waiting = self.ruin(routes, bank, self.rng.randint(minimum, upper), mode)
            trial, waiting = self.repair(trial, waiting,
                                        min(stop, time.perf_counter() + allowance * .4))
            if waiting and time.perf_counter() < stop:
                if (stagnation > .25 and len(waiting) < self.exact_size and
                        self.rng.random() < .12):
                    trial, waiting = self.exact_neighborhood(trial, waiting, stop)
                else:
                    trial, waiting = self.eject(trial, waiting, stop)
            self.remember(trial)
            if not waiting:
                gain = self.consider(trial)
                weights[mode] = .94 * weights[mode] + (2 if gain else .15)
                if len(self.best) <= self.lower_bound:
                    return
                routes, bank = self.eliminate(self.best)
                progress = time.perf_counter()
                next_restart = progress + restart_span
                continue
            old_missing = sum(self.penalty[u] * (1 + self.q[u] / self.capacity) for u in bank)
            new_missing = sum(self.penalty[u] * (1 + self.q[u] / self.capacity) for u in waiting)
            distance_change = (sum(r.cost for r in trial) - sum(r.cost for r in routes)) / self.unit
            change = new_missing - old_missing + .025 * distance_change / max(1, len(routes))
            temperature = (.1 + .65 * stagnation) * (1 + self.strength / 8)
            improved = len(waiting) < len(bank)
            accepted = improved or change <= 0 or self.rng.random() < math.exp(-min(700, change / temperature))
            if accepted:
                routes, bank = trial, waiting
                if improved:
                    progress = time.perf_counter()
            weights[mode] = .96 * weights[mode] + (.18 if improved else .06 if accepted else .03)
            for u in bank:
                self.penalty[u] += .05

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


def solve(instance: dict, time_limit_s: float) -> dict:
    """Always keep a complete solution on valid input; a timeout or failed optional search is not passed off as a proof of infeasibility."""
    started = time.perf_counter()
    backup = _empty("feasible")
    search = None
    try:
        backup = _backup(instance)
        if backup["status"] == "infeasible" or not instance["customer_demands"]:
            return backup
        budget = max(0.0, float(time_limit_s))
        if not math.isfinite(budget) or time.perf_counter() - started >= budget:
            return backup
        search = _Search(instance, budget, started)
        search.search()
    except Exception:
        pass
    if search is not None:
        try:
            return search.answer()
        except Exception:
            pass
    return backup
