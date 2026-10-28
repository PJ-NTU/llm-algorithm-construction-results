"""Hybrid subset dynamic programming and route-elimination search
Checks feasibility on an integer time scale and reduces the vehicle count with adaptive destroy-and-repair, ejection chains, tail exchange and pausable exact subset re-sequencing.
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
    # Integerise times and demands separately according to the precision of the raw data; do not round service times or windows to one decimal place.
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
        self.end = started + budget - min(budget * .08, .01 + self.n * .00001)
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
        self.elite_limit = max(2, int((1 + self.strength) * math.sqrt(max(1, self.n))))
        # This limits the memory of a single dynamic program, not the number of search rounds.
        state_space = min(8000000, int(50000 * (1 + self.strength) ** 2))
        self.block_width = 1
        while ((self.block_width + 1) * (1 << (self.block_width + 1)) <= state_space):
            self.block_width += 1
        self.block_stop = self.end
        self.exact_done = False
        self.block_failed = set()

    def check(self, stop=None):
        if time.perf_counter() >= (self.end if stop is None else min(stop, self.end)):
            raise _Stop()

    def row(self, u):
        row = self.rows[u]
        if row is None:
            self.check()
            # Build distance rows on demand; the single-customer fallback only needs the edges to and from the depot.
            if self.small_ints:
                try:
                    row = array('q', (_tenths(x) * self.factor for x in self.base[u]))
                except OverflowError:
                    row = [_tenths(x) * self.factor for x in self.base[u]]
            else:
                row = [_tenths(x) * self.factor for x in self.base[u]]
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
                # The effective-time triangle inequality guarantees feasibility after other customers are removed.
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
        for u in path[1:]:
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
        # When t_i <= b_i, max(a_j, t_i + max(tau+s_i, a_j-b_i))
        # equals max(a_j, t_i+tau+s_i), so taking the maximum for the ready time fully preserves the lifting rule.
        for p, (v, w) in enumerate(zip(route.path, route.path[1:])):
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

    def remember(self, routes):
        # Move the most recently used column to the end; the route pool can still absorb new partitions once full.
        for r in routes:
            if len(r.nodes) < 2:
                continue
            old = self.pool.pop(r.mask, None)
            self.pool[r.mask] = r if old is None or r.cost < old.cost else old
        while len(self.pool) > self.pool_limit:
            self.pool.pop(next(iter(self.pool)))

    def publish(self, routes):
        routes = [r for r in routes if r.nodes]
        if len(routes) > len(self.best):
            return False
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
            self.elites = []
        # Partitions with the same vehicle count are only used as restart seeds; they do not change the single optimisation objective.
        signature = tuple(sorted(r.mask for r in routes))
        cost = sum(r.cost for r in routes)
        found = False
        for k, (old_cost, old_signature, old_routes) in enumerate(self.elites):
            if old_signature == signature:
                if cost < old_cost:
                    self.elites[k] = (cost, signature, routes[:])
                found = True
                break
        if not found:
            self.elites.append((cost, signature, routes[:]))
        if len(self.elites) > self.elite_limit:
            self.elites.sort(key=lambda x: x[0])
            # Keep compact partitions while giving the remaining partitions a random chance to survive.
            self.elites.pop(self.rng.randrange(max(1, len(self.elites) // 2), len(self.elites)))
        self.remember(routes)
        return improved

    def restart_solution(self):
        if not self.elites or self.rng.random() < .35:
            return self.best
        return self.rng.choice(self.elites)[2]

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
        # If construction is interrupted, cover the remaining customers with single-customer routes.
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
                    # Routes that only approximately satisfy hard constraints are not accepted during search either.
                    break
                routes[k] = added
                bank.remove(u)
                del cache[u]
        except _Stop:
            pass
        return routes, bank

    def eliminate(self, routes):
        # Prefer eliminating routes with few customers and low load, while keeping a selection probability for other routes.
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
        """Time-limited one- and two-customer ejection chains; intermediate routes always satisfy hard constraints."""
        routes, bank = routes[:], set(bank)
        rng = self.rng
        initial_count = len(bank)
        tabu = {}
        width = max(2, int(math.sqrt(max(1, self.n)) * (1 + self.strength / 3)))
        pair_depth = self.strength >= math.log(3)
        try:
            while bank:
                self.check(stop)
                u = max(bank, key=lambda x: self.penalty[x] * rng.uniform(.7, 1.3))
                direct = []
                for k, r in enumerate(routes):
                    self.check(stop)
                    option = self.insertion(u, r, .1, .1)
                    if option is not None:
                        direct.append((option[0], k, option[1]))
                if direct:
                    _, k, pos = min(direct)
                    added = self.insert_at(u, routes[k], pos)
                    if added is not None:
                        routes[k] = added
                        bank.remove(u)
                        if len(bank) < initial_count:
                            return routes, bank
                        continue
                self.penalty[u] += .2
                candidates = []
                now = time.perf_counter()
                for k, r in enumerate(routes):
                    for v in r.nodes:
                        if tabu.get(v, 0) > now:
                            continue
                        if r.load - self.q[v] + self.q[u] > self.capacity:
                            continue
                        candidates.append((self.penalty[v] + self.row(u)[v] / self.unit
                                           + rng.random() * .3, k, v))
                candidates.sort()
                best = None
                for _, k, v in candidates:
                    self.check(stop)
                    r = routes[k]
                    reduced = self.make(x for x in r.nodes if x != v)
                    if reduced is None:
                        continue
                    option = self.insertion(u, reduced, .1, 0)
                    if option is not None:
                        score = self.penalty[v] + .03 * option[0] + rng.random() * .25
                        if best is None or score < best[0]:
                            added = self.insert_at(u, reduced, option[1])
                            if added is not None:
                                best = score, k, (v,), added
                if best is None and pair_depth:
                    # Neighbourhood width grows with the number of customers and the given time budget.
                    choices = sorted(range(len(routes)), key=lambda k:
                        min((self.row(u)[v] for v in routes[k].nodes), default=0))
                    for k in choices:
                        r = routes[k]
                        victims = sorted(r.nodes, key=lambda v:
                            self.penalty[v] + self.row(u)[v] / self.unit)[:width]
                        for p, v in enumerate(victims):
                            if tabu.get(v, 0) > now:
                                continue
                            for w in victims[p + 1:]:
                                self.check(stop)
                                if tabu.get(w, 0) > now:
                                    continue
                                if r.load - self.q[v] - self.q[w] + self.q[u] > self.capacity:
                                    continue
                                reduced = self.make(x for x in r.nodes if x != v and x != w)
                                if reduced is None:
                                    continue
                                option = self.insertion(u, reduced, .1, 0)
                                if option is not None:
                                    score = self.penalty[v] + self.penalty[w] + .03 * option[0]
                                    if best is None or score < best[0]:
                                        added = self.insert_at(u, reduced, option[1])
                                        if added is not None:
                                            best = score, k, (v, w), added
                if best is None:
                    break
                _, k, victims, added = best
                routes[k] = added
                bank.remove(u)
                bank.update(victims)
                tabu[u] = time.perf_counter() + max(.0001, (stop - time.perf_counter()) / (2 + self.strength))
        except _Stop:
            pass
        return routes, bank

    def recombine(self, stop):
        """Time-limited set covering over the discovered route pool; overlapping customers can be removed by the triangle inequality.

        The route pool does not contain all feasible routes, so exhausting it is not a proof of global optimality.
        """
        self.remember(self.best)
        pool = list(self.pool.values()) + self.singletons
        by_customer = [[] for _ in range(self.n)]
        for k, r in enumerate(pool):
            self.check(stop)
            for u in r.nodes:
                by_customer[u - 1].append(k)
        order = sorted(range(self.n), key=lambda u: len(by_customer[u]))
        target = len(self.best) - 1
        full = (1 << self.n) - 1
        total = sum(self.q)
        stack = [(full, total, ())]
        seen = {}
        width = max(2, int((1 + self.strength) * math.sqrt(max(1, len(pool)))))
        while stack:
            self.check(stop)
            left, demand, chosen = stack.pop()
            if not left:
                used = 0
                routes = []
                for k in chosen:
                    r = self.make(u for u in pool[k].nodes if not (used >> (u - 1)) & 1)
                    if r is None:
                        break
                    if r.nodes:
                        routes.append(r)
                        used |= r.mask
                if used == full and self.publish(routes):
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
            options = []
            for k in by_customer[pivot]:
                covered = left & pool[k].mask
                saved = sum(self.q[u] for u in pool[k].nodes if (left >> (u - 1)) & 1)
                options.append((covered.bit_count(), saved, k, covered))
            options.sort(reverse=True)
            for _, saved, k, covered in reversed(options[:width]):
                stack.append((left ^ covered, demand - saved, chosen + (k,)))

    def exchange(self, routes, bank, stop):
        """Jointly exchange the tail segments of two routes, avoiding blocking of single moves by full-load constraints."""
        if len(routes) < 2:
            return routes[:], set(bank)
        rng = self.rng
        waiting = sorted(bank, key=lambda u: -self.penalty[u])
        if not waiting:
            return routes[:], set(bank)
        u = waiting[0]
        order = list(range(len(routes)))
        rng.shuffle(order)
        order.sort(key=lambda k: min((self.row(u)[v] for v in routes[k].nodes), default=0))
        prefixes = []
        for r in routes:
            values = [0]
            for v in r.nodes:
                values.append(values[-1] + self.q[v])
            prefixes.append(values)
        chosen = None
        score_best = math.inf
        try:
            for ix, k in enumerate(order):
                a = routes[k]
                cuts_a = list(range(len(a.nodes) + 1))
                rng.shuffle(cuts_a)
                for h in order[ix + 1:]:
                    b = routes[h]
                    cuts_b = list(range(len(b.nodes) + 1))
                    rng.shuffle(cuts_b)
                    for p in cuts_a:
                        for q in cuts_b:
                            self.check(stop)
                            if (p == 0 and q == 0) or (p == len(a.nodes) and q == len(b.nodes)):
                                continue
                            qa = prefixes[k][p] + b.load - prefixes[h][q]
                            qb = prefixes[h][q] + a.load - prefixes[k][p]
                            if qa > self.capacity or qb > self.capacity:
                                continue
                            v, w = a.path[p], b.path[q + 1]
                            x, y = b.path[q], a.path[p + 1]
                            if a.early[p] + self.row(v)[w] + self.s[v] > b.late[q + 1]:
                                continue
                            if b.early[q] + self.row(x)[y] + self.s[x] > a.late[p + 1]:
                                continue
                            ra = self.make(a.nodes[:p] + b.nodes[q:])
                            rb = self.make(b.nodes[:q] + a.nodes[p:])
                            if ra is None or rb is None:
                                continue
                            # First check whether pending customers can be absorbed directly, rather than only reducing distance.
                            options = []
                            for z in waiting[:max(1, int(self.strength))]:
                                for dest, route in ((k, ra), (h, rb)):
                                    option = self.insertion(z, route, .1, .1)
                                    if option is not None:
                                        options.append((-self.penalty[z], option[0], z, dest, option[1]))
                            if options:
                                _, _, z, dest, pos = min(options)
                                trial = routes[:]
                                trial[k], trial[h] = ra, rb
                                added = self.insert_at(z, trial[dest], pos)
                                if added is not None:
                                    trial[dest] = added
                                    self.remember((ra, rb, added))
                                    return trial, set(bank) - {z}
                            score = (ra.cost + rb.cost - a.cost - b.cost) / self.unit
                            score += rng.random() * .08
                            if score < score_best:
                                score_best = score
                                chosen = k, h, ra, rb
        except _Stop:
            pass
        trial = routes[:]
        if chosen is not None:
            k, h, ra, rb = chosen
            trial[k], trial[h] = ra, rb
            self.remember((ra, rb))
        return trial, set(bank)

    def choose_block(self, routes, bank):
        """Re-sequence small instances as a whole; for large instances re-sequence pending customers together with several neighbouring routes."""
        if self.n <= self.block_width and not self.exact_done:
            return self.block_solver(self.best[:], [], (), True)
        if not bank or len(bank) >= self.block_width:
            return None
        extra = tuple(bank)
        total_extra = sum(self.q[u] for u in extra)
        seed = max(extra, key=lambda u: self.penalty[u])
        candidates = list(range(len(routes)))
        self.rng.shuffle(candidates)
        candidates.sort(key=lambda k: min((self.row(seed)[u] for u in routes[k].nodes), default=0))
        # The neighbourhood size is not hard-coded for a particular instance size; the time budget determines the affordable number of states.
        for first in candidates:
            self.check()
            picked = []
            size = len(extra)
            demand = total_extra
            for k in [first] + [h for h in candidates if h != first]:
                if not routes[k].nodes or size + len(routes[k].nodes) > self.block_width:
                    continue
                picked.append(k)
                size += len(routes[k].nodes)
                demand += routes[k].load
                if demand <= len(picked) * self.capacity:
                    selected = [routes[h] for h in picked]
                    local_mask = sum(1 << (u - 1) for r in selected for u in r.nodes)
                    local_mask |= sum(1 << (u - 1) for u in extra)
                    if (local_mask, len(picked)) in self.block_failed:
                        continue
                    outside = [r for h, r in enumerate(routes) if h not in picked]
                    return self.block_solver(selected, outside, extra, False)
        return None

    def block_solver(self, selected, outside, extra, whole):
        """Pausable earliest-arrival label dynamic programming, followed by an exact partition of this local customer set."""
        nodes = tuple(u for r in selected for u in r.nodes) + tuple(extra)
        m = len(nodes)
        if not m:
            return
        full = (1 << m) - 1
        count = full + 1
        inf = self.b[0] + 1
        size = count * m
        if -(1 << 63) <= inf < (1 << 63):
            earliest = array('q', [inf]) * size
        else:
            earliest = [inf] * size
        parent = bytearray(size)
        ending = bytearray([255]) * count
        demands = [self.q[u] for u in nodes]
        ready = [self.a[u] for u in nodes]
        due = [self.b[u] for u in nodes]
        back = [self.row(u)[0] + self.s[u] for u in nodes]
        travel = [[self.row(u)[v] + self.s[u] for v in nodes] for u in nodes]
        loads = [0] * count
        for j, u in enumerate(nodes):
            mask = 1 << j
            earliest[mask * m + j] = self.singletons[u - 1].early[1]
            ending[mask] = j
        max_card = 1
        max_load = max(demands)

        def decode(mask):
            last = ending[mask]
            sequence = []
            while mask:
                sequence.append(nodes[last])
                predecessor = parent[mask * m + last] - 1
                mask ^= 1 << last
                last = predecessor
            return self.make(reversed(sequence))

        for mask in range(1, count):
            self.check()
            if time.perf_counter() >= self.block_stop:
                yield None
            bit = mask & -mask
            j = bit.bit_length() - 1
            load = loads[mask ^ bit] + demands[j]
            loads[mask] = load
            if ending[mask] == 255:
                continue
            max_card = max(max_card, mask.bit_count())
            max_load = max(max_load, load)
            if mask == full:
                route = decode(mask)
                if route is not None:
                    self.publish(outside + [route])
                    if whole:
                        self.lower_bound = 1
                        self.exact_done = True
                return
            # Keep only the earliest label for each visited set and last customer; the load is uniquely determined by the set.
            labels = []
            left = mask
            offset = mask * m
            while left:
                bit = left & -left
                h = bit.bit_length() - 1
                t = earliest[offset + h]
                if t != inf:
                    labels.append((h, t))
                left ^= bit
            remaining = full ^ mask
            while remaining:
                bit = remaining & -remaining
                remaining ^= bit
                j = bit.bit_length() - 1
                if load + demands[j] > self.capacity:
                    continue
                bound = min(due[j], self.b[0] - back[j])
                tbest, prev = inf, -1
                for h, t in labels:
                    arrival = t + travel[h][j]
                    if arrival < tbest:
                        tbest, prev = arrival, h
                tbest = max(ready[j], tbest)
                if prev < 0 or tbest > bound:
                    continue
                larger = mask | bit
                earliest[larger * m + j] = tbest
                parent[larger * m + j] = prev + 1
                ending[larger] = j

        # All feasible subsets have been generated. Solve an exact non-overlapping cover of the subsets rather than greedy bin packing.
        pair_counts = [sum(ending[(1 << j) | (1 << h)] != 255
                           for h in range(m) if h != j) for j in range(m)]
        pivot_order = sorted(range(m), key=lambda j: (pair_counts[j], -demands[j]))
        failed = set()

        def cover(left, slots):
            self.check()
            if time.perf_counter() >= self.block_stop:
                yield None
            if not left:
                return []
            if slots <= 0 or loads[left] > slots * max_load:
                return None
            if left.bit_count() > slots * max_card:
                return None
            if ending[left] != 255:
                return [left]
            if slots == 1 or (left, slots) in failed:
                return None
            pivot = next(j for j in pivot_order if left & (1 << j))
            bit = 1 << pivot
            minimum = loads[left] - (slots - 1) * max_load
            options = []
            subset = left
            while subset:
                self.check()
                if time.perf_counter() >= self.block_stop:
                    yield None
                if subset & bit and ending[subset] != 255 and loads[subset] >= minimum:
                    rest = left ^ subset
                    if slots == 2:
                        if ending[rest] != 255:
                            return [subset, rest]
                    else:
                        options.append(subset)
                subset = (subset - 1) & left
            options.sort(key=lambda mask: (loads[mask], mask.bit_count()), reverse=True)
            for subset in options:
                result = yield from cover(left ^ subset, slots - 1)
                if result is not None:
                    return [subset] + result
            failed.add((left, slots))
            return None

        lower = max((loads[full] + self.capacity - 1) // self.capacity,
                    (m + max_card - 1) // max_card,
                    (loads[full] + max_load - 1) // max_load)
        limit = len(selected)
        for target in range(lower, limit + 1):
            if whole and target >= len(self.best):
                self.lower_bound = len(self.best)
                self.exact_done = True
                return
            partition = yield from cover(full, target)
            if partition is not None:
                replacement = [decode(mask) for mask in partition]
                if all(r is not None for r in replacement):
                    self.publish(outside + replacement)
                    if whole:
                        self.lower_bound = len(replacement)
                        self.exact_done = True
                return
        if whole:
            self.lower_bound = len(self.best)
            self.exact_done = True
        else:
            self.block_failed.add((sum(1 << (u - 1) for u in nodes), limit))

    def search(self):
        if self.n < 2 or len(self.best) == self.lower_bound:
            return
        self.check()
        beginning = time.perf_counter()
        construction_end = min(self.end, beginning + (self.end - beginning) * .10)
        self.construct(construction_end, True)
        # Start the whole-instance exact search early for small instances; keep multi-start construction for larger ones.
        if self.n > self.block_width:
            while time.perf_counter() < construction_end:
                self.construct(construction_end)
                if len(self.best) == self.lower_bound:
                    return
        if len(self.best) == self.lower_bound:
            return
        routes, bank = self.eliminate(self.restart_solution())
        weights = [1.0] * 6
        last_progress = time.perf_counter()
        restart_span = max(.005, self.budget / (2 + self.strength))
        next_pool = time.perf_counter() + restart_span * .7
        next_block = time.perf_counter()
        worker = None
        min_remove = max(1, int(math.sqrt(self.n / max(1, len(self.best)))))
        max_remove = min(self.n, max(min_remove,
                         int(math.sqrt(self.n) * (1 + self.strength))))
        # No iteration limit; the search stops only on the wall clock or a proven objective lower bound.
        while time.perf_counter() < self.end:
            now = time.perf_counter()
            if len(self.best) == self.lower_bound:
                return
            if now >= next_block:
                old_count = len(self.best)
                if worker is None:
                    worker = self.choose_block(routes, bank)
                portion = .30 if self.n <= self.block_width else .12
                self.block_stop = min(self.end, time.perf_counter() + restart_span * portion)
                if worker is not None:
                    try:
                        next(worker)
                    except StopIteration:
                        worker = None
                next_block = time.perf_counter() + restart_span * (.55 - portion)
                if len(self.best) < old_count:
                    routes, bank = self.eliminate(self.restart_solution())
                    last_progress = time.perf_counter()
            now = time.perf_counter()
            if len(self.best) == self.lower_bound:
                return
            if now >= next_pool:
                old_count = len(self.best)
                try:
                    self.recombine(min(self.end, now + restart_span * .12))
                except _Stop:
                    pass
                next_pool = time.perf_counter() + restart_span * .7
                if len(self.best) < old_count:
                    routes, bank = self.eliminate(self.restart_solution())
                    last_progress = time.perf_counter()
            now = time.perf_counter()
            stagnation = min(1.0, (now - last_progress) / restart_span)
            if stagnation >= 1:
                if self.rng.random() < .3:
                    self.construct(min(self.end, now + restart_span * .15))
                if len(self.best) == self.lower_bound:
                    return
                routes, bank = self.eliminate(self.restart_solution())
                last_progress = time.perf_counter()
                for u in range(1, self.n + 1):
                    self.penalty[u] = 1 + .7 * (self.penalty[u] - 1)
            operator = self.rng.choices(range(6), weights=weights)[0]
            slice_end = min(self.end, time.perf_counter() + restart_span * (.12 + .15 * stagnation))
            if operator == 5:
                trial, waiting = self.exchange(routes, bank,
                    min(slice_end, time.perf_counter() + (slice_end - time.perf_counter()) * .45))
            else:
                upper = max(min_remove, int(max_remove * (.25 + .75 * stagnation)))
                count = self.rng.randint(min_remove, upper)
                trial, waiting = self.ruin(routes, bank, count, operator)
            trial, waiting = self.repair(trial, waiting, slice_end)
            if not waiting:
                self.publish(trial)
                if len(self.best) == self.lower_bound:
                    return
                routes, bank = self.eliminate(self.restart_solution())
                weights[operator] += 2
                last_progress = time.perf_counter()
                continue
            if time.perf_counter() < slice_end and (len(waiting) <= len(bank) or self.rng.random() < .25):
                trial, waiting = self.eject(trial, waiting, slice_end)
            self.remember(trial)
            if not waiting:
                self.publish(trial)
                if len(self.best) == self.lower_bound:
                    return
                routes, bank = self.eliminate(self.restart_solution())
                weights[operator] += 2
                last_progress = time.perf_counter()
                continue
            old_missing = sum(self.penalty[u] * (1 + self.q[u] / self.capacity) for u in bank)
            new_missing = sum(self.penalty[u] * (1 + self.q[u] / self.capacity) for u in waiting)
            distance_change = (sum(r.cost for r in trial) - sum(r.cost for r in routes)) / self.unit
            change = new_missing - old_missing + .02 * distance_change / max(1, len(routes))
            temperature = (.08 + .6 * stagnation) * (1 + self.strength / 8)
            improved = len(waiting) < len(bank)
            if improved or change <= 0 or self.rng.random() < math.exp(-min(700, change / temperature)):
                routes, bank = trial, waiting
                if improved:
                    last_progress = time.perf_counter()
                weights[operator] = .95 * weights[operator] + (.2 if improved else .06)
            else:
                weights[operator] = .95 * weights[operator] + .05
            for u in bank:
                self.penalty[u] += .04

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


def _baseline(instance):
    """Complete fallback without search; by the triangle inequality, infeasibility of a single customer proves global infeasibility."""
    result = _empty("feasible")
    n = len(instance["customer_demands"])
    if not n:
        return result
    a0, b0 = (Fraction(str(x)) for x in instance["depot_time_window"])
    capacity = Fraction(str(instance["vehicle_capacity"]))
    base = instance["base_travel_times"]
    for k in range(n):
        q = Fraction(str(instance["customer_demands"][k]))
        a, b = (Fraction(str(x)) for x in instance["customer_time_windows"][k])
        s = Fraction(str(instance["customer_service_times"][k]))
        t = max(a, a0 + Fraction(str(base[0][k + 1])))
        returned = max(a0, t + Fraction(str(base[k + 1][0])) + s)
        if q > capacity or t > b or returned > b0:
            return _empty("infeasible")
        result["routes"].append([0, k + 1, 0])
        result["route_departure_times"].append(float(a0))
        result["route_return_times"].append(float(returned))
        result["customer_departure_times"].append(float(t))
        result["customer_loads"].append(float(q))
    return result


def solve(instance: dict, time_limit_s: float) -> dict:
    """Return a complete solution; an unfinished search is never taken as proof of infeasibility."""
    started = time.perf_counter()
    fallback = _baseline(instance)
    if fallback["status"] == "infeasible" or not fallback["routes"]:
        return fallback
    try:
        budget = max(0.0, float(time_limit_s))
        if not math.isfinite(budget) or time.perf_counter() - started >= budget:
            return fallback
        search = _Search(instance, budget, started)
        if not search.initialize():
            return _empty("infeasible")
    except Exception:
        return fallback
    try:
        search.search()
    except Exception:
        pass
    try:
        return search.answer()
    except Exception:
        return fallback
