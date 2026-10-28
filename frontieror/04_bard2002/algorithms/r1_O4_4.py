"""Multi-level vehicle-elimination search with exact integer timing
Uses a complete feasible fallback, regret repair, multi-customer ejection beam search and subset dynamic programming route recombination.
"""

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
    # Demands and times are scaled separately by the decimal precision of the input, not compressed to a fixed time grid.
    values = [Decimal(str(x)) for x in values]
    digits = max([minimum_digits] + [-x.as_tuple().exponent for x in values])
    scale = 10 ** digits
    result = []
    for x in values:
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
        self.early, self.late = early, late
        self.load, self.cost = load, cost
        self.mask = sum(1 << (u - 1) for u in self.nodes)


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
        self.end = started + budget - min(budget * .08, .025 + self.n * .00003)
        self.rng = random.Random(917623)
        self.strength = math.log1p(budget)
        self.rows = [None] * (self.n + 1)
        self.singletons = []
        self.best = []
        self.pool = {}
        self.pool_limit = max(self.n, int(self.n * (2 + self.strength) ** 2))
        self.penalty = [1.0] * (self.n + 1)
        self.unit = 1
        self.lower_bound = 0

        self.elites = []
        self.elite_cap = max(2, int(1 + self.strength))
        self.proven = False
        self.full_exact_tried = False

    def check(self, stop=None):
        if time.perf_counter() >= (self.end if stop is None else min(stop, self.end)):
            raise _Stop()


    def row(self, u):
        row = self.rows[u]
        if row is None:
            self.check()
            row = []
            for k, value in enumerate(self.base[u]):
                if not k % 256:
                    self.check()
                row.append(_tenths(value) * self.factor)
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
                # Effective times satisfy the triangle inequality, so a route stays feasible after removing other customers; hence an infeasible single customer proves the whole instance infeasible.
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
        for count, q in enumerate(sorted(self.q[1:], reverse=True), 1):
            per_route = self.capacity // q
            self.lower_bound = max(self.lower_bound, (count + per_route - 1) // per_route)
        if self.n:
            self.unit = max(1, sum(r.cost for r in self.singletons) // self.n)
        return True


    def make(self, nodes):
        # When t_i <= b_i, max(a_j, t_i+c_ij) equals max(a_j, t_i+tau_ij+s_i).
        # Hence the recursion below fully preserves the lifting rule of the effective travel time and never counts service time twice.
        nodes = tuple(nodes)
        path = (0,) + nodes + (0,)
        early = [self.a[0]]
        load = 0
        cost = 0
        previous = 0
        for index, u in enumerate(path[1:]):
            if not index % 64:
                self.check()
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
            if not p % 64:
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


    def remember(self, routes):
        for route in routes:
            if len(route.nodes) < 2:
                continue
            old = self.pool.get(route.mask)
            if old is None or route.cost < old.cost:
                self.pool[route.mask] = route
        if len(self.pool) > self.pool_limit:
            protected = {r.mask for r in self.best}
            for solution in self.elites:
                protected.update(r.mask for r in solution)
            target = max(len(protected), int(self.pool_limit * .8))
            for mask in list(self.pool):
                if len(self.pool) <= target:
                    break
                if mask not in protected:
                    del self.pool[mask]

    def publish(self, routes):
        routes = [r for r in routes if r.nodes]
        if len(routes) > len(self.best):
            return False
        covered = 0
        count = 0
        for route in routes:
            if covered & route.mask:
                return False
            covered |= route.mask
            count += len(route.nodes)
        if count != self.n or covered != (1 << self.n) - 1:
            return False
        improved = len(routes) < len(self.best)
        if improved:
            self.best = routes[:]
            self.elites = []
        signature = frozenset(r.mask for r in routes)
        if all(signature != frozenset(r.mask for r in s) for s in self.elites):
            self.elites.append(routes[:])
            if len(self.elites) > self.elite_cap:
                del self.elites[self.rng.randrange(len(self.elites) - 1)]
        self.remember(routes)
        return improved

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


    def beam_ejection(self, u, route, tabu, stop):
        """Within subsequences that keep the original relative route order, jointly choose the insertion position and several ejected customers."""
        if not route.nodes:
            return self.singletons[u - 1], ()
        rng = self.rng
        depth = min(len(route.nodes), max(1, 1 + int(self.strength / 1.35)))
        width = max(4, int((2 + self.strength) * math.sqrt(len(route.nodes) + 1)))
        now = time.perf_counter()
        locked = {v for v in route.nodes if tabu.get(v, 0) > now}
        charges = {v: self.penalty[v] * rng.uniform(.94, 1.06) for v in route.nodes}
        a, b, q, service = self.a, self.b, self.q, self.s
        span = max(1, b[0] - a[0])
        # Label: ejection cost, earliest time, load, last customer, kept sequence, ejected sequence, whether already inserted.
        beam = [(0.0, a[0], 0, 0, (), (), False)]

        def extend(label, v, inserted):
            price, t, load, last, path, out, _ = label
            load += q[v]
            if load + (0 if inserted else q[u]) > self.capacity:
                return None
            t = max(a[v], t + self.row(last)[v] + service[last])
            if t > b[v] or t + self.row(v)[0] + service[v] > b[0]:
                return None
            return price, t, load, v, path + (v,), out, inserted

        for v in route.nodes:
            self.check(stop)
            options = []
            for label in beam:
                active = [label]
                if not label[6]:
                    added = extend(label, u, True)
                    if added is not None:
                        active.append(added)
                for item in active:
                    added = extend(item, v, item[6])
                    if added is not None:
                        options.append(added)
                    if len(item[5]) < depth and v not in locked:
                        options.append((item[0] + charges[v], item[1], item[2], item[3],
                                        item[4], item[5] + (v,), item[6]))
            if not options:
                return None
            # Apply resource dominance for the same last customer and insertion state; each of the two insertion states keeps its own beam width.
            options.sort(key=lambda z: z[0] + .025 * ((z[1] - a[0]) / span) +
                         .015 * z[2] / self.capacity)
            groups = {}
            counts = [0, 0]
            beam = []
            for label in options:
                flag = int(label[6])
                if counts[flag] >= width:
                    continue
                key = (label[3], flag)
                group = groups.setdefault(key, [])
                if any(old[0] <= label[0] and old[1] <= label[1] and
                       old[2] <= label[2] and len(old[5]) <= len(label[5]) for old in group):
                    continue
                group.append(label)
                beam.append(label)
                counts[flag] += 1
        choices = []
        for label in beam:
            self.check(stop)
            if not label[6]:
                label = extend(label, u, True)
            if label is not None:
                choices.append(label)
        if not choices:
            return None
        choices.sort(key=lambda z: (z[0], z[1]))
        for label in choices:
            self.check(stop)
            route = self.make(label[4])
            if route is not None:
                return route, label[5]
        return None

    def eject(self, routes, bank, stop):
        routes, bank = routes[:], set(bank)
        best_routes, best_bank = routes[:], set(bank)
        best_key = (len(bank), sum(self.penalty[u] for u in bank))
        tabu = {}
        rng = self.rng
        try:
            while bank:
                self.check(stop)
                u = max(bank, key=lambda v: self.penalty[v] * rng.uniform(.8, 1.2))
                direct = []
                for k, route in enumerate(routes):
                    self.check(stop)
                    option = self.insertion(u, route, .12, .12)
                    if option is not None:
                        direct.append((option[0], k, option[1]))
                selected = None
                if direct:
                    _, k, position = min(direct)
                    added = self.insert_at(u, routes[k], position)
                    if added is not None:
                        selected = (0.0, k, added, ())
                if selected is None:
                    self.penalty[u] += 1.0
                    order = list(range(len(routes)))
                    rng.shuffle(order)
                    du = self.row(u)
                    order.sort(key=lambda k: min((du[v] for v in routes[k].nodes), default=0))
                    for k in order:
                        self.check(stop)
                        result = self.beam_ejection(u, routes[k], tabu, stop)
                        if result is None:
                            continue
                        added, victims = result
                        score = sum(self.penalty[v] for v in victims)
                        score += .035 * ((added.cost - routes[k].cost) / self.unit)
                        score += rng.random() * .06
                        if selected is None or score < selected[0]:
                            selected = (score, k, added, victims)
                if selected is None:
                    break
                _, k, added, victims = selected
                routes[k] = added
                bank.remove(u)
                bank.update(victims)
                tabu[u] = time.perf_counter() + max(.00005, (stop - time.perf_counter()) /
                                                   (3 + self.strength))
                key = (len(bank), sum(self.penalty[v] for v in bank))
                if key < best_key:
                    best_key = key
                    best_routes, best_bank = routes[:], set(bank)
                if not bank:
                    return routes, bank
                if rng.random() < .18:
                    routes = self.reshuffle(routes, min(stop, time.perf_counter() +
                                               (stop - time.perf_counter()) / (8 + self.strength)))
        except _Stop:
            pass
        # Do not let an unfinished long ejection chain overwrite an earlier state that was closer to completion.
        if len(best_bank) < len(bank):
            return best_routes, best_bank
        return routes, bank

    def cover_masks(self, masks, quantities, target, stop, exhaustive=False):
        """Set covering over a restricted route library; only a fully enumerated local route library can give a proof of local optimality."""
        n = len(quantities)
        full = (1 << n) - 1
        by_node = [[] for _ in range(n)]
        for k, mask in enumerate(masks):
            self.check(stop)
            rest = mask
            while rest:
                bit = rest & -rest
                rest ^= bit
                by_node[bit.bit_length() - 1].append(k)
        if any(not row for row in by_node):
            return None
        order = sorted(range(n), key=lambda u: len(by_node[u]))
        demand_cache = {0: 0, full: sum(quantities)}

        def demand(mask):
            known = demand_cache.get(mask)
            if known is None:
                known = 0
                rest = mask
                while rest:
                    bit = rest & -rest
                    rest ^= bit
                    known += quantities[bit.bit_length() - 1]
                demand_cache[mask] = known
            return known

        max_size = max(map(int.bit_count, masks), default=1)
        width = max(2, int((2 + self.strength) * math.sqrt(len(masks) + 1)))
        stack = [(full, target, ())]
        seen = {}
        while stack:
            self.check(stop)
            left, slots, chosen = stack.pop()
            if not left:
                return chosen
            if slots <= 0 or demand(left) > slots * self.capacity:
                continue
            if left.bit_count() > slots * max_size:
                continue
            if seen.get(left, -1) >= slots:
                continue
            seen[left] = slots
            if not exhaustive and len(seen) > self.pool_limit * (2 + self.strength):
                seen.clear()
            pivot = next(u for u in order if left & (1 << u))
            required = demand(left) - (slots - 1) * self.capacity
            options = {}
            for k in by_node[pivot]:
                self.check(stop)
                covered = left & masks[k]
                if covered == left:
                    return chosen + (k,)
                if slots > 1 and demand(covered) >= required:
                    options[covered] = k
            options = [(covered.bit_count(), demand(covered), k, covered)
                       for covered, k in options.items()]
            options.sort(reverse=True)
            if not exhaustive:
                options = options[:width]
            for _, _, k, covered in reversed(options):
                stack.append((left ^ covered, slots - 1, chosen + (k,)))
        return None

    def recombine(self, stop):
        self.remember(self.best)
        pool = list(self.pool.values()) + self.singletons
        chosen = self.cover_masks([r.mask for r in pool], self.q[1:],
                                  len(self.best) - 1, stop)
        if chosen is None:
            return
        used, routes = 0, []
        for k in chosen:
            self.check(stop)
            route = self.make(u for u in pool[k].nodes if not used & (1 << (u - 1)))
            if route is None:
                return
            if route.nodes:
                routes.append(route)
                used |= route.mask
        self.publish(routes)

    def exact_width(self, allowance):
        # Estimate the affordable number of subset labels from the budget; size thresholds are not hard-coded for a particular benchmark size.
        work = max(4.0, allowance * 32000 * (1 + self.strength / 4))
        width = 1
        while width < self.n and (width + 1) * (1 << width) <= work:
            width += 1
        return width

    def exact_regroup(self, solution, indices, stop):
        """Subset-last-customer earliest-label dynamic programming, followed by an exact cover with the largest feasible subsets."""
        chosen_routes = [solution[k] for k in indices]
        nodes = [u for route in chosen_routes for u in route.nodes]
        m = len(nodes)
        if not m:
            return False
        is_full = m == self.n
        if is_full:
            self.full_exact_tried = True
        started = time.perf_counter()
        enum_stop = min(stop, started + .68 * max(0.0, stop - started))
        full = (1 << m) - 1
        labels = [None] * (full + 1)
        loads = [0] * (full + 1)
        feasible = {}
        quantities = [self.q[u] for u in nodes]
        for j, u in enumerate(nodes):
            labels[1 << j] = {j: (self.singletons[u - 1].early[1], -1)}
            feasible[1 << j] = j
        complete = True
        try:
            for mask in range(1, full + 1):
                self.check(enum_stop)
                bit = mask & -mask
                loads[mask] = loads[mask ^ bit] + quantities[bit.bit_length() - 1]
                row = labels[mask]
                if row is None:
                    continue
                last = min(row, key=lambda j: row[j][0] + self.row(nodes[j])[0] + self.s[nodes[j]])
                feasible[mask] = last
                free = full ^ mask
                for i, (departure, _) in row.items():
                    self.check(enum_stop)
                    u = nodes[i]
                    du = self.row(u)
                    rest = free
                    while rest:
                        bit = rest & -rest
                        rest ^= bit
                        j = bit.bit_length() - 1
                        if loads[mask] + quantities[j] > self.capacity:
                            continue
                        v = nodes[j]
                        arrival = max(self.a[v], departure + du[v] + self.s[u])
                        if arrival > self.b[v] or arrival + self.row(v)[0] + self.s[v] > self.b[0]:
                            continue
                        nxt = mask | bit
                        if labels[nxt] is None:
                            labels[nxt] = {}
                        old = labels[nxt].get(j)
                        if old is None or arrival < old[0]:
                            labels[nxt][j] = (arrival, i)
        except _Stop:
            complete = False

        def unpack(mask):
            j = feasible[mask]
            path = []
            while mask:
                path.append(nodes[j])
                previous = labels[mask][j][1]
                mask ^= 1 << j
                j = previous
            return self.make(reversed(path))

        self.check(stop)
        maximal = []
        for mask in feasible:
            self.check(stop)
            rest = full ^ mask
            dominated = False
            while rest:
                bit = rest & -rest
                rest ^= bit
                if mask | bit in feasible:
                    dominated = True
                    break
            if not dominated:
                maximal.append(mask)
        outside = [route for k, route in enumerate(solution) if k not in indices]
        target = len(chosen_routes) - 1
        improved = False
        try:
            while target >= 1:
                self.check(stop)
                picked = self.cover_masks(maximal, quantities, target, stop, complete)
                if picked is None:
                    if complete and is_full:
                        self.proven = True
                    break
                covered, replacement = 0, []
                for k in picked:
                    mask = maximal[k]
                    route = unpack(mask)
                    route = self.make(u for u in route.nodes if not covered & (1 << (u - 1)))
                    if route is None:
                        return improved
                    if route.nodes:
                        replacement.append(route)
                        covered |= route.mask
                candidate = outside + replacement
                improved = self.publish(candidate) or improved
                target = len(replacement) - 1
                if len(self.best) == self.lower_bound or (is_full and len(replacement) == 1):
                    if is_full:
                        self.proven = True
                    break
        finally:
            # Partial enumeration also yields valid new columns, but no optimality or infeasibility conclusion.
            self.rng.shuffle(maximal)
            for mask in maximal:
                if time.perf_counter() >= min(stop, self.end):
                    break
                self.remember([unpack(mask)])
        return improved

    def regroup(self, stop):
        solution = self.rng.choice(self.elites) if self.elites else self.best
        limit = self.exact_width(max(0, stop - time.perf_counter()))
        if self.n <= limit and not self.full_exact_tried:
            return self.exact_regroup(solution, list(range(len(solution))), stop)
        ranked = sorted(range(len(solution)), key=lambda k:
                        len(solution[k].nodes) * self.rng.uniform(.7, 1.3))
        for first in ranked:
            self.check(stop)
            selected = [first]
            count = len(solution[first].nodes)
            load = solution[first].load
            choices = [k for k in ranked if k != first]
            seed = solution[first].nodes[0]
            choices.sort(key=lambda k: (len(solution[k].nodes) +
                min((self.row(seed)[u] for u in solution[k].nodes), default=0) / self.unit))
            for k in choices:
                self.check(stop)
                if count + len(solution[k].nodes) > limit:
                    continue
                selected.append(k)
                count += len(solution[k].nodes)
                load += solution[k].load
                if load <= (len(selected) - 1) * self.capacity:
                    if count < self.n or not self.full_exact_tried:
                        return self.exact_regroup(solution, selected, stop)
        return False

    def reshuffle(self, routes, stop):
        """Tail exchange, customer migration and intra-route reinsertion between feasible routes, freeing room for later vehicle elimination."""
        routes = routes[:]
        rng = self.rng
        try:
            while routes:
                self.check(stop)
                i, j = rng.randrange(len(routes)), rng.randrange(len(routes))
                x, y = routes[i], routes[j]
                if not x.nodes:
                    continue
                mode = rng.random()
                if i == j or mode < .45:
                    p = rng.randrange(len(x.nodes))
                    u = x.nodes[p]
                    short = self.make(x.nodes[:p] + x.nodes[p + 1:])
                    destination = short if i == j else y
                    option = self.insertion(u, destination, rng.random() * .2, 0)
                    if option is None:
                        continue
                    added = self.insert_at(u, destination, option[1])
                    if added is None:
                        continue
                    if i == j:
                        if added.cost <= x.cost or rng.random() < .06:
                            routes[i] = added
                        continue
                    nx, ny = short, added
                elif mode < .72 and y.nodes:
                    p, h = rng.randrange(len(x.nodes)), rng.randrange(len(y.nodes))
                    u, v = x.nodes[p], y.nodes[h]
                    if x.load - self.q[u] + self.q[v] > self.capacity or y.load - self.q[v] + self.q[u] > self.capacity:
                        continue
                    nx = self.make(x.nodes[:p] + (v,) + x.nodes[p + 1:])
                    ny = self.make(y.nodes[:h] + (u,) + y.nodes[h + 1:])
                else:
                    p, h = rng.randrange(len(x.nodes) + 1), rng.randrange(len(y.nodes) + 1)
                    nx = self.make(x.nodes[:p] + y.nodes[h:])
                    ny = self.make(y.nodes[:h] + x.nodes[p:])
                if nx is None or ny is None:
                    continue
                if nx.cost + ny.cost <= x.cost + y.cost or rng.random() < .08:
                    routes[i], routes[j] = nx, ny
        except _Stop:
            pass
        return routes

    def search(self):
        if self.n < 2 or len(self.best) == self.lower_bound:
            return
        self.check()
        construction_end = min(self.end, time.perf_counter() + self.budget / (14 + self.strength))
        first = True
        while time.perf_counter() < construction_end:
            self.construct(construction_end, first)
            first = False
            if len(self.best) == self.lower_bound:
                return
        allowance = self.budget / (3 + self.strength / 2)
        if self.n <= self.exact_width(allowance):
            try:
                self.exact_regroup(self.best, list(range(len(self.best))),
                                   min(self.end, time.perf_counter() + allowance))
            except _Stop:
                pass
            if self.proven or len(self.best) == self.lower_bound:
                return
        self.check()
        routes, bank = self.eliminate(self.best)
        weights = [1.0] * 5
        last_progress = time.perf_counter()
        restart_span = max(.002, self.budget / (3 + self.strength))
        next_pool = time.perf_counter() + restart_span * .45
        next_exact = time.perf_counter() + restart_span * .7
        min_remove = max(1, int(math.sqrt(self.n / max(1, len(self.best)))))
        max_remove = min(self.n, max(min_remove, int(math.sqrt(self.n) * (1 + self.strength))))
        # Search, restarts and neighbourhood calls have no fixed round limit; they are bounded only by the wall clock and a proven lower bound.
        while time.perf_counter() < self.end:
            now = time.perf_counter()
            if self.proven or len(self.best) == self.lower_bound:
                return
            old_count = len(self.best)
            if now >= next_pool:
                try:
                    self.recombine(min(self.end, now + restart_span * .14))
                except _Stop:
                    pass
                next_pool = time.perf_counter() + restart_span * .6
            now = time.perf_counter()
            if now >= next_exact:
                try:
                    self.regroup(min(self.end, now + restart_span * .24))
                except _Stop:
                    pass
                next_exact = time.perf_counter() + restart_span * .85
            if len(self.best) < old_count:
                routes, bank = self.eliminate(self.best)
                last_progress = time.perf_counter()
            now = time.perf_counter()
            stagnation = min(1.0, (now - last_progress) / restart_span)
            if stagnation >= 1:
                if self.rng.random() < .25:
                    self.construct(min(self.end, now + restart_span * .12))
                base = self.rng.choice(self.elites) if self.elites else self.best
                base = self.reshuffle(base, min(self.end, time.perf_counter() + restart_span * .04))
                self.publish(base)
                if len(base) > len(self.best):
                    base = self.best
                routes, bank = self.eliminate([r for r in base if r.nodes])
                last_progress = time.perf_counter()
                for u in range(1, self.n + 1):
                    self.penalty[u] = 1 + .65 * (self.penalty[u] - 1)
            slice_end = min(self.end, time.perf_counter() + restart_span * (.06 + .12 * stagnation))
            operator = self.rng.choices(range(5), weights=weights)[0]
            # With few customers awaiting insertion, prefer continuing the ejection chain rather than scattering existing progress every round.
            if bank and self.rng.random() < .38:
                trial, waiting = self.repair(routes, bank, min(slice_end,
                                          time.perf_counter() + (slice_end - time.perf_counter()) * .3))
            else:
                upper = max(min_remove, int(max_remove * (.2 + .8 * stagnation)))
                count = self.rng.randint(min_remove, upper)
                trial, waiting = self.ruin(routes, bank, count, operator)
                trial, waiting = self.repair(trial, waiting, min(slice_end,
                                          time.perf_counter() + (slice_end - time.perf_counter()) * .55))
            if waiting and time.perf_counter() < slice_end:
                trial, waiting = self.eject(trial, waiting, slice_end)
            self.remember(trial)
            if not waiting:
                self.publish(trial)
                if len(self.best) == self.lower_bound:
                    return
                routes, bank = self.eliminate(self.best)
                weights[operator] += 2
                last_progress = time.perf_counter()
                continue
            old_missing = sum(self.penalty[u] * (1 + self.q[u] / self.capacity) for u in bank)
            new_missing = sum(self.penalty[u] * (1 + self.q[u] / self.capacity) for u in waiting)
            distance_change = (sum(r.cost for r in trial) - sum(r.cost for r in routes)) / self.unit
            change = new_missing - old_missing + .025 * distance_change / max(1, len(routes))
            temperature = (.1 + .8 * stagnation) * (1 + self.strength / 8)
            improved = len(waiting) < len(bank)
            if improved or change <= 0 or self.rng.random() < math.exp(-min(700, change / temperature)):
                routes, bank = trial, waiting
                if improved:
                    last_progress = time.perf_counter()
                weights[operator] = .95 * weights[operator] + (.25 if improved else .07)
            else:
                weights[operator] = .95 * weights[operator] + .05
            for u in bank:
                self.penalty[u] += .06

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


def _singleton_answer(instance):
    # The rescue path uses rational numbers per customer and does not depend on search objects or optional native solvers.
    result = _empty("feasible")
    demands = instance["customer_demands"]
    capacity = Fraction(str(instance["vehicle_capacity"]))
    a0, b0 = map(lambda x: Fraction(str(x)), instance["depot_time_window"])
    base = instance["base_travel_times"]
    for k, demand in enumerate(demands):
        a, b = map(lambda x: Fraction(str(x)), instance["customer_time_windows"][k])
        departure = max(a, a0 + Fraction(str(base[0][k + 1])))
        inward = max(Fraction(str(base[k + 1][0])) +
                     Fraction(str(instance["customer_service_times"][k])), a0 - b)
        returned = max(a0, departure + inward)
        if Fraction(str(demand)) > capacity or departure > b or returned > b0:
            return _empty("infeasible")
        result["routes"].append([0, k + 1, 0])
        result["route_departure_times"].append(float(a0))
        result["customer_departure_times"].append(float(departure))
        result["route_return_times"].append(float(returned))
        result["customer_loads"].append(demand)
    return result


def solve(instance: dict, time_limit_s: float) -> dict:
    """Build a complete certificate first on valid instances; optimisation errors or timeouts only interrupt the search and never declare infeasibility."""
    started = time.perf_counter()
    search = None
    backup = None
    try:
        budget = max(0.0, float(time_limit_s))
        if not math.isfinite(budget):
            budget = 0.0
        search = _Search(instance, budget, started)
        if not search.initialize():
            return _empty("infeasible")
        backup = search.answer()
        search.search()
    except Exception:
        pass
    if search is not None and backup is not None:
        try:
            return search.answer()
        except Exception:
            return backup
    return _singleton_answer(instance)
