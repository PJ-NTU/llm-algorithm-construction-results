"""Adaptive route-elimination search with exact decimal timing
Uses individual round trips as a fallback and reduces the vehicle count with regret insertion, remove-and-reinsert, guided ejection, subset route enumeration and integer route-pool recombination.
"""

import math
import random
import time
from decimal import Decimal


def _scaled(values, minimum_places=0):
    parts = [Decimal(str(v)).as_tuple() for v in values]
    places = max([minimum_places] + [-p.exponent for p in parts])
    scale = 10 ** places
    result = []
    for p in parts:
        value = 0
        for digit in p.digits:
            value = value * 10 + digit
        result.append((-value if p.sign else value) * 10 ** (p.exponent + places))
    return result, scale


def _empty(status):
    return {"status": status, "routes": [], "route_departure_times": [],
            "customer_departure_times": [], "route_return_times": [],
            "customer_loads": []}


class _Route:
    __slots__ = ("seq", "load", "early", "late", "end", "mask")

    def __init__(self, seq, load, early, late, end, mask):
        self.seq, self.load, self.early = seq, load, early
        self.late, self.end, self.mask = late, end, mask


class _Search:
    def __init__(self, q, capacity, a, b, s, scale, qscale, base,
                 singleton, fallback, start, budget):
        self.q, self.Q, self.a, self.b, self.s = q, capacity, a, b, s
        self.scale, self.qscale, self.base = scale, qscale, base
        self.n = len(q) - 1
        self.clock = time.perf_counter
        self.start, self.budget = start, budget
        self.deadline = start + max(0.0, budget - min(0.3, budget * 0.025))
        self.strength = math.log1p(budget)
        self.rng = random.Random(72931)
        self.answer = fallback
        self.best = None
        self.single = singleton
        self.pool = {}
        self.pool_limit = max(self.n * 4, int(self.n * (12 + 8 * self.strength)))
        self.lower = max(1, (sum(q) + capacity - 1) // capacity)
        self.full_mask = (1 << self.n) - 1
        self.c = []
        self.norm = max(scale, max(r.end - a[0] for r in singleton))
        self.milp_tools = None
        self.milp_unavailable = False

    def make(self, seq):
        seq = tuple(seq)
        load, previous, t, mask = 0, 0, self.a[0], 0
        early = [t]
        for u in seq:
            load += self.q[u]
            if load > self.Q:
                return None
            t = max(self.a[u], t + self.c[previous][u])
            if t > self.b[u]:
                return None
            early.append(t)
            previous = u
            mask |= 1 << (u - 1)
        end = t + self.c[previous][0] if seq else self.a[0]
        if end > self.b[0]:
            return None
        late = [self.b[0]] * (len(seq) + 1)
        following = 0
        for k in range(len(seq) - 1, -1, -1):
            u = seq[k]
            late[k] = min(self.b[u], late[k + 1] - self.c[u][following])
            following = u
        return _Route(seq, load, tuple(early), tuple(late), end, mask)

    def save(self, routes):
        routes = [r for r in routes if r.seq]
        if self.best is not None and len(routes) >= len(self.best):
            return False
        seen = 0
        for r in routes:
            if seen & r.mask:
                return False
            seen |= r.mask
        if seen != self.full_mask:
            return False
        result = _empty("feasible")
        departures, loads = [0.0] * self.n, [0.0] * self.n
        for r in routes:
            result["routes"].append([0] + list(r.seq) + [0])
            result["route_departure_times"].append(self.a[0] / self.scale)
            result["route_return_times"].append(r.end / self.scale)
            load = 0
            for k, u in enumerate(r.seq):
                departures[u - 1] = r.early[k + 1] / self.scale
                load += self.q[u]
                loads[u - 1] = load / self.qscale
        result["customer_departure_times"] = departures
        result["customer_loads"] = loads
        self.best = routes[:]
        self.answer = result
        return True

    def collect(self, routes):
        for r in routes:
            if r.seq and r.mask not in self.pool:
                self.pool[r.mask] = r
        if len(self.pool) > self.pool_limit * 2:
            protected = {r.mask: r for r in self.single + self.best}
            items = [(k, r) for k, r in self.pool.items() if k not in protected]
            self.rng.shuffle(items)
            # Keep distinct coverage sets; route cost is not a secondary objective of the solution.
            items.sort(key=lambda item: -len(item[1].seq))
            keep = max(0, self.pool_limit - len(protected))
            # Keep some long routes and some random routes so that not only nearly identical large sets remain.
            half = keep // 2
            chosen = items[:half]
            rest = items[half:]
            self.rng.shuffle(rest)
            protected.update(chosen + rest[:keep - half])
            self.pool = protected

    def insertion(self, route, u, weights):
        if route.load + self.q[u] > self.Q:
            return None
        c, a, b = self.c, self.a, self.b
        seq, early, late = route.seq, route.early, route.late
        best = None
        prev = 0
        for pos in range(len(seq) + 1):
            nxt = seq[pos] if pos < len(seq) else 0
            t = max(a[u], early[pos] + c[prev][u])
            if t <= b[u] and t + c[u][nxt] <= late[pos]:
                delta = c[prev][u] + c[u][nxt] - c[prev][nxt]
                old_t = early[pos + 1] if nxt else route.end
                delay = max(0, t + c[u][nxt] - old_t)
                cost = delta / self.norm + weights[0] * (delay / self.norm)
                cost += weights[1] * ((t - a[u]) / max(self.scale, b[u] - a[u]))
                if best is None or cost < best[0]:
                    best = (cost, pos)
            prev = nxt
        return best

    def inserted(self, route, u, pos):
        return self.make(route.seq[:pos] + (u,) + route.seq[pos:])

    def repair(self, routes, pending, stop):
        routes, pending = routes[:], list(pending)
        cache = [{} for _ in routes]
        weights = (self.rng.uniform(0.05, 1.5), self.rng.uniform(0.0, 0.35))
        order = 2 + int(self.rng.random() * max(1.0, min(3.0, self.strength)))
        stranded = []
        while pending and self.clock() < stop:
            chosen, chosen_key, impossible = None, None, []
            for u in pending:
                if self.clock() >= stop:
                    return routes, pending + stranded
                options = []
                for ri, r in enumerate(routes):
                    if u not in cache[ri]:
                        cache[ri][u] = self.insertion(r, u, weights)
                    option = cache[ri][u]
                    if option is not None:
                        # The load bias is only used to generate candidates; it does not change the vehicle-count objective.
                        value = option[0] - 0.10 * (r.load / self.Q)
                        options.append((value, ri, option[1]))
                if not options:
                    impossible.append(u)
                    continue
                options.sort()
                first = options[0]
                regret = sum(z[0] - first[0] for z in options[1:order])
                key = (min(order, len(options)),
                       -regret - 0.05 * (self.q[u] / self.Q),
                       first[0] + self.rng.random() * 1e-7)
                if chosen_key is None or key < chosen_key:
                    chosen_key, chosen = key, (u, first[1], first[2])
            if impossible:
                excluded = set(impossible)
                stranded.extend(impossible)
                pending = [u for u in pending if u not in excluded]
            if chosen is None:
                break
            u, ri, pos = chosen
            new = self.inserted(routes[ri], u, pos)
            if new is None:
                return routes, pending + stranded
            routes[ri] = new
            cache[ri] = {}
            pending.remove(u)
        return routes, pending + stranded

    def construct(self, stop, mode):
        pending = list(range(1, self.n + 1))
        done = []
        weights = (self.rng.uniform(0.05, 0.9), self.rng.uniform(0.0, 0.25))
        radial = self.rng.uniform(0.6, 1.7)
        while pending and self.clock() < stop:
            if mode % 3 == 0:
                seed = min(pending, key=lambda u: (self.b[u], -self.c[0][u]))
            elif mode % 3 == 1:
                seed = max(pending, key=lambda u: self.c[0][u])
            else:
                seed = max(pending, key=lambda u: self.q[u] * self.rng.uniform(0.7, 1.3))
            route = self.single[seed - 1]
            pending.remove(seed)
            while pending and self.clock() < stop:
                chosen, key = None, float("inf")
                for u in pending:
                    if self.clock() >= stop:
                        break
                    option = self.insertion(route, u, weights)
                    if option is not None:
                        value = option[0] - radial * (self.c[0][u] / self.norm)
                        value -= 0.25 * (self.q[u] / self.Q)
                        if value < key:
                            key, chosen = value, (u, option[1])
                if chosen is None:
                    break
                u, pos = chosen
                new = self.inserted(route, u, pos)
                if new is None:
                    break
                route = new
                pending.remove(u)
            done.append(route)
            self.save(done + [self.single[u - 1] for u in pending])
        done += [self.single[u - 1] for u in pending]
        self.save(done)
        self.collect(done)
        return done

    def destroy(self, routes, pending, fraction):
        present = [u for r in routes for u in r.seq]
        if not present:
            return routes[:], pending[:]
        count = min(len(present), max(1, int(fraction * len(present))))
        mode = self.rng.random()
        anchor = self.rng.choice(pending if pending else present)
        if mode < 0.57:
            jitter = self.rng.uniform(0.04, 0.45)
            def related(u):
                return ((self.c[anchor][u] + self.c[u][anchor]) / self.norm
                        + 0.30 * (abs(self.a[anchor] - self.a[u]) / self.norm)
                        + 0.20 * (abs(self.q[anchor] - self.q[u]) / self.Q)
                        + jitter * self.rng.random())
            present.sort(key=related)
        elif mode < 0.79:
            costs = {}
            for r in routes:
                ext = (0,) + r.seq + (0,)
                for p, u in enumerate(r.seq, 1):
                    costs[u] = ((self.c[ext[p - 1]][u] + self.c[u][ext[p + 1]]
                                 - self.c[ext[p - 1]][ext[p + 1]]) / self.norm
                                + self.rng.random() * 0.2)
            present.sort(key=lambda u: -costs[u])
        elif mode < 0.89:
            order = routes[:]
            self.rng.shuffle(order)
            present = [u for r in order for u in r.seq]
        else:
            self.rng.shuffle(present)
        removed = set(present[:count])
        new_routes = []
        removed_mask = sum(1 << (u - 1) for u in removed)
        for r in routes:
            new_routes.append(self.make([u for u in r.seq if u not in removed])
                              if r.mask & removed_mask else r)
        return new_routes, pending + list(removed)

    def eject(self, routes, pending, penalties, tabu, stop, depth):
        if not pending:
            return routes, pending
        u = max(pending, key=lambda x: penalties[x] * self.rng.uniform(0.8, 1.2))
        penalties[u] += 1.0
        indices = list(range(len(routes)))
        self.rng.shuffle(indices)
        indices.sort(key=lambda k: min((self.c[u][v] + self.c[v][u]
                                       for v in routes[k].seq), default=0))
        indices = indices[:min(len(indices), 2 + int(self.strength))]
        best, best_key = None, float("inf")
        now = self.clock()
        weights = (self.rng.random(), 0.0)
        for ri in indices:
            r = routes[ri]
            seq = r.seq
            positions = list(range(len(seq)))
            self.rng.shuffle(positions)
            positions.sort(key=lambda p: self.c[u][seq[p]] + self.c[seq[p]][u])
            width = min(len(seq), 2 + int((1 + self.strength) * math.sqrt(len(seq))))
            positions = positions[:width]
            patterns = [(p,) for p in positions]
            for d in range(2, min(depth, len(seq)) + 1):
                patterns += [tuple(range(p, p + d)) for p in positions if p + d <= len(seq)]
                if len(positions) >= d:
                    patterns += [tuple(sorted(self.rng.sample(positions, d))) for _ in positions]
            visited = set()
            for pattern in patterns:
                if self.clock() >= stop:
                    break
                if pattern in visited:
                    continue
                visited.add(pattern)
                out = [seq[p] for p in pattern]
                key = sum(penalties[v] + (2.0 if tabu.get(v, 0.0) > now else 0.0)
                          for v in out)
                key += self.rng.random() * 0.25
                if key >= best_key or r.load - sum(self.q[v] for v in out) + self.q[u] > self.Q:
                    continue
                remove = set(pattern)
                reduced = self.make([v for p, v in enumerate(seq) if p not in remove])
                if reduced is None:
                    continue
                option = self.insertion(reduced, u, weights)
                if option is None:
                    continue
                new = self.inserted(reduced, u, option[1])
                if new is not None:
                    best_key, best = key, (ri, new, out)
            if self.clock() >= stop:
                break
        if best is None:
            return routes, pending
        ri, new, out = best
        result = routes[:]
        result[ri] = new
        tabu[u] = now + max(0.001, self.budget / (30 + self.n))
        return result, [v for v in pending if v != u] + out

    def enumerate_routes(self, nodes, stop):
        """Keep only the earliest-departure label for each visited set and last customer."""
        reps, layer = {}, {}
        for p, u in enumerate(nodes):
            layer[(1 << p, p)] = (self.single[u - 1].early[1], (u,), self.q[u])
        all_bits = (1 << len(nodes)) - 1
        limit = max(self.pool_limit, int((1 + self.strength) * self.n * self.n))
        while layer and self.clock() < stop:
            following = {}
            for (mask, last), (t, path, load) in layer.items():
                if self.clock() >= stop:
                    break
                if mask not in reps:
                    reps[mask] = path
                bits = all_bits ^ mask
                row = self.c[nodes[last]]
                while bits:
                    bit = bits & -bits
                    bits -= bit
                    j = bit.bit_length() - 1
                    u = nodes[j]
                    new_load = load + self.q[u]
                    if new_load > self.Q:
                        continue
                    end = max(self.a[u], t + row[u])
                    if end > self.b[u] or end + self.c[u][0] > self.b[0]:
                        continue
                    key = (mask | bit, j)
                    previous = following.get(key)
                    if previous is None or end < previous[0]:
                        following[key] = (end, path + (u,), new_load)
                if len(following) > limit:
                    break
            layer = following
        for path in reps.values():
            if self.clock() >= self.deadline:
                break
            r = self.make(path)
            if r is not None:
                self.pool.setdefault(r.mask, r)

    def recombine(self, stop):
        if self.milp_unavailable or self.clock() >= stop or len(self.best) <= self.lower:
            return False
        try:
            if self.milp_tools is None:
                import numpy as np
                from scipy.optimize import milp, Bounds, LinearConstraint
                from scipy.sparse import coo_matrix
                self.milp_tools = (np, milp, Bounds, LinearConstraint, coo_matrix)
            np, milp, Bounds, LinearConstraint, coo_matrix = self.milp_tools
            self.collect(self.best + self.single)
            columns = list(self.pool.values())
            row_indices, col_indices = [], []
            for j, r in enumerate(columns):
                row_indices.extend(u - 1 for u in r.seq)
                col_indices.extend([j] * len(r.seq))
            m = len(columns)
            row_indices.extend([self.n] * m)
            col_indices.extend(range(m))
            matrix = coo_matrix((np.ones(len(row_indices)),
                                 (np.asarray(row_indices, dtype=np.int32),
                                  np.asarray(col_indices, dtype=np.int32))),
                                shape=(self.n + 1, m)).tocsc()
            lower = np.ones(self.n + 1)
            upper = np.ones(self.n + 1)
            lower[-1], upper[-1] = 0, len(self.best) - 1
            remaining = min(stop, self.deadline) - self.clock()
            if remaining <= max(0.02, self.budget * 0.001):
                return False
            result = milp(np.ones(m), integrality=np.ones(m), bounds=Bounds(0, 1),
                          constraints=LinearConstraint(matrix, lower, upper),
                          options={"time_limit": remaining * 0.85,
                                   "mip_rel_gap": 0.0, "disp": False})
            if result.x is not None:
                selected = [r for r, x in zip(columns, result.x) if x > 0.5]
                return self.save(selected)
        except Exception:
            self.milp_unavailable = True
        return False

    def restart(self):
        routes = self.best[:]
        # Small routes are easy to eliminate; the random bias lets the search escape repeated failures on the same route.
        weights = [(len(r.seq) + 0.5 * r.load / self.Q)
                   * self.rng.uniform(0.35, 1.8) for r in routes]
        index = min(range(len(routes)), key=lambda i: weights[i])
        pending = list(routes.pop(index).seq)
        return routes, pending

    def run(self):
        # Validity of the base matrix is an input guarantee; distances are not repaired and customer coordinates are not read.
        ten_scale = self.scale // 10
        for i, row in enumerate(self.base):
            if self.clock() >= self.deadline:
                return
            values, factor = _scaled(row, 1)
            travel = [v * ten_scale // (factor // 10) for v in values]
            if i:
                travel = [max(v + self.s[i], self.a[j] - self.b[i]) if i != j else 0
                          for j, v in enumerate(travel)]
            self.c.append(travel)
        self.single = [self.make((i,)) for i in range(1, self.n + 1)]
        if any(r is None for r in self.single):
            return
        self.save(self.single)
        self.collect(self.single)
        warm_end = min(self.deadline, self.clock() + self.budget * 0.07)
        self.construct(warm_end, 0)
        if len(self.best) <= self.lower or self.clock() >= self.deadline:
            return
        # Small-set enumeration adds route columns; set size and time quota grow with the given time limit.
        enum_size = min(self.n, max(2, int(math.log2(max(1.0, self.budget) * 1800))))
        if self.n <= enum_size:
            enum_end = min(self.deadline, self.clock() + self.budget * 0.12)
            self.enumerate_routes(list(range(1, self.n + 1)), enum_end)
        if self.budget >= 2 and self.clock() < self.deadline - self.budget * 0.20:
            self.recombine(min(self.deadline, self.clock() + self.budget * 0.045))
        if len(self.best) <= self.lower:
            return
        routes, pending = self.restart()
        penalties = [1.0] * (self.n + 1)
        tabu = {}
        epoch = max(0.005, self.budget / (4 + math.sqrt(self.n)))
        restart_at = self.clock() + epoch
        pool_interval = max(0.01, self.budget / (3 + self.strength))
        next_pool = self.clock() + pool_interval
        next_construct = self.clock() + 1.6 * pool_interval
        depth = max(1, min(self.n, 1 + int(self.strength / 1.2)))
        min_pending = len(pending)
        last_progress = self.clock()
        while self.clock() < self.deadline:
            now = self.clock()
            if len(self.best) <= self.lower:
                return
            if now >= next_pool and self.deadline - now > self.budget * 0.06:
                old_count = len(self.best)
                # Large instances only enumerate customers on a group of neighbouring routes, avoiding exponential dependence on the total number of customers.
                if self.rng.random() < 0.5 and self.n > enum_size:
                    anchor = self.rng.randrange(1, self.n + 1)
                    nodes = sorted(range(1, self.n + 1),
                                   key=lambda u: self.c[anchor][u] + self.c[u][anchor])[:enum_size]
                    self.enumerate_routes(nodes, min(self.deadline, now + self.budget * 0.025))
                self.recombine(min(self.deadline, self.clock() + self.budget * 0.045))
                next_pool = self.clock() + pool_interval
                if len(self.best) < old_count:
                    routes, pending = self.restart()
                    restart_at = self.clock() + epoch
                    min_pending, last_progress = len(pending), self.clock()
                continue
            if now >= next_construct and self.deadline - now > self.budget * 0.05:
                self.construct(min(self.deadline, now + self.budget * 0.035),
                               self.rng.randrange(3))
                routes, pending = self.restart()
                restart_at = self.clock() + epoch
                min_pending, last_progress = len(pending), self.clock()
                next_construct = self.clock() + 1.6 * pool_interval
                continue
            if now >= restart_at:
                routes, pending = self.restart()
                if self.rng.random() < 0.5:
                    routes, pending = self.destroy(routes, pending,
                                                   self.rng.uniform(0.04, 0.12))
                restart_at = now + epoch
                min_pending, last_progress = len(pending), now
                penalties = [1.0 + (v - 1.0) * 0.5 for v in penalties]
            operation_end = min(self.deadline, self.clock() + epoch * 0.30)
            if self.rng.random() < 0.45 and pending:
                candidate, missing = self.eject(routes, pending, penalties, tabu,
                                                operation_end, depth)
            else:
                stagnation = min(1.0, (self.clock() - last_progress) / epoch)
                maximum = min(0.55, 0.10 + 0.035 * self.strength + 0.20 * stagnation)
                fraction = self.rng.uniform(1 / max(1, self.n), maximum)
                candidate, missing = self.destroy(routes, pending, fraction)
            candidate, missing = self.repair(candidate, missing, operation_end)
            self.collect(candidate)
            if not missing:
                self.save(candidate)
                if len(self.best) <= self.lower:
                    return
                routes, pending = self.restart()
                restart_at = self.clock() + epoch
                min_pending, last_progress = len(pending), self.clock()
                continue
            old_value = sum(penalties[u] for u in pending)
            new_value = sum(penalties[u] for u in missing)
            temperature = max(0.15, (1 + min_pending * 0.2)
                              * max(0.05, (restart_at - self.clock()) / epoch))
            if new_value <= old_value or self.rng.random() < math.exp(
                    -min(700.0, (new_value - old_value) / temperature)):
                routes, pending = candidate, missing
            if len(pending) < min_pending:
                min_pending, last_progress = len(pending), self.clock()
            for u in pending:
                penalties[u] += 0.05


def solve(instance: dict, time_limit_s: float) -> dict:
    start = time.perf_counter()
    n = len(instance["customer_demands"])
    if n == 0:
        return _empty("feasible")
    # Use integer arithmetic for finite decimal inputs; do not assume service times and windows lie on a one-decimal grid.
    amounts, qscale = _scaled(instance["customer_demands"] + [instance["vehicle_capacity"]])
    q, capacity = [0] + amounts[:-1], amounts[-1]
    windows = instance["customer_time_windows"]
    temporal = (list(instance["depot_time_window"])
                + [x for window in windows for x in window]
                + list(instance["customer_service_times"]))
    values, scale = _scaled(temporal, 1)
    a = [values[0]] + [values[2 + 2 * k] for k in range(n)]
    b = [values[1]] + [values[3 + 2 * k] for k in range(n)]
    s = [0] + values[2 + 2 * n:]
    base = instance["base_travel_times"]
    direct, factor = _scaled([base[0][i] for i in range(1, n + 1)], 1)
    direct = [v * (scale // 10) // (factor // 10) for v in direct]
    fallback = _empty("feasible")
    singletons = []
    for u in range(1, n + 1):
        t = max(a[u], a[0] + direct[u - 1])
        back = max(direct[u - 1] + s[u], a[0] - b[u])
        end = t + back
        if q[u] > capacity or t > b[u] or end > b[0]:
            # The triangle inequality makes individual round trips a necessary condition; with unlimited vehicles it is also sufficient.
            return _empty("infeasible")
        singletons.append(_Route((u,), q[u], (a[0], t),
                                  (min(b[u], b[0] - back), b[0]),
                                  end, 1 << (u - 1)))
        fallback["routes"].append([0, u, 0])
        fallback["route_departure_times"].append(a[0] / scale)
        fallback["route_return_times"].append(end / scale)
        fallback["customer_departure_times"].append(t / scale)
        fallback["customer_loads"].append(q[u] / qscale)
    if n == 1:
        return fallback
    search = None
    try:
        budget = max(0.0, float(time_limit_s))
        if not math.isfinite(budget) or time.perf_counter() >= start + budget:
            return fallback
        search = _Search(q, capacity, a, b, s, scale, qscale, base,
                         singletons, fallback, start, budget)
        search.run()
    except Exception:
        # If a search component is interrupted or an optional solver is unavailable, fall back only to a fully verified feasible solution.
        pass
    return search.answer if search is not None else fallback
