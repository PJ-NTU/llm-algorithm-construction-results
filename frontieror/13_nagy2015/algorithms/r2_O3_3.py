"""Large neighbourhood search with load caching and cooperative vehicle splitting
Combines savings construction, combined or split-vehicle regret repair, reinsertion and exchange, and elite route recombination, searching within the wall-clock budget and keeping the historical best feasible solution.
"""
import heapq
import math
import random
import time


class _Route:
    __slots__ = ('v', 'cost', 'delivery', 'pickup', 'loads', 'prefix',
                 'suffix', 'peak', 'low', 'where')

    def __init__(self, visits, engine):
        self.v = visits
        d, p, dist = engine.d, engine.p, engine.dist
        self.delivery = sum(d[i] for i, mask in visits if mask & 1)
        self.pickup = sum(p[i] for i, mask in visits if mask & 2)
        load = self.delivery
        loads = [load]
        cost, prev = 0, 0
        where = {}
        for pos, (i, mask) in enumerate(visits):
            cost += dist[prev][i]
            prev = i
            if mask & 1:
                load -= d[i]
            if mask & 2:
                load += p[i]
            loads.append(load)
            where[i] = pos
        self.cost = cost + dist[prev][0]
        self.loads = loads
        self.where = where
        prefix, high = [], -math.inf
        for load in loads:
            high = max(high, load)
            prefix.append(high)
        self.peak = high
        self.low = min(loads)
        self.prefix = prefix
        suffix = [0.0] * len(loads)
        high = -math.inf
        for j in range(len(loads) - 1, -1, -1):
            high = max(high, loads[j])
            suffix[j] = high
        self.suffix = suffix


class _Task:
    __slots__ = ('options', 'top')

    def __init__(self, depot_cost):
        self.options = {-1: (depot_cost, -1, 0)}
        self.top = []

    def update(self, rid, value, depth):
        affected = any(entry[1] == rid for entry in self.top)
        if value is None:
            self.options.pop(rid, None)
        else:
            entry = (value[0], rid, value[1])
            self.options[rid] = entry
            affected = affected or len(self.top) < depth or entry < self.top[-1]
        if affected:
            old = self.top
            self.top = heapq.nsmallest(depth, self.options.values())
            return old != self.top
        return False


class _Job:
    __slots__ = ('i', 'mask', 'parts', 'jitter', 'plan', 'priority')

    def __init__(self, i, mask, engine):
        self.i, self.mask = i, mask
        self.jitter = 0.8 + 0.4 * engine.rng.random()
        self.parts = {bit: _Task(engine.dist[0][i] + engine.dist[i][0])
                      for bit in ((3, 1, 2) if mask == 3 else (mask,))}
        self.plan, self.priority = (), 0.0

    def refresh(self, depth):
        # Each option completes all services this customer is currently missing in one go.
        bundles = [(op[0], ((self.mask, op),)) for op in self.parts[self.mask].top]
        if self.mask == 3:
            for a in self.parts[1].top:
                for b in self.parts[2].top:
                    if a[1] == b[1]:
                        continue
                    bundles.append((a[0] + b[0], ((1, a), (2, b))))
        bundles.sort(key=lambda x: (x[0], len(x[1])))
        first = bundles[0][0]
        regret = sum(x[0] - first for x in bundles[1:depth])
        if len(bundles) < depth:
            regret += (depth - len(bundles)) * (bundles[-1][0] - first)
        self.plan = bundles[0][1]
        self.priority = self.jitter * regret - 0.03 * first


class _Engine:
    def __init__(self, instance, budget, started):
        self.n = int(instance['n'])
        self.dist = instance['distance_matrix']
        raw_d = instance['delivery_demands']
        raw_p = instance['pickup_demands']
        self.d = ([0.0] + list(raw_d)) if len(raw_d) == self.n else list(raw_d)
        self.p = ([0.0] + list(raw_p)) if len(raw_p) == self.n else list(raw_p)
        self.capacity = float(instance['capacity'])
        self.cap = self.capacity + 1e-9 * max(1.0, self.capacity)
        self.budget = max(0.0, float(budget))
        self.effort = math.log1p(self.budget)
        self.started = started
        reserve = min(self.budget * 0.05, max(0.002, self.n * 0.000012,
                                             0.012 * self.budget))
        self.end = started + max(0.0, self.budget - reserve)
        self.rng = random.Random(731927 + self.n * 101)
        self.near = {}
        self.width = min(max(0, self.n - 1), max(4, int(
            5 + 3 * math.sqrt(self.budget) + 2 * math.log1p(self.n))))
        self.best = [_Route([(i, 3)], self) for i in range(1, self.n + 1)]
        self.best_cost = sum(route.cost for route in self.best)
        self.order_cache = {}
        self.cache_limit = max(1, int(self.n * (5 + math.sqrt(self.budget)) * 8))
        self.exchange_cache = set()
        self.elite = []
        self.elite_limit = max(2, int(2 + self.effort + math.log1p(self.n) / 2))

    def nearby(self, i):
        result = self.near.get(i)
        if result is None:
            row = self.dist[i]
            result = heapq.nsmallest(self.width, (j for j in range(1, self.n + 1)
                                                   if j != i), key=row.__getitem__)
            self.near[i] = result
        return result

    def keep(self, routes):
        value = sum(route.cost for route in routes)
        if value < self.best_cost:
            self.best_cost = value
            self.best = routes[:]
        return value

    def valid_route(self, visits):
        # After recombining two routes, complementary services may merge into one visit, while duplicate services are rejected.
        where, clean = {}, []
        for i, mask in visits:
            if i in where:
                pos = where[i]
                if clean[pos][1] & mask:
                    return None
                clean[pos] = (i, clean[pos][1] | mask)
            else:
                where[i] = len(clean)
                clean.append((i, mask))
        route = _Route(clean, self)
        return route if route.peak <= self.cap else None

    def insertion(self, route, i, mask):
        extra_d = self.d[i] if mask & 1 else 0.0
        extra_p = self.p[i] if mask & 2 else 0.0
        if route.delivery + extra_d > self.cap or route.pickup + extra_p > self.cap:
            return None
        existing = route.where.get(i)
        if existing is not None:
            if route.v[existing][1] & mask:
                return None
            if (route.prefix[existing] + extra_d <= self.cap and
                    route.suffix[existing + 1] + extra_p <= self.cap):
                return (0, -existing - 1)
            return None
        pd, sp = self.cap - extra_d, self.cap - extra_p
        best, position = math.inf, 0
        prev, row, dist = 0, self.dist[i], self.dist
        for k in range(len(route.v) + 1):
            if k % 128 == 0 and time.perf_counter() >= self.end:
                return None
            nxt = route.v[k][0] if k < len(route.v) else 0
            if route.prefix[k] <= pd and route.suffix[k] <= sp:
                delta = dist[prev][i] + row[nxt] - dist[prev][nxt]
                if delta < best:
                    best, position = delta, k
            prev = nxt
        return None if best == math.inf else (best, position)

    def insertion_bundle(self, route, i, mask, deadline):
        if mask != 3:
            return {mask: self.insertion(route, i, mask)}
        # The three service types share one position scan instead of recomputing route loads separately.
        qd, qp, cap = self.d[i], self.p[i], self.cap
        allow_d = route.delivery + qd <= cap
        allow_p = route.pickup + qp <= cap
        if not allow_d and not allow_p:
            return {1: None, 2: None, 3: None}
        existing = route.where.get(i)
        if existing is not None:
            return {bit: self.insertion(route, i, bit) for bit in (1, 2, 3)}
        best_d = best_p = best_both = math.inf
        pos_d = pos_p = pos_both = 0
        pd, pp = cap - qd, cap - qp
        previous, row, dist = 0, self.dist[i], self.dist
        for k in range(len(route.v) + 1):
            if k % 128 == 0 and time.perf_counter() >= deadline:
                break
            nxt = route.v[k][0] if k < len(route.v) else 0
            good_d = allow_d and route.prefix[k] <= pd
            good_p = allow_p and route.suffix[k] <= pp
            if good_d or good_p:
                delta = dist[previous][i] + row[nxt] - dist[previous][nxt]
                if good_d and delta < best_d:
                    best_d, pos_d = delta, k
                if good_p and delta < best_p:
                    best_p, pos_p = delta, k
                if good_d and good_p and delta < best_both:
                    best_both, pos_both = delta, k
            previous = nxt
        return {1: None if best_d == math.inf else (best_d, pos_d),
                2: None if best_p == math.inf else (best_p, pos_p),
                3: None if best_both == math.inf else (best_both, pos_both)}

    def inserted(self, route, i, mask, position):
        visits = route.v[:]
        if position < 0:
            k = -position - 1
            visits[k] = (i, visits[k][1] | mask)
        else:
            visits.insert(position, (i, mask))
        return _Route(visits, self)

    def savings(self, deadline):
        routes = self.best[:]
        owner = [0] + list(range(self.n))
        edges, dist = [], self.dist
        for i in range(1, self.n + 1):
            if time.perf_counter() >= deadline:
                break
            for j in self.nearby(i):
                if i < j:
                    saving = dist[0][i] + dist[0][j] - dist[i][j]
                    if saving > 0:
                        edges.append((-saving, i, j))
        edges.sort()
        for _, i, j in edges:
            if time.perf_counter() >= deadline:
                break
            a, b = owner[i], owner[j]
            if a == b:
                continue
            ra, rb = routes[a], routes[b]
            if i not in (ra.v[0][0], ra.v[-1][0]) or j not in (rb.v[0][0], rb.v[-1][0]):
                continue
            va = ra.v if ra.v[-1][0] == i else ra.v[::-1]
            vb = rb.v if rb.v[0][0] == j else rb.v[::-1]
            peak_a = ra.peak if va is ra.v else ra.delivery + ra.pickup - ra.low
            peak_b = rb.peak if vb is rb.v else rb.delivery + rb.pickup - rb.low
            forward = max(peak_a + rb.delivery, peak_b + ra.pickup)
            low_a = ra.low if va is ra.v else ra.delivery + ra.pickup - ra.peak
            low_b = rb.low if vb is rb.v else rb.delivery + rb.pickup - rb.peak
            low = min(low_a + rb.delivery, low_b + ra.pickup)
            reverse = ra.delivery + rb.delivery + ra.pickup + rb.pickup - low
            if min(forward, reverse) > self.cap:
                continue
            visits = va + vb
            if reverse < forward:
                visits.reverse()
            new_route = _Route(visits, self)
            if new_route.peak > self.cap:
                continue
            routes[a], routes[b] = new_route, None
            for node, mask in visits:
                owner[node] = a
        answer = [route for route in routes if route is not None]
        self.keep(answer)
        return answer

    def polish(self, route, deadline):
        key = tuple(sorted(route.v))
        cached = self.order_cache.get(key)
        if cached is not None and (cached.cost, cached.peak) < (route.cost, route.peak):
            route = cached
        while len(route.v) > 2 and time.perf_counter() < deadline:
            visits, dist = route.v, self.dist
            length, changed = len(visits), False
            # 2-opt exchange under symmetric distances; the peak after segment reversal follows from the minimum of the original load interval.
            for a in range(length - 1):
                if time.perf_counter() >= deadline:
                    break
                before = visits[a - 1][0] if a else 0
                first, minimum = visits[a][0], route.loads[a]
                for b in range(a + 1, length):
                    minimum = min(minimum, route.loads[b], route.loads[b + 1])
                    last = visits[b][0]
                    after = visits[b + 1][0] if b + 1 < length else 0
                    delta = dist[before][last] + dist[first][after] - dist[before][first] - dist[last][after]
                    if delta < 0 and route.loads[a] + route.loads[b + 1] - minimum <= self.cap:
                        candidate = _Route(visits[:a] + visits[a:b + 1][::-1] + visits[b + 1:], self)
                        if candidate.peak <= self.cap:
                            route, changed = candidate, True
                            break
                    if b % 64 == 0 and time.perf_counter() >= deadline:
                        break
                if changed:
                    break
            if changed:
                continue
            # Intra-route shift of one node or two adjacent nodes.
            for a in range(length):
                if time.perf_counter() >= deadline:
                    break
                for size in (1, 2):
                    if a + size > length:
                        continue
                    segment = visits[a:a + size]
                    rest = visits[:a] + visits[a + size:]
                    before = visits[a - 1][0] if a else 0
                    after = visits[a + size][0] if a + size < length else 0
                    first, last = segment[0][0], segment[-1][0]
                    remove = dist[before][after] - dist[before][first] - dist[last][after]
                    for b in range(len(rest) + 1):
                        if b == a:
                            continue
                        left = rest[b - 1][0] if b else 0
                        right = rest[b][0] if b < len(rest) else 0
                        delta = remove + dist[left][first] + dist[last][right] - dist[left][right]
                        if delta < 0:
                            candidate = _Route(rest[:b] + segment + rest[b:], self)
                            if candidate.peak <= self.cap:
                                route, changed = candidate, True
                                break
                        if b % 64 == 0 and time.perf_counter() >= deadline:
                            break
                    if changed:
                        break
                if changed:
                    break
            if not changed:
                break
        if route.delivery + route.pickup - route.low < route.peak - 1e-9:
            route = _Route(route.v[::-1], self)
        if cached is None or (route.cost, route.peak) < (cached.cost, cached.peak):
            if len(self.order_cache) >= self.cache_limit:
                self.order_cache.pop(next(iter(self.order_cache)))
            self.order_cache[key] = route
        return route

    def owners(self, routes):
        owners = [[] for _ in range(self.n + 1)]
        for rid, route in enumerate(routes):
            for pos, (i, mask) in enumerate(route.v):
                owners[i].append((rid, pos))
        return owners

    def cross_basic(self, routes, deadline):
        routes = routes[:]
        dist = self.dist
        while time.perf_counter() < deadline:
            owners = self.owners(routes)
            customers = list(range(1, self.n + 1))
            self.rng.shuffle(customers)
            changed = False
            for i in customers:
                if time.perf_counter() >= deadline:
                    break
                neighbors = [i] + self.nearby(i)
                for a, apos in owners[i]:
                    ra = routes[a]
                    item = ra.v[apos]
                    prev = ra.v[apos - 1][0] if apos else 0
                    nxt = ra.v[apos + 1][0] if apos + 1 < len(ra.v) else 0
                    removal = dist[prev][nxt] - dist[prev][i] - dist[i][nxt]
                    targets = {b for j in neighbors for b, _ in owners[j] if b != a}
                    move = None
                    gain = 0
                    for b in targets:
                        if time.perf_counter() >= deadline:
                            break
                        option = self.insertion(routes[b], i, item[1])
                        if option is not None and removal + option[0] < gain:
                            gain = removal + option[0]
                            move = (b, option[1])
                    if move is not None:
                        b, position = move
                        routes[a] = _Route(ra.v[:apos] + ra.v[apos + 1:], self)
                        routes[b] = self.inserted(routes[b], i, item[1], position)
                        changed = True
                        break
                    for j in neighbors:
                        if time.perf_counter() >= deadline:
                            break
                        for b, bpos in owners[j]:
                            if b == a or j == i:
                                continue
                            rb = routes[b]
                            # Whole-visit exchange; single-service requests remain indivisible.
                            if j not in ra.where and i not in rb.where:
                                p2 = rb.v[bpos - 1][0] if bpos else 0
                                n2 = rb.v[bpos + 1][0] if bpos + 1 < len(rb.v) else 0
                                delta = (dist[prev][j] + dist[j][nxt] - dist[prev][i] - dist[i][nxt]
                                         + dist[p2][i] + dist[i][n2] - dist[p2][j] - dist[j][n2])
                                if delta < 0:
                                    va, vb = ra.v[:], rb.v[:]
                                    va[apos], vb[bpos] = vb[bpos], va[apos]
                                    ca = self.valid_route(va)
                                    cb = self.valid_route(vb) if ca is not None else None
                                    if cb is not None:
                                        routes[a], routes[b] = ca, cb
                                        changed = True
                                        break
                            # Suffix exchange also allows merging two complementary services.
                            common = not ra.where.keys().isdisjoint(rb.where)
                            for acut, bcut in ((apos + 1, bpos + 1), (apos, bpos),
                                               (apos + 1, bpos), (apos, bpos + 1)):
                                al = ra.v[acut - 1][0] if acut else 0
                                ar = ra.v[acut][0] if acut < len(ra.v) else 0
                                bl = rb.v[bcut - 1][0] if bcut else 0
                                br = rb.v[bcut][0] if bcut < len(rb.v) else 0
                                delta = dist[al][br] + dist[bl][ar] - dist[al][ar] - dist[bl][br]
                                if delta >= 0 and not common:
                                    continue
                                ca = self.valid_route(ra.v[:acut] + rb.v[bcut:])
                                cb = self.valid_route(rb.v[:bcut] + ra.v[acut:]) if ca is not None else None
                                if cb is not None and ca.cost + cb.cost < ra.cost + rb.cost:
                                    routes[a], routes[b] = ca, cb
                                    changed = True
                                    break
                            if changed:
                                break
                        if changed:
                            break
                    if changed:
                        break
                if changed:
                    break
            if not changed:
                break
            routes = [route for route in routes if route.v]
            self.keep(routes)
        return routes

    def variants(self, route, deadline):
        positions = list(range(len(route.v)))
        self.rng.shuffle(positions)
        # Single-neighbourhood width grows with the budget; the next call still randomly covers other positions.
        positions = positions[:max(2, self.width)]
        result = []
        for pos in positions:
            if time.perf_counter() >= deadline:
                break
            i, mask = route.v[pos]
            for action in ((3, 1, 2) if mask == 3 else (mask,)):
                visits = route.v[:]
                if mask != action:
                    visits[pos] = (i, mask ^ action)
                else:
                    del visits[pos]
                rest = _Route(visits, self)
                result.append((i, action, rest.cost - route.cost, rest))
        return result

    def exchange_pair(self, a, b, deadline):
        va, vb = self.variants(a, deadline), self.variants(b, deadline)
        for variants, target, forward in ((va, b, True), (vb, a, False)):
            for i, mask, removal, rest in variants:
                if time.perf_counter() >= deadline:
                    return None
                choice = self.insertion(target, i, mask)
                if choice is not None and removal + choice[0] < 0:
                    new = self.inserted(target, i, mask, choice[1])
                    return (rest, new) if forward else (new, rest)
        # Both sides first remove services, then each finds its best position, not limited to swapping in place.
        for i, mask_i, da, na in va:
            if time.perf_counter() >= deadline:
                return None
            for j, mask_j, db, nb in vb:
                if time.perf_counter() >= deadline:
                    return None
                if i == j:
                    continue
                qdi = self.d[i] if mask_i & 1 else 0.0
                qpi = self.p[i] if mask_i & 2 else 0.0
                qdj = self.d[j] if mask_j & 1 else 0.0
                qpj = self.p[j] if mask_j & 2 else 0.0
                if (nb.delivery + qdi > self.cap or nb.pickup + qpi > self.cap
                        or na.delivery + qdj > self.cap or na.pickup + qpj > self.cap):
                    continue
                into_b = self.insertion(nb, i, mask_i)
                if into_b is None:
                    continue
                into_a = self.insertion(na, j, mask_j)
                if into_a is not None and da + db + into_a[0] + into_b[0] < 0:
                    return (self.inserted(na, j, mask_j, into_a[1]),
                            self.inserted(nb, i, mask_i, into_b[1]))
        return None

    def cross_search(self, routes, deadline):
        now = time.perf_counter()
        routes = self.cross_basic(routes, now + max(0.0, deadline - now) * 0.68)
        routes = routes[:]
        while len(routes) > 1 and time.perf_counter() < deadline:
            owners = self.owners(routes)
            order = list(range(len(routes)))
            self.rng.shuffle(order)
            changed = False
            for a in order:
                if time.perf_counter() >= deadline:
                    break
                ra = routes[a]
                anchor = self.rng.choice(ra.v)[0]
                targets = {b for i in [anchor] + self.nearby(anchor)
                           for b, _ in owners[i] if b != a}
                ordered = list(targets)
                self.rng.shuffle(ordered)
                for b in ordered:
                    if time.perf_counter() >= deadline:
                        break
                    rb = routes[b]
                    key = (ra, rb) if a < b else (rb, ra)
                    if key in self.exchange_cache:
                        continue
                    pair = self.exchange_pair(ra, rb, deadline)
                    if pair is not None:
                        routes[a], routes[b] = pair
                        routes = [r for r in routes if r.v]
                        self.keep(routes)
                        changed = True
                        break
                    # Only fully scanned position sets enter this round's cache.
                    if time.perf_counter() < deadline:
                        if len(self.exchange_cache) >= self.cache_limit:
                            self.exchange_cache.clear()
                        self.exchange_cache.add(key)
                if changed:
                    break
            if not changed:
                break
        return routes

    def destroy(self, routes, count, operator, split_probability):
        rng, n, dist = self.rng, self.n, self.dist
        seed = rng.randrange(1, n + 1)
        if operator == 0:
            ranked = sorted(range(1, n + 1), key=lambda i: dist[seed][i] * (0.65 + rng.random()))
            selected = ranked[:count]
        elif operator == 1:
            selected = rng.sample(range(1, n + 1), count)
        elif operator == 2:
            scores = [0.0] * (n + 1)
            for route in routes:
                for pos, (i, mask) in enumerate(route.v):
                    prev = route.v[pos - 1][0] if pos else 0
                    nxt = route.v[pos + 1][0] if pos + 1 < len(route.v) else 0
                    scores[i] += dist[prev][i] + dist[i][nxt] - dist[prev][nxt]
            selected = heapq.nlargest(count, range(1, n + 1),
                                      key=lambda i: scores[i] * (0.3 + rng.random()))
        elif operator == 5:
            counts = [0] * (n + 1)
            for route in routes:
                for i, mask in route.v:
                    counts[i] += 1
            selected = sorted(range(1, n + 1),
                              key=lambda i: (-counts[i], dist[seed][i] * (0.5 + rng.random())))[:count]
        else:
            selected, seen = [], set()
            ranked = list(routes)
            rng.shuffle(ranked)
            if operator == 3:
                ranked.sort(key=lambda route: min(dist[seed][i] for i, mask in route.v))
            elif operator == 6:
                ranked.sort(key=lambda route: -route.cost / max(1, len(route.v))
                            * (0.6 + rng.random()))
            for route in ranked:
                if operator == 4:
                    start = rng.randrange(len(route.v))
                    visits = route.v[start:] + route.v[:start]
                    size = max(1, int(math.sqrt(len(visits)) * (0.5 + rng.random())))
                    visits = visits[:size]
                else:
                    visits = route.v
                for i, mask in visits:
                    if i not in seen:
                        seen.add(i)
                        selected.append(i)
                        if len(selected) >= count:
                            break
                if len(selected) >= count:
                    break
            if len(selected) < count:
                rest = [i for i in range(1, n + 1) if i not in seen]
                selected.extend(rng.sample(rest, count - len(selected)))
        remove = {}
        one_kind = rng.choice((1, 2))
        partial = rng.random() < split_probability
        for i in selected:
            remove[i] = one_kind if partial and rng.random() < 0.8 else 3
        reduced, missing = [], {}
        for route in routes:
            visits, changed = [], False
            for i, mask in route.v:
                taken = mask & remove.get(i, 0)
                if taken:
                    missing[i] = missing.get(i, 0) | taken
                    changed = True
                remaining = mask & ~taken
                if remaining:
                    visits.append((i, remaining))
            if visits:
                reduced.append(_Route(visits, self) if changed else route)
        return reduced, list(missing.items())

    def repair(self, routes, jobs, deadline):
        routes = routes[:]
        depth = 2 if self.rng.random() < 0.5 else max(2, min(5, int(2 + self.effort / 2)))
        tasks = []
        for i, mask in jobs:
            if time.perf_counter() >= deadline:
                return None
            job = _Job(i, mask, self)
            for rid, route in enumerate(routes):
                if rid % 16 == 0 and time.perf_counter() >= deadline:
                    return None
                for action, value in self.insertion_bundle(route, i, mask, deadline).items():
                    if value is not None:
                        job.parts[action].options[rid] = (value[0], rid, value[1])
            for part in job.parts.values():
                part.top = heapq.nsmallest(depth, part.options.values())
            job.refresh(depth)
            tasks.append(job)
        while tasks:
            if time.perf_counter() >= deadline:
                return None
            index = max(range(len(tasks)), key=lambda j: tasks[j].priority)
            job = tasks.pop(index)
            changed = []
            for mask, (_, rid, position) in job.plan:
                if rid == -1:
                    rid = len(routes)
                    routes.append(_Route([(job.i, mask)], self))
                else:
                    routes[rid] = self.inserted(routes[rid], job.i, mask, position)
                changed.append(rid)
            for index, other in enumerate(tasks):
                if index % 16 == 0 and time.perf_counter() >= deadline:
                    return None
                affected = False
                for rid in changed:
                    options = self.insertion_bundle(routes[rid], other.i, other.mask, deadline)
                    for mask, value in options.items():
                        affected = other.parts[mask].update(rid, value, depth) or affected
                if affected:
                    other.refresh(depth)
        return routes

    def add_elite(self, routes):
        value = sum(r.cost for r in routes)
        if self.elite and value > self.best_cost * (1.035 + 0.02 * self.effort / (1 + self.effort)):
            return
        signature = tuple(sorted(tuple(r.v) for r in routes))
        if any(signature == item[2] for item in self.elite):
            return
        self.elite.append((value, routes[:], signature))
        self.elite.sort(key=lambda item: item[0])
        del self.elite[self.elite_limit:]

    def recombine(self, donor, receiver, fraction):
        covered = [0] * (self.n + 1)
        offspring, taken = [], 0
        ranked = donor[:]
        self.rng.shuffle(ranked)
        ranked.sort(key=lambda r: r.cost / max(1, len(r.v)) * (0.7 + self.rng.random()))
        target = 2 * self.n * fraction
        for route in ranked:
            if taken >= target or time.perf_counter() >= self.end:
                break
            offspring.append(route)
            for i, mask in route.v:
                covered[i] |= mask
                taken += int(bool(mask & 1)) + int(bool(mask & 2))
        for route in receiver:
            visits = [(i, mask & ~covered[i]) for i, mask in route.v if mask & ~covered[i]]
            if visits:
                offspring.append(route if visits == route.v else _Route(visits, self))
        return offspring

    def run(self):
        if self.n == 0 or time.perf_counter() >= self.end:
            return
        if any(self.d[i] > self.cap or self.p[i] > self.cap for i in range(1, self.n + 1)):
            return
        available = max(0.0, self.end - time.perf_counter())
        current = self.savings(min(self.end, time.perf_counter() + available * 0.10))
        polish_end = min(self.end, time.perf_counter() + available * 0.04)
        current = [self.polish(route, polish_end) for route in current]
        self.keep(current)
        current = self.cross_search(current, min(self.end, time.perf_counter() + available * 0.07))
        current_cost = self.keep(current)
        self.add_elite(current)
        weights = [1.0] * 7
        last_improvement = time.perf_counter()
        cycle_started = last_improvement
        epoch = max(0.002, self.budget / (2.0 + math.sqrt(1.0 + self.budget)))
        repair_average = 0.0
        while time.perf_counter() < self.end:
            now = time.perf_counter()
            stalled = (now - last_improvement) / epoch
            if now - cycle_started >= epoch:
                if len(self.elite) > 1 and self.rng.random() < 0.3:
                    pick = int(self.rng.random() ** 2 * len(self.elite))
                    current_cost, saved, _ = self.elite[pick]
                    current = saved[:]
                else:
                    current, current_cost = self.best[:], self.best_cost
                cycle_started = now
                self.exchange_cache.clear()
            progress = min(1.0, (now - cycle_started) / epoch)
            temperature = (max(1.0, self.best_cost / max(1.0, math.sqrt(self.n)))
                           * (0.07 + 0.015 * self.effort) * math.exp(-5.0 * progress))
            base = min(self.n * 0.24, math.sqrt(self.n) * (0.7 + 0.2 * math.sqrt(self.budget))
                       + self.n / max(1, len(current)))
            count = min(self.n, max(1, int(base * self.rng.uniform(0.45, 1.45)
                                           * (1.0 + 0.3 * min(2.0, stalled)))))
            if self.rng.random() < 1.0 / (8.0 + self.effort):
                count = min(self.n, max(count, int(self.n * self.rng.uniform(0.25, 0.55))))
            # Shrink the destruction size near the deadline according to the online repair time.
            remaining = self.end - now
            if repair_average > 0 and remaining < repair_average * 3:
                count = max(1, int(count * remaining / (repair_average * 3)))
            draw = self.rng.random() * sum(weights)
            operator = len(weights) - 1
            for index, weight in enumerate(weights):
                draw -= weight
                if draw <= 0:
                    operator = index
                    break
            source = current
            if len(self.elite) > 1 and self.rng.random() < 0.025 + 0.015 * self.effort:
                donor = self.elite[self.rng.randrange(len(self.elite))][1]
                source = self.recombine(donor, current, self.rng.uniform(0.15, 0.45))
                count = max(1, count // 2)
            split_probability = min(0.7, 0.15 + 0.18 * stalled + 0.2 * self.rng.random())
            reduced, jobs = self.destroy(source, count, operator, split_probability)
            repair_started = time.perf_counter()
            candidate = self.repair(reduced, jobs, self.end)
            if candidate is None:
                break
            repair_elapsed = time.perf_counter() - repair_started
            repair_average = repair_elapsed if not repair_average else 0.8 * repair_average + 0.2 * repair_elapsed
            old_best = self.best_cost
            self.keep(candidate)
            current_set = set(current)
            changed = [index for index, route in enumerate(candidate) if route not in current_set]
            self.rng.shuffle(changed)
            local_end = min(self.end, time.perf_counter() + max(self.budget * 0.00004, repair_elapsed * 0.25))
            for index in changed:
                if time.perf_counter() >= local_end:
                    break
                candidate[index] = self.polish(candidate[index], local_end)
            value = self.keep(candidate)
            if value < old_best or self.rng.random() < 0.09:
                local_slice = min(self.budget * (0.010 + 0.001 * self.effort) / max(1.0, math.log1p(self.n)),
                                  max(0.0, self.end - time.perf_counter()) * 0.12)
                candidate = self.cross_search(candidate, min(self.end, time.perf_counter() + local_slice))
                value = self.keep(candidate)
            if self.best_cost < old_best:
                last_improvement = time.perf_counter()
                reward = 7.0
                self.add_elite(self.best)
            elif value < current_cost:
                reward = 3.0
            else:
                reward = 0.25
            difference = value - current_cost
            if difference <= 0 or self.rng.random() < math.exp(-difference / max(1e-9, temperature)):
                current, current_cost = candidate, value
                reward = max(reward, 0.7)
                if self.rng.random() < 0.05:
                    self.add_elite(current)
            weights[operator] = 0.92 * weights[operator] + 0.08 * reward

    def audit(self):
        delivery, pickup = [0] * (self.n + 1), [0] * (self.n + 1)
        for route in self.best:
            if not route.v or len(route.where) != len(route.v):
                return False
            if route.peak > self.cap or route.low < -1e-8 * max(1.0, self.capacity):
                return False
            for i, mask in route.v:
                if not 1 <= i <= self.n or mask not in (1, 2, 3):
                    return False
                delivery[i] += bool(mask & 1)
                pickup[i] += bool(mask & 2)
        return all(delivery[i] == 1 and pickup[i] == 1 for i in range(1, self.n + 1))

    def result(self):
        return {'routes': [[{'customer_id': i,
                            'serve_delivery': bool(mask & 1),
                            'serve_pickup': bool(mask & 2)}
                           for i, mask in route.v]
                          for route in self.best if route.v]}


def solve(instance: dict, time_limit_s: float) -> dict:
    started = time.perf_counter()
    fallback = {'routes': []}
    engine = None
    try:
        n = max(0, int(instance['n']))
        fallback = {'routes': [[{'customer_id': i, 'serve_delivery': True,
                                 'serve_pickup': True}] for i in range(1, n + 1)]}
        budget = float(time_limit_s)
        if not math.isfinite(budget) or budget <= 0 or not n:
            return fallback
        engine = _Engine(instance, budget, started)
        engine.run()
        if engine.audit():
            return engine.result()
    except BaseException:
        if engine is not None:
            try:
                if engine.audit():
                    return engine.result()
            except BaseException:
                pass
    return fallback
