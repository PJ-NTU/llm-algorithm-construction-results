"""Bidirectional capacity insertion and route-pool recombination search
Extends regret repair with bidirectional insertion and merge relocation, combining adaptive large neighbourhood search with time-limited exact set-cover recombination over a historical route pool, always keeping the best feasible solution.
"""
import bisect
import heapq
import math
import random
import time


class _Route:
    __slots__ = ('v', 'cost', 'delivery', 'pickup', 'loads', 'prefix',
                 'suffix', 'peak', 'low', 'where', 'dcum', 'pcum',
                 'negative_suffix', 'prefix_low_negative', 'suffix_low',
                 'coverage', 'cache', 'polished')

    def __init__(self, visits, engine):
        self.v = visits
        d, p, dist = engine.d, engine.p, engine.dist
        self.delivery = sum(d[i] for i, mask in visits if mask & 1)
        self.pickup = sum(p[i] for i, mask in visits if mask & 2)
        load = self.delivery
        loads = [load]
        cost, prev = 0, 0
        where = {}
        coverage = 0
        dcum, pcum = [0.0], [0.0]
        served_d, served_p = 0.0, 0.0
        for pos, (i, mask) in enumerate(visits):
            cost += dist[prev][i]
            prev = i
            if mask & 1:
                load -= d[i]
                served_d += d[i]
            if mask & 2:
                load += p[i]
                served_p += p[i]
            loads.append(load)
            dcum.append(served_d)
            pcum.append(served_p)
            where[i] = pos
            coverage |= mask << (2 * (i - 1))
        self.cost = cost + dist[prev][0]
        self.loads = loads
        self.where = where
        self.coverage = coverage
        prefix, prefix_low_negative = [], []
        high, low = -math.inf, math.inf
        for load in loads:
            if load > high:
                high = load
            if load < low:
                low = load
            prefix.append(high)
            prefix_low_negative.append(-low)
        self.peak, self.low = high, low
        self.prefix = prefix
        suffix, suffix_low = [0.0] * len(loads), [0.0] * len(loads)
        high, low = -math.inf, math.inf
        for j in range(len(loads) - 1, -1, -1):
            value = loads[j]
            if value > high:
                high = value
            if value < low:
                low = value
            suffix[j], suffix_low[j] = high, low
        self.suffix = suffix
        self.dcum, self.pcum = dcum, pcum
        self.negative_suffix = [-value for value in suffix]
        self.prefix_low_negative = prefix_low_negative
        self.suffix_low = suffix_low
        self.cache = {}
        self.polished = False


class _Task:
    __slots__ = ('i', 'mask', 'options', 'top', 'jitter')

    def __init__(self, i, mask, depot_cost, jitter):
        self.i, self.mask, self.jitter = i, mask, jitter
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
            self.top = heapq.nsmallest(depth, self.options.values())


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
        self.started = started
        reserve = min(0.15, 0.015 * self.budget) + min(0.03, self.n * 0.000003)
        self.end = started + max(0.0, self.budget - reserve)
        self.rng = random.Random(731927 + self.n * 101)
        self.near = {}
        self.width = min(max(0, self.n - 1), max(4, int(
            5 + 3 * math.sqrt(self.budget) + 2 * math.log1p(self.n))))
        self.star_width = min(self.width, max(2, int(2 + math.log1p(self.budget))))
        self.insertion_cache_limit = max(8, self.width * (2 + int(math.log1p(self.budget))))
        self.best = [_Route([(i, 3)], self) for i in range(1, self.n + 1)]
        self.best_cost = sum(route.cost for route in self.best)
        self.pool = {route.coverage: route for route in self.best}
        self.pool_limit = max(1, int(self.n * (5 + 3 * math.sqrt(self.budget))))
        self.patch_width = max(2, int(2 + math.log1p(self.budget)))
        self.order_cache = {}
        self.cache_limit = max(1, int(self.n * (5 + math.sqrt(self.budget)) * 8))

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
        # Routes are deduplicated by the delivery/pickup requests they cover, keeping only the cheapest ordering.
        for route in routes:
            if not route.v:
                continue
            old = self.pool.get(route.coverage)
            if old is None or route.cost < old.cost:
                self.pool[route.coverage] = route
        if len(self.pool) > self.pool_limit:
            protected = {route.coverage for route in self.best}
            for key in list(self.pool):
                if len(self.pool) <= self.pool_limit:
                    break
                if key not in protected:
                    del self.pool[key]
        return value

    def valid_route(self, visits):
        if not visits:
            return _Route([], self)
        if len({i for i, mask in visits}) != len(visits):
            return None
        route = _Route(visits, self)
        return route if route.peak <= self.cap else None

    def insertion(self, route, i, mask):
        extra_d = self.d[i] if mask & 1 else 0.0
        extra_p = self.p[i] if mask & 2 else 0.0
        if route.delivery + extra_d > self.cap or route.pickup + extra_p > self.cap:
            return None
        key = (i, mask)
        if key in route.cache:
            return route.cache[key]
        length = len(route.v)
        total = route.delivery + route.pickup
        existing = route.where.get(i)
        if existing is not None:
            old_mask = route.v[existing][1]
            if old_mask & mask:
                return None
            if (route.prefix[existing] + extra_d <= self.cap and
                    route.suffix[existing + 1] + extra_p <= self.cap):
                return (0, -existing - 1)
            if (total - route.suffix_low[existing + 1] + extra_d <= self.cap and
                    total + route.prefix_low_negative[existing] + extra_p <= self.cap):
                return (0, -length - existing - 1)
            # A merged service need not stay at the original visit position: remove the old service first, then insert as a whole.
            reduced = _Route(route.v[:existing] + route.v[existing + 1:], self)
            option = self.insertion(reduced, i, old_mask | mask)
            result = None if option is None else (
                reduced.cost - route.cost + option[0], 2 * length + 2 + option[1])
        else:
            pd, sp = self.cap - extra_d, self.cap - extra_p
            left = bisect.bisect_left(route.negative_suffix, -sp)
            right = bisect.bisect_right(route.prefix, pd)
            # The peaks on both sides of the original boundary k when travelling in reverse follow directly from the load minimum.
            rev_left = bisect.bisect_left(route.suffix_low, total + extra_d - self.cap)
            rev_right = bisect.bisect_right(route.prefix_low_negative,
                                           self.cap - total - extra_p)
            intervals = []
            if left < right:
                intervals.append((left, right))
            if rev_left < rev_right:
                intervals.append((rev_left, rev_right))
            intervals.sort()
            best, position, covered = math.inf, 0, 0
            row, dist = self.dist[i], self.dist
            for lo, hi in intervals:
                lo = max(lo, covered)
                covered = max(covered, hi)
                prev = route.v[lo - 1][0] if 0 < lo <= length else 0
                for k in range(lo, hi):
                    if k % 32 == 0 and time.perf_counter() >= self.end:
                        return None
                    nxt = route.v[k][0] if k < length else 0
                    delta = dist[prev][i] + row[nxt] - dist[prev][nxt]
                    if delta < best:
                        best = delta
                        position = k if left <= k < right else length + 1 + k
                    prev = nxt
            result = None if best == math.inf else (best, position)
        if len(route.cache) >= self.insertion_cache_limit:
            route.cache.clear()
        route.cache[key] = result
        return result

    def inserted(self, route, i, mask, position):
        visits = route.v[:]
        length = len(visits)
        if position >= 2 * length + 2:
            k = route.where[i]
            combined = visits[k][1] | mask
            reduced = _Route(visits[:k] + visits[k + 1:], self)
            return self.inserted(reduced, i, combined, position - 2 * length - 2)
        if position < -length:
            k = -position - length - 1
            visits[k] = (i, visits[k][1] | mask)
            visits.reverse()
        elif position < 0:
            k = -position - 1
            visits[k] = (i, visits[k][1] | mask)
        elif position > length:
            k = position - length - 1
            visits.reverse()
            visits.insert(length - k, (i, mask))
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
        if route.polished or time.perf_counter() >= deadline:
            return route
        key = route.coverage
        cached = self.order_cache.get(key)
        if cached is not None and cached.cost <= route.cost:
            route = cached
        if route.polished:
            return route
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
                if time.perf_counter() < deadline:
                    route.polished = True
                break
        if cached is None or route.cost < cached.cost:
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

    def cross_search(self, routes, deadline):
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
                    reduced_a = None
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
                    for neighbor_rank, j in enumerate(neighbors):
                        if time.perf_counter() >= deadline:
                            break
                        for b, bpos in owners[j]:
                            if b == a or j == i:
                                continue
                            rb = routes[b]
                            p2 = rb.v[bpos - 1][0] if bpos else 0
                            n2 = rb.v[bpos + 1][0] if bpos + 1 < len(rb.v) else 0
                            # Whole-visit exchange; single-service requests remain indivisible.
                            if j not in ra.where and i not in rb.where:
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
                            # Swap requests rather than fixed positions: each is reinserted into the other route after removal.
                            if neighbor_rank <= self.star_width:
                                removal_b = dist[p2][n2] - dist[p2][j] - dist[j][n2]
                                mi, mj = item[1], rb.v[bpos][1]
                                di = self.d[i] if mi & 1 else 0.0
                                pi = self.p[i] if mi & 2 else 0.0
                                dj = self.d[j] if mj & 1 else 0.0
                                pj = self.p[j] if mj & 2 else 0.0
                                if (ra.delivery - di + dj <= self.cap and
                                        ra.pickup - pi + pj <= self.cap and
                                        rb.delivery - dj + di <= self.cap and
                                        rb.pickup - pj + pi <= self.cap):
                                    if reduced_a is None:
                                        reduced_a = _Route(ra.v[:apos] + ra.v[apos + 1:], self)
                                    into_a = self.insertion(reduced_a, j, mj)
                                    if into_a is not None:
                                        reduced_b = _Route(rb.v[:bpos] + rb.v[bpos + 1:], self)
                                        into_b = self.insertion(reduced_b, i, mi)
                                        if (into_b is not None and removal + removal_b
                                                + into_a[0] + into_b[0] < 0):
                                            routes[a] = self.inserted(reduced_a, j, mj, into_a[1])
                                            routes[b] = self.inserted(reduced_b, i, mi, into_b[1])
                                            changed = True
                                            break
                            # Two-route suffix exchange; combinations with duplicate customers are rejected.
                            for acut, bcut in ((apos + 1, bpos + 1), (apos, bpos),
                                               (apos + 1, bpos), (apos, bpos + 1)):
                                al = ra.v[acut - 1][0] if acut else 0
                                ar = ra.v[acut][0] if acut < len(ra.v) else 0
                                bl = rb.v[bcut - 1][0] if bcut else 0
                                br = rb.v[bcut][0] if bcut < len(rb.v) else 0
                                delta = dist[al][br] + dist[bl][ar] - dist[al][ar] - dist[bl][br]
                                if delta >= 0:
                                    continue
                                # After a suffix exchange the loads of both segments only shift by a constant, so overloads are screened out in constant time first.
                                da = ra.delivery - ra.dcum[acut]
                                db = rb.delivery - rb.dcum[bcut]
                                pa, pb = ra.pcum[acut], rb.pcum[bcut]
                                if (max(ra.prefix[acut] + db - da,
                                        rb.suffix[bcut] + pa - pb) > self.cap or
                                        max(rb.prefix[bcut] + da - db,
                                            ra.suffix[acut] + pb - pa) > self.cap):
                                    continue
                                ca = self.valid_route(ra.v[:acut] + rb.v[bcut:])
                                cb = self.valid_route(rb.v[:bcut] + ra.v[acut:]) if ca is not None else None
                                if cb is not None:
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
            # Remove a whole sparser route entirely, together with neighbouring customers, to make room for eliminating that route.
            anchor = min(routes, key=lambda route: len(route.v) * (0.5 + rng.random()))
            selected = [i for i, mask in anchor.v]
            seen = set(selected)
            seed = rng.choice(selected)
            need = max(0, count - len(selected))
            selected += heapq.nsmallest(need, (i for i in range(1, n + 1) if i not in seen),
                                       key=lambda i: dist[seed][i] * (0.7 + rng.random()))
        else:
            selected, seen = [], set()
            ranked = list(routes)
            rng.shuffle(ranked)
            if operator == 3:
                ranked.sort(key=lambda route: min(dist[seed][i] for i, mask in route.v))
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
        partial = operator != 5 and rng.random() < split_probability
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
        jobs = []
        for i, mask in missing.items():
            if mask == 3 and rng.random() < split_probability * 0.4:
                jobs.extend(((i, 1), (i, 2)))
            else:
                jobs.append((i, mask))
        return reduced, jobs

    def repair(self, routes, jobs, deadline):
        routes = routes[:]
        depth = 2 if self.rng.random() < 0.5 else max(2, min(4, int(2 + math.log1p(self.budget) / 2)))
        flexible = self.rng.random() < 0.25 + 0.25 * math.log1p(self.budget) / (1 + math.log1p(self.budget))
        tasks = []
        for i, mask in jobs:
            if time.perf_counter() >= deadline:
                return None
            task = _Task(i, mask, self.dist[0][i] + self.dist[i][0], 0.8 + 0.4 * self.rng.random())
            for rid, route in enumerate(routes):
                option = self.insertion(route, i, mask)
                if option is not None:
                    task.options[rid] = (option[0], rid, option[1])
                if rid % 32 == 0 and time.perf_counter() >= deadline:
                    return None
            task.top = heapq.nsmallest(depth, task.options.values())
            tasks.append(task)
        while tasks:
            if time.perf_counter() >= deadline:
                return None
            chosen, priority = 0, -math.inf
            for index, task in enumerate(tasks):
                if index % 64 == 0 and time.perf_counter() >= deadline:
                    return None
                first = task.top[0][0]
                regret = sum(option[0] - first for option in task.top[1:])
                if len(task.top) < depth:
                    regret += (depth - len(task.top)) * (task.top[-1][0] - first)
                score = task.jitter * regret - 0.03 * first
                if score > priority:
                    chosen, priority = index, score
            task = tasks.pop(chosen)
            first, rid, position = task.top[0]
            actions = [(task.mask, rid, position)]
            # Compute split-vehicle options only for the currently chosen expensive combined request, avoiding a threefold enumeration for every pending request.
            if task.mask == 3 and (rid == -1 or (flexible and
                    first > 0.25 * self.best_cost / max(1, self.n))):
                depot_cost = self.dist[0][task.i] + self.dist[task.i][0]
                alternatives = [[(depot_cost, -1, 0)], [(depot_cost, -1, 0)]]
                for r, route in enumerate(routes):
                    if r % 16 == 0 and time.perf_counter() >= deadline:
                        return None
                    for k, mask in enumerate((1, 2)):
                        option = self.insertion(route, task.i, mask)
                        if option is not None:
                            top = alternatives[k]
                            bisect.insort(top, (option[0], r, option[1]))
                            if len(top) > 2:
                                top.pop()
                for one in alternatives[0]:
                    for two in alternatives[1]:
                        if one[1] >= 0 and one[1] == two[1]:
                            continue
                        if one[0] + two[0] < first:
                            first = one[0] + two[0]
                            actions = [(1, one[1], one[2]), (2, two[1], two[2])]
            changed = []
            for mask, rid, position in actions:
                if rid == -1:
                    rid = len(routes)
                    routes.append(_Route([(task.i, mask)], self))
                else:
                    routes[rid] = self.inserted(routes[rid], task.i, mask, position)
                changed.append(rid)
            for rid in changed:
                route = routes[rid]
                for index, pending in enumerate(tasks):
                    option = self.insertion(route, pending.i, pending.mask)
                    pending.update(rid, option, depth)
                    if index % 32 == 0 and time.perf_counter() >= deadline:
                        return None
        return routes

    def cover_patch(self, remaining, ceiling, bank, deadline):
        """Search for a strictly cheaper complete route combination under a cap on uncovered requests."""
        if not remaining:
            return []
        local = {}
        for index, route in enumerate(bank):
            if index % 64 == 0 and time.perf_counter() >= deadline:
                return None
            mask = route.coverage
            if mask & remaining == mask and route.cost < ceiling:
                old = local.get(mask)
                if old is None or route.cost < old.cost:
                    local[mask] = route
        # Single-customer columns let half-request residuals be completed too; they are generated only within this local subproblem.
        rest = remaining
        while rest:
            if time.perf_counter() >= deadline:
                return None
            bit = rest & -rest
            shift = ((bit.bit_length() - 1) // 2) * 2
            mask = (remaining >> shift) & 3
            coverage = mask << shift
            if coverage not in local:
                route = _Route([(shift // 2 + 1, mask)], self)
                if route.cost < ceiling:
                    local[coverage] = route
            rest &= ~(3 << shift)
        columns = list(local.values())
        covers, weights = {}, {}
        masks, costs, sizes = [], [], []
        for index, route in enumerate(columns):
            if index % 32 == 0 and time.perf_counter() >= deadline:
                return None
            mask = route.coverage
            size = mask.bit_count()
            masks.append(mask)
            costs.append(route.cost)
            sizes.append(size)
            share = route.cost / size
            rest = mask
            while rest:
                bit = rest & -rest
                rest ^= bit
                covers.setdefault(bit, []).append(index)
                weights[bit] = min(weights.get(bit, math.inf), share)
        if len(covers) != remaining.bit_count():
            return None
        slack = []
        for mask, cost in zip(masks, costs):
            value, rest = 0.0, mask
            while rest:
                bit = rest & -rest
                rest ^= bit
                value += weights[bit]
            slack.append(cost - value)
        # Non-negative dual prices always keep the price sum of each candidate route no greater than its cost.
        for bit in sorted(covers, key=lambda b: len(covers[b])):
            if time.perf_counter() >= deadline:
                return None
            price = min(slack[index] for index in covers[bit])
            if price > 1e-8:
                weights[bit] += price
                for index in covers[bit]:
                    slack[index] -= price
        lower = sum(weights.values())
        if lower >= ceiling - 1e-7:
            return None
        covered_prices = [costs[index] - slack[index] for index in range(len(columns))]
        for bit, indices in covers.items():
            indices.sort(key=lambda index: (slack[index], costs[index] / sizes[index], -sizes[index]))
        stack = [(remaining, 0, lower, ())]
        seen, answer = {}, None
        while stack and time.perf_counter() < deadline:
            rest, paid, bound, path = stack.pop()
            if paid + bound >= ceiling - 1e-7:
                continue
            if not rest:
                ceiling = paid
                answer = [columns[index] for index in path]
                continue
            if paid >= seen.get(rest, math.inf):
                continue
            seen[rest] = paid
            # Pick the request with the fewest usable columns; each column must be entirely contained in the uncovered set.
            choices, bits = None, rest
            extra_bound = 0.0
            while bits:
                if time.perf_counter() >= deadline:
                    return answer
                bit = bits & -bits
                bits ^= bit
                compatible = [index for index in covers[bit] if masks[index] & rest == masks[index]]
                if not compatible:
                    choices = []
                    break
                extra_bound = max(extra_bound, min(slack[index] for index in compatible))
                if paid + bound + extra_bound >= ceiling - 1e-7:
                    choices = []
                    break
                if choices is None or len(compatible) < len(choices):
                    choices = compatible
                    if len(choices) == 1:
                        break
            if not choices:
                continue
            for index in reversed(choices):
                new_paid = paid + costs[index]
                new_bound = bound - covered_prices[index]
                if new_paid + new_bound < ceiling - 1e-7:
                    stack.append((rest ^ masks[index], new_paid, new_bound, path + (index,)))
        return answer

    def recombine(self, deadline):
        """Anchored on a historical route, exactly re-cover the few active routes it intersects."""
        started = time.perf_counter()
        slice_size = max(0.0, deadline - started) / (2 + math.log1p(self.n))
        while time.perf_counter() < deadline:
            routes = [self.pool.get(route.coverage, route) for route in self.best]
            self.keep(routes)
            request_owner = [-1] * (2 * self.n)
            incumbent_keys = set()
            for rid, route in enumerate(routes):
                incumbent_keys.add(route.coverage)
                for i, mask in route.v:
                    if mask & 1:
                        request_owner[2 * (i - 1)] = rid
                    if mask & 2:
                        request_owner[2 * (i - 1) + 1] = rid
            bank, plans = list(self.pool.values()), []
            for index, anchor in enumerate(bank):
                if index % 32 == 0 and time.perf_counter() >= deadline:
                    return
                if anchor.coverage in incumbent_keys or len(anchor.v) == 1:
                    continue
                touched, rest = set(), anchor.coverage
                while rest:
                    bit = rest & -rest
                    rest ^= bit
                    touched.add(request_owner[bit.bit_length() - 1])
                    if len(touched) > self.patch_width:
                        break
                if len(touched) > self.patch_width:
                    continue
                union, total, credit = 0, 0, 0.0
                for rid in touched:
                    route = routes[rid]
                    union |= route.coverage
                    total += route.cost
                    credit += (route.cost * (route.coverage & anchor.coverage).bit_count()
                               / route.coverage.bit_count())
                if anchor.cost >= total:
                    continue
                score = credit - anchor.cost
                score += (0.15 * self.best_cost / max(1, self.n)) * self.rng.random()
                plans.append((score, anchor, touched, union, total))
            plans.sort(key=lambda plan: plan[0], reverse=True)
            improved = False
            for _, anchor, touched, union, total in plans:
                now = time.perf_counter()
                if now >= deadline:
                    return
                completion = self.cover_patch(union ^ anchor.coverage, total - anchor.cost,
                                              bank, min(deadline, now + slice_size))
                if completion is None:
                    continue
                if anchor.cost + sum(route.cost for route in completion) < total:
                    candidate = [route for rid, route in enumerate(routes) if rid not in touched]
                    candidate.extend([anchor] + completion)
                    self.keep(candidate)
                    improved = True
                    break
            if not improved:
                return

    def run(self):
        if self.n == 0 or time.perf_counter() >= self.end:
            return
        if any(self.d[i] > self.cap or self.p[i] > self.cap for i in range(1, self.n + 1)):
            return
        available = max(0.0, self.end - time.perf_counter())
        current = self.savings(min(self.end, time.perf_counter() + available * 0.12))
        polish_end = min(self.end, time.perf_counter() + available * 0.05)
        current = [self.polish(route, polish_end) for route in current]
        self.keep(current)
        current = self.cross_search(current, min(self.end, time.perf_counter() + available * 0.08))
        current_cost = self.keep(current)
        weights = [1.0] * 6
        last_improvement = time.perf_counter()
        cycle_started = last_improvement
        epoch = max(0.02, self.budget / (2.0 + math.sqrt(1.0 + self.budget)))
        next_recombine = cycle_started + epoch * 0.7
        while time.perf_counter() < self.end:
            now = time.perf_counter()
            if now >= next_recombine:
                old_best = self.best_cost
                pool_slice = min(self.budget * 0.025, (self.end - now) * 0.15)
                self.recombine(min(self.end, now + pool_slice))
                now = time.perf_counter()
                next_recombine = now + epoch
                if self.best_cost < old_best:
                    last_improvement = now
                    current, current_cost = self.best[:], self.best_cost
                if now >= self.end:
                    break
            stalled = (now - last_improvement) / epoch
            if now - cycle_started >= epoch:
                current, current_cost = self.best[:], self.best_cost
                cycle_started = now
            progress = min(1.0, (now - cycle_started) / epoch)
            temperature = (max(1.0, self.best_cost / max(1.0, math.sqrt(self.n)))
                           * (0.07 + 0.015 * math.log1p(self.budget)) * math.exp(-5.0 * progress))
            base = min(self.n * 0.24, math.sqrt(self.n) * (0.7 + 0.2 * math.sqrt(self.budget))
                       + self.n / max(1, len(current)))
            count = min(self.n, max(1, int(base * self.rng.uniform(0.45, 1.45)
                                           * (1.0 + 0.3 * min(2.0, stalled)))))
            if self.rng.random() < 1.0 / (8.0 + math.log1p(self.budget)):
                count = min(self.n, max(count, int(self.n * self.rng.uniform(0.25, 0.55))))
            draw = self.rng.random() * sum(weights)
            operator = len(weights) - 1
            for index, weight in enumerate(weights):
                draw -= weight
                if draw <= 0:
                    operator = index
                    break
            split_probability = min(0.7, 0.15 + 0.18 * stalled + 0.2 * self.rng.random())
            reduced, jobs = self.destroy(current, count, operator, split_probability)
            repair_started = time.perf_counter()
            candidate = self.repair(reduced, jobs, self.end)
            if candidate is None:
                break
            repair_elapsed = time.perf_counter() - repair_started
            unchanged = set(current)
            changed = [index for index, route in enumerate(candidate) if route not in unchanged]
            self.rng.shuffle(changed)
            local_end = min(self.end, time.perf_counter() + max(0.0005, repair_elapsed * 0.3))
            for index in changed:
                if time.perf_counter() >= local_end:
                    break
                candidate[index] = self.polish(candidate[index], local_end)
            value = sum(route.cost for route in candidate)
            old_best = self.best_cost
            if value < old_best or self.rng.random() < 0.08:
                local_slice = min(self.budget * 0.012 / max(1.0, math.log1p(self.n)),
                                  max(0.0, self.end - time.perf_counter()) * 0.12)
                candidate = self.cross_search(candidate, min(self.end, time.perf_counter() + local_slice))
                value = sum(route.cost for route in candidate)
            self.keep(candidate)
            if self.best_cost < old_best:
                last_improvement = time.perf_counter()
                reward = 7.0
            elif value < current_cost:
                reward = 3.0
            else:
                reward = 0.25
            difference = value - current_cost
            if difference <= 0 or self.rng.random() < math.exp(-difference / max(1e-9, temperature)):
                current, current_cost = candidate, value
                reward = max(reward, 0.7)
            weights[operator] = 0.92 * weights[operator] + 0.08 * reward

    def result(self):
        # Finally audit only the complete historical best; repairs left unfinished midway never enter this snapshot.
        counts_d, counts_p = [0] * (self.n + 1), [0] * (self.n + 1)
        valid = True
        for route in self.best:
            if not route.v or route.peak > self.cap or len(route.where) != len(route.v):
                valid = False
                break
            for i, mask in route.v:
                if not 1 <= i <= self.n or mask not in (1, 2, 3):
                    valid = False
                    break
                counts_d[i] += bool(mask & 1)
                counts_p[i] += bool(mask & 2)
        if (not valid or any(counts_d[i] != 1 or counts_p[i] != 1
                             for i in range(1, self.n + 1))):
            return {'routes': [[{'customer_id': i, 'serve_delivery': True,
                                'serve_pickup': True}] for i in range(1, self.n + 1)]}
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
        return engine.result()
    except BaseException:
        if engine is not None:
            try:
                return engine.result()
            except BaseException:
                pass
        return fallback
