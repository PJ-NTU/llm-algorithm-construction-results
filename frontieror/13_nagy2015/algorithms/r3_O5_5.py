"""Pickup-and-delivery large neighbourhood search enhanced by route-pool exact covering
Keeps capacity-interval caching and on-demand split-vehicle regret repair, recombining service coverage with a historical route pool, dual lower bounds and time-limited branching, always saving the complete best feasible solution.
"""
import bisect
import heapq
import math
import random
import time


class _Route:
    __slots__ = ('v', 'cost', 'delivery', 'pickup', 'loads', 'prefix',
                 'suffix', 'peak', 'low', 'where', 'dcum', 'pcum',
                 'negative_suffix', 'cache', 'polished', 'cover')

    def __init__(self, visits, engine):
        self.v = visits
        d, p, dist = engine.d, engine.p, engine.dist
        self.delivery = sum(d[i] for i, mask in visits if mask & 1)
        self.pickup = sum(p[i] for i, mask in visits if mask & 2)
        load = self.delivery
        loads = [load]
        cost, prev = 0, 0
        where = {}
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
        self.dcum, self.pcum = dcum, pcum
        self.negative_suffix = [-value for value in suffix]
        self.cache = {}
        self.polished = False
        self.cover = None


class _Task:
    __slots__ = ('i', 'mask', 'options', 'top', 'jitter', 'priority')

    def __init__(self, i, mask, depot_cost, jitter):
        self.i, self.mask, self.jitter = i, mask, jitter
        self.options = {-1: (depot_cost, -1, 0)}
        self.top = []
        self.priority = 0.0

    def refresh(self, depth):
        first = self.top[0][0]
        regret = sum(option[0] - first for option in self.top[1:])
        regret += max(0, depth - len(self.top)) * (self.top[-1][0] - first)
        self.priority = self.jitter * regret - 0.03 * first

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
            self.refresh(depth)


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
        self.order_cache = {}
        self.cache_limit = max(1, int(self.n * (5 + math.sqrt(self.budget)) * 8))
        self.effort = math.log1p(self.budget)
        self.pool = {}
        self.pool_index = [set() for _ in range(self.n + 1)]
        self.pool_limit = max(32, int(self.n * (6 + 3 * self.effort)))
        self.pool_width = max(2, int(2 + self.effort + math.sqrt(math.log1p(self.n))))

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
        existing = route.where.get(i)
        if existing is not None:
            if route.v[existing][1] & mask:
                return None
            if (route.prefix[existing] + extra_d <= self.cap and
                    route.suffix[existing + 1] + extra_p <= self.cap):
                return (0, -existing - 1)
            return None
        pd, sp = self.cap - extra_d, self.cap - extra_p
        # Prefix peaks are non-decreasing and suffix peaks non-increasing, so feasible insertion positions form a contiguous interval.
        left = bisect.bisect_left(route.negative_suffix, -sp)
        right = bisect.bisect_right(route.prefix, pd)
        best, position = math.inf, 0
        row, dist = self.dist[i], self.dist
        prev = route.v[left - 1][0] if 0 < left <= len(route.v) else 0
        for k in range(left, right):
            if k % 64 == 0 and time.perf_counter() >= self.end:
                return None
            nxt = route.v[k][0] if k < len(route.v) else 0
            delta = dist[prev][i] + row[nxt] - dist[prev][nxt]
            if delta < best:
                best, position = delta, k
            prev = nxt
        result = None if best == math.inf else (best, position)
        if len(route.cache) >= self.insertion_cache_limit:
            route.cache.clear()
        route.cache[key] = result
        return result

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
        now = time.perf_counter()
        build_end = now + max(0.0, deadline - now) * 0.55
        for i in range(1, self.n + 1):
            if time.perf_counter() >= build_end:
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
        key = tuple(sorted(route.v))
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
            task.refresh(depth)
            tasks.append(task)
        while tasks:
            if time.perf_counter() >= deadline:
                return None
            chosen, priority = 0, -math.inf
            for index, task in enumerate(tasks):
                if index % 64 == 0 and time.perf_counter() >= deadline:
                    return None
                score = task.priority
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

    def cover_of(self, route):
        if route.cover is None:
            covered = 0
            for i, mask in route.v:
                covered |= mask << (2 * (i - 1))
            route.cover = covered
        return route.cover

    def collect(self, routes):
        # Feasible routes are deduplicated by service set; a more expensive route with the same set cannot improve the exact cover.
        for route in routes:
            if time.perf_counter() >= self.end:
                return
            if route.cover is not None or len(route.v) < 2:
                continue
            key = self.cover_of(route)
            previous = self.pool.get(key)
            if previous is not None:
                if route.cost < previous.cost:
                    self.pool[key] = route
                continue
            if route.peak > self.cap or len(route.where) != len(route.v):
                continue
            self.pool[key] = route
            for i, mask in route.v:
                self.pool_index[i].add(key)
            if len(self.pool) > self.pool_limit:
                old_key = next(iter(self.pool))
                old = self.pool.pop(old_key)
                for i, mask in old.v:
                    self.pool_index[i].discard(old_key)

    def recombine_pool(self, source, deadline):
        # The original routes are a feasible upper bound. Branches not fully solved never replace the original solution.
        if len(source) < 2 or not self.pool or time.perf_counter() >= deadline:
            return None
        rng = self.rng
        now = time.perf_counter()
        prepare_end = now + (deadline - now) * 0.55
        owner_d, owner_p = [-1] * (self.n + 1), [-1] * (self.n + 1)
        shares = []
        for rid, route in enumerate(source):
            size = sum((mask & 1 != 0) + (mask & 2 != 0) for _, mask in route.v)
            shares.append(route.cost / max(1, size))
            for i, mask in route.v:
                if mask & 1:
                    owner_d[i] = rid
                if mask & 2:
                    owner_p[i] = rid
        reach = min(len(source), self.pool_width)
        selected, best_score = None, -math.inf
        entries = list(self.pool.values())
        sample_size = min(len(entries), max(1, int(reach * (5 + 3 * self.effort))))
        for route in rng.sample(entries, sample_size):
            if time.perf_counter() >= prepare_end:
                return None
            owners, allocated = set(), 0.0
            for i, mask in route.v:
                if mask & 1:
                    owners.add(owner_d[i])
                    allocated += shares[owner_d[i]]
                if mask & 2:
                    owners.add(owner_p[i])
                    allocated += shares[owner_p[i]]
            if len(owners) > reach:
                continue
            if len(owners) == 1 and self.cover_of(source[next(iter(owners))]) == route.cover:
                continue
            score = (allocated - route.cost + 0.04 * allocated) * rng.uniform(0.7, 1.3)
            if score > best_score:
                selected, best_score = owners, score
        if selected is None:
            selected = {rng.randrange(len(source))}
        want = rng.randint(max(2, len(selected)), reach)
        seed = rng.choice(source[rng.choice(tuple(selected))].v)[0]
        for i in [seed] + self.nearby(seed):
            for rid in (owner_d[i], owner_p[i]):
                if len(selected) >= want:
                    break
                selected.add(rid)
            if len(selected) >= want:
                break
        if len(selected) < want:
            ranked = heapq.nsmallest(want - len(selected),
                                     (rid for rid in range(len(source)) if rid not in selected),
                                     key=lambda rid: min(self.dist[seed][source[rid].v[0][0]],
                                                         self.dist[seed][source[rid].v[-1][0]]))
            selected.update(ranked)
        covered, upper = 0, 0
        candidates, pool_keys, customers = {}, set(), set()
        for rid in selected:
            route = source[rid]
            key = self.cover_of(route)
            covered |= key
            upper += route.cost
            candidates[key] = route
            customers.update(route.where)
        if upper <= 0:
            return None
        for i in customers:
            if time.perf_counter() >= prepare_end:
                return None
            pool_keys.update(self.pool_index[i])
        for key in pool_keys:
            if time.perf_counter() >= prepare_end:
                return None
            if key & covered == key:
                route = self.pool[key]
                old = candidates.get(key)
                if old is None or route.cost < old.cost:
                    candidates[key] = route
        # Add single-service and combined single-customer routes so that historical split-vehicle routes can also close into a complete cover.
        for i in customers:
            if time.perf_counter() >= prepare_end:
                return None
            mask = (covered >> (2 * (i - 1))) & 3
            for part in ((1, 2, 3) if mask == 3 else (mask,)):
                key = part << (2 * (i - 1))
                route = _Route([(i, part)], self)
                old = candidates.get(key)
                if old is None or route.cost < old.cost:
                    candidates[key] = route
        bit_list, remaining = [], covered
        while remaining:
            bit = remaining & -remaining
            bit_list.append(bit)
            remaining ^= bit
        bit_index = {bit: k for k, bit in enumerate(bit_list)}
        rows = list(candidates.values())
        masks, costs, row_vars = [], [], []
        by_var = [[] for _ in bit_list]
        dual = [math.inf] * len(bit_list)
        for rid, route in enumerate(rows):
            if time.perf_counter() >= prepare_end:
                return None
            mask = self.cover_of(route)
            masks.append(mask)
            costs.append(route.cost)
            variables, remaining = [], mask
            while remaining:
                bit = remaining & -remaining
                variables.append(bit_index[bit])
                remaining ^= bit
            row_vars.append(variables)
            unit = route.cost / len(variables)
            for var in variables:
                by_var[var].append(rid)
                if unit < dual[var]:
                    dual[var] = unit
        # The dual price sum of each column does not exceed the route cost, giving a valid lower bound for the exact cover.
        slack = [costs[rid] - sum(dual[k] for k in variables)
                 for rid, variables in enumerate(row_vars)]
        order = list(range(len(bit_list)))
        rng.shuffle(order)
        order.sort(key=lambda var: len(by_var[var]))
        for var in order:
            if time.perf_counter() >= deadline:
                return None
            lift = min(slack[rid] for rid in by_var[var])
            if lift > 1e-9:
                dual[var] += lift
                for rid in by_var[var]:
                    slack[rid] -= lift
        prices = [sum(dual[var] for var in variables) for variables in row_vars]
        lower = sum(dual)
        if lower >= upper - 1e-6:
            return None
        # Iterative branching avoids recursion-depth limits; the search is terminated only by the wall clock.
        stack = [(covered, 0, (), lower)]
        dominance, best_path = {covered: 0}, None
        memory_limit = max(256, int((self.n + len(rows)) * (4 + self.effort)))
        while stack and time.perf_counter() < deadline:
            remaining, value, path, bound = stack.pop()
            if value > dominance.get(remaining, math.inf) or value + bound >= upper - 1e-6:
                continue
            choices = None
            for var in order:
                if not remaining & bit_list[var]:
                    continue
                options = [rid for rid in by_var[var]
                           if masks[rid] & remaining == masks[rid]
                           and value + costs[rid] < upper]
                if choices is None or len(options) < len(choices):
                    choices = options
                    if len(choices) <= 1:
                        break
                if time.perf_counter() >= deadline:
                    break
            if not choices:
                continue
            choices.sort(key=lambda rid: (costs[rid] - prices[rid],
                                           costs[rid] / len(row_vars[rid])))
            for rid in reversed(choices):
                if time.perf_counter() >= deadline:
                    break
                new_value = value + costs[rid]
                new_bound = bound - prices[rid]
                if new_value + new_bound >= upper - 1e-6:
                    continue
                rest = remaining ^ masks[rid]
                if not rest:
                    upper, best_path = new_value, path + (rid,)
                    continue
                if new_value >= dominance.get(rest, math.inf):
                    continue
                if len(dominance) >= memory_limit:
                    dominance.clear()
                dominance[rest] = new_value
                stack.append((rest, new_value, path + (rid,), new_bound))
        if best_path is None:
            return None
        result = [route for rid, route in enumerate(source) if rid not in selected]
        result.extend(rows[rid] for rid in best_path)
        return result

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
        self.collect(current)
        weights = [1.0] * 6
        last_improvement = time.perf_counter()
        cycle_started = last_improvement
        epoch = max(0.002, self.budget / (2.0 + math.sqrt(1.0 + self.budget)))
        next_pool = time.perf_counter() + epoch * 0.30
        pool_credit = 1.0
        repair_average = 0.0
        while time.perf_counter() < self.end:
            now = time.perf_counter()
            if now >= next_pool:
                pool_slice = min(self.budget / (100 + 4 * math.sqrt(self.n)) * pool_credit,
                                 max(0.0, self.end - now) * 0.15)
                before = self.best_cost
                candidate = self.recombine_pool(self.best, min(self.end, now + pool_slice))
                if candidate is not None:
                    value = self.keep(candidate)
                    if value < current_cost:
                        current, current_cost = candidate, value
                if self.best_cost < before:
                    last_improvement = time.perf_counter()
                    pool_credit = min(1.8, pool_credit * 1.25)
                else:
                    pool_credit = max(0.45, pool_credit * 0.85)
                next_pool = time.perf_counter() + epoch * 0.30 / pool_credit
                continue
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
            split_probability = min(0.7, 0.15 + 0.18 * stalled + 0.2 * self.rng.random())
            reduced, jobs = self.destroy(current, count, operator, split_probability)
            repair_started = time.perf_counter()
            candidate = self.repair(reduced, jobs, self.end)
            if candidate is None:
                break
            repair_elapsed = time.perf_counter() - repair_started
            repair_average = (repair_elapsed if not repair_average else
                              0.8 * repair_average + 0.2 * repair_elapsed)
            unchanged = set(current)
            changed = [index for index, route in enumerate(candidate) if route not in unchanged]
            self.rng.shuffle(changed)
            local_end = min(self.end, time.perf_counter() + max(self.budget * 0.00005, repair_elapsed * 0.3))
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
            self.collect(candidate)
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
