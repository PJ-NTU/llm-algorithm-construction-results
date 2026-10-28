"""Adaptive large neighbourhood search enhanced with ejection chains
Combines savings construction, cached regret repair, local improvement of feasible routes and cross-route beam-search ejection chains, maintaining the complete best feasible solution within the wall-clock budget.
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
        self.d = [0.0] + [float(x) for x in raw_d]
        self.p = [0.0] + [float(x) for x in raw_p]
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
        self.best = [_Route([(i, 3)], self) for i in range(1, self.n + 1)]
        self.best_cost = sum(route.cost for route in self.best)
        self.order_cache = {}
        self.cache_limit = max(1, int(self.n * (5 + math.sqrt(self.budget)) * 8))
        effort = math.log1p(self.budget)
        self.chain_depth = max(2, int(2 + effort))
        self.beam_width = max(2, int(2 + 1.5 * math.sqrt(effort)))

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
            nxt = route.v[k][0] if k < len(route.v) else 0
            if route.prefix[k] <= pd and route.suffix[k] <= sp:
                delta = dist[prev][i] + row[nxt] - dist[prev][nxt]
                if delta < best:
                    best, position = delta, k
            prev = nxt
        return None if best == math.inf else (best, position)

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
        if cached is not None and cached.cost < route.cost:
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
                    targets = {b for j in neighbors for b, _ in owners[j] if b != a}
                    move = None
                    gain = 0
                    for b in targets:
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

    def eject(self, routes, deadline):
        # The state holds only one not-yet-placed indivisible request; all other routes always satisfy capacity.
        # Only complete solutions with the request fully placed and lower total distance are handed to the main search.
        if len(routes) < 2 or time.perf_counter() >= deadline:
            return None
        dist, rng = self.dist, self.rng
        owners = self.owners(routes)
        targets_cache, removed_cache, price_cache = {}, {}, {}
        beam_width = min(self.beam_width, 2 * self.n)
        depth_limit = min(len(routes), self.chain_depth)
        split_moves = rng.random() < 0.3

        def targets(c):
            saved = targets_cache.get(c)
            if saved is None:
                seen, saved = set(), []
                for j in [c] + self.nearby(c):
                    for rid, _ in owners[j]:
                        if rid not in seen:
                            seen.add(rid)
                            saved.append(rid)
                if len(routes) <= self.width:
                    saved.extend(r for r in range(len(routes)) if r not in seen)
                targets_cache[c] = saved
            return saved

        def relaxed_price(c, excluded):
            # Ignore capacity and use insertion costs on both sides of nearby customers as the beam-ordering estimate; this is not a pruning proof.
            key = c, excluded
            value = price_cache.get(key)
            if value is not None:
                return value
            row = dist[c]
            value = row[0] + dist[0][c]
            for j in [c] + self.nearby(c):
                for rid, pos in owners[j]:
                    if rid == excluded:
                        continue
                    if c == j:
                        value = min(value, 0)
                        continue
                    v = routes[rid].v
                    left = v[pos - 1][0] if pos else 0
                    right = v[pos + 1][0] if pos + 1 < len(v) else 0
                    value = min(value, row[left] + row[j] - dist[left][j],
                                row[j] + row[right] - dist[j][right])
            price_cache[key] = value
            return value

        def remove(rid, pos, mask):
            key = rid, pos, mask
            saved = removed_cache.get(key)
            if saved is None:
                old = routes[rid]
                v = old.v[:]
                c, original = v[pos]
                rest = original & ~mask
                if rest:
                    v[pos] = c, rest
                else:
                    del v[pos]
                small = _Route(v, self)
                saved = small, small.cost - old.cost
                removed_cache[key] = saved
            return saved

        def finish(changes, rid, c, mask, position):
            result = routes[:]
            for r, route in changes.items():
                result[r] = route
            if rid < 0:
                result.append(_Route([(c, mask)], self))
            else:
                result[rid] = self.inserted(result[rid], c, mask, position)
            return [route for route in result if route.v]

        starts = []
        for rid, route in enumerate(routes):
            for pos, (c, mask) in enumerate(route.v):
                left = route.v[pos - 1][0] if pos else 0
                right = route.v[pos + 1][0] if pos + 1 < len(route.v) else 0
                saving = dist[left][c] + dist[c][right] - dist[left][right]
                pressure = max(self.d[c] if mask & 1 else 0,
                               self.p[c] if mask & 2 else 0) / max(self.cap, 1e-12)
                score = (saving + 0.1 * dist[0][c] * pressure) * (0.35 + rng.random())
                starts.append((score, rid, pos, mask))
                if split_moves and mask == 3:
                    starts.append((score * rng.random(), rid, pos, rng.choice((1, 2))))
        starts.sort(reverse=True)
        for _, source, at, first_mask in starts:
            if time.perf_counter() >= deadline:
                break
            c = routes[source].v[at][0]
            small, change = remove(source, at, first_mask)
            states = [(change, c, first_mask, {source: small})]
            depth = 1
            while states and depth <= depth_limit and time.perf_counter() < deadline:
                next_states = []
                for delta, pending, mask, edits in states:
                    if time.perf_counter() >= deadline:
                        break
                    nearby = targets(pending)
                    final_targets = list(dict.fromkeys(list(edits) + nearby))
                    best_gain, closure = -1e-9, None
                    singleton = dist[0][pending] + dist[pending][0]
                    if delta + singleton < best_gain:
                        best_gain = delta + singleton
                        closure = (-1, 0)
                    for rid in final_targets:
                        if time.perf_counter() >= deadline:
                            break
                        target = edits.get(rid, routes[rid])
                        slot = self.insertion(target, pending, mask)
                        if slot is not None and delta + slot[0] < best_gain:
                            best_gain = delta + slot[0]
                            closure = rid, slot[1]
                    if closure is not None:
                        return finish(edits, closure[0], pending, mask, closure[1])
                    if depth >= depth_limit:
                        continue
                    qd = self.d[pending] if mask & 1 else 0.0
                    qp = self.p[pending] if mask & 2 else 0.0
                    for rid in nearby:
                        if time.perf_counter() >= deadline:
                            break
                        # Each extension visits a route not yet changed, preventing cycles within the chain.
                        if rid in edits:
                            continue
                        target = routes[rid]
                        for pos, (outgoing, outmask) in enumerate(target.v):
                            if time.perf_counter() >= deadline:
                                break
                            variants = (outmask, 1, 2) if split_moves and outmask == 3 else (outmask,)
                            for take in variants:
                                od = self.d[outgoing] if take & 1 else 0.0
                                op = self.p[outgoing] if take & 2 else 0.0
                                if (target.delivery - od + qd > self.cap or
                                        target.pickup - op + qp > self.cap):
                                    continue
                                reduced, removal = remove(rid, pos, take)
                                slot = self.insertion(reduced, pending, mask)
                                if slot is None:
                                    continue
                                next_delta = delta + removal + slot[0]
                                # First try closing back to the starting route; a two-step chain is an exchange with repositioning.
                                closing = self.insertion(edits[source], outgoing, take)
                                if closing is not None and next_delta + closing[0] < -1e-9:
                                    extended = edits.copy()
                                    extended[rid] = self.inserted(reduced, pending, mask, slot[1])
                                    return finish(extended, source, outgoing, take, closing[1])
                                score = next_delta + relaxed_price(outgoing, rid)
                                if len(next_states) >= beam_width and score >= next_states[-1][0]:
                                    continue
                                extended = edits.copy()
                                extended[rid] = self.inserted(reduced, pending, mask, slot[1])
                                next_states.append((score, next_delta, outgoing, take, extended))
                                next_states.sort(key=lambda state: state[0])
                                if len(next_states) > beam_width:
                                    next_states.pop()
                states = [state[1:] for state in next_states]
                depth += 1
        return None

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
        jobs = []
        for i, mask in missing.items():
            if mask == 3 and rng.random() < split_probability:
                jobs.extend(((i, 1), (i, 2)))
            else:
                jobs.append((i, mask))
        return reduced, jobs

    def repair(self, routes, jobs, deadline):
        routes = routes[:]
        depth = 2 if self.rng.random() < 0.5 else max(2, min(4, int(2 + math.log1p(self.budget) / 2)))
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
                first = task.top[0][0]
                regret = sum(option[0] - first for option in task.top[1:])
                if len(task.top) < depth:
                    regret += (depth - len(task.top)) * (task.top[-1][0] - first)
                score = task.jitter * regret - 0.03 * first
                if score > priority:
                    chosen, priority = index, score
            task = tasks.pop(chosen)
            _, rid, position = task.top[0]
            if rid == -1:
                rid = len(routes)
                routes.append(_Route([(task.i, task.mask)], self))
            else:
                routes[rid] = self.inserted(routes[rid], task.i, task.mask, position)
            route = routes[rid]
            for index, task in enumerate(tasks):
                option = self.insertion(route, task.i, task.mask)
                task.update(rid, option, depth)
                if index % 32 == 0 and time.perf_counter() >= deadline:
                    return None
        return routes

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
        weights = [1.0] * 5
        last_improvement = time.perf_counter()
        cycle_started = last_improvement
        epoch = max(0.02, self.budget / (2.0 + math.sqrt(1.0 + self.budget)))
        search_started = last_improvement
        next_chain = search_started
        chain_spent = 0.0
        chain_slice = self.budget / (95.0 + 3.0 * math.sqrt(self.n))
        chain_interval = self.budget / (32.0 + math.sqrt(self.n) + math.log1p(self.budget))
        chain_share = 0.12 + 0.06 * math.log1p(self.budget) / (1.0 + math.log1p(self.budget))
        while time.perf_counter() < self.end:
            now = time.perf_counter()
            if (now >= next_chain and
                    chain_spent < chain_share * (now - search_started + epoch * 0.15)):
                source = self.best if self.rng.random() < 0.75 else current
                before = self.best_cost
                candidate = self.eject(source, min(self.end, now + chain_slice))
                chain_spent += time.perf_counter() - now
                next_chain = time.perf_counter() + chain_interval
                if candidate is not None:
                    value = self.keep(candidate)
                    if self.best_cost < before:
                        last_improvement = time.perf_counter()
                    if value <= current_cost:
                        current, current_cost = candidate, value
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
            changed = [index for index, route in enumerate(candidate) if route not in current]
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

    def audit(self):
        seen_d, seen_p = [0] * (self.n + 1), [0] * (self.n + 1)
        for route in self.best:
            seen = set()
            load = sum(self.d[i] for i, mask in route.v if mask & 1)
            if not route.v or load > self.cap:
                return False
            for i, mask in route.v:
                if not 1 <= i <= self.n or i in seen or mask not in (1, 2, 3):
                    return False
                seen.add(i)
                if mask & 1:
                    seen_d[i] += 1
                    load -= self.d[i]
                if mask & 2:
                    seen_p[i] += 1
                    load += self.p[i]
                if load > self.cap or load < -1e-9 * max(1.0, self.capacity):
                    return False
        return all(seen_d[i] == seen_p[i] == 1 for i in range(1, self.n + 1))

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
        return engine.result() if engine.audit() else fallback
    except BaseException:
        if engine is not None:
            try:
                if engine.audit():
                    return engine.result()
            except BaseException:
                pass
        return fallback
