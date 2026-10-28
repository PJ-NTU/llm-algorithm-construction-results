"""Giant-tour splitting and capacity-penalty-guided neighbourhood search
Builds routes by dynamic splitting and keeps searching with request transfers, short-chain exchanges, route reconnection and edge penalties, saving only complete feasible solutions.
"""
import heapq
import math
import random
import time


_ZERO = (0.0, 0.0, 0.0, 0.0)


def _join(a, b):
    return (a[0] + b[0], a[1] + b[1],
            max(a[2] + b[0], b[2] + a[1]),
            max(b[3] + a[0], a[3] + b[1]))


def _reverse(a):
    return (a[0], a[1], a[3], a[2])


def _over2(a, b, cap):
    return max(0.0, min(max(a[2] + b[0], b[2] + a[1]),
                        max(b[3] + a[0], a[3] + b[1])) - cap)


def _over3(a, b, c, cap):
    forward = max(a[2] + b[0] + c[0], b[2] + a[1] + c[0],
                  c[2] + a[1] + b[1])
    backward = max(c[3] + b[0] + a[0], b[3] + c[1] + a[0],
                   a[3] + c[1] + b[1])
    return max(0.0, min(forward, backward) - cap)


class _Tour:
    __slots__ = ('v', 'ids', 'where', 'pre', 'suf', 'cost', 'score',
                 'over', 'stamp', 'delta', 'lo', 'hi')

    def __init__(self, visits, search, stamp=0):
        atom = search.atom
        v = list(visits)
        pre = [_ZERO]
        for item in v:
            pre.append(_join(pre[-1], atom[item]))
        # Under symmetric distances reverse travel does not change the cost, so always pick the direction with the smaller peak.
        if pre[-1][3] < pre[-1][2] - search.eps:
            v.reverse()
            pre = [_ZERO]
            for item in v:
                pre.append(_join(pre[-1], atom[item]))
        self.v = v
        self.ids = [item >> 2 for item in v]
        self.where = {i: pos for pos, i in enumerate(self.ids)}
        self.pre = pre
        self.suf = [_ZERO] * (len(v) + 1)
        for pos in range(len(v) - 1, -1, -1):
            self.suf[pos] = _join(atom[v[pos]], self.suf[pos + 1])
        self.over = max(0.0, pre[-1][2] - search.cap)
        self.cost = 0
        self.score = 0.0
        last = 0
        for i in self.ids:
            self.cost += search.dist[last][i]
            self.score += search.edge[last][i]
            last = i
        self.cost += search.dist[last][0]
        self.score += search.edge[last][0]
        self.stamp = stamp
        self.delta = [s[1] - s[0] for s in pre]
        self.lo = None
        self.hi = None

    def segment(self, left, right, atom):
        if left == right:
            return _ZERO
        if right == left + 1:
            return atom[self.v[left]]
        if right == left + 2:
            return _join(atom[self.v[left]], atom[self.v[left + 1]])
        if left == 0:
            return self.pre[right]
        if right == len(self.v):
            return self.suf[left]
        if self.lo is None:
            self.lo, self.hi = [self.delta], [self.delta]
            span = 1
            size = len(self.delta)
            while span * 2 <= size:
                low, high = self.lo[-1], self.hi[-1]
                self.lo.append([min(low[k], low[k + span])
                                for k in range(size - 2 * span + 1)])
                self.hi.append([max(high[k], high[k + span])
                                for k in range(size - 2 * span + 1)])
                span *= 2
        level = (right - left + 1).bit_length() - 1
        start = right + 1 - (1 << level)
        lower = min(self.lo[level][left], self.lo[level][start])
        upper = max(self.hi[level][left], self.hi[level][start])
        d = self.pre[right][0] - self.pre[left][0]
        p = self.pre[right][1] - self.pre[left][1]
        return (d, p, d + upper - self.delta[left],
                p + self.delta[left] - lower)


class _Search:
    def __init__(self, instance, budget, started):
        self.n = int(instance['n'])
        self.dist = instance['distance_matrix']
        raw_d, raw_p = instance['delivery_demands'], instance['pickup_demands']
        self.d = [0.0] + list(raw_d) if len(raw_d) == self.n else list(raw_d)
        self.p = [0.0] + list(raw_p) if len(raw_p) == self.n else list(raw_p)
        self.capacity = float(instance['capacity'])
        self.eps = 1e-10 * max(1.0, abs(self.capacity))
        self.cap = self.capacity + self.eps
        self.started, self.budget = started, budget
        reserve = min(budget * 0.035, 0.2) + min(budget * 0.015, self.n * 0.000006)
        self.end = started + max(0.0, budget - reserve)
        self.rng = random.Random(173189 + 97 * self.n)
        self.strength = math.log1p(budget)
        self.width = min(max(0, self.n - 1), max(5, int(
            3 + 2 * math.sqrt(1 + self.strength)
            + math.sqrt(self.n) * (0.3 + 0.1 * self.strength))))
        self.depth = max(1, min(4, 1 + int(self.strength / 1.5)))
        self.near = {}
        self.atom = [_ZERO] * (4 * (self.n + 1))
        for i in range(1, self.n + 1):
            d, p = self.d[i], self.p[i]
            self.atom[4 * i + 1] = (d, 0.0, d, d)
            self.atom[4 * i + 2] = (0.0, p, p, p)
            self.atom[4 * i + 3] = (d, p, max(d, p), max(d, p))
        self.edge = [list(row) for row in self.dist]
        self.penalties = {}
        radial = sum(self.dist[0][1:self.n + 1]) / max(1, self.n)
        self.base_penalty = max(1e-8, 2.0 * radial / max(self.capacity, 1e-12))
        self.capacity_penalty = self.base_penalty * (1.5 + math.sqrt(1 + self.strength))
        self.tick = 1
        self.routes = [_Tour([4 * i + 3], self) for i in range(1, self.n + 1)]
        self.best = self.routes[:]
        self.best_cost = sum(r.cost for r in self.best)
        self.total_cost = self.best_cost
        self.total_over = sum(r.over for r in self.routes)
        self.rd, self.rp = [-1] * (self.n + 1), [-1] * (self.n + 1)
        self.pd, self.pp = [0] * (self.n + 1), [0] * (self.n + 1)
        self.scanned = [0] * (self.n + 1)
        self.empty = -1

    def neighbors(self, i):
        if i not in self.near:
            row = self.dist[i]
            self.near[i] = heapq.nsmallest(
                self.width, (j for j in range(1, self.n + 1) if j != i),
                key=row.__getitem__)
        return self.near[i]

    def remember(self):
        if self.total_over <= self.eps and self.total_cost < self.best_cost:
            if all(r.over == 0.0 for r in self.routes):
                self.best_cost = self.total_cost
                self.best = [r for r in self.routes if r.v]

    def index_route(self, rid):
        for pos, item in enumerate(self.routes[rid].v):
            i, mask = item >> 2, item & 3
            if mask & 1:
                self.rd[i], self.pd[i] = rid, pos
            if mask & 2:
                self.rp[i], self.pp[i] = rid, pos

    def install(self, routes):
        self.tick += 1
        self.routes = [r for r in routes if r.v]
        self.routes.append(_Tour([], self, self.tick))
        self.empty = len(self.routes) - 1
        for rid, route in enumerate(self.routes):
            route.stamp = self.tick
            last, score = 0, 0.0
            for i in route.ids:
                score += self.edge[last][i]
                last = i
            route.score = score + self.edge[last][0]
            self.index_route(rid)
        self.total_cost = sum(r.cost for r in self.routes)
        self.total_over = sum(r.over for r in self.routes)
        self.scanned = [0] * (self.n + 1)
        self.remember()

    def split(self, order, deadline):
        n = len(order)
        value, previous, backward = [math.inf] * (n + 1), [-1] * (n + 1), [False] * (n + 1)
        value[0] = 0
        dist, cap = self.dist, self.cap
        for left in range(n):
            if time.perf_counter() >= deadline:
                return None
            if not math.isfinite(value[left]):
                continue
            d, p, delta, high, low, length, last = 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0
            for right in range(left, n):
                if (right - left) % 64 == 0 and time.perf_counter() >= deadline:
                    return None
                i = order[right]
                d += self.d[i]
                p += self.p[i]
                if d > cap or p > cap:
                    break
                delta += self.p[i] - self.d[i]
                high, low = max(high, delta), min(low, delta)
                length += dist[last][i]
                last = i
                forward_peak, reverse_peak = d + high, p - low
                if min(forward_peak, reverse_peak) <= cap:
                    candidate = value[left] + length + dist[i][0]
                    if candidate < value[right + 1]:
                        value[right + 1] = candidate
                        previous[right + 1] = left
                        backward[right + 1] = reverse_peak < forward_peak
        if previous[n] < 0:
            return None
        answer, pos = [], n
        while pos:
            left = previous[pos]
            visits = [4 * i + 3 for i in order[left:pos]]
            if backward[pos]:
                visits.reverse()
            answer.append(_Tour(visits, self))
            pos = left
        return answer

    def construct(self, deadline):
        n, dist = self.n, self.dist
        anchor = max(range(1, n + 1), key=dist[0].__getitem__)
        radius = float(dist[0][anchor])
        if radius > 0:
            x = [0.0] * (n + 1)
            y2 = [0.0] * (n + 1)
            for i in range(1, n + 1):
                x[i] = (dist[0][i] ** 2 + radius ** 2 - dist[anchor][i] ** 2) / (2 * radius)
                y2[i] = max(0.0, dist[0][i] ** 2 - x[i] ** 2)
            other = max(range(1, n + 1), key=y2.__getitem__)
            height = math.sqrt(y2[other])
            if height > 1e-12:
                angles = [0.0] * (n + 1)
                for i in range(1, n + 1):
                    y = (dist[0][i] ** 2 + dist[0][other] ** 2
                         - dist[other][i] ** 2 - 2 * x[i] * x[other]) / (2 * height)
                    angles[i] = math.atan2(y, x[i])
                sweep = sorted(range(1, n + 1), key=lambda i: (angles[i], dist[0][i]))
            else:
                sweep = sorted(range(1, n + 1), key=lambda i: x[i])
        else:
            sweep = list(range(1, n + 1))
        routes = self.split(sweep, deadline)
        if routes is not None:
            self.install(routes)
        # Nearest-neighbour sequences and rotated polar-angle sequences alternate in splitting, with no cap on the number of constructions.
        use_nearest = True
        while time.perf_counter() < deadline:
            if use_nearest:
                remaining = set(range(1, n + 1))
                last = self.rng.choice(sweep)
                order = [last]
                remaining.remove(last)
                while remaining and time.perf_counter() < deadline:
                    nearby = [j for j in self.neighbors(last) if j in remaining]
                    if nearby:
                        last = nearby[0]
                    else:
                        last = min(remaining, key=dist[last].__getitem__)
                    remaining.remove(last)
                    order.append(last)
                if remaining:
                    break
            else:
                cut = self.rng.randrange(n)
                order = sweep[cut:] + sweep[:cut]
            candidate = self.split(order, deadline)
            if candidate is not None:
                cost = sum(r.cost for r in candidate)
                if cost < self.best_cost:
                    self.install(candidate)
            use_nearest = not use_nearest
        self.install(self.best)

    def commit(self, a, va, b=-1, vb=None):
        if len({item >> 2 for item in va}) != len(va):
            return False
        if b >= 0 and len({item >> 2 for item in vb}) != len(vb):
            return False
        self.tick += 1
        new_a = _Tour(va, self, self.tick)
        old_a = self.routes[a]
        old_score = old_a.score + self.capacity_penalty * old_a.over
        new_score = new_a.score + self.capacity_penalty * new_a.over
        new_b = None
        if b >= 0:
            old_b = self.routes[b]
            new_b = _Tour(vb, self, self.tick)
            old_score += old_b.score + self.capacity_penalty * old_b.over
            new_score += new_b.score + self.capacity_penalty * new_b.over
        if new_score >= old_score - 1e-8:
            return False
        self.total_cost += new_a.cost - old_a.cost
        self.total_over += new_a.over - old_a.over
        self.routes[a] = new_a
        self.index_route(a)
        if b >= 0:
            self.total_cost += new_b.cost - old_b.cost
            self.total_over += new_b.over - old_b.over
            self.routes[b] = new_b
            self.index_route(b)
        if self.empty == a or self.empty == b or self.empty < 0:
            self.empty = next((k for k, route in enumerate(self.routes) if not route.v), -1)
            if self.empty < 0:
                self.empty = len(self.routes)
                self.routes.append(_Tour([], self, self.tick))
        self.remember()
        return True

    def intra(self, rid, x, y):
        route = self.routes[rid]
        if x == y or len(route.v) < 2:
            return False
        v, ids, edge = route.v, route.ids, self.edge
        pre, suf, atom, cap = route.pre, route.suf, self.atom, self.cap
        penalty = self.capacity_penalty
        best, action = -1e-8, None
        left, right = min(x, y), max(x, y) + 1
        before = ids[left - 1] if left else 0
        after = ids[right] if right < len(v) else 0
        block = route.segment(left, right, atom)
        delta = (edge[before][ids[right - 1]] + edge[ids[left]][after]
                 - edge[before][ids[left]] - edge[ids[right - 1]][after])
        over = _over3(pre[left], _reverse(block), suf[right], cap)
        change = delta + penalty * (over - route.over)
        if change < best:
            best, action = change, (0, left, right)
        for size in range(1, min(self.depth, len(v) - x) + 1):
            block = route.segment(x, x + size, atom)
            first, last = ids[x], ids[x + size - 1]
            before = ids[x - 1] if x else 0
            after = ids[x + size] if x + size < len(v) else 0
            removal = edge[before][after] - edge[before][first] - edge[last][after]
            for pos in (y, y + 1):
                if x <= pos <= x + size:
                    continue
                prev = ids[pos - 1] if pos else 0
                nxt = ids[pos] if pos < len(v) else 0
                delta = removal + edge[prev][first] + edge[last][nxt] - edge[prev][nxt]
                if pos < x:
                    tail = _join(route.segment(pos, x, atom), suf[x + size])
                    over = _over3(pre[pos], block, tail, cap)
                else:
                    head = _join(pre[x], route.segment(x + size, pos, atom))
                    over = _over3(head, block, suf[pos], cap)
                change = delta + penalty * (over - route.over)
                if change < best:
                    best, action = change, (1, x, size, pos)
        if action is None:
            return False
        if action[0] == 0:
            _, left, right = action
            candidate = v[:left] + v[left:right][::-1] + v[right:]
        else:
            _, left, size, pos = action
            block = v[left:left + size]
            candidate = v[:left] + v[left + size:]
            if pos > left:
                pos -= size
            candidate = candidate[:pos] + block + candidate[pos:]
        return self.commit(rid, candidate)

    def cross(self, a, x, b, y):
        ra, rb = self.routes[a], self.routes[b]
        va, vb, ia, ib = ra.v, rb.v, ra.ids, rb.ids
        la, lb = len(va), len(vb)
        if not la:
            return False
        edge, atom, cap, penalty = self.edge, self.atom, self.cap, self.capacity_penalty
        ap, ass, bp, bs = ra.pre, ra.suf, rb.pre, rb.suf
        old_over = ra.over + rb.over
        best, action = -1e-8, None
        i, mask = ia[x], va[x] & 3
        before = ia[x - 1] if x else 0
        after = ia[x + 1] if x + 1 < la else 0
        removal = edge[before][after] - edge[before][i] - edge[i][after]
        existing = rb.where.get(i, -1)
        positions = (0,) if not lb else (y, y + 1)
        masks = (mask, 1, 2) if mask == 3 else (mask,)
        for taken in masks:
            rest = mask ^ taken
            over_a = _over3(ap[x], atom[4 * i + rest], ass[x + 1], cap)
            delta_a = removal if not rest else 0
            if existing >= 0:
                if vb[existing] & taken:
                    continue
                merged = vb[existing] | taken
                over_b = _over3(bp[existing], atom[merged], bs[existing + 1], cap)
                change = delta_a + penalty * (over_a + over_b - old_over)
                if change < best:
                    best, action = change, (0, taken, existing, True)
            else:
                block = atom[4 * i + taken]
                for pos in positions:
                    prev = ib[pos - 1] if pos else 0
                    nxt = ib[pos] if pos < lb else 0
                    delta_b = edge[prev][i] + edge[i][nxt] - edge[prev][nxt]
                    over_b = _over3(bp[pos], block, bs[pos], cap)
                    change = delta_a + delta_b + penalty * (over_a + over_b - old_over)
                    if change < best:
                        best, action = change, (0, taken, pos, False)
        for size in range(2, min(self.depth, la - x) + 1):
            if any(i in rb.where for i in ia[x:x + size]):
                continue
            block = ra.segment(x, x + size, atom)
            over_a = _over2(ap[x], ass[x + size], cap)
            first, last = ia[x], ia[x + size - 1]
            after = ia[x + size] if x + size < la else 0
            delta_a = edge[before][after] - edge[before][first] - edge[last][after]
            for pos in positions:
                prev = ib[pos - 1] if pos else 0
                nxt = ib[pos] if pos < lb else 0
                for rev in (False, True):
                    f, l = (last, first) if rev else (first, last)
                    summary = _reverse(block) if rev else block
                    over_b = _over3(bp[pos], summary, bs[pos], cap)
                    delta = delta_a + edge[prev][f] + edge[l][nxt] - edge[prev][nxt]
                    change = delta + penalty * (over_a + over_b - old_over)
                    if change < best:
                        best, action = change, (1, size, pos, rev)
        if lb:
            # When exchanging short chains only the four boundary arcs are recomputed; internal arcs cancel between the two routes.
            sizes = [(1, 1)]
            if self.depth > 1:
                sizes.extend((k, 1) for k in range(2, min(self.depth, la - x) + 1))
                sizes.extend((1, k) for k in range(2, min(self.depth, lb - y) + 1))
            for size_a, size_b in sizes:
                enda, endb = x + size_a, y + size_b
                if enda > la or endb > lb:
                    continue
                if any(j in rb.where and not y <= rb.where[j] < endb for j in ia[x:enda]):
                    continue
                if any(j in ra.where and not x <= ra.where[j] < enda for j in ib[y:endb]):
                    continue
                block_a, block_b = ra.segment(x, enda, atom), rb.segment(y, endb, atom)
                al, ar = (ia[x - 1] if x else 0), (ia[enda] if enda < la else 0)
                bl, br = (ib[y - 1] if y else 0), (ib[endb] if endb < lb else 0)
                af, az, bf, bz = ia[x], ia[enda - 1], ib[y], ib[endb - 1]
                delta = (edge[al][bf] + edge[bz][ar] + edge[bl][af] + edge[az][br]
                         - edge[al][af] - edge[az][ar] - edge[bl][bf] - edge[bz][br])
                over_a = _over3(ap[x], block_b, ass[enda], cap)
                over_b = _over3(bp[y], block_a, bs[endb], cap)
                change = delta + penalty * (over_a + over_b - old_over)
                if change < best:
                    best, action = change, (2, size_a, size_b)
            common = ra.where.keys() & rb.where.keys()
            for acut in (x, x + 1):
                al = ia[acut - 1] if acut else 0
                ar = ia[acut] if acut < la else 0
                for bcut in (y, y + 1):
                    bl = ib[bcut - 1] if bcut else 0
                    br = ib[bcut] if bcut < lb else 0
                    removed = edge[al][ar] + edge[bl][br]
                    over_a = _over2(ap[acut], bs[bcut], cap)
                    over_b = _over2(bp[bcut], ass[acut], cap)
                    change = (edge[al][br] + edge[bl][ar] - removed
                              + penalty * (over_a + over_b - old_over))
                    if change < best and all((ra.where[j] < acut) == (rb.where[j] < bcut) for j in common):
                        best, action = change, (3, acut, bcut)
                    over_a = _over2(ap[acut], _reverse(bp[bcut]), cap)
                    over_b = _over2(_reverse(ass[acut]), bs[bcut], cap)
                    change = (edge[al][bl] + edge[ar][br] - removed
                              + penalty * (over_a + over_b - old_over))
                    if change < best and all((ra.where[j] < acut) != (rb.where[j] < bcut) for j in common):
                        best, action = change, (4, acut, bcut)
        if action is None:
            return False
        kind = action[0]
        if kind == 0:
            _, taken, pos, merge = action
            rest = mask ^ taken
            new_a = va[:x] + ([4 * i + rest] if rest else []) + va[x + 1:]
            new_b = vb[:]
            if merge:
                new_b[pos] |= taken
            else:
                new_b.insert(pos, 4 * i + taken)
        elif kind == 1:
            _, size, pos, rev = action
            block = va[x:x + size]
            if rev:
                block = block[::-1]
            new_a = va[:x] + va[x + size:]
            new_b = vb[:pos] + block + vb[pos:]
        elif kind == 2:
            _, size_a, size_b = action
            new_a = va[:x] + vb[y:y + size_b] + va[x + size_a:]
            new_b = vb[:y] + va[x:x + size_a] + vb[y + size_b:]
        elif kind == 3:
            _, acut, bcut = action
            new_a, new_b = va[:acut] + vb[bcut:], vb[:bcut] + va[acut:]
        else:
            _, acut, bcut = action
            new_a, new_b = va[:acut] + vb[:bcut][::-1], va[acut:][::-1] + vb[bcut:]
        return self.commit(a, new_a, b, new_b)

    def sweep(self):
        order = list(range(1, self.n + 1))
        self.rng.shuffle(order)
        changed = False
        for i in order:
            if time.perf_counter() >= self.end:
                break
            stamp = self.tick
            moved = False
            sources = [(self.rd[i], self.pd[i])]
            if self.rp[i] != self.rd[i]:
                sources.append((self.rp[i], self.pp[i]))
            neighbors = [i] + self.neighbors(i)
            for a, x in sources:
                if moved:
                    break
                for j in neighbors:
                    if time.perf_counter() >= self.end:
                        return changed
                    targets = [(self.rd[j], self.pd[j])]
                    if self.rp[j] != self.rd[j]:
                        targets.append((self.rp[j], self.pp[j]))
                    for b, y in targets:
                        if max(self.routes[a].stamp, self.routes[b].stamp) <= self.scanned[i]:
                            continue
                        if a == b:
                            moved = self.intra(a, x, y)
                        else:
                            moved = self.cross(a, x, b, y)
                        if moved:
                            break
                    if moved:
                        break
                if not moved and self.empty != a and time.perf_counter() < self.end:
                    moved = self.cross(a, x, self.empty, 0)
            self.scanned[i] = stamp
            changed = changed or moved
        return changed

    def invalidate(self):
        self.tick += 1
        self.scanned = [0] * (self.n + 1)
        for route in self.routes:
            route.stamp = self.tick

    def guide(self):
        candidates = {}
        for route in self.routes:
            last = 0
            for nxt in route.ids + [0]:
                if last != nxt:
                    key = (min(last, nxt), max(last, nxt))
                    candidates[key] = self.dist[last][nxt] / (1.0 + self.penalties.get(key, 0))
                last = nxt
        count = max(1, int(1 + self.strength / 2))
        chosen = heapq.nlargest(count, candidates, key=candidates.__getitem__)
        used = sum(bool(r.v) for r in self.routes)
        weight = (0.04 + 0.04 * self.strength) * max(1.0, self.best_cost / max(1, self.n + used))
        for i, j in chosen:
            self.penalties[(i, j)] = self.penalties.get((i, j), 0) + 1
            value = self.dist[i][j] + weight * self.penalties[(i, j)]
            self.edge[i][j] = self.edge[j][i] = value
        for route in self.routes:
            last, score = 0, 0.0
            for nxt in route.ids:
                score += self.edge[last][nxt]
                last = nxt
            route.score = score + self.edge[last][0]
        self.invalidate()

    def run(self):
        if not self.n or time.perf_counter() >= self.end:
            return
        if any(self.d[i] > self.cap or self.p[i] > self.cap for i in range(1, self.n + 1)):
            return
        remaining = max(0.0, self.end - time.perf_counter())
        self.construct(min(self.end, time.perf_counter() + remaining * 0.13))
        epoch = max(0.002, self.budget / (1.0 + math.sqrt(1.0 + self.strength)))
        control = max(0.001, self.budget / (3.0 + math.log1p(self.n) + self.strength))
        next_restart, next_control = time.perf_counter() + epoch, time.perf_counter() + control
        while time.perf_counter() < self.end:
            now = time.perf_counter()
            if now >= next_restart:
                self.install(self.best)
                self.capacity_penalty = max(self.base_penalty, self.capacity_penalty * 0.6)
                next_restart = now + epoch
            if now >= next_control:
                self.total_over = sum(r.over for r in self.routes)
                if self.total_over > self.eps:
                    self.capacity_penalty *= 1.8
                    self.invalidate()
                next_control = now + control
            changed = self.sweep()
            if not changed and time.perf_counter() < self.end:
                self.total_over = sum(r.over for r in self.routes)
                if self.total_over > self.eps:
                    self.capacity_penalty *= 2.0
                    self.invalidate()
                else:
                    self.capacity_penalty = max(self.base_penalty * 0.35, self.capacity_penalty * 0.85)
                    self.guide()

    def result(self):
        deliveries, pickups = [0] * (self.n + 1), [0] * (self.n + 1)
        answer = []
        for route in self.best:
            if not route.v:
                continue
            if len(route.where) != len(route.v):
                return self.fallback()
            load = sum(self.d[item >> 2] for item in route.v if item & 1)
            if load > self.cap:
                return self.fallback()
            visits = []
            for item in route.v:
                i, mask = item >> 2, item & 3
                if not 1 <= i <= self.n or not mask:
                    return self.fallback()
                if mask & 1:
                    deliveries[i] += 1
                    load -= self.d[i]
                if mask & 2:
                    pickups[i] += 1
                    load += self.p[i]
                if load > self.cap:
                    return self.fallback()
                visits.append({'customer_id': i, 'serve_delivery': bool(mask & 1),
                               'serve_pickup': bool(mask & 2)})
            answer.append(visits)
        if any(deliveries[i] != 1 or pickups[i] != 1 for i in range(1, self.n + 1)):
            return self.fallback()
        return {'routes': answer}

    def fallback(self):
        return {'routes': [[{'customer_id': i, 'serve_delivery': True, 'serve_pickup': True}]
                           for i in range(1, self.n + 1)]}


def solve(instance: dict, time_limit_s: float) -> dict:
    started = time.perf_counter()
    fallback = {'routes': []}
    search = None
    try:
        n = max(0, int(instance['n']))
        fallback = {'routes': [[{'customer_id': i, 'serve_delivery': True,
                                'serve_pickup': True}] for i in range(1, n + 1)]}
        budget = float(time_limit_s)
        if not n or not math.isfinite(budget) or budget <= 0:
            return fallback
        search = _Search(instance, budget, started)
        search.run()
        return search.result()
    except BaseException:
        if search is not None:
            try:
                return search.result()
            except BaseException:
                pass
        return fallback
