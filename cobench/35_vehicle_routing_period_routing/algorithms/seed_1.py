"""Standalone, wall-clock PVRP search.

Exact integer capacity arithmetic and schedule/bin backtracking safeguard
feasibility. Dynamic regret repair and a cached variable-neighbourhood descent
optimize schedules and Euclidean tours jointly. Only distance selects the best
complete incumbent; exploratory acceptance never changes the objective.

A bounded-time search cannot in general certify feasibility or infeasibility
for every legal NP-hard input. As in the supplied reference, an unproved
no-incumbent timeout is explicitly unknown, never a false infeasibility proof.
"""

import fractions
import itertools
import math
import random
import time


class _Search:
    def __init__(self, instance, time_limit_s, started):
        self.n = instance['num_customers']
        self.T = instance['planning_period']
        self.K = instance['num_vehicles']
        self.H = min(self.K, max(1, self.n))
        self.Q = float(instance['vehicle_capacity'])
        self.rawq = [0.0] + [float(v) for v in instance['demands']]
        values = [fractions.Fraction(str(instance['vehicle_capacity']))]
        values += [fractions.Fraction(str(v)) for v in instance['demands']]
        self.unit = 1
        for v in values:
            self.unit = self.unit // math.gcd(self.unit, v.denominator) * v.denominator
        weights = [v.numerator * (self.unit // v.denominator) for v in values]
        self.W = weights[0]
        self.w = [0] + weights[1:]
        self.q = [v / self.W if v <= self.W else 2.0 for v in self.w]
        self.d = instance['distance_matrix']
        self.candidates = instance['candidate_schedules']
        self.budget = max(0.0, float(time_limit_s))
        self.started = started
        self.reserve = min(self.budget * 0.15, 0.003 + 0.0000003 * (
            (self.n + 1) ** 2 + self.n * self.K
            + self.n * self.T * (self.K + self.T)))
        self.end = started + max(0.0, self.budget - self.reserve)
        self.rng = random.Random(917345)
        self.best = None
        self.proven_infeasible = False
        self.scale = (sum(self.d[0][i] / max(1, self.n)
                          for i in range(1, self.n + 1)) or 1.0)
        self.eps = self.scale * 1e-12
        self.intensity = 1.0 + math.log1p(self.budget)
        self.route_cache = {}
        self.pair_cache = set()
        self.cache_limit = max(32, int((self.n + self.T + 1) * self.intensity * 32))
        self.days = [[]]
        self.masks = [[]]
        self.options = [[]]
        self.mandatory = [[]]
        vectors = instance['schedule_vectors']
        for row in self.candidates:
            days = [tuple(t for t, v in enumerate(vectors[s]) if v)
                    for s in row]
            masks = [sum(1 << t for t in ds) for ds in days]
            kept = []
            # Deleting visits never increases a Euclidean tour's distance.
            # Original local candidate indices are retained in the output.
            for r in sorted(range(len(row)), key=lambda r: len(days[r])):
                if not any(masks[a] & masks[r] == masks[a] for a in kept):
                    kept.append(r)
            common = masks[kept[0]]
            for r in kept:
                common &= masks[r]
            self.days.append(days)
            self.masks.append(masks)
            self.options.append(kept)
            self.mandatory.append(tuple(t for t in range(self.T)
                                        if common & (1 << t)))

    def cost(self, routes):
        value = 0.0
        d = self.d
        for day in routes:
            for route in day:
                prev = 0
                for i in route:
                    value += d[prev][i]
                    prev = i
                value += d[prev][0]
        return value

    @staticmethod
    def clone(routes):
        return [[r[:] for r in day] for day in routes]

    def valid(self, choices, routes):
        # Check the compact incumbent before publishing it; all C1--C10 then
        # follow from depot tours, capacities, and the load construction below.
        seen = [0] * (self.n + 1)
        for t, day in enumerate(routes):
            if len(day) > self.K:
                return False
            bit = 1 << t
            for route in day:
                load = 0
                for i in route:
                    if not 1 <= i <= self.n or seen[i] & bit:
                        return False
                    seen[i] |= bit
                    load += self.w[i]
                if load > self.W:
                    return False
        return all(0 <= choices[i] < len(self.candidates[i - 1])
                   and seen[i] == self.masks[i][choices[i]]
                   for i in range(1, self.n + 1))

    def save(self, choices, routes, value=None):
        if value is None:
            value = self.cost(routes)
        if (self.best is None or value < self.best[0]) and self.valid(choices, routes):
            self.best = (value, choices[:], self.clone(routes))

    def route_loads(self, routes):
        return [[sum(self.w[i] for i in r) for r in day] for day in routes]

    def insertion_options(self, i, routes, loads, until, noise, initial):
        """Rank all schedules and feasible bins, retaining cheapest positions.

        Initial feasibility search merges equal-load bins, retaining their
        cheapest representative. Improvement preserves spatially different
        nonempty routes even at equal loads. None denotes a timed-out slice.
        """
        wi, d = self.w[i], self.d
        by_day = {}
        for r in self.options[i]:
            for t in self.days[i][r]:
                if t in by_day:
                    continue
                if time.perf_counter() >= until:
                    return None
                reps = {}
                for h, route in enumerate(routes[t]):
                    load = loads[t][h]
                    if load + wi > self.W:
                        continue
                    delta, pos, prev = float('inf'), 0, 0
                    for p in range(len(route) + 1):
                        nxt = route[p] if p < len(route) else 0
                        extra = d[prev][i] + d[i][nxt] - d[prev][nxt]
                        if extra < delta:
                            delta, pos = extra, p
                        prev = nxt
                    entry = (delta, h, pos)
                    key = load if initial or not route else ('route', h)
                    if key not in reps or delta < reps[key][0]:
                        reps[key] = entry
                entries = list(reps.values())
                if initial:
                    entries.sort(key=lambda e: (-loads[t][e[1]], e[0]))
                else:
                    entries.sort(key=lambda e: (e[0], -loads[t][e[1]]))
                by_day[t] = entries
        variants = []
        for r in self.options[i]:
            ds = self.days[i][r]
            bins = [by_day[t] for t in ds]
            if all(bins):
                cost = sum(b[0][0] for b in bins)
                if initial:
                    score = (len(ds), sum(sum(loads[t]) / self.W for t in ds), cost)
                else:
                    score = (cost + noise * self.scale * len(ds)
                             * self.rng.uniform(-1.0, 1.0),)
                variants.append((score, cost, r, ds, bins))
        variants.sort(key=lambda v: v[0])
        return variants

    def complete(self, removed, choices, routes, until, noise=0.0, initial=False):
        """Joint schedule/bin repair: 1 complete, 0 exhausted, -1 timed out.

        Regret-guided variable order and insertion-cost branch order are used
        inside capacity backtracking, not as separate alternative solvers.
        Route order does not restrict feasibility, so one insertion position
        per bin and symmetry reduction preserve exhaustive feasibility search.
        """
        pending = list(removed)
        if not pending:
            return 1
        if time.perf_counter() >= until:
            return -1
        loads = self.route_loads(routes)
        totals = [sum(day) for day in loads]
        if initial:
            order = sorted(pending, key=lambda i: (
                -self.w[i], len(self.options[i]),
                -min(len(self.days[i][r]) for r in self.options[i]), -self.d[0][i]))
        else:
            ranking = []
            for i in pending:
                variants = self.insertion_options(i, routes, loads, until, noise, False)
                if variants is None:
                    return -1
                if not variants:
                    return 0
                _, cost, _, _, bins = variants[0]
                alternatives = [v[1] for v in variants[1:]]
                alternatives += [cost + b[1][0] - b[0][0] for b in bins if len(b) > 1]
                gap = max(0.0, min(alternatives) - cost) if alternatives else self.scale * 4
                priority = gap / self.scale + self.q[i] + 1.0 / len(variants)
                ranking.append((-priority, -self.w[i], i))
            order = [v[2] for v in sorted(ranking)]
        m, H, T, W, w = len(order), self.H, self.T, self.W, self.w
        forced = [[0] * T for _ in range(m + 1)]
        minimum = [0] * (m + 1)
        for p in range(m - 1, -1, -1):
            i = order[p]
            forced[p] = forced[p + 1][:]
            for t in self.mandatory[i]:
                forced[p][t] += w[i]
            minimum[p] = minimum[p + 1] + w[i] * min(
                len(self.days[i][r]) for r in self.options[i])
        if (sum(totals) + minimum[0] > H * T * W
                or any(totals[t] + forced[0][t] > H * W for t in range(T))):
            return 0
        assigned = [None] * m
        dead = set()
        memo_limit = max(64, int(30000 * self.intensity / (1 + H * T)))

        def placements(p):
            i = order[p]
            variants = self.insertion_options(i, routes, loads, until, noise, initial)
            if variants is None:
                yield None
                return
            used = sum(totals)
            for _, _, r, ds, bins in variants:
                if time.perf_counter() >= until:
                    yield None
                    return
                if used + w[i] * len(ds) + minimum[p + 1] > H * T * W:
                    continue
                mask = self.masks[i][r]
                if any(totals[t] + forced[p + 1][t]
                       + (w[i] if mask & (1 << t) else 0) > H * W for t in range(T)):
                    continue
                for combo in itertools.product(*bins):
                    if time.perf_counter() >= until:
                        yield None
                        return
                    yield r, [(t, e[1], e[2]) for t, e in zip(ds, combo)]

        def undo(p):
            i = order[p]
            for t, h, pos in assigned[p]:
                routes[t][h].pop(pos)
                loads[t][h] -= w[i]
                totals[t] -= w[i]

        frames = []
        depth = 0
        while time.perf_counter() < until:
            if len(frames) == depth:
                key = (depth, tuple(tuple(sorted(day)) for day in loads))
                if key in dead:
                    if depth == 0:
                        return 0
                    depth -= 1
                    undo(depth)
                    continue
                frames.append((placements(depth), key))
            try:
                move = next(frames[depth][0])
            except StopIteration:
                _, key = frames.pop()
                if len(dead) < memo_limit:
                    dead.add(key)
                if depth == 0:
                    return 0
                depth -= 1
                undo(depth)
                continue
            if move is None:
                return -1
            r, moves = move
            i = order[depth]
            choices[i] = r
            assigned[depth] = moves
            for t, h, pos in moves:
                routes[t][h].insert(pos, i)
                loads[t][h] += w[i]
                totals[t] += w[i]
            depth += 1
            # No deadline check may intervene between completion and its return.
            if depth == m:
                return 1
        return -1

    def initial(self):
        routes = [[[] for _ in range(self.H)] for _ in range(self.T)]
        choices = [-1] * (self.n + 1)
        if not self.n:
            self.save(choices, routes, 0.0)
            return
        if any(v > self.W for v in self.w[1:]):
            self.proven_infeasible = True
            return
        large = [0] * self.T
        for i in range(1, self.n + 1):
            if 2 * self.w[i] > self.W:
                for t in self.mandatory[i]:
                    large[t] += 1
        if any(v > self.H for v in large):
            self.proven_infeasible = True
            return
        # This case has an immediate certificate, independent of search time.
        if self.H >= self.n:
            for i in range(1, self.n + 1):
                r = min(self.options[i], key=lambda a: len(self.days[i][a]))
                choices[i] = r
                for t in self.days[i][r]:
                    routes[t][i - 1].append(i)
            self.save(choices, routes)
            return
        result = self.complete(range(1, self.n + 1), choices, routes, self.end, initial=True)
        if result == 1:
            self.save(choices, routes)
        elif result == 0:
            self.proven_infeasible = True

    def regret_repair(self, removed, choices, routes, until, noise):
        """Recompute regret after each insertion, caching unchanged days."""
        pending = set(removed)
        loads = self.route_loads(routes)
        versions = [0] * self.T
        cache = {}
        d, w, W = self.d, self.w, self.W
        regret_depth = max(2, int(1 + math.log1p(self.budget)))
        while pending and time.perf_counter() < until:
            selected, plan, priority = None, None, -float('inf')
            for i in pending:
                if time.perf_counter() >= until:
                    return False
                by_day = {}
                for r in self.options[i]:
                    for t in self.days[i][r]:
                        if t in by_day:
                            continue
                        key = (i, t)
                        old = cache.get(key)
                        if old is not None and old[0] == versions[t]:
                            by_day[t] = old[1]
                            continue
                        entries, empty_seen = [], False
                        for h, route in enumerate(routes[t]):
                            if loads[t][h] + w[i] > W:
                                continue
                            if not route:
                                if empty_seen:
                                    continue
                                empty_seen = True
                            best, pos, prev = float('inf'), 0, 0
                            for p in range(len(route) + 1):
                                nxt = route[p] if p < len(route) else 0
                                v = d[prev][i] + d[i][nxt] - d[prev][nxt]
                                if v < best:
                                    best, pos = v, p
                                prev = nxt
                            score = best + noise * self.scale * self.rng.uniform(-1, 1)
                            entries.append((score, best, h, pos))
                        entries.sort()
                        by_day[t] = entries
                        cache[key] = versions[t], entries
                variants = []
                for r in self.options[i]:
                    ds = self.days[i][r]
                    if all(by_day[t] for t in ds):
                        score = sum(by_day[t][0][0] for t in ds)
                        variants.append((score, r, ds))
                if not variants:
                    return False
                variants.sort()
                score, r, ds = variants[0]
                alternatives = [v[0] for v in variants[1:]]
                for t in ds:
                    entries = by_day[t]
                    alternatives.extend(score + e[0] - entries[0][0]
                                        for e in entries[1:regret_depth])
                alternatives.sort()
                if alternatives:
                    regret = sum(max(0.0, v - score)
                                 for v in alternatives[:regret_depth - 1])
                else:
                    regret = self.scale * (4 + len(ds))
                rank = regret / self.scale + self.q[i] + 1.0 / len(variants)
                if rank > priority:
                    priority, selected = rank, i
                    plan = r, [(t, by_day[t][0][2], by_day[t][0][3]) for t in ds]
            if selected is None:
                return False
            r, moves = plan
            choices[selected] = r
            for t, h, pos in moves:
                routes[t][h].insert(pos, selected)
                loads[t][h] += w[selected]
                versions[t] += 1
            pending.remove(selected)
        return not pending

    def repair(self, removed, choices, routes, loads, noise):
        until = min(self.end, time.perf_counter() +
                    self.budget / (35.0 + 3.0 * math.sqrt(max(1, self.n))))
        original = self.clone(routes)
        old_choices = choices[:]
        # Leave time for the exact capacity fallback, rather than repeatedly
        # trying greedy insertions into a neighbourhood known to be blocked.
        greedy_end = min(until, time.perf_counter() +
                         max(0.0, until - time.perf_counter()) * 0.45)
        if self.regret_repair(removed, choices, routes, greedy_end, noise):
            loads[:] = self.route_loads(routes)
            return True
        routes[:] = original
        choices[:] = old_choices
        ok = self.complete(removed, choices, routes, until, noise) == 1
        if ok:
            loads[:] = self.route_loads(routes)
        return ok

    def tour_cost(self, route):
        d, prev, value = self.d, 0, 0.0
        for i in route:
            value += d[prev][i]
            prev = i
        return value + d[prev][0]

    def intra(self, route, until):
        """Best 2-opt and Or-opt descent; cache by customer set, not day."""
        if len(route) < 3:
            return
        key = tuple(sorted(route))
        record = self.route_cache.get(key)
        value = self.tour_cost(route)
        if record is not None and record[0] <= value + self.eps:
            route[:] = record[1]
            if record[2]:
                return
            value = record[0]
        d, m, eps = self.d, len(route), self.eps
        span = min(m - 1, max(1, int(1 + math.log1p(self.budget))))
        optimal = False
        while time.perf_counter() < until:
            delta, move = -eps, None
            for a in range(m - 1):
                if time.perf_counter() >= until:
                    break
                p = route[a - 1] if a else 0
                i = route[a]
                for b in range(a + 1, m):
                    j = route[b]
                    z = route[b + 1] if b + 1 < m else 0
                    diff = d[p][j] + d[i][z] - d[p][i] - d[j][z]
                    if diff < delta:
                        delta, move = diff, (0, a, b)
            if move is not None:
                _, a, b = move
                route[a:b + 1] = reversed(route[a:b + 1])
                continue
            if time.perf_counter() >= until:
                break
            for length in range(1, span + 1):
                if time.perf_counter() >= until:
                    break
                for a in range(m - length + 1):
                    if time.perf_counter() >= until:
                        break
                    p = route[a - 1] if a else 0
                    z = route[a + length] if a + length < m else 0
                    first, last = route[a], route[a + length - 1]
                    removal = d[p][z] - d[p][first] - d[last][z]
                    for b in range(m + 1):
                        if a <= b <= a + length:
                            continue
                        left = route[b - 1] if b else 0
                        right = route[b] if b < m else 0
                        fwd = d[left][first] + d[last][right] - d[left][right]
                        rev = d[left][last] + d[first][right] - d[left][right]
                        diff = removal + min(fwd, rev)
                        if diff < delta:
                            delta, move = diff, (1, a, length, b, rev < fwd)
            if move is None:
                optimal = time.perf_counter() < until
                break
            _, a, length, b, reverse = move
            segment = route[a:a + length]
            if reverse:
                segment.reverse()
            del route[a:a + length]
            pos = b if b < a else b - length
            route[pos:pos] = segment
        value = self.tour_cost(route)
        if len(self.route_cache) >= self.cache_limit:
            self.route_cache.clear()
        if record is None or value <= record[0] + eps:
            self.route_cache[key] = value, route[:], optimal

    def pair_move(self, a, b, la, lb, until):
        """Relocate, SWAP* (independent best reinsertions), both 2-opt* joins."""
        d, w, W, eps = self.d, self.w, self.W, self.eps
        m, n = len(a), len(b)
        best, move = -eps, None
        pa, pb = [0], [0]
        for i in a:
            pa.append(pa[-1] + w[i])
        for i in b:
            pb.append(pb[-1] + w[i])
        # Prefix/prefix and prefix/suffix connections also include route merges.
        for x in range(m + 1):
            if time.perf_counter() >= until:
                return best, move, False
            p, i = a[x - 1] if x else 0, a[x] if x < m else 0
            for y in range(n + 1):
                q, j = b[y - 1] if y else 0, b[y] if y < n else 0
                old = d[p][i] + d[q][j]
                if pa[x] + lb - pb[y] <= W and pb[y] + la - pa[x] <= W:
                    delta = d[p][j] + d[q][i] - old
                    if delta < best:
                        best, move = delta, ('tail', x, y, False)
                if pa[x] + pb[y] <= W and la + lb - pa[x] - pb[y] <= W:
                    delta = d[p][q] + d[i][j] - old
                    if delta < best:
                        best, move = delta, ('tail', x, y, True)
        top_a, top_b = [], []
        rem_a = [d[a[x - 1] if x else 0][a[x + 1] if x + 1 < m else 0]
                 - d[a[x - 1] if x else 0][i]
                 - d[i][a[x + 1] if x + 1 < m else 0] for x, i in enumerate(a)]
        rem_b = [d[b[y - 1] if y else 0][b[y + 1] if y + 1 < n else 0]
                 - d[b[y - 1] if y else 0][j]
                 - d[j][b[y + 1] if y + 1 < n else 0] for y, j in enumerate(b)]
        for direction, src, dst, ls, ld, tops, removals in (
                (0, a, b, la, lb, top_a, rem_a),
                (1, b, a, lb, la, top_b, rem_b)):
            size = len(dst)
            for x, i in enumerate(src):
                if time.perf_counter() >= until:
                    return best, move, False
                entries = []
                prev = 0
                for y in range(size + 1):
                    nxt = dst[y] if y < size else 0
                    v = d[prev][i] + d[i][nxt] - d[prev][nxt]
                    entries.append((v, y))
                    prev = nxt
                entries.sort()
                tops.append(entries[:3])
                if ld + w[i] <= W:
                    delta = removals[x] + entries[0][0]
                    if delta < best:
                        best, move = delta, ('rel', direction, x, 1, entries[0][1], False)
                # Two-customer Or-opt bridges tight assignments without requiring
                # either individual relocation to improve the objective.
                if x + 1 < len(src) and ld + w[i] + w[src[x + 1]] <= W:
                    last = src[x + 1]
                    p = src[x - 1] if x else 0
                    z = src[x + 2] if x + 2 < len(src) else 0
                    remove = d[p][z] - d[p][i] - d[last][z]
                    prev = 0
                    for y in range(size + 1):
                        nxt = dst[y] if y < size else 0
                        fwd = d[prev][i] + d[last][nxt] - d[prev][nxt]
                        rev = d[prev][last] + d[i][nxt] - d[prev][nxt]
                        delta = remove + min(fwd, rev)
                        if delta < best:
                            best, move = delta, ('rel', direction, x, 2, y, rev < fwd)
                        prev = nxt
        for x, i in enumerate(a):
            if time.perf_counter() >= until:
                return best, move, False
            ap, az = a[x - 1] if x else 0, a[x + 1] if x + 1 < m else 0
            for y, j in enumerate(b):
                if la - w[i] + w[j] > W or lb - w[j] + w[i] > W:
                    continue
                bp, bz = b[y - 1] if y else 0, b[y + 1] if y + 1 < n else 0
                va, posa = d[ap][j] + d[j][az] - d[ap][az], x
                for v, pos in top_b[y]:
                    if pos != x and pos != x + 1:
                        if v < va:
                            va, posa = v, pos - int(pos > x)
                        break
                vb, posb = d[bp][i] + d[i][bz] - d[bp][bz], y
                for v, pos in top_a[x]:
                    if pos != y and pos != y + 1:
                        if v < vb:
                            vb, posb = v, pos - int(pos > y)
                        break
                delta = rem_a[x] + rem_b[y] + va + vb
                if delta < best:
                    best, move = delta, ('swap', x, y, posa, posb)
        return best, move, True

    @staticmethod
    def apply_pair(a, b, move):
        if move[0] == 'tail':
            _, x, y, reverse = move
            if reverse:
                return a[:x] + b[:y][::-1], a[x:][::-1] + b[y:]
            return a[:x] + b[y:], b[:y] + a[x:]
        aa, bb = a[:], b[:]
        if move[0] == 'rel':
            _, direction, x, length, y, reverse = move
            src, dst = (aa, bb) if direction == 0 else (bb, aa)
            segment = src[x:x + length]
            del src[x:x + length]
            if reverse:
                segment.reverse()
            dst[y:y] = segment
        else:
            _, x, y, posa, posb = move
            i, j = aa.pop(x), bb.pop(y)
            aa.insert(posa, j)
            bb.insert(posb, i)
        return aa, bb

    def day_descent(self, day, loads, until):
        while time.perf_counter() < until:
            for route in day:
                if time.perf_counter() >= until:
                    return
                self.intra(route, until)
            changed = False
            order = list(range(self.H))
            self.rng.shuffle(order)
            for p, h in enumerate(order):
                if time.perf_counter() >= until:
                    return
                for j in order[p + 1:]:
                    a, b = day[h], day[j]
                    if not a or not b:
                        continue
                    key = tuple(a), tuple(b)
                    if key in self.pair_cache:
                        continue
                    _, move, done = self.pair_move(a, b, loads[h], loads[j], until)
                    if move is not None:
                        day[h], day[j] = self.apply_pair(a, b, move)
                        loads[h] = sum(self.w[i] for i in day[h])
                        loads[j] = sum(self.w[i] for i in day[j])
                        changed = True
                        break
                    if done:
                        if len(self.pair_cache) >= self.cache_limit * 2:
                            self.pair_cache.clear()
                        self.pair_cache.add(key)
                        self.pair_cache.add((key[1], key[0]))
                    if time.perf_counter() >= until:
                        return
                if changed:
                    break
            if not changed:
                return

    def schedule_sweep(self, choices, routes, loads, until):
        """Evaluate a customer's entire schedule change, not separate visits."""
        customers = list(range(1, self.n + 1))
        self.rng.shuffle(customers)
        changed, d = False, self.d
        for i in customers:
            if time.perf_counter() >= until:
                break
            old = choices[i]
            removed, saving = [], 0.0
            for t in self.days[i][old]:
                for h, route in enumerate(routes[t]):
                    if i in route:
                        pos = route.index(i)
                        a = route[pos - 1] if pos else 0
                        b = route[pos + 1] if pos + 1 < len(route) else 0
                        saving += d[a][i] + d[i][b] - d[a][b]
                        route.pop(pos)
                        loads[t][h] -= self.w[i]
                        removed.append((t, h, pos))
                        break
            variants = self.insertion_options(i, routes, loads, until, 0.0, False)
            if variants and variants[0][1] < saving - self.eps:
                _, _, r, ds, bins = variants[0]
                choices[i] = r
                for t, entries in zip(ds, bins):
                    _, h, pos = entries[0]
                    routes[t][h].insert(pos, i)
                    loads[t][h] += self.w[i]
                changed = True
            else:
                for t, h, pos in removed:
                    routes[t][h].insert(pos, i)
                    loads[t][h] += self.w[i]
        return changed

    def polish(self, choices, routes, loads, until):
        while time.perf_counter() < until:
            days = list(range(self.T))
            self.rng.shuffle(days)
            for t in days:
                if time.perf_counter() >= until:
                    return
                self.day_descent(routes[t], loads[t], until)
            if not self.schedule_sweep(choices, routes, loads, until):
                return

    def improve(self):
        if self.best is None or not self.n or self.best[0] == 0.0:
            return
        n, rng = self.n, self.rng
        value, choices, routes = self.best
        choices, routes = choices[:], self.clone(routes)
        loads = self.route_loads(routes)
        self.polish(choices, routes, loads, min(self.end, time.perf_counter()
                    + self.budget / (5.0 + math.sqrt(n))))
        value = self.cost(routes)
        self.save(choices, routes, value)
        last_best = time.perf_counter()
        max_remove = min(n, max(1, int(math.sqrt(n * self.intensity))))
        local_slice = self.budget / (60.0 + 8.0 * math.sqrt(n))
        customers = list(range(1, n + 1))
        flexible = [i for i in customers if len(self.options[i]) > 1]
        weights = [1.0] * 6
        while time.perf_counter() < self.end:
            now = time.perf_counter()
            stagnation = (now - last_best) / max(self.budget, 1e-9)
            if stagnation > 1.0 / (2.0 + self.intensity) and rng.random() < 0.10:
                value, saved_choices, saved_routes = self.best
                choices, routes = saved_choices[:], self.clone(saved_routes)
            ceiling = min(n, max_remove + int(n * min(0.35, stagnation)))
            count = rng.randint(1, max(1, ceiling))
            mode = rng.choices(range(6), weights=weights)[0]
            seed = rng.choice(customers)
            if mode == 0:
                removed = set(rng.sample(customers, count))
            elif mode == 1:
                mask = self.masks[seed][choices[seed]]
                ranking = sorted(customers, key=lambda i:
                    self.d[seed][i] / self.scale
                    + 0.25 * (mask ^ self.masks[i][choices[i]]).bit_count()
                    + 0.25 * rng.random())
                removed = set(ranking[:count])
            elif mode == 2:
                gains = [0.0] * (n + 1)
                for day in routes:
                    for route in day:
                        for p, i in enumerate(route):
                            a = route[p - 1] if p else 0
                            b = route[p + 1] if p + 1 < len(route) else 0
                            gains[i] += self.d[a][i] + self.d[i][b] - self.d[a][b]
                removed = set(sorted(customers, key=lambda i:
                              -gains[i] * (0.6 + rng.random()))[:count])
            elif mode == 4 and flexible:
                seed = rng.choice(flexible)
                target = rng.choice(self.options[seed])
                mask = self.masks[seed][target]
                removed = {seed}
                ranking = sorted((i for i in customers if i != seed), key=lambda i:
                    self.d[seed][i] / self.scale + rng.random() * 0.3
                    - 0.5 * (mask & self.masks[i][choices[i]]).bit_count())
                removed.update(ranking[:max(0, count - 1)])
            else:
                active = [r for day in routes for r in day if r]
                route = rng.choice(active)
                if mode == 5:
                    start = rng.randrange(len(route))
                    removed = set((route[start:] + route[:start])[:count])
                else:
                    removed = set(route)
                if len(removed) < count:
                    pool = [i for i in customers if i not in removed]
                    removed.update(rng.sample(pool, min(count - len(removed), len(pool))))
            trial = [[[i for i in route if i not in removed] for route in day]
                     for day in routes]
            trial_choices = choices[:]
            trial_loads = self.route_loads(trial)
            noise = (rng.random() * min(0.25, 0.035 * self.intensity)
                     if rng.random() < 0.35 else 0.0)
            oldbest = self.best[0]
            if not self.repair(removed, trial_choices, trial, trial_loads, noise):
                weights[mode] = 0.97 * weights[mode] + 0.03 * 0.2
                continue
            trial_value = self.cost(trial)
            self.save(trial_choices, trial, trial_value)
            self.polish(trial_choices, trial, trial_loads,
                        min(self.end, time.perf_counter() + local_slice))
            trial_value = self.cost(trial)
            self.save(trial_choices, trial, trial_value)
            reward = 0.5
            if self.best[0] < oldbest - self.eps:
                last_best = time.perf_counter()
                reward = 5.0 + (oldbest - self.best[0]) / self.scale
            elif trial_value < value - self.eps:
                reward = 2.0
            weights[mode] = 0.97 * weights[mode] + 0.03 * reward
            progress = min(1.0, (time.perf_counter() - self.started)
                           / max(self.budget, 1e-9))
            visits = max(1, sum(len(self.days[i][trial_choices[i]]) for i in customers))
            temperature = max(self.eps, self.best[0] / visits * 0.055
                              * (0.02 ** progress) * (1 + min(2.0, 5 * stagnation)))
            if trial_value <= value or rng.random() < math.exp(
                    min(0.0, (value - trial_value) / temperature)):
                choices, routes, value = trial_choices, trial, trial_value

    def result(self):
        if self.best is None:
            return {'status': 'infeasible' if self.proven_infeasible else 'unknown',
                    'x': None, 'y': None, 'u': None}
        _, choices, routes = self.best
        x = [[int(r == choices[i]) for r in range(len(row))]
             for i, row in enumerate(self.candidates, 1)]
        # These are full dense lists, not sparse output. Shared zero leaves are
        # immutable except through the copy-on-write arc helper below.
        zero_t = [0] * self.T
        zero_kt = [zero_t] * self.K
        y = [[zero_kt] * (self.n + 1) for _ in range(self.n + 1)]
        bases = [[v] * self.T for v in self.rawq[1:]]
        u = [[base] * self.K for base in bases]

        def arc(i, j, h, t):
            if y[i][j] is zero_kt:
                y[i][j] = zero_kt[:]
            if y[i][j][h] is zero_t:
                y[i][j][h] = zero_t[:]
            y[i][j][h][t] = 1

        for t, day in enumerate(routes):
            for h, route in enumerate(day):
                if not route:
                    continue
                prefix, prev = 0, 0
                for i in route:
                    arc(prev, i, h, t)
                    prefix += self.w[i]
                    if u[i - 1][h] is bases[i - 1]:
                        u[i - 1][h] = bases[i - 1][:]
                    u[i - 1][h][t] = prefix / self.unit
                    prev = i
                arc(prev, 0, h, t)
        return {'status': 'feasible', 'x': x, 'y': y, 'u': u}


def solve(instance: dict, time_limit_s: float) -> dict:
    """Return a complete incumbent; a timeout is never an infeasibility proof."""
    started = time.perf_counter()
    search = None
    fallback = None
    try:
        search = _Search(instance, time_limit_s, started)
        search.initial()
        if search.best is not None:
            before = time.perf_counter()
            fallback = search.result()
            rendering = time.perf_counter() - before
            search.end = min(search.end, started + search.budget -
                             max(search.reserve, 2.0 * rendering + 0.001))
            search.improve()
    except Exception:
        # Failed trial work must never corrupt or discard a saved incumbent.
        pass
    if search is not None:
        try:
            return search.result()
        except Exception:
            pass
    if fallback is not None:
        return fallback
    return {'status': 'unknown', 'x': None, 'y': None, 'u': None}
