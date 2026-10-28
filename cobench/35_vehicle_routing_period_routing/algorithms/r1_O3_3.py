"""Capacity-oscillation and route-pool cooperative large neighbourhood search
Jointly explores with schedule regret repair, capacity penalties and local route optimisation, and uses branch-and-bound over a shared route pool to restore capacity and recombine routes.
"""

import array
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
        self.reserve = min(self.budget * 0.15, 0.003 + 3e-07 * ((self.n + 1) ** 2 + self.n * self.K + self.n * self.T * (self.K + self.T)))
        self.reserve = max(self.reserve, self.budget * 0.035)
        self.end = started + max(0.0, self.budget - self.reserve)
        self.rng = random.Random(917345)
        self.best = None
        self.proven_infeasible = False
        self.scale = sum((self.d[0][i] / max(1, self.n) for i in range(1, self.n + 1))) or 1.0
        self.eps = self.scale * 1e-12
        self.intensity = 1.0 + math.log1p(self.budget)
        self.route_cache = {}
        self.pair_cache = set()
        self.cache_limit = max(32, int((self.n + self.T + 1) * self.intensity * 32))
        self.insert_cache = {}
        self.pool = {}
        self.pool_version = 0
        self.pool_limit = max(self.n, int((self.n + self.T + 1) * self.intensity * 12))
        self.partitions = {}
        self.bits = [0] + [1 << i for i in range(self.n)]
        self.days = [[]]
        self.masks = [[]]
        self.options = [[]]
        self.mandatory = [[]]
        vectors = instance['schedule_vectors']
        for row in self.candidates:
            days = [tuple((t for t, v in enumerate(vectors[s]) if v)) for s in row]
            masks = [sum((1 << t for t in ds)) for ds in days]
            kept = []
            for r in sorted(range(len(row)), key=lambda r: len(days[r])):
                if not any((masks[a] & masks[r] == masks[a] for a in kept)):
                    kept.append(r)
            common = masks[kept[0]]
            for r in kept:
                common &= masks[r]
            self.days.append(days)
            self.masks.append(masks)
            self.options.append(kept)
            self.mandatory.append(tuple((t for t in range(self.T) if common & 1 << t)))
        self.flexible = [i for i in range(1, self.n + 1) if len(self.options[i]) > 1]
        self.penalty = self.scale * (2.0 + math.sqrt(self.n))
        self.seed = None
        self.exact_optimal = False

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
        return all((0 <= choices[i] < len(self.candidates[i - 1]) and seen[i] == self.masks[i][choices[i]] for i in range(1, self.n + 1)))

    def save(self, choices, routes, value=None):
        if value is None:
            value = self.cost(routes)
        self.collect(routes)
        if (self.best is None or value < self.best[0]) and self.valid(choices, routes):
            self.best = (value, choices[:], self.clone(routes))

    def insert_at(self, i, route):
        if not route:
            return (self.d[0][i] + self.d[i][0], 0)
        key = (i, tuple(route))
        old = self.insert_cache.get(key)
        if old is not None:
            return old
        d, previous, best, pos = (self.d, 0, math.inf, 0)
        for p in range(len(route) + 1):
            nxt = route[p] if p < len(route) else 0
            v = d[previous][i] + d[i][nxt] - d[previous][nxt]
            if v < best:
                best, pos = (v, p)
            previous = nxt
        if len(self.insert_cache) >= self.cache_limit * 2:
            self.insert_cache.clear()
        self.insert_cache[key] = (best, pos)
        return (best, pos)

    def collect(self, routes):
        for day in routes:
            for route in day:
                if time.perf_counter() >= self.end:
                    return
                if not route:
                    continue
                chunks, segment, mass = ([], [], 0)
                for i in route:
                    if mass + self.w[i] > self.W and segment:
                        chunks.append(segment)
                        segment, mass = ([], 0)
                    segment.append(i)
                    mass += self.w[i]
                if segment:
                    chunks.append(segment)
                for piece in chunks:
                    if sum((self.w[i] for i in piece)) > self.W:
                        continue
                    mask = sum((self.bits[i] for i in piece))
                    value = self.tour_cost(piece)
                    old = self.pool.get(mask)
                    if old is None or value < old[0]:
                        self.pool[mask] = (value, tuple(piece))
                        self.pool_version += 1
        if len(self.pool) > self.pool_limit * 1.2:
            keep = set()
            if self.best is not None:
                keep = {sum((self.bits[i] for i in r)) for day in self.best[2] for r in day if r}
            for mask in list(self.pool):
                if len(self.pool) <= self.pool_limit:
                    break
                if mask not in keep:
                    del self.pool[mask]
                    self.pool_version += 1

    def route_loads(self, routes):
        return [[sum((self.w[i] for i in r)) for r in day] for day in routes]

    def insertion_options(self, i, routes, loads, until, noise, initial):
        wi = self.w[i]
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
                    delta, pos = self.insert_at(i, route)
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
                cost = sum((b[0][0] for b in bins))
                if initial:
                    score = (len(ds), sum((sum(loads[t]) / self.W for t in ds)), cost)
                else:
                    score = (cost + noise * self.scale * len(ds) * self.rng.uniform(-1.0, 1.0),)
                variants.append((score, cost, r, ds, bins))
        variants.sort(key=lambda v: v[0])
        return variants

    def complete(self, removed, choices, routes, until, noise=0.0, initial=False):
        pending = list(removed)
        if not pending:
            return 1
        if time.perf_counter() >= until:
            return -1
        loads = self.route_loads(routes)
        totals = [sum(day) for day in loads]
        if initial:
            order = sorted(pending, key=lambda i: (-self.w[i], len(self.options[i]), -min((len(self.days[i][r]) for r in self.options[i])), -self.d[0][i]))
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
        m, H, T, W, w = (len(order), self.H, self.T, self.W, self.w)
        forced = [[0] * T for _ in range(m + 1)]
        minimum = [0] * (m + 1)
        for p in range(m - 1, -1, -1):
            i = order[p]
            forced[p] = forced[p + 1][:]
            for t in self.mandatory[i]:
                forced[p][t] += w[i]
            minimum[p] = minimum[p + 1] + w[i] * min((len(self.days[i][r]) for r in self.options[i]))
        if sum(totals) + minimum[0] > H * T * W or any((totals[t] + forced[0][t] > H * W for t in range(T))):
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
                if any((totals[t] + forced[p + 1][t] + (w[i] if mask & 1 << t else 0) > H * W for t in range(T))):
                    continue
                for combo in itertools.product(*bins):
                    if time.perf_counter() >= until:
                        yield None
                        return
                    yield (r, [(t, e[1], e[2]) for t, e in zip(ds, combo)])

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
                key = (depth, tuple((tuple(sorted(day)) for day in loads)))
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
            if depth == m:
                return 1
        return -1

    def initial(self):
        routes = [[[] for _ in range(self.H)] for _ in range(self.T)]
        choices = [-1] * (self.n + 1)
        if not self.n:
            self.save(choices, routes, 0.0)
            return
        if any((v > self.W for v in self.w[1:])):
            self.proven_infeasible = True
            return
        work = sum((self.w[i] * min((len(self.days[i][r]) for r in self.options[i])) for i in range(1, self.n + 1)))
        forced, large = ([0] * self.T, [0] * self.T)
        for i in range(1, self.n + 1):
            for t in self.mandatory[i]:
                forced[t] += self.w[i]
                large[t] += int(2 * self.w[i] > self.W)
        if work > self.H * self.T * self.W or any((v > self.H * self.W for v in forced)) or any((v > self.H for v in large)):
            self.proven_infeasible = True
            return
        if self.H >= self.n:
            for i in range(1, self.n + 1):
                r = min(self.options[i], key=lambda a: len(self.days[i][a]))
                choices[i] = r
                for t in self.days[i][r]:
                    routes[t][i - 1].append(i)
            self.save(choices, routes)
            return
        until = min(self.end, time.perf_counter() + self.budget / (2.0 + math.sqrt(self.intensity)))
        result = self.complete(range(1, self.n + 1), choices, routes, until, initial=True)
        if result == 1:
            self.save(choices, routes)
        elif result == 0:
            self.proven_infeasible = True
        elif time.perf_counter() < self.end:
            self.seed = self.soft_seed(self.end)
            if self.seed is not None:
                self.save(*self.seed)

    def regret_repair(self, removed, choices, routes, until, noise, preferred, penalty=math.inf):
        work = self.clone(routes)
        pick = choices[:]
        loads = self.route_loads(work)
        versions = [0] * self.T
        pending = sorted(removed)
        cached, perturb = ({}, {})
        d, w, W, rng = (self.d, self.w, self.W, self.rng)
        depth = max(2, int(1 + math.log1p(self.budget)))
        while pending and time.perf_counter() < until:
            winner, winner_key = (None, None)
            for i in pending:
                if time.perf_counter() >= until:
                    return False
                opts = [preferred[i]] if i in preferred else self.options[i]
                by_day = {}
                for r in opts:
                    for t in self.days[i][r]:
                        if t in by_day:
                            continue
                        key = (i, t)
                        old = cached.get(key)
                        if old is not None and old[0] == versions[t]:
                            by_day[t] = old[1]
                            continue
                        if time.perf_counter() >= until:
                            return False
                        entries, empty = ([], False)
                        for h, route in enumerate(work[t]):
                            if penalty == math.inf and loads[t][h] + w[i] > W:
                                continue
                            if not route:
                                if empty:
                                    continue
                                empty = True
                            extra, pos = self.insert_at(i, route)
                            nk = (i, t, h)
                            if nk not in perturb:
                                perturb[nk] = noise * self.scale * rng.uniform(-1, 1)
                            charge = 0.0 if penalty == math.inf else penalty * (self.excess(loads[t][h] + w[i]) - self.excess(loads[t][h]))
                            entries.append((extra + charge + perturb[nk], extra, h, pos))
                        entries.sort()
                        cached[key] = (versions[t], entries)
                        by_day[t] = entries
                variants = []
                for r in opts:
                    ds = self.days[i][r]
                    if all((by_day[t] for t in ds)):
                        score = sum((by_day[t][0][0] for t in ds))
                        variants.append((score, r))
                if not variants:
                    return False
                variants.sort()
                score, r = variants[0]
                alternatives = [v[0] for v in variants]
                for t in self.days[i][r]:
                    row = by_day[t]
                    alternatives.extend((score + v[0] - row[0][0] for v in row[1:depth]))
                alternatives.sort()
                forced = len(alternatives) == 1
                regret = sum((v - score for v in alternatives[1:depth]))
                key = (forced, regret / self.scale + self.q[i] + 1.0 / len(variants), score, self.w[i])
                if winner_key is None or key > winner_key:
                    winner_key = key
                    winner = (i, r, [(t, by_day[t][0][2], by_day[t][0][3]) for t in self.days[i][r]])
            i, r, moves = winner
            pick[i] = r
            for t, h, pos in moves:
                work[t][h].insert(pos, i)
                loads[t][h] += w[i]
                versions[t] += 1
            pending.remove(i)
        if pending:
            return False
        choices[:] = pick
        routes[:] = work
        return True

    def repair(self, removed, choices, routes, loads, noise, preferred, until=None):
        if until is None:
            until = min(self.end, time.perf_counter() + self.budget / (35.0 + 3.0 * math.sqrt(max(1, self.n))))
        old_options = {i: self.options[i] for i in preferred}
        try:
            for i, r in preferred.items():
                self.options[i] = [r]
            greedy_end = time.perf_counter() + 0.55 * max(0.0, until - time.perf_counter())
            if self.regret_repair(removed, choices, routes, greedy_end, noise, preferred):
                loads[:] = self.route_loads(routes)
                return True
            ok = self.complete(removed, choices, routes, until, noise) == 1
            if ok:
                loads[:] = self.route_loads(routes)
            return ok
        finally:
            for i, opts in old_options.items():
                self.options[i] = opts

    def tour_cost(self, route):
        d, prev, value = (self.d, 0, 0.0)
        for i in route:
            value += d[prev][i]
            prev = i
        return value + d[prev][0]

    def intra(self, route, until):
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
        d, m, eps = (self.d, len(route), self.eps)
        span = min(m - 1, max(1, int(1 + math.log1p(self.budget))))
        optimal = False
        while time.perf_counter() < until:
            delta, move = (-eps, None)
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
                        delta, move = (diff, (0, a, b))
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
                    first, last = (route[a], route[a + length - 1])
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
                            delta, move = (diff, (1, a, length, b, rev < fwd))
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
            self.route_cache[key] = (value, route[:], optimal)

    def pair_move(self, r, s, lr, ls, until, penalty=math.inf):
        key = (tuple(r), tuple(s))
        hard = penalty == math.inf
        if hard and key in self.pair_cache:
            return None
        base_excess = self.excess(lr) + self.excess(ls)

        def surcharge(a, b):
            if hard:
                return 0.0
            return penalty * (self.excess(a) + self.excess(b) - base_excess)
        d, w, W, eps = (self.d, self.w, self.W, self.eps)
        best, move = (-eps, None)
        complete = True
        span = max(1, int(math.log1p(self.budget)))
        for rev in (False, True):
            a, b = (s, r) if rev else (r, s)
            lb = lr if rev else ls
            la = ls if rev else lr
            for p in range(len(a)):
                if time.perf_counter() >= until:
                    complete = False
                    break
                mass = 0
                for length in range(1, min(span, len(a) - p) + 1):
                    mass += w[a[p + length - 1]]
                    if hard and lb + mass > W:
                        break
                    charge = surcharge(la - mass, lb + mass)
                    first, last = (a[p], a[p + length - 1])
                    left = a[p - 1] if p else 0
                    right = a[p + length] if p + length < len(a) else 0
                    rem = d[left][right] - d[left][first] - d[last][right]
                    for q in range(len(b) + 1):
                        prev, nxt = (b[q - 1] if q else 0, b[q] if q < len(b) else 0)
                        delta = rem + d[prev][first] + d[last][nxt] - d[prev][nxt] + charge
                        if delta < best:
                            best, move = (delta, (0, rev, p, length, q, False))
                        if length > 1:
                            delta = rem + d[prev][last] + d[first][nxt] - d[prev][nxt] + charge
                            if delta < best:
                                best, move = (delta, (0, rev, p, length, q, True))
            if not complete:
                break
        pr, ps = ([0], [0])
        for i in r:
            pr.append(pr[-1] + w[i])
        for j in s:
            ps.append(ps[-1] + w[j])
        if complete:
            for a in range(len(r) + 1):
                if time.perf_counter() >= until:
                    complete = False
                    break
                p = r[a - 1] if a else 0
                z = r[a] if a < len(r) else 0
                for b in range(len(s) + 1):
                    v = s[b - 1] if b else 0
                    x = s[b] if b < len(s) else 0
                    nr, ns = (pr[a] + ls - ps[b], ps[b] + lr - pr[a])
                    if not hard or max(nr, ns) <= W:
                        delta = d[p][x] + d[v][z] - d[p][z] - d[v][x] + surcharge(nr, ns)
                        if delta < best:
                            best, move = (delta, (1, a, b, False))
                    nr, ns = (pr[a] + ps[b], lr + ls - pr[a] - ps[b])
                    if not hard or max(nr, ns) <= W:
                        delta = d[p][v] + d[z][x] - d[p][z] - d[v][x] + surcharge(nr, ns)
                        if delta < best:
                            best, move = (delta, (1, a, b, True))
        if complete and r and s:

            def tops(nodes, target):
                table = {}
                for i in nodes:
                    if time.perf_counter() >= until:
                        return None
                    vals = []
                    prev = 0
                    for pos in range(len(target) + 1):
                        nxt = target[pos] if pos < len(target) else 0
                        val = d[prev][i] + d[i][nxt] - d[prev][nxt]
                        if len(vals) < 3 or val < vals[-1][0]:
                            vals.append((val, pos))
                            vals.sort()
                            if len(vals) > 3:
                                vals.pop()
                        prev = nxt
                    table[i] = vals
                return table
            into_r, into_s = (tops(s, r), tops(r, s))
            if into_r is None or into_s is None:
                complete = False
            else:
                for a, i in enumerate(r):
                    if time.perf_counter() >= until:
                        complete = False
                        break
                    p, z = (r[a - 1] if a else 0, r[a + 1] if a + 1 < len(r) else 0)
                    remr = d[p][z] - d[p][i] - d[i][z]
                    for b, j in enumerate(s):
                        nr, ns = (lr - w[i] + w[j], ls - w[j] + w[i])
                        if hard and max(nr, ns) > W:
                            continue
                        v, x = (s[b - 1] if b else 0, s[b + 1] if b + 1 < len(s) else 0)
                        rems = d[v][x] - d[v][j] - d[j][x]
                        br, ar = (d[p][j] + d[j][z] - d[p][z], a)
                        for delta, pos in into_r[j]:
                            if pos != a and pos != a + 1:
                                if delta < br:
                                    br, ar = (delta, pos - (pos > a))
                                break
                        bs, ass = (d[v][i] + d[i][x] - d[v][x], b)
                        for delta, pos in into_s[i]:
                            if pos != b and pos != b + 1:
                                if delta < bs:
                                    bs, ass = (delta, pos - (pos > b))
                                break
                        delta = remr + rems + br + bs + surcharge(nr, ns)
                        if delta < best:
                            best, move = (delta, (2, a, b, ar, ass))
        if move is None:
            if complete and hard:
                if len(self.pair_cache) >= self.cache_limit:
                    self.pair_cache.clear()
                self.pair_cache.add(key)
                self.pair_cache.add((key[1], key[0]))
            return None
        if move[0] == 0:
            _, rev, p, length, q, flip = move
            a, b = (s, r) if rev else (r, s)
            segment = a[p:p + length]
            if flip:
                segment.reverse()
            aa = a[:p] + a[p + length:]
            bb = b[:q] + segment + b[q:]
            return (bb, aa) if rev else (aa, bb)
        if move[0] == 1:
            _, a, b, flip = move
            if flip:
                return (r[:a] + s[:b][::-1], r[a:][::-1] + s[b:])
            return (r[:a] + s[b:], s[:b] + r[a:])
        _, a, b, ar, ass = move
        rr, ss = (r[:a] + r[a + 1:], s[:b] + s[b + 1:])
        rr.insert(ar, s[b])
        ss.insert(ass, r[a])
        return (rr, ss)

    def patterns(self, choices, routes, loads, until, penalty=math.inf):
        order = list(range(1, self.n + 1))
        self.rng.shuffle(order)
        d, w, W = (self.d, self.w, self.W)
        changed = False
        for i in order:
            if time.perf_counter() >= until:
                break
            removed, removal, virtual = ([], 0.0, {})
            for t in self.days[i][choices[i]]:
                for h, route in enumerate(routes[t]):
                    if i in route:
                        pos = route.index(i)
                        p = route[pos - 1] if pos else 0
                        z = route[pos + 1] if pos + 1 < len(route) else 0
                        removal += d[p][z] - d[p][i] - d[i][z]
                        if penalty != math.inf:
                            removal += penalty * (self.excess(loads[t][h] - w[i]) - self.excess(loads[t][h]))
                        virtual[t, h] = route[:pos] + route[pos + 1:]
                        removed.append((t, h, pos))
                        break
            by_day = {}
            for opt in self.options[i]:
                for t in self.days[i][opt]:
                    if t in by_day:
                        continue
                    if time.perf_counter() >= until:
                        return changed
                    best = (float('inf'), -1, -1)
                    for h, route in enumerate(routes[t]):
                        key = (t, h)
                        load = loads[t][h]
                        if key in virtual:
                            route, load = (virtual[key], load - w[i])
                        if penalty == math.inf and load + w[i] > W:
                            continue
                        charge = 0.0 if penalty == math.inf else penalty * (self.excess(load + w[i]) - self.excess(load))
                        extra, pos = self.insert_at(i, route)
                        if extra + charge < best[0]:
                            best = (extra + charge, h, pos)
                    by_day[t] = best
            delta, pick = (-self.eps, None)
            for opt in self.options[i]:
                value = removal + sum((by_day[t][0] for t in self.days[i][opt]))
                if value < delta:
                    delta, pick = (value, opt)
            if pick is not None:
                for t, h, pos in removed:
                    routes[t][h].pop(pos)
                    loads[t][h] -= w[i]
                for t in self.days[i][pick]:
                    _, h, pos = by_day[t]
                    routes[t][h].insert(pos, i)
                    loads[t][h] += w[i]
                choices[i] = pick
                changed = True
        return changed

    def polish(self, choices, routes, loads, until, penalty=math.inf):
        days = [t for t in range(self.T) if any(routes[t])]
        while time.perf_counter() < until:
            changed = False
            self.rng.shuffle(days)
            for t in days:
                day = routes[t]
                order = list(range(self.H))
                self.rng.shuffle(order)
                while time.perf_counter() < until:
                    for route in day:
                        if time.perf_counter() >= until:
                            return
                        self.intra(route, until)
                    moved = False
                    for p, a in enumerate(order):
                        if time.perf_counter() >= until:
                            return
                        for b in order[p + 1:]:
                            if penalty == math.inf and (not day[a] or not day[b]):
                                continue
                            if not day[a] and (not day[b]):
                                continue
                            pair = self.pair_move(day[a], day[b], loads[t][a], loads[t][b], until, penalty)
                            if pair is not None:
                                day[a], day[b] = pair
                                loads[t][a] = sum((self.w[i] for i in day[a]))
                                loads[t][b] = sum((self.w[i] for i in day[b]))
                                moved = changed = True
                                break
                            if time.perf_counter() >= until:
                                return
                        if moved:
                            break
                    if not moved:
                        break
            if time.perf_counter() >= until:
                return
            if self.patterns(choices, routes, loads, until, penalty):
                changed = True
                days = [t for t in range(self.T) if any(routes[t])]
            if penalty != math.inf and self.violation(loads) == 0.0:
                self.save(choices, routes)
                penalty = math.inf
            if not changed:
                return

    def small_exact(self, until):
        n, H, T = (self.n, self.H, self.T)
        seconds = max(0.0, until - time.perf_counter())
        work = seconds * 700000.0
        if n > math.log2(work + 1.0):
            return False
        size = 1 << n
        if n * n * size + 3 ** n > work:
            return False
        d, w, W = (self.d, self.w, self.W)
        inf = float('inf')
        mass = [0] * size
        tour = array.array('d', [inf]) * size
        tour[0] = 0.0
        dp = [array.array('d', [inf]) * size for _ in range(n)]
        for mask in range(1, size):
            if time.perf_counter() >= until:
                return False
            bit = mask & -mask
            mass[mask] = mass[mask ^ bit] + w[bit.bit_length()]
            if mass[mask] > W:
                continue
            ends = mask
            while ends:
                b = ends & -ends
                ends ^= b
                i = b.bit_length() - 1
                rest = mask ^ b
                val = inf
                if not rest:
                    val = d[0][i + 1]
                else:
                    prevs = rest
                    while prevs:
                        p = prevs & -prevs
                        prevs ^= p
                        j = p.bit_length() - 1
                        candidate = dp[j][rest] + d[j + 1][i + 1]
                        if candidate < val:
                            val = candidate
                dp[i][mask] = val
                tour[mask] = min(tour[mask], val + d[i + 1][0])
        memo, split = ({}, {})

        def pack(mask, k):
            if not mask:
                return 0.0
            if mass[mask] <= W:
                return tour[mask]
            if k <= 1 or mass[mask] > k * W:
                return inf
            k = min(k, mask.bit_count())
            key = (mask, k)
            if key in memo:
                return memo[key]
            if time.perf_counter() >= until:
                return None
            best, bestsub = (inf, 0)
            pivot = mask & -mask
            sub = mask
            while sub:
                if time.perf_counter() >= until:
                    return None
                if sub & pivot and mass[sub] <= W and (mass[mask ^ sub] <= (k - 1) * W):
                    if tour[sub] < best:
                        value = pack(mask ^ sub, k - 1)
                        if value is None:
                            return None
                        value += tour[sub]
                        if value < best:
                            best, bestsub = (value, sub)
                sub = sub - 1 & mask
            memo[key] = best
            split[key] = bestsub
            return best

        def route(mask):
            ends = [i for i in range(n) if mask & 1 << i]
            i = min(ends, key=lambda j: dp[j][mask] + d[j + 1][0])
            result = []
            while mask:
                result.append(i + 1)
                mask ^= 1 << i
                if mask:
                    i = min((j for j in range(n) if mask & 1 << j), key=lambda j: dp[j][mask] + d[j + 1][i + 1])
            result.reverse()
            return result

        def unpack(mask, k):
            day = []
            while mask:
                if mass[mask] <= W:
                    day.append(route(mask))
                    break
                k = min(k, mask.bit_count())
                sub = split[mask, k]
                day.append(route(sub))
                mask ^= sub
                k -= 1
            day.extend(([] for _ in range(H - len(day))))
            return day

        def bound(masks):
            total = 0.0
            for mask in masks:
                value = pack(mask, H)
                if value is None:
                    return None
                total += value
            return total
        chosen = self.best[1][:]
        incumbent_masks = [0] * T
        for i in range(1, n + 1):
            for t in self.days[i][chosen[i]]:
                incumbent_masks[t] |= 1 << i - 1
        value = bound(incumbent_masks)
        if value is None:
            return False
        if value < self.best[0] - self.eps:
            self.save(chosen, [unpack(mask, H) for mask in incumbent_masks])
        order = sorted(self.flexible, key=lambda i: (len(self.options[i]), -self.w[i], -self.d[0][i]))
        base = [0] * T
        for i in range(1, n + 1):
            if len(self.options[i]) == 1:
                chosen[i] = self.options[i][0]
                for t in self.days[i][chosen[i]]:
                    base[t] |= 1 << i - 1
        forced = [[0] * T for _ in range(len(order) + 1)]
        for depth in range(len(order) - 1, -1, -1):
            forced[depth] = forced[depth + 1][:]
            i = order[depth]
            for t in self.mandatory[i]:
                forced[depth][t] |= 1 << i - 1

        def visit(depth, masks, lower=None):
            if time.perf_counter() >= until:
                return False
            if lower is None:
                lower = bound([masks[t] | forced[depth][t] for t in range(T)])
            if lower is None:
                return False
            if lower >= self.best[0] - self.eps:
                return True
            if depth == len(order):
                self.save(chosen, [unpack(mask, H) for mask in masks])
                return True
            i = order[depth]
            branches = []
            for r in self.options[i]:
                nextm = masks[:]
                for t in self.days[i][r]:
                    nextm[t] |= 1 << i - 1
                value = bound([nextm[t] | forced[depth + 1][t] for t in range(T)])
                if value is None:
                    return False
                if value < self.best[0] - self.eps:
                    branches.append((value, r, nextm))
            branches.sort()
            for lower, r, nextm in branches:
                chosen[i] = r
                if not visit(depth + 1, nextm, lower):
                    return False
            return True
        self.exact_optimal = visit(0, base)
        return self.exact_optimal

    def excess(self, mass):
        if mass <= self.W:
            return 0.0
        v = (mass - self.W) / self.W
        return v + v * v

    def violation(self, loads):
        return sum((self.excess(v) for row in loads for v in row))

    def soft_seed(self, until):
        routes = [[[] for _ in range(self.H)] for _ in range(self.T)]
        loads = [[0] * self.H for _ in range(self.T)]
        choices = [-1] * (self.n + 1)
        order = sorted(range(1, self.n + 1), key=lambda i: (-self.w[i], len(self.options[i]), -self.d[0][i]))
        penalty = self.scale * (self.n + 1) ** 2
        d = self.d
        for i in order:
            if time.perf_counter() >= until:
                return None
            plans = {}
            for r in self.options[i]:
                for t in self.days[i][r]:
                    if t in plans:
                        continue
                    if time.perf_counter() >= until:
                        return None
                    best = (math.inf, -1, -1)
                    empty = False
                    for h, route in enumerate(routes[t]):
                        if not route:
                            if empty:
                                continue
                            empty = True
                        charge = penalty * (self.excess(loads[t][h] + self.w[i]) - self.excess(loads[t][h]))
                        extra, pos = self.insert_at(i, route)
                        if extra + charge < best[0]:
                            best = (extra + charge, h, pos)
                    plans[t] = best
            r = min(self.options[i], key=lambda r: sum((plans[t][0] for t in self.days[i][r])))
            choices[i] = r
            for t in self.days[i][r]:
                _, h, pos = plans[t]
                routes[t][h].insert(pos, i)
                loads[t][h] += self.w[i]
        return (choices, routes)

    def restore_capacity(self, choices, routes, loads, until):
        if time.perf_counter() >= until:
            return None
        choices, routes = (choices[:], self.clone(routes))
        loads = [row[:] for row in loads]
        self.collect(routes)
        overloaded = [t for t in range(self.T) if any((v > self.W for v in loads[t]))]
        pool_end = min(until, time.perf_counter() + 0.4 * max(0.0, until - time.perf_counter()))
        for p, t in enumerate(overloaded):
            now = time.perf_counter()
            if now >= pool_end:
                break
            if sum(loads[t]) <= self.H * self.W:
                self.partition_day(t, choices, routes, loads, now + (pool_end - now) / max(1, len(overloaded) - p))
        shadow = [row[:] for row in loads]
        locations = [[] for _ in range(self.n + 1)]
        for t, day in enumerate(routes):
            for h, route in enumerate(day):
                for i in route:
                    locations[i].append((t, h))
        removed = set()
        for t, day in enumerate(routes):
            for h, route in enumerate(day):
                if shadow[t][h] <= self.W:
                    continue
                order = sorted(route, key=lambda i: -self.w[i] * self.rng.uniform(0.7, 1.3))
                for i in order:
                    if shadow[t][h] <= self.W:
                        break
                    if time.perf_counter() >= until:
                        return None
                    if i not in removed:
                        removed.add(i)
                        for tt, hh in locations[i]:
                            shadow[tt][hh] -= self.w[i]
        if not removed:
            return (choices[:], self.clone(routes), self.route_loads(routes))
        seed = self.rng.choice(sorted(removed))
        neighbors = sorted((i for i in range(1, self.n + 1) if i not in removed), key=lambda i: self.d[seed][i] / self.scale + 0.3 * self.rng.random())
        extra = min(len(neighbors), max(1, int(math.sqrt(len(removed) * self.intensity))))
        removed.update(neighbors[:extra])
        trial = [[[i for i in route if i not in removed] for route in day] for day in routes]
        pick = choices[:]
        mass = self.route_loads(trial)
        if self.repair(removed, pick, trial, mass, 0.0, {}, until):
            return (pick, trial, mass)
        return None

    def partition_day(self, t, choices, routes, loads, until):
        day = routes[t]
        target = sum((self.bits[i] for r in day for i in r))
        if not target:
            return
        if sum(loads[t]) > self.H * self.W:
            return
        anchor = sum((self.tour_cost(r) for r in day))
        upper = anchor if all((v <= self.W for v in loads[t])) else math.inf

        def commit(tours, value):
            nonlocal upper
            if value >= upper - self.eps:
                return
            day[:] = [list(r) for r in tours] + [[] for _ in range(self.H - len(tours))]
            loads[t][:] = [sum((self.w[i] for i in r)) for r in day]
            upper = value
            self.save(choices, routes)
        cached = self.partitions.get(target)
        if cached is not None:
            version, value, tours, finished = cached
            commit(tours, value)
            if finished and version == self.pool_version:
                return
        version = self.pool_version
        catalogue = {}
        for r in day:
            if r and sum((self.w[i] for i in r)) <= self.W:
                catalogue[sum((self.bits[i] for i in r))] = (self.tour_cost(r), tuple(r))
        nodes = [i for r in day for i in r]
        for i in nodes:
            catalogue[self.bits[i]] = (self.d[0][i] + self.d[i][0], (i,))
        for mask, (value, tour) in self.pool.items():
            if time.perf_counter() >= until:
                return
            common = mask & target
            if common.bit_count() < 2:
                continue
            if common != mask:
                tour = tuple((i for i in tour if self.bits[i] & target))
                value = self.tour_cost(tour)
            old = catalogue.get(common)
            if old is None or value < old[0]:
                catalogue[common] = (value, tour)
        full = catalogue.get(target)
        if full is not None:
            commit((full[1],), full[0])
        if self.H == 1:
            if math.isfinite(upper):
                self.partitions[target] = (version, upper, tuple((tuple(r) for r in day if r)), True)
            return
        columns = [(mask, value, tour) for mask, (value, tour) in catalogue.items() if value < upper - self.eps]
        columns.sort(key=lambda c: (c[1] / c[0].bit_count(), -c[0].bit_count()))
        if not columns or time.perf_counter() >= until:
            return
        memberships = [0] * (self.n + 1)
        share = [math.inf] * (self.n + 1)
        for j, (mask, value, tour) in enumerate(columns):
            if time.perf_counter() >= until:
                return
            bit, price = (1 << j, value / len(tour))
            for i in tour:
                memberships[i] |= bit
                if price < share[i]:
                    share[i] = price
        if any((not memberships[i] for i in nodes)):
            return
        multiplier = (upper if math.isfinite(upper) else anchor) / (self.H + 1) * 0.25
        dual = share[:]
        slack = [max(0.0, c + multiplier - sum((share[i] for i in r))) for _, c, r in columns]
        for i in sorted(nodes, key=lambda i: (-self.w[i], -self.d[0][i])):
            if time.perf_counter() >= until:
                return
            bits, increase = (memberships[i], math.inf)
            while bits:
                b = bits & -bits
                bits ^= b
                increase = min(increase, slack[b.bit_length() - 1])
            if increase > 0.0:
                dual[i] += increase
                bits = memberships[i]
                while bits:
                    b = bits & -bits
                    bits ^= b
                    j = b.bit_length() - 1
                    slack[j] = max(0.0, slack[j] - increase)
        bounds, dominance, conflicts = ({}, {}, {})

        def measure(mask):
            old = bounds.get(mask)
            if old is not None:
                return old
            bits, mass, simple, priced, radial, farthest = (mask, 0, 0.0, 0.0, 0.0, 0.0)
            while bits:
                bit = bits & -bits
                bits ^= bit
                i = bit.bit_length()
                mass += self.w[i]
                simple += share[i]
                priced += dual[i]
                radial += 2.0 * self.q[i] * self.d[0][i]
                farthest = max(farthest, 2.0 * self.d[0][i])
            result = (mass, max(simple, radial, farthest), priced)
            bounds[mask] = result
            return result

        def frame(mask, k, available, cost, plan):
            if not k or time.perf_counter() >= until:
                return None
            k = min(k, mask.bit_count())
            mass, simple, priced = measure(mask)
            if mass > k * self.W or cost + max(simple, priced - k * multiplier) >= upper - self.eps:
                return None
            key = (mask, k)
            if cost >= dominance.get(key, math.inf) - self.eps:
                return None
            dominance[key] = cost
            bits, least, alternatives = (mask, len(columns) + 1, 0)
            while bits:
                bit = bits & -bits
                bits ^= bit
                choices_here = memberships[bit.bit_length()] & available
                count = choices_here.bit_count()
                if not count:
                    return None
                if count < least:
                    least, alternatives = (count, choices_here)
                    if count == 1:
                        break
            return [mask, k, available, cost, plan, alternatives]
        root = frame(target, self.H, (1 << len(columns)) - 1, 0.0, ())
        stack = [root] if root is not None else []
        while stack and time.perf_counter() < until:
            f = stack[-1]
            if not f[5]:
                stack.pop()
                continue
            bit = f[5] & -f[5]
            f[5] ^= bit
            j = bit.bit_length() - 1
            mask, value, tour = columns[j]
            cost = f[3] + value
            if cost >= upper - self.eps:
                continue
            remaining = f[0] ^ mask
            plan = f[4] + (j,)
            if not remaining:
                commit(tuple((columns[a][2] for a in plan)), cost)
                continue
            if f[1] <= 1:
                continue
            blocked = conflicts.get(j)
            if blocked is None:
                blocked = 0
                for i in tour:
                    blocked |= memberships[i]
                conflicts[j] = blocked
            child = frame(remaining, f[1] - 1, f[2] & ~blocked, cost, plan)
            if child is not None:
                stack.append(child)
        if len(self.partitions) >= self.cache_limit:
            self.partitions.clear()
        finished = not stack and time.perf_counter() < until
        if math.isfinite(upper):
            self.partitions[target] = (version, upper, tuple((tuple(r) for r in day if r)), finished)

    def recombine(self, choices, routes, loads, until):
        days = [t for t in range(self.T) if any(routes[t])]
        self.rng.shuffle(days)
        for p, t in enumerate(days):
            now = time.perf_counter()
            if now >= until:
                return
            self.partition_day(t, choices, routes, loads, now + (until - now) / math.sqrt(len(days) - p))

    def destroy(self, choices, routes, count, mode):
        n, rng, d = (self.n, self.rng, self.d)
        customers = list(range(1, n + 1))
        seed, preferred = (rng.choice(customers), {})
        if mode == 0:
            removed = set(rng.sample(customers, count))
        elif mode == 1:
            mask = self.masks[seed][choices[seed]]
            ranking = sorted(customers, key=lambda i: d[seed][i] / self.scale + 0.3 * (mask ^ self.masks[i][choices[i]]).bit_count() + 0.2 * rng.random())
            removed = set(ranking[:count])
        elif mode == 2:
            gains = [0.0] * (n + 1)
            for day in routes:
                for route in day:
                    for p, i in enumerate(route):
                        a = route[p - 1] if p else 0
                        b = route[p + 1] if p + 1 < len(route) else 0
                        gains[i] += d[a][i] + d[i][b] - d[a][b]
            ranking = sorted(customers, key=lambda i: -gains[i] * (0.5 + rng.random()))
            removed = set(ranking[:count])
        elif mode == 3:
            active = [r for day in routes for r in day if r]
            rng.shuffle(active)
            removed = set()
            for r in active:
                if len(removed) >= count:
                    break
                length = min(len(r), max(1, count - len(removed)))
                a = rng.randrange(len(r) - length + 1)
                removed.update(r[a:a + length])
        elif mode == 4 and self.flexible:
            seed = rng.choice(self.flexible)
            alt = rng.choice([r for r in self.options[seed] if r != choices[seed]])
            preferred[seed] = alt
            mask = self.masks[seed][alt]
            ranking = sorted((i for i in customers if i != seed), key=lambda i: d[seed][i] / self.scale + rng.random() * 0.25 + (0.0 if mask & self.masks[i][choices[i]] else 1.0))
            removed = {seed}
            removed.update(ranking[:max(0, count - 1)])
        else:
            active = [r for day in routes for r in day if r]
            removed = set(rng.choice(active))
            if len(removed) < count:
                pool = [i for i in customers if i not in removed]
                removed.update(rng.sample(pool, min(count - len(removed), len(pool))))
        return (removed, preferred)

    def improve(self):
        if self.proven_infeasible or not self.n:
            return
        n, rng = (self.n, self.rng)
        if self.best is not None:
            if self.best[0] == 0.0:
                return
            _, cc, rr = self.best
            choices, routes = (cc[:], self.clone(rr))
        elif self.seed is not None:
            choices, routes = self.seed
        else:
            return
        loads = self.route_loads(routes)
        local_slice = self.budget / (55.0 + 6.0 * math.sqrt(n))
        self.polish(choices, routes, loads, min(self.end, time.perf_counter() + self.budget / (5.0 + math.sqrt(n))), math.inf if self.violation(loads) == 0.0 else self.penalty)
        self.save(choices, routes)
        if self.best is not None and time.perf_counter() < self.end:
            self.small_exact(min(self.end, time.perf_counter() + (self.end - time.perf_counter()) / (3.0 + math.sqrt(self.intensity))))
        if self.exact_optimal:
            return
        if self.best is not None:
            value, cc, rr = self.best
            choices, routes = (cc[:], self.clone(rr))
            loads = self.route_loads(routes)
        value = self.cost(routes)
        violation = self.violation(loads)
        self.collect(routes)
        self.collect([[[i] for i in range(1, n + 1)]])
        last_best = time.perf_counter()
        intensity = self.intensity
        max_remove = min(n, max(1, int(math.sqrt(n * intensity))))
        weights = [1.0] * 6
        adaptation = self.budget / (3.0 + intensity + math.sqrt(n))
        next_adapt = time.perf_counter() + adaptation
        pool_interval = self.budget / (2.0 + intensity)
        pool_slice = self.budget / (24.0 + 4.0 * math.sqrt(n))
        next_pool = time.perf_counter() + pool_interval
        penalty = self.penalty
        feasible_time = observed_time = 0.0
        while time.perf_counter() < self.end:
            now = time.perf_counter()
            if self.best is not None and now >= next_pool:
                oldbest, cc, rr = self.best
                pc, pr = (cc[:], self.clone(rr))
                self.recombine(pc, pr, self.route_loads(pr), min(self.end, now + pool_slice))
                if self.best[0] < oldbest - self.eps:
                    value, cc, rr = self.best
                    choices, routes = (cc[:], self.clone(rr))
                    loads = self.route_loads(routes)
                    violation = 0.0
                    last_best = time.perf_counter()
                next_pool = time.perf_counter() + pool_interval
                continue
            stagnation = (now - last_best) / max(1e-09, self.budget)
            if stagnation > 1.0 / (2.0 + intensity) and self.best is not None and (rng.random() < 0.1):
                value, cc, rr = self.best
                choices, routes = (cc[:], self.clone(rr))
                loads = self.route_loads(routes)
                violation = 0.0
            ceiling = min(n, max_remove + int(n * min(0.35, stagnation)))
            count = rng.randint(1, max(1, ceiling))
            mode = rng.choices(range(len(weights)), weights=weights)[0]
            removed, preferred = self.destroy(choices, routes, count, mode)
            trial = [[[i for i in r if i not in removed] for r in day] for day in routes]
            trial_choices = choices[:]
            self.collect(trial)
            noise = rng.random() * min(0.25, 0.035 * intensity) if rng.random() < 0.35 else 0.0
            oldbest = self.best[0] if self.best is not None else math.inf
            repair_end = min(self.end, time.perf_counter() + self.budget / (20.0 + 2.0 * math.sqrt(n)))
            if not self.regret_repair(removed, trial_choices, trial, repair_end, noise, preferred, penalty):
                weights[mode] = 0.97 * weights[mode] + 0.03 * 0.2
                continue
            trial_loads = self.route_loads(trial)
            excess = self.violation(trial_loads)
            if excess == 0.0:
                self.save(trial_choices, trial)
            self.polish(trial_choices, trial, trial_loads, min(self.end, time.perf_counter() + local_slice), math.inf if excess == 0.0 else penalty)
            excess = self.violation(trial_loads)
            trial_value = self.cost(trial)
            if excess > 0.0 and time.perf_counter() < self.end:
                if self.best is None or trial_value < self.best[0] or rng.random() < 1.0 / (1.0 + intensity):
                    restored = self.restore_capacity(trial_choices, trial, trial_loads, min(self.end, time.perf_counter() + local_slice))
                    if restored is not None:
                        rc, rr, rl = restored
                        self.polish(rc, rr, rl, min(self.end, time.perf_counter() + local_slice))
                        rv = self.cost(rr)
                        self.save(rc, rr, rv)
                        if rv <= trial_value + penalty * excess:
                            trial_choices, trial, trial_loads = (rc, rr, rl)
                            trial_value, excess = (rv, 0.0)
            if excess == 0.0:
                self.save(trial_choices, trial, trial_value)
            else:
                self.collect(trial)
            progress = min(1.0, (time.perf_counter() - self.started) / max(1e-09, self.budget))
            visits = max(1, sum((len(self.days[i][trial_choices[i]]) for i in range(1, n + 1))))
            anchor = self.best[0] if self.best is not None else max(self.scale, trial_value)
            temperature = max(self.eps, anchor / visits * 0.055 * 0.02 ** progress * (1.0 + min(2.0, 5.0 * stagnation)))
            delta = trial_value - value + penalty * (excess - violation)
            accepted = delta <= 0.0 or rng.random() < math.exp(min(0.0, -delta / temperature))
            reward = 0.4
            if self.best is not None and self.best[0] < oldbest - self.eps:
                last_best = time.perf_counter()
                reward = 5.0 + (0.0 if not math.isfinite(oldbest) else (oldbest - self.best[0]) / self.scale)
            elif delta < -self.eps:
                reward = 2.0
            elif accepted:
                reward = 1.0
            weights[mode] = 0.97 * weights[mode] + 0.03 * reward
            if accepted:
                choices, routes, loads = (trial_choices, trial, trial_loads)
                value, violation = (trial_value, excess)
            elapsed = max(0.0, time.perf_counter() - now)
            observed_time += elapsed
            feasible_time += elapsed if excess == 0.0 else 0.0
            if time.perf_counter() >= next_adapt:
                ratio = feasible_time / max(1e-12, observed_time)
                penalty *= math.exp(0.9 * (0.65 - ratio))
                penalty = min(self.scale * (n + 1) ** 2, max(self.scale * 0.5, penalty))
                self.penalty = penalty
                feasible_time = observed_time = 0.0
                next_adapt = time.perf_counter() + adaptation

    def result(self):
        # Without a completed proof, a search timeout must not be falsely reported as mathematical infeasibility.
        if self.best is None:
            return {'status': 'infeasible' if self.proven_infeasible else 'unknown', 'x': None, 'y': None, 'u': None}
        _, choices, routes = self.best
        x = [[int(r == choices[i]) for r in range(len(row))] for i, row in enumerate(self.candidates, 1)]
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
                prefix, prev = (0, 0)
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
            search.end = min(search.end, started + search.budget - max(search.reserve, 2.0 * rendering + 0.001))
        if not search.proven_infeasible:
            search.improve()
    except Exception:
        pass
    if search is not None:
        try:
            return search.result()
        except Exception:
            pass
    if fallback is not None:
        return fallback
    return {'status': 'unknown', 'x': None, 'y': None, 'u': None}
