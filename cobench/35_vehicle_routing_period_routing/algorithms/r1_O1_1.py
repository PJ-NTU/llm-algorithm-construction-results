"""Schedule-coupled exact subset search with adaptive large neighbourhood search
Combines integer capacity backtracking, complementary schedule swaps, segment exchanges and subset dynamic programming, continually improving and keeping the shortest complete feasible solution.
"""

import array
import fractions
import itertools
import math
import random
import time



class _SubsetRoutes:
    """Enumerate capacity-feasible routes over a given customer set, then partition them exactly into vehicles."""

    def __init__(self, nodes, w, capacity, distances):
        self.nodes = tuple(nodes)
        self.m = len(nodes)
        self.size = 1 << self.m
        self.W = capacity
        self.w = [w[i] for i in nodes]
        self.d0 = [distances[0][i] for i in nodes]
        self.d = [[distances[i][j] for j in nodes] for i in nodes]
        self.mass = [0] * self.size
        self.radial = [0.0] * self.size
        self.farthest = [0.0] * self.size
        self.path = array.array('d', [math.inf]) * (self.size * self.m)
        self.tour = array.array('d', [math.inf]) * self.size
        self.tour[0] = 0.0
        self.last = [-1] * self.size
        self.cursor = 1
        self.fleets = {}
        self.route_cache = {}

    def build(self, until):
        """Held-Karp state can resume across time slices; no half-finished state is saved."""
        m, mass, d, d0, path = self.m, self.mass, self.d, self.d0, self.path
        while self.cursor < self.size and time.perf_counter() < until:
            mask = self.cursor
            bit = mask & -mask
            j = bit.bit_length() - 1
            rest = mask ^ bit
            mass[mask] = mass[rest] + self.w[j]
            self.radial[mask] = self.radial[rest] + 2.0 * d0[j] * (self.w[j] / self.W)
            self.farthest[mask] = max(self.farthest[rest], 2.0 * d0[j])
            if mass[mask] <= self.W:
                ends = mask
                best, end = math.inf, -1
                while ends:
                    bit = ends & -ends
                    ends ^= bit
                    j = bit.bit_length() - 1
                    previous = mask ^ bit
                    if not previous:
                        value = d0[j]
                    else:
                        value = math.inf
                        offset, bits = previous * m, previous
                        row = d[j]
                        while bits:
                            low = bits & -bits
                            bits ^= low
                            k = low.bit_length() - 1
                            candidate = path[offset + k] + row[k]
                            if candidate < value:
                                value = candidate
                    path[mask * m + j] = value
                    if value + d0[j] < best:
                        best, end = value + d0[j], j
                self.tour[mask], self.last[mask] = best, end
            self.cursor += 1
        return self.cursor == self.size

    def lower(self, mask):
        return max(self.radial[mask], self.farthest[mask])

    def fleet(self, mask, vehicles, until):
        """Return the exact distance and route subset; None internally means only that the time slice ran out."""
        if not mask:
            return 0.0, ()
        if time.perf_counter() >= until:
            return None
        if self.mass[mask] > vehicles * self.W or not vehicles:
            return math.inf, ()
        if self.mass[mask] <= self.W:
            return self.tour[mask], (mask,)
        vehicles = min(vehicles, mask.bit_count())
        key = mask, vehicles
        old = self.fleets.get(key)
        if old is not None:
            return old
        # Fix the route containing one customer to eliminate the permutation symmetry of vehicle labels.
        pivot = mask & -mask
        remaining, best, plan = mask ^ pivot, math.inf, ()
        sub = remaining
        while True:
            if time.perf_counter() >= until:
                return None
            route = sub | pivot
            residual = mask ^ route
            value = self.tour[route]
            if (value < best and self.mass[residual] <= (vehicles - 1) * self.W
                    and value + self.lower(residual) < best):
                tail = self.fleet(residual, vehicles - 1, until)
                if tail is None:
                    return None
                if value + tail[0] < best:
                    best, plan = value + tail[0], (route,) + tail[1]
            if not sub:
                break
            sub = (sub - 1) & remaining
        result = best, plan
        self.fleets[key] = result
        return result

    def route(self, mask):
        saved = self.route_cache.get(mask)
        if saved is not None:
            return list(saved)
        original, reverse = mask, []
        j = self.last[mask]
        while mask:
            reverse.append(self.nodes[j])
            mask ^= 1 << j
            if mask:
                offset = mask * self.m
                bits, best, previous = mask, math.inf, -1
                while bits:
                    bit = bits & -bits
                    bits ^= bit
                    k = bit.bit_length() - 1
                    value = self.path[offset + k] + self.d[k][j]
                    if value < best:
                        best, previous = value, k
                j = previous
        reverse.reverse()
        self.route_cache[original] = tuple(reverse)
        return reverse


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
        self.insert_cache = {}
        self.pool = {}
        self.pool_version = 0
        self.pool_limit = max(self.n, int((self.n + self.T + 1) * self.intensity * 12))
        self.partitions = {}
        self.exchange_span = max(2, int(1 + math.log1p(self.budget)))
        self.schedule_pair_cache = set()
        self.exact = None
        self.exact_stack = None
        self.exact_seen = set()
        self.exact_order = None
        self.exact_forced = None
        self.optimal = False
        self.block_cache = {}
        self.bits = [0] + [1 << i for i in range(self.n)]
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
        self.collect(routes)
        if (self.best is None or value < self.best[0]) and self.valid(choices, routes):
            self.best = (value, choices[:], self.clone(routes))

    def insert_at(self, i, route):
        """Geometry is shared across unchanged routes, dates and ALNS trials."""
        if not route:
            return self.d[0][i] + self.d[i][0], 0
        key = i, tuple(route)
        old = self.insert_cache.get(key)
        if old is not None:
            return old
        d, previous, best, pos = self.d, 0, math.inf, 0
        for p in range(len(route) + 1):
            nxt = route[p] if p < len(route) else 0
            v = d[previous][i] + d[i][nxt] - d[previous][nxt]
            if v < best:
                best, pos = v, p
            previous = nxt
        if len(self.insert_cache) >= self.cache_limit * 2:
            self.insert_cache.clear()
        self.insert_cache[key] = best, pos
        return best, pos

    def collect(self, routes):
        """A route represents original y arcs; no auxiliary output variables."""
        for day in routes:
            for route in day:
                if not route:
                    continue
                mask = sum(self.bits[i] for i in route)
                value = self.tour_cost(route)
                old = self.pool.get(mask)
                if old is None or value < old[0]:
                    self.pool[mask] = value, tuple(route)
                    self.pool_version += 1
        if len(self.pool) > self.pool_limit * 1.2:
            keep = set()
            if self.best is not None:
                keep = {sum(self.bits[i] for i in r)
                        for day in self.best[2] for r in day if r}
            for mask in list(self.pool):
                if len(self.pool) <= self.pool_limit:
                    break
                if mask not in keep:
                    del self.pool[mask]

    def route_loads(self, routes):
        return [[sum(self.w[i] for i in r) for r in day] for day in routes]

    def insertion_options(self, i, routes, loads, until, noise, initial):
        """Rank all schedules and feasible bins, retaining cheapest positions.

        Initial feasibility search merges equal-load bins, retaining their
        cheapest representative. Improvement preserves spatially different
        nonempty routes even at equal loads. None denotes a timed-out slice.
        """
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
        w, W = self.w, self.W
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
                            best, pos = self.insert_at(i, route)
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

    def repair(self, removed, choices, routes, loads, noise, preferred):
        # Restrict only this transactional trial, including capacity fallback.
        old_options = {i: self.options[i] for i in preferred}
        try:
            for i, r in preferred.items():
                self.options[i] = [r]
            return self.repair_free(removed, choices, routes, loads, noise)
        finally:
            for i, opts in old_options.items():
                self.options[i] = opts

    def repair_free(self, removed, choices, routes, loads, noise):
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
        if move is None:
            # Multi-customer contiguous segment swap; both segments may be reversed, overcoming the capacity barrier of single-customer swaps.
            span_a, span_b = min(m, self.exchange_span), min(n, self.exchange_span)
            for x in range(m):
                if time.perf_counter() >= until:
                    return best, move, False
                ap = a[x - 1] if x else 0
                first_a = a[x]
                for length_a in range(1, min(span_a, m - x) + 1):
                    last_a = a[x + length_a - 1]
                    az = a[x + length_a] if x + length_a < m else 0
                    wa = pa[x + length_a] - pa[x]
                    remove_a = d[ap][first_a] + d[last_a][az]
                    for y in range(n):
                        if time.perf_counter() >= until:
                            return best, move, False
                        bp = b[y - 1] if y else 0
                        first_b = b[y]
                        for length_b in range(1, min(span_b, n - y) + 1):
                            if length_a == length_b == 1:
                                continue
                            wb = pb[y + length_b] - pb[y]
                            if la - wa + wb > W or lb - wb + wa > W:
                                continue
                            last_b = b[y + length_b - 1]
                            bz = b[y + length_b] if y + length_b < n else 0
                            fa = d[ap][first_b] + d[last_b][az]
                            ra = d[ap][last_b] + d[first_b][az]
                            fb = d[bp][first_a] + d[last_a][bz]
                            rb = d[bp][last_a] + d[first_a][bz]
                            delta = (min(fa, ra) + min(fb, rb) - remove_a
                                     - d[bp][first_b] - d[last_b][bz])
                            if delta < best:
                                best = delta
                                move = ('cross', x, length_a, y, length_b, ra < fa, rb < fb)
        return best, move, True

    @staticmethod
    def apply_pair(a, b, move):
        if move[0] == 'tail':
            _, x, y, reverse = move
            if reverse:
                return a[:x] + b[:y][::-1], a[x:][::-1] + b[y:]
            return a[:x] + b[y:], b[:y] + a[x:]
        aa, bb = a[:], b[:]
        if move[0] == 'cross':
            _, x, length_a, y, length_b, reverse_a, reverse_b = move
            part_a, part_b = a[x:x + length_a], b[y:y + length_b]
            if reverse_a:
                part_b.reverse()
            if reverse_b:
                part_a.reverse()
            aa[x:x + length_a] = part_b
            bb[y:y + length_b] = part_a
        elif move[0] == 'rel':
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

    def schedule_exchange(self, choices, routes, loads, until):
        """Switch two complementary schedules simultaneously; shared service days stay fixed, net change computed exactly day by day."""
        groups = {}
        for i in range(1, self.n + 1):
            if time.perf_counter() >= until:
                return False
            old = self.masks[i][choices[i]]
            for r in self.options[i]:
                new = self.masks[i][r]
                if new != old:
                    key = old & ~new, new & ~old
                    groups.setdefault(key, []).append((i, r))
        keys = [key for key in groups if key[0] < key[1] and key[::-1] in groups]
        if not keys:
            return False
        self.rng.shuffle(keys)
        positions = [{} for _ in range(self.T)]
        for t, day in enumerate(routes):
            for h, route in enumerate(day):
                for p, i in enumerate(route):
                    positions[t][i] = h, p
        for remove_mask, add_mask in keys:
            left, right = groups[(remove_mask, add_mask)], groups[(add_mask, remove_mask)]
            self.rng.shuffle(left)
            self.rng.shuffle(right)
            affected = [t for t in range(self.T) if (remove_mask | add_mask) & (1 << t)]
            signature = tuple(tuple(tuple(r) for r in routes[t]) for t in affected)
            for i, ri in left:
                for j, rj in right:
                    if time.perf_counter() >= until:
                        return False
                    if i == j:
                        continue
                    cache_key = i, ri, j, rj, signature
                    if cache_key in self.schedule_pair_cache:
                        continue
                    delta, plan, possible = 0.0, [], True
                    for t in affected:
                        if time.perf_counter() >= until:
                            return False
                        outgoing, incoming = (i, j) if remove_mask & (1 << t) else (j, i)
                        h0, p0 = positions[t][outgoing]
                        route = routes[t][h0]
                        prev = route[p0 - 1] if p0 else 0
                        nxt = route[p0 + 1] if p0 + 1 < len(route) else 0
                        removal = self.d[prev][nxt] - self.d[prev][outgoing] - self.d[outgoing][nxt]
                        shortened = route[:p0] + route[p0 + 1:]
                        best, placement = math.inf, None
                        for h, other in enumerate(routes[t]):
                            load = loads[t][h] - (self.w[outgoing] if h == h0 else 0)
                            if load + self.w[incoming] > self.W:
                                continue
                            cost, pos = self.insert_at(incoming, shortened if h == h0 else other)
                            if cost < best:
                                best, placement = cost, (h, pos)
                        if placement is None:
                            possible = False
                            break
                        delta += removal + best
                        plan.append((t, h0, p0, outgoing, incoming, placement))
                    if possible and delta < -self.eps:
                        for t, h0, p0, outgoing, incoming, (h, pos) in plan:
                            routes[t][h0].pop(p0)
                            loads[t][h0] -= self.w[outgoing]
                            routes[t][h].insert(pos, incoming)
                            loads[t][h] += self.w[incoming]
                        choices[i], choices[j] = ri, rj
                        return True
                    if len(self.schedule_pair_cache) >= self.cache_limit:
                        self.schedule_pair_cache.clear()
                    self.schedule_pair_cache.add(cache_key)
        return False

    @staticmethod
    def subset_affordable(size, seconds):
        # Estimate admission from remaining wall-clock time and state count, rather than a hard-coded instance-size threshold.
        return (size > 0 and seconds > 0.0 and
                size * math.log(2.0) + 2.0 * math.log(size)
                <= math.log(max(1.0, seconds * 1000000.0)))

    def exact_global(self, until):
        """Jointly enumerate schedules on small sets; the vehicle partition for each day's set is solved exactly by subset DP."""
        if self.optimal or not self.n or time.perf_counter() >= until:
            return
        if self.exact is None:
            if not self.subset_affordable(self.n, self.budget / (2 + self.intensity)):
                return
            self.exact = _SubsetRoutes(tuple(range(1, self.n + 1)), self.w, self.W, self.d)
        oracle = self.exact
        if not oracle.build(until):
            return
        if self.exact_stack is None:
            self.exact_order = sorted(range(1, self.n + 1), key=lambda i: (
                len(self.options[i]), -len(self.mandatory[i]), -self.w[i], -self.d[0][i]))
            self.exact_forced = [[0] * self.T for _ in range(self.n + 1)]
            for depth in range(self.n - 1, -1, -1):
                self.exact_forced[depth] = self.exact_forced[depth + 1][:]
                i = self.exact_order[depth]
                for t in self.mandatory[i]:
                    self.exact_forced[depth][t] |= self.bits[i]
            self.exact_stack = [(0, (0,) * self.T, ())]
        # First improve the routes of existing complete schedules, without waiting for the joint enumeration to finish.
        _, current, incumbent = self.best
        routes = self.clone(incumbent)
        changed = False
        for t in range(self.T):
            target = sum(self.bits[i] for route in routes[t] for i in route)
            answer = oracle.fleet(target, self.H, until)
            if answer is None:
                if changed:
                    self.save(current, routes)
                return
            old = sum(self.tour_cost(r) for r in routes[t])
            if answer[0] < old - self.eps:
                tours = [oracle.route(mask) for mask in answer[1]]
                routes[t] = tours + [[] for _ in range(self.H - len(tours))]
                changed = True
        if changed:
            self.save(current, routes)
        while self.exact_stack and time.perf_counter() < until:
            depth, assigned, chosen = state = self.exact_stack.pop()
            key = depth, assigned
            if key in self.exact_seen:
                continue
            bound, plans, timed_out = 0.0, [], False
            for t in range(self.T):
                mask = assigned[t] | self.exact_forced[depth][t]
                answer = oracle.fleet(mask, self.H, until)
                if answer is None:
                    timed_out = True
                    break
                bound += answer[0]
                plans.append(answer[1])
                if bound >= self.best[0] - self.eps:
                    break
            if timed_out:
                self.exact_stack.append(state)
                return
            if len(self.exact_seen) < self.cache_limit * self.n:
                self.exact_seen.add(key)
            if bound >= self.best[0] - self.eps:
                continue
            if depth == self.n:
                selected = [-1] * (self.n + 1)
                for i, r in zip(self.exact_order, chosen):
                    selected[i] = r
                tours = []
                for masks in plans:
                    day = [oracle.route(mask) for mask in masks]
                    tours.append(day + [[] for _ in range(self.H - len(day))])
                self.save(selected, tours)
                continue
            i = self.exact_order[depth]
            preferred = self.best[1][i]
            options = sorted(self.options[i], key=lambda r: (
                r != preferred, len(self.days[i][r]),
                sum(oracle.mass[assigned[t]] for t in self.days[i][r])))
            for r in reversed(options):
                masks = list(assigned)
                for t in self.days[i][r]:
                    masks[t] |= self.bits[i]
                if all(oracle.mass[masks[t] | self.exact_forced[depth + 1][t]]
                       <= self.H * self.W for t in range(self.T)):
                    self.exact_stack.append((depth + 1, tuple(masks), chosen + (r,)))
        if not self.exact_stack:
            self.optimal = True

    def exact_blocks(self, t, choices, routes, loads, until):
        """On large instances, jointly re-optimise a group of spatially close routes without freezing vehicle assignments within the group."""
        available = max(0.0, until - time.perf_counter())
        if not available:
            return
        day = routes[t]
        active = [h for h, route in enumerate(day) if route]
        self.rng.shuffle(active)
        for seed in active:
            if time.perf_counter() >= until:
                return
            anchor = self.rng.choice(day[seed])
            order = sorted((h for h in active if h != seed), key=lambda h:
                           min(self.d[anchor][i] for i in day[h]))
            group, count = [seed], len(day[seed])
            if not self.subset_affordable(count, available):
                continue
            for h in order:
                if self.subset_affordable(count + len(day[h]), available):
                    group.append(h)
                    count += len(day[h])
            nodes = tuple(sorted(i for h in group for i in day[h]))
            oracle = self.block_cache.get(nodes)
            if oracle is None:
                if len(self.block_cache) >= max(1, int(self.intensity)):
                    del self.block_cache[next(iter(self.block_cache))]
                oracle = _SubsetRoutes(nodes, self.w, self.W, self.d)
                self.block_cache[nodes] = oracle
            if not oracle.build(until):
                return
            answer = oracle.fleet(oracle.size - 1, len(group), until)
            if answer is None:
                return
            old = sum(self.tour_cost(day[h]) for h in group)
            if answer[0] < old - self.eps:
                new = [oracle.route(mask) for mask in answer[1]]
                new += [[] for _ in range(len(group) - len(new))]
                for h, route in zip(group, new):
                    day[h] = route
                    loads[t][h] = sum(self.w[i] for i in route)
                self.save(choices, routes)
                return

    def polish(self, choices, routes, loads, until):
        while time.perf_counter() < until:
            days = list(range(self.T))
            self.rng.shuffle(days)
            for t in days:
                if time.perf_counter() >= until:
                    return
                self.day_descent(routes[t], loads[t], until)
            changed = self.schedule_sweep(choices, routes, loads, until)
            if not changed:
                changed = self.schedule_exchange(choices, routes, loads, until)
            if not changed:
                return

    def partition_day(self, t, choices, routes, loads, until):
        """Anytime exact partition over a restricted pool of original y tours.

        Columns are stored tours, not additional returned/model variables.
        Shortcutting permits their reuse on different service-day subsets.
        Every complete improvement is published before further branching.
        """
        day = routes[t]
        target = sum(self.bits[i] for r in day for i in r)
        if not target:
            return
        upper = sum(self.tour_cost(r) for r in day)

        def commit(tours, value):
            nonlocal upper
            if value >= upper - self.eps:
                return
            day[:] = [list(r) for r in tours] + [[] for _ in range(self.H - len(tours))]
            loads[t][:] = [sum(self.w[i] for i in r) for r in day]
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
            if r:
                catalogue[sum(self.bits[i] for i in r)] = self.tour_cost(r), tuple(r)
        nodes = [i for r in day for i in r]
        for i in nodes:
            catalogue[self.bits[i]] = self.d[0][i] + self.d[i][0], (i,)
        for mask, (value, tour) in self.pool.items():
            if time.perf_counter() >= until:
                return
            common = mask & target
            if common.bit_count() < 2:
                continue
            if common != mask:
                tour = tuple(i for i in tour if self.bits[i] & target)
                value = self.tour_cost(tour)
            old = catalogue.get(common)
            if old is None or value < old[0]:
                catalogue[common] = value, tour
        full = catalogue.get(target)
        if full is not None:
            commit((full[1],), full[0])
        if self.H == 1:
            self.partitions[target] = (version, upper, tuple(tuple(r) for r in day if r), True)
            return
        # A route individually costing >= the complete incumbent cannot be
        # part of a strict improvement (all distances are nonnegative).
        columns = [(mask, value, tour) for mask, (value, tour) in catalogue.items()
                   if value < upper - self.eps]
        columns.sort(key=lambda c: (c[1] / c[0].bit_count(), -c[0].bit_count()))
        if not columns or time.perf_counter() >= until:
            return
        memberships = [0] * (self.n + 1)
        share = [math.inf] * (self.n + 1)
        for j, (mask, value, tour) in enumerate(columns):
            if time.perf_counter() >= until:
                return
            bit, price = 1 << j, value / len(tour)
            for i in tour:
                memberships[i] |= bit
                if price < share[i]:
                    share[i] = price
        if any(not memberships[i] for i in nodes):
            return
        # Nonnegative route-cost shares form a dual-feasible lower bound.
        # A second dual includes the at-most-H vehicle restriction:
        # sum(price[i] in route) <= route_cost + multiplier.
        multiplier = upper / (self.H + 1) * 0.25
        dual = share[:]
        slack = [max(0.0, c + multiplier - sum(share[i] for i in r))
                 for _, c, r in columns]
        for i in sorted(nodes, key=lambda i: (-self.w[i], -self.d[0][i])):
            if time.perf_counter() >= until:
                return
            bits, increase = memberships[i], math.inf
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
        bounds, dominance, conflicts = {}, {}, {}

        def measure(mask):
            old = bounds.get(mask)
            if old is not None:
                return old
            bits, mass, simple, priced, radial, farthest = mask, 0, 0.0, 0.0, 0.0, 0.0
            while bits:
                bit = bits & -bits
                bits ^= bit
                i = bit.bit_length()
                mass += self.w[i]
                simple += share[i]
                priced += dual[i]
                radial += 2.0 * self.q[i] * self.d[0][i]
                farthest = max(farthest, 2.0 * self.d[0][i])
            result = mass, max(simple, radial, farthest), priced
            bounds[mask] = result
            return result

        def frame(mask, k, available, cost, plan):
            if not k or time.perf_counter() >= until:
                return None
            k = min(k, mask.bit_count())
            mass, simple, priced = measure(mask)
            if mass > k * self.W or cost + max(simple, priced - k * multiplier) >= upper - self.eps:
                return None
            key = mask, k
            if cost >= dominance.get(key, math.inf) - self.eps:
                return None
            dominance[key] = cost
            bits, least, alternatives = mask, len(columns) + 1, 0
            while bits:
                bit = bits & -bits
                bits ^= bit
                choices_here = memberships[bit.bit_length()] & available
                count = choices_here.bit_count()
                if not count:
                    return None
                if count < least:
                    least, alternatives = count, choices_here
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
                commit(tuple(columns[a][2] for a in plan), cost)
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
        self.partitions[target] = (version, upper, tuple(tuple(r) for r in day if r), finished)

    def recombine(self, choices, routes, loads, until):
        days = [t for t in range(self.T) if any(routes[t])]
        self.rng.shuffle(days)
        for p, t in enumerate(days):
            now = time.perf_counter()
            if now >= until:
                return
            day_end = now + (until - now) / math.sqrt(len(days) - p)
            self.exact_blocks(t, choices, routes, loads, now + (day_end - now) * 0.55)
            self.partition_day(t, choices, routes, loads, day_end)

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
        self.exact_global(min(self.end, time.perf_counter() +
                              self.budget / (2.0 + self.intensity)))
        if self.optimal:
            return
        value, saved_choices, saved_routes = self.best
        choices, routes = saved_choices[:], self.clone(saved_routes)
        last_best = time.perf_counter()
        max_remove = min(n, max(1, int(math.sqrt(n * self.intensity))))
        local_slice = self.budget / (60.0 + 8.0 * math.sqrt(n))
        customers = list(range(1, n + 1))
        flexible = [i for i in customers if len(self.options[i]) > 1]
        weights = [1.0] * 6
        pool_interval = self.budget / (2.0 + self.intensity)
        next_pool = time.perf_counter() + pool_interval
        pool_slice = self.budget / (24.0 + 4.0 * math.sqrt(n))
        while time.perf_counter() < self.end:
            now = time.perf_counter()
            if now >= next_pool:
                oldbest = self.best[0]
                if self.exact is not None:
                    self.exact_global(min(self.end, time.perf_counter() + pool_slice))
                    if self.optimal:
                        return
                _, pooled_choices, pooled_routes = self.best
                pooled_choices = pooled_choices[:]
                pooled_routes = self.clone(pooled_routes)
                self.recombine(pooled_choices, pooled_routes, self.route_loads(pooled_routes),
                               min(self.end, time.perf_counter() + pool_slice))
                if self.best[0] < oldbest - self.eps:
                    last_best = time.perf_counter()
                    value, saved_choices, saved_routes = self.best
                    choices, routes = saved_choices[:], self.clone(saved_routes)
                next_pool = time.perf_counter() + pool_interval
                continue
            stagnation = (now - last_best) / max(self.budget, 1e-9)
            if stagnation > 1.0 / (2.0 + self.intensity) and rng.random() < 0.10:
                value, saved_choices, saved_routes = self.best
                choices, routes = saved_choices[:], self.clone(saved_routes)
            ceiling = min(n, max_remove + int(n * min(0.35, stagnation)))
            count = rng.randint(1, max(1, ceiling))
            mode = rng.choices(range(6), weights=weights)[0]
            seed = rng.choice(customers)
            preferred = {}
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
                target = rng.choice([r for r in self.options[seed] if r != choices[seed]])
                preferred[seed] = target
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
            if not self.repair(removed, trial_choices, trial, trial_loads, noise, preferred):
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
    """Return the saved complete solution; a search timeout is never treated as a proof of infeasibility."""
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
