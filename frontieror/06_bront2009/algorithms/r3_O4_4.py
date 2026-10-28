"""Marginal decomposition with strengthened exact-pricing column generation
Coordinates capacity through a linear program over equivalent component durations, combining enumeration, two-point neighbourhoods, safe fixing, branch-and-bound and mixed-integer pricing, and outputs superimposed offer-set durations verified after scaling.
"""

import heapq
import math
import time

try:
    import numpy as np
except Exception:
    np = None
try:
    from scipy.optimize import linprog, milp, Bounds, LinearConstraint
    from scipy.sparse import csc_matrix, coo_matrix
except Exception:
    linprog = None
    milp = None


def _divide(a, b):
    return np.divide(a, b, out=np.zeros_like(a, dtype=float), where=b > 0)


class _Choice:
    def __init__(self, products, segments, routes, fares, caps, horizon, budget):
        self.products = products
        self.k = len(products)
        self.s = len(segments)
        self.budget = budget
        position = {j: i for i, j in enumerate(products)}
        self.legs = sorted({i for j in products for i in routes[j]})
        leg_position = {i: h for h, i in enumerate(self.legs)}
        rr, cc, vv = [], [], []
        self.rows = []
        for l, (_, _, items) in enumerate(segments):
            js = np.asarray([position[j] for j, v in items], dtype=int)
            vs = np.asarray([v for j, v in items], dtype=float)
            self.rows.append((js, vs))
            rr.extend([l] * len(js))
            cc.extend(js)
            vv.extend(vs)
        self.rr = np.asarray(rr, dtype=int)
        self.cc = np.asarray(cc, dtype=int)
        self.vv = np.asarray(vv, dtype=float)
        self.outside = np.asarray([s[1] for s in segments])
        self.rates = np.asarray([s[0] for s in segments])
        self.edge_rates = self.rates[self.rr]
        self.fares = fares[products]
        self.lp = np.asarray([p for p, j in enumerate(products) for i in routes[j]], dtype=int)
        self.li = np.asarray([leg_position[i] for j in products for i in routes[j]], dtype=int)
        self.scale = horizon / caps[self.legs]
        self.bits = [1 << j for j in range(self.k)]
        self.full = (1 << self.k) - 1
        self.cache = {}
        self.active = []
        self.previous = self.full
        self.table = None
        self.price = None

    def vector(self, mask):
        return np.fromiter((bool(mask & b) for b in self.bits), dtype=bool, count=self.k)

    def mask(self, x):
        return sum(self.bits[int(j)] for j in np.flatnonzero(x))

    def state(self, x, q):
        weights = self.vv * x[self.cc]
        den = self.outside + np.bincount(self.rr, weights=weights, minlength=self.s)
        num = np.bincount(self.rr, weights=weights * q[self.cc], minlength=self.s)
        return den, num, float(np.dot(self.rates, _divide(num, den)))

    def value(self, mask, q):
        saved = self.cache.get(mask)
        if saved is not None:
            return float(np.dot(saved[2], q))
        if self.table is not None:
            return float(np.dot(self.table[mask], q))
        return self.state(self.vector(mask), q)[2]

    def metrics(self, mask):
        saved = self.cache.get(mask)
        if saved is not None:
            return saved
        if self.table is not None:
            sales = self.table[mask].copy()
        else:
            weights = self.vv * self.vector(mask)[self.cc]
            den = self.outside + np.bincount(self.rr, weights=weights, minlength=self.s)
            flows = weights * _divide(self.rates, den)[self.rr]
            sales = np.bincount(self.cc, weights=flows, minlength=self.k)
        loads = np.bincount(self.li, weights=sales[self.lp], minlength=len(self.legs))
        saved = (float(np.dot(sales, self.fares)), loads, sales)
        self.cache[mask] = saved
        return saved

    def gains(self, x, q, free):
        den, num, value = self.state(x, q)
        selected = x[self.cc]
        sign = 1.0 - 2.0 * selected
        dd = den[self.rr] + sign * self.vv
        nn = num[self.rr] + sign * self.vv * q[self.cc]
        counts = np.bincount(self.rr, weights=selected, minlength=self.s)
        last = selected & (counts[self.rr] == 1)
        dd[last] = self.outside[self.rr[last]]
        nn[last] = 0.0
        difference = _divide(nn, dd) - _divide(num, den)[self.rr]
        gains = np.bincount(self.cc, weights=self.edge_rates * difference, minlength=self.k)
        gains[~free] = -np.inf
        return gains, value

    def local(self, seed, state, stop, rng):
        x = self.vector((seed & state.positive) | state.forced)
        free = state.free_vector
        tolerance = state.tol * 0.1
        width = min(int(np.count_nonzero(free)), max(2, int(
            math.sqrt(self.k) * (1.0 + math.log1p(self.budget)))))
        while time.perf_counter() < stop and np.any(free):
            gains, value = self.gains(x, state.q, free)
            j = int(np.argmax(gains))
            if gains[j] > tolerance:
                trial = x.copy()
                trial[j] = ~trial[j]
                if self.state(trial, state.q)[2] > value + tolerance:
                    x = trial
                    continue
            state.offer(self.mask(x))
            ranked = np.flatnonzero(free)
            ranked = ranked[np.argsort(-gains[ranked])]
            shortlist = ranked[:width].copy()
            if len(ranked) > width and width > 1:
                tail = min(width // 2, len(ranked) - width)
                shortlist[-tail:] = rng.choice(ranked[width:], size=tail, replace=False)
            best, pair = tolerance, None
            # The first flip point is screened by marginal loss; the second point checks all free products.
            for a in shortlist:
                if time.perf_counter() >= stop:
                    break
                x[a] = ~x[a]
                second, _ = self.gains(x, state.q, free)
                x[a] = ~x[a]
                second[a] = -np.inf
                b = int(np.argmax(second))
                improvement = float(gains[a] + second[b])
                if improvement > best:
                    best, pair = improvement, (int(a), b)
            if pair is None:
                break
            trial = x.copy()
            trial[pair[0]] = ~trial[pair[0]]
            trial[pair[1]] = ~trial[pair[1]]
            if self.state(trial, state.q)[2] <= value + tolerance:
                break
            x = trial
        state.offer(self.mask(x))


class _Price:
    def __init__(self, comp, q, stop):
        self.c = comp
        self.q = q.copy()
        self.tol = 4e-12 * max(1e-8, float(comp.rates.sum() * np.max(np.abs(q))))
        self.positive = comp.mask(q > 0)
        self.forced = 0
        self.free_vector = q > 0
        self.heap = []
        self.serial = 0
        self.best, self.mask = 0.0, 0
        self.visits = 0
        self.spent = 0.0
        self.certified = math.inf
        self.rows = []
        self.alternatives = []
        self.keys = set()
        self.pool_size = max(2, int(math.sqrt(comp.k) * (1.0 + math.log1p(comp.budget))))
        self.ready = False
        self.row_bound = np.zeros(comp.s)
        single_mask = 0
        for l, (js, vs) in enumerate(comp.rows):
            if time.perf_counter() >= stop:
                return
            order = np.argsort(-q[js], kind='stable')
            js, vs = js[order], vs[order]
            keep = q[js] > 0
            js, vs = js[keep], vs[keep]
            self.rows.append((js, vs))
            if len(js):
                values = _divide(np.cumsum(vs * q[js]), comp.outside[l] + np.cumsum(vs))
                cut = int(np.argmax(values)) + 1
                self.row_bound[l] = max(0.0, float(values[cut - 1]))
                if comp.s == 1:
                    single_mask = sum(comp.bits[int(j)] for j in js[:cut])
        threshold = np.zeros(comp.k)
        np.maximum.at(threshold, comp.cc, self.row_bound[comp.rr])
        self.forced = comp.mask((q > 0) & (q > threshold + self.tol))
        self.free_vector = comp.vector(self.positive & ~self.forced)
        self.offer(comp.previous)
        self.offer(self.positive)
        self.offer(self.forced)
        for mask in comp.active:
            if time.perf_counter() >= stop:
                break
            self.offer(mask)
        self.ready = True
        if comp.s == 1:
            self.offer(single_mask)
            self.certified = self.best
        elif comp.table is not None:
            values = np.einsum('ij,j->i', comp.table, q, optimize=False)
            count = min(len(values), self.pool_size)
            indices = np.argpartition(values, len(values) - count)[-count:]
            for i in indices[np.argsort(-values[indices])]:
                self.offer(int(i))
            self.certified = self.best
        else:
            self.push(float(np.dot(comp.rates, self.row_bound)) + self.tol,
                      self.forced, self.positive & ~self.forced)

    def offer(self, mask):
        mask = (mask & self.positive) | self.forced
        value = self.c.value(mask, self.q)
        if not math.isfinite(value):
            return
        if value > self.best:
            self.best, self.mask = value, mask
            self.c.previous = mask
        if mask not in self.keys:
            if len(self.alternatives) < self.pool_size or value > self.alternatives[-1][0]:
                self.keys.add(mask)
                self.alternatives.append((value, mask))
                self.alternatives.sort(reverse=True)
                if len(self.alternatives) > self.pool_size:
                    _, removed = self.alternatives.pop()
                    self.keys.remove(removed)

    def push(self, upper, forced, free):
        self.serial += 1
        heapq.heappush(self.heap, (-upper, self.serial, forced, free))

    def upper(self):
        if not self.ready:
            return math.inf
        while self.heap and -self.heap[0][0] <= self.best + self.tol:
            heapq.heappop(self.heap)
        tree = max(self.best, -self.heap[0][0]) if self.heap else self.best
        return max(self.best, min(tree, self.certified))

    def expand(self, stop):
        if not self.heap or time.perf_counter() >= stop:
            return
        item = heapq.heappop(self.heap)
        old_upper, _, forced, free = item
        if min(-old_upper, self.certified) <= self.best + self.tol:
            return
        c, q = self.c, self.q
        den, num, _ = c.state(c.vector(forced), q)
        possible = c.vector(free)
        upper = 0.0
        union, intersection = 0, free
        positive, negative = np.zeros(c.k), np.zeros(c.k)
        ceiling = np.full(c.k, -np.inf)
        floor = np.full(c.k, np.inf)
        for l, (js, vs) in enumerate(self.rows):
            if time.perf_counter() >= stop:
                heapq.heappush(self.heap, item)
                return
            keep = possible[js]
            jj, vv = js[keep], vs[keep]
            base = num[l] / den[l] if den[l] > 0 else 0.0
            values = np.empty(len(jj) + 1)
            values[0] = base
            if len(jj):
                values[1:] = _divide(num[l] + np.cumsum(vv * q[jj]), den[l] + np.cumsum(vv))
            cut = int(np.argmax(values))
            bound = max(0.0, float(values[cut]))
            upper += c.rates[l] * bound
            chosen = sum(c.bits[int(j)] for j in jj[:cut])
            row_free = sum(c.bits[int(j)] for j in jj)
            union |= chosen
            intersection &= ~(row_free & ~chosen)
            if len(jj):
                maximum_den = float(den[l] + vv.sum())
                regret = c.rates[l] * vv * (q[jj] - bound) / max(1e-300, maximum_den)
                positive[jj] += np.maximum(regret, 0.0)
                negative[jj] += np.maximum(-regret, 0.0)
                ceiling[jj] = np.maximum(ceiling[jj], bound)
                floor[jj] = np.minimum(floor[jj], num[l] / max(1e-300, maximum_den))
        upper = min(upper + self.tol, -old_upper, self.certified)
        if upper <= self.best + self.tol:
            return
        self.offer(forced | union)
        self.offer(forced | intersection)
        self.offer(forced | c.mask(possible & (positive >= negative)))
        if upper <= self.best + self.tol or not free:
            return
        # Upper and lower bounds on the per-segment ratios ensure these flips do not decrease the pricing objective over all remaining combinations.
        take = c.mask(possible & (q > ceiling + self.tol))
        leave = c.mask(possible & (q < floor - self.tol))
        if take or leave:
            self.push(upper, forced | take, free & ~(take | leave))
            return
        conflict = union & ~intersection & free
        eligible = c.vector(conflict if conflict else free)
        priority = np.minimum(positive, negative)
        priority[~eligible] = -np.inf
        j = int(np.argmax(priority))
        if priority[j] <= self.tol:
            priority = positive + negative
            priority[~eligible] = -np.inf
            j = int(np.argmax(priority))
        bit = c.bits[j]
        rest = free & ~bit
        self.push(upper, forced | bit, rest)
        self.push(upper, forced, rest)

    def mixed_integer(self, stop):
        if milp is None or time.perf_counter() >= stop:
            return
        c, q = self.c, self.q
        chosen = np.flatnonzero(self.free_vector)
        if not len(chosen):
            return
        den, num, _ = c.state(c.vector(self.forced), q)
        if np.any(den <= 0):
            return
        edge = self.free_vector[c.cc]
        rr, cc = c.rr[edge], c.cc[edge]
        edges = len(rr)
        if edges > max(0.0, stop - time.perf_counter()) * 90000:
            return
        mapping = np.full(c.k, -1, dtype=int)
        mapping[chosen] = np.arange(len(chosen))
        prod = mapping[cc]
        a = c.vv[edge] / den[rr]
        if not np.all(np.isfinite(a)) or (edges and float(np.max(a)) > 1e8):
            return
        k, s = len(chosen), c.s
        size = k + s + edges
        total = np.bincount(rr, weights=a, minlength=s)
        low = 1.0 / (1.0 + total)
        off = 1.0 / (1.0 + np.maximum(0.0, total[rr] - a))
        on = 1.0 / (1.0 + a)
        base = num / den
        u = k + np.arange(s)
        z = k + s + np.arange(edges)
        er = 2 * s + 4 * np.arange(edges)
        # Strengthened linear envelope of the normalised reciprocal variables and the binary products.
        rows = [np.arange(s), rr, s + np.arange(s), s + rr,
                er, er, er + 1, er + 1,
                er + 2, er + 2, er + 2, er + 3, er + 3, er + 3]
        cols = [u, z, u, z, z, prod, z, prod,
                k + rr, z, prod, k + rr, z, prod]
        vals = [np.ones(s), a, base, a * q[cc],
                np.ones(edges), -on, -np.ones(edges), low[rr],
                np.ones(edges), -np.ones(edges), np.ones(edges),
                -np.ones(edges), np.ones(edges), -off]
        nrows = 2 * s + 4 * edges
        lower = np.full(nrows + 1, -np.inf)
        upper = np.zeros(nrows + 1)
        lower[:s] = 1.0
        upper[:s] = 1.0
        upper[s:2 * s] = self.row_bound + self.tol
        upper[er + 2] = 1.0
        upper[er + 3] = -off
        scale = max(1e-12, float(c.rates.sum() * np.max(q)))
        objective = np.zeros(size)
        objective[u] = -c.rates * base / scale
        objective[z] = -c.rates[rr] * a * q[cc] / scale
        nz = np.flatnonzero(objective)
        rows.append(np.full(len(nz), nrows, dtype=int))
        cols.append(nz)
        vals.append(objective[nz])
        upper[nrows] = -(self.best - max(self.tol * 16.0, self.best * 1e-9)) / scale
        matrix = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                            shape=(nrows + 1, size)).tocsc()
        lb = np.zeros(size)
        ub = np.ones(size)
        lb[u] = low
        ub[z] = on
        integral = np.zeros(size, dtype=np.uint8)
        integral[:k] = 1
        remaining = stop - time.perf_counter()
        if remaining <= 0:
            return
        try:
            answer = milp(objective, integrality=integral, bounds=Bounds(lb, ub),
                          constraints=LinearConstraint(matrix, lower, upper),
                          options={'time_limit': remaining * 0.88, 'mip_rel_gap': 1e-9,
                                   'presolve': True, 'disp': False})
            if answer.x is not None and np.all(np.isfinite(answer.x[:k])):
                mask = self.forced | sum(c.bits[int(j)] for j in chosen[answer.x[:k] > 0.5])
                self.offer(mask)
            bound = getattr(answer, 'mip_dual_bound', None)
            if bound is not None and math.isfinite(float(bound)):
                self.certified = min(self.certified, -float(bound) * scale + self.tol * 8)
        except Exception:
            pass


class _Solver:
    def __init__(self, data, budget, started, output):
        self.output = output
        self.budget = budget
        self.started = started
        self.finish = started + budget * 0.97
        self.clock = time.perf_counter
        self.best = 0.0
        self.horizon = float(data['booking_horizon_T'])
        self.ids = [int(j) for j in data['products_product_id']]
        self.caps = np.asarray(data['network_legs_capacity'], dtype=float)
        self.n = len(self.ids)
        fares = np.asarray(data['products_fare'], dtype=float)
        fare_scale = max(1.0, float(np.max(np.abs(fares))))
        fares = fares / fare_scale
        position = {j: p for p, j in enumerate(self.ids)}
        leg_index = {int(j): p for p, j in enumerate(data['network_legs_leg_id'])}
        routes = [tuple(sorted({leg_index[int(i)] for i in raw}))
                  for raw in data['products_legs_used']]
        legal = [fares[j] > 0 and all(self.caps[i] > 0 for i in route)
                 for j, route in enumerate(routes)]
        parent = list(range(self.n))

        def root(j):
            while parent[j] != j:
                parent[j] = parent[parent[j]]
                j = parent[j]
            return j

        segments, merged_segments, relevant = [], {}, set()
        # The segment share is lambda_l / lambda, so the actual per-period sales rate uses lambda_l directly.
        for rate, outside, products, weights in zip(data['segments_lambda_l'],
                data['segments_no_purchase_preference'], data['segments_consideration_set'],
                data['segments_preference_vector']):
            if self.clock() >= self.finish:
                self.components = []
                return
            rate = float(rate)
            if rate <= 0:
                continue
            merged = {}
            for product, raw in zip(products, weights):
                j, weight = position[int(product)], float(raw)
                if legal[j] and weight > 0:
                    merged[j] = merged.get(j, 0.0) + weight
            if not merged:
                continue
            normalizer = max(float(outside), max(merged.values()))
            items = tuple((j, v / normalizer) for j, v in sorted(merged.items()))
            outside = float(outside) / normalizer
            key = (outside, items)
            if key in merged_segments:
                segments[merged_segments[key]][0] += rate
                continue
            merged_segments[key] = len(segments)
            segments.append([rate, outside, items])
            r = root(items[0][0])
            for j, _ in items:
                parent[root(j)] = r
                relevant.add(j)
        groups, seg_groups = {}, {}
        for j in sorted(relevant):
            groups.setdefault(root(j), []).append(j)
        for segment in segments:
            seg_groups.setdefault(root(segment[2][0][0]), []).append(segment)
        self.components = []
        for r, products in groups.items():
            if self.clock() >= self.finish:
                self.components = []
                return
            self.components.append(_Choice(products, seg_groups[r], routes, fares,
                                           self.caps, self.horizon, budget))
        self.useful = sorted({i for j in relevant for i in routes[j]})
        master_rows = {i: h for h, i in enumerate(self.useful)}
        for c in self.components:
            c.master_rows = np.asarray([master_rows[i] for i in c.legs], dtype=int)
        self.h = len(self.useful)
        self.d = self.h + len(self.components)
        self.columns = []
        self.known = set()
        self.matrix_rows, self.matrix_cols, self.matrix_values = [], [], []
        self.costs = []
        self.prices = np.zeros(self.h)
        self.tau = np.zeros(len(self.components))
        self.pending = False
        self.optimal = False
        self.master_cost = budget * 0.003
        self.rng = np.random.default_rng(20261003 + self.n * 31)
        work = max(1, sum(len(c.vv) for c in self.components))
        self.quantum = budget / (20.0 + math.sqrt(work))

    def add(self, ci, mask):
        c = self.components[ci]
        mask &= c.full
        if not mask or (ci, mask) in self.known:
            return False
        revenue, loads, _ = c.metrics(mask)
        if not math.isfinite(revenue) or revenue <= 0:
            return False
        index = len(self.columns)
        global_mask = sum(1 << c.products[j] for j in range(c.k) if mask & c.bits[j])
        self.columns.append((ci, mask, revenue, loads, global_mask))
        self.known.add((ci, mask))
        self.costs.append(-revenue)
        self.matrix_rows.append(self.h + ci)
        self.matrix_cols.append(index)
        self.matrix_values.append(1.0)
        scaled = loads * c.scale
        for r, value in zip(c.master_rows, scaled):
            if value > 0:
                self.matrix_rows.append(int(r))
                self.matrix_cols.append(index)
                self.matrix_values.append(float(value))
        self.pending, self.optimal = True, False
        fraction = 1.0 / max(1.0, float(np.max(scaled)) if len(scaled) else 1.0)
        if revenue * fraction > self.best:
            self.accept([(index, fraction)])
        return True

    def accept(self, weights):
        clean = [(int(i), float(v)) for i, v in weights if v > 0 and math.isfinite(float(v))]
        if not clean:
            return
        use = np.zeros(self.d)
        value = 0.0
        for index, weight in clean:
            ci, _, revenue, loads, _ = self.columns[index]
            c = self.components[ci]
            use[c.master_rows] += weight * loads * c.scale
            use[self.h + ci] += weight
            value += weight * revenue
        if not np.all(np.isfinite(use)):
            return
        factor = (1.0 - 1e-10) / max(1.0, float(np.max(use)))
        value *= factor
        if not math.isfinite(value) or value <= self.best:
            return
        # Independent choice components can be sold simultaneously; superimposing them along the time axis yields t[S] of the original model.
        offsets = np.zeros(len(self.components))
        events = []
        for index, weight in clean:
            ci, _, _, _, mask = self.columns[index]
            left = float(offsets[ci])
            right = left + weight * factor
            events.append((left, mask))
            events.append((right, mask))
            offsets[ci] = right
        events.sort()
        active, previous = 0, 0.0
        allocations = {}
        for boundary, mask in events:
            boundary = min(1.0, boundary)
            if boundary > previous and active:
                allocations[active] = allocations.get(active, 0.0) + (boundary - previous) * self.horizon
            active ^= mask
            previous = boundary
        answer = []
        for mask, duration in allocations.items():
            if duration > 0 and math.isfinite(duration):
                offered = []
                remaining = mask
                while remaining:
                    bit = remaining & -remaining
                    offered.append(self.ids[bit.bit_length() - 1])
                    remaining ^= bit
                answer.append({'offer_set': offered, 'time_allocated': float(duration)})
        if answer:
            self.best = value
            self.output['offer_set_allocations'] = answer

    def master(self, stop):
        if linprog is None or not self.columns or self.clock() >= stop:
            return False
        tick = self.clock()
        matrix = csc_matrix((self.matrix_values, (self.matrix_rows, self.matrix_cols)),
                            shape=(self.d, len(self.columns)))
        remaining = stop - self.clock()
        if remaining <= 0:
            return False
        try:
            solved = linprog(np.asarray(self.costs), A_ub=matrix, b_ub=np.ones(self.d),
                             bounds=(0.0, None), method='highs-ds',
                             options={'time_limit': remaining * 0.85,
                                      'primal_feasibility_tolerance': 1e-9,
                                      'dual_feasibility_tolerance': 1e-10,
                                      'presolve': True})
            if solved.x is not None:
                positive = np.flatnonzero(solved.x > 0)
                self.accept((int(j), solved.x[j]) for j in positive)
                for c in self.components:
                    c.active = []
                for j in positive:
                    ci, mask, _, _, _ = self.columns[int(j)]
                    self.components[ci].active.append(mask)
            self.optimal = bool(solved.success)
            if self.optimal:
                y = np.maximum(0.0, -np.asarray(solved.ineqlin.marginals))
                self.prices, self.tau = y[:self.h], y[self.h:]
                self.pending = False
        except Exception:
            self.optimal = False
        self.master_cost = max(self.clock() - tick, self.master_cost * 0.6)
        return self.optimal

    def tables(self, stop):
        allowance = self.budget * 2500000.0 / math.sqrt(max(1, len(self.components)))
        for c in sorted(self.components, key=lambda a: a.k):
            if self.clock() >= stop:
                return
            if c.s == 1 or c.k >= 63 or c.k > math.log2(max(1.0, allowance)):
                continue
            count = 1 << c.k
            work = count * (c.k + len(c.vv))
            if work > allowance or count * c.k * 8 > min(96000000, self.budget * 8000000):
                continue
            table = np.empty((count, c.k))
            block = max(1, min(count, int(200000 / max(1, c.k + len(c.vv)))))
            shifts = np.arange(c.k, dtype=np.uint64)
            offset = 0
            while offset < count and self.clock() < stop:
                last = min(count, offset + block)
                codes = np.arange(offset, last, dtype=np.uint64)
                choices = ((codes[:, None] >> shifts) & 1).astype(float)
                sales = np.zeros_like(choices)
                complete = True
                for l, (js, vs) in enumerate(c.rows):
                    if self.clock() >= stop:
                        complete = False
                        break
                    selected = choices[:, js] * vs
                    den = c.outside[l] + selected.sum(axis=1)
                    rates = _divide(np.full(len(codes), c.rates[l]), den)
                    sales[:, js] += selected * rates[:, None]
                if not complete:
                    return
                table[offset:last] = sales
                offset = last
            if offset == count:
                c.table = table
                allowance -= work

    def insert_pricing(self, ci):
        state = self.components[ci].price
        tolerance = max(state.tol * 8, 2e-10 * (1.0 + abs(float(self.tau[ci]))))
        added = False
        for value, mask in state.alternatives:
            if self.clock() >= self.finish or value <= self.tau[ci] + tolerance:
                break
            added = self.add(ci, mask) or added
        return added, tolerance

    def run(self):
        if not self.components or self.clock() >= self.finish:
            return
        initial = []
        for ci, c in enumerate(self.components):
            if self.clock() >= self.finish:
                break
            if self.add(ci, c.full):
                initial.append((len(self.columns) - 1, 1.0))
        self.accept(initial)
        seed_stop = min(self.finish, self.started + self.budget * 0.055)
        for ci, c in enumerate(self.components):
            mask = 0
            for j in np.argsort(-c.fares, kind='stable'):
                if self.clock() >= seed_stop:
                    break
                mask |= c.bits[int(j)]
                self.add(ci, mask)
                self.add(ci, c.bits[int(j)])
        self.master(min(self.finish, self.clock() + self.budget * 0.08))
        self.tables(min(self.finish, self.started + self.budget * 0.20))
        # The search has no cap on rounds, nodes or restarts; all time slices scale with the total budget.
        while self.clock() < self.finish:
            reserve = max(self.budget * 0.045, min(self.budget * 0.20, self.master_cost * 1.8))
            search_end = self.finish - reserve
            if self.clock() >= search_end:
                break
            if self.pending or not self.optimal:
                stop = min(search_end, self.clock() + max(
                    self.quantum, self.master_cost * 1.8, self.budget * 0.025))
                if not self.master(stop):
                    continue
            added, complete = False, True
            unresolved = []
            for ci, c in enumerate(self.components):
                if self.clock() >= search_end:
                    complete = False
                    break
                leg_price = self.prices[c.master_rows] * c.scale
                q = c.fares - np.bincount(c.lp, weights=leg_price[c.li], minlength=c.k)
                if c.price is None or not np.array_equal(q, c.price.q) or not c.price.ready:
                    c.price = _Price(c, q, search_end)
                state = c.price
                inserted, tolerance = self.insert_pricing(ci)
                added = inserted or added
                # Optimal candidates already in the master problem only absorb numerical residuals in the LP duals.
                target = float(self.tau[ci])
                if (ci, state.mask) in self.known:
                    target = max(target, state.best)
                if state.upper() > target + tolerance:
                    unresolved.append(ci)
            if added:
                continue
            if not complete or self.clock() >= search_end:
                break
            if not unresolved:
                break
            ci = max(unresolved, key=lambda i: (
                self.components[i].price.upper() - self.tau[i]) /
                (math.sqrt(1.0 + len(self.components[i].vv)) *
                 (1.0 + self.components[i].price.spent / max(self.quantum, 1e-12))))
            c = self.components[ci]
            state = c.price
            tick = self.clock()
            slice_time = max(self.quantum, self.budget * 0.006 * math.sqrt(1.0 + state.visits))
            stop = min(search_end, tick + slice_time)
            if state.visits % 4 == 3 and milp is not None:
                stop = min(search_end, tick + max(slice_time, self.budget * 0.045 *
                                                 math.sqrt(1.0 + state.spent / self.budget)))
                state.mixed_integer(stop)
            else:
                if state.visits == 0:
                    seed = state.mask
                elif state.visits % 3 == 0:
                    seed = c.mask(self.rng.random(c.k) < float(self.rng.random()))
                elif state.visits % 3 == 1 and c.active:
                    seed = c.active[int(self.rng.integers(len(c.active)))]
                else:
                    seed = state.mask
                    free = np.flatnonzero(state.free_vector)
                    width = min(len(free), max(1, int(math.sqrt(c.k * math.log1p(self.budget)))))
                    if width:
                        flips = int(self.rng.integers(1, width + 1))
                        for j in self.rng.choice(free, size=flips, replace=False):
                            seed ^= c.bits[int(j)]
                c.local(seed, state, min(stop, tick + slice_time * 0.4), self.rng)
                self.insert_pricing(ci)
                while not self.pending and self.clock() < stop and state.heap:
                    state.expand(stop)
                    inserted, tolerance = self.insert_pricing(ci)
                    if inserted or state.upper() <= max(float(self.tau[ci]), state.best) + tolerance:
                        break
            state.visits += 1
            state.spent += self.clock() - tick
            self.insert_pricing(ci)
        if self.columns and self.clock() < self.finish and (self.pending or not self.optimal):
            self.master(self.finish)


def solve(instance: dict, time_limit_s: float) -> dict:
    started = time.perf_counter()
    result = {'offer_set_allocations': []}
    try:
        budget = float(time_limit_s)
        if not math.isfinite(budget) or budget <= 0 or np is None:
            return result
        if (float(instance['booking_horizon_T']) <= 0 or
                float(instance['lambda']) <= 0 or not instance['products_product_id']):
            return result
        solver = _Solver(instance, budget, started, result)
        solver.run()
    except Exception:
        # Neither an exception nor running out of time overwrites the previously saved complete feasible solution.
        pass
    return result
