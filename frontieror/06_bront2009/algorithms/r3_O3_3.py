"""Column generation fusing smoothed duals with marginal decomposition
Optimises and superimposes component offer durations with warm-started simplex, combined with batch enumeration pricing, multi-start add-drop-swap under smoothed duals, and branch-and-bound with loss upper bounds.
"""

import heapq
import math
import time

try:
    import numpy as np
except Exception:
    np = None
try:
    from scipy.optimize import linprog
    from scipy.sparse import csc_matrix
except Exception:
    linprog = None
    csc_matrix = None


def _divide(a, b):
    return np.divide(a, b, out=np.zeros_like(a, dtype=float), where=b > 0)


class _Component:
    def __init__(self, products, segments, routes, fares, caps, horizon):
        self.products = products
        self.k = len(products)
        self.s = len(segments)
        position = {j: i for i, j in enumerate(products)}
        self.legs = sorted({i for j in products for i in routes[j]})
        leg_position = {i: h for h, i in enumerate(self.legs)}
        rr, cc, vv = [], [], []
        self.row_items = []
        for l, (_, _, items) in enumerate(segments):
            js = np.asarray([position[j] for j, v in items], dtype=int)
            ws = np.asarray([v for j, v in items], dtype=float)
            self.row_items.append((js, ws))
            rr.extend([l] * len(js))
            cc.extend(js)
            vv.extend(ws)
        self.rr = np.asarray(rr, dtype=int)
        self.cc = np.asarray(cc, dtype=int)
        self.vv = np.asarray(vv, dtype=float)
        self.outside = np.asarray([s[1] for s in segments])
        self.rates = np.asarray([s[0] for s in segments])
        self.edge_rates = self.rates[self.rr]
        self.fares = fares[products]
        a, b = [], []
        for p, j in enumerate(products):
            for i in routes[j]:
                a.append(p)
                b.append(leg_position[i])
        self.lp = np.asarray(a, dtype=int)
        self.li = np.asarray(b, dtype=int)
        self.scale = horizon / caps[self.legs]
        self.bits = [1 << j for j in range(self.k)]
        self.full = (1 << self.k) - 1
        self.cache = {}
        self.column_ids = []
        self.active = []
        self.previous = self.full
        self.table = None
        self.price_state = None
        self.restarts = 0

    def vector(self, mask):
        return np.fromiter((bool(mask & b) for b in self.bits), dtype=bool,
                           count=self.k)

    def mask(self, x):
        return sum(self.bits[int(j)] for j in np.flatnonzero(x))

    def state(self, x, q):
        v = self.vv * x[self.cc]
        den = self.outside + np.bincount(self.rr, weights=v, minlength=self.s)
        num = np.bincount(self.rr, weights=v * q[self.cc], minlength=self.s)
        return den, num, float(np.dot(self.rates, _divide(num, den)))

    def value(self, mask, q):
        saved = self.cache.get(mask)
        if saved is not None:
            return float(np.dot(saved[2], q))
        return self.state(self.vector(mask), q)[2]

    def metrics(self, mask):
        saved = self.cache.get(mask)
        if saved is not None:
            return saved
        x = self.vector(mask)
        weights = self.vv * x[self.cc]
        den = self.outside + np.bincount(self.rr, weights=weights, minlength=self.s)
        flow = weights * _divide(self.rates, den)[self.rr]
        sales = np.bincount(self.cc, weights=flow, minlength=self.k)
        loads = np.bincount(self.li, weights=sales[self.lp], minlength=len(self.legs))
        result = (float(np.dot(sales, self.fares)), loads, sales)
        self.cache[mask] = result
        return result

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
        gain = np.bincount(self.cc, weights=self.edge_rates *
                          (_divide(nn, dd) - _divide(num, den)[self.rr]),
                          minlength=self.k)
        gain[~free] = -np.inf
        return gain, value

    def local(self, seed, state, stop, budget, rng):
        q = state.q
        x = self.vector((seed & state.positive) | state.forced)
        free = state.free_vector
        tolerance = state.tol * 0.1
        width = min(int(np.count_nonzero(free)), max(2, int(
            math.sqrt(self.k) * (1.0 + math.log1p(budget)))))
        while time.perf_counter() < stop and np.any(free):
            gain, value = self.gains(x, q, free)
            j = int(np.argmax(gain))
            if gain[j] > tolerance:
                candidate = x.copy()
                candidate[j] = ~candidate[j]
                if self.state(candidate, q)[2] > value + tolerance:
                    x = candidate
                    continue
            ranked = np.flatnonzero(free)
            ranked = ranked[np.argsort(-gain[ranked])]
            shortlist = ranked[:width].copy()
            if len(ranked) > width and width > 1:
                tail = min(width // 2, len(ranked) - width)
                shortlist[-tail:] = rng.choice(ranked[width:], size=tail, replace=False)
            best, pair = tolerance, None
            for a in shortlist:
                if time.perf_counter() >= stop:
                    break
                x[a] = ~x[a]
                other, _ = self.gains(x, q, free)
                x[a] = ~x[a]
                other[a] = -np.inf
                b = int(np.argmax(other))
                improvement = float(gain[a] + other[b])
                if improvement > best:
                    best, pair = improvement, (int(a), b)
            if pair is None:
                break
            candidate = x.copy()
            candidate[pair[0]] = ~candidate[pair[0]]
            candidate[pair[1]] = ~candidate[pair[1]]
            if self.state(candidate, q)[2] <= value + tolerance:
                break
            x = candidate
        return self.mask(x)


class _Price:
    def __init__(self, comp, q, stop, batch=1, exact=True):
        self.c = comp
        self.q = q.copy()
        self.tol = 3e-12 * max(1.0, float(comp.rates.sum() * np.max(np.abs(q))))
        self.positive = comp.mask(q > 0)
        self.forced = 0
        self.heap = []
        self.serial = 0
        self.best, self.mask = 0.0, 0
        self.visits = 0
        self.rows = []
        self.candidates = []
        row_bound = np.zeros(comp.s)
        # Single-segment threshold upper bounds give safe must-select conditions independent of the other segments.
        for l, (js, vs) in enumerate(comp.row_items):
            order = np.argsort(-q[js])
            js, vs = js[order], vs[order]
            keep = q[js] > 0
            js, vs = js[keep], vs[keep]
            self.rows.append((js, vs))
            if len(js):
                values = _divide(np.cumsum(vs * q[js]), comp.outside[l] + np.cumsum(vs))
                row_bound[l] = max(0.0, float(np.max(values)))
        threshold = np.zeros(comp.k)
        np.maximum.at(threshold, comp.cc, row_bound[comp.rr])
        self.forced = comp.mask((q > 0) & (q >= threshold + self.tol))
        self.free_vector = comp.vector(self.positive & ~self.forced)
        self.offer(comp.previous)
        self.offer(self.positive)
        self.offer(self.forced)
        for mask in comp.active:
            if time.perf_counter() >= stop:
                break
            self.offer(mask)
        if exact and comp.table is not None:
            values = np.einsum('ij,j->i', comp.table, q)
            count = min(len(values), max(1, batch))
            codes = np.argpartition(values, len(values) - count)[-count:]
            codes = codes[np.argsort(-values[codes])]
            for code in codes:
                if time.perf_counter() >= stop and self.candidates:
                    break
                self.candidates.append(int(code))
                self.offer(int(code))
        elif exact:
            # The sum of single-segment threshold bounds is already a valid finite upper bound before variables are fixed.
            upper = float(np.dot(comp.rates, row_bound)) + self.tol
            self.heap.append((-upper, 0, self.forced, self.positive & ~self.forced))

    def offer(self, mask):
        mask = (mask & self.positive) | self.forced
        value = self.c.value(mask, self.q)
        if value > self.best:
            self.best, self.mask = value, mask
            self.c.previous = mask

    def upper(self):
        while self.heap and -self.heap[0][0] <= self.best + self.tol:
            heapq.heappop(self.heap)
        return max(self.best, -self.heap[0][0]) if self.heap else self.best

    def expand(self, stop):
        if not self.heap:
            return
        item = heapq.heappop(self.heap)
        neg_upper, _, forced, free = item
        if -neg_upper <= self.best + self.tol:
            return
        c, q = self.c, self.q
        selected = c.vector(forced)
        possible = c.vector(free)
        den, num, _ = c.state(selected, q)
        upper = 0.0
        union, intersection = 0, free
        positive, negative = np.zeros(c.k), np.zeros(c.k)
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
                values[1:] = _divide(num[l] + np.cumsum(vv * q[jj]),
                                     den[l] + np.cumsum(vv))
            cut = int(np.argmax(values))
            bound = max(0.0, float(values[cut]))
            upper += c.rates[l] * bound
            chosen = sum(c.bits[int(j)] for j in jj[:cut])
            row_free = sum(c.bits[int(j)] for j in jj)
            union |= chosen
            intersection &= ~(row_free & ~chosen)
            if len(jj):
                regret = c.rates[l] * vv * (q[jj] - bound) / max(
                    1e-300, float(den[l] + vv.sum()))
                positive[jj] += np.maximum(regret, 0.0)
                negative[jj] += np.maximum(-regret, 0.0)
        upper += self.tol
        if upper <= self.best + self.tol:
            return
        self.offer(forced | union)
        self.offer(forced | intersection)
        if upper <= self.best + self.tol or not free:
            return
        conflict = union & ~intersection & free
        if not conflict:
            # All segments use the same free-product selection, so the joint candidate attains the separable upper bound.
            if upper <= self.best + 4.0 * self.tol:
                return
            conflict = free
        eligible = c.vector(conflict)
        priority = np.minimum(positive, negative)
        priority[~eligible] = -np.inf
        j = int(np.argmax(priority))
        if priority[j] <= self.tol:
            priority = positive + negative
            priority[~eligible] = -np.inf
            j = int(np.argmax(priority))
        # When the segment's optimal ratio is u, choosing the opposite for j loses at least
        # v_j * abs(q_j-u) / the maximum attainable denominator; summing across segments is still a safe lower bound.
        # Hence branch bounds need not inherit the parent bound, and branches that cannot beat the current best can be fixed at the same time.
        force_more, forbid_more = 0, 0
        for a in np.flatnonzero(possible):
            if upper - positive[a] <= self.best + self.tol:
                force_more |= c.bits[int(a)]
            if upper - negative[a] <= self.best + self.tol:
                forbid_more |= c.bits[int(a)]
        if force_more & forbid_more:
            return
        if force_more or forbid_more:
            self.serial += 1
            heapq.heappush(self.heap, (-upper, self.serial, forced | force_more,
                                      free & ~(force_more | forbid_more)))
            return
        bit = c.bits[j]
        rest = free & ~bit
        self.serial += 2
        take_upper = upper - float(negative[j])
        leave_upper = upper - float(positive[j])
        if take_upper > self.best + self.tol:
            heapq.heappush(self.heap, (-take_upper, self.serial - 1, forced | bit, rest))
        if leave_upper > self.best + self.tol:
            heapq.heappush(self.heap, (-leave_upper, self.serial, forced, rest))


class _Solver:
    def __init__(self, data, budget, started, result):
        self.result = result
        self.budget = budget
        self.started = started
        self.finish = started + budget * 0.978
        self.clock = time.perf_counter
        self.best = 0.0
        self.horizon = float(data['booking_horizon_T'])
        self.ids = [int(p) for p in data['products_product_id']]
        self.caps = np.asarray(data['network_legs_capacity'], dtype=float)
        n, m = len(self.ids), len(self.caps)
        self.n, self.m = n, m
        fares = np.array(data['products_fare'], dtype=float, copy=True)
        self.fare_scale = max(1.0, float(np.max(np.abs(fares)))) if n else 1.0
        fares /= self.fare_scale
        positions = {p: j for j, p in enumerate(self.ids)}
        leg_index = {p: i for i, p in enumerate(data['network_legs_leg_id'])}
        routes = [tuple(sorted({leg_index[i] for i in raw}))
                  for raw in data['products_legs_used']]
        legal = [all(self.caps[i] > 0 for i in route) for route in routes]
        parent = list(range(n))

        def root(j):
            while parent[j] != j:
                parent[j] = parent[parent[j]]
                j = parent[j]
            return j

        segments, relevant = [], set()
        for rate, v0, products, weights in zip(data['segments_lambda_l'],
                data['segments_no_purchase_preference'], data['segments_consideration_set'],
                data['segments_preference_vector']):
            if self.clock() >= self.finish:
                break
            rate = float(rate)
            if rate <= 0:
                continue
            merged = {}
            for p, raw in zip(products, weights):
                j = positions[p]
                v = float(raw)
                if legal[j] and v > 0:
                    merged[j] = merged.get(j, 0.0) + v
            if not merged:
                continue
            norm = max(float(v0), max(merged.values()))
            items = [(j, v / norm) for j, v in merged.items()]
            segments.append((rate, float(v0) / norm, items))
            r = root(items[0][0])
            for j, _ in items:
                parent[root(j)] = r
                relevant.add(j)
        # Do not build solutions from a truncated demand model before data parsing has finished.
        if self.clock() >= self.finish:
            self.components = []
            return
        groups, seg_groups = {}, {}
        for j in sorted(relevant):
            groups.setdefault(root(j), []).append(j)
        for segment in segments:
            seg_groups.setdefault(root(segment[2][0][0]), []).append(segment)
        self.components = [_Component(products, seg_groups[r], routes, fares,
                                      self.caps, self.horizon)
                           for r, products in groups.items()]
        self.useful = sorted({i for j in relevant for i in routes[j]})
        row_id = {i: h for h, i in enumerate(self.useful)}
        for c in self.components:
            c.master_rows = np.asarray([row_id[i] for i in c.legs], dtype=int)
        self.h = len(self.useful)
        self.d = self.h + len(self.components)
        self.columns = []
        self.known = set()
        self.pending = []
        self.matrix_rows, self.matrix_cols, self.matrix_values = [], [], []
        self.costs = []
        self.prices = np.zeros(self.h)
        self.smooth = None
        self.batch = max(1, int(math.sqrt(1.0 + self.d) * math.log1p(budget)))
        self.tau = np.zeros(len(self.components))
        self.optimal = False
        self.master_cost = budget * 0.002
        self.warm = self.d * self.d <= 25000.0 * (1.0 + budget) or linprog is None
        if self.warm:
            self.tableau = np.zeros((self.d + 1, self.d))
            self.tableau[:self.d] = np.eye(self.d)
            self.rhs = np.ones(self.d + 1)
            self.rhs[-1] = 0.0
            self.basis = np.arange(self.d, dtype=int)
        self.rng = np.random.default_rng(71173 + n * 97 + m * 193)
        self.quantum = budget / (24.0 + math.sqrt(max(1, sum(len(c.vv) for c in self.components))))

    def add(self, ci, mask):
        c = self.components[ci]
        mask &= c.full
        if not mask or (ci, mask) in self.known:
            return False
        revenue, loads, _ = c.metrics(mask)
        if not math.isfinite(revenue) or revenue <= 0:
            return False
        index = len(self.columns)
        self.known.add((ci, mask))
        self.columns.append((ci, mask, revenue, loads))
        c.column_ids.append(index)
        self.costs.append(-revenue)
        self.pending.append(index)
        self.matrix_rows.append(self.h + ci)
        self.matrix_cols.append(index)
        self.matrix_values.append(1.0)
        scaled = loads * c.scale
        for r, v in zip(c.master_rows, scaled):
            if v > 0:
                self.matrix_rows.append(int(r))
                self.matrix_cols.append(index)
                self.matrix_values.append(float(v))
        frac = 1.0 / max(1.0, float(np.max(scaled)) if len(scaled) else 1.0)
        if revenue * frac > self.best:
            self.accept([(index, frac)])
        return True

    def accept(self, weights):
        clean = [(int(i), float(v)) for i, v in weights if v > 0 and math.isfinite(float(v))]
        if not clean:
            return
        use = np.zeros(self.d)
        value = 0.0
        for index, x in clean:
            ci, _, revenue, loads = self.columns[index]
            c = self.components[ci]
            use[c.master_rows] += loads * c.scale * x
            use[self.h + ci] += x
            value += revenue * x
        if not np.all(np.isfinite(use)):
            return
        factor = (1.0 - 2e-11) / max(1.0, float(np.max(use)))
        value *= factor
        if not math.isfinite(value) or value <= self.best:
            return
        offsets = np.zeros(len(self.components))
        events = []
        for index, x in clean:
            ci, mask, _, _ = self.columns[index]
            c = self.components[ci]
            global_mask = sum(1 << c.products[j] for j in range(c.k) if mask & c.bits[j])
            left = float(offsets[ci])
            right = left + x * factor
            events.append((left, global_mask))
            events.append((right, global_mask))
            offsets[ci] = right
        events.sort()
        active, previous = 0, 0.0
        allocation = {}
        for boundary, mask in events:
            boundary = min(1.0, boundary)
            if boundary > previous and active:
                allocation[active] = allocation.get(active, 0.0) + (boundary - previous) * self.horizon
            active ^= mask
            previous = boundary
        output = []
        for mask, duration in allocation.items():
            if duration > 0:
                output.append({'offer_set': [self.ids[j] for j in range(self.n)
                                             if mask & (1 << j)],
                               'time_allocated': float(duration)})
        if output:
            self.best = value
            self.result['offer_set_allocations'] = output

    def sparse_master(self, stop):
        if linprog is None or self.clock() >= stop:
            return False
        matrix = csc_matrix((self.matrix_values, (self.matrix_rows, self.matrix_cols)),
                            shape=(self.d, len(self.columns)))
        remaining = stop - self.clock()
        if remaining <= 0:
            return False
        solved = linprog(np.asarray(self.costs), A_ub=matrix, b_ub=np.ones(self.d),
                         bounds=(0.0, None), method='highs-ds',
                         options={'time_limit': remaining * 0.85,
                                  'primal_feasibility_tolerance': 1e-9,
                                  'dual_feasibility_tolerance': 1e-9})
        if solved.x is not None:
            self.accept(enumerate(solved.x))
            for c in self.components:
                c.active = []
            for j in np.flatnonzero(solved.x > 1e-12):
                ci, mask, _, _ = self.columns[int(j)]
                self.components[ci].active.append(mask)
        if solved.success:
            y = np.maximum(0.0, -np.asarray(solved.ineqlin.marginals))
            self.prices, self.tau = y[:self.h], y[self.h:]
            return True
        return False

    def master(self, stop, polish=False):
        tick = self.clock()
        if not self.columns or tick >= stop:
            return
        if not self.warm or polish:
            self.optimal = self.sparse_master(stop)
            if not self.warm and self.optimal:
                self.pending.clear()
        else:
            if self.pending:
                new = list(self.pending)
                a = np.zeros((self.d, len(new)))
                for k, index in enumerate(new):
                    ci, _, revenue, loads = self.columns[index]
                    c = self.components[ci]
                    a[c.master_rows, k] = loads * c.scale
                    a[self.h + ci, k] = 1.0
                transformed = np.einsum('ij,jk->ik', self.tableau[:, :self.d], a)
                transformed[-1] += np.asarray([self.costs[j] for j in new])
                self.tableau = np.concatenate((self.tableau, transformed), axis=1)
                self.pending.clear()
            self.optimal = False
            stalled = tick
            previous = self.rhs[-1]
            while self.clock() < stop:
                reduced = self.tableau[-1]
                tolerance = 2e-12 * max(1.0, abs(float(self.rhs[-1])))
                entering = int(np.argmin(reduced))
                if reduced[entering] >= -tolerance:
                    self.optimal = True
                    break
                if self.clock() - stalled > max(self.quantum, self.master_cost):
                    entering = int(np.flatnonzero(reduced < -tolerance)[0])
                coeff = self.tableau[:-1, entering]
                eligible = np.flatnonzero(coeff > 1e-13)
                if not len(eligible):
                    self.warm = False
                    break
                ratios = np.maximum(self.rhs[eligible], 0.0) / coeff[eligible]
                minimum = float(np.min(ratios))
                ties = eligible[ratios <= minimum + 2e-13 * max(1.0, minimum)]
                leaving = int(ties[np.argmin(self.basis[ties])])
                pivot = float(self.tableau[leaving, entering])
                row = self.tableau[leaving].copy() / pivot
                val = max(0.0, float(self.rhs[leaving])) / pivot
                factors = self.tableau[:, entering].copy()
                factors[leaving] = 0.0
                self.tableau -= factors[:, None] * row
                self.rhs -= factors * val
                self.tableau[leaving] = row
                self.rhs[leaving] = val
                self.tableau[:, entering] = 0.0
                self.tableau[leaving, entering] = 1.0
                self.basis[leaving] = entering
                if self.rhs[-1] > previous + tolerance:
                    previous, stalled = self.rhs[-1], self.clock()
            weights = []
            for c in self.components:
                c.active = []
            for i, variable in enumerate(self.basis):
                if variable >= self.d and self.rhs[i] > 0:
                    index = int(variable) - self.d
                    weights.append((index, float(self.rhs[i])))
                    ci, mask, _, _ = self.columns[index]
                    self.components[ci].active.append(mask)
            self.accept(weights)
            y = np.maximum(0.0, self.tableau[-1, :self.d])
            self.prices, self.tau = y[:self.h].copy(), y[self.h:].copy()
        if self.optimal:
            if self.smooth is None:
                self.smooth = self.prices.copy()
            else:
                self.smooth = 0.65 * self.smooth + 0.35 * self.prices
        self.master_cost = max(self.clock() - tick, self.master_cost * 0.6)

    def tables(self, stop):
        allowance = self.budget * 500000.0 / math.sqrt(max(1, len(self.components)))
        for c in sorted(self.components, key=lambda a: a.k):
            if self.clock() >= stop:
                return
            if c.k >= 63 or c.k > math.log2(max(1.0, allowance)):
                continue
            count = 1 << c.k
            work = count * (c.k + len(c.vv))
            if work > allowance:
                continue
            table = np.empty((count, c.k))
            size = max(1, min(count, int(30000 * math.sqrt(self.budget) / max(1, c.k + len(c.vv)))))
            shifts = np.arange(c.k, dtype=np.uint64)
            offset = 0
            while offset < count and self.clock() < stop:
                last = min(count, offset + size)
                codes = np.arange(offset, last, dtype=np.uint64)
                choices = ((codes[:, None] >> shifts) & 1).astype(float)
                sale = np.zeros_like(choices)
                for l, (js, vs) in enumerate(c.row_items):
                    chosen = choices[:, js] * vs
                    den = c.outside[l] + chosen.sum(axis=1)
                    multipliers = _divide(np.full(len(codes), c.rates[l]), den)
                    sale[:, js] += chosen * multipliers[:, None]
                table[offset:last] = sale
                offset = last
            if offset == count:
                c.table = table
                allowance -= work

    def run(self):
        if not self.components or self.clock() >= self.finish:
            return
        initial = []
        for ci, c in enumerate(self.components):
            self.add(ci, c.full)
            if (ci, c.full) in self.known:
                initial.append((c.column_ids[-1], 1.0))
        self.accept(initial)
        seed_stop = min(self.finish, self.started + self.budget * 0.085)
        for ci, c in enumerate(self.components):
            mask = 0
            for j in np.argsort(-c.fares):
                if self.clock() >= seed_stop:
                    break
                mask |= c.bits[int(j)]
                self.add(ci, mask)
                self.add(ci, c.bits[int(j)])
            for l, (js, vs) in enumerate(c.row_items):
                if self.clock() >= seed_stop:
                    break
                order = np.argsort(-c.fares[js])
                jj, vv = js[order], vs[order]
                values = _divide(np.cumsum(vv * c.fares[jj]),
                                 c.outside[l] + np.cumsum(vv))
                cut = int(np.argmax(values)) + 1
                self.add(ci, sum(c.bits[int(j)] for j in jj[:cut]))
        self.master(min(self.finish, self.clock() + self.budget * 0.09))
        self.tables(min(self.finish, self.clock() + self.budget * 0.09))
        # New columns are warm-started immediately; on stagnation the pricing tree is continued, with no cap on rounds or nodes.
        while self.clock() < self.finish:
            search_end = self.finish - max(self.budget * 0.025, self.master_cost * 1.25)
            if self.clock() >= search_end:
                break
            if self.pending or not self.optimal:
                self.master(min(search_end, self.clock() + max(
                    self.quantum, self.master_cost * 2.0, self.budget * 0.025)))
                if not self.optimal:
                    continue
            added = False
            unresolved = []
            for ci, c in enumerate(self.components):
                if self.clock() >= search_end:
                    break
                leg_price = self.prices[c.master_rows] * c.scale
                q = c.fares - np.bincount(c.lp, weights=leg_price[c.li], minlength=c.k)
                state = c.price_state
                if state is None or not np.array_equal(q, state.q):
                    state = _Price(c, q, search_end, self.batch)
                    c.price_state = state
                tolerance = state.tol * 8.0
                if state.best > self.tau[ci] + tolerance:
                    added = self.add(ci, state.mask) or added
                for mask in state.candidates:
                    if self.clock() >= search_end:
                        break
                    if c.value(mask, q) > self.tau[ci] + tolerance:
                        added = self.add(ci, mask) or added
                state.candidates.clear()
                if state.upper() > self.tau[ci] + tolerance:
                    unresolved.append(ci)
            if added:
                continue
            if self.clock() >= search_end:
                break
            if not unresolved:
                # Every marginal pricing upper bound is at most the corresponding time price, so a global certificate has been obtained.
                if linprog is not None:
                    self.master(min(self.finish, self.clock() + max(
                        self.budget * 0.05, self.master_cost * 2.0)), polish=True)
                break
            # Adjust priorities by the size of the unresolved upper bounds, while visited components gradually yield their time slices.
            ci = max(unresolved, key=lambda i: (
                self.components[i].price_state.upper() - self.tau[i]) /
                (math.sqrt(1.0 + len(self.components[i].vv)) *
                 (1.0 + self.components[i].price_state.visits)))
            c = self.components[ci]
            state = c.price_state
            stop = min(search_end, self.clock() + max(self.quantum, self.budget * 0.006))
            # REF_a's cross-round dual smoothing and multi-start search feed columns directly into the same marginal master problem.
            # Only the true dual state maintains the certificate; perturbed duals are used only to discover offer sets.
            mode = c.restarts % 7
            working = state
            if mode == 0:
                seed = state.mask
            elif mode == 1:
                order = np.flatnonzero(state.q > 0)
                order = order[np.argsort(-state.q[order])]
                cut = int(self.rng.integers(len(order) + 1))
                seed = sum(c.bits[int(j)] for j in order[:cut])
            elif mode == 2 and c.active:
                seed = c.active[int(self.rng.integers(len(c.active)))]
            elif mode == 3:
                seed = c.mask(self.rng.random(c.k) < float(self.rng.random()))
            elif mode == 4 and self.smooth is not None:
                amplitude = min(0.45, 0.06 * (1.0 + math.log1p(self.budget)))
                prices = self.smooth[c.master_rows] * self.rng.uniform(
                    1.0 - amplitude, 1.0 + amplitude, len(c.master_rows))
                prices *= c.scale
                q = c.fares - np.bincount(c.lp, weights=prices[c.li], minlength=c.k)
                working = _Price(c, q, stop, exact=False)
                seed = working.mask
            elif mode == 5 and c.column_ids:
                index = c.column_ids[int(self.rng.integers(len(c.column_ids)))]
                seed = self.columns[index][1]
            else:
                seed = state.forced
            if mode in (3, 5, 6):
                free = np.flatnonzero(working.free_vector)
                strength = min(len(free), max(1, int(math.sqrt(c.k) *
                               (1.0 + math.log1p(self.budget)))))
                if strength:
                    count = int(self.rng.integers(1, strength + 1))
                    for j in self.rng.choice(free, size=count, replace=False):
                        seed ^= c.bits[int(j)]
            local_stop = min(stop, self.clock() + (stop - self.clock()) * 0.5)
            candidate = c.local(seed, working, local_stop, self.budget, self.rng)
            state.offer(candidate)
            # Smoothed pricing columns also enter the master problem; they must not be treated merely as another algorithm's results.
            if working is not state:
                self.add(ci, candidate)
            state.visits += 1
            c.restarts += 1
            if state.best > self.tau[ci] + state.tol * 8 and self.add(ci, state.mask):
                continue
            while self.clock() < stop and state.heap:
                state.expand(stop)
                if state.best > self.tau[ci] + state.tol * 8:
                    if self.add(ci, state.mask):
                        break
                if state.upper() <= self.tau[ci] + state.tol * 8:
                    break
        if self.pending or not self.optimal:
            self.master(self.finish)
        if linprog is not None and self.clock() < self.finish and self.columns:
            self.master(self.finish, polish=True)


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
        # The empty solution and the solution saved after every replacement are feasible; an exception does not overwrite results already obtained.
        pass
    return result
