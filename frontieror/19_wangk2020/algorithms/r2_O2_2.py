"""Capacity shadow-price guided flight-path column generation
A global path master model couples the schedule, scenarios and connections, alternately adding low reduced-cost paths and solving the integer model, while keeping verified feasible solutions.
"""
import math
import random
import time
from collections import deque

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp
from scipy.sparse import coo_matrix, eye, hstack


class _Search:
    def __init__(self, data, budget, start):
        self.data = data
        self.start = start
        self.budget = budget
        self.end = start + max(0.0, budget * 0.93)
        self.rng = random.Random(190734863)
        self.n = int(data['n_flights'])
        self.h = int(data['n_periods'])
        self.k = int(data['n_airports'])
        self.original_w = int(data['n_scenarios'])
        self.req = list(map(int, data['requested_departure_period']))
        self.length = list(map(int, data['scheduled_enroute_duration']))
        self.mn = list(map(int, data['min_enroute_duration']))
        self.mx = list(map(int, data['max_enroute_duration']))
        self.dd = list(map(int, data['max_departure_delay']))
        self.ad = list(map(int, data['max_arrival_delay']))
        self.dc = list(map(float, data['departure_delay_cost']))
        self.ac = list(map(float, data['arrival_delay_cost']))
        self.origin = list(map(int, data['flight_departure_airport']))
        self.dest = list(map(int, data['flight_arrival_airport']))
        self.rho = float(data['rho'])
        self.sp = [(int(a), int(b), self.length[int(a)] + int(c))
                   for (a, b), c in zip(data['schedule_connection_pairs'],
                                        data['schedule_min_connection_time'])]
        self.ap = [(int(a), int(b), int(c))
                   for (a, b), c in zip(data['aircraft_connection_pairs'],
                                        data['aircraft_min_connection_time'])]
        delta = int(data['max_schedule_displacement'])
        self.lo = [max(1, 1 - self.length[f], self.req[f] - delta)
                   for f in range(self.n)]
        self.hi = [min(self.h, self.h - self.length[f], self.req[f] + delta)
                   for f in range(self.n)]
        self.sout = [[] for _ in range(self.n)]
        self.sin = [[] for _ in range(self.n)]
        self.aout = [[] for _ in range(self.n)]
        self.ain = [[] for _ in range(self.n)]
        for a, b, lag in self.sp:
            self.sout[a].append((b, lag))
            self.sin[b].append((a, lag))
        for a, b, lag in self.ap:
            self.aout[a].append((b, lag))
            self.ain[b].append((a, lag))
        # Bound propagation for integer difference constraints, without assuming the input indices are already sorted by time.
        queue = deque(range(self.n))
        queued = set(queue)
        while queue and time.monotonic() < self.end:
            f = queue.popleft()
            queued.discard(f)
            if self.lo[f] > self.hi[f]:
                break
            for g, lag in self.sout[f]:
                value = self.lo[f] + lag
                if value > self.lo[g]:
                    self.lo[g] = value
                    if g not in queued:
                        queue.append(g)
                        queued.add(g)
            for g, lag in self.sin[f]:
                value = self.hi[f] - lag
                if value < self.hi[g]:
                    self.hi[g] = value
                    if g not in queued:
                        queue.append(g)
                        queued.add(g)
        self.order = self.topology(self.sout, self.sin)
        self.aorder = self.topology(self.aout, self.ain)
        # Scenarios with identical weather share operating decisions and their probabilities are summed; the original scenario axis is restored on output.
        groups = {}
        self.weather = []
        self.prob = []
        self.mapping = []
        for w in range(self.original_w):
            weather = data['weather_condition'][w]
            key = tuple(tuple(map(int, row)) for row in weather)
            if key not in groups:
                groups[key] = len(self.weather)
                self.weather.append(key)
                self.prob.append(0.0)
            j = groups[key]
            self.mapping.append(j)
            self.prob[j] += float(data['scenario_probabilities'][w])
        self.w = len(self.weather)
        self.weights = [(1.0 - self.rho) * q for q in self.prob]
        self.lines = []
        for w in range(self.w):
            airports = []
            for airport in range(self.k):
                slots = [()]
                for t in range(self.h):
                    cond = self.weather[w][airport][t]
                    slots.append(tuple(zip(
                        map(float, data['capacity_departure_coefficients'][cond][airport]),
                        map(float, data['capacity_arrival_coefficients'][cond][airport]),
                        map(float, data['capacity_limits'][cond][airport]))))
                airports.append(slots)
            self.lines.append(airports)
        self.best = None
        self.best_value = math.inf
        self.dom = []
        self.by_plan = []
        self.full_domains = True
        # Only limits the density of the one-off pricing grid; the main loop still keeps adding random-time paths afterwards.
        grid_budget = max(1.0, 180000.0 * math.sqrt(max(0.001, budget)))
        estimate = sum(max(0, self.hi[f] - self.lo[f] + 1) *
                       (self.dd[f] + 1) * (self.mx[f] - self.mn[f] + 1)
                       for f in range(self.n))
        self.stride = max(1, int(math.ceil(math.sqrt(estimate / grid_budget))))
        self.full_domains = self.stride == 1
        for f in range(self.n):
            pp, dep, arr = [], [], []
            group = {}
            for p in range(self.lo[f], self.hi[f] + 1):
                first = len(pp)
                dl = max(p, 1 - self.mx[f])
                dh = min(self.h, p + self.dd[f], self.h - self.mn[f],
                         p + self.length[f] + self.ad[f] - self.mn[f])
                departures = set(range(dl, dh + 1, self.stride))
                if dh >= dl:
                    departures.add(dh)
                for d in sorted(departures):
                    al = max(1, d + self.mn[f])
                    ah = min(self.h, d + self.mx[f], p + self.length[f] + self.ad[f])
                    arrivals = set(range(al, ah + 1, self.stride))
                    if al <= ah:
                        arrivals.update((ah, min(ah, max(al, p + self.length[f]))))
                    for a in sorted(arrivals):
                        pp.append(p)
                        dep.append(d)
                        arr.append(a)
                group[p] = np.arange(first, len(pp), dtype=np.int32)
            p = np.asarray(pp, dtype=np.int32)
            d = np.asarray(dep, dtype=np.int32)
            a = np.asarray(arr, dtype=np.int32)
            delay_a = (np.full(len(a), self.ad[f], dtype=float)
                       if self.ac[f] < 0 else np.maximum(0, a - p - self.length[f]))
            cost = self.dc[f] * (d - p) + self.ac[f] * delay_a
            self.dom.append([p, d, a, cost])
            self.by_plan.append(group)
        self.allowed = [[None] * self.n for _ in range(self.w)]
        self.used = [[None] * self.n for _ in range(self.w)]
        for w in range(self.w):
            for f in range(self.n):
                p, d, a, _ = self.dom[f]
                ok = np.ones(len(p), dtype=bool)
                for j in range(len(p)):
                    dj, aj = int(d[j]), int(a[j])
                    same = self.origin[f] == self.dest[f] and dj == aj
                    for cd, ca, cap in self.lines[w][self.origin[f]][dj]:
                        if min(cd, ca) >= 0 and cd + (ca if same else 0) > cap + 1e-9:
                            ok[j] = False
                            break
                    if ok[j] and not same:
                        for cd, ca, cap in self.lines[w][self.dest[f]][aj]:
                            if min(cd, ca) >= 0 and ca > cap + 1e-9:
                                ok[j] = False
                                break
                self.allowed[w][f] = ok
                self.used[w][f] = np.zeros(len(p), dtype=bool)
        self.pool = [[[] for _ in range(self.n)] for _ in range(self.w)]
        self.model_ready = False
        self.generation = 0
        self.matrix_generation = -1
        self.last_improvement = start

    def topology(self, out, incoming):
        degree = [len(x) for x in incoming]
        queue = deque(f for f in range(self.n) if degree[f] == 0)
        answer = []
        while queue:
            f = queue.popleft()
            answer.append(f)
            for g, _ in out[f]:
                degree[g] -= 1
                if degree[g] == 0:
                    queue.append(g)
        return answer

    def plan(self, mode):
        answer = self.lo[:]
        for f in self.order:
            low = self.lo[f]
            for g, lag in self.sin[f]:
                low = max(low, answer[g] + lag)
            high = self.hi[f]
            if low > high:
                return None
            if mode == 0:
                target = self.req[f]
            elif mode == 1:
                target = low
            elif mode == 2:
                target = high
            else:
                target = self.rng.randint(low, high)
            answer[f] = min(high, max(low, target))
        return answer

    def fits(self, w, f, d, a, counts):
        o, v = self.origin[f], self.dest[f]
        cd, ca = counts
        nd, na = int(cd[o, d]) + 1, int(ca[o, d])
        if o == v and d == a:
            na += 1
        for x, y, limit in self.lines[w][o][d]:
            if x * nd + y * na > limit + 1e-8:
                return False
        if o != v or d != a:
            nd, na = int(cd[v, a]), int(ca[v, a]) + 1
            for x, y, limit in self.lines[w][v][a]:
                if x * nd + y * na > limit + 1e-8:
                    return False
        return True

    def construct(self, p, stop, noise):
        if p is None or len(self.aorder) != self.n:
            return None
        result = []
        for w in range(self.w):
            counts = (np.zeros((self.k, self.h + 1), dtype=np.int32),
                      np.zeros((self.k, self.h + 1), dtype=np.int32))
            departures, arrivals = [0] * self.n, [0] * self.n
            chosen = [-1] * self.n
            ready = [f for f in range(self.n) if not self.ain[f]]
            indegree = [len(row) for row in self.ain]
            lastd, lasta = [0] * self.n, [0] * self.n
            for f in reversed(self.aorder):
                lasta[f] = min(self.h, p[f] + self.length[f] + self.ad[f])
                for g, lag in self.aout[f]:
                    lasta[f] = min(lasta[f], lastd[g] - lag)
                lastd[f] = min(p[f] + self.dd[f], lasta[f] - self.mn[f])
            rank = [p[f] + 0.35 * (lastd[f] - p[f]) +
                    noise * self.rng.random() * (1 + self.dd[f])
                    for f in range(self.n)]
            while ready and time.monotonic() < stop:
                f = min(ready, key=lambda g: (rank[g], lastd[g], g))
                ready.remove(f)
                earliest = p[f]
                for g, lag in self.ain[f]:
                    earliest = max(earliest, arrivals[g] + lag)
                ids = self.by_plan[f].get(p[f], np.empty(0, dtype=np.int32))
                ps, ds, ars, costs = self.dom[f]
                ids = ids[self.allowed[w][f][ids] & (ds[ids] >= earliest) &
                          (ds[ids] <= lastd[f]) & (ars[ids] <= lasta[f])]
                if not len(ids):
                    return None
                scores = costs[ids] + 1e-7 * ars[ids]
                if noise:
                    scores = scores + np.asarray([self.rng.random() for _ in ids]) * noise
                found = False
                for j in ids[np.argsort(scores)]:
                    if time.monotonic() >= stop:
                        return None
                    d, a = int(ds[j]), int(ars[j])
                    if self.fits(w, f, d, a, counts):
                        chosen[f] = int(j)
                        departures[f], arrivals[f] = d, a
                        counts[0][self.origin[f], d] += 1
                        counts[1][self.dest[f], a] += 1
                        found = True
                        break
                if not found:
                    return None
                for g, _ in self.aout[f]:
                    indegree[g] -= 1
                    if indegree[g] == 0:
                        ready.append(g)
            if any(j < 0 for j in chosen):
                return None
            result.append(chosen)
        return result

    def check(self, p, chosen):
        if len(p) != self.n or len(chosen) != self.w:
            return False
        for f in range(self.n):
            if not self.lo[f] <= p[f] <= self.hi[f]:
                return False
        if any(p[b] - p[a] < lag for a, b, lag in self.sp):
            return False
        for w in range(self.w):
            if len(chosen[w]) != self.n:
                return False
            dc = np.zeros((self.k, self.h + 1), dtype=np.int32)
            ac = np.zeros_like(dc)
            ds, ars = [], []
            for f, j in enumerate(chosen[w]):
                if j < 0 or j >= len(self.dom[f][0]):
                    return False
                ps, d, a, _ = self.dom[f]
                if int(ps[j]) != p[f]:
                    return False
                d, a = int(d[j]), int(a[j])
                if not 1 <= d <= self.h or not 1 <= a <= self.h:
                    return False
                if not 0 <= d - p[f] <= self.dd[f]:
                    return False
                if not self.mn[f] <= a - d <= self.mx[f]:
                    return False
                if max(0, a - p[f] - self.length[f]) > self.ad[f]:
                    return False
                ds.append(d)
                ars.append(a)
                dc[self.origin[f], d] += 1
                ac[self.dest[f], a] += 1
            if any(ds[b] - ars[a] < lag for a, b, lag in self.ap):
                return False
            for airport in range(self.k):
                for t in range(1, self.h + 1):
                    for x, y, cap in self.lines[w][airport][t]:
                        if x * dc[airport, t] + y * ac[airport, t] > cap + 1e-7:
                            return False
        return True

    def accept(self, p, chosen):
        if not self.check(p, chosen):
            return False
        value = self.rho * sum(abs(p[f] - self.req[f]) for f in range(self.n))
        value += sum(self.weights[w] * float(self.dom[f][3][chosen[w][f]])
                     for w in range(self.w) for f in range(self.n))
        if value < self.best_value - 1e-9:
            self.best_value = value
            self.best = (p[:], [row[:] for row in chosen])
            self.last_improvement = time.monotonic()
            return True
        return False

    def initialize_model(self):
        self.eq_p = []
        self.z = []
        self.costs, self.meta = [], []
        self.row, self.col, self.val = [], [], []
        self.beq = [1.0] * self.n
        for w in range(self.w):
            rows = []
            for f in range(self.n):
                rows.append({})
                for p in self.by_plan[f]:
                    rows[-1][p] = len(self.beq)
                    self.beq.append(0.0)
            self.eq_p.append(rows)
        self.neq = len(self.beq)
        self.bub = []
        self.srows = []
        self.arows = []
        for _, _, lag in self.sp:
            self.srows.append(len(self.bub))
            self.bub.append(-float(lag))
        for w in range(self.w):
            rows = []
            for _, _, lag in self.ap:
                rows.append(len(self.bub))
                self.bub.append(-float(lag))
            self.arows.append(rows)
        self.crows = []
        for w in range(self.w):
            airport_rows = []
            for airport in range(self.k):
                slots = [[]]
                for t in range(1, self.h + 1):
                    rows = []
                    for cd, ca, cap in self.lines[w][airport][t]:
                        rows.append((len(self.bub), cd, ca))
                        self.bub.append(cap)
                    slots.append(rows)
                airport_rows.append(slots)
            self.crows.append(airport_rows)
        self.schedule_terms = [[] for _ in range(self.n)]
        self.departure_terms = [[[] for _ in range(self.n)] for _ in range(self.w)]
        self.arrival_terms = [[[] for _ in range(self.n)] for _ in range(self.w)]
        for row, (a, b, lag) in zip(self.srows, self.sp):
            self.schedule_terms[a].append((row, 1.0))
            self.schedule_terms[b].append((row, -1.0))
        for w in range(self.w):
            for row, (a, b, lag) in zip(self.arows[w], self.ap):
                self.arrival_terms[w][a].append((row, 1.0))
                self.departure_terms[w][b].append((row, -1.0))
        for f in range(self.n):
            ids = {}
            for p in self.by_plan[f]:
                terms = [(f, 1.0)]
                terms += [(self.eq_p[w][f][p], -1.0) for w in range(self.w)]
                terms += [(self.neq + r, v * p) for r, v in self.schedule_terms[f]]
                ids[p] = self.column(self.rho * abs(p - self.req[f]),
                                     (-1, f, p), terms)
            self.z.append(ids)
        self.model_ready = True
        if self.best is not None:
            for w in range(self.w):
                for f in range(self.n):
                    self.add(w, f, self.best[1][w][f])
        # For each scheduled time keep the cheapest path, plus scattered operating paths for global re-sequencing.
        for w in range(self.w):
            for f in range(self.n):
                p, d, a, c = self.dom[f]
                for plan, ids in self.by_plan[f].items():
                    ids = ids[self.allowed[w][f][ids]]
                    if not len(ids):
                        continue
                    self.add(w, f, int(ids[np.argmin(c[ids] + 1e-8 * a[ids])]))
                    if len(ids) > 1:
                        self.add(w, f, int(ids[self.rng.randrange(len(ids))]))

    def column(self, cost, meta, terms):
        j = len(self.costs)
        self.costs.append(float(cost))
        self.meta.append(meta)
        for row, value in terms:
            if value:
                self.row.append(row)
                self.col.append(j)
                self.val.append(float(value))
        self.generation += 1
        return j

    def add(self, w, f, j):
        if self.used[w][f][j] or not self.allowed[w][f][j]:
            return False
        p, d, a, c = self.dom[f]
        p, d, a = int(p[j]), int(d[j]), int(a[j])
        terms = [(self.eq_p[w][f][p], 1.0)]
        terms += [(self.neq + r, v * d) for r, v in self.departure_terms[w][f]]
        terms += [(self.neq + r, v * a) for r, v in self.arrival_terms[w][f]]
        terms += [(self.neq + r, cd) for r, cd, ca in self.crows[w][self.origin[f]][d]]
        terms += [(self.neq + r, ca) for r, cd, ca in self.crows[w][self.dest[f]][a]]
        col = self.column(self.weights[w] * float(c[j]), (w, f, int(j)), terms)
        self.pool[w][f].append((int(j), col))
        self.used[w][f][j] = True
        return True

    def matrix(self):
        if self.matrix_generation != self.generation:
            self.mat = coo_matrix((np.asarray(self.val, dtype=float),
                                   (np.asarray(self.row, dtype=np.int32),
                                    np.asarray(self.col, dtype=np.int32))),
                                  shape=(self.neq + len(self.bub), len(self.costs))).tocsc()
            self.mat.sum_duplicates()
            self.cvec = np.asarray(self.costs, dtype=float)
            self.be = np.asarray(self.beq, dtype=float)
            self.bu = np.asarray(self.bub, dtype=float)
            self.matrix_generation = self.generation
        return self.mat

    def relaxation(self, stop):
        mat = self.matrix()
        left = stop - time.monotonic()
        if left <= 0:
            return None
        try:
            answer = linprog(self.cvec, A_ub=mat[self.neq:], b_ub=self.bu,
                             A_eq=mat[:self.neq], b_eq=self.be, bounds=(0, None),
                             method='highs', options={'time_limit': left * 0.88})
            if answer.success:
                return answer.eqlin.marginals, answer.ineqlin.marginals, float(answer.fun)
            if answer.status != 2 or time.monotonic() >= stop:
                return None
            # Artificial slack is added only in the linear pricing subproblem; the integer model never allows constraint violations.
            nr, nc = mat.shape
            upper = hstack((mat, -eye(nr, format='csc'), eye(nr, format='csc')), format='csc')
            penalty = 1.0 + sum(abs(c) for c in self.costs)
            obj = np.concatenate((self.cvec, np.full(2 * nr, penalty)))
            left = stop - time.monotonic()
            if left <= 0:
                return None
            answer = linprog(obj, A_ub=upper[self.neq:], b_ub=self.bu,
                             A_eq=upper[:self.neq], b_eq=self.be, bounds=(0, None),
                             method='highs', options={'time_limit': left * 0.88})
            if answer.success:
                return answer.eqlin.marginals, answer.ineqlin.marginals, -math.inf
        except Exception:
            pass
        return None

    def pricing(self, dual, stop, diversify=False):
        if dual is None:
            eq = np.zeros(self.neq)
            ub = np.zeros(len(self.bub))
        else:
            eq, ub, _ = dual
        dp = np.zeros((self.w, self.k, self.h + 1))
        ap = np.zeros_like(dp)
        for w in range(self.w):
            for airport in range(self.k):
                for t in range(1, self.h + 1):
                    for row, cd, ca in self.crows[w][airport][t]:
                        dp[w, airport, t] += cd * ub[row]
                        ap[w, airport, t] += ca * ub[row]
        groups = [(w, f) for w in range(self.w) for f in range(self.n)]
        self.rng.shuffle(groups)
        batch = max(1, int(math.ceil(math.log1p(self.budget) *
                    (1.0 + (time.monotonic() - self.start) / max(self.budget, 1e-9)))))
        added = 0
        for w, f in groups:
            if time.monotonic() >= stop:
                break
            p, d, a, c = self.dom[f]
            ids = np.flatnonzero(self.allowed[w][f] & ~self.used[w][f])
            if not len(ids):
                continue
            price = np.asarray([eq[self.eq_p[w][f][int(t)]]
                                for t in range(self.lo[f], self.hi[f] + 1)])
            vd = sum(v * ub[r] for r, v in self.departure_terms[w][f])
            va = sum(v * ub[r] for r, v in self.arrival_terms[w][f])
            reduced = (self.weights[w] * c[ids] - price[p[ids] - self.lo[f]] -
                       dp[w, self.origin[f], d[ids]] - ap[w, self.dest[f], a[ids]] -
                       vd * d[ids] - va * a[ids])
            count = min(batch, len(ids))
            positions = np.argpartition(reduced, count - 1)[:count]
            for pos in positions:
                if reduced[pos] < -1e-7 or diversify or dual is None:
                    added += self.add(w, f, int(ids[pos]))
            if diversify and len(ids):
                added += self.add(w, f, int(ids[self.rng.randrange(len(ids))]))
        return added

    def integer(self, stop):
        mat = self.matrix()
        left = stop - time.monotonic()
        if left <= 0:
            return False
        low = np.concatenate((self.be, np.full(len(self.bub), -np.inf)))
        high = np.concatenate((self.be, self.bu))
        try:
            answer = milp(self.cvec, integrality=np.ones(len(self.costs), dtype=np.uint8),
                          bounds=Bounds(0.0, 1.0),
                          constraints=LinearConstraint(mat, low, high),
                          options={'time_limit': left * 0.87, 'mip_rel_gap': 0.0,
                                   'presolve': True, 'disp': False})
            if answer.x is None:
                return False
            p = [0] * self.n
            chosen = [[-1] * self.n for _ in range(self.w)]
            for j in np.flatnonzero(answer.x > 0.5):
                w, f, q = self.meta[int(j)]
                if w < 0:
                    p[f] = q
                else:
                    chosen[w][f] = q
            return self.accept(p, chosen)
        except Exception:
            return False

    def extend_grid(self, stop):
        if self.full_domains:
            return
        # Very wide time windows are not expanded as a Cartesian product at once; support is filled in gradually within the wall-clock loop.
        for f in self.rng.sample(range(self.n), self.n):
            if time.monotonic() >= stop:
                return
            pp, dd, aa, cc = [], [], [], []
            old_p, old_d, old_a, old_c = self.dom[f]
            existing = set(zip(map(int, old_p), map(int, old_d), map(int, old_a)))
            for p in self.by_plan[f]:
                high = min(self.h, p + self.dd[f], self.h - self.mn[f],
                           p + self.length[f] + self.ad[f] - self.mn[f])
                low = max(p, 1 - self.mx[f])
                if high < low:
                    continue
                d = self.rng.randint(low, high)
                al = max(1, d + self.mn[f])
                ah = min(self.h, d + self.mx[f], p + self.length[f] + self.ad[f])
                if ah < al:
                    continue
                a = self.rng.randint(al, ah)
                if (p, d, a) in existing:
                    continue
                pp.append(p)
                dd.append(d)
                aa.append(a)
                delay = self.ad[f] if self.ac[f] < 0 else max(0, a - p - self.length[f])
                cc.append(self.dc[f] * (d - p) + self.ac[f] * delay)
            if not pp:
                continue
            offset = len(old_p)
            self.dom[f] = [np.concatenate((old_p, np.asarray(pp, dtype=np.int32))),
                           np.concatenate((old_d, np.asarray(dd, dtype=np.int32))),
                           np.concatenate((old_a, np.asarray(aa, dtype=np.int32))),
                           np.concatenate((old_c, np.asarray(cc, dtype=float)))]
            for i, p in enumerate(pp):
                self.by_plan[f][p] = np.append(self.by_plan[f][p], offset + i)
            for w in range(self.w):
                self.allowed[w][f] = np.concatenate((self.allowed[w][f], np.ones(len(pp), dtype=bool)))
                self.used[w][f] = np.concatenate((self.used[w][f], np.zeros(len(pp), dtype=bool)))

    def run(self):
        stop = min(self.end, self.start + self.budget * 0.10)
        attempt = 0
        while self.best is None and time.monotonic() < stop:
            mode = attempt if attempt < 3 else 3
            p = self.plan(mode)
            chosen = self.construct(p, stop, 0.0 if attempt == 0 else math.log1p(attempt))
            if chosen is not None:
                self.accept(p, chosen)
            attempt += 1
        self.initialize_model()
        # The global column pool always keeps the initial feasible solution; neighbourhood improvements do not replace global capacity coordination.
        while time.monotonic() < self.end:
            now = time.monotonic()
            remaining = self.end - now
            quantum = min(remaining, self.budget /
                          (2.0 + math.log1p(max(1, self.n * self.w))))
            if quantum <= 0:
                break
            dual = self.relaxation(min(self.end, now + quantum * 0.42))
            added = self.pricing(dual, min(self.end, time.monotonic() + quantum * 0.23),
                                 diversify=dual is None)
            if added == 0:
                self.pricing(dual, min(self.end, time.monotonic() + quantum * 0.13), True)
            if not self.full_domains:
                self.extend_grid(min(self.end, time.monotonic() + quantum * 0.08))
            now = time.monotonic()
            remaining = self.end - now
            if remaining <= 0:
                break
            stagnant = (now - self.last_improvement) / max(self.budget, 1e-9)
            amount = min(remaining, quantum * (1.0 + math.sqrt(max(0.0, stagnant))))
            if self.best is None:
                amount = min(remaining, max(amount, remaining * 0.40))
            self.integer(now + amount)
        return self.pack()

    def pack(self):
        if self.best is not None:
            plan, selected = self.best
            dep = [[int(self.dom[f][1][selected[w][f]]) for f in range(self.n)]
                   for w in range(self.w)]
            arr = [[int(self.dom[f][2][selected[w][f]]) for f in range(self.n)]
                   for w in range(self.w)]
        else:
            # Without an integer incumbent this is only to keep the interface complete; it does not falsely claim to have passed the feasibility check.
            plan = self.plan(0)
            if plan is None:
                plan = self.lo[:]
            dep = [plan[:] for _ in range(self.w)]
            arr = [[min(self.h, max(1, plan[f] + self.mn[f])) for f in range(self.n)]
                   for _ in range(self.w)]
        sa = [int(plan[f] + self.length[f]) for f in range(self.n)]
        od = [dep[w][:] for w in self.mapping]
        oa = [arr[w][:] for w in self.mapping]
        return {'scheduled_departure_period': list(map(int, plan)),
                'scheduled_arrival_period': sa,
                'operated_departure_period': od,
                'operated_arrival_period': oa,
                'departure_delay': [[float(od[w][f] - plan[f]) for f in range(self.n)]
                                    for w in range(self.original_w)],
                'arrival_delay': [[float(self.ad[f] if self.ac[f] < 0 else
                                         max(0, oa[w][f] - sa[f])) for f in range(self.n)]
                                  for w in range(self.original_w)]}


def solve(instance: dict, time_limit_s: float) -> dict:
    started = time.monotonic()
    engine = None
    try:
        budget = max(0.0, float(time_limit_s))
        if not math.isfinite(budget):
            budget = 0.0
        engine = _Search(instance, budget, started)
        if engine.n == 0:
            engine.best = ([], [[] for _ in range(engine.w)])
            return engine.pack()
        return engine.run()
    except Exception:
        if engine is not None:
            try:
                return engine.pack()
            except Exception:
                pass
        n = max(0, int(instance.get('n_flights', 0)))
        w = max(0, int(instance.get('n_scenarios', 0)))
        p = [int(x) for x in instance.get('requested_departure_period', [1] * n)]
        l = [int(x) for x in instance.get('scheduled_enroute_duration', [0] * n)]
        a = [p[f] + l[f] for f in range(n)]
        return {'scheduled_departure_period': p,
                'scheduled_arrival_period': a,
                'operated_departure_period': [p[:] for _ in range(w)],
                'operated_arrival_period': [a[:] for _ in range(w)],
                'departure_delay': [[0.0] * n for _ in range(w)],
                'arrival_delay': [[0.0] * n for _ in range(w)]}
