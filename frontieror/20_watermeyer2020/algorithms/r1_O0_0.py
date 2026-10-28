"""Time-lag closure and period-wise resource hybrid search
Combines temporal constraint propagation, resource-repair neighbourhoods and piecewise-linear integer programming, keeping and improving feasible schedules within the wall-clock budget.
"""

import bisect
import math
import time
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix


_NEG = -(1 << 55)


class _Search:
    def __init__(self, instance, limit):
        self.started = time.perf_counter()
        self.budget = max(0.001, float(limit))
        self.deadline = self.started + self.budget - min(0.18, self.budget * 0.025)
        self.n = int(instance['n']) + 2
        self.end = self.n - 1
        self.k = int(instance['num_resources'])
        self.p = np.asarray(instance['activities_processing_time'], dtype=np.int64)
        self.cap = np.asarray(instance['resources_capacity'], dtype=np.int64)
        self.dem = np.asarray([instance['activities_resource_demands_' + str(k)]
                               for k in range(self.k)], dtype=np.int64).reshape(self.k, self.n).T
        self.periods = [np.asarray(sorted(set(int(t) for t in v if int(t) > 0)),
                                   dtype=np.int64) for v in instance['resources_periods']]
        self.horizon = max((int(v[-1]) for v in self.periods if len(v)), default=0)
        self.edges = [(int(i), int(j), int(d)) for i, j, d in zip(
            instance['temporal_constraints_i'], instance['temporal_constraints_j'],
            instance['temporal_constraints_delta'])]
        self.rng = np.random.default_rng(173)
        self.scale = np.maximum(1.0, self.cap.astype(float))
        self.best = None
        self.best_value = math.inf
        self.profiles = []
        self.table = None
        self.optimal = False
        self.strength = max(1.0, math.log1p(self.budget))
        self.width = max(2, min(self.n, int(math.sqrt(self.n) * self.strength)))
        self.tenure = max(2, int(math.sqrt(self.n) * self.strength))
        self.dist = np.full((self.n, self.n), _NEG, dtype=np.int64)
        np.fill_diagonal(self.dist, 0)
        for i, j, d in self.edges:
            self.dist[i, j] = max(self.dist[i, j], d)
        # Longest-path closure; the non-negative lower bounds of all activities also take part in propagation.
        for v in range(self.n):
            left = self.dist[:, v:v + 1]
            right = self.dist[v:v + 1, :]
            via = left + right
            via[(left == _NEG) | (right == _NEG)] = _NEG
            np.maximum(self.dist, via, out=self.dist)
        self.lo = np.maximum(0, np.max(self.dist, axis=0))
        span = max(0, int(np.max(self.dist)))
        # Start times beyond the last resource period can be compressed to this bound without losing optimal solutions.
        self.H = self.horizon + span
        self.hi = np.minimum(self.H - np.max(self.dist, axis=1), -self.dist[:, 0])
        self.hi[0] = 0
        self.lower_value = int(self.lo[-1])
        self.consider(self.lo)
        self.consider(self.hi)
        late = np.minimum(self.hi, self.horizon)
        late = np.maximum(late, self.lo)
        late = np.max(late[:, None] + self.dist, axis=0)
        self.consider(late)
        self.seed = late

    def usage(self, schedules, details=False):
        s = np.asarray(schedules, dtype=np.int64)
        if s.ndim == 1:
            s = s[None, :]
        if self.table is not None:
            c = self.table[np.arange(self.n)[None, :], np.minimum(s, self.horizon)]
            return c[0] if details else np.sum(c, axis=1)
        if details:
            out = np.zeros((self.n, self.k), dtype=np.int64)
        else:
            out = np.zeros((len(s), self.k), dtype=np.int64)
        for k, periods in enumerate(self.periods):
            if not len(periods):
                continue
            counts = np.searchsorted(periods, s + self.p, side='right')
            counts -= np.searchsorted(periods, s, side='right')
            costs = counts * self.dem[:, k]
            if details:
                out[:, k] = costs[0]
            else:
                out[:, k] = np.sum(costs, axis=1)
        return out

    def valid(self, s):
        if s[0] != 0 or np.any(s < 0):
            return False
        if any(s[j] - s[i] < d for i, j, d in self.edges):
            return False
        return bool(np.all(self.usage(s)[0] <= self.cap))

    def consider(self, s):
        if s[-1] < self.best_value and self.valid(s):
            self.best = np.asarray(s, dtype=np.int64).copy()
            self.best_value = int(s[-1])
            if self.best_value == self.lower_value:
                self.optimal = True
            return True
        return False

    def costs_at(self, i, starts):
        starts = np.asarray(starts, dtype=np.int64)
        ans = np.zeros((len(starts), self.k), dtype=np.int64)
        for k, periods in enumerate(self.periods):
            if self.dem[i, k] and self.p[i] and len(periods):
                ans[:, k] = self.dem[i, k] * (
                    np.searchsorted(periods, starts + self.p[i], side='right') -
                    np.searchsorted(periods, starts, side='right'))
        return ans

    def prepare_profiles(self):
        boundaries = []
        for periods in self.periods:
            if not len(periods):
                boundaries.append(np.empty(0, dtype=np.int64))
                continue
            cuts = np.flatnonzero(np.diff(periods) != 1)
            starts = np.r_[periods[0], periods[cuts + 1]] - 1
            ends = np.r_[periods[cuts], periods[-1]]
            boundaries.append(np.unique(np.r_[starts, ends]))
        for i in range(self.n):
            points = {int(self.lo[i]), int(self.hi[i])}
            if self.p[i]:
                for k in np.flatnonzero(self.dem[i]):
                    for t in boundaries[k]:
                        if self.lo[i] < t < self.hi[i]:
                            points.add(int(t))
                        t = int(t) - int(self.p[i])
                        if self.lo[i] < t < self.hi[i]:
                            points.add(t)
            x = np.asarray(sorted(points), dtype=np.int64)
            c = self.costs_at(i, x)
            if len(x) > 2:
                slopes = np.diff(c, axis=0) // np.diff(x)[:, None]
                keep = np.r_[True, np.any(slopes[1:] != slopes[:-1], axis=1), True]
                x, c = x[keep], c[keep]
            self.profiles.append((x, c))
            if time.perf_counter() >= self.deadline:
                return False
        entries = self.n * (self.horizon + 1) * max(1, self.k)
        if self.k and entries <= min(8_000_000, 350_000 * (1.0 + self.budget)):
            if time.perf_counter() + self.budget * 0.01 < self.deadline:
                self.table = np.zeros((self.n, self.horizon + 1, self.k), dtype=np.int64)
                starts = np.arange(self.horizon + 1, dtype=np.int64)
                for i in range(self.n):
                    self.table[i] = self.costs_at(i, starts)
                    if time.perf_counter() >= self.deadline:
                        self.table = None
                        break
        return True

    def bounds_for(self, value):
        upper = np.minimum(self.hi, int(value) - self.dist[:, self.end])
        return upper

    def clipped_profile(self, i, lower, upper):
        x, _ = self.profiles[i]
        a = np.searchsorted(x, lower, side='right')
        b = np.searchsorted(x, upper, side='left')
        t = np.unique(np.r_[lower, x[a:b], upper]).astype(np.int64)
        return t, self.costs_at(i, t)

    def resource_bound(self, stop):
        if self.best is None:
            return
        left, right = self.lower_value, self.best_value
        while left < right and time.perf_counter() < stop:
            middle = (left + right) // 2
            ub = self.bounds_for(middle)
            lb = np.maximum(self.lo, self.lower_value + self.dist[self.end])
            possible = bool(np.all(ub >= lb))
            total = np.zeros(self.k, dtype=np.int64)
            if possible:
                for i in range(self.n):
                    _, c = self.clipped_profile(i, int(lb[i]), int(ub[i]))
                    total += np.min(c, axis=0)
                    if np.any(total > self.cap):
                        possible = False
                        break
            if possible:
                right = middle
            else:
                left = middle + 1
        self.lower_value = left
        self.lo = np.maximum(self.lo, left + self.dist[self.end])
        if self.best_value == left:
            self.optimal = True

    def merit(self, used, weights):
        excess = np.maximum(used - self.cap, 0) / self.scale
        return (excess + excess * excess * 0.2) @ weights

    def local_search(self, stop):
        if self.optimal or time.perf_counter() >= stop:
            return
        target = self.best_value - 1 if self.best is not None else self.H
        if self.best is not None and self.best_value - self.lower_value > 2:
            target -= int((self.best_value - self.lower_value - 1) * self.rng.uniform(0, 0.5))
        upper = self.bounds_for(target)
        if np.any(upper < self.lo):
            return
        source = self.best if self.best is not None else self.seed
        s = np.minimum(source, upper).copy()
        weights = np.ones(self.k, dtype=float)
        tabu = {}
        iteration = 0
        stagnant = 0
        record = math.inf
        anchor = s.copy()
        while time.perf_counter() < stop and not self.optimal:
            used = self.usage(s)[0]
            feasible = bool(np.all(used <= self.cap))
            if feasible:
                self.consider(s)
                if self.optimal:
                    return
                target = min(target, self.best_value - 1)
                upper = self.bounds_for(target)
                if np.any(upper < self.lo):
                    return
                s = np.minimum(s, upper)
                tabu.clear()
                record = math.inf
                stagnant = 0
                continue
            value = float(self.merit(used[None, :], weights)[0])
            if value < record - 1e-12:
                record, anchor = value, s.copy()
                stagnant = 0
            else:
                stagnant += 1
            iteration += 1
            over = np.maximum(used - self.cap, 0) / self.scale
            prices = (weights * (over > 0) * (1 + 0.4 * over) + 0.0001) / self.scale
            individual = self.usage(s, details=True)
            priority = individual @ prices
            priority += max(1e-8, float(np.max(priority)) * 0.08)
            movable = np.flatnonzero(upper > self.lo)
            if not len(movable):
                return
            probs = priority[movable]
            probs /= np.sum(probs)
            count = min(self.width, len(movable))
            pivots = self.rng.choice(movable, size=count, replace=False, p=probs)
            moves = []
            for i in pivots:
                i = int(i)
                low, high, now = int(self.lo[i]), int(upper[i]), int(s[i])
                points, costs = self.profiles[i]
                a = bisect.bisect_left(points, low)
                b = bisect.bisect_right(points, high)
                choices = {low, high}
                if a < b:
                    vals = costs[a:b] @ prices
                    # Pick low-consumption positions on the left and right separately; the closure handles coupled adjustment of the other activities.
                    divide = bisect.bisect_left(points, now, a, b) - a
                    if divide:
                        q = vals[:divide]
                        merit = q + self.rng.random() * 0.03 * (now - points[a:a + divide]) / max(1, high - low)
                        choices.add(int(points[a + int(np.argmin(merit))]))
                    if divide < b - a:
                        q = vals[divide:]
                        merit = q + self.rng.random() * 0.03 * (points[a + divide:b] - now) / max(1, high - low)
                        choices.add(int(points[a + divide + int(np.argmin(merit))]))
                    pos = bisect.bisect_left(points, now)
                    if pos > a:
                        choices.add(int(points[pos - 1]))
                    if pos < b:
                        choices.add(int(points[pos]))
                    choices.add(int(points[int(self.rng.integers(a, b))]))
                choices.add(max(low, now - 1))
                choices.add(min(high, now + 1))
                for t in choices:
                    if t != now and low <= t <= high:
                        moves.append((i, t))
            if not moves:
                return
            ids = np.asarray([m[0] for m in moves], dtype=np.int64)
            ts = np.asarray([m[1] for m in moves], dtype=np.int64)
            push = ts > s[ids]
            candidates = np.where(push[:, None],
                                  np.maximum(s, ts[:, None] + self.dist[ids]),
                                  np.minimum(s, ts[:, None] - self.dist[:, ids].T))
            totals = self.usage(candidates)
            ok = np.all(totals <= self.cap, axis=1)
            if np.any(ok):
                good = np.flatnonzero(ok)
                chosen = int(good[np.argmin(candidates[good, self.end])])
                s = candidates[chosen].copy()
                self.consider(s)
                continue
            scores = self.merit(totals, weights)
            scores += 1e-8 * np.sum(totals / self.scale, axis=1)
            for q, move in enumerate(moves):
                if tabu.get(move, -1) > iteration:
                    scores[q] += max(1.0, record)
            selected = int(np.argmin(scores))
            old = (int(ids[selected]), int(s[ids[selected]]))
            tabu[old] = iteration + self.tenure
            s = candidates[selected].copy()
            if len(tabu) > self.tenure * self.width:
                tabu = {key: expires for key, expires in tabu.items() if expires > iteration}
            # On stagnation raise the price of overloaded resources and rebuild a time-feasible solution within the current bounds.
            if stagnant >= self.tenure:
                weights += (over > 0) * (0.3 + self.rng.random(self.k))
                weights /= max(1.0, float(np.mean(weights)))
                if self.best is not None and target < self.best_value - 1:
                    target += max(1, (self.best_value - 1 - target) // 2)
                    upper = self.bounds_for(target)
                mode = float(self.rng.random())
                if mode < 0.35:
                    alpha = float(self.rng.random())
                    s = np.ceil((1 - alpha) * self.lo + alpha * upper).astype(np.int64)
                elif mode < 0.7:
                    s = np.minimum(anchor, upper).copy()
                    perturb = max(1, int(math.sqrt(self.n) * self.rng.random() * self.strength))
                    for _ in range(perturb):
                        i = int(self.rng.choice(movable))
                        t = int(self.rng.integers(int(self.lo[i]), int(upper[i]) + 1))
                        if t > s[i]:
                            s = np.maximum(s, t + self.dist[i])
                        else:
                            s = np.minimum(s, t - self.dist[:, i])
                else:
                    s = np.minimum(self.best if self.best is not None else self.seed, upper).copy()
                record, stagnant = math.inf, 0
                tabu.clear()

    def integer_search(self, stop):
        if time.perf_counter() >= stop or self.optimal:
            return
        ub = self.bounds_for(self.best_value - 1) if self.best is not None else self.hi.copy()
        lb = self.lo.copy()
        if np.any(ub < lb):
            if self.best is not None:
                self.optimal = True
            return
        curves = []
        minimum = np.zeros((self.n, self.k), dtype=np.int64)
        for i in range(self.n):
            x, c = self.clipped_profile(i, int(lb[i]), int(ub[i]))
            curves.append((x, c))
            minimum[i] = np.min(c, axis=0)
            if time.perf_counter() >= stop:
                return
        sum_minimum = np.sum(minimum, axis=0)
        if np.any(sum_minimum > self.cap):
            if self.best is not None:
                self.optimal = True
            return
        rows, cols, data = [], [], []
        rlo, rhi = [], []
        lower, upper = list(lb.astype(float)), list(ub.astype(float))
        integrality = [1] * self.n
        objective = [0.0] * self.n
        objective[self.end] = 1.0
        resource_terms = [[] for _ in range(self.k)]
        resource_constant = np.zeros(self.k, dtype=np.int64)

        def row(terms, low, high):
            r = len(rlo)
            for c, v in terms:
                if v:
                    rows.append(r)
                    cols.append(c)
                    data.append(float(v))
            rlo.append(float(low))
            rhi.append(float(high))

        def variable(low, high, integer):
            index = len(lower)
            lower.append(float(low))
            upper.append(float(high))
            integrality.append(integer)
            objective.append(0.0)
            return index

        for i, (x, c) in enumerate(curves):
            if len(x) == 1:
                resource_constant += c[0]
                continue
            allowance = self.cap - sum_minimum + minimum[i]
            segments = []
            for j in range(len(x) - 1):
                a, b = int(x[j]), int(x[j + 1])
                slope = (c[j + 1] - c[j]) // (b - a)
                low, high = a, b
                for k in range(self.k):
                    delta = int(allowance[k] - c[j, k])
                    rate = int(slope[k])
                    if rate > 0:
                        high = min(high, a + delta // rate)
                    elif rate < 0:
                        low = max(low, a - (delta // (-rate)))
                    elif delta < 0:
                        high = low - 1
                        break
                if low <= high:
                    ca = c[j] + slope * (low - a)
                    segments.append((low, high, ca, slope))
            if not segments:
                if self.best is not None:
                    self.optimal = True
                return
            if len(segments) == 1:
                a, b, ca, slope = segments[0]
                lower[i], upper[i] = float(a), float(b)
                resource_constant += ca - slope * a
                for k in np.flatnonzero(slope):
                    resource_terms[k].append((i, int(slope[k])))
                continue
            indicators, start_eq = [], [(i, 1)]
            for a, b, ca, slope in segments:
                z = variable(0, 1, 1)
                indicators.append((z, 1))
                start_eq.append((z, -a))
                for k in np.flatnonzero(ca):
                    resource_terms[k].append((z, int(ca[k])))
                if b > a:
                    v = variable(0, b - a, 0)
                    row([(v, 1), (z, -(b - a))], -np.inf, 0)
                    start_eq.append((v, -1))
                    for k in np.flatnonzero(slope):
                        resource_terms[k].append((v, int(slope[k])))
            row(indicators, 1, 1)
            row(start_eq, 0, 0)
            if time.perf_counter() >= stop:
                return
        for i, j, d in self.edges:
            row([(j, 1), (i, -1)], d, np.inf)
        for k in range(self.k):
            row(resource_terms[k], -np.inf, int(self.cap[k] - resource_constant[k]))
        remaining = stop - time.perf_counter()
        if remaining <= 0.015:
            return
        matrix = coo_matrix((data, (rows, cols)), shape=(len(rlo), len(lower))).tocsc()
        remaining = stop - time.perf_counter()
        if remaining <= 0.015:
            return
        try:
            result = milp(np.asarray(objective), integrality=np.asarray(integrality),
                          bounds=Bounds(np.asarray(lower), np.asarray(upper)),
                          constraints=LinearConstraint(matrix, np.asarray(rlo), np.asarray(rhi)),
                          options={'time_limit': max(0.001, remaining * 0.98),
                                   'mip_rel_gap': 0.0, 'presolve': True})
            if result.x is not None:
                s = np.rint(result.x[:self.n]).astype(np.int64)
                self.consider(s)
            if result.status == 0 and result.x is not None and self.best is not None:
                if int(round(float(result.fun))) == self.best_value:
                    self.optimal = True
            elif result.status == 2 and self.best is not None:
                self.optimal = True
            dual = getattr(result, 'mip_dual_bound', None)
            if dual is not None and math.isfinite(float(dual)):
                self.lower_value = max(self.lower_value, int(math.ceil(float(dual) - 1e-6)))
                if self.best is not None and self.lower_value >= self.best_value:
                    self.optimal = True
        except Exception:
            # When the optional integer solver is unavailable, keep running the independent time-closure neighbourhood search.
            pass

    def run(self):
        if self.optimal:
            return self.best
        if not self.prepare_profiles():
            return self.best if self.best is not None else self.seed
        now = time.perf_counter()
        if self.best is not None:
            self.resource_bound(min(self.deadline, now + self.budget * 0.035))
        exact_used = False
        first = True
        while time.perf_counter() < self.deadline and not self.optimal:
            now = time.perf_counter()
            remaining = self.deadline - now
            if first:
                self.local_search(min(self.deadline, now + remaining * (0.16 if self.best is not None else 0.25)))
                first = False
            elif not exact_used and remaining > max(0.03, self.budget * 0.04):
                self.integer_search(self.deadline - min(self.budget * 0.09, remaining * 0.15))
                exact_used = True
            else:
                quantum = max(0.01, math.sqrt(self.budget) * 0.07)
                self.local_search(min(self.deadline, now + quantum))
        return self.best if self.best is not None else self.seed


def solve(instance: dict, time_limit_s: float) -> dict:
    search = _Search(instance, time_limit_s)
    starts = search.run()
    return {'start_times': {str(i): int(starts[i]) for i in range(search.n)}}
