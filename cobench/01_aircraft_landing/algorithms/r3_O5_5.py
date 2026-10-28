"""Joint search with structural decomposition and isotonic timing
Detects independent runways and reducible separation chains, accelerates reinsertion search with bounded weighted isotonic regression and component caching, and combines closure timing with joint perturbation, continually keeping the best feasible solution.
"""

import math
import random
import time
from collections import deque
from heapq import merge


_INF = float("inf")


def _closure(weights, links, forbidden, deadline):
    """Find a negative-weight closed set via min cut; on timeout the current times are not moved."""
    n = len(weights)
    source, sink = n, n + 1
    graph = [[] for _ in range(n + 2)]
    dest, capacity = [], []
    large = sum(abs(w) for w in weights) + 1.0

    def edge(u, v, c):
        graph[u].append(len(dest))
        dest.append(v)
        capacity.append(c)
        graph[v].append(len(dest))
        dest.append(u)
        capacity.append(0.0)

    for i, w in enumerate(weights):
        if w < 0.0:
            edge(source, i, -w)
        elif w > 0.0:
            edge(i, sink, w)
        if forbidden[i]:
            edge(i, sink, large)
    for u, v in links:
        edge(u, v, large)
    eps = 1.0e-12
    while time.perf_counter() < deadline:
        level = [-1] * (n + 2)
        level[source] = 0
        q = deque([source])
        while q:
            if time.perf_counter() >= deadline:
                return None
            u = q.popleft()
            for a in graph[u]:
                v = dest[a]
                if capacity[a] > eps and level[v] < 0:
                    level[v] = level[u] + 1
                    q.append(v)
        if level[sink] < 0:
            chosen = [i for i in range(n) if level[i] >= 0]
            if sum(weights[i] for i in chosen) < -eps:
                return chosen
            return None
        pointer = [0] * (n + 2)
        # An explicit path stack keeps recursion depth independent of instance size.
        while time.perf_counter() < deadline:
            nodes, path = [source], []
            while nodes and nodes[-1] != sink:
                u = nodes[-1]
                gu = graph[u]
                while pointer[u] < len(gu):
                    a = gu[pointer[u]]
                    if capacity[a] > eps and level[dest[a]] == level[u] + 1:
                        break
                    pointer[u] += 1
                if pointer[u] == len(gu):
                    level[u] = -1
                    nodes.pop()
                    if path:
                        path.pop()
                        pointer[nodes[-1]] += 1
                else:
                    a = gu[pointer[u]]
                    path.append(a)
                    nodes.append(dest[a])
            if not nodes:
                break
            amount = min(capacity[a] for a in path)
            for a in path:
                capacity[a] -= amount
                capacity[a ^ 1] += amount
    return None


class _Search:
    def __init__(self, data, budget, started):
        self.data = data
        self.clock = time.perf_counter
        self.budget = budget
        self.planes = list(data["planes"])
        self.runways = list(data["runways"])
        self.n, self.r = len(self.planes), len(self.runways)
        self.end = started + budget - min(0.08 * budget, 0.02 + self.n * self.n * 0.0000002)
        self.best = None
        self.best_cost = _INF
        self.backup = None
        self.backup_key = (_INF, _INF)
        self.elites = []
        self.rng = random.Random(0xA17F0 + self.n * 131 + self.r)
        self.strength = math.log1p(max(0.0, budget))
        self.pool_size = max(1, int(1 + self.strength * math.sqrt(math.log1p(self.n))))
        self.eval_seconds = 1.0e-5
        self.cache = {}
        self.components = {}
        self.exact = set()
        self.cache_size = max(16, int((1 + self.strength) * math.sqrt(max(1, self.n)) * 12))
        self.started = started
        self.last_improvement = started
        self.E = [float(data["earliest_landing_time"][p]) for p in self.planes]
        self.T = [float(data["target_landing_time"][p]) for p in self.planes]
        self.L = [float(data["latest_landing_time"][p]) for p in self.planes]
        self.A = [float(data["earliness_penalty_cost"][p]) for p in self.planes]
        self.B = [float(data["lateness_penalty_cost"][p]) for p in self.planes]
        self.M = float(data["big_m"])
        self.penalty = (1 + sum(max(a, b) * (l - e + abs(t - e) + abs(t - l))
                                  for a, b, e, t, l in zip(self.A, self.B, self.E, self.T, self.L))) * 1000
        scale = max([1.0] + [abs(x) for x in self.E + self.L])
        self.eps = max(1.0e-9, math.ulp(scale) * 8)
        self.S = []
        self.safe_m = True
        self.independent = self.r > 1
        raw = data["separation_matrix"]
        for i, p in enumerate(self.planes):
            row = []
            for j, q in enumerate(self.planes):
                flat = [float(raw[p][q][r][k]) for r in self.runways for k in self.runways]
                row.append(flat)
                if i != j and self.independent:
                    if any(flat[a * self.r + b] > 0.0 for a in range(self.r)
                           for b in range(self.r) if a != b):
                        self.independent = False
                if i != j and self.L[i] + max(flat) - self.E[j] > self.M:
                    self.safe_m = False
            self.S.append(row)
        self.independent = self.independent and self.safe_m
        if not self.safe_m:
            # After the assignment is fixed, the relaxed constraints for all unselected runways are also kept, rather than assuming M is large enough.
            for i in range(self.n):
                for j in range(self.n):
                    flat = self.S[i][j]
                    rows = [max(flat[a * self.r:(a + 1) * self.r]) for a in range(self.r)]
                    cols = [max(flat[b::self.r]) for b in range(self.r)]
                    all_max = max(flat) - 2 * self.M
                    self.S[i][j] = [max(flat[a * self.r + b], rows[a] - self.M,
                                             cols[b] - self.M, all_max)
                                     for a in range(self.r) for b in range(self.r)]
        self.s_in = [max([0.0] + [max(self.S[j][i]) for j in range(self.n) if j != i])
                     for i in range(self.n)]
        self.s_out = [max([0.0] + [max(self.S[i][j]) for j in range(self.n) if j != i])
                      for i in range(self.n)]

    def cost(self, seq, times):
        return sum(self.A[i] * max(0.0, self.T[i] - times[i])
                   + self.B[i] * max(0.0, times[i] - self.T[i]) for i in seq)

    def arcs(self, seq, rw, deadline=None):
        deadline = self.end if deadline is None else min(deadline, self.end)
        edges = []
        for a, i in enumerate(seq):
            if self.clock() >= deadline:
                return None
            for j in seq[a + 1:]:
                if self.independent and rw[i] != rw[j]:
                    continue
                d = max(0.0, self.S[i][j][rw[i] * self.r + rw[j]])
                if self.E[j] - self.L[i] < d:
                    edges.append((i, j, d))
                if not self.safe_m:
                    d = self.S[j][i][rw[j] * self.r + rw[i]] - self.M
                    if self.E[i] - self.L[j] < d:
                        edges.append((j, i, d))
        return edges

    def earliest(self, seq, rw, cut=_INF):
        times = self.E[:]
        lower_cost = 0.0
        if not self.safe_m:
            edges = self.arcs(seq, rw)
            if edges is None:
                return None
            graph = [[] for _ in range(self.n)]
            for u, v, d in edges:
                graph[u].append((v, d))
            queue = deque(seq)
            inside = [False] * self.n
            for i in seq:
                inside[i] = True
            depth = [0] * self.n
            while queue:
                if self.clock() >= self.end:
                    return None
                u = queue.popleft()
                inside[u] = False
                for v, d in graph[u]:
                    value = times[u] + d
                    if value > times[v] + self.eps:
                        times[v] = value
                        depth[v] = depth[u] + 1
                        if depth[v] >= len(seq):
                            return None
                        if not inside[v]:
                            queue.append(v)
                            inside[v] = True
        else:
            for pos, i in enumerate(seq):
                if self.clock() >= self.end:
                    return None
                lo = self.E[i]
                if pos:
                    lo = max(lo, times[seq[pos - 1]])
                lim = self.s_in[i]
                ri = rw[i]
                for k in range(pos - 1, -1, -1):
                    j = seq[k]
                    if times[j] + lim <= lo:
                        break
                    v = times[j] + self.S[j][i][rw[j] * self.r + ri]
                    if v > lo:
                        lo = v
                times[i] = lo
                lower_cost += self.B[i] * max(0.0, lo - self.T[i])
                if lower_cost > cut + self.eps:
                    return None
        violation = sum(max(0.0, times[i] - self.L[i]) for i in seq)
        return times, violation

    def coordinate(self, seq, rw, times, backward):
        positions = range(len(seq) - 1, -1, -1) if backward else range(len(seq))
        for pos in positions:
            if self.clock() >= self.end:
                return
            i = seq[pos]
            old = times[i]
            if backward:
                hi = min(self.L[i], self.T[i], times[seq[pos + 1]] if pos + 1 < len(seq) else _INF)
                if hi <= old:
                    continue
                for b in range(pos + 1, len(seq)):
                    j = seq[b]
                    if times[j] - self.s_out[i] >= hi:
                        break
                    hi = min(hi, times[j] - self.S[i][j][rw[i] * self.r + rw[j]])
                if hi > old:
                    times[i] = hi
            else:
                lo = max(self.E[i], self.T[i], times[seq[pos - 1]] if pos else -_INF)
                if lo >= old:
                    continue
                for a in range(pos - 1, -1, -1):
                    j = seq[a]
                    if times[j] + self.s_in[i] <= lo:
                        break
                    lo = max(lo, times[j] + self.S[j][i][rw[j] * self.r + rw[i]])
                if lo < old:
                    times[i] = lo

    def suffix(self, seq, times):
        slope, room = 0.0, _INF
        best_slope, start = -1.0e-12, -1
        for a in range(len(seq) - 1, -1, -1):
            i = seq[a]
            slope += -self.A[i] if times[i] < self.T[i] else self.B[i]
            room = min(room, self.L[i] - times[i])
            if slope < best_slope and room > self.eps:
                best_slope, start = slope, a
        if start < 0:
            return
        block = seq[start:]
        delta = min(self.L[i] - times[i] for i in block)
        slope = best_slope
        for kink, change in sorted((self.T[i] - times[i], self.A[i] + self.B[i])
                                   for i in block if times[i] < self.T[i]):
            if kink >= delta:
                break
            slope += change
            if slope >= -1.0e-12:
                delta = kink
                break
        if delta > self.eps:
            for i in block:
                times[i] += delta

    def key(self, seq, rw):
        if self.independent:
            groups = [[] for _ in range(self.r)]
            for i in seq:
                groups[rw[i]].append(i)
            return tuple(tuple(group) for group in groups)
        return tuple(seq), tuple(rw[i] for i in seq)

    def chain(self, seq, rw):
        """Solve the bounded isotonic subproblem exactly only when adjacent separation chains imply all constraints."""
        size = len(seq)
        if not size:
            return (0.0, 0.0, [], True)
        offsets = [0.0] * size
        for b in range(1, size):
            i, j = seq[b - 1], seq[b]
            offsets[b] = offsets[b - 1] + max(0.0, self.S[i][j][rw[i] * self.r + rw[j]])
        for b, j in enumerate(seq):
            if self.clock() >= self.end:
                return None
            for a in range(b - 2, -1, -1):
                distance = offsets[b] - offsets[a]
                if distance >= self.s_in[j] + self.eps:
                    break
                i = seq[a]
                gap = max(0.0, self.S[i][j][rw[i] * self.r + rw[j]])
                if distance < gap - self.eps and self.E[j] - self.L[i] < gap:
                    return False
        # The forward lower bound doubles as a continuous penalty for infeasible orders; it is never published as a feasible solution.
        floor, low, violation = -_INF, [], 0.0
        for k, i in enumerate(seq):
            floor = max(floor, self.E[i] - offsets[k])
            low.append(floor)
            violation += max(0.0, floor + offsets[k] - self.L[i])
        if violation > self.eps:
            values = [low[k] + offsets[k] for k in range(size)]
            cost = sum(self.A[i] * max(0.0, self.T[i] - v) + self.B[i] * max(0.0, v - self.T[i])
                       for i, v in zip(seq, values))
            return cost, violation, values, False
        high = [0.0] * size
        ceiling = _INF
        for k in range(size - 1, -1, -1):
            ceiling = min(ceiling, self.L[seq[k]] - offsets[k])
            high[k] = ceiling
        # Each block stores objective breakpoints and A+B weights; the derivative starts from -sum(A).
        def value(points, early, lo, hi):
            if early <= 0.0:
                return lo
            weight = 0.0
            x = points[-1][0]
            for target, amount in points:
                weight += amount
                if weight >= early:
                    x = target
                    break
            return max(lo, min(hi, x))

        blocks = []
        for k, i in enumerate(seq):
            if self.clock() >= self.end:
                return None
            points = [(self.T[i] - offsets[k], self.A[i] + self.B[i])]
            block = [k, k, points, self.A[i], low[k], high[k],
                     value(points, self.A[i], low[k], high[k])]
            blocks.append(block)
            while len(blocks) > 1 and blocks[-2][6] > blocks[-1][6] + self.eps:
                if self.clock() >= self.end:
                    return None
                right, left = blocks.pop(), blocks.pop()
                points = list(merge(left[2], right[2]))
                early = left[3] + right[3]
                lo, hi = max(left[4], right[4]), min(left[5], right[5])
                if lo > hi + self.eps:
                    return False
                blocks.append([left[0], right[1], points, early, lo, hi,
                               value(points, early, lo, hi)])
        values = [0.0] * size
        for left, right, _, _, _, _, x in blocks:
            for k in range(left, right + 1):
                values[k] = max(self.E[seq[k]], min(self.L[seq[k]], x + offsets[k]))
        cost = sum(self.A[i] * max(0.0, self.T[i] - v) + self.B[i] * max(0.0, v - self.T[i])
                   for i, v in zip(seq, values))
        return cost, 0.0, values, True

    def component(self, seq, rw, cut=_INF):
        result = self.chain(seq, rw) if self.safe_m else False
        if result is None or result is not False:
            return result
        initial = self.earliest(seq, rw, cut)
        if initial is None:
            return None
        times, violation = initial
        if violation <= self.eps and self.safe_m:
            self.coordinate(seq, rw, times, True)
            self.suffix(seq, times)
            self.coordinate(seq, rw, times, True)
            self.coordinate(seq, rw, times, False)
        return self.cost(seq, times), violation, [times[i] for i in seq], False

    def remember(self, key, result, exact=False):
        if key not in self.cache and len(self.cache) >= self.cache_size:
            old = next(iter(self.cache))
            self.cache.pop(old)
            self.exact.discard(old)
        self.cache[key] = result
        if exact:
            self.exact.add(key)

    def evaluate(self, seq, rw, cut=_INF):
        started = self.clock()
        if started >= self.end:
            return None
        key = self.key(seq, rw)
        hit = self.cache.get(key)
        if hit is not None:
            return hit
        times, cost, violation, exact = self.E[:], 0.0, 0.0, True
        if self.independent:
            groups = key
        else:
            groups = (seq,)
        for r, group in enumerate(groups):
            if self.clock() >= self.end:
                return None
            part_key = (r, tuple(group))
            part = self.components.get(part_key) if self.independent else None
            if part is None:
                part = self.component(group, rw, _INF if self.independent else cut)
                if part is None:
                    return None
                if self.independent:
                    if len(self.components) >= self.cache_size * (1 + self.r):
                        self.components.pop(next(iter(self.components)))
                    self.components[part_key] = part
            cost += part[0]
            violation += part[1]
            exact = exact and part[3]
            for i, v in zip(group, part[2]):
                times[i] = v
        result = (cost + self.penalty * violation, cost, violation, times)
        self.eval_seconds = 0.9 * self.eval_seconds + 0.1 * max(1.0e-7, self.clock() - started)
        self.remember(key, result, exact)
        if len(seq) == self.n and (violation, cost) < self.backup_key:
            self.backup_key = (violation, cost)
            self.backup = (list(seq), list(rw), times[:])
        return result

    def refine(self, seq, rw, times, deadline):
        """Starting from the fast timing result, bidirectional closure moves remove the remaining improvements coupled across blocks."""
        key = self.key(seq, rw)
        known = self.cache.get(key)
        if known is not None and known[1] < self.cost(seq, times):
            times = known[3]
        if key in self.exact:
            return times[:]
        times = times[:]
        edges = self.arcs(seq, rw, deadline)
        if edges is None:
            return times
        index = {i: a for a, i in enumerate(seq)}
        best_times, best_cost = times[:], self.cost(seq, times)
        complete = False
        while self.clock() < deadline:
            tight = [(index[u], index[v]) for u, v, d in edges
                     if times[v] - times[u] - d <= self.eps]
            moved = False
            for direction in (1, -1):
                if self.clock() >= deadline:
                    break
                if direction == 1:
                    weights = [-self.A[i] if times[i] < self.T[i] - self.eps else self.B[i] for i in seq]
                    banned = [times[i] >= self.L[i] - self.eps for i in seq]
                    links = tight
                else:
                    weights = [-self.B[i] if times[i] > self.T[i] + self.eps else self.A[i] for i in seq]
                    banned = [times[i] <= self.E[i] + self.eps for i in seq]
                    links = [(v, u) for u, v in tight]
                chosen = _closure(weights, links, banned, deadline)
                if not chosen:
                    continue
                selected = set(seq[a] for a in chosen)
                delta = _INF
                for i in selected:
                    if direction == 1:
                        delta = min(delta, self.L[i] - times[i])
                        if times[i] < self.T[i] - self.eps:
                            delta = min(delta, self.T[i] - times[i])
                    else:
                        delta = min(delta, times[i] - self.E[i])
                        if times[i] > self.T[i] + self.eps:
                            delta = min(delta, times[i] - self.T[i])
                for u, v, d in edges:
                    crossing = (u in selected and v not in selected) if direction == 1 else (v in selected and u not in selected)
                    if crossing:
                        delta = min(delta, times[v] - times[u] - d)
                if not math.isfinite(delta) or delta <= self.eps:
                    continue
                for i in selected:
                    times[i] += direction * delta
                cost = self.cost(seq, times)
                if cost < best_cost:
                    best_cost, best_times = cost, times[:]
                moved = True
                break
            if not moved:
                complete = self.clock() < deadline
                break
        self.remember(key, (best_cost, best_cost, 0.0, best_times[:]), complete)
        if self.independent:
            for r, group in enumerate(key):
                self.components[r, group] = (self.cost(group, best_times), 0.0,
                                             [best_times[i] for i in group], complete)
        return best_times

    def valid(self, seq, rw, times):
        eps = max(1.0e-7, self.eps * 8)
        if self.independent:
            seq = sorted(seq, key=lambda i: times[i])
        if len(seq) != self.n or len(set(seq)) != self.n:
            return False
        for i in seq:
            if not math.isfinite(times[i]) or times[i] < self.E[i] - eps or times[i] > self.L[i] + eps:
                return False
        for a, i in enumerate(seq):
            for j in seq[a + 1:]:
                if times[j] < times[i] - eps:
                    return False
                if times[j] - times[i] < self.S[i][j][rw[i] * self.r + rw[j]] - eps:
                    return False
                if not self.safe_m and times[i] - times[j] < self.S[j][i][rw[j] * self.r + rw[i]] - self.M - eps:
                    return False
        return True

    def publish(self, seq, rw, times):
        cost = self.cost(seq, times)
        if cost < self.best_cost - 1.0e-9 and self.valid(seq, rw, times):
            self.best_cost = cost
            order = sorted(seq, key=lambda i: times[i]) if self.independent else list(seq)
            self.best = (order, list(rw), list(times))
            self.last_improvement = self.clock()
            return True
        return False

    def consider(self, seq, rw, result=None, polish=False, until=None):
        if result is None:
            result = self.evaluate(seq, rw)
        if result is None or result[2] > self.eps:
            return False
        changed = self.publish(seq, rw, result[3])
        times = result[3]
        if self.clock() < self.end and (polish or changed):
            allowance = self.budget / (4.0 + math.sqrt(self.n))
            end = min(self.end, self.clock() + allowance, until if until is not None else self.end)
            times = self.refine(seq, rw, times, end)
            changed = self.publish(seq, rw, times) or changed
        if polish or changed:
            key = self.key(seq, rw)
            if all(entry[1] != key for entry in self.elites):
                order = sorted(seq, key=lambda i: times[i]) if self.independent else list(seq)
                self.elites.append((self.cost(seq, times), key, order, list(rw), times[:]))
                self.elites.sort(key=lambda x: x[0])
                del self.elites[self.pool_size:]
        return changed

    def construct(self, order, noise=0.0):
        seq, rw, times = [], [0] * self.n, self.E[:]
        for i in order:
            if self.clock() >= self.end:
                return None
            choices = []
            for r in range(self.r):
                lo = self.E[i]
                if seq and not self.independent:
                    lo = max(lo, times[seq[-1]])
                for j in reversed(seq):
                    if self.independent and rw[j] != r:
                        continue
                    if times[j] + self.s_in[i] <= lo and not self.independent:
                        break
                    lo = max(lo, times[j] + max(0.0, self.S[j][i][rw[j] * self.r + r]))
                key = (max(0.0, lo - self.L[i]), lo + noise * self.rng.random())
                choices.append((key, lo, r))
            _, times[i], rw[i] = min(choices)
            seq.append(i)
        return seq, rw

    def seed(self):
        ids = list(range(self.n))
        orders = [sorted(ids, key=lambda i: (self.T[i], self.L[i])),
                  sorted(ids, key=lambda i: (self.L[i], self.T[i])),
                  sorted(ids, key=lambda i: (self.E[i], self.T[i])),
                  sorted(ids, key=lambda i: (self.T[i], -self.B[i] / (self.A[i] + self.B[i] + 1.0e-12)))]
        seed_end = min(self.end, self.clock() + self.budget * 0.12)
        for seq in orders:
            if self.clock() >= seed_end and self.best is not None:
                break
            state = self.construct(seq)
            if state is None:
                break
            self.consider(*state)
            if self.best is None:
                for offset in range(self.r):
                    if self.clock() >= seed_end:
                        break
                    rw = [0] * self.n
                    for pos, i in enumerate(seq):
                        rw[i] = (pos + offset) % self.r
                    self.consider(seq, rw)
        if self.best is not None:
            self.consider(self.best[0], self.best[1], polish=True, until=seed_end)

    def feasibility(self):
        """Use the complete mixed-integer model of the problem when the heuristics have not yet found a feasible solution."""
        if self.best is not None or self.clock() >= self.end:
            return
        try:
            import numpy as np
            from scipy.optimize import Bounds, LinearConstraint, milp
            from scipy.sparse import coo_matrix

            n, r = self.n, self.r
            y0, z0 = 3 * n, 3 * n + n * r
            z = [[-1] * n for _ in range(n)]
            count = z0
            for i in range(n):
                for j in range(n):
                    if i != j:
                        z[i][j] = count
                        count += 1
            objective = np.zeros(count)
            objective[n:2 * n], objective[2 * n:3 * n] = self.A, self.B
            lower, upper = np.zeros(count), np.full(count, np.inf)
            lower[:n], upper[:n] = self.E, self.L
            upper[y0:] = 1.0
            integrality = np.zeros(count, dtype=int)
            integrality[y0:] = 1
            rows, cols, vals, lb, ub = [], [], [], [], []

            def add(terms, lo, hi=_INF):
                row = len(lb)
                for col, value in terms:
                    rows.append(row)
                    cols.append(col)
                    vals.append(value)
                lb.append(lo)
                ub.append(hi)

            raw = self.data["separation_matrix"]
            for i in range(n):
                if self.clock() >= self.end:
                    return
                add([(y0 + i * r + a, 1.0) for a in range(r)], 1.0, 1.0)
                add([(i, 1.0), (n + i, 1.0)], self.T[i])
                add([(i, -1.0), (2 * n + i, 1.0)], -self.T[i])
                for j in range(n):
                    if i == j:
                        continue
                    if i < j:
                        add([(z[i][j], 1.0), (z[j][i], 1.0)], 1.0, 1.0)
                    for a, ra in enumerate(self.runways):
                        for b, rb in enumerate(self.runways):
                            gap = float(raw[self.planes[i]][self.planes[j]][ra][rb])
                            mm = min(self.M, max(0.0, self.L[i] + gap - self.E[j])) if self.safe_m else self.M
                            add([(j, 1.0), (i, -1.0), (y0 + i * r + a, -mm),
                                 (y0 + j * r + b, -mm), (z[i][j], -mm)], gap - 3 * mm)
            remaining = self.end - self.clock()
            if remaining <= 0:
                return
            matrix = coo_matrix((vals, (rows, cols)), shape=(len(lb), count)).tocsc()
            result = milp(objective, integrality=integrality, bounds=Bounds(lower, upper),
                          constraints=LinearConstraint(matrix, np.array(lb), np.array(ub)),
                          options={"time_limit": max(0.001, remaining * 0.65), "mip_rel_gap": 0.0})
            if result.x is not None:
                x = result.x
                rw = [int(np.argmax(x[y0 + i * r:y0 + (i + 1) * r])) for i in range(n)]
                times = [float(x[i]) for i in range(n)]
                seq = sorted(range(n), key=lambda i: (times[i], sum(x[z[j][i]] for j in range(n) if j != i)))
                self.publish(seq, rw, times)
                self.consider(seq, rw, polish=True)
        except Exception:
            return

    def insertion_slots(self, rest, rw, tt, i):
        """Frozen times only decide the trial order; candidates still re-optimise all times."""
        proposals = []
        positions = {j: p for p, j in enumerate(rest)}
        average = sum(self.A[j] + self.B[j] for j in rest) / max(1, len(rest))
        for r in range(self.r):
            group = [j for j in rest if rw[j] == r] if self.independent else rest
            size = len(group)
            low, high = [self.E[i]] * (size + 1), [self.L[i]] * (size + 1)
            for k, j in enumerate(group):
                low[k + 1] = max(low[k], tt[j] + max(0.0, self.S[j][i][rw[j] * self.r + r]))
            for k in range(size - 1, -1, -1):
                j = group[k]
                high[k] = min(high[k + 1], tt[j] - max(0.0, self.S[i][j][r * self.r + rw[j]]))
            for k in range(size + 1):
                when = max(self.E[i], min(self.L[i], max(low[k], min(high[k], self.T[i]))))
                value = (self.A[i] * max(0.0, self.T[i] - when)
                         + self.B[i] * max(0.0, when - self.T[i])
                         + max(0.0, low[k] - high[k]) * (self.A[i] + self.B[i] + average) / 2)
                if not self.independent:
                    position = k
                elif k < size:
                    position = positions[group[k]]
                elif size:
                    position = positions[group[-1]] + 1
                else:
                    position = sum(tt[j] <= self.T[i] for j in rest)
                proposals.append((value, self.rng.random(), position, r))
        proposals.sort()
        return proposals

    def insertion_descent(self, seq, rw, result, deadline):
        while self.clock() < deadline:
            result = self.cache.get(self.key(seq, rw), result)
            penalties = {i: self.A[i] * max(0.0, self.T[i] - result[3][i])
                         + self.B[i] * max(0.0, result[3][i] - self.T[i]) for i in seq}
            scan = sorted(seq, key=lambda i: -penalties[i] * (0.5 + self.rng.random()))
            improved = False
            for i in scan:
                if self.clock() >= deadline:
                    return seq, rw, result
                if self.independent:
                    seq = sorted(seq, key=lambda j: result[3][j])
                old = seq.index(i)
                rest = seq[:old] + seq[old + 1:]
                proposals = self.insertion_slots(rest, rw, result[3], i)
                for _, _, position, r in proposals:
                    if self.clock() >= deadline:
                        return seq, rw, result
                    if position == old and r == rw[i]:
                        continue
                    trial = rest[:position] + [i] + rest[position:]
                    new_rw = rw[:]
                    new_rw[i] = r
                    candidate = self.evaluate(trial, new_rw, result[0])
                    if candidate is not None and candidate[0] < result[0] - self.eps:
                        seq, rw, result = trial, new_rw, candidate
                        self.consider(seq, rw, result, until=deadline)
                        result = self.cache.get(self.key(seq, rw), result)
                        improved = True
                        break
            if not improved:
                break
        return seq, rw, result

    def descent(self, seq, rw, result, deadline):
        """Joint sequence and runway changes; fast evaluation and closure refinement share the same retained solution."""
        seq, rw = seq[:], rw[:]
        if self.independent or self.r == 1:
            sub_end = min(deadline, self.clock() + (deadline - self.clock()) * 0.65)
            seq, rw, result = self.insertion_descent(seq, rw, result, sub_end)
        while self.clock() < deadline:
            result = self.cache.get(self.key(seq, rw), result)
            if self.independent:
                seq.sort(key=lambda i: result[3][i])
            improved = False
            for pos in range(self.n - 1):
                if self.clock() >= deadline:
                    break
                i, j = seq[pos], seq[pos + 1]
                trial = seq[:]
                trial[pos], trial[pos + 1] = j, i
                modes = (False, True) if rw[i] != rw[j] else (False,)
                for swap_lanes in modes:
                    new_rw = rw[:]
                    if swap_lanes:
                        new_rw[i], new_rw[j] = new_rw[j], new_rw[i]
                    candidate = self.evaluate(trial, new_rw, result[0])
                    if candidate is not None and candidate[0] < result[0] - 1.0e-9:
                        seq, rw, result = trial, new_rw, candidate
                        self.consider(seq, rw, result, until=deadline)
                        result = self.cache.get(self.key(seq, rw), result)
                        improved = True
                        break
            ids = list(range(self.n))
            self.rng.shuffle(ids)
            for i in ids:
                if self.clock() >= deadline:
                    break
                for r in range(self.r):
                    if r == rw[i] or self.clock() >= deadline:
                        continue
                    new_rw = rw[:]
                    new_rw[i] = r
                    candidate = self.evaluate(seq, new_rw, result[0])
                    if candidate is not None and candidate[0] < result[0] - 1.0e-9:
                        rw, result = new_rw, candidate
                        self.consider(seq, rw, result, until=deadline)
                        result = self.cache.get(self.key(seq, rw), result)
                        improved = True
            # Screen swaps by weighted cost under frozen times, avoiding random enumeration of distant bad neighbourhoods.
            width = max(1, min(self.n, int(math.sqrt(self.n) * (1 + self.strength))))
            for pos in self.rng.sample(range(self.n), self.n):
                if self.clock() >= deadline:
                    break
                i, times = seq[pos], result[3]
                proposals = []
                for other in self.rng.sample(range(self.n), width):
                    if other == pos:
                        continue
                    j = seq[other]
                    delta = (self.A[i] * max(0.0, self.T[i] - times[j]) + self.B[i] * max(0.0, times[j] - self.T[i])
                             + self.A[j] * max(0.0, self.T[j] - times[i]) + self.B[j] * max(0.0, times[i] - self.T[j])
                             - self.A[i] * max(0.0, self.T[i] - times[i]) - self.B[i] * max(0.0, times[i] - self.T[i])
                             - self.A[j] * max(0.0, self.T[j] - times[j]) - self.B[j] * max(0.0, times[j] - self.T[j]))
                    proposals.append((delta, other))
                if not proposals:
                    continue
                _, other = min(proposals)
                trial = seq[:]
                if self.rng.random() < 0.5:
                    trial[pos], trial[other] = trial[other], trial[pos]
                else:
                    trial.insert(other, trial.pop(pos))
                new_rw = rw[:]
                if self.r > 1 and self.rng.random() < 0.5:
                    new_rw[i] = self.rng.randrange(self.r)
                candidate = self.evaluate(trial, new_rw, result[0])
                if candidate is not None and candidate[0] < result[0] - 1.0e-9:
                    seq, rw, result = trial, new_rw, candidate
                    self.consider(seq, rw, result, until=deadline)
                    result = self.cache.get(self.key(seq, rw), result)
                    improved = True
            if not improved:
                break
        return seq, rw, result

    def reinsert(self, base, deadline, stagnation):
        seq, rw, times = base
        span = max(2, int(math.sqrt(self.n) * self.strength * (0.25 + stagnation)))
        count = min(self.n, span)
        penalties = [self.A[i] * max(0.0, self.T[i] - times[i]) + self.B[i] * max(0.0, times[i] - self.T[i]) for i in range(self.n)]
        mode = self.rng.random()
        if mode < 0.35:
            removed = self.rng.sample(seq, count)
        elif mode < 0.7:
            # Adjacent time blocks and high-cost flights jointly determine where to destroy.
            anchor = max(self.rng.sample(seq, min(self.n, max(1, count))), key=lambda i: penalties[i])
            removed = sorted(seq, key=lambda i: abs(times[i] - times[anchor]))[:count]
        else:
            removed = sorted(seq, key=lambda i: penalties[i] * (0.5 + self.rng.random()), reverse=True)[:count]
        removed_set = set(removed)
        current = [i for i in seq if i not in removed_set]
        rw = rw[:]
        old_position = {i: pos for pos, i in enumerate(seq)}
        removed.sort(key=lambda i: (self.L[i] - self.E[i]) / (1 + self.A[i] + self.B[i]))
        for step, i in enumerate(removed):
            remaining = deadline - self.clock()
            if remaining <= 0:
                return None
            length = len(current)
            affordable = max(1, int(remaining / max(1.0e-7, self.eval_seconds * self.r * (len(removed) - step) * 2)))
            positions = list(range(length + 1))
            if affordable < len(positions):
                target_pos = sum(self.T[j] <= self.T[i] for j in current)
                positions.sort(key=lambda p: min(abs(p - target_pos), abs(p - min(length, old_position[i]))) + self.rng.random() * max(1, length / affordable))
                positions = positions[:affordable]
            shortlist = []
            seen = set()
            keep = max(1, int(1 + self.strength / math.sqrt(max(1, self.n / 4))))
            for pos in positions:
                trial = current[:pos] + [i] + current[pos:]
                for r in range(self.r):
                    if self.clock() >= deadline:
                        return None
                    new_rw = rw[:]
                    new_rw[i] = r
                    key = self.key(trial, new_rw)
                    if key in seen:
                        continue
                    seen.add(key)
                    result = self.evaluate(trial, new_rw)
                    if result is None:
                        continue
                    if len(trial) == self.n and result[2] <= self.eps:
                        self.publish(trial, new_rw, result[3])
                    shortlist.append((result[0], trial, new_rw, result))
                    shortlist.sort(key=lambda x: x[0])
                    del shortlist[keep:]
            if not shortlist:
                return None
            choice = shortlist[0]
            # Closure cost directly drives insertion position/runway choice, not only the final display timing.
            exact_best = _INF
            per_choice = max(0.0, deadline - self.clock()) / max(1, (len(removed) - step) * (len(shortlist) + 1))
            for candidate in shortlist:
                _, trial, new_rw, result = candidate
                value = result[0]
                if result[2] <= self.eps and self.clock() < deadline:
                    tt = self.refine(trial, new_rw, result[3], min(deadline, self.clock() + per_choice))
                    value = self.cost(trial, tt)
                    result = (value, value, 0.0, tt)
                    candidate = (value, trial, new_rw, result)
                if value < exact_best:
                    exact_best, choice = value, candidate
            current, rw = choice[1], choice[2]
        result = self.evaluate(current, rw)
        if result is None:
            return None
        return current, rw, result

    def perturb(self, base, stagnation):
        seq, rw, times = base
        seq, rw = seq[:], rw[:]
        if self.independent and self.rng.random() < 0.25:
            p = self.rng.randrange(self.n)
            lane = rw[seq[p]]
            other = self.rng.choice([r for r in range(self.r) if r != lane])
            for i in seq[p:]:
                if rw[i] == lane:
                    rw[i] = other
                elif rw[i] == other:
                    rw[i] = lane
            result = self.evaluate(seq, rw)
            return (seq, rw, result) if result is not None else None
        count = max(1, min(self.n, int((1 + self.strength) * (1 + stagnation) * math.sqrt(self.n) / 3)))
        radius = max(1, min(self.n, int(math.sqrt(self.n) * (1 + self.strength))))
        for i in self.rng.sample(range(self.n), count):
            j = max(0, min(self.n - 1, i + self.rng.randint(-radius, radius)))
            if self.rng.random() < 0.5:
                seq.insert(j, seq.pop(i))
            else:
                seq[i], seq[j] = seq[j], seq[i]
            if self.r > 1:
                rw[seq[j]] = self.rng.randrange(self.r)
        result = self.evaluate(seq, rw)
        if result is None:
            return None
        return seq, rw, result

    def run(self):
        self.seed()
        self.feasibility()
        current = None
        while self.clock() < self.end:
            now = self.clock()
            if self.best is None:
                order = sorted(range(self.n), key=lambda i: self.L[i] + self.rng.uniform(-1, 1) * (self.L[i] - self.E[i]))
                state = self.construct(order)
                if state is not None:
                    result = self.evaluate(*state)
                    if result is not None:
                        local_end = min(self.end, now + self.budget / (2 + math.sqrt(self.n)))
                        seq, rw, result = self.descent(*state, result, local_end)
                        self.consider(seq, rw, result, polish=True)
                continue
            if current is None:
                seq, rw, _ = self.best
                result = self.evaluate(seq, rw)
                if result is None:
                    break
                current = (seq[:], rw[:], result)
            local_end = min(self.end, now + self.budget / (2 + math.log1p(self.n)))
            current = self.descent(*current, local_end)
            self.consider(current[0], current[1], current[2], polish=True)
            current = (current[0], current[1], self.cache.get(self.key(current[0], current[1]), current[2]))
            if self.clock() >= self.end:
                break
            stagnation = min(1.0, (self.clock() - self.last_improvement) / max(1.0e-9, self.budget * 0.25))
            base = self.best
            if self.elites and self.rng.random() < 0.3:
                entry = self.rng.choice(self.elites)
                base = (entry[2], entry[3], entry[4])
            if self.n > 1 and self.rng.random() < 0.35 + 0.4 * stagnation:
                repair_end = min(self.end, self.clock() + self.budget / (2 + math.sqrt(self.n)))
                candidate = self.reinsert(base, repair_end, stagnation)
            else:
                candidate = self.perturb(base, stagnation)
            if candidate is not None:
                current = candidate
                self.consider(current[0], current[1], current[2])
            else:
                current = None

    def answer(self):
        state = self.best if self.best is not None else self.backup
        if state is None:
            seq = sorted(range(self.n), key=lambda i: self.T[i])
            rw = [i % self.r for i in range(self.n)]
            times = [min(self.L[i], max(self.E[i], self.T[i])) for i in range(self.n)]
        else:
            seq, rw, times = state
        rank = {i: k for k, i in enumerate(seq)}
        order = sorted(seq, key=lambda i: (times[i], rank[i]))
        return {"runway_assignment": {str(p): int(self.runways[rw[i]]) for i, p in enumerate(self.planes)},
                "landing_times": {str(p): float(times[i]) for i, p in enumerate(self.planes)},
                "landing_sequence": [str(self.planes[i]) for i in order]}


def solve(instance: dict, time_limit_s: float) -> dict:
    started = time.perf_counter()
    search = None
    empty = {"runway_assignment": {}, "landing_times": {}, "landing_sequence": []}
    try:
        budget = max(0.0, float(time_limit_s))
        if not math.isfinite(budget):
            budget = 0.0
        if not instance["planes"]:
            return empty
        search = _Search(instance, budget, started)
        search.run()
    except Exception:
        pass
    if search is not None:
        try:
            return search.answer()
        except Exception:
            pass
    try:
        planes, runways = instance["planes"], instance["runways"]
        times = {str(p): float(min(instance["latest_landing_time"][p], max(instance["earliest_landing_time"][p],
                                                                         instance["target_landing_time"][p]))) for p in planes}
        return {"runway_assignment": {str(p): int(runways[i % len(runways)]) for i, p in enumerate(planes)},
                "landing_times": times,
                "landing_sequence": sorted(times, key=times.get)}
    except Exception:
        return empty
