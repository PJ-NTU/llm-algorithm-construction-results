"""Joint search with closure feedback and regret reinsertion
Writes bidirectional min-cut timing back into candidate evaluation, combined with joint sequence-runway shifts, regret-based reinsertion and independent timing under zero cross-runway separation, continually saving the best feasible solution.
"""

import math
import random
import time
from collections import deque


_INF = float("inf")


def _closure(weights, links, forbidden, deadline):
    """Find a negative-weight closed set via min cut; on timeout the current times are not moved."""
    if not any(w < -1.0e-12 and not forbidden[i] for i, w in enumerate(weights)):
        return None
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
        self.end = started + budget - min(0.08 * budget, 0.025 + self.n * self.n * 0.0000005)
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
        self.optimal_timing = set()
        self.refine_seconds = 1.0e-4
        self.mip_failed = False
        self.cache_size = max(16, int((1 + self.strength) * math.sqrt(max(1, self.n)) * 12))
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
        raw = data["separation_matrix"]
        for i, p in enumerate(self.planes):
            row = []
            for j, q in enumerate(self.planes):
                flat = [float(raw[p][q][r][k]) for r in self.runways for k in self.runways]
                row.append(flat)
                if i != j and self.L[i] + max(flat) - self.E[j] > self.M:
                    self.safe_m = False
            self.S.append(row)
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
        self.independent = self.safe_m and all(
            self.S[i][j][a * self.r + b] <= 0.0
            for i in range(self.n) for j in range(self.n) if i != j
            for a in range(self.r) for b in range(self.r) if a != b)
        self.s_in = [max([0.0] + [max(self.S[j][i]) for j in range(self.n) if j != i])
                     for i in range(self.n)]
        self.s_out = [max([0.0] + [max(self.S[i][j]) for j in range(self.n) if j != i])
                      for i in range(self.n)]

    def groups(self, seq, rw):
        if not self.independent:
            return [seq]
        lanes = [[] for _ in range(self.r)]
        for i in seq:
            lanes[rw[i]].append(i)
        return lanes

    def key(self, seq, rw):
        if self.independent:
            return tuple(tuple(g) for g in self.groups(seq, rw))
        return tuple(seq), tuple(rw[i] for i in seq)

    def remember(self, seq, rw, result, optimal=False):
        key = self.key(seq, rw)
        old = self.cache.get(key)
        if old is not None and old[0] < result[0] - self.eps:
            return old
        if key not in self.cache and len(self.cache) >= self.cache_size:
            expired = next(iter(self.cache))
            del self.cache[expired]
            self.optimal_timing.discard(expired)
        self.cache[key] = result
        if optimal:
            self.optimal_timing.add(key)
        return result

    def cost(self, seq, times):
        return sum(self.A[i] * max(0.0, self.T[i] - times[i])
                   + self.B[i] * max(0.0, times[i] - self.T[i]) for i in seq)

    def arcs(self, seq, rw):
        edges = []
        for group in self.groups(seq, rw):
            for a, i in enumerate(group):
                if self.clock() >= self.end:
                    return None
                for j in group[a + 1:]:
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
            for group in self.groups(seq, rw):
                for pos, i in enumerate(group):
                    if self.clock() >= self.end:
                        return None
                    lo = self.E[i]
                    if pos:
                        lo = max(lo, times[group[pos - 1]])
                    lim, ri = self.s_in[i], rw[i]
                    for k in range(pos - 1, -1, -1):
                        j = group[k]
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
                hi = min(self.L[i], self.T[i])
                if hi <= old:
                    continue
                for b in range(pos + 1, len(seq)):
                    j = seq[b]
                    if times[j] - self.s_out[i] >= hi:
                        break
                    hi = min(hi, times[j] - max(0.0, self.S[i][j][rw[i] * self.r + rw[j]]))
                if hi > old:
                    times[i] = hi
            else:
                lo = max(self.E[i], self.T[i])
                if lo >= old:
                    continue
                for a in range(pos - 1, -1, -1):
                    j = seq[a]
                    if times[j] + self.s_in[i] <= lo:
                        break
                    lo = max(lo, times[j] + max(0.0, self.S[j][i][rw[j] * self.r + rw[i]]))
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

    def evaluate(self, seq, rw, cut=_INF):
        started = self.clock()
        key = self.key(seq, rw)
        hit = self.cache.get(key)
        if hit is not None:
            return hit
        initial = self.earliest(seq, rw, cut)
        if initial is None:
            return None
        times, violation = initial
        if violation <= self.eps and self.safe_m:
            for group in self.groups(seq, rw):
                self.coordinate(group, rw, times, True)
                self.suffix(group, times)
                self.coordinate(group, rw, times, True)
                self.coordinate(group, rw, times, False)
        cost = self.cost(seq, times)
        result = (cost + self.penalty * violation, cost, violation, times)
        self.eval_seconds = 0.9 * self.eval_seconds + 0.1 * max(1.0e-7, self.clock() - started)
        self.remember(seq, rw, result)
        if len(seq) == self.n and (violation, cost) < self.backup_key:
            self.backup_key = (violation, cost)
            self.backup = (list(seq), list(rw), times[:])
        return result

    def refine(self, seq, rw, times, deadline):
        """Starting from the fast timing result, bidirectional closure moves remove the remaining improvements coupled across blocks."""
        begun = self.clock()
        key = self.key(seq, rw)
        cached = self.cache.get(key)
        if cached is not None and cached[2] <= self.eps and cached[1] <= self.cost(seq, times):
            times = cached[3]
        if key in self.optimal_timing:
            return times[:]
        times = times[:]
        edges = self.arcs(seq, rw)
        if edges is None:
            return times
        index = {i: a for a, i in enumerate(seq)}
        best_times, best_cost = times[:], self.cost(seq, times)
        optimal = False
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
                optimal = self.clock() < deadline
                break
        self.refine_seconds = 0.85 * self.refine_seconds + 0.15 * (self.clock() - begun)
        result = self.remember(seq, rw, (best_cost, best_cost, 0.0, best_times[:]), optimal)
        return result[3][:]

    def valid(self, seq, rw, times):
        eps = max(1.0e-7, self.eps * 8)
        if len(seq) != self.n or len(set(seq)) != self.n:
            return False
        for i in seq:
            if not math.isfinite(times[i]) or times[i] < self.E[i] - eps or times[i] > self.L[i] + eps:
                return False
        if self.independent:
            rank = {i: k for k, i in enumerate(seq)}
            seq = sorted(seq, key=lambda i: (times[i], rank[i]))
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
            rank = {i: k for k, i in enumerate(seq)}
            ordered = sorted(seq, key=lambda i: (times[i], rank[i]))
            self.best = (ordered, list(rw), list(times))
            self.last_improvement = self.clock()
            return True
        return False

    def consider(self, seq, rw, result=None, polish=False, until=None):
        if result is None:
            result = self.evaluate(seq, rw)
        cached = self.cache.get(self.key(seq, rw))
        if cached is not None and (result is None or cached[0] < result[0]):
            result = cached
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
            self.elites = [entry for entry in self.elites if entry[1] != key]
            self.elites.append((self.cost(seq, times), key, list(seq), list(rw), times[:]))
            self.elites.sort(key=lambda x: x[0])
            del self.elites[self.pool_size:]
        return changed

    def construct(self, order, noise=0.0):
        seq, rw, times = [], [0] * self.n, self.E[:]
        lanes = [[] for _ in range(self.r)]
        for i in order:
            if self.clock() >= self.end:
                return None
            choices = []
            for r in range(self.r):
                group = lanes[r] if self.independent else seq
                lo = self.E[i]
                if group:
                    lo = max(lo, times[group[-1]])
                for j in reversed(group):
                    if times[j] + self.s_in[i] <= lo:
                        break
                    lo = max(lo, times[j] + self.S[j][i][rw[j] * self.r + r])
                key = (max(0.0, lo - self.L[i]), lo + noise * self.rng.random())
                choices.append((key, lo, r))
            _, times[i], rw[i] = min(choices)
            seq.append(i)
            lanes[rw[i]].append(i)
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
        if self.best is not None or self.mip_failed or self.clock() >= self.end:
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
                            d = float(raw[self.planes[i]][self.planes[j]][ra][rb])
                            m = min(self.M, max(0.0, self.L[i] + d - self.E[j]))
                            add([(j, 1.0), (i, -1.0), (y0 + i * r + a, -m),
                                 (y0 + j * r + b, -m), (z[i][j], -m)], d - 3 * m)
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
            self.mip_failed = True
            return

    def insertion_options(self, i, current, rw, times, limit, deadline):
        """With other times fixed, linearly scan all positions and runways using prefix/suffix bounds."""
        m = len(current)
        options = []
        target = self.T[i]
        average = sum(self.A[j] + self.B[j] for j in current) / max(1, 2 * m)
        for lane in range(self.r):
            if self.clock() >= deadline:
                break
            pl, ph = [self.E[i]] * (m + 1), [self.L[i]] * (m + 1)
            sl, sh = [self.E[i]] * (m + 1), [self.L[i]] * (m + 1)
            for pos, j in enumerate(current):
                pl[pos + 1], ph[pos + 1] = pl[pos], ph[pos]
                if self.independent and rw[j] != lane:
                    continue
                forward = self.S[j][i][rw[j] * self.r + lane]
                reverse = self.S[i][j][lane * self.r + rw[j]]
                pl[pos + 1] = max(pl[pos], times[j] + max(0.0, forward))
                if not self.safe_m:
                    ph[pos + 1] = min(ph[pos], times[j] + self.M - reverse)
            for pos in range(m - 1, -1, -1):
                j = current[pos]
                sl[pos], sh[pos] = sl[pos + 1], sh[pos + 1]
                if self.independent and rw[j] != lane:
                    continue
                forward = self.S[i][j][lane * self.r + rw[j]]
                reverse = self.S[j][i][rw[j] * self.r + lane]
                sh[pos] = min(sh[pos + 1], times[j] - max(0.0, forward))
                if not self.safe_m:
                    sl[pos] = max(sl[pos + 1], times[j] + reverse - self.M)
            for pos in range(m + 1):
                # With independent runways, enumerate only distinct insertion positions within each runway.
                if self.independent and pos and rw[current[pos - 1]] != lane:
                    continue
                lo, hi = max(pl[pos], sl[pos]), min(ph[pos], sh[pos])
                gap = max(0.0, lo - hi)
                if gap <= self.eps:
                    tt = min(hi, max(lo, target))
                else:
                    tt = min(self.L[i], max(self.E[i], min(lo, max(hi, target))))
                own = self.A[i] * max(0.0, target - tt) + self.B[i] * max(0.0, tt - target)
                score = own + gap * (average + min(self.A[i], self.B[i]) + 1.0e-12)
                options.append((score, pos, lane, tt, gap <= self.eps))
        options.sort(key=lambda x: x[0])
        return options[:max(1, limit)]

    def challenge(self, seq, rw, incumbent, deadline, warm=None):
        candidate = self.evaluate(seq, rw, incumbent[0])
        if warm is not None:
            value = self.cost(seq, warm)
            if candidate is None or value < candidate[0]:
                candidate = self.remember(seq, rw, (value, value, 0.0, warm[:]))
        if candidate is None or candidate[2] > self.eps:
            return candidate
        # Do not reject nearby candidates directly by unrefined cost, so closure gains are not cut off.
        slack = max(self.eps, abs(incumbent[0]) / (2 + self.strength))
        if candidate[0] <= incumbent[0] + slack and self.clock() < deadline:
            allowance = min(self.budget / ((self.n + self.r) * (2 + math.sqrt(self.n))),
                            max(self.eval_seconds * (1 + self.strength), self.refine_seconds * 1.5))
            tt = self.refine(seq, rw, candidate[3], min(deadline, self.clock() + allowance))
            value = self.cost(seq, tt)
            candidate = self.remember(seq, rw, (value, value, 0.0, tt))
        return candidate

    def descent(self, seq, rw, result, deadline):
        """Jointly insert over full position and runway candidates, comparing by closure after screening."""
        seq, rw = seq[:], rw[:]
        while self.clock() < deadline:
            result = self.cache.get(self.key(seq, rw), result)
            improved = False
            penalties = {i: self.A[i] * max(0.0, self.T[i] - result[3][i])
                         + self.B[i] * max(0.0, result[3][i] - self.T[i]) for i in seq}
            ids = sorted(seq, key=lambda i: (penalties[i] + 1.0) * (0.5 + self.rng.random()), reverse=True)
            for number, i in enumerate(ids):
                if self.clock() >= deadline:
                    break
                base = [j for j in seq if j != i]
                width = max(2, int(1 + self.strength * math.sqrt(math.log1p(self.n))))
                affordable = int((deadline - self.clock()) / max(1.0e-7,
                    (len(ids) - number) * (self.eval_seconds + self.refine_seconds * 0.35)))
                width = min(width, max(1, affordable))
                options = self.insertion_options(i, base, rw, result[3], width, deadline)
                chosen = None
                incumbent_key = self.key(seq, rw)
                for _, pos, lane, tt, feasible in options:
                    if self.clock() >= deadline:
                        break
                    trial = base[:pos] + [i] + base[pos:]
                    new_rw = rw[:]
                    new_rw[i] = lane
                    if self.key(trial, new_rw) == incumbent_key:
                        continue
                    warm = None
                    if feasible and result[2] <= self.eps:
                        warm = result[3][:]
                        warm[i] = tt
                    candidate = self.challenge(trial, new_rw, result, deadline, warm)
                    if candidate is not None and candidate[0] < result[0] - self.eps:
                        if chosen is None or candidate[0] < chosen[2][0]:
                            chosen = (trial, new_rw, candidate)
                if chosen is not None:
                    seq, rw, result = chosen
                    self.consider(seq, rw, result, until=deadline)
                    result = self.cache.get(self.key(seq, rw), result)
                    improved = True
            # Swapping the runways of two flights can cross local barriers where single-flight rerouting is infeasible.
            for pos in self.rng.sample(range(max(0, self.n - 1)), max(0, self.n - 1)):
                if self.clock() >= deadline:
                    break
                i, j = seq[pos], seq[pos + 1]
                trial = seq[:]
                trial[pos], trial[pos + 1] = j, i
                modes = (False, True) if rw[i] != rw[j] else (False,)
                for swap_lanes in modes:
                    if self.clock() >= deadline:
                        break
                    new_rw = rw[:]
                    if swap_lanes:
                        new_rw[i], new_rw[j] = new_rw[j], new_rw[i]
                    candidate = self.challenge(trial, new_rw, result, deadline)
                    if candidate is not None and candidate[0] < result[0] - self.eps:
                        seq, rw, result = trial, new_rw, candidate
                        self.consider(seq, rw, result, until=deadline)
                        result = self.cache.get(self.key(seq, rw), result)
                        improved = True
                        break
            if not improved:
                break
        return seq, rw, result

    def reinsert(self, base, deadline, stagnation):
        """Destroy adjacent or high-cost flights and pick repair priority by second-best insertion loss."""
        seq, rw, times = base
        span = math.sqrt(self.n * max(0.1, self.strength)) * (0.35 + stagnation)
        effort = max(1.0e-7, (self.eval_seconds + self.refine_seconds) * (2 + self.strength))
        count = min(self.n, max(2, int(span)), max(2, int((deadline - self.clock()) / effort)))
        penalties = [self.A[i] * max(0.0, self.T[i] - times[i])
                     + self.B[i] * max(0.0, times[i] - self.T[i]) for i in range(self.n)]
        mode = self.rng.random()
        if mode < 0.3:
            removed = self.rng.sample(seq, count)
        elif mode < 0.7:
            anchor = max(self.rng.sample(seq, min(self.n, max(1, count))), key=lambda i: penalties[i])
            # Time proximity and same-runway links jointly form the destroy block.
            scale = max(1.0, self.s_in[anchor], self.s_out[anchor])
            removed = sorted(seq, key=lambda i: abs(times[i] - times[anchor])
                             + (0.0 if rw[i] == rw[anchor] else scale * self.rng.random()))[:count]
        else:
            removed = sorted(seq, key=lambda i: (penalties[i] + 1.0) * (0.5 + self.rng.random()), reverse=True)[:count]
        excluded = set(removed)
        current = [i for i in seq if i not in excluded]
        rw, times = rw[:], times[:]
        value = self.cost(current, times)
        result = self.remember(current, rw, (value, value, 0.0, times))
        removed.sort(key=lambda i: (self.L[i] - self.E[i]) / (1 + self.A[i] + self.B[i]))
        while removed and self.clock() < deadline:
            remaining = deadline - self.clock()
            probe = min(len(removed), max(1, int(math.ceil(math.sqrt(self.strength)))))
            affordable = int(remaining / max(1.0e-7,
                len(removed) * (self.eval_seconds + self.refine_seconds) * (1 + self.strength)))
            probe = min(probe, max(1, affordable))
            # Tight-window flights are always checked and others rotate, to avoid always repairing in the same order.
            pool = removed[:1]
            if probe > 1:
                pool += self.rng.sample(removed[1:], probe - 1)
            selected, selected_regret = None, -_INF
            for i in pool:
                width = max(2, int(1 + self.strength * math.sqrt(math.log1p(len(current) + 1))))
                affordable = int((deadline - self.clock()) / max(1.0e-7,
                    len(removed) * probe * (self.eval_seconds + self.refine_seconds * 0.3)))
                width = min(width, max(1, affordable))
                options = self.insertion_options(i, current, rw, result[3], width, deadline)
                choices = []
                for _, pos, lane, tt, feasible in options:
                    if self.clock() >= deadline:
                        break
                    trial = current[:pos] + [i] + current[pos:]
                    new_rw = rw[:]
                    new_rw[i] = lane
                    candidate = self.evaluate(trial, new_rw)
                    if feasible and result[2] <= self.eps:
                        warm = result[3][:]
                        warm[i] = tt
                        value = self.cost(trial, warm)
                        if candidate is None or value < candidate[0]:
                            candidate = self.remember(trial, new_rw, (value, value, 0.0, warm))
                    if candidate is not None:
                        choices.append((candidate[0], trial, new_rw, candidate))
                if not choices:
                    continue
                choices.sort(key=lambda x: x[0])
                # Keep both the refined cost and times, instead of discarding the refinement and reverting to approximate evaluation.
                quota = max(0.0, deadline - self.clock()) / max(1, len(removed) * probe * (len(choices) + 1))
                for idx, choice in enumerate(choices):
                    _, trial, new_rw, candidate = choice
                    if candidate[2] <= self.eps and self.clock() < deadline:
                        tt = self.refine(trial, new_rw, candidate[3], min(deadline, self.clock() + quota))
                        value = self.cost(trial, tt)
                        candidate = self.remember(trial, new_rw, (value, value, 0.0, tt))
                        choices[idx] = (candidate[0], trial, new_rw, candidate)
                choices.sort(key=lambda x: x[0])
                best = choices[0]
                if len(choices) > 1:
                    regret = choices[1][0] - best[0]
                else:
                    regret = abs(best[0] - result[0]) + 1.0
                regret += (1.0 + penalties[i]) / (1.0 + max(0.0, self.L[i] - self.E[i]))
                if regret > selected_regret:
                    selected_regret, selected = regret, (i, best)
            if selected is None:
                return None
            i, choice = selected
            _, current, rw, result = choice
            removed.remove(i)
        if removed:
            return None
        self.consider(current, rw, result, until=deadline)
        result = self.cache.get(self.key(current, rw), result)
        return current, rw, result

    def perturb(self, base, stagnation):
        seq, rw, times = base
        seq, rw = seq[:], rw[:]
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
            self.consider(current[0], current[1], current[2], polish=True, until=local_end)
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
