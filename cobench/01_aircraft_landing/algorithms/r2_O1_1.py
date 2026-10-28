"""Critical-block integer neighbourhood and closure timing search
Jointly optimises critical-block sequence and runways with globally timed mixed-integer subproblems, alternating closure refinement, joint-neighbourhood descent and destroy-reinsert while saving the best feasible solution.
"""

import math
import random
import time
from collections import deque


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
        self.end = started + budget - min(0.08 * budget, 0.025 + self.n * self.n * 0.000002)
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
        self.cache_size = max(16, int((1 + self.strength) * math.sqrt(max(1, self.n)) * 12))
        self.last_improvement = started
        self.solver_failed = False
        self.block_scale = 1.0
        self.block_visits = {}
        self.next_integer = started
        self.integer_seconds = 0.0
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
        self.s_in = [max([0.0] + [max(self.S[j][i]) for j in range(self.n) if j != i])
                     for i in range(self.n)]
        self.s_out = [max([0.0] + [max(self.S[i][j]) for j in range(self.n) if j != i])
                      for i in range(self.n)]

    def cost(self, seq, times):
        return sum(self.A[i] * max(0.0, self.T[i] - times[i])
                   + self.B[i] * max(0.0, times[i] - self.T[i]) for i in seq)

    def arcs(self, seq, rw):
        edges = []
        for a, i in enumerate(seq):
            if self.clock() >= self.end:
                return None
            for j in seq[a + 1:]:
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
                hi = min(self.L[i], self.T[i])
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
                lo = max(self.E[i], self.T[i])
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

    def evaluate(self, seq, rw, cut=_INF):
        started = self.clock()
        key = (tuple(seq), tuple(rw))
        hit = self.cache.get(key)
        if hit is not None:
            return hit
        initial = self.earliest(seq, rw, cut)
        if initial is None:
            return None
        times, violation = initial
        if violation <= self.eps and self.safe_m:
            self.coordinate(seq, rw, times, True)
            self.suffix(seq, times)
            self.coordinate(seq, rw, times, True)
            self.coordinate(seq, rw, times, False)
        cost = self.cost(seq, times)
        result = (cost + self.penalty * violation, cost, violation, times)
        self.eval_seconds = 0.9 * self.eval_seconds + 0.1 * max(1.0e-7, self.clock() - started)
        if len(self.cache) >= self.cache_size:
            self.cache.pop(next(iter(self.cache)))
        self.cache[key] = result
        if len(seq) == self.n and (violation, cost) < self.backup_key:
            self.backup_key = (violation, cost)
            self.backup = (list(seq), list(rw), times[:])
        return result

    def refine(self, seq, rw, times, deadline):
        """Starting from the fast timing result, bidirectional closure moves remove the remaining improvements coupled across blocks."""
        times = times[:]
        edges = self.arcs(seq, rw)
        if edges is None:
            return times
        index = {i: a for a, i in enumerate(seq)}
        best_times, best_cost = times[:], self.cost(seq, times)
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
                break
        value = self.cost(seq, best_times)
        key = (tuple(seq), tuple(rw))
        previous = self.cache.get(key)
        if previous is None or value < previous[0]:
            if len(self.cache) >= self.cache_size:
                self.cache.pop(next(iter(self.cache)))
            self.cache[key] = (value, value, 0.0, best_times[:])
        return best_times

    def valid(self, seq, rw, times):
        eps = max(1.0e-7, self.eps * 8)
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
            self.best = (list(seq), list(rw), list(times))
            self.last_improvement = self.clock()
            self.block_visits.clear()
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
            key = (tuple(seq), tuple(rw))
            if all(entry[1] != key for entry in self.elites):
                self.elites.append((self.cost(seq, times), key, list(seq), list(rw), times[:]))
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
                if seq:
                    lo = max(lo, times[seq[-1]])
                for j in reversed(seq):
                    if times[j] + self.s_in[i] <= lo:
                        break
                    lo = max(lo, times[j] + self.S[j][i][rw[j] * self.r + r])
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

    def challenge(self, seq, rw, incumbent, deadline):
        candidate = self.evaluate(seq, rw, incumbent[0])
        if candidate is None or candidate[2] > self.eps:
            return candidate
        slack = abs(incumbent[0]) / ((1 + self.strength) * math.sqrt(max(1, self.n)))
        if incumbent[0] <= candidate[0] <= incumbent[0] + slack and self.clock() < deadline:
            allowance = max(self.eval_seconds * math.sqrt(self.n),
                            self.budget / (1 + self.n * self.n * (self.r + 1)))
            tt = self.refine(seq, rw, candidate[3], min(deadline, self.clock() + allowance))
            value = self.cost(seq, tt)
            candidate = (value, value, 0.0, tt)
        return candidate

    def integer_neighborhood(self, base, block, free_order, free_lanes, deadline):
        """Fix discrete decisions outside the block; all flight times still take part in jointly optimising the problem's objective."""
        if self.solver_failed or self.clock() >= deadline:
            return None
        started = self.clock()
        try:
            import numpy as np
            from scipy.optimize import Bounds, LinearConstraint, milp
            from scipy.sparse import coo_matrix

            seq, rw, old_times = base
            n, r = self.n, self.r
            active = set(block)
            rank = {i: k for k, i in enumerate(seq)}
            lanes = active if free_lanes and r > 1 else set()
            y = {}
            count = 3 * n
            for i in block:
                if i in lanes:
                    for a in range(r):
                        y[i, a] = count
                        count += 1
            # Relative to the original order, variable value 0 means no reversal; the complementary direction is substituted out.
            z = {}
            if free_order:
                for pos, i in enumerate(block):
                    for j in block[pos + 1:]:
                        z[i, j] = count
                        count += 1
            lo, hi = self.E[:], self.L[:]
            if self.best is not None:
                floor = [self.A[i] * max(0.0, self.T[i] - hi[i])
                         + self.B[i] * max(0.0, lo[i] - self.T[i]) for i in range(n)]
                spare = max(0.0, self.best_cost - sum(floor))
                cushion = max(1.0e-7, abs(self.best_cost) * 1.0e-9)
                for i in range(n):
                    available = spare + floor[i] + cushion
                    if self.A[i] > 0:
                        lo[i] = max(lo[i], self.T[i] - available / self.A[i])
                    if self.B[i] > 0:
                        hi[i] = min(hi[i], self.T[i] + available / self.B[i])
            objective = np.zeros(count)
            objective[n:2 * n], objective[2 * n:3 * n] = self.A, self.B
            lower, upper = np.zeros(count), np.ones(count)
            lower[:n], upper[:n] = lo, hi
            upper[n:2 * n] = [max(0.0, self.T[i] - lo[i]) for i in range(n)]
            upper[2 * n:3 * n] = [max(0.0, hi[i] - self.T[i]) for i in range(n)]
            integrality = np.zeros(count, dtype=int)
            integrality[3 * n:] = 1
            rows, cols, values, lhs, rhs = [], [], [], [], []

            def add(terms, low, high=_INF):
                row = len(lhs)
                for col, value in terms:
                    if value:
                        rows.append(row)
                        cols.append(col)
                        values.append(value)
                lhs.append(low)
                rhs.append(high)

            for i in range(n):
                add([(i, 1.0), (n + i, 1.0)], self.T[i])
                add([(i, -1.0), (2 * n + i, 1.0)], -self.T[i])
                if i in lanes:
                    # The original runway slot also uses the complemented variable, so the all-zero integer vector corresponds to the current assignment.
                    add([(y[i, a], -1.0 if a == rw[i] else 1.0)
                         for a in range(r)], 0.0, 0.0)
            if self.best is not None:
                add([(n + i, self.A[i]) for i in range(n)]
                    + [(2 * n + i, self.B[i]) for i in range(n)],
                    -_INF, self.best_cost + cushion)

            for pos, u in enumerate(seq):
                if self.clock() >= deadline:
                    return None
                for v in seq[pos + 1:]:
                    flip = z.get((u, v))
                    for i, j, forward in ((u, v, True), (v, u, False)):
                        if flip is None and not forward and self.safe_m:
                            continue
                        aa = range(r) if i in lanes else (rw[i],)
                        bb = range(r) if j in lanes else (rw[j],)
                        for a in aa:
                            for b in bb:
                                d = self.S[i][j][a * r + b]
                                if flip is None and not forward:
                                    d -= self.M
                                needed = hi[i] + d - lo[j]
                                if needed <= 0.0:
                                    continue
                                terms = [(j, 1.0), (i, -1.0)]
                                bound = d
                                # First aggregate all unselected-runway constraints of the original model, then disjunct over the assignment
                                # using tight bounds derived from time windows; with small M the reverse residual is also kept.
                                if flip is not None:
                                    mz = min(self.M, needed)
                                    if forward:
                                        terms.append((flip, mz))
                                    else:
                                        terms.append((flip, -mz))
                                        bound -= mz
                                for q, a_q in ((i, a), (j, b)):
                                    if q in lanes:
                                        if a_q == rw[q]:
                                            terms.append((y[q, a_q], needed))
                                        else:
                                            terms.append((y[q, a_q], -needed))
                                            bound -= needed
                                add(terms, bound)
                    # Explicitly enforce sequence semantics; zero separations and simultaneous landings also get a consistent order.
                    if flip is None:
                        if lo[v] < hi[u]:
                            add([(v, 1.0), (u, -1.0)], 0.0)
                    else:
                        b1, b2 = max(0.0, hi[u] - lo[v]), max(0.0, hi[v] - lo[u])
                        add([(v, 1.0), (u, -1.0), (flip, b1)], 0.0)
                        add([(u, 1.0), (v, -1.0), (flip, -b2)], -b2)
            # Triangle inequalities constrain only z, excluding cyclic orders under zero separation.
            if free_order:
                for a, i in enumerate(block):
                    if self.clock() >= deadline:
                        return None
                    for b in range(a + 1, len(block)):
                        j = block[b]
                        for k in block[b + 1:]:
                            add([(z[i, j], 1.0), (z[j, k], 1.0), (z[i, k], -1.0)], 0.0, 1.0)
            if self.clock() >= deadline:
                return None
            matrix = coo_matrix((values, (rows, cols)), shape=(len(lhs), count)).tocsc()
            remaining = deadline - self.clock()
            if remaining <= max(0.001, self.budget * 0.0005):
                return None
            result = milp(objective, integrality=integrality, bounds=Bounds(lower, upper),
                          constraints=LinearConstraint(matrix, np.asarray(lhs), np.asarray(rhs)),
                          options={"time_limit": remaining * 0.85, "mip_rel_gap": 0.0,
                                   "presolve": True})
            self.integer_seconds = self.clock() - started
            if result.x is None:
                self.block_scale = max(0.55, self.block_scale / (1 + 1 / (1 + self.strength)))
                return None
            x = result.x
            new_rw = rw[:]
            for i in lanes:
                new_rw[i] = max(range(r), key=lambda a: (1.0 - x[y[i, a]]) if a == rw[i] else x[y[i, a]])
            new_seq = seq[:]
            if free_order:
                before = {i: 0 for i in block}
                for (i, j), col in z.items():
                    if x[col] > 0.5:
                        before[i] += 1
                    else:
                        before[j] += 1
                order = sorted(block, key=lambda i: (before[i], rank[i]))
                positions = sorted(rank[i] for i in block)
                for pos, i in zip(positions, order):
                    new_seq[pos] = i
            tt = [float(x[i]) for i in range(n)]
            # Tiny errors from solver tolerance are not passed to the retained solution; if needed, re-time with the discrete solution fixed.
            if self.valid(new_seq, new_rw, tt):
                value = self.cost(new_seq, tt)
                candidate = (value, value, 0.0, tt)
                self.cache[(tuple(new_seq), tuple(new_rw))] = candidate
                self.publish(new_seq, new_rw, tt)
            else:
                candidate = self.evaluate(new_seq, new_rw)
                if candidate is None or candidate[2] > self.eps:
                    return None
                self.consider(new_seq, new_rw, candidate, until=deadline)
            if result.status == 0:
                self.block_scale = min(1 + self.strength, self.block_scale * (1 + 1 / (2 + self.strength)))
            return new_seq, new_rw, candidate
        except Exception:
            self.solver_failed = True
            return None

    def feasibility(self):
        if self.best is not None or self.clock() >= self.end:
            return
        if self.backup is not None:
            base = self.backup
        else:
            seq = sorted(range(self.n), key=lambda i: (self.L[i], self.T[i]))
            base = (seq, [0] * self.n, self.E[:])
        deadline = self.clock() + (self.end - self.clock()) * 0.7
        self.integer_neighborhood(base, base[0], True, True, deadline)

    def critical_block(self):
        seq, rw, tt = self.best
        choice = self.rng.random()
        mode = 0 if self.r == 1 or choice < 0.6 else (1 if choice < 0.8 else 2)
        # Neighbourhood width changes with time limit, size, number of runways and the actual solve status of subproblems.
        width = (2 + self.strength) * math.sqrt(math.log1p(self.n))
        width *= self.block_scale / (1 + 0.2 * max(0, self.r - 1))
        if mode:
            width *= math.sqrt(2.0)
        width = min(self.n, max(2, int(width)))
        penalties = [self.A[i] * max(0.0, self.T[i] - tt[i])
                     + self.B[i] * max(0.0, tt[i] - self.T[i]) for i in seq]
        prefix = [0.0]
        for value in penalties:
            prefix.append(prefix[-1] + value)
        floor = max(1.0e-9, sum(penalties) / max(1, self.n))
        scores = []
        for start in range(self.n - width + 1):
            seen = self.block_visits.get((mode, start, width), 0)
            score = (prefix[start + width] - prefix[start] + floor) / (1 + seen)
            scores.append(score * (0.5 + self.rng.random()))
        start = max(range(len(scores)), key=scores.__getitem__)
        if self.rng.random() < 1 / (2 + self.strength):
            start = self.rng.randrange(len(scores))
        key = (mode, start, width)
        self.block_visits[key] = self.block_visits.get(key, 0) + 1
        return seq[start:start + width], mode != 1, mode != 2

    def descent(self, seq, rw, result, deadline):
        """Joint sequence and runway changes; fast evaluation and closure refinement share the same retained solution."""
        seq, rw = seq[:], rw[:]
        while self.clock() < deadline:
            result = self.cache.get((tuple(seq), tuple(rw)), result)
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
                    candidate = self.challenge(trial, new_rw, result, deadline)
                    if candidate is not None and candidate[0] < result[0] - 1.0e-9:
                        seq, rw, result = trial, new_rw, candidate
                        self.consider(seq, rw, result, until=deadline)
                        result = self.cache.get((tuple(seq), tuple(rw)), result)
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
                    candidate = self.challenge(seq, new_rw, result, deadline)
                    if candidate is not None and candidate[0] < result[0] - 1.0e-9:
                        rw, result = new_rw, candidate
                        self.consider(seq, rw, result, until=deadline)
                        result = self.cache.get((tuple(seq), tuple(rw)), result)
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
                candidate = self.challenge(trial, new_rw, result, deadline)
                if candidate is not None and candidate[0] < result[0] - 1.0e-9:
                    seq, rw, result = trial, new_rw, candidate
                    self.consider(seq, rw, result, until=deadline)
                    result = self.cache.get((tuple(seq), tuple(rw)), result)
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
            keep = max(1, int(1 + self.strength / math.sqrt(max(1, self.n / 4))))
            for pos in positions:
                trial = current[:pos] + [i] + current[pos:]
                for r in range(self.r):
                    if self.clock() >= deadline:
                        return None
                    new_rw = rw[:]
                    new_rw[i] = r
                    result = self.evaluate(trial, new_rw)
                    if result is None:
                        continue
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
        # Search termination depends only on the wall clock; integer and fast neighbourhoods alternate by actual elapsed time.
        while self.clock() < self.end:
            now = self.clock()
            if self.best is None:
                order = sorted(range(self.n), key=lambda i: self.L[i] + self.rng.uniform(-1, 1) * (self.L[i] - self.E[i]))
                state = self.construct(order)
                if state is not None:
                    result = self.evaluate(*state)
                    if result is not None:
                        local_end = min(self.end, now + self.budget / (3 + math.sqrt(self.n)))
                        seq, rw, result = self.descent(*state, result, local_end)
                        self.consider(seq, rw, result, polish=True)
                if self.best is None and not self.solver_failed:
                    self.feasibility()
                continue
            if current is None:
                seq, rw, _ = self.best
                result = self.evaluate(seq, rw)
                if result is None:
                    continue
                current = (seq[:], rw[:], result)
            local_end = min(self.end, now + self.budget / (3 + math.sqrt(self.n)))
            current = self.descent(*current, local_end)
            self.consider(current[0], current[1], current[2], polish=True, until=local_end)
            if self.clock() >= self.end:
                break
            if not self.solver_failed and self.clock() >= self.next_integer and self.n > 1:
                share = self.budget / (3 + math.sqrt(self.n))
                if self.end - self.clock() > max(0.04, share * 0.35):
                    before = self.clock()
                    block, free_order, free_lanes = self.critical_block()
                    candidate = self.integer_neighborhood(self.best, block, free_order, free_lanes,
                                                          min(self.end, before + share))
                    spent = self.clock() - before
                    self.next_integer = self.clock() + spent * (1 + self.n ** 0.25 / (1 + self.strength))
                    if candidate is not None:
                        current = candidate
                        self.consider(current[0], current[1], current[2])
                        continue
            stagnation = min(1.0, (self.clock() - self.last_improvement) / max(1.0e-9, self.budget * 0.25))
            base = self.best
            if self.elites and self.rng.random() < 0.3:
                entry = self.rng.choice(self.elites)
                base = (entry[2], entry[3], entry[4])
            if self.n > 1 and self.rng.random() < 0.35 + 0.4 * stagnation:
                repair_end = min(self.end, self.clock() + self.budget / (3 + math.sqrt(self.n)))
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
