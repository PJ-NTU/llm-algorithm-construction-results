"""Multi-label path column generation with integer recombination
Combines cost matching, duty-start propagation, multi-label shortest-path pricing and feasible full-path integer partitioning, expanding candidates within the wall-clock budget and keeping the best roster.
"""

import bisect
import math
import random
import time

try:
    import numpy as np
    from scipy.optimize import Bounds, LinearConstraint, linear_sum_assignment, linprog, milp
    from scipy.sparse import coo_matrix
except Exception:
    np = None


# diagnosis: the stronger member coordinates multiple paths with duty-start propagation and full-path integer partitioning.
# Scores are identical, so the weaker member cannot be judged to actually lag; scattered search budget and repeated single-label pricing are code-level limitations.
# This implementation deepens the full-path mechanism with batched multi-label pricing, a complete path library when enumerable, and hard crew-count integer recombination.


def _output(routes, k):
    rows = [list(r) for r in routes]
    rows.extend([] for _ in range(max(0, k - len(rows))))
    return {"crew_routes": rows,
            "start_tasks": [r[0] if r else -1 for r in rows],
            "end_tasks": [r[-1] if r else -1 for r in rows]}


class _Search:
    def __init__(self, instance, budget, started):
        self.n = int(instance["num_tasks"])
        self.k = int(instance["max_crews"])
        self.limit = float(instance["max_duty_time"])
        self.s = list(map(float, instance["start_times"]))
        self.f = list(map(float, instance["finish_times"]))
        self.c = instance["transition_costs"]
        self.v = instance["valid_transitions"]
        self.budget = max(0.0, budget)
        self.started = started
        self.deadline = started + self.budget * 0.94
        self.rng = random.Random(782347)
        self.best = None
        self.best_cost = float("inf")
        self.lower = -float("inf")
        self.last_improvement = started
        self.pool = []
        self.costs = []
        self.index = {}
        self.scale = 1.0
        self.arc_model = None
        self.price_cursor = 0
        self.price_speed = 1.0e7
        self.last_values = None
        self.path_cuts = set()
        self.all_columns = False
        self.enumeration = None
        self.enumeration_cursor = 0

    def remaining(self):
        return max(0.0, self.deadline - time.perf_counter())

    def valid_route(self, route):
        if not route:
            return False
        if self.f[route[-1]] - self.s[route[0]] > self.limit:
            return False
        return all(self.f[u] <= self.s[v] and self.v[u][v] == 1
                   for u, v in zip(route, route[1:]))

    def remember(self, routes):
        routes = [tuple(r) for r in routes if r]
        seen = set()
        for route in routes:
            if not self.valid_route(route):
                return False
            for t in route:
                if not 0 <= t < self.n or t in seen:
                    return False
                seen.add(t)
        if len(seen) != self.n:
            return False
        # With idle crews only positive-cost edges are cut; negative-cost edges are not wrongly forced apart.
        room = self.k - len(routes)
        if room > 0:
            cuts = sorted(((float(self.c[u][v]), q, j)
                           for q, r in enumerate(routes)
                           for j, (u, v) in enumerate(zip(r, r[1:]))
                           if float(self.c[u][v]) > 0), reverse=True)
            selected = {(q, j) for _, q, j in cuts[:room]}
            split = []
            for q, r in enumerate(routes):
                left = 0
                for j in range(len(r) - 1):
                    if (q, j) in selected:
                        split.append(r[left:j + 1])
                        left = j + 1
                split.append(r[left:])
            routes = split
        cost = sum(float(self.c[u][v]) for r in routes for u, v in zip(r, r[1:]))
        if len(routes) <= self.k and cost < self.best_cost - 1e-9:
            self.best = routes
            self.best_cost = cost
            self.last_improvement = time.perf_counter()
            for route in routes:
                self.add(route)
            return True
        return False

    def add(self, route):
        route = tuple(route)
        if not route or route in self.index:
            return 0
        if len(set(route)) != len(route) or not self.valid_route(route):
            return 0
        self.index[route] = len(self.pool)
        self.pool.append(route)
        self.costs.append(sum(float(self.c[u][v]) for u, v in
                              zip(route, route[1:])) / self.scale)
        return 1

    def include(self, routes, fragments=False):
        stop = min(self.deadline, time.perf_counter() + self.budget * 0.02)
        for route in routes:
            self.add(route)
            if fragments:
                for a in range(len(route)):
                    for b in range(a + 1, len(route) + 1):
                        if time.perf_counter() >= stop:
                            return
                        if self.f[route[b - 1]] - self.s[route[a]] > self.limit:
                            break
                        self.add(route[a:b])

    def prepare(self):
        n = self.n
        self.edges = []
        self.adj = [[] for _ in range(n)]
        self.allowed = [set() for _ in range(n)]
        self.incoming = [[] for _ in range(n)]
        self.nonnegative = True
        for i in range(n):
            for j in range(n):
                if (i != j and self.v[i][j] == 1 and self.f[i] <= self.s[j]
                        and self.f[j] - self.s[i] <= self.limit):
                    w = float(self.c[i][j])
                    self.scale = max(self.scale, abs(w))
                    self.nonnegative = self.nonnegative and w >= 0
                    e = len(self.edges)
                    self.edges.append((i, j, w))
                    self.allowed[i].add(j)
                    self.incoming[j].append((i, e))
        degree = [len(row) for row in self.incoming]
        ready = [i for i in range(n) if degree[i] == 0]
        self.order = []
        while ready:
            u = ready.pop()
            self.order.append(u)
            for v in self.allowed[u]:
                degree[v] -= 1
                if degree[v] == 0:
                    ready.append(v)
        self.dag = len(self.order) == n
        if not self.dag:
            self.order = sorted(range(n), key=lambda i: (self.s[i], self.f[i], i))
        self.rank = [0] * n
        for q, i in enumerate(self.order):
            self.rank[i] = q
        for e, (i, j, w) in enumerate(self.edges):
            self.adj[i].append((j, w / self.scale, e))
        self.width = max(2, int((2.0 + math.log1p(self.budget)) * math.sqrt(n + 1)))
        self.batch = max(1, int(math.ceil(math.log2(2.0 + self.budget))))
        self.pool_cap = max(3 * n, int(n * (8 + 6 * math.log1p(self.budget))
                                      * (1 + math.sqrt(n / max(1, self.k)))))
        self.pool_cap = max(self.pool_cap, n + self.k)
        for i in range(n):
            self.add((i,))
        if np is not None:
            self.in_nodes = [np.array([i for i, _ in row], dtype=np.int32)
                             for row in self.incoming]
            self.in_edges = [np.array([e for _, e in row], dtype=np.int32)
                             for row in self.incoming]
            self.weights = np.array([w / self.scale for _, _, w in self.edges])
        if self.k >= n:
            self.remember([(i,) for i in range(n)])
        # Enumerate completely when the number of paths is manageable; otherwise leave the budget to dual-guided multi-label pricing.
        if self.dag:
            counts = [0] * n
            for u in self.order:
                counts[u] = min(self.pool_cap + 1,
                                1 + sum(counts[i] for i, _ in self.incoming[u]))
            if sum(counts) <= self.pool_cap:
                self.enumeration = []

    def enumerate_routes(self, stop):
        if self.enumeration is None or self.all_columns:
            return
        while time.perf_counter() < stop:
            if not self.enumeration:
                if self.enumeration_cursor >= self.n:
                    self.all_columns = True
                    return
                first = self.order[self.enumeration_cursor]
                self.enumeration_cursor += 1
                self.enumeration.append((first,))
            route = self.enumeration.pop()
            self.add(route)
            for v, _, _ in self.adj[route[-1]]:
                if self.f[v] - self.s[route[0]] <= self.limit:
                    self.enumeration.append(route + (v,))

    def matching(self):
        if np is None or self.remaining() <= 0 or self.k <= 0:
            return False
        n, k = self.n, min(self.n, self.k)
        a = np.full((n + k, n + k), np.inf)
        a[:n, n:] = 0.0
        a[n:, :] = 0.0
        for i, j, w in self.edges:
            a[i, j] = w / self.scale
        try:
            rows, cols = linear_sum_assignment(a)
        except Exception:
            return False
        # The assignment relaxation allows cycles and over-duty paths, so its objective value is always a valid lower bound.
        self.lower = max(self.lower, float(a[rows, cols].sum()) * self.scale)
        succ = {int(i): int(j) for i, j in zip(rows, cols) if i < n and j < n}
        incoming = set(succ.values())
        seen, routes = set(), []
        cyclic = False
        for first in [i for i in range(n) if i not in incoming] + list(range(n)):
            if first in seen:
                continue
            r, u = [], first
            while u not in seen:
                seen.add(u)
                r.append(u)
                if u not in succ:
                    break
                u = succ[u]
            if r and r[-1] in succ and succ[r[-1]] == r[0]:
                cyclic = True
                self.path_cuts.add(tuple(zip(r, r[1:] + r[:1])))
                q = max(range(len(r)), key=lambda z: float(self.c[r[z]][r[(z + 1) % len(r)]]))
                r = r[q + 1:] + r[:q + 1]
            routes.append(tuple(r))
        self.remember(routes)
        if not cyclic and self.best is not None and self.best_cost <= self.lower + 1e-7 * self.scale:
            return True
        split = []
        for r in routes:
            left = 0
            for j, t in enumerate(r):
                if self.f[t] - self.s[r[left]] > self.limit:
                    split.append(r[left:j])
                    left = j
            if left < len(r):
                split.append(r[left:])
            # Cuts on the shortest over-duty sub-chains strengthen the subsequent start-propagation model.
            end = 0
            for start in range(len(r)):
                end = max(end, start)
                while end < len(r) and self.f[r[end]] - self.s[r[start]] <= self.limit:
                    end += 1
                if end < len(r):
                    self.path_cuts.add(tuple(zip(r[start:end], r[start + 1:end + 1])))
        self.remember(split)
        self.include(split, fragments=True)
        return False

    def beam(self, stop, randomized=False):
        """Deduplicate by endpoint state; keep candidates for every number of used crews, to avoid favouring only opening or merging crews."""
        states = [(0.0, ())]
        for t in self.order:
            if time.perf_counter() >= stop:
                break
            candidates = {}
            for cost, routes in states:
                if time.perf_counter() >= stop:
                    break
                for a, route in enumerate(routes):
                    if (t in self.allowed[route[-1]]
                            and self.f[t] - self.s[route[0]] <= self.limit):
                        new = routes[:a] + (route + (t,),) + routes[a + 1:]
                        value = cost + float(self.c[route[-1]][t]) / self.scale
                        key = tuple(sorted((r[0], r[-1]) for r in new))
                        old = candidates.get(key)
                        if old is None or value < old[0]:
                            candidates[key] = (value, new)
                if len(routes) < self.k and self.f[t] - self.s[t] <= self.limit:
                    new = routes + ((t,),)
                    key = tuple(sorted((r[0], r[-1]) for r in new))
                    old = candidates.get(key)
                    if old is None or cost < old[0]:
                        candidates[key] = (cost, new)
            if not candidates:
                break
            groups = {}
            for value, routes in candidates.values():
                score = value
                if randomized:
                    score += self.rng.random() * math.sqrt(1 + self.n)
                groups.setdefault(len(routes), []).append((score, value, routes))
            quota = max(1, self.width // len(groups))
            chosen, spare = [], []
            for bucket in groups.values():
                bucket.sort(key=lambda z: z[0])
                chosen.extend(bucket[:quota])
                spare.extend(bucket[quota:])
            if len(chosen) < self.width:
                spare.sort(key=lambda z: z[0])
                chosen.extend(spare[:self.width - len(chosen)])
            states = [(value, routes) for _, value, routes in chosen]
        for _, routes in states:
            self.remember(routes)
            self.include(routes)

    def build_arc_model(self):
        """Exact projection after eliminating crew labels; the output paths uniquely recover the corresponding x, y and first/last tasks."""
        n, m = self.n, len(self.edges)
        if not m or np is None or self.remaining() <= 0:
            return
        zero_edges = [e for e, (u, v, _) in enumerate(self.edges)
                      if self.s[u] == self.f[u] == self.s[v] == self.f[v]]
        extra = n if zero_edges else 0
        size = m + n + extra
        offset = min(self.s)
        unit = max(1.0, self.limit, max(self.f) - offset)
        starts = sorted(set(self.s))
        lows = [(starts[min(len(starts) - 1, bisect.bisect_left(starts, x - self.limit))]
                 - offset) / unit for x in self.f]
        highs = [(x - offset) / unit for x in self.s]
        rr, cc, vv = [], [], []
        lb = [-float("inf")] * (2 * n) + [max(0, n - self.k)]
        ub = [1.0] * (2 * n) + [float("inf")]
        row = 2 * n + 1
        for e, (i, j, _) in enumerate(self.edges):
            if e % max(1, n) == 0 and self.remaining() <= 0:
                return
            rr.extend((i, n + j, 2 * n))
            cc.extend((e, e, e))
            vv.extend((1.0, 1.0, 1.0))
            # h_j <= h_i: propagate along the path the upper bound on the start time of its first task.
            # F_i-L <= h_i <= S_i also enforces the span of the whole duty.
            big = max(0.0, highs[j] - lows[i])
            rr.extend((row, row, row))
            cc.extend((e, m + j, m + i))
            vv.extend((big, 1.0, -1.0))
            lb.append(-float("inf"))
            ub.append(big)
            row += 1
        for e in zero_edges:
            i, j, _ = self.edges[e]
            rr.extend((row, row, row))
            cc.extend((e, m + n + i, m + n + j))
            vv.extend((float(n), 1.0, -1.0))
            lb.append(-float("inf"))
            ub.append(float(n - 1))
            row += 1
        edge_index = {(i, j): e for e, (i, j, _) in enumerate(self.edges)}
        for cut in self.path_cuts:
            if cut and all(arc in edge_index for arc in cut):
                for arc in cut:
                    rr.append(row)
                    cc.append(edge_index[arc])
                    vv.append(1.0)
                lb.append(-float("inf"))
                ub.append(float(len(cut) - 1))
                row += 1
        matrix = coo_matrix((vv, (rr, cc)), shape=(row, size)).tocsc()
        bounds = Bounds(np.array([0.0] * m + lows + [0.0] * extra),
                        np.array([1.0] * m + highs + [float(n - 1)] * extra))
        integer = np.array([1] * m + [0] * (n + extra), dtype=np.uint8)
        objective = np.array([w / self.scale for _, _, w in self.edges]
                             + [0.0] * (n + extra))
        self.arc_model = (matrix, np.array(lb), np.array(ub), bounds,
                          integer, objective)

    def decode_arcs(self, values):
        out, inc = [-1] * self.n, [-1] * self.n
        for e, (i, j, _) in enumerate(self.edges):
            if values[e] > 0.5:
                if out[i] >= 0 or inc[j] >= 0:
                    return []
                out[i], inc[j] = j, i
        routes, visited = [], set()
        for i in range(self.n):
            if inc[i] < 0:
                route = []
                while i >= 0:
                    if i in visited:
                        return []
                    visited.add(i)
                    route.append(i)
                    i = out[i]
                routes.append(tuple(route))
        return routes if len(visited) == self.n else []

    def solve_arcs(self, allowance, feasibility=False):
        if self.arc_model is None or self.remaining() <= 0:
            return False
        matrix, lb, ub0, bounds, integer, objective = self.arc_model
        ub = ub0.copy()
        if feasibility or self.nonnegative:
            ub[2 * self.n] = max(0, self.n - self.k)
        c = np.zeros(len(objective)) if feasibility else objective
        try:
            result = milp(c, integrality=integer, bounds=bounds,
                          constraints=LinearConstraint(matrix, lb, ub),
                          options={"time_limit": max(1e-6, min(allowance, self.remaining()) * 0.92),
                                   "mip_rel_gap": 0.0, "presolve": True})
        except Exception:
            return False
        if result.x is not None:
            routes = self.decode_arcs(result.x)
            if routes:
                self.remember(routes)
                self.include(routes, fragments=True)
        if not feasibility:
            bound = getattr(result, "mip_dual_bound", None)
            if bound is not None and math.isfinite(bound):
                self.lower = max(self.lower, float(bound) * self.scale)
        return (not feasibility and self.best is not None
                and self.best_cost <= self.lower + 1e-7 * self.scale)

    def master(self, allowance, integer=False):
        if not self.pool or self.remaining() <= 0 or np is None:
            return None
        stop = min(self.deadline, time.perf_counter() + allowance)
        n, p = self.n, len(self.pool)
        rr, cc, vv = [], [], []
        for j, route in enumerate(self.pool):
            rr.extend(route)
            cc.extend([j] * len(route))
            vv.extend([1.0] * len(route))
            rr.append(n)
            cc.append(j)
            vv.append(1.0)
        rr.append(n)
        cc.append(p)
        vv.append(-1.0)
        a = coo_matrix((vv, (rr, cc)), shape=(n + 1, p + 1)).tocsc()
        # The continuous master problem allows artificial excess-crew variables to obtain directions while columns are missing; once a feasible solution exists the integer master problem fixes them to zero.
        c = np.array(self.costs + [float(4 * n + 1)])
        duration = (stop - time.perf_counter()) * 0.86
        if duration <= 0:
            return None
        try:
            if integer:
                slack_bound = 0.0 if self.best is not None else float(n)
                result = milp(c, integrality=np.array([1] * p + [0], dtype=np.uint8),
                              bounds=Bounds(np.zeros(p + 1), np.array([1.0] * p + [slack_bound])),
                              constraints=LinearConstraint(a, np.array([1.0] * n + [-np.inf]),
                                                           np.array([1.0] * n + [float(self.k)])),
                              options={"time_limit": max(1e-6, duration),
                                       "mip_rel_gap": 0.0, "presolve": True})
            else:
                result = linprog(c, A_ub=a[n:n + 1], b_ub=[self.k],
                                 A_eq=a[:n], b_eq=np.ones(n), bounds=(0.0, None),
                                 method="highs", options={"time_limit": max(1e-6, duration),
                                                          "presolve": True})
        except Exception:
            return None
        if result.x is not None:
            values = result.x[:p]
            if integer or np.max(np.abs(values - np.rint(values))) <= 1e-7:
                self.remember([self.pool[j] for j in range(p) if values[j] > 0.5])
            if not integer:
                self.last_values = {self.pool[j] for j in range(p) if values[j] > 1e-7}
        if integer and self.all_columns and self.best is not None:
            bound = getattr(result, "mip_dual_bound", None)
            # While the artificial excess-crew variables are open, the integer master problem is still a lower-bound relaxation of the original problem.
            if bound is not None and math.isfinite(bound):
                self.lower = max(self.lower, float(bound) * self.scale)
        if not integer and result.status == 0:
            return (list(result.eqlin.marginals), float(result.ineqlin.marginals[0]),
                    float(result.fun), result.x)
        return None

    def price(self, dual, crew_dual, stop, noisy=False, labels=1):
        if np is None or self.n == 0:
            return 0, False, -float("inf")
        n, order, width = self.n, self.order, max(1, labels)
        pi = np.array(dual, dtype=float)
        weights = self.weights.copy()
        age = (time.perf_counter() - self.last_improvement) / max(self.budget, 1e-9)
        if noisy:
            amplitude = (0.03 + min(0.6, age)) * (1 + abs(crew_dual) / max(1, n))
            pi += np.array([amplitude * (self.rng.random() - 0.5) for _ in range(n)])
            weights += np.array([amplitude * (self.rng.random() - 0.5) for _ in self.edges])
        if not self.dag:
            return self.price_simple(pi, weights, crew_dual, stop)
        starts = order[self.price_cursor:] + order[:self.price_cursor]
        added, lowest, cursor = 0, float("inf"), 0
        while cursor < n and time.perf_counter() < stop:
            available = stop - time.perf_counter()
            work = max(1, width * (len(self.edges) + n))
            # The batch width is bounded by the remaining wall clock, the measured rate and the adaptive memory size.
            memory = max(n, self.pool_cap * self.batch)
            batch = min(n - cursor, max(1, int(available * self.price_speed * 0.45 / work)),
                        max(1, memory // max(1, n * width)))
            sources = starts[cursor:cursor + batch]
            tick = time.perf_counter()
            dist = np.full((batch, n, width), np.inf)
            prev = np.full((batch, n, width), -1, dtype=np.int32)
            level = np.zeros((batch, n, width), dtype=np.int32)
            roots = {h: q for q, h in enumerate(sources)}
            source_time = np.array([self.s[h] for h in sources])
            done = 0
            for j in order:
                if time.perf_counter() >= stop:
                    elapsed = time.perf_counter() - tick
                    if done and elapsed > 0:
                        self.price_speed = min(self.price_speed,
                                               batch * work * done / max(1, n) / elapsed)
                    return added, False, lowest
                predecessors = self.in_nodes[j]
                if len(predecessors):
                    values = (dist[:, predecessors, :]
                              + weights[self.in_edges[j]][None, :, None]).reshape(batch, -1)
                    if width == 1:
                        ix = np.argmin(values, axis=1)[:, None]
                    else:
                        ix = np.argpartition(values, width - 1, axis=1)[:, :width]
                        selected = np.take_along_axis(values, ix, axis=1)
                        ix = np.take_along_axis(ix, np.argsort(selected, axis=1), axis=1)
                    dist[:, j, :] = np.take_along_axis(values, ix, axis=1) - pi[j]
                    prev[:, j, :] = predecessors[ix // width]
                    level[:, j, :] = ix % width
                    invalid = self.f[j] - source_time > self.limit
                    dist[invalid, j, :] = np.inf
                    prev[invalid, j, :] = -1
                root = roots.get(j)
                if root is not None:
                    dist[root, j, :] = np.inf
                    dist[root, j, 0] = -pi[j]
                    prev[root, j, :] = -1
                done += 1
            elapsed = time.perf_counter() - tick
            if elapsed > 0:
                speed = batch * work / elapsed
                self.price_speed = 0.5 * self.price_speed + 0.5 * speed
            limit = self.batch * width
            for q, first in enumerate(sources):
                if time.perf_counter() >= stop:
                    return added, False, lowest
                flat = (dist[q] - crew_dual).ravel()
                lowest = min(lowest, float(np.min(flat)))
                # The multi-label phase also collects paths with near-zero reduced cost, so that integer recombination is not limited to negative-price columns.
                threshold = (0.02 * width * (1 + abs(crew_dual) / max(1, n))
                             if width > 1 or noisy else -1e-8)
                candidates = np.flatnonzero(np.isfinite(flat) & (flat < threshold))
                candidates = candidates[np.argsort(flat[candidates])]
                count = 0
                for flat_index in candidates:
                    if time.perf_counter() >= stop:
                        return added, False, lowest
                    u, h = divmod(int(flat_index), width)
                    route = []
                    while u >= 0:
                        route.append(u)
                        u, h = int(prev[q, u, h]), int(level[q, u, h])
                    route.reverse()
                    if route and route[0] == first:
                        new = self.add(route)
                        added += new
                        count += new
                    if count >= limit:
                        break
            cursor += batch
            self.price_cursor = (self.price_cursor + batch) % n
        return added, cursor == n, lowest

    def price_simple(self, pi, weights, crew_dual, stop):
        # Zero-duration tasks at the same time may form cycles; pricing only extends to unvisited tasks and never deletes valid reverse edges.
        added = 0
        while time.perf_counter() < stop:
            first = self.order[self.price_cursor]
            self.price_cursor = (self.price_cursor + 1) % self.n
            stack = [((first,), 1 << first, -float(pi[first]))]
            local_stop = min(stop, time.perf_counter() + (stop - time.perf_counter())
                             / max(1.0, math.sqrt(self.n)))
            while stack and time.perf_counter() < local_stop:
                route, seen, value = stack.pop()
                if value - crew_dual < 1e-7:
                    added += self.add(route)
                choices = [(float(weights[e]) - float(pi[v]), v) for v, _, e in self.adj[route[-1]]
                           if not (seen >> v) & 1 and self.f[v] - self.s[first] <= self.limit]
                choices.sort(reverse=True)
                for delta, v in choices:
                    stack.append((route + (v,), seen | (1 << v), value + delta))
        return added, False, -float("inf")

    def prune(self, dual, crew_dual):
        if self.all_columns or len(self.pool) <= self.pool_cap:
            return
        mandatory = {i for i, r in enumerate(self.pool)
                     if len(r) == 1 or (self.last_values is not None and r in self.last_values)}
        for route in self.best or []:
            if route in self.index:
                mandatory.add(self.index[route])
        ranked = sorted((self.costs[i] - sum(dual[t] for t in r) - crew_dual,
                         self.rng.random(), i)
                        for i, r in enumerate(self.pool) if i not in mandatory)
        room = max(0, self.pool_cap - len(mandatory))
        core = int(room * 0.88)
        keep = mandatory | {i for _, _, i in ranked[:core]}
        rest = [i for _, _, i in ranked[core:]]
        if rest:
            keep.update(self.rng.sample(rest, min(room - core, len(rest))))
        self.pool = [r for i, r in enumerate(self.pool) if i in keep]
        self.costs = [c for i, c in enumerate(self.costs) if i in keep]
        self.index = {r: i for i, r in enumerate(self.pool)}

    def run(self):
        if self.n == 0:
            self.best = []
            self.best_cost = 0.0
            return
        self.prepare()
        if self.best is not None and self.nonnegative:
            return
        if self.matching():
            return
        if self.best is None:
            self.beam(min(self.deadline, time.perf_counter() + self.budget * 0.055))
        if np is None:
            while self.remaining() > 0:
                self.beam(min(self.deadline, time.perf_counter() + self.budget * 0.12), True)
            return
        self.enumerate_routes(min(self.deadline, time.perf_counter() + self.budget * 0.04))
        if self.all_columns:
            self.master(self.remaining() * 0.45, integer=True)
        if self.best is not None and self.best_cost <= self.lower + 1e-7 * self.scale:
            return
        self.build_arc_model()
        if self.best is None:
            self.solve_arcs(self.remaining() * 0.30, feasibility=True)
        if self.remaining() > 0 and self.solve_arcs(self.remaining() * 0.18):
            return
        dual = [0.0] * self.n
        crew_dual = -float(4 * self.n + 1)
        next_integer = time.perf_counter()
        safety_at = self.started + self.budget * 0.62
        last_lp = None
        dirty = True
        while self.remaining() > max(0.0001, self.budget * 0.0003):
            if self.best is not None and self.best_cost <= self.lower + 1e-7 * self.scale:
                break
            remaining = self.remaining()
            quantum = min(remaining * 0.55, self.budget / (5.0 + math.log1p(self.budget)))
            if self.best is None and time.perf_counter() >= safety_at:
                self.solve_arcs(self.remaining() * 0.58, feasibility=True)
                safety_at = time.perf_counter() + self.remaining() * 0.45
            if self.enumeration is not None and not self.all_columns:
                self.enumerate_routes(min(self.deadline, time.perf_counter() + quantum * 0.25))
                dirty = True
            if dirty:
                last_lp = self.master(quantum * 0.40)
                if last_lp is not None:
                    dual, crew_dual = last_lp[0], last_lp[1]
                dirty = False
            if self.all_columns:
                if last_lp is not None:
                    self.lower = max(self.lower, last_lp[2] * self.scale)
                self.master(self.remaining() * 0.90, integer=True)
                continue
            price_stop = min(self.deadline, time.perf_counter() + quantum * 0.52)
            added, complete, reduced = self.price(dual, crew_dual, price_stop)
            if last_lp is not None and complete and self.dag and math.isfinite(reduced):
                # Lowering the crew-count dual converts the negative-price error into a valid lower bound of the original problem.
                bound = (sum(dual) + self.k * (crew_dual + min(0.0, reduced))) * self.scale
                self.lower = max(self.lower, bound)
            dirty = added > 0
            if (time.perf_counter() >= next_integer or not added) and self.remaining() > 0:
                self.master(min(self.remaining() * 0.40, quantum), integer=True)
                next_integer = time.perf_counter() + quantum * 0.40
            if not added or not complete:
                age = (time.perf_counter() - self.last_improvement) / max(self.budget, 1e-9)
                labels = max(2, int(2 + math.log1p(self.budget) * (0.5 + min(1.0, age))))
                stop = min(self.deadline, time.perf_counter() + quantum * 0.65)
                extra, _, _ = self.price(dual, crew_dual, stop, labels=labels)
                if extra == 0 and self.remaining() > 0:
                    stop = min(self.deadline, time.perf_counter() + quantum * 0.45)
                    extra, _, _ = self.price(dual, crew_dual, stop, noisy=True, labels=labels)
                dirty = dirty or extra > 0
            if len(self.pool) > self.pool_cap:
                self.prune(dual, crew_dual)
                dirty = True
            if not self.dag and self.remaining() > 0:
                if self.solve_arcs(min(quantum, self.remaining() * 0.40)):
                    break
            if last_lp is None and not dirty and self.remaining() > 0:
                self.beam(min(self.deadline, time.perf_counter() + quantum), True)
                dirty = True

    def answer(self):
        if self.best is not None:
            return _output(self.best, self.k)
        # The final reserved budget is only used to complete feasible seeds; unverified task chains are not spliced.
        final_stop = self.started + self.budget * 0.985
        self.deadline = final_stop
        if self.remaining() > 0:
            try:
                if np is not None:
                    if self.arc_model is None:
                        self.build_arc_model()
                    self.solve_arcs(self.remaining() * 0.80, feasibility=True)
                if self.best is None and self.remaining() > 0:
                    self.beam(final_stop, True)
            except Exception:
                pass
        return _output(self.best if self.best is not None else [], max(0, self.k))


def solve(instance: dict, time_limit_s: float) -> dict:
    started = time.perf_counter()
    search = None
    try:
        search = _Search(instance, float(time_limit_s), started)
        search.run()
    except Exception:
        pass
    if search is not None:
        return search.answer()
    # Non-contract input also never raises an exception to the caller.
    try:
        k = max(0, int(instance.get("max_crews", 0)))
    except Exception:
        k = 0
    return _output([], k)
