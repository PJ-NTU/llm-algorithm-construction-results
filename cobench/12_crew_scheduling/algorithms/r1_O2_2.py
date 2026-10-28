"""Full-path column generation with duty-constrained integer recombination
Finds rosters with a duty-start propagation model, generates feasible task chains via shortest-path pricing, and then performs set-partitioning integer recombination within the wall-clock budget.
"""

import math
import random
import time

try:
    import numpy as np
    from scipy.optimize import Bounds, LinearConstraint, linprog, milp
    from scipy.sparse import coo_matrix
except Exception:
    np = None


# diagnosis: the matrix shows no item where the champion lags; instances 3, 6 and 8 only show larger differences between members.
# The short-segment local neighbourhood of the champion may miss coordinated recombination of multiple paths; the optimality gap cannot be inferred from scores.
# This implementation uses full-path integer recombination and a duty-constrained model, not the destroy-repair matching and tail-swap framework.


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
        self.fallback = [[i] for i in range(self.n)]
        self.fallback_key = (self.n + 1, float("inf"))

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
        cost = 0.0
        for route in routes:
            if not self.valid_route(route):
                return False
            for t in route:
                if not 0 <= t < self.n or t in seen:
                    return False
                seen.add(t)
            cost += sum(float(self.c[u][v]) for u, v in zip(route, route[1:]))
        if len(seen) != self.n:
            return False
        key = (max(0, len(routes) - self.k), cost)
        if key < self.fallback_key:
            self.fallback_key = key
            self.fallback = routes
        if len(routes) <= self.k and cost < self.best_cost - 1e-9:
            self.best = routes
            self.best_cost = cost
            self.last_improvement = time.perf_counter()
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
        for route in routes:
            self.add(route)
            if fragments:
                for a in range(len(route)):
                    if self.remaining() <= 0:
                        return
                    for b in range(a + 1, len(route) + 1):
                        self.add(route[a:b])

    def prepare(self):
        n = self.n
        self.order = sorted(range(n), key=lambda i: (self.s[i], self.f[i], i))
        self.rank = [0] * n
        for a, i in enumerate(self.order):
            self.rank[i] = a
        self.edges = []
        self.adj = [[] for _ in range(n)]
        self.allowed = [set() for _ in range(n)]
        self.dag = True
        self.nonnegative = True
        for i in range(n):
            for j in range(n):
                if (i != j and self.v[i][j] == 1 and self.f[i] <= self.s[j]
                        and self.f[j] - self.s[i] <= self.limit):
                    w = float(self.c[i][j])
                    self.scale = max(self.scale, abs(w))
                    self.nonnegative = self.nonnegative and w >= 0
                    self.edges.append((i, j, w))
                    self.allowed[i].add(j)
                    if self.rank[i] >= self.rank[j]:
                        self.dag = False
        for e, (i, j, w) in enumerate(self.edges):
            if self.rank[i] < self.rank[j]:
                self.adj[i].append((j, w / self.scale, e))
        self.width = max(2, int((1.0 + math.log1p(self.budget)) * math.sqrt(n + 1)))
        self.batch = max(1, int(math.ceil(math.log2(2.0 + self.budget))))
        self.pool_cap = max(3 * n, int(n * (4 + 3 * math.log1p(self.budget))
                                      * (1 + math.sqrt(n / max(1, self.k)))))
        for i in range(n):
            self.add((i,))
        if self.k >= n:
            self.remember([(i,) for i in range(n)])

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
        lows = [(x - self.limit - offset) / unit for x in self.f]
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
        # Artificial columns are only used to find directions while columns are missing; rosters exceeding the crew count never update the best feasible solution.
        c = np.array(self.costs + [float(4 * n + 1)])
        duration = min(allowance, self.remaining()) * 0.90
        if duration <= 0:
            return None
        try:
            if integer:
                result = milp(c, integrality=np.array([1] * p + [0], dtype=np.uint8),
                              bounds=Bounds(np.zeros(p + 1), np.array([1.0] * p + [float(n)])),
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
        if not integer and result.status == 0:
            return (list(result.eqlin.marginals), float(result.ineqlin.marginals[0]),
                    float(result.fun), result.x)
        return None

    def price(self, dual, crew_dual, stop, noisy=False):
        n, order = self.n, self.order
        adjusted = list(dual)
        perturb = None
        if noisy:
            age = (time.perf_counter() - self.last_improvement) / max(self.budget, 1e-9)
            amplitude = (0.05 + min(1.0, age)) * (1 + abs(crew_dual) / max(1, n))
            adjusted = [x + amplitude * (self.rng.random() - 0.5) for x in adjusted]
            perturb = [amplitude * (self.rng.random() - 0.5) for _ in self.edges]
        added, lowest = 0, float("inf")
        sources = order[self.price_cursor:] + order[:self.price_cursor]
        for first in sources:
            pos = self.rank[first]
            if time.perf_counter() >= stop:
                return added, False, lowest
            dist = [float("inf")] * n
            pred = [-1] * n
            dist[first] = -adjusted[first]
            start_time = self.s[first]
            for u in order[pos:]:
                if time.perf_counter() >= stop:
                    return added, False, lowest
                du = dist[u]
                if not math.isfinite(du):
                    continue
                for v, w, e in self.adj[u]:
                    if self.f[v] - start_time > self.limit:
                        continue
                    candidate = du + w - adjusted[v]
                    if noisy:
                        candidate += perturb[e]
                    if candidate < dist[v] - 1e-12:
                        dist[v], pred[v] = candidate, u
            self.price_cursor = (pos + 1) % n
            endpoints = [(dist[u] - crew_dual, u) for u in order[pos:]
                         if math.isfinite(dist[u])]
            endpoints.sort()
            if endpoints:
                lowest = min(lowest, endpoints[0][0])
            count = 0
            for reduced, last in endpoints:
                if not noisy and reduced >= -1e-7:
                    break
                route = []
                u = last
                while u >= 0:
                    route.append(u)
                    u = pred[u]
                route.reverse()
                new = self.add(route)
                added += new
                count += new
                if count >= self.batch or time.perf_counter() >= stop:
                    break
        return added, True, lowest

    def prune(self, dual, crew_dual):
        if len(self.pool) <= self.pool_cap:
            return
        mandatory = {i for i, r in enumerate(self.pool) if len(r) == 1}
        for route in self.best or []:
            if route in self.index:
                mandatory.add(self.index[route])
        ranked = sorted((self.costs[i] - sum(dual[t] for t in r) - crew_dual, i)
                        for i, r in enumerate(self.pool) if i not in mandatory)
        keep = mandatory | {i for _, i in ranked[:max(0, self.pool_cap - len(mandatory))]}
        self.pool = [r for i, r in enumerate(self.pool) if i in keep]
        self.costs = [c for i, c in enumerate(self.costs) if i in keep]
        self.index = {r: i for i, r in enumerate(self.pool)}

    def run(self):
        if self.n == 0:
            self.best = []
            return
        self.prepare()
        if self.best is not None and self.nonnegative:
            return
        self.beam(min(self.deadline, self.started + self.budget * 0.08))
        if np is None:
            while self.remaining() > 0:
                self.beam(min(self.deadline, time.perf_counter() + self.budget * 0.12), True)
            return
        self.build_arc_model()
        if self.best is None:
            self.solve_arcs(self.remaining() * 0.38, feasibility=True)
        if self.remaining() > 0 and self.solve_arcs(self.remaining() * 0.22):
            return
        dual = [0.0] * self.n
        crew_dual = -float(4 * self.n + 1)
        next_integer = time.perf_counter()
        safety_at = self.started + self.budget * 0.68
        last_lp = None
        dirty = True
        while self.remaining() > max(0.001, self.budget * 0.001):
            if self.best is not None and self.best_cost <= self.lower + 1e-7 * self.scale:
                break
            if self.best is None and time.perf_counter() >= safety_at:
                self.solve_arcs(self.remaining() * 0.72, feasibility=True)
                safety_at = time.perf_counter() + self.remaining() * 0.55
            if dirty:
                last_lp = self.master(min(self.remaining() * 0.30, self.budget * 0.10))
                if last_lp is not None:
                    dual, crew_dual = last_lp[0], last_lp[1]
                dirty = False
            price_stop = min(self.deadline, time.perf_counter()
                             + min(self.remaining() * 0.25, self.budget * 0.10))
            added, complete, reduced = self.price(dual, crew_dual, price_stop)
            if last_lp is not None and complete and reduced >= -1e-7 and self.dag:
                self.lower = max(self.lower, last_lp[2] * self.scale)
            dirty = added > 0
            if (time.perf_counter() >= next_integer or not added) and self.remaining() > 0:
                allowance = min(self.remaining() * 0.42, self.budget * 0.15)
                self.master(allowance, integer=True)
                next_integer = time.perf_counter() + self.budget * 0.06
            if self.best is not None and self.best_cost <= self.lower + 1e-7 * self.scale:
                break
            if not added or not complete:
                stop = min(self.deadline, time.perf_counter()
                           + min(self.remaining() * 0.25, self.budget * 0.10))
                extra, _, _ = self.price(dual, crew_dual, stop, noisy=True)
                dirty = dirty or extra > 0
            if len(self.pool) > self.pool_cap:
                self.prune(dual, crew_dual)
                dirty = True
            if not self.dag and self.remaining() > 0:
                if self.solve_arcs(min(self.budget * 0.12, self.remaining() * 0.35)):
                    break
            if last_lp is None and not dirty and self.remaining() > 0:
                self.beam(min(self.deadline, time.perf_counter() + self.budget * 0.10), True)
                dirty = True

    def answer(self):
        if self.best is not None:
            return _output(self.best, self.k)
        # This point can only be reached when no feasible roster was found or the input is infeasible; keep full task coverage.
        routes = [list(r) for r in self.fallback]
        if self.k > 0 and len(routes) > self.k:
            routes = routes[:self.k - 1] + [[t for r in routes[self.k - 1:] for t in r]]
        return _output(routes, max(0, self.k))


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
