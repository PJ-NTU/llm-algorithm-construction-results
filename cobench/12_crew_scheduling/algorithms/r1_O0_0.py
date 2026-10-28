"""Hybrid matching cutting-plane and path-neighbourhood search
Builds paths with min-cost matching, repairs them with duty-span cuts and integer programming, and performs path exchanges and local exact re-sequencing within the wall-clock budget.
"""

import math
import random
import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linear_sum_assignment, milp
from scipy.sparse import coo_matrix


class _Search:
    def __init__(self, data, budget, started):
        self.n = int(data["num_tasks"])
        self.k = int(data["max_crews"])
        self.limit = float(data["max_duty_time"])
        self.s = np.asarray(data["start_times"], dtype=float)
        self.f = np.asarray(data["finish_times"], dtype=float)
        self.c = np.asarray(data["transition_costs"], dtype=float)
        self.budget = max(0.0, float(budget))
        self.started = started
        self.deadline = started + self.budget - min(0.12, 0.02 * self.budget)
        self.rng = random.Random(7919 * self.n + 104729 * self.k)
        self.best = None
        self.best_cost = math.inf
        self.current = [[i] for i in range(self.n)]
        self.cuts = set()
        self.last_gain = started
        self.proven = False
        self.allowed = ((np.asarray(data["valid_transitions"]) == 1)
                        & (self.f[:, None] <= self.s[None, :])
                        & (self.f[None, :] - self.s[:, None] <= self.limit))
        np.fill_diagonal(self.allowed, False)
        self.order = sorted(range(self.n), key=lambda i: (self.s[i], self.f[i], i))
        self.ei, self.ej = np.nonzero(self.allowed)
        ec = np.abs(self.c[self.ei, self.ej])
        self.scale = max(1.0, float(np.median(ec)) if ec.size else 1.0)
        self.penalty = 1.0 + 2.0 * self.n * (float(np.max(ec)) if ec.size else 1.0)
        self.lower_bound = -math.inf

    def remaining(self):
        return self.deadline - time.perf_counter()

    def cost(self, routes):
        return sum(float(self.c[a, b]) for r in routes for a, b in zip(r, r[1:]))

    def route_ok(self, r):
        return (not r or (self.f[r[-1]] - self.s[r[0]] <= self.limit
                         and all(self.allowed[a, b] for a, b in zip(r, r[1:]))))

    def accept(self, routes):
        rr = [r for r in routes if r]
        if len(rr) > self.k or any(not self.route_ok(r) for r in rr):
            return False
        flat = [i for r in rr for i in r]
        if len(flat) != self.n or len(set(flat)) != self.n:
            return False
        value = self.cost(rr)
        if self.best is None or value < self.best_cost:
            self.best = [r[:] for r in rr]
            self.best_cost = value
            self.last_gain = time.perf_counter()
            if value <= self.lower_bound + 1e-9 * max(1.0, abs(value)):
                self.proven = True
            return True
        return False

    def remember(self, routes):
        rr = [r for r in routes if r]
        if not all(self.route_ok(r) for r in rr):
            return
        old = (max(0, len(self.current) - self.k), self.cost(self.current))
        new = (max(0, len(rr) - self.k), self.cost(rr))
        if new < old:
            self.current = [r[:] for r in rr]
        self.accept(rr)

    def greedy(self, randomized=False):
        routes = []
        for i in self.order:
            candidates = []
            for p, r in enumerate(routes):
                if self.allowed[r[-1], i] and self.f[i] - self.s[r[0]] <= self.limit:
                    edge = float(self.c[r[-1], i]) / self.scale
                    age = (self.s[i] - self.s[r[0]]) / max(1.0, self.limit)
                    gap = (self.s[i] - self.f[r[-1]]) / max(1.0, self.limit)
                    score = edge + age + gap
                    if randomized:
                        score = (self.rng.random() * edge
                                 + self.rng.uniform(-2.0, 2.0) * age
                                 + self.rng.uniform(-1.0, 1.0) * gap
                                 + self.rng.random())
                    candidates.append((score, p))
            if candidates:
                routes[min(candidates)[1]].append(i)
            else:
                routes.append([i])
        self.remember(routes)
        return routes

    def decode(self, nodes, pairs):
        successor = dict(pairs)
        predecessor = {j: i for i, j in pairs}
        used = set()
        routes = []
        for root in nodes:
            if root in predecessor or root in used:
                continue
            r = []
            i = root
            while i not in used:
                used.add(i)
                r.append(i)
                if i not in successor:
                    break
                i = successor[i]
            routes.append(r)
        cycles = []
        for root in nodes:
            if root in used:
                continue
            cycle = []
            i = root
            while i not in used:
                used.add(i)
                cycle.append(i)
                i = successor.get(i, root)
            if cycle:
                cycles.append(cycle)
        return routes, cycles

    def add_cuts(self, routes, cycles):
        for r in cycles:
            edges = tuple(sorted(zip(r, r[1:] + r[:1])))
            self.cuts.add(edges)
        for r in routes:
            left = 0
            for right in range(1, len(r)):
                if self.f[r[right]] - self.s[r[left]] <= self.limit:
                    continue
                while left + 1 < right and self.f[r[right]] - self.s[r[left + 1]] > self.limit:
                    left += 1
                if self.f[r[right - 1]] - self.s[r[left]] <= self.limit:
                    self.cuts.add(tuple(sorted(zip(r[left:right], r[left + 1:right + 1]))))

    def split_and_merge(self, routes, cycles=()):
        parts = []
        for r in cycles:
            cut = max(range(len(r)), key=lambda j: self.c[r[j], r[(j + 1) % len(r)]])
            routes = routes + [r[cut + 1:] + r[:cut + 1]]
        for r in routes:
            # Dynamic programming: first minimise the number of segments, then the connection cost.
            m = len(r)
            counts = [m + 1] * (m + 1)
            values = [math.inf] * (m + 1)
            prev = [-1] * (m + 1)
            prefix = [0.0] * m
            for j in range(1, m):
                prefix[j] = prefix[j - 1] + float(self.c[r[j - 1], r[j]])
            counts[0], values[0] = 0, 0.0
            for j in range(1, m + 1):
                for i in range(j - 1, -1, -1):
                    if self.f[r[j - 1]] - self.s[r[i]] > self.limit:
                        break
                    candidate = (counts[i] + 1, values[i] + prefix[j - 1] - prefix[i])
                    if candidate < (counts[j], values[j]):
                        counts[j], values[j] = candidate
                        prev[j] = i
            if prev[m] < 0:
                parts.extend([[i] for i in r])
            else:
                j = m
                while j:
                    i = prev[j]
                    parts.append(r[i:j])
                    j = i
        # Merging only changes the junction, keeping the span of each path feasible.
        while len(parts) > 1 and self.remaining() > 0:
            chosen = None
            for p, a in enumerate(parts):
                if time.perf_counter() >= self.deadline:
                    break
                for q, b in enumerate(parts):
                    if p == q or not self.allowed[a[-1], b[0]]:
                        continue
                    if self.f[b[-1]] - self.s[a[0]] > self.limit:
                        continue
                    value = float(self.c[a[-1], b[0]])
                    if len(parts) <= self.k and value >= 0:
                        continue
                    if chosen is None or value < chosen[0]:
                        chosen = (value, p, q)
            if chosen is None:
                break
            _, p, q = chosen
            joined = parts[p] + parts[q]
            parts = [r for j, r in enumerate(parts) if j not in (p, q)] + [joined]
        spare = self.k - len(parts)
        if spare > 0:
            breaks = sorted(((float(self.c[a, b]), p, j)
                             for p, r in enumerate(parts)
                             for j, (a, b) in enumerate(zip(r, r[1:]))
                             if self.c[a, b] > 0), reverse=True)
            selected = {(p, j) for _, p, j in breaks[:spare]}
            expanded = []
            for p, r in enumerate(parts):
                start = 0
                for j in range(len(r) - 1):
                    if (p, j) in selected:
                        expanded.append(r[start:j + 1])
                        start = j + 1
                expanded.append(r[start:])
            parts = expanded
        return parts

    def matching(self):
        n = self.n
        mat = np.full((n, n + min(n, max(0, self.k))), np.inf)
        mat[self.ei, self.ej] = self.c[self.ei, self.ej]
        mat[:, n:] = 0.0
        try:
            rows, cols = linear_sum_assignment(mat)
        except (ValueError, MemoryError):
            return
        self.lower_bound = float(mat[rows, cols].sum())
        pairs = [(int(i), int(j)) for i, j in zip(rows, cols) if j < n]
        routes, cycles = self.decode(range(n), pairs)
        self.add_cuts(routes, cycles)
        if not cycles and all(self.route_ok(r) for r in routes):
            self.accept(routes)
        if not self.proven:
            self.remember(self.split_and_merge(routes, cycles))

    def local_search(self, stop):
        routes = [r[:] for r in self.current if r]
        if self.best is not None and len(routes) > self.k:
            routes = [r[:] for r in self.best]
        while time.perf_counter() < stop and not self.proven:
            count = len(routes)
            if count == 0:
                break
            p = self.rng.randrange(count)
            can_open = count < min(self.k, self.n)
            q = self.rng.randrange(count + int(can_open))
            if p == q:
                continue
            a = routes[p]
            b = routes[q] if q < count else []
            progress = min(1.0, (time.perf_counter() - self.started) / max(self.budget, 1e-9))
            operation = self.rng.random()
            if operation < 0.45:
                i = self.rng.randrange(len(a) + 1)
                j = self.rng.randrange(len(b) + 1)
                aa, bb = a[:i] + b[j:], b[:j] + a[i:]
            elif operation < 0.85 or not b:
                i = self.rng.randrange(len(a))
                width = min(len(a) - i, max(1, int(1 + math.log1p(self.budget) * (1 + progress))))
                end = i + self.rng.randint(1, width)
                j = self.rng.randrange(len(b) + 1)
                aa, bb = a[:i] + a[end:], b[:j] + a[i:end] + b[j:]
            else:
                i, j = self.rng.randrange(len(a)), self.rng.randrange(len(b))
                aa, bb = a[:], b[:]
                aa[i], bb[j] = bb[j], aa[i]
            if not self.route_ok(aa) or not self.route_ok(bb):
                continue
            change_count = int(bool(aa)) + int(bool(bb)) - 1 - int(bool(b))
            delta = self.cost([aa, bb]) - self.cost([a, b])
            delta += self.penalty * (max(0, count + change_count - self.k) - max(0, count - self.k))
            temperature = self.scale * (0.02 + (1.0 - progress) ** 2)
            if delta > 0 and self.rng.random() >= math.exp(-min(700.0, delta / temperature)):
                continue
            routes = [r for h, r in enumerate(routes) if h != p and h != q]
            if aa:
                routes.append(aa)
            if bb:
                routes.append(bb)
            self.accept(routes)
        self.current = routes

    def optimize(self, nodes, crews, duration_model, seconds, global_problem=False):
        if seconds <= 0 or self.remaining() <= 0:
            return None
        ids = np.asarray(nodes, dtype=int)
        n = len(nodes)
        if n == 0:
            return []
        if global_problem:
            ii, jj = self.ei, self.ej
        else:
            ii, jj = np.nonzero(self.allowed[np.ix_(ids, ids)])
        full_edges = True
        m = len(ii)
        # For large graphs, enlarge the candidate edges step by step; edges of the current feasible solution are always kept.
        edge_budget = max(n * 4, int(1200 * max(0.1, self.budget)))
        if global_problem and m > edge_budget:
            progress = min(1.0, (time.perf_counter() - self.started) / max(self.budget, 1e-9))
            width = min(n, max(2, int(math.sqrt(max(0.1, self.budget))
                                     * math.log2(n + 2) * (1 + progress))))
            keep = set()
            for i in range(n):
                if self.remaining() <= 0:
                    return None
                dest = np.flatnonzero(self.allowed[i])
                if len(dest) > width:
                    vals = self.c[i, dest]
                    dest = dest[np.argpartition(vals, width - 1)[:width]]
                keep.update((i, int(j)) for j in dest)
                src = np.flatnonzero(self.allowed[:, i])
                if len(src) > width:
                    vals = self.c[src, i]
                    src = src[np.argpartition(vals, width - 1)[:width]]
                keep.update((int(j), i) for j in src)
            for r in (self.best or self.current):
                keep.update(zip(r, r[1:]))
            if keep:
                ii, jj = np.asarray(sorted(keep), dtype=int).T
                full_edges = len(ii) == m
                m = len(ii)
        if not m:
            return [[int(i)] for i in ids] if n <= crews else None
        oi, oj = ids[ii], ids[jj]
        raw_cost = self.c[oi, oj]
        c = raw_cost / self.scale
        nvars = m + (n if duration_model else 0)
        lower = np.zeros(nvars)
        upper = np.ones(nvars)
        integrality = np.zeros(nvars, dtype=np.uint8)
        integrality[:m] = 1
        objective = np.zeros(nvars)
        objective[:m] = c
        indices = np.arange(m)
        row = [ii, n + jj, np.full(m, 2 * n, dtype=int)]
        col = [indices, indices, indices]
        val = [np.ones(m), np.ones(m), np.ones(m)]
        lb = [-np.inf] * (2 * n) + [max(0, n - crews)]
        ub = [1.0] * (2 * n) + [np.inf]
        edge_index = {(int(a), int(b)): h for h, (a, b) in enumerate(zip(oi, oj))}
        for cut in tuple(self.cuts):
            positions = [edge_index.get(e, -1) for e in cut]
            if positions and min(positions) >= 0:
                r = len(lb)
                row.append(np.full(len(positions), r, dtype=int))
                col.append(np.asarray(positions, dtype=int))
                val.append(np.ones(len(positions)))
                lb.append(-np.inf)
                ub.append(len(positions) - 1.0)
        if global_problem and self.best is not None:
            r = len(lb)
            row.append(np.full(m, r, dtype=int))
            col.append(indices)
            val.append(c)
            lb.append(-np.inf)
            ub.append(self.best_cost / self.scale + 1e-9)
        if duration_model:
            unit = max(abs(self.limit), 1.0)
            age_ub = np.maximum(0.0, (self.limit - (self.f[ids] - self.s[ids])) / unit)
            upper[m:] = age_ub
            big_m = age_ub[ii] + (self.s[oj] - self.s[oi]) / unit
            r = np.arange(m) + len(lb)
            row.extend([r, r, r])
            col.extend([indices, m + ii, m + jj])
            val.extend([big_m, np.ones(m), -np.ones(m)])
            lb.extend([-np.inf] * m)
            ub.extend(age_ub[ii].tolist())
        if self.remaining() <= 0:
            return None
        matrix = coo_matrix((np.concatenate(val), (np.concatenate(row), np.concatenate(col))),
                            shape=(len(lb), nvars)).tocsc()
        allowed_time = min(seconds, self.remaining())
        if allowed_time <= 0:
            return None
        try:
            result = milp(objective, integrality=integrality, bounds=Bounds(lower, upper),
                          constraints=LinearConstraint(matrix, np.asarray(lb), np.asarray(ub)),
                          options={"time_limit": allowed_time, "mip_rel_gap": 0.0,
                                   "presolve": True, "disp": False})
        except Exception:
            return None
        if result.x is None:
            return None
        chosen = np.flatnonzero(result.x[:m] > 0.5)
        pairs = [(int(oi[h]), int(oj[h])) for h in chosen]
        routes, cycles = self.decode(nodes, pairs)
        self.add_cuts(routes, cycles)
        valid = not cycles and len(routes) <= crews and all(self.route_ok(r) for r in routes)
        if valid:
            if global_problem:
                self.accept(routes)
                if full_edges and result.status == 0:
                    self.proven = True
            return routes
        if global_problem:
            self.remember(self.split_and_merge(routes, cycles))
        return None

    def neighborhood(self, seconds):
        base = self.best if self.best is not None else self.current
        if len(base) < 2:
            return
        progress = min(1.0, (time.perf_counter() - self.started) / max(self.budget, 1e-9))
        stagnant = (time.perf_counter() - self.last_gain) / max(self.budget, 1e-9)
        width = min(len(base), max(2, int(1 + math.log1p(self.budget)
                                         * (1 + progress + stagnant))))
        seed = self.rng.randrange(len(base))
        chosen = [seed]
        rest = [i for i in range(len(base)) if i != seed]
        # Prefer paths whose time ranges overlap, while keeping random diversity.
        a = base[seed]
        rest.sort(key=lambda p: (max(0.0, float(self.s[base[p][0]] - self.f[a[-1]]),
                                             float(self.s[a[0]] - self.f[base[p][-1]]))
                                 / max(1.0, self.limit) + self.rng.random()))
        chosen.extend(rest[:width - 1])
        selected = set(chosen)
        fixed = [r[:] for i, r in enumerate(base) if i not in selected]
        nodes = sorted(i for p in chosen for i in base[p])
        crews = min(len(nodes), self.k - len(fixed))
        if crews < 1:
            crews = max(1, len(chosen) - 1)
        answer = self.optimize(nodes, crews, True, seconds)
        if answer is not None:
            candidate = fixed + answer
            self.remember(candidate)
            if len(candidate) <= self.k and self.cost(candidate) <= self.cost(self.current):
                self.current = [r[:] for r in candidate]

    def run(self):
        if not self.n:
            self.best = []
            return
        self.greedy()
        if self.remaining() > 0:
            self.matching()
        if self.proven:
            return
        initial_stop = min(self.deadline, self.started + 0.07 * self.budget)
        while time.perf_counter() < initial_stop:
            self.greedy(True)
            self.local_search(min(initial_stop, time.perf_counter() + self.budget / (20 + self.n)))
        # While no feasible solution exists yet, first give the model with full span constraints a larger time slice.
        if self.best is None and self.remaining() > 0:
            self.optimize(list(range(self.n)), self.k, True, self.remaining() * 0.58, True)
        while self.remaining() > 0 and not self.proven:
            remaining = self.remaining()
            progress = min(1.0, (time.perf_counter() - self.started) / max(self.budget, 1e-9))
            local_seconds = min(remaining * 0.12, self.budget / (12 + math.sqrt(max(1, self.n))))
            self.local_search(min(self.deadline, time.perf_counter() + local_seconds))
            if self.remaining() <= 0 or self.proven:
                break
            remaining = self.remaining()
            slice_seconds = min(remaining * 0.72,
                                self.budget * (0.06 + 0.20 * progress)
                                + remaining / (4 + math.log2(self.n + 2)))
            if self.best is not None and self.rng.random() < 0.4 + 0.25 * progress:
                self.neighborhood(slice_seconds)
            else:
                full_duration = self.best is None or self.rng.random() < progress
                self.optimize(list(range(self.n)), self.k, full_duration, slice_seconds, True)
            if self.best is not None and self.rng.random() < 0.3:
                self.current = [r[:] for r in self.best]
            elif self.best is None and self.remaining() > 0:
                self.greedy(True)

    def answer(self):
        routes = self.best if self.best is not None else self.current
        routes = [list(map(int, r)) for r in routes if r]
        # For infeasible input or when no feasible solution is found within the budget, still keep full coverage and the interface shape.
        # The best feasible solution returned normally has always had all constraints verified by accept.
        if len(routes) > self.k and self.k > 0:
            head = routes[:self.k]
            for r in routes[self.k:]:
                p = min(range(self.k), key=lambda j: len(head[j]))
                head[p].extend(r)
            routes = head
        routes.extend([[] for _ in range(max(0, self.k - len(routes)))])
        return {"crew_routes": routes,
                "start_tasks": [r[0] if r else -1 for r in routes],
                "end_tasks": [r[-1] if r else -1 for r in routes]}


def solve(instance: dict, time_limit_s: float) -> dict:
    started = time.perf_counter()
    if int(instance["num_tasks"]) == 0:
        k = max(0, int(instance["max_crews"]))
        return {"crew_routes": [[] for _ in range(k)],
                "start_tasks": [-1] * k, "end_tasks": [-1] * k}
    search = _Search(instance, time_limit_s, started)
    try:
        search.run()
    except Exception:
        # A failure of the external numerical solver does not discard the verified best feasible solution.
        pass
    return search.answer()
