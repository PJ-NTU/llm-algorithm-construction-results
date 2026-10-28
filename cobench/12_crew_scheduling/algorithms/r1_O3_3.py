"""Duty-conflict learning with path-segment rematching
Combines min-cost matching, overtime path cuts, duty-feasible augmenting repair and segment large-neighbourhood search, continually improving on the wall clock and saving the best feasible solution.
"""
import heapq
import math
import random
import time

try:
    import numpy as np
    from scipy.optimize import Bounds, LinearConstraint, linear_sum_assignment, milp
    from scipy.sparse import coo_matrix
except Exception:
    np = None
    milp = None


def _format(routes, k):
    out = [list(r) for r in routes if r]
    out += [[] for _ in range(max(0, k - len(out)))]
    return {"crew_routes": out, "start_tasks": [r[0] if r else -1 for r in out],
            "end_tasks": [r[-1] if r else -1 for r in out]}


def _flow(n, k, edges, stop):
    """Variable-cardinality min-cost matching; negative-cost connections are not discarded when the crew limit is reached."""
    size = 2 * n + 2
    source, sink = 2 * n, 2 * n + 1
    adj = [[] for _ in range(size)]
    to, cap, cost = [], [], []

    def add(a, b, w):
        q = len(to)
        adj[a].append(q)
        adj[b].append(q + 1)
        to.extend((b, a))
        cap.extend((1, 0))
        cost.extend((w, -w))
        return q

    for i in range(n):
        add(source, i, 0.0)
        add(n + i, sink, 0.0)
    shift = max(0.0, -min((w for i, j, w in edges), default=0.0))
    refs = [(add(i, n + j, w + shift), i, j) for i, j, w in edges]
    potential = [0.0] * size
    flow = 0
    complete = False
    while time.monotonic() < stop:
        dist = [math.inf] * size
        previous = [-1] * size
        dist[source] = 0.0
        queue = [(0.0, source)]
        aborted = False
        while queue:
            if time.monotonic() >= stop:
                aborted = True
                break
            d, u = heapq.heappop(queue)
            if d > dist[u] + 1e-11:
                continue
            for e in adj[u]:
                if not cap[e]:
                    continue
                v = to[e]
                nd = d + cost[e] + potential[u] - potential[v]
                if nd < dist[v] - 1e-11:
                    dist[v] = nd
                    previous[v] = e
                    heapq.heappush(queue, (nd, v))
        if aborted:
            break
        if dist[sink] == math.inf:
            complete = True
            break
        increment = dist[sink] + potential[sink] - potential[source] - shift
        if flow >= max(0, n - k) and increment >= -1e-10:
            complete = True
            break
        ds = dist[sink]
        for v in range(size):
            potential[v] += dist[v] if dist[v] < math.inf else ds
        v = sink
        while v != source:
            e = previous[v]
            cap[e] = 0
            cap[e ^ 1] = 1
            v = to[e ^ 1]
        flow += 1
    successor = [-1] * n
    for e, i, j in refs:
        if not cap[e]:
            successor[i] = j
    return successor, complete


class _Search:
    def __init__(self, data, budget, started):
        self.n = int(data["num_tasks"])
        self.k = int(data["max_crews"])
        self.limit = float(data["max_duty_time"])
        self.s = [float(v) for v in data["start_times"]]
        self.f = [float(v) for v in data["finish_times"]]
        self.c = data["transition_costs"]
        self.v = data["valid_transitions"]
        self.budget = max(0.0, budget)
        self.started = started
        self.stop = started + max(0.0, self.budget - min(0.10, self.budget * 0.035))
        self.rng = random.Random(503917)
        self.best = None
        self.best_cost = math.inf
        self.partial = [[i] for i in range(self.n)]
        self.partial_key = (self.n + 1, math.inf)
        self.cuts = set()
        self.penalty = {}
        self.lower = -math.inf
        self.ok = [set() for _ in range(self.n)]
        self.out = [[] for _ in range(self.n)]
        self.inc = [[] for _ in range(self.n)]
        self.order = sorted(range(self.n), key=lambda i: (self.s[i], self.f[i], i))
        self.succ = [-1] * self.n
        self.pred = [-1] * self.n
        self.head = list(range(self.n))
        self.tail = list(range(self.n))
        self.count = self.n
        self.scale = 1.0
        self.depth = min(self.n, max(2, int(2 + math.log1p(self.budget) * math.log2(self.n + 2))))
        self.segment = min(self.n, max(1, int(1 + math.log1p(self.budget) *
                                             math.sqrt(self.n + 1) / math.log2(self.n + 2))))

    def prepare(self):
        total, number = 0.0, 0
        for i in range(self.n):
            for j in range(self.n):
                if i != j and self.v[i][j] == 1 and self.f[i] <= self.s[j] and \
                        self.f[j] - self.s[i] <= self.limit:
                    w = float(self.c[i][j])
                    self.ok[i].add(j)
                    self.out[i].append((w, j))
                    self.inc[j].append((w, i))
                    total += abs(w)
                    number += 1
            self.out[i].sort()
        for row in self.inc:
            row.sort()
        self.scale = max(1e-8, total / max(1, number))

    def cost(self, routes):
        return sum(float(self.c[a][b]) for r in routes for a, b in zip(r, r[1:]))

    def valid(self, routes, crew_cap=True):
        if crew_cap and len(routes) > self.k:
            return False
        seen = set()
        for r in routes:
            if not r:
                continue
            if self.f[r[-1]] - self.s[r[0]] > self.limit:
                return False
            for x in r:
                if x < 0 or x >= self.n or x in seen:
                    return False
                seen.add(x)
            if any(b not in self.ok[a] for a, b in zip(r, r[1:])):
                return False
        return len(seen) == self.n

    def consider(self, routes):
        routes = [list(r) for r in routes if r]
        if not self.valid(routes, False):
            return False
        # Surplus idle crews can be used to cut positive-cost connections; negative-cost connections are kept.
        spare = self.k - len(routes)
        if spare > 0:
            links = sorted(((float(self.c[a][b]), a) for r in routes
                            for a, b in zip(r, r[1:]) if self.c[a][b] > 0), reverse=True)
            cut_at = {a for _, a in links[:spare]}
            split = []
            for r in routes:
                part = []
                for a in r:
                    part.append(a)
                    if a in cut_at:
                        split.append(part)
                        part = []
                if part:
                    split.append(part)
            routes = split
        value = self.cost(routes)
        key = (len(routes), value)
        if key < self.partial_key:
            self.partial, self.partial_key = routes, key
        if len(routes) <= self.k and value < self.best_cost - 1e-10:
            self.best, self.best_cost = [r[:] for r in routes], value
            return True
        return False

    def rebuild(self):
        n = self.n
        pred = [-1] * n
        for i, j in enumerate(self.succ):
            if j >= 0:
                if pred[j] >= 0:
                    return False
                pred[j] = i
        self.pred = pred
        visited = 0
        self.count = 0
        for h in range(n):
            if pred[h] >= 0:
                continue
            self.count += 1
            path, x = [], h
            while x >= 0:
                path.append(x)
                if len(path) > n:
                    return False
                x = self.succ[x]
            t = path[-1]
            visited += len(path)
            for x in path:
                self.head[x], self.tail[x] = h, t
        return visited == n

    def load(self, routes):
        self.succ = [-1] * self.n
        for r in routes:
            for i, j in zip(r, r[1:]):
                self.succ[i] = j
        return self.rebuild()

    def routes(self):
        result = []
        for i in range(self.n):
            if self.pred[i] < 0:
                r, x = [], i
                while x >= 0:
                    r.append(x)
                    x = self.succ[x]
                result.append(r)
        return result

    def link(self, i, j):
        self.succ[i], self.pred[j] = j, i
        h, t = self.head[i], self.tail[j]
        x = h
        while x >= 0:
            self.head[x], self.tail[x] = h, t
            x = self.succ[x]
        self.count -= 1

    def unlink(self, i, j):
        h, t = self.head[i], self.tail[j]
        self.succ[i], self.pred[j] = -1, -1
        x = h
        while x >= 0:
            self.head[x], self.tail[x] = h, i
            x = self.succ[x]
        x = j
        while x >= 0:
            self.head[x], self.tail[x] = j, t
            x = self.succ[x]
        self.count += 1

    def joinable(self, i, j):
        return self.head[i] != self.head[j] and \
            self.f[self.tail[j]] - self.s[self.head[i]] <= self.limit

    def construct(self, mode, noise, stop):
        routes = []
        for t in self.order:
            choices = []
            for z, r in enumerate(routes):
                u = r[-1]
                if t not in self.ok[u] or self.f[t] - self.s[r[0]] > self.limit:
                    continue
                if mode == 0:
                    score = float(self.c[u][t])
                elif mode == 1:
                    score = self.s[r[0]]
                elif mode == 2:
                    score = self.s[t] - self.f[u]
                else:
                    score = -self.s[r[0]]
                score += noise * self.scale * self.rng.random()
                choices.append((score, z))
            if choices:
                routes[min(choices)[1]].append(t)
            else:
                routes.append([t])
            if time.monotonic() >= stop:
                return
        self.consider(routes)
        self.load(routes)
        if self.count > self.k:
            self.repair_state(stop)
        self.consider(self.routes())

    def augment(self, i, seen, depth, noisy, stop):
        if depth > self.depth or time.monotonic() >= stop:
            return False
        choices = self.out[i]
        if noisy:
            choices = choices[:]
            self.rng.shuffle(choices)
        for _, j in choices:
            if time.monotonic() >= stop:
                return False
            if j in seen:
                continue
            seen.add(j)
            p = self.pred[j]
            if p < 0:
                if self.joinable(i, j):
                    self.link(i, j)
                    return True
            elif p != i:
                self.unlink(p, j)
                if self.joinable(i, j):
                    self.link(i, j)
                    if self.augment(p, seen, depth + 1, noisy, stop):
                        return True
                    self.unlink(i, j)
                self.link(p, j)
        return False

    def insertion(self, t, pool):
        best = None
        for w, u in self.inc[t]:
            if u in pool:
                continue
            v = self.succ[u]
            if v >= 0 and v not in self.ok[t]:
                continue
            if v < 0 and self.f[t] - self.s[self.head[u]] > self.limit:
                continue
            d = w if v < 0 else w + self.c[t][v] - self.c[u][v]
            candidate = (d, u, v)
            if best is None or candidate < best:
                best = candidate
        for w, v in self.out[t]:
            if v in pool or self.pred[v] >= 0:
                continue
            if self.f[self.tail[v]] - self.s[t] > self.limit:
                continue
            candidate = (w, -1, v)
            if best is None or candidate < best:
                best = candidate
        return best

    def eliminate(self, stop):
        chains = self.routes()
        self.rng.shuffle(chains)
        chains.sort(key=len)
        for chain in chains:
            if time.monotonic() >= stop:
                return False
            saved = self.succ[:]
            before = self.count
            for a, b in zip(chain, chain[1:]):
                self.unlink(a, b)
            pool = set(chain)
            ejected = set(chain)
            while pool and time.monotonic() < stop:
                # Handle tasks with few insertion positions first.
                t = min(pool, key=lambda x: (len(self.inc[x]) + len(self.out[x]), self.rng.random()))
                choice = self.insertion(t, pool)
                if choice is not None:
                    _, u, v = choice
                    if u >= 0 and v >= 0:
                        self.unlink(u, v)
                    if u >= 0:
                        self.link(u, t)
                    if v >= 0:
                        self.link(t, v)
                    pool.remove(t)
                    continue
                options = []
                for _, u in self.inc[t]:
                    if u in pool:
                        continue
                    v = self.succ[u]
                    if v < 0 or v in ejected:
                        continue
                    z = self.succ[v]
                    if z >= 0 and z not in self.ok[t]:
                        continue
                    end = t if z < 0 else self.tail[u]
                    if self.f[end] - self.s[self.head[u]] > self.limit:
                        continue
                    d = self.c[u][t] - self.c[u][v]
                    if z >= 0:
                        d += self.c[t][z] - self.c[v][z]
                    options.append((d + self.scale * self.rng.random(), u, v, z))
                if not options:
                    break
                _, u, v, z = min(options)
                self.unlink(u, v)
                if z >= 0:
                    self.unlink(v, z)
                self.link(u, t)
                if z >= 0:
                    self.link(t, z)
                pool.remove(t)
                pool.add(v)
                ejected.add(v)
            if not pool and self.count < before:
                return True
            self.succ = saved
            self.rebuild()
        return False

    def repair_state(self, stop):
        progress = True
        while self.count > self.k and progress and time.monotonic() < stop:
            progress = False
            free = [i for i in range(self.n) if self.succ[i] < 0]
            self.rng.shuffle(free)
            free.sort(key=lambda i: len(self.out[i]))
            for i in free:
                if self.count <= self.k or time.monotonic() >= stop:
                    break
                if self.succ[i] < 0 and self.augment(i, set(), 0, True, stop):
                    progress = True
            if self.count > self.k and time.monotonic() < stop:
                progress = self.eliminate(stop) or progress
        self.consider(self.routes())

    def repair_routes(self, routes, stop):
        pieces = []
        for route in routes:
            if not route:
                continue
            current = []
            for t in route:
                if current and self.f[t] - self.s[current[0]] > self.limit:
                    pieces.append(current)
                    current = []
                current.append(t)
            if current:
                pieces.append(current)
        if self.valid(pieces, False):
            self.consider(pieces)
            self.load(pieces)
            self.repair_state(stop)

    def tail_pass(self, stop, perturb=False):
        changed = False
        nodes = list(range(self.n))
        if perturb:
            self.rng.shuffle(nodes)
        for a in nodes:
            if time.monotonic() >= stop:
                break
            choices = self.out[a]
            if perturb:
                choices = choices[:]
                self.rng.shuffle(choices)
            for w, y in choices:
                if time.monotonic() >= stop:
                    return changed
                b, x = self.succ[a], self.pred[y]
                if y == b or self.head[a] == self.head[y]:
                    continue
                if b >= 0 and x >= 0 and b not in self.ok[x]:
                    continue
                d = w
                if b >= 0:
                    d -= self.c[a][b]
                if x >= 0:
                    d -= self.c[x][y]
                if b >= 0 and x >= 0:
                    d += self.c[x][b]
                if not perturb and d >= -1e-10:
                    continue
                if self.f[self.tail[y]] - self.s[self.head[a]] > self.limit:
                    continue
                h = self.head[y] if x >= 0 else b
                t = self.tail[a] if b >= 0 else x
                if h >= 0 and t >= 0 and self.f[t] - self.s[h] > self.limit:
                    continue
                self.succ[a] = y
                if x >= 0:
                    self.succ[x] = b
                self.rebuild()
                changed = True
                if perturb:
                    return True
        return changed

    def segment_pass(self, stop):
        changed = False
        for first in range(self.n):
            if time.monotonic() >= stop:
                break
            last = first
            size = 1
            while size <= self.segment and time.monotonic() < stop:
                p, nxt = self.pred[first], self.succ[last]
                if p >= 0 and nxt >= 0 and nxt not in self.ok[p]:
                    pass
                else:
                    delta = 0.0
                    if p >= 0:
                        delta -= self.c[p][first]
                    if nxt >= 0:
                        delta -= self.c[last][nxt]
                    if p >= 0 and nxt >= 0:
                        delta += self.c[p][nxt]
                    h = self.head[first]
                    move = None
                    # Move the segment into the interior or the tail of another path.
                    for w, u in self.inc[first]:
                        if time.monotonic() >= stop:
                            return changed
                        if self.head[u] == h:
                            continue
                        v = self.succ[u]
                        if v >= 0 and v not in self.ok[last]:
                            continue
                        extra = w + (self.c[last][v] - self.c[u][v] if v >= 0 else 0.0)
                        end = self.tail[u] if v >= 0 else last
                        if delta + extra < -1e-10 and self.f[end] - self.s[self.head[u]] <= self.limit:
                            move = (u, v)
                            break
                    if move is None:
                        for w, v in self.out[last]:
                            if self.pred[v] >= 0 or self.head[v] == h:
                                continue
                            if delta + w < -1e-10 and self.f[self.tail[v]] - self.s[first] <= self.limit:
                                move = (-1, v)
                                break
                    if move is None and delta < -1e-10 and self.count < self.k and (p >= 0 or nxt >= 0):
                        move = (-1, -1)
                    if move is not None:
                        u, v = move
                        if p >= 0:
                            self.succ[p] = nxt
                        self.succ[last] = v
                        if u >= 0:
                            self.succ[u] = first
                        self.rebuild()
                        changed = True
                        break
                last = self.succ[last]
                if last < 0:
                    break
                size += 1
        return changed

    def polish(self, stop):
        while time.monotonic() < stop:
            a = self.tail_pass(stop)
            b = self.segment_pass(stop) if time.monotonic() < stop else False
            self.consider(self.routes())
            if not a and not b:
                break

    def learn(self, route, cycle=False):
        if cycle:
            cut = tuple(sorted(a * self.n + b for a, b in zip(route, route[1:] + route[:1])))
            if cut:
                self.cuts.add(cut)
            return
        left = 0
        for right in range(1, len(route)):
            while left + 1 < right and self.f[route[right]] - self.s[route[left + 1]] > self.limit:
                left += 1
            if self.f[route[right]] - self.s[route[left]] > self.limit:
                cut = tuple(sorted(route[z] * self.n + route[z + 1] for z in range(left, right)))
                if cut not in self.cuts:
                    self.cuts.add(cut)
                    bump = self.scale / max(1, len(cut))
                    for code in cut:
                        self.penalty[code] = self.penalty.get(code, 0.0) + bump

    def decode(self, successor, parts):
        m = len(parts)
        incoming = [False] * m
        for j in successor:
            if j >= 0:
                incoming[j] = True
        seen, result = set(), []
        starts = [i for i in range(m) if not incoming[i]] + [i for i in range(m) if incoming[i]]
        clean = True
        for start in starts:
            if start in seen:
                continue
            chain, route, u = [], [], start
            while u >= 0 and u not in seen:
                seen.add(u)
                chain.append(u)
                route.extend(parts[u])
                u = successor[u]
            if u >= 0:
                clean = False
                self.learn(route, True)
                # Break the most expensive edge on the cycle to obtain an actual path that can be repaired.
                z = max(range(len(chain)), key=lambda p: self.c[parts[chain[p]][-1]][parts[chain[(p + 1) % len(chain)]][0]])
                chain = chain[z + 1:] + chain[:z + 1]
                route = [t for i in chain for t in parts[i]]
            if self.f[route[-1]] - self.s[route[0]] > self.limit:
                clean = False
                self.learn(route)
            result.append(route)
        return result, clean

    def recombine(self, parts, outside, crews, stop, global_bound=False):
        """For the same segment model, first solve the matching relaxation, then eliminate infeasible chains with shared path cuts."""
        if not parts or time.monotonic() >= stop:
            return
        m = len(parts)
        edges, codes = [], []
        for i, a in enumerate(parts):
            if time.monotonic() >= stop:
                return
            u = a[-1]
            for j, b in enumerate(parts):
                if i != j and b[0] in self.ok[u] and self.f[b[-1]] - self.s[a[0]] <= self.limit:
                    edges.append((i, j, float(self.c[u][b[0]])))
                    codes.append(u * self.n + b[0])
        if not edges:
            self.consider(outside + parts)
            return
        successor, exact = None, False
        remaining = stop - time.monotonic()
        dimension = m + min(m, max(0, crews))
        # Dense assignment is used only when the estimated matrix workload fits the remaining wall clock.
        if np is not None and dimension * dimension <= max(1.0, remaining) * 4e6 and \
                dimension ** 3 <= max(0.0, remaining) * 2e9:
            try:
                matrix = np.full((dimension, dimension), np.inf)
                matrix[m:, :] = 0.0
                matrix[:, m:] = 0.0
                for i, j, w in edges:
                    matrix[i, j] = w
                rows, columns = linear_sum_assignment(matrix)
                successor = [-1] * m
                for i, j in zip(rows.tolist(), columns.tolist()):
                    if i < m and j < m:
                        successor[i] = j
                exact = True
            except Exception:
                successor = None
        if successor is None and time.monotonic() < stop:
            successor, exact = _flow(m, crews, edges, min(stop, time.monotonic() +
                                                     (stop - time.monotonic()) * 0.45))
        if successor is None:
            return
        raw, clean = self.decode(successor, parts)
        matching_cost = sum(float(self.c[parts[i][-1]][parts[j][0]])
                            for i, j in enumerate(successor) if j >= 0)
        constant = self.cost(outside) + self.cost(parts)
        if exact and global_bound:
            self.lower = max(self.lower, constant + matching_cost)
        self.consider(outside + raw)
        if clean and len(raw) <= crews and exact:
            return
        # Broken-chain repair of the matching and local search share the same feasible path state.
        repair_stop = min(stop, time.monotonic() + max(0.0, stop - time.monotonic()) * 0.12)
        self.repair_routes(outside + raw, repair_stop)
        if time.monotonic() >= stop:
            return
        if milp is None:
            # Without an integer solver, still rematch with conflict penalties and select by the original cost.
            while time.monotonic() < stop:
                penalized = [(i, j, w + self.penalty.get(code, 0.0) * (0.5 + self.rng.random()))
                             for (i, j, w), code in zip(edges, codes)]
                suc, _ = _flow(m, crews, penalized, min(stop, time.monotonic() +
                                                    (stop - time.monotonic()) * 0.5))
                candidate, _ = self.decode(suc, parts)
                self.repair_routes(outside + candidate, min(stop, time.monotonic() +
                                                          (stop - time.monotonic()) * 0.3))
            return
        # All internal segment connections are fixed to 1; the remaining cross-segment connections are binary variables.
        lookup = {code: e for e, code in enumerate(codes)}
        fixed = {a * self.n + b for r in parts + outside for a, b in zip(r, r[1:])}
        count = len(edges)
        rr, cc, vv = [], [], []
        for e, (i, j, w) in enumerate(edges):
            rr.extend((i, m + j, 2 * m))
            cc.extend((e, e, e))
            vv.extend((1.0, 1.0, 1.0))
        lower = [-math.inf] * (2 * m) + [max(0, m - crews)]
        upper = [1.0] * (2 * m) + [math.inf]
        included = set()
        objective = np.array([w for i, j, w in edges], dtype=float)
        while time.monotonic() < stop:
            for cut in self.cuts:
                if cut in included:
                    continue
                variables = []
                relevant = True
                for code in cut:
                    if code in lookup:
                        variables.append(lookup[code])
                    elif code not in fixed:
                        relevant = False
                        break
                if relevant and variables:
                    row = len(lower)
                    rr.extend([row] * len(variables))
                    cc.extend(variables)
                    vv.extend([1.0] * len(variables))
                    lower.append(-math.inf)
                    upper.append(len(variables) - 1)
                    included.add(cut)
            if time.monotonic() >= stop:
                return
            matrix = coo_matrix((vv, (rr, cc)), shape=(len(lower), count)).tocsc()
            rest = stop - time.monotonic()
            if rest <= 0:
                return
            try:
                answer = milp(objective, integrality=np.ones(count), bounds=Bounds(0.0, 1.0),
                              constraints=LinearConstraint(matrix, np.array(lower), np.array(upper)),
                              options={"time_limit": max(0.001, rest * 0.8), "mip_rel_gap": 0.0,
                                       "presolve": True})
            except Exception:
                return
            if answer.x is None:
                return
            successor = [-1] * m
            for e, value in enumerate(answer.x):
                if value > 0.5:
                    i, j, _ = edges[e]
                    successor[i] = j
            old_cuts = len(self.cuts)
            candidate, clean = self.decode(successor, parts)
            self.consider(outside + candidate)
            if clean and len(candidate) <= crews:
                if global_bound and answer.status == 0:
                    self.lower = max(self.lower, constant + float(answer.fun))
                return
            if len(self.cuts) == old_cuts:
                return
            self.repair_routes(outside + candidate, min(stop, time.monotonic() +
                                                       max(0.0, stop - time.monotonic()) * 0.10))

    def neighborhood(self, stagnant):
        routes = self.best if self.best is not None else self.partial
        if self.best is None:
            return [[t] for r in routes for t in r], [], self.k
        r = self.rng.randrange(len(routes))
        chosen = {r}
        anchor = routes[r]
        scores = []
        for j, route in enumerate(routes):
            if j == r:
                continue
            overlap = max(0.0, min(self.f[anchor[-1]], self.f[route[-1]]) -
                          max(self.s[anchor[0]], self.s[route[0]]))
            span = max(1e-8, self.limit)
            scores.append((self.rng.random() + overlap / span, j))
        scores.sort(reverse=True)
        elapsed = (time.monotonic() - self.started) / max(1e-8, self.budget)
        aim = min(self.n, max(len(anchor) + 1, int(math.sqrt(self.n + 1) *
                  math.log1p(self.budget + 1) * (1 + elapsed + math.log1p(stagnant)))))
        tasks = len(anchor)
        for _, j in scores:
            if tasks >= aim:
                break
            chosen.add(j)
            tasks += len(routes[j])
        parts, outside = [], []
        destroy = min(1.0, 0.25 + self.rng.random() * 0.5 +
                      math.log1p(stagnant) / (4 + math.log1p(self.n)))
        for j, route in enumerate(routes):
            if j not in chosen:
                outside.append(route[:])
                continue
            piece = [route[0]]
            for a, b in zip(route, route[1:]):
                score = abs(float(self.c[a][b])) / self.scale
                if self.rng.random() < min(1.0, destroy * (0.8 + 0.2 * score)):
                    parts.append(piece)
                    piece = []
                piece.append(b)
            parts.append(piece)
        return parts, outside, self.k - len(outside)

    def feasibility_dfs(self, stop):
        """While no feasible solution exists, supplement the repair search with backtracking crew assignment in time order."""
        order = self.order
        paths, options, choices = [], [], []
        pos = 0
        while time.monotonic() < stop:
            if pos == self.n:
                self.consider(paths)
                return
            t = order[pos]
            if len(options) <= pos:
                candidates = []
                equivalent = set()
                for j, r in enumerate(paths):
                    key = (r[0], r[-1])
                    if key in equivalent:
                        continue
                    equivalent.add(key)
                    if t in self.ok[r[-1]] and self.f[t] - self.s[r[0]] <= self.limit:
                        candidates.append((float(self.c[r[-1]][t]), j))
                candidates.sort(reverse=True)
                row = [j for _, j in candidates]
                if len(paths) < self.k:
                    row.insert(0, len(paths))
                options.append(row)
            if options[pos]:
                j = options[pos].pop()
                if j == len(paths):
                    paths.append([t])
                else:
                    paths[j].append(t)
                choices.append(j)
                pos += 1
            else:
                options.pop()
                if pos == 0:
                    return
                pos -= 1
                j = choices.pop()
                paths[j].pop()
                if not paths[j]:
                    paths.pop()

    def run(self):
        if self.n == 0:
            return _format([], max(0, self.k))
        self.prepare()
        if self.k >= self.n:
            self.consider([[i] for i in range(self.n)])
            if all(w >= 0 for row in self.out for w, _ in row):
                return _format(self.best, self.k)
        # Keep a low-cost constructed solution first, then start matching; every phase is bounded by the total wall clock.
        modes = [1, 2, 0, 3]
        for mode in modes:
            now = time.monotonic()
            if now >= self.stop:
                break
            self.construct(mode, 0.0, min(self.stop, now + self.budget * 0.035))
        if time.monotonic() < self.stop:
            fraction = 0.24 if self.best is not None else 0.38
            self.recombine([[i] for i in range(self.n)], [], self.k,
                           min(self.stop, time.monotonic() + self.budget * fraction), True)
        if self.best is None and time.monotonic() < self.stop:
            self.feasibility_dfs(min(self.stop, time.monotonic() + self.budget * 0.15))
        if self.best is not None:
            self.load(self.best)
            self.polish(min(self.stop, time.monotonic() + self.budget * 0.10))
        stagnant = 0
        while time.monotonic() < self.stop:
            if self.best is not None and self.best_cost <= self.lower + 1e-9:
                break
            old = self.best_cost
            now = time.monotonic()
            remaining = self.stop - now
            quantum = min(remaining, max(self.budget / (3 + math.sqrt(self.n + 1)),
                                         remaining / (4 + math.log1p(self.n))))
            if self.best is None:
                if self.rng.random() < 0.45:
                    self.construct(self.rng.randrange(4), self.rng.random(), min(self.stop, now + quantum))
                else:
                    parts, outside, crews = self.neighborhood(stagnant)
                    self.recombine(parts, outside, crews, min(self.stop, now + quantum))
            elif self.rng.random() < 0.65:
                parts, outside, crews = self.neighborhood(stagnant)
                self.recombine(parts, outside, crews, min(self.stop, now + quantum))
                self.load(self.best)
                self.polish(min(self.stop, time.monotonic() + quantum * 0.35))
            else:
                self.load(self.best)
                strength = max(1, int(math.log1p(self.budget) * (1 + math.log1p(stagnant))))
                kicks = 0
                end = min(self.stop, now + quantum * 0.25)
                while kicks < strength and time.monotonic() < end:
                    if not self.tail_pass(end, True):
                        break
                    kicks += 1
                if self.rng.random() < 0.35:
                    used = [i for i in range(self.n) if self.succ[i] >= 0]
                    self.rng.shuffle(used)
                    for i in used[:strength]:
                        self.unlink(i, self.succ[i])
                    self.repair_state(min(self.stop, now + quantum * 0.5))
                if self.count <= self.k:
                    self.polish(min(self.stop, now + quantum))
                self.consider(self.routes())
            stagnant = 0 if self.best_cost < old - 1e-10 else stagnant + 1
        return _format(self.best if self.best is not None else self.partial, self.k)


def solve(instance: dict, time_limit_s: float) -> dict:
    started = time.monotonic()
    engine = None
    try:
        engine = _Search(instance, float(time_limit_s), started)
        return engine.run()
    except Exception:
        # The verified feasible archive is stored independently and is not lost when a search neighbourhood fails.
        if engine is not None and engine.best is not None:
            return _format(engine.best, engine.k)
        if engine is not None:
            return _format(engine.partial, max(0, engine.k))
        try:
            n, k = int(instance["num_tasks"]), max(0, int(instance["max_crews"]))
            return _format([[i] for i in range(n)], k)
        except Exception:
            return {"crew_routes": [], "start_tasks": [], "end_tasks": []}
