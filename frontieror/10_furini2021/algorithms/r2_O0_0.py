"""Clique-cut driven edge-interdiction hybrid search
Combines triangle reduction, weighted clique-covering repair, time-limited integer programming and partition tabu search, verifying candidate plans with bitset maximum clique search.
"""

import time
import math
import random
import heapq

try:
    import numpy as np
    from scipy.optimize import milp, Bounds, LinearConstraint
    from scipy.sparse import csc_matrix
except Exception:
    np = None
    milp = None


_clock = time.perf_counter


def _bits(mask):
    while mask:
        bit = mask & -mask
        yield bit.bit_length() - 1
        mask ^= bit


def _turan(s, r):
    a, b = divmod(s, r)
    return r * a * (a - 1) // 2 + b * a


def _greedy_clique(adj, deadline, rng=None, first=None):
    n = len(adj)
    p = (1 << n) - 1
    clique = []
    if first is not None:
        clique.append(first)
        p = adj[first]
    while p and _clock() < deadline:
        vertex = (p & -p).bit_length() - 1
        score = -1.0
        for j, v in enumerate(_bits(p)):
            value = (adj[v] & p).bit_count()
            if rng is not None:
                value += rng.random() * 1.5
            if value > score:
                vertex, score = v, value
            if (j & 127) == 127 and _clock() >= deadline:
                break
        clique.append(vertex)
        p &= adj[vertex]
    return clique


class _CliqueSearch:
    """Resumable colouring-bound search; when target is non-zero, enumerate cliques of that size."""

    def __init__(self, adj, best=(), target=0, upper=None):
        self.adj = adj
        self.best = list(best)
        self.target = target
        self.upper = len(adj) if upper is None else upper
        self.stack = []
        self.pending = ((1 << len(adj)) - 1, ())
        self.complete = not target and len(self.best) >= self.upper

    def _colors(self, p, deadline):
        order, bounds = [], []
        color = 0
        while p:
            color += 1
            available = p
            while available:
                bit = available & -available
                v = bit.bit_length() - 1
                order.append(v)
                bounds.append(color)
                p ^= bit
                available &= ~bit
                available &= ~self.adj[v]
            if _clock() >= deadline:
                return None
        return order, bounds

    def run(self, deadline, cap=1):
        found = []
        while not self.complete and _clock() < deadline:
            if self.pending is not None:
                p, prefix = self.pending
                need = self.target if self.target else len(self.best) + 1
                if len(prefix) + p.bit_count() < need:
                    self.pending = None
                    continue
                coloring = self._colors(p, deadline)
                if coloring is None:
                    break
                self.pending = None
                order, bounds = coloring
                if order:
                    self.stack.append([p, prefix, order, bounds, len(order) - 1])
                continue
            if not self.stack:
                self.complete = True
                break
            frame = self.stack[-1]
            p, prefix, order, bounds, index = frame
            need = self.target if self.target else len(self.best) + 1
            if index < 0 or len(prefix) + bounds[index] < need:
                self.stack.pop()
                continue
            v = order[index]
            frame[4] -= 1
            frame[0] &= ~(1 << v)
            extended = prefix + (v,)
            if len(extended) > len(self.best):
                self.best = list(extended)
            if self.target and len(extended) == self.target:
                found.append(extended)
                if len(found) >= cap:
                    break
                continue
            if not self.target and len(self.best) >= self.upper:
                self.complete = True
                break
            child = p & self.adj[v]
            need = self.target if self.target else len(self.best) + 1
            if child and len(extended) + child.bit_count() >= need:
                self.pending = (child, extended)
        return found


def _remaining(adj, edges, removed):
    result = adj.copy()
    for e in removed:
        u, v = edges[e]
        result[u] &= ~(1 << v)
        result[v] &= ~(1 << u)
    return result


def _triangle_start(adj, edges, edge_ids, k, deadline, rng):
    graph = adj.copy()
    m = len(edges)
    triangle = [(graph[u] & graph[v]).bit_count() for u, v in edges]
    noise = [rng.random() for _ in edges]
    queue = [(-triangle[e], noise[e], e) for e in range(m)]
    heapq.heapify(queue)
    removed = set()
    triangle_free = False
    while len(removed) < k and queue and _clock() < deadline:
        negative, _, e = heapq.heappop(queue)
        if e in removed or -negative != triangle[e]:
            continue
        if negative == 0:
            triangle_free = True
            break
        u, v = edges[e]
        common = graph[u] & graph[v]
        removed.add(e)
        graph[u] &= ~(1 << v)
        graph[v] &= ~(1 << u)
        for w in _bits(common):
            for other in (edge_ids[u][w], edge_ids[v][w]):
                triangle[other] -= 1
                heapq.heappush(queue, (-triangle[other], noise[other], other))
    return removed, graph, triangle_free


class _Cover:
    """Cover clique cuts with edge-removal variables, incrementally maintaining weighted deficits and single-edge swap costs."""

    def __init__(self, edges, edge_ids, removed, k, r, central, rng, scale):
        self.edges, self.edge_ids = edges, edge_ids
        self.k, self.r, self.rng = k, r, rng
        self.central = central
        self.selected = set(removed)
        self.cuts, self.rhs, self.weight, self.count = [], [], [], []
        self.incidence = [[] for _ in edges]
        self.gain = [0] * len(edges)
        self.loss = [0] * len(edges)
        self.uncovered, self.position = [], []
        self.seen = set()
        self.age = [0] * len(edges)
        self.move = 0
        self.tenure = max(1, int(math.sqrt(k + 1) * scale))
        self.value = 0

    def _mark(self, c, deficient):
        p = self.position[c]
        if deficient and p < 0:
            self.position[c] = len(self.uncovered)
            self.uncovered.append(c)
        elif not deficient and p >= 0:
            last = self.uncovered.pop()
            if p < len(self.uncovered):
                self.uncovered[p] = last
                self.position[last] = p
            self.position[c] = -1

    def add(self, vertices):
        s = len(vertices)
        if s <= self.r:
            return False
        mask = sum(1 << v for v in vertices)
        if mask in self.seen:
            return False
        self.seen.add(mask)
        cut = []
        for i, u in enumerate(vertices):
            for v in vertices[i + 1:]:
                cut.append(self.edge_ids[u][v])
        required = _turan(s, self.r)
        count = sum(e in self.selected for e in cut)
        c = len(self.cuts)
        self.cuts.append(tuple(cut))
        self.rhs.append(required)
        self.count.append(count)
        self.weight.append(1)
        self.position.append(-1)
        self._mark(c, count < required)
        self.value += max(0, required - count)
        for e in cut:
            self.incidence[e].append(c)
            if count < required:
                self.gain[e] += 1
            if count <= required:
                self.loss[e] += 1
        return True

    def toggle(self, e, adding):
        if adding:
            self.selected.add(e)
        else:
            self.selected.remove(e)
        change = 1 if adding else -1
        for c in self.incidence[e]:
            old, required, weight = self.count[c], self.rhs[c], self.weight[c]
            new = old + change
            self.count[c] = new
            self.value += weight * (max(0, required - new) - max(0, required - old))
            gain_delta = int(new < required) - int(old < required)
            loss_delta = int(new <= required) - int(old <= required)
            if gain_delta or loss_delta:
                for other in self.cuts[c]:
                    self.gain[other] += gain_delta * weight
                    self.loss[other] += loss_delta * weight
            self._mark(c, new < required)

    def reset(self, selected):
        self.selected = set(selected)
        self.gain = [0] * len(self.edges)
        self.loss = [0] * len(self.edges)
        self.uncovered = []
        self.position = [-1] * len(self.cuts)
        self.value = 0
        for c, cut in enumerate(self.cuts):
            count = sum(e in self.selected for e in cut)
            self.count[c] = count
            required, weight = self.rhs[c], self.weight[c]
            self.value += weight * max(0, required - count)
            self._mark(c, count < required)
            for e in cut:
                if count < required:
                    self.gain[e] += weight
                if count <= required:
                    self.loss[e] += weight

    def repair(self, deadline):
        rng = self.rng
        while self.uncovered and _clock() < deadline:
            before = self.value
            self.move += 1
            dropped = -1
            if len(self.selected) >= self.k:
                eligible = [e for e in self.selected
                            if self.move - self.age[e] > self.tenure]
                if not eligible:
                    eligible = list(self.selected)
                if not eligible:
                    return
                if rng.random() < 1.0 / (2.0 + math.sqrt(self.k + 1)):
                    dropped = rng.choice(eligible)
                else:
                    dropped = min(eligible, key=lambda e: (
                        self.loss[e], self.age[e], self.central[e]))
                self.toggle(dropped, False)
                self.age[dropped] = self.move
            if not self.uncovered:
                break
            c = rng.choice(self.uncovered)
            choices = [e for e in self.cuts[c]
                       if e not in self.selected and e != dropped]
            if not choices:
                choices = [e for e in self.cuts[c] if e not in self.selected]
            if choices:
                added = max(choices, key=lambda e: (
                    self.gain[e], -self.age[e], self.central[e] + rng.random()))
                self.toggle(added, True)
                self.age[added] = self.move
            if self.value >= before and self.uncovered:
                # Increase penalty weights on currently unsatisfied clique cuts to escape fixed-weight plateaus.
                sample = min(len(self.uncovered), max(1, int(math.sqrt(len(self.uncovered)))))
                for _ in range(sample):
                    if _clock() >= deadline:
                        break
                    c = rng.choice(self.uncovered)
                    self.weight[c] += 1
                    self.value += self.rhs[c] - self.count[c]
                    for e in self.cuts[c]:
                        self.gain[e] += 1
                        self.loss[e] += 1

    def integer_repair(self, deadline):
        if milp is None or not self.cuts or _clock() >= deadline:
            return None, False
        try:
            m = len(self.edges)
            rows, cols = [], []
            for c, cut in enumerate(self.cuts):
                if _clock() >= deadline:
                    return None, False
                rows.extend([c] * len(cut))
                cols.extend(cut)
            rows.extend([len(self.cuts)] * m)
            cols.extend(range(m))
            matrix = csc_matrix((np.ones(len(rows)), (rows, cols)),
                                shape=(len(self.cuts) + 1, m))
            lower = np.asarray(self.rhs + [-np.inf], dtype=float)
            upper = np.asarray([np.inf] * len(self.cuts) + [self.k], dtype=float)
            peak = max(self.central, default=1) + 1
            objective = np.asarray([1.0 - 0.01 * c / peak for c in self.central])
            remaining = deadline - _clock()
            if remaining <= 0:
                return None, False
            result = milp(objective, integrality=np.ones(m), bounds=Bounds(0, 1),
                          constraints=LinearConstraint(matrix, lower, upper),
                          options={"time_limit": remaining, "mip_rel_gap": 0.0,
                                   "presolve": True, "disp": False})
            if result.x is not None:
                selected = {e for e in range(m) if result.x[e] > 0.5}
                if len(selected) <= self.k and all(
                    sum(e in selected for e in cut) >= self.rhs[c]
                    for c, cut in enumerate(self.cuts)
                ):
                    return selected, False
            return None, result.status == 2
        except Exception:
            return None, False


class _Partition:
    """Minimise the number of same-group edges; finding a feasible partition gives a constructive upper bound on the maximum clique."""

    def __init__(self, adj, edges, r, k, rng, scale):
        self.adj, self.edges, self.r, self.k, self.rng = adj, edges, r, k, rng
        n = len(adj)
        order = list(range(n))
        rng.shuffle(order)
        order.sort(key=lambda v: adj[v].bit_count(), reverse=True)
        self.color = [-1] * n
        self.count = [[0] * r for _ in adj]
        size = [0] * r
        for v in order:
            c = min(range(r), key=lambda c: (self.count[v][c], size[c], rng.random()))
            self.color[v] = c
            size[c] += 1
            for u in _bits(adj[v]):
                self.count[u][c] += 1
        self.value = sum(self.count[v][self.color[v]] for v in range(n)) // 2
        self.best = self.value
        self.tabu = [[0] * r for _ in adj]
        self.step = 0
        self.tenure = max(1, int(math.sqrt(n) * scale))

    def search(self, deadline):
        n, r, rng = len(self.adj), self.r, self.rng
        while _clock() < deadline:
            if self.value <= self.k:
                return {e for e, (u, v) in enumerate(self.edges)
                        if self.color[u] == self.color[v]}
            self.step += 1
            move = None
            best_key = (math.inf, math.inf)
            for v in range(n):
                old = self.color[v]
                if self.count[v][old] == 0:
                    continue
                for new in range(r):
                    if new == old:
                        continue
                    delta = self.count[v][new] - self.count[v][old]
                    if self.tabu[v][new] > self.step and self.value + delta >= self.best:
                        continue
                    key = (delta, rng.random())
                    if key < best_key:
                        best_key, move = key, (v, old, new, delta)
                if (v & 63) == 63 and _clock() >= deadline:
                    return None
            if move is None:
                v = rng.randrange(n)
                old = self.color[v]
                new = rng.randrange(r)
                if new == old:
                    continue
                move = v, old, new, self.count[v][new] - self.count[v][old]
            v, old, new, delta = move
            self.color[v] = new
            self.value += delta
            self.best = min(self.best, self.value)
            self.tabu[v][old] = self.step + self.tenure + rng.randrange(self.tenure + 1)
            for u in _bits(self.adj[v]):
                self.count[u][old] -= 1
                self.count[u][new] += 1
        return None


def solve(instance: dict, time_limit_s: float) -> dict:
    started = _clock()
    vertices = list(instance.get("vertices", range(instance.get("num_vertices", 0))))
    raw_edges = list(instance.get("edges", []))
    n, m = len(vertices), len(raw_edges)
    answer = {"interdiction_status": [0] * m,
              "clique_membership": [int(i == 0) for i in range(n)],
              "max_clique_size": int(n > 0)}
    try:
        budget = max(0, min(m, int(instance.get("interdiction_budget_k", 0))))
        total = max(0.0, float(time_limit_s))
        margin = min(0.05, total * 0.015)
        deadline = started + max(0.0, total - margin)
        scale = 1.0 + math.log1p(total)
        rng = random.Random((n * 1000003 + m * 9176 + budget * 37) & 0xFFFFFFFF)
        positions = {v: i for i, v in enumerate(vertices)}
        degree = [0] * n
        indexed = [(positions[u], positions[v]) for u, v in raw_edges]
        for u, v in indexed:
            degree[u] += 1
            degree[v] += 1
        order = sorted(range(n), key=lambda v: degree[v], reverse=True)
        inverse = [0] * n
        for i, v in enumerate(order):
            inverse[v] = i
        edges = [(inverse[u], inverse[v]) for u, v in indexed]
        adj = [0] * n
        edge_ids = [{} for _ in range(n)]
        for e, (u, v) in enumerate(edges):
            adj[u] |= 1 << v
            adj[v] |= 1 << u
            edge_ids[u][v] = edge_ids[v][u] = e

        def pack(removed, clique):
            membership = [0] * n
            for v in clique:
                membership[order[v]] = 1
            return {"interdiction_status": [int(e in removed) for e in range(m)],
                    "clique_membership": membership,
                    "max_clique_size": len(clique)}

        if not n or not m:
            return pack(set(), [0] if n else [])
        if budget == m:
            return pack(set(range(m)), [0])
        if m == n * (n - 1) // 2:
            r = 1
            while _turan(n, r) > budget:
                r += 1
            removed = {e for e, (u, v) in enumerate(edges) if u % r == v % r}
            return pack(removed, list(range(r)))

        original_clique = _greedy_clique(adj, min(deadline, started + total * 0.025))
        if not original_clique:
            original_clique = [edges[0][0], edges[0][1]]
        lower_bound = 2
        while _turan(len(original_clique), lower_bound) > budget:
            lower_bound += 1
        if budget == 0:
            oracle = _CliqueSearch(adj, original_clique)
            oracle.run(deadline)
            return pack(set(), oracle.best)

        best_removed, graph, triangle_free = _triangle_start(
            adj, edges, edge_ids, budget, min(deadline, started + total * 0.10), rng)
        initial_clique = _greedy_clique(graph, min(deadline, started + total * 0.12))
        if not initial_clique:
            initial_clique = [0]
        if triangle_free:
            for e, (u, v) in enumerate(edges):
                if e not in best_removed:
                    return pack(best_removed, [u, v])
        baseline = _CliqueSearch(graph, initial_clique)
        baseline.run(min(deadline, started + total * 0.24))
        certified = baseline.complete
        best_clique = baseline.best.copy()
        answer = pack(best_removed, best_clique)
        best_size = len(best_clique)
        if certified and best_size <= lower_bound:
            return answer

        central = [(adj[u] & adj[v]).bit_count() for u, v in edges]
        pool = []
        pool_seen = set()

        def remember(clique):
            mask = sum(1 << v for v in clique)
            if mask not in pool_seen:
                pool_seen.add(mask)
                pool.append(tuple(clique))
            return tuple(clique)

        remember(original_clique)
        remember(best_clique)
        sampling_end = min(deadline, _clock() + total * 0.025)
        while _clock() < sampling_end:
            clique = _greedy_clique(adj, sampling_end, rng, rng.randrange(n))
            remember(clique)
            while _turan(len(clique), lower_bound) > budget:
                lower_bound += 1

        target = max(lower_bound, best_size - 1)
        cover = None
        partition = None
        separator = None
        verification = None
        candidate_removed = None
        next_integer = _clock() + total * 0.04
        next_partition = _clock()
        batch = max(1, int(math.sqrt(n + m) * scale))
        quantum = total / (12.0 + math.log1p(n + m))
        uncertain_end = deadline - total * 0.12

        while _clock() < (deadline if certified else uncertain_end):
            now = _clock()
            if certified and best_size <= lower_bound:
                break
            wanted = max(lower_bound, best_size - 1)
            if cover is None or wanted != target:
                target = wanted
                cover = _Cover(edges, edge_ids, best_removed, budget, target,
                               central, rng, scale)
                for clique in pool:
                    if _clock() >= deadline:
                        break
                    cover.add(clique)
                cover.add(best_clique)
                partition = None
                separator = None
                verification = None
                next_partition = now

            if verification is not None:
                verification.run(min(deadline, now + quantum))
                if verification.complete:
                    size = len(verification.best)
                    if not certified or size < best_size:
                        best_removed = set(candidate_removed)
                        best_clique = verification.best.copy()
                        best_size = size
                        certified = True
                        answer = pack(best_removed, best_clique)
                        remember(best_clique)
                        cover = None
                    verification = None
                    separator = None
                continue

            if separator is not None:
                found = separator.run(min(deadline, now + quantum), batch)
                if found:
                    for clique in found:
                        cover.add(remember(clique))
                    separator = None
                elif separator.complete:
                    verification = _CliqueSearch(separator.adj, separator.best, upper=target)
                    separator = None
                continue

            if _clock() >= next_partition:
                part_end = min(deadline, _clock() + quantum * 0.6)
                if partition is None:
                    partition = _Partition(adj, edges, target, budget, rng, scale)
                partition_removed = partition.search(part_end)
                next_partition = _clock() + quantum * (3.0 + scale)
                if partition_removed is not None:
                    candidate_removed = partition_removed
                    candidate_graph = _remaining(adj, edges, candidate_removed)
                    clique = _greedy_clique(candidate_graph, min(deadline, _clock() + quantum * 0.25))
                    verification = _CliqueSearch(candidate_graph, clique, upper=target)
                    continue

            if cover.uncovered:
                cover.repair(min(deadline, _clock() + quantum))
                if cover.uncovered and _clock() >= next_integer:
                    now = _clock()
                    integer_time = min(total * 0.10, (deadline - now) * 0.25)
                    selected, impossible = cover.integer_repair(now + max(0.0, integer_time))
                    next_integer = _clock() + quantum * scale
                    if impossible and certified and target == best_size - 1:
                        break
                    if selected is not None:
                        cover.reset(selected)
                if cover.uncovered:
                    continue

            # When the budget is not used up, prefer removing edges that still lie in the most triangles.
            if len(cover.selected) < budget:
                current_graph = _remaining(adj, edges, cover.selected)
                candidates = [e for e in range(m) if e not in cover.selected]
                candidates.sort(key=lambda e: (
                    (current_graph[edges[e][0]] & current_graph[edges[e][1]]).bit_count(),
                    central[e]), reverse=True)
                for e in candidates:
                    if len(cover.selected) >= budget or _clock() >= deadline:
                        break
                    cover.toggle(e, True)
            candidate_removed = set(cover.selected)
            candidate_graph = _remaining(adj, edges, candidate_removed)
            clique = _greedy_clique(candidate_graph, min(deadline, _clock() + quantum * 0.2))
            if len(clique) > target:
                cover.add(remember(clique))
                # Besides the Turan cut for large cliques, also add a specific clique-forbidding cut.
                subset = rng.sample(clique, target + 1)
                cover.add(remember(subset))
                continue
            separator = _CliqueSearch(candidate_graph, clique, target=target + 1)

        if not certified and _clock() < deadline:
            baseline.run(deadline)
            answer = pack(best_removed, baseline.best)
        return answer
    except Exception:
        return answer
