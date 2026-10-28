"""Clique-cut covering with adaptive edge-removal search
Separates covering constraints with bitset clique separation, and improves the edge-removal plan with weighted tabu search, time-limited integer programming and triangle-greedy construction.
"""

import heapq
import math
import random
import time


def _turan_loss(size, target):
    if size <= target:
        return 0
    a, b = divmod(size, target)
    return target * a * (a - 1) // 2 + a * b


def _color_order(adj, candidates):
    order = []
    bounds = []
    color = 0
    remaining = candidates
    while remaining:
        color += 1
        available = remaining
        while available:
            bit = available & -available
            v = bit.bit_length() - 1
            order.append(v)
            bounds.append(color)
            remaining ^= bit
            available = (available ^ bit) & ~adj[v]
    return order, bounds


def _greedy_clique(adj, deadline, rng=None):
    candidates = (1 << len(adj)) - 1
    clique = []
    while candidates:
        scan = candidates
        best_v = (scan & -scan).bit_length() - 1
        best_score = -1.0
        while scan:
            bit = scan & -scan
            scan ^= bit
            v = bit.bit_length() - 1
            degree = (adj[v] & candidates).bit_count()
            score = degree
            if rng is not None:
                score += rng.random() * math.sqrt(degree + 1)
            if score > best_score:
                best_v, best_score = v, score
            if time.perf_counter() >= deadline:
                break
        clique.append(best_v)
        candidates &= adj[best_v]
        if time.perf_counter() >= deadline:
            # Subsequent bit operations only complete a valid clique; no extra search is performed.
            while candidates:
                bit = candidates & -candidates
                v = bit.bit_length() - 1
                clique.append(v)
                candidates &= adj[v]
            break
    return clique


class _CliqueSearch:
    """Resumable colouring-bound branch and bound; maximality is confirmed only when the search finishes."""

    def __init__(self, adj, deadline, hint=()):
        self.adj = adj
        self.best = _greedy_clique(adj, deadline)
        mask = 0
        valid = True
        for v in hint:
            if mask & ~adj[v]:
                valid = False
                break
            mask |= 1 << v
        if valid and len(hint) > len(self.best):
            self.best = list(hint)
        self.stack = []
        self.pending = (((1 << len(adj)) - 1), ())
        self.complete = False

    def run(self, deadline, stop_at=0):
        adj = self.adj
        while time.perf_counter() < deadline:
            if stop_at and len(self.best) >= stop_at:
                return
            if self.pending is not None:
                candidates, clique = self.pending
                self.pending = None
                if len(clique) + candidates.bit_count() > len(self.best):
                    if not candidates:
                        self.best = list(clique)
                    else:
                        order, bounds = _color_order(adj, candidates)
                        if len(clique) + bounds[-1] > len(self.best):
                            if bounds[-1] == len(order):
                                self.best = list(clique) + order
                            else:
                                self.stack.append([candidates, clique, order,
                                                   bounds, len(order) - 1])
                continue
            if not self.stack:
                self.complete = True
                return
            frame = self.stack[-1]
            candidates, clique, order, bounds, i = frame
            if i < 0 or len(clique) + bounds[i] <= len(self.best):
                self.stack.pop()
                continue
            v = order[i]
            bit = 1 << v
            frame[0] = candidates ^ bit
            frame[4] = i - 1
            next_clique = clique + (v,)
            next_candidates = candidates & adj[v]
            if not next_candidates:
                if len(next_clique) > len(self.best):
                    self.best = list(next_clique)
            elif len(next_clique) + next_candidates.bit_count() > len(self.best):
                self.pending = (next_candidates, next_clique)


class _CoverSearch:
    """The Turan necessary edge-removal count of a clique forms a multi-cover constraint."""

    def __init__(self, edge_count, budget, target, index, priority,
                 chosen, rng, total_time):
        self.m = edge_count
        self.k = budget
        self.target = target
        self.index = index
        self.priority = priority
        self.rng = rng
        self.total_time = total_time
        self.rows = []
        self.sizes = []
        self.keys = set()
        self.need = []
        self.weight = []
        self.count = []
        self.incidence = [[] for _ in range(edge_count)]
        self.gain = [0] * edge_count
        self.loss = [0] * edge_count
        self.chosen = set(chosen)
        self.flags = [False] * edge_count
        for e in chosen:
            self.flags[e] = True
        self.unsatisfied = set()
        self.deficit = 0
        self.tabu_add = [0.0] * edge_count
        self.tabu_remove = [0.0] * edge_count
        self.best_deficit = edge_count + 1
        self.best_chosen = set(chosen)
        self.last_progress = time.perf_counter()

    def add_clique(self, clique):
        size = len(clique)
        if size <= self.target:
            return False
        mask = 0
        for v in clique:
            mask |= 1 << v
        if mask in self.keys:
            return False
        self.keys.add(mask)
        edges = []
        for i, u in enumerate(clique):
            row = self.index[u]
            for v in clique[:i]:
                edges.append(row[v])
        j = len(self.rows)
        need = _turan_loss(size, self.target)
        count = sum(self.flags[e] for e in edges)
        self.rows.append(edges)
        self.sizes.append(size)
        self.need.append(need)
        self.weight.append(1)
        self.count.append(count)
        for e in edges:
            self.incidence[e].append(j)
            if count < need:
                self.gain[e] += 1
            if count <= need:
                self.loss[e] += 1
        if count < need:
            self.unsatisfied.add(j)
            self.deficit += need - count
        self.best_deficit = self.deficit
        self.best_chosen = set(self.chosen)
        self.last_progress = time.perf_counter()
        return True

    def reset(self, chosen, target=None):
        if target is not None and target != self.target:
            self.target = target
            self.need = [_turan_loss(s, target) for s in self.sizes]
            self.weight = [1] * len(self.rows)
        self.chosen = set(chosen)
        self.flags = [False] * self.m
        for e in self.chosen:
            self.flags[e] = True
        self.gain = [0] * self.m
        self.loss = [0] * self.m
        self.unsatisfied = set()
        self.deficit = 0
        self.count = []
        for j, edges in enumerate(self.rows):
            count = sum(self.flags[e] for e in edges)
            self.count.append(count)
            need, weight = self.need[j], self.weight[j]
            if count < need:
                self.unsatisfied.add(j)
                self.deficit += need - count
                for e in edges:
                    self.gain[e] += weight
            if count <= need:
                for e in edges:
                    self.loss[e] += weight
        self.tabu_add = [0.0] * self.m
        self.tabu_remove = [0.0] * self.m
        self.best_deficit = self.deficit
        self.best_chosen = set(self.chosen)
        self.last_progress = time.perf_counter()

    def add_edge(self, edge):
        if self.flags[edge]:
            return
        self.flags[edge] = True
        self.chosen.add(edge)
        for j in self.incidence[edge]:
            count, need, weight = self.count[j], self.need[j], self.weight[j]
            if count < need:
                self.deficit -= 1
            if count == need - 1:
                self.unsatisfied.discard(j)
                for e in self.rows[j]:
                    self.gain[e] -= weight
            if count == need:
                for e in self.rows[j]:
                    self.loss[e] -= weight
            self.count[j] = count + 1

    def remove_edge(self, edge):
        self.flags[edge] = False
        self.chosen.remove(edge)
        for j in self.incidence[edge]:
            count, need, weight = self.count[j], self.need[j], self.weight[j]
            if count <= need:
                self.deficit += 1
            if count == need:
                self.unsatisfied.add(j)
                for e in self.rows[j]:
                    self.gain[e] += weight
            if count == need + 1:
                for e in self.rows[j]:
                    self.loss[e] += weight
            self.count[j] = count - 1

    def penalize(self):
        for j in self.unsatisfied:
            self.weight[j] += 1
            for e in self.rows[j]:
                self.gain[e] += 1
                self.loss[e] += 1

    def advance(self, deadline):
        rng = self.rng
        tenure = (self.total_time / (30.0 + math.sqrt(self.m + 1)) /
                  (1.0 + math.sqrt(self.k + 1)))
        restart_delay = self.total_time / (4.0 + math.log2(self.m + 2))
        while self.unsatisfied and time.perf_counter() < deadline:
            now = time.perf_counter()
            if now - self.last_progress > restart_delay:
                strength = min(0.8, math.log1p(self.total_time) /
                               (2.0 + math.log1p(self.k + 1)))
                seed = {e for e in self.best_chosen if rng.random() > strength}
                self.reset(seed)
                self.penalize()
            removed = -1
            if len(self.chosen) >= self.k:
                candidates = [e for e in self.chosen if self.tabu_remove[e] <= now]
                if not candidates:
                    candidates = list(self.chosen)
                removed = min(candidates,
                              key=lambda e: (self.loss[e],
                                             self.priority[e] + rng.random()))
                self.remove_edge(removed)
                self.tabu_add[removed] = now + tenure * (0.5 + rng.random())
            best = -1
            best_score = -1.0
            for e, gain in enumerate(self.gain):
                if gain <= 0 or self.flags[e] or self.tabu_add[e] > now:
                    continue
                score = gain + 0.1 * self.priority[e] + 0.05 * rng.random()
                if score > best_score:
                    best, best_score = e, score
            if best < 0:
                for e, gain in enumerate(self.gain):
                    if not self.flags[e] and gain > best_score:
                        best, best_score = e, gain
            if best >= 0:
                self.add_edge(best)
                self.tabu_remove[best] = now + tenure * (0.5 + rng.random())
            elif removed >= 0:
                self.add_edge(removed)
            if self.deficit < self.best_deficit:
                self.best_deficit = self.deficit
                self.best_chosen = set(self.chosen)
                self.last_progress = time.perf_counter()
            if self.unsatisfied:
                self.penalize()
        return not self.unsatisfied


def _residual(original, endpoints, removed):
    adj = original.copy()
    for e in removed:
        u, v = endpoints[e]
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
    return adj


def _fill_budget(original, endpoints, removed, budget, deadline, rng):
    removed = set(removed)
    adj = _residual(original, endpoints, removed)
    if len(removed) >= budget or time.perf_counter() >= deadline:
        return removed, adj
    heap = []
    for e, (u, v) in enumerate(endpoints):
        if e not in removed:
            common = (adj[u] & adj[v]).bit_count()
            degree = adj[u].bit_count() + adj[v].bit_count()
            heap.append((-common, -degree, rng.random(), e))
    heapq.heapify(heap)
    while len(removed) < budget and heap and time.perf_counter() < deadline:
        a, b, tie, e = heapq.heappop(heap)
        u, v = endpoints[e]
        score = (-(adj[u] & adj[v]).bit_count(),
                 -(adj[u].bit_count() + adj[v].bit_count()))
        if score != (a, b):
            heapq.heappush(heap, (score[0], score[1], tie, e))
            continue
        removed.add(e)
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
    return removed, adj


def _collect_cuts(adj, cover, deadline, pool_limit, rng):
    """Short enumeration of threshold cliques; vary the vertex order to diversify the sources of cuts."""
    n = len(adj)
    threshold = cover.target + 1
    permutation = list(range(n))
    rng.shuffle(permutation)
    # Renumbering with random priorities serves only this enumeration and does not change solution indices.
    rank = [0] * n
    for i, v in enumerate(permutation):
        rank[v] = i
    graph = [0] * n
    for i, v in enumerate(permutation):
        scan = adj[v]
        mask = 0
        while scan:
            bit = scan & -scan
            scan ^= bit
            mask |= 1 << rank[bit.bit_length() - 1]
        graph[i] = mask
        if time.perf_counter() >= deadline:
            return
    pending = (((1 << n) - 1), ())
    stack = []
    while time.perf_counter() < deadline and len(cover.rows) < pool_limit:
        if pending is not None:
            candidates, clique = pending
            pending = None
            if len(clique) == threshold:
                cover.add_clique([permutation[v] for v in clique])
            elif len(clique) + candidates.bit_count() >= threshold:
                order, bounds = _color_order(graph, candidates)
                if order and len(clique) + bounds[-1] >= threshold:
                    stack.append([candidates, clique, order, bounds, len(order) - 1])
            continue
        if not stack:
            return
        frame = stack[-1]
        candidates, clique, order, bounds, i = frame
        if i < 0 or len(clique) + bounds[i] < threshold:
            stack.pop()
            continue
        v = order[i]
        frame[0] = candidates ^ (1 << v)
        frame[4] = i - 1
        pending = (candidates & graph[v], clique + (v,))


def _milp_cover(cover, deadline, backend):
    """Use only the separated necessary constraints; an infeasibility conclusion can serve as a global lower bound."""
    if backend is False:
        return None, False, False
    if backend is None:
        try:
            import numpy as np
            from scipy.optimize import milp, Bounds, LinearConstraint
            from scipy.sparse import coo_matrix
            backend = (np, milp, Bounds, LinearConstraint, coo_matrix)
        except Exception:
            return None, False, False
    np, milp, Bounds, LinearConstraint, coo_matrix = backend
    if time.perf_counter() >= deadline:
        return None, False, backend
    columns = sorted({e for row in cover.rows for e in row})
    if not columns:
        return set(), False, backend
    position = {e: j for j, e in enumerate(columns)}
    ii, jj = [], []
    for r, edges in enumerate(cover.rows):
        ii.extend([r] * len(edges))
        jj.extend(position[e] for e in edges)
        if time.perf_counter() >= deadline:
            return None, False, backend
    row_count = len(cover.rows)
    ii.extend([row_count] * len(columns))
    jj.extend(range(len(columns)))
    matrix = coo_matrix((np.ones(len(ii)), (np.asarray(ii, dtype=np.int32),
                                          np.asarray(jj, dtype=np.int32))),
                        shape=(row_count + 1, len(columns))).tocsc()
    lower = np.asarray(cover.need + [0], dtype=float)
    upper = np.full(row_count + 1, np.inf)
    upper[-1] = cover.k
    left = deadline - time.perf_counter()
    if left <= 0:
        return None, False, backend
    try:
        result = milp(np.ones(len(columns)),
                      integrality=np.ones(len(columns), dtype=np.uint8),
                      bounds=Bounds(0, 1),
                      constraints=LinearConstraint(matrix, lower, upper),
                      options={"time_limit": left, "mip_rel_gap": 0.0,
                               "presolve": True, "disp": False})
        if result.status == 2:
            return None, True, backend
        if result.x is not None:
            chosen = {columns[j] for j, x in enumerate(result.x) if x > 0.5}
            if len(chosen) <= cover.k and all(
                    sum(e in chosen for e in row) >= need
                    for row, need in zip(cover.rows, cover.need)):
                return chosen, False, backend
    except Exception:
        return None, False, False
    return None, False, backend


def _solve_core(instance, time_limit_s, start, safe):
    total = max(0.0, float(time_limit_s))
    deadline = start + total - min(0.1, total * 0.025)
    vertices = instance["vertices"]
    input_edges = instance["edges"]
    n, m = len(vertices), len(input_edges)
    if not n:
        return safe
    vertex_position = {v: i for i, v in enumerate(vertices)}
    degree = [0] * n
    raw_edges = []
    for edge in input_edges:
        u, v = vertex_position[edge[0]], vertex_position[edge[1]]
        raw_edges.append((u, v))
        degree[u] += 1
        degree[v] += 1
    permutation = sorted(range(n), key=lambda v: (-degree[v], v))
    rank = [0] * n
    for i, v in enumerate(permutation):
        rank[v] = i
    endpoints = [(rank[u], rank[v]) for u, v in raw_edges]
    original = [0] * n
    index = [{} for _ in range(n)]
    for e, (u, v) in enumerate(endpoints):
        original[u] |= 1 << v
        original[v] |= 1 << u
        index[u][v] = e
        index[v][u] = e
    budget = min(m, max(0, int(instance["interdiction_budget_k"])))

    def publish(removed, clique):
        status = [0] * m
        for e in removed:
            status[e] = 1
        membership = [0] * n
        for v in clique:
            membership[permutation[v]] = 1
        safe["interdiction_status"] = status
        safe["clique_membership"] = membership
        safe["max_clique_size"] = len(clique)
        return safe

    if budget == m:
        return publish(range(m), [0])

    # If every connected component is a complete graph, the Turan partition directly gives the global optimum.
    components = []
    unseen = (1 << n) - 1
    all_complete = True
    while unseen and time.perf_counter() < deadline:
        frontier = unseen & -unseen
        component = []
        unseen ^= frontier
        while frontier:
            bit = frontier & -frontier
            frontier ^= bit
            v = bit.bit_length() - 1
            component.append(v)
            added = original[v] & unseen
            unseen ^= added
            frontier |= added
        size = len(component)
        if any(original[v].bit_count() != size - 1 for v in component):
            all_complete = False
            break
        components.append(component)
    if all_complete and not unseen:
        lo, hi = 1, max(map(len, components))
        while lo < hi:
            mid = (lo + hi) // 2
            cost = sum(_turan_loss(len(c), mid) for c in components)
            if cost <= budget:
                hi = mid
            else:
                lo = mid + 1
        removed = set()
        witness = []
        for component in components:
            groups = [[] for _ in range(min(lo, len(component)))]
            for i, v in enumerate(component):
                groups[i % len(groups)].append(v)
            for group in groups:
                for i, u in enumerate(group):
                    for v in group[:i]:
                        removed.add(index[u][v])
            if len(groups) > len(witness):
                witness = [group[0] for group in groups]
        return publish(removed, witness)

    rng = random.Random(917231 + 1009 * n + 37 * m + budget)
    removed, adj = _fill_budget(original, endpoints, set(), budget,
                               min(deadline, start + total * 0.06), rng)
    oracle = _CliqueSearch(adj, min(deadline, start + total * 0.08))
    oracle.run(min(deadline, start + total * 0.23))
    if not oracle.complete:
        # First complete the initial follower solve, to avoid replacing a verified solution with an unverified candidate.
        oracle.run(deadline)
    best_removed = set(removed)
    best_clique = list(oracle.best)
    publish(best_removed, best_clique)
    if not oracle.complete or not budget or time.perf_counter() >= deadline:
        return safe
    best_size = len(best_clique)
    remaining_edges = m - budget
    lo, hi = 1, best_size
    complete_edges = n * (n - 1) // 2
    while lo < hi:
        mid = (lo + hi) // 2
        if remaining_edges <= complete_edges - _turan_loss(n, mid):
            hi = mid
        else:
            lo = mid + 1
    lower_bound = lo
    if best_size <= lower_bound:
        return safe

    priority = [(original[u] & original[v]).bit_count() / max(1, n)
                for u, v in endpoints]
    cover = _CoverSearch(m, budget, best_size - 1, index, priority,
                         best_removed, rng, total)
    cover.add_clique(best_clique)
    pool_limit = max(n, int(math.sqrt(m + n + 1) *
                           math.sqrt(1 + total) * 24))
    batch_time = total / (100.0 + math.sqrt(n + 1))
    local_time = total / (16.0 + math.log2(m + 2))
    exact_time = total / (8.0 + math.log2(n + 2))
    mip_time = total / (6.0 + math.log2(n + 2))
    _collect_cuts(adj, cover, min(deadline, time.perf_counter() + batch_time),
                  pool_limit, rng)
    backend = None
    last_mip = start
    pending_oracle = None
    candidate_removed = None
    candidate_adj = None
    previous_witness = best_clique
    search_deadline = deadline - min(total * 0.025, 0.08)

    while time.perf_counter() < search_deadline:
        now = time.perf_counter()
        if pending_oracle is not None:
            pending_oracle.run(min(deadline, now + exact_time), best_size)
            if len(pending_oracle.best) >= best_size:
                previous_witness = list(pending_oracle.best)
                cover.add_clique(previous_witness)
                pending_oracle = None
                if len(cover.rows) < pool_limit:
                    _collect_cuts(candidate_adj, cover,
                                  min(search_deadline,
                                      time.perf_counter() + batch_time),
                                  pool_limit, rng)
            elif pending_oracle.complete:
                best_clique = list(pending_oracle.best)
                best_removed = set(candidate_removed)
                best_size = len(best_clique)
                publish(best_removed, best_clique)
                pending_oracle = None
                if best_size <= lower_bound:
                    return safe
                cover.reset(best_removed, best_size - 1)
                cover.add_clique(best_clique)
                previous_witness = best_clique
            continue

        if cover.unsatisfied:
            solved = cover.advance(min(search_deadline, now + local_time))
            now = time.perf_counter()
            if not solved and now - last_mip >= local_time + mip_time:
                # Leave a margin for modelling and solver exit; unproven model results are not used as lower bounds.
                if search_deadline - now > mip_time + total * 0.03:
                    chosen, impossible, backend = _milp_cover(
                        cover, min(search_deadline - total * 0.02,
                                   now + mip_time), backend)
                    last_mip = time.perf_counter()
                    if impossible:
                        return safe
                    if chosen is not None:
                        cover.reset(chosen)
            continue

        candidate_removed, candidate_adj = _fill_budget(
            original, endpoints, cover.chosen, budget,
            min(search_deadline, now + batch_time), rng)
        if candidate_removed != cover.chosen:
            cover.reset(candidate_removed)
        pending_oracle = _CliqueSearch(
            candidate_adj, min(search_deadline, time.perf_counter() + batch_time),
            previous_witness)

    if pending_oracle is not None and len(pending_oracle.best) < best_size:
        pending_oracle.run(deadline, best_size)
        if pending_oracle.complete and len(pending_oracle.best) < best_size:
            publish(candidate_removed, pending_oracle.best)
    return safe


def solve(instance: dict, time_limit_s: float) -> dict:
    start = time.perf_counter()
    # Interface fallback keeps fields and arrays complete; normal inputs always go through the time-limited solve above.
    edges = instance.get("edges", [])
    vertices = instance.get("vertices", [])
    n = len(vertices)
    safe = {"interdiction_status": [0] * len(edges),
            "clique_membership": [1 if i == 0 else 0 for i in range(n)],
            "max_clique_size": int(n > 0)}
    try:
        return _solve_core(instance, time_limit_s, start, safe)
    except Exception:
        return safe
