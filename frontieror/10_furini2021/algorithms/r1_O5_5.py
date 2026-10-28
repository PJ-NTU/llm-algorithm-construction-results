"""Target clique-cut generation with weighted tabu covering
Generates edge-removal constraints from cliques of the target size, searches candidates with incremental weighted edge swaps and partition-based edge removal, and certifies the maximum clique by bitset colouring branch and bound.
"""

import math
import random
import time


def _greedy_clique(adj, deadline, seed=()):
    n = len(adj)
    clique = []
    candidates = (1 << n) - 1
    for v in seed:
        if candidates & (1 << v):
            clique.append(v)
            candidates &= adj[v]
    while candidates and time.perf_counter() < deadline:
        scan = candidates
        chosen = -1
        score = -1
        while scan:
            bit = scan & -scan
            v = bit.bit_length() - 1
            scan ^= bit
            value = (adj[v] & candidates).bit_count()
            if value > score:
                chosen, score = v, value
        if chosen < 0:
            break
        clique.append(chosen)
        candidates &= adj[chosen]
    return clique


def _maximum_clique(adj, deadline, stop_at=0, seed=(), upper=None):
    """Reaching stop_at can give a counterexample early; certification only when bounds meet or the search completes."""
    n = len(adj)
    if not n:
        return [], True, 0
    upper = n if upper is None else upper
    best = [0]
    warm = _greedy_clique(adj, deadline, seed)
    if len(warm) > len(best):
        best = warm
    if len(best) >= upper:
        return best, True, len(best)
    if stop_at and len(best) >= stop_at:
        return best, False, upper

    def color_sort(p):
        order, bounds = [], []
        color = 0
        while p:
            if time.perf_counter() >= deadline:
                return None
            color += 1
            available = p
            while available:
                bit = available & -available
                v = bit.bit_length() - 1
                p ^= bit
                available &= ~bit & ~adj[v]
                order.append(v)
                bounds.append(color)
        return order, bounds

    # The second phase starts only when the threshold search has proven an upper bound but still lacks a witness of the same size.
    while time.perf_counter() < deadline:
        root = color_sort((1 << n) - 1)
        if root is None:
            return best, False, upper
        upper = min(upper, root[1][-1])
        if len(best) >= upper:
            return best, True, len(best)
        stack = [[(1 << n) - 1, root[0], root[1], n - 1, ()]]
        while stack:
            if time.perf_counter() >= deadline:
                return best, False, upper
            frame = stack[-1]
            p, order, bounds, idx, cur = frame
            threshold = max(len(best), stop_at - 1)
            if idx < 0 or len(cur) + bounds[idx] <= threshold:
                stack.pop()
                continue
            v = order[idx]
            frame[0] = p & ~(1 << v)
            frame[3] -= 1
            child = cur + (v,)
            if len(child) > len(best):
                best = list(child)
                if len(best) >= upper:
                    return best, True, len(best)
                if stop_at and len(best) >= stop_at:
                    return best, False, upper
            next_p = p & adj[v]
            if len(child) + next_p.bit_count() <= threshold:
                continue
            colored = color_sort(next_p)
            if colored is None:
                return best, False, upper
            if colored[0] and len(child) + colored[1][-1] > threshold:
                stack.append([next_p, colored[0], colored[1],
                              len(colored[0]) - 1, child])
        if not stop_at:
            return best, True, len(best)
        upper = min(upper, stop_at - 1)
        if len(best) >= upper:
            return best, True, len(best)
        stop_at = 0
    return best, False, upper


def _turan_deletions(n, colors):
    a, b = divmod(n, colors)
    return colors * a * (a - 1) // 2 + b * a


class _Cover:
    """Incremental covering ledger of target cliques; weights only guide, never replace, the true objective."""

    def __init__(self, adj, endpoints, edge_index, target, removed, importance):
        self.adj = list(adj)
        self.endpoints = endpoints
        self.edge_index = edge_index
        self.target = target
        self.m = len(endpoints)
        self.chosen = set(removed)
        self.on = [False] * self.m
        self.inc = [[] for _ in endpoints]
        self.gain = [0] * self.m
        self.loss = [0] * self.m
        self.cuts = []
        self.vertices = []
        self.keys = set()
        self.count = []
        self.owner = []
        self.weight = []
        self.uncovered = set()
        self.active = set()
        self.tabu_add = [0] * self.m
        self.tabu_drop = [0] * self.m
        self.importance = importance
        self.step = 0
        for e in removed:
            self.on[e] = True
            u, v = endpoints[e]
            self.adj[u] &= ~(1 << v)
            self.adj[v] &= ~(1 << u)

    def add(self, vertices):
        vertices = tuple(sorted(vertices))
        if len(vertices) != self.target:
            return False
        key = sum(1 << v for v in vertices)
        if key in self.keys:
            return False
        edges = []
        for i, u in enumerate(vertices):
            for v in vertices[i + 1:]:
                edges.append(self.edge_index[(u, v)])
        if not edges:
            return False
        ci = len(self.cuts)
        self.keys.add(key)
        self.cuts.append(tuple(edges))
        self.vertices.append(vertices)
        self.weight.append(1)
        count = owner = 0
        for e in edges:
            self.inc[e].append(ci)
            self.active.add(e)
            if self.on[e]:
                count += 1
                owner ^= e + 1
        self.count.append(count)
        self.owner.append(owner)
        if count == 0:
            self.uncovered.add(ci)
            for e in edges:
                self.gain[e] += 1
        elif count == 1:
            self.loss[owner - 1] += 1
        return True

    def change(self, e, adding):
        if self.on[e] == adding:
            return
        self.on[e] = adding
        u, v = self.endpoints[e]
        if adding:
            self.chosen.add(e)
            self.adj[u] &= ~(1 << v)
            self.adj[v] &= ~(1 << u)
        else:
            self.chosen.remove(e)
            self.adj[u] |= 1 << v
            self.adj[v] |= 1 << u
        for ci in self.inc[e]:
            old = self.count[ci]
            weight = self.weight[ci]
            owner = self.owner[ci]
            if adding:
                if old == 0:
                    self.uncovered.remove(ci)
                    for f in self.cuts[ci]:
                        self.gain[f] -= weight
                    self.loss[e] += weight
                elif old == 1:
                    self.loss[owner - 1] -= weight
                self.count[ci] = old + 1
            else:
                if old == 1:
                    self.loss[e] -= weight
                    self.uncovered.add(ci)
                    for f in self.cuts[ci]:
                        self.gain[f] += weight
                elif old == 2:
                    self.loss[(owner ^ (e + 1)) - 1] += weight
                self.count[ci] = old - 1
            self.owner[ci] = owner ^ (e + 1)

    def move(self, k, rng, tenure, noise):
        if not self.uncovered or k <= 0:
            return
        self.step += 1
        ci = rng.choice(tuple(self.uncovered))
        candidates = self.cuts[ci]
        if len(self.chosen) < k:
            candidates = self.active
        allowed = [e for e in candidates
                   if not self.on[e] and self.tabu_add[e] <= self.step]
        if not allowed:
            allowed = [e for e in candidates if not self.on[e]]
        if not allowed:
            return
        if rng.random() < noise:
            incoming = rng.choice(allowed)
        else:
            incoming = max(allowed, key=lambda e: (
                self.gain[e], self.importance[e] * rng.uniform(0.8, 1.2),
                rng.random()))
        self.change(incoming, True)
        self.tabu_drop[incoming] = self.step + tenure
        if len(self.chosen) > k:
            allowed = [e for e in self.chosen
                       if e != incoming and self.tabu_drop[e] <= self.step]
            if not allowed:
                allowed = [e for e in self.chosen if e != incoming]
            outgoing = min(allowed, key=lambda e: (
                self.loss[e], self.importance[e] * rng.uniform(0.8, 1.2),
                rng.random()))
            self.change(outgoing, False)
            self.tabu_add[outgoing] = self.step + tenure
        # Persistently violated cliques get increased penalties, so there is an improving direction even on plateaus.
        if self.uncovered:
            ci = rng.choice(tuple(self.uncovered))
            self.weight[ci] += 1
            for e in self.cuts[ci]:
                self.gain[e] += 1


class _Partition:
    """Reduce within-group edges by moving vertices between groups; deleting within-group edges yields a colouring upper bound on clique size."""

    def __init__(self, neighbors, groups, rng, tenure):
        self.neighbors = neighbors
        self.n = len(neighbors)
        self.groups = groups
        self.rng = rng
        self.tenure = tenure
        self.step = 0
        self.labels = [-1] * self.n
        self.counts = [[0] * groups for _ in neighbors]
        sizes = [0] * groups
        for v in range(self.n):
            g = min(range(groups), key=lambda c: (
                self.counts[v][c], sizes[c], rng.random()))
            self.labels[v] = g
            sizes[g] += 1
            for u in neighbors[v]:
                self.counts[u][g] += 1
        self.cost = sum(self.counts[v][self.labels[v]]
                        for v in range(self.n)) // 2
        self.best = self.cost
        self.best_labels = list(self.labels)
        self.tabu = [[0] * groups for _ in neighbors]

    def run(self, k, deadline):
        rng = self.rng
        while self.cost > k and time.perf_counter() < deadline:
            self.step += 1
            best_move = None
            best_key = None
            for v in range(self.n):
                if time.perf_counter() >= deadline:
                    return None
                a = self.labels[v]
                row = self.counts[v]
                for b in range(self.groups):
                    if a == b:
                        continue
                    delta = row[b] - row[a]
                    if (self.tabu[v][b] > self.step
                            and self.cost + delta >= self.best):
                        continue
                    key = (delta, rng.random())
                    if best_key is None or key < best_key:
                        best_key, best_move = key, (v, a, b, delta)
            if best_move is None:
                self.tabu = [[0] * self.groups for _ in range(self.n)]
                continue
            v, a, b, delta = best_move
            self.labels[v] = b
            self.cost += delta
            self.tabu[v][a] = self.step + self.tenure + rng.randrange(self.tenure + 1)
            for u in self.neighbors[v]:
                self.counts[u][a] -= 1
                self.counts[u][b] += 1
            if self.cost < self.best:
                self.best = self.cost
                self.best_labels = list(self.labels)
        return list(self.labels) if self.cost <= k else None


def solve(instance: dict, time_limit_s: float) -> dict:
    started = time.perf_counter()
    vertices = instance.get('vertices', list(range(instance.get('num_vertices', 0))))
    n = len(vertices)
    edges = instance.get('edges', [])
    output_m = len(edges)
    empty = {'interdiction_status': [0] * output_m,
             'clique_membership': [0] * n, 'max_clique_size': 0}
    if n:
        empty['clique_membership'][0] = 1
        empty['max_clique_size'] = 1
    answer = empty
    try:
        duration = max(0.0, float(time_limit_s))
        if not math.isfinite(duration):
            duration = 0.0
        span = duration * 0.985
        deadline = started + span
        strength = 1.0 + math.log1p(duration)
        k = max(0, int(instance['interdiction_budget_k']))
        pos = {v: i for i, v in enumerate(vertices)}
        pair_first = {}
        degree = [0] * n
        for i, e in enumerate(edges):
            u, v = pos[e[0]], pos[e[1]]
            if u == v:
                continue
            pair = (min(u, v), max(u, v))
            if pair not in pair_first:
                pair_first[pair] = i
                degree[u] += 1
                degree[v] += 1
        order = sorted(range(n), key=lambda v: degree[v], reverse=True)
        inverse = [0] * n
        for i, v in enumerate(order):
            inverse[v] = i
        adj = [0] * n
        endpoints, representatives = [], []
        neighbors = [[] for _ in range(n)]
        edge_index = {}
        for (a, b), index in pair_first.items():
            u, v = sorted((inverse[a], inverse[b]))
            edge_index[(u, v)] = len(endpoints)
            endpoints.append((u, v))
            representatives.append(index)
            adj[u] |= 1 << v
            adj[v] |= 1 << u
            neighbors[u].append(v)
            neighbors[v].append(u)
        m = len(endpoints)
        k = min(k, m)
        rng = random.Random((n * 1000003 + m * 9176 + k * 97) & 0xffffffff)

        def package(removed, clique):
            w = [0] * output_m
            x = [0] * n
            for e in removed:
                w[representatives[e]] = 1
            for v in clique:
                x[order[v]] = 1
            return {'interdiction_status': w, 'clique_membership': x,
                    'max_clique_size': len(clique)}

        if n == 0 or m == 0:
            return package(set(), [0] if n else [])
        if k == m:
            return package(set(range(m)), [0])
        # A balanced partition of a complete graph attains the graph-theoretic lower bound; no heuristic search needed.
        if m == n * (n - 1) // 2:
            r = 1
            while _turan_deletions(n, r) > k:
                r += 1
            removed = {e for e, (u, v) in enumerate(endpoints) if u % r == v % r}
            return package(removed, list(range(r)))

        base, certified, base_upper = _maximum_clique(
            adj, min(deadline, started + span * (1.0 if k == 0 else 0.14)))
        answer = package(set(), base)
        if k == 0:
            return answer
        best_removed = set()
        best_clique = list(base)
        best_certified = certified
        best_size = len(base) if certified else n + 1
        target = len(base) if certified else min(base_upper, max(2, len(base) + 1))
        lower = 2
        while lower < len(base) and _turan_deletions(len(base), lower) > k:
            lower += 1
        while lower < n and m - k > n * (n - 1) // 2 - _turan_deletions(n, lower):
            lower += 1
        if certified and best_size <= lower:
            return answer
        importance = [(adj[u] & adj[v]).bit_count() + 1 for u, v in endpoints]
        tenure = max(1, int(math.sqrt(n + k) * strength / 6.0))
        capacity = max(n, int((n + math.sqrt(m + 1)) * strength * 8))
        epoch = span / (6.0 + 2.0 * strength)
        stall_span = span / (14.0 + 3.0 * strength)
        oracle_slice = span / (10.0 + strength)
        archive = [tuple(base)]
        cover = _Cover(adj, endpoints, edge_index, target, set(), importance)
        partition = None
        partition_target = None
        next_partition = time.perf_counter()
        last_progress = time.perf_counter()
        fewest = n + m + 1
        pending = None

        def add_witness(state, clique, until):
            if len(clique) < state.target:
                return
            # Contiguous circular subsets generate exactly all target subcliques when they differ by only one vertex.
            seq = list(clique)
            for shift in range(len(seq)):
                if time.perf_counter() >= until:
                    break
                subset = (seq[shift:] + seq[:shift])[:state.target]
                state.add(subset)
                if len(seq) == state.target:
                    break

        def accept(removed, clique, cert):
            nonlocal answer, best_removed, best_clique, best_size, best_certified
            if cert and (not best_certified or len(clique) < best_size):
                best_removed = set(removed)
                best_clique = list(clique)
                best_size = len(clique)
                best_certified = True
                answer = package(best_removed, best_clique)
                return True
            return False

        add_witness(cover, base, deadline)
        while time.perf_counter() < deadline:
            now = time.perf_counter()
            if best_certified and best_size <= lower:
                break
            if best_certified and target != best_size:
                target = best_size
                cover = _Cover(adj, endpoints, edge_index, target,
                               best_removed, importance)
                load_end = min(deadline, now + epoch * 0.18)
                for witness in reversed(archive):
                    add_witness(cover, witness, load_end)
                    if time.perf_counter() >= load_end:
                        break
                add_witness(cover, best_clique, deadline)
                partition = None
                pending = None
                fewest = n + m + 1
                last_progress = time.perf_counter()

            # Partition search and covering search alternate; the partition state is kept across time slices.
            if now >= next_partition and target > 2:
                groups = target - 1
                if partition is None or partition_target != groups:
                    partition = _Partition(neighbors, groups, rng, tenure)
                    partition_target = groups
                part_end = min(deadline, now + epoch * 0.22)
                labels = partition.run(k, part_end)
                next_partition = time.perf_counter() + epoch
                if labels is not None:
                    removed = {e for e, (u, v) in enumerate(endpoints)
                               if labels[u] == labels[v]}
                    residual = list(adj)
                    for e in removed:
                        u, v = endpoints[e]
                        residual[u] &= ~(1 << v)
                        residual[v] &= ~(1 << u)
                    clique, cert, ub = _maximum_clique(
                        residual, min(deadline, time.perf_counter() + oracle_slice),
                        upper=groups)
                    if accept(removed, clique, cert):
                        archive.append(tuple(clique))
                        continue
                    if not cert:
                        pending = (removed, list(clique), ub)
                    # Uncertified candidates can also guide, but do not overwrite a certified answer.
                    cover = _Cover(adj, endpoints, edge_index, target,
                                   removed, importance)
                    for witness in reversed(archive):
                        add_witness(cover, witness, min(deadline, next_partition))
                        if time.perf_counter() >= next_partition:
                            break
                    partition = None
            now = time.perf_counter()
            if now >= deadline:
                break
            if pending is not None and deadline - now < span * 0.12:
                removed, warm, ub = pending
                residual = list(adj)
                for e in removed:
                    u, v = endpoints[e]
                    residual[u] &= ~(1 << v)
                    residual[v] &= ~(1 << u)
                clique, cert, _ = _maximum_clique(residual, deadline, seed=warm, upper=ub)
                accept(removed, clique, cert)
                break

            if not cover.uncovered:
                until = min(deadline, now + oracle_slice)
                clique, cert, ub = _maximum_clique(cover.adj, until, stop_at=target)
                improved = accept(cover.chosen, clique, cert)
                if improved:
                    archive.append(tuple(clique))
                    continue
                if len(clique) >= target:
                    archive.append(tuple(clique))
                    add_witness(cover, clique, deadline)
                    if len(archive) > capacity:
                        archive = archive[-capacity:]
                    pending = None
                else:
                    pending = (set(cover.chosen), list(clique), ub)
                    # The current threshold could not be certified; continue with the remaining budget to construct a residual graph that is easier to certify.
                    if len(cover.chosen) < k:
                        available = [e for e in range(m) if not cover.on[e]]
                        if available:
                            e = max(available, key=lambda e: importance[e] * rng.random())
                            cover.change(e, True)
                    else:
                        removable = list(cover.chosen)
                        if removable:
                            cover.change(rng.choice(removable), False)
                fewest = len(cover.uncovered)
                last_progress = time.perf_counter()
                continue

            phase_end = min(deadline, next_partition)
            while cover.uncovered and time.perf_counter() < phase_end:
                now = time.perf_counter()
                count = len(cover.uncovered)
                if count < fewest:
                    fewest, last_progress = count, now
                stagnation = (now - last_progress) / max(stall_span, 1e-12)
                noise = min(0.25, 0.015 + 0.06 * stagnation)
                cover.move(k, rng, tenure, noise)
                if now - last_progress >= stall_span:
                    size = max(1, int(math.sqrt(k) * (1.0 + strength / 5.0)))
                    selected = list(cover.chosen)
                    rng.shuffle(selected)
                    selected.sort(key=lambda e: cover.loss[e] * rng.uniform(0.5, 1.5))
                    for e in selected[:size]:
                        if time.perf_counter() >= phase_end:
                            break
                        cover.change(e, False)
                        cover.tabu_add[e] = 0
                    fewest = n + m + 1
                    last_progress = time.perf_counter()
            # Bounded cut-pool capacity controls memory and does not limit the main loop or the number of restarts.
            if len(cover.cuts) > capacity and time.perf_counter() < deadline:
                retained = cover.vertices[-capacity // 2:]
                chosen = set(cover.chosen)
                cover = _Cover(adj, endpoints, edge_index, target, chosen, importance)
                until = min(deadline, time.perf_counter() + epoch * 0.18)
                for witness in retained:
                    if time.perf_counter() >= until:
                        break
                    cover.add(witness)
                fewest = n + m + 1
        return answer
    except Exception:
        # Certified answers are saved as a whole; internal exceptions do not break the previous incumbent or the output structure.
        return answer
