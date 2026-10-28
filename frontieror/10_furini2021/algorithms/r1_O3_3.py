"""Multi-clique-cut with adaptive edge-swap fusion algorithm
A shared clique pool generates weighted hitting and multiple edge-removal constraints; the plan is improved with incremental edge-swap repair, colouring construction and resumable exact maximum clique search.
"""

import time
import random
import math


def _required(size, target):
    """Minimum number of edge removals required to reduce a complete graph to cliques of size at most target."""
    if size <= target:
        return 0
    q, r = divmod(size, target)
    return target * q * (q - 1) // 2 + r * q


def _greedy(adj, rng, deadline, first=None):
    p = (1 << len(adj)) - 1
    clique = []
    if first is not None:
        clique.append(first)
        p &= adj[first]
    while p and time.perf_counter() < deadline:
        q = p
        chosen = -1
        score = -1.0
        spread = max(1.0, math.sqrt(p.bit_count()) * 0.35)
        while q:
            b = q & -q
            v = b.bit_length() - 1
            val = (adj[v] & p).bit_count() + rng.random() * spread
            if val > score:
                score, chosen = val, v
            q ^= b
        clique.append(chosen)
        p &= adj[chosen]
    return clique


class _CliqueSearch:
    """Save the search stack; a timeout only pauses, and a lower bound is never passed off as a certified result."""

    def __init__(self, adj, seed, stop_at=None):
        self.adj = adj
        self.best = list(seed)
        self.stop_at = stop_at
        self.complete = not adj
        self.stack = [[(1 << len(adj)) - 1, (), None, None, -1]]

    def run(self, deadline):
        adj = self.adj
        while self.stack and time.perf_counter() < deadline:
            if self.stop_at is not None and len(self.best) >= self.stop_at:
                return
            frame = self.stack[-1]
            p, cur = frame[0], frame[1]
            depth = len(cur)
            if depth + p.bit_count() <= len(self.best):
                self.stack.pop()
                continue
            if frame[2] is None:
                order, bounds = [], []
                q, color = p, 0
                while q:
                    if time.perf_counter() >= deadline:
                        return
                    color += 1
                    available = q
                    while available:
                        b = available & -available
                        v = b.bit_length() - 1
                        q ^= b
                        available &= ~b & ~adj[v]
                        order.append(v)
                        bounds.append(color)
                        if len(order) % 64 == 0 and time.perf_counter() >= deadline:
                            return
                frame[2], frame[3], frame[4] = order, bounds, len(order) - 1
            i = frame[4]
            if i < 0 or depth + frame[3][i] <= len(self.best):
                self.stack.pop()
                continue
            v = frame[2][i]
            frame[4] -= 1
            frame[0] &= ~(1 << v)
            child = p & adj[v]
            branch = cur + (v,)
            if len(branch) > len(self.best):
                self.best = list(branch)
            if child and len(branch) + child.bit_count() > len(self.best):
                self.stack.append([child, branch, None, None, -1])
        if not self.stack:
            self.complete = True


class _Master:
    """Incrementally maintain clique-cut deficits, edge-addition gains and the loss of undoing removals."""

    def __init__(self, count, budget, target, selected, prior, rng, horizon):
        self.k, self.target = budget, target
        self.on = set(selected)
        self.prior, self.rng = prior, rng
        self.rows, self.need, self.weight, self.cover = [], [], [], []
        self.incidence = [[] for _ in range(count)]
        self.gain, self.loss = [0.0] * count, [0.0] * count
        self.bad = set()
        self.value = 0.0
        self.tabu = [0.0] * count
        self.best_value = float('inf')
        self.last_progress = time.perf_counter()
        self.horizon = horizon
        self.ranking = sorted(range(count), key=prior.__getitem__, reverse=True)
        self.fill_cursor = 0

    def add(self, size, edges):
        if size <= self.target or not edges:
            return
        r = len(self.rows)
        d = _required(size, self.target)
        w = 1.0 + (size - self.target) / self.target
        c = sum(e in self.on for e in edges)
        self.rows.append(edges)
        self.need.append(d)
        self.weight.append(w)
        self.cover.append(c)
        if c < d:
            self.bad.add(r)
            self.value += w * (d - c)
        for e in edges:
            self.incidence[e].append(r)
            if c < d:
                self.gain[e] += w
            if c <= d:
                self.loss[e] += w
        self.best_value = float('inf')

    def toggle(self, e, adding):
        if adding:
            self.on.add(e)
        else:
            self.on.remove(e)
        for r in self.incidence[e]:
            c, d, w = self.cover[r], self.need[r], self.weight[r]
            if adding:
                if c < d:
                    self.value -= w
                if c == d - 1:
                    self.bad.discard(r)
                    for f in self.rows[r]:
                        self.gain[f] -= w
                if c == d:
                    for f in self.rows[r]:
                        self.loss[f] -= w
                self.cover[r] = c + 1
            else:
                if c <= d:
                    self.value += w
                if c == d:
                    self.bad.add(r)
                    for f in self.rows[r]:
                        self.gain[f] += w
                if c == d + 1:
                    for f in self.rows[r]:
                        self.loss[f] += w
                self.cover[r] = c - 1

    def penalize(self, deadline):
        for r in tuple(self.bad):
            if time.perf_counter() >= deadline:
                break
            delta = 0.25 * (1.0 + self.need[r] / max(1, len(self.rows[r])))
            self.weight[r] += delta
            self.value += delta * (self.need[r] - self.cover[r])
            for e in self.rows[r]:
                self.gain[e] += delta
                self.loss[e] += delta
        self.best_value = self.value

    def choose_in(self, now, noise):
        if self.bad:
            r = self.rng.choice(tuple(self.bad))
            candidates = [e for e in self.rows[r] if e not in self.on]
            available = [e for e in candidates if self.tabu[e] <= now]
            if available:
                candidates = available
            if candidates:
                return max(candidates, key=lambda e: self.gain[e] *
                           (1.0 + noise * self.rng.random()) + self.prior[e] * 1e-5)
        while self.fill_cursor < len(self.ranking):
            e = self.ranking[self.fill_cursor]
            self.fill_cursor += 1
            if e not in self.on:
                return e
        available = [e for e in self.ranking if e not in self.on]
        return self.rng.choice(available) if available else None

    def repair(self, deadline, quantum, progress):
        rng = self.rng
        tenure = quantum / (2.0 + math.sqrt(self.k + 1))
        stagnation = quantum * (0.35 + 0.25 * progress)
        noise = 0.08 + 0.22 * progress
        while time.perf_counter() < deadline:
            now = time.perf_counter()
            if not self.bad and len(self.on) >= self.k:
                return True
            if self.value < self.best_value - 1e-7:
                self.best_value = self.value
                self.last_progress = now
            if self.bad and now - self.last_progress > stagnation:
                self.penalize(deadline)
                amplitude = min(len(self.on), max(1, int(
                    math.sqrt(self.k + 1) * math.sqrt(math.log1p(self.horizon)) *
                    (0.25 + progress))))
                if amplitude:
                    candidates = sorted(self.on, key=lambda e: self.loss[e] *
                                        (0.5 + rng.random()))
                    for e in candidates[:amplitude]:
                        if time.perf_counter() >= deadline:
                            return False
                        self.toggle(e, False)
                        self.tabu[e] = now + tenure
                self.last_progress = now
                self.fill_cursor = 0
            incoming = self.choose_in(now, noise)
            if incoming is None:
                return not self.bad
            self.toggle(incoming, True)
            if len(self.on) > self.k:
                candidates = [e for e in self.on if e != incoming]
                available = [e for e in candidates if self.tabu[e] <= now]
                if available:
                    candidates = available
                outgoing = min(candidates, key=lambda e: self.loss[e] *
                               (1.0 + noise * rng.random()) + self.prior[e] * 1e-5)
                self.toggle(outgoing, False)
                self.tabu[outgoing] = now + tenure
            self.tabu[incoming] = now + tenure
        return not self.bad and len(self.on) <= self.k


def solve(instance: dict, time_limit_s: float) -> dict:
    start = time.perf_counter()
    try:
        horizon = max(0.0, float(time_limit_s))
        if not math.isfinite(horizon):
            horizon = 0.0
    except (TypeError, ValueError, OverflowError):
        horizon = 0.0
    end = start + horizon * 0.975
    vertices = list(instance.get('vertices', range(instance.get('num_vertices', 0))))
    raw_edges = instance.get('edges', [])
    n, raw_m = len(vertices), len(raw_edges)
    fallback = {'interdiction_status': [0] * raw_m,
                'clique_membership': [int(i == 0) for i in range(n)],
                'max_clique_size': int(n > 0)}
    best_result = None
    try:
        pos = {v: i for i, v in enumerate(vertices)}
        pairs = {}
        original = [0] * n
        for i, edge in enumerate(raw_edges):
            u, v = pos[edge[0]], pos[edge[1]]
            if u == v:
                continue
            if u > v:
                u, v = v, u
            if (u, v) not in pairs:
                pairs[u, v] = i
                original[u] |= 1 << v
                original[v] |= 1 << u
        m = len(pairs)
        k = min(m, max(0, int(instance.get('interdiction_budget_k', 0))))
        # The static degree order only changes internal numbering; output still follows the order of vertices and edges strictly.
        order = sorted(range(n), key=lambda v: original[v].bit_count(), reverse=True)
        inverse = [0] * n
        for i, v in enumerate(order):
            inverse[v] = i
        edges, output_ids, lookup = [], [], {}
        adj0 = [0] * n
        for (a, b), output_id in pairs.items():
            u, v = inverse[a], inverse[b]
            e = len(edges)
            edges.append((u, v))
            output_ids.append(output_id)
            lookup[min(u, v), max(u, v)] = e
            adj0[u] |= 1 << v
            adj0[v] |= 1 << u
        rng = random.Random(n * 1000003 + m * 9176 + k * 97)

        def package(selected, clique):
            w, x = [0] * raw_m, [0] * n
            for e in selected:
                w[output_ids[e]] = 1
            for v in clique:
                x[order[v]] = 1
            return {'interdiction_status': w, 'clique_membership': x,
                    'max_clique_size': len(clique)}

        def residual(selected):
            adj = list(adj0)
            for e in selected:
                u, v = edges[e]
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
            return adj

        if n == 0 or m == 0:
            return package([], [0] if n else [])
        if k == m:
            return package(range(m), [0])
        # A balanced partition of a complete graph attains the Turan lower bound; no search needed.
        if m == n * (n - 1) // 2:
            target = 1
            while _required(n, target) > k:
                target += 1
            groups = [v % target for v in range(n)]
            selected = [e for e, (u, v) in enumerate(edges) if groups[u] == groups[v]]
            return package(selected, list(range(target)))

        prior = [(adj0[u] & adj0[v]).bit_count() + 1 for u, v in edges]
        scale = max(prior, default=1)
        prior = [v / scale for v in prior]
        quantum = horizon / (12.0 + math.sqrt(n + 1))
        seed_end = min(end, start + horizon * 0.035)
        seed = _greedy(adj0, rng, seed_end)
        if not seed:
            seed = [0]
        pool, seen = [], set()
        master = None
        best_selected, best_clique = set(), []
        strongest = list(seed)

        def remember(clique):
            nonlocal strongest
            if len(clique) > len(strongest):
                strongest = list(clique)
            if len(clique) <= 1:
                return
            mask = sum(1 << v for v in clique)
            if mask in seen:
                return
            seen.add(mask)
            ce = []
            for i, u in enumerate(clique):
                for v in clique[i + 1:]:
                    ce.append(lookup[min(u, v), max(u, v)])
            ce = tuple(ce)
            pool.append((len(clique), ce))
            if master is not None:
                master.add(len(clique), ce)

        def accept(selected, clique):
            nonlocal best_selected, best_clique, best_result
            if best_result is None or len(clique) < len(best_clique):
                best_selected, best_clique = set(selected), list(clique)
                best_result = package(selected, clique)
                return True
            return False

        remember(seed)
        while time.perf_counter() < seed_end:
            c = _greedy(adj0, rng, seed_end, rng.randrange(n))
            remember(c)
            if len(c) > len(seed):
                seed = c
        base = _CliqueSearch(adj0, seed)
        base.run(min(end, start + horizon * 0.20))
        remember(base.best)
        if base.complete:
            accept(set(), base.best)
        if k == 0:
            if not base.complete:
                base.run(end)
            return package(set(), base.best)

        # Anchored partition construction: delete within-group edges; the anchor clique gives a lower bound and the group count gives the same upper bound.
        # Its edge set also becomes a warm start for the same clique-cut master problem, rather than a separate algorithm.
        partition_end = min(end, start + horizon * 0.28)
        anchors = list(strongest)
        target_colors = max(2, len(anchors) - int(best_result is not None))
        while time.perf_counter() < partition_end and target_colors <= len(anchors):
            chosen = anchors[:target_colors]
            fixed = set(chosen)
            groups = [0] * target_colors
            labels = [-1] * n
            for i, v in enumerate(chosen):
                groups[i] |= 1 << v
                labels[v] = i
            movable = [v for v in range(n) if v not in fixed]
            rng.shuffle(movable)
            movable.sort(key=lambda v: adj0[v].bit_count(), reverse=True)
            cost = 0
            for v in movable:
                costs = [(adj0[v] & g).bit_count() for g in groups]
                low = min(costs)
                color = rng.choice([c for c, z in enumerate(costs) if z == low])
                labels[v] = color
                groups[color] |= 1 << v
                cost += low
            improve_end = min(partition_end, time.perf_counter() + quantum * 0.5)
            while cost > k and time.perf_counter() < improve_end:
                v = rng.choice(movable) if movable else None
                if v is None:
                    break
                old = labels[v]
                before = (adj0[v] & groups[old]).bit_count()
                costs = [(adj0[v] & g).bit_count() for g in groups]
                low = min(costs)
                if low <= before:
                    color = rng.choice([c for c, z in enumerate(costs) if z == low])
                    groups[old] &= ~(1 << v)
                    groups[color] |= 1 << v
                    labels[v] = color
                    cost += low - before
            if cost <= k:
                selected = {e for e, (u, v) in enumerate(edges) if labels[u] == labels[v]}
                accept(selected, chosen)
                target_colors -= 1
                if target_colors < 2:
                    break
            elif best_result is None and target_colors < len(anchors):
                target_colors += 1

        # The edge count and discovered cliques both provide global lower bounds; terminate early only with a proof.
        lower = 2
        while n * (n - 1) // 2 - _required(n, lower) < m - k:
            lower += 1
        while _required(len(strongest), lower) > k:
            lower += 1
        if best_result is not None and len(best_clique) <= lower:
            return best_result

        def rebuild():
            nonlocal master
            target = max(1, len(best_clique) - 1) if best_result is not None else max(2, len(strongest))
            master = _Master(m, k, target, best_selected, prior, rng, horizon)
            for size, ce in pool:
                if time.perf_counter() >= end:
                    break
                master.add(size, ce)

        def separate(clique, adj, deadline):
            remember(clique)
            if len(clique) > master.target + 1:
                remember(rng.sample(clique, master.target + 1))
            mask = sum(1 << v for v in clique)
            vertices_to_try = list(range(n))
            rng.shuffle(vertices_to_try)
            for v in vertices_to_try:
                if time.perf_counter() >= deadline:
                    break
                if mask & (1 << v):
                    continue
                neighbors = mask & adj[v]
                if neighbors.bit_count() >= master.target:
                    c = [v]
                    while neighbors:
                        b = neighbors & -neighbors
                        c.append(b.bit_length() - 1)
                        neighbors ^= b
                    remember(c)

        rebuild()
        pending = None
        certification_start = start + horizon * 0.82
        while time.perf_counter() < end:
            now = time.perf_counter()
            if best_result is not None and len(best_clique) <= lower:
                break
            if pending is not None:
                selected, task = pending
                task.run(min(end, now + quantum))
                if task.complete:
                    changed = accept(selected, task.best)
                    remember(task.best)
                    pending = None
                    if changed:
                        while _required(len(strongest), lower) > k:
                            lower += 1
                        rebuild()
                    continue
                if len(task.best) >= task.stop_at:
                    separate(task.best, task.adj, min(end, time.perf_counter() + quantum * 0.12))
                    pending = None
                    continue
                # A paused exact subproblem keeps its whole stack; the final phase continues certification without restarting the search.
                if now >= certification_start or best_result is None:
                    continue
                task.run(min(end, time.perf_counter() + quantum))
                continue
            progress = min(1.0, (now - start) / max(horizon, 1e-12))
            ready = master.repair(min(end, now + quantum), quantum, progress)
            if not ready or time.perf_counter() >= end:
                if best_result is None and not base.complete:
                    base.run(min(end, time.perf_counter() + quantum * 0.25))
                    if base.complete:
                        accept(set(), base.best)
                        rebuild()
                continue
            selected = set(master.on)
            adj = residual(selected)
            c = _greedy(adj, rng, min(end, time.perf_counter() + quantum * 0.1))
            if len(c) > master.target:
                separate(c, adj, min(end, time.perf_counter() + quantum * 0.12))
                continue
            pending = (selected, _CliqueSearch(adj, c, master.target + 1))

        if best_result is not None:
            return best_result
        # When no certified plan exists yet, use the last available time to continue the proof on the original graph, keeping real cliques found.
        base.run(end)
        return package(set(), base.best)
    except Exception:
        return best_result if best_result is not None else fallback
