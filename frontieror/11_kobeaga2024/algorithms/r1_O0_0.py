"""Adaptive destroy-and-repair with route neighbourhood search
Combines randomised insertion, 2-opt, consecutive node relocation and visited-node replacement, uses subset dynamic programming for small instances, and stores the best feasible closed tour.
"""

import math
import time
import numpy as np


class _Search:
    def __init__(self, instance, seconds, started):
        self.n = int(instance['n_locations'])
        self.r = int(instance['depot_index'])
        self.budget = float(instance['distance_budget'])
        self.p = np.asarray(instance['scores'], dtype=float)
        self.d = np.asarray(instance['distance_matrix'], dtype=float)
        self.seconds = max(0.0, float(seconds))
        self.start = started
        self.end = started + self.seconds - min(self.seconds * .06, .008 + self.seconds * .012)
        self.rng = np.random.default_rng(73129)
        self.best = []
        self.best_value = -math.inf
        self.best_length = math.inf
        self.last_gain = started
        self.scale = max(1.e-12, abs(self.budget) / max(1, self.n))
        self.eps = self.scale * 1.e-10
        self.strength = math.log1p(self.seconds)
        self.pool_size = max(2, int(2 + 2 * math.sqrt(self.seconds) + math.log1p(self.n)))
        self.pool = []
        self.seeds = []
        self.weights = np.ones(5, dtype=float)

    def length(self, route):
        return math.fsum(float(self.d[route[i - 1], route[i]]) for i in range(len(route)))

    def value(self, route):
        return math.fsum(float(self.p[v]) for v in route)

    def save(self, route):
        if len(route) < 3:
            return False
        value = self.value(route)
        if value < self.best_value:
            return False
        length = self.length(route)
        if length > self.budget:
            return False
        if value > self.best_value or (value == self.best_value and length < self.best_length):
            if value > self.best_value:
                self.last_gain = time.perf_counter()
            self.best = list(route)
            self.best_value = value
            self.best_length = length
            return True
        return False

    def initial(self):
        # Meetings of different sources in multi-source shortest paths form simple cycles, without relying on the triangle inequality.
        n, r, d = self.n, self.r, self.d
        if n < 3:
            return
        dist = d[r].copy()
        dist[r] = math.inf
        labels = np.arange(n)
        parents = np.full(n, -1, dtype=int)
        settled = np.zeros(n, dtype=bool)
        settled[r] = True
        shortest = math.inf
        shortest_route = None
        while not np.all(settled):
            if self.best and time.perf_counter() >= self.end:
                break
            v = int(np.argmin(np.where(settled, math.inf, dist)))
            if not math.isfinite(float(dist[v])):
                break
            cross = (labels != labels[v])
            cross[r] = False
            cross[v] = False
            costs = dist[v] + d[v] + dist
            costs[~cross] = math.inf
            u = int(np.argmin(costs))
            if float(costs[u]) < shortest:
                left, right = [], []
                x = v
                while x != -1:
                    left.append(x)
                    x = int(parents[x])
                x = u
                while x != -1:
                    right.append(x)
                    x = int(parents[x])
                route = [r] + left[::-1] + right
                if len(set(route)) == len(route):
                    length = self.length(route)
                    if length < shortest:
                        shortest = length
                        shortest_route = route
                        if length <= self.budget:
                            self.save(route)
            settled[v] = True
            candidate = dist[v] + d[v]
            improve = (~settled) & (candidate < dist)
            dist[improve] = candidate[improve]
            labels[improve] = labels[v]
            parents[improve] = v
        if shortest_route is not None and shortest <= self.budget:
            self.seeds.append(shortest_route)
            self.save(shortest_route)
        if not self.best:
            return
        # Pick high-profit triangles for different depot neighbours, giving starting points in different directions.
        order = np.argsort(-(self.p / np.maximum(d[r], self.scale * .01)))
        target = max(2, int(self.strength + math.sqrt(self.n)))
        for v in order:
            if time.perf_counter() >= self.end or len(self.seeds) >= target:
                break
            v = int(v)
            if v == r:
                continue
            costs = d[r, v] + d[v] + d[:, r]
            feasible = costs <= self.budget
            feasible[r] = feasible[v] = False
            if not np.any(feasible):
                continue
            merit = (self.p[v] + self.p) / np.maximum(costs, self.scale * .001) ** .65
            merit[~feasible] = -math.inf
            u = int(np.argmax(merit))
            route = [r, v, u]
            self.seeds.append(route)
            self.save(route)

    def optimize(self, route, deadline, deep=True):
        # All neighbourhoods use true distance increments; only strict shortening is accepted.
        route = list(route)
        d, eps = self.d, self.eps
        while len(route) > 3 and time.perf_counter() < deadline:
            a = np.asarray(route, dtype=int)
            k = len(route)
            b = np.roll(a, -1)
            costs = d[a, b]
            delta = d[a[:, None], a] + d[b[:, None], b] - costs[:, None] - costs
            delta[np.tril_indices(k, 1)] = math.inf
            delta[0, -1] = math.inf
            flat = int(np.argmin(delta))
            i, j = divmod(flat, k)
            if delta[i, j] < -eps:
                route[i + 1:j + 1] = route[i + 1:j + 1][::-1]
                continue
            if time.perf_counter() >= deadline:
                break
            prev = np.roll(a, 1)
            saving = d[prev, a] + d[a, b] - d[prev, b]
            delta = d[a[:, None], a] + d[a[:, None], b] - costs - saving[:, None]
            idx = np.arange(k)
            delta[0, :] = math.inf
            delta[idx, idx] = math.inf
            delta[idx, (idx - 1) % k] = math.inf
            flat = int(np.argmin(delta))
            i, j = divmod(flat, k)
            if delta[i, j] < -eps:
                v = route.pop(i)
                route.insert(j + 1 if j < i else j, v)
                continue
            if not deep or k < 5 or time.perf_counter() >= deadline:
                break
            # Keep or reverse two adjacent non-depot nodes, then move them into another edge.
            first, last = a[1:-1], a[2:]
            pre, post = a[:-2], np.concatenate((a[3:], a[:1]))
            saved = d[pre, first] + d[first, last] + d[last, post] - d[pre, post]
            inner = d[first, last]
            normal = d[first[:, None], a] + d[last[:, None], b]
            reverse = d[last[:, None], a] + d[first[:, None], b]
            delta = np.minimum(normal, reverse) + inner[:, None] - costs - saved[:, None]
            rows = np.arange(k - 2)
            delta[rows, rows] = math.inf
            delta[rows, rows + 1] = math.inf
            delta[rows, rows + 2] = math.inf
            flat = int(np.argmin(delta))
            row, j = divmod(flat, k)
            if delta[row, j] >= -eps:
                break
            i = row + 1
            group = route[i:i + 2]
            if reverse[row, j] < normal[row, j]:
                group.reverse()
            del route[i:i + 2]
            pos = j + 1 if j < i else j - 1
            route[pos:pos] = group
        return route

    def fill(self, route, deadline, randomized):
        route = list(route)
        available = np.ones(self.n, dtype=bool)
        available[route] = False
        length = self.length(route)
        if randomized:
            beta = math.exp(float(self.rng.normal(0, .28 + .08 * self.strength)))
            utility = self.p * np.exp(self.rng.normal(0, .12 + .05 * self.strength, self.n))
        else:
            beta, utility = 1.0, self.p
        while time.perf_counter() < deadline:
            nodes = np.flatnonzero(available)
            if not len(nodes):
                break
            a = np.asarray(route, dtype=int)
            b = np.roll(a, -1)
            increments = self.d[nodes[:, None], a] + self.d[nodes[:, None], b] - self.d[a, b]
            where = np.argmin(increments, axis=1)
            delta = increments[np.arange(len(nodes)), where]
            fits = delta <= self.budget - length
            if not np.any(fits):
                break
            free = fits & (delta <= self.eps)
            if np.any(free):
                merit = np.where(free, utility[nodes] + np.maximum(0, -delta) / self.scale, -math.inf)
            else:
                # Logarithmic form avoids numerical overflow from tiny distances and random exponents.
                merit = np.log(utility[nodes]) - beta * np.log(np.maximum(delta, self.scale * 1.e-9))
                merit[~fits] = -math.inf
            row = int(np.argmax(merit))
            node = int(nodes[row])
            route.insert(int(where[row]) + 1, node)
            available[node] = False
            length = self.length(route)
        return route

    def exchange(self, route, deadline):
        # After removing a node, allow the new node to be inserted at any position, not only replacing the original position.
        route = list(route)
        if len(route) < 3 or time.perf_counter() >= deadline:
            return route, False
        a = np.asarray(route, dtype=int)
        k = len(a)
        available = np.ones(self.n, dtype=bool)
        available[a] = False
        nodes = np.flatnonzero(available)
        if not len(nodes):
            return route, False
        d = self.d
        prev, nxt = np.roll(a, 1), np.roll(a, -1)
        edge = d[a, nxt]
        ins = d[nodes[:, None], a] + d[nodes[:, None], nxt] - edge
        # Removing one node invalidates at most two edges, so the top three insertion positions suffice to find the best remaining position.
        top = np.argpartition(ins, 2, axis=1)[:, :3]
        vals = np.take_along_axis(ins, top, axis=1)
        ranks = np.argsort(vals, axis=1)
        top = np.take_along_axis(top, ranks, axis=1)
        vals = np.take_along_axis(vals, ranks, axis=1)
        positions = np.arange(k)
        best_insert = np.full((len(nodes), k), math.inf)
        best_edge = np.full((len(nodes), k), -1, dtype=int)
        for rank in (2, 1, 0):
            j = top[:, rank, None]
            legal = (j != positions) & (j != (positions - 1) % k)
            best_insert = np.where(legal, vals[:, rank, None], best_insert)
            best_edge = np.where(legal, j, best_edge)
        merged = d[nodes[:, None], prev] + d[nodes[:, None], nxt] - d[prev, nxt]
        use_merge = merged < best_insert
        best_insert = np.minimum(best_insert, merged)
        saving = d[prev, a] + d[a, nxt] - d[prev, nxt]
        delta = best_insert - saving
        gains = self.p[nodes, None] - self.p[a]
        legal = (delta <= self.budget - self.length(route)) & ((gains > 0) | ((gains == 0) & (delta < -self.eps)))
        legal[:, 0] = False
        if not np.any(legal):
            return route, False
        max_gain = float(np.max(np.where(legal, gains, -math.inf)))
        choices = legal & (gains == max_gain)
        flat = int(np.argmin(np.where(choices, delta, math.inf)))
        row, i = divmod(flat, k)
        node = int(nodes[row])
        if use_merge[row, i]:
            route[i] = node
        else:
            j = int(best_edge[row, i])
            route.pop(i)
            route.insert(j + 1 if j < i else j, node)
        return route, True

    def repair(self, route, deadline):
        route = list(route)
        while len(route) > 3 and time.perf_counter() < deadline:
            length = self.length(route)
            if length <= self.budget:
                return route
            a = np.asarray(route, dtype=int)
            prev, nxt = np.roll(a, 1), np.roll(a, -1)
            saved = self.d[prev, a] + self.d[a, nxt] - self.d[prev, nxt]
            merit = np.where(saved > self.eps, self.p[a] / np.maximum(saved, self.eps), math.inf)
            merit[0] = math.inf
            pos = int(np.argmin(merit))
            if not math.isfinite(float(merit[pos])):
                break
            route.pop(pos)
        return route

    def improve(self, route, deadline, randomized):
        route = self.optimize(route, deadline, deep=False)
        route = self.fill(route, deadline, randomized)
        if self.length(route) > self.budget:
            route = self.repair(route, deadline)
        while time.perf_counter() < deadline:
            before = (tuple(route), self.length(route))
            route = self.optimize(route, deadline)
            route = self.fill(route, deadline, randomized)
            self.save(route)
            if time.perf_counter() >= deadline:
                break
            route, changed = self.exchange(route, deadline)
            self.save(route)
            if not changed and (tuple(route), self.length(route)) == before:
                break
        self.save(route)
        return route

    def archive(self, route):
        length = self.length(route)
        if len(route) < 3 or length > self.budget:
            return
        key = frozenset(route)
        value = self.value(route)
        for i, entry in enumerate(self.pool):
            if entry[3] == key:
                if length < entry[1]:
                    self.pool[i] = (value, length, list(route), key)
                return
        self.pool.append((value, length, list(route), key))
        if len(self.pool) > self.pool_size:
            best_set = frozenset(self.best)
            # Besides quality, keep routes whose visited sets differ from that of the current best solution.
            def merit(entry):
                novelty = len(entry[3] ^ best_set) / max(1, len(entry[3] | best_set))
                return entry[0] + .07 * abs(self.best_value) * novelty
            victim = min(range(len(self.pool)), key=lambda i: (merit(self.pool[i]), -self.pool[i][1]))
            self.pool.pop(victim)

    def destroy(self, route, operator, fraction):
        route = list(route)
        k = len(route)
        if k <= 3:
            return list(self.seeds[int(self.rng.integers(len(self.seeds)))])
        maximum = max(1, min(k - 3, int(math.ceil((k - 1) * fraction))))
        q = int(self.rng.integers(1, maximum + 1))
        candidates = np.asarray(route[1:], dtype=int)
        if operator == 0:
            removed = set(int(v) for v in self.rng.choice(candidates, q, replace=False))
        elif operator == 1:
            start = int(self.rng.integers(k - 1))
            removed = {int(candidates[(start + j) % (k - 1)]) for j in range(q)}
        elif operator == 2:
            removed = set()
            while len(removed) < q:
                a = np.asarray(route, dtype=int)
                prev, nxt = np.roll(a, 1), np.roll(a, -1)
                saved = self.d[prev, a] + self.d[a, nxt] - self.d[prev, nxt]
                badness = saved / np.maximum(self.p[a], 1.e-100)
                badness *= self.rng.uniform(.65, 1.35, len(a))
                badness[0] = -math.inf
                pos = int(np.argmax(badness))
                removed.add(route.pop(pos))
                if time.perf_counter() >= self.end:
                    break
            return route
        elif operator == 3:
            anchor = int(self.rng.choice(candidates))
            metric = self.d[anchor, candidates] * self.rng.uniform(.5, 1.5, len(candidates))
            removed = set(int(v) for v in candidates[np.argsort(metric)[:q]])
        else:
            a = np.asarray(route, dtype=int)
            badness = self.d[np.roll(a, 1), a] + self.d[a, np.roll(a, -1)]
            badness[0] = 0
            badness *= self.rng.uniform(.4, 1.6, len(a))
            removed = set(int(v) for v in a[np.argsort(-badness)[:q]])
        return [v for v in route if v not in removed]

    def exact(self):
        m = self.n - 1
        if m < 2:
            return False
        # The state space is controlled by remaining time and an explicit memory limit, not by a fixed node count.
        if m * math.log(2) + math.log(max(1, m)) > math.log(max(1, self.seconds * 65000)):
            return False
        states = 1 << m
        if states * m * 12 > 64 * 1024 * 1024:
            return False
        nodes = [v for v in range(self.n) if v != self.r]
        nodes.sort(key=lambda v: -float(self.p[v] / max(self.d[self.r, v], self.scale * .01)))
        ids = np.asarray(nodes, dtype=int)
        dist = self.d[np.ix_(ids, ids)]
        returning = self.d[ids, self.r]
        dp = np.full((states, m), math.inf)
        parent = np.full((states, m), -1, dtype=np.int16)
        values = np.zeros(states)
        for j, v in enumerate(nodes):
            dp[1 << j, j] = self.d[self.r, v]
        mask = 1
        while mask < states and time.perf_counter() < self.end:
            low = mask & -mask
            values[mask] = values[mask ^ low] + self.p[nodes[low.bit_length() - 1]]
            if mask != low:
                bits = mask
                while bits:
                    bit = bits & -bits
                    j = bit.bit_length() - 1
                    prev = mask ^ bit
                    costs = dp[prev] + dist[:, j]
                    last = int(np.argmin(costs))
                    dp[mask, j] = costs[last]
                    parent[mask, j] = last
                    bits ^= bit
                if values[mask] + self.p[self.r] >= self.best_value - self.eps:
                    costs = dp[mask] + returning
                    last = int(np.argmin(costs))
                    if costs[last] <= self.budget:
                        route, rest = [], mask
                        while rest:
                            route.append(nodes[last])
                            previous = int(parent[rest, last])
                            rest ^= 1 << last
                            last = previous
                        self.save([self.r] + route[::-1])
            mask += 1
        return mask == states

    def run(self):
        self.initial()
        if not self.best or time.perf_counter() >= self.end:
            return
        quantum = max(.001, self.seconds / (12 + 2 * math.sqrt(self.n)))
        current = self.improve(self.seeds[0], min(self.end, time.perf_counter() + quantum), False)
        if self.length(current) > self.budget:
            current = list(self.best)
        self.archive(current)
        if len(self.best) == self.n:
            return
        if self.exact() or time.perf_counter() >= self.end:
            return
        while time.perf_counter() < self.end:
            now = time.perf_counter()
            stale = min(1.0, (now - self.last_gain) / max(.001, self.seconds * .20))
            draw = float(self.rng.random())
            if draw < .12 + .12 * stale:
                base = list(self.seeds[int(self.rng.integers(len(self.seeds)))])
            elif draw < .35 and self.pool:
                base = list(self.pool[int(self.rng.integers(len(self.pool)))][2])
            elif draw < .60:
                base = list(self.best)
            else:
                base = list(current)
            operator = int(self.rng.choice(len(self.weights), p=self.weights / self.weights.sum()))
            exploration = self.strength / (1 + self.strength)
            fraction = .08 + exploration * (.25 + .50 * stale)
            candidate = self.destroy(base, operator, fraction)
            # Even when the visited set is unchanged, route perturbation can escape 2-opt local optima.
            if len(candidate) > 5 and self.rng.random() < .12 * exploration:
                cut = np.sort(self.rng.choice(np.arange(1, len(candidate)), 3, replace=False))
                i, j, k = (int(v) for v in cut)
                candidate = candidate[:i] + candidate[j:k] + candidate[i:j] + candidate[k:]
            old_best = self.best_value
            deadline = min(self.end, time.perf_counter() + quantum * (1 + exploration + stale))
            candidate = self.improve(candidate, deadline, True)
            length = self.length(candidate)
            self.weights[operator] = .97 * self.weights[operator] + .03
            if length <= self.budget and len(candidate) >= 3:
                self.archive(candidate)
                value = self.value(candidate)
                delta = value - self.value(current)
                remaining = max(0.0, (self.end - time.perf_counter()) / max(self.seconds, .001))
                average = max(self.scale * 1.e-9, (self.best_value - self.p[self.r]) / max(1, len(self.best) - 1))
                temperature = average * (.025 + .35 * remaining * exploration) * (1 + stale)
                if delta >= 0 or self.rng.random() < math.exp(max(-700.0, delta / temperature)):
                    current = candidate
                    self.weights[operator] += .05
                if self.best_value > old_best:
                    self.weights[operator] += 1
                if len(self.best) == self.n:
                    return


def solve(instance: dict, time_limit_s: float) -> dict:
    started = time.perf_counter()
    n = int(instance['n_locations'])
    depot = int(instance['depot_index'])
    result = {'visited': [0] * n, 'selected_links': [[0] * n for _ in range(n)]}
    if 0 <= depot < n:
        result['visited'][depot] = 1
    search = None
    try:
        search = _Search(instance, time_limit_s, started)
        search.run()
    except Exception:
        # When the search is interrupted, output only historical feasible solutions that passed the true-distance check.
        pass
    if search is not None and search.best:
        route = search.best
        for i, v in enumerate(route):
            u = route[i - 1]
            result['visited'][v] = 1
            result['selected_links'][u][v] = 1
            result['selected_links'][v][u] = 1
    return result
