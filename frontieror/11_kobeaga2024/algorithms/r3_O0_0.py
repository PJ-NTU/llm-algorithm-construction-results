"""Adaptive destroy-repair and closed-tour neighbourhood search
Combines shortest-closed-tour initialisation, incremental profit insertion, route reordering and node swaps, and uses subset dynamic programming on suitable small instances.
"""

import math
import random
import time
from array import array

import numpy as np


class _Search:
    def __init__(self, instance, limit, started):
        self.n = int(instance["n_locations"])
        self.r = int(instance["depot_index"])
        self.budget = float(instance["distance_budget"])
        self.d = np.asarray(instance["distance_matrix"], dtype=float)
        self.p = np.asarray(instance["scores"], dtype=float)
        self.limit = max(0.0, float(limit))
        self.start = started
        self.end = started + self.limit - min(0.1, 0.008 + self.limit * 0.003)
        self.rng = random.Random(20261002)
        self.nrng = np.random.default_rng(20261002)
        self.best = []
        self.best_score = -math.inf
        self.best_length = math.inf
        self.improved = started
        self.scale = max(self.budget / max(3, self.n), 1e-12)
        self.eps = max(1e-12, abs(self.budget) * 2e-14)
        self.depth = max(1, int(math.ceil(math.log1p(self.limit))))
        self.pool_cap = max(2, int(math.ceil(math.log1p(self.n) * math.sqrt(math.log1p(self.limit) + 1))))
        self.pool = []
        self.work = None
        self.rows = np.arange(self.n)
        self.slice = max(0.001, self.limit / (10.0 + math.sqrt(self.n) * math.log1p(self.n)))

    def length(self, route):
        return math.fsum(float(self.d[route[i - 1], v]) for i, v in enumerate(route))

    def score(self, route):
        return math.fsum(float(self.p[v]) for v in route)

    def offer(self, route, pool=False):
        if len(route) < 3:
            return False
        length = self.length(route)
        if length > self.budget:
            return False
        value = self.score(route)
        if value > self.best_score or (value == self.best_score and length < self.best_length):
            if value > self.best_score:
                self.improved = time.perf_counter()
            self.best = route[:]
            self.best_score = value
            self.best_length = length
        if pool:
            key = frozenset(route)
            for z, entry in enumerate(self.pool):
                if entry[3] == key:
                    if length < entry[1]:
                        self.pool[z] = (value, length, route[:], key)
                    return True
            self.pool.append((value, length, route[:], key))
            self.pool.sort(key=lambda x: (-x[0], x[1]))
            if len(self.pool) > self.pool_cap:
                self.pool.pop()
        return True

    def seed(self):
        # The simple undirected binary-edge model needs at least three distinct nodes.
        n, r, d = self.n, self.r, self.d
        if n < 3:
            return []
        radial = d[r]
        costs = d + radial[:, None] + radial[None, :]
        np.fill_diagonal(costs, np.inf)
        costs[r, :] = np.inf
        costs[:, r] = np.inf
        at = int(np.argmin(costs))
        a, b = divmod(at, n)
        route = [r, a, b]
        if self.offer(route):
            return route

        # Under non-metric distances the shortest feasible closed tour may not be a triangle.
        # After removing the depot, run dense multi-source shortest paths with each depot neighbour as a distinct source.
        dist = radial.copy()
        dist[r] = np.inf
        label = np.arange(n)
        parent = np.full(n, -1, dtype=int)
        done = np.zeros(n, dtype=bool)
        done[r] = True
        shortest = math.inf
        found = []
        while not bool(np.all(done)):
            u = int(np.argmin(np.where(done, np.inf, dist)))
            if not math.isfinite(float(dist[u])):
                break
            done[u] = True
            crossing = dist[u] + d[u] + dist
            crossing[(label == label[u]) | (self.rows == r)] = np.inf
            v = int(np.argmin(crossing))
            if float(crossing[v]) < shortest:
                left, right = [], []
                x = u
                while x >= 0:
                    left.append(int(x))
                    x = int(parent[x])
                x = v
                while x >= 0:
                    right.append(int(x))
                    x = int(parent[x])
                candidate = [r] + left[::-1] + right
                length = self.length(candidate)
                if len(candidate) == len(set(candidate)) and length < shortest:
                    shortest, found = length, candidate
                    self.offer(candidate)
            relax = dist[u] + d[u]
            change = (~done) & (relax < dist)
            dist[change] = relax[change]
            parent[change] = u
            label[change] = label[u]
            if self.best and time.perf_counter() >= self.end:
                break
        return self.best[:] if self.best else found

    def grow(self, route, stop, exponent=1.0, noise=0.0, discouraged=()):
        if time.perf_counter() >= stop or len(route) == self.n:
            return route
        n, d, p = self.n, self.d, self.p
        k = len(route)
        length = self.length(route)
        if length > self.budget:
            return route
        if self.work is None:
            self.work = np.empty((n, n), dtype=float)
        c = self.work
        aa = np.empty(n, dtype=int)
        bb = np.empty(n, dtype=int)
        aa[:k] = route
        bb[:k] = route[1:] + route[:1]
        c[:, :k] = d[:, aa[:k]] + d[:, bb[:k]] - d[aa[:k], bb[:k]]
        arg = np.argmin(c[:, :k], axis=1)
        delta = c[self.rows, arg].copy()
        active = np.ones(n, dtype=bool)
        active[route] = False
        succ = [-1] * n
        for a, b in zip(route, route[1:] + route[:1]):
            succ[a] = b
        weight = p.copy()
        if noise > 0:
            weight *= np.exp(self.nrng.uniform(-noise, noise, n))
        if discouraged:
            weight[list(discouraged)] *= math.exp(-noise)
        floor = self.scale * 0.015
        while k < n and time.perf_counter() < stop:
            feasible = active & (delta <= self.budget - length - self.eps)
            if not bool(np.any(feasible)):
                break
            merit = weight / np.maximum(delta, floor) ** exponent
            # Negative-increment insertions both add score and free budget, so handle them first.
            negative = feasible & (delta < -self.eps)
            if bool(np.any(negative)):
                merit = np.where(negative, weight * (1.0 - delta / self.scale), -np.inf)
            else:
                merit[~feasible] = -np.inf
            v = int(np.argmax(merit))
            h = int(arg[v])
            a, b = int(aa[h]), int(bb[h])
            length += float(delta[v])
            succ[a], succ[v] = v, b
            active[v] = False
            bb[h] = v
            aa[k], bb[k] = v, b
            c[:, h] = d[:, a] + d[:, v] - d[a, v]
            c[:, k] = d[:, v] + d[:, b] - d[v, b]
            broken = active & (arg == h)
            affected = self.rows[broken]
            if len(affected):
                new_arg = np.argmin(c[affected, :k + 1], axis=1)
                arg[affected] = new_arg
                delta[affected] = c[affected, new_arg]
            for col in (h, k):
                better = active & (c[:, col] < delta)
                delta[better] = c[better, col]
                arg[better] = col
            k += 1
        result = [self.r]
        v = succ[self.r]
        while v != self.r and v >= 0 and len(result) < k:
            result.append(v)
            v = succ[v]
        # Float accumulation is only used during search; before committing, re-sum over the original edges.
        return result if self.length(result) <= self.budget else route

    def shorten(self, route, stop):
        k = len(route)
        if k < 4 or time.perf_counter() >= stop:
            return route
        d = self.d
        ii, jj = np.triu_indices(k, 2)
        keep = ~((ii == 0) & (jj == k - 1))
        ii, jj = ii[keep], jj[keep]
        positions = np.arange(k)
        while time.perf_counter() < stop:
            rr = np.asarray(route, dtype=int)
            nxt = np.roll(rr, -1)
            edge = d[rr, nxt]
            gain = d[rr[ii], rr[jj]] + d[nxt[ii], nxt[jj]] - edge[ii] - edge[jj]
            if len(gain):
                at = int(np.argmin(gain))
                if float(gain[at]) < -self.eps:
                    i, j = int(ii[at]), int(jj[at])
                    route = route[:i + 1] + route[i + 1:j + 1][::-1] + route[j + 1:]
                    continue
            moved = False
            # Relocate consecutive blocks that do not cross the depot; block length is determined by available time.
            for block in range(1, min(self.depth, k - 2) + 1):
                if time.perf_counter() >= stop:
                    break
                starts = np.arange(1, k - block + 1)
                ends = starts + block - 1
                first, last = rr[starts], rr[ends]
                prev, after = rr[starts - 1], rr[(ends + 1) % k]
                saving = d[prev, first] + d[last, after] - d[prev, after]
                change = d[first[:, None], rr[None, :]] + d[last[:, None], nxt[None, :]]
                change -= edge[None, :] + saving[:, None]
                forbidden = (positions[None, :] >= starts[:, None] - 1) & (positions[None, :] <= ends[:, None])
                change[forbidden] = np.inf
                at = int(np.argmin(change))
                q, h = divmod(at, k)
                if float(change[q, h]) < -self.eps:
                    s = int(starts[q])
                    segment = route[s:s + block]
                    rest = route[:s] + route[s + block:]
                    insert = h + 1 if h < s else h + 1 - block
                    route = rest[:insert] + segment + rest[insert:]
                    moved = True
                    break
            if not moved:
                break
        return route

    def exchange(self, route, stop):
        if time.perf_counter() >= stop or len(route) == self.n:
            return route, False
        d, p = self.d, self.p
        rr = np.asarray(route, dtype=int)
        k = len(rr)
        active = np.ones(self.n, dtype=bool)
        active[rr] = False
        uu = self.rows[active]
        nxt = np.roll(rr, -1)
        ins = d[uu[:, None], rr[None, :]] + d[uu[:, None], nxt[None, :]] - d[rr, nxt]
        # Removing one node breaks at most two old edges, so the top three insertion positions suffice.
        top = np.argpartition(ins, 2, axis=1)[:, :3]
        removed = np.arange(1, k)
        cost = np.full((len(uu), k - 1), np.inf)
        where = np.full((len(uu), k - 1), -1, dtype=int)
        row = np.arange(len(uu))
        for rank in range(3):
            edge = top[:, rank]
            values = ins[row, edge][:, None]
            valid = (edge[:, None] != removed[None, :]) & (edge[:, None] != removed[None, :] - 1)
            better = valid & (values < cost)
            cost = np.where(better, values, cost)
            where = np.where(better, edge[:, None], where)
        before, node, after = rr[removed - 1], rr[removed], rr[(removed + 1) % k]
        saving = d[before, node] + d[node, after] - d[before, after]
        cost -= saving[None, :]
        replace = d[uu[:, None], before[None, :]] + d[uu[:, None], after[None, :]]
        replace -= (d[before, node] + d[node, after])[None, :]
        direct = replace < cost
        cost = np.minimum(cost, replace)
        prize = p[uu, None] - p[node][None, :]
        feasible = (prize > 0) & (cost <= self.budget - self.length(route) - self.eps)
        if not bool(np.any(feasible)):
            return route, False
        gain = np.where(feasible, prize, -np.inf)
        maximum = float(np.max(gain))
        at = int(np.argmin(np.where(gain == maximum, cost, np.inf)))
        a, b = divmod(at, k - 1)
        q = int(removed[b])
        v = int(uu[a])
        if bool(direct[a, b]):
            result = route[:]
            result[q] = v
        else:
            h = int(where[a, b])
            result = route[:q] + route[q + 1:]
            pos = h + 1 if h < q else h
            result.insert(pos, v)
        if self.length(result) <= self.budget:
            return result, True
        return route, False

    def polish(self, route, stop):
        while time.perf_counter() < stop:
            route = self.shorten(route, stop)
            route = self.grow(route, stop)
            self.offer(route)
            if len(route) == self.n:
                break
            route, changed = self.exchange(route, stop)
            if not changed:
                break
        self.offer(route)
        return route

    def restore(self, route, stop):
        if self.length(route) <= self.budget:
            return route
        route = self.shorten(route, stop)
        while len(route) > 3 and time.perf_counter() < stop:
            if self.length(route) <= self.budget:
                return route
            rr = np.asarray(route)
            saving = self.d[np.roll(rr, 1), rr] + self.d[rr, np.roll(rr, -1)] - self.d[np.roll(rr, 1), np.roll(rr, -1)]
            saving[0] = -np.inf
            valid = saving > self.eps
            if not bool(np.any(valid)):
                break
            ratio = np.where(valid, self.p[rr] / np.maximum(saving, self.eps), np.inf)
            route.pop(int(np.argmin(ratio)))
        return route

    def destroy(self, route, amount, mode):
        k = len(route)
        if amount <= 0:
            return route[:], set()
        if mode == 0:
            removed = set(self.rng.sample(route[1:], amount))
        elif mode == 1:
            start = self.rng.randrange(k - 1)
            removed = {route[1 + (start + h) % (k - 1)] for h in range(amount)}
        elif mode == 2:
            active = np.ones(self.n, dtype=bool)
            active[route] = False
            missing = self.rows[active]
            anchor = int(self.rng.choice(missing)) if len(missing) else self.rng.choice(route[1:])
            nodes = np.asarray(route[1:])
            metric = self.d[anchor, nodes] * self.nrng.uniform(0.65, 1.35, len(nodes))
            removed = set(int(x) for x in nodes[np.argsort(metric)[:amount]])
        else:
            result = route[:]
            removed = set()
            while len(removed) < amount:
                rr = np.asarray(result)
                saving = self.d[np.roll(rr, 1), rr] + self.d[rr, np.roll(rr, -1)] - self.d[np.roll(rr, 1), np.roll(rr, -1)]
                metric = self.p[rr] / np.maximum(saving, self.scale * 0.02)
                metric *= self.nrng.uniform(0.45, 1.55, len(rr))
                metric[0] = np.inf
                q = int(np.argmin(metric))
                removed.add(result.pop(q))
            return result, removed
        return [v for v in route if v not in removed], removed

    def new_seed(self):
        r = self.r
        missing = self.rows[self.rows != r]
        weights = self.p[missing] / np.maximum(self.d[r, missing], self.scale * 0.05) ** self.rng.uniform(0, 1)
        weights /= np.sum(weights)
        a = int(self.nrng.choice(missing, p=weights))
        length = self.d[r, a] + self.d[a] + self.d[r]
        valid = (length <= self.budget) & (self.rows != r) & (self.rows != a)
        if not bool(np.any(valid)):
            return self.best[:]
        merit = (self.p[a] + self.p) * self.nrng.uniform(0.5, 1.5, self.n) / np.maximum(length, self.scale * 0.05)
        merit[~valid] = -np.inf
        b = int(np.argmax(merit))
        return [r, a, b]

    def exact(self):
        m = self.n - 1
        remaining = self.end - time.perf_counter()
        # Decide by state-transition workload and remaining wall-clock time, rather than switching only by a fixed instance size.
        if m > 19 or m < 2:
            return False
        count = 1 << m
        work = m * (m - 1) * count / 4
        if work > max(0.0, remaining) * 500000:
            return False
        vertices = [v for v in range(self.n) if v != self.r]
        bit_index = {1 << j: j for j in range(m)}
        dp = array('d', [math.inf]) * (count * m)
        totals = array('d', [0.0]) * count
        distances = [[float(self.d[a, b]) for b in vertices] for a in vertices]
        radial = [float(self.d[self.r, a]) for a in vertices]
        prizes = [float(self.p[a]) for a in vertices]
        depot_prize = float(self.p[self.r])
        for mask in range(1, count):
            if (mask & 63) == 1 and time.perf_counter() >= self.end:
                return False
            bit = mask & -mask
            j = bit_index[bit]
            previous = mask ^ bit
            totals[mask] = totals[previous] + prizes[j]
            base = mask * m
            bits = mask
            closing, last = math.inf, -1
            while bits:
                bit = bits & -bits
                bits ^= bit
                j = bit_index[bit]
                previous = mask ^ bit
                if previous == 0:
                    value = radial[j]
                else:
                    row = previous * m
                    rest = previous
                    value = math.inf
                    while rest:
                        h = rest & -rest
                        rest ^= h
                        q = bit_index[h]
                        candidate = dp[row + q] + distances[q][j]
                        if candidate < value:
                            value = candidate
                if value <= self.budget:
                    dp[base + j] = value
                    candidate = value + radial[j]
                    if candidate < closing:
                        closing, last = candidate, j
            if (mask & (mask - 1)) and closing <= self.budget and totals[mask] + depot_prize >= self.best_score - 1e-10:
                route, rest, j = [], mask, last
                while rest:
                    route.append(vertices[j])
                    rest ^= 1 << j
                    if not rest:
                        break
                    row = rest * m
                    bits = rest
                    value, next_j = math.inf, -1
                    while bits:
                        bit = bits & -bits
                        bits ^= bit
                        q = bit_index[bit]
                        candidate = dp[row + q] + distances[q][j]
                        if candidate < value:
                            value, next_j = candidate, q
                    j = next_j
                self.offer([self.r] + route[::-1])
        return True

    def run(self):
        route = self.seed()
        if not self.best or time.perf_counter() >= self.end:
            return
        stop = min(self.end, time.perf_counter() + self.slice * (1 + math.log1p(self.n)))
        route = self.grow(route, stop)
        route = self.polish(route, stop)
        self.offer(route, True)
        if len(self.best) == self.n or self.exact():
            return
        current = self.best[:]
        current_value = self.best_score
        current_length = self.best_length
        while time.perf_counter() < self.end:
            now = time.perf_counter()
            progress = min(1.0, (now - self.start) / max(self.limit, 1e-9))
            stagnant = (now - self.improved) / max(self.slice, self.limit / (1.0 + math.log1p(self.n)))
            choose = self.rng.random()
            restart = 1.0 / (2.0 + math.sqrt(self.n) * math.log1p(self.limit))
            removed = set()
            if choose < restart:
                trial = self.new_seed()
                amount = max(1, len(current) - 3)
            else:
                if choose < 0.42:
                    parent = self.best
                elif choose < 0.85 or not self.pool:
                    parent = current
                else:
                    parent = self.rng.choice(self.pool)[2]
                k = len(parent)
                width = math.sqrt(k) * (1.0 + math.log1p(self.limit)) * (0.3 + min(1.0, stagnant))
                maximum = min(k - 3, max(1, int(math.ceil(width))))
                amount = 0 if maximum <= 0 else 1 + int(self.rng.random() ** 1.7 * maximum)
                amount = min(amount, k - 3)
                trial, removed = self.destroy(parent, amount, self.rng.randrange(4))
            stop = min(self.end, time.perf_counter() + self.slice)
            trial = self.restore(trial, stop)
            if self.length(trial) > self.budget:
                continue
            noise = self.rng.random() * (0.25 + math.log1p(self.limit) * 0.35) * (1.0 + min(stagnant, 1.0))
            exponent = self.rng.uniform(0.65, 1.5)
            trial = self.grow(trial, stop, exponent, noise, removed)
            self.offer(trial)
            trial = self.polish(trial, min(self.end, time.perf_counter() + self.slice))
            if not self.offer(trial, True):
                continue
            if len(self.best) == self.n:
                return
            value, length = self.score(trial), self.length(trial)
            temperature = float(np.mean(self.p)) * (0.05 + (1.0 - progress) ** 2) * math.sqrt(max(1, amount))
            difference = value - current_value
            accept = difference > 0 or (difference == 0 and length <= current_length)
            if not accept and temperature > 0:
                energy = difference - float(np.mean(self.p)) * max(0.0, length - current_length) / max(self.budget, self.eps)
                accept = self.rng.random() < math.exp(max(-700.0, min(0.0, energy / temperature)))
            if accept:
                current, current_value, current_length = trial, value, length


def solve(instance: dict, time_limit_s: float) -> dict:
    started = time.perf_counter()
    n = int(instance["n_locations"])
    depot = int(instance["depot_index"])
    result = {"visited": [0] * n, "selected_links": [[0] * n for _ in range(n)]}
    search = None
    try:
        search = _Search(instance, time_limit_s, started)
        search.run()
    except Exception:
        # Numerical or resource exceptions during search do not overwrite the feasible closed tour already found.
        pass
    route = search.best if search is not None else []
    if route:
        visited, links = result["visited"], result["selected_links"]
        for j, v in enumerate(route):
            visited[v] = 1
            u = route[j - 1]
            links[u][v] = links[v][u] = 1
    elif 0 <= depot < n:
        # When no feasible closed tour exists, the problem statement defines no output encoding for an infeasible status.
        result["visited"][depot] = 1
    return result
