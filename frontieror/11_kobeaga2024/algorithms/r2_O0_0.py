"""Elite-guided adaptive closed-tour neighbourhood search
Builds the initial tour with shortest paths, combines incremental insertion, segment reordering, node replacement and time-limited destroy-repair, and keeps the best feasible tour.
"""
import math
import random
import time
import numpy as np


def solve(instance: dict, time_limit_s: float) -> dict:
    started = time.perf_counter()
    n = int(instance["n_locations"])
    depot = int(instance["depot_index"])
    budget = float(instance["distance_budget"])
    result = {"visited": [0] * n,
              "selected_links": [[0] * n for _ in range(n)]}
    result["visited"][depot] = 1
    incumbent = [None, -math.inf, math.inf]
    limit = max(0.0, float(time_limit_s))
    deadline = started + limit - min(0.04, 0.012 * limit)
    try:
        _search(instance, budget, depot, n, limit, deadline, incumbent)
    except Exception:
        # Interrupting the search does not damage the already stored feasible tour.
        pass
    route = incumbent[0]
    if route is not None:
        for u in route:
            result["visited"][u] = 1
        for i, u in enumerate(route):
            v = route[(i + 1) % len(route)]
            result["selected_links"][u][v] = 1
            result["selected_links"][v][u] = 1
    return result


def _search(instance, budget, depot, n, limit, deadline, incumbent):
    if n < 3:
        return  # In this case no simple undirected closed tour exists.
    d = np.asarray(instance["distance_matrix"], dtype=float)
    p = np.asarray(instance["scores"], dtype=float)
    ids = np.arange(n)
    rng = random.Random(718921)
    now = time.perf_counter
    mean_p = max(float(np.mean(p)), 1e-100)
    eps = max(1e-13, abs(budget) * 2e-13)
    tiny = max(1e-100, abs(budget) * 1e-8)
    last_improvement = [now()]
    strength = math.log1p(limit)
    # Chunking only controls temporary matrix memory; it does not limit the number of search rounds.
    matrix_entries = 131072

    def length(route):
        return math.fsum(float(d[u, route[(i + 1) % len(route)]])
                         for i, u in enumerate(route))

    def remember(route):
        cost = length(route)
        score = math.fsum(float(p[u]) for u in route)
        if len(route) >= 3 and cost <= budget:
            if score > incumbent[1] or (score == incumbent[1] and cost < incumbent[2]):
                if score > incumbent[1]:
                    last_improvement[0] = now()
                incumbent[:] = [route[:], score, cost]
        return score, cost

    # Meeting multi-source labels yield two internally disjoint depot paths; non-metric distances are allowed.
    dist = d[depot].copy()
    dist[depot] = np.inf
    owner = ids.copy()
    parent = [-1] * n
    settled = np.zeros(n, dtype=bool)
    settled[depot] = True
    seed = None
    while not bool(np.all(settled)):
        u = int(np.argmin(np.where(settled, np.inf, dist)))
        if not math.isfinite(float(dist[u])):
            break
        settled[u] = True
        values = dist[u] + d[u] + dist
        values[(owner == owner[u]) | (ids == depot)] = np.inf
        v = int(np.argmin(values))
        if float(values[v]) <= budget + eps:
            left, right = [], []
            a = u
            while a >= 0:
                left.append(a)
                a = parent[a]
            a = v
            while a >= 0:
                right.append(a)
                a = parent[a]
            candidate = [depot] + left[::-1] + right
            if length(candidate) <= budget:
                seed = candidate
                remember(seed)
                break
        improved = (~settled) & (dist[u] + d[u] < dist)
        for v in np.flatnonzero(improved):
            parent[int(v)] = u
        dist[improved] = dist[u] + d[u, improved]
        owner[improved] = owner[u]
    if seed is None or now() >= deadline:
        return

    def polish(route, until):
        route = route[:]
        k = len(route)
        block = max(1, matrix_entries // k)
        columns = np.arange(k)[None, :]
        while now() < until:
            rr = np.asarray(route, dtype=int)
            nxt = np.roll(rr, -1)
            prev = np.roll(rr, 1)
            edges = d[rr, nxt]
            best_delta, move = -eps, None
            # Two-edge exchange.
            for start in range(0, k - 2, block):
                if now() >= until:
                    break
                rows = np.arange(start, min(k - 2, start + block))
                a, b = rr[rows], nxt[rows]
                changes = (d[a[:, None], rr] + d[b[:, None], nxt]
                           - edges[rows, None] - edges[None, :])
                invalid = columns <= rows[:, None] + 1
                invalid |= (rows[:, None] == 0) & (columns == k - 1)
                changes[invalid] = np.inf
                flat = int(np.argmin(changes))
                ii, jj = divmod(flat, k)
                delta = float(changes[ii, jj])
                if delta < best_delta:
                    best_delta, move = delta, (int(rows[ii]), jj)
            if move is not None:
                i, j = move
                route[i + 1:j + 1] = reversed(route[i + 1:j + 1])
                continue
            if now() >= until:
                break
            # Single-node relocation; the depot position never moves.
            saving = d[prev, rr] + edges - d[prev, nxt]
            for start in range(1, k, block):
                if now() >= until:
                    break
                rows = np.arange(start, min(k, start + block))
                changes = (d[rr[rows, None], rr] + d[rr[rows, None], nxt]
                           - edges[None, :] - saving[rows, None])
                changes[(columns == rows[:, None]) |
                        (columns == rows[:, None] - 1)] = np.inf
                flat = int(np.argmin(changes))
                ii, jj = divmod(flat, k)
                delta = float(changes[ii, jj])
                if delta < best_delta:
                    best_delta, move = delta, (int(rows[ii]), jj)
            if move is not None:
                i, j = move
                u = route.pop(i)
                route.insert(j + 1 - int(i < j), u)
                continue
            if now() >= until or k < 4:
                break
            # Adjacent two-node segments may be relocated forward or reversed.
            for start in range(1, k - 1, block):
                if now() >= until:
                    break
                rows = np.arange(start, min(k - 1, start + block))
                a, b = rr[rows], rr[rows + 1]
                before, after = rr[rows - 1], rr[(rows + 2) % k]
                saving2 = d[before, a] + d[b, after] - d[before, after]
                forward = d[a[:, None], rr] + d[b[:, None], nxt]
                reverse = d[b[:, None], rr] + d[a[:, None], nxt]
                changes = np.minimum(forward, reverse) - edges - saving2[:, None]
                changes[(columns >= rows[:, None] - 1) &
                        (columns <= rows[:, None] + 1)] = np.inf
                flat = int(np.argmin(changes))
                ii, jj = divmod(flat, k)
                delta = float(changes[ii, jj])
                if delta < best_delta:
                    best_delta = delta
                    move = (int(rows[ii]), jj, bool(reverse[ii, jj] < forward[ii, jj]))
            if move is None:
                break
            i, j, rev = move
            pair = route[i:i + 2]
            if rev:
                pair.reverse()
            del route[i:i + 2]
            pos = j + 1 - (2 if j > i else 0)
            route[pos:pos] = pair
        return route

    def fill(route, target, weights, alpha, until):
        route = route[:]
        total = length(route)
        outside = np.ones(n, dtype=bool)
        outside[route] = False
        gain = np.full(n, np.inf)
        anchor = np.full(n, -1, dtype=int)

        def refresh(nodes):
            rr = np.asarray(route, dtype=int)
            nxt = np.roll(rr, -1)
            edges = d[rr, nxt]
            chunk = max(1, matrix_entries // len(route))
            for start in range(0, len(nodes), chunk):
                vv = nodes[start:start + chunk]
                costs = d[vv[:, None], rr] + d[vv[:, None], nxt] - edges
                arg = np.argmin(costs, axis=1)
                gain[vv] = costs[np.arange(len(vv)), arg]
                anchor[vv] = rr[arg]
                if now() >= until:
                    break

        refresh(np.flatnonzero(outside))
        while now() < until and bool(np.any(outside)):
            eligible = outside & (gain <= target - total)
            if not bool(np.any(eligible)):
                # Under non-metric distances a two-node path may be insertable even when no single node is.
                vv = np.flatnonzero(outside)
                if len(vv) < 2:
                    break
                rr = np.asarray(route, dtype=int)
                nxt = np.roll(rr, -1)
                edge_cost = d[rr, nxt]
                neighbors = min(len(vv) - 1, max(1, int(strength * math.sqrt(math.log1p(n)))))
                best_pair, best_priority = None, -math.inf
                for u in vv:
                    if now() >= until:
                        break
                    distances = d[u, vv].copy()
                    distances[vv == u] = np.inf
                    choices = vv[np.argpartition(distances, neighbors - 1)[:neighbors]]
                    costs = (d[rr, u][None, :] + d[u, choices][:, None]
                             + d[choices[:, None], nxt] - edge_cost)
                    positions = np.argmin(costs, axis=1)
                    delta = costs[np.arange(len(choices)), positions]
                    rating = (weights[u] + weights[choices]) / np.maximum(delta, tiny) ** alpha
                    rating[delta > target - total] = -np.inf
                    j = int(np.argmax(rating))
                    if float(rating[j]) > best_priority:
                        best_priority = float(rating[j])
                        best_pair = (int(u), int(choices[j]), int(positions[j]), float(delta[j]))
                if best_pair is None:
                    break
                u, v, pos, delta = best_pair
                route[pos + 1:pos + 1] = [u, v]
                total += delta
                outside[u] = outside[v] = False
                gain[:] = np.inf
                refresh(np.flatnonzero(outside))
                continue
            priority = weights / np.maximum(gain, tiny) ** alpha
            priority[~eligible] = -np.inf
            v = int(np.argmax(priority))
            a = int(anchor[v])
            pos = route.index(a) + 1
            b = route[pos % len(route)]
            total += float(gain[v])
            route.insert(pos, v)
            outside[v] = False
            gain[v] = np.inf
            stale = np.flatnonzero(outside & (anchor == a))
            if len(stale):
                refresh(stale)
            vv = np.flatnonzero(outside)
            if not len(vv):
                break
            left = d[a, vv] + d[vv, v] - d[a, v]
            right = d[v, vv] + d[vv, b] - d[v, b]
            use_left = (left < gain[vv]) & (left <= right)
            use_right = (right < gain[vv]) & (right < left)
            gain[vv[use_left]] = left[use_left]
            anchor[vv[use_left]] = a
            gain[vv[use_right]] = right[use_right]
            anchor[vv[use_right]] = v
        return route

    def repair(route, until):
        route = route[:]
        total = length(route)
        while total > budget and len(route) > 3 and now() < until:
            rr = np.asarray(route, dtype=int)
            prev, nxt = np.roll(rr, 1), np.roll(rr, -1)
            saving = d[prev, rr] + d[rr, nxt] - d[prev, nxt]
            noise = np.asarray([rng.uniform(0.8, 1.2) for _ in route])
            rating = p[rr] * noise / np.maximum(saving, tiny)
            rating[0] = np.inf
            i = int(np.argmin(rating))
            route.pop(i)
            total = length(route)
        return route

    def exchange(route, until):
        rr = np.asarray(route, dtype=int)
        k = len(route)
        if k == n:
            return route, False
        outside = np.ones(n, dtype=bool)
        outside[rr] = False
        vv = np.flatnonzero(outside)
        prev, nxt = np.roll(rr, 1), np.roll(rr, -1)
        edges = d[rr, nxt]
        saving = d[prev, rr] + edges - d[prev, nxt]
        total = length(route)
        chunk = max(1, matrix_entries // k)
        col = np.arange(k)[None, :]
        best_rank, chosen = -np.inf, None
        for start in range(0, len(vv), chunk):
            if now() >= until:
                break
            nodes = vv[start:start + chunk]
            costs = d[nodes[:, None], rr] + d[nodes[:, None], nxt] - edges
            # Removing one node invalidates only two old insertion edges, so the top three suffice.
            order = np.argpartition(costs, 2, axis=1)[:, :3]
            insertion = np.full((len(nodes), k), np.inf)
            for h in range(3):
                edge = order[:, h]
                valid = (edge[:, None] != col) & (edge[:, None] != (col - 1) % k)
                value = costs[np.arange(len(nodes)), edge]
                insertion = np.minimum(insertion, np.where(valid, value[:, None], np.inf))
            bypass = d[nodes[:, None], prev] + d[nodes[:, None], nxt] - d[prev, nxt]
            delta = np.minimum(insertion, bypass) - saving
            profit = p[nodes, None] - p[rr]
            valid = (delta <= budget - total) & ((profit > 0) | ((profit == 0) & (delta < -eps)))
            valid[:, 0] = False
            rank = profit / mean_p - 1e-8 * delta / max(abs(budget), tiny)
            rank[~valid] = -np.inf
            flat = int(np.argmax(rank))
            i, j = divmod(flat, k)
            if float(rank[i, j]) > best_rank:
                best_rank, chosen = float(rank[i, j]), (int(nodes[i]), j)
        if chosen is None:
            return route, False
        v, j = chosen
        trial = route[:]
        trial.pop(j)
        rr = np.asarray(trial, dtype=int)
        nxt = np.roll(rr, -1)
        costs = d[rr, v] + d[v, nxt] - d[rr, nxt]
        trial.insert(int(np.argmin(costs)) + 1, v)
        if length(trial) > budget:
            return route, False
        return trial, True

    def refine(route, weights, alpha, target, until):
        route = polish(route, min(until, now() + 0.20 * max(0.0, until - now())))
        route = fill(route, target, weights, alpha,
                     min(until, now() + 0.60 * max(0.0, until - now())))
        route = polish(route, min(until, now() + 0.50 * max(0.0, until - now())))
        if length(route) > budget:
            route = repair(route, until)
        if length(route) > budget:
            return None
        remember(route)
        while now() < until and len(route) < n:
            size = len(route)
            route = fill(route, budget, weights, alpha, until)
            route = polish(route, until)
            remember(route)
            if len(route) > size:
                continue
            route, changed = exchange(route, until)
            if not changed:
                break
            remember(route)
        return route

    base_weights = p / mean_p
    first_until = min(deadline, now() + 0.16 * limit)
    initial = refine(seed, base_weights, 1.0, budget, first_until)
    current = incumbent[0][:] if initial is None else initial
    current_score = math.fsum(float(p[u]) for u in current)
    pool_capacity = max(2, int(2 + strength * math.sqrt(math.log1p(n))))
    pool = [(incumbent[1], incumbent[2], incumbent[0][:], frozenset(incumbent[0]))]
    slice_size = max(0.001, limit / (32 + 8 * math.sqrt(n)))
    max_destroy = 1 - math.exp(-0.35 * math.sqrt(limit + 0.01))
    while now() < deadline and len(incumbent[0]) < n:
        stall = min(1.0, (now() - last_improvement[0]) / max(0.001, 0.14 * limit))
        if rng.random() < 0.42:
            source = current
        else:
            source = pool[min(len(pool) - 1, int(len(pool) * rng.random() ** 2))][2]
        route = source[:]
        donor = pool[rng.randrange(len(pool))][3]
        mode = rng.randrange(5)
        if rng.random() < 0.09 + 0.16 * stall:
            # New starting points allow direct exploration of regions different from the existing route.
            a = rng.randrange(n - 1)
            if a >= depot:
                a += 1
            cost = d[depot, a] + d[a] + d[:, depot]
            allowed = (ids != a) & (ids != depot) & (cost <= budget)
            if bool(np.any(allowed)):
                value = (p[a] + p) / np.maximum(cost, tiny) ** rng.uniform(0.4, 1.5)
                value *= np.asarray([rng.uniform(0.5, 1.5) for _ in range(n)])
                value[~allowed] = -np.inf
                route = [depot, a, int(np.argmax(value))]
            else:
                route = seed[:]
        elif len(route) > 3:
            k = len(route)
            fraction = min(max_destroy, (0.03 + 0.5 * rng.random() ** 2) * (0.7 + stall))
            remove_count = min(k - 3, max(1, int((k - 1) * fraction)))
            if rng.random() < 0.24 / (1 + stall):
                remove_count = min(k - 3, 1 + int(rng.random() * math.sqrt(1 + strength)))
            candidates = list(range(1, k))
            if mode == 0:
                rr = np.asarray(route, dtype=int)
                saving = d[np.roll(rr, 1), rr] + d[rr, np.roll(rr, -1)] - d[np.roll(rr, 1), np.roll(rr, -1)]
                candidates.sort(key=lambda i: float(p[route[i]]) * rng.uniform(0.5, 1.5) / max(float(saving[i]), tiny))
            elif mode == 1:
                center = rng.randrange(n)
                candidates.sort(key=lambda i: float(d[center, route[i]]) * rng.uniform(0.7, 1.3))
            elif mode == 2:
                start = rng.randrange(1, k)
                candidates.sort(key=lambda i: (i - start) % (k - 1))
            elif mode == 3:
                rng.shuffle(candidates)
            else:
                candidates.sort(key=lambda i: (route[i] in donor, rng.random()))
            removed = set(candidates[:remove_count])
            route = [u for i, u in enumerate(route) if i not in removed]
        alpha = rng.uniform(0.55, 1.45)
        if rng.random() < 0.08:
            alpha = 0.0
        noise = 0.08 + 0.4 * rng.random() * (0.5 + stall)
        weights = base_weights * np.asarray([math.exp(rng.uniform(-noise, noise)) for _ in range(n)])
        if mode == 4:
            for u in donor:
                weights[u] *= 1.0 + 0.25 * rng.random()
        relaxed = rng.random() < 0.32
        target = budget * (1.0 + (0.02 + 0.18 * rng.random()) * max_destroy) if relaxed else budget
        until = min(deadline, now() + slice_size * (0.8 + 0.8 * stall))
        trial = refine(route, weights, alpha, target, until)
        if trial is None:
            continue
        score, cost = remember(trial)
        if cost > budget:
            continue
        temperature = mean_p * (0.15 + 1.2 * stall) / (1 + 0.12 * strength)
        if score >= current_score or rng.random() < math.exp(max(-700.0, (score - current_score) / temperature)):
            current, current_score = trial, score
        signature = frozenset(trial)
        same = next((i for i, entry in enumerate(pool) if entry[3] == signature), None)
        if same is not None:
            if cost < pool[same][1]:
                pool[same] = (score, cost, trial[:], signature)
        elif len(pool) < pool_capacity or score >= pool[-1][0]:
            pool.append((score, cost, trial[:], signature))
        pool.sort(key=lambda entry: (-entry[0], entry[1]))
        if len(pool) > pool_capacity:
            pool.pop()
