"""Adaptive elite large neighbourhood search with exact path patching
Combines heap-based randomised regret insertion, multi-mode destruction, cross-position node swaps, path relinking and subset dynamic programming, maintaining the best feasible tour within the wall-clock limit.
"""
import heapq
import math
import random
import time
from array import array

import numpy as np


def solve(instance: dict, time_limit_s: float) -> dict:
    clock = time.perf_counter
    started = clock()
    result = {"visited": [], "selected_links": []}
    best_route = None
    n = 0
    depot = 0
    try:
        limit = float(time_limit_s)
        if not math.isfinite(limit) or limit < 0.0:
            limit = 0.0
        n = int(instance["n_locations"])
        depot = int(instance["depot_index"])
        result = {"visited": [0] * n,
                  "selected_links": [[0] * n for _ in range(n)]}
        if not 0 <= depot < n:
            return result
        result["visited"][depot] = 1
        if n < 3:
            return result
        p = [float(v) for v in instance["scores"]]
        d = instance["distance_matrix"]
        B = float(instance["distance_budget"])
        if len(p) != n or len(d) != n or any(len(row) != n for row in d):
            return result
        if not math.isfinite(B) or B <= 0.0:
            return result
        D = np.asarray(d, dtype=float)
        P = np.asarray(p, dtype=float)
        preparation = clock() - started
        reserve = min(0.10 * limit,
                      max(0.002, 0.007 * limit, 0.5 * preparation))
        deadline = started + max(0.0, limit - reserve)
        horizon = math.log1p(limit)
        eps = 1.0e-11
        nodes = [v for v in range(n) if v != depot]
        unit = max(1.0e-12, B / n)
        avg_prize = max(1.0e-12, math.fsum(p) / n)
        near = sorted(nodes, key=lambda v: (d[depot][v], -p[v]))
        rich = sorted(nodes, key=lambda v: (-p[v], d[depot][v]))
        density = sorted(nodes, key=lambda v: -p[v] / (d[depot][v] + unit))
        seed = (n + 1) * 1000003 + depot * 9176
        for v in nodes:
            seed = (seed * 6364136223846793005 + int(p[v] * 1009)
                    + int(d[depot][v] * 9176) + v) & ((1 << 64) - 1)
        rng = random.Random(seed)
        best_score = -math.inf
        best_cost = math.inf
        last_improvement = started
        capacity = max(4, int(3 + math.sqrt(n) * math.sqrt(1 + horizon)))
        archive = []

        def cost(route):
            return math.fsum(d[route[i - 1]][v] for i, v in enumerate(route))

        def value(route):
            return math.fsum(p[v] for v in route)

        def register(route, store=True):
            nonlocal best_route, best_cost, best_score, last_improvement, archive
            if len(route) < 3 or route[0] != depot or len(set(route)) != len(route):
                return False
            c = cost(route)
            if c > B or not math.isfinite(c):
                return False
            s = value(route)
            changed = s > best_score or (s == best_score and c < best_cost)
            if changed:
                if s > best_score:
                    last_improvement = clock()
                best_route, best_score, best_cost = route[:], s, c
            if not store:
                return changed
            signature = frozenset(route)
            for idx, entry in enumerate(archive):
                if entry[3] == signature:
                    if c < entry[2]:
                        archive[idx] = (route[:], s, c, signature)
                    return changed
            archive.append((route[:], s, c, signature))
            if len(archive) > capacity:
                archive.sort(key=lambda e: (-e[1], e[2]))
                elite = max(2, capacity // 2)
                # Keep high-score solutions, and at the tail evict the solution most similar to the elite visited sets.
                remove = max(range(elite, len(archive)), key=lambda j:
                             max(len(archive[j][3] & e[3]) /
                                 max(1, len(archive[j][3] | e[3]))
                                 for e in archive[:elite])
                             + 0.03 * rng.random())
                archive.pop(remove)
            return changed

        # First find one feasible tour; only then spend time building a diversified archive.
        seed_until = min(deadline, started + 0.025 * limit)
        for ai, a in enumerate(near):
            if ai and clock() >= (seed_until if best_route else deadline):
                break
            tail = near[ai + 1:]
            if not tail:
                break
            costs = D[depot, a] + D[a, tail] + D[tail, depot]
            feasible = np.flatnonzero(costs <= B)
            if feasible.size:
                j = int(feasible[np.argmin(costs[feasible])])
                register([depot, a, tail[j]])
                j = int(feasible[np.argmax(P[np.asarray(tail)[feasible]])])
                register([depot, a, tail[j]])
        if best_route is None:
            # Positive-distance non-metric graph: shortest non-depot paths can form a tour even when no feasible triangle exists.
            for source in near:
                if clock() >= deadline:
                    break
                first = d[depot][source]
                if first >= B:
                    continue
                labels = [math.inf] * n
                parents = [-1] * n
                settled = [False] * n
                labels[source] = 0.0
                heap = [(0.0, source)]
                while heap and clock() < deadline:
                    du, u = heapq.heappop(heap)
                    if settled[u] or du != labels[u]:
                        continue
                    settled[u] = True
                    if u != source and first + du + d[u][depot] <= B:
                        path, v = [], u
                        while v != -1:
                            path.append(v)
                            v = parents[v]
                        path.reverse()
                        register([depot] + path)
                        if best_route is not None:
                            break
                    for index, v in enumerate(nodes):
                        if settled[v]:
                            continue
                        nd = du + d[u][v]
                        if nd < labels[v] and first + nd < B:
                            labels[v] = nd
                            parents[v] = u
                            heapq.heappush(heap, (nd, v))
                        if index % 256 == 0 and clock() >= deadline:
                            break
                if best_route is not None:
                    break
        if best_route is None:
            return result
        safe_route = best_route[:]
        # Prefix construction by linear scan balances distance, score and profit density.
        for order in (near, density, rich):
            route = [depot]
            path_cost = 0.0
            for v in order:
                if clock() >= seed_until:
                    break
                path_cost += d[route[-1]][v]
                route.append(v)
                if len(route) >= 3 and path_cost + d[v][depot] <= B:
                    register(route)

        def quantum():
            # Time budget and instance size jointly determine the computational granularity of each neighbourhood.
            m = len(best_route)
            return min(max(0.0, deadline - clock()),
                       max(0.001, min(0.065 * limit,
                           0.003 * (1 + math.sqrt(limit))
                           + 1.0e-7 * (n * m + m * m))))

        def insertion(route, v):
            best, position = math.inf, -1
            for i, a in enumerate(route):
                b = route[(i + 1) % len(route)]
                delta = d[a][v] + d[v][b] - d[a][b]
                if delta < best:
                    best, position = delta, i + 1
            return best, position

        def repair(route, until, randomized=True, preferred=None, cap=None):
            """An incremental edge heap shares randomised restricted candidates, score density and second-order regret information."""
            route = route[:]
            current_cost = cost(route)
            bound = B if cap is None else cap
            present = set(route)
            unused = [v for v in nodes if v not in present]
            if not unused or clock() >= until:
                return route
            rng.shuffle(unused)
            successor = {a: route[(i + 1) % len(route)]
                         for i, a in enumerate(route)}
            heaps = {}
            edges = list(successor.items())
            for vi, v in enumerate(unused):
                h = []
                for ei, (a, b) in enumerate(edges):
                    h.append((d[a][v] + d[v][b] - d[a][b], a, b))
                    if ei % 256 == 0 and clock() >= until:
                        return route
                heapq.heapify(h)
                heaps[v] = h
                if vi % 32 == 0 and clock() >= until:
                    return route
            alpha = rng.uniform(0.68, 1.48) if randomized else 1.0
            eta = rng.uniform(0.015, 0.10) if randomized else 0.02
            width = (1 + int(math.sqrt(1 + horizon) *
                            math.sqrt(max(1, n - len(route))) / 5)
                     if randomized else 1)
            preferred = set() if preferred is None else preferred
            mode = rng.randrange(3) if randomized else 0
            while unused and clock() < until:
                choices = []
                slack = bound - current_cost
                for index, v in enumerate(unused):
                    h = heaps[v]
                    while h and successor.get(h[0][1]) != h[0][2]:
                        heapq.heappop(h)
                    if not h:
                        continue
                    delta, a, b = h[0]
                    if delta > slack:
                        continue
                    first = heapq.heappop(h)
                    while h and successor.get(h[0][1]) != h[0][2]:
                        heapq.heappop(h)
                    regret = max(0.0, h[0][0] - delta) if h else 0.0
                    heapq.heappush(h, first)
                    denom = max(0.0, delta) + eta * unit
                    rating = alpha * math.log(max(p[v] / avg_prize, 1e-300))
                    rating -= math.log(max(denom / unit, 1e-300))
                    rating += 0.10 * min(5.0, regret / unit)
                    if mode == 1:
                        rating += 0.4 * p[v] / avg_prize
                    elif mode == 2:
                        rating += 0.30 * min(5.0, regret / max(denom, unit * 0.01))
                    if delta < 0.0:
                        rating += min(8.0, -delta / unit)
                    if v in preferred:
                        rating += 0.45
                    item = (rating, rng.random() if randomized else p[v], v, a, b)
                    if len(choices) < width:
                        heapq.heappush(choices, item)
                    elif item > choices[0]:
                        heapq.heapreplace(choices, item)
                    if index % 128 == 0 and clock() >= until:
                        break
                if not choices:
                    break
                choices.sort(reverse=True)
                pick = min(len(choices) - 1,
                           int(rng.random() ** 2 * len(choices))) if randomized else 0
                inserted = False
                for item in choices[pick:] + choices[:pick]:
                    _, _, v, a, b = item
                    pos = route.index(a) + 1
                    trial = route[:pos] + [v] + route[pos:]
                    c = cost(trial)
                    if c <= bound:
                        route, current_cost = trial, c
                        inserted = True
                        break
                if not inserted:
                    break
                successor[a] = v
                successor[v] = b
                unused.remove(v)
                # Only add the two new edges; old edges are lazily deleted when popped from the heap top.
                for index, u in enumerate(unused):
                    h = heaps[u]
                    heapq.heappush(h, (d[a][u] + d[u][v] - d[a][v], a, v))
                    heapq.heappush(h, (d[v][u] + d[u][b] - d[v][b], v, b))
                    if index % 128 == 0 and clock() >= until:
                        if current_cost <= B:
                            register(route, False)
                        return route
                if current_cost <= B:
                    register(route, False)
            return route

        def order_improve(route, until, deep=False):
            """Check 2-opt and single/two-node relocation in chunks, allowing repair of temporarily over-budget routes."""
            route = route[:]
            c = cost(route)
            while len(route) >= 4 and clock() < until:
                m = len(route)
                r = np.asarray(route, dtype=int)
                nxt = np.roll(r, -1)
                ed = D[r, nxt]
                positions = np.arange(m)
                best_delta = -eps * max(1.0, c)
                move = None
                # Chunking bounds each uninterruptible vector operation and its temporary memory.
                block = max(1, min(m, int(math.sqrt(max(1, n) * (1 + horizon)))))
                offset = rng.randrange(m)
                for start in range(0, m, block):
                    if clock() >= until:
                        break
                    ix = (np.arange(start, min(m, start + block)) + offset) % m
                    delta = (D[r[ix, None], r[None, :]]
                             + D[nxt[ix, None], nxt[None, :]]
                             - ed[ix, None] - ed[None, :])
                    invalid = (positions[None, :] <= ix[:, None] + 1)
                    invalid |= ((ix[:, None] == 0) & (positions[None, :] == m - 1))
                    delta[invalid] = np.inf
                    at = int(np.argmin(delta))
                    row, j = divmod(at, m)
                    gain = float(delta[row, j])
                    if gain < best_delta:
                        best_delta = gain
                        move = (0, int(ix[row]), j)
                if move is None and clock() < until:
                    # For relocation, the insertion edge may not be adjacent to the node being moved.
                    prev = np.roll(r, 1)
                    removal = D[prev, nxt] - D[prev, r] - D[r, nxt]
                    for start in range(1, m, block):
                        if clock() >= until:
                            break
                        ix = np.arange(start, min(m, start + block))
                        delta = (removal[ix, None]
                                 + D[r[None, :], r[ix, None]]
                                 + D[r[ix, None], nxt[None, :]] - ed[None, :])
                        delta[(positions[None, :] == ix[:, None]) |
                              (positions[None, :] == ix[:, None] - 1)] = np.inf
                        at = int(np.argmin(delta))
                        row, j = divmod(at, m)
                        gain = float(delta[row, j])
                        if gain < best_delta:
                            best_delta = gain
                            move = (1, int(ix[row]), j)
                if move is None and deep and m >= 5 and clock() < until:
                    for i in range(1, m - 1):
                        if clock() >= until:
                            break
                        u, v = route[i:i + 2]
                        a, b = route[i - 1], route[(i + 2) % m]
                        removal = d[a][b] - d[a][u] - d[u][v] - d[v][b]
                        for rev in (False, True):
                            x, y = (v, u) if rev else (u, v)
                            delta = removal + D[r, x] + d[x][y] + D[y, nxt] - ed
                            delta[i - 1:i + 2] = np.inf
                            j = int(np.argmin(delta))
                            gain = float(delta[j])
                            if gain < best_delta:
                                best_delta, move = gain, (2, i, j, rev)
                if move is None:
                    break
                kind, i, j = move[:3]
                if kind == 0:
                    trial = route[:i + 1] + route[i + 1:j + 1][::-1] + route[j + 1:]
                else:
                    count = kind
                    anchor = route[j]
                    chunk = route[i:i + count]
                    if kind == 2 and move[3]:
                        chunk.reverse()
                    trial = route[:i] + route[i + count:]
                    where = trial.index(anchor) + 1
                    trial[where:where] = chunk
                nc = cost(trial)
                if nc >= c - eps * max(1.0, c):
                    break
                route, c = trial, nc
                if c <= B:
                    register(route, False)
            return route

        def exchange(route, until):
            """Remove one node then insert at any position: the original tour's top three insertion edges suffice to cover the unremoved edges."""
            used = set(route)
            outsiders = [v for v in nodes if v not in used]
            if not outsiders or clock() >= until:
                return route
            rng.shuffle(outsiders)
            r = np.asarray(route, dtype=int)
            m = len(route)
            nxt = np.roll(r, -1)
            c = cost(route)
            old = r[1:]
            prev = r[:-1]
            following = np.roll(r, -1)[1:]
            removal = D[prev, following] - D[prev, old] - D[old, following]
            best_gain, best_c, chosen = 0.0, c, None
            old_pos = np.arange(1, m)
            block = max(1, min(len(outsiders),
                              int(math.sqrt(n) * math.sqrt(1 + horizon))))
            for start in range(0, len(outsiders), block):
                if clock() >= until:
                    break
                out = np.asarray(outsiders[start:start + block], dtype=int)
                ins = (D[out[:, None], r[None, :]]
                       + D[out[:, None], nxt[None, :]] - D[r, nxt][None, :])
                top = np.argpartition(ins, 2, axis=1)[:, :3]
                top_val = np.take_along_axis(ins, top, axis=1)
                choose_order = np.argsort(top_val, axis=1)
                top = np.take_along_axis(top, choose_order, axis=1)
                vals = np.take_along_axis(ins, top, axis=1)
                survived = np.full((len(out), m - 1), np.inf)
                edge_at = np.full((len(out), m - 1), -1, dtype=int)
                for k in range(3):
                    edge = top[:, k, None]
                    valid = ((edge != old_pos[None, :]) &
                             (edge != old_pos[None, :] - 1))
                    improve = valid & (vals[:, k, None] < survived)
                    survived[improve] = np.broadcast_to(vals[:, k, None], survived.shape)[improve]
                    edge_at[improve] = np.broadcast_to(edge, edge_at.shape)[improve]
                bridge = (D[out[:, None], prev[None, :]]
                          + D[out[:, None], following[None, :]]
                          - D[prev, following][None, :])
                at_bridge = bridge < survived
                survived = np.minimum(survived, bridge)
                edge_at[at_bridge] = -1
                new_costs = c + removal[None, :] + survived
                gains = P[out, None] - P[old][None, :]
                admissible = (new_costs <= B) & ((gains > best_gain) |
                              ((gains == best_gain) & (new_costs < best_c - eps)))
                if not np.any(admissible):
                    continue
                best_block_gain = float(np.max(np.where(admissible, gains, -np.inf)))
                choices = np.where(admissible & (gains == best_block_gain), new_costs, np.inf)
                at = int(np.argmin(choices))
                x, y = divmod(at, m - 1)
                nc = float(choices[x, y])
                if best_block_gain > best_gain or nc < best_c:
                    best_gain, best_c = best_block_gain, nc
                    chosen = (int(out[x]), y + 1, int(edge_at[x, y]))
            if chosen is None:
                return route
            v, position, edge = chosen
            trial = route[:position] + route[position + 1:]
            if edge == -1:
                trial.insert(position, v)
            else:
                where = trial.index(route[edge]) + 1
                trial.insert(where, v)
            nc = cost(trial)
            if nc <= B and (value(trial) > value(route) or nc < c - eps):
                register(trial, False)
                return trial
            return route

        def polish(route, until, randomize=True, preferred=None, deep=False):
            route = route[:]
            while clock() < until:
                old_s, old_c = value(route), cost(route)
                order_until = min(until, clock() + 0.28 * max(0.0, until - clock()))
                route = order_improve(route, order_until, deep)
                if cost(route) > B:
                    break
                route = repair(route, until, randomize, preferred)
                register(route)
                if len(route) == n or clock() >= until:
                    break
                if value(route) == old_s and cost(route) >= old_c - eps:
                    route2 = exchange(route, until)
                    if route2 == route:
                        break
                    route = route2
            return route

        def destroy(route, until, stagnation, donor=None):
            route = route[:]
            m = len(route)
            if m <= 3:
                return route, set()
            strength = min(0.75, 0.06 + 0.045 * horizon + 0.4 * stagnation)
            fraction = strength * rng.random() ** 1.7
            count = min(m - 3, max(1, int(math.ceil((m - 3) * fraction))))
            mode = rng.randrange(5)
            anchor = route[rng.randrange(1, m)]
            start = rng.randrange(1, m)
            priority = []
            for i in range(1, m):
                v = route[i]
                saving = d[route[i - 1]][v] + d[v][route[(i + 1) % m]] - d[route[i - 1]][route[(i + 1) % m]]
                if mode == 0:
                    key = rng.random()
                elif mode == 1:
                    key = p[v] * rng.uniform(0.75, 1.25)
                elif mode == 2:
                    key = p[v] / (max(0.0, saving) + 0.025 * unit) * rng.uniform(0.8, 1.2)
                elif mode == 3:
                    key = (i - start) % (m - 1)
                else:
                    key = d[anchor][v] * rng.uniform(0.8, 1.2)
                if donor is not None and v in donor:
                    key += max(1.0, abs(key)) * (2 + horizon)
                priority.append((key, v))
            priority.sort()
            removed = {v for _, v in priority[:count]}
            child = [v for v in route if v not in removed]
            if cost(child) <= B:
                return child, removed
            # In non-metric graphs removing a node may increase cost; allow reordering first, then repair by negative-increment insertion.
            child = order_improve(child, min(until, clock() + 0.2 * max(0.0, until - clock())))
            if cost(child) <= B:
                return child, removed
            child = repair(child, until, True, removed)
            if cost(child) <= B:
                return child, removed
            return route, set()

        def eject_target(route, until, stagnation):
            original = route[:]
            used = set(route)
            unused = [v for v in rich if v not in used]
            if not unused:
                return route
            width = min(len(unused), max(2, int(math.sqrt(n) * (1 + horizon))))
            target = unused[min(width - 1, int(rng.random() ** 2 * width))]
            candidate = route[:]
            dc, pos = insertion(candidate, target)
            candidate.insert(pos, target)
            c = cost(candidate)
            protected = {target}
            # Insert the target, then evict low-loss nodes, instead of requiring every intermediate state to be feasible.
            while c > B and len(candidate) > 3 and clock() < until:
                choices = []
                deficit = c - B
                for i in range(1, len(candidate)):
                    v = candidate[i]
                    if v in protected:
                        continue
                    a, b = candidate[i - 1], candidate[(i + 1) % len(candidate)]
                    saving = d[a][v] + d[v][b] - d[a][b]
                    key = p[v] / max(unit * 0.015, min(deficit, saving))
                    key *= rng.uniform(0.88, 1.12)
                    choices.append((key, -saving, i))
                if not choices:
                    break
                choices.sort()
                pick_width = min(len(choices), 1 + int(stagnation * (1 + horizon)))
                pos = choices[min(pick_width - 1, int(rng.random() ** 2 * pick_width))][2]
                candidate.pop(pos)
                c = cost(candidate)
            if c <= B:
                register(candidate)
                return candidate
            return original

        def oscillate(route, until, stagnation):
            c = cost(route)
            excess = B + max(c / len(route), B * (0.02 + 0.025 * horizon + 0.1 * stagnation))
            addition_until = min(until, clock() + 0.32 * max(0.0, until - clock()))
            candidate = repair(route, addition_until, True, cap=excess)
            candidate = order_improve(candidate, min(until, clock() + 0.3 * max(0.0, until - clock())), True)
            c = cost(candidate)
            while c > B and len(candidate) > 3 and clock() < until:
                deficit = c - B
                choices = []
                for i in range(1, len(candidate)):
                    a, v, b = candidate[i - 1], candidate[i], candidate[(i + 1) % len(candidate)]
                    saving = d[a][v] + d[v][b] - d[a][b]
                    key = p[v] / max(unit * 0.01, min(deficit, saving))
                    choices.append((key * rng.uniform(0.93, 1.07), -saving, i))
                position = min(choices)[2]
                candidate.pop(position)
                c = cost(candidate)
            if c <= B:
                register(candidate)
                return candidate
            return route

        dp_rate = 1500000.0
        global_done = False

        def exact_patch(route, until, donor=None):
            nonlocal dp_rate, global_done
            remaining = until - clock()
            if remaining <= 0.0:
                return route
            # Depth is derived from remaining time and the observed transition rate; the memory limit is not an iteration cap.
            k = 2
            while (k + 1) <= n - 1:
                next_k = k + 1
                states = (1 << next_k) * next_k
                work = next_k * (next_k - 1) * (1 << max(0, next_k - 2))
                if work > remaining * dp_rate * 0.7 or states * 10 > 32 * 1024 * 1024:
                    break
                k = next_k
            full = k >= n - 1 and not global_done
            m = len(route)
            if full:
                left, right = 1, m
                pool = nodes[:]
            else:
                segment_size = min(m - 1, max(1, k // 2))
                segment_size = rng.randint(1, segment_size)
                left = rng.randrange(1, m - segment_size + 1)
                right = left + segment_size
                pool = route[left:right]
            a = route[left - 1]
            b = route[right] if right < m else depot
            old_segment = route[left:right]
            old_path = [a] + old_segment + [b]
            fixed = math.fsum(d[route[i - 1]][v]
                             for i, v in enumerate(route)
                             if not left <= i <= right and not (right == m and i == 0))
            # Explicitly accumulate fixed edges to avoid cancellation error from subtraction at the budget boundary.
            if right < m:
                # The expression above already excludes a -> segment, segment interior, and segment -> b.
                allowance = B - fixed
            else:
                allowance = B - fixed
            if allowance <= 0.0:
                return route
            if not full:
                used = set(route)
                unused = [v for v in nodes if v not in used]
                preference = set() if donor is None else set(donor)
                alpha = rng.uniform(0.65, 1.3)
                ranked = []
                for v in unused:
                    via = d[a][v] + d[v][b]
                    rating = alpha * math.log(max(p[v], 1e-300)) - math.log(max(unit * 0.02, via))
                    rating += rng.uniform(-0.35, 0.35)
                    if v in preference:
                        rating += 0.5
                    ranked.append((rating, v))
                ranked.sort(reverse=True)
                pool.extend(v for _, v in ranked[:max(0, k - len(pool))])
            k = len(pool)
            if not k or clock() >= until:
                return route
            state_count = 1 << k
            dp = array('d', [math.inf]) * (state_count * k)
            parents = array('b', [-2]) * (state_count * k)
            subset_score = array('d', [0.0]) * state_count
            for j, v in enumerate(pool):
                idx = (1 << j) * k + j
                dp[idx] = d[a][v]
                parents[idx] = -1
            local_d = [[d[u][v] for v in pool] for u in pool]
            ending = [d[v][b] for v in pool]
            prizes = [p[v] for v in pool]
            outside_count = m - len(old_segment)
            winner_mask, winner_last = -1, -1
            winner_score, winner_cost = -math.inf, math.inf
            full_mask = state_count - 1
            begin = clock()
            transitions = 0
            completed = True
            for mask in range(1, state_count):
                if mask % 16 == 0 and clock() >= until:
                    completed = False
                    break
                low = mask & -mask
                subset_score[mask] = subset_score[mask ^ low] + prizes[low.bit_length() - 1]
                base = mask * k
                bits = mask
                while bits:
                    bit = bits & -bits
                    j = bit.bit_length() - 1
                    bits ^= bit
                    path = dp[base + j]
                    if path >= allowance:
                        continue
                    closed = path + ending[j]
                    sc = subset_score[mask]
                    if (outside_count + mask.bit_count() >= 3 and closed <= allowance
                            and (sc > winner_score or (sc == winner_score and closed < winner_cost))):
                        winner_mask, winner_last = mask, j
                        winner_score, winner_cost = sc, closed
                    missing = full_mask ^ mask
                    row = local_d[j]
                    while missing:
                        nb = missing & -missing
                        v = nb.bit_length() - 1
                        missing ^= nb
                        nxt_index = (mask | nb) * k + v
                        nd = path + row[v]
                        if nd < dp[nxt_index]:
                            dp[nxt_index] = nd
                            parents[nxt_index] = j
                        transitions += 1
            elapsed = clock() - begin
            if transitions and elapsed > 0:
                dp_rate = 0.6 * dp_rate + 0.4 * transitions / elapsed
            if full and completed:
                global_done = True
            if winner_mask < 0:
                return route
            order = []
            mask, j = winner_mask, winner_last
            while j >= 0:
                order.append(pool[j])
                previous = parents[mask * k + j]
                mask ^= 1 << j
                j = previous
            order.reverse()
            trial = route[:left] + order + route[right:]
            sc, c = value(trial), cost(trial)
            if c <= B and (sc > value(route) or (sc == value(route) and c < cost(route))):
                register(trial)
                return trial
            return route

        # Deterministic start within the same search pipeline, so random destruction does not precede sufficient filling.
        initial_until = min(deadline, clock() + 0.14 * max(0.0, deadline - clock()))
        candidate = polish(best_route, initial_until, False, deep=True)
        register(candidate)
        current = best_route[:]
        weights = [3.0, 1.5, 1.0, 0.65, 0.8, 0.7]
        while clock() < deadline:
            try:
                now = clock()
                age = max(0.0, now - last_improvement)
                stagnation = min(1.0, age / max(0.001, 0.18 * limit))
                q = quantum()
                if q <= 0:
                    break
                ranked = sorted(archive, key=lambda e: (-e[1], e[2]))
                choose = rng.random()
                if choose < 0.38:
                    parent = best_route[:]
                elif choose < 0.60:
                    parent = current[:]
                else:
                    ix = min(len(ranked) - 1, int(rng.random() ** 2 * len(ranked)))
                    parent = ranked[ix][0][:]
                donor = ranked[rng.randrange(len(ranked))][0]
                adjusted = [w * (1.0 + stagnation * (0.3 if i < 2 else 1.0))
                            for i, w in enumerate(weights)]
                pick = rng.random() * math.fsum(adjusted)
                operator = 0
                for i, w in enumerate(adjusted):
                    pick -= w
                    if pick <= 0:
                        operator = i
                        break
                stop = min(deadline, now + q * (1.4 if operator == 3 else 1.0))
                before_score, before_cost = best_score, best_cost
                parent_score = value(parent)
                preferred = None
                if operator == 0:
                    candidate, removed = destroy(parent, min(stop, now + q * 0.25), stagnation)
                    # Do not reward just-removed nodes, avoiding destroy-repair round trips back to the same state.
                    preferred = set(donor) - removed if rng.random() < 0.3 else None
                elif operator == 1:
                    candidate = eject_target(parent, min(stop, now + q * 0.4), stagnation)
                elif operator == 2:
                    candidate = oscillate(parent, min(stop, now + q * 0.65), stagnation)
                elif operator == 3:
                    candidate = exact_patch(parent, min(stop, now + q * 1.1), donor)
                elif operator == 4:
                    candidate, removed = destroy(parent, min(stop, now + q * 0.3), stagnation, set(donor))
                    preferred = set(donor)
                else:
                    candidate = safe_route[:]
                    pool_size = min(n - 1, max(2, int(math.sqrt(n) * (1 + horizon))))
                    sampling_end = min(stop, now + q * 0.15)
                    seed_key = -math.inf
                    while clock() < sampling_end:
                        a = (near if rng.random() < 0.5 else density)[rng.randrange(pool_size)]
                        b = (rich if rng.random() < 0.5 else near)[rng.randrange(pool_size)]
                        if a == b:
                            continue
                        trial = [depot, a, b]
                        c = cost(trial)
                        if c > B:
                            continue
                        key = (p[a] + p[b]) / (c + unit) ** rng.uniform(0.5, 1.3)
                        if key > seed_key:
                            seed_key, candidate = key, trial
                if clock() < stop:
                    candidate = polish(candidate, stop, True, preferred,
                                       deep=rng.random() < 0.25 + 0.35 * stagnation)
                register(candidate)
                c, sc = cost(candidate), value(candidate)
                if c > B:
                    continue
                reward = 0.1
                if best_score > before_score:
                    reward = 7.0
                elif best_cost < before_cost and best_score == before_score:
                    reward = 2.0
                elif sc > parent_score:
                    reward = 1.2
                weights[operator] = max(0.15, 0.90 * weights[operator] + 0.10 * reward)
                old_s = value(current)
                temp = avg_prize * (0.03 + 0.35 * stagnation) * math.sqrt(1 + horizon)
                if sc >= old_s or rng.random() < math.exp(-min(60.0, (old_s - sc) / temp)):
                    current = candidate[:]
                elif rng.random() < 0.05 + 0.1 * stagnation:
                    current = best_route[:]
            except BaseException:
                # An exception only abandons the current neighbourhood; the verified best tour is unaffected.
                continue
    except BaseException:
        pass
    # The full matrix is pre-allocated; only feasible tours re-checked against the original distances can become the incumbent.
    try:
        if best_route is not None:
            visited = result["visited"]
            links = result["selected_links"]
            for v in best_route:
                visited[v] = 1
            for i, a in enumerate(best_route):
                b = best_route[(i + 1) % len(best_route)]
                links[a][b] = 1
                links[b][a] = 1
    except BaseException:
        pass
    return result
