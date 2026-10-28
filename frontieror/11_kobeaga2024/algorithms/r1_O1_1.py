"""Adaptive orienteering search with incremental neighbourhoods and joint repair
Improves elite routes using an insertion cache, group node swaps, repair of temporarily over-budget routes and time-limited subset dynamic programming.
"""

import numpy as np
from array import array

def solve(instance: dict, time_limit_s: float) -> dict:
    safe_result = {"visited": [], "selected_links": []}
    try:
        import heapq
        import math
        import random
        import time

        clock = time.perf_counter
        started = clock()

        try:
            limit = float(time_limit_s)
        except BaseException:
            limit = 0.0
        if not math.isfinite(limit) or limit < 0.0:
            limit = 0.0

        n = max(0, int(instance.get("n_locations", 0)))
        depot = int(instance.get("depot_index", 0))
        visited0 = [0] * n
        if 0 <= depot < n:
            visited0[depot] = 1
        safe_result = {
            "visited": visited0,
            "selected_links": [[0] * n for _ in range(n)],
        }

        if n < 3 or not (0 <= depot < n):
            return safe_result

        scores = [float(v) for v in instance["scores"]]
        raw_distance = instance["distance_matrix"]
        budget = float(instance["distance_budget"])
        if len(scores) != n or len(raw_distance) != n:
            return safe_result

        distance = []
        for row in raw_distance:
            if len(row) != n:
                return safe_result
            distance.append([float(v) for v in row])

        hard_deadline = started + limit
        output_reserve = max(0.004, min(1.2, 0.012 + 7.0e-7 * n * n))
        output_reserve = min(output_reserve, 0.08 * limit)
        deadline = max(started, hard_deadline - output_reserve)

        seed_value = 0x9E3779B97F4A7C15 ^ (n * 1000003) ^ (depot * 9176)
        for i in range(min(n, 64)):
            seed_value ^= int(abs(scores[i]) * 1000003.0) + i * 2654435761
            seed_value ^= int(abs(distance[depot][i]) * 10007.0)
            seed_value = (
                seed_value * 6364136223846793005 + 1442695040888963407
            ) & ((1 << 64) - 1)
        rng = random.Random(seed_value)

        nodes = [v for v in range(n) if v != depot]
        tiny = 1.0e-12
        score_scale = max(tiny, sum(scores[v] for v in nodes) / max(1, len(nodes)))
        distance_scale = max(tiny, abs(budget) / max(3, n))

        def route_length(route):
            m = len(route)
            return math.fsum(distance[route[i]][route[(i + 1) % m]] for i in range(m))

        def route_score(route):
            total = 0.0
            for v in route:
                total += scores[v]
            return total

        def valid_route(route, cost=None):
            if route is None or len(route) < 3 or route[0] != depot:
                return False
            if len(set(route)) != len(route):
                return False
            for v in route:
                if v < 0 or v >= n:
                    return False
            if cost is None:
                cost = route_length(route)
            return math.isfinite(cost) and cost <= budget

        def pack(route):
            visited = [0] * n
            links = [[0] * n for _ in range(n)]
            if route is not None and len(route) >= 3:
                for v in route:
                    visited[v] = 1
                m = len(route)
                for i in range(m):
                    a = route[i]
                    b = route[(i + 1) % m]
                    links[a][b] = 1
                    links[b][a] = 1
            else:
                visited[depot] = 1
            return {"visited": visited, "selected_links": links}

        def better(value, cost, old_value, old_cost):
            value_tol = 1.0e-12 * max(1.0, abs(value), abs(old_value))
            cost_tol = 1.0e-12 * max(1.0, abs(cost), abs(old_cost))
            return value > old_value + value_tol or (
                abs(value - old_value) <= value_tol and cost < old_cost - cost_tol
            )

        near_order = sorted(
            nodes, key=lambda v: (distance[depot][v], -scores[v], v)
        )
        prize_order = sorted(
            nodes, key=lambda v: (-scores[v], distance[depot][v], v)
        )
        efficiency_order = sorted(
            nodes,
            key=lambda v: (
                -scores[v] / (distance[depot][v] + 0.20 * distance_scale),
                distance[depot][v],
                v,
            ),
        )
        mixed_order = []
        mixed_seen = set()
        for k in range(len(nodes)):
            for order in (near_order, efficiency_order, prize_order):
                v = order[k]
                if v not in mixed_seen:
                    mixed_seen.add(v)
                    mixed_order.append(v)

        safe_route = None
        safe_cost = math.inf
        richest_seed = None
        shortest_seed = None
        efficient_seed = None
        seed_pool = []
        seed_signatures = set()
        feasible_seed_count = 0
        seed_pool_cap = 4 + int(math.sqrt(max(1, n)) * math.log1p(limit))

        def record_seed(route, known_cost=None):
            nonlocal safe_route, safe_cost, safe_result
            nonlocal richest_seed, shortest_seed, efficient_seed
            nonlocal feasible_seed_count

            candidate = list(route)
            cost = route_length(candidate)
            if not valid_route(candidate, cost):
                return False
            value = route_score(candidate)
            efficiency = value / max(cost, tiny)

            if safe_route is None:
                safe_route = candidate[:]
                safe_cost = cost
                safe_result = pack(candidate)
            elif cost < safe_cost:
                safe_route = candidate[:]
                safe_cost = cost

            if shortest_seed is None or cost < shortest_seed[0]:
                shortest_seed = (cost, candidate[:])
            if richest_seed is None or better(
                value, cost, richest_seed[0], richest_seed[1]
            ):
                richest_seed = (value, cost, candidate[:])
            if efficient_seed is None or efficiency > efficient_seed[0] or (
                efficiency == efficient_seed[0] and value > efficient_seed[1]
            ):
                efficient_seed = (efficiency, value, cost, candidate[:])

            feasible_seed_count += 1
            signature = frozenset(candidate)
            if signature not in seed_signatures:
                if len(seed_pool) < seed_pool_cap:
                    seed_pool.append(candidate[:])
                    seed_signatures.add(signature)
                else:
                    replacement = rng.randrange(feasible_seed_count)
                    if replacement < seed_pool_cap:
                        seed_signatures.discard(frozenset(seed_pool[replacement]))
                        seed_pool[replacement] = candidate[:]
                        seed_signatures.add(signature)
            return True

        seed_end = min(
            deadline,
            started + 0.045 * limit,
        )

        def scan_prefix(order, until):
            route = [depot]
            current = depot
            open_cost = 0.0
            for v in order:
                if clock() >= until:
                    break
                open_cost += distance[current][v]
                current = v
                route.append(v)
                if len(route) >= 3:
                    cost = open_cost + distance[current][depot]
                    if cost <= budget:
                        record_seed(route, cost)

        scan_prefix(near_order, seed_end)
        scan_prefix(efficiency_order, seed_end)
        scan_prefix(prize_order, seed_end)
        scan_prefix(mixed_order, seed_end)

        pair_i = 0
        pair_j = 1
        checked_pairs = 0
        while pair_i < len(mixed_order) - 1 and clock() < seed_end:
            if pair_j >= len(mixed_order):
                pair_i += 1
                pair_j = pair_i + 1
                continue
            a = mixed_order[pair_i]
            b = mixed_order[pair_j]
            pair_j += 1
            checked_pairs += 1
            cost = distance[depot][a] + distance[a][b] + distance[b][depot]
            if cost <= budget:
                record_seed([depot, a, b], cost)
            if (checked_pairs & 255) == 0 and clock() >= seed_end:
                break

        def shortest_cycle_seed(until):
            labels = [math.inf] * n
            owner = [-1] * n
            parent = [-1] * n
            settled = [False] * n
            heap = []
            for v in nodes:
                labels[v] = distance[depot][v]
                owner[v] = v
                heapq.heappush(heap, (labels[v], v))

            operations = 0

            def chain_to_root(v):
                chain = []
                seen = set()
                while v != -1 and v not in seen and len(chain) <= n:
                    seen.add(v)
                    chain.append(v)
                    v = parent[v]
                return chain

            while heap and clock() < until:
                du, u = heapq.heappop(heap)
                if settled[u] or du != labels[u]:
                    continue
                settled[u] = True
                for v in nodes:
                    if v == u:
                        continue
                    operations += 1
                    if owner[v] != owner[u] and owner[v] >= 0:
                        bound = labels[u] + distance[u][v] + labels[v]
                        if bound <= budget:
                            left = chain_to_root(u)
                            right = chain_to_root(v)
                            if left and right:
                                candidate = [depot] + list(reversed(left)) + right
                                if record_seed(candidate):
                                    return True
                    nd = labels[u] + distance[u][v]
                    if not settled[v] and nd < labels[v]:
                        labels[v] = nd
                        owner[v] = owner[u]
                        parent[v] = u
                        heapq.heappush(heap, (nd, v))
                    if (operations & 255) == 0 and clock() >= until:
                        return safe_route is not None
            return safe_route is not None

        if safe_route is None and clock() < deadline:
            shortest_cycle_seed(deadline)

        if safe_route is None:
            return safe_result

        best_route = safe_route[:]
        best_cost = route_length(best_route)
        best_score = route_score(best_route)
        last_improvement = clock()
        archive = []

        def archive_capacity():
            elapsed = max(0.0, clock() - started)
            return (
                10
                + int(0.35 * math.sqrt(max(1, n)))
                + int(2.0 * math.sqrt(elapsed + 1.0) + math.log1p(limit))
            )

        def register(route):
            nonlocal best_route, best_cost, best_score, last_improvement, archive
            candidate = list(route)
            cost = route_length(candidate)
            if not valid_route(candidate, cost):
                return None
            value = route_score(candidate)
            improved = better(value, cost, best_score, best_cost)
            score_tol = 1.0e-12 * max(1.0, abs(value), abs(best_score))
            score_improved = value > best_score + score_tol
            if improved:
                best_route = candidate[:]
                best_cost = cost
                best_score = value
                if score_improved:
                    last_improvement = clock()

            signature = frozenset(candidate)
            entry = (candidate[:], cost, value, signature)
            for i, old in enumerate(archive):
                if old[3] == signature:
                    if cost < old[1]:
                        archive[i] = entry
                    return value, cost, improved

            archive.append(entry)
            capacity = archive_capacity()
            if len(archive) > capacity:
                ranked = sorted(archive, key=lambda e: (-e[2], e[1]))
                elite_count = max(4, int(0.70 * capacity))
                keep = ranked[:elite_count]
                tail = ranked[elite_count:]
                rng.shuffle(tail)
                archive = keep + tail[: capacity - elite_count]
            return value, cost, improved

        initial_routes = []
        initial_seen = set()
        special = []
        if richest_seed is not None:
            special.append(richest_seed[2])
        if efficient_seed is not None:
            special.append(efficient_seed[3])
        if shortest_seed is not None:
            special.append(shortest_seed[1])
        special.extend(seed_pool)
        for route in special:
            signature = frozenset(route)
            if signature not in initial_seen:
                initial_seen.add(signature)
                initial_routes.append(route[:])
                register(route)

        # Arrays only speed up neighbourhood computation; routes are always stored as closed tours of distinct nodes.
        dist = np.asarray(distance, dtype=float)
        prize = np.asarray(scores, dtype=float)
        nrng = np.random.default_rng(seed_value)
        intensity = math.log1p(limit)
        cell_budget = 262144
        exact_rate = 1400000.0
        global_exact_done = False

        def insertion_table(vertices, route, until):
            vertices = np.asarray(vertices, dtype=np.intp)
            rr = np.asarray(route, dtype=np.intp)
            ss = np.roll(rr, -1)
            values = np.full((len(vertices), len(rr)), math.inf)
            step = max(1, cell_budget // max(1, len(rr)))
            for lo in range(0, len(vertices), step):
                if clock() >= until:
                    break
                vv = vertices[lo:lo + step]
                values[lo:lo + step] = (
                    dist[vv[:, None], rr] + dist[vv[:, None], ss]
                    - dist[rr, ss]
                )
            return values

        def improve_order(route, cost, until):
            candidate = list(route)
            changed = False
            while len(candidate) >= 4 and clock() < until:
                m = len(candidate)
                rr = np.asarray(candidate, dtype=np.intp)
                ss = np.roll(rr, -1)
                ee = dist[rr, ss]
                tol = 1.e-11 * max(1.0, abs(cost))
                best_delta, move = -tol, None
                step = max(1, cell_budget // m)
                columns = np.arange(m)[None, :]
                for lo in range(0, m - 2, step):
                    if clock() >= until:
                        break
                    hi = min(m - 2, lo + step)
                    rows = np.arange(lo, hi)[:, None]
                    delta = (dist[rr[lo:hi, None], rr]
                             + dist[ss[lo:hi, None], ss]
                             - ee[lo:hi, None] - ee)
                    delta[columns <= rows + 1] = math.inf
                    if lo == 0:
                        delta[0, m - 1] = math.inf
                    flat = int(np.argmin(delta))
                    value = float(delta.flat[flat])
                    if value < best_delta:
                        i, j = divmod(flat, m)
                        best_delta, move = value, (0, lo + i, j, 0)
                # After reaching a 2-opt local optimum, check relocation of single nodes and two-node blocks.
                if move is None and clock() < until:
                    for size in (1, 2):
                        if size + 2 > m or clock() >= until:
                            break
                        positions = np.arange(1, m - size + 1)
                        for lo in range(0, len(positions), step):
                            if clock() >= until:
                                break
                            pos = positions[lo:lo + step]
                            u, v = rr[pos], rr[pos + size - 1]
                            before, after = rr[pos - 1], rr[(pos + size) % m]
                            inner = np.zeros(len(pos)) if size == 1 else dist[u, v]
                            removal = (dist[before, after] - dist[before, u]
                                       - inner - dist[v, after])
                            invalid = ((columns >= pos[:, None] - 1)
                                       & (columns <= pos[:, None] + size - 1))
                            for reverse in range(size):
                                first, last = (u, v) if reverse == 0 else (v, u)
                                delta = (removal[:, None] + inner[:, None]
                                         + dist[first[:, None], rr]
                                         + dist[last[:, None], ss] - ee)
                                delta[invalid] = math.inf
                                flat = int(np.argmin(delta))
                                value = float(delta.flat[flat])
                                if value < best_delta:
                                    x, edge = divmod(flat, m)
                                    best_delta = value
                                    move = (size, int(pos[x]), edge, reverse)
                if move is None:
                    break
                kind, i, edge, reverse = move
                if kind == 0:
                    trial = candidate[:]
                    trial[i + 1:edge + 1] = reversed(trial[i + 1:edge + 1])
                else:
                    block = candidate[i:i + kind]
                    if reverse:
                        block.reverse()
                    anchor = candidate[edge]
                    trial = candidate[:i] + candidate[i + kind:]
                    at = trial.index(anchor) + 1
                    trial[at:at] = block
                actual = route_length(trial)
                if actual >= cost - tol:
                    break
                candidate, cost = trial, actual
                changed = True
            return candidate, cost, changed

        def candidate_pool(available, marginal, width, randomized):
            if len(available) <= width:
                return list(available)
            efficiency = sorted(available, key=lambda v: (
                -scores[v] / (max(0.0, marginal[v]) + 0.03 * distance_scale), v))
            by_prize = sorted(available, key=lambda v: -scores[v])
            chosen = []
            seen = set()
            deterministic = max(2, int(width * (0.72 if randomized else 1.0)))
            for a, b in zip(efficiency, by_prize):
                for v in (a, b):
                    if v not in seen:
                        chosen.append(v)
                        seen.add(v)
                if len(chosen) >= deterministic:
                    break
            if randomized and len(chosen) < width:
                rest = [v for v in available if v not in seen]
                chosen.extend(rng.sample(rest, min(width - len(chosen), len(rest))))
            return chosen[:width]

        def pair_insert(route, cost, until, randomized, marginal=None):
            used = set(route)
            available = [v for v in nodes if v not in used]
            if len(available) < 2 or clock() >= until:
                return route, cost, False
            if marginal is None:
                table = insertion_table(available, route, until)
                dd = np.min(table, axis=1)
                marginal = {v: float(dd[i]) for i, v in enumerate(available)}
            width = min(len(available), max(2, int(math.sqrt(n) * (1.0 + intensity))))
            pool = candidate_pool(available, marginal, width, randomized)
            vv = np.asarray(pool, dtype=np.intp)
            pp = prize[vv, None] + prize[vv]
            middle = dist[vv[:, None], vv]
            best_merit, best_move = -math.inf, None
            floor = 0.03 * distance_scale
            alpha = 0.7 + 0.6 * rng.random() if randomized else 1.0
            edges = list(range(len(route)))
            if randomized:
                rng.shuffle(edges)
            for e in edges:
                if clock() >= until:
                    break
                a, b = route[e], route[(e + 1) % len(route)]
                delta = dist[a, vv, None] + middle + dist[vv, b] - distance[a][b]
                feasible = delta + cost <= budget
                np.fill_diagonal(feasible, False)
                merit = pp / np.power(np.maximum(delta, 0.0) + floor, alpha)
                merit[~feasible] = -math.inf
                flat = int(np.argmax(merit))
                val = float(merit.flat[flat])
                if val > best_merit:
                    i, j = divmod(flat, len(pool))
                    best_merit, best_move = val, (e, pool[i], pool[j])
            if best_move is not None:
                e, a, b = best_move
                trial = route[:e + 1] + [a, b] + route[e + 1:]
                new_cost = route_length(trial)
                if new_cost <= budget:
                    return trial, new_cost, True
            return route, cost, False

        def greedy_fill(route, cost, until, randomized, mode, preferred=None):
            candidate = list(route)
            used = np.zeros(n, dtype=bool)
            used[candidate] = True
            best_delta = np.full(n, math.inf)
            best_anchor = np.full(n, -1, dtype=np.intp)
            changed = False
            alpha = (0.65 + 0.85 * rng.random()) if randomized else (0.75, 1.0, 1.3)[mode % 3]
            bias = nrng.uniform(-0.20, 0.20, n) if randomized else np.zeros(n)
            if preferred:
                bias[list(preferred)] += 0.22

            def refresh(vertices):
                if not len(vertices):
                    return
                best_delta[vertices] = math.inf
                best_anchor[vertices] = -1
                step = max(1, cell_budget // max(1, len(candidate)))
                rr = np.asarray(candidate)
                for lo in range(0, len(vertices), step):
                    if clock() >= until:
                        break
                    vv = vertices[lo:lo + step]
                    table = insertion_table(vv, candidate, until)
                    where = np.argmin(table, axis=1)
                    best_delta[vv] = table[np.arange(len(vv)), where]
                    best_anchor[vv] = rr[where]

            refresh(np.flatnonzero(~used))
            while clock() < until and not bool(np.all(used)):
                feasible = (~used) & (cost + best_delta <= budget)
                if not bool(np.any(feasible)):
                    candidate, cost, paired = pair_insert(
                        candidate, cost, until, randomized, best_delta)
                    if paired:
                        changed = True
                        used[candidate] = True
                        refresh(np.flatnonzero(~used))
                        continue
                    break
                scale = max(distance_scale, abs(cost) / len(candidate), tiny)
                if mode % 3 == 1:
                    merit = prize / score_scale - alpha * np.maximum(best_delta, 0.0) / scale + bias
                else:
                    merit = np.log(np.maximum(prize, tiny)) - alpha * np.log(
                        np.maximum(best_delta, 0.0) + 0.015 * scale) + bias
                merit[~feasible] = -math.inf
                v = int(np.argmax(merit))
                a = int(best_anchor[v])
                edge = candidate.index(a)
                b = candidate[(edge + 1) % len(candidate)]
                trial = candidate[:edge + 1] + [v] + candidate[edge + 1:]
                actual = route_length(trial)
                if actual > budget:
                    best_delta[v] = math.inf
                    continue
                candidate, cost = trial, actual
                changed = True
                used[v] = True
                best_delta[v] = math.inf
                # Only nodes whose cheapest insertion edge was removed need to rescan the whole route.
                damaged = np.flatnonzero((~used) & (best_anchor == a))
                refresh(damaged)
                remaining = np.flatnonzero(~used)
                for left, right in ((a, v), (v, b)):
                    inc = dist[remaining, left] + dist[remaining, right] - distance[left][right]
                    mask = inc < best_delta[remaining]
                    affected = remaining[mask]
                    best_delta[affected] = inc[mask]
                    best_anchor[affected] = left
            return candidate, cost, changed

        def generalized_exchange(route, cost, until, randomized):
            used = set(route)
            available = [v for v in nodes if v not in used]
            m = len(route)
            if not available or clock() >= until:
                return list(route), cost, False
            rr = np.asarray(route, dtype=np.intp)
            positions = np.arange(1, m)
            old = rr[positions]
            before = rr[positions - 1]
            after = rr[(positions + 1) % m]
            saving = dist[before, old] + dist[old, after] - dist[before, after]
            best_gain, best_cost2, best_move = 0.0, cost, None
            marginal = {}
            step = max(1, cell_budget // max(m, 1))
            for lo in range(0, len(available), step):
                if clock() >= until:
                    break
                vv = np.asarray(available[lo:lo + step], dtype=np.intp)
                table = insertion_table(vv, route, until)
                first = np.argpartition(table, 2, axis=1)[:, :3]
                first_values = np.take_along_axis(table, first, axis=1)
                mins = np.min(first_values, axis=1)
                for j, v in enumerate(vv):
                    marginal[int(v)] = float(mins[j])
                existing = np.full((len(vv), m - 1), math.inf)
                existing_edge = np.full((len(vv), m - 1), -1, dtype=np.intp)
                for k in range(3):
                    edge = first[:, k, None]
                    value = first_values[:, k, None]
                    legal = (edge != positions - 1) & (edge != positions)
                    take = legal & (value < existing)
                    existing = np.where(take, value, existing)
                    existing_edge = np.where(take, edge, existing_edge)
                bridge = (dist[vv[:, None], before] + dist[vv[:, None], after]
                          - dist[before, after])
                via_bridge = bridge < existing
                inc = np.minimum(bridge, existing)
                proposed = cost - saving + inc
                gain = prize[vv, None] - prize[old]
                legal = (proposed <= budget) & ((gain > best_gain + 1.e-10)
                         | ((np.abs(gain - best_gain) <= 1.e-10) & (proposed < best_cost2 - 1.e-10)))
                if bool(np.any(legal)):
                    merit = np.where(legal, gain, -math.inf)
                    high = float(np.max(merit))
                    tied = legal & (np.abs(gain - high) <= 1.e-10)
                    tie_cost = np.where(tied, proposed, math.inf)
                    flat = int(np.argmin(tie_cost))
                    row, col = divmod(flat, m - 1)
                    pos = int(positions[col])
                    edge = pos - 1 if via_bridge[row, col] else int(existing_edge[row, col])
                    best_gain, best_cost2 = high, float(proposed[row, col])
                    best_move = (1, pos, [int(vv[row])], int(rr[edge]))
                # Remove two consecutive nodes and insert one, without requiring the intermediate route after removal to be feasible.
                if m >= 4 and clock() < until:
                    pos2 = np.arange(1, m - 1)
                    aa, uu, ww, bb = rr[pos2 - 1], rr[pos2], rr[pos2 + 1], rr[(pos2 + 2) % m]
                    removed_edges = dist[aa, uu] + dist[uu, ww] + dist[ww, bb]
                    proposed2 = cost - removed_edges + dist[vv[:, None], aa] + dist[vv[:, None], bb]
                    gain2 = prize[vv, None] - prize[uu] - prize[ww]
                    legal2 = (proposed2 <= budget) & (gain2 > best_gain + 1.e-10)
                    if bool(np.any(legal2)):
                        flat = int(np.argmax(np.where(legal2, gain2, -math.inf)))
                        row, col = divmod(flat, len(pos2))
                        pos = int(pos2[col])
                        best_gain, best_cost2 = float(gain2[row, col]), float(proposed2[row, col])
                        best_move = (2, pos, [int(vv[row])], int(rr[pos - 1]))
            # The joint one-for-two neighbourhood can cross barriers where no single-node swap-in increases the score.
            if len(available) >= 2 and clock() < until:
                for v in available:
                    marginal.setdefault(v, distance[depot][v])
                width = min(len(available), max(2, int(math.sqrt(n) * (1.0 + intensity))))
                pool = candidate_pool(available, marginal, width, randomized)
                vv = np.asarray(pool, dtype=np.intp)
                pair_prize = prize[vv, None] + prize[vv]
                middle = dist[vv[:, None], vv]
                order = list(range(1, m))
                if randomized:
                    rng.shuffle(order)
                for pos in order:
                    if clock() >= until:
                        break
                    a, v, b = route[pos - 1], route[pos], route[(pos + 1) % m]
                    proposed = (cost - distance[a][v] - distance[v][b]
                                + dist[a, vv, None] + middle + dist[vv, b])
                    gain = pair_prize - scores[v]
                    legal = (proposed <= budget) & (gain > best_gain + 1.e-10)
                    np.fill_diagonal(legal, False)
                    if bool(np.any(legal)):
                        high = float(np.max(np.where(legal, gain, -math.inf)))
                        tie_cost = np.where(legal & (np.abs(gain - high) <= 1.e-10), proposed, math.inf)
                        flat = int(np.argmin(tie_cost))
                        i, j = divmod(flat, len(pool))
                        best_gain, best_cost2 = high, float(proposed.flat[flat])
                        best_move = (1, pos, [pool[i], pool[j]], a)
            if best_move is None:
                return list(route), cost, False
            size, pos, block, anchor = best_move
            trial = route[:pos] + route[pos + size:]
            at = trial.index(anchor) + 1
            trial[at:at] = block
            actual = route_length(trial)
            if actual <= budget and better(route_score(trial), actual, route_score(route), cost):
                return trial, actual, True
            return list(route), cost, False

        def repair(route, cost, until, protected=None):
            candidate = list(route)
            protected = set() if protected is None else protected
            if cost > budget and clock() < until:
                order_end = clock() + 0.28 * max(0.0, until - clock())
                candidate, cost, _ = improve_order(candidate, cost, order_end)
            while cost > budget and len(candidate) > 3 and clock() < until:
                best_key, move = -math.inf, None
                m = len(candidate)
                # Block removal can cross non-metric barriers where single-node removal increases distance.
                block_depth = min(m - 3, max(1, int(1 + math.sqrt(intensity))))
                for size in range(1, block_depth + 1):
                    for i in range(1, m - size + 1):
                        block = candidate[i:i + size]
                        a = candidate[i - 1]
                        b = candidate[(i + size) % m]
                        old_path = distance[a][block[0]] + distance[block[-1]][b]
                        old_path += sum(distance[block[j]][block[j + 1]] for j in range(size - 1))
                        saving = old_path - distance[a][b]
                        if saving <= 0.0:
                            continue
                        loss = sum(scores[v] * (4.0 if v in protected else 1.0) for v in block)
                        key = min(saving, max(cost - budget, 0.05 * distance_scale)) / max(loss, tiny)
                        key *= 0.88 + 0.24 * rng.random()
                        if key > best_key:
                            best_key, move = key, (i, size)
                    if clock() >= until:
                        break
                if move is None:
                    break
                i, size = move
                candidate = candidate[:i] + candidate[i + size:]
                cost = route_length(candidate)
            return candidate, cost

        def polish(route, until, randomized, preferred=None):
            candidate = list(route)
            cost = route_length(candidate)
            if cost > budget:
                candidate, cost = repair(candidate, cost, until, preferred)
            if cost > budget:
                return candidate, cost
            phase = rng.randrange(3) if randomized else 0
            while clock() < until:
                old_value, old_cost = route_score(candidate), cost
                # Reserve filling time first, so sorting on long routes does not monopolise a single search.
                order_end = min(until, clock() + 0.35 * max(0.0, until - clock()))
                candidate, cost, moved = improve_order(candidate, cost, order_end)
                candidate, cost, added = greedy_fill(candidate, cost, until, randomized, phase, preferred)
                exchanged = False
                if not added and clock() < until:
                    candidate, cost, exchanged = generalized_exchange(candidate, cost, until, randomized)
                if not (moved or added or exchanged):
                    break
                if not better(route_score(candidate), cost, old_value, old_cost):
                    break
                phase = (phase + 1) % 3
            return candidate, cost

        def destroy(route, cost, target, mode, until, protected=None):
            candidate = list(route)
            removed = set()
            protected = set() if protected is None else set(protected)
            anchor = candidate[rng.randrange(1, len(candidate))]
            cut_anchor = rng.randrange(1, len(candidate))
            while len(candidate) > 3 and len(removed) < target and clock() < until:
                choices = []
                m = len(candidate)
                for i in range(1, m):
                    v = candidate[i]
                    if v in protected:
                        continue
                    a, b = candidate[i - 1], candidate[(i + 1) % m]
                    saving = distance[a][v] + distance[v][b] - distance[a][b]
                    if mode == 0:
                        key = rng.random()
                    elif mode == 1:
                        key = (max(0.0, saving) + 0.03 * distance_scale) / (scores[v] + 0.02 * score_scale)
                        key *= 0.75 + 0.50 * rng.random()
                    elif mode == 2:
                        key = -scores[v] / score_scale + 0.3 * rng.random()
                    elif mode == 3:
                        key = -abs(i - min(cut_anchor, m - 1)) + rng.random()
                    else:
                        key = -distance[anchor][v] / distance_scale + 0.15 * rng.random()
                    choices.append((key, i, v))
                if not choices:
                    break
                _, i, v = max(choices)
                candidate.pop(i)
                removed.add(v)
                # Allow the private working route to be temporarily over budget, leaving repair to joint sorting and block removal.
                cost = route_length(candidate)
            return candidate, cost, removed

        def force_target(route, cost, until, elapsed):
            used = set(route)
            available = [v for v in nodes if v not in used]
            if not available or clock() >= until:
                return list(route), cost, set()
            edges = np.zeros(len(available), dtype=np.intp)
            delta = np.full(len(available), math.inf)
            step = max(1, cell_budget // max(1, len(route)))
            for lo in range(0, len(available), step):
                if clock() >= until:
                    break
                table = insertion_table(available[lo:lo + step], route, until)
                ee = np.argmin(table, axis=1)
                edges[lo:lo + len(ee)] = ee
                delta[lo:lo + len(ee)] = table[np.arange(len(ee)), ee]
            if not bool(np.any(np.isfinite(delta))):
                return list(route), cost, set()
            alpha = 0.4 + rng.random()
            merit = (np.log(prize[available]) - alpha * np.log(
                np.maximum(delta, 0.0) + 0.1 * distance_scale)
                + nrng.uniform(-0.65, 0.65, len(available)))
            j = int(np.argmax(merit))
            v, edge = available[j], int(edges[j])
            trial = route[:edge + 1] + [v] + route[edge + 1:]
            cost2 = route_length(trial)
            trial, cost2 = repair(trial, cost2, until, {v})
            if cost2 <= budget:
                return trial, cost2, {v}
            return list(route), cost, {v}

        def exact_capacity(seconds):
            k = 2
            # Transition count and memory jointly bound a single neighbourhood; the wall clock can still interrupt at any time.
            work = exact_rate * max(0.0, seconds) * 0.8
            while k < n - 1 and ((1 << (k + 1)) * (k + 1) * k / 4 <= work
                                and (1 << (k + 1)) * (k + 1) <= 2400000):
                k += 1
            return min(n - 1, k)

        def exact_segment(route, cost, until):
            nonlocal exact_rate, global_exact_done
            m = len(route)
            remaining_time = until - clock()
            if m < 3 or remaining_time <= 0.001:
                return list(route), cost, False
            target_k = exact_capacity(remaining_time)
            is_global = target_k >= n - 1
            if is_global and global_exact_done:
                return list(route), cost, False
            if is_global:
                left, right = 1, m - 1
                pool = list(nodes)
            else:
                segment_cap = min(m - 1, max(1, target_k // 2))
                segment_size = 1 + rng.randrange(segment_cap)
                left = 1 + rng.randrange(m - segment_size)
                right = left + segment_size - 1
                pool = list(route[left:right + 1])
                used = set(route)
                available = [v for v in nodes if v not in used]
                aa = route[left - 1:right + 1]
                bb = route[left:right + 1] + [route[(right + 1) % m]]
                marginal = {}
                for v in available:
                    marginal[v] = min(distance[a][v] + distance[v][b] - distance[a][b]
                                      for a, b in zip(aa, bb))
                slots = max(0, target_k - len(pool))
                if slots:
                    pool.extend(candidate_pool(available, marginal, slots, True))
            segment = route[left:right + 1]
            a, b = route[left - 1], route[(right + 1) % m]
            old_path = distance[a][segment[0]] + distance[segment[-1]][b]
            old_path += sum(distance[segment[j]][segment[j + 1]] for j in range(len(segment) - 1))
            fixed = cost - old_path
            allowance = budget - fixed
            k = len(pool)
            if k < 1 or allowance <= 0.0 or clock() >= until:
                return list(route), cost, False
            states = 1 << k
            full = states - 1
            dp = array('d', [math.inf]) * (states * k)
            parent = array('b', [-2]) * (states * k)
            subset_score = array('d', [0.0]) * states
            bits_index = {1 << j: j for j in range(k)}
            pp = [scores[v] for v in pool]
            dd = [[distance[u][v] for v in pool] for u in pool]
            ends = [distance[v][b] for v in pool]
            for j in range(k):
                dp[(1 << j) * k + j] = distance[a][pool[j]]
                parent[(1 << j) * k + j] = -1
            outside = m - len(segment)
            winning_mask, winning_last = 0, -1
            winning_score, winning_cost = -math.inf, math.inf
            transitions = 0
            dp_started = clock()
            complete = True
            for mask in range(1, states):
                if (mask & 31) == 0 and clock() >= until:
                    complete = False
                    break
                bit = mask & -mask
                subset_score[mask] = subset_score[mask ^ bit] + pp[bits_index[bit]]
                base = mask * k
                rest = full ^ mask
                last_bits = mask
                enough = outside + mask.bit_count() >= 3
                value_score = subset_score[mask]
                while last_bits:
                    bit = last_bits & -last_bits
                    last_bits ^= bit
                    j = bits_index[bit]
                    value = dp[base + j]
                    if not math.isfinite(value):
                        continue
                    completion = value + ends[j]
                    if enough and completion <= allowance and (
                        value_score > winning_score or (value_score == winning_score and completion < winning_cost)):
                        winning_mask, winning_last = mask, j
                        winning_score, winning_cost = value_score, completion
                    if value >= allowance:
                        continue
                    next_bits = rest
                    row = dd[j]
                    while next_bits:
                        bit2 = next_bits & -next_bits
                        next_bits ^= bit2
                        nxt = bits_index[bit2]
                        nv = value + row[nxt]
                        idx = (mask | bit2) * k + nxt
                        if nv < dp[idx]:
                            dp[idx] = nv
                            parent[idx] = j
                        transitions += 1
            elapsed_dp = clock() - dp_started
            if elapsed_dp > 0.001 and transitions:
                exact_rate = 0.7 * exact_rate + 0.3 * transitions / elapsed_dp
            if is_global and complete:
                global_exact_done = True
            if winning_last < 0:
                return list(route), cost, False
            order = []
            mask, last = winning_mask, winning_last
            while last >= 0:
                order.append(pool[last])
                prev = parent[mask * k + last]
                mask ^= 1 << last
                last = prev
            order.reverse()
            trial = route[:left] + order + route[right + 1:]
            new_cost = route_length(trial)
            if valid_route(trial, new_cost) and better(route_score(trial), new_cost, route_score(route), cost):
                return trial, new_cost, True
            return list(route), cost, False

        baseline_end = min(
            deadline,
            started + 0.16 * limit,
        )
        baseline_index = 0
        while baseline_index < len(initial_routes) and clock() < baseline_end:
            remaining = len(initial_routes) - baseline_index
            local_end = min(
                baseline_end,
                clock() + max(0.004, (baseline_end - clock()) / max(1, remaining)),
            )
            candidate, _ = polish(
                initial_routes[baseline_index], local_end, False
            )
            register(candidate)
            baseline_index += 1

        exact_end = min(deadline, clock() + 0.14 * limit)
        if not global_exact_done and n - 1 <= exact_capacity(exact_end - clock()):
            candidate, _, _ = exact_segment(best_route, best_cost, exact_end)
            register(candidate)

        current_route = best_route[:]
        current_cost = best_cost
        current_score = best_score
        destroy_weights = [1.0] * 5

        def choose_parent():
            if not archive:
                return best_route[:]
            draw = rng.random()
            if draw < 0.22:
                return current_route[:]
            if draw < 0.48:
                return best_route[:]
            ranked = sorted(archive, key=lambda e: (-e[2], e[1]))
            rank = min(
                len(ranked) - 1,
                int((rng.random() ** 2.15) * len(ranked)),
            )
            return ranked[rank][0][:]

        def choose_destroy_mode():
            total = sum(destroy_weights)
            draw = rng.random() * total
            cumulative = 0.0
            for i, weight in enumerate(destroy_weights):
                cumulative += weight
                if draw <= cumulative:
                    return i
            return len(destroy_weights) - 1

        def fresh_seed(until, depth):
            if seed_pool:
                base = seed_pool[rng.randrange(len(seed_pool))][:]
            else:
                base = safe_route[:]
            base_cost = route_length(base)
            price = rng.random() * score_scale / max(distance_scale, tiny)
            best_key = route_score(base) - price * base_cost
            total = len(mixed_order)
            while total >= 2 and clock() < until:
                exponent = 1.15 + 1.9 * depth
                ia = (
                    rng.randrange(total)
                    if rng.random() < 0.20
                    else min(total - 1, int((rng.random() ** exponent) * total))
                )
                ib = (
                    rng.randrange(total)
                    if rng.random() < 0.20
                    else min(total - 1, int((rng.random() ** exponent) * total))
                )
                if ia == ib:
                    continue
                a = mixed_order[ia]
                b = mixed_order[ib]
                candidate_cost = (
                    distance[depot][a] + distance[a][b] + distance[b][depot]
                )
                if candidate_cost <= budget:
                    value = scores[depot] + scores[a] + scores[b]
                    key = (value - price * candidate_cost) * (
                        0.90 + 0.20 * rng.random()
                    )
                    if key > best_key:
                        base = [depot, a, b]
                        base_cost = candidate_cost
                        best_key = key
            return base, base_cost

        while clock() < deadline:
            try:
                now = clock()
                elapsed = max(0.0, now - started)
                stagnation_age = max(0.0, now - last_improvement)
                depth = min(1.0, (math.log1p(limit) + math.log1p(elapsed + 1.0)) / 7.0)
                stagnation = min(
                    1.0,
                    stagnation_age
                    / max(1.0, 0.75 + math.sqrt(elapsed + 1.0)),
                )

                parent = choose_parent()
                parent_cost = route_length(parent)
                parent_score = route_score(parent)
                candidate = parent[:]
                candidate_cost = parent_cost
                before_best_score = best_score
                mode_used = -1

                exact_probability = 0.0 if global_exact_done else 0.08 + 0.14 * depth + 0.06 * stagnation
                force_probability = 0.14 + 0.10 * stagnation
                fresh_probability = 0.06 + 0.08 * stagnation
                crossover_probability = 0.11 + 0.10 * depth
                roll = rng.random()

                if roll < exact_probability:
                    quantum = limit * (0.007 + 0.011 * depth) * (0.78 + 0.48 * rng.random())
                    local_end = min(deadline, clock() + quantum)
                    candidate, candidate_cost, _ = exact_segment(
                        candidate, candidate_cost, local_end
                    )
                    if clock() < local_end:
                        candidate, candidate_cost = polish(
                            candidate, local_end, True
                        )

                else:
                    quantum = min(0.045 * limit, (0.006 + 0.000012 * n
                        * math.sqrt(max(3, len(candidate)))) * (1.0 + 0.3 * intensity))
                    quantum *= 0.70 + 0.78 * rng.random()
                    local_end = min(deadline, clock() + quantum)
                    preferred = None

                    if roll < exact_probability + force_probability:
                        candidate, candidate_cost, preferred = force_target(
                            candidate, candidate_cost, local_end, elapsed
                        )

                    elif roll < exact_probability + force_probability + fresh_probability:
                        seed_end = min(
                            local_end,
                            clock() + 0.20 * max(0.0, local_end - clock()),
                        )
                        candidate, candidate_cost = fresh_seed(seed_end, depth)

                    else:
                        protected = None
                        cross_limit = (
                            exact_probability
                            + force_probability
                            + fresh_probability
                            + crossover_probability
                        )
                        if roll < cross_limit and len(archive) >= 2:
                            other = choose_parent()
                            parent_set = set(candidate)
                            other_set = set(other)
                            preferred = parent_set | other_set
                            preferred.discard(depot)
                            protected = (parent_set & other_set) - {depot}

                        removable = max(0, len(candidate) - 3)
                        if removable > 0:
                            fraction = min(
                                0.82,
                                0.055
                                + 0.060 * math.log1p(elapsed + 1.0)
                                + 0.30 * stagnation
                                + 0.18 * rng.random(),
                            )
                            target_cap = min(
                                removable,
                                max(1, int((len(candidate) - 1) * fraction)),
                            )
                            target = 1 + min(
                                target_cap - 1,
                                int((rng.random() ** 1.45) * target_cap),
                            )
                            mode_used = choose_destroy_mode()
                            destroy_end = min(
                                local_end,
                                clock() + 0.25 * max(0.0, local_end - clock()),
                            )
                            candidate, candidate_cost, removed = destroy(
                                candidate,
                                candidate_cost,
                                target,
                                mode_used,
                                destroy_end,
                                protected,
                            )
                            if preferred is None:
                                preferred = set(nodes) - set(parent)

                        if (
                            len(candidate) > 4
                            and rng.random() < 0.16 + 0.32 * stagnation
                        ):
                            i = 1 + rng.randrange(len(candidate) - 2)
                            j = i + 1 + rng.randrange(len(candidate) - i - 1)
                            trial = candidate[:]
                            trial[i : j + 1] = reversed(trial[i : j + 1])
                            trial_cost = route_length(trial)
                            if trial_cost <= budget:
                                candidate = trial
                                candidate_cost = trial_cost

                    candidate, candidate_cost = polish(
                        candidate, local_end, True, preferred
                    )

                outcome = register(candidate)
                if outcome is None:
                    continue
                candidate_score, candidate_cost, globally_improved = outcome

                if mode_used >= 0:
                    if globally_improved and best_score > before_best_score:
                        reward = 6.0
                    elif candidate_score > parent_score:
                        reward = 2.5
                    elif candidate_score == parent_score and candidate_cost < parent_cost:
                        reward = 1.2
                    else:
                        reward = 0.35
                    destroy_weights[mode_used] = (
                        0.86 * destroy_weights[mode_used] + 0.14 * reward
                    )

                tolerance = 1.0e-12 * max(
                    1.0, abs(candidate_score), abs(current_score)
                )
                accept = candidate_score > current_score + tolerance or (
                    abs(candidate_score - current_score) <= tolerance
                    and candidate_cost < current_cost
                )
                if not accept:
                    average_prize = max(
                        tiny,
                        abs(best_score - scores[depot])
                        / max(1, len(best_route) - 1),
                    )
                    temperature = average_prize * (0.035 + 0.95 * stagnation)
                    loss = max(0.0, current_score - candidate_score)
                    accept = rng.random() < math.exp(
                        -min(60.0, loss / max(temperature, tiny))
                    )
                if accept:
                    current_route = candidate[:]
                    current_cost = candidate_cost
                    current_score = candidate_score
                elif rng.random() < 0.04 + 0.16 * stagnation:
                    current_route = best_route[:]
                    current_cost = best_cost
                    current_score = best_score
            except BaseException:
                continue

        final_cost = route_length(best_route)
        final_route = best_route if valid_route(best_route, final_cost) else safe_route
        return pack(final_route)

    except BaseException:
        try:
            if valid_route(best_route):
                return pack(best_route)
            if valid_route(safe_route):
                return pack(safe_route)
        except BaseException:
            pass
        return safe_result
