"""Shortest-path augmentation and mixed-integer neighbourhood search for the quota tree
Combines multi-start path augmentation, subtree pruning and time-limited flow-model neighbourhood optimisation, continually saving the best feasible tree.
"""

import heapq
import math
import random
import time

try:
    import mip
except Exception:
    mip = None


def solve(instance: dict, time_limit_s: float) -> dict:
    started = time.perf_counter()
    n = len(instance["node_types"])
    original_edges = instance["edges"]
    m = len(original_edges)
    answer = {"node_selected": [0] * n, "edge_selected": [0] * m}
    if not n:
        return answer
    labels = [str(t).strip().lower() for t in instance["node_types"]]
    stations = [i for i, t in enumerate(labels)
                if "substation" in t or "sub_station" in t or "substation" in t
                or t in ("station", "sub", "ss", "s", "root", "source")]
    if not stations:
        stations = [i for i, t in enumerate(labels)
                    if "steiner" not in t and "turbine" not in t
                    and "wind" not in t and "wind" not in t and "steiner" not in t
                    and instance["annual_energy_yields"][i] == 0]
    if not stations:
        stations = [0]
    root = stations[0]
    original_adj = [[] for _ in range(n)]
    for e, pair in enumerate(original_edges):
        u, v = int(pair[0]), int(pair[1])
        original_adj[u].append((v, e))
        original_adj[v].append((u, e))
    # Before any optimisation, save a spanning tree of the connected component containing the root.
    order = [root]
    answer["node_selected"][root] = 1
    for u in order:
        for v, e in original_adj[u]:
            if not answer["node_selected"][v]:
                answer["node_selected"][v] = 1
                answer["edge_selected"][e] = 1
                order.append(v)
    try:
        _search(instance, float(time_limit_s), started, stations, order, answer)
    except Exception:
        # If the optimiser is unavailable or fails midway, keep the complete solution verified earlier.
        pass
    return answer


def _search(data, budget, started, stations, reachable, answer):
    deadline = started + max(0.0, budget)
    reserve = min(0.15, max(0.002, budget * 0.025))
    stop = deadline - reserve
    if time.perf_counter() >= stop:
        return
    rng = random.Random(81371)
    original_n = len(data["node_types"])
    original_m = len(data["edges"])
    alpha = float(data["alpha"])
    quota = float(data["minimum_energy_quota"])
    station_set = set(stations)
    # The zero-weight complete graph between substations is contracted into the root; a zero-weight star tree is restored on output.
    vertex = [stations[0]] + [v for v in reachable if v not in station_set]
    index = {v: i for i, v in enumerate(vertex)}
    for v in stations:
        index[v] = 0
    size = len(vertex)
    energy = [0.0] + [float(data["annual_energy_yields"][v]) for v in vertex[1:]]
    cost = [0.0] + [alpha * float(data["installation_costs"][v])
                       + (1.0 - alpha) * float(data["node_scenic_impacts"][v])
                       for v in vertex[1:]]
    star = {}
    parallel = {}
    for e, pair in enumerate(data["edges"]):
        a, b = int(pair[0]), int(pair[1])
        w = alpha * float(data["edge_cable_costs"][e]) + (1 - alpha) * float(data["edge_scenic_impacts"][e])
        if a == stations[0] and b in station_set and a != b:
            if b not in star or w < star[b][0]:
                star[b] = (w, e)
        if b == stations[0] and a in station_set and a != b:
            if a not in star or w < star[a][0]:
                star[a] = (w, e)
        if a not in index or b not in index:
            continue
        u, v = index[a], index[b]
        if u == v:
            continue
        if u > v:
            u, v = v, u
        if (u, v) not in parallel or w < parallel[u, v][0]:
            parallel[u, v] = (w, e)
    if len(star) != len(stations) - 1:
        return
    edges = [(u, v, w, e) for (u, v), (w, e) in parallel.items()]
    edge_count = len(edges)
    adj = [[] for _ in range(size)]
    for e, (u, v, w, _) in enumerate(edges):
        adj[u].append((v, e))
        adj[v].append((u, e))
    sorted_edges = sorted(range(edge_count), key=lambda e: edges[e][2])
    nonnegative = all(c >= 0 for c in cost) and all(e[2] >= 0 for e in edges)
    strength = math.sqrt(math.log1p(max(0.0, budget)))
    capacity = max(2, int(2 + strength * math.log2(size + 1)))
    best = None
    pool = []

    def value(nodes, tree):
        return math.fsum(cost[v] for v in nodes) + math.fsum(edges[e][2] for e in tree)

    def remember(candidate):
        nonlocal best
        if candidate is None:
            return False
        nodes, tree = candidate
        if 0 not in nodes or len(tree) != len(nodes) - 1:
            return False
        if math.fsum(energy[v] for v in nodes) < quota:
            return False
        seen = {0}
        local = {v: [] for v in nodes}
        for e in tree:
            u, v = edges[e][:2]
            if u not in nodes or v not in nodes:
                return False
            local[u].append(v)
            local[v].append(u)
        todo = [0]
        for u in todo:
            for v in local[u]:
                if v not in seen:
                    seen.add(v)
                    todo.append(v)
        if len(seen) != len(nodes):
            return False
        score = value(nodes, tree)
        if best is None or score < best[0] - 1e-10 * max(1.0, abs(best[0])):
            best = (score, set(nodes), set(tree))
            ns = [0] * original_n
            es = [0] * original_m
            for v in stations:
                ns[v] = 1
            for v in nodes:
                ns[vertex[v]] = 1
            for _, e in star.values():
                es[e] = 1
            for e in tree:
                es[edges[e][3]] = 1
            answer["node_selected"] = ns
            answer["edge_selected"] = es
            improved = True
        else:
            improved = False
        signature = frozenset(nodes)
        if not any(item[3] == signature for item in pool):
            pool.append((score, set(nodes), set(tree), signature))
            pool.sort(key=lambda item: item[0])
            if len(pool) > capacity:
                pool.pop()
        return improved

    def mst(nodes):
        parent = list(range(size))
        rank = [0] * size

        def find(a):
            while a != parent[a]:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        tree = set()
        for e in sorted_edges:
            u, v = edges[e][:2]
            if u not in nodes or v not in nodes:
                continue
            a, b = find(u), find(v)
            if a != b:
                if rank[a] < rank[b]:
                    a, b = b, a
                parent[b] = a
                if rank[a] == rank[b]:
                    rank[a] += 1
                tree.add(e)
                if len(tree) == len(nodes) - 1:
                    break
        if len(tree) != len(nodes) - 1:
            return None
        return set(nodes), tree

    def rooted(nodes, tree):
        ta = {v: [] for v in nodes}
        for e in tree:
            u, v = edges[e][:2]
            ta[u].append((v, e))
            ta[v].append((u, e))
        parent = {0: -1}
        pe = {}
        order = [0]
        children = {v: [] for v in nodes}
        for u in order:
            for v, e in ta[u]:
                if v not in parent:
                    parent[v] = u
                    pe[v] = e
                    children[u].append(v)
                    order.append(v)
        return parent, pe, order, children

    def prune(candidate, exponent=1.0):
        if candidate is None:
            return None
        nodes, tree = set(candidate[0]), set(candidate[1])
        parent, pe, order, children = rooted(nodes, tree)
        se = {v: energy[v] for v in nodes}
        sc = {v: cost[v] + (edges[pe[v]][2] if v else 0.0) for v in nodes}
        for v in reversed(order[1:]):
            se[parent[v]] += se[v]
            sc[parent[v]] += sc[v]
        total = math.fsum(energy[v] for v in nodes)
        unit = max(1e-12, math.fsum(max(0.0, e) for e in energy) / max(1, size))
        version = [0] * size
        heap = []

        def push(v):
            if v and sc[v] >= -1e-12:
                ratio = sc[v] / max(se[v], unit * 1e-9) ** exponent
                heapq.heappush(heap, (-ratio, rng.random(), v, version[v]))

        for v in order[1:]:
            push(v)
        while heap and time.perf_counter() < stop:
            _, _, v, ver = heapq.heappop(heap)
            if v not in nodes or ver != version[v]:
                continue
            eps = 1e-11 * max(1.0, abs(total), abs(quota))
            if total - se[v] < quota - eps:
                continue
            removed = [v]
            for u in removed:
                removed.extend(c for c in children[u] if c in nodes)
            if abs(total - se[v] - quota) <= eps:
                removed_set = set(removed)
                after = math.fsum(energy[u] for u in nodes if u not in removed_set)
                if after < quota:
                    continue
            else:
                after = total - se[v]
            lost_e, lost_c = se[v], sc[v]
            for u in removed:
                nodes.remove(u)
                tree.discard(pe[u])
            total = after
            u = parent[v]
            while u != -1:
                se[u] -= lost_e
                sc[u] -= lost_c
                version[u] += 1
                push(u)
                u = parent[u]
        if math.fsum(energy[v] for v in nodes) < quota:
            return candidate
        return nodes, tree

    baseline = mst(set(range(size)))
    remember(baseline)
    remember(prune(baseline))
    if best is None or time.perf_counter() >= stop:
        return

    def grow(seed_nodes, seed_tree, noisy, exponent, preferred, forced=None):
        nodes, tree = set(seed_nodes), set(seed_tree)
        total = math.fsum(energy[v] for v in nodes)
        dist = [math.inf] * size
        pred = [-1] * size
        prev_edge = [-1] * size
        heap = []
        amplitude = min(1.6, (0.2 + 0.22 * strength) * noisy)
        ec = [max(0.0, item[2]) * math.exp(amplitude * (rng.random() - 0.5))
              * (0.75 if noisy and e in preferred else 1.0)
              for e, item in enumerate(edges)]
        nc = [max(0.0, c) * math.exp(amplitude * (rng.random() - 0.5)) for c in cost]
        target_noise = [math.exp(amplitude * 0.35 * (rng.random() - 0.5)) for _ in range(size)]
        for v in nodes:
            dist[v] = 0.0
            heapq.heappush(heap, (0.0, v))
        while total < quota or forced is not None:
            if time.perf_counter() >= stop:
                return None
            pops = 0
            while heap:
                d, u = heapq.heappop(heap)
                if d != dist[u]:
                    continue
                for v, e in adj[u]:
                    if v in nodes:
                        continue
                    nd = d + ec[e] + nc[v]
                    if nd < dist[v]:
                        dist[v] = nd
                        pred[v] = u
                        prev_edge[v] = e
                        heapq.heappush(heap, (nd, v))
                pops += 1
                if pops % 256 == 0 and time.perf_counter() >= stop:
                    return None
            if forced is not None and forced not in nodes and math.isfinite(dist[forced]):
                target = forced
            elif total >= quota:
                break
            else:
                gains = {v: 0.0 for v in nodes}
                target = -1
                best_ratio = math.inf
                remaining = quota - total
                for v in range(1, size):
                    if v in nodes or energy[v] <= 0 or not math.isfinite(dist[v]):
                        continue
                    chain = []
                    u = v
                    while u not in gains and u >= 0:
                        chain.append(u)
                        u = pred[u]
                    gain = gains.get(u, 0.0)
                    for u in reversed(chain):
                        gain += energy[u]
                        gains[u] = gain
                    useful = min(remaining, gains[v])
                    if useful <= 0:
                        continue
                    ratio = dist[v] * target_noise[v] / useful ** exponent
                    if ratio < best_ratio:
                        best_ratio, target = ratio, v
                if target < 0:
                    return None
            forced = None
            u = target
            while u not in nodes:
                if prev_edge[u] < 0:
                    return None
                tree.add(prev_edge[u])
                nodes.add(u)
                p = pred[u]
                dist[u] = 0.0
                pred[u] = -1
                heapq.heappush(heap, (0.0, u))
                u = p
            total = math.fsum(energy[v] for v in nodes)
        return nodes, tree

    def destroy(nodes, tree):
        nodes, tree = set(nodes), set(tree)
        parent, pe, order, children = rooted(nodes, tree)
        fraction = rng.uniform(0.04, min(0.85, 0.16 + 0.16 * strength))
        target_loss = fraction * math.fsum(energy[v] for v in nodes)
        removed_energy = 0.0
        candidates = order[1:]
        rng.shuffle(candidates)
        while candidates and (removed_energy < target_loss or len(nodes) == len(order)):
            v = candidates.pop()
            if v not in nodes:
                continue
            todo = [v]
            for u in todo:
                todo.extend(c for c in children[u] if c in nodes)
            for u in todo:
                nodes.remove(u)
                tree.discard(pe[u])
                removed_energy += energy[u]
        return nodes, tree

    # Single-commodity signed flow guarantees connectivity, and the edge-count equality excludes cycles.
    mip_enabled = mip is not None
    globally_solved = False
    first_model = True

    def exact_neighborhood(duration, full):
        nonlocal mip_enabled, globally_solved
        local_stop = min(stop, time.perf_counter() + duration)
        incumbent = best
        current_nodes, current_tree = incumbent[1], incumbent[2]
        if full:
            allowed = set(range(size))
            fixed = {0}
        else:
            positive = [v for v in range(1, size) if energy[v] > 0]
            anchor = rng.choice(positive) if positive else 0
            distances = [math.inf] * size
            distances[anchor] = 0.0
            heap = [(0.0, anchor)]
            while heap and time.perf_counter() < local_stop:
                d, u = heapq.heappop(heap)
                if d != distances[u]:
                    continue
                for v, e in adj[u]:
                    nd = d + max(0.0, edges[e][2]) + max(0.0, cost[v]) * 0.5
                    if nd < distances[v]:
                        distances[v] = nd
                        heapq.heappush(heap, (nd, v))
            fraction = rng.uniform(0.12, min(0.95, 0.25 + 0.2 * strength))
            count = max(1, math.ceil(size * fraction))
            free = set(sorted(range(size), key=lambda v: distances[v])[:count])
            allowed = current_nodes | free
            fixed = (current_nodes - free) | {0}
        eligible = [e for e, item in enumerate(edges) if item[0] in allowed and item[1] in allowed]
        if time.perf_counter() >= local_stop or len(allowed) < 2:
            return
        model = None
        try:
            model = mip.Model(solver_name=mip.CBC)
            model.verbose = 0
            model.threads = 1
            model.seed = rng.randrange(1, 2147483647)
            model.max_mip_gap = 0.0
            model.max_mip_gap_abs = 1e-9
            positive_costs = [cost[v] for v in allowed if cost[v] > 0]
            positive_costs.extend(edges[e][2] for e in eligible if edges[e][2] > 0)
            scale = math.fsum(positive_costs) / max(1, len(positive_costs))
            scale = max(scale, 1e-9)
            x = {v: model.add_var(var_type=mip.BINARY, lb=1.0 if v in fixed else 0.0,
                                  obj=cost[v] / scale) for v in allowed}
            y = {e: model.add_var(var_type=mip.BINARY, obj=edges[e][2] / scale) for e in eligible}
            upper = max(1, len(allowed) - 1)
            f = {e: model.add_var(lb=-upper, ub=upper, var_type=mip.CONTINUOUS) for e in eligible}
            flow = {v: [] for v in allowed}
            incident = {v: [] for v in allowed}
            for j, e in enumerate(eligible):
                u, v = edges[e][:2]
                model += y[e] <= x[u]
                model += y[e] <= x[v]
                model += f[e] <= upper * y[e]
                model += f[e] >= -upper * y[e]
                flow[u].append(-f[e])
                flow[v].append(f[e])
                incident[u].append(y[e])
                incident[v].append(y[e])
                if j % 128 == 0 and time.perf_counter() >= local_stop:
                    return
            for v in allowed:
                if v:
                    model += mip.xsum(flow[v]) == x[v]
                    degree = 2 if nonnegative and energy[v] == 0 else 1
                    model += mip.xsum(incident[v]) >= degree * x[v]
            model += mip.xsum(y.values()) == mip.xsum(x.values()) - 1
            e_scale = max(1e-9, math.fsum(energy[v] for v in allowed) / max(1, len(allowed)))
            model += mip.xsum((energy[v] / e_scale) * x[v] for v in allowed if energy[v]) >= quota / e_scale
            parent, pe, order, _ = rooted(current_nodes, current_tree)
            subtree = {v: 1 for v in current_nodes}
            flow_start = {}
            for v in reversed(order[1:]):
                e = pe[v]
                flow_start[e] = subtree[v] if edges[e][0] == parent[v] else -subtree[v]
                subtree[parent[v]] += subtree[v]
            model.start = ([(x[v], int(v in current_nodes)) for v in allowed]
                           + [(y[e], int(e in current_tree)) for e in eligible]
                           + [(f[e], flow_start.get(e, 0)) for e in eligible])
            model.cutoff = incumbent[0] / scale + 1e-7 * max(1.0, abs(incumbent[0] / scale))
            remaining = local_stop - time.perf_counter()
            if remaining <= max(0.005, reserve * 0.15):
                return
            status = model.optimize(max_seconds=remaining * 0.88)
            if model.num_solutions:
                nodes = {v for v in allowed if x[v].x is not None and x[v].x > 0.5}
                candidate = mst(nodes)
                if candidate is not None and math.fsum(energy[v] for v in nodes) < quota:
                    candidate = grow(candidate[0], candidate[1], 0.0, 1.0, set())
                remember(prune(candidate, 1.0))
                if full and status == mip.OptimizationStatus.OPTIMAL:
                    globally_solved = True
        except Exception:
            mip_enabled = False
        finally:
            if model is not None:
                model.clear()

    next_mip = started + max(0.0, budget) * 0.16
    deterministic = True
    last_improvement = time.perf_counter()
    while time.perf_counter() < stop:
        now = time.perf_counter()
        if mip_enabled and not globally_solved and now >= next_mip:
            duration = min((stop - now) * 0.48, max(0.0, budget) * (0.23 if first_model else 0.16))
            full = first_model or rng.random() < 1.0 / (2.0 + strength)
            before = best[0]
            exact_neighborhood(duration, full)
            first_model = False
            if best[0] < before:
                last_improvement = time.perf_counter()
            next_mip = time.perf_counter() + max(0.0, budget) * 0.075
            continue
        base = pool[0] if rng.random() < 0.65 else rng.choice(pool)
        stale = (now - last_improvement) / max(1e-9, budget)
        mode = 0.0 if deterministic else rng.random()
        if mode < min(0.5, 0.2 + stale):
            seed_nodes, seed_tree = {0}, set()
            forced = None
        elif mode < 0.78:
            seed_nodes, seed_tree = destroy(base[1], base[2])
            forced = None
        else:
            seed_nodes, seed_tree = set(base[1]), set(base[2])
            options = [v for v in range(1, size) if v not in seed_nodes and energy[v] > 0]
            forced = rng.choice(options) if options else None
        noisy = 0.0 if deterministic else rng.uniform(0.3, 1.0 + min(1.0, stale * strength))
        exponent = 1.0 if deterministic else rng.uniform(max(0.4, 1 - 0.2 * strength), 1.0 + 0.3 * strength)
        candidate = grow(seed_nodes, seed_tree, noisy, exponent, base[2], forced)
        if candidate is not None:
            candidate = mst(candidate[0])
            candidate = prune(candidate, 1.0 if deterministic else rng.uniform(0.0, 1.0 + 0.25 * strength))
            if remember(candidate):
                last_improvement = time.perf_counter()
        deterministic = False
