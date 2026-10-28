"""Adaptive branch recombination with sparse quota tree search
Secures a feasible tree as fallback and combines preorder subtree-skipping dynamic programming, reconnection over the union of elite branches and wall-clock-driven perturbation search.
"""

def solve(instance: dict, time_limit_s: float) -> dict:
    import time
    import math
    import random
    import heapq
    from collections import deque

    clock = time.perf_counter
    t0 = clock()
    started = t0
    answer = {"node_selected": [], "edge_selected": []}

    try:
        try:
            budget = float(time_limit_s)
        except BaseException:
            budget = 0.0
        if not math.isfinite(budget) or budget < 0.0:
            budget = 0.0

        deadline = t0 + 0.93 * budget
        search_deadline = deadline

        try:
            types = list(instance.get("node_types", []))
        except BaseException:
            types = []
        try:
            raw_edges = list(instance.get("edges", []))
        except BaseException:
            raw_edges = []

        n = len(types)
        m = len(raw_edges)
        answer = {
            "node_selected": [0] * n,
            "edge_selected": [0] * m,
        }
        if n == 0:
            return answer

        def read_array(name, count):
            try:
                raw = list(instance.get(name, []))
            except BaseException:
                raw = []
            result = [0.0] * count
            for i in range(min(count, len(raw))):
                try:
                    value = float(raw[i])
                    if math.isfinite(value):
                        result[i] = max(0.0, value)
                except BaseException:
                    pass
            return result

        installation = read_array("installation_costs", n)
        yields = read_array("annual_energy_yields", n)
        node_scenic = read_array("node_scenic_impacts", n)
        cable = read_array("edge_cable_costs", m)
        edge_scenic = read_array("edge_scenic_impacts", m)

        try:
            alpha = float(instance.get("alpha", 0.5))
        except BaseException:
            alpha = 0.5
        if not math.isfinite(alpha):
            alpha = 0.5
        alpha = min(1.0, max(0.0, alpha))

        try:
            quota = float(instance.get("minimum_energy_quota", 0.0))
        except BaseException:
            quota = 0.0
        if not math.isfinite(quota):
            quota = 0.0
        quota = max(0.0, quota)

        edges = [(-1, -1)] * m
        adjacency = [[] for _ in range(n)]
        valid_edges = []
        for e, pair in enumerate(raw_edges):
            try:
                u = int(pair[0])
                v = int(pair[1])
            except BaseException:
                continue
            if 0 <= u < n and 0 <= v < n and u != v:
                edges[e] = (u, v)
                adjacency[u].append((v, e))
                adjacency[v].append((u, e))
                valid_edges.append(e)

        substations = [i for i, kind in enumerate(types) if kind == "substation"]
        turbines = [i for i, kind in enumerate(types) if kind == "turbine"]
        is_turbine = [kind == "turbine" for kind in types]
        root = substations[0] if substations else 0
        if not substations:
            substations = [root]
        sub_set = set(substations)

        node_weight = [0.0] * n
        for v in turbines:
            node_weight[v] = (
                alpha * installation[v]
                + (1.0 - alpha) * node_scenic[v]
            )
        edge_weight = [
            alpha * cable[e] + (1.0 - alpha) * edge_scenic[e]
            for e in range(m)
        ]

        def energy_of(nodes):
            return math.fsum(yields[v] for v in nodes if is_turbine[v])

        def objective(nodes, selected_edges):
            return (
                math.fsum(node_weight[v] for v in nodes)
                + math.fsum(edge_weight[e] for e in selected_edges)
            )

        def make_output(nodes, selected_edges):
            node_out = [0] * n
            edge_out = [0] * m
            for v in nodes:
                if 0 <= v < n:
                    node_out[v] = 1
            for e in selected_edges:
                if 0 <= e < m:
                    edge_out[e] = 1
            return {
                "node_selected": node_out,
                "edge_selected": edge_out,
            }

        reached = {root}
        fallback_edges = set()
        queue = deque([root])
        while queue:
            u = queue.popleft()
            for v, e in adjacency[u]:
                if v not in reached:
                    reached.add(v)
                    fallback_edges.add(e)
                    queue.append(v)
        fallback_nodes = set(reached)
        answer = make_output(fallback_nodes, fallback_edges)

        def is_tree(nodes, selected_edges):
            if not nodes or len(selected_edges) != len(nodes) - 1:
                return False
            parent = list(range(n))
            size = [1] * n

            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            for e in selected_edges:
                if not (0 <= e < m):
                    return False
                u, v = edges[e]
                if u < 0 or u not in nodes or v not in nodes:
                    return False
                ru = find(u)
                rv = find(v)
                if ru == rv:
                    return False
                if size[ru] < size[rv]:
                    ru, rv = rv, ru
                parent[rv] = ru
                size[ru] += size[rv]
            representative = find(next(iter(nodes)))
            return all(find(v) == representative for v in nodes)

        def feasible(nodes, selected_edges):
            return (
                sub_set.issubset(nodes)
                and energy_of(nodes) >= quota
                and is_tree(nodes, selected_edges)
            )

        def prune_tree(nodes, selected_edges, prune_rng=None):
            nodes = set(nodes)
            active = set(selected_edges)
            incident = [[] for _ in range(n)]
            degree = [0] * n
            for e in active:
                u, v = edges[e]
                if u in nodes and v in nodes:
                    incident[u].append(e)
                    incident[v].append(e)
                    degree[u] += 1
                    degree[v] += 1

            energy = energy_of(nodes)
            safety = 1e-13 * max(1.0, quota)
            heap = []
            serial = 0

            def push(v):
                nonlocal serial
                if v not in nodes or v in sub_set or degree[v] > 1:
                    return
                leaf_edge = -1
                if degree[v] == 1:
                    for e in incident[v]:
                        if e in active:
                            leaf_edge = e
                            break
                edge_cost = edge_weight[leaf_edge] if leaf_edge >= 0 else 0.0
                serial += 1
                if not is_turbine[v]:
                    key = (0, -edge_cost, serial, v)
                else:
                    saving = node_weight[v] + edge_cost
                    badness = saving / max(yields[v], 1e-15)
                    if prune_rng is not None:
                        badness *= math.exp(0.32 * (2.0 * prune_rng.random() - 1.0))
                    key = (1, -badness, serial, v)
                heapq.heappush(heap, key)

            for v in list(nodes):
                push(v)

            while heap:
                _, _, _, v = heapq.heappop(heap)
                if v not in nodes or v in sub_set or degree[v] > 1:
                    continue
                loss = yields[v] if is_turbine[v] else 0.0
                new_energy = energy - loss
                if is_turbine[v] and new_energy < quota + safety:
                    continue
                if degree[v] == 0:
                    nodes.remove(v)
                    energy = new_energy
                    continue

                chosen = -1
                other = -1
                for e in incident[v]:
                    if e in active:
                        chosen = e
                        u, w = edges[e]
                        other = w if u == v else u
                        break
                if chosen < 0:
                    continue
                active.remove(chosen)
                nodes.remove(v)
                degree[v] = 0
                degree[other] -= 1
                energy = new_energy
                push(other)
            return nodes, active

        best_nodes = set(fallback_nodes)
        best_edges = set(fallback_edges)
        best_valid = feasible(best_nodes, best_edges)
        best_value = objective(best_nodes, best_edges) if best_valid else float("inf")

        if best_valid:
            pn, pe = prune_tree(best_nodes, best_edges)
            if feasible(pn, pe):
                pv = objective(pn, pe)
                if pv < best_value:
                    best_nodes, best_edges, best_value = pn, pe, pv
                    answer = make_output(best_nodes, best_edges)

        base_nodes = set(substations)
        base_edges = set()
        base_ok = True
        for s in substations[1:]:
            chosen = -1
            chosen_cost = float("inf")
            for v, e in adjacency[root]:
                if v == s and edge_weight[e] < chosen_cost:
                    chosen = e
                    chosen_cost = edge_weight[e]
            if chosen < 0:
                base_ok = False
                break
            base_edges.add(chosen)

        edge_order = None
        remaining = search_deadline - clock()
        sort_estimate = 3e-7 * max(1, len(valid_edges)) * math.log2(len(valid_edges) + 2.0)
        if remaining > max(0.01, 2.0 * sort_estimate):
            edge_order = sorted(valid_edges, key=lambda e: (edge_weight[e], e))

        def induced_mst(nodes, cutoff):
            if edge_order is None:
                return None
            nodes = set(nodes)
            if len(nodes) <= 1:
                return set()
            parent = list(range(n))
            size = [1] * n

            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            result = set()
            need = len(nodes) - 1
            for pos, e in enumerate(edge_order):
                if (pos & 2047) == 0 and clock() >= cutoff:
                    return None
                u, v = edges[e]
                if u not in nodes or v not in nodes:
                    continue
                ru = find(u)
                rv = find(v)
                if ru == rv:
                    continue
                if size[ru] < size[rv]:
                    ru, rv = rv, ru
                parent[rv] = ru
                size[ru] += size[rv]
                result.add(e)
                if len(result) == need:
                    return result
            return None

        infinity = float("inf")

        def shortest_forest(sources, search_edge, search_node, existing, cutoff):
            dist = [infinity] * n
            actual = [infinity] * n
            gain = [0.0] * n
            parent_node = [-1] * n
            parent_edge = [-1] * n
            heap = []
            for s in sources:
                dist[s] = 0.0
                actual[s] = 0.0
                heapq.heappush(heap, (0.0, s))

            settled = [False] * n
            pops = 0
            while heap:
                value, u = heapq.heappop(heap)
                if settled[u] or value != dist[u]:
                    continue
                settled[u] = True
                pops += 1
                if (pops & 255) == 0 and clock() >= cutoff:
                    return None
                for v, e in adjacency[u]:
                    if settled[v]:
                        continue
                    node_part = 0.0 if v in existing else search_node[v]
                    nd = value + search_edge[e] + node_part
                    if nd < dist[v]:
                        dist[v] = nd
                        parent_node[v] = u
                        parent_edge[v] = e
                        actual[v] = actual[u] + edge_weight[e]
                        gain[v] = gain[u]
                        if v not in existing:
                            actual[v] += node_weight[v]
                            if is_turbine[v]:
                                gain[v] += yields[v]
                        heapq.heappush(heap, (nd, v))
            return dist, actual, gain, parent_node, parent_edge

        def full_tree_from_forest(forest):
            if forest is None:
                return None
            parent_edge = forest[4]
            nodes = set(base_nodes)
            selected_edges = set(base_edges)
            for v in range(n):
                e = parent_edge[v]
                if e >= 0:
                    nodes.add(v)
                    selected_edges.add(e)
            if is_tree(nodes, selected_edges):
                return nodes, selected_edges
            return None

        def attach_path(target, nodes, selected_edges, parent_node, parent_edge):
            chain = []
            current = target
            steps = 0
            while current not in nodes:
                e = parent_edge[current]
                previous = parent_node[current]
                if e < 0 or previous < 0:
                    return None
                chain.append((current, e))
                current = previous
                steps += 1
                if steps > n:
                    return None
            added_energy = 0.0
            for v, e in reversed(chain):
                if v not in nodes:
                    nodes.add(v)
                    selected_edges.add(e)
                    if is_turbine[v]:
                        added_energy += yields[v]
            return added_energy

        def static_construct(forest, beta, noise, seed_targets, cutoff, rng):
            dist, _, path_gain, parent_node, parent_edge = forest
            nodes = set(base_nodes)
            selected_edges = set(base_edges)

            if seed_targets:
                seeds = [t for t in seed_targets if math.isfinite(dist[t])]
                seeds.sort(key=lambda t: dist[t])
                for pos, target in enumerate(seeds):
                    if (pos & 63) == 0 and clock() >= cutoff:
                        return None
                    if target not in nodes:
                        if attach_path(target, nodes, selected_edges, parent_node, parent_edge) is None:
                            return None

            current_energy = energy_of(nodes)
            candidates = []
            remaining_quota = max(1e-15, quota - current_energy)
            for pos, t in enumerate(turbines):
                if (pos & 255) == 0 and clock() >= cutoff:
                    return None
                if t in nodes or path_gain[t] <= 0.0 or not math.isfinite(dist[t]):
                    continue
                useful = max(1e-15, min(path_gain[t], remaining_quota))
                score = dist[t] / (useful ** beta)
                if noise > 0.0:
                    score *= math.exp(noise * (2.0 * rng.random() - 1.0))
                candidates.append((score, t))
            candidates.sort()

            for pos, (_, target) in enumerate(candidates):
                if (pos & 63) == 0 and clock() >= cutoff:
                    return None
                if current_energy >= quota:
                    break
                if target in nodes:
                    continue
                added = attach_path(target, nodes, selected_edges, parent_node, parent_edge)
                if added is not None:
                    current_energy += added
            if energy_of(nodes) < quota:
                return None
            return nodes, selected_edges

        def dynamic_grow(start_nodes, start_edges, search_edge, search_node,
                         beta, rcl, batch_fraction, cutoff, rng):
            nodes = set(start_nodes)
            selected_edges = set(start_edges)
            current_energy = energy_of(nodes)

            while current_energy < quota and clock() < cutoff:
                forest = shortest_forest(nodes, search_edge, search_node, nodes, cutoff)
                if forest is None:
                    return None
                _, actual, path_gain, parent_node, parent_edge = forest
                remaining_quota = quota - current_energy
                candidates = []
                for pos, t in enumerate(turbines):
                    if (pos & 255) == 0 and clock() >= cutoff:
                        return None
                    if t in nodes or path_gain[t] <= 0.0 or not math.isfinite(actual[t]):
                        continue
                    useful = max(1e-15, min(path_gain[t], remaining_quota))
                    candidates.append((actual[t] / (useful ** beta), t))
                if not candidates:
                    return None

                pool_size = min(len(candidates), max(1, rcl * 3))
                pool = heapq.nsmallest(pool_size, candidates)
                choice_limit = min(max(1, rcl), len(pool))
                chosen_index = min(
                    choice_limit - 1,
                    int((rng.random() ** 2.2) * choice_limit),
                )
                ordered = [pool[chosen_index]] + pool[:chosen_index] + pool[chosen_index + 1:]
                target_energy = min(
                    quota,
                    current_energy + max(1e-15, remaining_quota * batch_fraction),
                )
                old_energy = current_energy
                for _, target in ordered:
                    if current_energy >= target_energy:
                        break
                    if target in nodes:
                        continue
                    added = attach_path(target, nodes, selected_edges, parent_node, parent_edge)
                    if added is not None:
                        current_energy += added
                current_energy = energy_of(nodes)
                if current_energy <= old_energy:
                    return None

            if current_energy < quota:
                return None
            return nodes, selected_edges

        def enrich_tree(source_nodes, source_edges, search_edge, search_node,
                        width, cutoff):

            if clock() >= cutoff:
                return None
            forest = shortest_forest(
                source_nodes, search_edge, search_node, source_nodes, cutoff
            )
            if forest is None:
                return None
            _, actual, path_gain, parent_node, parent_edge = forest
            ranked = []
            need = max(1e-15, quota - energy_of(source_nodes))
            for t in turbines:
                if clock() >= cutoff:
                    return None
                if t in source_nodes or path_gain[t] <= 0.0:
                    continue
                useful = min(path_gain[t], max(need, yields[t]))
                ranked.append((actual[t] / max(useful, 1e-15), actual[t], t))
            ranked.sort()
            nodes = set(source_nodes)
            selected_edges = set(source_edges)
            added = 0

            for _, _, target in ranked:
                if clock() >= cutoff:
                    return None
                before = len(nodes)
                if attach_path(
                    target, nodes, selected_edges, parent_node, parent_edge
                ) is not None and len(nodes) > before:
                    added += 1
                    if added >= width:
                        break
            if added and is_tree(nodes, selected_edges):
                return nodes, selected_edges
            return None

        def cut_branch_and_repair(source, search_edge, search_node,
                                  strength, cutoff, rng):

            if source is None or clock() >= cutoff:
                return None
            source_nodes, source_edges = source[1], source[2]
            local = [[] for _ in range(n)]
            for e in source_edges:
                u, v = edges[e]
                local[u].append((v, e))
                local[v].append((u, e))

            parent = [-2] * n
            parent_edge = [-1] * n
            children = [[] for _ in range(n)]
            parent[root] = -1
            order = [root]
            for u in order:
                if clock() >= cutoff:
                    return None
                for v, e in local[u]:
                    if parent[v] == -2:
                        parent[v] = u
                        parent_edge[v] = e
                        children[u].append(v)
                        order.append(v)
            if len(order) != len(source_nodes):
                return None

            subtree_energy = [0.0] * n
            subtree_cost = [0.0] * n
            protected = [False] * n
            subtree_size = [0] * n
            for v in reversed(order):
                energy = yields[v] if is_turbine[v] else 0.0
                cost = node_weight[v]
                protected_here = v in sub_set
                size_here = 1
                for child in children[v]:
                    energy += subtree_energy[child]
                    cost += subtree_cost[child] + edge_weight[parent_edge[child]]
                    protected_here = protected_here or protected[child]
                    size_here += subtree_size[child]
                subtree_energy[v] = energy
                subtree_cost[v] = cost
                protected[v] = protected_here
                subtree_size[v] = size_here

            candidates = []
            total_energy = energy_of(source_nodes)
            for v in order[1:]:
                if clock() >= cutoff:
                    return None
                lost = subtree_energy[v]
                if protected[v] or lost <= 0.0:
                    continue

                ratio = (
                    subtree_cost[v] + edge_weight[parent_edge[v]]
                ) / max(lost, 1e-15)
                scale = math.exp(strength * (2.0 * rng.random() - 1.0))
                balance = 1.0 + 0.20 * min(3.0, lost / max(quota, 1e-15))
                candidates.append((ratio * scale * balance, v))
            if not candidates:
                return None
            candidates.sort(reverse=True)
            pool = min(len(candidates), max(1, int(2 + 5 * strength)))
            cut = candidates[min(pool - 1, int((rng.random() ** 2) * pool))][1]

            removed = set()
            stack = [cut]
            while stack:
                if clock() >= cutoff:
                    return None
                v = stack.pop()
                removed.add(v)
                stack.extend(children[v])
            nodes = set(source_nodes) - removed
            selected_edges = {
                e for e in source_edges
                if edges[e][0] not in removed and edges[e][1] not in removed
            }
            if not is_tree(nodes, selected_edges):
                return None
            if total_energy - subtree_energy[cut] >= quota:
                return nodes, selected_edges
            return dynamic_grow(
                nodes, selected_edges, search_edge, search_node,
                0.72 + 0.30 * rng.random(),
                max(1, int(1 + 4 * strength)),
                0.08 + 0.12 * rng.random(), cutoff, rng,
            )

        def prim_full(search_edge, search_node, cutoff):
            if not base_ok:
                return None
            nodes = set(base_nodes)
            selected_edges = set(base_edges)
            heap = []
            for u in base_nodes:
                for v, e in adjacency[u]:
                    if v not in nodes:
                        key = search_edge[e] + 0.25 * search_node[v]
                        heapq.heappush(heap, (key, e, u, v))
            pops = 0
            while heap:
                key, e, u, v = heapq.heappop(heap)
                if v in nodes:
                    continue
                nodes.add(v)
                selected_edges.add(e)
                pops += 1
                if (pops & 255) == 0 and clock() >= cutoff:
                    return None
                for w, f in adjacency[v]:
                    if w not in nodes:
                        new_key = search_edge[f] + 0.25 * search_node[w]
                        heapq.heappush(heap, (new_key, f, v, w))
            if is_tree(nodes, selected_edges):
                return nodes, selected_edges
            return None

        def metric_reconnect(terminals, search_edge, search_node, cutoff):
            terminals = set(terminals)
            terminals.update(substations)
            terminals = {v for v in terminals if 0 <= v < n}
            if not terminals:
                return None
            if terminals == base_nodes:
                return set(base_nodes), set(base_edges)
            if len(terminals) == 1:
                return set(terminals), set()
            if clock() >= cutoff:
                return None

            dist = [infinity] * n
            owner = [-1] * n
            parent_node = [-1] * n
            parent_edge = [-1] * n
            heap = []
            for t in terminals:
                dist[t] = 0.0
                owner[t] = t
                heapq.heappush(heap, (0.0, t))

            settled = [False] * n
            pops = 0
            while heap:
                value, u = heapq.heappop(heap)
                if settled[u] or value != dist[u]:
                    continue
                settled[u] = True
                pops += 1
                if (pops & 255) == 0 and clock() >= cutoff:
                    return None
                u_part = 0.0 if u in terminals else search_node[u]
                for v, e in adjacency[u]:
                    if settled[v]:
                        continue
                    v_part = 0.0 if v in terminals else search_node[v]
                    nd = value + search_edge[e] + 0.5 * (u_part + v_part)
                    if nd < dist[v]:
                        dist[v] = nd
                        owner[v] = owner[u]
                        parent_node[v] = u
                        parent_edge[v] = e
                        heapq.heappush(heap, (nd, v))

            boundary = []
            for pos, e in enumerate(valid_edges):
                if (pos & 2047) == 0 and clock() >= cutoff:
                    return None
                u, v = edges[e]
                if owner[u] < 0 or owner[v] < 0 or owner[u] == owner[v]:
                    continue
                u_part = 0.0 if u in terminals else search_node[u]
                v_part = 0.0 if v in terminals else search_node[v]
                step = search_edge[e] + 0.5 * (u_part + v_part)
                boundary.append((dist[u] + step + dist[v], e, u, v))
            boundary.sort()

            uf = {t: t for t in terminals}
            uf_size = {t: 1 for t in terminals}

            def find_terminal(x):
                while uf[x] != x:
                    uf[x] = uf[uf[x]]
                    x = uf[x]
                return x

            chosen = set()
            merges = 0
            for pos, (_, e, u, v) in enumerate(boundary):
                if (pos & 1023) == 0 and clock() >= cutoff:
                    return None
                a = find_terminal(owner[u])
                b = find_terminal(owner[v])
                if a == b:
                    continue
                if uf_size[a] < uf_size[b]:
                    a, b = b, a
                uf[b] = a
                uf_size[a] += uf_size[b]
                chosen.add(e)

                for start_node in (u, v):
                    x = start_node
                    steps = 0
                    while parent_edge[x] >= 0:
                        chosen.add(parent_edge[x])
                        x = parent_node[x]
                        steps += 1
                        if steps > n:
                            return None
                merges += 1
                if merges == len(terminals) - 1:
                    break

            if merges != len(terminals) - 1:
                return None
            nodes = set(terminals)
            for e in chosen:
                u, v = edges[e]
                nodes.add(u)
                nodes.add(v)
            if is_tree(nodes, chosen):
                return nodes, chosen
            mst = induced_mst(nodes, cutoff)
            if mst is not None and is_tree(nodes, mst):
                return nodes, mst
            return None

        def tree_quota_dp(nodes, selected_edges, cap, cutoff, cost_bound=None):
            if clock() >= cutoff or root not in nodes:
                return None
            local = [[] for _ in range(n)]
            for e in selected_edges:
                u, v = edges[e]
                local[u].append((v, e))
                local[v].append((u, e))
            parent = [-2] * n
            parent_edge = [-1] * n
            children = [[] for _ in range(n)]
            parent[root] = -1
            order = [root]
            for u in order:
                if clock() >= cutoff:
                    return None
                for v, e in local[u]:
                    if parent[v] == -2:
                        parent[v] = u
                        parent_edge[v] = e
                        children[u].append(v)
                        order.append(v)
            if len(order) != len(nodes):
                return None

            required = [False] * n
            sizes = [1] * n
            subtree_gain = [0.0] * n
            subtree_cost = [0.0] * n
            for v in reversed(order):
                required[v] = v in sub_set
                subtree_gain[v] = yields[v] if is_turbine[v] else 0.0
                subtree_cost[v] = node_weight[v]
                if v != root:
                    subtree_cost[v] += edge_weight[parent_edge[v]]
                for w in children[v]:
                    required[v] = required[v] or required[w]
                    sizes[v] += sizes[w]
                    subtree_gain[v] += subtree_gain[w]
                    subtree_cost[v] += subtree_cost[w]
                # Process branches with high energy per cost first; slight perturbations change the retention order of the approximate frontier.
                children[v].sort(key=lambda w: (
                    not required[w],
                    subtree_cost[w] / max(subtree_gain[w], 1e-30)
                    * (0.85 + 0.30 * rng.random()),
                ))
            preorder = []
            stack = [root]
            while stack:
                v = stack.pop()
                preorder.append(v)
                stack.extend(reversed(children[v]))
            length = len(preorder)
            suffix_gain = [0.0] * (length + 1)
            for i in range(length - 1, -1, -1):
                v = preorder[i]
                suffix_gain[i] = suffix_gain[i + 1] + (
                    yields[v] if is_turbine[v] else 0.0
                )

            cap = max(16, int(cap))
            tolerance = 1e-12 * max(1.0, quota)
            saturation = quota + tolerance if quota > 0.0 else 0.0
            ceiling = infinity if cost_bound is None else (
                cost_bound + 1e-10 * max(1.0, abs(cost_bound))
            )

            def compress(options, position):
                if not options:
                    return []
                remaining = suffix_gain[position]
                options = [x for x in options
                           if x[1] <= ceiling and x[0] + remaining >= quota - tolerance]
                options.sort(key=lambda x: (-x[0], x[1]))
                kept = []
                cheapest = infinity
                for x in options:
                    if x[1] < cheapest:
                        kept.append(x)
                        cheapest = x[1]
                if len(kept) <= cap:
                    return kept
                # Keep both ends of each energy bin and evenly ranked points, rather than only combinations on the convex hull.
                slots = max(2, cap // 4)
                chosen = {0, len(kept) - 1}
                span = max(kept[0][0] - kept[-1][0], 1e-30)
                last_bin = -1
                for j, x in enumerate(kept):
                    cell = min(slots - 1, int((kept[0][0] - x[0]) / span * slots))
                    if cell != last_bin:
                        chosen.add(j)
                        if j:
                            chosen.add(j - 1)
                        last_bin = cell
                extra = max(2, cap - len(chosen))
                for j in range(extra):
                    chosen.add(j * (len(kept) - 1) // (extra - 1))
                return [kept[j] for j in sorted(chosen)]

            # Preorder positions are the vertices of the state graph; selecting the current node advances one step, skipping it jumps over the whole subtree.
            pending = [None] * (length + 1)
            current = [(0.0, 0.0, 0)]
            for i, v in enumerate(preorder):
                if clock() >= cutoff:
                    return None
                if pending[i] is not None:
                    current.extend(pending[i])
                    pending[i] = None
                current = compress(current, i)
                after = i + sizes[v]
                if not required[v] and after > i + 1 and current:
                    if pending[after] is None:
                        pending[after] = list(current)
                    else:
                        pending[after].extend(current)
                        if len(pending[after]) > 2 * cap:
                            pending[after] = compress(pending[after], after)
                own_gain = yields[v] if is_turbine[v] else 0.0
                own_cost = node_weight[v]
                if v != root:
                    own_cost += edge_weight[parent_edge[v]]
                bit = 1 << v
                advanced = [(min(saturation, x[0] + own_gain),
                             x[1] + own_cost, x[2] | bit) for x in current]
                if not required[v] and after == i + 1:
                    advanced.extend(current)
                current = advanced
            if pending[length] is not None:
                current.extend(pending[length])
            candidates = compress(current, length)
            candidates.sort(key=lambda x: x[1])
            for energy, value, mask in candidates:
                if clock() >= cutoff:
                    return None
                cn = set()
                while mask:
                    bit = mask & -mask
                    cn.add(bit.bit_length() - 1)
                    mask ^= bit
                ce = {parent_edge[v] for v in cn if v != root}
                if feasible(cn, ce):
                    return cn, ce
            return None

        positive_values = [edge_weight[e] for e in valid_edges if edge_weight[e] > 0.0]
        positive_values.extend(node_weight[v] for v in turbines if node_weight[v] > 0.0)
        tie_scale = math.fsum(positive_values) / len(positive_values) if positive_values else 1.0
        if not math.isfinite(tie_scale) or tie_scale <= 0.0:
            tie_scale = 1.0
        positive_yields = [yields[v] for v in turbines if yields[v] > 0.0]
        yield_scale = math.fsum(positive_yields) / len(positive_yields) if positive_yields else 1.0
        if not math.isfinite(yield_scale) or yield_scale <= 0.0:
            yield_scale = 1.0

        seed = (
            (n + 1) * 0x9E3779B97F4A7C15
            ^ (m + 3) * 0xBF58476D1CE4E5B9
            ^ int(quota * 1000003.0)
            ^ int(alpha * 1000000007.0)
        ) & ((1 << 64) - 1)
        for i in range(min(n, 48)):
            seed ^= (
                int(yields[i] * 1000033.0)
                + int(node_weight[i] * 10007.0)
                + (i + 1) * 0x94D049BB133111EB
            ) & ((1 << 64) - 1)
        rng = random.Random(seed)

        elites = []
        last_improvement = started

        def normalize(nodes, selected_edges, cutoff, deep, dp_cap, prune_rng=None):
            nodes = set(nodes)
            selected_edges = set(selected_edges)
            if not is_tree(nodes, selected_edges):
                replacement = induced_mst(nodes, cutoff)
                if replacement is None or not is_tree(nodes, replacement):
                    return None
                selected_edges = replacement
            if energy_of(nodes) < quota or not sub_set.issubset(nodes):
                return None

            local_best = None
            local_value = float("inf")

            def retain(candidate_nodes, candidate_edges):
                nonlocal local_best, local_value
                if not feasible(candidate_nodes, candidate_edges):
                    return
                value = objective(candidate_nodes, candidate_edges)
                if value < local_value:
                    local_value = value
                    local_best = (set(candidate_nodes), set(candidate_edges))

            pn, pe = prune_tree(nodes, selected_edges, prune_rng)
            retain(pn, pe)

            if deep and clock() < cutoff:
                slice_end = min(cutoff, clock() + move_slice)
                dp_candidate = tree_quota_dp(
                    nodes, selected_edges, dp_cap, slice_end, local_value
                )
                if dp_candidate is not None:
                    dn, de = prune_tree(dp_candidate[0], dp_candidate[1])
                    retain(dn, de)
                    if edge_order is not None and clock() < cutoff:
                        rewired = induced_mst(dn, cutoff)
                        if rewired is not None:
                            rn, re = prune_tree(dn, rewired)
                            retain(rn, re)

            if deep and local_best is not None and clock() < cutoff:
                rewired = induced_mst(local_best[0], cutoff)
                if rewired is not None:
                    rn, re = prune_tree(local_best[0], rewired)
                    retain(rn, re)
            if (deep and local_best is not None and clock() < cutoff
                    and (local_value < best_value or rng.random() < 0.16)):
                terminals = sub_set | {v for v in local_best[0] if is_turbine[v]}
                reconnected = metric_reconnect(
                    terminals, edge_weight, node_weight,
                    min(cutoff, clock() + move_slice),
                )
                if reconnected is not None:
                    rn, re = prune_tree(*reconnected)
                    retain(rn, re)
            return local_best

        def consider(nodes, selected_edges, cutoff, deep, dp_cap, elite_cap, prune_rng=None):
            nonlocal best_nodes, best_edges, best_value, best_valid
            nonlocal answer, elites, last_improvement
            candidate = normalize(
                nodes, selected_edges, cutoff, deep, dp_cap, prune_rng
            )
            if candidate is None:
                return False
            cn, ce = candidate
            value = objective(cn, ce)
            tolerance = 1e-12 * max(1.0, abs(best_value)) if best_valid else 0.0
            improved = not best_valid or value < best_value - tolerance
            if improved:
                best_nodes = set(cn)
                best_edges = set(ce)
                best_value = value
                best_valid = True
                answer = make_output(best_nodes, best_edges)
                last_improvement = clock()

            signature = frozenset(v for v in cn if is_turbine[v])
            replacement = -1
            for i, record in enumerate(elites):
                if record[3] == signature:
                    replacement = i
                    break
            record = (value, set(cn), set(ce), signature)
            if replacement >= 0:
                if value < elites[replacement][0]:
                    elites[replacement] = record
            else:
                elites.append(record)
            elites.sort(key=lambda item: item[0])
            if len(elites) > elite_cap:
                del elites[elite_cap:]
            return improved

        def pick_elite():
            if not elites:
                return None
            index = min(len(elites) - 1, int((rng.random() ** 2.4) * len(elites)))
            return elites[index]

        def terminal_swap(source, search_edge, search_node, explore, cutoff):

            if source is None or clock() >= cutoff:
                return None
            chosen_turbines = list(source[3])
            outside = [t for t in turbines if t not in source[3]]
            if not chosen_turbines or not outside:
                return None
            forest = shortest_forest(
                source[1], search_edge, search_node, source[1], cutoff
            )
            if forest is None:
                return None
            _, actual, path_gain, _, _ = forest

            degree = [0] * n
            leaf_edge = [-1] * n
            for e in source[2]:
                u, v = edges[e]
                degree[u] += 1
                degree[v] += 1
                leaf_edge[u] = e
                leaf_edge[v] = e
            total_energy = energy_of(source[1])
            pairs = []
            for old in chosen_turbines:
                if clock() >= cutoff:
                    return None
                saving = node_weight[old]
                if degree[old] == 1 and leaf_edge[old] >= 0:
                    saving += edge_weight[leaf_edge[old]]
                required = quota - (total_energy - yields[old])
                for new in outside:
                    if yields[new] + 1e-12 < required:
                        continue
                    if not math.isfinite(actual[new]) or path_gain[new] <= 0.0:
                        continue
                    delta = actual[new] - saving
                    if explore > 0.0:
                        delta += tie_scale * explore * rng.random()
                    pairs.append((delta, old, new))
            if not pairs:
                return None
            pairs.sort()
            pool = min(len(pairs), max(1, int(1 + 10 * explore)))
            _, old, new = pairs[
                min(pool - 1, int((rng.random() ** 2.5) * pool))
            ]
            terminals = set(source[3])
            terminals.discard(old)
            terminals.add(new)
            return metric_reconnect(
                set(substations) | terminals,
                search_edge, search_node, cutoff,
            )

        def recombine_envelope(first, second, search_edge, search_node, width, cutoff):
            terminals = sub_set | set(first[3]) | set(second[3])
            candidate = metric_reconnect(
                terminals, search_edge, search_node, cutoff
            )
            if candidate is None:
                candidate = (set(first[1]), set(first[2]))
            expanded = enrich_tree(
                candidate[0], candidate[1], search_edge, search_node,
                width, cutoff,
            )
            if expanded is not None:
                candidate = expanded
            # Rewire on the complete candidate network first, then filter by quota, avoiding optimising only an already pruned tree.
            if clock() < cutoff and rng.random() < 0.5:
                rewired = induced_mst(candidate[0], cutoff)
                if rewired is not None:
                    candidate = (candidate[0], rewired)
            return candidate

        def make_weights(amplitude, guide, prize_pressure, cutoff):
            search_edge = list(edge_weight)
            guide_nodes = guide[1] if guide is not None else None
            guide_edges = guide[2] if guide is not None else None
            epsilon = tie_scale * 1e-12

            for pos, e in enumerate(valid_edges):
                if (pos & 2047) == 0 and clock() >= cutoff:
                    return None
                value = edge_weight[e] * math.exp(
                    amplitude * (2.0 * rng.random() - 1.0)
                )
                if edge_weight[e] == 0.0:
                    value += epsilon * (0.1 + amplitude) * rng.random()
                if guide_edges is not None and e in guide_edges:
                    value *= 0.82 + 0.10 * rng.random()
                search_edge[e] = value

            search_node = list(node_weight)
            for pos, v in enumerate(range(n)):
                if (pos & 2047) == 0 and clock() >= cutoff:
                    return None
                if v in sub_set:
                    search_node[v] = 0.0
                    continue
                value = node_weight[v] * math.exp(
                    amplitude * (2.0 * rng.random() - 1.0)
                )
                if is_turbine[v] and prize_pressure > 0.0:
                    value /= 1.0 + prize_pressure * yields[v] / yield_scale
                if value == 0.0:
                    value = epsilon * (0.1 + amplitude) * rng.random()
                if guide_nodes is not None and v in guide_nodes:
                    value *= 0.82 + 0.10 * rng.random()
                search_node[v] = value
            return search_edge, search_node

        effort = math.log1p(budget)
        scale = 1.0 + (n + m) / max(1.0, 8000.0 * (1.0 + effort))
        initial_cap = max(32, int((64.0 + 150.0 * effort) / math.sqrt(scale)))
        initial_elites = max(3, int(3 + 1.6 * effort))
        branch_width = max(1, int((1.0 + effort) * math.sqrt(max(1, len(turbines)))))
        move_slice = min(search_deadline - clock(), budget / (12.0 + 4.0 * effort))
        if best_valid and clock() < search_deadline:
            consider(
                best_nodes, best_edges, search_deadline,
                True, initial_cap, initial_elites
            )

        actual_forest = None
        if base_ok and clock() < search_deadline:
            actual_forest = shortest_forest(
                base_nodes, edge_weight, node_weight, base_nodes, search_deadline
            )
        if actual_forest is not None and clock() < search_deadline:
            full_candidate = full_tree_from_forest(actual_forest)
            if full_candidate is not None:
                consider(
                    full_candidate[0], full_candidate[1], search_deadline,
                    True, initial_cap, initial_elites
                )
        if actual_forest is not None and clock() < search_deadline:
            candidate = static_construct(
                actual_forest, 1.0, 0.0, None, search_deadline, rng
            )
            if candidate is not None:
                consider(
                    candidate[0], candidate[1], search_deadline,
                    True, initial_cap, initial_elites
                )

        if base_ok and clock() < search_deadline:
            candidate = prim_full(edge_weight, node_weight, search_deadline)
            if candidate is not None:
                consider(
                    candidate[0], candidate[1], search_deadline,
                    True, initial_cap, initial_elites
                )

        if edge_order is not None and clock() < search_deadline:
            all_mst = induced_mst(fallback_nodes, search_deadline)
            if all_mst is not None and is_tree(fallback_nodes, all_mst):
                consider(
                    fallback_nodes, all_mst, search_deadline,
                    True, initial_cap, initial_elites
                )

        if base_ok and clock() < search_deadline:
            biased_node = list(node_weight)
            for v in turbines:
                biased_node[v] /= 1.0 + 2.5 * yields[v] / yield_scale
            biased_forest = shortest_forest(
                base_nodes, edge_weight, biased_node, base_nodes, search_deadline
            )
            if biased_forest is not None:
                candidate = full_tree_from_forest(biased_forest)
                if candidate is not None:
                    consider(
                        candidate[0], candidate[1], search_deadline,
                        True, initial_cap, initial_elites
                    )

        if base_ok and clock() < search_deadline:
            candidate = dynamic_grow(
                base_nodes, base_edges, edge_weight, node_weight,
                1.0, 1, 0.16, search_deadline, rng
            )
            if candidate is not None:
                consider(
                    candidate[0], candidate[1], search_deadline,
                    True, initial_cap, initial_elites
                )

        if best_valid and clock() < search_deadline:
            candidate = enrich_tree(
                best_nodes, best_edges, edge_weight, node_weight,
                branch_width, min(search_deadline, clock() + move_slice),
            )
            if candidate is not None:
                consider(
                    candidate[0], candidate[1], search_deadline,
                    True, initial_cap, initial_elites,
                )

        if elites and clock() < search_deadline:
            candidate = terminal_swap(
                elites[0], edge_weight, node_weight, 0.0, search_deadline
            )
            if candidate is not None:
                consider(
                    candidate[0], candidate[1], search_deadline,
                    True, initial_cap, initial_elites,
                )

        search_span = max(1e-9, search_deadline - started)
        while base_ok and clock() < search_deadline:
            if clock() > deadline:
                break
            now = clock()
            elapsed = max(0.0, now - started)
            stagnation = max(0.0, now - last_improvement)
            progress = min(1.0, elapsed / search_span)
            stale_fraction = min(1.0, stagnation / search_span)

            level = math.log1p(1.0 + effort * (1.0 + 3.0 * progress + 5.0 * stale_fraction))
            time_level = math.log1p(1.0 + effort * (1.0 + 3.0 * progress))

            dp_cap = max(32, int(initial_cap * (1.0 + progress + 1.5 * stale_fraction)))
            elite_cap = max(initial_elites, int(initial_elites + effort * (progress + stale_fraction)))
            rcl = max(1, int(1 + 1.05 * level))
            batch_fraction = max(0.035, 0.22 / (1.0 + 0.11 * level))
            amplitude_ceiling = 0.10 + 0.15 * level
            amplitude = 0.015 + min(1.45, amplitude_ceiling) * rng.random()
            prize_pressure = (0.15 + 0.75 * level) * (rng.random() ** 1.4)

            guide = pick_elite()
            if guide is not None and rng.random() >= min(0.62, 0.28 + 0.045 * level):
                guide = None
            weights = make_weights(
                amplitude, guide, prize_pressure, search_deadline
            )
            if weights is None:
                break
            search_edge, search_node = weights
            candidate = None
            deep = True
            roll = rng.random()

            if roll < 0.22 and elites:
                first = pick_elite()
                second = pick_elite()
                width = max(1, int(branch_width * (0.4 + rng.random() + stale_fraction)))
                candidate = recombine_envelope(
                    first, second, search_edge, search_node, width,
                    min(search_deadline, clock() + move_slice),
                )

            elif roll < 0.34:
                forest = shortest_forest(
                    base_nodes, search_edge, search_node,
                    base_nodes, search_deadline
                )
                if forest is not None:
                    candidate = full_tree_from_forest(forest)

            elif roll < 0.42:
                candidate = prim_full(search_edge, search_node, search_deadline)

            elif roll < 0.51:
                if rng.random() < 0.48:
                    forest = shortest_forest(
                        base_nodes, search_edge, search_node,
                        base_nodes, search_deadline
                    )
                    if forest is not None:
                        seeds = None
                        source = pick_elite()
                        if source is not None and rng.random() < 0.55:
                            keep_probability = 0.08 + 0.72 * rng.random()
                            seeds = [
                                t for t in source[3]
                                if rng.random() < keep_probability
                            ]
                        candidate = static_construct(
                            forest,
                            0.58 + 0.92 * rng.random(),
                            min(0.85, 0.05 + 0.08 * level),
                            seeds,
                            search_deadline,
                            rng,
                        )
                else:
                    candidate = dynamic_grow(
                        base_nodes, base_edges, search_edge, search_node,
                        0.66 + 0.78 * rng.random(), rcl,
                        batch_fraction, search_deadline, rng
                    )

            elif roll < 0.65 and elites:

                source = pick_elite()
                forest = shortest_forest(
                    source[1], search_edge, search_node,
                    source[1], search_deadline
                )
                if forest is not None:
                    _, actual, path_gain, parent_node, parent_edge = forest
                    additions = []
                    beta = 0.70 + 0.55 * rng.random()
                    for pos, t in enumerate(turbines):
                        if (pos & 255) == 0 and clock() >= search_deadline:
                            break
                        if t in source[1] or path_gain[t] <= 0.0:
                            continue
                        additions.append((
                            actual[t] / (max(path_gain[t], 1e-15) ** beta)
                            * math.exp(0.35 * (2.0 * rng.random() - 1.0)),
                            t,
                        ))
                    if additions and clock() < search_deadline:
                        pool_size = min(
                            len(additions),
                            max(rcl * 5, int(8 + 4.0 * level)),
                        )
                        pool = heapq.nsmallest(pool_size, additions)
                        rng.shuffle(pool)
                        cn = set(source[1])
                        ce = set(source[2])
                        target_additions = max(1, int(1 + level * (0.35 + rng.random())))
                        added_count = 0
                        for _, target in pool:
                            if clock() >= search_deadline or added_count >= target_additions:
                                break
                            before = len(cn)
                            result = attach_path(
                                target, cn, ce, parent_node, parent_edge
                            )
                            if result is not None and len(cn) > before:
                                added_count += 1
                        if added_count > 0:
                            candidate = (cn, ce)

            elif roll < 0.75 and elites:
                source = pick_elite()
                candidate = cut_branch_and_repair(
                    source, search_edge, search_node,
                    min(1.0, 0.18 + 0.30 * level),
                    search_deadline, rng,
                )

            elif roll < 0.84 and elites:
                source = pick_elite()
                candidate = terminal_swap(
                    source, search_edge, search_node,
                    min(0.8, 0.08 + 0.18 * level), search_deadline,
                )

            elif roll < 0.93 and elites:

                source = pick_elite()
                source_turbines = list(source[3])
                if source_turbines:
                    degree = [0] * n
                    leaf_edge = [-1] * n
                    for e in source[2]:
                        u, v = edges[e]
                        degree[u] += 1
                        degree[v] += 1
                        leaf_edge[u] = e
                        leaf_edge[v] = e
                    ranked = []
                    randomness = 0.28 + 0.11 * level
                    for t in source_turbines:
                        marginal = node_weight[t]
                        if degree[t] == 1 and leaf_edge[t] >= 0:
                            marginal += edge_weight[leaf_edge[t]]
                        badness = marginal / max(yields[t], 1e-15)
                        ranked.append((
                            math.log1p(max(0.0, badness))
                            + randomness * rng.random(),
                            t,
                        ))
                    ranked.sort(reverse=True)
                    max_fraction = 0.10 + 0.76 * (1.0 - math.exp(-0.16 * level))
                    remove_fraction = 0.02 + max_fraction * rng.random()
                    remove_count = max(
                        1,
                        min(len(source_turbines), int(len(source_turbines) * remove_fraction)),
                    )
                    removed = {t for _, t in ranked[:remove_count]}
                    retained = set(source_turbines) - removed
                    candidate = metric_reconnect(
                        set(substations) | retained,
                        search_edge, search_node, search_deadline
                    )
                    if candidate is not None and energy_of(candidate[0]) < quota:
                        candidate = dynamic_grow(
                            candidate[0], candidate[1],
                            search_edge, search_node,
                            0.64 + 0.80 * rng.random(), rcl,
                            batch_fraction, search_deadline, rng
                        )

            elif elites:

                first = pick_elite()
                second = pick_elite() if len(elites) > 1 else first
                common = set(first[3]) & set(second[3])
                differing = list((set(first[3]) | set(second[3])) - common)
                retained = set(common)
                retained_energy = math.fsum(yields[t] for t in retained)
                target_energy = quota * (0.15 + 0.76 * rng.random())
                differing.sort(key=lambda t: (
                    (node_weight[t] + tie_scale * 1e-12)
                    / max(yields[t], 1e-15)
                    * math.exp(0.55 * (2.0 * rng.random() - 1.0))
                ))
                for t in differing:
                    if retained_energy >= target_energy:
                        break
                    retained.add(t)
                    retained_energy += yields[t]
                candidate = metric_reconnect(
                    set(substations) | retained,
                    search_edge, search_node, search_deadline
                )
                if candidate is not None and energy_of(candidate[0]) < quota:
                    candidate = dynamic_grow(
                        candidate[0], candidate[1],
                        search_edge, search_node,
                        0.64 + 0.80 * rng.random(), rcl,
                        batch_fraction, search_deadline, rng
                    )

            if candidate is not None and clock() < search_deadline:
                consider(
                    candidate[0], candidate[1], search_deadline,
                    deep, dp_cap, elite_cap,
                    rng if rng.random() < 0.30 else None,
                )

        return answer

    except BaseException:
        return answer
