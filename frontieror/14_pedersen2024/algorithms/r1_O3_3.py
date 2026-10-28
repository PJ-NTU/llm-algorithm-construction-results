"""Dual-guided quota tree search
Injects Lagrangian quota-boundary subtrees into the dynamic-programming frontier and keeps improving the feasible tree with branch expansion, destroy-and-repair and shortest-path reconnection.
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
        
        deadline = t0 + 0.94 * budget
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

            while heap and clock() < search_deadline:
                _, _, _, v = heapq.heappop(heap)
                if v not in nodes or v in sub_set or degree[v] > 1:
                    continue
                loss = yields[v] if is_turbine[v] else 0.0
                new_energy = energy - loss
                if is_turbine[v] and new_energy < quota + safety:
                    new_energy = math.fsum(yields[w] for w in nodes
                                           if is_turbine[w] and w != v)
                    if new_energy < quota:
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
            repair_node = list(search_node)
            surcharge = (subtree_cost[cut] + edge_weight[parent_edge[cut]]) / max(subtree_energy[cut], 1e-15)
            for v in removed:
                if is_turbine[v]:
                    repair_node[v] += strength * surcharge * yields[v]
            return dynamic_grow(
                nodes, selected_edges, search_edge, repair_node,
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

        def fused_tree_search(nodes, selected_edges, cap, cutoff):
            """Fuse dual-supported solutions with quota dynamic programming in the same tree frontier."""
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
                for v, e in local[u]:
                    if parent[v] == -2:
                        parent[v] = u
                        parent_edge[v] = e
                        children[u].append(v)
                        order.append(v)
            if len(order) != len(nodes):
                return None, 0.0
            bit_of = {v: 1 << i for i, v in enumerate(order)}
            required = [False] * n
            own_cost = [0.0] * n
            own_energy = [0.0] * n
            for v in reversed(order):
                own_energy[v] = yields[v] if is_turbine[v] else 0.0
                own_cost[v] = node_weight[v]
                if v != root:
                    own_cost[v] += edge_weight[parent_edge[v]]
                required[v] = v in sub_set or any(required[c] for c in children[v])

            def decode(mask):
                ns, es = set(), set()
                while mask:
                    bit = mask & -mask
                    v = order[bit.bit_length() - 1]
                    ns.add(v)
                    if v != root:
                        es.add(parent_edge[v])
                    mask ^= bit
                return ns, es

            candidate = None
            candidate_value = infinity
            seen = set()

            def retain(state):
                nonlocal candidate, candidate_value
                if state is None or state[2] in seen:
                    return
                seen.add(state[2])
                ns, es = decode(state[2])
                if energy_of(ns) < quota:
                    return
                ns, es = prune_tree(ns, es)
                if feasible(ns, es):
                    value = objective(ns, es)
                    if value < candidate_value:
                        candidate, candidate_value = (ns, es), value

            # The dual phase works directly on the unpruned expanded tree, keeping whole subtrees on both sides of the quota.
            dual_cutoff = min(cutoff, clock() + 0.27 * max(0.0, cutoff - clock()))

            def evaluate(lam):
                states = [None] * n
                for pos, v in enumerate(reversed(order)):
                    if (pos & 255) == 0 and clock() >= dual_cutoff:
                        return None
                    energy, cost, mask = own_energy[v], own_cost[v], bit_of[v]
                    for c in children[v]:
                        ec, cc, mc = states[c]
                        if required[c] or cc <= lam * ec:
                            energy += ec
                            cost += cc
                            mask |= mc
                    states[v] = (energy, cost, mask)
                ns, _ = decode(states[root][2])
                states[root] = (energy_of(ns), states[root][1], states[root][2])
                retain(states[root])
                return states

            low_lam = 0.0
            low = evaluate(low_lam)
            high = None
            high_lam = max(1e-12, objective(nodes, selected_edges) / max(quota, 1e-12))
            if low is not None and low[root][0] < quota:
                high = evaluate(high_lam)
                while high is not None and high[root][0] < quota and clock() < dual_cutoff:
                    low_lam, low = high_lam, high
                    high_lam *= 2.0
                    if not math.isfinite(high_lam):
                        break
                    high = evaluate(high_lam)
                # The secant through adjacent support points gives the next multiplier, more directly than fixed-precision bisection.
                while high is not None and low is not None and clock() < dual_cutoff:
                    le, lc, lm = low[root]
                    he, hc, hm = high[root]
                    if he < quota or he <= le:
                        break
                    trial = (hc - lc) / (he - le)
                    if not (low_lam < trial < high_lam):
                        trial = 0.5 * (low_lam + high_lam)
                    if trial == low_lam or trial == high_lam:
                        break
                    middle = evaluate(trial)
                    if middle is None:
                        break
                    if middle[root][2] == lm or middle[root][2] == hm:
                        break
                    if middle[root][0] >= quota:
                        high_lam, high = trial, middle
                    else:
                        low_lam, low = trial, middle
            else:
                high = low
                high_lam = 0.0
            policies = [p for p in (low, high) if p is not None]
            lam = high_lam if math.isfinite(high_lam) else low_lam
            if low is not None and high is not None and high[root][0] > low[root][0]:
                lam = max(0.0, (high[root][1] - low[root][1]) /
                          (high[root][0] - low[root][0]))
            if clock() >= cutoff:
                return candidate, lam

            # A state holds true energy, true cost and a local node mask; the quota is not discretised.
            frontiers = [None] * n
            cap = max(8, int(cap))

            def compress(options, anchors):
                options.extend(anchors)
                options.sort(key=lambda z: (-min(quota, z[0]), z[1]))
                kept = []
                least = infinity
                for state in options:
                    if state[1] < least:
                        kept.append(state)
                        least = state[1]
                if len(kept) <= cap:
                    return kept
                protected = {state[2] for state in anchors}
                indexes = {0, len(kept) - 1}
                indexes.update(i for i, st in enumerate(kept) if st[2] in protected)
                # Keep both energy coverage and non-support points near the multiplier, rather than sampling uniformly only.
                spread = max(2, cap // 2)
                indexes.update(round(j * (len(kept) - 1) / (spread - 1))
                               for j in range(spread))
                slots = max(0, cap - len(indexes))
                ranked = heapq.nsmallest(slots,
                    ((st[1] - lam * min(quota, st[0]), i)
                     for i, st in enumerate(kept) if i not in indexes))
                indexes.update(i for _, i in ranked)
                return [kept[i] for i in sorted(indexes)]

            operations = 0
            for v in reversed(order):
                if clock() >= cutoff:
                    return candidate, lam
                own = (own_energy[v], own_cost[v], bit_of[v])
                frontier = [own]
                prefixes = [own for _ in policies]
                for child in sorted(children[v], key=lambda c: len(frontiers[c])):
                    choices = frontiers[child]
                    if not required[child]:
                        choices = [(0.0, 0.0, 0)] + choices
                    merged = []
                    for ea, ca, ma in frontier:
                        for eb, cb, mb in choices:
                            merged.append((ea + eb, ca + cb, ma | mb))
                            operations += 1
                            if (operations & 2047) == 0 and clock() >= cutoff:
                                return candidate, lam
                    for i, policy in enumerate(policies):
                        child_state = policy[child]
                        if policy[v][2] & bit_of[child]:
                            ea, ca, ma = prefixes[i]
                            eb, cb, mb = child_state
                            prefixes[i] = (ea + eb, ca + cb, ma | mb)
                    frontier = compress(merged, prefixes)
                    frontiers[child] = None
                anchors = [policy[v] for policy in policies]
                frontiers[v] = compress(frontier, anchors)
            for state in frontiers[root]:
                if clock() >= cutoff:
                    break
                if state[0] >= quota - 1e-12 * max(1.0, quota):
                    retain(state)
            return candidate, lam

        positive = [edge_weight[e] for e in valid_edges if edge_weight[e] > 0.0]
        positive.extend(node_weight[v] for v in turbines if node_weight[v] > 0.0)
        tie_scale = math.fsum(positive) / len(positive) if positive else 1.0
        seed = ((n + 1) * 1000003) ^ ((m + 7) * 1000033)
        for v in turbines:
            seed = ((seed * 1000003) ^ int(yields[v] * 1009)
                    ^ int(node_weight[v] * 101)) & ((1 << 64) - 1)
        rng = random.Random(seed)
        elites = []
        last_improvement = clock()
        dual_price = best_value / max(quota, 1e-12) if best_valid else 0.0
        effort = math.log1p(budget)
        scale = 1.0 + math.log1p(n + m)
        population = max(3, int(3 + 2 * effort))
        base_cap = max(12, int((24 + 32 * effort) /
                              max(1.0, math.sqrt(n / max(1.0, 40 * (1 + effort))))))

        def register(ns, es):
            nonlocal best_nodes, best_edges, best_value, best_valid, answer, last_improvement
            if not feasible(ns, es):
                return
            value = objective(ns, es)
            if not best_valid or value < best_value:
                best_nodes, best_edges = set(ns), set(es)
                best_value, best_valid = value, True
                answer = make_output(ns, es)
                last_improvement = clock()
            signature = frozenset(v for v in ns if is_turbine[v])
            record = (value, set(ns), set(es), signature)
            for i, old in enumerate(elites):
                if old[3] == signature:
                    if value < old[0]:
                        elites[i] = record
                        elites.sort(key=lambda z: z[0])
                    return
            elites.append(record)
            elites.sort(key=lambda z: z[0])
            del elites[population:]

        def consider(candidate, cap, cutoff):
            nonlocal dual_price
            if candidate is None:
                return
            ns, es = candidate
            ns, es = set(ns), set(es)
            if not is_tree(ns, es):
                es = induced_mst(ns, cutoff)
                if es is None:
                    return
            if energy_of(ns) < quota or not sub_set.issubset(ns):
                return
            pn, pe = prune_tree(ns, es)
            register(pn, pe)
            # Save feasible improvements first, then run the interruptible deep optimisation.
            if clock() < cutoff:
                mst = induced_mst(pn, cutoff)
                if mst is not None:
                    rn, re = prune_tree(pn, mst)
                    register(rn, re)
            if clock() < cutoff:
                improved, price = fused_tree_search(ns, es, cap, cutoff)
                if math.isfinite(price) and price >= 0.0:
                    dual_price = 0.65 * dual_price + 0.35 * price
                if improved is not None:
                    dn, de = improved
                    register(dn, de)
                    if clock() < cutoff:
                        mst = induced_mst(dn, cutoff)
                        if mst is not None:
                            register(*prune_tree(dn, mst))

        def pick_elite():
            if not elites:
                return None
            return elites[min(len(elites) - 1, int(rng.random() ** 2 * len(elites)))]

        def weights(amplitude, guide, pressure, cutoff):
            se, sn = list(edge_weight), list(node_weight)
            gn = guide[1] if guide is not None else set()
            ge = guide[2] if guide is not None else set()
            eps = tie_scale * 1e-12
            for pos, e in enumerate(valid_edges):
                if (pos & 1023) == 0 and clock() >= cutoff:
                    return None
                se[e] = max(eps, edge_weight[e]) * math.exp(amplitude * rng.uniform(-1, 1))
                if e in ge:
                    se[e] *= 0.85
            for v in turbines:
                # Dual energy prices are turned into non-negative search costs, keeping the shortest-path algorithm valid.
                discount = pressure * dual_price * yields[v]
                sn[v] = max(eps, node_weight[v] - discount)
                sn[v] *= math.exp(amplitude * rng.uniform(-1, 1))
                if v in gn:
                    sn[v] *= 0.85
            return se, sn

        def swap_terminals(source, se, sn, width, amplitude, cutoff):
            if source is None:
                return None
            forest = shortest_forest(source[1], se, sn, source[1], cutoff)
            if forest is None:
                return None
            actual = forest[1]
            outside = [v for v in turbines if v not in source[3] and math.isfinite(actual[v])]
            if not outside:
                return None
            degree, incident = [0] * n, [-1] * n
            for e in source[2]:
                u, v = edges[e]
                degree[u] += 1
                degree[v] += 1
                incident[u] = incident[v] = e
            old_rank = []
            for v in source[3]:
                saving = node_weight[v] + (edge_weight[incident[v]] if degree[v] == 1 else 0)
                old_rank.append((-(saving / max(yields[v], 1e-15)) *
                                 math.exp(amplitude * rng.uniform(-1, 1)), v, saving))
            old_rank = heapq.nsmallest(width, old_rank)
            energy = energy_of(source[1])
            chosen = None
            best_delta = infinity
            for _, old, saving in old_rank:
                need = quota - energy + yields[old]
                for j, new in enumerate(outside):
                    if (j & 255) == 0 and clock() >= cutoff:
                        return None
                    if yields[new] < need:
                        continue
                    delta = actual[new] * math.exp(amplitude * rng.uniform(-1, 1)) - saving
                    if delta < best_delta:
                        best_delta, chosen = delta, (old, new)
            if chosen is None:
                return None
            terminals = set(source[3])
            terminals.remove(chosen[0])
            terminals.add(chosen[1])
            return metric_reconnect(terminals, se, sn, cutoff)

        if best_valid:
            register(best_nodes, best_edges)
        starters = deque(('grow', 'forest', 'prim', 'enrich', 'reconnect'))
        search_span = max(1e-12, search_deadline - t0)
        # No cap on restarts; as the budget grows, frontier width, neighbourhood width and elite capacity grow together.
        while base_ok and clock() < search_deadline:
            now = clock()
            stale = min(1.0, (now - last_improvement) / search_span)
            progress = min(1.0, (now - t0) / search_span)
            cap = max(8, int(base_cap * (1.0 + 0.65 * progress + stale)))
            population = max(3, int(3 + effort * (2.0 + progress + stale)))
            width = max(1, int(1 + effort * (1.2 + progress + 2 * stale)))
            rcl = max(1, int(1 + effort * (0.4 + stale)))
            batch = max(0.02, 0.24 / (1 + 0.2 * effort + stale))
            amplitude = (0.04 + 0.10 * effort + 0.65 * stale) * rng.random()
            # Each neighbourhood gets a wall-clock time slice set by the total budget and graph size.
            quantum = budget / (8.0 + 2.0 * scale)
            cutoff = min(search_deadline, now + quantum)
            generator_cutoff = min(cutoff, now + 0.58 * quantum)
            source = pick_elite()
            if starters:
                mode = starters.popleft()
                se, sn = edge_weight, node_weight
                local_amplitude = 0.0
            else:
                rolled = rng.random()
                if rolled < 0.17:
                    mode = 'forest'
                elif rolled < 0.25:
                    mode = 'prim'
                elif rolled < 0.39:
                    mode = 'grow'
                elif rolled < 0.61:
                    mode = 'enrich'
                elif rolled < 0.74:
                    mode = 'branch'
                elif rolled < 0.85:
                    mode = 'ruin'
                elif rolled < 0.95:
                    mode = 'swap'
                else:
                    mode = 'relink'
                search_weights = weights(amplitude, source if rng.random() < 0.45 else None,
                                         rng.random() ** 2, generator_cutoff)
                if search_weights is None:
                    continue
                se, sn = search_weights
                local_amplitude = amplitude
            candidate = None
            if mode == 'forest':
                forest = shortest_forest(base_nodes, se, sn, base_nodes, generator_cutoff)
                if forest is not None:
                    candidate = full_tree_from_forest(forest)
            elif mode == 'prim':
                candidate = prim_full(se, sn, generator_cutoff)
            elif mode == 'grow':
                candidate = dynamic_grow(base_nodes, base_edges, se, sn,
                                         1.0 if local_amplitude == 0 else rng.uniform(0.65, 1.3),
                                         rcl, batch, generator_cutoff, rng)
            elif mode == 'enrich' and source is not None:
                candidate = enrich_tree(source[1], source[2], se, sn, width, generator_cutoff)
            elif mode == 'reconnect' and source is not None:
                candidate = metric_reconnect(source[3], se, sn, generator_cutoff)
            elif mode == 'branch' and source is not None:
                candidate = cut_branch_and_repair(source, se, sn,
                    min(1.0, 0.15 + 0.10 * effort + stale), generator_cutoff, rng)
            elif mode == 'swap' and source is not None:
                candidate = swap_terminals(source, se, sn, width, amplitude, generator_cutoff)
            elif mode in ('ruin', 'relink') and source is not None:
                terminals = set(source[3])
                if mode == 'relink':
                    second = pick_elite()
                    common = terminals & set(second[3])
                    different = list((terminals | set(second[3])) - common)
                    rng.shuffle(different)
                    terminals = common
                    desired = quota * rng.uniform(0.3, 0.85)
                    for v in different:
                        if energy_of(terminals) >= desired:
                            break
                        terminals.add(v)
                else:
                    fraction = min(0.85, (0.08 + 0.06 * effort + stale * 0.45) * rng.random())
                    ranked = sorted(terminals, key=lambda v:
                        (node_weight[v] / max(yields[v], 1e-15) + tie_scale * rng.random())
                        * math.exp(amplitude * rng.uniform(-1, 1)), reverse=True)
                    count = max(1, int(len(ranked) * fraction))
                    terminals.difference_update(ranked[:count])
                candidate = metric_reconnect(terminals, se, sn, generator_cutoff)
                if candidate is not None and energy_of(candidate[0]) < quota:
                    candidate = dynamic_grow(candidate[0], candidate[1], se, sn,
                                             rng.uniform(0.7, 1.2), rcl, batch,
                                             generator_cutoff, rng)
            if candidate is not None:
                consider(candidate, cap, cutoff)
        return answer
    except BaseException:
        return answer
