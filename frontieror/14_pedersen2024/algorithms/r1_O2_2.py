"""Connectivity-cut-strengthened integer search for the quota tree
First reduces zero-cost substations and Steiner chains and secures a feasible tree as fallback, then jointly optimises node selection and cabling with flow constraints, separated connectivity cuts and adaptive turbine-swap neighbourhoods.
"""

import heapq
import math
import random
import time


def solve(instance: dict, time_limit_s: float) -> dict:
    start = time.perf_counter()
    answer = {"node_selected": [], "edge_selected": []}
    try:
        budget = max(0.0, float(time_limit_s))
        if not math.isfinite(budget):
            budget = 0.0
        reserve = min(0.15 * budget, 0.08 + 0.025 * budget)
        deadline = start + budget - reserve
        kinds = instance["node_types"]
        raw_edges = instance["edges"]
        n, m = len(kinds), len(raw_edges)
        answer = {"node_selected": [0] * n, "edge_selected": [0] * m}
        if not n:
            return answer
        stations = {v for v in range(n) if kinds[v] == "substation"}
        root = min(stations) if stations else 0
        stations.add(root)
        turbine = [k == "turbine" for k in kinds]
        alpha = float(instance["alpha"])
        energy = [float(instance["annual_energy_yields"][v]) if turbine[v] else 0.0
                  for v in range(n)]
        nw = [alpha * float(instance["installation_costs"][v])
              + (1.0 - alpha) * float(instance["node_scenic_impacts"][v])
              if turbine[v] else 0.0 for v in range(n)]
        ew = [alpha * float(instance["edge_cable_costs"][e])
              + (1.0 - alpha) * float(instance["edge_scenic_impacts"][e])
              for e in range(m)]
        quota = float(instance["minimum_energy_quota"])
        endpoints = [(int(p[0]), int(p[1])) for p in raw_edges]
        adj0 = [[] for _ in range(n)]
        for e, (u, v) in enumerate(endpoints):
            if u != v and 0 <= u < n and 0 <= v < n:
                adj0[u].append((v, e))
                adj0[v].append((u, e))

        # The fallback is a single linear traversal and depends on no optimisation library.
        seen = {root}
        walk = [root]
        initial = set()
        for u in walk:
            for v, e in adj0[u]:
                if v not in seen:
                    seen.add(v)
                    walk.append(v)
                    initial.add(e)
        for v in seen:
            answer["node_selected"][v] = 1
        for e in initial:
            answer["edge_selected"][e] = 1
        best_cost = math.fsum(nw[v] for v in seen) + math.fsum(ew[e] for e in initial)
        best_valid = stations <= seen and math.fsum(energy[v] for v in seen) >= quota
        if not best_valid:
            best_cost = math.inf
        if time.perf_counter() >= deadline:
            return answer
        rng = random.Random(7919 * n + 104729 * m + 17)
        nonnegative = min(nw + ew + [0.0]) >= 0.0

        # Once the zero-cost tree between stations is fixed, all substations can be contracted; with negative costs the original graph is kept.
        star = {}
        for v, e in adj0[root]:
            if v in stations and v != root and ew[e] == 0.0:
                star[v] = e
        contract = nonnegative and len(star) == len(stations) - 1
        fixed_edges = set(star.values()) if contract else set()
        mandatory = {root} if contract else set(stations)
        live = (set(seen) - stations) | {root} if contract else set(seen)
        graph = {v: {} for v in live}
        records = []

        def add_edge(u, v, weight, original, left=-1, right=-1):
            old = graph[u].get(v)
            if old is not None and records[old][2] <= weight:
                return
            j = len(records)
            records.append((u, v, weight, original, left, right))
            graph[u][v] = j
            graph[v][u] = j

        for e, (u, v) in enumerate(endpoints):
            if contract:
                u = root if u in stations else u
                v = root if v in stations else v
            if u != v and u in live and v in live:
                add_edge(u, v, ew[e], e)
        if nonnegative:
            queue = [v for v in live if v not in mandatory and not turbine[v]
                     and len(graph[v]) <= 2]
            while queue and time.perf_counter() < deadline:
                v = queue.pop()
                if v not in live or v in mandatory or turbine[v] or len(graph[v]) > 2:
                    continue
                neighbors = list(graph[v].items())
                for u, _ in neighbors:
                    del graph[u][v]
                graph[v].clear()
                live.remove(v)
                if len(neighbors) == 2:
                    (u, a), (w, b) = neighbors
                    add_edge(u, w, records[a][2] + records[b][2], -1, a, b)
                for u, _ in neighbors:
                    if u not in mandatory and not turbine[u] and len(graph[u]) <= 2:
                        queue.append(u)
        vertices = sorted(live)
        index = {v: i for i, v in enumerate(vertices)}
        r = index[root]
        size = len(vertices)
        req = {index[v] for v in mandatory}
        gain = [energy[v] for v in vertices]
        cost = [nw[v] for v in vertices]
        wind = [turbine[v] for v in vertices]
        edge_ids = [j for u in vertices for v, j in graph[u].items() if u < v]
        links = [(index[records[j][0]], index[records[j][1]], records[j][2]) for j in edge_ids]
        adj = [[] for _ in vertices]
        for e, (u, v, w) in enumerate(links):
            adj[u].append((v, e))
            adj[v].append((u, e))
        best_nodes = {index[v] for v in vertices if answer["node_selected"][v]}
        last_improvement = time.perf_counter()

        def accept(chosen):
            nonlocal answer, best_cost, best_nodes, best_valid, last_improvement
            selected = set(req)
            parent = list(range(size))

            def find(v):
                while parent[v] != v:
                    parent[v] = parent[parent[v]]
                    v = parent[v]
                return v

            value = 0.0
            for e in chosen:
                u, v, w = links[e]
                a, b = find(u), find(v)
                if a == b:
                    return False
                parent[a] = b
                selected.add(u)
                selected.add(v)
                value += w
            if len(chosen) != len(selected) - 1 or any(find(v) != find(r) for v in selected):
                return False
            if math.fsum(gain[v] for v in selected) < quota:
                return False
            value += math.fsum(cost[v] for v in selected)
            if best_valid and value >= best_cost - 1e-10 * max(1.0, abs(best_cost)):
                return False
            original_edges = set(fixed_edges)
            stack = [edge_ids[e] for e in chosen]
            while stack:
                item = records[stack.pop()]
                if item[3] >= 0:
                    original_edges.add(item[3])
                else:
                    stack.extend((item[4], item[5]))
            node_out = [0] * n
            edge_out = [0] * m
            for v in stations:
                node_out[v] = 1
            for e in original_edges:
                edge_out[e] = 1
                u, v = endpoints[e]
                node_out[u] = node_out[v] = 1
            if sum(edge_out) != sum(node_out) - 1:
                return False
            if math.fsum(energy[v] for v in range(n) if node_out[v]) < quota:
                return False
            actual = math.fsum(nw[v] for v in range(n) if node_out[v]) + math.fsum(ew[e] for e in original_edges)
            if best_valid and actual >= best_cost - 1e-10 * max(1.0, abs(best_cost)):
                return False
            best_cost = actual
            best_nodes = selected
            best_valid = True
            last_improvement = time.perf_counter()
            answer = {"node_selected": node_out, "edge_selected": edge_out}
            return True

        def spanning_tree(weights):
            reached = {r}
            picked = set()
            heap = [(weights[e], e, v) for v, e in adj[r]]
            heapq.heapify(heap)
            while heap:
                _, e, v = heapq.heappop(heap)
                if v in reached:
                    continue
                reached.add(v)
                picked.add(e)
                for u, f in adj[v]:
                    if u not in reached:
                        heapq.heappush(heap, (weights[f], f, u))
            return picked

        def trim(picked, cutoff, noise=0.0, preference=None):
            # Remove a whole subtree while updating the removable energy and true savings of its ancestors.
            local = [[] for _ in vertices]
            for e in picked:
                u, v, _ = links[e]
                local[u].append((v, e))
                local[v].append((u, e))
            parent = [-1] * size
            pe = [-1] * size
            order = [r]
            parent[r] = r
            for u in order:
                for v, e in local[u]:
                    if parent[v] < 0:
                        parent[v] = u
                        pe[v] = e
                        order.append(v)
            total = math.fsum(gain[v] for v in order)
            amount = list(gain)
            saving = list(cost)
            protected = [v in req for v in range(size)]
            children = [[] for _ in vertices]
            for v in reversed(order[1:]):
                p = parent[v]
                saving[v] += links[pe[v]][2]
                amount[p] += amount[v]
                saving[p] += saving[v]
                protected[p] = protected[p] or protected[v]
                children[p].append(v)
            active = [parent[v] >= 0 for v in range(size)]
            version = [0] * size
            factor = [math.exp(noise * (2.0 * rng.random() - 1.0)) for _ in vertices]
            if preference is not None:
                factor = [factor[v] * (1.5 - preference[v]) for v in range(size)]
            heap = []

            def push(v):
                if v != r and active[v] and not protected[v]:
                    priority = saving[v] / max(amount[v], 1e-12) * factor[v]
                    heapq.heappush(heap, (-priority, version[v], v))

            for v in order[1:]:
                push(v)
            kept = set(picked)
            while heap and time.perf_counter() < cutoff:
                _, stamp, v = heapq.heappop(heap)
                if not active[v] or stamp != version[v]:
                    continue
                loss, saved = amount[v], saving[v]
                if total - loss < quota or saved < 0.0:
                    continue
                if saved == 0.0 and loss > 0.0:
                    continue
                removed = []
                stack = [v]
                while stack:
                    u = stack.pop()
                    if active[u]:
                        removed.append(u)
                        stack.extend(children[u])
                # Confirm the quota by direct summation, avoiding overshooting a tight quota through accumulated subtraction.
                removed_set = set(removed)
                remaining_energy = math.fsum(gain[u] for u in order if active[u] and u not in removed_set)
                if remaining_energy < quota:
                    continue
                for u in removed:
                    active[u] = False
                    kept.discard(pe[u])
                total = remaining_energy
                u = parent[v]
                while True:
                    amount[u] -= loss
                    saving[u] -= saved
                    version[u] += 1
                    push(u)
                    if u == r:
                        break
                    u = parent[u]
            return kept

        tree = spanning_tree([w for _, _, w in links])
        accept(trim(tree, min(deadline, start + 0.09 * budget)))
        if size <= 1 or time.perf_counter() >= deadline:
            return answer

        def random_tree(cutoff, preference=None):
            intensity = math.log1p(budget) * (0.15 + 0.25 * rng.random())
            weights = []
            for u, v, w in links:
                discount = 1.0
                if preference is not None:
                    discount = 1.05 - 0.45 * (preference[u] + preference[v])
                elif u in best_nodes and v in best_nodes:
                    discount = 0.7 + 0.3 * rng.random()
                weights.append(w * discount * math.exp(intensity * (2.0 * rng.random() - 1.0)))
            candidate = spanning_tree(weights)
            accept(trim(candidate, cutoff, intensity, preference))

        optimizer = None
        try:
            if deadline - time.perf_counter() > 0.08 + 0.03 * budget:
                import numpy as np
                from scipy.optimize import Bounds, LinearConstraint, milp
                from scipy.sparse import csc_matrix, csr_matrix, vstack
                from scipy.sparse.csgraph import maximum_flow
                optimizer = milp
        except BaseException:
            optimizer = None

        if optimizer is not None and time.perf_counter() < deadline:
            # x are implicitly integer node variables, a are directed parent edges and f is a counting flow.
            arcs = []
            incoming = [[] for _ in vertices]
            outgoing = [[] for _ in vertices]
            by_edge = [[] for _ in links]
            for e, (u, v, _) in enumerate(links):
                for a, b in ((u, v), (v, u)):
                    if b != r:
                        j = len(arcs)
                        arcs.append((a, b, e))
                        incoming[b].append(j)
                        outgoing[a].append(j)
                        by_edge[e].append(j)
            count = len(arcs)
            width = size + 2 * count
            ar = size
            fl = size + count
            objective = np.zeros(width)
            scale = max(1.0, max([abs(w) for w in cost] + [abs(w) for _, _, w in links]))
            objective[:size] = np.asarray(cost) / scale
            for j, (_, _, e) in enumerate(arcs):
                objective[ar + j] = links[e][2] / scale
            lower = np.zeros(width)
            upper = np.ones(width)
            bound_flow = max(1, size - 1)
            upper[fl:] = bound_flow
            for v in req:
                lower[v] = 1.0
            integrality = np.zeros(width, dtype=np.uint8)
            integrality[ar:fl] = 1
            rows, cols, data, lows, highs = [], [], [], [], []

            def add_row(entries, lo, hi):
                k = len(lows)
                for col, value in entries:
                    if value:
                        rows.append(k)
                        cols.append(col)
                        data.append(float(value))
                lows.append(lo)
                highs.append(hi)

            if quota > 0.0:
                add_row(((v, gain[v] / quota) for v in range(size)), 1.0, np.inf)
            for v in range(size):
                if v == r:
                    continue
                add_row([(v, -1.0)] + [(ar + j, 1.0) for j in incoming[v]], 0.0, 0.0)
                add_row([(v, -1.0)] + [(fl + j, 1.0) for j in incoming[v]]
                        + [(fl + j, -1.0) for j in outgoing[v]], 0.0, 0.0)
                if nonnegative and not wind[v] and v not in req:
                    add_row([(ar + j, 1.0) for j in outgoing[v]] + [(v, -1.0)], 0.0, np.inf)
            for j, (u, v, _) in enumerate(arcs):
                if u != r:
                    add_row(((ar + j, 1.0), (u, -1.0)), -np.inf, 0.0)
                cap = 1 if len(adj[v]) == 1 else bound_flow - (u != r)
                cap = max(1, cap)
                add_row(((fl + j, 1.0), (ar + j, -cap)), -np.inf, 0.0)
                add_row(((fl + j, 1.0), (ar + j, -1.0)), 0.0, np.inf)
            for group in by_edge:
                if len(group) > 1:
                    add_row(((ar + j, 1.0) for j in group), -np.inf, 1.0)
            bounds = Bounds(lower, upper)

            def matrix():
                return csc_matrix((data, (rows, cols)), shape=(len(lows), width))

            model_matrix = matrix()
            extra_cuts = set()
            relaxation_end = min(deadline, start + 0.21 * budget)
            relaxation = None
            # Root-to-node minimum cuts strengthen the counting-flow relaxation without extra integer variables.
            while time.perf_counter() < relaxation_end:
                allowance = relaxation_end - time.perf_counter()
                if allowance < 0.015:
                    break
                lp = optimizer(objective, integrality=np.zeros(width, dtype=np.uint8),
                               bounds=bounds,
                               constraints=LinearConstraint(model_matrix, lows, highs),
                               options={"time_limit": max(0.001, 0.65 * allowance), "presolve": True})
                if lp.x is None:
                    break
                relaxation = lp.x
                fractional = np.maximum(0.0, relaxation[ar:fl])
                capacities = csr_matrix((np.rint(fractional * 1000000.0).astype(np.int64),
                                         ([a for a, _, _ in arcs], [b for _, b, _ in arcs])),
                                        shape=(size, size))
                targets = sorted((v for v in range(size) if v != r and relaxation[v] > 1e-5),
                                 key=lambda v: -relaxation[v])
                added = False
                for target in targets:
                    if time.perf_counter() >= relaxation_end:
                        break
                    flow = maximum_flow(capacities, r, target)
                    if flow.flow_value / 1000000.0 >= relaxation[target] - 1e-5:
                        continue
                    residual = (capacities - flow.flow).tocsr()
                    reached = {r}
                    queue = [r]
                    for u in queue:
                        for p in range(residual.indptr[u], residual.indptr[u + 1]):
                            v = int(residual.indices[p])
                            if residual.data[p] > 0 and v not in reached:
                                reached.add(v)
                                queue.append(v)
                    outside = frozenset(set(range(size)) - reached)
                    signature = (outside, target)
                    if not outside or signature in extra_cuts:
                        continue
                    crossing = [j for j, (u, v, _) in enumerate(arcs) if u in reached and v in outside]
                    if sum(fractional[j] for j in crossing) >= relaxation[target] - 1e-6:
                        continue
                    add_row([(ar + j, 1.0) for j in crossing] + [(target, -1.0)], 0.0, np.inf)
                    extra_cuts.add(signature)
                    added = True
                if not added:
                    break
                model_matrix = matrix()
            if relaxation is not None and time.perf_counter() < deadline:
                random_tree(min(deadline, time.perf_counter() + 0.035 * budget),
                            [float(min(1.0, max(0.0, relaxation[v]))) for v in range(size)])
            model_matrix = matrix()
            base_low = np.asarray(lows)
            base_high = np.asarray(highs)
            objective_row = csc_matrix(objective.reshape(1, -1))
            terminals = [v for v in range(size) if wind[v] and v not in req]
            first = True
            local_radius = max(2, int(math.sqrt(max(1, len(terminals))) * math.log1p(budget) / 3.0))
            quantum = math.sqrt(max(budget, 1e-9)) * (0.1 + 0.02 * math.log1p(size + count))
            while time.perf_counter() < deadline:
                remaining = deadline - time.perf_counter()
                if remaining < reserve * 0.12 + 0.003:
                    break
                stale = (time.perf_counter() - last_improvement) / max(budget, 1e-9)
                local = not first and best_valid and rng.random() < 0.72 and bool(terminals)
                matrices = [model_matrix]
                lb = [base_low]
                ub = [base_high]
                if best_valid:
                    matrices.append(objective_row)
                    lb.append(np.array([-np.inf]))
                    ub.append(np.array([best_cost / scale + 1e-8]))
                radius = min(len(terminals), max(local_radius, int(local_radius * (1.0 + 4.0 * stale))))
                if local and radius < len(terminals):
                    selected = set(terminals) & best_nodes
                    row = csc_matrix(([(-1.0 if v in selected else 1.0) for v in terminals],
                                      ([0] * len(terminals), terminals)), shape=(1, width))
                    matrices.append(row)
                    lb.append(np.array([-np.inf]))
                    ub.append(np.array([radius - len(selected)], dtype=float))
                else:
                    local = False
                allowance = min(remaining * 0.92, max(quantum, 0.30 * budget) if first else quantum * (1.0 if local else 2.0))
                assembled = vstack(matrices, format="csc")
                result = optimizer(objective, integrality=integrality, bounds=bounds,
                                   constraints=LinearConstraint(assembled, np.concatenate(lb), np.concatenate(ub)),
                                   options={"time_limit": max(0.001, allowance), "mip_rel_gap": 0.0,
                                            "presolve": True})
                first = False
                improved = False
                if result.x is not None:
                    chosen = {arcs[j][2] for j in range(count) if result.x[ar + j] > 0.5}
                    improved = accept(chosen)
                    if time.perf_counter() < deadline:
                        improved = accept(trim(chosen, min(deadline, time.perf_counter() + 0.025 * budget))) or improved
                if result.status == 0 and not local and result.x is not None and best_valid:
                    # The complete model has proven optimality, so further search cannot give a better objective.
                    if result.fun * scale >= best_cost - 1e-7 * max(1.0, abs(best_cost)):
                        return answer
                if local and result.status == 0 and not improved:
                    local_radius = min(len(terminals), max(local_radius + 1, int(1.4 * local_radius)))
                if improved:
                    local_radius = max(2, int(math.sqrt(max(1, len(terminals))) * math.log1p(budget) / 3.0))
                if result.x is None and time.perf_counter() < deadline:
                    random_tree(min(deadline, time.perf_counter() + 0.04 * budget))
        # Without the optimisation library, complete feasible trees are still generated and improved.
        while time.perf_counter() < deadline:
            random_tree(min(deadline, time.perf_counter() + max(0.005, 0.05 * budget)))
        return answer
    except BaseException:
        return answer
