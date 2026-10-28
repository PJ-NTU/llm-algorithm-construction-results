# Library slot 3/10  id=alg_2a00b5d3033a
# score=-164.2  feasible_rate=1.0
# score_vec=[-369.0, -128.0, -100.0, -80.0, -113.0, -55.0, -117.0, -98.0, -443.0, -139.0]

import heapq
import random
import time
from collections import deque


def solve(instance: dict, time_limit_s: float) -> dict:
    t0 = time.time()
    try:
        limit = max(0.0, float(time_limit_s))
    except Exception:
        limit = 0.0
    deadline = t0 + limit * 0.85

    try:
        n = max(0, int(instance.get("num_vertices", 0)))
    except Exception:
        n = 0
    fallback = {"vertex_selection": [0] * n}
    if n == 0 or time.time() >= deadline:
        return fallback

    try:
        # A self-loop makes its endpoint impossible to select under the checker.
        banned = set()
        raw_edges = []
        for edge in instance.get("edges", []):
            if len(edge) < 2:
                continue
            u, v = int(edge[0]), int(edge[1])
            if not (0 <= u < n and 0 <= v < n):
                continue
            if u == v:
                banned.add(u)
            else:
                raw_edges.append((u, v))

        adj = [set() for _ in range(n)]
        for u, v in raw_edges:
            if u not in banned and v not in banned:
                adj[u].add(v)
                adj[v].add(u)

        # Degree-zero and degree-one reductions are exact for maximum
        # independent set: some optimum always contains the leaf.
        active = [v not in banned for v in range(n)]
        degree = [len(adj[v]) if active[v] else 0 for v in range(n)]
        heap = [(degree[v], v) for v in range(n) if active[v]]
        heapq.heapify(heap)
        forced = set()

        def remove_active(v):
            if not active[v]:
                return
            active[v] = False
            for u in adj[v]:
                if active[u]:
                    degree[u] -= 1
                    heapq.heappush(heap, (degree[u], u))

        while heap and time.time() < deadline:
            d, v = heapq.heappop(heap)
            if not active[v] or d != degree[v]:
                continue
            if d > 1:
                break
            if d == 0:
                forced.add(v)
                remove_active(v)
            else:
                u = -1
                for w in adj[v]:
                    if active[w]:
                        u = w
                        break
                forced.add(v)
                remove_active(v)
                if u >= 0:
                    remove_active(u)

        if time.time() >= deadline:
            out = [0] * n
            for v in forced:
                out[v] = 1
            return {"vertex_selection": out}

        # Split the residual graph. Bipartite components are solved exactly
        # through Konig's theorem; only genuinely hard kernels remain.
        components = []
        color = [-1] * n
        bipartite = []
        seen = set()
        for root in range(n):
            if not active[root] or root in seen:
                continue
            q = deque([root])
            seen.add(root)
            color[root] = 0
            comp = []
            is_bip = True
            while q:
                v = q.popleft()
                comp.append(v)
                for u in adj[v]:
                    if not active[u]:
                        continue
                    if u not in seen:
                        seen.add(u)
                        color[u] = color[v] ^ 1
                        q.append(u)
                    elif color[u] == color[v]:
                        is_bip = False
            if is_bip:
                bipartite.append(comp)
            else:
                components.append(comp)

        exact_fixed = set(forced)

        def bipartite_mis(comp):
            comp_set = set(comp)
            left = [v for v in comp if color[v] == 0]
            right = [v for v in comp if color[v] == 1]
            pair = [-1] * n
            dist = [-1] * n

            def bfs():
                q = deque()
                found = False
                for u in left:
                    if pair[u] < 0:
                        dist[u] = 0
                        q.append(u)
                    else:
                        dist[u] = -1
                while q:
                    u = q.popleft()
                    for v in adj[u]:
                        if v not in comp_set:
                            continue
                        mate = pair[v]
                        if mate < 0:
                            found = True
                        elif dist[mate] < 0:
                            dist[mate] = dist[u] + 1
                            q.append(mate)
                return found

            def dfs(u):
                for v in adj[u]:
                    if v not in comp_set:
                        continue
                    mate = pair[v]
                    if mate < 0 or (dist[mate] == dist[u] + 1 and dfs(mate)):
                        pair[u] = v
                        pair[v] = u
                        return True
                dist[u] = -1
                return False

            matching_complete = False
            while time.time() < deadline:
                if not bfs():
                    matching_complete = True
                    break
                for u in left:
                    if pair[u] < 0:
                        dfs(u)

            if not matching_complete:
                return set(left if len(left) >= len(right) else right)

            # Vertices reachable from unmatched left vertices by alternating
            # paths determine a minimum vertex cover and hence its complement.
            z_left = set()
            z_right = set()
            q = deque()
            for u in left:
                if pair[u] < 0:
                    z_left.add(u)
                    q.append((u, 0))
            while q:
                v, side = q.popleft()
                if side == 0:
                    for u in adj[v]:
                        if u not in comp_set:
                            continue
                        if pair[v] != u and u not in z_right:
                            z_right.add(u)
                            q.append((u, 1))
                else:
                    u = pair[v]
                    if u >= 0 and u not in z_left:
                        z_left.add(u)
                        q.append((u, 0))
            answer = set(z_left)
            for v in comp:
                if color[v] == 1 and v not in z_right:
                    answer.add(v)
            return answer

        for comp in bipartite:
            if time.time() >= deadline:
                break
            try:
                exact_fixed.update(bipartite_mis(comp))
            except Exception:
                left = {v for v in comp if color[v] == 0}
                right = {v for v in comp if color[v] == 1}
                exact_fixed.update(left if len(left) >= len(right) else right)

        hard_vertices = []
        for comp in components:
            hard_vertices.extend(comp)
        hard_set = set(hard_vertices)
        if not hard_vertices:
            out = [0] * n
            for v in exact_fixed:
                out[v] = 1
            return {"vertex_selection": out}

        seed_value = (n * 1000003 + len(raw_edges) * 9176 + len(banned) * 37) & 0xFFFFFFFF
        rng = random.Random(seed_value)
        static_neighbor_weight = [0] * n
        for v in hard_vertices:
            total = 0
            for u in adj[v]:
                if u in hard_set:
                    total += len(adj[u])
            static_neighbor_weight[v] = total

        def greedy(vertices, width):
            available = set(vertices)
            deg = {}
            h = []
            for v in available:
                d = 0
                for u in adj[v]:
                    if u in available:
                        d += 1
                deg[v] = d
                heapq.heappush(h, (d, -static_neighbor_weight[v], v))
            chosen = set()
            while available and time.time() < deadline:
                candidates = []
                candidate_seen = set()
                while h and len(candidates) < width:
                    d, neg_weight, v = heapq.heappop(h)
                    if v not in available or v in candidate_seen or d != deg[v]:
                        continue
                    candidates.append((d, neg_weight, v))
                    candidate_seen.add(v)
                if not candidates:
                    break
                pick_at = 0 if width == 1 else rng.randrange(len(candidates))
                v = candidates[pick_at][2]
                for item in candidates:
                    if item[2] != v:
                        heapq.heappush(h, item)
                chosen.add(v)
                removed = [v]
                for u in adj[v]:
                    if u in available:
                        removed.append(u)
                for u in removed:
                    if u not in available:
                        continue
                    available.remove(u)
                    for w in adj[u]:
                        if w in available:
                            deg[w] -= 1
                            heapq.heappush(h, (deg[w], -static_neighbor_weight[w], w))
            return chosen

        def maximalize(selected, vertices, width):
            candidates = []
            for v in vertices:
                if v in selected:
                    continue
                free = True
                for u in adj[v]:
                    if u in selected:
                        free = False
                        break
                if free:
                    candidates.append(v)
            if candidates and time.time() < deadline:
                selected = set(selected)
                selected.update(greedy(candidates, width))
            return selected

        def improve_exchange(selected, vertices, local_deadline):
            selected = maximalize(selected, vertices, 1)
            while time.time() < local_deadline:
                one = {}
                two = {}
                for v in vertices:
                    if v in selected:
                        continue
                    conflicts = []
                    for u in adj[v]:
                        if u in selected:
                            conflicts.append(u)
                            if len(conflicts) > 2:
                                break
                    if len(conflicts) == 1:
                        one.setdefault(conflicts[0], []).append(v)
                    elif len(conflicts) == 2:
                        a, b = conflicts
                        if a > b:
                            a, b = b, a
                        two.setdefault((a, b), []).append(v)

                blockers = list(one)
                rng.shuffle(blockers)
                moved = False
                for blocker in blockers:
                    cand = one[blocker]
                    cand.sort(key=lambda v: (len(adj[v]), v))
                    for i, a in enumerate(cand):
                        if time.time() >= local_deadline:
                            return selected
                        for b in cand[i + 1:]:
                            if b not in adj[a]:
                                selected.remove(blocker)
                                selected.add(a)
                                selected.add(b)
                                selected = maximalize(selected, vertices, 1)
                                moved = True
                                break
                        if moved:
                            break
                    if moved:
                        break
                if moved:
                    continue

                # A 2-for-3 exchange reaches improvements hidden behind a
                # locally optimal 1-for-2 neighborhood.
                pairs = list(two)
                rng.shuffle(pairs)
                for a, b in pairs:
                    if time.time() >= local_deadline:
                        return selected
                    pool = set(two[(a, b)])
                    pool.update(one.get(a, []))
                    pool.update(one.get(b, []))
                    cand = sorted(pool, key=lambda v: (len(adj[v]), v))
                    for i, x in enumerate(cand):
                        compatible = []
                        for y in cand[i + 1:]:
                            if y not in adj[x]:
                                compatible.append(y)
                        for j, y in enumerate(compatible):
                            if time.time() >= local_deadline:
                                return selected
                            for z in compatible[j + 1:]:
                                if z not in adj[y]:
                                    selected.remove(a)
                                    selected.remove(b)
                                    selected.add(x)
                                    selected.add(y)
                                    selected.add(z)
                                    selected = maximalize(selected, vertices, 1)
                                    moved = True
                                    break
                            if moved:
                                break
                        if moved:
                            break
                    if moved:
                        break
                if not moved:
                    return selected
            return selected

        # Secure a complete incumbent immediately, then diversify construction
        # for a short wall-clock slice before exact search.
        best_hard = greedy(hard_vertices, 1)
        seed_deadline = min(deadline, t0 + limit * 0.10)
        while time.time() < seed_deadline:
            width = rng.choice((2, 3, 4, 6, 8, 12))
            candidate = greedy(hard_vertices, width)
            candidate = improve_exchange(candidate, hard_vertices, min(seed_deadline, time.time() + 0.04))
            if len(candidate) > len(best_hard):
                best_hard = candidate

        # Tomita-style maximum-clique search on the complement. It is exact
        # when it finishes and remains an anytime improver when interrupted.
        def exact_component(comp, incumbent, stop):
            # Avoid Python recursion-depth failure on exceptionally large,
            # very sparse kernels; the iterative local search handles these.
            if len(comp) > 1400 or len(incumbent) > 800:
                return set(incumbent), False
            order_vertices = sorted(comp, key=lambda v: (len(adj[v]), v))
            size = len(order_vertices)
            all_bits = (1 << size) - 1
            comp_adj = [0] * size
            position = {v: i for i, v in enumerate(order_vertices)}
            for i, v in enumerate(order_vertices):
                if time.time() >= stop:
                    return set(incumbent), False
                neighbor_bits = 0
                for u in adj[v]:
                    if u in position:
                        neighbor_bits |= 1 << position[u]
                comp_adj[i] = all_bits & ~(neighbor_bits | (1 << i))

            best_local = []
            incumbent_set = set(incumbent)
            for i, v in enumerate(order_vertices):
                if v in incumbent_set:
                    best_local.append(i)
            current = []
            interrupted = [False]

            def color_sort(bits):
                colored_order = []
                bounds = []
                uncolored = bits
                color_number = 0
                while uncolored:
                    if time.time() >= stop:
                        interrupted[0] = True
                        return None, None
                    color_number += 1
                    available = uncolored
                    while available:
                        if time.time() >= stop:
                            interrupted[0] = True
                            return None, None
                        bit = available & -available
                        v = bit.bit_length() - 1
                        uncolored ^= bit
                        available ^= bit
                        available &= ~comp_adj[v]
                        colored_order.append(v)
                        bounds.append(color_number)
                return colored_order, bounds

            def expand(bits):
                if time.time() >= stop:
                    interrupted[0] = True
                    return
                colored_order, bounds = color_sort(bits)
                if interrupted[0]:
                    return
                for i in range(len(colored_order) - 1, -1, -1):
                    if time.time() >= stop:
                        interrupted[0] = True
                        return
                    if len(current) + bounds[i] <= len(best_local):
                        return
                    v = colored_order[i]
                    bit = 1 << v
                    if not bits & bit:
                        continue
                    current.append(v)
                    next_bits = bits & comp_adj[v]
                    if next_bits:
                        expand(next_bits)
                    elif len(current) > len(best_local):
                        best_local[:] = current
                    current.pop()
                    bits &= ~bit
                    if interrupted[0]:
                        return

            expand(all_bits)
            result = {order_vertices[i] for i in best_local}
            return result, not interrupted[0]

        exact_phase_end = min(deadline, t0 + limit * 0.42)
        unresolved = []
        solved_exactly = set()
        components.sort(key=len)
        for index, comp in enumerate(components):
            now = time.time()
            if now >= exact_phase_end:
                for later_comp in components[index:]:
                    unresolved.extend(later_comp)
                break
            comp_set = set(comp)
            incumbent = best_hard.intersection(comp_set)
            remaining_count = len(components) - index
            slice_end = min(exact_phase_end, now + (exact_phase_end - now) / remaining_count)
            result, complete = exact_component(comp, incumbent, slice_end)
            if len(result) > len(incumbent):
                best_hard.difference_update(comp_set)
                best_hard.update(result)
            if complete:
                solved_exactly.update(result)
            else:
                unresolved.extend(comp)
        else:
            unresolved = list(unresolved)

        unresolved_set = set(unresolved)
        if unresolved_set:
            incumbent_dynamic = best_hard.intersection(unresolved_set)
            best_dynamic = set(incumbent_dynamic)
            # Repeated perturb-and-repair: the outer loop has no iteration cap
            # and turns every remaining millisecond into another neighborhood.
            while time.time() < deadline:
                local_stop = min(deadline, time.time() + max(0.025, limit * 0.022))
                if rng.random() < 0.28:
                    candidate = greedy(unresolved, rng.choice((2, 3, 4, 6, 8, 12)))
                else:
                    candidate = set(best_dynamic)
                    if candidate:
                        strength = 2 + rng.randrange(max(2, min(10, len(candidate) // 8 + 2)))
                        removed = rng.sample(tuple(candidate), min(strength, len(candidate)))
                        for v in removed:
                            candidate.remove(v)
                        # Occasional neutral 1-for-1 moves diversify which
                        # exchange pairs are exposed after maximalization.
                        one_tight = []
                        for v in unresolved:
                            if v in candidate:
                                continue
                            conflicts = [u for u in adj[v] if u in candidate]
                            if len(conflicts) == 1:
                                one_tight.append((v, conflicts[0]))
                        if one_tight and rng.random() < 0.5:
                            v, blocker = rng.choice(one_tight)
                            candidate.remove(blocker)
                            candidate.add(v)
                        candidate = maximalize(candidate, unresolved, rng.choice((1, 2, 3, 4)))
                candidate = improve_exchange(candidate, unresolved, local_stop)
                if len(candidate) > len(best_dynamic):
                    best_dynamic = candidate
            best_hard.difference_update(unresolved_set)
            best_hard.update(best_dynamic)

        answer = set(exact_fixed)
        answer.update(best_hard)
        answer.update(solved_exactly)
        out = [0] * n
        for v in answer:
            out[v] = 1
        return {"vertex_selection": out}
    except Exception:
        return fallback
