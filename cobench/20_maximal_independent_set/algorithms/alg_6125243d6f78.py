# Library slot 2/10  id=alg_6125243d6f78
# score=-164.4  feasible_rate=1.0
# score_vec=[-369.0, -128.0, -100.0, -80.0, -112.0, -55.0, -118.0, -99.0, -443.0, -140.0]

import heapq
import random
import time
from collections import deque


def solve(instance: dict, time_limit_s: float) -> dict:
    """Anytime maximum-independent-set solver using only the standard library."""
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
    best_vector = [0] * n
    best_size = 0
    if n == 0:
        return {"vertex_selection": best_vector}

    try:
        # Parse defensively.  Self-loops make their endpoint unavailable under
        # the supplied checker; duplicate edges are harmlessly coalesced.
        original_adj = [set() for _ in range(n)]
        banned = set()
        for edge in instance.get("edges", []):
            if time.time() >= deadline:
                return {"vertex_selection": best_vector}
            if not isinstance(edge, (list, tuple)) or len(edge) < 2:
                continue
            try:
                u, v = int(edge[0]), int(edge[1])
            except Exception:
                continue
            if not (0 <= u < n and 0 <= v < n):
                continue
            if u == v:
                banned.add(u)
            else:
                original_adj[u].add(v)
                original_adj[v].add(u)

        # A complete feasible incumbent is established before any expensive
        # work.  Static smallest-degree order is deliberately cheap.
        baseline = set()
        order = sorted(
            (v for v in range(n) if v not in banned),
            key=lambda v: (len(original_adj[v]), v),
        )
        for v in order:
            if not (original_adj[v] & baseline):
                baseline.add(v)
        for v in baseline:
            best_vector[v] = 1
        best_size = len(baseline)
        if time.time() >= deadline:
            return {"vertex_selection": best_vector}

        # The kernel graph can acquire synthetic vertices through exact degree
        # two folding.  Every reduction is recorded for reverse reconstruction.
        adj = [set(neighbors) for neighbors in original_adj]
        active = [v not in banned for v in range(n)]
        degree = [len(adj[v]) if active[v] else 0 for v in range(n)]
        reductions = []
        heap = [(degree[v], v) for v in range(n) if active[v]]
        heapq.heapify(heap)

        def remove_vertex(v):
            if v >= len(active) or not active[v]:
                return
            active[v] = False
            for u in adj[v]:
                if active[u]:
                    degree[u] -= 1
                    heapq.heappush(heap, (degree[u], u))

        # Degree 0/1 forcing and both degree-2 cases are exact.  If the two
        # neighbors of a degree-2 vertex are adjacent, that vertex can be
        # forced.  Otherwise the three vertices are folded into one.
        while heap and time.time() < deadline:
            d, v = heapq.heappop(heap)
            if not active[v] or d != degree[v]:
                continue
            if d > 2:
                break
            neighbors = [u for u in adj[v] if active[u]]
            if d == 0:
                reductions.append((0, v))
                remove_vertex(v)
            elif d == 1:
                u = neighbors[0]
                reductions.append((0, v))
                remove_vertex(v)
                remove_vertex(u)
            else:
                u, w = neighbors[0], neighbors[1]
                if w in adj[u]:
                    reductions.append((0, v))
                    remove_vertex(v)
                    remove_vertex(u)
                    remove_vertex(w)
                else:
                    new_neighbors = {
                        z
                        for z in (adj[u] | adj[w])
                        if active[z] and z != v and z != u and z != w
                    }
                    x = len(adj)
                    reductions.append((1, x, v, u, w))
                    remove_vertex(v)
                    remove_vertex(u)
                    remove_vertex(w)
                    adj.append(set(new_neighbors))
                    active.append(True)
                    degree.append(len(new_neighbors))
                    for z in new_neighbors:
                        adj[z].add(x)
                        degree[z] += 1
                        heapq.heappush(heap, (degree[z], z))
                    heapq.heappush(heap, (degree[x], x))

        def reconstruct(reduced_selected):
            chosen = set(reduced_selected)
            for reduction in reversed(reductions):
                if reduction[0] == 0:
                    chosen.add(reduction[1])
                else:
                    _, x, v, u, w = reduction
                    if x in chosen:
                        chosen.remove(x)
                        chosen.add(u)
                        chosen.add(w)
                    else:
                        chosen.add(v)
            return {v for v in chosen if 0 <= v < n and v not in banned}

        def commit(reduced_selected):
            """Validate/repair a candidate and greedily make it maximal."""
            nonlocal best_vector, best_size
            proposed = reconstruct(reduced_selected)
            safe = set()
            for v in sorted(proposed, key=lambda z: (len(original_adj[z]), z)):
                if not (original_adj[v] & safe):
                    safe.add(v)
            for v in order:
                if v not in safe and not (original_adj[v] & safe):
                    safe.add(v)
            if len(safe) > best_size:
                vector = [0] * n
                for v in safe:
                    vector[v] = 1
                best_vector = vector
                best_size = len(safe)

        # Even an interrupted kernelization immediately yields another full
        # incumbent after reconstruction.
        quick_kernel = set()
        for v in sorted(
            (z for z in range(len(active)) if active[z]),
            key=lambda z: (degree[z], z),
        ):
            if not any(u in quick_kernel for u in adj[v] if active[u]):
                quick_kernel.add(v)
        commit(quick_kernel)
        if time.time() >= deadline:
            return {"vertex_selection": best_vector}

        # Split the kernel.  Bipartite components are solved exactly by
        # Hopcroft-Karp plus Konig's theorem.  The remaining components form
        # the heuristic hard kernel.
        resolved = set()
        hard_components = []
        seen = set()
        color = [-1] * len(active)

        def bipartite_mis(comp):
            comp_set = set(comp)
            left = [v for v in comp if color[v] == 0]
            pair = {}
            dist = {}

            def bfs():
                q = deque()
                found = False
                for u in left:
                    if u not in pair:
                        dist[u] = 0
                        q.append(u)
                    else:
                        dist[u] = -1
                while q and time.time() < deadline:
                    u = q.popleft()
                    for v in adj[u]:
                        if v not in comp_set:
                            continue
                        mate = pair.get(v, -1)
                        if mate < 0:
                            found = True
                        elif dist.get(mate, -1) < 0:
                            dist[mate] = dist[u] + 1
                            q.append(mate)
                return found

            def dfs(u):
                if time.time() >= deadline:
                    return False
                for v in adj[u]:
                    if v not in comp_set:
                        continue
                    mate = pair.get(v, -1)
                    if mate < 0 or (
                        dist.get(mate, -1) == dist[u] + 1 and dfs(mate)
                    ):
                        pair[u] = v
                        pair[v] = u
                        return True
                dist[u] = -1
                return False

            complete = False
            while time.time() < deadline:
                if not bfs():
                    complete = True
                    break
                for u in left:
                    if u not in pair:
                        dfs(u)
            if not complete:
                left_set = set(left)
                right_set = comp_set - left_set
                return left_set if len(left_set) >= len(right_set) else right_set

            z_left = set()
            z_right = set()
            q = deque()
            for u in left:
                if u not in pair:
                    z_left.add(u)
                    q.append((u, 0))
            while q:
                v, side = q.popleft()
                if side == 0:
                    for u in adj[v]:
                        if u in comp_set and pair.get(v, -1) != u and u not in z_right:
                            z_right.add(u)
                            q.append((u, 1))
                else:
                    u = pair.get(v, -1)
                    if u >= 0 and u not in z_left:
                        z_left.add(u)
                        q.append((u, 0))
            answer = set(z_left)
            answer.update(v for v in comp if color[v] == 1 and v not in z_right)
            return answer

        for root in range(len(active)):
            if time.time() >= deadline:
                break
            if not active[root] or root in seen:
                continue
            q = deque([root])
            seen.add(root)
            color[root] = 0
            comp = []
            is_bipartite = True
            while q and time.time() < deadline:
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
                        is_bipartite = False
            if q:
                hard_components.append(comp + list(q))
            elif is_bipartite:
                try:
                    resolved.update(bipartite_mis(comp))
                except Exception:
                    side0 = {v for v in comp if color[v] == 0}
                    side1 = set(comp) - side0
                    resolved.update(side0 if len(side0) >= len(side1) else side1)
            else:
                hard_components.append(comp)

        # Any vertices not visited because of the deadline remain safely in the
        # hard kernel.  Normally this branch is only relevant to tiny budgets.
        hard_vertices = []
        for comp in hard_components:
            hard_vertices.extend(comp)
        hard_seen = set(hard_vertices)
        for v in range(len(active)):
            if active[v] and v not in resolved and v not in hard_seen:
                hard_vertices.append(v)
                hard_seen.add(v)

        if not hard_vertices:
            commit(resolved)
            while time.time() < deadline:
                # Honor the anytime contract without changing an exact answer.
                pass
            return {"vertex_selection": best_vector}

        # Compact bitsets make adjacency tests and exchange neighborhoods fast.
        vertices = hard_vertices
        index = {v: i for i, v in enumerate(vertices)}
        k = len(vertices)
        graph = [0] * k
        for i, v in enumerate(vertices):
            mask = 0
            for u in adj[v]:
                j = index.get(u)
                if j is not None:
                    mask |= 1 << j
            graph[i] = mask
        all_mask = (1 << k) - 1
        seed = (
            n * 1000003
            + sum(len(original_adj[v]) for v in range(n)) * 9176
            + len(banned) * 131
        ) & ((1 << 63) - 1)
        rng = random.Random(seed)

        def greedy_mask(allowed, noise):
            available = allowed
            selected = 0
            deg = [0] * k
            version = [0] * k
            hp = []
            bits = available
            while bits:
                b = bits & -bits
                v = b.bit_length() - 1
                d = (graph[v] & available).bit_count()
                deg[v] = d
                tie = rng.random() * noise if noise else v * 1e-12
                heapq.heappush(hp, (d + tie, version[v], v))
                bits ^= b
            while available and hp and time.time() < deadline:
                _, ver, v = heapq.heappop(hp)
                bit_v = 1 << v
                if not (available & bit_v) or ver != version[v]:
                    continue
                selected |= bit_v
                removed = available & (graph[v] | bit_v)
                available &= ~removed
                scan = removed
                while scan:
                    b = scan & -scan
                    u = b.bit_length() - 1
                    affected = graph[u] & available
                    while affected:
                        c = affected & -affected
                        w = c.bit_length() - 1
                        deg[w] -= 1
                        version[w] += 1
                        tie = rng.random() * noise if noise else w * 1e-12
                        heapq.heappush(hp, (deg[w] + tie, version[w], w))
                        affected ^= c
                    scan ^= b
            return selected

        def maximalize(selected, noise):
            blocked = selected
            scan = selected
            while scan:
                b = scan & -scan
                v = b.bit_length() - 1
                blocked |= graph[v]
                scan ^= b
            free = all_mask & ~blocked
            if free and time.time() < deadline:
                selected |= greedy_mask(free, noise)
            return selected

        def one_two_swap(selected):
            buckets = [0] * k
            outside = all_mask & ~selected
            scan = outside
            while scan and time.time() < deadline:
                b = scan & -scan
                v = b.bit_length() - 1
                blockers = graph[v] & selected
                if blockers and not (blockers & (blockers - 1)):
                    s = blockers.bit_length() - 1
                    buckets[s] |= b
                scan ^= b
            selected_scan = selected
            while selected_scan and time.time() < deadline:
                sb = selected_scan & -selected_scan
                s = sb.bit_length() - 1
                candidates = buckets[s]
                left = candidates
                while left and time.time() < deadline:
                    vb = left & -left
                    v = vb.bit_length() - 1
                    left ^= vb
                    partners = left & ~graph[v]
                    if partners:
                        wb = partners & -partners
                        return (selected ^ sb) | vb | wb
                selected_scan ^= sb
            return selected

        def two_three_swap(selected, phase_end):
            ones = [0] * k
            twos = {}
            outside = all_mask & ~selected
            scan = outside
            while scan and time.time() < phase_end:
                b = scan & -scan
                v = b.bit_length() - 1
                blockers = graph[v] & selected
                count = blockers.bit_count()
                if count == 1:
                    ones[blockers.bit_length() - 1] |= b
                elif count == 2:
                    first = blockers & -blockers
                    second = blockers ^ first
                    key = (first.bit_length() - 1, second.bit_length() - 1)
                    twos[key] = twos.get(key, 0) | b
                scan ^= b

            pairs = list(twos)
            nonempty = [i for i, mask in enumerate(ones) if mask]
            a_pos = 0
            b_pos = 1
            known = set(pairs)
            while time.time() < phase_end:
                if a_pos >= len(nonempty):
                    break
                if b_pos >= len(nonempty):
                    a_pos += 1
                    b_pos = a_pos + 1
                    continue
                key = (nonempty[a_pos], nonempty[b_pos])
                b_pos += 1
                if key not in known:
                    known.add(key)
                    pairs.append(key)

            for s, t in pairs:
                if time.time() >= phase_end:
                    break
                candidates = ones[s] | ones[t] | twos.get((s, t), 0)
                left = candidates
                while left and time.time() < phase_end:
                    ab = left & -left
                    a = ab.bit_length() - 1
                    left ^= ab
                    second_pool = left & ~graph[a]
                    middle = second_pool
                    while middle and time.time() < phase_end:
                        bb = middle & -middle
                        b = bb.bit_length() - 1
                        middle ^= bb
                        third = middle & ~graph[b]
                        if third:
                            cb = third & -third
                            return (
                                selected
                                & ~(1 << s)
                                & ~(1 << t)
                            ) | ab | bb | cb
            return selected

        def local_search(selected):
            selected = maximalize(selected, 0.15)
            while time.time() < deadline:
                improved = one_two_swap(selected)
                if improved == selected:
                    phase_end = min(deadline, time.time() + max(0.002, limit * 0.004))
                    improved = two_three_swap(selected, phase_end)
                if improved == selected:
                    break
                selected = maximalize(improved, 0.1)
            return selected

        def commit_mask(mask):
            candidate = set(resolved)
            scan = mask
            while scan:
                b = scan & -scan
                candidate.add(vertices[b.bit_length() - 1])
                scan ^= b
            commit(candidate)

        current = local_search(greedy_mask(all_mask, 0.0))
        hard_best = current
        hard_best_size = current.bit_count()
        commit_mask(hard_best)

        # ARW-style iterated local search.  Kicks eject selected vertices, then
        # randomized min-degree repair and 1->2 / 2->3 exchanges rebuild the
        # solution.  The outer loop is governed solely by the wall clock.
        stagnation = 0
        while time.time() < deadline:
            selected_list = []
            scan = current
            while scan:
                b = scan & -scan
                selected_list.append(b.bit_length() - 1)
                scan ^= b
            if not selected_list:
                current = local_search(greedy_mask(all_mask, 1.2))
                continue

            max_kick = max(2, min(14, len(selected_list) // 8 + 2))
            kick_size = 2 + (stagnation % (max_kick - 1))
            if kick_size > len(selected_list):
                kick_size = len(selected_list)
            removed_vertices = rng.sample(selected_list, kick_size)
            retained = current
            for v in removed_vertices:
                retained &= ~(1 << v)

            blocked = retained
            scan = retained
            while scan:
                b = scan & -scan
                blocked |= graph[b.bit_length() - 1]
                scan ^= b
            eligible = all_mask & ~blocked
            trial = retained | greedy_mask(eligible, 0.6 + 0.15 * (stagnation % 7))
            trial = local_search(trial)
            trial_size = trial.bit_count()

            if trial_size > hard_best_size:
                hard_best = trial
                hard_best_size = trial_size
                current = trial
                stagnation = 0
                commit_mask(hard_best)
            else:
                stagnation += 1
                # Equal plateaus are valuable; rare one-vertex drops let the
                # walk leave deep local optima while the incumbent stays safe.
                if trial_size >= current.bit_count() or (
                    trial_size + 1 == current.bit_count() and rng.random() < 0.08
                ):
                    current = trial
                if stagnation % 9 == 0 and time.time() < deadline:
                    restart = local_search(
                        greedy_mask(all_mask, 0.4 + rng.random() * 1.8)
                    )
                    if restart.bit_count() >= current.bit_count() - 1:
                        current = restart
                    if restart.bit_count() > hard_best_size:
                        hard_best = restart
                        hard_best_size = restart.bit_count()
                        commit_mask(hard_best)

        return {"vertex_selection": best_vector}
    except Exception:
        # best_vector is initialized before all fallible work and is updated
        # only with independently repaired feasible sets.
        return {"vertex_selection": best_vector}
