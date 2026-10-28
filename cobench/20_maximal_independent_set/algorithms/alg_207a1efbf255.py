# Library slot 10/10  id=alg_207a1efbf255
# score=-163.5  feasible_rate=1.0
# score_vec=[-369.0, -128.0, -100.0, -80.0, -111.0, -55.0, -116.0, -98.0, -440.0, -138.0]

import heapq
import random
import time


def solve(instance: dict, time_limit_s: float) -> dict:
    """Anytime maximum-independent-set heuristic using only the standard library."""
    t0 = time.time()
    try:
        limit = max(0.0, float(time_limit_s))
    except Exception:
        limit = 0.0
    deadline = t0 + limit * 0.85

    try:
        n = int(instance.get("num_vertices", 0))
    except Exception:
        n = 0
    if n <= 0:
        return {"vertex_selection": []}

    # An all-zero vector is the always-available emergency incumbent.
    best_mask = 0

    try:
        adj_sets = [set() for _ in range(n)]
        banned_mask = 0
        raw_edges = instance.get("edges", [])
        for edge in raw_edges:
            if len(edge) < 2:
                continue
            u, v = edge[0], edge[1]
            if not isinstance(u, int) or not isinstance(v, int):
                continue
            if u < 0 or u >= n or v < 0 or v >= n:
                continue
            if u == v:
                # The supplied checker makes a self-looped vertex unselectable.
                banned_mask |= 1 << u
            else:
                adj_sets[u].add(v)
                adj_sets[v].add(u)

        adj = [0] * n
        for v in range(n):
            mask = 0
            for u in adj_sets[v]:
                mask |= 1 << u
            adj[v] = mask

        usable_mask = ((1 << n) - 1) & ~banned_mask

        # Exact degree-0/1 reductions.  Every reduction has an optimum that
        # contains v, so the choices can safely be fixed before local search.
        active_mask = usable_mask
        degree = [(adj[v] & active_mask).bit_count() for v in range(n)]
        heap = [(degree[v], v) for v in range(n) if (active_mask >> v) & 1]
        heapq.heapify(heap)
        forced_mask = 0

        while heap and time.time() < deadline:
            d, v = heapq.heappop(heap)
            vbit = 1 << v
            if not (active_mask & vbit) or d != degree[v]:
                continue
            if d > 12:
                break

            neighbors = adj[v] & active_mask
            simplicial = True
            check = neighbors
            while check:
                ubit = check & -check
                check ^= ubit
                u = ubit.bit_length() - 1
                if (neighbors ^ ubit) & ~adj[u]:
                    simplicial = False
                    break
            if not simplicial:
                continue

            forced_mask |= vbit
            remove_mask = vbit | neighbors

            active_mask &= ~remove_mask
            removed = remove_mask
            while removed:
                wbit = removed & -removed
                removed ^= wbit
                w = wbit.bit_length() - 1
                affected = adj[w] & active_mask
                while affected:
                    zbit = affected & -affected
                    affected ^= zbit
                    z = zbit.bit_length() - 1
                    degree[z] -= 1
                    heapq.heappush(heap, (degree[z], z))

        best_mask = forced_mask
        kernel_mask = active_mask
        if not kernel_mask or time.time() >= deadline:
            return {
                "vertex_selection": [1 if (best_mask >> v) & 1 else 0 for v in range(n)]
            }

        kernel_vertices = []
        bits = kernel_mask
        while bits:
            bit = bits & -bits
            bits ^= bit
            kernel_vertices.append(bit.bit_length() - 1)

        kernel_degree = [(adj[v] & kernel_mask).bit_count() for v in range(n)]
        neighbor_pressure = [0] * n
        for v in kernel_vertices:
            total = 0
            nbrs = adj[v] & kernel_mask
            while nbrs:
                bit = nbrs & -nbrs
                nbrs ^= bit
                total += kernel_degree[bit.bit_length() - 1]
            neighbor_pressure[v] = total

        edge_count = sum(kernel_degree[v] for v in kernel_vertices) // 2
        seed = (n * 1000003 + edge_count * 9176 + len(raw_edges) * 97) & 0xFFFFFFFF
        rng = random.Random(seed)

        def choose_random_bit(mask):
            """Uniformly choose one set bit without materializing all vertices."""
            target = rng.randrange(mask.bit_count())
            while target:
                mask &= mask - 1
                target -= 1
            return mask & -mask

        def available_after(selected):
            available = kernel_mask & ~selected
            scan = selected & kernel_mask
            while scan:
                bit = scan & -scan
                scan ^= bit
                available &= ~adj[bit.bit_length() - 1]
            return available

        def greedy_extend(selected, noise):
            """Adaptive minimum residual-degree completion of a partial set."""
            available = available_after(selected)
            while available and time.time() < deadline:
                scan = available
                chosen_bit = scan & -scan
                chosen_key = float("inf")
                while scan:
                    bit = scan & -scan
                    scan ^= bit
                    v = bit.bit_length() - 1
                    d = (adj[v] & available).bit_count()
                    # Residual degree dominates.  Pressure breaks ties toward
                    # deleting mutually redundant, high-degree neighbors.
                    pressure = neighbor_pressure[v] / max(1, kernel_degree[v])
                    key = d + noise * rng.random() * (1.0 + 0.20 * d)
                    key -= min(0.20, pressure * 0.002)
                    if key < chosen_key:
                        chosen_key = key
                        chosen_bit = bit
                v = chosen_bit.bit_length() - 1
                selected |= chosen_bit
                available &= ~chosen_bit
                available &= ~adj[v]
            return selected

        def find_pair(candidate_mask):
            """Return two nonadjacent candidate bits, or zero."""
            scan = candidate_mask
            while scan and time.time() < deadline:
                first = scan & -scan
                scan ^= first
                v = first.bit_length() - 1
                mates = scan & ~adj[v]
                if mates:
                    return first | (mates & -mates)
            return 0

        def find_triple(candidate_mask):
            """Return three pairwise nonadjacent candidate bits, or zero."""
            outer = candidate_mask
            while outer and time.time() < deadline:
                first = outer & -outer
                outer ^= first
                u = first.bit_length() - 1
                seconds = outer & ~adj[u]
                scan = seconds
                while scan and time.time() < deadline:
                    second = scan & -scan
                    scan ^= second
                    v = second.bit_length() - 1
                    thirds = scan & ~adj[v]
                    if thirds:
                        return first | second | (thirds & -thirds)
            return 0

        def improve(selected):
            """Climb through complete (1,2), then (2,3), exchanges."""
            selected = greedy_extend(selected, 0.04)
            while time.time() < deadline:
                one_groups = {}
                two_groups = {}
                outside = kernel_mask & ~selected
                zero = 0
                scan = outside
                while scan and time.time() < deadline:
                    bit = scan & -scan
                    scan ^= bit
                    v = bit.bit_length() - 1
                    conflicts = adj[v] & selected
                    count = conflicts.bit_count()
                    if count == 0:
                        zero |= bit
                    elif count == 1:
                        blocker = conflicts.bit_length() - 1
                        one_groups[blocker] = one_groups.get(blocker, 0) | bit
                    elif count == 2:
                        first = conflicts & -conflicts
                        second = conflicts ^ first
                        key = (first.bit_length() - 1, second.bit_length() - 1)
                        two_groups[key] = two_groups.get(key, 0) | bit

                if zero:
                    selected = greedy_extend(selected | (zero & -zero), 0.02)
                    continue

                blockers = list(one_groups)
                rng.shuffle(blockers)
                move = 0
                removed = 0
                for blocker in blockers:
                    move = find_pair(one_groups[blocker])
                    if move:
                        removed = 1 << blocker
                        break
                if move:
                    selected = greedy_extend((selected & ~removed) | move, 0.03)
                    continue

                # A 2-for-3 exchange is the smallest useful neighborhood that
                # none of the incumbent algorithms searches completely.
                pairs = list(two_groups)
                rng.shuffle(pairs)
                for a, b in pairs:
                    candidates = two_groups[(a, b)]
                    candidates |= one_groups.get(a, 0)
                    candidates |= one_groups.get(b, 0)
                    move = find_triple(candidates)
                    if move:
                        removed = (1 << a) | (1 << b)
                        break
                if move:
                    selected = greedy_extend((selected & ~removed) | move, 0.03)
                    continue
                break
            return selected

        def perturb(selected, strength):
            """ARW-style plateau/ejection moves followed by greedy repair."""
            result = selected
            for _ in range(strength):
                if time.time() >= deadline:
                    break
                outside = kernel_mask & ~result
                if not outside:
                    break

                # Prefer low-tightness vertices, but randomize within the best
                # observed class so repeated kicks explore different plateaus.
                scan = outside
                best_tightness = n + 1
                choices = 0
                while scan:
                    bit = scan & -scan
                    scan ^= bit
                    v = bit.bit_length() - 1
                    tightness = (adj[v] & result).bit_count()
                    if tightness < best_tightness:
                        best_tightness = tightness
                        choices = bit
                    elif tightness == best_tightness:
                        choices |= bit
                chosen = choose_random_bit(choices)
                v = chosen.bit_length() - 1
                result &= ~adj[v]
                result |= chosen
            return greedy_extend(result, rng.uniform(0.05, 0.40))

        def bipartite_recombine(parent_a, parent_b):
            """Find the exact best independent set inside two parents' union."""
            common = parent_a & parent_b
            left = parent_a & ~parent_b
            right = parent_b & ~parent_a
            if not left or not right:
                return parent_a if parent_a.bit_count() >= parent_b.bit_count() else parent_b

            match_left = [-1] * n
            match_right = [-1] * n
            roots = []
            scan = left
            while scan:
                bit = scan & -scan
                scan ^= bit
                roots.append(bit.bit_length() - 1)
            rng.shuffle(roots)

            # Repeated shortest augmenting paths.  The union is bipartite
            # because each parent is itself independent.
            for root in roots:
                if time.time() >= deadline:
                    return parent_a
                queue = [root]
                qpos = 0
                seen_left = 1 << root
                seen_right = 0
                parent_right = {}
                free_right = -1
                while qpos < len(queue) and free_right < 0 and time.time() < deadline:
                    u = queue[qpos]
                    qpos += 1
                    candidates = adj[u] & right & ~seen_right
                    if match_left[u] >= 0:
                        candidates &= ~(1 << match_left[u])
                    while candidates:
                        bit = candidates & -candidates
                        candidates ^= bit
                        v = bit.bit_length() - 1
                        seen_right |= bit
                        parent_right[v] = u
                        mate = match_right[v]
                        if mate < 0:
                            free_right = v
                            break
                        mate_bit = 1 << mate
                        if not (seen_left & mate_bit):
                            seen_left |= mate_bit
                            queue.append(mate)
                if free_right >= 0:
                    v = free_right
                    while v >= 0:
                        u = parent_right[v]
                        previous = match_left[u]
                        match_left[u] = v
                        match_right[v] = u
                        v = previous

            # Konig: alternating reachability gives a minimum vertex cover;
            # its complement is a maximum independent set of the union.
            visited_left = 0
            visited_right = 0
            queue = []
            for u in roots:
                if match_left[u] < 0:
                    visited_left |= 1 << u
                    queue.append(u)
            qpos = 0
            while qpos < len(queue) and time.time() < deadline:
                u = queue[qpos]
                qpos += 1
                candidates = adj[u] & right & ~visited_right
                if match_left[u] >= 0:
                    candidates &= ~(1 << match_left[u])
                while candidates:
                    bit = candidates & -candidates
                    candidates ^= bit
                    v = bit.bit_length() - 1
                    visited_right |= bit
                    mate = match_right[v]
                    if mate >= 0 and not (visited_left & (1 << mate)):
                        visited_left |= 1 << mate
                        queue.append(mate)

            return common | (left & visited_left) | (right & ~visited_right)

        def add_elite(pool, candidate):
            if candidate not in pool:
                pool.append(candidate)
                pool.sort(key=int.bit_count, reverse=True)
                if len(pool) > 12:
                    pool.pop()

        def exact_finish(incumbent):
            """Interruptible maximum-clique B&B on the complement graph."""
            complement = [0] * n
            for v in kernel_vertices:
                complement[v] = kernel_mask & ~adj[v] & ~(1 << v)

            exact_best = incumbent
            exact_size = incumbent.bit_count()

            def color_sort(candidates):
                order = []
                colors = []
                uncolored = candidates
                color = 0
                while uncolored and time.time() < deadline:
                    color += 1
                    color_class = uncolored
                    while color_class:
                        bit = color_class & -color_class
                        color_class ^= bit
                        uncolored ^= bit
                        v = bit.bit_length() - 1
                        order.append(bit)
                        colors.append(color)
                        color_class &= uncolored
                        color_class &= ~complement[v]
                return order, colors

            def expand(clique, clique_size, candidates):
                nonlocal exact_best, exact_size
                if time.time() >= deadline:
                    return
                if clique_size > exact_size:
                    exact_best = clique
                    exact_size = clique_size
                if not candidates or clique_size + candidates.bit_count() <= exact_size:
                    return

                order, colors = color_sort(candidates)
                for pos in range(len(order) - 1, -1, -1):
                    if time.time() >= deadline:
                        return
                    if clique_size + colors[pos] <= exact_size:
                        return
                    bit = order[pos]
                    if not (candidates & bit):
                        continue
                    v = bit.bit_length() - 1
                    expand(clique | bit, clique_size + 1, candidates & complement[v])
                    candidates &= ~bit

            try:
                expand(0, 0, kernel_mask)
            except RecursionError:
                pass
            return exact_best

        # Fast first incumbent, strengthened before any diversification.
        current = improve(greedy_extend(0, 0.0))
        best_kernel = current
        best_mask = forced_mask | best_kernel
        elites = []
        add_elite(elites, current)
        iteration = 0
        stagnant = 0
        use_exact_finish = len(kernel_vertices) <= 320
        heuristic_deadline = t0 + limit * 0.62 if use_exact_finish else deadline

        # No iteration cap: every spare millisecond becomes a restart, kick,
        # larger neighborhood, or exact elite recombination.
        while time.time() < heuristic_deadline:
            iteration += 1
            if len(elites) >= 2 and iteration % 9 == 0:
                a, b = rng.sample(elites, 2)
                candidate = bipartite_recombine(a, b)
                candidate = improve(candidate)
            elif iteration % 6 == 0:
                candidate = improve(greedy_extend(0, rng.uniform(0.05, 0.55)))
            else:
                source = best_kernel if rng.random() < 0.75 else rng.choice(elites)
                strength = 1 + min(5, stagnant // 8)
                candidate = improve(perturb(source, strength))

            add_elite(elites, candidate)
            size = candidate.bit_count()
            if size > best_kernel.bit_count():
                best_kernel = candidate
                best_mask = forced_mask | best_kernel
                stagnant = 0
            else:
                stagnant += 1

        if use_exact_finish and time.time() < deadline:
            exact_candidate = exact_finish(best_kernel)
            if exact_candidate.bit_count() > best_kernel.bit_count():
                best_kernel = exact_candidate
                best_mask = forced_mask | best_kernel

        return {
            "vertex_selection": [1 if (best_mask >> v) & 1 else 0 for v in range(n)]
        }
    except Exception:
        # best_mask is updated only with feasible independent sets.
        return {"vertex_selection": [1 if (best_mask >> v) & 1 else 0 for v in range(n)]}
