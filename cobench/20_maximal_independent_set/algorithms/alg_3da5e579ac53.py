# Library slot 4/10  id=alg_3da5e579ac53
# score=-164.2  feasible_rate=1.0
# score_vec=[-369.0, -128.0, -100.0, -80.0, -112.0, -55.0, -116.0, -98.0, -446.0, -138.0]

import heapq
import random
import time


def solve(instance: dict, time_limit_s: float) -> dict:
    t0 = time.time()
    try:
        n = int(instance.get("num_vertices", 0))
    except Exception:
        n = 0
    if n < 0:
        n = 0

    best_mask = 0

    def make_answer(mask):
        return {
            "vertex_selection": [1 if (mask >> v) & 1 else 0 for v in range(n)]
        }

    try:
        budget = max(0.0, float(time_limit_s))
        deadline = t0 + budget * 0.85
        if n == 0 or time.time() >= deadline:
            return make_answer(best_mask)

        # Build both set and bitset views.  Invalid edges are ignored exactly as
        # by the supplied checker; a self-loop makes its vertex unselectable.
        adj = [set() for _ in range(n)]
        self_loop = [False] * n
        edges = instance.get("edges", [])
        for edge_no, edge in enumerate(edges):
            if (edge_no & 1023) == 0 and time.time() >= deadline:
                return make_answer(best_mask)
            if not isinstance(edge, (list, tuple)) or len(edge) < 2:
                continue
            u, v = edge[0], edge[1]
            if not isinstance(u, int) or not isinstance(v, int):
                continue
            if u < 0 or u >= n or v < 0 or v >= n:
                continue
            if u == v:
                self_loop[u] = True
            else:
                adj[u].add(v)
                adj[v].add(u)

        adj_bits = [0] * n
        for v in range(n):
            bits = 0
            for u in adj[v]:
                bits |= 1 << u
            adj_bits[v] = bits

        allowed_mask = 0
        for v in range(n):
            if not self_loop[v]:
                allowed_mask |= 1 << v

        # Immediate complete incumbent: static min-degree greedy is deliberately
        # simple and fast, so every later phase has a feasible fallback.
        quick = 0
        quick_order = sorted(
            (v for v in range(n) if not self_loop[v]),
            key=lambda v: (len(adj[v]), v),
        )
        for v in quick_order:
            if time.time() >= deadline:
                break
            if (adj_bits[v] & quick) == 0:
                quick |= 1 << v
        best_mask = quick
        if time.time() >= deadline:
            return make_answer(best_mask)

        # Exact degree-0/1 reductions.  A leaf can always replace its only
        # neighbor in some optimum, hence selecting it is lossless.
        active = [not self_loop[v] for v in range(n)]
        degree = [0] * n
        heap = []
        for v in range(n):
            if active[v]:
                degree[v] = sum(1 for u in adj[v] if active[u])
                heapq.heappush(heap, (degree[v], v))

        forced_mask = 0

        def deactivate(v):
            if not active[v]:
                return
            active[v] = False
            for u in adj[v]:
                if active[u]:
                    degree[u] -= 1
                    heapq.heappush(heap, (degree[u], u))

        reduction_steps = 0
        while heap and time.time() < deadline:
            d, v = heapq.heappop(heap)
            if not active[v] or d != degree[v]:
                continue
            if d > 1:
                break
            forced_mask |= 1 << v
            if d == 0:
                deactivate(v)
            else:
                neighbor = -1
                for u in adj[v]:
                    if active[u]:
                        neighbor = u
                        break
                deactivate(v)
                if neighbor >= 0:
                    deactivate(neighbor)
            reduction_steps += 1
            if (reduction_steps & 255) == 0 and time.time() >= deadline:
                if forced_mask.bit_count() > best_mask.bit_count():
                    best_mask = forced_mask
                return make_answer(best_mask)

        active_mask = 0
        for v in range(n):
            if active[v]:
                active_mask |= 1 << v

        if forced_mask.bit_count() > best_mask.bit_count():
            best_mask = forced_mask
        if active_mask == 0:
            if forced_mask.bit_count() >= best_mask.bit_count():
                best_mask = forced_mask
            return make_answer(best_mask)

        seed_value = (
            n * 1000003
            + len(edges) * 9176
            + sum((v + 1) * len(adj[v]) for v in range(n))
        )
        rng = random.Random(seed_value)

        def construct_from_base(base, pool, alpha, stop_at):
            """Dynamic residual-degree GRASP completion of an independent base."""
            selected = base & active_mask
            candidates = pool & active_mask & ~selected
            selected_scan = selected
            while selected_scan:
                bit = selected_scan & -selected_scan
                v = bit.bit_length() - 1
                candidates &= ~adj_bits[v]
                selected_scan ^= bit

            while candidates and time.time() < stop_at:
                scan = candidates
                min_degree = n + 1
                max_degree = -1
                scored = []
                while scan:
                    bit = scan & -scan
                    v = bit.bit_length() - 1
                    d = (adj_bits[v] & candidates).bit_count()
                    scored.append((v, d))
                    if d < min_degree:
                        min_degree = d
                    if d > max_degree:
                        max_degree = d
                    scan ^= bit
                cutoff = min_degree + alpha * (max_degree - min_degree)
                rcl = [v for v, d in scored if d <= cutoff]
                if len(rcl) == 1:
                    chosen = rcl[0]
                else:
                    # Within the RCL, prefer a vertex that eliminates difficult
                    # high-residual-degree neighbors, with randomness for breadth.
                    sample_size = min(len(rcl), 12)
                    sampled = rng.sample(rcl, sample_size)
                    if rng.random() < 0.72:
                        chosen = max(
                            sampled,
                            key=lambda v: sum(
                                (adj_bits[u] & candidates).bit_count()
                                for u in adj[v]
                                if (candidates >> u) & 1
                            ),
                        )
                    else:
                        chosen = rng.choice(sampled)
                chosen_bit = 1 << chosen
                selected |= chosen_bit
                candidates &= ~chosen_bit
                candidates &= ~adj_bits[chosen]
            return selected

        def local_improve(selected, stop_at):
            """Maximal fill followed by genuine 1-for-many improving swaps."""
            selected = construct_from_base(selected, active_mask, 0.0, stop_at)
            while time.time() < stop_at:
                outside = active_mask & ~selected
                groups = {}
                scan = outside
                while scan:
                    bit = scan & -scan
                    v = bit.bit_length() - 1
                    conflicts = adj_bits[v] & selected
                    count = conflicts.bit_count()
                    if count == 0:
                        selected |= bit
                        break
                    if count == 1:
                        blocker = conflicts.bit_length() - 1
                        if blocker in groups:
                            groups[blocker].append(v)
                        else:
                            groups[blocker] = [v]
                    scan ^= bit
                    if (scan.bit_count() & 127) == 0 and time.time() >= stop_at:
                        return selected
                else:
                    blockers = [u for u in groups if len(groups[u]) >= 2]
                    rng.shuffle(blockers)
                    blockers.sort(key=lambda u: len(groups[u]), reverse=True)
                    improved = False
                    for blocker in blockers:
                        if time.time() >= stop_at:
                            return selected
                        candidate_mask = 0
                        for v in groups[blocker]:
                            candidate_mask |= 1 << v
                        additions = 0
                        remaining = candidate_mask
                        while remaining and time.time() < stop_at:
                            choice = -1
                            choice_degree = n + 1
                            scan2 = remaining
                            while scan2:
                                bit2 = scan2 & -scan2
                                v = bit2.bit_length() - 1
                                d = (adj_bits[v] & remaining).bit_count()
                                if d < choice_degree or (
                                    d == choice_degree and rng.random() < 0.25
                                ):
                                    choice = v
                                    choice_degree = d
                                scan2 ^= bit2
                            if choice < 0:
                                break
                            choice_bit = 1 << choice
                            additions |= choice_bit
                            remaining &= ~choice_bit
                            remaining &= ~adj_bits[choice]
                        if additions.bit_count() >= 2:
                            selected &= ~(1 << blocker)
                            selected |= additions
                            selected = construct_from_base(
                                selected, active_mask, 0.0, stop_at
                            )
                            improved = True
                            break
                    if not improved:
                        return selected
                    continue
                # The scan found an unblocked vertex; refill before rebuilding
                # conflict groups, since conflict counts have changed.
                selected = construct_from_base(selected, active_mask, 0.0, stop_at)
            return selected

        # The first dynamic construction is A's strongest mechanism, immediately
        # fed into B's strongest mechanism rather than treated as an alternative.
        heuristic_stop = min(deadline, t0 + budget * 0.63)
        kernel_best = construct_from_base(0, active_mask, 0.0, heuristic_stop)
        kernel_best = local_improve(kernel_best, heuristic_stop)
        combined = forced_mask | kernel_best
        if combined.bit_count() > best_mask.bit_count():
            best_mask = combined

        elites = [kernel_best]

        def remember(mask):
            if mask not in elites and mask.bit_count() + 2 >= kernel_best.bit_count():
                elites.append(mask)
                elites.sort(key=lambda item: item.bit_count(), reverse=True)
                del elites[8:]

        def two_tight_kick(selected, stop_at):
            """Cross a local-search plateau through a forced 2-conflict move."""
            options = []
            scan = active_mask & ~selected
            while scan and time.time() < stop_at:
                bit = scan & -scan
                v = bit.bit_length() - 1
                conflicts = adj_bits[v] & selected
                if conflicts.bit_count() == 2:
                    # Low graph degree and a large pool exposed by removing the
                    # two blockers are useful, but retain randomness across runs.
                    exposure = 0
                    blockers = conflicts
                    while blockers:
                        blocker_bit = blockers & -blockers
                        blocker = blocker_bit.bit_length() - 1
                        exposure += len(adj[blocker])
                        blockers ^= blocker_bit
                    options.append((v, conflicts, exposure - len(adj[v])))
                scan ^= bit
            if not options:
                return selected
            sample = rng.sample(options, min(16, len(options)))
            if rng.random() < 0.75:
                v, conflicts, unused_score = max(sample, key=lambda item: item[2])
            else:
                v, conflicts, unused_score = rng.choice(sample)
            kicked = (selected & ~conflicts) | (1 << v)
            return construct_from_base(
                kicked, active_mask, rng.uniform(0.0, 0.20), stop_at
            )

        stagnation = 0
        # Time-driven iterated search: adaptive destruction, full restarts, and
        # elite crossover all finish with the same dynamic GRASP + swap chain.
        while time.time() < heuristic_stop:
            roll = rng.random()
            if len(elites) >= 2 and roll < 0.18:
                parent_a, parent_b = rng.sample(elites, 2)
                common = parent_a & parent_b
                child = construct_from_base(
                    common, parent_a | parent_b, rng.uniform(0.0, 0.16), heuristic_stop
                )
                current = construct_from_base(
                    child, active_mask, rng.uniform(0.0, 0.20), heuristic_stop
                )
            elif roll < 0.34:
                current = construct_from_base(
                    0, active_mask, rng.uniform(0.0, 0.30), heuristic_stop
                )
            elif roll < 0.62:
                current = two_tight_kick(kernel_best, heuristic_stop)
            else:
                current = kernel_best
                chosen = []
                scan = current
                while scan:
                    bit = scan & -scan
                    chosen.append(bit.bit_length() - 1)
                    scan ^= bit
                if chosen:
                    natural_cap = max(1, len(chosen) // 3)
                    drop_count = min(
                        len(chosen),
                        natural_cap,
                        1 + int(rng.expovariate(0.28)) + min(6, stagnation // 7),
                    )
                    for v in rng.sample(chosen, drop_count):
                        current &= ~(1 << v)
                current = construct_from_base(
                    current, active_mask, rng.uniform(0.0, 0.25), heuristic_stop
                )

            current = local_improve(current, heuristic_stop)
            if current.bit_count() > kernel_best.bit_count():
                kernel_best = current
                best_mask = forced_mask | kernel_best
                stagnation = 0
            else:
                stagnation += 1
            remember(current)

        if time.time() >= deadline:
            return make_answer(best_mask)

        # A bitset maximum-clique search in the complement is an exact MIS
        # search.  It is used only on manageable connected kernels and starts
        # from the heuristic incumbent, so interruption can never hurt quality.
        def exact_component(component_mask, incumbent, stop_at):
            vertices = []
            scan = component_mask
            while scan:
                bit = scan & -scan
                vertices.append(bit.bit_length() - 1)
                scan ^= bit
            vertices.sort(
                key=lambda v: (adj_bits[v] & component_mask).bit_count(),
                reverse=True,
            )
            size = len(vertices)
            local_adj = [0] * size
            index = {v: i for i, v in enumerate(vertices)}
            full = (1 << size) - 1
            for i, v in enumerate(vertices):
                original_neighbors = adj_bits[v] & component_mask
                original_local = 0
                scan2 = original_neighbors
                while scan2:
                    bit2 = scan2 & -scan2
                    original_local |= 1 << index[bit2.bit_length() - 1]
                    scan2 ^= bit2
                local_adj[i] = full & ~original_local & ~(1 << i)

            local_seed = 0
            for i, v in enumerate(vertices):
                if (incumbent >> v) & 1:
                    local_seed |= 1 << i
            exact_best = local_seed
            exact_size = exact_best.bit_count()
            timed_out = False
            node_count = 0

            def color_sort(candidates):
                order = []
                bounds = []
                remaining = candidates
                color = 0
                while remaining:
                    if time.time() >= stop_at:
                        return order, bounds, False
                    color += 1
                    available = remaining
                    while available:
                        bit = available & -available
                        v = bit.bit_length() - 1
                        order.append(v)
                        bounds.append(color)
                        remaining ^= bit
                        available &= remaining
                        available &= ~local_adj[v]
                return order, bounds, True

            def expand(candidates, chosen, chosen_size):
                nonlocal exact_best, exact_size, timed_out, node_count
                node_count += 1
                if (node_count & 63) == 0 and time.time() >= stop_at:
                    timed_out = True
                    return
                if chosen_size + candidates.bit_count() <= exact_size:
                    return
                order, bounds, complete = color_sort(candidates)
                if not complete:
                    timed_out = True
                    return
                i = len(order) - 1
                while i >= 0:
                    if time.time() >= stop_at:
                        timed_out = True
                        return
                    if chosen_size + bounds[i] <= exact_size:
                        return
                    v = order[i]
                    bit = 1 << v
                    new_chosen = chosen | bit
                    new_candidates = candidates & local_adj[v]
                    if new_candidates:
                        expand(new_candidates, new_chosen, chosen_size + 1)
                        if timed_out:
                            return
                    elif chosen_size + 1 > exact_size:
                        exact_best = new_chosen
                        exact_size = chosen_size + 1
                    candidates &= ~bit
                    i -= 1

            expand(full, 0, 0)
            result = 0
            for i, v in enumerate(vertices):
                if (exact_best >> i) & 1:
                    result |= 1 << v
            return result, not timed_out

        # Split the original graph into independent components before exact
        # search.  Small components are often proved optimal almost instantly.
        components = []
        unseen = active_mask
        while unseen and time.time() < deadline:
            root = unseen & -unseen
            frontier = root
            component = 0
            unseen ^= root
            while frontier and time.time() < deadline:
                bit = frontier & -frontier
                frontier ^= bit
                v = bit.bit_length() - 1
                component |= bit
                neighbors = adj_bits[v] & unseen
                unseen &= ~neighbors
                frontier |= neighbors
            components.append(component)
        components.sort(key=lambda mask: mask.bit_count())

        all_exact = True
        attempted_exact = False
        for component in components:
            if time.time() >= deadline:
                all_exact = False
                break
            component_size = component.bit_count()
            if component_size > 360:
                all_exact = False
                continue
            attempted_exact = True
            exact_part, proved = exact_component(
                component, kernel_best & component, deadline
            )
            if exact_part.bit_count() > (kernel_best & component).bit_count():
                kernel_best = (kernel_best & ~component) | exact_part
                best_mask = forced_mask | kernel_best
            if not proved:
                all_exact = False
                break

        if attempted_exact and all_exact:
            return make_answer(best_mask)

        # Large or interrupted kernels spend every remaining instant in the
        # hybrid anytime search instead of idling after the exact cutoff.
        while time.time() < deadline:
            if rng.random() < 0.45:
                current = two_tight_kick(kernel_best, deadline)
            else:
                current = kernel_best
                chosen = []
                scan = current
                while scan:
                    bit = scan & -scan
                    chosen.append(bit.bit_length() - 1)
                    scan ^= bit
                if chosen:
                    drop_cap = max(1, len(chosen) // 3)
                    drop_count = min(
                        len(chosen), drop_cap, 2 + int(rng.expovariate(0.22))
                    )
                    for v in rng.sample(chosen, drop_count):
                        current &= ~(1 << v)
                current = construct_from_base(
                    current, active_mask, rng.uniform(0.0, 0.30), deadline
                )
            current = local_improve(current, deadline)
            if current.bit_count() > kernel_best.bit_count():
                kernel_best = current
                best_mask = forced_mask | kernel_best

        return make_answer(best_mask)
    except Exception:
        # best_mask is updated only with independent sets, so even unexpected
        # malformed input or a heuristic failure returns a complete safe answer.
        return make_answer(best_mask)
