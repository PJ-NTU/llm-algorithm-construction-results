# Library slot 7/10  id=alg_850f77a9c8f2
# score=-163.9  feasible_rate=1.0
# score_vec=[-369.0, -128.0, -100.0, -80.0, -113.0, -54.0, -117.0, -95.0, -444.0, -139.0]

import random
import time


def solve(instance: dict, time_limit_s: float) -> dict:
    """Anytime maximum-independent-set solver using only Python bit sets."""
    t0 = time.time()
    try:
        budget = max(0.0, float(time_limit_s))
    except Exception:
        budget = 0.0
    deadline = t0 + budget * 0.85

    try:
        n = max(0, int(instance.get("num_vertices", 0)))
    except Exception:
        n = 0

    # The empty set is the immediate, always-feasible fallback.
    best_bits = 0
    best_size = 0
    if n == 0 or time.time() >= deadline:
        return {"vertex_selection": [0] * n}

    try:
        adj = [0] * n
        forbidden = 0
        fingerprint = (n * 0x9E3779B1) & 0xFFFFFFFFFFFFFFFF
        for edge in instance.get("edges", []):
            if len(edge) < 2:
                continue
            u, v = edge[0], edge[1]
            if not isinstance(u, int) or not isinstance(v, int):
                continue
            if u < 0 or u >= n or v < 0 or v >= n:
                continue
            fingerprint ^= ((u + 1) * 0x85EBCA77 + (v + 1) * 0xC2B2AE3D)
            fingerprint &= 0xFFFFFFFFFFFFFFFF
            if u == v:
                # The supplied checker interprets a loop as 2*x[u] <= 1.
                forbidden |= 1 << u
            else:
                adj[u] |= 1 << v
                adj[v] |= 1 << u

        all_vertices = ((1 << n) - 1) & ~forbidden
        if not all_vertices:
            return {"vertex_selection": [0] * n}
        degrees = [adj[v].bit_count() for v in range(n)]
        rng = random.Random(fingerprint)

        def iter_bits(bits):
            while bits:
                bit = bits & -bits
                yield bit.bit_length() - 1
                bits ^= bit

        def choose_from_rcl(candidates, random_width):
            """Residual min-degree choice, randomized only among near ties."""
            min_degree = n + 1
            choices = []
            scan = candidates
            while scan:
                bit = scan & -scan
                v = bit.bit_length() - 1
                d = (adj[v] & candidates).bit_count()
                if d < min_degree:
                    min_degree = d
                    choices = [v]
                elif d == min_degree:
                    choices.append(v)
                scan ^= bit
            if random_width and candidates.bit_count() > 2:
                threshold = min_degree + random_width
                scan = candidates
                choices = []
                while scan:
                    bit = scan & -scan
                    v = bit.bit_length() - 1
                    if (adj[v] & candidates).bit_count() <= threshold:
                        choices.append(v)
                    scan ^= bit
            return choices[rng.randrange(len(choices))]

        def maximalize(selected, random_width=0):
            """Complete a feasible seed by dynamic residual-degree greedy."""
            candidates = all_vertices & ~selected
            occupied = selected
            scan = selected
            while scan:
                bit = scan & -scan
                v = bit.bit_length() - 1
                candidates &= ~adj[v]
                scan ^= bit
            candidates &= ~occupied
            while candidates and time.time() < deadline:
                v = choose_from_rcl(candidates, random_width)
                bit = 1 << v
                occupied |= bit
                candidates &= ~bit & ~adj[v]
            return occupied

        def find_nonedge_pair(candidates):
            """Find a promising independent pair in a candidate subgraph."""
            scan = candidates
            chosen_v = -1
            chosen_options = 0
            best_residual_degree = n + 1
            while scan:
                bit = scan & -scan
                v = bit.bit_length() - 1
                options = candidates & ~adj[v] & ~bit
                if options:
                    d = (adj[v] & candidates).bit_count()
                    if d < best_residual_degree:
                        best_residual_degree = d
                        chosen_v = v
                        chosen_options = options
                        if d == 0:
                            break
                scan ^= bit
            if chosen_v < 0:
                return 0
            best_w = -1
            best_degree = n + 1
            scan = chosen_options
            while scan:
                bit = scan & -scan
                w = bit.bit_length() - 1
                d = (adj[w] & candidates).bit_count()
                if d < best_degree:
                    best_degree = d
                    best_w = w
                scan ^= bit
            return (1 << chosen_v) | (1 << best_w)

        def find_independent_triple(candidates):
            """Return three mutually nonadjacent vertices, or zero."""
            scan = candidates
            while scan and time.time() < deadline:
                bit = scan & -scan
                v = bit.bit_length() - 1
                compatible = candidates & ~adj[v] & ~bit
                pair = find_nonedge_pair(compatible)
                if pair:
                    return bit | pair
                scan ^= bit
            return 0

        def improve(selected):
            """Reach a maximal set and close all discovered 1-for-2 moves."""
            selected = maximalize(selected)
            while time.time() < deadline:
                outside = all_vertices & ~selected
                unique_groups = {}
                double_groups = {}
                scan = outside
                while scan:
                    bit = scan & -scan
                    v = bit.bit_length() - 1
                    conflicts = adj[v] & selected
                    count = conflicts.bit_count()
                    if count == 0:
                        selected |= bit
                        break
                    if count == 1:
                        blocker = conflicts.bit_length() - 1
                        unique_groups[blocker] = unique_groups.get(blocker, 0) | bit
                    elif count == 2:
                        first_bit = conflicts & -conflicts
                        first = first_bit.bit_length() - 1
                        second = (conflicts ^ first_bit).bit_length() - 1
                        key = (first, second) if first < second else (second, first)
                        double_groups[key] = double_groups.get(key, 0) | bit
                    scan ^= bit
                else:
                    group_items = list(unique_groups.items())
                    rng.shuffle(group_items)
                    group_items.sort(key=lambda item: item[1].bit_count(), reverse=True)
                    move_remove = -1
                    move_add = 0
                    for blocker, candidates in group_items:
                        if time.time() >= deadline:
                            break
                        if candidates.bit_count() >= 2:
                            pair = find_nonedge_pair(candidates)
                            if pair:
                                move_remove = blocker
                                move_add = pair
                                break
                    move_remove_bits = 0
                    if move_remove >= 0:
                        move_remove_bits = 1 << move_remove
                    else:
                        # Once all 1-for-2 moves are closed, a profitable 2-for-3
                        # move must involve a vertex blocked by exactly that pair.
                        pair_items = list(double_groups.items())
                        pair_items.sort(
                            key=lambda item: (
                                item[1]
                                | unique_groups.get(item[0][0], 0)
                                | unique_groups.get(item[0][1], 0)
                            ).bit_count(),
                            reverse=True,
                        )
                        for blockers, double_candidates in pair_items:
                            if time.time() >= deadline:
                                break
                            candidates = (
                                double_candidates
                                | unique_groups.get(blockers[0], 0)
                                | unique_groups.get(blockers[1], 0)
                            )
                            if candidates.bit_count() < 3:
                                continue
                            triple = find_independent_triple(candidates)
                            if triple:
                                move_remove_bits = (1 << blockers[0]) | (1 << blockers[1])
                                move_add = triple
                                break
                    if not move_remove_bits:
                        return selected
                    selected = (selected & ~move_remove_bits) | move_add
                    selected = maximalize(selected)
                    continue
                selected = maximalize(selected)
            return selected

        def update_best(candidate):
            nonlocal best_bits, best_size
            size = candidate.bit_count()
            if size > best_size:
                best_bits = candidate
                best_size = size

        # Fast deterministic incumbent, followed by complementary randomized ones.
        incumbent = improve(maximalize(0, 0))
        update_best(incumbent)
        population = [incumbent]

        population_target = 10
        while len(population) < population_target and time.time() < deadline:
            candidate = improve(maximalize(0, 1 + (len(population) % 3)))
            update_best(candidate)
            if candidate not in population:
                population.append(candidate)
            elif len(population) == 1:
                # Avoid stalling population construction on highly structured graphs.
                population.append(candidate)
            else:
                break

        # Spend roughly the first third of the safe budget on fused evolutionary/VNS
        # search, leaving the rest to an exact bit-set branch-and-bound continuation.
        # Complement construction is quadratic in the number of vertices.  On large
        # instances the population search is both safer and more useful for the full
        # budget; on moderate instances reserve time for exact continuation.
        exact_size_limit = 640
        if all_vertices.bit_count() <= exact_size_limit:
            heuristic_deadline = min(deadline, t0 + budget * 0.32)
        else:
            heuristic_deadline = deadline
        stagnation = 0
        while time.time() < heuristic_deadline:
            if len(population) >= 2 and rng.random() < 0.48:
                p1, p2 = rng.sample(population, 2)
                # A's stable common backbone is retained.  Unlike intersection-only
                # crossover, differing parent genes then compete by low degree.
                seed = p1 & p2
                union_left = (p1 | p2) & ~seed
                while union_left and time.time() < heuristic_deadline:
                    v = choose_from_rcl(union_left, 1)
                    bit = 1 << v
                    if not (adj[v] & seed):
                        seed |= bit
                        union_left &= ~adj[v]
                    union_left &= ~bit
                child = maximalize(seed, 1)
            else:
                # B's drop-and-refill becomes a variable-neighborhood mutation.
                parent = population[rng.randrange(len(population))]
                vertices = list(iter_bits(parent))
                if vertices:
                    scale = 1 + (stagnation % max(1, min(12, len(vertices) // 4 + 1)))
                    drop_count = min(len(vertices), scale)
                    dropped = rng.sample(vertices, drop_count)
                    child = parent
                    for v in dropped:
                        child &= ~(1 << v)
                    child = maximalize(child, 1 + (stagnation % 3))
                else:
                    child = maximalize(0, 2)

            child = improve(child)
            previous_best = best_size
            update_best(child)
            stagnation = 0 if best_size > previous_best else stagnation + 1

            if child not in population:
                if len(population) < population_target:
                    population.append(child)
                else:
                    # Preserve quality while evicting the most redundant weak member.
                    child_size = child.bit_count()
                    worst_index = 0
                    worst_key = None
                    for i, member in enumerate(population):
                        similarity = max(
                            (member & other).bit_count()
                            for j, other in enumerate(population)
                            if i != j
                        )
                        key = (member.bit_count(), -similarity)
                        if worst_key is None or key < worst_key:
                            worst_key = key
                            worst_index = i
                    if child_size >= population[worst_index].bit_count():
                        population[worst_index] = child

        if time.time() >= deadline:
            result = [0] * n
            for v in iter_bits(best_bits):
                result[v] = 1
            return {"vertex_selection": result}

        # Exact continuation: maximum clique in the complement.  A strong heuristic
        # incumbent makes its greedy-color upper bounds much more effective.
        original_vertices = list(iter_bits(all_vertices))
        original_vertices.sort(key=lambda v: (degrees[v], v), reverse=True)
        mapped_count = len(original_vertices)
        mapped_adj = [0] * mapped_count
        original_to_mapped = [-1] * n
        for mapped, original in enumerate(original_vertices):
            original_to_mapped[original] = mapped
        mapped_all = (1 << mapped_count) - 1
        for mapped, original in enumerate(original_vertices):
            original_neighbors = adj[original]
            independent_neighbors = 0
            scan = all_vertices & ~original_neighbors & ~(1 << original)
            while scan:
                bit = scan & -scan
                v = bit.bit_length() - 1
                independent_neighbors |= 1 << original_to_mapped[v]
                scan ^= bit
            mapped_adj[mapped] = independent_neighbors

        exact_best = 0
        scan = best_bits
        while scan:
            bit = scan & -scan
            original = bit.bit_length() - 1
            exact_best |= 1 << original_to_mapped[original]
            scan ^= bit
        exact_best_size = best_size
        search_nodes = 0
        timed_out = False

        def color_sort(candidates):
            order = []
            bounds = []
            uncolored = candidates
            color = 0
            while uncolored:
                if time.time() >= deadline:
                    return None, None
                color += 1
                color_class = uncolored
                while color_class:
                    bit = color_class & -color_class
                    v = bit.bit_length() - 1
                    order.append(v)
                    bounds.append(color)
                    uncolored ^= bit
                    color_class ^= bit
                    color_class &= ~mapped_adj[v]
            return order, bounds

        def expand(candidates, chosen, chosen_size):
            nonlocal exact_best, exact_best_size, search_nodes, timed_out
            search_nodes += 1
            if (search_nodes & 31) == 0 and time.time() >= deadline:
                timed_out = True
                return
            order, bounds = color_sort(candidates)
            if order is None:
                timed_out = True
                return
            for index in range(len(order) - 1, -1, -1):
                if chosen_size + bounds[index] <= exact_best_size:
                    return
                v = order[index]
                bit = 1 << v
                if not (candidates & bit):
                    continue
                new_chosen = chosen | bit
                new_candidates = candidates & mapped_adj[v]
                if new_candidates:
                    expand(new_candidates, new_chosen, chosen_size + 1)
                    if timed_out:
                        return
                elif chosen_size + 1 > exact_best_size:
                    exact_best = new_chosen
                    exact_best_size = chosen_size + 1
                candidates &= ~bit
                if time.time() >= deadline:
                    timed_out = True
                    return

        expand(mapped_all, 0, 0)
        if exact_best_size > best_size:
            best_bits = 0
            scan = exact_best
            while scan:
                bit = scan & -scan
                mapped = bit.bit_length() - 1
                best_bits |= 1 << original_vertices[mapped]
                scan ^= bit
            best_size = exact_best_size

    except Exception:
        # best_bits always denotes a feasible set: mutations only remove vertices,
        # and all additions are conflict-checked before this catch can be reached.
        pass

    result = [0] * n
    scan = best_bits
    while scan:
        bit = scan & -scan
        v = bit.bit_length() - 1
        if 0 <= v < n:
            result[v] = 1
        scan ^= bit
    return {"vertex_selection": result}
