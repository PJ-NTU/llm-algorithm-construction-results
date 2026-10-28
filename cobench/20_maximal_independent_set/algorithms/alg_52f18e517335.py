# Library slot 6/10  id=alg_52f18e517335
# score=-163.9  feasible_rate=1.0
# score_vec=[-369.0, -128.0, -100.0, -80.0, -113.0, -55.0, -116.0, -97.0, -446.0, -135.0]

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
    best_global = 0

    def make_result(mask):
        answer = [0] * n
        while mask:
            bit = mask & -mask
            answer[bit.bit_length() - 1] = 1
            mask ^= bit
        return {"vertex_selection": answer}

    if n == 0 or time.time() >= deadline:
        return make_result(0)

    try:
        adj_sets = [set() for _ in range(n)]
        banned = set()
        checksum = (n * 1000003) & 0xFFFFFFFFFFFFFFFF
        for edge_number, edge in enumerate(instance.get("edges", [])):
            if edge_number & 4095 == 0 and time.time() >= deadline:
                return make_result(best_global)
            if len(edge) < 2:
                continue
            u, v = int(edge[0]), int(edge[1])
            if not (0 <= u < n and 0 <= v < n):
                continue
            if u == v:
                banned.add(u)
            else:
                adj_sets[u].add(v)
                adj_sets[v].add(u)
                checksum = (
                    checksum * 1000003 + u * 9176 + v * 6113
                ) & 0xFFFFFFFFFFFFFFFF

        adj = [0] * n
        for v in range(n):
            mask = 0
            for u in adj_sets[v]:
                mask |= 1 << u
            adj[v] = mask

        active = [v not in banned for v in range(n)]
        degree = [0] * n
        heap = []
        for v in range(n):
            if active[v]:
                degree[v] = sum(1 for u in adj_sets[v] if active[u])
                heap.append((degree[v], v))
        heapq.heapify(heap)
        forced = 0

        def remove_active(v):
            if not active[v]:
                return
            active[v] = False
            for u in adj_sets[v]:
                if active[u]:
                    degree[u] -= 1
                    heapq.heappush(heap, (degree[u], u))

        # Isolated vertices, leaves, and low-degree simplicial vertices are
        # safe inclusions: an optimum containing each chosen vertex exists.
        def reduce_easy():
            nonlocal forced
            changed = False
            while heap and time.time() < deadline:
                d, v = heapq.heappop(heap)
                if not active[v] or d != degree[v]:
                    continue
                choose = d <= 1
                neighbors = []
                if not choose and d <= 6:
                    neighbors = [u for u in adj_sets[v] if active[u]]
                    choose = True
                    for i in range(len(neighbors)):
                        for j in range(i + 1, len(neighbors)):
                            if not (adj[neighbors[i]] >> neighbors[j]) & 1:
                                choose = False
                                break
                        if not choose:
                            break
                if not choose:
                    if d > 6:
                        heapq.heappush(heap, (d, v))
                        break
                    continue
                forced |= 1 << v
                victims = [v]
                victims.extend(u for u in adj_sets[v] if active[u])
                for u in victims:
                    remove_active(u)
                changed = True
            return changed

        reduce_easy()

        # Closed-neighborhood domination is another exact reduction.  It is
        # deliberately limited to a few passes; the hard search is more useful
        # than pursuing a fixed point on large kernels.
        for _ in range(3):
            if time.time() >= deadline:
                break
            active_mask = 0
            for v in range(n):
                if active[v]:
                    active_mask |= 1 << v
            if active_mask.bit_count() > 1400:
                break
            removed = False
            scan = active_mask
            checks = 0
            while scan and time.time() < deadline:
                vb = scan & -scan
                v = vb.bit_length() - 1
                closed_v = (adj[v] | vb) & active_mask
                neighbors = adj[v] & active_mask
                while neighbors:
                    ub = neighbors & -neighbors
                    u = ub.bit_length() - 1
                    closed_u = (adj[u] | ub) & active_mask
                    if closed_u & ~closed_v == 0:
                        remove_active(v)
                        active_mask ^= vb
                        removed = True
                        break
                    neighbors ^= ub
                    checks += 1
                    if checks & 1023 == 0 and time.time() >= deadline:
                        break
                scan ^= vb
            if not removed:
                break
            reduce_easy()

        best_global = forced
        if time.time() >= deadline:
            return make_result(best_global)

        active_mask = 0
        for v in range(n):
            if active[v]:
                active_mask |= 1 << v

        # Split the kernel.  Bipartite components are polynomial, and solving
        # components separately also makes the exact neighborhoods much tighter.
        components = []
        colors = [-1] * n
        unseen = active_mask
        while unseen and time.time() < deadline:
            root_bit = unseen & -unseen
            root = root_bit.bit_length() - 1
            q = deque([root])
            unseen ^= root_bit
            colors[root] = 0
            comp_mask = 0
            bipartite = True
            while q:
                v = q.popleft()
                comp_mask |= 1 << v
                nbrs = adj[v] & active_mask
                scan = nbrs
                while scan:
                    bit = scan & -scan
                    u = bit.bit_length() - 1
                    if colors[u] < 0:
                        colors[u] = colors[v] ^ 1
                        unseen &= ~bit
                        q.append(u)
                    elif colors[u] == colors[v]:
                        bipartite = False
                    scan ^= bit
            components.append((comp_mask, bipartite))

        rng = random.Random(checksum ^ (len(banned) << 37))

        def greedy_fill(selected, universe, randomized=False):
            available = universe & ~selected
            scan = selected
            while scan:
                bit = scan & -scan
                available &= ~adj[bit.bit_length() - 1]
                scan ^= bit
            while available and time.time() < deadline:
                scan = available
                min_degree = n + 1
                candidates = []
                near_candidates = []
                inspected = 0
                while scan:
                    bit = scan & -scan
                    v = bit.bit_length() - 1
                    d = (adj[v] & available).bit_count()
                    if d < min_degree:
                        min_degree = d
                        candidates = [v]
                    elif d == min_degree:
                        candidates.append(v)
                    scan ^= bit
                    inspected += 1
                    if inspected & 1023 == 0 and time.time() >= deadline:
                        return selected
                if randomized:
                    scan = available
                    width = 1 + rng.randrange(3)
                    while scan:
                        bit = scan & -scan
                        candidate = bit.bit_length() - 1
                        if (adj[candidate] & available).bit_count() <= min_degree + width:
                            near_candidates.append(candidate)
                        scan ^= bit
                    v = near_candidates[rng.randrange(len(near_candidates))]
                else:
                    # Among minimum-degree vertices, prefer the one whose
                    # neighbors are themselves expensive to keep.
                    v = candidates[0]
                    best_pressure = -1
                    for candidate in candidates:
                        pressure = 0
                        nbrs = adj[candidate] & available
                        while nbrs:
                            bit = nbrs & -nbrs
                            pressure += (adj[bit.bit_length() - 1] & available).bit_count()
                            nbrs ^= bit
                        if pressure > best_pressure:
                            best_pressure = pressure
                            v = candidate
                bit = 1 << v
                selected |= bit
                available &= ~(bit | adj[v])
            return selected

        def bipartite_mis(comp_mask):
            vertices = []
            scan = comp_mask
            while scan:
                bit = scan & -scan
                vertices.append(bit.bit_length() - 1)
                scan ^= bit
            left = [v for v in vertices if colors[v] == 0]
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
                    nbrs = adj[u] & comp_mask
                    while nbrs:
                        bit = nbrs & -nbrs
                        v = bit.bit_length() - 1
                        mate = pair[v]
                        if mate < 0:
                            found = True
                        elif dist[mate] < 0:
                            dist[mate] = dist[u] + 1
                            q.append(mate)
                        nbrs ^= bit
                return found

            def dfs(u):
                nbrs = adj[u] & comp_mask
                while nbrs:
                    bit = nbrs & -nbrs
                    v = bit.bit_length() - 1
                    mate = pair[v]
                    if mate < 0 or (dist[mate] == dist[u] + 1 and dfs(mate)):
                        pair[u] = v
                        pair[v] = u
                        return True
                    nbrs ^= bit
                dist[u] = -1
                return False

            complete = False
            while time.time() < deadline:
                if not bfs():
                    complete = True
                    break
                for u in left:
                    if pair[u] < 0 and time.time() < deadline:
                        dfs(u)
            if not complete:
                left_mask = 0
                for v in left:
                    left_mask |= 1 << v
                return left_mask if len(left) * 2 >= len(vertices) else comp_mask ^ left_mask

            z_left = 0
            z_right = 0
            q = deque()
            for u in left:
                if pair[u] < 0:
                    z_left |= 1 << u
                    q.append((u, 0))
            while q:
                v, side = q.popleft()
                if side == 0:
                    nbrs = adj[v] & comp_mask & ~z_right
                    if pair[v] >= 0:
                        nbrs &= ~(1 << pair[v])
                    while nbrs:
                        bit = nbrs & -nbrs
                        z_right |= bit
                        q.append((bit.bit_length() - 1, 1))
                        nbrs ^= bit
                else:
                    mate = pair[v]
                    if mate >= 0 and not ((z_left >> mate) & 1):
                        z_left |= 1 << mate
                        q.append((mate, 0))
            right_mask = comp_mask
            for v in left:
                right_mask &= ~(1 << v)
            return z_left | (right_mask & ~z_right)

        hard_components = []
        for comp_mask, is_bipartite in components:
            if time.time() >= deadline:
                break
            if is_bipartite:
                best_global |= bipartite_mis(comp_mask)
            else:
                hard_components.append(comp_mask)

        if time.time() >= deadline:
            return make_result(best_global)

        def find_independent_triple(candidates, stop_time):
            first_scan = candidates
            checks = 0
            while first_scan and time.time() < stop_time:
                first = first_scan & -first_scan
                u = first.bit_length() - 1
                after_u = candidates & ~adj[u] & ~((first << 1) - 1)
                second_scan = after_u
                while second_scan:
                    second = second_scan & -second_scan
                    v = second.bit_length() - 1
                    third = after_u & ~adj[v] & ~((second << 1) - 1)
                    if third:
                        return first | second | (third & -third)
                    second_scan ^= second
                    checks += 1
                    if checks & 127 == 0 and time.time() >= stop_time:
                        return 0
                first_scan ^= first
            return 0

        def local_improve(selected, universe, allow_triples=True):
            while time.time() < deadline:
                tight_one = {}
                tight_two = {}
                outside = universe & ~selected
                scan = outside
                while scan:
                    bit = scan & -scan
                    v = bit.bit_length() - 1
                    conflicts = adj[v] & selected
                    count = conflicts.bit_count()
                    if count == 0:
                        selected |= bit
                    elif count == 1:
                        owner = conflicts.bit_length() - 1
                        tight_one[owner] = tight_one.get(owner, 0) | bit
                    elif count == 2 and allow_triples:
                        first = conflicts & -conflicts
                        a = first.bit_length() - 1
                        b = (conflicts ^ first).bit_length() - 1
                        tight_two[(a, b)] = tight_two.get((a, b), 0) | bit
                    scan ^= bit
                moved = False
                owners = list(tight_one)
                rng.shuffle(owners)
                for owner in owners:
                    candidates = tight_one[owner]
                    scan = candidates
                    while scan:
                        first = scan & -scan
                        u = first.bit_length() - 1
                        partner = candidates & ~adj[u] & ~first
                        if partner:
                            selected &= ~(1 << owner)
                            selected |= first | (partner & -partner)
                            selected = greedy_fill(selected, universe, False)
                            moved = True
                            break
                        scan ^= first
                    if moved or time.time() >= deadline:
                        break
                if moved:
                    continue
                if allow_triples:
                    pairs = list(tight_two)
                    rng.shuffle(pairs)
                    triple_stop = min(deadline, time.time() + 0.08)
                    for a, b in pairs:
                        candidates = (
                            tight_two[(a, b)]
                            | tight_one.get(a, 0)
                            | tight_one.get(b, 0)
                        )
                        triple = find_independent_triple(candidates, triple_stop)
                        if triple:
                            selected &= ~((1 << a) | (1 << b))
                            selected |= triple
                            selected = greedy_fill(selected, universe, False)
                            moved = True
                            break
                        if time.time() >= triple_stop:
                            break
                if not moved:
                    return selected
            return selected

        # Maximum clique in the complement, used as a bounded exact solver for
        # large neighborhoods.  Greedy coloring supplies a valid upper bound.
        def exact_neighborhood(universe, incumbent, stop_time):
            vertices = []
            scan = universe
            while scan:
                bit = scan & -scan
                vertices.append(bit.bit_length() - 1)
                scan ^= bit
            vertices.sort(
                key=lambda v: ((adj[v] & universe).bit_count(), v), reverse=True
            )
            qn = len(vertices)
            if qn == 0:
                return 0
            local_index = {v: i for i, v in enumerate(vertices)}
            all_local = (1 << qn) - 1
            comp_adj = [0] * qn
            for i, v in enumerate(vertices):
                nonneighbors = universe & ~adj[v] & ~(1 << v)
                mask = 0
                while nonneighbors:
                    bit = nonneighbors & -nonneighbors
                    mask |= 1 << local_index[bit.bit_length() - 1]
                    nonneighbors ^= bit
                comp_adj[i] = mask

            local_best = 0
            scan = incumbent & universe
            while scan:
                bit = scan & -scan
                local_best |= 1 << local_index[bit.bit_length() - 1]
                scan ^= bit
            local_best_size = local_best.bit_count()
            nodes = 0

            def color_sort(candidates):
                order = []
                bounds = []
                remaining = candidates
                color = 0
                while remaining:
                    color += 1
                    colorable = remaining
                    while colorable:
                        bit = colorable & -colorable
                        v = bit.bit_length() - 1
                        order.append(v)
                        bounds.append(color)
                        remaining ^= bit
                        colorable &= ~bit & ~comp_adj[v]
                return order, bounds

            def expand(clique, candidates):
                nonlocal local_best, local_best_size, nodes
                nodes += 1
                if nodes & 255 == 0 and time.time() >= stop_time:
                    return
                order, bounds = color_sort(candidates)
                for i in range(len(order) - 1, -1, -1):
                    if clique.bit_count() + bounds[i] <= local_best_size:
                        return
                    if time.time() >= stop_time:
                        return
                    v = order[i]
                    bit = 1 << v
                    if not (candidates & bit):
                        continue
                    new_clique = clique | bit
                    new_candidates = candidates & comp_adj[v]
                    if new_candidates:
                        expand(new_clique, new_candidates)
                    elif new_clique.bit_count() > local_best_size:
                        local_best = new_clique
                        local_best_size = new_clique.bit_count()
                    candidates ^= bit

            expand(0, all_local)
            answer = 0
            scan = local_best
            while scan:
                bit = scan & -scan
                answer |= 1 << vertices[bit.bit_length() - 1]
                scan ^= bit
            return answer

        def cover_search(universe, initial_independent, stop_time):
            """Weighted local search on the complementary vertex-cover view."""
            vertices = []
            scan = universe
            while scan:
                bit = scan & -scan
                vertices.append(bit.bit_length() - 1)
                scan ^= bit
            edges = []
            incident = {}
            for v in vertices:
                incident[v] = []
            for v in vertices:
                nbrs = adj[v] & universe & ~((1 << (v + 1)) - 1)
                while nbrs:
                    bit = nbrs & -nbrs
                    u = bit.bit_length() - 1
                    index = len(edges)
                    edges.append((v, u))
                    incident[v].append(index)
                    incident[u].append(index)
                    nbrs ^= bit
            if not edges:
                return universe

            cover = universe & ~initial_independent
            best_independent = initial_independent
            weights = [1] * len(edges)
            positions = [-1] * len(edges)
            uncovered = []
            changed_at = [0] * n
            step = 0
            last_added = -1

            def add_uncovered(edge_index):
                if positions[edge_index] < 0:
                    positions[edge_index] = len(uncovered)
                    uncovered.append(edge_index)

            def remove_uncovered(edge_index):
                position = positions[edge_index]
                if position >= 0:
                    tail = uncovered.pop()
                    if position < len(uncovered):
                        uncovered[position] = tail
                        positions[tail] = position
                    positions[edge_index] = -1

            def remove_from_cover(v):
                nonlocal cover
                cover &= ~(1 << v)
                for edge_index in incident[v]:
                    a, b = edges[edge_index]
                    other = b if a == v else a
                    if not ((cover >> other) & 1):
                        add_uncovered(edge_index)

            def add_to_cover(v):
                nonlocal cover
                cover |= 1 << v
                for edge_index in incident[v]:
                    remove_uncovered(edge_index)

            def score(v):
                value = 0
                if (cover >> v) & 1:
                    for edge_index in incident[v]:
                        a, b = edges[edge_index]
                        other = b if a == v else a
                        if not ((cover >> other) & 1):
                            value -= weights[edge_index]
                else:
                    for edge_index in incident[v]:
                        if positions[edge_index] >= 0:
                            value += weights[edge_index]
                return value

            def choose_removal():
                choice = -1
                choice_key = None
                for v in vertices:
                    if not ((cover >> v) & 1) or v == last_added:
                        continue
                    key = (score(v), -changed_at[v], rng.random())
                    if choice_key is None or key > choice_key:
                        choice_key = key
                        choice = v
                if choice < 0:
                    for v in vertices:
                        if (cover >> v) & 1:
                            return v
                return choice

            first_removal = choose_removal()
            if first_removal < 0:
                return best_independent
            remove_from_cover(first_removal)

            while time.time() < stop_time:
                step += 1
                if not uncovered:
                    independent = universe & ~cover
                    if independent.bit_count() > best_independent.bit_count():
                        best_independent = independent
                    v = choose_removal()
                    if v < 0:
                        break
                    remove_from_cover(v)
                    changed_at[v] = step
                    last_added = -1
                    continue

                v = choose_removal()
                if v >= 0:
                    remove_from_cover(v)
                    changed_at[v] = step

                edge_index = uncovered[rng.randrange(len(uncovered))]
                a, b = edges[edge_index]
                score_a = score(a)
                score_b = score(b)
                if score_a > score_b:
                    chosen = a
                elif score_b > score_a:
                    chosen = b
                else:
                    chosen = a if changed_at[a] <= changed_at[b] else b
                add_to_cover(chosen)
                changed_at[chosen] = step
                last_added = chosen

                for edge_index in uncovered:
                    weights[edge_index] += 1
                if step % 1000 == 0 and sum(weights) > 50 * len(weights):
                    for edge_index in range(len(weights)):
                        weights[edge_index] = max(1, weights[edge_index] // 3)

            return best_independent

        hard_solutions = []
        for comp_mask in hard_components:
            if time.time() >= deadline:
                break
            solution = greedy_fill(0, comp_mask, False)
            solution = local_improve(solution, comp_mask, True)
            hard_solutions.append([comp_mask, solution, solution])
            best_global |= solution

        # Any components not reached because of an extremely short budget still
        # receive an immediate feasible maximal set.
        if len(hard_solutions) < len(hard_components):
            for comp_mask in hard_components[len(hard_solutions):]:
                solution = greedy_fill(0, comp_mask, False)
                hard_solutions.append([comp_mask, solution, solution])
                best_global |= solution

        if time.time() >= deadline:
            return make_result(best_global)

        # Small components are often solved outright.  Give each only a bounded
        # slice so no unexpectedly hard component can consume the whole exam.
        for item in hard_solutions:
            comp_mask, solution = item[0], item[1]
            size = comp_mask.bit_count()
            if size <= 220 and time.time() < deadline:
                slice_s = min(0.35, max(0.01, (deadline - time.time()) * 0.08))
                exact = exact_neighborhood(
                    comp_mask, solution, min(deadline, time.time() + slice_s)
                )
                if exact.bit_count() > solution.bit_count():
                    best_global ^= solution
                    best_global |= exact
                    item[1] = exact
                    item[2] = exact

        if hard_solutions and time.time() < deadline:
            cover_budget_end = time.time() + (deadline - time.time()) * 0.42
            for position, item in enumerate(hard_solutions):
                if time.time() >= cover_budget_end:
                    break
                remaining_components = len(hard_solutions) - position
                stop_time = time.time() + (
                    cover_budget_end - time.time()
                ) / remaining_components
                cover_answer = cover_search(item[0], item[1], stop_time)
                if cover_answer.bit_count() > item[1].bit_count():
                    best_global ^= item[1]
                    best_global |= cover_answer
                    item[1] = cover_answer
                    item[2] = cover_answer

        # Iterated exact large-neighborhood search.  Removing a group R from a
        # maximal set exposes exactly the vertices whose selected conflicts are
        # contained in R; optimizing that induced subgraph realizes general
        # k-for-(k+1) exchanges, not just the reference solver's 1/2 exchanges.
        turn = 0
        while time.time() < deadline:
            if not hard_solutions:
                break
            item = hard_solutions[turn % len(hard_solutions)]
            turn += 1
            universe, incumbent, current = item
            selected_vertices = []
            scan = current
            while scan:
                bit = scan & -scan
                selected_vertices.append(bit.bit_length() - 1)
                scan ^= bit
            if not selected_vertices:
                break

            remaining_time = deadline - time.time()
            if remaining_time <= 0:
                break
            remove_count = min(
                len(selected_vertices), 2 + (turn % 9)
            )
            seed = selected_vertices[rng.randrange(len(selected_vertices))]
            removal = 1 << seed

            # Grow R along co-conflict links so the exposed neighborhood is rich
            # enough to contain improving exchanges.
            owner_scores = {}
            outside = universe & ~current
            scan = outside
            while scan and removal.bit_count() < remove_count:
                bit = scan & -scan
                conflicts = adj[bit.bit_length() - 1] & current
                if conflicts & removal:
                    owners = conflicts & ~removal
                    while owners:
                        owner_bit = owners & -owners
                        owner = owner_bit.bit_length() - 1
                        owner_scores[owner] = owner_scores.get(owner, 0) + 1
                        owners ^= owner_bit
                scan ^= bit
            while removal.bit_count() < remove_count and owner_scores:
                peak = max(owner_scores.values())
                choices = [v for v, score in owner_scores.items() if score == peak]
                v = choices[rng.randrange(len(choices))]
                removal |= 1 << v
                del owner_scores[v]
            while removal.bit_count() < remove_count:
                v = selected_vertices[rng.randrange(len(selected_vertices))]
                removal |= 1 << v

            fixed = current & ~removal
            neighborhood = removal
            scan = universe & ~current
            while scan:
                bit = scan & -scan
                v = bit.bit_length() - 1
                if adj[v] & fixed == 0:
                    neighborhood |= bit
                scan ^= bit
                if neighborhood.bit_count() > 150:
                    break

            old_part = current & neighborhood
            slice_s = min(0.10, max(0.012, remaining_time * 0.03))
            replacement = exact_neighborhood(
                neighborhood, old_part, min(deadline, time.time() + slice_s)
            )
            candidate = fixed | replacement
            candidate = greedy_fill(candidate, universe, turn % 3 != 0)
            candidate = local_improve(candidate, universe, turn % 2 == 0)

            if candidate.bit_count() > incumbent.bit_count():
                best_global ^= incumbent
                best_global |= candidate
                item[1] = candidate
                item[2] = candidate
            elif candidate.bit_count() >= incumbent.bit_count() - 1:
                # Keep equal-quality (and occasionally one-worse) states: this
                # is what lets the walk cross plateaus that defeat pure descent.
                item[2] = candidate
            elif time.time() < deadline:
                # A diversified ruin-and-recreate attempt is cheap and provides
                # useful movement when exact LNS repeatedly proves no small gain.
                ruin_count = min(
                    len(selected_vertices), 2 + (turn % max(3, len(selected_vertices) // 6 + 1))
                )
                ruined = current
                for _ in range(ruin_count):
                    if not ruined:
                        break
                    bits = []
                    scan = ruined
                    while scan:
                        bit = scan & -scan
                        bits.append(bit)
                        scan ^= bit
                    ruined ^= bits[rng.randrange(len(bits))]
                rebuilt = greedy_fill(ruined, universe, True)
                rebuilt = local_improve(rebuilt, universe, turn % 4 == 0)
                if rebuilt.bit_count() > incumbent.bit_count():
                    best_global ^= incumbent
                    best_global |= rebuilt
                    item[1] = rebuilt
                    item[2] = rebuilt
                elif rebuilt.bit_count() >= incumbent.bit_count() - 1:
                    item[2] = rebuilt

            if time.time() < deadline and turn % 3 == 0:
                # An explicit 1-for-1 move changes the maximal set even when an
                # exact neighborhood merely reconstructs its incumbent.
                walk = item[2]
                choices = []
                scan = universe & ~walk
                while scan:
                    bit = scan & -scan
                    v = bit.bit_length() - 1
                    conflicts = adj[v] & walk
                    if conflicts.bit_count() == 1:
                        choices.append((bit, conflicts))
                    scan ^= bit
                if choices:
                    bit, conflict = choices[rng.randrange(len(choices))]
                    walked = (walk & ~conflict) | bit
                    walked = greedy_fill(walked, universe, True)
                    walked = local_improve(walked, universe, turn % 6 == 0)
                    item[2] = walked
                    if walked.bit_count() > item[1].bit_count():
                        best_global ^= item[1]
                        best_global |= walked
                        item[1] = walked

        return make_result(best_global)
    except Exception:
        return make_result(best_global)
