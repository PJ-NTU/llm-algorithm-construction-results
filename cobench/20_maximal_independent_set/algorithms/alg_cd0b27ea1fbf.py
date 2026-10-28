# Library slot 1/10  id=alg_cd0b27ea1fbf
# score=-164.7  feasible_rate=1.0
# score_vec=[-369.0, -128.0, -100.0, -80.0, -114.0, -55.0, -117.0, -99.0, -445.0, -140.0]

"""Anytime maximum-independent-set solver using only the Python standard library."""

import random
import time
from collections import deque


def solve(instance: dict, time_limit_s: float) -> dict:
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

    best = 0

    def answer(mask):
        return {
            "vertex_selection": [1 if mask & (1 << v) else 0 for v in range(n)]
        }

    if n == 0 or time.time() >= deadline:
        return answer(best)

    try:
        # Bit sets make conflict counting, exchange evaluation, and graph
        # reduction inexpensive at the exam's medium-sized graph scales.
        adj = [0] * n
        forbidden = 0
        checksum = n * 1000003
        edge_no = 0
        for edge in instance.get("edges", []):
            edge_no += 1
            if (edge_no & 2047) == 0 and time.time() >= deadline:
                return answer(best)
            if not isinstance(edge, (list, tuple)) or len(edge) < 2:
                continue
            u, v = edge[0], edge[1]
            if not isinstance(u, int) or not isinstance(v, int):
                continue
            if not (0 <= u < n and 0 <= v < n):
                continue
            if u == v:
                forbidden |= 1 << u
            else:
                ub = 1 << u
                vb = 1 << v
                adj[u] |= vb
                adj[v] |= ub
                checksum = (
                    checksum * 1000003 + (u + 1) * 9176 + (v + 1) * 6361
                ) & ((1 << 64) - 1)

        valid = ((1 << n) - 1) & ~forbidden
        if not valid or time.time() >= deadline:
            return answer(best)
        rng = random.Random(checksum)

        def blocked_by(selected):
            blocked = selected
            scan = selected
            while scan:
                bit = scan & -scan
                scan ^= bit
                blocked |= adj[bit.bit_length() - 1]
            return blocked

        static_degree = [(adj[v] & valid).bit_count() for v in range(n)]
        pressure = [0] * n
        for v in range(n):
            scan = adj[v] & valid
            total = 0
            while scan:
                bit = scan & -scan
                scan ^= bit
                total += static_degree[bit.bit_length() - 1]
            pressure[v] = total

        def greedy_fill(selected, pool, noise, stop_at):
            """Dynamic min-degree completion, randomized only when requested."""
            selected &= pool
            available = pool & ~blocked_by(selected)
            while available and time.time() < stop_at:
                scan = available
                chosen = 0
                chosen_key = None
                inspected = 0
                while scan:
                    bit = scan & -scan
                    scan ^= bit
                    v = bit.bit_length() - 1
                    d = (adj[v] & available).bit_count()
                    if noise <= 0.0:
                        key = (d, -pressure[v], static_degree[v], v)
                    else:
                        # Residual degree still dominates small perturbations;
                        # larger noise admits near-minimum choices on restarts.
                        key = (
                            d + rng.random() * noise * (1.0 + 0.22 * d),
                            -pressure[v],
                        )
                    if chosen_key is None or key < chosen_key:
                        chosen_key = key
                        chosen = bit
                    inspected += 1
                    if (inspected & 255) == 0 and time.time() >= stop_at:
                        break
                if not chosen:
                    break
                v = chosen.bit_length() - 1
                selected |= chosen
                available &= ~(chosen | adj[v])
            return selected

        # A full feasible incumbent is established before preprocessing.
        best = greedy_fill(0, valid, 0.0, deadline)
        if time.time() >= deadline:
            return answer(best)

        # Safe kernelization.  A simplicial vertex belongs to some optimum:
        # its selected neighbor, if any, can be replaced by the vertex itself.
        active = valid
        fixed = 0
        changed = True
        while changed and active and time.time() < deadline:
            changed = False
            scan = active
            inspected = 0
            while scan and time.time() < deadline:
                bit = scan & -scan
                scan ^= bit
                v = bit.bit_length() - 1
                neighbors = adj[v] & active
                d = neighbors.bit_count()
                if d <= 14:
                    clique = True
                    check = neighbors
                    while check:
                        ubit = check & -check
                        check ^= ubit
                        u = ubit.bit_length() - 1
                        if (neighbors ^ ubit) & ~adj[u]:
                            clique = False
                            break
                    if clique:
                        fixed |= bit
                        active &= ~(bit | neighbors)
                        changed = True
                        break
                inspected += 1
                if (inspected & 127) == 0 and time.time() >= deadline:
                    break

            if changed or time.time() >= deadline:
                continue

            # Closed-neighborhood domination: if N[v] is contained in N[u],
            # every optimum using u has an equally large replacement using v,
            # so u may be deleted.  This complements simplicial reduction on
            # sparse kernels whose low-degree neighborhoods are not cliques.
            scan = active
            while scan and time.time() < deadline:
                vbit = scan & -scan
                scan ^= vbit
                v = vbit.bit_length() - 1
                closed_v = (adj[v] & active) | vbit
                neighbors = adj[v] & active
                while neighbors:
                    ubit = neighbors & -neighbors
                    neighbors ^= ubit
                    u = ubit.bit_length() - 1
                    if closed_v & ~((adj[u] & active) | ubit) == 0:
                        active &= ~ubit
                        changed = True
                        break
                if changed:
                    break

        reduced_candidate = fixed | greedy_fill(0, active, 0.0, deadline)
        if reduced_candidate.bit_count() > best.bit_count():
            best = reduced_candidate
        if not active or time.time() >= deadline:
            return answer(best)

        # Solve every residual bipartite component exactly using matching and
        # Konig's theorem.  Non-bipartite components remain for local search.
        seen = 0
        color = [-1] * n
        components = []
        roots = active
        while roots and time.time() < deadline:
            root_bit = roots & -roots
            root = root_bit.bit_length() - 1
            q = deque([root])
            seen |= root_bit
            roots ^= root_bit
            color[root] = 0
            comp = []
            comp_mask = 0
            is_bipartite = True
            while q and time.time() < deadline:
                v = q.popleft()
                vbit = 1 << v
                comp.append(v)
                comp_mask |= vbit
                neighbors = adj[v] & active
                scan = neighbors
                while scan:
                    ubit = scan & -scan
                    scan ^= ubit
                    u = ubit.bit_length() - 1
                    if not (seen & ubit):
                        seen |= ubit
                        roots &= ~ubit
                        color[u] = color[v] ^ 1
                        q.append(u)
                    elif color[u] == color[v]:
                        is_bipartite = False
            if q:
                break
            components.append((comp, comp_mask, is_bipartite))

        exact_fixed = fixed
        kernel = active

        def bipartite_independent_set(comp, comp_mask, stop_at):
            left = [v for v in comp if color[v] == 0]
            pair = [-1] * n
            dist = [-1] * n

            def bfs():
                queue = deque()
                augmenting = False
                for u in left:
                    if pair[u] < 0:
                        dist[u] = 0
                        queue.append(u)
                    else:
                        dist[u] = -1
                while queue and time.time() < stop_at:
                    u = queue.popleft()
                    scan = adj[u] & comp_mask
                    while scan:
                        bit = scan & -scan
                        scan ^= bit
                        v = bit.bit_length() - 1
                        mate = pair[v]
                        if mate < 0:
                            augmenting = True
                        elif dist[mate] < 0:
                            dist[mate] = dist[u] + 1
                            queue.append(mate)
                return augmenting

            def dfs(u):
                scan = adj[u] & comp_mask
                while scan and time.time() < stop_at:
                    bit = scan & -scan
                    scan ^= bit
                    v = bit.bit_length() - 1
                    mate = pair[v]
                    if mate < 0 or (dist[mate] == dist[u] + 1 and dfs(mate)):
                        pair[u] = v
                        pair[v] = u
                        return True
                dist[u] = -1
                return False

            complete = False
            while time.time() < stop_at:
                if not bfs():
                    complete = True
                    break
                for u in left:
                    if pair[u] < 0 and time.time() < stop_at:
                        dfs(u)
            if not complete:
                return 0, False

            # Alternating reachability gives a minimum vertex cover.  Its
            # complement is a maximum independent set.
            z_left = 0
            z_right = 0
            queue = deque()
            for u in left:
                if pair[u] < 0:
                    z_left |= 1 << u
                    queue.append((u, 0))
            while queue and time.time() < stop_at:
                v, side = queue.popleft()
                if side == 0:
                    scan = adj[v] & comp_mask & ~z_right
                    while scan:
                        bit = scan & -scan
                        scan ^= bit
                        u = bit.bit_length() - 1
                        if pair[v] != u:
                            z_right |= bit
                            queue.append((u, 1))
                else:
                    u = pair[v]
                    if u >= 0 and not (z_left & (1 << u)):
                        z_left |= 1 << u
                        queue.append((u, 0))
            if queue:
                return 0, False
            right_mask = comp_mask
            for u in left:
                right_mask &= ~(1 << u)
            return z_left | (right_mask & ~z_right), True

        for comp, comp_mask, is_bipartite in components:
            if time.time() >= deadline:
                break
            if is_bipartite:
                independent, complete = bipartite_independent_set(
                    comp, comp_mask, deadline
                )
                if complete:
                    exact_fixed |= independent
                    kernel &= ~comp_mask

        candidate = exact_fixed | greedy_fill(0, kernel, 0.0, deadline)
        if candidate.bit_count() > best.bit_count():
            best = candidate
        if not kernel or time.time() >= deadline:
            return answer(best)

        kernel_degrees = [(adj[v] & kernel).bit_count() for v in range(n)]

        def independent_pair(mask, stop_at):
            scan = mask
            while scan and time.time() < stop_at:
                first = scan & -scan
                scan ^= first
                v = first.bit_length() - 1
                mates = scan & ~adj[v]
                if mates:
                    return first | (mates & -mates)
            return 0

        def independent_triple(mask, stop_at):
            firsts = mask
            while firsts and time.time() < stop_at:
                first = firsts & -firsts
                firsts ^= first
                u = first.bit_length() - 1
                seconds = firsts & ~adj[u]
                scan = seconds
                while scan and time.time() < stop_at:
                    second = scan & -scan
                    scan ^= second
                    v = second.bit_length() - 1
                    thirds = scan & ~adj[v]
                    if thirds:
                        return first | second | (thirds & -thirds)
            return 0

        def local_improve(selected, stop_at):
            """Reach a maximal, 1->2 and targeted 2->3 local optimum."""
            selected = greedy_fill(selected, kernel, 0.0, stop_at)
            while time.time() < stop_at:
                tight_one = {}
                tight_two = {}
                outside = kernel & ~selected
                scan = outside
                inspected = 0
                while scan:
                    bit = scan & -scan
                    scan ^= bit
                    v = bit.bit_length() - 1
                    conflicts = adj[v] & selected
                    count = conflicts.bit_count()
                    if count == 0:
                        selected |= bit
                        selected = greedy_fill(selected, kernel, 0.0, stop_at)
                        break
                    if count == 1:
                        owner = conflicts.bit_length() - 1
                        tight_one[owner] = tight_one.get(owner, 0) | bit
                    elif count == 2:
                        low = conflicts & -conflicts
                        a = low.bit_length() - 1
                        b = (conflicts ^ low).bit_length() - 1
                        if a > b:
                            a, b = b, a
                        tight_two[(a, b)] = tight_two.get((a, b), 0) | bit
                    inspected += 1
                    if (inspected & 255) == 0 and time.time() >= stop_at:
                        return selected
                else:
                    moved = False
                    owners = list(tight_one)
                    owners.sort(
                        key=lambda u: tight_one[u].bit_count(), reverse=True
                    )
                    for owner in owners:
                        pair_bits = independent_pair(tight_one[owner], stop_at)
                        if pair_bits:
                            selected &= ~(1 << owner)
                            selected |= pair_bits
                            selected = greedy_fill(selected, kernel, 0.0, stop_at)
                            moved = True
                            break
                        if time.time() >= stop_at:
                            return selected
                    if moved:
                        continue

                    pair_items = list(tight_two.items())
                    pair_items.sort(
                        key=lambda item: item[1].bit_count(), reverse=True
                    )
                    for pair_key, group in pair_items:
                        a, b = pair_key
                        candidates = (
                            group
                            | tight_one.get(a, 0)
                            | tight_one.get(b, 0)
                        )
                        triple = independent_triple(candidates, stop_at)
                        if triple:
                            selected &= ~((1 << a) | (1 << b))
                            selected |= triple
                            selected = greedy_fill(selected, kernel, 0.0, stop_at)
                            moved = True
                            break
                        if time.time() >= stop_at:
                            return selected
                    if not moved:
                        return selected
            return selected

        # First phase: ARW-style independent-set search.  Strict exchanges are
        # interleaved with plateau swaps, force-in kicks, and broad ejections.
        search_stop = min(deadline, t0 + budget * 0.62)
        kernel_best = best & kernel
        seed = local_improve(greedy_fill(0, kernel, 0.0, search_stop), search_stop)
        if seed.bit_count() > kernel_best.bit_count():
            kernel_best = seed
            candidate = exact_fixed | kernel_best
            if candidate.bit_count() > best.bit_count():
                best = candidate
        walk = seed
        stagnation = 0

        while time.time() < search_stop:
            outside = kernel & ~walk
            one_tight = []
            forceable = []
            scan = outside
            while scan and time.time() < search_stop:
                bit = scan & -scan
                scan ^= bit
                conflicts = adj[bit.bit_length() - 1] & walk
                count = conflicts.bit_count()
                if count == 1:
                    one_tight.append((bit, conflicts))
                elif 2 <= count <= 8:
                    forceable.append((bit, conflicts))

            roll = rng.random()
            if one_tight and roll < 0.34:
                # A neutral 1-for-1 move walks to another basin without losing
                # cardinality; local_improve then tests its new tight groups.
                bit, conflicts = rng.choice(one_tight)
                candidate_kernel = (walk & ~conflicts) | bit
                candidate_kernel = greedy_fill(
                    candidate_kernel, kernel, 0.45, search_stop
                )
            elif forceable and roll < 0.73:
                bit, conflicts = rng.choice(forceable)
                candidate_kernel = (walk & ~conflicts) | bit
                candidate_kernel = greedy_fill(
                    candidate_kernel, kernel, 0.75, search_stop
                )
            elif roll < 0.92 and kernel_best:
                selected_vertices = []
                scan = kernel_best
                while scan:
                    bit = scan & -scan
                    scan ^= bit
                    selected_vertices.append(bit)
                width = 2 + min(12, stagnation // 7)
                width = min(width, len(selected_vertices))
                removed = 0
                if width:
                    for bit in rng.sample(selected_vertices, width):
                        removed |= bit
                candidate_kernel = greedy_fill(
                    kernel_best & ~removed, kernel, 1.15, search_stop
                )
            else:
                candidate_kernel = greedy_fill(
                    0, kernel, 0.20 + 1.35 * rng.random(), search_stop
                )

            candidate_kernel = local_improve(candidate_kernel, search_stop)
            walk = candidate_kernel
            if candidate_kernel.bit_count() > kernel_best.bit_count():
                kernel_best = candidate_kernel
                stagnation = 0
                full_candidate = exact_fixed | kernel_best
                if full_candidate.bit_count() > best.bit_count():
                    best = full_candidate
            else:
                stagnation += 1
                # Periodically return the walk to the best basin, while most
                # neutral/worse moves are retained to cross local plateaus.
                if rng.random() < min(0.35, 0.015 * stagnation):
                    walk = kernel_best

        if time.time() >= deadline:
            return answer(best)

        # Second phase: search the same kernel as a weighted minimum vertex
        # cover.  It has a different neighborhood and can cross plateaus that
        # defeat bounded independent-set exchanges.
        edge_list = []
        incident = [[] for _ in range(n)]
        scan = kernel
        while scan and time.time() < deadline:
            bit = scan & -scan
            scan ^= bit
            u = bit.bit_length() - 1
            higher = adj[u] & kernel & ~((1 << (u + 1)) - 1)
            while higher:
                vbit = higher & -higher
                higher ^= vbit
                v = vbit.bit_length() - 1
                eid = len(edge_list)
                edge_list.append((u, v))
                incident[u].append(eid)
                incident[v].append(eid)

        if not edge_list or time.time() >= deadline:
            return answer(best)

        cover = kernel & ~kernel_best
        weights = [1] * len(edge_list)
        dscore = [0] * n
        uncovered = set()
        age = [0] * n

        def rebuild_cover_state():
            nonlocal dscore, uncovered
            dscore = [0] * n
            uncovered = set()
            for eid, edge in enumerate(edge_list):
                u, v = edge
                u_in = bool(cover & (1 << u))
                v_in = bool(cover & (1 << v))
                w = weights[eid]
                if not u_in and not v_in:
                    uncovered.add(eid)
                    dscore[u] += w
                    dscore[v] += w
                elif u_in and not v_in:
                    dscore[u] -= w
                elif not u_in and v_in:
                    dscore[v] -= w

        def remove_cover_vertex(v):
            nonlocal cover
            bit = 1 << v
            if not (cover & bit):
                return
            old = dscore[v]
            cover &= ~bit
            dscore[v] = -old
            for eid in incident[v]:
                u, wv = edge_list[eid]
                other = wv if u == v else u
                weight = weights[eid]
                if cover & (1 << other):
                    dscore[other] -= weight
                else:
                    dscore[other] += weight
                    uncovered.add(eid)

        def add_cover_vertex(v):
            nonlocal cover
            bit = 1 << v
            if cover & bit:
                return
            old = dscore[v]
            cover |= bit
            dscore[v] = -old
            for eid in incident[v]:
                u, wv = edge_list[eid]
                other = wv if u == v else u
                weight = weights[eid]
                if cover & (1 << other):
                    dscore[other] += weight
                else:
                    dscore[other] -= weight
                    uncovered.discard(eid)

        rebuild_cover_state()
        step = 0
        last_added = -1
        total_weight = len(edge_list)

        while time.time() < deadline:
            step += 1
            if not uncovered:
                independent = kernel & ~cover
                if independent.bit_count() > kernel_best.bit_count():
                    kernel_best = independent
                    full_candidate = exact_fixed | kernel_best
                    if full_candidate.bit_count() > best.bit_count():
                        best = full_candidate

                # Lower the target cover size by removing its least costly
                # member; redundant members can yield immediate improvements.
                choices = cover
                if not choices:
                    break
                chosen = -1
                chosen_key = None
                scan = choices
                while scan:
                    bit = scan & -scan
                    scan ^= bit
                    v = bit.bit_length() - 1
                    key = (dscore[v], -age[v], -kernel_degrees[v])
                    if chosen_key is None or key > chosen_key:
                        chosen_key = key
                        chosen = v
                remove_cover_vertex(chosen)
                age[chosen] = step
                continue

            # Keep the current target cardinality while replacing one cover
            # vertex.  The just-added vertex is tabu for one move.
            choices = cover
            if last_added >= 0 and choices & ~(1 << last_added):
                choices &= ~(1 << last_added)
            remove_v = -1
            remove_key = None
            scan = choices
            inspected = 0
            while scan:
                bit = scan & -scan
                scan ^= bit
                v = bit.bit_length() - 1
                key = (dscore[v], -age[v], -kernel_degrees[v])
                if remove_key is None or key > remove_key:
                    remove_key = key
                    remove_v = v
                inspected += 1
                if (inspected & 255) == 0 and time.time() >= deadline:
                    break
            if remove_v < 0 or time.time() >= deadline:
                break
            remove_cover_vertex(remove_v)
            age[remove_v] = step

            if not uncovered:
                last_added = -1
                continue
            edge_id = rng.choice(tuple(uncovered))
            u, v = edge_list[edge_id]
            if u == remove_v and v != remove_v:
                add_v = v
            elif v == remove_v and u != remove_v:
                add_v = u
            else:
                u_key = (dscore[u], -age[u], rng.random())
                v_key = (dscore[v], -age[v], rng.random())
                add_v = u if u_key >= v_key else v
            add_cover_vertex(add_v)
            age[add_v] = step
            last_added = add_v

            # NuMVC-style edge weighting focuses future repairs on conflicts
            # that persist.  Periodic forgetting prevents weights exploding.
            pending = tuple(uncovered)
            for eid in pending:
                weights[eid] += 1
                total_weight += 1
                a, b = edge_list[eid]
                dscore[a] += 1
                dscore[b] += 1
                if time.time() >= deadline:
                    break
            if total_weight > 48 * len(edge_list) and time.time() < deadline:
                total_weight = 0
                for eid in range(len(weights)):
                    weights[eid] = max(1, weights[eid] // 3)
                    total_weight += weights[eid]
                rebuild_cover_state()

        return answer(best)
    except Exception:
        # best is updated only with complete independent sets.  Even malformed
        # input or an unexpected internal error therefore returns feasibly.
        return answer(best)
