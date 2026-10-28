# Library slot 5/10  id=alg_5a0c914e0af2
# score=-164.1  feasible_rate=1.0
# score_vec=[-369.0, -128.0, -100.0, -80.0, -112.0, -55.0, -118.0, -99.0, -442.0, -138.0]

import random
import time
from collections import deque


def solve(instance: dict, time_limit_s: float) -> dict:
    """Anytime maximum-independent-set solver using reduction and local search."""
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
    zero_result = {"vertex_selection": [0] * n}
    if n == 0:
        return zero_result

    best_mask = 0

    try:
        adjacency = [0] * n
        forbidden = 0
        edge_fingerprint = n * 1000003
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
                forbidden |= 1 << u
                continue
            ub, vb = 1 << u, 1 << v
            adjacency[u] |= vb
            adjacency[v] |= ub
            edge_fingerprint = (
                edge_fingerprint * 1000003 + (u + 1) * 9176 + (v + 1) * 6361
            ) & ((1 << 63) - 1)

        full_mask = (1 << n) - 1
        valid_mask = full_mask & ~forbidden

        # Immediate complete fallback: static-degree greedy.  It is available
        # before any expensive reduction or search begins.
        order = []
        vertices = valid_mask
        while vertices:
            bit = vertices & -vertices
            v = bit.bit_length() - 1
            order.append(v)
            vertices ^= bit
        order.sort(key=lambda v: (adjacency[v] & valid_mask).bit_count())
        for v in order:
            bit = 1 << v
            if not (adjacency[v] & best_mask):
                best_mask |= bit
        best_size = best_mask.bit_count()

        if time.time() >= deadline:
            return {
                "vertex_selection": [int(bool(best_mask & (1 << v))) for v in range(n)]
            }

        rng = random.Random(edge_fingerprint)

        # Exact, safe kernelization rules.  Isolated vertices and leaves can
        # always be selected; the same is true for a degree-two vertex whose
        # two neighbors are adjacent.
        active = valid_mask
        fixed = 0
        degrees = [(adjacency[v] & active).bit_count() for v in range(n)]
        queue = deque(v for v in range(n) if active & (1 << v) and degrees[v] <= 2)

        def remove_active(bits):
            nonlocal active
            removed = active & bits
            active &= ~removed
            scan = removed
            while scan:
                bit = scan & -scan
                v = bit.bit_length() - 1
                affected = adjacency[v] & active
                while affected:
                    wbit = affected & -affected
                    w = wbit.bit_length() - 1
                    degrees[w] -= 1
                    if degrees[w] <= 2:
                        queue.append(w)
                    affected ^= wbit
                degrees[v] = 0
                scan ^= bit

        while queue and time.time() < deadline:
            v = queue.popleft()
            vbit = 1 << v
            if not (active & vbit):
                continue
            neighbors = adjacency[v] & active
            degree = neighbors.bit_count()
            degrees[v] = degree
            take_v = degree <= 1
            if degree == 2:
                first = neighbors & -neighbors
                a = first.bit_length() - 1
                second = neighbors ^ first
                take_v = bool(adjacency[a] & second)
            if take_v:
                fixed |= vbit
                remove_active(vbit | neighbors)

        reduced_best = fixed

        def update_best(kernel_selection):
            nonlocal best_mask, best_size, reduced_best
            candidate = fixed | (kernel_selection & active)
            candidate_size = candidate.bit_count()
            if candidate_size > reduced_best.bit_count():
                reduced_best = candidate
            if candidate_size > best_size:
                best_mask = candidate
                best_size = candidate_size

        def blocked_by(selection):
            blocked = selection
            scan = selection
            while scan:
                bit = scan & -scan
                blocked |= adjacency[bit.bit_length() - 1]
                scan ^= bit
            return blocked

        def greedy_fill(selection, mode, until):
            selection &= active
            candidates = active & ~blocked_by(selection)
            while candidates and time.time() < until:
                scan = candidates
                chosen_bit = 0
                chosen_score = None
                inspected = 0
                while scan:
                    bit = scan & -scan
                    v = bit.bit_length() - 1
                    degree = (adjacency[v] & candidates).bit_count()
                    if mode == 0:
                        score = (degree, (adjacency[v] & active).bit_count(), v)
                    elif mode == 1:
                        score = degree + rng.random() * (1.25 + 0.12 * degree)
                    else:
                        # A wider randomized ordering is valuable after an
                        # ejection, where recreating the same greedy choices
                        # would simply return to the previous local optimum.
                        score = degree + rng.random() * (3.0 + 0.25 * degree)
                    if chosen_score is None or score < chosen_score:
                        chosen_score = score
                        chosen_bit = bit
                    scan ^= bit
                    inspected += 1
                    if inspected & 63 == 0 and time.time() >= until:
                        break
                if not chosen_bit:
                    break
                v = chosen_bit.bit_length() - 1
                selection |= chosen_bit
                candidates &= ~(chosen_bit | adjacency[v])
            return selection

        def first_independent_pair(bits):
            scan = bits
            while scan:
                abit = scan & -scan
                a = abit.bit_length() - 1
                scan ^= abit
                partners = scan & ~adjacency[a]
                if partners:
                    return abit, partners & -partners
            return 0, 0

        def first_independent_triple(bits, until):
            scan = bits
            checked = 0
            while scan and time.time() < until:
                abit = scan & -scan
                a = abit.bit_length() - 1
                scan ^= abit
                compatible = scan & ~adjacency[a]
                second_scan = compatible
                while second_scan:
                    bbit = second_scan & -second_scan
                    b = bbit.bit_length() - 1
                    second_scan ^= bbit
                    third = second_scan & ~adjacency[b]
                    if third:
                        return abit, bbit, third & -third
                    checked += 1
                    if checked & 63 == 0 and time.time() >= until:
                        return 0, 0, 0
            return 0, 0, 0

        def local_improve(selection, until):
            selection = greedy_fill(selection, 0, until)
            while time.time() < until:
                one_tight = [0] * n
                two_tight = {}
                outside = active & ~selection
                scanned = 0
                while outside:
                    bit = outside & -outside
                    v = bit.bit_length() - 1
                    conflicts = adjacency[v] & selection
                    count = conflicts.bit_count()
                    if count == 1:
                        u = conflicts.bit_length() - 1
                        one_tight[u] |= bit
                    elif count == 2:
                        first = conflicts & -conflicts
                        u = first.bit_length() - 1
                        w = (conflicts ^ first).bit_length() - 1
                        key = (u, w) if u < w else (w, u)
                        two_tight[key] = two_tight.get(key, 0) | bit
                    outside ^= bit
                    scanned += 1
                    if scanned & 127 == 0 and time.time() >= until:
                        return selection

                # Exhaust the complete 1-for-2 improving neighborhood.
                improved = False
                selected_scan = selection
                while selected_scan and time.time() < until:
                    ubit = selected_scan & -selected_scan
                    u = ubit.bit_length() - 1
                    first, second = first_independent_pair(one_tight[u])
                    if first:
                        selection = (selection & ~ubit) | first | second
                        selection = greedy_fill(selection, 0, until)
                        update_best(selection)
                        improved = True
                        break
                    selected_scan ^= ubit
                if improved:
                    continue

                # Then exhaust 2-for-3 exchanges.  Candidate pools contain
                # precisely vertices freed by removing the selected pair.
                useful = []
                selected_scan = selection
                while selected_scan:
                    bit = selected_scan & -selected_scan
                    u = bit.bit_length() - 1
                    if one_tight[u]:
                        useful.append(u)
                    selected_scan ^= bit

                pair_keys = []
                seen_pairs = set()
                for key in two_tight:
                    pair_keys.append(key)
                    seen_pairs.add(key)
                for i in range(len(useful)):
                    u = useful[i]
                    for j in range(i + 1, len(useful)):
                        w = useful[j]
                        key = (u, w) if u < w else (w, u)
                        if key not in seen_pairs and (
                            one_tight[u] | one_tight[w]
                        ).bit_count() >= 3:
                            pair_keys.append(key)
                            seen_pairs.add(key)
                    if time.time() >= until:
                        return selection

                for index, key in enumerate(pair_keys):
                    if index & 31 == 0 and time.time() >= until:
                        return selection
                    u, w = key
                    pool = one_tight[u] | one_tight[w] | two_tight.get(key, 0)
                    if pool.bit_count() < 3:
                        continue
                    a, b, c = first_independent_triple(pool, until)
                    if a:
                        selection &= ~((1 << u) | (1 << w))
                        selection |= a | b | c
                        selection = greedy_fill(selection, 0, until)
                        update_best(selection)
                        improved = True
                        break
                if not improved:
                    return selection
            return selection

        # Build a strong reduced incumbent before entering the longer search.
        quick_until = min(deadline, time.time() + max(0.002, budget * 0.05))
        initial = greedy_fill(0, 0, quick_until)
        update_best(initial)

        def random_bit(bits):
            count = bits.bit_count()
            if count == 0:
                return 0
            target = rng.randrange(count)
            scan = bits
            while target:
                scan ^= scan & -scan
                target -= 1
            return scan & -scan

        def perturb(selection, strength, until):
            selection &= active
            outside = active & ~selection
            if not outside:
                removed = random_bit(selection)
                return greedy_fill(selection & ~removed, 2, until)

            desired_conflicts = 1
            roll = rng.random()
            if strength >= 5 and roll < 0.25:
                desired_conflicts = 3
            elif strength >= 3 and roll < 0.55:
                desired_conflicts = 2

            chosen = 0
            chosen_score = None
            fallback = 0
            fallback_score = None
            scan = outside
            inspected = 0
            while scan:
                bit = scan & -scan
                v = bit.bit_length() - 1
                conflicts = (adjacency[v] & selection).bit_count()
                degree = (adjacency[v] & active).bit_count()
                score = degree + rng.random() * 8.0
                if conflicts == desired_conflicts and (
                    chosen_score is None or score < chosen_score
                ):
                    chosen, chosen_score = bit, score
                broad_score = conflicts * 1000 + score
                if conflicts and (fallback_score is None or broad_score < fallback_score):
                    fallback, fallback_score = bit, broad_score
                scan ^= bit
                inspected += 1
                if inspected & 127 == 0 and time.time() >= until:
                    break
            pivot = chosen or fallback or random_bit(outside)
            if not pivot:
                return selection
            v = pivot.bit_length() - 1
            selection &= ~adjacency[v]
            selection |= pivot

            # Increasing ejection strength after stagnation opens a larger
            # neighborhood without ever risking feasibility.
            extra = max(0, strength // 3 - 1)
            while extra and selection and time.time() < until:
                removable = selection & ~pivot
                bit = random_bit(removable)
                if not bit:
                    break
                selection ^= bit
                extra -= 1
            return greedy_fill(selection, 2, until)

        def ils_until(until, current, stagnation):
            current = local_improve(current, until)
            update_best(current)
            while time.time() < until:
                strength = 1 + min(8, stagnation // 3)
                if rng.random() < 0.16:
                    candidate = greedy_fill(0, 1 + int(rng.random() < 0.5), until)
                else:
                    base = current if current.bit_count() + 3 >= reduced_best.bit_count() else (reduced_best & active)
                    candidate = perturb(base, strength, until)
                candidate = local_improve(candidate, until)
                previous_best = best_size
                update_best(candidate)
                if best_size > previous_best:
                    stagnation = 0
                    current = candidate
                else:
                    stagnation += 1
                    tolerance = min(4, 1 + stagnation // 6)
                    if candidate.bit_count() + tolerance >= (reduced_best & active).bit_count():
                        current = candidate
                    else:
                        current = reduced_best & active
            return current, stagnation

        search_span = max(0.0, deadline - t0)
        heuristic_end = min(deadline, t0 + search_span * 0.58)
        current, stagnation = ils_until(heuristic_end, initial, 0)

        # A bounded exact maximum-clique search on the complement is a useful
        # intensification for small/medium kernels.  It starts from the local
        # search incumbent and is itself interruptible at every recursion and
        # coloring pass.
        exact_end = min(deadline, t0 + search_span * 0.84)
        kernel_size = active.bit_count()
        if kernel_size and kernel_size <= 450 and time.time() < exact_end:
            kernel_vertices = []
            scan = active
            while scan:
                bit = scan & -scan
                kernel_vertices.append(bit.bit_length() - 1)
                scan ^= bit
            kernel_vertices.sort(key=lambda v: (adjacency[v] & active).bit_count())
            local_n = len(kernel_vertices)
            complement = [0] * local_n
            for i, v in enumerate(kernel_vertices):
                row = 0
                for j, w in enumerate(kernel_vertices):
                    if i != j and not (adjacency[v] & (1 << w)):
                        row |= 1 << j
                complement[i] = row
                if i & 15 == 0 and time.time() >= exact_end:
                    break

            if time.time() < exact_end:
                incumbent_bits = 0
                reduced_kernel = reduced_best & active
                for i, v in enumerate(kernel_vertices):
                    if reduced_kernel & (1 << v):
                        incumbent_bits |= 1 << i
                incumbent_size = incumbent_bits.bit_count()
                timed_out = False

                def commit_exact(local_selection):
                    global_selection = 0
                    scan_local = local_selection
                    while scan_local:
                        bit = scan_local & -scan_local
                        global_selection |= 1 << kernel_vertices[bit.bit_length() - 1]
                        scan_local ^= bit
                    update_best(global_selection)

                def color_sort(candidates):
                    nonlocal timed_out
                    order_local = []
                    bounds = []
                    uncolored = candidates
                    color = 0
                    while uncolored:
                        if time.time() >= exact_end:
                            timed_out = True
                            return order_local, bounds
                        color += 1
                        available = uncolored
                        while available:
                            bit = available & -available
                            v = bit.bit_length() - 1
                            order_local.append(v)
                            bounds.append(color)
                            uncolored ^= bit
                            available &= ~bit
                            available &= ~complement[v]
                    return order_local, bounds

                def expand(candidates, clique_size, clique_bits):
                    nonlocal incumbent_bits, incumbent_size, timed_out
                    if timed_out or time.time() >= exact_end:
                        timed_out = True
                        return
                    ordered, bounds = color_sort(candidates)
                    if timed_out:
                        return
                    for index in range(len(ordered) - 1, -1, -1):
                        if clique_size + bounds[index] <= incumbent_size:
                            return
                        v = ordered[index]
                        bit = 1 << v
                        if not (candidates & bit):
                            continue
                        next_bits = clique_bits | bit
                        next_candidates = candidates & complement[v]
                        if next_candidates:
                            expand(next_candidates, clique_size + 1, next_bits)
                        elif clique_size + 1 > incumbent_size:
                            incumbent_size = clique_size + 1
                            incumbent_bits = next_bits
                            commit_exact(incumbent_bits)
                        candidates &= ~bit
                        if timed_out:
                            return

                expand((1 << local_n) - 1, 0, 0)
                commit_exact(incumbent_bits)

        # Convert every remaining millisecond into additional diversified
        # neighborhoods after exact intensification finishes or times out.
        if time.time() < deadline:
            current, stagnation = ils_until(deadline, current, stagnation)

        result = [0] * n
        scan = best_mask
        while scan:
            bit = scan & -scan
            result[bit.bit_length() - 1] = 1
            scan ^= bit
        return {"vertex_selection": result}
    except Exception:
        # best_mask is always independent: it is either zero, the initial
        # greedy fallback, or a solution accepted by feasibility-preserving
        # moves.  Even an unexpected internal error therefore returns safely.
        result = [0] * n
        scan = best_mask
        while scan:
            bit = scan & -scan
            v = bit.bit_length() - 1
            if 0 <= v < n:
                result[v] = 1
            scan ^= bit
        return {"vertex_selection": result}
