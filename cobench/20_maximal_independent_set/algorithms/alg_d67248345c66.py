# Library slot 8/10  id=alg_d67248345c66
# score=-163.9  feasible_rate=1.0
# score_vec=[-369.0, -128.0, -100.0, -80.0, -111.0, -55.0, -117.0, -98.0, -442.0, -139.0]

import heapq
import random
import time
from collections import deque


def solve(instance: dict, time_limit_s: float) -> dict:
    """Anytime maximum-independent-set solver using exchange and recombination."""
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
    best_full = 0

    def make_answer(mask):
        return {
            "vertex_selection": [1 if (mask >> v) & 1 else 0 for v in range(n)]
        }

    if n == 0 or time.time() >= deadline:
        return make_answer(0)

    try:
        # Normalize the conflict graph. A loop forbids its endpoint under the
        # supplied checker, while malformed/out-of-range edges are irrelevant.
        adj_bits = [0] * n
        adj_list = [[] for _ in range(n)]
        banned = 0
        edge_fingerprint = n * 1000003
        for pos, edge in enumerate(instance.get("edges", [])):
            if (pos & 1023) == 0 and time.time() >= deadline:
                return make_answer(0)
            if not isinstance(edge, (list, tuple)) or len(edge) < 2:
                continue
            try:
                u, v = int(edge[0]), int(edge[1])
            except Exception:
                continue
            if not (0 <= u < n and 0 <= v < n):
                continue
            if u == v:
                banned |= 1 << u
                continue
            if (adj_bits[u] >> v) & 1:
                continue
            adj_bits[u] |= 1 << v
            adj_bits[v] |= 1 << u
            adj_list[u].append(v)
            adj_list[v].append(u)
            edge_fingerprint = (
                edge_fingerprint * 1000003 + (u + 1) * 9176 + (v + 1) * 6361
            ) & ((1 << 63) - 1)

        universe = ((1 << n) - 1) & ~banned
        if universe == 0:
            return make_answer(0)
        degree = [len(adj_list[v]) for v in range(n)]
        rng = random.Random(edge_fingerprint)

        def vertices(mask):
            while mask:
                bit = mask & -mask
                yield bit.bit_length() - 1
                mask ^= bit

        def heap_greedy(allowed, noisy):
            """Dynamic minimum-residual-degree construction in O((V+E)logV)."""
            active = [False] * n
            residual_degree = [0] * n
            heap = []
            for v in vertices(allowed):
                active[v] = True
            for v in vertices(allowed):
                d = (adj_bits[v] & allowed).bit_count()
                residual_degree[v] = d
                noise = rng.random() * (2.5 if noisy else 0.0)
                heapq.heappush(heap, (d + noise, d, rng.random(), v))

            chosen = 0
            ticks = 0
            while heap and time.time() < deadline:
                _, old_d, _, v = heapq.heappop(heap)
                if not active[v] or old_d != residual_degree[v]:
                    continue
                bit_v = 1 << v
                chosen |= bit_v
                removed = allowed & (adj_bits[v] | bit_v)
                allowed &= ~removed
                for r in vertices(removed):
                    active[r] = False
                    for w in adj_list[r]:
                        if active[w]:
                            residual_degree[w] -= 1
                            d = residual_degree[w]
                            noise = rng.random() * (2.5 if noisy else 0.0)
                            heapq.heappush(heap, (d + noise, d, rng.random(), w))
                ticks += 1
                if (ticks & 31) == 0 and time.time() >= deadline:
                    break
            return chosen

        # The first complete feasible incumbent is established before structural
        # analysis or deeper search consumes meaningful time.
        best_full = heap_greedy(universe, False)
        if time.time() >= deadline:
            return make_answer(best_full)

        degree_order = sorted(vertices(universe), key=lambda v: (degree[v], v))

        def maximalize(selected, allowed, noisy=False):
            free = allowed & ~selected
            possible = 0
            for v in vertices(free):
                if not (adj_bits[v] & selected):
                    possible |= 1 << v
            if noisy and possible.bit_count() > 1:
                order = list(vertices(possible))
                order.sort(key=lambda v: degree[v] + rng.random() * 3.5)
            else:
                order = degree_order
            for pos, v in enumerate(order):
                bit_v = 1 << v
                if possible & bit_v:
                    selected |= bit_v
                    possible &= ~(adj_bits[v] | bit_v)
                if (pos & 127) == 0 and time.time() >= deadline:
                    break
            return selected

        # A bounded structural pass removes solved components from the heuristic
        # search. It has a small budget so it can never starve the anytime core.
        structure_deadline = min(deadline, t0 + budget * 0.20)
        solved_mask = 0
        hard_mask = 0
        unseen = universe

        def bipartite_independent(left_mask, right_mask, cutoff):
            left = list(vertices(left_mask))
            pair_left = [-1] * n
            pair_right = [-1] * n

            # Repeated shortest augmenting paths. This iterative version avoids
            # recursion-depth failures on long alternating paths.
            for start in left:
                if time.time() >= cutoff:
                    return 0, False
                if pair_left[start] >= 0:
                    continue
                q = deque([start])
                seen_left = {start}
                previous_right = {}
                finish = -1
                while q and finish < 0:
                    u = q.popleft()
                    for v in adj_list[u]:
                        if not ((right_mask >> v) & 1) or v in previous_right:
                            continue
                        previous_right[v] = u
                        mate = pair_right[v]
                        if mate < 0:
                            finish = v
                            break
                        if mate not in seen_left:
                            seen_left.add(mate)
                            q.append(mate)
                    if time.time() >= cutoff:
                        return 0, False
                if finish >= 0:
                    v = finish
                    while v >= 0:
                        u = previous_right[v]
                        old_v = pair_left[u]
                        pair_left[u] = v
                        pair_right[v] = u
                        v = old_v

            # Complement of a Konig minimum vertex cover.
            z_left = 0
            z_right = 0
            q = deque()
            for u in left:
                if pair_left[u] < 0:
                    z_left |= 1 << u
                    q.append(u)
            while q:
                if time.time() >= cutoff:
                    return 0, False
                u = q.popleft()
                for v in adj_list[u]:
                    bit_v = 1 << v
                    if not (right_mask & bit_v) or pair_left[u] == v or z_right & bit_v:
                        continue
                    z_right |= bit_v
                    mate = pair_right[v]
                    if mate >= 0 and not (z_left >> mate) & 1:
                        z_left |= 1 << mate
                        q.append(mate)
            return z_left | (right_mask & ~z_right), True

        def exact_small_component(comp, initial, cutoff):
            """Tomita-style maximum clique search on the complement graph."""
            size = len(comp)
            index = {v: i for i, v in enumerate(comp)}
            local_all = (1 << size) - 1
            complement = [0] * size
            for i, v in enumerate(comp):
                conflicts = 0
                for w in adj_list[v]:
                    j = index.get(w)
                    if j is not None:
                        conflicts |= 1 << j
                complement[i] = local_all & ~(conflicts | (1 << i))

            incumbent = 0
            for i, v in enumerate(comp):
                if (initial >> v) & 1:
                    incumbent |= 1 << i
            best_local = incumbent
            best_size = incumbent.bit_count()
            aborted = False

            def color_sort(candidates):
                order = []
                bounds = []
                remaining = candidates
                color = 0
                while remaining:
                    color += 1
                    q = remaining
                    while q:
                        bit = q & -q
                        v = bit.bit_length() - 1
                        order.append(v)
                        bounds.append(color)
                        remaining ^= bit
                        q &= ~bit
                        q &= ~complement[v]
                return order, bounds

            def expand(clique, candidates):
                nonlocal best_local, best_size, aborted
                if time.time() >= cutoff:
                    aborted = True
                    return
                order, bounds = color_sort(candidates)
                for pos in range(len(order) - 1, -1, -1):
                    if clique.bit_count() + bounds[pos] <= best_size:
                        return
                    v = order[pos]
                    bit_v = 1 << v
                    if not (candidates & bit_v):
                        continue
                    grown = clique | bit_v
                    next_candidates = candidates & complement[v]
                    if next_candidates:
                        expand(grown, next_candidates)
                        if aborted:
                            return
                    elif grown.bit_count() > best_size:
                        best_local = grown
                        best_size = grown.bit_count()
                    candidates &= ~bit_v

            expand(0, local_all)
            answer = 0
            for i in vertices(best_local):
                answer |= 1 << comp[i]
            return answer, not aborted

        while unseen and time.time() < structure_deadline:
            root_bit = unseen & -unseen
            root = root_bit.bit_length() - 1
            unseen ^= root_bit
            q = deque([root])
            comp = []
            comp_mask = root_bit
            colors = {root: 0}
            left_mask = 0
            right_mask = 0
            is_bipartite = True
            interrupted = False
            while q:
                v = q.popleft()
                comp.append(v)
                if colors[v] == 0:
                    left_mask |= 1 << v
                else:
                    right_mask |= 1 << v
                for w in adj_list[v]:
                    bit_w = 1 << w
                    if not (universe & bit_w):
                        continue
                    if w not in colors:
                        colors[w] = colors[v] ^ 1
                        unseen &= ~bit_w
                        comp_mask |= bit_w
                        q.append(w)
                    elif colors[w] == colors[v]:
                        is_bipartite = False
                if time.time() >= structure_deadline:
                    interrupted = True
                    break
            if interrupted:
                hard_mask |= comp_mask | unseen
                unseen = 0
                break

            if is_bipartite:
                answer, complete = bipartite_independent(
                    left_mask, right_mask, structure_deadline
                )
                if complete:
                    solved_mask |= answer
                else:
                    hard_mask |= comp_mask
            elif len(comp) <= 72:
                answer, complete = exact_small_component(
                    comp, best_full & comp_mask, structure_deadline
                )
                if complete:
                    solved_mask |= answer
                else:
                    hard_mask |= comp_mask
            else:
                hard_mask |= comp_mask
        hard_mask |= unseen

        current = best_full & hard_mask
        current = maximalize(current, hard_mask)
        candidate_full = solved_mask | current
        if candidate_full.bit_count() > best_full.bit_count():
            best_full = candidate_full
        if time.time() >= deadline or hard_mask == 0:
            return make_answer(best_full)

        def tight_sets(selected):
            one = {}
            two = {}
            three = {}
            zero = 0
            outside = hard_mask & ~selected
            for pos, v in enumerate(vertices(outside)):
                blockers = adj_bits[v] & selected
                count = blockers.bit_count()
                if count == 0:
                    zero |= 1 << v
                elif count == 1:
                    u = blockers.bit_length() - 1
                    one[u] = one.get(u, 0) | (1 << v)
                elif count == 2:
                    first_bit = blockers & -blockers
                    u = first_bit.bit_length() - 1
                    w = (blockers ^ first_bit).bit_length() - 1
                    two[(u, w)] = two.get((u, w), 0) | (1 << v)
                elif count == 3:
                    blocker_vertices = tuple(vertices(blockers))
                    three[blocker_vertices] = three.get(blocker_vertices, 0) | (1 << v)
                if (pos & 255) == 0 and time.time() >= deadline:
                    break
            return one, two, three, zero

        def nonadjacent_pair(mask):
            order = list(vertices(mask))
            order.sort(key=lambda v: degree[v] + rng.random() * 2.0)
            remaining = mask
            for a in order:
                bit_a = 1 << a
                remaining &= ~bit_a
                choices = remaining & ~adj_bits[a]
                if choices:
                    best_b = -1
                    best_score = n + 1
                    for b in vertices(choices):
                        score = degree[b]
                        if score < best_score:
                            best_score = score
                            best_b = b
                    if best_b >= 0:
                        return a, best_b
                if time.time() >= deadline:
                    break
            return None

        def nonadjacent_triple(mask):
            order = list(vertices(mask))
            order.sort(key=lambda v: degree[v] + rng.random() * 2.5)
            remaining_a = mask
            for a in order:
                bit_a = 1 << a
                remaining_a &= ~bit_a
                possible_b = remaining_a & ~adj_bits[a]
                while possible_b:
                    bit_b = possible_b & -possible_b
                    b = bit_b.bit_length() - 1
                    possible_c = possible_b & ~((1 << (b + 1)) - 1)
                    possible_c &= ~adj_bits[b]
                    if possible_c:
                        bit_c = possible_c & -possible_c
                        return a, b, bit_c.bit_length() - 1
                    possible_b ^= bit_b
                    if time.time() >= deadline:
                        return None
                if time.time() >= deadline:
                    return None
            return None

        def nonadjacent_quad(mask, cutoff):
            order = list(vertices(mask))
            order.sort(key=lambda v: degree[v] + rng.random() * 3.0)
            remaining_a = mask
            for a in order:
                bit_a = 1 << a
                remaining_a &= ~bit_a
                possible_b = remaining_a & ~adj_bits[a]
                while possible_b:
                    bit_b = possible_b & -possible_b
                    b = bit_b.bit_length() - 1
                    possible_c = possible_b & ~((1 << (b + 1)) - 1)
                    possible_c &= ~adj_bits[b]
                    while possible_c:
                        bit_c = possible_c & -possible_c
                        c = bit_c.bit_length() - 1
                        possible_d = possible_c & ~((1 << (c + 1)) - 1)
                        possible_d &= ~adj_bits[c]
                        if possible_d:
                            bit_d = possible_d & -possible_d
                            return a, b, c, bit_d.bit_length() - 1
                        possible_c ^= bit_c
                        if time.time() >= cutoff:
                            return None
                    possible_b ^= bit_b
                    if time.time() >= cutoff:
                        return None
                if time.time() >= cutoff:
                    return None
            return None

        def improve(selected):
            """Exhaust improving 1->2 and 2->3 exchanges."""
            selected = maximalize(selected, hard_mask)
            while time.time() < deadline:
                one, two, three, zero = tight_sets(selected)
                if zero:
                    selected = maximalize(selected, hard_mask)
                    continue

                blockers = list(one)
                rng.shuffle(blockers)
                blockers.sort(key=lambda u: -one[u].bit_count())
                move = None
                for u in blockers:
                    if one[u].bit_count() >= 2:
                        pair = nonadjacent_pair(one[u])
                        if pair is not None:
                            move = (u, pair[0], pair[1])
                            break
                    if time.time() >= deadline:
                        break
                if move is not None:
                    u, a, b = move
                    selected &= ~(1 << u)
                    selected |= (1 << a) | (1 << b)
                    selected = maximalize(selected, hard_mask, True)
                    continue

                pairs = list(two)
                rng.shuffle(pairs)
                pairs.sort(
                    key=lambda pair: -(
                        two[pair]
                        | one.get(pair[0], 0)
                        | one.get(pair[1], 0)
                    ).bit_count()
                )
                triple_move = None
                for u, w in pairs:
                    candidates = two[(u, w)] | one.get(u, 0) | one.get(w, 0)
                    if candidates.bit_count() >= 3:
                        triple = nonadjacent_triple(candidates)
                        if triple is not None:
                            triple_move = (u, w, triple)
                            break
                    if time.time() >= deadline:
                        break
                if triple_move is not None:
                    u, w, triple = triple_move
                    selected &= ~((1 << u) | (1 << w))
                    for v in triple:
                        selected |= 1 << v
                    selected = maximalize(selected, hard_mask, True)
                    continue

                # A short, wall-clock-bounded 3->4 probe catches deeper gains
                # without letting a combinatorial neighborhood starve restarts.
                probe_cutoff = min(deadline, time.time() + max(0.002, budget * 0.006))
                triplets = set(three)
                incident = {}
                for u, w in two:
                    incident.setdefault(u, set()).add(w)
                    incident.setdefault(w, set()).add(u)
                for center, neighbors in incident.items():
                    ordered = list(neighbors)
                    for i in range(len(ordered)):
                        for j in range(i + 1, len(ordered)):
                            triplets.add(tuple(sorted((center, ordered[i], ordered[j]))))
                            if time.time() >= probe_cutoff:
                                break
                        if time.time() >= probe_cutoff:
                            break
                    if time.time() >= probe_cutoff:
                        break

                quad_move = None
                triplet_order = list(triplets)
                rng.shuffle(triplet_order)
                for key in triplet_order:
                    u, w, z = key
                    candidates = three.get(key, 0)
                    candidates |= one.get(u, 0) | one.get(w, 0) | one.get(z, 0)
                    candidates |= two.get(tuple(sorted((u, w))), 0)
                    candidates |= two.get(tuple(sorted((u, z))), 0)
                    candidates |= two.get(tuple(sorted((w, z))), 0)
                    if candidates.bit_count() >= 4:
                        quad = nonadjacent_quad(candidates, probe_cutoff)
                        if quad is not None:
                            quad_move = (key, quad)
                            break
                    if time.time() >= probe_cutoff:
                        break
                if quad_move is not None:
                    key, quad = quad_move
                    for v in key:
                        selected &= ~(1 << v)
                    for v in quad:
                        selected |= 1 << v
                    selected = maximalize(selected, hard_mask, True)
                    continue
                break
            return selected

        def combine(first, second):
            """Exact bipartite recombination of two independent-set parents."""
            common = first & second
            left = first & ~second
            right = second & ~first
            if not left or not right:
                return maximalize(first if first.bit_count() >= second.bit_count() else second,
                                   hard_mask, True)
            merged, complete = bipartite_independent(left, right, deadline)
            if not complete:
                return first if first.bit_count() >= second.bit_count() else second
            return maximalize(common | merged, hard_mask, True)

        def perturb(selected, strength):
            one, two, _, _ = tight_sets(selected)
            roll = rng.random()
            if one and roll < 0.55:
                options = []
                for blocker, mask in one.items():
                    for v in vertices(mask):
                        options.append((degree[v] + rng.random() * 5.0, blocker, v))
                if options:
                    _, blocker, v = min(options)
                    selected &= ~(1 << blocker)
                    selected |= 1 << v
                    return maximalize(selected, hard_mask, True)
            if two and roll < 0.88:
                options = []
                for pair, mask in two.items():
                    for v in vertices(mask):
                        options.append((degree[v] + rng.random() * 6.0, pair, v))
                if options:
                    _, pair, v = min(options)
                    selected &= ~((1 << pair[0]) | (1 << pair[1]))
                    selected |= 1 << v
                    return maximalize(selected, hard_mask, True)

            chosen = list(vertices(selected))
            if not chosen:
                return heap_greedy(hard_mask, True)
            rng.shuffle(chosen)
            kick = 2 + min(6, strength // 7)
            for v in chosen[:kick]:
                selected &= ~(1 << v)
            return maximalize(selected, hard_mask, True)

        current = improve(current)
        best_hard = current
        candidate_full = solved_mask | best_hard
        if candidate_full.bit_count() > best_full.bit_count():
            best_full = candidate_full
        elites = [best_hard]
        stagnation = 0

        def remember(candidate):
            nonlocal elites
            if candidate in elites:
                return
            elites.append(candidate)
            elites.sort(key=int.bit_count, reverse=True)
            if len(elites) > 10:
                elites = elites[:10]

        # No iteration ceiling: restarts, exchanges, and exact recombinations
        # continue until the wall-clock deadline.
        while time.time() < deadline:
            if len(elites) >= 2 and rng.random() < 0.22:
                first = elites[rng.randrange(min(4, len(elites)))]
                second = elites[rng.randrange(len(elites))]
                if first == second:
                    current = perturb(best_hard, stagnation)
                else:
                    current = combine(first, second)
            elif stagnation >= 18:
                current = heap_greedy(hard_mask, True)
                current = maximalize(current, hard_mask, True)
                stagnation = 0
            else:
                base = current
                if base.bit_count() + 3 < best_hard.bit_count():
                    base = best_hard
                current = perturb(base, stagnation)

            if time.time() >= deadline:
                break
            current = improve(current)
            remember(current)
            if current.bit_count() > best_hard.bit_count():
                best_hard = current
                candidate_full = solved_mask | best_hard
                if candidate_full.bit_count() > best_full.bit_count():
                    best_full = candidate_full
                stagnation = 0
            else:
                stagnation += 1

        candidate_full = solved_mask | best_hard
        if candidate_full.bit_count() > best_full.bit_count():
            best_full = candidate_full
    except Exception:
        # best_full is always an already constructed independent set (or zero).
        pass

    return make_answer(best_full)
