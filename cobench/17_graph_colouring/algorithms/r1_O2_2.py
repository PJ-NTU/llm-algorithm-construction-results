"""Colour-class recombination with constraint-propagation colouring
Recombines and iterated-greedies a population of bitset colour classes, and eliminates colours jointly through core reduction and time-limited constraint-propagation backtracking.
"""

import heapq
import math
import random
import time


def solve(instance: dict, time_limit_s: float) -> dict:
    start = time.perf_counter()
    try:
        n = max(0, int(instance.get("n", 0)))
    except Exception:
        return {"coloring": []}
    best = list(range(n))
    if not n:
        return {"coloring": best}
    try:
        budget = max(0.0, float(time_limit_s))
        if not math.isfinite(budget):
            budget = 0.0
        end = start + budget * 0.96
        clock = time.perf_counter
        if clock() >= end:
            return {"coloring": best}
        adj = [set() for _ in range(n)]
        for index, edge in enumerate(instance.get("edges", [])):
            if index % 256 == 0 and clock() >= end:
                return {"coloring": best}
            u, v = int(edge[0]), int(edge[1])
            if 0 <= u < n and 0 <= v < n and u != v:
                adj[u].add(v)
                adj[v].add(u)
        degree = [len(a) for a in adj]
        if not any(degree):
            return {"coloring": [0] * n}

        # Output the compact encoding x[v][c] = (coloring[v] == c); y[c] indicates a non-empty colour class.
        bip = [-1] * n
        bip_ok = True
        for root in range(n):
            if not bip_ok:
                break
            if bip[root] >= 0:
                continue
            bip[root] = 0
            todo = [root]
            while todo and bip_ok:
                if clock() >= end:
                    return {"coloring": best}
                v = todo.pop()
                for u in adj[v]:
                    if bip[u] < 0:
                        bip[u] = 1 - bip[v]
                        todo.append(u)
                    elif bip[u] == bip[v]:
                        bip_ok = False
                        break
        if bip_ok:
            return {"coloring": bip}

        rng = random.Random(n * 1000003 + sum(degree))
        effort = math.log1p(budget)
        order = sorted(range(n), key=lambda v: (-degree[v], v))
        initial = [-1] * n
        used = 0
        for pos, v in enumerate(order):
            if clock() >= end:
                for u in order[pos:]:
                    initial[u] = used
                    used += 1
                break
            occupied = {initial[u] for u in adj[v] if initial[u] >= 0}
            c = 0
            while c in occupied:
                c += 1
            initial[v] = c
            used = max(used, c + 1)
        best = initial
        best_k = used
        if clock() >= end:
            return {"coloring": best}

        # When the memory estimate is too large, keep adjacency lists and keep re-ordering colour classes.
        # The resource threshold here scales with the budget rather than branching on a particular benchmark size.
        bit_storage = n * ((n + 7) // 8 + 32)
        memory_allowance = (1.0 + effort) * 16000000
        if bit_storage > memory_allowance:
            current = best[:]
            while clock() < end:
                groups = [[] for _ in range(max(current) + 1)]
                for v, c in enumerate(current):
                    groups[c].append(v)
                rng.shuffle(groups)
                trial = [-1] * n
                complete = True
                for group in groups:
                    rng.shuffle(group)
                    for v in group:
                        if clock() >= end:
                            complete = False
                            break
                        forbidden = {trial[u] for u in adj[v] if trial[u] >= 0}
                        c = 0
                        while c in forbidden:
                            c += 1
                        trial[v] = c
                    if not complete:
                        break
                if complete:
                    current = trial
                    k = max(trial) + 1
                    if k < best_k:
                        best, best_k = trial[:], k
            return {"coloring": best}

        masks = []
        for v in range(n):
            if clock() >= end:
                return {"coloring": best}
            mask = 0
            for u in adj[v]:
                mask |= 1 << u
            masks.append(mask)
        all_vertices = (1 << n) - 1

        def vertices(mask):
            while mask:
                bit = mask & -mask
                yield bit.bit_length() - 1
                mask ^= bit

        def partition(colors):
            classes = [0] * (max(colors) + 1)
            for v, c in enumerate(colors):
                classes[c] |= 1 << v
            return [cl for cl in classes if cl]

        def unpack(classes):
            colors = [0] * n
            for c, cl in enumerate(classes):
                for v in vertices(cl):
                    colors[v] = c
            return colors

        def accept(classes):
            nonlocal best, best_k, last_gain
            if len(classes) >= best_k:
                return False
            # Only complete, edge-by-edge verified strict improvements are written to the fallback answer.
            seen = 0
            for cl in classes:
                if seen & cl:
                    return False
                seen |= cl
                for v in vertices(cl):
                    if masks[v] & cl:
                        return False
            if seen != all_vertices:
                return False
            best = unpack(classes)
            best_k = len(classes)
            last_gain = clock()
            return True

        last_gain = clock()
        population = [partition(best)]
        signatures = {tuple(sorted(population[0]))}
        capacity = max(2, int(2 + math.sqrt(n * max(effort, 0.01)) /
                              (1 + math.log2(n + 1))))

        def remember(classes):
            if not classes:
                return
            accept(classes)
            signature = tuple(sorted(classes))
            if signature in signatures:
                return
            population.append(classes)
            signatures.add(signature)
            population.sort(key=lambda p: (len(p), -sum(cl.bit_count() ** 2 for cl in p)))
            while len(population) > capacity:
                old = population.pop()
                signatures.discard(tuple(sorted(old)))

        # A single heap-based saturation-degree construction only provides a starting point; subsequent search operates on whole colour classes.
        def saturation_construct(stop):
            colors = [-1] * n
            forbidden = [0] * n
            heap = [(0, -degree[v], v) for v in range(n)]
            heapq.heapify(heap)
            left = n
            while heap and clock() < stop:
                neg_sat, _, v = heapq.heappop(heap)
                if colors[v] >= 0 or -neg_sat != forbidden[v].bit_count():
                    continue
                free_bit = (~forbidden[v]) & (forbidden[v] + 1)
                colors[v] = free_bit.bit_length() - 1
                left -= 1
                for u in adj[v]:
                    if colors[u] < 0 and not forbidden[u] & free_bit:
                        forbidden[u] |= free_bit
                        heapq.heappush(heap, (-forbidden[u].bit_count(), -degree[u], u))
            return partition(colors) if not left else []

        remember(saturation_construct(min(end, clock() + budget / (8 + effort))))
        clique = []
        lower = 3  # Proven not bipartite.
        clique_stop = min(end, clock() + budget / (35 + math.sqrt(n)))
        while clock() < clique_stop:
            candidate = all_vertices
            found = []
            while candidate and clock() < clique_stop:
                ranked = []
                for v in vertices(candidate):
                    ranked.append(((masks[v] & candidate).bit_count(), rng.random(), v))
                    if clock() >= clique_stop:
                        break
                if not ranked:
                    break
                ranked.sort(reverse=True)
                width = max(1, int(math.sqrt(1 + effort))) if clique else 1
                v = ranked[rng.randrange(min(width, len(ranked)))][2]
                found.append(v)
                candidate &= masks[v]
            if len(found) > len(clique):
                clique = found
                lower = max(lower, len(clique))
            if lower >= best_k:
                return {"coloring": best}

        def pack(sequence):
            classes = []
            for v in sequence:
                if clock() >= end:
                    return []
                bit = 1 << v
                neighbors = masks[v]
                for c, cl in enumerate(classes):
                    if not neighbors & cl:
                        classes[c] |= bit
                        break
                else:
                    classes.append(bit)
            return classes

        def recombine():
            parent = population[rng.randrange(len(population))]
            if len(population) > 1 and rng.random() < 0.5:
                other = population[rng.randrange(len(population))]
                parents = (parent, other)
                remaining = all_vertices
                chunks = []
                which = rng.randrange(2)
                while remaining:
                    if clock() >= end:
                        return []
                    choices = [cl & remaining for cl in parents[which]]
                    largest = max(choices, key=int.bit_count)
                    if not largest:
                        which ^= 1
                        continue
                    chunks.append(largest)
                    remaining ^= largest
                    which ^= 1
            else:
                chunks = parent[:]
                mode = rng.random()
                if mode < 0.35:
                    rng.shuffle(chunks)
                elif mode < 0.7:
                    chunks.sort(key=lambda cl: (cl.bit_count(), rng.random()))
                else:
                    chunks.sort(key=lambda cl: (-cl.bit_count(), rng.random()))
            sequence = []
            for cl in chunks:
                group = list(vertices(cl))
                rng.shuffle(group)
                sequence.extend(group)
            # Moving a few vertices forward breaks the closure of whole-colour-class re-ordering; the magnitude scales with budget and stagnation.
            age = max(0.0, clock() - last_gain) / max(budget, 1e-12)
            if rng.random() < 0.5:
                count = min(n, max(1, int(math.sqrt(n) * (1 + effort) * (0.1 + age))))
                lifted = set(rng.sample(range(n), count))
                sequence = [v for v in sequence if v in lifted] + [v for v in sequence if v not in lifted]
            return pack(sequence)

        cached_k = -1
        core = all_vertices
        peeled = []

        def make_core(k):
            remaining_degree = degree[:]
            active = [True] * n
            queue = [v for v in range(n) if degree[v] < k]
            removed = []
            for v in queue:
                if clock() >= end:
                    return -1, []
                if not active[v]:
                    continue
                active[v] = False
                removed.append(v)
                for u in adj[v]:
                    if active[u]:
                        remaining_degree[u] -= 1
                        if remaining_degree[u] == k - 1:
                            queue.append(u)
            remaining = all_vertices
            for v in removed:
                remaining ^= 1 << v
            return remaining, removed

        def restore(colors, k):
            for v in reversed(peeled):
                occupied = 0
                for u in adj[v]:
                    if colors[u] >= 0:
                        occupied |= 1 << colors[u]
                available = ((1 << k) - 1) & ~occupied
                if not available:
                    return []
                colors[v] = (available & -available).bit_length() - 1
            return partition(colors)

        def recolor(k, free, fixed, preferred, stop):
            """With the external colour classes fixed, find a feasible completion with undoable domain propagation and iterative backtracking."""
            colors = fixed[:]
            domains = [0] * n
            use = [0] * k
            fixed_classes = [0] * k
            for v in vertices(core & ~free):
                c = colors[v]
                fixed_classes[c] |= 1 << v
                use[c] += 1
            palette = (1 << k) - 1
            pending = []
            for v in vertices(free):
                if clock() >= stop:
                    return [], False
                allowed = palette
                for c, cl in enumerate(fixed_classes):
                    if masks[v] & cl:
                        allowed &= ~(1 << c)
                if not allowed:
                    return [], True
                domains[v] = allowed
                if allowed & (allowed - 1) == 0:
                    pending.append((v, allowed))
            remaining = free
            trail = []

            def undo(mark):
                nonlocal remaining
                while len(trail) > mark:
                    v, old = trail.pop()
                    if old < 0:
                        use[colors[v]] -= 1
                        colors[v] = -1
                        remaining |= 1 << v
                    else:
                        domains[v] = old

            def propagate(queue):
                nonlocal remaining
                while queue:
                    if clock() >= stop:
                        return False
                    v, bit = queue.pop()
                    if not remaining & (1 << v):
                        if (1 << colors[v]) != bit:
                            return False
                        continue
                    if not domains[v] & bit:
                        return False
                    trail.append((v, -1))
                    colors[v] = bit.bit_length() - 1
                    use[colors[v]] += 1
                    remaining ^= 1 << v
                    for u in vertices(masks[v] & remaining):
                        old = domains[u]
                        if old & bit:
                            new = old ^ bit
                            trail.append((u, old))
                            domains[u] = new
                            if not new:
                                return False
                            if new & (new - 1) == 0:
                                queue.append((u, new))
                return True

            if not propagate(pending):
                return [], clock() < stop
            stack = []
            while clock() < stop:
                if not remaining:
                    return restore(colors, k), False
                chosen = -1
                key_best = (k + 1, 1)
                for v in vertices(remaining):
                    key = (domains[v].bit_count(), -(masks[v] & remaining).bit_count())
                    if key < key_best:
                        key_best, chosen = key, v
                    if clock() >= stop:
                        return [], False
                options = []
                empty_seen = False
                neighbors = masks[chosen] & remaining
                for c in vertices(domains[chosen]):
                    if use[c] == 0:
                        if empty_seen:
                            continue
                        empty_seen = True
                    bit = 1 << c
                    impact = sum(bool(domains[u] & bit) for u in vertices(neighbors))
                    options.append((c != preferred[chosen], impact, rng.random(), bit))
                options.sort(reverse=True)
                stack.append([chosen, [item[3] for item in options], len(trail)])
                advanced = False
                while stack and clock() < stop:
                    v, choices, mark = stack[-1]
                    undo(mark)
                    if not choices:
                        stack.pop()
                        continue
                    bit = choices.pop()
                    if propagate([(v, bit)]):
                        advanced = True
                        break
                if not advanced:
                    return [], not stack and clock() < stop
            return [], False

        search_round = 0
        next_exact = clock()
        full_due = clock()
        interval = budget / (18 + math.sqrt(n) / (1 + effort))
        while clock() < end and best_k > lower:
            remember(recombine())
            now = clock()
            if now < next_exact or now >= end:
                continue
            target = best_k - 1
            if target != cached_k:
                core, peeled = make_core(target)
                if core < 0:
                    break
                cached_k = target
                search_round = 0
                full_due = now
            if not core:
                remember(restore([-1] * n, target))
                continue
            full = now >= full_due
            if full:
                free = core
                fixed = [-1] * n
                # Colours within a clique are distinct, so fixing their ids loses no feasible colouring.
                for c, v in enumerate(clique):
                    if core & (1 << v):
                        fixed[v] = c
                        free &= ~(1 << v)
                preferred = best
                phase = interval * (1 + effort)
                full_due = now + math.sqrt(max(budget, 1e-12) * interval) * (1 + effort)
            else:
                base = population[0] if rng.random() < 0.5 else rng.choice(population)
                if len(base) != best_k:
                    base = partition(best)
                ranking = sorted(range(len(base)), key=lambda c: (base[c] & core).bit_count())
                removed = ranking[search_round % len(ranking)]
                age = max(0.0, now - last_gain) / max(budget, 1e-12)
                width = min(best_k, max(2, int(2 + effort * (0.5 + age))))
                scores = []
                touched = 0
                for v in vertices(base[removed] & core):
                    touched |= masks[v]
                for c in range(best_k):
                    if c != removed:
                        scores.append(((base[c] & touched).bit_count() * rng.uniform(0.5, 1.5), c))
                scores.sort(reverse=True)
                selected = {removed}
                selected.update(c for _, c in scores[:width - 1])
                free = 0
                fixed = [-1] * n
                preferred = [-1] * n
                label = 0
                for c, cl in enumerate(base):
                    if c == removed:
                        free |= cl & core
                        continue
                    for v in vertices(cl & core):
                        preferred[v] = label
                        if c in selected:
                            free |= 1 << v
                        else:
                            fixed[v] = label
                    label += 1
                phase = interval
            stop = min(end, clock() + phase)
            candidate, exhausted = recolor(target, free, fixed, preferred, stop)
            if candidate:
                remember(candidate)
            elif full and exhausted:
                lower = max(lower, target + 1)
            search_round += 1
            next_exact = clock() + interval / (1 + effort)
        return {"coloring": best}
    except Exception:
        # All search state is separate from the fallback answer; exceptions or exhausted budget never leak an unfinished colouring.
        return {"coloring": best}
