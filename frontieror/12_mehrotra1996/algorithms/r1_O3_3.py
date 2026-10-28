"""Core peeling and dual-state adaptive tabu colouring
Combines saturation-degree and independent-set construction and low-degree core peeling, fusing incremental partial colouring with conflict search via state transitions and adaptive candidate sampling.
"""

import heapq
import math
import random
import time


def solve(instance: dict, time_limit_s: float) -> dict:
    clock = time.perf_counter
    started = clock()
    best = []
    try:
        n = max(0, int(instance.get("n", 0)))
        best = list(range(n))
        if n == 0:
            return {"coloring": best}
        budget = max(0.0, float(time_limit_s))
        if not math.isfinite(budget):
            return {"coloring": best}
        deadline = started + 0.97 * budget
        if clock() >= deadline:
            return {"coloring": best}
        adj = [set() for _ in range(n)]
        for index, edge in enumerate(instance.get("edges", [])):
            if index % 256 == 0 and clock() >= deadline:
                return {"coloring": best}
            u, v = int(edge[0]), int(edge[1])
            if 0 <= u < n and 0 <= v < n and u != v:
                adj[u].add(v)
                adj[v].add(u)
        degree = [len(row) for row in adj]
        if not any(degree):
            return {"coloring": [0] * n}
        best_k = n
        scale = math.log1p(budget)
        rng = random.Random(n * 1000003 + sum(degree) * 9176)

        def canonical(colors):
            mapping = {}
            return [mapping.setdefault(c, len(mapping)) for c in colors]

        def accept(colors):
            nonlocal best, best_k
            if colors is None or len(colors) != n:
                return False
            k = len(set(colors))
            if k >= best_k or min(colors) < 0:
                return False
            for v, row in enumerate(adj):
                if v % 64 == 0 and clock() >= deadline:
                    return False
                if any(colors[u] == colors[v] for u in row if u > v):
                    return False
            best = canonical(colors)
            best_k = k
            return True

        # Bipartite graphs get the optimum directly; any unfinished check keeps the original feasible solution.
        bip = [-1] * n
        bipartite = True
        for root in range(n):
            if clock() >= deadline:
                return {"coloring": best}
            if bip[root] >= 0:
                continue
            bip[root] = 0
            stack = [root]
            while stack and bipartite:
                if clock() >= deadline:
                    return {"coloring": best}
                v = stack.pop()
                for u in adj[v]:
                    if bip[u] < 0:
                        bip[u] = 1 - bip[v]
                        stack.append(u)
                    elif bip[u] == bip[v]:
                        bipartite = False
                        break
            if not bipartite:
                break
        if bipartite:
            return {"coloring": bip}
        lower = 3

        def greedy(order, stop):
            colors = [-1] * n
            for v in order:
                if clock() >= stop:
                    return None
                forbidden = {colors[u] for u in adj[v] if colors[u] >= 0}
                c = 0
                while c in forbidden:
                    c += 1
                colors[v] = c
            return colors

        accept(greedy(sorted(range(n), key=lambda v: -degree[v]), deadline))
        if best_k == lower:
            return {"coloring": best}

        def dsatur(randomized, stop):
            colors = [-1] * n
            masks = [0] * n
            weights = [degree[v] * (1 + rng.random() * scale / (8 + scale))
                       if randomized else degree[v] for v in range(n)]
            heap = [(0, -weights[v], v) for v in range(n)]
            heapq.heapify(heap)
            left = n
            while left:
                if clock() >= stop:
                    return None
                sat, _, v = heapq.heappop(heap)
                if colors[v] >= 0 or -sat != masks[v].bit_count():
                    continue
                bit = (masks[v] + 1) & ~masks[v]
                colors[v] = bit.bit_length() - 1
                left -= 1
                for u in adj[v]:
                    if colors[u] < 0 and not masks[u] & bit:
                        masks[u] |= bit
                        heapq.heappush(heap, (-masks[u].bit_count(), -weights[u], u))
            return colors

        def rlf(randomized, stop):
            colors = [-1] * n
            uncolored = set(range(n))
            color = 0
            while uncolored:
                candidates = uncolored.copy()
                excluded = set()
                first = True
                while candidates:
                    selected = -1
                    top = None
                    ties = 0
                    for index, v in enumerate(candidates):
                        if index % 32 == 0 and clock() >= stop:
                            return None
                        score = ((len(adj[v] & uncolored), degree[v]) if first
                                 else (len(adj[v] & excluded),
                                       -len(adj[v] & candidates)))
                        if top is None or score > top:
                            top, selected, ties = score, v, 1
                        elif score == top:
                            ties += 1
                            if randomized and rng.randrange(ties) == 0:
                                selected = v
                    v = selected
                    colors[v] = color
                    uncolored.remove(v)
                    candidates.remove(v)
                    blocked = candidates & adj[v]
                    candidates.difference_update(blocked)
                    excluded.update(blocked)
                    first = False
                color += 1
            return colors

        construction_end = min(deadline, started + budget * 0.17)
        accept(dsatur(False, construction_end))
        if best_k == lower:
            return {"coloring": best}
        # Only clique sizes actually found are used as lower bounds; heuristic failure is not treated as a proof of infeasibility.
        clique_end = min(construction_end, clock() + budget * 0.015)
        candidates = set(range(n))
        clique_size = 0
        while candidates and clock() < clique_end:
            chosen, score = -1, -1
            for v in candidates:
                if clock() >= clique_end:
                    break
                d = len(adj[v] & candidates)
                if d > score:
                    chosen, score = v, d
            if chosen < 0:
                break
            clique_size += 1
            lower = max(lower, clique_size)
            candidates.intersection_update(adj[chosen])
        if best_k == lower:
            return {"coloring": best}
        turn = 0
        while clock() < construction_end:
            stop = min(construction_end, clock() + budget * 0.035)
            candidate = rlf(turn > 0, stop) if turn % 2 == 0 else dsatur(True, stop)
            accept(candidate)
            if best_k == lower:
                return {"coloring": best}
            turn += 1

        def kernel(k):
            residual = degree[:]
            alive = [True] * n
            queue = [v for v in range(n) if residual[v] < k]
            for v in queue:
                alive[v] = False
            peeled = []
            while queue:
                if clock() >= deadline:
                    return None
                v = queue.pop()
                peeled.append(v)
                for u in adj[v]:
                    if alive[u]:
                        residual[u] -= 1
                        if residual[u] < k:
                            alive[u] = False
                            queue.append(u)
            vertices = [v for v in range(n) if alive[v]]
            ids = {v: i for i, v in enumerate(vertices)}
            rows = []
            for v in vertices:
                if clock() >= deadline:
                    return None
                rows.append([ids[u] for u in adj[v] if alive[u]])
            return vertices, rows, peeled

        def extend(core_colors, vertices, peeled):
            colors = [-1] * n
            for i, v in enumerate(vertices):
                colors[v] = core_colors[i]
            for v in reversed(peeled):
                if clock() >= deadline:
                    return None
                forbidden = {colors[u] for u in adj[v] if colors[u] >= 0}
                c = 0
                while c in forbidden:
                    c += 1
                colors[v] = c
            return colors

        def kempe(base, rows, restart, stop):
            colors = base[:]
            k = max(colors) + 1
            swaps = int(1 + math.sqrt(scale) * math.log1p(restart))
            for _ in range(swaps):
                if clock() >= stop:
                    break
                v = rng.randrange(len(colors))
                a = colors[v]
                b = rng.randrange(k - 1)
                b += b >= a
                seen = {v}
                stack = [v]
                while stack:
                    if clock() >= stop:
                        return canonical(colors)
                    w = stack.pop()
                    for u in rows[w]:
                        if u not in seen and (colors[u] == a or colors[u] == b):
                            seen.add(u)
                            stack.append(u)
                for v in seen:
                    colors[v] = b if colors[v] == a else a
            return canonical(colors)

        def hybrid(base, rows, k, restart, stop):
            """Both states share neighbour colour counts, tabu checks and sampling rules, with elite states passed in both directions."""
            q = len(rows)
            classes = [[] for _ in range(k + 1)]
            for v, c in enumerate(base):
                classes[c].append(v)
            ranked = sorted(range(k + 1),
                            key=lambda c: (len(classes[c]),
                                           sum(len(rows[v]) for v in classes[c])))
            removed = ranked[restart % (k + 1)]
            colors = [(-1 if c == removed else c - (c > removed)) for c in base]
            mode = 0
            counts = []
            bad = []
            position = []
            total = 0

            def rebuild():
                nonlocal counts, bad, position, total
                counts = []
                for v in range(q):
                    if clock() >= stop:
                        return False
                    row = [0] * k
                    for u in rows[v]:
                        if colors[u] >= 0:
                            row[colors[u]] += 1
                    counts.append(row)
                total = sum(counts[v][c] for v, c in enumerate(colors) if c >= 0) // 2
                bad = []
                position = [-1] * q
                for v, c in enumerate(colors):
                    if (c < 0 if mode == 0 else c >= 0 and counts[v][c] > 0):
                        position[v] = len(bad)
                        bad.append(v)
                return True

            def mark(v, active):
                p = position[v]
                if active and p < 0:
                    position[v] = len(bad)
                    bad.append(v)
                elif not active and p >= 0:
                    last = bad.pop()
                    if p < len(bad):
                        bad[p] = last
                        position[last] = p
                    position[v] = -1

            def move(v, new):
                nonlocal total
                old = colors[v]
                total += ((counts[v][new] if new >= 0 else 0)
                          - (counts[v][old] if old >= 0 else 0))
                colors[v] = new
                for u in rows[v]:
                    if old >= 0:
                        counts[u][old] -= 1
                    if new >= 0:
                        counts[u][new] += 1
                    if mode:
                        c = colors[u]
                        mark(u, c >= 0 and counts[u][c] > 0)
                mark(v, new < 0 if mode == 0 else new >= 0 and counts[v][new] > 0)

            if not rebuild():
                return None
            sample_base = max(1, int(math.sqrt(q) * (1 + scale)))
            jitter = max(2, int(math.sqrt(q) * (1 + scale) / 5))
            segment = max(0.0, (stop - clock()) / (2 + math.sqrt(scale)))
            switch_at = clock() + segment
            tabu = [[0] * k for _ in range(q)]
            iteration = 0
            elite_cost = len(bad)
            elite = colors[:]
            last_gain = clock()
            while clock() < stop:
                cost = len(bad) if mode == 0 else total
                if cost == 0:
                    return colors
                now = clock()
                if now >= switch_at:
                    colors = elite[:]
                    if not rebuild():
                        return None
                    if mode == 0:
                        # Fill empty slots with the least-conflict colour: hand the progress of partial colouring over to conflict search.
                        holes = bad[:]
                        holes.sort(key=lambda v: -len(rows[v]))
                        for v in holes:
                            if clock() >= stop:
                                return None
                            row = counts[v]
                            least = min(row)
                            options = [c for c in range(k) if row[c] == least]
                            move(v, rng.choice(options))
                        mode = 1
                    else:
                        # Greedy vertex cover of conflict edges for removal: keep the other legal colours instead of regenerating randomly.
                        while total:
                            if clock() >= stop:
                                return None
                            choices = (rng.sample(bad, sample_base)
                                       if len(bad) > sample_base else bad)
                            v = max(choices, key=lambda u: (counts[u][colors[u]], -len(rows[u])))
                            move(v, -1)
                        mode = 0
                    if not rebuild():
                        return None
                    tabu = [[0] * k for _ in range(q)]
                    elite_cost = len(bad) if mode == 0 else total
                    elite = colors[:]
                    last_gain = clock()
                    switch_at = last_gain + segment
                    continue

                iteration += 1
                stall = max(0.0, (now - last_gain) / max(segment, 1e-12))
                width = min(q, max(1, int(sample_base * (1 + stall))))
                candidates = rng.sample(bad, width) if len(bad) > width else bad
                chosen_v, chosen_c = -1, -1
                best_delta, ties = q + 1, 0
                fallback = None
                for index, v in enumerate(candidates):
                    if index % 16 == 0 and clock() >= stop:
                        return None
                    old = colors[v]
                    row = counts[v]
                    old_cost = 1 if mode == 0 else row[old]
                    for c in range(k):
                        if c == old:
                            continue
                        delta = row[c] - old_cost
                        if fallback is None or delta < fallback[0]:
                            fallback = (delta, v, c)
                        if tabu[v][c] > iteration and cost + delta >= elite_cost:
                            continue
                        if delta < best_delta:
                            best_delta, chosen_v, chosen_c, ties = delta, v, c, 1
                        elif delta == best_delta:
                            ties += 1
                            if rng.randrange(ties) == 0:
                                chosen_v, chosen_c = v, c
                if chosen_v < 0:
                    if fallback is None:
                        return None
                    best_delta, chosen_v, chosen_c = fallback
                tenure = (int(0.6 * len(bad) + math.sqrt(q) * stall)
                          + rng.randint(1, jitter))
                if mode == 0:
                    expelled = [u for u in rows[chosen_v] if colors[u] == chosen_c]
                    for u in expelled:
                        move(u, -1)
                        tabu[u][chosen_c] = iteration + tenure
                    move(chosen_v, chosen_c)
                    cost = len(bad)
                else:
                    old = colors[chosen_v]
                    move(chosen_v, chosen_c)
                    tabu[chosen_v][old] = iteration + tenure
                    cost = total
                if cost < elite_cost:
                    elite_cost = cost
                    elite = colors[:]
                    last_gain = clock()
                if cost == 0:
                    return colors
            return None

        target = -1
        restart = 0
        vertices, rows, peeled = [], [], []
        base = []
        while clock() < deadline and best_k > lower:
            k = best_k - 1
            if target != k:
                reduced = kernel(k)
                if reduced is None:
                    break
                vertices, rows, peeled = reduced
                base = canonical([best[v] for v in vertices])
                target, restart = k, 0
                if not base or max(base) < k:
                    if accept(extend(base, vertices, peeled)):
                        continue
                    break
            span = budget / (4 + math.sqrt(scale))
            span *= 1 + math.log1p(restart) / (1 + scale + math.log1p(restart))
            stop = min(deadline, clock() + span)
            seed = kempe(base, rows, restart, stop) if restart else base
            if max(seed) < k:
                candidate = seed
            else:
                candidate = hybrid(seed, rows, k, restart, stop)
            if candidate is not None and accept(extend(candidate, vertices, peeled)):
                continue
            restart += 1
        return {"coloring": best}
    except Exception:
        # Only verified complete solutions may replace the safeguard solution; unexpected interruptions do not leak conflicting search state.
        return {"coloring": best}
