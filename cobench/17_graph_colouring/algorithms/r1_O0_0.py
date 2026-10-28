"""Core peeling with hybrid tabu graph colouring
Combines saturation-degree construction, clique lower bounds, low-degree peeling, time-limited exact search and colour-class crossover tabu search, continually keeping the feasible solution with the fewest colours.
"""

import heapq
import math
import random
import time
from collections import deque


def _compact(colors):
    labels = {}
    result = []
    for c in colors:
        if c not in labels:
            labels[c] = len(labels)
        result.append(labels[c])
    return result, len(labels)


def _greedy(adj, order, deadline):
    colors = [-1] * len(adj)
    used = 0
    for v in order:
        if time.perf_counter() >= deadline:
            break
        mask = 0
        for u in adj[v]:
            if colors[u] >= 0:
                mask |= 1 << colors[u]
        c = ((mask + 1) & ~mask).bit_length() - 1
        colors[v] = c
        used = max(used, c + 1)
    for v in range(len(adj)):
        if colors[v] < 0:
            colors[v] = used
            used += 1
    return colors, used


def _dsatur(adj, deadline, rng=None):
    n = len(adj)
    colors = [-1] * n
    masks = [0] * n
    saturation = [0] * n
    priorities = [len(a) for a in adj]
    if rng is not None:
        priorities = [d * (0.8 + 0.4 * rng.random()) for d in priorities]
    heap = [(0, -priorities[v], v) for v in range(n)]
    heapq.heapify(heap)
    used = 0
    while heap and time.perf_counter() < deadline:
        neg_sat, _, v = heapq.heappop(heap)
        if colors[v] >= 0 or -neg_sat != saturation[v]:
            continue
        mask = masks[v]
        c = ((mask + 1) & ~mask).bit_length() - 1
        colors[v] = c
        used = max(used, c + 1)
        bit = 1 << c
        for u in adj[v]:
            if colors[u] < 0 and not masks[u] & bit:
                masks[u] |= bit
                saturation[u] += 1
                heapq.heappush(heap, (-saturation[u], -priorities[u], u))
    for v in range(n):
        if colors[v] < 0:
            colors[v] = used
            used += 1
    return colors, used


def _clique_bound(adj, order, upper, deadline, rng):
    degree = [len(a) for a in adj]
    best = []
    first = True
    while time.perf_counter() < deadline and len(best) < upper:
        seed = order[0] if first else rng.choice(order)
        first = False
        clique = [seed]
        candidates = set(adj[seed])
        while candidates and time.perf_counter() < deadline:
            if rng.random() < 0.15:
                v = rng.choice(tuple(candidates))
            else:
                v = max(candidates, key=degree.__getitem__)
            clique.append(v)
            candidates.intersection_update(adj[v])
        if len(clique) > len(best):
            best = clique
    return best


def _core(adj, k, deadline):
    degree = [len(a) for a in adj]
    active = [True] * len(adj)
    queue = deque(v for v, d in enumerate(degree) if d < k)
    removed = []
    while queue:
        if time.perf_counter() >= deadline:
            return None
        v = queue.popleft()
        if not active[v]:
            continue
        active[v] = False
        removed.append(v)
        for u in adj[v]:
            if active[u]:
                degree[u] -= 1
                if degree[u] == k - 1:
                    queue.append(u)
    vertices = [v for v, live in enumerate(active) if live]
    index = {v: i for i, v in enumerate(vertices)}
    local = []
    for v in vertices:
        if time.perf_counter() >= deadline:
            return None
        local.append([index[u] for u in adj[v] if active[u]])
    return vertices, index, local, removed


def _extend(adj, vertices, local_colors, removed):
    colors = [-1] * len(adj)
    for i, v in enumerate(vertices):
        colors[v] = local_colors[i]
    for v in reversed(removed):
        mask = 0
        for u in adj[v]:
            if colors[u] >= 0:
                mask |= 1 << colors[u]
        colors[v] = ((mask + 1) & ~mask).bit_length() - 1
    return _compact(colors)


def _exact(adj, k, clique, hint, deadline):
    n = len(adj)
    full = (1 << k) - 1
    colors = [-1] * n
    masks = [0] * n
    degree = [len(a) for a in adj]
    if len(clique) > k:
        return 0, []
    for c, v in enumerate(clique):
        colors[v] = c
        for u in adj[v]:
            masks[u] |= 1 << c

    def visit(left, used):
        if time.perf_counter() >= deadline:
            return -1
        if not left:
            return 1
        v = -1
        key = (-1, -1)
        for u in range(n):
            if colors[u] < 0:
                if masks[u] == full:
                    return 0
                score = (masks[u].bit_count(), degree[u])
                if score > key:
                    key, v = score, u
        available = ((1 << min(k, used + 1)) - 1) & ~masks[v]
        preferred = 1 << hint[v]
        while available:
            bit = preferred if available & preferred else available & -available
            available ^= bit
            c = bit.bit_length() - 1
            colors[v] = c
            changed = []
            blocked = False
            for u in adj[v]:
                if colors[u] < 0 and not masks[u] & bit:
                    changed.append((u, masks[u]))
                    masks[u] |= bit
                    if masks[u] == full:
                        blocked = True
                        break
            status = 0 if blocked else visit(left - 1, max(used, c + 1))
            if status == 1:
                return 1
            for u, old in changed:
                masks[u] = old
            colors[v] = -1
            if status < 0:
                return -1
        return 0

    status = visit(n - len(clique), len(clique))
    return status, colors if status == 1 else []


def _eliminate(adj, incumbent, k, rng, first):
    classes = [[] for _ in range(k + 1)]
    for v, c in enumerate(incumbent):
        classes[c].append(v)
    if first or rng.random() < 0.5:
        remove = min(range(k + 1), key=lambda c: len(classes[c]))
    else:
        remove = rng.randrange(k + 1)
    colors = [c - (c > remove) if c != remove else -1 for c in incumbent]
    for v in classes[remove]:
        counts = [0] * k
        for u in adj[v]:
            c = colors[u]
            if c >= 0:
                counts[c] += 1
        minimum = min(counts)
        colors[v] = rng.choice([c for c, count in enumerate(counts) if count == minimum])
    return colors


def _crossover(adj, first, second, k, rng, deadline):
    n = len(adj)
    parents = []
    for parent in (first, second):
        classes = [set() for _ in range(k)]
        for v, c in enumerate(parent):
            classes[c].add(v)
        parents.append(classes)
    colors = [-1] * n
    side = rng.randrange(2)
    for c in range(k):
        if time.perf_counter() >= deadline:
            return first[:]
        classes = parents[(side + c) % 2]
        chosen = max(classes, key=len).copy()
        for v in chosen:
            colors[v] = c
        for groups in parents:
            for group in groups:
                group.difference_update(chosen)
    remaining = [v for v, c in enumerate(colors) if c < 0]
    rng.shuffle(remaining)
    for v in remaining:
        if time.perf_counter() >= deadline:
            return first[:]
        counts = [0] * k
        for u in adj[v]:
            if colors[u] >= 0:
                counts[colors[u]] += 1
        minimum = min(counts)
        colors[v] = rng.choice([c for c in range(k) if counts[c] == minimum])
    return colors


def _tabu(adj, initial, k, deadline, rng, strength):
    n = len(adj)
    colors = initial[:]
    counts = [[0] * k for _ in range(n)]
    tabu = [{} for _ in range(n)]
    bad = []
    position = [-1] * n
    conflicts = 0
    for v in range(n):
        if time.perf_counter() >= deadline:
            return None, initial
        row = counts[v]
        for u in adj[v]:
            row[colors[u]] += 1
        conflicts += row[colors[v]]
        if row[colors[v]]:
            position[v] = len(bad)
            bad.append(v)
    conflicts //= 2
    best_value = conflicts
    best = colors[:]
    if not conflicts:
        return 0, best
    step = 0
    scan_width = max(8, int((800 + 1800 * strength) / k))
    random_tenure = max(2, int(3 + 3 * strength))

    def refresh(v):
        p = position[v]
        if counts[v][colors[v]]:
            if p < 0:
                position[v] = len(bad)
                bad.append(v)
        elif p >= 0:
            last = bad.pop()
            if p < len(bad):
                bad[p] = last
                position[last] = p
            position[v] = -1

    last_improvement = time.perf_counter()
    started = last_improvement
    while time.perf_counter() < deadline:
        step += 1
        sample = bad if len(bad) <= scan_width else rng.sample(bad, scan_width)
        delta_best = n + 1
        chosen_v = -1
        chosen_c = -1
        ties = 0
        for v in sample:
            row = counts[v]
            old = colors[v]
            old_conflicts = row[old]
            bans = tabu[v]
            for c in range(k):
                if c == old:
                    continue
                delta = row[c] - old_conflicts
                if delta > delta_best:
                    continue
                if bans.get(c, 0) > step and conflicts + delta >= best_value:
                    continue
                if delta < delta_best:
                    delta_best, chosen_v, chosen_c, ties = delta, v, c, 1
                else:
                    ties += 1
                    if rng.randrange(ties) == 0:
                        chosen_v, chosen_c = v, c
            if time.perf_counter() >= deadline:
                return best_value, best
        if chosen_v < 0:
            chosen_v = rng.choice(bad)
            old = colors[chosen_v]
            chosen_c = rng.randrange(k - 1)
            if chosen_c >= old:
                chosen_c += 1
            delta_best = counts[chosen_v][chosen_c] - counts[chosen_v][old]
        v, new = chosen_v, chosen_c
        old = colors[v]
        now = time.perf_counter()
        plateau = (now - last_improvement) / max(0.001, deadline - started)
        tenure = int(0.6 * len(bad) + plateau * math.sqrt(n))
        tabu[v][old] = step + tenure + rng.randrange(1, random_tenure + 1)
        colors[v] = new
        conflicts += delta_best
        for u in adj[v]:
            counts[u][old] -= 1
            counts[u][new] += 1
            refresh(u)
        refresh(v)
        if conflicts < best_value:
            best_value = conflicts
            best = colors[:]
            last_improvement = now
            if conflicts == 0:
                return 0, best
    return best_value, best


def solve(instance: dict, time_limit_s: float) -> dict:
    started = time.perf_counter()
    best = []
    try:
        n = max(0, int(instance["n"]))
        best = list(range(n))
        budget = float(time_limit_s)
        if n == 0 or not math.isfinite(budget) or budget <= 0:
            return {"coloring": best}
        deadline = started + budget - min(0.06, 0.04 * budget)
        rng = random.Random(721347)
        strength = math.log1p(budget)
        adj = [set() for _ in range(n)]
        for edge in instance["edges"]:
            if time.perf_counter() >= deadline:
                return {"coloring": best}
            u, v = int(edge[0]), int(edge[1])
            if u == v or not (0 <= u < n and 0 <= v < n):
                return {"coloring": best}
            adj[u].add(v)
            adj[v].add(u)
        edge_twice = sum(map(len, adj))
        if edge_twice == 0:
            return {"coloring": [0] * n}
        if edge_twice == n * (n - 1):
            return {"coloring": best}

        # Bipartite graphs can be solved optimally directly; once an odd cycle is found, the lower bound is at least three.
        bipartite = [-1] * n
        valid = True
        for root in range(n):
            if bipartite[root] >= 0:
                continue
            bipartite[root] = 0
            queue = deque([root])
            while queue and valid:
                if time.perf_counter() >= deadline:
                    return {"coloring": best}
                v = queue.popleft()
                for u in adj[v]:
                    if bipartite[u] < 0:
                        bipartite[u] = bipartite[v] ^ 1
                        queue.append(u)
                    elif bipartite[u] == bipartite[v]:
                        valid = False
                        break
            if not valid:
                break
        if valid:
            return {"coloring": bipartite}
        lower = 3
        order = sorted(range(n), key=lambda v: len(adj[v]), reverse=True)
        best, upper = _greedy(adj, order, min(deadline, time.perf_counter() + 0.12 * budget))
        if upper == lower:
            return {"coloring": best}
        candidate, used = _dsatur(adj, min(deadline, time.perf_counter() + 0.18 * budget))
        if used < upper:
            best, upper = candidate, used
        if upper == lower:
            return {"coloring": best}
        clique = _clique_bound(adj, order, upper,
                               min(deadline, time.perf_counter() + 0.035 * budget), rng)
        lower = max(lower, len(clique))
        exact_size = min(600, max(20, int(25 * math.log2(2 + budget))))
        population_size = max(2, int(2 + 1.5 * strength))
        slice_length = budget / (5 + math.log2(n + 2))

        while upper > lower and time.perf_counter() < deadline:
            k = upper - 1
            reduced = _core(adj, k, deadline)
            if reduced is None:
                break
            vertices, index, local, removed = reduced
            base, base_count = _compact([best[v] for v in vertices])
            if base_count <= k:
                best, upper = _extend(adj, vertices, base, removed)
                continue
            if len(vertices) <= exact_size:
                local_clique = [index[v] for v in clique if v in index]
                exact_deadline = min(deadline, time.perf_counter() + 0.16 * (deadline - time.perf_counter()))
                status, result = _exact(local, k, local_clique, base, exact_deadline)
                if status == 1:
                    best, upper = _extend(adj, vertices, result, removed)
                    continue
                if status == 0:
                    break

            population = []
            first = True
            improved = False
            next_construction = time.perf_counter() + 2 * slice_length
            while time.perf_counter() < deadline:
                now = time.perf_counter()
                # Overly large dense count tables are replaced by a directly feasible randomised construction.
                if now >= next_construction or len(vertices) * k > 1500000:
                    candidate, used = _dsatur(adj, min(deadline, now + slice_length), rng)
                    next_construction = time.perf_counter() + 2 * slice_length
                    if used < upper:
                        best, upper = candidate, used
                        improved = True
                        break
                    if used == upper:
                        base, base_count = _compact([candidate[v] for v in vertices])
                        if base_count <= k:
                            best, upper = _extend(adj, vertices, base, removed)
                            improved = True
                            break
                    if len(vertices) * k > 1500000:
                        continue
                if time.perf_counter() >= deadline:
                    break
                if len(population) >= 2 and rng.random() < 0.7:
                    a, b = rng.sample(population, 2)
                    seed = _crossover(local, a[1], b[1], k, rng, deadline)
                elif population and rng.random() < 0.45:
                    seed = rng.choice(population)[1][:]
                    amount = min(len(seed), max(1, int(math.sqrt(len(seed)) * strength / 3)))
                    for v in rng.sample(range(len(seed)), amount):
                        seed[v] = rng.randrange(k)
                else:
                    seed = _eliminate(local, base, k, rng, first)
                first = False
                score, result = _tabu(local, seed, k,
                                      min(deadline, time.perf_counter() + slice_length), rng, strength)
                if score == 0:
                    best, upper = _extend(adj, vertices, result, removed)
                    improved = True
                    break
                if score is not None and time.perf_counter() < deadline:
                    normalized, _ = _compact(result)
                    if not any(normalized == entry[1] for entry in population):
                        population.append((score, normalized))
                        population.sort(key=lambda entry: entry[0])
                        if len(population) > population_size:
                            # Keep the best members while allowing new members with equal score to replace old ones.
                            worst = population[-1][0]
                            choices = [i for i in range(1, len(population)) if population[i][0] == worst]
                            population.pop(rng.choice(choices))
            if not improved:
                break
    except Exception:
        # Any runtime error interrupting the search falls back to the already constructed complete feasible colouring.
        pass
    return {"coloring": best}
