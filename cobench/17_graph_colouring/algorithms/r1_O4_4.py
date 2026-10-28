"""Core peeling with complementary tabu graph colouring
Combines bipartite recognition, saturation-degree construction, clique lower bounds, time-limited exact search, and partial-colouring and conflict tabu searches that exchange high-quality states to reduce colours step by step.
"""

import heapq
import math
import random
import time


def _canonical(colors):
    names = {}
    return [names.setdefault(c, len(names)) for c in colors]


def _greedy(adj, order, stop):
    colors = [-1] * len(adj)
    for i, v in enumerate(order):
        if i % 64 == 0 and time.perf_counter() >= stop:
            return None
        used = {colors[u] for u in adj[v] if colors[u] >= 0}
        c = 0
        while c in used:
            c += 1
        colors[v] = c
    return colors


def _dsatur(adj, stop, rng, noise=0.0):
    n = len(adj)
    colors = [-1] * n
    masks = [0] * n
    priorities = [-len(adj[v]) * (1.0 + noise * rng.random()) for v in range(n)]
    heap = [(0, priorities[v], v) for v in range(n)]
    heapq.heapify(heap)
    left = n
    while heap and left:
        if time.perf_counter() >= stop:
            return None
        sat, _, v = heapq.heappop(heap)
        if colors[v] >= 0 or -sat != masks[v].bit_count():
            continue
        available = ~masks[v]
        bit = available & -available
        colors[v] = bit.bit_length() - 1
        left -= 1
        for u in adj[v]:
            if colors[u] < 0 and not masks[u] & bit:
                masks[u] |= bit
                heapq.heappush(heap, (-masks[u].bit_count(), priorities[u], u))
    return colors if not left else None


def _rlf(adj, stop, rng):
    colors = [-1] * len(adj)
    left = set(range(len(adj)))
    color = 0
    while left:
        if time.perf_counter() >= stop:
            return None
        v, score = -1, None
        for i, u in enumerate(left):
            if i % 32 == 0 and time.perf_counter() >= stop:
                return None
            key = (len(adj[u] & left), rng.random())
            if score is None or key > score:
                v, score = u, key
        candidates = set(left)
        excluded = set()
        while candidates:
            if time.perf_counter() >= stop:
                return None
            colors[v] = color
            left.remove(v)
            candidates.remove(v)
            blocked = candidates & adj[v]
            candidates.difference_update(blocked)
            excluded.update(blocked)
            if candidates:
                score = None
                for i, u in enumerate(candidates):
                    if i % 32 == 0 and time.perf_counter() >= stop:
                        return None
                    key = (len(adj[u] & excluded), -len(adj[u] & candidates), rng.random())
                    if score is None or key > score:
                        score, v = key, u
        color += 1
    return colors


def _clique(adj, stop, rng, wanted):
    """Only append vertices adjacent to all vertices of the current clique, so the resulting lower bound is always valid."""
    n = len(adj)
    best = []
    roots = sorted(range(n), key=lambda v: -len(adj[v]))
    attempt = 0
    while time.perf_counter() < stop and len(best) < wanted:
        root = roots[attempt] if attempt < n else rng.randrange(n)
        attempt += 1
        clique = [root]
        possible = set(adj[root])
        while possible and len(clique) + len(possible) > len(best):
            if time.perf_counter() >= stop:
                return best if len(best) >= len(clique) else clique
            chosen, score = -1, None
            for i, v in enumerate(possible):
                if i % 32 == 0 and time.perf_counter() >= stop:
                    return best if len(best) >= len(clique) else clique
                key = (len(adj[v] & possible), len(adj[v]), rng.random())
                if score is None or key > score:
                    chosen, score = v, key
            clique.append(chosen)
            possible.intersection_update(adj[chosen])
        if len(clique) > len(best):
            best = clique
    return best


def _exact(adj, k, clique, preferred, deadline):
    """Resumable saturation-degree branching search kept on an explicit stack; infeasibility is reported only when the tree is exhausted."""
    n = len(adj)
    colors = [-1] * n
    counts = [[0] * k for _ in range(n)]
    masks = [0] * n
    uncolored = set(range(n))
    full = (1 << k) - 1
    degrees = [len(a) for a in adj]

    def assign(v, c, sign):
        bit = 1 << c
        for u in adj[v]:
            counts[u][c] += sign
            if counts[u][c]:
                masks[u] |= bit
            else:
                masks[u] &= ~bit
        if sign > 0:
            colors[v] = c
            uncolored.remove(v)
        else:
            colors[v] = -1
            uncolored.add(v)

    def select(used):
        chosen, score = -1, (-1, -1)
        for i, v in enumerate(uncolored):
            if i % 64 == 0 and time.perf_counter() >= deadline:
                return -2, 0
            if masks[v] == full:
                return -1, 0
            key = (masks[v].bit_count(), degrees[v])
            if key > score:
                chosen, score = v, key
        choices = ((1 << min(k, used + 1)) - 1) & ~masks[chosen]
        return chosen, choices

    # The colours of a clique can be fixed to distinct consecutive ids without losing feasible solutions.
    rename = {preferred[v]: c for c, v in enumerate(clique)}
    for c in set(preferred):
        if c not in rename:
            rename[c] = len(rename)
    preference = [rename[c] for c in preferred]
    for c, v in enumerate(clique):
        if time.perf_counter() >= deadline:
            return
        assign(v, c, 1)
    used = len(clique)
    if not uncolored:
        yield (True, colors[:])
        return
    v, choices = select(used)
    if v == -2:
        return
    if v == -1:
        yield (False, None)
        return
    stack = [[v, choices, used, -1]]
    while stack:
        yield None
        if time.perf_counter() >= deadline:
            return
        frame = stack[-1]
        v, choices, previous_used, active = frame
        if active >= 0:
            assign(v, active, -1)
            used = previous_used
            frame[3] = -1
        if not choices:
            stack.pop()
            continue
        preferred_bit = 1 << preference[v]
        bit = preferred_bit if choices & preferred_bit else choices & -choices
        c = bit.bit_length() - 1
        frame[1] = choices ^ bit
        frame[3] = c
        assign(v, c, 1)
        used = max(used, c + 1)
        if not uncolored:
            yield (True, colors[:])
            return
        w, options = select(used)
        if w == -2:
            return
        if w >= 0:
            stack.append([w, options, used, -1])
    yield (False, None)


def _partial(adj, seed, k, stop, rng, strength):
    """Legal partial colouring: minimise the number of empty slots, breaking ties by empty-slot degree sum."""
    n = len(adj)
    colors = seed[:]
    holes = {v for v, c in enumerate(colors) if c < 0}
    base_weight = sum(map(len, adj)) + 1
    weights = [base_weight + len(a) for a in adj]
    cost = sum(weights[v] for v in holes)
    rows = [[0] * k for _ in range(n)]
    tabu = [[0] * k for _ in range(n)]
    for v, c in enumerate(colors):
        if v % 64 == 0 and time.perf_counter() >= stop:
            return None, seed
        if c >= 0:
            for u in adj[v]:
                rows[u][c] += weights[v]
    best, best_cost = colors[:], cost
    started = last_improve = time.perf_counter()
    span = max(1e-9, stop - started)
    jitter = max(1, int(math.sqrt(n) * strength))
    iteration = 0
    last_kick = started
    while time.perf_counter() < stop:
        if not holes:
            return colors, colors
        iteration += 1
        delta_best = float('inf')
        fallback = None
        move = None
        ties = 0
        for index, v in enumerate(holes):
            if index % 16 == 0 and time.perf_counter() >= stop:
                return None, best
            for c, count in enumerate(rows[v]):
                delta = count - weights[v]
                if fallback is None or delta < fallback[0]:
                    fallback = (delta, v, c)
                if tabu[v][c] > iteration and cost + delta >= best_cost:
                    continue
                if delta < delta_best:
                    delta_best, move, ties = delta, (v, c), 1
                elif delta == delta_best:
                    ties += 1
                    if rng.randrange(ties) == 0:
                        move = (v, c)
        if move is None:
            _, v, c = fallback
        else:
            v, c = move
        now = time.perf_counter()
        if now - max(last_improve, last_kick) > span / (2.0 + strength):
            v = rng.choice(tuple(holes))
            c = rng.randrange(k)
            last_kick = now
        expelled = [u for u in adj[v] if colors[u] == c]
        cost += rows[v][c] - weights[v]
        colors[v] = c
        holes.remove(v)
        for u in adj[v]:
            rows[u][c] += weights[v]
        reactive = int(jitter * (now - last_improve) / span)
        tenure = int(0.6 * len(holes)) + rng.randint(1, jitter) + reactive
        for w in expelled:
            colors[w] = -1
            holes.add(w)
            for u in adj[w]:
                rows[u][c] -= weights[w]
            tabu[w][c] = iteration + tenure
        if cost < best_cost:
            best_cost, best = cost, colors[:]
            last_improve = now
        if not holes:
            return colors, colors
    return None, best


def _tabu(adj, seed, k, stop, rng, strength):
    """Incremental neighbour-colour count updates, full conflict neighbourhood, aspiration and stagnation perturbation."""
    n = len(adj)
    colors = seed[:]
    rows = [[0] * k for _ in range(n)]
    for v in range(n):
        if v % 64 == 0 and time.perf_counter() >= stop:
            return None, seed, float('inf')
        for u in adj[v]:
            rows[v][colors[u]] += 1
    conflicting = {v for v in range(n) if rows[v][colors[v]]}
    total = sum(rows[v][colors[v]] for v in conflicting) // 2
    tabu = [[0] * k for _ in range(n)]
    best, best_total = colors[:], total
    started = last_improve = last_kick = time.perf_counter()
    span = max(1e-9, stop - started)
    jitter = max(1, int(math.sqrt(n) * strength))
    iteration = 0

    def change(v, c):
        nonlocal total
        old = colors[v]
        total += rows[v][c] - rows[v][old]
        colors[v] = c
        for u in adj[v]:
            rows[u][old] -= 1
            rows[u][c] += 1
            if rows[u][colors[u]]:
                conflicting.add(u)
            else:
                conflicting.discard(u)
        if rows[v][c]:
            conflicting.add(v)
        else:
            conflicting.discard(v)
        return old

    while time.perf_counter() < stop:
        if not total:
            return colors, colors, 0
        iteration += 1
        move = None
        delta_best = n + 1
        fallback = None
        ties = 0
        for index, v in enumerate(conflicting):
            if index % 16 == 0 and time.perf_counter() >= stop:
                return None, best, best_total
            old = colors[v]
            row = rows[v]
            old_cost = row[old]
            for c in range(k):
                if c == old:
                    continue
                delta = row[c] - old_cost
                if fallback is None or delta < fallback[0]:
                    fallback = (delta, v, c)
                if tabu[v][c] > iteration and total + delta >= best_total:
                    continue
                if delta < delta_best:
                    delta_best, move, ties = delta, (v, c), 1
                elif delta == delta_best:
                    ties += 1
                    if rng.randrange(ties) == 0:
                        move = (v, c)
        if move is None:
            _, v, c = fallback
        else:
            v, c = move
        now = time.perf_counter()
        old = change(v, c)
        reactive = int(jitter * (now - last_improve) / span)
        tabu[v][old] = iteration + int(0.6 * len(conflicting)) + rng.randint(1, jitter) + reactive
        if total < best_total:
            best_total, best = total, colors[:]
            last_improve = now
        if not total:
            return colors, colors, 0
        if now - max(last_improve, last_kick) > span / (2.0 + strength):
            kicks = max(1, int(math.sqrt(len(conflicting)) * strength))
            for _ in range(kicks):
                if not conflicting or time.perf_counter() >= stop:
                    break
                v = rng.choice(tuple(conflicting))
                c = rng.randrange(k - 1)
                if c >= colors[v]:
                    c += 1
                change(v, c)
                if total < best_total:
                    best_total, best = total, colors[:]
                    last_improve = time.perf_counter()
            last_kick = time.perf_counter()
    return (best if best_total == 0 else None), best, best_total


def _fill(adj, partial, k, rng, stop):
    colors = partial[:]
    holes = [v for v, c in enumerate(colors) if c < 0]
    rng.shuffle(holes)
    holes.sort(key=lambda v: -len(adj[v]))
    for v in holes:
        if time.perf_counter() >= stop:
            return None
        costs = [0] * k
        for u in adj[v]:
            if colors[u] >= 0:
                costs[colors[u]] += 1
        least = min(costs)
        colors[v] = rng.choice([c for c, value in enumerate(costs) if value == least])
    return colors


def _uncolor(adj, full, stop):
    """Turn a complete conflicting colouring into a legal partial colouring via a vertex cover of conflict edges."""
    colors = full[:]
    conflicts = []
    for v in range(len(adj)):
        if v % 64 == 0 and time.perf_counter() >= stop:
            return None
        conflicts.append(sum(colors[v] == colors[u] for u in adj[v]))
    heap = [(-conflicts[v], len(adj[v]), v) for v in range(len(adj)) if conflicts[v]]
    heapq.heapify(heap)
    while heap:
        if time.perf_counter() >= stop:
            return None
        neg, _, v = heapq.heappop(heap)
        if colors[v] < 0 or -neg != conflicts[v]:
            continue
        old = colors[v]
        colors[v] = -1
        conflicts[v] = 0
        for u in adj[v]:
            if colors[u] == old:
                conflicts[u] -= 1
                if conflicts[u]:
                    heapq.heappush(heap, (-conflicts[u], len(adj[u]), u))
    return colors


def _fixed_k(adj, initial, k, deadline, rng, strength):
    """Return (feasible solution, proven infeasible); a timeout only returns the unproven state."""
    start = time.perf_counter()
    span = max(0.0, deadline - start)
    clique = _clique(adj, min(deadline, start + span * 0.025), rng, k + 1)
    if len(clique) > k:
        return None, True
    exact = _exact(adj, k, clique, initial, deadline)
    exact_share = min(0.25, strength / math.sqrt(max(1, len(adj))))
    quantum = span / (3.0 + strength)
    classes = [[] for _ in range(k + 1)]
    for v, c in enumerate(initial):
        classes[c].append(v)
    ranked = sorted(range(k + 1), key=lambda c: (len(classes[c]), sum(len(adj[v]) for v in classes[c])))
    elite = None
    elite_score = float('inf')
    last_full = None
    restart = 0
    while time.perf_counter() < deadline:
        cycle_end = min(deadline, time.perf_counter() + quantum)
        exact_end = min(cycle_end, time.perf_counter() + quantum * exact_share)
        while exact is not None and time.perf_counter() < exact_end:
            try:
                result = next(exact)
            except StopIteration:
                exact = None
                break
            if result is not None:
                return (result[1], not result[0])
        if time.perf_counter() >= deadline:
            break
        partial = None
        if restart % 3 and last_full is not None:
            source = elite if restart % 3 == 2 and elite is not None else last_full
            partial = _uncolor(adj, source, cycle_end)
        if partial is None:
            removed = ranked[(restart // 3) % (k + 1)]
            partial = [-1 if c == removed else c - (c > removed) for c in initial]
        partial_end = min(cycle_end, time.perf_counter() + quantum * (1.0 - exact_share) * 0.5)
        feasible, partial = _partial(adj, partial, k, partial_end, rng, strength)
        if feasible is not None:
            return feasible, False
        full = _fill(adj, partial, k, rng, cycle_end)
        if full is not None and time.perf_counter() < cycle_end:
            feasible, last_full, score = _tabu(adj, full, k, cycle_end, rng, strength)
            if feasible is not None:
                return feasible, False
            if score < elite_score:
                elite_score, elite = score, last_full[:]
        restart += 1
    return None, False


def solve(instance: dict, time_limit_s: float) -> dict:
    started = time.perf_counter()
    try:
        n = max(0, int(instance.get('n', 0)))
    except Exception:
        return {'coloring': []}
    best = list(range(n))
    if not n:
        return {'coloring': best}
    try:
        budget = max(0.0, float(time_limit_s))
        if not math.isfinite(budget) or not budget:
            return {'coloring': best}
        deadline = started + budget * 0.96
        validation_deadline = started + budget * 0.995
        strength = 1.0 + math.log1p(budget) / 2.0
        rng = random.Random(n * 1000003 + len(instance.get('edges', [])) * 9176)
        adj = [set() for _ in range(n)]
        for i, edge in enumerate(instance.get('edges', [])):
            if i % 256 == 0 and time.perf_counter() >= deadline:
                return {'coloring': best}
            u, v = int(edge[0]), int(edge[1])
            if 0 <= u < n and 0 <= v < n and u != v:
                adj[u].add(v)
                adj[v].add(u)
        degrees = [len(a) for a in adj]
        if not any(degrees):
            return {'coloring': [0] * n}
        best_k = n

        def consider(candidate):
            nonlocal best, best_k
            if candidate is None or len(candidate) != n or any(c < 0 for c in candidate):
                return False
            k = len(set(candidate))
            if k >= best_k:
                return False
            for v in range(n):
                if v % 64 == 0 and time.perf_counter() >= validation_deadline:
                    return False
                if any(candidate[v] == candidate[u] for u in adj[v]):
                    return False
            candidate = _canonical(candidate)
            best, best_k = candidate, k
            return True

        # Connected components share colour ids; each bipartite component directly gets an optimal colouring.
        bip = [-1] * n
        parts = []
        all_bipartite = True
        for root in range(n):
            if time.perf_counter() >= deadline:
                return {'coloring': best}
            if bip[root] >= 0:
                continue
            bip[root] = 0
            queue = [root]
            proper = True
            index = 0
            while index < len(queue):
                if index % 64 == 0 and time.perf_counter() >= deadline:
                    return {'coloring': best}
                v = queue[index]
                index += 1
                for u in adj[v]:
                    if bip[u] < 0:
                        bip[u] = 1 - bip[v]
                        queue.append(u)
                    elif bip[u] == bip[v]:
                        proper = False
            parts.append((queue, proper))
            all_bipartite = all_bipartite and proper
        if all_bipartite:
            return {'coloring': bip}
        lower = 3
        initial = [-1] * n
        for vertices, proper in parts:
            if proper:
                for v in vertices:
                    initial[v] = bip[v]
            else:
                for v in sorted(vertices, key=lambda u: -degrees[u]):
                    if time.perf_counter() >= deadline:
                        return {'coloring': best}
                    used = {initial[u] for u in adj[v] if initial[u] >= 0}
                    c = 0
                    while c in used:
                        c += 1
                    initial[v] = c
        consider(initial)
        if best_k == lower:
            return {'coloring': best}
        construction_end = min(deadline, started + budget * 0.15)
        consider(_dsatur(adj, construction_end, rng))
        if best_k == lower:
            return {'coloring': best}
        clique_stop = min(construction_end, time.perf_counter() + budget * 0.025)
        lower = max(lower, len(_clique(adj, clique_stop, rng, best_k)))
        round_id = 0
        while time.perf_counter() < construction_end and best_k > lower:
            stop = min(construction_end, time.perf_counter() + budget / (12.0 + strength))
            if round_id % 3 == 0:
                candidate = _rlf(adj, stop, rng)
            elif round_id % 3 == 1:
                candidate = _dsatur(adj, stop, rng, strength / (1.0 + strength))
            else:
                classes = [[] for _ in range(best_k)]
                for v, c in enumerate(best):
                    classes[c].append(v)
                rng.shuffle(classes)
                if rng.random() < 0.5:
                    classes.sort(key=len)
                order = []
                for group in classes:
                    rng.shuffle(group)
                    order.extend(group)
                candidate = _greedy(adj, order, stop)
            consider(candidate)
            round_id += 1

        # For target k, vertices with degree below k can be greedily restored later; only the remaining core is searched.
        while time.perf_counter() < deadline and best_k > lower:
            k = best_k - 1
            residual = degrees[:]
            removed = [False] * n
            peel = [v for v in range(n) if residual[v] < k]
            for v in peel:
                removed[v] = True
            index = 0
            while index < len(peel):
                if index % 64 == 0 and time.perf_counter() >= deadline:
                    return {'coloring': best}
                v = peel[index]
                index += 1
                for u in adj[v]:
                    if not removed[u]:
                        residual[u] -= 1
                        if residual[u] < k:
                            removed[u] = True
                            peel.append(u)
            seen = removed[:]
            core_parts = []
            for root in range(n):
                if seen[root]:
                    continue
                seen[root] = True
                vertices = [root]
                index = 0
                while index < len(vertices):
                    if index % 64 == 0 and time.perf_counter() >= deadline:
                        return {'coloring': best}
                    v = vertices[index]
                    index += 1
                    for u in adj[v]:
                        if not seen[u]:
                            seen[u] = True
                            vertices.append(u)
                core_parts.append(vertices)
            core_parts.sort(key=lambda vs: (len(set(best[v] for v in vs)), sum(residual[v] for v in vs) / len(vs)), reverse=True)
            trial = [-1] * n
            for vertices in core_parts:
                seed = _canonical([best[v] for v in vertices])
                if max(seed) < k:
                    result = seed
                else:
                    labels = {v: i for i, v in enumerate(vertices)}
                    local_adj = []
                    for i, v in enumerate(vertices):
                        if i % 64 == 0 and time.perf_counter() >= deadline:
                            return {'coloring': best}
                        local_adj.append({labels[u] for u in adj[v] if not removed[u]})
                    result, impossible = _fixed_k(local_adj, seed, k, deadline, rng, strength)
                    if impossible:
                        lower = best_k
                        return {'coloring': best}
                    if result is None:
                        return {'coloring': best}
                for v, c in zip(vertices, result):
                    trial[v] = c
            for i, v in enumerate(reversed(peel)):
                if i % 64 == 0 and time.perf_counter() >= deadline:
                    return {'coloring': best}
                used = {trial[u] for u in adj[v] if trial[u] >= 0}
                c = 0
                while c in used:
                    c += 1
                trial[v] = c
            if not consider(trial):
                return {'coloring': best}
        return {'coloring': best}
    except Exception:
        # Only fully checked feasible solutions are saved; unfinished searches or exceptions never overwrite the fallback solution.
        return {'coloring': best}
