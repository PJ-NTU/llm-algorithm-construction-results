# Library slot 9/10  id=alg_a7d129669a2d
# score=-163.6  feasible_rate=1.0
# score_vec=[-369.0, -128.0, -100.0, -80.0, -111.0, -55.0, -116.0, -99.0, -441.0, -137.0]

"""Anytime maximum-independent-set solver using only the Python standard library."""

import heapq
import random
import time
from collections import deque


def _remove_vertex(v, active, degree, adj, queue):
    if not active[v]:
        return
    active[v] = False
    for w in adj[v]:
        if active[w]:
            degree[w] -= 1
            if degree[w] <= 2:
                queue.append(w)


def _safe_reductions(adj, forbidden, deadline):
    """Apply reductions whose choices occur in some maximum independent set."""
    n = len(adj)
    active = [not forbidden[v] for v in range(n)]
    degree = [sum(1 for w in adj[v] if active[w]) if active[v] else 0
              for v in range(n)]
    queue = deque(v for v in range(n) if active[v] and degree[v] <= 2)
    chosen = [False] * n

    while queue and time.time() < deadline:
        v = queue.popleft()
        if not active[v] or degree[v] > 2:
            continue
        neighbors = [w for w in adj[v] if active[w]]
        if not neighbors:
            chosen[v] = True
            _remove_vertex(v, active, degree, adj, queue)
        elif len(neighbors) == 1:
            # A leaf can always replace its only neighbor in an optimum.
            chosen[v] = True
            _remove_vertex(v, active, degree, adj, queue)
            _remove_vertex(neighbors[0], active, degree, adj, queue)
        elif neighbors[1] in adj[neighbors[0]]:
            # v and its two adjacent neighbors form a triangle.  Since v has
            # no outside neighbors, an optimum can choose v from the triangle.
            chosen[v] = True
            _remove_vertex(v, active, degree, adj, queue)
            _remove_vertex(neighbors[0], active, degree, adj, queue)
            _remove_vertex(neighbors[1], active, degree, adj, queue)

    return chosen, [v for v in range(n) if active[v]]


def _components(vertices, adj, deadline):
    allowed = set(vertices)
    seen = set()
    result = []
    for start in vertices:
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        component = []
        while stack:
            v = stack.pop()
            component.append(v)
            for w in adj[v]:
                if w in allowed and w not in seen:
                    seen.add(w)
                    stack.append(w)
        result.append(component)
        if time.time() >= deadline:
            # Components already discovered are still valid, but callers need
            # every vertex.  Put the unseen remainder into one safe work set.
            remainder = [v for v in vertices if v not in seen]
            if remainder:
                result.append(remainder)
            break
    return result


def _bipartite_independent_set(component, adj, deadline):
    """Return an exact MIS for a bipartite component, or None."""
    members = set(component)
    color = {}
    for start in component:
        if start in color:
            continue
        color[start] = 0
        queue = deque([start])
        while queue:
            if time.time() >= deadline:
                return None
            v = queue.popleft()
            for w in adj[v]:
                if w not in members:
                    continue
                if w not in color:
                    color[w] = 1 - color[v]
                    queue.append(w)
                elif color[w] == color[v]:
                    return None

    left = [v for v in component if color[v] == 0]
    pair_left = {v: -1 for v in left}
    pair_right = {v: -1 for v in component if color[v] == 1}
    distance = {}
    timed_out = [False]

    def bfs():
        queue = deque()
        found = False
        for u in left:
            if pair_left[u] == -1:
                distance[u] = 0
                queue.append(u)
            else:
                distance[u] = -1
        while queue:
            if time.time() >= deadline:
                timed_out[0] = True
                return False
            u = queue.popleft()
            for v in adj[u]:
                if v not in members:
                    continue
                mate = pair_right[v]
                if mate == -1:
                    found = True
                elif distance[mate] < 0:
                    distance[mate] = distance[u] + 1
                    queue.append(mate)
        return found

    def dfs(u):
        if time.time() >= deadline:
            timed_out[0] = True
            return False
        for v in adj[u]:
            if v not in members:
                continue
            mate = pair_right[v]
            if mate == -1 or (distance.get(mate, -1) == distance[u] + 1 and dfs(mate)):
                pair_left[u] = v
                pair_right[v] = u
                return True
        distance[u] = -1
        return False

    while bfs():
        for u in left:
            if pair_left[u] == -1:
                dfs(u)
                if timed_out[0]:
                    return None
    if timed_out[0]:
        return None

    # Konig's theorem: derive a minimum vertex cover from the maximum matching.
    reachable_left = {u for u in left if pair_left[u] == -1}
    reachable_right = set()
    queue = deque(reachable_left)
    while queue:
        if time.time() >= deadline:
            return None
        u = queue.popleft()
        for v in adj[u]:
            if v not in members:
                continue
            if pair_left[u] == v or v in reachable_right:
                continue
            reachable_right.add(v)
            mate = pair_right[v]
            if mate != -1 and mate not in reachable_left:
                reachable_left.add(mate)
                queue.append(mate)
    return reachable_left | (set(pair_right) - reachable_right)


def _greedy_construct(vertices, adj, initial, rcl_size, rng, deadline):
    """Dynamic minimum-residual-degree construction, optionally randomized."""
    n = len(adj)
    selected = bytearray(n)
    for v in initial:
        selected[v] = 1

    blocked = [0] * n
    for v in initial:
        for w in adj[v]:
            blocked[w] += 1

    available = bytearray(n)
    for v in vertices:
        if not selected[v] and blocked[v] == 0:
            available[v] = 1
    residual_degree = [0] * n
    heap = []
    for v in vertices:
        if available[v]:
            residual_degree[v] = sum(1 for w in adj[v] if available[w])
            tie = rng.random() if rcl_size > 1 else float(v)
            heapq.heappush(heap, (residual_degree[v], tie, v))

    while heap and time.time() < deadline:
        choices = []
        while heap and len(choices) < rcl_size:
            item = heapq.heappop(heap)
            d, _, v = item
            if available[v] and d == residual_degree[v]:
                choices.append(item)
        if not choices:
            continue
        pick = int((rng.random() ** 2) * len(choices)) if len(choices) > 1 else 0
        _, _, v = choices[pick]
        for i, item in enumerate(choices):
            if i != pick:
                heapq.heappush(heap, item)

        selected[v] = 1
        doomed = [v]
        doomed.extend(w for w in adj[v] if available[w])
        for removed in doomed:
            if not available[removed]:
                continue
            available[removed] = 0
            for w in adj[removed]:
                if available[w]:
                    residual_degree[w] -= 1
                    tie = rng.random() if rcl_size > 1 else float(w)
                    heapq.heappush(heap, (residual_degree[w], tie, w))
    return selected


def _one_to_two_local_search(selected, vertices, adj, rng, deadline):
    """Make the set maximal, then repeatedly replace one vertex by two."""
    while time.time() < deadline:
        initial = [v for v in vertices if selected[v]]
        selected = _greedy_construct(vertices, adj, initial, 2, rng, deadline)
        if time.time() >= deadline:
            return selected

        buckets = {}
        for v in vertices:
            if selected[v]:
                continue
            sole = -1
            conflicts = 0
            for w in adj[v]:
                if selected[w]:
                    conflicts += 1
                    sole = w
                    if conflicts > 1:
                        break
            if conflicts == 1:
                buckets.setdefault(sole, []).append(v)

        owners = list(buckets)
        rng.shuffle(owners)
        improved = False
        for owner in owners:
            if time.time() >= deadline:
                return selected
            candidates = buckets[owner]
            candidates.sort(key=lambda v: (len(adj[v]), rng.random()))
            for i, a in enumerate(candidates):
                if time.time() >= deadline:
                    return selected
                for b in candidates[i + 1:]:
                    if b not in adj[a]:
                        selected[owner] = 0
                        selected[a] = 1
                        selected[b] = 1
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            return selected
    return selected


def _perturb(selected, vertices, adj, rng, deadline):
    current = bytearray(selected)
    inside = [v for v in vertices if current[v]]
    if not inside:
        return _greedy_construct(vertices, adj, [], 5, rng, deadline)

    # Destroy a varying neighborhood.  Removing nearby selected vertices is
    # more useful than independent random flips on sparse graphs.
    outsiders = [v for v in vertices if not current[v]]
    removed = set()
    if outsiders and rng.random() < 0.65:
        pivot = rng.choice(outsiders)
        removed.update(w for w in adj[pivot] if current[w])
    target = 2 + int(rng.random() * max(2, min(12, len(inside) // 12 + 1)))
    while len(removed) < target and time.time() < deadline:
        removed.add(rng.choice(inside))
    for v in removed:
        current[v] = 0
    initial = [v for v in vertices if current[v]]
    return _greedy_construct(vertices, adj, initial, 3 + rng.randrange(6), rng, deadline)


def _exact_component(component, adj, incumbent, deadline):
    """Bit-parallel maximum-clique branch-and-bound on the complement graph."""
    if not component or time.time() >= deadline:
        return set(incumbent), False

    # Low-degree original vertices are promising MIS vertices.  This ordering
    # also tends to give the greedy coloring a useful first incumbent.
    ordered = sorted(component, key=lambda v: (len(adj[v]), v), reverse=True)
    position = {v: i for i, v in enumerate(ordered)}
    size = len(ordered)
    full = (1 << size) - 1
    complement = [0] * size
    for i, v in enumerate(ordered):
        if time.time() >= deadline:
            return set(incumbent), False
        original_neighbors = 0
        for w in adj[v]:
            j = position.get(w)
            if j is not None:
                original_neighbors |= 1 << j
        complement[i] = full & ~(original_neighbors | (1 << i))

    best_vertices = [position[v] for v in incumbent if v in position]
    best_size = [len(best_vertices)]
    best_bits = [sum(1 << v for v in best_vertices)]
    timed_out = [False]

    def color_sort(candidates):
        order = []
        bounds = []
        color = 0
        remaining = candidates
        while remaining:
            if time.time() >= deadline:
                timed_out[0] = True
                return None, None
            color += 1
            color_class = remaining
            while color_class:
                if time.time() >= deadline:
                    timed_out[0] = True
                    return None, None
                bit = color_class & -color_class
                v = bit.bit_length() - 1
                order.append(v)
                bounds.append(color)
                remaining ^= bit
                color_class ^= bit
                color_class &= ~complement[v]
        return order, bounds

    calls = [0]

    def expand(candidates, clique_bits, clique_size):
        calls[0] += 1
        if (calls[0] & 31) == 0 and time.time() >= deadline:
            timed_out[0] = True
            return
        order, bounds = color_sort(candidates)
        if timed_out[0]:
            return
        for index in range(len(order) - 1, -1, -1):
            if clique_size + bounds[index] <= best_size[0]:
                return
            if time.time() >= deadline:
                timed_out[0] = True
                return
            v = order[index]
            bit = 1 << v
            new_candidates = candidates & complement[v]
            new_bits = clique_bits | bit
            if new_candidates:
                expand(new_candidates, new_bits, clique_size + 1)
                if timed_out[0]:
                    return
            elif clique_size + 1 > best_size[0]:
                best_size[0] = clique_size + 1
                best_bits[0] = new_bits
            candidates &= ~bit

    expand(full, 0, 0)
    answer = {ordered[i] for i in range(size) if (best_bits[0] >> i) & 1}
    return answer, not timed_out[0]


def solve(instance: dict, time_limit_s: float) -> dict:
    t0 = time.time()
    try:
        n = max(0, int(instance.get("num_vertices", 0)))
    except Exception:
        n = 0
    fallback = {"vertex_selection": [0] * n}
    try:
        budget = max(0.0, float(time_limit_s))
        deadline = t0 + budget * 0.85
        if n == 0 or time.time() >= deadline:
            return fallback

        adj = [set() for _ in range(n)]
        forbidden = [False] * n
        for edge in instance.get("edges", []):
            if len(edge) < 2:
                continue
            u, v = edge[0], edge[1]
            if not isinstance(u, int) or not isinstance(v, int):
                continue
            if not (0 <= u < n and 0 <= v < n):
                continue
            if u == v:
                forbidden[u] = True
            else:
                adj[u].add(v)
                adj[v].add(u)

        fixed, remaining = _safe_reductions(adj, forbidden, deadline)
        if time.time() >= deadline:
            return {"vertex_selection": [int(v) for v in fixed]}

        # Exact polynomial-time handling of bipartite connected components.
        hard_components = []
        for component in _components(remaining, adj, deadline):
            try:
                exact = _bipartite_independent_set(component, adj, deadline)
            except RecursionError:
                exact = None
            if exact is None:
                hard_components.append(component)
            else:
                for v in exact:
                    fixed[v] = True
        hard_vertices = [v for component in hard_components for v in component]
        if not hard_vertices or time.time() >= deadline:
            return {"vertex_selection": [int(v) for v in fixed]}

        seed = n * 1000003 + sum((u + 1) * 97 + (v + 1) * 193
                                 for u in range(n) for v in adj[u] if u < v)
        rng = random.Random(seed)

        best = _greedy_construct(hard_vertices, adj, [], 1, rng, deadline)
        best = _one_to_two_local_search(best, hard_vertices, adj, rng, deadline)
        best_size = sum(best[v] for v in hard_vertices)
        current = bytearray(best)

        # Reserve most of the budget for exact branch-and-bound, but build a
        # strong incumbent first.  The loop has no iteration cap.
        heuristic_deadline = min(deadline, t0 + budget * 0.32)
        while time.time() < heuristic_deadline:
            if rng.random() < 0.24:
                candidate = _greedy_construct(
                    hard_vertices, adj, [], 2 + rng.randrange(10), rng,
                    heuristic_deadline)
            else:
                base = current if rng.random() < 0.55 else best
                candidate = _perturb(base, hard_vertices, adj, rng,
                                     heuristic_deadline)
            candidate = _one_to_two_local_search(
                candidate, hard_vertices, adj, rng, heuristic_deadline)
            candidate_size = sum(candidate[v] for v in hard_vertices)
            if candidate_size > best_size:
                best = bytearray(candidate)
                best_size = candidate_size
            if candidate_size >= best_size - 2 or rng.random() < 0.08:
                current = candidate
            else:
                current = bytearray(best)

        # Components are additive.  Prove/improve small components first so a
        # single hard component cannot starve all easy exact wins.
        complete = True
        for component in sorted(hard_components, key=len):
            if time.time() >= deadline:
                complete = False
                break
            incumbent = {v for v in component if best[v]}
            answer, proven = _exact_component(component, adj, incumbent, deadline)
            if len(answer) > len(incumbent):
                for v in component:
                    best[v] = int(v in answer)
                best_size += len(answer) - len(incumbent)
            if not proven:
                complete = False
                break

        # If exact search finished early, keep converting spare wall time into
        # additional diversified attempts (unless all components were proved).
        while not complete and time.time() < deadline:
            candidate = _perturb(best, hard_vertices, adj, rng, deadline)
            candidate = _one_to_two_local_search(
                candidate, hard_vertices, adj, rng, deadline)
            candidate_size = sum(candidate[v] for v in hard_vertices)
            if candidate_size > best_size:
                best = candidate
                best_size = candidate_size

        result = [int(fixed[v] or best[v]) for v in range(n)]
        # A final inexpensive feasibility guard protects the fallback contract.
        for u in range(n):
            if not result[u]:
                continue
            if forbidden[u]:
                result[u] = 0
                continue
            for v in adj[u]:
                if v > u and result[v]:
                    result[v] = 0
        return {"vertex_selection": result}
    except Exception:
        return fallback
