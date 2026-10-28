import math
import random
import time


def solve(instance: dict, time_limit_s: float) -> dict:
    """Anytime heuristic for the Euclidean Steiner tree problem."""
    t0 = time.time()
    deadline = t0 + max(0.0, float(time_limit_s)) * 0.85

    terminals = [
        (float(point[0]), float(point[1])) for point in instance.get("points", [])
    ]
    n = len(terminals)
    max_steiner = max(0, int(instance.get("max_steiner_points", 0)))

    if n <= 1:
        return {"steiner_points": [], "edges": []}

    def distance(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def tree_cost(coords, edges):
        return sum(distance(coords[u], coords[v]) for u, v in edges)

    def prim_tree(coords, noise, rng, stop_at=None):
        count = len(coords)
        used = [False] * count
        key = [float("inf")] * count
        parent = [-1] * count
        key[0] = 0.0
        edges = []
        stopped = False
        for _ in range(count):
            if stop_at is not None and time.time() >= stop_at:
                stopped = True
                break
            u = -1
            best_key = float("inf")
            for i in range(count):
                if not used[i] and key[i] < best_key:
                    best_key = key[i]
                    u = i
            if u < 0:
                break
            used[u] = True
            if parent[u] >= 0:
                edges.append((parent[u], u))
            ux, uy = coords[u]
            for v in range(count):
                if used[v]:
                    continue
                vx, vy = coords[v]
                d = math.hypot(ux - vx, uy - vy)
                weighted = d
                if noise > 0.0:
                    weighted *= 1.0 + noise * (2.0 * rng.random() - 1.0)
                if weighted < key[v]:
                    key[v] = weighted
                    parent[v] = u
        if stopped:
            # A plain star completes a partial Prim tree at the cutoff.
            for v in range(1, count):
                if not used[v]:
                    edges.append((0, v))
        return edges

    # A complete feasible answer exists before any expensive search begins.
    fallback_edges = [(i - 1, i) for i in range(1, n)]
    best_coords = terminals[:]
    best_edges = fallback_edges
    best_cost = tree_cost(best_coords, best_edges)

    seed = n * 1000003 + max_steiner * 9176
    for i, (x, y) in enumerate(terminals):
        seed ^= int(abs(x) * 1009.0 + abs(y) * 917.0 + i * 131) & 0xFFFFFFFF
    rng = random.Random(seed)

    base_edges = prim_tree(terminals, 0.0, rng, deadline)
    base_cost = tree_cost(terminals, base_edges)
    if base_cost < best_cost:
        best_edges = base_edges
        best_cost = base_cost

    if max_steiner == 0 or time.time() >= deadline:
        return {"steiner_points": [], "edges": best_edges}

    scale = max(
        max(point[0] for point in terminals) - min(point[0] for point in terminals),
        max(point[1] for point in terminals) - min(point[1] for point in terminals),
        1.0,
    )
    improve_eps = 1e-10 * scale

    def adjacency(count, edges):
        adj = [[] for _ in range(count)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        return adj

    def geometric_median(points, start=None, steps=18):
        if not points:
            return (0.0, 0.0)
        if len(points) == 1:
            return points[0]
        if len(points) == 3:
            for i in range(3):
                a = points[i]
                b = points[(i + 1) % 3]
                c = points[(i + 2) % 3]
                abx, aby = b[0] - a[0], b[1] - a[1]
                acx, acy = c[0] - a[0], c[1] - a[1]
                lab = math.hypot(abx, aby)
                lac = math.hypot(acx, acy)
                if lab == 0.0 or lac == 0.0:
                    return a
                if abx * acx + aby * acy <= -0.5 * lab * lac:
                    return a
        if start is None:
            x = sum(p[0] for p in points) / len(points)
            y = sum(p[1] for p in points) / len(points)
        else:
            x, y = start
        for _ in range(steps):
            num_x = 0.0
            num_y = 0.0
            den = 0.0
            coincident = None
            for px, py in points:
                d = math.hypot(x - px, y - py)
                if d < 1e-13 * scale:
                    coincident = (px, py)
                    break
                w = 1.0 / d
                num_x += px * w
                num_y += py * w
                den += w
            if coincident is not None:
                return coincident
            nx, ny = num_x / den, num_y / den
            if math.hypot(nx - x, ny - y) < 1e-11 * scale:
                return (nx, ny)
            x, y = nx, ny
        return (x, y)

    def compact(coords, edges):
        coords = coords[:]
        edges = [tuple(edge) for edge in edges]
        while True:
            adj = adjacency(len(coords), edges)
            remove = -1
            for vertex in range(n, len(coords)):
                if len(adj[vertex]) <= 2:
                    remove = vertex
                    break
            if remove < 0:
                break
            neighbors = adj[remove]
            edges = [edge for edge in edges if remove not in edge]
            if len(neighbors) == 2:
                edges.append((neighbors[0], neighbors[1]))
            coords.pop(remove)
            remapped = []
            for u, v in edges:
                if u > remove:
                    u -= 1
                if v > remove:
                    v -= 1
                remapped.append((u, v))
            edges = remapped
        return coords, edges

    def optimize_geometry(coords, edges, passes=12):
        coords = coords[:]
        for _ in range(passes):
            if time.time() >= deadline:
                break
            adj = adjacency(len(coords), edges)
            largest_move = 0.0
            for vertex in range(n, len(coords)):
                if vertex % 32 == 0 and time.time() >= deadline:
                    break
                neighbor_points = [coords[other] for other in adj[vertex]]
                new_point = geometric_median(
                    neighbor_points, coords[vertex], 16 if len(neighbor_points) == 3 else 24
                )
                move = distance(coords[vertex], new_point)
                if move > largest_move:
                    largest_move = move
                coords[vertex] = new_point
            if largest_move < 2e-10 * scale:
                break
        return coords, edges, tree_cost(coords, edges)

    def best_split(coords, edges):
        adj = adjacency(len(coords), edges)
        choice = None
        best_gain = improve_eps
        for center, neighbors in enumerate(adj):
            if center >= n and len(neighbors) <= 3:
                continue
            degree = len(neighbors)
            for ia in range(degree):
                if time.time() >= deadline:
                    break
                a = neighbors[ia]
                for ib in range(ia + 1, degree):
                    b = neighbors[ib]
                    point = geometric_median(
                        [coords[center], coords[a], coords[b]], None, 20
                    )
                    old = distance(coords[center], coords[a]) + distance(
                        coords[center], coords[b]
                    )
                    new = (
                        distance(point, coords[center])
                        + distance(point, coords[a])
                        + distance(point, coords[b])
                    )
                    gain = old - new
                    if gain > best_gain:
                        best_gain = gain
                        choice = (center, a, b, point)
            if time.time() >= deadline:
                break
        return choice

    def add_split(coords, edges, choice):
        center, a, b, point = choice
        new_index = len(coords)
        new_edges = []
        for u, v in edges:
            if (u == center and v in (a, b)) or (v == center and u in (a, b)):
                continue
            new_edges.append((u, v))
        new_edges.extend([(new_index, center), (new_index, a), (new_index, b)])
        return coords + [point], new_edges

    def steinerize(coords, edges):
        coords = coords[:]
        edges = edges[:]
        while len(coords) - n < max_steiner and time.time() < deadline:
            choice = best_split(coords, edges)
            if choice is None:
                break
            coords, edges = add_split(coords, edges, choice)
            coords, edges, _ = optimize_geometry(coords, edges, 10)
        return optimize_geometry(coords, edges, 18)

    def euclidean_mst_refresh(coords, edges):
        refreshed = prim_tree(coords, 0.0, rng, deadline)
        refreshed_coords, refreshed = compact(coords, refreshed)
        return optimize_geometry(refreshed_coords, refreshed, 14)

    def improve_nni(coords, edges):
        current_coords = coords
        current_edges = edges
        current_cost = tree_cost(coords, edges)
        changed = True
        while changed and time.time() < deadline:
            changed = False
            adj = adjacency(len(current_coords), current_edges)
            for left, right in current_edges[:]:
                if left < n or right < n:
                    continue
                if len(adj[left]) != 3 or len(adj[right]) != 3:
                    continue
                left_neighbors = [v for v in adj[left] if v != right]
                right_neighbors = [v for v in adj[right] if v != left]
                for swap_right in right_neighbors:
                    if time.time() >= deadline:
                        return current_coords, current_edges, current_cost
                    swap_left = left_neighbors[1]
                    trial_edges = []
                    for u, v in current_edges:
                        pair = (u, v)
                        if pair in ((left, swap_left), (swap_left, left)):
                            continue
                        if pair in ((right, swap_right), (swap_right, right)):
                            continue
                        trial_edges.append((u, v))
                    trial_edges.extend([(left, swap_right), (right, swap_left)])
                    trial_coords, trial_edges, trial_cost = optimize_geometry(
                        current_coords, trial_edges, 10
                    )
                    if trial_cost + improve_eps < current_cost:
                        current_coords = trial_coords
                        current_edges = trial_edges
                        current_cost = trial_cost
                        changed = True
                        break
                if changed:
                    break
        return current_coords, current_edges, current_cost

    def candidate_triples(coords, edges, limit):
        count = len(coords)
        candidates = []
        seen = set()
        adj = adjacency(count, edges)
        for center in range(count):
            if time.time() >= deadline:
                break
            near = sorted(
                (distance(coords[center], coords[other]), other)
                for other in range(count)
                if other != center
            )[: min(7, count - 1)]
            pool = list(adj[center])
            for _, other in near:
                if other not in pool:
                    pool.append(other)
            pool = pool[:9]
            for i in range(len(pool)):
                for j in range(i + 1, len(pool)):
                    triple = tuple(sorted((center, pool[i], pool[j])))
                    if len(set(triple)) == 3 and triple not in seen:
                        seen.add(triple)
                        candidates.append(triple)
        rng.shuffle(candidates)
        candidates.sort(
            key=lambda tri: sum(
                distance(coords[tri[i]], coords[tri[(i + 1) % 3]])
                for i in range(3)
            )
        )
        if len(candidates) > limit:
            regular = candidates[: limit * 2 // 3]
            tail = candidates[limit * 2 // 3 :]
            rng.shuffle(tail)
            candidates = regular + tail[: limit - len(regular)]
        return candidates

    def augmented_tree(coords, edges, point):
        new_index = len(coords)
        weighted = [
            (distance(coords[u], coords[v]), u, v) for u, v in edges
        ]
        weighted.extend(
            (distance(point, coords[v]), new_index, v) for v in range(new_index)
        )
        weighted.sort(key=lambda item: item[0])
        parent = list(range(new_index + 1))
        rank = [0] * (new_index + 1)

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        result = []
        for _, u, v in weighted:
            ru, rv = find(u), find(v)
            if ru == rv:
                continue
            if rank[ru] < rank[rv]:
                ru, rv = rv, ru
            parent[rv] = ru
            if rank[ru] == rank[rv]:
                rank[ru] += 1
            result.append((u, v))
            if len(result) == new_index:
                break
        return result

    def one_steiner_pass(coords, edges):
        if len(coords) - n >= max_steiner:
            return coords, edges, tree_cost(coords, edges), False
        current_cost = tree_cost(coords, edges)
        best_trial = None
        # Cap each pass so large instances retain time for restarts and polishing.
        limit = max(24, min(180, 3600 // max(3, len(coords))))
        for triple in candidate_triples(coords, edges, limit):
            if time.time() >= deadline:
                break
            point = geometric_median([coords[i] for i in triple], None, 22)
            trial_edges = augmented_tree(coords, edges, point)
            trial_coords, trial_edges = compact(coords + [point], trial_edges)
            if len(trial_coords) == len(coords):
                continue
            raw_cost = tree_cost(trial_coords, trial_edges)
            if raw_cost + improve_eps < current_cost:
                if best_trial is None or raw_cost < best_trial[0]:
                    best_trial = (raw_cost, trial_coords, trial_edges)
        if best_trial is None:
            return coords, edges, current_cost, False
        trial_coords, trial_edges, trial_cost = optimize_geometry(
            best_trial[1], best_trial[2], 16
        )
        return trial_coords, trial_edges, trial_cost, trial_cost + improve_eps < current_cost

    def record(coords, edges, cost):
        nonlocal best_coords, best_edges, best_cost
        if cost + improve_eps < best_cost:
            best_coords = coords[:]
            best_edges = edges[:]
            best_cost = cost

    try:
        coords, edges, cost = steinerize(terminals, base_edges)
        record(coords, edges, cost)

        if time.time() < deadline:
            coords, edges, cost = euclidean_mst_refresh(coords, edges)
            coords, edges, cost = steinerize(coords, edges)
            record(coords, edges, cost)

        if time.time() < deadline:
            coords, edges, cost = improve_nni(coords, edges)
            record(coords, edges, cost)

        # Iterated 1-Steiner insertion can create useful triples that are not
        # adjacent in the terminal MST.
        probe_coords, probe_edges = terminals[:], base_edges[:]
        while time.time() < deadline and len(probe_coords) - n < max_steiner:
            probe_coords, probe_edges, probe_cost, changed = one_steiner_pass(
                probe_coords, probe_edges
            )
            if not changed:
                break
            probe_coords, probe_edges, probe_cost = steinerize(
                probe_coords, probe_edges
            )
            record(probe_coords, probe_edges, probe_cost)

        restart = 0
        while time.time() < deadline:
            # Near-MST restarts expose different terminal triples; unlike point
            # jitter, this changes the combinatorial topology deliberately.
            noise_levels = (0.008, 0.02, 0.05, 0.11, 0.20)
            noise = noise_levels[restart % len(noise_levels)]
            trial_edges = prim_tree(terminals, noise, rng, deadline)
            trial_coords, trial_edges, trial_cost = steinerize(
                terminals, trial_edges
            )
            if time.time() < deadline:
                trial_coords, trial_edges, trial_cost = improve_nni(
                    trial_coords, trial_edges
                )
            if time.time() < deadline:
                refreshed_coords, refreshed_edges, refreshed_cost = euclidean_mst_refresh(
                    trial_coords, trial_edges
                )
                refreshed_coords, refreshed_edges, refreshed_cost = steinerize(
                    refreshed_coords, refreshed_edges
                )
                if refreshed_cost < trial_cost:
                    trial_coords, trial_edges, trial_cost = (
                        refreshed_coords,
                        refreshed_edges,
                        refreshed_cost,
                    )
            record(trial_coords, trial_edges, trial_cost)
            restart += 1
    except Exception:
        # The baseline MST remains a complete feasible answer even if an
        # optional improvement operator encounters a numerical corner case.
        pass

    steiner_points = [
        (float(x), float(y)) for x, y in best_coords[n:]
    ]
    output_edges = [(int(u), int(v)) for u, v in best_edges]
    return {"steiner_points": steiner_points, "edges": output_edges}
