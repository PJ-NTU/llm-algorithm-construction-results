"""Iterative search with analytic Fermat points and subtree re-hanging
Combines minimum spanning trees, analytic geometric relaxation, two-node swaps and subtree splice-and-rehang, with local destroy-and-rebuild for time-limited search.
"""
import heapq
import math
import random
import time

try:
    import numpy as np
except ImportError:
    np = None


def solve(instance: dict, time_limit_s: float) -> dict:
    clock = time.perf_counter
    started = clock()
    try:
        allowance = max(0.0, float(time_limit_s))
    except Exception:
        allowance = 0.0
    end = started + allowance * 0.975
    search_end = started + allowance * 0.94
    terminals = []
    maximum = 0
    # First save a complete tree that can be returned without optimisation.
    try:
        terminals = [(float(p[0]), float(p[1])) for p in instance.get("points", [])]
        maximum = max(0, int(instance.get("max_steiner_points", 0)))
    except Exception:
        return {"steiner_points": [], "edges": []}
    n = len(terminals)
    best_points = terminals[:]
    best_edges = [(v - 1, v) for v in range(1, n)]
    origin = (0.0, 0.0)
    scale = 1.0
    normalized = False

    def answer():
        if normalized:
            extra = [[float(origin[0] + scale * x), float(origin[1] + scale * y)]
                     for x, y in best_points[n:]]
        else:
            extra = [[float(x), float(y)] for x, y in best_points[n:]]
        result_edges = [[int(u), int(v)] for u, v in best_edges]
        anchor = list(terminals[0]) if n else [0.0, 0.0]
        # Padding coordinates does not increase length; each padding point attaches with a single zero-length edge.
        while len(extra) < maximum:
            vertex = n + len(extra)
            extra.append(anchor[:])
            if vertex:
                result_edges.append([0, vertex])
        return {"steiner_points": extra, "edges": result_edges}

    if n < 2 or clock() >= search_end:
        return answer()

    try:
        minx = min(p[0] for p in terminals)
        miny = min(p[1] for p in terminals)
        span = max(max(p[0] for p in terminals) - minx,
                   max(p[1] for p in terminals) - miny)
        if span == 0.0:
            return answer()
        if not math.isfinite(span):
            return answer()
        origin = (minx, miny)
        scale = span
        points = [((x - minx) / span, (y - miny) / span) for x, y in terminals]
        best_points = points[:]
        normalized = True
        capacity = min(maximum, max(0, n - 2))
        eps = 2e-12
        root3 = math.sqrt(3.0)
        horizon = math.log1p(allowance)
        width = min(max(1, 2 * n - 3),
                    max(3, int(math.ceil(math.sqrt(1.0 + allowance) * math.log2(n + 2)))))
        local_depth = max(1, int(math.ceil(horizon)))
        epoch = allowance / (2.0 + math.log2(n + 2))
        quantum = allowance / ((n + 1) * (8.0 + 2.0 * horizon))
        seed = n * 1000003 + capacity * 9176
        for i, (x, y) in enumerate(points):
            seed ^= (int(x * 10000019) + int(y * 10000169) + i * 131) & 0xffffffff
        rng = random.Random(seed)
        nrng = np.random.default_rng(seed) if np is not None else None
        hypot = math.hypot

        def dist(a, b):
            return hypot(a[0] - b[0], a[1] - b[1])

        def cost(p, e):
            return sum(hypot(p[a][0] - p[b][0], p[a][1] - p[b][1]) for a, b in e)

        best_cost = cost(points, best_edges)

        def record(p, e, value=None):
            nonlocal best_points, best_edges, best_cost
            if value is None:
                value = cost(p, e)
            if value < best_cost - eps:
                best_points, best_edges, best_cost = p[:], e[:], value
            return value

        def adjacency(count, edges):
            adj = [[] for _ in range(count)]
            for a, b in edges:
                adj[a].append(b)
                adj[b].append(a)
            return adj

        def mst(p, noise, stop):
            count = len(p)
            result = []
            used = [False] * count
            parent = [-1] * count
            if np is not None:
                xy = np.asarray(p, dtype=float)
                keys = np.full(count, np.inf)
                marked = np.zeros(count, dtype=bool)
                par = np.full(count, -1, dtype=int)
                keys[0] = 0.0
                for _ in range(count):
                    if clock() >= stop:
                        break
                    u = int(np.argmin(keys))
                    if marked[u]:
                        break
                    used[u] = True
                    marked[u] = True
                    if par[u] >= 0:
                        result.append((int(par[u]), u))
                    delta = xy - xy[u]
                    length = np.hypot(delta[:, 0], delta[:, 1])
                    if noise:
                        length *= nrng.uniform(1.0 - noise, 1.0 + noise, count)
                    improve = (length < keys) & ~marked
                    keys[improve] = length[improve]
                    par[improve] = u
                    keys[marked] = np.inf
            else:
                keys = [float("inf")] * count
                keys[0] = 0.0
                for _ in range(count):
                    if clock() >= stop:
                        break
                    u = min((v for v in range(count) if not used[v]), key=keys.__getitem__)
                    used[u] = True
                    if parent[u] >= 0:
                        result.append((parent[u], u))
                    for v in range(count):
                        if not used[v]:
                            length = dist(p[u], p[v])
                            if noise:
                                length *= 1.0 + noise * (2.0 * rng.random() - 1.0)
                            if length < keys[v]:
                                keys[v], parent[v] = length, u
            # When Prim is interrupted at the deadline, the selected part plus star edges to the root is still a tree.
            result.extend((0, v) for v in range(1, count) if not used[v])
            return result

        def fermat(a, b, c):
            abx, aby = b[0] - a[0], b[1] - a[1]
            acx, acy = c[0] - a[0], c[1] - a[1]
            bcx, bcy = c[0] - b[0], c[1] - b[1]
            cc = abx * abx + aby * aby
            bb = acx * acx + acy * acy
            aa = bcx * bcx + bcy * bcy
            if cc == 0.0 or bb == 0.0:
                return a
            if aa == 0.0:
                return b
            da, db, dc = bb + cc - aa, aa + cc - bb, aa + bb - cc
            if da <= -math.sqrt(bb * cc):
                return a
            if db <= -math.sqrt(aa * cc):
                return b
            if dc <= -math.sqrt(aa * bb):
                return c
            area2 = abs(abx * acy - aby * acx)
            va, vb, vc = 2.0 * area2 + root3 * da, 2.0 * area2 + root3 * db, 2.0 * area2 + root3 * dc
            if min(va, vb, vc) <= 0.0:
                return min((a, b, c), key=lambda z: dist(z, a) + dist(z, b) + dist(z, c))
            # Normalised reciprocals avoid overflow near 120 degrees.
            small = min(va, vb, vc)
            wa, wb, wc = small / va, small / vb, small / vc
            total = wa + wb + wc
            return ((wa * a[0] + wb * b[0] + wc * c[0]) / total,
                    (wa * a[1] + wb * b[1] + wc * c[1]) / total)

        def median(neighbors, old, stop):
            if len(neighbors) == 3:
                return fermat(*neighbors)
            if not neighbors:
                return old
            if len(neighbors) == 1:
                return neighbors[0]
            if len(neighbors) == 2:
                return ((neighbors[0][0] + neighbors[1][0]) * 0.5,
                        (neighbors[0][1] + neighbors[1][1]) * 0.5)
            x, y = old
            while clock() < stop:
                sx = sy = sw = rx = ry = 0.0
                coincident = 0
                for px, py in neighbors:
                    d = hypot(px - x, py - y)
                    if d < 1e-14:
                        coincident += 1
                    else:
                        w = 1.0 / d
                        sx += w * px
                        sy += w * py
                        sw += w
                        rx += w * (px - x)
                        ry += w * (py - y)
                if sw == 0.0:
                    return (x, y)
                r = hypot(rx, ry)
                if r <= coincident:
                    return (x, y)
                fraction = coincident / r if coincident else 0.0
                nx = (1.0 - fraction) * sx / sw + fraction * x
                ny = (1.0 - fraction) * sy / sw + fraction * y
                if hypot(nx - x, ny - y) < 2e-11:
                    return (nx, ny)
                x, y = nx, ny
            return (x, y)

        def compact(p, e, collapse=False):
            adj = [set(a) for a in adjacency(len(p), e)]
            live = [True] * len(p)
            queue = list(range(n, len(p)))
            while queue:
                u = queue.pop()
                if not live[u]:
                    continue
                neighbors = list(adj[u])
                merge = None
                if collapse:
                    merge = next((v for v in neighbors if dist(p[u], p[v]) < 2e-13), None)
                if len(neighbors) > 2 and merge is None:
                    continue
                for v in neighbors:
                    adj[v].discard(u)
                if merge is not None:
                    for v in neighbors:
                        if v != merge:
                            adj[merge].add(v)
                            adj[v].add(merge)
                elif len(neighbors) == 2:
                    a, b = neighbors
                    adj[a].add(b)
                    adj[b].add(a)
                live[u] = False
                adj[u].clear()
                queue.extend(v for v in neighbors if v >= n)
            ids = {}
            result = []
            for u, alive in enumerate(live):
                if alive:
                    ids[u] = len(result)
                    result.append(p[u])
            edges = [(ids[u], ids[v]) for u in ids for v in adj[u] if u < v]
            return result, edges

        def relax(p, e, stop, seeds=None):
            q = p[:]
            adj = adjacency(len(q), e)
            if seeds is None:
                vertices = list(range(n, len(q)))
            else:
                active = {u for u in seeds if u >= n}
                active.update(v for u in seeds for v in adj[u] if v >= n)
                front = set(active)
                for _ in range(local_depth):
                    front = {v for u in front for v in adj[u] if v >= n} - active
                    active.update(front)
                    if not front:
                        break
                vertices = sorted(active)
            if not vertices:
                return q, cost(q, e)
            while clock() < stop:
                movement = 0.0
                for index, u in enumerate(vertices):
                    if index % 16 == 0 and clock() >= stop:
                        break
                    new = median([q[v] for v in adj[u]], q[u], min(stop, clock() + quantum))
                    movement = max(movement, dist(q[u], new))
                    q[u] = new
                if movement < 2e-10:
                    break
                vertices.reverse()
            return q, cost(q, e)

        def grow(p, e, stop):
            p, e = p[:], e[:]
            while len(p) - n < capacity and clock() < stop:
                adj = adjacency(len(p), e)
                choice = None
                gain = eps
                for u, neighbors in enumerate(adj):
                    if clock() >= stop:
                        break
                    if u >= n and len(neighbors) <= 3:
                        continue
                    for i, a in enumerate(neighbors):
                        for b in neighbors[i + 1:]:
                            z = fermat(p[u], p[a], p[b])
                            value = dist(p[u], p[a]) + dist(p[u], p[b]) - dist(z, p[u]) - dist(z, p[a]) - dist(z, p[b])
                            if value > gain:
                                gain, choice = value, (u, a, b, z)
                        if clock() >= stop:
                            break
                if choice is None:
                    break
                u, a, b, z = choice
                new = len(p)
                e = [(v, w) for v, w in e
                     if not ((v == u and w in (a, b)) or (w == u and v in (a, b)))]
                e.extend(((u, new), (a, new), (b, new)))
                p.append(z)
                p, value = relax(p, e, min(stop, clock() + quantum), (u, a, b, new))
                record(p, e, value)
            p, value = relax(p, e, min(stop, clock() + quantum * math.sqrt(1 + len(p) - n)))
            record(p, e, value)
            return p, e, value

        def refresh(p, e, stop):
            value = cost(p, e)
            candidate = mst(p, 0.0, stop)
            if cost(p, candidate) > value + eps:
                return p, e, value
            q, f = compact(p, candidate, True)
            q, value2 = relax(q, f, min(stop, clock() + quantum * math.sqrt(1 + len(q) - n)))
            if value2 < value + eps:
                record(q, f, value2)
                return q, f, value2
            return p, e, value

        def nni(p, e, stop):
            value = cost(p, e)
            while clock() < stop:
                adj = adjacency(len(p), e)
                links = [(u, v) for u, v in e if u >= n and v >= n and len(adj[u]) == len(adj[v]) == 3]
                rng.shuffle(links)
                changed = False
                for u, v in links:
                    a, b = [w for w in adj[u] if w != v]
                    c, d = [w for w in adj[v] if w != u]
                    oldlocal = dist(p[u], p[a]) + dist(p[u], p[b]) + dist(p[u], p[v]) + dist(p[v], p[c]) + dist(p[v], p[d])
                    for w, other in ((c, d), (d, c)):
                        if clock() >= stop:
                            return p, e, value
                        # Separate the two nodes again, so zero-length edges do not stall coordinate descent.
                        su = ((p[a][0] + p[w][0]) * 0.5, (p[a][1] + p[w][1]) * 0.5)
                        sv = ((p[b][0] + p[other][0]) * 0.5, (p[b][1] + p[other][1]) * 0.5)
                        pair_end = min(stop, clock() + quantum)
                        while clock() < pair_end:
                            nu = fermat(p[a], p[w], sv)
                            nv = fermat(p[b], p[other], nu)
                            delta = dist(nu, su) + dist(nv, sv)
                            su, sv = nu, nv
                            if delta < 2e-10:
                                break
                        newlocal = dist(su, p[a]) + dist(su, p[w]) + dist(su, sv) + dist(sv, p[b]) + dist(sv, p[other])
                        # Adjacent branches may also move; only clearly worsening swaps are discarded.
                        if newlocal > oldlocal * (1.0 + 0.02 * horizon):
                            continue
                        f = [(i, j) for i, j in e if not ((i == u and j == b) or (i == b and j == u) or (i == v and j == w) or (i == w and j == v))]
                        f.extend(((u, w), (v, b)))
                        q = p[:]
                        q[u], q[v] = su, sv
                        q, trial = relax(q, f, min(stop, clock() + quantum), (u, v, a, b, c, d))
                        if trial < value - eps:
                            p, e, value = q, f, trial
                            record(p, e, value)
                            changed = True
                            break
                    if changed:
                        break
                if not changed:
                    break
            return p, e, value

        def regraft(p, e, stop):
            value = cost(p, e)
            adj = adjacency(len(p), e)
            # Use DFS intervals to identify the cut subtree, ensuring re-hanging creates no cycle.
            parent = [-1] * len(p)
            tin, tout = [0] * len(p), [0] * len(p)
            stack = [(0, -1, False)]
            tick = 0
            while stack:
                u, par, closing = stack.pop()
                if closing:
                    tout[u] = tick
                    continue
                parent[u], tin[u] = par, tick
                tick += 1
                stack.append((u, par, True))
                stack.extend((v, u, False) for v in adj[u] if v != par)
            nodes = [u for u in range(n, len(p)) if len(adj[u]) == 3]
            rng.shuffle(nodes)
            lengths = [dist(p[c], p[d]) for c, d in e]
            attempts = max(1, int(math.ceil(horizon * math.log2(n + 2) / math.sqrt(n))))
            for u in nodes:
                if clock() >= stop:
                    break
                choices = []
                for t in adj[u]:
                    a, b = [v for v in adj[u] if v != t]
                    saving = dist(p[u], p[a]) + dist(p[u], p[b]) + dist(p[u], p[t]) - dist(p[a], p[b])
                    ranked = []
                    tx, ty = p[t]
                    for index, (c, d) in enumerate(e):
                        if index % 128 == 0 and clock() >= stop:
                            break
                        if u == c or u == d:
                            continue
                        if parent[t] == u:
                            forbidden = tin[t] <= tin[c] < tout[t]
                        else:
                            forbidden = not (tin[u] <= tin[c] < tout[u])
                        if forbidden:
                            continue
                        dx, dy = p[d][0] - p[c][0], p[d][1] - p[c][1]
                        norm = dx * dx + dy * dy
                        alpha = max(0.0, min(1.0, ((tx - p[c][0]) * dx + (ty - p[c][1]) * dy) / norm)) if norm else 0.0
                        separation = (p[c][0] + alpha * dx - tx) ** 2 + (p[c][1] + alpha * dy - ty) ** 2
                        ranked.append((separation, index))
                    for _, index in heapq.nsmallest(width, ranked):
                        c, d = e[index]
                        z = fermat(p[t], p[c], p[d])
                        gain = saving + lengths[index] - dist(z, p[t]) - dist(z, p[c]) - dist(z, p[d])
                        choices.append((-gain, t, a, b, c, d, z))
                choices.sort(key=lambda item: item[0])
                for negative_gain, t, a, b, c, d, z in choices[:attempts]:
                    if clock() >= stop:
                        break
                    local_scale = dist(p[u], p[t]) + dist(p[u], p[a]) + dist(p[u], p[b]) + dist(p[c], p[d])
                    if negative_gain > local_scale * min(0.5, 0.04 * (1.0 + horizon)):
                        continue
                    f = [(v, w) for v, w in e if not (
                        (v == u and w in (a, b)) or (w == u and v in (a, b)) or
                        (v == c and w == d) or (v == d and w == c))]
                    f.extend(((a, b), (u, c), (u, d)))
                    q = p[:]
                    q[u] = z
                    q, trial = relax(q, f, min(stop, clock() + quantum), (u, t, a, b, c, d))
                    if trial < value - eps:
                        q, f = compact(q, f, True)
                        trial = record(q, f)
                        return q, f, trial, True
            return p, e, value, False

        def destroy(p, e, fraction):
            q = p[:]
            adj = [set(a) for a in adjacency(len(q), e)]
            active = set(range(n, len(q)))
            if not active:
                return q, e[:]
            anchor = q[rng.choice(sorted(active))]
            amount = min(len(active), max(1, int(math.ceil(fraction * math.sqrt(len(active)) * (1.0 + horizon)))))
            selected = heapq.nsmallest(amount, active, key=lambda u: dist(q[u], anchor) * (0.5 + rng.random()))
            for u in selected:
                if not adj[u]:
                    continue
                v = min(adj[u], key=lambda w: dist(q[u], q[w]))
                for w in list(adj[u]):
                    adj[w].discard(u)
                    if w != v:
                        adj[w].add(v)
                        adj[v].add(w)
                adj[u].clear()
                active.discard(u)
            live = list(range(n)) + sorted(active)
            ids = {u: i for i, u in enumerate(live)}
            f = [(ids[u], ids[v]) for u in live for v in adj[u] if u < v]
            q, f = compact([q[u] for u in live], f, True)
            return q, f

        def polish(p, e, stop):
            now = clock()
            p, e, value = grow(p, e, min(stop, now + max(0.0, stop - now) * 0.35))
            if clock() < stop:
                p, e, value = refresh(p, e, min(stop, clock() + max(quantum, (stop - clock()) * 0.12)))
                p, e, value = grow(p, e, min(stop, clock() + max(0.0, stop - clock()) * 0.2))
            while clock() < stop:
                before = value
                p, e, value = nni(p, e, min(stop, clock() + max(0.0, stop - clock()) * 0.35))
                p, e, value, changed = regraft(p, e, stop)
                if not changed and value >= before - eps:
                    break
                p, e, value = grow(p, e, min(stop, clock() + quantum * math.sqrt(1 + len(p) - n)))
            record(p, e, value)
            return p, e, value

        base = mst(points, 0.0, search_end)
        record(points, base)
        if capacity == 0:
            return answer()
        initial_end = min(search_end, started + allowance * 0.28)
        current, edges, current_cost = polish(points, base, initial_end)
        failures = 0
        while clock() < search_end:
            old_best = best_cost
            if rng.random() < 1.0 / (2.0 + horizon):
                noise = min(0.45, (0.01 + 0.025 * horizon) * math.exp(rng.uniform(-1.5, 1.5)))
                trial_points = points[:]
                trial_edges = mst(points, noise, min(search_end, clock() + epoch * 0.25))
            else:
                # Prefer keeping effective branches of the champion, then gradually enlarge the region to rebuild.
                if rng.random() < 0.7:
                    trial_points, trial_edges = best_points, best_edges
                else:
                    trial_points, trial_edges = current, edges
                strength = min(1.0, (0.12 + rng.random() * 0.3) * math.sqrt(1.0 + failures / (1.0 + horizon)))
                trial_points, trial_edges = destroy(trial_points, trial_edges, strength)
            trial_points, trial_edges, trial_cost = polish(trial_points, trial_edges, min(search_end, clock() + epoch))
            temperature = best_cost / max(1, n) * 0.025 / (1.0 + failures / (1.0 + horizon))
            if trial_cost < current_cost or rng.random() < math.exp(min(0.0, (current_cost - trial_cost) / max(1e-15, temperature))):
                current, edges, current_cost = trial_points, trial_edges, trial_cost
            if best_cost < old_best - eps:
                failures = 0
            else:
                failures += 1
        # Leave the final time to process only the current champion, so unfinished perturbations cannot overwrite it.
        if clock() < end:
            q, f, value = refresh(best_points, best_edges, end)
            q, value = relax(q, f, end)
            record(q, f, value)
    except Exception:
        # If a numerical or optional acceleration path fails, return the previously saved complete feasible tree.
        pass
    return answer()
