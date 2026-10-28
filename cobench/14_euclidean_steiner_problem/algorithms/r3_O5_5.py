"""Full-component candidate pool and elite recombination search
Combines analytic Fermat points, four-terminal components, group insertion, budget pruning and geometric relaxation on the tree, saving the shortest feasible tree within the wall-clock limit.
"""

import heapq
import math
import random
import time

try:
    import numpy as np
except Exception:
    np = None
try:
    from scipy.spatial import Delaunay, cKDTree
except Exception:
    Delaunay = cKDTree = None


def solve(instance: dict, time_limit_s: float) -> dict:
    started = time.perf_counter()
    answer = {"steiner_points": [], "edges": []}
    try:
        raw = instance.get("points", [])
        n = len(raw)
        m = max(0, int(instance.get("max_steiner_points", 0)))
        origin = [float(raw[0][0]), float(raw[0][1])] if n else [0.0, 0.0]
        answer = {"steiner_points": [origin[:] for _ in range(m)],
                  "edges": [[i - 1, i] for i in range(1, n + m)]}
        seconds = max(0.0, float(time_limit_s))
        if n < 2 or seconds <= 0.0 or not math.isfinite(seconds):
            return answer
        points = [(float(p[0]), float(p[1])) for p in raw]
        if not all(math.isfinite(x) and math.isfinite(y) for x, y in points):
            return answer
        _search(points, m, seconds, started, answer)
    except Exception:
        # The published complete tree does not depend on any candidate operation still in progress.
        pass
    return answer


def _search(original, m, seconds, started, answer):
    clock = time.perf_counter
    n = len(original)
    cap = min(m, max(0, n - 2))
    hard_end = started + seconds * 0.985
    search_end = started + seconds * 0.925
    ox = min(p[0] for p in original)
    oy = min(p[1] for p in original)
    scale = max(max(p[0] for p in original) - ox,
                max(p[1] for p in original) - oy)
    if scale <= 0.0 or not math.isfinite(scale) or clock() >= hard_end:
        return
    terminals = [((x - ox) / scale, (y - oy) / scale) for x, y in original]
    root3 = math.sqrt(3.0)
    eps = 2e-12
    effort = math.log1p(seconds)
    rng = random.Random(1000003 * n + 9176 * cap)
    geometry_time = seconds / (90.0 + 6.0 * math.sqrt(n))
    slice_time = seconds / (5.0 + math.log1p(n) + effort)
    width = min(n - 1, max(2, int(math.ceil(2.0 + effort + math.log1p(n) / 4.0))))
    elite_size = max(2, int(math.ceil(1.0 + effort)))
    best_x = terminals[:]
    best_e = [(i - 1, i) for i in range(1, n)]

    def dist(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def length(x, e):
        return math.fsum(dist(x[a], x[b]) for a, b in e)

    best_value = length(best_x, best_e)

    def publish(x, e, equal=False):
        nonlocal best_x, best_e, best_value
        if len(x) - n > cap or len(e) != len(x) - 1:
            return float("inf")
        value = length(x, e)
        if math.isfinite(value) and (value < best_value - eps or (equal and value <= best_value)):
            out = [[float(ox + p[0] * scale), float(oy + p[1] * scale)] for p in x[n:]]
            used = len(out)
            out.extend([list(original[0]) for _ in range(m - used)])
            edges = [[int(min(a, b)), int(max(a, b))] for a, b in e]
            edges.extend([[0, n + j] for j in range(used, m)])
            best_x, best_e, best_value = x[:], e[:], value
            answer.update(steiner_points=out, edges=edges)
        return value

    def adjacency(x, e):
        adj = [set() for _ in x]
        for a, b in e:
            adj[a].add(b)
            adj[b].add(a)
        return adj

    def fermat(a, b, c):
        ux, uy = b[0] - a[0], b[1] - a[1]
        vx, vy = c[0] - a[0], c[1] - a[1]
        aa = (ux - vx) ** 2 + (uy - vy) ** 2
        bb = vx * vx + vy * vy
        cc = ux * ux + uy * uy
        if bb < 1e-30 or cc < 1e-30:
            return a
        if aa < 1e-30:
            return b
        da, db, dc = bb + cc - aa, aa + cc - bb, aa + bb - cc
        if da <= -math.sqrt(bb * cc):
            return a
        if db <= -math.sqrt(aa * cc):
            return b
        if dc <= -math.sqrt(aa * bb):
            return c
        area = 2.0 * abs(ux * vy - uy * vx)
        qa = max(1e-30, area + root3 * da)
        qb = max(1e-30, area + root3 * db)
        qc = max(1e-30, area + root3 * dc)
        z = min(qa, qb, qc)
        wa, wb, wc = z / qa, z / qb, z / qc
        total = wa + wb + wc
        return (a[0] + (wb * ux + wc * vx) / total,
                a[1] + (wb * uy + wc * vy) / total)

    def median(ps, at, stop):
        if len(ps) == 3:
            return fermat(*ps)
        if not ps:
            return at
        if len(ps) == 1:
            return ps[0]
        if len(ps) == 2:
            return ((ps[0][0] + ps[1][0]) * 0.5, (ps[0][1] + ps[1][1]) * 0.5)
        x, y = at
        while clock() < stop:
            sx = sy = sw = rx = ry = 0.0
            coincident = 0
            for px, py in ps:
                dx, dy = px - x, py - y
                d = math.hypot(dx, dy)
                if d < 1e-14:
                    coincident += 1
                else:
                    w = 1.0 / d
                    sx += w * px
                    sy += w * py
                    sw += w
                    rx += w * dx
                    ry += w * dy
            norm = math.hypot(rx, ry)
            if sw == 0.0 or norm <= coincident + 1e-13:
                break
            alpha = 1.0 - coincident / norm
            nx = x + alpha * (sx / sw - x)
            ny = y + alpha * (sy / sw - y)
            move = math.hypot(nx - x, ny - y)
            x, y = nx, ny
            if move < 2e-11:
                break
        return (x, y)

    def dense_mst(x, stop, noise=0.0):
        q = len(x)
        used = [False] * q
        parent = [-1] * q
        keys = [float("inf")] * q
        keys[0] = 0.0
        out = []
        if np is not None:
            xy = np.asarray(x, dtype=float)
            dk = np.full(q, np.inf)
            dk[0] = 0.0
            visited = np.zeros(q, dtype=bool)
            pa = np.full(q, -1, dtype=int)
            nrng = np.random.default_rng(rng.getrandbits(63)) if noise else None
            for _ in x:
                if clock() >= stop:
                    return None
                u = int(np.argmin(dk))
                if not math.isfinite(float(dk[u])):
                    return None
                visited[u] = True
                if pa[u] >= 0:
                    out.append((int(pa[u]), u))
                d = xy - xy[u]
                weights = d[:, 0] * d[:, 0] + d[:, 1] * d[:, 1]
                if noise:
                    weights *= (1.0 + noise * (2.0 * nrng.random(q) - 1.0)) ** 2
                mask = (~visited) & (weights < dk)
                dk[mask], pa[mask] = weights[mask], u
                dk[visited] = np.inf
            return out
        for _ in x:
            if clock() >= stop:
                return None
            u = min((i for i in range(q) if not used[i]), key=keys.__getitem__)
            used[u] = True
            if parent[u] >= 0:
                out.append((parent[u], u))
            for v in range(q):
                if not used[v]:
                    w = dist(x[u], x[v])
                    if noise:
                        w *= 1.0 + noise * (2.0 * rng.random() - 1.0)
                    if w < keys[v]:
                        keys[v], parent[v] = w, u
        return out

    def mst(x, stop, noise=0.0):
        q = len(x)
        if clock() >= stop:
            return None
        if Delaunay is None or np is None or q < 4:
            return dense_mst(x, stop, noise)
        try:
            xy = np.asarray(x, dtype=float)
            tri = Delaunay(xy).simplices
            pairs = set()
            for a, b, c in tri:
                a, b, c = int(a), int(b), int(c)
                pairs.update((tuple(sorted((a, b))), tuple(sorted((a, c))), tuple(sorted((b, c)))))
            order = sorted(range(q), key=lambda u: (x[u][0], x[u][1]))
            pairs.update(tuple(sorted((a, b))) for a, b in zip(order, order[1:]))
            # The sorted chain adds zero-length edges for duplicate coordinates omitted by the triangulation.
            weighted = [(dist(x[a], x[b]) * (1.0 + noise * (2.0 * rng.random() - 1.0))
                         if noise else dist(x[a], x[b]), a, b) for a, b in pairs]
            weighted.sort()
            parent = list(range(q))
            size = [1] * q

            def find(u):
                while parent[u] != u:
                    parent[u] = parent[parent[u]]
                    u = parent[u]
                return u

            out = []
            for _, a, b in weighted:
                if clock() >= stop:
                    return None
                u, v = find(a), find(b)
                if u != v:
                    if size[u] < size[v]:
                        u, v = v, u
                    parent[v] = u
                    size[u] += size[v]
                    out.append((a, b))
                    if len(out) == q - 1:
                        return out
        except Exception:
            pass
        return dense_mst(x, stop, noise)

    def compact(x, e):
        g = adjacency(x, e)
        alive = [True] * len(x)
        todo = list(range(n, len(x)))
        while todo:
            s = todo.pop()
            if not alive[s]:
                continue
            ns = list(g[s])
            target = next((v for v in ns if dist(x[s], x[v]) < 1e-13), None)
            if len(ns) > 2 and target is None:
                continue
            for v in ns:
                g[v].discard(s)
            if target is not None:
                for v in ns:
                    if v != target:
                        g[target].add(v)
                        g[v].add(target)
            elif len(ns) == 2:
                a, b = ns
                g[a].add(b)
                g[b].add(a)
            g[s].clear()
            alive[s] = False
            todo.extend(v for v in ns if v >= n)
        ids = [u for u in range(len(x)) if alive[u]]
        remap = {u: i for i, u in enumerate(ids)}
        return [x[u] for u in ids], [(remap[u], remap[v]) for u in ids for v in g[u] if u < v]

    def relax(x, e, stop, joint=False):
        if len(x) == n or clock() >= stop:
            return x[:]
        x = x[:]
        g = adjacency(x, e)
        old = length(x, e)
        order = []
        parent = [-2] * len(x)
        if joint:
            for root in range(n, len(x)):
                if parent[root] != -2:
                    continue
                parent[root] = -1
                stack = [root]
                while stack:
                    u = stack.pop()
                    order.append(u)
                    for v in g[u]:
                        if v >= n and parent[v] == -2:
                            parent[v] = u
                            stack.append(v)
        reverse = False
        smoothing = max(1e-12, old / len(e) * 1e-4)
        while clock() < stop:
            before = old
            for u in (range(len(x) - 1, n - 1, -1) if reverse else range(n, len(x))):
                if clock() >= stop:
                    return x
                ps = [x[v] for v in g[u]]
                q = median(ps, x[u], stop)
                if sum(dist(q, a) for a in ps) <= sum(dist(x[u], a) for a in ps):
                    x[u] = q
            reverse = not reverse
            old = length(x, e)
            if joint and clock() < stop:
                # The Steiner-induced graph is a forest, so the weighted Laplacian can be eliminated in linear time.
                diagonal = [0.0] * len(x)
                bx, by = [0.0] * len(x), [0.0] * len(x)
                link = [0.0] * len(x)
                for a, b in e:
                    w = 1.0 / math.hypot(dist(x[a], x[b]), smoothing)
                    if a >= n:
                        diagonal[a] += w
                        if b < n:
                            bx[a] += w * x[b][0]
                            by[a] += w * x[b][1]
                    if b >= n:
                        diagonal[b] += w
                        if a < n:
                            bx[b] += w * x[a][0]
                            by[b] += w * x[a][1]
                    if a >= n and b >= n:
                        link[a if parent[a] == b else b] = w
                for u in reversed(order):
                    p = parent[u]
                    if p >= n:
                        r = link[u] / max(1e-25, diagonal[u])
                        diagonal[p] -= link[u] * r
                        bx[p] += bx[u] * r
                        by[p] += by[u] * r
                trial = x[:]
                for u in order:
                    p = parent[u]
                    d = max(1e-25, diagonal[u])
                    if p >= n:
                        trial[u] = ((bx[u] + link[u] * trial[p][0]) / d,
                                    (by[u] + link[u] * trial[p][1]) / d)
                    else:
                        trial[u] = (bx[u] / d, by[u] / d)
                value = length(trial, e)
                if math.isfinite(value) and value < old:
                    x, old = trial, value
                smoothing = max(1e-13, smoothing * 0.25)
            if before - old <= 2e-12 * max(1.0, before):
                break
        return x

    def grow(x, e, stop, chaos=0.0):
        x, e = x[:], e[:]
        while len(x) - n < cap and clock() < stop:
            g = adjacency(x, e)
            best_score, choice = eps, None
            for u, neighbors in enumerate(g):
                if len(neighbors) < 2 or (u >= n and len(neighbors) <= 3):
                    continue
                ns = list(neighbors)
                for i, a in enumerate(ns):
                    if clock() >= stop:
                        break
                    for b in ns[:i]:
                        if clock() >= stop:
                            break
                        f = fermat(x[u], x[a], x[b])
                        gain = dist(x[u], x[a]) + dist(x[u], x[b])
                        gain -= dist(f, x[u]) + dist(f, x[a]) + dist(f, x[b])
                        score = gain * (1.0 + chaos * rng.random())
                        if gain > eps and score > best_score:
                            best_score, choice = score, (u, a, b, f)
                if clock() >= stop:
                    break
            if choice is None:
                break
            u, a, b, f = choice
            s = len(x)
            e = [(v, w) for v, w in e if not ((v == u and w in (a, b)) or (w == u and v in (a, b)))]
            e.extend(((s, u), (s, a), (s, b)))
            x.append(f)
            x = relax(x, e, min(stop, clock() + geometry_time / math.sqrt(1 + len(x) - n)))
            publish(x, e)
        x = relax(x, e, min(stop, clock() + geometry_time), True)
        x, e = compact(x, e)
        publish(x, e)
        return x, e

    def quartet(a, b, c, d):
        # Enumerate the orientations of the equilateral auxiliary points of both pairs, and back-project onto their respective circumcircles.
        def apices(p, q):
            mx, my = (p[0] + q[0]) * 0.5, (p[1] + q[1]) * 0.5
            dx, dy = (q[0] - p[0]) * root3 * 0.5, (q[1] - p[1]) * root3 * 0.5
            return ((mx - dy, my + dx), (mx + dy, my - dx))

        result, best = None, float("inf")
        for aa in apices(a, b):
            ca = ((a[0] + b[0] + aa[0]) / 3.0, (a[1] + b[1] + aa[1]) / 3.0)
            for bb in apices(c, d):
                dx, dy = bb[0] - aa[0], bb[1] - aa[1]
                dd = dx * dx + dy * dy
                if dd < 1e-28:
                    continue
                cb = ((c[0] + d[0] + bb[0]) / 3.0, (c[1] + d[1] + bb[1]) / 3.0)
                t = 2.0 * ((ca[0] - aa[0]) * dx + (ca[1] - aa[1]) * dy) / dd
                v = -2.0 * ((cb[0] - bb[0]) * dx + (cb[1] - bb[1]) * dy) / dd
                if t <= 0.0 or v <= 0.0 or t + v >= 1.0:
                    continue
                p = (aa[0] + t * dx, aa[1] + t * dy)
                q = (bb[0] - v * dx, bb[1] - v * dy)
                # Measure length directly by the five real edges, never using auxiliary line lengths as the objective value.
                value = dist(p, a) + dist(p, b) + dist(p, q) + dist(q, c) + dist(q, d)
                if value < best:
                    best, result = value, (p, q)
        return result

    def improve_nni(x, e, stop):
        x, e = x[:], e[:]
        while clock() < stop:
            g = adjacency(x, e)
            candidates = [(u, v) for u, v in e if u >= n or v >= n]
            rng.shuffle(candidates)
            changed = False
            for s, t in candidates:
                if clock() >= stop:
                    break
                if s < n:
                    s, t = t, s
                if len(g[s]) != 3:
                    continue
                left = list(g[s] - {t})
                right = list(g[t] - {s})
                if not right:
                    continue
                swaps = [(left[0], left[1], b) for b in right]
                if t < n:
                    swaps.extend((left[1], left[0], b) for b in right)
                for a, c, b in swaps:
                    if clock() >= stop:
                        break
                    old = dist(x[s], x[a]) + dist(x[s], x[c]) + dist(x[s], x[t]) + dist(x[t], x[b])
                    px, qx = x[s], x[t]
                    if t >= n and len(right) == 2:
                        d = next(v for v in right if v != b)
                        old += dist(x[t], x[d])
                        pair = quartet(x[c], x[b], x[a], x[d])
                        if pair is not None:
                            px, qx = pair
                        else:
                            px = fermat(x[c], x[b], qx)
                            qx = fermat(x[a], x[d], px)
                        local_stop = min(stop, clock() + geometry_time / (2.0 + math.sqrt(len(x))))
                        previous = float("inf")
                        while clock() < local_stop:
                            px = fermat(x[c], x[b], qx)
                            qx = fermat(x[a], x[d], px)
                            value = dist(px, x[c]) + dist(px, x[b]) + dist(px, qx) + dist(qx, x[a]) + dist(qx, x[d])
                            if previous - value < eps:
                                break
                            previous = value
                        value = dist(px, x[c]) + dist(px, x[b]) + dist(px, qx) + dist(qx, x[a]) + dist(qx, x[d])
                    elif t < n:
                        px = fermat(x[c], x[b], x[t])
                        value = dist(px, x[c]) + dist(px, x[b]) + dist(px, x[t]) + dist(x[t], x[a])
                    else:
                        continue
                    if value >= old - eps:
                        continue
                    removed = {tuple(sorted((s, a))), tuple(sorted((t, b)))}
                    te = [(u, v) for u, v in e if tuple(sorted((u, v))) not in removed]
                    te.extend(((s, b), (t, a)))
                    tx = x[:]
                    tx[s] = px
                    if t >= n:
                        tx[t] = qx
                    tx = relax(tx, te, min(stop, clock() + geometry_time), True)
                    x, e = compact(tx, te)
                    publish(x, e)
                    changed = True
                    break
                if changed:
                    break
            if not changed:
                break
        return x, e

    def trim(x, e, stop):
        # Excess points compete by deletion delta; local reconnection keeps the whole tree connected and acyclic.
        x, e = compact(x, e)
        if len(x) - n <= cap:
            return x, e
        g = adjacency(x, e)
        alive = [True] * len(x)
        versions = [0] * len(x)
        heap = []

        def offer(u):
            versions[u] += 1
            if not alive[u] or u < n:
                return
            ns = list(g[u])
            edges = []
            todo = set(ns[1:])
            reached = ns[:1]
            while todo:
                if clock() >= stop:
                    return
                a, b = min(((a, b) for a in reached for b in todo), key=lambda z: dist(x[z[0]], x[z[1]]))
                edges.append((a, b))
                reached.append(b)
                todo.remove(b)
            loss = sum(dist(x[a], x[b]) for a, b in edges) - sum(dist(x[u], x[a]) for a in ns)
            heapq.heappush(heap, (loss, u, versions[u], edges))

        for u in range(n, len(x)):
            if clock() >= stop:
                return None
            offer(u)
        active = len(x) - n
        while active > cap and heap and clock() < stop:
            _, u, version, edges = heapq.heappop(heap)
            if not alive[u] or versions[u] != version:
                continue
            ns = list(g[u])
            for v in ns:
                g[v].remove(u)
            for a, b in edges:
                g[a].add(b)
                g[b].add(a)
            g[u].clear()
            alive[u] = False
            active -= 1
            for v in ns:
                if v >= n:
                    offer(v)
        if active > cap:
            return None
        ids = [u for u in range(len(x)) if alive[u]]
        remap = {u: i for i, u in enumerate(ids)}
        return compact([x[u] for u in ids], [(remap[u], remap[v]) for u in ids for v in g[u] if u < v])

    def refresh(x, e, stop):
        te = mst(x, stop)
        if te is not None and length(x, te) <= length(x, e) + eps:
            x, e = compact(x, te)
            x = relax(x, e, min(stop, clock() + geometry_time), True)
            publish(x, e)
        return x, e

    before = clock()
    base_e = mst(terminals, search_end)
    mst_time = clock() - before
    if base_e is None:
        return
    publish(terminals, base_e, True)
    if cap == 0:
        return
    search_end = min(search_end, hard_end - min(seconds * 0.22, mst_time * 2.0))
    x, e = grow(terminals, base_e, min(search_end, clock() + seconds * 0.18))
    x, e = refresh(x, e, min(search_end, clock() + slice_time * 0.15))
    x, e = grow(x, e, min(search_end, clock() + slice_time * 0.15))
    x, e = improve_nni(x, e, min(search_end, clock() + slice_time * 0.35))

    # Candidate components depend only on terminals; each entry stores the gain, centre and the whole group of Steiner coordinates.
    catalog = []
    catalog_stop = min(search_end, clock() + seconds / (7.0 + math.log1p(n)))
    triples, quads = set(), set()
    try:
        if Delaunay is not None and np is not None and n >= 4 and clock() < catalog_stop:
            triangles = Delaunay(np.asarray(terminals)).simplices
            faces = {}
            for tri in triangles:
                if clock() >= catalog_stop:
                    break
                a, b, c = sorted(map(int, tri))
                triples.add((a, b, c))
                for u, v, w in ((a, b, c), (a, c, b), (b, c, a)):
                    pair = (u, v)
                    if pair in faces:
                        quads.add(tuple(sorted((u, v, w, faces[pair]))))
                    else:
                        faces[pair] = w
        near_rows = None
        if cKDTree is not None and np is not None and clock() < catalog_stop:
            _, near_rows = cKDTree(np.asarray(terminals)).query(terminals, k=min(n, width + 1))
        for u in range(n):
            if clock() >= catalog_stop:
                break
            ns = ([int(v) for v in near_rows[u] if int(v) != u] if near_rows is not None else
                  heapq.nsmallest(width, (v for v in range(n) if v != u), key=lambda v: dist(terminals[u], terminals[v])))
            for i, a in enumerate(ns):
                for b in ns[:i]:
                    triples.add(tuple(sorted((u, a, b))))
            if len(ns) >= 3:
                quads.add(tuple(sorted((u, ns[0], ns[1], ns[2]))))
        for a, b, c in sorted(triples):
            if clock() >= catalog_stop:
                break
            pa, pb, pc = terminals[a], terminals[b], terminals[c]
            f = fermat(pa, pb, pc)
            if min(dist(f, pa), dist(f, pb), dist(f, pc)) < 1e-10:
                continue
            ds = (dist(pa, pb), dist(pa, pc), dist(pb, pc))
            gain = sum(ds) - max(ds) - dist(f, pa) - dist(f, pb) - dist(f, pc)
            if gain > eps:
                catalog.append((gain, f, (f,)))
        for ids in sorted(quads):
            if clock() >= catalog_stop:
                break
            ps = [terminals[u] for u in ids]
            local_e = dense_mst(ps, catalog_stop)
            if local_e is None:
                break
            old = length(ps, local_e)
            winner = None
            for a, b, c, d in ((0, 1, 2, 3), (0, 2, 1, 3), (0, 3, 1, 2)):
                pair = quartet(ps[a], ps[b], ps[c], ps[d])
                if pair is None:
                    continue
                p, q = pair
                value = dist(p, ps[a]) + dist(p, ps[b]) + dist(p, q) + dist(q, ps[c]) + dist(q, ps[d])
                if value < old - eps:
                    old, winner = value, pair
            if winner is not None:
                p, q = winner
                gain = length(ps, local_e) - old
                catalog.append((gain / 2.0, ((p[0] + q[0]) * 0.5, (p[1] + q[1]) * 0.5), winner))
    except Exception:
        pass
    catalog.sort(key=lambda item: item[0], reverse=True)
    pool_limit = max(1, int(n * width * (1.0 + effort)))
    catalog = catalog[:pool_limit]
    elites = [(best_value, best_x[:], best_e[:])]
    stalled = 0

    def pooled(source, stop):
        q = len(source) - n
        center = source[rng.randrange(len(source))]
        amplitude = (1.0 + effort) * math.sqrt(max(1, q)) * (0.25 + 0.12 * math.log1p(stalled))
        amount = max(1, int(1.0 + rng.random() * amplitude))
        extras = []
        if catalog:
            if rng.random() < 0.35:
                entry = catalog[int(len(catalog) * rng.random() ** (1.0 + effort))]
                center = entry[1]
            # Mainly local group activation, while keeping the chance of cross-region combinations.
            if rng.random() < 0.8:
                ranked = heapq.nsmallest(amount, catalog,
                    key=lambda z: (dist(z[1], center) + 0.03 / math.sqrt(n)) /
                    (max(z[0], eps) ** 0.18 * (0.65 + rng.random() * 0.7)))
            else:
                ranked = rng.sample(catalog, min(amount, len(catalog)))
            for item in ranked:
                extras.extend(item[2])
        if len(elites) > 1 and rng.random() < 0.6:
            donor = rng.choice(elites)[1][n:]
            extras.extend(heapq.nsmallest(amount, donor, key=lambda p: dist(p, center)))
        if not extras:
            return None
        # Coincident candidates need not be activated again; terminals and retained auxiliary points keep their original indices.
        removed = set()
        if q and rng.random() < 0.55:
            count = min(q, max(1, int(amount * (0.5 + rng.random()))))
            removed = set(heapq.nsmallest(count, range(n, len(source)),
                          key=lambda u: dist(source[u], center)))
        trial = [p for u, p in enumerate(source) if u not in removed]
        seen = {(round(p[0], 11), round(p[1], 11)) for p in trial}
        for p in extras:
            key = (round(p[0], 11), round(p[1], 11))
            if key not in seen:
                trial.append(p)
                seen.add(key)
        te = mst(trial, stop)
        if te is None:
            return None
        result = trim(trial, te, stop)
        if result is None:
            return None
        tx, te = result
        tx = relax(tx, te, min(stop, clock() + geometry_time), True)
        publish(tx, te)
        return tx, te

    while clock() < search_end:
        before = best_value
        stop = min(search_end, clock() + slice_time)
        progress = min(1.0, (clock() - started) / seconds)
        mode = rng.random()
        chaos = rng.random() * effort * (0.04 + 0.25 * progress)
        source = best_x if rng.random() < 0.7 else rng.choice(elites)[1]
        tx, te = best_x[:], best_e[:]
        if mode < 0.22:
            noise = min(0.42, (0.005 + 0.022 * effort) * (1.0 + math.sqrt(stalled))) * rng.random()
            fresh = mst(terminals, stop, noise)
            if fresh is not None:
                tx, te = terminals[:], fresh
        elif mode < 0.64:
            proposal = pooled(source, min(stop, clock() + slice_time * 0.48))
            if proposal is not None:
                tx, te = proposal
        elif len(source) > n:
            k = len(source) - n
            amount = min(k, max(1, int(1.0 + math.sqrt(k) * (0.15 + 0.16 * effort)
                                         * (1.0 + math.log1p(stalled)) * rng.random())))
            pivot = source[rng.randrange(n, len(source))]
            if rng.random() < 0.8:
                removed = set(heapq.nsmallest(amount, range(n, len(source)),
                              key=lambda u: dist(source[u], pivot) * (0.65 + 0.7 * rng.random())))
            else:
                removed = set(rng.sample(range(n, len(source)), amount))
            trial = [p for i, p in enumerate(source) if i not in removed]
            fresh = mst(trial, stop)
            if fresh is not None:
                tx, te = compact(trial, fresh)
        if clock() >= search_end:
            break
        tx, te = grow(tx, te, min(stop, clock() + slice_time * 0.38), chaos)
        tx, te = improve_nni(tx, te, min(stop, clock() + slice_time * 0.28))
        tx, te = refresh(tx, te, min(stop, clock() + slice_time * 0.14))
        tx, te = grow(tx, te, stop, chaos * 0.35)
        value = publish(tx, te)
        if value <= best_value * (1.0 + 0.025 / (1.0 + effort)):
            if all(abs(value - item[0]) > 1e-9 for item in elites):
                elites.append((value, tx[:], te[:]))
                elites.sort(key=lambda item: item[0])
                del elites[elite_size:]
        stalled = 0 if best_value < before - eps else stalled + 1
        # The reserve for the final MST adapts to the actual construction time and current number of points.
        reserve = min(seconds * 0.22, mst_time * (len(best_x) / n) ** 1.5 * 2.0)
        search_end = min(search_end, hard_end - reserve)

    finish_mst = max(seconds * 0.018, mst_time * (len(best_x) / n) ** 1.5 * 1.5)
    tx = relax(best_x, best_e, hard_end - finish_mst, True)
    publish(tx, best_e)
    final_e = mst(best_x, hard_end)
    if final_e is not None:
        publish(best_x, final_e, True)
        tx, te = compact(best_x, final_e)
        publish(tx, te, True)
