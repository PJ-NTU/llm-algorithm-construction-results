"""Point insertion fused with complete subtree reconstruction search
On the same tree, alternates Fermat-point insertion, adjacency swaps, equilateral-triangle topology reconstruction and weighted coordinate optimisation on the tree, searching continually with regional destroy-and-repair and perturbation restarts.
"""
import math
import random
import time
import itertools
import heapq


def solve(instance: dict, time_limit_s: float) -> dict:
    start = time.perf_counter()
    answer = {"steiner_points": [], "edges": []}
    try:
        terminals = [(float(p[0]), float(p[1])) for p in instance.get("points", [])]
        n = len(terminals)
        answer["edges"] = [[i - 1, i] for i in range(1, n)]
        budget = max(0.0, float(time_limit_s))
        limit = max(0, min(int(instance.get("max_steiner_points", 0)), n - 2))
        if n < 2 or budget <= 0.0:
            return answer
        clock = time.perf_counter
        deadline = start + budget * 0.94
        finish = start + budget * 0.985
        if clock() >= deadline:
            return answer
        xmin = min(p[0] for p in terminals)
        ymin = min(p[1] for p in terminals)
        scale = max(max(p[0] for p in terminals) - xmin,
                    max(p[1] for p in terminals) - ymin)
        if not math.isfinite(scale) or scale <= 0.0:
            return answer
        points = [((x - xmin) / scale, (y - ymin) / scale) for x, y in terminals]
        rng = random.Random(104729 * n + 1009 * limit)
        root3 = math.sqrt(3.0)
        np = None
        try:
            import numpy as np
        except Exception:
            pass

        def dist(a, b):
            return math.hypot(a[0] - b[0], a[1] - b[1])

        def mst(ps, stop, noise=0.0):
            """Return an empty marker on timeout, never replacing the existing solution with an incomplete tree."""
            size = len(ps)
            if size < 2:
                return []
            result = []
            if np is not None and size > 24:
                xy = np.asarray(ps, dtype=float)
                keys = np.full(size, np.inf)
                parents = np.full(size, -1, dtype=int)
                active = np.ones(size, dtype=bool)
                keys[0] = 0.0
                nrng = np.random.default_rng(rng.getrandbits(63)) if noise else None
                for _ in range(size):
                    if clock() >= stop:
                        return None
                    u = int(np.argmin(keys))
                    active[u] = False
                    if parents[u] >= 0:
                        result.append((int(parents[u]), u))
                    delta = xy - xy[u]
                    ds = delta[:, 0] * delta[:, 0] + delta[:, 1] * delta[:, 1]
                    if noise:
                        ds *= (1.0 + noise * (2.0 * nrng.random(size) - 1.0)) ** 2
                    mask = active & (ds < keys)
                    keys[mask] = ds[mask]
                    parents[mask] = u
                    keys[~active] = np.inf
                return result
            used = [False] * size
            keys = [float("inf")] * size
            parents = [-1] * size
            keys[0] = 0.0
            for _ in range(size):
                if clock() >= stop:
                    return None
                u = min((i for i in range(size) if not used[i]), key=keys.__getitem__)
                used[u] = True
                if parents[u] >= 0:
                    result.append((parents[u], u))
                for v in range(size):
                    if not used[v]:
                        d = dist(ps[u], ps[v])
                        if noise:
                            d *= 1.0 + noise * (2.0 * rng.random() - 1.0)
                        if d < keys[v]:
                            keys[v], parents[v] = d, u
            return result

        edges = mst(points, deadline)
        if edges is None:
            return answer
        answer = {"steiner_points": [], "edges": [[a, b] for a, b in edges]}
        if limit == 0:
            return answer
        coords = dict(enumerate(points))
        adj = {i: set() for i in coords}
        for a, b in edges:
            adj[a].add(b)
            adj[b].add(a)
        value = sum(dist(points[a], points[b]) for a, b in edges)
        best_value = value
        best_coords = coords.copy()
        best_adj = {u: vs.copy() for u, vs in adj.items()}
        next_id = -1

        def all_edges():
            return [(u, v) for u, vs in adj.items() for v in vs if u < v]

        def export(cs, ad):
            ids = list(range(n)) + sorted(u for u in cs if u < 0)
            remap = {u: i for i, u in enumerate(ids)}
            return {"steiner_points": [[float(cs[u][0] * scale + xmin),
                                        float(cs[u][1] * scale + ymin)] for u in ids[n:]],
                    "edges": [[remap[u], remap[v]] for u in ids for v in ad[u] if u < v]}

        def record():
            nonlocal best_value, best_coords, best_adj, answer
            if len(coords) - n <= limit and value < best_value - 1e-12 * max(1.0, best_value):
                value_check = sum(dist(coords[u], coords[v])
                                  for u, vs in adj.items() for v in vs if u < v)
                if not math.isfinite(value_check) or value_check >= best_value - 1e-12:
                    return
                best_value = value_check
                best_coords = coords.copy()
                best_adj = {u: vs.copy() for u, vs in adj.items()}
                answer = export(best_coords, best_adj)

        def fermat(a, b, c):
            ab = (b[0] - a[0], b[1] - a[1])
            ac = (c[0] - a[0], c[1] - a[1])
            bc = (c[0] - b[0], c[1] - b[1])
            aa = bc[0] * bc[0] + bc[1] * bc[1]
            bb = ac[0] * ac[0] + ac[1] * ac[1]
            cc = ab[0] * ab[0] + ab[1] * ab[1]
            if min(aa, bb, cc) < 1e-28:
                if cc < 1e-28 or bb < 1e-28:
                    return a
                return b
            dots = ((bb + cc - aa) * 0.5, (aa + cc - bb) * 0.5,
                    (aa + bb - cc) * 0.5)
            if dots[0] <= -0.5 * math.sqrt(bb * cc):
                return a
            if dots[1] <= -0.5 * math.sqrt(aa * cc):
                return b
            if dots[2] <= -0.5 * math.sqrt(aa * bb):
                return c
            area = abs(ab[0] * ac[1] - ab[1] * ac[0])
            ws = [1.0 / max(1e-30, area + root3 * d) for d in dots]
            total = sum(ws)
            return ((ws[0] * a[0] + ws[1] * b[0] + ws[2] * c[0]) / total,
                    (ws[0] * a[1] + ws[1] * b[1] + ws[2] * c[1]) / total)

        def median(ps, initial, stop):
            if len(ps) == 3:
                return fermat(*ps)
            x, y = initial
            while clock() < stop:
                sx = sy = sw = rx = ry = 0.0
                same = 0
                for px, py in ps:
                    d = math.hypot(px - x, py - y)
                    if d < 1e-13:
                        same += 1
                    else:
                        w = 1.0 / d
                        sx += w * px
                        sy += w * py
                        sw += w
                        rx += (px - x) * w
                        ry += (py - y) * w
                norm = math.hypot(rx, ry)
                if sw == 0.0 or norm <= same + 1e-12:
                    return (x, y)
                fraction = max(0.0, 1.0 - same / norm)
                xx = x + fraction * (sx / sw - x)
                yy = y + fraction * (sy / sw - y)
                if math.hypot(xx - x, yy - y) < 1e-11:
                    return (xx, yy)
                x, y = xx, yy
            return (x, y)

        def prune(seeds):
            nonlocal value
            queue = list(seeds)
            while queue:
                u = queue.pop()
                if u >= 0 or u not in adj:
                    continue
                vs = list(adj[u])
                merge = None
                for v in vs:
                    if dist(coords[u], coords[v]) < 1e-12:
                        merge = v
                        break
                if len(vs) > 2 and merge is None:
                    continue
                value -= sum(dist(coords[u], coords[v]) for v in vs)
                for v in vs:
                    adj[v].discard(u)
                if merge is not None:
                    for v in vs:
                        if v != merge:
                            adj[merge].add(v)
                            adj[v].add(merge)
                            value += dist(coords[merge], coords[v])
                    queue.append(merge)
                elif len(vs) == 2:
                    a, b = vs
                    adj[a].add(b)
                    adj[b].add(a)
                    value += dist(coords[a], coords[b])
                del adj[u]
                del coords[u]
                queue.extend(vs)

        def optimize(stop):
            """On a fixed topology, alternate exact medians with weighted Laplacian elimination on the forest."""
            nonlocal value
            ss = [u for u in coords if u < 0]
            if not ss or clock() >= stop:
                return
            es = all_edges()
            parent = {}
            order = []
            for root in ss:
                if root in parent:
                    continue
                parent[root] = None
                stack = [root]
                while stack:
                    u = stack.pop()
                    order.append(u)
                    for v in adj[u]:
                        if v < 0 and v not in parent:
                            parent[v] = u
                            stack.append(v)
            smoothing = max(1e-12, value / max(1, len(es)) * 1e-4)
            while clock() < stop:
                previous = value
                for u in ss:
                    if clock() >= stop:
                        break
                    ps = [coords[v] for v in adj[u]]
                    new = median(ps, coords[u], stop)
                    old_cost = sum(dist(coords[u], p) for p in ps)
                    new_cost = sum(dist(new, p) for p in ps)
                    if new_cost <= old_cost:
                        coords[u] = new
                value = sum(dist(coords[a], coords[b]) for a, b in es)
                if clock() >= stop:
                    break
                diagonal = {u: 0.0 for u in ss}
                bx = {u: 0.0 for u in ss}
                by = {u: 0.0 for u in ss}
                link = {}
                for a, b in es:
                    w = 1.0 / max(smoothing, dist(coords[a], coords[b]))
                    if a < 0:
                        diagonal[a] += w
                        if b >= 0:
                            bx[a] += w * coords[b][0]
                            by[a] += w * coords[b][1]
                    if b < 0:
                        diagonal[b] += w
                        if a >= 0:
                            bx[b] += w * coords[a][0]
                            by[b] += w * coords[a][1]
                    if a < 0 and b < 0:
                        link[a if parent[a] == b else b] = w
                for u in reversed(order):
                    p = parent[u]
                    if p is not None:
                        ratio = link[u] / max(1e-20, diagonal[u])
                        diagonal[p] -= link[u] * ratio
                        bx[p] += bx[u] * ratio
                        by[p] += by[u] * ratio
                trial = {}
                for u in order:
                    p = parent[u]
                    d = max(1e-20, diagonal[u])
                    if p is None:
                        trial[u] = (bx[u] / d, by[u] / d)
                    else:
                        trial[u] = ((bx[u] + link[u] * trial[p][0]) / d,
                                    (by[u] + link[u] * trial[p][1]) / d)
                trial_value = sum(dist(trial.get(a, coords[a]), trial.get(b, coords[b]))
                                  for a, b in es)
                if trial_value < value:
                    coords.update(trial)
                    value = trial_value
                smoothing = max(1e-13, smoothing * 0.35)
                if previous - value <= 1e-11 * max(1.0, value):
                    break
            prune(ss)
            record()

        def tree_topologies(k):
            if k == 3:
                yield ((-1, 0), (-1, 1), (-1, 2))
            else:
                for old in tree_topologies(k - 1):
                    s = -(k - 2)
                    for j, (a, b) in enumerate(old):
                        yield old[:j] + old[j + 1:] + ((a, s), (b, s), (k - 1, s))

        def compile_topology(es):
            ad = {}
            for a, b in es:
                ad.setdefault(a, set()).add(b)
                ad.setdefault(b, set()).add(a)
            reductions = []
            branches = {u for u in ad if u < 0}
            while len(branches) > 1:
                for s in sorted(branches):
                    leaves = sorted(v for v in ad[s] if len(ad[v]) == 1)
                    if len(leaves) >= 2:
                        a, b = leaves[:2]
                        q = next(v for v in ad[s] if v != a and v != b)
                        reductions.append((a, b, s, q))
                        del ad[a]
                        del ad[b]
                        ad[s] = {q}
                        branches.remove(s)
                        break
            root = next(iter(branches))
            return (es, reductions, root, tuple(ad[root]))

        cache = {}
        generators = {}
        completed = set()

        def plans(k, stop):
            if k not in cache:
                cache[k] = []
                generators[k] = tree_topologies(k)
            while k not in completed and clock() < stop:
                try:
                    es = next(generators[k])
                except StopIteration:
                    completed.add(k)
                    break
                cache[k].append(compile_topology(es))
            return cache[k]

        def component(ps, slots, old_length, stop):
            """Keep boundary coordinates fixed and return the best replacement subtree with actually measured length."""
            k = len(ps)
            incumbent = old_length
            result = None

            def consider(cs, es):
                nonlocal incumbent, result
                length = sum(dist(cs[a], cs[b]) for a, b in es)
                if length < incumbent - 1e-12:
                    incumbent = length
                    result = (cs[:], list(es), length)

            base = mst(ps, stop)
            if base is None:
                return result
            consider(ps, base)
            if slots <= 0 or clock() >= stop:
                return result
            if k == 3:
                q = fermat(*ps)
                consider(ps + [q], [(0, 3), (1, 3), (2, 3)])
                return result
            # Also use insertion repair inside full topology enumeration, covering cases with partial slots and boundary branching.
            local = ps[:]
            local_edges = list(base)
            build_stop = min(stop, clock() + max(0.0, stop - clock()) * 0.14)
            while len(local) - k < slots and clock() < build_stop:
                local_adj = [[] for _ in local]
                for a, b in local_edges:
                    local_adj[a].append(b)
                    local_adj[b].append(a)
                choice = None
                gain_best = 1e-12
                for u, neighbors in enumerate(local_adj):
                    if u >= k and len(neighbors) <= 3:
                        continue
                    for a, b in itertools.combinations(neighbors, 2):
                        if clock() >= build_stop:
                            break
                        q = fermat(local[u], local[a], local[b])
                        gain = (dist(local[u], local[a]) + dist(local[u], local[b])
                                - dist(q, local[u]) - dist(q, local[a]) - dist(q, local[b]))
                        if gain > gain_best:
                            gain_best, choice = gain, (u, a, b, q)
                if choice is None:
                    break
                u, a, b, q = choice
                sid = len(local)
                local.append(q)
                removed_edges = {tuple(sorted((u, a))), tuple(sorted((u, b)))}
                local_edges = [(x, y) for x, y in local_edges
                               if tuple(sorted((x, y))) not in removed_edges]
                local_edges.extend(((sid, u), (sid, a), (sid, b)))
                local_adj.append([u, a, b])
                local_adj[u].remove(a)
                local_adj[u].remove(b)
                local_adj[u].append(sid)
                for v in (a, b):
                    local_adj[v].remove(u)
                    local_adj[v].append(sid)
                polish_stop = min(build_stop, clock() + max(0.0, build_stop - clock()) * 0.25)
                while clock() < polish_stop:
                    movement = 0.0
                    for v in range(k, len(local)):
                        q = median([local[w] for w in local_adj[v]], local[v], polish_stop)
                        movement = max(movement, dist(q, local[v]))
                        local[v] = q
                    if movement < 1e-10:
                        break
                consider(local, local_edges)
            # Degenerate components allow terminals to act as branch points, avoiding forcing all boundary points to be leaves.
            if slots < k - 2 or k <= 4:
                for triple in itertools.combinations(range(k), 3):
                    if clock() >= stop:
                        return result
                    q = fermat(*(ps[i] for i in triple))
                    cs = ps + [q]
                    es = mst(cs, stop)
                    if es is not None:
                        consider(cs, es)
                if slots < k - 2 and clock() < stop:
                    center = (sum(p[0] for p in ps) / k, sum(p[1] for p in ps) / k)
                    q = median(ps, center, stop)
                    consider(ps + [q], [(i, k) for i in range(k)])
            if slots < k - 2 or clock() >= stop:
                return result
            cached = plans(k, min(stop, clock() + max(0.0, stop - clock()) * 0.12))
            order = list(range(len(cached)))
            rng.shuffle(order)
            for index in order:
                if clock() >= stop:
                    break
                es, reductions, root, leaves = cached[index]
                positions = dict(enumerate(ps))
                circles = []

                def embed(level):
                    if clock() >= stop:
                        return
                    if level < len(reductions):
                        a, b, s, q = reductions[level]
                        pa, pb = positions[a], positions[b]
                        dx, dy = pb[0] - pa[0], pb[1] - pa[1]
                        if dx * dx + dy * dy < 1e-26:
                            return
                        mx, my = (pa[0] + pb[0]) * 0.5, (pa[1] + pb[1]) * 0.5
                        for sign in (-1.0, 1.0):
                            ex = mx - sign * root3 * dy * 0.5
                            ey = my + sign * root3 * dx * 0.5
                            positions[s] = (ex, ey)
                            circles.append((s, q, ex, ey, (pa[0] + pb[0] + ex) / 3.0,
                                            (pa[1] + pb[1] + ey) / 3.0, pa, pb))
                            embed(level + 1)
                            circles.pop()
                        return
                    center = fermat(*(positions[u] for u in leaves))
                    bound = sum(dist(center, positions[u]) for u in leaves)
                    if bound >= incumbent - 1e-12:
                        return
                    restored = positions.copy()
                    restored[root] = center
                    for s, q, ex, ey, ox, oy, pa, pb in reversed(circles):
                        dx, dy = restored[q][0] - ex, restored[q][1] - ey
                        d2 = dx * dx + dy * dy
                        if d2 < 1e-26:
                            return
                        t = 2.0 * ((ox - ex) * dx + (oy - ey) * dy) / d2
                        if t <= 1e-12 or t >= 1.0 - 1e-12:
                            return
                        sx, sy = ex + t * dx, ey + t * dy
                        abx, aby = pb[0] - pa[0], pb[1] - pa[1]
                        side_e = abx * (ey - pa[1]) - aby * (ex - pa[0])
                        side_s = abx * (sy - pa[1]) - aby * (sx - pa[0])
                        if side_e * side_s > 1e-24:
                            return
                        restored[s] = (sx, sy)
                    cs = ps + [restored[-i] for i in range(1, k - 1)]
                    mapped = [(a if a >= 0 else k - a - 1,
                               b if b >= 0 else k - b - 1) for a, b in es]
                    consider(cs, mapped)

                embed(0)
            return result

        def boundary(selected):
            inside = {}
            for a, b in selected:
                inside[a] = inside.get(a, 0) + 1
                inside[b] = inside.get(b, 0) + 1
            removed = {u for u, degree in inside.items() if u < 0 and degree == len(adj[u])}
            anchors = sorted(u for u in inside if u not in removed)
            return anchors, removed, inside

        def region(center, target, stop, destructive=False):
            if center not in adj or len(adj[center]) < 2:
                return None
            neighbors = list(adj[center])
            if center >= 0 and len(neighbors) > 2:
                if destructive or rng.random() < 0.3:
                    neighbors = rng.sample(neighbors, 2)
                else:
                    def wedge(pair):
                        a, b = pair
                        q = fermat(coords[center], coords[a], coords[b])
                        return (dist(coords[center], coords[a]) + dist(coords[center], coords[b])
                                - dist(q, coords[center]) - dist(q, coords[a]) - dist(q, coords[b]))
                    neighbors = list(max(itertools.combinations(neighbors, 2), key=wedge))
            selected = {tuple(sorted((center, v))) for v in neighbors}
            anchors, removed, inside = boundary(selected)
            free = limit - (len(coords) - n)
            while len(anchors) < target and clock() < stop:
                options = []
                for u in inside:
                    missing = [v for v in adj[u] if tuple(sorted((u, v))) not in selected]
                    if not missing:
                        continue
                    additions = [missing] if u < 0 else [[v] for v in missing]
                    for vs in additions:
                        extra = {tuple(sorted((u, v))) for v in vs}
                        aa, rr, ii = boundary(selected | extra)
                        if len(aa) > target:
                            continue
                        if not destructive and len(aa) - 2 > free + len(rr):
                            continue
                        weight = sum(dist(coords[u], coords[v]) for v in vs)
                        options.append((weight * (0.35 + rng.random()), extra, aa, rr, ii))
                if not options:
                    break
                _, extra, anchors, removed, inside = min(options, key=lambda item: item[0])
                selected |= extra
            if len(anchors) < 3:
                return None
            return selected, anchors, removed

        def replace(selected, anchors, removed, proposal):
            nonlocal next_id, value
            cs, es, length = proposal
            value -= sum(dist(coords[a], coords[b]) for a, b in selected)
            for a, b in selected:
                adj[a].remove(b)
                adj[b].remove(a)
            for u in removed:
                del coords[u]
                del adj[u]
            ids = anchors[:]
            for p in cs[len(anchors):]:
                ids.append(next_id)
                coords[next_id] = p
                adj[next_id] = set()
                next_id -= 1
            for a, b in es:
                u, v = ids[a], ids[b]
                adj[u].add(v)
                adj[v].add(u)
            value += length
            prune(ids)

        def refresh(stop):
            nonlocal adj, value
            ids = list(coords)
            es = mst([coords[u] for u in ids], stop)
            if es is None:
                return
            fresh_value = sum(dist(coords[ids[a]], coords[ids[b]]) for a, b in es)
            if fresh_value <= value + 1e-12:
                adj = {u: set() for u in ids}
                for a, b in es:
                    u, v = ids[a], ids[b]
                    adj[u].add(v)
                    adj[v].add(u)
                value = fresh_value
                prune(ids)
                record()

        def local_relax(seeds, stop):
            """Propagate coordinate changes along the tree and return the centres whose insertion gains need recomputing."""
            nonlocal value
            queue = [u for u in seeds if u < 0 and u in adj]
            pending = set(queue)
            touched = set(seeds)
            cursor = 0
            while cursor < len(queue) and clock() < stop:
                u = queue[cursor]
                cursor += 1
                pending.discard(u)
                if u not in adj or not adj[u]:
                    continue
                ps = [coords[v] for v in adj[u]]
                q = median(ps, coords[u], stop)
                movement = dist(q, coords[u])
                if movement < 2e-11:
                    continue
                old = sum(dist(coords[u], p) for p in ps)
                new = sum(dist(q, p) for p in ps)
                if new > old:
                    continue
                coords[u] = q
                value += new - old
                touched.add(u)
                touched.update(adj[u])
                for v in adj[u]:
                    if v < 0 and v not in pending:
                        pending.add(v)
                        queue.append(v)
            return touched

        def grow(stop):
            """An incremental gain heap selects adjacent-edge splits; coordinate changes invalidate related candidates for recomputation."""
            nonlocal next_id, value
            heap = []
            versions = {}

            def push(u):
                versions[u] = versions.get(u, 0) + 1
                if u not in adj or len(adj[u]) < 2 or (u < 0 and len(adj[u]) <= 3):
                    return
                best = 1e-12
                choice = None
                for a, b in itertools.combinations(adj[u], 2):
                    if clock() >= stop:
                        break
                    q = fermat(coords[u], coords[a], coords[b])
                    gain = (dist(coords[u], coords[a]) + dist(coords[u], coords[b])
                            - dist(q, coords[u]) - dist(q, coords[a]) - dist(q, coords[b]))
                    if gain > best:
                        best = gain
                        choice = (a, b, q)
                if choice is not None:
                    heapq.heappush(heap, (-best, u, versions[u], choice))

            if len(coords) - n >= limit:
                return
            for u in list(adj):
                if clock() >= stop:
                    break
                push(u)
            while heap and len(coords) - n < limit and clock() < stop:
                negative_gain, u, version, choice = heapq.heappop(heap)
                if versions.get(u) != version or u not in adj:
                    continue
                a, b, q = choice
                if a not in adj[u] or b not in adj[u]:
                    continue
                s = next_id
                next_id -= 1
                coords[s] = q
                adj[s] = {u, a, b}
                adj[u].difference_update((a, b))
                adj[u].add(s)
                for v in (a, b):
                    adj[v].remove(u)
                    adj[v].add(s)
                value += negative_gain
                touched = local_relax((s, u, a, b),
                                      min(stop, clock() + budget / (20.0 + 12.0 * n)))
                for v in touched:
                    push(v)
                record()
            if clock() < stop:
                optimize(stop)
            else:
                prune(list(coords))
                record()

        def nni(stop):
            """Swap the branches on both sides of an internal edge and judge real improvement by whole-tree coordinate elimination."""
            nonlocal coords, adj, value
            candidates = [(u, v) for u, v in all_edges()
                          if u < 0 and v < 0 and len(adj[u]) == len(adj[v]) == 3]
            candidates.sort(key=lambda e: dist(coords[e[0]], coords[e[1]]) *
                            (0.5 + rng.random()))
            for left, right in candidates:
                if clock() >= stop:
                    return
                if left not in adj or right not in adj or right not in adj[left]:
                    continue
                if len(adj[left]) != 3 or len(adj[right]) != 3:
                    continue
                a = next(v for v in adj[left] if v != right)
                choices = [v for v in adj[right] if v != left]
                for b in choices:
                    if clock() >= stop:
                        return
                    saved_coords = coords.copy()
                    saved_adj = {u: vs.copy() for u, vs in adj.items()}
                    saved_value = value
                    value += (dist(coords[left], coords[b]) + dist(coords[right], coords[a])
                              - dist(coords[left], coords[a]) - dist(coords[right], coords[b]))
                    adj[left].remove(a)
                    adj[a].remove(left)
                    adj[right].remove(b)
                    adj[b].remove(right)
                    adj[left].add(b)
                    adj[b].add(left)
                    adj[right].add(a)
                    adj[a].add(right)
                    optimize(min(stop, clock() + budget / (40.0 + 3.0 * n)))
                    if value < saved_value - 1e-11:
                        record()
                        break
                    coords, adj, value = saved_coords, saved_adj, saved_value

        def insert_anywhere(stop):
            """Non-adjacent triples generate Fermat points that compete for slots via Kruskal reconnection of the augmented tree."""
            nonlocal coords, adj, value, next_id
            centers = list(coords)
            rng.shuffle(centers)
            seen = set()
            width = min(len(centers) - 1, 3 + int(math.log1p(budget) + math.log1p(n) / 3.0))
            for center in centers:
                if clock() >= stop:
                    return
                if center not in coords:
                    continue
                near = heapq.nsmallest(width, (v for v in coords if v != center),
                                      key=lambda v: dist(coords[center], coords[v]))
                pool = list(adj[center])
                pool.extend(v for v in near if v not in adj[center])
                triples = list(itertools.combinations(pool, 2))
                rng.shuffle(triples)
                for a, b in triples:
                    if clock() >= stop:
                        return
                    if a not in coords or b not in coords:
                        continue
                    signature = tuple(sorted((center, a, b)))
                    if signature in seen:
                        continue
                    seen.add(signature)
                    q = fermat(coords[center], coords[a], coords[b])
                    if min(dist(q, coords[v]) for v in signature) < 1e-11:
                        continue
                    ids = list(coords)
                    s = next_id
                    weighted = [(dist(coords[u], coords[v]), u, v) for u, v in all_edges()]
                    weighted.extend((dist(q, coords[v]), s, v) for v in ids)
                    weighted.sort()
                    parents = {u: u for u in ids}
                    parents[s] = s
                    sizes = {u: 1 for u in parents}

                    def find(u):
                        while parents[u] != u:
                            parents[u] = parents[parents[u]]
                            u = parents[u]
                        return u

                    es = []
                    length = 0.0
                    for w, u, v in weighted:
                        ru, rv = find(u), find(v)
                        if ru == rv:
                            continue
                        if sizes[ru] < sizes[rv]:
                            ru, rv = rv, ru
                        parents[rv] = ru
                        sizes[ru] += sizes[rv]
                        es.append((u, v))
                        length += w
                        if len(es) == len(ids):
                            break
                    if length >= value - 1e-11 or clock() >= stop:
                        continue
                    old_coords, old_adj, old_value = coords, adj, value
                    coords = coords.copy()
                    coords[s] = q
                    adj = {u: set() for u in coords}
                    for u, v in es:
                        adj[u].add(v)
                        adj[v].add(u)
                    value = length
                    prune(list(coords))
                    if len(coords) - n > limit or value >= old_value - 1e-11:
                        coords, adj, value = old_coords, old_adj, old_value
                        continue
                    next_id -= 1
                    record()
                    optimize(min(stop, clock() + budget / (24.0 + math.sqrt(n))))
                    return

        # Full topology enumeration depth scales with budget and size; actual execution is still bounded by the wall clock.
        depth = 3
        work = 1.0
        capacity = max(1.0, budget * 12000.0 / (12.0 + math.sqrt(n)))
        while depth < n and work * (2 * depth - 3) * 2 <= capacity:
            work *= (2 * depth - 3) * 2
            depth += 1

        def reconstruct(stop, epoch):
            centers = [u for u in adj if len(adj[u]) >= 2]
            rng.shuffle(centers)
            for center in centers:
                if clock() >= stop:
                    break
                if center not in adj:
                    continue
                upper = min(depth, 4 + epoch)
                target = upper if rng.random() < 0.55 else rng.randint(3, upper)
                patch_stop = min(stop, clock() + budget / (32.0 + 2.0 * n))
                patch = region(center, target, patch_stop)
                if patch is None:
                    continue
                selected, anchors, removed = patch
                slots = limit - (len(coords) - n) + len(removed)
                old_length = sum(dist(coords[a], coords[b]) for a, b in selected)
                proposal = component([coords[u] for u in anchors], slots, old_length, patch_stop)
                if proposal is None or len(proposal[0]) - len(anchors) > slots:
                    continue
                replace(selected, anchors, removed, proposal)
                record()
                optimize(min(stop, clock() + budget / (40.0 + 4.0 * n)))

        def restore_best():
            nonlocal coords, adj, value
            coords = best_coords.copy()
            adj = {u: vs.copy() for u, vs in best_adj.items()}
            value = best_value

        def perturb(stalled, epoch, stop):
            nonlocal coords, adj, value
            restore_best()
            # Regional destruction and noisy terminal trees are both repaired by the same insertion, geometry and subtree operators.
            if epoch % 3 == 2:
                noise = min(0.45, 0.008 * math.log1p(budget) *
                            (1.0 + math.sqrt(stalled)) * (0.5 + 2.0 * rng.random()))
                es = mst(points, stop, noise)
                if es is not None:
                    coords = dict(enumerate(points))
                    adj = {u: set() for u in coords}
                    for a, b in es:
                        adj[a].add(b)
                        adj[b].add(a)
                    value = sum(dist(points[a], points[b]) for a, b in es)
                return
            steiners = [u for u in coords if u < 0]
            eligible = steiners or [u for u in adj if len(adj[u]) >= 2]
            if not eligible:
                return
            strength = max(1, int(math.sqrt(max(1, len(steiners))) *
                                  math.log1p(budget) * (0.15 + 0.05 * math.sqrt(stalled))))
            target = min(n, 3 + rng.randrange(1 + strength))
            patch = region(rng.choice(eligible), target, stop, True)
            if patch is None:
                return
            selected, anchors, removed = patch
            ps = [coords[u] for u in anchors]
            es = mst(ps, stop)
            if es is not None:
                length = sum(dist(ps[a], ps[b]) for a, b in es)
                replace(selected, anchors, removed, (ps, es, length))
                record()

        # The first tree is built quickly; each later round applies both reference mechanisms to the same search state.
        grow(min(deadline, clock() + budget * 0.13))
        refresh(min(deadline, clock() + budget / (30.0 + math.sqrt(n))))
        grow(min(deadline, clock() + budget * 0.04))
        epoch = 0
        stalled = 0
        while clock() < deadline:
            before = best_value
            cycle = budget / (4.0 + math.log1p(n) + math.log1p(budget))
            cycle_stop = min(deadline, clock() + cycle)
            grow(min(cycle_stop, clock() + cycle * 0.18))
            optimize(min(cycle_stop, clock() + cycle * 0.10))
            nni(min(cycle_stop, clock() + cycle * 0.22))
            reconstruct(min(cycle_stop, clock() + cycle * 0.30), epoch)
            refresh(min(cycle_stop, clock() + cycle * 0.08))
            grow(min(cycle_stop, clock() + cycle * 0.10))
            insert_anywhere(cycle_stop)
            record()
            if best_value < before - 1e-10 * max(1.0, before):
                stalled = 0
            else:
                stalled += 1
                perturb(stalled, epoch, min(deadline, clock() + cycle * 0.10))
            epoch += 1
        # Refine only the saved best tree; unfinished candidates never overwrite the complete feasible solution.
        restore_best()
        if clock() < finish:
            optimize(min(finish, clock() + max(0.0, finish - clock()) * 0.55))
        if clock() < finish:
            refresh(finish)
        record()
    except Exception:
        # All search happens after a complete feasible solution exists; on numerical exceptions that solution is kept as is.
        pass
    return answer
