"""Fermat-point insertion and tree-topology neighbourhood search
Starts from the Euclidean minimum spanning tree and alternates Fermat-point insertion, continuous coordinate optimisation, branch swaps and local destroy-and-rebuild, keeping the best feasible tree.
"""

import math
import time
import random
import numpy as np
from scipy.spatial import Delaunay, cKDTree
from scipy.optimize import minimize


class _Timeout(Exception):
    pass


def _fermat(a, b, c):
    ab, ac, bc = b - a, c - a, c - b
    aa = np.einsum('ij,ij->i', ab, ab)
    bb = np.einsum('ij,ij->i', ac, ac)
    cc = np.einsum('ij,ij->i', bc, bc)
    da = np.einsum('ij,ij->i', ab, ac)
    db = -np.einsum('ij,ij->i', ab, bc)
    dc = np.einsum('ij,ij->i', ac, bc)
    area = np.abs(ab[:, 0] * ac[:, 1] - ab[:, 1] * ac[:, 0])
    den = area[:, None] + math.sqrt(3.0) * np.column_stack((da, db, dc))
    w = 1.0 / np.maximum(den, 1e-30)
    w /= w.sum(axis=1)[:, None]
    f = w[:, :1] * a + w[:, 1:2] * b + w[:, 2:] * c
    for mask, p in ((da <= -0.5 * np.sqrt(aa * bb), a),
                    (db <= -0.5 * np.sqrt(aa * cc), b),
                    (dc <= -0.5 * np.sqrt(bb * cc), c)):
        f[mask] = p[mask]
    return f


def solve(instance: dict, time_limit_s: float) -> dict:
    started = time.perf_counter()
    result = {"steiner_points": [], "edges": []}
    try:
        raw = instance.get("points", [])
        n = len(raw)
        maximum = max(0, int(instance.get("max_steiner_points", 0)))
        anchor = [float(raw[0][0]), float(raw[0][1])] if n else [0.0, 0.0]
        result = {"steiner_points": [anchor[:] for _ in range(maximum)],
                  "edges": [[0, i] for i in range(1, n + maximum)]}
        if n < 2:
            return result
        budget = max(0.0, float(time_limit_s))
        if not math.isfinite(budget):
            budget = 0.0
        if budget <= 0.0:
            return result
        p = np.asarray(raw, dtype=float)[:, :2].copy()
        if not np.isfinite(p).all():
            return result
        origin = p[0].copy()
        scale = float(np.max(np.abs(p - origin)))
        if scale == 0.0:
            return result
        p = (p - origin) / scale
        cap = min(maximum, n - 2)
        intensity = 1.0 + math.log1p(budget)
        rng = random.Random(704193)
        reserve = min(0.06 * budget, 0.025 + 0.000015 * (n + maximum))
        end = started + budget - reserve
        hard_end = started + budget
        best_q = p.copy()
        best_e = np.asarray([[0, i] for i in range(1, n)], dtype=np.int64)
        best_value = float(np.linalg.norm(p[1:] - p[0], axis=1).sum())
        eps = 1e-11
        breadth = max(4, int(2 + math.sqrt(intensity * math.log2(n + 1))))

        def check(until):
            if time.perf_counter() >= until:
                raise _Timeout()

        def length(q, e):
            return float(np.linalg.norm(q[e[:, 0]] - q[e[:, 1]], axis=1).sum())

        def remember(q, e):
            nonlocal best_q, best_e, best_value
            value = length(q, e)
            if math.isfinite(value) and value < best_value - eps * (1.0 + best_value):
                best_value, best_q, best_e = value, q.copy(), e.copy()
            return value

        def adjacency(k, e):
            adj = [[] for _ in range(k)]
            for j, (a, b) in enumerate(e):
                a, b = int(a), int(b)
                adj[a].append((b, j))
                adj[b].append((a, j))
            return adj

        def mst(q, until):
            check(until)
            k = len(q)
            if k <= 1:
                return np.empty((0, 2), dtype=np.int64)
            # Large point sets use the Delaunay edge set, which contains the Euclidean minimum spanning tree.
            if k > 48:
                try:
                    uq, representatives, inverse = np.unique(
                        q, axis=0, return_index=True, return_inverse=True)
                    u = len(uq)
                    extra = [(int(representatives[inverse[i]]), i)
                             for i in range(k) if representatives[inverse[i]] != i]
                    if u == 1:
                        return np.asarray(extra, dtype=np.int64).reshape(-1, 2)
                    if u == 2:
                        extra.append((int(representatives[0]), int(representatives[1])))
                        return np.asarray(extra, dtype=np.int64)
                    check(until)
                    tri = Delaunay(uq).simplices
                    pairs = np.vstack((tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]))
                    pairs = np.unique(np.sort(pairs, axis=1), axis=0)
                    d = uq[pairs[:, 0]] - uq[pairs[:, 1]]
                    order = np.argsort(np.einsum('ij,ij->i', d, d), kind='stable')
                    parent = list(range(u))
                    size = [1] * u

                    def root(a):
                        while parent[a] != a:
                            parent[a] = parent[parent[a]]
                            a = parent[a]
                        return a

                    edges = list(extra)
                    for step, j in enumerate(order):
                        if step % 256 == 0:
                            check(until)
                        a, b = map(int, pairs[j])
                        ra, rb = root(a), root(b)
                        if ra != rb:
                            if size[ra] < size[rb]:
                                ra, rb = rb, ra
                            parent[rb] = ra
                            size[ra] += size[rb]
                            edges.append((int(representatives[a]), int(representatives[b])))
                            if len(edges) == k - 1:
                                return np.asarray(edges, dtype=np.int64)
                except _Timeout:
                    raise
                except Exception:
                    pass
            # Small point sets and degenerate geometry use dense Prim, correctly handling coincident points.
            d = np.full(k, np.inf)
            parent = np.zeros(k, dtype=np.int64)
            used = np.zeros(k, dtype=bool)
            d[0] = 0.0
            edges = []
            for step in range(k):
                check(until)
                v = int(np.argmin(d))
                if step:
                    edges.append((int(parent[v]), v))
                used[v] = True
                delta = q - q[v]
                nd = np.einsum('ij,ij->i', delta, delta)
                improve = (nd < d) & ~used
                parent[improve] = v
                d[improve] = nd[improve]
                d[used] = np.inf
            return np.asarray(edges, dtype=np.int64)

        def strip(q, e):
            # Delete non-terminal leaves and contract non-terminal degree-2 points; tree length does not increase.
            adj = [set() for _ in range(len(q))]
            for a, b in e:
                adj[int(a)].add(int(b))
                adj[int(b)].add(int(a))
            alive = np.ones(len(q), dtype=bool)
            pending = [i for i in range(n, len(q)) if len(adj[i]) <= 2]
            while pending:
                v = pending.pop()
                if not alive[v] or len(adj[v]) > 2:
                    continue
                nb = list(adj[v])
                alive[v] = False
                for a in nb:
                    adj[a].discard(v)
                if len(nb) == 2:
                    a, b = nb
                    adj[a].add(b)
                    adj[b].add(a)
                adj[v].clear()
                for a in nb:
                    if a >= n and len(adj[a]) <= 2:
                        pending.append(a)
            if alive.all():
                return q, e
            indices = np.flatnonzero(alive)
            mapping = np.full(len(q), -1, dtype=np.int64)
            mapping[indices] = np.arange(len(indices))
            edges = [(int(mapping[a]), int(mapping[b]))
                     for a in indices for b in adj[a] if a < b]
            return q[indices].copy(), np.asarray(edges, dtype=np.int64).reshape(-1, 2)

        def opt(q, e, until):
            if len(q) == n or time.perf_counter() >= until:
                return q
            work = q.copy()
            u, v = e[:, 0], e[:, 1]
            um, vm = u >= n, v >= n
            ui, vi = u[um] - n, v[vm] - n
            m = len(q) - n
            best_x = q[n:].ravel().copy()
            value = length(q, e)
            initial_value = value
            smooth = max(1e-12, value / max(1, len(e)) * 1e-8 / intensity)

            def fun(x):
                nonlocal value, best_x
                check(until)
                work[n:] = x.reshape(m, 2)
                delta = work[u] - work[v]
                squared = np.einsum('ij,ij->i', delta, delta)
                actual = float(np.sqrt(squared).sum())
                if actual < value:
                    value, best_x = actual, x.copy()
                dist = np.sqrt(squared + smooth * smooth)
                force = delta / dist[:, None]
                grad = np.empty((m, 2))
                for axis in (0, 1):
                    grad[:, axis] = (
                        np.bincount(ui, weights=force[um, axis], minlength=m)
                        - np.bincount(vi, weights=force[vm, axis], minlength=m))
                return float(dist.sum()), grad.ravel()

            try:
                minimize(fun, best_x, jac=True, method='L-BFGS-B',
                         options={'ftol': 1e-12 / intensity, 'gtol': 1e-7 / intensity,
                                  'maxcor': max(5, int(3 * intensity))})
            except Exception:
                pass
            if value < initial_value:
                q = q.copy()
                q[n:] = best_x.reshape(m, 2)
            remember(q, e)
            return q

        def opt_slice(until, k, fraction=0.18):
            now = time.perf_counter()
            duration = 0.0008 * (k + 1) ** 0.65 * intensity
            return min(until, now + min(max(0.0, until - now) * fraction, duration))

        def path_data(q, e, until):
            check(until)
            k = len(q)
            adj = adjacency(k, e)
            weights = np.linalg.norm(q[e[:, 0]] - q[e[:, 1]], axis=1)
            order = np.argsort(weights, kind='stable')
            rank = np.empty(len(e), dtype=np.int64)
            rank[order] = np.arange(len(e))
            parent = np.zeros(k, dtype=np.int64)
            depth = np.zeros(k, dtype=np.int64)
            mx = np.full(k, -1, dtype=np.int64)
            stack = [0]
            while stack:
                a = stack.pop()
                for b, j in adj[a]:
                    if b != parent[a]:
                        parent[b] = a
                        depth[b] = depth[a] + 1
                        mx[b] = rank[j]
                        stack.append(b)
            up, top = [parent], [mx]
            while (1 << len(up)) < k:
                up.append(up[-1][up[-1]])
                top.append(np.maximum(top[-1], top[-1][up[-2]]))
            check(until)

            def query(a, b):
                a, b = a.copy(), b.copy()
                swap = depth[a] < depth[b]
                a[swap], b[swap] = b[swap].copy(), a[swap].copy()
                difference = depth[a] - depth[b]
                ans = np.full(len(a), -1, dtype=np.int64)
                for level in range(len(up)):
                    mask = (difference & (1 << level)) != 0
                    ans[mask] = np.maximum(ans[mask], top[level][a[mask]])
                    a[mask] = up[level][a[mask]]
                for level in range(len(up) - 1, -1, -1):
                    mask = up[level][a] != up[level][b]
                    ans[mask] = np.maximum(ans[mask], np.maximum(
                        top[level][a[mask]], top[level][b[mask]]))
                    a[mask], b[mask] = up[level][a[mask]], up[level][b[mask]]
                mask = a != b
                ans[mask] = np.maximum(ans[mask], np.maximum(top[0][a[mask]], top[0][b[mask]]))
                return ans

            return adj, weights[order], order, query

        def triples(q, adj, until, noisy):
            candidates = []
            for v, neighbors in enumerate(adj):
                if v % 128 == 0:
                    check(until)
                nb = [a for a, _ in neighbors]
                if len(nb) > breadth:
                    nb = rng.sample(nb, breadth)
                for i, a in enumerate(nb):
                    for b in nb[i + 1:]:
                        candidates.append((v, a, b))
            check(until)
            try:
                if len(q) >= 3:
                    candidates.extend(Delaunay(q).simplices.tolist())
            except Exception:
                pass
            if noisy and len(q) > 3 and time.perf_counter() < until:
                k = min(len(q), breadth)
                _, near = cKDTree(q).query(q, k=k)
                for v, row in enumerate(near):
                    nb = [int(a) for a in row if a != v]
                    if len(nb) >= 2:
                        for _ in range(max(1, int(math.sqrt(intensity)))):
                            a, b = rng.sample(nb, 2)
                            candidates.append((v, a, b))
            if not candidates:
                return np.empty((0, 3), dtype=np.int64)
            return np.unique(np.sort(np.asarray(candidates, dtype=np.int64), axis=1), axis=0)

        def insert(q, e, until, noise):
            adj, ranked_weights, edge_order, query = path_data(q, e, until)
            tri = triples(q, adj, until, noise > 0.0)
            if not len(tri):
                return None
            check(until)
            a, b, c = q[tri[:, 0]], q[tri[:, 1]], q[tri[:, 2]]
            f = _fermat(a, b, c)
            cost = (np.linalg.norm(f - a, axis=1) + np.linalg.norm(f - b, axis=1)
                    + np.linalg.norm(f - c, axis=1))
            bottleneck = np.column_stack((query(tri[:, 0], tri[:, 1]),
                                          query(tri[:, 0], tri[:, 2]),
                                          query(tri[:, 1], tri[:, 2])))
            lo, hi = bottleneck.min(axis=1), bottleneck.max(axis=1)
            # Minimum and maximum bottleneck ranks of the three paths, corresponding to two tree edges that can be removed together.
            gain = ranked_weights[lo] + ranked_weights[hi] - cost
            valid = (lo >= 0) & (lo != hi) & (gain > eps * (1.0 + cost))
            if not valid.any():
                return None
            score = gain.copy()
            if noise:
                score *= np.asarray([1.0 + noise * rng.random() for _ in score])
            score[~valid] = -np.inf
            j = int(np.argmax(score))
            keep = np.ones(len(e), dtype=bool)
            keep[edge_order[lo[j]]] = False
            keep[edge_order[hi[j]]] = False
            z = len(q)
            edges = np.vstack((e[keep], np.column_stack((tri[j], np.full(3, z)))))
            coords = np.vstack((q, f[j]))
            coords, edges = strip(coords, edges)
            remember(coords, edges)
            return coords, edges

        def complete(q, e, until, noise):
            # Zero-gain splits are also kept as search topology, so adjacent Steiner points can move cooperatively.
            while len(q) - n < cap and time.perf_counter() < until:
                adj = adjacency(len(q), e)
                choices = []
                for v, neighbors in enumerate(adj):
                    if len(neighbors) < (2 if v < n else 4):
                        continue
                    nb = neighbors if len(neighbors) <= breadth else rng.sample(neighbors, breadth)
                    for i, (a, ea) in enumerate(nb):
                        for b, eb in nb[i + 1:]:
                            choices.append((v, a, b, ea, eb))
                if not choices:
                    break
                ch = np.asarray(choices, dtype=np.int64)
                a, b, c = q[ch[:, 0]], q[ch[:, 1]], q[ch[:, 2]]
                f = _fermat(a, b, c)
                gain = (np.linalg.norm(a - b, axis=1) + np.linalg.norm(a - c, axis=1)
                        - np.linalg.norm(f - a, axis=1) - np.linalg.norm(f - b, axis=1)
                        - np.linalg.norm(f - c, axis=1))
                score = gain + np.asarray([rng.random() for _ in gain]) * (
                    eps + noise * 0.005 * (1.0 + length(q, e)) / n)
                j = int(np.argmax(score))
                v, a, b, ea, eb = map(int, ch[j])
                keep = np.ones(len(e), dtype=bool)
                keep[[ea, eb]] = False
                z = len(q)
                e = np.vstack((e[keep], [[v, z], [a, z], [b, z]]))
                q = np.vstack((q, f[j]))
                remember(q, e)
            return q, e

        def fill(q, e, until, noise):
            batch = max(1, int(math.sqrt(max(1, cap)) / math.sqrt(intensity)))
            since_opt = 0
            try:
                while len(q) - n < cap and time.perf_counter() < until:
                    addition = insert(q, e, until, noise)
                    if addition is None:
                        break
                    q, e = addition
                    since_opt += 1
                    if since_opt >= batch:
                        q = opt(q, e, opt_slice(until, len(q)))
                        since_opt = 0
                q = opt(q, e, opt_slice(until, len(q)))
                q, e = complete(q, e, until, noise)
                q = opt(q, e, opt_slice(until, len(q)))
            except _Timeout:
                pass
            remember(q, e)
            return q, e

        def nni(q, e, until):
            adj = adjacency(len(q), e)
            moves = []
            for u, v in e:
                u, v = int(u), int(v)
                if u < n:
                    u, v = v, u
                if u < n or len(adj[u]) != 3 or len(adj[v]) < 2:
                    continue
                if v >= n and len(adj[v]) != 3:
                    continue
                un = [(a, j) for a, j in adj[u] if a != v]
                vn = [(b, j) for b, j in adj[v] if b != u]
                for a, ea in (un[:1] if v >= n else un):
                    other_u = next(t for t, _ in un if t != a)
                    for b, eb in vn:
                        other_v = next((t for t, _ in vn if t != b), v) if v >= n else v
                        moves.append((u, v, a, b, other_u, other_v, ea, eb))
            if not moves or time.perf_counter() >= until:
                return q, e, False
            mv = np.asarray(moves, dtype=np.int64)
            u, v, a, b, ou, ov = (mv[:, j] for j in range(6))
            free_v = v >= n
            A, B, C, D = q[ou], q[b], q[ov], q[a]
            s = (A + B + q[v]) / 3.0
            t = q[v].copy()
            t[free_v] = ((C + D + q[u]) / 3.0)[free_v]
            old = (np.linalg.norm(q[u] - q[ou], axis=1)
                   + np.linalg.norm(q[u] - q[a], axis=1)
                   + np.linalg.norm(q[u] - q[v], axis=1)
                   + np.linalg.norm(q[v] - q[b], axis=1)
                   + free_v * np.linalg.norm(q[v] - q[ov], axis=1))
            local_until = time.perf_counter() + max(0.0, until - time.perf_counter()) * 0.22
            previous = np.full(len(moves), np.inf)
            while time.perf_counter() < local_until:
                s = _fermat(A, B, t)
                if free_v.any():
                    t[free_v] = _fermat(C[free_v], D[free_v], s[free_v])
                costs = (np.linalg.norm(s - A, axis=1) + np.linalg.norm(s - B, axis=1)
                         + np.linalg.norm(s - t, axis=1) + np.linalg.norm(t - D, axis=1)
                         + free_v * np.linalg.norm(t - C, axis=1))
                if np.max(np.abs(previous - costs)) <= 1e-10:
                    break
                previous = costs
            costs = (np.linalg.norm(s - A, axis=1) + np.linalg.norm(s - B, axis=1)
                     + np.linalg.norm(s - t, axis=1) + np.linalg.norm(t - D, axis=1)
                     + free_v * np.linalg.norm(t - C, axis=1))
            order = np.argsort(costs - old, kind='stable')
            value = length(q, e)
            # After ranking by local score, optimise the whole tree, allowing outside Steiner points to adjust jointly.
            for j in order:
                if time.perf_counter() >= until:
                    break
                uu, vv, aa, bb, _, _, ea, eb = map(int, mv[j])
                trial_q, trial_e = q.copy(), e.copy()
                trial_q[uu] = s[j]
                if vv >= n:
                    trial_q[vv] = t[j]
                trial_e[ea], trial_e[eb] = (uu, bb), (vv, aa)
                trial_q = opt(trial_q, trial_e, opt_slice(until, len(q), 0.6))
                score = remember(trial_q, trial_e)
                if score < value - eps * (1.0 + value):
                    return trial_q, trial_e, True
            return q, e, False

        try:
            e = mst(p, end)
            q = p.copy()
            remember(q, e)
            terminal_e = e.copy()
            if cap > 0:
                first_end = min(end, time.perf_counter() + max(0.0, end - time.perf_counter()) * 0.45)
                q, e = fill(q, e, first_end, 0.0)
                stagnation = 0
                while time.perf_counter() < end:
                    now = time.perf_counter()
                    phase = (0.006 + 0.0015 * (len(q) + 1) ** 0.65) * intensity
                    phase_end = min(end, now + min((end - now) * 0.32, phase))
                    q, e, improved = nni(q, e, phase_end)
                    if improved:
                        stagnation = 0
                        continue
                    check(end)
                    # Recomputing the point-set MST can change several edges at once and release useless Steiner points.
                    mq, me = strip(q, mst(q, end))
                    mvalue = remember(mq, me)
                    if mvalue < length(q, e) - eps * (1.0 + mvalue):
                        q, e = fill(mq, me, min(end, time.perf_counter() + phase), 0.0)
                        stagnation = 0
                        continue
                    stagnation += 1
                    source_q, source_e = (best_q, best_e) if rng.random() < 0.7 else (q, e)
                    m = len(source_q) - n
                    if not m or rng.random() < 1.0 / (2.0 + intensity):
                        tq, te = p.copy(), terminal_e.copy()
                        removed = max(1, m)
                    else:
                        fraction = min(0.75, 0.06 + 0.38 * rng.random() ** 2
                                       + 0.045 * math.log1p(stagnation) / math.sqrt(intensity))
                        removed = min(m, max(1, int(math.ceil(m * fraction))))
                        if rng.random() < 0.7:
                            center = source_q[n + rng.randrange(m)]
                            distance = np.sum((source_q[n:] - center) ** 2, axis=1)
                            deleted = np.argsort(distance, kind='stable')[:removed] + n
                        else:
                            deleted = np.asarray(rng.sample(range(n, len(source_q)), removed))
                        keep = np.ones(len(source_q), dtype=bool)
                        keep[deleted] = False
                        tq = source_q[keep].copy()
                        tq, te = strip(tq, mst(tq, end))
                    now = time.perf_counter()
                    rebuild_end = min(end, now + min((end - now) * 0.48,
                                                      phase * (1.0 + math.sqrt(removed))))
                    progress = min(1.0, max(0.0, (now - started) / budget))
                    noise = (0.25 + rng.random()) * (1.0 - 0.7 * progress) / math.sqrt(intensity)
                    tq, te = fill(tq, te, rebuild_end, noise)
                    value, current_value = remember(tq, te), length(q, e)
                    temperature = (best_value / max(1, n - 1) * 0.025
                                   * (1.0 - 0.8 * progress) / math.sqrt(intensity))
                    if value <= current_value or rng.random() < math.exp(
                            min(0.0, (current_value - value) / max(temperature, 1e-15))):
                        q, e = tq, te
        except Exception:
            pass

        # Take a true MST once more before output; zero-length padding points do not change its optimal value.
        if time.perf_counter() < hard_end - reserve * 0.3:
            try:
                final_e = mst(best_q, hard_end - reserve * 0.3)
                remember(best_q, final_e)
            except Exception:
                pass

        coords = (best_q[n:] * scale + origin).tolist()
        edges = best_e.tolist()
        for j in range(len(coords), maximum):
            coords.append(anchor[:])
            edges.append([0, n + j])
        result = {"steiner_points": coords, "edges": edges}
    except Exception:
        pass
    return result
