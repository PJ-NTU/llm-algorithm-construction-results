"""Hybrid search with persistent edge penalties and exchange chains
Combines vectorised guided descent on augmented distances, exchange chains on original distances, segment dynamic programming and regret reconstruction, always saving the complete tour with the shortest original length.
"""

import math
import random
import time

try:
    import numpy as np
except Exception:
    np = None


def solve(instance: dict, time_limit_s: float) -> dict:
    clock = time.perf_counter
    started = clock()
    best_tour = []
    try:
        dist = instance["distance_matrix"]
        n = len(dist)
        best_tour = list(range(n))
        limit = max(0.0, float(time_limit_s))
        if not math.isfinite(limit):
            limit = 0.0
        deadline = started + limit * 0.97
        if n <= 2 or clock() >= deadline:
            return {"tour": best_tour}

        def cost(tour):
            return sum(dist[tour[i - 1]][tour[i]] for i in range(n))

        best_cost = cost(best_tour)
        eps = 1e-12 * max(1.0, abs(best_cost) / n)
        rng = random.Random(1784892296 ^ (n * 1000003))

        def remember(tour):
            nonlocal best_tour, best_cost
            value = cost(tour)
            if value < best_cost:
                best_cost, best_tour = value, tour[:]
            return value

        # Keep one complete greedy tour before building the distance array and candidate lists.
        remaining = set(range(1, n))
        greedy = [0]
        stop = min(deadline, started + limit * 0.06)
        while remaining and clock() < stop:
            nxt = min(remaining, key=dist[greedy[-1]].__getitem__)
            greedy.append(nxt)
            remaining.remove(nxt)
        greedy.extend(sorted(remaining))
        remember(greedy)
        if clock() >= deadline:
            return {"tour": best_tour}

        if np is None:
            current = best_tour[:]
            while clock() < deadline:
                i, j = sorted(rng.sample(range(n), 2))
                if rng.random() < 0.5:
                    trial = current[:i] + current[i:j + 1][::-1] + current[j + 1:]
                else:
                    trial = current[:]
                    trial.insert(j, trial.pop(i))
                value = remember(trial)
                if value <= cost(current):
                    current = trial
                elif rng.random() < 1.0 / max(2, n):
                    current = best_tour[:]
                    a, b = rng.sample(range(n), 2)
                    current[a], current[b] = current[b], current[a]
            z = best_tour.index(0)
            return {"tour": best_tour[z:] + best_tour[:z]}
        D = np.asarray(dist, dtype=float)
        if clock() >= deadline:
            return {"tour": best_tour}
        ids = np.arange(n)
        budget_scale = math.log1p(limit)
        width = min(n - 1, max(6, int(6 + 3 * budget_scale + 2 * math.log1p(n))))
        chain_length = min(n - 2, max(1, 2 + int(budget_scale / 1.5)))
        pool_size = max(2, int(2 + budget_scale + math.log1p(n) ** 0.5))
        max_ruin = min(n - 2, max(2, int(math.sqrt(n) * (1 + budget_scale / 4))))
        block = max(1, min(32, 65536 // n))

        symmetric = True
        for lo in range(0, n, block):
            if clock() >= deadline:
                return {"tour": best_tour}
            hi = min(n, lo + block)
            if not np.array_equal(D[lo:hi, :], D[:, lo:hi].T):
                symmetric = False
                break

        near = []
        incoming = []
        for v in range(n):
            if clock() >= deadline:
                return {"tour": best_tour}
            row = D[v]
            q = np.argpartition(row, width)[:width + 1]
            q = q[q != v]
            q = q[np.argsort(row[q], kind="stable")][:width]
            near.append([int(x) for x in q])
            if not symmetric:
                row = D[:, v]
                q = np.argpartition(row, width)[:width + 1]
                q = q[q != v]
                q = q[np.argsort(row[q], kind="stable")][:width]
                incoming.append([int(x) for x in q])
        if symmetric:
            incoming = near

        def reverse_prefix(tour):
            prefix = [0.0] * n
            if not symmetric:
                for k in range(n - 1):
                    a, b = tour[k], tour[k + 1]
                    prefix[k + 1] = prefix[k] + dist[b][a] - dist[a][b]
            return prefix

        def candidate_descent(tour, stop):
            pos = [0] * n
            for i, v in enumerate(tour):
                pos[v] = i
            prefix = reverse_prefix(tour)
            order = list(range(n))
            rng.shuffle(order)
            while clock() < stop:
                moved = False
                for a in order:
                    if clock() >= stop:
                        return
                    for c in near[a]:
                        accepted = False
                        # Scan both endpoint orientations so that neither kind of new short edge is missed.
                        for shift in (0, -1):
                            i, j = (pos[a] + shift) % n, (pos[c] + shift) % n
                            if i > j:
                                i, j = j, i
                            if j <= i + 1 or (symmetric and i == 0 and j == n - 1):
                                continue
                            aa, b = tour[i], tour[i + 1]
                            cc, d = tour[j], tour[(j + 1) % n]
                            delta = dist[aa][cc] + dist[b][d] - dist[aa][b] - dist[cc][d]
                            if not symmetric:
                                delta += prefix[j] - prefix[i + 1]
                            if delta < -eps:
                                tour[i + 1:j + 1] = tour[i + 1:j + 1][::-1]
                                for p in range(i + 1, j + 1):
                                    pos[tour[p]] = p
                                if not symmetric:
                                    prefix = reverse_prefix(tour)
                                moved = accepted = True
                                break
                        if accepted:
                            break
                if not moved:
                    return

        def full_two(tour, stop):
            p = np.asarray(tour, dtype=int)
            q = np.roll(p, -1)
            edge = D[p, q]
            prefix = np.asarray(reverse_prefix(tour))
            best_delta, move = -eps, None
            for lo in range(0, n - 1, block):
                if clock() >= stop:
                    break
                ii = ids[lo:min(n - 1, lo + block)]
                delta = (D[p[ii, None], p[None, :]] + D[q[ii, None], q[None, :]]
                         - edge[ii, None] - edge[None, :])
                if not symmetric:
                    delta += prefix[None, :] - prefix[ii + 1, None]
                valid = ids[None, :] > ii[:, None] + 1
                if symmetric and lo == 0:
                    valid[0, -1] = False
                delta[~valid] = np.inf
                flat = int(np.argmin(delta))
                r, j = divmod(flat, n)
                value = float(delta[r, j])
                if value < best_delta:
                    best_delta, move = value, (int(ii[r]), j)
            if move is None:
                return False
            i, j = move
            tour[i + 1:j + 1] = tour[i + 1:j + 1][::-1]
            return True

        def relocate(tour, stop, exhaustive):
            p = np.asarray(tour, dtype=int)
            q = np.roll(p, -1)
            edge = D[p, q]
            prefix = np.asarray(reverse_prefix(tour))
            pos = [0] * n
            for i, v in enumerate(tour):
                pos[v] = i
            for length in range(1, chain_length + 1):
                for lo in range(0, n - length + 1, block if exhaustive else 1):
                    if clock() >= stop:
                        return False
                    ii = ids[lo:min(n - length + 1, lo + (block if exhaustive else 1))]
                    jj = ids
                    if not exhaustive:
                        targets = {pos[v] for v in incoming[tour[lo]]}
                        targets.update((pos[v] - 1) % n for v in near[tour[lo + length - 1]])
                        jj = np.asarray(sorted(targets), dtype=int)
                        if len(jj) == 0:
                            continue
                    first, last = p[ii], p[ii + length - 1]
                    before, after = p[(ii - 1) % n], p[(ii + length) % n]
                    base = D[before, after] - D[before, first] - D[last, after]
                    delta = (base[:, None] + D[p[jj][None, :], first[:, None]]
                             + D[last[:, None], q[jj][None, :]] - edge[jj][None, :])
                    valid = ((jj[None, :] < ii[:, None]) | (jj[None, :] >= ii[:, None] + length))
                    valid &= jj[None, :] != ((ii - 1) % n)[:, None]
                    delta[~valid] = np.inf
                    flat = int(np.argmin(delta))
                    r, c = divmod(flat, len(jj))
                    value, rev = float(delta[r, c]), False
                    if length > 1:
                        back = (base[:, None] + D[p[jj][None, :], last[:, None]]
                                + D[first[:, None], q[jj][None, :]] - edge[jj][None, :])
                        if not symmetric:
                            back += (prefix[ii + length - 1] - prefix[ii])[:, None]
                        back[~valid] = np.inf
                        other = int(np.argmin(back))
                        rr, cc = divmod(other, len(jj))
                        if float(back[rr, cc]) < value:
                            r, c, value, rev = rr, cc, float(back[rr, cc]), True
                    if value < -eps:
                        i, j = int(ii[r]), int(jj[c])
                        segment = tour[i:i + length]
                        if rev:
                            segment.reverse()
                        rest = tour[:i] + tour[i + length:]
                        at = (j if j < i else j - length) + 1
                        tour[:] = rest[:at] + segment + rest[at:]
                        return True
            return False

        def swap(tour, stop, exhaustive):
            p = np.asarray(tour, dtype=int)
            prev, nxt = np.roll(p, 1), np.roll(p, -1)
            incident = D[prev, p] + D[p, nxt]
            pos = [0] * n
            for i, v in enumerate(tour):
                pos[v] = i
            for lo in range(0, n - 2, block if exhaustive else 1):
                if clock() >= stop:
                    return False
                ii = ids[lo:min(n - 2, lo + (block if exhaustive else 1))]
                jj = ids
                if not exhaustive:
                    targets = {pos[v] for v in near[tour[(lo - 1) % n]]}
                    targets.update(pos[v] for v in incoming[tour[(lo + 1) % n]])
                    jj = np.asarray(sorted(targets), dtype=int)
                    if len(jj) == 0:
                        continue
                delta = (D[prev[ii, None], p[jj][None, :]] + D[p[jj][None, :], nxt[ii, None]]
                         + D[prev[jj][None, :], p[ii, None]] + D[p[ii, None], nxt[jj][None, :]]
                         - incident[ii, None] - incident[jj][None, :])
                valid = jj[None, :] > ii[:, None] + 1
                valid &= ~((ii[:, None] == 0) & (jj[None, :] == n - 1))
                delta[~valid] = np.inf
                flat = int(np.argmin(delta))
                r, c = divmod(flat, len(jj))
                if float(delta[r, c]) < -eps:
                    i, j = int(ii[r]), int(jj[c])
                    tour[i], tour[j] = tour[j], tour[i]
                    return True
            return False

        def three_opt(tour, stop):
            if n < 6:
                return False
            p = np.asarray(tour, dtype=int)
            q = np.roll(p, -1)
            edge = D[p, q]
            prefix = np.asarray(reverse_prefix(tour))
            pos = [0] * n
            for i, v in enumerate(tour):
                pos[v] = i
            exhaustive = n ** 3 <= max(limit, 0.001) * 600000
            order = list(range(n - 4))
            rng.shuffle(order)
            for i in order:
                if clock() >= stop:
                    return False
                a, b = tour[i], tour[i + 1]
                if exhaustive:
                    js = list(range(i + 2, n - 2))
                else:
                    candidates = set()
                    for v in near[a] + incoming[b]:
                        candidates.add(pos[v])
                        candidates.add((pos[v] - 1) % n)
                    js = sorted(j for j in candidates if i + 2 <= j < n - 2)
                for lo in range(0, len(js), block):
                    if clock() >= stop:
                        return False
                    jj = np.asarray(js[lo:lo + block], dtype=int)
                    c, d = p[jj, None], q[jj, None]
                    old = edge[i] + edge[jj, None] + edge[None, :]
                    values = [
                        D[a, c] + D[b, p][None, :] + D[d, q[None, :]] - old,
                        D[a, d] + D[p, b][None, :] + D[c, q[None, :]] - old,
                        D[a, d] + D[p[None, :], c] + D[b, q][None, :] - old,
                        D[a, p][None, :] + D[d, b] + D[c, q[None, :]] - old,
                    ]
                    if not symmetric:
                        rb = prefix[jj, None] - prefix[i + 1]
                        rc = prefix[None, :] - prefix[jj + 1, None]
                        values[0] += rb + rc
                        values[2] += rb
                        values[3] += rc
                    valid = ids[None, :] >= jj[:, None] + 2
                    if i == 0:
                        valid[:, -1] = False
                    winner, delta_best = None, -eps
                    for mode, delta in enumerate(values):
                        delta[~valid] = np.inf
                        flat = int(np.argmin(delta))
                        r, k = divmod(flat, n)
                        if float(delta[r, k]) < delta_best:
                            delta_best = float(delta[r, k])
                            winner = int(jj[r]), k, mode
                    if winner is not None:
                        j, k, mode = winner
                        left, right = tour[i + 1:j + 1], tour[j + 1:k + 1]
                        if mode == 0:
                            middle = left[::-1] + right[::-1]
                        elif mode == 1:
                            middle = right + left
                        elif mode == 2:
                            middle = right + left[::-1]
                        else:
                            middle = right[::-1] + left
                        tour[i + 1:k + 1] = middle
                        return True
            return False

        exchange_depth = min(n - 2, max(2, int(2 + budget_scale + math.log1p(n) / 3)))
        exchange_width = max(1, int(1 + math.sqrt(budget_scale)))

        def exchange_chain(tour, stop):
            # Keep an open Hamiltonian path, attach candidate edges and reverse the tail segment.
            # A positive-gain chain that is not yet closed can cross ordinary 2-opt and 3-opt local optima.
            if not symmetric or n < 5:
                return False

            def edge_key(a, b):
                return min(a, b) * n + max(a, b)

            def extend(path, gain, removed, added, depth):
                if clock() >= stop:
                    return None
                start, end = path[0], path[-1]
                position = [0] * n
                for j, v in enumerate(path):
                    position[v] = j
                choices = []
                for c in near[end]:
                    j = position[c]
                    if j == 0 or j >= n - 2:
                        continue
                    insertion = dist[end][c]
                    if gain - insertion <= eps:
                        continue
                    d = path[j + 1]
                    put, cut = edge_key(end, c), edge_key(c, d)
                    if put in removed or cut in added:
                        continue
                    new_gain = gain - insertion + dist[c][d]
                    closed_gain = new_gain - dist[d][start]
                    if closed_gain > eps:
                        return path[:j + 1] + path[j + 1:][::-1]
                    if depth < exchange_depth:
                        choices.append((-closed_gain, -new_gain, j, put, cut))
                choices.sort()
                breadth = exchange_width if depth <= exchange_width else 1
                for _, minus_gain, j, put, cut in choices[:breadth]:
                    if clock() >= stop:
                        return None
                    result = extend(path[:j + 1] + path[j + 1:][::-1],
                                    -minus_gain, removed | {cut}, added | {put}, depth + 1)
                    if result is not None:
                        return result
                return None

            base = tour[:] if rng.random() < 0.5 else tour[::-1]
            roots = list(range(n))
            rng.shuffle(roots)
            original = cost(tour)
            for i in roots:
                if clock() >= stop:
                    return False
                path = base[i:] + base[:i]
                a, b = path[0], path[-1]
                result = extend(path, dist[a][b], {edge_key(a, b)}, set(), 1)
                if result is not None and cost(result) < original - eps:
                    tour[:] = result
                    return True
            return False

        # All constructions and perturbations share the same descent chain.
        def improve(tour, stop, strong):
            exhaustive = n * n <= max(limit, 0.001) * 120000
            while clock() < stop:
                candidate_descent(tour, stop)
                remember(tour)
                if clock() >= stop:
                    break
                if full_two(tour, stop):
                    continue
                if relocate(tour, stop, exhaustive):
                    continue
                if swap(tour, stop, exhaustive):
                    continue
                if strong and symmetric and exchange_chain(
                        tour, min(stop, clock() + limit / (30 + math.sqrt(n)))):
                    continue
                if strong and three_opt(tour, stop):
                    continue
                break
            return tour, remember(tour)

        def nearest(start, randomized, stop):
            used = np.zeros(n, dtype=bool)
            used[start] = True
            tour = [start]
            while len(tour) < n:
                if clock() >= stop:
                    tour.extend(int(x) for x in ids[~used])
                    break
                choices = [v for v in near[tour[-1]] if not used[v]]
                if not choices:
                    row = np.where(used, np.inf, D[tour[-1]])
                    node = int(np.argmin(row))
                elif randomized and len(choices) > 1:
                    rank = min(len(choices) - 1, int(-math.log(max(rng.random(), 1e-15)) / 1.3))
                    node = choices[rank]
                else:
                    node = choices[0]
                tour.append(node)
                used[node] = True
            return tour

        def farthest(stop):
            far = int(np.argmax(D[0] + D[:, 0]))
            if far == 0:
                far = 1
            tour = [0, far]
            used = np.zeros(n, dtype=bool)
            used[tour] = True
            proximity = np.minimum(D[0] + D[:, 0], D[far] + D[:, far])
            while len(tour) < n:
                if clock() >= stop:
                    tour.extend(int(x) for x in ids[~used])
                    break
                v = int(np.argmax(np.where(used, -np.inf, proximity)))
                p = np.asarray(tour, dtype=int)
                q = np.roll(p, -1)
                at = int(np.argmin(D[p, v] + D[v, q] - D[p, q])) + 1
                tour.insert(at, v)
                used[v] = True
                proximity = np.minimum(proximity, D[v] + D[:, v])
            return tour

        def bridge(tour):
            if n < 5:
                result = tour[:]
                i, j = rng.sample(range(n), 2)
                result[i], result[j] = result[j], result[i]
                return result
            a, b, c, d = sorted(rng.sample(range(n), 4))
            return tour[:a] + tour[c:d] + tour[b:c] + tour[a:b] + tour[d:]

        def recreate(tour, strength, stop):
            strength = min(n - 2, strength)
            roll = rng.random()
            if roll < 0.35:
                seed = rng.randrange(n)
                candidates = [tour[seed]] + near[tour[seed]]
                removed = candidates[:strength]
                if len(removed) < strength:
                    rest = list(set(tour) - set(removed))
                    removed.extend(rng.sample(rest, strength - len(removed)))
            elif roll < 0.7:
                removed = rng.sample(tour, strength)
            else:
                sample = rng.sample(range(n), min(n, strength * 3))
                sample.sort(key=lambda i: dist[tour[i - 1]][tour[i]]
                            + dist[tour[i]][tour[(i + 1) % n]]
                            - dist[tour[i - 1]][tour[(i + 1) % n]], reverse=True)
                removed = [tour[i] for i in sample[:strength]]
            removed_set = set(removed)
            base = [v for v in tour if v not in removed_set]
            rng.shuffle(removed)
            regret = rng.random() < 0.75
            while removed:
                if clock() >= stop:
                    base.extend(removed)
                    break
                p = np.asarray(base, dtype=int)
                q = np.roll(p, -1)
                pending = np.asarray(removed if regret else removed[:1], dtype=int)
                delta = D[p[None, :], pending[:, None]] + D[pending[:, None], q[None, :]] - D[p, q][None, :]
                at = np.argmin(delta, axis=1)
                if regret and len(removed) > 1:
                    pair = np.partition(delta, 1, axis=1)[:, :2]
                    regrets = pair[:, 1] - pair[:, 0]
                    chosen = int(np.lexsort((delta[np.arange(len(pending)), at], -regrets))[0])
                else:
                    chosen = 0
                base.insert(int(at[chosen]) + 1, removed.pop(chosen))
            return base

        # Penalties and paths of the guided search persist across restarts; original distances and the best record are not contaminated.
        guide_matrix = None
        guide_tour = None
        guide_penalties = {}
        guide_near = guide_incoming = guide_pairs = None
        guide_lambda = 0.0
        guide_shift = 0.0
        guide_block = max(1, min(n, 16384 // max(1, 2 * width)))

        def guided_move(tour, stop):
            W = guide_matrix
            p = np.asarray(tour, dtype=int)
            q = np.roll(p, -1)
            prev = np.roll(p, 1)
            edge = W[p, q]
            position = np.empty(n, dtype=int)
            position[p] = ids
            prefix = None
            if not symmetric:
                prefix = np.concatenate(([0.0], np.cumsum(W[q, p] - edge)))
            winner, lowest = None, -eps
            # Consider both endpoint orientations of a candidate edge as the new edge.
            for shift in (0, -1):
                for lo in range(0, n, guide_block):
                    if clock() >= stop:
                        return winner
                    hi = min(n, lo + guide_block)
                    ii = (position[ids[lo:hi], None] + shift) % n
                    jj = (position[guide_pairs[lo:hi]] + shift) % n
                    left, right = np.minimum(ii, jj), np.maximum(ii, jj)
                    a, b = p[left], p[(left + 1) % n]
                    c, d = p[right], q[right]
                    delta = W[a, c] + W[b, d] - edge[left] - edge[right]
                    if not symmetric:
                        delta += prefix[right] - prefix[left + 1]
                    valid = right > left + 1
                    if symmetric:
                        valid &= ~((left == 0) & (right == n - 1))
                    delta[~valid] = np.inf
                    index = int(np.argmin(delta))
                    r, col = divmod(index, delta.shape[1])
                    value = float(delta[r, col])
                    if value < lowest:
                        lowest = value
                        winner = (0, int(left[r, col]), int(right[r, col]), 0, False)
            if winner is not None:
                return winner

            # In directed graphs, a shift keeps the direction within the segment; a reversed shift recosts all internal arcs.
            for length in range(1, chain_length + 1):
                for lo in range(0, n - length + 1, guide_block):
                    if clock() >= stop:
                        return None
                    ii = ids[lo:min(n - length + 1, lo + guide_block)]
                    first, last = p[ii], p[ii + length - 1]
                    before, after = prev[ii], p[(ii + length) % n]
                    targets = np.concatenate((position[guide_incoming[first]],
                                              (position[guide_near[last]] - 1) % n), axis=1)
                    tails, heads = p[targets], q[targets]
                    base = (W[before, after] - W[before, first] - W[last, after])[:, None]
                    delta = (base + W[tails, first[:, None]] + W[last[:, None], heads]
                             - edge[targets])
                    valid = ((targets < ii[:, None]) | (targets >= ii[:, None] + length))
                    valid &= targets != ((ii - 1) % n)[:, None]
                    delta[~valid] = np.inf
                    index = int(np.argmin(delta))
                    r, col = divmod(index, delta.shape[1])
                    value, backward = float(delta[r, col]), False
                    if length > 1:
                        reverse = (base + W[tails, last[:, None]] + W[first[:, None], heads]
                                   - edge[targets])
                        if not symmetric:
                            reverse += (prefix[ii + length - 1] - prefix[ii])[:, None]
                        reverse[~valid] = np.inf
                        index = int(np.argmin(reverse))
                        rr, cc = divmod(index, reverse.shape[1])
                        if float(reverse[rr, cc]) < value:
                            r, col, value, backward = rr, cc, float(reverse[rr, cc]), True
                    if value < -eps:
                        return (1, int(ii[r]), int(targets[r, col]), length, backward)

            incident = W[prev, p] + edge
            for lo in range(0, n, guide_block):
                if clock() >= stop:
                    return None
                ii = ids[lo:min(n, lo + guide_block)]
                targets = np.concatenate((position[guide_near[prev[ii]]],
                                          position[guide_incoming[q[ii]]]), axis=1)
                left = ii[:, None]
                delta = (W[prev[left], p[targets]] + W[p[targets], q[left]]
                         + W[prev[targets], p[left]] + W[p[left], q[targets]]
                         - incident[left] - incident[targets])
                valid = (targets != left) & (targets != (left - 1) % n)
                valid &= targets != (left + 1) % n
                delta[~valid] = np.inf
                index = int(np.argmin(delta))
                r, col = divmod(index, delta.shape[1])
                if float(delta[r, col]) < -eps:
                    return (2, int(ii[r]), int(targets[r, col]), 0, False)
            return None

        def guided_search(stop):
            nonlocal guide_matrix, guide_tour, guide_near, guide_incoming
            nonlocal guide_pairs, guide_lambda, guide_shift
            if guide_matrix is None:
                # Initialise in blocks; an incomplete augmented matrix is never used.
                working = np.empty_like(D)
                for lo in range(0, n, block):
                    if clock() >= stop:
                        return best_tour[:]
                    working[lo:lo + block] = D[lo:lo + block]
                guide_matrix = working
                guide_near = np.asarray(near, dtype=int)
                guide_incoming = guide_near if symmetric else np.asarray(incoming, dtype=int)
                guide_pairs = (guide_near if symmetric else
                               np.concatenate((guide_near, guide_incoming), axis=1))
                guide_tour = best_tour[:]
                scale = sum(abs(dist[guide_tour[i - 1]][guide_tour[i]]) for i in range(n)) / n
                guide_lambda = max(eps, 0.12 * scale)
                guide_shift = min(0.0, float(np.min(D)))
            while clock() < stop:
                move = guided_move(guide_tour, stop)
                if move is not None:
                    kind, i, j, length, backward = move
                    if kind == 0:
                        guide_tour[i + 1:j + 1] = guide_tour[i + 1:j + 1][::-1]
                    elif kind == 1:
                        segment = guide_tour[i:i + length]
                        if backward:
                            segment.reverse()
                        rest = guide_tour[:i] + guide_tour[i + length:]
                        at = (j if j < i else j - length) + 1
                        guide_tour = rest[:at] + segment + rest[at:]
                    else:
                        guide_tour[i], guide_tour[j] = guide_tour[j], guide_tour[i]
                    # A descent in guided distance does not necessarily shorten the original tour, so each step is evaluated independently.
                    remember(guide_tour)
                    continue
                if clock() >= stop:
                    break
                edges, utilities = [], []
                for i, b in enumerate(guide_tour):
                    a = guide_tour[i - 1]
                    key = (min(a, b), max(a, b)) if symmetric else (a, b)
                    count = guide_penalties.get(key, 0)
                    edges.append((a, b, key, count))
                    utilities.append((dist[a][b] - guide_shift) / (1 + count))
                highest = max(utilities)
                for utility, (a, b, key, count) in zip(utilities, edges):
                    if clock() >= stop:
                        break
                    if utility >= highest - eps:
                        count += 1
                        guide_penalties[key] = count
                        guide_matrix[a, b] = dist[a][b] + guide_lambda * count
                        if symmetric:
                            guide_matrix[b, a] = guide_matrix[a, b]
                # Rotate where the cycle is cut so short chain shifts gradually cover the former start/end boundary.
                cut = rng.randrange(n)
                guide_tour = guide_tour[cut:] + guide_tour[:cut]
            return guide_tour[:]

        subset_layers = {}

        def reconnect(tour, stop):
            # Anchor at a single node; the other contiguous segments can be reordered and reversed, keeping each segment's internal arc cost.
            available = max(0.0, stop - clock())
            allowance = available * 3000000
            m = 1
            while (m < n - 1 and (m + 1) ** 2 * (1 << (m + 1)) <= allowance
                   and 16 * (m + 1) * (1 << (m + 1)) <= 16777216):
                m += 1
            if m < 3 or clock() >= stop:
                return tour[:], False
            anchor = rng.randrange(n)
            route = tour[anchor:] + tour[:anchor]
            position = [0] * n
            for i, v in enumerate(route):
                position[v] = i
            cuts = {0, n - 1}
            # Mix unusually long edges with candidate neighbour edges so the same reconnection subproblem contains edges that may replace each other.
            ranked = list(range(1, n - 1))
            ranked.sort(key=lambda i: (
                dist[route[i]][route[i + 1]]
                - 0.5 * (dist[route[i]][near[route[i]][0]]
                         + dist[incoming[route[i + 1]][0]][route[i + 1]]))
                * (0.5 + rng.random()), reverse=True)
            cursor = 0
            while len(cuts) < m + 1:
                if clock() >= stop:
                    return tour[:], False
                if rng.random() < 0.7:
                    e = rng.choice(tuple(cuts))
                    v = route[(e + rng.randrange(2)) % n]
                    neighbors = near[v] if rng.random() < 0.5 else incoming[v]
                    rank = min(len(neighbors) - 1,
                               int(-math.log(max(rng.random(), 1e-15)) * math.sqrt(m)))
                    j = position[neighbors[rank]]
                    for cut in (j, (j - 1) % n):
                        if len(cuts) < m + 1:
                            cuts.add(cut)
                if len(cuts) < m + 1:
                    while ranked[cursor] in cuts:
                        cursor += 1
                    cuts.add(ranked[cursor])
            bounds = sorted(cuts)
            pieces = [route[bounds[j] + 1:bounds[j + 1] + 1] for j in range(m)]
            starts, ends, internal = [], [], []
            for segment in pieces:
                starts.extend((segment[0], segment[-1]))
                ends.extend((segment[-1], segment[0]))
                forward = sum(dist[a][b] for a, b in zip(segment, segment[1:]))
                backward = forward if symmetric else sum(
                    dist[b][a] for a, b in zip(segment, segment[1:]))
                internal.extend((forward, backward))
            starts = np.asarray(starts, dtype=int)
            ends = np.asarray(ends, dtype=int)
            internal = np.asarray(internal)
            transitions = D[ends[:, None], starts[None, :]] + internal[None, :]
            states = 1 << m
            table = np.full((states, 2 * m), np.inf)
            for j in range(m):
                table[1 << j, 2 * j:2 * j + 2] = (
                    D[route[0], starts[2 * j:2 * j + 2]] + internal[2 * j:2 * j + 2])
            if m not in subset_layers:
                layers = [[] for _ in range(m + 1)]
                for mask in range(1, states):
                    layers[mask.bit_count()].append(mask)
                subset_layers[m] = [np.asarray(layer, dtype=int) for layer in layers]
            layers = subset_layers[m]
            batch = max(1, 32768 // (2 * m))
            for size in range(1, m):
                for j in range(m):
                    if clock() >= stop:
                        return tour[:], False
                    bit = 1 << j
                    masks = layers[size][(layers[size] & bit) == 0]
                    for lo in range(0, len(masks), batch):
                        if clock() >= stop:
                            return tour[:], False
                        indices = masks[lo:lo + batch]
                        previous = table[indices]
                        table[indices | bit, 2 * j] = np.min(
                            previous + transitions[:, 2 * j], axis=1)
                        table[indices | bit, 2 * j + 1] = np.min(
                            previous + transitions[:, 2 * j + 1], axis=1)
            state = int(np.argmin(table[-1] + D[ends, route[0]]))
            mask, sequence = states - 1, []
            while mask:
                j, backward = divmod(state, 2)
                sequence.append(pieces[j][::-1] if backward else pieces[j])
                mask ^= 1 << j
                if mask:
                    state = int(np.argmin(table[mask] + transitions[:, state]))
            candidate = [route[0]]
            for segment in reversed(sequence):
                candidate.extend(segment)
            value = remember(candidate)
            # A global optimality certificate exists only when every segment is a single node and the dynamic programme finishes.
            proved = m == n - 1
            if value < cost(tour) - eps:
                return candidate, proved
            return tour[:], proved

        elites = []

        def keep(tour, value):
            z = tour.index(0)
            route = tour[z:] + tour[:z]
            key = tuple(route)
            if symmetric:
                key = min(key, (0,) + tuple(reversed(route[1:])))
            if any(item[2] == key for item in elites):
                return
            elites.append((value, tour[:], key))
            elites.sort(key=lambda item: item[0])
            del elites[pool_size:]

        # When the budget allows, first cover the whole small problem with the same reconnection operator.
        if n - 1 <= 16 and (n - 1) ** 2 * (1 << (n - 1)) <= limit * 240000:
            _, proved = reconnect(best_tour, min(deadline, clock() + limit * 0.12))
            if proved:
                z = best_tour.index(0)
                return {"tour": best_tour[z:] + best_tour[:z]}

        opening = min(deadline, started + limit * 0.22)
        now = clock()
        tour, value = improve(best_tour[:], min(opening, now + limit * 0.09), False)
        keep(tour, value)
        if clock() < opening:
            tour = farthest(min(opening, clock() + limit * 0.025))
            remember(tour)
            tour, value = improve(tour, min(opening, clock() + limit * 0.09), False)
            keep(tour, value)
        while clock() < opening:
            stop = min(opening, clock() + limit / (8 + math.sqrt(n)))
            tour = nearest(rng.randrange(n), True, stop)
            tour, value = improve(tour, stop, False)
            keep(tour, value)

        if clock() < deadline:
            tour, value = improve(best_tour[:], min(deadline, started + limit * 0.34), True)
            keep(tour, value)
        current, current_cost = best_tour[:], best_cost
        last_gain = clock()
        deep_at = clock()
        # No cap on restarts; on stagnation, enlarge the destruction range, relax the acceptance threshold and deepen the neighbourhood together.
        while clock() < deadline:
            now = clock()
            before = best_cost
            stagnant = (now - last_gain) / max(limit * 0.10, 1e-9)
            roll = rng.random()
            if roll < 0.50:
                source = best_tour
            elif roll < 0.82 or not elites:
                source = current
            else:
                source = rng.choice(elites)[1]
            stop = min(deadline, now + limit / (8 + math.sqrt(n)) * (1 + min(stagnant, 2) / 2))
            roll = rng.random()
            if roll < 0.56:
                candidate = guided_search(min(stop, now + (stop - now) * 0.78))
            elif roll < 0.69:
                candidate, proved = reconnect(source, min(stop, now + (stop - now) * 0.55))
                if proved:
                    break
                if cost(candidate) >= cost(source) - eps:
                    candidate = bridge(source)
            elif roll < 0.85:
                candidate = bridge(source)
                if rng.random() < min(0.5, stagnant / 8):
                    candidate = bridge(candidate)
            elif roll < 0.97:
                strength = min(max_ruin, max(2, int(2 + (1 + stagnant) * math.sqrt(max_ruin))))
                candidate = recreate(source, strength, stop)
            else:
                candidate = nearest(rng.randrange(n), True, stop)
            # Change where the cycle is cut so fixed-length segments can also cover the former start/end boundary.
            cut = rng.randrange(n)
            candidate = candidate[cut:] + candidate[:cut]
            strong = now >= deep_at
            candidate, value = improve(candidate, stop, strong)
            if best_cost < before - eps and clock() < deadline:
                candidate, value = improve(best_tour[:], min(deadline, clock() + limit / (6 + math.sqrt(n))), True)
            if strong:
                deep_at = clock() + limit / (12 + math.sqrt(n))
            keep(candidate, value)
            if best_cost < before - eps:
                last_gain = clock()
                current, current_cost = best_tour[:], best_cost
            else:
                tolerance = abs(best_cost) * (0.001 + min(0.012, stagnant * 0.002))
                if value < current_cost or value <= best_cost + tolerance:
                    current, current_cost = candidate, value
                else:
                    current, current_cost = best_tour[:], best_cost
    except Exception:
        # The best record is never modified in place, so an error or timeout never leaves half a tour.
        pass
    if best_tour:
        z = best_tour.index(0)
        best_tour = best_tour[z:] + best_tour[:z]
    return {"tour": best_tour}
