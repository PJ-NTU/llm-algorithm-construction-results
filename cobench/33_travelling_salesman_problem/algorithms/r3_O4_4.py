"""Two-ended exchange chains with elite-guided exact segment reconnection
Uses two-ended backtracking exchanges, elite-difference edge cutting and a subset dynamic programme with deduplicated orientation states, deepening the search adaptively with the remaining wall clock.
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

        exchange_depth = min(n - 2, max(2, int(3 + 2 * budget_scale + math.log1p(n) / 2)))
        exchange_width = max(2, int(2 + math.sqrt(budget_scale)))

        def exchange_chain(tour, stop):
            # Both ends of the open path can take new edges; in-place reversal and rollback reuse node positions.
            # gain always equals the original tour length minus the current open-path length.
            if not symmetric or n < 5:
                return False

            def edge_key(a, b):
                return min(a, b) * n + max(a, b)

            def flip(side, j):
                lo, hi = (j + 1, n) if side == 0 else (0, j)
                path[lo:hi] = path[lo:hi][::-1]
                for k in range(lo, hi):
                    position[path[k]] = k

            def extend(gain, removed, added, depth):
                if clock() >= stop:
                    return None
                choices = []
                winner, winner_gain = None, eps
                for side in (0, 1):
                    endpoint = path[-1] if side == 0 else path[0]
                    other = path[0] if side == 0 else path[-1]
                    for c in near[endpoint]:
                        insertion = dist[endpoint][c]
                        # Candidate lists are sorted by increasing distance, so later candidates cannot keep a positive prefix gain either.
                        if gain - insertion <= eps:
                            break
                        j = position[c]
                        if (side == 0 and not 0 < j < n - 2) or (side == 1 and not 1 < j < n - 1):
                            continue
                        d = path[j + 1] if side == 0 else path[j - 1]
                        put, cut = edge_key(endpoint, c), edge_key(c, d)
                        if put in removed or put in added or cut in added or cut in removed:
                            continue
                        new_gain = gain - insertion + dist[c][d]
                        closed_gain = new_gain - dist[d][other]
                        if closed_gain > winner_gain:
                            winner_gain = closed_gain
                            winner = side, j
                        if depth < exchange_depth:
                            choices.append((-closed_gain, -new_gain, side, j, put, cut))
                if winner is not None:
                    flip(*winner)
                    result = path[:]
                    flip(*winner)
                    return result
                choices.sort()
                breadth = max(1, exchange_width - (depth - 1) // 3)
                for _, minus_gain, side, j, put, cut in choices[:breadth]:
                    if clock() >= stop:
                        return None
                    flip(side, j)
                    result = extend(-minus_gain, removed | {cut}, added | {put}, depth + 1)
                    flip(side, j)
                    if result is not None:
                        return result
                return None

            roots = list(range(n))
            # Loosen arcs that are expensive relative to candidate edges first; a random factor avoids starting at the same place each time.
            roots.sort(key=lambda i: (
                dist[tour[i - 1]][tour[i]]
                - 0.5 * (dist[tour[i - 1]][near[tour[i - 1]][0]]
                         + dist[tour[i]][near[tour[i]][0]]))
                * (0.5 + rng.random()), reverse=True)
            original = cost(tour)
            for i in roots:
                if clock() >= stop:
                    return False
                path = tour[i:] + tour[:i]
                position = [0] * n
                for j, v in enumerate(path):
                    position[v] = j
                a, b = path[0], path[-1]
                result = extend(dist[a][b], {edge_key(a, b)}, set(), 1)
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

        subset_layers = {}
        dp_rate = 8000000.0

        def reconnect_once(tour, stop, guide):
            nonlocal dp_rate
            available = max(0.0, stop - clock())
            allowance = available * dp_rate * 0.55
            m = 1
            # Time estimates set the neighbourhood size; the memory constraint only prevents the exponential state table from growing too large.
            while (m < n - 1 and (m + 1) ** 2 * (1 << (m + 1)) <= allowance
                   and 16 * (m + 1) * (1 << (m + 1)) <= 33554432):
                m += 1
            if m < 2 or clock() >= stop:
                return tour[:], False, False
            fragment_count = m + 1
            position = [0] * n
            for i, v in enumerate(tour):
                position[v] = i
            cuts = set()
            anchor = rng.randrange(n)
            if fragment_count == n:
                cuts.update(range(n))
            else:
                # Directed problems fix a single-node anchor; for symmetric problems fixing the orientation of any segment suffices.
                if not symmetric:
                    cuts.update(((anchor - 1) % n, anchor))
                difference = []
                if guide is not None:
                    links = set()
                    for i, a in enumerate(guide):
                        b = guide[(i + 1) % n]
                        links.add(a * n + b)
                        if symmetric:
                            links.add(b * n + a)
                    difference = [i for i, a in enumerate(tour)
                                  if a * n + tour[(i + 1) % n] not in links]
                    # If all differences fit, both elite solutions are contained in this exact neighbourhood.
                    if len(cuts | set(difference)) <= fragment_count:
                        cuts.update(difference)
                    elif difference:
                        cuts.add(rng.choice(difference))
                ranked = list(range(n))
                different = set(difference)
                ranked.sort(key=lambda i: (
                    dist[tour[i]][tour[(i + 1) % n]]
                    - 0.5 * (dist[tour[i]][near[tour[i]][0]]
                             + dist[incoming[tour[(i + 1) % n]][0]][tour[(i + 1) % n]]))
                    * (0.5 + rng.random()) * (1.5 if i in different else 1.0), reverse=True)
                if not cuts:
                    cuts.add(ranked[0] if rng.random() < 0.6 else rng.randrange(n))
                cursor = 0
                while len(cuts) < fragment_count:
                    if clock() >= stop:
                        return tour[:], False, False
                    if rng.random() < 0.8:
                        edge = rng.choice(tuple(cuts))
                        v = tour[(edge + rng.randrange(2)) % n]
                        neighbors = near[v] if rng.random() < 0.5 else incoming[v]
                        rank = min(len(neighbors) - 1,
                                   int(-math.log(max(rng.random(), 1e-15)) * math.sqrt(m)))
                        j = position[neighbors[rank]]
                        adjacent = [j, (j - 1) % n]
                        rng.shuffle(adjacent)
                        for cut in adjacent:
                            if len(cuts) < fragment_count:
                                cuts.add(cut)
                    if len(cuts) < fragment_count:
                        while ranked[cursor] in cuts:
                            cursor += 1
                        cuts.add(ranked[cursor])

            bounds = sorted(cuts)
            pieces = []
            for j, left in enumerate(bounds):
                right = bounds[(j + 1) % fragment_count]
                segment = (tour[left + 1:right + 1] if right > left
                           else tour[left + 1:] + tour[:right + 1])
                pieces.append(segment)
            if symmetric:
                root_index = max(range(fragment_count), key=lambda j: len(pieces[j]))
            else:
                root_index = next(j for j, segment in enumerate(pieces)
                                  if len(segment) == 1 and segment[0] == anchor)
            root = pieces.pop(root_index)
            starts, ends, internal, owner, backward, offsets = [], [], [], [], [], [0]
            # A single-node segment has only one orientation, avoiding duplicate states; asymmetric internal arcs are costed per direction.
            for j, segment in enumerate(pieces):
                forward = sum(dist[a][b] for a, b in zip(segment, segment[1:]))
                starts.append(segment[0])
                ends.append(segment[-1])
                internal.append(forward)
                owner.append(j)
                backward.append(False)
                if len(segment) > 1:
                    starts.append(segment[-1])
                    ends.append(segment[0])
                    internal.append(forward if symmetric else sum(
                        dist[b][a] for a, b in zip(segment, segment[1:])))
                    owner.append(j)
                    backward.append(True)
                offsets.append(len(starts))
            starts = np.asarray(starts, dtype=int)
            ends = np.asarray(ends, dtype=int)
            internal = np.asarray(internal)
            transitions = D[ends[:, None], starts[None, :]] + internal[None, :]
            states = 1 << m
            dp_started = clock()
            table = np.full((states, len(starts)), np.inf)
            for j in range(m):
                lo, hi = offsets[j], offsets[j + 1]
                table[1 << j, lo:hi] = D[root[-1], starts[lo:hi]] + internal[lo:hi]
            if m not in subset_layers:
                masks = np.arange(states, dtype=np.int32)
                sizes = np.zeros(states, dtype=np.uint8)
                for j in range(m):
                    sizes += ((masks >> j) & 1).astype(np.uint8)
                subset_layers[m] = [masks[sizes == size] for size in range(m + 1)]
            layers = subset_layers[m]
            batch = max(1, 32768 // len(starts))
            for size in range(1, m):
                for j in range(m):
                    if clock() >= stop:
                        dp_rate = max(10000.0, dp_rate * 0.65)
                        return tour[:], False, False
                    bit = 1 << j
                    masks = layers[size][(layers[size] & bit) == 0]
                    for lo in range(0, len(masks), batch):
                        if clock() >= stop:
                            dp_rate = max(10000.0, dp_rate * 0.65)
                            return tour[:], False, False
                        indices = masks[lo:lo + batch]
                        previous = table[indices]
                        for state in range(offsets[j], offsets[j + 1]):
                            table[indices | bit, state] = np.min(
                                previous + transitions[:, state], axis=1)
            elapsed = max(clock() - dp_started, 1e-9)
            measured = m * m * states / elapsed
            dp_rate = 0.7 * dp_rate + 0.3 * min(dp_rate * 2, measured)
            state = int(np.argmin(table[-1] + D[ends, root[0]]))
            mask, sequence = states - 1, []
            while mask:
                j = owner[state]
                sequence.append(pieces[j][::-1] if backward[state] else pieces[j])
                mask ^= 1 << j
                if mask:
                    state = int(np.argmin(table[mask] + transitions[:, state]))
            candidate = root[:]
            for segment in reversed(sequence):
                candidate.extend(segment)
            value = remember(candidate)
            proved = fragment_count == n
            if value < cost(tour) - eps:
                return candidate, proved, True
            return tour[:], proved, True

        def reconnect(tour, stop):
            # Explore different edge cuts in succession within one time slice; save the complete best tour immediately after each.
            current = tour[:]
            while clock() < stop:
                guide = None
                if elites and rng.random() < 0.7:
                    guide = rng.choice(elites)[1]
                candidate, proved, completed = reconnect_once(current, stop, guide)
                if cost(candidate) < cost(current) - eps:
                    current = candidate
                if proved or not completed:
                    return current, proved
            return current, False

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
        if ((n - 1) < 24 and 16 * (n - 1) * (1 << (n - 1)) <= 33554432
                and (n - 1) ** 2 * (1 << (n - 1)) <= limit * dp_rate * 0.08):
            _, proved = reconnect(best_tour, min(deadline, clock() + limit * 0.16))
            if proved:
                z = best_tour.index(0)
                return {"tour": best_tour[z:] + best_tour[:z]}

        opening = min(deadline, started + limit * 0.20)
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
            tour, value = improve(best_tour[:], min(deadline, started + limit * 0.35), True)
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
            if roll < 0.60:
                source = best_tour
            elif roll < 0.82 or not elites:
                source = current
            else:
                source = rng.choice(elites)[1]
            stop = min(deadline, now + limit / (8 + math.sqrt(n)) * (1 + min(stagnant, 2) / 2))
            roll = rng.random()
            if roll < 0.48:
                candidate, proved = reconnect(source, min(stop, now + (stop - now) * 0.60))
                if proved:
                    break
                if cost(candidate) >= cost(source) - eps:
                    candidate = bridge(source)
            elif roll < 0.78:
                candidate = bridge(source)
                if rng.random() < min(0.5, stagnant / 8):
                    candidate = bridge(candidate)
            elif roll < 0.96:
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
