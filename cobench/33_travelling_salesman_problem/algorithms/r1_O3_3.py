"""Variable neighbourhood search fusing candidate edges and regret reconstruction
Accelerates descent with candidate edges and combines blockwise complete neighbourhoods, orientation-corrected 3-opt, regret insertion and adaptive multi-basin perturbation.
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
        limit = max(0.0, float(time_limit_s))
        if not math.isfinite(limit):
            limit = 0.0
        deadline = started + limit * 0.97
        dist = instance["distance_matrix"]
        # Each row of the matrix corresponds to a node that must be visited, including the depot.
        n = len(dist)
        best_tour = list(range(n))
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
            return {"tour": best_tour}
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

        # All constructions and perturbations share this one descent chain; no reference algorithm is chosen per instance.
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

        opening = min(deadline, started + limit * 0.25)
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
            tour, value = improve(best_tour[:], min(deadline, started + limit * 0.40), True)
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
            if roll < 0.48:
                candidate = bridge(source)
                if rng.random() < min(0.5, stagnant / 8):
                    candidate = bridge(candidate)
            elif roll < 0.93:
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
