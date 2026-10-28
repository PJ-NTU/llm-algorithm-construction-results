import itertools
import random
import time


def solve(instance: dict, time_limit_s: float) -> dict:
    """Anytime iterated local search for metric/symmetric TSP instances."""
    t0 = time.time()
    try:
        limit = max(0.0, float(time_limit_s))
        deadline = t0 + limit * 0.85
        n = int(instance["num_nodes"])
        dist = instance["distance_matrix"]
    except Exception:
        return {"tour": []}

    if n <= 0:
        return {"tour": []}
    if n == 1:
        return {"tour": [0]}

    # This is deliberately available before any expensive setup: every later
    # early exit and every exception can still return a complete permutation.
    best_tour = list(range(n))

    def tour_cost(tour):
        total = 0.0
        previous = tour[-1]
        for node in tour:
            total += dist[previous][node]
            previous = node
        return total

    try:
        best_cost = tour_cost(best_tour)
    except Exception:
        return {"tour": best_tour}

    def normalized(tour):
        try:
            p = tour.index(0)
            return tour[p:] + tour[:p]
        except Exception:
            return best_tour[:]

    if time.time() >= deadline:
        return {"tour": best_tour}

    rng = random.Random(1784892296 ^ (n * 1000003))
    eps = 1e-10

    try:
        # Exhaustion is preferable to spending the whole allowance after an
        # optimum has been proved on a tiny instance.
        if n <= 10:
            completed = True
            for tail in itertools.permutations(range(1, n)):
                if time.time() >= deadline:
                    completed = False
                    break
                candidate = [0] + list(tail)
                value = tour_cost(candidate)
                if value < best_cost - eps:
                    best_tour, best_cost = candidate, value
            if completed:
                return {"tour": best_tour}

        # Detect whether reversal deltas are valid.  The expensive symmetric
        # neighborhoods are disabled rather than mis-scored on directed data.
        symmetric = True
        for i in range(n):
            if time.time() >= deadline:
                return {"tour": normalized(best_tour)}
            for j in range(i):
                a = dist[i][j]
                b = dist[j][i]
                if abs(a - b) > 1e-9 * (1.0 + abs(a) + abs(b)):
                    symmetric = False
                    break
            if not symmetric:
                break

        # Candidate edges make repeated local optimization much cheaper than
        # the incumbents' all-pairs scans.  Full 2-opt is still used as a final
        # certification pass on promising tours.
        candidate_count = min(n - 1, 28 if n <= 500 else 20)
        near = []
        for i in range(n):
            if time.time() >= deadline:
                return {"tour": normalized(best_tour)}
            order = [j for j in range(n) if j != i]
            order.sort(key=lambda node, i=i: dist[i][node])
            near.append(order[:candidate_count])

        if symmetric:
            incoming = near
        else:
            incoming = []
            for target in range(n):
                if time.time() >= deadline:
                    return {"tour": normalized(best_tour)}
                order = [node for node in range(n) if node != target]
                order.sort(key=lambda node, target=target: dist[node][target])
                incoming.append(order[:candidate_count])

        def nearest_neighbor(start, choices, end_time):
            tour = [start]
            unused = set(range(n))
            unused.remove(start)
            current = start
            while unused:
                if time.time() >= end_time:
                    return None
                shortlist = []
                for node in near[current]:
                    if node in unused:
                        shortlist.append(node)
                        if len(shortlist) >= choices:
                            break
                if not shortlist:
                    node = min(unused, key=lambda candidate: dist[current][candidate])
                elif choices == 1:
                    node = shortlist[0]
                else:
                    node = shortlist[rng.randrange(len(shortlist))]
                tour.append(node)
                unused.remove(node)
                current = node
            return tour

        def farthest_insertion(end_time):
            far = max(range(1, n), key=lambda node: dist[0][node])
            cycle = [0, far]
            unused = set(range(1, n))
            unused.remove(far)
            proximity = {
                node: min(dist[node][0], dist[node][far]) for node in unused
            }
            while unused:
                if time.time() >= end_time:
                    return None
                node = max(unused, key=lambda candidate: proximity[candidate])
                best_pos = 0
                best_delta = float("inf")
                for p in range(len(cycle)):
                    if (p & 63) == 0 and time.time() >= end_time:
                        return None
                    a = cycle[p]
                    b = cycle[(p + 1) % len(cycle)]
                    delta = dist[a][node] + dist[node][b] - dist[a][b]
                    if delta < best_delta:
                        best_delta, best_pos = delta, p + 1
                cycle.insert(best_pos, node)
                unused.remove(node)
                proximity.pop(node)
                for other in unused:
                    value = dist[other][node]
                    if value < proximity[other]:
                        proximity[other] = value
            return cycle

        def candidate_two_opt(tour, end_time):
            if not symmetric or n < 4:
                return tour
            position = [0] * n
            for p, node in enumerate(tour):
                position[node] = p
            while time.time() < end_time:
                changed = False
                for edge1 in range(n):
                    if time.time() >= end_time:
                        return tour
                    a = tour[edge1]
                    for c in near[a]:
                        edge2 = position[c]
                        lo, hi = (edge1, edge2) if edge1 < edge2 else (edge2, edge1)
                        if hi == lo + 1 or (lo == 0 and hi == n - 1):
                            continue
                        aa = tour[lo]
                        bb = tour[lo + 1]
                        cc = tour[hi]
                        dd = tour[(hi + 1) % n]
                        delta = dist[aa][cc] + dist[bb][dd] - dist[aa][bb] - dist[cc][dd]
                        if delta < -eps:
                            tour[lo + 1:hi + 1] = reversed(tour[lo + 1:hi + 1])
                            for p in range(lo + 1, hi + 1):
                                position[tour[p]] = p
                            changed = True
                            break
                    if changed:
                        break
                if not changed:
                    break
            return tour

        def full_two_opt(tour, end_time):
            if not symmetric or n < 4:
                return tour
            while time.time() < end_time:
                changed = False
                for i in range(n - 2):
                    a = tour[i]
                    b = tour[i + 1]
                    for j in range(i + 2, n):
                        if (j & 63) == 0 and time.time() >= end_time:
                            return tour
                        if i == 0 and j == n - 1:
                            continue
                        c = tour[j]
                        d = tour[(j + 1) % n]
                        delta = dist[a][c] + dist[b][d] - dist[a][b] - dist[c][d]
                        if delta < -eps:
                            tour[i + 1:j + 1] = reversed(tour[i + 1:j + 1])
                            changed = True
                            break
                    if changed:
                        break
                if not changed:
                    break
            return tour

        def or_opt(tour, end_time):
            """Relocate strings of 1, 2, or 3 nodes; valid for directed data too."""
            position = [0] * n
            for p, node in enumerate(tour):
                position[node] = p
            while time.time() < end_time:
                changed = False
                for length in (1, 2, 3):
                    if length >= n:
                        continue
                    for i in range(1, n - length + 1):
                        if time.time() >= end_time:
                            return tour
                        first = tour[i]
                        last = tour[i + length - 1]
                        before = tour[i - 1]
                        after = tour[(i + length) % n]
                        for anchor in incoming[first]:
                            j = position[anchor]
                            if i - 1 <= j < i + length:
                                continue
                            following = tour[(j + 1) % n]
                            old = dist[before][first] + dist[last][after] + dist[anchor][following]
                            new = dist[before][after] + dist[anchor][first] + dist[last][following]
                            if new < old - eps:
                                segment = tour[i:i + length]
                                remainder = tour[:i] + tour[i + length:]
                                anchor_pos = remainder.index(anchor)
                                tour[:] = (remainder[:anchor_pos + 1] + segment +
                                           remainder[anchor_pos + 1:])
                                for p, node in enumerate(tour):
                                    position[node] = p
                                changed = True
                                break
                        if changed:
                            break
                    if changed:
                        break
                if not changed:
                    break
            return tour

        def swap_opt(tour, end_time):
            """Exchange two nodes, catching improvements needing two relocations."""
            position = [0] * n
            for p, node in enumerate(tour):
                position[node] = p
            while time.time() < end_time:
                changed = False
                for i in range(1, n):
                    if time.time() >= end_time:
                        return tour
                    previous = tour[i - 1]
                    # A good replacement at i normally has a cheap incoming
                    # edge from its new predecessor.
                    for replacement in near[previous]:
                        j = position[replacement]
                        if j == 0 or j == i:
                            continue
                        lo, hi = (i, j) if i < j else (j, i)
                        left = tour[lo]
                        right = tour[hi]
                        if hi == lo + 1:
                            p = tour[lo - 1]
                            q = tour[(hi + 1) % n]
                            old = dist[p][left] + dist[left][right] + dist[right][q]
                            new = dist[p][right] + dist[right][left] + dist[left][q]
                        else:
                            p = tour[lo - 1]
                            q = tour[lo + 1]
                            r = tour[hi - 1]
                            s = tour[(hi + 1) % n]
                            old = (dist[p][left] + dist[left][q] +
                                   dist[r][right] + dist[right][s])
                            new = (dist[p][right] + dist[right][q] +
                                   dist[r][left] + dist[left][s])
                        if new < old - eps:
                            tour[lo], tour[hi] = tour[hi], tour[lo]
                            position[tour[lo]] = lo
                            position[tour[hi]] = hi
                            changed = True
                            break
                    if changed:
                        break
                if not changed:
                    break
            return tour

        def three_opt(tour, end_time):
            """Candidate-restricted genuine 3-edge exchanges for symmetric TSP."""
            if not symmetric or n < 6:
                return tour
            position = [0] * n
            for p, node in enumerate(tour):
                position[node] = p
            width = min(12, candidate_count)
            while time.time() < end_time:
                changed = False
                for i in range(n - 4):
                    if time.time() >= end_time:
                        return tour
                    a = tour[i]
                    b = tour[i + 1]
                    for c in near[a][:width]:
                        j = position[c]
                        if j <= i + 1 or j >= n - 2:
                            continue
                        d = tour[j + 1]
                        for e in near[b][:width]:
                            k = position[e]
                            if k <= j + 1 or (i == 0 and k == n - 1):
                                continue
                            f = tour[(k + 1) % n]
                            old = dist[a][b] + dist[c][d] + dist[e][f]
                            # All four options below change three edges; simple
                            # segment reversals have already been exhausted by
                            # 2-opt and are intentionally omitted.
                            alternatives = (
                                dist[a][c] + dist[b][e] + dist[d][f],
                                dist[a][d] + dist[e][b] + dist[c][f],
                                dist[a][e] + dist[d][b] + dist[c][f],
                                dist[a][d] + dist[e][c] + dist[b][f],
                            )
                            option = min(range(4), key=lambda z: alternatives[z])
                            if alternatives[option] < old - eps:
                                prefix = tour[:i + 1]
                                middle1 = tour[i + 1:j + 1]
                                middle2 = tour[j + 1:k + 1]
                                suffix = tour[k + 1:]
                                if option == 0:
                                    tour[:] = prefix + middle1[::-1] + middle2[::-1] + suffix
                                elif option == 1:
                                    tour[:] = prefix + middle2 + middle1 + suffix
                                elif option == 2:
                                    tour[:] = prefix + middle2[::-1] + middle1 + suffix
                                else:
                                    tour[:] = prefix + middle2 + middle1[::-1] + suffix
                                for p, node in enumerate(tour):
                                    position[node] = p
                                changed = True
                                break
                        if changed:
                            break
                    if changed:
                        break
                if not changed:
                    break
            return tour

        def improve(tour, end_time, certify):
            previous = tour_cost(tour)
            while time.time() < end_time:
                candidate_two_opt(tour, end_time)
                or_opt(tour, end_time)
                swap_opt(tour, end_time)
                value = tour_cost(tour)
                if value >= previous - eps:
                    break
                previous = value
            if certify and time.time() < end_time:
                while time.time() < end_time:
                    certified_before = tour_cost(tour)
                    full_two_opt(tour, end_time)
                    or_opt(tour, end_time)
                    swap_opt(tour, end_time)
                    candidate_two_opt(tour, end_time)
                    three_opt(tour, end_time)
                    if tour_cost(tour) >= certified_before - eps:
                        break
            return tour, tour_cost(tour)

        def consider(tour, end_time, certify):
            nonlocal best_tour, best_cost
            if tour is None:
                return
            tour, value = improve(tour, end_time, certify)
            if value < best_cost - eps:
                best_tour, best_cost = tour[:], value

        # Diverse constructions occupy only the first part of the allowance.
        setup_end = min(deadline, t0 + limit * 0.30)
        starts = [0]
        farthest_from_zero = max(range(1, n), key=lambda node: dist[0][node])
        if farthest_from_zero not in starts:
            starts.append(farthest_from_zero)
        starts.append(rng.randrange(n))
        for start in starts:
            if time.time() >= setup_end:
                break
            consider(nearest_neighbor(start, 1, setup_end), setup_end, True)
        if time.time() < setup_end:
            consider(farthest_insertion(setup_end), setup_end, True)
        while time.time() < setup_end:
            choices = 2 + rng.randrange(min(5, n - 1))
            start = rng.randrange(n)
            consider(nearest_neighbor(start, choices, setup_end), setup_end, False)

        def double_bridge(tour):
            if n < 8:
                result = tour[:]
                if n > 2:
                    i, j = sorted(rng.sample(range(1, n), 2))
                    result[i:j + 1] = reversed(result[i:j + 1])
                return result
            a, b, c = sorted(rng.sample(range(1, n), 3))
            return tour[:a] + tour[b:c] + tour[a:b] + tour[c:]

        def ruin_and_recreate(tour, strength, end_time):
            removable = list(range(1, n))
            strength = min(strength, len(removable))
            removed_positions = set(rng.sample(removable, strength))
            removed = [tour[p] for p in removed_positions]
            rng.shuffle(removed)
            partial = [node for p, node in enumerate(tour) if p not in removed_positions]
            for r, node in enumerate(removed):
                if time.time() >= end_time:
                    partial.extend(removed[r:])
                    break
                best_pos = 0
                best_delta = float("inf")
                m = len(partial)
                for p in range(m):
                    a = partial[p]
                    b = partial[(p + 1) % m]
                    delta = dist[a][node] + dist[node][b] - dist[a][b]
                    if delta < best_delta:
                        best_delta, best_pos = delta, p + 1
                partial.insert(best_pos, node)
            return partial

        current_tour = best_tour[:]
        current_cost = best_cost
        last_best_time = time.time()

        # The only outer search bound is wall time.  Stagnation increases the
        # destroy size, so unused time becomes progressively broader search.
        while time.time() < deadline:
            now = time.time()
            stagnant = max(0.0, now - last_best_time)
            scale = stagnant / max(0.05, limit * 0.12)
            source = best_tour if rng.random() < 0.72 else current_tour
            roll = rng.random()
            if roll < 0.46:
                candidate = double_bridge(source[:])
            elif roll < 0.82:
                strength = 3 + int(min(12.0, scale * 3.0))
                candidate = ruin_and_recreate(source, strength, deadline)
            else:
                candidate = nearest_neighbor(rng.randrange(n), 2 + rng.randrange(min(6, n - 1)), deadline)
                if candidate is None:
                    break

            candidate, value = improve(candidate, deadline, False)
            if value < best_cost - eps:
                # Spend the all-pairs pass only when there is a real chance to
                # replace the incumbent best.
                candidate, value = improve(candidate, deadline, True)
                if value < best_cost - eps:
                    best_tour, best_cost = candidate[:], value
                    last_best_time = time.time()
                    current_tour, current_cost = candidate[:], value
                    continue

            # Mild threshold acceptance keeps an alternative basin alive while
            # retaining a strong bias toward the incumbent basin.
            tolerance = 0.0015 + min(0.008, scale * 0.001)
            if value < current_cost or value <= best_cost * (1.0 + tolerance):
                current_tour, current_cost = candidate, value
            else:
                current_tour, current_cost = best_tour[:], best_cost

    except Exception:
        # The input contract is trusted, but a heuristic must never turn an
        # unusual numerical case into a missing answer.
        return {"tour": normalized(best_tour)}

    return {"tour": normalized(best_tour)}
