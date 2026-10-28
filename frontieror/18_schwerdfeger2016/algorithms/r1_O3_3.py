"""Annealing-guided multi-machine reconstruction with group balancing
Fuses randomised longest-processing-time-first construction, move/swap descent, two-machine subset-sum reassignment and wall-clock annealing acceptance, continually saving the best feasible assignment.
"""

import bisect
import heapq
import math
import random
import time


def solve(instance: dict, time_limit_s: float) -> dict:
    start = time.perf_counter()
    budget = max(0.0, float(time_limit_s))
    m = int(instance["num_machines"])
    n = int(instance["num_jobs"])
    p = [int(v) for v in instance["processing_times"]]

    # First build a complete backup solution; no search step modifies this saved assignment.
    best = [[] for _ in range(m)]
    for j in range(n):
        best[j % m].append(j)

    def result():
        return {
            "assignment": {str(i): best[i] for i in range(m)},
            "machine_completion_times": {
                str(i): float(sum(p[j] for j in best[i])) for i in range(m)
            },
        }

    if n == 0 or m <= 1 or n <= m:
        return result()
    setup = time.perf_counter() - start
    reserve = min(budget * 0.12, 0.003 + budget * 0.02 + setup)
    deadline = start + budget - reserve
    if time.perf_counter() >= deadline:
        return result()

    try:
        unit = 0
        for value in p:
            unit = math.gcd(unit, value)
        if unit == 0:
            return result()
        w = [value // unit for value in p]
        total = sum(w)
        best_score = sum(sum(w[j] for j in b) ** 2 for b in best)
        order = sorted(range(n), key=w.__getitem__, reverse=True)

        # The large-job constraint and the integer equal split together give a lower bound on the sum of squares.
        remaining, machines, bound = total, m, 0
        for j in order:
            if machines <= 1 or w[j] * machines <= remaining:
                break
            bound += w[j] * w[j]
            remaining -= w[j]
            machines -= 1
        level, extra = divmod(remaining, machines)
        lower_bound = bound + (machines - extra) * level * level
        lower_bound += extra * (level + 1) * (level + 1)

        groups = [[] for _ in range(m)]
        loads = [0] * m
        heap = [(0, i) for i in range(m)]
        for k, j in enumerate(order):
            if k % 128 == 0 and time.perf_counter() >= deadline:
                return result()
            load, i = heapq.heappop(heap)
            groups[i].append(j)
            loads[i] = load + w[j]
            heapq.heappush(heap, (loads[i], i))
        score = sum(v * v for v in loads)
        if score < best_score:
            best_score = score
            best = [b[:] for b in groups]
        if best_score == lower_bound:
            return result()

        rng = random.Random((n * 1000003 + m * 9176 + total) & 0xFFFFFFFF)
        effort = math.log1p(budget)
        mean_count = n / m
        usable = max(0.000001, deadline - time.perf_counter())
        quantum = usable / (6.0 + math.sqrt(n + m) * (1.0 + effort))
        epoch = usable / (2.0 + math.sqrt(1.0 + budget))
        bit_cap = int(50000 * (1.0 + math.sqrt(budget)))
        history_cap = int(8000000 * (1.0 + effort))
        mitm_cap = int(2 * math.log2(2.0 + 1000.0 * math.sqrt(budget)))
        width_cap = min(m, max(2, 2 + int(math.sqrt(effort * (1 + mean_count)))))
        shake_cap = max(1, int((1.0 + effort) * math.sqrt(mean_count)))
        mean_square = sum(v * v for v in w) / n
        search_start = time.perf_counter()
        last_best = search_start
        next_restart = search_start + epoch

        def pair_candidate(a, b, la, lb, end):
            """First do the best single move or swap, then try a complete two-machine reassignment."""
            if la < lb:
                answer = pair_candidate(b, a, lb, la, end)
                if answer is None:
                    return None
                x, y, lx, ly = answer
                return y, x, ly, lx
            difference = la - lb
            if difference <= 1 or time.perf_counter() >= end:
                return None
            gain = 0
            move = None
            for ia, ja in enumerate(a):
                d = w[ja]
                improvement = d * (difference - d)
                if improvement > gain:
                    gain, move = improvement, (ia, -1)
                if ia % 128 == 0 and time.perf_counter() >= end:
                    return None
            if b:
                sorted_b = sorted((w[j], ib) for ib, j in enumerate(b))
                values = [v for v, _ in sorted_b]
                for ia, ja in enumerate(a):
                    target = w[ja] - difference / 2.0
                    pos = bisect.bisect_left(values, target)
                    for ib_sorted in (pos - 1, pos):
                        if 0 <= ib_sorted < len(values):
                            d = w[ja] - values[ib_sorted]
                            improvement = d * (difference - d)
                            if improvement > gain:
                                gain = improvement
                                move = (ia, sorted_b[ib_sorted][1])
                    if ia % 64 == 0 and time.perf_counter() >= end:
                        return None
            if move is not None:
                ia, ib = move
                x, y = a[:], b[:]
                if ib < 0:
                    ja = x.pop(ia)
                    y.append(ja)
                    d = w[ja]
                else:
                    d = w[x[ia]] - w[y[ib]]
                    x[ia], y[ib] = y[ib], x[ia]
                return x, y, la - d, lb + d

            pool = a + b
            divisor = 0
            for j in pool:
                divisor = math.gcd(divisor, w[j])
            if divisor == 0 or time.perf_counter() >= end:
                return None
            half = (la + lb) // (2 * divisor)
            chosen = None
            count = len(pool)
            if half <= bit_cap and (half + 1) * (count + 1) <= history_cap:
                rng.shuffle(pool)
                bits = 1
                mask = (1 << (half + 1)) - 1
                history = []
                for j in pool:
                    if time.perf_counter() >= end:
                        return None
                    history.append(bits)
                    value = w[j] // divisor
                    if value <= half:
                        bits |= (bits << value) & mask
                    if (bits >> half) & 1:
                        break
                found = bits.bit_length() - 1
                new_la = found * divisor
                if abs(2 * new_la - la - lb) >= difference:
                    return None
                chosen = set()
                for k in range(len(history) - 1, -1, -1):
                    if time.perf_counter() >= end:
                        return None
                    if not ((history[k] >> found) & 1):
                        chosen.add(pool[k])
                        found -= w[pool[k]] // divisor
            elif count <= mitm_cap:
                middle = count // 2
                left = [(0, 0)]
                right = [(0, 0)]
                for side, jobs in ((left, pool[:middle]), (right, pool[middle:])):
                    for k, j in enumerate(jobs):
                        if time.perf_counter() >= end:
                            return None
                        value = w[j] // divisor
                        flag = 1 << k
                        length = len(side)
                        for pos in range(length):
                            if pos % 256 == 0 and time.perf_counter() >= end:
                                return None
                            value0, bits0 = side[pos]
                            if value0 + value <= half:
                                side.append((value0 + value, bits0 | flag))
                right.sort()
                values = [v for v, _ in right]
                found, masks = -1, (0, 0)
                for k, (value, flags) in enumerate(left):
                    if k % 128 == 0 and time.perf_counter() >= end:
                        return None
                    pos = bisect.bisect_right(values, half - value) - 1
                    if pos >= 0 and value + right[pos][0] > found:
                        found = value + right[pos][0]
                        masks = flags, right[pos][1]
                        if found == half:
                            break
                new_la = found * divisor
                if abs(2 * new_la - la - lb) >= difference:
                    return None
                chosen = {
                    j for k, j in enumerate(pool[:middle]) if masks[0] & (1 << k)
                }
                chosen.update(
                    j for k, j in enumerate(pool[middle:]) if masks[1] & (1 << k)
                )
            if chosen is None:
                return None
            x = [j for j in pool if j in chosen]
            y = [j for j in pool if j not in chosen]
            lx = sum(w[j] for j in x)
            return x, y, lx, la + lb - lx

        def polish(buckets, weights, end):
            # Descent has no fixed iteration cap; it stops at a local optimum or when the wall clock expires.
            while time.perf_counter() < end:
                ranked = sorted(range(len(weights)), key=weights.__getitem__)
                improved = False
                for high in range(len(ranked) - 1, 0, -1):
                    i = ranked[high]
                    for low in range(high):
                        if time.perf_counter() >= end:
                            return
                        j = ranked[low]
                        answer = pair_candidate(
                            buckets[i], buckets[j], weights[i], weights[j], end
                        )
                        if answer is not None:
                            buckets[i], buckets[j], weights[i], weights[j] = answer
                            improved = True
                            break
                    if improved:
                        break
                if not improved:
                    return

        polish(groups, loads, min(deadline, time.perf_counter() + quantum))
        score = sum(v * v for v in loads)
        if score < best_score:
            best_score = score
            best = [b[:] for b in groups]
            last_best = time.perf_counter()

        # Descent solutions after random reconstruction/perturbation are accepted by the same annealing criterion, rather than switching between two solvers.
        while time.perf_counter() < deadline and best_score > lower_bound:
            now = time.perf_counter()
            if now >= next_restart:
                groups = [b[:] for b in best]
                loads = [sum(w[j] for j in b) for b in groups]
                score = best_score
                next_restart = now + epoch
            phase = ((now - search_start) % epoch) / epoch
            progress = min(1.0, (now - search_start) / usable)
            temperature = max(
                1.0,
                (best_score - lower_bound) / m + mean_square / (1.0 + mean_count),
            )
            temperature *= math.exp(-9.0 * phase) * (1.0 - 0.8 * progress)
            stale = min(1.0, (now - last_best) / epoch)
            width = rng.randint(2, min(m, max(2, int(width_cap * (1.0 + stale)))))
            ids = rng.sample(range(m), width)
            if rng.random() < 0.65:
                heavy = max(range(m), key=loads.__getitem__)
                light = min(range(m), key=loads.__getitem__)
                selected = {heavy, light}
                ids = list(selected) + [i for i in ids if i not in selected]
                ids = ids[:width]
            trial = [groups[i][:] for i in ids]
            trial_loads = [loads[i] for i in ids]
            old_part = sum(v * v for v in trial_loads)
            end = min(deadline, now + quantum)

            if rng.random() < 0.5:
                # Randomised longest-processing-time-first reconstruction is used inside the same local search.
                jobs = [j for b in trial for j in b]
                noise = (0.05 + 0.65 * phase + stale) * effort / (1.0 + effort)
                jobs.sort(key=lambda j: w[j] * (1.0 + noise * rng.random()), reverse=True)
                rebuilt = [[] for _ in ids]
                new_loads = [0] * len(ids)
                heap = [(0, k) for k in range(len(ids))]
                complete = True
                for k, j in enumerate(jobs):
                    if k % 64 == 0 and time.perf_counter() >= end:
                        complete = False
                        break
                    load, dest = heapq.heappop(heap)
                    rebuilt[dest].append(j)
                    new_loads[dest] = load + w[j]
                    heapq.heappush(heap, (new_loads[dest], dest))
                if complete:
                    trial, trial_loads = rebuilt, new_loads
            else:
                strength = rng.randint(1, max(1, int(shake_cap * (1.0 + stale))))
                # The perturbation strength is determined by instance density, time budget and stagnation length.
                for _ in range(strength):
                    if time.perf_counter() >= end:
                        break
                    a, b = rng.sample(range(len(ids)), 2)
                    if not trial[a]:
                        continue
                    ia = rng.randrange(len(trial[a]))
                    ja = trial[a][ia]
                    if trial[b] and rng.random() < 0.7:
                        ib = rng.randrange(len(trial[b]))
                        jb = trial[b][ib]
                        trial[a][ia], trial[b][ib] = jb, ja
                        d = w[ja] - w[jb]
                    else:
                        trial[a][ia] = trial[a][-1]
                        trial[a].pop()
                        trial[b].append(ja)
                        d = w[ja]
                    trial_loads[a] -= d
                    trial_loads[b] += d

            polish(trial, trial_loads, end)
            delta = sum(v * v for v in trial_loads) - old_part
            accepted = delta <= 0
            if not accepted and temperature > 0.0:
                # The logarithmic form avoids exponential underflow; the best solution never deteriorates through annealing.
                accepted = delta < -temperature * math.log(max(rng.random(), 1e-300))
            if accepted:
                for k, i in enumerate(ids):
                    groups[i], loads[i] = trial[k], trial_loads[k]
                score += delta
                if score < best_score:
                    best_score = score
                    best = [b[:] for b in groups]
                    last_best = time.perf_counter()
    except Exception:
        # Search exceptions do not affect the already saved complete feasible solution.
        pass
    return result()
