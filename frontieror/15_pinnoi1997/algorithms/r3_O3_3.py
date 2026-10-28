"""Bottleneck-chain-guided bidirectional closure rebuilding
Integrates shift-and-swap repair chains into grouped beam search and closure subset-sum reassignment, rebuilds neighbourhoods with time-limited integer programming, and continuously keeps the best verified feasible solution.
"""
import bisect
import math
import random
import time

try:
    import numpy as np
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import coo_matrix
except Exception:
    np = None


def _orientation(data, reverse):
    p, m, C, e, l, pred, succ, order, guide = data
    if reverse:
        e, l = [m - 1 - x for x in l], [m - 1 - x for x in e]
        pred, succ, order = succ, pred, order[::-1]
        guide = None if guide is None else [m - 1 - x for x in guide]
    n = len(p)
    pm = [sum(1 << j for j in pred[i]) for i in range(n)]
    tail = p[:]
    for i in reversed(order):
        if succ[i]:
            tail[i] += max(tail[j] for j in succ[i])
    released, due, guided = [0] * m, [0] * m, [0] * m
    for i in range(n):
        released[e[i]] |= 1 << i
        due[l[i]] |= 1 << i
        if guide is not None:
            guided[guide[i]] |= 1 << i
    for s in range(1, m):
        released[s] |= released[s - 1]
        due[s] |= due[s - 1]
        guided[s] |= guided[s - 1]
    return e, l, pred, succ, order, pm, tail, released, due, guided, guide


def _beam(data, target, reverse, width, local_width, branches, salt, rng, end, emit):
    p, m, C = data[:3]
    n, total = len(p), sum(p)
    e, l, pred, succ, order, pm, tail, eligible, due, guided, guide = _orientation(data, reverse)
    full = (1 << n) - 1
    clock = time.perf_counter
    cache = {}

    def boundary(mask, work, station):
        key = (mask, station)
        if key in cache:
            return cache[key]
        count = m - station
        remaining = total - work
        if count == 0:
            return 0.0 if mask == full else None
        if not count * target <= remaining <= count * C:
            cache[key] = None
            return None
        release, finish = [0] * count, [0] * count
        bits = full ^ mask
        while bits:
            if clock() >= end:
                return None
            bit = bits & -bits
            bits ^= bit
            i = bit.bit_length() - 1
            if l[i] < station:
                cache[key] = None
                return None
            release[max(0, e[i] - station)] += p[i]
            finish[l[i] - station] += p[i]
        risk, forced, usable = 0.0, 0, 0
        for k in range(count):
            forced += finish[k]
            usable += release[k]
            cap, need = (k + 1) * C, (k + 1) * target
            if forced > cap or usable < need:
                cache[key] = None
                return None
            risk = max(risk, forced / max(1, cap), need / max(1, usable))
        forced, usable = 0, 0
        for k in range(count - 1, -1, -1):
            forced += release[k]
            usable += finish[k]
            cap, need = (count - k) * C, (count - k) * target
            if forced > cap or usable < need:
                cache[key] = None
                return None
            risk = max(risk, forced / max(1, cap), need / max(1, usable))
        cache[key] = risk
        return risk

    def groups(mask, work, station):
        remaining = total - work
        later = m - station - 1
        low, high = max(target, remaining - later * C), min(C, remaining - later * target)
        if low > high:
            return []
        unassigned = full ^ mask
        if later == 0:
            valid = not (unassigned & ~eligible[station]) and not ((due[station - 1] if station else 0) & unassigned)
            return [(full, remaining, 0.0, 0.0)] if valid and low <= remaining <= high else []
        # Propagated deadline stations make the set of due tasks closed under predecessors, so it can be added in one go.
        mandatory = due[station] & unassigned
        if mandatory & ~eligible[station]:
            return []
        base, base_load = mask | mandatory, 0
        bits = mandatory
        while bits:
            bit = bits & -bits
            bits ^= bit
            base_load += p[bit.bit_length() - 1]
        if base_load > high:
            return []
        ready, bits = 0, full ^ base
        while bits:
            bit = bits & -bits
            bits ^= bit
            i = bit.bit_length() - 1
            if pm[i] & ~base == 0:
                ready |= bit
        desired = min(high, max(low, remaining / (later + 1)))
        found = {}

        def add(current, load, penalty):
            if not low <= load <= high:
                return
            risk = boundary(current, work + load, station + 1)
            if risk is not None:
                score = 4.0 * risk + abs(load - desired) / (C + 1) + 0.12 * penalty / (n + 1)
                old = found.get(current)
                if old is None or score < old[0]:
                    found[current] = (score, current, load, risk)

        guide_mask = None
        if guide is not None:
            guide_mask = base | guided[station]
            bits, load = guide_mask ^ mask, 0
            while bits:
                bit = bits & -bits
                bits ^= bit
                load += p[bit.bit_length() - 1]
            if not ((guide_mask ^ mask) & ~eligible[station]):
                add(guide_mask, load, 0)
        frontier = [(0.0, base, base_load, ready, 0)]
        depth = 0
        while frontier and clock() < end:
            next_by_mask = {}
            for _, current, load, available, penalty in frontier:
                if clock() >= end:
                    break
                add(current, load, penalty)
                candidates, bits = [], available & eligible[station]
                variant = (salt + station + depth) % 7
                while bits:
                    if clock() >= end:
                        break
                    bit = bits & -bits
                    bits ^= bit
                    i = bit.bit_length() - 1
                    if load + p[i] > high:
                        continue
                    bad = int(guide is not None and guide[i] != station)
                    fit = abs(desired - load - p[i])
                    noise = rng.random()
                    if variant == 0:
                        key = (l[i], bad, -tail[i], fit, noise)
                    elif variant == 1:
                        key = (l[i], -p[i], bad, noise)
                    elif variant == 2:
                        key = (l[i] - e[i], l[i], -tail[i], noise)
                    elif variant == 3:
                        key = (bad, l[i], fit, noise)
                    elif variant == 4:
                        key = (fit, l[i], -tail[i], noise)
                    elif variant == 5:
                        key = (l[i], noise, -tail[i])
                    else:
                        key = (-tail[i], l[i], noise)
                    candidates.append((key, i, bit, bad))
                candidates.sort()
                chosen = candidates[:branches]
                if len(candidates) > branches and branches > 1:
                    chosen[-1] = candidates[rng.randrange(branches, len(candidates))]
                for _, i, bit, bad in chosen:
                    current2, load2 = current | bit, load + p[i]
                    available2 = available ^ bit
                    for j in succ[i]:
                        if not (current2 >> j) & 1 and pm[j] & ~current2 == 0:
                            available2 |= 1 << j
                    penalty2 = penalty + bad
                    score = abs(load2 - desired) / (C + 1) + 0.10 * penalty2 / (n + 1) + rng.random() * 0.035
                    state = (score, current2, load2, available2, penalty2)
                    old = next_by_mask.get(current2)
                    if old is None or score < old[0]:
                        next_by_mask[current2] = state
            frontier = sorted(next_by_mask.values())[:local_width]
            depth += 1
        packed = sorted(found.values())
        if len(packed) > branches:
            selected, used, loads_used = [], set(), set()
            if guide_mask in found:
                item = found[guide_mask]
                selected.append(item)
                used.add(item[1])
                loads_used.add(item[2])
            # Keep candidates with different loads and different task compositions, so local candidates do not all crowd at the same load.
            for item in packed:
                if len(selected) >= max(1, (branches * 2) // 3):
                    break
                if item[1] not in used and (item[2] not in loads_used or not selected):
                    selected.append(item)
                    used.add(item[1])
                    loads_used.add(item[2])
            for item in packed:
                if len(selected) >= branches:
                    break
                if item[1] not in used:
                    selected.append(item)
                    used.add(item[1])
            packed = selected
        return [(x[1], x[2], x[3], x[0]) for x in packed]

    if boundary(0, 0, 0) is None:
        return False
    states = [(0, 0, (), ())]
    improved = False
    for station in range(m):
        if clock() >= end:
            break
        following = {}
        for mask, work, blocks, loads in states:
            if clock() >= end:
                break
            for next_mask, load, risk, quality in groups(mask, work, station):
                blocks2 = blocks + (next_mask ^ mask,)
                loads2 = tuple(sorted(loads + (load,)))
                work2 = work + load
                if station == m - 1:
                    assignment = [0] * n
                    for s, bits in enumerate(blocks2):
                        actual = m - 1 - s if reverse else s
                        while bits:
                            bit = bits & -bits
                            bits ^= bit
                            assignment[bit.bit_length() - 1] = actual
                    if emit(assignment):
                        return True
                    continue
                deviation = abs(work2 - total * (station + 1) / m) / (C + 1)
                mean = work2 / (station + 1)
                spread = sum(abs(x - mean) for x in loads2) / ((station + 1) * (C + 1))
                score = 4.5 * risk + deviation + 0.18 * spread + 0.08 * quality + rng.random() * 0.025
                state = (score, work2, blocks2, loads2)
                old = following.get(next_mask)
                # Identical remaining tasks define identical future subproblems; keep the one whose completed loads are lexicographically dominant.
                if old is None or loads2 > old[3] or (loads2 == old[3] and score < old[0]):
                    following[next_mask] = state
        if not following:
            break
        ranked = sorted(following.items(), key=lambda item: item[1][0])[:width]
        if guide is not None and guided[station] in following and all(k != guided[station] for k, _ in ranked):
            ranked.append((guided[station], following[guided[station]]))
        states = [(mask, item[1], item[2], item[3]) for mask, item in ranked]
    return improved


def _two_station(data, end, budget, diversify, emit):
    p, _, C, e, l, pred, succ, order, guide = data
    n, total = len(p), sum(p)
    full = (1 << n) - 1
    clock = time.perf_counter
    ancestors, descendants = [0] * n, [0] * n
    for i in order:
        bits = 1 << i
        for j in pred[i]:
            bits |= ancestors[j]
        ancestors[i] = bits
    for i in reversed(order):
        bits = 1 << i
        for j in succ[i]:
            bits |= descendants[j]
        descendants[i] = bits
    left, right = 0, 0
    for i in range(n):
        if l[i] == 0:
            left |= ancestors[i]
        if e[i] == 1:
            right |= descendants[i]
    if left & right:
        return False
    weights = {0: 0}

    def weight(bits):
        if bits in weights:
            return weights[bits]
        original, value = bits, 0
        while bits:
            bit = bits & -bits
            bits ^= bit
            value += p[bit.bit_length() - 1]
        weights[original] = value
        return value

    incumbent_load = sum(p[i] for i in range(n) if guide[i] == 0)
    threshold = min(incumbent_load, total - incumbent_load) + (0 if diversify else 1)
    improved = False

    def offer(bits):
        nonlocal threshold, improved
        value = weight(bits)
        quality = min(value, total - value)
        if quality < threshold or max(value, total - value) > C:
            return
        candidate = [0 if (bits >> i) & 1 else 1 for i in range(n)]
        improved = emit(candidate) or improved
        threshold = max(threshold, quality + 1)

    def independent_dp(free, fixed, fixed_work, low, high):
        items, g, bits = [], 0, free
        while bits:
            bit = bits & -bits
            bits ^= bit
            i = bit.bit_length() - 1
            items.append((i, bit))
            g = math.gcd(g, p[i])
        if not g:
            offer(fixed)
            return True
        low = max(0, (low - fixed_work + g - 1) // g)
        high = (high - fixed_work) // g
        if high < low:
            return True
        # Bitset capacity scales with the budget and the number of undecided tasks, avoiding memory blow-up from large processing times.
        bit_budget = int(2000000 * (1 + math.log1p(max(0.0, budget))))
        if (high + 1) * (len(items) + 1) > bit_budget:
            return False
        cap = (1 << (high + 1)) - 1
        reachable, history = 1, [1]
        for i, bit in items:
            if clock() >= end:
                return True
            units = p[i] // g
            if units <= high:
                reachable = (reachable | (reachable << units)) & cap
            history.append(reachable)
        legal = (reachable >> low) << low
        if not legal:
            return True
        middle = max(low, min(high, (total - 2 * fixed_work) // (2 * g)))
        below = legal & ((1 << (middle + 1)) - 1)
        above = legal >> (middle + 1)
        choices = []
        if below:
            choices.append(below.bit_length() - 1)
        if above:
            choices.append(middle + 1 + (above & -above).bit_length() - 1)
        target = min(choices, key=lambda x: abs(total - 2 * (fixed_work + x * g)))
        chosen, remaining = fixed, target
        for k in range(len(items) - 1, -1, -1):
            if not (history[k] >> remaining) & 1:
                i, bit = items[k]
                chosen |= bit
                remaining -= p[i] // g
        offer(chosen)
        return True

    stack = [(left, right, weight(left), weight(right))]
    while stack and clock() < end:
        left, right, wl, wr = stack.pop()
        low, high = max(total - C, threshold), min(C, total - threshold)
        if low > high:
            break
        if wl > high or total - wr < low:
            continue
        if low <= wl <= high:
            offer(left)
        if low <= total - wr <= high:
            offer(full ^ right)
        low, high = max(total - C, threshold), min(C, total - threshold)
        if low > high or wl > high or total - wr < low:
            continue
        free = full ^ (left | right)
        if not free:
            continue
        choice, key, independent, bits = -1, None, True, free
        while bits:
            if clock() >= end:
                return improved
            bit = bits & -bits
            bits ^= bit
            i = bit.bit_length() - 1
            a, d = ancestors[i] & free, descendants[i] & free
            impact = (a | d).bit_count()
            independent = independent and impact == 1
            score = (impact, p[i])
            if key is None or score > key:
                key, choice = score, i
        if independent and independent_dp(free, left, wl, low, high):
            continue
        add_left = ancestors[choice] & free
        add_right = descendants[choice] & free
        wa, wd = weight(add_left), weight(add_right)
        options = []
        if not ancestors[choice] & right and wl + wa <= high:
            options.append((left | add_left, right, wl + wa, wr))
        if not descendants[choice] & left and total - wr - wd >= low:
            options.append((left, right | add_right, wl, wr + wd))
        options.sort(key=lambda x: abs((total - x[2] - x[3]) / 2 + x[2] - total / 2), reverse=True)
        stack.extend(options)
    return improved


def _integer_rebuild(data, target, end, rng, emit):
    """Cumulative station binary-variable model; release bounds, deadline bounds and external precedence arcs are already folded into the local data."""
    if np is None or time.perf_counter() >= end:
        return False
    try:
        p, m, C, e, l, pred, succ, order, guide = data
        n = len(p)
        if not n:
            return False
        divisor = 0
        for value in p:
            divisor = math.gcd(divisor, value)
        divisor = max(1, divisor)
        offset, count = [], 0
        for i in range(n):
            offset.append(count)
            count += l[i] - e[i]
        if count == 0:
            return emit(e[:])
        upper = min(C, sum(p) / m) / divisor
        if target / divisor > upper:
            return False
        row, col, val, lower, higher = [], [], [], [], []
        station_terms = [[] for _ in range(m)]
        fixed = [0.0] * m
        objective = np.zeros(count + 1)
        objective[count] = -1.0
        epsilon = 0.12 / (count + 1)

        def add(terms, lo, hi):
            r = len(lower)
            for k, value in terms:
                row.append(r)
                col.append(k)
                val.append(value)
            lower.append(lo)
            higher.append(hi)

        for i in range(n):
            if time.perf_counter() >= end:
                return False
            weight = p[i] / divisor
            fixed[l[i]] += weight
            for s in range(e[i], l[i]):
                k = offset[i] + s - e[i]
                station_terms[s].append((k, weight))
                station_terms[s + 1].append((k, -weight))
                if s + 1 < l[i]:
                    add(((k, 1.0), (k + 1, -1.0)), -np.inf, 0.0)
                if guide is not None:
                    objective[k] = epsilon * (1.0 + rng.random()) * (1 if guide[i] > s else -1)
            for j in pred[i]:
                # y[j,s] >= y[i,s] is a strengthened linear description of the precedence relation.
                for s in range(e[i], l[j]):
                    add(((offset[i] + s - e[i], 1.0),
                         (offset[j] + s - e[j], -1.0)), -np.inf, 0.0)
        for s in range(m):
            add(station_terms[s], -np.inf, C / divisor - fixed[s])
            add(station_terms[s] + [(count, -1.0)], -fixed[s], np.inf)
        if time.perf_counter() >= end:
            return False
        matrix = coo_matrix((val, (row, col)), shape=(len(lower), count + 1)).tocsc()
        lb, ub = np.zeros(count + 1), np.ones(count + 1)
        lb[count], ub[count] = max(0.0, target / divisor), upper
        integral = np.ones(count + 1)
        integral[count] = 0
        remaining = end - time.perf_counter()
        if remaining <= 0:
            return False
        answer = milp(
            objective, integrality=integral, bounds=Bounds(lb, ub),
            constraints=LinearConstraint(matrix, np.asarray(lower), np.asarray(higher)),
            options={"time_limit": remaining * 0.88, "mip_rel_gap": 0.0, "disp": False},
        )
        if answer.x is None:
            return False
        assignment = l[:]
        for i in range(n):
            for s in range(e[i], l[i]):
                if answer.x[offset[i] + s - e[i]] > 0.5:
                    assignment[i] = s
                    break
        return emit(assignment)
    except Exception:
        return False


def solve(instance: dict, time_limit_s: float) -> dict:
    started = time.perf_counter()
    result = {"task_stations": [], "worst_case_idle_time": 0}
    try:
        n, m = int(instance["num_tasks"]), int(instance["num_stations"])
        C = int(instance["cycle_time"])
        p = [int(x) for x in instance["processing_times"]]
        E = [int(x) - 1 for x in instance["earliest_stations"]]
        L = [int(x) - 1 for x in instance["latest_stations"]]
        result = {"task_stations": [x + 1 for x in E], "worst_case_idle_time": max(0, C)}
        limit = float(time_limit_s)
        if not math.isfinite(limit):
            limit = 0.0
        limit = max(0.0, limit)
        clock = time.perf_counter
        reserve = min(limit * 0.06, 0.004 + limit * 0.012)
        deadline = started + max(0.0, limit - reserve)
        if n == 0:
            return {"task_stations": [], "worst_case_idle_time": C}
        if m <= 0 or C <= 0 or len(p) != n or len(E) != n or len(L) != n:
            return result
        if any(p[i] < 0 or p[i] > C or not 0 <= E[i] <= L[i] < m for i in range(n)):
            return result
        arcs = sorted(set((int(a[0]) - 1, int(a[1]) - 1) for a in instance["precedence_arcs"]))
        pred, succ = [[] for _ in range(n)], [[] for _ in range(n)]
        for u, v in arcs:
            if not (0 <= u < n and 0 <= v < n):
                return result
            if u != v:
                pred[v].append(u)
                succ[u].append(v)
        degree = [len(x) for x in pred]
        ready = [i for i in range(n) if not degree[i]]
        order = []
        while ready:
            i = ready.pop()
            order.append(i)
            for j in succ[i]:
                degree[j] -= 1
                if not degree[j]:
                    ready.append(j)
        if len(order) != n:
            return result
        for i in order:
            for j in pred[i]:
                E[i] = max(E[i], E[j])
        for i in reversed(order):
            for j in succ[i]:
                L[i] = min(L[i], L[j])
        if any(E[i] > L[i] for i in range(n)):
            return result
        total = sum(p)
        idle_lower = max(0, (m * C - total) // C)
        upper = min(C, total // m, C - idle_lower)
        step = 0
        for value in p:
            step = math.gcd(step, value)
        step = max(1, step)
        upper = upper // step * step
        seed = n * 1000003 + m * 9176 + C
        for i in range(n):
            seed = ((seed ^ (p[i] + 131 * E[i] + 9176 * L[i])) * 1099511628211) & ((1 << 64) - 1)
        rng = random.Random(seed)
        best, best_loads, best_key = None, None, None
        pool = []
        pool_size = max(2, int(2 + math.log1p(limit) * math.sqrt(n) / 3))

        def consider(a):
            nonlocal best, best_loads, best_key, result, pool
            if a is None or len(a) != n:
                return False
            loads = [0] * m
            for i, s in enumerate(a):
                if not isinstance(s, int) or not E[i] <= s <= L[i]:
                    return False
                loads[s] += p[i]
                if loads[s] > C:
                    return False
            if any(a[u] > a[v] for u, v in arcs):
                return False
            key = tuple(sorted(loads))
            if C - key[0] < idle_lower:
                return False
            improved = best_key is None or key > best_key
            if improved:
                best, best_loads, best_key = a[:], loads[:], key
                result = {"task_stations": [s + 1 for s in a], "worst_case_idle_time": C - key[0]}
                pool = [item for item in pool if item[2][0] >= key[0]]
            if best_key is not None and key[0] >= best_key[0] and not any(a == item[0] for item in pool):
                item = (a[:], loads[:], key)
                if len(pool) < pool_size:
                    pool.append(item)
                elif improved or rng.random() < 0.2:
                    pool[rng.randrange(len(pool))] = item
            return improved

        consider(E)
        consider(L)
        full_data = (p, m, C, E, L, pred, succ, order, None)
        orientations = [_orientation(full_data, reverse) for reverse in (False, True)]

        def greedy(reverse, mode, end):
            e, l, pr, su, topo, pm, tail, eligible, due, guided, guide = orientations[int(reverse)]
            a, loads = [-1] * n, [0] * m
            degree = [len(x) for x in pr]
            available = [i for i in range(n) if not degree[i]]
            variant = mode % 7
            for _ in range(n):
                if not available or clock() >= end:
                    return None
                if variant == 0:
                    i = min(available, key=lambda t: (l[t], -tail[t], -p[t]))
                elif variant == 1:
                    i = min(available, key=lambda t: (l[t] - e[t], l[t], -p[t], rng.random()))
                elif variant == 2:
                    i = min(available, key=lambda t: (l[t], -p[t], rng.random()))
                elif variant == 3:
                    i = min(available, key=lambda t: (-tail[t], l[t], -p[t]))
                elif variant == 4:
                    i = min(available, key=lambda t: (l[t], rng.random()))
                elif variant == 5:
                    i = min(available, key=lambda t: (l[t] + rng.random() * max(1, m / 5), -tail[t]))
                else:
                    i = min(available, key=lambda t: (rng.random(), l[t]))
                lo = max([e[i]] + [a[j] for j in pr[i]])
                stations = [s for s in range(lo, l[i] + 1) if loads[s] + p[i] <= C]
                if not stations:
                    return None
                if variant == 0:
                    s = min(stations, key=lambda s: (abs(loads[s] + p[i] - total / m), s))
                elif variant == 1:
                    s = min(stations, key=lambda s: (-loads[s], s))
                elif variant == 2:
                    s = min(stations, key=lambda s: (loads[s], s))
                elif variant in (3, 4):
                    s = stations[0]
                else:
                    s = min(stations, key=lambda s: abs(loads[s] + p[i] - total / m) + rng.random() * C * 0.12)
                a[i], loads[s] = s, loads[s] + p[i]
                available.remove(i)
                for j in su[i]:
                    degree[j] -= 1
                    if not degree[j]:
                        available.append(j)
            return [m - 1 - s for s in a] if reverse else a

        def region(stations, source):
            stations = sorted(set(stations))
            selected = set(stations)
            tasks = [i for i in order if source[i] in selected]
            if not tasks:
                return None
            index = {i: k for k, i in enumerate(tasks)}
            pp, ee, ll, local_guide = [], [], [], []
            pr, su = [[] for _ in tasks], [[] for _ in tasks]
            for k, i in enumerate(tasks):
                if clock() >= deadline:
                    return None
                lo, hi = E[i], L[i]
                for j in pred[i]:
                    if j in index:
                        pr[k].append(index[j])
                    else:
                        lo = max(lo, source[j])
                for j in succ[i]:
                    if j in index:
                        su[k].append(index[j])
                    else:
                        hi = min(hi, source[j])
                pp.append(p[i])
                ee.append(bisect.bisect_left(stations, lo))
                ll.append(bisect.bisect_right(stations, hi) - 1)
                local_guide.append(bisect.bisect_left(stations, source[i]))
            for i in range(len(tasks)):
                for j in pr[i]:
                    ee[i] = max(ee[i], ee[j])
            for i in range(len(tasks) - 1, -1, -1):
                for j in su[i]:
                    ll[i] = min(ll[i], ll[j])
            if any(ee[i] > ll[i] for i in range(len(tasks))):
                return None
            data = (pp, len(stations), C, ee, ll, pr, su, list(range(len(tasks))), local_guide)

            def emit(local):
                a = source[:]
                for k, i in enumerate(tasks):
                    a[i] = stations[local[k]]
                return consider(a)
            return data, emit

        def move_ok(i, s, assignment):
            return (E[i] <= s <= L[i]
                    and all(assignment[j] <= s for j in pred[i])
                    and all(s <= assignment[j] for j in succ[i]))

        def repair_chain(source, source_loads, stop, strength):
            # Every intermediate state keeps capacity and precedence feasibility; bottlenecks shifted temporarily are removed by subsequent closure rebuilding.
            a, loads = source[:], source_loads[:]
            target = best_key[0] + step
            used, touched = set(), set()
            depth = max(1, int(math.log1p(limit) * strength + math.sqrt(n) / (1 + math.log1p(n))))
            while clock() < stop and len(used) < min(n, depth):
                receivers = [s for s in range(m) if loads[s] == min(loads)]
                receiver = rng.choice(receivers)
                touched.add(receiver)
                candidates = []

                def add(i, j, donor, change):
                    old_r, old_d = loads[receiver], loads[donor]
                    new_r, new_d = old_r + change, old_d - change
                    if change <= 0 or new_r > C or new_d > C or new_d < 0:
                        return
                    deficit = (max(0, target - new_r) + max(0, target - new_d)
                               - max(0, target - old_r) - max(0, target - old_d))
                    balance = (new_r * new_r + new_d * new_d - old_r * old_r - old_d * old_d) / (C * C + 1)
                    candidates.append(((deficit, balance, rng.random()), i, j, donor, change))

                for i in range(n):
                    if clock() >= stop:
                        break
                    donor = a[i]
                    if donor == receiver or i in used or loads[donor] <= loads[receiver]:
                        continue
                    if loads[receiver] + p[i] <= C and move_ok(i, receiver, a):
                        add(i, -1, donor, p[i])
                if not candidates or min(x[0][0] for x in candidates) >= 0:
                    for j in range(n):
                        if clock() >= stop:
                            break
                        if a[j] != receiver or j in used:
                            continue
                        for i in range(n):
                            if clock() >= stop:
                                break
                            donor = a[i]
                            change = p[i] - p[j]
                            if donor == receiver or i in used or change <= 0:
                                continue
                            if loads[receiver] + change > C or loads[donor] - change < 0:
                                continue
                            a[i], a[j] = receiver, donor
                            valid = move_ok(i, receiver, a) and move_ok(j, donor, a)
                            a[i], a[j] = donor, receiver
                            if valid:
                                add(i, j, donor, change)
                if not candidates or clock() >= stop:
                    break
                candidates.sort()
                restricted = max(1, min(len(candidates), int(1 + math.log1p(limit) * (strength - 1))))
                chosen = int((rng.random() ** 3) * restricted)
                _, i, j, donor, change = candidates[chosen]
                a[i] = receiver
                used.add(i)
                if j >= 0:
                    a[j] = donor
                    used.add(j)
                loads[receiver] += change
                loads[donor] -= change
                touched.add(donor)
                consider(a)
                if min(loads) >= target:
                    break
            return a, loads, touched

        attempt, failures = 0, 0
        initial_end = min(deadline, started + limit * 0.025)
        while clock() < initial_end:
            consider(greedy(bool(attempt & 1), attempt, initial_end))
            attempt += 1
        scale = math.sqrt(limit)
        if best is None and clock() < deadline:
            _beam(full_data, 0, False, max(2, int(3 + scale)), max(2, int(3 + scale)),
                  max(2, int(2 + math.log1p(limit))), attempt, rng,
                  min(deadline, clock() + limit * 0.08), consider)
        if clock() < deadline:
            duration = limit * (0.36 if best is None else 0.16)
            data = full_data[:-1] + (best,)
            _integer_rebuild(data, 0 if best is None else best_key[0],
                             min(deadline, clock() + duration), rng, consider)
        next_restart = clock() + limit * 0.1
        next_global = clock() + limit * 0.22
        while clock() < deadline:
            now = clock()
            before = best_key
            progress = max(0.0, (now - started) / max(limit, 1e-9))
            strength = 1 + math.sqrt(progress) + 0.14 * math.log1p(failures)
            complexity = 1 + math.log1p(n + len(arcs)) / (2 + math.log1p(limit))
            width = max(2, int((8 + 10 * scale) * strength / math.sqrt(complexity)))
            local_width = max(2, int((4 + 3 * scale) * math.sqrt(strength)))
            branches = max(2, int(2 + math.log2(1 + limit) + 0.25 * math.log1p(failures)))
            if best is None:
                data = full_data
                stop = min(deadline, now + limit * min(0.28, 0.07 * (1 + math.sqrt(failures))))
                _beam(data, 0, bool(attempt & 1), width, local_width, branches,
                      attempt, rng, stop, consider)
                if best is None and clock() < deadline:
                    consider(greedy(bool(attempt & 1), attempt, min(deadline, clock() + limit * 0.01)))
                if best is None and clock() < deadline and np is not None:
                    _integer_rebuild(data, 0, min(deadline, clock() + limit * 0.18), rng, consider)
            elif m > 1:
                source, source_loads = best, best_loads
                if pool and rng.random() < 0.25:
                    source, source_loads, _ = rng.choice(pool)
                repair_stop = min(deadline, clock() + limit * 0.006 * math.sqrt(strength))
                source, source_loads, affected = repair_chain(source, source_loads, repair_stop, strength)
                ranked = sorted(range(m), key=lambda s: (source_loads[s], rng.random()))
                low = ranked[0]
                affected.add(low)
                choice = rng.random()
                if len(affected) <= 2 and choice < 0.48:
                    if len(affected) == 1:
                        neighbors = [s for s in (low - 1, low + 1) if 0 <= s < m]
                        other = rng.choice(neighbors) if rng.random() < 0.65 else rng.choice([s for s in range(m) if s != low])
                        affected.add(other)
                    sub = region(affected, source)
                    if sub is not None:
                        stop = min(deadline, clock() + limit * 0.01 * math.sqrt(strength))
                        _two_station(sub[0], stop, limit, failures > 0, sub[1])
                else:
                    span_max = min(m, max(2, int(2 + math.log1p(limit) * math.sqrt(strength))))
                    span = rng.randint(2, span_max)
                    lo = max(0, min(m - span, low - rng.randrange(span)))
                    affected.update(range(lo, lo + span))
                    if choice > 0.8:
                        affected.add(ranked[-1])
                    sub = region(affected, source)
                    if sub is not None:
                        data, emit = sub
                        bound = min(C, sum(data[0]) // len(affected))
                        base_min = min(source_loads[s] for s in affected)
                        target = min(bound, max(best_key[0], base_min))
                        if rng.random() < 0.35 and target + step <= bound:
                            target += step
                        duration = limit * 0.04 * math.sqrt(strength)
                        end = min(deadline, clock() + duration)
                        # Within the same repair-chain neighbourhood, keep grouped candidates first, then supplement rebuilding with the exact cumulative-variable model.
                        beam_end = min(end, clock() + duration * (0.35 if np is not None else 1.0))
                        _beam(data, target, bool(attempt & 1), width, local_width, branches,
                              attempt, rng, beam_end, emit)
                        if np is not None and clock() < end:
                            if best_key != before:
                                refreshed = region(affected, best)
                                if refreshed is not None:
                                    data, emit = refreshed
                            _integer_rebuild(data, min(bound, best_key[0]), end, rng, emit)
                if clock() >= next_global and clock() < deadline:
                    data = full_data[:-1] + (best,)
                    gap = max(0, (upper - best_key[0]) // step)
                    jump = 1 if rng.random() < 0.7 else rng.randint(1, max(1, gap))
                    target = min(upper, best_key[0] + jump * step)
                    end = min(deadline, clock() + limit * 0.095 * math.sqrt(strength))
                    _beam(data, target, bool(attempt & 1), width, local_width, branches,
                          attempt, rng, end, consider)
                    next_global = clock() + limit * 0.18 / math.sqrt(strength)
            if clock() >= next_restart and clock() < deadline:
                consider(greedy(bool(attempt & 1), attempt, min(deadline, clock() + limit * 0.01)))
                next_restart = clock() + limit * 0.1
            failures = failures + 1 if before == best_key else 0
            attempt += 1
        return result
    except BaseException:
        return result
