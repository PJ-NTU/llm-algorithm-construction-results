"""Recursive precedence-closure splitting with bottleneck reassignment
Combines bidirectional grouped beam search, exact two-station reassignment and multi-station closure splitting with subset-sum pruning and subproblem caching, continuously improving feasible solutions within the wall-clock budget.
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
    milp = None


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
            return [(full, remaining, 0.0, 0.0)] if low <= remaining <= high else []
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


def _two_station(data, end, budget, diversify, emit, proof=None):
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

    if proof is not None:
        proof["complete"] = False
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
    if proof is not None:
        proof["complete"] = clock() < end
    return improved



def _closed_partition(data, target, end, budget, rng, emit, proof=None):
    """Recursively splits a multi-station problem into a predecessor-closed set and a successor-closed set."""
    p, m, C, e, l, pred, succ, order, guide = data
    clock = time.perf_counter
    n = len(p)
    if proof is not None:
        proof["infeasible"] = False
    if not n or clock() >= end:
        return False
    anc, desc = [0] * n, [0] * n
    divisor = 0
    for i in order:
        if clock() >= end:
            return False
        bits = 1 << i
        for j in pred[i]:
            bits |= anc[j]
        anc[i] = bits
        divisor = math.gcd(divisor, p[i])
    for i in reversed(order):
        bits = 1 << i
        for j in succ[i]:
            bits |= desc[j]
        desc[i] = bits
    divisor = max(1, divisor)
    weights = {0: 0}
    memo = {}
    stopped = False
    # This cap limits bitset storage, not search nodes or restarts.
    bit_budget = int(1200000 * (1 + math.log1p(max(0.0, budget))))

    def weight(bits):
        cached = weights.get(bits)
        if cached is not None:
            return cached
        original, value = bits, 0
        while bits:
            bit = bits & -bits
            bits ^= bit
            value += p[bit.bit_length() - 1]
        weights[original] = value
        return value

    def subset(free, fixed, lower, upper, desired):
        # Subset sum ignoring the remaining precedence arcs is a necessary condition; closure is verified again after rebuilding.
        count = free.bit_count()
        lower = max(0, (lower + divisor - 1) // divisor)
        upper = min(weight(free) // divisor, upper // divisor)
        if lower > upper:
            return False, None
        if (upper + 1) * (count + 1) > bit_budget:
            return True, None
        items, bits = [], free
        while bits:
            bit = bits & -bits
            bits ^= bit
            items.append(bit.bit_length() - 1)
        rng.shuffle(items)
        cap = (1 << (upper + 1)) - 1
        history, reachable = [], 1
        for i in items:
            if clock() >= end:
                return True, None
            history.append(reachable)
            units = p[i] // divisor
            if units <= upper:
                reachable = (reachable | (reachable << units)) & cap
        legal = (reachable >> lower) << lower
        if not legal:
            return False, None
        middle = max(lower, min(upper, int(desired / divisor)))
        below = legal & ((1 << (middle + 1)) - 1)
        above = legal >> (middle + 1)
        candidates = []
        if below:
            candidates.append(below.bit_length() - 1)
        if above:
            candidates.append(middle + (above & -above).bit_length())
        remaining = min(candidates, key=lambda v: abs(v * divisor - desired))
        chosen = fixed
        for k in range(len(items) - 1, -1, -1):
            i = items[k]
            units = p[i] // divisor
            can_skip = (history[k] >> remaining) & 1
            can_take = remaining >= units and (history[k] >> (remaining - units)) & 1
            if can_take and (not can_skip or rng.random() < 0.5):
                chosen |= 1 << i
                remaining -= units
        return True, chosen

    def arrange(mask, first, last):
        nonlocal stopped
        if clock() >= end:
            stopped = True
            return None
        key = (mask, first, last)
        if key in memo:
            return memo[key]
        count, work = last - first + 1, weight(mask)
        if work < count * target or work > count * C:
            memo[key] = None
            return None
        release, due = [0] * count, [0] * count
        bits = mask
        while bits:
            bit = bits & -bits
            bits ^= bit
            i = bit.bit_length() - 1
            if e[i] > last or l[i] < first:
                memo[key] = None
                return None
            release[max(0, e[i] - first)] += p[i]
            due[min(count - 1, l[i] - first)] += p[i]
        used = available = 0
        for k in range(count):
            used += due[k]
            available += release[k]
            if used > (k + 1) * C or available < (k + 1) * target:
                memo[key] = None
                return None
        used = available = 0
        for k in range(count - 1, -1, -1):
            used += release[k]
            available += due[k]
            if used > (count - k) * C or available < (count - k) * target:
                memo[key] = None
                return None
        if count == 1:
            memo[key] = (mask,)
            return (mask,)
        # Randomly choose the middle cut of an odd window, keeping the recursion depth logarithmic.
        left_count = count // 2 + int(count % 2 and rng.random() < 0.5)
        cut = first + left_count - 1
        right_count = count - left_count
        low = max(left_count * target, work - right_count * C)
        high = min(left_count * C, work - right_count * target)
        forced_left = forced_right = 0
        bits = mask
        while bits:
            bit = bits & -bits
            bits ^= bit
            i = bit.bit_length() - 1
            if l[i] <= cut:
                forced_left |= anc[i] & mask
            if e[i] > cut:
                forced_right |= desc[i] & mask
        if forced_left & forced_right:
            memo[key] = None
            return None
        desired = work * left_count / count
        tried = set()

        def attempt(left):
            if left in tried:
                return None
            tried.add(left)
            wl = weight(left)
            if not low <= wl <= high:
                return None
            right = mask ^ left
            # The caches of the left and right subproblems depend only on the task set and station interval, not on the search path.
            a = arrange(left, first, cut)
            if a is None:
                return None
            b = arrange(right, cut + 1, last)
            if b is None:
                return None
            return a + b

        if guide is not None:
            guided, bits = 0, mask
            while bits:
                bit = bits & -bits
                bits ^= bit
                if guide[bit.bit_length() - 1] <= cut:
                    guided |= bit
            if forced_left & ~guided == 0 and forced_right & guided == 0:
                solution = attempt(guided)
                if solution is not None:
                    memo[key] = solution
                    return solution
                if stopped:
                    return None
        stack = [(forced_left, forced_right, weight(forced_left), weight(forced_right))]
        while stack and clock() < end:
            left, right, wl, wr = stack.pop()
            if wl > high or work - wr < low:
                continue
            free = mask ^ (left | right)
            possible, candidate = subset(free, left, low - wl, high - wl, desired - wl)
            if not possible:
                continue
            if clock() >= end:
                stopped = True
                return None
            conflict = -1
            if candidate is not None:
                bits = candidate
                while bits:
                    bit = bits & -bits
                    bits ^= bit
                    i = bit.bit_length() - 1
                    if anc[i] & mask & ~candidate:
                        conflict = i
                        break
                if conflict < 0:
                    solution = attempt(candidate)
                    if solution is not None:
                        memo[key] = solution
                        return solution
                    if stopped:
                        return None
                    if count == 2:
                        continue
            for candidate in (left, mask ^ right):
                solution = attempt(candidate)
                if solution is not None:
                    memo[key] = solution
                    return solution
                if stopped:
                    return None
            if not free:
                continue
            choice, best_score, bits = conflict, None, free
            if choice < 0:
                while bits:
                    if clock() >= end:
                        stopped = True
                        return None
                    bit = bits & -bits
                    bits ^= bit
                    i = bit.bit_length() - 1
                    a, d = anc[i] & free, desc[i] & free
                    score = ((a | d).bit_count(), p[i], -(l[i] - e[i]))
                    if best_score is None or score > best_score:
                        best_score, choice = score, i
            add_left, add_right = anc[choice] & free, desc[choice] & free
            wa, wd = weight(add_left), weight(add_right)
            options = []
            if anc[choice] & right == 0 and wl + wa <= high:
                options.append((left | add_left, right, wl + wa, wr))
            if desc[choice] & left == 0 and work - wr - wd >= low:
                options.append((left, right | add_right, wl, wr + wd))
            if len(options) == 2:
                take_left = guide is not None and guide[choice] <= cut
                if guide is None:
                    take_left = wl < desired and rng.random() < left_count / count
                elif rng.random() < 1 / (2 + math.sqrt(max(0.0, budget))):
                    take_left = not take_left
                if take_left:
                    options.reverse()
            stack.extend(options)
        if clock() >= end:
            stopped = True
        if not stopped:
            memo[key] = None
        return None

    groups = arrange((1 << n) - 1, 0, m - 1)
    if groups is None:
        if proof is not None:
            proof["infeasible"] = not stopped
        return False
    assignment = [0] * n
    for station, bits in enumerate(groups):
        while bits:
            bit = bits & -bits
            bits ^= bit
            assignment[bit.bit_length() - 1] = station
    return emit(assignment)


def _feasibility(data, end):
    """Calls the cumulative-assignment integer model only when there is no initial feasible solution."""
    if milp is None or time.perf_counter() >= end:
        return None
    p, m, C, e, l, pred, succ, order, guide = data
    try:
        starts, count = [], 0
        for i in range(len(p)):
            starts.append(count)
            count += l[i] - e[i]
        if not count:
            return e[:]
        rows, cols, values, lower, upper = [], [], [], [], []

        def add(terms, lo, hi):
            row = len(lower)
            for col, value in terms:
                rows.append(row)
                cols.append(col)
                values.append(value)
            lower.append(lo)
            upper.append(hi)

        fixed = [0] * m
        terms = [[] for _ in range(m)]
        for i in range(len(p)):
            if time.perf_counter() >= end:
                return None
            fixed[l[i]] += p[i]
            for s in range(e[i], l[i]):
                col = starts[i] + s - e[i]
                terms[s].append((col, p[i]))
                terms[s + 1].append((col, -p[i]))
                if s > e[i]:
                    add([(col - 1, 1), (col, -1)], -np.inf, 0)
            for j in pred[i]:
                for s in range(e[i], l[j]):
                    add([(starts[i] + s - e[i], 1),
                         (starts[j] + s - e[j], -1)], -np.inf, 0)
        for s in range(m):
            add(terms[s], -fixed[s], C - fixed[s])
        matrix = coo_matrix((values, (rows, cols)), shape=(len(lower), count)).tocsc()
        remaining = end - time.perf_counter()
        if remaining <= 0:
            return None
        solution = milp(np.zeros(count), integrality=np.ones(count, dtype=np.uint8),
                        bounds=Bounds(np.zeros(count), np.ones(count)),
                        constraints=LinearConstraint(matrix, np.asarray(lower), np.asarray(upper)),
                        options={"time_limit": remaining * 0.82, "mip_rel_gap": 0.0, "presolve": True})
        if solution.x is None:
            return None
        assignment = l[:]
        for i in range(len(p)):
            for s in range(e[i], l[i]):
                if solution.x[starts[i] + s - e[i]] > 0.5:
                    assignment[i] = s
                    break
        return assignment
    except Exception:
        return None

def solve(instance: dict, time_limit_s: float) -> dict:
    started = time.perf_counter()
    result = {"task_stations": [], "worst_case_idle_time": 0}
    try:
        n = int(instance["num_tasks"])
        m = int(instance["num_stations"])
        C = int(instance["cycle_time"])
        p = [int(x) for x in instance["processing_times"]]
        E = [int(x) - 1 for x in instance["earliest_stations"]]
        L = [int(x) - 1 for x in instance["latest_stations"]]
        result = {"task_stations": [x + 1 for x in E], "worst_case_idle_time": max(0, C)}
        limit = float(time_limit_s)
        if not math.isfinite(limit):
            limit = 0.0
        limit = max(0.0, limit)
        reserve = min(limit * 0.04, 0.002 + limit * 0.005)
        deadline = started + max(0.0, limit - reserve)
        clock = time.perf_counter
        if n == 0:
            return {"task_stations": [], "worst_case_idle_time": C}
        if m <= 0 or C <= 0 or len(p) != n or len(E) != n or len(L) != n:
            return result
        if any(p[i] < 0 or p[i] > C or E[i] < 0 or E[i] > L[i] or L[i] >= m for i in range(n)):
            return result
        arcs = sorted(set((int(a[0]) - 1, int(a[1]) - 1) for a in instance["precedence_arcs"]))
        pred, succ = [[] for _ in range(n)], [[] for _ in range(n)]
        for u, v in arcs:
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
        divisor = 0
        for value in p:
            divisor = math.gcd(divisor, value)
        step = max(1, divisor)
        upper = (upper // step) * step
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
                best, best_loads, best_key = a[:], loads, key
                result = {"task_stations": [s + 1 for s in a], "worst_case_idle_time": int(C - key[0])}
                pool = [item for item in pool if item[2][0] >= key[0]]
            if best_key is not None and key[0] >= best_key[0] and not any(a == item[0] for item in pool):
                item = (a[:], loads[:], key)
                if len(pool) < pool_size:
                    pool.append(item)
                elif improved or rng.random() < 0.25:
                    pool[rng.randrange(len(pool))] = item
            return improved

        consider(E)
        consider(L)
        orientations = []
        for reverse in (False, True):
            data = (p, m, C, E, L, pred, succ, order, None)
            orientations.append(_orientation(data, reverse))

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
                candidates = [s for s in range(lo, l[i] + 1) if loads[s] + p[i] <= C]
                if not candidates:
                    return None
                if variant == 0:
                    station = min(candidates, key=lambda s: (abs(loads[s] + p[i] - total / m), s))
                elif variant == 1:
                    station = min(candidates, key=lambda s: (-loads[s], s))
                elif variant == 2:
                    station = min(candidates, key=lambda s: (loads[s], s))
                elif variant in (3, 4):
                    station = candidates[0]
                else:
                    station = min(candidates, key=lambda s: abs(loads[s] + p[i] - total / m) + rng.random() * C * 0.12)
                a[i], loads[station] = station, loads[station] + p[i]
                available.remove(i)
                for j in su[i]:
                    degree[j] -= 1
                    if degree[j] == 0:
                        available.append(j)
            return [m - 1 - s for s in a] if reverse else a

        def region(stations, source):
            stations = sorted(set(stations))
            selected = set(stations)
            tasks = [i for i in order if source[i] in selected]
            index = {i: k for k, i in enumerate(tasks)}
            count = len(tasks)
            if not count:
                return None
            pp, ee, ll = [], [], []
            pr, su = [[] for _ in tasks], [[] for _ in tasks]
            local_guide = []
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
            for i in range(count):
                for j in pr[i]:
                    ee[i] = max(ee[i], ee[j])
            for i in range(count - 1, -1, -1):
                for j in su[i]:
                    ll[i] = min(ll[i], ll[j])
            if any(ee[i] > ll[i] for i in range(count)):
                return None
            data = (pp, len(stations), C, ee, ll, pr, su, list(range(count)), local_guide)

            def emit(local):
                a = source[:]
                for k, i in enumerate(tasks):
                    a[i] = stations[local[k]]
                return consider(a)
            return data, emit

        attempt, failures = 0, 0
        initial_end = min(deadline, started + limit * 0.04)
        while best is None and clock() < initial_end:
            consider(greedy(bool(attempt & 1), attempt, initial_end))
            attempt += 1
        scale = math.sqrt(limit)
        next_restart = clock() + limit * 0.11
        pair_proofs, closed_failures = set(), set()
        cache_size = max(2, int((1 + math.log1p(limit)) * (n + m)))
        next_mip = started + limit * 0.17
        while clock() < deadline:
            now = clock()
            progress = (now - started) / max(limit, 1e-9)
            before = best_key
            source, source_loads = best, best_loads
            if best is not None and pool and rng.random() < 0.24:
                source, source_loads, _ = rng.choice(pool)
            strength = 1 + math.sqrt(max(0.0, progress)) + 0.15 * math.log1p(failures)
            width = max(2, int((10 + 9 * scale) * strength))
            local_width = max(2, int((5 + 3 * scale) * math.sqrt(strength)))
            branches = max(2, int(2 + math.log2(1 + limit) + 0.3 * math.log1p(failures)))
            quantum = limit * (1 + math.log1p(failures) / 4) / (18 + math.sqrt(n / max(1, m)))
            if best is None:
                data = (p, m, C, E, L, pred, succ, order, None)
                if clock() >= next_mip and milp is not None:
                    end = min(deadline, clock() + (deadline - clock()) * 0.42)
                    consider(_feasibility(data, end))
                    next_mip = clock() + limit * 0.2
                elif failures % 2:
                    _closed_partition(data, 0, min(deadline, now + quantum * 2), limit, rng, consider)
                else:
                    width = max(2, int((4 + 2 * scale) * (1 + math.log1p(failures))))
                    _beam(data, 0, bool(attempt & 1), width, local_width, branches, attempt, rng,
                          min(deadline, now + quantum * 2), consider)
                if best is None and clock() < deadline:
                    consider(greedy(bool(attempt & 1), attempt + 2, min(deadline, clock() + limit * 0.012)))
            elif m > 1:
                choice = rng.random()
                ranked = sorted(range(m), key=lambda s: (source_loads[s], rng.random()))
                low = ranked[min(m - 1, int((rng.random() ** 3) * max(1, math.sqrt(m))))]
                if m == 2 or choice < 0.42:
                    if rng.random() < 0.63:
                        neighbors = [s for s in (low - 1, low + 1) if 0 <= s < m]
                        other = rng.choice(neighbors)
                    else:
                        donors = sorted((s for s in range(m) if s != low),
                                        key=lambda s: (-source_loads[s], abs(s - low)))
                        other = donors[int((rng.random() ** 2) * len(donors))]
                    stations = tuple(sorted((low, other)))
                    signature = (stations, tuple(source))
                    diversify = failures > 0 and rng.random() < 0.25
                    if signature not in pair_proofs or diversify:
                        sub = region(stations, source)
                        if sub is not None:
                            proof = {}
                            pair_slice = quantum / (3 + math.log1p(n / m))
                            _two_station(sub[0], min(deadline, clock() + pair_slice), limit,
                                         diversify, sub[1], proof)
                            if proof.get("complete"):
                                pair_proofs.add(signature)
                elif choice < 0.92:
                    max_span = min(m, max(3, int(2 + math.log1p(limit) * strength)))
                    span = 3 + int((rng.random() ** 1.5) * (max_span - 2))
                    lo = max(0, min(m - span, low - rng.randrange(span)))
                    stations = list(range(lo, lo + span))
                    if rng.random() < 0.2 and span < m:
                        stations = sorted(set(stations + [ranked[-1]]))
                    sub = region(stations, source)
                    if sub is not None:
                        sub_total, q = sum(sub[0][0]), len(stations)
                        base_min = min(source_loads[s] for s in stations)
                        bound = (min(C, sub_total // q) // step) * step
                        target = min(bound, base_min + step)
                        if target > base_min and rng.random() < 0.18:
                            target = bound
                        signature = (tuple(stations), tuple(source), target)
                        if target > base_min and rng.random() < 0.78 and signature not in closed_failures:
                            proof = {}
                            _closed_partition(sub[0], target, min(deadline, clock() + quantum),
                                              limit, rng, sub[1], proof)
                            if proof.get("infeasible"):
                                closed_failures.add(signature)
                        else:
                            if signature in closed_failures:
                                target = base_min
                            _beam(sub[0], target, bool(attempt & 1), width, local_width, branches, attempt,
                                  rng, min(deadline, clock() + quantum), sub[1])
                else:
                    gap = max(0, (upper - best_key[0]) // step)
                    jump = 1 if rng.random() < 0.8 else rng.randint(1, max(1, gap))
                    target = min(upper, best_key[0] + jump * step)
                    data = (p, m, C, E, L, pred, succ, order, source)
                    if target > best_key[0] and rng.random() < 0.6:
                        _closed_partition(data, target, min(deadline, clock() + quantum * 2),
                                          limit, rng, consider)
                    else:
                        _beam(data, target, bool(attempt & 1), width, local_width, branches, attempt,
                              rng, min(deadline, clock() + quantum * 2), consider)
            if clock() >= next_restart and clock() < deadline:
                consider(greedy(bool(attempt & 1), attempt, min(deadline, clock() + limit * 0.012)))
                next_restart = clock() + limit * 0.11
            if len(pair_proofs) > cache_size:
                pair_proofs.clear()
            if len(closed_failures) > cache_size:
                closed_failures.clear()
            failures = failures + 1 if before == best_key else 0
            attempt += 1
        return result
    except BaseException:
        return result
