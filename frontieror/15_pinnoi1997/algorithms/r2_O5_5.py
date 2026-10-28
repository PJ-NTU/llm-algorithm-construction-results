"""Multi-station constraint propagation with cooperative closure balancing
Builds solutions by bidirectional grouped beam search, and keeps improving with multi-station reassignment under load-bound and subset-sum propagation, two-station closure search and a pool of feasible solutions.
"""
import bisect
import math
import random
import time


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


def _constraint_search(data, target, end, budget, rng, emit):
    """Time-limited search over multi-station task domains; returns whether a solution was found and whether the search was exhausted."""
    p0, q, capacity, e, l, pred0, succ0, order, guide = data
    n = len(p0)
    clock = time.perf_counter
    if clock() >= end:
        return False, False
    divisor = 0
    for v in p0:
        divisor = math.gcd(divisor, v)
    divisor = max(1, divisor)
    p = [v // divisor for v in p0]
    C = capacity // divisor
    target = (target + divisor - 1) // divisor
    total = sum(p)
    if total < q * target or total > q * C:
        return False, True
    domains = [((1 << (l[i] + 1)) - 1) ^ ((1 << e[i]) - 1) for i in range(n)]
    pred, succ = [v[:] for v in pred0], [v[:] for v in succ0]
    # Tasks with equal weight, equal bounds and equal adjacency are interchangeable; break this symmetry using the order in the current solution.
    twins = {}
    for i in range(n):
        key = (p[i], e[i], l[i], tuple(sorted(pred[i])), tuple(sorted(succ[i])))
        twins.setdefault(key, []).append(i)
    for group in twins.values():
        if len(group) > 1:
            group.sort(key=lambda i: (guide[i] if guide is not None else 0, i))
            for u, v in zip(group, group[1:]):
                pred[v].append(u)
                succ[u].append(v)
    arcs = [(u, v) for v in range(n) for u in pred[v]]
    noise = [rng.random() for _ in p]
    bit_budget = int(2000000 * (1 + math.log1p(max(0.0, budget))) / math.sqrt(q))
    support_size = max(2, int(math.sqrt(n + q) + math.log1p(max(0.0, budget))))
    stack = [domains]
    while stack and clock() < end:
        ds = stack.pop()
        failed = False
        changed = True
        while changed and not failed:
            if clock() >= end:
                return False, False
            changed = False
            for u, v in arcs:
                du, dv = ds[u], ds[v]
                if not du or not dv:
                    failed = True
                    break
                nu = du & ((1 << dv.bit_length()) - 1)
                if not nu:
                    failed = True
                    break
                nv = dv & ~((nu & -nu) - 1)
                if not nv:
                    failed = True
                    break
                if nu != du or nv != dv:
                    ds[u], ds[v] = nu, nv
                    changed = True
            if failed:
                break
            forced, possible = [0] * q, [0] * q
            flexible = [[] for _ in range(q)]
            for i, d in enumerate(ds):
                if not d:
                    failed = True
                    break
                if not d & (d - 1):
                    s = d.bit_length() - 1
                    forced[s] += p[i]
                    possible[s] += p[i]
                else:
                    while d:
                        b = d & -d
                        d ^= b
                        s = b.bit_length() - 1
                        possible[s] += p[i]
                        flexible[s].append(i)
            if failed:
                break
            lower = [max(target, v) for v in forced]
            upper = [min(C, v) for v in possible]
            if any(lower[s] > upper[s] for s in range(q)):
                failed = True
                break
            # Bounds on the reachable load of each station; an unreachable capacity tail cannot count as remaining capacity.
            dp_rows = []
            for s in range(q):
                items = flexible[s]
                cap = min(C - forced[s], possible[s] - forced[s])
                if cap < 0:
                    failed = True
                    break
                if cap * max(1, len(items)) > bit_budget:
                    continue
                mask = (1 << (cap + 1)) - 1
                reach = 1
                for i in items:
                    if clock() >= end:
                        return False, False
                    if p[i] <= cap:
                        reach = (reach | (reach << p[i])) & mask
                low = max(0, lower[s] - forced[s])
                valid = reach >> low
                if not valid:
                    failed = True
                    break
                lower[s] = forced[s] + low + (valid & -valid).bit_length() - 1
                upper[s] = forced[s] + reach.bit_length() - 1
                if len(items) <= support_size:
                    dp_rows.append((s, cap, mask))
            if failed:
                break
            sl, su = sum(lower), sum(upper)
            if sl > total or su < total:
                failed = True
                break
            for s in range(q):
                oldlo, oldhi = lower[s], upper[s]
                lower[s] = max(oldlo, total - su + oldhi)
                upper[s] = min(oldhi, total - sl + oldlo)
            # Fixed loads and possible loads give necessary conditions for taking or not taking each task.
            for s in range(q):
                bit = 1 << s
                for i in flexible[s]:
                    d = ds[i]
                    if not d & bit:
                        continue
                    if forced[s] + p[i] > upper[s]:
                        nd = d & ~bit
                    elif possible[s] - p[i] < lower[s]:
                        nd = bit
                    else:
                        continue
                    if not nd:
                        failed = True
                        break
                    if nd != d:
                        ds[i] = nd
                        changed = True
                if failed:
                    break
            if failed:
                break
            if changed:
                continue
            # Small candidate sets have the support of each value checked one by one; equal weights share a single subset-sum computation.
            for s, cap, mask in dp_rows:
                bit, cache = 1 << s, {}
                low, high = lower[s] - forced[s], upper[s] - forced[s]
                for i in flexible[s]:
                    if clock() >= end:
                        return False, False
                    w = p[i]
                    if w not in cache:
                        reach, skipped = 1, False
                        for j in flexible[s]:
                            if p[j] == w and not skipped:
                                skipped = True
                            elif p[j] <= cap:
                                reach = (reach | (reach << p[j])) & mask
                        def supported(a, b):
                            a = max(0, a)
                            return b >= a and bool((reach >> a) & ((1 << (b - a + 1)) - 1))
                        cache[w] = supported(low, high), supported(low - w, high - w)
                    excluded, included = cache[w]
                    d = ds[i]
                    nd = d
                    if not included:
                        nd &= ~bit
                    if not excluded:
                        nd &= bit
                    if not nd:
                        failed = True
                        break
                    if nd != d:
                        ds[i] = nd
                        changed = True
                if failed:
                    break
        if failed:
            continue
        if clock() >= end:
            return False, False
        # Check the projected guide solution first, to avoid walking variable by variable through an existing complete solution.
        assignment, loads, undecided = [], [0] * q, []
        for i, d in enumerate(ds):
            if guide is not None and d & (1 << guide[i]):
                s = guide[i]
            else:
                s = (d & -d).bit_length() - 1
            assignment.append(s)
            loads[s] += p[i]
            if d & (d - 1):
                undecided.append(i)
        if all(target <= v <= C for v in loads) and all(assignment[u] <= assignment[v] for u, v in arcs):
            emit(assignment)
            return True, False
        if not undecided:
            continue
        choice = None
        # Branch first on tasks that may relieve the current bottleneck; the binary domain split keeps all other values.
        deficient = [s for s in range(q) if loads[s] < target]
        if deficient:
            s = min(deficient, key=lambda v: (loads[v], len(flexible[v])))
            need, bit = target - loads[s], 1 << s
            ranked = []
            for i in flexible[s]:
                old = assignment[i]
                if old == s or not ds[i] & bit:
                    continue
                blockers = sum(assignment[j] > s for j in pred[i]) + sum(assignment[j] < s for j in succ[i])
                loss = max(0, target - loads[old] + p[i])
                key = (blockers, loss, abs(p[i] - need), ds[i].bit_count(), noise[i])
                ranked.append((key, i))
            if ranked:
                _, i = min(ranked)
                choice = i, bit, True
        if choice is None:
            overloaded = [s for s in range(q) if loads[s] > C]
            if overloaded:
                s = max(overloaded, key=lambda v: loads[v] - C)
                items = [i for i in undecided if assignment[i] == s]
                if items:
                    i = min(items, key=lambda j: (ds[j].bit_count(), -p[j], noise[j]))
                    choice = i, 1 << s, False
        if choice is None:
            i = min(undecided, key=lambda j: (ds[j].bit_count(), -len(pred[j]) - len(succ[j]), -p[j], noise[j]))
            choice = i, 1 << assignment[i], True
        i, bit, prefer_in = choice
        include, exclude = ds[:], ds[:]
        include[i], exclude[i] = bit, ds[i] & ~bit
        if prefer_in:
            stack.append(exclude)
            stack.append(include)
        else:
            stack.append(include)
            stack.append(exclude)
    return False, not stack


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
        initial_end = min(deadline, started + limit * 0.025)
        while best is None and clock() < initial_end:
            consider(greedy(bool(attempt & 1), attempt, initial_end))
            attempt += 1
        scale = math.sqrt(limit)
        next_restart = clock() + limit * 0.13
        infeasible = {}
        proof_capacity = max(2, int((1 + math.log1p(limit)) * math.sqrt(n + m)))

        def exact(data, target, end, emit):
            if clock() >= end:
                return False
            pp, q, cap, ee, ll, pr = data[:6]
            key = (q, tuple(pp), tuple(ee), tuple(ll), tuple(tuple(x) for x in pr))
            if key in infeasible and target >= infeasible[key]:
                return False
            found, complete = _constraint_search(data, target, end, limit, rng, emit)
            if complete and not found:
                if len(infeasible) >= proof_capacity and key not in infeasible:
                    infeasible.pop(next(iter(infeasible)))
                infeasible[key] = min(target, infeasible.get(key, target))
            return found

        while clock() < deadline:
            now = clock()
            progress = (now - started) / max(limit, 1e-9)
            before = best_key
            source, source_loads = best, best_loads
            if best is not None and pool and rng.random() < 0.24:
                source, source_loads, _ = rng.choice(pool)
            strength = 1 + math.sqrt(max(0.0, progress)) + 0.12 * math.log1p(failures)
            width = max(2, int((10 + 9 * scale) * strength))
            local_width = max(2, int((5 + 3 * scale) * math.sqrt(strength)))
            branches = max(2, int(2 + math.log2(1 + limit) + 0.3 * math.log1p(failures)))
            if best is None:
                data = (p, m, C, E, L, pred, succ, order, None)
                slice_s = limit * min(0.35, 0.065 * (1 + math.sqrt(failures)))
                if failures % 2 == 1:
                    exact(data, 0, min(deadline, now + slice_s), consider)
                else:
                    width = max(2, int((4 + 2 * scale) * (1 + math.log1p(failures))))
                    _beam(data, 0, bool(attempt & 1), width, local_width, branches, attempt, rng,
                          min(deadline, now + slice_s), consider)
                if best is None and clock() < deadline:
                    consider(greedy(bool(attempt & 1), attempt + 2,
                                    min(deadline, clock() + limit * 0.008)))
            elif m > 1:
                choice = rng.random()
                ranked = sorted(range(m), key=lambda s: (source_loads[s], rng.random()))
                low = ranked[min(m - 1, int((rng.random() ** 3) * max(1, math.sqrt(m))))]
                if choice < 0.38 or m == 2:
                    if rng.random() < 0.68:
                        neighbors = [s for s in (low - 1, low + 1) if 0 <= s < m]
                        other = rng.choice(neighbors)
                    else:
                        other = rng.choice([s for s in range(m) if s != low])
                    sub = region([low, other], source)
                    if sub is not None:
                        pair_slice = limit * 0.008 * (1 + math.log1p(failures) / 3)
                        _two_station(sub[0], min(deadline, clock() + pair_slice), limit,
                                     failures > 0 and rng.random() < 0.4, sub[1])
                else:
                    max_span = min(m, max(3, int(2 + math.log1p(limit) * strength)))
                    span = rng.randint(min(3, m), max_span)
                    if choice > 0.93:
                        span = m
                    lo = max(0, min(m - span, low - rng.randrange(span)))
                    stations = list(range(lo, lo + span))
                    if choice < 0.84 and rng.random() < 0.22 and span < m:
                        stations = sorted(set(stations + [ranked[-1]]))
                    sub = region(stations, source)
                    if sub is not None:
                        data, emit = sub
                        q = len(stations)
                        base_min = min(source_loads[s] for s in stations)
                        bound = min(C, sum(data[0]) // q)
                        local_step = 0
                        for value in data[0]:
                            local_step = math.gcd(local_step, value)
                        local_step = max(1, local_step)
                        bound = (bound // local_step) * local_step
                        if q == m:
                            bound = min(bound, upper)
                        use_exact = choice < 0.84 or choice > 0.965
                        if bound > base_min and use_exact:
                            target = base_min + local_step
                            # Try the smallest improvement first; after success, tighten step by step within the same remaining time slot.
                            slice_s = limit * (0.028 + 0.025 * math.log1p(q))
                            slice_s *= 1 + math.log1p(failures) / 4
                            stop = min(deadline, clock() + slice_s)
                            while target <= bound and clock() < stop:
                                if not exact(data, target, stop, emit):
                                    break
                                target += local_step
                                if best is not None:
                                    renewed = region(stations, best)
                                    if renewed is not None:
                                        data, emit = renewed
                        else:
                            target = base_min
                            if bound > base_min:
                                target = bound if rng.random() < 0.25 else base_min + local_step
                            slice_s = limit * (0.035 if q < m else 0.11)
                            slice_s *= 1 + math.log1p(failures) / 5
                            _beam(data, target, bool(attempt & 1), width, local_width, branches,
                                  attempt, rng, min(deadline, clock() + slice_s), emit)
            if clock() >= next_restart and clock() < deadline:
                consider(greedy(bool(attempt & 1), attempt, min(deadline, clock() + limit * 0.01)))
                next_restart = clock() + limit * 0.13
            failures = failures + 1 if before == best_key else 0
            attempt += 1
        return result
    except BaseException:
        return result
