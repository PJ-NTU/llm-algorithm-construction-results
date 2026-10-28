"""Adaptive subset-reassignment load balancing
Combines longest-processing-time-first construction, two-machine swaps and subset-sum reassignment, multi-machine perturbation reconstruction and small-scale time-limited branch and bound, keeping the best feasible assignment.
"""

import bisect
import heapq
import math
import random
import time


def solve(instance: dict, time_limit_s: float) -> dict:
    start = time.perf_counter()
    p = [int(v) for v in instance["processing_times"]]
    n = int(instance["num_jobs"])
    m = int(instance["num_machines"])
    # The input follows the dimension and non-negative processing time conventions of the problem statement.
    bags = [[] for _ in range(m)]
    raw_loads = [0] * m
    for j in range(n):
        i = j % m
        bags[i].append(j)
        raw_loads[i] += p[j]
    best_bags = bags[:]
    best_loads = raw_loads[:]
    divisor = 1

    def output():
        return {
            "assignment": {str(i): best_bags[i] for i in range(m)},
            "machine_completion_times": {
                str(i): float(best_loads[i] * divisor) for i in range(m)
            },
        }

    try:
        budget = max(0.0, float(time_limit_s))
        reserve = min(budget * 0.12, max(0.003, budget * 0.025,
                                        (n + m) * 0.00000025))
        deadline = start + max(0.0, budget - reserve)
        if m <= 1 or n == 0 or time.perf_counter() >= deadline:
            return output()
        g = 0
        for v in p:
            g = math.gcd(g, v)
        if g == 0:
            return output()
        weights = [v // g for v in p]
        best_loads = [v // g for v in raw_loads]
        divisor = g
        loads = best_loads[:]
        best_cost = sum(v * v for v in loads)
        current_cost = best_cost
        total = sum(weights)
        order = sorted(range(n), key=weights.__getitem__, reverse=True)
        rng = random.Random(1729)
        scale = math.log2(2.0 + budget)
        max_bins = min(m, max(2, int(2 + 2 * scale)))
        bit_limit = min(16000000, max(4096, int(120000 * (budget + 0.05) ** 0.65)))
        mitm_limit = min(38, max(12, 20 + 2 * int(scale)))
        memory_limit = int(min(96000000, 4000000 * (1 + scale)))
        slice_length = max(0.001, budget / (18 + math.sqrt(n / m + 1)))
        kick_interval = max(0.002, budget / (20 + 4 * scale))
        last_improvement = time.perf_counter()
        last_kick = last_improvement

        # Majorisation lower bound from large jobs, plus a balanced lower bound on the remaining integer load.
        remaining, machines, lower = total, m, 0
        for j in order:
            v = weights[j]
            if machines and v * machines > remaining:
                lower += v * v
                remaining -= v
                machines -= 1
            else:
                break
        if machines:
            a, r = divmod(remaining, machines)
            lower += (machines - r) * a * a + r * (a + 1) ** 2

        initial = [[] for _ in range(m)]
        initial_loads = [0] * m
        heap = [(0, i) for i in range(m)]
        heapq.heapify(heap)
        complete = True
        for k, j in enumerate(order):
            if k % 256 == 0 and time.perf_counter() >= deadline:
                complete = False
                break
            value, i = heapq.heappop(heap)
            initial[i].append(j)
            initial_loads[i] = value + weights[j]
            heapq.heappush(heap, (initial_loads[i], i))
        if complete:
            cost = sum(v * v for v in initial_loads)
            if cost <= best_cost:
                bags, loads = initial, initial_loads
                best_bags, best_loads = bags[:], loads[:]
                current_cost = best_cost = cost
        if best_cost == lower or time.perf_counter() >= deadline:
            return output()

        def subset(jobs, target, stop):
            """Find a subset not exceeding the target; keep the greedy subset when computing power is insufficient."""
            if target <= 0:
                return [], 0
            items = [j for j in jobs if 0 < weights[j] <= target]
            if not items:
                return [], 0
            rng.shuffle(items)
            chosen, attained = [], 0
            for j in items:
                if attained + weights[j] <= target:
                    chosen.append(j)
                    attained += weights[j]
            if attained == target or time.perf_counter() >= stop:
                return chosen, attained
            unit = 0
            for j in items:
                unit = math.gcd(unit, weights[j])
            t = target // unit
            values = [weights[j] // unit for j in items]
            if attained // unit == t:
                return chosen, attained
            block = math.isqrt(len(items)) + 1
            memory = (2 * block + 4) * (t + 1) // 8
            work = len(items) * (t + 1)
            available = max(0.0, stop - time.perf_counter())
            if (t <= bit_limit and memory <= memory_limit
                    and work <= 2500000000 * available):
                mask = (1 << (t + 1)) - 1
                bits = 1
                checkpoints = []
                used = 0
                for k, value in enumerate(values):
                    if k % block == 0:
                        if time.perf_counter() >= stop:
                            return chosen, attained
                        checkpoints.append(bits)
                    bits |= (bits << value) & mask
                    used = k + 1
                    if bits.bit_length() == t + 1:
                        break
                best_sum = bits.bit_length() - 1
                if best_sum * unit < attained:
                    return chosen, attained
                result = []
                residue = best_sum
                block_start = ((used - 1) // block) * block
                while block_start >= 0 and residue:
                    if time.perf_counter() >= stop:
                        return chosen, attained
                    history = []
                    bits = checkpoints[block_start // block]
                    end = min(used, block_start + block)
                    for k in range(block_start, end):
                        history.append(bits)
                        bits |= (bits << values[k]) & mask
                    for k in range(end - 1, block_start - 1, -1):
                        if not ((history[k - block_start] >> residue) & 1):
                            result.append(items[k])
                            residue -= values[k]
                    block_start -= block
                return result, best_sum * unit

            if len(items) <= mitm_limit and time.perf_counter() < stop:
                middle = len(items) // 2

                def enumerate_half(vals):
                    sums = [(0, 0)]
                    for k, value in enumerate(vals):
                        if time.perf_counter() >= stop:
                            return None
                        new = []
                        flag = 1 << k
                        for z, (s, flags) in enumerate(sums):
                            if z % 1024 == 0 and time.perf_counter() >= stop:
                                return None
                            if s + value <= t:
                                new.append((s + value, flags | flag))
                        sums.extend(new)
                    return sums

                left = enumerate_half(values[:middle])
                if left is None:
                    return chosen, attained
                right = enumerate_half(values[middle:])
                if right is None:
                    return chosen, attained
                right.sort()
                right_sums = [entry[0] for entry in right]
                best_sum, flags_left, flags_right = attained // unit, None, None
                for k, (s, flags) in enumerate(left):
                    if k % 256 == 0 and time.perf_counter() >= stop:
                        break
                    pos = bisect.bisect_right(right_sums, t - s) - 1
                    if pos >= 0 and s + right[pos][0] > best_sum:
                        best_sum = s + right[pos][0]
                        flags_left, flags_right = flags, right[pos][1]
                        if best_sum == t:
                            break
                if flags_left is not None:
                    result = [items[k] for k in range(middle)
                              if (flags_left >> k) & 1]
                    result.extend(items[middle + k] for k in range(len(items) - middle)
                                  if (flags_right >> k) & 1)
                    return result, best_sum * unit
            return chosen, attained

        def fast_pair(a, b, stop):
            if loads[a] < loads[b]:
                a, b = b, a
            difference = loads[a] - loads[b]
            if difference <= 1:
                return None
            small = sorted((weights[j], j) for j in bags[b])
            vals = [v for v, _ in small]
            gain, move = 0, None
            for k, j in enumerate(bags[a]):
                if k % 128 == 0 and time.perf_counter() >= stop:
                    break
                value = weights[j]
                change = 2 * value * (difference - value)
                if change > gain:
                    gain, move = change, (j, None)
                pos = bisect.bisect_left(vals, value - difference / 2)
                for h in (pos - 1, pos):
                    if 0 <= h < len(small):
                        v, other = small[h]
                        delta = value - v
                        change = 2 * delta * (difference - delta)
                        if change > gain:
                            gain, move = change, (j, other)
            if move is None:
                return None
            j, other = move
            ga = [v for v in bags[a] if v != j]
            gb = [v for v in bags[b] if v != other]
            gb.append(j)
            delta = weights[j]
            if other is not None:
                ga.append(other)
                delta -= weights[other]
            return [a, b], [ga, gb], [loads[a] - delta, loads[b] + delta]

        def accept(indices, groups, values, force=False):
            nonlocal current_cost, best_cost, best_bags, best_loads, last_improvement
            old = sum(loads[i] ** 2 for i in indices)
            new = sum(v * v for v in values)
            if new > old and not force:
                return False
            for i, group, value in zip(indices, groups, values):
                bags[i] = group
                loads[i] = value
            current_cost += new - old
            if current_cost < best_cost:
                best_cost = current_cost
                best_bags, best_loads = bags[:], loads[:]
                last_improvement = time.perf_counter()
            return True

        def rebuild(indices, stop, perturb=False):
            pool = [j for i in indices for j in bags[i]]
            count = len(indices)
            if not pool:
                return
            if perturb:
                rng.shuffle(pool)
                # Random-order perturbation that grows with the time limit, without changing job or machine constraints.
                noise = 0.2 + scale / (2 + scale)
                pool.sort(key=lambda j: weights[j] * (1 + noise * rng.random()),
                          reverse=True)
                groups, values = [[] for _ in indices], [0] * count
                h = [(0, i) for i in range(count)]
                heapq.heapify(h)
                for z, j in enumerate(pool):
                    if z % 256 == 0 and time.perf_counter() >= stop:
                        return
                    value, k = heapq.heappop(h)
                    groups[k].append(j)
                    values[k] = value + weights[j]
                    heapq.heappush(h, (values[k], k))
                accept(indices, groups, values, True)
                return
            groups, values = [], []
            left_total = sum(weights[j] for j in pool)
            while len(groups) < count - 1:
                now = time.perf_counter()
                if now >= stop:
                    return
                slots = count - len(groups)
                largest = max(pool, key=weights.__getitem__) if pool else None
                if largest is not None and weights[largest] * slots > left_total:
                    group, value = [largest], weights[largest]
                else:
                    target = left_total // slots
                    if slots > 2 and left_total % slots and rng.random() < 0.5:
                        target += 1
                    local_stop = now + (stop - now) / max(1, slots - 1)
                    group, value = subset(pool, target, local_stop)
                selected = set(group)
                pool = [j for j in pool if j not in selected]
                groups.append(group)
                values.append(value)
                left_total -= value
            groups.append(pool)
            values.append(left_total)
            accept(indices, groups, values)

        # Exact search on small instances keeps its search stack in a generator and shares the wall clock with neighbourhood search.
        exact_done = False
        positives = [j for j in order if weights[j] > 0]
        zeros = [j for j in order if weights[j] == 0]

        def exact_search():
            nonlocal best_bags, best_loads, best_cost, last_improvement
            local_bags = [[] for _ in range(m)]
            local_loads = [0] * m
            suffix = [0] * (len(positives) + 1)
            squares = [0] * (len(positives) + 1)
            for k in range(len(positives) - 1, -1, -1):
                value = weights[positives[k]]
                suffix[k] = suffix[k + 1] + value
                squares[k] = squares[k + 1] + value * value

            def visit(k, cost):
                nonlocal best_bags, best_loads, best_cost, last_improvement
                yield None
                if best_cost == lower:
                    return
                if k == len(positives):
                    if cost < best_cost:
                        candidate = [group[:] for group in local_bags]
                        candidate[0].extend(zeros)
                        best_bags, best_loads = candidate, local_loads[:]
                        best_cost = cost
                        last_improvement = time.perf_counter()
                    return
                levels = sorted(local_loads)
                rest = suffix[k]
                if cost + squares[k] + 2 * levels[0] * rest >= best_cost:
                    return
                prefix = 0
                high_cost = cost
                bound = cost
                for h, level in enumerate(levels):
                    prefix += level
                    high_cost -= level * level
                    width = h + 1
                    water = rest + prefix
                    if width == m or water <= width * levels[width]:
                        q, r = divmod(water, width)
                        bound = high_cost + (width - r) * q * q + r * (q + 1) ** 2
                        break
                if bound >= best_cost:
                    return
                j = positives[k]
                value = weights[j]
                seen = set()
                choices = sorted(range(m), key=local_loads.__getitem__)
                if k + 1 == len(positives):
                    choices = choices[:1]
                for i in choices:
                    old = local_loads[i]
                    if old in seen:
                        continue
                    seen.add(old)
                    new_cost = cost + value * (2 * old + value)
                    if new_cost >= best_cost:
                        continue
                    local_loads[i] += value
                    local_bags[i].append(j)
                    yield from visit(k + 1, new_cost)
                    local_bags[i].pop()
                    local_loads[i] = old
            yield from visit(0, 0)

        exact = None
        if len(positives) <= mitm_limit and m <= max(2, int(2 + scale)):
            exact = exact_search()
        exact_clock = start
        while time.perf_counter() < deadline and best_cost > lower:
            now = time.perf_counter()
            if exact is not None and now >= exact_clock:
                stop = min(deadline, now + slice_length / (2 + scale))
                while time.perf_counter() < stop:
                    try:
                        next(exact)
                    except StopIteration:
                        exact_done = True
                        break
                if exact_done or best_cost == lower:
                    break
                exact_clock = time.perf_counter() + slice_length
            now = time.perf_counter()
            if now >= deadline:
                break
            if current_cost < best_cost:
                best_bags, best_loads = bags[:], loads[:]
                best_cost = current_cost
            if now - max(last_improvement, last_kick) >= kick_interval:
                bags, loads = best_bags[:], best_loads[:]
                current_cost = best_cost
                indices = rng.sample(range(m), max_bins)
                rebuild(indices, min(deadline, now + slice_length), True)
                last_kick = time.perf_counter()
                continue
            ranked = sorted(range(m), key=loads.__getitem__)
            spread = max(1, min(m, int(math.sqrt(m) * (1 + scale / 4))))
            if rng.random() < 0.55:
                a, b = ranked[-1], ranked[0]
            else:
                a = ranked[m - 1 - rng.randrange(spread)]
                b = ranked[rng.randrange(spread)]
            if a == b:
                b = (a + 1 + rng.randrange(m - 1)) % m
            stop = min(deadline, time.perf_counter() + slice_length)
            if m == 2 or rng.random() < 0.6:
                fast = fast_pair(a, b, stop)
                if fast is not None:
                    accept(*fast)
                if best_cost == lower:
                    break
                if time.perf_counter() >= stop:
                    continue
                jobs = bags[a] + bags[b]
                target = (loads[a] + loads[b]) // 2
                group, value = subset(jobs, target, stop)
                selected = set(group)
                other = [j for j in jobs if j not in selected]
                accept([a, b], [group, other],
                       [value, loads[a] + loads[b] - value])
            else:
                stale = (time.perf_counter() - last_improvement) / kick_interval
                upper = min(max_bins, max(3, int(3 + stale * scale)))
                count = rng.randint(3, upper)
                indices = [a, b]
                chosen = {a, b}
                while len(indices) < count and time.perf_counter() < stop:
                    i = rng.randrange(m)
                    if i not in chosen:
                        chosen.add(i)
                        indices.append(i)
                if len(indices) == count:
                    rebuild(indices, stop)
        return output()
    except Exception:
        # When any search branch is interrupted, fall back to the already fully saved feasible solution.
        return output()
