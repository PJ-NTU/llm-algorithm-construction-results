"""Differencing-guided adaptive subset reassignment
Combines longest-processing-time-first, compressed-bitset and meet-in-the-middle reassignment, multi-machine differencing construction, continual exact search and time-limited perturbation, saving the best feasible assignment.
"""

import bisect
import heapq
import math
import random
import time


def solve(instance: dict, time_limit_s: float) -> dict:
    started = time.perf_counter()
    m = int(instance["num_machines"])
    n = int(instance["num_jobs"])
    original = [int(v) for v in instance["processing_times"]]
    # For valid input, first build a complete backup solution; the search replaces only complete buckets.
    initial = [list(range(n))] + [[] for _ in range(m - 1)]
    best = [tuple(b) for b in initial]

    def result():
        return {
            "assignment": {str(i): list(best[i]) for i in range(m)},
            "machine_completion_times": {
                str(i): float(sum(original[j] for j in best[i]))
                for i in range(m)
            },
        }

    try:
        budget = max(0.0, float(time_limit_s))
        if not math.isfinite(budget):
            budget = 0.0
        reserve = min(budget * 0.15, max(budget * 0.035, 0.002,
                                       (n + m) * 0.0000005))
        deadline = started + max(0.0, budget - reserve)
        if m <= 1 or not n or time.perf_counter() >= deadline:
            return result()
        positive = [j for j in range(n) if original[j] > 0]
        zeros = tuple(j for j in range(n) if original[j] == 0)
        if len(positive) <= m:
            best = [(j,) for j in positive] + [()] * (m - len(positive))
            best[0] += zeros
            return result()
        unit = 0
        for j in positive:
            unit = math.gcd(unit, original[j])
        p = [v // unit for v in original]
        order = sorted(positive, key=p.__getitem__, reverse=True)
        bags = [[] for _ in range(m)]
        loads = [0] * m
        heap = [(0, i) for i in range(m)]
        for h, j in enumerate(order):
            if h % 128 == 0 and time.perf_counter() >= deadline:
                return result()
            load, i = heapq.heappop(heap)
            bags[i].append(j)
            loads[i] = load + p[j]
            heapq.heappush(heap, (loads[i], i))
        bags = [tuple(reversed(b)) for b in bags]
        bags[0] = zeros + bags[0]
        score = sum(v * v for v in loads)
        old_score = sum(sum(p[j] for j in b) ** 2 for b in best)
        if score <= old_score:
            best = bags[:]
        best_loads = [sum(p[j] for j in b) for b in best]
        best_score = sum(v * v for v in best_loads)

    # The total load is fixed, so the sum of squares orders solutions exactly as the normalised root variance under a positive mean.
    # A large job exceeding the remaining average may occupy a machine alone in an optimal assignment.
        remaining, count, lower = sum(p), m, 0
        solitary = set()
        for j in order:
            v = p[j]
            if count <= 1 or v * count <= remaining:
                break
            solitary.add(j)
            lower += v * v
            remaining -= v
            count -= 1
        q, r = divmod(remaining, count)
        lower += (count - r) * q * q + r * (q + 1) ** 2
        if best_score == lower:
            return result()
        active = []
        for i, bucket in enumerate(bags):
            nonzero = [j for j in bucket if p[j]]
            if len(nonzero) != 1 or nonzero[0] not in solitary:
                active.append(i)
        k_active = len(active)
        if k_active < 2:
            return result()
        rng = random.Random(32452843 + 1009 * n + m)
        effort = math.log1p(budget)
        density = len(positive) / m
        width = min(k_active, max(2, int(2 + 2 * effort + math.sqrt(budget))))
        depth = max(2, min(700, int(12 + 5 * effort + math.log1p(density))))
        quantum = max(0.0005, budget / (50 + 10 * math.sqrt(m) + 4 * effort))
        kick_period = max(quantum, budget / (12 + effort + math.sqrt(density)))
        memory_bytes = int(8_000_000 * (1 + effort))
        version = [0] * m
        failed_pairs = {}
        last_best = last_kick = time.perf_counter()

        def install(ids, groups, values=None, force=False, equal=False):
            nonlocal score, best_score, best, best_loads, last_best
            if values is None:
                values = [sum(p[j] for j in b) for b in groups]
            delta = sum(v * v - loads[i] * loads[i]
                        for i, v in zip(ids, values))
            if not force and (delta > 0 or (delta == 0 and not equal)):
                return False
            candidate = [tuple(sorted(b, key=p.__getitem__)) for b in groups]
            for i, b, v in zip(ids, candidate, values):
                bags[i], loads[i] = b, v
                version[i] += 1
            score += delta
            if score < best_score:
                best_score = score
                best, best_loads = bags[:], loads[:]
                last_best = time.perf_counter()
            return True

        def subset(jobs, target2, stop):
            """Subset closest to target2 / 2; also reports whether the exact solve completed."""
            if time.perf_counter() >= stop:
                return None
            jobs = [j for j in jobs if p[j] > 0]
            total = sum(p[j] for j in jobs)
            if target2 <= 0:
                return [], True
            if target2 >= 2 * total:
                return jobs, True
            by_value = {}
            factor = 0
            for j in jobs:
                factor = math.gcd(factor, p[j])
                by_value.setdefault(p[j], []).append(j)
            chunks = []
            for value, members in by_value.items():
                offset, size = 0, 1
                while offset < len(members):
                    end = min(len(members), offset + size)
                    chunks.append((value // factor * (end - offset), members[offset:end]))
                    offset, size = end, size * 2
            half = target2 == total
            denominator = 2 * factor
            target = target2 // denominator
            cap = (total // (2 * factor) if half else
                   min(total // factor, target + max(by_value) // factor + 1))
            block = math.isqrt(len(chunks)) + 1
            copies = (len(chunks) + block - 1) // block + block + 4
            available = max(0.0, stop - time.perf_counter())
            if (cap * len(chunks) <= available * 450_000_000
                    and cap * copies <= 8 * memory_bytes
                    and cap <= 2_000_000 * (1 + effort)):
                rng.shuffle(chunks)
                mask = (1 << (cap + 1)) - 1
                bits = 1
                checkpoints = []
                used = 0
                ideal = min(cap, (target2 + factor) // denominator)
                for h, (v, _) in enumerate(chunks):
                    if time.perf_counter() >= stop:
                        return None
                    if h % block == 0:
                        checkpoints.append(bits)
                    if v <= cap:
                        bits = (bits | (bits << v)) & mask
                    used = h + 1
                    if (bits >> ideal) & 1:
                        break
                below = (bits & ((1 << (min(target, cap) + 1)) - 1)).bit_length() - 1
                attained = below
                if target + 1 <= cap:
                    tail = bits >> (target + 1)
                    if tail:
                        above = target + 1 + (tail & -tail).bit_length() - 1
                        if abs(above * denominator - target2) < abs(below * denominator - target2):
                            attained = above
                chosen = []
                residue = attained
                start_block = ((used - 1) // block) * block
                while start_block >= 0 and residue:
                    if time.perf_counter() >= stop:
                        return None
                    history = []
                    bits = checkpoints[start_block // block]
                    end = min(used, start_block + block)
                    for h in range(start_block, end):
                        history.append(bits)
                        v = chunks[h][0]
                        if v <= cap:
                            bits = (bits | (bits << v)) & mask
                    for h in range(end - 1, start_block - 1, -1):
                        before = history[h - start_block]
                        v, members = chunks[h]
                        skip = (before >> residue) & 1
                        take = residue >= v and ((before >> (residue - v)) & 1)
                        if take and (not skip or rng.getrandbits(1)):
                            chosen.extend(members)
                            residue -= v
                    start_block -= block
                return chosen, True

            # For large values, the size of meet-in-the-middle enumeration is limited jointly by remaining wall clock and memory budget.
            capacity = min(memory_bytes // 160, int(available * 160_000))
            split = len(jobs) // 2
            if capacity < 2 or len(jobs) - split > capacity.bit_length() - 1:
                return None
            rng.shuffle(jobs)
            tables = []
            for side in (jobs[:split], jobs[split:]):
                entries = [(0, 0)]
                for h, j in enumerate(side):
                    flag, v = 1 << h, p[j]
                    old_length = len(entries)
                    for z in range(old_length):
                        if z % 256 == 0 and time.perf_counter() >= stop:
                            return None
                        value, code = entries[z]
                        entries.append((value + v, code | flag))
                entries.sort()
                tables.append(entries)
                if time.perf_counter() >= stop:
                    return None
            left, right = tables
            a, b = 0, len(right) - 1
            error, codes = None, (0, 0)
            while a < len(left) and b >= 0:
                if (a + b) % 128 == 0 and time.perf_counter() >= stop:
                    return None
                delta = 2 * (left[a][0] + right[b][0]) - target2
                if error is None or abs(delta) < error:
                    error, codes = abs(delta), (left[a][1], right[b][1])
                    if error <= target2 % 2:
                        break
                if delta < 0:
                    a += 1
                else:
                    b -= 1
            chosen = [j for h, j in enumerate(jobs[:split]) if (codes[0] >> h) & 1]
            chosen.extend(j for h, j in enumerate(jobs[split:]) if (codes[1] >> h) & 1)
            return chosen, True

        def differencing(jobs, count, stop):
            """Merge load vectors by span, compress differences by reverse-order pairing, then restore the jobs."""
            queue = []
            serial = 0
            for h, j in enumerate(jobs):
                if h % 128 == 0 and time.perf_counter() >= stop:
                    return None
                values = (0,) * (count - 1) + (p[j],)
                queue.append((-p[j], rng.random(), serial, values, j))
                serial += 1
            if not queue:
                return [[] for _ in range(count)]
            heapq.heapify(queue)
            while len(queue) > 1:
                if time.perf_counter() >= stop:
                    return None
                _, _, _, a, left = heapq.heappop(queue)
                _, _, _, b, right = heapq.heappop(queue)
                merged = [a[h] + b[count - 1 - h] for h in range(count)]
                perm = sorted(range(count), key=merged.__getitem__)
                values = tuple(merged[h] for h in perm)
                node = (left, right, perm)
                heapq.heappush(queue, (values[0] - values[-1], rng.random(),
                                       serial, values, node))
                serial += 1
            groups = [[] for _ in range(count)]
            stack = [(queue[0][4], list(range(count)))]
            while stack:
                if time.perf_counter() >= stop:
                    return None
                node, mapping = stack.pop()
                if isinstance(node, int):
                    groups[mapping[-1]].append(node)
                else:
                    left, right, perm = node
                    ma, mb = [0] * count, [0] * count
                    for h, v in enumerate(perm):
                        ma[v] = mapping[h]
                        mb[count - 1 - v] = mapping[h]
                    stack.append((left, ma))
                    stack.append((right, mb))
            return groups

        def fast_pair(i, j, stop):
            if loads[i] < loads[j]:
                i, j = j, i
            difference = loads[i] - loads[j]
            if difference <= 1:
                return False
            other = bags[j]
            vals = [p[x] for x in other]
            gain, move = 0, None
            previous = -1
            for h, job in enumerate(bags[i]):
                if h % 64 == 0 and time.perf_counter() >= stop:
                    break
                v = p[job]
                if v == previous:
                    continue
                previous = v
                change = v * (difference - v)
                if change > gain:
                    gain, move = change, (job, None)
                at = bisect.bisect_left(vals, (2 * v - difference + 1) // 2)
                for pos in (at - 1, at):
                    if 0 <= pos < len(other):
                        delta = v - vals[pos]
                        change = delta * (difference - delta)
                        if change > gain:
                            gain, move = change, (job, other[pos])
            if move is None:
                return False
            job, swap = move
            first = [x for x in bags[i] if x != job]
            second = [x for x in bags[j] if x != swap]
            second.append(job)
            if swap is not None:
                first.append(swap)
            return install((i, j), (first, second))

        def pair(i, j, stop):
            if fast_pair(i, j, stop):
                return False
            key = (min(i, j), max(i, j))
            state = (version[key[0]], version[key[1]])
            if failed_pairs.get(key) == state:
                return k_active == 2
            pool = list(bags[i] + bags[j])
            answer = subset(pool, loads[i] + loads[j], stop)
            if answer is not None:
                chosen, certified = answer
                chosen = set(chosen)
                groups = ([x for x in pool if x in chosen],
                          [x for x in pool if x not in chosen])
                changed = install((i, j), groups)
                if not changed and certified:
                    failed_pairs[key] = state
                return certified and k_active == 2
            if time.perf_counter() >= stop:
                return False
            # Fix the base load and draw only a job pool that can be solved exactly within the current time slice.
            available = max(0.0, stop - time.perf_counter())
            capacity = max(2, min(memory_bytes // 160, int(available * 100_000)))
            pool_size = min(len(pool), 2 * (capacity.bit_length() - 1))
            chosen_pool = rng.sample(pool, pool_size)
            selected = set(chosen_pool)
            kept_i = [x for x in bags[i] if x not in selected]
            kept_j = [x for x in bags[j] if x not in selected]
            base_i = sum(p[x] for x in kept_i)
            base_j = sum(p[x] for x in kept_j)
            target2 = base_j + sum(p[x] for x in chosen_pool) - base_i
            answer = subset(chosen_pool, target2, stop)
            if answer is not None:
                chosen = set(answer[0])
                install((i, j), (kept_i + [x for x in chosen_pool if x in chosen],
                                 kept_j + [x for x in chosen_pool if x not in chosen]))
            return False

        def select_machines(count):
            if count >= k_active:
                return active[:]
            sample_size = min(k_active, max(count, int(math.sqrt(k_active) * (2 + effort))))
            sample = rng.sample(active, sample_size)
            sample.sort(key=loads.__getitem__)
            return [sample[0], sample[-1]] + rng.sample(sample[1:-1], count - 2)

        def rebuild(ids, stop, use_difference=False):
            pool = [j for i in ids for j in bags[i]]
            if use_difference:
                candidate = differencing(pool, len(ids), stop)
                if candidate is not None:
                    install(ids, candidate, equal=True)
                return
            groups = []
            remaining = sum(p[j] for j in pool)
            while len(groups) + 1 < len(ids):
                if time.perf_counter() >= stop:
                    return
                slots = len(ids) - len(groups)
                if not pool:
                    groups.append([])
                    continue
                anchor = max(pool, key=p.__getitem__)
                target, r = divmod(remaining, slots)
                if slots > 2 and rng.randrange(slots) < r:
                    target += 1
                if p[anchor] > target and slots > 2:
                    chosen = {anchor}
                else:
                    if slots == 2:
                        jobs = pool
                        target2 = remaining
                    else:
                        jobs = [j for j in pool if j != anchor]
                        target2 = 2 * (target - p[anchor])
                    now = time.perf_counter()
                    answer = subset(jobs, target2, now + (stop - now) / (slots - 1))
                    if answer is None:
                        return
                    chosen = set(answer[0])
                    if slots > 2:
                        chosen.add(anchor)
                group = [j for j in pool if j in chosen]
                pool = [j for j in pool if j not in chosen]
                groups.append(group)
                remaining -= sum(p[j] for j in group)
            groups.append(pool)
            install(ids, groups, equal=True)

        def shake(pressure):
            nonlocal bags, loads, score
            # Immutable buckets allow cheap restoration of the best assignment; perturbation only changes the current trajectory.
            bags, loads, score = best[:], best_loads[:], best_score
            failed_pairs.clear()
            for i in active:
                version[i] += 1
            count = min(k_active, max(2, int(2 + rng.random() * width * pressure)))
            ids = select_machines(count)
            pool = [j for i in ids for j in bags[i] if p[j]]
            take = min(len(pool), max(2, int((1 + effort) * pressure * math.sqrt(density))))
            removed = rng.sample(pool, take)
            selected = set(removed)
            groups = [[j for j in bags[i] if j not in selected] for i in ids]
            values = [sum(p[j] for j in b) for b in groups]
            noise = min(1.0, pressure / (2 + effort))
            removed.sort(key=lambda j: p[j] * (1 + noise * rng.random()), reverse=True)
            queue = [(v, rng.random(), h) for h, v in enumerate(values)]
            heapq.heapify(queue)
            for j in removed:
                if time.perf_counter() >= deadline:
                    return
                v, _, h = heapq.heappop(queue)
                groups[h].append(j)
                values[h] = v + p[j]
                heapq.heappush(queue, (values[h], rng.random(), h))
            install(ids, groups, values, force=True)

        def complete_differencing():
            """Two-machine complete differencing: the difference and sum branches exhaust all partitions; a generator keeps the search stack."""
            ids = active[:]
            jobs = [j for i in ids for j in bags[i] if p[j]]
            zero_jobs = [j for i in ids for j in bags[i] if not p[j]]
            entries = sorted(((p[j], j) for j in jobs), key=lambda x: x[0], reverse=True)
            seen = set()
            capacity = int(20_000 * (1 + effort))

            def publish(node):
                groups = [[], []]
                stack = [(node, 1)]
                while stack:
                    v, sign = stack.pop()
                    if isinstance(v, int):
                        groups[0 if sign > 0 else 1].append(v)
                    else:
                        a, b, direction = v
                        stack.append((a, sign))
                        stack.append((b, sign * direction))
                groups[0].extend(zero_jobs)
                install(ids, groups)

            def visit(state, total):
                yield None
                gap = abs(best_loads[ids[0]] - best_loads[ids[1]])
                largest = state[0][0]
                if max(2 * largest - total, total % 2) >= gap:
                    return
                signature = tuple(v for v, _ in state)
                if signature in seen:
                    return
                if len(seen) < capacity:
                    seen.add(signature)
                if 2 * largest >= total:
                    node = state[0][1]
                    for _, other in state[1:]:
                        node = (node, other, -1)
                    publish(node)
                    return
                (a, left), (b, right) = state[:2]
                tail = state[2:]
                for value, sign in ((a - b, -1), (a + b, 1)):
                    entry = (value, (left, right, sign))
                    at = 0
                    while at < len(tail) and tail[at][0] > value:
                        at += 1
                    child = tail[:at] + [entry] + tail[at:]
                    yield from visit(child, total - a - b + value)
            yield from visit(entries, sum(p[j] for j in jobs))

        def exact_assignment():
            """Symmetry-breaking branch and bound for small job sets, sharing the best upper bound with local search."""
            jobs = sorted((j for i in active for j in bags[i] if p[j]),
                          key=p.__getitem__, reverse=True)
            zero_jobs = [j for i in active for j in bags[i] if not p[j]]
            count = len(active)
            suffix = [0] * (len(jobs) + 1)
            squares = suffix[:]
            for h in range(len(jobs) - 1, -1, -1):
                suffix[h] = suffix[h + 1] + p[jobs[h]]
                squares[h] = squares[h + 1] + p[jobs[h]] ** 2
            fixed = score - sum(loads[i] ** 2 for i in active)
            local = [0] * count
            groups = [[] for _ in range(count)]
            visited = set()
            capacity = int(20_000 * (1 + effort))

            def visit(h, cost):
                yield None
                bound = best_score - fixed
                if h == len(jobs):
                    if cost < bound:
                        candidate = [b[:] for b in groups]
                        candidate[0].extend(zero_jobs)
                        install(active, candidate, local[:])
                    return
                levels = sorted(local)
                rest = suffix[h]
                if cost + squares[h] + 2 * levels[0] * rest >= bound:
                    return
                high_cost, prefix = cost, 0
                for z, value in enumerate(levels):
                    prefix += value
                    high_cost -= value * value
                    width0 = z + 1
                    mass = prefix + rest
                    if width0 == count or mass <= width0 * levels[width0]:
                        q, r = divmod(mass, width0)
                        relaxed = high_cost + (width0 - r) * q * q + r * (q + 1) ** 2
                        if relaxed >= bound:
                            return
                        break
                state = (h, tuple(levels))
                if state in visited:
                    return
                if len(visited) < capacity:
                    visited.add(state)
                job, seen = jobs[h], set()
                choices = sorted(range(count), key=local.__getitem__)
                if h + 1 == len(jobs):
                    choices = choices[:1]
                for i in choices:
                    old = local[i]
                    if old in seen:
                        continue
                    seen.add(old)
                    updated = cost + p[job] * (2 * old + p[job])
                    if updated >= best_score - fixed:
                        continue
                    local[i] += p[job]
                    groups[i].append(job)
                    yield from visit(h + 1, updated)
                    groups[i].pop()
                    local[i] = old
            yield from visit(0, 0)

        # The initial differencing candidate does not replace longest-processing-time-first; it is committed only if the true sum of squares improves.
        ids = select_machines(width)
        pool = [j for i in ids for j in bags[i]]
        candidate = differencing(pool, len(ids), min(deadline, time.perf_counter() + 2 * quantum))
        if candidate is not None:
            install(ids, candidate)
        active_n = sum(sum(p[j] > 0 for j in bags[i]) for i in active)
        exact = None
        if k_active == 2 and active_n <= min(700, int(24 + 24 * (1 + effort))):
            exact = complete_differencing()
        elif active_n <= depth and k_active <= width:
            exact = exact_assignment()
        exact_clock = time.perf_counter()

        # No cap on restarts or iterations; only the wall-clock deadline or an optimality proof terminates the search.
        while time.perf_counter() < deadline and best_score > lower:
            now = time.perf_counter()
            if exact is not None and now >= exact_clock:
                stop = min(deadline, now + quantum / (2 + effort))
                exhausted = False
                while time.perf_counter() < stop and best_score > lower:
                    try:
                        next(exact)
                    except StopIteration:
                        exhausted = True
                        break
                if exhausted or best_score == lower:
                    break
                exact_clock = time.perf_counter() + quantum
            now = time.perf_counter()
            if now >= deadline:
                break
            pressure = 1 + math.log1p((now - last_best) / max(quantum, kick_period))
            if now - max(last_best, last_kick) >= kick_period:
                shake(pressure)
                last_kick = time.perf_counter()
                continue
            stop = min(deadline, now + quantum * math.sqrt(pressure))
            if k_active == 2 or rng.random() < 0.62:
                ids = select_machines(2)
                if pair(ids[0], ids[1], stop):
                    break
            else:
                count = rng.randint(3, min(k_active, max(3, int(width * math.sqrt(pressure)))))
                ids = select_machines(count)
                rebuild(ids, stop, use_difference=rng.random() < 0.35)
    except Exception:
        # Interruption, insufficient resources or a failed search branch never overwrites the already fully saved feasible solution.
        pass
    return result()
