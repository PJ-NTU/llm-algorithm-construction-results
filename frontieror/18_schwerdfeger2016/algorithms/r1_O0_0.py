"""Adaptive subset-sum large neighbourhood load balancing
Uses a longest-processing-time-first initial solution, two-machine subset-sum reassignment, time-limited branch and bound and multi-machine ruin-and-recreate, keeping the current best feasible solution.
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
    p = [int(v) for v in instance["processing_times"]]
    groups = [[] for _ in range(m)]
    loads = [0] * m
    order = sorted(range(n), key=p.__getitem__, reverse=True)
    heap = [(0, i) for i in range(m)]
    for j in order:
        load, i = heapq.heappop(heap)
        groups[i].append(j)
        loads[i] = load + p[j]
        heapq.heappush(heap, (loads[i], i))

    def output():
        return {
            "assignment": {str(i): groups[i] for i in range(m)},
            "machine_completion_times": {
                str(i): float(sum(p[j] for j in groups[i])) for i in range(m)
            },
        }

    try:
        budget = max(0.0, float(time_limit_s))
        if not math.isfinite(budget):
            budget = 0.0
        reserve = max(0.002, budget * 0.025, (n + m) * 0.000001)
        end = started + max(0.0, budget - reserve)
        if m <= 1 or n <= m or time.perf_counter() >= end:
            return output()
        common = 0
        for value in p:
            common = math.gcd(common, value)
        if common == 0:
            return output()
        w = [value // common for value in p]
        loads = [value // common for value in loads]
        positive = [j for j in order if w[j] > 0]
        total = sum(w)
        # A job larger than the remaining average load necessarily occupies a machine alone in an optimal solution.
        frozen_jobs = set()
        remaining_total = total
        remaining_machines = m
        for j in positive:
            if w[j] * remaining_machines <= remaining_total:
                break
            frozen_jobs.add(j)
            remaining_total -= w[j]
            remaining_machines -= 1
            if remaining_machines == 0:
                return output()
        active = [i for i in range(m)
                  if not any(j in frozen_jobs for j in groups[i])]
        if len(active) <= 1:
            return output()
        # Zero-duration jobs do not take part in the search and are added back at the end, to avoid enlarging the neighbourhood.
        zero_jobs = [j for j in order if w[j] == 0]
        work = [[j for j in groups[i] if w[j] > 0] for i in active]
        weight = [sum(w[j] for j in jobs) for jobs in work]
        size = len(work)
        q, r = divmod(sum(weight), size)
        lower = (size - r) * q * q + r * (q + 1) * (q + 1)
        score = sum(value * value for value in weight)
        rng = random.Random(641729)
        strength = math.log1p(budget)
        memory_bits = int(8 * 1024 * 1024 * min(16.0, 1.0 + strength))
        exact_depth = max(2, min(160, int(10 + 7 * strength)))

        def subset(jobs, target, deadline):
            """Find a subset not exceeding the target; exact when resources allow, otherwise keep the greedy subset."""
            if target <= 0:
                return []
            chosen = []
            value = 0
            for j in sorted(jobs, key=w.__getitem__, reverse=True):
                if value + w[j] <= target:
                    chosen.append(j)
                    value += w[j]
            if value == target or time.perf_counter() >= deadline:
                return chosen
            eligible = [j for j in jobs if w[j] <= target]
            divisor = 0
            for j in eligible:
                divisor = math.gcd(divisor, w[j])
            if not divisor:
                return chosen
            cap = target // divisor
            by_value = {}
            for j in eligible:
                by_value.setdefault(w[j] // divisor, []).append(j)
            chunks = []
            for v, ids in by_value.items():
                offset, block = 0, 1
                while offset < len(ids):
                    take = min(block, len(ids) - offset)
                    chunks.append((v * take, ids[offset:offset + take]))
                    offset += take
                    block *= 2
            rng.shuffle(chunks)
            left = deadline - time.perf_counter()
            effort = cap * max(1, len(chunks))
            if left > 0 and effort <= min(memory_bits, left * 350000000):
                mask = (1 << (cap + 1)) - 1
                bits = 1
                history = []
                for v, ids in chunks:
                    if time.perf_counter() >= deadline:
                        return chosen
                    if v <= cap:
                        history.append((bits, v, ids))
                        bits |= (bits << v) & mask
                        if (bits >> cap) & 1:
                            break
                best_sum = bits.bit_length() - 1
                if best_sum * divisor < value:
                    return chosen
                result = []
                for before, v, ids in reversed(history):
                    if not ((before >> best_sum) & 1):
                        result.extend(ids)
                        best_sum -= v
                    if best_sum == 0:
                        break
                return result
            # When the numeric range is large but there are few jobs, switch to meet-in-the-middle enumeration.
            left = deadline - time.perf_counter()
            capacity = int(min(memory_bits / 1024, max(0.0, left) * 180000))
            count = len(eligible)
            half = count // 2
            if capacity < 2 or count > 2 * int(math.log2(capacity)):
                return chosen
            if (1 << (count - half)) > capacity:
                return chosen
            halves = []
            for ids in (eligible[:half], eligible[half:]):
                states = [(0, 0)]
                for bit, j in enumerate(ids):
                    if time.perf_counter() >= deadline:
                        return chosen
                    flag = 1 << bit
                    addition = [(s + w[j], code | flag) for s, code in states
                                if s + w[j] <= target]
                    states.extend(addition)
                halves.append(states)
            first, second = halves
            first.sort()
            sums = [item[0] for item in first]
            best_sum, codes = value, None
            for index, (s, code) in enumerate(second):
                if index % 256 == 0 and time.perf_counter() >= deadline:
                    break
                pos = bisect.bisect_right(sums, target - s) - 1
                if pos >= 0 and s + sums[pos] > best_sum:
                    best_sum = s + sums[pos]
                    codes = (first[pos][1], code)
                    if best_sum == target:
                        break
            if codes is None:
                return chosen
            return [j for bit, j in enumerate(eligible[:half])
                    if (codes[0] >> bit) & 1] + [
                        j for bit, j in enumerate(eligible[half:])
                        if (codes[1] >> bit) & 1]

        def pair(a, b, deadline):
            sa, sb = sum(w[j] for j in a), sum(w[j] for j in b)
            if sa < sb:
                a, b, sa, sb = b, a, sb, sa
            difference = sa - sb
            if difference <= 1:
                return a, b
            # First do the best single-job move or one-for-one swap, as a fallback for sparse numeric instances.
            ordered_b = sorted((w[j], j) for j in b)
            values = [v for v, _ in ordered_b]
            best_diff, change = difference, None
            for index, j in enumerate(a):
                if index % 128 == 0 and time.perf_counter() >= deadline:
                    break
                d = abs(difference - 2 * w[j])
                if d < best_diff:
                    best_diff, change = d, (j, None)
                pos = bisect.bisect_left(values, (2 * w[j] - difference) // 2)
                for location in (pos - 1, pos, pos + 1):
                    if 0 <= location < len(ordered_b):
                        v, other = ordered_b[location]
                        d = abs(difference - 2 * (w[j] - v))
                        if d < best_diff:
                            best_diff, change = d, (j, other)
                if best_diff <= 1:
                    break
            if change is not None:
                j, other = change
                aa = [v for v in a if v != j]
                bb = [v for v in b if v != other]
                bb.append(j)
                if other is not None:
                    aa.append(other)
            else:
                aa, bb = a, b
            if best_diff > 1 and time.perf_counter() < deadline:
                pool = a + b
                selected = subset(pool, (sa + sb) // 2, deadline)
                new_sum = sum(w[j] for j in selected)
                if abs(sa + sb - 2 * new_sum) <= best_diff:
                    members = set(selected)
                    aa = selected
                    bb = [j for j in pool if j not in members]
            return aa, bb

        def exact(parts, deadline):
            """Time-limited, symmetry-breaking branch and bound over a limited machine neighbourhood."""
            jobs = sorted((j for part in parts for j in part),
                          key=w.__getitem__, reverse=True)
            k = len(parts)
            best = [part[:] for part in parts]
            best_score = sum(sum(w[j] for j in part) ** 2 for part in parts)
            s = sum(w[j] for j in jobs)
            quotient, remainder = divmod(s, k)
            bound = (k - remainder) * quotient ** 2 + remainder * (quotient + 1) ** 2
            if best_score == bound or len(jobs) > exact_depth:
                return best, best_score == bound
            tail = [0] * (len(jobs) + 1)
            for pos in range(len(jobs) - 1, -1, -1):
                tail[pos] = tail[pos + 1] + w[jobs[pos]]
            current = [[] for _ in range(k)]
            current_load = [0] * k
            interrupted = False

            def visit(pos):
                nonlocal best, best_score, interrupted
                if interrupted or best_score == bound:
                    return
                if time.perf_counter() >= deadline:
                    interrupted = True
                    return
                if pos == len(jobs):
                    value = sum(v * v for v in current_load)
                    if value < best_score:
                        best_score = value
                        best = [part[:] for part in current]
                    return
                levels = sorted(current_load)
                fixed_square = sum(v * v for v in levels)
                prefix = 0
                for c, v in enumerate(levels, 1):
                    prefix += v
                    fixed_square -= v * v
                    mass = prefix + tail[pos]
                    if c == k or mass <= c * levels[c]:
                        base, extra = divmod(mass, c)
                        relaxed = fixed_square + (c - extra) * base ** 2 + extra * (base + 1) ** 2
                        if relaxed >= best_score:
                            return
                        break
                j = jobs[pos]
                seen = set()
                for i in sorted(range(k), key=current_load.__getitem__):
                    old = current_load[i]
                    if old in seen:
                        continue
                    seen.add(old)
                    current_load[i] += w[j]
                    current[i].append(j)
                    visit(pos + 1)
                    current[i].pop()
                    current_load[i] = old
                    if interrupted or best_score == bound:
                        return

            visit(0)
            return best, not interrupted or best_score == bound

        def accept(indices, candidate):
            nonlocal score
            new_load = [sum(w[j] for j in part) for part in candidate]
            old_score = sum(weight[i] ** 2 for i in indices)
            new_score = sum(v * v for v in new_load)
            if new_score > old_score:
                return False
            for i, part, value in zip(indices, candidate, new_load):
                work[i] = part
                weight[i] = value
            score += new_score - old_score
            return new_score < old_score

        def rebuild(indices, deadline, stagnation):
            old = [work[i] for i in indices]
            k = len(indices)
            pool = [j for part in old for j in part]
            if rng.random() < 0.5:
                # Sequential subset-sum packing, recomputing the target from the remaining load each time.
                candidate = []
                residual = sum(w[j] for j in pool)
                while len(candidate) < k - 1 and pool and time.perf_counter() < deadline:
                    slots = k - len(candidate)
                    largest = max(pool, key=w.__getitem__)
                    target = residual // slots
                    if w[largest] > target:
                        chosen = [largest]
                    else:
                        now = time.perf_counter()
                        chosen = subset(pool, target, now + max(0.0, deadline - now) / slots)
                    chosen_set = set(chosen)
                    candidate.append(chosen)
                    pool = [j for j in pool if j not in chosen_set]
                    residual -= sum(w[j] for j in chosen)
                candidate.extend([[] for _ in range(k - len(candidate))])
                candidate_load = [sum(w[j] for j in part) for part in candidate]
            else:
                # Keep part of the original assignment, remove jobs at random, then reinsert them by perturbed lengths.
                fraction = min(1.0, 1.0 / math.sqrt(1.0 + strength) + rng.random())
                candidate, pool = [], []
                for part in old:
                    retained = []
                    for j in part:
                        if rng.random() < fraction:
                            pool.append(j)
                        else:
                            retained.append(j)
                    candidate.append(retained)
                candidate_load = [sum(w[j] for j in part) for part in candidate]
            noise = min(0.7, math.log1p(stagnation) / (8.0 + strength)) * rng.random()
            pool.sort(key=lambda j: w[j] * (1.0 + noise * rng.random()), reverse=True)
            queue = [(v, i) for i, v in enumerate(candidate_load)]
            heapq.heapify(queue)
            for j in pool:
                value, i = heapq.heappop(queue)
                candidate[i].append(j)
                candidate_load[i] = value + w[j]
                heapq.heappush(queue, (candidate_load[i], i))
            # Within the remaining neighbourhood time, eliminate improvable partitions between the heaviest and lightest machines.
            while time.perf_counter() < deadline:
                high = max(range(k), key=candidate_load.__getitem__)
                low = min(range(k), key=candidate_load.__getitem__)
                if candidate_load[high] - candidate_load[low] <= 1:
                    break
                before = candidate_load[high] ** 2 + candidate_load[low] ** 2
                aa, bb = pair(candidate[high], candidate[low], deadline)
                sa, sb = sum(w[j] for j in aa), sum(w[j] for j in bb)
                candidate[high], candidate[low] = aa, bb
                candidate_load[high], candidate_load[low] = sa, sb
                if sa * sa + sb * sb >= before:
                    break
            return candidate

        # Small instances try a full exact search first; otherwise it is called only on selected small neighbourhoods.
        if len(positive) - len(frozen_jobs) <= exact_depth and score > lower:
            now = time.perf_counter()
            candidate, proved = exact(work, now + max(0.0, end - now) / (2.0 + strength))
            accept(list(range(size)), candidate)
            if proved:
                lower = score
        stagnation = 0
        while score > lower and time.perf_counter() < end:
            now = time.perf_counter()
            remaining = end - now
            local_end = min(end, now + remaining / (2.0 + strength))
            max_width = min(size, max(2, int(2 + (1 + strength) * math.log1p(stagnation + 1))))
            width = 2 if size == 2 or rng.random() < 0.55 else rng.randint(3, max(3, max_width))
            width = min(width, size)
            probe_size = min(size, max(width, int(math.sqrt(size) * (1 + strength))))
            probe = rng.sample(range(size), probe_size)
            probe.sort(key=weight.__getitem__)
            indices = [probe[0], probe[-1]]
            if width > 2:
                middle = [i for i in probe if i not in indices]
                indices.extend(rng.sample(middle, width - 2))
            if width == 2:
                a, b = indices
                candidate = list(pair(work[a], work[b], local_end))
            else:
                parts = [work[i] for i in indices]
                if sum(len(part) for part in parts) <= exact_depth and rng.random() < 1.0 / (1.0 + strength):
                    candidate, _ = exact(parts, local_end)
                else:
                    candidate = rebuild(indices, local_end, stagnation)
            if accept(indices, candidate):
                stagnation = 0
            else:
                stagnation += 1
        final_groups = groups[:]
        for i, part in zip(active, work):
            final_groups[i] = part
        # Zero jobs were earlier removed from active machines; existing zero jobs on frozen machines need not be repeated.
        active_set = set(active)
        present_zero = {j for i in range(m) if i not in active_set
                        for j in groups[i] if w[j] == 0}
        final_groups[active[0]] = final_groups[active[0]] + [
            j for j in zero_jobs if j not in present_zero]
        groups = final_groups
    except Exception:
        # Search failure does not affect the separately saved, structurally complete initial feasible solution.
        pass
    return output()
