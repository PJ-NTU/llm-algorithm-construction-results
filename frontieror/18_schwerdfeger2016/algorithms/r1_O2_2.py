"""Adaptive subset-partition large neighbourhood search
Reassigns jobs with bitset or meet-in-the-middle enumeration, and improves load balance with multi-machine recursive partitioning, time-limited branch and bound and equal-value reordering.
"""
import heapq
import math
import random
import time


def solve(instance: dict, time_limit_s: float) -> dict:
    start = time.perf_counter()
    budget = max(0.0, float(time_limit_s))
    m = int(instance["num_machines"])
    n = int(instance["num_jobs"])
    original = instance["processing_times"]
    weights = [int(original[j]) for j in range(n)]
    divisor = 0
    for value in weights:
        divisor = math.gcd(divisor, value)
    divisor = max(1, divisor)
    weights = [value // divisor for value in weights]
    bins = [[] for _ in range(m)]
    loads = [0] * m

    # First build a complete feasible solution; later neighbourhoods commit only after all computation finishes.
    queue = [(0, i) for i in range(m)]
    for job in sorted(range(n), key=weights.__getitem__, reverse=True):
        load, machine = heapq.heappop(queue)
        bins[machine].append(job)
        loads[machine] = load + weights[job]
        heapq.heappush(queue, (loads[machine], machine))

    def result():
        return {
            "assignment": {str(i): bins[i] for i in range(m)},
            "machine_completion_times": {
                str(i): float(loads[i] * divisor) for i in range(m)
            },
        }

    total = sum(weights)
    q, r = divmod(total, m)
    lower_bound = (m - r) * q * q + r * (q + 1) * (q + 1)
    score = sum(value * value for value in loads)
    reserve = min(budget * 0.15, 0.003 + 0.000003 * (n + m))
    deadline = start + max(0.0, budget - reserve)
    if (score == lower_bound or m == 1 or
            m >= sum(value > 0 for value in weights) or
            time.perf_counter() >= deadline):
        return result()

    # The total load is fixed, so the sum of squares orders solutions exactly as the normalised root variance in the problem statement.
    effort = math.log1p(budget)
    rng = random.Random((total << 9) ^ (n << 4) ^ m)
    stale = 0

    def meet_in_middle(jobs, numerator, denominator, until):
        """When integer magnitudes are large, expand states by job count rather than by numeric size."""
        available = until - time.perf_counter()
        state_limit = min(1 << 19, max(1, int(available * 160000)))
        count = len(jobs)
        half = count // 2
        if max(half, count - half) > state_limit.bit_length() - 1:
            return None
        shuffled = list(jobs)
        rng.shuffle(shuffled)
        sides = (shuffled[:half], shuffled[half:])
        tables = []
        for side in sides:
            table = [(0, 0)]
            for pos, job in enumerate(side):
                added = []
                bit = 1 << pos
                value = weights[job]
                for idx, (amount, mask) in enumerate(table):
                    if (idx & 255) == 0 and time.perf_counter() >= until:
                        return None
                    added.append((amount + value, mask | bit))
                table.extend(added)
            table.sort()
            tables.append(table)
            if time.perf_counter() >= until:
                return None
        left, right = tables
        a, b = 0, len(right) - 1
        best_error = None
        best_masks = (0, 0)
        while a < len(left) and b >= 0:
            if time.perf_counter() >= until:
                return None
            delta = (left[a][0] + right[b][0]) * denominator - numerator
            error = abs(delta)
            if best_error is None or error < best_error:
                best_error = error
                best_masks = (left[a][1], right[b][1])
                if error == 0:
                    break
            if delta < 0:
                a += 1
            else:
                b -= 1
        chosen = set()
        for side, mask in zip(sides, best_masks):
            chosen.update(job for pos, job in enumerate(side) if mask & (1 << pos))
        return chosen

    def split(jobs, numerator, denominator, until):
        """Find the subset closest to a given rational target; on failure the current assignment is unchanged."""
        if time.perf_counter() >= until:
            return None
        amount = sum(weights[job] for job in jobs)
        if numerator <= 0:
            return [], list(jobs)
        if numerator >= amount * denominator:
            return list(jobs), []
        mirrored = 2 * numerator > amount * denominator
        if mirrored:
            numerator = amount * denominator - numerator
        by_value = {}
        zeros = []
        local_gcd = 0
        for job in jobs:
            value = weights[job]
            if value:
                by_value.setdefault(value, []).append(job)
                local_gcd = math.gcd(local_gcd, value)
            else:
                zeros.append(job)
        groups = []
        for value, members in by_value.items():
            rng.shuffle(members)
            pos, size = 0, 1
            while pos < len(members):
                take = min(size, len(members) - pos)
                groups.append((value // local_gcd * take, members[pos:pos + take]))
                pos += take
                size *= 2
        rng.shuffle(groups)
        scaled_total = amount // local_gcd
        scaled_den = denominator * local_gcd
        if 2 * numerator == amount * denominator:
            cap = scaled_total // 2
        else:
            # For any minimal subset exceeding the target, the excess is at most its largest job.
            cap = min(scaled_total, (numerator + scaled_den - 1) // scaled_den
                      + max(by_value) // local_gcd)
        stride = math.isqrt(len(groups)) + 1
        copies = (len(groups) + stride - 1) // stride + stride + 4
        memory_cap = (64 * 1024 * 1024 * 8) // copies
        available = max(0.0, until - time.perf_counter())
        work_cap = int(available * 600000000 / max(1, 2 * len(groups)))
        if cap > min(memory_cap, work_cap):
            chosen = meet_in_middle([j for j in jobs if weights[j]],
                                    numerator, denominator, until)
            if chosen is None:
                return None
        else:
            mask = (1 << (cap + 1)) - 1
            reachable = 1
            checkpoints = []
            for idx, (value, _) in enumerate(groups):
                if time.perf_counter() >= until:
                    return None
                if idx % stride == 0:
                    checkpoints.append(reachable)
                if value <= cap:
                    reachable = (reachable | (reachable << value)) & mask
            floor_target = min(cap, numerator // scaled_den)
            below = (reachable & ((1 << (floor_target + 1)) - 1)).bit_length() - 1
            target = below
            above_start = (numerator + scaled_den - 1) // scaled_den
            if above_start <= cap:
                tail = reachable >> above_start
                if tail:
                    above = above_start + (tail & -tail).bit_length() - 1
                    if abs(above * scaled_den - numerator) < abs(below * scaled_den - numerator):
                        target = above
            chosen = set()
            # Checkpoints plus in-block recomputation avoid storing a full bitset for every job.
            for block in range(len(checkpoints) - 1, -1, -1):
                if target == 0:
                    break
                begin = block * stride
                end = min(begin + stride, len(groups))
                history = [checkpoints[block]]
                for idx in range(begin, end):
                    if time.perf_counter() >= until:
                        return None
                    value = groups[idx][0]
                    previous = history[-1]
                    history.append(((previous | (previous << value)) & mask)
                                   if value <= cap else previous)
                for idx in range(end - 1, begin - 1, -1):
                    value, members = groups[idx]
                    previous = history[idx - begin]
                    skip = (previous >> target) & 1
                    take = target >= value and ((previous >> (target - value)) & 1)
                    if take and (not skip or rng.getrandbits(1)):
                        chosen.update(members)
                        target -= value
        chosen.update(job for job in zeros if rng.getrandbits(1))
        left = [job for job in jobs if job in chosen]
        right = [job for job in jobs if job not in chosen]
        return (right, left) if mirrored else (left, right)

    def recursive_partition(jobs, count, until):
        if time.perf_counter() >= until:
            return None
        if count == 1:
            return [list(jobs)]
        left_count = count // 2
        partition = split(jobs, sum(weights[j] for j in jobs) * left_count, count, until)
        if partition is None:
            return None
        first = recursive_partition(partition[0], left_count, until)
        if first is None:
            return None
        second = recursive_partition(partition[1], count - left_count, until)
        return None if second is None else first + second

    def exact_block(machines, active, until):
        """Fix unselected jobs and run multi-machine branch and bound with convex-relaxation bounds on the selected jobs."""
        chosen = set(active)
        retained = [[j for j in bins[i] if j not in chosen] for i in machines]
        base = [sum(weights[j] for j in group) for group in retained]
        running = list(base)
        active = sorted(active, key=weights.__getitem__, reverse=True)
        width = len(machines)
        final_total = sum(base) + sum(weights[j] for j in active)
        incumbent = sum(loads[i] * loads[i] for i in machines)
        best_path = None
        path = [0] * len(active)
        halted = False

        def bound():
            remaining, slots, fixed = final_total, width, 0
            for value in sorted(running, reverse=True):
                if value * slots <= remaining:
                    break
                fixed += value * value
                remaining -= value
                slots -= 1
            if not slots:
                return fixed
            quotient, remainder = divmod(remaining, slots)
            return (fixed + (slots - remainder) * quotient * quotient
                    + remainder * (quotient + 1) * (quotient + 1))

        root_bound = bound()

        def visit(pos, squares):
            nonlocal incumbent, best_path, halted
            if time.perf_counter() >= until:
                halted = True
                return
            if incumbent == root_bound or bound() >= incumbent:
                return
            if pos == len(active):
                incumbent = squares
                best_path = list(path)
                return
            value = weights[active[pos]]
            seen = set()
            for machine in sorted(range(width), key=running.__getitem__):
                old = running[machine]
                if old in seen:
                    continue
                seen.add(old)
                updated = squares + value * (2 * old + value)
                if updated >= incumbent:
                    continue
                running[machine] = old + value
                path[pos] = machine
                visit(pos + 1, updated)
                running[machine] = old
                if halted or incumbent == root_bound:
                    return

        visit(0, sum(value * value for value in running))
        if best_path is None:
            return None, not halted
        for job, machine in zip(active, best_path):
            retained[machine].append(job)
        return retained, not halted

    def install(machines, candidate):
        nonlocal score
        candidate_loads = [sum(weights[j] for j in group) for group in candidate]
        old = sum(loads[i] * loads[i] for i in machines)
        new = sum(value * value for value in candidate_loads)
        if new > old:
            return False
        # Equal-value reordering changes subsequent neighbourhoods without losing the current best objective value.
        for machine, group, load in zip(machines, candidate, candidate_loads):
            bins[machine] = group
            loads[machine] = load
        score += new - old
        return new < old

    try:
        while time.perf_counter() < deadline and score > lower_bound:
            now = time.perf_counter()
            remaining = deadline - now
            depth = min(n, max(2, int(8 + 4 * effort +
                        effort * min(effort, math.log1p(stale)))))
            max_width = min(m, depth, max(2, int(2 + 2 * effort)))
            pool_size = min(m, max_width + max(2, int(math.sqrt(m) * effort)))
            pool = rng.sample(range(m), pool_size)
            high = max(pool, key=loads.__getitem__)
            low = min((i for i in pool if i != high), key=loads.__getitem__)
            pair_probability = 1.0 / (1.0 + math.log1p(stale) / (1.0 + effort))
            until = min(deadline, now + min(remaining * 0.4,
                        budget / (2.0 + math.sqrt(m))))
            improved = False
            if rng.random() < pair_probability:
                if rng.random() < 1.0 / (2.0 + effort):
                    high, low = rng.sample(range(m), 2)
                machines = [high, low]
                jobs = bins[high] + bins[low]
                candidate = split(jobs, loads[high] + loads[low], 2, until)
                if candidate is not None:
                    improved = install(machines, candidate)
                    if m == 2:
                        break  # A complete two-machine subset partition has already given the global optimum.
                elif time.perf_counter() < until:
                    active = rng.sample(jobs, min(depth, len(jobs)))
                    candidate, _ = exact_block(machines, active, until)
                    if candidate is not None:
                        improved = install(machines, candidate)
            else:
                width = rng.randint(2, max_width)
                others = [i for i in pool if i != high and i != low]
                machines = [high, low] + rng.sample(others, width - 2)
                jobs = [j for i in machines for j in bins[i]]
                complete = False
                if len(jobs) <= depth:
                    candidate, complete = exact_block(machines, jobs, until)
                    if candidate is not None:
                        improved = install(machines, candidate)
                    if width == m and complete:
                        break
                else:
                    # As budget and stagnation grow, enlarge the scope of joint re-optimisation.
                    if rng.random() < 0.5:
                        candidate = recursive_partition(jobs, width, until)
                        if candidate is not None:
                            improved = install(machines, candidate)
                    if not improved and time.perf_counter() < until:
                        active = rng.sample(jobs, min(depth, len(jobs)))
                        candidate, _ = exact_block(machines, active, until)
                        if candidate is not None:
                            improved = install(machines, candidate)
            stale = 0 if improved else stale + 1
    except Exception:
        # Resource exceptions during search do not affect the saved complete feasible assignment.
        pass
    return result()
