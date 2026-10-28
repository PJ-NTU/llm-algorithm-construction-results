"""Bidirectional accelerated beam-reconstruction iterated greedy
Uses batched prefix/suffix insertion evaluation, local search on the partial sequence and adaptive beam reconstruction, and keeps block neighbourhoods and small-scale branch and bound.
"""

import math
import random
import time

try:
    import numpy as np
except Exception:
    np = None


class _Expired(Exception):
    pass


class _Kernel:
    def __init__(self, processing, deadline, matrix=None):
        self.p = processing
        self.m = len(processing)
        self.deadline = deadline
        self.matrix = matrix

    def check(self, until):
        if time.perf_counter() >= until:
            raise _Expired

    def value(self, sequence):
        c = [0.0] * self.m
        p = self.p
        for k, job in enumerate(sequence):
            if k % 64 == 0:
                self.check(self.deadline)
            c[0] += p[0][job]
            for i in range(1, self.m):
                c[i] = max(c[i], c[i - 1]) + p[i][job]
        return c[-1]

    def profiles(self, sequence, until):
        self.check(until)
        length, m, p = len(sequence), self.m, self.p
        if self.matrix is not None:
            a = self.matrix[:, sequence]
            prefix = np.cumsum(a, axis=1)
            h = np.zeros((m, length + 1))
            h[0, 1:] = prefix[0]
            for i in range(1, m):
                self.check(until)
                h[i, 1:] = prefix[i] + np.maximum.accumulate(
                    h[i - 1, 1:] - prefix[i] + a[i]
                )
            a = a[::-1, ::-1]
            prefix = np.cumsum(a, axis=1)
            q = np.zeros((m, length + 1))
            q[0, 1:] = prefix[0]
            for i in range(1, m):
                self.check(until)
                q[i, 1:] = prefix[i] + np.maximum.accumulate(
                    q[i - 1, 1:] - prefix[i] + a[i]
                )
            return h, q[::-1, ::-1]
        h = [[0.0] * (length + 1) for _ in range(m)]
        q = [[0.0] * (length + 1) for _ in range(m)]
        for i in range(m):
            row, values = h[i], p[i]
            prior = h[i - 1] if i else None
            c = 0.0
            for k, job in enumerate(sequence):
                if k % 128 == 0:
                    self.check(until)
                if i and prior[k + 1] > c:
                    c = prior[k + 1]
                c += values[job]
                row[k + 1] = c
        for i in range(m - 1, -1, -1):
            row, values = q[i], p[i]
            following = q[i + 1] if i + 1 < m else None
            c = 0.0
            for k in range(length - 1, -1, -1):
                if k % 128 == 0:
                    self.check(until)
                if following is not None and following[k] > c:
                    c = following[k]
                c += values[sequence[k]]
                row[k] = c
        return h, q

    def options(self, profiles, block, until):
        self.check(until)
        h, q = profiles
        m, p, length = self.m, self.p, len(h[0])
        if np is not None and isinstance(h, np.ndarray):
            c = h.copy()
            for job in block:
                c[0] += p[0][job]
                for i in range(1, m):
                    self.check(until)
                    c[i] = np.maximum(c[i], c[i - 1]) + p[i][job]
            joined = c + q
            return np.max(joined, axis=0), np.sum(joined, axis=0)
        values, secondary = [0.0] * length, [0.0] * length
        if len(block) == 1:
            job = block[0]
            for k in range(length):
                if k % 64 == 0:
                    self.check(until)
                c = h[0][k] + p[0][job]
                value = total = c + q[0][k]
                for i in range(1, m):
                    c = max(c, h[i][k]) + p[i][job]
                    joined = c + q[i][k]
                    if joined > value:
                        value = joined
                    total += joined
                values[k], secondary[k] = value, total
        else:
            for k in range(length):
                if k % 32 == 0:
                    self.check(until)
                c = [h[i][k] for i in range(m)]
                for job in block:
                    c[0] += p[0][job]
                    for i in range(1, m):
                        c[i] = max(c[i], c[i - 1]) + p[i][job]
                joined = [c[i] + q[i][k] for i in range(m)]
                values[k], secondary[k] = max(joined), sum(joined)
        return values, secondary

    def insert(self, sequence, job, until, rng=None):
        values, secondary = self.options(
            self.profiles(sequence, until), (job,), until
        )
        if np is not None and isinstance(values, np.ndarray):
            best = float(values.min())
            tied = np.flatnonzero(values <= best + 1e-12 * max(1.0, abs(best)))
        else:
            best = min(values)
            tol = 1e-12 * max(1.0, abs(best))
            tied = [k for k, v in enumerate(values) if v <= best + tol]
        if rng is None:
            pos = min(tied, key=lambda k: secondary[k])
        else:
            pos = tied[rng.randrange(len(tied))]
        pos = int(pos)
        return pos, float(values[pos])


def _johnson(jobs, first, second):
    early = sorted((j for j in jobs if first[j] <= second[j]),
                   key=lambda j: (first[j], j))
    late = sorted((j for j in jobs if first[j] > second[j]),
                  key=lambda j: (-second[j], j))
    return early + late


def _orders(p, totals):
    jobs, m = list(range(len(totals))), len(p)
    yield sorted(jobs, key=lambda j: (-totals[j], j))
    yield sorted(jobs, key=lambda j: -(totals[j] + 0.35 * max(r[j] for r in p)))
    for direction in (1, -1):
        weights = [sum((i + 1 if direction > 0 else m - i) * p[i][j]
                       for i in range(m)) for j in jobs]
        yield sorted(jobs, key=lambda j: -weights[j])
    for cut in sorted({1, max(1, m // 4), max(1, m // 2), m - 1}):
        first = [sum(p[i][j] for i in range(cut)) for j in jobs]
        second = [sum(p[i][j] for i in range(m - cut, m)) for j in jobs]
        yield _johnson(jobs, first, second)


def _neh(order, kernel, until):
    sequence = []
    for k, job in enumerate(order):
        try:
            pos, value = kernel.insert(sequence, job, until)
        except _Expired:
            return sequence + order[k:], None
        sequence.insert(pos, job)
    return sequence, value


def _descent(sequence, value, kernel, until, rng, offer=None):
    sequence = list(sequence)
    work = 0
    try:
        while time.perf_counter() < until:
            improved = False
            jobs = sequence[:]
            rng.shuffle(jobs)
            for job in jobs:
                old = sequence.index(job)
                base = sequence[:old] + sequence[old + 1:]
                pos, new_value = kernel.insert(base, job, until, rng)
                work += 1
                if new_value < value - 1e-11 * max(1.0, abs(value)):
                    base.insert(pos, job)
                    sequence, value = base, new_value
                    improved = True
                    if offer is not None:
                        offer(sequence, value)
            if not improved:
                break
    except _Expired:
        pass
    return sequence, value, work


def _blocks(sequence, value, kernel, until, rng, depth, offer):
    sequence = list(sequence)
    n = len(sequence)
    try:
        while time.perf_counter() < until:
            improved = False
            sizes = list(range(2, min(depth, n - 1) + 1))
            rng.shuffle(sizes)
            for size in sizes:
                starts = list(range(n - size + 1))
                rng.shuffle(starts)
                for old in starts:
                    block = sequence[old:old + size]
                    base = sequence[:old] + sequence[old + size:]
                    profiles = kernel.profiles(base, until)
                    for oriented in (block, block[::-1]):
                        values, secondary = kernel.options(profiles, oriented, until)
                        if np is not None and isinstance(values, np.ndarray):
                            pos = int(np.argmin(values))
                        else:
                            pos = min(range(len(values)), key=values.__getitem__)
                        new_value = float(values[pos])
                        if new_value < value - 1e-11 * max(1.0, abs(value)):
                            sequence = base[:pos] + oriented + base[pos:]
                            value = new_value
                            offer(sequence, value)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                break
    except _Expired:
        pass
    return sequence, value


def _reconstruct(base, removed, kernel, until, rng, width):
    if width <= 1:
        sequence = list(base)
        value = 0.0
        for job in removed:
            pos, value = kernel.insert(sequence, job, until, rng)
            sequence.insert(pos, job)
        return sequence, value
    # Keep several partial permutations for the same insertion order, not only the current best insertion position.
    beam = [(0.0, 0.0, list(base))]
    for job in removed:
        candidates = []
        for _, _, sequence in beam:
            values, secondary = kernel.options(
                kernel.profiles(sequence, until), (job,), until
            )
            if np is not None and isinstance(values, np.ndarray):
                positions = [int(k) for k in np.lexsort((secondary, values))[:width]]
            else:
                positions = sorted(range(len(values)),
                                   key=lambda k: (values[k], secondary[k]))[:width]
            for pos in positions:
                candidate = sequence[:pos] + [job] + sequence[pos:]
                candidates.append((float(values[pos]), float(secondary[pos]), candidate))
        candidates.sort(key=lambda x: (x[0], x[1]))
        beam = candidates[:width]
    return beam[0][2], beam[0][0]


def _critical_positions(sequence, kernel, count, rng, until, average):
    h, _ = kernel.profiles(sequence, until)
    weights = [1.0] * len(sequence)
    k, i = len(sequence) - 1, kernel.m - 1
    while k >= 0 and i >= 0:
        weights[k] += kernel.p[i][sequence[k]] / max(average, 1e-12)
        if i == 0 or (k > 0 and h[i][k] >= h[i - 1][k + 1]):
            k -= 1
        else:
            i -= 1
    return sorted(range(len(sequence)),
                  key=lambda k: rng.random() ** (1.0 / weights[k]), reverse=True)[:count]


def _exact(sequence, value, kernel, until, offer, memory_width):
    n, m, p = len(sequence), kernel.m, kernel.p
    tails = [[0.0] * m for _ in range(n)]
    for j in range(n):
        for i in range(m - 2, -1, -1):
            tails[j][i] = tails[j][i + 1] + p[i + 1][j]
    frontiers = {}
    path = []
    best_value = value

    def visit(mask, completion):
        nonlocal best_value
        kernel.check(until)
        if not mask:
            if completion[-1] < best_value:
                best_value = completion[-1]
                offer(path, best_value)
            return
        known = frontiers.get(mask, [])
        for vector in known:
            if all(a <= b for a, b in zip(vector, completion)):
                return
        kept = [v for v in known if not all(a <= b for a, b in zip(completion, v))]
        if len(kept) < memory_width:
            kept.append(tuple(completion))
        frontiers[mask] = kept
        load, earliest, tail = [0.0] * m, [math.inf] * m, [math.inf] * m
        children = []
        for job in sequence:
            if not mask & (1 << job):
                continue
            c = [0.0] * m
            for i in range(m):
                start = max(completion[i], c[i - 1] if i else 0.0)
                c[i] = start + p[i][job]
                load[i] += p[i][job]
                earliest[i] = min(earliest[i], start)
                tail[i] = min(tail[i], tails[job][i])
            children.append((sum(c), job, c))
        if any(earliest[i] + load[i] + tail[i] >= best_value for i in range(m)):
            return
        children.sort(key=lambda item: item[0])
        for _, job, c in children:
            if c[-1] < best_value:
                path.append(job)
                visit(mask ^ (1 << job), c)
                path.pop()

    try:
        visit((1 << n) - 1, [0.0] * m)
        return True
    except _Expired:
        return False


def solve(instance: dict, time_limit_s: float) -> dict:
    started = time.perf_counter()
    best = []
    try:
        n = max(0, int(instance.get("n", 0)))
        best = list(range(n))
        m = int(instance.get("n_machines", 0))
        budget = max(0.0, float(time_limit_s))
        if not math.isfinite(budget):
            budget = 0.0
        deadline = started + budget - min(0.03, 0.025 * budget)
        if n <= 1 or m <= 1 or time.perf_counter() >= deadline:
            return {"perm": best}
        proc = instance["proc"]
        p = list(zip(*proc))
        if m == 2:
            return {"perm": _johnson(best, p[0], p[1])}
        matrix = None
        if np is not None and n >= 64:
            matrix = np.asarray(p, dtype=float)
        forward = _Kernel(p, deadline, matrix)
        backward = _Kernel(p[::-1], deadline, None if matrix is None else matrix[::-1])
        kernels = (forward, backward)
        best_value = forward.value(best)
        totals = [sum(row) for row in proc]
        average = sum(totals) / (n * m)
        seed = (n * 1000003 + m * 9176) & 0xFFFFFFFF
        for total in totals[:16]:
            seed = (seed * 1000003 + int(abs(total) * 1009)) & 0xFFFFFFFF
        rng = random.Random(seed)
        intensity = math.log1p(budget)
        pool_size = max(2, min(n, 2 + int(intensity * math.sqrt(n / (n + m)))))
        pool = {}

        def offer(sequence, value, reverse=False):
            nonlocal best, best_value
            if value < best_value - 1e-11 * max(1.0, abs(best_value)):
                candidate = list(reversed(sequence)) if reverse else list(sequence)
                actual = forward.value(candidate)
                if actual < best_value:
                    best, best_value = candidate, actual

        def archive(sequence, value):
            key = tuple(sequence)
            pool[key] = float(value)
            if len(pool) > pool_size:
                worst = max(pool, key=pool.__getitem__)
                del pool[worst]

        # Install a complete order first; even if construction is cut short, a valid permutation of all jobs exists.
        order = sorted(best, key=lambda j: (-totals[j], j))
        offer(order, forward.value(order))
        archive(best, best_value)
        construction_end = min(deadline, started + 0.17 * budget)
        for order in _orders(p, totals):
            if time.perf_counter() >= construction_end:
                break
            for reverse, kernel in enumerate(kernels):
                if time.perf_counter() >= construction_end:
                    break
                sequence, value = _neh(order, kernel, construction_end)
                if value is None:
                    value = kernel.value(sequence)
                canonical = sequence[::-1] if reverse else sequence
                offer(canonical, value)
                archive(canonical, value)
        local_end = min(deadline, started + 0.30 * budget)
        sweep_cost = max(1e-6, 0.005 * budget)
        seeds = sorted(pool.items(), key=lambda item: item[1])
        for index, (sequence, value) in enumerate(seeds):
            now = time.perf_counter()
            if now >= local_end:
                break
            limit = min(local_end, now + (local_end - now) / max(1, len(seeds) - index))
            candidate, candidate_value, work = _descent(
                sequence, value, forward, limit, rng, offer
            )
            if work:
                sweep_cost = (time.perf_counter() - now) * n / work
            offer(candidate, candidate_value)
            archive(candidate, candidate_value)

        exact_size = min(18, 9 + int(math.log2(1.0 + budget)) + int(m <= 4))
        if n <= exact_size:
            if _exact(best[:], best_value, forward, deadline, offer,
                      max(8, int(8 + 8 * intensity))):
                return {"perm": best}

        current, current_value = best[:], best_value
        archive(best, best_value)
        minimum_destroy = min(n - 1, max(2, int(2 + intensity / 3)))
        maximum_destroy = min(n - 1, minimum_destroy + max(
            1, int(intensity * math.sqrt(n / (m + 1)))
        ))
        beam_width = max(1, min(n, 1 + int(intensity / 1.5)))
        block_depth = min(n - 1, 2 + int(intensity / 2))
        weights = [1.0, 1.0, 1.0, 1.0]
        last_best = last_restart = last_block = time.perf_counter()
        restart_period = max(0.10 * budget, 3.0 * sweep_cost)
        temperature = max(1e-12, 0.06 * average)

        # The main loop is driven only by the wall-clock deadline; there is no cap on iterations or restarts.
        while time.perf_counter() < deadline:
            now = time.perf_counter()
            old_best = best_value
            stale = min(1.0, (now - last_best) / max(restart_period, 1e-9))
            if now - last_restart > restart_period and stale >= 0.8:
                parent = list(rng.choice(list(pool))) if pool else best[:]
                left, right = sorted(rng.sample(range(n), 2))
                middle = best[left:right]
                present = set(middle)
                fill = [j for j in parent if j not in present]
                current = fill[:left] + middle + fill[left:]
                current_value = forward.value(current)
                last_restart = now
            reverse = rng.random() < 0.5
            kernel = kernels[int(reverse)]
            sequence = current[::-1] if reverse else current[:]
            local_offer = lambda s, v: offer(s, v, reverse)
            count = minimum_destroy + int(
                (maximum_destroy - minimum_destroy) * stale * rng.random()
            )
            operator = rng.choices(range(4), weights=weights, k=1)[0]
            if operator == 1:
                start = rng.randrange(n - count + 1)
                positions = list(range(start, start + count))
            elif operator == 2:
                positions = _critical_positions(sequence, kernel, count, rng,
                                                deadline, average)
            else:
                positions = rng.sample(range(n), count)
            base = sequence[:]
            removed = [base.pop(pos) for pos in sorted(positions, reverse=True)]
            rng.shuffle(removed)
            window = max(0.025 * budget, 1.5 * sweep_cost)
            # First optimise the partial sequence after jobs are removed, to break insertion local optima of the complete permutation.
            if base and (operator == 1 or rng.random() < 0.30):
                reduced_value = kernel.value(base)
                base, _, _ = _descent(base, reduced_value, kernel,
                                     min(deadline, time.perf_counter() + 0.22 * window), rng)
            width = 1
            if operator == 3:
                if pool:
                    parent = list(rng.choice(list(pool)))
                    if reverse:
                        parent.reverse()
                    rank = {j: k for k, j in enumerate(parent)}
                    removed.sort(key=rank.__getitem__)
                width = beam_width
            candidate, candidate_value = _reconstruct(
                base, removed, kernel, deadline, rng, width
            )
            local_offer(candidate, candidate_value)
            before_descent = time.perf_counter()
            candidate, candidate_value, work = _descent(
                candidate, candidate_value, kernel,
                min(deadline, before_descent + window), rng, local_offer
            )
            if work:
                estimate = (time.perf_counter() - before_descent) * n / work
                sweep_cost = 0.8 * sweep_cost + 0.2 * estimate
            now = time.perf_counter()
            if (candidate_value <= best_value * (1.0 + 1e-12)
                    and now - last_block > max(0.035 * budget, sweep_cost)):
                candidate, candidate_value = _blocks(
                    candidate, candidate_value, kernel,
                    min(deadline, now + max(0.015 * budget, 0.35 * sweep_cost)),
                    rng, block_depth, local_offer
                )
                last_block = time.perf_counter()
            canonical = candidate[::-1] if reverse else candidate
            archive(canonical, candidate_value)
            delta = candidate_value - current_value
            accepted = delta <= 0 or rng.random() < math.exp(
                -delta / (temperature * (1.0 + 3.0 * stale))
            )
            reward = 1.0
            if accepted:
                current, current_value = canonical, candidate_value
                reward = 2.0
            if best_value < old_best:
                last_best = time.perf_counter()
                reward = 8.0
                archive(best, best_value)
            weights[operator] = 0.85 * weights[operator] + 0.15 * reward
            restart_period = max(0.10 * budget, 3.0 * sweep_cost)
    except Exception:
        # A timeout or computation failure only stops improvement and does not damage the saved complete permutation.
        pass
    return {"perm": best}
