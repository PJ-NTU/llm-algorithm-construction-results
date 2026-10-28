"""Batched insertion iterated greedy with local-optimum memory
Combines bidirectional construction, adaptive batched insertion descent, a local-optimum cache, short tabu search and destruction-reconstruction, keeping the two-machine order and small-scale branch and bound.
"""

import itertools
import math
import random
import time


class _Stop(Exception):
    pass


def _check(stop):
    if time.perf_counter() >= stop:
        raise _Stop


class _Engine:
    def __init__(self, data, np, incumbent, reverse=False):
        self.np = np
        self.p = data[:, ::-1] if reverse else data
        self.rp = self.p[:, ::-1]
        self.cum = np.cumsum(self.p, axis=1)
        self.off = self.cum - self.p
        self.rcum = np.cumsum(self.rp, axis=1)
        self.roff = self.rcum - self.rp
        self.n, self.m = data.shape
        self.incumbent = incumbent
        self.reverse = reverse
        self.move_time = 0.0
        self.batch_width = 1
        self.optima = {}
        self.cache_capacity = 1

    def key(self, seq):
        return tuple(reversed(seq)) if self.reverse else tuple(seq)

    def certify(self, seq):
        self.optima[self.key(seq)] = True
        while len(self.optima) > self.cache_capacity:
            del self.optima[next(iter(self.optima))]

    def keep(self, seq, value):
        # Only a complete permutation can replace the returnable solution; partial repair states never leak out.
        if len(seq) == self.n and value < self.incumbent[1]:
            self.incumbent[0] = seq[::-1] if self.reverse else list(seq)
            self.incumbent[1] = float(value)

    def extend(self, initial, jobs, stop, backward=False):
        np = self.np
        length = len(jobs)
        out = np.empty((self.m, length))
        if not length:
            return out
        _check(stop)
        p = self.rp if backward else self.p
        cum = self.rcum if backward else self.cum
        off = self.roff if backward else self.off
        # Recurse along the shorter axis and use cumulative maxima on the other axis for the exact completion-time recursion.
        if length < self.m:
            prev = initial
            for k, job in enumerate(jobs):
                if k % 8 == 0:
                    _check(stop)
                prev = np.maximum.accumulate(prev - off[job]) + cum[job]
                out[:, k] = prev
        else:
            block = p[jobs].T
            sums = np.cumsum(block, axis=1)
            out[0] = initial[0] + sums[0]
            for i in range(1, self.m):
                if i % 8 == 0:
                    _check(stop)
                row = np.maximum.accumulate(out[i - 1] - (sums[i] - block[i]))
                np.maximum(row, initial[i], out=row)
                out[i] = row + sums[i]
        return out

    def tables(self, seq, stop):
        np = self.np
        length = len(seq)
        heads = np.zeros((self.m, length + 1))
        tails = np.zeros_like(heads)
        heads[:, 1:] = self.extend(heads[:, 0], seq, stop)
        tails[:, :length] = self.extend(
            tails[:, -1], seq[::-1], stop, True
        )[::-1, ::-1]
        return heads, tails

    def value(self, seq, stop):
        if not seq:
            return 0.0
        return float(self.extend(self.np.zeros(self.m), seq, stop)[-1, -1])

    def without(self, seq, heads, tails, left, right, stop):
        length = len(seq) - (right - left)
        h = self.np.empty((self.m, length + 1))
        t = self.np.empty_like(h)
        h[:, :left + 1] = heads[:, :left + 1]
        t[:, left:] = tails[:, right:]
        h[:, left + 1:] = self.extend(heads[:, left], seq[right:], stop)
        t[:, :left] = self.extend(
            tails[::-1, right], seq[:left][::-1], stop, True
        )[::-1, ::-1]
        return h, t

    def splice(self, base, h, t, job, pos, stop):
        seq = base[:pos] + [job] + base[pos:]
        nh = self.np.empty((self.m, len(seq) + 1))
        nt = self.np.empty_like(nh)
        nh[:, :pos + 1] = h[:, :pos + 1]
        nt[:, pos + 1:] = t[:, pos:]
        nh[:, pos + 1:] = self.extend(h[:, pos], seq[pos:], stop)
        nt[:, :pos + 1] = self.extend(
            t[::-1, pos], seq[:pos + 1][::-1], stop, True
        )[::-1, ::-1]
        return seq, nh, nt

    def costs(self, h, t, block, stop):
        np = self.np
        _check(stop)
        f = h.copy()
        for job in block:
            f -= self.off[job, :, None]
            np.maximum.accumulate(f, axis=0, out=f)
            f += self.cum[job, :, None]
        f += t
        return np.max(f, axis=0), np.sum(f, axis=0)

    def position(self, values, ties, rng=None):
        np = self.np
        best = float(np.min(values))
        eps = 1e-12 * max(abs(best), 1e-100)
        choices = np.flatnonzero(values <= best + eps)
        secondary = ties[choices]
        min_tie = float(np.min(secondary))
        choices = choices[secondary <= min_tie + 1e-12 * max(abs(min_tie), 1e-100)]
        pos = int(choices[rng.randrange(len(choices))] if rng else choices[0])
        return pos, float(values[pos])

    def neh(self, order, stop, rng):
        # If construction is unfinished, save the complete permutation formed by the built prefix plus the original-order suffix.
        _, suffix = self.tables(order, stop)
        self.keep(order, float(suffix[0, 0]))
        seq = []
        h = self.np.zeros((self.m, 1))
        t = self.np.zeros_like(h)
        value = 0.0
        for k, job in enumerate(order):
            values, ties = self.costs(h, t, (job,), stop)
            pos, value = self.position(values, ties, rng)
            seq, h, t = self.splice(seq, h, t, job, pos, stop)
            full_value = float(self.np.max(h[:, -1] + suffix[:, k + 1]))
            self.keep(seq + order[k + 1:], full_value)
        return seq, value

    def batch_costs(self, seq, jobs, stop):
        # A row corresponds to a removed job and a column to its insertion position in the partial permutation.
        # Candidates in a batch share array recursions, but every value is the true makespan of a complete permutation.
        np = self.np
        _check(stop)
        batch, length = len(jobs), len(seq) - 1
        old = np.asarray([seq.index(job) for job in jobs], dtype=np.intp)
        columns = np.arange(length)[None, :]
        indices = columns + (columns >= old[:, None])
        bases = np.asarray(seq, dtype=np.intp)[indices]
        h = np.zeros((self.m, 2 * batch, length + 1))
        if length < self.m:
            for k in range(length):
                _check(stop)
                first, last = bases[:, k], bases[:, length - 1 - k]
                h[:, :batch, k + 1] = np.maximum.accumulate(
                    h[:, :batch, k] - self.off[first].T, axis=0
                ) + self.cum[first].T
                h[:, batch:, k + 1] = np.maximum.accumulate(
                    h[:, batch:, k] - self.roff[last].T, axis=0
                ) + self.rcum[last].T
        else:
            block = self.p[bases].transpose(2, 0, 1)
            both = np.concatenate((block, block[::-1, :, ::-1]), axis=1)
            sums = np.cumsum(both, axis=2)
            h[0, :, 1:] = sums[0]
            for i in range(1, self.m):
                _check(stop)
                h[i, :, 1:] = sums[i] + np.maximum.accumulate(
                    h[i - 1, :, 1:] - sums[i] + both[i], axis=1
                )
        _check(stop)
        joined = h[:, :batch].copy()
        joined -= self.off[jobs].T[:, :, None]
        np.maximum.accumulate(joined, axis=0, out=joined)
        joined += self.cum[jobs].T[:, :, None]
        joined += h[::-1, batch:, ::-1]
        return np.max(joined, axis=0)

    def descent(self, seq, value, stop, rng, focus=()):
        seq = list(seq)
        if self.key(seq) in self.optima:
            return seq, value
        started = time.perf_counter()
        scanned = 0
        np = self.np
        try:
            h, t = self.tables(seq, stop)
            width = 1
            while time.perf_counter() < stop:
                improved = False
                jobs = list(seq)
                rng.shuffle(jobs)
                if focus:
                    preferred = set(focus)
                    jobs = [j for j in jobs if j in preferred] + [
                        j for j in jobs if j not in preferred
                    ]
                    focus = ()
                offset = 0
                while offset < len(jobs):
                    _check(stop)
                    size = min(width, self.batch_width, len(jobs) - offset)
                    selected = jobs[offset:offset + size]
                    if size == 1:
                        job = selected[0]
                        old = seq.index(job)
                        bh, bt = self.without(seq, h, t, old, old + 1, stop)
                        costs, _ = self.costs(bh, bt, (job,), stop)
                        values = costs[None, :]
                    else:
                        values = self.batch_costs(seq, selected, stop)
                    scanned += size
                    positions = np.argmin(values, axis=1)
                    minima = values[np.arange(size), positions]
                    eps = 1e-12 * max(abs(value), 1e-100)
                    rows = np.flatnonzero(minima < value - eps)
                    if len(rows):
                        # Take the first improvement in the original scan order; later candidates are discarded immediately.
                        row = int(rows[0])
                        job, at = selected[row], int(positions[row])
                        old = seq.index(job)
                        base = seq[:old] + seq[old + 1:]
                        seq = base[:at] + [job] + base[at:]
                        value = float(minima[row])
                        self.keep(seq, value)
                        improved = True
                        offset += row + 1
                        width = 1
                        if self.key(seq) in self.optima:
                            return seq, value
                        if size == 1:
                            _, h, t = self.splice(base, bh, bt, job, at, stop)
                        else:
                            h, t = self.tables(seq, stop)
                    else:
                        offset += size
                        # Enlarge the batch only after consecutive failures, avoiding wasted candidates early in the descent.
                        width = min(self.batch_width, max(width + 1, 2 * width))
                if not improved:
                    self.certify(seq)
                    break
        except _Stop:
            pass
        if scanned:
            estimate = (time.perf_counter() - started) / scanned
            self.move_time = estimate if not self.move_time else (
                0.75 * self.move_time + 0.25 * estimate
            )
        return seq, value

    def tabu(self, seq, value, stop, rng, budget):
        # Within a small time slice, worsening moves are allowed and moving the same job repeatedly is briefly forbidden.
        # Tabu only restricts the search path; moves better than the global record are exempt (aspiration).
        seq = list(seq)
        best, best_value = list(seq), value
        n = len(seq)
        if n < 3:
            return best, best_value
        np = self.np
        tenure = min(n - 1, max(1, int(math.sqrt(n) *
                                      (0.6 + 0.18 * math.log1p(budget)))))
        probes = min(n, max(self.batch_width, int(math.sqrt(n) *
                                                 (1.0 + math.log1p(budget)))))
        forbidden = [0] * self.n
        step = 0
        try:
            while time.perf_counter() < stop:
                selected = rng.sample(seq, probes)
                choice = None
                candidate_value = math.inf
                eps = 1e-12 * max(abs(value), 1e-100)
                for offset in range(0, len(selected), self.batch_width):
                    _check(stop)
                    jobs = selected[offset:offset + self.batch_width]
                    values = self.batch_costs(seq, jobs, stop)
                    for r, job in enumerate(jobs):
                        values[r, seq.index(job)] = math.inf
                        if forbidden[job] > step:
                            values[r, values[r] >= self.incumbent[1] - eps] = math.inf
                    v = float(np.min(values))
                    if not math.isfinite(v):
                        continue
                    tied = np.flatnonzero(values.ravel() <= v + eps)
                    flat = int(tied[rng.randrange(len(tied))])
                    row, at = divmod(flat, len(seq))
                    v = float(values[row, at])
                    if v < candidate_value - eps or (
                            abs(v - candidate_value) <= eps and rng.random() < 0.5):
                        choice = (jobs[row], at)
                        candidate_value = v
                if choice is None:
                    step += 1
                    continue
                job, at = choice
                seq.remove(job)
                seq.insert(at, job)
                value = candidate_value
                step += 1
                forbidden[job] = step + min(n - 1, tenure + rng.randrange(1 + tenure // 2))
                self.keep(seq, value)
                if value < best_value:
                    best, best_value = list(seq), value
        except _Stop:
            pass
        return best, best_value

    def blocks(self, seq, value, depth, stop, rng):
        seq = list(seq)
        try:
            while time.perf_counter() < stop:
                h, t = self.tables(seq, stop)
                improved = False
                for size in range(2, min(depth, len(seq) - 1) + 1):
                    starts = list(range(len(seq) - size + 1))
                    rng.shuffle(starts)
                    for left in starts:
                        _check(stop)
                        right = left + size
                        bh, bt = self.without(seq, h, t, left, right, stop)
                        base = seq[:left] + seq[right:]
                        block = seq[left:right]
                        # Check not only forward and reverse orders but every permutation within the block.
                        for oriented in itertools.permutations(block):
                            values, ties = self.costs(bh, bt, oriented, stop)
                            at = int(self.np.argmin(values))
                            candidate = float(values[at])
                            if candidate < value - 1e-12 * max(abs(value), 1e-100):
                                seq = base[:at] + list(oriented) + base[at:]
                                value = candidate
                                self.keep(seq, value)
                                seq, value = self.descent(seq, value, stop, rng)
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if not improved:
                    break
        except _Stop:
            pass
        return seq, value

    def reconstruct(self, seq, count, regret, partial_slice, stop, rng):
        length = len(seq)
        if rng.random() < 0.22:
            left = rng.randrange(length - count + 1)
            spots = set(range(left, left + count))
        else:
            spots = set(rng.sample(range(length), count))
        pending = [job for k, job in enumerate(seq) if k in spots]
        base = [job for k, job in enumerate(seq) if k not in spots]
        rng.shuffle(pending)
        h, t = self.tables(base, stop)
        if partial_slice > 0.0 and len(base) > 2:
            base, _ = self.descent(
                base, float(h[-1, -1]),
                min(stop, time.perf_counter() + partial_slice), rng
            )
            h, t = self.tables(base, stop)
        value = 0.0
        while pending:
            _check(stop)
            if regret and len(pending) > 1:
                choice = None
                for job in pending:
                    values, ties = self.costs(h, t, (job,), stop)
                    pos, candidate = self.position(values, ties, rng)
                    second = float(self.np.partition(values, 1)[1]) if len(values) > 1 else candidate
                    key = (second - candidate, float(self.np.mean(values)) - candidate, rng.random())
                    if choice is None or key > choice[0]:
                        choice = (key, job, pos, candidate)
                _, job, pos, value = choice
            else:
                job = pending[-1]
                values, ties = self.costs(h, t, (job,), stop)
                pos, value = self.position(values, ties, rng)
            pending.remove(job)
            if not pending:
                base = base[:pos] + [job] + base[pos:]
                self.keep(base, value)
            else:
                base, h, t = self.splice(base, h, t, job, pos, stop)
        return base, value


def _johnson(first, second):
    early = [j for j in range(len(first)) if first[j] <= second[j]]
    late = [j for j in range(len(first)) if first[j] > second[j]]
    early.sort(key=lambda j: (first[j], j))
    late.sort(key=lambda j: (-second[j], j))
    return early + late


def _elite_add(pool, seq, value, capacity):
    key = tuple(seq)
    for old_value, old_seq in pool:
        if old_seq == key:
            return
    edges = set(zip(key, key[1:]))
    for k, (old_value, old_seq) in enumerate(pool):
        common = sum((a, b) in edges for a, b in zip(old_seq, old_seq[1:]))
        if len(key) - 1 - common < max(2, len(key) // 12):
            if value < old_value:
                pool[k] = (float(value), key)
                pool.sort()
            return
    pool.append((float(value), key))
    pool.sort()
    del pool[capacity:]


def _exact(engine, incumbent, stop, budget):
    # Prune states with the same remaining set by completion-vector dominance; machine loads and earliest arrival times give lower bounds.
    p = engine.p.tolist()
    n, m = engine.n, engine.m
    tail = [[0.0] * m for _ in range(n)]
    for j in range(n):
        acc = 0.0
        for i in range(m - 1, -1, -1):
            tail[j][i] = acc
            acc += p[j][i]
    load = [sum(p[j][i] for j in range(n)) for i in range(m)]
    fronts = {}
    front_limit = max(8, int(8 + math.sqrt(budget * n)))
    path = []
    order = list(incumbent[0])

    def visit(mask, completion, remaining_load):
        _check(stop)
        if not mask:
            engine.keep(path, completion[-1])
            return
        known = fronts.get(mask, [])
        if any(all(a <= b for a, b in zip(row, completion)) for row in known):
            return
        kept = [row for row in known if not all(a <= b for a, b in zip(completion, row))]
        kept.append(tuple(completion))
        if len(kept) > front_limit:
            kept.sort(key=sum)
            del kept[front_limit:]
        fronts[mask] = kept
        remaining = [j for j in order if mask & (1 << j)]
        first_start = [math.inf] * m
        min_tail = [math.inf] * m
        children = []
        for job in remaining:
            row = p[job]
            child = [0.0] * m
            a = 0.0
            for i in range(m):
                start = max(a, completion[i])
                first_start[i] = min(first_start[i], start)
                min_tail[i] = min(min_tail[i], tail[job][i])
                a = start + row[i]
                child[i] = a
            if child[-1] >= incumbent[1]:
                return
            children.append((child[-1], sum(child), job, child))
        bound = max(first_start[i] + remaining_load[i] + min_tail[i] for i in range(m))
        if bound >= incumbent[1]:
            return
        children.sort()
        for _, _, job, child in children:
            path.append(job)
            visit(mask ^ (1 << job), child,
                  [max(0.0, remaining_load[i] - p[job][i]) for i in range(m)])
            path.pop()

    visit((1 << n) - 1, [0.0] * m, load)


def solve(instance: dict, time_limit_s: float) -> dict:
    started = time.perf_counter()
    incumbent = [[], math.inf]
    try:
        n = max(0, int(instance["n"]))
        # perm implicitly represents x[j][k] = 1 if and only if perm[k] = j, which guarantees both assignment constraints.
        # The earliest-completion recursion satisfies both the previous-position and previous-machine constraints; Cmax is the last entry.
        incumbent[0] = list(range(n))
        m = int(instance["n_machines"])
        budget = max(0.0, float(time_limit_s))
        if n <= 1 or m <= 1 or not math.isfinite(budget) or budget <= 0.0:
            return {"perm": incumbent[0]}
        stop = started + 0.975 * budget
        _check(stop)
        # The single file depends only on allowed general modules; an import failure still keeps the complete initial permutation.
        import numpy as np

        _check(stop)
        data = np.asarray(instance["proc"], dtype=float).reshape(n, m)
        if m == 2:
            incumbent[0] = _johnson(data[:, 0], data[:, 1])
            return {"perm": incumbent[0]}
        forward = _Engine(data, np, incumbent)
        backward = _Engine(data, np, incumbent, True)
        # The number of parallel candidates scales with budget and size; the element cap only bounds temporary array memory.
        width = max(1, min(n, int(2 + math.log1p(budget) *
                                  math.sqrt(n / (m + 1))),
                           max(1, 262144 // (m * n))))
        cache_capacity = max(2, min(max(2, 1048576 // n),
                                    int(16 * (1 + math.log1p(budget)))))
        backward.optima = forward.optima
        for engine in (forward, backward):
            engine.batch_width = width
            engine.cache_capacity = cache_capacity
        forward.keep(incumbent[0], forward.value(incumbent[0], stop))
        rng = random.Random(n * 1000003 + m * 9176)
        total = np.sum(data, axis=1)
        capacity = min(n, max(2, int(2 + math.log1p(budget) * math.log1p(n) / (1 + math.log1p(m)))))
        block_depth = min(n - 1, 2 + int(math.log1p(budget / (1 + n * m / 500.0)) / math.log(3.0)))
        exact_size = min(18, 9 + int(math.log2(1 + budget)))
        pool = []
        initial_stop = min(stop, started + 0.16 * budget)
        mode = 0
        while time.perf_counter() < initial_stop or mode == 0:
            engine = forward if mode % 2 == 0 else backward
            kind = (mode // 2) % 6
            p = engine.p
            if kind == 0:
                scores = total
            elif kind == 1:
                scores = total + 0.35 * np.max(p, axis=1)
            elif kind == 2:
                scores = np.sum(p * np.arange(1, m + 1), axis=1)
            elif kind == 3:
                scores = total + m * np.std(p, axis=1)
            elif kind == 4:
                cut = rng.randrange(1, m)
                order = _johnson(np.sum(p[:, :cut], axis=1), np.sum(p[:, -cut:], axis=1))
                scores = None
            else:
                scores = total * np.asarray([rng.uniform(0.8, 1.2) for _ in range(n)])
            if scores is not None:
                if mode >= 12:
                    scores = scores * np.asarray([rng.uniform(0.94, 1.06) for _ in range(n)])
                order = [int(j) for j in np.argsort(-scores, kind="stable")]
            phase_stop = min(stop, started + (0.23 if mode == 0 else 0.16) * budget)
            try:
                seq, value = engine.neh(order, phase_stop, rng)
                canonical = seq[::-1] if engine.reverse else seq
                _elite_add(pool, canonical, value, capacity)
            except _Stop:
                break
            mode += 1
            if n <= exact_size:
                break

        _elite_add(pool, incumbent[0], incumbent[1], capacity)
        local_stop = min(stop, started + 0.30 * budget)
        for value, canonical in list(pool):
            if time.perf_counter() >= local_stop:
                break
            seq, value = forward.descent(list(canonical), value, local_stop, rng)
            _elite_add(pool, seq, value, capacity)
        if n <= exact_size and time.perf_counter() < stop:
            _exact(forward, incumbent, stop, budget)
            return {"perm": list(incumbent[0])}

        forward.blocks(incumbent[0], incumbent[1], block_depth,
                       min(stop, started + 0.35 * budget), rng)
        _elite_add(pool, incumbent[0], incumbent[1], capacity)
        current = list(incumbent[0])
        current_value = incumbent[1]
        average = float(np.mean(data))
        lower_bound = max(
            float(np.max(total)),
            float(np.max(np.sum(data, axis=0) + np.min(forward.off, axis=0)
                         + np.min(total[:, None] - forward.cum, axis=0)))
        )
        last_best = time.perf_counter()
        last_restart = last_best
        search_started = last_best
        tabu_used = 0.0
        tabu_gain = 0.0
        greedy_gain = 0.0
        observed = max(forward.move_time, backward.move_time, 1e-6)
        # The main search stops only at the wall-clock deadline, with no cap on restarts or total iterations.
        while time.perf_counter() < stop:
            if incumbent[1] <= lower_bound:
                break
            now = time.perf_counter()
            stalled = min(1.0, (now - last_best) / max(0.18 * budget, 1e-9))
            local_slice = max(1.4 * n * observed, budget / (8 + math.sqrt(n)))
            local_slice = min(local_slice, 0.12 * budget)
            base_count = min(n - 1, max(2, int(2 + 0.5 * math.log1p(budget / (n * observed)))))
            extra = max(1, int(math.sqrt(n) * math.log1p(budget) / (4 + math.log1p(m))))
            upper = min(n - 1, base_count + int(stalled * extra))
            count = rng.randint(base_count, max(base_count, upper))
            engine = backward if rng.random() < 0.5 else forward
            oriented = current[::-1] if engine.reverse else list(current)
            record = incumbent[1]
            # Allocate tabu search by its actual share of wall clock; its share is not increased when it yields no gain.
            now = time.perf_counter()
            share = min(0.16, 0.025 * math.log1p(budget) + 0.04 * stalled)
            if tabu_gain > greedy_gain:
                share = min(0.20, 1.4 * share)
            if (stalled > 0.15 and
                    tabu_used < share * (now - search_started)):
                tick = time.perf_counter()
                seed = incumbent[0][::-1] if engine.reverse else incumbent[0]
                engine.tabu(
                    seed, incumbent[1],
                    min(stop, tick + min(0.035 * budget,
                                        max(0.3 * local_slice, n * observed))),
                    rng, budget
                )
                tabu_used += time.perf_counter() - tick
                if incumbent[1] < record:
                    tabu_gain += record - incumbent[1]
                    last_best = time.perf_counter()
                    current = list(incumbent[0])
                    current_value = incumbent[1]
                    _elite_add(pool, current, current_value, capacity)
                    oriented = current[::-1] if engine.reverse else list(current)
                record = incumbent[1]
            if len(pool) > 1 and rng.random() < 0.12:
                parent = list(rng.choice(pool)[1])
                if engine.reverse:
                    parent.reverse()
                left, right = sorted(rng.sample(range(n), 2))
                right += 1
                block = oriented[left:right]
                present = set(block)
                fill = [j for j in parent if j not in present]
                candidate = fill[:left] + block + fill[left:]
                value = engine.value(candidate, stop)
                engine.keep(candidate, value)
            else:
                candidate, value = engine.reconstruct(
                    oriented, count, rng.random() < 0.30,
                    0.2 * local_slice if rng.random() < 0.22 else 0.0,
                    stop, rng
                )
            focus = [j for k, j in enumerate(candidate) if j != oriented[k]]
            candidate, value = engine.descent(
                candidate, value, min(stop, time.perf_counter() + local_slice), rng, focus
            )
            observed = max(1e-6, 0.8 * observed + 0.2 * max(engine.move_time, 1e-6))
            if incumbent[1] < record or rng.random() < 0.10 * (1 + stalled):
                candidate, value = engine.blocks(
                    candidate, value, block_depth,
                    min(stop, time.perf_counter() + max(0.3 * local_slice, budget / (100 + 5 * math.sqrt(n)))), rng
                )
            canonical = candidate[::-1] if engine.reverse else candidate
            _elite_add(pool, canonical, value, capacity)
            if incumbent[1] < record:
                greedy_gain += record - incumbent[1]
                last_best = time.perf_counter()
                _elite_add(pool, incumbent[0], incumbent[1], capacity)
            delta = value - current_value
            temperature = max(average * (0.04 + 0.12 * stalled), 1e-100)
            if delta <= 0.0 or rng.random() < math.exp(-min(700.0, delta / temperature)):
                current, current_value = list(canonical), value
            now = time.perf_counter()
            if (now - last_best > budget / (3 + math.log1p(budget))
                    and now - last_restart > budget / (6 + math.sqrt(n))):
                current_value, parent = rng.choice(pool)
                current = list(parent)
                last_restart = now
    except Exception:
        # On interruption or error, return the most recently saved complete permutation, never a partial construction state.
        pass
    return {"perm": list(incumbent[0])}
