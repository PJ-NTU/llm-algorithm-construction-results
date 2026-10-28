"""Bidirectional beam reconstruction with window branch-and-bound iterated greedy
Combines accelerated insertion descent, multi-branch destruction-reconstruction, window reordering bounded by prefixes and suffixes, and elite restarts, keeping the two-machine order and small-scale branch and bound.
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

    def descent(self, seq, value, stop, rng, focus=()):
        seq = list(seq)
        started = time.perf_counter()
        scanned = 0
        try:
            h, t = self.tables(seq, stop)
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
                for job in jobs:
                    _check(stop)
                    old = seq.index(job)
                    bh, bt = self.without(seq, h, t, old, old + 1, stop)
                    scanned += 1
                    # If removing a job does not reduce the makespan, no reinsertion can strictly improve it.
                    if float(bh[-1, -1]) >= value:
                        continue
                    values, ties = self.costs(bh, bt, (job,), stop)
                    at = int(self.np.argmin(values))
                    new_value = float(values[at])
                    if new_value < value - 1e-12 * max(abs(value), 1e-100):
                        base = seq[:old] + seq[old + 1:]
                        seq = base[:at] + [job] + base[at:]
                        value = new_value
                        self.keep(seq, value)
                        _, h, t = self.splice(base, bh, bt, job, at, stop)
                        improved = True
                if not improved:
                    break
        except _Stop:
            pass
        if scanned:
            estimate = (time.perf_counter() - started) / scanned
            self.move_time = estimate if not self.move_time else (
                0.75 * self.move_time + 0.25 * estimate
            )
        return seq, value

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

    def beam_reconstruct(self, seq, count, width, stop, rng):
        # Each level inserts the same job, keeping several repair branches at different positions.
        # Partial permutations are used only for search; only the last level can update the complete feasible solution.
        np = self.np
        if rng.random() < 0.25:
            left = rng.randrange(len(seq) - count + 1)
            spots = set(range(left, left + count))
        else:
            spots = set(rng.sample(range(len(seq)), count))
        pending = [j for k, j in enumerate(seq) if k in spots]
        base = [j for k, j in enumerate(seq) if k not in spots]
        rng.shuffle(pending)
        h, t = self.tables(base, stop)
        beam = [(base, h, t)]
        result, result_value = list(seq), math.inf
        while pending:
            _check(stop)
            # The first branch picks the job with the largest regret; the other branches use the same job.
            if len(pending) > 1:
                choice = None
                for job in pending:
                    values, ties = self.costs(beam[0][1], beam[0][2], (job,), stop)
                    best = float(np.min(values))
                    second = float(np.partition(values, 1)[1]) if len(values) > 1 else best
                    key = (second - best, float(np.mean(values)) - best, rng.random())
                    if choice is None or key > choice[0]:
                        choice = (key, job)
                job = choice[1]
            else:
                job = pending[0]
            pending.remove(job)
            proposals = []
            for parent, (base, h, t) in enumerate(beam):
                values, ties = self.costs(h, t, (job,), stop)
                # The secondary measure only breaks ties of equal completion time and does not replace the true objective.
                positions = np.lexsort((ties, values))
                if not pending:
                    pos = int(positions[0])
                    value = float(values[pos])
                    candidate = base[:pos] + [job] + base[pos:]
                    self.keep(candidate, value)
                    if value < result_value:
                        result, result_value = candidate, value
                    continue
                # Besides the best position, keep one distant candidate to prevent the beam from degenerating.
                chosen = [int(k) for k in positions[:width]]
                if len(positions) > width:
                    first = chosen[0]
                    distance = max(1, len(base) // (2 * width))
                    for at in positions:
                        at = int(at)
                        if abs(at - first) >= distance and at not in chosen:
                            chosen.append(at)
                            break
                for at in chosen:
                    proposals.append((float(values[at]), float(ties[at]),
                                      rng.random(), parent, at))
            if not pending:
                return result, result_value
            proposals.sort()
            selected = []
            seen = set()
            for value, tie, _, parent, at in proposals:
                _check(stop)
                base, h, t = beam[parent]
                key = tuple(base[:at] + [job] + base[at:])
                if key in seen:
                    continue
                seen.add(key)
                selected.append((parent, at))
                if len(selected) >= width:
                    break
            beam = [self.splice(beam[parent][0], beam[parent][1],
                                beam[parent][2], job, at, stop)
                    for parent, at in selected]
        return result, result_value

    def windows(self, seq, value, depth, stop, rng):
        # The order outside the window is fixed; the prefix completion vector and suffix tail lengths enter the objective exactly.
        seq = list(seq)
        np = self.np
        if len(seq) < 3 or depth < 3:
            return seq, value
        try:
            while time.perf_counter() < stop:
                h, t = self.tables(seq, stop)
                size = min(len(seq), rng.randint(max(3, depth - 2), max(3, depth)))
                # Critical jobs come first, while every window keeps a chance of being explored.
                if rng.random() < 0.65:
                    critical = h[:, 1:] + t[:, :-1] - self.p[seq].T
                    eps = 1e-10 * max(abs(value), 1e-100)
                    weight = np.sum(self.p[seq].T * (critical >= value - eps), axis=0)
                    weights = (weight + np.mean(weight) + 1e-100).tolist()
                    center = rng.choices(range(len(seq)), weights=weights, k=1)[0]
                    left = max(0, min(len(seq) - size, center - rng.randrange(size)))
                else:
                    left = rng.randrange(len(seq) - size + 1)
                right = left + size
                jobs = seq[left:right]
                rows = self.p[jobs].tolist()
                suffix = t[:, right].tolist()
                loads = [sum(row[i] for row in rows) for i in range(self.m)]
                downstream = []
                for row in rows:
                    tail = [0.0] * self.m
                    total = 0.0
                    for i in range(self.m - 1, -1, -1):
                        tail[i] = total
                        total += row[i]
                    downstream.append(tail)
                fronts = {}
                path = []
                local = [value, jobs]
                # Each window takes only part of the remaining time slice; improvements already found are kept on timeout.
                node_stop = min(stop, time.perf_counter() +
                                max(self.move_time * len(seq), (stop - time.perf_counter()) / 3.0))

                def visit(mask, completion, load):
                    _check(node_stop)
                    if not mask:
                        candidate_value = max(completion[i] + suffix[i] for i in range(self.m))
                        if candidate_value < local[0]:
                            order = [jobs[k] for k in path]
                            local[0], local[1] = candidate_value, order
                            self.keep(seq[:left] + order + seq[right:], candidate_value)
                        return
                    known = fronts.get(mask, ())
                    if any(all(a <= b for a, b in zip(row, completion)) for row in known):
                        return
                    # Keep only non-dominated completion vectors; dominance still holds for a fixed suffix.
                    fronts[mask] = [row for row in known
                                    if not all(a <= b for a, b in zip(completion, row))] + [tuple(completion)]
                    first = [math.inf] * self.m
                    tail_min = [math.inf] * self.m
                    children = []
                    bits = mask
                    while bits:
                        bit = bits & -bits
                        bits -= bit
                        k = bit.bit_length() - 1
                        row = rows[k]
                        child = []
                        finish = 0.0
                        score = 0.0
                        for i in range(self.m):
                            start = max(finish, completion[i])
                            first[i] = min(first[i], start)
                            tail_min[i] = min(tail_min[i], downstream[k][i])
                            finish = start + row[i]
                            child.append(finish)
                            score = max(score, finish + load[i] - row[i] + suffix[i])
                        # Delaying any job can never make it finish earlier than processing it immediately.
                        if max(child[i] + suffix[i] for i in range(self.m)) >= local[0]:
                            return
                        children.append((score, sum(child), k, bit, child))
                    bound = max(first[i] + load[i] +
                                max(suffix[i], tail_min[i] + suffix[-1])
                                for i in range(self.m))
                    if bound >= local[0]:
                        return
                    children.sort()
                    for _, _, k, bit, child in children:
                        path.append(k)
                        visit(mask ^ bit, child,
                              [max(0.0, load[i] - rows[k][i]) for i in range(self.m)])
                        path.pop()

                try:
                    visit((1 << size) - 1, h[:, left].tolist(), loads)
                except _Stop:
                    pass
                if local[0] < value:
                    seq = seq[:left] + local[1] + seq[right:]
                    value = local[0]
                    seq, value = self.descent(seq, value, stop, rng)
        except _Stop:
            pass
        return seq, value


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
                       min(stop, started + 0.33 * budget), rng)
        observed = max(forward.move_time, backward.move_time, 1e-6)
        window_depth = min(n, 4 + int(math.log1p(
            budget / max(m * observed, 1e-9)) / math.log(4.0)))
        forward.windows(incumbent[0], incumbent[1], window_depth,
                        min(stop, started + 0.35 * budget), rng)
        _elite_add(pool, incumbent[0], incumbent[1], capacity)
        current = list(incumbent[0])
        current_value = incumbent[1]
        average = float(np.mean(data))
        # Single-machine load plus the shortest arrival time of any first job and the shortest departure time of the last job.
        prefix = np.cumsum(data, axis=1) - data
        suffix = total[:, None] - prefix - data
        lower_bound = max(float(np.max(np.sum(data, axis=0) +
                                       np.min(prefix, axis=0) + np.min(suffix, axis=0))),
                          float(np.max(total)))
        last_best = time.perf_counter()
        last_restart = last_best
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
            elif rng.random() < 0.20 + 0.35 * stalled:
                width = min(count + 1, max(2, int(1 + math.log1p(
                    budget / max(n * observed, 1e-9)) / math.log(4.0))))
                candidate, value = engine.beam_reconstruct(
                    oriented, count, width, stop, rng
                )
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
                extra_stop = min(stop, time.perf_counter() + max(
                    0.3 * local_slice, budget / (100 + 5 * math.sqrt(n))))
                if rng.random() < 0.45 + 0.25 * stalled:
                    window_depth = min(n, 4 + int(math.log1p(
                        budget / max(m * observed, 1e-9)) / math.log(4.0)))
                    candidate, value = engine.windows(
                        candidate, value, window_depth, extra_stop, rng
                    )
                else:
                    candidate, value = engine.blocks(
                        candidate, value, block_depth, extra_stop, rng
                    )
            canonical = candidate[::-1] if engine.reverse else candidate
            _elite_add(pool, canonical, value, capacity)
            if incumbent[1] < record:
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
    except (_Stop, Exception):
        # On interruption or error, return the most recently saved complete permutation, never a partial construction state.
        pass
    return {"perm": list(incumbent[0])}
