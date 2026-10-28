"""Joint multi-job side-assignment search for the common due date problem
Combines quadratic subset enumeration, straddling-job switching, structured swap and insertion, and positional dynamic programming, searching within the wall-clock budget and keeping the best complete permutation.
"""

import math
import random
import time
from functools import cmp_to_key

try:
    import numpy as np
except Exception:
    np = None


def solve(instance: dict, time_limit_s: float) -> dict:
    started = time.perf_counter()
    best_order = []
    try:
        n = int(instance["n"])
        best_order = list(range(n))
        budget = max(0.0, float(time_limit_s))
        deadline = started + budget - min(0.10, max(0.002, 0.015 * budget))
        if n < 2 or time.perf_counter() >= deadline:
            return {"order": best_order}
        p = list(instance["processing_times"])
        a = list(instance["earliness_penalties"])
        b = list(instance["tardiness_penalties"])
        d = instance["due_date"]
        rng = random.Random(1729)
        clock = time.perf_counter

        def value(order):
            t = 0
            cost = 0
            for j in order:
                t += p[j]
                cost += a[j] * (d - t) if t < d else b[j] * (t - d)
            return cost

        best_cost = value(best_order)

        def save(order, cost=None):
            nonlocal best_order, best_cost
            if cost is None:
                cost = value(order)
            if cost < best_cost:
                best_order = list(order)
                best_cost = cost
            return cost

        def early_cmp(i, j):
            if (p[i] == 0) != (p[j] == 0):
                return 1 if p[i] == 0 else -1
            z = a[i] * p[j] - a[j] * p[i]
            return (z > 0) - (z < 0) if z else (i > j) - (i < j)

        def late_cmp(i, j):
            if (p[i] == 0) != (p[j] == 0):
                return -1 if p[i] == 0 else 1
            z = p[i] * b[j] - p[j] * b[i]
            return (z > 0) - (z < 0) if z else (i > j) - (i < j)

        early_order = sorted(range(n), key=cmp_to_key(early_cmp))
        if clock() >= deadline:
            save(early_order)
            return {"order": best_order}
        late_order = sorted(range(n), key=cmp_to_key(late_cmp))
        total_p = sum(p)
        if d >= total_p:
            return {"order": early_order}
        if d <= 0:
            return {"order": late_order}
        early_rank = [0] * n
        late_rank = [0] * n
        for k, j in enumerate(early_order):
            early_rank[j] = k
        for k, j in enumerate(late_order):
            late_rank[j] = k

        def split(order):
            s = 0
            k = 0
            while k < n and s + p[order[k]] <= d:
                s += p[order[k]]
                k += 1
            return k, s

        def canonical(order):
            k, s = split(order)
            e = sorted(order[:k], key=early_rank.__getitem__)
            tail = sorted(order[k:], key=late_rank.__getitem__)
            if not tail:
                return e
            pp = 0
            bb = 0
            h = tail[0]
            extra = None
            for j in tail:
                if s + p[j] >= d:
                    z = p[j] * bb - b[j] * pp
                    if extra is None or z < extra:
                        extra, h = z, j
                pp += p[j]
                bb += b[j]
            tail.remove(h)
            if s + p[h] == d:
                return sorted(e + [h], key=early_rank.__getitem__) + tail
            return e + [h] + tail

        seeds = []
        initial_orders = [best_order, early_order, late_order]
        initial_orders.append(sorted(range(n), key=lambda j: (a[j] / max(1, b[j]), -p[j])))
        initial_orders.append(sorted(range(n), key=lambda j: (-p[j], a[j] - b[j])))
        for order in initial_orders:
            if clock() >= deadline:
                break
            q = canonical(order)
            c = save(q)
            seeds.append((c, q))
        # Skip jobs that temporarily do not fit, adding constructed solutions with different early sets.
        if clock() < deadline:
            packed, rest, duration = [], [], 0
            for j in initial_orders[3]:
                if duration + p[j] <= d:
                    packed.append(j)
                    duration += p[j]
                else:
                    rest.append(j)
            q = canonical(packed + rest)
            seeds.append((save(q), q))
        if clock() >= deadline:
            return {"order": best_order}

        if np is None:
            current = best_order[:]
            while clock() < deadline:
                trial = current[:]
                i, j = rng.sample(range(n), 2)
                if rng.random() < 0.5:
                    trial[i], trial[j] = trial[j], trial[i]
                else:
                    trial.insert(j, trial.pop(i))
                trial = canonical(trial)
                c = save(trial)
                if c <= value(current) or rng.random() < 1.0 / (n + 1):
                    current = trial
            return {"order": best_order}

        P = np.asarray(p, dtype=np.float64)
        A = np.asarray(a, dtype=np.float64)
        B = np.asarray(b, dtype=np.float64)
        eo = np.asarray(early_order, dtype=np.int64)
        lo = np.asarray(late_order, dtype=np.int64)
        er = np.asarray(early_rank, dtype=np.int64)
        lr = np.asarray(late_rank, dtype=np.int64)

        def prefix(v):
            return np.concatenate(([0.0], np.cumsum(v)))

        def structural_move(order, until):
            k, s = split(order)
            if k >= n:
                return None
            h = order[k]
            ei = np.asarray(order[:k], dtype=np.int64)
            li = np.asarray(order[k + 1:], dtype=np.int64)
            em = np.zeros(n, dtype=bool)
            lm = np.zeros(n, dtype=bool)
            em[ei] = True
            lm[li] = True
            ep = prefix(P[eo] * em[eo])
            ea = prefix(A[eo] * em[eo])
            lp = prefix(P[lo] * lm[lo])
            lb = prefix(B[lo] * lm[lo])
            before_e = ep[er]
            after_a = ea[-1] - ea[er + 1]
            before_l = lp[lr]
            before_b = lb[lr]
            re = -A * (d - before_e - P) + P * after_a
            ae = A * (d - before_e - P) - P * after_a
            remove = re - P * (B[h] + before_b) + B * (s + P[h] + before_l - d)
            add = ae + P * (B[h] + before_b) - B * (s + P[h] + before_l + P - d)
            gain = -1e-7
            move = None

            def consider(vals, ids, kind):
                nonlocal gain, move
                if len(ids):
                    z = int(np.argmin(vals))
                    if float(vals[z]) < gain:
                        gain = float(vals[z])
                        move = (kind, int(ids[z]))

            if len(ei):
                vals = np.where(s - P[ei] + P[h] >= d, remove[ei], np.inf)
                consider(vals, ei, "remove")
                corr = np.maximum(P[ei] * A[h], P[h] * A[ei])
                vals = re[ei] + ae[h] + corr + (B[ei] - B[h]) * (s + P[h] - d)
                vals = np.where(s - P[ei] + P[h] <= d, vals, np.inf)
                consider(vals, ei, "pivot")
            if len(li):
                vals = np.where(s + P[li] <= d, add[li], np.inf)
                consider(vals, li, "add")
            if len(ei) and len(li):
                # Chunking limits the size of temporary matrices; the wall clock is checked between chunks.
                block = max(1, min(len(ei), 160000 // len(li)))
                pj = P[li][None, :]
                aj = A[li][None, :]
                bj = B[li][None, :]
                for start in range(0, len(ei), block):
                    if clock() >= until:
                        break
                    ids = ei[start:start + block]
                    pi = P[ids][:, None]
                    vals = remove[ids][:, None] + add[li][None, :]
                    vals += np.maximum(pi * aj, pj * A[ids][:, None])
                    vals += np.maximum(pi * bj, pj * B[ids][:, None])
                    new_s = s - pi + pj
                    vals[(new_s > d) | (new_s + P[h] < d)] = np.inf
                    z = int(np.argmin(vals))
                    v = float(vals.flat[z])
                    if v < gain:
                        x, y = divmod(z, len(li))
                        gain = v
                        move = ("swap", int(ids[x]), int(li[y]))
            if move is None:
                return None
            kind, i = move[:2]
            if kind == "remove":
                em[i] = False
            elif kind == "add":
                em[i] = True
            elif kind == "pivot":
                em[i] = False
                em[h] = True
                h = i
            else:
                em[i] = False
                em[move[2]] = True
            e = [j for j in early_order if em[j]]
            tail = [j for j in late_order if not em[j] and j != h]
            return canonical(e + [h] + tail)

        def insertion_move(order, until):
            ids = np.asarray(order, dtype=np.int64)
            pp, aa, bb = P[ids], A[ids], B[ids]
            c = np.cumsum(pp)
            starts = c - pp
            old = aa * np.maximum(d - c, 0) + bb * np.maximum(c - d, 0)
            best_gain = -1e-7
            best_pair = None
            # Rotate the scan start, so that on large instances the search does not cover only the front of the permutation near the time limit.
            offset = rng.randrange(n)
            for step in range(n):
                if clock() >= until:
                    break
                i = (offset + step) % n
                shift = pp[i]
                plus = aa * np.maximum(d - c - shift, 0) + bb * np.maximum(c + shift - d, 0) - old
                minus = aa * np.maximum(d - c + shift, 0) + bb * np.maximum(c - shift - d, 0) - old
                cp = prefix(plus)
                cm = prefix(minus)
                dest = np.where(np.arange(n) < i, starts + shift, c)
                vals = aa[i] * np.maximum(d - dest, 0) + bb[i] * np.maximum(dest - d, 0) - old[i]
                vals[:i] += cp[i] - cp[:i]
                vals[i + 1:] += cm[i + 2:] - cm[i + 1]
                vals[i] = 0
                j = int(np.argmin(vals))
                if float(vals[j]) < best_gain:
                    best_gain = float(vals[j])
                    best_pair = (i, j)
            if best_pair is None:
                return None
            trial = list(order)
            i, j = best_pair
            trial.insert(j, trial.pop(i))
            return canonical(trial)

        def descend(order, until):
            q = order
            cost = value(q)
            while clock() < until:
                trial = structural_move(q, until)
                if trial is not None:
                    z = save(trial)
                    if z < cost:
                        q, cost = trial, z
                        continue
                if clock() >= until:
                    break
                trial = insertion_move(q, until)
                if trial is not None:
                    z = save(trial)
                    if z < cost:
                        q, cost = trial, z
                        continue
                break
            return q, cost

        partition_rate = 3000000.0
        certified_pivots = set()

        def select_block(order, r, stall, until):
            """Selects jobs whose side may change, by swap cost and random exploration."""
            k, s = split(order)
            old_h = order[k]
            early = order[:k]
            late = order[k + 1:]
            if r == n - 1:
                # Fixing each possible straddling job and enumerating all other side assignments can cover the global optimum.
                h = old_h
                if h in certified_pivots:
                    h = next(j for j in late_order if j not in certified_pivots)
                return h, [j for j in range(n) if j != h]

            h = old_h
            switch_chance = min(0.65, (1 + math.log1p(stall)) / (3 + math.log1p(budget)))
            if rng.random() < switch_chance:
                # Longer straddling jobs allow a wider interval for the total early processing time.
                options = [j for j in late if p[j] >= d - s]
                options += [j for j in early if s - p[j] + p[old_h] <= d]
                if options:
                    if rng.random() < 0.5:
                        width = max(1, min(len(options), int(math.sqrt(n) * (1 + math.log1p(budget)))))
                        options = sorted(options, key=lambda j: -p[j])[:width]
                    h = rng.choice(options)
            selected = [old_h] if h != old_h else []
            eids = np.asarray([j for j in early if j != h], dtype=np.int64)
            lids = np.asarray([j for j in late if j != h], dtype=np.int64)
            needed = r - len(selected)
            count_e = min(needed, len(eids), max(0, needed // 2 + rng.randint(-int(math.sqrt(r)), int(math.sqrt(r)))))
            count_l = min(len(lids), needed - count_e)
            count_e = min(len(eids), needed - count_l)
            count_l = min(len(lids), needed - count_e)

            score_e = np.full(len(eids), np.inf)
            score_l = np.full(len(lids), np.inf)
            ranked_mode = rng.random() < 0.75 and len(eids) and len(lids)
            if ranked_mode and clock() < until:
                em = np.zeros(n, dtype=bool)
                lm = np.zeros(n, dtype=bool)
                em[early] = True
                lm[late] = True
                ep = prefix(P[eo] * em[eo])
                ea = prefix(A[eo] * em[eo])
                lp = prefix(P[lo] * lm[lo])
                lb = prefix(B[lo] * lm[lo])
                ae = A * (d - ep[er] - P) - P * (ea[-1] - ea[er + 1])
                remove = -ae - P * (B[old_h] + lb[lr]) + B * (s + P[old_h] + lp[lr] - d)
                add = ae + P * (B[old_h] + lb[lr]) - B * (s + P[old_h] + lp[lr] + P - d)
                anchors = max(1, r // 2)
                # Swap screening only affects the search order; the final subset enumeration allows any number of additions and removals.
                for j in rng.sample(list(lids), min(len(lids), anchors)):
                    if clock() >= until:
                        break
                    vals = remove[eids] + add[j]
                    vals += np.maximum(P[eids] * A[j], P[j] * A[eids])
                    vals += np.maximum(P[eids] * B[j], P[j] * B[eids])
                    ss = s - P[eids] + P[j]
                    vals[(ss > d) | (ss + P[old_h] < d)] = np.inf
                    np.minimum(score_e, vals, out=score_e)
                for i in rng.sample(list(eids), min(len(eids), anchors)):
                    if clock() >= until:
                        break
                    vals = remove[i] + add[lids]
                    vals += np.maximum(P[i] * A[lids], P[lids] * A[i])
                    vals += np.maximum(P[i] * B[lids], P[lids] * B[i])
                    ss = s - P[i] + P[lids]
                    vals[(ss > d) | (ss + P[old_h] < d)] = np.inf
                    np.minimum(score_l, vals, out=score_l)

            def choose(ids, scores, count):
                if count == 0:
                    return []
                jobs = [int(j) for j in ids]
                if count >= len(jobs):
                    return jobs
                if not ranked_mode:
                    return rng.sample(jobs, count)
                # Randomly shuffle ties and let the candidate bandwidth widen with the degree of stagnation.
                indices = list(range(len(jobs)))
                rng.shuffle(indices)
                indices.sort(key=lambda i: float(scores[i]))
                width = min(len(jobs), max(count, int(count * (1 + math.log1p(stall + 1)))))
                core = min(count, max(1, count // (2 + int(math.log1p(stall)))))
                picked = indices[:core] + rng.sample(indices[core:width], count - core)
                return [jobs[i] for i in picked]

            selected += choose(eids, score_e, count_e)
            selected += choose(lids, score_l, count_l)
            return h, selected

        def partition_neighborhood(order, h, jobs, until):
            """Fixes assignments outside the block; exactly enumerates the quadratic objective and total early processing time within the block."""
            nonlocal partition_rate
            begin = clock()
            r = len(jobs)
            if r == 0 or begin >= until:
                return order, False
            k, _ = split(order)
            em = np.zeros(n, dtype=bool)
            em[order[:k]] = True
            em[jobs] = False
            em[h] = False
            lm = ~em
            lm[h] = False
            fixed_s = sum(p[j] for j in order[:k] if em[j])
            ep = prefix(P[eo] * em[eo])
            ea = prefix(A[eo] * em[eo])
            lp = prefix(P[lo] * lm[lo])
            lb = prefix(B[lo] * lm[lo])
            ids = np.asarray(jobs, dtype=np.int64)
            pp, aa, bb = P[ids], A[ids], B[ids]
            # The base point puts all jobs in the block on the tardy side. Linear term for moving one job to the early side.
            unary = aa * (d - ep[er[ids]] - pp) - pp * (ea[-1] - ea[er[ids] + 1])
            unary += pp * (B[h] + lb[lr[ids]])
            unary -= bb * (fixed_s + P[h] + lp[lr[ids]] + pp - d)
            # Interaction term when two jobs become early together, independent of the fixed straddling job.
            pair = -np.maximum(pp[:, None] * aa[None, :], aa[:, None] * pp[None, :])
            pair -= np.maximum(pp[:, None] * bb[None, :], bb[:, None] * pp[None, :])
            size = 1 << r
            costs = np.empty(size, dtype=np.float64)
            durations = np.empty(size, dtype=np.float64)
            interaction = np.empty(max(1, size // 2), dtype=np.float64)
            costs[0] = 0.0
            durations[0] = fixed_s
            for j in range(r):
                if clock() >= until:
                    return order, False
                bit = 1 << j
                interaction[0] = unary[j]
                for i in range(j):
                    if clock() >= until:
                        return order, False
                    z = 1 << i
                    interaction[z:2 * z] = interaction[:z] + pair[j, i]
                costs[bit:2 * bit] = costs[:bit] + interaction[:bit]
                durations[bit:2 * bit] = durations[:bit] + pp[j]

            # Chunked screening avoids long operations over whole large arrays near the time limit.
            chunk = min(size, max(1024, int(partition_rate * min(0.004, max(0.0001, budget / 1000)))))
            mask = None
            lowest = np.inf
            scanned = 0
            for start in range(0, size, chunk):
                if clock() >= until:
                    break
                stop = min(size, start + chunk)
                ss = durations[start:stop]
                vals = costs[start:stop].copy()
                vals[(ss > d) | (ss + P[h] < d)] = np.inf
                z = int(np.argmin(vals))
                if float(vals[z]) < lowest:
                    lowest = float(vals[z])
                    mask = start + z
                scanned = stop
            complete = scanned == size
            elapsed = clock() - begin
            if complete and elapsed > 0:
                partition_rate = 0.5 * partition_rate + 0.5 * size / elapsed
            if mask is None:
                return order, complete
            for i, j in enumerate(jobs):
                em[j] = bool(mask & (1 << i))
            e = [j for j in early_order if em[j]]
            tail = [j for j in late_order if not em[j] and j != h]
            trial = canonical(e + [h] + tail)
            save(trial)
            return trial, complete

        dp_rate = 2500000.0
        skeleton = None

        def exact_neighborhood(order, slots, until):
            nonlocal dp_rate, skeleton
            begin = clock()
            r = len(slots)
            size = 1 << r
            work = 0
            if skeleton is None or skeleton[0] != r:
                counts = np.zeros(size, dtype=np.uint8)
                for j in range(r):
                    if clock() >= until:
                        return order, False
                    z = 1 << j
                    counts[z:2 * z] = counts[:z] + 1
                layers = []
                for k in range(1, r + 1):
                    if clock() >= until:
                        return order, False
                    layers.append(np.flatnonzero(counts == k).astype(np.int32))
                skeleton = (r, layers)
            layers = skeleton[1]
            chosen_jobs = [order[x] for x in slots]
            sums = np.empty(size, dtype=np.float64)
            sums[0] = 0
            for j, job in enumerate(chosen_jobs):
                z = 1 << j
                sums[z:2 * z] = sums[:z] + P[job]
            dp = np.full(size, np.inf)
            parent = np.zeros(size, dtype=np.uint8)
            dp[0] = 0
            fixed_time = sum(p[j] for j in order[:slots[0]])
            for layer, states in enumerate(layers):
                if clock() >= until:
                    return order, False
                ss = sums[states]
                completion = fixed_time + ss
                early = np.maximum(d - completion, 0)
                tardy = np.maximum(completion - d, 0)
                vals = np.full(len(states), np.inf)
                arg = np.zeros(len(states), dtype=np.uint8)
                for j, job in enumerate(chosen_jobs):
                    if clock() >= until:
                        return order, False
                    bit = 1 << j
                    positions = np.flatnonzero((states & bit) != 0)
                    prev = states[positions] ^ bit
                    cand = dp[prev] + A[job] * early[positions] + B[job] * tardy[positions]
                    improve = cand < vals[positions]
                    ii = positions[improve]
                    vals[ii] = cand[improve]
                    arg[ii] = j
                    work += len(positions)
                # Unselected jobs keep their relative positions; the completion times of the whole block are determined by the total processing time of the current subset.
                if layer + 1 < r:
                    gap = order[slots[layer] + 1:slots[layer + 1]]
                    if gap:
                        gi = np.asarray(gap, dtype=np.int64)
                        fc = fixed_time + np.cumsum(P[gi])
                        cut = np.searchsorted(fc, d - ss, side="right")
                        pa, pb = prefix(A[gi]), prefix(B[gi])
                        pac, pbc = prefix(A[gi] * fc), prefix(B[gi] * fc)
                        vals += (d - ss) * pa[cut] - pac[cut]
                        vals += pbc[-1] - pbc[cut] + (ss - d) * (pb[-1] - pb[cut])
                        fixed_time = float(fc[-1])
                dp[states] = vals
                parent[states] = arg
            mask = size - 1
            selected = []
            while mask:
                j = int(parent[mask])
                selected.append(chosen_jobs[j])
                mask ^= 1 << j
            selected.reverse()
            trial = list(order)
            for pos, job in zip(slots, selected):
                trial[pos] = job
            elapsed = clock() - begin
            if elapsed > 0:
                dp_rate = 0.5 * dp_rate + 0.5 * work / elapsed
            trial = canonical(trial)
            save(trial)
            return trial, True

        pool_capacity = max(2, int(2 + math.log1p(budget) * math.sqrt(math.log2(n + 1))))
        pool = []

        def add_pool(order, cost):
            signature = tuple(order)
            if any(entry[2] == signature for entry in pool):
                return
            pool.append((cost, list(order), signature))
            pool.sort(key=lambda entry: entry[0])
            if len(pool) > pool_capacity:
                pool.pop()

        for cost, order in seeds:
            add_pool(order, cost)
        q = best_order[:]
        stall = 0
        while clock() < deadline:
            old_best = best_cost
            remaining = deadline - clock()
            slice_s = min(remaining, max(0.005, budget / (2 + math.log2(n + 1))))
            q, cost = descend(q, min(deadline, clock() + slice_s))
            add_pool(q, cost)
            if clock() >= deadline:
                break
            remaining = deadline - clock()
            target = min(remaining * 0.30, budget / (4 + math.sqrt(n)))
            # Quadratic side-assignment enumeration does not fix the number of early jobs; its complexity is proportional to the number of subsets of the chosen block.
            r = 1
            memory_states = (128 * 1024 * 1024) // 32
            while r < n - 1 and (1 << (r + 1)) <= memory_states:
                if (1 << (r + 1)) > partition_rate * target:
                    break
                r += 1
            block_until = min(deadline, clock() + max(target * 1.8, 0.001))
            h, jobs = select_block(q, r, stall, block_until)
            trial, complete = partition_neighborhood(q, h, jobs, block_until)
            z = save(trial)
            if complete and len(jobs) == n - 1:
                certified_pivots.add(h)
                if len(certified_pivots) == n:
                    return {"order": best_order}
            if z < cost:
                q, cost = trial, z
                add_pool(q, cost)
                stall = 0 if best_cost < old_best else stall + 1
                continue
            if clock() >= deadline:
                break
            # When full side-assignment enumeration can cover all straddling jobs, no more time is spent on random restarts.
            if r == n - 1:
                stall += 1
                continue

            # Keep positional subset dynamic programming as a complementary neighbourhood, allowing jobs outside the block to cross the due date.
            if rng.random() < 1.0 / (2 + math.log1p(budget)):
                remaining = deadline - clock()
                target = min(remaining * 0.30, budget / (6 + math.sqrt(n)))
                depth = 1
                while depth < n and (1 << (depth + 1)) <= memory_states // 2:
                    nd = depth + 1
                    if nd * (1 << (nd - 1)) > dp_rate * target:
                        break
                    depth = nd
                if depth >= 2 and clock() < deadline:
                    if depth == n:
                        slots = list(range(n))
                    else:
                        k, _ = split(q)
                        boundary = min(n - 1, k)
                        if rng.random() < 0.5:
                            left = max(0, min(n - depth, boundary - rng.randrange(depth)))
                            slots = list(range(left, left + depth))
                        else:
                            chosen = set(rng.sample(range(n), depth - 1))
                            chosen.add(boundary)
                            if len(chosen) < depth:
                                candidates = [i for i in range(n) if i not in chosen]
                                chosen.add(rng.choice(candidates))
                            slots = sorted(chosen)
                    trial, complete = exact_neighborhood(q, slots, min(deadline, clock() + max(target * 1.8, 0.001)))
                    if complete:
                        z = save(trial)
                        if depth == n:
                            return {"order": best_order}
                        if z < cost:
                            q, cost = trial, z
                            add_pool(q, cost)
                            stall = 0 if best_cost < old_best else stall + 1
                            continue
            stall = 0 if best_cost < old_best else stall + 1
            if clock() >= deadline:
                break
            # First try different multi-job combinations at the same local optimum, then perturb adaptively.
            if rng.random() < 1 - 1 / math.sqrt(1 + math.log1p(budget)):
                continue
            if rng.random() < min(0.45, stall / (stall + 2 + 3 * math.log1p(budget))):
                noise = 0.25 + math.log1p(stall) / (1 + math.log1p(budget))
                q = sorted(range(n), key=lambda j: math.log((a[j] + 1) / (b[j] + 1)) + noise * rng.random())
                q = canonical(q)
            else:
                if pool and rng.random() < 0.35:
                    q = list(rng.choice(pool)[1])
                else:
                    q = best_order[:]
                k, _ = split(q)
                strength = min(n, max(2, int((1 + math.log1p(budget)) * math.sqrt(stall + 1))))
                moves = rng.randint(1, strength)
                for _ in range(moves):
                    if clock() >= deadline:
                        break
                    if 0 < k < n:
                        i, j = rng.randrange(k), rng.randrange(k, n)
                    else:
                        i, j = rng.sample(range(n), 2)
                    q[i], q[j] = q[j], q[i]
                q = canonical(q)
            save(q)
        return {"order": best_order}
    except Exception:
        # A complete permutation is always kept; numerical or optional-dependency failures do not affect interface feasibility.
        return {"order": best_order}
