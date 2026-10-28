"""Structured neighbourhood search for the common due date problem
Combines early/tardy ordering, incremental swap and insertion search, random perturbation and subset dynamic programming, keeping the best complete permutation within the wall-clock limit.
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
            target = min(remaining * 0.45, budget / (4 + math.sqrt(n)))
            # Depth scales with available time, size and measured throughput; the number of states is limited only by the memory guard.
            r = 1
            memory_states = (64 * 1024 * 1024) // 32
            while r < n and (1 << (r + 1)) <= memory_states:
                nr = r + 1
                if nr * (1 << (nr - 1)) > dp_rate * target:
                    break
                r = nr
            if r >= 2 and clock() < deadline:
                if r == n:
                    slots = list(range(n))
                else:
                    k, _ = split(q)
                    boundary = min(n - 1, k)
                    mode = rng.randrange(3)
                    if mode == 0:
                        left = max(0, min(n - r, boundary - rng.randrange(r)))
                        slots = list(range(left, left + r))
                    elif mode == 1:
                        slots = sorted(rng.sample(range(n), r))
                    else:
                        width = min(n, max(r, int(r * (1 + math.log1p(stall + 1)))))
                        left = max(0, min(n - width, boundary - width // 2))
                        local_count = max(1, r // 2)
                        selected = set(rng.sample(range(left, left + width), local_count))
                        selected.add(boundary)
                        while len(selected) < r:
                            selected.add(rng.randrange(n))
                        slots = sorted(selected)
                trial, complete = exact_neighborhood(q, slots, min(deadline, clock() + min(remaining * 0.75, max(target * 2, 0.005))))
                if complete:
                    z = save(trial)
                    if r == n:
                        return {"order": best_order}
                    if z < cost:
                        q = trial
                        add_pool(q, z)
                        stall = 0 if best_cost < old_best else stall + 1
                        continue
            stall = 0 if best_cost < old_best else stall + 1
            if clock() >= deadline:
                break
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
