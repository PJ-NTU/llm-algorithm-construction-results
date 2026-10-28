"""Joint early-set search under straddling constraints
Combines two-sided dominance ordering, incremental swap and insertion descent, quadratic subset side-assignment enumeration and straddling-job replacement, saving the best complete permutation within the wall-clock budget.
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
        n = max(0, int(instance["n"]))
        best_order = list(range(n))
        budget = max(0.0, float(time_limit_s))
        if not math.isfinite(budget):
            budget = 0.0
        clock = time.perf_counter
        deadline = started + budget - min(0.12, max(0.002, budget * 0.02))
        if n < 2 or clock() >= deadline:
            return {"order": best_order}
        p = [int(v) for v in instance["processing_times"]]
        a = [int(v) for v in instance["earliness_penalties"]]
        b = [int(v) for v in instance["tardiness_penalties"]]
        d = int(instance["due_date"])
        rng = random.Random(1729)

        def value(order):
            elapsed = result = 0
            for j in order:
                elapsed += p[j]
                gap = elapsed - d
                result += b[j] * gap if gap > 0 else -a[j] * gap
            return result

        best_cost = value(best_order)
        last_improvement = clock()

        def save(order, cost=None):
            nonlocal best_order, best_cost, last_improvement
            if cost is None:
                cost = value(order)
            if cost < best_cost:
                best_order, best_cost = list(order), cost
                last_improvement = clock()
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
            save(early_order)
            return {"order": best_order}
        if d <= min(p):
            save(late_order)
            return {"order": best_order}
        early_rank, late_rank = [0] * n, [0] * n
        for k, j in enumerate(early_order):
            early_rank[j] = k
        for k, j in enumerate(late_order):
            late_rank[j] = k

        def split(order):
            k = s = 0
            while k < n and s + p[order[k]] <= d:
                s += p[order[k]]
                k += 1
            return k, s

        def canonical(order):
            k, s = split(order)
            early = sorted(order[:k], key=early_rank.__getitem__)
            tail = sorted(order[k:], key=late_rank.__getitem__)
            if not tail:
                return early
            # With the early set fixed, enumerate the first tardy job that may straddle the due date.
            pp = bb = 0
            h, extra = tail[0], None
            for j in tail:
                if s + p[j] >= d:
                    z = p[j] * bb - b[j] * pp
                    if extra is None or z < extra:
                        extra, h = z, j
                pp += p[j]
                bb += b[j]
            tail.remove(h)
            if s + p[h] == d:
                return sorted(early + [h], key=early_rank.__getitem__) + tail
            return early + [h] + tail

        seeds = []
        initial = [best_order, early_order, late_order,
                   sorted(range(n), key=lambda j: (a[j] / max(1, b[j]), -p[j])),
                   sorted(range(n), key=lambda j: (-p[j], a[j] - b[j])),
                   sorted(range(n), key=lambda j: (a[j] - b[j]) / max(1, p[j]))]
        for order in initial:
            if clock() >= deadline:
                break
            q = canonical(order)
            seeds.append((save(q), q))
        # Capacity filling and prefix truncation complement each other, preventing long jobs from blocking later jobs that could be early.
        for order in initial[3:]:
            if clock() >= deadline:
                break
            left, right, s = [], [], 0
            for j in order:
                if s + p[j] <= d:
                    left.append(j)
                    s += p[j]
                else:
                    right.append(j)
            q = canonical(left + right)
            seeds.append((save(q), q))
        if clock() >= deadline:
            return {"order": best_order}

        intensity = math.log1p(budget)
        if np is None:
            q, cost = list(best_order), best_cost
            while clock() < deadline:
                trial = list(q)
                i, j = rng.sample(range(n), 2)
                if rng.random() < 0.5:
                    trial[i], trial[j] = trial[j], trial[i]
                else:
                    trial.insert(j, trial.pop(i))
                trial = canonical(trial)
                z = save(trial)
                temperature = max(1.0, best_cost / n / (2.0 + intensity))
                if z <= cost or rng.random() < math.exp(-min(700.0, (z - cost) / temperature)):
                    q, cost = trial, z
            return {"order": best_order}

        bound = (total_p + abs(d)) * (sum(abs(x) for x in a) + sum(abs(x) for x in b))
        exact_numbers = max(bound, total_p, abs(d), max(a), max(b)) < (1 << 58)
        dtype = np.int64 if exact_numbers else np.float64
        infinity = (1 << 62) if exact_numbers else math.inf
        P, A, B = (np.asarray(v, dtype=dtype) for v in (p, a, b))
        eo, lo = np.asarray(early_order), np.asarray(late_order)
        er, lr = np.asarray(early_rank), np.asarray(late_rank)
        zero = np.zeros(1, dtype=dtype)

        def prefix(v):
            return np.concatenate((zero, np.cumsum(v)))

        def side_data(order, pivot=None):
            k, s = split(order)
            h = order[k] if pivot is None else pivot
            em = np.zeros(n, dtype=bool)
            em[order[:k]] = True
            if em[h]:
                em[h] = False
                s -= p[h]
            lm = ~em
            lm[h] = False
            ei, li = eo[em[eo]], lo[lm[lo]]
            ep = prefix(P[eo] * em[eo])
            ea = prefix(A[eo] * em[eo])
            lp = prefix(P[lo] * lm[lo])
            lb = prefix(B[lo] * lm[lo])
            before_e, after_a = ep[er], ea[-1] - ea[er + 1]
            before_l, before_b = lp[lr], lb[lr]
            re = -A * (d - before_e - P) + P * after_a
            ae = -re
            remove = re - P * (B[h] + before_b) + B * (s + P[h] + before_l - d)
            add = ae + P * (B[h] + before_b) - B * (s + P[h] + before_l + P - d)
            return em, ei, li, s, h, re, ae, remove, add

        def assemble(em, h):
            q = [j for j in early_order if em[j]]
            q.append(int(h))
            q.extend(j for j in late_order if not em[j] and j != h)
            return canonical(q)

        guide_key, guide_score = None, None

        def structural_move(order, until):
            nonlocal guide_key, guide_score
            em, ei, li, s, h, re, ae, remove, add = side_data(order)
            gain, move = 0, None

            def consider(vals, ids, kind):
                nonlocal gain, move
                if len(ids):
                    pos = int(np.argmin(vals))
                    z = vals[pos]
                    if z < gain:
                        gain, move = z, (kind, int(ids[pos]))

            if len(ei):
                consider(np.where(s - P[ei] + P[h] >= d, remove[ei], infinity), ei, "remove")
                vals = re[ei] + ae[h] + np.maximum(P[ei] * A[h], P[h] * A[ei])
                vals += (B[ei] - B[h]) * (s + P[h] - d)
                consider(np.where(s - P[ei] + P[h] <= d, vals, infinity), ei, "pivot")
            if len(li):
                consider(np.where(s + P[li] <= d, add[li], infinity), li, "add")
            # Finish cheap single-job moves first, then pay for the full swap matrix.
            if move is None and len(ei) and len(li):
                scores = np.full(n, infinity, dtype=dtype)
                block = max(1, min(len(ei), 131072 // len(li)))
                offset = rng.randrange(len(ei))
                scan = np.concatenate((ei[offset:], ei[:offset]))
                pj, aj, bj = P[li][None, :], A[li][None, :], B[li][None, :]
                complete = True
                for start in range(0, len(ei), block):
                    if clock() >= until:
                        complete = False
                        break
                    ids = scan[start:start + block]
                    pi = P[ids][:, None]
                    vals = remove[ids][:, None] + add[li][None, :]
                    vals += np.maximum(pi * aj, pj * A[ids][:, None])
                    vals += np.maximum(pi * bj, pj * B[ids][:, None])
                    new_s = s - pi + pj
                    vals[(new_s > d) | (new_s + P[h] < d)] = infinity
                    scores[ids] = np.min(vals, axis=1)
                    scores[li] = np.minimum(scores[li], np.min(vals, axis=0))
                    z = int(np.argmin(vals))
                    if vals.flat[z] < gain:
                        row, col = divmod(z, len(li))
                        gain, move = vals.flat[z], ("swap", int(ids[row]), int(li[col]))
                if complete and move is None:
                    guide_key, guide_score = tuple(order), scores
            if move is None:
                return None
            kind, i = move[:2]
            if kind == "remove":
                em[i] = False
            elif kind == "add":
                em[i] = True
            elif kind == "pivot":
                em[i], em[h], h = False, True, i
            else:
                em[i], em[move[2]] = False, True
            return assemble(em, h)

        def insertion_move(order, until):
            ids = np.asarray(order)
            pp, aa, bb = P[ids], A[ids], B[ids]
            completion = np.cumsum(pp)
            starts = completion - pp
            old = aa * np.maximum(d - completion, 0) + bb * np.maximum(completion - d, 0)
            positions = np.arange(n)
            offset = rng.randrange(n)
            for step in range(n):
                if clock() >= until:
                    return None, False
                i = (offset + step) % n
                shift = pp[i]
                plus = aa * np.maximum(d - completion - shift, 0) + bb * np.maximum(completion + shift - d, 0) - old
                minus = aa * np.maximum(d - completion + shift, 0) + bb * np.maximum(completion - shift - d, 0) - old
                cp, cm = prefix(plus), prefix(minus)
                dest = np.where(positions < i, starts + shift, completion)
                vals = aa[i] * np.maximum(d - dest, 0) + bb[i] * np.maximum(dest - d, 0) - old[i]
                vals[:i] += cp[i] - cp[:i]
                vals[i + 1:] += cm[i + 2:] - cm[i + 1]
                vals[i] = 0
                j = int(np.argmin(vals))
                if vals[j] < 0:
                    trial = list(order)
                    trial.insert(j, trial.pop(i))
                    return canonical(trial), True
            return None, True

        capacity = max(2, int(2 + intensity * math.sqrt(math.log2(n + 1))))
        optima = set()

        def descend(order, until):
            q, cost = list(order), value(order)
            if tuple(q) in optima:
                return q, cost
            while clock() < until:
                trial = structural_move(q, until)
                if trial is not None:
                    z = save(trial)
                    if z < cost:
                        q, cost = trial, z
                        continue
                if clock() >= until:
                    break
                trial, complete = insertion_move(q, until)
                if trial is not None:
                    z = save(trial)
                    if z < cost:
                        q, cost = trial, z
                        continue
                if complete and trial is None:
                    if len(optima) >= 4 * capacity:
                        optima.clear()
                    optima.add(tuple(q))
                break
            return q, cost

        enum_rate = 3000000.0
        memory_states = (96 * 1024 * 1024) // 24

        def joint_membership(order, data, selected, until):
            nonlocal enum_rate
            begin = clock()
            em, ei, li, s, h, re, ae, remove, add = data
            ids = np.asarray(selected, dtype=np.intp)
            r = len(ids)
            size = 1 << r
            if clock() >= until or size > memory_states:
                return order, False
            pp, aa, bb = P[ids], A[ids], B[ids]
            pair = np.maximum(pp[:, None] * aa[None, :], aa[:, None] * pp[None, :])
            pair += np.maximum(pp[:, None] * bb[None, :], bb[:, None] * pp[None, :])
            np.fill_diagonal(pair, 0)
            flags = em[ids]
            # First put all selected jobs on the tardy side, then enumerate the subsets that rejoin the early side.
            linear = np.where(flags, -remove[ids], add[ids]) + np.sum(pair[:, flags], axis=1)
            base_time = s - sum(p[int(j)] for j in ids[flags])
            lower, upper = d - p[h] - base_time, d - base_time
            costs = np.empty(size, dtype=dtype)
            times = np.empty(size, dtype=dtype)
            interaction = np.empty(max(1, size // 2), dtype=dtype)
            costs[0], times[0] = 0, 0
            for j in range(r):
                if clock() >= until:
                    return order, False
                bit = 1 << j
                interaction[0] = 0
                for i in range(j):
                    if clock() >= until:
                        return order, False
                    step = 1 << i
                    interaction[step:2 * step] = interaction[:step] + pair[j, i]
                costs[bit:2 * bit] = costs[:bit] + linear[j] - interaction[:bit]
                times[bit:2 * bit] = times[:bit] + pp[j]
            best_mask, minimum, complete = None, infinity, True
            # Screen feasible subsets in chunks, so improvements already found can still be returned near the deadline.
            for start in range(0, size, 131072):
                if clock() >= until:
                    complete = False
                    break
                end = min(size, start + 131072)
                tt = times[start:end]
                vals = np.where((tt >= lower) & (tt <= upper), costs[start:end], infinity)
                pos = int(np.argmin(vals))
                if vals[pos] < minimum:
                    minimum, best_mask = vals[pos], start + pos
            elapsed = clock() - begin
            if complete and elapsed > 0:
                enum_rate = 0.6 * enum_rate + 0.4 * size / elapsed
            elif not complete:
                enum_rate *= 0.8
            if best_mask is None:
                return order, complete
            new_em = em.copy()
            for i, j in enumerate(selected):
                new_em[j] = bool(best_mask & (1 << i))
            # Even in the floating-point fallback, capacity is re-verified with integers, forbidding invalid straddling partitions.
            early_time = sum(p[j] for j in range(n) if new_em[j])
            if early_time > d or early_time + p[h] < d:
                return order, False
            trial = assemble(new_em, h)
            save(trial)
            return trial, complete

        def select_jobs(order, data, r, stall, old_h):
            em, ei, li, s, h, re, ae, remove, add = data
            if r == n - 1:
                return [j for j in range(n) if j != h]
            ne = min(len(ei), r // 2)
            nl = min(len(li), r - ne)
            ne = min(len(ei), r - nl)
            mode = rng.randrange(4)
            if mode == 0 and guide_key == tuple(order) and h == old_h:
                score = guide_score
            else:
                score = np.where(em, remove, add) / np.maximum(P, 1)
            chosen = []
            for group, count in ((ei, ne), (li, nl)):
                if count == 0:
                    continue
                jobs = [int(j) for j in group]
                if mode == 1:
                    # Jobs near the due date on both sides, plus a small amount of global sampling.
                    ranked = list(reversed(jobs)) if em[jobs[0]] else jobs
                elif mode == 2:
                    ranked = list(jobs)
                    rng.shuffle(ranked)
                else:
                    ranked = sorted(jobs, key=lambda j: score[j])
                width = min(len(jobs), max(count, int(count * (1.4 + math.log1p(stall) / (1 + intensity)))))
                biased = min(count, max(1, (2 * count) // 3))
                take = rng.sample(ranked[:width], biased)
                used = set(take)
                remaining = [j for j in jobs if j not in used]
                take.extend(rng.sample(remaining, count - biased))
                chosen.extend(take)
            if old_h != h and old_h not in chosen:
                chosen[rng.randrange(len(chosen))] = old_h
            return chosen

        pool = []

        def add_pool(order, cost):
            signature = tuple(order)
            if any(entry[2] == signature for entry in pool):
                return
            pool.append((cost, list(order), signature))
            pool.sort(key=lambda entry: entry[0])
            if len(pool) > capacity:
                pool.pop()

        for cost, order in seeds:
            add_pool(order, cost)
        quantum = budget / (6.0 + math.sqrt(n) + intensity)
        q, current_cost = descend(best_order, min(deadline, clock() + budget * 0.30))
        add_pool(q, current_cost)
        stall, covered = 0, set()
        while clock() < deadline:
            old_best = best_cost
            remaining = deadline - clock()
            target = min(remaining * 0.38, quantum * (0.8 + math.log1p(stall) / (1 + intensity)))
            r = 1
            while r < n - 1 and (1 << (r + 1)) <= memory_states:
                if (1 << (r + 1)) > enum_rate * target:
                    break
                r += 1
            k, _ = split(q)
            old_h = q[k]
            h = old_h
            if r == n - 1 and exact_numbers:
                pending = [j for j in range(n) if j not in covered]
                if not pending:
                    return {"order": best_order}
                h = old_h if old_h in pending else rng.choice(pending)
            elif rng.random() < min(0.5, stall / (stall + 2.0 + intensity)):
                h = rng.randrange(n)
            data = side_data(q, h)
            selected = select_jobs(q, data, r, stall, old_h)
            trial, complete = joint_membership(q, data, selected,
                                               min(deadline, clock() + max(target * 1.7, budget * 0.001)))
            z = save(trial)
            if complete and r == n - 1 and exact_numbers:
                covered.add(h)
                if len(covered) == n:
                    return {"order": best_order}
                q, current_cost = list(best_order), best_cost
                continue
            if z < current_cost:
                q, current_cost = trial, z
                if clock() < deadline:
                    q, current_cost = descend(q, min(deadline, clock() + quantum))
                add_pool(q, current_cost)
                stall = 0 if best_cost < old_best else stall + 1
                continue
            stall = 0 if best_cost < old_best else stall + 1
            if clock() >= deadline:
                break
            # Keep the elite while perturbing adaptively between different early sets.
            if pool and rng.random() < 0.28:
                base = rng.choice(pool)[1]
            else:
                base = best_order
            trial = list(base)
            k, _ = split(trial)
            strength = min(n, max(2, int((1 + intensity) * math.sqrt(1 + math.log1p(stall)))))
            for _ in range(rng.randint(1, strength)):
                if clock() >= deadline:
                    break
                if 0 < k < n and rng.random() < 0.85:
                    i, j = rng.randrange(k), rng.randrange(k, n)
                else:
                    i, j = rng.sample(range(n), 2)
                if rng.random() < 0.8:
                    trial[i], trial[j] = trial[j], trial[i]
                else:
                    trial.insert(j, trial.pop(i))
            trial = canonical(trial)
            z = save(trial)
            if clock() < deadline:
                trial, z = descend(trial, min(deadline, clock() + quantum))
            add_pool(trial, z)
            stalled_time = (clock() - last_improvement) / max(quantum, budget * 0.01)
            temperature = max(1.0, best_cost / max(1, n) * (0.02 + 0.08 * math.log1p(stalled_time)))
            gap = z - current_cost
            if gap <= 0 or rng.random() < math.exp(-min(700.0, gap / temperature)):
                q, current_cost = trial, z
            else:
                q, current_cost = list(best_order), best_cost
        return {"order": best_order}
    except Exception:
        # The complete permutation is established before any optional search, numerical conversion or dependency use.
        return {"order": best_order}
