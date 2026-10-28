"""Set-based large-neighbourhood search guided by continuous relaxation
Builds early sets from a fixed-processing-time continuous relaxation, enumerates adaptive binary subspaces and exactly selects the job straddling the due date, keeping the best complete permutation by wall clock.
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
    clock = time.perf_counter
    start = clock()
    best_order = []
    try:
        n = max(0, int(instance["n"]))
        best_order = list(range(n))
        budget = max(0.0, float(time_limit_s))
        end = start + budget - min(0.12, budget * 0.025)
        if n < 2 or clock() >= end:
            return {"order": best_order}
        p = [int(v) for v in instance["processing_times"]]
        a = [int(v) for v in instance["earliness_penalties"]]
        b = [int(v) for v in instance["tardiness_penalties"]]
        d = int(instance["due_date"])
        rng = random.Random(561903)

        def cost(order):
            t = z = 0
            for j in order:
                t += p[j]
                z += a[j] * max(0, d - t) + b[j] * max(0, t - d)
            return z

        best_cost = cost(best_order)

        def save(order):
            nonlocal best_order, best_cost
            z = cost(order)
            if z < best_cost:
                best_order, best_cost = list(order), z
            return z

        def ec(i, j):
            if (p[i] == 0) != (p[j] == 0):
                return 1 if p[i] == 0 else -1
            z = a[i] * p[j] - a[j] * p[i]
            return (z > 0) - (z < 0) if z else (i > j) - (i < j)

        def lc(i, j):
            if (p[i] == 0) != (p[j] == 0):
                return -1 if p[i] == 0 else 1
            z = p[i] * b[j] - p[j] * b[i]
            return (z > 0) - (z < 0) if z else (i > j) - (i < j)

        early = sorted(range(n), key=cmp_to_key(ec))
        late = sorted(range(n), key=cmp_to_key(lc))
        save(early)
        save(late)
        if d >= sum(p) or d <= 0 or clock() >= end:
            return {"order": best_order}
        if np is None:
            current = best_order[:]
            current_cost = best_cost
            while clock() < end:
                trial = current[:]
                i, j = rng.sample(range(n), 2)
                trial.insert(j, trial.pop(i))
                z = save(trial)
                temperature = max(1.0, best_cost / n) / (1.0 + (clock() - start) / max(budget, 1e-9))
                if z <= current_cost or rng.random() < math.exp(min(0.0, (current_cost - z) / temperature)):
                    current, current_cost = trial, z
            return {"order": best_order}

        P = np.asarray(p, dtype=np.float64)
        A = np.asarray(a, dtype=np.float64)
        B = np.asarray(b, dtype=np.float64)
        eo = np.asarray(early, dtype=np.int64)
        lo = np.asarray(late, dtype=np.int64)
        pe, ae = P[eo], A[eo]
        pl, bl = P[lo], B[lo]
        pp = np.cumsum(pl) - pl
        bb = np.cumsum(bl) - bl
        linear = np.empty(n)
        linear[lo] = (A[lo] + bl) * (d - pl) + pl * bb - bl * pp
        nrng = np.random.default_rng(561903)
        rows = max(1, min(int(8 + 8 * math.log1p(budget)), max(1, 200000 // n)))
        maxp = float(np.max(P))
        denom = np.maximum(P, 1.0)

        def product(x):
            # Matrix product of two max-ratio kernels without building an n×n matrix; diagonal entries are zero.
            out = np.empty(n)
            v = x[eo]
            xp, xa = v * pe, v * ae
            out[eo] = ae * (np.cumsum(xp) - xp) + pe * (np.sum(xa) - np.cumsum(xa))
            v = x[lo]
            xp, xb = v * pl, v * bl
            out[lo] += pl * (np.cumsum(xb) - xb) + bl * (np.sum(xp) - np.cumsum(xp))
            return out

        def membership(order):
            mask = np.zeros(n, dtype=bool)
            t = 0
            for j in order:
                t += p[j]
                if t <= d:
                    mask[j] = True
                else:
                    break
            return mask

        def repair(masks, preference):
            # Used only for construction; sets in the enumeration neighbourhood do not go through this repair.
            masks = masks.copy()
            excess = np.maximum(0, masks @ P - d)
            if np.any(excess > 0):
                rank = np.argsort(preference)
                selected = masks[:, rank]
                lengths = selected * P[rank]
                remove = (np.cumsum(lengths, axis=1) - lengths) < excess[:, None]
                masks[:, rank] = selected & ~remove
            s = masks @ P
            tail = ~masks[:, lo]
            completion = s[:, None] + np.cumsum(tail * pl, axis=1)
            masks[:, lo] |= tail & (completion <= d)
            return masks

        def evaluate(masks):
            # F(E) is a quadratic lower bound; the actual cost adds the cost of moving a valid straddling job to the first tardy position.
            em = masks[:, eo]
            completion = np.cumsum(em * pe, axis=1)
            s = np.sum(em * pe, axis=1)
            values = np.sum(em * ae * (d - completion), axis=1)
            lm = ~masks[:, lo]
            lp = lm * pl
            lb = lm * bl
            before_p = np.cumsum(lp, axis=1) - lp
            before_b = np.cumsum(lb, axis=1) - lb
            values += np.sum(lb * (s[:, None] + before_p + pl - d), axis=1)
            correction = pl * before_b - bl * before_p
            valid = lm & (s[:, None] + pl > d)
            correction[~valid] = np.inf
            pivots = np.argmin(correction, axis=1)
            values += correction[np.arange(len(masks)), pivots]
            values[s > d] = np.inf
            return values, lo[pivots]

        def order_from(mask, pivot):
            h = int(pivot)
            return [j for j in early if mask[j]] + [h] + [j for j in late if not mask[j] and j != h]

        def inspect(masks):
            values, pivots = evaluate(masks)
            ix = int(np.argmin(values))
            if not np.isfinite(values[ix]):
                return None
            q = order_from(masks[ix], pivots[ix])
            z = save(q)
            return masks[ix].copy(), z

        best_mask = membership(best_order)
        preference = np.log((B + 1.0) / (A + 1.0))
        rank = np.argsort(-preference)
        seed = np.zeros(n, dtype=bool)
        seed[rank] = np.cumsum(P[rank]) <= d
        inspect(repair(np.stack((best_mask, seed)), preference))
        best_mask = membership(best_order)

        # On the hyperplane of fixed total early processing time, the quadratic extension is convex.
        # The Frank-Wolfe linear subproblem is a continuous knapsack; its solution is used for randomised rounding and subspace selection.
        relaxation_end = min(end, start + budget * 0.15)
        target = min(float(d), float(np.sum(P)))
        x = np.full(n, target / max(float(np.sum(P)), 1.0))
        relaxed = x.copy()
        while clock() < relaxation_end:
            g = linear - product(x)
            ranking = np.argsort(g / denom)
            v = np.zeros(n)
            positive = ranking[P[ranking] > 0]
            cumulative = np.cumsum(P[positive])
            v[positive] = np.clip((target - cumulative + P[positive]) / P[positive], 0, 1)
            v[(P == 0) & (g < 0)] = 1.0
            direction = v - x
            slope = float(np.dot(g, direction))
            curvature = -float(np.dot(direction, product(direction)))
            step = min(1.0, max(0.0, -slope / curvature)) if curvature > 1e-12 else 1.0
            x = np.clip(x + step * direction, 0, 1)
            relaxed = x.copy()
            if clock() >= relaxation_end:
                break
            samples = nrng.random((rows, n)) < x
            inspect(repair(samples, x + nrng.random(n) / max(1.0, math.log1p(budget))))
            if -slope <= 1e-8 * max(1.0, abs(float(best_cost))) or step <= 1e-10:
                target = max(0.0, d - nrng.random() * min(maxp, float(d)))
                x = np.full(n, target / max(float(np.sum(P)), 1.0))
        if clock() >= end:
            return {"order": best_order}

        current = membership(best_order)
        current_cost = best_cost
        enum_rate = 1000000.0
        stalled = 0
        rounds = 0
        memory_states = (48 * 1024 * 1024) // 40

        def neighborhood(mask, chosen, until, incumbent):
            nonlocal enum_rate
            begin = clock()
            r = len(chosen)
            size = 1 << r
            base = mask.copy()
            base[chosen] = False
            gradient = linear - product(base)
            em = base[eo]
            s0 = float(np.dot(base, P))
            f0 = float(np.sum(em * ae * (d - np.cumsum(em * pe))))
            lm = ~base[lo]
            f0 += float(np.sum(lm * bl * (s0 + np.cumsum(lm * pl) - d)))
            cp, ca, cb = P[chosen], A[chosen], B[chosen]
            pair = np.maximum(cp[:, None] * ca, ca[:, None] * cp)
            pair += np.maximum(cp[:, None] * cb, cb[:, None] * cp)
            values = np.empty(size)
            work = np.empty(size)
            values[0], work[0] = f0, s0
            for j in range(r):
                if clock() >= until:
                    return mask, incumbent, False
                length = 1 << j
                interaction = np.empty(length)
                interaction[0] = 0.0
                for k in range(j):
                    bit = 1 << k
                    interaction[bit:2 * bit] = interaction[:bit] + pair[j, k]
                values[length:2 * length] = values[:length] + gradient[chosen[j]] - interaction
                work[length:2 * length] = work[:length] + cp[j]
            valid = (work <= d) & (work > d - maxp)
            tolerance = 1e-8 * max(1.0, abs(float(incumbent)))
            valid &= values <= float(incumbent) + tolerance
            ids = np.flatnonzero(valid)
            if clock() >= until:
                return mask, incumbent, False
            ids = ids[np.argsort(values[ids])]
            elapsed = clock() - begin
            enum_rate = 0.65 * enum_rate + 0.35 * size / max(elapsed, 1e-6)
            answer, answer_cost = mask, incumbent
            complete = True
            for left in range(0, len(ids), rows):
                if clock() >= until:
                    complete = False
                    break
                states = ids[left:left + rows]
                if values[states[0]] > float(answer_cost) + tolerance:
                    break
                masks = np.broadcast_to(base, (len(states), n)).copy()
                masks[:, chosen] = (states[:, None] & (1 << np.arange(r))) != 0
                result = inspect(masks)
                if result is not None and result[1] < answer_cost:
                    answer, answer_cost = result
            return answer, answer_cost, complete

        while clock() < end:
            remaining = end - clock()
            window = min(remaining * 0.55, budget * (1 + math.log1p(stalled)) / (5 + math.log2(n + 1)))
            allowed = max(2, min(memory_states, int(enum_rate * max(window * 0.45, 1e-6))))
            depth = min(n, max(1, int(math.log2(allowed))))
            if depth == n:
                chosen = np.arange(n)
            else:
                g = (linear - product(current)) / denom
                inside = np.flatnonzero(current)
                outside = np.flatnonzero(~current)
                scale = max(1e-8, float(np.std(g)))
                noise = scale * (1 + math.log1p(stalled)) / (1 + math.log1p(budget))
                # Alternate among marginal opportunities, relaxation fractions and global random subspaces.
                mode = rounds % 3
                if mode == 0:
                    priority = np.where(current, g, -g) + nrng.gumbel(size=n) * noise
                elif mode == 1:
                    priority = np.where(current, 1.0 - relaxed, relaxed) + nrng.gumbel(size=n) / (1 + math.log1p(budget))
                else:
                    priority = nrng.random(n)
                ni = min(len(inside), max(0, depth // 2))
                no = min(len(outside), depth - ni)
                ni = min(len(inside), depth - no)
                ci = inside[np.argsort(priority[inside])[-ni:]] if ni else np.empty(0, dtype=np.int64)
                co = outside[np.argsort(priority[outside])[-no:]] if no else np.empty(0, dtype=np.int64)
                chosen = np.concatenate((ci, co))
                nrng.shuffle(chosen)
            previous = best_cost
            current, current_cost, complete = neighborhood(current, chosen, min(end, clock() + window), current_cost)
            rounds += 1
            if len(chosen) == n and complete:
                return {"order": best_order}
            if best_cost < previous:
                stalled = 0
            else:
                stalled += 1
            if clock() >= end:
                break
            if stalled and rng.random() < stalled / (stalled + 2 + math.log1p(budget)):
                anchor = membership(best_order)
                strength = min(0.5, (1 + math.log1p(stalled)) * (1 + math.log1p(budget)) / max(n, 1))
                samples = np.broadcast_to(anchor, (rows, n)).copy()
                samples ^= nrng.random((rows, n)) < strength
                samples = repair(samples, relaxed + nrng.random(n))
                result = inspect(samples)
                if result is not None:
                    trial_mask, trial_cost = result
                    temperature = max(1.0, best_cost / max(n, 1)) * math.log1p(stalled) / (1 + math.log1p(budget))
                    if trial_cost <= current_cost or rng.random() < math.exp(min(0.0, (current_cost - trial_cost) / temperature)):
                        current, current_cost = trial_mask, trial_cost
                    else:
                        current = membership(best_order)
                        current_cost = best_cost
        return {"order": best_order}
    except Exception:
        return {"order": best_order}
