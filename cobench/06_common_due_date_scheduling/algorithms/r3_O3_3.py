"""Large-neighbourhood search with bidirectional set-sequence feedback
Continuous relaxation guides early-set enumeration, combined with exact straddling-job selection, destroy-and-rebuild and block insertion; sequence changes feed back into the next set neighbourhood, and the best permutation is kept by wall clock.
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
    started = clock()
    best_order = []
    try:
        n = max(0, int(instance["n"]))
        best_order = list(range(n))
        budget = max(0.0, float(time_limit_s))
        end = started + budget - min(0.10, budget * 0.03)
        if n < 2 or clock() >= end:
            return {"order": best_order}
        p = [int(v) for v in instance["processing_times"]]
        a = [int(v) for v in instance["earliness_penalties"]]
        b = [int(v) for v in instance["tardiness_penalties"]]
        d = int(instance["due_date"])
        rng = random.Random(561903)
        effort = math.log1p(budget)

        def cost(order):
            t = z = 0
            for j in order:
                t += p[j]
                z += a[j] * (d - t) if t < d else b[j] * (t - d)
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
        save(early)
        if d >= sum(p) or clock() >= end:
            return {"order": best_order}
        late = sorted(range(n), key=cmp_to_key(lc))
        save(late)
        if d <= min(p) or clock() >= end:
            return {"order": best_order}

        if np is None:
            # Without the numerical library, insertion search still runs with the same complete permutation and wall clock.
            current, current_cost = best_order[:], best_cost
            while clock() < end:
                j = rng.randrange(n)
                job = current[j]
                base = current[:j] + current[j + 1:]
                trial_best, trial_value = current, current_cost
                positions = list(range(n))
                rng.shuffle(positions)
                for k in positions:
                    if clock() >= end:
                        break
                    trial = base[:k] + [job] + base[k:]
                    z = save(trial)
                    if z < trial_value:
                        trial_best, trial_value = trial, z
                if trial_value < current_cost:
                    current, current_cost = trial_best, trial_value
                elif clock() < end:
                    current = best_order[:]
                    strength = min(n, max(2, int(1 + effort)))
                    ids = rng.sample(range(n), strength)
                    values = [current[k] for k in ids]
                    rng.shuffle(values)
                    for k, value in zip(ids, values):
                        current[k] = value
                    current_cost = save(current)
            return {"order": best_order}

        P = np.asarray(p, dtype=np.float64)
        A = np.asarray(a, dtype=np.float64)
        B = np.asarray(b, dtype=np.float64)
        eo = np.asarray(early, dtype=np.int64)
        lo = np.asarray(late, dtype=np.int64)
        pe, ae = P[eo], A[eo]
        pl, bl = P[lo], B[lo]
        maxp = float(np.max(P))
        totalp = float(np.sum(P))
        denom = np.maximum(P, 1.0)
        pp = np.cumsum(pl) - pl
        bb = np.cumsum(bl) - bl
        linear = np.empty(n)
        linear[lo] = (A[lo] + bl) * (d - pl) + pl * bb - bl * pp
        nrng = np.random.default_rng(561903)
        rows = max(1, min(int(12 + 16 * effort), max(1, 160000 // n)))
        memory_states = (48 * 1024 * 1024) // 48

        def product(x):
            # Implicit quadratic-kernel multiplication using only linear space.
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
                if t > d:
                    break
                mask[j] = True
            return mask

        def repair(masks, preference):
            masks = masks.copy()
            excess = np.maximum(0, np.sum(masks * P, axis=1) - d)
            if np.any(excess > 0):
                rank = np.argsort(preference)
                selected = masks[:, rank]
                lengths = selected * P[rank]
                remove = (np.cumsum(lengths, axis=1) - lengths) < excess[:, None]
                masks[:, rank] = selected & ~remove
            s = np.sum(masks * P, axis=1)
            tail = ~masks[:, lo]
            completion = s[:, None] + np.cumsum(tail * pl, axis=1)
            masks[:, lo] |= tail & (completion <= d)
            return masks

        def evaluate(masks, lower=None, work=None):
            # For a given early set, enumerate the exact corrections for all valid straddling jobs.
            if work is None:
                work = np.sum(masks * P, axis=1)
            lm = ~masks[:, lo]
            cp = np.cumsum(lm * pl, axis=1)
            cb = np.cumsum(lm * bl, axis=1)
            if lower is None:
                em = masks[:, eo]
                ce = np.cumsum(em * pe, axis=1)
                lower = np.sum(em * ae * (d - ce), axis=1)
                lower += np.sum(lm * bl * (work[:, None] + cp - d), axis=1)
            correction = pl * cb - bl * cp
            correction[~(lm & (work[:, None] + pl > d))] = np.inf
            pivots = np.argmin(correction, axis=1)
            values = lower + correction[np.arange(len(masks)), pivots]
            values[work > d] = np.inf
            return values, lo[pivots]

        def order_from(mask, pivot):
            h = int(pivot)
            return [j for j in early if mask[j]] + [h] + [j for j in late if not mask[j] and j != h]

        def inspect(masks):
            values, pivots = evaluate(masks)
            k = int(np.argmin(values))
            if not np.isfinite(values[k]):
                return None
            q = order_from(masks[k], pivots[k])
            return masks[k].copy(), save(q), q

        preference = np.log((B + 1.0) / (A + 1.0))
        priorities = (preference, B / denom, (B - A) / denom, B * P / (A + 1.0))
        for priority in priorities:
            if clock() >= end:
                return {"order": best_order}
            ranking = np.argsort(-priority)
            mask = np.zeros(n, dtype=bool)
            used = 0
            for j in ranking:
                if used + p[j] <= d:
                    mask[j] = True
                    used += p[j]
            prefix = np.zeros(n, dtype=bool)
            prefix[ranking] = np.cumsum(P[ranking]) <= d
            inspect(repair(np.stack((mask, prefix, membership(best_order))), priority))

        target = min(float(d), totalp)
        x = np.full(n, target / max(totalp, 1.0))
        relaxed = x.copy()

        def relaxation_step(until):
            nonlocal x, target, relaxed
            if clock() >= until:
                return
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
            if clock() < until:
                masks = nrng.random((rows, n)) < x
                inspect(repair(masks, x + nrng.random(n) / max(1.0, effort)))
            if -slope <= 1e-9 * max(1.0, abs(float(best_cost))) or step <= 1e-10:
                best_s = float(np.dot(membership(best_order), P))
                target = min(float(d), max(0.0, best_s + (nrng.random() - 0.5) * min(maxp, float(d))))
                x = np.full(n, target / max(totalp, 1.0))

        opening_end = min(end, started + budget * 0.12)
        while clock() < opening_end:
            relaxation_step(opening_end)

        def place_block(base, block, until, randomized=False):
            # All insertion positions are computed at once via suffix cost differences, without recomputing the permutation per position.
            if clock() >= until:
                return None
            ids = np.asarray(base, dtype=np.int64)
            completion = np.cumsum(P[ids])
            old = np.where(completion < d, A[ids] * (d - completion), B[ids] * (completion - d))
            length = sum(p[j] for j in block)
            shifted = completion + length
            changes = np.where(shifted < d, A[ids] * (d - shifted), B[ids] * (shifted - d)) - old
            values = np.empty(len(base) + 1)
            values[:-1] = np.cumsum(changes[::-1])[::-1]
            values[-1] = 0.0
            prefix = np.empty(len(base) + 1)
            prefix[0] = 0.0
            prefix[1:] = completion
            t = 0
            for j in block:
                if clock() >= until:
                    return None
                t += p[j]
                c = prefix + t
                values += np.where(c < d, a[j] * (d - c), b[j] * (c - d))
            if randomized:
                width = min(len(values), max(1, int(1 + effort)))
                top = np.argpartition(values, width - 1)[:width]
                position = int(top[rng.randrange(width)])
            else:
                position = int(np.argmin(values))
            return base[:position] + block + base[position:]

        def due_position(order):
            t = 0
            for k, j in enumerate(order):
                t += p[j]
                if t > d:
                    return k
            return n - 1

        def sequence_search(order, chosen, depth, stalled, until):
            # Set subspaces guide removal; membership changes after rebuilding in turn guide enumeration.
            old_mask = membership(order)
            count = min(n - 1, max(2, int((1 + effort) * math.sqrt(max(1, depth)) * (1 + 0.15 * math.log1p(stalled)))))
            boundary = due_position(order)
            near = order[max(0, boundary - count):min(n, boundary + count + 1)]
            guided = [int(j) for j in chosen]
            rng.shuffle(near)
            rng.shuffle(guided)
            removed = set(guided[:max(1, count // 3)])
            removed.update(near[:max(1, count // 3)])
            while len(removed) < count:
                if clock() >= until:
                    return old_mask, cost(order), order, []
                removed.add(rng.randrange(n))
            missing = list(removed)
            rng.shuffle(missing)
            if rng.random() < 0.5:
                missing.sort(key=lambda j: (a[j] + b[j]) * p[j], reverse=True)
            rebuilt = [j for j in order if j not in removed]
            for k, j in enumerate(missing):
                candidate = place_block(rebuilt, [j], until, k + 1 < len(missing) and rng.random() < 0.2)
                if candidate is None:
                    return old_mask, cost(order), order, missing
                rebuilt = candidate
            value = save(rebuilt)
            mask = membership(rebuilt)
            if clock() < until:
                canonical = inspect(mask[None, :])
                if canonical is not None and canonical[1] <= value:
                    mask, value, rebuilt = canonical
            block_depth = min(n - 1, max(1, int(1 + effort)))
            improved = True
            while improved and clock() < until:
                improved = False
                boundary = due_position(rebuilt)
                positions = [k for k, j in enumerate(rebuilt) if j in removed]
                positions += list(range(max(0, boundary - block_depth), min(n, boundary + block_depth + 1)))
                rng.shuffle(positions)
                for position in positions:
                    if clock() >= until:
                        break
                    length = rng.randint(1, min(block_depth, n - position))
                    block = rebuilt[position:position + length]
                    base = rebuilt[:position] + rebuilt[position + length:]
                    trial = place_block(base, block, until)
                    if trial is None:
                        break
                    z = save(trial)
                    if z < value:
                        rebuilt, value = trial, z
                        improved = True
                        break
            mask = membership(rebuilt)
            if clock() < until:
                canonical = inspect(mask[None, :])
                if canonical is not None and canonical[1] <= value:
                    mask, value, rebuilt = canonical
            changed = [int(j) for j in np.flatnonzero(mask != old_mask)]
            feedback = changed + [j for j in missing if j not in changed]
            return mask, value, rebuilt, feedback

        enum_rate = 1000000.0

        def neighborhood(mask, order, chosen, until, incumbent):
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
                    enum_rate *= 0.7
                    return mask, incumbent, order, False
                length = 1 << j
                interaction = np.empty(length)
                interaction[0] = 0.0
                for k in range(j):
                    bit = 1 << k
                    interaction[bit:2 * bit] = interaction[:bit] + pair[j, k]
                values[length:2 * length] = values[:length] + gradient[chosen[j]] - interaction
                work[length:2 * length] = work[:length] + cp[j]
            tolerance = 1e-7 * max(1.0, abs(float(incumbent)))
            valid = (work <= d) & (work > d - maxp) & (values <= float(incumbent) + tolerance)
            ids = np.flatnonzero(valid)
            if clock() >= until:
                enum_rate *= 0.7
                return mask, incumbent, order, False
            ids = ids[np.argsort(values[ids])]
            enum_rate = 0.65 * enum_rate + 0.35 * size / max(clock() - begin, 1e-6)
            answer, answer_value, answer_order = mask, incumbent, order
            bits = 1 << np.arange(r)
            complete = True
            for left in range(0, len(ids), rows):
                if clock() >= until:
                    complete = False
                    break
                states = ids[left:left + rows]
                if values[states[0]] > float(answer_value) + tolerance:
                    break
                masks = np.broadcast_to(base, (len(states), n)).copy()
                masks[:, chosen] = (states[:, None] & bits) != 0
                exact, pivots = evaluate(masks, values[states], work[states])
                ix = int(np.argmin(exact))
                if np.isfinite(exact[ix]) and exact[ix] <= float(answer_value) + tolerance:
                    q = order_from(masks[ix], pivots[ix])
                    z = save(q)
                    if z < answer_value:
                        answer, answer_value, answer_order = masks[ix].copy(), z, q
            return answer, answer_value, answer_order, complete

        def choose(mask, order, depth, stalled, rounds, feedback):
            if depth >= n:
                return np.arange(n, dtype=np.int64)
            g = (linear - product(mask)) / denom
            scale = max(1e-8, float(np.std(g)))
            noise = (1 + math.log1p(stalled)) / (1 + effort)
            mode = rounds % 4
            if mode == 0:
                priority = np.where(mask, g, -g) / scale + nrng.gumbel(size=n) * noise
            elif mode == 1:
                priority = np.where(mask, 1.0 - relaxed, relaxed) + nrng.gumbel(size=n) * noise
            elif mode == 2:
                boundary = due_position(order)
                distance = np.empty(n)
                distance[np.asarray(order, dtype=np.int64)] = np.abs(np.arange(n) - boundary)
                priority = -distance / max(1, depth) + nrng.gumbel(size=n) * noise
            else:
                priority = nrng.random(n)
            # Force some sequence changes to be accommodated; the remaining positions are still contested by relaxation and marginal opportunity.
            if feedback:
                forced = list(dict.fromkeys(feedback))[:max(1, depth // 2)]
                priority[forced] = float(np.max(priority)) + 1.0 + np.arange(len(forced), 0, -1)
            inside = np.flatnonzero(mask)
            outside = np.flatnonzero(~mask)
            ni = min(len(inside), depth // 2)
            no = min(len(outside), depth - ni)
            ni = min(len(inside), depth - no)
            ci = inside[np.argpartition(priority[inside], len(inside) - ni)[-ni:]] if ni else np.empty(0, dtype=np.int64)
            co = outside[np.argpartition(priority[outside], len(outside) - no)[-no:]] if no else np.empty(0, dtype=np.int64)
            selected = np.concatenate((ci, co))
            nrng.shuffle(selected)
            return selected

        result = inspect(membership(best_order)[None, :]) if clock() < end else None
        if result is None:
            return {"order": best_order}
        current, current_cost, current_order = result
        stalled = rounds = 0
        feedback = []
        while clock() < end:
            previous = best_cost
            remaining = end - clock()
            window = min(remaining * 0.60, budget * (1 + math.log1p(stalled)) / (6 + math.log2(n + 1)))
            allowed = max(2, min(memory_states, int(enum_rate * max(window * 0.35, 1e-6))))
            depth = min(n, max(1, int(math.log2(allowed))))
            chosen = choose(current, current_order, depth, stalled, rounds, feedback)
            sequence_end = min(end, clock() + window * 0.20)
            trial_mask, trial_cost, trial_order, feedback = sequence_search(current_order, chosen, depth, stalled, sequence_end)
            temperature = max(1.0, best_cost / max(1, n)) * (1 + math.log1p(stalled)) / (1 + effort)
            if trial_cost <= current_cost or rng.random() < math.exp(min(0.0, (current_cost - trial_cost) / temperature)):
                current, current_cost, current_order = trial_mask, trial_cost, trial_order
            if clock() >= end:
                break
            chosen = choose(current, current_order, depth, stalled, rounds, feedback)
            current, current_cost, current_order, complete = neighborhood(current, current_order, chosen, min(end, clock() + window * 0.80), current_cost)
            if len(chosen) == n and complete:
                return {"order": best_order}
            rounds += 1
            stalled = 0 if best_cost < previous else stalled + 1
            if clock() >= end:
                break
            # When the sequence local basin stagnates, apply relaxation-guided membership perturbation from the same best set.
            if stalled and rng.random() < stalled / (stalled + 2 + effort):
                anchor = membership(best_order)
                strength = min(0.5, (1 + math.log1p(stalled)) * (1 + effort) / n)
                samples = np.broadcast_to(anchor, (rows, n)).copy()
                samples ^= nrng.random((rows, n)) < strength
                result = inspect(repair(samples, relaxed + nrng.random(n)))
                if result is not None:
                    mask, z, order = result
                    if z <= current_cost or rng.random() < math.exp(min(0.0, (current_cost - z) / temperature)):
                        current, current_cost, current_order = mask, z, order
                    else:
                        current, current_cost, current_order = anchor, best_cost, best_order[:]
            if clock() < end:
                relaxation_step(min(end, clock() + window * 0.10))
            if best_cost < previous:
                current, current_cost, current_order = membership(best_order), best_cost, best_order[:]
        return {"order": best_order}
    except Exception:
        return {"order": best_order}
