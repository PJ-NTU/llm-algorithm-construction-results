"""Pattern column generation with fractional integer allocation
Generates patterns with edge-contact packing and beam search, extends them guided by demand shadow prices, and uses fractional integer optimisation to jointly choose sheet types, pattern counts and piece quantities.
"""
import math
import random
import time


def solve(instance: dict, time_limit_s: float) -> dict:
    start = time.perf_counter()
    best = {"selected_stock_types": [], "stock_instances": [], "piece_counts": {}}
    try:
        S, m = int(instance["S"]), int(instance["m"])
        best = {"selected_stock_types": [0] * S, "stock_instances": [],
                "piece_counts": {"piece_" + str(i): 0 for i in range(m)}}
        budget = max(0.0, float(time_limit_s))
        deadline = start + budget * 0.965
        stocks = instance.get("stocks", [])
        pieces = instance.get("pieces", [])
        L = [float(stocks[s]["length"] if stocks else instance["length"][s]) for s in range(S)]
        W = [float(stocks[s]["width"] if stocks else instance["width"][s]) for s in range(S)]
        A = [L[s] * W[s] for s in range(S)]
        pl = [float(instance["piece_length"][i] if "piece_length" in instance else pieces[i]["length"]) for i in range(m)]
        pw = [float(instance["piece_width"][i] if "piece_width" in instance else pieces[i]["width"]) for i in range(m)]
        pa = [float(instance["piece_area"][i]) if "piece_area" in instance else pl[i] * pw[i] for i in range(m)]
        lo = [int(instance["min"][i] if "min" in instance else pieces[i]["min"]) for i in range(m)]
        hi = [int(instance["max"][i] if "max" in instance else pieces[i]["max"]) for i in range(m)]
        eps = 1e-9
        if S == 0 or m == 0 or not any(hi):
            return best
        rng = random.Random(72463 + m * 37 + S)
        strength = 1.0 + math.log1p(budget)
        masks, fits = [], []
        required = sum(1 << i for i in range(m) if lo[i] > 0)
        for s in range(S):
            row = [i for i in range(m) if hi[i] > 0 and
                   ((pl[i] <= L[s] + eps and pw[i] <= W[s] + eps) or
                    (pw[i] <= L[s] + eps and pl[i] <= W[s] + eps))]
            fits.append(row)
            masks.append(sum(1 << i for i in row))
        rank = sorted(range(S), key=lambda s: (-sum(pa[i] for i in fits[s]) / max(A[s], eps), A[s]))
        allowed = None
        for s in rank:
            if masks[s] & required == required and fits[s]:
                allowed = (s,)
                break
        if allowed is None:
            for a_index, s in enumerate(rank):
                for t in rank[a_index + 1:]:
                    if (masks[s] | masks[t]) & required == required:
                        allowed = (s, t)
                        break
                if allowed is not None:
                    break
        if allowed is None:
            return best

        # Piece types within a pattern are distinct; after integer replication any piece can be removed independently.
        pools = [[] for _ in range(S)]
        best_ratio = -1.0
        best_pair = allowed
        pattern_limit = max(12, int((4.0 + 5.0 * strength) * math.sqrt(max(1, m))))
        weights = pa[:]

        def publish(entries):
            nonlocal best, best_ratio, best_pair
            counts = [0] * m
            coverage = [0] * m
            used = set()
            for s, rects, number in entries:
                if number <= 0:
                    continue
                used.add(s)
                for p in rects:
                    coverage[p[0]] += number
            if len(used) > 2 or any(coverage[i] < lo[i] for i in range(m)):
                return
            target = [min(hi[i], coverage[i]) for i in range(m)]
            denominator = sum(A[s] * number for s, rects, number in entries)
            numerator = sum(pa[i] * target[i] for i in range(m))
            if denominator <= 0 or numerator / denominator < best_ratio - 1e-12:
                return
            bins = []
            selected = [0] * S
            for s, rects, number in entries:
                for _ in range(number):
                    places = []
                    for pid, x, y, w, h, rot in rects:
                        if counts[pid] < target[pid]:
                            counts[pid] += 1
                            places.append({"piece_type_id": pid, "x": float(x), "y": float(y),
                                           "is_rotated": int(rot)})
                    if places:
                        bins.append({"stock_type_id": int(s), "placements": places})
                        selected[s] = 1
            denominator = sum(A[b["stock_type_id"]] for b in bins)
            ratio = numerator / denominator if denominator > 0 else 0.0
            if ratio > best_ratio + 1e-12 or (abs(ratio - best_ratio) <= 1e-12 and len(bins) < len(best["stock_instances"])):
                best_ratio = ratio
                best_pair = tuple(s for s in range(S) if selected[s]) or allowed
                best = {"selected_stock_types": selected, "stock_instances": bins,
                        "piece_counts": {"piece_" + str(i): counts[i] for i in range(m)}}

        def add_pattern(s, rects):
            if not rects:
                return False
            mask = sum(1 << p[0] for p in rects)
            pool = pools[s]
            if any(mask & item[0] == mask for item in pool):
                return False
            pool[:] = [item for item in pool if item[0] & mask != item[0]]
            pool.append((mask, tuple(rects)))
            if len(pool) > pattern_limit:
                # Keep carriers of different scarce types and discard crowded, low-value patterns.
                freq = [0] * m
                for _, ps in pool:
                    for p in ps:
                        freq[p[0]] += 1
                victim = min(range(len(pool)), key=lambda j: sum(
                    (pa[p[0]] + weights[p[0]]) / max(1, freq[p[0]]) for p in pool[j][1]))
                pool.pop(victim)
            return True

        initial = []
        for i in range(m):
            if lo[i] <= 0:
                continue
            s = min((s for s in allowed if masks[s] & (1 << i)), key=lambda s: A[s])
            rot = int(not (pl[i] <= L[s] + eps and pw[i] <= W[s] + eps))
            w, h = (pw[i], pl[i]) if rot else (pl[i], pw[i])
            rects = ((i, 0.0, 0.0, w, h, rot),)
            initial.append((s, rects, lo[i]))
            add_pattern(s, rects)
        if not initial:
            candidates = [(pa[i] / A[s], s, i) for s in rank for i in fits[s] if A[s] > 0]
            if candidates:
                _, s, i = max(candidates)
                rot = int(not (pl[i] <= L[s] + eps and pw[i] <= W[s] + eps))
                w, h = (pw[i], pl[i]) if rot else (pl[i], pw[i])
                initial = [(s, ((i, 0.0, 0.0, w, h, rot),), 1)]
                add_pattern(s, initial[0][1])
        publish(initial)
        if time.perf_counter() >= deadline:
            return best

        def locations(s, rects, pid, axis, mode, limit, stop):
            # Candidate x-coordinates come from sheet edges or existing piece edges; the lowest collision-free height is found per segment.
            U, V = (W[s], L[s]) if axis else (L[s], W[s])
            local = [(p[2], p[1], p[4], p[3]) if axis else (p[1], p[2], p[3], p[4]) for p in rects]
            ordered = sorted(local, key=lambda p: p[1])
            mx = max((p[0] + p[2] for p in local), default=0.0)
            my = max((p[1] + p[3] for p in local), default=0.0)
            choices = []
            for rot in (0, 1):
                if rot and abs(pl[pid] - pw[pid]) < eps:
                    continue
                ow, oh = (pw[pid], pl[pid]) if rot else (pl[pid], pw[pid])
                w, h = (oh, ow) if axis else (ow, oh)
                if w > U + eps or h > V + eps:
                    continue
                xs = {0.0, max(0.0, U - w)}
                for x, y, a, b in local:
                    if x + a + w <= U + eps:
                        xs.add(min(x + a, max(0.0, U - w)))
                    if x >= w - eps:
                        xs.add(max(0.0, x - w))
                for x in sorted(xs):
                    if time.perf_counter() >= stop:
                        return []
                    y = 0.0
                    for rx, ry, rw, rh in ordered:
                        if x + w <= rx + eps or rx + rw <= x + eps:
                            continue
                        if y + h <= ry + eps:
                            break
                        if y < ry + rh - eps:
                            y = ry + rh
                    if y + h > V + eps:
                        continue
                    if mode == 0:
                        key = ((y + h) / V, x / U, max(mx, x + w) / U)
                    elif mode == 1:
                        key = (max(mx, x + w) * max(my, y + h) / (U * V), (y + h) / V, x / U)
                    else:
                        contact = (h if x < eps or abs(x + w - U) < eps else 0.0)
                        contact += w if y < eps or abs(y + h - V) < eps else 0.0
                        for rx, ry, rw, rh in local:
                            if abs(x + w - rx) < eps or abs(rx + rw - x) < eps:
                                contact += max(0.0, min(y + h, ry + rh) - max(y, ry))
                            if abs(y + h - ry) < eps or abs(ry + rh - y) < eps:
                                contact += max(0.0, min(x + w, rx + rw) - max(x, rx))
                        key = (-contact / (w + h), (y + h) / V, x / U)
                    p = (pid, y, x, ow, oh, rot) if axis else (pid, x, y, ow, oh, rot)
                    choices.append((key, p))
            choices.sort(key=lambda c: c[0])
            return [p for _, p in choices[:limit]]

        def grow(s, order, axis, mode, stop):
            rects = []
            for i in order:
                if time.perf_counter() >= stop:
                    break
                spots = locations(s, rects, i, axis, mode, 1, stop)
                if spots:
                    rects.append(spots[0])
            return add_pattern(s, rects)

        def greedy(pair, stop, noisy=False):
            candidates = [(s, rects) for s in pair for _, rects in pools[s]]
            if not candidates:
                return
            coverage = [0] * m
            entries = []
            cost = 0.0
            scarcity = [sum(1 for _, ps in candidates if any(p[0] == i for p in ps)) for i in range(m)]
            noise = [math.exp(rng.uniform(-0.5, 0.5)) if noisy else 1.0 for _ in range(m)]
            while any(coverage[i] < lo[i] for i in range(m)) and time.perf_counter() < stop:
                chosen, value = None, -1.0
                for s, rects in candidates:
                    need = sum(noise[p[0]] * pa[p[0]] * (1.0 + 1.0 / max(1, scarcity[p[0]]))
                               for p in rects if coverage[p[0]] < lo[p[0]])
                    if need <= 0:
                        continue
                    useful = sum(pa[p[0]] for p in rects if coverage[p[0]] < hi[p[0]])
                    score = (need * (1.0 + strength) + useful) / max(A[s], eps)
                    if score > value:
                        chosen, value = (s, rects), score
                if chosen is None:
                    return
                s, rects = chosen
                k = min(lo[p[0]] - coverage[p[0]] for p in rects if coverage[p[0]] < lo[p[0]])
                entries.append((s, rects, k))
                cost += A[s] * k
                for p in rects:
                    coverage[p[0]] += k
            if any(coverage[i] < lo[i] for i in range(m)):
                return
            publish(entries)
            value = sum(pa[i] * min(hi[i], coverage[i]) for i in range(m))
            while time.perf_counter() < stop:
                chosen = None
                ratio = value / cost if cost else -1.0
                for s, rects in candidates:
                    levels = {hi[p[0]] - coverage[p[0]] for p in rects if hi[p[0]] > coverage[p[0]]}
                    for k in levels:
                        gain = sum(pa[p[0]] * min(k, max(0, hi[p[0]] - coverage[p[0]])) for p in rects)
                        new_ratio = (value + gain) / (cost + k * A[s])
                        if new_ratio > ratio + 1e-12:
                            ratio, chosen = new_ratio, (s, rects, k, gain)
                if chosen is None:
                    break
                s, rects, k, gain = chosen
                entries.append((s, rects, k))
                cost += k * A[s]
                value += gain
                for p in rects:
                    coverage[p[0]] += k
                publish(entries)

        # The first patterns do not depend on the solver, so a usable combined solution exists early.
        seed_stop = min(deadline, start + budget * 0.17)
        for s in rank:
            if time.perf_counter() >= seed_stop:
                break
            order = sorted(fits[s], key=lambda i: (-pa[i], -max(pl[i], pw[i]), i))
            grow(s, order, 0, 0, seed_stop)
            missing = masks[s]
            for mask, _ in pools[s]:
                missing &= ~mask
            for i in fits[s]:
                if missing & (1 << i):
                    rot = int(not (pl[i] <= L[s] + eps and pw[i] <= W[s] + eps))
                    w, h = (pw[i], pl[i]) if rot else (pl[i], pw[i])
                    add_pattern(s, ((i, 0.0, 0.0, w, h, rot),))
        greedy(allowed, min(deadline, start + budget * 0.21))
        if time.perf_counter() >= deadline:
            return best
        try:
            import numpy as np
            from scipy.optimize import milp, linprog, Bounds, LinearConstraint
            from scipy.sparse import csc_matrix
            numerical = True
        except Exception:
            numerical = False

        def master(stop):
            nonlocal weights, numerical
            if not numerical or time.perf_counter() >= stop:
                return
            patterns = [(s, rects) for s in range(S) for _, rects in pools[s]]
            P = len(patterns)
            if P == 0:
                return
            try:
                n = P + m + S
                rows, cols, data = [], [], []
                c = np.zeros(n)
                lower = np.zeros(n)
                upper = np.ones(n)
                unit = max(max(A), max(pa), eps)
                density = max(best_ratio, 0.0)
                for j, (s, rects) in enumerate(patterns):
                    cap = max(hi[p[0]] for p in rects)
                    upper[j] = cap
                    c[j] = density * A[s] / unit
                    for p in rects:
                        rows.append(p[0]); cols.append(j); data.append(-1.0)
                    rows.extend((m + j, m + j, m + P + 1))
                    cols.extend((j, P + m + s, j))
                    data.extend((1.0, -float(cap), -1.0))
                for i in range(m):
                    lower[P + i], upper[P + i] = lo[i], hi[i]
                    c[P + i] = -pa[i] / unit
                    rows.append(i); cols.append(P + i); data.append(1.0)
                for s in range(S):
                    rows.append(m + P); cols.append(P + m + s); data.append(1.0)
                matrix = csc_matrix((data, (rows, cols)), shape=(m + P + 2, n))
                rhs = np.zeros(m + P + 2)
                rhs[m + P], rhs[m + P + 1] = 2.0, -1.0
                remain = stop - time.perf_counter()
                if remain <= 0:
                    return
                result = milp(c, integrality=np.ones(n), bounds=Bounds(lower, upper),
                              constraints=LinearConstraint(matrix, -np.inf, rhs),
                              options={"time_limit": max(0.001, remain * 0.78), "mip_rel_gap": 0.0})
                if result.x is not None:
                    quantities = [max(0, int(round(v))) for v in result.x[:P]]
                    if all(abs(result.x[j] - quantities[j]) < 1e-5 for j in range(P)):
                        publish([(patterns[j][0], patterns[j][1], k) for j, k in enumerate(quantities) if k])
                remain = stop - time.perf_counter()
                if remain > budget * 0.002:
                    relaxation = linprog(c, A_ub=matrix, b_ub=rhs,
                                         bounds=list(zip(lower, upper)), method="highs",
                                         options={"time_limit": max(0.001, remain * 0.85)})
                    if relaxation.success:
                        prices = [-float(v) * unit for v in relaxation.ineqlin.marginals[:m]]
                        weights = [max(pa[i] * 0.05, prices[i]) for i in range(m)]
            except Exception:
                numerical = False

        def beam(s, stop):
            # Beam width and number of candidate branches scale with total budget and instance size; tree depth is set by the number of piece types.
            width = max(2, int(strength * math.sqrt(max(1, len(fits[s])))))
            branches = max(2, int(strength * 2))
            axis, mode = rng.randrange(2), rng.randrange(3)
            order = sorted(fits[s], key=lambda i: -(weights[i] + pa[i] * 0.3) * math.exp(rng.uniform(-0.3, 0.3)))
            states = [(0.0, 0, ())]
            while states and time.perf_counter() < stop:
                next_states = []
                for profit, mask, rects in states:
                    offered = 0
                    for i in order:
                        if mask & (1 << i):
                            continue
                        spots = locations(s, rects, i, axis, mode, 2, stop)
                        if not spots:
                            if time.perf_counter() >= stop:
                                return
                            continue
                        offered += 1
                        for p in spots:
                            new_rects = rects + (p,)
                            new_profit = profit + weights[i] + 0.3 * pa[i]
                            next_states.append((new_profit, mask | (1 << i), new_rects))
                        if offered >= branches:
                            break
                    add_pattern(s, rects)
                    if time.perf_counter() >= stop:
                        return
                if not next_states:
                    return
                next_states.sort(key=lambda item: -item[0])
                states, seen = [], {}
                for state in next_states:
                    mask = state[1]
                    if seen.get(mask, 0) < 2:
                        states.append(state)
                        seen[mask] = seen.get(mask, 0) + 1
                        add_pattern(s, state[2])
                        if len(states) >= width:
                            break

        turn = 0
        next_master = time.perf_counter()
        period = max(0.015, budget / (4.0 + 2.0 * strength))
        # The only search stopping condition is the wall clock; the best published solution is kept and unfinished work never replaces it.
        while time.perf_counter() < deadline:
            now = time.perf_counter()
            if numerical and now >= next_master:
                master(min(deadline, now + period * 0.8))
                next_master = time.perf_counter() + period * 0.5
                if time.perf_counter() >= deadline:
                    break
            s = rank[turn % S] if turn % 3 != 2 else rng.choice(best_pair if rng.random() < 0.65 else rank)
            turn += 1
            stop = min(deadline, time.perf_counter() + period * 0.35)
            mode = turn % 3
            alpha = rng.uniform(0.0, 1.0)
            priorities = {i: (alpha * weights[i] + (1.0 - alpha) * pa[i]) *
                          math.exp(rng.uniform(-strength * 0.3, strength * 0.3)) /
                          max(pl[i] * pw[i], eps) ** rng.uniform(0.0, 0.45) for i in fits[s]}
            order = sorted(fits[s], key=lambda i: -priorities[i])
            grow(s, order, turn % 2, mode, stop)
            if turn % 3 == 0 and time.perf_counter() < deadline:
                beam(s, min(deadline, time.perf_counter() + period * 0.45))
            pair = best_pair
            if masks[s] & required == required:
                pair = (s,)
            elif s not in pair:
                compatible = [t for t in rank if t != s and (masks[s] | masks[t]) & required == required]
                if compatible:
                    pair = (s, rng.choice(compatible))
            greedy(pair, min(deadline, time.perf_counter() + period * 0.2), noisy=True)
        return best
    except Exception:
        return best
