"""Whole-sheet pattern reuse with fractional integer recombination
Generates and locally re-packs whole-sheet patterns with MaxRects, recombines sheets and quantities with a fractional integer objective, and keeps better feasible solutions within the wall-clock limit.
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
                "piece_counts": {f"piece_{i}": 0 for i in range(m)}}
        budget = max(0.0, float(time_limit_s))
        deadline = start + 0.94 * budget
        now = time.perf_counter
        sl = [float(v) for v in instance["length"]]
        sw = [float(v) for v in instance["width"]]
        sa = [float(v) for v in instance["area"]]
        pl = [float(v) for v in instance["piece_length"]]
        pw = [float(v) for v in instance["piece_width"]]
        pa = [float(v) for v in instance["piece_area"]]
        lo = [int(v) for v in instance["min"]]
        hi = [int(v) for v in instance["max"]]
        if not S or not m or not any(hi):
            return best
        eps = 1e-10
        effort = math.log1p(budget)
        free_limit = max(16, int((m + 1) * (2.0 + effort)))
        column_limit = max(m + S, int((m + S + 1) * (8 + 7 * effort)))
        strength = min(1.0, effort / (1.0 + math.log1p(m)))
        rng = random.Random(817193 + m * 1009 + S * 9176)
        orientations = []
        covers = []
        required = sum(1 << i for i in range(m) if lo[i] > 0)
        for s in range(S):
            row, mask = [], 0
            for i in range(m):
                opts = []
                if pl[i] <= sl[s] + eps and pw[i] <= sw[s] + eps:
                    opts.append((pl[i], pw[i], 0))
                if pl[i] != pw[i] and pw[i] <= sl[s] + eps and pl[i] <= sw[s] + eps:
                    opts.append((pw[i], pl[i], 1))
                row.append(opts)
                if opts:
                    mask |= 1 << i
            orientations.append(row)
            covers.append(mask)
        stock_order = sorted(range(S), key=lambda s: sa[s])
        allowed0 = None
        for s in stock_order:
            if covers[s] & required == required:
                allowed0 = (s,)
                break
        if allowed0 is None:
            for j, s in enumerate(stock_order):
                for t in stock_order[j + 1:]:
                    if (covers[s] | covers[t]) & required == required:
                        allowed0 = (s, t)
                        break
                if allowed0 is not None:
                    break
        if allowed0 is None:
            return best

        # A pattern holds one copy of each piece type; repetitions denote different stock-sheet instances.
        patterns, pattern_index = [], {}
        best_plan, best_density, best_area = {}, -math.inf, math.inf
        dual = pa[:]
        singleton_ids = set()

        def register(s, placements):
            ids = tuple(p[0] for p in placements)
            if not ids:
                return None
            mask = sum(1 << i for i in ids)
            key = (s, mask)
            if key in pattern_index:
                return pattern_index[key]
            j = len(patterns)
            patterns.append((s, tuple(placements), ids, sum(pa[i] for i in ids)))
            pattern_index[key] = j
            return j

        def singleton(s, i):
            p = register(s, [(i, 0.0, 0.0, orientations[s][i][0][2])])
            singleton_ids.add(p)
            return p

        def consider(plan):
            nonlocal best, best_plan, best_density, best_area
            counts, selected = [0] * m, set()
            total, used = 0.0, 0.0
            clean = {j: int(k) for j, k in plan.items() if k > 0}
            for j, k in clean.items():
                s, _, ids, filled = patterns[j]
                selected.add(s)
                total += sa[s] * k
                used += filled * k
                for i in ids:
                    counts[i] += k
            if (len(selected) > 2 or total <= 0 or
                    any(counts[i] < lo[i] or counts[i] > hi[i] for i in range(m))):
                return False
            density = used / total
            if (density < best_density - 1e-12 or
                    (density <= best_density + 1e-12 and total >= best_area)):
                return False
            flags = [0] * S
            bins = []
            for j, k in clean.items():
                s, places, _, _ = patterns[j]
                flags[s] = 1
                for _ in range(k):
                    bins.append({"stock_type_id": s, "placements": [
                        {"piece_type_id": i, "x": float(x), "y": float(y), "is_rotated": r}
                        for i, x, y, r in places]})
            candidate = {"selected_stock_types": flags, "stock_instances": bins,
                         "piece_counts": {f"piece_{i}": counts[i] for i in range(m)}}
            best = candidate
            best_plan, best_density, best_area = clean, density, total
            return True

        # Build a complete solver-free fallback plan first, then run the interruptible search.
        fallback = {}
        for i in range(m):
            if lo[i]:
                s = min((s for s in allowed0 if orientations[s][i]), key=lambda s: sa[s])
                fallback[singleton(s, i)] = lo[i]
        if not fallback:
            candidates = [(pa[i] / sa[s], s, i) for s in range(S) for i in range(m)
                          if hi[i] > 0 and orientations[s][i] and sa[s] > 0]
            if not candidates:
                return best
            _, s, i = max(candidates)
            fallback[singleton(s, i)] = 1
            allowed0 = (s,)
        consider(fallback)
        if now() >= deadline:
            return best

        def split_free(free, rect):
            x, y, w, h = rect
            right, top = x + w, y + h
            pieces = []
            for fx, fy, fw, fh in free:
                fr, ft = fx + fw, fy + fh
                if right <= fx + eps or x >= fr - eps or top <= fy + eps or y >= ft - eps:
                    pieces.append((fx, fy, fw, fh))
                    continue
                if x > fx + eps:
                    pieces.append((fx, fy, x - fx, fh))
                if right < fr - eps:
                    pieces.append((right, fy, fr - right, fh))
                if y > fy + eps:
                    pieces.append((fx, fy, fw, y - fy))
                if top < ft - eps:
                    pieces.append((fx, top, fw, ft - top))
            pieces.sort(key=lambda f: f[2] * f[3], reverse=True)
            pruned = []
            for f in pieces:
                if now() >= deadline:
                    break
                if f[2] <= eps or f[3] <= eps:
                    continue
                if any(f[0] >= g[0] - eps and f[1] >= g[1] - eps and
                       f[0] + f[2] <= g[0] + g[2] + eps and
                       f[1] + f[3] <= g[1] + g[3] + eps for g in pruned):
                    continue
                pruned.append(f)
                if len(pruned) >= free_limit:
                    break
            return pruned

        def pack(s, first, second, weights, mode, fixed=()):
            free = [(0.0, 0.0, sl[s], sw[s])]
            places = list(fixed)
            seen = {p[0] for p in places}
            for i, x, y, r in places:
                w, h = (pw[i], pl[i]) if r else (pl[i], pw[i])
                free = split_free(free, (x, y, w, h))
            dim = max(sl[s], sw[s], eps)
            scale = max((abs(weights[i]) for i in first + second), default=1.0)
            scale = max(scale, eps)

            def order(ids):
                ranked = []
                for i in ids:
                    if i in seen or not orientations[s][i]:
                        continue
                    value = weights[i] / scale
                    longest = max(pl[i], pw[i]) / dim
                    shortest = min(pl[i], pw[i]) / dim
                    if mode == 0:
                        score = value + 1.8 * longest
                    elif mode == 1:
                        score = 2.0 * value + longest
                    elif mode == 2:
                        score = value + 1.5 * shortest
                    else:
                        score = (max(eps, value + 0.3 * longest) ** (0.5 + 0.6 * rng.random())
                                 * (0.35 + 1.3 * rng.random()))
                    ranked.append((score, i))
                ranked.sort(reverse=True)
                return [i for _, i in ranked]

            for group in (first, second):
                for i in order(group):
                    if now() >= deadline:
                        return register(s, places)
                    chosen = None
                    for fx, fy, fw, fh in free:
                        for w, h, r in orientations[s][i]:
                            if w > fw + eps or h > fh + eps:
                                continue
                            dx, dy = max(0.0, fw - w), max(0.0, fh - h)
                            x, y = fx, fy
                            if mode == 3:
                                x += dx
                            elif mode == 4:
                                y += dy
                            short, long = min(dx, dy), max(dx, dy)
                            if mode in (0, 3):
                                score = (short, long, fy + h, fx)
                            elif mode == 1:
                                score = (fw * fh - w * h, short, fy + h, fx)
                            elif mode == 2:
                                score = (fy + h, fx, short, long)
                            else:
                                score = (long, short, -y, -x)
                            if chosen is None or score < chosen[0]:
                                chosen = (score, x, y, w, h, r)
                    if chosen is None:
                        continue
                    _, x, y, w, h, r = chosen
                    places.append((i, max(0.0, x), max(0.0, y), r))
                    seen.add(i)
                    free = split_free(free, (x, y, w, h))
            return register(s, places)

        def construct(allowed, target, mode):
            counts, plan = [0] * m, {}
            while now() < deadline:
                need = [i for i in range(m) if counts[i] < target[i]]
                if not need:
                    break
                extra = [i for i in range(m) if counts[i] >= target[i] and counts[i] < hi[i]]
                peak = max(target[i] - counts[i] for i in need)
                weights = [pa[i] * (1.0 + (target[i] - counts[i]) / peak) for i in range(m)]
                options = []
                for s in allowed:
                    p = pack(s, need, extra, weights, mode)
                    if p is None:
                        continue
                    ids = patterns[p][2]
                    missing = [target[i] - counts[i] for i in ids if counts[i] < target[i]]
                    if not missing:
                        continue
                    repeat = min(min(missing), min(hi[i] - counts[i] for i in ids))
                    if repeat <= 0:
                        continue
                    urgent = sum(pa[i] * (target[i] - counts[i]) / peak for i in ids
                                 if counts[i] < target[i])
                    score = (patterns[p][3] + 0.35 * urgent) / sa[s]
                    options.append((score, -sa[s], p, repeat))
                if not options:
                    break
                _, _, p, repeat = max(options)
                plan[p] = plan.get(p, 0) + repeat
                for i in patterns[p][2]:
                    counts[i] += repeat
            # When construction is interrupted, only hard demands are filled; unfinished candidates are never mistaken for feasible solutions.
            for i in range(m):
                if counts[i] < lo[i]:
                    s = min((s for s in allowed if orientations[s][i]), key=lambda s: sa[s])
                    p = singleton(s, i)
                    plan[p] = plan.get(p, 0) + lo[i] - counts[i]
            consider(plan)

        construct(allowed0, lo, 0)
        if now() >= deadline:
            return best

        # If the general numerical library is unavailable or too little time remains, the standalone geometric search is still used.
        optimizer = None
        if deadline - now() > 0.4:
            try:
                import numpy as np
                from scipy.optimize import Bounds, LinearConstraint, linprog, milp
                from scipy.sparse import coo_matrix, vstack
                optimizer = milp
            except Exception:
                optimizer = None

        def optimize_pool():
            nonlocal dual
            remaining = deadline - now()
            if optimizer is None or remaining <= 0.02:
                return
            count = len(patterns)
            if count <= column_limit:
                chosen = list(range(count))
            else:
                essential = set(best_plan) | singleton_ids
                room = max(0, column_limit - len(essential))
                recent = list(range(max(0, count - room // 3), count))
                essential.update(recent)
                ranking = sorted((j for j in range(count) if j not in essential),
                                 key=lambda j: patterns[j][3] / sa[patterns[j][0]]
                                 + 0.4 * rng.random(), reverse=True)
                essential.update(ranking[:max(0, column_limit - len(essential))])
                chosen = sorted(essential)
            P = len(chosen)
            stocks = sorted({patterns[j][0] for j in chosen})
            stock_var = {s: P + k for k, s in enumerate(stocks)}
            N = P + len(stocks)
            area_scale = max(max(sa[s] for s in stocks), eps)
            costs = np.zeros(N)
            upper = np.ones(N)
            rows, cols, vals = [], [], []
            for k, j in enumerate(chosen):
                s, _, ids, filled = patterns[j]
                bound = min(hi[i] for i in ids)
                upper[k] = bound
                costs[k] = (best_density * sa[s] - filled) / area_scale
                for i in ids:
                    rows.append(i)
                    cols.append(k)
                    vals.append(1.0)
                rows.extend((m + k, m + k, m + P + 1))
                cols.extend((k, stock_var[s], k))
                vals.extend((1.0, -float(bound), 1.0))
            for s in stocks:
                rows.append(m + P)
                cols.append(stock_var[s])
                vals.append(1.0)
            matrix = coo_matrix((vals, (rows, cols)), shape=(m + P + 2, N)).tocsc()
            lower_row = np.concatenate((np.asarray(lo, dtype=float),
                                        np.full(P, -np.inf), [-np.inf, 1.0]))
            upper_row = np.concatenate((np.asarray(hi, dtype=float), np.zeros(P), [2.0, np.inf]))
            seconds = min((deadline - now()) * 0.45,
                          budget / (5.0 + math.sqrt(m + S)))
            if seconds <= 0.005:
                return
            try:
                result = optimizer(costs, integrality=np.ones(N),
                                   bounds=Bounds(np.zeros(N), upper),
                                   constraints=LinearConstraint(matrix, lower_row, upper_row),
                                   options={"time_limit": seconds, "mip_rel_gap": 0.0})
                if result.x is not None:
                    candidate = {}
                    integral = True
                    for k, j in enumerate(chosen):
                        value = float(result.x[k])
                        copies = int(round(value))
                        if abs(value - copies) > 1e-5 or copies < 0:
                            integral = False
                            break
                        if copies:
                            candidate[j] = copies
                    if integral:
                        consider(candidate)
            except Exception:
                return
            # The linear relaxation of the current sheet combination gives piece scarcity prices for the next round of pattern generation.
            remaining = deadline - now()
            if remaining <= 0.03:
                return
            active = {patterns[j][0] for j in best_plan}
            keep = [k for k, j in enumerate(chosen) if patterns[j][0] in active]
            if not keep:
                return
            B = matrix[:m, keep]
            c = np.array([(best_density * sa[patterns[chosen[k]][0]] - patterns[chosen[k]][3])
                          / area_scale for k in keep])
            try:
                relaxation = linprog(c, A_ub=vstack((-B, B), format="csc"),
                                     b_ub=np.array([-v for v in lo] + hi, dtype=float),
                                     bounds=(0.0, None), method="highs",
                                     options={"time_limit": min(remaining * 0.15,
                                                               budget / (20 + m + S))})
                if relaxation.success:
                    prices = relaxation.ineqlin.marginals
                    dual = [pa[i] + area_scale * (-prices[i] + prices[m + i]) for i in range(m)]
            except Exception:
                pass

        all_ids = [[i for i in range(m) if hi[i] > 0 and orientations[s][i]] for s in range(S)]
        viable_stocks = [s for s in range(S) if all_ids[s]]
        window = max(0.002, budget / (24 + 8 * effort))
        next_master = now() + window
        next_construct = now() + 2 * window
        last_pool = -1
        turn = 0
        stock_cursor = 0
        while now() < deadline:
            current = now()
            if optimizer is not None and current >= next_master and len(patterns) != last_pool:
                last_pool = len(patterns)
                optimize_pool()
                next_master = now() + window
                continue
            if current >= next_construct:
                active = tuple(sorted({patterns[j][0] for j in best_plan}))
                if rng.random() < 0.5:
                    allowed = active
                else:
                    s = rng.choice(viable_stocks)
                    t = rng.choice(stock_order)
                    allowed = tuple(sorted({s, t}))
                    union = covers[s] | covers[t]
                    if union & required != required:
                        allowed = allowed0
                roll = rng.random()
                if roll < 0.4:
                    target = lo[:]
                elif roll < 0.65:
                    target = [hi[i] if any(orientations[s][i] for s in allowed) else lo[i]
                              for i in range(m)]
                else:
                    rate = rng.random()
                    target = [lo[i] + int(rate * (hi[i] - lo[i]))
                              if any(orientations[s][i] for s in allowed) else lo[i] for i in range(m)]
                construct(allowed, target, rng.randrange(5))
                next_construct = now() + window * (3.0 if optimizer is not None else 0.5)
                continue

            # Keep extending patterns by wall clock: rotate through stock sheets for exploration, or destroy and re-pack existing patterns.
            mode = turn % 5
            turn += 1
            fixed = ()
            if patterns and rng.random() < 0.55:
                source = rng.choice(list(best_plan)) if best_plan and rng.random() < 0.6 else rng.randrange(len(patterns))
                s, places, ids, _ = patterns[source]
                fraction = (0.15 + 0.65 * strength) * rng.random()
                keep_count = max(0, len(places) - max(1, int(math.ceil(len(places) * fraction))))
                fixed = tuple(rng.sample(places, keep_count))
                register(s, fixed)
            else:
                s = viable_stocks[stock_cursor % len(viable_stocks)]
                stock_cursor += 1
            available = all_ids[s]
            if turn % 3 == 0:
                weights = dual[:]
                ids = [i for i in available if weights[i] > eps or lo[i] > 0]
            elif turn % 3 == 1:
                weights = [pa[i] * (1.0 + lo[i] / max(1, hi[i])) for i in range(m)]
                ids = available[:]
            else:
                weights = [pa[i] * (0.2 + 1.8 * rng.random()) for i in range(m)]
                keep_probability = 0.45 + 0.55 * rng.random()
                ids = [i for i in available if rng.random() < keep_probability]
            fixed_ids = {p[0] for p in fixed}
            ids = [i for i in ids if i not in fixed_ids]
            p = pack(s, ids, [], weights, mode, fixed)
            if p is not None:
                # When a single high-density pattern already covers the hard demands, there is no need to wait for integer programming.
                ids = patterns[p][2]
                present = set(ids)
                if all(lo[i] == 0 or i in present for i in range(m)):
                    repeat = max(1, max(lo))
                    if repeat <= min(hi[i] for i in ids):
                        consider({p: repeat})
        return best
    except Exception:
        return best
