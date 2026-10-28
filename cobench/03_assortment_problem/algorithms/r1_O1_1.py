"""Pattern-pool fractional integer optimisation
Combines maximal-free-rectangle packing, demand-price guidance, destroy-and-repair and integer recombination of patterns to keep lowering the total waste-area ratio.
"""

import math
import random
import time

try:
    import numpy as np
    from scipy.optimize import Bounds, LinearConstraint, linprog, milp
    from scipy.sparse import csc_matrix, vstack
except Exception:
    np = None


def solve(instance: dict, time_limit_s: float) -> dict:
    clock = time.perf_counter
    start = clock()
    result = {"selected_stock_types": [], "stock_instances": [], "piece_counts": {}}
    pool = {}
    incumbent = {}
    try:
        budget = max(0.0, float(time_limit_s))
        S, m = int(instance["S"]), int(instance["m"])
        result["selected_stock_types"] = [0] * S
        result["piece_counts"] = {f"piece_{i}": 0 for i in range(m)}
        sl = [float(v) for v in instance["length"]]
        sw = [float(v) for v in instance["width"]]
        sa = [float(v) for v in instance["area"]]
        pl = [float(v) for v in instance["piece_length"]]
        pw = [float(v) for v in instance["piece_width"]]
        pa = [float(v) for v in instance["piece_area"]]
        lo = [int(v) for v in instance["min"]]
        hi = [int(v) for v in instance["max"]]
        eps = 1e-9
        deadline = start + budget - min(0.15, budget * 0.035)
        strength = math.sqrt(1.0 + budget)
        perturbation = math.log1p(budget) / (2.0 + math.log1p(budget))
        column_limit = max(1, int((m + S) * (5.0 + 7.0 * strength)))
        scale = max(sa, default=1.0)
        scale = max(scale, eps)
        seed = 20260930
        for v in sl + sw + pl + pw:
            seed = ((seed * 1000003) ^ int(abs(v) * 1000.0)) & 0xFFFFFFFFFFFFFFFF
        rng = random.Random(seed)
        orientations = []
        masks = []
        required = sum(1 << i for i in range(m) if lo[i] > 0)
        for s in range(S):
            choices = {}
            mask = 0
            for i in range(m):
                if hi[i] <= 0:
                    continue
                turns = []
                if pl[i] <= sl[s] + eps and pw[i] <= sw[s] + eps:
                    turns.append((pl[i], pw[i], 0))
                if (abs(pl[i] - pw[i]) > eps and
                        pw[i] <= sl[s] + eps and pl[i] <= sw[s] + eps):
                    turns.append((pw[i], pl[i], 1))
                if turns:
                    choices[i] = turns
                    mask |= 1 << i
            orientations.append(choices)
            masks.append(mask)

        stocks = sorted((s for s in range(S) if masks[s]), key=lambda s: sa[s])
        allowed = next(((s,) for s in stocks if masks[s] & required == required), ())
        if not allowed:
            for p, a in enumerate(stocks):
                for b in stocks[p + 1:]:
                    if (masks[a] | masks[b]) & required == required:
                        allowed = (a, b)
                        break
                if allowed:
                    break
        if not allowed:
            return result

        # A pattern stores only a geometrically feasible set of pieces; repetition counts are decided in the master problem.
        def register(s, places):
            if not places:
                return None
            ids = tuple(sorted(p[0] for p in places))
            mask = sum(1 << i for i in ids)
            key = (s, mask)
            if key not in pool:
                pool[key] = (ids, tuple(places), sum(pa[i] for i in ids))
            return key

        for i in range(m):
            if lo[i] <= 0:
                continue
            s = min((s for s in allowed if i in orientations[s]), key=lambda s: sa[s])
            rot = orientations[s][i][0][2]
            key = register(s, [(i, 0.0, 0.0, rot)])
            incumbent[key] = lo[i]
        if not incumbent:
            candidates = [(pa[i] / max(sa[s], eps), s, i)
                          for s in stocks for i in orientations[s]]
            if not candidates:
                return result
            _, s, i = max(candidates)
            key = register(s, [(i, 0.0, 0.0, orientations[s][i][0][2])])
            incumbent[key] = 1

        best_counts = lo[:]
        best_ratio = -math.inf
        best_area = math.inf

        def consider(plan):
            nonlocal incumbent, best_ratio, best_area, best_counts, deadline
            counts = [0] * m
            used = set()
            area = 0.0
            clean = {}
            for key, amount in plan.items():
                if amount <= 0:
                    continue
                if amount != int(amount):
                    return False
                amount = int(amount)
                used.add(key[0])
                if len(used) > 2:
                    return False
                clean[key] = amount
                area += amount * sa[key[0]]
                for i in pool[key][0]:
                    counts[i] += amount
            if area <= 0 or any(not lo[i] <= counts[i] <= hi[i] for i in range(m)):
                return False
            ratio = sum(counts[i] * pa[i] for i in range(m)) / area
            if ratio > best_ratio + 1e-12 or (abs(ratio - best_ratio) <= 1e-12 and area < best_area):
                incumbent, best_counts = clean, counts
                best_ratio, best_area = ratio, area
                output_size = sum(amount * (len(pool[key][0]) + 1) for key, amount in clean.items())
                reserve = min(budget * 0.2, max(budget * 0.035, output_size * 0.000002))
                deadline = min(deadline, start + budget - reserve)
                return True
            return False

        consider(incumbent)

        def split(free, x, y, w, h, stop):
            right, top = x + w, y + h
            rects = []
            for fx, fy, fw, fh in free:
                fr, ft = fx + fw, fy + fh
                if x >= fr - eps or right <= fx + eps or y >= ft - eps or top <= fy + eps:
                    rects.append((fx, fy, fw, fh))
                    continue
                if x > fx + eps:
                    rects.append((fx, fy, x - fx, fh))
                if right < fr - eps:
                    rects.append((right, fy, fr - right, fh))
                if y > fy + eps:
                    rects.append((fx, fy, fw, y - fy))
                if top < ft - eps:
                    rects.append((fx, top, fw, ft - top))
            # Removing contained rectangles does not affect feasibility; on timeout the unprocessed free rectangles are kept.
            rects = list(dict.fromkeys(rects))
            rects.sort(key=lambda r: r[2] * r[3], reverse=True)
            keep = []
            for j, r in enumerate(rects):
                if clock() >= stop:
                    return keep + rects[j:]
                if not any(r[0] >= a[0] - eps and r[1] >= a[1] - eps and
                           r[0] + r[2] <= a[0] + a[2] + eps and
                           r[1] + r[3] <= a[1] + a[3] + eps for a in keep):
                    keep.append(r)
            return keep

        def pack(s, order, mode, stop, initial=()):
            free = [(0.0, 0.0, sl[s], sw[s])]
            places = list(initial)
            seen = set()
            for i, x, y, rot in initial:
                seen.add(i)
                w, h = (pw[i], pl[i]) if rot else (pl[i], pw[i])
                free = split(free, x, y, w, h, stop)
            for i in order:
                if clock() >= stop:
                    break
                if i in seen or i not in orientations[s]:
                    continue
                chosen = None
                noise = rng.random() * perturbation if mode == 3 else 0.0
                for fx, fy, fw, fh in free:
                    if clock() >= stop:
                        break
                    for w, h, rot in orientations[s][i]:
                        if w > fw + eps or h > fh + eps:
                            continue
                        dw, dh = max(0.0, fw - w), max(0.0, fh - h)
                        if mode == 0:
                            rank = (min(dw, dh), max(dw, dh), fy + h, fx)
                        elif mode == 1:
                            rank = (fw * fh - w * h, min(dw, dh), fy + h, fx)
                        elif mode == 2:
                            rank = (fy + h, fx, min(dw, dh), max(dw, dh))
                        else:
                            rank = ((1.0 - noise) * min(dw, dh) / max(sl[s], sw[s], eps)
                                    + noise * (fw * fh - w * h) / max(sl[s] * sw[s], eps),
                                    fy + h, fx, max(dw, dh))
                        if chosen is None or rank < chosen[0]:
                            chosen = (rank, fx, fy, w, h, rot)
                if chosen is not None:
                    _, x, y, w, h, rot = chosen
                    places.append((i, float(x), float(y), rot))
                    seen.add(i)
                    free = split(free, x, y, w, h, stop)
            return places

        def counts_of(plan):
            counts = [0] * m
            for key, amount in plan.items():
                for i in pool[key][0]:
                    counts[i] += amount
            return counts

        def enrich(plan, counts, stop, randomized):
            # Split patterns by repetition count so that topping up never exceeds any piece type's upper bound.
            for key, total in list(plan.items()):
                if clock() >= stop:
                    break
                left = total
                while left > 0 and clock() < stop:
                    s = key[0]
                    ids, original, _ = pool[key]
                    present = set(ids)
                    order = [i for i in orientations[s] if i not in present and counts[i] < hi[i]]
                    if not order:
                        break
                    order.sort(key=lambda i: (counts[i] < lo[i],
                               pa[i] * (rng.uniform(0.7, 1.3) if randomized else 1.0)), reverse=True)
                    places = pack(s, order, rng.randrange(4) if randomized else 0, stop, original)
                    additions = [p[0] for p in places if p[0] not in present]
                    if not additions:
                        break
                    amount = min([left] + [hi[i] - counts[i] for i in additions])
                    new = register(s, places)
                    plan[key] -= amount
                    if plan[key] == 0:
                        del plan[key]
                    plan[new] = plan.get(new, 0) + amount
                    for i in additions:
                        counts[i] += amount
                    left -= amount
            return plan, counts

        def augment(plan, counts, pair, stop):
            area = sum(sa[k[0]] * v for k, v in plan.items())
            packed = sum(pa[i] * counts[i] for i in range(m))
            keys = [k for k in pool if k[0] in pair]
            keys.sort(key=lambda k: pool[k][2] / max(sa[k[0]], eps), reverse=True)
            for key in keys:
                if clock() >= stop:
                    break
                if pool[key][2] * area <= packed * sa[key[0]] + eps:
                    break
                amount = min(hi[i] - counts[i] for i in pool[key][0])
                if amount <= 0:
                    continue
                plan[key] = plan.get(key, 0) + amount
                for i in pool[key][0]:
                    counts[i] += amount
                area += sa[key[0]] * amount
                packed += pool[key][2] * amount
            return plan

        max_pa = max(pa, default=1.0)
        max_dim = max(pl + pw, default=1.0)
        norm_pa = [v / max(max_pa, eps) for v in pa]
        norm_dim = [max(pl[i], pw[i]) / max(max_dim, eps) for i in range(m)]
        prices = {s: pa[:] for s in stocks}

        def construct(pair, target, stop, randomized=False, kept=None, optional=True):
            plan = dict(kept or {})
            counts = counts_of(plan)
            if plan:
                plan, counts = enrich(plan, counts, stop, randomized)
            while clock() < stop:
                rem = [max(0, target[i] - counts[i]) for i in range(m)]
                if not any(rem):
                    if not plan:
                        return
                    consider(plan)
                    plan, counts = enrich(plan, counts, stop, randomized)
                    consider(plan)
                    consider(augment(plan, counts, pair, stop))
                    return
                largest = max(rem)
                candidates = []
                mode = rng.randrange(4) if randomized else 0
                for s in pair:
                    order = [i for i in orientations[s] if rem[i] > 0]
                    order.sort(key=lambda i: 1.8 * norm_dim[i] + norm_pa[i]
                               + rem[i] / largest
                               + (rng.random() * (1.0 + perturbation) if randomized else 0.0),
                               reverse=True)
                    if optional:
                        extra = [i for i in orientations[s] if rem[i] == 0 and counts[i] < hi[i]]
                        extra.sort(key=lambda i: pa[i] * (rng.uniform(0.5, 1.5) if randomized else 1.0), reverse=True)
                        order += extra
                    places = pack(s, order, mode, stop)
                    need = [p[0] for p in places if rem[p[0]] > 0]
                    if not need:
                        continue
                    key = register(s, places)
                    urgency = sum(pa[i] * rem[i] / largest for i in need)
                    score = (pool[key][2] + urgency) / max(sa[s], eps)
                    candidates.append((score, -sa[s], key, need))
                if not candidates:
                    return
                _, _, key, need = max(candidates)
                amount = min([rem[i] for i in need] + [hi[i] - counts[i] for i in pool[key][0]])
                if amount <= 0:
                    return
                plan[key] = plan.get(key, 0) + amount
                for i in pool[key][0]:
                    counts[i] += amount
                # A feasible solution is stored immediately, regardless of whether the last top-up step finishes.
                if all(counts[i] >= lo[i] for i in range(m)):
                    consider(plan)

        def pick_pair(s=None):
            if s is None and rng.random() < 0.5:
                return tuple(sorted({k[0] for k in incumbent}))
            if s is None:
                s = rng.choice(stocks)
            partners = [b for b in stocks if (masks[s] | masks[b]) & required == required]
            if not partners:
                return allowed
            if masks[s] & required == required and rng.random() < 0.5:
                return (s,)
            b = rng.choice(partners)
            return tuple(sorted({s, b}))

        initial_stop = min(deadline, clock() + budget * 0.09)
        construct(allowed, lo, initial_stop, optional=False)
        # Single-piece columns keep the master problem demand-feasible for every selectable stock sheet.
        for s in stocks:
            for i in orientations[s]:
                register(s, [(i, 0.0, 0.0, orientations[s][i][0][2])])

        def generate(s, tick, stop):
            available = list(orientations[s])
            if not available:
                return
            style = tick % 8
            forced = None
            if style == 0:
                order = sorted(available, key=lambda i: (pa[i], norm_dim[i]), reverse=True)
            elif style == 1:
                order = sorted(available, key=lambda i: (norm_dim[i], pa[i]), reverse=True)
            elif style == 2:
                order = sorted(available, key=lambda i: pl[i] + pw[i], reverse=True)
            elif style == 3:
                values = prices[s]
                order = [i for i in available if values[i] > eps]
                order.sort(key=lambda i: values[i] * rng.uniform(1.0 - perturbation, 1.0 + perturbation), reverse=True)
            elif style == 4:
                order = [i for i in available if lo[i] > 0]
                power = rng.uniform(0.2, 1.5)
                order.sort(key=lambda i: (norm_pa[i] + norm_dim[i]) * lo[i] ** power * rng.uniform(0.5, 1.5), reverse=True)
            elif style == 5:
                order = [i for i in available if best_counts[i] < hi[i] or rng.random() < 0.3]
                order.sort(key=lambda i: (norm_pa[i] + norm_dim[i]) * rng.random(), reverse=True)
            elif style == 6 and incumbent:
                # Swap pieces out of existing patterns and re-pack them on other stock sheets.
                first = rng.choice(list(incumbent))
                second = rng.choice(list(incumbent))
                focus = set(pool[first][0]) | set(pool[second][0])
                order = [i for i in available if i in focus and rng.random() > 0.35 * perturbation]
                order.sort(key=lambda i: (norm_pa[i] + norm_dim[i]) * rng.uniform(0.4, 1.6), reverse=True)
                order += [i for i in available if i not in focus and rng.random() < 0.5]
            else:
                fraction = rng.uniform(0.25, 1.0)
                order = [i for i in available if rng.random() <= fraction]
                rng.shuffle(order)
                forced = rng.choice(available)
            if forced is not None:
                order = [forced] + [i for i in order if i != forced]
            places = pack(s, order, tick % 4, stop)
            register(s, places)
            if len(places) > 1:
                # Non-maximal patterns avoid binding low-bound pieces to every sheet.
                cut = rng.randrange(1, len(places))
                register(s, places[:cut])
                removed = rng.randrange(len(places))
                register(s, places[:removed] + places[removed + 1:])

        def active_keys(pair):
            keys = [k for k in pool if pair is None or k[0] in pair]
            mandatory = {k for k in keys if len(pool[k][0]) == 1 or k in incumbent}
            other = [k for k in keys if k not in mandatory]
            room = max(column_limit, len(mandatory)) - len(mandatory)
            if len(other) > room:
                rng.shuffle(other)
                other.sort(key=lambda k: (sum(max(0.0, prices[k[0]][i]) for i in pool[k][0])
                                          + pool[k][2]) / max(sa[k[0]], eps), reverse=True)
                cut = int(room * 0.7)
                other = other[:cut] + rng.sample(other[cut:], room - cut)
            return sorted(mandatory) + other

        def master(pair, seconds):
            if np is None or seconds <= 0:
                return
            stop = min(deadline, clock() + seconds)
            keys = active_keys(pair)
            p = len(keys)
            if p == 0:
                return
            used_stocks = sorted({k[0] for k in keys})
            selection = pair is None and len(used_stocks) > 2
            stock_col = {s: p + j for j, s in enumerate(used_stocks)} if selection else {}
            rows, cols, data = [], [], []
            lower, upper = list(lo) + [1.0], list(hi) + [math.inf]
            links = {}
            if selection:
                # Per-type linking is tighter than a big M based on total demand.
                for s in used_stocks:
                    for i in orientations[s]:
                        row = len(lower)
                        links[s, i] = row
                        lower.append(-math.inf)
                        upper.append(0.0)
                        rows.append(row)
                        cols.append(stock_col[s])
                        data.append(-float(hi[i]))
                row = len(lower)
                lower.append(0.0)
                upper.append(2.0)
                for s in used_stocks:
                    rows.append(row)
                    cols.append(stock_col[s])
                    data.append(1.0)
            objective, bounds = [], []
            for j, key in enumerate(keys):
                if clock() >= stop:
                    return
                ids, _, area = pool[key]
                objective.append((best_ratio * sa[key[0]] - area) / scale)
                bounds.append(float(min(hi[i] for i in ids)))
                rows.append(m)
                cols.append(j)
                data.append(1.0)
                for i in ids:
                    rows.append(i)
                    cols.append(j)
                    data.append(1.0)
                    if selection:
                        rows.append(links[key[0], i])
                        cols.append(j)
                        data.append(1.0)
            n = p + (len(used_stocks) if selection else 0)
            objective += [0.0] * (n - p)
            bounds += [1.0] * (n - p)
            matrix = csc_matrix((data, (rows, cols)), shape=(len(lower), n))
            remaining = stop - clock()
            if remaining <= 0:
                return
            try:
                answer = milp(np.asarray(objective), integrality=np.ones(n),
                              bounds=Bounds(np.zeros(n), np.asarray(bounds)),
                              constraints=LinearConstraint(matrix, np.asarray(lower), np.asarray(upper)),
                              options={"time_limit": remaining, "mip_rel_gap": 0.0, "presolve": True})
                if answer.x is not None:
                    plan = {}
                    for j, key in enumerate(keys):
                        v = float(answer.x[j])
                        amount = int(round(v))
                        if abs(v - amount) > 1e-5 or amount < 0:
                            return
                        if amount:
                            plan[key] = amount
                    consider(plan)
            except Exception:
                return

        def price_pair(pair, seconds):
            if np is None or seconds <= 0 or not pair:
                return
            stop = min(deadline, clock() + seconds)
            keys = active_keys(pair)
            rows, cols = [], []
            for j, key in enumerate(keys):
                for i in pool[key][0]:
                    rows.append(i)
                    cols.append(j)
            if not keys or clock() >= stop:
                return
            A = csc_matrix((np.ones(len(rows)), (rows, cols)), shape=(m, len(keys)))
            constraint = vstack([A, -A, -csc_matrix(np.ones((1, len(keys))))], format="csc")
            objective = np.asarray([(best_ratio * sa[k[0]] - pool[k][2]) / scale for k in keys])
            try:
                answer = linprog(objective, A_ub=constraint,
                                 b_ub=np.asarray(hi + [-v for v in lo] + [-1]),
                                 bounds=(0, None), method="highs-ds",
                                 options={"time_limit": max(0.0001, stop - clock())})
                if answer.success:
                    dual = answer.ineqlin.marginals
                    weights = [pa[i] + scale * (dual[i] - dual[m + i]) for i in range(m)]
                    for s in pair:
                        prices[s] = weights[:]
            except Exception:
                return

        tick = 0
        rounds = 0
        next_s = 0
        # The main loop has no iteration cap; pool width, perturbation strength and subproblem budgets all scale with the time limit.
        while clock() < deadline:
            remaining = deadline - clock()
            generation_stop = min(deadline, clock() + min(remaining * 0.35, budget * 0.055))
            while clock() < generation_stop:
                s = stocks[next_s % len(stocks)]
                next_s += 1
                generate(s, tick // len(stocks), generation_stop)
                tick += 1
            if clock() >= deadline:
                break
            pair = pick_pair()
            build_stop = min(deadline, clock() + budget * 0.045)
            if rounds % 3 == 1 and incumbent:
                kept = {}
                destroy = rng.uniform(0.2, 0.35 + 0.4 * perturbation)
                for key, amount in incumbent.items():
                    keep = int(amount * (1.0 - destroy)) if rng.random() < destroy else amount
                    if keep:
                        kept[key] = keep
                pair = tuple(sorted({k[0] for k in incumbent}))
                construct(pair, lo, build_stop, True, kept)
            else:
                if rounds % 3 == 0:
                    target = lo[:]
                else:
                    fraction = rng.random()
                    target = [lo[i] + int((hi[i] - lo[i]) * fraction) if any(i in orientations[s] for s in pair)
                              else lo[i] for i in range(m)]
                construct(pair, target, build_stop, True)
            if clock() >= deadline:
                break
            solve_pair = None if rounds % 2 == 0 else tuple(sorted({k[0] for k in incumbent}))
            allocation = min((deadline - clock()) * 0.65,
                             budget * (0.025 + 0.022 * math.log1p(budget)))
            master(solve_pair, allocation)
            if clock() < deadline:
                price_pair(tuple(sorted({k[0] for k in incumbent})),
                           min((deadline - clock()) * 0.2, budget * 0.018))
            rounds += 1
    except Exception:
        pass

    # Only integer pattern combinations whose demand and stock choice have been checked become the incumbent.
    try:
        selected = [0] * S
        counts = [0] * m
        bins = []
        for key, amount in incumbent.items():
            s = key[0]
            selected[s] = 1
            ids, places, _ = pool[key]
            for i in ids:
                counts[i] += amount
            for _ in range(amount):
                bins.append({"stock_type_id": int(s), "placements": [
                    {"piece_type_id": int(i), "x": float(x), "y": float(y), "is_rotated": int(rot)}
                    for i, x, y, rot in places]})
        if bins:
            result = {"selected_stock_types": selected, "stock_instances": bins,
                      "piece_counts": {f"piece_{i}": int(counts[i]) for i in range(m)}}
    except Exception:
        pass
    return result
