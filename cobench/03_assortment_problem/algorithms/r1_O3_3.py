"""Pattern recombination with adaptive destroy-and-repair rectangle packing
Combines whole-sheet utilisation trial packing, cross-sheet reuse and multi-corner maximal-empty-rectangle placement, and improves the waste ratio by integer pattern recombination and demand perturbation.
"""

import math
import random
import time


def solve(instance: dict, time_limit_s: float) -> dict:
    started = time.perf_counter()
    budget = max(0.0, float(time_limit_s))
    deadline = started + budget * 0.96
    m = int(instance.get("m", 0))
    S = int(instance.get("S", 0))
    best = {"selected_stock_types": [0] * S, "stock_instances": [],
            "piece_counts": {f"piece_{i}": 0 for i in range(m)}}

    try:
        sl = [float(v) for v in instance["length"]]
        sw = [float(v) for v in instance["width"]]
        sa = [float(v) for v in instance["area"]]
        pl = [float(v) for v in instance["piece_length"]]
        pw = [float(v) for v in instance["piece_width"]]
        pa = [float(v) for v in instance["piece_area"]]
        lo = [int(v) for v in instance["min"]]
        hi = [int(v) for v in instance["max"]]
        eps = 1e-9
        if not S or not m or not any(hi):
            return best

        bits = [1 << i for i in range(m)]
        required = sum(bits[i] for i in range(m) if lo[i] > 0)
        orient = []
        masks = []
        for s in range(S):
            row = []
            mask = 0
            for i in range(m):
                opts = []
                if hi[i] > 0:
                    if pl[i] <= sl[s] + eps and pw[i] <= sw[s] + eps:
                        opts.append((pl[i], pw[i], 0))
                    if (abs(pl[i] - pw[i]) > eps and
                            pw[i] <= sl[s] + eps and pl[i] <= sw[s] + eps):
                        opts.append((pw[i], pl[i], 1))
                if opts:
                    mask |= bits[i]
                row.append(opts)
            orient.append(row)
            masks.append(mask)

        stocks = sorted(range(S), key=lambda s: sa[s])
        initial = next(((s,) for s in stocks
                        if required & ~masks[s] == 0 and masks[s]), None)
        if initial is None:
            for a, s in enumerate(stocks):
                for t in stocks[a + 1:]:
                    if required & ~(masks[s] | masks[t]) == 0:
                        initial = (s, t)
                        break
                if initial is not None:
                    break
        if initial is None:
            return best

        # Before searching, build a complete fallback solution with one sheet per required piece.
        fallback = []
        for i in range(m):
            if not lo[i]:
                continue
            s = min((s for s in initial if orient[s][i]), key=lambda s: sa[s])
            p = (i, 0.0, 0.0, orient[s][i][0][2])
            fallback.extend((s, (p,)) for _ in range(lo[i]))
        if not fallback:
            possibilities = ((pa[i] / sa[s], s, i) for s in stocks
                             for i in range(m) if orient[s][i])
            _, s, i = max(possibilities)
            initial = (s,)
            fallback = [(s, ((i, 0.0, 0.0, orient[s][i][0][2]),))]

        def output(snapshot, counts):
            selected = [0] * S
            sheets = []
            for s, placements in snapshot:
                selected[s] = 1
                sheets.append({"stock_type_id": s, "placements": [
                    {"piece_type_id": i, "x": float(x), "y": float(y),
                     "is_rotated": int(r)} for i, x, y, r in placements]})
            return {"selected_stock_types": selected, "stock_instances": sheets,
                    "piece_counts": {f"piece_{i}": int(counts[i]) for i in range(m)}}

        best_snap = fallback
        best_counts = [0] * m
        for _, ps in fallback:
            for i, _, _, _ in ps:
                best_counts[i] += 1
        best_area = sum(sa[s] for s, _ in fallback)
        best_density = sum(pa[i] * best_counts[i] for i in range(m)) / best_area
        best = output(best_snap, best_counts)
        if time.perf_counter() >= deadline:
            return best

        seed = 146959810
        for v in sl + sw + pl + pw + lo + hi:
            seed = (seed * 1000003 + int(round(v * 1000))) & 0xFFFFFFFF
        rng = random.Random(seed)
        strength = math.log1p(budget)
        free_cap = max(16, int(math.sqrt(m + 1) * (8 + 4 * strength)))
        column_cap = max(m * S, int((m + S + 1) * (8 + 8 * math.sqrt(budget))))
        pool = {}
        max_pa = max(max(pa), eps)
        max_dim = max(max(pl), max(pw), eps)
        fit_count = [sum(bool(orient[s][i]) for s in range(S)) for i in range(m)]

        def remember(s, placements):
            if placements:
                mask = 0
                area = 0.0
                for i, _, _, _ in placements:
                    mask |= bits[i]
                    area += pa[i]
                key = (s, mask)
                if key not in pool:
                    pool[key] = (s, tuple(placements), area, mask)

        for s in range(S):
            for i in range(m):
                if orient[s][i]:
                    remember(s, [(i, 0.0, 0.0, orient[s][i][0][2])])

        def split(free, used):
            x, y, w, h = used
            right, top = x + w, y + h
            result = []
            for fx, fy, fw, fh in free:
                fr, ft = fx + fw, fy + fh
                if x >= fr - eps or right <= fx + eps or y >= ft - eps or top <= fy + eps:
                    result.append((fx, fy, fw, fh))
                    continue
                if x > fx + eps:
                    result.append((fx, fy, x - fx, fh))
                if right < fr - eps:
                    result.append((right, fy, fr - right, fh))
                if y > fy + eps:
                    result.append((fx, fy, fw, y - fy))
                if top < ft - eps:
                    result.append((fx, top, fw, ft - top))
            # Discarding empty rectangles cannot create illegal positions; capacity grows with size and budget.
            result.sort(key=lambda r: r[2] * r[3], reverse=True)
            kept = []
            for r in result:
                if r[2] <= eps or r[3] <= eps:
                    continue
                if not any(r[0] >= q[0] - eps and r[1] >= q[1] - eps and
                           r[0] + r[2] <= q[0] + q[2] + eps and
                           r[1] + r[3] <= q[1] + q[3] + eps for q in kept):
                    kept.append(r)
                    if len(kept) >= free_cap:
                        break
            return kept

        def new_bin(s):
            return {"s": s, "p": [], "f": [(0.0, 0.0, sl[s], sw[s])],
                    "mask": 0, "a": 0.0}

        def position(b, i, mode):
            if b["mask"] & bits[i]:
                return None
            answer = None
            s = b["s"]
            scale = max(sl[s], sw[s], eps)
            for w, h, r in orient[s][i]:
                for x, y, fw, fh in b["f"]:
                    if w > fw + eps or h > fh + eps:
                        continue
                    dw, dh = max(0.0, fw - w), max(0.0, fh - h)
                    short, long = min(dw, dh) / scale, max(dw, dh) / scale
                    left = (fw * fh - w * h) / max(sa[s], eps)
                    if mode == 0:
                        corners = ((x, y),)
                        geom = (short, long, left)
                    elif mode == 1:
                        corners = ((x, y + dh),)
                        geom = (left, short, long)
                    elif mode == 2:
                        corners = ((x, y),)
                        geom = ()
                    elif mode == 3:
                        corners = ((x + dw, y + dh),)
                        geom = (long, short, left)
                    else:
                        corners = ((x, y), (x + dw, y), (x, y + dh), (x + dw, y + dh))
                        geom = ()
                    for px, py in corners:
                        if mode == 2:
                            key = ((py + h) / scale, px / scale, short, left)
                        elif mode == 4:
                            contact = 0.0
                            if px < eps or abs(px + w - sl[s]) < eps:
                                contact += h
                            if py < eps or abs(py + h - sw[s]) < eps:
                                contact += w
                            for j, ax, ay, ar in b["p"]:
                                aw, ah = (pw[j], pl[j]) if ar else (pl[j], pw[j])
                                if abs(ax + aw - px) < eps or abs(px + w - ax) < eps:
                                    contact += max(0.0, min(py + h, ay + ah) - max(py, ay))
                                if abs(ay + ah - py) < eps or abs(py + h - ay) < eps:
                                    contact += max(0.0, min(px + w, ax + aw) - max(px, ax))
                            key = (-contact / scale, short, left, py / scale, px / scale)
                        else:
                            key = geom + ((py + h) / scale, px / scale)
                        candidate = (key, px, py, w, h, r)
                        if answer is None or key < answer[0]:
                            answer = candidate
            return answer

        def place(b, i, pos):
            _, x, y, w, h, r = pos
            b["p"].append((i, x, y, r))
            b["mask"] |= bits[i]
            b["a"] += pa[i]
            b["f"] = split(b["f"], (x, y, w, h))

        def restore(s, ps):
            b = new_bin(s)
            for i, x, y, r in ps:
                w, h = (pw[i], pl[i]) if r else (pl[i], pw[i])
                place(b, i, ((), x, y, w, h, r))
            return b

        def priorities(remaining, strategy, noisy):
            largest = max(max(remaining), 1)
            priorities_ = []
            for i in range(m):
                a = pa[i] / max_pa
                long = max(pl[i], pw[i]) / max_dim
                short = min(pl[i], pw[i]) / max_dim
                demand = remaining[i] / largest
                if strategy == 0:
                    score = 2 * demand + a + long
                elif strategy == 1:
                    score = 3 * a + long + 0.3 * demand
                elif strategy == 2:
                    score = 3 * long + short + 0.3 * demand
                elif strategy == 3:
                    score = 3 * short + a + 0.5 / max(fit_count[i], 1)
                else:
                    score = a + long + demand + rng.random()
                priorities_.append(score * (1.0 + noisy * (rng.random() - 0.5)))
            return priorities_

        def pattern(s, remaining, mode, priority, forced=None):
            b = new_bin(s)
            if forced is not None:
                pos = position(b, forced, mode)
                if pos is None:
                    return b
                place(b, forced, pos)
            order = sorted((i for i in range(m) if remaining[i] > 0 and orient[s][i]),
                           key=lambda i: priority[i], reverse=True)
            for i in order:
                if time.perf_counter() >= deadline:
                    break
                pos = position(b, i, mode)
                if pos is not None:
                    place(b, i, pos)
            remember(s, b["p"])
            return b

        def offer(bins):
            nonlocal best, best_snap, best_counts, best_area, best_density
            counts = [0] * m
            total = 0.0
            used = set()
            for b in bins:
                if not b["p"]:
                    continue
                used.add(b["s"])
                total += sa[b["s"]]
                for i, _, _, _ in b["p"]:
                    counts[i] += 1
            if (not total or len(used) > 2 or
                    any(counts[i] < lo[i] or counts[i] > hi[i] for i in range(m))):
                return False
            density = sum(counts[i] * pa[i] for i in range(m)) / total
            if (density < best_density - 1e-12 or
                    (density <= best_density + 1e-12 and total >= best_area - eps)):
                return False
            # Only solutions that may improve the incumbent are verified; patterns and their geometric layouts are independent of solver variables.
            for b in bins:
                s = b["s"]
                seen = set()
                rects = []
                for i, x, y, r in b["p"]:
                    w, h = (pw[i], pl[i]) if r else (pl[i], pw[i])
                    if (i in seen or not math.isfinite(x + y) or x < -eps or y < -eps or
                            x + w > sl[s] + eps or y + h > sw[s] + eps):
                        return False
                    if any(x < ax + aw - eps and x + w > ax + eps and
                           y < ay + ah - eps and y + h > ay + eps
                           for ax, ay, aw, ah in rects):
                        return False
                    seen.add(i)
                    rects.append((x, y, w, h))
            snapshot = [(b["s"], tuple(b["p"])) for b in bins if b["p"]]
            solution = output(snapshot, counts)
            best, best_snap = solution, snapshot
            best_counts, best_area, best_density = counts, total, density
            for s, ps in snapshot:
                remember(s, ps)
            return True

        def augment(bins, counts, mode, noisy):
            order = sorted(bins, key=lambda b: b["a"] / sa[b["s"]])
            for b in order:
                while time.perf_counter() < deadline:
                    chosen = None
                    for i in range(m):
                        if counts[i] >= hi[i] or b["mask"] & bits[i]:
                            continue
                        if time.perf_counter() >= deadline:
                            break
                        pos = position(b, i, mode)
                        if pos is not None:
                            gain = pa[i] * (1.0 + noisy * (rng.random() - 0.5))
                            key = (-gain, pos[0])
                            if chosen is None or key < chosen[0]:
                                chosen = (key, i, pos)
                    if chosen is None:
                        break
                    _, i, pos = chosen
                    place(b, i, pos)
                    counts[i] += 1
                remember(b["s"], b["p"])
                if time.perf_counter() >= deadline:
                    break

        def improve_quantity(bins, counts, allowed, mode, noisy):
            if not bins:
                return
            total = sum(sa[b["s"]] for b in bins)
            packed = sum(pa[i] * counts[i] for i in range(m))
            # Remove sheets below the current average utilisation that are not needed to meet the lower bounds.
            for b in sorted(bins[:], key=lambda b: b["a"] / sa[b["s"]]):
                if time.perf_counter() >= deadline:
                    return
                s = b["s"]
                if (len(bins) > 1 and b["a"] / sa[s] <= packed / total + 1e-12 and
                        all(counts[i] > lo[i] for i, _, _, _ in b["p"])):
                    bins.remove(b)
                    total -= sa[s]
                    packed -= b["a"]
                    for i, _, _, _ in b["p"]:
                        counts[i] -= 1
            # The ratio objective allows opening extra high-utilisation sheets rather than only minimising the number of sheets.
            while time.perf_counter() < deadline:
                remain = [hi[i] - counts[i] for i in range(m)]
                if not any(remain):
                    break
                priority = priorities(remain, 1, noisy)
                candidates = [pattern(s, remain, mode, priority) for s in allowed
                              if time.perf_counter() < deadline]
                if not candidates:
                    break
                b = max(candidates, key=lambda b: b["a"] / sa[b["s"]])
                if not b["p"] or b["a"] / sa[b["s"]] <= packed / total + 1e-12:
                    break
                copies = min(remain[i] for i, _, _, _ in b["p"])
                while copies and time.perf_counter() < deadline:
                    bins.append({"s": b["s"], "p": b["p"][:], "f": b["f"][:],
                                 "mask": b["mask"], "a": b["a"]})
                    for i, _, _, _ in b["p"]:
                        counts[i] += 1
                    packed += b["a"]
                    total += sa[b["s"]]
                    copies -= 1

        def construct(allowed, target, retained, strategy, mode, noisy):
            bins = []
            counts = [0] * m
            for s, ps in retained:
                if time.perf_counter() >= deadline:
                    return
                bins.append(restore(s, ps))
                for i, _, _, _ in ps:
                    counts[i] += 1
            remaining = [max(0, target[i] - counts[i]) for i in range(m)]
            priority = priorities(remaining, strategy, noisy)
            while any(remaining) and time.perf_counter() < deadline:
                i = max((i for i in range(m) if remaining[i]),
                        key=lambda i: priority[i])
                chosen = None
                # Reuse across sheets first, then compare sheet types by trial-packing a whole sheet containing the currently required pieces.
                for b in bins:
                    if time.perf_counter() >= deadline:
                        return
                    pos = position(b, i, mode)
                    if pos is not None and (chosen is None or pos[0] < chosen[0]):
                        chosen = (pos[0], b, pos)
                if chosen is not None:
                    place(chosen[1], i, chosen[2])
                    remaining[i] -= 1
                    counts[i] += 1
                    continue
                candidates = []
                for s in allowed:
                    if orient[s][i] and time.perf_counter() < deadline:
                        b = pattern(s, remaining, mode, priority, i)
                        urgent = sum(remaining[j] for j, _, _, _ in b["p"])
                        score = b["a"] / sa[s] + 0.025 * urgent / max(1, sum(remaining))
                        candidates.append((score, -sa[s], b))
                if not candidates:
                    return
                b = max(candidates, key=lambda item: item[:2])[2]
                bins.append(b)
                for j, _, _, _ in b["p"]:
                    remaining[j] -= 1
                    counts[j] += 1
            if any(remaining):
                return
            offer(bins)
            if not bins:
                remaining = [hi[i] if any(orient[s][i] for s in allowed) else 0
                             for i in range(m)]
                candidates = [pattern(s, remaining, mode, priority) for s in allowed]
                b = max(candidates, key=lambda b: b["a"] / sa[b["s"]])
                if not b["p"]:
                    return
                bins.append(b)
                for i, _, _, _ in b["p"]:
                    counts[i] += 1
            augment(bins, counts, mode, noisy)
            offer(bins)
            improve_quantity(bins, counts, allowed, mode, noisy)
            offer(bins)

        milp_api = None
        master_disabled = False

        def recombine():
            nonlocal milp_api, master_disabled
            if master_disabled or time.perf_counter() >= deadline:
                return
            try:
                if milp_api is None:
                    from scipy.optimize import Bounds, LinearConstraint, milp
                    from scipy.sparse import csc_matrix
                    milp_api = (Bounds, LinearConstraint, milp, csc_matrix)
                Bounds, LinearConstraint, milp, csc_matrix = milp_api
                left = deadline - time.perf_counter()
                if left <= budget * 0.025:
                    return
                protected = {(s, sum(bits[i] for i, _, _, _ in ps)) for s, ps in best_snap}
                entries = list(pool.items())
                entries.sort(key=lambda kv: (
                    kv[0] in protected or len(kv[1][1]) == 1,
                    kv[1][2] / sa[kv[1][0]] * (0.85 + 0.3 * rng.random())), reverse=True)
                columns = [v for _, v in entries[:column_cap]]
                n = len(columns)
                rows, cols, vals = [], [], []
                upper = []
                objective = []
                scale = max(max(sa), eps)
                for j, (s, ps, area, _) in enumerate(columns):
                    upper.append(min(hi[i] for i, _, _, _ in ps))
                    objective.append((best_density * sa[s] - area) / scale)
                    for i, _, _, _ in ps:
                        rows.append(i)
                        cols.append(j)
                        vals.append(1.0)
                    rows.extend((m + s, m + S + 1))
                    cols.extend((j, j))
                    vals.extend((1.0, 1.0))
                for s in range(S):
                    bound = sum(hi[i] for i in range(m) if orient[s][i])
                    rows.extend((m + s, m + S))
                    cols.extend((n + s, n + s))
                    vals.extend((-float(bound), 1.0))
                    upper.append(1)
                    objective.append(0.0)
                matrix = csc_matrix((vals, (rows, cols)), shape=(m + S + 2, n + S))
                lower_rows = lo + [-math.inf] * (S + 1) + [1.0]
                upper_rows = hi + [0.0] * S + [2.0, math.inf]
                allowance = min((deadline - time.perf_counter()) * 0.18,
                                budget / (5.0 + math.log1p(n)))
                if allowance <= 0:
                    return
                result = milp(objective, integrality=[1] * (n + S),
                              bounds=Bounds([0.0] * (n + S), upper),
                              constraints=LinearConstraint(matrix, lower_rows, upper_rows),
                              options={"time_limit": allowance, "mip_rel_gap": 0.0,
                                       "presolve": True})
                if result.x is None:
                    return
                multiplicities = [max(0, int(round(v))) for v in result.x[:n]]
                counts = [0] * m
                stock_area = 0.0
                used = set()
                for number, (s, ps, area, _) in zip(multiplicities, columns):
                    if number:
                        used.add(s)
                        stock_area += number * sa[s]
                        for i, _, _, _ in ps:
                            counts[i] += number
                if (not stock_area or len(used) > 2 or
                        any(counts[i] < lo[i] or counts[i] > hi[i] for i in range(m))):
                    return
                density = sum(counts[i] * pa[i] for i in range(m)) / stock_area
                if density < best_density - 1e-12:
                    return
                bins = []
                for number, (s, ps, area, mask) in zip(multiplicities, columns):
                    while number and time.perf_counter() < deadline:
                        bins.append({"s": s, "p": ps, "f": [], "mask": mask, "a": area})
                        number -= 1
                    if number:
                        return
                offer(bins)
            except Exception:
                # Without an integer solver, geometric search and the feasible fallback are unaffected.
                master_disabled = True

        def pair_stream():
            for s in stocks:
                if masks[s] and required & ~masks[s] == 0:
                    yield (s,)
            for a, s in enumerate(stocks):
                for t in stocks[a + 1:]:
                    if time.perf_counter() >= deadline:
                        return
                    if required & ~(masks[s] | masks[t]) == 0:
                        yield (s, t)

        # Obtain one fused constructive solution first, then hand the generated geometric patterns to integer recombination.
        construct(initial, lo[:], [], 0, 0, 0.0)
        iteration = 0
        stream = pair_stream()
        master_interval = budget / (5.0 + math.sqrt(m + S))
        next_master = time.perf_counter()
        last_pool = -1
        last_density = -math.inf
        while time.perf_counter() < deadline:
            now = time.perf_counter()
            if (now >= next_master and deadline - now > budget * 0.08 and
                    (len(pool) != last_pool or best_density > last_density + 1e-12)):
                last_pool, last_density = len(pool), best_density
                recombine()
                next_master = time.perf_counter() + master_interval
                if time.perf_counter() >= deadline:
                    break

            iteration += 1
            mode = iteration % 5
            strategy = (iteration // 5) % 5
            noisy = 0.15 + 0.65 * rng.random()
            retained = []
            use_repair = bool(best_snap) and rng.random() < 0.55
            if use_repair:
                allowed = tuple(sorted({s for s, _ in best_snap}))
                fraction = min(1.0, (0.12 + 0.5 * rng.random()) *
                               (1.0 + strength / (4.0 + math.sqrt(len(best_snap)))))
                destroy = max(1, int(math.ceil(len(best_snap) * fraction)))
                ranked = sorted(range(len(best_snap)), key=lambda j:
                                sum(pa[p[0]] for p in best_snap[j][1]) /
                                sa[best_snap[j][0]] + rng.random() * noisy)
                removed = set(ranked[:destroy])
                retained = [b for j, b in enumerate(best_snap) if j not in removed]
            else:
                allowed = next(stream, None)
                if allowed is None:
                    stream = pair_stream()
                    allowed = next(stream, initial)

            phase = iteration % 6
            if phase == 0:
                target = hi[:]
            elif phase == 1 or phase == 2:
                target = lo[:]
            elif phase == 3:
                target = best_counts[:]
            elif phase == 4:
                height = rng.randint(max(lo), max(hi))
                target = [min(hi[i], max(lo[i], height)) for i in range(m)]
            else:
                target = best_counts[:]
                moves = max(1, int(math.sqrt(m) * (1 + strength / 3)))
                for _ in range(moves):
                    i = rng.randrange(m)
                    target[i] = rng.randint(lo[i], hi[i])
            for i in range(m):
                if not any(orient[s][i] for s in allowed):
                    target[i] = 0
            construct(allowed, target, retained, strategy, mode, noisy)
        return best
    except Exception:
        return best
