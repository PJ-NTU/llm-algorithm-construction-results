"""Trimmable pattern library with fractional integer optimisation
Generates feasible patterns with maximal empty rectangles and time-limited branch search, adds patterns guided by demand prices, and uses an integer master problem to jointly optimise pattern counts, quantities and the choice between the two stock types.
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
                "piece_counts": {"piece_%d" % i: 0 for i in range(m)}}
        budget = max(0.0, float(time_limit_s))
        end = start + budget * 0.94
        sl = list(map(float, instance["length"]))
        sw = list(map(float, instance["width"]))
        sa = list(map(float, instance["area"]))
        pl = list(map(float, instance["piece_length"]))
        pw = list(map(float, instance["piece_width"]))
        pa = list(map(float, instance["piece_area"]))
        lo = list(map(int, instance["min"]))
        hi = list(map(int, instance["max"]))
        if not S or not m:
            return best
        eps = 1e-9
        ga = [pl[i] * pw[i] for i in range(m)]
        bits = [1 << i for i in range(m)]
        required = sum(bits[i] for i in range(m) if lo[i] > 0)
        orientations = [[] for _ in range(m)]
        for i in range(m):
            orientations[i].append((pl[i], pw[i], 0))
            if pl[i] != pw[i]:
                orientations[i].append((pw[i], pl[i], 1))
        fits, cover = [], []
        for s in range(S):
            ids = [i for i in range(m) if hi[i] > 0 and any(
                w <= sl[s] + eps and h <= sw[s] + eps
                for w, h, _ in orientations[i])]
            fits.append(ids)
            cover.append(sum(bits[i] for i in ids))
        stocks = sorted(range(S), key=lambda s: sa[s])
        pair = next(((s,) for s in stocks if cover[s] & required == required), None)
        if pair is None:
            for a_index, a in enumerate(stocks):
                for b in stocks[a_index + 1:]:
                    if (cover[a] | cover[b]) & required == required:
                        pair = (a, b)
                        break
                if pair is not None:
                    break
        if pair is None:
            return best

        # A pattern is recorded as (stock sheet, type bitset, type tuple, geometry tuple, piece area).
        def single(s, i):
            w, h, r = next(o for o in orientations[i]
                           if o[0] <= sl[s] + eps and o[1] <= sw[s] + eps)
            return (s, bits[i], (i,), ((i, 0.0, 0.0, w, h, r),), pa[i])

        baseline = []
        for i in range(m):
            if lo[i] > 0:
                s = min((s for s in pair if cover[s] & bits[i]), key=lambda s: sa[s])
                baseline.append((single(s, i), lo[i]))
        if not baseline:
            candidates = [(pa[i] / sa[s], s, i) for s in range(S)
                          for i in fits[s] if sa[s] > 0]
            if not candidates:
                return best
            _, s, i = max(candidates)
            baseline = [(single(s, i), 1)]
        best_ratio = -math.inf
        best_area = math.inf
        best_records = []

        def consider(plan):
            nonlocal best, best_ratio, best_area, best_records
            if not plan:
                return
            used = {p[0] for p, n in plan if n > 0}
            if len(used) > 2:
                return
            remaining = hi[:]
            trimmed = []
            area = 0.0
            # Placements beyond the quantity bound can be trimmed piece by piece without changing the geometric feasibility of the rest.
            for p, number in sorted(plan, key=lambda pn: pn[0][4] / sa[pn[0][0]],
                                    reverse=True):
                number = int(number)
                if number <= 0:
                    continue
                number = min(number, max((remaining[i] for i in p[2]), default=0))
                if number <= 0:
                    continue
                take = {i: min(number, remaining[i]) for i in p[2]}
                for i, n in take.items():
                    remaining[i] -= n
                trimmed.append((p, number, take))
                area += number * sa[p[0]]
            counts = [hi[i] - remaining[i] for i in range(m)]
            if area <= 0 or any(counts[i] < lo[i] for i in range(m)):
                return
            ratio = sum(counts[i] * pa[i] for i in range(m)) / area
            if ratio < best_ratio + 1e-12 and not (
                    abs(ratio - best_ratio) <= 1e-12 and area < best_area - eps):
                return
            selected = [0] * S
            bins = []
            records = []
            for p, n, take in trimmed:
                s = p[0]
                selected[s] = 1
                records.append(p)
                for k in range(n):
                    places = [{"piece_type_id": i, "x": float(x), "y": float(y),
                               "is_rotated": int(r)}
                              for i, x, y, w, h, r in p[3] if k < take[i]]
                    bins.append({"stock_type_id": s, "placements": places})
            candidate = {"selected_stock_types": selected, "stock_instances": bins,
                         "piece_counts": {"piece_%d" % i: counts[i] for i in range(m)}}
            best, best_ratio, best_area, best_records = candidate, ratio, area, records

        consider(baseline)
        if time.perf_counter() >= end:
            return best
        seed = 173
        for v in sl + sw + pl + pw:
            seed = (seed * 1000003 + int(abs(v) * 1000)) & 0xFFFFFFFF
        rng = random.Random(seed)
        pool = [dict() for _ in range(S)]
        capacity = max(2 * m, int(30 * math.sqrt(m + 1) *
                                  (1 + math.sqrt(budget)) / math.sqrt(S)))
        branch_width = max(2, int(2 + math.sqrt(budget / (1 + m / 12))))
        prices = pa[:]
        serial = 0
        visit = [0] * S

        def register(s, placed):
            if not placed:
                return None
            mask = 0
            for j, (i, x, y, w, h, r) in enumerate(placed):
                if time.perf_counter() >= end:
                    return None
                if (mask & bits[i] or not math.isfinite(x + y) or
                        x < -eps or y < -eps or x + w > sl[s] + eps or
                        y + h > sw[s] + eps):
                    return None
                for _, a, b, c, d, _ in placed[:j]:
                    if not (x + w <= a + eps or a + c <= x + eps or
                            y + h <= b + eps or b + d <= y + eps):
                        return None
                mask |= bits[i]
            ids = tuple(p[0] for p in placed)
            p = (s, mask, ids, tuple(placed), sum(pa[i] for i in ids))
            library = pool[s]
            # The master problem allows trimming: a feasible superset on the same sheet strictly dominates its subsets.
            for old in library:
                if mask & old == mask:
                    return p
            redundant = [old for old in library if mask & old == old]
            for old in redundant:
                del library[old]
            library[mask] = p
            if len(library) > capacity:
                rows = sorted(library.values(),
                              key=lambda p: p[4] / sa[s], reverse=True)
                protected = {}
                for i in fits[s]:
                    q = next((q for q in rows if q[1] & bits[i]), None)
                    if q is not None:
                        protected[q[1]] = q
                for old in best_records:
                    if old[0] == s:
                        q = next((q for q in rows if q[1] & old[1] == old[1]), None)
                        if q is not None:
                            protected[q[1]] = q
                protected[mask] = p
                # Balance utilisation, demand prices and combination diversity, keeping patterns that cover every piece type.
                rows.sort(key=lambda q: (0.4 * q[4] +
                                         sum(prices[i] for i in q[2])) *
                          (0.8 + 0.4 * rng.random()), reverse=True)
                for q in rows:
                    if len(protected) >= capacity:
                        break
                    protected[q[1]] = q
                pool[s] = protected
            return p

        for s in stocks:
            for i in fits[s]:
                p = single(s, i)
                pool[s][p[1]] = p
            if time.perf_counter() >= end:
                return best

        def split(free, used, stop):
            _, x, y, w, h, _ = used
            right, top = x + w, y + h
            result = []
            for a, b, c, d in free:
                ar, bt = a + c, b + d
                if right <= a + eps or ar <= x + eps or top <= b + eps or bt <= y + eps:
                    result.append((a, b, c, d))
                    continue
                if x > a + eps:
                    result.append((a, b, x - a, d))
                if right < ar - eps:
                    result.append((right, b, ar - right, d))
                if y > b + eps:
                    result.append((a, b, c, y - b))
                if top < bt - eps:
                    result.append((a, top, c, bt - top))
            result.sort(key=lambda f: f[2] * f[3], reverse=True)
            kept = []
            for a, b, c, d in result:
                if time.perf_counter() >= stop:
                    return None
                if c <= eps or d <= eps:
                    continue
                if not any(a >= u - eps and b >= v - eps and
                           a + c <= u + w + eps and b + d <= v + h + eps
                           for u, v, w, h in kept):
                    kept.append((a, b, c, d))
            return kept

        def positions(s, i, free, placed, mode, stop, all_corners=False):
            result, seen = [], set()
            for x, y, fw, fh in free:
                if time.perf_counter() >= stop:
                    break
                for w, h, r in orientations[i]:
                    if w > fw + eps or h > fh + eps:
                        continue
                    corners = [(x, y)]
                    if all_corners or mode == 5:
                        corners += [(max(x, x + fw - w), y),
                                    (x, max(y, y + fh - h)),
                                    (max(x, x + fw - w), max(y, y + fh - h))]
                    dw, dh = max(0.0, fw - w), max(0.0, fh - h)
                    for a, b in corners:
                        key = (a, b, r)
                        if key in seen:
                            continue
                        seen.add(key)
                        if mode == 0:
                            score = (min(dw, dh), max(dw, dh), b, a)
                        elif mode == 1:
                            score = (fw * fh - w * h, min(dw, dh), b, a)
                        elif mode == 2:
                            score = (b + h, a, min(dw, dh))
                        elif mode == 3:
                            score = (max(dw, dh), min(dw, dh), b, a)
                        elif mode == 4:
                            contact = (h if a <= eps or abs(a + w - sl[s]) <= eps else 0)
                            contact += (w if b <= eps or abs(b + h - sw[s]) <= eps else 0)
                            for _, px, py, qw, qh, _ in placed:
                                if abs(a + w - px) <= eps or abs(px + qw - a) <= eps:
                                    contact += max(0, min(b + h, py + qh) - max(b, py))
                                if abs(b + h - py) <= eps or abs(py + qh - b) <= eps:
                                    contact += max(0, min(a + w, px + qw) - max(a, px))
                            score = (-contact, fw * fh - w * h, b, a)
                        else:
                            score = ((fw * fh - w * h) * (0.8 + 0.4 * rng.random()),
                                     min(dw, dh), b, a)
                        result.append((score, (i, a, b, w, h, r)))
            result.sort(key=lambda v: v[0])
            return result

        def pack(s, weights, variant, stop, initial=()):
            placed = list(initial)
            free = [(0.0, 0.0, sl[s], sw[s])]
            for p in placed:
                free = split(free, p, stop)
                if free is None:
                    return register(s, placed)
            present = {p[0] for p in placed}
            ids = [i for i in fits[s] if i not in present]
            mode = variant % 6
            order_mode = variant // 6 % 7
            if order_mode == 0:
                ids.sort(key=lambda i: (weights[i], ga[i]), reverse=True)
            elif order_mode == 1:
                ids.sort(key=lambda i: (max(pl[i], pw[i]), weights[i]), reverse=True)
            elif order_mode == 2:
                ids.sort(key=lambda i: (min(pl[i], pw[i]), weights[i]), reverse=True)
            elif order_mode == 3:
                ids.sort(key=lambda i: weights[i] / max(ga[i], eps) ** 0.5,
                         reverse=True)
            else:
                exponent = rng.random()
                ids.sort(key=lambda i: max(weights[i], eps) /
                         max(ga[i], eps) ** exponent * (0.3 + rng.random()), reverse=True)
            for i in ids:
                if time.perf_counter() >= stop:
                    break
                options = positions(s, i, free, placed, mode, stop)
                if options:
                    p = options[0][1]
                    newfree = split(free, p, stop)
                    if newfree is None:
                        break
                    free = newfree
                    placed.append(p)
            return register(s, placed)

        pairs = [(s,) for s in stocks if cover[s] & required == required]
        pairs.append(pair)
        for j, a in enumerate(stocks):
            if time.perf_counter() >= start + budget * 0.06:
                break
            for b in stocks[j + 1:]:
                if (cover[a] | cover[b]) & required == required:
                    pairs.append((a, b))
        pairs = list(dict.fromkeys(pairs))

        def greedy(allowed, variant, stop):
            patterns = [p for s in allowed for p in pool[s].values()]
            room = hi[:]
            need = lo[:]
            plan = []
            stock_area = 0.0
            piece_area = 0.0
            while time.perf_counter() < stop:
                mandatory = any(n > 0 for n in need)
                choice = None
                for p in patterns:
                    if time.perf_counter() >= stop:
                        return
                    gain = sum(pa[i] for i in p[2] if room[i] > 0)
                    demanded = [i for i in p[2] if need[i] > 0]
                    if mandatory:
                        if not demanded:
                            continue
                        urgent = sum(pa[i] * (1 + need[i] / max(1, lo[i]))
                                     for i in demanded)
                        score = (urgent + (0.2 + 0.5 * (variant % 3)) * gain) / sa[p[0]]
                        number = min(need[i] for i in demanded)
                    else:
                        if gain <= 0 or (stock_area > 0 and
                                         gain * stock_area <= piece_area * sa[p[0]] + eps):
                            continue
                        score = gain / sa[p[0]]
                        number = min(room[i] for i in p[2] if room[i] > 0)
                    if variant >= 3:
                        score *= 0.85 + 0.3 * rng.random()
                    if choice is None or score > choice[0]:
                        choice = (score, p, number)
                if choice is None:
                    break
                _, p, number = choice
                plan.append((p, number))
                stock_area += sa[p[0]] * number
                for i in p[2]:
                    n = min(number, room[i])
                    room[i] -= n
                    need[i] = max(0, need[i] - n)
                    piece_area += n * pa[i]
                if not any(need):
                    consider(plan)
            consider(plan)

        quick_end = min(end, start + budget * 0.16)
        for s in stocks:
            if time.perf_counter() >= quick_end:
                break
            pack(s, pa, 0, quick_end)
        for allowed in pairs:
            if time.perf_counter() >= quick_end:
                break
            greedy(allowed, 0, quick_end)

        # The general scientific-computing module is loaded only after a feasible fallback exists; if unavailable, pure-Python search continues.
        numeric = False
        if end - time.perf_counter() > max(0.08, budget * 0.025):
            try:
                import numpy as np
                from scipy.optimize import Bounds, LinearConstraint, linprog, milp
                from scipy.sparse import coo_matrix
                numeric = True
            except Exception:
                numeric = False

        def master(stop):
            nonlocal prices, numeric
            if not numeric or time.perf_counter() >= stop:
                return
            patterns = [p for lib in pool for p in lib.values()]
            n = len(patterns)
            if not n:
                return
            try:
                scale = max(max(sa), max(pa), eps)
                count = n + m + S
                c = np.zeros(count)
                lower = np.zeros(count)
                upper = np.zeros(count)
                integer = np.zeros(count, dtype=int)
                rr, cc, vv = [], [], []
                caps = [max(1, sum(hi[i] for i in fits[s])) for s in range(S)]
                for j, p in enumerate(patterns):
                    if time.perf_counter() >= stop:
                        return
                    s = p[0]
                    c[j] = best_ratio * sa[s] / scale
                    upper[j] = max(hi[i] for i in p[2])
                    integer[j] = 1
                    for i in p[2]:
                        rr.append(i)
                        cc.append(j)
                        vv.append(-1.0)
                    rr.append(m + s)
                    cc.append(j)
                    vv.append(1.0 / caps[s])
                for i in range(m):
                    j = n + i
                    c[j] = -pa[i] / scale
                    lower[j], upper[j] = lo[i], hi[i]
                    rr.append(i)
                    cc.append(j)
                    vv.append(1.0)
                for s in range(S):
                    j = n + m + s
                    upper[j] = 1
                    integer[j] = 1
                    rr.extend((m + s, m + S))
                    cc.extend((j, j))
                    vv.extend((-1.0, 1.0))
                matrix = coo_matrix((vv, (rr, cc)), shape=(m + S + 1, count)).tocsc()
                rhs = np.zeros(m + S + 1)
                rhs[-1] = 2
                left = stop - time.perf_counter()
                if left <= 0:
                    return
                # Demand shadow prices of the linear relaxation guide later single-sheet packing; they do not replace the true objective.
                lp_time = min(left * 0.18, budget * 0.025)
                result = linprog(c, A_ub=matrix, b_ub=rhs,
                                 bounds=list(zip(lower, upper)), method="highs",
                                 options={"time_limit": max(0.001, lp_time)})
                if result.success:
                    dual = result.ineqlin.marginals
                    prices = [max(0.0, -float(dual[i]) * scale) for i in range(m)]
                left = stop - time.perf_counter()
                if left <= 0:
                    return
                result = milp(c, integrality=integer, bounds=Bounds(lower, upper),
                              constraints=LinearConstraint(matrix, -np.inf, rhs),
                              options={"time_limit": max(0.001, left * 0.94),
                                       "mip_rel_gap": 0.0})
                if result.x is not None:
                    plan = [(p, max(0, int(round(float(result.x[j])))))
                            for j, p in enumerate(patterns) if result.x[j] > 0.5]
                    consider(plan)
            except Exception:
                # Keep the existing solution; construction and geometric search can still improve it later.
                numeric = False

        def branch_pack(s, ids, stop):
            if sum(ga[i] for i in ids) > sl[s] * sw[s] + eps:
                return None
            if len(ids) == len(set(ids)):
                target = tuple(sorted(ids, key=lambda i: -ga[i]))
            else:
                return None
            stack = [(target, [(0.0, 0.0, sl[s], sw[s])], ())]
            while stack and time.perf_counter() < stop:
                remaining, free, placed = stack.pop()
                if not remaining:
                    return placed
                alternatives = []
                impossible = False
                shapes = set()
                for i in remaining:
                    if time.perf_counter() >= stop:
                        return None
                    options = positions(s, i, free, placed, 0, stop, True)
                    if not options:
                        impossible = True
                        break
                    shape = (min(pl[i], pw[i]), max(pl[i], pw[i]))
                    if shape not in shapes:
                        shapes.add(shape)
                        alternatives.append((len(options), -ga[i], i, options))
                if impossible:
                    continue
                alternatives.sort(key=lambda a: (a[0], a[1]))
                children = []
                for _, _, i, options in alternatives[:branch_width]:
                    rest = tuple(j for j in remaining if j != i)
                    for _, p in options[:branch_width * branch_width]:
                        if time.perf_counter() >= stop:
                            return None
                        newfree = split(free, p, stop)
                        if newfree is not None:
                            children.append((rest, newfree, placed + (p,)))
                stack.extend(reversed(children))
            return None

        next_master = time.perf_counter()
        while time.perf_counter() < end:
            now = time.perf_counter()
            if numeric and now >= next_master:
                share = min(end - now, budget * (0.07 +
                            0.035 * math.log1p(1 + sum(len(lib) for lib in pool) / (m + S))))
                master(min(end, now + share))
                next_master = time.perf_counter() + budget * 0.045
                if time.perf_counter() >= end:
                    break
            # Rotation ensures every raw stock is tried; price guidance and random diversity alternate.
            if serial % 3 == 0:
                s = stocks[(serial // 3) % S]
            elif best_records and rng.random() < 0.5:
                s = rng.choice(best_records)[0]
            else:
                s = rng.randrange(S)
            if not fits[s]:
                serial += 1
                continue
            visit[s] += 1
            roll = rng.random()
            if roll < 0.45:
                weights = [prices[i] + 0.03 * pa[i] for i in range(m)]
            elif roll < 0.75:
                weights = [pa[i] * (0.3 + rng.random()) for i in range(m)]
            else:
                weights = [pa[i] * (1 + lo[i]) / (1 + hi[i]) *
                           (0.3 + rng.random()) for i in range(m)]
            local_end = min(end, time.perf_counter() +
                            budget / (8 + math.sqrt(m + S)))
            variant = (visit[s] - 1) % 42 if visit[s] <= 42 else rng.randrange(42)
            p = pack(s, weights, variant, local_end)
            serial += 1
            if serial % 3 == 1 and time.perf_counter() < end:
                allowed = pairs[(serial // 3) % len(pairs)] if rng.random() < 0.5 else rng.choice(pairs)
                greedy(allowed, serial % 6,
                       min(end, time.perf_counter() + budget * 0.025))
            # Try adding unplaced pieces to existing patterns; if needed, swap out low-price pieces and re-pack by branching.
            if p is not None and time.perf_counter() < end and serial % 2 == 0:
                missing = [i for i in fits[s] if not p[1] & bits[i]]
                if missing:
                    i = max(missing, key=lambda i: (weights[i] + eps) * (0.2 + rng.random()))
                    target = list(p[2]) + [i]
                    geometric = sum(ga[j] for j in target)
                    removable = sorted(p[2], key=lambda j: (weights[j] + eps) /
                                        max(ga[j], eps) * (0.7 + 0.6 * rng.random()))
                    while geometric > sl[s] * sw[s] + eps and removable:
                        j = removable.pop(0)
                        target.remove(j)
                        geometric -= ga[j]
                    if geometric <= sl[s] * sw[s] + eps:
                        sub_end = min(end, time.perf_counter() +
                                      budget / (35 + m + S))
                        placed = branch_pack(s, target, sub_end)
                        if placed is not None:
                            pack(s, weights, rng.randrange(42),
                                 min(end, time.perf_counter() + budget * 0.01), placed)
            if not numeric and serial % 2 == 0:
                greedy(rng.choice(pairs), serial % 6, end)
        return best
    except Exception:
        return best
