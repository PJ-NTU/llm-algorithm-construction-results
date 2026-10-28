"""Contact construction inside a circle with removal-and-repack search
Builds layouts from contact positions with the circular arc and existing rectangles, combines directional compaction, minimum-conflict replacement and adaptive local repacking, and keeps the best solution that passes hard-constraint checks.
"""

import math
import random
import time


def solve(instance: dict, time_limit_s: float) -> dict:
    started = time.perf_counter()
    answer = {"alpha": [], "x": [], "y": [], "r": []}
    try:
        n = int(instance["n"])
        answer = {"alpha": [0] * n, "x": [-1] * n,
                  "y": [-1] * n, "r": [0] * n}
        budget = max(0.0, float(time_limit_s))
        if not math.isfinite(budget) or budget <= 0.0 or n == 0:
            return answer
        end = started + budget
        deadline = end - min(0.06 * budget, 0.002 + 0.000002 * n)
        clock = time.perf_counter
        cx, cy, radius = (float(instance[k]) for k in ("cx", "cy", "R"))
        rotate = int(instance["rotation_enabled"])
        if radius <= 0.0 or not all(map(math.isfinite, (cx, cy, radius))):
            return answer
        raw = instance["items"]
        if len(raw) != n:
            return answer
        radius2 = radius * radius
        gap = max(3.0e-14, 8.0 * (math.ulp(cx) / radius + math.ulp(cy) / radius))
        halfgap = gap * 0.5
        safe_radius = max(0.0, 1.0 - 4.0 * gap)
        safe2 = safe_radius * safe_radius
        rng = random.Random(937451)
        dims = {}
        weight = {}
        original = {}
        eligible = []
        best_pack = []
        best_value = 0.0

        def inside_original(px, py, width, height):
            if not math.isfinite(px) or not math.isfinite(py):
                return False
            for xx in (px - width * 0.5, px + width * 0.5):
                for yy in (py - height * 0.5, py + height * 0.5):
                    dx, dy = xx - cx, yy - cy
                    if not math.isfinite(dx) or not math.isfinite(dy):
                        return False
                    if math.hypot(dx, dy) > radius:
                        return False
                    if 0.0 < radius2 < math.inf and dx * dx + dy * dy > radius2:
                        return False
            return True

        # First build a single-item feasible lower bound that can be returned at any time; radius normalisation avoids area overflow.
        for i, item in enumerate(raw):
            if clock() >= deadline:
                return answer
            width, height = float(item[0]), float(item[1])
            if not (width > 0.0 and height > 0.0):
                continue
            if not math.isfinite(width) or not math.isfinite(height):
                continue
            a, b = (width * 0.5) / radius, (height * 0.5) / radius
            if math.hypot(a, b) > 1.0 + 8.0e-15:
                continue
            original[i] = (width, height)
            dims[i] = (a, b)
            weight[i] = 4.0 * a * b
            eligible.append(i)
            if weight[i] > best_value and inside_original(cx, cy, width, height):
                if best_pack:
                    old = best_pack[0][0]
                    answer["alpha"][old] = 0
                    answer["x"][old] = answer["y"][old] = -1
                answer["alpha"][i] = 1
                answer["x"][i], answer["y"][i] = cx, cy
                best_pack = [(i, 0.0, 0.0, a, b, 0)]
                best_value = weight[i]
        if not eligible or safe_radius <= 0.0:
            return answer
        if len(eligible) == 1:
            return answer
        m = len(eligible)
        strength = math.sqrt(1.0 + math.log1p(budget))
        quantum = budget / (4.0 + math.sqrt(m) / (1.0 + math.log1p(budget)))
        pool_size = max(2, int(2.0 + math.log1p(budget) * math.sqrt(m) /
                               (1.0 + math.sqrt(m) / 4.0)))
        pool = []
        area_order = sorted(eligible, key=weight.__getitem__, reverse=True)
        logs = {i: (math.log(max(weight[i], 5e-324)),
                    math.log(max(max(dims[i]), 5e-324)),
                    math.log(max(min(dims[i]), 5e-324))) for i in eligible}

        def pack_value(pack):
            return math.fsum(weight[p[0]] for p in pack)

        def commit(pack):
            nonlocal answer, best_pack, best_value
            value = pack_value(pack)
            if value < best_value:
                return
            if value == best_value:
                old_ids = {p[0] for p in best_pack}
                new_ids = {p[0] for p in pack}
                if not old_ids < new_ids:
                    return
            converted = []
            for p in pack:
                if clock() >= end:
                    return
                i, x, y, a, b, r = p
                px, py = cx + x * radius, cy + y * radius
                width, height = original[i]
                if r:
                    width, height = height, width
                if r > rotate or not inside_original(px, py, width, height):
                    return
                for _, qx, qy, qw, qh, _ in converted:
                    if (abs(px - qx) < width * 0.5 + qw * 0.5 and
                            abs(py - qy) < height * 0.5 + qh * 0.5):
                        return
                converted.append((i, px, py, width, height, r))
            result = {"alpha": [0] * n, "x": [-1] * n,
                      "y": [-1] * n, "r": [0] * n}
            for i, x, y, _, _, r in converted:
                result["alpha"][i] = 1
                result["x"][i], result["y"][i] = x, y
                result["r"][i] = r
            answer, best_pack, best_value = result, list(pack), value

        def view_of(pack, mode):
            swap, sx, sy = mode
            if swap:
                view = [(sx * p[2], sy * p[1], p[4], p[3], p[0]) for p in pack]
            else:
                view = [(sx * p[1], sy * p[2], p[3], p[4], p[0]) for p in pack]
            view.sort(key=lambda q: q[1] - q[3])
            return view

        def from_view(i, x, y, a, b, r, mode):
            swap, sx, sy = mode
            if swap:
                return (i, sy * y, sx * x, b, a, r)
            return (i, sx * x, sy * y, a, b, r)

        def x_events(a, b, view, beta, stop):
            if a * a + b * b > safe2:
                return []
            xmax = max(0.0, math.sqrt(max(0.0, safe2 - b * b)) - a)
            xs = {0.0, -xmax, xmax}
            tangent = safe_radius * abs(beta) / math.sqrt(1.0 + beta * beta) - a
            if tangent > 0.0:
                xs.add(-math.copysign(tangent, beta))
            for qx, qy, qa, qb, _ in view:
                if clock() >= stop:
                    break
                for x in (qx - qa - a - gap, qx + qa + a + gap,
                          qx - qa + a, qx + qa - a, qx):
                    if -xmax <= x <= xmax:
                        xs.add(x)
                for y in (qy - qb - b - gap, qy + qb + b + gap):
                    edge = abs(y) + b
                    if edge <= safe_radius:
                        x = math.sqrt(max(0.0, safe2 - edge * edge)) - a
                        if x >= 0.0:
                            xs.add(x)
                            xs.add(-x)
            return list(xs)

        def place(i, pack, mode, beta, stop, view=None):
            if clock() >= stop:
                return None
            if view is None:
                view = view_of(pack, mode)
            aa, bb = dims[i]
            best, best_key = None, math.inf
            orientations = (0, 1) if rotate and aa != bb else (0,)
            for r in orientations:
                a, b = (bb, aa) if r else (aa, bb)
                if mode[0]:
                    a, b = b, a
                xs = x_events(a, b, view, beta, stop)
                candidates = []
                for x in xs:
                    ylimit = math.sqrt(max(0.0, safe2 - (abs(x) + a) ** 2)) - b
                    if ylimit >= -gap:
                        ylimit = max(0.0, ylimit)
                        candidates.append((-ylimit + b + beta * x, x, ylimit))
                candidates.sort()
                for lower_key, x, ylimit in candidates:
                    if clock() >= stop:
                        return best
                    if lower_key >= best_key:
                        break
                    y = -ylimit
                    for qx, qy, qa, qb, _ in view:
                        if y <= qy - qb - b - halfgap:
                            break
                        if abs(x - qx) < a + qa + halfgap:
                            if y < qy + qb + b + halfgap:
                                y = qy + qb + b + gap
                                if y > ylimit:
                                    break
                    if y <= ylimit:
                        key = y + b + beta * x
                        if key < best_key:
                            best_key = key
                            best = from_view(i, x, y, a, b, r, mode)
            return best

        def order_items(pack, initial=False):
            used = {p[0] for p in pack}
            pending = [i for i in eligible if i not in used]
            if initial:
                pending.sort(key=weight.__getitem__, reverse=True)
                return pending
            kind = rng.randrange(6)
            noise = rng.random() * strength
            keys = {}
            for i in pending:
                la, lm, ls = logs[i]
                if kind < 2:
                    key = la
                elif kind == 2:
                    key = lm
                elif kind == 3:
                    key = ls
                elif kind == 4:
                    key = 0.65 * la + 0.7 * ls
                else:
                    key = 0.5 * la + 0.7 * lm
                keys[i] = key + noise * (rng.random() - 0.5)
            pending.sort(key=keys.__getitem__, reverse=True)
            return pending

        def fill(pack, order, mode, beta, stop):
            failed = set()
            value = pack_value(pack)
            view = view_of(pack, mode)
            for i in order:
                if clock() >= stop:
                    break
                a, b = dims[i]
                shape = (min(a, b), max(a, b)) if rotate else (a, b)
                if shape in failed or value + weight[i] > math.pi:
                    continue
                p = place(i, pack, mode, beta, stop, view)
                if p is None:
                    failed.add(shape)
                else:
                    pack.append(p)
                    value += weight[i]
                    view = view_of(pack, mode)
            return pack

        def slide_axis(pack, axis, sign, stop):
            changed = False
            order = sorted(range(len(pack)), key=lambda j: sign * pack[j][axis + 1])
            for j in order:
                if clock() >= stop:
                    break
                p = pack[j]
                pos = sign * p[axis + 1]
                cross = p[2 - axis]
                half = p[axis + 3]
                otherhalf = p[4 - axis]
                edge = abs(cross) + otherhalf
                if edge > safe_radius:
                    continue
                low = half - math.sqrt(max(0.0, safe2 - edge * edge))
                for k, q in enumerate(pack):
                    if k == j:
                        continue
                    qp = sign * q[axis + 1]
                    if qp < pos and abs(cross - q[2 - axis]) < otherhalf + q[4 - axis] + halfgap:
                        low = max(low, qp + q[axis + 3] + half + gap)
                if low < pos - gap:
                    p = list(p)
                    p[axis + 1] = sign * low
                    pack[j] = tuple(p)
                    changed = True
            return changed

        def compact(pack, mode, stop):
            swap, sx, sy = mode
            primary = 0 if swap else 1
            changed = True
            while changed and clock() < stop:
                changed = slide_axis(pack, primary, sy, stop)
                changed = slide_axis(pack, 1 - primary, sx, stop) or changed
            return pack

        def eject_place(i, pack, mode, beta, stop):
            # Scan the conflict intervals for each candidate vertical line and find the position with minimum total removed area.
            view = view_of(pack, mode)
            aa, bb = dims[i]
            best, best_cost, best_key = None, math.inf, math.inf
            orientations = (0, 1) if rotate and aa != bb else (0,)
            for r in orientations:
                a, b = (bb, aa) if r else (aa, bb)
                if mode[0]:
                    a, b = b, a
                xs = x_events(a, b, view, beta, stop)
                rng.shuffle(xs)
                for x in xs:
                    if clock() >= stop:
                        break
                    limit = math.sqrt(max(0.0, safe2 - (abs(x) + a) ** 2)) - b
                    if limit < -gap:
                        continue
                    limit = max(0.0, limit)
                    bottom = -limit
                    events, starting = [], []
                    for qx, qy, qa, qb, qi in view:
                        if abs(x - qx) >= a + qa + halfgap:
                            continue
                        lo, hi = qy - qb - b - gap, qy + qb + b + gap
                        w = weight[qi]
                        if lo < bottom <= hi:
                            starting.append(w)
                        if bottom <= lo <= limit:
                            events.append((lo, 1, w))
                        if bottom <= hi <= limit:
                            events.append((hi, -1, w))
                    cost = math.fsum(starting)
                    events.append((bottom, 0, 0.0))
                    events.sort()
                    k = 0
                    while k < len(events):
                        y = events[k][0]
                        entered = 0.0
                        while k < len(events) and events[k][0] == y:
                            _, kind, w = events[k]
                            if kind < 0:
                                cost -= w
                            elif kind > 0:
                                entered += w
                            k += 1
                        c = max(0.0, cost)
                        key = y + b + beta * x
                        if c < best_cost or (c == best_cost and key < best_key):
                            best_cost, best_key = c, key
                            best = from_view(i, x, y, a, b, r, mode)
                        cost += entered
            if best is None:
                return None, []
            _, x, y, a, b, _ = best
            removed = [q[0] for q in pack
                       if abs(x - q[1]) < a + q[3] + halfgap and
                       abs(y - q[2]) < b + q[4] + halfgap]
            return best, removed

        def ruin(pack, count, mode):
            if not pack:
                return []
            kind = rng.randrange(4)
            anchor = rng.choice(pack)
            if kind == 0:
                ranked = sorted(pack, key=lambda p: (p[1] - anchor[1]) ** 2 +
                                (p[2] - anchor[2]) ** 2)
            elif kind == 1:
                ranked = sorted(pack, key=lambda p: rng.random())
            elif kind == 2:
                ranked = sorted(pack, key=lambda p: weight[p[0]] * (0.5 + rng.random()))
            else:
                axis, sign = (1, mode[2]) if mode[0] else (2, mode[2])
                ranked = sorted(pack, key=lambda p: -sign * p[axis])
            removed = {p[0] for p in ranked[:count]}
            return [p for p in pack if p[0] not in removed]

        def remember(pack):
            value = pack_value(pack)
            if not pack or value < best_value * (1.0 - 0.5 / strength):
                return
            signature = tuple(sorted((p[0], p[5]) for p in pack))
            for k, (v, old, sig) in enumerate(pool):
                if sig == signature:
                    if rng.random() < 0.5:
                        pool[k] = (value, list(pack), signature)
                    return
            pool.append((value, list(pack), signature))
            if len(pool) > pool_size:
                pool.sort(key=lambda q: q[0], reverse=True)
                pool.pop()

        first = True
        stale_since = clock()
        while clock() < deadline:
            previous_best = best_value
            initial = first
            mode = (False, 1, 1) if initial else (
                bool(rng.getrandbits(1)), rng.choice((-1, 1)), rng.choice((-1, 1)))
            beta = 0.035 if initial else rng.uniform(-0.22, 0.22)
            stalled = min(1.0, (clock() - stale_since) / max(quantum, budget * 0.15))
            restart = initial or not best_pack or rng.random() < 0.12 + 0.18 * stalled
            if restart:
                pack = []
            else:
                if pool and rng.random() < 0.4:
                    pack = list(rng.choice(pool)[1])
                else:
                    pack = list(best_pack)
                used = {p[0] for p in pack}
                missing = [i for i in area_order if i not in used]
                if missing and rng.random() < 0.45:
                    rank = min(len(missing) - 1,
                               int(len(missing) * rng.random() ** strength))
                    target = missing[rank]
                    p, removed = eject_place(target, pack, mode, beta,
                                             min(deadline, clock() + quantum * 0.35))
                    if p is not None:
                        blocked = set(removed)
                        pack = [q for q in pack if q[0] not in blocked]
                        pack.append(p)
                    else:
                        count = min(len(pack), 1 + int(rng.random() * math.sqrt(len(pack)) * strength))
                        pack = ruin(pack, count, mode)
                else:
                    count = min(len(pack), 1 + int(rng.random() * math.sqrt(len(pack)) *
                                                 strength * (1.0 + stalled)))
                    pack = ruin(pack, count, mode)
                # Sometimes keep the contact structure, sometimes open new gaps via consecutive feasible shifts.
                if rng.random() < 0.75:
                    compact(pack, mode, min(deadline, clock() + quantum * 0.15))
            order = order_items(pack, initial)
            fill(pack, order, mode, beta, min(deadline, clock() + quantum))
            commit(pack)
            remember(pack)
            if best_value > previous_best:
                stale_since = clock()
            first = False
            if len(best_pack) == m:
                break
        return answer
    except Exception:
        # Even on unexpected numerical problems, do not lose the already verified complete feasible solution.
        return answer
