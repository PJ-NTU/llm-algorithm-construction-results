"""Incremental contact-graph beam search with adaptive ejection and rebuilding
Shares an incremental tangent placement-point cache, selects layout branches by third-nearest gap and centroid balance, and improves continually with spatial destruction, targeted ejection and annealing acceptance.
"""

import math
import random
import time


# Follows the selection semantics of the two reference algorithms: only check whether circles that are both selected overlap.
# The original squared formula still requires a positive distance when two positive-radius circles are both unselected, which conflicts with these semantics;
# for returned values containing two unselected positive-radius circles, satisfying the formula's extra constraint is not claimed.


def solve(instance: dict, time_limit_s: float) -> dict:
    started = time.perf_counter()
    saved = [[]]
    result = {"packing_status": [], "center_x_coordinates": [],
              "center_y_coordinates": []}
    try:
        n = max(0, int(instance["n"]))
        cx, cy = float(instance["cx"]), float(instance["cy"])
        if not math.isfinite(cx):
            cx = 0.0
        if not math.isfinite(cy):
            cy = 0.0
        result = {"packing_status": [0] * n,
                  "center_x_coordinates": [cx] * n,
                  "center_y_coordinates": [cy] * n}
        raw = [float(r) for r in instance["radii"]]
        radius = float(instance["R"])
        budget = float(time_limit_s)
        if not math.isfinite(budget):
            budget = 0.0
        if n and len(raw) >= n and radius > 0.0 and math.isfinite(radius):
            _search(raw[:n], radius, cx, cy, max(0.0, budget), started, saved)
    except Exception:
        # When the search is interrupted, keep the historical best solution already verified point by point.
        pass
    for p in saved[0]:
        result["packing_status"][p[0]] = 1
        result["center_x_coordinates"][p[0]] = p[3]
        result["center_y_coordinates"][p[0]] = p[4]
    return result


def _search(raw, R, cx, cy, budget, started, saved):
    clock = time.perf_counter
    n = len(raw)
    reserve = min(0.12 * budget, 0.015 * budget + 0.000003 * n)
    deadline = started + budget - reserve
    order = sorted((i for i, r in enumerate(raw)
                    if math.isfinite(r) and 0.0 < r <= R),
                   key=lambda i: -raw[i])
    if not order:
        return
    radii = [r / R if math.isfinite(r) and 0.0 <= r <= R else 0.0
             for r in raw]
    weights = [r * r for r in radii]
    first = order[0]
    best = [(first, 0.0, 0.0, cx, cy)]
    best_area = weights[first]
    saved[0] = best
    if raw[first] == R or len(order) == 1 or clock() >= deadline:
        return

    eps = max(5e-10, 16.0 * math.ulp(cx) / R, 16.0 * math.ulp(cy) / R)
    threshold = eps * 0.35
    strength = math.log1p(budget)
    root_strength = math.sqrt(strength)
    breadth = max(3, int(7.0 + 11.0 * root_strength))
    beam_width = max(2, int(2.0 + 3.0 * root_strength /
                            math.sqrt(1.0 + len(order) / 35.0)))
    elite_cap = max(2, int(2.0 + 2.0 * root_strength))
    rng = random.Random(1785947586)
    sqrt, hypot = math.sqrt, math.hypot
    tau, inf = 2.0 * math.pi, float("inf")
    total_area = sum(weights[i] for i in order)
    last_gain = clock()
    optimal = False
    elites = []

    def remember(pl, area):
        nonlocal best, best_area, last_gain, optimal
        if area > best_area + 1e-15:
            best, best_area = list(pl), area
            saved[0] = best
            last_gain = clock()
        if len(pl) == len(order):
            optimal = True

    def safe_point(idx, x, y, pl, dl):
        wx, wy = cx + R * x, cy + R * y
        if not math.isfinite(wx) or not math.isfinite(wy):
            return None
        dx, dy = wx - cx, wy - cy
        limit = R - raw[idx]
        if dx * dx + dy * dy > limit * limit or hypot(dx, dy) + raw[idx] > R:
            return None
        for j, p in enumerate(pl):
            if j % 64 == 0 and clock() >= dl:
                return None
            dx, dy = wx - p[3], wy - p[4]
            sep = raw[idx] + raw[p[0]]
            if dx * dx + dy * dy < sep * sep or hypot(dx, dy) < sep:
                return None
        return (idx, (wx - cx) / R, (wy - cy) / R, wx, wy)

    def intersections(x, y, a, xx, yy, b):
        dx, dy = xx - x, yy - y
        d2 = dx * dx + dy * dy
        if d2 <= 0.0 or d2 > (a + b) ** 2 or d2 < (a - b) ** 2:
            return ()
        d = sqrt(d2)
        offset = (a * a - b * b + d2) / (2.0 * d)
        h = sqrt(max(0.0, a * a - offset * offset))
        ux, uy = dx / d, dy / d
        mx, my = x + offset * ux, y + offset * uy
        return ((mx - h * uy, my + h * ux),
                (mx + h * uy, my - h * ux))

    def positions(r, pl, start, dl, phase):
        inner = 1.0 - r - eps
        if inner < 0.0:
            return
        if start == 0:
            yield (0.0, 0.0)
            count = max(4, int(3.0 + root_strength))
            for q in range(count):
                angle = phase + tau * q / count
                x, y = inner * math.cos(angle), inner * math.sin(angle)
                yield (x, y)
                if len(pl) < 3:
                    yield (0.55 * x, 0.55 * y)
        for j in range(start, len(pl)):
            if clock() >= dl:
                return
            pj = pl[j]
            dj = radii[pj[0]] + r + eps
            for p in intersections(0.0, 0.0, inner, pj[1], pj[2], dj):
                yield p
            for i in range(j):
                if i % 32 == 0 and clock() >= dl:
                    return
                pi = pl[i]
                di = radii[pi[0]] + r + eps
                for p in intersections(pi[1], pi[2], di, pj[1], pj[2], dj):
                    yield p

    def evaluate(x, y, r, pl, dl):
        g1 = 1.0 - r - hypot(x, y)
        if g1 < threshold:
            return None
        g2, g3 = inf, inf
        for j, p in enumerate(pl):
            if j % 64 == 0 and clock() >= dl:
                return None
            dx, dy = x - p[1], y - p[2]
            rr = r + radii[p[0]]
            d2 = dx * dx + dy * dy
            if d2 < (rr + threshold) ** 2:
                return None
            gap = sqrt(d2) - rr
            if gap < g1:
                g1, g2, g3 = gap, g1, g2
            elif gap < g2:
                g2, g3 = gap, g2
            elif gap < g3:
                g3 = gap
        return (x, y, g1, g2, g3)

    def update_pool(r, pool, pl, dl, phase):
        version, candidates = pool
        k = len(pl)
        if version == k:
            return pool
        valid = []
        # Branches share the old cache but do not modify it in place; only incremental results are added to the new cache.
        for number, c in enumerate(candidates):
            if number % 32 == 0 and clock() >= dl:
                return None
            x, y, g1, g2, g3 = c
            okay = True
            for j in range(version, k):
                if j % 64 == 0 and clock() >= dl:
                    return None
                p = pl[j]
                dx, dy = x - p[1], y - p[2]
                rr = r + radii[p[0]]
                d2 = dx * dx + dy * dy
                if d2 < (rr + threshold) ** 2:
                    okay = False
                    break
                gap = sqrt(d2) - rr
                if gap < g1:
                    g1, g2, g3 = gap, g1, g2
                elif gap < g2:
                    g2, g3 = gap, g2
                elif gap < g3:
                    g3 = gap
            if okay:
                valid.append((x, y, g1, g2, g3))
        for x, y in positions(r, pl, max(0, version), dl, phase):
            if clock() >= dl:
                return None
            c = evaluate(x, y, r, pl, dl)
            if c is not None:
                valid.append(c)
        if clock() >= dl:
            return None
        return (k, valid)

    def state_for(keep):
        used = {p[0] for p in keep}
        groups = {}
        for idx in order:
            if idx not in used:
                groups.setdefault(radii[idx], []).append(idx)
        left = {r: tuple(ids) for r, ids in groups.items()}
        area = sum(weights[p[0]] for p in keep)
        sx = sum(weights[p[0]] * p[1] for p in keep)
        sy = sum(weights[p[0]] * p[2] for p in keep)
        # Layout, area, remaining radius groups, cache, area-centroid numerator, contact quality.
        return (list(keep), area, left, {}, sx, sy, 0.0)

    def expand(state, branch, noise, area_bias, balance_bias, phase, dl):
        pl, area, left, old_pools, sx, sy, quality = state
        pools = dict(old_pools)
        available = list(left)
        cap = min(len(available), max(3, int(breadth /
                  sqrt(1.0 + len(pl) / 24.0))))
        if len(available) > cap:
            head = max(1, 2 * cap // 3)
            trial = available[:head] + rng.sample(available[head:], cap - head)
        else:
            trial = available
        choices = []
        dead = set()
        for r in trial:
            if clock() >= dl:
                break
            pool = update_pool(r, pools.get(r, (-1, [])), pl, dl, phase)
            if pool is None:
                break
            pools[r] = pool
            if not pool[1]:
                dead.add(r)
                continue
            bias = area_bias * r / radii[first]
            local = []
            local_cap = max(1, 3 * branch)
            for number, c in enumerate(pool[1]):
                if number % 128 == 0 and clock() >= dl:
                    break
                g = c[4] if c[4] < inf else (c[3] if c[3] < inf else c[2])
                balance = hypot(sx + r * r * c[0], sy + r * r * c[1]) / (area + r * r)
                merit = 1.0 - g / r - balance_bias * balance
                merit += 0.015 * ((c[2] < 6.0 * eps) + (c[3] < 6.0 * eps))
                score = merit + bias + noise * rng.random()
                item = (score, r, c, merit)
                if len(local) < local_cap or score > local[-1][0]:
                    local.append(item)
                    local.sort(key=lambda z: z[0], reverse=True)
                    if len(local) > local_cap:
                        local.pop()
            chosen_xy = []
            for item in local:
                x, y = item[2][:2]
                if any((x - xx) ** 2 + (y - yy) ** 2 < (0.18 * r) ** 2
                       for xx, yy in chosen_xy):
                    continue
                choices.append(item)
                chosen_xy.append((x, y))
                if len(chosen_xy) >= branch:
                    break
        # Only a radius group that has been fully checked can be judged to have no usable placement point.
        remaining = {r: ids for r, ids in left.items() if r not in dead}
        for r in dead:
            pools.pop(r, None)
        choices.sort(key=lambda z: z[0], reverse=True)
        children = []
        for _, r, c, merit in choices:
            if clock() >= dl:
                break
            idx = remaining[r][-1]
            p = safe_point(idx, c[0], c[1], pl, dl)
            if p is None:
                continue
            newleft = dict(remaining)
            if len(newleft[r]) == 1:
                del newleft[r]
            else:
                newleft[r] = newleft[r][:-1]
            child = (pl + [p], area + weights[idx], newleft, dict(pools),
                     sx + weights[idx] * p[1], sy + weights[idx] * p[2],
                     quality + merit)
            children.append(child)
            remember(child[0], child[1])
            if len(children) >= branch or optimal:
                break
        # When candidate sampling does not cover all remaining groups, keep the other groups for further attempts.
        retry = None
        if not children and dead and remaining and clock() < dl:
            retry = (pl, area, remaining, pools, sx, sy, quality)
        return children, retry

    def rebuild(keep, width, noise, area_bias, balance_bias, phase, dl):
        state = state_for(keep)
        states = [state]
        local_best, local_area = state[0], state[1]
        remember(local_best, local_area)
        while states and clock() < dl and not optimal:
            pool = []
            for s in states:
                children, retry = expand(s, 1 if width == 1 else
                                         max(2, int(sqrt(width) + 1)),
                                         noise, area_bias, balance_bias, phase, dl)
                for child in children:
                    if child[1] > local_area:
                        local_best, local_area = child[0], child[1]
                    pool.append(child)
                if retry is not None:
                    pool.append(retry)
                if clock() >= dl or optimal:
                    break
            if not pool:
                break
            # Area comes first; contact quality keeps alternative layouts with different local geometry.
            pool.sort(key=lambda s: s[1] + 0.006 * min(1.0, total_area) *
                      math.tanh(s[6] / max(1, len(s[0]) - len(keep))), reverse=True)
            states, signatures = [], {}
            for s in pool:
                signature = tuple(sorted(p[0] for p in s[0]))
                count = signatures.get(signature, 0)
                if count < max(2, (width + 1) // 2):
                    states.append(s)
                    signatures[signature] = count + 1
                if len(states) >= width:
                    break
        return local_best, local_area

    def add_elite(pl, area):
        if area < 0.94 * best_area or not pl:
            return
        # For the same selection set, keep layouts with different spatial structure.
        signature = tuple(sorted((p[0], round(hypot(p[1], p[2]), 3)) for p in pl))
        if any(e[2] == signature for e in elites):
            return
        elites.append((list(pl), area, signature))
        elites.sort(key=lambda e: e[1], reverse=True)
        del elites[elite_cap:]

    def eject(base, dl):
        used = {p[0] for p in base}
        missing = [i for i in order if i not in used]
        if not missing:
            return None
        # Probe circles with larger area first, while giving other radii a chance to be tried.
        if rng.random() < 0.6:
            missing.sort(key=lambda i: rng.random() / max(weights[i], 1e-30))
        move, best_gain = None, -inf
        seen_radii = set()
        for idx in missing:
            if clock() >= dl:
                break
            r = radii[idx]
            if r in seen_radii:
                continue
            seen_radii.add(r)
            inner = 1.0 - r - eps
            if inner < 0.0:
                continue
            best_cost, best_xy, best_gone = inf, None, None

            def consider(x, y):
                nonlocal best_cost, best_xy, best_gone
                if x * x + y * y > inner * inner:
                    return
                gone, cost = [], 0.0
                for j, p in enumerate(base):
                    if j % 64 == 0 and clock() >= dl:
                        return
                    dx, dy = x - p[1], y - p[2]
                    sep = r + radii[p[0]] + threshold
                    if dx * dx + dy * dy < sep * sep:
                        gone.append(j)
                        cost += weights[p[0]]
                        if cost >= best_cost:
                            return
                if cost < best_cost:
                    best_cost, best_xy, best_gone = cost, (x, y), gone

            for p in base:
                if clock() >= dl:
                    break
                consider(p[1], p[2])
            for x, y in positions(r, base, 0, dl, rng.random() * tau):
                if clock() >= dl or best_cost == 0.0:
                    break
                consider(x, y)
            if best_xy is not None and weights[idx] - best_cost > best_gain:
                removed = set(best_gone)
                keep = [p for j, p in enumerate(base) if j not in removed]
                p = safe_point(idx, best_xy[0], best_xy[1], keep, deadline)
                if p is not None:
                    best_gain = weights[idx] - best_cost
                    move = keep + [p]
                    remember(move, sum(weights[p[0]] for p in move))
            if best_gain > 0.0:
                break
        return move

    # First build a cheap complete layout; later wide beams and rebuilds share exactly the same cache engine.
    initial_dl = min(deadline, clock() + max(0.0, deadline - clock()) /
                     (2.0 + root_strength))
    current, current_area = rebuild([], 1, 0.0, 0.0, 0.02, 0.0, initial_dl)
    if current_area < best_area:
        current, current_area = list(best), best_area
    add_elite(current, current_area)
    operator_weights = [1.2, 0.8, 0.8, 1.5, 0.45]

    while clock() < deadline and not optimal:
        now = clock()
        remaining = max(0.0, deadline - now)
        stall = min(1.0, (now - last_gain) / max(budget * 0.18, 1e-12))
        draw = rng.random()
        if draw < 0.45:
            base = best
        elif draw < 0.7 and elites:
            base = rng.choice(elites)[0]
        else:
            base = current
        nb = len(base)
        depth = min(nb, max(1, int((1.0 + root_strength) *
                    sqrt(max(nb, 1)) * (0.45 + stall))))
        q = min(nb, 1 + int(max(0, depth - 1) * rng.random() ** 2))
        pick = rng.random() * sum(operator_weights)
        op = len(operator_weights) - 1
        for j, weight in enumerate(operator_weights):
            pick -= weight
            if pick <= 0.0:
                op = j
                break
        keep = None
        slice_length = min(remaining, (deadline - started) /
                           (3.0 + sqrt(max(len(order), 1)) + root_strength))
        before = best_area
        if op == 3:
            keep = eject(base, min(deadline, clock() + slice_length * 0.35))
        if keep is None:
            if not nb or op == 4:
                keep = []
            elif op == 1:
                angle = rng.random() * tau
                ranked = sorted(range(nb), key=lambda j:
                                (math.atan2(base[j][2], base[j][1]) - angle) % tau)
                gone = set(ranked[:q])
                keep = [p for j, p in enumerate(base) if j not in gone]
            elif op == 2:
                gone = set(rng.sample(range(nb), q))
                keep = [p for j, p in enumerate(base) if j not in gone]
            else:
                center = rng.choice(base)
                ranked = sorted(range(nb), key=lambda j:
                                hypot(base[j][1] - center[1], base[j][2] - center[2])
                                - radii[base[j][0]])
                gone = set(ranked[:q])
                keep = [p for j, p in enumerate(base) if j not in gone]
        noise = (0.02 + 0.055 * root_strength) * rng.random() ** 2
        if rng.random() < 0.25:
            noise = 0.0
        draw = rng.random()
        if draw < 0.55:
            area_bias = 0.0
        elif draw < 0.88:
            area_bias = (0.04 + 0.12 * root_strength) * rng.random()
        else:
            area_bias = 1.0 + (1.0 + root_strength) * rng.random()
        balance_bias = 0.13 * rng.random() ** 2
        width = 1
        if rng.random() < (0.2 + 0.2 * stall) and remaining > slice_length:
            width = max(2, int(beam_width * (0.5 + 0.5 * rng.random())))
        local_dl = min(deadline, clock() + slice_length * sqrt(width))
        new, new_area = rebuild(keep, width, noise, area_bias, balance_bias,
                                rng.random() * tau, local_dl)
        add_elite(new, new_area)
        gain = new_area - current_area
        temperature = (0.018 * best_area * min(1.0, sqrt(budget / max(n, 1))) *
                       (0.1 + (remaining / max(budget, 1e-12)) ** 2) * (1.0 + stall))
        accepted = gain >= 0.0
        if not accepted and temperature > 0.0:
            accepted = rng.random() < math.exp(max(-700.0, gain / temperature))
        if accepted:
            current, current_area = new, new_area
        reward = 5.0 if best_area > before else (1.2 if gain >= 0.0 else 0.3)
        operator_weights[op] = 0.85 * operator_weights[op] + 0.15 * reward
        if stall >= 1.0 and rng.random() < 0.2:
            current, current_area = list(best), best_area
