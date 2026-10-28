"""Blocking-witness cache with multi-target ejection search
Reuses tangent placement points and blocking witnesses across neighbourhoods, fills holes by third-nearest gap, and improves continually with net-area ejection, local destruction and annealing acceptance.
"""

import math
import random
import time


# Consistent with the two provided reference codes, non-overlap is checked only for circles that are both selected.
# The squared formula in the problem statement would forbid two positive-radius circles from both being unselected, and could make the whole problem infeasible;
# this implementation does not claim to satisfy the extra constraint that the formula imposes when both circles are unselected.


def solve(instance: dict, time_limit_s: float) -> dict:
    started = time.perf_counter()
    n, cx, cy = 0, 0.0, 0.0
    saved = [[]]
    result = {"packing_status": [], "center_x_coordinates": [],
              "center_y_coordinates": []}
    try:
        n = max(0, int(instance.get("n", len(instance.get("radii", [])))))
        cx = float(instance.get("cx", 0.0))
        cy = float(instance.get("cy", 0.0))
        if not math.isfinite(cx):
            cx = 0.0
        if not math.isfinite(cy):
            cy = 0.0
        result = {"packing_status": [0] * n,
                  "center_x_coordinates": [cx] * n,
                  "center_y_coordinates": [cy] * n}
        raw = [float(v) for v in instance["radii"]]
        R = float(instance["R"])
        budget = float(time_limit_s)
        if not math.isfinite(budget):
            budget = 0.0
        if (n and len(raw) >= n and R > 0.0 and math.isfinite(R)):
            _search(raw[:n], R, cx, cy, max(0.0, budget), started, saved)
    except Exception:
        # A search exception never overwrites the best layout previously verified and saved.
        pass
    for p in saved[0]:
        result["packing_status"][p[0]] = 1
        result["center_x_coordinates"][p[0]] = p[3]
        result["center_y_coordinates"][p[0]] = p[4]
    return result


def _search(raw, R, cx, cy, budget, started, saved):
    clock = time.perf_counter
    n = len(raw)
    reserve = min(0.15 * budget, 0.02 * budget + 0.000002 * n)
    deadline = started + budget - reserve
    order = [i for i, r in enumerate(raw)
             if math.isfinite(r) and 0.0 < r <= R]
    if not order:
        return
    order.sort(key=lambda i: -raw[i])
    radii = [r / R if math.isfinite(r) else 0.0 for r in raw]
    weights = [r * r for r in radii]
    first = order[0]
    saved[0] = [(first, 0.0, 0.0, cx, cy)]
    best = saved[0]
    best_area = weights[first]
    if raw[first] == R or len(order) == 1 or clock() >= deadline:
        return

    # Normalisation removes the container scale; the margin also covers rounding error when translating to actual coordinates.
    eps = max(3e-10, 16.0 * math.ulp(cx) / R,
              16.0 * math.ulp(cy) / R)
    threshold = 0.4 * eps
    strength = math.log1p(budget)
    breadth = max(3, int(8.0 + 10.0 * math.sqrt(strength)))
    # Only limits cache memory, not the number of search rounds; after eviction, entries are fully rebuilt on demand.
    cache_limit = max(256, int(2048 * breadth * (1.0 + math.sqrt(strength))))
    cache = {}
    cache_size = 0
    rng = random.Random(20261003)
    sqrt = math.sqrt
    inf = float("inf")
    tau = 2.0 * math.pi
    last_gain = clock()
    optimal = False

    def remember(pl, area):
        nonlocal best, best_area, last_gain, optimal
        if area > best_area:
            best = list(pl)
            best_area = area
            saved[0] = best
            last_gain = clock()
        if len(pl) == len(order):
            optimal = True

    def safe_point(idx, x, y, pl):
        # Verify at the coordinate scale of the final output, so any sub-layout kept afterwards remains feasible.
        wx, wy = cx + R * x, cy + R * y
        if not math.isfinite(wx) or not math.isfinite(wy):
            return None
        dx, dy = wx - cx, wy - cy
        lim = R - raw[idx]
        if dx * dx + dy * dy > lim * lim:
            return None
        if math.hypot(dx, dy) + raw[idx] > R:
            return None
        for j, p in enumerate(pl):
            if j % 64 == 0 and clock() >= deadline:
                return None
            dx, dy = wx - p[3], wy - p[4]
            need = raw[idx] + raw[p[0]]
            if dx * dx + dy * dy < need * need:
                return None
            if math.hypot(dx, dy) < need:
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

    def positions(r, pl, added, dl, phase, initial):
        inner = 1.0 - r - eps
        if inner < 0.0:
            return
        if initial:
            yield (0.0, 0.0, -1, -1)
            for a in (phase, phase + 0.5 * math.pi,
                      phase + math.pi, phase + 1.5 * math.pi):
                yield (inner * math.cos(a), inner * math.sin(a), -1, -1)
        # Contact points between kept circles are already cached; only generate contact pairs involving at least one newly added circle.
        earlier = []
        for p in pl:
            if p[0] not in added:
                earlier.append(p)
        for pj in pl:
            if pj[0] not in added:
                continue
            if clock() >= dl:
                return
            dj = radii[pj[0]] + r + eps
            for x, y in intersections(0.0, 0.0, inner, pj[1], pj[2], dj):
                yield (x, y, pj[0], -1)
            for number, pi in enumerate(earlier):
                if number % 32 == 0 and clock() >= dl:
                    return
                di = radii[pi[0]] + r + eps
                for x, y in intersections(pi[1], pi[2], di, pj[1], pj[2], dj):
                    yield (x, y, pi[0], pj[0])
            earlier.append(pj)

    def evaluate(x, y, r, pl, s1, s2, dl):
        g1 = 1.0 - r - math.hypot(x, y)
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
                # A single blocking circle still in place is enough to prove the placement point infeasible.
                return (x, y, 0.0, 0.0, 0.0, p[0], s1, s2)
            # Beyond the third-nearest gap, no square root needs to be computed.
            if g3 < inf and d2 >= (rr + g3) ** 2:
                continue
            gap = sqrt(d2) - rr
            if gap < g1:
                g1, g2, g3 = gap, g1, g2
            elif gap < g2:
                g2, g3 = gap, g2
            elif gap < g3:
                g3 = gap
        return (x, y, g1, g2, g3, -1, s1, s2)

    def update_pool(r, pl, phase, dl):
        nonlocal cache_size
        state = {p[0]: p for p in pl}
        old = cache.get(r)
        if old is None:
            previous, candidates = {}, []
        else:
            previous, candidates = old
        removed = {i for i, p in previous.items() if state.get(i) != p}
        added = {i for i, p in state.items() if previous.get(i) != p}
        if old is not None and not removed and not added:
            # Refresh the most-recently-used order.
            cache.pop(r)
            cache[r] = old
            return candidates
        extra = [p for p in pl if p[0] in added]
        updated = []
        for number, c in enumerate(candidates):
            if number % 32 == 0 and clock() >= dl:
                return None
            x, y, g1, g2, g3, blocker, s1, s2 = c
            if s1 in removed or s2 in removed:
                continue
            if blocker >= 0 and blocker not in removed:
                updated.append(c)
                continue
            if removed:
                # Blocking witness invalidated or neighbour removed: recompute feasibility and the exact third-nearest gap.
                cc = evaluate(x, y, r, pl, s1, s2, dl)
                if cc is not None:
                    updated.append(cc)
                continue
            for j, p in enumerate(extra):
                if j % 64 == 0 and clock() >= dl:
                    return None
                dx, dy = x - p[1], y - p[2]
                rr = r + radii[p[0]]
                d2 = dx * dx + dy * dy
                if d2 < (rr + threshold) ** 2:
                    blocker = p[0]
                    break
                if g3 < inf and d2 >= (rr + g3) ** 2:
                    continue
                gap = sqrt(d2) - rr
                if gap < g1:
                    g1, g2, g3 = gap, g1, g2
                elif gap < g2:
                    g2, g3 = gap, g2
                elif gap < g3:
                    g3 = gap
            updated.append((x, y, g1, g2, g3, blocker, s1, s2))
        for x, y, s1, s2 in positions(r, pl, added, dl, phase, old is None):
            if clock() >= dl:
                return None
            c = evaluate(x, y, r, pl, s1, s2, dl)
            if c is not None:
                updated.append(c)
        if clock() >= dl:
            return None
        # The cache is committed only after a full update; an exhausted local time slice must not leave half-updated geometric state.
        if old is not None:
            cache_size -= len(old[1])
            cache.pop(r)
        cache[r] = (state, updated)
        cache_size += len(updated)
        while cache_size > cache_limit and cache:
            key = next(iter(cache))
            cache_size -= len(cache.pop(key)[1])
        return updated

    def fill(keep, noise, area_weight, phase, dl):
        pl = list(keep)
        used = {p[0] for p in pl}
        groups = {}
        for idx in order:
            if idx not in used:
                groups.setdefault(radii[idx], []).append(idx)
        live = list(groups)
        area = sum(weights[p[0]] for p in pl)
        remember(pl, area)
        while live and clock() < dl and not optimal:
            cap = min(len(live), max(3, int(breadth / sqrt(1.0 + len(pl) / 24.0))))
            if len(live) <= cap:
                trial = list(live)
            else:
                head = max(1, (2 * cap) // 3)
                trial = live[:head] + rng.sample(live[head:], cap - head)
            chosen = None
            top = -inf
            dead = set()
            for r in trial:
                if clock() >= dl:
                    break
                pool = update_pool(r, pl, phase, dl)
                if pool is None:
                    break
                any_valid = False
                bias = area_weight * r / radii[first]
                for number, c in enumerate(pool):
                    if number % 64 == 0 and clock() >= dl:
                        break
                    if c[5] >= 0:
                        continue
                    any_valid = True
                    g = c[4] if c[4] < inf else (c[3] if c[3] < inf else c[2])
                    score = 1.0 - g / r + bias
                    if noise:
                        score += noise * rng.random()
                    if score > top:
                        top, chosen = score, (r, c)
                if not any_valid:
                    dead.add(r)
            if dead:
                live = [r for r in live if r not in dead]
            if chosen is None:
                if clock() >= dl or not dead:
                    break
                continue
            # Even if the candidate-comparison time slice runs out, keep an insertion already found that passed the strict check.
            r, c = chosen
            idx = groups[r][-1]
            p = safe_point(idx, c[0], c[1], pl)
            if p is None:
                cached = cache.get(r)
                if cached is not None and c in cached[1]:
                    cached[1].remove(c)
                    # Overestimating cache_size only causes conservative eviction and does not affect correctness.
                else:
                    dead.add(r)
                    live = [v for v in live if v not in dead]
                continue
            pl.append(p)
            area += weights[idx]
            groups[r].pop()
            if not groups[r]:
                live.remove(r)
            remember(pl, area)
        return pl, area

    def eject(base, dl):
        used = {p[0] for p in base}
        missing = []
        seen_radii = set()
        for i in order:
            if i not in used and radii[i] not in seen_radii:
                seen_radii.add(radii[i])
                missing.append(i)
        if not missing or clock() >= dl:
            return None
        width = min(len(missing), max(2, int((1.0 + sqrt(strength)) *
                                             sqrt(len(missing)))))
        # Mix the largest circles to pack with stratified sampling, to avoid trying only one random target.
        h = max(1, width // 2)
        targets = missing[:h]
        if width > h:
            targets += rng.sample(missing[h:], width - h)
        rng.shuffle(targets)
        best_gain, best_xy, best_gone, best_idx = -inf, None, None, None
        for number, idx in enumerate(targets):
            now = clock()
            if now >= dl:
                break
            sub_dl = min(dl, now + (dl - now) / max(1, len(targets) - number))
            r = radii[idx]
            inner = 1.0 - r - eps
            if inner < 0.0:
                continue
            own_weight = weights[idx]

            def consider(x, y):
                nonlocal best_gain, best_xy, best_gone, best_idx
                if x * x + y * y > inner * inner:
                    return
                gone, cost = [], 0.0
                for j, p in enumerate(base):
                    if j % 64 == 0 and clock() >= sub_dl:
                        return
                    dx, dy = x - p[1], y - p[2]
                    rr = r + radii[p[0]] + threshold
                    if dx * dx + dy * dy < rr * rr:
                        gone.append(j)
                        cost += weights[p[0]]
                        if own_weight - cost < best_gain:
                            return
                # Swapping indices in place between equal-radius circles does not change the layout and does not occupy the ejection neighbourhood.
                if len(gone) == 1:
                    p = base[gone[0]]
                    if radii[p[0]] == r and math.hypot(x - p[1], y - p[2]) <= eps:
                        return
                gain = own_weight - cost
                if gain > best_gain or (gain == best_gain and rng.random() < 0.2):
                    best_gain, best_xy, best_gone, best_idx = gain, (x, y), gone, idx

            # Examine cheap seeds first, so replacement suggestions can be formed within a short time slice even without a cache.
            consider(0.0, 0.0)
            for p in base:
                if clock() >= sub_dl:
                    break
                consider(p[1], p[2])
            pool = update_pool(r, base, rng.random() * tau, sub_dl)
            if pool is None:
                continue
            for c in pool:
                if clock() >= sub_dl:
                    break
                # The blocking witness gives a loss lower bound; skip directly when it is not good enough.
                if c[5] >= 0 and own_weight - weights[c[5]] < best_gain:
                    continue
                consider(c[0], c[1])
        if best_xy is None or clock() >= deadline:
            return None
        removed = set(best_gone)
        keep = [p for j, p in enumerate(base) if j not in removed]
        p = safe_point(best_idx, best_xy[0], best_xy[1], keep)
        if p is None:
            return None
        keep.append(p)
        remember(keep, sum(weights[p[0]] for p in keep))
        return keep

    current, current_area = fill([], 0.0, 0.0, 0.0, deadline)
    if current_area < best_area:
        current, current_area = list(best), best_area
    operator_weights = [1.2, 0.8, 0.8, 1.6, 0.3]

    # No fixed iteration cap; candidate width, neighbourhood depth, noise and local time slices all scale with the budget.
    while clock() < deadline and not optimal:
        now = clock()
        before = best_area
        remaining = max(0.0, deadline - now)
        stall = min(1.0, (now - last_gain) / max(budget * 0.2, 1e-12))
        base = best if rng.random() < 0.5 else current
        nb = len(base)
        depth = min(nb, max(1, int((1.0 + sqrt(strength)) * sqrt(max(nb, 1)) *
                                    (0.5 + stall))))
        q = min(nb, 1 + int((depth - 1) * rng.random() ** 2))
        draw = rng.random() * sum(operator_weights)
        op = len(operator_weights) - 1
        for j, weight in enumerate(operator_weights):
            draw -= weight
            if draw <= 0.0:
                op = j
                break
        keep = None
        if op == 3:
            local_deadline = min(deadline, now + remaining /
                                 (4.0 + sqrt(max(n, 1)) + strength))
            keep = eject(base, local_deadline)
        if keep is None:
            if not nb or op == 4:
                keep = []
            elif op == 1:
                start_angle = rng.random() * tau
                ranked = sorted(range(nb), key=lambda j:
                                (math.atan2(base[j][2], base[j][1]) - start_angle) % tau)
                gone = set(ranked[:q])
                keep = [p for j, p in enumerate(base) if j not in gone]
            elif op == 2:
                gone = set(rng.sample(range(nb), q))
                keep = [p for j, p in enumerate(base) if j not in gone]
            else:
                center = base[rng.randrange(nb)]
                ranked = sorted(range(nb), key=lambda j:
                                math.hypot(base[j][1] - center[1],
                                           base[j][2] - center[2]) - radii[base[j][0]])
                gone = set(ranked[:q])
                keep = [p for j, p in enumerate(base) if j not in gone]
        noise = (0.025 + 0.055 * sqrt(strength)) * rng.random() ** 2
        if rng.random() < 0.25:
            noise = 0.0
        area_weight = (0.03 + 0.07 * sqrt(strength)) * rng.random() ** 2
        if rng.random() < 0.5:
            area_weight = 0.0
        new, new_area = fill(keep, noise, area_weight, rng.random() * tau, deadline)
        gain = new_area - current_area
        temperature = (0.02 * best_area * min(1.0, sqrt(budget / max(n, 1))) *
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
