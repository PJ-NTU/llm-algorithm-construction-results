"""Incremental hole filling with elastic circle-enlargement search
Combines tangent-point caching, ruin-and-recreate and targeted ejection, tries inserting circles and upgrading circle radii via time-limited quasi-Newton displacement, and verifies point by point to save the best layout.
"""

import math
import random
import time

try:
    import numpy as np
except Exception:
    np = None


class _SliceEnd(Exception):
    pass


# Semantic restriction: consistent with the provided reference algorithms, non-overlap is checked only for circles that are both selected.
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
              16.0 * math.ulp(cy) / R, 16.0 * math.ulp(R) / R)
    threshold = 0.4 * eps
    strength = math.log1p(budget)
    breadth = max(3, int(8.0 + 10.0 * math.sqrt(strength)))
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

    def positions(r, pl, start, dl, phase):
        inner = 1.0 - r - eps
        if inner < 0.0:
            return
        if start == 0:
            yield (0.0, 0.0)
            # Degenerate cases: no contacts, concentric circles, or few circles.
            for a in (phase, phase + 0.5 * math.pi,
                      phase + math.pi, phase + 1.5 * math.pi):
                yield (inner * math.cos(a), inner * math.sin(a))
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
                for p in intersections(pi[1], pi[2], di,
                                       pj[1], pj[2], dj):
                    yield p

    def evaluate(x, y, r, pl):
        g1 = 1.0 - r - math.hypot(x, y)
        if g1 < threshold:
            return None
        g2, g3 = inf, inf
        for j, p in enumerate(pl):
            if j % 64 == 0 and clock() >= deadline:
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

    def update_pool(r, pool, pl, phase):
        old_version, candidates = pool
        k = len(pl)
        if old_version == k:
            return pool
        valid = []
        # Old placement points only need to be checked against the newly placed circle; the third-nearest gap is also updated incrementally.
        for number, c in enumerate(candidates):
            if number % 32 == 0 and clock() >= deadline:
                return (k, valid)
            x, y, g1, g2, g3 = c
            okay = True
            for j in range(old_version, k):
                if j % 64 == 0 and clock() >= deadline:
                    return (k, valid)
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
        # Only generate contact pairs involving the new circle, instead of re-enumerating all old contact pairs.
        for x, y in positions(r, pl, max(0, old_version), deadline, phase):
            if clock() >= deadline:
                break
            c = evaluate(x, y, r, pl)
            if c is not None:
                valid.append(c)
        return (k, valid)

    def fill(keep, noise, area_weight, phase):
        pl = list(keep)
        used = {p[0] for p in pl}
        groups = {}
        for idx in order:
            if idx not in used:
                groups.setdefault(radii[idx], []).append(idx)
        live = list(groups)
        pools = {}
        area = sum(weights[p[0]] for p in pl)
        remember(pl, area)
        while live and clock() < deadline and not optimal:
            # The time budget widens the candidate width; the larger the layout, the narrower each single comparison.
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
                if clock() >= deadline:
                    break
                # -1 denotes the initial placement point of an empty layout not yet generated.
                pool = update_pool(r, pools.get(r, (-1, [])), pl, phase)
                pools[r] = pool
                if clock() >= deadline:
                    break
                if not pool[1]:
                    dead.add(r)
                    continue
                bias = area_weight * r / radii[first]
                for number, c in enumerate(pool[1]):
                    if number % 128 == 0 and clock() >= deadline:
                        break
                    g = c[4] if c[4] < inf else (c[3] if c[3] < inf else c[2])
                    score = 1.0 - g / r + bias
                    if noise:
                        score += noise * rng.random()
                    if score > top:
                        top, chosen = score, (r, c)
            if dead:
                live = [r for r in live if r not in dead]
                for r in dead:
                    pools.pop(r, None)
            if chosen is None or clock() >= deadline:
                if clock() >= deadline or not dead:
                    break
                continue
            r, c = chosen
            idx = groups[r][-1]
            p = safe_point(idx, c[0], c[1], pl)
            if p is None:
                pools[r][1].remove(c)
                continue
            pl.append(p)
            area += weights[idx]
            groups[r].pop()
            if not groups[r]:
                live.remove(r)
                pools.pop(r, None)
            remember(pl, area)
        return pl, area

    def eject(base, dl):
        used = {p[0] for p in base}
        missing = [i for i in order if i not in used]
        if not missing:
            return None
        # Favour larger unpacked circles while giving every unpacked circle a chance to be tried.
        idx = missing[int(rng.random() ** (1.0 + sqrt(strength)) * len(missing))]
        r = radii[idx]
        inner = 1.0 - r - eps
        if inner < 0.0:
            return None
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
                rr = r + radii[p[0]] + threshold
                if dx * dx + dy * dy < rr * rr:
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
        if best_xy is None or clock() >= deadline:
            return None
        removed = set(best_gone)
        keep = [p for j, p in enumerate(base) if j not in removed]
        p = safe_point(idx, best_xy[0], best_xy[1], keep)
        if p is None:
            return None
        keep.append(p)
        return keep


    def elastic(base, dl, exchange):
        """Try adding a circle, or replacing a packed circle with a slightly larger one, then move the local circle cluster."""
        if np is None or not base or clock() >= dl:
            return None
        used = {p[0] for p in base}
        # Equal-radius circles are equivalent; generate one target per radius to avoid retrying the same geometry.
        missing = []
        seen = set()
        base_area = math.fsum(weights[p[0]] for p in base)
        for idx in order:
            r = radii[idx]
            if idx not in used and r not in seen:
                seen.add(r)
                missing.append(idx)
        if not missing:
            return None
        dropped = None
        if exchange:
            upgrades = []
            by_radius = sorted(base, key=lambda p: radii[p[0]])
            # Prefer upgrades with smaller relative enlargement, without excluding harder upgrades from the search.
            for idx in missing:
                if clock() >= dl:
                    return None
                rt = radii[idx]
                eligible = [p for p in by_radius if radii[p[0]] < rt]
                if not eligible:
                    continue
                old_radius = radii[eligible[-1][0]]
                gain = weights[idx] - old_radius * old_radius
                if base_area + gain > 1.0 + 1e-12:
                    continue
                ties = [p for p in eligible if radii[p[0]] == old_radius]
                old = rng.choice(ties)
                upgrades.append(((rt - old_radius) / rt, idx, old))
            if not upgrades:
                return None
            upgrades.sort(key=lambda z: z[0])
            chosen = upgrades[int(rng.random() ** (1.0 + sqrt(strength)) * len(upgrades))]
            _, idx, dropped = chosen
            tx, ty = dropped[1], dropped[2]
            seed = [p for p in base if p[0] != dropped[0]]
            limit = max(0.0, 1.0 - radii[idx] - 8.0 * eps)
            dist = math.hypot(tx, ty)
            if dist > limit and dist > 0.0:
                tx, ty = tx * limit / dist, ty * limit / dist
        else:
            available = [i for i in missing if base_area + weights[i] <= 1.0 + 1e-12]
            if not available:
                return None
            idx = available[int(rng.random() ** (1.0 + sqrt(strength)) * len(available))]
            seed = list(base)
            rt = radii[idx]
            limit = 1.0 - rt - eps
            if limit < 0.0:
                return None
            # First use the existing contact geometry to find a position with small total penetration, rather than directly removing obstacle circles.
            seed_end = min(dl, clock() + (dl - clock()) / (3.0 + sqrt(strength)))
            best_penalty = inf
            best_xy = None

            def consider(tx, ty):
                nonlocal best_penalty, best_xy
                if tx * tx + ty * ty > limit * limit + eps * eps:
                    return
                penalty = 0.0
                for j, p in enumerate(seed):
                    if j % 64 == 0 and clock() >= seed_end:
                        return
                    dx, dy = tx - p[1], ty - p[2]
                    need = rt + radii[p[0]] + 8.0 * eps
                    dsq = dx * dx + dy * dy
                    if dsq < need * need:
                        overlap = need - sqrt(dsq)
                        penalty += overlap * overlap
                        if penalty >= best_penalty:
                            return
                if penalty < best_penalty:
                    best_penalty, best_xy = penalty, (tx, ty)

            phase = rng.random() * tau
            for tx, ty in positions(rt, seed, 0, seed_end, phase):
                if clock() >= seed_end:
                    break
                consider(tx, ty)
                if best_penalty == 0.0:
                    break
            if best_xy is None:
                return None
            tx, ty = best_xy
            if best_penalty == 0.0:
                p = safe_point(idx, tx, ty, seed)
                if p is not None:
                    answer = seed + [p]
                    remember(answer, base_area + weights[idx])
                    return answer
            # Avoid the zero-gradient degeneracy of the distance function when two circle centres coincide exactly.
            if any(math.hypot(tx - p[1], ty - p[2]) < eps for p in seed):
                tx += max(16.0 * eps, radii[idx] * 1e-5)

        if clock() >= dl:
            return None
        rt = radii[idx]
        # Only move circles near the target; the distant layout serves as fixed obstacles, controlling the size of each gradient step.
        # Depth and matrix workload are both determined by the total budget and the current number of packed circles; the number of searches is not limited.
        size = len(seed)
        depth = max(4, int((2.0 + sqrt(strength)) * sqrt(max(1, size))))
        work_cap = max(1, int(5000.0 * (1.0 + strength) / max(1, size)))
        depth = min(size, depth, work_cap)
        near = sorted(seed, key=lambda p: math.hypot(p[1] - tx, p[2] - ty) - radii[p[0]])
        movable = near[:depth]
        fixed = near[depth:]
        ids = [p[0] for p in movable] + [idx]
        m = len(ids)
        z = np.array([(p[1], p[2]) for p in movable] + [(tx, ty)], dtype=float).reshape(-1)
        rr = np.array([radii[i] for i in ids], dtype=float)
        wall = 1.0 - rr - 8.0 * eps
        a, b = np.triu_indices(m, 1)
        separation = rr[a] + rr[b] + 8.0 * eps
        fxy = np.array([(p[1], p[2]) for p in fixed], dtype=float).reshape((-1, 2))
        fr = np.array([radii[p[0]] for p in fixed], dtype=float)
        fixed_separation = rr[:, None] + fr[None, :] + 8.0 * eps
        # Reserve part of the current time slice for converting continuous coordinates and strictly verifying them.
        local_end = clock() + max(0.0, dl - clock()) * 0.9

        def energy(v):
            if clock() >= local_end:
                raise _SliceEnd()
            xy = v.reshape((m, 2))
            dist = np.sqrt(np.sum(xy * xy, axis=1))
            penetration = np.maximum(dist - wall, 0.0)
            value = float(np.dot(penetration, penetration))
            peak = float(np.max(penetration))
            grad = (2.0 * penetration / np.maximum(dist, 1e-30))[:, None] * xy
            difference = xy[a] - xy[b]
            dist = np.sqrt(np.sum(difference * difference, axis=1))
            penetration = np.maximum(separation - dist, 0.0)
            if len(penetration):
                value += float(np.dot(penetration, penetration))
                peak = max(peak, float(np.max(penetration)))
                force = (-2.0 * penetration / np.maximum(dist, 1e-30))[:, None] * difference
                grad[:, 0] += (np.bincount(a, weights=force[:, 0], minlength=m)
                               - np.bincount(b, weights=force[:, 0], minlength=m))
                grad[:, 1] += (np.bincount(a, weights=force[:, 1], minlength=m)
                               - np.bincount(b, weights=force[:, 1], minlength=m))
            if len(fixed):
                difference = xy[:, None, :] - fxy[None, :, :]
                dist = np.sqrt(np.sum(difference * difference, axis=2))
                penetration = np.maximum(fixed_separation - dist, 0.0)
                value += float(np.sum(penetration * penetration))
                peak = max(peak, float(np.max(penetration)))
                force = -2.0 * penetration / np.maximum(dist, 1e-30)
                grad += np.sum(force[:, :, None] * difference, axis=1)
            return value, grad.reshape(-1), peak

        # Hand-written time-limited L-BFGS: analytic gradient, two-loop recursion and backtracking line search, with no fixed iteration cap.
        history = []
        memory = max(3, int(3.0 + 2.0 * sqrt(strength)))
        try:
            value, gradient, peak = energy(z)
            while clock() < local_end and peak > 2.0 * eps:
                if not math.isfinite(value):
                    return None
                direction = gradient.copy()
                coefficients = []
                for s, y, inv_sy in reversed(history):
                    coefficient = inv_sy * float(np.dot(s, direction))
                    coefficients.append(coefficient)
                    direction -= coefficient * y
                scale = 0.25
                if history:
                    s, y, inv_sy = history[-1]
                    scale = 1.0 / max(1e-30, inv_sy * float(np.dot(y, y)))
                direction *= scale
                for (s, y, inv_sy), coefficient in zip(history, reversed(coefficients)):
                    direction += s * (coefficient - inv_sy * float(np.dot(y, direction)))
                direction = -direction
                slope = float(np.dot(gradient, direction))
                if slope >= 0.0 or not math.isfinite(slope):
                    history.clear()
                    direction = -gradient
                    slope = -float(np.dot(gradient, gradient))
                if slope >= 0.0:
                    break
                length = float(np.max(np.sqrt(np.sum(direction.reshape((m, 2)) ** 2, axis=1))))
                cap = max(32.0 * eps, 0.5 * rt)
                if length > cap:
                    direction *= cap / length
                    slope *= cap / length
                step = 1.0
                accepted = False
                while clock() < local_end and step > 1e-15:
                    trial = z + step * direction
                    if np.array_equal(trial, z):
                        break
                    new_value, new_gradient, new_peak = energy(trial)
                    if math.isfinite(new_value) and new_value <= value + 1e-4 * step * slope:
                        accepted = True
                        break
                    step *= 0.5
                if not accepted:
                    break
                s = trial - z
                y = new_gradient - gradient
                sy = float(np.dot(s, y))
                yy = float(np.dot(y, y))
                if sy > 1e-12 * sqrt(max(0.0, float(np.dot(s, s)) * yy)) and yy > 0.0:
                    history.append((s, y, 1.0 / sy))
                    if len(history) > memory:
                        del history[0]
                z, value, gradient, peak = trial, new_value, new_gradient, new_peak
        except _SliceEnd:
            pass
        if clock() >= deadline or not bool(np.all(np.isfinite(z))):
            return None
        xy = z.reshape((m, 2))
        candidate = list(fixed)
        # Circles that fail verification do not enter the layout; only strict area gains are kept, and failures do not pollute the current solution.
        for j in sorted(range(m), key=lambda j: -weights[ids[j]]):
            p = safe_point(ids[j], float(xy[j, 0]), float(xy[j, 1]), candidate)
            if p is not None:
                candidate.append(p)
        area = math.fsum(weights[p[0]] for p in candidate)
        if area > base_area + 1e-14:
            remember(candidate, area)
            return candidate
        return None

    current, current_area = fill([], 0.0, 0.0, 0.0)
    if current_area < best_area:
        current, current_area = list(best), best_area
    operator_weights = [1.2, 0.8, 0.8, 1.6, 0.3, 1.4 if np is not None else 0.0,
                        0.7 if np is not None else 0.0]

    # No fixed iteration cap; candidate width, neighbourhood depth, noise and local time slices all scale with the budget.
    while clock() < deadline and not optimal:
        now = clock()
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
        before = best_area
        if op >= 5:
            local_span = budget * (0.03 + 0.02 * sqrt(strength)) / (1.0 + nb / 120.0)
            local_deadline = min(deadline, now + min(remaining * 0.25, local_span))
            try:
                keep = elastic(base, local_deadline, op == 5)
            except Exception:
                keep = None
            if keep is None:
                operator_weights[op] = 0.9 * operator_weights[op] + 0.1 * 0.2
                continue
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
        new, new_area = fill(keep, noise, area_weight, rng.random() * tau)
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
