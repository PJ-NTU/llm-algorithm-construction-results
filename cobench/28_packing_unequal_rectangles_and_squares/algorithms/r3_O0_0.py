"""Contact construction and perturbation-repair search for rectangles in a circle
Combines contact-position insertion, local tear-down and rearrangement, and continuous coordinate repair; searches within the wall-clock limit and keeps the best feasible solution that passes the geometric check.
"""

import math
import random
import time


def solve(instance: dict, time_limit_s: float) -> dict:
    started = time.perf_counter()
    answer = {"alpha": [], "x": [], "y": [], "r": []}
    try:
        n = int(instance["num_rectangles"])
        answer = {"alpha": [0] * n, "x": [0.0] * n,
                  "y": [0.0] * n, "r": [0] * n}
        budget = max(0.0, float(time_limit_s))
        if n == 0 or not math.isfinite(budget) or budget <= 0.0:
            return answer
        _search(instance, budget, started, answer)
    except Exception:
        # The published solution is always complete, so it can be returned directly even if the search or an optional numerical module fails.
        pass
    return answer


def _search(instance, budget, started, answer):
    clock = time.perf_counter
    deadline = started + budget - min(0.04, 0.025 * budget)
    radius = float(instance["container_radius"])
    widths = instance["widths"]
    heights = instance["heights"]
    n = len(answer["alpha"])
    rng = random.Random(719381)
    strength = math.log1p(budget)
    raw = {}
    dims = {}
    areas = {}
    eligible = []
    best = []

    def publish(layout):
        """Checks in the actual output units; only publishes complete solutions with a strictly larger count."""
        nonlocal best
        good = []
        physical = []
        for item in layout:
            i, cx, cy, a, b, rotation = item
            xx, yy = float(cx * radius), float(cy * radius)
            hw = float((heights[i] if rotation else widths[i]) * 0.5)
            hh = float((widths[i] if rotation else heights[i]) * 0.5)
            if not (math.isfinite(xx) and math.isfinite(yy)):
                continue
            if math.hypot(abs(xx) + hw, abs(yy) + hh) > radius:
                continue
            if any(abs(xx - qx) < hw + qa and abs(yy - qy) < hh + qb
                   for qx, qy, qa, qb in physical):
                continue
            physical.append((xx, yy, hw, hh))
            good.append(item)
        if len(good) > len(best):
            for item in best:
                i = item[0]
                answer["alpha"][i] = 0
                answer["x"][i] = 0.0
                answer["y"][i] = 0.0
                answer["r"][i] = 0
            for item, position in zip(good, physical):
                i = item[0]
                answer["alpha"][i] = 1
                answer["x"][i] = position[0]
                answer["y"][i] = position[1]
                answer["r"][i] = int(item[5])
            best = good[:]
        return good

    for i in range(n):
        a = (float(widths[i]) * 0.5) / radius
        b = (float(heights[i]) * 0.5) / radius
        diagonal = math.hypot(a, b)
        if diagonal <= 1.0 and a > 0.0 and b > 0.0:
            raw[i] = (a, b)
            pad = min(1.0e-9, min(a, b) * 1.0e-5,
                      max(0.0, (1.0 - diagonal) * 0.25))
            dims[i] = (a + pad, b + pad)
            areas[i] = 4.0 * a * b
            eligible.append(i)
            if not best:
                publish([(i, 0.0, 0.0, a + pad, b + pad, 0)])
        if clock() >= deadline:
            return
    if not eligible:
        return
    eligible.sort(key=areas.__getitem__)
    upper = 0
    area_sum = 0.0
    for i in eligible:
        area_sum += areas[i]
        if area_sum > math.pi * (1.0 + 1.0e-12):
            break
        upper += 1
    if len(best) >= upper or clock() >= deadline:
        return

    # Two-item layouts admit a simple exact separation test that also handles exactly touching boundaries.
    if upper <= 2:
        for pos, i in enumerate(eligible):
            for j in eligible[pos + 1:]:
                for ri in (0, 1):
                    a, b = raw[i][::(-1 if ri else 1)]
                    for rj in (0, 1):
                        c, d = raw[j][::(-1 if rj else 1)]
                        for vertical in (False, True):
                            aa, bb = (b, a) if vertical else (a, b)
                            cc, dd = (d, c) if vertical else (c, d)
                            left = math.sqrt(max(0.0, 1.0 - bb * bb)) - aa
                            right = math.sqrt(max(0.0, 1.0 - dd * dd)) - cc
                            if left + right >= aa + cc:
                                xi, yi = (0.0, -left) if vertical else (-left, 0.0)
                                xj, yj = (0.0, right) if vertical else (right, 0.0)
                                publish([(i, xi, yi, a, b, ri),
                                         (j, xj, yj, c, d, rj)])
                                if len(best) >= upper:
                                    return
                if clock() >= deadline:
                    return
    if deadline - clock() < 0.02:
        return
    import numpy as np
    nrng = np.random.default_rng(381719)
    tiny = min(1.0e-12, min(min(raw[i]) for i in eligible) * 1.0e-7)
    pool = []
    pool_size = max(2, int(2 + strength * math.sqrt(math.log1p(len(eligible)))))

    def insert(i, layout, slope, height_weight, until):
        """Scans obstacle edges and arc contact events to find a low insertion position in the given direction."""
        if clock() >= until:
            return None
        k = len(layout)
        if k:
            geom = np.asarray([(q[1], q[2], q[3], q[4]) for q in layout])
            px, py, pa, pb = geom.T
            order = np.argsort(py - pb)
        else:
            px = py = pa = pb = np.empty(0)
            order = ()
        chosen = None
        value = float("inf")
        point_limit = max(24, int(60 + 300 * math.sqrt(budget / (k + 1))))
        for rotation in (0, 1):
            a, b = dims[i] if rotation == 0 else dims[i][::-1]
            if rotation and a == b:
                continue
            xmax = math.sqrt(max(0.0, 1.0 - b * b)) - a
            if xmax < -tiny:
                continue
            xmax = max(0.0, xmax)
            smooth = -math.copysign(max(0.0, abs(slope) /
                                      math.sqrt(1.0 + slope * slope) - a), slope)
            essential = np.array([0.0, -xmax, xmax, smooth])
            if k:
                ye = np.concatenate((py - pb - b, py + pb + b))
                ye = ye[np.abs(ye) + b <= 1.0]
                arc = np.sqrt(np.maximum(0.0, 1.0 - (np.abs(ye) + b) ** 2)) - a
                arc = arc[arc >= -tiny]
                xs = np.concatenate((essential, px - pa - a, px + pa + a,
                                     px - pa + a, px + pa - a, arc, -arc))
            else:
                xs = essential
            xs = np.unique(np.clip(xs[np.abs(xs) <= xmax + tiny], -xmax, xmax))
            if len(xs) > point_limit:
                xs = np.concatenate((essential[:3],
                                     xs[nrng.choice(len(xs), point_limit, replace=False)]))
            low = b - np.sqrt(np.maximum(0.0, 1.0 - (np.abs(xs) + a) ** 2))
            high = -low
            for j in order:
                blocked = ((np.abs(xs - px[j]) < a + pa[j] - tiny) &
                           (low > py[j] - pb[j] - b + tiny) &
                           (low < py[j] + pb[j] + b - tiny))
                low = np.where(blocked, py[j] + pb[j] + b, low)
                if clock() >= until:
                    return chosen
            valid = low <= high + tiny
            if np.any(valid):
                low = np.minimum(low, high)
                costs = np.where(valid, low + height_weight * b + slope * xs, np.inf)
                p = int(np.argmin(costs))
                cost = float(costs[p])
                if cost < value:
                    value = cost
                    chosen = (i, float(xs[p]), float(low[p]), a, b, rotation)
        return chosen

    def construct(order, layout, slope, height_weight, until):
        used = {q[0] for q in layout}
        for i in order:
            if clock() >= until:
                break
            if i in used:
                continue
            item = insert(i, layout, slope, height_weight, until)
            if item is not None:
                layout.append(item)
                used.add(i)
                if len(layout) > len(best):
                    publish(layout)
                if len(best) >= upper:
                    break
        return layout

    def remember(layout):
        layout = publish(layout)
        if len(layout) < max(1, len(best) - 1):
            return
        pool.append(layout)
        if len(pool) > pool_size:
            least = min(len(q) for q in pool)
            choices = [j for j, q in enumerate(pool) if len(q) == least]
            del pool[rng.choice(choices)]

    initial_until = min(deadline, clock() + budget * 0.18)
    initial = construct(eligible, [], 0.015, 1.0, initial_until)
    remember(initial)
    if len(best) >= upper:
        return

    def seed_position(i, others, until):
        """Chooses starting points for continuous repair inside the circle with few conflicts with the current layout."""
        k = len(others)
        if k == 0:
            a, b = dims[i]
            return (i, 0.0, 0.0, a, b, 0)
        geom = np.asarray([(q[1], q[2], q[3], q[4]) for q in others])
        px, py, pa, pb = geom.T
        amount = max(16, int(40 * math.sqrt((budget + 1.0) * (k + 1))))
        amount = min(amount, max(16, int(25000 * (1 + strength) / (k + 1))))
        chosen, merit = None, float("inf")
        for rotation in (0, 1):
            if clock() >= until:
                break
            a, b = dims[i] if not rotation else dims[i][::-1]
            xmax = max(0.0, math.sqrt(max(0.0, 1.0 - b * b)) - a)
            xe = np.concatenate(([0.0], px, px - pa - a, px + pa + a))
            ye = np.concatenate(([0.0], py, py - pb - b, py + pb + b))
            if len(xe) * len(ye) <= amount:
                xx, yy = np.meshgrid(xe, ye)
                xx, yy = xx.ravel(), yy.ravel()
            else:
                xx = xe[nrng.integers(len(xe), size=amount)]
                yy = ye[nrng.integers(len(ye), size=amount)]
            xx = np.concatenate((xx, nrng.uniform(-xmax, xmax, size=amount)))
            yy = np.concatenate((yy, nrng.uniform(-1.0, 1.0, size=amount)))
            xx = np.clip(xx, -xmax, xmax)
            ymax = np.maximum(0.0, np.sqrt(np.maximum(0.0, 1.0 - (np.abs(xx) + a) ** 2)) - b)
            yy = np.clip(yy, -ymax, ymax)
            dx = a + pa[None, :] - np.abs(xx[:, None] - px[None, :])
            dy = b + pb[None, :] - np.abs(yy[:, None] - py[None, :])
            overlap = np.maximum(0.0, np.minimum(dx, dy))
            scores = np.sum(overlap * overlap, axis=1)
            p = int(np.argmin(scores))
            if float(scores[p]) < merit:
                merit = float(scores[p])
                chosen = (i, float(xx[p]), float(yy[p]), a, b, rotation)
        return chosen

    def minimize_overlap(layout, until):
        """Limited-memory quasi-Newton repair with backtracking; every loop is bounded by the wall clock."""
        k = len(layout)
        a = np.array([q[3] for q in layout])
        b = np.array([q[4] for q in layout])
        ii, jj = np.triu_indices(k, 1)
        sa, sb = a[ii] + a[jj], b[ii] + b[jj]
        z = np.array([q[1] for q in layout] + [q[2] for q in layout])

        def evaluate(v):
            xx, yy = v[:k], v[k:]
            ux, uy = np.abs(xx) + a, np.abs(yy) + b
            radial = np.sqrt(ux * ux + uy * uy)
            excess = np.maximum(0.0, radial - 1.0)
            factor = 2.0 * excess / np.maximum(radial, 1.0e-300)
            gx = factor * ux * np.sign(xx)
            gy = factor * uy * np.sign(yy)
            xd, yd = xx[ii] - xx[jj], yy[ii] - yy[jj]
            dx, dy = sa - np.abs(xd), sb - np.abs(yd)
            active = (dx > 0.0) & (dy > 0.0)
            mx = active & (dx <= dy)
            my = active & ~mx
            wx = -2.0 * dx[mx] * np.where(xd[mx] >= 0.0, 1.0, -1.0)
            wy = -2.0 * dy[my] * np.where(yd[my] >= 0.0, 1.0, -1.0)
            gx += np.bincount(ii[mx], weights=wx, minlength=k)
            gx -= np.bincount(jj[mx], weights=wx, minlength=k)
            gy += np.bincount(ii[my], weights=wy, minlength=k)
            gy -= np.bincount(jj[my], weights=wy, minlength=k)
            f = float(np.dot(excess, excess) + np.dot(dx[mx], dx[mx]) + np.dot(dy[my], dy[my]))
            return f, np.concatenate((gx, gy))

        f, g = evaluate(z)
        history = []
        memory = max(2, int(2 + 2 * strength + math.log1p(k)))
        while clock() < until and f > 0.0:
            q = g.copy()
            alphas = []
            for s, y, rho in reversed(history):
                alpha = rho * float(np.dot(s, q))
                alphas.append(alpha)
                q -= alpha * y
            if history:
                s, y, rho = history[-1]
                q *= float(np.dot(s, y)) / max(1.0e-300, float(np.dot(y, y)))
            else:
                q /= 2.0 + math.sqrt(k)
            for (s, y, rho), alpha in zip(history, reversed(alphas)):
                q += s * (alpha - rho * float(np.dot(y, q)))
            direction = -q
            derivative = float(np.dot(g, direction))
            if not math.isfinite(derivative):
                break
            if derivative >= 0.0:
                direction = -g / (2.0 + math.sqrt(k))
                derivative = float(np.dot(g, direction))
                history.clear()
            norm = float(np.max(np.abs(direction)))
            if norm < 1.0e-15 or derivative >= 0.0:
                break
            step = min(1.0, 0.5 / norm)
            accepted = False
            while clock() < until and step * norm > 1.0e-15:
                nz = z + step * direction
                nf, ng = evaluate(nz)
                if nf <= f + 1.0e-4 * step * derivative:
                    accepted = True
                    break
                step *= 0.5
            if not accepted:
                break
            s, y = nz - z, ng - g
            curvature = float(np.dot(s, y))
            if curvature > 1.0e-12 * math.sqrt(max(0.0, float(np.dot(s, s) * np.dot(y, y)))):
                history.append((s, y, 1.0 / curvature))
                if len(history) > memory:
                    del history[0]
            z, f, g = nz, nf, ng
            if f < 1.0e-26:
                break
        repaired = [(q[0], float(z[j]), float(z[j + k]), q[3], q[4], q[5])
                    for j, q in enumerate(layout)]
        return repaired, f

    def expand(parent, until):
        selected = {q[0] for q in parent}
        unused = [i for i in eligible if i not in selected]
        if not unused:
            return
        window = min(len(unused), max(1, int(1 + math.sqrt(len(parent) + 1) * strength)))
        i = unused[min(window - 1, int(window * rng.random() ** 2))]
        item = seed_position(i, parent, until)
        if item is None:
            return
        trial = parent[:] + [item]
        best_trial = trial
        best_energy = float("inf")
        while clock() < until:
            slice_end = min(until, clock() + budget / (30.0 + 3.0 * math.sqrt(len(trial))))
            trial, energy = minimize_overlap(trial, slice_end)
            good = publish(trial)
            if len(good) == len(trial):
                remember(good)
                return
            if energy < best_energy:
                best_energy, best_trial = energy, trial[:]
            else:
                trial = best_trial[:]
            if clock() >= until:
                break
            k = len(trial)
            geom = np.array([(q[1], q[2], q[3], q[4]) for q in trial])
            xx, yy, aa, bb = geom.T
            overlap = np.maximum(0.0, np.minimum(aa[:, None] + aa[None, :] -
                                 np.abs(xx[:, None] - xx[None, :]),
                                 bb[:, None] + bb[None, :] -
                                 np.abs(yy[:, None] - yy[None, :])))
            np.fill_diagonal(overlap, 0.0)
            stress = np.sum(overlap * overlap, axis=1)
            stress += np.maximum(0.0, np.hypot(np.abs(xx) + aa, np.abs(yy) + bb) - 1.0) ** 2
            j = int(np.argmax(stress)) if rng.random() < 0.75 else rng.randrange(k)
            q = trial[j]
            if q[3] != q[4] and rng.random() < 0.55:
                trial[j] = (q[0], q[1], q[2], q[4], q[3], 1 - q[5])
            elif rng.random() < 0.45:
                alternatives = [u for u in unused if u not in {v[0] for v in trial}]
                idx = alternatives[int(len(alternatives) * rng.random() ** 3)] if alternatives else q[0]
                replacement = seed_position(idx, trial[:j] + trial[j + 1:], until)
                if replacement is not None:
                    trial[j] = replacement
            else:
                scale = min(0.08, max(1.0e-7, math.sqrt(max(energy, 0.0) / k)))
                noise = nrng.normal(0.0, scale, size=(k, 2))
                trial = [(q[0], q[1] + float(noise[t, 0]), q[2] + float(noise[t, 1]),
                          q[3], q[4], q[5]) for t, q in enumerate(trial)]

    last_improvement = clock()
    previous_count = len(best)
    while clock() < deadline and len(best) < upper:
        remaining = deadline - clock()
        if len(best) > previous_count:
            last_improvement, previous_count = clock(), len(best)
        depth = max(1, int(math.sqrt(len(best) + 1) * math.sqrt(1.0 + strength)))
        stale = min(1.0, (clock() - last_improvement) / max(0.001, budget * 0.2))
        local_time = min(remaining * 0.3, budget * (0.04 + 0.025 * stale))
        if (len(best) > 1 and remaining > 0.01 and
                len(best) ** 2 < 120000 * local_time and
                rng.random() < 0.17 + 0.16 * stale):
            choices = [q for q in pool if len(q) == len(best)]
            parent = rng.choice(choices) if choices and rng.random() < 0.5 else best
            expand(parent, min(deadline, clock() + local_time))
            continue
        restart = not pool or rng.random() < 0.26
        slope = rng.uniform(-0.6, 0.6) if rng.random() < 0.6 else rng.uniform(-0.035, 0.035)
        height_weight = rng.choice((0.0, 1.0, 1.0, 2.0))
        until = min(deadline, clock() + max(budget * 0.045,
                                            budget * 0.15 / math.sqrt(1.0 + len(best) / 25.0)))
        if restart:
            layout = []
            removed = set()
        else:
            parent = rng.choice(pool) if rng.random() < 0.65 else best
            swap = rng.random() < 0.5
            sx, sy = rng.choice((-1, 1)), rng.choice((-1, 1))
            if swap:
                layout = [(q[0], sx * q[2], sy * q[1], q[4], q[3], 1 - q[5]) for q in parent]
            else:
                layout = [(q[0], sx * q[1], sy * q[2], q[3], q[4], q[5]) for q in parent]
            count = min(len(layout), 1 + int(depth * (0.5 + stale) * rng.random() ** 1.5))
            pivot = rng.choice(layout)
            mode = rng.random()
            if mode < 0.6:
                ranked = sorted(layout, key=lambda q: (q[1] - pivot[1]) ** 2 + (q[2] - pivot[2]) ** 2)
            elif mode < 0.8:
                ranked = sorted(layout, key=lambda q: -areas[q[0]] * rng.uniform(0.5, 1.5))
            else:
                ranked = rng.sample(layout, len(layout))
            removed = {q[0] for q in ranked[:count]}
            layout = [q for q in layout if q[0] not in removed]
            if layout and rng.random() < strength / (strength + 4.0):
                moving = sorted(layout, key=lambda q: q[2] + slope * q[1])
                for q in moving:
                    if clock() >= until:
                        break
                    others = [v for v in layout if v[0] != q[0]]
                    moved = insert(q[0], others, slope, height_weight, until)
                    if moved is not None and moved[2] + slope * moved[1] <= q[2] + slope * q[1] + tiny:
                        layout = others + [moved]
        used = {q[0] for q in layout}
        candidates = [i for i in eligible if i not in used]
        noise = rng.uniform(0.0, 0.5 + 0.2 * strength)
        shape_weight = rng.uniform(0.0, 0.7)
        priorities = {i: (areas[i] ** (1.0 - shape_weight) *
                         max(raw[i]) ** (2.0 * shape_weight) * math.exp(noise * rng.uniform(-1.0, 1.0)) *
                         (rng.uniform(1.0, 1.8) if i in removed else 1.0)) for i in candidates}
        candidates.sort(key=priorities.__getitem__)
        if rng.random() < 0.72:
            target = max(1, len(best) + 1 - len(layout))
            target += int(depth * rng.random() ** 2)
            prefix = candidates[:target]
            metric = rng.random()
            if metric < 0.5:
                prefix.sort(key=lambda i: -areas[i])
            elif metric < 0.85:
                prefix.sort(key=lambda i: -max(raw[i]))
            else:
                prefix.sort(key=lambda i: -min(raw[i]))
            candidates = prefix + candidates[target:]
        layout = construct(candidates, layout, slope, height_weight, until)
        remember(layout)
