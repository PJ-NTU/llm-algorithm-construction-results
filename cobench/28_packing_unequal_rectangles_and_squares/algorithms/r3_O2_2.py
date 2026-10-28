"""Incremental-capacity elastic rectangle packing search
Uses a strip-based initial solution, joint coordinate relaxation and rotate-and-swap perturbations, saving only feasible layouts that pass the original geometric check.
"""
import math
import random
import time

try:
    import numpy as np
    from scipy.optimize import minimize
    from scipy.spatial import cKDTree
except Exception:
    np = None
    minimize = None
    cKDTree = None


def solve(instance: dict, time_limit_s: float) -> dict:
    started = time.perf_counter()
    answer = {"alpha": [], "x": [], "y": [], "r": []}
    best = []
    try:
        n = int(instance["num_rectangles"])
        answer = {"alpha": [0] * n, "x": [0.0] * n,
                  "y": [0.0] * n, "r": [0] * n}
        budget = float(time_limit_s)
        if n == 0 or not math.isfinite(budget) or budget <= 0:
            return answer
        now = time.perf_counter
        deadline = started + budget - min(0.08 * budget, 0.003 + 0.000006 * n)
        radius = float(instance["container_radius"])
        widths, heights = instance["widths"], instance["heights"]
        rng = random.Random(804719)
        effort = math.log1p(budget)
        dims, physical = {}, {}
        scanned = True
        for i in range(n):
            if now() >= deadline:
                scanned = False
                break
            w, h = float(widths[i]) * 0.5, float(heights[i]) * 0.5
            if math.hypot(w, h) > radius:
                continue
            a, b = w / radius, h / radius
            if not math.isfinite(a + b):
                continue
            dims[i], physical[i] = (a, b), (w, h)
            if not best:
                best = [(i, 0.0, 0.0, 0)]
        items = sorted(dims, key=lambda i: dims[i][0] * dims[i][1])
        if not items:
            return answer
        upper, total = 0, 0.0
        for i in items:
            total += 4.0 * dims[i][0] * dims[i][1]
            if total > math.pi * (1.0 + 1.0e-12 * (upper + 1)):
                break
            upper += 1
        upper = max(1, upper)
        short = sorted(min(dims[i]) for i in items)
        typical = max(1.0e-12, short[len(short) // 2])
        clearance = max(1.0e-14, min(1.0e-9, typical * 1.0e-7))
        best_area = sum(dims[p[0]][0] * dims[p[0]][1] for p in best)

        def save(ids, points, rotations, end):
            """Certifies using the actual output coordinates only; a failed candidate does not change the existing feasible solution."""
            nonlocal best, best_area
            accepted = []
            boxes = []
            order = sorted(range(len(ids)), key=lambda j: dims[ids[j]][0] * dims[ids[j]][1])
            for j in order:
                if now() >= end:
                    break
                i, r = int(ids[j]), int(rotations[j])
                x, y = float(points[j][0]) * radius, float(points[j][1]) * radius
                if not (math.isfinite(x) and math.isfinite(y)):
                    continue
                w, h = physical[i]
                a, b = dims[i]
                if r:
                    w, h, a, b = h, w, b, a
                if math.hypot(abs(x) + w, abs(y) + h) > radius:
                    continue
                nx, ny = x / radius, y / radius
                if math.hypot(abs(nx) + a, abs(ny) + b) > 1.0:
                    continue
                valid = True
                for k, q in enumerate(boxes):
                    if k % 64 == 0 and now() >= end:
                        valid = False
                        break
                    if ((abs(x - q[0]) < w + q[2] and abs(y - q[1]) < h + q[3]) or
                        (abs(nx - q[4]) < a + q[6] and abs(ny - q[5]) < b + q[7])):
                        valid = False
                        break
                if valid:
                    accepted.append((i, x, y, r))
                    boxes.append((x, y, w, h, nx, ny, a, b))
            area = sum(dims[p[0]][0] * dims[p[0]][1] for p in accepted)
            if len(accepted) > len(best) or (len(accepted) == len(best) and area < best_area):
                best, best_area = accepted, area

        def shelf(height, phase, end):
            """Arranges items by available horizontal length within equal-height strips, expanding strips outward from the centre on both sides."""
            options = []
            for i in items:
                if now() >= end:
                    return
                a, b = dims[i]
                choices = []
                if 2.0 * (b + clearance) <= height:
                    choices.append((a + clearance, b + clearance, 0))
                if 2.0 * (a + clearance) <= height:
                    choices.append((b + clearance, a + clearance, 1))
                if choices:
                    aa, bb, r = min(choices)
                    options.append((aa, bb, i, r))
            options.sort()
            ids, points, rotations = [], [], []
            level = 0
            while options and now() < end:
                row = (level + 1) // 2
                cy = (row if level % 2 else -row) * height + phase * height
                level += 1
                far = abs(cy) + height * 0.5
                if far >= 1.0 - clearance:
                    if row * height - abs(phase * height) + height * 0.5 >= 1.0:
                        break
                    continue
                capacity = 2.0 * math.sqrt(max(0.0, (1.0 - clearance) ** 2 - far * far))
                chosen, rest, used = [], [], 0.0
                for entry in options:
                    if now() >= end:
                        return
                    if used + 2.0 * entry[0] <= capacity:
                        chosen.append(entry)
                        used += 2.0 * entry[0]
                    else:
                        rest.append(entry)
                options = rest
                cx = -used * 0.5
                for a, b, i, r in chosen:
                    ids.append(i)
                    points.append((cx + a, cy))
                    rotations.append(r)
                    cx += 2.0 * a
                if chosen:
                    save(ids, points, rotations, end)
                if not chosen and row * height > abs(phase * height):
                    break

        # Strips only provide a cheap initial solution; most of the time goes to the search that can move all centres simultaneously.
        seed_end = min(deadline, now() + budget / (8.0 + 2.0 * effort))
        seed_turn = 0
        candidate_heights = sorted({2.0 * (v + clearance) for i in items for v in dims[i]
                                    if v + clearance < 1.0})
        while now() < seed_end and candidate_heights:
            if scanned and len(best) >= upper:
                break
            if seed_turn == 0:
                h = 2.0 * (short[min(len(short) - 1, max(0, upper // 2))] + clearance)
            else:
                h = candidate_heights[int(rng.random() ** 1.5 * len(candidate_heights))]
            shelf(h, 0.0 if seed_turn % 2 == 0 else 0.5, seed_end)
            seed_turn += 1
        if np is None or minimize is None:
            while now() < deadline and candidate_heights and len(best) < upper:
                shelf(rng.choice(candidate_heights), rng.uniform(-0.5, 0.5), deadline)
        else:
            generator = np.random.default_rng(804719)
            scale = 1.0 / max(typical * typical, 1.0e-16)
            area = {i: dims[i][0] * dims[i][1] for i in items}

            class LocalDeadline(Exception):
                pass

            def halves(ids, rotations):
                return np.array([dims[i][::-1] if int(r) else dims[i]
                                 for i, r in zip(ids, rotations)], dtype=float)

            def pair_indices(z, h):
                # Large layouts generate possibly intersecting rectangle pairs from circumcircle neighbourhoods, avoiding quadratic memory.
                if len(z) ** 2 <= (1.0 + effort) * 65536.0:
                    return np.triu_indices(len(z), 1)
                distance = 2.0 * float(np.max(np.hypot(h[:, 0], h[:, 1]))) + 4.0 * clearance
                pairs = cKDTree(z).query_pairs(distance, output_type='ndarray')
                return pairs[:, 0], pairs[:, 1]

            def relax(ids, start, rotations, end):
                h = halves(ids, rotations) + clearance
                k = len(ids)
                dense = k * k <= (1.0 + effort) * 65536.0
                fixed = np.triu_indices(k, 1) if dense else None
                winner = start.copy()
                lowest = math.inf

                def energy(flat):
                    nonlocal winner, lowest
                    if now() >= end:
                        raise LocalDeadline()
                    z = flat.reshape(k, 2)
                    p, q = fixed if dense else pair_indices(z, h)
                    d = z[p] - z[q]
                    pen = h[p] + h[q] - np.abs(d)
                    horizontal = pen[:, 0] <= pen[:, 1]
                    overlap = np.maximum(0.0, np.minimum(pen[:, 0], pen[:, 1]))
                    force = -2.0 * overlap
                    gx = force * horizontal * np.where(d[:, 0] >= 0.0, 1.0, -1.0)
                    gy = force * (~horizontal) * np.where(d[:, 1] >= 0.0, 1.0, -1.0)
                    g = np.empty_like(z)
                    g[:, 0] = np.bincount(p, gx, minlength=k) - np.bincount(q, gx, minlength=k)
                    g[:, 1] = np.bincount(p, gy, minlength=k) - np.bincount(q, gy, minlength=k)
                    corners = np.abs(z) + h
                    lengths = np.hypot(corners[:, 0], corners[:, 1])
                    outside = np.maximum(0.0, lengths - (1.0 - clearance))
                    g += (4.0 * outside / np.maximum(lengths, 1.0e-300))[:, None] * corners * np.sign(z)
                    value = float(np.dot(overlap, overlap) + 2.0 * np.dot(outside, outside)) * scale
                    if value < lowest:
                        lowest, winner = value, z.copy()
                    return value, (g * scale).ravel()

                try:
                    minimize(energy, start.ravel(), jac=True, method='L-BFGS-B',
                             options={'ftol': 0.0, 'gtol': 1.0e-10,
                                      'maxiter': math.inf, 'maxfun': math.inf,
                                      'maxls': max(12, int(10 + 5 * effort)),
                                      'maxcor': max(4, int(4 + 3 * effort))})
                except LocalDeadline:
                    pass
                return winner, lowest

            def position(i, rotation):
                a, b = dims[i][::-1] if rotation else dims[i]
                extent = max(0.0, math.sqrt(max(0.0, 1.0 - b * b)) - a)
                x = rng.uniform(-extent, extent)
                extent = max(0.0, math.sqrt(max(0.0, 1.0 - (abs(x) + a) ** 2)) - b)
                return x, rng.uniform(-extent, extent)

            def augment(ids, z, rotations, target):
                selected = set(ids)
                while len(ids) < target and now() < deadline:
                    i = next((j for j in items if j not in selected), None)
                    if i is None:
                        break
                    count = max(8, int((3.0 + 4.0 * effort) * math.sqrt(len(ids) + 1)))
                    old_h = halves(ids, rotations) if ids else np.empty((0, 2))
                    value, choice = math.inf, (0.0, 0.0, 0)
                    for r in (0, 1):
                        if now() >= deadline:
                            break
                        samples = np.array([position(i, r) for _ in range(count)])
                        if ids:
                            h = np.array(dims[i][::-1] if r else dims[i])
                            pen = h + old_h[None, :, :] - np.abs(samples[:, None, :] - z[None, :, :])
                            overlap = np.maximum(0.0, np.minimum(pen[:, :, 0], pen[:, :, 1]))
                            values = np.sum(overlap * overlap, axis=1)
                            j = int(np.argmin(values))
                            score = float(values[j])
                        else:
                            j, score = 0, 0.0
                        if score < value:
                            value, choice = score, (samples[j, 0], samples[j, 1], r)
                    ids.append(i)
                    rotations = np.append(rotations, choice[2])
                    z = np.vstack((z, choice[:2]))
                    selected.add(i)
                return ids, z, rotations

            def certify_layout(ids, z, rotations, end):
                save(ids, z, rotations, end)
                if now() >= end or len(ids) <= len(best):
                    return
                # Slightly scale the centres outward to remove convergence residuals; containment in the circle must still be re-certified.
                h = halves(ids, rotations)
                p, q = pair_indices(z, h)
                d = np.abs(z[p] - z[q])
                ratios = (h[p] + h[q]) / np.maximum(d, 1.0e-300)
                factor = max(1.0, float(np.max(np.min(ratios, axis=1))) if len(p) else 1.0)
                if 1.0 < factor < 1.001:
                    save(ids, z * (factor + 4.0e-13), rotations, end)

            def perturb(ids, z, rotations, stagnation):
                ids, z, rotations = ids[:], z.copy(), rotations.copy()
                k = len(ids)
                h = halves(ids, rotations)
                p, q = pair_indices(z, h)
                pen = h[p] + h[q] - np.abs(z[p] - z[q])
                ov = np.maximum(0.0, np.minimum(pen[:, 0], pen[:, 1])) ** 2
                load = np.bincount(p, ov, minlength=k) + np.bincount(q, ov, minlength=k)
                load += np.maximum(0.0, np.hypot(*(np.abs(z) + h).T) - 1.0) ** 2
                j = rng.choices(range(k), weights=(load + max(1.0e-15, float(np.mean(load)) * 0.1)).tolist())[0]
                depth = min(k, max(1, int(math.sqrt(k) * (0.15 + 0.12 * effort) * (1.0 + stagnation))))
                mode = rng.random()
                if mode < 0.27:
                    chosen = {j}
                    chosen.update(rng.sample(range(k), rng.randrange(1, depth + 1)))
                    for t in chosen:
                        rotations[t] = 1 - rotations[t]
                elif mode < 0.51:
                    missing = [i for i in items if i not in set(ids)]
                    if missing:
                        if rng.random() < 0.6:
                            j = max(range(k), key=lambda t: area[ids[t]] * (0.8 + rng.random() * 0.4))
                        pool = min(len(missing), max(1, int(math.sqrt(len(items)) * (0.5 + effort))))
                        replacement = missing[int(rng.random() ** 3 * pool)]
                        ids[j] = replacement
                        rotations[j] = rng.randrange(2)
                    else:
                        rotations[j] = 1 - rotations[j]
                elif mode < 0.72:
                    z[j] = position(ids[j], int(rotations[j]))
                    if depth > 1 and rng.random() < 0.5:
                        t = rng.randrange(k)
                        z[t] = position(ids[t], int(rotations[t]))
                elif mode < 0.88:
                    amplitude = typical * (0.1 + rng.random() * (0.4 + stagnation))
                    z += generator.normal(0.0, amplitude, z.shape)
                else:
                    t = rng.randrange(k)
                    z[[j, t]] = z[[t, j]]
                    if rng.random() < 0.5:
                        rotations[j] = 1 - rotations[j]
                return ids, z, rotations

            target = 0
            current = elite = None
            current_value = elite_value = math.inf
            progress_at = now()
            first = True
            while now() < deadline:
                if scanned and len(best) >= upper:
                    break
                wanted = min(len(items), max(len(best) + 1, 1))
                if wanted != target or current is None:
                    target = wanted
                    ids = [p[0] for p in best]
                    z = np.array([(p[1] / radius, p[2] / radius) for p in best], dtype=float).reshape(-1, 2)
                    rotations = np.array([p[3] for p in best], dtype=int)
                    current = augment(ids, z, rotations, target)
                    elite = current
                    current_value = elite_value = math.inf
                    progress_at, first = now(), True
                stagnation = (now() - progress_at) / max(0.001, budget / (3.0 + effort))
                if stagnation > 1.0:
                    # Restarts search a subset of fixed size directly, without reusing contact insertion or removal-based rebuilding.
                    pool_size = min(len(items), target + max(1, int(effort * math.sqrt(target))))
                    pool = items[:pool_size]
                    ranking = sorted(pool, key=lambda i: math.log(max(area[i], 1.0e-300)) + rng.gauss(0.0, 0.15 + 0.12 * effort))
                    ids = ranking[:target]
                    rotations = generator.integers(0, 2, len(ids))
                    z = np.array([position(i, int(r)) for i, r in zip(ids, rotations)])
                    trial = (ids, z, rotations)
                    progress_at = now()
                    current_value = elite_value = math.inf
                    first = True
                elif first:
                    trial = current
                else:
                    source = elite if rng.random() < 0.7 else current
                    trial = perturb(*source, stagnation)
                ids, z, rotations = trial
                local_span = budget / (8.0 + (2.0 + effort) * math.sqrt(len(ids) + 1))
                local_end = min(deadline, now() + max(0.001, local_span))
                z, value = relax(ids, z, rotations, local_end)
                certify_layout(ids, z, rotations, deadline)
                trial = (ids, z, rotations)
                if value < elite_value:
                    if value < elite_value * 0.97:
                        progress_at = now()
                    elite, elite_value = trial, value
                temperature = max(1.0e-9, elite_value * (0.08 + 0.1 * min(2.0, stagnation)))
                if first or value <= current_value or rng.random() < math.exp(min(0.0, (current_value - value) / temperature)):
                    current, current_value = trial, value
                first = False
    except Exception:
        # If any optional search component fails, the certified complete feasible solution is still returned.
        pass
    try:
        for i, x, y, r in best:
            answer["alpha"][i] = 1
            answer["x"][i] = float(x)
            answer["y"][i] = float(y)
            answer["r"][i] = int(r)
    except Exception:
        pass
    return answer
