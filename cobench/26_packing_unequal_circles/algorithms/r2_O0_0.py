"""Prefix expansion with multi-start circle packing search
Uses tangent-point construction, analytic-gradient overlap elimination and adaptive perturbation search, saving and expanding the verified feasible prefix within the wall-clock budget.
"""

import math
import time

try:
    import numpy as np
    from scipy.optimize import minimize
    from scipy.spatial import cKDTree
except Exception:
    np = None


class _StopLocal(Exception):
    pass


def solve(instance: dict, time_limit_s: float) -> dict:
    started = time.perf_counter()
    answer = {"packing_status": [], "centers": []}
    try:
        n = max(0, int(instance["n"]))
        answer = {
            "packing_status": [0] * n,
            "centers": [[-1.0, -1.0] for _ in range(n)],
        }
        cx, cy, radius = (float(instance[s]) for s in ("cx", "cy", "R"))
        radii = [float(v) for v in instance["radii"][:n]]
        if len(radii) != n or not all(math.isfinite(v) for v in [cx, cy, radius] + radii):
            return answer
        if radius < 0.0 or any(v < 0.0 for v in radii):
            return answer
        zeros = 0
        while zeros < n and radii[zeros] == 0.0:
            answer["packing_status"][zeros] = 1
            answer["centers"][zeros] = [cx, cy]
            zeros += 1
        if zeros == n or radius == 0.0:
            return answer
        if zeros == 0 and radii[0] <= radius:
            answer["packing_status"][0] = 1
            answer["centers"][0] = [cx, cy]
        budget = max(0.0, float(time_limit_s))
        reserve = min(0.15, max(0.001, budget * 0.025))
        deadline = started + budget - reserve
        if np is not None and time.perf_counter() < deadline:
            _search(radii, zeros, cx, cy, radius, budget, deadline, answer)
    except Exception:
        pass
    return answer


def _search(original, zeros, cx, cy, radius, budget, deadline, answer):
    r = np.asarray(original[zeros:], dtype=float) / radius
    upper = int(np.searchsorted(np.cumsum(r * r), 1.0 + 8e-15, side="right"))
    while upper > 0 and r[upper - 1] > 1.0:
        upper -= 1
    while upper > 1 and r[upper - 2] + r[upper - 1] > 1.0 + 4e-15:
        upper -= 1
    if upper == 0:
        return
    r = r[:upper]
    guard = min(2e-8, float(r[0]) * 1e-5)
    rng = np.random.default_rng(812367)
    effort = math.log1p(budget)
    dense_limit = max(48, int(math.sqrt(min(400000.0, 5000.0 * (1.0 + budget)))))
    population_size = max(2, int(2 + effort))
    best = np.zeros((1, 2)) if zeros == 0 and answer["packing_status"][0] else np.empty((0, 2))
    shift = np.array([cx, cy])

    def pairs(p, radii):
        k = len(p)
        if k <= dense_limit:
            return np.triu_indices(k, 1)
        found = cKDTree(p).query_pairs(2.0 * float(np.max(radii)) + 2.0 * guard, output_type="ndarray")
        return found[:, 0], found[:, 1]

    def commit(p):
        nonlocal best
        k = len(p)
        if k <= len(best) or not np.all(np.isfinite(p)):
            return False
        physical = p * radius + shift
        actual = (physical - shift) / radius
        if np.any(np.hypot(actual[:, 0], actual[:, 1]) + r[:k] > 1.0):
            return False
        a, b = pairs(actual, r[:k])
        if len(a):
            delta = (physical[a] - physical[b]) / radius
            if np.any(np.hypot(delta[:, 0], delta[:, 1]) < r[a] + r[b]):
                return False
        zero_position = None
        if zeros:
            for length in (1.0 - guard * 0.25, 1.0):
                for v in ((length, 0.0), (0.0, length), (-length, 0.0), (0.0, -length)):
                    zphys = np.asarray(v) * radius + shift
                    zrel = (zphys - shift) / radius
                    diff = (physical - zphys) / radius
                    if math.hypot(*zrel) <= 1.0 and np.all(np.hypot(diff[:, 0], diff[:, 1]) >= r[:k]):
                        zero_position = zphys.tolist()
                        break
                if zero_position is not None:
                    break
            if zero_position is None:
                return False
        coordinates = physical.tolist()
        for i in range(zeros):
            answer["centers"][i] = zero_position[:]
        for i, point in enumerate(coordinates, zeros):
            answer["centers"][i] = point
            answer["packing_status"][i] = 1
        best = actual.copy()
        return True

    commit(np.zeros((1, 2)))
    cumulative = np.cumsum(r)
    line_k = int(np.searchsorted(cumulative, 1.0, side="right"))
    if line_k > len(best):
        slack = max(0.0, 1.0 - float(cumulative[line_k - 1]))
        gap = min(guard, slack / max(1, line_k))
        half = float(cumulative[line_k - 1]) + 0.5 * (line_k - 1) * gap
        p = np.zeros((line_k, 2))
        p[:, 0] = 2.0 * cumulative[:line_k] - r[:line_k] - half + np.arange(line_k) * gap
        commit(p)
    if len(best) == upper or time.perf_counter() >= deadline:
        return

    def lattice(k):
        side = int(math.ceil(math.sqrt(k))) + 1
        row, col = np.meshgrid(np.arange(-side, side + 1), np.arange(-side, side + 1), indexing="ij")
        points = np.column_stack(((col + 0.5 * (row % 2)).ravel(), (row * (math.sqrt(3.0) / 2.0)).ravel()))
        norm2 = np.einsum("ij,ij->i", points, points)
        use = np.argpartition(norm2, k - 1)[:k]
        use = use[np.argsort(norm2[use])]
        return points[use]

    uniform = np.flatnonzero(np.arange(1, upper + 1) * r * r <= 0.65)
    if len(uniform) and int(uniform[-1]) + 1 > len(best):
        k = int(uniform[-1]) + 1
        p = lattice(k) * (2.0 * r[k - 1] + guard)
        bad = np.flatnonzero(np.hypot(p[:, 0], p[:, 1]) + r[:k] > 1.0 - guard)
        commit(p[:int(bad[0])] if len(bad) else p)
    if len(best) == upper:
        return

    def hole(p, radii, new_radius, until, direction=0.0):
        cap = max(0.0, 1.0 - new_radius - guard)
        if len(p) == 0:
            return np.array([-cap, 0.0]), True
        if time.perf_counter() >= until:
            return None, False
        reach = radii + new_radius + guard
        candidates = [np.array([[-cap, 0.0], [cap, 0.0], [0.0, -cap], [0.0, cap], [0.0, 0.0]])]
        distance = np.hypot(p[:, 0], p[:, 1])
        mask = (distance > 1e-30) & (distance <= cap + reach) & (distance >= np.abs(cap - reach))
        if np.any(mask):
            d, s, q = distance[mask], reach[mask], p[mask]
            along = (cap * cap + d * d - s * s) / (2.0 * d)
            high = np.sqrt(np.maximum(0.0, cap * cap - along * along))
            unit = q / d[:, None]
            perpendicular = np.column_stack((-unit[:, 1], unit[:, 0]))
            candidates.extend((along[:, None] * unit + high[:, None] * perpendicular,
                               along[:, None] * unit - high[:, None] * perpendicular))
        if len(p) <= dense_limit:
            a, b = np.triu_indices(len(p), 1)
        else:
            links = cKDTree(p).query_pairs(2.0 * float(np.max(reach)), output_type="ndarray")
            a, b = links[:, 0], links[:, 1]
        delta = p[b] - p[a]
        distance = np.hypot(delta[:, 0], delta[:, 1])
        mask = (distance > 1e-30) & (distance <= reach[a] + reach[b]) & (distance >= np.abs(reach[a] - reach[b]))
        if np.any(mask):
            a, b, delta, distance = a[mask], b[mask], delta[mask], distance[mask]
            along = (reach[a] ** 2 - reach[b] ** 2 + distance ** 2) / (2.0 * distance)
            high = np.sqrt(np.maximum(0.0, reach[a] ** 2 - along ** 2))
            unit = delta / distance[:, None]
            base = p[a] + along[:, None] * unit
            perpendicular = np.column_stack((-unit[:, 1], unit[:, 0]))
            candidates.extend((base + high[:, None] * perpendicular, base - high[:, None] * perpendicular))
        candidates = np.concatenate(candidates)
        norm = np.hypot(candidates[:, 0], candidates[:, 1])
        candidates = candidates[norm <= cap + guard * 0.2]
        norm = np.hypot(candidates[:, 0], candidates[:, 1])
        candidates *= np.minimum(1.0, cap / np.maximum(norm, 1e-300))[:, None]
        projection = candidates[:, 0] * math.cos(direction) + candidates[:, 1] * math.sin(direction)
        candidates = candidates[np.argsort(projection)]
        best_point, best_value = candidates[0].copy(), math.inf
        batch = max(1, 32768 // (len(p) + 1))
        for begin in range(0, len(candidates), batch):
            if time.perf_counter() >= until:
                break
            trial = candidates[begin:begin + batch]
            delta = trial[:, None, :] - p[None, :, :]
            overlap = np.maximum(0.0, reach[None, :] - np.hypot(delta[:, :, 0], delta[:, :, 1]))
            good = np.flatnonzero(np.max(overlap, axis=1) <= guard * 0.2)
            if len(good):
                return trial[int(good[0])].copy(), True
            cost = np.einsum("ij,ij->i", overlap, overlap)
            j = int(np.argmin(cost))
            if cost[j] < best_value:
                best_value, best_point = float(cost[j]), trial[j].copy()
        return best_point, False

    def construct(k, order, until, complete):
        p = np.zeros((k, 2))
        chosen = []
        direction = float(rng.uniform(-math.pi, math.pi)) if complete else 0.0
        for j in order:
            if time.perf_counter() >= until:
                break
            chosen_array = np.asarray(chosen, dtype=int)
            point, fits = hole(p[chosen_array], r[chosen_array], r[j], until, direction)
            if point is None or (not fits and not complete):
                break
            p[j] = point
            chosen.append(int(j))
        if len(chosen) == k:
            return p
        if not complete:
            return p[:len(chosen)]
        missing = np.ones(k, dtype=bool)
        missing[chosen] = False
        angles = rng.uniform(0.0, 2.0 * math.pi, int(np.sum(missing)))
        lengths = np.sqrt(rng.random(len(angles))) * np.maximum(0.0, 1.0 - r[:k][missing])
        p[missing] = np.column_stack((np.cos(angles), np.sin(angles))) * lengths[:, None]
        return p

    construction_end = min(deadline, time.perf_counter() + budget * 0.12)
    if time.perf_counter() < construction_end:
        commit(construct(upper, range(upper), construction_end, False))

    def relax(seed, radii, until):
        k = len(seed)
        dense = k <= dense_limit
        if dense:
            a0, b0 = np.triu_indices(k, 1)
            sums = radii[a0] + radii[b0] + guard
        caps = 1.0 - radii - guard
        incumbent = seed.copy()
        minimum = math.inf

        def objective(flat):
            nonlocal incumbent, minimum
            if time.perf_counter() >= until:
                raise _StopLocal()
            p = flat.reshape(k, 2)
            length = np.hypot(p[:, 0], p[:, 1])
            wall = np.maximum(0.0, length - caps)
            gradient = (2.0 * wall / np.maximum(length, 1e-30))[:, None] * p
            value = float(np.dot(wall, wall))
            violation = float(np.max(wall))
            if dense:
                a, b, required = a0, b0, sums
            else:
                links = cKDTree(p).query_pairs(2.0 * float(np.max(radii)) + guard, output_type="ndarray")
                a, b = links[:, 0], links[:, 1]
                required = radii[a] + radii[b] + guard
            if len(a):
                delta = p[a] - p[b]
                distance = np.hypot(delta[:, 0], delta[:, 1])
                overlap = np.maximum(0.0, required - distance)
                value += float(np.dot(overlap, overlap))
                violation = max(violation, float(np.max(overlap)))
                active = overlap > 0.0
                if np.any(active):
                    aa, bb = a[active], b[active]
                    force = (2.0 * overlap[active] / np.maximum(distance[active], 1e-30))[:, None] * delta[active]
                    for dimension in (0, 1):
                        gradient[:, dimension] += np.bincount(bb, weights=force[:, dimension], minlength=k)
                        gradient[:, dimension] -= np.bincount(aa, weights=force[:, dimension], minlength=k)
            if value < minimum:
                incumbent, minimum = p.copy(), value
            if violation <= guard * 0.15:
                raise _StopLocal()
            return value, gradient.ravel()

        try:
            minimize(objective, np.asarray(seed).ravel(), method="L-BFGS-B", jac=True,
                     options={"ftol": 1e-15, "gtol": max(1e-12, guard * 0.003),
                              "maxiter": math.inf, "maxfun": math.inf,
                              "maxcor": max(5, min(30, int(math.sqrt(k) + effort)))})
        except Exception:
            pass
        return incumbent, minimum

    def perturb(p, radii, heat):
        k = len(p)
        p = p.copy()
        mean_radius = float(np.sqrt(np.mean(radii * radii)))
        depth = min(k, max(1, int((1.0 + effort) * math.sqrt(k) * (0.07 + 0.13 * heat / (1.0 + heat)))))
        mode = float(rng.random())
        if mode < 0.43:
            amplitude = mean_radius * (0.10 + 0.07 * heat) * float(rng.uniform(0.5, 1.5))
            p += rng.normal(0.0, amplitude, p.shape)
        elif mode < 0.80:
            stress = np.maximum(0.0, np.hypot(p[:, 0], p[:, 1]) + radii - 1.0) ** 2
            a, b = pairs(p, radii)
            if len(a):
                d = p[a] - p[b]
                cost = np.maximum(0.0, radii[a] + radii[b] - np.hypot(d[:, 0], d[:, 1])) ** 2
                stress += np.bincount(a, weights=cost, minlength=k) + np.bincount(b, weights=cost, minlength=k)
            stress += max(1e-30, float(np.sum(stress)) / max(k, 1) * 0.4)
            stress /= float(np.sum(stress))
            chosen = rng.choice(k, size=depth, replace=False, p=stress)
            samples = max(12, int(8.0 * (math.sqrt(k) + effort)))
            for j in chosen:
                if time.perf_counter() >= deadline:
                    break
                angles = rng.uniform(0.0, 2.0 * math.pi, samples)
                lengths = (1.0 - radii[j] - guard) * np.sqrt(rng.random(samples))
                lengths[:samples // 3] = max(0.0, 1.0 - radii[j] - guard)
                trial = np.column_stack((np.cos(angles), np.sin(angles))) * lengths[:, None]
                d = trial[:, None, :] - p[None, :, :]
                overlap = np.maximum(0.0, radii[j] + radii[None, :] + guard - np.hypot(d[:, :, 0], d[:, :, 1]))
                overlap[:, j] = 0.0
                costs = np.einsum("ij,ij->i", overlap, overlap)
                p[j] = trial[int(np.argmin(costs))]
        else:
            chosen = rng.choice(k, size=min(k, max(2, 2 * depth)), replace=False)
            p[chosen] = p[np.roll(chosen, 1)].copy()
            p += rng.normal(0.0, mean_radius * 0.06 * heat, p.shape)
        lengths = np.hypot(p[:, 0], p[:, 1])
        p *= np.minimum(1.0, np.maximum(0.0, 1.0 - radii - guard) / np.maximum(lengths, 1e-30))[:, None]
        return p

    target = -1
    population = []
    last_improvement = time.perf_counter()
    best_energy = math.inf
    while time.perf_counter() < deadline and len(best) < upper:
        k = len(best) + 1
        now = time.perf_counter()
        local_span = budget / (10.0 + math.sqrt(k))
        until = min(deadline, now + local_span)
        if k != target:
            target = k
            population = []
            best_energy = math.inf
            last_improvement = now
            point, fits = hole(best, r[:k - 1], r[k - 1], min(until, now + local_span * 0.35))
            if point is None:
                point = np.array([0.0, 0.0])
            seed = np.vstack((best, point))
            if fits and commit(seed):
                continue
            shrink = False
        else:
            heat = 1.0 + min(4.0, (now - last_improvement) / max(0.001, budget * 0.055))
            mode = float(rng.random())
            shrink = False
            if mode < 0.14:
                order = np.argsort(-(r[:k] * np.exp(rng.normal(0.0, 0.05 + 0.03 * heat, k))))
                seed = construct(k, order, min(until, now + local_span * 0.45), True)
                if commit(seed):
                    continue
            elif mode < 0.24 or not population:
                seed = lattice(k)[::-1].copy()
                extent = max(1e-30, float(np.max(np.hypot(seed[:, 0], seed[:, 1]))))
                seed *= (1.0 - float(np.mean(r[:k]))) / extent
                if rng.random() < 0.4:
                    rng.shuffle(seed)
                seed += rng.normal(0.0, float(np.mean(r[:k])) * 0.08, seed.shape)
            else:
                index = int((rng.random() ** 2) * len(population))
                seed = perturb(population[index][1], r[:k], heat)
                shrink = mode < 0.48
        if time.perf_counter() >= deadline:
            break
        if shrink:
            factor = 1.0 - float(rng.uniform(0.025, min(0.28, 0.08 + 0.025 * heat)))
            small_until = min(until, time.perf_counter() + local_span * 0.28)
            seed, _ = relax(seed, r[:k] * factor, small_until)
        candidate, energy = relax(seed, r[:k], until)
        if commit(candidate):
            continue
        if math.isfinite(energy):
            if energy < best_energy * (1.0 - 1e-5):
                best_energy = energy
                last_improvement = time.perf_counter()
            distinct = all(abs(energy - old[0]) > max(1e-20, 1e-7 * max(energy, old[0])) for old in population)
            if distinct or not population:
                population.append((energy, candidate.copy()))
                population.sort(key=lambda item: item[0])
                del population[population_size:]
