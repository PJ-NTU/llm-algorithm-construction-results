"""Tangent-point construction and elastic basin search for prefix circle packing
Combines triangular lattice and tangent-point construction, eliminates overlaps with analytic gradients, and grows the feasible prefix through swaps, reinsertion and adaptive perturbation.
"""

import math
import time

try:
    import numpy as np
    from scipy.optimize import minimize
    from scipy.spatial import cKDTree
except Exception:
    np = None


class _Stop(Exception):
    pass


def solve(instance: dict, time_limit_s: float) -> dict:
    start = time.perf_counter()
    try:
        n = max(0, int(instance["n"]))
    except Exception:
        n = 0
    answer = {"packing_status": [0] * n,
              "centers": [[-1.0, -1.0] for _ in range(n)]}
    try:
        budget = max(0.0, float(time_limit_s))
        if not math.isfinite(budget):
            budget = 0.0
        deadline = start + budget - min(0.08, budget * 0.025)
        cx, cy, radius = (float(instance[t]) for t in ("cx", "cy", "R"))
        original = [float(v) for v in instance["radii"]]
        if n == 0 or len(original) != n:
            return answer
        if not all(math.isfinite(v) for v in (cx, cy, radius)):
            return answer
        if radius < 0 or any(not math.isfinite(v) or v < 0 for v in original):
            return answer
        if original[0] <= radius:
            answer["packing_status"][0] = 1
            answer["centers"][0] = [cx, cy]
        if radius == 0:
            for i, r in enumerate(original):
                if r != 0:
                    break
                answer["packing_status"][i] = 1
                answer["centers"][i] = [cx, cy]
            return answer
        if np is None or time.perf_counter() >= deadline:
            return answer
        _search(original, radius, cx, cy, budget, start, deadline, answer)
    except Exception:
        # Whether or not the numeric solve is interrupted, the committed feasible prefix is always kept.
        pass
    return answer


def _search(original, radius, cx, cy, budget, start, deadline, answer):
    clock = time.perf_counter
    n = len(original)
    raw = np.asarray(original, dtype=float)
    r = raw / radius
    if not np.all(np.isfinite(r)):
        return
    upper = 0
    area = 0.0
    for i in range(n):
        if clock() >= deadline:
            return
        area += float(r[i] * r[i])
        if r[i] > 1.0 or area > 1.0 + 2e-14:
            break
        if i and r[i - 1] + r[i] > 1.0 + 2e-14:
            break
        upper = i + 1
    if upper == 0:
        return
    r = r[:upper]
    guard = max(2e-9, 8.0 * max(math.ulp(cx), math.ulp(cy)) / radius)
    effort = math.log1p(budget)
    rng = np.random.default_rng(20261003 + n * 313)
    best = np.zeros((int(answer["packing_status"][0]), 2), dtype=float)

    def pairs(p, cutoff):
        m = len(p)
        if m <= 224:
            return np.triu_indices(m, 1)
        ij = cKDTree(p).query_pairs(float(cutoff), output_type="ndarray")
        if len(ij) == 0:
            return np.empty(0, dtype=int), np.empty(0, dtype=int)
        return ij[:, 0], ij[:, 1]

    def accept(p):
        nonlocal best
        k = len(p)
        if k <= len(best) or clock() >= deadline or not np.all(np.isfinite(p)):
            return False
        world = p * radius + np.array([cx, cy])
        if not np.all(np.isfinite(world)):
            return False
        if np.any(np.hypot(world[:, 0] - cx, world[:, 1] - cy) + raw[:k] > radius):
            return False
        actual = (world - np.array([cx, cy])) / radius
        ii, jj = pairs(actual, 2.0 * float(r[k - 1]) + 4.0 * guard)
        if len(ii):
            d = np.hypot(world[ii, 0] - world[jj, 0], world[ii, 1] - world[jj, 1])
            if np.any(d < raw[ii] + raw[jj]):
                return False
        coordinates = world.tolist()
        # Update the externally visible solution only after all geometric checks pass.
        answer["centers"][:k] = coordinates
        answer["packing_status"][:k] = [1] * k
        best = actual.copy()
        return True

    # A chain along the diameter provides a cheap deterministic backup solution, with an extra check for the exact tangency case.
    for gap in (guard * 2.0, 0.0):
        total = 0.0
        count = 0
        for ri in r:
            new_total = total + float(ri) + (gap * 0.5 if count else 0.0)
            if new_total > 1.0 - (guard if gap else 0.0):
                break
            total = new_total
            count += 1
        if count > len(best):
            q = np.zeros((count, 2))
            left = -total
            for i in range(count):
                q[i, 0] = left + r[i]
                left += 2.0 * r[i] + gap
            accept(q)
    if len(best) >= upper or clock() >= deadline:
        return

    # Use a triangular lattice with the same safety spacing to quickly handle many small circles.
    init_end = min(deadline, start + budget * 0.14)
    indices = np.arange(1, upper + 1, dtype=float)
    admissible = np.flatnonzero((np.sqrt(indices * (2.0 * math.sqrt(3.0) / math.pi)) + 1.0)
                               * (r + guard) <= 1.0 - guard)
    if len(admissible) and clock() < init_end:
        k = int(admissible[-1]) + 1
        a = float(r[k - 1]) + guard
        if k > len(best) and a > 0.0:
            dx, dy = 2.0 * a, math.sqrt(3.0) * a
            extent = min(1.0 - a - guard, dx * math.sqrt(k + 1.0))
            rows = int(extent / dy)
            points = []
            for row in range(-rows, rows + 1):
                if clock() >= init_end:
                    break
                y = row * dy
                width = math.sqrt(max(0.0, extent * extent - y * y))
                offset = (row % 2) * a
                lo = math.ceil((-width - offset) / dx)
                hi = math.floor((width - offset) / dx)
                if hi >= lo:
                    xx = np.arange(lo, hi + 1, dtype=float) * dx + offset
                    points.append(np.column_stack((xx, np.full(len(xx), y))))
            if points:
                q = np.concatenate(points)
                order = np.argsort(np.sum(q * q, axis=1))
                accept(q[order[:k]])
    if len(best) >= upper:
        return

    def insertion(p, rr, new_r, until, direction=None):
        """Choose a low-conflict position among container-boundary points, two-circle tangent points and a few interior candidates."""
        if clock() >= until:
            return None, False
        cap = 1.0 - float(new_r) - 2.0 * guard
        if cap < 0.0:
            return np.zeros(2), False
        m = len(p)
        if not m:
            return np.array([0.0, -cap]), True
        sample_count = max(12, int((2.0 + effort) * math.sqrt(m + 1)))
        angle = np.arange(sample_count) * (2.0 * math.pi / sample_count)
        candidate = [np.column_stack((cap * np.cos(angle), cap * np.sin(angle))),
                     np.zeros((1, 2))]
        radii = rr + new_r + 2.0 * guard
        d = np.hypot(p[:, 0], p[:, 1])
        mask = (d > 1e-15) & (d <= cap + radii) & (d >= np.abs(cap - radii))
        if np.any(mask):
            v, dd, ss = p[mask], d[mask], radii[mask]
            aa = (cap * cap - ss * ss + dd * dd) / (2.0 * dd)
            hh = np.sqrt(np.maximum(0.0, cap * cap - aa * aa))
            base = v * (aa / dd)[:, None]
            off = np.column_stack((-v[:, 1], v[:, 0])) * (hh / dd)[:, None]
            candidate.extend((base + off, base - off))
        ii, jj = pairs(p, 2.0 * float(np.max(radii)))
        if len(ii):
            v = p[jj] - p[ii]
            dd = np.hypot(v[:, 0], v[:, 1])
            mask = ((dd > 1e-15) & (dd <= radii[ii] + radii[jj]) &
                    (dd >= np.abs(radii[ii] - radii[jj])))
            ii, jj, v, dd = ii[mask], jj[mask], v[mask], dd[mask]
            if len(ii):
                aa = (radii[ii] ** 2 - radii[jj] ** 2 + dd ** 2) / (2.0 * dd)
                hh = np.sqrt(np.maximum(0.0, radii[ii] ** 2 - aa ** 2))
                base = p[ii] + v * (aa / dd)[:, None]
                off = np.column_stack((-v[:, 1], v[:, 0])) * (hh / dd)[:, None]
                candidate.extend((base + off, base - off))
        c = np.concatenate(candidate)
        c = c[np.hypot(c[:, 0], c[:, 1]) <= cap + guard * 0.1]
        if len(c) == 0:
            return np.zeros(2), False
        if direction is None:
            direction = np.array([1e-5, 1.0])
        chosen = c[0].copy()
        low_energy, low_rank = float("inf"), float("inf")
        found = False
        block = max(1, min(len(c), 32768 // max(1, m)))
        for begin in range(0, len(c), block):
            if clock() >= until:
                break
            cc = c[begin:begin + block]
            distances = np.hypot(cc[:, None, 0] - p[None, :, 0],
                                 cc[:, None, 1] - p[None, :, 1])
            violation = np.maximum(0.0, rr[None, :] + new_r + guard - distances)
            energy = np.sum(violation * violation, axis=1)
            rank = cc[:, 0] * direction[0] + cc[:, 1] * direction[1]
            feasible = np.flatnonzero(energy <= guard * guard * 0.01)
            if len(feasible):
                j = int(feasible[np.argmin(rank[feasible])])
                if not found or rank[j] < low_rank:
                    chosen, low_rank = cc[j].copy(), float(rank[j])
                found = True
            elif not found:
                j = int(np.argmin(energy))
                if energy[j] < low_energy:
                    chosen, low_energy = cc[j].copy(), float(energy[j])
        return chosen, found

    # Tangent-point construction adds circles by index and does not break the existing prefix on failure.
    greedy = np.empty((0, 2))
    while len(greedy) < upper and clock() < init_end:
        k = len(greedy)
        point, feasible = insertion(greedy, r[:k], r[k], init_end)
        if point is None or not feasible:
            break
        greedy = np.vstack((greedy, point))
    accept(greedy)
    if len(best) >= upper or clock() >= deadline:
        return

    def relax(p, rr, until, wall=1.0):
        """Squared-penetration potential and its analytic gradient; a wall-clock interruption returns the lowest-potential point visited."""
        k = len(rr)
        fixed = np.triu_indices(k, 1) if k <= 224 else None
        caps = np.maximum(0.0, wall - rr - 2.0 * guard)
        cutoff = 2.0 * float(np.max(rr)) + 4.0 * guard
        low = [float("inf"), p.copy()]

        def fun(z):
            if clock() >= until:
                raise _Stop()
            q = z.reshape(k, 2)
            ii, jj = fixed if fixed is not None else pairs(q, cutoff)
            xx, yy = q[:, 0], q[:, 1]
            norm = np.hypot(xx, yy)
            boundary = np.maximum(0.0, norm - caps)
            mult = 2.0 * boundary / np.maximum(norm, 1e-30)
            gx, gy = mult * xx, mult * yy
            energy = float(np.sum(boundary * boundary))
            if len(ii):
                dx, dy = xx[ii] - xx[jj], yy[ii] - yy[jj]
                distances = np.hypot(dx, dy)
                overlap = np.maximum(0.0, rr[ii] + rr[jj] + 2.0 * guard - distances)
                active = overlap > 0.0
                ia, ja = ii[active], jj[active]
                dd = distances[active]
                vx, vy = dx[active].copy(), dy[active].copy()
                coincident = dd < 1e-25
                if np.any(coincident):
                    theta = (ia[coincident] * 2.399963229728653 + ja[coincident])
                    vx[coincident], vy[coincident] = np.cos(theta), np.sin(theta)
                    dd = dd.copy()
                    dd[coincident] = 1.0
                force = -2.0 * overlap[active] / np.maximum(dd, 1e-30)
                fx, fy = force * vx, force * vy
                gx += np.bincount(ia, weights=fx, minlength=k) - np.bincount(ja, weights=fx, minlength=k)
                gy += np.bincount(ia, weights=fy, minlength=k) - np.bincount(ja, weights=fy, minlength=k)
                energy += float(np.sum(overlap * overlap))
            if energy < low[0]:
                low[0], low[1] = energy, q.copy()
            if energy <= guard * guard * 0.005:
                raise _Stop()
            return energy, np.column_stack((gx, gy)).ravel()

        try:
            minimize(fun, p.ravel(), method="L-BFGS-B", jac=True,
                     options={"ftol": 1e-15, "gtol": 1e-11,
                              "maxiter": math.inf, "maxfun": math.inf,
                              "maxcor": max(5, int(5 + 3 * effort))})
        except Exception:
            pass
        return low[1], low[0]

    def fresh(rr, until):
        k = len(rr)
        angle = rng.uniform(0.0, 2.0 * math.pi, k)
        norm = np.sqrt(rng.random(k)) * np.maximum(0.0, 1.0 - rr - 3.0 * guard)
        q = np.column_stack((norm * np.cos(angle), norm * np.sin(angle)))
        # Large circles first, but the order is slightly randomised to change the contact structure.
        order = np.argsort(-(rr + guard) * (1.0 + rng.random(k) * (0.2 + effort * 0.08)))
        done = []
        theta = rng.uniform(0.0, 2.0 * math.pi)
        direction = np.array([math.cos(theta), math.sin(theta)])
        for i in order:
            if clock() >= until:
                break
            point, _ = insertion(q[done], rr[done], rr[i], until, direction)
            if point is None:
                break
            q[i] = point
            done.append(int(i))
        return q

    target = -1
    current = None
    current_energy = float("inf")
    pool = []
    last_improvement = clock()
    while clock() < deadline and len(best) < upper:
        k = len(best) + 1
        rr = r[:k]
        mean_r = max(float(np.mean(rr)), 1e-10)
        base_slice = budget / (24.0 + 2.0 * math.sqrt(k))
        now = clock()
        if target != k:
            target = k
            point, _ = insertion(best, rr[:-1], rr[-1], min(deadline, now + base_slice * 0.3))
            if point is None:
                point = np.array([0.0, -(1.0 - rr[-1])])
            current = np.vstack((best, point))
            current_energy = float("inf")
            pool = []
            last_improvement = now
            q = current.copy()
        else:
            stalled = max(0.0, now - last_improvement)
            scale = 0.15 + effort * 0.06 + min(1.3, stalled / max(base_slice * (2.0 + effort), 1e-9))
            restart_probability = min(0.35, 0.06 + stalled / max(budget, 1e-9) * 0.5)
            mode = rng.random()
            if mode < restart_probability:
                if rng.random() < 0.7:
                    q = fresh(rr, min(deadline, now + base_slice * 0.45))
                else:
                    theta = np.arange(k) * 2.399963229728653 + rng.uniform(0.0, 6.283185307179586)
                    rho = np.sqrt((rng.permutation(k) + 0.5) / k) * (1.0 - rr)
                    q = np.column_stack((rho * np.cos(theta), rho * np.sin(theta)))
            else:
                if pool and rng.random() < 0.65:
                    parent = 0 if rng.random() < 0.65 else int(rng.integers(len(pool)))
                    q = pool[parent][1].copy()
                else:
                    q = current.copy()
                choice = rng.random()
                if choice < 0.26 and k > 1:
                    count = min(k, max(2, int(2 + rng.random() * math.sqrt(k) * (0.3 + effort * 0.1))))
                    chosen = rng.choice(k, count, replace=False)
                    q[chosen] = q[np.roll(chosen, 1)]
                    q[chosen] += rng.normal(0.0, mean_r * 0.05, (count, 2))
                elif choice < 0.48 and k > 1:
                    ii, jj = pairs(q, 2.0 * float(rr[-1]) + 4.0 * guard)
                    d = np.hypot(q[ii, 0] - q[jj, 0], q[ii, 1] - q[jj, 1])
                    overlap = np.maximum(0.0, rr[ii] + rr[jj] - d)
                    loads = np.bincount(ii, weights=overlap, minlength=k) + np.bincount(jj, weights=overlap, minlength=k)
                    j = int(np.argmax(loads)) if rng.random() < 0.65 else int(rng.integers(k))
                    keep = np.arange(k) != j
                    theta = rng.uniform(0.0, 6.283185307179586)
                    point, _ = insertion(q[keep], rr[keep], rr[j],
                                         min(deadline, clock() + base_slice * 0.25),
                                         np.array([math.cos(theta), math.sin(theta)]))
                    if point is not None:
                        q[j] = point
                    if rng.random() < 0.5:
                        q[j] = (1.0 - rr[j] - 3.0 * guard) * np.array([math.cos(theta), math.sin(theta)])
                    q += rng.normal(0.0, mean_r * scale * 0.08, q.shape)
                else:
                    count = min(k, max(1, int(k * min(1.0, 0.1 + scale * rng.random()))))
                    chosen = rng.choice(k, count, replace=False)
                    q[chosen] += rng.normal(0.0, mean_r * scale, (count, 2))
                    if choice > 0.9:
                        q *= rng.uniform(0.8, 1.05)
                norms = np.hypot(q[:, 0], q[:, 1])
                caps = np.maximum(0.0, 1.0 - rr - 3.0 * guard)
                q *= np.minimum(1.0, caps / np.maximum(norms, 1e-30))[:, None]
        now = clock()
        stalled = max(0.0, now - last_improvement)
        local_end = min(deadline, now + base_slice * (1.0 + math.log1p(stalled / max(base_slice, 1e-9))))
        # Occasionally increase container pressure to change the local contact structure, then restore the original container.
        if pool and rng.random() < min(0.25, 0.04 * effort):
            q, _ = relax(q, rr, min(local_end, now + (local_end - now) * 0.3),
                         wall=1.0 - rng.uniform(0.01, 0.04 + effort * 0.015))
        q, energy = relax(q, rr, local_end)
        if accept(q):
            continue
        if not math.isfinite(energy):
            continue
        if not pool or energy < pool[0][0]:
            last_improvement = clock()
        capacity = max(2, int(2.0 + math.sqrt(max(effort, 0.0)) * math.log1p(k) * 0.5))
        if not pool or energy < pool[-1][0] or len(pool) < capacity:
            if not any(abs(energy - e) <= 1e-13 * max(1.0, energy) for e, _ in pool):
                pool.append((energy, q.copy()))
                pool.sort(key=lambda item: item[0])
                del pool[capacity:]
        heat = max(mean_r * mean_r * (0.003 + 0.003 * effort),
                   (pool[0][0] if pool else energy) * 0.3, 1e-18)
        if energy <= current_energy or rng.random() < math.exp(min(0.0, (current_energy - energy) / heat)):
            current, current_energy = q.copy(), energy
