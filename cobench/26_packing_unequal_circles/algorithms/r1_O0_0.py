"""Prefix-incremental packing with geometric perturbation
Combines lattice and tangency construction, overlap-penalty relaxation, gap insertion and adaptive perturbation, saving within the time limit the longest feasible prefix that passed the geometric check.
"""

import math
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree


class _SearchStop(Exception):
    pass


class _PackingFound(Exception):
    pass


def solve(instance: dict, time_limit_s: float) -> dict:
    started = time.monotonic()
    answer = {"packing_status": [], "centers": []}
    try:
        n = max(0, int(instance["n"]))
        answer = {"packing_status": [0] * n,
                  "centers": [[-1.0, -1.0] for _ in range(n)]}
        budget = max(0.0, float(time_limit_s))
        cx, cy, radius = (float(instance[s]) for s in ("cx", "cy", "R"))
        original = np.asarray(instance["radii"], dtype=float)[:n]
        if n == 0 or len(original) != n or radius < 0:
            return answer
        if not np.all(np.isfinite(original)) or np.any(original < 0):
            return answer
        zeros = 0
        while zeros < n and original[zeros] == 0:
            zeros += 1
        for i in range(zeros):
            answer["packing_status"][i] = 1
            answer["centers"][i] = [cx, cy]
        if zeros == n or radius <= 0 or budget <= 0:
            return answer
        rr_all = original[zeros:] / radius
        upper, area = 0, 0.0
        for i, r in enumerate(rr_all):
            area += r * r
            if r > 1 or area > 1 + 1e-12:
                break
            if i and rr_all[i - 1] + r > 1 + 1e-12:
                break
            upper = i + 1
        if upper == 0:
            return answer
        rr_all = rr_all[:upper]
        reserve = min(budget * 0.07, 0.015 + 0.00015 * n)
        end = started + budget - reserve
        strength = math.log1p(budget)
        rng = np.random.default_rng(20261002)
        gap = min(1e-5, max(2e-9, 16 * np.finfo(float).eps *
                           (1 + abs(cx / radius) + abs(cy / radius))))
        dense_limit = max(32, int(32 * math.sqrt(1 + budget)))
        best_m, best_p = 0, np.empty((0, 2), dtype=float)
        angle_step = math.pi * (3 - math.sqrt(5))
        sample_count = max(24, int((12 + 3 * math.sqrt(upper)) * (1 + strength)))

        def pairs(p, cutoff):
            m = len(p)
            if m <= dense_limit:
                return np.triu_indices(m, 1)
            edges = cKDTree(p).query_pairs(float(cutoff), output_type="ndarray")
            return edges[:, 0], edges[:, 1]

        def keep(p):
            nonlocal best_m, best_p
            m = len(p)
            if m <= best_m or not np.all(np.isfinite(p)):
                return False
            r = original[zeros:zeros + m]
            world = p * radius + np.array([cx, cy])
            if not np.all(np.isfinite(world)):
                return False
            bad = np.flatnonzero(np.hypot(world[:, 0] - cx,
                                          world[:, 1] - cy) + r > radius)
            count = int(bad[0]) if len(bad) else m
            if count > best_m:
                # Recheck with the actual output coordinates rather than trusting only the optimiser's convergence flag.
                a, b = pairs(p[:count], 2 * float(rr_all[count - 1]) + 8 * gap)
                d = np.hypot(world[a, 0] - world[b, 0],
                             world[a, 1] - world[b, 1])
                bad_pairs = b[d < r[a] + r[b]]
                if len(bad_pairs):
                    count = min(count, int(np.min(bad_pairs)))
            if count <= best_m:
                return False
            zero_point = [cx, cy]
            if zeros:
                found = False
                for dx, dy in ((1., 0.), (-1., 0.), (0., 1.), (0., -1.)):
                    z = np.array([cx + radius * dx, cy + radius * dy])
                    if math.hypot(z[0] - cx, z[1] - cy) > radius:
                        z = np.nextafter(z, np.array([cx, cy]))
                    if (math.hypot(z[0] - cx, z[1] - cy) <= radius and
                            np.all(np.hypot(world[:count, 0] - z[0],
                                            world[:count, 1] - z[1]) >= r[:count])):
                        zero_point = z.tolist()
                        found = True
                        break
                if not found:
                    return False
            new_centers = ([list(zero_point) for _ in range(zeros)] +
                           world[:count].tolist() +
                           [[-1.0, -1.0] for _ in range(n - zeros - count)])
            answer["centers"] = new_centers
            answer["packing_status"] = ([1] * (zeros + count) +
                                         [0] * (n - zeros - count))
            best_m, best_p = count, p[:count].copy()
            return True

        # The diameter arrangement provides an initial feasible solution without numeric optimisation and also covers the critical two-circle case.
        keep(np.zeros((1, 2)))
        sums = np.cumsum(rr_all)
        line_m = int(np.searchsorted(sums, 1.0, side="right"))
        if line_m:
            radii = rr_all[:line_m].copy()
            slack = max(0.0, 1 - float(np.sum(radii)))
            radii += min(gap, slack / (2 * line_m))
            p = np.zeros((line_m, 2))
            p[:, 0] = 2 * np.cumsum(radii) - radii - np.sum(radii)
            keep(p)
        if best_m >= upper or time.monotonic() >= end:
            return answer

        def lattice(m, shifted=False):
            width = int(math.ceil(math.sqrt(m)))
            yy, xx = np.mgrid[-width:width + 1, -width:width + 1]
            spacing = float(rr_all[m - 1]) + 2 * gap
            x = (2 * xx + (yy % 2)) * spacing
            y = math.sqrt(3) * yy * spacing
            if shifted:
                x = x + rng.uniform(-1, 1) * spacing
                y = y + rng.uniform(-1, 1) * spacing
            p = np.column_stack((x.ravel(), y.ravel()))
            order = np.argsort(np.sum(p * p, axis=1))[:m]
            return p[order[::-1]].copy()

        # This bisection is only over the count the lattice construction can certify; it is not a proof of problem infeasibility.
        lo, hi = best_m, upper
        lattice_end = min(end, started + 0.06 * budget)
        while lo < hi and time.monotonic() < lattice_end:
            m = (lo + hi + 1) // 2
            p = lattice(m)
            good = bool(np.all(np.hypot(p[:, 0], p[:, 1]) + rr_all[:m] <= 1))
            if good:
                keep(p)
                lo = m
            else:
                hi = m - 1
        if best_m >= upper:
            return answer

        def contact_candidates(p, r, new_r):
            m = len(p)
            bound = max(0.0, 1 - new_r - gap)
            t = np.arange(max(12, int(6 * (1 + strength))))
            theta = t * (2 * math.pi / len(t))
            candidates = [np.column_stack((bound * np.cos(theta),
                                           bound * np.sin(theta))),
                          np.zeros((1, 2))]
            if not m:
                return np.vstack(candidates)
            d = np.hypot(p[:, 0], p[:, 1])
            b = r + new_r + 2 * gap
            mask = (d > 1e-14) & (d <= bound + b) & (d >= np.abs(bound - b))
            q, distance, radii = p[mask], d[mask], b[mask]
            if len(q):
                foot = (bound * bound + distance * distance - radii * radii) / (2 * distance)
                height = np.sqrt(np.maximum(0, bound * bound - foot * foot))
                unit = q / distance[:, None]
                perp = np.column_stack((-unit[:, 1], unit[:, 0]))
                candidates += [unit * foot[:, None] + perp * height[:, None],
                               unit * foot[:, None] - perp * height[:, None]]
            a, bidx = pairs(p, 2 * (float(np.max(r)) + new_r + 2 * gap))
            delta = p[bidx] - p[a]
            d = np.hypot(delta[:, 0], delta[:, 1])
            ra, rb = r[a] + new_r + 2 * gap, r[bidx] + new_r + 2 * gap
            mask = (d > 1e-14) & (d <= ra + rb) & (d >= np.abs(ra - rb))
            a, delta, d, ra, rb = a[mask], delta[mask], d[mask], ra[mask], rb[mask]
            if len(a):
                foot = (ra * ra + d * d - rb * rb) / (2 * d)
                height = np.sqrt(np.maximum(0, ra * ra - foot * foot))
                unit = delta / d[:, None]
                mid = p[a] + unit * foot[:, None]
                perp = np.column_stack((-unit[:, 1], unit[:, 0])) * height[:, None]
                candidates += [mid + perp, mid - perp]
            return np.vstack(candidates)

        # Small-to-large tangency construction always keeps a prefix; after construction fails, continuous search takes over.
        constructive_end = min(end, started + 0.16 * budget)
        placed = []
        while len(placed) < upper and time.monotonic() < constructive_end:
            m = len(placed)
            p = np.asarray(placed, dtype=float).reshape((-1, 2))
            r = rr_all[:m]
            nr = float(rr_all[m])
            candidates = contact_candidates(p, r, nr)
            inside = np.hypot(candidates[:, 0], candidates[:, 1]) + nr <= 1 - 0.3 * gap
            candidates = candidates[inside]
            candidates = candidates[np.argsort(candidates[:, 0] + 0.013 * candidates[:, 1])]
            found = None
            block = max(1, 32768 // (m + 1))
            for first in range(0, len(candidates), block):
                if time.monotonic() >= constructive_end:
                    break
                q = candidates[first:first + block]
                d = np.hypot(q[:, None, 0] - p[None, :, 0],
                             q[:, None, 1] - p[None, :, 1])
                feasible = np.all(d >= r[None, :] + nr + 0.5 * gap, axis=1)
                good = np.flatnonzero(feasible)
                if len(good):
                    found = q[good[0]]
                    break
            if found is None:
                break
            placed.append(found)
        if placed:
            keep(np.asarray(placed))
        if best_m >= upper or time.monotonic() >= end:
            return answer

        def hole(r, other_p, other_r, stop):
            bound = max(0.0, 1 - r - 2 * gap)
            ids = np.arange(sample_count)
            theta = ids * angle_step + rng.uniform(0, 2 * math.pi)
            lengths = bound * np.sqrt((ids + 0.5) / sample_count)
            q = np.column_stack((lengths * np.cos(theta), lengths * np.sin(theta)))
            outer = max(8, int(math.sqrt(sample_count) * (1 + strength)))
            theta = np.arange(outer) * (2 * math.pi / outer) + rng.uniform(0, 2 * math.pi)
            q = np.vstack((q, np.column_stack((bound * np.cos(theta), bound * np.sin(theta))),
                           np.zeros((1, 2))))
            if not len(other_p):
                return q[-1].copy()
            chosen, value = q[0], float("inf")
            block = max(1, 32768 // len(other_p))
            for first in range(0, len(q), block):
                if time.monotonic() >= stop:
                    break
                candidates = q[first:first + block]
                d = np.hypot(candidates[:, None, 0] - other_p[None, :, 0],
                             candidates[:, None, 1] - other_p[None, :, 1])
                overlap = np.maximum(0, r + other_r[None, :] + 2 * gap - d)
                energies = np.sum(overlap * overlap, axis=1)
                index = int(np.argmin(energies))
                if energies[index] < value:
                    value, chosen = float(energies[index]), candidates[index]
            return chosen.copy()

        def seed(m, base, stop, structured=False):
            r = rr_all[:m]
            if structured:
                p = lattice(m, shifted=True)
                lengths = np.hypot(p[:, 0], p[:, 1])
                p *= np.minimum(1, np.maximum(0, 1 - r - gap) /
                                np.maximum(lengths, 1e-100))[:, None]
                return p
            theta = rng.uniform(0, 2 * math.pi, m)
            lengths = np.sqrt(rng.random(m)) * np.maximum(0, 1 - r - gap)
            p = np.column_stack((lengths * np.cos(theta), lengths * np.sin(theta)))
            used = np.zeros(m, dtype=bool)
            if base is not None:
                b = min(len(base), m)
                p[:b], used[:b] = base[:b], True
            for i in range(m - 1, -1, -1):
                if time.monotonic() >= stop:
                    break
                if not used[i]:
                    p[i] = hole(float(r[i]), p[used], r[used], stop)
                    used[i] = True
            return p

        def relax(p, bound, stop):
            m = len(p)
            r = rr_all[:m]
            dense = m <= dense_limit
            fixed = np.triu_indices(m, 1) if dense else None
            cutoff = 2 * float(r[-1]) + 4 * gap
            cache_p, cache_value = p.copy(), float("inf")
            found = False

            def energy(z):
                nonlocal cache_p, cache_value, found
                if time.monotonic() >= stop:
                    raise _SearchStop()
                q = z.reshape((m, 2))
                a, b = fixed if dense else pairs(q, cutoff)
                dx, dy = q[a, 0] - q[b, 0], q[a, 1] - q[b, 1]
                d = np.hypot(dx, dy)
                u = np.maximum(0, r[a] + r[b] + 2 * gap - d)
                factor = -2 * u / np.maximum(d, 1e-100)
                gx, gy = factor * dx, factor * dy
                gradient = np.empty_like(q)
                gradient[:, 0] = (np.bincount(a, weights=gx, minlength=m) -
                                  np.bincount(b, weights=gx, minlength=m))
                gradient[:, 1] = (np.bincount(a, weights=gy, minlength=m) -
                                  np.bincount(b, weights=gy, minlength=m))
                distance = np.hypot(q[:, 0], q[:, 1])
                v = np.maximum(0, distance + r + gap - bound)
                gradient += (4 * v / np.maximum(distance, 1e-100))[:, None] * q
                value = float(np.dot(u, u) + 2 * np.dot(v, v))
                if value < cache_value:
                    cache_p, cache_value = q.copy(), value
                if bound <= 1 and max(float(np.max(u, initial=0)),
                                      float(np.max(v, initial=0))) < 0.5 * gap:
                    cache_p, cache_value, found = q.copy(), value, True
                    raise _PackingFound()
                return value, gradient.ravel()

            try:
                minimize(energy, p.ravel(), jac=True, method="L-BFGS-B",
                         options={"ftol": 0.0, "gtol": max(1e-13, 0.02 * gap),
                                  # Remove the library's iteration/evaluation caps; the deadline is checked at each evaluation.
                                  "maxiter": math.inf, "maxfun": math.inf,
                                  "maxcor": max(3, min(32, int((1 + strength) * math.log1p(m)))),
                                  "maxls": max(20, int(8 * (1 + strength)))})
            except (_SearchStop, _PackingFound):
                pass
            return cache_p, cache_value, found

        target_guess, accumulated = 1, 0.0
        for i, r in enumerate(rr_all):
            accumulated += r * r
            if accumulated <= 0.91 * (1 - 0.45 * r) ** 2:
                target_guess = i + 1
        target = min(upper, max(best_m + 1, target_guess))
        champion, current = None, None
        champion_value = float("inf")
        stage_started = last_improved = time.monotonic()
        stage_window = max(0, end - stage_started) / (2 + math.sqrt(target - best_m))

        # No cap on total restarts or perturbations; only the wall clock and a proven count upper bound end the search.
        while time.monotonic() < end and best_m < upper:
            now = time.monotonic()
            remaining = end - now
            if target <= best_m:
                target = best_m + 1
                champion = current = None
                champion_value = float("inf")
                stage_started = last_improved = now
                stage_window = remaining / (2 + math.sqrt(target - best_m))
            if target > best_m + 1 and now - stage_started >= stage_window:
                target = max(best_m + 1, (target + best_m + 1) // 2)
                champion = champion[:target].copy() if champion is not None else None
                current = None
                champion_value = float("inf")
                stage_started = last_improved = now
                stage_window = remaining / (2 + math.sqrt(target - best_m))
            span = min(remaining, budget / (12 + math.sqrt(target)))
            local_end = min(end, now + span)
            r = rr_all[:target]
            if champion is None:
                base = best_p if best_m * 3 >= target * 2 else None
                p = seed(target, base, now + 0.3 * span)
            elif rng.random() < 1 / (3 + strength):
                p = seed(target, None, now + 0.3 * span, structured=bool(rng.integers(2)))
            else:
                p = (current if current is not None and rng.random() < 0.35 else champion).copy()
                stagnation = max(0.0, now - last_improved) / max(span, stage_window)
                depth = min(target, max(1, int(1 + strength * math.sqrt(stagnation))))
                mode = rng.random()
                if mode < 0.36:
                    a, b = pairs(p, 2 * float(r[-1]) + 4 * gap)
                    d = np.hypot(p[a, 0] - p[b, 0], p[a, 1] - p[b, 1])
                    overlaps = np.maximum(0, r[a] + r[b] + 2 * gap - d)
                    trouble = (np.bincount(a, weights=overlaps, minlength=target) +
                               np.bincount(b, weights=overlaps, minlength=target) +
                               np.maximum(0, np.hypot(p[:, 0], p[:, 1]) + r - 1))
                    trouble += max(float(np.max(trouble)), gap) / (2 + strength)
                    selected = rng.choice(target, size=depth, replace=False, p=trouble / trouble.sum())
                    mask = np.ones(target, dtype=bool)
                    for i in selected:
                        if time.monotonic() >= now + 0.35 * span:
                            break
                        mask[i] = False
                        p[i] = hole(float(r[i]), p[mask], r[mask], now + 0.35 * span)
                        mask[i] = True
                elif mode < 0.62 and target > 1:
                    for _ in range(depth):
                        i, j = rng.choice(target, 2, replace=False)
                        p[[i, j]] = p[[j, i]]
                    p += rng.normal(size=p.shape) * float(np.median(r)) * 0.025
                else:
                    amplitude = float(np.median(r)) * (0.15 + rng.random())
                    amplitude *= 1 + min(strength, math.sqrt(stagnation))
                    p += rng.normal(size=p.shape) * amplitude
                if rng.random() < 0.15 + 0.15 * stagnation / (1 + stagnation):
                    expansion = 1 + float(np.median(r)) * (0.2 + rng.random())
                    p, _, _ = relax(p, expansion, min(local_end, now + span * 0.45))
            p, value, feasible = relax(p, 1.0, local_end)
            previous_best = best_m
            keep(p)
            if best_m >= target:
                continue
            if value < champion_value:
                champion, champion_value = p.copy(), value
                current = p.copy()
                last_improved = time.monotonic()
            elif current is None or rng.random() < 1 / (2 + strength):
                current = p.copy()
            if best_m > previous_best:
                stage_window = min(stage_window, max(0, end - time.monotonic()) /
                                   (2 + math.sqrt(target - best_m)))
    except Exception:
        # If any numeric routine fails, deliver the solution previously checked and fully saved.
        pass
    return answer
