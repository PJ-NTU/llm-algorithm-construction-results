"""Contact-front and global-relaxation fusion packing
Constructs and resets layouts using boundary and two-circle contact points, combined with incremental insertion, global continuous relaxation and projection repair, maintaining a verified feasible prefix within the wall clock.
"""

import math
import random
import time

import numpy as np
from scipy.optimize import minimize


def solve(instance: dict, time_limit_s: float) -> dict:
    start = time.perf_counter()
    n = 0
    answer = {"packing_status": [], "centers": []}
    try:
        n = max(0, int(instance["n"]))
        answer = {"packing_status": [0] * n,
                  "centers": [[-1.0, -1.0] for _ in range(n)]}
        budget = float(time_limit_s)
        if not math.isfinite(budget):
            budget = 0.0
        budget = max(0.0, budget)
        # Reserve a budget-scaled margin for output and the final geometric check.
        deadline = start + budget * 0.965
        cx, cy = float(instance["cx"]), float(instance["cy"])
        radius = float(instance["R"])
        raw = [float(instance["radii"][i]) for i in range(n)]
        if (not n or not all(math.isfinite(v) for v in [cx, cy, radius])
                or radius < 0.0
                or any(not math.isfinite(v) or v < 0.0 for v in raw)):
            return answer
        if radius == 0.0:
            for i, r in enumerate(raw):
                if r > 0.0:
                    break
                answer["packing_status"][i] = 1
                answer["centers"][i] = [cx, cy]
            return answer

        r = np.asarray(raw, dtype=float) / radius
        if raw[0] > radius:
            return answer
        best_count = 1
        best_xy = np.zeros((1, 2))
        answer["packing_status"][0] = 1
        answer["centers"][0] = [cx, cy]
        if time.perf_counter() >= deadline:
            return answer

        gap = max(2.0e-9, 16.0 * max(math.ulp(cx), math.ulp(cy),
                                     math.ulp(radius)) / radius)
        effort = math.log1p(budget)
        rng = random.Random(71933 + n * 101 +
                            sum(int(min(1.0, v) * 100003) for v in r))
        tau = 2.0 * math.pi
        upper = 0
        area = 0.0
        largest = 0.0
        for i in range(n):
            area += r[i] * r[i]
            if (r[i] > 1.0 or area > 1.0 + 1.0e-14
                    or (i and largest + r[i] > 1.0 + 1.0e-14)):
                break
            upper = i + 1
            largest = max(largest, r[i])

        def promote(xy):
            """Check the output coordinates directly, without using a tolerance to let intersections or boundary violations pass."""
            nonlocal best_count, best_xy, answer
            k = len(xy)
            if k <= best_count or time.perf_counter() >= deadline:
                return False
            world = []
            for i in range(k):
                if time.perf_counter() >= deadline:
                    return False
                x = cx + float(xy[i, 0]) * radius
                y = cy + float(xy[i, 1]) * radius
                if (not math.isfinite(x) or not math.isfinite(y)
                        or math.hypot(x - cx, y - cy) + raw[i] > radius):
                    return False
                for j, (a, b) in enumerate(world):
                    if math.hypot(x - a, y - b) < raw[i] + raw[j]:
                        return False
                world.append([x, y])
            best_count = k
            best_xy = np.array(xy, dtype=float, copy=True)
            answer = {"packing_status": [1] * k + [0] * (n - k),
                      "centers": world + [[-1.0, -1.0] for _ in range(n - k)]}
            return True

        # The arrangement along the diameter provides an initial feasible prefix without search.
        total = 0.0
        line_count = 0
        for i in range(upper):
            if total + r[i] + gap > 1.0 - gap:
                break
            total += r[i] + gap
            line_count = i + 1
        if line_count > 1:
            line = np.zeros((line_count, 2))
            cursor = -total
            for i in range(line_count):
                line[i, 0] = cursor + r[i] + gap
                cursor += 2.0 * (r[i] + gap)
            promote(line)
        if best_count >= upper:
            return answer

        class SliceEnd(Exception):
            pass

        def intersections(a, ar, b, br):
            """Compute in batch the intersection points of both kinds of contact circles."""
            delta = b - a
            d2 = np.sum(delta * delta, axis=1)
            d = np.sqrt(d2)
            valid = ((d > 1.0e-14) & (d <= ar + br)
                     & (d >= np.abs(ar - br)))
            if not np.any(valid):
                return np.empty((0, 2))
            a, delta, d = a[valid], delta[valid], d[valid]
            ar, br = ar[valid], br[valid]
            along = (ar * ar - br * br + d * d) / (2.0 * d)
            height = np.sqrt(np.maximum(0.0, ar * ar - along * along))
            unit = delta / d[:, None]
            base = a + along[:, None] * unit
            side = height[:, None] * np.column_stack((-unit[:, 1], unit[:, 0]))
            return np.vstack((base + side, base - side))

        def front(idx, ids, xy, mode, stop, allow_overlap=False):
            """The same front provides both feasible insertion points and low-conflict seeds for global repair."""
            if time.perf_counter() >= stop:
                raise SliceEnd
            m = len(ids)
            room = max(0.0, 1.0 - r[idx] - gap)
            samples = max(12, int((4.0 + effort) * math.sqrt(m + 1)))
            phase = rng.random() * tau
            angle = phase + np.arange(samples) * (tau / samples)
            ring = room * np.column_stack((np.cos(angle), np.sin(angle)))
            parts = [ring, np.zeros((1, 2))]
            if allow_overlap:
                parts.append(ring * math.sqrt(rng.random()))
            if m:
                rr = r[ids] + r[idx] + gap
                parts.append(intersections(np.zeros_like(xy),
                                           np.full(m, room), xy, rr))
                # Single-circle contacts fill out the front early in construction and after local resets.
                directions = angle[:min(samples, max(6, int(3.0 + effort)))]
                directions = directions + np.arange(len(directions)) * 2.399963229728653
                unit = np.column_stack((np.cos(directions), np.sin(directions)))
                parts.append((xy[:, None, :] + rr[:, None, None] * unit).reshape(-1, 2))
                p, q = np.triu_indices(m, 1)
                chunk = max(64, int(8192 / max(1.0, math.sqrt(m))))
                for base in range(0, len(p), chunk):
                    if time.perf_counter() >= stop:
                        raise SliceEnd
                    pi, qi = p[base:base + chunk], q[base:base + chunk]
                    parts.append(intersections(xy[pi], rr[pi], xy[qi], rr[qi]))
            points = np.vstack(parts)
            norms = np.sqrt(np.sum(points * points, axis=1))
            good = norms <= room + 1.0e-13
            points, norms = points[good], norms[good]
            chosen = None
            chosen_score = float("inf")
            overlap_chosen = None
            overlap_score = float("inf")
            block = max(16, int(32768 / max(1, m)))
            anchor = np.array([-math.cos(phase), -math.sin(phase)])
            for base in range(0, len(points), block):
                if time.perf_counter() >= stop:
                    raise SliceEnd
                pts = points[base:base + block]
                ds = norms[base:base + block]
                # Small circles grow tightly along the boundary; larger circles also consider compacting inward.
                scores = (mode * (room - ds) + (1.0 - mode) * ds
                          + 0.035 * np.sqrt(np.sum((pts - anchor) ** 2, axis=1)))
                scores += np.asarray([rng.random() for _ in range(len(pts))]) * (0.001 + 0.0004 * effort)
                if m:
                    delta = pts[:, None, :] - xy[None, :, :]
                    d2 = np.sum(delta * delta, axis=2)
                    needs = r[ids] + r[idx] + gap * 0.65
                    feasible = np.all(d2 >= needs[None, :] ** 2, axis=1)
                else:
                    feasible = np.ones(len(pts), dtype=bool)
                if np.any(feasible):
                    value = np.where(feasible, scores, np.inf)
                    pos = int(np.argmin(value))
                    if value[pos] < chosen_score:
                        chosen_score = float(value[pos])
                        chosen = pts[pos].copy()
                if allow_overlap and chosen is None:
                    if m:
                        pen = np.maximum(0.0, needs[None, :] - np.sqrt(d2))
                        value = np.sum(pen * pen, axis=1) + 1.0e-7 * scores
                    else:
                        value = scores
                    pos = int(np.argmin(value))
                    if value[pos] < overlap_score:
                        overlap_score = float(value[pos])
                        overlap_chosen = pts[pos].copy()
            return chosen if chosen is not None else overlap_chosen

        # First pack the smallest circles consecutively, so a valid prefix can be saved at every step.
        quick_stop = min(deadline, start + budget * 0.16)
        forward = []
        try:
            for i in range(upper):
                xy = np.asarray(forward, dtype=float).reshape(-1, 2)
                point = front(i, list(range(i)), xy, 0.93, quick_stop)
                if point is None:
                    break
                forward.append(point)
                if i + 1 > best_count:
                    promote(np.asarray(forward))
        except SliceEnd:
            pass

        def repair(seed, k, stop):
            """After continuous relaxation, distribute displacements by collision mass and return the lowest-conflict layout of this period."""
            rr = r[:k]
            p, q = np.triu_indices(k, 1)
            need = rr[p] + rr[q] + gap
            room = np.maximum(0.0, 1.0 - rr - gap)
            best_local = np.array(seed, copy=True)
            best_energy = float("inf")
            best_worst = float("inf")
            finished = False
            local_stop = time.perf_counter() + max(0.0, stop - time.perf_counter()) * 0.88

            def evaluate(flat):
                nonlocal best_local, best_energy, best_worst, finished
                if time.perf_counter() >= local_stop:
                    raise SliceEnd
                z = np.asarray(flat).reshape(k, 2)
                delta = z[p] - z[q]
                dist = np.sqrt(np.sum(delta * delta, axis=1) + 1.0e-30)
                pen = np.maximum(0.0, need - dist)
                norms = np.sqrt(np.sum(z * z, axis=1) + 1.0e-30)
                wall = np.maximum(0.0, norms - room)
                energy = float(np.dot(pen, pen) + np.dot(wall, wall))
                worst = max(float(np.max(pen, initial=0.0)), float(np.max(wall, initial=0.0)))
                if energy < best_energy:
                    best_energy, best_worst = energy, worst
                    best_local = z.copy()
                if worst < gap * 0.25 and promote(z):
                    finished = True
                    raise SliceEnd
                force = (-2.0 * pen / dist)[:, None] * delta
                grad = (2.0 * wall / norms)[:, None] * z
                grad[:, 0] += np.bincount(p, weights=force[:, 0], minlength=k) - np.bincount(q, weights=force[:, 0], minlength=k)
                grad[:, 1] += np.bincount(p, weights=force[:, 1], minlength=k) - np.bincount(q, weights=force[:, 1], minlength=k)
                return energy, grad.ravel()

            try:
                # No cap on iterations or function evaluations; the wall clock in the callback ends this round.
                minimize(evaluate, np.asarray(seed).ravel(), method="L-BFGS-B", jac=True,
                         options={"ftol": 0.0, "gtol": 1.0e-12,
                                  "maxiter": float("inf"), "maxfun": float("inf"),
                                  "maxcor": max(5, int(3.0 * effort + math.sqrt(k)))})
            except SliceEnd:
                pass
            if finished:
                return best_local, best_energy, True

            # Projection polishing only near the feasible region; keep the large-circle skeleton so small circles move more easily.
            z = best_local.copy()
            if best_worst < max(gap * 8.0, 0.03 / math.sqrt(k)):
                while time.perf_counter() < stop:
                    d = z[p] - z[q]
                    dist = np.sqrt(np.sum(d * d, axis=1) + 1.0e-30)
                    active = np.flatnonzero(dist < need)
                    rng.shuffle(active)
                    for h in active:
                        if time.perf_counter() >= stop:
                            break
                        i, j = p[h], q[h]
                        dx, dy = z[i] - z[j]
                        length = math.hypot(dx, dy)
                        overlap = need[h] - length
                        if overlap <= 0.0:
                            continue
                        if length < 1.0e-15:
                            angle = rng.random() * tau
                            ux, uy = math.cos(angle), math.sin(angle)
                        else:
                            ux, uy = dx / length, dy / length
                        wi = (rr[j] ** 2 + gap) / (rr[i] ** 2 + rr[j] ** 2 + 2.0 * gap)
                        shift = np.array([ux, uy]) * (overlap * 1.02)
                        z[i] += wi * shift
                        z[j] -= (1.0 - wi) * shift
                    norms = np.sqrt(np.sum(z * z, axis=1))
                    z *= np.minimum(1.0, room / np.maximum(norms, 1.0e-30))[:, None]
                    if promote(z):
                        return z, 0.0, True
                    if not len(active):
                        break
            return best_local, best_energy, False

        def reset_patch(seed, k, depth, stop):
            """Remove several circles from a high-conflict neighbourhood, then reinsert them through the same contact front."""
            z = np.array(seed, copy=True)
            d = z[:, None, :] - z[None, :, :]
            distances = np.sqrt(np.sum(d * d, axis=2))
            overlap = np.maximum(0.0, r[:k, None] + r[None, :k] + gap - distances)
            np.fill_diagonal(overlap, 0.0)
            conflict = np.sum(overlap * overlap, axis=1)
            focus = int(np.argmax(conflict)) if rng.random() < 0.7 else rng.randrange(k)
            proximity = distances[focus] / np.maximum(r[:k] + r[focus], gap)
            removed = list(np.argsort(proximity)[:depth])
            rng.shuffle(removed)
            removed_set = set(removed)
            kept = [i for i in range(k) if i not in removed_set]
            for i in removed:
                pos = front(i, kept, z[kept], 0.2 + rng.random() * 0.75,
                            stop, allow_overlap=True)
                if pos is not None:
                    z[i] = pos
                kept.append(i)
            return z

        target = -1
        elite = None
        elite_energy = float("inf")
        restart = 0
        while best_count < upper and time.perf_counter() < deadline:
            k = best_count + 1
            if target != k:
                target, elite, elite_energy, restart = k, None, float("inf"), 0
            now = time.perf_counter()
            remaining = deadline - now
            # As the budget grows, the local solve period and the reset neighbourhood grow together.
            slice_length = budget * (0.014 + 0.004 * effort) * (1.0 + math.sqrt(k) / 12.0)
            stop = min(deadline, now + min(remaining, slice_length))
            seed_stop = now + (stop - now) * 0.30
            seed = None
            try:
                if elite is None or restart % 4 == 0:
                    point = front(k - 1, list(range(best_count)), best_xy,
                                  rng.random(), seed_stop, allow_overlap=True)
                    if point is not None:
                        seed = np.vstack((best_xy, point))
                elif restart % 4 == 1:
                    depth = min(k, max(2, int(math.sqrt(k) *
                                (0.35 + 0.10 * effort + 0.04 * math.log1p(restart)))))
                    seed = reset_patch(elite, k, depth, seed_stop)
                elif restart % 4 == 2:
                    # The reversed skeleton still uses the same front; when blocked, keep low-conflict points for global repair.
                    seed = np.zeros((k, 2))
                    ids = []
                    for i in range(k - 1, -1, -1):
                        mix = 0.15 + 0.70 * (k - 1 - i) / max(1, k - 1)
                        point = front(i, ids, seed[ids], mix, seed_stop, True)
                        if point is None:
                            raise SliceEnd
                        seed[i] = point
                        ids.append(i)
                else:
                    seed = elite.copy()
                    amplitude = min(0.25, (0.008 + 0.005 * effort) * math.sqrt(1.0 + restart))
                    for i in range(k):
                        a = rng.random() * tau
                        seed[i] += amplitude * np.array([math.cos(a), math.sin(a)])
                    # Swap positions rather than radii, so all decisions still correspond to the original circle indices.
                    i, j = rng.randrange(k), rng.randrange(k)
                    seed[[i, j]] = seed[[j, i]]
            except SliceEnd:
                seed = None
            if time.perf_counter() >= deadline:
                break
            if seed is None:
                if elite is not None:
                    seed = elite.copy()
                    amplitude = (0.015 + effort * 0.007) / math.sqrt(max(1.0, k / 16.0))
                    for i in range(k):
                        a = rng.random() * tau
                        seed[i] += amplitude * np.array([math.cos(a), math.sin(a)])
                else:
                    seed = np.zeros((k, 2))
                    phase = rng.random() * tau
                    for rank, i in enumerate(range(k - 1, -1, -1)):
                        length = max(0.0, 1.0 - r[i] - gap) * math.sqrt((rank + 0.5) / k)
                        a = phase + rank * 2.399963229728653
                        seed[i] = [length * math.cos(a), length * math.sin(a)]
            if time.perf_counter() < stop:
                layout, energy, success = repair(seed, k, stop)
                if not success and energy < elite_energy:
                    elite, elite_energy = layout.copy(), energy
            restart += 1
        return answer
    except Exception:
        # Always keep and return the most recent complete solution that passed geometric verification.
        return answer
