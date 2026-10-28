"""Whole-prefix joint optimisation with layout rearrangement
Maintains a feasible prefix within the wall-clock budget through geometric point-set seeding, vectorised quasi-Newton overlap elimination and local position rearrangement.
"""
import math
import random
import time


class _Stop(Exception):
    pass


def solve(instance: dict, time_limit_s: float) -> dict:
    start = time.perf_counter()
    try:
        n = max(0, int(instance.get("n", 0)))
    except Exception:
        n = 0
    answer = {"packing_status": [0] * n,
              "centers": [[-1.0, -1.0] for _ in range(n)]}
    try:
        _search(instance, float(time_limit_s), start, answer)
    except Exception:
        # The saved solution does not take part in tentative updates, so exceptions cannot corrupt it.
        pass
    return answer


def _search(instance, budget, start, answer):
    n = len(answer["packing_status"])
    if n == 0 or not math.isfinite(budget) or budget <= 0:
        return
    deadline = start + budget * 0.96
    cx, cy, radius = float(instance["cx"]), float(instance["cy"]), float(instance["R"])
    raw = [float(instance["radii"][i]) for i in range(n)]
    if not all(math.isfinite(v) for v in [cx, cy, radius] + raw):
        return
    if radius < 0 or any(v < 0 for v in raw):
        return
    if radius == 0:
        k = 0
        while k < n and raw[k] == 0:
            k += 1
        answer["packing_status"][:k] = [1] * k
        answer["centers"][:k] = [[cx, cy] for _ in range(k)]
        return

    rlist = [v / radius for v in raw]
    best_k = 0
    best_pos = []
    # Dependency-free, linear-time initial feasible prefix: arrange along the diameter.
    total = 0.0
    for i, r in enumerate(rlist):
        if time.perf_counter() >= deadline or total + r > 1.0:
            break
        total += r
        best_k = i + 1
    cursor = -total
    positions = []
    for r in rlist[:best_k]:
        positions.append((cursor + r, 0.0))
        cursor += 2.0 * r
    if best_k:
        # The error of the construction below is only floating-point rounding; save after leaving a margin.
        if total > 0 and total < 1:
            factor = 1.0 + min(1e-8, (1.0 - total) / (2.0 * total))
            positions = [(x * factor, y) for x, y in positions]
        answer["packing_status"][:best_k] = [1] * best_k
        answer["centers"][:best_k] = [[cx + radius * x, cy + radius * y]
                                           for x, y in positions]
        best_pos = positions
    if best_k == n or time.perf_counter() >= deadline:
        return

    import numpy as np
    from scipy.spatial import cKDTree

    if time.perf_counter() >= deadline:
        return
    r = np.asarray(rlist, dtype=float)
    area = np.cumsum(r * r)
    upper = int(np.searchsorted(area, 1.0 + 2e-14, side="right"))
    upper = min(upper, int(np.searchsorted(r, 1.0 + 2e-14, side="right")), n)
    # Any two circles require the sum of their radii to be at most the container radius.
    if upper >= 2:
        bad = np.flatnonzero(r[1:upper] + r[:upper - 1] > 1.0 + 2e-14)
        if len(bad):
            upper = int(bad[0]) + 1
    if best_k >= upper:
        return

    rng = np.random.default_rng(n * 1009 + 73)
    prng = random.Random(n * 9176 + 31)
    safety = 3e-9
    strength = math.log1p(budget)
    memory = max(4, min(24, int(4 + 3 * strength)))
    pool_size = max(2, int(2 + math.sqrt(strength)))
    dense_limit = max(64, int(160 * math.sqrt(1.0 + strength)))

    def check_time(stop=deadline):
        if time.perf_counter() >= stop:
            raise _Stop()

    def pair_indices(p, rr):
        k = len(rr)
        if k <= dense_limit:
            return np.triu_indices(k, 1)
        pairs = cKDTree(p).query_pairs(2.0 * float(np.max(rr)) + 2 * safety,
                                      output_type="ndarray")
        return pairs[:, 0], pairs[:, 1]

    def publish(p):
        nonlocal best_k, best_pos
        k = len(p)
        if k <= best_k or not np.isfinite(p).all():
            return False
        check_time()
        rr = r[:k]
        if np.any(np.hypot(p[:, 0], p[:, 1]) + rr > 1.0 + 2e-14):
            return False
        ii, jj = pair_indices(p, rr)
        diff = p[ii] - p[jj]
        if np.any(np.hypot(diff[:, 0], diff[:, 1]) < rr[ii] + rr[jj] - 2e-14):
            return False
        world = p * radius + np.array([cx, cy])
        if not np.isfinite(world).all():
            return False
        # Recheck at the scale of the returned coordinates, preventing translation from losing significant digits.
        tolerance = 8.0 * math.ulp(max(radius, abs(cx), abs(cy), 1e-300))
        center_dist = np.hypot(world[:, 0] - cx, world[:, 1] - cy)
        if np.any(center_dist + np.asarray(raw[:k]) > radius + tolerance):
            return False
        diff = world[ii] - world[jj]
        if np.any(np.hypot(diff[:, 0], diff[:, 1]) + tolerance <
                  (rr[ii] + rr[jj]) * radius):
            return False
        best_pos = p.copy()
        best_k = k
        answer["packing_status"] = [1] * k + [0] * (n - k)
        answer["centers"] = world.tolist() + [[-1.0, -1.0] for _ in range(n - k)]
        return True

    def lattice_sites(k, offset):
        extent = int(math.ceil(math.sqrt(k))) + 2
        rows = np.arange(-extent, extent + 1, dtype=float)
        xx, yy = np.meshgrid(rows, rows)
        q = np.column_stack(((xx + yy * 0.5 + offset[0]).ravel(),
                             (yy * (math.sqrt(3.0) / 2.0) + offset[1]).ravel()))
        ds = np.einsum("ij,ij->i", q, q)
        selected = np.argpartition(ds, k - 1)[:k]
        selected = selected[np.argsort(ds[selected])]
        return q[selected], np.sqrt(ds[selected])

    def lattice_bound(q, distances):
        lo, hi = best_k, upper
        while lo < hi:
            check_time()
            k = (lo + hi + 1) // 2
            spacing = (r[k - 1] + r[k - 2] + safety) if k > 1 else 0.0
            if np.all(distances[:k] * spacing + r[:k][::-1] <= 1.0 - safety):
                lo = k
            else:
                hi = k - 1
        if lo > best_k:
            spacing = (r[lo - 1] + r[lo - 2] + safety) if lo > 1 else 0.0
            publish(q[:lo][::-1].copy() * spacing)

    def seed(k, q, mode):
        rr = r[:k]
        if mode == 0:
            p = q[:k][::-1].copy()
            norms = np.hypot(p[:, 0], p[:, 1])
            spacing = min(float(rr[-1] + (rr[-2] if k > 1 else 0)),
                          float(np.min((1.0 - rr - safety) / np.maximum(norms, 1e-15))))
            return p * max(0.0, spacing)
        angle = np.arange(k) * (math.pi * (3.0 - math.sqrt(5.0)))
        angle += rng.uniform(0, 2 * math.pi)
        norms = np.sqrt((np.arange(k)[::-1] + 0.5) / k) * (1 - rr)
        if mode == 2:
            angle = rng.uniform(0, 2 * math.pi, k)
            norms = np.sqrt(rng.random(k)) * (1 - rr)
        return np.column_stack((norms * np.cos(angle), norms * np.sin(angle)))

    class Energy:
        def __init__(self, k, stop):
            self.k, self.rr, self.stop = k, r[:k], stop
            self.fixed = np.triu_indices(k, 1) if k <= dense_limit else None
            self.floor = max(1e-5, float(np.mean(self.rr)) * 0.35)
            self.wall_w = 2.0 / np.maximum(self.rr, self.floor) ** 2
            self.best_value, self.best = float("inf"), None

        def evaluate(self, flat):
            check_time(self.stop)
            p = flat.reshape(self.k, 2)
            rr = self.rr
            ii, jj = self.fixed if self.fixed is not None else pair_indices(p, rr)
            delta = p[ii] - p[jj]
            d = np.hypot(delta[:, 0], delta[:, 1])
            need = rr[ii] + rr[jj]
            gaps = np.maximum(need + safety * (need > 0) - d, 0.0)
            wall_d = np.hypot(p[:, 0], p[:, 1])
            wall_gap = np.maximum(wall_d + rr + safety - 1.0, 0.0)
            weights = 1.0 / np.maximum(need, self.floor) ** 2
            value = float(np.dot(gaps * weights, gaps) +
                          np.dot(wall_gap * self.wall_w, wall_gap))
            grad = (2 * self.wall_w * wall_gap / np.maximum(wall_d, 1e-30))[:, None] * p
            active = gaps > 0
            a, b = ii[active], jj[active]
            dist, delta = d[active], delta[active].copy()
            coincident = dist < 1e-14
            if np.any(coincident):
                # Choose a deterministic separation direction at coincident circle centres.
                angles = (a[coincident] * 1.7 + b[coincident] * 2.3)
                delta[coincident] = np.column_stack((np.cos(angles), np.sin(angles))) * 1e-14
                dist = np.maximum(dist, 1e-14)
            force = (-2.0 * weights[active] * gaps[active] /
                     np.maximum(dist, 1e-30))[:, None] * delta
            grad[:, 0] += np.bincount(a, force[:, 0], minlength=self.k)
            grad[:, 0] -= np.bincount(b, force[:, 0], minlength=self.k)
            grad[:, 1] += np.bincount(a, force[:, 1], minlength=self.k)
            grad[:, 1] -= np.bincount(b, force[:, 1], minlength=self.k)
            worst = max(float(np.max(wall_gap, initial=0)),
                        float(np.max(gaps, initial=0)))
            if value < self.best_value:
                self.best_value, self.best = value, p.copy()
            return value, grad.ravel(), worst

    def local_search(initial, stop):
        energy = Energy(len(initial), stop)
        x = initial.ravel().copy()
        history = []
        try:
            value, gradient, worst = energy.evaluate(x)
            while time.perf_counter() < stop:
                if worst < safety * 0.20:
                    return x.reshape(-1, 2), value, True
                # Limited-memory quasi-Newton; iterations and backtracking are both controlled by the wall clock and numeric convergence.
                direction = gradient.copy()
                coefficients = []
                for s, y, inv in reversed(history):
                    alpha = inv * float(np.dot(s, direction))
                    coefficients.append(alpha)
                    direction -= alpha * y
                if history:
                    s, y, inv = history[-1]
                    direction *= float(np.dot(s, y)) / max(float(np.dot(y, y)), 1e-30)
                else:
                    direction *= max(1e-8, float(np.mean(energy.rr)) ** 2) * 0.1
                for (s, y, inv), alpha in zip(history, reversed(coefficients)):
                    direction += s * (alpha - inv * float(np.dot(y, direction)))
                direction = -direction
                slope = float(np.dot(gradient, direction))
                if slope >= 0 or not math.isfinite(slope):
                    history.clear()
                    direction = -gradient
                    slope = -float(np.dot(gradient, gradient))
                if slope >= -1e-28:
                    break
                step = min(1.0, 0.25 / max(float(np.max(np.abs(direction))), 1e-30))
                accepted = False
                while time.perf_counter() < stop and step > 1e-14:
                    nx = x + step * direction
                    nv, ng, nw = energy.evaluate(nx)
                    if nv <= value + 1e-4 * step * slope:
                        accepted = True
                        break
                    step *= 0.5
                if not accepted:
                    break
                s, y = nx - x, ng - gradient
                sy = float(np.dot(s, y))
                if sy > 1e-12 * math.sqrt(float(np.dot(s, s)) * float(np.dot(y, y))):
                    history.append((s, y, 1.0 / sy))
                    if len(history) > memory:
                        history.pop(0)
                if abs(value - nv) <= 1e-15 * max(1e-5, value) and nw >= worst:
                    break
                x, value, gradient, worst = nx, nv, ng, nw
        except _Stop:
            pass
        if energy.best is None:
            return initial.copy(), float("inf"), False
        return energy.best, energy.best_value, False

    def perturb(p, k, progress):
        p = p.copy()
        rr = r[:k]
        # Change several positions at once, rather than freezing the old prefix and inserting a new circle.
        width = min(k, max(2, int(math.sqrt(k) * (0.35 + strength / 5))))
        mode = prng.randrange(3)
        if mode == 0:
            indices = rng.choice(k, size=width, replace=False)
            shuffled = rng.permutation(indices)
            p[indices] = p[shuffled]
            p += rng.normal(0, 1, p.shape) * (float(np.mean(rr)) * 0.03)
        elif mode == 1:
            anchor = p[prng.randrange(k)]
            dist = np.sum((p - anchor) ** 2, axis=1)
            indices = np.argpartition(dist, width - 1)[:width]
            center = np.mean(p[indices], axis=0)
            angle = rng.uniform(-math.pi, math.pi)
            rotation = np.array([[math.cos(angle), -math.sin(angle)],
                                 [math.sin(angle), math.cos(angle)]])
            p[indices] = (p[indices] - center) @ rotation + center
        else:
            amplitude = float(np.mean(rr)) * (0.10 + 0.40 * progress)
            amplitude *= 1.0 + strength / (2.0 + math.log1p(k))
            p += rng.normal(0, amplitude, p.shape)
            indices = rng.choice(k, size=max(1, width // 3), replace=False)
            angle = rng.uniform(0, 2 * math.pi, len(indices))
            norms = np.sqrt(rng.random(len(indices))) * (1 - rr[indices])
            p[indices] = np.column_stack((norms * np.cos(angle), norms * np.sin(angle)))
        norm = np.hypot(p[:, 0], p[:, 1])
        p *= np.minimum(1.0, np.maximum(0.0, 1.0 - rr - safety) /
                        np.maximum(norm, 1e-30))[:, None]
        return p

    q, distances = lattice_sites(upper, (0.0, 0.0))
    lattice_bound(q, distances)
    if best_k >= upper:
        return
    hint = min(upper, int(np.searchsorted(area, 0.80, side="right")))
    target = max(best_k + 1, hint)
    pool = []
    attempt = 0
    # Unknown feasibility is not used as an upper bound: a failed target can still be re-explored in later restarts.
    while time.perf_counter() < deadline and best_k < upper:
        try:
            if target <= best_k:
                target, pool = best_k + 1, []
            mode = attempt % 4
            if mode == 0:
                offset = (rng.uniform(-0.5, 0.5), rng.uniform(-math.sqrt(3) / 4, math.sqrt(3) / 4))
                q, distances = lattice_sites(upper, offset)
                lattice_bound(q, distances)
                if best_k >= upper:
                    break
                if target <= best_k:
                    target, pool = best_k + 1, []
                initial = seed(target, q, 0)
            elif pool:
                rank = min(len(pool) - 1, int((prng.random() ** 2) * len(pool)))
                progress = min(1.0, (time.perf_counter() - start) / budget)
                initial = perturb(pool[rank][1], target, progress)
            elif best_k and target - best_k <= max(1, int(math.sqrt(target))):
                initial = seed(target, q, 2)
                initial[:best_k] = np.asarray(best_pos)
                initial = perturb(initial, target, 0.25)
            else:
                initial = seed(target, q, 1 + (attempt % 2))

            now = time.perf_counter()
            slice_length = budget / (4.0 + math.sqrt(target) + strength)
            stop = min(deadline, now + slice_length)
            trial, value, solved = local_search(initial, stop)
            if time.perf_counter() >= deadline:
                break
            if (solved or value < 1e-12) and publish(trial):
                target = min(upper, max(best_k + 1, hint))
                pool = []
                attempt += 1
                continue
            if math.isfinite(value):
                pool.append((value, trial.copy()))
                pool.sort(key=lambda entry: entry[0])
                del pool[pool_size:]
            if target > best_k + 1:
                target = best_k + max(1, (target - best_k) // 2)
                pool = []
            attempt += 1
        except _Stop:
            break
