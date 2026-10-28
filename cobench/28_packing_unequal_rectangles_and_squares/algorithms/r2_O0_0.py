"""Circle-contact construction with adaptive repacking search
Combines contact-point insertion in both orientations, local removal and repacking, and continuous coordinate repair, keeping the feasible layout that packs the most items within the wall-clock limit.
"""

import math
import random
import time


class _SearchStop(Exception):
    pass


class _Packing:
    def __init__(self, instance, budget, started, answer):
        self.answer = answer
        self.n = len(answer['alpha'])
        self.radius = float(instance['container_radius'])
        self.widths = instance['widths']
        self.heights = instance['heights']
        self.budget = budget
        self.clock = time.perf_counter
        self.deadline = started + budget - min(budget * 0.04, 0.015 + budget * 0.02)
        self.rng = random.Random(71983)
        self.items = {}
        self.best = []
        self.pool = []
        self.pool_size = max(2, int(2 + 2 * math.log1p(budget)))
        self.last_gain = started
        self.np = None
        self.minimize = None
        self.import_tried = False
        for i in range(self.n):
            a = (float(self.widths[i]) / self.radius) * 0.5
            b = (float(self.heights[i]) / self.radius) * 0.5
            if math.hypot(a, b) <= 1.0:
                self.items[i] = (a, b, 4.0 * a * b)
        self.ids = sorted(self.items, key=lambda i: self.items[i][2])
        self.bound = len(self.ids)
        total = 0.0
        for k, i in enumerate(self.ids):
            total += self.items[i][2]
            if total > math.pi + 1e-12:
                self.bound = k
                break
        if self.ids:
            i = self.ids[0]
            a, b, _ = self.items[i]
            self.best = [(i, 0.0, 0.0, a, b, 0)]
            self.pool = [self.best[:]]
        self.scale = (sorted(max(v[0], v[1]) for v in self.items.values())
                      [len(self.items) // 2] if self.items else 1.0)
        self.gap = 3e-10 * min(1.0, max(self.scale, 1e-100))

    def feasible(self, layout):
        physical = []
        for j, p in enumerate(layout):
            _, x, y, a, b, _ = p
            if not (math.isfinite(x) and math.isfinite(y)):
                return False
            if math.hypot(abs(x) + a, abs(y) + b) > 1.0:
                return False
            for q in layout[:j]:
                if abs(x - q[1]) < a + q[3] and abs(y - q[2]) < b + q[4]:
                    return False
            i, _, _, _, _, rot = p
            w, h = float(self.widths[i]), float(self.heights[i])
            if rot:
                w, h = h, w
            xx, yy = x * self.radius, y * self.radius
            if math.hypot(abs(xx) + w * 0.5, abs(yy) + h * 0.5) > self.radius:
                return False
            if any(abs(xx - q[0]) < w * 0.5 + q[2] * 0.5 and
                   abs(yy - q[1]) < h * 0.5 + q[3] * 0.5 for q in physical):
                return False
            physical.append((xx, yy, w, h))
        return True

    def record(self, layout):
        count = len(layout)
        if count > len(self.best):
            if not self.feasible(layout):
                return
            self.best = layout[:]
            self.last_gain = self.clock()
            self.pool = [p for p in self.pool if len(p) >= count - 1]
        if count >= max(1, len(self.best) - 1):
            self.pool.append(layout[:])
            if len(self.pool) > self.pool_size:
                # The layout pool is only for search diversity and does not change the count objective.
                worst = min(range(len(self.pool)), key=lambda j: len(self.pool[j]))
                self.pool.pop(worst)

    def placement(self, i, layout, gravity, until, exact=False):
        """Scans the x-coordinates of contact events, then skips blocked intervals along the y-axis."""
        a0, b0, _ = self.items[i]
        swap, sign, slope, top_weight = gravity
        if swap:
            obstacles = [(p[2], sign * p[1], p[4], p[3]) for p in layout]
        else:
            obstacles = [(p[1], sign * p[2], p[3], p[4]) for p in layout]
        obstacles.sort(key=lambda p: p[1] - p[3])
        gap = 0.0 if exact else self.gap
        rad = 1.0 - gap
        rad2 = rad * rad
        winner = None
        best_score = float('inf')
        for rot in (0, 1):
            if rot and a0 == b0:
                continue
            aa, bb = (b0, a0) if rot else (a0, b0)
            a, b = (bb, aa) if swap else (aa, bb)
            if math.hypot(a, b) > rad:
                continue
            xmax = math.sqrt(max(0.0, rad2 - b * b)) - a
            if xmax < 0.0:
                continue
            xs = {0.0, -xmax, xmax}
            norm = math.hypot(1.0, slope)
            tangent = max(0.0, rad * abs(slope) / norm - a)
            xs.add(-math.copysign(tangent, slope))
            for ox, oy, oa, ob in obstacles:
                d = a + oa + gap
                xs.add(ox - d)
                xs.add(ox + d)
                for yy in (oy - ob - b - gap, oy + ob + b + gap):
                    v = abs(yy) + b
                    if v <= rad:
                        xx = math.sqrt(max(0.0, rad2 - v * v)) - a
                        if xx >= 0.0:
                            xs.add(-xx)
                            xs.add(xx)
            for x in xs:
                if self.clock() >= until:
                    return winner
                if abs(x) > xmax:
                    continue
                height = math.sqrt(max(0.0, rad2 - (abs(x) + a) ** 2)) - b
                if height < 0.0:
                    continue
                y = -height
                for ox, oy, oa, ob in obstacles:
                    if abs(x - ox) >= a + oa:
                        continue
                    lower = oy - ob - b
                    upper = oy + ob + b
                    if y <= lower:
                        break
                    if y < upper:
                        y = upper + gap
                        if y > height:
                            break
                if y > height:
                    continue
                score = y + top_weight * b + slope * x
                if score >= best_score:
                    continue
                px, py = (sign * y, x) if swap else (x, sign * y)
                p = (i, px, py, aa, bb, rot)
                # Zero-gap attempts must also pass the actual geometric check.
                if math.hypot(abs(px) + aa, abs(py) + bb) > 1.0:
                    continue
                if any(abs(px - q[1]) < aa + q[3] and
                       abs(py - q[2]) < bb + q[4] for q in layout):
                    continue
                winner, best_score = p, score
        return winner

    def gravity(self, initial=False):
        if initial:
            return False, 1, 0.03, 1.0
        r = self.rng
        return (r.random() < 0.5, -1 if r.random() < 0.5 else 1,
                r.uniform(-0.8, 0.8) if r.random() < 0.45 else r.choice((-0.02, 0.02)),
                r.choice((0.0, 0.5, 1.0)))

    def fill(self, order, layout, gravity, until, exact=False):
        layout = layout[:]
        used = {p[0] for p in layout}
        for i in order:
            if self.clock() >= until or len(layout) >= self.bound:
                break
            if i in used:
                continue
            p = self.placement(i, layout, gravity, until, exact)
            if p is not None:
                layout.append(p)
                used.add(i)
        return layout

    def order(self, selected_count, rebuilding=False):
        r = self.rng
        kind = r.randrange(6)
        noise = r.uniform(0.02, 0.22) * math.log1p(self.budget)
        scores = {}
        shape = r.uniform(-0.18, 0.65) if kind == 1 else 0.0
        for i in self.ids:
            a, b, area = self.items[i]
            if kind == 2:
                value = math.log(max(a + b, 1e-300))
            else:
                value = math.log(max(area, 1e-300))
            value += shape * math.log(max(a, b) / max(min(a, b), 1e-300))
            scores[i] = value + noise * r.gauss(0.0, 1.0)
        order = sorted(self.ids, key=scores.__getitem__)
        if kind >= 3:
            reach = max(1, int(math.sqrt(max(1, selected_count)) * math.log1p(self.budget)))
            target = min(self.bound, selected_count + 1 + r.randrange(reach))
            chosen, rest = order[:target], order[target:]
            if kind == 3:
                chosen.sort(key=lambda i: self.items[i][2], reverse=True)
            elif kind == 4:
                chosen.sort(key=lambda i: max(self.items[i][:2]), reverse=True)
            else:
                r.shuffle(chosen)
            order = chosen + rest
        elif rebuilding and r.random() < 0.3:
            front = min(len(order), selected_count + 1)
            order[:front] = sorted(order[:front], key=lambda i: self.items[i][2], reverse=True)
        return order

    def compact(self, layout, gravity, until, exact):
        layout = layout[:]
        swap, sign, _, _ = gravity
        order = [p[0] for p in sorted(layout,
                 key=lambda p: sign * (p[1] if swap else p[2]))]
        for i in order:
            if self.clock() >= until:
                break
            old = next(p for p in layout if p[0] == i)
            others = [p for p in layout if p[0] != i]
            new = self.placement(i, others, gravity, until, exact)
            layout = others + [new if new is not None else old]
        return layout

    def rebuild(self, until):
        r = self.rng
        source = self.best if r.random() < 0.6 else r.choice(self.pool)
        source = source[:]
        size = len(source)
        if not size:
            return []
        stale = (self.clock() - self.last_gain) / max(self.budget, 1e-12)
        amplitude = min(0.8, 0.12 + 0.10 * math.log1p(self.budget) + stale)
        remove = max(1, min(size, int(1 + size * r.uniform(0.08, amplitude))))
        mode = r.randrange(4)
        if mode == 0:
            center = r.choice(source)
            source.sort(key=lambda p: (p[1] - center[1]) ** 2 + (p[2] - center[2]) ** 2)
        elif mode == 1:
            source.sort(key=lambda p: p[3] * p[4] * r.uniform(0.7, 1.3), reverse=True)
        elif mode == 2:
            angle = r.random() * math.tau
            gx, gy = math.cos(angle), math.sin(angle)
            source.sort(key=lambda p: gx * p[1] + gy * p[2])
        else:
            r.shuffle(source)
        survivor = source[remove:]
        g = self.gravity()
        exact = r.random() < 0.10
        if survivor and r.random() < 0.45:
            now = self.clock()
            survivor = self.compact(survivor, g, min(until, now + (until - now) * 0.27), exact)
        order = self.order(len(self.best), True)
        if r.random() < 0.25:
            removed = {p[0] for p in source[:remove]}
            order.sort(key=lambda i: i in removed)
        return self.fill(order, survivor, g, until, exact)

    def load_optimizer(self):
        if self.import_tried:
            return self.minimize is not None
        self.import_tried = True
        try:
            import numpy as np
            from scipy.optimize import minimize
            self.np = np
            self.minimize = minimize
        except Exception:
            return False
        return True

    def repair(self, until):
        """Tries to pack one more; intermediate points of the continuous search are never output as feasible solutions."""
        if len(self.best) >= self.bound or self.clock() >= until:
            return None
        k = len(self.best) + 1
        if k * k > max(1.0, until - self.clock()) * 220000:
            return None
        if not self.load_optimizer() or self.clock() >= until:
            return None
        np, r = self.np, self.rng
        source = self.best[:]
        used = {p[0] for p in source}
        unused = [i for i in self.ids if i not in used]
        if not unused:
            return None
        width = min(len(unused), max(1, int(math.sqrt(k) * math.log1p(self.budget))))
        chosen = unused[min(width - 1, int(r.expovariate(1.4)))]
        # Allow a smaller unselected rectangle to replace a larger rectangle in the current layout.
        if len(unused) > 1 and len(source) > 1 and r.random() < 0.24:
            victim = max(source, key=lambda p: p[3] * p[4] * r.uniform(0.6, 1.4))
            substitutes = [i for i in unused[:width] if i != chosen]
            if substitutes:
                replacement = r.choice(substitutes)
                a, b, _ = self.items[replacement]
                source.remove(victim)
                source.append((replacement, victim[1], victim[2], a, b, 0))
        a, b, _ = self.items[chosen]
        if sum(4 * p[3] * p[4] for p in source) + 4 * a * b > math.pi + 1e-12:
            return None
        ids = [p[0] for p in source] + [chosen]
        base_a = np.array([self.items[i][0] for i in ids], dtype=float)
        base_b = np.array([self.items[i][1] for i in ids], dtype=float)
        rotations = np.array([p[5] for p in source] + [0], dtype=int)
        positions = np.array([p[1] for p in source] + [0.0] +
                             [p[2] for p in source] + [0.0], dtype=float)
        ii, jj = np.triu_indices(k, 1)
        clear = 8e-8 * min(1.0, max(self.scale, 1e-100))
        best_start = None
        best_cost = float('inf')
        # Place the new rectangle at a low-conflict contact position, then let all centres move jointly.
        for rot in (0, 1):
            aa, bb = (b, a) if rot else (a, b)
            xlim = max(0.0, math.sqrt(max(0.0, 1.0 - bb * bb)) - aa)
            samples = [(0.0, 0.0)]
            for p in source:
                samples.extend(((p[1] - p[3] - aa, p[2]),
                                (p[1] + p[3] + aa, p[2]),
                                (p[1], p[2] - p[4] - bb),
                                (p[1], p[2] + p[4] + bb)))
            sample_count = max(1, int(math.sqrt(k) * (1.0 + math.log1p(self.budget))))
            for _ in range(sample_count):
                xx = r.uniform(-xlim, xlim)
                ylim = max(0.0, math.sqrt(max(0.0, 1.0 - (abs(xx) + aa) ** 2)) - bb)
                samples.append((xx, r.uniform(-ylim, ylim)))
            for xx, yy in samples:
                if self.clock() >= until:
                    return None
                xx = min(xlim, max(-xlim, xx))
                ylim = max(0.0, math.sqrt(max(0.0, 1.0 - (abs(xx) + aa) ** 2)) - bb)
                yy = min(ylim, max(-ylim, yy))
                cost = 0.0
                for p in source:
                    dx = aa + p[3] - abs(xx - p[1])
                    dy = bb + p[4] - abs(yy - p[2])
                    if dx > 0.0 and dy > 0.0:
                        cost += min(dx, dy) ** 2
                cost *= r.uniform(0.85, 1.15)
                if cost < best_cost:
                    best_cost = cost
                    best_start = xx, yy, rot
        if best_start is None:
            return None
        positions[k - 1], positions[2 * k - 1], rotations[k - 1] = best_start
        best_energy = float('inf')
        best_positions = positions.copy()
        best_rotations = rotations.copy()
        while self.clock() < until:
            ha = np.where(rotations == 0, base_a, base_b)
            hb = np.where(rotations == 0, base_b, base_a)
            pa, pb = ha + clear, hb + clear
            record = [float('inf'), positions.copy(), None]

            def energy(z):
                if self.clock() >= until:
                    raise _SearchStop()
                xx, yy = z[:k], z[k:]
                dx, dy = xx[ii] - xx[jj], yy[ii] - yy[jj]
                ox = pa[ii] + pa[jj] - np.abs(dx)
                oy = pb[ii] + pb[jj] - np.abs(dy)
                active = (ox > 0.0) & (oy > 0.0)
                vx = np.where(active & (ox <= oy), ox, 0.0)
                vy = np.where(active & (ox > oy), oy, 0.0)
                tx = -2.0 * vx * np.where(dx >= 0.0, 1.0, -1.0)
                ty = -2.0 * vy * np.where(dy >= 0.0, 1.0, -1.0)
                gx = np.bincount(ii, weights=tx, minlength=k) - np.bincount(jj, weights=tx, minlength=k)
                gy = np.bincount(ii, weights=ty, minlength=k) - np.bincount(jj, weights=ty, minlength=k)
                ax, ay = np.abs(xx) + pa, np.abs(yy) + pb
                radius = np.hypot(ax, ay)
                outside = np.maximum(0.0, radius - (1.0 - clear))
                factor = 6.0 * outside / np.maximum(radius, 1e-300)
                gx += factor * ax * np.where(xx >= 0.0, 1.0, -1.0)
                gy += factor * ay * np.where(yy >= 0.0, 1.0, -1.0)
                value = float(np.dot(vx, vx) + np.dot(vy, vy) + 3.0 * np.dot(outside, outside))
                if value < record[0]:
                    record[0], record[1] = value, z.copy()
                    if value <= 16.0 * clear * clear * k:
                        layout = [(ids[t], float(xx[t]), float(yy[t]), float(ha[t]),
                                   float(hb[t]), int(rotations[t])) for t in range(k)]
                        if self.feasible(layout):
                            record[2] = layout
                            raise _SearchStop()
                return value, np.concatenate((gx, gy))

            try:
                self.minimize(energy, positions, jac=True, method='L-BFGS-B',
                              options={'ftol': 1e-15, 'gtol': 1e-11,
                                       'maxiter': float('inf'), 'maxfun': float('inf'),
                                       'maxcor': max(3, min(2 * k, int(3 + 2 * math.sqrt(self.budget))))})
            except _SearchStop:
                pass
            except Exception:
                return None
            if record[2] is not None:
                return record[2]
            positions = record[1]
            if record[0] < best_energy:
                best_energy = record[0]
                best_positions, best_rotations = positions.copy(), rotations.copy()
            elif r.random() < 0.6:
                positions, rotations = best_positions.copy(), best_rotations.copy()
            if self.clock() >= until:
                break
            ha = np.where(rotations == 0, base_a, base_b)
            hb = np.where(rotations == 0, base_b, base_a)
            dx = ha[ii] + ha[jj] - np.abs(positions[ii] - positions[jj])
            dy = hb[ii] + hb[jj] - np.abs(positions[k + ii] - positions[k + jj])
            bad = np.maximum(0.0, np.minimum(dx, dy))
            stress = (np.bincount(ii, weights=bad, minlength=k) +
                      np.bincount(jj, weights=bad, minlength=k) +
                      np.maximum(0.0, np.hypot(np.abs(positions[:k]) + ha,
                                               np.abs(positions[k:]) + hb) - 1.0))
            order = sorted(range(k), key=lambda t: float(stress[t]) * r.uniform(0.6, 1.4), reverse=True)
            depth = max(1, min(k, int(1 + r.random() * math.log1p(self.budget) * math.sqrt(k) / 2)))
            for t in order[:depth]:
                if base_a[t] != base_b[t] and r.random() < 0.7:
                    rotations[t] = 1 - rotations[t]
                step = self.scale * r.uniform(0.04, 0.35)
                positions[t] += r.uniform(-step, step)
                positions[k + t] += r.uniform(-step, step)
        return None

    def run(self):
        if not self.ids or len(self.best) >= self.bound or self.clock() >= self.deadline:
            return
        now = self.clock()
        initial_end = min(self.deadline, now + max(self.budget * 0.13, (self.deadline - now) * 0.10))
        first = self.fill(self.ids, [], self.gravity(True), initial_end)
        self.record(first)
        while self.clock() < self.deadline and len(self.best) < self.bound:
            now = self.clock()
            remaining = self.deadline - now
            # Each neighbourhood uses only part of the total budget; the main search has no restart or iteration cap.
            window = min(remaining, self.budget * (0.025 + 0.025 * math.log1p(self.budget)))
            until = now + window
            choice = self.rng.random()
            if choice < 0.26 and remaining > self.budget * 0.06:
                if self.import_tried or remaining > 0.3 + self.budget * 0.08:
                    repaired = self.repair(until)
                    if repaired is not None:
                        self.record(repaired)
                    continue
            if choice < 0.70 and self.pool:
                layout = self.rebuild(until)
            else:
                order = self.order(len(self.best))
                layout = self.fill(order, [], self.gravity(), until, self.rng.random() < 0.12)
            self.record(layout)

    def export(self):
        """Re-checks in the original length units; unselected rectangles always keep the dummy origin."""
        accepted = []
        for i, x, y, a, b, rot in self.best:
            xx, yy = float(x * self.radius), float(y * self.radius)
            if not (math.isfinite(xx) and math.isfinite(yy)):
                continue
            w, h = float(self.widths[i]), float(self.heights[i])
            if rot:
                w, h = h, w
            # Use hypot rather than squared length, avoiding overflow when squaring finite inputs.
            if math.hypot(abs(xx) + w * 0.5, abs(yy) + h * 0.5) > self.radius:
                continue
            if any(abs(xx - q[0]) < w * 0.5 + q[2] * 0.5 and
                   abs(yy - q[1]) < h * 0.5 + q[3] * 0.5 for q in accepted):
                continue
            self.answer['alpha'][i] = 1
            self.answer['x'][i] = xx
            self.answer['y'][i] = yy
            self.answer['r'][i] = int(rot)
            accepted.append((xx, yy, w, h))
        return self.answer


def solve(instance: dict, time_limit_s: float) -> dict:
    started = time.perf_counter()
    answer = {'alpha': [], 'x': [], 'y': [], 'r': []}
    engine = None
    try:
        n = max(0, int(instance['num_rectangles']))
        answer = {'alpha': [0] * n, 'x': [0.0] * n,
                  'y': [0.0] * n, 'r': [0] * n}
        budget = float(time_limit_s)
        if not math.isfinite(budget) or budget <= 0.0 or n == 0:
            return answer
        engine = _Packing(instance, budget, started, answer)
        engine.run()
    except Exception:
        # A failed search must not lose the feasible layout already saved.
        pass
    if engine is not None:
        try:
            return engine.export()
        except Exception:
            pass
    return answer
