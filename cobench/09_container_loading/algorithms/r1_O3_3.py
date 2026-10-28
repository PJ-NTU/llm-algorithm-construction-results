"""Contact-guided free-space beam search
Fuses multi-corner contact scoring, adaptive beam search, and spatial and directed destroy-repair, continuously keeping the feasible solution with the largest packed volume by wall clock.
"""

import bisect
import math
import random
import time


def _empty(n):
    return {
        "placement_status": [0] * n,
        "coordinates": [[0.0, 0.0, 0.0] for _ in range(n)],
        "orientations": [-1] * n,
    }


def solve(instance: dict, time_limit_s: float) -> dict:
    start = time.perf_counter()
    try:
        n = max(0, int(instance.get("total_box_instances", 0)))
    except Exception:
        n = 0
    result = _empty(n)
    engine = None
    try:
        budget = float(time_limit_s)
        if n == 0 or not math.isfinite(budget) or budget <= 0:
            return result
        deadline = start + budget * 0.95
        engine = _Search(instance, n, budget, deadline)
        engine.run()
    except Exception:
        # Even if a step is interrupted, keep the historical best solution already confirmed feasible.
        pass
    if engine is not None:
        for p in engine.best:
            j, x, y, z, dx, dy, dz, k = p
            result["placement_status"][j] = 1
            result["coordinates"][j] = [x, y, z]
            result["orientations"][j] = k
    return result


class _Search:
    def __init__(self, instance, n, budget, deadline):
        self.best = ()
        self.best_value = 0.0
        self.deadline = deadline
        self.budget = budget
        self.n = n
        self.dims = tuple(float(instance[k]) for k in
                          ("container_length", "container_width", "container_height"))
        if not all(math.isfinite(d) and d > 0 for d in self.dims):
            raise ValueError("container dimensions must be finite positive numbers")
        self.cv = math.prod(self.dims)
        if not math.isfinite(self.cv) or self.cv <= 0:
            raise ValueError("invalid container volume")
        self.inv = tuple(1.0 / d for d in self.dims)
        self.root = (0.0, 0.0, 0.0) + self.dims
        self.values = [0.0] * n
        self.options = [[] for _ in range(n)]
        self.physical = [0.0] * n
        self.shapes = [None] * n
        self.feasible = []
        lengths = instance["boxes.rotated_lengths"]
        widths = instance["boxes.rotated_widths"]
        heights = instance["boxes.rotated_heights"]
        forbidden = instance["boxes.forbidden_orientations"]
        raw_values = instance["boxes.volume"]
        for j in range(n):
            if time.perf_counter() >= deadline:
                break
            try:
                value = float(raw_values[j])
                if not math.isfinite(value) or value <= 0:
                    continue
                blocked = set()
                for raw in forbidden[j]:
                    try:
                        blocked.add(int(raw))
                    except (ValueError, TypeError, OverflowError):
                        pass
                seen = set()
                for k in range(min(6, len(lengths[j]), len(widths[j]), len(heights[j]))):
                    if k in blocked:
                        continue
                    d = (float(lengths[j][k]), float(widths[j][k]), float(heights[j][k]))
                    if (d in seen or not all(math.isfinite(a) and 0 < a <= b
                                            for a, b in zip(d, self.dims))):
                        continue
                    seen.add(d)
                    self.options[j].append((k,) + d)
                if self.options[j]:
                    self.values[j] = value
                    self.physical[j] = min(math.prod(o[1:]) for o in self.options[j])
                    self.shapes[j] = tuple(sorted(o[1:] for o in self.options[j]))
                    self.feasible.append(j)
            except (ValueError, TypeError, IndexError, OverflowError):
                self.options[j] = []
        m = len(self.feasible)
        strength = math.log1p(budget)
        size_factor = 1.0 + math.sqrt(m) / 12.0
        self.space_cap = max(24, int(64 + 160 * strength / size_factor))
        self.beam_width = max(2, int(2 + 3 * strength / size_factor))
        self.branches = max(2, int(1 + strength / math.sqrt(size_factor)))
        self.pool_size = max(2, int(2 + strength))
        self.ruin_scale = strength / (2.0 + strength)
        self.rng = random.Random(n * 1000003 + int(sum(self.dims)))
        self.pool = []
        if m:
            self.minimum = tuple(min(o[a + 1] for j in self.feasible
                                     for o in self.options[j]) for a in range(3))
            self.mean_density = sum(self.values[j] for j in self.feasible) / sum(
                self.physical[j] for j in self.feasible)
        else:
            self.minimum = self.dims
            self.mean_density = 1.0

    @staticmethod
    def volume(e):
        return (e[3] - e[0]) * (e[4] - e[1]) * (e[5] - e[2])

    def subtract(self, spaces, p, stop):
        x, y, z, dx, dy, dz = p[1:7]
        X, Y, Z = x + dx, y + dy, z + dz
        pieces = set()
        for i, e in enumerate(spaces):
            if i % 32 == 0 and time.perf_counter() >= stop:
                return None
            a, b, c, A, B, C = e
            if X <= a or x >= A or Y <= b or y >= B or Z <= c or z >= C:
                pieces.add(e)
                continue
            if x > a:
                pieces.add((a, b, c, x, B, C))
            if X < A:
                pieces.add((X, b, c, A, B, C))
            if y > b:
                pieces.add((a, b, c, A, y, C))
            if Y < B:
                pieces.add((a, Y, c, A, B, C))
            if z > c:
                pieces.add((a, b, c, A, B, z))
            if Z < C:
                pieces.add((a, b, Z, A, B, C))
        minimum = self.minimum
        candidates = [e for e in pieces if all(
            e[a + 3] - e[a] >= minimum[a] for a in range(3))]
        candidates.sort(key=self.volume, reverse=True)
        maximal = []
        comparisons = 0
        for e in candidates:
            covered = False
            for q in maximal:
                comparisons += 1
                if comparisons % 1024 == 0 and time.perf_counter() >= stop:
                    return None
                if (q[0] <= e[0] and q[1] <= e[1] and q[2] <= e[2]
                        and q[3] >= e[3] and q[4] >= e[4] and q[5] >= e[5]):
                    covered = True
                    break
            if not covered:
                maximal.append(e)
        if time.perf_counter() >= stop:
            return None
        cap = self.space_cap
        if len(maximal) > cap:
            # Keep both large cavities and the cavities at both ends of each axis, rather than only the large-volume spaces.
            kept = set(maximal[:cap // 2])
            quota = max(1, (cap - len(kept)) // 6)
            for axis in range(3):
                for direction in (-1, 1):
                    ranked = sorted(maximal, key=lambda e: (
                        direction * e[axis], e[(axis + 1) % 3], -self.volume(e)))
                    added = 0
                    for e in ranked:
                        if e not in kept:
                            kept.add(e)
                            added += 1
                            if added >= quota:
                                break
            maximal = sorted(kept, key=self.volume, reverse=True)[:cap]
        return maximal

    def face_index(self, placed):
        faces = [dict() for _ in range(6)]
        for p in placed:
            lo = p[1:4]
            hi = tuple(p[a + 1] + p[a + 4] for a in range(3))
            for a in range(3):
                b, c = (a + 1) % 3, (a + 2) % 3
                rectangle = (lo[b], lo[c], hi[b], hi[c])
                faces[2 * a].setdefault(round(lo[a] * self.inv[a], 10), []).append(rectangle)
                faces[2 * a + 1].setdefault(round(hi[a] * self.inv[a], 10), []).append(rectangle)
        return faces

    def contact(self, point, dims, faces):
        ends = tuple(point[a] + dims[a] for a in range(3))
        area = 0.0
        for a in range(3):
            b, c = (a + 1) % 3, (a + 2) % 3
            face_area = dims[b] * dims[c]
            if point[a] == 0.0:
                area += face_area
            if ends[a] == self.dims[a]:
                area += face_area
            for side, plane in ((1, point[a]), (0, ends[a])):
                for r in faces[2 * a + side].get(round(plane * self.inv[a], 10), ()):
                    ob = min(ends[b], r[2]) - max(point[b], r[0])
                    oc = min(ends[c], r[3]) - max(point[c], r[1])
                    if ob > 0 and oc > 0:
                        area += ob * oc
        dx, dy, dz = dims
        return area / (2.0 * (dx * dy + dx * dz + dy * dz))

    def candidates(self, item, spaces, placed, bounds, policy, weights, axes, stop, limit):
        ranked = []
        keys = []
        faces = None
        seen = {}
        checks = 0
        inv = self.inv
        for space in spaces:
            checks += 1
            if checks % 16 == 0 and time.perf_counter() >= stop:
                break
            sx, sy, sz = (space[a + 3] - space[a] for a in range(3))
            space_volume = sx * sy * sz
            for option in self.options[item]:
                k, dx, dy, dz = option
                d = (dx, dy, dz)
                if dx > sx or dy > sy or dz > sz:
                    continue
                gaps = (sx - dx, sy - dy, sz - dz)
                fit = sum(gaps[a] * inv[a] for a in range(3))
                dead = sum(gaps[a] * inv[a] for a in range(3)
                           if 0 < gaps[a] < self.minimum[a])
                waste = space_volume / (dx * dy * dz)
                lower = space[:3]
                if policy in (0, 6, 7):
                    anchors = (lower,)
                else:
                    high = []
                    for a in range(3):
                        v = space[a + 3] - d[a]
                        if v + d[a] > space[a + 3]:
                            v = math.nextafter(v, -math.inf)
                        high.append(v)
                    anchors = ((x, y, z) for x in (lower[0], high[0])
                               for y in (lower[1], high[1]) for z in (lower[2], high[2]))
                for point in anchors:
                    checks += 1
                    if checks % 64 == 0 and time.perf_counter() >= stop:
                        return [r[1] for r in ranked]
                    if any(point[a] < space[a] or point[a] + d[a] > space[a + 3]
                           for a in range(3)):
                        continue
                    ends = tuple((point[a] + d[a]) * inv[a] for a in range(3))
                    norm = tuple(point[a] * inv[a] for a in range(3))
                    b = tuple(max(bounds[a] * inv[a], ends[a]) for a in range(3))
                    envelope = b[0] * b[1] * b[2]
                    low = norm[axes[0]] + 0.35 * norm[axes[1]] + 0.15 * norm[axes[2]]
                    if policy in (0, 7):
                        prefix = tuple(norm[a] for a in axes) + (fit, dead)
                    elif policy == 1:
                        prefix = (envelope,) + tuple(b[a] for a in axes) + (dead,)
                    elif policy == 2:
                        prefix = (waste, dead, fit)
                    elif policy == 4:
                        prefix = (dead, max(b), envelope, fit)
                    elif policy == 6:
                        prefix = (norm[axes[0]], ends[axes[0]], envelope, norm[axes[1]])
                    else:
                        prefix = ()
                    if len(keys) >= limit and prefix and prefix > keys[-1][:len(prefix)]:
                        continue
                    if faces is None:
                        faces = self.face_index(placed)
                    contact = self.contact(point, d, faces)
                    if policy == 3:
                        key = (-contact + 0.15 * dead, envelope, low, fit)
                    elif policy == 5:
                        score = (weights[0] * low + weights[1] * envelope
                                 + weights[2] * fit + weights[4] * dead - weights[3] * contact)
                        key = (score, envelope, -contact, fit)
                    else:
                        key = prefix + (-contact, low, envelope, k)
                    marker = point + (k,)
                    previous = seen.get(marker)
                    if previous is not None and previous <= key:
                        continue
                    seen[marker] = key
                    if len(keys) >= limit and key >= keys[-1]:
                        continue
                    for pos, entry in enumerate(ranked):
                        if entry[0] == marker:
                            del ranked[pos]
                            del keys[pos]
                            break
                    position = bisect.bisect_right(keys, key)
                    p = (item,) + point + d + (k,)
                    keys.insert(position, key)
                    ranked.insert(position, (marker, p))
                    if len(keys) > limit:
                        keys.pop()
                        ranked.pop()
        return [r[1] for r in ranked]

    def remember(self, records, value):
        if value > self.best_value:
            self.best_value = value
            self.best = tuple(records)

    def construct(self, order, policy, weights, axes, stop, fixed=(), width=1):
        placed = tuple(fixed)
        value = sum(self.values[p[0]] for p in placed)
        self.remember(placed, value)
        best = (value, placed)
        spaces = [self.root]
        bounds = (0.0, 0.0, 0.0)
        used = 0.0
        for p in placed:
            if time.perf_counter() >= stop:
                return best
            spaces = self.subtract(spaces, p, stop)
            if spaces is None:
                return best
            bounds = tuple(max(bounds[a], p[a + 1] + p[a + 4]) for a in range(3))
            used += p[4] * p[5] * p[6]
        beam = [(value, spaces, placed, bounds, used)]
        fixed_ids = {p[0] for p in placed}
        failed = set()
        remaining_profit = sum(self.values[j] for j in order if j not in fixed_ids)
        for j in order:
            if time.perf_counter() >= stop:
                break
            if j in fixed_ids:
                continue
            remaining_profit -= self.values[j]
            if width == 1 and self.shapes[j] in failed:
                continue
            children = []
            skipped = []
            for total, free, records, extent, physical in beam:
                if time.perf_counter() >= stop:
                    break
                options = self.candidates(j, free, records, extent, policy, weights,
                                          axes, stop, self.branches if width > 1 else 1)
                parent = (total, free, records, extent, physical)
                if width > 1 or not options:
                    children.append(parent)
                    if options:
                        skipped.append(parent)
                if not options and width == 1:
                    failed.add(self.shapes[j])
                for p in options:
                    new_total = total + self.values[j]
                    new_records = records + (p,)
                    # Update the current best feasible solution first; if the free-space update times out, this placement is not lost.
                    self.remember(new_records, new_total)
                    if new_total > best[0]:
                        best = (new_total, new_records)
                    new_free = self.subtract(free, p, stop)
                    if new_free is None:
                        return best
                    new_extent = tuple(max(extent[a], p[a + 1] + p[a + 4]) for a in range(3))
                    children.append((new_total, new_free, new_records, new_extent,
                                     physical + p[4] * p[5] * p[6]))
            if not children:
                break
            if width == 1:
                beam = children[:1]
                continue

            def rank(state):
                total, free, records, extent, physical = state
                envelope = math.prod(extent)
                void = max(0.0, envelope - physical)
                largest = max((self.volume(e) for e in free), default=0.0)
                geometry = self.mean_density * (0.045 * void - 0.006 * largest)
                return (total - geometry, total, -envelope)

            children.sort(key=rank, reverse=True)
            # Remove identical layouts and keep one branch that skips the current box to explore subsets.
            beam = []
            signatures = set()
            for state in children:
                signature = tuple(sorted(state[2]))
                if signature not in signatures:
                    signatures.add(signature)
                    beam.append(state)
                    if len(beam) >= width:
                        break
            if skipped and len(beam) == width:
                alternative = max(skipped, key=rank)
                signature = tuple(sorted(alternative[2]))
                if signature not in signatures and alternative[0] + remaining_profit > best[0]:
                    beam[-1] = alternative
        return best

    def make_order(self, iteration, parent=None, missing=None):
        rng = self.rng
        items = list(self.feasible if missing is None else missing)
        if parent is not None and rng.random() < 0.55:
            allowed = set(items)
            order = [j for j in parent if j in allowed]
            selected = set(order)
            order.extend(j for j in items if j not in selected)
            count = len(order)
            if count > 1:
                fraction = (0.04 + self.ruin_scale * 0.28 * rng.random())
                moves = min(count, max(1, int(count * fraction)))
                segment = rng.sample(range(count), moves)
                shuffled = [order[p] for p in segment]
                rng.shuffle(shuffled)
                for position, j in zip(segment, shuffled):
                    order[position] = j
                loaded = {p[0] for p in self.best}
                outsiders = [j for j in order if j not in loaded]
                if outsiders:
                    promoted = rng.choice(outsiders)
                    order.remove(promoted)
                    order.insert(rng.randrange(max(1, count // 3)), promoted)
            return order
        if iteration == 0:
            return sorted(items, key=lambda j: (-self.values[j], -self.physical[j], len(self.options[j]), j))
        if iteration == 1:
            return sorted(items, key=lambda j: (-self.physical[j], len(self.options[j]), -self.values[j]))
        if iteration == 2:
            return sorted(items, key=lambda j: (-self.values[j] / self.physical[j],
                                                len(self.options[j]), -self.values[j]))
        if iteration == 3:
            return sorted(items, key=lambda j: (len(self.options[j]),
                -max(max(o[a + 1] * self.inv[a] for a in range(3)) for o in self.options[j]),
                -self.values[j]))
        if iteration == 4:
            return sorted(items, key=lambda j: (min(o[3] for o in self.options[j]), -self.values[j]))
        a, b = rng.uniform(0.35, 1.65), rng.uniform(-0.45, 0.65)
        noise = rng.uniform(0.05, 0.65 + 0.25 * self.ruin_scale)
        return sorted(items, key=lambda j: -(a * math.log(self.values[j])
                      + b * math.log(self.physical[j]) - 0.08 * len(self.options[j])
                      + noise * rng.random()))

    def ruin(self, records, method):
        rng = self.rng
        fraction = 0.08 + (0.2 + 0.25 * self.ruin_scale) * rng.random()
        count = min(len(records), max(1, int(len(records) * fraction)))
        if method == 0:
            removed = {p[0] for p in rng.sample(list(records), count)}
        elif method == 1:
            seed = rng.choice(records)
            center = tuple((seed[a + 1] + seed[a + 4] * 0.5) * self.inv[a] for a in range(3))
            ranked = sorted(records, key=lambda p: sum(
                ((p[a + 1] + p[a + 4] * 0.5) * self.inv[a] - center[a]) ** 2
                for a in range(3)))
            removed = {p[0] for p in ranked[:count]}
        else:
            axis = rng.randrange(3)
            ranked = sorted(records, key=lambda p: p[axis + 1] + p[axis + 4] * 0.5,
                            reverse=bool(rng.randrange(2)))
            start = 0 if rng.random() < 0.5 else rng.randrange(len(records) - count + 1)
            removed = {p[0] for p in ranked[start:start + count]}
        return tuple(p for p in records if p[0] not in removed)

    def force_insertion(self, records, stop):
        selected = {p[0] for p in records}
        missing = [j for j in self.feasible if j not in selected]
        if not missing:
            return None
        rng = self.rng
        missing.sort(key=lambda j: -self.values[j] * (0.5 + rng.random()))
        item = missing[0]
        anchors = [(0.0, 0.0, 0.0)] + [p[1:4] for p in records]
        rng.shuffle(anchors)
        options = list(self.options[item])
        rng.shuffle(options)
        best = None
        best_key = None
        for option in options:
            k, dx, dy, dz = option
            d = option[1:]
            for anchor in anchors:
                if time.perf_counter() >= stop:
                    return best
                point = tuple(min(anchor[a], self.dims[a] - d[a]) for a in range(3))
                point = tuple(math.nextafter(point[a], -math.inf)
                              if point[a] + d[a] > self.dims[a] else point[a] for a in range(3))
                if min(point) < 0:
                    continue
                ends = tuple(point[a] + d[a] for a in range(3))
                if any(ends[a] > self.dims[a] for a in range(3)):
                    continue
                lost = 0.0
                conflicts = set()
                for index, p in enumerate(records):
                    if index % 64 == 0 and time.perf_counter() >= stop:
                        return best
                    if any(ends[a] <= p[a + 1] or p[a + 1] + p[a + 4] <= point[a]
                           for a in range(3)):
                        continue
                    conflicts.add(p[0])
                    lost += self.values[p[0]]
                    if best_key is not None and lost > best_key[0]:
                        break
                else:
                    key = (lost, len(conflicts), sum(point[a] * self.inv[a] for a in range(3)))
                    if best_key is None or key < best_key:
                        best_key = key
                        new_record = (item,) + point + d + (k,)
                        best = tuple(p for p in records if p[0] not in conflicts) + (new_record,)
        return best

    def run(self):
        if not self.feasible or time.perf_counter() >= self.deadline:
            return
        rng = self.rng
        iteration = 0
        initial_order = self.make_order(0)
        average_duration = self.budget / (8.0 + math.log1p(len(self.feasible)))
        axis_orders = ((2, 1, 0), (2, 0, 1), (1, 2, 0),
                       (1, 0, 2), (0, 2, 1), (0, 1, 2))
        while time.perf_counter() < self.deadline:
            now = time.perf_counter()
            remaining = self.deadline - now
            weights = (rng.uniform(0.4, 2.0), rng.uniform(0.2, 2.2),
                       rng.uniform(0.05, 1.2), rng.uniform(0.2, 2.5), rng.uniform(0.2, 1.0))
            axes = axis_orders[0] if iteration < 6 else rng.choice(axis_orders)
            policy = iteration % 6 if iteration < 6 else rng.randrange(8)
            width = 1
            fixed = ()
            parent = rng.choice(self.pool) if self.pool else None
            base_order = parent[2] if parent else initial_order
            if iteration == 0:
                allowance = remaining * 0.24
                order = initial_order
            else:
                # The per-round budget adapts to the total budget, remaining budget and measured construction time.
                allowance = min(remaining, max(average_duration * 2.5,
                                               self.budget / (6 + math.sqrt(iteration + 1))))
                if iteration < 6:
                    order = self.make_order(iteration)
                else:
                    action = rng.random()
                    if parent is not None and action < 0.66:
                        records = parent[1]
                        if action < 0.19:
                            fixed = self.force_insertion(records, min(self.deadline, now + allowance * 0.18))
                            if fixed is None:
                                fixed = self.ruin(records, rng.randrange(3))
                        else:
                            fixed = self.ruin(records, rng.randrange(3))
                        selected = {p[0] for p in fixed}
                        missing = [j for j in self.feasible if j not in selected]
                        order = self.make_order(iteration, base_order, missing)
                        if rng.random() < 0.5:
                            width = self.beam_width
                    else:
                        order = self.make_order(iteration, base_order if action < 0.88 else None)
                        if action >= 0.88:
                            width = self.beam_width
            stop = min(self.deadline, now + allowance)
            candidate = self.construct(order, policy, weights, axes, stop, fixed, width)
            value, records = candidate
            duration = time.perf_counter() - now
            average_duration = 0.7 * average_duration + 0.3 * duration
            if records:
                complete_order = [p[0] for p in fixed] + order
                signature = tuple(sorted(records))
                if not any(signature == entry[3] for entry in self.pool):
                    self.pool.append((value, records, complete_order, signature))
                    self.pool.sort(key=lambda entry: entry[0], reverse=True)
                    del self.pool[self.pool_size:]
            iteration += 1
