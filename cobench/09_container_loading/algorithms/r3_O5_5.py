"""Inventory-aware complementary double-block packing search
Combines maximal free spaces, double-block lookahead over mixed types and orientations, multi-corner placement and elite-layout remove-and-repack, keeping the feasible layout with the largest packed volume by wall clock.
"""

import bisect
import heapq
import math
import random
import time


class _Stop(Exception):
    pass


def _number_of_boxes(block):
    return (len(block[2]) - 1) * (len(block[3]) - 1) * (len(block[4]) - 1)


def _bounds(block):
    return (block[2][0], block[3][0], block[4][0],
            block[2][-1], block[3][-1], block[4][-1])


def _space_volume(space):
    return ((space[3] - space[0]) * (space[4] - space[1]) *
            (space[5] - space[2]))


class _Search:
    def __init__(self, dims, groups, budget, deadline, n):
        self.dims = dims
        self.groups = groups
        self.budget = budget
        self.deadline = deadline
        self.cutoff = deadline
        self.rng = random.Random(947381 + n * 97 + len(groups))
        self.volume = math.prod(dims)
        self.epsilon = max(dims) * 1e-10
        self.minimum = tuple(min(d[a] for g in groups for d in g['dims'])
                             for a in range(3))
        effort = math.log1p(budget)
        self.strength = effort / (1.0 + effort)
        self.space_cap = max(24, int(24 + 15 * math.sqrt(1 + budget) *
                                    math.log2(len(groups) + 2)))
        self.breadth = max(1, int(effort) + 1)
        self.beam = max(1, int(1 + effort / (1 + 0.2 * math.log1p(len(groups)))))
        self.shortlist = max(2, int(2 + 2 * math.sqrt(effort)))
        self.anchor_cap = min(8, max(1, int(1 + 2 * math.sqrt(effort))))
        self.contact_cap = max(8, int(8 + 12 * effort))
        self.target_tries = max(2, int(2 + 2 * effort))
        self.pool_cap = max(2, int(2 + effort))
        self.cache_cap = max(128, int(128 * (1 + effort) * math.sqrt(len(groups) + 1)))
        self.pattern_cache = {}
        self.pool = []
        self.best = []
        self.best_value = 0.0
        self.best_counts = [0] * len(groups)
        self.active = []
        self.counts = [0] * len(groups)
        self.value = 0.0
        self.run_slice = budget / (3.0 + math.log2(n + 2))
        self.stale = 0
        self.axis_order = (2, 1, 0)
        self.bounds = []
        self.extent = (0.0, 0.0, 0.0)
        self.weights = [1.0] * len(groups)
        self.finished = False
        self.runs = 0
        self.paired = False
        self.pair_width = max(2, int(3 + 2.5 * math.sqrt(effort) /
                                      (1 + 0.1 * math.log1p(len(groups)))))
        self.pair_steps = max(2, int(2 + 3 * effort))
        self.pair_slice = 0.0

    def check(self):
        if time.perf_counter() >= self.cutoff:
            raise _Stop()

    def remember(self):
        # Instances in the same group are geometrically identical, so the highest-value instances of the group can always be chosen.
        value = math.fsum(g['prefix'][c] for g, c in zip(self.groups, self.counts))
        self.value = value
        if value > self.best_value:
            self.best_value = value
            self.best = self.active[:]
            self.best_counts = self.counts[:]
            self.stale = 0
        else:
            self.stale += 1
        if not self.active or time.perf_counter() >= self.deadline:
            return
        signature = tuple((b[0], b[1], len(b[2]), len(b[3]), len(b[4]),
                           b[2][0], b[3][0], b[4][0]) for b in self.active)
        sig = hash(signature)
        if any(entry[1] == sig for entry in self.pool):
            return
        self.pool.append((value, sig, self.active[:]))
        # Also update on equal-value layouts, avoiding mistaking different geometric layouts for the same search state.
        self.pool.sort(key=lambda entry: entry[0])
        if len(self.pool) > self.pool_cap:
            del self.pool[0]

    def split(self, spaces, box):
        px, py, pz, qx, qy, qz = box
        pieces = []
        for index, s in enumerate(spaces):
            if index % 32 == 0:
                self.check()
            x, y, z, u, v, w = s
            if px >= u or qx <= x or py >= v or qy <= y or pz >= w or qz <= z:
                pieces.append(s)
                continue
            if px > x:
                pieces.append((x, y, z, px, v, w))
            if qx < u:
                pieces.append((qx, y, z, u, v, w))
            if py > y:
                pieces.append((x, y, z, u, py, w))
            if qy < v:
                pieces.append((x, qy, z, u, v, w))
            if pz > z:
                pieces.append((x, y, z, u, v, pz))
            if qz < w:
                pieces.append((x, y, qz, u, v, w))
        minimum = self.minimum
        pieces = list(set(s for s in pieces
                          if s[3] - s[0] >= minimum[0] and
                          s[4] - s[1] >= minimum[1] and
                          s[5] - s[2] >= minimum[2]))
        pieces.sort(key=_space_volume, reverse=True)
        maximal = []
        for index, s in enumerate(pieces):
            if index % 16 == 0:
                self.check()
            covered = False
            for ti, t in enumerate(maximal):
                if ti % 128 == 127:
                    self.check()
                if (t[0] <= s[0] and t[1] <= s[1] and t[2] <= s[2] and
                        t[3] >= s[3] and t[4] >= s[4] and t[5] >= s[5]):
                    covered = True
                    break
            if not covered:
                maximal.append(s)
        if len(maximal) > self.space_cap:
            a, b, c = self.axis_order
            large = self.space_cap // 2
            selected = maximal[:large]
            tail = maximal[large:]
            tail.sort(key=lambda s: (s[a] / self.dims[a] +
                                    0.38 * s[b] / self.dims[b] +
                                    0.16 * s[c] / self.dims[c], -_space_volume(s)))
            maximal = selected + tail[:self.space_cap - large]
        return maximal

    def patterns(self, caps, quantity, breadth):
        key = (caps, quantity, breadth)
        saved = self.pattern_cache.get(key)
        if saved is not None:
            return saved
        choices = []
        for m in caps:
            values = {1, m}
            for k in range(1, breadth):
                values.add(max(1, m - k))
                values.add(max(1, m // (k + 1)))
            choices.append(sorted(values, reverse=True))
        found = {(1, 1, 1)}
        for a, b, c in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
            for na in choices[a]:
                self.check()
                for nb in choices[b]:
                    nc = min(caps[c], quantity // (na * nb))
                    if nc:
                        p = [0, 0, 0]
                        p[a], p[b], p[c] = na, nb, nc
                        found.add(tuple(p))
        saved = tuple((x, y, z, x * y * z) for x, y, z in
                      sorted(found, key=lambda p: p[0] * p[1] * p[2], reverse=True))
        if len(self.pattern_cache) >= self.cache_cap:
            self.pattern_cache.clear()
        self.pattern_cache[key] = saved
        return saved

    def candidates(self, space, breadth, loss_weight, fill_weight, fragmentation):
        # Single boxes and multi-box grids share one candidate pool; the more expensive face contact is computed only for a shortlist.
        side = tuple(space[a + 3] - space[a] for a in range(3))
        sv = math.prod(side)
        fitting = []
        minfit = [math.inf, math.inf, math.inf]
        for gi, group in enumerate(self.groups):
            if gi % 16 == 0:
                self.check()
            remaining = group['quantity'] - self.counts[gi]
            if remaining <= 0:
                continue
            for oi, d in enumerate(group['dims']):
                if any(d[a] > side[a] for a in range(3)):
                    continue
                caps = tuple(min(remaining, max(1, int(side[a] / d[a]))) for a in range(3))
                fitting.append((gi, oi, d, caps, remaining))
                for a in range(3):
                    minfit[a] = min(minfit[a], d[a])
        heap = []
        for index, (gi, oi, d, caps, remaining) in enumerate(fitting):
            if index % 4 == 0:
                self.check()
            count = self.counts[gi]
            prefix = self.groups[gi]['prefix']
            for pi, (nx, ny, nz, amount) in enumerate(self.patterns(caps, remaining, breadth)):
                if pi % 64 == 0:
                    self.check()
                dx, dy, dz = nx * d[0], ny * d[1], nz * d[2]
                if dx > side[0] or dy > side[1] or dz > side[2]:
                    continue
                rx, ry, rz = side[0] - dx, side[1] - dy, side[2] - dz
                ux = rx if 0 < rx < minfit[0] else 0.0
                uy = ry if 0 < ry < minfit[1] else 0.0
                uz = rz if 0 < rz < minfit[2] else 0.0
                loss = sv - (side[0] - ux) * (side[1] - uy) * (side[2] - uz)
                occupied = dx * dy * dz
                gain = prefix[count + amount] - prefix[count]
                priority = gain * self.weights[gi] / (amount ** fragmentation)
                fill = dx / side[0] + dy / side[1] + dz / side[2]
                merit = priority * (1.0 + fill_weight * fill -
                                    loss_weight * loss / max(occupied, 1e-300))
                slack = rx / self.dims[0] + ry / self.dims[1] + rz / self.dims[2]
                parts = ((gi, oi, (nx, ny, nz)),)
                item = (merit, gain, parts, (dx, dy, dz), priority, slack, -1)
                if len(heap) < self.shortlist:
                    heapq.heappush(heap, item)
                elif item > heap[0]:
                    heapq.heapreplace(heap, item)
        if self.paired and len(fitting) > 1:
            previous = self.cutoff
            self.cutoff = min(previous, time.perf_counter() + self.pair_slice)
            try:
                self.complements(side, fitting, minfit, breadth, loss_weight,
                                 fill_weight, fragmentation, heap)
            except _Stop:
                pass
            finally:
                self.cutoff = previous
        return heap

    def plane(self, first, second, quantity):
        # Jump along integer-quotient breakpoints on both grid side lengths to find the largest single-layer rectangle within inventory.
        x = min(first, quantity)
        best = (0, 0, 0)
        while x > 0:
            self.check()
            y = min(second, quantity // x)
            candidate = (x * y, x, y)
            if candidate > best:
                best = candidate
            if best[0] == quantity or (x - 1) * second <= best[0]:
                break
            quotient = quantity // x
            x = min(x - 1, quantity // (quotient + 1))
        return best

    def layer_choices(self, maximum):
        if maximum <= self.pair_steps:
            return set(range(1, maximum + 1))
        # Cover both ends, uniform quantiles and denominator breakpoints together; the width grows with the total wall-clock budget.
        result = {1, maximum}
        for i in range(1, self.pair_steps):
            result.add(max(1, maximum * i // self.pair_steps))
            result.add(max(1, maximum // (i + 1)))
            result.add(max(1, maximum - i))
        return result

    def complements(self, side, fitting, minfit, breadth, loss_weight,
                    fill_weight, fragmentation, heap):
        sv = math.prod(side)
        plane_cache = {}
        remainder_cache = {}
        seen = set()
        for axis in self.axis_order:
            self.check()
            b, c = (axis + 1) % 3, (axis + 2) % 3
            templates = []
            for index, fit in enumerate(fitting):
                if index % 8 == 0:
                    self.check()
                gi, oi, d, caps, remaining = fit
                marker = (caps[b], caps[c], remaining)
                plane = plane_cache.get(marker)
                if plane is None:
                    plane = self.plane(*marker)
                    plane_cache[marker] = plane
                amount, nb, nc = plane
                if not amount:
                    continue
                layers = min(caps[axis], remaining // amount)
                count = self.counts[gi]
                prefix = self.groups[gi]['prefix']
                coverage = amount * d[b] * d[c] / (side[b] * side[c])
                layer_gain = prefix[count + amount] - prefix[count]
                quality = layer_gain * self.weights[gi] / d[axis] * (0.25 + coverage)
                potential = (prefix[count + amount * layers] - prefix[count]) * self.weights[gi]
                grid = [1, 1, 1]
                grid[b], grid[c] = nb, nc
                templates.append((quality, potential, fit, tuple(grid), amount, layers))
            if len(templates) < 2:
                continue
            templates.sort(key=lambda t: (t[0], t[1]), reverse=True)
            half = max(1, self.pair_width // 2)
            selected = templates[:half]
            markers = {(t[2][0], t[2][1]) for t in selected}
            for t in sorted(templates, key=lambda t: (t[1], t[0]), reverse=True):
                marker = (t[2][0], t[2][1])
                if marker not in markers:
                    selected.append(t)
                    markers.add(marker)
                    if len(selected) >= self.pair_width:
                        break
            # Later restarts let orientations that did not enter the deterministic shortlist also join combinations.
            if self.runs > 1 and len(templates) > len(selected):
                alternative = self.rng.choice(templates)
                if (alternative[2][0], alternative[2][1]) not in markers:
                    selected[-1] = alternative
            for ta in selected:
                self.check()
                gi, oi, d, caps, quantity = ta[2]
                first_plane, layer_amount, max_layers = ta[3:]
                for tb in selected:
                    self.check()
                    hj, oj, e, other_caps, other_quantity = tb[2]
                    if gi == hj and oi == oj:
                        continue
                    levels = self.layer_choices(max_layers)
                    # Back-derive the layer count of the first block from full-layer boundaries of the second length, adding complementary cut points.
                    for second_layers in self.layer_choices(tb[5]):
                        remaining_length = side[axis] - second_layers * e[axis]
                        level = min(max_layers, int(remaining_length / d[axis]))
                        if level > 0:
                            levels.add(level)
                    for level in sorted(levels, reverse=True):
                        self.check()
                        take = layer_amount * level
                        left = other_quantity - (take if gi == hj else 0)
                        gap = side[axis] - level * d[axis]
                        if left <= 0 or gap < e[axis]:
                            continue
                        tail_caps = list(other_caps)
                        tail_caps[axis] = min(left, max(0, int(gap / e[axis])))
                        if tail_caps[axis] == 0:
                            continue
                        tail_caps = tuple(min(left, k) for k in tail_caps)
                        marker = (tail_caps, left, breadth)
                        tail = remainder_cache.get(marker)
                        if tail is None:
                            patterns = self.patterns(tail_caps, left, breadth)
                            tail = patterns[0]
                            remainder_cache[marker] = tail
                        nx, ny, nz, amount = tail
                        grid = list(first_plane)
                        grid[axis] = level
                        grid = tuple(grid)
                        second_grid = (nx, ny, nz)
                        parts = ((gi, oi, grid), (hj, oj, second_grid))
                        identity = (axis, parts)
                        if identity in seen:
                            continue
                        seen.add(identity)
                        first_span = tuple(grid[a] * d[a] for a in range(3))
                        second_span = tuple(second_grid[a] * e[a] for a in range(3))
                        extent = tuple(first_span[a] + second_span[a] if a == axis else
                                       max(first_span[a], second_span[a]) for a in range(3))
                        if any(extent[a] > side[a] for a in range(3)):
                            continue
                        count = self.counts[gi]
                        prefix = self.groups[gi]['prefix']
                        gain_a = prefix[count + take] - prefix[count]
                        count_b = self.counts[hj] + (take if gi == hj else 0)
                        prefix_b = self.groups[hj]['prefix']
                        gain_b = prefix_b[count_b + amount] - prefix_b[count_b]
                        gain = gain_a + gain_b
                        occupied = math.prod(first_span) + math.prod(second_span)
                        gaps = tuple(side[a] - extent[a] for a in range(3))
                        dead = tuple(gaps[a] if 0 < gaps[a] < minfit[a] else 0.0
                                     for a in range(3))
                        loss = sv - math.prod(side[a] - dead[a] for a in range(3))
                        internal = max(0.0, math.prod(extent) - occupied)
                        priority = ((gain_a * self.weights[gi] + gain_b * self.weights[hj]) /
                                    ((take + amount) ** fragmentation))
                        fill = sum(extent[a] / side[a] for a in range(3))
                        merit = priority * (1.0 + fill_weight * fill -
                                           loss_weight * (loss + 0.5 * internal) /
                                           max(occupied, 1e-300))
                        slack = sum(gaps[a] / self.dims[a] for a in range(3))
                        item = (merit, gain, parts, extent, priority, slack, axis)
                        if len(heap) < self.shortlist:
                            heapq.heappush(heap, item)
                        elif item > heap[0]:
                            heapq.heapreplace(heap, item)

    def contact(self, box):
        x, y, z, u, v, w = box
        dx, dy, dz = u - x, v - y, w - z
        eps = self.epsilon
        area = 0.0
        if x <= eps:
            area += dy * dz
        if u >= self.dims[0] - eps:
            area += dy * dz
        if y <= eps:
            area += dx * dz
        if v >= self.dims[1] - eps:
            area += dx * dz
        if z <= eps:
            area += dx * dy
        if w >= self.dims[2] - eps:
            area += dx * dy
        for p, q, r, s, t, f in self.bounds[-self.contact_cap:]:
            if abs(u - p) <= eps or abs(s - x) <= eps:
                area += max(0.0, min(v, t) - max(y, q)) * max(0.0, min(w, f) - max(z, r))
            if abs(v - q) <= eps or abs(t - y) <= eps:
                area += max(0.0, min(u, s) - max(x, p)) * max(0.0, min(w, f) - max(z, r))
            if abs(w - r) <= eps or abs(f - z) <= eps:
                area += max(0.0, min(u, s) - max(x, p)) * max(0.0, min(v, t) - max(y, q))
        return area / max(2 * (dx * dy + dx * dz + dy * dz), 1e-300)

    def select(self, spaces, breadth, loss_weight, fill_weight, fragmentation, geometry, tight):
        a, b, c = self.axis_order
        if tight:
            indices = sorted(range(len(spaces)), key=lambda i: (
                _space_volume(spaces[i]), spaces[i][a], spaces[i][b], spaces[i][c]))
        else:
            indices = sorted(range(len(spaces)), key=lambda i: (
                spaces[i][a], spaces[i][b], spaces[i][c], -_space_volume(spaces[i])))
        selected = None
        best_key = None
        dead = set()
        promising = 0
        wp, wb, ws, wc = geometry
        old_bbox = math.prod(self.extent)
        seen = set()
        for si in indices:
            self.check()
            space = spaces[si]
            candidates = self.candidates(space, breadth, loss_weight, fill_weight, fragmentation)
            if not candidates:
                dead.add(si)
                continue
            promising += 1
            for merit, gain, parts, d, priority, slack, split_axis in candidates:
                self.check()
                for mask in range(self.anchor_cap):
                    origin = list(space[:3])
                    # First try the far ends of the two minor axes, then the far end of the major axis.
                    for bit, axis in enumerate((c, b, a)):
                        if mask & (1 << bit):
                            origin[axis] = max(space[axis], space[axis + 3] - d[axis])
                    origin = tuple(origin)
                    identity = (parts, split_axis, origin)
                    if identity in seen:
                        continue
                    seen.add(identity)
                    box = origin + tuple(origin[t] + d[t] for t in range(3))
                    extent = tuple(max(self.extent[t], box[t + 3]) for t in range(3))
                    bounding = (math.prod(extent) - old_bbox) / self.volume
                    position = (origin[a] / self.dims[a] +
                                0.38 * origin[b] / self.dims[b] +
                                0.16 * origin[c] / self.dims[c])
                    compact = wc * self.contact(box) - wp * position - wb * bounding - ws * slack
                    score = merit + priority * compact
                    key = (score, gain, -position, -bounding)
                    if best_key is None or key > best_key:
                        best_key = key
                        selected = (space, parts, origin, split_axis)
            if promising >= self.beam:
                break
        if dead:
            spaces[:] = [s for i, s in enumerate(spaces) if i not in dead]
        return selected

    def make_block(self, space, candidate, origin=None):
        gi, oi, sizes = candidate
        dims = self.groups[gi]['dims'][oi]
        if origin is None:
            origin = space[:3]
        edges = []
        for axis in range(3):
            values = [max(space[axis], origin[axis])]
            for index in range(sizes[axis]):
                if index % 128 == 0:
                    self.check()
                endpoint = values[-1] + dims[axis]
                if endpoint > space[axis + 3] or endpoint <= values[-1]:
                    break
                values.append(endpoint)
            if len(values) < 2:
                return None
            edges.append(tuple(values))
        return (gi, oi, edges[0], edges[1], edges[2])

    def construct(self, kept, first, slice_s):
        self.cutoff = min(self.deadline, time.perf_counter() + slice_s)
        self.active = kept[:]
        self.counts = [0] * len(self.groups)
        self.bounds = []
        self.extent = (0.0, 0.0, 0.0)
        for block in kept:
            self.counts[block[0]] += _number_of_boxes(block)
            bound = _bounds(block)
            self.bounds.append(bound)
            self.extent = tuple(max(self.extent[a], bound[a + 3]) for a in range(3))
        rng = self.rng
        self.paired = not first and (self.runs == 1 or rng.random() < 0.45 + 0.3 * self.strength)
        self.pair_slice = slice_s / (4.0 + math.sqrt(sum(g['quantity'] for g in self.groups)))
        if first or self.runs == 1:
            self.axis_order = (2, 1, 0)
            breadth, loss_weight, fill_weight, fragmentation = 1, 4.0, 0.12, 0.0
            geometry = (0.24, 0.30, 0.08, 0.45)
            self.weights = [1.0] * len(self.groups)
        else:
            axes = [0, 1, 2]
            rng.shuffle(axes)
            self.axis_order = tuple(axes)
            breadth = rng.randint(1, self.breadth)
            loss_weight = rng.choice((0.0, 1.0, 2.0, 4.0, 8.0))
            fill_weight = rng.uniform(0.0, 0.3)
            fragmentation = rng.random() ** 3 * self.strength
            geometry = (rng.uniform(0.08, 0.6), rng.uniform(0.1, 0.8),
                        rng.uniform(0.0, 0.2), rng.uniform(0.15, 1.0))
            amplitude = rng.random() * (0.25 + self.strength)
            restriction = rng.uniform(0.0, 0.55)
            size_bias = rng.uniform(-0.35, 0.35) * self.strength
            density_bias = rng.uniform(0.0, 0.55)
            self.weights = []
            for g in self.groups:
                typical = max(g['prefix'][-1] / g['quantity'], 1e-300)
                physical = max(math.prod(g['dims'][0]) / self.volume, 1e-300)
                log_weight = (rng.uniform(-amplitude, amplitude) -
                              restriction * math.log(len(g['dims']) + 0.25) +
                              size_bias * math.log(typical) +
                              density_bias * (math.log(typical) - math.log(physical)))
                self.weights.append(math.exp(max(-8.0, min(8.0, log_weight))))
            # Promote unpacked groups forward, acting on the same grid-block candidate scoring rather than a separate decoder.
            missing = [gi for gi, g in enumerate(self.groups)
                       if self.best_counts[gi] < g['quantity']]
            if missing and rng.random() < 0.65:
                gi = rng.choice(missing)
                self.weights[gi] *= math.exp(0.3 + self.strength * rng.random())
        spaces = [(0.0, 0.0, 0.0) + self.dims]
        start = time.perf_counter()
        self.finished = False
        try:
            for bound in self.bounds:
                self.check()
                spaces = self.split(spaces, bound)
            tight = not first and rng.random() < 0.2
            while spaces:
                self.check()
                candidate = self.select(spaces, breadth, loss_weight, fill_weight,
                                        fragmentation, geometry, tight)
                if candidate is None:
                    break
                space, choices, origin, split_axis = candidate
                first_block = self.make_block(space, choices[0], origin)
                if first_block is None:
                    spaces.remove(space)
                    continue
                blocks = [first_block]
                if len(choices) == 2:
                    second_origin = list(origin)
                    # Use the actually accumulated end points, not multiplicative estimates, to guarantee the two grids are strictly separated.
                    second_origin[split_axis] = first_block[split_axis + 2][-1]
                    second_block = self.make_block(space, choices[1], second_origin)
                    if second_block is not None:
                        blocks.append(second_block)
                # Both blocks lie in the same free space with a separating axis; commit first, then subtract each actually occupied region.
                for block in blocks:
                    self.active.append(block)
                    self.counts[block[0]] += _number_of_boxes(block)
                    bound = _bounds(block)
                    self.bounds.append(bound)
                    self.extent = tuple(max(self.extent[a], bound[a + 3]) for a in range(3))
                for block in blocks:
                    spaces = self.split(spaces, _bounds(block))
            self.finished = True
        except _Stop:
            pass
        self.remember()
        self.runs += 1
        elapsed = time.perf_counter() - start
        if not self.finished and time.perf_counter() < self.deadline:
            self.run_slice = max(self.run_slice, slice_s * 1.5)
        elif self.finished and not kept:
            self.run_slice = max(self.run_slice, elapsed * 1.5)

    def erase_region(self, base, lows, highs):
        # Cut intersecting boxes along grid-cell boundaries; the remainder splits into at most six blocks.
        kept = []
        for block in base:
            self.check()
            edges = block[2:]
            if any(edges[a][-1] <= lows[a] or edges[a][0] >= highs[a] for a in range(3)):
                kept.append(block)
                continue
            first = [max(0, bisect.bisect_right(edges[a], lows[a]) - 1) for a in range(3)]
            last = [min(len(edges[a]) - 1, bisect.bisect_left(edges[a], highs[a]))
                    for a in range(3)]
            middle = list(edges)
            for axis in range(3):
                if first[axis] > 0:
                    part = middle[:]
                    part[axis] = edges[axis][:first[axis] + 1]
                    kept.append((block[0], block[1], part[0], part[1], part[2]))
                if last[axis] < len(edges[axis]) - 1:
                    part = middle[:]
                    part[axis] = edges[axis][last[axis]:]
                    kept.append((block[0], block[1], part[0], part[1], part[2]))
                middle[axis] = edges[axis][first[axis]:last[axis] + 1]
        return kept

    def insert_missing(self, base):
        counts = [0] * len(self.groups)
        for b in base:
            counts[b[0]] += _number_of_boxes(b)
        missing = [gi for gi, g in enumerate(self.groups) if counts[gi] < g['quantity']]
        if not missing:
            return None
        rng = self.rng
        gi = rng.choice(missing)
        group = self.groups[gi]
        selected = None
        best_key = None
        for _ in range(self.target_tries):
            self.check()
            oi = rng.randrange(len(group['dims']))
            d = group['dims'][oi]
            anchor = _bounds(rng.choice(base))
            origin = []
            for a in range(3):
                coordinate = rng.choice((anchor[a], anchor[a + 3],
                                         anchor[a + 3] - d[a], anchor[a] - d[a]))
                origin.append(max(0.0, min(self.dims[a] - d[a], coordinate)))
            # Cut using the actually accumulated boundaries to keep strict non-overlap.
            block = self.make_block((0.0, 0.0, 0.0) + self.dims, (gi, oi, (1, 1, 1)), origin)
            if block is None:
                continue
            box = _bounds(block)
            kept = self.erase_region(base, box[:3], box[3:])
            next_counts = [0] * len(self.groups)
            for b in kept:
                next_counts[b[0]] += _number_of_boxes(b)
            next_counts[gi] += 1
            value = math.fsum(g['prefix'][c] for g, c in zip(self.groups, next_counts))
            position = sum(box[a] / self.dims[a] for a in range(3))
            key = (value, -position)
            if best_key is None or key > best_key:
                best_key = key
                selected = kept + [block]
        return selected

    def destroy(self, base):
        self.check()
        if not base:
            return []
        amplitude = 0.15 + 0.55 * self.strength
        fraction = 0.05 + amplitude * self.rng.random()
        method = self.rng.randrange(4)
        if method == 3:
            inserted = self.insert_missing(base)
            if inserted is not None:
                return inserted
            method = 2
        if method == 0:
            # Keep the construction prefix; a truncated grid can also be cut inside rather than only removed as a whole block.
            target = max(0, int(sum(_number_of_boxes(b) for b in base) * (1 - fraction)))
            kept = []
            total = 0
            for block in base:
                self.check()
                count = _number_of_boxes(block)
                if total + count <= target:
                    kept.append(block)
                    total += count
                else:
                    axis = self.rng.randrange(3)
                    edges = block[axis + 2]
                    layer = count // (len(edges) - 1)
                    take = min(len(edges) - 1, (target - total) // layer)
                    if take > 0:
                        piece = list(block)
                        piece[axis + 2] = edges[:take + 1]
                        kept.append(tuple(piece))
                    break
            return kept
        if method == 1:
            axis = self.rng.randrange(3)
            level = self.dims[axis] * (1 - fraction)
            high = self.rng.random() < 0.5
            if high:
                level = self.dims[axis] * fraction
            kept = []
            for block in base:
                self.check()
                edges = block[axis + 2]
                if high:
                    index = bisect.bisect_left(edges, level)
                    retained = edges[index:]
                else:
                    index = bisect.bisect_right(edges, level)
                    retained = edges[:index]
                if len(retained) >= 2:
                    piece = list(block)
                    piece[axis + 2] = retained
                    kept.append(tuple(piece))
            return kept
        # Remove a spatial neighbourhood; the rest of each grid decomposes exactly into at most six solid blocks.
        selected = self.rng.choice(base)
        center = [self.rng.uniform(selected[a + 2][0], selected[a + 2][-1])
                  for a in range(3)]
        scale = fraction ** (1.0 / 3.0)
        lows = [max(0.0, center[a] - 0.5 * scale * self.dims[a]) for a in range(3)]
        highs = [min(self.dims[a], center[a] + 0.5 * scale * self.dims[a])
                 for a in range(3)]
        return self.erase_region(base, lows, highs)

    def run(self):
        self.construct([], True, max(self.run_slice, 0.22 * self.budget))
        while time.perf_counter() < self.deadline:
            # Once all legal boxes with positive contribution are packed, no selection variable can be improved.
            if all(c == g['quantity'] for g, c in zip(self.groups, self.best_counts)):
                break
            self.cutoff = self.deadline
            try:
                if not self.finished and self.best:
                    kept = self.best[:]
                elif self.runs == 1:
                    kept = []
                else:
                    restart_probability = 0.2 + 0.25 * self.stale / (1.0 + self.stale)
                    if self.pool and self.rng.random() > restart_probability:
                        base = self.best if self.rng.random() < 0.65 else self.rng.choice(self.pool)[2]
                        kept = self.destroy(base)
                    else:
                        kept = []
                self.construct(kept, False, self.run_slice)
            except _Stop:
                break


def solve(instance: dict, time_limit_s: float) -> dict:
    start = time.perf_counter()
    result = {'placement_status': [], 'coordinates': [], 'orientations': []}
    search = None
    try:
        n = max(0, int(instance['total_box_instances']))
        result = {'placement_status': [0] * n,
                  'coordinates': [[0.0, 0.0, 0.0] for _ in range(n)],
                  'orientations': [-1] * n}
        budget = float(time_limit_s)
        if not math.isfinite(budget) or budget <= 0 or n == 0:
            return result
        reserve = min(budget * 0.2, 0.005 + 0.000008 * n + 0.005 * budget)
        deadline = start + budget - reserve
        dims = (float(instance['container_length']), float(instance['container_width']),
                float(instance['container_height']))
        if not all(math.isfinite(d) and d > 0 for d in dims):
            return result
        container_volume = math.prod(dims)
        if not math.isfinite(container_volume) or container_volume <= 0:
            return result
        lengths = instance['boxes.rotated_lengths']
        widths = instance['boxes.rotated_widths']
        heights = instance['boxes.rotated_heights']
        forbidden = instance['boxes.forbidden_orientations']
        volumes = instance['boxes.volume']
        grouped = {}
        for j in range(n):
            if time.perf_counter() >= deadline:
                break
            value = float(volumes[j])
            if not math.isfinite(value) or value <= 0:
                continue
            legal = {}
            blocked = set()
            for entry in forbidden[j]:
                try:
                    blocked.add(int(entry))
                except (TypeError, ValueError, OverflowError):
                    pass
            for k in range(min(6, len(lengths[j]), len(widths[j]), len(heights[j]))):
                if k in blocked:
                    continue
                d = (float(lengths[j][k]), float(widths[j][k]), float(heights[j][k]))
                if all(math.isfinite(d[a]) and 0 < d[a] <= dims[a] for a in range(3)):
                    if d not in legal:
                        legal[d] = k
            if not legal:
                continue
            key = tuple(sorted(legal))
            entry = (value / container_volume, j, tuple(legal[d] for d in key))
            grouped.setdefault(key, []).append(entry)
        groups = []
        for key, members in grouped.items():
            if time.perf_counter() >= deadline:
                break
            members.sort(key=lambda item: (-item[0], item[1]))
            prefix = [0.0]
            for mi, (value, _, _) in enumerate(members):
                if mi % 256 == 0 and time.perf_counter() >= deadline:
                    break
                prefix.append(prefix[-1] + value)
            members = members[:len(prefix) - 1]
            if not members:
                break
            groups.append({'dims': key, 'members': members, 'prefix': prefix,
                           'quantity': len(members)})
        if not groups or time.perf_counter() >= deadline:
            return result
        search = _Search(dims, groups, budget, deadline, n)
        try:
            search.run()
        except Exception:
            # Any interruption publishes only the feasible blocks already built; free-space updates do not affect their feasibility.
            search.remember()
    except Exception:
        pass
    if search is not None:
        try:
            used = [0] * len(search.groups)
            emitted = 0
            for block in search.best:
                gi, oi, xs, ys, zs = block
                group = search.groups[gi]
                dx, dy, dz = group['dims'][oi]
                for x in xs[:-1]:
                    for y in ys[:-1]:
                        for z in zs[:-1]:
                            if emitted % 128 == 0 and time.perf_counter() >= start + budget:
                                return result
                            emitted += 1
                            _, j, mapping = group['members'][used[gi]]
                            used[gi] += 1
                            if (x >= 0 and y >= 0 and z >= 0 and
                                    x + dx <= search.dims[0] and
                                    y + dy <= search.dims[1] and z + dz <= search.dims[2]):
                                result['coordinates'][j] = [float(x), float(y), float(z)]
                                result['orientations'][j] = int(mapping[oi])
                                result['placement_status'][j] = 1
        except Exception:
            pass
    return result
