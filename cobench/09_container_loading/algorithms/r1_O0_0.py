"""Maximal free space and grid-block neighbourhood search
Groups interchangeable boxes into grid blocks and searches with randomised multi-start construction and spatial remove-and-repack, keeping the best feasible solution within the wall-clock limit.
"""

import bisect
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
        self.rng = random.Random(947381)
        self.volume = math.prod(dims)
        self.minimum = tuple(min(d[a] for g in groups for d in g['dims'])
                             for a in range(3))
        self.space_cap = max(24, int(24 + 15 * math.sqrt(1 + budget) *
                                    math.log2(len(groups) + 2)))
        self.breadth = max(1, int(math.log1p(budget)) + 1)
        self.pool_cap = max(2, int(2 + math.log1p(budget)))
        self.cache_cap = max(128, int(128 * (1 + math.log1p(budget)) *
                                      math.sqrt(len(groups) + 1)))
        self.pattern_cache = {}
        self.pool = []
        self.best = []
        self.best_value = 0.0
        self.active = []
        self.counts = [0] * len(groups)
        self.value = 0.0
        self.upper = math.fsum(g['prefix'][-1] for g in groups)
        self.run_slice = budget / (3.0 + math.log2(n + 2))
        self.strength = math.log1p(budget) / (1.0 + math.log1p(budget))
        self.stale = 0

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
            self.stale = 0
        else:
            self.stale += 1
        if not self.active:
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
            if not any(t[0] <= s[0] and t[1] <= s[1] and t[2] <= s[2] and
                       t[3] >= s[3] and t[4] >= s[4] and t[5] >= s[5]
                       for t in maximal):
                maximal.append(s)
        if len(maximal) > self.space_cap:
            # Space pruning only shrinks the candidate set and never mistakes occupied space for free space.
            a, b, c = self.axis_order
            maximal.sort(key=lambda s: (s[a] / self.dims[a] +
                                        s[b] / self.dims[b] +
                                        s[c] / self.dims[c], -_space_volume(s)))
            maximal = maximal[:self.space_cap]
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
        found = set()
        for a, b, c in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
            for na in choices[a]:
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

    def choose(self, space, breadth, weights, loss_weight, atomic, contact_weight):
        side = tuple(space[a + 3] - space[a] for a in range(3))
        sv = side[0] * side[1] * side[2]
        fitting = []
        minfit = [math.inf, math.inf, math.inf]
        for gi, group in enumerate(self.groups):
            if gi % 16 == 0:
                self.check()
            remaining = group['quantity'] - self.counts[gi]
            if remaining <= 0:
                continue
            for oi, d in enumerate(group['dims']):
                if d[0] > side[0] or d[1] > side[1] or d[2] > side[2]:
                    continue
                caps = tuple(min(remaining, max(1, int(side[a] / d[a])))
                             for a in range(3))
                fitting.append((gi, oi, d, caps, remaining))
                for a in range(3):
                    minfit[a] = min(minfit[a], d[a])
        if not fitting:
            return None
        best_key = None
        selected = None
        inv = tuple(1.0 / t for t in side)
        for index, (gi, oi, d, caps, remaining) in enumerate(fitting):
            if index % 4 == 0:
                self.check()
            count = self.counts[gi]
            prefix = self.groups[gi]['prefix']
            patterns = ((1, 1, 1, 1),) if atomic else self.patterns(caps, remaining, breadth)
            for pi, (nx, ny, nz, amount) in enumerate(patterns):
                if pi % 128 == 0:
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
                contact = dx * inv[0] + dy * inv[1] + dz * inv[2]
                merit = gain * weights[gi] * (1.0 + contact_weight * contact)
                merit -= loss_weight * gain * loss / max(occupied, 1e-300)
                key = (merit, gain, contact, -dz, -dy, -dx)
                if best_key is None or key > best_key:
                    best_key = key
                    selected = (gi, oi, (nx, ny, nz))
        return selected

    def make_block(self, space, candidate):
        gi, oi, sizes = candidate
        dims = self.groups[gi]['dims'][oi]
        edges = []
        for axis in range(3):
            values = [space[axis]]
            for index in range(sizes[axis]):
                if index % 128 == 0:
                    self.check()
                endpoint = values[-1] + dims[axis]
                if endpoint > space[axis + 3]:
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
        for block in kept:
            self.counts[block[0]] += _number_of_boxes(block)
        self.value = math.fsum(g['prefix'][c] for g, c in zip(self.groups, self.counts))
        if first:
            self.axis_order = (2, 1, 0)
            breadth, loss_weight, atomic, contact_weight = 1, 4.0, False, 0.12
            weights = [1.0] * len(self.groups)
        else:
            self.axis_order = self.rng.choice(((2, 1, 0), (2, 0, 1), (1, 2, 0),
                                               (1, 0, 2), (0, 1, 2), (0, 2, 1)))
            breadth = self.rng.randint(1, self.breadth)
            loss_weight = self.rng.choice((0.0, 1.0, 2.0, 4.0, 8.0))
            atomic = self.rng.random() < 1.0 / (4.0 + math.log1p(self.budget))
            contact_weight = self.rng.uniform(0.0, 0.25)
            amplitude = self.rng.random() * (0.3 + self.strength)
            weights = [math.exp(self.rng.uniform(-amplitude, amplitude))
                       for _ in self.groups]
        spaces = [(0.0, 0.0, 0.0) + self.dims]
        start = time.perf_counter()
        completed = False
        try:
            for block in kept:
                self.check()
                spaces = self.split(spaces, _bounds(block))
            a, b, c = self.axis_order
            tight = not first and self.rng.random() < 0.25
            while spaces:
                self.check()
                if tight:
                    si = min(range(len(spaces)), key=lambda i: (
                        _space_volume(spaces[i]), spaces[i][a], spaces[i][b], spaces[i][c]))
                else:
                    si = min(range(len(spaces)), key=lambda i: (
                        spaces[i][a], spaces[i][b], spaces[i][c], -_space_volume(spaces[i])))
                space = spaces[si]
                candidate = self.choose(space, breadth, weights, loss_weight,
                                        atomic, contact_weight)
                if candidate is None:
                    spaces.pop(si)
                    continue
                block = self.make_block(space, candidate)
                if block is None:
                    spaces.pop(si)
                    continue
                self.active.append(block)
                gi = block[0]
                self.counts[gi] += _number_of_boxes(block)
                spaces = self.split(spaces, _bounds(block))
            completed = True
        except _Stop:
            pass
        self.remember()
        elapsed = time.perf_counter() - start
        if not completed and time.perf_counter() < self.deadline:
            self.run_slice = max(self.run_slice, slice_s * 1.5)
        elif completed and not kept:
            self.run_slice = max(self.run_slice, elapsed * 1.5)

    def destroy(self, base):
        self.check()
        if not base:
            return []
        amplitude = 0.15 + 0.55 * self.strength
        fraction = 0.05 + amplitude * self.rng.random()
        method = self.rng.randrange(3)
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

    def run(self):
        self.construct([], True, max(self.run_slice, 0.3 * self.budget))
        while time.perf_counter() < self.deadline:
            # When every instance with positive objective contribution and a legal orientation is packed, a provable upper bound has been reached.
            if all(c == g['quantity'] for g, c in zip(self.groups, self.counts)):
                break
            self.cutoff = self.deadline
            try:
                restart_probability = 0.25 + 0.25 * self.stale / (1.0 + self.stale)
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
        reserve = min(budget * 0.2, 0.003 + 0.000008 * n)
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
            for k in range(6):
                if k in forbidden[j]:
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
            for value, _, _ in members:
                prefix.append(prefix[-1] + value)
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
            for block in search.best:
                gi, oi, xs, ys, zs = block
                group = search.groups[gi]
                dx, dy, dz = group['dims'][oi]
                for x in xs[:-1]:
                    for y in ys[:-1]:
                        for z in zs[:-1]:
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
