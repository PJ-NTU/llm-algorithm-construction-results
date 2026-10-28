"""Complementary strip support-forest packing search
Maintains support and cumulative load-bearing with exact integers; combines lateral heterogeneous forests with vertical support trees, alternating rearrangement and destroy-repair, keeping the maximum loaded-volume solution under a wall clock.
"""
import time
import math
import random
from decimal import Decimal


def _integers(values):
    parts = [Decimal(str(v)).as_tuple() for v in values]
    exponent = min([0] + [p.exponent for p in parts])
    scale = 10 ** (-exponent)
    result = []
    for p in parts:
        n = 0
        for d in p.digits:
            n = n * 10 + d
        result.append(n * 10 ** (p.exponent - exponent))
    return result, scale


def _cut(rectangles, used, deadline=math.inf):
    """Maintain maximal empty rectangles without relaxing geometric feasibility."""
    ux, uy, vx, vy = used
    result = []
    for x, y, xx, yy in rectangles:
        if time.monotonic() >= deadline:
            return []
        if vx <= x or xx <= ux or vy <= y or yy <= uy:
            result.append((x, y, xx, yy))
            continue
        if x < ux:
            result.append((x, y, ux, yy))
        if vx < xx:
            result.append((vx, y, xx, yy))
        if y < uy:
            result.append((x, y, xx, uy))
        if vy < yy:
            result.append((x, vy, xx, yy))
    # Keep only maximal rectangles and remove duplicates.
    result = list(set(result))
    result.sort(key=lambda r: (r[2] - r[0]) * (r[3] - r[1]), reverse=True)
    maximal = []
    for r in result:
        if time.monotonic() >= deadline:
            break
        if not any(t[0] <= r[0] and t[1] <= r[1]
                   and r[2] <= t[2] and r[3] <= t[3] for t in maximal):
            maximal.append(r)
    return maximal


class _Block:
    """Support forest with coplanar bases; each root's descendants remain within that root's vertical projection."""
    __slots__ = ("nodes", "counts", "a", "b", "h", "weight", "volume", "key", "roots", "area")

    def __init__(self, nodes, kinds):
        self.nodes = tuple(nodes)
        self.a = self.b = self.h = self.area = 0
        self.roots = tuple(j for j, node in enumerate(nodes) if node[4] < 0)
        self.weight = 0
        self.volume = 0
        counts = {}
        for r, x, y, z, parent, load in nodes:
            k = kinds[r]
            self.h = max(self.h, z + k[4])
            if parent < 0:
                self.a = max(self.a, x + k[2])
                self.b = max(self.b, y + k[3])
                self.area += k[2] * k[3]
            self.weight += k[5]
            self.volume += k[7]
            counts[k[0]] = counts.get(k[0], 0) + 1
        self.counts = tuple(sorted(counts.items()))
        self.key = (nodes[0][0], self.a, self.b, self.h, self.counts)


def _subtree(snapshot, children, root, kinds, deadline, limit=math.inf):
    """Extract a complete subtree from an existing feasible solution, with coordinates relative to the root box."""
    origin = snapshot[root]
    stack = [root]
    mapping = {}
    nodes = []
    while stack:
        if len(nodes) >= limit or time.monotonic() >= deadline:
            return None
        j = stack.pop()
        r, x, y, z, p, load = snapshot[j]
        mapping[j] = len(nodes)
        nodes.append((r, x - origin[1], y - origin[2], z - origin[3],
                      -1 if j == root else mapping[p], load))
        stack.extend(reversed(children[j]))
    return _Block(nodes, kinds)


class _Packing:
    def __init__(self, kinds, quantities, bounds, snapshot=(), keep=None, deadline=math.inf):
        self.kinds = kinds
        self.bounds = bounds
        self.left = quantities[:]
        self.boxes = []  # orientation record, x, y, z, parent box, total descendant weight
        self.children = {-1: []}
        self.free = {}
        self.volume = 0
        mapping = {-1: -1}
        for j, old in enumerate(snapshot):
            if time.monotonic() >= deadline:
                break
            if keep is not None and not keep[j]:
                continue
            r, x, y, z, p, _ = old
            if p not in mapping:
                continue
            p = mapping[p]
            index = len(self.boxes)
            mapping[j] = index
            self.boxes.append([r, x, y, z, p, 0])
            self.children[p].append(index)
            self.children[index] = []
            k = kinds[r]
            self.left[k[0]] -= 1
            self.volume += k[7]
        for b in reversed(self.boxes):
            if b[4] >= 0:
                self.boxes[b[4]][5] += kinds[b[0]][5] + b[5]

    def surface(self, p):
        if p < 0:
            return (0, 0, self.bounds[0], self.bounds[1], 0)
        b = self.boxes[p]
        k = self.kinds[b[0]]
        return (b[1], b[2], b[1] + k[2], b[2] + k[3], b[3] + k[4])

    def rectangles(self, p, deadline):
        if p not in self.free:
            s = self.surface(p)
            rectangles = [s[:4]]
            for j in self.children[p]:
                if time.monotonic() >= deadline:
                    rectangles = []
                    break
                b = self.boxes[j]
                k = self.kinds[b[0]]
                rectangles = _cut(rectangles, (b[1], b[2], b[1] + k[2], b[2] + k[3]), deadline)
            self.free[p] = rectangles
        return self.free[p]

    def allowance(self, p, total_weight):
        limit = total_weight
        while p >= 0:
            b = self.boxes[p]
            limit = min(limit, self.kinds[b[0]][6] - b[5])
            p = b[4]
        return limit

    def place(self, p, action, deadline):
        r, x, y, count, layers, nx = action
        k = self.kinds[r]
        typ, _, a, b, h, w, cap, v = k
        z = self.surface(p)[4]
        added = 0
        columns = 0
        tops = []
        while added < count and time.monotonic() < deadline:
            cx = x + (columns % nx) * a
            cy = y + (columns // nx) * b
            parent = p
            cz = z
            chain = []
            desired = min(layers, count - added)
            for _ in range(desired):
                if time.monotonic() >= deadline:
                    break
                j = len(self.boxes)
                self.boxes.append([r, cx, cy, cz, parent, 0])
                self.children[parent].append(j)
                self.children[j] = []
                if parent != p:
                    self.free[parent] = []
                chain.append(j)
                parent = j
                cz += h
                added += 1
                self.left[typ] -= 1
                self.volume += v
            if not chain:
                break
            for t, j in enumerate(chain):
                self.boxes[j][5] = (len(chain) - t - 1) * w
            tops.append(chain[-1])
            columns += 1
        if added:
            ancestor = p
            while ancestor >= 0:
                self.boxes[ancestor][5] += added * w
                ancestor = self.boxes[ancestor][4]
            # A prefix of a row-major grid occupies at most two rectangles.
            rows, remainder = divmod(columns, nx)
            free = self.free[p]
            if rows:
                free = _cut(free, (x, y, x + nx * a, y + rows * b), deadline)
            if remainder:
                free = _cut(free, (x, y + rows * b,
                                   x + remainder * a, y + (rows + 1) * b), deadline)
            self.free[p] = free
        return tops, added

    def place_block(self, p, block, x, y, count, nx, deadline):
        """Translate-copy a complete support tree; interrupt only between two trees."""
        z = self.surface(p)[4]
        added = 0
        surfaces = []
        occupied = []
        while added < count and time.monotonic() < deadline:
            cx = x + (added % nx) * block.a
            cy = y + (added // nx) * block.b
            offset = len(self.boxes)
            for r, bx, by, bz, parent, load in block.nodes:
                ancestor = p if parent < 0 else offset + parent
                j = len(self.boxes)
                self.boxes.append([r, cx + bx, cy + by, z + bz, ancestor, load])
                self.children[ancestor].append(j)
                self.children[j] = []
                if parent < 0 and len(block.roots) > 1:
                    k = self.kinds[r]
                    occupied.append((cx + bx, cy + by, cx + bx + k[2], cy + by + k[3]))
                self.left[self.kinds[r][0]] -= 1
                if load < self.kinds[r][6] and z + bz + self.kinds[r][4] < self.bounds[2]:
                    surfaces.append(j)
            self.volume += block.volume
            added += 1
        if added:
            ancestor = p
            while ancestor >= 0:
                self.boxes[ancestor][5] += added * block.weight
                ancestor = self.boxes[ancestor][4]
            free = self.free[p]
            if len(block.roots) > 1:
                # Deduct only real root boxes; do not seal usable gaps inside heterogeneous strips.
                for rectangle in occupied:
                    if time.monotonic() >= deadline:
                        free = []
                        break
                    free = _cut(free, rectangle, deadline)
            else:
                rows, remainder = divmod(added, nx)
                if rows:
                    free = _cut(free, (x, y, x + nx * block.a, y + rows * block.b), deadline)
                if remainder:
                    free = _cut(free, (x, y + rows * block.b,
                                       x + remainder * block.a, y + (rows + 1) * block.b), deadline)
            self.free[p] = free
        return surfaces, added


def solve(instance: dict, time_limit_s: float) -> dict:
    start = time.monotonic()
    quantities = []
    answer = {key: [] for key in ("s", "x", "y", "z", "o")}
    best = ()
    kinds = []
    scale = 1
    try:
        quantities = instance["box_quantities"]
        for key in answer:
            answer[key] = [[0] * q for q in quantities]
        budget = max(0.0, float(time_limit_s))
        if not math.isfinite(budget):
            budget = 0.0
        reserve = min(budget * 0.25, max(budget * 0.045,
                      (time.monotonic() - start) * 3 + sum(quantities) * 0.000002))
        end = start + budget - reserve
        if time.monotonic() >= end or not any(quantities):
            return answer
        ntypes = instance["num_box_types"]
        dimensions, scale = _integers(
            [instance["container_length"], instance["container_width"], instance["container_height"]]
            + instance["box_lengths"] + instance["box_widths"] + instance["box_heights"])
        L, W, H = dimensions[:3]
        lengths = dimensions[3:3 + ntypes]
        widths = dimensions[3 + ntypes:3 + 2 * ntypes]
        heights = dimensions[3 + 2 * ntypes:]
        weights, weight_scale = _integers(instance["box_weights"])
        pressures, pressure_scale = _integers(
            [p for row in instance["load_bearing_by_orientation"] for p in row])
        capacity_denominator = scale * scale * pressure_scale
        volumes = [lengths[i] * widths[i] * heights[i] for i in range(ntypes)]
        for i in range(ntypes):
            if quantities[i] == 0:
                continue
            a, b, h = lengths[i], widths[i], heights[i]
            unique = {}
            for o, (dx, dy, dz) in enumerate(((b, h, a), (a, h, b), (a, b, h)), 1):
                if not instance["orientation_allowed"][i][o - 1] or dx > L or dy > W or dz > H:
                    continue
                capacity = dx * dy * pressures[3 * i + o - 1] * weight_scale // capacity_denominator
                key = (dx, dy, dz)
                record = (i, o, dx, dy, dz, weights[i], capacity, volumes[i])
                if key not in unique or capacity > unique[key][6]:
                    unique[key] = record
            kinds.extend(unique.values())
        if not kinds or time.monotonic() >= end:
            return answer
        total_volume = L * W * H
        log_container_volume = math.log(total_volume)
        possible = {k[0] for k in kinds}
        available_volume = sum(quantities[i] * volumes[i] for i in possible)
        available_weight = sum(quantities[i] * weights[i] for i in possible)
        volume_bound = min(total_volume, available_volume)
        m = len(kinds)
        effort = 1.0 + math.log1p(budget)
        pool_size = max(1, int(math.sqrt(1.0 + budget)))
        look_width = max(1, int(effort))
        rng = random.Random(60793)
        fallback = sorted(range(m), key=lambda r: kinds[r][4])[:look_width]
        fallback += sorted(range(m), key=lambda r: math.log(kinds[r][7]) - math.log(kinds[r][5]),
                           reverse=True)[:look_width]
        links = [None] * m
        preparation_end = min(end, time.monotonic() + budget / (effort * effort + 1))
        for r, k in enumerate(kinds):
            if time.monotonic() >= preparation_end:
                break
            candidates = []
            for j, t in enumerate(kinds):
                if j % 32 == 0 and time.monotonic() >= preparation_end:
                    break
                slots = (k[2] // t[2]) * (k[3] // t[3])
                layers = min((H - k[4]) // t[4], 1 + t[6] // t[5])
                count = min(quantities[t[0]] - (t[0] == k[0]),
                            slots * layers, k[6] // t[5])
                if count > 0:
                    candidates.append((count * t[7], j))
            candidates.sort(reverse=True)
            if len(candidates) <= effort * effort:
                links[r] = [j for _, j in candidates]
            else:
                chosen = []
                seen_types = set()
                for _, j in candidates:
                    if kinds[j][0] not in seen_types:
                        chosen.append(j)
                        seen_types.add(kinds[j][0])
                        if len(chosen) >= look_width:
                            break
                chosen += [j for _, j in sorted(candidates, key=lambda pair: kinds[pair[1]][4])[:look_width]]
                chosen += [j for _, j in candidates[:look_width]]
                links[r] = list(dict.fromkeys(chosen))
        for r in range(m):
            if links[r] is None:
                links[r] = list(dict.fromkeys([r] + fallback))

        # Number and size of combination blocks vary with time limit and inventory size; combination search has no round limit.
        catalogue = [[] for _ in kinds]
        singletons = [_Block(((r, 0, 0, 0, -1, 0),), kinds) for r in range(m)]
        catalogue_width = max(2, int(effort * math.sqrt(effort + math.log1p(m))))
        block_limit = max(2, min(sum(quantities), int(effort * effort
                          * math.sqrt(1.0 + math.log1p(sum(quantities))))))
        pairs = [(r, j) for r in range(m) for j in links[r]]
        pair_cursor = 0

        def remember(block):
            if block is None or len(block.nodes) <= 1 or len(block.nodes) > block_limit:
                return
            group = catalogue[block.nodes[0][0]]
            for j, old in enumerate(group):
                if old.key == block.key:
                    # Equal usage still keeps the chance to change branch positions.
                    if rng.random() < 1.0 / effort:
                        group[j] = block
                    return
            group.append(block)
            group.sort(key=lambda b: b.volume, reverse=True)
            if len(group) > catalogue_width:
                # Keep large-volume patterns while leaving room for low-usage patterns of different heights.
                protected = max(1, catalogue_width // 2)
                del group[rng.randrange(protected, len(group))]

        # Multi-root forests are archived separately from single-root trees, so complementary bases are not wrongly dropped by first-root box type.
        mosaics = []
        mosaic_width = max(2, int(effort * effort * math.sqrt(1 + math.log1p(m))))
        target_width = max(2, int(effort * math.sqrt(m + 1)))
        targets = [(L, W, H)]
        target_set = set(targets)
        mosaic_cursor = 0
        stem_cursor = 0

        def remember_target(a, b, h):
            key = (a, b, h)
            if a <= 0 or b <= 0 or h <= 0 or key in target_set:
                return
            if len(targets) >= target_width:
                index = rng.randrange(1, len(targets))
                target_set.discard(targets[index])
                targets[index] = key
            else:
                targets.append(key)
            target_set.add(key)

        def remember_mosaic(block):
            if block is None or len(block.roots) < 2:
                return
            if any(old.key == block.key for old in mosaics):
                return
            mosaics.append(block)
            if len(mosaics) <= mosaic_width:
                return
            # Interleave candidates of different shapes, low usage and high density, without changing the single objective.
            tile = sorted(mosaics, key=lambda b:
                          (L // b.a) * (W // b.b) * b.volume, reverse=True)
            density = sorted(mosaics, key=lambda b:
                             b.volume / (b.a * b.b * H), reverse=True)
            bands = {}
            bins = max(2, int(math.sqrt(mosaic_width)))
            for b in density:
                bands.setdefault((b.a * bins // L, b.b * bins // W,
                                  b.h * bins // H), b)
            rankings = [tile, list(bands.values()), density,
                        sorted(mosaics, key=lambda b: len(b.nodes))]
            kept, seen = [], set()
            for row in zip(*[r + [None] * (len(mosaics) - len(r)) for r in rankings]):
                for b in row:
                    if b is not None and b not in seen:
                        seen.add(b)
                        kept.append(b)
                        if len(kept) >= mosaic_width:
                            mosaics[:] = kept
                            return

        def join_strips(groups, deadline):
            # Every group's roots are at z=0; only translate, no rotation or merging of support surfaces.
            nodes = []
            for block, copies, nx, ox, oy in groups:
                for c in range(copies):
                    if time.monotonic() >= deadline:
                        return None
                    x = ox + (c % nx) * block.a
                    y = oy + (c // nx) * block.b
                    offset = len(nodes)
                    for r, bx, by, bz, parent, load in block.nodes:
                        nodes.append((r, x + bx, y + by, bz,
                                      -1 if parent < 0 else offset + parent, load))
            return _Block(nodes, kinds)

        def grow_mosaics(deadline):
            nonlocal mosaic_cursor, stem_cursor
            # The largest same-type column gives lateral combinations realised rather than guessed vertical fill.
            while stem_cursor < m and time.monotonic() < deadline:
                r = stem_cursor
                stem_cursor += 1
                k = kinds[r]
                count = min(quantities[k[0]], H // k[4],
                            1 + k[6] // k[5], block_limit)
                if count > 1:
                    nodes = [(r, 0, 0, j * k[4], j - 1, (count - j - 1) * k[5])
                             for j in range(count)]
                    remember(_Block(nodes, kinds))
            representatives = [group[0] if group else singletons[r]
                               for r, group in enumerate(catalogue)]
            bank = singletons + representatives
            for group in catalogue:
                if time.monotonic() >= deadline:
                    return
                if group:
                    bank.append(rng.choice(group))
            bank += mosaics
            while time.monotonic() < deadline:
                initial = mosaic_cursor < m * m
                if initial:
                    i, j = divmod(mosaic_cursor, m)
                    first, second = representatives[i], representatives[j]
                    target = (L, W, H)
                else:
                    first, second = rng.choice(bank), rng.choice(bank)
                    target = (L, W, H) if rng.randrange(2) else rng.choice(targets)
                mosaic_cursor += 1
                if first.key == second.key:
                    continue
                rw, rh, headroom = target
                if max(first.h, second.h) > headroom:
                    continue
                na, nb = len(first.nodes), len(second.nodes)
                if na + nb > block_limit:
                    continue
                need_a, need_b = dict(first.counts), dict(second.counts)
                maximum_a = min((quantities[i] - need_b.get(i, 0)) // q
                                for i, q in first.counts)
                if maximum_a < 1:
                    continue
                best_volume, best_density = None, None
                value_key, density_key = None, None
                for axis in (0, 1):
                    if axis == 0:
                        R, P = rw, rh
                        ax, ay, bx, by = first.a, first.b, second.a, second.b
                    else:
                        R, P = rh, rw
                        ax, ay, bx, by = first.b, first.a, second.b, second.a
                    if ay > P or by > P:
                        continue
                    max_across = min((R - bx) // ax, maximum_a,
                                     (block_limit - nb) // na)
                    # The finite set of strip widths is set by inventory and adaptive block size; search rounds are not limited.
                    for across in range(1, max_across + 1):
                        if time.monotonic() >= deadline:
                            break
                        high = min(P // ay, maximum_a // across,
                                   (block_limit - nb) // (na * across))
                        if high < 1:
                            continue
                        heights_to_try = {1, high}
                        if high > 1 and not initial:
                            heights_to_try.add(rng.randint(1, high))
                        for along in heights_to_try:
                            ca = across * along
                            maximum_b = min((quantities[i] - ca * need_a.get(i, 0)) // q
                                            for i, q in second.counts)
                            maximum_b = min(maximum_b, (block_limit - ca * na) // nb)
                            max_bcross = min((R - across * ax) // bx, maximum_b)
                            max_balong = P // by
                            if max_bcross < 1 or max_balong < 1:
                                continue
                            bwidths = {max_bcross, max(1, min(max_bcross, maximum_b // max_balong))}
                            for bacross in bwidths:
                                balong = min(max_balong, maximum_b // bacross)
                                cb = bacross * balong
                                if cb < 1:
                                    continue
                                envelope = (across * ax + bacross * bx) * max(along * ay, balong * by)
                                value = ca * first.volume + cb * second.volume
                                if axis == 0:
                                    plan = ((first, ca, across, 0, 0),
                                            (second, cb, bacross, across * ax, 0))
                                else:
                                    plan = ((first, ca, along, 0, 0),
                                            (second, cb, balong, 0, across * ax))
                                vk = (value, -envelope)
                                dk = (value / (envelope * headroom), value)
                                if value_key is None or vk > value_key:
                                    value_key, best_volume = vk, plan
                                if density_key is None or dk > density_key:
                                    density_key, best_density = dk, plan
                if best_volume is not None:
                    remember_mosaic(join_strips(best_volume, deadline))
                if best_density is not None and best_density != best_volume:
                    remember_mosaic(join_strips(best_density, deadline))

        def grow_blocks(deadline):
            nonlocal pair_cursor
            if not pairs:
                return
            while time.monotonic() < deadline:
                r, t = pairs[pair_cursor % len(pairs)]
                initial = pair_cursor < len(pairs)
                pair_cursor += 1
                if not initial and rng.random() < 1.0 / effort:
                    t = rng.randrange(m)
                choices = catalogue[t]
                if choices and not initial and rng.random() > 1.0 / (effort + 1.0):
                    upper = rng.choice(choices)
                else:
                    upper = singletons[t]
                if not initial and mosaics and rng.random() < 1.0 / (effort + 1.0):
                    upper = rng.choice(mosaics)
                k = kinds[r]
                if k[4] + upper.h > H or upper.a > k[2] or upper.b > k[3]:
                    continue
                gx, gy = k[2] // upper.a, k[3] // upper.b
                copies = min(gx * gy, k[6] // upper.weight,
                             (block_limit - 1) // len(upper.nodes))
                for typ, q in upper.counts:
                    copies = min(copies, (quantities[typ] - (typ == k[0])) // q)
                if copies <= 0:
                    continue
                if not initial and copies > 1:
                    choice = rng.random()
                    if choice < 1.0 / effort:
                        copies = 1
                    elif choice < 2.0 / (effort + 1.0):
                        copies = rng.randint(1, copies)
                nx = min(gx, copies)
                if not initial and rng.randrange(2):
                    nx = (copies + gy - 1) // gy
                ny = (copies + nx - 1) // nx
                ox = 0 if initial or rng.randrange(2) else k[2] - nx * upper.a
                oy = 0 if initial or rng.randrange(2) else k[3] - ny * upper.b
                nodes = [(r, 0, 0, 0, -1, copies * upper.weight)]
                for c in range(copies):
                    offset = len(nodes)
                    x, y = ox + (c % nx) * upper.a, oy + (c // nx) * upper.b
                    for rr, xx, yy, zz, pp, load in upper.nodes:
                        nodes.append((rr, x + xx, y + yy, k[4] + zz,
                                      0 if pp < 0 else offset + pp, load))
                remember(_Block(nodes, kinds))

        def learn(state, deadline):
            if len(state.boxes) < 2:
                return
            sizes = [1] * len(state.boxes)
            for j in range(len(state.boxes) - 1, -1, -1):
                p = state.boxes[j][4]
                if p >= 0:
                    sizes[p] += sizes[j]
            roots = [j for j, b in enumerate(state.boxes) if 1 < sizes[j] <= block_limit]
            rng.shuffle(roots)
            for j in roots:
                if time.monotonic() >= deadline:
                    return
                remember(_subtree(state.boxes, state.children, j, kinds, deadline, block_limit))

        def relocate(snapshot, config, deadline):
            """Rearrange whole trees on the floor while preserving internal support and load-bearing."""
            state = _Packing(kinds, quantities, (L, W, H))
            children = {j: [] for j in range(len(snapshot))}
            roots = []
            for j, b in enumerate(snapshot):
                if b[4] < 0:
                    roots.append(j)
                else:
                    children[b[4]].append(j)
            blocks = []
            for j in roots:
                if time.monotonic() >= deadline:
                    break
                block = _subtree(snapshot, children, j, kinds, deadline)
                if block is not None:
                    blocks.append(block)
            mode = rng.randrange(3)
            def priority(b):
                metric = b.a * b.b if mode == 0 else b.volume if mode == 1 else max(b.a * W, b.b * L)
                return math.log(metric) + rng.uniform(-1.0, 1.0) * (1.0 - 1.0 / effort)
            blocks.sort(key=priority, reverse=True)
            for block in blocks:
                if time.monotonic() >= deadline:
                    break
                chosen = None
                best_fit = None
                for rx, ry, ex, ey in state.rectangles(-1, deadline):
                    rw, rh = ex - rx, ey - ry
                    if block.a > rw or block.b > rh:
                        continue
                    short = min((rw - block.a) / L, (rh - block.b) / W)
                    waste = (rw * rh - block.a * block.b) / (L * W)
                    fit = (waste, short) if mode == 1 else (short, waste)
                    if best_fit is None or fit < best_fit:
                        best_fit = fit
                        chosen = (rx if not config["corner"][0] else ex - block.a,
                                  ry if not config["corner"][1] else ey - block.b)
                if chosen is not None:
                    state.place_block(-1, block, chosen[0], chosen[1], 1, 1, deadline)
            return state

        def potential(state, r, top, capacity, used, columns, allowance):
            if capacity <= 0 or top >= H:
                return 0
            k = kinds[r]
            value = 0
            for j in links[r]:
                t = kinds[j]
                q = state.left[t[0]] - (used if t[0] == k[0] else 0)
                if q <= 0:
                    continue
                slots = (k[2] // t[2]) * (k[3] // t[3])
                levels = min((H - top) // t[4], 1 + t[6] // t[5])
                # Multiple columns share one inventory, and the total added weight must not exceed any ancestor's remaining load capacity.
                count = min(q, slots * levels * columns,
                            (capacity // t[5]) * columns, allowance // t[5])
                value = max(value, count * t[7])
            return value

        def configuration(generation, stale):
            mode = (2, 0, 1, 3)[generation] if generation < 4 else rng.randrange(4)
            noise = (1.0 - 1.0 / effort) * rng.random()
            use_blocks = generation == 1 or (generation > 3 and rng.random() < 0.6)
            return {
                "mode": mode,
                "order": generation % 3 if generation < 4 else rng.randrange(3),
                "look": 0.0 if generation == 0 or use_blocks else rng.uniform(0.3, 1.0),
                "blocks": use_blocks,
                "vh": rng.uniform(0.7, 2.7),
                "va": rng.uniform(0.05, 0.95),
                "vv": rng.uniform(0.05, 0.8),
                "contact": rng.uniform(0.0, 0.8),
                "noise": [rng.uniform(-noise, noise) for _ in kinds],
                "corner": (rng.randrange(2), rng.randrange(2)),
                "first": generation == 0,
                "partial": rng.random() < stale / (stale + effort + 1.0),
            }

        def construct(state, config, deadline):
            import heapq
            queue = []
            serial = 0
            remaining_volume = available_volume - state.volume
            remaining_weight = sum(state.left[i] * weights[i] for i in possible)
            patterns = []
            if config["blocks"]:
                for group in catalogue:
                    choices = group[:look_width]
                    if len(group) > look_width:
                        choices += rng.sample(group[look_width:], min(look_width, len(group) - look_width))
                    for block in choices:
                        if all(state.left[i] >= q for i, q in block.counts):
                            patterns.append(block)
                for block in mosaics:
                    if all(state.left[i] >= q for i, q in block.counts):
                        patterns.append(block)

            def push(p):
                nonlocal serial
                if p in state.free and not state.free[p]:
                    return
                x, y, xx, yy, z = state.surface(p)
                if z >= H:
                    return
                if p >= 0 and state.boxes[p][5] >= kinds[state.boxes[p][0]][6]:
                    return
                area = (xx - x) * (yy - y)
                if config["order"] == 0:
                    key = (z / H, -area / (L * W))
                elif config["order"] == 1:
                    key = (-z / H, -area / (L * W))
                else:
                    key = (-area * (H - z) / total_volume, z / H)
                serial += 1
                heapq.heappush(queue, (key, serial, p))

            for p in state.children:
                if time.monotonic() >= deadline:
                    return
                push(p)
            while queue and time.monotonic() < deadline:
                _, _, p = heapq.heappop(queue)
                if remaining_volume <= 0:
                    return
                z = state.surface(p)[4]
                headroom = H - z
                allow = state.allowance(p, remaining_weight)
                if allow <= 0:
                    continue
                rectangles = state.rectangles(p, deadline)
                if not rectangles:
                    continue
                if config["blocks"]:
                    for rx, ry, ex, ey in rectangles:
                        remember_target(ex - rx, ey - ry, headroom)
                chosen = None
                best_score = -math.inf
                for r, k in enumerate(kinds):
                    if time.monotonic() >= deadline:
                        break
                    typ, _, a, b, h, w, cap, v = k
                    count_limit = min(state.left[typ], allow // w)
                    if count_limit <= 0 or h > headroom:
                        continue
                    max_layers = min(headroom // h, 1 + cap // w, count_limit)
                    cache = {}
                    for rect in rectangles:
                        if time.monotonic() >= deadline:
                            break
                        rx, ry, ex, ey = rect
                        rw, rh = ex - rx, ey - ry
                        gx, gy = rw // a, rh // b
                        if gx == 0 or gy == 0:
                            continue
                        mode = config["mode"]
                        if mode == 0 or mode == 3:
                            layer_choices = (1,)
                        elif mode == 1:
                            layer_choices = (max_layers,)
                        else:
                            broad_layers = min(max_layers, max(1, count_limit // (gx * gy)))
                            layer_choices = (max_layers, broad_layers)
                        if config["partial"] and max_layers > 1 and mode in (1, 2):
                            layer_choices = tuple(set(layer_choices + (rng.randint(1, max_layers),)))
                        for layers in layer_choices:
                            if mode == 0:
                                count = 1
                            elif mode == 1:
                                count = layers
                            else:
                                count = min(count_limit, gx * gy * layers)
                            columns = (count + layers - 1) // layers
                            nx = min(gx, columns)
                            ny = (columns + nx - 1) // nx
                            area = columns * a * b
                            fill = count * v
                            future = 0
                            if config["look"]:
                                ck = (layers, count)
                                if ck not in cache:
                                    cache[ck] = potential(state, r, z + layers * h,
                                                          cap - (layers - 1) * w, count,
                                                          columns, allow - count * w)
                                future = min(cache[ck],
                                             max(0, area * (headroom - layers * h)),
                                             max(0, remaining_volume - fill))
                            vertical = fill / (area * headroom) + config["look"] * (future / (area * headroom))
                            short_loss = min((rw - nx * a) / rw, (rh - ny * b) / rh)
                            if config["first"]:
                                score = math.log(fill) - log_container_volume
                            else:
                                progress = fill / total_volume + config["look"] * (future / total_volume)
                                score = (config["vh"] * math.log(max(vertical, 1e-300))
                                         + config["va"] * (math.log(area) - math.log(rw * rh))
                                         + config["vv"] * math.log(max(progress, 1e-300))
                                         + config["contact"] * (1.0 - short_loss)
                                         + config["noise"][r])
                            if score > best_score:
                                best_score = score
                                x = rx if not config["corner"][0] else ex - nx * a
                                y = ry if not config["corner"][1] else ey - ny * b
                                chosen = (r, x, y, count, layers, nx)
                # Verified single-root trees and lateral multi-root forests compete in the same empty rectangle.
                for block in patterns:
                    if time.monotonic() >= deadline:
                        break
                    if block.h > headroom or block.weight > allow:
                        continue
                    count_limit = allow // block.weight
                    for typ, q in block.counts:
                        count_limit = min(count_limit, state.left[typ] // q)
                    if count_limit <= 0:
                        continue
                    for rx, ry, ex, ey in rectangles:
                        if time.monotonic() >= deadline:
                            break
                        rw, rh = ex - rx, ey - ry
                        gx, gy = rw // block.a, rh // block.b
                        if gx == 0 or gy == 0:
                            continue
                        count = min(count_limit, gx * gy) if config["mode"] in (2, 3) else 1
                        nx = min(gx, count)
                        ny = (count + nx - 1) // nx
                        area = count * block.area
                        fill = count * block.volume
                        vertical = fill / (area * headroom)
                        short_loss = min((rw - nx * block.a) / rw, (rh - ny * block.b) / rh)
                        r = block.nodes[0][0]
                        score = (config["vh"] * math.log(max(vertical, 1e-300))
                                 + config["va"] * (math.log(area) - math.log(rw * rh))
                                 + config["vv"] * (math.log(fill) - log_container_volume)
                                 + config["contact"] * (1.0 - short_loss)
                                 + config["noise"][r])
                        if score > best_score:
                            best_score = score
                            x = rx if not config["corner"][0] else ex - nx * block.a
                            y = ry if not config["corner"][1] else ey - ny * block.b
                            chosen = (block, x, y, count, 0, nx)
                if chosen is None:
                    continue
                if isinstance(chosen[0], _Block):
                    block, x, y, count, _, nx = chosen
                    tops, added = state.place_block(p, block, x, y, count, nx, deadline)
                    remaining_volume -= added * block.volume
                    remaining_weight -= added * block.weight
                else:
                    tops, added = state.place(p, chosen, deadline)
                    k = kinds[chosen[0]]
                    remaining_volume -= added * k[7]
                    remaining_weight -= added * k[5]
                if not added:
                    return
                push(p)
                for j in tops:
                    push(j)

        def destroy(snapshot, stale, generation):
            size = len(snapshot)
            if not size:
                return []
            # Every time limit keeps chances for both small repairs and large-scale reconstruction.
            strength = 1.0 / (1.0 + effort)
            fraction = min(1.0, strength + rng.random() ** 2
                           * (1.0 - strength) * (1.0 + stale / (stale + effort)))
            target = max(1, int(size * fraction))
            mode = generation % 5
            removed = set()
            roots = [j for j, b in enumerate(snapshot) if b[4] < 0]
            if mode == 0:
                # Remove spatially adjacent complete trees to free contiguous floor area.
                anchor = rng.choice(roots)
                b = snapshot[anchor]
                k = kinds[b[0]]
                cx, cy = 2 * b[1] + k[2], 2 * b[2] + k[3]
                ranking = sorted(roots, key=lambda j:
                    abs((2 * snapshot[j][1] + kinds[snapshot[j][0]][2]) - cx) / L
                    + abs((2 * snapshot[j][2] + kinds[snapshot[j][0]][3]) - cy) / W)
                root_count = max(1, int(len(roots) * fraction))
                removed.update(ranking[:root_count])
            elif mode == 1:
                removed.update(rng.sample(range(size), target))
            elif mode == 2:
                typ = kinds[snapshot[rng.randrange(size)][0]][0]
                removed.update(j for j, b in enumerate(snapshot) if kinds[b[0]][0] == typ)
            elif mode == 3:
                # Prefer reconstructing branches with lower vertical space utilisation.
                subtree = [kinds[b[0]][7] for b in snapshot]
                for j in range(size - 1, -1, -1):
                    p = snapshot[j][4]
                    if p >= 0:
                        subtree[p] += subtree[j]
                ranked = []
                for j, b in enumerate(snapshot):
                    k = kinds[b[0]]
                    efficiency = subtree[j] / (k[2] * k[3] * (H - b[3]))
                    ranked.append((efficiency * rng.uniform(0.5, 1.5), j))
                ranked.sort()
                removed.update(j for _, j in ranked[:target])
            else:
                # Reconstruct the suffix of the build sequence; orientation and stacking height may change.
                cut = rng.randrange(size)
                removed.update(range(cut, size))
            keep = [True] * size
            for j, b in enumerate(snapshot):
                keep[j] = j not in removed and (b[4] < 0 or keep[b[4]])
            return keep

        incumbent_volume = 0
        pool = []
        generation = 0
        stale = 0
        typical_duration = budget / (effort + math.log1p(m) + 1.0)
        while time.monotonic() < end:
            fresh = not pool or generation < 4 or rng.random() < 1.0 / (effort + 1.0)
            config = configuration(generation, stale)
            if config["blocks"]:
                growth = min(budget / ((effort + 1.0) ** 2), typical_duration / (effort + 1.0))
                growth_end = min(end, time.monotonic() + growth)
                split_end = min(growth_end, time.monotonic() + growth / (1.0 + effort))
                grow_blocks(split_end)
                grow_mosaics(growth_end)
            now = time.monotonic()
            # Per-construction budget follows the total time limit and measured cost; all rounds are wall-clock driven.
            slice_length = max(budget / (effort + math.log1p(m) + 1.0), typical_duration * 1.5)
            run_end = min(end, now + slice_length)
            if fresh:
                state = _Packing(kinds, quantities, (L, W, H))
            elif generation % 4 == 0:
                seed = best if rng.random() < 0.7 else rng.choice(pool)[1]
                move_end = min(run_end, now + slice_length / (1.0 + effort))
                state = relocate(seed, config, move_end)
            else:
                seed = best if rng.random() < 0.7 else rng.choice(pool)[1]
                keep = destroy(seed, stale, generation)
                state = _Packing(kinds, quantities, (L, W, H), seed, keep, run_end)
            construct(state, config, run_end)
            duration = time.monotonic() - now
            typical_duration = 0.75 * typical_duration + 0.25 * duration
            value = state.volume
            if value > incumbent_volume:
                incumbent_volume = value
                best = tuple(tuple(b) for b in state.boxes)
                stale = 0
            else:
                stale += 1
            if value > 0:
                snapshot = tuple(tuple(b) for b in state.boxes)
                signature = (tuple(state.left), tuple(sorted((b[0], b[1], b[2])
                              for b in state.boxes if b[4] < 0)))
                entry = (value, snapshot, signature)
                duplicate = next((j for j, e in enumerate(pool) if e[2] == signature), None)
                if duplicate is None:
                    pool.append(entry)
                elif value > pool[duplicate][0] or rng.random() < 1.0 / effort:
                    pool[duplicate] = entry
                pool.sort(key=lambda e: e[0], reverse=True)
                if len(pool) > pool_size:
                    pool.pop()
                learn(state, min(end, time.monotonic() + duration / (effort + 1.0)))
            generation += 1
            # Stop early only on reaching a provable volume upper bound, not by iteration count.
            if incumbent_volume >= volume_bound:
                break
    except Exception:
        # Search state is isolated from the incumbent; on exceptions the completed feasible solution is kept.
        pass
    try:
        used = [0] * len(quantities)
        serialized = {}
        valid = []
        for r, x, y, z, parent, _ in best:
            if parent >= 0 and not valid[parent]:
                valid.append(False)
                continue
            coordinates = []
            for coordinate in (x, y, z):
                if coordinate not in serialized:
                    if coordinate % scale == 0:
                        serialized[coordinate] = coordinate // scale
                    else:
                        value = coordinate / scale
                        numerator, denominator = Decimal(str(value)).as_integer_ratio()
                        # JSON numbers must exactly preserve the constructed decimal coordinates.
                        serialized[coordinate] = (value if numerator * scale == coordinate * denominator else None)
                coordinates.append(serialized[coordinate])
            if any(c is None for c in coordinates):
                valid.append(False)
                continue
            valid.append(True)
            k = kinds[r]
            typ = k[0]
            j = used[typ]
            used[typ] += 1
            answer["s"][typ][j] = 1
            answer["o"][typ][j] = k[1]
            answer["x"][typ][j], answer["y"][typ][j], answer["z"][typ][j] = coordinates
    except Exception:
        # On serialisation errors fall back to a complete all-zero feasible solution.
        for key in answer:
            for row in answer[key]:
                for j in range(len(row)):
                    row[j] = 0
    return answer
