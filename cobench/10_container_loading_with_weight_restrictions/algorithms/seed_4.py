"""Anytime packing with resource-adaptive, heterogeneous support-tree grammar.

Only the standard library is required.  The incumbent objective is exclusively
original packed volume.  All numerical guards are one-sided, with no epsilon.

In addition to direct MaxRects construction and destroy/repair, the search
builds a resource-aware archive of whole support trees. Trees can have mixed
orientations, mixed levels and multiple branches. Elite subtrees enter the
same archive and can be recombined or moved as intact units. Every actual
placement still passes the exact geometry and ancestor-load checks below.
The grammar also crosses over heterogeneous crowns, fills their remaining
rectangles, and prunes archived trees against current stock, height and load.
"""
import heapq
import math
import random
import time
from fractions import Fraction


def _fraction(x):
    return Fraction(str(x))


def _lcm(a, b):
    return a // math.gcd(a, b) * b


class _Variant:
    __slots__ = ('i', 'o', 'dx', 'dy', 'dz', 'nx', 'ny', 'nz', 'weight',
                 'cap', 'volume', 'area', 'height', 'logarea', 'profiles')


class _Node:
    __slots__ = ('v', 'x', 'y', 'z', 'x1', 'y1', 'top', 'parent', 'link',
                 'slack', 'rects', 'children', 'box')


def _number(data, value):
    """Convert scaled decimal geometry to a losslessly serializable number."""
    cache = data['numbers']
    if value in cache:
        return cache[value]
    scale = data['scale']
    if value % scale == 0:
        answer = value // scale
    else:
        try:
            answer = value / scale
            f = _fraction(answer)
            if not math.isfinite(answer) or f.numerator * scale != value * f.denominator:
                answer = None
        except (OverflowError, ValueError, ZeroDivisionError):
            answer = None
    cache[value] = answer
    return answer


def _prepare(ins, end):
    q = list(ins['box_quantities'])
    dims = [_fraction(ins[k]) for k in
            ('container_length', 'container_width', 'container_height')]
    scale = 1
    for f in dims:
        scale = _lcm(scale, f.denominator)
    raw = []
    wscale = 1
    complete = True
    for i, count in enumerate(q):
        if time.monotonic() >= end:
            complete = False
            break
        if not count:
            continue
        a, b, c = [_fraction(ins[k][i]) for k in
                   ('box_lengths', 'box_widths', 'box_heights')]
        choices = {}
        for o, ds in enumerate(((b, c, a), (a, c, b), (a, b, c)), 1):
            if not ins['orientation_allowed'][i][o - 1] or any(ds[k] > dims[k] for k in range(3)):
                continue
            rate = ins['load_bearing_by_orientation'][i][o - 1]
            native = tuple(f.numerator if f.denominator == 1 else float(f) for f in ds)
            cap = min(ds[0] * ds[1] * _fraction(rate),
                      Fraction(native[0]) * Fraction(native[1]) * Fraction(rate))
            nc = native[0] * native[1] * rate
            if isinstance(nc, int) or math.isfinite(nc):
                cap = min(cap, Fraction(nc), _fraction(nc))
            if ds not in choices or cap > choices[ds][1]:
                choices[ds] = (o, cap, native)
        if not choices:
            continue
        weight = max(_fraction(ins['box_weights'][i]), Fraction(ins['box_weights'][i]))
        wscale = _lcm(wscale, weight.denominator)
        for f in (a, b, c):
            scale = _lcm(scale, f.denominator)
        raw.append((i, choices, weight))
    if not raw:
        return None
    L, W, H = (f.numerator * (scale // f.denominator) for f in dims)
    weights = {i: w.numerator * (wscale // w.denominator) for i, _, w in raw}
    mass = sum(q[i] * weights[i] for i in weights)
    data = {'q': q, 'scale': scale, 'L': L, 'W': W, 'H': H,
            'mass': mass, 'numbers': {0: 0}, 'best': [0, None], 'cursor': 0}
    variants = []
    inventory = 0
    for i, choices, _ in raw:
        volume = 0
        for ds, (o, cap, native) in choices.items():
            v = _Variant()
            v.i, v.o = i, o
            v.dx, v.dy, v.dz = (f.numerator * (scale // f.denominator) for f in ds)
            v.nx, v.ny, v.nz = native
            v.weight = weights[i]
            v.cap = min(mass, cap.numerator * wscale // cap.denominator)
            v.volume = volume = v.dx * v.dy * v.dz
            v.area = (v.dx / L) * (v.dy / W)
            v.height = v.dz / H
            v.logarea = math.log(v.dx) - math.log(L) + math.log(v.dy) - math.log(W)
            v.profiles = []
            variants.append(v)
        inventory += volume * q[i]
    data['variants'] = variants
    data['container'] = L * W * H
    data['upper'] = min(data['container'], inventory) if complete else data['container']
    data['inventory'] = inventory
    data['native'] = tuple(_number(data, x) for x in (L, W, H))
    return data


def _subtract(rects, x, y, xx, yy, end):
    pieces = []
    for j, (a, b, c, d) in enumerate(rects):
        if j % 32 == 0 and time.monotonic() >= end:
            return []  # Retiring a free face is conservative, not a relaxation.
        if xx <= a or c <= x or yy <= b or d <= y:
            pieces.append((a, b, c, d))
        else:
            if a < x:
                pieces.append((a, b, x, d))
            if xx < c:
                pieces.append((xx, b, c, d))
            if b < y:
                pieces.append((a, b, c, y))
            if yy < d:
                pieces.append((a, yy, c, d))
    if len(pieces) < 2:
        return pieces
    pieces = sorted(set(pieces), key=lambda r: (r[0], r[1], -r[2], -r[3]))
    kept = []
    for j, r in enumerate(pieces):
        if time.monotonic() >= end:
            kept.extend(pieces[j:])
            return kept
        contained = False
        for k, s in enumerate(kept):
            if k % 128 == 0 and time.monotonic() >= end:
                return pieces
            if s[0] <= r[0] and s[1] <= r[1] and r[2] <= s[2] and r[3] <= s[3]:
                contained = True
                break
        if not contained:
            kept.append(r)
    return kept


class _State:
    def __init__(self, data):
        self.data = data
        self.remaining = data['q'][:]
        self.value = 0
        n = _Node()
        n.v, n.x, n.y, n.z = -1, 0, 0, 0
        n.x1, n.y1, n.top = data['L'], data['W'], 0
        n.parent, n.link, n.slack = -1, 0, data['mass']
        n.rects = [(0, 0, n.x1, n.y1)]
        n.children = []
        n.box = (0, 0, 0, data['native'][0], data['native'][1], 0)
        self.nodes = [n]

    def budget(self, parent):
        nodes, mass = self.nodes, self.data['mass']
        p = parent if nodes[parent].slack < mass else nodes[parent].link
        result = mass
        while p:
            result = min(result, nodes[p].slack)
            p = nodes[p].link
        return result

    def geometry(self, vid, x, y, parent, end):
        d, nodes = self.data, self.nodes
        v, a = d['variants'][vid], nodes[parent]
        z, xx, yy = a.top, x + v.dx, y + v.dy
        top = z + v.dz
        if x < a.x or y < a.y or xx > a.x1 or yy > a.y1 or top > d['H']:
            return None
        nx, ny, nz = _number(d, x), _number(d, y), _number(d, z)
        if nx is None or ny is None or nz is None:
            return None
        b = (nx, ny, nz, nx + v.nx, ny + v.ny, nz + v.nz)
        ab = a.box
        if not (b[2] == ab[5] and ab[0] <= b[0] < b[3] <= ab[3]
                and ab[1] <= b[1] < b[4] <= ab[4]
                and b[2] < b[5] <= d['native'][2]):
            return None
        for j, child in enumerate(a.children):
            if j % 32 == 0 and time.monotonic() >= end:
                return None
            c = nodes[child]
            if not (xx <= c.x or c.x1 <= x or yy <= c.y or c.y1 <= y):
                return None
            cb = c.box
            if not (b[3] <= cb[0] or cb[3] <= b[0] or b[4] <= cb[1] or cb[4] <= b[1]):
                return None
        return b

    def add(self, vid, x, y, parent, end, box=None):
        if time.monotonic() >= end:
            return -1
        d, nodes = self.data, self.nodes
        v, a = d['variants'][vid], nodes[parent]
        if self.remaining[v.i] <= 0:
            return -1
        ancestors = []
        p = parent if a.slack < d['mass'] else a.link
        while p:
            if nodes[p].slack < v.weight or time.monotonic() >= end:
                return -1
            ancestors.append(p)
            p = nodes[p].link
        if box is None:
            box = self.geometry(vid, x, y, parent, end)
            if box is None:
                return -1
        n = _Node()
        n.v, n.x, n.y, n.z = vid, x, y, a.top
        n.x1, n.y1, n.top = x + v.dx, y + v.dy, a.top + v.dz
        n.parent = parent
        n.link = parent if a.slack < d['mass'] else a.link
        n.slack, n.children, n.box = v.cap, [], box
        n.rects = ([(x, y, n.x1, n.y1)] if v.cap and n.top < d['H']
                   and _number(d, n.top) == box[5] else [])
        free = _subtract(a.rects, x, y, n.x1, n.y1, end)
        # No deadline interruption inside this short, feasible-state commit.
        index = len(nodes)
        nodes.append(n)
        a.rects = free
        a.children.append(index)
        for p in ancestors:
            nodes[p].slack -= v.weight
        self.remaining[v.i] -= 1
        self.value += v.volume
        if self.value > d['best'][0]:
            d['best'][:] = [self.value, self]
        return index


def _profiles(data, end):
    """Incremental, wall-clock-budgeted shortlist of realizable child grids."""
    vs, q, H = data['variants'], data['q'], data['H']
    m, cursor, width = len(vs), data['cursor'], data['width']
    while cursor < m * m and time.monotonic() < end:
        i, j = divmod(cursor, m)
        cursor += 1
        v, u = vs[i], vs[j]
        if u.dx > v.dx or u.dy > v.dy or v.dz + u.dz > H or u.weight > v.cap:
            continue
        stock = q[u.i] - (u.i == v.i)
        levels = min((H - v.dz) // u.dz, 1 + u.cap // u.weight)
        count = min(stock, (v.dx // u.dx) * (v.dy // u.dy) * levels, v.cap // u.weight)
        if count <= 0:
            continue
        item = (count * u.volume, j)
        if len(v.profiles) < width:
            heapq.heappush(v.profiles, item)
        elif item > v.profiles[0]:
            heapq.heapreplace(v.profiles, item)
    data['cursor'] = cursor


class _Recipe:
    __slots__ = ('vid', 'levels', 'groups', 'need', 'mass', 'height', 'value')


def _recipe(d, vid, levels, groups, end):
    """A homogeneous stem with arbitrary already-feasible trees on its top.

    A group is (child recipe, copies, grid columns, relative x, relative y).
    Groups originate either from disjoint observed children or a single grid.
    Nothing here permits joint support by two boxes.
    """
    if levels < 1 or time.monotonic() >= end:
        return None
    v, q = d['variants'][vid], d['q']
    need = {v.i: levels}
    mass, value, extra_height = levels * v.weight, levels * v.volume, 0
    for child, copies, cols, x, y in groups:
        if time.monotonic() >= end:
            return None
        u = d['variants'][child.vid]
        if (copies <= 0 or cols <= 0 or x < 0 or y < 0
                or x + min(cols, copies) * u.dx > v.dx
                or y + ((copies + cols - 1) // cols) * u.dy > v.dy):
            return None
        extra_height = max(extra_height, child.height)
        mass += copies * child.mass
        value += copies * child.value
        for i, count in child.need:
            need[i] = need.get(i, 0) + copies * count
            if need[i] > q[i]:
                return None
    height = levels * v.dz + extra_height
    if (height > d['H'] or mass - v.weight > v.cap
            or need[v.i] > q[v.i]):
        return None
    p = _Recipe()
    p.vid, p.levels, p.groups = vid, levels, tuple(groups)
    p.need, p.mass, p.height, p.value = tuple(sorted(need.items())), mass, height, value
    return p


def _remember(d, p):
    """Bounded multi-view archive; never use its heuristics as the objective."""
    if p is None:
        return
    pool = d['trees'][p.vid]
    for k, old in enumerate(pool):
        if old.need == p.need:
            if old.height <= p.height:
                return
            pool.pop(k)
            break
    pool.append(p)
    width = d['tree_width']
    if len(pool) <= width:
        pool.sort(key=lambda r: r.value, reverse=True)
        return
    # Keep alternatives for low ceilings, tight loads and different scarce
    # resources, not only the globally highest-volume (usually tallest) tree.
    bands, resources = {}, {}
    nb = max(2, int(math.sqrt(width)))
    root_type = d['variants'][p.vid].i
    by_value = sorted(pool, key=lambda r: r.value, reverse=True)
    for r in by_value:
        bands.setdefault(r.height * nb // d['H'], r)
        extra = [(n / d['q'][i], i) for i, n in r.need if i != root_type]
        key = max(extra)[1] if extra else root_type
        resources.setdefault(key, r)
    rankings = [by_value,
                list(bands.values()), list(resources.values()),
                sorted(pool, key=lambda r: r.value / r.height, reverse=True),
                sorted(pool, key=lambda r: r.value / r.mass, reverse=True)]
    kept, seen, pos = [], set(), [0] * len(rankings)
    while len(kept) < width:
        advanced = False
        for k, ranking in enumerate(rankings):
            while pos[k] < len(ranking) and ranking[pos[k]] in seen:
                pos[k] += 1
            if pos[k] < len(ranking):
                r = ranking[pos[k]]
                pos[k] += 1
                seen.add(r)
                kept.append(r)
                advanced = True
                if len(kept) == width:
                    break
        if not advanced:
            break
    kept.sort(key=lambda r: r.value, reverse=True)
    d['trees'][p.vid] = kept


def _seed_trees(d, end):
    d['leaves'] = [None] * len(d['variants'])
    for j, v in enumerate(d['variants']):
        if time.monotonic() >= end:
            break
        leaf = _recipe(d, j, 1, (), end)
        d['leaves'][j] = leaf
        _remember(d, leaf)
        levels = min(d['q'][v.i], d['H'] // v.dz, 1 + v.cap // v.weight)
        if levels > 1:
            _remember(d, _recipe(d, j, levels, (), end))
    d['tree_cursor'] = 0


def _prune_tree(d, p, stock, height, allowance, depth, end):
    """Make an archived tree usable without ever breaking support closure.

    Whole child grids can lose copies, children can shrink recursively, and
    homogeneous stems can shorten. Relative horizontal coordinates do not
    change, so distinct branches remain disjoint. No real state is modified.
    """
    if time.monotonic() >= end:
        return None
    if (p.height <= height and p.mass <= allowance
            and all(stock[i] >= n for i, n in p.need)):
        return p
    v = d['variants'][p.vid]
    maximum = min(p.levels, stock[v.i], height // v.dz,
                  allowance // v.weight, 1 + v.cap // v.weight)
    if maximum < 1:
        return None
    if not p.groups or depth <= 0:
        return _recipe(d, p.vid, maximum, (), end)
    left = {i: stock[i] for i, _ in p.need}
    left[v.i] -= 1
    spare = min(v.cap, allowance - v.weight)
    groups, crown_mass, crown_height = [], 0, 0
    # Preserve the productive branches before restoring the repeated stem.
    ordered = sorted(p.groups, key=lambda g: g[0].value / g[0].mass,
                     reverse=True)
    for child, copies, cols, x, y in ordered:
        if time.monotonic() >= end:
            return None
        if (child.height > height - v.dz or child.mass > spare
                or any(left[i] < n for i, n in child.need)):
            child = _prune_tree(d, child, left, height - v.dz, spare,
                               depth - 1, end)
            if child is None:
                continue
        copies = min(copies, spare // child.mass,
                     min(left[i] // n for i, n in child.need))
        if copies < 1:
            continue
        groups.append((child, copies, cols, x, y))
        spare -= copies * child.mass
        crown_mass += copies * child.mass
        crown_height = max(crown_height, child.height)
        for i, n in child.need:
            left[i] -= copies * n
    levels = min(p.levels, 1 + left[v.i],
                 (height - crown_height) // v.dz,
                 (allowance - crown_mass) // v.weight,
                 1 + (v.cap - crown_mass) // v.weight)
    return _recipe(d, p.vid, levels, groups, end)


def _tree_choice(d, vid, stock, height, allowance, lower, end):
    """Compare full trees first; spend the remaining slice repairing rejects."""
    rejected, best = [], None
    for p in d['trees'][vid]:
        if p.value <= lower or time.monotonic() >= end:
            break
        if (p.height <= height and p.mass <= allowance
                and all(stock[i] >= n for i, n in p.need)):
            best, lower = p, p.value
            break
        rejected.append(p)
    for p in rejected:
        if p.value <= lower or time.monotonic() >= end:
            break
        trimmed = _prune_tree(d, p, stock, height, allowance,
                             d['adapt_depth'], end)
        if trimmed is not None and trimmed.value > lower:
            best, lower = trimmed, trimmed.value
    return best


def _mixed_crown(d, vid, rng, end):
    """Crossover and MaxRects repair of a real, heterogeneous tree crown.

    Unlike repeating just one child, each free rectangle can receive another
    archived tree. Entire descendant inventories and masses are debited before
    the next choice. Grids are compressed; a partial final row stays available
    to a different child type. The loop ends only at a deadline or exhaustion.
    """
    vs, v = d['variants'], d['variants'][vid]
    if v.cap <= 0 or v.dz >= d['H'] or time.monotonic() >= end:
        return None
    pool = d['trees'][vid]
    seed = rng.choice(pool) if pool and rng.random() < 0.7 else None
    levels = seed.levels if seed is not None else 1
    if levels > 1 and rng.random() < 0.65:
        levels = rng.randint(1, levels)
    available = d['q'][:]
    available[v.i] -= levels
    spare = v.cap - (levels - 1) * v.weight
    height = d['H'] - levels * v.dz
    rects, groups = [(0, 0, v.dx, v.dy)], []
    best = _recipe(d, vid, levels, groups, end)
    if best is None:
        return None
    keep = 1.0 / (1.0 + 0.3 * d['strength'])
    if seed is not None:
        for child, copies, cols, x, y in seed.groups:
            if time.monotonic() >= end:
                return best
            if rng.random() > keep or child.height > height:
                continue
            copies = min(copies, spare // child.mass,
                         min(available[i] // n for i, n in child.need))
            if copies <= 0:
                continue
            groups.append((child, copies, cols, x, y))
            spare -= copies * child.mass
            for i, count in child.need:
                available[i] -= copies * count
            p = _recipe(d, vid, levels, groups, end)
            if p is not None:
                best = p
            u = vs[child.vid]
            rects = _subtract(rects, x, y, x + min(cols, copies) * u.dx,
                              y + ((copies + cols - 1) // cols) * u.dy, end)
    order = list(range(len(vs)))
    rng.shuffle(order)
    # The learned support shortlist is inspected first, but unrelated tree
    # roots are still explored: complementary dimensions need not be common.
    preferred = [j for _, j in sorted(v.profiles, reverse=True)]
    order = list(dict.fromkeys(preferred + order))
    biases = {u.i: rng.uniform(-0.2 * d['strength'], 0.2 * d['strength'])
              for u in vs}
    wa, wt = rng.uniform(0.05, 0.85), rng.uniform(0.3, 1.8)
    while rects and spare > 0 and time.monotonic() < end:
        winner, score_best = None, -math.inf
        scan_end = min(end, time.monotonic() + d['scan_time'])
        found = 0
        for j in order:
            now = time.monotonic()
            if now >= end or (winner is not None and
                              (now >= scan_end or found >= d['breadth'])):
                break
            u = vs[j]
            if not available[u.i] or u.weight > spare or u.dz > height:
                continue
            fitting = [r for r in rects if u.dx <= r[2] - r[0]
                       and u.dy <= r[3] - r[1]]
            if not fitting:
                continue
            tree_end = min(scan_end, time.monotonic() + d['scan_time'] /
                           (1 + math.sqrt(d['breadth'])))
            child = _tree_choice(d, j, available, height, spare, 0, tree_end)
            if child is None:
                leaf = d['leaves'][j]
                if leaf is None:
                    leaf = _recipe(d, j, 1, (), end)
                child = leaf
            if child is None:
                continue
            maximum = min(spare // child.mass,
                          min(available[i] // n for i, n in child.need))
            if maximum <= 0:
                continue
            density = child.value / (u.dx * u.dy * height)
            for a, b, c, e in fitting:
                if time.monotonic() >= end:
                    break
                cols, rows = (c - a) // u.dx, (e - b) // u.dy
                copies = min(maximum, cols * rows)
                # Two complementary guillotine outlines, not extra rotations.
                for columns in {min(cols, copies), (copies + rows - 1) // rows}:
                    occupied_rows = (copies + columns - 1) // columns
                    tile = ((cols * u.dx / (c - a)) *
                            (rows * u.dy / (e - b)))
                    coverage = copies * u.dx * u.dy / (v.dx * v.dy)
                    edge = max(columns * u.dx / (c - a),
                               occupied_rows * u.dy / (e - b))
                    score = (math.log(max(density, 1e-300)) +
                             wa * math.log(max(coverage, 1e-300)) +
                             wt * math.log(max(tile, 1e-300)) +
                             0.2 * edge + biases[u.i])
                    if score > score_best:
                        score_best = score
                        winner = child, copies, columns, a, b
            found += 1
        if winner is None:
            break
        child, copies, cols, x, y = winner
        u = vs[child.vid]
        rows, tail = divmod(copies, cols)
        additions, occupied = [], []
        if rows:
            additions.append((child, rows * cols, cols, x, y))
            occupied.append((x, y, x + cols * u.dx, y + rows * u.dy))
        if tail:
            additions.append((child, tail, tail, x, y + rows * u.dy))
            occupied.append((x, y + rows * u.dy, x + tail * u.dx,
                             y + (rows + 1) * u.dy))
        groups.extend(additions)
        p = _recipe(d, vid, levels, groups, end)
        if p is None:
            break
        best = p
        spare -= copies * child.mass
        for i, count in child.need:
            available[i] -= copies * count
        for box in occupied:
            rects = _subtract(rects, *box, end)
    return best


def _catalogue(d, rng, end):
    """Anytime grammar search: stem + repeated mixed tree, or root substitution.

    Unlike a two-type block, the child here can itself contain arbitrary mixed
    levels and branches. Its *whole* mass is charged against the new stem.
    """
    vs, q = d['variants'], d['q']
    m = len(vs)
    while time.monotonic() < end:
        if rng.random() < 0.18 + 0.32 * d['strength'] / (1 + d['strength']):
            incumbent = d['best'][1]
            if incumbent is not None and rng.random() < 0.55:
                root = rng.choice(incumbent.nodes[0].children)
                vid = incumbent.nodes[root].v
            else:
                vid = rng.randrange(m)
            local_end = min(end, time.monotonic() +
                            max(d['scan_time'], (end - time.monotonic()) /
                                (2 + d['strength'])))
            _remember(d, _mixed_crown(d, vid, rng, local_end))
            continue
        cursor = d['tree_cursor']
        d['tree_cursor'] += 1
        if cursor < m * m:
            vid, j = divmod(cursor, m)
        else:
            vid = rng.randrange(m)
            choices = vs[vid].profiles
            j = rng.choice(choices)[1] if choices and rng.random() < 0.75 else rng.randrange(m)
        v, u = vs[vid], vs[j]
        if not d['trees'][j]:
            continue
        child = (d['leaves'][j] if cursor < m * m or rng.random() < 0.2
                 else rng.choice(d['trees'][j]))
        if child is None:
            continue
        # Re-seat an observed branching crown on a different bottom box.
        if child.groups and rng.random() < 0.3:
            _remember(d, _recipe(d, vid, 1, child.groups, end))
        if u.dx > v.dx or u.dy > v.dy or v.dz + child.height > d['H']:
            continue
        cols, rows = v.dx // u.dx, v.dy // u.dy
        maximum = min(cols * rows, v.cap // child.mass)
        for i, count in child.need:
            maximum = min(maximum, (q[i] - (i == v.i)) // count)
        if maximum < 1:
            continue
        # Endpoints plus a sampled intermediate count cover partial layers.
        copies = maximum if rng.random() < 0.55 else rng.randint(1, maximum)
        own = next((count for i, count in child.need if i == v.i), 0)
        levels = min((d['H'] - child.height) // v.dz,
                     1 + (v.cap - copies * child.mass) // v.weight,
                     q[v.i] - copies * own)
        columns = min(cols, copies)
        if rng.random() < 0.25:
            columns = rng.randint((copies + rows - 1) // rows, columns)
        groups = ((child, copies, columns, 0, 0),)
        _remember(d, _recipe(d, vid, 1, groups, end))
        if levels > 1:
            if rng.random() < 0.35:
                levels = rng.randint(1, levels)
            _remember(d, _recipe(d, vid, levels, groups, end))


def _learn(state, end):
    """Mine actual heterogeneous subtrees, compressing homogeneous stems."""
    if hasattr(state, 'root_trees'):
        return state.root_trees
    d, nodes, made = state.data, state.nodes, {}
    for j in range(len(nodes) - 1, 0, -1):
        if time.monotonic() >= end:
            return []
        n = nodes[j]
        groups = []
        for c in n.children:
            if c not in made:
                break
            b = nodes[c]
            groups.append((made[c], 1, 1, b.x - n.x, b.y - n.y))
        else:
            levels = 1
            if len(groups) == 1:
                child, _, _, x, y = groups[0]
                if child.vid == n.v and x == 0 and y == 0:
                    levels, groups = child.levels + 1, child.groups
            p = _recipe(d, n.v, levels, groups, end)
            if p is not None:
                made[j] = p
                _remember(d, p)
    roots = [made[j] for j in nodes[0].children if j in made]
    state.root_trees = roots
    return roots


def _execute_tree(state, root, recipe, corner, end):
    """Expand lazily: even a huge recipe never allocates all its placements."""
    stack = []
    current, p = recipe, root
    while time.monotonic() < end:
        n = state.nodes[p]
        complete = True
        for level in range(1, current.levels):
            index = state.add(current.vid, n.x, n.y, p, end)
            if index < 0:
                complete = False
                break
            p = index
        if complete:
            for group in reversed(current.groups):
                stack.append([group, p, 0])
        found = False
        while stack and time.monotonic() < end:
            group, parent, cell = stack[-1]
            child, copies, cols, ox, oy = group
            if cell >= copies:
                stack.pop()
                continue
            stack[-1][2] += 1
            v, base = state.data['variants'][child.vid], state.nodes[parent]
            iy, ix = divmod(cell, cols)
            rx, ry = ox + ix * v.dx, oy + iy * v.dy
            x = base.x1 - rx - v.dx if corner & 1 else base.x + rx
            y = base.y1 - ry - v.dy if corner & 2 else base.y + ry
            index = state.add(child.vid, x, y, parent, end)
            if index >= 0:
                current, p, found = child, index, True
                break
            if not state.remaining[v.i] or state.budget(parent) < v.weight:
                stack.pop()
        if not found:
            return


def _repack(source, rng, fraction, end):
    """Repack intact trees; unlike ruin/repair this also moves retained boxes."""
    d = source.data
    result = _State(d)
    trees = list(_learn(source, end))
    if not trees:
        return result
    mode = rng.randrange(4)
    if mode == 0:
        trees.sort(key=lambda p: d['variants'][p.vid].logarea + rng.uniform(-fraction, fraction), reverse=True)
    elif mode == 1:
        trees.sort(key=lambda p: p.value / (d['variants'][p.vid].dx * d['variants'][p.vid].dy), reverse=True)
    elif mode == 2:
        trees.sort(key=lambda p: p.height * rng.uniform(1 - fraction, 1 + fraction), reverse=True)
    else:
        rng.shuffle(trees)
    corner = rng.randrange(4)
    for recipe in trees:
        if time.monotonic() >= end:
            break
        if rng.random() < fraction * 0.3:
            continue
        if any(result.remaining[i] < n for i, n in recipe.need):
            continue
        v = d['variants'][recipe.vid]
        best, best_key = None, None
        scan_end = min(end, time.monotonic() + d['scan_time'])
        for p, face in enumerate(result.nodes):
            if time.monotonic() >= end or (best is not None and time.monotonic() >= scan_end):
                break
            if face.top + recipe.height > d['H'] or result.budget(p) < recipe.mass:
                continue
            for a, b, c, e in face.rects:
                if time.monotonic() >= end:
                    break
                rw, rh = c - a, e - b
                if v.dx > rw or v.dy > rh:
                    continue
                key = (p == 0, min((rw - v.dx) / rw, (rh - v.dy) / rh),
                       1 - v.dx * v.dy / (rw * rh))
                if best_key is not None and key >= best_key:
                    continue
                for flip in (corner, corner ^ 1, corner ^ 2, corner ^ 3):
                    x, y = (c - v.dx if flip & 1 else a), (e - v.dy if flip & 2 else b)
                    box = result.geometry(recipe.vid, x, y, p, end)
                    if box is not None:
                        best, best_key = (p, x, y, box), key
                        break
        if best is not None:
            p, x, y, box = best
            root = result.add(recipe.vid, x, y, p, end, box)
            if root >= 0:
                _execute_tree(result, root, recipe, corner, end)
    return result


def _config(data, rng, first):
    strength = data['strength']
    if first:
        return (0, 0.95, 1.15, 1.4, 0.35, 0.2, 0.9, 0, True,
                [0.0] * len(data['q']))
    spread = min(1.4, 0.18 + 0.2 * strength)
    if data['use_trees']:
        # Tree density and residual tiling matter more than single-box size.
        return (rng.choice((0, 0, 1, 2, 3)), rng.uniform(0.12, 1.1),
                rng.uniform(0.85, 1.9), rng.uniform(0.4, 2.3),
                rng.uniform(0.0, 0.9), rng.uniform(0.0, 0.45),
                rng.uniform(0.8, 1.05), rng.randrange(4), True,
                [rng.uniform(-spread, spread) for _ in data['q']])
    return (rng.randrange(4), rng.uniform(0.4, 1.5), rng.uniform(0.45, 1.7),
            rng.uniform(0.2, 2.3), rng.uniform(0.0, 0.9), rng.uniform(0.0, 0.65),
            rng.uniform(0.45, 1.05), rng.randrange(4), rng.random() < 0.7,
            [rng.uniform(-spread, spread) for _ in data['q']])


def _choose(state, parent, cfg, order, end):
    d, vs = state.data, state.data['variants']
    n = state.nodes[parent]
    allowance = state.budget(parent)
    height = d['H'] - n.top
    mode, wa, wh, wt, we, wf, optimism, corner, macro, biases = cfg
    best, best_score = None, -math.inf
    scan_end = min(end, time.monotonic() + d['scan_time'])
    found = 0
    for vid in order:
        now = time.monotonic()
        if now >= end or (best is not None and (now >= scan_end or found >= d['breadth'])):
            break
        v = vs[vid]
        count = state.remaining[v.i]
        if not count or v.weight > allowance or v.dz > height:
            continue
        fitting = [r for r in n.rects if v.dx <= r[2] - r[0] and v.dy <= r[3] - r[1]]
        if not fitting:
            continue
        k = min(count, height // v.dz, allowance // v.weight, 1 + v.cap // v.weight)
        projected = k * v.volume
        plan = (vid, k - 1, k - 1) if k > 1 else None
        child_budget = min(v.cap, allowance - v.weight)
        for _, j in v.profiles:
            if time.monotonic() >= end:
                return best
            u = vs[j]
            stock = state.remaining[u.i] - (u.i == v.i)
            levels = min((height - v.dz) // u.dz, 1 + u.cap // u.weight)
            if stock <= 0 or levels <= 0 or child_budget < u.weight:
                continue
            slots = (v.dx // u.dx) * (v.dy // u.dy)
            number = min(stock, slots * levels, child_budget // u.weight)
            value = v.volume + number * u.volume
            if value > projected:
                projected, plan = value, (j, number, levels)
        if d['use_trees']:
            tree_end = min(scan_end, time.monotonic() + d['scan_time'] /
                           (1 + math.sqrt(d['breadth'])))
            recipe = _tree_choice(d, vid, state.remaining, height, allowance,
                                  projected, tree_end)
            if recipe is not None:
                projected, plan = recipe.value, recipe
        effective_height = v.height + optimism * ((projected - v.volume) / (v.dx * v.dy * d['H']))
        base = wa * v.logarea + wh * math.log(max(effective_height, 1e-300)) + biases[v.i]
        # Every candidate is checked in BOTH geometries before it can win.
        for a, b, c, e in fitting:
            if time.monotonic() >= end:
                return best
            rw, rh = c - a, e - b
            tile = ((rw // v.dx) * v.dx / rw) * ((rh // v.dy) * v.dy / rh)
            coverage = (v.dx / rw) * (v.dy / rh)
            score = (base + wt * math.log(max(0.02 + 0.98 * tile, 1e-300))
                     + we * max(v.dx / rw, v.dy / rh) + wf * math.log(max(coverage, 1e-300)))
            if score <= best_score:
                continue
            for flip in (corner, corner ^ 1, corner ^ 2, corner ^ 3):
                x = c - v.dx if flip & 1 else a
                y = e - v.dy if flip & 2 else b
                box = state.geometry(vid, x, y, parent, end)
                if box is not None:
                    best_score, best = score, (vid, x, y, plan, box)
                    break
        found += 1
    return best


def _block(state, root, plan, corner, end):
    """Execute a mixed child grid, each grid cell being its own supported tower."""
    if plan is None:
        return
    if isinstance(plan, _Recipe):
        _execute_tree(state, root, plan, corner, end)
        return
    vid, wanted, levels = plan
    if wanted <= 0 or levels <= 0:
        return
    v = state.data['variants'][vid]
    base = state.nodes[root]
    cols, rows = (base.x1 - base.x) // v.dx, (base.y1 - base.y) // v.dy
    cell = 0
    while wanted > 0 and cell < cols * rows and time.monotonic() < end:
        iy, ix = divmod(cell, cols)
        cell += 1
        x = base.x1 - (ix + 1) * v.dx if corner & 1 else base.x + ix * v.dx
        y = base.y1 - (iy + 1) * v.dy if corner & 2 else base.y + iy * v.dy
        p, level = root, 0
        while wanted > 0 and level < levels and time.monotonic() < end:
            child = state.add(vid, x, y, p, end)
            if child < 0:
                break
            p = child
            level += 1
            wanted -= 1
        if not state.remaining[v.i] or state.budget(root) < v.weight:
            return


def _fill(state, cfg, rng, end):
    d, vs, nodes = state.data, state.data['variants'], state.nodes
    mode, _, _, _, _, _, _, _, macro, biases = cfg
    order = list(range(len(vs)))
    if d['use_trees']:
        order.sort(key=lambda j: math.log(d['trees'][j][0].value if d['trees'][j]
                                         else vs[j].volume) + biases[vs[j].i], reverse=True)
    else:
        order.sort(key=lambda j: vs[j].logarea + math.log(max(vs[j].height, 1e-300)) + biases[vs[j].i], reverse=True)
    if mode == 3:
        rng.shuffle(order)
    queue = []

    def push(p):
        n = nodes[p]
        if not n.rects:
            return
        area = ((n.x1 - n.x) / d['L']) * ((n.y1 - n.y) / d['W'])
        z = n.top / d['H']
        if mode == 0:
            priority = (-z, -area)
        elif mode == 1:
            priority = (z, -area)
        elif mode == 2:
            priority = (-area * (1 - z), z)
        else:
            priority = (-z * rng.random(), -area)
        heapq.heappush(queue, (priority, p))

    for p in range(len(nodes)):
        if time.monotonic() >= end:
            return
        push(p)
    while queue and time.monotonic() < end:
        _, p = heapq.heappop(queue)
        choice = _choose(state, p, cfg, order, end)
        if choice is None:
            continue
        vid, x, y, plan, box = choice
        first = len(nodes)
        root = state.add(vid, x, y, p, end, box)
        if root < 0:
            continue
        if macro:
            _block(state, root, plan, cfg[7], end)
        push(p)
        for j in range(first, len(nodes)):
            if time.monotonic() >= end:
                return
            push(j)
        if d['best'][0] >= d['upper']:
            return


def _ruin(source, rng, fraction, mode, end):
    nodes, d = source.nodes, source.data
    n = len(nodes) - 1
    result = _State(d)
    if not n:
        return result, [], []
    volumes = [0] + [d['variants'][b.v].volume for b in nodes[1:]]
    weights = [0] + [d['variants'][b.v].weight for b in nodes[1:]]
    roots = nodes[0].children
    for j in range(n, 0, -1):
        if time.monotonic() >= end:
            return result, [], []
        p = nodes[j].parent
        volumes[p] += volumes[j]
        weights[p] += weights[j]
    target = max(1, math.ceil(n * fraction))
    if mode == 0:
        anchor = nodes[rng.choice(roots)]
        candidates = sorted(roots, key=lambda j:
                            abs((nodes[j].x - anchor.x) / d['L'])
                            + abs((nodes[j].y - anchor.y) / d['W']))
    elif mode == 1:
        # Worst occupied column density, with noise to avoid deterministic traps.
        candidates = sorted(roots, key=lambda j:
                            (volumes[j] / ((nodes[j].x1 - nodes[j].x)
                                           * (nodes[j].y1 - nodes[j].y) * d['H']))
                            * rng.uniform(0.65, 1.45))
    elif mode == 2:
        candidates = sorted(range(1, n + 1), key=lambda j: (-nodes[j].z, rng.random()))
    elif mode == 4:
        # Release ancestor capacity by removing inefficient loaded branches,
        # rather than merely moving a random set of geometrically nearby boxes.
        loaded = [j for j in range(1, n + 1) if nodes[j].children]
        if loaded:
            loaded.sort(key=lambda j: nodes[j].slack / max(1, d['variants'][nodes[j].v].cap))
            anchor = loaded[min(len(loaded) - 1, int(rng.random() ** 2 * len(loaded)))]
            children = set(nodes[anchor].children)
        else:
            children = set()
        candidates = sorted(range(1, n + 1), key=lambda j:
                            (j not in children, volumes[j] / weights[j] * rng.uniform(0.7, 1.3)))
    elif mode == 5:
        candidates = sorted(range(1, n + 1), key=lambda j:
                            volumes[j] / ((nodes[j].x1 - nodes[j].x)
                                          * (nodes[j].y1 - nodes[j].y)
                                          * (d['H'] - nodes[j].z)) * rng.uniform(0.7, 1.3))
    else:
        candidates = list(range(1, n + 1))
        rng.shuffle(candidates)
    deleted = [False] * (n + 1)
    removed, picked = 0, []
    for root in candidates:
        if removed >= target or time.monotonic() >= end:
            break
        if deleted[root]:
            continue
        picked.append(root)
        stack = [root]
        while stack and time.monotonic() < end:
            j = stack.pop()
            if not deleted[j]:
                deleted[j] = True
                removed += 1
                stack.extend(nodes[j].children)
    mapping = [0] * (n + 1)
    for j in range(1, n + 1):
        if time.monotonic() >= end:
            return result, deleted, picked
        b = nodes[j]
        if deleted[j] or (b.parent and not mapping[b.parent]):
            deleted[j] = True
            continue
        # This is a support-closed subset at unchanged coordinates. The saved
        # native box was already verified; removing other trees cannot hurt it.
        index = result.add(b.v, b.x, b.y, mapping[b.parent], end, b.box)
        if index >= 0:
            mapping[j] = index
        else:
            deleted[j] = True
    return result, deleted, picked


def _transplant(state, source, deleted, picked, rng, end):
    """Relocate a removed subtree as a unit, then let repair fill its gaps."""
    if not picked or not deleted:
        return
    source_nodes, d = source.nodes, state.data
    root = rng.choice(picked)
    members, stack = [], [root]
    total_weight, max_top = 0, 0
    while stack and time.monotonic() < end:
        j = stack.pop()
        if not deleted[j]:
            return
        members.append(j)
        b = source_nodes[j]
        total_weight += d['variants'][b.v].weight
        max_top = max(max_top, b.top)
        stack.extend(b.children)
    if stack or time.monotonic() >= end:
        return
    members.sort()
    base = source_nodes[root]
    v = d['variants'][base.v]
    tree_height = max_top - base.z
    faces = list(range(len(state.nodes)))
    rng.shuffle(faces)
    best, best_key = None, None
    for p in faces:
        if time.monotonic() >= end:
            break
        n = state.nodes[p]
        if n.top + tree_height > d['H'] or state.budget(p) < total_weight:
            continue
        for a, b, c, e in n.rects:
            if time.monotonic() >= end:
                break
            if c - a < v.dx or e - b < v.dy or (a == base.x and b == base.y and n.top == base.z):
                continue
            key = ((c - a) * (e - b) - v.dx * v.dy, n.top)
            if best_key is not None and key >= best_key:
                continue
            box = state.geometry(base.v, a, b, p, end)
            if box is not None:
                best, best_key = (p, a, b), key
    if best is None:
        return
    p, x, y = best
    dx, dy = x - base.x, y - base.y
    mapping = {}
    for j in members:
        if time.monotonic() >= end:
            return
        b = source_nodes[j]
        if j != root and b.parent not in mapping:
            continue
        parent = p if j == root else mapping[b.parent]
        index = state.add(b.v, b.x + dx, b.y + dy, parent, end)
        if index >= 0:
            mapping[j] = index


def _search(data, duration, end):
    rng = random.Random(719329)
    strength = math.log1p(duration)
    data['strength'] = strength
    m = len(data['variants'])
    data['width'] = min(m, max(1, int((1 + strength) * math.sqrt(m))))
    data['breadth'] = min(m, max(1, int((2 + strength) * math.sqrt(m))))
    expected = sum(data['q']) / max(1.0, data['inventory'] / data['container'])
    data['scan_time'] = duration / ((2 + strength) * (1 + expected))
    data['adapt_depth'] = max(1, int((1 + strength) * math.log1p(expected)))
    data['tree_width'] = max(4, int((1 + strength) * (2 + math.log1p(m))))
    data['trees'] = [[] for _ in range(m)]
    data['use_trees'] = False
    _seed_trees(data, min(end, time.monotonic() + duration * 0.02))
    _profiles(data, min(end, time.monotonic() + duration * 0.035))
    pool_size = max(2, int(1 + strength + math.log1p(m) / (1 + strength)))
    pool = []
    attempt = 0
    typical = duration / (2 + strength)
    improved_at = time.monotonic()
    # Search has no restart/iteration ceiling. The only early exit is a proved
    # original-volume upper bound; extra wall time funds wider and deeper moves.
    while time.monotonic() < end:
        if data['best'][0] >= data['upper']:
            break
        data['use_trees'] = attempt == 1 or (attempt > 1 and rng.random() < 0.72)
        if data['use_trees']:
            catalogue_time = min(duration / (12 + 4 * strength),
                                 typical * (0.12 + 0.03 * strength))
            _catalogue(data, rng, min(end, time.monotonic() + catalogue_time))
        begin = time.monotonic()
        old_best = data['best'][0]
        stagnation = min(1.0, (begin - improved_at) / max(duration * 0.2, 1e-12))
        local_end = min(end, begin + (duration / (2 + strength * 0.5) if attempt == 0
                        else max(duration / (4 + strength), typical * (1.5 + 0.2 * strength))))
        cfg = _config(data, rng, attempt == 0)
        source = None
        if attempt != 1 and pool and rng.random() > (0.5 + 0.3 * stagnation) / (2 + strength):
            if rng.random() < 0.45:
                source = data['best'][1]
            else:
                rank = min(len(pool) - 1, int(rng.random() ** 1.5 * len(pool)))
                source = pool[rank][1]
        if source is None:
            state = _State(data)
        else:
            n = len(source.nodes) - 1
            spread = min(0.85, 0.12 + 0.12 * strength + 0.3 * stagnation)
            # Log-uniform move sizes include both fine repairs and broad escapes.
            fraction = max(1.0 / max(1, n),
                           math.expm1(rng.random() * math.log1p(n * spread)) / max(1, n))
            if rng.random() < (0.12 + 0.15 * strength / (1 + strength) + 0.12 * stagnation):
                data['use_trees'] = True
                cfg = _config(data, rng, False)
                state = _repack(source, rng, fraction, local_end)
            else:
                state, deleted, picked = _ruin(source, rng, fraction, rng.randrange(6), local_end)
                if rng.random() < 0.15 + 0.15 * strength / (1 + strength):
                    move_end = min(local_end, time.monotonic() + duration / ((3 + strength) * (2 + expected)))
                    _transplant(state, source, deleted, picked, rng, move_end)
        _fill(state, cfg, rng, local_end)
        elapsed = time.monotonic() - begin
        typical = 0.7 * typical + 0.3 * elapsed
        if state.value and state.value * 10 >= old_best * 9 and time.monotonic() < end:
            learn_time = min(duration / (20 + strength),
                             max(elapsed * 0.1, duration / (200 * (1 + strength))))
            _learn(state, min(end, time.monotonic() + learn_time))
        if state.value:
            # Equal-volume states are search alternatives, not secondary goals.
            pool.append((state.value, state))
            pool.sort(key=lambda pair: pair[0], reverse=True)
            if len(pool) > pool_size:
                if state.value == pool[pool_size - 1][0] and rng.random() < 0.5:
                    pool[pool_size - 1] = (state.value, state)
                del pool[pool_size:]
        if data['best'][0] > old_best:
            improved_at = time.monotonic()
        if data['cursor'] < m * m:
            _profiles(data, min(end, time.monotonic() + duration * 0.01 / (1 + strength)))
        attempt += 1


def solve(instance: dict, time_limit_s: float) -> dict:
    """Return complete JSON-compatible s, x, y, z, o arrays at any time budget."""
    started = time.monotonic()
    answer = {k: [] for k in ('s', 'x', 'y', 'z', 'o')}
    data = None
    try:
        quantities = instance['box_quantities']
        for key in answer:
            answer[key] = [[0] * q for q in quantities]
        duration = max(0.0, float(time_limit_s))
        if not math.isfinite(duration) or duration <= 0 or not any(quantities):
            return answer
        reserve = min(duration * 0.2, max(duration * 0.025,
                      0.001 + sum(quantities) * 0.000002,
                      2 * (time.monotonic() - started)))
        end = started + duration - reserve
        if time.monotonic() >= end:
            return answer
        data = _prepare(instance, end)
        if data is not None and time.monotonic() < end:
            _search(data, duration, end)
    except Exception:
        # All search changes are feasible insertions into fresh states. A
        # numerical/resource failure never replaces the valid incumbent.
        pass
    try:
        if data is not None and data['best'][1] is not None:
            used = [0] * len(answer['s'])
            for node in data['best'][1].nodes[1:]:
                v = data['variants'][node.v]
                i, k = v.i, used[v.i]
                answer['x'][i][k], answer['y'][i][k], answer['z'][i][k] = node.box[:3]
                answer['o'][i][k] = v.o
                answer['s'][i][k] = 1
                used[i] += 1
    except Exception:
        # Parents precede children, so any completely exported prefix remains
        # support-closed and has no greater loads than the full incumbent.
        pass
    return answer
