"""Expanded-slot augmenting-matching packing
Combines empty-rectangle repair, bitmask ejection and two-axis compaction, expands adjacent gaps into mutually exclusive slots, chains piece swaps along augmenting paths and keeps the best feasible layout.
"""

import math
import random
import time


_clock = time.perf_counter
_inf = float('inf')


def _empty(n):
    return {
        'selection': [0] * n,
        'bottom_left_x': [0.0] * n,
        'bottom_left_y': [0.0] * n,
        'left_relations': [[0] * n for _ in range(n)],
        'below_relations': [[0] * n for _ in range(n)],
    }


def _sub(end, size):
    x = end - size
    return math.nextafter(x, -_inf) if x + size > end else x


def _edge_add(edges, q):
    _, x, y, X, Y = q
    for table, key, segment in ((edges[0], x, (y, Y)), (edges[1], X, (y, Y)),
                                (edges[2], y, (x, X)), (edges[3], Y, (x, X))):
        table.setdefault(key, []).append(segment)


def _contact(edges, x, y, X, Y, W, H):
    value = (Y-y) * ((x == 0) + (X == W)) + (X-x) * ((y == 0) + (Y == H))
    for table, key, a, b in ((edges[1], x, y, Y), (edges[0], X, y, Y),
                            (edges[3], y, x, X), (edges[2], Y, x, X)):
        for c, d in table.get(key, ()):
            value += max(0.0, min(b, d) - max(a, c))
    return value


def _parameters(rnd, strength, base=None):
    # Parameters act on the same sorter and placer at once, without switching between two algorithms.
    ranges = {
        'alpha': (0.82, 1.45, 0.09), 'price': (0.60, 1.16, 0.10),
        'power': (0.0, 0.92, 0.16), 'mix': (0.04, 0.86, 0.17),
        'eta': (0.0, 0.25, 0.06), 'noise': (0.0, 0.09, 0.025),
        'balance': (0.015, 0.38, 0.08), 'waste': (0.0, 0.45, 0.10),
        'contact': (0.06, 0.48, 0.10), 'preserve': (0.03, 3.5, 0.40),
        'memory': (0.0, min(0.30, 0.05 + 0.055*strength), 0.05),
    }
    p = {}
    for key, (low, high, step) in ranges.items():
        if base is None:
            u = rnd.random()
            p[key] = low + (high-low) * (u*u if key == 'preserve' else u)
        else:
            p[key] = min(high, max(low, base[key] + rnd.uniform(-step, step)))
    p['geo'] = rnd.randrange(5) if base is None or rnd.random() < 0.23 else base['geo']
    p['corner'] = rnd.randrange(4) if base is None or rnd.random() < 0.25 else base['corner']
    p['vertical'] = rnd.random() < 0.25 if base is None else base['vertical']
    return p


class _Packing:
    def __init__(self, w, h, v, W, H, cand, budget):
        self.w, self.h, self.v = w, h, v
        self.W, self.H, self.cand = W, H, cand
        self.area = [a*b for a, b in zip(w, h)]
        self.rho = [0.0] * len(w)
        self.logarea = [0.0] * len(w)
        for i in cand:
            self.rho[i] = v[i] / self.area[i]
            self.logarea[i] = math.log(self.area[i])
        self.density = sorted(cand, key=lambda i: (-self.rho[i], -v[i]))
        self.rich = sorted(cand, key=lambda i: (-v[i], self.area[i], i))
        self.groups = {}
        for i in self.rich:
            self.groups.setdefault((w[i], h[i]), []).append(i)
        self.mw, self.mh = min(w[i] for i in cand), min(h[i] for i in cand)
        strength = math.log1p(budget)
        self.cap = max(24, int(14 * math.sqrt(len(cand)) * math.sqrt(1 + strength)))
        self.hosts = max(3, int(3 + math.sqrt(strength) + 0.3*math.log1p(len(cand))))
        self.big_count = max(1, int(1 + math.sqrt(strength)))
        self.price = self._price(self.density, W*H)

    def _price(self, ids, room):
        acc = 0.0
        for i in ids:
            acc += self.area[i]
            if acc >= room:
                return self.rho[i]
        return 0.0

    def split(self, free, box, deadline):
        x, y, X, Y = box
        keep, pieces = [], []
        mw, mh = self.mw, self.mh
        for a, b, c, d in free:
            if x >= c or X <= a or y >= d or Y <= b:
                keep.append((a, b, c, d))
            else:
                if x-a >= mw and d-b >= mh:
                    pieces.append((a, b, x, d))
                if c-X >= mw and d-b >= mh:
                    pieces.append((X, b, c, d))
                if y-b >= mh and c-a >= mw:
                    pieces.append((a, b, c, y))
                if d-Y >= mh and c-a >= mw:
                    pieces.append((a, Y, c, d))
        pieces = sorted(set(pieces), key=lambda q: -(q[2]-q[0])*(q[3]-q[1]))
        for j, q in enumerate(pieces):
            if (j & 15) == 0 and _clock() >= deadline:
                return []
            a, b, c, d = q
            if not any(u <= a and v <= b and s >= c and t >= d for u, v, s, t in keep):
                keep.append(q)
        if len(keep) > self.cap:
            keep.sort(key=lambda q: -(q[2]-q[0])*(q[3]-q[1]))
            del keep[self.cap:]
        return keep

    def ordering(self, fixed, rnd, p, history):
        used = {q[0] for q in fixed}
        ids = [i for i in self.density if i not in used]
        if not ids:
            return []
        area, rho, w, h = self.area, self.rho, self.w, self.h
        room = max(0.0, self.W*self.H - sum(area[i] for i in used))
        price = self._price(ids, room) * p['price']
        marginal = []
        for i in ids:
            gain = self.v[i] - price*area[i]
            score = (gain*math.exp(-p['power']*self.logarea[i]) *
                     (0.65 + max(w[i]/self.W, h[i]/self.H))) if gain >= 0 else gain/area[i]
            marginal.append((score*(1 + rnd.uniform(-p['eta'], p['eta'])), i))
        marginal.sort(reverse=True)
        rank = {i: k for k, (_, i) in enumerate(marginal)}
        poolorder = sorted(ids, key=lambda i: -rho[i]*(1 + rnd.uniform(-p['eta'], p['eta'])))
        pool, acc = [], 0.0
        for i in poolorder:
            pool.append(i)
            acc += area[i]
            if acc >= p['alpha']*room:
                break
        geo = p['geo']
        if geo == 0:
            key = lambda i: -area[i]
        elif geo == 1:
            key = lambda i: -max(w[i], h[i])
        elif geo == 2:
            key = lambda i: -max(w[i]/self.W, h[i]/self.H)
        elif geo == 3:
            key = lambda i: -(w[i]+h[i])
        else:
            key = lambda i: -min(w[i], h[i])
        geometry = {i: k for k, i in enumerate(sorted(pool, key=key))}
        density_rank = {i: k for k, i in enumerate(ids)}
        pool.sort(key=lambda i: 0.82*geometry[i] + 0.18*density_rank[i])
        chosen = set(pool)
        pool.extend(i for i in ids if i not in chosen)
        pooled_rank = {i: k for k, i in enumerate(pool)}
        memory_rank = {q[0]: k for k, q in enumerate(history)}
        memory = p['memory'] if history else 0.0
        amp, mix = p['noise']*len(ids), p['mix']
        ids.sort(key=lambda i: (1-memory)*((1-mix)*rank[i] + mix*pooled_rank[i]) +
                 memory*memory_rank.get(i, len(history)+rank[i]) + amp*rnd.random())
        return ids

    def position(self, free, w, h, edges, p, deadline):
        W, H = self.W, self.H
        area = W*H
        hosts = []
        big = sorted(free, key=lambda r: -(r[2]-r[0])*(r[3]-r[1]))[:self.big_count]
        for a, b, c, d in free:
            if a+w > c or b+h > d:
                continue
            dx, dy = c-a-w, d-b-h
            nx, ny = dx/W, dy/H
            score = min(nx, ny) + p['balance']*max(nx, ny)
            score += p['waste']*((c-a)*(d-b)-w*h)/area
            if 0 < dx < self.mw:
                score += dx*(d-b)/area
            if 0 < dy < self.mh:
                score += dy*(c-a)/area
            hosts.append((score, (a, b, c, d)))
        hosts.sort()
        best, bestscore, seen = None, _inf, set()
        for base, r in hosts[:self.hosts]:
            if _clock() >= deadline:
                break
            a, b, c, d = r
            for x in (a, _sub(c, w)):
                if x < a or x+w > c or x > W-w:
                    continue
                for y in (b, _sub(d, h)):
                    if y < b or y+h > d or y > H-h or (x, y) in seen:
                        continue
                    seen.add((x, y))
                    X, Y = x+w, y+h
                    loss = 0.0
                    for u, v, s, t in big:
                        if x >= s or X <= u or y >= t or Y <= v:
                            continue
                        rw, rh = s-u, t-v
                        remain = max(0.0, (x-u)*rh, (s-X)*rh, rw*(y-v), rw*(t-Y))
                        loss += rw*rh - remain
                    nx = x/W if not (p['corner'] & 1) else (W-X)/W
                    ny = y/H if not (p['corner'] & 2) else (H-Y)/H
                    compact = ny+0.03*nx if p['vertical'] else nx+ny
                    score = base + p['preserve']*loss/area + 0.04*compact
                    score -= p['contact']*_contact(edges, x, y, X, Y, W, H)/(w+h)
                    if score < bestscore:
                        bestscore, best = score, (x, y, X, Y)
        return best

    def repair(self, fixed, rnd, p, deadline, history=()):
        placed = list(fixed)
        free, edges = [(0.0, 0.0, self.W, self.H)], ({}, {}, {}, {})
        for q in fixed:
            if _clock() >= deadline:
                return placed
            free = self.split(free, q[1:], deadline)
            _edge_add(edges, q)
        if not free or _clock() >= deadline:
            return placed
        order = self.ordering(fixed, rnd, p, history)
        failed = set()
        for i in order:
            if not free or _clock() >= deadline:
                break
            shape = (self.w[i], self.h[i])
            if shape in failed:
                continue
            pos = self.position(free, shape[0], shape[1], edges, p, deadline)
            if pos is None:
                failed.add(shape)
            else:
                q = (i,) + pos
                placed.append(q)
                _edge_add(edges, q)
                free = self.split(free, pos, deadline)
        return placed

    def canonical(self, layout):
        # Coordinates of equal sizes stay the same; only the index changes to the highest-value available piece of that size.
        positions = {}
        for q in layout:
            positions.setdefault((self.w[q[0]], self.h[q[0]]), []).append(q)
        result = []
        for shape, rows in positions.items():
            top = self.groups[shape][:len(rows)]
            allowed = set(top)
            present = {q[0] for q in rows if q[0] in allowed}
            missing = iter(i for i in top if i not in present)
            for q in rows:
                result.append(q if q[0] in allowed else (next(missing),) + q[1:])
        return result

    def reassign(self, source, rnd, deadline):
        """Expanded slots stay mutually exclusive; every alternating-path swap commits a complete feasible matching."""
        if not source or len(source) == len(self.cand) or _clock() >= deadline:
            return list(source)
        slots = [list(q[1:]) for q in source]
        count = len(slots)
        assigned = [q[0] for q in source]
        owner = {i: j for j, i in enumerate(assigned)}
        order = list(range(count))
        rnd.shuffle(order)
        if rnd.random() < 0.5:
            order.sort(key=lambda j: self.rho[assigned[j]] * rnd.uniform(0.5, 1.5))
        # Expansion uses only part of this neighbourhood's time slice, leaving time for matching.
        now = _clock()
        grow_until = now + 0.24 * max(0.0, deadline-now)
        axes = (0, 1) if rnd.random() < 0.5 else (1, 0)
        for axis in axes:
            other = 1-axis
            end = self.W if axis == 0 else self.H
            for j in order:
                if _clock() >= grow_until:
                    break
                box = slots[j]
                lower, upper = 0.0, end
                complete = True
                for k, q in enumerate(slots):
                    if (k & 31) == 0 and _clock() >= grow_until:
                        complete = False
                        break
                    if j == k or box[other] >= q[other+2] or q[other] >= box[other+2]:
                        continue
                    if q[axis+2] <= box[axis]:
                        lower = max(lower, q[axis+2])
                    elif q[axis] >= box[axis+2]:
                        upper = min(upper, q[axis])
                # An expanded boundary cannot be committed before all obstacles are checked.
                if complete:
                    box[axis], box[axis+2] = lower, upper

        masks = {}
        minimum = min(self.v[i] for i in assigned)
        for target in self.rich:
            if _clock() >= deadline or self.v[target] <= minimum:
                break
            if target in owner:
                continue
            parents = [-1] * count
            queue, head, visited = [target], 0, 0
            victim, loss = -1, self.v[target]
            reached_minimum = False
            while head < len(queue) and _clock() < deadline and not reached_minimum:
                item = queue[head]
                head += 1
                mask = masks.get(item)
                if mask is None:
                    mask = 0
                    w, h = self.w[item], self.h[item]
                    complete = True
                    for j, (x, y, X, Y) in enumerate(slots):
                        if (j & 31) == 0 and _clock() >= deadline:
                            complete = False
                            break
                        if (x+w <= X and y+h <= Y and
                                x <= self.W-w and y <= self.H-h):
                            mask |= 1 << j
                    if not complete:
                        break
                    masks[item] = mask
                fresh = mask & ~visited
                visited |= fresh
                while fresh:
                    bit = fresh & -fresh
                    fresh ^= bit
                    slot = bit.bit_length()-1
                    parents[slot] = item
                    incumbent = assigned[slot]
                    value = self.v[incumbent]
                    if value < loss:
                        victim, loss = slot, value
                        if value == minimum:
                            reached_minimum = True
                            break
                    queue.append(incumbent)
            if victim < 0:
                continue
            # Backtrack from the lowest-value exchangeable item; path length is not limited to one-for-one swaps.
            del owner[assigned[victim]]
            slot = victim
            while slot >= 0:
                item = parents[slot]
                previous = owner.get(item, -1)
                assigned[slot] = item
                owner[item] = slot
                slot = previous
            minimum = min(self.v[i] for i in assigned)

        # Output only the pieces themselves, not the expanded slots, so later refilling can reuse the remaining gaps.
        result = []
        corner = rnd.randrange(4)
        for i, (x, y, X, Y) in zip(assigned, slots):
            w, h = self.w[i], self.h[i]
            if corner & 1:
                x = max(x, min(self.W-w, _sub(X, w)))
            if corner & 2:
                y = max(y, min(self.H-h, _sub(Y, h)))
            result.append((i, x, y, x+w, y+h))
        return result

    def compact(self, source, rnd, deadline):
        result = source[:]
        axes = (1, 2) if rnd.random() < 0.5 else (2, 1)
        for a in axes:
            b = 3-a
            high = rnd.random() < 0.5
            size = self.w if a == 1 else self.h
            end = self.W if a == 1 else self.H
            ordered = sorted(result, key=lambda q: q[a+2] if high else q[a], reverse=high)
            settled = []
            for q in ordered:
                if _clock() >= deadline:
                    return result
                if high:
                    edge = end
                    for r in settled:
                        if q[b] < r[b+2] and r[b] < q[b+2]:
                            edge = min(edge, r[a])
                    start = max(q[a], min(end-size[q[0]], _sub(edge, size[q[0]])))
                else:
                    edge = 0.0
                    for r in settled:
                        if q[b] < r[b+2] and r[b] < q[b+2]:
                            edge = max(edge, r[a+2])
                    start = min(q[a], edge)
                moved = list(q)
                moved[a], moved[a+2] = start, start+size[q[0]]
                settled.append(tuple(moved))
            result = settled
        return result

    def ruin(self, source, rnd, strength, mode):
        if not source:
            return []
        count = len(source)
        max_fraction = min(0.82, 0.13 + 0.10*strength)
        depth = min(count, int(1 + count*max_fraction*rnd.random()**1.7))
        q = rnd.choice(source)
        cx, cy = (q[1]+q[3])*0.5, (q[2]+q[4])*0.5
        if mode == 1:
            key = lambda r: (max(0.0, r[1]-cx, cx-r[3])/self.W)**2 + (max(0.0, r[2]-cy, cy-r[4])/self.H)**2
        elif mode == 2:
            axis = 1 if rnd.random() < 0.5 else 2
            mid = cx if axis == 1 else cy
            key = lambda r: abs((r[axis]+r[axis+2])*0.5-mid)
        elif rnd.random() < 0.75:
            key = lambda r: self.rho[r[0]]*rnd.uniform(0.5, 1.5)
        else:
            key = lambda r: rnd.random()
        removed = {q[0] for q in sorted(source, key=key)[:depth]}
        return [q for q in source if q[0] not in removed]

    def axis_masks(self, source, size, length, axis, rnd, limit, deadline, high):
        upper = min(length-size, _sub(length, size))
        coords = {0.0, upper}
        for q in source:
            coords.update((q[axis], q[axis+2], _sub(q[axis], size), _sub(q[axis+2], size)))
        coords = [x for x in coords if 0 <= x <= upper and x+size <= length]
        if len(coords) > limit:
            coords = [0.0, upper] + rnd.sample(coords, limit-2)
        rnd.shuffle(coords)
        masks = {}
        for x in coords:
            if _clock() >= deadline:
                break
            mask, end = 0, x+size
            for j, q in enumerate(source):
                if x < q[axis+2] and q[axis] < end:
                    mask |= 1 << j
            old = masks.get(mask)
            if old is None or (x > old if high else x < old):
                masks[mask] = x
        return list(masks.items())

    def eject(self, source, rnd, p, budget, deadline):
        used = {q[0] for q in source}
        missing = [i for i in self.cand if i not in used]
        if not missing:
            return source[:]
        power = rnd.random()
        missing.sort(key=lambda i: -(math.log(self.v[i])-power*self.logarea[i]))
        target = missing[int(len(missing)*rnd.random()**2)]
        w, h = self.w[target], self.h[target]
        limit = max(6, int(math.sqrt(180000*budget/(len(source)+1))))
        xs = self.axis_masks(source, w, self.W, 1, rnd, limit, deadline, p['corner'] & 1)
        ys = self.axis_masks(source, h, self.H, 2, rnd, limit, deadline, p['corner'] & 2)
        # Charge an opportunity cost for recoverable area, while candidates are still accepted by true total value.
        credit = rnd.uniform(0.0, 0.30)*self.price
        jitter = rnd.uniform(0.02, 0.15)
        costs = [max(0.0, self.v[q[0]]-credit*self.area[q[0]]) *
                 rnd.uniform(1-jitter, 1+jitter) for q in source]
        cache, best, score = {0: 0.0}, None, _inf
        bx, by = rnd.uniform(-0.01, 0.01), rnd.uniform(-0.01, 0.01)
        for xm, x in xs:
            if _clock() >= deadline:
                break
            for j, (ym, y) in enumerate(ys):
                if (j & 31) == 0 and _clock() >= deadline:
                    break
                mask = xm & ym
                loss = cache.get(mask)
                if loss is None:
                    loss, bits = 0.0, mask
                    while bits:
                        bit = bits & -bits
                        loss += costs[bit.bit_length()-1]
                        bits ^= bit
                    cache[mask] = loss
                value = loss + self.v[target]*(bx*x/self.W + by*y/self.H)
                if value < score:
                    score, best = value, (mask, x, y)
        if best is None:
            return self.ruin(source, rnd, math.log1p(budget), 1)
        mask, x, y = best
        kept = [q for j, q in enumerate(source) if not (mask >> j) & 1]
        return kept + [(target, x, y, x+w, y+h)]


def _valid(layout, e, deadline):
    seen = set()
    for a, (i, x, y, X, Y) in enumerate(layout):
        if _clock() >= deadline:
            return False
        if i in seen or not (0 <= x <= e.W-e.w[i] and 0 <= y <= e.H-e.h[i]):
            return False
        if X != x+e.w[i] or Y != y+e.h[i] or X > e.W or Y > e.H:
            return False
        seen.add(i)
        for _, u, v, U, V in layout[:a]:
            if x < U and u < X and y < V and v < Y:
                return False
    return True


def _search(e, tr, budget, deadline, saved):
    rnd = random.Random(734197 + len(e.w)*193 + int(e.W)*17 + int(e.H)*31)
    strength = math.log1p(budget)
    elite_cap = max(2, int(2 + strength*math.sqrt(len(e.cand)/(len(e.cand)+4))))
    p = dict(alpha=1.12, price=0.86, power=0.22, mix=0.14, eta=0.0, noise=0.0,
             balance=0.12, waste=0.10, contact=0.24, preserve=0.22, memory=0.0,
             geo=2, corner=0, vertical=False)
    bestp = p
    current, curval = saved[1][:], saved[0]
    archive = []
    spent, gain = [0.0]*7, [0.0]*7
    shares = (0.22, 0.16, 0.11, 0.11, 0.08, 0.20, 0.12)
    first = True
    started = _clock()
    span = max(1e-12, deadline-started)
    while _clock() < deadline:
        now = _clock()
        phase = min(1.0, (now-started)/span)
        op = 0 if first else min(range(7), key=lambda a: spent[a] / (
            shares[a]*(1+min(2.0, 30*gain[a]/max(1.0, saved[0])))))
        source, parent = current, bestp
        r = rnd.random()
        if r < 0.35:
            source = saved[1]
        elif r < 0.70 and archive:
            _, source, _, parent = archive[int(len(archive)*rnd.random()**2)]
        if not first:
            p = _parameters(rnd, strength, parent if rnd.random() < 0.65 else None)
        engine, flip = e, False
        if op == 0:
            kept = []
            if not first and rnd.random() < 0.5:
                engine, flip = tr, True
        elif op < 4:
            kept = e.ruin(source, rnd, strength, op)
            if rnd.random() < 0.35:
                kept = e.compact(kept, rnd, deadline)
        elif op == 4:
            kept = e.compact(source, rnd, deadline)
            if rnd.random() < 0.55:
                kept = e.ruin(kept, rnd, 0.5*strength, rnd.randrange(1, 3))
        elif op == 5:
            slice_s = min((deadline-now)*0.18, 0.014*math.sqrt(budget))
            kept = e.eject(source, rnd, p, budget, min(deadline, now+slice_s))
            if rnd.random() < 0.18:
                kept = e.compact(kept, rnd, deadline)
        else:
            slice_s = min((deadline-now)*0.22, budget*(0.013+0.007*strength))
            kept = e.reassign(source, rnd, min(deadline, now+slice_s))
        history = source if not first and rnd.random() < 0.60 else ()
        candidate = engine.repair(kept, rnd, p, deadline, history)
        if flip:
            candidate = [(i, y, x, Y, X) for i, x, y, X, Y in candidate]
        candidate = e.canonical(candidate)
        first = False
        val = math.fsum(e.v[q[0]] for q in candidate)
        spent[op] += max(1e-9, _clock()-now)
        remaining = max(0.0, 1-phase)
        temp = max(1e-12, saved[0]*(0.001+0.012*remaining*remaining) /
                   math.sqrt(max(1, len(candidate)))/(1+0.12*strength))
        accept = val >= curval or rnd.random() < math.exp(max(-700.0, (val-curval)/temp))
        promising = val > saved[0] or accept or len(archive) < elite_cap or (archive and val > archive[-1][0])
        if not promising or not _valid(candidate, e, deadline):
            continue
        if val > saved[0]:
            gain[op] += val-saved[0]
            saved[0], saved[1], bestp = val, candidate[:], p
        if accept:
            current, curval = candidate, val
        if candidate and (len(archive) < elite_cap or val >= archive[-1][0]):
            signature = frozenset(q[0] for q in candidate)
            found = next((j for j, row in enumerate(archive) if row[2] == signature), None)
            row = (val, candidate[:], signature, p)
            if found is None:
                archive.append(row)
            elif rnd.random() < 0.25:
                archive[found] = row
            archive.sort(key=lambda row: -row[0])
            del archive[elite_cap:]


def _emit(out, layout):
    sel, xx, yy = out['selection'], out['bottom_left_x'], out['bottom_left_y']
    left, below = out['left_relations'], out['below_relations']
    for i, x, y, _, _ in layout:
        sel[i], xx[i], yy[i] = 1, float(x), float(y)
    for a, (i, x, y, X, Y) in enumerate(layout):
        for j, u, v, U, V in layout[a+1:]:
            if X <= u:
                left[i][j] = 1
            elif U <= x:
                left[j][i] = 1
            if Y <= v:
                below[i][j] = 1
            elif V <= y:
                below[j][i] = 1
    return out


def solve(instance: dict, time_limit_s: float) -> dict:
    start = _clock()
    n, out = 0, None
    saved, fixed = [0.0, []], []
    try:
        n = max(0, int(instance['num_parts']))
        out = _empty(n)
        W, H, M = (float(instance[k]) for k in ('sheet_width', 'sheet_height', 'big_M'))
        w = [float(instance['part_widths'][i]) for i in range(n)]
        h = [float(instance['part_heights'][i]) for i in range(n)]
        v = [float(instance['part_values'][i]) for i in range(n)]
        budget = max(0.0, float(time_limit_s))
        if not all(math.isfinite(z) for z in [W, H, M, budget]+w+h+v):
            return out
        if min(W, H, M) < 0 or any(z < 0 for z in w+h):
            return out
        # This range bounds all inactive big-M linking constraints, not only geometric non-overlap.
        BW, BH = (min(W, M), min(H, M)) if n > 1 else (W, H)
        cand = []
        for i in range(n):
            if v[i] <= 0 or w[i] > BW or h[i] > BH:
                continue
            if w[i] == 0 or h[i] == 0:
                fixed.append((i, 0.0, 0.0, w[i], h[i]))
            elif math.isfinite(w[i]*h[i]) and w[i]*h[i] > 0:
                cand.append(i)
        if cand:
            i = max(cand, key=lambda j: v[j])
            saved = [v[i], [(i, 0.0, 0.0, w[i], h[i])]]
        reserve = min(0.30*budget, max(0.015*budget, 2*(_clock()-start)+n*n*1.5e-7))
        deadline = start + max(0.0, budget-reserve)
        if cand and _clock() < deadline:
            e = _Packing(w, h, v, BW, BH, cand, budget)
            tr = _Packing(h, w, v, BH, BW, cand, budget)
            _search(e, tr, budget, deadline, saved)
    except Exception:
        pass
    if out is None:
        out = _empty(n)
    try:
        return _emit(out, saved[1]+fixed)
    except Exception:
        return _empty(n)
