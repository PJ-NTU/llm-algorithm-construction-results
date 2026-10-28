"""Bipartite minimum-cut fusion with adaptive packing
Combines residual-value and density-based selection, contact-edge free-rectangle repair and local removal with compaction, and fuses compatible positions of two feasible layouts with a minimum cut.
"""

import math
import random
import time
from collections import deque


_CLOCK = time.perf_counter
_INF = float('inf')


def _empty(n):
    return {
        'selection': [0] * n,
        'bottom_left_x': [0.0] * n,
        'bottom_left_y': [0.0] * n,
        'left_relations': [[0] * n for _ in range(n)],
        'below_relations': [[0] * n for _ in range(n)],
    }


def _sub(a, b):
    z = a - b
    if z + b > a:
        z = math.nextafter(z, -_INF)
    return z


def _overlap(p, q):
    return p[1] < q[3] and q[1] < p[3] and p[2] < q[4] and q[2] < p[4]


def _value(layout, c):
    return math.fsum(c['v'][p[0]] for p in layout)


def _cut(spaces, box, c, until):
    x, y, X, Y = box
    mw, mh = c['mw'], c['mh']
    keep, new = [], []
    for a, b, d, e in spaces:
        if _CLOCK() >= until:
            return []
        if x >= d or X <= a or y >= e or Y <= b:
            keep.append((a, b, d, e))
        else:
            if x - a >= mw:
                new.append((a, b, x, e))
            if d - X >= mw:
                new.append((X, b, d, e))
            if y - b >= mh:
                new.append((a, b, d, y))
            if e - Y >= mh:
                new.append((a, Y, d, e))
    # Maximal rectangles that did not intersect before cannot be contained in newly generated sub-rectangles.
    new = sorted(set(new), key=lambda r: (r[2]-r[0])*(r[3]-r[1]), reverse=True)
    for r in new:
        if _CLOCK() >= until:
            return []
        a, b, d, e = r
        if d-a >= mw and e-b >= mh and not any(
                u <= a and v <= b and s >= d and t >= e for u, v, s, t in keep):
            keep.append(r)
    if len(keep) > c['space_cap']:
        keep.sort(key=lambda r: (r[2]-r[0])*(r[3]-r[1]), reverse=True)
        del keep[c['space_cap']:]
    return keep


def _edge_add(edges, p):
    _, x, y, X, Y = p
    for table, key, interval in ((edges[0], x, (y, Y)), (edges[1], X, (y, Y)),
                                 (edges[2], y, (x, X)), (edges[3], Y, (x, X))):
        table.setdefault(key, []).append(interval)


def _contact(edges, x, y, X, Y, W, H):
    total = (Y-y)*((x == 0.0)+(X == W)) + (X-x)*((y == 0.0)+(Y == H))
    for table, key, a, b in ((edges[1], x, y, Y), (edges[0], X, y, Y),
                             (edges[3], y, x, X), (edges[2], Y, x, X)):
        for u, v in table.get(key, ()):
            total += max(0.0, min(b, v)-max(a, u))
    return total


def _position(spaces, w, h, edges, c, style, until):
    W, H = c['W'], c['H']
    balance, waste, contact, corner, bottom, preserve, rule = style
    hosts = []
    for r in spaces:
        if _CLOCK() >= until:
            return None
        a, b, d, e = r
        if a+w > d or b+h > e:
            continue
        dx, dy = (d-a-w)/W, (e-b-h)/H
        if rule == 0:
            fit = min(dx, dy) + balance*max(dx, dy)
        elif rule == 1:
            fit = max(dx, dy) + balance*min(dx, dy)
        elif rule == 2:
            fit = ((d-a)*(e-b)-w*h)/(W*H)
        else:
            fit = b/H + 0.03*a/W
        fit += waste*((d-a)*(e-b)-w*h)/(W*H)
        fit += ((d-a-w)*(e-b) if 0 < d-a-w < c['mw'] else 0.0)/(W*H)
        fit += ((e-b-h)*(d-a) if 0 < e-b-h < c['mh'] else 0.0)/(W*H)
        hosts.append((fit, r))
    if not hosts:
        return None
    bigs = []
    if preserve:
        hosts.sort()
        del hosts[c['breadth']:]
        bigs = sorted(spaces, key=lambda r: (r[2]-r[0])*(r[3]-r[1]), reverse=True)[:c['breadth']]
    best, best_score, seen = None, _INF, set()
    for fit, (a, b, d, e) in hosts:
        if _CLOCK() >= until:
            break
        rx, ry = _sub(d, w), _sub(e, h)
        for x, y in ((a, b), (rx, b), (a, ry), (rx, ry)):
            if (x, y) in seen or x < a or y < b or x > W-w or y > H-h:
                continue
            seen.add((x, y))
            X, Y = x+w, y+h
            if X > d or Y > e:
                continue
            nx = x/W if not corner & 1 else (W-X)/W
            ny = y/H if not corner & 2 else (H-Y)/H
            score = fit + 0.04*(ny+0.03*nx if bottom else nx+ny)
            score -= contact*_contact(edges, x, y, X, Y, W, H)/(w+h)
            if preserve:
                damage = 0.0
                for u, v, s, t in bigs:
                    if x >= s or X <= u or y >= t or Y <= v:
                        continue
                    retained = max(max(0.0, x-u)*(t-v), max(0.0, s-X)*(t-v),
                                   max(0.0, y-v)*(s-u), max(0.0, t-Y)*(s-u))
                    damage += (s-u)*(t-v)-retained
                score += preserve*damage/(W*H)
            if score < best_score:
                best_score, best = score, (x, y, X, Y)
    return best


def _order(c, fixed, rnd, policy, initial, until):
    have = {p[0] for p in fixed}
    rest = [i for i in c['density'] if i not in have]
    if not rest:
        return []
    a, v, rho = c['area'], c['v'], c['rho']
    room = max(0.0, c['W']*c['H']-math.fsum(a[i] for i in have))
    if policy == 0:
        used, price = 0.0, 0.0
        for i in rest:
            used += a[i]
            if used >= room:
                price = rho[i]
                break
        price *= 0.86 if initial else rnd.uniform(0.60, 1.15)
        power = 0.22 if initial else rnd.uniform(0.0, 0.95)
        jitter = 0.0 if initial else rnd.uniform(0.0, 0.32)
        ranked = []
        for i in rest:
            if _CLOCK() >= until:
                break
            gain = v[i]-price*a[i]
            hard = 0.65+max(c['w'][i]/c['W'], c['h'][i]/c['H'])
            key = gain*math.exp(-power*c['loga'][i])*hard if gain >= 0 else gain/a[i]
            ranked.append((key*math.exp(jitter*(rnd.random()-0.5)), v[i], i))
        ranked.sort(reverse=True)
        return [i for _, _, i in ranked]
    eta = rnd.uniform(0.0, 0.26)
    rich = sorted(rest, key=lambda i: rho[i]*(1.0+rnd.uniform(-eta, eta)), reverse=True)
    threshold = rnd.triangular(0.82, 1.36, 0.96)*room
    pool, used = [], 0.0
    for i in rich:
        pool.append(i)
        used += a[i]
        if used >= threshold:
            break
    mode = rnd.randrange(5)
    if mode == 0:
        key = lambda i: a[i]
    elif mode == 1:
        key = lambda i: max(c['w'][i], c['h'][i])
    elif mode == 2:
        key = lambda i: max(c['w'][i]/c['W'], c['h'][i]/c['H'])
    elif mode == 3:
        key = lambda i: c['w'][i]+c['h'][i]
    else:
        key = lambda i: min(c['w'][i], c['h'][i])
    gr = {i: j for j, i in enumerate(sorted(pool, key=key, reverse=True))}
    vr = {i: j for j, i in enumerate(sorted(pool, key=lambda i: rho[i], reverse=True))}
    beta, noise = rnd.uniform(0.0, 0.42), rnd.uniform(0.0, c['noise'])*len(pool)
    pool.sort(key=lambda i: (1-beta)*gr[i]+beta*vr[i]+rnd.random()*noise)
    chosen = set(pool)
    pool.extend(i for i in rest if i not in chosen)
    return pool


def _repair(fixed, c, rnd, until, initial=False, policy=0):
    placed = fixed[:]
    spaces = [(0.0, 0.0, c['W'], c['H'])]
    edges = ({}, {}, {}, {})
    for p in fixed:
        if _CLOCK() >= until:
            return placed
        spaces = _cut(spaces, p[1:], c, until)
        _edge_add(edges, p)
    if not spaces:
        return placed
    order = _order(c, fixed, rnd, policy, initial, until)
    if initial:
        style = (0.12, 0.10, 0.24, 0, False, 0.0, 0)
    else:
        style = (rnd.uniform(0.015, 0.32), rnd.uniform(0.0, 0.45),
                 rnd.uniform(0.06, 0.48), rnd.randrange(4), rnd.random() < 0.25,
                 math.exp(rnd.uniform(-1.6, 1.7)) if policy else 0.0,
                 rnd.randrange(4) if policy else 0)
    failed = set()
    for i in order:
        if _CLOCK() >= until or not spaces:
            break
        shape = c['w'][i], c['h'][i]
        if shape in failed:
            continue
        pos = _position(spaces, shape[0], shape[1], edges, c, style, until)
        if pos is None:
            failed.add(shape)
            continue
        p = (i,)+pos
        placed.append(p)
        _edge_add(edges, p)
        spaces = _cut(spaces, pos, c, until)
    return placed


def _canonical(layout, c):
    by_shape = {}
    for p in layout:
        by_shape.setdefault((c['w'][p[0]], c['h'][p[0]]), []).append(p)
    out = []
    for shape, positions in by_shape.items():
        top = c['groups'][shape][:len(positions)]
        allowed = set(top)
        already = {p[0] for p in positions if p[0] in allowed}
        missing = iter(i for i in top if i not in already)
        for p in positions:
            out.append(p if p[0] in allowed else (next(missing),)+p[1:])
    return out


def _compact(layout, c, rnd, until):
    result = layout[:]
    for axis in ((0, 1) if rnd.random() < 0.5 else (1, 0)):
        high = rnd.random() < 0.5
        a, b = (1, 2) if axis == 0 else (2, 1)
        length = c['W'] if axis == 0 else c['H']
        sizes = c['w'] if axis == 0 else c['h']
        ordered = sorted(result, key=lambda p: p[a+2] if high else p[a], reverse=high)
        settled = []
        for p in ordered:
            if _CLOCK() >= until:
                return result
            size = sizes[p[0]]
            edge = length if high else 0.0
            for q in settled:
                if p[b] < q[b+2] and q[b] < p[b+2]:
                    edge = min(edge, q[a]) if high else max(edge, q[a+2])
            x = max(p[a], min(length-size, _sub(edge, size))) if high else min(p[a], edge)
            pp = list(p)
            pp[a], pp[a+2] = x, x+size
            settled.append(tuple(pp))
        result = settled
    return result


def _destroy(layout, mode, fraction, c, rnd):
    if not layout:
        return []
    count = max(1, min(len(layout), int(math.ceil(fraction*len(layout)))))
    anchor = rnd.choice(layout)
    ax, ay = (anchor[1]+anchor[3])*0.5, (anchor[2]+anchor[4])*0.5
    if mode == 0:
        aspect = math.exp(rnd.uniform(-1.0, 1.0))
        def key(p):
            dx = max(0.0, p[1]-ax, ax-p[3])/c['W']
            dy = max(0.0, p[2]-ay, ay-p[4])/c['H']
            return aspect*dx*dx+dy*dy/aspect+0.0001*rnd.random()
    elif mode == 1:
        vertical = rnd.random() < 0.5
        def key(p):
            return abs((p[1]+p[3])*0.5-ax)/c['W'] if vertical else abs((p[2]+p[4])*0.5-ay)/c['H']
    elif mode == 2:
        def key(p):
            return c['rho'][p[0]]*rnd.uniform(0.5, 1.5)
    else:
        def key(p):
            return rnd.random()
    return sorted(layout, key=key)[count:]


def _eject(layout, c, rnd, until):
    have = {p[0] for p in layout}
    missing = [i for i in c['density'] if i not in have]
    if not missing:
        return layout[:]
    if rnd.random() < 0.5:
        power = rnd.random()
        missing.sort(key=lambda i: math.log(c['v'][i])-power*c['loga'][i], reverse=True)
    target = missing[int(len(missing)*rnd.random()**2)]
    w, h, W, H = c['w'][target], c['h'][target], c['W'], c['H']
    xmax, ymax = _sub(W, w), _sub(H, h)
    anchors = [(0.0, 0.0), (xmax, 0.0), (0.0, ymax), (xmax, ymax)]
    xs, ys = {0.0, xmax}, {0.0, ymax}
    shuffled = layout[:]
    rnd.shuffle(shuffled)
    for _, x, y, X, Y in shuffled:
        if _CLOCK() >= until:
            return layout[:]
        for u in (x, X, _sub(x, w), _sub(X, w)):
            if 0 <= u <= xmax:
                xs.add(u)
        for v in (y, Y, _sub(y, h), _sub(Y, h)):
            if 0 <= v <= ymax:
                ys.add(v)
        anchors.extend(((min(x, xmax), min(y, ymax)),
                        (max(0.0, min(_sub(X, w), xmax)), max(0.0, min(_sub(Y, h), ymax)))))
    xs, ys = tuple(xs), tuple(ys)
    seen, least, position, removed = set(), _INF, None, set()
    cursor = 0
    while _CLOCK() < until:
        if cursor < len(anchors):
            x, y = anchors[cursor]
            cursor += 1
        else:
            if len(seen) >= len(xs)*len(ys):
                break
            x, y = rnd.choice(xs), rnd.choice(ys)
        if (x, y) in seen:
            continue
        seen.add((x, y))
        X, Y = x+w, y+h
        if x < 0 or y < 0 or X > W or Y > H or x > W-w or y > H-h:
            continue
        loss, blockers = 0.0, set()
        for p in layout:
            if x < p[3] and p[1] < X and y < p[4] and p[2] < Y:
                loss += c['v'][p[0]]
                blockers.add(p[0])
                if loss >= least:
                    break
        if loss < least:
            least, position, removed = loss, (target, x, y, X, Y), blockers
            if loss == 0.0:
                break
    if position is None:
        return layout[:]
    return [p for p in layout if p[0] not in removed]+[position]


def _fuse(first, second, c, until):
    """Maximum-weight independent set over the union of finite candidate positions; the original layout is kept if the flow computation does not finish."""
    fallback = first if _value(first, c) >= _value(second, c) else second
    if _CLOCK() >= until:
        return fallback
    lookup = {p[0]: p for p in first}
    common = [p for p in second if lookup.get(p[0]) == p]
    same = {p[0] for p in common}
    aa = [p for p in first if p[0] not in same]
    bb = [p for p in second if p[0] not in same]
    if not aa or not bb:
        return common+aa+bb
    na, nb = len(aa), len(bb)
    source, sink = na+nb, na+nb+1
    graph = [[] for _ in range(sink+1)]

    def edge(u, v, cap):
        graph[u].append([v, cap, len(graph[v])])
        graph[v].append([u, 0, len(graph[u])-1])

    weights = c['iv']
    infinity = sum(weights[p[0]] for p in aa+bb)+1
    for i, p in enumerate(aa):
        edge(source, i, weights[p[0]])
    for j, q in enumerate(bb):
        edge(na+j, sink, weights[q[0]])
    for i, p in enumerate(aa):
        if _CLOCK() >= until:
            return fallback
        for j, q in enumerate(bb):
            if p[0] == q[0] or _overlap(p, q):
                edge(i, na+j, infinity)
    # Integer capacities prevent floating-point residuals from wrongly treating conflict edges as cut.
    # An explicit path stack replaces recursion, so layout size is not limited by Python's recursion depth.
    while _CLOCK() < until:
        level = [-1]*len(graph)
        level[source] = 0
        queue = deque([source])
        while queue:
            if _CLOCK() >= until:
                return fallback
            u = queue.popleft()
            for v, cap, _ in graph[u]:
                if cap > 0 and level[v] < 0:
                    level[v] = level[u]+1
                    queue.append(v)
        if level[sink] < 0:
            result = common+[p for i, p in enumerate(aa) if level[i] >= 0]
            result.extend(q for j, q in enumerate(bb) if level[na+j] < 0)
            return result if _value(result, c) >= _value(fallback, c) else fallback
        cursor = [0]*len(graph)
        stack, path = [source], []
        while stack:
            if _CLOCK() >= until:
                return fallback
            u = stack[-1]
            if u == sink:
                amount = min(graph[v][k][1] for v, k in path)
                first_zero = len(path)
                for index, (v, k) in enumerate(path):
                    e = graph[v][k]
                    e[1] -= amount
                    graph[e[0]][e[2]][1] += amount
                    if not e[1] and index < first_zero:
                        first_zero = index
                del path[first_zero:]
                del stack[first_zero+1:]
                continue
            row = graph[u]
            while cursor[u] < len(row):
                v, cap, _ = row[cursor[u]]
                if cap > 0 and level[v] == level[u]+1:
                    break
                cursor[u] += 1
            if cursor[u] == len(row):
                level[u] = -1
                stack.pop()
                if path:
                    path.pop()
            else:
                path.append((u, cursor[u]))
                stack.append(row[cursor[u]][0])
    return fallback


def _valid(layout, c, until):
    seen = set()
    for p in layout:
        i, x, y, X, Y = p
        if i in seen or not (0 <= x <= c['W']-c['w'][i] and 0 <= y <= c['H']-c['h'][i]):
            return False
        if X != x+c['w'][i] or Y != y+c['h'][i] or X > c['W'] or Y > c['H']:
            return False
        seen.add(i)
    for a, p in enumerate(layout):
        if _CLOCK() >= until:
            return False
        for b in range(a+1, len(layout)):
            if _overlap(p, layout[b]):
                return False
    return True


def _write(out, layout):
    selected, xx, yy = out['selection'], out['bottom_left_x'], out['bottom_left_y']
    left, below = out['left_relations'], out['below_relations']
    for i, x, y, _, _ in layout:
        selected[i], xx[i], yy[i] = 1, float(x), float(y)
    for a, p in enumerate(layout):
        i, x, y, X, Y = p
        for b in range(a+1, len(layout)):
            j, u, v, U, V = layout[b]
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
    started = _CLOCK()
    n, out, best, degenerate = 0, None, [], []
    try:
        n = max(0, int(instance['num_parts']))
        out = _empty(n)
        allocated = _CLOCK()-started
        w = [float(x) for x in instance['part_widths']]
        h = [float(x) for x in instance['part_heights']]
        v = [float(x) for x in instance['part_values']]
        W, H, M = (float(instance[k]) for k in ('sheet_width', 'sheet_height', 'big_M'))
        budget = max(0.0, float(time_limit_s))
        if n == 0 or min(len(w), len(h), len(v)) < n:
            return out
        if not all(math.isfinite(z) for z in w+h+v+[W, H, M, budget]):
            return out
        if W < 0 or H < 0 or M < 0 or any(z < 0 for z in w+h):
            return out
        # All inactive linking constraints are also bounded by M; spans anchored at the origin explicitly do not exceed M.
        BW, BH = (min(W, M), min(H, M)) if n > 1 else (W, H)
        candidates, groups = [], {}
        area, rho, loga = [0.0]*n, [0.0]*n, [0.0]*n
        for i in range(n):
            if v[i] <= 0 or w[i] > BW or h[i] > BH:
                continue
            if w[i] == 0 or h[i] == 0:
                degenerate.append((i, 0.0, 0.0, w[i], h[i]))
            else:
                a = w[i]*h[i]
                if not (a > 0 and math.isfinite(a)):
                    continue
                candidates.append(i)
                area[i], rho[i], loga[i] = a, v[i]/a, math.log(a)
                groups.setdefault((w[i], h[i]), []).append(i)
        if not candidates:
            return _write(out, degenerate)
        for group in groups.values():
            group.sort(key=lambda i: (-v[i], i))
        single = max(candidates, key=lambda i: v[i])
        best = [(single, 0.0, 0.0, w[single], h[single])]
        best_value = v[single]
        strength = math.log1p(budget)
        m = len(candidates)
        reserve = min(budget*0.30, max(0.015*budget, 2.0*allocated, 0.002+n*n*1.8e-7))
        deadline = started+max(0.0, budget-reserve)
        density = sorted(candidates, key=lambda i: (rho[i], v[i]), reverse=True)
        # Floating-point values are binary rationals; after a common denominator, value comparisons of the original objective are kept exact.
        ratios = {i: v[i].as_integer_ratio() for i in candidates}
        exponent = max(q.bit_length()-1 for _, q in ratios.values())
        iv = {i: p << (exponent-(q.bit_length()-1)) for i, (p, q) in ratios.items()}
        c = {'W': BW, 'H': BH, 'w': w, 'h': h, 'v': v, 'area': area, 'rho': rho,
             'loga': loga, 'density': density, 'groups': groups, 'iv': iv,
             'mw': min(w[i] for i in candidates), 'mh': min(h[i] for i in candidates),
             'space_cap': max(32, int((6.0+3.0*strength)*math.sqrt(m))),
             'breadth': max(2, int(2.0+math.sqrt(strength))),
             'noise': min(0.18, 0.025+0.026*strength)}
        rnd = random.Random(734197+n*193+int(W)*17+int(H)*31)
        max_fraction = min(0.90, 0.18+0.105*strength)
        archive_cap = max(2, int(2+math.sqrt(strength*math.log1p(m))))
        archive = [(best_value, best[:], frozenset([single]))]
        current, current_value = best[:], best_value
        weights = [1.0]*7
        first = True
        last_gain = started
        # No cap on restarts or iterations; all improvement operations share the wall-clock deadline.
        while _CLOCK() < deadline:
            now = _CLOCK()
            if first:
                source, source_value, fixed, op = best, best_value, [], 6
            else:
                r = rnd.random()
                if r < 0.52:
                    source, source_value = best, best_value
                elif r < 0.76:
                    source, source_value = current, current_value
                else:
                    source_value, source, _ = rnd.choice(archive)
                draw, op = rnd.random()*sum(weights), 6
                for j, weight in enumerate(weights):
                    draw -= weight
                    if draw <= 0:
                        op = j
                        break
                stagnant = min(1.0, (now-last_gain)/max(budget*0.2, 1e-12))
                fraction = min(1.0, 1.0/max(1, len(source))+
                               max_fraction*(0.7+0.6*stagnant)*rnd.random()**1.7)
                if op < 4:
                    fixed = _destroy(source, op, fraction, c, rnd)
                    if rnd.random() < 0.38:
                        fixed = _compact(fixed, c, rnd, deadline)
                elif op == 4:
                    allowance = min((deadline-now)*0.12,
                                    (1.0+math.sqrt(budget))/(250.0*math.sqrt(max(1, len(source)))))
                    fixed = _eject(source, c, rnd, min(deadline, now+allowance))
                elif op == 5:
                    fixed = _compact(source, c, rnd, deadline)
                    if rnd.random() < 0.5:
                        fixed = _destroy(fixed, rnd.randrange(2), fraction*0.6, c, rnd)
                else:
                    fixed = []
            policy = 0 if first or rnd.random() < 0.73 else 1
            raw = _canonical(_repair(fixed, c, rnd, deadline, first, policy), c)
            first = False
            if not _valid(raw, c, deadline):
                weights[op] = 0.94*weights[op]+0.06
                continue
            raw_value = _value(raw, c)
            candidate, val = raw, raw_value
            # Each fusion combines only two internally feasible layouts; duplicate indices and overlaps both become conflict edges.
            now = _CLOCK()
            allowance = min((deadline-now)*0.35,
                            budget*(0.6+0.20*strength)/(25.0+math.sqrt(len(source)+len(raw))))
            if allowance > 0:
                candidate = _fuse(source, raw, c, min(deadline, now+allowance))
                candidate = _canonical(candidate, c)
                val = _value(candidate, c)
            if source is not best and val > source_value and _CLOCK() < deadline:
                now = _CLOCK()
                candidate = _fuse(best, candidate, c,
                                  min(deadline, now+max(0.0, allowance)))
                candidate = _canonical(candidate, c)
                val = _value(candidate, c)
            # Fusion changes the shape of holes; refilling happens only when a genuinely new layout is formed.
            if val > max(source_value, raw_value) and _CLOCK() < deadline:
                candidate = _canonical(_repair(candidate, c, rnd, deadline, policy=0), c)
                val = _value(candidate, c)
            reward = 1.0
            if val > best_value:
                # Construction, compaction and fusion all preserve feasibility; the best solution is checked geometrically once more on its own.
                if _valid(candidate, c, deadline):
                    best, best_value = candidate[:], val
                    last_gain = _CLOCK()
                    reward = 12.0
            remaining = max(0.0, (deadline-_CLOCK())/max(budget, 1e-12))
            temperature = max(1e-12, best_value*(0.001+0.012*remaining*remaining)
                              /math.sqrt(max(1, len(raw))))
            # A non-fused repair can still serve as a non-monotone exploration start and does not overwrite the global best.
            if rnd.random() < 0.28:
                walk, walk_value = raw, raw_value
            else:
                walk, walk_value = candidate, val
            if walk_value >= current_value or rnd.random() < math.exp(
                    max(-700.0, (walk_value-current_value)/temperature)):
                current, current_value = walk, walk_value
                reward = max(reward, 3.0)
            if val >= best_value*0.96:
                signature = frozenset(p[0] for p in candidate)
                match = next((j for j, entry in enumerate(archive) if entry[2] == signature), None)
                if match is None:
                    archive.append((val, candidate[:], signature))
                elif val > archive[match][0] or rnd.random() < 0.25:
                    archive[match] = (val, candidate[:], signature)
                archive.sort(key=lambda entry: entry[0], reverse=True)
                del archive[archive_cap:]
            weights[op] = 0.94*weights[op]+0.06*reward
    except Exception:
        # A search error can never overwrite a previously verified layout.
        pass
    if out is None:
        out = _empty(n)
    try:
        return _write(out, degenerate+best)
    except Exception:
        return _empty(n)
