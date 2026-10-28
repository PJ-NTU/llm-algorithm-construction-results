"""Structure-preserving minimum-cut fusion packing
Ranks pieces jointly by density geometry and opportunity cost, repairs layouts with continuous space-loss and contact scoring, and keeps improving with adaptive removal, sweep-line exchange, two-axis compaction and minimum-cut recombination.
"""

import math
import random
import time
from bisect import bisect_left, bisect_right


def _blank(n):
    return {
        "selection": [0] * n,
        "bottom_left_x": [0.0] * n,
        "bottom_left_y": [0.0] * n,
        "left_relations": [[0] * n for _ in range(n)],
        "below_relations": [[0] * n for _ in range(n)],
    }


def _cut(spaces, box, mw, mh, deadline, cap):
    x, y, X, Y = box
    kept, fresh = [], []
    for a, b, c, d in spaces:
        if x >= c or X <= a or y >= d or Y <= b:
            kept.append((a, b, c, d))
            continue
        if x-a >= mw:
            fresh.append((a, b, x, d))
        if c-X >= mw:
            fresh.append((X, b, c, d))
        if y-b >= mh:
            fresh.append((a, b, c, y))
        if d-Y >= mh:
            fresh.append((a, Y, c, d))
    # Existing spaces do not contain one another, so only containment among new slices and between them and existing spaces is checked.
    fresh = list(dict.fromkeys(fresh))
    dead = [False]*len(fresh)
    for i, r in enumerate(fresh):
        if time.perf_counter() >= deadline:
            return []
        if dead[i]:
            continue
        a, b, c, d = r
        if c-a < mw or d-b < mh:
            dead[i] = True
            continue
        for j in range(i+1, len(fresh)):
            if dead[j]:
                continue
            u, v, U, V = fresh[j]
            if a <= u and b <= v and c >= U and d >= V:
                dead[j] = True
            elif u <= a and v <= b and U >= c and V >= d:
                dead[i] = True
                break
    for i, r in enumerate(fresh):
        if time.perf_counter() >= deadline:
            return []
        if not dead[i]:
            a, b, c, d = r
            if not any(u <= a and v <= b and U >= c and V >= d
                       for u, v, U, V in kept):
                kept.append(r)
    if len(kept) > cap:
        kept.sort(key=lambda r: (r[2]-r[0])*(r[3]-r[1]), reverse=True)
        del kept[cap:]
    return kept


def _edge_add(edges, p):
    _, x, y, X, Y = p
    for table, key, interval in ((edges[0], x, (y, Y)), (edges[1], X, (y, Y)),
                                 (edges[2], y, (x, X)), (edges[3], Y, (x, X))):
        table.setdefault(key, []).append(interval)


def _contact(edges, x, y, X, Y, W, H):
    c = (Y-y) * ((x == 0.0) + (X == W)) + (X-x) * ((y == 0.0) + (Y == H))
    for table, key, a, b in ((edges[1], x, y, Y), (edges[0], X, y, Y),
                             (edges[3], y, x, X), (edges[2], Y, x, X)):
        for u, v in table.get(key, ()):
            c += max(0.0, min(b, v)-max(a, u))
    return c


def _subtract(a, b):
    x = a-b
    if x+b > a:
        x = math.nextafter(x, -math.inf)
    return x


def _position(spaces, w, h, edges, ctx, params, host_count, deadline):
    W, H, _, _, _, mw, mh = ctx
    rule, lam, contact, balance, corner = params[5:10]
    inv_area = 1.0/(W*H)
    short, long = min(W, H), max(W, H)
    hosts, largest = [], []
    for k, r in enumerate(spaces):
        if k % 32 == 0 and time.perf_counter() >= deadline:
            return None
        a, b, c, d = r
        fw, fh = c-a, d-b
        area = fw*fh
        largest.append((area, r))
        largest.sort(reverse=True)
        if len(largest) > host_count:
            largest.pop()
        if a+w > c or b+h > d:
            continue
        dw, dh = fw-w, fh-h
        sm, lg = min(dw, dh), max(dw, dh)
        if rule == 0:
            fit = sm/short + balance*lg/long
        elif rule == 1:
            fit = lg/long + balance*sm/short
        elif rule == 2:
            fit = (area-w*h)*inv_area + balance*sm/short
        else:
            nx = a/W if not (corner & 1) else (W-c)/W
            ny = b/H if not (corner & 2) else (H-d)/H
            fit = ny+0.1*nx+balance*sm/short
        if 0 < dw < mw:
            fit += dw*fh*inv_area
        if 0 < dh < mh:
            fit += dh*fw*inv_area
        hosts.append((fit, r))
        hosts.sort()
        if len(hosts) > host_count:
            hosts.pop()
    best, score = None, math.inf
    seen = set()
    bigs = [r for _, r in largest]
    for fit, r in hosts:
        if time.perf_counter() >= deadline:
            break
        a, b, c, d = r
        check = bigs if r in bigs else bigs+[r]
        right, top = _subtract(c, w), _subtract(d, h)
        for x, y in ((a, b), (right, b), (a, top), (right, top)):
            if (x, y) in seen:
                continue
            seen.add((x, y))
            X, Y = x+w, y+h
            if x < a or y < b or X > c or Y > d or x > W-w or y > H-h:
                continue
            loss = 0.0
            for u, v, U, V in check:
                if X <= u or U <= x or Y <= v or V <= y:
                    continue
                fw, fh = U-u, V-v
                largest_piece = max(max(0.0, x-u)*fh, max(0.0, U-X)*fh,
                                    max(0.0, y-v)*fw, max(0.0, V-Y)*fw)
                loss += fw*fh-largest_piece
            nx = x/W if not (corner & 1) else (W-X)/W
            ny = y/H if not (corner & 2) else (H-Y)/H
            sc = fit + lam*loss*inv_area
            sc -= contact*_contact(edges, x, y, X, Y, W, H)/(w+h)
            sc += 0.025*(nx+ny)
            if sc < score:
                score, best = sc, (x, y, X, Y)
    return best


def _parameters(rnd, base, strength):
    # Parameters are inherited together with layouts; continuous perturbation strength grows with available time.
    spread = min(1.0, math.sqrt(strength/(1.0+strength)))
    if base is None or rnd.random() < 0.24:
        return (rnd.uniform(0.82, 1.18+0.12*spread),
                rnd.uniform(0.0, 0.28*spread), rnd.uniform(0.0, 0.40),
                rnd.uniform(0.0, 0.10*spread), rnd.randrange(5), rnd.randrange(4),
                math.exp(rnd.uniform(math.log(0.15), math.log(9.0))),
                rnd.uniform(0.06, 0.40), rnd.uniform(0.001, 0.24), rnd.randrange(4),
                rnd.uniform(0.08, 0.65), rnd.uniform(0.62, 1.12), rnd.uniform(0.0, 0.85))
    p = list(base)
    for k, width, lo, hi in ((0, 0.07, 0.76, 1.5), (1, 0.07, 0.0, 0.32),
                             (2, 0.10, 0.0, 0.5), (3, 0.035, 0.0, 0.12),
                             (7, 0.08, 0.03, 0.48), (8, 0.05, 0.001, 0.35),
                             (10, 0.12, 0.03, 0.8), (11, 0.09, 0.55, 1.2),
                             (12, 0.15, 0.0, 1.0)):
        p[k] = min(hi, max(lo, p[k]+rnd.uniform(-width, width)*spread))
    p[6] = min(12.0, max(0.08, p[6]*math.exp(rnd.uniform(-0.5, 0.5)*spread)))
    for k, choices in ((4, 5), (5, 4), (9, 4)):
        if rnd.random() < 0.20:
            p[k] = rnd.randrange(choices)
    return tuple(p)


def _repair(fixed, ctx, rnd, deadline, params, cap, host_count, fill_only=False):
    W, H, items, density_order, values, mw, mh = ctx
    placed = fixed[:]
    have = {p[0] for p in placed}
    spaces = [(0.0, 0.0, W, H)]
    edges = ({}, {}, {}, {})
    area_left = W*H
    for p in placed:
        if time.perf_counter() >= deadline:
            return placed
        i, x, y, X, Y = p
        area_left -= items[i][2]
        spaces = _cut(spaces, (x, y, X, Y), mw, mh, deadline, cap)
        _edge_add(edges, p)
        if not spaces:
            return placed
    maxw = max(r[2]-r[0] for r in spaces)
    maxh = max(r[3]-r[1] for r in spaces)
    remaining = [i for i in density_order if i not in have
                 and items[i][0] <= maxw and items[i][1] <= maxh]
    if not remaining or time.perf_counter() >= deadline:
        return placed
    alpha, jitter, beta, noise, geo = params[:5]
    shadow, price_factor, power = params[10:13]
    used, price = 0.0, 0.0
    for i in remaining:
        used += items[i][2]
        if used >= area_left:
            price = items[i][3]*price_factor
            break
    if fill_only:
        order = remaining
    else:
        ranked = sorted(remaining, key=lambda i: items[i][3]*(1.0+rnd.uniform(-jitter, jitter)),
                        reverse=True)
        pool, used = [], 0.0
        for i in ranked:
            pool.append(i)
            used += items[i][2]
            if used >= alpha*max(0.0, area_left):
                break
        def geometric(i):
            w, h, a = items[i][:3]
            if geo == 0:
                return a
            if geo == 1:
                return max(w, h)
            if geo == 2:
                return max(w/W, h/H)
            if geo == 3:
                return w+h
            return min(w, h)
        grank = {i: k for k, i in enumerate(sorted(pool, key=geometric, reverse=True))}
        drank = {i: k for k, i in enumerate(sorted(pool, key=lambda i: items[i][3], reverse=True))}
        def surplus(i):
            w, h, a, rho, loga = items[i]
            gain = values[i]-price*a
            return (gain*math.exp(-power*loga)*(0.65+max(w/W, h/H))
                    if gain >= 0.0 else gain/a)
        srank = {i: k for k, i in enumerate(sorted(pool, key=surplus, reverse=True))}
        size = max(1, len(pool))
        order = sorted(pool, key=lambda i:
                       (1.0-shadow)*((1.0-beta)*grank[i]+beta*drank[i])
                       + shadow*srank[i]+noise*size*rnd.random())
        pooled = set(pool)
        # A piece that failed cannot fit again while free space only shrinks; the tail is refilled once.
        order.extend(i for i in remaining if i not in pooled)
    for i in order:
        if time.perf_counter() >= deadline or not spaces:
            break
        w, h = items[i][:2]
        pos = _position(spaces, w, h, edges, ctx, params, host_count, deadline)
        if pos is not None:
            p = (i,)+pos
            placed.append(p)
            _edge_add(edges, p)
            spaces = _cut(spaces, pos, mw, mh, deadline, cap)
    return placed


def _compact(layout, ctx, rnd, deadline):
    W, H, items = ctx[:3]
    result = layout[:]
    axes = (0, 1) if rnd.random() < 0.5 else (1, 0)
    for axis in axes:
        toward_high = rnd.random() < 0.5
        a, b = (1, 2) if axis == 0 else (2, 1)
        end = W if axis == 0 else H
        ordered = sorted(result, key=lambda p: p[a+2] if toward_high else p[a], reverse=toward_high)
        settled = []
        for p in ordered:
            if time.perf_counter() >= deadline:
                return result
            size = items[p[0]][axis]
            if toward_high:
                edge = end
                for q in settled:
                    if p[b] < q[b+2] and q[b] < p[b+2]:
                        edge = min(edge, q[a])
                start = max(p[a], _subtract(edge, size))
            else:
                start = 0.0
                for q in settled:
                    if p[b] < q[b+2] and q[b] < p[b+2]:
                        start = max(start, q[a+2])
                start = min(start, p[a])
            pp = list(p)
            pp[a], pp[a+2] = start, start+size
            settled.append(tuple(pp))
        result = settled
    return result


def _destroy(layout, mode, fraction, ctx, rnd):
    if not layout:
        return []
    W, H, items, _, values = ctx[:5]
    count = max(1, min(len(layout), int(math.ceil(fraction*len(layout)))))
    anchor = rnd.choice(layout)
    ax, ay = (anchor[1]+anchor[3])*0.5, (anchor[2]+anchor[4])*0.5
    if mode == 0:
        def key(p):
            dx = max(0.0, p[1]-ax, ax-p[3])/W
            dy = max(0.0, p[2]-ay, ay-p[4])/H
            return dx*dx+dy*dy+0.0001*rnd.random()
    elif mode == 1:
        vertical = rnd.random() < 0.5
        def key(p):
            return abs((p[1]+p[3])*0.5-ax)/W if vertical else abs((p[2]+p[4])*0.5-ay)/H
    elif mode == 2:
        def key(p):
            return items[p[0]][3]*rnd.uniform(0.5, 1.5)
    else:
        def key(p):
            return rnd.random()
    ordered = sorted(layout, key=key)
    return ordered[count:]


def _eject(layout, ctx, rnd, until):
    """Sweep overlap events at the target origin, maintaining the lowest ejection value with range additions."""
    W, H, items, density_order, values = ctx[:5]
    have = {p[0] for p in layout}
    missing = [i for i in density_order if i not in have]
    if not missing or time.perf_counter() >= until:
        return layout[:]
    if rnd.random() < 0.5:
        missing.sort(key=lambda i: values[i], reverse=True)
    target = missing[int(len(missing)*rnd.random()**2)]
    w, h = items[target][:2]
    xmax, ymax = _subtract(W, w), _subtract(H, h)
    ys = {0.0, ymax}
    for p in layout:
        for y in (_subtract(p[2], h), p[4]):
            if 0.0 <= y <= ymax:
                ys.add(y)
    ys = sorted(ys)
    tops = [y+h for y in ys]
    size = len(ys)
    low = [0.0]*(4*size)
    lazy = [0.0]*(4*size)
    arg = [0]*(4*size)

    def build(node, lo, hi):
        arg[node] = lo
        if lo < hi:
            mid = (lo+hi)//2
            build(2*node, lo, mid)
            build(2*node+1, mid+1, hi)

    def add(node, lo, hi, left, right, amount):
        if left <= lo and hi <= right:
            low[node] += amount
            lazy[node] += amount
            return
        mid = (lo+hi)//2
        if left <= mid:
            add(2*node, lo, mid, left, right, amount)
        if right > mid:
            add(2*node+1, mid+1, hi, left, right, amount)
        child = 2*node if low[2*node] <= low[2*node+1] else 2*node+1
        low[node] = lazy[node]+low[child]
        arg[node] = arg[child]

    build(1, 0, size-1)
    events = {0.0: [[], []], xmax: [[], []]}
    for p in layout:
        if time.perf_counter() >= until:
            return layout[:]
        left = bisect_right(tops, p[2])
        right = bisect_left(ys, p[4])-1
        if left > right:
            continue
        amount = values[p[0]]
        begin, end = _subtract(p[1], w), p[3]
        change = (left, right, amount)
        if p[1] < w and end > 0.0:
            add(1, 0, size-1, left, right, amount)
        elif 0.0 <= begin <= xmax:
            events.setdefault(begin, [[], []])[1].append(change)
        if 0.0 < end <= xmax:
            events.setdefault(end, [[], []])[0].append(change)
    least, position = math.inf, None
    # Remove end events first, then evaluate contact points, then add start events; touching boundaries do not count as overlap.
    for x in sorted(events):
        if time.perf_counter() >= until:
            break
        ending, starting = events[x]
        for left, right, amount in ending:
            add(1, 0, size-1, left, right, -amount)
        if low[1] < least:
            y = ys[arg[1]]
            if x+w <= W and y+h <= H:
                least, position = low[1], (target, x, y, x+w, y+h)
                if least <= 0.0:
                    break
        for left, right, amount in starting:
            add(1, 0, size-1, left, right, amount)
    if position is None:
        return layout[:]
    _, x, y, X, Y = position
    # Recheck blockers with the actual floating-point coordinates rather than the rounded sweep-line accumulation.
    kept = [p for p in layout
            if not (x < p[3] and p[1] < X and y < p[4] and p[2] < Y)]
    return kept+[position]


def _recombine(first, second, values, until):
    """Two feasible layouts are internally conflict-free, so joint selection is equivalent to a minimum-weight vertex cover in a bipartite graph."""
    va = sum(values[p[0]] for p in first)
    vb = sum(values[p[0]] for p in second)
    fallback = first if va >= vb else second
    if time.perf_counter() >= until:
        return fallback
    common = set(first).intersection(second)
    aa = [p for p in first if p not in common]
    bb = [p for p in second if p not in common]
    if not aa or not bb:
        return fallback
    na, nb = len(aa), len(bb)
    source, sink = na+nb, na+nb+1
    count = sink+1
    adjacent = [[] for _ in range(count)]
    destination, capacity = [], []
    infinite = va+vb+max(1.0, (va+vb)*1e-9)
    if not math.isfinite(infinite):
        return fallback

    def arc(u, v, c):
        adjacent[u].append(len(destination))
        destination.append(v)
        capacity.append(c)
        adjacent[v].append(len(destination))
        destination.append(u)
        capacity.append(0.0)

    for i, p in enumerate(aa):
        arc(source, i, values[p[0]])
        if time.perf_counter() >= until:
            return fallback
        for j, q in enumerate(bb):
            if p[0] == q[0] or (p[1] < q[3] and q[1] < p[3]
                               and p[2] < q[4] and q[2] < p[4]):
                arc(i, na+j, infinite)
    for j, q in enumerate(bb):
        arc(na+j, sink, values[q[0]])
    # Non-recursive Dinic; when the wall clock runs out, fall back to the better parent layout and never output an unfinished cut.
    while time.perf_counter() < until:
        level = [-1]*count
        level[source] = 0
        queue = [source]
        for u in queue:
            if time.perf_counter() >= until:
                return fallback
            for e in adjacent[u]:
                v = destination[e]
                if capacity[e] > 0.0 and level[v] < 0:
                    level[v] = level[u]+1
                    queue.append(v)
        if level[sink] < 0:
            result = [p for p in first if p in common]
            result.extend(p for i, p in enumerate(aa) if level[i] >= 0)
            result.extend(q for j, q in enumerate(bb) if level[na+j] < 0)
            return result if sum(values[p[0]] for p in result) >= max(va, vb) else fallback
        cursor = [0]*count
        stack, path, limits = [source], [], [infinite]
        while stack:
            if time.perf_counter() >= until:
                return fallback
            u = stack[-1]
            if u == sink:
                amount = limits[-1]
                for e in path:
                    capacity[e] -= amount
                    capacity[e ^ 1] += amount
                stack, path, limits = [source], [], [infinite]
                continue
            edges = adjacent[u]
            while cursor[u] < len(edges):
                e = edges[cursor[u]]
                v = destination[e]
                if capacity[e] > 0.0 and level[v] == level[u]+1:
                    break
                cursor[u] += 1
            if cursor[u] == len(edges):
                level[u] = -1
                stack.pop()
                limits.pop()
                if path:
                    path.pop()
                    cursor[stack[-1]] += 1
            else:
                stack.append(v)
                path.append(e)
                limits.append(min(limits[-1], capacity[e]))
    return fallback


def _valid(layout, ws, hs, W, H, M, n, until=math.inf):
    seen = set()
    for i, x, y, X, Y in layout:
        if i in seen or not (0 <= x <= W-ws[i] and 0 <= y <= H-hs[i]):
            return False
        if not (X == x+ws[i] and Y == y+hs[i] and X <= W and Y <= H):
            return False
        seen.add(i)
    for a, p in enumerate(layout):
        if time.perf_counter() >= until:
            return False
        for q in layout[a+1:]:
            if p[1] < q[3] and q[1] < p[3] and p[2] < q[4] and q[2] < p[4]:
                return False
            if p[3] > q[1]+M or q[3] > p[1]+M or p[4] > q[2]+M or q[4] > p[2]+M:
                return False
    if len(layout) < n and n > 1:
        for _, x, y, X, Y in layout:
            if X > M or Y > M:
                return False
        for i in range(n):
            if i not in seen:
                if ws[i] > M or hs[i] > M:
                    return False
    return True


def _write(answer, layout, ws, hs):
    sel, xx, yy = answer["selection"], answer["bottom_left_x"], answer["bottom_left_y"]
    left, below = answer["left_relations"], answer["below_relations"]
    for i, x, y, _, _ in layout:
        sel[i], xx[i], yy[i] = 1, float(x), float(y)
    for a, p in enumerate(layout):
        i, x, y, X, Y = p
        for q in layout[a+1:]:
            j, u, v, U, V = q
            if X <= u:
                left[i][j] = 1
            elif U <= x:
                left[j][i] = 1
            if Y <= v:
                below[i][j] = 1
            elif V <= y:
                below[j][i] = 1
    return answer


def solve(instance: dict, time_limit_s: float) -> dict:
    start = time.perf_counter()
    n, answer = 0, None
    best, ws, hs = [], [], []
    try:
        n = max(0, int(instance["num_parts"]))
        answer = _blank(n)
        W, H, M = (float(instance[k]) for k in ("sheet_width", "sheet_height", "big_M"))
        ws = [float(v) for v in instance["part_widths"]]
        hs = [float(v) for v in instance["part_heights"]]
        values = [float(v) for v in instance["part_values"]]
        budget = max(0.0, float(time_limit_s))
        if not math.isfinite(budget):
            return answer
        if n == 0 or min(len(ws), len(hs), len(values)) < n or W <= 0 or H <= 0 or M < 0:
            return answer
        if not all(math.isfinite(v) for v in [W, H, M]+ws+hs+values):
            return answer
        # When some pieces are unselected, their zero coordinates also enter the inactive big-M inequalities.
        BW, BH = (min(W, M), min(H, M)) if n > 1 else (W, H)
        if BW <= 0.0 or BH <= 0.0:
            return answer
        items, zeros = {}, []
        for i in range(n):
            if 0 <= ws[i] <= BW and 0 <= hs[i] <= BH and values[i] > 0:
                area = ws[i]*hs[i]
                if ws[i] == 0.0 or hs[i] == 0.0:
                    zeros.append((i, 0.0, 0.0, ws[i], hs[i]))
                elif area > 0.0 and math.isfinite(area):
                    items[i] = (ws[i], hs[i], area, values[i]/area, math.log(area))
        if not items:
            if _valid(zeros, ws, hs, W, H, M, n):
                best = zeros
            return _write(answer, best, ws, hs)
        first = max(items, key=lambda i: values[i])
        best = [(first, 0.0, 0.0, ws[first], hs[first])]
        if not _valid(best+zeros, ws, hs, W, H, M, n):
            best = []
        best_value = sum(values[p[0]] for p in best)
        # The output matrix is pre-allocated; wall-clock time is reserved separately for coordinates, writing back relations and the final check.
        reserve = min(budget*0.18, max(budget*0.008, 0.003+n*n*0.00000032))
        deadline = start+max(0.0, budget-reserve)
        density_order = sorted(items, key=lambda i: (items[i][3], values[i]), reverse=True)
        ctx = (BW, BH, items, density_order, values,
               min(items[i][0] for i in items), min(items[i][1] for i in items))
        rnd = random.Random(734197+n*193+int(W)*17+int(H)*31)
        strength = math.log1p(budget)
        maximum_fraction = min(0.92, 0.15+0.12*strength)
        archive_limit = max(2, int(2+math.sqrt(strength*(1.0+math.log1p(len(items))))))
        host_count = max(2, int(2+math.sqrt(strength)))
        cap = max(24, int((len(items)+1)**0.65*(5.0+strength)))
        initial = (0.96, 0.0, 0.15, 0.0, 2, 0, 1.5, 0.16, 0.008, 0, 0.20, 0.86, 0.22)
        best_params = initial
        archive = [(best_value, best[:], initial)]
        weights = [1.0]*7
        current, current_value, current_params = best[:], best_value, initial
        first_pass = True
        last_gain = time.perf_counter()
        while time.perf_counter() < deadline:
            now = time.perf_counter()
            if first_pass:
                fixed, op, params = [], 6, initial
            else:
                r = rnd.random()
                if r < 0.22:
                    current_value, base, current_params = rnd.choice(archive)
                    current = base[:]
                elif r < 0.54:
                    current, current_value, current_params = best[:], best_value, best_params
                # Gradually raise the probability of a global rebuild during stagnation; the rebuild still uses the same fusion repairer.
                stagnation = min(1.0, (now-last_gain)/max(budget*0.22, 1e-12))
                if rnd.random() < 0.12+0.22*stagnation:
                    op = 6
                else:
                    pick, op = rnd.random()*sum(weights), 6
                    for j, weight in enumerate(weights):
                        pick -= weight
                        if pick <= 0:
                            op = j
                            break
                fraction = min(1.0, 1.0/max(1, len(current))
                               + maximum_fraction*rnd.random()**(1.7-0.7*stagnation))
                if op < 4:
                    fixed = _destroy(current, op, fraction, ctx, rnd)
                    if rnd.random() < 0.25:
                        fixed = _compact(fixed, ctx, rnd, deadline)
                elif op == 4:
                    allowance = min((deadline-now)*0.12,
                                    budget/(45.0*math.sqrt(1+len(current)))+budget*0.002)
                    fixed = _eject(current, ctx, rnd, min(deadline, now+allowance))
                elif op == 5:
                    fixed = _compact(current, ctx, rnd, deadline)
                    if rnd.random() < 0.5:
                        fixed = _destroy(fixed, rnd.randrange(2), fraction*0.6, ctx, rnd)
                else:
                    fixed = []
                base_params = best_params if rnd.random() < 0.5 else current_params
                params = _parameters(rnd, base_params, strength)
            repair_start = time.perf_counter()
            candidate = _repair(fixed, ctx, rnd, deadline, params, cap, host_count)
            first_pass = False
            if not _valid(candidate+zeros, ws, hs, W, H, M, n, deadline):
                weights[op] = 0.95*weights[op]+0.05
                continue
            val = sum(values[p[0]] for p in candidate)
            old_best = best_value
            # Save verified improvements before the expensive recombination, so this result is not lost near the deadline.
            if val > best_value:
                best, best_value, best_params = candidate[:], val, params
                last_gain = time.perf_counter()
            parent, parent_value = current, current_value
            if op == 6 and len(archive) > 1 and rnd.random() < 0.4:
                parent_value, parent, _ = rnd.choice(archive)
            elif rnd.random() < 0.35:
                parent, parent_value = best, best_value
            now = time.perf_counter()
            allowance = min(max(0.0, deadline-now)*0.18,
                            max(budget/(40.0*math.sqrt(len(items))), (now-repair_start)*0.5))
            combined = _recombine(parent, candidate, values, now+allowance)
            combined_value = sum(values[p[0]] for p in combined)
            if combined_value > max(val, parent_value):
                # Holes freed by the minimum cut are refilled with the same space-preserving score.
                topoff_until = min(deadline, time.perf_counter()+max(allowance, now-repair_start))
                combined = _repair(combined, ctx, rnd, topoff_until, params, cap, host_count, True)
                if _valid(combined+zeros, ws, hs, W, H, M, n, deadline):
                    candidate = combined
                    val = sum(values[p[0]] for p in candidate)
            reward = 1.0
            if val > best_value:
                best, best_value, best_params = candidate[:], val, params
                last_gain = time.perf_counter()
            if best_value > old_best:
                reward = 12.0
                archive.append((best_value, best[:], best_params))
                archive.sort(key=lambda s: s[0], reverse=True)
                del archive[archive_limit:]
            remaining = max(0.0, (deadline-time.perf_counter())/max(budget, 1e-12))
            temperature = max(1e-12, best_value*(0.001+0.014*remaining*remaining)
                              /math.sqrt(max(1, len(candidate))))
            if val >= current_value or rnd.random() < math.exp(max(-700.0, (val-current_value)/temperature)):
                current, current_value, current_params = candidate, val, params
                reward = max(reward, 3.0)
                if len(archive) < archive_limit and val > 0:
                    archive.append((val, candidate[:], params))
            weights[op] = 0.95*weights[op]+0.05*reward
        if zeros and _valid(best+zeros, ws, hs, W, H, M, n):
            best = best+zeros
    except Exception:
        pass
    if answer is None:
        answer = _blank(n)
    try:
        return _write(answer, best, ws, hs)
    except Exception:
        return _blank(n)
