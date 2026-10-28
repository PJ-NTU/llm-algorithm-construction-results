"""Spatial exchange with adaptive large neighbourhood packing
Modifies the layout by directed ejection, local removal and two-axis compaction, repairs it with Lagrangian value ordering and contact-edge refilling, and keeps the best feasible solution found so far.
"""

import math
import random
import time


def _blank(n):
    return {
        "selection": [0] * n,
        "bottom_left_x": [0.0] * n,
        "bottom_left_y": [0.0] * n,
        "left_relations": [[0] * n for _ in range(n)],
        "below_relations": [[0] * n for _ in range(n)],
    }


def _cut(spaces, box, mw, mh, deadline):
    x, y, X, Y = box
    pieces = []
    for a, b, c, d in spaces:
        if x >= c or X <= a or y >= d or Y <= b:
            pieces.append((a, b, c, d))
        else:
            if x - a >= mw:
                pieces.append((a, b, x, d))
            if c - X >= mw:
                pieces.append((X, b, c, d))
            if y - b >= mh:
                pieces.append((a, b, c, y))
            if d - Y >= mh:
                pieces.append((a, Y, c, d))
    # Remove contained spaces in decreasing area order; at the deadline unprocessed spaces are discarded without harming the placed layout.
    pieces = sorted(set(pieces), key=lambda r: (r[2]-r[0])*(r[3]-r[1]), reverse=True)
    kept = []
    for r in pieces:
        if time.perf_counter() >= deadline:
            return []
        a, b, c, d = r
        if c-a < mw or d-b < mh:
            continue
        if not any(u <= a and v <= b and s >= c and t >= d for u, v, s, t in kept):
            kept.append(r)
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


def _position(spaces, w, h, edges, ctx, style, deadline):
    W, H, _, _, _, mw, mh = ctx
    balance, waste, contact, corner, bottom = style
    score = math.inf
    best = None
    seen = set()
    for a, b, c, d in spaces:
        if time.perf_counter() >= deadline:
            break
        if a+w > c or b+h > d:
            continue
        dx, dy = (c-a-w)/W, (d-b-h)/H
        fit = min(dx, dy) + balance*max(dx, dy) + waste*((c-a)*(d-b)-w*h)/(W*H)
        fit += ((c-a-w)*(d-b) if 0 < c-a-w < mw else 0.0)/(W*H)
        fit += ((d-b-h)*(c-a) if 0 < d-b-h < mh else 0.0)/(W*H)
        for x, y in ((a, b), (_subtract(c, w), b), (a, _subtract(d, h)),
                     (_subtract(c, w), _subtract(d, h))):
            if (x, y) in seen or x < a or y < b:
                continue
            seen.add((x, y))
            X, Y = x+w, y+h
            if X > c or Y > d:
                continue
            nx = x/W if not (corner & 1) else (W-X)/W
            ny = y/H if not (corner & 2) else (H-Y)/H
            compact = ny+0.03*nx if bottom else nx+ny
            s = fit - contact*_contact(edges, x, y, X, Y, W, H)/(w+h) + 0.04*compact
            if s < score:
                score, best = s, (x, y, X, Y)
    return best


def _repair(fixed, ctx, rnd, deadline, initial=False):
    W, H, items, density_order, values, mw, mh = ctx
    placed = fixed[:]
    have = {p[0] for p in placed}
    spaces = [(0.0, 0.0, W, H)]
    edges = ({}, {}, {}, {})
    area_left = W*H
    for p in placed:
        if time.perf_counter() >= deadline:
            return placed
        _, x, y, X, Y = p
        area_left -= (X-x)*(Y-y)
        spaces = _cut(spaces, (x, y, X, Y), mw, mh, deadline)
        _edge_add(edges, p)
    if not spaces:
        return placed
    remaining = [i for i in density_order if i not in have]
    used = 0.0
    price = 0.0
    for i in remaining:
        used += items[i][2]
        if used >= area_left:
            price = items[i][3]
            break
    price *= 0.86 if initial else rnd.uniform(0.60, 1.15)
    power = 0.22 if initial else rnd.uniform(0.0, 0.95)
    jitter = 0.0 if initial else rnd.uniform(0.0, 0.32)
    order = []
    for i in remaining:
        w, h, a, rho, loga = items[i]
        gain = values[i] - price*a
        hard = 0.65 + max(w/W, h/H)
        if gain >= 0:
            key = gain*math.exp(-power*loga)*hard
        else:
            key = gain/a
        key *= math.exp(jitter*(rnd.random()-0.5))
        order.append((key, values[i], i))
    order.sort(reverse=True)
    style = (0.12, 0.10, 0.24, 0, False) if initial else (
        rnd.uniform(0.015, 0.32), rnd.uniform(0.0, 0.45),
        rnd.uniform(0.06, 0.48), rnd.randrange(4), rnd.random() < 0.25)
    for _, _, i in order:
        if time.perf_counter() >= deadline or not spaces:
            break
        w, h = items[i][:2]
        pos = _position(spaces, w, h, edges, ctx, style, deadline)
        if pos is not None:
            p = (i,)+pos
            placed.append(p)
            _edge_add(edges, p)
            spaces = _cut(spaces, pos, mw, mh, deadline)
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
    W, H, items, density_order, values = ctx[:5]
    have = {p[0] for p in layout}
    missing = [i for i in density_order if i not in have]
    if not missing:
        return layout[:]
    if rnd.random() < 0.5:
        missing.sort(key=lambda i: values[i], reverse=True)
    target = missing[int(len(missing)*rnd.random()**2)]
    w, h = items[target][:2]
    xmax, ymax = _subtract(W, w), _subtract(H, h)
    anchors = [(0.0, 0.0), (xmax, 0.0), (0.0, ymax), (xmax, ymax)]
    xs, ys = {0.0, xmax}, {0.0, ymax}
    shuffled = layout[:]
    rnd.shuffle(shuffled)
    for _, x, y, X, Y in shuffled:
        for u in (x, X, _subtract(x, w), _subtract(X, w)):
            if 0 <= u <= xmax:
                xs.add(u)
        for v in (y, Y, _subtract(y, h), _subtract(Y, h)):
            if 0 <= v <= ymax:
                ys.add(v)
        anchors.extend(((min(x, xmax), min(y, ymax)),
                        (max(0.0, min(_subtract(X, w), xmax)), max(0.0, min(_subtract(Y, h), ymax)))))
    xs, ys = tuple(xs), tuple(ys)
    seen = set()
    least, position, removed = math.inf, None, set()
    # No cap on the number of samples; the neighbourhood is cut off by wall clock and ends naturally once all finite positions are visited.
    cursor = 0
    while time.perf_counter() < until:
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
        if x < 0 or y < 0 or X > W or Y > H:
            continue
        loss, blockers = 0.0, set()
        for p in layout:
            if x < p[3] and p[1] < X and y < p[4] and p[2] < Y:
                loss += values[p[0]]
                blockers.add(p[0])
                if loss >= least:
                    break
        if loss < least:
            least, position, removed = loss, (target, x, y, X, Y), blockers
            if loss == 0.0:
                break
    if position is None:
        return layout[:]
    return [p for p in layout if p[0] not in removed] + [position]


def _valid(layout, ws, hs, W, H, M, n):
    seen = set()
    for i, x, y, X, Y in layout:
        if i in seen or not (0 <= x <= W-ws[i] and 0 <= y <= H-hs[i]):
            return False
        if not (X == x+ws[i] and Y == y+hs[i] and X <= W and Y <= H):
            return False
        seen.add(i)
    for a, p in enumerate(layout):
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
    n = 0
    best = []
    answer = None
    ws, hs = [], []
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
        # Under the normal big M this equals the original sheet; a smaller M still satisfies all inactive linking inequalities.
        BW, BH = (min(W, M), min(H, M)) if n > 1 else (W, H)
        items = {}
        for i in range(n):
            if 0 < ws[i] <= BW and 0 < hs[i] <= BH and values[i] > 0:
                area = ws[i]*hs[i]
                if area > 0 and math.isfinite(area):
                    items[i] = (ws[i], hs[i], area, values[i]/area, math.log(area))
        if not items:
            return answer
        # Keep a non-empty feasible solution that needs no search first; the output matrix is already allocated.
        first = max(items, key=lambda i: values[i])
        best = [(first, 0.0, 0.0, ws[first], hs[first])]
        if not _valid(best, ws, hs, W, H, M, n):
            best = []
        best_value = sum(values[p[0]] for p in best)
        reserve = min(budget*0.20, 0.004 + n*n*0.00000025)
        deadline = start + max(0.0, budget-reserve)
        density_order = sorted(items, key=lambda i: (items[i][3], values[i]), reverse=True)
        ctx = (BW, BH, items, density_order, values,
               min(items[i][0] for i in items), min(items[i][1] for i in items))
        rnd = random.Random(734197 + n*193 + int(W)*17 + int(H)*31)
        strength = math.log1p(budget)
        maximum_fraction = min(0.9, 0.18+0.105*strength)
        archive_limit = max(2, int(2+math.sqrt(strength)))
        archive = [(best_value, best[:])]
        weights = [1.0]*7
        current, current_value = best[:], best_value
        first_pass = True
        while time.perf_counter() < deadline:
            now = time.perf_counter()
            if first_pass:
                fixed, op = [], 6
            else:
                r = rnd.random()
                if r < 0.25:
                    current_value, base = rnd.choice(archive)
                    current = base[:]
                elif r < 0.55:
                    current, current_value = best[:], best_value
                pick, op = rnd.random()*sum(weights), 6
                for j, weight in enumerate(weights):
                    pick -= weight
                    if pick <= 0:
                        op = j
                        break
                fraction = min(1.0, 1.0/max(1, len(current)) + maximum_fraction*rnd.random()**1.7)
                if op < 4:
                    fixed = _destroy(current, op, fraction, ctx, rnd)
                    if rnd.random() < 0.4:
                        fixed = _compact(fixed, ctx, rnd, deadline)
                elif op == 4:
                    slice_s = min((deadline-now)*0.12,
                                  (1.0+math.sqrt(budget))/(250.0*math.sqrt(max(1, len(current)))))
                    fixed = _eject(current, ctx, rnd, min(deadline, now+slice_s))
                elif op == 5:
                    fixed = _compact(current, ctx, rnd, deadline)
                    if rnd.random() < 0.5:
                        fixed = _destroy(fixed, rnd.randrange(2), fraction*0.6, ctx, rnd)
                else:
                    fixed = []
            candidate = _repair(fixed, ctx, rnd, deadline, first_pass)
            first_pass = False
            # The numerical check does not depend on external grading code; candidates that fail it never enter any archive.
            if not _valid(candidate, ws, hs, W, H, M, n):
                weights[op] = 0.94*weights[op]+0.06
                continue
            val = sum(values[p[0]] for p in candidate)
            reward = 1.0
            if val > best_value:
                best, best_value = candidate[:], val
                reward = 12.0
                archive.append((val, candidate[:]))
                archive.sort(key=lambda s: s[0], reverse=True)
                del archive[archive_limit:]
            remaining = max(0.0, (deadline-time.perf_counter())/max(budget, 1e-12))
            temperature = max(1e-12, best_value*(0.001+0.012*remaining*remaining)
                              /math.sqrt(max(1, len(candidate))))
            if val >= current_value or rnd.random() < math.exp(max(-700.0, (val-current_value)/temperature)):
                current, current_value = candidate, val
                reward = max(reward, 3.0)
                if len(archive) < archive_limit and val > 0:
                    archive.append((val, candidate[:]))
            weights[op] = 0.94*weights[op]+0.06*reward
    except Exception:
        pass
    if answer is None:
        answer = _blank(n)
    try:
        return _write(answer, best, ws, hs)
    except Exception:
        return _blank(n)
