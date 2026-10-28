"""Anytime heuristic for selective unequal-circle packing.

The search combines a diverse beam of contact-graph constructions with spatial
ruin/recreate and profitable ejection moves.  All internal geometry is scaled
to a unit container; returned coordinates are shifted and scaled back.
"""

import math
import random
import time


def solve(instance: dict, time_limit_s: float) -> dict:
    t0 = time.time()
    try:
        return _solve(instance, float(time_limit_s), t0)
    except Exception:
        return _empty(instance)


def _empty(instance):
    try:
        n = max(0, int(instance.get("n", 0)))
        cx = float(instance.get("cx", 0.0))
        cy = float(instance.get("cy", 0.0))
    except Exception:
        n, cx, cy = 0, 0.0, 0.0
    return {
        "packing_status": [0] * n,
        "center_x_coordinates": [cx] * n,
        "center_y_coordinates": [cy] * n,
    }


def _solve(inst, time_limit_s, t0):
    deadline = t0 + max(0.0, time_limit_s) * 0.85
    search_deadline = deadline
    n = max(0, int(inst.get("n", 0)))
    cx = float(inst.get("cx", 0.0))
    cy = float(inst.get("cy", 0.0))
    radius = float(inst.get("R", 0.0))
    raw = list(inst.get("radii") or [])
    rr = []
    for i in range(n):
        try:
            rr.append(float(raw[i]))
        except Exception:
            rr.append(radius + 1.0)

    def make_output(placed):
        status = [0] * n
        xs = [cx] * n
        ys = [cy] * n
        if radius <= 0.0:
            return {
                "packing_status": status,
                "center_x_coordinates": xs,
                "center_y_coordinates": ys,
            }
        # A final independent, conservative feasibility pass.  A questionable
        # circle is omitted instead of risking an all-or-nothing invalid result.
        accepted = []
        tol = 2.0e-8
        for idx, x, y in placed:
            r = rr[idx] / radius
            if r <= 0.0 or r > 1.0:
                continue
            if math.hypot(x, y) + r > 1.0 - tol:
                continue
            good = True
            for j, u, v in accepted:
                if math.hypot(x - u, y - v) < r + rr[j] / radius + tol:
                    good = False
                    break
            if good:
                accepted.append((idx, x, y))
        for idx, x, y in accepted:
            status[idx] = 1
            xs[idx] = cx + radius * x
            ys[idx] = cy + radius * y
        return {
            "packing_status": status,
            "center_x_coordinates": xs,
            "center_y_coordinates": ys,
        }

    if n == 0 or radius <= 0.0 or time.time() >= deadline:
        return make_output([])

    rads = [r / radius for r in rr]
    eligible = [i for i in range(n) if 0.0 < rads[i] < 1.0]
    exactly = [i for i in range(n) if rads[i] == 1.0]
    if not eligible:
        if exactly:
            # Give an exact-container circle a tiny inward-safe special case.
            i = exactly[0]
            status = [0] * n
            xs = [cx] * n
            ys = [cy] * n
            status[i] = 1
            return {
                "packing_status": status,
                "center_x_coordinates": xs,
                "center_y_coordinates": ys,
            }
        return make_output([])

    eps = 8.0e-8
    rng = random.Random(1785947586)
    weights = [r * r for r in rads]
    total_weight = sum(weights[i] for i in eligible)

    def value(pl):
        return sum(weights[p[0]] for p in pl)

    def feasible_position(idx, x, y, pl):
        r = rads[idx]
        lim = 1.0 - r - eps
        if x * x + y * y > lim * lim + 1.0e-13:
            return False
        for j, u, v in pl:
            sep = r + rads[j] + eps
            dx = x - u
            dy = y - v
            if dx * dx + dy * dy < sep * sep - 1.0e-13:
                return False
        return True

    def intersections(x0, y0, a, x1, y1, b, target):
        dx = x1 - x0
        dy = y1 - y0
        d2 = dx * dx + dy * dy
        if d2 <= 1.0e-22:
            return
        hi = a + b
        lo = abs(a - b)
        if d2 > hi * hi + 1.0e-14 or d2 < lo * lo - 1.0e-14:
            return
        d = math.sqrt(d2)
        q = (a * a - b * b + d2) / (2.0 * d)
        h2 = a * a - q * q
        if h2 < -1.0e-13:
            return
        h = math.sqrt(max(0.0, h2))
        ux, uy = dx / d, dy / d
        mx, my = x0 + q * ux, y0 + q * uy
        target.append((mx - uy * h, my + ux * h))
        if h > 1.0e-12:
            target.append((mx + uy * h, my - ux * h))

    def raw_points(idx, pl, add_grid):
        """All maximally-supported contact locations for one circle."""
        r = rads[idx]
        lim = 1.0 - r - eps
        pts = [(0.0, 0.0)]
        if lim < 0.0:
            return []
        # A small polar net supplies different contact-graph components; exact
        # tangent intersections then refine them without numerical relaxation.
        if add_grid:
            rings = (0.35, 0.68, 1.0)
            phase = rng.random() * math.pi * 2.0
            for ring in rings:
                count = 10 if ring < 1.0 else 18
                for q in range(count):
                    a = phase + q * 2.0 * math.pi / count
                    pts.append((lim * ring * math.cos(a),
                                lim * ring * math.sin(a)))
        else:
            for q in range(8):
                a = q * math.pi / 4.0
                pts.append((lim * math.cos(a), lim * math.sin(a)))

        k = len(pl)
        for j in range(k):
            _, x, y = pl[j]
            sep = r + rads[pl[j][0]] + eps
            intersections(0.0, 0.0, lim, x, y, sep, pts)
        for a in range(k):
            ia, xa, ya = pl[a]
            da = r + rads[ia] + eps
            for b in range(a + 1, k):
                if (b & 15) == 0 and time.time() >= search_deadline:
                    return pts
                ib, xb, yb = pl[b]
                db = r + rads[ib] + eps
                intersections(xa, ya, da, xb, yb, db, pts)
        return pts

    def position_options(idx, pl, branch, noisy):
        pts = raw_points(idx, pl, len(pl) < 3)
        options = []
        r = rads[idx]
        old_w = value(pl)
        sx = sum(weights[j] * x for j, x, _ in pl)
        sy = sum(weights[j] * y for j, _, y in pl)
        new_w = old_w + weights[idx]
        for pno, (x, y) in enumerate(pts):
            if (pno & 63) == 0 and time.time() >= search_deadline:
                break
            if not feasible_position(idx, x, y, pl):
                continue
            gaps = [1.0 - r - math.hypot(x, y)]
            contacts = 0
            for j, u, v in pl:
                g = math.hypot(x - u, y - v) - r - rads[j]
                gaps.append(g)
                if g < 6.0 * eps:
                    contacts += 1
            gaps.sort()
            third = gaps[min(2, len(gaps) - 1)]
            balance = math.hypot(sx + weights[idx] * x,
                                 sy + weights[idx] * y) / max(new_w, 1.0e-12)
            # Tight third support predicts little unusable crescent waste, while
            # balance prevents a whole beam from collapsing to one boundary arc.
            merit = (-third / max(r, 0.02) - 0.13 * balance
                     + 0.025 * contacts)
            if noisy:
                merit += rng.uniform(-0.12, 0.12)
            options.append((merit, x, y, balance, math.hypot(x, y)))
        if not options:
            return []
        options.sort(reverse=True)
        chosen = options[:branch]
        # Preserve geometrically distinct alternatives, not merely the top few
        # variants of the same boundary pocket.
        if branch >= 3:
            chosen.append(min(options, key=lambda z: z[3]))
            chosen.append(max(options, key=lambda z: z[4]))
        answer = []
        seen = set()
        for merit, x, y, _, _ in chosen:
            key = (round(x, 6), round(y, 6))
            if key not in seen:
                seen.add(key)
                answer.append((merit, x, y))
            if len(answer) >= branch:
                break
        return answer

    # Immediate incumbent: even an externally tiny budget returns a valid item.
    first = max(eligible, key=lambda i: weights[i])
    best = [(first, 0.0, 0.0)]
    best_value = weights[first]
    if time.time() >= deadline:
        return make_output(best)

    def greedy(order, base, mode):
        pl = list(base)
        for idx in order:
            if time.time() >= search_deadline:
                break
            if any(q[0] == idx for q in pl):
                continue
            opts = position_options(idx, pl, 4 if mode else 1, bool(mode))
            if opts:
                pick = 0 if not mode else rng.randrange(min(3, len(opts)))
                _, x, y = opts[pick]
                pl.append((idx, x, y))
        return pl

    def adaptive_fill(base, mode):
        """Jointly choose the next circle and its contact location.

        A fixed insertion order throws away useful information: a pocket which
        is mediocre for today's circle can be the sole good home of another.
        Here all still-missing circles compete for every next placement.
        """
        pl = list(base)
        present = {p[0] for p in pl}
        missing = [i for i in eligible if i not in present]
        while missing and time.time() < search_deadline:
            if len(missing) > 28:
                by_size = sorted(missing, key=lambda i: rads[i])
                sample = by_size[:8] + by_size[-12:]
                sample_set = set(sample)
                rest = [i for i in missing if i not in sample_set]
                rng.shuffle(rest)
                sample += rest[:8]
            else:
                sample = missing
            choices = []
            for idx in sample:
                opts = position_options(idx, pl, 2, mode == 3)
                if opts:
                    merit, x, y = opts[0]
                    if mode == 0:       # raw area: hard to dislodge big circles
                        key = weights[idx] + 1.0e-5 * merit
                    elif mode == 1:     # best-fitting hole: minimize dead space
                        key = merit + 0.15 * weights[idx] / max(total_weight, 1e-12)
                    elif mode == 2:     # balanced value / geometric opportunity
                        key = weights[idx] / max(total_weight, 1e-12) + 0.035 * merit
                    else:               # randomized GRASP version of the above
                        key = (weights[idx] / max(total_weight, 1e-12)
                               + 0.04 * merit + rng.uniform(-0.025, 0.025))
                    choices.append((key, idx, x, y))
                if time.time() >= search_deadline:
                    break
            if not choices:
                break
            choices.sort(reverse=True)
            take = 0
            if mode == 3 and len(choices) > 1 and rng.random() < 0.3:
                take = 1
            _, idx, x, y = choices[take]
            pl.append((idx, x, y))
            missing.remove(idx)
        return pl

    # Fast complete constructions before the more expensive beam.
    descending = sorted(eligible, key=lambda i: -rads[i])
    ascending = list(reversed(descending))
    orders = [descending, ascending]
    zigzag = []
    lo, hi = 0, len(ascending) - 1
    while lo <= hi:
        zigzag.append(ascending[lo])
        lo += 1
        if lo <= hi:
            zigzag.append(ascending[hi])
            hi -= 1
    orders.append(zigzag)
    for order in orders:
        pl = greedy(order, [], 0)
        z = value(pl)
        if z > best_value:
            best, best_value = pl, z
        if time.time() >= deadline:
            return make_output(best)

    for mode in range(3):
        pl = adaptive_fill([], mode)
        z = value(pl)
        if z > best_value:
            best, best_value = pl, z
        if time.time() >= deadline:
            return make_output(best)

    def beam_construct(order, width):
        # state = (packing, area, cumulative topology merit)
        states = [([], 0.0, 0.0)]
        for idx in order:
            if time.time() >= search_deadline:
                break
            pool = []
            for pl, val, quality in states:
                pool.append((pl, val, quality - 0.002))  # explicit skip branch
                opts = position_options(idx, pl, 3, False)
                for merit, x, y in opts:
                    pool.append((pl + [(idx, x, y)], val + weights[idx],
                                 quality + merit))
                if time.time() >= search_deadline:
                    break
            # Area dominates, but a bounded topology term keeps alternatives
            # alive long enough for later circles to expose their true value.
            pool.sort(key=lambda s: s[1] + 0.006 * total_weight
                      * math.tanh(s[2] / max(1, len(s[0]))), reverse=True)
            states = []
            mask_count = {}
            for state in pool:
                mask = tuple(sorted(p[0] for p in state[0]))
                count = mask_count.get(mask, 0)
                if count < 3:
                    states.append(state)
                    mask_count[mask] = count + 1
                if len(states) >= width:
                    break
        if not states:
            return []
        return max(states, key=lambda s: s[1])[0]

    # Descending and zig-zag beams attack the high-area and high-density sides.
    for bno, order in enumerate((descending, zigzag, ascending)):
        if time.time() >= search_deadline:
            break
        width = 22 if len(eligible) <= 35 else 10
        pl = beam_construct(order, width)
        z = value(pl)
        if z > best_value:
            best, best_value = pl, z

    # Random-key greedy restarts diversify the incumbent before LNS.
    for _ in range(5):
        if time.time() >= deadline:
            break
        bias = rng.uniform(-1.5, 2.5)
        order = sorted(eligible,
                       key=lambda i: rng.random() / (rads[i] ** bias))
        pl = greedy(order, [], 1)
        z = value(pl)
        if z > best_value:
            best, best_value = pl, z

    current = list(best)
    current_value = best_value
    stagnant = 0
    search_deadline = deadline

    def refill(pl, mode):
        present = {p[0] for p in pl}
        missing = [i for i in eligible if i not in present]
        if mode == 0:
            missing.sort(key=lambda i: -rads[i])
        elif mode == 1:
            missing.sort(key=lambda i: rads[i])
        else:
            missing.sort(key=lambda i: rng.random() / max(weights[i], 1.0e-12))
        # Alternate cheap fixed-order rebuilding with the more expensive joint
        # circle/location decision.  Both are useful under a strict wall clock.
        if mode == 0:
            return greedy(missing, pl, 0)
        return adaptive_fill(pl, mode + 1)

    def ruin(pl, count, style):
        if not pl:
            return []
        count = min(count, len(pl))
        seed = rng.choice(pl)
        if style == 0:
            ranked = sorted(pl, key=lambda p: (p[1] - seed[1]) ** 2
                            + (p[2] - seed[2]) ** 2)
        elif style == 1:
            angle = math.atan2(seed[2], seed[1])
            ranked = sorted(pl, key=lambda p: abs(math.sin(
                0.5 * (math.atan2(p[2], p[1]) - angle))))
        else:
            ranked = list(pl)
            rng.shuffle(ranked)
        removed = {p[0] for p in ranked[:count]}
        return [p for p in pl if p[0] not in removed]

    def ejection_move(pl):
        """Insert one missing disk while ejecting lower-area blockers."""
        present = {p[0] for p in pl}
        missing = [i for i in eligible if i not in present]
        missing.sort(key=lambda i: -weights[i])
        best_gain = 0.0
        best_move = None
        for idx in missing[:12]:
            if time.time() >= deadline:
                break
            r = rads[idx]
            for pno, (x, y) in enumerate(raw_points(idx, pl, True)):
                if (pno & 63) == 0 and time.time() >= deadline:
                    break
                if math.hypot(x, y) + r > 1.0 - eps:
                    continue
                blockers = []
                loss = 0.0
                for j, u, v in pl:
                    if math.hypot(x - u, y - v) < r + rads[j] + eps:
                        blockers.append(j)
                        loss += weights[j]
                gain = weights[idx] - loss
                if blockers and gain > best_gain + 1.0e-12:
                    best_gain = gain
                    best_move = (idx, x, y, blockers)
        if best_move is None:
            return pl
        idx, x, y, blockers = best_move
        blocked = set(blockers)
        return [p for p in pl if p[0] not in blocked] + [(idx, x, y)]

    # Anytime large-neighbourhood search.  It has no iteration cap: surplus
    # budget is always converted into another topology perturbation.
    iteration = 0
    while time.time() < deadline:
        iteration += 1
        if iteration % 5 == 0:
            candidate = ejection_move(current)
            candidate = refill(candidate, iteration % 3)
        else:
            upper = min(6, max(2, len(current) // 2))
            q = 1 + rng.randrange(upper)
            candidate = refill(ruin(current, q, iteration % 3), iteration % 3)
        z = value(candidate)
        if z > best_value + 1.0e-12:
            best, best_value = list(candidate), z
            current, current_value = list(candidate), z
            stagnant = 0
        elif z >= current_value - (0.0025 * total_weight
                                    if stagnant > 8 else 1.0e-12):
            current, current_value = list(candidate), z
            stagnant += 1
        else:
            stagnant += 1
        if stagnant > 18:
            current = ruin(best, min(4, max(1, len(best) // 3)), 2)
            current = refill(current, 2)
            current_value = value(current)
            stagnant = 0

    return make_output(best)
