"""Anytime cardinality packing of axis-aligned rectangles in a circle."""
import math
import random
import time


def solve(instance: dict, time_limit_s: float) -> dict:
    started = time.perf_counter()
    answer = {"alpha": [], "x": [], "y": [], "r": []}
    best = []
    try:
        n = instance["num_rectangles"]
        answer = {"alpha": [0] * n, "x": [0.0] * n,
                  "y": [0.0] * n, "r": [0] * n}
        budget = max(0.0, float(time_limit_s))
        if not math.isfinite(budget) or budget <= 0.0 or n == 0:
            return answer
        widths, heights = instance["widths"], instance["heights"]
        radius = float(instance["container_radius"])
        reserve = min(budget * 0.12, 0.004 + n * 0.000003)
        deadline = started + budget - reserve
        now = time.perf_counter
        effort = math.log1p(budget)
        rng = random.Random(173921)
        # Work in radius units. Inflated boxes give roundoff clearance, but
        # every accepted move is also checked using its actual output numbers.
        gap = 4.0e-13
        tolerance = gap * 0.125
        inner_radius = 1.0 - gap
        rr = inner_radius * inner_radius
        dims = {}
        items = []
        fully_scanned = True
        for i in range(n):
            if now() >= deadline:
                fully_scanned = False
                break
            a = (float(widths[i]) / radius) * 0.5
            b = (float(heights[i]) / radius) * 0.5
            if not math.isfinite(a + b) or math.hypot(a, b) > 1.0:
                continue
            dims[i] = (a, b)
            la = math.log(max(a, 1.0e-300))
            lb = math.log(max(b, 1.0e-300))
            items.append((i, la + lb, math.log(max(a + b, 1.0e-300)),
                          max(la, lb)))
            if not best:
                # A centered rectangle is feasible whenever any translation is.
                if math.hypot(float(widths[i]) * 0.5,
                              float(heights[i]) * 0.5) <= radius:
                    best = [(i, 0.0, 0.0, a + gap, b + gap,
                             0, 0.0, 0.0, a, b)]
        if not items:
            return answer

        def realize(i, x, y, rotation, placed, end):
            """Return a certified placement, or reject without changing state."""
            a, b = dims[i]
            if rotation:
                a, b = b, a
            ox, oy = x * radius, y * radius
            x, y = ox / radius, oy / radius
            if not (math.isfinite(ox) and math.isfinite(oy)):
                return None
            if math.hypot(abs(x) + a, abs(y) + b) > 1.0:
                return None
            w = float(heights[i] if rotation else widths[i]) * 0.5
            h = float(widths[i] if rotation else heights[i]) * 0.5
            if math.hypot(abs(ox) + w, abs(oy) + h) > radius:
                return None
            for j, q in enumerate(placed):
                if j % 64 == 0 and now() >= end:
                    return None
                if abs(x - q[1]) < a + q[8] and abs(y - q[2]) < b + q[9]:
                    return None
                qw = float(heights[q[0]] if q[5] else widths[q[0]]) * 0.5
                qh = float(widths[q[0]] if q[5] else heights[q[0]]) * 0.5
                if abs(ox - q[6]) < w + qw and abs(oy - q[7]) < h + qh:
                    return None
            return (i, x, y, a + gap, b + gap, rotation, ox, oy, a, b)

        def locate(i, placed, by_y, by_x, beta, theta, end):
            """Scan contact lines; solve the remaining 1-D gaps exactly."""
            winner = None
            best_score = float("inf")
            sign = 1.0 if beta >= 0.0 else -1.0
            a0, b0 = dims[i]
            orientations = (0,) if a0 == b0 else (0, 1)
            cap = max(8, int((16.0 + 24.0 * effort) /
                             (1.0 + len(placed) / 160.0) ** 0.25))
            for rotation in orientations:
                if now() >= end:
                    break
                a, b = (b0 + gap, a0 + gap) if rotation else (a0 + gap, b0 + gap)
                if math.hypot(a, b) > inner_radius:
                    continue
                xmax = math.sqrt(max(0.0, rr - b * b)) - a
                ymax = math.sqrt(max(0.0, rr - a * a)) - b
                norm = math.hypot(1.0, beta)
                tx = -sign * max(0.0, inner_radius * abs(beta) / norm - a)
                ty = -max(0.0, inner_radius / norm - b)
                xs = [0.0, -xmax, xmax, tx]
                ys = [0.0, -ymax, ymax, ty]
                for q in placed:
                    xs.extend((q[1] - q[3] - a, q[1] + q[3] + a))
                    ys.extend((q[2] - q[4] - b, q[2] + q[4] + b))
                if len(xs) > cap:
                    xs = xs[:4] + rng.sample(xs[4:], cap - 4)
                    ys = ys[:4] + rng.sample(ys[4:], cap - 4)
                for x in xs:
                    if now() >= end:
                        break
                    if abs(x) > xmax + tolerance:
                        continue
                    extent = math.sqrt(max(0.0, rr - (abs(x) + a) ** 2)) - b
                    y = -extent
                    valid = True
                    for j, q in enumerate(by_y):
                        if j % 64 == 0 and now() >= end:
                            valid = False
                            break
                        if abs(x - q[1]) >= a + q[3] - tolerance:
                            continue
                        low = q[2] - q[4] - b
                        if y <= low + tolerance:
                            break
                        y = max(y, q[2] + q[4] + b)
                        if y > extent + tolerance:
                            valid = False
                            break
                    if valid and y <= extent + tolerance:
                        score = y + theta * b + beta * x + abs(beta) * theta * a
                        if score < best_score:
                            candidate = realize(i, x, y, rotation, placed, end)
                            if candidate is not None:
                                winner, best_score = candidate, score
                for y in ys:
                    if now() >= end:
                        break
                    if abs(y) > ymax + tolerance:
                        continue
                    extent = math.sqrt(max(0.0, rr - (abs(y) + b) ** 2)) - a
                    u = -extent
                    valid = True
                    for j, q in enumerate(by_x):
                        if j % 64 == 0 and now() >= end:
                            valid = False
                            break
                        if abs(y - q[2]) >= b + q[4] - tolerance:
                            continue
                        low = sign * q[1] - q[3] - a
                        if u <= low + tolerance:
                            break
                        u = max(u, sign * q[1] + q[3] + a)
                        if u > extent + tolerance:
                            valid = False
                            break
                    if valid and u <= extent + tolerance:
                        x = sign * u
                        score = y + theta * b + beta * x + abs(beta) * theta * a
                        if score < best_score:
                            candidate = realize(i, x, y, rotation, placed, end)
                            if candidate is not None:
                                winner, best_score = candidate, score
            return winner

        def compact(placed, beta, end):
            # Two coordinate sweeps form one neighborhood move, not a capped
            # optimization loop. Each successful individual move is certified.
            sign = 1.0 if beta >= 0.0 else -1.0
            for axis in (2, 1):
                direction = 1.0 if axis == 2 else sign
                order = sorted(range(len(placed)),
                               key=lambda j: direction * placed[j][axis])
                other_axis = 3 - axis
                for j in order:
                    if now() >= end:
                        return placed
                    p = placed[j]
                    half = p[axis + 2]
                    other_half = p[other_axis + 2]
                    corner = abs(p[other_axis]) + other_half
                    if corner * corner + half * half > rr:
                        continue
                    value = -math.sqrt(max(0.0, rr - corner * corner)) + half
                    old_value = direction * p[axis]
                    for k, q in enumerate(placed):
                        if k % 64 == 0 and now() >= end:
                            return placed
                        if k == j or abs(p[other_axis] - q[other_axis]) >= other_half + q[other_axis + 2] - tolerance:
                            continue
                        if direction * q[axis] < old_value:
                            value = max(value, direction * q[axis] + q[axis + 2] + half)
                    if value < old_value - gap:
                        x = direction * value if axis == 1 else p[1]
                        y = direction * value if axis == 2 else p[2]
                        others = placed[:j] + placed[j + 1:]
                        q = realize(p[0], x, y, p[5], others, end)
                        if q is not None:
                            placed[j] = q
            return placed

        population = [best[:]] if best else []
        population_limit = max(1, min(len(items), int(1.0 + 2.0 * effort)))
        turn = 0
        stagnant = 0
        while now() < deadline:
            if fully_scanned and len(best) == len(items):
                break  # Proven optimal: every individually feasible item fits.
            remaining = deadline - now()
            job_end = min(deadline, now() + max(0.001, budget / (2.0 + effort)))
            beta = 0.12 if turn == 0 else rng.uniform(-0.85, 0.85)
            theta = 1.0 if turn == 0 else rng.random()
            old_best = len(best)
            fresh = turn < math.ceil(1.0 + effort) or not population or rng.random() < 1.0 / (2.0 + effort)
            if fresh:
                placed = []
            else:
                base = best if rng.random() < 0.5 else rng.choice(population)
                k = len(base)
                depth = min(k, max(1, int(math.sqrt(k) * (0.6 + 0.4 * effort) *
                                         (1.0 + math.log1p(stagnant) / (2.0 + effort)))))
                remove_count = rng.randint(1, depth) if depth else 0
                mode = rng.randrange(4)
                if mode == 0 and base:
                    anchor = rng.choice(base)
                    ranking = sorted(base, key=lambda p: (p[1] - anchor[1]) ** 2 + (p[2] - anchor[2]) ** 2)
                elif mode == 1:
                    ranking = sorted(base, key=lambda p: -(p[2] + beta * p[1]))
                elif mode == 2:
                    ranking = sorted(base, key=lambda p: -(p[3] * p[4]) * (0.5 + rng.random()))
                else:
                    ranking = rng.sample(base, k)
                removed = {p[0] for p in ranking[:remove_count]}
                placed = [p for p in base if p[0] not in removed]
                if placed and rng.random() < effort / (1.0 + effort):
                    compact_end = min(job_end, now() + remaining / (4.0 + effort))
                    placed = compact(placed, beta, compact_end)
            selected = {p[0] for p in placed}
            order_mode = turn % 4
            noise = 0.0 if turn == 0 else rng.random() * (0.15 + 0.25 * effort)
            if order_mode == 0:
                priority = lambda t: t[1] + noise * rng.gauss(0.0, 1.0)
            elif order_mode == 1:
                priority = lambda t: 2.0 * t[2] + noise * rng.gauss(0.0, 1.0)
            elif order_mode == 2:
                priority = lambda t: 2.0 * t[3] + noise * rng.gauss(0.0, 1.0)
            else:
                mix = rng.random()
                priority = lambda t: (1.0 - mix) * t[1] + 2.0 * mix * t[3] + noise * rng.gauss(0.0, 1.0)
            order = sorted((t for t in items if t[0] not in selected), key=priority)
            sign = 1.0 if beta >= 0.0 else -1.0
            by_y = sorted(placed, key=lambda p: p[2] - p[4])
            by_x = sorted(placed, key=lambda p: sign * p[1] - p[3])
            for t in order:
                if now() >= job_end:
                    break
                p = locate(t[0], placed, by_y, by_x, beta, theta, job_end)
                if p is None:
                    continue
                placed.append(p)
                if len(placed) > len(best):
                    best = placed[:]
                by_y.append(p)
                by_x.append(p)
                by_y.sort(key=lambda q: q[2] - q[4])
                by_x.sort(key=lambda q: sign * q[1] - q[3])
                if fully_scanned and len(best) == len(items):
                    break
            if placed:
                if len(population) < population_limit:
                    population.append(placed[:])
                else:
                    worst = min(range(len(population)), key=lambda j: len(population[j]))
                    if len(placed) >= len(population[worst]):
                        population[worst] = placed[:]
            stagnant = 0 if len(best) > old_best else stagnant + 1
            turn += 1
    except Exception:
        # The incumbent consists only of individually certified placements.
        # A failed optional search step must never destroy the feasible answer.
        pass
    try:
        for p in best:
            i = p[0]
            answer["x"][i] = p[6]
            answer["y"][i] = p[7]
            answer["r"][i] = p[5]
            answer["alpha"][i] = 1
    except Exception:
        # For valid inputs the arrays and indices above are always consistent.
        pass
    return answer
