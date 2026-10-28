"""Adaptive-neighbourhood landing scheduling with exact timing
Combines feasible insertion, sequence perturbation, linear-programming timing and mixed-integer optimisation over flight blocks, always keeping the best feasible solution found.
"""

import math
import time
import numpy as np
from scipy.optimize import linprog, milp, Bounds, LinearConstraint
from scipy.sparse import coo_matrix


def solve(instance: dict, time_limit_s: float) -> dict:
    start = time.perf_counter()
    duration = max(0.001, float(time_limit_s))
    deadline = start + duration - min(0.25, 0.035 * duration)
    planes = instance['planes']
    runways = instance['runways']
    n, m = len(planes), len(runways)
    if not n:
        return {'runway_assignment': {}, 'landing_times': {}, 'landing_sequence': []}
    E = np.asarray(instance['earliest_landing_time'], dtype=float)
    T = np.asarray(instance['target_landing_time'], dtype=float)
    L = np.asarray(instance['latest_landing_time'], dtype=float)
    alpha = np.asarray(instance['earliness_penalty_cost'], dtype=float)
    beta = np.asarray(instance['lateness_penalty_cost'], dtype=float)
    S = np.asarray(instance['separation_matrix'], dtype=float)
    M = float(instance['big_m'])
    ids = [str(p) for p in planes]
    rng = np.random.default_rng(73129)
    indices = np.arange(n)
    safe_m = bool(np.all(np.max(S, axis=(2, 3)) + L[:, None] - E[None, :] <= M + 1e-9))
    pair_u, pair_v = np.triu_indices(n, 1)
    objective = np.r_[np.zeros(n), alpha, beta]
    best = None
    best_cost = math.inf
    last_improvement = start
    lp_seconds = max(0.015, 0.012 * math.sqrt(duration) * math.log1p(n))

    def cost(t):
        return float(np.dot(alpha, np.maximum(T - t, 0.0)) +
                     np.dot(beta, np.maximum(t - T, 0.0)))

    def gaps(a):
        if safe_m:
            return S[indices[:, None], indices[None, :], a[:, None], a[None, :]]
        d = np.full((n, n), -math.inf)
        for r in range(m):
            for k in range(m):
                d = np.maximum(d, S[:, :, r, k] - M *
                               (2 - (a == r)[:, None].astype(float) - (a == k)[None, :]))
        return d

    def feasible(q, a, t, d=None):
        if not np.all(np.isfinite(t)) or np.any(t < E - 1e-6) or np.any(t > L + 1e-6):
            return False
        if np.any(np.diff(t[q]) < -1e-7):
            return False
        if d is None:
            d = gaps(a)
        u, v = q[pair_u], q[pair_v]
        delta = t[v] - t[u]
        if np.any(delta < np.maximum(0.0, d[u, v]) - 1e-6):
            return False
        return safe_m or not np.any(delta > M - d[v, u] + 1e-6)

    def record(q, a, t, checked=False):
        nonlocal best, best_cost, last_improvement
        value = cost(t)
        if value < best_cost - 1e-8 and (checked or feasible(q, a, t)):
            best_cost = value
            best = (q.copy(), a.copy(), t.copy(), value)
            last_improvement = time.perf_counter()
        return value

    def timing(q, a, allowance, ceiling=math.inf):
        if time.perf_counter() >= deadline:
            return None
        d = gaps(a)
        h = np.maximum(0.0, d[np.ix_(q, q)])
        lo, hi = E[q].copy(), L[q].copy()
        if safe_m:
            for j in range(1, n):
                lo[j] = max(lo[j], float(np.max(lo[:j] + h[:j, j])))
                if lo[j] > hi[j] + 1e-7:
                    return None
            for i in range(n - 2, -1, -1):
                hi[i] = min(hi[i], float(np.min(hi[i + 1:] - h[i, i + 1:])))
            if np.any(lo > hi + 1e-7):
                return None
            lower = float(np.dot(alpha[q], np.maximum(T[q] - hi, 0.0)) +
                          np.dot(beta[q], np.maximum(lo - T[q], 0.0)))
            if lower > ceiling + 1e-7:
                return None
            forward = lo.copy()
            for j in range(n):
                low = lo[j] if not j else max(lo[j], float(np.max(forward[:j] + h[:j, j])))
                forward[j] = min(hi[j], max(low, T[q[j]]))
            backward = hi.copy()
            for i in range(n - 1, -1, -1):
                high = hi[i] if i == n - 1 else min(hi[i], float(np.min(backward[i + 1:] - h[i, i + 1:])))
                backward[i] = max(lo[i], min(high, T[q[i]]))
            tf = np.empty(n)
            tb = np.empty(n)
            tf[q], tb[q] = forward, backward
            t = tf if cost(tf) <= cost(tb) else tb
            candidate = (q.copy(), a.copy(), t, cost(t)) if feasible(q, a, t, d) else None
        else:
            candidate = None
        remaining = min(allowance, deadline - time.perf_counter())
        if remaining <= 0.002:
            return candidate
        u, v = q[pair_u], q[pair_v]
        g = np.maximum(0.0, d[u, v])
        mask = L[u] + g > E[v] + 1e-10
        u, v, g = u[mask], v[mask], g[mask]
        if not safe_m:
            uu, vv = q[pair_u], q[pair_v]
            reverse = d[vv, uu] - M
            mask = L[vv] + reverse > E[uu] + 1e-10
            u, v, g = np.r_[u, vv[mask]], np.r_[v, uu[mask]], np.r_[g, reverse[mask]]
        cnt = len(u)
        row = np.r_[np.arange(cnt), np.arange(cnt), cnt + indices, cnt + indices,
                    cnt + n + indices, cnt + n + indices]
        col = np.r_[u, v, indices, n + indices, indices, 2 * n + indices]
        val = np.r_[np.ones(cnt), -np.ones(cnt), -np.ones(n), -np.ones(n), np.ones(n), -np.ones(n)]
        matrix = coo_matrix((val, (row, col)), shape=(cnt + 2 * n, 3 * n)).tocsc()
        rhs = np.r_[-g, -T, T]
        lower_bounds, upper_bounds = E.copy(), L.copy()
        if safe_m:
            lower_bounds[q], upper_bounds[q] = lo, np.maximum(lo, hi)
        bounds = list(zip(lower_bounds, upper_bounds)) + [(0, None)] * (2 * n)
        try:
            result = linprog(objective, A_ub=matrix, b_ub=rhs, bounds=bounds,
                             method='highs', options={'time_limit': max(0.001, min(remaining, deadline - time.perf_counter()))})
            if result.x is not None:
                t = np.clip(result.x[:n], E, L)
                for j in range(1, n):
                    if t[q[j]] < t[q[j - 1]] and t[q[j - 1]] - t[q[j]] <= 1e-7:
                        t[q[j]] = t[q[j - 1]]
                value = cost(t)
                if feasible(q, a, t, d) and (candidate is None or value < candidate[3]):
                    candidate = (q.copy(), a.copy(), t, value)
        except Exception:
            pass
        return candidate

    def construct(attempt):
        if attempt == 0:
            priority = T
        elif attempt == 1:
            priority = L
        elif attempt == 2:
            priority = E
        else:
            weight = rng.random()
            spread = max(1.0, float(np.median(L - E)))
            priority = (1 - weight) * T + weight * L + rng.normal(0, spread / (2 + math.sqrt(n)), n)
        q = np.argsort(priority, kind='stable')
        a = np.zeros(n, dtype=int)
        earliest = E.copy()
        preferred = E.copy()
        lag = 0.0 if attempt == 0 else (1.0 if attempt == 1 else rng.random())
        for pos, i in enumerate(q):
            prev = q[:pos]
            if pos:
                sep = S[prev, i, a[prev], :]
                low = np.maximum(E[i], np.max(earliest[prev, None] + sep, axis=0))
                wanted = np.maximum(E[i] + (1 - lag) * (T[i] - E[i]),
                                    np.max(preferred[prev, None] + sep, axis=0))
            else:
                low = np.full(m, E[i])
                wanted = np.full(m, np.clip(T[i], E[i], L[i]))
            scores = beta[i] * np.maximum(wanted - T[i], 0) + 0.001 * np.maximum(low - E[i], 0)
            scores += np.maximum(low - L[i], 0) * (1 + float(np.sum(alpha + beta)))
            if attempt > 2:
                scores += rng.exponential(max(1e-6, float(np.mean(np.abs(scores))) * 0.1), m)
            r = int(np.argmin(scores))
            a[i], earliest[i], preferred[i] = r, low[r], wanted[r]
        return q, a

    def relocate(state, temperature):
        q, a, t, value = state
        penalties = alpha * np.maximum(T - t, 0) + beta * np.maximum(t - T, 0)
        if rng.random() < 0.7 and float(penalties.sum()) > 0:
            p = int(rng.choice(n, p=(penalties + penalties.mean() * 0.15) / (penalties.sum() * 1.15)))
        else:
            p = int(rng.integers(n))
        other = q[q != p]
        tt = t[other]
        aa = a[other]
        chosen = None
        lowest = math.inf
        for r in range(m):
            incoming = S[other, p, aa, r].copy()
            outgoing = S[p, other, r, aa].copy()
            if not safe_m:
                incoming.fill(-math.inf)
                outgoing.fill(-math.inf)
                for rr in range(m):
                    for kk in range(m):
                        incoming = np.maximum(incoming, S[other, p, rr, kk] - M * (2 - (aa == rr) - int(r == kk)))
                        outgoing = np.maximum(outgoing, S[p, other, rr, kk] - M * (2 - int(r == rr) - (aa == kk)))
            before = tt + np.maximum(0, incoming)
            after = tt - np.maximum(0, outgoing)
            lower = np.maximum(E[p], np.r_[-math.inf, np.maximum.accumulate(before)])
            upper = np.minimum(L[p], np.r_[np.minimum.accumulate(after[::-1])[::-1], math.inf])
            if not safe_m:
                lower = np.maximum(lower, np.r_[np.maximum.accumulate((tt + incoming - M)[::-1])[::-1], -math.inf])
                upper = np.minimum(upper, np.r_[math.inf, np.minimum.accumulate(tt + M - outgoing)])
            valid = lower <= upper + 1e-9
            if not np.any(valid):
                continue
            times = np.minimum(upper, np.maximum(lower, T[p]))
            own = alpha[p] * np.maximum(T[p] - times, 0) + beta[p] * np.maximum(times - T[p], 0)
            scores = own + rng.gumbel(0, max(1e-9, temperature), n) if temperature > 0 else own + rng.random(n) * 1e-9
            scores[~valid] = math.inf
            slot = int(np.argmin(scores))
            if scores[slot] < lowest:
                lowest = float(scores[slot])
                chosen = (r, slot, times[slot])
        if chosen is None:
            return state
        r, slot, when = chosen
        qa, ar, tr = np.insert(other, slot, p), a.copy(), t.copy()
        ar[p], tr[p] = r, when
        if not safe_m and not feasible(qa, ar, tr):
            return state
        return (qa, ar, tr, cost(tr))

    def integer_block(state, left, width, allowance):
        build_end = min(deadline - 0.002, time.perf_counter() + allowance)
        if build_end <= time.perf_counter():
            return None
        if state is None:
            q = np.argsort(T, kind='stable')
            a = np.zeros(n, dtype=int)
            ceiling = math.inf
        else:
            q, a, old_t, ceiling = state
        free = list(map(int, q[left:left + width]))
        free_set = set(free)
        y = {}
        count = 3 * n
        for i in free:
            for r in range(m):
                y[i, r] = count
                count += 1
        z = {}
        for ii, i in enumerate(free):
            for j in free[ii + 1:]:
                z[i, j] = count
                count += 1
        obj = np.r_[objective, np.zeros(count - 3 * n)]
        lb = np.r_[E, np.zeros(count - n)]
        ub = np.r_[L, np.full(2 * n, math.inf), np.ones(count - 3 * n)]
        integer = np.r_[np.zeros(3 * n, dtype=np.uint8), np.ones(count - 3 * n, dtype=np.uint8)]
        rows, cols, data, rhs = [], [], [], []

        def add(columns, values, bound):
            row = len(rhs)
            rows.extend([row] * len(columns))
            cols.extend(columns)
            data.extend(values)
            rhs.append(bound)

        for i in range(n):
            add([i, n + i], [-1.0, -1.0], -T[i])
            add([i, 2 * n + i], [1.0, -1.0], T[i])
        for i in free:
            columns = [y[i, r] for r in range(m)]
            add(columns, [1.0] * m, 1.0)
            add(columns, [-1.0] * m, -1.0)
        if math.isfinite(ceiling):
            nonzero = np.flatnonzero(obj)
            add(nonzero.tolist(), obj[nonzero].tolist(), ceiling + 1e-7)
        for pi in range(n):
            if time.perf_counter() >= build_end - 0.002:
                return None
            i = int(q[pi])
            fi = i in free_set
            for pj in range(pi + 1, n):
                j = int(q[pj])
                fj = j in free_set
                zv = z.get((i, j))
                ri = range(m) if fi or not safe_m else [int(a[i])]
                rj = range(m) if fj or not safe_m else [int(a[j])]
                for r in ri:
                    for k in rj:
                        for reverse in (False, True):
                            if reverse and zv is None and safe_m:
                                continue
                            first, second = (j, i) if reverse else (i, j)
                            sep = S[j, i, k, r] if reverse else S[i, j, r, k]
                            big = max(0.0, float(L[first] + sep - E[second])) if safe_m else M
                            if safe_m and big <= 1e-10:
                                continue
                            cc, vv = [first, second], [1.0, -1.0]
                            absent = 0
                            if fi:
                                cc.append(y[i, r]); vv.append(big); absent += 1
                            elif r != a[i]:
                                absent += 1
                            if fj:
                                cc.append(y[j, k]); vv.append(big); absent += 1
                            elif k != a[j]:
                                absent += 1
                            if zv is not None:
                                cc.append(zv)
                                vv.append(-big if reverse else big)
                                if not reverse:
                                    absent += 1
                            elif reverse:
                                absent += 1
                            add(cc, vv, big * absent - sep)
        # Triangle inequalities ensure local binary precedence relations can be restored into a landing sequence.
        for ii, i in enumerate(free):
            for jj in range(ii + 1, len(free)):
                j = free[jj]
                for k in free[jj + 1:]:
                    columns = [z[i, j], z[j, k], z[i, k]]
                    add(columns, [1, 1, -1], 1)
                    add(columns, [-1, -1, 1], 0)
        remaining = min(build_end, deadline) - time.perf_counter()
        if remaining <= 0.005:
            return None
        matrix = coo_matrix((data, (rows, cols)), shape=(len(rhs), count)).tocsc()
        try:
            result = milp(obj, integrality=integer, bounds=Bounds(lb, ub),
                          constraints=LinearConstraint(matrix, -math.inf, np.asarray(rhs)),
                          options={'time_limit': max(0.001, min(build_end, deadline) - time.perf_counter()),
                                   'mip_rel_gap': 0.00001, 'presolve': True})
            if result.x is None:
                return None
            x = result.x
            ar, tr = a.copy(), np.clip(x[:n], E, L)
            for i in free:
                ar[i] = int(np.argmax([x[y[i, r]] for r in range(m)]))
            ranks = np.empty(n)
            ranks[q] = np.arange(n)
            for i in free:
                ranks[i] = left
            for (i, j), index in z.items():
                if x[index] > 0.5:
                    ranks[j] += 1
                else:
                    ranks[i] += 1
            qa = np.array(sorted(range(n), key=lambda i: (round(float(tr[i]), 7), ranks[i])), dtype=int)
            for j in range(1, n):
                if tr[qa[j]] < tr[qa[j - 1]] and tr[qa[j - 1]] - tr[qa[j]] <= 1e-7:
                    tr[qa[j]] = tr[qa[j - 1]]
            if feasible(qa, ar, tr):
                return (qa, ar, tr, cost(tr))
        except Exception:
            pass
        return None

    current = None
    attempt = 0
    construction_end = start + 0.16 * duration
    while time.perf_counter() < min(construction_end, deadline) and (current is None or attempt < 3):
        q, a = construct(attempt)
        state = timing(q, a, lp_seconds)
        if state is not None:
            record(*state[:3], checked=True)
            if current is None or state[3] < current[3]:
                current = state
        attempt += 1
    if current is None and time.perf_counter() < deadline:
        state = integer_block(None, 0, n, max(0.001, (deadline - time.perf_counter()) * 0.82))
        if state is not None:
            record(*state[:3], checked=True)
            current = state
    # If the initial construction fails, keep searching; unverified candidates are not written to the best solution.
    while current is None and time.perf_counter() < deadline:
        q, a = construct(attempt)
        attempt += 1
        state = timing(q, a, lp_seconds)
        if state is not None:
            record(*state[:3], checked=True)
            current = state

    base_width = min(n, max(2, int(2 + 1.5 * math.sqrt(duration) + 0.8 * math.log1p(n))))
    mip_interval = max(0.015, 0.018 * math.sqrt(duration) * math.log1p(n))
    next_mip = time.perf_counter() + mip_interval
    next_lp = time.perf_counter()
    dirty = False
    while current is not None and time.perf_counter() < deadline:
        now = time.perf_counter()
        if best_cost <= 1e-9:
            break
        elapsed = now - start
        remaining = deadline - now
        stagnation = (now - last_improvement) / max(0.01, duration)
        scale = max(1e-4, best_cost / max(1, n), float(np.mean(alpha + beta)))
        phase = (elapsed / max(0.01, duration * 0.23)) % 1.0
        temperature = scale * (0.03 + 0.30 * stagnation) * (1 - phase) ** 2
        if now >= next_mip and remaining > 0.03:
            width = min(n, max(2, int(base_width * (0.65 + rng.random() * 0.6 + stagnation))))
            source = best if rng.random() < 0.85 else current
            q, a, t, value = source
            penalties = alpha * np.maximum(T - t, 0) + beta * np.maximum(t - T, 0)
            weights = penalties[q] + max(1e-9, float(penalties.mean()) * 0.2)
            center = int(rng.choice(n, p=weights / weights.sum()))
            left = max(0, min(n - width, center - int(rng.integers(width))))
            budget = min(remaining * 0.45, max(0.025, mip_interval * (1.8 + 4 * stagnation)))
            state = integer_block(source, left, width, budget)
            if state is not None:
                record(*state[:3], checked=True)
                if state[3] <= current[3] + 1e-7:
                    current = state
                    dirty = False
            next_mip = time.perf_counter() + mip_interval
        elif dirty and now >= next_lp:
            state = timing(current[0], current[1], min(lp_seconds, remaining * 0.2))
            if state is not None:
                current = state
                record(*state[:3], checked=True)
            dirty = False
            next_lp = time.perf_counter() + max(0.002, lp_seconds * 0.25)
        elif rng.random() < 0.64:
            current = relocate(current, temperature if rng.random() < 0.3 else 0.0)
            record(*current[:3])
            dirty = True
        else:
            q, a = current[0].copy(), current[1].copy()
            radius = min(n - 1, max(1, int(math.sqrt(n) * (1 + stagnation + math.log1p(duration) * 0.15))))
            p = int(rng.integers(n))
            j = max(0, min(n - 1, p + int(rng.integers(-radius, radius + 1))))
            plane = int(q[p])
            if rng.random() < 0.5:
                q = np.insert(np.delete(q, p), j, plane)
            else:
                q[p], q[j] = q[j], q[p]
            if m > 1:
                a[plane] = int(rng.integers(m))
                if rng.random() < 0.35:
                    a[int(q[p])] = int(rng.integers(m))
            threshold = current[3] + rng.exponential(max(temperature, 1e-9))
            state = timing(q, a, min(lp_seconds, remaining * 0.2), threshold)
            if state is not None:
                record(*state[:3], checked=True)
                if state[3] <= threshold:
                    current = state
                    dirty = False
            if rng.random() < min(0.3, 0.025 + stagnation * 0.3):
                current = best
                dirty = False

    if best is None:
        # Keep the interface complete even for abnormally short time limits; for normal feasible inputs the construction above or the integer model provides a solution.
        q = np.argsort(T, kind='stable')
        a = np.zeros(n, dtype=int)
        t = np.clip(T, E, L)
    else:
        q, a, t, _ = best
    sequence = sorted(map(int, q), key=lambda i: float(t[i]))
    return {'runway_assignment': {ids[i]: int(runways[int(a[i])]) for i in range(n)},
            'landing_times': {ids[i]: float(t[i]) for i in range(n)},
            'landing_sequence': [ids[i] for i in sequence]}
