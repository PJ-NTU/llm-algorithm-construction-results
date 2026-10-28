"""Lagrangian-guided ordered-cover large neighbourhood search
Screens facilities with Lagrangian lower bounds, jointly adjusts the opening plan with an ordered cost-chain mixed-integer model, and keeps the current best feasible solution.
"""

import math
import numbers
import time

import numpy as np

try:
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import coo_matrix
except Exception:
    milp = None


def _fallback(instance):
    """Independent fallback for valid instances; invalid dimensions are rejected with empty decisions."""
    try:
        m, n = instance["m_facilities"], instance["n"]
        if (isinstance(m, bool) or isinstance(n, bool)
                or not isinstance(m, numbers.Integral)
                or not isinstance(n, numbers.Integral) or m < 1 or n < 1):
            return {"y": [], "x": []}
        y = [0] * int(m)
        y[0] = 1
        x = [[0] * int(n) for _ in range(int(m))]
        x[0] = [1] * int(n)
        return {"y": y, "x": x}
    except Exception:
        return {"y": [], "x": []}


def _assignment(c, opened):
    ids = np.flatnonzero(opened)
    a = ids[np.argmin(c[ids], axis=0)]
    return a, c[a, np.arange(c.shape[1])]


def _value(f, c, opened):
    return float(f[opened].sum() + _assignment(c, opened)[1].sum())


def _polish(f, c, opened, stop):
    """Open/close coordinate descent only; joint relocation is left to the cover subproblem."""
    opened = opened.copy()
    m, n = c.shape
    if not opened.any():
        opened[int(np.argmin(f + c.sum(axis=1)))] = True
    while time.perf_counter() < stop:
        ids = np.flatnonzero(opened)
        sub = c[ids]
        local = np.argmin(sub, axis=0)
        owners = ids[local]
        nearest = sub[local, np.arange(n)]
        value = float(f[ids].sum() + nearest.sum())
        tolerance = 2e-12 * max(1.0, value)
        gain, chosen, adding = tolerance, -1, False
        if ids.size > 1:
            second = np.partition(sub, 1, axis=0)[1]
            removal = f - np.bincount(owners, weights=second - nearest,
                                      minlength=m)
            i = int(ids[np.argmax(removal[ids])])
            if removal[i] > gain:
                gain, chosen = float(removal[i]), i
        block = max(1, 65536 // n)
        for begin in range(0, m, block):
            if time.perf_counter() >= stop:
                break
            end = min(m, begin + block)
            delta = np.maximum(nearest - c[begin:end], 0.0).sum(axis=1)
            delta -= f[begin:end]
            delta[opened[begin:end]] = -np.inf
            k = int(np.argmax(delta))
            if delta[k] > gain:
                gain, chosen, adding = float(delta[k]), begin + k, True
        if chosen < 0:
            break
        opened[chosen] = adding
    return opened, _value(f, c, opened)


def _dual(f, c, v, stop):
    """Lagrangian relaxation of demand equalities; returns lower bound, facility reduced costs and subgradient."""
    m, n = c.shape
    q = f.copy()
    g = np.ones(n)
    block = max(1, 65536 // n)
    for begin in range(0, m, block):
        if time.perf_counter() >= stop:
            return None
        end = min(m, begin + block)
        cc = c[begin:end]
        q[begin:end] -= np.maximum(v - cc, 0.0).sum(axis=1)
        active = q[begin:end] < 0.0
        if active.any():
            g -= np.count_nonzero(cc[active] < v, axis=0)
    lower = float(v.sum() + np.minimum(q, 0.0).sum())
    return lower, q, g


def _ordered_model(f, c, pool, outside, upper, stop, allowance):
    """One uncovered-cost chain per customer, without explicit customer assignment variables."""
    started = time.perf_counter()
    k, n = len(pool), c.shape[1]
    if k == 0:
        return outside.copy(), True
    costs = c[pool].T
    order = np.argsort(costs, axis=1, kind="stable")
    sorted_cost = np.take_along_axis(costs, order, axis=1)
    if outside.any():
        baseline = c[outside].min(axis=0)
    else:
        baseline = sorted_cost[:, -1]
    constant = float(f[outside].sum()
                     + np.minimum(sorted_cost[:, 0], baseline).sum())
    flat = np.flatnonzero((sorted_cost < baseline[:, None]).reshape(-1))
    nz = len(flat)
    if time.perf_counter() >= stop:
        return None, False
    if nz == 0:
        result = outside.copy()
        if not result.any():
            result[pool[int(np.argmin(f[pool]))]] = True
        return result, True
    # u_t >= u_(t-1) - y_(t-th cheapest facility), where u_0 = 1.
    next_cost = np.empty_like(sorted_cost)
    next_cost[:, :-1] = np.minimum(sorted_cost[:, 1:], baseline[:, None])
    next_cost[:, -1] = baseline
    weights = (next_cost - sorted_cost).reshape(-1)[flat]
    objective = np.concatenate((f[pool], weights))
    r = np.arange(nz, dtype=np.int64)
    first = flat % k == 0
    rr = [r, r, r[~first]]
    cc = [order.reshape(-1)[flat], k + r, k + r[~first] - 1]
    dd = [-np.ones(nz), -np.ones(nz), np.ones(np.count_nonzero(~first))]
    limits = [-first.astype(float)]
    nr = nz
    if not outside.any():
        rr.append(np.full(k, nr, dtype=np.int64))
        cc.append(np.arange(k, dtype=np.int64))
        dd.append(-np.ones(k))
        limits.append(np.array([-1.0]))
        nr += 1
    # The existing plan is representable in the subproblem; this truncation only excludes plans worse than it.
    nonzero = np.flatnonzero(objective)
    rr.append(np.full(len(nonzero), nr, dtype=np.int64))
    cc.append(nonzero)
    dd.append(objective[nonzero])
    limits.append(np.array([upper - constant + 2e-9 * max(1.0, upper)]))
    nr += 1
    matrix = coo_matrix((np.concatenate(dd),
                         (np.concatenate(rr), np.concatenate(cc))),
                        shape=(nr, k + nz)).tocsc()
    remaining = stop - time.perf_counter()
    seconds = min(allowance - (time.perf_counter() - started), 0.8 * remaining)
    if seconds <= 0.001:
        return None, False
    integrality = np.zeros(k + nz, dtype=np.uint8)
    integrality[:k] = 1
    result = milp(objective, integrality=integrality,
                  bounds=Bounds(0.0, 1.0),
                  constraints=LinearConstraint(matrix, -np.inf,
                                               np.concatenate(limits)),
                  options={"time_limit": float(seconds), "mip_rel_gap": 1e-10,
                           "disp": False})
    if result.x is None:
        return None, False
    opened = outside.copy()
    opened[pool[np.asarray(result.x[:k]) > 0.5]] = True
    if not opened.any():
        return None, False
    return opened, result.status == 0


def _neighborhood(f, c, best, q, lower, upper, width, rng, broad, stop):
    """Keep some facilities fixed; the rest are screened jointly by lower bounds and reassignment gains."""
    m, n = c.shape
    safety = 2e-8 * max(1.0, abs(upper), abs(lower))
    prohibited = ((lower + np.maximum(q, 0.0) > upper + safety) & ~best)
    required = ((lower + np.maximum(-q, 0.0) > upper + safety) & best)
    required |= f == 0.0
    free = np.flatnonzero(~prohibited & ~required)
    width = min(len(free), max(1, width))
    if width == len(free):
        return free, required, True
    variable_open = np.flatnonzero(best & ~required)
    if broad and len(variable_open) < width:
        removed = variable_open
    else:
        count = min(len(variable_open), max(1, width // 2))
        removed = rng.choice(variable_open, size=count, replace=False)
    outside = best.copy()
    outside[removed] = False
    outside |= required
    available = ~outside & ~prohibited
    available[removed] = False
    slots = width - len(removed)
    candidates = np.flatnonzero(available)
    if slots >= len(candidates):
        return np.concatenate((removed, candidates)), outside, False
    if slots <= 0:
        return removed, outside, False
    if outside.any():
        baseline = c[outside].min(axis=0)
    else:
        baseline = _assignment(c, best)[1]
    score = f.copy()
    block = max(1, 65536 // n)
    completed = True
    for begin in range(0, m, block):
        if time.perf_counter() >= stop:
            completed = False
            break
        end = min(m, begin + block)
        score[begin:end] -= np.maximum(baseline - c[begin:end], 0.0).sum(axis=1)
    rank_dual = np.empty(len(candidates), dtype=float)
    rank_dual[np.argsort(q[candidates], kind="stable")] = np.arange(len(candidates))
    if completed:
        rank_gain = np.empty(len(candidates), dtype=float)
        rank_gain[np.argsort(score[candidates], kind="stable")] = np.arange(len(candidates))
        score = (rank_dual + rank_gain) / max(1, 2 * len(candidates))
    else:
        score = rank_dual / max(1, len(candidates))
    noise = 0.04 if broad else 0.5
    score += noise * rng.standard_normal(len(candidates))
    selected = candidates[np.argpartition(score, slots - 1)[:slots]]
    return np.concatenate((removed, selected)), outside, False


def _encode(c, opened):
    m, n = c.shape
    assignments, _ = _assignment(c, opened)
    y = [0] * m
    x = [[0] * n for _ in range(m)]
    # Idle facilities need not be charged; the output opens only facilities that actually serve customers.
    for j, i in enumerate(assignments.tolist()):
        y[i] = 1
        x[i][j] = 1
    return {"y": y, "x": x}


def solve(instance: dict, time_limit_s: float) -> dict:
    started = time.perf_counter()
    best = None
    try:
        duration = float(time_limit_s)
        if not math.isfinite(duration) or duration <= 0.0:
            return _fallback(instance)
        m, n = instance["m_facilities"], instance["n"]
        if (isinstance(m, bool) or isinstance(n, bool)
                or not isinstance(m, numbers.Integral)
                or not isinstance(n, numbers.Integral) or m < 1 or n < 1):
            return {"y": [], "x": []}
        m, n = int(m), int(n)
        f = np.asarray(instance["fixed_cost"], dtype=float)
        c = np.asarray(instance["alloc_cost"], dtype=float)
        if (f.shape != (m,) or c.shape != (m, n)
                or not np.isfinite(f).all() or not np.isfinite(c).all()
                or np.any(f < 0.0) or np.any(c < 0.0)):
            return {"y": [], "x": []}
        # Subtract the cost common to all customers, then normalise, so large constants do not hinder bound comparison.
        c = c - c.min(axis=0)
        scale = max(float(f.max()), float(c.max()))
        if scale > 0.0:
            f = f / scale
            c /= scale
        best = np.zeros(m, dtype=bool)
        best[int(np.argmin(f + c.sum(axis=1)))] = True
        best |= f == 0.0
        upper = _value(f, c, best)
        parsed = time.perf_counter() - started
        reserve = min(0.35 * duration,
                      max(0.003, 0.04 * duration, 0.7 * parsed, 3e-8 * m * n))
        stop = started + duration - reserve
        usable = max(0.0, stop - started)
        if upper == 0.0 or m == 1 or time.perf_counter() >= stop:
            return _encode(c, best)
        rng = np.random.default_rng(718321)
        # Build another feasible plan by apportioning fixed costs among customers, and initialise the multipliers.
        v = np.full(n, np.inf)
        ceiling = np.full(n, np.inf)
        owner = np.zeros(n, dtype=np.int64)
        block = max(1, 65536 // n)
        for begin in range(0, m, block):
            if time.perf_counter() >= stop:
                return _encode(c, best)
            end = min(m, begin + block)
            piece = c[begin:end] + f[begin:end, None] / n
            choices = np.argmin(piece, axis=0)
            values = piece[choices, np.arange(n)]
            improves = values < v
            v[improves] = values[improves]
            owner[improves] = begin + choices[improves]
            ceiling = np.minimum(ceiling,
                                 (c[begin:end] + f[begin:end, None]).min(axis=0))
        candidate = np.zeros(m, dtype=bool)
        candidate[owner] = True
        candidate |= f == 0.0
        construction_stop = min(stop, started + 0.13 * usable)
        candidate, cost = _polish(f, c, candidate, construction_stop)
        if cost < upper:
            best, upper = candidate, cost
        best, upper = _polish(f, c, best, construction_stop)
        lower, saved_q = -np.inf, np.zeros(m)
        saved_v = v.copy()
        last_bound = time.perf_counter()
        last_recovery = started
        warm_stop = min(stop, started + 0.27 * usable)
        dual_seconds = max(1e-5, parsed)
        first_model = True
        optimizer_available = milp is not None
        last_model_seconds = max(0.001, usable / math.sqrt(m + n + 1))
        # The main search is limited only by the wall clock; it may stop early on reaching a valid lower bound or a proof of global optimality.
        while time.perf_counter() < stop:
            now = time.perf_counter()
            dual_result = _dual(f, c, v, stop)
            dual_seconds = max(1e-6, time.perf_counter() - now)
            if dual_result is None:
                break
            bound, q, gradient = dual_result
            if bound > lower:
                lower, saved_q, saved_v = bound, q.copy(), v.copy()
                last_bound = time.perf_counter()
            gap_tolerance = 3e-10 * max(1.0, upper)
            if upper - lower <= gap_tolerance:
                break
            norm = float(gradient @ gradient)
            if norm > 0.0:
                stale = (time.perf_counter() - last_bound) / max(0.08 * usable, dual_seconds)
                step = (1.8 / (1.0 + stale)) * max(0.0, upper - bound) / norm
                v = np.clip(v + step * gradient, 0.0, ceiling)
            else:
                v = np.clip(saved_v + rng.uniform(-1.0, 1.0, n)
                            * max(upper - lower, gap_tolerance) / n, 0.0, ceiling)
            now = time.perf_counter()
            recovery_period = max(dual_seconds * math.log1p(m + n),
                                  usable / math.sqrt((m + n + 1) * (1.0 + duration)))
            if now - last_recovery >= recovery_period:
                candidate = (q < 0.0) | (f == 0.0)
                if not candidate.any():
                    candidate[int(np.argmin(q))] = True
                repair_stop = min(stop, now + recovery_period * 0.35)
                candidate, cost = _polish(f, c, candidate, repair_stop)
                if cost < upper:
                    best, upper = candidate, cost
                last_recovery = time.perf_counter()
            if time.perf_counter() < warm_stop:
                continue
            remaining = stop - time.perf_counter()
            if remaining <= max(0.002, 3.0 * dual_seconds):
                continue
            # The estimated per-second processing scale and the total budget jointly set neighbourhood capacity; memory also has a guard limit.
            pair_budget = min(1500000, int(40000.0 * remaining
                              * math.sqrt(duration) / (1.0 + math.sqrt(duration))))
            width = max(1, pair_budget // n)
            if not first_model:
                width = max(1, int(width * rng.uniform(0.4, 1.0)))
            pool, outside, global_model = _neighborhood(
                f, c, best, saved_q, lower, upper, width, rng, first_model, stop)
            if time.perf_counter() >= stop:
                break
            if len(pool) == 0:
                if global_model:
                    break
                continue
            before_model = time.perf_counter()
            remaining = stop - before_model
            quantum = math.sqrt(max(0.0, usable * last_model_seconds))
            allowance = min(0.60 * remaining,
                            0.60 * remaining if first_model and global_model else quantum)
            candidate, proven = None, False
            if optimizer_available and allowance > max(0.001, dual_seconds):
                try:
                    candidate, proven = _ordered_model(
                        f, c, pool, outside, upper, stop, allowance)
                except Exception:
                    optimizer_available = False
            if candidate is None:
                # When the solver has no integer solution or is unavailable, still build a feasible plan within the same candidate facility pool.
                candidate = outside.copy()
                probability = min(0.9, max(1.0 / len(pool),
                                          np.count_nonzero(best[pool]) / len(pool)))
                candidate[pool[rng.random(len(pool)) < probability]] = True
                if not candidate.any():
                    candidate[pool[int(np.argmin(f[pool] + c[pool].sum(axis=1)))]] = True
            cost = _value(f, c, candidate)
            if cost < upper:
                best, upper = candidate.copy(), cost
            if proven and global_model:
                break
            last_model_seconds = max(dual_seconds,
                                     time.perf_counter() - before_model)
            if time.perf_counter() < stop:
                candidate, cost = _polish(f, c, candidate,
                                          min(stop, time.perf_counter() + 0.025 * usable))
                if cost < upper:
                    best, upper = candidate, cost
            first_model = False
        return _encode(c, best)
    except Exception:
        try:
            if best is not None and best.any():
                return _encode(c, best)
        except Exception:
            pass
        return _fallback(instance)
