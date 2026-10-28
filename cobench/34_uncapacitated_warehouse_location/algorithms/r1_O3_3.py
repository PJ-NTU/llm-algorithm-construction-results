"""Tabu search fused with incremental integer programming for facility location
Maintains nearest and second-nearest facilities, and keeps improving with add/drop/swap moves, tabu perturbation and integer-programming neighbourhoods anchored at the current best solution.
"""

import math
import time

import numpy as np

try:
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import coo_matrix
except Exception:
    milp = None


def _state(f, c, opened):
    ids = np.flatnonzero(opened)
    sub = c[ids]
    n = c.shape[1]
    ar = np.arange(n)
    p = np.argmin(sub, axis=0)
    owner = ids[p]
    d1 = sub[p, ar].copy()
    if len(ids) > 1:
        sub.partition(1, axis=0)
        d2 = sub[1].copy()
    else:
        d2 = np.full(n, np.inf)
    return (opened.copy(), owner, d1, d2,
            float(f[ids].sum() + d1.sum()))


def _moves(f, c, st, tabu, step, best_cost, until, block):
    """Compute exact add/drop/swap costs in chunks, while storing integer-programming candidate scores."""
    opened, owner, d1, d2, cost = st
    m, n = c.shape
    oi = np.flatnonzero(opened)
    ci = np.flatnonzero(~opened)
    eps = 1e-11 * max(1.0, abs(best_cost))
    blocked = tabu > step
    best = (np.inf, -1, -1)
    free = best
    score = np.full(m, np.inf)
    if len(oi) > 1:
        drop = np.bincount(owner, weights=d2 - d1, minlength=m) - f
        score[oi] = drop[oi]
        for allowed in (False, True):
            values = drop[oi].copy()
            if allowed:
                values[blocked[oi] & (cost + values >= best_cost - eps)] = np.inf
            p = int(np.argmin(values))
            mv = (float(values[p]), int(oi[p]), -1)
            if allowed:
                best = mv
            else:
                free = mv
    if not len(ci):
        return best if math.isfinite(best[0]) else free, score, True

    perm = np.argsort(owner, kind="stable")
    ordered = owner[perm]
    starts = np.flatnonzero(np.r_[True, ordered[1:] != ordered[:-1]])
    groups = ordered[starts]
    positions = np.searchsorted(oi, groups)
    gap = d2 - d1
    complete = True
    begin = 0
    while begin < len(ci):
        if time.perf_counter() >= until:
            complete = False
            break
        ids = ci[begin:begin + block]
        a = c[ids]
        adds = f[ids] - np.maximum(d1 - a, 0.0).sum(axis=1)
        values = adds.copy()
        p = int(np.argmin(values))
        if values[p] < free[0]:
            free = (float(values[p]), -1, int(ids[p]))
        values[blocked[ids] & (cost + values >= best_cost - eps)] = np.inf
        p = int(np.argmin(values))
        if values[p] < best[0]:
            best = (float(values[p]), -1, int(ids[p]))

        extra = np.maximum(a - d1, 0.0)
        np.minimum(extra, gap, out=extra)
        grouped = np.add.reduceat(extra[:, perm], starts, axis=1)
        swaps = adds[:, None] - f[oi][None, :]
        swaps[:, positions] += grouped
        score[ids] = np.minimum(adds, swaps.min(axis=1))
        p = int(np.argmin(swaps))
        r, k = divmod(p, len(oi))
        if swaps[r, k] < free[0]:
            free = (float(swaps[r, k]), int(oi[k]), int(ids[r]))
        forbidden = blocked[ids, None] | blocked[oi][None, :]
        swaps[forbidden & (cost + swaps >= best_cost - eps)] = np.inf
        p = int(np.argmin(swaps))
        r, k = divmod(p, len(oi))
        if swaps[r, k] < best[0]:
            best = (float(swaps[r, k]), int(oi[k]), int(ids[r]))
        begin += len(ids)
    return best if math.isfinite(best[0]) else free, score, complete


def _neighborhood(best, cur, score, frequency, count, rng):
    """Let swap scores, search trajectory and random exploration jointly decide the free facilities."""
    opened = best[0]
    m = len(opened)
    if count >= m:
        return np.arange(m)
    oi = np.flatnonzero(opened)
    ci = np.flatnonzero(~opened)
    # Release at least one open facility; the remaining slots balance closing and opening.
    no = min(len(oi), max(1, count // 2))
    if len(ci) < count - no:
        no = count - len(ci)
    nc = count - no
    chosen = []
    for pool, size in ((oi, no), (ci, nc)):
        if size <= 0:
            continue
        if size >= len(pool):
            chosen.extend(pool.tolist())
            continue
        vals = score[pool].copy()
        finite = np.isfinite(vals)
        scale = float(np.std(vals[finite])) if finite.any() else 1.0
        scale = max(scale, 1e-9)
        vals[~finite] = (float(np.max(vals[finite])) + scale
                         if finite.any() else 0.0)
        changed = cur[0][pool] != opened[pool]
        merit = vals - scale * (0.35 * changed + 0.15 * frequency[pool])
        deterministic = max(1, int(math.ceil(size * 0.7)))
        order = np.argsort(merit, kind="stable")
        take = order[:deterministic]
        rest = order[deterministic:]
        if len(take) < size:
            take = np.r_[take, rng.choice(rest, size=size - len(take), replace=False)]
        chosen.extend(pool[take].tolist())
    return np.asarray(sorted(chosen), dtype=np.int64)


def _integer_neighborhood(f, c, incumbent, free_ids, until):
    """Eliminate the original assignment variables, using facility flips and substitute assignments as variables; the zero vector is the existing solution."""
    if milp is None or time.perf_counter() >= until:
        return None, False
    opened, owner, base, _, base_cost = incumbent
    m, n = c.shape
    q = len(free_ids)
    local = np.full(m, -1, dtype=np.int64)
    local[free_ids] = np.arange(q)
    variable_owner = local[owner]
    fixed = opened.copy()
    fixed[free_ids] = False
    fixed_ids = np.flatnonzero(fixed)
    if len(fixed_ids):
        outside = c[fixed_ids].min(axis=0)
    else:
        outside = None
    sign = np.where(opened[free_ids], -1.0, 1.0)

    # Each substitute arc records the local facility index, customer index and cost relative to the original assignment.
    ais, ajs, costs = [], [], []
    k = 0
    block = max(1, 262144 // max(1, n))
    while k < q:
        if time.perf_counter() >= until:
            return None, False
        ids = free_ids[k:k + block]
        cc = c[ids]
        keep = ids[:, None] != owner[None, :]
        if outside is not None:
            keep &= cc < outside[None, :]
        rr, jj = np.nonzero(keep)
        ais.append(rr + k)
        ajs.append(jj)
        costs.append(cc[rr, jj] - base[jj])
        k += len(ids)
    ai = np.concatenate(ais)
    aj = np.concatenate(ajs)
    edge_cost = np.concatenate(costs)
    edges = len(ai)
    if outside is not None:
        backup = np.flatnonzero(variable_owner >= 0)
    else:
        backup = np.empty(0, dtype=np.int64)
    all_j = np.r_[aj, backup]
    all_cost = np.r_[edge_cost, outside[backup] - base[backup]
                     if outside is not None else np.empty(0)]
    nv = q + len(all_j)
    objective = np.r_[sign * f[free_ids], all_cost]
    magnitude = float(np.max(np.abs(objective)))
    if magnitude > 0:
        objective *= 1000.0 / magnitude

    # Rows: customer substitution upper bounds, substitution lower bounds when the original facility closes, and arc-facility linking.
    need = np.flatnonzero(variable_owner >= 0)
    need_row = np.full(n, -1, dtype=np.int64)
    need_row[need] = n + np.arange(len(need))
    link_start = n + len(need)
    rows = [all_j]
    cols = [q + np.arange(len(all_j))]
    vals = [np.ones(len(all_j))]
    relevant = need_row[all_j] >= 0
    rows.extend([need_row[all_j[relevant]], need_row[need]])
    cols.extend([q + np.flatnonzero(relevant), variable_owner[need]])
    vals.extend([-np.ones(int(relevant.sum())), np.ones(len(need))])
    link_rows = link_start + np.arange(edges)
    rows.extend([link_rows, link_rows])
    cols.extend([q + np.arange(edges), ai])
    vals.extend([np.ones(edges), -sign[ai]])
    upper = np.r_[np.ones(n), np.zeros(len(need)), opened[free_ids[ai]].astype(float)]
    lower = np.full(len(upper), -np.inf)
    if not len(fixed_ids):
        rows.append(np.full(q, len(upper), dtype=np.int64))
        cols.append(np.arange(q))
        vals.append(sign)
        lower = np.r_[lower, 1.0 - int(opened.sum())]
        upper = np.r_[upper, np.inf]
    nonzero = np.flatnonzero(objective)
    rows.append(np.full(len(nonzero), len(upper), dtype=np.int64))
    cols.append(nonzero)
    vals.append(objective[nonzero])
    lower = np.r_[lower, -np.inf]
    upper = np.r_[upper, 1e-8]
    if time.perf_counter() >= until:
        return None, False
    matrix = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                        shape=(len(upper), nv)).tocsc()
    integrality = np.zeros(nv, dtype=np.uint8)
    integrality[:q] = 1
    remaining = until - time.perf_counter()
    if remaining <= 0:
        return None, False
    result = milp(objective, integrality=integrality, bounds=Bounds(0.0, 1.0),
                  constraints=LinearConstraint(matrix, lower, upper),
                  options={"time_limit": max(0.001, 0.88 * remaining),
                           "mip_rel_gap": 1e-9, "disp": False})
    if result.x is None or not np.isfinite(result.x[:q]).all():
        return None, False
    new_open = opened.copy()
    flip = result.x[:q] > 0.5
    new_open[free_ids[flip]] = ~new_open[free_ids[flip]]
    if not new_open.any():
        return None, False
    return new_open, bool(result.status == 0 and q == m)


def _pack(opened, assignment, m, n):
    x = np.zeros((m, n), dtype=np.int8)
    x[assignment, np.arange(n)] = 1
    return {"y": opened.astype(np.int8).tolist(), "x": x.tolist()}


def solve(instance: dict, time_limit_s: float) -> dict:
    start = time.perf_counter()
    # Reject invalid input outright; do not fabricate costs or silently change the problem.
    try:
        if not isinstance(instance, dict):
            return {"y": [], "x": []}
        m, n = instance["m_facilities"], instance["n"]
        if (isinstance(m, bool) or isinstance(n, bool)
                or not isinstance(m, (int, np.integer))
                or not isinstance(n, (int, np.integer)) or m < 1 or n < 1):
            return {"y": [], "x": []}
        m, n = int(m), int(n)
        f = np.asarray(instance["fixed_cost"], dtype=np.float64)
        c = np.asarray(instance["alloc_cost"], dtype=np.float64)
        if (f.shape != (m,) or c.shape != (m, n)
                or not np.isfinite(f).all() or not np.isfinite(c).all()
                or (f < 0).any() or (c < 0).any()):
            return {"y": [], "x": []}
    except Exception:
        return {"y": [], "x": []}
    try:
        total = float(time_limit_s)
        if not math.isfinite(total) or total < 0:
            total = 0.0
    except Exception:
        total = 0.0

    # Save a complete feasible solution first; any later search failure only affects optimisation quality.
    best_open = np.zeros(m, dtype=bool)
    best_open[0] = True
    best_assignment = np.zeros(n, dtype=np.int64)
    try:
        margin = min(0.25 * total, 0.008 + 0.035 * total + 2e-7 * m * n)
        deadline = start + total - margin
        if time.perf_counter() >= deadline:
            return _pack(best_open, best_assignment, m, n)
        scale = max(float(f.max()), float(c.max()))
        if scale > 0:
            f = f / scale
            c = c / scale
        else:
            f = f.copy()
            c = c.copy()
        # A per-customer constant shift does not change the cost difference between any opening plans.
        c -= c.min(axis=0)
        single = f + c.sum(axis=1)
        b0 = int(np.argmin(single))
        best_open[:] = False
        best_open[b0] = True
        best_assignment[:] = b0
        if m == 1 or n == 1 or time.perf_counter() >= deadline:
            return _pack(best_open, best_assignment, m, n)
        near = np.argmin(c, axis=0)
        all_near = np.zeros(m, dtype=bool)
        all_near[near] = True
        if float(f[all_near].sum()) < float(single[b0]):
            best_open = all_near
            best_assignment = near.copy()
        cur = _state(f, c, best_open)
        best = cur
        best_cost = best[4]
        rng = np.random.default_rng(716231)
        tabu = np.zeros(m, dtype=np.int64)
        frequency = best_open.astype(float)
        score = np.full(m, np.inf)
        step = 0
        exploration = math.log1p(total)
        tenure = max(1, int(math.sqrt(m) * (1.0 + exploration) / 3.0))
        block = max(1, int(min(1048576, 65536 * (1.0 + exploration))) // max(1, n + m))
        last_improve = time.perf_counter()
        last_restart = last_improve
        period = total / (5.0 + exploration)
        next_mip = start + 0.16 * total
        full_attempted = False
        mip_enabled = milp is not None
        descent = True
        move_seconds = max(1e-6, 1e-8 * m * n)

        while time.perf_counter() < deadline:
            now = time.perf_counter()
            eps = 1e-11 * max(1.0, abs(best_cost))
            # Non-negative shifted assignment costs give a valid simple global lower bound.
            if best_cost <= float(f.min()) + eps:
                break
            remaining = deadline - now
            if (mip_enabled and now >= next_mip
                    and remaining > max(0.005, 0.02 * total)):
                # Model size and exact-neighbourhood duration both scale with available time instead of being hard-coded to the evaluation limit.
                preliminary_slice = 0.72 * remaining
                arc_budget = int(26000 * preliminary_slice ** 0.65 * (1.0 + exploration))
                count = min(m, max(2, arc_budget // max(1, n)))
                if full_attempted and m > 2:
                    count = min(count, max(2, int(math.ceil(m * 0.65))))
                q = _neighborhood(best, cur, score, frequency, count, rng)
                is_full = len(q) == m
                if is_full:
                    full_attempted = True
                slice_s = preliminary_slice if is_full else remaining / (2.0 + exploration)
                end = min(deadline, now + slice_s)
                try:
                    candidate, certified = _integer_neighborhood(f, c, best, q, end)
                    if candidate is not None:
                        new = _state(f, c, candidate)
                        if new[4] < best_cost - eps:
                            best, best_cost = new, new[4]
                            best_open, best_assignment = new[0].copy(), new[1].copy()
                            frequency = 0.9 * frequency + 0.1 * best_open
                            last_improve = time.perf_counter()
                        cur = best
                        tabu[:] = 0
                        descent = True
                        if certified:
                            break
                except Exception:
                    # When exact optimisation is unavailable, continue the same tabu search without discarding the existing solution.
                    mip_enabled = False
                next_mip = time.perf_counter() + max(period, 2.0 * move_seconds)
                continue

            stop_scan = deadline
            if mip_enabled and now < next_mip:
                stop_scan = min(stop_scan, next_mip)
            began = time.perf_counter()
            mv, new_score, complete = _moves(f, c, cur, tabu, step, best_cost,
                                            stop_scan, block)
            move_seconds = 0.8 * move_seconds + 0.2 * (time.perf_counter() - began)
            known = np.isfinite(new_score)
            score[known] = new_score[known]
            if not math.isfinite(mv[0]):
                if not complete:
                    continue
                tabu[:] = 0
                descent = False
                continue
            if descent and mv[0] >= -eps:
                descent = False
                tabu[:] = 0
            if time.perf_counter() >= deadline:
                break
            step += 1
            _, out_id, in_id = mv
            opened = cur[0].copy()
            duration = tenure + int(rng.integers(tenure + 1))
            if out_id >= 0:
                opened[out_id] = False
                tabu[out_id] = step + duration
            if in_id >= 0:
                opened[in_id] = True
                tabu[in_id] = step + duration
            cur = _state(f, c, opened)
            if cur[4] < best_cost - eps:
                best, best_cost = cur, cur[4]
                best_open, best_assignment = cur[0].copy(), cur[1].copy()
                frequency = 0.9 * frequency + 0.1 * best_open
                last_improve = time.perf_counter()
            elif time.perf_counter() - max(last_improve, last_restart) > max(period, 3 * move_seconds):
                # On stagnation, perturb from the historical best solution; perturbation strength grows with time budget and stagnation time.
                now = time.perf_counter()
                stagnation = (now - last_improve) / max(period, 1e-9)
                strength = min(m, max(1, int(math.sqrt(int(best_open.sum()))
                                  * (1.0 + exploration) * math.sqrt(1.0 + stagnation))))
                flip_count = int(rng.integers(1, strength + 1))
                opened = best_open.copy()
                flips = rng.choice(m, size=flip_count, replace=False)
                opened[flips] = ~opened[flips]
                if not opened.any():
                    opened[b0] = True
                cur = _state(f, c, opened)
                tabu[:] = 0
                descent = True
                last_restart = time.perf_counter()
                if cur[4] < best_cost - eps:
                    best, best_cost = cur, cur[4]
                    best_open, best_assignment = cur[0].copy(), cur[1].copy()
                    last_improve = last_restart
    except Exception:
        pass
    try:
        return _pack(best_open, best_assignment, m, n)
    except Exception:
        return {"y": [1] + [0] * (m - 1),
                "x": [[1] * n] + [[0] * n for _ in range(m - 1)]}
