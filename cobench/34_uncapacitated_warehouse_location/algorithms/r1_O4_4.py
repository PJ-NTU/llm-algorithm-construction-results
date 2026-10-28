"""Expanding integer programming and full-swap hybrid search
Builds a solution by full add/drop/swap descent, then alternates expanding sparse integer programs that keep the current assignment, adaptive large-neighbourhood programs and perturbed descent.
"""

import math
import numbers
import time

import numpy as np

try:
    from scipy.optimize import milp, LinearConstraint, Bounds
    from scipy import sparse
except Exception:
    milp = None


def _fallback(m, n):
    y = [0] * m
    x = [[0] * n for _ in range(m)]
    if m and n:
        y[0] = 1
        x[0] = [1] * n
    return {"y": y, "x": x}


def _state(f, c, mask, block):
    """Compute nearest and second-nearest facilities in chunks, and remove facilities that serve no customer."""
    m, n = c.shape
    opened = np.flatnonzero(mask)
    if opened.size == 0:
        opened = np.array([int(np.argmin(f + c.sum(axis=1)))])
    first = np.full(n, np.inf)
    second = first.copy()
    owner = np.zeros(n, dtype=np.int64)
    ar = np.arange(n)
    step = max(1, block // n)
    for start in range(0, len(opened), step):
        ids = opened[start:start + step]
        values = c[ids].copy()
        p = np.argmin(values, axis=0)
        v1 = values[p, ar].copy()
        values[p, ar] = np.inf
        v2 = values.min(axis=0)
        better = v1 < first
        second = np.where(better, np.minimum(first, v2),
                          np.minimum(second, v1))
        owner = np.where(better, ids[p], owner)
        first = np.minimum(first, v1)
    used = np.zeros(m, dtype=bool)
    used[owner] = True
    # After removing idle facilities the old second-nearest distances may be invalid, so recompute only when idle ones exist.
    if np.count_nonzero(used) != len(opened):
        return _state(f, c, used, block)
    return (used, opened, owner, first, second,
            float(f[opened].sum() + first.sum()))


def _descent(f, c, st, deadline, block):
    """Each round fully scans the add, drop and one-for-one swap neighbourhoods; check the wall clock between chunks."""
    m, n = c.shape
    step = max(1, block // n)
    while time.perf_counter() < deadline:
        mask, opened, owner, d1, d2, value = st
        gain = 0.0
        outgoing = incoming = -1
        if len(opened) > 1:
            drop = f - np.bincount(owner, weights=d2 - d1, minlength=m)
            k = int(opened[np.argmax(drop[opened])])
            if drop[k] > gain:
                gain, outgoing = float(drop[k]), k
            perm = np.argsort(owner, kind="stable")
            sorted_owner = owner[perm]
            starts = np.flatnonzero(np.r_[True, sorted_owner[1:] != sorted_owner[:-1]])
            groups = np.searchsorted(opened, sorted_owner[starts])
        closed = np.flatnonzero(~mask)
        scanned = True
        for start in range(0, len(closed), step):
            if time.perf_counter() >= deadline:
                scanned = False
                break
            ids = closed[start:start + step]
            rows = c[ids]
            add = np.maximum(d1 - rows, 0.0).sum(axis=1) - f[ids]
            a = int(np.argmax(add))
            if add[a] > gain:
                gain, outgoing, incoming = float(add[a]), -1, int(ids[a])
            if len(opened) == 1:
                swaps = value - f[ids] - rows.sum(axis=1)
                a = int(np.argmax(swaps))
                if swaps[a] > gain:
                    gain, outgoing, incoming = float(swaps[a]), int(opened[0]), int(ids[a])
            else:
                loss = np.minimum(rows, d2) - np.minimum(rows, d1)
                grouped = np.add.reduceat(loss[:, perm], starts, axis=1)
                swaps = add[:, None] + f[opened][None, :]
                swaps[:, groups] -= grouped
                a, b = np.unravel_index(int(np.argmax(swaps)), swaps.shape)
                if swaps[a, b] > gain:
                    gain, outgoing, incoming = float(swaps[a, b]), int(opened[b]), int(ids[a])
        eps = 1e-11 * max(1.0, abs(value))
        if gain <= eps:
            return st, scanned
        new_mask = mask.copy()
        if outgoing >= 0:
            new_mask[outgoing] = False
        if incoming >= 0:
            new_mask[incoming] = True
        candidate = _state(f, c, new_mask, block)
        if candidate[-1] >= value - eps:
            return st, False
        st = candidate
    return st, False


def _make_edges(f, c, ids, locked, st, rank, deadline, block, safe_cap):
    """Candidate arcs include nearest neighbours, nearest neighbours by amortised fixed cost, and arcs used by the current solution."""
    m, n = c.shape
    q = len(ids)
    has_locked = bool(locked.any())
    baseline = np.zeros(n)
    if has_locked:
        baseline[:] = np.inf
        step = max(1, block // n)
        locked_ids = np.flatnonzero(locked)
        for k in range(0, len(locked_ids), step):
            if time.perf_counter() >= deadline:
                return None
            baseline = np.minimum(baseline, c[locked_ids[k:k + step]].min(axis=0))
    constant = float(f[locked].sum() + baseline.sum())
    rank = min(q, max(1, rank))
    full = rank == q
    local = np.full(m, -1, dtype=np.int64)
    local[ids] = np.arange(q)
    parts = []
    customer_step = max(1, block // max(q, 1))
    share = max(1, len(st[1])) / max(1, n)
    for first in range(0, n, customer_step):
        if time.perf_counter() >= deadline:
            return None
        last = min(n, first + customer_step)
        customers = np.arange(first, last)
        costs = c[np.ix_(ids, customers)]
        if full:
            if has_locked:
                a, b = np.nonzero(costs < baseline[customers])
            else:
                a, b = np.nonzero(costs <= safe_cap[customers] + 1e-12)
            parts.append(a.astype(np.int64) * n + customers[b])
        else:
            chosen = np.argpartition(costs, rank - 1, axis=0)[:rank]
            parts.append((chosen * n + customers[None, :]).ravel())
            adjusted = costs + f[ids, None] * share
            r2 = max(1, rank // 3)
            chosen = np.argpartition(adjusted, r2 - 1, axis=0)[:r2]
            parts.append((chosen * n + customers[None, :]).ravel())
    # This term ensures that, with no fixed facilities, the current best solution is always representable in the model.
    own = local[st[2]]
    j = np.flatnonzero(own >= 0)
    parts.append(own[j] * n + j)
    if time.perf_counter() >= deadline:
        return None
    encoded = np.unique(np.concatenate(parts))
    facility = encoded // n
    customer = encoded % n
    coefficients = c[ids[facility], customer].copy()
    if has_locked:
        coefficients -= baseline[customer]
        keep = coefficients < 0.0
        facility, customer, coefficients = facility[keep], customer[keep], coefficients[keep]
    active, facility = np.unique(facility, return_inverse=True)
    ids = ids[active]
    return ids, facility, customer, coefficients, constant, has_locked, full


def _mip_search(f, c, ids, locked, st, rank, deadline, block, safe_cap):
    """Only opening variables are integer; final assignments are re-rounded to the nearest facility."""
    result = _make_edges(f, c, ids, locked, st, rank, deadline, block, safe_cap)
    if result is None:
        return None, False
    ids, facility, customer, costs, constant, has_locked, full = result
    n = c.shape[1]
    q, e = len(ids), len(costs)
    if e == 0:
        return (locked.copy(), False) if has_locked else (None, False)
    if deadline - time.perf_counter() <= 0.02:
        return None, False
    obj = np.concatenate((f[ids], costs))
    nv = q + e
    ar = np.arange(e, dtype=np.int64)
    nonzero = np.flatnonzero(obj)
    rows = np.concatenate((customer, n + ar, n + ar,
                           np.full(len(nonzero), n + e, dtype=np.int64)))
    cols = np.concatenate((q + ar, q + ar, facility, nonzero))
    values = np.concatenate((np.ones(2 * e), -np.ones(e), obj[nonzero]))
    matrix = sparse.coo_matrix((values, (rows, cols)),
                               shape=(n + e + 1, nv)).tocsc()
    lower = np.full(n + e + 1, -np.inf)
    upper = np.zeros(n + e + 1)
    upper[:n] = 1.0
    if not has_locked:
        lower[:n] = 1.0
    # The current upper bound restricts the search region without changing the optimum within it.
    upper[-1] = st[-1] - constant + 1e-8 * max(1.0, abs(st[-1]))
    integrality = np.zeros(nv, dtype=np.uint8)
    integrality[:q] = 1
    remaining = deadline - time.perf_counter()
    if remaining <= 0.025:
        return None, False
    res = milp(obj, integrality=integrality,
               bounds=Bounds(0.0, 1.0),
               constraints=LinearConstraint(matrix, lower, upper),
               options={"time_limit": 0.92 * remaining,
                        "mip_rel_gap": 1e-9, "disp": False})
    if res.x is None:
        return None, False
    y = locked.copy()
    y[ids] = np.asarray(res.x[:q]) > 0.5
    if not y.any():
        return None, False
    # Only an unlocked model containing all facilities and all safe arcs may declare global convergence.
    proven = bool(res.status == 0 and full and not has_locked)
    return y, proven


def _neighborhood(f, c, st, rng, width, block, deadline):
    """Keep some facilities as assignment backup; release the rest along with promising substitute facilities."""
    m, n = c.shape
    k = len(st[1])
    free_count = min(k, max(1, int(math.ceil(math.sqrt(k) * width))))
    released = rng.choice(st[1], free_count, replace=False)
    locked = st[0].copy()
    locked[released] = False
    if not locked.any():
        return np.arange(m), locked
    baseline = _state(f, c, locked, block)[3]
    score = np.full(m, -np.inf)
    step = max(1, block // n)
    for first in range(0, m, step):
        if time.perf_counter() >= deadline:
            break
        last = min(m, first + step)
        score[first:last] = np.maximum(baseline - c[first:last], 0.0).sum(axis=1) - f[first:last]
    score[locked] = -np.inf
    count = min(m - int(locked.sum()),
                max(free_count, int(math.ceil(free_count * width + math.sqrt(m) * width))))
    chosen = np.argpartition(-score, count - 1)[:count]
    random_count = min(m, max(1, int(math.ceil(math.sqrt(count) * width))))
    sampled = rng.choice(m, random_count, replace=False)
    ids = np.unique(np.concatenate((released, chosen, sampled)))
    ids = ids[~locked[ids]]
    return ids, locked


def _perturb(st, best, rng, width, age):
    base = best if rng.random() < 0.65 else st
    mask = base[0].copy()
    m = len(mask)
    strength = min(m, max(1, int(math.ceil(width * (1.0 + math.log1p(age))))))
    count = int(rng.integers(1, strength + 1))
    change = rng.choice(m, count, replace=False)
    mask[change] = ~mask[change]
    if not mask.any():
        mask[int(rng.integers(m))] = True
    return mask


def _build(c, st):
    m, n = c.shape
    y = st[0].astype(int).tolist()
    x = [[0] * n for _ in range(m)]
    for j, i in enumerate(st[2]):
        x[int(i)][j] = 1
    return {"y": y, "x": x}


def solve(instance: dict, time_limit_s: float) -> dict:
    start = time.perf_counter()
    m = n = 0
    best = None
    c = None
    validated = False
    try:
        if not isinstance(instance, dict):
            return {"y": [], "x": []}
        m, n = instance.get("m_facilities"), instance.get("n")
        if (not isinstance(m, numbers.Integral) or isinstance(m, bool) or m < 1
                or not isinstance(n, numbers.Integral) or isinstance(n, bool) or n < 1):
            return {"y": [], "x": []}
        m, n = int(m), int(n)
        f = np.asarray(instance.get("fixed_cost"))
        c = np.asarray(instance.get("alloc_cost"))
        if (f.shape != (m,) or c.shape != (m, n)
                or f.dtype.kind not in "iuf" or c.dtype.kind not in "iuf"):
            return {"y": [], "x": []}
        f = f.astype(float, copy=True)
        c = c.astype(float, copy=True)
        if (not np.isfinite(f).all() or not np.isfinite(c).all()
                or (f < 0).any() or (c < 0).any()):
            return {"y": [], "x": []}
        validated = True
        try:
            budget = float(time_limit_s)
            if not math.isfinite(budget):
                budget = 0.0
            budget = max(0.0, budget)
        except Exception:
            budget = 0.0
        if budget <= time.perf_counter() - start:
            return _fallback(m, n)
        reserve = min(0.30 * budget, 0.025 + 0.055 * budget + 1.5e-7 * m * n)
        deadline = start + budget - reserve
        # Remove the constant common to all customers, then rescale, so large constants do not mask opening costs.
        c -= c.min(axis=0)
        row_bound = np.maximum(f, c.max(axis=1))
        i0 = int(np.argmin(row_bound))
        scale = float(row_bound[i0])
        if scale == 0.0:
            y = [0] * m
            y[i0] = 1
            x = [[0] * n for _ in range(m)]
            x[i0] = [1] * n
            return {"y": y, "x": x}
        # A single-facility solution costs at most (n + 1) * scale; truncating larger single terms
        # does not affect any plan better than it, and keeps irrelevant huge costs from ruining precision.
        cap = scale * (n + 2)
        np.minimum(f, cap, out=f)
        np.minimum(c, cap, out=c)
        f /= scale
        c /= scale
        block = max(n, int(min(2000000, 180000 * (1.0 + math.sqrt(budget)))))
        mask = np.zeros(m, dtype=bool)
        mask[int(np.argmin(f + c.sum(axis=1)))] = True
        best = _state(f, c, mask, block)
        if m == 1 or n == 1 or not np.any(c):
            return _build(c, best)
        if not np.any(f):
            best = _state(f, c, np.ones(m, dtype=bool), block)
            return _build(c, best)
        initial_end = min(deadline, start + 0.12 * budget)
        best, _ = _descent(f, c, best, initial_end, block)
        current = best
        rng = np.random.default_rng(15485863)
        width = 1.0 + math.log1p(budget) / 2.0
        # Model size and candidate width both grow with the time limit; the search is not truncated by restart count.
        edge_budget = max(n, int(16000 * (1.0 + budget) ** 0.8))
        max_rank = min(m, max(1, edge_budget // max(n, 1)))
        rank = min(max_rank, max(2, int(math.ceil(math.log2(m + 1) * width))))
        if m * n <= edge_budget:
            rank = m
        safe_cap = np.full(n, np.inf)
        step = max(1, block // n)
        for first in range(0, m, step):
            if time.perf_counter() >= deadline:
                return _build(c, best)
            last = min(m, first + step)
            safe_cap = np.minimum(safe_cap, (c[first:last] + f[first:last, None]).min(axis=0))
        age = 0
        global_turn = True
        first_mip = True
        mip_enabled = milp is not None and n <= int(16000 * (1.0 + budget) ** 0.8)
        while time.perf_counter() < deadline:
            now = time.perf_counter()
            remaining = deadline - now
            old_value = best[-1]
            if mip_enabled and remaining > max(0.06, 0.018 * budget):
                if global_turn:
                    ids = np.arange(m)
                    locked = np.zeros(m, dtype=bool)
                    model_rank = rank
                    share = 0.80 if rank == m else 0.40
                else:
                    ids, locked = _neighborhood(f, c, best, rng,
                                                width * (1.0 + 0.15 * math.log1p(age)),
                                                block, min(deadline, now + 0.12 * remaining))
                    model_rank = min(len(ids), max_rank)
                    share = 0.40
                phase_end = min(deadline, now + share * remaining)
                try:
                    y, proven = _mip_search(f, c, ids, locked, best, model_rank,
                                            phase_end, block, safe_cap)
                except Exception:
                    y, proven = None, False
                    mip_enabled = False
                if y is not None:
                    candidate = _state(f, c, y, block)
                    if candidate[-1] < best[-1]:
                        best = candidate
                        current = candidate
                    if proven and len(ids) == m:
                        return _build(c, best)
                    polish_end = min(deadline, time.perf_counter() + 0.05 * budget)
                    candidate, _ = _descent(f, c, candidate, polish_end, block)
                    if candidate[-1] < best[-1]:
                        best = candidate
                        current = candidate
                if global_turn:
                    previous_rank = rank
                    rank = min(max_rank, max(rank + 1, int(math.ceil(rank * width))))
                    # After the first sparse model, prefer expansion; once the model budget is reached, alternate with large neighbourhoods.
                    global_turn = first_mip and previous_rank < max_rank
                else:
                    global_turn = True
                first_mip = False
            if time.perf_counter() >= deadline:
                break
            mask = _perturb(current, best, rng, width, age)
            candidate = _state(f, c, mask, block)
            local_end = min(deadline, time.perf_counter() + 0.07 * budget)
            candidate, _ = _descent(f, c, candidate, local_end, block)
            if candidate[-1] < best[-1]:
                best = candidate
            if candidate[-1] <= current[-1] or rng.random() < 1.0 / (2.0 + width):
                current = candidate
            if best[-1] < old_value - 1e-11 * max(1.0, abs(old_value)):
                age = 0
            else:
                age += 1
        return _build(c, best)
    except Exception:
        if best is not None and c is not None:
            try:
                return _build(c, best)
            except Exception:
                pass
        try:
            if validated and isinstance(m, numbers.Integral) and isinstance(n, numbers.Integral) and m > 0 and n > 0:
                return _fallback(int(m), int(n))
        except Exception:
            pass
        return {"y": [], "x": []}
