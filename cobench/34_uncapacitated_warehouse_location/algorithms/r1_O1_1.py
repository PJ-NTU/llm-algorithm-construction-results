"""Dual screening and adaptive multi-facility neighbourhood search
Combines add/drop/swap local search, Lagrangian lower-bound screening, progressively expanding sparse integer programs and targeted perturbation, keeping the best feasible solution within the wall-clock budget.
"""

import math
import numbers
import time

import numpy as np

try:
    from scipy.optimize import milp as _milp, LinearConstraint as _LC, Bounds as _Bounds
    from scipy import sparse as _sparse
    _HAVE_MILP = True
except Exception:
    _HAVE_MILP = False


def _parse(instance):
    m, n = instance["m_facilities"], instance["n"]
    if (isinstance(m, bool) or isinstance(n, bool)
            or not isinstance(m, numbers.Integral)
            or not isinstance(n, numbers.Integral) or m < 1 or n < 1):
        return None
    f = np.asarray(instance["fixed_cost"])
    c = np.asarray(instance["alloc_cost"])
    if (f.shape != (m,) or c.shape != (m, n)
            or f.dtype.kind not in "iuf" or c.dtype.kind not in "iuf"):
        return None
    f = f.astype(float, copy=True)
    c = c.astype(float, copy=True)
    if (not np.isfinite(f).all() or not np.isfinite(c).all()
            or (f < 0).any() or (c < 0).any()):
        return None
    # Remove the mandatory cost of each column, then rescale uniformly to avoid overflow in sums.
    c -= c.min(axis=0)
    scale = max(float(f.max()), float(c.max()))
    if scale > 0:
        f /= scale
        c /= scale
    return f, c


def _build(c, opened):
    m, n = c.shape
    ids = np.flatnonzero(opened)
    if not ids.size:
        ids = np.array([0])
    owner = ids[np.argmin(c[ids], axis=0)]
    # With non-negative fixed costs, removing idle facilities never worsens the objective.
    y = [0] * m
    x = [[0] * n for _ in range(m)]
    for j, i in enumerate(owner.tolist()):
        y[i] = 1
        x[i][j] = 1
    return {"y": y, "x": x}


class _State:
    def __init__(self, f, c, opened):
        self.f, self.c = f, c
        self.m, self.n = c.shape
        self.opened = opened.copy()
        self.ids = np.flatnonzero(opened)
        sub = c[self.ids]
        am = np.argmin(sub, axis=0)
        self.owner = self.ids[am]
        self.first = sub[am, np.arange(self.n)]
        if len(self.ids) == 1:
            self.second = None
        else:
            self.second = np.partition(sub, 1, axis=0)[1]
        self.cost = float(f[self.ids].sum() + self.first.sum())

    def move(self, deadline, anchor=-1):
        # Return gain, facility moved out, facility moved in; -1 means the corresponding operation is absent.
        best = (0.0, -1, -1)
        eligible = self.ids[self.ids != anchor]
        if self.second is not None:
            drop = self.f - np.bincount(
                self.owner, weights=self.second - self.first, minlength=self.m)
            if eligible.size:
                i = int(eligible[np.argmax(drop[eligible])])
                if drop[i] > best[0]:
                    best = (float(drop[i]), i, -1)
            order = np.argsort(self.owner, kind="stable")
            owners = self.owner[order]
            starts = np.flatnonzero(np.r_[True, owners[1:] != owners[:-1]])
            owners = owners[starts]
            pos = np.full(self.m, -1, dtype=int)
            pos[self.ids] = np.arange(len(self.ids))
        else:
            drop = None
        closed = np.flatnonzero(~self.opened)
        # Chunk by memory and check the clock between chunks, without building the whole swap tensor.
        block = max(1, min(len(closed), 180000 // max(1, self.n + len(self.ids))))
        for start in range(0, len(closed), block):
            if time.perf_counter() >= deadline:
                return best, False
            ids = closed[start:start + block]
            cc = self.c[ids]
            gain = np.maximum(self.first - cc, 0.0).sum(axis=1) - self.f[ids]
            k = int(np.argmax(gain))
            if gain[k] > best[0]:
                best = (float(gain[k]), -1, int(ids[k]))
            if drop is not None and eligible.size:
                v = self.second - np.maximum(cc, self.first)
                np.maximum(v, 0.0, out=v)
                ext = np.add.reduceat(v[:, order], starts, axis=1)
                swaps = gain[:, None] + drop[self.ids]
                swaps[:, pos[owners]] += ext
                if anchor >= 0 and self.opened[anchor]:
                    swaps[:, pos[anchor]] = -np.inf
                flat = int(np.argmax(swaps))
                row, col = divmod(flat, len(self.ids))
                if swaps[row, col] > best[0]:
                    best = (float(swaps[row, col]), int(self.ids[col]), int(ids[row]))
            elif len(self.ids) == 1 and self.ids[0] != anchor:
                gains = self.cost - (self.f[ids] + cc.sum(axis=1))
                k = int(np.argmax(gains))
                if gains[k] > best[0]:
                    best = (float(gains[k]), int(self.ids[0]), int(ids[k]))
        return best, True


def _descent(st, deadline, anchor=-1):
    while time.perf_counter() < deadline:
        move, complete = st.move(deadline, anchor)
        gain, out, into = move
        tol = 2e-12 * max(1.0, st.cost)
        if gain <= tol:
            return st
        opened = st.opened.copy()
        if out >= 0:
            opened[out] = False
        if into >= 0:
            opened[into] = True
        cand = _State(st.f, st.c, opened)
        if cand.cost >= st.cost - tol:
            return st
        st = cand
        if not complete:
            return st
    return st


def _dual(st, deadline):
    """Relax customer equality constraints; every returned lower bound comes from an actually evaluated dual function."""
    f, c = st.f, st.c
    counts = np.bincount(st.owner, minlength=st.m)
    alpha = st.first + f[st.owner] / counts[st.owner]
    best_lb = 0.0
    best_alpha = np.zeros(st.n)
    best_red = f.copy()
    start = time.perf_counter()
    span = max(deadline - start, 1e-6)
    last_gain = start
    while time.perf_counter() < deadline:
        red = f - np.maximum(alpha - c, 0.0).sum(axis=1)
        lb = float(alpha.sum() + np.minimum(red, 0.0).sum())
        now = time.perf_counter()
        if lb > best_lb:
            best_lb, best_alpha, best_red = lb, alpha.copy(), red.copy()
            last_gain = now
        gap = max(0.0, st.cost - lb)
        if st.cost - best_lb <= 1e-11 * max(1.0, st.cost):
            break
        ids = np.flatnonzero(red < 0.0)
        if ids.size:
            gradient = 1.0 - (c[ids] < alpha).sum(axis=0)
        else:
            gradient = np.ones(st.n)
        norm = float(np.dot(gradient, gradient))
        if norm == 0.0:
            break
        # Polyak step size decays with wall-clock stagnation time; the search is not ended by an iteration count.
        factor = 0.12 + 1.65 * math.exp(-5.0 * (now - last_gain) / span)
        alpha = np.maximum(0.0, alpha + factor * gap / norm * gradient)
        np.minimum(alpha, max(st.cost, 1e-15), out=alpha)
    return best_lb, best_alpha, best_red


def _mip_search(st, dual, deadline, seconds, coverage, radius, strength):
    """Sparse assignment model; global optimality is preserved when it fully covers the safely screened arcs."""
    f, c = st.f, st.c
    m, n = c.shape
    now = time.perf_counter()
    finish = min(deadline, now + seconds)
    # Model capacity grows with the current time slice; overly large customer dimensions are left to vectorised neighbourhood search.
    edge_budget = max(1, int(42000 * max(seconds, 1e-4) ** 0.75))
    if n > edge_budget or finish - now <= 0.01:
        return None, False, coverage
    k = min(m, max(1, edge_budget // n),
            max(1, int(math.ceil(math.sqrt(m) * strength * coverage))))
    lb, alpha, red = dual
    safety = 2e-8 * max(1.0, st.cost)
    gap = max(0.0, st.cost - lb) + safety
    positive_red = np.maximum(red, 0.0)
    facilities, customers = [], []
    complete = True
    block = max(1, min(n, 180000 // max(1, m)))
    build_stop = min(finish, now + 0.4 * seconds)
    amortization = len(st.ids) / max(1, n)
    for begin in range(0, n, block):
        if time.perf_counter() >= build_stop:
            return None, False, coverage
        end = min(n, begin + block)
        cc = c[:, begin:end]
        # The increment from forcing x_ij=1 in the Lagrangian relaxation gives a safe lower bound.
        valid = (positive_red[:, None]
                 + np.maximum(cc - alpha[begin:end], 0.0) <= gap)
        valid[st.owner[begin:end], np.arange(end - begin)] = True
        if k >= m or int(valid.sum(axis=0).max()) <= k:
            chosen = valid
        else:
            # Select arcs by both assignment cost and amortised fixed cost, to avoid favouring only expensive facilities.
            chosen = np.zeros_like(valid)
            raw_count = max(1, (k + 1) // 2)
            adjusted_count = k - raw_count
            scores = np.where(valid, cc, np.inf)
            top = np.argpartition(scores, raw_count - 1, axis=0)[:raw_count]
            chosen[top, np.arange(end - begin)] = True
            if adjusted_count:
                scores = np.where(valid, cc + f[:, None] * amortization, np.inf)
                top = np.argpartition(scores, adjusted_count - 1, axis=0)[:adjusted_count]
                chosen[top, np.arange(end - begin)] = True
            chosen &= valid
            chosen[st.owner[begin:end], np.arange(end - begin)] = True
            complete = complete and not bool((valid & ~chosen).any())
        ii, jj = np.nonzero(chosen)
        facilities.append(ii)
        customers.append(jj + begin)
    fi = np.concatenate(facilities)
    cj = np.concatenate(customers)
    e = len(fi)
    var = m + e
    obj = np.concatenate((f, c[fi, cj]))
    ar = np.arange(e)
    rows = [cj, n + ar, n + ar, np.full(var, n + e, dtype=int)]
    cols = [m + ar, m + ar, fi, np.arange(var)]
    data = [np.ones(e), np.ones(e), -np.ones(e), obj]
    lower = np.r_[np.ones(n), np.full(e + 1, -np.inf)]
    upper = np.r_[np.ones(n), np.zeros(e), st.cost + safety]
    nr = n + e + 1
    if radius is not None:
        rows.append(np.full(m, nr, dtype=int))
        cols.append(np.arange(m))
        data.append(np.where(st.opened, -1.0, 1.0))
        lower = np.r_[lower, -np.inf]
        upper = np.r_[upper, radius - len(st.ids)]
        nr += 1
    matrix = _sparse.coo_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
        shape=(nr, var)).tocsc()
    remain = finish - time.perf_counter()
    if remain <= 0.01:
        return None, False, coverage
    integrality = np.zeros(var, dtype=np.uint8)
    integrality[:m] = 1
    result = _milp(obj, integrality=integrality, bounds=_Bounds(0.0, 1.0),
                   constraints=_LC(matrix, lower, upper),
                   options={"time_limit": max(0.001, 0.88 * remain),
                            "mip_rel_gap": 1e-10, "disp": False})
    opened = None
    if result.x is not None:
        opened = np.asarray(result.x[:m]) > 0.5
        if not opened.any():
            opened = None
    proven = bool(complete and radius is None and result.status == 0
                  and opened is not None)
    if radius is None and not complete:
        coverage *= 1.0 + 0.35 * strength
    return opened, proven, coverage


def _perturb(st, best, elite, rng, strength, stagnation, deadline):
    m = st.m
    if rng.random() < 0.25:
        st = best
    opened = st.opened.copy()
    cap = min(m, max(2, int(math.ceil(math.sqrt(len(st.ids)) * strength))))
    depth = max(1, min(cap, 1 + int(stagnation * cap)))
    count = int(rng.integers(1, depth + 1))
    mode = rng.random()
    anchor = -1
    if mode < 0.22 and len(elite) > 1:
        other = elite[int(rng.integers(len(elite)))].opened
        differing = np.flatnonzero(opened != other)
        if differing.size:
            chosen = differing[rng.random(len(differing)) < 0.5]
            opened[chosen] = other[chosen]
    elif mode < 0.64:
        closed = np.flatnonzero(~opened)
        if closed.size:
            scores = np.empty(len(closed))
            block = max(1, 160000 // max(1, st.n))
            for start in range(0, len(closed), block):
                if time.perf_counter() >= deadline:
                    return opened, -1
                ids = closed[start:start + block]
                scores[start:start + block] = (
                    np.maximum(st.first - st.c[ids], 0.0).sum(axis=1) - st.f[ids])
            width = min(len(closed), max(1, int(math.ceil(strength * depth))))
            short = np.argpartition(-scores, width - 1)[:width]
            anchor = int(closed[int(rng.choice(short))])
            opened[anchor] = True
            # Targeted destruction: prefer closing customer clusters that a new facility can cheaply take over.
            losses = np.bincount(
                st.owner, weights=np.maximum(st.c[anchor] - st.first, 0.0),
                minlength=m) - st.f
            ids = st.ids
            noise = max(float(np.std(losses[ids])), 1e-12)
            priority = losses[ids] + rng.gumbel(size=len(ids)) * noise * 0.25
            take = min(count, len(ids))
            removed = ids[np.argpartition(priority, take - 1)[:take]]
            opened[removed] = False
        elif len(st.ids) > 1:
            take = min(count, len(st.ids) - 1)
            opened[rng.choice(st.ids, size=take, replace=False)] = False
    else:
        # Combine wide-range destruction with local targeted destruction; restart strength is set by time budget and stagnation time.
        take = min(count, m)
        flip = rng.choice(m, size=take, replace=False)
        opened[flip] = ~opened[flip]
    if not opened.any():
        opened[int(rng.choice(st.ids))] = True
    return opened, anchor


def solve(instance: dict, time_limit_s: float) -> dict:
    started = time.perf_counter()
    try:
        parsed = _parse(instance)
    except Exception:
        parsed = None
    # Reject invalid input outright instead of modifying the data and continuing; empty fields are a rejection, not a feasible solution.
    if parsed is None:
        return {"y": [], "x": []}
    f, c = parsed
    m, n = c.shape
    best = None
    opened = np.zeros(m, dtype=bool)
    opened[0] = True
    try:
        try:
            budget = float(time_limit_s)
            if not math.isfinite(budget):
                budget = 0.0
            budget = max(0.0, budget)
        except Exception:
            budget = 0.0
        reserve = min(0.3 * budget, 0.035 * budget + 3e-7 * m * n + 0.002)
        deadline = started + budget - reserve
        strength = 1.0 + math.log1p(budget)
        rng = np.random.default_rng(918273)
        single = f + c.sum(axis=1)
        opened[:] = False
        opened[int(np.argmin(single))] = True
        best = _State(f, c, opened)
        if m == 1 or n == 1 or best.cost == 0.0 or time.perf_counter() >= deadline:
            return _build(c, best.opened)
        best = _descent(best, min(deadline, started + 0.16 * budget))
        current = best
        dual = _dual(best, min(deadline, time.perf_counter() + 0.075 * budget))
        if best.cost - dual[0] <= 1e-11 * max(1.0, best.cost):
            return _build(c, best.opened)
        lag_open = dual[2] < 0.0
        if lag_open.any() and time.perf_counter() < deadline:
            cand = _State(f, c, lag_open)
            cand = _descent(cand, min(deadline, time.perf_counter() + 0.035 * budget))
            if cand.cost < best.cost:
                best, current = cand, cand
        elite = [best]
        elite_cap = max(2, int(math.ceil(strength * math.sqrt(max(1, m)))))
        coverage = 1.0
        local_mip = False
        next_mip = time.perf_counter()
        last_improvement = time.perf_counter()
        mip_enabled = _HAVE_MILP
        while time.perf_counter() < deadline:
            now = time.perf_counter()
            remain = deadline - now
            stagnation = min(1.0, (now - last_improvement) / max(0.12 * budget, 1e-9))
            improved = False
            if mip_enabled and now >= next_mip and remain > 0.025 * budget:
                slot = min(0.44 * remain, budget * (0.10 + 0.11 * stagnation))
                radius = None
                if local_mip:
                    radius = min(m, max(2, int(math.ceil(
                        math.sqrt(len(best.ids)) * strength * (0.5 + stagnation)))))
                try:
                    yo, proven, coverage = _mip_search(
                        best, dual, deadline, slot, coverage, radius, strength)
                except Exception:
                    yo, proven = None, False
                    mip_enabled = False
                spent = time.perf_counter() - now
                local_mip = not local_mip
                next_mip = time.perf_counter() + max(0.025 * budget, 0.22 * spent)
                if yo is not None:
                    cand = _State(f, c, yo)
                    if cand.cost < best.cost:
                        best, current = cand, cand
                        improved = True
                    if proven:
                        break
                    cand = _descent(cand, min(deadline, time.perf_counter() + 0.045 * budget))
                    if cand.cost < best.cost:
                        best, current = cand, cand
                        improved = True
            else:
                yo, anchor = _perturb(current, best, elite, rng, strength, stagnation, deadline)
                if time.perf_counter() >= deadline:
                    break
                cand = _State(f, c, yo)
                slice_end = min(deadline, time.perf_counter() + budget * (0.025 + 0.06 * stagnation))
                if anchor >= 0:
                    # Temporarily protect the new facility, allow closing several old facilities in a row, then release the protection.
                    cand = _descent(cand, min(slice_end, time.perf_counter() + 0.4 * (slice_end - time.perf_counter())), anchor)
                cand = _descent(cand, slice_end)
                if cand.cost < best.cost:
                    best = cand
                    improved = True
                temperature = max(1e-12, 0.012 * max(best.cost, 1e-6) * stagnation / strength)
                if (cand.cost <= current.cost
                        or rng.random() < math.exp(-min(700.0, (cand.cost - current.cost) / temperature))):
                    current = cand
                if not any(np.array_equal(cand.opened, item.opened) for item in elite):
                    elite.append(cand)
                    elite.sort(key=lambda item: item.cost)
                    if len(elite) > elite_cap:
                        elite.pop()
                if rng.random() < 0.1 * stagnation:
                    current = elite[int(rng.integers(len(elite)))]
            if improved:
                last_improvement = time.perf_counter()
                if not any(np.array_equal(best.opened, item.opened) for item in elite):
                    elite.append(best)
                    elite.sort(key=lambda item: item.cost)
                    if len(elite) > elite_cap:
                        elite.pop()
            if best.cost - dual[0] <= 1e-11 * max(1.0, best.cost):
                break
        return _build(c, best.opened)
    except Exception:
        try:
            return _build(c, best.opened if best is not None else opened)
        except Exception:
            y = [0] * m
            x = [[0] * n for _ in range(m)]
            y[0] = 1
            x[0] = [1] * n
            return {"y": y, "x": x}
