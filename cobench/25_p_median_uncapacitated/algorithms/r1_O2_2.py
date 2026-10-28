"""Adaptive joint median re-selection algorithm
Guarantees coverage via sink strongly connected components and combines greedy construction, intra-cluster median updates, two-facility enumeration and a distance-level covering integer program for time-limited large-neighbourhood improvement.
"""

import math
import time
import itertools
import numpy as np

try:
    from scipy.optimize import milp, Bounds, LinearConstraint
    from scipy.sparse import coo_matrix
except Exception:
    milp = None


def _plain_feasible(raw, n, p):
    """Fallback using only the reachability relation; for valid input the number of sink components does not exceed p."""
    unseen = set(range(n))
    centers = []
    while unseen:
        v = next(iter(unseen))
        row = raw[v]
        component = [j for j in unseen
                     if row[j] is not None and raw[j][v] is not None]
        unseen.difference_update(component)
        reach = sum(x is not None for x in row)
        if reach == len(component):
            centers.append(v)
    taken = set(centers)
    for j in range(n):
        if len(centers) == p:
            break
        if j not in taken:
            centers.append(j)
    return centers


def _sinks(reach):
    n = len(reach)
    labels = np.full(n, -1, dtype=np.int64)
    if bool(reach.all()):
        return labels, []
    unseen = np.ones(n, dtype=bool)
    groups = []
    while unseen.any():
        v = int(np.flatnonzero(unseen)[0])
        component = np.flatnonzero(reach[v] & reach[:, v])
        unseen[component] = False
        if int(np.count_nonzero(reach[v])) == len(component):
            labels[component] = len(groups)
            groups.append(component)
    return labels, groups


def _cost(A, centers):
    return np.sum(np.min(A[:, centers], axis=1), dtype=np.longdouble)


def _remember(A, centers, p, saved):
    if len(centers) != p or len(set(int(j) for j in centers)) != p:
        return False
    c = _cost(A, centers)
    if np.isfinite(c) and c < saved[1]:
        saved[0] = [int(j) for j in centers]
        saved[1] = c
        return True
    return False


def _greedy(D, seeds, p, stop):
    n = len(D)
    centers = list(seeds)
    used = np.zeros(n, dtype=bool)
    used[centers] = True
    if centers:
        nearest = np.min(D[:, centers], axis=1)
    else:
        j = int(np.argmin(np.sum(D, axis=0, dtype=np.longdouble)))
        centers.append(j)
        used[j] = True
        nearest = D[:, j].copy()
    block = max(1, min(n, 524288 // max(n, 1)))
    while len(centers) < p and time.perf_counter() < stop:
        winner = -1
        gain = -1.0
        for a in range(0, n, block):
            if time.perf_counter() >= stop:
                break
            b = min(n, a + block)
            values = np.maximum(nearest[:, None] - D[:, a:b], 0.0).sum(axis=0)
            values[used[a:b]] = -1.0
            k = int(np.argmax(values))
            if values[k] > gain:
                gain = float(values[k])
                winner = a + k
        if winner < 0:
            break
        centers.append(winner)
        used[winner] = True
        np.minimum(nearest, D[:, winner], out=nearest)
    if len(centers) < p:
        priority = nearest.copy()
        priority[used] = -np.inf
        extra = np.argsort(priority)[::-1][:p - len(centers)]
        centers.extend(int(j) for j in extra)
    return centers


def _partition(D, centers):
    distances = D[:, centers]
    assigned = np.argmin(distances, axis=1)
    assigned[np.asarray(centers)] = np.arange(len(centers))
    nearest = distances[np.arange(len(D)), assigned]
    return assigned, nearest


def _relocate(D, A, centers, saved, stop):
    """Each cluster picks its new centre only within itself, keeping centres distinct and intra-cluster service finite."""
    centers = list(centers)
    while time.perf_counter() < stop:
        assigned, unused = _partition(D, centers)
        proposed = centers.copy()
        changed = False
        for r in range(len(centers)):
            if time.perf_counter() >= stop:
                break
            members = np.flatnonzero(assigned == r)
            if len(members) <= 1:
                continue
            values = np.zeros(len(members), dtype=np.longdouble)
            block = max(1, 262144 // len(members))
            complete = True
            for a in range(0, len(members), block):
                if time.perf_counter() >= stop:
                    complete = False
                    break
                values += np.sum(D[np.ix_(members[a:a + block], members)],
                                 axis=0, dtype=np.longdouble)
            if complete:
                j = int(members[int(np.argmin(values))])
                if j != centers[r]:
                    proposed[r] = j
                    changed = True
        if not changed:
            return
        before = _cost(A, centers)
        after = _cost(A, proposed)
        if not np.isfinite(after) or after >= before:
            return
        centers = proposed
        _remember(A, centers, len(centers), saved)


def _pool(D, centers, assigned, nearest, slots, width, stop):
    """Select high-insertion-gain candidates per cluster so that candidates are not all concentrated in one cluster."""
    n = len(D)
    removed = [int(centers[r]) for r in slots]
    fixed = [int(centers[r]) for r in range(len(centers)) if r not in slots]
    allowed = np.ones(n, dtype=bool)
    allowed[fixed] = False
    available = np.flatnonzero(allowed)
    if len(available) <= width:
        return fixed, available
    chosen = set(removed)
    quota = max(1, int(math.ceil(width / len(slots))))
    for r in slots:
        if time.perf_counter() >= stop:
            break
        rows = np.flatnonzero(assigned == r)
        gain = np.zeros(n)
        block = max(1, 262144 // n)
        complete = True
        for a in range(0, len(rows), block):
            if time.perf_counter() >= stop:
                complete = False
                break
            ix = rows[a:a + block]
            gain += np.maximum(nearest[ix, None] - D[ix], 0.0).sum(axis=0)
        if not complete:
            break
        gain[~allowed] = -np.inf
        count = min(quota, len(available))
        top = np.argpartition(gain, n - count)[n - count:]
        chosen.update(int(j) for j in top)
    return fixed, np.asarray(sorted(chosen), dtype=np.int64)


def _pairs(D, A, fixed, candidates, saved, stop):
    """Fully scan two-facility combinations from the given candidate pool, checking the wall clock per block."""
    n = len(D)
    base = np.min(D[:, fixed], axis=1) if fixed else np.full(n, np.inf)
    data = np.minimum(D[:, candidates], base[:, None])
    pairs = itertools.combinations(range(len(candidates)), 2)
    block = max(1, 262144 // n)
    while time.perf_counter() < stop:
        batch = list(itertools.islice(pairs, block))
        if not batch:
            return
        pair = np.asarray(batch, dtype=np.int64)
        values = np.minimum(data[:, pair[:, 0]], data[:, pair[:, 1]])
        costs = values.sum(axis=0, dtype=np.longdouble)
        k = int(np.argmin(costs))
        if np.isfinite(costs[k]):
            trial = fixed + [int(candidates[pair[k, 0]]),
                             int(candidates[pair[k, 1]])]
            _remember(A, trial, len(fixed) + 2, saved)


def _joint(D, A, centers, fixed, candidates, labels, groups,
           saved, link_budget, stop, guard):
    """Recursive distance-level covering model; no assignment decisions and no unreachability penalty."""
    n = len(D)
    m = len(candidates)
    q = len(centers) - len(fixed)
    if m <= q or time.perf_counter() + guard >= stop:
        return False
    baseline = np.min(D[:, fixed], axis=1) if fixed else np.full(n, np.inf)
    data = D[:, candidates]
    relevant = (data < baseline[:, None]) & np.isfinite(data)
    active = max(1, int(np.count_nonzero(relevant.any(axis=1))))
    per_row = max(2, int(link_budget / active))
    keep = np.isin(candidates, np.asarray(centers))
    old_distance = np.where(keep[None, :], data, np.inf)
    old_nearest = np.argmin(old_distance, axis=1)
    old_value = old_distance[np.arange(n), old_nearest]

    # Each distance level measures how far no selected facility lies within that level; the positive gap between levels is its cost.
    rr = [np.zeros(m, dtype=np.int64)]
    cc = [np.arange(m, dtype=np.int64)]
    vv = [np.ones(m)]
    lower = [float(q)]
    upper = [float(q)]
    weights = []
    z_count = 0
    row_count = 1
    complete = True
    fixed_labels = set(int(labels[j]) for j in fixed if labels[j] >= 0)
    for g in range(len(groups)):
        if g not in fixed_labels:
            ix = np.flatnonzero(labels[candidates] == g)
            if not len(ix):
                return False
            rr.append(np.full(len(ix), row_count, dtype=np.int64))
            cc.append(ix)
            vv.append(np.ones(len(ix)))
            lower.append(1.0)
            upper.append(np.inf)
            row_count += 1
    for i in range(n):
        if time.perf_counter() + guard >= stop:
            return False
        ix = np.flatnonzero(relevant[i])
        if len(ix) > per_row:
            complete = False
            take = np.argpartition(data[i, ix], per_row - 1)[:per_row]
            ix = ix[take]
            old = int(old_nearest[i])
            if old_value[i] < baseline[i] and not np.any(ix == old):
                ix = np.append(ix, old)
        if not len(ix):
            if not np.isfinite(baseline[i]):
                return False
            continue
        levels, inverse = np.unique(data[i, ix], return_inverse=True)
        count = len(levels)
        finite_base = np.isfinite(baseline[i])
        nz = count if finite_base else count - 1
        differences = np.diff(np.append(levels, baseline[i])) if finite_base else np.diff(levels)
        weights.append(differences)
        rr.append(row_count + inverse)
        cc.append(ix)
        vv.append(np.ones(len(ix)))
        if nz:
            rr.append(row_count + np.arange(nz, dtype=np.int64))
            cc.append(m + z_count + np.arange(nz, dtype=np.int64))
            vv.append(np.ones(nz))
        if count > 1:
            rr.append(row_count + np.arange(1, count, dtype=np.int64))
            cc.append(m + z_count + np.arange(count - 1, dtype=np.int64))
            vv.append(-np.ones(count - 1))
        lower.extend([1.0] + [0.0] * (count - 1))
        upper.extend([np.inf] * count)
        row_count += count
        z_count += nz

    if time.perf_counter() + guard >= stop:
        return False
    c = np.zeros(m + z_count)
    if z_count:
        c[m:] = np.concatenate(weights)
        positive = c[c > 0.0]
        if len(positive):
            scale = min(1.0 / max(float(np.median(positive)), 1e-300),
                        1e8 / max(float(positive.max()), 1e-300))
            c *= scale
    matrix = coo_matrix((np.concatenate(vv), (np.concatenate(rr), np.concatenate(cc))),
                        shape=(row_count, m + z_count)).tocsc()
    integer = np.zeros(m + z_count, dtype=np.uint8)
    integer[:m] = 1
    remaining = stop - time.perf_counter() - guard
    if remaining <= 0.0:
        return False
    result = milp(c, integrality=integer,
                  bounds=Bounds(np.zeros(m + z_count), np.ones(m + z_count)),
                  constraints=LinearConstraint(matrix, np.asarray(lower), np.asarray(upper)),
                  options={"time_limit": remaining, "mip_rel_gap": 0.0,
                           "presolve": True})
    if result.x is not None and np.isfinite(result.x[:m]).all():
        chosen = np.flatnonzero(result.x[:m] > 0.5)
        if len(chosen) == q:
            trial = fixed + [int(j) for j in candidates[chosen]]
            _remember(A, trial, len(centers), saved)
            # Only when the untruncated global model is solved to optimality may the whole search stop early.
            if result.status == 0 and complete and not fixed and m == n:
                return True
    return False


def _search(D, A, p, labels, groups, saved, start, limit, deadline):
    n = len(D)
    guard = max(0.001, 0.035 * limit)
    rng = np.random.default_rng(724531)
    strength = math.log1p(max(0.0, limit))
    depth = min(p, max(2, int(math.ceil(math.sqrt(p) * strength))))
    width_factor = (1.0 + strength) * math.sqrt(n / p)
    links = max(2 * n, int(2600 * max(limit, 0.001) ** 0.65))
    call_span = limit / (2.0 + math.sqrt(1.0 + strength) * math.log1p(n / p))
    last_improvement = time.perf_counter()
    joint_available = milp is not None
    tried_global = False
    spent_build = 0.0
    while time.perf_counter() < deadline:
        now = time.perf_counter()
        left = deadline - now
        if left <= guard:
            return
        if saved[1] == 0:
            return
        centers = np.asarray(saved[0], dtype=np.int64)
        old_cost = saved[1]
        assigned, nearest = _partition(D, centers)
        can_global = n * n <= links and joint_available and not tried_global
        if can_global:
            slots = list(range(p))
            fixed = []
            candidates = np.arange(n, dtype=np.int64)
            tried_global = True
        else:
            stall = max(0.0, now - last_improvement) / max(limit, 1e-9)
            growing = min(depth, max(2, int(math.ceil(2.0 + depth * math.sqrt(stall)))))
            if joint_available and growing > 2 and rng.random() < min(0.75, strength / (1.0 + strength)):
                size = int(rng.integers(3, growing + 1))
            else:
                size = 2
            size = min(size, p)
            anchor = int(rng.integers(p))
            affinity = D[centers[anchor], centers] + D[centers, centers[anchor]]
            affinity[anchor] = -1.0
            order = np.argsort(affinity, kind="stable")
            nearby = min(p, max(size, int(math.ceil(size * (1.0 + strength)))))
            rest = order[order != anchor][:max(0, nearby - 1)]
            slots = [anchor]
            if size > 1:
                if rng.random() < 1.0 / (1.0 + strength):
                    rest = np.asarray([j for j in range(p) if j != anchor])
                slots.extend(int(j) for j in rng.choice(rest, size - 1, replace=False))
            width = min(n - p + size, max(size + 1, int(math.ceil(size * width_factor))))
            build_start = time.perf_counter()
            fixed, candidates = _pool(D, centers, assigned, nearest, slots, width,
                                      min(deadline - guard, now + call_span * 0.4))
            spent_build = max(spent_build * 0.8, time.perf_counter() - build_start)
        if time.perf_counter() >= deadline - guard:
            return
        end = min(deadline, time.perf_counter() + max(call_span, 2.0 * spent_build))
        if len(slots) == 2:
            _pairs(D, A, fixed, candidates, saved, end - guard)
        elif joint_available and end - time.perf_counter() > 2.0 * guard:
            try:
                proven = _joint(D, A, centers, fixed, candidates, labels, groups,
                                saved, links, end, guard)
                if proven:
                    return
            except Exception:
                joint_available = False
        if saved[1] < old_cost:
            last_improvement = time.perf_counter()
            refine_stop = min(deadline - guard, last_improvement + call_span * 0.12)
            _relocate(D, A, saved[0], saved, refine_stop)


def solve(instance: dict, time_limit_s: float) -> dict:
    start = time.perf_counter()
    saved = [None, np.longdouble(np.inf)]
    n = int(instance["n"])
    p = int(instance["p"])
    raw = instance["dist"]
    if p == n:
        return {"medians": list(range(n))}
    try:
        limit = float(time_limit_s)
        if not math.isfinite(limit) or limit < 0:
            limit = 0.0
        deadline = start + 0.96 * limit
        A = np.asarray(raw, dtype=np.float64)
        A = A.copy()
        A[np.isnan(A)] = np.inf
        reach = np.isfinite(A)
        labels, groups = _sinks(reach)
        seeds = [int(g[0]) for g in groups]
        taken = set(seeds)
        fallback = seeds + [j for j in range(n) if j not in taken][:p - len(seeds)]
        saved[0] = fallback
        saved[1] = _cost(A, fallback)
        if time.perf_counter() >= deadline:
            return {"medians": saved[0]}
        maximum = float(np.max(A, where=reach, initial=0.0))
        D = A / maximum if maximum > 0.0 else A.copy()
        if p == 1:
            eligible = np.flatnonzero(reach.all(axis=0))
            values = np.sum(A[:, eligible], axis=0, dtype=np.longdouble)
            return {"medians": [int(eligible[int(np.argmin(values))])]}
        if p == n - 1:
            without_self = A.copy()
            np.fill_diagonal(without_self, np.inf)
            removed = int(np.argmin(np.min(without_self, axis=1)))
            return {"medians": [j for j in range(n) if j != removed]}
        if groups:
            score = np.sum(np.where(reach, D, 0.0), axis=0, dtype=np.longdouble)
            seeds = [int(g[int(np.argmin(score[g]))]) for g in groups]
        constructed = _greedy(D, seeds, p, min(deadline, start + 0.18 * limit))
        _remember(A, constructed, p, saved)
        _relocate(D, A, saved[0], saved, min(deadline, start + 0.25 * limit))
        _search(D, A, p, labels, groups, saved, start, limit, deadline)
    except Exception:
        if saved[0] is None:
            saved[0] = _plain_feasible(raw, n, p)
    return {"medians": [int(j) for j in saved[0]]}
