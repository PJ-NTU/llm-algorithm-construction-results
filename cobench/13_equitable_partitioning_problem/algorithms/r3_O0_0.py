"""Integer-deviation-guided multi-neighbourhood equitable grouping
Combines rare-attribute-first construction, move and swap descent, time-limited integer-programming repair and adaptive perturbation, continuously keeping the complete grouping that is best under the original objective.
"""
import math
import random
import time

try:
    import numpy as np
except Exception:
    np = None

try:
    from scipy.optimize import milp, Bounds, LinearConstraint
    from scipy.sparse import coo_matrix, csr_matrix, hstack, vstack
except Exception:
    milp = None


def solve(instance: dict, time_limit_s: float) -> dict:
    """Return, within the wall-clock budget, a grouping in which each person belongs to exactly one group."""
    started = time.perf_counter()
    result = {"group_assignment": {}}
    try:
        n = int(instance["num_individuals"])
        k = int(instance["num_groups"])
        if n <= 0:
            return result
        if k <= 0:
            k = 1
        result["group_assignment"] = {f"person_{i}": i % k for i in range(n)}
        budget = max(0.0, float(time_limit_s))
        deadline = started + budget - max(0.002, 0.04 * budget)
        if np is None or k == 1 or k >= n or time.perf_counter() >= deadline:
            return result
        _search(instance, n, k, budget, deadline, result)
    except Exception:
        # The stored complete grouping also serves as a fallback when dependencies are unavailable or numerical solving fails.
        pass
    return result


def _search(instance, n, k, budget, deadline, result):
    a = np.asarray(instance["attribute_matrix"], dtype=np.int64).reshape(
        n, int(instance["num_attributes"]))
    total = a.sum(axis=0)
    active = total > 0
    a = a[:, active]
    total = total[active]
    m = len(total)
    if not m or time.perf_counter() >= deadline:
        return
    af = a.astype(np.float64)
    rows = [np.flatnonzero(a[i]) for i in range(n)]
    degree = a.sum(axis=1)
    q, r = total // k, total % k
    lower_bound = int((2 * r * (k - r)).sum())
    rng = np.random.default_rng(932417)
    py_rng = random.Random(932417)
    eps = 1.0 / (1.0 + 8.0 * n * max(1, m))
    pool_cap = max(4, int(math.sqrt(1800.0 * budget / (1.0 + m / 32.0))))
    variable_cap = max(12, int(80.0 * math.sqrt(budget) *
                                  (1.0 + math.sqrt(n / (k * max(1, m))))))
    max_depth = min(k, max(2, int(2.0 + math.sqrt(budget))))

    def counts(z):
        c = np.zeros((k, m), dtype=np.int64)
        np.add.at(c, z, a)
        return c

    def value(c):
        return int(np.abs(k * c - total).sum())

    z = np.arange(n, dtype=np.int64) % k
    c = counts(z)
    best_z, best_c = z.copy(), c.copy()
    best_f = value(c)
    best_q = float(np.square(c.astype(float)).sum())

    def remember(z1, c1):
        nonlocal best_z, best_c, best_f, best_q
        f = value(c1)
        sq = float(np.square(c1.astype(float)).sum())
        if f < best_f or (f == best_f and sq < best_q - 1e-9):
            best_z, best_c = z1.copy(), c1.copy()
            best_f, best_q = f, sq
            result["group_assignment"] = {
                f"person_{i}": int(best_z[i]) for i in range(n)}
        return f

    if best_f == lower_bound:
        return

    def marginals(c1):
        e = k * c1 - total
        minus = np.abs(e - k) - np.abs(e)
        plus = np.abs(e + k) - np.abs(e)
        return (minus + eps * (1.0 - 2.0 * c1),
                plus + eps * (1.0 + 2.0 * c1))

    def refresh(c1, minus, plus, g):
        e = k * c1[g] - total
        minus[g] = np.abs(e - k) - np.abs(e) + eps * (1.0 - 2.0 * c1[g])
        plus[g] = np.abs(e + k) - np.abs(e) + eps * (1.0 + 2.0 * c1[g])

    def pair_gaps(c1, g):
        s = c1 + c1[g]
        low = s // 2
        pair_lb = (np.abs(k * low - total) +
                   np.abs(k * (s - low) - total)).sum(axis=1)
        f = np.abs(k * c1 - total).sum(axis=1)
        gaps = f + f[g] - pair_lb
        gaps[g] = -1
        return gaps

    def polish(z1, c1, end):
        minus, plus = marginals(c1)
        while time.perf_counter() < end:
            moved = False
            for i0 in rng.permutation(n):
                if time.perf_counter() >= end:
                    return z1, c1
                i = int(i0)
                idx = rows[i]
                if not len(idx):
                    continue
                old = int(z1[i])
                delta = plus[:, idx].sum(axis=1) + minus[old, idx].sum()
                delta[old] = np.inf
                new = int(np.argmin(delta))
                if delta[new] < -1e-9:
                    c1[old, idx] -= 1
                    c1[new, idx] += 1
                    z1[i] = new
                    refresh(c1, minus, plus, old)
                    refresh(c1, minus, plus, new)
                    moved = True
            if moved:
                continue
            # Each group picks one complementary group, and objective increments of candidate swaps are computed exactly in batch.
            tried = set()
            for g0 in rng.permutation(k):
                if time.perf_counter() >= end:
                    return z1, c1
                g = int(g0)
                gaps = pair_gaps(c1, g).astype(float)
                ranking = gaps + rng.random(k) * max(1.0, float(gaps.max()) * 0.2)
                ranking[g] = -np.inf
                h = int(np.argmax(ranking))
                pair = (min(g, h), max(g, h))
                if pair in tried:
                    continue
                tried.add(pair)
                ia = np.flatnonzero(z1 == g)
                ib = np.flatnonzero(z1 == h)
                if not len(ia) or not len(ib):
                    continue
                if len(ia) > pool_cap:
                    ia = rng.choice(ia, pool_cap, replace=False)
                if len(ib) > pool_cap:
                    ib = rng.choice(ib, pool_cap, replace=False)
                aa, ab = af[ia], af[ib]
                u = minus[g] + plus[h]
                v = plus[g] + minus[h]
                delta = ((aa @ u)[:, None] + (ab @ v)[None, :] -
                         np.einsum('im,jm,m->ij', aa, ab, u + v, optimize=False))
                flat = int(np.argmin(delta))
                if delta.flat[flat] < -1e-9:
                    p, t = divmod(flat, len(ib))
                    i, j = int(ia[p]), int(ib[t])
                    change = a[j] - a[i]
                    c1[g] += change
                    c1[h] -= change
                    z1[i], z1[j] = h, g
                    refresh(c1, minus, plus, g)
                    refresh(c1, minus, plus, h)
                    moved = True
                    break
            if not moved:
                return z1, c1
        return z1, c1

    # Rare attributes first; construction only produces starting points, and final comparisons always use the original objective.
    rarity = 1.0 / np.sqrt(np.maximum(total, 1))
    priority = af @ rarity + rng.random(n) * (float(rarity.sum()) / max(1, m))
    order = np.argsort(-priority)
    trial = z.copy()
    cc = np.zeros_like(c)
    sizes = np.zeros(k, dtype=np.int64)
    construction_end = min(deadline, time.perf_counter() + 0.09 * budget)
    for i0 in order:
        if time.perf_counter() >= construction_end:
            break
        i = int(i0)
        idx = rows[i]
        scores = ((cc[:, idx] * rarity[idx]).sum(axis=1) if len(idx)
                  else np.zeros(k))
        choices = np.flatnonzero(scores <= scores.min() + 1e-10)
        if len(choices) > 1:
            sizes_min = sizes[choices].min()
            choices = choices[sizes[choices] == sizes_min]
        g = int(rng.choice(choices))
        trial[i] = g
        cc[g, idx] += 1
        sizes[g] += 1
    cc = counts(trial)
    if value(cc) <= value(c):
        z, c = trial, cc
    remember(z, c)
    z, c = polish(z, c, min(deadline, time.perf_counter() + 0.20 * budget))
    remember(z, c)
    if best_f == lower_bound:
        return

    def repair(z1, c1, groups, selected, end):
        """Eliminate the indicator variables of the last group and solve a local problem strictly equivalent to the original absolute deviation."""
        if milp is None or time.perf_counter() >= end:
            return None
        groups = np.asarray(groups, dtype=np.int64)
        selected = np.asarray(selected, dtype=np.int64)
        ng, p = len(groups), len(selected)
        if ng < 2 or not p:
            return None
        width = ng - 1
        nx = p * width
        sub = a[selected]
        fixed = c1[groups].copy()
        group_index = {int(g): j for j, g in enumerate(groups)}
        original_local = np.array([group_index[int(z1[i])] for i in selected])
        np.add.at(fixed, original_local, -sub)
        base = fixed.copy()
        base[-1] += sub.sum(axis=0)
        ii, mm = np.nonzero(sub)
        rr, cols, data = [], [], []
        for g in range(width):
            variable = ii * width + g
            rr.extend((g * m + mm, (ng - 1) * m + mm))
            cols.extend((variable, variable))
            data.extend((np.ones(len(ii)), -np.ones(len(ii))))
        mat = coo_matrix((np.concatenate(data),
                          (np.concatenate(rr), np.concatenate(cols))),
                         shape=(ng * m, nx)).tocsr()
        lows = np.flatnonzero(np.tile(q > 0, ng))
        highs = np.flatnonzero(np.tile(r > 0, ng))
        nl, nh = len(lows), len(highs)
        ns = nl + nh
        if not ns:
            return None
        obj = np.concatenate((np.zeros(nx),
                              np.tile(k - r, ng)[lows],
                              np.tile(r, ng)[highs])).astype(float)
        count_matrix = vstack((mat[lows], mat[highs]), format='csr')
        slack = coo_matrix((np.concatenate((np.ones(nl), -np.ones(nh))),
                            (np.arange(ns), np.arange(ns))),
                           shape=(ns, ns)).tocsr()
        constraint = hstack((count_matrix, slack), format='csr')
        lo = np.concatenate((np.tile(q, ng)[lows] - base.ravel()[lows],
                             np.full(nh, -np.inf)))
        hi = np.concatenate((np.full(nl, np.inf),
                             np.tile(q + 1, ng)[highs] - base.ravel()[highs]))
        if width > 1:
            assignment = coo_matrix((np.ones(nx),
                                     (np.repeat(np.arange(p), width), np.arange(nx))),
                                    shape=(p, nx + ns)).tocsr()
            constraint = vstack((constraint, assignment), format='csr')
            lo = np.concatenate((lo, np.full(p, -np.inf)))
            hi = np.concatenate((hi, np.ones(p)))
        # For integer counts, this weighted excess equals half of the original scaled objective minus a constant.
        current = int(((k - r) * np.maximum(q - c1[groups], 0) +
                       r * np.maximum(c1[groups] - q - 1, 0)).sum())
        if current <= 0:
            return None
        constraint = vstack((constraint, csr_matrix(obj.reshape(1, -1))), format='csr')
        lo = np.concatenate((lo, [-np.inf]))
        hi = np.concatenate((hi, [current - 1]))
        lb = np.zeros(nx + ns)
        ub = np.concatenate((np.ones(nx), np.tile(q, ng)[lows],
                             np.maximum(np.tile(total - q - 1, ng)[highs], 0))).astype(float)
        if not np.any(fixed):
            # When whole groups are selected, group labels are fully symmetric, so fixing one label loses no feasible grouping.
            lb[0] = 1.0
            ub[0] = 1.0
        integrality = np.concatenate((np.ones(nx, dtype=np.uint8),
                                      np.zeros(ns, dtype=np.uint8)))
        remaining = end - time.perf_counter()
        if remaining <= 0.005:
            return None
        try:
            answer = milp(obj, integrality=integrality, bounds=Bounds(lb, ub),
                          constraints=LinearConstraint(constraint, lo, hi),
                          options={"time_limit": remaining, "mip_rel_gap": 0.0,
                                   "presolve": True, "disp": False})
        except Exception:
            return None
        if answer.x is None:
            return None
        x = answer.x[:nx].reshape(p, width)
        if not np.all(np.isfinite(x)) or np.max(np.abs(x - np.rint(x))) > 1e-4:
            return None
        chosen = np.full(p, ng - 1, dtype=np.int64)
        used = x.max(axis=1) > 0.5
        if np.any(x.sum(axis=1) > 1.00001):
            return None
        chosen[used] = np.argmax(x[used], axis=1)
        zz = z1.copy()
        zz[selected] = groups[chosen]
        newc = counts(zz)
        if value(newc) < value(c1):
            return zz, newc
        return None

    failures = 0
    pair_failures = {}
    base_slice = budget / (5.0 + math.sqrt(1.0 + budget))
    while time.perf_counter() < deadline:
        if best_f == lower_bound:
            return
        before = best_f
        left = deadline - time.perf_counter()
        violation = ((k - r) * np.maximum(q - c, 0) +
                     r * np.maximum(c - q - 1, 0)).sum(axis=1).astype(float)
        weights = violation + max(1.0, float(violation.sum()) * 0.02 / k)
        g = int(rng.choice(k, p=weights / weights.sum()))
        gaps = pair_gaps(c, g).astype(float)
        for h0 in range(k):
            pair = (min(g, h0), max(g, h0))
            gaps[h0] /= 1.0 + pair_failures.get(pair, 0)
        gaps += rng.random(k) * max(1.0, float(gaps.max()) * 0.15)
        gaps[g] = -np.inf
        h = int(np.argmax(gaps))
        groups = [g, h]
        depth = min(max_depth, 2 + int(math.log2(1 + failures)))
        if rng.random() < 1.0 / (2.0 + math.sqrt(k)):
            depth = k
        if depth > 2:
            candidates = np.array([j for j in range(k) if j not in groups])
            w = weights[candidates]
            extras = rng.choice(candidates, min(depth - 2, len(candidates)),
                                replace=False, p=w / w.sum())
            groups.extend(int(j) for j in extras)
        selected = np.flatnonzero(np.isin(z, groups))
        cap = max(2, variable_cap // max(1, len(groups) - 1))
        if len(selected) > cap:
            e = k * c - total
            remove = np.abs(e - k) - np.abs(e)
            benefit = -np.einsum('ij,ij->i', af[selected], remove[z[selected]],
                                 optimize=False)
            importance = 1.0 + np.maximum(benefit, 0) + np.sqrt(degree[selected])
            keys = np.log(importance) - np.log(-np.log(rng.uniform(1e-12, 1.0, len(selected))))
            selected = selected[np.argpartition(keys, -cap)[-cap:]]
        dimension = len(selected) * (len(groups) - 1)
        allotted = min(left * 0.55, base_slice * math.sqrt(
            max(1, dimension) / (32.0 + m)))
        repaired = repair(z, c, groups, selected,
                          min(deadline, time.perf_counter() + allotted))
        if repaired is not None:
            z, c = repaired
            remember(z, c)
            pair_failures.clear()
        else:
            pair = (min(g, h), max(g, h))
            pair_failures[pair] = pair_failures.get(pair, 0) + 1
        if time.perf_counter() >= deadline:
            break
        if repaired is None:
            # After failures, perturb from the best known solution; the magnitude grows with budget and stagnation.
            z, c = best_z.copy(), best_c.copy()
            amplitude = min(n, max(1, int(math.sqrt(budget) *
                math.sqrt(1.0 + failures) * math.sqrt(n / max(1, k * m)))))
            for i0 in rng.choice(n, amplitude, replace=False):
                if time.perf_counter() >= deadline:
                    break
                i = int(i0)
                old = int(z[i])
                new = py_rng.randrange(k - 1)
                if new >= old:
                    new += 1
                c[old] -= a[i]
                c[new] += a[i]
                z[i] = new
        local_time = min((deadline - time.perf_counter()) * 0.4,
                         base_slice * (0.16 + 0.04 * math.sqrt(1.0 + failures)))
        if local_time > 0:
            z, c = polish(z, c, min(deadline, time.perf_counter() + local_time))
            remember(z, c)
        if best_f < before:
            failures = 0
        else:
            failures += 1
        if value(c) > best_f and rng.random() < 1.0 / math.sqrt(1.0 + failures):
            z, c = best_z.copy(), best_c.copy()
