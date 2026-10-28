"""Dual core and ring local-branching search
Combines linear-relaxation pricing, safe reduced-cost fixing, integer core search and local branching that excludes neighbourhoods proven non-improving layer by layer, continuously keeping a solution feasible under the original constraints.
"""

import math
import random
import time

try:
    import numpy as np
except Exception:
    np = None

try:
    from scipy.optimize import Bounds, LinearConstraint, linprog, milp
except Exception:
    Bounds = LinearConstraint = linprog = milp = None


def _plain(instance, deadline, saved):
    """Without a numerical backend, drive destroy-repair by the wall clock."""
    n = len(saved[0])
    m = int(instance['m'])
    p = [float(v) for v in instance['profit']]
    w = [[float(v) for v in row] for row in instance['weights']]
    b = [float(v) for v in instance['capacity']]
    rng = random.Random(104729 + n * 1009 + m)
    best = saved[0][:]
    value = math.fsum(p[j] for j in range(n) if best[j])
    strength = math.log1p(max(0.0, deadline - time.perf_counter()))
    while time.perf_counter() < deadline:
        x = best[:]
        chosen = [j for j in range(n) if x[j]]
        count = min(len(chosen), max(1, int((1.0 + strength) * math.sqrt(len(chosen)) * rng.random())))
        for j in rng.sample(chosen, count):
            x[j] = 0
        load = [math.fsum(w[r][j] for j in range(n) if x[j]) for r in range(m)]
        lam = [math.exp(rng.uniform(-0.7, 0.7)) / max(abs(v), 1.0) for v in b]
        order = sorted(range(n), key=lambda j: p[j] * math.exp(rng.uniform(-0.2, 0.2)) /
                       max(1e-100, math.fsum(lam[r] * max(0.0, w[r][j]) for r in range(m))),
                       reverse=True)
        for j in order:
            if time.perf_counter() >= deadline:
                break
            if not x[j] and p[j] > 0 and all(load[r] + w[r][j] <= b[r] for r in range(m)):
                x[j] = 1
                for r in range(m):
                    load[r] += w[r][j]
        score = math.fsum(p[j] for j in range(n) if x[j])
        if score > value and all(math.fsum(w[r][j] for j in range(n) if x[j]) <= b[r] for r in range(m)):
            best, value = x, score
            saved[0] = x[:]


def _search(instance, budget, start, saved):
    # Modelling, backend return and solution serialisation also use the same wall-clock budget.
    deadline = start + budget * 0.93
    n = len(saved[0])
    m = int(instance['m'])
    if not n or time.perf_counter() >= deadline:
        return
    if np is None:
        _plain(instance, deadline, saved)
        return
    raw_p = np.asarray(instance['profit'], dtype=float).reshape(n)
    raw_a = np.asarray(instance['weights'], dtype=float).reshape(m, n)
    raw_b = np.asarray(instance['capacity'], dtype=float).reshape(m)
    if not (np.all(np.isfinite(raw_p)) and np.all(np.isfinite(raw_a)) and np.all(np.isfinite(raw_b))):
        return
    nonnegative = bool(np.all(raw_a >= 0) and np.all(raw_b >= 0))
    usable = ((raw_p > 0) & np.all(raw_a <= raw_b[:, None], axis=0)
              if nonnegative else np.ones(n, dtype=bool))
    ids = np.flatnonzero(usable)
    k = len(ids)
    if not k:
        return
    original = raw_a[:, ids]
    profit = raw_p[ids]
    pscale = max(float(np.max(np.abs(profit))), 1e-100)
    q = profit / pscale
    scales = np.maximum(np.abs(raw_b), np.max(np.abs(original), axis=1)) if m else np.ones(0)
    scales[scales == 0] = 1.0
    a = original / scales[:, None]
    b = raw_b / scales
    strength = math.log1p(budget)
    rng = np.random.default_rng(104729 + n * 1009 + m * 9176)
    best = np.zeros(k, dtype=bool)
    best_raw = 0.0 if np.all(raw_b >= 0) else -float('inf')
    best_value = best_raw
    generation = 0
    last_gain = time.perf_counter()
    archive = []
    archive_limit = max(2, int(2.0 + strength))
    lp_x = np.zeros(k)
    dual = np.ones(m)
    rc = q.copy()
    upper = float('inf')
    certified = False
    proof_tol = 1e-8 * (1.0 + float(np.abs(q).sum()))
    row_error = np.finfo(float).eps * (k + 2) * np.maximum(np.abs(raw_b), np.abs(original).sum(axis=1))

    # Use the objective lattice only for exactly integer profits; near-integers or fractions are never rounded arbitrarily.
    unit = 0.0
    if np.all(profit == np.floor(profit)) and pscale <= 2 ** 50:
        divisor = 0
        for v in profit:
            divisor = math.gcd(divisor, abs(int(v)))
        if divisor:
            unit = divisor / pscale

    def loads(matrix, x):
        return np.einsum('ij,j->i', matrix, x, optimize=False)

    def costs(lam):
        return np.einsum('ij,i->j', a, lam, optimize=False)

    def feasible(x):
        load = loads(original, x)
        # Critical rows are verified with independent high-precision summation; the tolerance only screens rows to verify and does not relax capacity.
        error = row_error
        if np.any(load > raw_b + error):
            return False
        for r0 in np.flatnonzero(load >= raw_b - error):
            r = int(r0)
            if math.fsum(float(original[r, j]) for j in np.flatnonzero(x)) > raw_b[r]:
                return False
        return True

    def offer(x):
        nonlocal best, best_raw, best_value, generation, last_gain
        x = np.asarray(x, dtype=bool)
        if not feasible(x):
            return False
        chosen = np.flatnonzero(x)
        raw_value = math.fsum(float(profit[j]) for j in chosen)
        if raw_value > best_raw:
            best, best_raw, best_value = x.copy(), raw_value, raw_value / pscale
            answer = [0] * n
            for j in ids[chosen]:
                answer[int(j)] = 1
            saved[0] = answer
            generation += 1
            last_gain = time.perf_counter()
        key = x.tobytes()
        if not any(entry[2] == key for entry in archive):
            archive.append((raw_value, x.copy(), key))
            archive.sort(key=lambda entry: entry[0], reverse=True)
            if len(archive) > archive_limit:
                archive.pop()
        return True

    def repair_fill(x, order, stop):
        x = x.copy()
        load = loads(original, x)
        while nonnegative and np.any(load > raw_b) and time.perf_counter() < stop:
            chosen = np.flatnonzero(x)
            if not len(chosen):
                break
            over = load > raw_b
            deficit = np.maximum(load[over] - raw_b[over], np.maximum(scales[over] * 1e-12, 1e-100))
            relief = np.minimum(original[over][:, chosen] / deficit[:, None], 1.0).sum(axis=0)
            j = int(chosen[np.argmax(relief / np.maximum(q[chosen], 1e-100))])
            x[j] = False
            load = loads(original, x)
        if np.any(load > raw_b):
            return x
        for j0 in order:
            if time.perf_counter() >= stop:
                break
            j = int(j0)
            if not x[j] and q[j] > 0 and np.all(load + original[:, j] <= raw_b):
                x[j] = True
                load += original[:, j]
        return x

    def density_order(lam, noise=0.0):
        score = q / np.maximum(costs(lam), 1e-100)
        if noise:
            score *= np.exp(rng.uniform(-noise, noise, k))
        return np.argsort(-score, kind='stable')

    base_order = density_order(dual)
    offer(repair_fill(best, base_order, deadline))
    if nonnegative and feasible(np.ones(k, dtype=bool)):
        offer(np.ones(k, dtype=bool))
        return
    if time.perf_counter() >= deadline:
        return

    if linprog is not None:
        seconds = min(budget * 0.12, (deadline - time.perf_counter()) * 0.25)
        if seconds > 0:
            try:
                result = linprog(-q, A_ub=a if m else None, b_ub=b if m else None,
                                 bounds=(0.0, 1.0), method='highs',
                                 options={'time_limit': seconds, 'presolve': True})
                if result.x is not None:
                    lp_x = np.clip(result.x, 0.0, 1.0)
                    offer(repair_fill(lp_x >= 1.0 - 1e-7, base_order, deadline))
                if result.success:
                    dual = np.maximum(0.0, -np.asarray(result.ineqlin.marginals))
                    rc = q - costs(dual)
                    upper = float(dual @ b + np.maximum(rc, 0.0).sum())
                    proof_tol += 1e-8 * abs(upper)
                    certified = True
                    base_order = density_order(dual)
                    offer(repair_fill(lp_x >= 1.0 - 1e-7, base_order, deadline))
                    offer(repair_fill(lp_x >= 0.5, base_order, deadline))
                    offer(repair_fill(np.zeros(k, dtype=bool), base_order, deadline))
            except Exception:
                pass
    if not certified:
        if m:
            dual = np.maximum(1e-8, np.sum(np.maximum(a, 0.0), axis=1))
            dual /= max(float(dual.sum()), 1e-100)
            positive_cost = costs(dual)
            ratios = q[positive_cost > 0] / positive_cost[positive_cost > 0]
            if len(ratios):
                dual *= max(float(np.median(ratios)), 1e-8)
        rc = q - costs(dual)
        base_order = density_order(dual)

    # Short refinement provides a stronger lower bound, directly shrinking the subsequent integer subproblems.
    polish_end = min(deadline, time.perf_counter() + budget * 0.025)
    while nonnegative and time.perf_counter() < polish_end:
        chosen = np.flatnonzero(best)
        outsiders = np.flatnonzero(~best)
        slack = raw_b - loads(original, best)
        changed = False
        for j in outsiders[np.argsort(-q[outsiders])]:
            if time.perf_counter() >= polish_end:
                break
            cheap = chosen[q[chosen] < q[j]]
            fit = cheap[np.all(original[:, cheap] + slack[:, None] >= original[:, j, None], axis=0)]
            if len(fit):
                old = int(fit[np.argmin(q[fit])])
                x = best.copy()
                x[old], x[int(j)] = False, True
                previous = generation
                offer(repair_fill(x, base_order, polish_end))
                changed = generation != previous
                if changed:
                    break
        if not changed:
            break

    def universe():
        if not certified or not math.isfinite(best_value):
            return np.arange(k)
        gap = max(0.0, upper - best_value - unit) + proof_tol
        disagree = best != (rc > 0)
        return np.flatnonzero((np.abs(rc) <= gap) | disagree)

    def integer_neighborhood(parent, core, seconds, metric=None, inner=-1, outer=None):
        """Build the same binary model on flip variables, adding an objective lower bound and a dual-loss budget."""
        if milp is None or not len(core) or seconds <= 0:
            return False, False
        sign = 1.0 - 2.0 * parent[core].astype(float)
        gain = q[core] * sign
        matrix = [a[:, core] * sign[None, :]]
        lower = [np.full(m, -np.inf)]
        upper_rows = [b - loads(a, parent)]
        parent_value = math.fsum(float(q[j]) for j in np.flatnonzero(parent))
        if math.isfinite(best_value):
            target = best_value + unit
            # Lattice rows are scaled per unit so that "improve by at least one unit" is not swallowed by the absolute tolerance.
            objective_scale = max(unit, 1e-6) if unit else 1.0
            margin = min(unit * 1e-4, proof_tol) if unit else proof_tol
            matrix.append((gain / objective_scale)[None, :])
            lower.append(np.array([(target - parent_value - margin) / objective_scale]))
            upper_rows.append(np.array([np.inf]))
            if certified:
                loss = float(np.where(parent, np.maximum(-rc, 0.0), np.maximum(rc, 0.0)).sum())
                matrix.append((-rc[core] * sign)[None, :])
                lower.append(np.array([-np.inf]))
                upper_rows.append(np.array([upper - target - loss + proof_tol]))
        if metric is not None:
            distance = np.ones(len(core)) if metric == 'hamming' else parent[core].astype(float)
            matrix.append(distance[None, :])
            lower.append(np.array([float(inner + 1) if inner >= 0 else -np.inf]))
            upper_rows.append(np.array([float(outer)]))
        constraint = LinearConstraint(np.vstack(matrix), np.concatenate(lower), np.concatenate(upper_rows))
        seconds = min(seconds, deadline - time.perf_counter())
        if seconds <= 0:
            return False, False
        try:
            result = milp(c=-gain, integrality=np.ones(len(core), dtype=np.uint8),
                          bounds=Bounds(np.zeros(len(core)), np.ones(len(core))),
                          constraints=constraint,
                          options={'time_limit': seconds, 'mip_rel_gap': 0.0, 'presolve': True, 'disp': False})
            exact_candidate = False
            accepted = False
            if result.x is not None:
                x = parent.copy()
                flip = np.asarray(result.x) >= 0.5
                x[core] = np.logical_xor(parent[core], flip)
                exact_candidate = offer(x)
                accepted = exact_candidate
                if not accepted and nonnegative:
                    accepted = offer(repair_fill(x, base_order, deadline))
            # No feasible improvement also constitutes a neighbourhood certificate; a repaired solution cannot serve as an optimality certificate.
            complete = result.status == 2 or (result.status == 0 and exact_candidate)
            return bool(complete), accepted
        except Exception:
            return False, False

    width_base = min(k, max(2 * m + 2, int((3.0 + 2.0 * strength) * math.sqrt(k))))
    width = width_base
    radius_base = max(2, int(2.0 + 2.0 * strength + math.log1p(k) / (2.0 + strength)))
    radius = {'hamming': radius_base, 'remove': max(1, radius_base // 2)}
    proven = {'hamming': -1, 'remove': -1}
    center_generation = generation
    last_global = start
    first_global = True
    neighborhood_number = 0
    failures = 0

    while time.perf_counter() < deadline:
        now = time.perf_counter()
        remaining = deadline - now
        if certified and math.isfinite(best_value):
            if unit and upper - best_value + proof_tol < unit:
                break
            if not unit and upper + proof_tol <= best_value:
                break
        allowed = universe()
        if not len(allowed):
            break
        if center_generation != generation:
            proven = {'hamming': -1, 'remove': -1}
            center_generation = generation
            radius = {'hamming': radius_base, 'remove': max(1, radius_base // 2)}
        stagnation = min(1.0, (now - last_gain) / max(budget, 1e-100))
        parent = best.copy()
        metric, inner, outer = None, -1, None
        unrestricted = False
        global_due = first_global or not math.isfinite(best_value) or now - last_global > budget * (0.34 + 0.035 * strength)
        if global_due:
            core = allowed
            seconds = min(remaining * 0.38, budget * (0.19 + 0.025 * strength))
            last_global = now
            first_global = False
            unrestricted = True
        elif neighborhood_number % 3 == 0:
            # Ring neighbourhoods over the full candidate set may cross the core boundary; inner layers already proven non-improving are not repeated.
            metric = 'hamming' if (neighborhood_number // 3) % 2 == 0 else 'remove'
            core = allowed
            total_distance = len(core) if metric == 'hamming' else int(parent[core].sum())
            inner = proven[metric]
            outer = min(total_distance, max(inner + 1, radius[metric]))
            unrestricted = outer >= total_distance
            seconds = min(remaining * 0.46, budget * (0.045 + 0.055 * stagnation) *
                          (1.0 + strength / (4.0 + strength)))
            neighborhood_number += 1
        else:
            target_width = min(len(allowed), max(2, int(width * rng.uniform(0.85, 1.15))))
            flip_loss = np.maximum(0.0, np.where(parent, rc, -rc))
            mode = float(rng.random())
            if mode < 0.22 and m:
                alternate_rc = q - costs(dual * np.exp(rng.uniform(-0.5, 0.5, m)))
                flip_loss = np.maximum(0.0, np.where(parent, alternate_rc, -alternate_rc))
            positive = flip_loss[allowed][flip_loss[allowed] > 0]
            offset = float(np.median(positive)) if len(positive) else max(1e-8, float(np.abs(q).mean()))
            keys = flip_loss + offset * (0.02 + 0.15 * stagnation)
            keys *= np.exp(rng.uniform(-0.4 - stagnation, 0.4 + stagnation, k))
            if mode < 0.5:
                # Release first where the relaxation and the parent solution disagree, avoiding choosing the core by a single profit density.
                keys /= 1.0 + (1.0 + strength) * np.abs(lp_x - parent)
            elif archive and mode < 0.85:
                other = archive[int(rng.integers(len(archive)))][1]
                keys[other != parent] *= 1.0 / (2.0 + strength)
            elif mode > 0.94:
                keys[allowed] = rng.random(len(allowed))
            ranked = allowed[np.argsort(keys[allowed], kind='stable')]
            outgoing, incoming = ranked[parent[ranked]], ranked[~parent[ranked]]
            out_count = min(len(outgoing), target_width // 2)
            in_count = min(len(incoming), target_width - out_count)
            out_count = min(len(outgoing), target_width - in_count)
            core = np.concatenate((outgoing[:out_count], incoming[:in_count]))
            seconds = min(remaining * 0.46, budget * (0.03 + 0.045 * stagnation) *
                          (1.0 + math.log1p(failures) / (2.0 + strength)))
            unrestricted = len(core) == len(allowed)
            neighborhood_number += 1
        previous = generation
        complete, accepted = integer_neighborhood(parent, core, seconds, metric, inner, outer)
        improved = generation != previous
        failures = 0 if improved else failures + 1
        if complete and unrestricted:
            break
        if metric is not None:
            if complete and not improved:
                proven[metric] = outer
                radius[metric] = max(outer + 1, int((outer + 1) * (1.1 + strength / (8.0 + strength))))
            elif not complete:
                radius[metric] = max(proven[metric] + 1, int(radius[metric] * 0.88))
        elif not global_due:
            if complete:
                width = min(k, max(width + 1, int(width * 1.18)))
            elif not accepted:
                width = max(min(k, 2 * m + 2), int(width * 0.92))
        if improved:
            width = max(width, width_base)
        if time.perf_counter() >= deadline:
            break

        # Diversified parent solutions are only candidates; no worse or repair-failed rebuild ever overwrites the best solution.
        heuristic_end = min(deadline, time.perf_counter() + budget * 0.012)
        if nonnegative:
            parent = best.copy()
            chosen = np.flatnonzero(parent)
            if len(chosen):
                amplitude = (1.0 + strength) * math.sqrt(len(chosen)) * (0.3 + stagnation)
                count = min(len(chosen), max(1, int(amplitude * rng.uniform(0.4, 1.0))))
                jitter = max(float(np.median(np.abs(rc[chosen]))), 1e-8)
                order = np.argsort(np.maximum(0.0, rc[chosen]) + rng.exponential(jitter, len(chosen)))
                parent[chosen[order[:count]]] = False
            order = density_order(dual * np.exp(rng.uniform(-0.5, 0.5, m)), 0.08 + 0.25 * stagnation)
            offer(repair_fill(parent, order, heuristic_end))
        elif milp is None:
            _plain(instance, deadline, saved)
            break


def solve(instance: dict, time_limit_s: float) -> dict:
    start = time.perf_counter()
    try:
        n = max(0, int(instance['n']))
    except Exception:
        n = 0
    saved = [[0] * n]
    try:
        budget = float(time_limit_s)
        if math.isfinite(budget) and budget > 0:
            _search(instance, budget, start, saved)
    except Exception:
        # If the optional backend or search fails, return only the previously verified complete solution.
        pass
    return {'selection_status': [int(v) for v in saved[0]]}
