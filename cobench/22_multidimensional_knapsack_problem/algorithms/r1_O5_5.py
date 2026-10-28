"""Dual-guided adaptive integer large-neighbourhood search
Chooses the swap core by linear-relaxation dual prices and reduced costs, and improves within the time limit by combining integer branch-and-cut, greedy rebuild and feasible-solution retention.
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
    """Keep performing feasible destroy-repair even when the numerical library is unavailable."""
    n = len(saved[0])
    m = int(instance['m'])
    p = [float(v) for v in instance['profit']]
    w = [[float(v) for v in row] for row in instance['weights']]
    cap = [float(v) for v in instance['capacity']]
    rng = random.Random(104729 + n * 1009 + m)
    best = saved[0][:]
    best_value = sum(p[j] for j in range(n) if best[j])
    prices = [1.0 / max(abs(cap[r]), 1.0) for r in range(m)]
    strength = 1.0 + math.log1p(max(0.0, deadline - time.perf_counter()))
    while time.perf_counter() < deadline:
        x = best[:]
        chosen = [j for j in range(n) if x[j]]
        remove = min(len(chosen), max(1, int(math.sqrt(len(chosen)) * strength * rng.random())))
        for j in rng.sample(chosen, remove):
            x[j] = 0
        load = [sum(w[r][j] for j in range(n) if x[j]) for r in range(m)]
        lam = [v * math.exp(rng.uniform(-0.7, 0.7)) for v in prices]
        order = sorted(range(n), key=lambda j: p[j] * math.exp(rng.uniform(-0.2, 0.2)) /
                       max(1e-15, sum(lam[r] * max(0.0, w[r][j]) for r in range(m))), reverse=True)
        for j in order:
            if time.perf_counter() >= deadline:
                break
            if not x[j] and p[j] > 0 and all(load[r] + w[r][j] <= cap[r] for r in range(m)):
                x[j] = 1
                for r in range(m):
                    load[r] += w[r][j]
        value = sum(p[j] for j in range(n) if x[j])
        if value > best_value and all(load[r] <= cap[r] for r in range(m)):
            best, best_value = x, value
            saved[0] = x[:]


def _search(instance, budget, start, saved):
    deadline = start + 0.90 * budget
    n = len(saved[0])
    m = int(instance['m'])
    if n == 0 or time.perf_counter() >= deadline:
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
    if nonnegative:
        usable = (raw_p > 0) & np.all(raw_a <= raw_b[:, None], axis=0)
    else:
        # Signed data does not apply the standard knapsack pruning rules.
        usable = np.ones(n, dtype=bool)
    ids = np.flatnonzero(usable)
    k = len(ids)
    if not k:
        return
    original = raw_a[:, ids]
    profit = raw_p[ids]
    pscale = max(float(np.max(np.abs(profit))), 1e-100)
    q = profit / pscale
    if m:
        scales = np.maximum(np.abs(raw_b), np.max(np.abs(original), axis=1))
        scales[scales == 0] = 1.0
    else:
        scales = np.ones(0)
    a = original / scales[:, None]
    b = raw_b / scales
    rng = np.random.default_rng(104729 + n * 1009 + m * 9176)
    best = np.zeros(k, dtype=bool)
    best_value = 0.0 if np.all(raw_b >= 0) else -float('inf')
    lp_x = np.zeros(k)
    dual = np.ones(m)
    rc = q.copy()
    upper = float('inf')
    certified = False
    last_gain = time.perf_counter()
    strength = math.log1p(budget)
    width_base = min(k, max(2 * m + 2, int((3.0 + 2.0 * strength) * math.sqrt(k))))
    width = width_base
    archive = []
    archive_limit = max(2, int(2.0 + strength))

    def loads(matrix, x):
        return np.einsum('ij,j->i', matrix, x, optimize=False)

    def costs(lam):
        return np.einsum('ij,i->j', a, lam, optimize=False)

    def offer(x):
        nonlocal best, best_value, last_gain
        x = np.asarray(x, dtype=bool)
        if not np.all(loads(original, x) <= raw_b):
            return False
        value = float(q @ x)
        key = x.tobytes()
        if value > best_value:
            best, best_value = x.copy(), value
            result = [0] * n
            for j in ids[x]:
                result[int(j)] = 1
            saved[0] = result
            last_gain = time.perf_counter()
        if not any(entry[2] == key for entry in archive):
            archive.append((value, x.copy(), key))
            archive.sort(key=lambda entry: entry[0], reverse=True)
            if len(archive) > archive_limit:
                archive.pop()
        return True

    def repair_fill(x, order, stop):
        x = x.copy()
        # Repair and fill in the original units, avoiding reliance on normalised tolerances alone.
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
        cost = costs(lam)
        score = q / np.maximum(cost, 1e-100)
        if noise:
            score *= np.exp(rng.uniform(-noise, noise, k))
        return np.argsort(-score, kind='stable')

    initial_order = density_order(dual)
    offer(repair_fill(best, initial_order, deadline))
    if nonnegative and np.all(loads(original, np.ones(k)) <= raw_b):
        offer(np.ones(k, dtype=bool))
        return
    if time.perf_counter() >= deadline:
        return

    # The duals of the full LP give a valid Lagrangian upper bound and safe reduced-cost fixing.
    if linprog is not None:
        lp_time = min(budget * 0.12, (deadline - time.perf_counter()) * 0.25)
        if lp_time > 0:
            try:
                relaxation = linprog(-q, A_ub=a if m else None, b_ub=b if m else None,
                                     bounds=(0.0, 1.0), method='highs',
                                     options={'time_limit': lp_time, 'presolve': True})
                if relaxation.x is not None:
                    lp_x = np.clip(relaxation.x, 0.0, 1.0)
                    offer(repair_fill(lp_x >= 1.0 - 1e-7, initial_order, deadline))
                if relaxation.success:
                    dual = np.maximum(0.0, -np.asarray(relaxation.ineqlin.marginals))
                    rc = q - costs(dual)
                    upper = float(dual @ b + np.maximum(rc, 0.0).sum())
                    certified = True
                    order = density_order(dual)
                    offer(repair_fill(lp_x >= 1.0 - 1e-7, order, deadline))
                    offer(repair_fill(lp_x >= 0.5, order, deadline))
                    offer(repair_fill(np.zeros(k, dtype=bool), order, deadline))
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

    # Vectorised one-in-one-out is only a short preprocessing step; the main budget goes to multi-item swaps.
    polish_end = min(deadline, time.perf_counter() + budget * 0.025)
    while nonnegative and time.perf_counter() < polish_end:
        selected = np.flatnonzero(best)
        if not len(selected):
            break
        outsiders = np.flatnonzero(~best)
        outsiders = outsiders[np.argsort(-q[outsiders])]
        slack = raw_b - loads(original, best)
        changed = False
        for j in outsiders:
            if time.perf_counter() >= polish_end:
                break
            cheap = selected[q[selected] < q[j]]
            if not len(cheap):
                continue
            fit = np.all(original[:, cheap] + slack[:, None] >= original[:, j, None], axis=0)
            feasible = cheap[fit]
            if len(feasible):
                old = int(feasible[np.argmin(q[feasible])])
                x = best.copy()
                x[old], x[int(j)] = False, True
                value_before = best_value
                offer(repair_fill(x, base_order, polish_end))
                changed = best_value > value_before
                if changed:
                    break
        if not changed:
            break

    def universe():
        if not certified or not math.isfinite(best_value):
            return np.arange(k)
        tolerance = 1e-7 * (1.0 + abs(upper) + float(np.abs(q).sum()))
        gap = max(0.0, upper - best_value) + tolerance
        disagree = best != (rc > 0)
        return np.flatnonzero((np.abs(rc) <= gap) | disagree)

    def integer_core(parent, core, seconds):
        """Fix the parent solution outside the core; the subproblem keeps all original resource constraints."""
        if milp is None or not len(core) or seconds <= 0:
            return False, False
        outside = parent.copy()
        outside[core] = False
        rhs = b - loads(a, outside)
        cq = q[core]
        matrix = np.vstack((a[:, core], cq[None, :]))
        local_value = float(cq @ parent[core])
        lower = np.full(m + 1, -np.inf)
        if np.all(loads(original, parent) <= raw_b):
            lower[-1] = local_value - 1e-9 * (1.0 + abs(local_value))
        upper_constraints = np.concatenate((rhs, [np.inf]))
        # Modelling overhead also counts toward the same deadline, so multiple subproblems do not each reset the total time limit.
        seconds = min(seconds, deadline - time.perf_counter())
        if seconds <= 0:
            return False, False
        try:
            result = milp(c=-cq, integrality=np.ones(len(core), dtype=np.uint8),
                          bounds=Bounds(np.zeros(len(core)), np.ones(len(core))),
                          constraints=LinearConstraint(matrix, lower, upper_constraints),
                          options={'time_limit': seconds, 'mip_rel_gap': 0.0, 'presolve': True,
                                   'disp': False})
            accepted = False
            exact_candidate = False
            if result.x is not None:
                x = outside.copy()
                x[core] = np.asarray(result.x) >= 0.5
                accepted = offer(x)
                exact_candidate = accepted
                if not accepted and nonnegative:
                    accepted = offer(repair_fill(x, base_order, deadline))
            # Use the optimal status only if the optimal integer solution itself passes the original-constraint check.
            return bool(result.status == 0 and exact_candidate), accepted
        except Exception:
            return False, False

    first_global = True
    last_global = start
    failures = 0
    while time.perf_counter() < deadline:
        now = time.perf_counter()
        remaining = deadline - now
        if certified and math.isfinite(best_value):
            gap_tolerance = 1e-9 * (1.0 + abs(upper))
            if upper - best_value <= gap_tolerance:
                break
        allowed = universe()
        if not len(allowed):
            break
        stagnation = min(1.0, (now - last_gain) / max(budget, 1e-100))
        global_due = first_global or (now - last_global > budget * (0.30 + 0.05 * strength))
        if global_due:
            parent = best.copy()
            core = allowed
            seconds = min(remaining * 0.38, budget * (0.20 + 0.025 * strength))
            first_global = False
            last_global = now
        else:
            parent = best.copy()
            if archive and rng.random() < 0.18:
                parent = archive[int(rng.integers(len(archive)))][1].copy()
                # Safe fixing is based on the current best solution; fixed values from an old parent solution are not carried into the model.
                locked = np.ones(k, dtype=bool)
                locked[allowed] = False
                parent[locked] = best[locked]
                if not np.all(loads(original, parent) <= raw_b):
                    parent = best.copy()
            target = min(len(allowed), max(2, int(width * rng.uniform(0.85, 1.15))))
            flip_loss = np.maximum(0.0, np.where(parent, rc, -rc))
            mode = float(rng.random())
            if mode < 0.25 and m:
                perturbed = dual * np.exp(rng.uniform(-0.5, 0.5, m))
                alternate_rc = q - costs(perturbed)
                flip_loss = np.maximum(0.0, np.where(parent, alternate_rc, -alternate_rc))
            positive = flip_loss[allowed][flip_loss[allowed] > 0]
            offset = float(np.median(positive)) if len(positive) else max(1e-8, float(np.abs(q).mean()))
            keys = (flip_loss + offset * (0.015 + 0.15 * stagnation))
            keys *= np.exp(rng.uniform(-0.4 - stagnation, 0.4 + stagnation, k))
            if mode > 0.9:
                keys[allowed] = rng.random(len(allowed))
            ranked = allowed[np.argsort(keys[allowed], kind='stable')]
            outgoing = ranked[parent[ranked]]
            incoming = ranked[~parent[ranked]]
            out_count = min(len(outgoing), target // 2)
            in_count = min(len(incoming), target - out_count)
            out_count = min(len(outgoing), target - in_count)
            core = np.concatenate((outgoing[:out_count], incoming[:in_count]))
            seconds = min(remaining * 0.48, budget * (0.035 + 0.045 * stagnation) *
                          (1.0 + math.log1p(failures) / (2.0 + strength)))
        before = best_value
        full_scope = len(core) == len(allowed)
        completed, accepted = integer_core(parent, core, seconds)
        if completed and full_scope and np.array_equal(parent[np.setdiff1d(np.arange(k), allowed)],
                                                       best[np.setdiff1d(np.arange(k), allowed)]):
            break
        improved = best_value > before
        failures = 0 if improved else failures + 1
        if completed:
            width = min(k, max(width + 1, int(width * 1.20)))
        elif not global_due:
            width = max(min(k, 2 * m + 2), int(width * (0.92 if not accepted else 1.03)))
        if improved:
            width = max(width, width_base)
        if time.perf_counter() >= deadline:
            break

        # Produce different feasible parent solutions between integer searches; the longer the stagnation, the larger the destruction.
        heuristic_end = min(deadline, time.perf_counter() + budget * 0.012)
        if nonnegative:
            parent = best.copy()
            chosen = np.flatnonzero(parent)
            if len(chosen):
                amplitude = (1.0 + strength) * math.sqrt(len(chosen)) * (0.3 + stagnation)
                count = min(len(chosen), max(1, int(amplitude * rng.uniform(0.4, 1.0))))
                badness = np.maximum(0.0, rc[chosen])
                jitter = max(float(np.median(np.abs(rc[chosen]))), 1e-8)
                removal_order = np.argsort(badness + rng.exponential(jitter, len(chosen)))
                parent[chosen[removal_order[:count]]] = False
            lam = dual * np.exp(rng.uniform(-0.5, 0.5, m))
            order = density_order(lam, 0.08 + 0.25 * stagnation)
            offer(repair_fill(parent, order, heuristic_end))
        elif milp is None:
            break
        # Without an integer backend, the main loop is still terminated by the wall clock and keeps performing the rebuild above.

    # Last line of defence in original units with independent summation; items are removed only in the non-negative knapsack domain.
    if nonnegative:
        answer = saved[0]
        for r in range(m):
            total = math.fsum(float(raw_a[r, j]) for j in range(n) if answer[j])
            if total > raw_b[r]:
                chosen = [j for j in range(n) if answer[j] and raw_a[r, j] > 0]
                chosen.sort(key=lambda j: raw_p[j] / raw_a[r, j])
                for j in chosen:
                    answer[j] = 0
                    total = math.fsum(float(raw_a[r, h]) for h in range(n) if answer[h])
                    if total <= raw_b[r]:
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
        # If any optional solve step fails, keep the previously verified complete feasible solution.
        pass
    return {'selection_status': [int(v) for v in saved[0]]}
