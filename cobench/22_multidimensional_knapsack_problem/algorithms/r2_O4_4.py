"""Dual-loss-budget-driven adaptive integer swap search
Combines an integer profit lattice, reduced-cost fixing, limited-swap neighbourhoods and diversified core rebuilds, searching continuously by wall clock and keeping the best solution that passes the original capacity check.
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
    """Time-limited feasible rebuild when optional numerical dependencies are missing."""
    n = len(saved[0])
    m = int(instance['m'])
    p = [float(v) for v in instance['profit']]
    w = [[float(v) for v in row] for row in instance['weights']]
    cap = [float(v) for v in instance['capacity']]
    rng = random.Random(104729 + n * 1009 + m)
    best = saved[0][:]
    value = math.fsum(p[j] for j in range(n) if best[j])
    strength = 1.0 + math.log1p(max(0.0, deadline - time.perf_counter()))
    while time.perf_counter() < deadline:
        x = best[:]
        chosen = [j for j in range(n) if x[j]]
        amount = min(len(chosen), max(1, int(math.sqrt(len(chosen)) * strength * rng.random())))
        for j in rng.sample(chosen, amount):
            x[j] = 0
        load = [math.fsum(w[r][j] for j in range(n) if x[j]) for r in range(m)]
        prices = [math.exp(rng.uniform(-0.6, 0.6)) / max(abs(cap[r]), 1.0) for r in range(m)]
        order = sorted(range(n), key=lambda j: p[j] * math.exp(rng.uniform(-0.2, 0.2)) /
                       max(1e-100, math.fsum(prices[r] * max(0.0, w[r][j]) for r in range(m))),
                       reverse=True)
        for j in order:
            if time.perf_counter() >= deadline:
                break
            if not x[j] and p[j] > 0 and all(load[r] + w[r][j] <= cap[r] for r in range(m)):
                x[j] = 1
                for r in range(m):
                    load[r] += w[r][j]
        candidate = math.fsum(p[j] for j in range(n) if x[j])
        if candidate > value and all(
                max(math.fsum(w[r][j] for j in range(n) if x[j]),
                    sum(w[r][j] for j in range(n) if x[j])) <= cap[r] for r in range(m)):
            best, value = x, candidate
            saved[0] = x[:]


def _search(instance, budget, started, saved):
    deadline = started + 0.92 * budget
    n = len(saved[0])
    m = int(instance['m'])
    if n == 0 or time.perf_counter() >= deadline:
        return
    if m == 0:
        saved[0] = [int(float(v) > 0) for v in instance['profit']]
        return
    if np is None:
        _plain(instance, deadline, saved)
        return

    raw_p = np.asarray(instance['profit'], dtype=float).reshape(n)
    raw_w = np.asarray(instance['weights'], dtype=float).reshape(m, n)
    cap = np.asarray(instance['capacity'], dtype=float).reshape(m)
    if not (np.all(np.isfinite(raw_p)) and np.all(np.isfinite(raw_w)) and np.all(np.isfinite(cap))):
        return
    ordinary = bool(np.all(raw_w >= 0) and np.all(cap >= 0))
    ids = np.flatnonzero((raw_p > 0) & np.all(raw_w <= cap[:, None], axis=0)) if ordinary else np.arange(n)
    k = len(ids)
    if not k:
        return
    w = np.ascontiguousarray(raw_w[:, ids])
    profit = raw_p[ids]
    pscale = max(float(np.max(np.abs(profit))), 1e-100)
    q = profit / pscale
    scales = np.maximum(np.abs(cap), np.max(np.abs(w), axis=1))
    scales[scales == 0] = 1.0
    a = w / scales[:, None]
    b = cap / scales
    rng = np.random.default_rng(104729 + n * 1009 + m * 9176)
    strength = math.log1p(budget)
    best = np.zeros(k, dtype=bool)
    best_profit = 0.0 if np.all(cap >= 0) else -math.inf
    best_value = best_profit
    last_gain = time.perf_counter()
    archive = []
    archive_limit = max(2, int(2 + strength))
    bound_error = 1e-9 * (1.0 + float(np.abs(q).sum()))

    # Integer profits keep integer objective coefficients, so the profit lattice is not lost to normalisation.
    lattice = bool(np.all(profit == np.floor(profit)) and float(np.abs(profit).sum()) < 2.0 ** 50)
    unit = 0
    if lattice:
        for v in profit:
            unit = math.gcd(unit, abs(int(v)))
        lattice = unit > 0
    objective = profit / unit if lattice else q
    objective_factor = pscale / unit if lattice else 1.0

    def loads(matrix, x):
        return np.einsum('ij,j->i', matrix, x, optimize=False)

    def costs(prices):
        return np.einsum('ij,i->j', a, prices, optimize=False)

    def exact_load(x):
        result = loads(w, x)
        # Far from the capacity boundary use the summation error bound; boundary resources are then verified exactly and independently.
        magnitude = loads(np.abs(w), x) if not ordinary else np.abs(result)
        error = (4.0 * k + 16.0) * np.finfo(float).eps * magnitude
        near = np.flatnonzero(result + error >= cap)
        if len(near):
            chosen = np.flatnonzero(x)
            for r in near:
                values = w[int(r), chosen].tolist()
                result[r] = max(math.fsum(values), sum(values), float(result[r]))
        return result

    def offer(x):
        nonlocal best, best_profit, best_value, last_gain
        x = np.asarray(x, dtype=bool)
        value = math.fsum(profit[x].tolist())
        if len(archive) >= archive_limit and value < archive[-1][0]:
            return False
        if np.any(exact_load(x) > cap):
            return False
        key = x.tobytes()
        if value > best_profit:
            best = x.copy()
            best_profit = value
            best_value = value / pscale
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

    def fill(x, order, stop):
        x = x.copy()
        load = exact_load(x)
        while ordinary and np.any(load > cap) and time.perf_counter() < stop:
            chosen = np.flatnonzero(x)
            if not len(chosen):
                break
            bad = load > cap
            excess = np.maximum(load[bad] - cap[bad], scales[bad] * 1e-13)
            relief = np.minimum(w[bad][:, chosen] / excess[:, None], 1.0).sum(axis=0)
            j = int(chosen[np.argmax(relief / np.maximum(q[chosen], 1e-100))])
            x[j] = False
            load = exact_load(x)
        if np.any(load > cap):
            return x
        for j0 in order:
            if time.perf_counter() >= stop:
                break
            j = int(j0)
            if not x[j] and q[j] > 0 and np.all(load + w[:, j] <= cap):
                x[j] = True
                load += w[:, j]
        return x

    def density_order(prices, noise=0.0):
        score = q / np.maximum(costs(prices), 1e-100)
        if noise:
            score *= np.exp(rng.uniform(-noise, noise, k))
        return np.argsort(-score, kind='stable')

    dual = np.ones(m)
    order = density_order(dual)
    offer(fill(best, order, min(deadline, started + budget * 0.08)))
    if ordinary and np.all(exact_load(np.ones(k, dtype=bool)) <= cap):
        offer(np.ones(k, dtype=bool))
        return
    lp_x = np.zeros(k)
    lp_ok = False
    if linprog is not None and time.perf_counter() < deadline:
        try:
            seconds = min(budget * 0.12, (deadline - time.perf_counter()) * 0.22)
            relaxation = linprog(-q, A_ub=a, b_ub=b, bounds=(0.0, 1.0), method='highs',
                                 options={'time_limit': seconds, 'presolve': True})
            if relaxation.x is not None:
                lp_x = np.clip(relaxation.x, 0.0, 1.0)
                offer(fill(lp_x >= 1.0 - 1e-8, order, deadline))
            if relaxation.success:
                dual = np.maximum(0.0, -np.asarray(relaxation.ineqlin.marginals))
                lp_ok = True
        except Exception:
            pass
    if not lp_ok:
        dual = np.maximum(1e-8, np.sum(np.maximum(a, 0.0), axis=1))
        dual /= max(float(dual.sum()), 1e-100)
        cost = costs(dual)
        ratios = q[cost > 0] / cost[cost > 0]
        if len(ratios):
            dual *= max(float(np.median(ratios)), 1e-8)
    rc = q - costs(dual)
    preferred = rc > 0
    positive_rc = np.maximum(rc, 0.0)
    upper = math.fsum((dual * b).tolist()) + math.fsum(positive_rc.tolist())
    bound_error += 1e-9 * (1.0 + abs(upper))
    upper += bound_error
    proof_upper = upper
    order = density_order(dual)
    if lp_ok and time.perf_counter() < deadline:
        offer(fill(lp_x >= 0.5, order, deadline))
        offer(fill(lp_x >= 1.0 - 1e-8, order, deadline))
    if time.perf_counter() < deadline:
        offer(fill(np.zeros(k, dtype=bool), order, deadline))

    # Cheap single swaps provide a tighter profit lower bound for the integer neighbourhood.
    polish_end = min(deadline, time.perf_counter() + budget * 0.02)
    while ordinary and time.perf_counter() < polish_end:
        chosen = np.flatnonzero(best)
        outsiders = np.flatnonzero(~best)
        outsiders = outsiders[np.argsort(-q[outsiders])]
        slack = cap - exact_load(best)
        improved = False
        for j in outsiders:
            if time.perf_counter() >= polish_end:
                break
            cheap = chosen[q[chosen] < q[j]]
            feasible = cheap[np.all(w[:, cheap] + slack[:, None] >= w[:, j, None], axis=0)]
            if len(feasible):
                x = best.copy()
                x[int(feasible[np.argmin(q[feasible])])] = False
                x[int(j)] = True
                old = best_profit
                offer(fill(x, order, polish_end))
                if best_profit > old:
                    improved = True
                    break
        if not improved:
            break

    def target_value():
        if not math.isfinite(best_profit):
            return -math.inf
        if lattice:
            return (round(best_profit / unit) + 1) / objective_factor
        return best_value - bound_error

    def universe(target):
        if not math.isfinite(target):
            return np.arange(k)
        gap = max(0.0, upper - target) + bound_error
        return np.flatnonzero((np.abs(rc) <= gap) | (best != preferred))

    def integer_core(parent, core, seconds, radius, full_scope, target):
        """Fix outside the core; keep the resource constraints and propagate the remaining dual-loss budget."""
        nonlocal proof_upper
        if milp is None or not len(core) or seconds <= 0:
            return False, False
        stop = min(deadline, time.perf_counter() + seconds)
        outside = parent.copy()
        outside[core] = False
        cq = objective[core]
        rows = [a[:, core]]
        lower_rows = [np.full(m, -np.inf)]
        upper_rows = [b - loads(a, outside)]
        lo = np.zeros(len(core))
        hi = np.ones(len(core))
        if math.isfinite(target):
            fixed_objective = math.fsum(objective[outside].tolist())
            cutoff = target * objective_factor - fixed_objective
            if lattice:
                cutoff = round(cutoff) - min(0.125, 1e-9 * (1.0 + abs(cutoff)))
            rows.append(cq[None, :])
            lower_rows.append(np.array([cutoff]))
            upper_rows.append(np.array([np.inf]))
            # Each variable deviating from its dual-preferred value consumes |rc|; the loss outside the core is deducted first.
            losses = np.where(parent, np.maximum(-rc, 0.0), positive_rc)
            fixed_loss = math.fsum(losses.tolist()) - math.fsum(losses[core].tolist())
            loss_budget = upper - target - fixed_loss + bound_error
            if loss_budget < -bound_error:
                return True, False
            forced = np.abs(rc[core]) > loss_budget + bound_error
            lo[forced & preferred[core]] = 1.0
            hi[forced & ~preferred[core]] = 0.0
            rows.append(-rc[core][None, :])
            lower_rows.append(np.array([-np.inf]))
            upper_rows.append(np.array([loss_budget - math.fsum(positive_rc[core].tolist())]))
        if radius is not None:
            selected = parent[core].astype(float)
            rows.append(-selected[None, :])
            lower_rows.append(np.array([-np.inf]))
            upper_rows.append(np.array([float(radius) - float(selected.sum())]))
        matrix = np.vstack(rows)
        constraints = LinearConstraint(matrix, np.concatenate(lower_rows), np.concatenate(upper_rows))
        remaining = stop - time.perf_counter()
        if remaining <= 0:
            return False, False
        try:
            result = milp(-cq, integrality=np.ones(len(core), dtype=np.uint8),
                          bounds=Bounds(lo, hi), constraints=constraints,
                          options={'time_limit': remaining, 'mip_rel_gap': 0.0,
                                   'presolve': True, 'disp': False})
            accepted = False
            valid_optimum = False
            if result.x is not None:
                x = outside.copy()
                x[core] = np.asarray(result.x) >= 0.5
                valid_optimum = bool(np.all(exact_load(x) <= cap))
                accepted = offer(x) if valid_optimum else False
                if ordinary and time.perf_counter() < deadline:
                    accepted = offer(fill(x, order, min(deadline, time.perf_counter() + budget * 0.008))) or accepted
            if full_scope and radius is None:
                value = getattr(result, 'mip_dual_bound', None)
                if value is not None and math.isfinite(float(value)):
                    fixed = math.fsum(q[outside].tolist())
                    proof_upper = min(proof_upper, fixed - float(value) / objective_factor + bound_error)
            completed = bool((result.status == 0 and valid_optimum) or
                             (result.status == 2 and lattice and math.isfinite(target)))
            return completed, accepted
        except Exception:
            return False, False

    width_base = min(k, max(2 * m + 2, int((2.5 + 2.0 * strength) * math.sqrt(k))))
    width = width_base
    radius_base = max(2, int(math.sqrt(m + 1) * (1.0 + 0.5 * strength)))
    radius = radius_base
    failures = 0
    first_global = True
    last_global = started
    branch_due = False
    while time.perf_counter() < deadline:
        now = time.perf_counter()
        remaining = deadline - now
        target = target_value()
        if lattice and math.isfinite(target) and proof_upper < target - bound_error:
            break
        if not lattice and math.isfinite(best_value) and proof_upper <= best_value + bound_error:
            break
        allowed = universe(target)
        if not len(allowed):
            break
        stagnation = min(1.0, (now - last_gain) / max(budget, 1e-100))
        global_due = first_global or now - last_global > budget * (0.28 + 0.045 * strength)
        parent = best.copy()
        local_radius = None
        if global_due:
            core = allowed
            seconds = min(remaining * 0.38, budget * (0.17 + 0.035 * strength))
            first_global = False
            last_global = now
        elif branch_due and len(allowed) > width:
            # Limit the number removed but not the number inserted, allowing compensating swaps outside the core.
            core = allowed
            local_radius = radius
            seconds = min(remaining * 0.43, budget * (0.035 + 0.04 * stagnation))
            branch_due = False
        else:
            target_width = min(len(allowed), max(2, int(width * rng.uniform(0.85, 1.15))))
            alternate = rc
            mode = float(rng.random())
            if mode < 0.3:
                alternate = q - costs(dual * np.exp(rng.uniform(-0.45, 0.45, m)))
            flip_loss = np.where(parent, alternate, -alternate)
            spread = max(float(np.median(np.abs(flip_loss[allowed]))), 1e-9)
            keys = flip_loss * np.exp(rng.uniform(-0.35 - stagnation, 0.35 + stagnation, k))
            keys += rng.exponential(spread * (0.015 + 0.12 * stagnation), k)
            if mode > 0.88:
                keys[allowed] = rng.random(len(allowed)) * spread
            # The differing variables of elite solutions form recombinable swap blocks.
            if len(archive) > 1 and 0.3 <= mode < 0.55:
                elite = archive[int(rng.integers(1, len(archive)))][1]
                keys[parent != elite] -= (1.0 + strength) * spread
            ranked = allowed[np.argsort(keys[allowed], kind='stable')]
            outgoing = ranked[parent[ranked]]
            incoming = ranked[~parent[ranked]]
            out_count = min(len(outgoing), target_width // 2)
            in_count = min(len(incoming), target_width - out_count)
            out_count = min(len(outgoing), target_width - in_count)
            core = np.concatenate((outgoing[:out_count], incoming[:in_count]))
            seconds = min(remaining * 0.43, budget * (0.027 + 0.04 * stagnation) *
                          (1.0 + math.log1p(failures) / (3.0 + strength)))
            branch_due = True
        before = best_profit
        full_scope = len(core) == len(allowed)
        completed, accepted = integer_core(parent, core, seconds, local_radius, full_scope, target)
        if completed and full_scope and local_radius is None:
            break
        improved = best_profit > before
        failures = 0 if improved else failures + 1
        if local_radius is not None:
            if completed:
                radius = min(k, max(radius + 1, int(radius * (1.0 + 0.4 / (1.0 + strength)))))
            elif not accepted:
                radius = max(2, int(radius * 0.8))
            if improved:
                radius = max(radius, radius_base)
        elif not global_due:
            if completed:
                width = min(k, max(width + 1, int(width * (1.0 + 0.65 / (1.0 + strength)))))
            elif not accepted:
                width = max(min(k, m + 2), int(width * 0.86))
            if improved:
                width = max(width, width_base)
        if time.perf_counter() >= deadline:
            break
        # The rebuild phase also runs by wall clock, with no limit on the number of restarts.
        heuristic_end = min(deadline, time.perf_counter() + budget * (0.008 + 0.01 * stagnation))
        if ordinary:
            while time.perf_counter() < heuristic_end:
                x = best.copy()
                if archive and rng.random() < 0.2:
                    x = archive[int(rng.integers(len(archive)))][1].copy()
                chosen = np.flatnonzero(x)
                if len(chosen):
                    amplitude = (1.0 + strength) * math.sqrt(len(chosen)) * (0.2 + stagnation)
                    count = min(len(chosen), max(1, int(amplitude * rng.uniform(0.3, 1.0))))
                    jitter = max(float(np.median(np.abs(rc[chosen]))), 1e-9)
                    keys = rc[chosen] + rng.exponential(jitter * (0.2 + stagnation), len(chosen))
                    x[chosen[np.argsort(keys)[:count]]] = False
                prices = dual * np.exp(rng.uniform(-0.45, 0.45, m))
                offer(fill(x, density_order(prices, 0.05 + 0.25 * stagnation), heuristic_end))
        elif milp is None:
            break


def solve(instance: dict, time_limit_s: float) -> dict:
    started = time.perf_counter()
    try:
        n = max(0, int(instance['n']))
    except Exception:
        n = 0
    saved = [[0] * n]
    try:
        budget = float(time_limit_s)
        if math.isfinite(budget) and budget > 0:
            _search(instance, budget, started, saved)
    except Exception:
        # Failure of an optional step does not overwrite the previously verified complete feasible solution.
        pass
    return {'selection_status': [int(v) for v in saved[0]]}
