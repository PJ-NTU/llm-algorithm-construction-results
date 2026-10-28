"""Dual pricing and adaptive integer large-neighbourhood knapsack algorithm
Combines the linear relaxation, safe reduced-cost fixing, time-limited integer neighbourhood search and perturbation-refill, continuously keeping the best feasible solution verified against the original constraints.
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


def _basic(instance, n, deadline, budget, answer):
    """Independent wall-clock-driven fallback search when the numerical solver is unavailable."""
    try:
        p = [float(v) for v in instance["profit"]]
        a = [[float(v) for v in row] for row in instance["weights"]]
        b = [float(v) for v in instance["capacity"]]
        m = int(instance["m"])
        rng = random.Random(6197 + n * 31 + m)
        best = sum(p[j] for j in range(n) if answer[j])
        nonnegative = all(v >= 0 for row in a for v in row)
        ids = [j for j in range(n) if p[j] > 0 and
               (not nonnegative or all(a[r][j] <= b[r] for r in range(m)))]
        intensity = 1.0 + math.log1p(budget)
        while time.perf_counter() < deadline:
            prices = [math.exp(rng.uniform(-intensity, intensity)) /
                      max(abs(b[r]), 1e-12) for r in range(m)]
            order = sorted(ids, key=lambda j: p[j] * rng.uniform(0.75, 1.25) /
                           max(1e-15, sum(prices[r] * a[r][j] for r in range(m))),
                           reverse=True)
            x = answer[:] if rng.random() < 0.8 else [0] * n
            selected = [j for j in ids if x[j]]
            if selected:
                count = min(len(selected), 1 + int(rng.random() *
                            math.sqrt(len(selected)) * intensity))
                for j in rng.sample(selected, count):
                    x[j] = 0
            loads = [math.fsum(a[r][j] for j in range(n) if x[j]) for r in range(m)]
            if any(loads[r] > b[r] for r in range(m)):
                continue
            for j in order:
                if time.perf_counter() >= deadline:
                    break
                if not x[j] and all(loads[r] + a[r][j] <= b[r] for r in range(m)):
                    x[j] = 1
                    for r in range(m):
                        loads[r] += a[r][j]
            value = math.fsum(p[j] for j in range(n) if x[j])
            if value > best and all(math.fsum(a[r][j] for j in range(n) if x[j])
                                    <= b[r] for r in range(m)):
                answer, best = x, value
    except Exception:
        pass
    return {"selection_status": answer}


def solve(instance: dict, time_limit_s: float) -> dict:
    start = time.perf_counter()
    try:
        budget = max(0.0, float(time_limit_s))
        if not math.isfinite(budget):
            budget = 0.0
    except Exception:
        budget = 0.0
    try:
        n = max(0, int(instance["n"]))
    except Exception:
        n = 0
    answer = [0] * n
    deadline = start + budget * 0.94
    if n == 0 or budget <= 0:
        return {"selection_status": answer}
    if np is None:
        return _basic(instance, n, deadline, budget, answer)

    try:
        m = int(instance["m"])
        raw_p = np.asarray(instance["profit"], dtype=float).reshape(n)
        raw_a = np.asarray(instance["weights"], dtype=float).reshape(m, n)
        raw_b = np.asarray(instance["capacity"], dtype=float).reshape(m)
        if not (np.isfinite(raw_p).all() and np.isfinite(raw_a).all()
                and np.isfinite(raw_b).all()):
            return {"selection_status": answer}
        if m == 0:
            return {"selection_status": (raw_p > 0).astype(int).tolist()}

        # Apply single-item exclusion only under non-negative consumption; signed coefficients keep the original model as is.
        ordinary = bool(np.all(raw_a >= 0) and np.all(raw_b >= 0))
        if ordinary:
            ids = np.flatnonzero((raw_p > 0) & np.all(raw_a <= raw_b[:, None], axis=0))
        else:
            ids = np.arange(n)
        k = len(ids)
        if k == 0:
            return {"selection_status": answer}
        p = raw_p[ids]
        original = raw_a[:, ids]
        scale = np.maximum(np.maximum(np.abs(raw_b), np.max(np.abs(original), axis=1)),
                           1e-100)
        a = original / scale[:, None]
        b = raw_b / scale
        w = np.ascontiguousarray(a.T)
        pscale = max(float(np.max(np.abs(p))), 1e-100)
        pc = p / pscale
        rng = np.random.default_rng(7817 + 131 * n + 17 * m)
        strength = math.log1p(budget)
        elite_limit = max(2, int(2 + 2 * math.sqrt(strength)))
        best_x = np.zeros(k, dtype=np.int8)
        best_value = 0.0 if np.all(raw_b >= 0) else -math.inf
        elites = []
        price = np.maximum(1e-12, np.sum(np.maximum(a, 0), axis=1))
        price *= pscale / max(float(np.sum(price)), 1e-100)
        reduced = p - a.T @ price
        lp_x = None
        dual_upper = math.inf
        milp_enabled = milp is not None
        improved_at = start

        def exact_feasible(x):
            used = np.flatnonzero(x)
            loads = original @ x
            if np.any(loads > raw_b + 1e-12 * scale):
                return False
            # Re-check with compensated summation only the resources where rounding error could affect the decision.
            near = np.flatnonzero(np.abs(loads - raw_b) <= 1e-10 * scale)
            for r in near:
                values = [float(original[r, j]) for j in used]
                if math.fsum(values) > raw_b[r] or sum(values) > raw_b[r]:
                    return False
            return True

        def repair(x):
            x = np.asarray(x, dtype=np.int8).copy()
            if exact_feasible(x):
                return x
            if not ordinary:
                return None
            while time.perf_counter() < deadline:
                load = a @ x
                bad = np.flatnonzero(load > b)
                if len(bad) == 0:
                    if exact_feasible(x):
                        return x
                    bad = np.flatnonzero(load >= b - 1e-11)
                chosen = np.flatnonzero(x)
                if len(chosen) == 0:
                    return x
                relief = np.sum(np.maximum(a[bad][:, chosen], 0), axis=0)
                if not np.any(relief > 0):
                    return np.zeros(k, dtype=np.int8)
                score = np.maximum(p[chosen], 1e-100) / np.maximum(relief, 1e-100)
                x[chosen[int(np.argmin(score))]] = 0
                if exact_feasible(x):
                    return x
            return np.zeros(k, dtype=np.int8)

        def remember(x):
            nonlocal best_x, best_value, answer, improved_at
            if x is None:
                return False
            value = float(p @ x)
            threshold = elites[-1][0] if len(elites) >= elite_limit else -math.inf
            if value <= threshold and value <= best_value:
                return False
            if not exact_feasible(x):
                return False
            improved = value > best_value
            if improved:
                best_value, best_x = value, x.copy()
                full = np.zeros(n, dtype=np.int8)
                full[ids] = x
                answer = full.astype(int).tolist()
                improved_at = time.perf_counter()
            sig = x.tobytes()
            if all(sig != e[2] for e in elites):
                elites.append((value, x.copy(), sig))
                elites.sort(key=lambda e: e[0], reverse=True)
                del elites[elite_limit:]
            return improved

        def fill(x, order, stop, forbidden=None):
            x = x.copy()
            load = a @ x
            if np.any(load > b):
                return repair(x)
            for pos, j in enumerate(order):
                if pos % 32 == 0 and time.perf_counter() >= stop:
                    break
                if x[j] or p[j] <= 0 or (forbidden is not None and forbidden[j]):
                    continue
                trial = load + w[j]
                if np.all(trial <= b):
                    x[j] = 1
                    load = trial
            return repair(x)

        def ranking(prices, noise=0.0):
            cost = a.T @ prices
            score = p / np.maximum(cost, pscale * 1e-14)
            if noise:
                score *= np.exp(rng.uniform(-noise, noise, k))
            return np.argsort(-score, kind="stable")

        # Build a complete feasible fallback solution before any more expensive solve.
        order = ranking(price)
        remember(fill(best_x, order, min(deadline, start + budget * 0.055)))
        if time.perf_counter() >= deadline:
            return {"selection_status": answer}

        # The dual upper bound uses an explicit Lagrangian expression; safe fixing does not rely on guessing the sign of numerical reduced costs.
        if linprog is not None:
            try:
                seconds = min(deadline - time.perf_counter(), budget * 0.12)
                lp = linprog(-pc, A_ub=a, b_ub=b, bounds=(0, 1), method="highs",
                             options={"time_limit": max(1e-5, seconds), "presolve": True})
                if lp.x is not None and np.isfinite(lp.x).all():
                    lp_x = np.clip(lp.x, 0, 1)
                    if lp.status == 0:
                        dual = np.maximum(0.0, -np.asarray(lp.ineqlin.marginals)) * pscale
                        if np.isfinite(dual).all():
                            price = dual
                            reduced = p - a.T @ price
                            dual_upper = float(b @ price + np.sum(np.maximum(reduced, 0)))
                    order = ranking(price)
                    rounded = repair((lp_x >= 1 - 1e-8).astype(np.int8))
                    if rounded is not None:
                        remember(fill(rounded, order, deadline))
                    remember(fill(np.zeros(k, dtype=np.int8), np.argsort(-lp_x), deadline))
            except Exception:
                pass

        bound_guard = 1e-8 * max(pscale, float(np.sum(np.abs(p))))
        proven = False

        def integer_search(parent, free, seconds, global_search):
            nonlocal milp_enabled, proven
            if not milp_enabled or seconds <= 0 or time.perf_counter() >= deadline:
                return False, False
            fixed = parent.copy()
            fixed[free] = 0
            residual = b - a @ fixed
            fixed_value = float(p @ fixed)
            sub_a = a[:, free]
            sub_p = pc[free]
            lower = np.full(m, -np.inf)
            upper = residual
            # The objective lower bound keeps the current best solution and only excludes combinations that cannot bring improvement.
            if math.isfinite(best_value):
                sub_a = np.vstack((sub_a, sub_p))
                lower = np.append(lower, (best_value - fixed_value - bound_guard) / pscale)
                upper = np.append(upper, np.inf)
            remaining = min(seconds, deadline - time.perf_counter())
            if remaining <= 0:
                return False, False
            try:
                result = milp(-sub_p, integrality=np.ones(len(free), dtype=np.uint8),
                              bounds=Bounds(np.zeros(len(free)), np.ones(len(free))),
                              constraints=LinearConstraint(sub_a, lower, upper),
                              options={"time_limit": remaining, "mip_rel_gap": 0.0,
                                       "presolve": True, "disp": False})
            except Exception:
                milp_enabled = False
                return False, False
            changed = False
            if result.x is not None and np.isfinite(result.x).all():
                trial = fixed.copy()
                trial[free] = (result.x >= 0.5).astype(np.int8)
                changed = remember(repair(trial))
            complete = result.status in (0, 2)
            if global_search and result.status == 0 and result.fun is not None:
                # If numerical repair lost objective value, the solve is not treated as proven optimal.
                objective = fixed_value - float(result.fun) * pscale
                if best_value >= objective - 1e-10 * pscale:
                    proven = True
            return changed, complete

        def available_core():
            if not math.isfinite(dual_upper) or not math.isfinite(best_value):
                return np.arange(k)
            gap = max(0.0, dual_upper - best_value) + bound_guard
            fixed_one = (reduced > gap) & (best_x == 1)
            fixed_zero = (reduced < -gap) & (best_x == 0)
            return np.flatnonzero(~(fixed_one | fixed_zero))

        # The first global integer search keeps long-range swap opportunities that the reference algorithm does not cover.
        free = available_core()
        if len(free) and time.perf_counter() < deadline:
            integer_search(best_x, free, min(budget * 0.22,
                           max(0.0, deadline - time.perf_counter())), True)
        if proven:
            return {"selection_status": answer}

        # Neighbourhood width, time slice and perturbation strength all depend on both instance size and the given time limit.
        core_size = min(k, max(8, int(8 + 2.2 * math.sqrt(k * max(strength, 0.01))
                                     + math.sqrt(m) * strength)))
        slice_base = budget / (10.0 + math.sqrt(k))
        stagnation = 0
        while time.perf_counter() < deadline:
            free = available_core()
            if len(free) == 0:
                break
            if elites and rng.random() < 0.25:
                e = min(len(elites) - 1, int((rng.random() ** 2) * len(elites)))
                parent = elites[e][1].copy()
            else:
                parent = best_x.copy()
            loads = a @ parent
            slack = np.maximum(0.02, b - loads)
            dynamic_price = (price + pscale * 1e-8) / np.sqrt(slack)
            dynamic_price *= np.exp(rng.uniform(-1.0, 1.0, m))
            order = ranking(dynamic_price, 0.1 + 0.15 * strength)
            chosen = np.flatnonzero(parent)
            if len(chosen):
                max_drop = min(len(chosen), max(1, int(math.sqrt(len(chosen)) *
                                                     (0.5 + math.sqrt(strength)))))
                drop = int(rng.integers(1, max_drop + 1))
                if rng.random() < 0.7:
                    burden = a[:, chosen].T @ dynamic_price
                    merit = p[chosen] / np.maximum(burden, pscale * 1e-14)
                    merit *= np.exp(rng.uniform(-0.7, 0.7, len(chosen)))
                    removed = chosen[np.argsort(merit)[:drop]]
                else:
                    removed = rng.choice(chosen, drop, replace=False)
                parent[removed] = 0
            else:
                removed = np.empty(0, dtype=int)
            forbidden = np.zeros(k, dtype=bool)
            if len(removed) and rng.random() < 0.6:
                forbidden[removed[int(rng.integers(len(removed)))]] = True
            changed = remember(fill(parent, order, deadline, forbidden))
            if time.perf_counter() >= deadline:
                break

            # Also try feasible one-for-one swaps, offsetting start-up overhead when the integer subproblem is very short.
            swap_stop = min(deadline, time.perf_counter() + slice_base * 0.12)
            work = best_x.copy()
            load = a @ work
            chosen = np.flatnonzero(work)
            omitted = np.flatnonzero((work == 0) & (p > 0))
            for j in omitted[np.argsort(-p[omitted])]:
                if time.perf_counter() >= swap_stop:
                    break
                eligible = chosen[p[chosen] < p[j]]
                if len(eligible) == 0:
                    continue
                fits = np.all(load + w[j] - w[eligible] <= b, axis=1)
                eligible = eligible[fits]
                if len(eligible):
                    out = eligible[int(np.argmin(p[eligible]))]
                    work[out], work[j] = 0, 1
                    changed = remember(fill(work, order, deadline)) or changed
                    break
            if not milp_enabled:
                continue
            if time.perf_counter() >= deadline:
                break

            parent = best_x.copy()
            count = min(len(free), max(2, int(core_size)))
            # Expand to the global core as stagnation grows, rather than staying fixed in one local neighbourhood forever.
            elapsed = time.perf_counter() - improved_at
            full_search = count == len(free) or rng.random() < min(
                0.45, 0.08 + elapsed / max(budget, 1e-12))
            if full_search:
                core = free
            else:
                gap_score = np.abs(reduced[free])
                typical = max(float(np.median(gap_score)), pscale * 1e-7)
                if rng.random() < 0.55:
                    score = gap_score / typical + rng.exponential(0.8, len(free))
                else:
                    # Exponential random priorities give items far from the relaxation boundary a non-zero chance of entering the core.
                    score = rng.exponential(1.0, len(free)) * (0.1 + gap_score / typical)
                if lp_x is not None:
                    score *= np.maximum(0.12, 1.0 - np.abs(parent[free] - lp_x[free]))
                if len(elites) > 1 and rng.random() < 0.5:
                    other = elites[int(rng.integers(1, len(elites)))][1]
                    score[parent[free] != other[free]] *= 0.15
                core = free[np.argpartition(score, count - 1)[:count]]
                # Both sides may change, avoiding an ineffective neighbourhood containing only selected items.
                for state in (0, 1):
                    side = free[parent[free] == state]
                    needed = min(len(side), max(1, count // 4))
                    if needed:
                        priority = np.abs(reduced[side]) * rng.uniform(0.7, 1.3, len(side))
                        core = np.union1d(core, side[np.argsort(priority)[:needed]])
            seconds = slice_base * (1.0 + math.sqrt(stagnation + 1) * 0.18)
            if full_search:
                seconds *= 1.8 + math.sqrt(strength)
            gain, complete = integer_search(parent, core, seconds, full_search)
            changed = changed or gain
            if proven:
                break
            if changed:
                stagnation = 0
                core_size = min(k, max(core_size, len(core) * 0.95))
            else:
                stagnation += 1
            if not full_search:
                if complete:
                    core_size = min(k, core_size * (1.08 + 0.03 * math.sqrt(strength)))
                elif not gain:
                    core_size = max(8, core_size * 0.88)
        return {"selection_status": answer}
    except Exception:
        # answer is updated only after the original-constraint re-check passes; exceptions never overwrite the existing best solution.
        return {"selection_status": answer}
