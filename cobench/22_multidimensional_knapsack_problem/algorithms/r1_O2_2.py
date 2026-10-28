"""Linear-relaxation-guided adaptive branch-and-cut search
Builds a feasible initial solution from the relaxation, alternates between solving the full model and full-variable Hamming neighbourhoods, and keeps the best solution re-checked against the original constraints.
"""

import math
import time


def solve(instance: dict, time_limit_s: float) -> dict:
    started = time.perf_counter()
    answer = []
    try:
        n = max(0, int(instance["n"]))
        answer = [0] * n
        budget = float(time_limit_s)
        if not math.isfinite(budget) or budget <= 0 or n == 0:
            return {"selection_status": answer}
        # Reserve time for result re-checking, data conversion and solver shutdown.
        deadline = started + budget * 0.94
        import numpy as np

        m = int(instance["m"])
        original_profit = np.asarray(instance["profit"], dtype=float).reshape(n)
        original_weight = np.asarray(instance["weights"], dtype=float).reshape(m, n)
        capacity = np.asarray(instance["capacity"], dtype=float).reshape(m)
        if not (np.isfinite(original_profit).all()
                and np.isfinite(original_weight).all()
                and np.isfinite(capacity).all()):
            return {"selection_status": answer}
        if m == 0:
            return {"selection_status": (original_profit > 0).astype(int).tolist()}

        nonnegative = bool((original_weight >= 0).all() and (capacity >= 0).all())
        # Apply these two safe reductions only for non-negative consumption; signed data keeps all variables.
        if nonnegative:
            usable = (original_profit > 0) & (original_weight <= capacity[:, None]).all(axis=0)
            indices = np.flatnonzero(usable)
        else:
            indices = np.arange(n)
        k = len(indices)
        if k == 0:
            return {"selection_status": answer}
        p = original_profit[indices]
        w = original_weight[:, indices]
        best = np.zeros(k, dtype=np.int8)
        best_value = 0.0 if (capacity >= 0).all() else -math.inf
        row_scale = np.maximum(1.0, np.maximum(np.abs(capacity), np.max(np.abs(w), axis=1)))
        a = w / row_scale[:, None]
        b = capacity / row_scale
        profit_scale = max(1.0, float(np.max(np.abs(p))))
        cost = p / profit_scale
        positive = p > 0
        now = time.perf_counter
        rng = np.random.default_rng((n * 1000003 + m * 9176) & 0xFFFFFFFF)

        def feasible(y):
            load = w @ y
            bad = np.flatnonzero(load > capacity)
            if len(bad):
                chosen = np.flatnonzero(y)
                # The last digits of a dot product may change with summation order; re-check boundary cases with exact accumulation.
                for r in bad:
                    if math.fsum(float(w[r, j]) for j in chosen) > float(capacity[r]):
                        return False
            return True

        def remember(y):
            nonlocal best, best_value, answer
            if y is None or len(y) != k:
                return False
            values = np.asarray(y, dtype=float)
            if not np.isfinite(values).all():
                return False
            rounded = np.rint(values)
            if np.max(np.abs(values - rounded)) > 1e-5:
                return False
            if (rounded < 0).any() or (rounded > 1).any():
                return False
            candidate = rounded.astype(np.int8)
            value = float(p @ candidate)
            if value <= best_value or not feasible(candidate):
                return False
            best = candidate.copy()
            best_value = value
            result = [0] * n
            for j in np.flatnonzero(best):
                result[int(indices[j])] = 1
            answer = result
            return True

        # The first feasible solution does not depend on the optimisation library; it is kept through any later exception.
        base_cost = np.sum(np.maximum(a, 0), axis=0)
        density = p / np.maximum(base_cost, 1e-12)
        default_order = np.argsort(-density, kind="stable")

        def repair_fill(y, order):
            y = np.asarray(y, dtype=np.int8).copy()
            load = w @ y
            while (load > capacity).any():
                if now() >= deadline:
                    return
                if not nonnegative:
                    # Under signed constraints, removing items does not necessarily restore feasibility; leave it to the full integer model.
                    remember(y)
                    return
                bad = load > capacity
                relief = np.sum(np.maximum(a[bad], 0), axis=0)
                score = np.where(y != 0, relief / np.maximum(p, 1e-12), -1.0)
                outgoing = int(np.argmax(score))
                if score[outgoing] <= 0:
                    return
                y[outgoing] = 0
                load -= w[:, outgoing]
            for pos, j in enumerate(order):
                if (pos & 31) == 0 and now() >= deadline:
                    break
                if y[j] or not positive[j]:
                    continue
                if np.all(load + w[:, j] <= capacity):
                    y[j] = 1
                    load += w[:, j]
            remember(y)

        repair_fill(best, default_order)
        if now() >= deadline:
            return {"selection_status": answer}
        if nonnegative and feasible(np.ones(k, dtype=np.int8)):
            remember(np.ones(k, dtype=np.int8))
            return {"selection_status": answer}

        from scipy.optimize import Bounds, LinearConstraint, linprog, milp
        from scipy.sparse import csr_matrix, vstack

        matrix = csr_matrix(a)
        matrix.indices = matrix.indices.astype(np.int32)
        matrix.indptr = matrix.indptr.astype(np.int32)
        try:
            from scipy.optimize._highspy import _core as highs
            native = True
        except Exception:
            highs = None
            native = False
        effort = math.log1p(budget)
        call_number = 0

        def run_model(seconds, center=None, radius=None, relaxation=False):
            """Limited only by the wall clock each time; the original profit is always the sole optimisation objective."""
            nonlocal native, call_number
            stop = min(deadline, now() + max(0.0, seconds))
            if stop <= now():
                return None, False, None, None
            call_number += 1
            if native:
                try:
                    h = highs._Highs()
                    h.setOptionValue("output_flag", False)
                    h.setOptionValue("threads", 1)
                    h.setOptionValue("random_seed", int(call_number % 2147483647))
                    h.setOptionValue("mip_rel_gap", 0.0)
                    h.setOptionValue("mip_abs_gap", 0.0)
                    h.setOptionValue("mip_feasibility_tolerance", 1e-8)
                    h.setOptionValue("primal_feasibility_tolerance", 1e-8)
                    h.setOptionValue("mip_heuristic_effort", 0.1 + 0.1 * effort / (1.0 + effort))
                    model = highs.HighsLp()
                    model.num_col_ = k
                    model.num_row_ = m
                    model.col_cost_ = -cost
                    model.col_lower_ = np.zeros(k)
                    model.col_upper_ = np.ones(k)
                    model.row_lower_ = np.full(m, -np.inf)
                    model.row_upper_ = b
                    model.a_matrix_.format_ = highs.MatrixFormat.kRowwise
                    model.a_matrix_.num_col_ = k
                    model.a_matrix_.num_row_ = m
                    model.a_matrix_.start_ = matrix.indptr
                    model.a_matrix_.index_ = matrix.indices
                    model.a_matrix_.value_ = matrix.data
                    if not relaxation:
                        model.integrality_ = [highs.HighsVarType.kInteger] * k
                    else:
                        h.setOptionValue("solver", "simplex")
                    status = h.passModel(model)
                    if status == highs.HighsStatus.kError:
                        raise ValueError("solver did not accept the model")
                    if center is not None:
                        # sum(unselected variables) + sum(1 - selected variables) <= radius.
                        coeff = 1.0 - 2.0 * center
                        h.addRow(-np.inf, float(radius - np.sum(center)), k,
                                 np.arange(k, dtype=np.int32), coeff)
                    if not relaxation and math.isfinite(best_value):
                        h.setSolution(k, np.arange(k, dtype=np.int32), best.astype(float))

                    def callback(kind, message, data_out, data_in, user_data):
                        try:
                            if now() >= stop:
                                data_in.user_interrupt = True
                            if kind == highs.cb.kCallbackMipImprovingSolution:
                                values = data_out.mip_solution
                                if hasattr(values, "to_array"):
                                    values = values.to_array(k)
                                remember(values)
                        except Exception:
                            pass

                    h.setCallback(callback, None)
                    h.startCallback(highs.cb.kCallbackMipInterrupt)
                    h.startCallback(highs.cb.kCallbackSimplexInterrupt)
                    h.startCallback(highs.cb.kCallbackIpmInterrupt)
                    if not relaxation:
                        h.startCallback(highs.cb.kCallbackMipImprovingSolution)
                    remaining = stop - now()
                    if remaining <= 0:
                        return None, False, None, None
                    h.setOptionValue("time_limit", remaining)
                    h.run()
                    solution = h.getSolution()
                    values = np.asarray(solution.col_value, dtype=float) if solution.value_valid else None
                    optimal = h.getModelStatus() == highs.HighsModelStatus.kOptimal
                    dual = None
                    upper = None
                    if relaxation:
                        if optimal and values is not None:
                            upper = float(p @ values)
                            if solution.dual_valid:
                                dual = -np.asarray(solution.row_dual, dtype=float)[:m]
                    else:
                        remember(values)
                        bound = float(h.getInfo().mip_dual_bound)
                        if math.isfinite(bound):
                            upper = -bound * profit_scale
                    return values, optimal, upper, dual
                except Exception:
                    # Some SciPy versions do not provide the embedded warm-start interface; fall back to the public interface.
                    native = False
            remaining = stop - now()
            if remaining <= 0:
                return None, False, None, None
            try:
                if relaxation:
                    result = linprog(-cost, A_ub=matrix, b_ub=b, bounds=(0.0, 1.0),
                                     method="highs", options={"time_limit": remaining})
                    optimal = result.status == 0
                    dual = -np.asarray(result.ineqlin.marginals) if optimal else None
                    upper = -float(result.fun) * profit_scale if optimal else None
                    return result.x, optimal, upper, dual
                rows = matrix
                lower = np.full(m, -np.inf)
                upper_rows = b
                if center is not None:
                    rows = vstack([matrix, csr_matrix((1.0 - 2.0 * center)[None, :])], format="csr")
                    lower = np.append(lower, -np.inf)
                    upper_rows = np.append(b, float(radius - np.sum(center)))
                remaining = stop - now()
                if remaining <= 0:
                    return None, False, None, None
                result = milp(-cost, integrality=np.ones(k), bounds=Bounds(0.0, 1.0),
                              constraints=LinearConstraint(rows, lower, upper_rows),
                              options={"time_limit": remaining, "mip_rel_gap": 0.0, "disp": False})
                remember(result.x)
                bound = getattr(result, "mip_dual_bound", None)
                upper = -float(bound) * profit_scale if bound is not None and math.isfinite(bound) else None
                return result.x, result.status == 0, upper, None
            except Exception:
                return None, False, None, None

        lp, lp_optimal, global_upper, dual = run_model(budget * 0.11, relaxation=True)
        if dual is not None and len(dual) == m:
            resource_cost = np.maximum(a.T @ np.maximum(dual, 0.0), 1e-12)
            density = cost / resource_cost
            default_order = np.argsort(-density, kind="stable")
        if lp is not None and len(lp) == k:
            lp = np.clip(lp, 0.0, 1.0)
            repair_fill((lp >= 1.0 - 1e-7).astype(np.int8), default_order)
            if now() < deadline:
                repair_fill((lp >= 0.5).astype(np.int8), default_order)
        else:
            lp = best.astype(float)

        integer_profit = bool(np.all(p == np.rint(p)))

        def proven(upper):
            if upper is None or not math.isfinite(best_value):
                return False
            if integer_profit:
                return math.floor(upper + 1e-6) <= best_value
            return upper <= best_value + 1e-10 * max(1.0, abs(best_value))

        if proven(global_upper):
            return {"selection_status": answer}

        # Randomised rounding takes only one short budget-allocated slot, with no cap on the number of attempts.
        rounding_stop = min(deadline, now() + budget * 0.025)
        while now() < rounding_stop:
            perturbed = density * np.exp(rng.normal(0.0, 1.0 / math.sqrt(1.0 + effort), k))
            order = np.argsort(-perturbed, kind="stable")
            repair_fill((rng.random(k) < lp).astype(np.int8), order)

        radius_base = min(k, max(2, int(math.ceil(1.8 * math.sqrt(max(1, m)) * math.sqrt(1.0 + effort)))))
        radius = radius_base
        quantum = budget / (4.0 + math.log2(2.0 + k) / math.sqrt(1.0 + effort))
        next_global = now() + min(quantum, budget * 0.10)
        last_gain = now()
        # All variables can always take part in replacement; only the Hamming distance is limited, candidate items are not truncated.
        while now() < deadline:
            before = best_value
            remaining = deadline - now()
            use_global = (not math.isfinite(best_value) or radius >= k or now() >= next_global)
            if use_global:
                seconds = min(remaining, quantum * (1.5 + effort / (1.0 + effort)))
                values, optimal, bound, _ = run_model(seconds)
                if bound is not None:
                    global_upper = bound if global_upper is None else min(global_upper, bound)
                if proven(global_upper):
                    break
                next_global = now() + budget * 0.27
            else:
                center = best.copy()
                seconds = min(remaining, quantum * (1.0 + math.log2(max(1.0, radius / radius_base))))
                values, optimal, bound, _ = run_model(seconds, center, radius)
            if values is not None and now() < deadline:
                repair_fill((np.asarray(values) >= 0.5).astype(np.int8), default_order)
            if best_value > before:
                radius = radius_base
                last_gain = now()
            elif not use_global:
                radius = min(k, max(radius + 1, int(math.ceil(radius * (1.65 if optimal else 1.25)))))
            elif now() - last_gain > budget * 0.2:
                radius = min(k, max(radius_base, int(math.ceil(radius * 1.2))))
            if now() >= deadline:
                break
            # When no integer solver is available, keep the budget-driven feasible construction going instead of stopping early.
            if values is None:
                scale = 0.25 + math.sqrt(max(0.0, (now() - last_gain) / max(budget, 1e-12)))
                order = np.argsort(-(density * np.exp(rng.normal(0.0, scale, k))), kind="stable")
                random_selection = (rng.random(k) < np.clip(0.8 * lp + 0.1, 0.0, 1.0)).astype(np.int8)
                repair_fill(random_selection, order)
        return {"selection_status": answer}
    except Exception:
        return {"selection_status": answer}
