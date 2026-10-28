"""Warm-start integer programming with adaptive re-partitioning search
After safe column deletion and forced selection, constructs a complete partition, uses the linear relaxation to guide warm-started integer programming, and enlarges the re-partitioning neighbourhood according to the remaining time.
"""

import math
import numbers
import random
import time
from collections import deque

try:
    import numpy as np
    from scipy.sparse import csc_matrix
    from scipy.optimize import Bounds, LinearConstraint, milp
except Exception:
    np = None

try:
    from scipy.optimize._highspy import _core as _hc
except Exception:
    _hc = None


def _optimize(matrix, costs, warm, deadline, scale, seed, relaxation=False):
    """One solve with a wall-clock budget; falls back to the public interface if the backend fails."""
    if np is None or time.perf_counter() >= deadline:
        return None, None, False, False
    count = len(costs)
    c = np.asarray(costs, dtype=float) / scale
    if _hc is not None:
        try:
            h = _hc._Highs()
            h.setOptionValue('output_flag', False)
            h.setOptionValue('threads', 1)
            h.setOptionValue('random_seed', int(seed))
            h.setOptionValue('mip_rel_gap', 0.0)
            h.setOptionValue('mip_abs_gap', max(1e-12, 1e-8 / scale))
            h.setOptionValue('mip_feasibility_tolerance', 1e-7)
            duration = max(0.0, deadline - time.perf_counter())
            h.setOptionValue('mip_heuristic_effort', 0.1 + 0.1 * duration / (1.0 + duration))
            model = _hc.HighsLp()
            model.num_col_ = count
            model.num_row_ = matrix.shape[0]
            model.col_cost_ = c
            model.col_lower_ = np.zeros(count)
            model.col_upper_ = np.ones(count)
            model.row_lower_ = np.ones(matrix.shape[0])
            model.row_upper_ = np.ones(matrix.shape[0])
            model.a_matrix_.format_ = _hc.MatrixFormat.kColwise
            model.a_matrix_.num_col_ = count
            model.a_matrix_.num_row_ = matrix.shape[0]
            model.a_matrix_.start_ = np.asarray(matrix.indptr, dtype=np.int32)
            model.a_matrix_.index_ = np.asarray(matrix.indices, dtype=np.int32)
            model.a_matrix_.value_ = matrix.data
            if not relaxation:
                model.integrality_ = [_hc.HighsVarType.kInteger] * count
            if h.passModel(model) == _hc.HighsStatus.kError:
                raise ValueError('model initialisation failed')
            if warm is not None and not relaxation:
                h.setSolution(count, np.arange(count, dtype=np.int32), np.asarray(warm, dtype=float))
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                return None, None, False, False
            h.setOptionValue('time_limit', remaining)
            if h.run() == _hc.HighsStatus.kError:
                raise ValueError('solver run failed')
            status = h.getModelStatus()
            sol = h.getSolution()
            optimal = status == _hc.HighsModelStatus.kOptimal
            infeasible = status == _hc.HighsModelStatus.kInfeasible
            values = np.asarray(sol.col_value, dtype=float) if sol.value_valid else None
            bound = None
            if relaxation and optimal:
                bound = float(h.getInfo().objective_function_value) * scale
            elif not relaxation:
                value = float(h.getInfo().mip_dual_bound) * scale
                if math.isfinite(value):
                    bound = value
            return values, bound, optimal, infeasible
        except Exception:
            pass
    if time.perf_counter() >= deadline:
        return None, None, False, False
    try:
        result = milp(
            c, integrality=np.zeros(count) if relaxation else np.ones(count),
            bounds=Bounds(np.zeros(count), np.ones(count)),
            constraints=LinearConstraint(matrix, np.ones(matrix.shape[0]), np.ones(matrix.shape[0])),
            options={'time_limit': max(1e-6, deadline - time.perf_counter()),
                     'mip_rel_gap': 0.0, 'presolve': True, 'disp': False})
        bound = result.fun if relaxation and result.status == 0 else getattr(result, 'mip_dual_bound', None)
        if bound is not None:
            bound = float(bound) * scale
        return result.x, bound, result.status == 0, result.status == 2
    except Exception:
        return None, None, False, False


def _exact_search(rows, costs, m, incumbent, deadline, guide=None, first_only=False):
    """Bitset branching on the fewest candidates; returns the current solution and whether the search was exhaustive."""
    n = len(rows)
    if m * n > 400000000 or time.perf_counter() >= deadline:
        return incumbent, False
    rowbits = [0] * m
    masks = []
    prices = [math.inf] * m
    for j, cover in enumerate(rows):
        mask = 0
        bit = 1 << j
        unit = costs[j] / len(cover)
        for r in cover:
            mask |= 1 << r
            rowbits[r] |= bit
            prices[r] = min(prices[r], unit)
        masks.append(mask)
    best = incumbent
    bound = math.fsum(costs[j] for j in best) if best is not None else math.inf
    scores = [costs[j] / len(rows[j]) for j in range(n)]
    if guide is not None:
        unit = max(scores, default=1.0)
        scores = [scores[j] - unit * float(guide[j]) for j in range(n)]
    stack = [((1 << m) - 1, (1 << n) - 1, 0.0, ())]
    seen = {}
    while stack and time.perf_counter() < deadline:
        remain, active, value, path = stack.pop()
        if not remain:
            exact_value = math.fsum(costs[j] for j in path)
            if best is None or exact_value < bound:
                bound, best = exact_value, list(path)
            if first_only:
                return best, False
            continue
        if value >= bound or value >= seen.get(remain, math.inf):
            continue
        seen[remain] = value
        todo = remain
        options = 0
        minimum = n + 1
        lower = 0.0
        while todo:
            bit = todo & -todo
            r = bit.bit_length() - 1
            todo ^= bit
            candidates = rowbits[r] & active
            degree = candidates.bit_count()
            if degree < minimum:
                minimum, options = degree, candidates
            if not degree:
                break
            lower += prices[r]
        if not minimum or value + lower > bound + 1e-9 * max(1.0, abs(bound)):
            continue
        choices = []
        while options:
            bit = options & -options
            choices.append(bit.bit_length() - 1)
            options ^= bit
        choices.sort(key=scores.__getitem__, reverse=True)
        for j in choices:
            new_value = value + costs[j]
            if new_value >= bound:
                continue
            forbidden = 0
            for r in rows[j]:
                forbidden |= rowbits[r]
            stack.append((remain ^ masks[j], active & ~forbidden, new_value, path + (j,)))
    return best, not stack


def solve(instance: dict, time_limit_s: float) -> dict:
    started = time.perf_counter()
    answer = {'feasible': False, 'x': None, 'total_cost': None}
    try:
        budget = float(time_limit_s)
        if not math.isfinite(budget) or budget < 0:
            budget = 0.0
        reserve = min(budget * 0.12, 0.025 + budget * 0.035)
        deadline = started + max(0.0, budget - reserve)
        n, m = instance['n'], instance['m']
        raw_costs, raw_covers = instance['costs'], instance['covers']
        if (not isinstance(n, numbers.Integral) or isinstance(n, bool) or n < 0 or
                not isinstance(m, numbers.Integral) or isinstance(m, bool) or m < 0 or
                len(raw_costs) != n or len(raw_covers) != n):
            return dict(answer, error='Input error: invalid size or array lengths')
        n, m = int(n), int(m)
        unique = {}
        original_costs = []
        for j in range(n):
            if not isinstance(raw_costs[j], numbers.Real):
                return dict(answer, error='Input error: costs must be finite non-negative reals')
            cost = float(raw_costs[j])
            if not math.isfinite(cost) or cost < 0:
                return dict(answer, error='Input error: costs must be finite non-negative reals')
            original_costs.append(cost)
            cover = raw_covers[j]
            if any(not isinstance(r, numbers.Integral) or isinstance(r, bool) or r < 0 or r >= m for r in cover):
                return dict(answer, error='Input error: invalid task index')
            key = tuple(sorted(int(r) for r in cover))
            if len(set(key)) != len(key):
                return dict(answer, error='Input error: a scheme contains a duplicate task')
            if key and (key not in unique or cost < unique[key][0]):
                unique[key] = (cost, j)
        if m == 0:
            return {'feasible': True, 'x': [0] * n, 'total_cost': 0.0}

        singles = {cover[0]: value[0] for cover, value in unique.items() if len(cover) == 1}
        records = []
        for cover, (cost, original) in unique.items():
            if len(cover) > 1 and all(r in singles for r in cover):
                if cost >= math.fsum(singles[r] for r in cover):
                    continue
            records.append((cover, cost, original))
        del unique
        incidence = [[] for _ in range(m)]
        for j, (cover, _, _) in enumerate(records):
            for r in cover:
                incidence[r].append(j)
        counts = [len(cols) for cols in incidence]
        if any(count == 0 for count in counts):
            return answer
        queue = deque(r for r in range(m) if counts[r] == 1)
        active = [True] * len(records)
        covered = [False] * m
        fixed = []
        while queue:
            r = queue.popleft()
            if covered[r]:
                continue
            if counts[r] == 0:
                return answer
            column = next(j for j in incidence[r] if active[j])
            fixed.append(records[column][2])
            for q in records[column][0]:
                covered[q] = True
            for q in records[column][0]:
                for j in incidence[q]:
                    if active[j]:
                        active[j] = False
                        for s in records[j][0]:
                            counts[s] -= 1
                            if not covered[s]:
                                if counts[s] == 0:
                                    return answer
                                if counts[s] == 1:
                                    queue.append(s)
        remaining_rows = [r for r in range(m) if not covered[r]]
        mapping = {r: i for i, r in enumerate(remaining_rows)}
        rows, costs, originals = [], [], []
        for j, (cover, cost, original) in enumerate(records):
            if active[j]:
                rows.append(tuple(mapping[r] for r in cover))
                costs.append(cost)
                originals.append(original)
        del records, incidence
        size, count = len(remaining_rows), len(rows)
        best = None
        best_cost = math.inf

        def accept(selection):
            nonlocal best, best_cost, answer
            if selection is None:
                return False
            occupied = set()
            for j in selection:
                if j < 0 or j >= count:
                    return False
                for r in rows[j]:
                    if r in occupied:
                        return False
                    occupied.add(r)
            if len(occupied) != size:
                return False
            value = math.fsum(costs[j] for j in selection)
            if best is not None and value >= best_cost:
                return False
            chosen = fixed + [originals[j] for j in selection]
            x = [0] * n
            for j in chosen:
                x[j] = 1
            total = math.fsum(original_costs[j] for j in chosen)
            if not math.isfinite(total):
                return False
            best, best_cost = list(selection), value
            answer = {'feasible': True, 'x': x, 'total_cost': total}
            return True

        if size == 0:
            accept([])
            return answer
        singleton = {cover[0]: j for j, cover in enumerate(rows) if len(cover) == 1}
        if len(singleton) == size:
            accept(list(singleton.values()))
        for j, cover in enumerate(rows):
            if len(cover) == size:
                accept([j])
        rng = random.Random(8675309)
        rowcols = [[] for _ in range(size)]
        for j, cover in enumerate(rows):
            for r in cover:
                rowcols[r].append(j)
        savings = [math.fsum(costs[singleton[r]] for r in cover if r in singleton) - costs[j]
                   for j, cover in enumerate(rows)]

        def construct(order):
            used = set()
            chosen = []
            for j in order:
                if all(r not in used for r in rows[j]):
                    chosen.append(j)
                    used.update(rows[j])
            for r in range(size):
                if r not in used:
                    if r not in singleton:
                        return
                    chosen.append(singleton[r])
            accept(chosen)

        order = [j for j in range(count) if len(rows[j]) > 1 and
                 (savings[j] > 0 or any(r not in singleton for r in rows[j]))]
        order.sort(key=lambda j: (sum(r not in singleton for r in rows[j]),
                                  savings[j] / math.sqrt(len(rows[j]))), reverse=True)
        construct(order)
        initial_end = min(deadline, started + budget * (0.08 if best is None else 0.025))
        if best is None:
            candidate, complete = _exact_search(rows, costs, size, None, initial_end, first_only=True)
            accept(candidate)
            if complete and best is None:
                return answer
        if time.perf_counter() >= deadline:
            if best is None:
                answer['error'] = 'No feasible solution found within the time limit; infeasibility not proven'
            return answer
        if np is None:
            candidate, complete = _exact_search(rows, costs, size, best, deadline)
            accept(candidate)
            if best is None and not complete:
                answer['error'] = 'No feasible solution found within the time limit; infeasibility not proven'
            return answer

        starts = [0]
        indices = []
        for cover in rows:
            indices.extend(cover)
            starts.append(len(indices))
        matrix = csc_matrix((np.ones(len(indices)), np.asarray(indices, dtype=np.int32),
                             np.asarray(starts, dtype=np.int32)), shape=(size, count))
        scale = max(max(costs, default=1.0), 1e-100)
        # A median-scale normaliser prevents extremely large column costs from squashing typical objective coefficients.
        positive = sorted(c for c in costs if c > 0)
        if positive:
            scale = positive[len(positive) // 2]
        scale = max(scale, max(costs, default=0.0) / 1e12, 1e-100)
        lp_guide = np.zeros(count)
        lower_bound = -math.inf
        remaining = deadline - time.perf_counter()
        lp_end = min(deadline, time.perf_counter() + remaining * 0.10)
        values, bound, optimal, impossible = _optimize(matrix, costs, None, lp_end, scale, 0, True)
        if impossible:
            return answer
        if values is not None and len(values) == count:
            lp_guide = np.maximum(0.0, np.minimum(1.0, values))
            if np.max(np.abs(values - np.rint(values))) <= 1e-7:
                accept(np.flatnonzero(values > 0.5).tolist())
            if optimal and bound is not None:
                lower_bound = bound
            lp_order = sorted(order, key=lambda j: (lp_guide[j], savings[j] / math.sqrt(len(rows[j]))), reverse=True)
            construct(lp_order)
        if best is not None and best_cost <= lower_bound + 1e-7:
            return answer
        if best is None and values is not None:
            until = min(deadline, time.perf_counter() + budget * 0.025)
            candidate, complete = _exact_search(rows, costs, size, None, until, lp_guide, True)
            accept(candidate)
            if complete and best is None:
                return answer

        # The first round keeps the full model; afterwards only a group of schemes is released from a complete feasible partition.
        global_run = True
        bit_mode = count * size <= 400000000
        masks = [sum(1 << r for r in cover) for cover in rows] if bit_mode else None
        base_fraction = min(0.70, 0.16 + 0.065 * math.log1p(budget))
        fraction = base_fraction
        last_improvement = time.perf_counter()
        anchor_order = sorted(range(count), key=lambda j: (lp_guide[j], savings[j] / len(rows[j])), reverse=True)
        anchor_pool = max(1, min(count, math.ceil(math.sqrt(count) * (1.0 + math.log1p(budget)))))
        while time.perf_counter() < deadline:
            now = time.perf_counter()
            left = deadline - now
            if best is None or global_run:
                eligible = list(range(count))
                free_rows = list(range(size))
                keep = []
                local_matrix = matrix
                until = min(deadline, now + left * (0.70 if best is None else 0.62)) if global_run else deadline
                global_run = False
            else:
                if best_cost <= lower_bound + 1e-7 or best_cost == 0.0:
                    break
                stagnation = (now - last_improvement) / max(budget, 1e-12)
                target = min(len(best), max(2, math.ceil(len(best) * min(1.0, fraction + stagnation) *
                                                       (0.75 + 0.5 * rng.random()))))
                if target >= len(best):
                    eligible = list(range(count))
                    free_rows = list(range(size))
                    keep = []
                    local_matrix = matrix
                else:
                    owner = [-1] * size
                    for j in best:
                        for r in rows[j]:
                            owner[r] = j
                    chosen_set = set(best)
                    if rng.random() < 0.75:
                        anchor = anchor_order[int(anchor_pool * rng.random() ** 2)]
                        released = {owner[r] for r in rows[anchor]}
                    else:
                        released = {rng.choice(best)}
                    fringe = list(released)
                    construction_end = min(deadline, now + left * 0.06)
                    while len(released) < target and time.perf_counter() < construction_end:
                        old_size = len(released)
                        parent = rng.choice(fringe)
                        r = rng.choice(rows[parent])
                        alternatives = rowcols[r]
                        j = rng.choice(alternatives)
                        if rng.random() < 0.65:
                            k = rng.choice(alternatives)
                            if lp_guide[k] > lp_guide[j]:
                                j = k
                        added = {owner[r] for r in rows[j]} - released
                        released.update(added)
                        fringe.extend(added)
                        if len(released) == old_size:
                            # Each expansion adds one scheme, so isolated components do not leave neighbourhood construction idling.
                            j = rng.choice(tuple(chosen_set - released))
                            released.add(j)
                            fringe.append(j)
                    free_rows = sorted(r for j in released for r in rows[j])
                    keep = [j for j in best if j not in released]
                    if masks is not None:
                        free_mask = 0
                        for j in released:
                            free_mask |= masks[j]
                        eligible = [j for j in range(count) if masks[j] & free_mask == masks[j]]
                    else:
                        free_set = set(free_rows)
                        eligible = [j for j, cover in enumerate(rows) if all(r in free_set for r in cover)]
                    local_matrix = matrix[free_rows, :][:, eligible].tocsc()
                relative = len(free_rows) / size
                slice_length = math.sqrt(max(0.0, left * budget)) * (0.065 + 0.10 * relative)
                until = min(deadline, time.perf_counter() + max(budget * 0.015, slice_length))
            if time.perf_counter() >= deadline:
                break
            warm = None
            if best is not None:
                chosen_set = set(best)
                warm = [1.0 if j in chosen_set else 0.0 for j in eligible]
            local_costs = [costs[j] for j in eligible]
            values, bound, optimal, impossible = _optimize(
                local_matrix, local_costs, warm, until, scale, rng.randrange(2147483647))
            full = not keep and len(eligible) == count
            changed = False
            if values is not None and len(values) == len(eligible):
                candidate = keep + [eligible[k] for k in np.flatnonzero(values > 0.5)]
                changed = accept(candidate)
            if full:
                if bound is not None and math.isfinite(bound):
                    lower_bound = max(lower_bound, bound)
                if impossible and best is None:
                    return answer
                if optimal and best is not None:
                    break
            if changed:
                last_improvement = time.perf_counter()
                fraction = max(base_fraction, fraction * 0.8)
            else:
                fraction = min(1.0, fraction * (1.0 + 0.12 / math.sqrt(1.0 + budget)))
            if best is None and values is None:
                until = min(deadline, time.perf_counter() + budget * 0.04)
                candidate, complete = _exact_search(rows, costs, size, None, until, lp_guide, True)
                accept(candidate)
                if complete and best is None:
                    return answer
        if best is None:
            answer['error'] = 'No feasible solution found within the time limit; infeasibility not proven'
        return answer
    except Exception:
        if not answer['feasible']:
            answer['error'] = 'Input or solve failure; infeasibility not proven'
        return answer
