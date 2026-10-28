"""Time-threshold encoding and adaptive grid optimisation
Builds an initial solution by global shifting, encodes resource consumption and temporal constraints exactly with monotone threshold variables, solves under a time limit and progressively refines the search grid.
"""

import time
import math
import random
from bisect import bisect_left, bisect_right
from collections import deque

try:
    import numpy as np
    from scipy.optimize import milp, Bounds, LinearConstraint
    from scipy.sparse import coo_matrix
except Exception:
    np = None
    milp = None


def solve(instance: dict, time_limit_s: float) -> dict:
    started = time.monotonic()
    budget = max(0.0, float(time_limit_s))
    deadline = started + budget - min(budget * 0.12, 0.12 + budget * 0.015)
    count = int(instance['n']) + 2
    end = count - 1
    best = None
    earliest = [0] * count

    def answer(values):
        return {'start_times': {str(i): int(values[i]) for i in range(count)}}

    try:
        durations = [int(x) for x in instance['activities_processing_time']]
        edges = [(int(a), int(b), int(d)) for a, b, d in zip(
            instance['temporal_constraints_i'], instance['temporal_constraints_j'],
            instance['temporal_constraints_delta'])]
        resource_count = int(instance['num_resources'])
        periods = [sorted(set(int(t) for t in row if int(t) > 0))
                   for row in instance['resources_periods'][:resource_count]]
        capacities = [int(x) for x in instance['resources_capacity'][:resource_count]]
        demands = [[int(x) for x in instance['activities_resource_demands_%d' % k]]
                   for k in range(resource_count)]
        outgoing = [[] for _ in range(count)]
        incoming = [[] for _ in range(count)]
        for a, b, d in edges:
            outgoing[a].append((b, d))
            incoming[b].append((a, d))
        uses = [[] for _ in range(count)]
        for k in range(resource_count):
            if periods[k]:
                for i in range(count):
                    if demands[k][i] and durations[i]:
                        uses[i].append((k, demands[k][i]))
        last_period = max((row[-1] for row in periods if row), default=0)
        max_lag = max((d for _, _, d in edges), default=0)
        span = (count - 1) * max(0, max_lag)

        def lower_closure(values):
            values = list(values)
            queue = deque(range(count))
            queued = [True] * count
            depths = [0] * count
            while queue:
                a = queue.popleft()
                queued[a] = False
                for b, d in outgoing[a]:
                    v = values[a] + d
                    if v > values[b]:
                        if b == 0:
                            return None
                        values[b] = v
                        depths[b] = depths[a] + 1
                        if depths[b] >= count:
                            return None
                        if not queued[b]:
                            queue.append(b)
                            queued[b] = True
            return values

        def upper_closure(horizon):
            ceiling = max(last_period, max(earliest), horizon) + span + 1
            values = [ceiling] * count
            values[0] = 0
            values[end] = horizon
            queue = deque(range(count))
            queued = [True] * count
            while queue:
                b = queue.popleft()
                queued[b] = False
                for a, d in incoming[b]:
                    v = values[b] - d
                    if v < values[a]:
                        if v < earliest[a]:
                            return None
                        values[a] = v
                        if not queued[a]:
                            queue.append(a)
                            queued[a] = True
            return values

        def totals(values):
            result = [0] * resource_count
            for i, row in enumerate(uses):
                s = values[i]
                for k, demand in row:
                    ps = periods[k]
                    result[k] += demand * (bisect_right(ps, s + durations[i])
                                           - bisect_right(ps, s))
            return result

        def valid(values):
            if values is None or len(values) != count or values[0] != 0:
                return False
            if any(s < 0 for s in values):
                return False
            if any(values[b] - values[a] < d for a, b, d in edges):
                return False
            return all(v <= c for v, c in zip(totals(values), capacities))

        def keep(values):
            nonlocal best
            if values is not None and (best is None or values[end] < best[end]):
                if valid(values):
                    best = list(values)
                    return True
            return False

        temporal = lower_closure(earliest)
        if temporal is None:
            return answer(earliest)
        earliest = temporal
        if keep(earliest):
            return answer(best)

        # This construction is fully checked; a shift violating the upper-bound constraint pointing to activity 0 must not be taken as the safeguard solution.
        shifted = [0] + [earliest[i] + last_period for i in range(1, count)]
        keep(shifted)
        safe_bounds = lower_closure([0] + [last_period] * (count - 1))
        keep(safe_bounds)
        if best is None:
            keep(upper_closure(last_period + max(earliest) + span + 1))

        # Resource costs change only at these integer shift events; sort the events to find the earliest feasible shift.
        construction_deadline = min(deadline, started + budget * 0.12)
        shift_limit = last_period
        for a, b, d in edges:
            if b == 0 and a != 0:
                shift_limit = min(shift_limit, -d - earliest[a])
        events = {}
        complete = True
        for i in range(1, count):
            if time.monotonic() >= construction_deadline:
                complete = False
                break
            for k, demand in uses[i]:
                base = earliest[i]
                duration = durations[i]
                for t in periods[k]:
                    enter, leave = t - base - duration, t - base
                    if 0 < enter <= shift_limit:
                        changes = events.setdefault(enter, {})
                        changes[k] = changes.get(k, 0) + demand
                    if 0 < leave <= shift_limit:
                        changes = events.setdefault(leave, {})
                        changes[k] = changes.get(k, 0) - demand
        if complete:
            usage = totals(earliest)
            violated = sum(v > c for v, c in zip(usage, capacities))
            for shift in sorted(events):
                if time.monotonic() >= construction_deadline:
                    break
                for k, delta in events[shift].items():
                    violated -= usage[k] > capacities[k]
                    usage[k] += delta
                    violated += usage[k] > capacities[k]
                if violated == 0:
                    keep([0] + [s + shift for s in earliest[1:]])
                    break

        rng = random.Random(count * 1000003 + len(edges) * 1009 + sum(durations))
        lower_objective = earliest[end]
        # Grid size adapts to the time limit and graph density; when refinement fails, change the grid rather than exiting after a fixed count.
        density = 1.0 + len(edges) / max(1, count)
        target_size = max(count, int(2600 * math.sqrt(max(0.001, budget) / density)))
        mesh_divisor = 1.0
        phase = 0

        def make_domains(upper, size):
            widths = [upper[i] - earliest[i] for i in range(count)]
            total_width = sum(widths)
            step = max(1, int(math.ceil(total_width / max(1, size))))
            radius = max(1, int(math.log2(2.0 + budget)))
            domains = []
            exact = True
            for i in range(count):
                lo, hi = earliest[i], upper[i]
                if step == 1 or hi - lo <= 2 * radius + 2:
                    domain = list(range(lo, hi + 1))
                else:
                    points = {lo, hi}
                    offset = 0 if phase == 0 else rng.randrange(step)
                    points.update(range(lo + offset, hi + 1, step))
                    points.update(range(lo, min(hi, lo + radius) + 1))
                    points.update(range(max(lo, hi - radius), hi + 1))
                    if best is not None:
                        center = max(lo, min(hi, best[i]))
                        points.update(range(max(lo, center - radius),
                                            min(hi, center + radius) + 1))
                    domain = sorted(points)
                    exact = False
                domains.append(domain)
            return domains, exact

        def optimize(domains, cutoff):
            # z[i,q]=1 means S_i >= domains[i][q], with q starting at 1.
            offsets = []
            variable_count = 0
            for domain in domains:
                offsets.append(variable_count)
                variable_count += len(domain) - 1
            if variable_count == 0:
                point = [d[0] for d in domains]
                return (point if valid(point) else None), True, None
            if milp is None:
                return None, False, None
            objective = np.zeros(variable_count, dtype=float)
            lb = np.zeros(variable_count, dtype=float)
            ub = np.ones(variable_count, dtype=float)
            rows, cols, data, bounds = [], [], [], []

            def row(indices, coefficients, bound):
                index = len(bounds)
                rows.extend([index] * len(indices))
                cols.extend(indices)
                data.extend(coefficients)
                bounds.append(float(bound))

            for i, domain in enumerate(domains):
                offset = offsets[i]
                for q in range(1, len(domain) - 1):
                    row([offset + q, offset + q - 1], [1.0, -1.0], 0)
                if i == end:
                    objective[offset:offset + len(domain) - 1] = np.diff(domain)

            # Each time-lag constraint is a set of threshold implications; for identical consequents keep only the strongest antecedent.
            for a, b, lag in edges:
                if time.monotonic() >= cutoff:
                    return None, False, None
                da, db = domains[a], domains[b]
                first = bisect_left(db, da[0] + lag)
                if first >= len(db):
                    return None, True, None
                if first:
                    lb[offsets[b] + first - 1] = 1.0
                previous = first
                for q in range(1, len(da)):
                    required = bisect_left(db, da[q] + lag)
                    if required == previous:
                        continue
                    previous = required
                    if required >= len(db):
                        ub[offsets[a] + q - 1] = 0.0
                        break
                    if required:
                        row([offsets[a] + q - 1, offsets[b] + required - 1],
                            [1.0, -1.0], 0)

            for k in range(resource_count):
                if time.monotonic() >= cutoff:
                    return None, False, None
                ps = np.asarray(periods[k], dtype=np.int64)
                constant = 0
                indices, coefficients = [], []
                for i in range(count):
                    demand = demands[k][i]
                    if not demand or not durations[i] or not len(ps):
                        continue
                    ds = np.asarray(domains[i], dtype=np.int64)
                    costs = demand * (np.searchsorted(ps, ds + durations[i], side='right')
                                      - np.searchsorted(ps, ds, side='right'))
                    constant += int(costs[0])
                    difference = np.diff(costs)
                    nonzero = np.flatnonzero(difference)
                    indices.extend((offsets[i] + nonzero).tolist())
                    coefficients.extend(difference[nonzero].astype(float).tolist())
                if not indices:
                    if constant > capacities[k]:
                        return None, True, None
                else:
                    row(indices, coefficients, capacities[k] - constant)
            if np.any(lb > ub):
                return None, True, None
            if time.monotonic() >= cutoff:
                return None, False, None
            matrix = coo_matrix((np.asarray(data, dtype=float),
                                 (np.asarray(rows, dtype=np.int32),
                                  np.asarray(cols, dtype=np.int32))),
                                shape=(len(bounds), variable_count)).tocsc()
            seconds = cutoff - time.monotonic()
            if seconds <= 0:
                return None, False, None
            result = milp(objective, integrality=np.ones(variable_count, dtype=np.uint8),
                          bounds=Bounds(lb, ub),
                          constraints=LinearConstraint(matrix, -np.inf, np.asarray(bounds)),
                          options={'time_limit': seconds, 'mip_rel_gap': 0.0,
                                   'presolve': True, 'disp': False})
            point = None
            if result.x is not None:
                point = []
                for i, domain in enumerate(domains):
                    values = result.x[offsets[i]:offsets[i] + len(domain) - 1]
                    q = int(np.count_nonzero(values > 0.5))
                    point.append(domain[q])
                if not valid(point):
                    point = None
            dual = getattr(result, 'mip_dual_bound', None)
            if dual is not None and math.isfinite(dual):
                dual = int(math.ceil(dual - 1e-6)) + domains[end][0]
            else:
                dual = None
            return point, result.status in (0, 2), dual

        while time.monotonic() < deadline:
            if best is not None and best[end] <= lower_objective:
                return answer(best)
            horizon = (best[end] - 1 if best is not None
                       else last_period + max(earliest) + span + 1)
            upper = upper_closure(horizon)
            if upper is None:
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            domains, exact = make_domains(upper, int(target_size * mesh_divisor))
            # The first round leaves room for refinement; later budgets are allocated according to the remaining wall-clock time.
            share = 0.62 if phase == 0 else 0.82
            cutoff = min(deadline, time.monotonic() + remaining * share)
            if milp is not None:
                point, proved, dual = optimize(domains, cutoff)
                improved = keep(point)
                if exact:
                    if dual is not None:
                        lower_objective = max(lower_objective, dual)
                    if proved and best is not None:
                        return answer(best)
                if not exact:
                    # The allowed model size scales with the total budget and the current remaining time.
                    available_scale = max(1.0, math.sqrt(max(0.0, remaining)))
                    mesh_divisor = min(mesh_divisor * (1.7 if proved else 1.15),
                                       available_scale + math.log2(2.0 + budget))
                elif not improved:
                    # Already the full grid: the next solve uses the remaining budget, with no cap on restarts.
                    target_size = max(target_size, sum(len(d) - 1 for d in domains))
            else:
                # When the general numerical library is unavailable, keep searching random lower bounds checked by the time closure.
                seed = list(earliest)
                for i in range(1, count):
                    seed[i] = rng.randint(earliest[i], upper[i])
                keep(lower_closure(seed))
            phase += 1

        return answer(best if best is not None else earliest)
    except Exception:
        return answer(best if best is not None else earliest)
