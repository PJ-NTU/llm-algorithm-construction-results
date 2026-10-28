"""Bidirectional propagation-cone event repair algorithm
Caches time-lag propagation cones with lazy shortest paths, and combines event jumps, chained advancing, tabu repair within the makespan upper bound and large-neighbourhood rebuilding to store the best feasible solution.
"""

from bisect import bisect_right, bisect_left
from collections import deque
import heapq
import math
import random
import time


def solve(instance: dict, time_limit_s: float) -> dict:
    clock = time.perf_counter
    started = clock()
    budget = max(0.0, float(time_limit_s))
    margin = min(0.05, 0.015 * budget)
    deadline = started + budget - margin
    size = int(instance["n"]) + 2
    end = size - 1
    best = None
    emergency = [0] * size

    def emit(starts):
        return {"start_times": {str(i): int(starts[i]) for i in range(size)}}

    try:
        resources = int(instance["num_resources"])
        duration = [int(x) for x in instance["activities_processing_time"]]
        capacity = [int(x) for x in instance["resources_capacity"]]
        periods = [sorted(set(int(t) for t in row))
                   for row in instance["resources_periods"]]
        edges = [(int(a), int(b), int(d)) for a, b, d in zip(
            instance["temporal_constraints_i"],
            instance["temporal_constraints_j"],
            instance["temporal_constraints_delta"])]
        outgoing = [[] for _ in range(size)]
        incoming = [[] for _ in range(size)]
        for a, b, lag in edges:
            outgoing[a].append((b, lag))
            incoming[b].append((a, lag))
        uses = [[] for _ in range(size)]
        users = [[] for _ in range(resources)]
        for k in range(resources):
            row = instance["activities_resource_demands_%d" % k]
            for i in range(size):
                demand = int(row[i])
                if demand and duration[i]:
                    uses[i].append((k, demand))
                    users[k].append((i, demand))
        scale = [1.0 / max(1, capacity[k]) for k in range(resources)]
        cache = [{} for _ in range(size)]

        def contribution(i, start):
            result = cache[i].get(start)
            if result is None:
                finish = start + duration[i]
                result = tuple(demand * (bisect_right(periods[k], finish)
                                          - bisect_right(periods[k], start))
                               for k, demand in uses[i])
                cache[i][start] = result
            return result

        def consumption(starts):
            totals = [0] * resources
            for i in range(size):
                for (k, _), value in zip(uses[i], contribution(i, starts[i])):
                    totals[k] += value
            return totals

        def feasible(totals):
            return all(totals[k] <= capacity[k] for k in range(resources))

        def penalty(totals, weights=None):
            result = 0.0
            for k in range(resources):
                if totals[k] > capacity[k]:
                    excess = (totals[k] - capacity[k]) * scale[k]
                    value = excess * (1.0 + excess)
                    result += value if weights is None else weights[k] * value
            return result

        def temporal(bounds, stop=None):
            starts = list(bounds)
            starts[0] = 0
            queue = deque(range(size))
            queued = [True] * size
            depth = [0] * size
            while queue:
                if stop is not None and clock() >= stop:
                    return None
                a = queue.popleft()
                queued[a] = False
                for b, lag in outgoing[a]:
                    value = starts[a] + lag
                    if value > starts[b]:
                        if b == 0:
                            return None
                        starts[b] = value
                        depth[b] = depth[a] + 1
                        if depth[b] >= size:
                            return None
                        if not queued[b]:
                            queued[b] = True
                            queue.append(b)
            return starts

        earliest = temporal([0] * size)
        if earliest is None:
            return emit(emergency)
        emergency = list(earliest)
        earliest_totals = consumption(earliest)
        if feasible(earliest_totals):
            # Every feasible schedule is component-wise no earlier than this time-lag earliest solution, so the objective lower bound has been reached.
            return emit(earliest)
        lower_objective = earliest[end]
        best_key = (math.inf, math.inf)

        def record(starts, totals):
            nonlocal best, best_key
            if not feasible(totals):
                return False
            key = (starts[end], sum(starts))
            if key < best_key:
                if starts[0] != 0 or min(starts) < 0:
                    return False
                if any(starts[b] - starts[a] < lag for a, b, lag in edges):
                    return False
                best = list(starts)
                best_key = key
            return True

        # With earliest times as the potential function, all time-lag edges become non-negative slacks.
        # Hence the longest time lags in both directions can be obtained by lazy Dijkstra, without an all-pairs triple loop.
        forward_graph = [[] for _ in range(size)]
        reverse_graph = [[] for _ in range(size)]
        for a, b, lag in edges:
            slack = earliest[b] - earliest[a] - lag
            forward_graph[a].append((b, slack))
            reverse_graph[b].append((a, slack))
        forward_cache = [None] * size
        reverse_cache = [None] * size

        def cone(node, direction, force=False):
            saved = forward_cache if direction > 0 else reverse_cache
            if saved[node] is not None:
                return saved[node]
            graph = forward_graph if direction > 0 else reverse_graph
            distance = [None] * size
            distance[node] = 0
            heap = [(0, node)]
            links = []
            lag_to_end = None
            lag_to_zero = None
            while heap:
                if not force and clock() >= deadline:
                    return None
                value, i = heapq.heappop(heap)
                if value != distance[i]:
                    continue
                lag = (earliest[i] - earliest[node] - value if direction > 0
                       else earliest[node] - earliest[i] - value)
                links.append((i, lag))
                if i == end:
                    lag_to_end = lag
                if i == 0:
                    lag_to_zero = lag
                for j, slack in graph[i]:
                    candidate = value + slack
                    if distance[j] is None or candidate < distance[j]:
                        distance[j] = candidate
                        heapq.heappush(heap, (candidate, j))
            result = (links, lag_to_end, lag_to_zero)
            saved[node] = result
            return result

        upper = [math.inf] * size
        upper[0] = 0
        if incoming[0]:
            for i, lag in cone(0, -1, True)[0]:
                upper[i] = -lag
        last_period = max((row[-1] for row in periods if row), default=0)
        offset = max(0, last_period) + 1
        late = [int(min(earliest[i] + offset, upper[i])) for i in range(size)]
        late_totals = consumption(late)
        record(late, late_totals)
        if penalty(late_totals) < penalty(earliest_totals):
            emergency = list(late)

        seed = (size * 1000003 + len(edges) * 1009 + sum(duration) * 97
                + sum(capacity) * 17 + sum(sum(row) for row in periods))
        rng = random.Random(seed)
        effort = math.log1p(budget)
        breadth = min(size - 1, max(2, int(math.sqrt(size) * (0.8 + 0.25 * effort))))
        event_width = max(2, int(1 + effort))
        focus_width = min(resources, max(1, int(1 + math.sqrt(effort))))
        slice_length = budget * (1.0 + 0.2 * effort) / (4.0 + math.sqrt(size))
        end_cone = cone(end, -1)
        to_end = [None] * size
        if end_cone is not None:
            for i, lag in end_cone[0]:
                to_end[i] = lag

        def limits(cap):
            result = list(upper)
            if cap is not None:
                for i, lag in enumerate(to_end):
                    if lag is not None:
                        result[i] = min(result[i], cap - lag)
            return result

        def moved(starts, totals, node, target, cap=None):
            if node == 0 or target == starts[node] or target < earliest[node]:
                return None
            direction = 1 if target > starts[node] else -1
            data = cone(node, direction)
            if data is None:
                return None
            links, end_lag, zero_lag = data
            if direction > 0:
                if zero_lag is not None and target + zero_lag > 0:
                    return None
                if cap is not None and end_lag is not None and target + end_lag > cap:
                    return None
            result = list(starts)
            new_totals = list(totals)
            displacement = 0
            for i, lag in links:
                old = starts[i]
                value = max(old, target + lag) if direction > 0 else min(old, target - lag)
                if value == old:
                    continue
                if value < 0 or (i == 0 and value != 0):
                    return None
                result[i] = value
                displacement += abs(value - old)
                previous = contribution(i, old)
                after = contribution(i, value)
                for pos, (k, _) in enumerate(uses[i]):
                    new_totals[k] += after[pos] - previous[pos]
            if cap is not None and result[end] > cap:
                return None
            return result, new_totals, displacement

        def sampled_indices(lo, hi, width):
            length = hi - lo
            if length <= 0:
                return ()
            if length <= width:
                return range(lo, hi)
            return sorted({lo + j * (length - 1) // (width - 1) for j in range(width)})

        def targets(k, i, start, direction, high):
            row = periods[k]
            pi = duration[i]
            lo = bisect_right(row, start)
            hi = bisect_right(row, start + pi)
            if direction > 0:
                values = {start + 1}
                values.update(row[q] for q in sampled_indices(lo, hi, event_width))
                values.update(row[q] for q in range(hi, min(len(row), hi + event_width)))
                if row:
                    values.add(row[-1])
                if high != math.inf:
                    values.add(int(high))
                return sorted(v for v in values if start < v <= high)
            values = {earliest[i], start - 1}
            values.update(row[q] - pi - 1 for q in sampled_indices(lo, hi, event_width))
            values.update(row[q] - pi - 1 for q in range(max(0, lo - event_width), lo))
            return sorted(v for v in values if earliest[i] <= v < start)

        def quick_repair(starts, totals, stop):
            randomized = rng.random() < 0.6
            while clock() < stop:
                bad = [k for k in range(resources) if totals[k] > capacity[k]]
                if not bad:
                    record(starts, totals)
                    return starts, totals
                k = (max(bad, key=lambda q: (totals[q] - capacity[q]) * scale[q])
                     if rng.random() < 0.75 else rng.choice(bad))
                choices = []
                for i, demand in users[k]:
                    if i and starts[i] < upper[i]:
                        use = demand * (bisect_right(periods[k], starts[i] + duration[i])
                                        - bisect_right(periods[k], starts[i]))
                        if use:
                            choices.append((use, i))
                if not choices:
                    return None
                if randomized:
                    draw = rng.random() * sum(x[0] for x in choices)
                    node = choices[-1][1]
                    for weight, i in choices:
                        draw -= weight
                        if draw <= 0:
                            node = i
                            break
                else:
                    node = max(choices)[1]
                step = rng.randint(1, max(1, int(math.sqrt(duration[node] + 1) * (1 + effort / 4))))
                target = int(min(starts[node] + step, upper[node]))
                trial = moved(starts, totals, node, target)
                if trial is None:
                    return None
                starts, totals = trial[0], trial[1]
            record(starts, totals)
            return None

        def repair(starts, totals, stop, cap=None, randomness=0.0):
            high = limits(cap)
            weights = [1.0] * resources
            tabu = {}
            visited = {}
            local_best = penalty(totals)
            improved_at = clock()
            weighted_at = improved_at
            allotted = max(0.0, stop - improved_at)
            tenure = allotted / (3.0 + math.sqrt(size))
            restricted = max(1, int(1 + effort * randomness * 2))
            while clock() < stop:
                if record(starts, totals):
                    return starts, totals
                now = clock()
                visited[tuple(starts)] = now + max(tenure, allotted / 3)
                bad = [k for k in range(resources) if totals[k] > capacity[k]]
                bad.sort(key=lambda k: weights[k] * (totals[k] - capacity[k]) * scale[k],
                         reverse=True)
                focus = bad[:focus_width]
                if len(bad) > len(focus) and rng.random() < randomness:
                    focus[-1] = rng.choice(bad[len(focus):])
                current_penalty = penalty(totals, weights)
                pressure = {}
                for k in focus:
                    multiplier = weights[k] * scale[k] * (1 + (totals[k] - capacity[k]) * scale[k])
                    for i, demand in users[k]:
                        if i == 0:
                            continue
                        overlap = (bisect_right(periods[k], starts[i] + duration[i])
                                   - bisect_right(periods[k], starts[i]))
                        if overlap:
                            pressure[i] = pressure.get(i, 0.0) + multiplier * demand * overlap
                ranked = sorted(pressure, key=lambda i: pressure[i], reverse=True)
                selected = ranked[:breadth]
                if len(ranked) > breadth:
                    selected += rng.sample(ranked[breadth:], min(len(ranked) - breadth,
                                                               max(1, breadth // 3)))
                choices = []
                tried = set()
                for i in selected:
                    if clock() >= stop:
                        break
                    for k in focus:
                        if not any(resource == k for resource, _ in uses[i]):
                            continue
                        directions = (1, -1) if cap is not None else (1,)
                        for direction in directions:
                            for target in targets(k, i, starts[i], direction, high[i]):
                                if clock() >= stop:
                                    break
                                marker = (i, target)
                                if marker in tried:
                                    continue
                                tried.add(marker)
                                trial = moved(starts, totals, i, target, cap)
                                if trial is None:
                                    continue
                                ns, nt, displacement = trial
                                raw = penalty(nt)
                                aspiration = raw + 1e-12 < local_best
                                if cap is not None and not aspiration:
                                    if tabu.get(marker, 0.0) > now or visited.get(tuple(ns), 0.0) > now:
                                        continue
                                value = penalty(nt, weights)
                                if cap is None:
                                    damage = max(0, ns[end] - starts[end])
                                    gain = current_penalty - value
                                    cost = 1.0 + damage + displacement / max(1, 10 * size)
                                    rank = ((0, -gain / cost, value, damage) if gain > 1e-12
                                            else (1, value, damage, displacement))
                                else:
                                    noise = randomness * current_penalty * 0.04 * rng.random()
                                    rank = (value + noise, ns[end], displacement)
                                choices.append((rank, ns, nt, i, starts[i]))
                                choices.sort(key=lambda item: item[0])
                                del choices[restricted:]
                            if clock() >= stop:
                                break
                        if clock() >= stop:
                            break
                if not choices:
                    return None
                chosen = choices[rng.randrange(len(choices))] if rng.random() < randomness else choices[0]
                starts, totals = chosen[1], chosen[2]
                now = clock()
                tabu[(chosen[3], chosen[4])] = now + tenure
                raw = penalty(totals)
                if raw < local_best - 1e-12:
                    local_best = raw
                    improved_at = now
                if cap is not None and now - weighted_at > allotted / (3 + effort):
                    for k in bad:
                        weights[k] += 0.2 * (1 + (totals[k] - capacity[k]) * scale[k])
                    weighted_at = now
                if cap is not None and now - improved_at > allotted * 0.6:
                    break
            record(starts, totals)
            return None

        left_events = [None] * size

        def activity_events(i):
            if left_events[i] is None:
                points = set()
                for k, _ in uses[i]:
                    for t in periods[k]:
                        points.add(t - 1)
                        points.add(t - duration[i] - 1)
                left_events[i] = sorted(points)
            return left_events[i]

        def compress(starts, totals, stop):
            changed = True
            while changed and clock() < stop:
                changed = False
                order = list(range(1, end))
                rng.shuffle(order)
                order.insert(0, end)
                for node in order:
                    if clock() >= stop:
                        break
                    low, old = earliest[node], starts[node]
                    if low >= old:
                        continue
                    trial = moved(starts, totals, node, low)
                    if trial is not None and feasible(trial[1]):
                        starts, totals = trial[0], trial[1]
                        changed = True
                        record(starts, totals)
                        continue
                    data = cone(node, -1)
                    if data is None:
                        break
                    points = {old - 1}
                    for i, lag in data[0]:
                        if clock() >= stop:
                            break
                        if not uses[i] or low - lag >= starts[i]:
                            continue
                        events = activity_events(i)
                        lo = bisect_left(events, low - lag)
                        hi = bisect_left(events, min(old - lag, starts[i]))
                        for q in sampled_indices(lo, hi, max(2, breadth + event_width)):
                            points.add(events[q] + lag)
                    candidates = sorted(v for v in points if low <= v < old)
                    width = max(2, breadth * event_width)
                    candidates = [candidates[q] for q in sampled_indices(0, len(candidates), width)]
                    for target in candidates:
                        if clock() >= stop:
                            break
                        trial = moved(starts, totals, node, target)
                        if trial is not None and feasible(trial[1]):
                            starts, totals = trial[0], trial[1]
                            changed = True
                            record(starts, totals)
                            break
                if best_key[0] == lower_objective:
                    break
            return starts, totals

        # Search time is partitioned only after the safe solution is built; every search phase is cut off by the wall clock.
        bootstrap_stop = min(deadline, started + budget * 0.12)
        while clock() < bootstrap_stop:
            quick_repair(list(earliest), list(earliest_totals), bootstrap_stop)
            if best_key[0] == lower_objective:
                return emit(best)
        if best is not None and clock() < deadline:
            compress(list(best), consumption(best), min(deadline, clock() + slice_length))

        first = True
        while clock() < deadline:
            if best_key[0] == lower_objective:
                break
            now = clock()
            stop = min(deadline, now + slice_length)
            cap = None
            randomness = 0.1 + 0.6 * rng.random()
            if best is not None and not first and rng.random() < 0.68:
                gap = best[end] - lower_objective
                if rng.random() < 0.65:
                    target = best[end] - 1
                else:
                    target = best[end] - max(1, int(gap * rng.random()))
                cap = max(lower_objective, target)
                trial = moved(best, consumption(best), end, cap, cap)
                if trial is None:
                    continue
                current, totals = trial[0], trial[1]
            else:
                if best is not None and not first and rng.random() < 0.6:
                    release = 0.2 + 0.65 * rng.random()
                    bounds = [earliest[i] if i == end or rng.random() < release else best[i]
                              for i in range(size)]
                    current = temporal(bounds, stop)
                    if current is None:
                        continue
                    totals = consumption(current)
                else:
                    current, totals = list(earliest), list(earliest_totals)
                if first:
                    randomness = 0.0
            first = False
            repaired = repair(current, totals, stop, cap, randomness)
            if repaired is not None and clock() < deadline:
                compress(repaired[0], repaired[1], min(deadline, clock() + slice_length * 0.5))

        return emit(best if best is not None else emergency)
    except BaseException:
        return emit(best if best is not None else emergency)
