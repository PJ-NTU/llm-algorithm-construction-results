"""Weighted event neighbourhood search with bidirectional time-lag propagation
Combines consumption-weighted sampling, event-jump repair, coupled advancing and shrinking, and repeatedly rebuilds and tightens the project end time within the wall-clock budget.
"""

import time
import math
import random
import heapq
from bisect import bisect_left, bisect_right
from collections import deque


def solve(instance: dict, time_limit_s: float) -> dict:
    started = time.perf_counter()
    clock = time.perf_counter
    answer = {"start_times": {}}
    try:
        count = int(instance["n"]) + 2
        end = count - 1
        answer = {"start_times": {str(i): 0 for i in range(count)}}
        budget = max(0.0, float(time_limit_s))
        deadline = started + budget * 0.985
        duration = list(map(int, instance["activities_processing_time"]))
        nr = int(instance["num_resources"])
        capacity = list(map(int, instance["resources_capacity"]))
        periods = [sorted(set(map(int, ps))) for ps in instance["resources_periods"]]
        edges = [(int(a), int(b), int(d)) for a, b, d in zip(
            instance["temporal_constraints_i"], instance["temporal_constraints_j"],
            instance["temporal_constraints_delta"])]
        outgoing = [[] for _ in range(count)]
        incoming = [[] for _ in range(count)]
        for a, b, d in edges:
            outgoing[a].append((b, d))
            incoming[b].append((a, d))
        users = [[] for _ in range(nr)]
        needs = [[] for _ in range(count)]
        for k in range(nr):
            for i, r in enumerate(instance["activities_resource_demands_%d" % k]):
                if r and duration[i] and periods[k]:
                    needs[i].append((k, int(r)))
                    users[k].append((i, int(r)))

        # Deterministic time-lag closure; path lengths are used only to detect infeasible positive-weight cycles.
        def close(values, forward, bounds=None, stop=None, nodes=None):
            s = list(values)
            graph = outgoing if forward else incoming
            q = deque(range(count) if nodes is None else nodes)
            queued = [False] * count
            depth = [0] * count
            for i in q:
                queued[i] = True
            while q:
                if stop is not None and clock() >= stop:
                    return None
                a = q.popleft()
                queued[a] = False
                for b, d in graph[a]:
                    value = s[a] + d if forward else s[a] - d
                    changes = value > s[b] if forward else value < s[b]
                    if not changes:
                        continue
                    if b == 0 and value != 0:
                        return None
                    if bounds is not None:
                        if (forward and value > bounds[b]) or (not forward and value < bounds[b]):
                            return None
                    s[b] = value
                    depth[b] = depth[a] + 1
                    if depth[b] >= count:
                        return None
                    if not queued[b]:
                        q.append(b)
                        queued[b] = True
            return s

        earliest = close([0] * count, True)
        if earliest is None:
            return answer
        last_period = max([0] + [ps[-1] for ps in periods if ps])
        # The coordinate-wise minimum of any feasible solution and earliest + last_period is still feasible:
        # truncated activities all lie after the last charging period; this does not additionally assume an upper bound d_bar.
        upper = [v + last_period for v in earliest]
        upper[0] = 0
        upper = close(upper, False, earliest)
        if upper is None:
            return answer
        fallback = list(upper)
        answer = {"start_times": {str(i): int(fallback[i]) for i in range(count)}}
        level = 1.0 + math.log1p(budget)
        rng = random.Random(count * 1000003 + len(edges) * 1009 +
                            sum(duration) * 97 + sum(capacity) * 17 +
                            sum(sum(ps) for ps in periods))
        mean_duration = max(1, sum(duration) // max(1, count - 2))
        breadth = min(count - 1, max(2, int(math.sqrt(count) * level / 2)))
        event_width = max(2, int(math.ceil(level)))
        tenure = max(2, int(math.sqrt(count) * level))
        cache_limit = max(1, int((count + len(edges)) * level * level))
        profiles = [{} for _ in range(count)]

        def profile(i, start):
            cache = profiles[i]
            value = cache.get(start)
            if value is None:
                value = tuple(r * (bisect_right(periods[k], start + duration[i]) -
                                   bisect_right(periods[k], start)) for k, r in needs[i])
                if len(cache) >= cache_limit:
                    cache.clear()
                cache[start] = value
            return value

        def totals(s):
            total = [0] * nr
            for i in range(count):
                for (k, _), use in zip(needs[i], profile(i, s[i])):
                    total[k] += use
            return total

        def feasible(total):
            return all(v <= c for v, c in zip(total, capacity))

        best = None
        best_key = (float("inf"), float("inf"))
        elite = []
        pool_size = max(2, int(math.ceil(level)))

        def remember(s, total):
            nonlocal answer, best, best_key, elite
            if not feasible(total):
                return False
            key = (s[end], sum(s))
            if key < best_key:
                # Independently re-check the original hard constraints only when rewriting the output.
                if s[0] or min(s) < 0 or any(s[b] - s[a] < d for a, b, d in edges):
                    return False
                best = list(s)
                best_key = key
                answer = {"start_times": {str(i): int(s[i]) for i in range(count)}}
            if best is not None and s[end] <= best[end] + mean_duration:
                signature = tuple(s)
                if all(signature != item[1] for item in elite):
                    elite.append((key, signature))
                    elite.sort()
                    elite = elite[:pool_size]
            return True

        first_totals = totals(earliest)
        if remember(earliest, first_totals):
            return answer  # The global lower bound obtained by ignoring resources has been reached.
        remember(fallback, totals(fallback))

        # Small and medium graphs use the transitive closure to speed up coupled moves; large graphs propagate incrementally along the original adjacency lists.
        paths = backpaths = None
        pre_end = min(deadline, started + budget * 0.09)
        if clock() < pre_end and count * count <= max(1.0, budget) * 180000:
            try:
                import numpy as np
                lag_bound = max([1] + [abs(d) for _, _, d in edges])
                neg = -4 * (count + 1) * lag_bound - 1
                if neg > -(1 << 60):
                    matrix = np.full((count, count), neg, dtype=np.int64)
                    np.fill_diagonal(matrix, 0)
                    for a, b, d in edges:
                        if d > matrix[a, b]:
                            matrix[a, b] = d
                    complete = True
                    for k in range(count):
                        if clock() >= pre_end:
                            complete = False
                            break
                        np.maximum(matrix, matrix[:, k:k + 1] + matrix[k:k + 1, :], out=matrix)
                    if complete:
                        rows = matrix.tolist()
                        paths = [[(j, v) for j, v in enumerate(row) if v > neg // 2]
                                 for row in rows]
                        backpaths = [[] for _ in range(count)]
                        for i, row in enumerate(paths):
                            for j, d in row:
                                backpaths[j].append((i, d))
            except Exception:
                paths = backpaths = None

        def limits(bound):
            if bound is None:
                return upper
            if bound < earliest[end]:
                return None
            u = list(upper)
            u[end] = min(u[end], bound)
            return close(u, False, earliest, deadline, [end])

        def move(s, total, node, target, caps, stop):
            if node == 0 or target == s[node] or clock() >= stop:
                return None
            forward = target > s[node]
            if target < earliest[node] or target > caps[node]:
                return None
            v = list(s)
            if paths is not None:
                for j, d in (paths[node] if forward else backpaths[node]):
                    value = target + d if forward else target - d
                    if (forward and value > v[j]) or (not forward and value < v[j]):
                        if value < earliest[j] or value > caps[j] or (j == 0 and value != 0):
                            return None
                        v[j] = value
            else:
                v[node] = target
                v = close(v, forward, caps if forward else earliest, stop, [node])
                if v is None:
                    return None
            new_total = list(total)
            displacement = 0
            for i, value in enumerate(v):
                if value == s[i]:
                    continue
                displacement += abs(value - s[i])
                old_use = profile(i, s[i])
                new_use = profile(i, value)
                for h, (k, _) in enumerate(needs[i]):
                    new_total[k] += new_use[h] - old_use[h]
            return v, new_total, displacement

        def overlap(k, i, s):
            return bisect_right(periods[k], s + duration[i]) - bisect_right(periods[k], s)

        def targets(k, i, s, caps, stop, backward):
            lo, hi = earliest[i], caps[i]
            step = rng.randint(1, max(1, int(mean_duration * level / 2)))
            values = {min(hi, s + 1), min(hi, s + step), hi}
            ps = periods[k]
            use = overlap(k, i, s)
            q = bisect_right(ps, s)
            found = 0
            while q < len(ps) and ps[q] <= hi and clock() < stop:
                v = ps[q]
                if overlap(k, i, v) < use:
                    values.add(v)
                    found += 1
                    if found >= event_width:
                        break
                q += 1
            if ps:
                values.add(min(hi, max(s + 1, ps[-1])))
            if backward:
                values.update((lo, max(lo, s - 1), max(lo, s - step)))
                q = bisect_right(ps, s + duration[i] - 1) - 1
                found = 0
                while q >= 0 and clock() < stop:
                    v = ps[q] - duration[i] - 1
                    if v < lo:
                        break
                    if v < s and overlap(k, i, v) < use:
                        values.add(v)
                        found += 1
                        if found >= event_width:
                            break
                    q -= 1
            return sorted(v for v in values if lo <= v <= hi and v != s)

        def loss(total, weights):
            value = 0.0
            for k in range(nr):
                excess = total[k] - capacity[k]
                if excess > 0:
                    x = excess / max(1, capacity[k])
                    value += weights[k] * (x + x * x)
            return value

        def repair(s, total, caps, stop, randomness, width, backward):
            weights = [1.0] * nr
            recent = deque([tuple(s)])
            tabu = {tuple(s)}
            best_loss = loss(total, weights)
            progress_time = clock()
            stall = max(budget / (8 + math.sqrt(count)), (stop - clock()) / 3)
            while clock() < stop:
                if remember(s, total):
                    return s, total
                bad = [k for k in range(nr) if total[k] > capacity[k]]
                pressures = [weights[k] * (total[k] - capacity[k]) / max(1, capacity[k])
                             for k in bad]
                selected_k = rng.choices(bad, weights=pressures, k=1)[0]
                focus = {selected_k, bad[max(range(len(bad)), key=lambda h: pressures[h])]}
                activity_weight = {}
                activity_resource = {}
                for k in focus:
                    for i, r in users[k]:
                        if i == 0:
                            continue
                        w = r * overlap(k, i, s[i]) / max(1, capacity[k])
                        if w > 0:
                            activity_weight[i] = activity_weight.get(i, 0.0) + w * weights[k]
                            if w > activity_resource.get(i, (-1.0, k))[0]:
                                activity_resource[i] = (w, k)
                if not activity_weight:
                    return None
                ordered = sorted(activity_weight, key=lambda i:
                                 -math.log(max(1e-15, rng.random())) / activity_weight[i])
                if width > 1 and rng.random() > randomness:
                    first = max(activity_weight, key=activity_weight.get)
                    ordered.remove(first)
                    ordered.insert(0, first)
                current_loss = loss(total, weights)
                candidates = []
                for i in ordered[:width]:
                    k = activity_resource[i][1]
                    for value in targets(k, i, s[i], caps, stop, backward):
                        if clock() >= stop:
                            break
                        trial = move(s, total, i, value, caps, stop)
                        if trial is None:
                            continue
                        v, nt, displacement = trial
                        key = tuple(v)
                        if key in tabu:
                            continue
                        new_loss = loss(nt, weights)
                        if new_loss == 0 and remember(v, nt):
                            return v, nt
                        improvement = current_loss - new_loss
                        damage = 1.0 + max(0, v[end] - s[end]) + displacement / count
                        noise = 1.0 + randomness * rng.random()
                        if improvement > 1e-12:
                            score = (0, -improvement / damage * noise, new_loss, v[end])
                        else:
                            score = (1, new_loss * noise, displacement / count, v[end])
                        candidates.append((score, v, nt))
                    if clock() >= stop:
                        break
                if not candidates:
                    return None
                _, s, total = min(candidates, key=lambda item: item[0])
                key = tuple(s)
                recent.append(key)
                tabu.add(key)
                if len(recent) > tenure:
                    tabu.discard(recent.popleft())
                actual = loss(total, [1.0] * nr)
                if actual < best_loss - 1e-12:
                    best_loss = actual
                    progress_time = clock()
                elif clock() - progress_time > stall:
                    return None
                for k in bad:
                    weights[k] += 1.0 / (count * level)
            return None

        def compress(s, total, stop):
            changed = True
            while changed and clock() < stop:
                changed = False
                order = list(range(1, count - 1))
                rng.shuffle(order)
                order = [end] + order + [end]
                for i in order:
                    if clock() >= stop:
                        break
                    old = s[i]
                    lo = max([earliest[i]] + [s[a] + d for a, d in incoming[i]])
                    if lo < old:
                        values = {lo}
                        for k, _ in needs[i]:
                            ps = periods[k]
                            for shift in (0, -duration[i]):
                                a = bisect_left(ps, lo - shift)
                                b = bisect_left(ps, old - shift)
                                values.update(t + shift for t in ps[a:b])
                        for value in sorted(values):
                            if clock() >= stop:
                                break
                            trial = move(s, total, i, value, upper, stop)
                            if trial is not None and feasible(trial[1]):
                                s, total = trial[:2]
                                changed = True
                                remember(s, total)
                                break
                    # Coupled shrinking can pass through zero-slack time-lag cycles, unlike coordinate-wise Shrink.
                    if s[i] > earliest[i] and clock() < stop:
                        values = {earliest[i], s[i] - 1,
                                  max(earliest[i], s[i] - mean_duration),
                                  max(earliest[i], s[i] - int(mean_duration * level))}
                        for value in sorted(values):
                            trial = move(s, total, i, value, upper, stop)
                            if trial is not None and feasible(trial[1]):
                                s, total = trial[:2]
                                changed = True
                                remember(s, total)
                                break
            return s, total

        # When the origin has an upper bound, shifting everything later is not necessarily feasible; find the first solution by branching on necessary delays of overloaded resources.
        if best is None:
            heap = [(loss(first_totals, [1.0] * nr), earliest[end], 0, earliest, first_totals)]
            seen = {tuple(earliest)}
            serial = 0
            secure_end = min(deadline, clock() + budget * 0.40)
            while heap and clock() < secure_end and best is None:
                _, _, _, s, total = heapq.heappop(heap)
                bad = [k for k in range(nr) if total[k] > capacity[k]]
                if not bad:
                    remember(s, total)
                    break
                k = max(bad, key=lambda k: (total[k] - capacity[k]) / max(1, capacity[k]))
                for i, r in users[k]:
                    if i == 0 or clock() >= secure_end:
                        continue
                    use = overlap(k, i, s[i])
                    if not use:
                        continue
                    q = bisect_right(periods[k], s[i])
                    while q < len(periods[k]) and clock() < secure_end:
                        value = periods[k][q]
                        if value > upper[i]:
                            break
                        if overlap(k, i, value) < use:
                            trial = move(s, total, i, value, upper, secure_end)
                            if trial is not None:
                                v, nt, _ = trial
                                key = tuple(v)
                                if key not in seen:
                                    seen.add(key)
                                    serial += 1
                                    if remember(v, nt):
                                        break
                                    heapq.heappush(heap, (loss(nt, [1.0] * nr), v[end], serial, v, nt))
                            break
                        q += 1

        bootstrap_end = min(deadline, started + budget * 0.18)
        slice_length = budget / (6.0 + math.sqrt(count))
        while clock() < bootstrap_end:
            stop = min(bootstrap_end, clock() + slice_length / level)
            result = repair(list(earliest), list(first_totals), upper, stop,
                            rng.random() * 0.8, max(1, breadth // int(2 * level)), False)
            if result is not None:
                compress(*result, min(bootstrap_end, clock() + slice_length / level))
            if best is not None and best[end] == earliest[end]:
                return answer

        if best is not None and clock() < deadline:
            compress(list(best), totals(best), min(deadline, clock() + slice_length))

        # Each round uses the event repairer uniformly; the end-time bound and release size change with state and budget.
        while clock() < deadline:
            if best is not None and best[end] == earliest[end]:
                break
            stop = min(deadline, clock() + slice_length)
            tighten = best is not None and rng.random() < 0.70
            bound = best[end] - 1 if tighten else (best[end] if best is not None else None)
            caps = limits(bound)
            if caps is None:
                continue
            mode = rng.random()
            if best is not None and mode < 0.48:
                base = list(rng.choice(elite)[1]) if elite else list(best)
                if base[end] > caps[end]:
                    trial = move(base, totals(base), end, caps[end], upper, stop)
                    s = trial[0] if trial is not None else list(earliest)
                else:
                    s = base
                # Truncate to the latest feasible time stamp: the coordinate-wise minimum preserves difference constraints.
                s = [min(s[i], caps[i]) for i in range(count)]
            elif best is not None and mode < 0.84:
                release = min(0.9, 1.0 / level + rng.random() * 0.55)
                lower = [earliest[i] if i == end or rng.random() < release else
                         min(best[i], caps[i]) for i in range(count)]
                lower[0] = 0
                s = close(lower, True, caps, stop)
                if s is None:
                    s = list(earliest)
            else:
                amplitude = max(1, int(mean_duration * level * rng.random()))
                lower = [min(caps[i], earliest[i] + rng.randint(0, amplitude))
                         if needs[i] and rng.random() < 1.0 / level else earliest[i]
                         for i in range(count)]
                lower[0] = 0
                s = close(lower, True, caps, stop)
                if s is None:
                    s = list(earliest)
            result = repair(s, totals(s), caps, stop, 0.1 + rng.random() * 0.7,
                            breadth, True)
            if result is not None and clock() < deadline:
                compress(*result, min(deadline, clock() + slice_length / level))
        return answer
    except Exception:
        return answer
