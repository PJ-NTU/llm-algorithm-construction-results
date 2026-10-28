"""Bidirectional time closure and bounded resource-repair search
Keeps a fast delay-only construction and continuously improves the makespan via coupled advancing and delaying, event jumps, tabu repair and bounded rebuilding.
"""

import time
import random
import math
import heapq
from bisect import bisect_left, bisect_right
from collections import deque


def solve(instance: dict, time_limit_s: float) -> dict:
    started = time.perf_counter()
    count = int(instance["n"]) + 2
    best = None
    emergency = [0] * count
    try:
        budget = max(0.0, float(time_limit_s))
        deadline = started + max(0.0, budget - min(0.06, max(0.001, budget * 0.015)))
        clock = time.perf_counter
        p = [int(x) for x in instance["activities_processing_time"]]
        m = int(instance["num_resources"])
        caps = [int(x) for x in instance["resources_capacity"]]
        periods = [sorted(set(int(t) for t in row))
                   for row in instance["resources_periods"]]
        out = [[] for _ in range(count)]
        edges = []
        for a, b, d in zip(instance["temporal_constraints_i"],
                           instance["temporal_constraints_j"],
                           instance["temporal_constraints_delta"]):
            a, b, d = int(a), int(b), int(d)
            out[a].append((b, d))
            edges.append((a, b, d))

        def close(bounds):
            s = list(bounds)
            s[0] = 0
            q = deque(range(count))
            queued = [True] * count
            depth = [0] * count
            while q:
                a = q.popleft()
                queued[a] = False
                for b, d in out[a]:
                    v = s[a] + d
                    if v > s[b]:
                        if b == 0:
                            return None
                        s[b] = v
                        depth[b] = depth[a] + 1
                        if depth[b] >= count:
                            return None
                        if not queued[b]:
                            queued[b] = True
                            q.append(b)
            return s

        earliest = close([0] * count)
        if earliest is None:
            return {"start_times": {str(i): 0 for i in range(count)}}
        emergency = list(earliest)
        rows = [[int(x) for x in instance["activities_resource_demands_%d" % k]]
                for k in range(m)]
        uses = [[(k, rows[k][i]) for k in range(m)
                 if rows[k][i] and p[i] > 0] for i in range(count)]
        users = [[i for i in range(1, count) if rows[k][i] and p[i] > 0]
                 for k in range(m)]
        strength = math.log1p(budget)
        cache_limit = max(32, int(16 * math.sqrt(count) * (1 + strength)))
        cache = [{} for _ in range(count)]

        def costs(i, s):
            saved = cache[i].get(s)
            if saved is not None:
                return saved
            saved = tuple(d * (bisect_right(periods[k], s + p[i])
                               - bisect_right(periods[k], s))
                          for k, d in uses[i])
            if len(cache[i]) >= cache_limit:
                cache[i].clear()
            cache[i][s] = saved
            return saved

        def consumption(s):
            total = [0] * m
            for i in range(count):
                for (k, _), v in zip(uses[i], costs(i, s[i])):
                    total[k] += v
            return total

        def feasible(total):
            return all(v <= caps[k] for k, v in enumerate(total))

        best_sum = math.inf

        def remember(s, total):
            nonlocal best, best_sum
            if feasible(total):
                ss = sum(s)
                if best is None or (s[-1], ss) < (best[-1], best_sum):
                    best = list(s)
                    best_sum = ss
                return True
            return False

        early_total = consumption(earliest)
        if remember(earliest, early_total):
            return {"start_times": {str(i): int(best[i]) for i in range(count)}}

        # Reweight arcs by earliest times so all arc lengths are non-negative; cache bidirectional time closures on demand.
        graph = [[] for _ in range(count)]
        reverse = [[] for _ in range(count)]
        for a, b, d in edges:
            slack = earliest[b] - earliest[a] - d
            graph[a].append((b, slack))
            reverse[b].append((a, slack))
        closures = [{}, {}]

        def closure(node, backward):
            bank = closures[int(backward)]
            if node in bank:
                return bank[node]
            adj = reverse if backward else graph
            dist = {node: 0}
            heap = [(0, node)]
            while heap:
                da, a = heapq.heappop(heap)
                if da != dist[a]:
                    continue
                for b, length in adj[a]:
                    db = da + length
                    if db < dist.get(b, math.inf):
                        dist[b] = db
                        heapq.heappush(heap, (db, b))
            if backward:
                result = [(j, earliest[node] - earliest[j] - d)
                          for j, d in dist.items()]
            else:
                result = [(j, earliest[j] - earliest[node] - d)
                          for j, d in dist.items()]
            bank[node] = result
            return result

        upper0 = [math.inf] * count
        for i, d in closure(0, True):
            upper0[i] = -d
        to_end = dict(closure(count - 1, True))

        def limits(cap):
            upper = list(upper0)
            if cap is not None:
                for i, d in to_end.items():
                    upper[i] = min(upper[i], cap - d)
            return upper

        def move(s, total, node, value, upper):
            if node == 0 or value == s[node]:
                return None
            backward = value < s[node]
            changed = []
            for j, d in closure(node, backward):
                v = value - d if backward else value + d
                if (backward and v < s[j]) or (not backward and v > s[j]):
                    if v < earliest[j] or v > upper[j] or (j == 0 and v != 0):
                        return None
                    changed.append((j, v))
            ns = list(s)
            nt = list(total)
            for j, v in changed:
                for (k, _), old, new in zip(uses[j], costs(j, s[j]), costs(j, v)):
                    nt[k] += new - old
                ns[j] = v
            return ns, nt

        # First build and verify a safeguard solution outside the resource charging intervals; upper bounds pointing to the origin are also propagated.
        shift = max([0] + [periods[k][-1] - earliest[i]
                            for i in range(1, count) for k, _ in uses[i]
                            if periods[k]])
        shifted = [0] + [min(earliest[i] + shift, upper0[i])
                         for i in range(1, count)]
        late = close(shifted)
        if late is not None:
            late_total = consumption(late)
            remember(late, late_total)
            if feasible(late_total):
                emergency = list(late)
        uniform_limit = min([shift] + [upper0[i] - earliest[i]
                                      for i in range(1, count)])
        if uniform_limit >= shift and best is not None:
            # Under non-monotone resources, bisection is used only for candidate generation, not to infer infeasibility.
            low, high = 0, shift
            while low < high and clock() < deadline:
                mid = (low + high) // 2
                s = [0] + [earliest[i] + mid for i in range(1, count)]
                total = consumption(s)
                if remember(s, total):
                    high = mid
                else:
                    low = mid + 1

        seed = count * 1000003 + sum(p) * 97 + sum(caps) * 17 + len(edges) * 1009
        rng = random.Random(seed)
        scales = [1.0 / max(1, c) for c in caps]
        event_cache = {}
        event_width = max(1, int(1 + strength / math.log(4)))
        node_width = max(2, int(math.sqrt(count) * (1 + strength / 3)))
        focus_width = max(1, int(math.ceil(math.sqrt(strength))))
        tenure = max(2, int(math.sqrt(count) * (1 + strength / 2)))

        def events(k, duration):
            key = (k, duration)
            saved = event_cache.get(key)
            if saved is None:
                changes = {}
                for t in periods[k]:
                    changes[t - duration] = changes.get(t - duration, 0) + 1
                    changes[t] = changes.get(t, 0) - 1
                right = sorted(t for t, d in changes.items() if d < 0)
                left = sorted(t - 1 for t, d in changes.items() if d > 0)
                saved = (right, left)
                event_cache[key] = saved
            return saved

        def targets(k, i, s, upper, backward=True, wide=True):
            right, left = events(k, p[i])
            lo, hi = earliest[i], upper[i]
            result = set()
            a = bisect_right(right, s)
            b = len(right) if math.isinf(hi) else bisect_right(right, hi)
            if a < b:
                result.update(right[a:min(b, a + event_width)])
                if wide:
                    result.add(right[b - 1])
                    result.add(right[rng.randrange(a, b)])
            if s + 1 <= hi:
                result.add(s + 1)
            if backward:
                a = bisect_left(left, lo)
                b = bisect_left(left, s)
                if a < b:
                    result.update(left[max(a, b - event_width):b])
                    if wide:
                        result.add(left[a])
                        result.add(left[rng.randrange(a, b)])
                if lo < s:
                    result.add(lo)
                    result.add(s - 1)
            return sorted(result)

        def violation(total, weights):
            score = 0.0
            for k in range(m):
                if total[k] > caps[k]:
                    x = (total[k] - caps[k]) * scales[k]
                    score += weights[k] * x * (1 + x)
            return score

        # A fast delay-only construction over a small candidate set gives a tighter initial upper bound for the subsequent bounded search.
        boot_end = min(deadline, started + budget * 0.16)
        run = 0
        weights = [1.0] * m
        while clock() < boot_end:
            s, total = list(earliest), list(early_total)
            upper = limits(best[-1] - 1 if best is not None else None)
            while clock() < boot_end:
                if remember(s, total):
                    break
                bad = [k for k in range(m) if total[k] > caps[k]]
                k = max(bad, key=lambda k: (total[k] - caps[k]) * scales[k])
                active = [i for i in users[k] if
                          bisect_right(periods[k], s[i] + p[i]) > bisect_right(periods[k], s[i])]
                if not active:
                    break
                active.sort(key=lambda i: rows[k][i] *
                            (bisect_right(periods[k], s[i] + p[i]) - bisect_right(periods[k], s[i])),
                            reverse=True)
                if run:
                    rng.shuffle(active)
                candidates = active[:max(2, int(2 + strength))]
                choice = None
                chosen_key = None
                old_score = violation(total, weights)
                for i in candidates:
                    for v in targets(k, i, s[i], upper, False, False):
                        trial = move(s, total, i, v, upper)
                        if trial is None:
                            continue
                        ns, nt = trial
                        gain = old_score - violation(nt, weights)
                        damage = 1 + max(0, ns[-1] - s[-1]) + 0.01 * (sum(ns) - sum(s))
                        key = (gain <= 0, -gain / damage, ns[-1])
                        if chosen_key is None or key < chosen_key:
                            chosen_key, choice = key, trial
                        if clock() >= boot_end:
                            break
                    if clock() >= boot_end:
                        break
                if choice is None:
                    break
                s, total = choice
                remember(s, total)
            run += 1

        quantum = max(0.01, budget / (4 + math.sqrt(count)))

        def compress(s, total, stop):
            upper = limits(s[-1])
            changed = True
            while changed and clock() < stop:
                changed = False
                order = list(range(1, count - 1))
                rng.shuffle(order)
                order.insert(0, count - 1)
                for i in order:
                    if clock() >= stop:
                        break
                    low, high = earliest[i], s[i]
                    # Advance time-feasible blocks jointly, including all activities on zero-slack cycles.
                    while low < high and clock() < stop:
                        v = low if low == earliest[i] else (low + high) // 2
                        trial = move(s, total, i, v, upper)
                        if trial is not None and feasible(trial[1]):
                            s, total = trial
                            high = v
                            changed = True
                            remember(s, total)
                        else:
                            if v == earliest[i]:
                                low = v + 1
                            else:
                                low = v + 1
                    if best is not None and best[-1] == earliest[-1]:
                        return s, total
            return s, total

        if best is not None and clock() < deadline:
            compress(list(best), consumption(best), min(deadline, clock() + quantum / 2))

        restart = 0
        while clock() < deadline:
            if best is not None and best[-1] == earliest[-1]:
                break
            stop = min(deadline, clock() + quantum)
            if best is None:
                cap = None
                s = list(earliest if restart % 2 == 0 or late is None else late)
            else:
                gap = best[-1] - earliest[-1]
                reduction = 1
                if restart % 4 == 0 and gap > 2:
                    reduction = max(1, int(gap * rng.uniform(0.15, 0.65)))
                cap = best[-1] - reduction
                upper = limits(cap)
                projected = move(best, consumption(best), count - 1, cap, upper)
                s = projected[0] if projected is not None else list(earliest)
                if restart % 3 == 1:
                    release = 0.15 + (1 - math.exp(-strength / 2)) * rng.random() * 0.7
                    bounds = [earliest[i] if i == count - 1 or rng.random() < release else s[i]
                              for i in range(count)]
                    rebuilt = close(bounds)
                    if rebuilt is not None:
                        s = rebuilt
                elif restart % 3 == 2:
                    s = list(earliest)
            upper = limits(cap)
            total = consumption(s)
            weights = [rng.uniform(0.7, 1.3) for _ in range(m)]
            seen = {tuple(s): 0}
            tabu = {}
            step = 0
            record_score = violation(total, weights)
            progressed = clock()
            noise = rng.uniform(0.0, 0.025) * min(1.0, strength)
            while clock() < stop:
                if remember(s, total):
                    compress(s, total, min(deadline, clock() + quantum / 2))
                    break
                score = violation(total, weights)
                bad = [k for k in range(m) if total[k] > caps[k]]
                bad.sort(key=lambda k: weights[k] * (total[k] - caps[k]) * scales[k], reverse=True)
                focus = bad[:focus_width]
                if len(bad) > focus_width:
                    focus.append(rng.choice(bad[focus_width:]))
                proposals = set()
                for k in focus:
                    active = []
                    for i in users[k]:
                        use = rows[k][i] * (bisect_right(periods[k], s[i] + p[i])
                                           - bisect_right(periods[k], s[i]))
                        if use > 0:
                            active.append((use, i))
                    active.sort(reverse=True)
                    selected = active[:node_width]
                    if len(active) > node_width:
                        selected += rng.sample(active[node_width:],
                                               min(len(active) - node_width, max(1, node_width // 3)))
                    for _, i in selected:
                        for v in targets(k, i, s[i], upper):
                            proposals.add((i, v))
                candidates = sorted(proposals)
                rng.shuffle(candidates)
                selected = None
                selected_key = None
                selected_node = None
                selected_score = None
                for i, v in candidates:
                    if clock() >= stop:
                        break
                    trial = move(s, total, i, v, upper)
                    if trial is None:
                        continue
                    ns, nt = trial
                    value = violation(nt, weights)
                    if value == 0:
                        remember(ns, nt)
                        selected, selected_node, selected_score = trial, i, value
                        break
                    aspiration = value < record_score - 1e-12
                    if not aspiration and (tabu.get((i, v), -1) > step or
                                           seen.get(tuple(ns), -tenure - 1) + tenure > step):
                        continue
                    key = (value + noise * score * rng.random(), sum(ns))
                    if selected_key is None or key < selected_key:
                        selected_key, selected = key, trial
                        selected_node, selected_score = i, value
                if selected is None:
                    break
                old_value = s[selected_node]
                s, total = selected
                tabu[(selected_node, old_value)] = step + tenure
                step += 1
                seen[tuple(s)] = step
                if selected_score < record_score - 1e-12:
                    record_score = selected_score
                    progressed = clock()
                elif clock() - progressed > quantum / (2 + strength):
                    break
                if len(seen) > cache_limit:
                    seen = {key: tick for key, tick in seen.items() if tick + tenure > step}
                remember(s, total)
            restart += 1
    except Exception:
        pass
    answer = best if best is not None else emergency
    return {"start_times": {str(i): int(answer[i]) for i in range(count)}}
