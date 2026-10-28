"""Reachability-safeguarded adaptive iterated swap search
Guarantees coverage via sink components and combines randomized greedy, batched exact single swaps and destroy-repair, keeping the best feasible solution within the wall-clock limit.
"""

import math
import time


def solve(instance: dict, time_limit_s: float) -> dict:
    start = time.perf_counter()
    budget = max(0.0, float(time_limit_s))
    n = int(instance["n"])
    p = int(instance["p"])
    raw = instance["dist"]
    if p == n:
        return {"medians": list(range(n))}

    # Reachability is transitive. Keep mutually unreachable representatives and replace them with their downstream vertices.
    # After the scan, exactly one representative remains for each sink strongly connected component.
    representatives = []
    for i in range(n):
        row = raw[i]
        if any(row[j] is not None for j in representatives):
            continue
        representatives = [j for j in representatives if raw[j][i] is None]
        representatives.append(i)
    chosen = set(representatives)
    initial = representatives[:]
    for j in range(n):
        if len(initial) == p:
            break
        if j not in chosen:
            initial.append(j)
            chosen.add(j)
    answer = {"medians": initial[:]}
    deadline = start + budget - min(0.08, 0.015 * budget)
    if time.perf_counter() >= deadline:
        return answer
    try:
        _improve(raw, n, p, representatives, initial, answer, budget, deadline)
    except Exception:
        # On initialisation, numerical-library or resource failures, still return the complete feasible solution already saved.
        pass
    return answer


def _improve(raw, n, p, representatives, initial, answer, budget, deadline):
    import numpy as np

    clock = time.perf_counter
    d = np.asarray(raw, dtype=np.float64)
    d[np.isnan(d)] = np.inf
    scale = float(np.max(d, where=np.isfinite(d), initial=0.0))
    if scale == 0.0 or clock() >= deadline:
        return
    # Only uniform positive scaling; unreachable distances always stay infinite.
    d /= scale
    rng = np.random.default_rng(13007 + 97 * n + p)
    rows = np.arange(n)
    best_cost = float(np.min(d[:, initial], axis=1).sum())
    if best_cost == 0.0:
        return
    if p == 1:
        costs = d.sum(axis=0)
        j = int(np.argmin(costs))
        if np.isfinite(costs[j]):
            answer["medians"] = [j]
        return

    groups = np.full(n, -1, dtype=np.int64)
    members = []
    for g, representative in enumerate(representatives):
        vertices = np.flatnonzero(np.isfinite(d[representative]))
        members.append(vertices)
        groups[vertices] = g
    group_count = len(members)
    # Block size bounds temporary array memory; the number of searches is governed solely by the wall clock.
    block = max(1, min(n, 262144 // max(1, n)))
    greed_sample = min(n, max(16, math.ceil(n / p * (2.0 + math.log1p(budget)))))
    pool_limit = max(2, math.ceil(2.0 + math.log1p(budget) * math.log1p(p)))
    max_depth = min(p, n - p, max(2, math.ceil(math.sqrt(p) * math.log1p(budget))))
    elite = []

    def remember(s, cost):
        nonlocal best_cost
        if math.isfinite(cost) and cost < best_cost:
            best_cost = cost
            answer["medians"] = [int(j) for j in s]

    def statistics(s):
        distances = d[:, s].copy()
        owner = np.argmin(distances, axis=1)
        first = distances[rows, owner].copy()
        distances[rows, owner] = np.inf
        second = np.min(distances, axis=1)
        return owner, first, second, float(first.sum())

    def greedy(base, stop, randomized=False):
        s = list(base)
        selected = np.zeros(n, dtype=bool)
        selected[s] = True
        nearest = np.min(d[:, s], axis=1)
        while len(s) < p and clock() < stop:
            candidates = np.flatnonzero(~selected)
            count = min(len(candidates), greed_sample)
            if count < len(candidates):
                candidates = rng.choice(candidates, size=count, replace=False)
            gain = np.empty(len(candidates))
            evaluated = 0
            for a in range(0, len(candidates), block):
                if clock() >= stop:
                    break
                part = candidates[a:a + block]
                gain[a:a + len(part)] = np.minimum(d[:, part] - nearest[:, None], 0.0).sum(axis=0)
                evaluated = a + len(part)
            if evaluated == 0:
                break
            candidates = candidates[:evaluated]
            gain = gain[:evaluated]
            k = int(np.argmin(gain))
            if randomized:
                cutoff = gain[k] + (-gain[k]) * rng.uniform(0.0, 1.0 / (2.0 + math.log1p(budget)))
                restricted = np.flatnonzero(gain <= cutoff)
                k = int(rng.choice(restricted))
            j = int(candidates[k])
            s.append(j)
            selected[j] = True
            np.minimum(nearest, d[:, j], out=nearest)
        # Fill up the count if construction was cut off; base already covers all sink components.
        if len(s) < p:
            extra = np.flatnonzero(~selected)[:p - len(s)]
            s.extend(int(j) for j in extra)
        return s

    def descent(s, stop):
        s = list(s)
        owner, first, second, cost = statistics(s)
        remember(s, cost)
        while clock() < stop and cost > 0.0:
            selected = np.zeros(n, dtype=bool)
            selected[s] = True
            candidates = rng.permutation(np.flatnonzero(~selected))
            order = np.argsort(owner, kind="stable")
            sizes = np.bincount(owner, minlength=p)
            active = np.flatnonzero(sizes)
            starts = np.cumsum(sizes[active]) - sizes[active]
            a_first = first[order, None]
            a_second = second[order, None]
            move = None
            best_delta = 0.0
            complete = True
            for a in range(0, len(candidates), block):
                if clock() >= stop:
                    complete = False
                    break
                part = candidates[a:a + block]
                candidate_dist = d[order[:, None], part[None, :]]
                after_add = np.minimum(candidate_dist, a_first)
                addition = (after_add - a_first).sum(axis=0)
                correction = np.minimum(candidate_dist, a_second)
                correction -= after_add
                delta = np.broadcast_to(addition, (p, len(part))).copy()
                delta[active] += np.add.reduceat(correction, starts, axis=0)
                index = int(np.argmin(delta))
                drop, add = divmod(index, len(part))
                value = float(delta[drop, add])
                if value < best_delta:
                    best_delta = value
                    move = (drop, int(part[add]))
            if move is None:
                return s, cost, complete
            trial = s[:]
            trial[move[0]] = move[1]
            new_owner, new_first, new_second, new_cost = statistics(trial)
            # Accept moves on the fully recomputed true distance sum, preventing cycling caused by cancellation error.
            if not new_cost < cost:
                return s, cost, False
            s, cost = trial, new_cost
            owner, first, second = new_owner, new_first, new_second
            remember(s, cost)
        return s, cost, False

    def keep_elite(s, cost):
        key = tuple(sorted(s))
        if any(entry[1] == key for entry in elite):
            return
        elite.append((cost, key))
        elite.sort(key=lambda entry: entry[0])
        if len(elite) > pool_limit:
            elite.pop()

    def shake(s, depth):
        s = list(s)
        selected = np.zeros(n, dtype=bool)
        selected[s] = True
        counts = np.bincount(groups[s][groups[s] >= 0], minlength=group_count)
        # Never remove the same position twice, and every swap preserves sink coverage.
        positions = rng.permutation(p)
        changed = 0
        for position in positions:
            if changed >= depth or clock() >= deadline:
                break
            old = s[int(position)]
            g = int(groups[old])
            if g >= 0 and counts[g] == 1:
                candidates = members[g][~selected[members[g]]]
            else:
                candidates = np.flatnonzero(~selected)
            if len(candidates) == 0:
                continue
            new = int(rng.choice(candidates))
            selected[old] = False
            selected[new] = True
            if g >= 0:
                counts[g] -= 1
            h = int(groups[new])
            if h >= 0:
                counts[h] += 1
            s[int(position)] = new
            changed += 1
        return s

    def rebuild(s, depth, stop):
        counts = np.bincount(groups[s][groups[s] >= 0], minlength=group_count)
        removed = set()
        for position in rng.permutation(p):
            if len(removed) >= depth or clock() >= stop:
                break
            j = s[int(position)]
            g = int(groups[j])
            if g < 0 or counts[g] > 1:
                removed.add(j)
                if g >= 0:
                    counts[g] -= 1
        if not removed:
            return shake(s, depth)
        base = [j for j in s if j not in removed]
        return greedy(base, stop, randomized=True)

    # Within a sink, first pick a representative that is relatively central for the vertices that can reach it.
    base = representatives[:]
    setup_stop = min(deadline, clock() + 0.06 * budget)
    for g, vertices in enumerate(members):
        if clock() >= setup_stop:
            break
        reachable = np.flatnonzero(np.isfinite(d[:, base[g]]))
        value = math.inf
        for a in range(0, len(vertices), block):
            if clock() >= setup_stop:
                break
            part = vertices[a:a + block]
            costs = d[reachable[:, None], part[None, :]].sum(axis=0)
            k = int(np.argmin(costs))
            if float(costs[k]) < value:
                value = float(costs[k])
                base[g] = int(part[k])
    construction_stop = min(deadline, clock() + 0.14 * budget)
    current = greedy(base, construction_stop)
    current_cost = float(np.min(d[:, current], axis=1).sum())
    remember(current, current_cost)
    # After the initial local optimisation, no cap on restarts or iterations is set.
    current, current_cost, _ = descent(current, deadline)
    keep_elite(current, current_cost)
    stagnation = 0
    while clock() < deadline and best_cost > 0.0:
        before = best_cost
        depth = min(max_depth, 2 + int(math.sqrt(stagnation)))
        draw = float(rng.random())
        if draw < 0.55:
            anchor = answer["medians"][:]
        elif draw < 0.8 or not elite:
            anchor = current[:]
        else:
            anchor = list(elite[int(rng.integers(len(elite)))][1])
        remaining = deadline - clock()
        if remaining <= 0.0:
            break
        action = float(rng.random())
        repair_stop = min(deadline, clock() + remaining / (3.0 + math.log1p(p)))
        if action < 1.0 / (3.0 + math.sqrt(p)):
            trial = greedy(base, repair_stop, randomized=True)
            trial = shake(trial, min(depth, max_depth))
        elif action < 0.5:
            trial = rebuild(anchor, depth, repair_stop)
        else:
            trial = shake(anchor, depth)
        current, current_cost, _ = descent(trial, deadline)
        keep_elite(current, current_cost)
        if best_cost < before:
            stagnation = 0
        else:
            stagnation += 1
