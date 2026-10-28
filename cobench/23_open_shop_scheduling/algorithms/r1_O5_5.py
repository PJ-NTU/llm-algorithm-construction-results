"""Joint dual-resource reinsertion tabu search
Builds an initial solution by matching, evaluates dual-resource reinsertion with longest paths and reachability bounds, and keeps improving through perturbed gap reconstruction.
"""

import bisect
import math
import random
import time


def solve(instance: dict, time_limit_s: float) -> dict:
    begin = time.perf_counter()
    best_start = []
    try:
        nj = max(0, int(instance["n_jobs"]))
        nm = max(0, int(instance["n_machines"]))
        best_start = [[0] * nm for _ in range(nj)]
        p = [int(instance["ptimes"][j][m])
             for j in range(nj) for m in range(nm)]
        cursor = 0
        for u, duration in enumerate(p):
            best_start[u // nm][u % nm] = cursor
            cursor += duration
        best_value = cursor
        budget = max(0.0, float(time_limit_s))
        if not math.isfinite(budget):
            budget = 0.0
        deadline = begin + budget - max(0.001, budget * 0.025)
        clock = time.perf_counter
        if not nj or not nm or clock() >= deadline:
            return {"start": best_start}

        size = nj * nm
        job = [u // nm for u in range(size)]
        machine = [u % nm for u in range(size)]
        jl = [sum(p[j * nm:(j + 1) * nm]) for j in range(nj)]
        ml = [sum(p[m::nm]) for m in range(nm)]
        lower = max(jl + ml)
        mean = max(1.0, cursor / size)
        strength = 1.0 + math.log1p(budget)
        rng = random.Random(nj * 1000003 + nm * 9176 +
                            sum((u + 1) * x for u, x in enumerate(p)))
        best_state = None
        pool = []
        pool_capacity = max(2, int(strength + math.log1p(size)))

        # Four chains directly encode the order within the same job and the same machine; their transitive closure encodes all zJ, zM.
        def from_sequence(sequence):
            jp, jn, mp, mn = ([-1] * size for _ in range(4))
            lastj, lastm = [-1] * nj, [-1] * nm
            for u in sequence:
                j, m = job[u], machine[u]
                a, c = lastj[j], lastm[m]
                jp[u], mp[u] = a, c
                if a >= 0:
                    jn[a] = u
                if c >= 0:
                    mn[c] = u
                lastj[j], lastm[m] = u, u
            return [jp, jn, mp, mn]

        def decode(state, stop, omitted=-1, chains=None):
            jp, jn, mp, mn = state
            degree = [int(jp[u] >= 0) + int(mp[u] >= 0)
                      for u in range(size)]
            queue = [u for u in range(size)
                     if not degree[u] and u != omitted]
            head = [0] * size
            aj = am = None
            if chains is not None:
                aj, am = [-1] * size, [-1] * size
                for k, u in enumerate(chains[0]):
                    aj[u] = k
                for k, u in enumerate(chains[1]):
                    am[u] = k
            pos, value = 0, 0
            while pos < len(queue):
                if not (pos & 255) and clock() >= stop:
                    return None
                u = queue[pos]
                pos += 1
                end = head[u] + p[u]
                if end > value:
                    value = end
                for v in (jn[u], mn[u]):
                    if v < 0:
                        continue
                    if end > head[v]:
                        head[v] = end
                    if aj is not None:
                        if aj[u] > aj[v]:
                            aj[v] = aj[u]
                        if am[u] > am[v]:
                            am[v] = am[u]
                    degree[v] -= 1
                    if not degree[v]:
                        queue.append(v)
            if pos != size - int(omitted >= 0):
                return None
            tail = [0] * size
            for k in range(pos - 1, -1, -1):
                if not (k & 255) and clock() >= stop:
                    return None
                u = queue[k]
                a, b = jn[u], mn[u]
                ta = tail[a] if a >= 0 else 0
                tb = tail[b] if b >= 0 else 0
                tail[u] = p[u] + max(ta, tb)
            return value, head, tail, queue, aj, am

        def remember(state, result, keep=False):
            nonlocal best_value, best_start, best_state
            if result is None:
                return
            value, starts = result[:2]
            if best_state is None or value < best_value:
                if value <= best_value:
                    best_value = value
                    best_start = [starts[j * nm:(j + 1) * nm]
                                  for j in range(nj)]
                    best_state = [a[:] for a in state]
            if keep:
                signature = (tuple(state[1]), tuple(state[3]))
                if not any(item[2] == signature for item in pool):
                    pool.append((value, [a[:] for a in state], signature))
                    pool.sort(key=lambda item: item[0])
                    del pool[pool_capacity:]

        def layer_state():
            js = sorted(range(nj), key=lambda j: -jl[j])
            ms = sorted(range(nm), key=lambda m: -ml[m])
            sequence = []
            if nj <= nm:
                for layer in range(nm):
                    for k, j in enumerate(js):
                        sequence.append(j * nm + ms[(k + layer) % nm])
            else:
                for layer in range(nj):
                    for k, m in enumerate(ms):
                        sequence.append(js[(k + layer) % nj] * nm + m)
            return from_sequence(sequence)

        state = layer_state()
        current = decode(state, deadline)
        remember(state, current, True)
        if best_value <= lower or clock() >= deadline:
            return {"start": best_start}

        def matching_state(mode, noise, stop):
            remaining = [True] * size
            rj, rm = jl[:], ml[:]
            fj, fm = [0] * nj, [0] * nm
            sequence = []
            now = 0
            while len(sequence) < size:
                if clock() >= stop:
                    return None
                idlej = [j for j in range(nj) if fj[j] <= now]
                idlem = [m for m in range(nm) if fm[m] <= now]
                preferences, urgency = {}, {}
                for j in idlej:
                    if clock() >= stop:
                        return None
                    ranked = []
                    for m in idlem:
                        u = j * nm + m
                        if not remaining[u]:
                            continue
                        if mode == 0:
                            score = 4 * p[u] + rj[j] + rm[m]
                        elif mode == 1:
                            score = 2 * p[u] + 2.5 * min(rj[j], rm[m])
                        elif mode == 2:
                            score = p[u] + 3 * rm[m] + rj[j]
                        else:
                            score = p[u] + 3 * rj[j] + rm[m]
                        score += noise * (abs(score) + 1) * rng.random()
                        ranked.append((score, m))
                    ranked.sort(reverse=True)
                    preferences[j] = [m for _, m in ranked]
                    urgency[j] = ranked[0][0] if ranked else -1
                idlej.sort(key=lambda j: urgency[j], reverse=True)
                matchj, matchm = {}, {}
                for root in idlej:
                    if clock() >= stop:
                        return None
                    queue, seenj, seenm, parent = [root], {root}, set(), {}
                    pos, free = 0, -1
                    while pos < len(queue) and free < 0:
                        j = queue[pos]
                        pos += 1
                        for m in preferences[j]:
                            if m in seenm:
                                continue
                            seenm.add(m)
                            parent[m] = j
                            old = matchm.get(m)
                            if old is None:
                                free = m
                                break
                            if old not in seenj:
                                seenj.add(old)
                                queue.append(old)
                    while free >= 0:
                        j = parent[free]
                        old = matchj.get(j, -1)
                        matchj[j], matchm[free] = free, j
                        free = old
                for j, m in matchj.items():
                    u = j * nm + m
                    remaining[u] = False
                    sequence.append(u)
                    fj[j] = fm[m] = now + p[u]
                    rj[j] -= p[u]
                    rm[m] -= p[u]
                if len(sequence) == size:
                    break
                # After a zero-duration operation completes, keep filling the current time first.
                if any(p[j * nm + m] == 0 for j, m in matchj.items()):
                    continue
                future = [x for x in fj + fm if x > now]
                if not future:
                    return None
                now = min(future)
            return from_sequence(sequence)

        # The construction share varies continuously with the budget and is not capped by a fixed number of constructions.
        construction_stop = min(deadline, begin + budget / (4.0 + strength))
        seed_index = 0
        while clock() < construction_stop and best_value > lower:
            trial = matching_state(seed_index % 4,
                                   0.035 * math.log1p(seed_index // 4),
                                   construction_stop)
            if trial is None:
                break
            result = decode(trial, deadline)
            remember(trial, result, True)
            seed_index += 1
        if best_state is None or best_value <= lower or clock() >= deadline:
            return {"start": best_start}

        def detach(state, u):
            jp, jn, mp, mn = state
            a, b, c, d = jp[u], jn[u], mp[u], mn[u]
            if a >= 0:
                jn[a] = b
            if b >= 0:
                jp[b] = a
            if c >= 0:
                mn[c] = d
            if d >= 0:
                mp[d] = c
            jp[u] = jn[u] = mp[u] = mn[u] = -1
            return a, b, c, d

        def attach(state, u, a, b, c, d):
            jp, jn, mp, mn = state
            jp[u], jn[u], mp[u], mn[u] = a, b, c, d
            if a >= 0:
                jn[a] = u
            if b >= 0:
                jp[b] = u
            if c >= 0:
                mn[c] = u
            if d >= 0:
                mp[d] = u

        def reinsert(u, rows, tabu, step, ceiling, stop):
            """Jointly enumerate the two insertion slots, using ancestor bounds to exclude all cyclic combinations."""
            rowj = [v for v in rows[job[u]] if v != u]
            rowm = [v for v in rows[nj + machine[u]] if v != u]
            old = detach(state, u)
            answer = None
            try:
                result = decode(state, stop, u, (rowj, rowm))
                if result is None or result[0] > ceiling:
                    return None
                base, head, tail, _, aj, am = result
                jpred, jsucc = [-1] + rowj, rowj + [-1]
                mpred, msucc = [-1] + rowm, rowm + [-1]
                hj = [0] + [head[v] + p[v] for v in rowj]
                hm = [0] + [head[v] + p[v] for v in rowm]
                tj = [tail[v] for v in rowj] + [0]
                tm = [tail[v] for v in rowm] + [0]
                ancestor_j = [aj[v] for v in rowm]
                best, best_through, ties = ceiling, float("inf"), 0
                pu = p[u]
                for k, a in enumerate(jpred):
                    if clock() >= stop:
                        return answer
                    # Predecessor a must lie before the machine successor; the machine predecessor must not depend on the job successor.
                    lo = am[a] + 1 if a >= 0 else 0
                    hi = bisect.bisect_left(ancestor_j, k)
                    hja, tja = hj[k], tj[k]
                    for l in range(lo, hi + 1):
                        if not (l & 255) and clock() >= stop:
                            return answer
                        c = mpred[l]
                        if a == old[0] and c == old[2]:
                            continue
                        start = hja if hja >= hm[l] else hm[l]
                        finish_tail = tja if tja >= tm[l] else tm[l]
                        through = start + pu + finish_tail
                        value = base if base >= through else through
                        if value > best or (value == best and through > best_through):
                            continue
                        if tabu.get((u, a, c), -1) > step and value >= best_value:
                            continue
                        if value == best and through == best_through:
                            ties += 1
                            if rng.randrange(ties) != 0:
                                continue
                        else:
                            ties = 1
                        best, best_through = value, through
                        answer = (value, through, u, a, jsucc[k], c, msucc[l])
                return answer
            finally:
                attach(state, u, *old)

        def rebuild(source, noise, reverse, stop):
            """Perturb the existing time priorities and rebuild a complete schedule in the common gaps of both resource types."""
            source_state = ([source[1][:], source[0][:],
                             source[3][:], source[2][:]] if reverse
                            else [a[:] for a in source])
            result = decode(source_state, stop)
            if result is None:
                return None
            head = result[1]
            sequence = sorted(range(size), key=lambda u:
                              (head[u] + rng.uniform(-noise, noise) * mean, -p[u]))
            calendars = [[] for _ in range(nj + nm)]
            starts = [0] * size
            for index, u in enumerate(sequence):
                if not (index & 31) and clock() >= stop:
                    return None
                first, second = calendars[job[u]], calendars[nj + machine[u]]
                i = k = t = 0
                duration = p[u]
                while i < len(first) or k < len(second):
                    if k >= len(second) or (i < len(first) and first[i] <= second[k]):
                        a, b, _ = first[i]
                        i += 1
                    else:
                        a, b, _ = second[k]
                        k += 1
                    if b <= t:
                        continue
                    if a >= t + duration:
                        break
                    t = b
                starts[u] = t
                interval = (t, t + duration, u)
                bisect.insort(first, interval)
                bisect.insort(second, interval)
            sequence.sort(key=lambda u: (starts[u], starts[u] + p[u], u))
            rebuilt = from_sequence(sequence)
            if reverse:
                rebuilt = [rebuilt[1], rebuilt[0], rebuilt[3], rebuilt[2]]
            return rebuilt

        state = [a[:] for a in best_state]
        current = decode(state, deadline)
        if current is None:
            return {"start": best_start}
        tabu, step, restart = {}, 0, 0
        last_gain = clock()
        epoch = budget / (2.0 + math.sqrt(strength) + math.log1p(nj + nm))
        trial_seconds = max(1e-6, (clock() - begin) / max(1, size))
        tenure = max(2, int(strength + math.sqrt(nj + nm)))

        # The only search termination conditions are the wall-clock deadline or reaching a valid lower bound.
        while clock() < deadline and best_value > lower:
            value, head, tail, topology = current[:4]
            rows = [[] for _ in range(nj + nm)]
            critical, nearby = [], []
            slack_window = mean * min(1.0, (clock() - last_gain) / max(epoch, 1e-6))
            for u in topology:
                rows[job[u]].append(u)
                rows[nj + machine[u]].append(u)
                slack = value - head[u] - tail[u]
                if slack == 0:
                    critical.append(u)
                elif slack <= slack_window:
                    nearby.append(u)
            rng.shuffle(critical)
            rng.shuffle(nearby)
            breadth = min(len(critical), max(1, int(strength * math.sqrt(len(critical)))))
            targets = critical[:breadth]
            targets.extend(nearby[:max(1, int(strength))])
            chosen = None
            neighborhood_stop = min(deadline, clock() + max(epoch / strength,
                                                            trial_seconds * strength))
            for u in targets:
                if clock() >= neighborhood_stop:
                    break
                tick = clock()
                candidate = reinsert(u, rows, tabu, step,
                                     chosen[0] if chosen is not None else float("inf"),
                                     neighborhood_stop)
                trial_seconds = 0.8 * trial_seconds + 0.2 * (clock() - tick)
                if candidate is not None:
                    if (chosen is None or candidate[:2] < chosen[:2]
                            or (candidate[:2] == chosen[:2] and rng.random() < 0.5)):
                        chosen = candidate
                    if chosen[0] < value:
                        break
            if chosen is not None and clock() + trial_seconds < deadline:
                _, _, u, a, b, c, d = chosen
                previous = detach(state, u)
                attach(state, u, a, b, c, d)
                updated = decode(state, deadline)
                if updated is None:
                    break
                step += 1
                tabu[(u, previous[0], previous[2])] = step + tenure + rng.randrange(tenure)
                if len(tabu) > size + tenure * 2:
                    tabu = {key: expiry for key, expiry in tabu.items() if expiry > step}
                old_best = best_value
                remember(state, updated)
                if best_value < old_best:
                    last_gain = clock()
                current = updated
            elif clock() + trial_seconds >= deadline:
                break

            if chosen is None or clock() - last_gain >= max(epoch, trial_seconds * strength):
                if clock() + trial_seconds >= deadline:
                    break
                remember(state, current, True)
                restart += 1
                repair_stop = min(deadline, clock() + max(epoch, trial_seconds * strength))
                if restart % 4 == 0:
                    fresh = matching_state(restart % 3,
                                           0.08 * math.log1p(restart) * strength,
                                           repair_stop)
                else:
                    source = best_state if restart % 3 else pool[rng.randrange(len(pool))][1]
                    noise = (0.25 + rng.random()) * strength * math.sqrt(1 + restart / strength)
                    fresh = rebuild(source, noise, bool(restart % 2), repair_stop)
                if fresh is not None:
                    result = decode(fresh, deadline)
                    if result is not None:
                        state, current = fresh, result
                        remember(state, current, True)
                tabu.clear()
                last_gain = clock()
        return {"start": best_start}
    except Exception:
        # A search exception never overwrites an already completely constructed feasible solution.
        return {"start": best_start}
