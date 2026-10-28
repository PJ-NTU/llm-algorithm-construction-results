"""Critical cross-section destroy-repair with joint reinsertion search
Combines matching construction, dual-resource tabu reinsertion, multi-operation removal at critical cross-sections and regret repair, and keeps improving with forward and backward gap re-sequencing.
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

        def decode(state, stop, excluded=(), chains=None):
            jp, jn, mp, mn = state
            degree = [int(jp[u] >= 0) + int(mp[u] >= 0)
                      for u in range(size)]
            queue = [u for u in range(size)
                     if not degree[u] and u not in excluded]
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
            if pos != size - len(excluded):
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

        def insertion(work, u, rowj, rowm, excluded, stop,
                      old=None, tabu=None, step=0, ceiling=float("inf"),
                      regret=False):
            """Compute dual-resource slots exactly; during multi-operation repair also record the second-best slot."""
            result = decode(work, stop, excluded, (rowj, rowm))
            if result is None or result[0] > ceiling:
                return None
            base, head, tail, _, aj, am = result
            jpred, jsucc = [-1] + rowj, rowj + [-1]
            mpred, msucc = [-1] + rowm, rowm + [-1]
            hj = [0] + [head[v] + p[v] for v in rowj]
            hm = [0] + [head[v] + p[v] for v in rowm]
            tj = [tail[v] for v in rowj] + [0]
            tm = [tail[v] for v in rowm] + [0]
            negative_tm = [-x for x in tm]
            ancestor_j = [aj[v] for v in rowm]
            best = (ceiling, float("inf"))
            second = (float("inf"), float("inf"))
            answer, ties = None, 0
            pu = p[u]
            for k, a in enumerate(jpred):
                if clock() >= stop:
                    break
                lo = am[a] + 1 if a >= 0 else 0
                hi = bisect.bisect_left(ancestor_j, k)
                hja, tja = hj[k], tj[k]
                if not regret:
                    # The head/tail distances are monotone, so the current best evaluation shrinks the slot interval; no better solution is lost.
                    cap = min(best) - pu
                    if hja + tja > cap:
                        continue
                    lo = max(lo, bisect.bisect_left(negative_tm, hja - cap))
                    hi = min(hi, bisect.bisect_right(hm, cap - tja) - 1)
                for l in range(lo, hi + 1):
                    if not (l & 127) and clock() >= stop:
                        break
                    c = mpred[l]
                    if old is not None and a == old[0] and c == old[2]:
                        continue
                    start = hja if hja >= hm[l] else hm[l]
                    finish_tail = tja if tja >= tm[l] else tm[l]
                    through = start + pu + finish_tail
                    value = base if base >= through else through
                    key = (value, through)
                    if value > ceiling:
                        continue
                    if (tabu is not None and tabu.get((u, a, c), -1) > step
                            and value >= best_value):
                        continue
                    if key < best:
                        if answer is not None:
                            second = best
                        best, ties = key, 1
                        answer = (value, through, u, a, jsucc[k], c,
                                  msucc[l], k, l)
                    elif key == best:
                        second = best
                        ties += 1
                        if answer is None or rng.randrange(ties) == 0:
                            answer = (value, through, u, a, jsucc[k], c,
                                      msucc[l], k, l)
                    elif regret and key < second:
                        second = key
            if answer is None:
                return None
            if regret:
                if math.isfinite(second[0]):
                    loss = second[0] - best[0] + (second[1] - best[1]) / size
                else:
                    loss = best_value + pu
            else:
                loss = 0.0
            return answer + (loss,)

        def reinsert(u, rows, tabu, step, ceiling, stop):
            rowj = [v for v in rows[job[u]] if v != u]
            rowm = [v for v in rows[nj + machine[u]] if v != u]
            old = detach(state, u)
            try:
                return insertion(state, u, rowj, rowm, (u,), stop,
                                 old, tabu, step, ceiling)
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

        def justify(source, stop):
            """Backward then forward noise-free gap re-sequencing; only results that do not increase the makespan are accepted."""
            original = decode(source, stop)
            if original is None:
                return None
            work, result = source, original
            for reverse in (True, False):
                if clock() >= stop:
                    break
                trial = rebuild(work, 0.0, reverse, stop)
                if trial is None:
                    break
                decoded = decode(trial, stop)
                if decoded is not None and decoded[0] <= result[0]:
                    work, result = trial, decoded
                    remember(work, result)
            return work, result

        def cut_repair(source, depth, stop):
            """Remove as a group the operations spanning a critical time cross-section, then expand related neighbours and jointly reinsert them."""
            reverse = rng.random() < 0.5
            work = ([source[1][:], source[0][:], source[3][:], source[2][:]]
                    if reverse else [a[:] for a in source])
            result = decode(work, stop)
            if result is None:
                return None
            value, head, tail, topology = result[:4]
            critical = [u for u in topology if head[u] + tail[u] == value]
            if not critical:
                return None
            pivot = rng.choice(critical)
            crossing = head[pivot] + p[pivot] * rng.random()
            removed = {u for u in critical
                       if head[u] <= crossing < head[u] + p[u]}
            if not removed:
                removed.add(pivot)
            # Do not truncate the cross-section set, so that all critical paths crossing this time are destroyed simultaneously.
            target = min(size, max(depth, len(removed)))
            frontier = list(removed)
            ranked = sorted(critical,
                            key=lambda u: (abs(head[u] + 0.5 * p[u] - crossing),
                                           rng.random()))
            while len(removed) < target:
                if clock() >= stop:
                    return None
                if frontier:
                    u = frontier.pop(rng.randrange(len(frontier)))
                    neighbors = [chain[u] for chain in work if chain[u] >= 0]
                    rng.shuffle(neighbors)
                    neighbors.sort(key=lambda v: value - head[v] - tail[v])
                    for v in neighbors:
                        if v not in removed:
                            removed.add(v)
                            frontier.append(v)
                            if len(removed) >= target:
                                break
                else:
                    u = next((v for v in ranked if v not in removed), -1)
                    if u < 0:
                        u = rng.randrange(size)
                    if u not in removed:
                        removed.add(u)
                        frontier.append(u)
            rows = [[] for _ in range(nj + nm)]
            for u in topology:
                if u not in removed:
                    rows[job[u]].append(u)
                    rows[nj + machine[u]].append(u)
            for u in removed:
                detach(work, u)
            pending = list(removed)
            rng.shuffle(pending)
            # For the sampled operations awaiting insertion, compare the regret between the best and second-best slots and repair restricted operations first.
            width = max(1, int(math.ceil(math.sqrt(strength))))
            while pending and clock() < stop:
                chosen, priority = None, None
                sample = min(len(pending), width + int(rng.random() < 0.5))
                for u in pending[:sample]:
                    candidate = insertion(work, u, rows[job[u]],
                                          rows[nj + machine[u]], removed,
                                          stop, regret=True)
                    if candidate is None:
                        return None
                    rank = (candidate[9], p[u], -candidate[0])
                    if priority is None or rank > priority:
                        chosen, priority = candidate, rank
                    if clock() >= stop:
                        return None
                _, _, u, a, b, c, d, k, l, _ = chosen
                attach(work, u, a, b, c, d)
                rows[job[u]].insert(k, u)
                rows[nj + machine[u]].insert(l, u)
                removed.remove(u)
                pending.remove(u)
                if pending:
                    # Operations not selected are kept, while the next round of sampling is changed, avoiding a fixed repair order.
                    offset = rng.randrange(len(pending))
                    pending = pending[offset:] + pending[:offset]
            if pending:
                return None
            if reverse:
                work = [work[1], work[0], work[3], work[2]]
            result = decode(work, stop)
            if result is None:
                return None
            remember(work, result)
            return work, result

        state = [a[:] for a in best_state]
        current = decode(state, deadline)
        if current is None:
            return {"start": best_start}
        tabu, step, restart = {}, 0, 0
        last_gain = clock()
        epoch = budget / (2.0 + math.sqrt(strength) + math.log1p(nj + nm))
        trial_seconds = max(1e-6, (clock() - begin) / max(1, size))
        tenure = max(2, int(strength + math.sqrt(nj + nm)))
        repair_failures = 0
        last_repair = clock()
        repair_spacing = epoch / (1.0 + math.sqrt(strength))
        max_depth = min(size, max(2, int(strength * math.sqrt(nj + nm))))
        polished = justify(state, min(deadline, clock() + repair_spacing))
        if polished is not None:
            state, current = polished
            remember(state, current, True)

        # The search stops only at the wall-clock deadline or on reaching a valid lower bound; the best complete solution and the working solution are stored separately.
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
            if chosen is not None and clock() < deadline:
                _, _, u, a, b, c, d = chosen[:7]
                previous = detach(state, u)
                attach(state, u, a, b, c, d)
                updated = decode(state, deadline)
                if updated is None:
                    detach(state, u)
                    attach(state, u, *previous)
                    # A timeout only abandons this move; the existing complete best solution is kept.
                    if clock() >= deadline:
                        break
                    chosen = None
                else:
                    step += 1
                    tabu[(u, previous[0], previous[2])] = step + tenure + rng.randrange(tenure)
                    if len(tabu) > size + tenure * 2:
                        tabu = {key: expiry for key, expiry in tabu.items() if expiry > step}
                    old_best = best_value
                    remember(state, updated)
                    if best_value < old_best:
                        last_gain = clock()
                        repair_failures = 0
                    current = updated

            now = clock()
            if best_value <= lower or now >= deadline:
                break
            # The multi-operation neighbourhood uses its own time slice; each failure gradually increases the removal depth.
            if chosen is None or now - max(last_gain, last_repair) >= repair_spacing:
                remember(state, current, True)
                repair_stop = min(deadline, now + max(repair_spacing / math.sqrt(strength),
                                                     trial_seconds * strength))
                previous_best = best_value
                candidate_best = None
                while clock() < repair_stop and best_value > lower:
                    scale = math.sqrt(1.0 + repair_failures / strength)
                    depth = min(max_depth, max(2, int(math.sqrt(nj + nm) *
                                    math.sqrt(strength) * scale * rng.uniform(0.5, 1.0))))
                    draw = rng.random()
                    if draw < 0.5:
                        source = best_state
                    elif draw < 0.8:
                        source = state
                    else:
                        source = pool[rng.randrange(len(pool))][1]
                    repaired = cut_repair(source, depth, repair_stop)
                    if repaired is None:
                        break
                    work, result = repaired
                    if candidate_best is None or result[0] < candidate_best[1][0]:
                        candidate_best = repaired
                    if result[0] <= current[0]:
                        squeezed = justify(work, repair_stop)
                        if squeezed is not None and squeezed[1][0] <= result[0]:
                            work, result = squeezed
                            if result[0] <= candidate_best[1][0]:
                                candidate_best = squeezed
                    if best_value < previous_best:
                        break
                    repair_failures += 1
                if candidate_best is not None:
                    work, result = candidate_best
                    allowance = mean * math.sqrt(strength) * (1.0 + math.log1p(repair_failures))
                    gap = max(0, result[0] - current[0])
                    if result[0] <= current[0] or rng.random() < math.exp(-gap / max(1.0, allowance)):
                        state, current = work, result
                        tabu.clear()
                        remember(state, current, True)
                if best_value < previous_best:
                    last_gain = clock()
                    repair_failures = 0
                last_repair = clock()

            if clock() >= deadline or best_value <= lower:
                break
            if clock() - last_gain >= max(epoch, trial_seconds * strength):
                remember(state, current, True)
                restart += 1
                repair_stop = min(deadline, clock() + max(repair_spacing,
                                                        trial_seconds * strength))
                if restart % 4 == 0:
                    fresh = matching_state(restart % 4,
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
                last_repair = clock()
        return {"start": best_start}
    except Exception:
        # Input is as given by the problem; a search exception never overwrites an already constructed complete feasible solution.
        return {"start": best_start}
