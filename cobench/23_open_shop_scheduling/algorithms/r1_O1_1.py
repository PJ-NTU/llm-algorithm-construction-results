"""Critical-block incremental tabu search with bidirectional insertion
Builds a feasible schedule by event matching, and continually improves the makespan with insertion of all critical blocks, local topological incremental evaluation and bidirectional gap re-sequencing.
"""

import bisect
import heapq
import math
import random
import time


def solve(instance: dict, time_limit_s: float) -> dict:
    begin = time.perf_counter()
    try:
        budget = max(0.0, float(time_limit_s))
        if not math.isfinite(budget):
            budget = 0.0
    except Exception:
        budget = 0.0
    deadline = begin + max(0.0, budget - min(0.06, max(0.001, budget * 0.015)))
    now = time.perf_counter
    try:
        nj = int(instance["n_jobs"])
        nm = int(instance["n_machines"])
    except Exception:
        return {"start": []}
    if nj <= 0:
        return {"start": []}
    if nm <= 0:
        return {"start": [[] for _ in range(nj)]}
    best_grid = [[0] * nm for _ in range(nj)]
    try:
        p = [int(instance["ptimes"][j][m]) for j in range(nj) for m in range(nm)]
        total = 0
        for u, duration in enumerate(p):
            best_grid[u // nm][u % nm] = total
            total += duration
    except Exception:
        return {"start": best_grid}
    best_value = total
    if now() >= deadline:
        return {"start": best_grid}

    try:
        size = nj * nm
        job_of = [u // nm for u in range(size)]
        mac_of = [u % nm for u in range(size)]
        jl = [sum(p[j * nm:(j + 1) * nm]) for j in range(nj)]
        ml = [sum(p[m::nm]) for m in range(nm)]
        lower = max(jl + ml)
        if best_value <= lower:
            return {"start": best_grid}
        effort = math.log1p(budget)
        pool_cap = max(2, int(2 + effort * math.sqrt(nj + nm)))
        breadth = max(4, int((1 + effort) * math.sqrt(size)))
        reach = max(2, int(1 + effort * math.sqrt(max(nj, nm))))
        seed = size + 1000003 * nj + 9176 * nm
        for u, duration in enumerate(p):
            seed = (seed * 1000003 + (u + 17) * (duration + 31)) & 0xFFFFFFFFFFFF
        rng = random.Random(seed)
        mean = total / size
        pool = []
        signatures = set()
        best_rows = None

        # The resource permutations implicitly determine all zJ, zM; the longest path gives s and Cmax.
        # The longest path of every acyclic permutation does not exceed the total processing time, satisfying the big-M disjunctive constraints of the problem.
        def make_rows(sequence):
            rows = [[] for _ in range(nj + nm)]
            for u in sequence:
                rows[job_of[u]].append(u)
                rows[nj + mac_of[u]].append(u)
            return rows

        def rows_from_starts(starts):
            sequence = sorted(range(size), key=lambda u: (starts[u], p[u], u))
            return make_rows(sequence)

        def graph(rows):
            pred = [[-1] * size, [-1] * size]
            succ = [[-1] * size, [-1] * size]
            for r, row in enumerate(rows):
                k = int(r >= nj)
                for a, b in zip(row, row[1:]):
                    succ[k][a] = b
                    pred[k][b] = a
            return pred, succ

        def full(pred, succ, stop):
            pa, pb = pred
            sa, sb = succ
            degree = [int(pa[u] >= 0) + int(pb[u] >= 0) for u in range(size)]
            queue = [u for u in range(size) if degree[u] == 0]
            heads = [0] * size
            index = 0
            value = 0
            while index < len(queue):
                if not (index & 255) and now() >= stop:
                    return None
                u = queue[index]
                index += 1
                end = heads[u] + p[u]
                if end > value:
                    value = end
                a, b = sa[u], sb[u]
                if a >= 0:
                    if end > heads[a]:
                        heads[a] = end
                    degree[a] -= 1
                    if degree[a] == 0:
                        queue.append(a)
                if b >= 0:
                    if end > heads[b]:
                        heads[b] = end
                    degree[b] -= 1
                    if degree[b] == 0:
                        queue.append(b)
            if len(queue) != size:
                return None
            tails = [0] * size
            rank = [0] * size
            for i in range(size - 1, -1, -1):
                if not (i & 511) and now() >= stop:
                    return None
                u = queue[i]
                rank[u] = i
                a, b = sa[u], sb[u]
                ta = p[a] + tails[a] if a >= 0 else 0
                tb = p[b] + tails[b] if b >= 0 else 0
                tails[u] = max(ta, tb)
            return value, heads, tails, queue, rank

        def save(rows, state, archive=False):
            nonlocal best_value, best_grid, best_rows
            if state is None:
                return
            value, heads = state[:2]
            if value < best_value:
                best_value = value
                best_grid = [heads[j * nm:(j + 1) * nm] for j in range(nj)]
                best_rows = [row[:] for row in rows]
            if archive:
                signature = tuple(tuple(row) for row in rows)
                if signature not in signatures:
                    signatures.add(signature)
                    pool.append((value, [row[:] for row in rows], signature))
                    pool.sort(key=lambda item: item[0])
                    if len(pool) > pool_cap:
                        old = pool.pop()
                        signatures.discard(old[2])

        def install(rows, stop, archive=True):
            pred, succ = graph(rows)
            state = full(pred, succ, stop)
            save(rows, state, archive)
            return state

        def cyclic(randomized):
            jobs = sorted(range(nj), key=lambda j: -jl[j])
            machines = sorted(range(nm), key=lambda m: -ml[m])
            width = max(nj, nm)
            layers = list(range(width))
            if randomized:
                rng.shuffle(jobs)
                rng.shuffle(machines)
                rng.shuffle(layers)
            sequence = []
            for layer in layers:
                if now() >= deadline:
                    return None
                if nj <= nm:
                    sequence.extend(jobs[j] * nm + machines[(j + layer) % width]
                                    for j in range(nj))
                else:
                    sequence.extend(jobs[(m - layer) % width] * nm + machines[m]
                                    for m in range(nm))
            return make_rows(sequence)

        def matching(policy, stop):
            remaining = [True] * size
            jf, mf = [0] * nj, [0] * nm
            rj, rm = jl[:], ml[:]
            sequence = []
            current = 0
            mode = policy % 6
            noise = min(0.8, (policy // 6) / (4 + effort))
            while len(sequence) < size and now() < stop:
                idlej = [j for j in range(nj) if jf[j] <= current]
                idlem = [m for m in range(nm) if mf[m] <= current]
                preferences = {}
                urgency = {}
                for j in idlej:
                    if now() >= stop:
                        return None
                    ranked = []
                    for m in idlem:
                        u = j * nm + m
                        if not remaining[u]:
                            continue
                        d = p[u]
                        if mode == 0:
                            score = 4 * d + rj[j] + rm[m]
                        elif mode == 1:
                            score = 2.5 * min(rj[j], rm[m]) + 2 * d
                        elif mode == 2:
                            score = 3 * rm[m] + rj[j] + d
                        elif mode == 3:
                            score = 3 * rj[j] + rm[m] + d
                        elif mode == 4:
                            score = 2 * max(rj[j], rm[m]) + 3 * d
                        else:
                            score = 0.8 * (rj[j] + rm[m]) - d
                        if noise:
                            score += rng.random() * (abs(score) + d + 1) * noise
                        ranked.append((score, m))
                    ranked.sort(reverse=True)
                    preferences[j] = [m for score, m in ranked]
                    urgency[j] = ranked[0][0] if ranked else -1
                roots = sorted(idlej, key=lambda j: (urgency[j], rng.random()), reverse=True)
                mj, mm = {}, {}
                for root in roots:
                    if now() >= stop:
                        return None
                    queue = [root]
                    seenj, seenm = {root}, set()
                    parent = {}
                    free = -1
                    index = 0
                    while index < len(queue) and free < 0:
                        j = queue[index]
                        index += 1
                        for m in preferences[j]:
                            if m in seenm:
                                continue
                            seenm.add(m)
                            parent[m] = j
                            old = mm.get(m, -1)
                            if old < 0:
                                free = m
                                break
                            if old not in seenj:
                                seenj.add(old)
                                queue.append(old)
                    while free >= 0:
                        j = parent[free]
                        old = mj.get(j, -1)
                        mj[j] = free
                        mm[free] = j
                        free = old
                if mj:
                    has_zero = False
                    for j, m in mj.items():
                        u = j * nm + m
                        sequence.append(u)
                        remaining[u] = False
                        jf[j] = mf[m] = current + p[u]
                        rj[j] -= p[u]
                        rm[m] -= p[u]
                        has_zero = has_zero or p[u] == 0
                    if has_zero:
                        continue
                if len(sequence) < size:
                    future = [t for t in jf + mf if t > current]
                    if not future:
                        return None
                    current = min(future)
            return make_rows(sequence) if len(sequence) == size else None

        # Search for the earliest common gap of both resources, rather than only appending at the end of a resource.
        def gap_decode(sequence, stop):
            calendars = [[] for _ in range(nj + nm)]
            starts = [0] * size
            for count, u in enumerate(sequence):
                if not (count & 31) and now() >= stop:
                    return None
                left = calendars[job_of[u]]
                right = calendars[nj + mac_of[u]]
                i = k = t = 0
                duration = p[u]
                while True:
                    while i < len(left) and left[i][1] <= t:
                        i += 1
                    while k < len(right) and right[k][1] <= t:
                        k += 1
                    end = t + duration
                    advance = t
                    if i < len(left) and end > left[i][0]:
                        advance = max(advance, left[i][1])
                    if k < len(right) and end > right[k][0]:
                        advance = max(advance, right[k][1])
                    if advance == t:
                        break
                    t = advance
                starts[u] = t
                entry = (t, t + duration, u)
                bisect.insort(left, entry)
                bisect.insort(right, entry)
            return starts

        rows = cyclic(False)
        if rows is not None:
            install(rows, deadline)
        if best_value <= lower or now() >= deadline:
            return {"start": best_grid}
        construction_stop = min(deadline, begin + budget * min(0.17, 1 / (3 + effort)))
        policy = 0
        while now() < construction_stop and best_value > lower:
            rows = matching(policy, construction_stop)
            if rows is None:
                break
            install(rows, deadline)
            policy += 1

        def improve(rows, stop):
            pred, succ = graph(rows)
            state = full(pred, succ, stop)
            if state is None:
                return
            save(rows, state, True)
            tabu = {}
            iteration = 0
            changed_at = now()
            session_best = state[0]
            stamps = [0] * size
            ticket = 0
            phase_start = now()
            stagnation_span = max(0.001, (stop - phase_start) / (1.5 + effort / 4))

            # Temporarily change a contiguous segment of one resource; the order on other resources stays unchanged.
            def put_segment(resource, first, segment):
                row = rows[resource]
                kind = int(resource >= nj)
                pr, su = pred[kind], succ[kind]
                before = row[first - 1] if first else -1
                after_index = first + len(segment)
                after = row[after_index] if after_index < len(row) else -1
                row[first:after_index] = segment
                if before >= 0:
                    su[before] = segment[0]
                if after >= 0:
                    pr[after] = segment[-1]
                for k, u in enumerate(segment):
                    pr[u] = segment[k - 1] if k else before
                    su[u] = segment[k + 1] if k + 1 < len(segment) else after
                return after

            def incremental(lo, hi, extra, ceiling):
                # New reverse arcs only lie within this topological interval; outside it propagate along the original topological order.
                nonlocal ticket
                ticket += 1
                tag = ticket
                value, heads, tails, order, rank = state
                window = order[lo:hi + 1]
                degrees = []
                queue = []
                pa, pb = pred
                sa, sb = succ
                for u in window:
                    a, b = pa[u], pb[u]
                    degree = int(a >= 0 and lo <= rank[a] <= hi)
                    degree += int(b >= 0 and lo <= rank[b] <= hi)
                    degrees.append(degree)
                    if degree == 0:
                        queue.append(u)
                updates = []
                heap = []

                def enqueue(u):
                    if u >= 0 and rank[u] > hi and stamps[u] != tag:
                        stamps[u] = tag
                        heapq.heappush(heap, rank[u])

                enqueue(extra)
                index = 0
                try:
                    while index < len(queue):
                        if not (index & 127) and now() >= stop:
                            return None
                        u = queue[index]
                        index += 1
                        a, b = pa[u], pb[u]
                        h = max(heads[a] + p[a] if a >= 0 else 0,
                                heads[b] + p[b] if b >= 0 else 0)
                        altered = h != heads[u]
                        if altered:
                            updates.append((u, heads[u]))
                            heads[u] = h
                        for v in (sa[u], sb[u]):
                            if v < 0:
                                continue
                            rv = rank[v]
                            if lo <= rv <= hi:
                                k = rv - lo
                                degrees[k] -= 1
                                if degrees[k] == 0:
                                    queue.append(v)
                            elif altered:
                                enqueue(v)
                    if len(queue) != len(window):
                        return None
                    count = 0
                    while heap:
                        if not (count & 127) and now() >= stop:
                            return None
                        count += 1
                        u = order[heapq.heappop(heap)]
                        a, b = pa[u], pb[u]
                        h = max(heads[a] + p[a] if a >= 0 else 0,
                                heads[b] + p[b] if b >= 0 else 0)
                        if h != heads[u]:
                            updates.append((u, heads[u]))
                            heads[u] = h
                            if h + p[u] + tails[u] > ceiling:
                                return None
                            enqueue(sa[u])
                            enqueue(sb[u])
                    ends = [heads[row[-1]] + p[row[-1]] for row in rows]
                    return max(ends), sum(ends)
                finally:
                    for u, old in updates:
                        heads[u] = old

            while now() < stop and best_value > lower:
                value, heads, tails, order, rank = state
                critical = [heads[u] + p[u] + tails[u] == value for u in range(size)]
                moves = set()
                for r, row in enumerate(rows):
                    if now() >= stop:
                        break
                    length = len(row)
                    left = 0
                    for i in range(length):
                        u = row[i]
                        if critical[u]:
                            if i:
                                moves.add((r, i - 1, i))
                            if i + 1 < length:
                                moves.add((r, i, i + 1))
                        tight = (i + 1 < length and critical[u] and critical[row[i + 1]]
                                 and heads[row[i + 1]] == heads[u] + p[u])
                        if tight:
                            continue
                        if i > left:
                            for k in range(left + 1, i + 1):
                                if k - left <= reach or k == i:
                                    moves.add((r, left, k))
                                    moves.add((r, k, left))
                                if i - k + 1 <= reach or k == left + 1:
                                    moves.add((r, i, k - 1))
                                    moves.add((r, k - 1, i))
                        left = i + 1
                extras = max(1, int(effort * math.sqrt(nj + nm)))
                for _ in range(extras):
                    r = rng.randrange(nj + nm)
                    length = len(rows[r])
                    if length > 1:
                        a = rng.randrange(length)
                        radius = min(length - 1, reach)
                        b = max(0, min(length - 1, a + rng.randint(-radius, radius)))
                        if a != b:
                            moves.add((r, a, b))
                ranked = []
                for r, a, b in moves:
                    if now() >= stop:
                        break
                    row = rows[r]
                    first, last = min(a, b), max(a, b)
                    segment = row[first:last + 1]
                    moved = row[a]
                    segment.insert(b - first, segment.pop(a - first))
                    other = 1 - int(r >= nj)
                    cursor = heads[row[first - 1]] + p[row[first - 1]] if first else 0
                    estimate = 0
                    for u in segment:
                        v = pred[other][u]
                        cursor = max(cursor, heads[v] + p[v] if v >= 0 else 0) + p[u]
                        v = succ[other][u]
                        estimate = max(estimate, cursor + (p[v] + tails[v] if v >= 0 else 0))
                    if last + 1 < len(row):
                        u = row[last + 1]
                        estimate = max(estimate, cursor + p[u] + tails[u])
                    crossed = row[a + 1:b + 1] if a < b else row[b:a]
                    # Forbid recently reversed relative orders; the tabu is overridden when the global best is truly improved.
                    forbidden = any(tabu.get((r, moved, v) if a < b else (r, v, moved), -1)
                                    > iteration for v in crossed)
                    ranked.append((estimate, rng.random(), r, a, b, forbidden))
                ranked.sort()
                chosen = None
                chosen_score = (float("inf"), float("inf"))
                # Top-ranked moves and a random tail both receive exact evaluation, so estimation errors do not exclude good moves.
                prefix = ranked[:breadth]
                rest = ranked[breadth:]
                rng.shuffle(rest)
                candidates = prefix + rest[:max(1, int(math.sqrt(breadth) * (1 + effort)))]
                for estimate, tie, r, a, b, forbidden in candidates:
                    if now() >= stop:
                        break
                    row = rows[r]
                    first, last = min(a, b), max(a, b)
                    original = row[first:last + 1]
                    lo, hi = rank[original[0]], rank[original[-1]]
                    replacement = original[:]
                    replacement.insert(b - first, replacement.pop(a - first))
                    extra = put_segment(r, first, replacement)
                    ceiling = min(chosen_score[0], best_value - 1) if forbidden else chosen_score[0]
                    try:
                        score = incremental(lo, hi, extra, ceiling)
                    finally:
                        put_segment(r, first, original)
                    if score is not None and (not forbidden or score[0] < best_value):
                        if score < chosen_score:
                            chosen_score = score
                            chosen = (r, a, b, replacement)
                if chosen is None or now() >= stop:
                    break
                r, a, b, replacement = chosen
                row = rows[r]
                moved = row[a]
                crossed = row[a + 1:b + 1] if a < b else row[b:a]
                tenure = max(2, int((1 + effort) * math.sqrt(nj + nm)))
                expiry = iteration + tenure + rng.randrange(tenure)
                for v in crossed:
                    key = (r, v, moved) if a < b else (r, moved, v)
                    tabu[key] = expiry
                put_segment(r, min(a, b), replacement)
                state = full(pred, succ, stop)
                if state is None:
                    break
                iteration += 1
                save(rows, state)
                if state[0] < session_best:
                    session_best = state[0]
                    changed_at = now()
                    save(rows, state, True)
                if now() - changed_at > stagnation_span:
                    break
                if len(tabu) > size * (2 + effort):
                    tabu = {key: expiry for key, expiry in tabu.items() if expiry > iteration}
            if state is not None:
                save(rows, state, True)

        # Each phase ends on the remaining wall-clock budget; the number of restarts is unlimited.
        restart = 0
        failures = 0
        while now() < deadline and best_value > lower:
            previous_best = best_value
            remaining_time = deadline - now()
            phase_stop = min(deadline, now() + min(budget / (2 + effort), remaining_time))
            if restart == 0 and pool:
                rows = [row[:] for row in pool[0][1]]
            elif restart % 4 == 3:
                seed_stop = min(phase_stop, now() + remaining_time / (4 + effort))
                rows = matching(policy, seed_stop)
                policy += 1
                if rows is None and now() < phase_stop:
                    rows = cyclic(True)
            else:
                if best_rows is None:
                    rows = cyclic(True)
                else:
                    if pool and rng.random() < 1 / (2 + failures):
                        source = pool[rng.randrange(len(pool))][1]
                    else:
                        source = best_rows
                    rows = [row[:] for row in source]
                    state = install(rows, phase_stop, False)
                    if state is not None:
                        heads = state[1]
                        if restart % 2:
                            sequence = sorted(range(size), key=lambda u: (-(heads[u] + p[u]), -heads[u], u))
                            reverse = gap_decode(sequence, phase_stop)
                            if reverse is not None:
                                reverse_rows = rows_from_starts(reverse)
                                install(reverse_rows, phase_stop)
                                sequence = sorted(range(size), key=lambda u: (-(reverse[u] + p[u]), -reverse[u], u))
                                forward = gap_decode(sequence, phase_stop)
                                if forward is not None:
                                    rows = rows_from_starts(forward)
                        else:
                            amplitude = mean * (0.1 + math.log1p(failures + effort) / (2 + effort))
                            sequence = sorted(range(size), key=lambda u: (heads[u] + rng.uniform(-amplitude, amplitude), u))
                            decoded = gap_decode(sequence, phase_stop)
                            if decoded is not None:
                                rows = rows_from_starts(decoded)
            if rows is not None and now() < phase_stop:
                improve(rows, phase_stop)
            failures = failures + 1 if best_value >= previous_best else 0
            restart += 1
    except Exception:
        # Saved solutions come from serial construction or from complete schedules that passed the acyclicity check.
        pass
    return {"start": best_grid}
