"""All-critical-block insertion tabu search
Builds starting points by forward and backward active scheduling, screens critical-block insertions with head/tail summaries, verifies on the full graph and improves the makespan with time-adaptive elite restarts.
"""

import heapq
import math
import random
import time


def _evaluate(p, jp, js, mp, ms):
    n = len(p)
    degree = [(jp[u] >= 0) + (mp[u] >= 0) for u in range(n)]
    ready = [u for u in range(n) if degree[u] == 0]
    h = [0.0] * n
    order = []
    cmax = 0.0
    while ready:
        u = ready.pop()
        order.append(u)
        end = h[u] + p[u]
        if end > cmax:
            cmax = end
        v = js[u]
        if v >= 0:
            if h[v] < end:
                h[v] = end
            degree[v] -= 1
            if degree[v] == 0:
                ready.append(v)
        v = ms[u]
        if v >= 0:
            if h[v] < end:
                h[v] = end
            degree[v] -= 1
            if degree[v] == 0:
                ready.append(v)
    if len(order) != n:
        return None
    tail = p[:]
    for u in reversed(order):
        q = 0.0
        v = js[u]
        if v >= 0:
            q = tail[v]
        v = ms[u]
        if v >= 0 and tail[v] > q:
            q = tail[v]
        tail[u] = p[u] + q
    return h, tail, cmax


def _links(seqs, n):
    mp = [-1] * n
    ms = [-1] * n
    roots = [-1] * len(seqs)
    for m, seq in enumerate(seqs):
        if seq:
            roots[m] = seq[0]
        for i in range(1, len(seq)):
            a, b = seq[i - 1], seq[i]
            ms[a] = b
            mp[b] = a
    return mp, ms, roots


def _move(u, v, after, mp, ms, roots, machine):
    a, b = mp[u], ms[u]
    if a >= 0:
        ms[a] = b
    else:
        roots[machine[u]] = b
    if b >= 0:
        mp[b] = a
    if after:
        x, y = v, ms[v]
    else:
        x, y = mp[v], v
    mp[u], ms[u] = x, y
    if x >= 0:
        ms[x] = u
    else:
        roots[machine[u]] = u
    if y >= 0:
        mp[y] = u
    return a, b


def _undo(u, a, b, mp, ms, roots, machine):
    x, y = mp[u], ms[u]
    if x >= 0:
        ms[x] = y
    else:
        roots[machine[u]] = y
    if y >= 0:
        mp[y] = x
    mp[u], ms[u] = a, b
    if a >= 0:
        ms[a] = u
    else:
        roots[machine[u]] = u
    if b >= 0:
        mp[b] = u


def _blocks(p, mp, ms, roots, h, tail, cmax):
    eps = 1e-9 * max(1.0, cmax)
    critical = [abs(h[u] + tail[u] - cmax) <= eps for u in range(len(p))]
    blocks = []
    for root in roots:
        u = root
        block = []
        while u >= 0:
            v = ms[u]
            if (v >= 0 and critical[u] and critical[v]
                    and abs(h[u] + p[u] - h[v]) <= eps):
                if not block:
                    block.append(u)
                block.append(v)
            elif block:
                blocks.append(block)
                block = []
            u = v
    return blocks


def _join(a, b):
    # Summary (P,R,Q,C): the completion time is max(t+P,R) and the path estimate is max(t+Q,C).
    return (a[0] + b[0], max(a[1] + b[0], b[1]),
            max(a[2], a[0] + b[2]), max(a[3], b[3], a[1] + b[2]))


def _neighborhood(blocks, p, jp, js, mp, ms, h, tail, rng, deadline):
    candidates = []
    empty = (0.0, -math.inf, -math.inf, -math.inf)
    for block in blocks:
        if time.perf_counter() >= deadline:
            break
        length = len(block)
        atom = []
        for u in block:
            a, b = jp[u], js[u]
            r = h[a] + p[a] if a >= 0 else 0.0
            q = tail[b] if b >= 0 else 0.0
            atom.append((p[u], r + p[u], p[u] + q, r + p[u] + q))
        prefix = [empty]
        shifted = [empty, empty]
        for i, a in enumerate(atom):
            prefix.append(_join(prefix[-1], a))
            if i:
                shifted.append(_join(shifted[-1], a))
        suffix = [empty] * (length + 1)
        shortened = [empty] * length
        for i in range(length - 1, -1, -1):
            suffix[i] = _join(atom[i], suffix[i + 1])
            if i < length - 1:
                shortened[i] = _join(atom[i], shortened[i + 1])
        seen = set()
        for k in range(1, length):
            # Four move types: block head to interior, interior to block head, block tail to interior, interior to block tail.
            variants = ((0, k, _join(shifted[k + 1], atom[0])),
                        (k, 0, _join(atom[k], prefix[k])),
                        (length - 1, k - 1,
                         _join(atom[-1], shortened[k - 1])),
                        (k - 1, length - 1,
                         _join(suffix[k], atom[k - 1])))
            for i, j, summary in variants:
                lo, hi = min(i, j), max(i, j)
                key = (lo, hi) if hi - lo == 1 else (i, j)
                if key in seen:
                    continue
                seen.add(key)
                a, b = mp[block[lo]], ms[block[hi]]
                r = h[a] + p[a] if a >= 0 else 0.0
                q = tail[b] if b >= 0 else 0.0
                value = max(r + summary[2], summary[3],
                            max(r + summary[0], summary[1]) + q)
                candidates.append((value, rng.random(), block, i, j))
    heapq.heapify(candidates)
    return candidates


def _construct(nj, nm, machine, p, jp, js, rule, reverse, rng, noise, deadline):
    # With the same operation indices, construct along reversed job chains and then reverse every machine sequence.
    nxt = js if reverse else jp
    succ = jp if reverse else js
    active = [u for u in range(len(p)) if nxt[u] < 0]
    ready_job = [0.0] * len(p)
    ready_machine = [0.0] * nm
    work = [0.0] * len(p)
    traversal = range(len(p)) if reverse else range(len(p) - 1, -1, -1)
    for u in traversal:
        v = succ[u]
        work[u] = p[u] + (work[v] if v >= 0 else 0.0)
    seqs = [[] for _ in range(nm)]
    while active:
        if time.perf_counter() >= deadline:
            return None
        earliest = []
        ect = math.inf
        pivot = active[0]
        for u in active:
            start = max(ready_job[u], ready_machine[machine[u]])
            earliest.append(start)
            finish = start + p[u]
            if finish < ect:
                ect, pivot = finish, u
        m = machine[pivot]
        chosen = -1
        best_key = math.inf
        for idx, u in enumerate(active):
            if machine[u] != m or (earliest[idx] >= ect and u != pivot):
                continue
            if rule == 0:
                key = -work[u]
            elif rule == 1:
                key = p[u]
            elif rule == 2:
                key = -p[u]
            else:
                key = earliest[idx] - 0.25 * work[u]
            if noise:
                key += noise * (abs(key) + 1.0) * (rng.random() - 0.5)
            if key < best_key:
                best_key, chosen = key, idx
        u = active[chosen]
        finish = earliest[chosen] + p[u]
        ready_machine[m] = finish
        seqs[m].append(u)
        v = succ[u]
        if v >= 0:
            ready_job[v] = finish
            active[chosen] = v
        else:
            active.pop(chosen)
    if reverse:
        for seq in seqs:
            seq.reverse()
    return _links(seqs, len(p))


def _snapshot(ev, mp, ms, roots):
    return (ev[2], ev[0], mp[:], ms[:], roots[:])


def _add_elite(pool, state, capacity, separation):
    signature = state[3]
    for i, old in enumerate(pool):
        distance = sum(a != b for a, b in zip(signature, old[3]))
        if distance <= separation:
            if state[0] < old[0]:
                pool[i] = state
                pool.sort(key=lambda x: x[0])
            return
    pool.append(state)
    pool.sort(key=lambda x: x[0])
    if len(pool) > capacity:
        pool.pop()


def _search(nj, nm, machine, p, jp, js, incumbent, budget, start, deadline):
    n = len(p)
    rng = random.Random(0xC0FFEE)
    intensity = math.log1p(budget)
    capacity = max(2, int(2 + intensity * math.sqrt(nm + 1) / 2))
    separation = max(1, int(math.sqrt(n) / 3))
    pool = [tuple(incumbent)]
    init_end = min(deadline, start + budget *
                   (0.025 + 0.025 / (1.0 + math.sqrt(n) / 10.0)))
    launches = 0
    while time.perf_counter() < init_end:
        mode = launches % 4 if launches < 8 else rng.randrange(4)
        reverse = (launches // 4) % 2 == 1
        noise = 0.0 if launches < 8 else 0.15 + 0.75 * rng.random()
        links = _construct(nj, nm, machine, p, jp, js, mode, reverse,
                           rng, noise, init_end)
        launches += 1
        if links is None:
            break
        mp, ms, roots = links
        ev = _evaluate(p, jp, js, mp, ms)
        if ev is not None:
            state = _snapshot(ev, mp, ms, roots)
            _add_elite(pool, state, capacity, separation)
            if ev[2] < incumbent[0]:
                incumbent[:] = state
    current = tuple(incumbent)
    mp, ms, roots = current[2][:], current[3][:], current[4][:]
    ev = _evaluate(p, jp, js, mp, ms)
    tabu = {}
    seen_states = {}
    iteration = 0
    restart = 0
    tenure_base = max(4, int(4 + math.sqrt(n) / 2 +
                             intensity * math.sqrt(nj) / 3))
    tenure = tenure_base
    span = budget * (0.8 + 0.4 * math.sqrt(n) / (math.sqrt(n) + 20)) / (3 + intensity)
    last_gain = epoch_start = time.perf_counter()
    epoch_best = _snapshot(ev, mp, ms, roots)
    kick_base = max(2, int((1.0 + intensity) * math.sqrt(n) / 12))

    while time.perf_counter() < deadline:
        now = time.perf_counter()
        h, tail, cmax = ev
        blocks = _blocks(p, mp, ms, roots, h, tail, cmax)
        if not blocks or now - last_gain > span or now - epoch_start > 3 * span:
            _add_elite(pool, epoch_best, capacity, separation)
            restart += 1
            if rng.random() < 1.0 / (2.0 + math.sqrt(len(pool))):
                construction_end = min(deadline, now + span * 0.15)
                links = _construct(nj, nm, machine, p, jp, js, rng.randrange(4),
                                   rng.random() < 0.5, rng,
                                   0.2 + rng.random(), construction_end)
                if links is not None:
                    mp, ms, roots = links
                    trial = _evaluate(p, jp, js, mp, ms)
                    if trial is not None:
                        ev = trial
                    else:
                        links = None
                if links is None:
                    state = tuple(incumbent)
                    mp, ms, roots = state[2][:], state[3][:], state[4][:]
                    ev = _evaluate(p, jp, js, mp, ms)
            else:
                rank = int((rng.random() ** 2) * len(pool))
                state = pool[rank] if rng.random() < 0.65 else tuple(incumbent)
                mp, ms, roots = state[2][:], state[3][:], state[4][:]
                ev = _evaluate(p, jp, js, mp, ms)
                strength = min(n, kick_base + int(math.sqrt(restart)))
                kick_end = min(deadline, now + span * 0.08)
                for _ in range(strength):
                    if time.perf_counter() >= kick_end:
                        break
                    bks = _blocks(p, mp, ms, roots, ev[0], ev[1], ev[2])
                    if bks and rng.random() < 0.85:
                        block = rng.choice(bks)
                        i = 0 if rng.random() < 0.5 else len(block) - 1
                        j = rng.randrange(len(block) - 1)
                        if j >= i:
                            j += 1
                        u, v, after = block[i], block[j], i < j
                    else:
                        u = rng.randrange(n)
                        v = ms[u]
                        if v < 0:
                            continue
                        after = True
                    a, b = _move(u, v, after, mp, ms, roots, machine)
                    trial = _evaluate(p, jp, js, mp, ms)
                    if trial is None:
                        _undo(u, a, b, mp, ms, roots, machine)
                    else:
                        ev = trial
                        if ev[2] < incumbent[0]:
                            incumbent[:] = _snapshot(ev, mp, ms, roots)
            if ev[2] < incumbent[0]:
                incumbent[:] = _snapshot(ev, mp, ms, roots)
            tabu.clear()
            seen_states.clear()
            tenure = tenure_base
            last_gain = epoch_start = time.perf_counter()
            epoch_best = _snapshot(ev, mp, ms, roots)
            continue

        candidates = _neighborhood(blocks, p, jp, js, mp, ms, h, tail, rng, deadline)
        iteration += 1
        accepted = False
        deferred = []
        while candidates and time.perf_counter() < deadline:
            candidate = heapq.heappop(candidates)
            estimate, _, block, i, j = candidate
            u, v = block[i], block[j]
            if i < j:
                crossed = block[i + 1:j + 1]
                forbidden = any(tabu.get((w, u), 0) > iteration for w in crossed)
            else:
                crossed = block[j:i]
                forbidden = any(tabu.get((u, w), 0) > iteration for w in crossed)
            if forbidden and estimate >= incumbent[0]:
                deferred.append(candidate)
                continue
            a, b = _move(u, v, i < j, mp, ms, roots, machine)
            trial = _evaluate(p, jp, js, mp, ms)
            if trial is None or (forbidden and trial[2] >= incumbent[0]):
                _undo(u, a, b, mp, ms, roots, machine)
                continue
            ev = trial
            expiry = iteration + tenure + rng.randrange(1 + tenure // 2)
            for w in crossed:
                tabu[(u, w) if i < j else (w, u)] = expiry
            accepted = True
            break
        if not accepted and deferred and time.perf_counter() < deadline:
            # When all candidates are tabu, allow the best feasible candidate to avoid getting stuck in a tabu dead end.
            for _, _, block, i, j in deferred:
                if time.perf_counter() >= deadline:
                    break
                u, v = block[i], block[j]
                a, b = _move(u, v, i < j, mp, ms, roots, machine)
                trial = _evaluate(p, jp, js, mp, ms)
                if trial is None:
                    _undo(u, a, b, mp, ms, roots, machine)
                    continue
                ev = trial
                crossed = block[i + 1:j + 1] if i < j else block[j:i]
                expiry = iteration + tenure + rng.randrange(1 + tenure // 2)
                for w in crossed:
                    tabu[(u, w) if i < j else (w, u)] = expiry
                accepted = True
                break
        if accepted:
            if ev[2] < incumbent[0]:
                incumbent[:] = _snapshot(ev, mp, ms, roots)
            if ev[2] < epoch_best[0]:
                epoch_best = _snapshot(ev, mp, ms, roots)
                last_gain = time.perf_counter()
                tenure = tenure_base
            signature = hash(tuple(ms))
            previous = seen_states.get(signature)
            if previous is not None and iteration - previous < 2 * tenure:
                tenure = min(3 * tenure_base, tenure + max(1, tenure_base // 4))
            seen_states[signature] = iteration
        else:
            last_gain = -math.inf
        if len(tabu) > 4 * n + tenure:
            tabu = {pair: expiry for pair, expiry in tabu.items() if expiry > iteration}
        if len(seen_states) > n + 4 * tenure:
            seen_states = {sig: k for sig, k in seen_states.items()
                           if iteration - k <= 4 * tenure}


def _fill_output(result, state, nj, nm, p):
    result['start_times'] = [[float(state[1][j * nm + k]) for k in range(nm)]
                             for j in range(nj)]
    result['makespan'] = float(state[0])
    matrices = result['machine_precedence']
    for m in range(nm):
        seq = []
        u = state[4][m]
        while u >= 0:
            seq.append(u // nm)
            u = state[3][u]
        matrix = matrices['machine_%d' % m]
        position = [0] * nj
        for i, j in enumerate(seq):
            position[j] = i
        for j in range(nj):
            row = matrix[j]
            r = position[j]
            for k in range(nj):
                row[k] = int(r < position[k])
    return result


def solve(instance: dict, time_limit_s: float) -> dict:
    start = time.perf_counter()
    try:
        budget = float(time_limit_s)
        if not math.isfinite(budget) or budget < 0:
            budget = 0.0
    except Exception:
        budget = 0.0
    # Build a complete serial fallback before optimising; any later search error keeps the best verified solution.
    raw = instance.get('jobs', [])
    nj = max(0, int(instance.get('n_jobs', len(raw))))
    nm = max(0, int(instance.get('n_machines', len(raw[0]) if raw else 0)))
    result = {'start_times': [[0.0] * nm for _ in range(nj)],
              'machine_precedence': {'machine_%d' % m:
                                     [[int(j < k) for k in range(nj)] for j in range(nj)]
                                     for m in range(nm)},
              'makespan': 0.0}
    p, machine = [], []
    clock = 0.0
    try:
        for j in range(nj):
            for k in range(nm):
                m, duration = int(raw[j][k][0]), float(raw[j][k][1])
                result['start_times'][j][k] = clock
                clock += duration
                machine.append(m)
                p.append(duration)
        result['makespan'] = float(clock)
        if not p or time.perf_counter() - start >= budget:
            return result
        n = len(p)
        jp = [u - 1 if u % nm else -1 for u in range(n)]
        js = [u + 1 if u % nm < nm - 1 else -1 for u in range(n)]
        seqs = [[] for _ in range(nm)]
        for u, m in enumerate(machine):
            seqs[m].append(u)
        mp, ms, roots = _links(seqs, n)
        ev = _evaluate(p, jp, js, mp, ms)
        incumbent = list(_snapshot(ev, mp, ms, roots))
        spent = time.perf_counter() - start
        reserve = max(0.025 * budget, 2.0 * spent)
        deadline = start + max(0.0, budget - reserve)
        try:
            if time.perf_counter() < deadline:
                _search(nj, nm, machine, p, jp, js, incumbent,
                        budget, start, deadline)
        except Exception:
            pass
        return _fill_output(result, incumbent, nj, nm, p)
    except Exception:
        return result
