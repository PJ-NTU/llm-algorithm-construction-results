"""Shifting-bottleneck reconstruction with all-critical-block reinsertion search
On the same disjunctive graph, fuses shifting-bottleneck reordering with indirect precedences, exact operation reinsertion and adaptive tabu search, continually improving the makespan with elite saving and multi-machine release-and-rebuild.
"""
import heapq
import math
import random
import time


def _links(sequences, count):
    pred = [-1] * count
    succ = [-1] * count
    for row in sequences:
        for a, b in zip(row, row[1:]):
            succ[a] = b
            pred[b] = a
    return pred, succ


def _forward(p, jp, js, mp, ms, deadline):
    count = len(p)
    degree = [int(jp[u] >= 0) + int(mp[u] >= 0) for u in range(count)]
    stack = [u for u in range(count) if degree[u] == 0]
    head = [0.0] * count
    order = []
    makespan = 0.0
    while stack:
        u = stack.pop()
        order.append(u)
        finish = head[u] + p[u]
        if finish > makespan:
            makespan = finish
        v = js[u]
        if v >= 0:
            if finish > head[v]:
                head[v] = finish
            degree[v] -= 1
            if degree[v] == 0:
                stack.append(v)
        v = ms[u]
        if v >= 0:
            if finish > head[v]:
                head[v] = finish
            degree[v] -= 1
            if degree[v] == 0:
                stack.append(v)
        if len(order) % 512 == 0 and time.perf_counter() >= deadline:
            return None
    if len(order) != count:
        return None
    return head, order, makespan


def _tails(p, js, ms, order):
    tail = [0.0] * len(p)
    for u in reversed(order):
        v = js[u]
        a = tail[v] + p[v] if v >= 0 else 0.0
        v = ms[u]
        b = tail[v] + p[v] if v >= 0 else 0.0
        tail[u] = max(a, b)
    return tail


def _insert(u, v, after, mp, ms):
    a, b = mp[u], ms[u]
    if a >= 0:
        ms[a] = b
    if b >= 0:
        mp[b] = a
    if after:
        c, d = v, ms[v]
    else:
        c, d = mp[v], v
    if c >= 0:
        ms[c] = u
    if d >= 0:
        mp[d] = u
    mp[u], ms[u] = c, d
    return a, b


def _undo(u, a, b, mp, ms):
    c, d = mp[u], ms[u]
    if c >= 0:
        ms[c] = d
    if d >= 0:
        mp[d] = c
    if a >= 0:
        ms[a] = u
    if b >= 0:
        mp[b] = u
    mp[u], ms[u] = a, b


def _forbidden(u, v, after, ms, tabu, iteration):
    """Every pair of machine precedences changed by a long insertion must be checked."""
    if after:
        x = ms[u]
        while x >= 0:
            if tabu.get((x, u), 0) > iteration:
                return True
            if x == v:
                break
            x = ms[x]
    else:
        x = v
        while x >= 0 and x != u:
            if tabu.get((u, x), 0) > iteration:
                return True
            x = ms[x]
    return False


def _reinsert(u, p, jp, js, mp, ms, tabu, iteration, best_value, rng, deadline):
    """Remove the machine arcs of one operation and exactly evaluate all insertion positions on that machine in one pass."""
    a, b = mp[u], ms[u]
    if a < 0 and b < 0:
        return None
    root = u
    while mp[root] >= 0:
        root = mp[root]
    row = []
    original = 0
    while root >= 0:
        if root == u:
            original = len(row)
        else:
            row.append(root)
        root = ms[root]

    if a >= 0:
        ms[a] = b
    if b >= 0:
        mp[b] = a
    mp[u] = ms[u] = -1
    try:
        result = _forward(p, jp, js, mp, ms, deadline)
        if result is None or time.perf_counter() >= deadline:
            return None
        h, order, base_value = result
        q = _tails(p, js, ms, order)

        # A successor cannot become the new machine predecessor, and an ancestor cannot become the new machine successor.
        marked = [0] * len(p)
        for previous, following, bit in ((js, ms, 1), (jp, mp, 2)):
            marked[u] |= bit
            stack = [u]
            steps = 0
            while stack:
                x = stack.pop()
                for y in (previous[x], following[x]):
                    if y >= 0 and not marked[y] & bit:
                        marked[y] |= bit
                        stack.append(y)
                steps += 1
                if steps % 256 == 0 and time.perf_counter() >= deadline:
                    return None

        length = len(row)
        expiry = [0] * (length + 1)
        for pos in range(original + 1, length + 1):
            expiry[pos] = max(expiry[pos - 1], tabu.get((row[pos - 1], u), 0))
        for pos in range(original - 1, -1, -1):
            expiry[pos] = max(expiry[pos + 1], tabu.get((u, row[pos]), 0))
        chosen = None
        for pos in range(length + 1):
            if pos == original:
                continue
            left = row[pos - 1] if pos else -1
            right = row[pos] if pos < length else -1
            if (left >= 0 and marked[left] & 1) or (right >= 0 and marked[right] & 2):
                continue
            release = max(h[u], h[left] + p[left]) if left >= 0 else h[u]
            delivery = max(q[u], p[right] + q[right]) if right >= 0 else q[u]
            span = release + p[u] + delivery
            value = max(base_value, span)
            if expiry[pos] > iteration and value >= best_value:
                continue
            # Among equal objective values, prefer shortening paths through the moved operation to help cross plateaus.
            v, after = (left, True) if pos > original else (right, False)
            candidate = (value, span, rng.random(), u, v, after)
            if chosen is None or candidate < chosen:
                chosen = candidate
        return chosen
    finally:
        if a >= 0:
            ms[a] = u
        if b >= 0:
            mp[b] = u
        mp[u], ms[u] = a, b


def _moves(p, jp, js, mp, ms, head, tail, makespan):
    """Insertions at both ends of every critical machine block; estimates are generated linearly with prefix summaries."""
    count = len(p)
    eps = 1e-9 * max(1.0, makespan)
    tight = [False] * count
    for u in range(count):
        v = ms[u]
        if (v >= 0 and abs(head[u] + p[u] - head[v]) <= eps
                and abs(head[u] + p[u] + tail[u] - makespan) <= eps
                and abs(tail[u] - p[v] - tail[v]) <= eps):
            tight[u] = True
    proposals = {}

    def emit(value, u, v, after):
        # Two adjacent ways of writing an insertion are merged into the same swap.
        if not after and ms[v] == u:
            u, v, after = v, u, True
        key = (u, v, after)
        old = proposals.get(key)
        if old is None or value < old:
            proposals[key] = value

    for root in range(count):
        if not tight[root] or (mp[root] >= 0 and tight[mp[root]]):
            continue
        block = [root]
        while tight[block[-1]]:
            block.append(ms[block[-1]])
        for reverse in (False, True):
            sequence = block[::-1] if reverse else block
            h, q = (tail, head) if reverse else (head, tail)
            prev, nxt = (js, jp) if reverse else (jp, js)
            mprev, mnxt = (ms, mp) if reverse else (mp, ms)
            first = sequence[0]
            a = mprev[first]
            boundary = h[a] + p[a] if a >= 0 else 0.0
            a = prev[first]
            r0 = h[a] + p[a] if a >= 0 else 0.0
            a = nxt[first]
            q0 = q[a] + p[a] if a >= 0 else 0.0
            finish, peak = boundary, 0.0
            total = p[first]
            release = r0 + p[first]
            delivery = p[first] + q0
            internal = release + q0
            for u in sequence[1:]:
                a = prev[u]
                r = h[a] + p[a] if a >= 0 else 0.0
                a = nxt[u]
                d = q[a] + p[a] if a >= 0 else 0.0
                a = mnxt[u]
                outside = q[a] + p[a] if a >= 0 else 0.0
                # Move the first operation of the block backwards.
                finish = max(finish, r) + p[u]
                peak = max(peak, finish + d)
                last = max(finish, r0) + p[first]
                estimate = max(peak, last + max(q0, outside))
                emit(estimate, first, u, not reverse)
                # Move the current operation to the block head; prefix quadruples summarise the unchanged subchain.
                initial = max(boundary, r) + p[u]
                last = max(initial + total, release)
                estimate = max(initial + d, initial + delivery,
                               internal, last + outside)
                emit(estimate, u, first, reverse)
                total += p[u]
                release = max(release, r) + p[u]
                delivery = max(delivery, total + d)
                internal = max(internal, release + d)
    return proposals


def _construct(nj, nm, p, machine, jp, js, rule, reverse, noise, rng, deadline):
    """Forward or backward conflict-set active scheduling; pivots are kept even for zero processing times."""
    step = jp if reverse else js
    nxt = [j * nm + (nm - 1 if reverse else 0) for j in range(nj)]
    ready = [0.0] * nj
    mr = [0.0] * nm
    remaining = [sum(p[j * nm:(j + 1) * nm]) for j in range(nj)]
    active = list(range(nj))
    sequences = [[] for _ in range(nm)]
    est = [0.0] * nj
    while active:
        if time.perf_counter() >= deadline:
            return None
        earliest, pivot, target = math.inf, -1, -1
        for j in active:
            u = nxt[j]
            m = machine[u]
            start = max(ready[j], mr[m])
            est[j] = start
            finish = start + p[u]
            if finish < earliest:
                earliest, pivot, target = finish, j, m
        pick, priority = pivot, math.inf
        for j in active:
            u = nxt[j]
            if machine[u] != target or (j != pivot and est[j] >= earliest):
                continue
            if rule == 0:
                value = -remaining[j]
            elif rule == 1:
                value = p[u]
            elif rule == 2:
                value = -p[u]
            else:
                value = est[j]
            if noise:
                value += noise * (abs(value) + 1.0) * (rng.random() - 0.5)
            if value < priority:
                pick, priority = j, value
        u = nxt[pick]
        ready[pick] = est[pick] + p[u]
        mr[target] = ready[pick]
        remaining[pick] -= p[u]
        sequences[target].append(u)
        nxt[pick] = step[u]
        if nxt[pick] < 0:
            active.remove(pick)
    if reverse:
        for row in sequences:
            row.reverse()
    return _links(sequences, len(p))


def _justify(nj, nm, p, machine, jp, js, priority, reverse, deadline):
    """Sort by existing head or tail lengths and insert into the earliest available gap on each machine."""
    step = jp if reverse else js
    heap = [(priority[j * nm + (nm - 1 if reverse else 0)],
             j * nm + (nm - 1 if reverse else 0)) for j in range(nj)]
    heapq.heapify(heap)
    ready = [0.0] * nj
    slots = [[] for _ in range(nm)]
    while heap:
        if time.perf_counter() >= deadline:
            return None
        _, u = heapq.heappop(heap)
        j = u // nm
        start = ready[j]
        row = slots[machine[u]]
        position = 0
        for a, b, _ in row:
            if start + p[u] <= a:
                break
            start = max(start, b)
            position += 1
        finish = start + p[u]
        row.insert(position, (start, finish, u))
        ready[j] = finish
        v = step[u]
        if v >= 0:
            heapq.heappush(heap, (priority[v], v))
    sequences = [[entry[2] for entry in row] for row in slots]
    if reverse:
        for row in sequences:
            row.reverse()
    return _links(sequences, len(p))


def _schrage(ops, head, tail, p, rng, noise, deadline):
    """Estimate how much of a bottleneck a machine is from single-machine release times and tail lengths."""
    waiting = sorted(ops, key=lambda u: head[u])
    ready, sequence = [], []
    i, clock, value = 0, 0.0, 0.0
    while i < len(waiting) or ready:
        if time.perf_counter() >= deadline:
            return None
        if not ready:
            clock = max(clock, head[waiting[i]])
        while i < len(waiting) and head[waiting[i]] <= clock:
            u = waiting[i]
            priority = tail[u] * (1.0 + noise * (rng.random() - 0.5))
            heapq.heappush(ready, (-priority, p[u], u))
            i += 1
        _, _, u = heapq.heappop(ready)
        sequence.append(u)
        clock += p[u]
        value = max(value, clock + tail[u])
    return value, sequence


def _precedence_schrage(ops, p, js, ms, head, tail, order, rng, noise, deadline):
    """Project indirect precedences and propagate the effect of scheduled operations on later release times."""
    bit = {u: 1 << i for i, u in enumerate(ops)}
    ancestors = [0] * len(p)
    for i, u in enumerate(order):
        mask = ancestors[u] | bit.get(u, 0)
        if mask:
            v = js[u]
            if v >= 0:
                ancestors[v] |= mask
            v = ms[u]
            if v >= 0:
                ancestors[v] |= mask
        if i % 256 == 0 and time.perf_counter() >= deadline:
            return None
    pending = list(ops)
    release = head[:]
    priorities = {u: tail[u] * (1.0 + noise * (rng.random() - 0.5)) for u in ops}
    done, clock = 0, 0.0
    sequence = []
    while pending:
        if time.perf_counter() >= deadline:
            return None
        eligible = [u for u in pending if not (ancestors[u] & ~done)]
        if not eligible:
            return None
        clock = max(clock, min(release[u] for u in eligible))
        u = max((u for u in eligible if release[u] <= clock),
                key=lambda u: (priorities[u], -p[u], -u))
        if clock > release[u]:
            release[u] = clock
            stack = [u]
            queued = {u}
            steps = 0
            while stack:
                x = stack.pop()
                queued.remove(x)
                finish = release[x] + p[x]
                for v in (js[x], ms[x]):
                    if v >= 0 and finish > release[v]:
                        release[v] = finish
                        if v not in queued:
                            stack.append(v)
                            queued.add(v)
                steps += 1
                if steps % 256 == 0 and time.perf_counter() >= deadline:
                    return None
        clock += p[u]
        sequence.append(u)
        pending.remove(u)
        done |= bit[u]
    return sequence


def _refine_row(ops, p, jp, js, state, rng, deadline):
    """Connect full-machine reordering with exact cross-block reinsertion, accepting only moves that truly shorten the makespan."""
    value, mp, ms, head, tail = state
    while time.perf_counter() < deadline:
        eps = 1e-9 * max(1.0, value)
        sources = [u for u in ops if head[u] + p[u] + tail[u] >= value - eps]
        rng.shuffle(sources)
        chosen = None
        for u in sources:
            if time.perf_counter() >= deadline:
                break
            candidate = _reinsert(u, p, jp, js, mp, ms, {}, 0,
                                  value, rng, deadline)
            if candidate is not None and candidate[0] < value - eps:
                if chosen is None or candidate < chosen:
                    chosen = candidate
        if chosen is None or time.perf_counter() >= deadline:
            break
        _, _, _, u, v, after = chosen
        a, b = _insert(u, v, after, mp, ms)
        result = _forward(p, jp, js, mp, ms, deadline)
        if result is None or result[2] >= value - eps:
            _undo(u, a, b, mp, ms)
            break
        head, order, value = result
        tail = _tails(p, js, ms, order)
    return value, mp, ms, head, tail


def _bottleneck(state, released, mops, p, jp, js, rng, noise, scale, deadline):
    """After releasing machines, rebuild them one by one; each round picks the machine with the largest current single-machine objective and improves it exactly."""
    mp, ms = state[1][:], state[2][:]
    todo = list(released)
    for m in todo:
        for u in mops[m]:
            mp[u] = ms[u] = -1
    while todo and time.perf_counter() < deadline:
        result = _forward(p, jp, js, mp, ms, deadline)
        if result is None:
            return None
        head, order, value = result
        tail = _tails(p, js, ms, order)
        pick, highest = -1, -math.inf
        for m in todo:
            rough = _schrage(mops[m], head, tail, p, rng, noise, deadline)
            if rough is None:
                return None
            pressure = rough[0] * (1.0 + 0.15 * noise * rng.random())
            if pressure > highest:
                pick, highest = m, pressure
        row = _precedence_schrage(mops[pick], p, js, ms, head, tail,
                                 order, rng, noise, deadline)
        if row is None:
            return None
        for i, u in enumerate(row):
            mp[u] = row[i - 1] if i else -1
            ms[u] = row[i + 1] if i + 1 < len(row) else -1
        result = _forward(p, jp, js, mp, ms, deadline)
        if result is None:
            return None
        head, order, value = result
        tail = _tails(p, js, ms, order)
        todo.remove(pick)
        now = time.perf_counter()
        refine_end = min(deadline, now + (deadline - now) /
                         ((len(todo) + 2.0) * (2.0 + scale)))
        state = _refine_row(row, p, jp, js, (value, mp, ms, head, tail), rng, refine_end)
    return state if not todo else None


def _search(nj, nm, p, machine, jp, js, mp, ms, head, budget, deadline):
    rng = random.Random(982451653)
    count = len(p)
    # The initial machine order follows job indices, so operation indices form a valid topological order.
    tail = _tails(p, js, ms, range(count))
    mk = max((head[u] + p[u] for u in range(count)), default=0.0)
    best = (mk, mp[:], ms[:], head, tail)
    scale = math.log1p(budget)
    pool_size = 2 + int(scale + math.log1p(nj + nm) / 3.0)
    width = 1 + int(scale / (1.0 + math.sqrt(count) / 16.0))
    broad_width = width + max(1, int(math.sqrt(scale)))
    broad_stride = max(2, int(2.0 + math.sqrt(count) / (2.0 * (1.0 + scale))))
    strength = max(2, int(math.sqrt(nj + nm) * (1.0 + scale) / 3.0))
    base_tenure = 3 + int(0.35 * math.sqrt(count) + scale)
    patience = budget / (3.0 + scale + math.log1p(count) / 2.0)
    diversity = max(2, int(math.sqrt(count) / (1.0 + scale)))
    pool = [best]
    mops = [[] for _ in range(nm)]
    for u, m in enumerate(machine):
        mops[m].append(u)
    mean_duration = max(1e-12, sum(p) / count)
    release_width = max(1, int(math.sqrt(nm) * (1.0 + scale) / 2.0))
    full_period = max(2, 2 + int(math.sqrt(scale)))

    def machine_order(state):
        value, _, _, h, q = state
        pressure = [0.0] * nm
        for u in range(count):
            slack = max(0.0, value - h[u] - p[u] - q[u])
            pressure[machine[u]] += p[u] / (1.0 + slack / mean_duration)
        return sorted(range(nm), key=lambda m: pressure[m], reverse=True)

    def archive(state):
        closest, distance = -1, count + 1
        for i, old in enumerate(pool):
            d = sum(a != b for a, b in zip(old[2], state[2]))
            if d < distance:
                closest, distance = i, d
        if distance < diversity:
            if state[0] < pool[closest][0]:
                pool[closest] = state
        elif len(pool) < pool_size:
            pool.append(state)
        else:
            worst = max(range(len(pool)), key=lambda i: pool[i][0])
            if state[0] < pool[worst][0]:
                pool[worst] = state

    def evaluate(links):
        if links is None or time.perf_counter() >= deadline:
            return None
        a, b = links
        result = _forward(p, jp, js, a, b, deadline)
        if result is None:
            return None
        h, order, value = result
        return value, a, b, h, _tails(p, js, b, order)

    try:
        seed_end = min(deadline, time.perf_counter() + budget / (6.0 + scale))
        for rule, reverse in ((0, False), (0, True), (1, False), (2, False),
                              (1, True), (2, True), (3, False), (3, True)):
            if time.perf_counter() >= seed_end:
                break
            state = evaluate(_construct(nj, nm, p, machine, jp, js, rule,
                                        reverse, 0.0, rng, seed_end))
            if state is not None:
                archive(state)
                if state[0] < best[0]:
                    best = state
        # Active scheduling and shifting-bottleneck construction share machine chains, head/tail lengths and the exact reinsertion evaluation.
        state = _bottleneck(best, list(range(nm)), mops, p, jp, js,
                            rng, 0.0, scale, seed_end)
        if state is not None:
            archive(state)
            if state[0] < best[0]:
                best = state
        mk, a, b, head, tail = best
        mp, ms = a[:], b[:]
        tabu, seen = {}, {}
        tenure = base_tenure
        iteration, restarts = 0, 0
        local_best = mk
        last_progress = time.perf_counter()
        last_machine = last_progress
        machine_failures, machine_probe = 0, 0
        force_restart = False
        while time.perf_counter() < deadline:
            now = time.perf_counter()
            if force_restart or now - last_progress >= patience:
                archive((mk, mp[:], ms[:], head, tail))
                restarts += 1
                base = best if rng.random() < 0.55 else rng.choice(pool)
                important = machine_order(base)
                depth = min(nm, max(1, int(release_width * (0.5 + rng.random()))))
                released = important[:max(1, depth // 2)]
                rest = [m for m in important if m not in released]
                rng.shuffle(rest)
                released.extend(rest[:depth - len(released)])
                if restarts % full_period == 0:
                    released = list(range(nm))
                phase_end = min(deadline, now + patience / (1.0 + math.sqrt(scale)))
                state = _bottleneck(base, released, mops, p, jp, js, rng,
                                    0.12 + 0.5 * rng.random(), scale, phase_end)
                if state is None:
                    state = base
                changed = sum(a != b for a, b in zip(base[2], state[2]))
                kick_strength = strength if changed < diversity else max(1, strength // (1 + int(scale)))
                mk, a, b, head, tail = state
                mp, ms = a[:], b[:]
                if mk < best[0]:
                    best = (mk, mp[:], ms[:], head, tail)
                    archive(best)
                # Gap insertion in both directions can change the order on non-critical machines.
                for reverse in (True, False):
                    candidate = evaluate(_justify(nj, nm, p, machine, jp, js,
                                                  tail if reverse else head,
                                                  reverse, min(deadline, now + patience)))
                    if candidate is not None and candidate[0] <= mk:
                        mk, mp, ms, head, tail = candidate
                    if mk < best[0]:
                        best = (mk, mp[:], ms[:], head, tail)
                        archive(best)
                # Perturbation size comes from the time budget and instance size; moves that create cycles are always rejected exactly.
                kick = 0
                while kick < kick_strength and time.perf_counter() < min(deadline, now + patience):
                    options = _moves(p, jp, js, mp, ms, head, tail, mk)
                    if options and rng.random() < 0.8:
                        u, v, after = rng.choice(list(options))
                    else:
                        u = rng.randrange(count)
                        v, after = ms[u], True
                        if v < 0:
                            kick += 1
                            continue
                    a, b = _insert(u, v, after, mp, ms)
                    result = _forward(p, jp, js, mp, ms, deadline)
                    if result is None:
                        _undo(u, a, b, mp, ms)
                    else:
                        head, order, mk = result
                        tail = _tails(p, js, ms, order)
                        if mk < best[0]:
                            best = (mk, mp[:], ms[:], head, tail)
                            archive(best)
                    kick += 1
                tabu.clear()
                seen.clear()
                tenure = base_tenure
                local_best, last_progress = mk, time.perf_counter()
                last_machine = last_progress
                machine_failures = 0
                force_restart = False
                continue

            # Use the whole-machine neighbourhood directly inside tabu search instead of only choosing between two solvers.
            machine_gap = patience * (1.0 + machine_failures) / (2.0 + scale)
            if (now - last_machine >= machine_gap and
                    now - last_progress >= patience / (3.0 + scale)):
                current = (mk, mp, ms, head, tail)
                important = machine_order(current)
                target = important[machine_probe % nm]
                machine_probe += 1
                phase_end = min(deadline, now + patience / (3.0 + scale))
                candidate = _bottleneck(current, [target], mops, p, jp, js,
                                        rng, 0.0, scale, phase_end)
                last_machine = time.perf_counter()
                if candidate is not None and candidate[0] < mk:
                    mk, mp, ms, head, tail = candidate
                    machine_failures = 0
                    tabu.clear()
                    if mk < best[0]:
                        best = (mk, mp[:], ms[:], head, tail)
                        archive(best)
                    if mk < local_best:
                        local_best, last_progress = mk, time.perf_counter()
                    continue
                machine_failures += 1

            iteration += 1
            proposals = _moves(p, jp, js, mp, ms, head, tail, mk)
            ranked, blocked = [], []
            for (u, v, after), estimate in proposals.items():
                forbidden = _forbidden(u, v, after, ms, tabu, iteration)
                entry = (estimate, rng.random(), u, v, after, forbidden)
                if not forbidden or estimate < best[0]:
                    ranked.append(entry)
                else:
                    blocked.append(entry)
            ranked.sort()
            # When no non-tabu candidate is available, allow one forced escape instead of stopping the search.
            forced = not ranked
            if forced:
                ranked = sorted(blocked)
            choice, choice_key, checked = None, None, 0
            if iteration % broad_stride == 0 or not ranked:
                sources = []
                for entry in sorted(proposals.items(), key=lambda entry: entry[1]):
                    u = entry[0][0]
                    if u not in sources:
                        sources.append(u)
                        if len(sources) >= broad_width - 1:
                            break
                eps = 1e-9 * max(1.0, mk)
                critical = [u for u in range(count)
                            if abs(head[u] + p[u] + tail[u] - mk) <= eps
                            and u not in sources and (mp[u] >= 0 or ms[u] >= 0)]
                if critical:
                    rng.shuffle(critical)
                    sources.extend(critical[:max(1, broad_width - len(sources))])
                extended = None
                for u in sources:
                    if time.perf_counter() >= deadline:
                        break
                    candidate = _reinsert(u, p, jp, js, mp, ms, tabu, iteration,
                                          best[0], rng, deadline)
                    if candidate is not None and (extended is None or candidate < extended):
                        extended = candidate
                if extended is not None and time.perf_counter() < deadline:
                    _, _, tie, u, v, after = extended
                    a, b = _insert(u, v, after, mp, ms)
                    try:
                        result = _forward(p, jp, js, mp, ms, deadline)
                    finally:
                        _undo(u, a, b, mp, ms)
                    if result is not None:
                        h, order, value = result
                        choice = (u, v, after, h, order, value)
                        choice_key = (value, sum(h), tie)
            # The cross-block phase already scans several complete insertion neighbourhoods; other phases use cheap critical-block screening.
            for _, tie, u, v, after, forbidden in (() if choice is not None else ranked):
                if time.perf_counter() >= deadline:
                    break
                a, b = _insert(u, v, after, mp, ms)
                result = _forward(p, jp, js, mp, ms, deadline)
                _undo(u, a, b, mp, ms)
                if result is None:
                    continue
                h, order, value = result
                if forbidden and not forced and value >= best[0]:
                    continue
                key = (value, sum(h), tie)
                if choice_key is None or key < choice_key:
                    choice_key = key
                    choice = (u, v, after, h, order, value)
                checked += 1
                if checked >= width:
                    break
            if choice is None:
                force_restart = True
                continue
            u, v, after, head, order, mk = choice
            expiration = iteration + tenure + rng.randrange(base_tenure + 1)
            # Record every operation pair crossed by this move in the tabu list, discouraging immediate undoing of long insertions.
            if after:
                x = ms[u]
                while x >= 0:
                    tabu[(u, x)] = expiration
                    if x == v:
                        break
                    x = ms[x]
            else:
                x = v
                while x >= 0 and x != u:
                    tabu[(x, u)] = expiration
                    x = ms[x]
            _insert(u, v, after, mp, ms)
            tail = _tails(p, js, ms, order)
            if mk < best[0]:
                best = (mk, mp[:], ms[:], head, tail)
                archive(best)
            if mk < local_best:
                local_best, last_progress = mk, time.perf_counter()
            signature = hash(tuple(ms))
            old = seen.get(signature)
            seen[signature] = iteration
            if old is not None and iteration - old <= 2 * tenure:
                tenure = min(int(base_tenure * (2.0 + scale)), tenure + base_tenure)
            elif iteration % max(1, 2 * base_tenure) == 0:
                tenure = max(base_tenure, tenure - 1)
            if len(tabu) > count * (2 + strength):
                tabu = {key: expiry for key, expiry in tabu.items() if expiry > iteration}
            if len(seen) > count * (2 + pool_size):
                cutoff = iteration - count
                seen = {key: when for key, when in seen.items() if when >= cutoff}
    except Exception:
        # Stored elites use separate machine linked lists, unaffected by unfinished neighbourhood operations.
        pass
    return best


def solve(instance: dict, time_limit_s: float) -> dict:
    started = time.perf_counter()
    answer = {"start_times": [], "machine_precedence": {}, "makespan": 0.0}
    try:
        try:
            budget = float(time_limit_s)
        except (TypeError, ValueError, OverflowError):
            budget = 0.0
        if not math.isfinite(budget) or budget < 0.0:
            budget = 0.0
        nj, nm = int(instance["n_jobs"]), int(instance["n_machines"])
        jobs = instance["jobs"]
        count = nj * nm
        p, machine = [], []
        for j in range(nj):
            for k in range(nm):
                machine.append(int(jobs[j][k][0]))
                p.append(float(jobs[j][k][1]))
        jp = [u - 1 if u % nm else -1 for u in range(count)]
        js = [u + 1 if u % nm != nm - 1 else -1 for u in range(count)]
        head = [0.0] * count
        sequences = [[] for _ in range(nm)]
        machine_ready = [0.0] * nm
        makespan = 0.0
        # Save a complete feasible solution first; use it directly when there is no search budget.
        for j in range(nj):
            ready = 0.0
            for k in range(nm):
                u = j * nm + k
                m = machine[u]
                head[u] = max(ready, machine_ready[m])
                ready = head[u] + p[u]
                machine_ready[m] = ready
                sequences[m].append(u)
            makespan = max(makespan, ready)
        answer = {
            "start_times": [head[j * nm:(j + 1) * nm] for j in range(nj)],
            "machine_precedence": {
                "machine_%d" % m: [[int(j < k) for k in range(nj)] for j in range(nj)]
                for m in range(nm)
            },
            "makespan": float(makespan),
        }
        if not count:
            return answer
        setup_elapsed = time.perf_counter() - started
        # Reserve time for building the quadratic output matrix; do not start improvement if the budget is insufficient.
        reserve = max(0.03 * budget, 1.5 * setup_elapsed)
        deadline = started + max(0.0, budget - reserve)
        if time.perf_counter() >= deadline:
            return answer
        mp, ms = _links(sequences, count)
        best = _search(nj, nm, p, machine, jp, js, mp, ms, head, budget, deadline)
        value, mp, ms, head, _ = best
        if value >= makespan:
            return answer
        matrices = {}
        for m in range(nm):
            row = sequences[m]
            root = next((u for u in row if mp[u] < 0), -1)
            ordered_jobs = []
            while root >= 0:
                ordered_jobs.append(root // nm)
                root = ms[root]
            rank = [0] * nj
            for position, j in enumerate(ordered_jobs):
                rank[j] = position
            matrices["machine_%d" % m] = [
                [int(rank[j] < rank[k]) for k in range(nj)] for j in range(nj)
            ]
        return {"start_times": [head[j * nm:(j + 1) * nm] for j in range(nj)],
                "machine_precedence": matrices, "makespan": float(value)}
    except Exception:
        return answer
