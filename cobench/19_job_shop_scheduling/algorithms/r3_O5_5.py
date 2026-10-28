"""Multi-critical-window branch-repair tabu search
Mainly critical-block insertion tabu search; on stagnation, releases several machine windows along the longest path and jointly reorders them with head/tail lower bounds and time-limited branch and bound.
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
    degree = [(jp[u] >= 0) + (mp[u] >= 0) for u in range(count)]
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
        tail[u] = a if a > b else b
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


def _partial(p, edges, degree, deadline):
    """Head lengths, tail lengths and one longest path of a partial disjunctive graph; give up immediately on a cycle or at the deadline."""
    n = len(p)
    degree = degree[:]
    stack = [u for u in range(n) if degree[u] == 0]
    h = [0.0] * n
    parent = [-1] * n
    order = []
    value = 0.0
    while stack:
        u = stack.pop()
        order.append(u)
        finish = h[u] + p[u]
        if finish > value:
            value = finish
        for v in edges[u]:
            if finish >= h[v]:
                h[v], parent[v] = finish, u
            degree[v] -= 1
            if degree[v] == 0:
                stack.append(v)
        if len(order) % 256 == 0 and time.perf_counter() >= deadline:
            return None
    if len(order) != n or time.perf_counter() >= deadline:
        return None
    q = [0.0] * n
    for u in reversed(order):
        longest = 0.0
        for v in edges[u]:
            candidate = p[v] + q[v]
            if candidate > longest:
                longest = candidate
        q[u] = longest
    return value, h, q, order, parent


def _relaxed_graph(rows, free, js):
    """Release only the machine order inside contiguous windows; all precedences between windows are kept."""
    edges = [[v] if v >= 0 else [] for v in js]
    blocks = []
    for row in rows:
        if not row:
            continue
        groups = [[row[0]]]
        for a, b in zip(row, row[1:]):
            if a in free:
                groups[-1].append(b)
            else:
                groups.append([b])
        blocks.extend(group for group in groups if len(group) > 1)
        for left, right in zip(groups, groups[1:]):
            for u in left:
                edges[u].extend(right)
    degree = [0] * len(js)
    for row in edges:
        for v in row:
            degree[v] += 1
    return edges, degree, blocks


def _repair(nj, nm, p, machine, jp, js, state, scale, rng, deadline):
    """Longest-path-guided multi-window branch and bound; only fully verified strict improvements are accepted."""
    best = state
    value, mp, ms, head, tail = state
    n = len(p)
    if n == 0 or time.perf_counter() >= deadline:
        return best
    rows = [[] for _ in range(nm)]
    position = [0] * n
    for u in range(n):
        if mp[u] < 0:
            row = rows[machine[u]]
            while u >= 0:
                position[u] = len(row)
                row.append(u)
                u = ms[u]
    free = set()
    # The release size depends on the instance and the total budget; it is not an iteration cap on the global search.
    edge_limit = max(2, int((1.0 + scale) * math.sqrt(nj + nm) *
                            (1.0 + 0.5 * rng.random())))
    pair_limit = max(4, int((2.0 + scale) * math.sqrt(n) *
                            (1.0 + rng.random())))
    margin = sum(p) / n * rng.random() * scale / (2.0 + scale)
    target = value - margin
    focus = rng.random() * value
    eps = 1e-10 * max(1.0, value)
    relaxed = None
    while time.perf_counter() < deadline:
        base, degree, blocks = _relaxed_graph(rows, free, js)
        relaxed = _partial(p, base, degree, deadline)
        if relaxed is None:
            return best
        lower, h, q, order, parent = relaxed
        if lower < target - eps or len(free) >= edge_limit:
            break
        u = max(order, key=lambda x: h[x] + p[x])
        options = []
        current_pairs = sum(len(b) * (len(b) - 1) // 2 for b in blocks)
        while parent[u] >= 0:
            v = parent[u]
            if machine[v] == machine[u] and js[v] != u:
                row = rows[machine[u]]
                left, right = position[v], position[u]
                if left < right:
                    while left > 0 and row[left - 1] in free:
                        left -= 1
                    while right + 1 < len(row) and row[right] in free:
                        right += 1
                    old_pairs = 0
                    run = 1
                    for k in range(left, right):
                        if row[k] in free:
                            run += 1
                        else:
                            old_pairs += run * (run - 1) // 2
                            run = 1
                    old_pairs += run * (run - 1) // 2
                    added_pairs = (right - left + 1) * (right - left) // 2 - old_pairs
                    added = [row[k] for k in range(left, right) if row[k] not in free]
                    if (added and len(free) + len(added) <= edge_limit and
                            current_pairs + added_pairs <= pair_limit):
                        cost = (1.0 + added_pairs) * (0.5 + rng.random())
                        cost *= 1.0 + abs(h[u] - focus) / max(1.0, value)
                        options.append((cost, added))
            u = v
        if not options:
            break
        free.update(min(options, key=lambda item: item[0])[1])
    if relaxed is None or relaxed[0] >= value - eps or time.perf_counter() >= deadline:
        return best

    pairs = [(u, v) for block in blocks for i, u in enumerate(block)
             for v in block[i + 1:]]
    # The stack stores the fixed z relations and introduces no other decision variables.
    stack = [((), 0)]
    while stack and time.perf_counter() < deadline:
        decisions, decided = stack.pop()
        edges = [row[:] for row in base]
        indegree = degree[:]
        for u, v in decisions:
            edges[u].append(v)
            indegree[v] += 1
        while time.perf_counter() < deadline:
            evaluated = _partial(p, edges, indegree, deadline)
            if evaluated is None:
                break
            lower, h, q, order, _ = evaluated
            ceiling = best[0] - eps
            if lower >= ceiling:
                break
            impossible = False
            # Single-machine subset load lower bound: every suffix after sorting by release time or tail length is a valid subset.
            for block in blocks:
                for key, other in ((h, q), (q, h)):
                    work, minimum = 0.0, math.inf
                    for u in sorted(block, key=lambda x: key[x], reverse=True):
                        work += p[u]
                        minimum = min(minimum, other[u])
                        if key[u] + work + minimum >= ceiling:
                            impossible = True
                            break
                    if impossible:
                        break
                if impossible:
                    break
            if impossible or time.perf_counter() >= deadline:
                break
            forced = []
            branch = None
            branch_key = None
            for i, (u, v) in enumerate(pairs):
                if decided & (1 << i):
                    continue
                uv = h[u] + p[u] + p[v] + q[v]
                vu = h[v] + p[v] + p[u] + q[u]
                if uv >= ceiling and vu >= ceiling:
                    impossible = True
                    break
                if uv >= ceiling:
                    forced.append((v, u, i))
                elif vu >= ceiling:
                    forced.append((u, v, i))
                elif h[u] + p[u] > h[v] and h[v] + p[v] > h[u]:
                    key = (min(uv, vu), max(uv, vu), min(p[u], p[v]))
                    if branch_key is None or key > branch_key:
                        branch_key = key
                        first, second = ((u, v), (v, u)) if uv < vu else ((v, u), (u, v))
                        branch = first, second, i
            if impossible:
                break
            if forced:
                extension = []
                for u, v, i in forced:
                    edges[u].append(v)
                    indegree[v] += 1
                    extension.append((u, v))
                    decided |= 1 << i
                decisions += tuple(extension)
                continue
            if branch is not None:
                first, second, i = branch
                mask = decided | (1 << i)
                stack.append((decisions + (second,), mask))
                stack.append((decisions + (first,), mask))
                break
            # Earliest start times are now free of machine conflicts; complete the machine orders and verify the full graph again.
            rank = [0] * n
            for i, u in enumerate(order):
                rank[u] = i
            sequences = [sorted(row, key=lambda u: (h[u], h[u] + p[u], rank[u]))
                         for row in rows]
            a, b = _links(sequences, n)
            result = _forward(p, jp, js, a, b, deadline)
            if result is not None and result[2] < best[0]:
                starts, topo, makespan = result
                best = (makespan, a, b, starts, _tails(p, js, b, topo))
            break
    return best


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
    strength = max(2, int(math.sqrt(nj + nm) * (1.0 + scale) / 3.0))
    base_tenure = 3 + int(0.35 * math.sqrt(count) + scale)
    patience = budget / (3.0 + scale + math.log1p(count) / 2.0)
    diversity = max(2, int(math.sqrt(count) / (1.0 + scale)))
    pool = [best]

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
        seed_end = min(deadline, time.perf_counter() + budget / (4.0 + scale))
        for rule in (0, 1, 2, 3):
            if time.perf_counter() >= seed_end:
                break
            state = evaluate(_construct(nj, nm, p, machine, jp, js, rule,
                                        False, 0.0, rng, seed_end))
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
        epoch_best = best
        repair_share = scale / (7.0 + 2.0 * scale)
        last_progress = time.perf_counter()
        force_restart = False
        while time.perf_counter() < deadline:
            now = time.perf_counter()
            if force_restart or now - last_progress >= patience:
                archive((mk, mp[:], ms[:], head, tail))
                restarts += 1
                # Reorder several critical windows at once on the best local state, rather than perturbing a single operation.
                source = best if restarts % 2 else epoch_best
                repair_end = min(deadline, now + patience * repair_share)
                repaired = _repair(nj, nm, p, machine, jp, js, source,
                                   scale, rng, repair_end)
                if repaired[0] < source[0]:
                    archive(repaired)
                    if repaired[0] < best[0]:
                        best = repaired
                    mk, a, b, head, tail = repaired
                    mp, ms = a[:], b[:]
                    tabu.clear()
                    seen.clear()
                    tenure = base_tenure
                    local_best, epoch_best = mk, repaired
                    last_progress = time.perf_counter()
                    force_restart = False
                    continue
                state = None
                # Alternate new constructions with elite backtracking so the search is never stuck in one set of machine orders.
                if restarts % 3 == 0:
                    phase_end = min(deadline, now + patience)
                    state = evaluate(_construct(nj, nm, p, machine, jp, js,
                                                rng.randrange(4), bool(restarts % 2),
                                                0.3 + rng.random(), rng, phase_end))
                if state is None:
                    state = best if rng.random() < 0.5 else rng.choice(pool)
                mk, a, b, head, tail = state
                mp, ms = a[:], b[:]
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
                while kick < strength and time.perf_counter() < min(deadline, now + patience):
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
                epoch_best = (mk, mp[:], ms[:], head, tail)
                force_restart = False
                continue

            iteration += 1
            proposals = _moves(p, jp, js, mp, ms, head, tail, mk)
            ranked, blocked = [], []
            for (u, v, after), estimate in proposals.items():
                relation = (v, u) if after else (u, v)
                expiry = tabu.get(relation, 0)
                forbidden = expiry > iteration
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
            for _, tie, u, v, after, forbidden in ranked:
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
                epoch_best = (mk, mp[:], ms[:], head, tail)
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
        budget = float(time_limit_s)
        if not math.isfinite(budget) or budget < 0.0:
            budget = 0.0
    except Exception:
        budget = 0.0
    try:
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
