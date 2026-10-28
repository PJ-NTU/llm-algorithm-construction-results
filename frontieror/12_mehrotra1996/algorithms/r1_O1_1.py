"""Core decomposition and partition crossover graph colouring
Combines bipartite detection, low-degree peeling, bitset construction, dual tabu search, partition crossover and pausable exact search, continuously keeping a complete feasible colouring.
"""

import heapq
import math
import random
import time


def _vertices(mask):
    while mask:
        bit = mask & -mask
        yield bit.bit_length() - 1
        mask ^= bit


def _canonical(colors):
    mapping = {}
    result = []
    for c in colors:
        if c not in mapping:
            mapping[c] = len(mapping)
        result.append(mapping[c])
    return result


def _groups(colors, k):
    groups = [0] * k
    for v, c in enumerate(colors):
        if c >= 0:
            groups[c] |= 1 << v
    return groups


def _greedy(a, order, stop):
    colors = [-1] * len(a)
    groups = []
    for i, v in enumerate(order):
        if i & 63 == 0 and time.perf_counter() >= stop:
            return None
        for c, group in enumerate(groups):
            if not a[v] & group:
                break
        else:
            c = len(groups)
            groups.append(0)
        colors[v] = c
        groups[c] |= 1 << v
    return colors


def _dsatur(a, stop, rng, noise=0.0):
    n = len(a)
    colors = [-1] * n
    masks = [0] * n
    priority = [-a[v].bit_count() * (1.0 + noise * rng.random())
                for v in range(n)]
    heap = [(0, priority[v], v) for v in range(n)]
    heapq.heapify(heap)
    left = n
    while heap and left:
        if time.perf_counter() >= stop:
            return None
        sat, _, v = heapq.heappop(heap)
        if colors[v] >= 0 or -sat != masks[v].bit_count():
            continue
        bit = (~masks[v]) & (masks[v] + 1)
        colors[v] = bit.bit_length() - 1
        left -= 1
        for u in _vertices(a[v]):
            if colors[u] < 0 and not masks[u] & bit:
                masks[u] |= bit
                heapq.heappush(heap, (-masks[u].bit_count(), priority[u], u))
    return colors


def _rlf(a, stop, rng, randomized=False):
    n = len(a)
    remaining = (1 << n) - 1
    colors = [-1] * n
    c = 0
    while remaining:
        if time.perf_counter() >= stop:
            return None
        start = max(_vertices(remaining), key=lambda v:
                    ((a[v] & remaining).bit_count(), rng.random() if randomized else -v))
        candidates = remaining
        excluded = 0
        v = start
        while candidates:
            colors[v] = c
            vb = 1 << v
            remaining ^= vb
            blocked = candidates & a[v]
            candidates &= ~(blocked | vb)
            excluded |= blocked
            if not candidates:
                break
            score = None
            for i, u in enumerate(_vertices(candidates)):
                if i & 63 == 0 and time.perf_counter() >= stop:
                    return None
                value = ((a[u] & excluded).bit_count(),
                         -(a[u] & candidates).bit_count(),
                         rng.random() if randomized else -u)
                if score is None or value > score:
                    score, v = value, u
        c += 1
    return colors


def _clique(a, stop, rng, previous):
    best = previous[:]
    first = not best
    while time.perf_counter() < stop:
        candidates = (1 << len(a)) - 1
        clique = []
        while candidates:
            score = -1.0
            chosen = -1
            for i, v in enumerate(_vertices(candidates)):
                if i & 63 == 0 and time.perf_counter() >= stop:
                    return best
                d = (a[v] & candidates).bit_count()
                value = d if first else (d + 1) * (0.75 + 0.5 * rng.random())
                if value > score:
                    score, chosen = value, v
            clique.append(chosen)
            candidates &= a[chosen]
        if len(clique) > len(best):
            best = clique
        first = False
    return best


class _Exact:
    """Save the progress of fixed-colour-count search with rollback-capable domain propagation and an explicit stack."""

    def __init__(self, a, k, clique, guide):
        self.a, self.k, self.guide = a, k, guide
        self.colors = [-1] * len(a)
        self.domains = [(1 << k) - 1] * len(a)
        self.remaining = (1 << len(a)) - 1
        self.used = len(clique)
        self.roots = [(v, c) for c, v in enumerate(clique)]
        self.trail = []
        self.frames = []
        self.initialized = False
        self.expand = True

    def _rollback(self, mark, remaining, used):
        while len(self.trail) > mark:
            v, domain, color = self.trail.pop()
            self.domains[v] = domain
            self.colors[v] = color
        self.remaining, self.used = remaining, used

    def _place(self, queue, stop):
        while queue:
            if time.perf_counter() >= stop:
                return None
            v, c = queue.pop()
            if self.colors[v] >= 0:
                if self.colors[v] != c:
                    return False
                continue
            bit = 1 << c
            if not self.domains[v] & bit:
                return False
            self.trail.append((v, self.domains[v], -1))
            self.domains[v] = bit
            self.colors[v] = c
            self.remaining &= ~(1 << v)
            self.used = max(self.used, c + 1)
            for u in _vertices(self.a[v] & self.remaining):
                old = self.domains[u]
                if old & bit:
                    self.trail.append((u, old, -1))
                    new = old ^ bit
                    self.domains[u] = new
                    if not new:
                        return False
                    if new & (new - 1) == 0:
                        queue.append((u, new.bit_length() - 1))
        return True

    def advance(self, stop):
        if not self.initialized:
            rem, used = self.remaining, self.used
            result = self._place(self.roots[:], stop)
            if result is None:
                self._rollback(0, rem, used)
                return 0, None
            if not result:
                return -1, None
            self.initialized = True
        while time.perf_counter() < stop:
            if self.expand:
                if not self.remaining:
                    return 1, self.colors[:]
                chosen, score = -1, None
                for i, v in enumerate(_vertices(self.remaining)):
                    if i & 63 == 0 and time.perf_counter() >= stop:
                        return 0, None
                    key = (self.domains[v].bit_count(),
                           -(self.a[v] & self.remaining).bit_count())
                    if score is None or key < score:
                        chosen, score = v, key
                options = self.domains[chosen] & ((1 << min(self.k, self.used + 1)) - 1)
                self.frames.append([chosen, options, len(self.trail),
                                    self.remaining, self.used])
                self.expand = False
            if not self.frames:
                return -1, None
            frame = self.frames[-1]
            v, options, mark, remaining, used = frame
            self._rollback(mark, remaining, used)
            if not options:
                self.frames.pop()
                continue
            preferred = 1 << (self.guide[v] % self.k)
            bit = preferred if options & preferred else options & -options
            frame[1] ^= bit
            result = self._place([(v, bit.bit_length() - 1)], stop)
            if result is None:
                self._rollback(mark, remaining, used)
                frame[1] |= bit
                return 0, None
            if result:
                self.expand = True
        return 0, None


def _seed(a, base, k, round_id, rng, stop):
    classes = [[] for _ in range(max(base) + 1)]
    for v, c in enumerate(base):
        classes[c].append(v)
    ranking = sorted(range(len(classes)), key=lambda c:
                     (len(classes[c]), sum(a[v].bit_count() for v in classes[c])))
    removed = ranking[round_id % len(ranking)]
    colors = [(-1 if c == removed else c - (c > removed)) for c in base]
    return _fill(a, colors, k, rng, stop)


def _fill(a, colors, k, rng, stop):
    groups = _groups(colors, k)
    holes = [v for v, c in enumerate(colors) if c < 0]
    rng.shuffle(holes)
    holes.sort(key=lambda v: -a[v].bit_count())
    for i, v in enumerate(holes):
        if i & 63 == 0 and time.perf_counter() >= stop:
            return None
        costs = [(a[v] & g).bit_count() for g in groups]
        low = min(costs)
        c = rng.choice([c for c in range(k) if costs[c] == low])
        colors[v] = c
        groups[c] |= 1 << v
    return colors


def _fitness(a, colors, k):
    groups = _groups(colors, k)
    return sum((a[v] & groups[c]).bit_count() for v, c in enumerate(colors)) // 2


def _crossover(a, first, second, k, rng, stop):
    parents = [_groups(first, k), _groups(second, k)]
    colors = [-1] * len(a)
    remaining = (1 << len(a)) - 1
    offset = rng.randrange(2)
    for c in range(k):
        if time.perf_counter() >= stop:
            return None
        parent = parents[(c + offset) % 2]
        largest = max((group & remaining).bit_count() for group in parent)
        choices = [i for i, group in enumerate(parent)
                   if (group & remaining).bit_count() == largest]
        selected = parent[rng.choice(choices)] & remaining
        for v in _vertices(selected):
            colors[v] = c
        remaining &= ~selected
    return _fill(a, colors, k, rng, stop)


def _tabu(a, initial, k, rng, stop, strength):
    colors = initial[:]
    groups = _groups(colors, k)
    bad = 0
    total = 0
    for v, c in enumerate(colors):
        d = (a[v] & groups[c]).bit_count()
        total += d
        if d:
            bad |= 1 << v
    total //= 2
    best_value, best = total, colors[:]
    tabu = [[0] * k for _ in a]
    iteration = 0
    start = last_gain = time.perf_counter()
    window = max((stop - start) / (2.0 + math.log1p(len(a))), 1e-6)
    jitter = max(2, int(math.log1p(len(a)) + math.log1p(strength)))
    while total and time.perf_counter() < stop:
        iteration += 1
        delta_best = len(a) + 1
        chosen_v = chosen_c = -1
        ties = 0
        fallback = (len(a) + 1, -1, -1)
        for i, v in enumerate(_vertices(bad)):
            if i & 31 == 0 and time.perf_counter() >= stop:
                return best_value, best
            old = colors[v]
            old_cost = (a[v] & groups[old]).bit_count()
            for c in range(k):
                if c == old:
                    continue
                delta = (a[v] & groups[c]).bit_count() - old_cost
                if delta < fallback[0]:
                    fallback = (delta, v, c)
                if tabu[v][c] > iteration and total + delta >= best_value:
                    continue
                if delta < delta_best:
                    delta_best, chosen_v, chosen_c, ties = delta, v, c, 1
                elif delta == delta_best:
                    ties += 1
                    if rng.randrange(ties) == 0:
                        chosen_v, chosen_c = v, c
        if chosen_v < 0:
            delta_best, chosen_v, chosen_c = fallback
        if chosen_v < 0:
            break
        v, c = chosen_v, chosen_c
        old = colors[v]
        bit = 1 << v
        affected = (a[v] & (groups[old] | groups[c])) | bit
        groups[old] ^= bit
        groups[c] |= bit
        colors[v] = c
        for u in _vertices(affected):
            if a[u] & groups[colors[u]]:
                bad |= 1 << u
            else:
                bad &= ~(1 << u)
        total += delta_best
        now = time.perf_counter()
        reactive = 1.0 + 0.25 * math.log1p((now - last_gain) / window)
        tenure = int(0.6 * bad.bit_count() * reactive) + rng.randint(1, jitter)
        tabu[v][old] = iteration + tenure
        if total < best_value:
            best_value, best, last_gain = total, colors[:], now
    return best_value, best


def _partial(a, initial, k, rng, stop, strength):
    colors = initial[:]
    original = _groups(colors, k)
    order = list(range(len(a)))
    rng.shuffle(order)
    order.sort(key=lambda v: (a[v] & original[colors[v]]).bit_count())
    groups = [0] * k
    holes = 0
    for v in order:
        c = colors[v]
        if a[v] & groups[c]:
            colors[v] = -1
            holes |= 1 << v
        else:
            groups[c] |= 1 << v
    best_holes, best = holes.bit_count(), colors[:]
    tabu = [[0] * k for _ in a]
    iteration = 0
    jitter = max(2, int(math.log1p(len(a)) + math.log1p(strength)))
    while holes and time.perf_counter() < stop:
        iteration += 1
        count = holes.bit_count()
        chosen_v = chosen_c = -1
        delta_best = len(a) + 1
        fallback = (len(a) + 1, -1, -1)
        ties = 0
        for i, v in enumerate(_vertices(holes)):
            if i & 31 == 0 and time.perf_counter() >= stop:
                return None, best
            for c in range(k):
                delta = (a[v] & groups[c]).bit_count() - 1
                if delta < fallback[0]:
                    fallback = (delta, v, c)
                if tabu[v][c] > iteration and count + delta >= best_holes:
                    continue
                if delta < delta_best:
                    delta_best, chosen_v, chosen_c, ties = delta, v, c, 1
                elif delta == delta_best:
                    ties += 1
                    if rng.randrange(ties) == 0:
                        chosen_v, chosen_c = v, c
        if chosen_v < 0:
            _, chosen_v, chosen_c = fallback
        if chosen_v < 0:
            break
        v, c = chosen_v, chosen_c
        expelled = a[v] & groups[c]
        groups[c] = (groups[c] & ~expelled) | (1 << v)
        holes = (holes & ~(1 << v)) | expelled
        colors[v] = c
        tenure = int(0.6 * holes.bit_count()) + rng.randint(1, jitter)
        for u in _vertices(expelled):
            colors[u] = -1
            tabu[u][c] = iteration + tenure
        if holes.bit_count() < best_holes:
            best_holes, best = holes.bit_count(), colors[:]
    return (0 if not holes else None), (colors if not holes else best)


def _target(part, stop):
    a, k = part['a'], part['k'] - 1
    degree = [mask.bit_count() for mask in a]
    queue = [v for v, d in enumerate(degree) if d < k]
    alive = (1 << len(a)) - 1
    removed = []
    while queue:
        if len(removed) & 63 == 0 and time.perf_counter() >= stop:
            return None
        v = queue.pop()
        bit = 1 << v
        if not alive & bit:
            continue
        alive ^= bit
        removed.append(v)
        for u in _vertices(a[v] & alive):
            degree[u] -= 1
            if degree[u] == k - 1:
                queue.append(u)
    if not removed:
        return {'a': a, 'k': k, 'nodes': list(range(len(a))), 'removed': [],
                'base': part['best'][:], 'clique': part['clique'][:],
                'pool': [], 'round': 0, 'exact': None}
    nodes = list(_vertices(alive))
    index = {v: i for i, v in enumerate(nodes)}
    core = []
    for i, v in enumerate(nodes):
        if i & 31 == 0 and time.perf_counter() >= stop:
            return None
        core.append(sum(1 << index[u] for u in _vertices(a[v] & alive)))
    base = _canonical([part['best'][v] for v in nodes])
    clique = [index[v] for v in part['clique'] if v in index]
    return {'a': core, 'k': k, 'nodes': nodes, 'removed': removed,
            'base': base, 'clique': clique, 'pool': [], 'round': 0, 'exact': None}


def _restore(part, task, core_colors):
    k = task['k']
    colors = [-1] * len(part['a'])
    groups = [0] * k
    for v, c in zip(task['nodes'], core_colors):
        colors[v] = c
        groups[c] |= 1 << v
    for v in reversed(task['removed']):
        for c in range(k):
            if not part['a'][v] & groups[c]:
                colors[v] = c
                groups[c] |= 1 << v
                break
        if colors[v] < 0:
            return None
    return colors


def solve(instance: dict, time_limit_s: float) -> dict:
    start = time.perf_counter()
    answer = []
    try:
        n = max(0, int(instance.get('n', 0)))
        answer = list(range(n))
        duration = max(0.0, float(time_limit_s))
        if n == 0 or not duration:
            return {'coloring': answer}
        deadline = start + 0.97 * duration
        rng = random.Random(20261002 + 1000003 * n)
        adj = [set() for _ in range(n)]
        for i, edge in enumerate(instance.get('edges', [])):
            if i & 1023 == 0 and time.perf_counter() >= deadline:
                return {'coloring': answer}
            if len(edge) >= 2:
                u, v = int(edge[0]), int(edge[1])
                if 0 <= u < n and 0 <= v < n and u != v:
                    adj[u].add(v)
                    adj[v].add(u)

        # There are no edges between components, so the same colour numbers can safely be reused.
        seen = [-1] * n
        components = []
        for root in range(n):
            if time.perf_counter() >= deadline:
                return {'coloring': answer}
            if seen[root] >= 0:
                continue
            seen[root] = 0
            stack, nodes = [root], []
            bipartite = True
            while stack:
                if len(nodes) & 63 == 0 and time.perf_counter() >= deadline:
                    return {'coloring': answer}
                v = stack.pop()
                nodes.append(v)
                for u in adj[v]:
                    if seen[u] < 0:
                        seen[u] = 1 - seen[v]
                        stack.append(u)
                    elif seen[u] == seen[v]:
                        bipartite = False
            if bipartite:
                for v in nodes:
                    answer[v] = seen[v]
            else:
                components.append(nodes)
        if not components:
            return {'coloring': answer}

        parts = []

        def accept(part, candidate):
            if candidate is None or len(candidate) != len(part['a']):
                return False
            if any(c < 0 for c in candidate):
                return False
            candidate = _canonical(candidate)
            k = max(candidate) + 1
            if k >= part['k']:
                return False
            groups = _groups(candidate, k)
            if any(part['a'][v] & groups[c] for v, c in enumerate(candidate)):
                return False
            part['best'], part['k'], part['task'] = candidate, k, None
            for v, c in zip(part['nodes'], candidate):
                answer[v] = c
            return True

        for nodes in sorted(components, key=len, reverse=True):
            if time.perf_counter() >= deadline:
                return {'coloring': _canonical(answer)}
            index = {v: i for i, v in enumerate(nodes)}
            a = []
            for i, v in enumerate(nodes):
                if i & 31 == 0 and time.perf_counter() >= deadline:
                    return {'coloring': _canonical(answer)}
                a.append(sum(1 << index[u] for u in adj[v]))
            q = len(a)
            part = {'nodes': nodes, 'a': a, 'best': list(range(q)), 'k': q,
                    'lb': 3, 'clique': [], 'stage': 0, 'task': None, 'visits': 0}
            for i, v in enumerate(nodes):
                answer[v] = i
            parts.append(part)
            accept(part, _greedy(a, sorted(range(q), key=lambda v: -a[v].bit_count()), deadline))

        # Each output colouring corresponds to a one-hot encoding of x[v][c], and every colour that appears has y[c]=1.
        # Only strict improvements that pass the adjacency check are accepted; a mid-way failure does not overwrite the feasible solution.
        pool_capacity = max(2, int(2 + math.log1p(duration)))
        while time.perf_counter() < deadline:
            upper = max(p['k'] for p in parts)
            lower = max(p['lb'] for p in parts)
            if upper <= lower:
                break
            eligible = [p for p in parts if p['k'] == upper and p['lb'] < upper]
            part = min(eligible, key=lambda p: p['visits'])
            part['visits'] += 1
            a = part['a']
            now = time.perf_counter()
            phase = duration / (3.0 + math.sqrt(duration) + 0.75 * math.log1p(len(a)))
            stop = min(deadline, now + phase)
            if part['stage'] == 0:
                part['stage'] = 1
                accept(part, _dsatur(a, stop, rng))
                continue
            if part['stage'] == 1:
                part['stage'] = 2
                accept(part, _rlf(a, stop, rng))
                continue
            if part['stage'] == 2:
                part['stage'] = 3
                part['clique'] = _clique(a, min(stop, now + phase / (2 + math.log1p(len(a)))),
                                        rng, part['clique'])
                part['lb'] = max(part['lb'], len(part['clique']))
                continue
            if part['task'] is None:
                part['task'] = _target(part, stop)
                if part['task'] is None:
                    continue
            task = part['task']
            core, k, base = task['a'], task['k'], task['base']
            if not core or max(base) < k:
                accept(part, _restore(part, task, base))
                continue
            r = task['round']
            task['round'] += 1
            if r % 6 == 3:
                if task['exact'] is None:
                    task['exact'] = _Exact(core, k, task['clique'],
                                           task['pool'][0][1] if task['pool'] else base)
                status, candidate = task['exact'].advance(stop)
                if status == 1:
                    accept(part, _restore(part, task, candidate))
                elif status == -1:
                    part['lb'] = part['k']
                continue
            if r % 6 == 5:
                order_classes = list(range(part['k']))
                rng.shuffle(order_classes)
                rank = {c: i for i, c in enumerate(order_classes)}
                order = list(range(len(a)))
                rng.shuffle(order)
                order.sort(key=lambda v: rank[part['best'][v]])
                candidate = _greedy(a, order, stop)
                if accept(part, candidate):
                    continue
                # New partitions with an equal colour count can also provide different colour-elimination seeds for the next round.
                if candidate is not None and max(candidate) + 1 == part['k']:
                    task['base'] = _canonical([candidate[v] for v in task['nodes']])
                continue
            pool = task['pool']
            if len(pool) >= 2 and r % 3 == 2:
                first, second = rng.sample(pool, 2)
                seed = _crossover(core, first[1], second[1], k, rng, stop)
            elif pool and r % 3 == 1:
                seed = pool[rng.randrange(len(pool))][1][:]
            else:
                seed = _seed(core, base, k, r, rng, stop)
            if seed is None or time.perf_counter() >= stop:
                continue
            if r % 6 in (0, 4):
                # Reserve time to fill empty slots, turning partial colourings into reusable crossover parents.
                local_stop = min(stop, time.perf_counter() +
                                 0.93 * max(0.0, stop - time.perf_counter()))
                value, candidate = _partial(core, seed, k, rng, local_stop, duration)
                if value is None:
                    candidate = _fill(core, candidate, k, rng, stop)
                    if candidate is None:
                        continue
                    value = _fitness(core, candidate, k)
            else:
                value, candidate = _tabu(core, seed, k, rng, stop, duration)
            if value == 0:
                accept(part, _restore(part, task, candidate))
                continue
            candidate = _canonical(candidate)
            if not any(candidate == item[1] for item in pool):
                pool.append((value, candidate))
                pool.sort(key=lambda item: item[0])
                if len(pool) > pool_capacity:
                    pool.pop()
        return {'coloring': _canonical(answer)}
    except Exception:
        return {'coloring': answer}
