"""Group-linked adaptive large-neighbourhood search for two-echelon routing
Combines capacity-based construction, dynamic giant-tour splitting, whole-group relocation and destroy-and-repair, optimising the splittable first-echelon deliveries and the routes of both echelons.
"""
import itertools
import math
import random
import time


def solve(instance: dict, time_limit_s: float) -> dict:
    start = time.perf_counter()
    budget = max(0.001, float(time_limit_s))
    reserve = min(0.08, max(0.001, budget * 0.018))
    end = start + max(0.0001, budget - reserve)
    rng = random.Random(19349663)
    eps = 1e-8
    inf = float('inf')
    q1, q2 = float(instance['Q1']), float(instance['Q2'])
    t1, t2 = int(instance['T1']), int(instance['T2'])
    ns, nc, ng = (int(instance[k]) for k in ('num_satellites', 'num_customers', 'num_groups'))
    depot = instance['depot_id']
    sid, cid = list(instance['satellites_id']), list(instance['customers_id'])
    dem = list(map(float, instance['customers_demand']))
    cap = list(map(float, instance['satellites_capacity']))
    handling = list(map(float, instance['satellites_handling_cost']))
    vmax = [min(t2, int(v)) for v in instance['satellites_max_vehicles']]
    cxy = list(zip(instance['customers_x'], instance['customers_y']))
    sxy = list(zip(instance['satellites_x'], instance['satellites_y']))
    dxy = (instance['depot_x'], instance['depot_y'])
    index = {c: i for i, c in enumerate(cid)}
    groups = [[index[c] for c in row] for row in instance['groups_customer_ids']]
    gd = [sum(dem[i] for i in row) for row in groups]
    belong = [0] * nc
    for g, row in enumerate(groups):
        for i in row:
            belong[i] = g

    def eu(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    ds = [[eu(p, x) for x in cxy] for p in sxy]
    sd = [eu(dxy, p) for p in sxy]
    ss = [[eu(a, b) for b in sxy] for a in sxy]
    # Large instances cache distances on demand, so initialisation alone does not exhaust memory and time.
    if nc * nc <= max(4096, min(4000000, int(200000 * budget))):
        matrix = [[eu(a, b) for b in cxy] for a in cxy]

        def cc(i, j):
            return matrix[i][j]
    else:
        distance_cache = {}

        def cc(i, j):
            if i > j:
                i, j = j, i
            key = (i, j)
            v = distance_cache.get(key)
            if v is None:
                v = eu(cxy[i], cxy[j])
                if len(distance_cache) < max(4096, int(50000 * budget)):
                    distance_cache[key] = v
            return v

    angles = [[math.atan2(p[1] - sxy[s][1], p[0] - sxy[s][0]) for p in cxy] for s in range(ns)]
    effective = [min(cap[s], vmax[s] * q2) for s in range(ns)]

    def edge(s, a, b):
        if a < 0:
            return 0.0 if b < 0 else ds[s][b]
        return ds[s][a] if b < 0 else cc(a, b)

    def route_cost(s, r):
        if not r:
            return 0.0
        return ds[s][r[0]] + ds[s][r[-1]] + sum(cc(a, b) for a, b in zip(r, r[1:]))

    def cost2(routes):
        return sum(route_cost(s, r) for s, rr in enumerate(routes) for r in rr)

    def clone(routes):
        return [[r[:] for r in rr] for rr in routes]

    def nearest(s, nodes):
        left = set(nodes)
        r = []
        last = -1
        while left:
            x = min(left, key=lambda i: edge(s, last, i))
            r.append(x)
            left.remove(x)
            last = x
        return r

    def opt2(s, r, stop):
        r = list(r)
        changed = True
        while changed and time.perf_counter() < stop:
            changed = False
            for i in range(len(r) - 1):
                a = r[i - 1] if i else -1
                b = r[i]
                for j in range(i + 1, len(r)):
                    c = r[j]
                    d = r[j + 1] if j + 1 < len(r) else -1
                    if edge(s, a, c) + edge(s, b, d) + eps < edge(s, a, b) + edge(s, c, d):
                        r[i:j + 1] = reversed(r[i:j + 1])
                        changed = True
                        break
                if changed or time.perf_counter() >= stop:
                    break
        return r

    tour_cache = {0: (0.0, [])}
    hk_cache = {}
    exact_tours = ns < 20 and ns * ns * (1 << ns) <= min(350000, 12000 * budget)

    def held(mask, j):
        key = (mask, j)
        old = hk_cache.get(key)
        if old is not None:
            return old
        rem = mask ^ (1 << j)
        if not rem:
            ans = (sd[j], -1)
        else:
            ans = (inf, -1)
            bits = rem
            while bits:
                bit = bits & -bits
                p = bit.bit_length() - 1
                value = held(rem, p)[0] + ss[p][j]
                if value < ans[0]:
                    ans = (value, p)
                bits ^= bit
        hk_cache[key] = ans
        return ans

    def tour(mask):
        old = tour_cache.get(mask)
        if old is not None:
            return old
        nodes = [s for s in range(ns) if mask & (1 << s)]
        if exact_tours and time.perf_counter() < end:
            last = min(nodes, key=lambda s: held(mask, s)[0] + sd[s])
            value = held(mask, last)[0] + sd[last]
            order, rem = [], mask
            while last >= 0:
                order.append(last)
                nxt = held(rem, last)[1]
                rem ^= 1 << last
                last = nxt
            order.reverse()
        else:
            left = set(nodes)
            order = []
            while left:
                s = min(left, key=lambda j: ss[order[-1]][j] if order else sd[j])
                left.remove(s)
                order.append(s)
            changed = True
            while changed and time.perf_counter() < end:
                changed = False
                for i in range(len(order) - 1):
                    a = order[i - 1] if i else -1
                    b = order[i]
                    for j in range(i + 1, len(order)):
                        c = order[j]
                        d = order[j + 1] if j + 1 < len(order) else -1
                        oldv = (sd[b] if a < 0 else ss[a][b]) + (sd[c] if d < 0 else ss[c][d])
                        newv = (sd[c] if a < 0 else ss[a][c]) + (sd[b] if d < 0 else ss[b][d])
                        if newv + eps < oldv:
                            order[i:j + 1] = reversed(order[i:j + 1])
                            changed = True
                            break
                    if changed:
                        break
            value = sd[order[0]] + sd[order[-1]] + sum(ss[a][b] for a, b in zip(order, order[1:]))
        tour_cache[mask] = value, order
        return value, order

    first_cache = {}

    def first_plan(loads, stop, polish=False):
        key = tuple(loads)
        cached = first_cache.get(key)
        if cached is not None and not polish:
            return cached
        active = [s for s in range(ns) if loads[s] > eps]
        if not active:
            ans = (0.0, [[0.0] * ns for _ in range(t1)])
            first_cache[key] = ans
            return ans
        best = cached

        def offer(deliveries):
            nonlocal best
            value = 0.0
            for row in deliveries:
                mask = sum(1 << s for s in active if row[s] > eps)
                value += tour(mask)[0]
            if best is None or value + eps < best[0]:
                best = (value, [r[:] for r in deliveries])

        # Consecutive loading always allows splitting; the same satellite may be resupplied by several vehicles together.
        mask = sum(1 << s for s in active)
        base = tour(mask)[1]
        orders = [base, base[::-1], sorted(active, key=lambda s: -loads[s]), sorted(active, key=lambda s: -sd[s])]
        for order in orders:
            rows = [[0.0] * ns for _ in range(t1)]
            k, room = 0, q1
            for s in order:
                rem = loads[s]
                while rem > eps and k < t1:
                    qty = min(rem, room)
                    rows[k][s] += qty
                    rem -= qty
                    room -= qty
                    if room <= eps:
                        k += 1
                        room = q1
            offer(rows)
            # Choose loading positions by the actual added first-echelon distance, preferring unsplit deliveries.
            rows = [[0.0] * ns for _ in range(t1)]
            rooms, masks = [q1] * t1, [0] * t1
            for s in order:
                rem = loads[s]
                while rem > eps:
                    choices = [k for k in range(t1) if rooms[k] + eps >= rem]
                    if not choices:
                        choices = [k for k in range(t1) if rooms[k] > eps]
                    if not choices:
                        break
                    k = min(choices, key=lambda k: (tour(masks[k] | (1 << s))[0] - tour(masks[k])[0], -min(rooms[k], rem), rooms[k]))
                    qty = min(rem, rooms[k])
                    rows[k][s] += qty
                    rooms[k] -= qty
                    masks[k] |= 1 << s
                    rem -= qty
            offer(rows)
            if time.perf_counter() >= stop:
                break

        # For small satellite sets, enumerate unsplit partitions; compare them with split solutions instead of accepting unsplit ones early.
        m = len(active)
        if m < 14 and 3 ** m <= max(100, 25000 * max(0.0, stop - time.perf_counter())) and max(loads) <= q1 + eps:
            size = 1 << m
            subload, globalmask = [0.0] * size, [0] * size
            for a in range(1, size):
                bit = a & -a
                j = bit.bit_length() - 1
                subload[a] = subload[a ^ bit] + loads[active[j]]
                globalmask[a] = globalmask[a ^ bit] | (1 << active[j])
            memo = {}

            def partition(rem, k):
                if not rem:
                    return 0.0, []
                if k <= 0 or subload[rem] > k * q1 + eps or time.perf_counter() >= stop:
                    return inf, []
                state = (rem, k)
                if state in memo:
                    return memo[state]
                if subload[rem] <= q1 + eps:
                    ans = (tour(globalmask[rem])[0], [rem])
                else:
                    ans = (inf, [])
                anchor = rem & -rem
                sub = rem
                while sub:
                    if sub & anchor and subload[sub] <= q1 + eps:
                        v, parts = partition(rem ^ sub, k - 1)
                        value = v + tour(globalmask[sub])[0]
                        if value < ans[0]:
                            ans = (value, parts + [sub])
                    sub = (sub - 1) & rem
                    if time.perf_counter() >= stop:
                        break
                memo[state] = ans
                return ans

            value, parts = partition(size - 1, min(t1, m))
            if best is None or value + eps < best[0]:
                rows = [[0.0] * ns for _ in range(t1)]
                for k, part in enumerate(parts):
                    for j, s in enumerate(active):
                        if part & (1 << j):
                            rows[k][s] = loads[s]
                offer(rows)

        # Two-by-two cyclic adjustments of shipped quantities keep satellite totals and vehicle loads unchanged and can eliminate repeated visits.
        if polish and best is not None:
            rows = [r[:] for r in best[1]]
            changed = True
            while changed and time.perf_counter() < stop:
                changed = False
                used = [k for k in range(t1) if sum(rows[k]) > eps]
                for ai, a in enumerate(used):
                    ma = sum(1 << s for s in active if rows[a][s] > eps)
                    for b in used[ai + 1:]:
                        mb = sum(1 << s for s in active if rows[b][s] > eps)
                        old = tour(ma)[0] + tour(mb)[0]
                        for source, target, msource, mtarget in ((a, b, ma, mb), (b, a, mb, ma)):
                            room = q1 - sum(rows[target])
                            for sat in active:
                                qty = rows[source][sat]
                                if qty > eps and qty <= room + eps:
                                    na = msource & ~(1 << sat)
                                    nb = mtarget | (1 << sat)
                                    if tour(na)[0] + tour(nb)[0] + eps < old:
                                        rows[source][sat] = 0.0
                                        rows[target][sat] += qty
                                        changed = True
                                        break
                            if changed:
                                break
                        if changed:
                            break
                        for s in active:
                            if rows[a][s] <= eps:
                                continue
                            for h in active:
                                if h == s or rows[b][h] <= eps:
                                    continue
                                qty = min(rows[a][s], rows[b][h])
                                na = (ma | (1 << h))
                                nb = (mb | (1 << s))
                                if rows[a][s] - qty <= eps:
                                    na &= ~(1 << s)
                                if rows[b][h] - qty <= eps:
                                    nb &= ~(1 << h)
                                if tour(na)[0] + tour(nb)[0] + eps < old:
                                    rows[a][s] -= qty
                                    rows[b][s] += qty
                                    rows[b][h] -= qty
                                    rows[a][h] += qty
                                    changed = True
                                    break
                            if changed or time.perf_counter() >= stop:
                                break
                        if changed or time.perf_counter() >= stop:
                            break
                    if changed or time.perf_counter() >= stop:
                        break
                if changed:
                    offer(rows)
        first_cache[key] = best
        return best

    def bfd(nodes, s, randomized=False):
        if randomized:
            order = sorted(nodes, key=lambda i: (-dem[i], rng.random()))
        else:
            order = sorted(nodes, key=lambda i: (-dem[i], angles[s][i]))
        bins, loads = [], []
        for i in order:
            options = [j for j in range(len(bins)) if loads[j] + dem[i] <= q2 + eps]
            if options:
                j = min(options, key=lambda j: (q2 - loads[j] - dem[i], min(cc(i, x) for x in bins[j])))
                bins[j].append(i)
                loads[j] += dem[i]
            else:
                bins.append([i])
                loads.append(dem[i])
        return bins

    def exact_pack(nodes, count, stop):
        # Iterative bin-packing backtracking, avoiding recursion-stack overflow when there are many customers.
        order = sorted(nodes, key=lambda i: -dem[i])
        if not order:
            return []
        if count <= 0 or sum(dem[i] for i in order) > count * q2 + eps:
            return None
        rooms = [q2] * count
        placement = [-1] * len(order)
        options = [None] * len(order)
        pos = 0
        while pos >= 0 and time.perf_counter() < stop:
            if pos == len(order):
                bins = [[] for _ in range(count)]
                for i, b in zip(order, placement):
                    bins[b].append(i)
                return [r for r in bins if r]
            if options[pos] is None:
                seen = set()
                choices = []
                for b in sorted(range(count), key=lambda b: rooms[b], reverse=True):
                    if rooms[b] + eps >= dem[order[pos]] and rooms[b] not in seen:
                        seen.add(rooms[b])
                        choices.append(b)
                options[pos] = choices
            if options[pos]:
                b = options[pos].pop()
                rooms[b] -= dem[order[pos]]
                placement[pos] = b
                pos += 1
                if pos < len(order):
                    options[pos] = None
            else:
                options[pos] = None
                pos -= 1
                if pos >= 0:
                    rooms[placement[pos]] += dem[order[pos]]
        return None

    plan_cache = [dict() for _ in range(ns)]

    def plans(s, mask, stop, force=False):
        cached = plan_cache[s].get(mask)
        if cached is not None and not force:
            return cached
        nodes = [i for g in range(ng) if mask & (1 << g) for i in groups[g]]
        if not nodes:
            return {0: (0.0, [])}
        result = dict(cached) if cached else {}

        def offer(rr, reorder=False):
            if len(rr) > vmax[s]:
                return
            if reorder:
                rr = [opt2(s, nearest(s, r), stop) for r in rr]
            value = sum(route_cost(s, r) for r in rr)
            old = result.get(len(rr))
            if old is None or value < old[0]:
                result[len(rr)] = (value, [r[:] for r in rr])

        bins = bfd(nodes, s)
        offer(bins, True)
        lb = max(1, int(math.ceil(sum(dem[i] for i in nodes) / q2 - eps)))
        if force or not result:
            target = min(vmax[s], len(bins) - 1)
            targets = range(lb, target + 1) if force else (target,)
            for count in targets:
                if count >= lb and time.perf_counter() < stop:
                    packed = exact_pack(nodes, count, min(stop, time.perf_counter() + 0.06 * budget))
                    if packed is not None:
                        offer(packed, True)
                        break
        if time.perf_counter() < stop:
            # Compute savings on the current customer set, without pre-copying all customer pairs of every satellite.
            savings = []
            near_count = min(len(nodes), max(8, int((1 + math.log1p(budget)) * math.sqrt(len(nodes)))))
            for p, i in enumerate(nodes):
                other = nodes[p + 1:]
                if len(other) > near_count and len(nodes) ** 2 > 15000 * budget:
                    other = sorted(other, key=lambda j: cc(i, j))[:near_count]
                savings.extend((ds[s][i] + ds[s][j] - cc(i, j), i, j) for j in other)
                if time.perf_counter() >= stop:
                    break
            savings.sort(reverse=True)
            rr = {i: [i] for i in nodes}
            loads = {i: dem[i] for i in nodes}
            owner = {i: i for i in nodes}
            for _, i, j in savings:
                a, b = owner[i], owner[j]
                if a == b or loads[a] + loads[b] > q2 + eps:
                    continue
                ra, rb = rr[a], rr[b]
                if i not in (ra[0], ra[-1]) or j not in (rb[0], rb[-1]):
                    continue
                if ra[-1] != i:
                    ra = ra[::-1]
                if rb[0] != j:
                    rb = rb[::-1]
                rr[a] = ra + rb
                loads[a] += loads[b]
                del rr[b], loads[b]
                for x in rb:
                    owner[x] = a
                if len(rr) <= vmax[s]:
                    offer(list(rr.values()))
                if time.perf_counter() >= stop:
                    break
            offer([opt2(s, r, stop) for r in rr.values()])

        # Split the giant tour by a shortest path with a vehicle-count dimension, avoiding the boundary losses of sequential filling.
        angular = sorted(nodes, key=lambda i: angles[s][i])
        giant_orders = [angular, angular[::-1]]
        if time.perf_counter() < stop:
            giant_orders.append(nearest(s, nodes))
        rotations = min(len(nodes), max(1, int(math.log1p(budget) + math.sqrt(len(nodes)) / 4)))
        for a in range(1, rotations):
            p = a * len(nodes) // rotations
            giant_orders.append(angular[p:] + angular[:p])
        for order in giant_orders:
            if time.perf_counter() >= stop:
                break
            n = len(order)
            total, internal = [0.0], [0.0]
            for j, i in enumerate(order):
                total.append(total[-1] + dem[i])
                internal.append(internal[-1] + (cc(order[j - 1], i) if j else 0.0))
            segments = [[] for _ in range(n + 1)]
            for j in range(1, n + 1):
                for i in range(j - 1, -1, -1):
                    if total[j] - total[i] > q2 + eps:
                        break
                    c = ds[s][order[i]] + ds[s][order[j - 1]] + internal[j] - internal[i + 1]
                    segments[j].append((i, c))
            prev = [inf] * (n + 1)
            prev[0] = 0.0
            parents = []
            for k in range(1, min(vmax[s], n) + 1):
                vals, par = [inf] * (n + 1), [-1] * (n + 1)
                for j in range(k, n + 1):
                    for i, c in segments[j]:
                        value = prev[i] + c
                        if value < vals[j]:
                            vals[j], par[j] = value, i
                parents.append(par)
                if vals[n] < inf:
                    rr, j = [], n
                    for p in range(k - 1, -1, -1):
                        i = parents[p][j]
                        rr.append(order[i:j])
                        j = i
                    offer([opt2(s, r, stop) for r in rr])
                prev = vals
                if time.perf_counter() >= stop:
                    break
        if result:
            plan_cache[s][mask] = result
        return result

    def reconcile(assignment, stop, force=False):
        loads, masks = [0.0] * ns, [0] * ns
        for g, s in enumerate(assignment):
            loads[s] += gd[g]
            masks[s] |= 1 << g
        if any(loads[s] > effective[s] + eps for s in range(ns)):
            return None
        nonempty = [any(groups[g] for g in range(ng) if assignment[g] == s) for s in range(ns)]
        if sum(max(1, int(math.ceil(loads[s] / q2 - eps))) for s in range(ns) if nonempty[s]) > t2:
            return None
        options = []
        for s in range(ns):
            local_stop = min(stop, time.perf_counter() + budget * 0.008 / max(1.0, math.sqrt(ns)))
            p = plans(s, masks[s], local_stop, force)
            if not p:
                return None
            options.append(p)
        dp = {0: (0.0, [])}
        for p in options:
            nxt = {}
            for used, (value, chosen) in dp.items():
                for count, (c, rr) in p.items():
                    k = used + count
                    if k <= t2 and (k not in nxt or value + c < nxt[k][0]):
                        nxt[k] = (value + c, chosen + [rr])
            dp = nxt
            if not dp:
                return None
        value, routes = min(dp.values(), key=lambda x: x[0])
        first = first_plan(loads, min(stop, time.perf_counter() + budget * 0.003))
        return (value + first[0] + sum(loads[s] * handling[s] for s in range(ns)), list(assignment), clone(routes), loads, first)

    proxy = [[gd[g] * handling[s] + sum(ds[s][i] * max(dem[i] / max(q2, 1.0), 1.0 / max(1, len(groups[g]))) for i in groups[g]) * 2 for s in range(ns)] for g in range(ng)]
    best = None

    def retain(state):
        nonlocal best
        if state is not None and (best is None or state[0] + eps < best[0]):
            best = (state[0], state[1][:], clone(state[2]), state[3][:], state[4])
            return True
        return False

    def construct(noise, stop):
        # Pack bins while constructing the group assignment, rather than relying only on demand lower bounds.
        assignment, loads = [-1] * ng, [0.0] * ns
        bins = [[] for _ in range(ns)]
        order = list(range(ng))
        regrets = [sorted(proxy[g])[1] - min(proxy[g]) if ns > 1 else gd[g] for g in range(ng)]
        order.sort(key=lambda g: -(gd[g] + regrets[g]) * (1 + noise * rng.random()))
        for g in order:
            options = []
            oldcount = sum(map(len, bins))
            for s in range(ns):
                if loads[s] + gd[g] > effective[s] + eps:
                    continue
                nodes = [i for r in bins[s] for i in r] + groups[g]
                rr = bfd(nodes, s, noise > 0)
                maxcount = min(vmax[s], t2 - oldcount + len(bins[s]))
                if len(rr) > maxcount:
                    if time.perf_counter() < stop:
                        rr = exact_pack(nodes, maxcount, min(stop, time.perf_counter() + budget * 0.001))
                    else:
                        rr = None
                if rr is None:
                    continue
                fixed = 2 * sd[s] if not bins[s] and groups[g] else 0.0
                value = (proxy[g][s] + fixed) * (1 + noise * (rng.random() - 0.5))
                options.append((value, s, rr))
            if not options:
                return None
            _, s, rr = min(options, key=lambda x: x[0])
            assignment[g], bins[s] = s, rr
            loads[s] += gd[g]
        routes = [[opt2(s, nearest(s, r), stop) for r in bins[s]] for s in range(ns)]
        first = first_plan(loads, stop)
        return (cost2(routes) + first[0] + sum(loads[s] * handling[s] for s in range(ns)), assignment, routes, loads, first)

    initial_stop = min(end, start + budget * 0.18)
    retain(construct(0.0, initial_stop))
    while best is None and time.perf_counter() < initial_stop:
        retain(construct(0.3 + rng.random(), initial_stop))

    if best is None:
        # Capacity-pruned group backtracking is started only when the fast joint construction fails.
        order = sorted(range(ng), key=lambda g: (-gd[g], -len(groups[g])))
        assignment, loads = [-1] * ng, [0.0] * ns
        options = [None] * ng
        pos = 0
        while pos >= 0 and time.perf_counter() < end:
            if pos == ng:
                retain(reconcile(assignment, end, True))
                if best is not None:
                    break
                pos -= 1
                if pos >= 0:
                    loads[assignment[order[pos]]] -= gd[order[pos]]
                continue
            g = order[pos]
            if options[pos] is None:
                options[pos] = sorted((s for s in range(ns) if loads[s] + gd[g] <= effective[s] + eps), key=lambda s: -proxy[g][s])
            if options[pos]:
                s = options[pos].pop()
                assignment[g] = s
                loads[s] += gd[g]
                pos += 1
                if pos < ng:
                    options[pos] = None
            else:
                options[pos] = None
                pos -= 1
                if pos >= 0:
                    loads[assignment[order[pos]]] -= gd[order[pos]]

    # The search phase always starts from a complete feasible joint plan.
    assignment_end = min(end, start + budget * 0.37)
    if best is not None:
        retain(reconcile(best[1], assignment_end))
        current = best
        enum = itertools.product(range(ns), repeat=ng) if ng * math.log(max(1, ns)) <= math.log(max(1, budget * 1000)) else None
        seen = set()
        while time.perf_counter() < assignment_end:
            if enum is not None:
                a = next(enum, None)
                if a is None:
                    enum = None
                    continue
                a = list(a)
            else:
                a = current[1][:]
                if not ng or ns < 2:
                    break
                if ng > 1 and rng.random() < 0.35:
                    g, h = rng.sample(range(ng), 2)
                    a[g], a[h] = a[h], a[g]
                else:
                    g = rng.randrange(ng)
                    old = a[g]
                    s = rng.randrange(ns - 1)
                    a[g] = s + (s >= old)
            key = tuple(a)
            if key in seen:
                if time.perf_counter() - start > budget * 0.29:
                    break
                continue
            seen.add(key)
            state = reconcile(a, assignment_end)
            if state is not None:
                retain(state)
                temp = max(eps, best[0] * 0.008 * (1 - (time.perf_counter() - start) / budget))
                if state[0] <= current[0] or rng.random() < math.exp(max(-700.0, (current[0] - state[0]) / temp)):
                    current = state
            if rng.random() < 0.03:
                current = best

    def insert_nodes(s, rr, pending, total_other, stop, noise=0.0):
        rr = [r[:] for r in rr]
        loads = [sum(dem[i] for i in r) for r in rr]
        pending = list(pending)
        while pending:
            if time.perf_counter() >= stop:
                return None
            selected = None
            for i in pending:
                if time.perf_counter() >= stop:
                    return None
                placements = []
                for k, r in enumerate(rr):
                    if loads[k] + dem[i] > q2 + eps:
                        continue
                    best_insertion = (inf, -1)
                    for p in range(len(r) + 1):
                        a = r[p - 1] if p else -1
                        b = r[p] if p < len(r) else -1
                        delta = edge(s, a, i) + edge(s, i, b) - edge(s, a, b)
                        if delta < best_insertion[0]:
                            best_insertion = delta, p
                    placements.append((best_insertion[0], k, best_insertion[1]))
                if len(rr) < vmax[s] and len(rr) + total_other < t2:
                    placements.append((2 * ds[s][i], len(rr), 0))
                if not placements:
                    return None
                placements.sort()
                v, k, p = placements[0]
                regret = (placements[1][0] - v) if len(placements) > 1 else (1 + v + max(ds[s], default=0))
                score = regret + dem[i] / max(q2, 1.0) * (1 + v) + noise * rng.random() * (1 + v)
                if selected is None or score > selected[0]:
                    selected = (score, i, k, p)
            _, i, k, p = selected
            if k == len(rr):
                rr.append([i])
                loads.append(dem[i])
            else:
                rr[k].insert(p, i)
                loads[k] += dem[i]
            pending.remove(i)
        return rr

    def local_search(routes, stop):
        sat_order = [s for s in range(ns) if routes[s]]
        rng.shuffle(sat_order)
        for s in sat_order:
            routes[s] = [opt2(s, r, stop) for r in routes[s]]
        changed = True
        while changed and time.perf_counter() < stop:
            changed = False
            for s in sat_order:
                rr = routes[s]
                pairs = [(a, b) for a in range(len(rr)) for b in range(a + 1, len(rr))]
                rng.shuffle(pairs)
                for a, b in pairs:
                    ra, rb = rr[a], rr[b]
                    la, lb = sum(dem[i] for i in ra), sum(dem[i] for i in rb)
                    move, gain = None, -eps
                    # Cross-route single-customer relocation and swaps are both evaluated by true edge deltas.
                    for flip in (False, True):
                        aa, bb = (rb, ra) if flip else (ra, rb)
                        destload = la if flip else lb
                        for i, x in enumerate(aa):
                            if time.perf_counter() >= stop:
                                break
                            if destload + dem[x] > q2 + eps:
                                continue
                            p = aa[i - 1] if i else -1
                            n = aa[i + 1] if i + 1 < len(aa) else -1
                            removal = edge(s, p, n) - edge(s, p, x) - edge(s, x, n)
                            for j in range(len(bb) + 1):
                                u = bb[j - 1] if j else -1
                                v = bb[j] if j < len(bb) else -1
                                delta = removal + edge(s, u, x) + edge(s, x, v) - edge(s, u, v)
                                if delta < gain:
                                    gain, move = delta, ('rel', flip, i, j)
                    for i, x in enumerate(ra):
                        if time.perf_counter() >= stop:
                            break
                        p = ra[i - 1] if i else -1
                        n = ra[i + 1] if i + 1 < len(ra) else -1
                        for j, y in enumerate(rb):
                            if la - dem[x] + dem[y] > q2 + eps or lb - dem[y] + dem[x] > q2 + eps:
                                continue
                            u = rb[j - 1] if j else -1
                            v = rb[j + 1] if j + 1 < len(rb) else -1
                            delta = edge(s, p, y) + edge(s, y, n) + edge(s, u, x) + edge(s, x, v) - edge(s, p, x) - edge(s, x, n) - edge(s, u, y) - edge(s, y, v)
                            if delta < gain:
                                gain, move = delta, ('swap', False, i, j)
                    # Two routes exchange their tails, allowing whole-segment moves and route elimination.
                    pa, pb = [0.0], [0.0]
                    for x in ra:
                        pa.append(pa[-1] + dem[x])
                    for x in rb:
                        pb.append(pb[-1] + dem[x])
                    for i in range(len(ra) + 1):
                        if time.perf_counter() >= stop:
                            break
                        p = ra[i - 1] if i else -1
                        n = ra[i] if i < len(ra) else -1
                        for j in range(len(rb) + 1):
                            if pa[i] + lb - pb[j] > q2 + eps or pb[j] + la - pa[i] > q2 + eps:
                                continue
                            u = rb[j - 1] if j else -1
                            v = rb[j] if j < len(rb) else -1
                            delta = edge(s, p, v) + edge(s, u, n) - edge(s, p, n) - edge(s, u, v)
                            if delta < gain:
                                gain, move = delta, ('tail', False, i, j)
                    if move is not None:
                        kind, flip, i, j = move
                        if kind == 'rel':
                            aa, bb = (rb, ra) if flip else (ra, rb)
                            x = aa.pop(i)
                            bb.insert(j, x)
                        elif kind == 'swap':
                            ra[i], rb[j] = rb[j], ra[i]
                        else:
                            rr[a], rr[b] = ra[:i] + rb[j:], rb[:j] + ra[i:]
                        routes[s] = [opt2(s, r, stop) for r in rr if r]
                        changed = True
                        break
                    if time.perf_counter() >= stop:
                        break
                if changed or time.perf_counter() >= stop:
                    break

    def group_repair(state, stop, strength):
        assignment, routes, loads = state[1][:], clone(state[2]), state[3][:]
        count = min(ng, max(1, int(1 + rng.random() * strength / max(1.0, math.sqrt(nc / max(1, ng))))))
        g0 = rng.randrange(ng)
        related = sorted(range(ng), key=lambda g: (0 if g == g0 else 1, min(proxy[g]), rng.random()))
        if rng.random() < 0.5:
            removed = [g0] + rng.sample([g for g in range(ng) if g != g0], count - 1)
        else:
            removed = sorted(related, key=lambda g: (g != g0, assignment[g] != assignment[g0], rng.random()))[:count]
        selected = set(removed)
        for s in range(ns):
            routes[s] = [[i for i in r if belong[i] not in selected] for r in routes[s]]
            routes[s] = [r for r in routes[s] if r]
        for g in removed:
            loads[assignment[g]] -= gd[g]
        removed.sort(key=lambda g: -gd[g])
        for g in removed:
            options = []
            total = sum(map(len, routes))
            for s in range(ns):
                if loads[s] + gd[g] > effective[s] + eps:
                    continue
                rr = insert_nodes(s, routes[s], groups[g], total - len(routes[s]), stop, 0.05)
                if rr is None:
                    continue
                delta = sum(route_cost(s, r) for r in rr) - sum(route_cost(s, r) for r in routes[s]) + gd[g] * handling[s]
                # First-echelon costs also take part in group insertion choices, rather than being scored only at the end.
                trial = loads[:]
                trial[s] += gd[g]
                f = first_plan(trial, min(stop, time.perf_counter() + budget * 0.0006))
                delta += f[0]
                options.append((delta * (1 + 0.035 * rng.random()), s, rr))
                if time.perf_counter() >= stop:
                    return None
            if not options:
                return None
            _, s, rr = min(options, key=lambda x: x[0])
            routes[s] = rr
            assignment[g] = s
            loads[s] += gd[g]
        return assignment, routes, loads

    def customer_repair(state, stop, strength):
        routes = clone(state[2])
        sats = [s for s in range(ns) if routes[s]]
        if not sats:
            return state[1][:], routes, state[3][:]
        s = rng.choice(sats)
        nodes = [i for r in routes[s] for i in r]
        count = min(len(nodes), max(1, int(1 + rng.random() * strength)))
        mode = rng.random()
        if mode < 0.25:
            removed = rng.choice(routes[s])[:]
        elif mode < 0.75:
            x = rng.choice(nodes)
            removed = sorted(nodes, key=lambda i: cc(x, i) * (0.8 + 0.4 * rng.random()))[:count]
        else:
            scores = []
            for r in routes[s]:
                for p, i in enumerate(r):
                    a = r[p - 1] if p else -1
                    b = r[p + 1] if p + 1 < len(r) else -1
                    gain = edge(s, a, i) + edge(s, i, b) - edge(s, a, b)
                    scores.append((gain * (0.6 + rng.random()), i))
            scores.sort(reverse=True)
            removed = [i for _, i in scores[:count]]
        selected = set(removed)
        routes[s] = [[i for i in r if i not in selected] for r in routes[s]]
        routes[s] = [r for r in routes[s] if r]
        rr = insert_nodes(s, routes[s], removed, sum(len(routes[h]) for h in range(ns) if h != s), stop, 0.1)
        if rr is None:
            return None
        routes[s] = rr
        return state[1][:], routes, state[3][:]

    if best is not None:
        current = best
        last_improvement = time.perf_counter()
        epoch = max(budget / (2 + math.log1p(budget)), 0.001)
        restart_at = time.perf_counter() + epoch
        # The main search has no iteration cap; the perturbation size and local-search time both scale with the total time limit.
        while time.perf_counter() < end:
            now = time.perf_counter()
            remaining = end - now
            local_slice = min(remaining, budget * 0.007 * (1 + math.log1p(nc)) / (1 + math.log1p(budget)))
            stop = min(end, now + max(0.0001, local_slice))
            stagnation = min(2.0, (now - last_improvement) / epoch)
            strength = min(max(1, nc), (1 + math.sqrt(max(1, nc))) * (0.4 + 0.18 * math.log1p(budget) + stagnation * 0.25))
            if ng and ns > 1 and rng.random() < 0.32:
                candidate = group_repair(current, stop, strength)
            else:
                candidate = customer_repair(current, stop, strength)
            if candidate is not None:
                assignment, routes, loads = candidate
                local_search(routes, stop)
                first = first_plan(loads, min(end, stop))
                value = cost2(routes) + first[0] + sum(loads[s] * handling[s] for s in range(ns))
                state = (value, assignment, routes, loads, first)
                if retain(state):
                    last_improvement = time.perf_counter()
                phase = max(0.0, min(1.0, (time.perf_counter() - start) / budget))
                temp = max(eps, best[0] / max(1, nc) * 0.2 * (0.025 ** phase))
                if value <= current[0] or rng.random() < math.exp(max(-700.0, (current[0] - value) / temp)):
                    current = state
            if time.perf_counter() >= restart_at:
                current = best
                improved_first = first_plan(best[3], min(end, time.perf_counter() + budget * 0.015), True)
                if improved_first[0] + eps < best[4][0]:
                    retain((best[0] - best[4][0] + improved_first[0], best[1], best[2], best[3], improved_first))
                    current = best
                restart_at = time.perf_counter() + epoch

    # Even under an extremely tight limit, completing a joint feasible construction comes first; no placeholder solution missing customers is output.
    while best is None:
        retain(construct(0.5 + rng.random(), time.perf_counter() + max(0.001, budget * 0.02)))

    _, assignment, routes, loads, first = best
    first_routes = []
    for row in first[1]:
        mask = sum(1 << s for s in range(ns) if row[s] > eps)
        first_routes.append([depot] + [sid[s] for s in tour(mask)[1]] + [depot])
    return {
        'first_echelon_routes': first_routes,
        'first_echelon_deliveries': [
            {'satellite_id': sid[s], 'vehicle_id': k, 'quantity': float(first[1][k][s])}
            for k in range(t1) for s in range(ns)
        ],
        'second_echelon_routes': [[sid[s]] + [cid[i] for i in r] + [sid[s]] for s in range(ns) for r in routes[s]],
        'group_assignments': {f'group_{g}': sid[assignment[g]] for g in range(ng)},
        'customer_satellite_assignments': {f'customer_{cid[i]}': sid[assignment[belong[i]]] for i in range(nc)},
    }
