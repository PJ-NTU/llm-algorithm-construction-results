"""Anytime heuristic for tax-efficient corporate hierarchy design.

Why the incumbent is strong / where it leaks (3 lines):
1. The strongest incumbent is strong because it never trusts a decomposable
   surrogate: every trial regraft is scored with the exact bottom-up
   recurrence, so the Subtree() coupling inside codes 3/4 is always honoured.
2. It leaks throughput because each trial costs a full O(N) re-evaluation of
   the whole tree, and it leaks basin quality because its restarts are
   uniformly random trees -- the worst possible sampler of good basins.
3. The weakest one leaks correctness of search direction: it scores moves with
   a shortest-path / Prufer surrogate that ignores the credit limits entirely,
   so a large share of its "improving" moves do not improve the real objective.

Pushed further here:
- Exact recurrence kept, but a regraft costs O(depth) instead of O(N) by
  maintaining per-node aggregates (subtree profit, raw received, per-branch
  credited receipts) and bubbling only the two affected root paths.
- The credit limits are rewritten as caps -- a code 3 parent credits
  min(F_k(1-W), (1-F_rate)*SubtreeP_k) per branch, a code 4 parent caps the
  pooled total -- which makes the uncapped relaxation an exact max-product
  path problem.  Dijkstra on that relaxation gives both a valid upper bound
  and a family of very good seed trees (provably optimal whenever no cap
  binds).  Random-tree restarts are replaced by noise-perturbed Dijkstra
  trees, which sample good basins instead of hopeless ones.
"""

import math
import random
import time


def solve(instance, time_limit_s):
    t0 = time.time()
    try:
        tl = float(time_limit_s)
    except Exception:
        tl = 10.0
    if not (tl > 0.0) or tl != tl:
        tl = 10.0
    deadline = t0 + tl * 0.85

    # ------------------------------------------------------------------ data
    raw_c = instance.get("countries") or {}
    raw_w = instance.get("withholding") or {}
    try:
        n = int(instance.get("N"))
    except Exception:
        n = 0
    if n <= 0:
        n = len(raw_c)
    if n <= 0:
        return {"structure": {}, "net_dividends": {}, "foreign_tax_paid": {}}
    try:
        target = int(instance.get("target"))
    except Exception:
        target = 1
    if target < 1 or target > n:
        target = 1

    code = [1] * (n + 1)
    dom = [0.0] * (n + 1)
    frate = [0.0] * (n + 1)
    prof = [0.0] * (n + 1)
    for key, rec in raw_c.items():
        try:
            i = int(key)
        except Exception:
            continue
        if i < 1 or i > n:
            continue
        try:
            code[i] = int(rec[0])
            dom[i] = float(rec[1])
            frate[i] = float(rec[2])
            prof[i] = float(rec[3])
        except Exception:
            pass

    pw = [prof[i] * (1.0 - dom[i]) for i in range(n + 1)]
    omf = [1.0 - frate[i] for i in range(n + 1)]

    # mult[k][j] = 1 - W[k][j], the fraction of k's dividend that survives the
    # withholding tax on the k -> j edge.
    mult = [[1.0] * (n + 1) for _ in range(n + 1)]
    for key, row in raw_w.items():
        try:
            k = int(key)
        except Exception:
            continue
        if k < 1 or k > n or not isinstance(row, dict):
            continue
        mk = mult[k]
        for key2, wv in row.items():
            try:
                j = int(key2)
                w = float(wv)
            except Exception:
                continue
            if 1 <= j <= n:
                if w < 0.0:
                    w = 0.0
                elif w > 1.0:
                    w = 1.0
                mk[j] = 1.0 - w

    nodes = list(range(1, n + 1))
    nonroot = [i for i in nodes if i != target]

    def build_answer(parent_list):
        """Recompute everything with the checker's own formulas."""
        ch = [[] for _ in range(n + 1)]
        for i in nodes:
            if i != target:
                p = parent_list[i]
                if p < 1 or p > n or p == i:
                    p = target
                ch[p].append(i)
        order = []
        stack = [target]
        while stack:
            u = stack.pop()
            order.append(u)
            for c in ch[u]:
                stack.append(c)
        spx = [0.0] * (n + 1)
        fx = [0.0] * (n + 1)
        tx = [0.0] * (n + 1)
        for u in reversed(order):
            cu = code[u]
            om = omf[u]
            s = prof[u]
            r = 0.0
            t3 = 0.0
            for c in ch[u]:
                x = fx[c] * mult[c][u]
                r += x
                s += spx[c]
                if cu == 3:
                    d = x - om * spx[c]
                    if d > 0.0:
                        t3 += d
            spx[u] = s
            if cu == 1:
                t = 0.0
            elif cu == 2:
                t = frate[u] * r
            elif cu == 3:
                t = t3
            else:
                d = r - om * (s - prof[u])
                t = d if d > 0.0 else 0.0
            tx[u] = t
            fx[u] = pw[u] + r - t
        struct = {}
        divs = {}
        taxes = {}
        for i in nodes:
            struct[str(i)] = 0 if i == target else int(parent_list[i])
            divs[str(i)] = float(fx[i])
            taxes[str(i)] = float(tx[i])
        return {
            "structure": struct,
            "net_dividends": divs,
            "foreign_tax_paid": taxes,
        }, fx[target]

    if not nonroot:
        ans, _ = build_answer([0] * (n + 1))
        return ans

    rng = random.Random(0x5EED07 + n * 131 + target)

    # ------------------------------------------------- max-product seed trees
    def dijkstra_tree(g, noise):
        """Best-multiplier tree under an uncapped pass-through model.

        g[p] is the fraction of an arriving dividend that country p forwards.
        With noise == 0 this is exactly optimal whenever no credit limit binds,
        and its value is a valid upper bound on the true optimum.
        """
        val = [-1.0] * (n + 1)
        pa = [target] * (n + 1)
        done = [False] * (n + 1)
        val[target] = 1.0
        pa[target] = 0
        for _ in range(n):
            bi = -1
            bv = -1.0
            for v in nodes:
                if not done[v] and val[v] > bv:
                    bv = val[v]
                    bi = v
            if bi < 0 or bv <= 0.0:
                break
            done[bi] = True
            onward = g[bi] * bv
            if onward <= 0.0:
                continue
            for v in nodes:
                if not done[v]:
                    cand = mult[v][bi] * onward
                    if noise > 0.0:
                        cand *= math.exp(-noise * rng.random())
                    if cand > val[v]:
                        val[v] = cand
                        pa[v] = bi
        for v in nodes:
            if v != target and (pa[v] == v or pa[v] < 1 or pa[v] > n):
                pa[v] = target
        pa[target] = 0
        return pa, val

    g_pass = [1.0] * (n + 1)
    g_all = [1.0] * (n + 1)
    g_c3 = [1.0] * (n + 1)
    g_c4 = [1.0] * (n + 1)
    for i in nodes:
        if code[i] == 2:
            g_pass[i] = omf[i]
            g_all[i] = omf[i]
            g_c3[i] = omf[i]
            g_c4[i] = omf[i]
        elif code[i] == 3:
            g_all[i] = omf[i]
            g_c3[i] = omf[i]
        elif code[i] == 4:
            g_all[i] = omf[i]
            g_c4[i] = omf[i]

    seed_pa, seed_val = dijkstra_tree(g_pass, 0.0)
    upper_bound = 0.0
    for i in nodes:
        v = seed_val[i]
        if v > 0.0:
            upper_bound += pw[i] * v

    # ------------------------------------------------------------ live state
    par = [0] * (n + 1)
    kids = [set() for _ in range(n + 1)]
    sp = [0.0] * (n + 1)
    recv = [0.0] * (n + 1)
    cr3 = [0.0] * (n + 1)
    fval = [0.0] * (n + 1)

    def sync(parent_list):
        for i in range(n + 1):
            par[i] = parent_list[i]
            kids[i].clear()
        par[target] = 0
        for i in nonroot:
            kids[par[i]].add(i)
        order = []
        stack = [target]
        while stack:
            u = stack.pop()
            order.append(u)
            for c in kids[u]:
                stack.append(c)
        for u in reversed(order):
            cu = code[u]
            om = omf[u]
            s = prof[u]
            r = 0.0
            t3 = 0.0
            for c in kids[u]:
                x = fval[c] * mult[c][u]
                r += x
                s += sp[c]
                if cu == 3:
                    b = om * sp[c]
                    t3 += x if x < b else b
            sp[u] = s
            recv[u] = r
            cr3[u] = t3
            if cu == 1:
                fval[u] = pw[u] + r
            elif cu == 2:
                fval[u] = pw[u] + om * r
            elif cu == 3:
                fval[u] = pw[u] + t3
            else:
                cap = om * (s - prof[u])
                fval[u] = pw[u] + (r if r < cap else cap)
        return fval[target]

    def node_value(u):
        cu = code[u]
        if cu == 1:
            return pw[u] + recv[u]
        if cu == 2:
            return pw[u] + omf[u] * recv[u]
        if cu == 3:
            return pw[u] + cr3[u]
        cap = omf[u] * (sp[u] - prof[u])
        r = recv[u]
        return pw[u] + (r if r < cap else cap)

    def bubble(v, k, old_f, old_s, new_f, new_s):
        """Push a child's (F, subtree-profit) change up the root path."""
        while v != 0:
            m = mult[k][v]
            recv[v] += (new_f - old_f) * m
            if code[v] == 3:
                om = omf[v]
                a = new_f * m
                b = om * new_s
                nt = a if a < b else b
                a = old_f * m
                b = om * old_s
                ot = a if a < b else b
                cr3[v] += nt - ot
            prev_s = sp[v]
            prev_f = fval[v]
            sp[v] = prev_s + (new_s - old_s)
            nf = node_value(v)
            fval[v] = nf
            if nf == prev_f and sp[v] == prev_s:
                return
            k = v
            old_f = prev_f
            new_f = nf
            old_s = prev_s
            new_s = sp[v]
            v = par[v]

    def relocate(i, newp):
        """Move the whole subtree rooted at i under newp.  O(depth)."""
        fi = fval[i]
        si = sp[i]
        oldp = par[i]

        kids[oldp].discard(i)
        m = mult[i][oldp]
        recv[oldp] -= fi * m
        if code[oldp] == 3:
            om = omf[oldp]
            a = fi * m
            b = om * si
            cr3[oldp] -= a if a < b else b
        prev_s = sp[oldp]
        prev_f = fval[oldp]
        sp[oldp] = prev_s - si
        nf = node_value(oldp)
        fval[oldp] = nf
        bubble(par[oldp], oldp, prev_f, prev_s, nf, sp[oldp])

        par[i] = newp
        kids[newp].add(i)
        m = mult[i][newp]
        recv[newp] += fi * m
        if code[newp] == 3:
            om = omf[newp]
            a = fi * m
            b = om * si
            cr3[newp] += a if a < b else b
        prev_s = sp[newp]
        prev_f = fval[newp]
        sp[newp] = prev_s + si
        nf = node_value(newp)
        fval[newp] = nf
        bubble(par[newp], newp, prev_f, prev_s, nf, sp[newp])

    def is_ancestor_of(i, p):
        """True if p sits inside the subtree of i (so i -> p would cycle)."""
        v = p
        while v != 0:
            if v == i:
                return True
            v = par[v]
        return False

    # ------------------------------------------------------- candidate lists
    ncand = 14 if n > 90 else (20 if n > 40 else n)
    ncand2 = 8 if n > 90 else (12 if n > 40 else n)
    _, val_all = dijkstra_tree(g_all, 0.0)
    cand = [None] * (n + 1)
    for i in nonroot:
        scored = []
        scored2 = []
        mi = mult[i]
        for p in nodes:
            if p == i:
                continue
            v1 = val_all[p]
            b1 = seed_val[p]
            if b1 > 0.0:
                scored.append((mi[p] * g_pass[p] * b1, p))
            if v1 > 0.0:
                scored2.append((mi[p] * g_all[p] * v1, p))
        scored.sort(reverse=True)
        scored2.sort(reverse=True)
        seen = set()
        lst = []
        for _, p in scored[:ncand]:
            if p not in seen:
                seen.add(p)
                lst.append(p)
        for _, p in scored2[:ncand2]:
            if p not in seen:
                seen.add(p)
                lst.append(p)
        if target not in seen:
            lst.append(target)
        cand[i] = lst

    nextra = 3 if n > 60 else 5

    # ------------------------------------------------------------ heuristics
    moves = [0]

    def local_search(stop_at):
        """First-improvement prune-and-regraft until a local optimum."""
        base = fval[target]
        improved = True
        while improved:
            if time.time() >= stop_at:
                return
            improved = False
            rng.shuffle(nonroot)
            for i in nonroot:
                if time.time() >= stop_at:
                    return
                base = fval[target]
                eps = 1e-7 * (1.0 + (base if base > 0.0 else -base))
                oldp = par[i]
                cl = cand[i]
                ln = len(cl)
                start = rng.randrange(ln)
                hit = False
                for t in range(ln + nextra):
                    if t < ln:
                        p = cl[(start + t) % ln]
                    else:
                        p = rng.randint(1, n)
                    if p == i or p == oldp:
                        continue
                    if is_ancestor_of(i, p):
                        continue
                    relocate(i, p)
                    moves[0] += 2
                    if fval[target] > base + eps:
                        hit = True
                        improved = True
                        break
                    relocate(i, oldp)
                if hit:
                    continue

    def lift_pass(stop_at):
        """Move a single country out of its position, re-hanging its children.

        Complements the subtree regraft: it dissolves an intermediate holding
        company without dragging its branch along.
        """
        gain = False
        order = list(nonroot)
        rng.shuffle(order)
        for i in order:
            if time.time() >= stop_at:
                return gain
            ch = kids[i]
            if not ch or len(ch) > 6:
                continue
            base = fval[target]
            eps = 1e-7 * (1.0 + (base if base > 0.0 else -base))
            oldp = par[i]
            children = list(ch)
            for c in children:
                relocate(c, oldp)
            cl = cand[i]
            ln = len(cl)
            start = rng.randrange(ln)
            hit = False
            for t in range(ln):
                p = cl[(start + t) % ln]
                if p == i or p == oldp:
                    continue
                if is_ancestor_of(i, p):
                    continue
                relocate(i, p)
                moves[0] += 2
                if fval[target] > base + eps:
                    hit = True
                    gain = True
                    break
                relocate(i, oldp)
            if not hit:
                for c in children:
                    relocate(c, i)
        return gain

    def random_tree():
        pa = [0] * (n + 1)
        pool = [target]
        order = list(nonroot)
        rng.shuffle(order)
        for v in order:
            pa[v] = pool[rng.randrange(len(pool))]
            pool.append(v)
        pa[target] = 0
        return pa

    def kick(strength):
        for _ in range(strength):
            i = nonroot[rng.randrange(len(nonroot))]
            if rng.random() < 0.55:
                cl = cand[i]
                p = cl[rng.randrange(len(cl))]
            else:
                p = rng.randint(1, n)
            if p == i or p == par[i]:
                continue
            if is_ancestor_of(i, p):
                continue
            relocate(i, p)
            moves[0] += 1

    # ----------------------------------------------------------- seed search
    best_par = None
    best_val = -1e300

    star = [target] * (n + 1)
    star[target] = 0
    for cand_tree in (seed_pa, star,
                      dijkstra_tree(g_all, 0.0)[0],
                      dijkstra_tree(g_c3, 0.0)[0],
                      dijkstra_tree(g_c4, 0.0)[0]):
        v = sync(cand_tree)
        if v > best_val:
            best_val = v
            best_par = par[:]

    tol = 1e-7 * (1.0 + (upper_bound if upper_bound > 0.0 else -upper_bound))
    if best_val >= upper_bound - tol:
        ans, _ = build_answer(best_par)
        return ans

    sync(best_par)

    # --------------------------------------------------------- ILS main loop
    restarts = 0
    stagnation = 0
    restart_after = 10 if n > 60 else 14
    resync_every = 3000

    while True:
        now = time.time()
        if now >= deadline:
            break
        local_search(deadline)
        if fval[target] <= best_val:
            # a stalled descent still deserves the cheap second neighbourhood
            if time.time() < deadline and lift_pass(deadline):
                local_search(deadline)
        cur = fval[target]
        if cur > best_val + 1e-9:
            best_val = cur
            best_par = par[:]
            stagnation = 0
        else:
            stagnation += 1
        if time.time() >= deadline:
            break

        if moves[0] > resync_every:
            moves[0] = 0
            sync(par[:])

        if stagnation >= restart_after:
            stagnation = 0
            restarts += 1
            r = restarts % 4
            if r == 3:
                sync(random_tree())
            else:
                noise = 0.05 + 0.30 * rng.random()
                base_g = g_pass if r == 0 else (g_all if r == 1 else g_c3)
                sync(dijkstra_tree(base_g, noise)[0])
        else:
            sync(best_par)
            lo = 2
            hi = 3 + (n // 12)
            if hi > 14:
                hi = 14
            kick(rng.randint(lo, hi))

    if best_par is None:
        best_par = star
    ans, final_obj = build_answer(best_par)
    return ans
