"""Converter for vehicle_routing_period_routing, for the portfolio updated by continued evolution (its format differs from the original).

Official: depot, customers [{id, x, y, demand, schedules}], vehicles_per_day, vehicle_capacity, period_length;
solution selected_schedules {customer id: chosen 0/1 schedule vector}, tours {day 1..T: [[0, id, ..., id, 0], ...]}.
Ours (input_format.json / output_format.json in the problem folder): node 0 = depot, node p + 1 = customers[p];
candidate_schedules[p] = global indices into schedule_vectors; num_vehicles K for every day; solution
status, x[p][r] (choice of candidate r of customer p), y[i][j][h][t] (vehicle h drives i -> j on day t, 0-based),
u (cumulative loads).

Input: the original entry produces exactly these fields (plus the unused _cust_ids) and is reused, except that
a fleet varying by day cannot be expressed by the single K: the original entry took K = min(vehicles_per_day),
which makes prvp1 ([2, 3], total demand 777 > 4 x 160) infeasible, so such an instance raises instead.
Output: the 1 in x[p] picks schedule_vectors[candidate_schedules[p][r]] for customer id customers[p]['id'];
the arcs of vehicle h on day t are followed from the depot back to the depot, giving one tour of day t + 1.
u has no official counterpart and is not passed on (the official eval_func needs only schedules and tours).
A solution whose x / y cannot be read this way raises instead of being repaired: status not "feasible",
wrong shape, a value other than 0/1, a customer without exactly one chosen schedule, or arcs of one vehicle-day
that are not a single walk from the depot back to it (two arcs leaving a node, a dead end, arcs left over).
"""
from adapters import _lst, _vrp_to_lib


def to_lib(off):
    if len(set(off["vehicles_per_day"])) > 1:
        raise ValueError("not representable in the evolved format: vehicles_per_day %s" % (off["vehicles_per_day"],))
    return _vrp_to_lib(off)


def to_off(sol, off, lib):
    if sol.get("status") != "feasible":
        raise ValueError("solution reports no plan (status=%r)" % (sol.get("status"),))
    ids = [int(c["id"]) for c in off["customers"]]
    n, k, days = len(ids), lib["num_vehicles"], lib["planning_period"]
    cand, vecs = lib["candidate_schedules"], lib["schedule_vectors"]

    x = _lst(sol.get("x"))
    if len(x) != n:
        raise ValueError("x has %d rows, expected %d customers" % (len(x), n))
    selected = {}
    for p, row in enumerate(x):
        row = _lst(row)
        if len(row) != len(cand[p]) or any(v not in (0, 1) for v in row) or sum(v == 1 for v in row) != 1:
            raise ValueError("x[%d] = %r is not one choice among %d candidates" % (p, row, len(cand[p])))
        selected[ids[p]] = list(vecs[cand[p][[v == 1 for v in row].index(True)]])

    y = sol.get("y")
    if (len(y) != n + 1 or any(len(r) != n + 1 for r in y)
            or any(len(c) != k for r in y for c in r) or any(len(d) != days for r in y for c in r for d in c)):
        raise ValueError("y is not %d x %d x %d x %d" % (n + 1, n + 1, k, days))
    tours = {}
    for t in range(days):
        tours[t + 1] = []
        for h in range(k):
            succ = {}
            for i in range(n + 1):
                for j in range(n + 1):
                    v = y[i][j][h][t]
                    if v not in (0, 1):
                        raise ValueError("y[%d][%d][%d][%d] = %r is not 0/1" % (i, j, h, t, v))
                    if v == 1:
                        if i in succ:
                            raise ValueError("vehicle %d day %d: two arcs leave node %d" % (h, t, i))
                        succ[i] = j
            if not succ:
                continue
            if 0 not in succ:
                raise ValueError("vehicle %d day %d: arcs used but none leaves the depot" % (h, t))
            walk, cur = [0], succ.pop(0)
            while cur != 0:
                if cur not in succ:
                    raise ValueError("vehicle %d day %d: the walk from the depot stops at node %d" % (h, t, cur))
                walk.append(cur)
                cur = succ.pop(cur)
            if succ:
                raise ValueError("vehicle %d day %d: arcs not on the depot walk: %s" % (h, t, sorted(succ.items())[:5]))
            tours[t + 1].append([0] + [ids[c - 1] for c in walk[1:]] + [0])
    return {"selected_schedules": selected, "tours": tours}
