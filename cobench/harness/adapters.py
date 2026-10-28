"""Pure format translation between official instance/solution ↔ library instance/solution.

Iron rule (2026-07-06 CO-Bench hard-learned lesson): only field moving/renaming/re-basing/structural reordering is allowed, no algorithm logic;
shape validation must go directly through the official eval_func — even with the right shape, the semantics may still be wrong.

One entry per problem:
    "to_lib": off_inst -> lib_inst
    "to_off": (lib_sol, off_inst, lib_inst) -> off_sol
Defaults to identity.
"""
import math


def _id_inst(off):
    return off


def _id_sol(sol, off, lib):
    return sol


def _g(sol, *keys):
    """From the library solution, take the first existing field among the candidate names."""
    if not isinstance(sol, dict):
        return None
    for k in keys:
        if k in sol and sol[k] is not None:
            return sol[k]
    return None


def _lst(x):
    if x is None:
        return []
    return list(x.tolist() if hasattr(x, "tolist") else x)


# ═══════════════ travelling_salesman_problem ═══════════════
def _tsp_to_lib(off):
    """Build row by row, avoiding an n×n×2 intermediate array (tsp10000 would eat 1.6GB)."""
    import numpy as np
    a = np.asarray(off["nodes"], dtype=np.float64)
    n = len(a)
    d = np.empty((n, n), dtype=np.float32)
    for i in range(n):
        d[i] = np.sqrt(((a - a[i]) ** 2).sum(1))
    return {"num_nodes": int(n), "distance_matrix": d}


def _tsp_to_off(sol, off, lib):
    return {"tour": [int(v) for v in _lst(_g(sol, "tour"))]}


# ═══════════════ assignment_problem ═══════════════
def _ap_to_lib(off):
    cm = off["cost_matrix"]
    return {"n": int(off["n"]), "cost": cm.tolist() if hasattr(cm, "tolist") else cm}


def _ap_to_off(sol, off, lib):
    a = _lst(_g(sol, "assignment", "assignments"))
    return {"assignment": [(i + 1, int(j) + 1) for i, j in enumerate(a)]}


# ═══════════════ common_due_date_scheduling ═══════════════
def _cdd_to_lib(off):
    jobs = off["jobs"]
    h = float(off.get("h", 0.6))
    p = [j[0] for j in jobs]
    return {"n": len(jobs), "processing_times": p,
            "earliness_penalties": [j[1] for j in jobs],
            "tardiness_penalties": [j[2] for j in jobs],
            "due_date": int(sum(p) * h)}


def _cdd_to_off(sol, off, lib):
    return {"schedule": [int(v) + 1 for v in _lst(_g(sol, "order", "schedule", "sequence"))]}


# ═══════════════ bin_packing_one_dimensional ═══════════════
def _bp_to_lib(off):
    """The library checker strictly requires capacity to be int (official gives float); without conversion all 1600 cells are judged as violations."""
    cap = off["bin_capacity"]
    return {"n": int(off["num_items"]),
            "capacity": int(cap) if float(cap).is_integer() else cap,
            "weights": [int(w) if float(w).is_integer() else w for w in _lst(off["items"])]}


def _bp_to_off(sol, off, lib):
    asg = _lst(_g(sol, "item_assignments", "assignments"))
    bins = {}
    for i, b in enumerate(asg):
        bins.setdefault(int(b), []).append(i + 1)
    lst = [v for _, v in sorted(bins.items())]
    return {"num_bins": len(lst), "bins": lst}


# ═══════════════ flow_shop_scheduling ═══════════════
def _fs_to_lib(off):
    return {"n": int(off["n"]), "n_machines": int(off["m"]), "proc": off["matrix"]}


def _fs_to_off(sol, off, lib):
    return {"job_sequence": [int(v) + 1 for v in _lst(_g(sol, "perm", "job_sequence", "order"))]}


# ═══════════════ multidimensional_knapsack_problem ═══════════════
def _mkp_to_lib(off):
    return {"n": int(off["n"]), "m": int(off["m"]), "profit": _lst(off["p"]),
            "weights": [_lst(r) for r in off["r"]], "capacity": _lst(off["b"])}


def _mkp_to_off(sol, off, lib):
    return {"x": [int(v) for v in _lst(_g(sol, "selection_status", "x", "chosen"))]}


# ═══════════════ multi_demand_multidimensional_knapsack_problem ═══════════════
def _mdmkp_to_off(sol, off, lib):
    return {"x": [int(v) for v in _lst(_g(sol, "chosen", "x", "selection_status"))]}


# ═══════════════ set_covering ═══════════════
def _sc_to_lib(off):
    return {"m": int(off["m"]), "n": int(off["n"]), "costs": _lst(off["costs"]),
            "covers": [[int(c) - 1 for c in row] for row in off["row_cover"]]}


def _sc_to_off(sol, off, lib):
    ss = _lst(_g(sol, "selection_status", "selected_columns", "selection"))
    return {"selected_columns": [i + 1 for i, v in enumerate(ss) if v]}


# ═══════════════ p_median_uncapacitated ═══════════════
def _pmu_to_lib(off):
    return {"n": int(off["n"]), "p": int(off["p"]), "dist": off["dist"]}


def _pmu_to_off(sol, off, lib):
    n, p = int(off["n"]), int(off["p"])
    med = [int(v) for v in _lst(_g(sol, "selected_medians", "medians", "selected_facilities"))]
    if len(med) == n and set(med) <= {0, 1}:
        med = [i for i, v in enumerate(med) if v]        # 0/1 indicator vector → indices
    if med and min(med) == 0:
        med = [v + 1 for v in med]                       # 0-based → official 1..n
    return {"medians": med}


# ═══════════════ corporate_structuring ═══════════════
def _cs_to_lib(off):
    """Library algorithms fetch countries/withholding by string key (the form after library JSON serialization)."""
    co = {str(k): list(v) for k, v in off["countries"].items()}
    wh = {str(i): {str(j): v for j, v in row.items()} for i, row in off["withholding"].items()}
    return {"N": int(off["N"]), "target": int(off["target"]), "countries": co, "withholding": wh}


def _cs_to_off(sol, off, lib):
    st = _g(sol, "structure") or {}
    return {"structure": {int(k): int(v) for k, v in st.items()}}


# ═══════════════ graph_colouring ═══════════════
def _gc_to_lib(off):
    """Official vertices are 1..n; the library stores edges 0-based."""
    return {"n": int(off["n"]), "edges": [[int(u) - 1, int(v) - 1] for u, v in off["edges"]]}


def _gc_to_off(sol, off, lib):
    """The official side unpacks the solution dict directly into kwargs; keys are vertex numbers (1..n), values are colors."""
    col = [int(c) for c in _lst(_g(sol, "coloring", "colors", "colouring"))]
    if col and min(col) == 0:
        col = [c + 1 for c in col]          # Official requires positive integer colors
    return {i + 1: c for i, c in enumerate(col)}


# ═══════════════ maximal_independent_set ═══════════════
def _mis_to_lib(off):
    g = off["graph"]
    return {"num_vertices": int(g.number_of_nodes()),
            "edges": [[int(u), int(v)] for u, v in g.edges()]}


def _mis_to_off(sol, off, lib):
    sel = _lst(_g(sol, "vertex_selection", "mis_nodes", "selection"))
    if sel and set(map(int, sel)) <= {0, 1} and len(sel) == lib["num_vertices"]:
        nodes = [i for i, v in enumerate(sel) if v]      # 0/1 indicator vector
    else:
        nodes = [int(v) for v in sel]                    # Vertex ids given directly
    return {"mis_nodes": nodes}


# ═══════════════ equitable_partitioning_problem ═══════════════
def _eq_to_lib(off):
    data = off["data"]
    mat = [list(r) for r in data]
    return {"num_individuals": len(mat), "num_attributes": len(mat[0]) if mat else 0,
            "num_groups": 8, "attribute_matrix": mat}


def _eq_to_off(sol, off, lib):
    """Official requires assignment = the list of group numbers of each individual (positive integers, exactly 8 groups)."""
    n = lib["num_individuals"]
    ga = _g(sol, "group_assignment", "assignment")
    out = [1] * n
    if isinstance(ga, dict):
        for k, v in ga.items():
            i = int(str(k).split("_")[-1])
            if 0 <= i < n:
                out[i] = int(v)
    else:
        for i, v in enumerate(_lst(ga)[:n]):
            out[i] = int(v)
    if min(out) == 0:
        out = [v + 1 for v in out]
    return {"assignment": out}


# ═══════════════ euclidean_steiner_problem ═══════════════
def _est_to_lib(off):
    pts = [list(p) for p in off["points"]]
    return {"points": pts, "max_steiner_points": max(0, len(pts) - 2)}


def _est_to_off(sol, off, lib):
    sp = _g(sol, "steiner_points") or []
    return {"steiner_points": [list(p) for p in sp]}


# ═══════════════ job_shop_scheduling ═══════════════
def _js_to_lib(off):
    t, m = off["times"], off["machines"]
    jobs = [[[int(m[j][k]), t[j][k]] for k in range(len(t[j]))] for j in range(len(t))]
    return {"n": int(off["n_jobs"]), "n_jobs": int(off["n_jobs"]),
            "n_machines": int(off["n_machines"]), "jobs": jobs}


def _js_to_off(sol, off, lib):
    nm = int(off["n_machines"])
    st = _g(sol, "start_times", "start") or []
    out = []
    for row in st:
        r = [int(round(v)) for v in row][:nm]
        r += [0] * (nm - len(r))
        out.append(r)
    return {"start_times": out}


# ═══════════════ open_shop_scheduling ═══════════════
def _os_to_lib(off):
    """Official times[j][k] is the duration of the k-th operation, machines[j][k] is the machine it uses;
    library ptimes is indexed by [job][machine], so reorder according to machines."""
    t, m = off["times"], off["machines"]
    nm = int(off["n_machines"])
    base = min(int(x) for r in m for x in r)      # Machine index base (0 or 1)
    pt = []
    for j in range(len(t)):
        row = [0] * nm
        for k in range(len(t[j])):
            row[int(m[j][k]) - base] = t[j][k]
        pt.append(row)
    return {"n": int(off["n_jobs"]), "n_jobs": int(off["n_jobs"]), "n_machines": nm, "ptimes": pt}


def _os_to_off(sol, off, lib):
    """Library start[job][machine] → official start_times[job][k] (k is the operation index)."""
    st = _g(sol, "start", "start_times") or []
    m = off["machines"]
    base = min(int(x) for r in m for x in r)
    out = []
    for j in range(len(m)):
        row = []
        for k in range(len(m[j])):
            mid = int(m[j][k]) - base
            try:
                row.append(int(round(st[j][mid])))
            except Exception:
                row.append(0)
        out.append(row)
    return {"start_times": out}


# ═══════════════ hybrid_reentrant_shop_scheduling ═══════════════
def _hr_to_lib(off):
    n = int(off["n_jobs"])
    it = off["init_time"]
    if not isinstance(it, (list, tuple)):
        it = [it] * n                         # Official gives a scalar; library stores one per job
    return {"num_jobs": n, "num_machines": int(off["n_machines"]),
            "init_processing_times": _lst(it),
            "setup_processing_times": off["setup_times"],
            "main_processing_times": off["processing_times"]}


def _hr_to_off(sol, off, lib):
    """Official requires permutation (job order) + batch_assignment (which machine each job is assigned to)."""
    n = int(off["n_jobs"])
    seq = _lst(_g(sol, "server_sequence", "permutation", "sequence"))
    perm = []
    for s in seq:
        try:
            perm.append(int(str(s).split("_")[-1]))
        except Exception:
            pass
    if sorted(perm) != list(range(n)):
        perm = list(range(n))
    perm = [v + 1 for v in perm]              # Official expects 1..n_jobs
    ma = _g(sol, "machine_assignment", "batch_assignment") or {}
    batch = []
    for j in range(n):
        v = ma.get(f"job_{j}", ma.get(str(j), ma.get(j, 0))) if isinstance(ma, dict) else (
            ma[j] if j < len(ma) else 0)
        try:
            batch.append(int(v))
        except Exception:
            batch.append(0)
    return {"permutation": perm, "batch_assignment": batch}


# ═══════════════ generalised_assignment_problem ═══════════════
def _gap_to_lib(off):
    return {"num_machines": int(off["m"]), "num_jobs": int(off["n"]),
            "cost_matrix": off["cost_matrix"], "resource_matrix": off["consumption_matrix"],
            "capacities": _lst(off["capacities"])}


def _gap_to_off(sol, off, lib):
    a = _g(sol, "assignments", "assignment") or {}
    n = int(off["n"])
    if isinstance(a, dict):
        out = []
        for j in range(n):
            v = a.get(f"job_{j}", a.get(str(j), a.get(j, 0)))
            out.append(int(v))
    else:
        out = [int(v) for v in _lst(a)]
    if out and min(out) == 0:
        out = [v + 1 for v in out]            # Library 0-based → official 1..m
    return {"assignments": out}


# ═══════════════ uncapacitated_warehouse_location ═══════════════
def _uwl_to_lib(off):
    wh, cu = off["warehouses"], off["customers"]
    fixed = [w["fixed_cost"] if isinstance(w, dict) else w[0] for w in wh]
    alloc = []
    for c in cu:
        row = c["costs"] if isinstance(c, dict) else c[1]
        alloc.append(_lst(row))
    alloc = [list(col) for col in zip(*alloc)]           # facility x customer, as in the problem format
    return {"n": int(off["n"]), "m_facilities": int(off["m"]),
            "fixed_cost": fixed, "alloc_cost": alloc}


def _uwl_to_off(sol, off, lib):
    m = int(off["m"])
    op = [int(v) for v in _lst(_g(sol, "facility_open_status", "warehouse_open", "open_plants"))]
    asg = _lst(_g(sol, "customer_assignments", "assignments"))
    rows = []
    for a in asg:
        if isinstance(a, (list, tuple)):
            rows.append([float(x) for x in a])          # Already an m-dimensional assignment vector
        else:
            r = [0.0] * m
            j = int(a)
            if 0 <= j < m:
                r[j] = 1.0
            rows.append(r)
    return {"warehouse_open": op, "assignments": rows}



# ═══════════════ packing_unequal_circles (count version) ═══════════════
def _puc_to_off(sol, off, lib):
    """Official requires coords: n pairs (x,y), with (-1,-1) for those not packed."""
    n = int(off["n"])
    st = _lst(_g(sol, "packing_status", "selection_status", "item_selection"))
    ctr = _g(sol, "centers")
    xs = _lst(_g(sol, "center_x_coordinates"))
    ys = _lst(_g(sol, "center_y_coordinates"))
    out = []
    for i in range(n):
        packed = bool(st[i]) if i < len(st) else False
        if not packed:
            out.append((-1.0, -1.0)); continue
        if ctr is not None and i < len(ctr):
            p = ctr[i]
            out.append((float(p[0]), float(p[1])))
        elif i < len(xs) and i < len(ys):
            out.append((float(xs[i]), float(ys[i])))
        else:
            out.append((-1.0, -1.0))
    return {"coords": out}


# ═══════════════ packing_unequal_rectangles_and_squares (count version) ═══════════════
def _purs_to_lib(off):
    items = off["items"]
    R = float(off["R"])
    return {"num_rectangles": int(off["n"]),
            "widths": [float(it[0]) for it in items],
            "heights": [float(it[1]) for it in items],
            "container_radius": R,
            "big_m": 4.0 * R}


def _purs_to_off(sol, off, lib):
    """Official requires placements: n triples (x, y, theta), (-1, -1, 0) for those not packed."""
    n = int(off["n"])
    st = _lst(_g(sol, "selection_status", "item_selection", "packing_status"))
    xs = _lst(_g(sol, "center_x_coordinates"))
    ys = _lst(_g(sol, "center_y_coordinates"))
    rf = _lst(_g(sol, "rotation_flags", "rotation_status"))
    out = []
    for i in range(n):
        packed = bool(st[i]) if i < len(st) else False
        if not packed or i >= len(xs) or i >= len(ys):
            out.append((-1.0, -1.0, 0)); continue
        th = 90 if (i < len(rf) and int(rf[i]) == 1) else 0
        # _purs_to_lib centres the container at (0, 0); the official data centre it at (cx, cy). Problems that
        # pass the official instance through unchanged already use the official frame.
        dx, dy = (0.0, 0.0) if "cx" in lib else (float(off.get("cx", 0.0)), float(off.get("cy", 0.0)))
        out.append((float(xs[i]) + dx, float(ys[i]) + dy, th))
    return {"placements": out}


# ═══════════════ set_partitioning ═══════════════
def _sp_to_lib(off):
    """Official columns_info: {col(1-indexed): (cost, {rows})} → library stores costs/covers by column."""
    ci = off["columns_info"]
    nc = int(off["num_columns"])
    costs, covers = [], []
    for j in range(1, nc + 1):
        c, rows = ci[j]
        costs.append(c)
        covers.append(sorted(int(r) - 1 for r in rows))     # 1-indexed rows → 0-based
    return {"n": nc, "m": int(off["num_rows"]), "costs": costs, "covers": covers}


def _sp_to_off(sol, off, lib):
    sel = _lst(_g(sol, "selected_plans", "selection_status", "selected_columns"))
    nc = int(off["num_columns"])
    if len(sel) == nc and set(int(v) for v in sel) <= {0, 1}:
        cols = [i + 1 for i, v in enumerate(sel) if v]       # 0/1 indicator vector
    else:
        cols = sorted({int(v) + 1 for v in sel})             # Column indices given directly (0-based)
    return {"selected_columns": sorted(cols)}


# ═══════════════ p_median_capacitated ═══════════════
def _pmc_to_lib(off):
    """Candidate points = all customer points; distances are Euclidean (same convention as the official eval_func)."""
    cu = off["customers"]
    n = len(cu)
    xs = [float(c[1]) for c in cu]
    ys = [float(c[2]) for c in cu]
    dm = [[math.hypot(xs[i] - xs[j], ys[i] - ys[j]) for j in range(n)] for i in range(n)]
    Q = float(off["Q"])
    return {"num_communities": n, "num_candidates": n, "distance_matrix": dm,
            "capacity_per_center": [Q] * n,
            "demands": [float(c[3]) for c in cu], "p": int(off["p"])}


def _pmc_to_off(sol, off, lib):
    """The library uses 0-based candidate indices; official requires customer_id."""
    cu = off["customers"]
    ids = [int(c[0]) for c in cu]
    n = len(ids)

    def cid(v):
        v = int(v)
        return ids[v] if 0 <= v < n else v

    raw = [int(v) for v in _lst(_g(sol, "selected_facilities", "selected_medians", "medians"))]
    if len(raw) == n and set(raw) <= {0, 1}:
        raw = [i for i, v in enumerate(raw) if v]        # 0/1 indicator vector → indices
    med = [cid(v) for v in raw]
    a = _g(sol, "assignments", "assignment")
    asg = [0] * n
    if isinstance(a, dict):
        for k, v in a.items():
            try:
                i = int(str(k).split("_")[-1])
            except Exception:
                continue
            if 0 <= i < n:
                asg[i] = cid(v)
    else:
        for i, v in enumerate(_lst(a)[:n]):
            asg[i] = cid(v)
    return {"medians": med, "assignments": asg, "objective": 0.0}


# ═══════════════ aircraft_landing ═══════════════
def _al_to_lib(off):
    pl = off["planes"]
    R = int(off["num_runways"])
    sep2 = off["separation"]
    n = len(pl)
    # Library separation_matrix has shape [i][j][runway_i][runway_j]: no separation required across runways
    sep4 = [[[[float(sep2[i][j]) if r1 == r2 else 0.0 for r2 in range(R)]
              for r1 in range(R)] for j in range(n)] for i in range(n)]
    return {"planes": list(range(n)), "runways": list(range(R)),
            "earliest_landing_time": [p["earliest"] for p in pl],
            "target_landing_time": [p["target"] for p in pl],
            "latest_landing_time": [p["latest"] for p in pl],
            "earliness_penalty_cost": [p["penalty_early"] for p in pl],
            "lateness_penalty_cost": [p["penalty_late"] for p in pl],
            "separation_matrix": sep4,
            "big_m": float(max(p["latest"] for p in pl)) * 2.0 if pl else 1e6,
            "freeze_time": off.get("freeze_time", 0)}


def _al_to_off(sol, off, lib):
    """Official schedule: {plane_id(1-indexed): {landing_time, runway(1..R)}}."""
    n = len(off["planes"])
    lt = _g(sol, "landing_times") or {}
    ra = _g(sol, "runway_assignment") or {}

    def pick(d, i):
        if isinstance(d, dict):
            for k in (f"plane_{i}", str(i), i, f"P{i}"):
                if k in d:
                    return d[k]
            return None
        return d[i] if i < len(d) else None

    sched = {}
    for i in range(n):
        t = pick(lt, i)
        r = pick(ra, i)
        sched[i + 1] = {"landing_time": float(t) if t is not None else float(off["planes"][i]["target"]),
                        "runway": (int(r) + 1) if r is not None else 1}
    return {"schedule": sched}


# ═══════════════ resource_constrained_shortest_path ═══════════════
def _rcsp_to_lib(off):
    g = off["graph"]
    n = int(off["n"]); K = int(off["K"])
    arcs, costs, arcres = [], [], []
    for u, lst in g.items():
        for a in lst:
            v, c = a[0], a[1]
            r = a[2] if len(a) > 2 else [0.0] * K
            arcs.append([int(u) - 1, int(v) - 1])
            costs.append(float(c))
            arcres.append([float(x) for x in r])
    vr = [[float(x) for x in row] for row in off["vertex_resources"]]
    lb, ub = [float(x) for x in off["lower_bounds"]], [float(x) for x in off["upper_bounds"]]
    return {"n": n, "resource_types": K, "arcs": arcs, "arc_costs": costs,
            "vertex_resource_consumption": vr, "arc_resource_consumption": arcres,
            "lower_bounds": [lb] * n, "upper_bounds": [ub] * n}


def _rcsp_to_off(sol, off, lib):
    path = [int(v) + 1 for v in _lst(_g(sol, "path"))]      # 0-based → 1-indexed
    return {"path": path, "total_cost": 0.0}


def _scale(row, demand):
    """The library may give proportions (summing to 1) or amounts (summing to demand); official requires amounts, so normalize by the sum and multiply by demand."""
    tot = sum(row)
    if tot <= 0:
        return row
    return [v * demand / tot for v in row]


# ═══════════════ capacitated_warehouse_location ═══════════════
def _cwl_to_lib(off):
    wh, cu = off["warehouses"], off["customers"]
    ship = [_lst(c["costs"]) for c in cu]
    n, m = len(cu), len(wh)
    return {"num_customers": n, "num_plants": m,
            "customers_demands": [c["demand"] for c in cu],
            "plants_capacities": [w["capacity"] for w in wh],
            "plants_fixed_costs": [w["fixed_cost"] for w in wh],
            "shipping_costs": ship,
            "customers_x_coords": [0.0] * n, "customers_y_coords": [0.0] * n,
            "plants_x_coords": [0.0] * m, "plants_y_coords": [0.0] * m,
            "total_demand": sum(c["demand"] for c in cu),
            "total_capacity": sum(w["capacity"] for w in wh),
            "capacity_tightness_ratio": 1}


def _cwl_to_off(sol, off, lib):
    m, n = int(off["m"]), int(off["n"])
    op = [int(v) for v in _lst(_g(sol, "open_plants", "warehouse_open", "facility_open_status"))]
    if op and len(op) != m and max(op) < m:
        v = [0] * m
        for j in op:
            v[j] = 1
        op = v
    dem = [float(c["demand"]) for c in off["customers"]]
    a = _g(sol, "assignments", "assignment")
    rows = []
    if isinstance(a, dict):
        for i in range(n):
            r = [0.0] * m
            d = None
            for k in (f"customer_{i}", str(i), i):
                if k in a:
                    d = a[k]; break
            if isinstance(d, dict):
                for pk, pv in d.items():
                    try:
                        j = int(str(pk).split("_")[-1])
                    except Exception:
                        continue
                    if 0 <= j < m:
                        r[j] = float(pv)
            elif d is not None:
                j = int(d)
                if 0 <= j < m:
                    r[j] = dem[i]
            rows.append(_scale(r, dem[i]))
    else:
        for i, x in enumerate(_lst(a)[:n]):
            r = [0.0] * m
            if isinstance(x, (list, tuple)):
                r = [float(v) for v in x]
            else:
                j = int(x)
                if 0 <= j < m:
                    r[j] = dem[i]
            rows.append(_scale(r, dem[i]))
    return {"warehouse_open": op, "assignments": rows}



# ═══════════════ constrained_non_guillotine_cutting ═══════════════
def _cngc_to_lib(off):
    pcs = off["pieces"]
    L, W = float(off["stock_length"]), float(off["stock_width"])
    return {"stock_length": L, "stock_width": W, "num_piece_types": len(pcs),
            "piece_lengths": [float(p["length"]) for p in pcs],
            "piece_widths": [float(p["width"]) for p in pcs],
            "piece_values": [float(p["value"]) for p in pcs],
            "min_usage": [int(p["min"]) for p in pcs],
            "max_usage": [int(p["max"]) for p in pcs],
            "max_slots": int(sum(p["max"] for p in pcs)),
            "big_m": 2.0 * max(L, W)}


def _cngc_to_off(sol, off, lib):
    """The library lists by "slot"; official requires 4-tuples (piece_type(1-indexed), x, y, r)."""
    out = []
    for p in (_g(sol, "placements") or []):
        if not isinstance(p, dict):
            if len(p) >= 4:
                out.append((int(p[0]) + 1, int(round(float(p[1]))), int(round(float(p[2]))), int(p[3])))
            continue
        if not int(p.get("is_used", 1)):
            continue
        t = int(p.get("piece_type_id", p.get("piece_type", -1)))
        if t < 0:
            continue
        out.append((t + 1, int(round(float(p.get("x", 0)))), int(round(float(p.get("y", 0)))),
                    int(p.get("is_rotated", 0))))
    return {"placements": out}


# ═══════════════ container_loading ═══════════════
# Orientation encoding (defined by this adapter; just needs to be consistent on both sides): o = v*2 + hswap
#   v     = original dimension index used as vertical (0/1/2)  → height = dims[v]
#   hswap = whether the other two dimensions are swapped (0/1)  → (length, width)
def _cl_orient(dims, o):
    v, hs = o // 2, o % 2
    rest = [dims[k] for k in range(3) if k != v]
    l, w = (rest[1], rest[0]) if hs else (rest[0], rest[1])
    return float(l), float(w), float(dims[v])


def _cl_to_lib(off):
    L, W, H = [float(x) for x in off["container"]]
    rl, rw, rh, vol, forb, meta = [], [], [], [], [], []
    for bt in sorted(off["box_types"]):
        b = off["box_types"][bt]
        dims, flags, cnt = b["dims"], b["flags"], int(b["count"])
        ors = [_cl_orient(dims, o) for o in range(6)]
        bad = [o for o in range(6) if not int(flags[o // 2])]
        v = float(dims[0]) * float(dims[1]) * float(dims[2])
        for _ in range(cnt):
            rl.append([x[0] for x in ors])
            rw.append([x[1] for x in ors])
            rh.append([x[2] for x in ors])
            vol.append(v)
            forb.append(list(bad))
            meta.append(int(bt))
    return {"container_length": L, "container_width": W, "container_height": H,
            "total_box_instances": len(vol),
            "boxes.volume": vol, "boxes.rotated_lengths": rl,
            "boxes.rotated_widths": rw, "boxes.rotated_heights": rh,
            "boxes.forbidden_orientations": forb,
            "_box_type_of": meta}


def _cl_to_off(sol, off, lib):
    meta = lib["_box_type_of"]
    st = _lst(_g(sol, "placement_status", "selection_status"))
    co = _g(sol, "coordinates", "positions") or []
    ors = _lst(_g(sol, "orientations"))
    out = []
    for i in range(len(meta)):
        if i >= len(st) or not int(st[i]):
            continue
        if i >= len(co):
            continue
        x, y, z = (list(co[i]) + [0, 0, 0])[:3]
        o = int(ors[i]) if i < len(ors) else 0
        if o < 0 or o > 5:
            o = 0
        out.append({"box_type": meta[i], "container_id": 1,
                    "x": int(round(x)), "y": int(round(y)), "z": int(round(z)),
                    "v": o // 2, "hswap": o % 2})
    return {"placements": out}



# ═══════════════ crew_scheduling ═══════════════
def _crew_to_lib(off):
    tasks = off["tasks"]
    N = int(off["N"])
    ids = sorted(tasks)
    st = [float(tasks[i][0]) for i in ids]
    ft = [float(tasks[i][1]) for i in ids]
    pos = {t: k for k, t in enumerate(ids)}
    tc = [[0.0] * N for _ in range(N)]
    vt = [[0] * N for _ in range(N)]
    for (a, b), c in off["arcs"].items():
        if a in pos and b in pos:
            tc[pos[a]][pos[b]] = float(c)
            vt[pos[a]][pos[b]] = 1
    return {"num_tasks": N, "max_crews": int(off["K"]),
            "max_duty_time": float(off["time_limit"]),
            "start_times": st, "finish_times": ft,
            "transition_costs": tc, "valid_transitions": vt,
            "_task_ids": ids}


def _crew_to_off(sol, off, lib):
    ids = lib["_task_ids"]
    routes = _g(sol, "crew_routes", "crews", "routes") or []
    out = []
    for r in routes:
        seq = []
        for v in _lst(r):
            k = int(v)
            seq.append(ids[k] if 0 <= k < len(ids) else k)
        if seq:
            out.append(seq)
    return {"crews": out}


# ═══════════════ container_loading_with_weight_restrictions ═══════════════
def _clw_to_lib(off):
    L, W, H = [float(x) for x in off["container"]]
    bt = off["box_types"]
    return {"container_length": L, "container_width": W, "container_height": H,
            "num_box_types": len(bt),
            "box_quantities": [int(b["count"]) for b in bt],
            "box_lengths": [float(b["length"]) for b in bt],
            "box_widths": [float(b["width"]) for b in bt],
            "box_heights": [float(b["height"]) for b in bt],
            "box_volumes": [float(b["length"]) * float(b["width"]) * float(b["height"]) for b in bt],
            "box_weights": [float(b["weight"]) for b in bt],
            # The library checker requires one scalar load-bearing value per box type; official gives three directions lb1/lb2/lb3.
            # Taking the minimum = most conservative; never treats a non-load-bearing direction as load-bearing (better strict than lax).
            "load_bearing_capacities": [min(float(b["lb1"]), float(b["lb2"]), float(b["lb3"])) for b in bt]}


def _clw_to_off(sol, off, lib):
    """The library groups by [box type][index within that type]; official requires a flat placements list."""
    sel = _g(sol, "selection_status") or []
    pos = _g(sol, "box_positions") or []
    ors = _g(sol, "box_orientations") or []
    out = []
    for t in range(len(sel)):
        row = _lst(sel[t])
        for i in range(len(row)):
            if not int(row[i]):
                continue
            try:
                x, y, z = (list(pos[t][i]) + [0, 0, 0])[:3]
            except Exception:
                continue
            try:
                o = int(ors[t][i])
            except Exception:
                o = 0
            out.append({"box_type": t + 1, "orientation": o,
                        "x": float(x), "y": float(y), "z": float(z)})
    return {"placements": out, "instance": 0, "util": 0.0, "m": len(out)}


# ═══════════════ constrained_guillotine_cutting ═══════════════
def _cgc_to_lib(off):
    pt = off["piece_types"]
    return {"sheet_length": float(off["stock_length"]), "sheet_width": float(off["stock_width"]),
            "num_piece_types": len(pt),
            "piece_lengths": [float(p["length"]) for p in pt],
            "piece_widths": [float(p["width"]) for p in pt],
            "max_quantities": [int(p["max"]) for p in pt],
            "piece_values": [float(p["value"]) for p in pt]}


def _cgc_to_off(sol, off, lib):
    """Official placements are 6-tuples (piece_type_idx, x, y, placed_len, placed_wid, orient_flag);
    orient_flag must be 0 (fixed orientation in this problem), and placed_len/wid must equal the type's original dimensions."""
    pt = off["piece_types"]
    ss = _g(sol, "selection_status") or []
    pl = _g(sol, "placements") or []
    out, tv = [], 0
    for i in range(min(len(ss), len(pt))):
        row = _lst(ss[i])
        L, W, V = int(pt[i]["length"]), int(pt[i]["width"]), int(pt[i]["value"])
        for k in range(len(row)):
            if not int(row[k]):
                continue
            try:
                c = pl[i][k]
                out.append((i + 1, int(round(float(c[0]))), int(round(float(c[1]))), L, W, 0))
                tv += V
            except Exception:
                pass
    return {"placements": out, "total_value": tv}


# ═══════════════ vehicle_routing_period_routing ═══════════════
def _vrp_to_lib(off):
    dep, cu = off["depot"], off["customers"]
    pts = [(float(dep["x"]), float(dep["y"]))] + [(float(c["x"]), float(c["y"])) for c in cu]
    n = len(pts)
    dm = [[math.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1]) for j in range(n)] for i in range(n)]
    vecs, cand = [], []
    for c in cu:
        idx = []
        for sch in c["schedules"]:
            v = [int(x) for x in sch]
            if v not in vecs:
                vecs.append(v)
            idx.append(vecs.index(v))
        cand.append(idx)
    return {"num_customers": len(cu), "planning_period": int(off["period_length"]),
            "num_vehicles": int(min(off["vehicles_per_day"])),
            "vehicle_capacity": float(off["vehicle_capacity"]),
            "demands": [float(c["demand"]) for c in cu],
            "distance_matrix": dm, "candidate_schedules": cand,
            "schedule_vectors": vecs, "_cust_ids": [int(c["id"]) for c in cu]}


def _vrp_to_off(sol, off, lib):
    ids, vecs = lib["_cust_ids"], lib["schedule_vectors"]
    cand = lib["candidate_schedules"]
    sa = _g(sol, "schedule_assignments", "selected_schedules") or {}
    sel = {}
    for i, cid in enumerate(ids):
        v = None
        if isinstance(sa, dict):
            # the problem format keys customers from 1 ("customer_1" is the first customer)
            for k in (f"customer_{i + 1}", f"customer_{cid}", str(cid), cid):
                if k in sa:
                    v = sa[k]; break
        elif i < len(sa):
            v = sa[i]
        if v is None:
            v = cand[i][0] if cand[i] else 0
        vi = int(v)
        vec = None
        if 0 <= vi < len(vecs) and vi in cand[i]:
            vec = vecs[vi]                     # A global schedule index, as the problem's checker reads it
        if vec is None and 0 <= vi < len(cand[i]):
            vec = vecs[cand[i][vi]]            # Otherwise "which candidate of this customer"
        if vec is None:
            vec = vecs[cand[i][0]] if cand[i] else [0] * lib["planning_period"]
        sel[cid] = list(vec)
    routes = _g(sol, "daily_routes", "tours") or []
    tours = {}
    for d, day in enumerate(routes):
        lst = []
        for tour in (day or []):
            t = [int(v) for v in _lst(tour)]
            if t and t[0] == 0:
                t = t[1:]
            if t and t[-1] == 0:
                t = t[:-1]
            if not t:
                continue                       # A trip with only the depot is judged "tour too short" by the official side
            lst.append([0] + t + [0])
        tours[d + 1] = lst
    return {"selected_schedules": sel, "tours": tours}



# ═══════════════ assortment_problem ═══════════════
def _asrt_to_lib(off):
    st, pc = off["stocks"], off["pieces"]
    return {"m": len(pc), "S": len(st),
            "length": [float(s["length"]) for s in st],
            "width": [float(s["width"]) for s in st],
            "area": [float(s["length"]) * float(s["width"]) for s in st],
            "piece_length": [float(p["length"]) for p in pc],
            "piece_width": [float(p["width"]) for p in pc],
            "piece_area": [float(p["length"]) * float(p["width"]) for p in pc],
            "min": [int(p["min"]) for p in pc],
            "max": [int(p["max"]) for p in pc],
            "stocks": st, "pieces": pc,
            "waste_cost": off.get("waste_cost", 1.0)}


def _asrt_to_off(sol, off, lib):
    """Official placements: {stock sheet instance number (1-indexed): {stock_type, placements:[{piece,x,y,orientation}]}}."""
    inst = _g(sol, "stock_instances") or []
    out = {}
    for k, si in enumerate(inst):
        if not isinstance(si, dict):
            continue
        pls = []
        for p in (si.get("placements") or []):
            if not isinstance(p, dict):
                continue
            t = int(p.get("piece_type_id", p.get("piece", 0)))
            if "piece_type_id" in p:
                t += 1                     # Library 0-based → official 1-indexed
            pls.append({"piece": max(1, t),
                        "x": float(p.get("x", 0)), "y": float(p.get("y", 0)),
                        "orientation": int(p.get("is_rotated", p.get("orientation", 0)))})
        stype = int(si.get("stock_type_id", si.get("stock_type", 0)))
        if "stock_type_id" in si:
            stype += 1                     # Library 0-based → official 1-indexed
        out[k + 1] = {"stock_type": max(1, stype), "placements": pls}
    return {"placements": out, "objective": 0.0}


TABLE = {
    "travelling_salesman_problem": {"to_lib": _tsp_to_lib, "to_off": _tsp_to_off},
    "assignment_problem": {"to_lib": _ap_to_lib, "to_off": _ap_to_off},
    "common_due_date_scheduling": {"to_lib": _cdd_to_lib, "to_off": _cdd_to_off},
    "bin_packing_one_dimensional": {"to_lib": _bp_to_lib, "to_off": _bp_to_off},
    "flow_shop_scheduling": {"to_lib": _fs_to_lib, "to_off": _fs_to_off},
    "multidimensional_knapsack_problem": {"to_lib": _mkp_to_lib, "to_off": _mkp_to_off},
    "multi_demand_multidimensional_knapsack_problem": {"to_off": _mdmkp_to_off},
    "set_covering": {"to_lib": _sc_to_lib, "to_off": _sc_to_off},
    "p_median_uncapacitated": {"to_lib": _pmu_to_lib, "to_off": _pmu_to_off},
    "corporate_structuring": {"to_lib": _cs_to_lib, "to_off": _cs_to_off},
    "graph_colouring": {"to_lib": _gc_to_lib, "to_off": _gc_to_off},
    "maximal_independent_set": {"to_lib": _mis_to_lib, "to_off": _mis_to_off},
    "equitable_partitioning_problem": {"to_lib": _eq_to_lib, "to_off": _eq_to_off},
    "euclidean_steiner_problem": {"to_lib": _est_to_lib, "to_off": _est_to_off},
    "job_shop_scheduling": {"to_lib": _js_to_lib, "to_off": _js_to_off},
    "open_shop_scheduling": {"to_lib": _os_to_lib, "to_off": _os_to_off},
    "hybrid_reentrant_shop_scheduling": {"to_lib": _hr_to_lib, "to_off": _hr_to_off},
    "generalised_assignment_problem": {"to_lib": _gap_to_lib, "to_off": _gap_to_off},
    "uncapacitated_warehouse_location": {"to_lib": _uwl_to_lib, "to_off": _uwl_to_off},
    "packing_unequal_circles": {"to_off": _puc_to_off},
    "packing_unequal_circles_area": {"to_off": _puc_to_off},
    "packing_unequal_rectangles_and_squares": {"to_lib": _purs_to_lib, "to_off": _purs_to_off},
    "packing_unequal_rectangles_and_squares_area": {"to_off": _purs_to_off},
    "set_partitioning": {"to_lib": _sp_to_lib, "to_off": _sp_to_off},
    "p_median_capacitated": {"to_lib": _pmc_to_lib, "to_off": _pmc_to_off},
    "aircraft_landing": {"to_lib": _al_to_lib, "to_off": _al_to_off},
    "resource_constrained_shortest_path": {"to_lib": _rcsp_to_lib, "to_off": _rcsp_to_off},
    "capacitated_warehouse_location": {"to_lib": _cwl_to_lib, "to_off": _cwl_to_off},
    "constrained_non_guillotine_cutting": {"to_lib": _cngc_to_lib, "to_off": _cngc_to_off},
    "container_loading": {"to_lib": _cl_to_lib, "to_off": _cl_to_off},
    "crew_scheduling": {"to_lib": _crew_to_lib, "to_off": _crew_to_off},
    "container_loading_with_weight_restrictions": {"to_lib": _clw_to_lib, "to_off": _clw_to_off},
    "constrained_guillotine_cutting": {"to_lib": _cgc_to_lib, "to_off": _cgc_to_off},
    "vehicle_routing_period_routing": {"to_lib": _vrp_to_lib, "to_off": _vrp_to_off},
    "assortment_problem": {"to_lib": _asrt_to_lib, "to_off": _asrt_to_off},
}


def get(official):
    e = TABLE.get(official, {})
    return {"to_lib": e.get("to_lib", _id_inst), "to_off": e.get("to_off", _id_sol)}
