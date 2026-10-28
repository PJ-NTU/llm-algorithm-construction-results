"""Format adapter for roberti2018 (two-echelon capacitated vehicle routing with grouping constraints).

Official instance: {"num_satellites", "num_customers", "num_groups", "Q1", "Q2", "T1", "T2", "depot": {"id", "x", "y"},
                    "satellites": [{"id", "x", "y", "capacity", "max_vehicles", "handling_cost"}],
                    "customers": [{"id", "x", "y", "demand", "group"}], "groups": [{"group_id", "customer_ids"}],
                    "instance_type", ...}.
Task format:       the same data flattened to "<block>_<field>" lists (satellites_id, customers_demand,
                   groups_customer_ids, ...). Node ids (depot, satellites, customers) and group ids are the official
                   ones, as in the task's own instances.

Official solution: {"objective_value", "first_echelon_routes": [{"vehicle", "route", "deliveries": {"<sat>": q}}],
"second_echelon_routes": [{"satellite", "route"}], "group_assignments": {"<group_id>": sat}}.
- first echelon: task route k (vehicle k) becomes vehicle k; its deliveries are the task's first_echelon_deliveries
  with vehicle_id k and a non-zero quantity. Unused vehicles (route [depot, depot], nothing delivered) are left out.
- second echelon: every task route that visits a customer becomes one route from its first node (the satellite).
- group_assignments: task key "group_<id>" becomes "<id>".
The objective (first-echelon travel + satellite handling + second-echelon travel, Euclidean distances) is recomputed
the way the official checker does, i.e. from the arc sets of the two echelons and the delivered quantities.
"""
import math


def to_ours(official_instance: dict) -> dict:
    d = official_instance
    sats, custs, groups = d["satellites"], d["customers"], d["groups"]
    return {
        "Q1": d["Q1"], "Q2": d["Q2"], "T1": d["T1"], "T2": d["T2"],
        "customers_demand": [c["demand"] for c in custs],
        "customers_group": [c["group"] for c in custs],
        "customers_id": [c["id"] for c in custs],
        "customers_x": [c["x"] for c in custs],
        "customers_y": [c["y"] for c in custs],
        "depot_id": d["depot"]["id"], "depot_x": d["depot"]["x"], "depot_y": d["depot"]["y"],
        "groups_customer_ids": [list(g["customer_ids"]) for g in groups],
        "groups_group_id": [g["group_id"] for g in groups],
        "instance_type": d.get("instance_type", ""),
        "num_customers": len(custs), "num_groups": len(groups), "num_satellites": len(sats),
        "satellites_capacity": [s["capacity"] for s in sats],
        "satellites_handling_cost": [s["handling_cost"] for s in sats],
        "satellites_id": [s["id"] for s in sats],
        "satellites_max_vehicles": [s["max_vehicles"] for s in sats],
        "satellites_x": [s["x"] for s in sats],
        "satellites_y": [s["y"] for s in sats],
    }


def to_official(official_instance: dict, our_solution: dict) -> dict:
    depot = official_instance["depot"]["id"]
    deliveries = {}
    for rec in our_solution.get("first_echelon_deliveries", []):
        q = float(rec["quantity"])
        if q != 0.0:
            deliveries.setdefault(int(rec["vehicle_id"]), {})[str(int(rec["satellite_id"]))] = q
    first = []
    for k, route in enumerate(our_solution["first_echelon_routes"]):
        if any(v != depot for v in route) or k in deliveries:
            first.append({"vehicle": k, "route": [int(v) for v in route], "deliveries": deliveries.get(k, {})})
    customers = {c["id"] for c in official_instance["customers"]}
    second = [{"satellite": int(r[0]), "route": [int(v) for v in r]}
              for r in our_solution["second_echelon_routes"] if any(v in customers for v in r)]
    groups = {}
    for key, sat in our_solution["group_assignments"].items():
        gid = str(key)[len("group_"):] if str(key).startswith("group_") else str(key)
        groups[gid] = int(sat)
    solution = {"first_echelon_routes": first, "second_echelon_routes": second, "group_assignments": groups}
    solution["objective_value"] = official_objective(official_instance, solution)
    return solution


def official_objective(official_instance: dict, official_solution: dict) -> float:
    d = official_instance
    xy = {d["depot"]["id"]: (d["depot"]["x"], d["depot"]["y"])}
    xy.update({s["id"]: (s["x"], s["y"]) for s in d["satellites"]})
    xy.update({c["id"]: (c["x"], c["y"]) for c in d["customers"]})
    handling = {s["id"]: s["handling_cost"] for s in d["satellites"]}

    def cost(i, j):
        return math.hypot(xy[i][0] - xy[j][0], xy[i][1] - xy[j][1])

    arcs1, arcs2, w = set(), set(), {}
    for r in official_solution["first_echelon_routes"]:
        k, route = r["vehicle"], r["route"]
        arcs1.update((route[p], route[p + 1], k) for p in range(len(route) - 1))
        for s, q in r.get("deliveries", {}).items():
            w[int(s), k] = float(q)
    for r in official_solution["second_echelon_routes"]:
        route = r["route"]
        arcs2.update((route[p], route[p + 1]) for p in range(len(route) - 1))
    T1 = d["T1"]
    return (sum(cost(i, j) for i, j, _ in arcs1)
            + sum(handling[s] * q for (s, k), q in w.items() if s in handling and 0 <= k < T1)
            + sum(cost(i, j) for i, j in arcs2))
