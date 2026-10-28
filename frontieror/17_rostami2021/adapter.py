"""Format adapter for rostami2021 (stochastic single-allocation hub location, DEF_V).

Official instance: {"n", "num_scenarios", "cost_parameters": {"chi", "alpha", "delta"}, "p_hubs", "node_coordinates",
                    "distances", "base_demands", "fixed_costs", "hub_capacities", "scenarios": [{"probability",
                    "pi_factors", "demands"}], "problem_info": {...}, "generation_metadata": {...}}.
Task format:       the same data flattened to "<block>_<field>" names (cost_parameters_alpha, scenarios_demands,
                   scenarios_probability, problem_info_variant, ...); nodes and scenarios keep their 0-based indices.

Official solution: {"objective_value", "hubs", "z_values", "allocations": {"<s>": {"<node>": hub}},
"x_values": {"<s>": {"<node>": {"<hub>": 1.0}}}}. The task solution gives the hub indicator of every node
("hub_selection") and, per scenario, the hub serving every node ("scenario_allocations", a hub serving itself).
x^s_ik = 1 exactly when node i is not a hub and is served by hub k in scenario s. The objective (fixed costs plus
the expected collection, transfer and distribution costs; fixed costs dropped for the SApHMP variant) is recomputed
with the official DEF_V formula.
"""


def to_ours(official_instance: dict) -> dict:
    d = official_instance
    cp, info, gen = d["cost_parameters"], d.get("problem_info", {}), d.get("generation_metadata", {})
    sc = d["scenarios"]
    return {
        "n": d["n"],
        "num_scenarios": d["num_scenarios"],
        "node_coordinates": d.get("node_coordinates"),
        "distances": d["distances"],
        "fixed_costs": d["fixed_costs"],
        "scenarios_probability": [s["probability"] for s in sc],
        "cost_parameters_alpha": cp["alpha"],
        "cost_parameters_chi": cp["chi"],
        "cost_parameters_delta": cp["delta"],
        "scenarios_demands": [s["demands"] for s in sc],
        "base_demands": d.get("base_demands"),
        "p_hubs": d.get("p_hubs"),
        "hub_capacities": d.get("hub_capacities"),
        "scenarios_pi_factors": [s.get("pi_factors") for s in sc],
        "generation_metadata_attempt": gen.get("attempt"),
        "generation_metadata_generator": gen.get("generator"),
        "problem_info_instance_type": info.get("instance_type"),
        "problem_info_name": info.get("name"),
        "problem_info_paper": info.get("paper"),
        "problem_info_variant": info.get("variant"),
    }


def to_official(official_instance: dict, our_solution: dict) -> dict:
    n, S = official_instance["n"], official_instance["num_scenarios"]
    z = [int(round(float(v))) for v in our_solution["hub_selection"][:n]]
    alloc = our_solution["scenario_allocations"]
    allocations, x_values = {}, {}
    for s in range(S):
        h = alloc.get(str(s), alloc.get(s))
        allocations[str(s)] = {str(i): int(h[i]) for i in range(n)}
        x_values[str(s)] = {str(i): ({str(int(h[i])): 1.0} if z[i] == 0 and int(h[i]) != i else {})
                            for i in range(n)}
    solution = {"hubs": [k for k in range(n) if z[k] == 1], "z_values": [float(v) for v in z],
                "allocations": allocations, "x_values": x_values}
    solution["objective_value"] = official_objective(official_instance, solution)
    return solution


def official_objective(official_instance: dict, official_solution: dict) -> float:
    d = official_instance
    n = d["n"]
    dist = d["distances"]
    chi, alpha, delta = (d["cost_parameters"][k] for k in ("chi", "alpha", "delta"))
    z = [float(v) for v in official_solution["z_values"]]
    total = 0.0 if d.get("problem_info", {}).get("variant") == "SApHMP" else \
        sum(f * zk for f, zk in zip(d["fixed_costs"], z))
    for s, sc in enumerate(d["scenarios"]):
        w = sc["demands"]
        out_flow = [sum(w[i]) for i in range(n)]
        in_flow = [sum(w[j][i] for j in range(n)) for i in range(n)]
        xs = official_solution["x_values"].get(str(s), {})
        X = []                                   # X[i] = {k: value}, X[i][i] = z_i
        cost = 0.0
        for i in range(n):
            row = {int(k): float(v) for k, v in xs.get(str(i), {}).items() if int(k) != i and float(v) != 0.0}
            for k, v in row.items():             # collection + distribution
                cost += dist[i][k] * (chi * out_flow[i] + delta * in_flow[i]) * v
            if z[i] != 0.0:
                row[i] = z[i]
            X.append(row)
        via = []                                 # via[i][l] = sum_k X[i][k] * dist[k][l]
        for i in range(n):
            acc = [0.0] * n
            for k, v in X[i].items():
                dk = dist[k]
                for l in range(n):
                    acc[l] += v * dk[l]
            via.append(acc)
        transfer = 0.0
        for i in range(n):
            vi, wi = via[i], w[i]
            for j in range(n):
                if wi[j]:
                    transfer += wi[j] * sum(v * vi[l] for l, v in X[j].items())
        total += float(sc["probability"]) * (cost + alpha * transfer)
    return total
