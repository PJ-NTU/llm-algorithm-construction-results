# rostami2021 · Mathematical model

> Stochastic single-allocation hub location problem
>
> Sense: min

---

A two-stage stochastic hub location problem where the first stage involves selecting which nodes to open as hubs, and the second stage involves assigning each non-hub node to exactly one open hub for each demand scenario. The goal is to minimize the sum of fixed setup costs and the expected transportation costs (collection, inter-hub transfer, and distribution). All indices are 0-indexed. N is the number of nodes, and S is the number of scenarios.

== DECISION VARIABLES (authoritative; model with exactly these) ==
- z[k]: Binary indicator for whether node k is selected as a hub.  |  z[k] ∈ {0,1}, k ∈ {0,...,N-1}
- x[s][i][k]: Binary indicator for whether node i is assigned to hub k in scenario s.  |  x[s][i][k] ∈ {0,1}, s ∈ {0,...,S-1}, i,k ∈ {0,...,N-1}, i ≠ k

== CONSTRAINTS (authoritative; ALL must be modeled) ==
- [hard] Each non-hub node must be assigned to exactly one hub in each scenario.  |  sum_{k=0, k≠i}^{N-1} x[s][i][k] = 1 - z[i], \forall i \in {0,...,N-1}, s \in {0,...,S-1}
- [hard] A node can only be assigned to a hub if that hub has been opened.  |  x[s][i][k] \le z[k], \forall i,k \in {0,...,N-1}, i 
eq k, s \in {0,...,S-1}
- [hard] At least one hub must be opened in the network.  |  sum_{k=0}^{N-1} z[k] \ge 1

== DATA FIELD SEMANTICS (key meanings and axis conventions) ==
- n: N; unit=nodes
- num_scenarios: S; unit=scenarios
- cost_parameters_alpha: alpha; unit=cost/unit-flow/distance
- cost_parameters_chi: chi; unit=cost/unit-flow/distance
- cost_parameters_delta: delta; unit=cost/unit-flow/distance
- distances: c; unit=cost; axis: scenario × node × candidate hub
- fixed_costs: fixed_costs; unit=currency; axis: node
- scenarios_probability: scenarios_probability; unit=probability; axis: scenario
- scenarios_demands: w; unit=flow units; axis: scenario × origin × destination

== OBJECTIVE (authoritative; the solver MUST optimize exactly this) ==
Sense: minimize
Formula: min sum_{k=0}^{N-1} (fixed_costs[k] * z[k]) + sum_{s=0}^{S-1} (scenarios_probability[s] * (sum_{i=0}^{N-1} sum_{k=0, k≠i}^{N-1} c[s][i][k] * x[s][i][k] + sum_{i=0}^{N-1} sum_{j=0}^{N-1} alpha * w[s][i][j] * [dist[i][j]*z[i]*z[j] + sum_{l≠j} dist[i][l]*z[i]*x[s][j][l] + sum_{k≠i} dist[k][j]*x[s][i][k]*z[j] + sum_{k≠i} sum_{l≠j} dist[k][l]*x[s][i][k]*x[s][j][l]]))
Plain: Minimize the total fixed setup costs for hubs plus the expected transportation costs across all scenarios. Transportation costs include collection from origin to first hub, inter-hub transfer, and distribution from second hub to destination.
If any earlier sentence conflicts with this section, this section wins.

== IO SCHEMA ==
{"n": int, "num_scenarios": int, "node_coordinates": [[float, ...], ...], "distances": [[float, ...], ...], "fixed_costs": [int, ...], "scenarios_probability": [float, ...], "cost_parameters_alpha": float, "cost_parameters_chi": float, "cost_parameters_delta": float, "scenarios_demands": [[[float, ...], ...], ...], "base_demands": [[float, ...], ...], "p_hubs": any, "hub_capacities": any, "scenarios_pi_factors": [[float, ...], ...], "generation_metadata_attempt": int, "generation_metadata_generator": str, "problem_info_instance_type": str, "problem_info_name": str, "problem_info_paper": str, "problem_info_variant": str}

OUTPUT SCHEMA — the solution dict you return MUST use EXACTLY these keys and shapes (this was confirmed upstream; do NOT rename, add, or drop keys):
- "hub_selection": list[int] — whether the node is selected as a hub. example: [1, 0, 1, 0, 0, 0, 0, 0, 0, 0]. List all n nodes; 1 means selected as a hub, 0 means not selected
- "scenario_allocations": dict[str, list[int]] — index of the hub node to which each node is assigned under each demand scenario. example: {"0": [0, 0, 2, 0, 2, 2, 0, 2, 0, 2], "1": [0, 2, 2, 2, 0, 0, 0, 2, 2, 0], "2": [0, 0, 2, 2, 2, 2, 0, 0, 0, 2], "3": [0, 2, 2, 0, 0, 0, 2, 2, 0, 2], "4": [0, 0, 2, 0, 2, 0, 0, 2, 2, 2]}. Include all num_scenarios scenarios; for each scenario the list has length n and records the hub index assigned to each node (a node that is itself a hub points to itself)
Fill every value for every entity; keep the keys EXACTLY as above.