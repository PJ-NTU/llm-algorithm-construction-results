# roberti2018 · Mathematical model

> Two-echelon capacitated vehicle routing problem with grouping constraints
>
> Sense: min

---

A two-echelon freight distribution problem where large vehicles transport goods from a central depot to intermediate satellites, and small vehicles deliver goods from satellites to customers. Customers are partitioned into disjoint groups, and all customers in a single group must be served by the same satellite. The goal is to minimize the total cost, including travel costs in both echelons and handling costs at satellites. All indices for nodes, vehicles, and groups start from 0.

== DECISION VARIABLES (authoritative; model with exactly these) ==
- x: Binary variable indicating if first-echelon vehicle k travels from node i to node j in the first echelon network (depot and satellites).  |  x[i][j][k] ∈ {0,1}
- y: Binary variable indicating if a second-echelon vehicle travels from node i to node j in the second echelon network (satellites and customers).  |  y[i][j] ∈ {0,1}
- w: Continuous variable representing the quantity of freight delivered to satellite s by first-echelon vehicle k.  |  w[s][k] ≥ 0
- f: Continuous variable representing the load of a second-echelon vehicle when traversing arc (i, j).  |  f[i][j] ≥ 0
- v: Binary variable indicating if customer group h is assigned to satellite s.  |  v[h][s] ∈ {0,1}
- z: Binary variable indicating if customer i is served by satellite s.  |  z[i][s] ∈ {0,1}
- u: Continuous variable representing the visit order of satellite s in the route of first-echelon vehicle k.  |  u[s][k] ≥ 0

== CONSTRAINTS (authoritative; ALL must be modeled) ==
- [hard] First-echelon flow conservation: each vehicle must leave the node it enters, including the depot and satellites.  |  sum_{j} x[i][j][k] = sum_{j} x[j][i][k], ∀ i ∈ {depot} ∪ V_S, ∀ k ∈ {1..T1}
- [hard] Each satellite can be visited at most once by any specific first-echelon vehicle.  |  sum_{j} x[i][j][k] ≤ 1, ∀ i ∈ V_S, ∀ k ∈ {1..T1}
- [hard] First-echelon sub-tour elimination using MTZ constraints.  |  u[i][k] + 1 ≤ u[j][k] + num_satellites * (1 - x[i][j][k]), ∀ i, j ∈ V_S, ∀ k ∈ {1..T1}
- [hard] A first-echelon vehicle can only deliver freight to a satellite if it visits that satellite.  |  w[s][k] ≤ Q1 * sum_{j} x[s][j][k], ∀ s ∈ V_S, ∀ k ∈ {1..T1}
- [hard] The total freight carried by a first-echelon vehicle cannot exceed its capacity.  |  sum_{s ∈ V_S} w[s][k] ≤ Q1, ∀ k ∈ {1..T1}
- [hard] The total freight handled by a satellite cannot exceed its throughput capacity.  |  sum_{k ∈ {1..T1}} w[s][k] ≤ satellites_capacity[s], ∀ s ∈ V_S
- [hard] Every customer must be visited exactly once by exactly one second-echelon vehicle.  |  sum_{j} y[i][j] = 1 and sum_{j} y[j][i] = 1, ∀ i ∈ V_C
- [hard] The number of second-echelon vehicles operating from a satellite cannot exceed its limit.  |  sum_{i ∈ V_C} y[s][i] ≤ satellites_max_vehicles[s], ∀ s ∈ V_S
- [hard] The total number of second-echelon vehicles used across all satellites cannot exceed the system limit.  |  sum_{s ∈ V_S} sum_{i ∈ V_C} y[s][i] ≤ T2, ∀ s ∈ V_S
- [hard] Second-echelon flow conservation and demand satisfaction (load tracking).  |  sum_{j} f[j][i] = sum_{j} f[i][j] + customers_demand[i], ∀ i ∈ V_C
- [hard] Second-echelon vehicle capacity and arc flow logic.  |  customers_demand[j] * y[i][j] ≤ f[i][j] ≤ (Q2 - customers_demand[i]) * y[i][j], ∀ (i,j) ∈ A2
- [hard] Consistency between second-echelon routes and customer-to-satellite assignments.  |  z[j][s] ≥ z[i][s] + y[i][j] - 1, ∀ i, j ∈ V_S ∪ V_C, ∀ s ∈ V_S
- [hard] Grouping constraint: all customers in a group must be served by the same satellite.  |  z[i][s] = v[h][s], ∀ i ∈ groups_customer_ids[h], ∀ h ∈ {1..num_groups}, ∀ s ∈ V_S
- [hard] Echelon flow balance: the quantity delivered to a satellite by first-echelon vehicles must equal the total demand of customers served by that satellite.  |  sum_{k} w[s][k] = sum_{i ∈ V_C} sum_{j} f[s][i], ∀ s ∈ V_S

== DATA FIELD SEMANTICS (key meanings and axis conventions) ==
- Q1: Q1; unit=units
- Q2: Q2; unit=units
- T1: T1; unit=vehicles
- T2: T2; unit=vehicles
- customers_demand: customers_demand; unit=units; axis: customer
- Euclidean distance: c; unit=cost; axis: node_i, node_j
- satellites_capacity: satellites_capacity; unit=units; axis: satellite
- satellites_handling_cost: satellites_handling_cost; unit=cost/unit; axis: satellite
- satellites_max_vehicles: satellites_max_vehicles; unit=vehicles; axis: satellite
- groups_customer_ids: groups_customer_ids; axis: group
- num_satellites: num_satellites

== OBJECTIVE (authoritative; the solver MUST optimize exactly this) ==
Sense: minimize
Formula: min (sum_{k, i, j in A1} c[i][j] * x[i][j][k]) + (sum_{s, k} h[s] * w[s][k]) + (sum_{i, j in A2} c[i][j] * y[i][j])
Plain: Minimize the sum of first-echelon travel costs, satellite handling costs (proportional to throughput), and second-echelon travel costs.
If any earlier sentence conflicts with this section, this section wins.

== IO SCHEMA ==
{"Q1": int, "Q2": int, "T1": int, "T2": int, "customers_demand": [int, ...], "customers_group": [int, ...], "customers_id": [int, ...], "customers_x": [int, ...], "customers_y": [int, ...], "depot_id": int, "depot_x": int, "depot_y": int, "groups_customer_ids": [[int, ...], ...], "groups_group_id": [int, ...], "instance_type": str, "num_customers": int, "num_groups": int, "num_satellites": int, "satellites_capacity": [int, ...], "satellites_handling_cost": [float, ...], "satellites_id": [int, ...], "satellites_max_vehicles": [int, ...], "satellites_x": [int, ...], "satellites_y": [int, ...]}

OUTPUT SCHEMA — the solution dict you return MUST use EXACTLY these keys and shapes (this was confirmed upstream; do NOT rename, add, or drop keys):
- "first_echelon_routes": list[list[int]] — routes of the first-echelon large trucks (leave the depot, visit satellites, then return). example: [[0, 1, 2, 0], [0, 2, 0]]. Include the routes of all T1 vehicles; an unused vehicle has route [0, 0]
- "first_echelon_deliveries": list[dict[str, float]] — quantity of freight delivered by each first-echelon vehicle to each satellite. example: [{"satellite_id": 1, "vehicle_id": 0, "quantity": 50.0}, {"satellite_id": 2, "vehicle_id": 0, "quantity": 30.0}]. List every combination of vehicle and satellite, including those with a delivered quantity of 0
- "second_echelon_routes": list[list[int]] — routes of the second-echelon small trucks (leave a satellite, visit customers, then return to the same satellite). example: [[1, 10, 11, 1], [2, 15, 16, 2]]. List all second-echelon routes actually operated; each route starts and ends with the satellite ID
- "group_assignments": dict[str, int] — satellite to which each customer group is assigned. example: {"group_0": 1, "group_1": 1, "group_2": 2, "group_3": 2}. Include the assignments of all num_groups customer groups
- "customer_satellite_assignments": dict[str, int] — serving satellite to which each customer is assigned. example: {"customer_10": 1, "customer_11": 1, "customer_15": 2}. Include the assignments of all num_customers customers
Fill every value for every entity; keep the keys EXACTLY as above.