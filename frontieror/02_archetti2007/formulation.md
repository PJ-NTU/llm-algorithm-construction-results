# archetti2007 · Mathematical model

> Single-depot inventory routing problem (IRP)
>
> Sense: min

---

Mathematical model of the problem type as revised for the portfolio updated by continued evolution (parameters, decision variables, objective and constraints).

A supplier manages the inventory of N retailers over a time horizon of H periods. The supplier must decide which retailers to visit in each period, the quantity to deliver (following an order-up-to level policy), and the optimal vehicle routes. The goal is to minimize the total cost, which includes inventory holding costs at the supplier and all retailers, plus transportation costs. Indices for retailers and time periods start from 0, where node 0 is the supplier and nodes 1 to N are retailers.

== DECISION VARIABLES (authoritative; model with exactly these) ==
- y[t][i][j]: Binary or integer variable indicating if the vehicle travels from node i to node j in period t.  |  y[t][i][j] ∈ {0,1} for i,j > 0; y[t][i][0] ∈ {0,1,2}
- z[t][i]: Binary variable indicating if node i is visited in period t.  |  z[t][i] ∈ {0,1}
- x[t][s]: Quantity of product delivered to retailer s in period t.  |  x[t][s] ≥ 0
- I[t][s]: Inventory level of retailer s at the start of period t.  |  I[t][s] ≥ 0
- B[t]: Inventory level at the supplier at the start of period t.  |  B[t] ≥ 0

== CONSTRAINTS (authoritative; ALL must be modeled) ==
- [hard] The supplier's inventory at the start of period t+1 is the inventory at the start of period t plus production minus total shipments to retailers in period t.  |  B[t+1] = B[t] + r_0 - \sum_{s=1}^{n} x[t][s]
- [hard] In each period, the supplier's current inventory must be sufficient to cover the total quantity shipped to all retailers.  |  B[t] \geq \sum_{s=1}^{n} x[t][s]
- [hard] A retailer's inventory at the start of period t+1 is the inventory at the start of period t plus the quantity received minus the quantity consumed in period t.  |  I[t+1][s] = I[t][s] + x[t][s] - r_s
- [hard] Retailer inventory must never fall below zero, and must not exceed the maximum inventory level at the start of any period.  |  0 \leq I[t][s] \leq U_s
- [hard] If a retailer is visited, the quantity shipped must raise its inventory to the maximum level; otherwise, the quantity shipped is zero.  |  x[t][s] \geq U_s z[t][s] - I[t][s] \text{ and } x[t][s] \leq U_s - I[t][s] \text{ and } x[t][s] \leq U_s z[t][s]
- [hard] The total quantity loaded onto the vehicle in any period must not exceed its capacity.  |  \sum_{s=1}^{n} x[t][s] \leq C
- [hard] The vehicle can only deliver product if it leaves the supplier.  |  \sum_{s=1}^{n} x[t][s] \leq C z[t][0]
- [hard] Each node in the route must have exactly two incident edges (inflow and outflow) if visited.  |  \sum_{j < i} y[t][i][j] + \sum_{j > i} y[t][j][i] = 2 z[t][i]
- [hard] Subtours among retailers must be eliminated.  |  \sum_{i \in S} \sum_{j \in S, j < i} y[t][i][j] \leq \sum_{i \in S} z[t][i] - z[t][k] \quad \forall S \subseteq \mathcal{M}, k \in S

== DATA FIELD SEMANTICS (key meanings and axis conventions) ==
- num_retailers: n; unit=count
- time_horizon: H; unit=periods
- supplier_production_rate: r_0; unit=units
- retailers_demand: r_s; unit=units; axis: retailer
- supplier_starting_inventory: B_1; unit=units
- retailers_max_inventory: U_s; unit=units; axis: retailer
- retailers_starting_inventory: I_1s; unit=units; axis: retailer
- supplier_inventory_cost: h_0; unit=currency/unit/period
- retailers_inventory_cost: h_s; unit=currency/unit/period; axis: retailer
- vehicle_capacity: C; unit=units
- distance_matrix: c_ij; unit=currency; axis: node×node

== OBJECTIVE (authoritative; the solver MUST optimize exactly this) ==
Sense: minimize
Formula: min ∑_{t=1}^{H+1} (h_0 * B_t + ∑_{s=1}^{N} h_s * I_{ts}) + ∑_{t=1}^{H} ∑_{i,j ∈ M', j<i} c_{ij} * y_{tij}
Plain: Minimize the sum of inventory holding costs at the supplier, inventory holding costs at all retailers across the extended horizon, and the total transportation costs of the vehicle routes.
If any earlier sentence conflicts with this section, this section wins.

== IO SCHEMA ==
{"cost_config": str, "distance_matrix": [[float, ...], ...], "num_retailers": int, "problem_description": str, "retailers_demand": [int, ...], "retailers_id": [int, ...], "retailers_inventory_cost": [float, ...], "retailers_max_inventory": [int, ...], "retailers_starting_inventory": [int, ...], "retailers_x": [int, ...], "retailers_y": [int, ...], "seed": int, "supplier_id": int, "supplier_inventory_cost": float, "supplier_production_rate": int, "supplier_starting_inventory": int, "supplier_x": int, "supplier_y": int, "time_horizon": int, "vehicle_capacity": int}

OUTPUT SCHEMA — the solution dict you return MUST use EXACTLY these keys and shapes (this was confirmed upstream; do NOT rename, add, or drop keys):
- "routes": dict[str, list[list[int]]] — route segments travelled by the vehicle in each period (including start node, end node and number of traversals). example: {"1": [[0, 1, 1], [1, 2, 1], [2, 0, 1]], "2": [[0, 3, 2]], "3": []}. Includes routes for all periods in the horizon; an empty list if a period has no deliveries
- "retailer_visit_status": dict[str, list[int]] — whether each retailer is visited in each period. example: {"1": [1, 1, 0, 0, 0], "2": [0, 0, 1, 0, 0], "3": [0, 0, 0, 0, 0]}. Each period must list the visit status (0 or 1) of all retailers
- "deliveries": dict[str, dict[str, float]] — quantity of product delivered to each retailer in each period. example: {"1": {"1": 50.0, "2": 30.0, "3": 0.0, "4": 0.0, "5": 0.0}, "2": {"1": 0.0, "2": 0.0, "3": 45.0, "4": 0.0, "5": 0.0}}. Each period must list the delivery quantity of all retailers, 0 if not visited
- "retailer_inventory": dict[str, dict[str, float]] — retailer inventory levels at the start of each period. example: {"1": {"1": 10.0, "2": 15.0}, "2": {"1": 5.0, "2": 10.0}, "3": {"1": 100.0, "2": 100.0}}. Includes the inventory of all retailers from period 1 to period H+1
- "supplier_inventory": dict[str, float] — supplier inventory level at the start of each period. example: {"1": 500.0, "2": 420.0, "3": 450.0}. Includes the supplier inventory from period 1 to period H+1
Fill every value for every entity; keep the keys EXACTLY as above.
