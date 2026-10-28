# Capacitated warehouse location: mathematical model

Problem statement of the prototype's problem type used for this CO-Bench problem (decision variables, constraints, data field semantics, objective and I/O schema).

```text
A company must decide which potential plant sites to open and how to allocate customer demands to these open plants. Each plant has a fixed opening cost and a maximum capacity. Each customer has a specific demand that can be split across multiple plants. The goal is to satisfy all customer demands while minimizing the sum of fixed opening costs and variable shipping costs. All indices for plants and customers start from 0.

== DECISION VARIABLES (authoritative; model with exactly these) ==
- y[j]: Binary indicator for whether plant j is opened  |  y[j] ∈ {0,1}
- x[i][j]: The fraction of customer i's demand served by plant j  |  x[i][j] ∈ [0,1]

== CONSTRAINTS (authoritative; ALL must be modeled) ==
- [hard] Every customer's demand must be met in full, meaning the sum of fractions assigned to a customer across all plants must be exactly one.  |  Σ_{j=0}^{num_plants-1} x[i][j] = 1, ∀i ∈ {0, ..., num_customers-1}
- [hard] The total demand assigned to each plant must not exceed its capacity if opened, and must be zero if the plant is closed.  |  Σ_{i=0}^{num_customers-1} (customers_demands[i] * x[i][j]) ≤ plants_capacities[j] * y[j], ∀j ∈ {0, ..., num_plants-1}

== DATA FIELD SEMANTICS (key meanings and axis conventions) ==
- num_customers: num_customers; axis: scalar
- num_plants: num_plants; axis: scalar
- plants_fixed_costs: plants_fixed_costs; unit=currency; axis: plant
- shipping_costs: shipping_costs; unit=currency; axis: customer × plant
- customers_demands: customers_demands; unit=demand units; axis: customer
- plants_capacities: plants_capacities; unit=capacity units; axis: plant

== OBJECTIVE (authoritative; the solver MUST optimize exactly this) ==
Sense: minimize
Formula: min Σ_{j=0}^{num_plants-1} (plants_fixed_costs[j] * y[j]) + Σ_{i=0}^{num_customers-1} Σ_{j=0}^{num_plants-1} (shipping_costs[i][j] * x[i][j])
Plain: The sum of the fixed costs for all opened plants plus the total shipping costs, where shipping cost for a pair is proportional to the fraction of demand served.
If any earlier sentence conflicts with this section, this section wins.

== IO SCHEMA ==
{"num_customers": int, "num_plants": int, "customers_demands": [int, ...], "customers_x_coords": [float, ...], "customers_y_coords": [float, ...], "plants_capacities": [float, ...], "plants_fixed_costs": [int, ...], "plants_x_coords": [float, ...], "plants_y_coords": [float, ...], "shipping_costs": [[float, ...], ...], "total_demand": int, "total_capacity": float, "capacity_tightness_ratio": int}

OUTPUT SCHEMA — the solution dict you return MUST use EXACTLY these keys and shapes (this was confirmed upstream; do NOT rename, add, or drop keys):
- "open_plants": list[int] — whether each candidate plant is opened. example: [1, 0, 1, 1, 0]. Lists the open status of all num_plants plants, 1 = open, 0 = closed
- "assignments": dict[str, dict[str, float]] — fraction of each customer's demand supplied by each plant. example: {"0": {"0": 0.5, "1": 0.0, "2": 0.5}, "1": {"0": 0.0, "1": 1.0, "2": 0.0}}. Includes all num_customers customers; each customer dict contains the allocation fractions for all num_plants plants (including entries with fraction 0)
Fill every value for every entity; keep the keys EXACTLY as above.
```

## Objective

**Direction**: min

**Formula**
```
min Σ_{j=0}^{num_plants-1} (plants_fixed_costs[j] * y[j]) + Σ_{i=0}^{num_customers-1} Σ_{j=0}^{num_plants-1} (shipping_costs[i][j] * x[i][j])
```

**Plain-language statement**
The sum of the fixed costs for all opened plants plus the total shipping costs, where shipping cost for a pair is proportional to the fraction of demand served.

**Rationale**
The objective is to minimize the total cost, which consists of two parts: the one-time cost of opening facilities and the variable cost of transporting goods to meet customer demands.
