# Multidimensional knapsack problem: mathematical model

Mathematical model of the problem type as revised for the portfolio updated by continued evolution (parameters, decision variables, objective and constraints).

Given N items and M types of resources, each item j has a profit and consumes a specific amount of each resource r. The goal is to select a subset of items to maximize the total profit while ensuring that the total consumption of each resource r does not exceed its capacity. All indices start from 0.

== DECISION VARIABLES (authoritative; model with exactly these) ==
- x[j]: A binary variable indicating whether item j is selected.  |  x[j] ∈ {0,1}, j=0..N-1

== CONSTRAINTS (authoritative; ALL must be modeled) ==
- [hard] The total consumption of each resource must not exceed that resource's rated capacity.  |  Σ_{j=0}^{n-1} weights[r][j] * x[j] ≤ capacity[r], ∀r=0..m-1

== DATA FIELD SEMANTICS (key meanings and axis conventions) ==
- n: n; axis: scalar
- m: m; axis: scalar
- profit: profit; axis: item
- weights: weights; axis: resource×item
- capacity: capacity; axis: resource

== OBJECTIVE (authoritative; the solver MUST optimize exactly this) ==
Sense: maximize
Formula: maximize Σ_{j=0}^{N-1} profit[j] * x[j]
Plain: The sum of profits of all selected items.
If any earlier sentence conflicts with this section, this section wins.

== IO SCHEMA ==
{"n": int, "m": int, "profit": [float, ...], "weights": [[float, ...], ...], "capacity": [float, ...]}

OUTPUT SCHEMA — the solution dict you return MUST use EXACTLY these keys and shapes (this was confirmed upstream; do NOT rename, add, or drop keys):
- "selection_status": list[int] — whether each item is selected into the knapsack (1 means selected, 0 means not selected). example: [1, 0, 1, 1, 0]. Must contain the status of all n items; items not selected must also be marked 0
Fill every value for every entity; keep the keys EXACTLY as above.
