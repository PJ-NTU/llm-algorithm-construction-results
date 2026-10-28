# Resource constrained shortest path: mathematical model

Problem statement of the prototype's problem type used for this CO-Bench problem (decision variables, constraints, data field semantics, objective and I/O schema).

```text
Find a path from a source vertex to a destination vertex in a directed graph that minimizes total arc cost while satisfying cumulative resource consumption constraints at both vertices and arcs. The path must respect lower and upper bounds for each resource type. All indices for vertices, arcs, and resource types start from 0 (e.g., vertex 0 to vertex N-1).

== DECISION VARIABLES (authoritative; model with exactly these) ==
- x[i][j]: Binary indicator whether the arc from vertex i to vertex j is included in the path  |  x[i][j] ∈ {0,1}
- w[i][k]: The cumulative consumption of resource k when arriving at vertex i  |  w[i][k] ≥ 0, continuous

== CONSTRAINTS (authoritative; ALL must be modeled) ==
- [hard] The path must start at node 1 and end at node n, and intermediate nodes must satisfy flow balance (in-degree equals out-degree).  |  Σ_{j: (1,j)∈A} x[1][j] = 1; Σ_{i: (i,n)∈A} x[i][n] = 1; Σ_{i: (i,j)∈A} x[i][j] - Σ_{k: (j,k)∈A} x[j][k] = 0, ∀j ∈ V \ {1, n}
- [hard] The cumulative resource consumption at each node must lie within the given lower and upper bounds.  |  L[i][k] ≤ w[i][k] ≤ U[i][k], ∀i ∈ V, ∀k ∈ K
- [hard] Resource consumption accumulates along the path: the resource amount on arrival at node j equals the resource amount at predecessor node i plus the consumption of node i and of arc (i,j).  |  x[i][j] = 1 ⇒ w[j][k] ≥ w[i][k] + v_res[i][k] + a_res[i][j][k], ∀(i,j) ∈ A, ∀k ∈ K
- [hard] The initial resource consumption at the origin includes at least the origin's own resource consumption.  |  w[1][k] = v_res[1][k], ∀k ∈ K

== DATA FIELD SEMANTICS (key meanings and axis conventions) ==
- arc_costs: c[i][j]; axis: arc_index
- arcs: A; axis: arc_list
- n: V; unit=vertex_count
- resource_types: K; unit=resource_count
- lower_bounds: L[i][k]; axis: vertex×resource
- upper_bounds: U[i][k]; axis: vertex×resource
- vertex_resource_consumption: v_res[i][k]; axis: vertex×resource
- arc_resource_consumption: a_res[i][j][k]; axis: arc×resource

== OBJECTIVE (authoritative; the solver MUST optimize exactly this) ==
Sense: minimize
Formula: min Σ_{(i,j) ∈ A} c[i][j] * x[i][j]
Plain: Minimize the sum of costs of all arcs included in the path.
If any earlier sentence conflicts with this section, this section wins.

== IO SCHEMA ==
{"n": int, "resource_types": int, "arcs": [[int, ...], ...], "arc_costs": [float, ...], "vertex_resource_consumption": [[float, ...], ...], "arc_resource_consumption": [[float, ...], ...], "lower_bounds": [[float, ...], ...], "upper_bounds": [[float, ...], ...]}

OUTPUT SCHEMA — the solution dict you return MUST use EXACTLY these keys and shapes (this was confirmed upstream; do NOT rename, add, or drop keys):
- "path": list[int] — Sequence of vertices visited in order from origin to destination. example: [1, 4, 2, 5]. List all vertices on the path in order; return an empty list if no feasible path exists
- "resource_accumulations": dict[str, list[float]] — Cumulative consumption of each resource at each vertex upon arrival. example: {"1": [10.0, 0.0], "2": [25.5, 5.0], "3": [0.0, 0.0], "4": [15.0, 2.0], "5": [40.0, 10.0]}. List the cumulative values of all n vertices in the graph across each resource dimension; vertices not visited can usually be marked 0 or their initial value
Fill every value for every entity; keep the keys EXACTLY as above.
```

## Objective

**Direction**: min

**Formula**
```
min Σ_{(i,j) ∈ A} c[i][j] * x[i][j]
```

**Plain-language statement**
Minimize the sum of costs of all arcs included in the path.

**Rationale**
The objective is explicitly stated as minimizing the total arc cost of the valid path from vertex 1 to vertex n.
