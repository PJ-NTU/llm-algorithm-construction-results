# Travelling salesman problem: mathematical model

Problem statement of the prototype's problem type used for this CO-Bench problem (decision variables, constraints, data field semantics, objective and I/O schema).

```text
Given an origin (base) and N target points, the drone must depart from the origin, visit each target point exactly once (it only needs to fly within a certain range of a target point, which can be treated as reaching the target point itself), and finally return to the origin; find the route with the shortest total flight distance. Indices start at 0; 0 denotes the origin and 1..N denote the target points. The distance matrix d[i][j] is known.

== DECISION VARIABLES (authoritative; model with exactly these) ==
- x[i][j]: whether the drone flies directly from point i to point j (i≠j)  |  x[i][j] ∈ {0,1}

== CONSTRAINTS (authoritative; ALL must be modeled) ==
- [hard] Each node (including the origin) must have exactly one outgoing arc, ensuring the drone leaves each node once.  |  ∑_{j=0, j≠i}^{N} x[i][j] = 1, ∀ i ∈ {0,1,…,N}
- [hard] Each node (including the origin) must have exactly one incoming arc, ensuring the drone arrives at each node once.  |  ∑_{i=0, i≠j}^{N} x[i][j] = 1, ∀ j ∈ {0,1,…,N}
- [hard] Avoid subtours that do not contain the origin, ensuring the route is a single complete tour.  |  u[i] - u[j] + 1 ≤ (N-1) * (1 - x[i][j]), ∀ i,j ∈ {1,…,N}, i≠j

== DATA FIELD SEMANTICS (key meanings and axis conventions) ==
- num_nodes: N
- distance_matrix: d; unit=km; axis: i,j node indices (0 is the start point, 1~N are the other nodes)

== OBJECTIVE (authoritative; the solver MUST optimize exactly this) ==
Sense: minimize
Formula: min ∑_{i=0}^{N} ∑_{j=0, j≠i}^{N} d[i][j] * x[i][j]
Plain: Minimize the sum of distances of all selected arcs, i.e., the total flight distance.
If any earlier sentence conflicts with this section, this section wins.

== IO SCHEMA ==
{"num_nodes": int, "distance_matrix": [[float, ...], ...]}

OUTPUT SCHEMA — the solution dict you return MUST use EXACTLY these keys and shapes (this was confirmed upstream; do NOT rename, add, or drop keys):
- "tour": list[int] — the order in which the drone visits the targets (including the start point), forming a closed loop that departs from and returns to the start point. example: [0, 3, 1, 2]. List all N+1 nodes (start point 0 plus N targets), with the start point first and every other target appearing exactly once; the return to the start point at the end is implicit
Fill every value for every entity; keep the keys EXACTLY as above.
```

## Objective

**Direction**: min

**Formula**
```
min ∑_{i=0}^{N} ∑_{j=0, j≠i}^{N} d[i][j] * x[i][j]
```

**Plain-language statement**
Minimize the sum of distances of all selected arcs, i.e., the total flight distance.

**Rationale**
The goal is to minimize the total flight distance for the drone to inspect all targets and return to the start point, so the total flight distance is minimized.
