# Maximal independent set: mathematical model

Problem statement of the prototype's problem type used for this CO-Bench problem (decision variables, constraints, data field semantics, objective and I/O schema).

```text
Given an undirected graph G = (V, E) with N vertices and M edges, the goal is to select a subset of vertices such that no two selected vertices are connected by an edge, while maximizing the total number of selected vertices. Indices for vertices start from 0.

== DECISION VARIABLES (authoritative; model with exactly these) ==
- x[i]: A binary indicator whether vertex i is included in the independent set S  |  x[i] ∈ {0,1}, i ∈ {0, 1, ..., N-1}

== CONSTRAINTS (authoritative; ALL must be modeled) ==
- [hard] If two vertices are connected by an edge, at most one of them can be included in the independent set.  |  x[i] + x[j] \le 1, \forall (i, j) \in E

== DATA FIELD SEMANTICS (key meanings and axis conventions) ==
- num_vertices: N; axis: 1
- edges: E; axis: edge_index

== OBJECTIVE (authoritative; the solver MUST optimize exactly this) ==
Sense: maximize
Formula: max Σ_{i=0}^{N-1} x[i]
Plain: Maximize the total number of vertices selected to be in the independent set.
If any earlier sentence conflicts with this section, this section wins.

== IO SCHEMA ==
{"num_vertices": int, "edges": [[int, ...], ...]}

OUTPUT SCHEMA — the solution dict you return MUST use EXACTLY these keys and shapes (this was confirmed upstream; do NOT rename, add, or drop keys):
- "vertex_selection": list[int] — Whether each vertex is included in the maximum independent set. example: [1, 0, 1, 0, 0]. List the selection status of all N vertices in the graph (1 = selected, 0 = not selected)
Fill every value for every entity; keep the keys EXACTLY as above.
```

## Objective

**Direction**: max

**Formula**
```
max Σ_{i=0}^{N-1} x[i]
```

**Plain-language statement**
Maximize the total number of vertices selected to be in the independent set.

**Rationale**
The objective of the Maximum Independent Set problem is to find the largest possible subset of vertices that satisfies the independence constraint.
