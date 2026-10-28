# mehrotra1996 · Mathematical model

> Graph coloring problem
>
> Sense: min

---

Given an undirected graph with N vertices and a set of edges, assign a color (represented by an integer from 0 to K-1) to each vertex such that no two adjacent vertices share the same color. The goal is to minimize the total number of distinct colors used. All indices for vertices and colors start from 0.

== DECISION VARIABLES (authoritative; model with exactly these) ==
- x[v][c]: A binary variable indicating whether vertex v is assigned color c.  |  x[v][c] ∈ {0,1}, v ∈ {0..N-1}, c ∈ {0..K-1}
- y[c]: A binary auxiliary variable indicating whether color c is used by at least one vertex.  |  y[c] ∈ {0,1}, c ∈ {0..K-1}

== CONSTRAINTS (authoritative; ALL must be modeled) ==
- [hard] Each vertex must be assigned exactly one color.  |  Σ_{c=0}^{K-1} x[v][c] = 1, ∀v ∈ {0..N-1}
- [hard] Two adjacent vertices must not be assigned the same color.  |  x[u][c] + x[v][c] ≤ 1, ∀[u,v] ∈ E, ∀c ∈ {0..K-1}
- [hard] If any vertex uses color c, the auxiliary variable y[c] must be set to 1.  |  y[c] ≥ x[v][c], ∀v ∈ {0..N-1}, ∀c ∈ {0..K-1}
- [hard] (Optional) symmetry-breaking constraint: color indices should be used consecutively, i.e., lower-indexed colors are used first.  |  y[c] ≥ y[c+1], ∀c ∈ {0..K-2}

== DATA FIELD SEMANTICS (key meanings and axis conventions) ==
- n: K; unit=upper bound on the number of colors; axis: scalar
- edges: E; unit=edge set; axis: edge index × [vertex u, vertex v]

== OBJECTIVE (authoritative; the solver MUST optimize exactly this) ==
Sense: minimize
Formula: min Σ_{c=0}^{K-1} y[c]
Plain: Minimize the total count of distinct colors used across all vertices.
If any earlier sentence conflicts with this section, this section wins.

== IO SCHEMA ==
{"n": int, "edges": [[int, ...], ...]}

OUTPUT SCHEMA — the solution dict you return MUST use EXACTLY these keys and shapes (this was confirmed upstream; do NOT rename, add, or drop keys):
- "coloring": list[int] — final color index assigned to each vertex. example: [0, 1, 0, 2, 1]. List the color assignment of all n vertices; the list index corresponds to the vertex index
Fill every value for every entity; keep the keys EXACTLY as above.