# Euclidean Steiner problem: mathematical model

Problem statement of the prototype's problem type used for this CO-Bench problem (decision variables, constraints, data field semantics, objective and I/O schema).

```text
Given a set of $N$ terminal points in a 2D plane, the goal is to determine the coordinates of a set of $M$ additional points (Steiner points) such that the total length of the Minimum Spanning Tree (MST) connecting all $N+M$ points is minimized. The distance between any two points is the Euclidean distance. The number of Steiner points $M$ is typically at most $N-2$. Indices for terminals and Steiner points start from 0.

== DECISION VARIABLES (authoritative; model with exactly these) ==
- s[j]: The coordinates $(x_j, y_j)$ of the $j$-th Steiner point  |  s[j] ∈ ℝ², j ∈ {0, ..., M-1}
- e[u][v]: Binary indicator whether an edge exists between point $u$ and point $v$ in the spanning tree, where $u, v$ belong to the union of terminals and Steiner points  |  e[u][v] ∈ {0, 1}

== CONSTRAINTS (authoritative; ALL must be modeled) ==
- [hard] The selected edges must form a connected graph that spans all original terminals.  |  The graph G=(V, E) where V = P ∪ S and E = { (u,v) | e[u][v]=1 } must contain a path between any two terminals p_i, p_k ∈ P
- [hard] The total number of selected edges must be exactly one less than the total number of points (terminals plus Steiner points) to ensure a tree structure without cycles.  |  ∑_{u,v: u<v} e[u][v] = (|P| + M) - 1
- [hard] The number of Steiner points used is limited by a predefined maximum M.  |  j \in \{0, \dots, M-1\}

== DATA FIELD SEMANTICS (key meanings and axis conventions) ==
- points: y_u; unit=coordinate; axis: y-coordinate of terminal or Steiner point
- max_steiner_points: M; unit=count

== OBJECTIVE (authoritative; the solver MUST optimize exactly this) ==
Sense: minimize
Formula: min ∑_{u,v} e[u][v] * sqrt((x_u - x_v)^2 + (y_u - y_v)^2)
Plain: Minimize the sum of Euclidean distances of all edges selected to form the Minimum Spanning Tree over the union of original terminals and Steiner points.
If any earlier sentence conflicts with this section, this section wins.

== IO SCHEMA ==
{"points": [[float, ...], ...], "max_steiner_points": int}

OUTPUT SCHEMA — the solution dict you return MUST use EXACTLY these keys and shapes (this was confirmed upstream; do NOT rename, add, or drop keys):
- "steiner_points": list[tuple[float,float]] — List of coordinates of the added Steiner points. example: [[1.5, 2.3], [4.0, 5.1]]. List the coordinates of all Steiner points selected to build the minimum spanning tree
- "edges": list[tuple[int,int]] — Set of edges connecting all points (terminals and Steiner points). example: [[0, 3], [3, 1], [3, 4], [4, 2]]. List all edges forming the minimum spanning tree; each edge consists of the indices of two points (indices 0 to N-1 are the original terminals, N onward are Steiner points)
Fill every value for every entity; keep the keys EXACTLY as above.
```

## Objective

**Direction**: min

**Formula**
```
min ∑_{u,v} e[u][v] * sqrt((x_u - x_v)^2 + (y_u - y_v)^2)
```

**Plain-language statement**
Minimize the sum of Euclidean distances of all edges selected to form the Minimum Spanning Tree over the union of original terminals and Steiner points.

**Rationale**
The objective of the Steiner Tree problem is to minimize the total construction cost (length) of the network connecting all required terminals.
