# furini2021 · Mathematical model

> Edge interdiction maximum clique problem
>
> Sense: min

---

Mathematical model of the problem type as revised for the portfolio updated by continued evolution (parameters, decision variables, objective and constraints).

Given an undirected graph G=(V, E) and an interdiction budget K, the goal is to select a subset of edges to remove (interdict) such that the size of the maximum clique in the remaining graph is minimized. A clique is a subset of vertices where every pair is connected by an edge. The problem is a bilevel optimization task where the leader removes edges and the follower identifies the maximum clique in the resulting graph. Indices for vertices and edges start from 0.

== DECISION VARIABLES (authoritative; model with exactly these) ==
- w[u][v]: Binary variable indicating if the edge between vertex u and vertex v is interdicted (removed).  |  w[u][v] ∈ {0,1} for all (u,v) ∈ E
- x[v]: Binary variable indicating if vertex v is included in the maximum clique of the remaining graph.  |  x[v] ∈ {0,1} for all v ∈ V
- theta: An auxiliary variable representing the size of the maximum clique in the interdicted graph.  |  theta ∈ Z+

== CONSTRAINTS (authoritative; ALL must be modeled) ==
- [hard] The total number of edges removed must not exceed the interdiction budget.  |  sum_{(u,v) in E} w[u][v] <= k
- [hard] The size of the maximum clique in the interdicted graph is the sum of the vertices selected by the follower.  |  theta >= sum_{v in V} x[v]
- [hard] In the interdicted graph, two vertices can both be in a clique only if the edge connecting them was not removed.  |  x[u] + x[v] <= 2 - w[u][v], forall (u,v) in E
- [hard] Two vertices cannot both be in a clique if there was no edge between them in the original graph.  |  x[u] + x[v] <= 1, forall (u,v) in E_complement

== DATA FIELD SEMANTICS (key meanings and axis conventions) ==
- vertices: V; axis: vertex index
- edges: E; axis: edge list
- interdiction_budget_k: k; unit=edges
- num_vertices: num_vertices; unit=count
- num_edges: num_edges; unit=count

== OBJECTIVE (authoritative; the solver MUST optimize exactly this) ==
Sense: minimize
Formula: min theta
Plain: Minimize the size of the maximum clique in the graph remaining after edge removal.
If any earlier sentence conflicts with this section, this section wins.

== IO SCHEMA ==
{"actual_edge_density": float, "budget_mu_percent": any, "budget_rule": str, "edges": [[int, ...], ...], "generation_seed": int, "generation_style": str, "interdiction_budget_k": int, "num_edges": int, "num_vertices": int, "problem": str, "target_edge_density": float, "vertices": [int, ...]}

OUTPUT SCHEMA — the solution dict you return MUST use EXACTLY these keys and shapes (this was confirmed upstream; do NOT rename, add, or drop keys):
- "interdiction_status": list[int] — for each edge of the original graph, decide whether to remove (interdict) it. example: [0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]. Following the order of the edges list in the input data, gives the interdiction status of all 17 edges (1 = removed, 0 = kept)
- "clique_membership": list[int] — which vertices form the maximum clique in the graph after interdiction. example: [1, 1, 0, 0, 0, 0, 0, 0]. Following the order of the vertices list in the input data, gives the selection status of all 8 vertices (1 = in the clique, 0 = not in it)
- "max_clique_size": int — size of the maximum clique in the graph after interdiction. example: 2. A single value, the optimal value of the objective function theta
Fill every value for every entity; keep the keys EXACTLY as above.
