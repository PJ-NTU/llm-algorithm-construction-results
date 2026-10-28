# p-median - uncapacitated: mathematical model

Mathematical model of the problem type as revised for the portfolio updated by continued evolution (parameters, decision variables, objective and constraints).

Select exactly \(p\) distinct vertices out of \(N\) as medians (facilities); every vertex is automatically served by its nearest selected median, and the sum of shortest-path distances from all vertices to their nearest medians is minimised. This is a pure location problem: it only decides which vertices to select and introduces no assignment variables.

## Parameters and Sets

| Symbol | Meaning | Unit or range |
|---|---|---|
| \(N\) | Number of vertices (data field `n`) | Integer, \(N \ge 1\) |
| \(p\) | Number of medians to select exactly (data field `p`) | Integer, \(1 \le p \le N\) |
| \(V\) | Vertex set \(\{0,1,\dots,N-1\}\), indexed from 0 | \(N\) elements |
| \(i\) | Index of a served vertex, i.e. the origin of a distance | \(i \in V\) |
| \(j\) | Index of a candidate median, i.e. the destination of a distance | \(j \in V\) |
| \(d_{ij}\) | Data field `dist[i][j]` (\(N \times N\) matrix, first dimension origin, second dimension destination): precomputed shortest-path distance from vertex \(i\) to vertex \(j\); \(+\infty\) means \(j\) is unreachable from \(i\) | \(\mathbb{R}_{\ge 0} \cup \{+\infty\}\), integers also valid; unit not specified by the problem statement |
| \(R_i\) | Set of vertices reachable from vertex \(i\), \(\{j \in V : d_{ij} < +\infty\}\), derived from \(d\) | \(R_i \subseteq V\), and \(i \in R_i\) |

Input guarantees on the distance data (confirmed with the user):
- Zero diagonal: \(d_{ii} = 0,\ \forall i \in V\).
- Already a shortest-path closure, satisfying the directed triangle inequality (with the convention \(a + (+\infty) = +\infty\)): \(d_{ik} \le d_{ij} + d_{jk},\ \forall i,j,k \in V\).
- Symmetry is not required; \(d_{ij} \ne d_{ji}\) is allowed; \(d_{ij}\) is always used in the direction from \(i\) to \(j\).
- \(+\infty\) is the only unreachability marker; any finite value, however large, is treated as a reachable distance; \(+\infty\) is written as `null` in the `dist` of the input JSON (see input_schema.json).

## Decision Variables

| Symbol | Meaning | Range |
|---|---|---|
| \(y_j\) | Whether vertex \(j\) is selected as a median: 1 means selected, 0 means not selected | \(y_j \in \{0,1\},\ \forall j \in V\) |

The model has only this one set of decision variables; each vertex belongs to its nearest selected median, determined directly by the \(\min\) in the objective, and no assignment variables are introduced.

## Objective Function

Minimise the sum of distances from all vertices (including the selected medians themselves, whose own distance is \(d_{ii} = 0\)) to their nearest selected medians, i.e. the total service distance; this is the only objective, with no secondary objective.
\[ \min_{y} \; Z(y) = \sum_{i \in V} \; \min_{j \in V:\, y_j = 1} d_{ij} \]

## Constraints

**C1** (hard constraint) Exactly \(p\) medians are selected.
\[ \sum_{j \in V} y_j = p \]

**C2** (hard constraint) The selected medians are distinct: each vertex is selected at most once and no vertex may appear twice in the solution, guaranteed by the 0-1 domain.
\[ y_j \in \{0,1\} \quad \forall j \in V \]

**C3** (hard constraint) Every vertex must have finite distance to its nearest selected median, i.e. every vertex can reach at least one selected median (is covered); partial coverage is not allowed.
\[ \min_{j \in V:\, y_j = 1} d_{ij} < +\infty \quad \forall i \in V \]
Equivalent covering form:
\[ \sum_{j \in R_i} y_j \ge 1 \quad \forall i \in V \]

## Solution and Boundary Conventions

- Complete solution: a 0-1 vector \(y\) satisfying C1–C3, equivalent to a median set \(S = \{j \in V : y_j = 1\}\) of exactly \(p\) distinct vertex indices, \(\lvert S \rvert = p\), output as `{"medians": [...]}` (the list holds the vertex indices of \(S\) in any order; see solution_form.json). The solution contains only the locations, not the assignments; when a vertex is equidistant to several selected medians, which one it is assigned to does not affect the objective value.
- The objective value \(Z(y)\) is uniquely determined by the locations; when C3 holds it is a finite non-negative real number with the same unit as the distances.
- Valid input: \(N \ge 1\), \(1 \le p \le N\), distances satisfy the ranges and input guarantees above, and at least one location set satisfying C1–C3 is guaranteed to exist; instances with no feasible location set are invalid input, and the solution need not express "no feasible solution". The empty instance with \(N = 0\) is invalid.
- Degenerate cases: \(N = 1\) (then \(p = 1\)) and \(p = N\) are both valid input; the unique feasible solution is \(S = V\), with objective value \(0\).
- Ties: when several feasible location sets have the same objective value, any one attaining the minimum objective value is optimal.
