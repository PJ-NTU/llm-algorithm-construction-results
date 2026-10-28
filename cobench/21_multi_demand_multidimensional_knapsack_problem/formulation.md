# Multi-Demand Multidimensional Knapsack problem: mathematical model

Mathematical model of the problem type as revised for the portfolio updated by continued evolution (parameters, decision variables, objective and constraints).

Select a subset of \(N\) items: the total resource consumption of the selected items must not exceed each capacity limit, and their total contribution must reach each demand lower bound; subject to this, maximise the total value.
## Parameters and sets
| Symbol | Meaning | Unit or range |
|---|---|---|
| \(N\) (field `n`) | Number of items | integer, \(N \ge 1\) |
| \(M\) (field `m`) | Number of capacity (resource) constraints | integer, \(M \ge 0\) |
| \(Q\) (field `q`) | Number of demand constraints | integer, \(Q \ge 0\) |
| \(J=\{0,\dots,N-1\}\) | Item index set, index \(j\) | numbered from 0 |
| \(I=\{0,\dots,M-1\}\) | Capacity constraint index set, index \(i\); empty when \(M=0\) | numbered from 0 |
| \(K=\{0,\dots,Q-1\}\) | Demand constraint index set, index \(k\); empty when \(Q=0\) | numbered from 0 |
| \(c_j\) (`cost_vector[j]`) | Value coefficient of item \(j\), length \(N\) | integer, may be positive, negative or 0; unit not given |
| \(a^{\le}_{i,j}\) (`A_leq[i][j]`) | Consumption of resource \(i\) by item \(j\), shape \(M\times N\) | integer, \(\ge 0\); unit not given |
| \(b^{\le}_i\) (`b_leq[i]`) | Available capacity of resource \(i\), length \(M\) | integer, \(\ge 0\); unit not given |
| \(a^{\ge}_{k,j}\) (`A_geq[k][j]`) | Contribution of item \(j\) to demand \(k\), shape \(Q\times N\) | integer, \(\ge 0\); unit not given |
| \(b^{\ge}_k\) (`b_geq[k]`) | Minimum required amount of demand \(k\), length \(Q\) | integer, \(\ge 0\); unit not given |
## Decision variables
| Symbol | Meaning | Range |
|---|---|---|
| \(x_j\) (`x[j]`), \(j \in J\) | Whether item \(j\) is selected: 1 selected, 0 not selected | \(\{0,1\}\) |
## Objective function
Maximise the sum of value coefficients of the selected items. This is the only objective; there is no secondary objective.
\[
\max \; \sum_{j=0}^{N-1} c_j\, x_j
\]
## Constraints
**C1 Resource capacity upper bound (hard constraint)**: for each resource \(i\), the total consumption of the selected items must not exceed that resource's available capacity.
\[
\sum_{j=0}^{N-1} a^{\le}_{i,j}\, x_j \le b^{\le}_i, \quad \forall i \in \{0,\dots,M-1\}
\]
**C2 Demand lower bound (hard constraint)**: for each demand \(k\), the total contribution of the selected items must reach at least that demand.
\[
\sum_{j=0}^{N-1} a^{\ge}_{k,j}\, x_j \ge b^{\ge}_k, \quad \forall k \in \{0,\dots,Q-1\}
\]
**C3 Selection is a 0-1 decision (hard constraint)**: each item is either selected whole or not selected; partial selection is not allowed.
\[
x_j \in \{0,1\}, \quad \forall j \in \{0,\dots,N-1\}
\]
## Solution and boundary conventions
- Complete solution: gives \(x_j \in \{0,1\}\) for all \(N\) items and satisfies C1-C3 simultaneously. The objective value is \(\sum_j c_j x_j\). All data are integers; constraint checks and the objective value are computed exactly in integers.
- Ties: feasible solutions with equal objective value are all optimal; there is no secondary ordering rule.
- No feasible solution: the input may have no feasible solution. In that case "no feasible solution" must be reported explicitly; a solution violating any hard constraint must not be output.
- When \(M=0\) there is no C1, and when \(Q=0\) there is no C2; the corresponding `A_leq`/`b_leq` or `A_geq`/`b_geq` are empty. \(N=0\) is not valid input.
- A demand constraint with \(b^{\ge}_k = 0\) is automatically satisfied. Selecting nothing (all \(x_j=0\)) is feasible only when \(Q=0\) or all \(b^{\ge}_k=0\).
- Dimensions must match (`cost_vector` has length \(N\), `A_leq` is \(M\times N\), `b_leq` has length \(M\), `A_geq` is \(Q\times N\), `b_geq` has length \(Q\)); otherwise the input is invalid.
