# p-median - capacitated: mathematical model

Mathematical model of the problem type as revised for the portfolio updated by continued evolution (parameters, decision variables, objective and constraints).

Select exactly p of the N customers as medians (facilities), then assign each customer to one selected median. The demand served by each median may not exceed a uniform capacity limit, and the objective is to minimise the total distance from all customers to their assigned medians.
## Parameters and Sets
| Symbol | Meaning | Unit or range |
|---|---|---|
| \( N \) | Number of customers (data field `n`) | Integer, \( N \ge 1 \) |
| \( I = \{0,1,\dots,N-1\} \) | Customer set, index \( i \); also the set of median candidates, index \( j \) | Indices start at 0 |
| \( p \) | Number of medians that must be selected (data field `p`) | Integer, \( 1 \le p \le N \) |
| \( Q \) | Uniform capacity limit of each median (data field `capacity`) | Positive integer; unit not given, same as demand |
| \( q_i \) | Demand of customer \( i \) (data field `demands[i]`) | Non-negative integer (may be 0); unit not given |
| \( d_{ij} \) | Distance when customer \( i \) is assigned to median \( j \) (data field `distance_matrix[i][j]`; rows are customers, columns are median candidates) | Non-negative integer; unit not given; not required to be symmetric; diagonal \( d_{jj} \) not required to be 0 |
## Decision Variables
| Symbol | Meaning | Range |
|---|---|---|
| \( y_j \) | Whether customer \( j \) is selected as a median: 1 means selected, 0 means not selected | \( y_j \in \{0,1\},\ \forall j \in I \) |
| \( x_{ij} \) | Whether customer \( i \) is assigned to median \( j \): 1 means assigned, 0 means not assigned | \( x_{ij} \in \{0,1\},\ \forall i \in I,\ j \in I \) |
## Objective Function
Minimise the total distance from all customers to their assigned medians. This is the only objective; there are no secondary objectives or weights.
\[
\min \; \sum_{i=0}^{N-1} \sum_{j=0}^{N-1} d_{ij}\, x_{ij}
\]
## Constraints
**C1 Exactly p medians selected** (hard constraint): the number of customers selected as medians must equal exactly \( p \), no more and no fewer.
\[
\sum_{j=0}^{N-1} y_j = p
\]
**C2 Each customer assigned exactly once** (hard constraint): every customer must be assigned to one median and only one; no customer may be left out.
\[
\sum_{j=0}^{N-1} x_{ij} = 1, \quad \forall i \in I
\]
**C3 Assign only to selected medians** (hard constraint): when customer \( j \) is not selected as a median, no customer may be assigned to it.
\[
x_{ij} \le y_j, \quad \forall i \in I,\ \forall j \in I
\]
**C4 Median capacity limit** (hard constraint): the total demand of all customers assigned to median \( j \) may not exceed the uniform capacity \( Q \); an unselected \( j \) serves zero demand.
\[
\sum_{i=0}^{N-1} q_i\, x_{ij} \le Q\, y_j, \quad \forall j \in I
\]
## Solution and Boundary Conventions
- **Valid input**: \( N \ge 1 \), \( 1 \le p \le N \) (\( p = N \) is valid); `demands` has length \( N \) with non-negative integer elements; `distance_matrix` is an \( N \times N \) non-negative integer matrix; `capacity` is a positive integer. \( N = 0 \), \( p = 0 \) or \( p > N \) are invalid inputs.
- **Complete solution**: exactly \( p \) medians are selected, every customer is assigned to a selected median, and C1–C4 all hold. Partial coverage is not allowed, nor is violating capacity.
- **Medians are not forced to serve themselves**: the model has no constraint \( x_{jj} = y_j \). A selected median \( j \) may be assigned to another selected median \( k \), in which case \( q_j \) counts toward the capacity of \( k \) and the distance is \( d_{jk} \). A selected median may also serve no customers at all, including itself.
- **Distance values**: always computed from the values given in `distance_matrix[i][j]`, without assuming symmetry and without assuming \( d_{jj} = 0 \).
- **Infeasible instances**: inputs are not guaranteed to have a feasible solution, e.g. some \( q_i > Q \), \( \sum_i q_i > pQ \), or demands cannot be split and so do not fit. When no feasible solution truly exists, "no feasible solution" must be reported explicitly, and no plan violating any constraint may be output.
- **Tied optima**: there is no secondary objective. When several feasible plans have the minimum total distance, any one of them is acceptable; acceptance compares only feasibility and objective value.
