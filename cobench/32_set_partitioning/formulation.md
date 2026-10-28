# Set partitioning: mathematical model

Mathematical model of the problem type as revised for the portfolio updated by continued evolution (parameters, decision variables, objective and constraints).

Select a subset of the \(n\) candidate schemes so that every task is covered by exactly one selected scheme, with no task missed or covered twice, while minimising the total cost of the selected schemes.
## Parameters and sets
| Symbol | Meaning | Unit or range |
|---|---|---|
| \(m\) | Number of tasks | count; non-negative integer, \(m=0\) allowed |
| \(n\) | Number of candidate schemes | count; non-negative integer, \(n=0\) allowed |
| \(R=\{0,1,\dots,m-1\}\) | Set of tasks, index \(r\) (0-based) | empty when \(m=0\) |
| \(J=\{0,1,\dots,n-1\}\) | Set of schemes, index \(j\) (0-based) | empty when \(n=0\) |
| \(c_j\) (costs[j]) | Cost of selecting scheme \(j\) | currency units; finite real, may be fractional, \(c_j \ge 0\) (zero allowed, negative not allowed) |
| \(S_j\) (covers[j]) | Set of task indices covered by scheme \(j\) | \(S_j \subseteq R\); may be empty; no index may appear twice in the list, and indices must lie between \(0\) and \(m-1\), otherwise the input is invalid |
| \(a_{rj}\) | Coverage relation: 1 if \(r \in S_j\), otherwise 0 | \(\{0,1\}\), derived from \(S_j\) |
## Decision variables
| Symbol | Meaning | Range |
|---|---|---|
| \(x_j\) | Whether scheme \(j\) is selected: 1 means selected, 0 means not selected | \(x_j \in \{0,1\},\ \forall j \in J\) |
## Objective function
Minimise the total cost of the selected schemes; this is the only objective, with no secondary objective.
\[ \min \sum_{j \in J} c_j \, x_j \]
## Constraints
**C1 Each task is covered exactly once (hard constraint)**: every task must be covered by one and only one selected scheme, neither missed nor duplicated.
\[ \sum_{j \in J} a_{rj}\, x_j \;=\; \sum_{j \in J:\ r \in S_j} x_j \;=\; 1, \qquad \forall r \in R \]
## Solutions and boundary conventions
- Complete solution: give \(x_j \in \{0,1\}\) for every scheme \(j \in J\) (i.e. the set of selected schemes) such that C1 holds for all tasks; partial or duplicate coverage is not allowed.
- No feasible solution: if no selection satisfies C1 (e.g. some task belongs to no \(S_j\), or \(n=0\) and \(m>0\)), the instance is still valid input and the result must explicitly report "no feasible solution" without giving a scheme selection.
- \(m=0\): C1 is empty, selecting nothing is feasible, and the optimal total cost is 0.
- Empty-cover scheme (\(S_j=\emptyset\)): a valid candidate; it does not affect C1, only the cost.
- Invalid input: \(c_j<0\), duplicate indices within \(S_j\), indices outside \(\{0,\dots,m-1\}\), or costs or covers whose length differs from \(n\).
- Numerical tolerance: total costs are compared with an absolute tolerance of \(10^{-6}\). The reported total cost is consistent if it differs from \(\sum_j c_j x_j\) recomputed from the selected schemes by at most \(10^{-6}\); two feasible solutions whose total costs differ by at most \(10^{-6}\) are equally good.
- Ties for the optimum: any optimal solution is acceptable.
