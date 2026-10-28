# Assignment problem: mathematical model

Problem statement of the prototype's problem type used for this CO-Bench problem (decision variables, constraints, data field semantics, objective and I/O schema).

```text
Given N workers and N tasks, where each worker i performing task j incurs a specific cost, the goal is to assign exactly one task to each worker and exactly one worker to each task such that the total cost is minimized. All indices for workers and tasks start from 0.

== DECISION VARIABLES (authoritative; model with exactly these) ==
- x[i][j]: A binary indicator variable that equals 1 if worker i is assigned to task j, and 0 otherwise.  |  x[i][j] ∈ {0,1}

== CONSTRAINTS (authoritative; ALL must be modeled) ==
- [hard] Each worker must be assigned to exactly one task.  |  Σ_{j=0}^{n-1} x[i][j] = 1, ∀i ∈ {0, ..., n-1}
- [hard] Each task must be completed by exactly one worker.  |  Σ_{i=0}^{n-1} x[i][j] = 1, ∀j ∈ {0, ..., n-1}

== DATA FIELD SEMANTICS (key meanings and axis conventions) ==
- n: n; axis: scalar
- cost: cost; axis: worker × task

== OBJECTIVE (authoritative; the solver MUST optimize exactly this) ==
Sense: minimize
Formula: minimize Σ_{i=0}^{n-1} Σ_{j=0}^{n-1} cost[i][j] * x[i][j]
Plain: The sum of the costs associated with all chosen worker-task assignments.
If any earlier sentence conflicts with this section, this section wins.

== IO SCHEMA ==
{"n": int, "cost": [[float, ...], ...]}

OUTPUT SCHEMA — the solution dict you return MUST use EXACTLY these keys and shapes (this was confirmed upstream; do NOT rename, add, or drop keys):
- "assignment": list[int] — task index assigned to each worker (an integer from 0 to n-1). example: [2, 0, 1]. The list length must equal the number of workers n; the value at index i is the task index assigned to worker i, and the list must be a complete permutation of 0 to n-1
Fill every value for every entity; keep the keys EXACTLY as above.
```

## Objective

**Direction**: min

**Formula**
```
minimize Σ_{i=0}^{n-1} Σ_{j=0}^{n-1} cost[i][j] * x[i][j]
```

**Plain-language statement**
The sum of the costs associated with all chosen worker-task assignments.

**Rationale**
The problem objective is to find the most cost-efficient configuration of assignments. By multiplying the cost of each pair by its selection status (0 or 1) and summing them up, we obtain the total cost to be minimized.
