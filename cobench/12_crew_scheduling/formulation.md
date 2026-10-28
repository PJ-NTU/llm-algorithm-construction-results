# Crew scheduling: mathematical model

Problem statement of the prototype's problem type used for this CO-Bench problem (decision variables, constraints, data field semantics, objective and I/O schema).

```text
Assign N tasks to at most K crews such that each task is covered exactly once, tasks within a crew do not overlap, and transitions are valid. The total duty time for each crew (last task finish - first task start) must not exceed a limit T. Indices start from 0. Let N be the number of tasks and K be the maximum number of crews.

== DECISION VARIABLES (authoritative; model with exactly these) ==
- x[i][j][k]: Binary variable indicating if crew k moves directly from task i to task j  |  x[i][j][k] ∈ {0,1}
- y[i][k]: Binary variable indicating if task i is assigned to crew k  |  y[i][k] ∈ {0,1}
- start_node[k]: The index of the first task performed by crew k  |  integer ∈ {0...N-1}
- end_node[k]: The index of the last task performed by crew k  |  integer ∈ {0...N-1}

== CONSTRAINTS (authoritative; ALL must be modeled) ==
- [hard] Each task must be assigned to exactly one crew.  |  ∑_{k=0}^{K-1} y_{ik} = 1, ∀i ∈ {0...N-1}
- [hard] Task flow balance within a crew: if task i is assigned to crew k, it must have exactly one successor task j (unless it is the last task).  |  ∑_{j=0, j≠i}^{N-1} x_{ijk} ≤ y_{ik}, ∀i, k; ∑_{i=0, i≠j}^{N-1} x_{ijk} ≤ y_{jk}, ∀j, k
- [hard] Consecutive tasks within the same crew must not overlap in time.  |  x_{ijk} = 1 ⇒ F_i ≤ S_j, ∀i, j, k
- [hard] A valid connection must exist between consecutive tasks (i.e. the corresponding path in the transition cost matrix is allowed).  |  x_{ijk} = 1 ⇒ V_{ij} = 1, ∀i, j, k
- [hard] The total duty time of each crew (from the start of its first task to the end of its last task) must not exceed the specified upper limit.  |  F_{end\_node[k]} - S_{start\_node[k]} ≤ L, ∀k
- [hard] The total number of crews used must not exceed K.  |  k < K

== DATA FIELD SEMANTICS (key meanings and axis conventions) ==
- transition_costs: C_{ij}; unit=cost; axis: task_i × task_j
- start_times: S_i; unit=time; axis: task_index
- finish_times: F_i; unit=time; axis: task_index
- valid_transitions: V_{ij}; unit=bool; axis: task_i × task_j
- max_duty_time: L; unit=time
- max_crews: K; unit=count
- num_tasks: N; unit=count

== OBJECTIVE (authoritative; the solver MUST optimize exactly this) ==
Sense: minimize
Formula: min ∑_{k=0}^{K-1} ∑_{i=0}^{N-1} ∑_{j=0}^{N-1} C_{ij} * x_{ijk}
Plain: Minimize the total transition costs across all crews and all consecutive task pairs.
If any earlier sentence conflicts with this section, this section wins.

== IO SCHEMA ==
{"num_tasks": int, "max_crews": int, "max_duty_time": float, "start_times": [float, ...], "finish_times": [float, ...], "transition_costs": [[float, ...], ...], "valid_transitions": [[int, ...], ...]}

OUTPUT SCHEMA — the solution dict you return MUST use EXACTLY these keys and shapes (this was confirmed upstream; do NOT rename, add, or drop keys):
- "crew_routes": list[list[int]] — sequence of tasks handled by each crew (in execution order). example: [[0, 3, 5], [1, 4], [2], []]. Lists the task routes of all K crews; a crew with no assigned tasks gets an empty list
- "start_tasks": list[int] — Start task index of each crew's duty period. example: [0, 1, 2, -1]. List the start tasks of all K crews; mark unused crews as -1
- "end_tasks": list[int] — End task index of each crew's duty period. example: [5, 4, 2, -1]. List the end tasks of all K crews; mark unused crews as -1
Fill every value for every entity; keep the keys EXACTLY as above.
```

## Objective

**Direction**: min

**Formula**
```
min ∑_{k=0}^{K-1} ∑_{i=0}^{N-1} ∑_{j=0}^{N-1} C_{ij} * x_{ijk}
```

**Plain-language statement**
Minimize the total transition costs across all crews and all consecutive task pairs.

**Rationale**
The problem explicitly states the goal is to minimize the sum of transition costs between consecutive tasks assigned to the same crew.
