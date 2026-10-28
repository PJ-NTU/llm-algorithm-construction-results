# Flow shop scheduling: mathematical model

Problem statement of the prototype's problem type used for this CO-Bench problem (decision variables, constraints, data field semantics, objective and I/O schema).

```text
In a Permutation Flow Shop Scheduling Problem (PFSP), N jobs must be processed on M machines in the same fixed order (from machine 0 to M-1). Each machine can process only one job at a time, and a job can only start on machine i after it has finished on machine i-1. The goal is to find a single permutation of jobs that minimizes the total completion time (makespan) of the last job on the last machine. All indices for jobs, machines, and positions start from 0.

== DECISION VARIABLES (authoritative; model with exactly these) ==
- x[j][k]: A binary variable indicating whether job j is assigned to the k-th position in the processing sequence.  |  x[j][k] ∈ {0,1}, j ∈ {0..N-1}, k ∈ {0..N-1}

== CONSTRAINTS (authoritative; ALL must be modeled) ==
- [hard] Each job must be assigned to exactly one processing position.  |  Σ_{k=0}^{n-1} x[j][k] = 1, ∀j ∈ {0..n-1}
- [hard] Each processing position must hold exactly one job.  |  Σ_{j=0}^{n-1} x[j][k] = 1, ∀k ∈ {0..n-1}
- [hard] The completion time of a job on a machine is bounded by the completion time of the same job on the previous machine and by the completion time of the job in the previous position on the current machine.  |  C[k][i] ≥ C[k-1][i] + Σ_{j=0}^{n-1} (x[j][k] · proc[j][i]) and C[k][i] ≥ C[k][i-1] + Σ_{j=0}^{n-1} (x[j][k] · proc[j][i]), ∀k ∈ {0..n-1}, ∀i ∈ {0..n_machines-1}
- [hard] The total completion time (Makespan) must be greater than or equal to the completion time of the job in the last position on the last machine.  |  Cmax ≥ C[n-1][n_machines-1]

== DATA FIELD SEMANTICS (key meanings and axis conventions) ==
- n: n; unit=count; axis: scalar
- n_machines: n_machines; unit=machines; axis: scalar
- proc: proc[j][i]; unit=time units; axis: job × machine

== OBJECTIVE (authoritative; the solver MUST optimize exactly this) ==
Sense: minimize
Formula: min Cmax
Plain: Minimize the makespan, which is the completion time of the job at the last position (N-1) on the last machine (M-1).
If any earlier sentence conflicts with this section, this section wins.

== IO SCHEMA ==
{"n": int, "n_machines": int, "proc": [[float, ...], ...]}

OUTPUT SCHEMA — the solution dict you return MUST use EXACTLY these keys and shapes (this was confirmed upstream; do NOT rename, add, or drop keys):
- "perm": list[int] — Processing order permutation of jobs (the job numbers processed in order from position 0 to position n-1). example: [2, 0, 3, 1]. Must contain the indices of all n jobs and be a permutation of 0 to n-1
Fill every value for every entity; keep the keys EXACTLY as above.
```

## Objective

**Direction**: min

**Formula**
```
min Cmax
```

**Plain-language statement**
Minimize the makespan, which is the completion time of the job at the last position (N-1) on the last machine (M-1).

**Rationale**
The objective of flow shop scheduling is typically to maximize throughput or minimize idle time, both of which are achieved by minimizing the total time required to complete all tasks (makespan).
