# Job shop scheduling: mathematical model

Problem statement of the prototype's problem type used for this CO-Bench problem (decision variables, constraints, data field semantics, objective and I/O schema).

```text
The Job Shop Scheduling Problem (JSSP) involves scheduling n_jobs on n_machines. Each job consists of a sequence of operations that must be processed in a specific order on designated machines. Each machine can process only one operation at a time. The goal is to determine the start times of all operations to minimize the total completion time (makespan). All indices for jobs, machines, and operations start from 0.

== DECISION VARIABLES (authoritative; model with exactly these) ==
- s[j][k]: The start time of the k-th operation of job j  |  s[j][k] ≥ 0, continuous
- z[j][j'][m]: Binary indicator whether job j precedes job j' on machine m  |  z[j][j'][m] ∈ {0,1}
- Cmax: The maximum completion time across all jobs (makespan)  |  Cmax ≥ 0, continuous

== CONSTRAINTS (authoritative; ALL must be modeled) ==
- [hard] Intra-job operation order constraint: operation k+1 of each job can start only after operation k is completed.  |  s[j][k+1] ≥ s[j][k] + p[j][k], ∀j ∈ {0...n_jobs-1}, k ∈ {0...n_machines-2}
- [hard] Machine non-overlap constraint (disjunctive constraint): any two operations on the same machine cannot be processed simultaneously and must be sequenced.  |  s[j][k] ≥ s[j'][k'] + p[j'][k'] - M * z[j][j'][m], s[j'][k'] ≥ s[j][k] + p[j][k] - M * (1 - z[j][j'][m]), where m = machine(j,k) = machine(j',k')
- [hard] Makespan definition: Cmax must be greater than or equal to the completion time of the last operation of every job.  |  Cmax ≥ s[j][n_machines-1] + p[j][n_machines-1], ∀j ∈ {0...n_jobs-1}

== DATA FIELD SEMANTICS (key meanings and axis conventions) ==
- n_jobs: n_jobs; unit=count; axis: scalar
- n_machines: n_machines; unit=count; axis: scalar
- jobs: machine(j,k); unit=ID; axis: job_index × operation_index (extracting the first element of the pair)

== OBJECTIVE (authoritative; the solver MUST optimize exactly this) ==
Sense: minimize
Formula: min Cmax
Plain: Minimize the maximum completion time (makespan) of all jobs.
If any earlier sentence conflicts with this section, this section wins.

== IO SCHEMA ==
{"n": int, "n_jobs": int, "n_machines": int, "jobs": [[[int, ...], ...], ...]}

OUTPUT SCHEMA — the solution dict you return MUST use EXACTLY these keys and shapes (this was confirmed upstream; do NOT rename, add, or drop keys):
- "start_times": list[list[float]] — Start time of each operation in each job. example: [[0.0, 10.0, 25.0], [5.0, 15.0, 30.0]]. Includes all n_jobs jobs; each job contains the start times of all n_machines operations
- "machine_precedence": dict[str, list[list[int]]] — Processing order between jobs on the same machine (0 means j' precedes j, 1 means j precedes j'). example: {"machine_0": [[0, 1], [0, 0]], "machine_1": [[0, 0], [1, 0]]}. Provide an n_jobs × n_jobs matrix for each machine, recording the precedence relation of all job pairs
- "makespan": float — Makespan of the whole schedule. example: 45.5. A single value, the time at which the last operation finishes
Fill every value for every entity; keep the keys EXACTLY as above.
```

## Objective

**Direction**: min

**Formula**
```
min Cmax
```

**Plain-language statement**
Minimize the maximum completion time (makespan) of all jobs.

**Rationale**
In manufacturing and scheduling, minimizing makespan improves throughput and machine utilization by ensuring all tasks are finished as early as possible.
