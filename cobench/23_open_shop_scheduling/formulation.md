# Open shop scheduling: mathematical model

Problem statement of the prototype's problem type used for this CO-Bench problem (decision variables, constraints, data field semantics, objective and I/O schema).

```text
In an Open Shop Scheduling Problem (OSSP), there are N jobs and M machines. Each job must be processed on every machine exactly once. Unlike Job Shop Scheduling, there is no fixed sequence of machines for any job; the processing order is entirely flexible. The constraints are: (1) a job can be processed on at most one machine at a time, and (2) a machine can process at most one job at a time. Operations are non-preemptive. The goal is to minimize the makespan (the time when all jobs are completed). All indices for jobs j and machines m start from 0.

== DECISION VARIABLES (authoritative; model with exactly these) ==
- s[j][m]: The start time of job j on machine m  |  s[j][m] ≥ 0, continuous (or non-negative integer)
- zJ[j][m1][m2]: Binary indicator for the relative order of two operations of the same job j on different machines m1 and m2  |  zJ[j][m1][m2] ∈ {0,1}
- zM[m][j1][j2]: Binary indicator for the relative order of two different jobs j1 and j2 on the same machine m  |  zM[m][j1][j2] ∈ {0,1}
- Cmax: The maximum completion time (makespan) across all jobs and machines  |  Cmax ≥ 0

== CONSTRAINTS (authoritative; ALL must be modeled) ==
- [hard] The completion time of each job on each machine cannot exceed the makespan Cmax.  |  Cmax \ge s[j][m] + ptimes[j][m], \forall j \in \{0..n\_jobs-1\}, m \in \{0..n\_machines-1\}
- [hard] Operations of the same job on different machines cannot overlap (order is free, determined by zJ).  |  s[j][m2] \ge s[j][m1] + ptimes[j][m1] - M \cdot zJ[j][m1][m2] \text{ and } s[j][m1] \ge s[j][m2] + ptimes[j][m2] - M \cdot (1 - zJ[j][m1][m2]), \forall j, m1 < m2
- [hard] Operations of different jobs on the same machine cannot overlap (order is free, determined by zM).  |  s[j2][m] \ge s[j1][m] + ptimes[j1][m] - M \cdot zM[m][j1][j2] \text{ and } s[j1][m] \ge s[j2][m] + ptimes[j2][m] - M \cdot (1 - zM[m][j1][j2]), \forall m, j1 < j2

== DATA FIELD SEMANTICS (key meanings and axis conventions) ==
- n_jobs: n_jobs; unit=count; axis: scalar
- n_machines: n_machines; unit=count; axis: scalar
- ptimes: M; unit=time; axis: sum of all elements in ptimes

== OBJECTIVE (authoritative; the solver MUST optimize exactly this) ==
Sense: minimize
Formula: minimize Cmax
Plain: Minimize the total time required to complete all operations across all jobs and machines.
If any earlier sentence conflicts with this section, this section wins.

== IO SCHEMA ==
{"n": int, "n_jobs": int, "n_machines": int, "ptimes": [[int, ...], ...]}

OUTPUT SCHEMA — the solution dict you return MUST use EXACTLY these keys and shapes (this was confirmed upstream; do NOT rename, add, or drop keys):
- "start": list[list[int]] — Start time of the operation of each job on each machine. example: [[0, 10, 5], [5, 0, 12]]. Must have n_jobs rows and n_machines columns, listing the start times of all jobs on all machines, even if the start time is 0.
Fill every value for every entity; keep the keys EXACTLY as above.
```

## Objective

**Direction**: min

**Formula**
```
minimize Cmax
```

**Plain-language statement**
Minimize the total time required to complete all operations across all jobs and machines.

**Rationale**
In scheduling, minimizing the makespan is the standard objective to maximize throughput and resource utilization.
