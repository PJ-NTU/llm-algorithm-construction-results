# Hybrid Reentrant Shop Scheduling: mathematical model

Problem statement of the prototype's problem type used for this CO-Bench problem (decision variables, constraints, data field semantics, objective and I/O schema).

```text
A hybrid reentrant shop scheduling problem where N jobs must pass through three stages: initialization on one of M identical primary machines, setup on a single remote server, and main processing on the same primary machine. Jobs follow a fixed natural order for initialization, but their sequence on the remote server and their assignment to primary machines for the main processing phase must be optimized. All indices start from 0.

== DECISION VARIABLES (authoritative; model with exactly these) ==
- y[j][k]: Binary variable indicating if job j is assigned to primary machine k  |  y[j][k] ∈ {0,1}
- sigma[j]: The position (rank) of job j in the processing sequence on the remote server  |  sigma[j] ∈ {0, ..., N-1}
- s1[j]: Start time of the initialization phase for job j  |  s1[j] ≥ 0
- s2[j]: Start time of the setup phase for job j on the remote server  |  s2[j] ≥ 0
- s3[j]: Start time of the main processing phase for job j  |  s3[j] ≥ 0
- C_max: The completion time of the last job to finish its main processing phase  |  C_max ≥ 0

== CONSTRAINTS (authoritative; ALL must be modeled) ==
- [hard] Each job must be assigned to exactly one primary machine.  |  sum(y[j][k] for k in range(M)) == 1, ∀j
- [hard] The initialization phase on primary machines follows a fixed natural order (list scheduling).  |  s1[j] >= s1[j-1] + p1[j-1] if y[j][k] == 1 and y[j-1][k] == 1, ∀j > 0, ∀k
- [hard] A primary machine can process only one initialization phase at a time.  |  s1[j] >= s1[i] + p1[i] or s1[i] >= s1[j] + p1[j] if y[i][k] == 1 and y[j][k] == 1, ∀i != j
- [hard] The setup phase for a job can only start after its initialization phase is completed.  |  s2[j] >= s1[j] + p1[j], ∀j
- [hard] The remote server processes setup phases one at a time according to the permutation order.  |  s2[j] >= s2[i] + p2[i] if sigma[j] > sigma[i], ∀i, j
- [hard] The main processing phase can only start after the setup phase is completed.  |  s3[j] >= s2[j] + p2[j], ∀j
- [hard] On each primary machine, the main processing phase follows the natural order and cannot overlap with other main processing or initialization phases.  |  s3[j] >= s3[j-1] + p3[j-1] if y[j][k] == 1 and y[j-1][k] == 1, ∀j > 0, ∀k; s3[j] >= s1[i] + p1[i] if y[j][k] == 1 and y[i][k] == 1, ∀i, j
- [hard] The makespan must be greater than or equal to the completion time of the main processing phase of every job.  |  C_max >= s3[j] + p3[j], ∀j
- [hard] The permutation positions on the remote server must be unique and cover all jobs.  |  AllDifferent(sigma[j]), ∀j

== DATA FIELD SEMANTICS (key meanings and axis conventions) ==
- num_jobs: N; unit=jobs
- num_machines: M; unit=machines
- init_processing_times: p1; unit=time; axis: job_index
- setup_processing_times: p2; unit=time; axis: job_index
- main_processing_times: p3; unit=time; axis: job_index

== OBJECTIVE (authoritative; the solver MUST optimize exactly this) ==
Sense: minimize
Formula: min C_max
Plain: Minimize the maximum completion time (makespan) among all jobs.
If any earlier sentence conflicts with this section, this section wins.

== IO SCHEMA ==
{"num_jobs": int, "num_machines": int, "init_processing_times": [float, ...], "setup_processing_times": [float, ...], "main_processing_times": [float, ...]}

OUTPUT SCHEMA — the solution dict you return MUST use EXACTLY these keys and shapes (this was confirmed upstream; do NOT rename, add, or drop keys):
- "machine_assignment": dict[str,int] — Main machine number each task is assigned to. example: {"job_0": 0, "job_1": 1, "job_2": 0}. Includes the machine assignment of all n tasks
- "server_sequence": list[str] — Processing order of tasks in the preparation stage on the remote server. example: ["job_1", "job_0", "job_2"]. Includes the ordering of all n tasks
- "start_times": dict[str,list[float]] — Start times of the three stages (initialization, preparation, main processing) of each task. example: {"job_0": [0.0, 10.0, 25.0], "job_1": [0.0, 15.0, 30.0], "job_2": [5.0, 20.0, 40.0]}. Includes all n tasks; each task maps to a list of length 3 representing s1, s2, s3 respectively
- "makespan": float — Total completion time of all tasks. example: 55.5. Objective function value of the optimal solution
Fill every value for every entity; keep the keys EXACTLY as above.
```

## Objective

**Direction**: min

**Formula**
```
min C_max
```

**Plain-language statement**
Minimize the maximum completion time (makespan) among all jobs.

**Rationale**
The goal is to optimize system efficiency by completing the entire batch of jobs as early as possible.
