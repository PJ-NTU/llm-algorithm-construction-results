# schwerdfeger2016 · Mathematical model

> Identical parallel machine scheduling with workload balancing
>
> Sense: min

---

The problem involves assigning a set of N independent jobs, each with a processing time p_j, to M identical parallel machines. The goal is to balance the workload across machines by minimizing the normalized sum of squared workload deviations (NSSWD). Each job must be assigned to exactly one machine, and each machine processes its assigned jobs without interruption. Indices for jobs j and machines i start from 0.

== DECISION VARIABLES (authoritative; model with exactly these) ==
- x[i][j]: Binary indicator whether job j is assigned to machine i  |  x[i][j] ∈ {0,1}, i ∈ {0,...,num_machines-1}, j ∈ {0,...,num_jobs-1}
- C[i]: Total completion time (workload) of machine i  |  C[i] ≥ 0, continuous (derived from integer processing times)

== CONSTRAINTS (authoritative; ALL must be modeled) ==
- [hard] Each job must be assigned to exactly one machine.  |  sum_{i=0}^{num_machines-1} x[i][j] = 1, \forall j \in {0,...,num_jobs-1}
- [hard] The completion time of each machine is the sum of the processing times of all jobs assigned to it.  |  C[i] = sum_{j=0}^{num_jobs-1} p[j] * x[i][j], \forall i \in {0,...,num_machines-1}

== DATA FIELD SEMANTICS (key meanings and axis conventions) ==
- num_machines: num_machines; unit=machines; axis: scalar
- num_jobs: num_jobs; unit=jobs; axis: scalar
- processing_times: p[j]; unit=time units; axis: jobs
- average_machine_completion_time: average_machine_completion_time; unit=time units; axis: scalar

== OBJECTIVE (authoritative; the solver MUST optimize exactly this) ==
Sense: minimize
Formula: min (1 / average_machine_completion_time) * sqrt(sum_{i=0}^{num_machines-1} (C[i] - average_machine_completion_time)^2)
Plain: Minimize the normalized root sum of squared differences between each machine's total workload and the global average workload.
If any earlier sentence conflicts with this section, this section wins.

== IO SCHEMA ==
{"num_machines": int, "num_jobs": int, "processing_times": [int, ...], "total_processing_time": int, "average_machine_completion_time": float, "processing_time_range": [int, ...], "processing_time_distribution": str, "generation_iteration": int}

OUTPUT SCHEMA — the solution dict you return MUST use EXACTLY these keys and shapes (this was confirmed upstream; do NOT rename, add, or drop keys):
- "assignment": dict[str, list[int]] — list of job indices assigned to each machine. example: {"0": [0, 3, 4], "1": [1, 2]}. List all num_machines machines; keep an empty list for a machine with no assigned jobs
- "machine_completion_times": dict[str, float] — total completion time of each machine (i.e., the total processing time of its assigned jobs). example: {"0": 45.0, "1": 42.0}. List the completion times of all num_machines machines
Fill every value for every entity; keep the keys EXACTLY as above.