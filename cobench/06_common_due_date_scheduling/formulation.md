# Common due date scheduling: mathematical model

Problem statement of the prototype's problem type used for this CO-Bench problem (decision variables, constraints, data field semantics, objective and I/O schema).

```text
A set of N jobs must be scheduled on a single machine starting at time zero without idle time. All jobs share a common due date d. Each job j has a processing time p_j, an earliness penalty weight a_j, and a tardiness penalty weight b_j. The goal is to determine the processing sequence that minimizes the total weighted earliness and tardiness. Indices for jobs and positions range from 0 to N-1.

== DECISION VARIABLES (authoritative; model with exactly these) ==
- x[j][k]: Binary indicator if job j is assigned to the k-th position in the sequence  |  x[j][k] ∈ {0,1}
- C[k]: Completion time of the job assigned to position k  |  C[k] ≥ 0, continuous
- E[j]: Earliness of job j  |  E[j] ≥ 0, continuous
- T[j]: Tardiness of job j  |  T[j] ≥ 0, continuous

== CONSTRAINTS (authoritative; ALL must be modeled) ==
- [hard] Each job must be assigned to exactly one position in the sequence.  |  sum_{k=0}^{n-1} x[j][k] = 1, ∀j ∈ {0,...,n-1}
- [hard] Each position in the sequence must be occupied by exactly one job.  |  sum_{j=0}^{n-1} x[j][k] = 1, ∀k ∈ {0,...,n-1}
- [hard] The completion time of the first position is the processing time of the job assigned to it.  |  C[0] = sum_{j=0}^{n-1} processing_times[j] * x[j][0]
- [hard] The completion time of any subsequent position is the completion time of the previous position plus the processing time of the current job.  |  C[k] = C[k-1] + sum_{j=0}^{n-1} processing_times[j] * x[j][k], ∀k ∈ {1,...,n-1}
- [hard] The earliness of a job is the maximum of zero and the difference between the due date and its completion time.  |  E[j] >= due_date - sum_{k=0}^{n-1} C[k] * x[j][k], ∀j ∈ {0,...,n-1}
- [hard] The tardiness of a job is the maximum of zero and the difference between its completion time and the due date.  |  T[j] >= sum_{k=0}^{n-1} C[k] * x[j][k] - due_date, ∀j ∈ {0,...,n-1}

== DATA FIELD SEMANTICS (key meanings and axis conventions) ==
- due_date: due_date; unit=time
- earliness_penalties: earliness_penalties; unit=cost per time unit; axis: job
- n: n; unit=count
- processing_times: processing_times; unit=time; axis: job
- tardiness_penalties: tardiness_penalties; unit=cost per time unit; axis: job

== OBJECTIVE (authoritative; the solver MUST optimize exactly this) ==
Sense: minimize
Formula: min ∑_{j=0}^{N-1} (earliness_penalties[j] * E[j] + tardiness_penalties[j] * T[j])
Plain: Minimize the sum of weighted earliness and weighted tardiness across all jobs.
If any earlier sentence conflicts with this section, this section wins.

== IO SCHEMA ==
{"n": int, "due_date": int, "processing_times": [int, ...], "earliness_penalties": [int, ...], "tardiness_penalties": [int, ...]}

OUTPUT SCHEMA — the solution dict you return MUST use EXACTLY these keys and shapes (this was confirmed upstream; do NOT rename, add, or drop keys):
- "order": list[int] — processing sequence of the jobs (job indices from 0 to n-1 arranged in order). example: [2, 0, 3, 1]. Must contain the indices of all n jobs and be a complete permutation
Fill every value for every entity; keep the keys EXACTLY as above.
```

## Objective

**Direction**: min

**Formula**
```
min ∑_{j=0}^{N-1} (earliness_penalties[j] * E[j] + tardiness_penalties[j] * T[j])
```

**Plain-language statement**
Minimize the sum of weighted earliness and weighted tardiness across all jobs.

**Rationale**
The problem defines a non-regular objective where both finishing too early and too late relative to the common due date incur costs based on job-specific weights.
