# Generalised assignment problem: mathematical model

Problem statement of the prototype's problem type used for this CO-Bench problem (decision variables, constraints, data field semantics, objective and I/O schema).

```text
A set of jobs J must be assigned to a set of machines I. Each job j is assigned to exactly one machine i. Assigning job j to machine i incurs a cost cost_matrix[i][j] and consumes resource_matrix[i][j]. Each machine i has a total resource capacity capacities[i]. The goal is to minimize the total assignment cost while respecting all machine capacities. All indices are 0-indexed.

== DECISION VARIABLES (authoritative; model with exactly these) ==
- x[i][j]: Binary indicator for whether job j is assigned to machine i  |  x[i][j] ∈ {0,1}, for i ∈ {0...num_machines-1}, j ∈ {0...num_jobs-1}

== CONSTRAINTS (authoritative; ALL must be modeled) ==
- [hard] Each job must be assigned to exactly one machine.  |  sum_{i=0}^{num_machines-1} x[i][j] = 1, ∀ j ∈ {0...num_jobs-1}
- [hard] The total resource consumed by all jobs assigned to a machine must not exceed its capacity.  |  sum_{j=0}^{num_jobs-1} resource_matrix[i][j] * x[i][j] <= capacities[i], ∀ i ∈ {0...num_machines-1}

== DATA FIELD SEMANTICS (key meanings and axis conventions) ==
- num_machines: num_machines; unit=machines
- num_jobs: num_jobs; unit=jobs
- cost_matrix: cost_matrix; unit=cost units; axis: machine × job
- resource_matrix: resource_matrix; unit=resource units; axis: machine × job
- capacities: capacities; unit=resource units; axis: machine

== OBJECTIVE (authoritative; the solver MUST optimize exactly this) ==
Sense: minimize
Formula: min Σ_{i=0}^{num_machines-1} Σ_{j=0}^{num_jobs-1} cost_matrix[i][j] * x[i][j]
Plain: The sum of costs for all selected job-to-machine assignments.
If any earlier sentence conflicts with this section, this section wins.

== IO SCHEMA ==
{"instance_name": str, "num_machines": int, "num_jobs": int, "cost_matrix": [[int, ...], ...], "resource_matrix": [[int, ...], ...], "capacities": [int, ...], "capacity_factor_used": float, "num_binary_variables": int, "algorithm_parameters_gamma": float, "algorithm_parameters_initial_stepsize_s0": float, "algorithm_parameters_multiplier_initialization": str, "algorithm_parameters_nu": int, "algorithm_parameters_zeta": float, "formulation_constraints": [str, ...], "formulation_objective": str, "gap_type": str, "generation_seed": int}

OUTPUT SCHEMA — the solution dict you return MUST use EXACTLY these keys and shapes (this was confirmed upstream; do NOT rename, add, or drop keys):
- "assignments": dict[str, int] — The machine assigned to each job. example: {"0": 2, "1": 0, "2": 2, "3": 1, "4": 4}. Include all jobs from 0 to num_jobs-1 as keys, with their corresponding machine index (0 to num_machines-1) as the value.
Fill every value for every entity; keep the keys EXACTLY as above.
```

## Objective

**Direction**: min

**Formula**
```
min Σ_{i=0}^{num_machines-1} Σ_{j=0}^{num_jobs-1} cost_matrix[i][j] * x[i][j]
```

**Plain-language statement**
The sum of costs for all selected job-to-machine assignments.

**Rationale**
The problem objective is to minimize the total cost incurred by the assignment of all jobs.
