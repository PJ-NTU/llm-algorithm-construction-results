# Results

Per-instance results of the prototype on FrontierOR and CO-Bench (the results figures in the repository README).

- FrontierOR self-evolution subset: 20 tasks x 5 official large instances (100 instances, 750 algorithm runs).
- FrontierOR, all tasks: 179 tasks, of which 169 are compared with the Gurobi reference on their official instances (984 instances, 901 compared); Fig. 3e.
- CO-Bench: 36 problems, 6,155 instances.

The tables are in [`data/`](data/) as CSV and JSON (same columns). Empty CSV cells (JSON `null`) mean "not applicable" or "not recorded".
The `best_algorithm_file` and `algorithm_file` columns give each algorithm's path from the repository root; for the 49 problems whose portfolio was updated by continued evolution (all except FrontierOR 01 and 05 and CO-Bench 04, 07, 18, 20 and 30), they refer to the recorded algorithms.

## FrontierOR

### Metrics (Fig. 3)

| Metric | Definition | Denominator for the prototype |
|---|---|---|
| Execution | Fraction of tasks whose program runs without runtime errors. FrontierOR scores the first program each method generates; for the prototype, the delivered algorithms are scored. | 20 tasks |
| Feasibility | Fraction of official large instances on which a solution accepted by the official checker is returned within the time limit. | 100 instances |
| Solution quality | Fraction of instances whose solution is feasible and whose signed gap to the Gurobi reference is at most 0.01. Instances whose Gurobi reference is rejected by the official checker are excluded. | 95 instances |
| QTE (quality-time efficiency) | As solution quality, and the method finishes no later than Gurobi. For the prototype, Gurobi's one-hour solve-time budget is the time criterion. | 95 instances |

Signed gap: `g = (u - b) / |b|` for minimization and `g = (b - u) / |b|` for maximization, where `u` is the prototype's objective and `b` the Gurobi reference. Negative values favour the prototype. Each algorithm was given 60 s per instance; the prototype's result on an instance is the best feasible objective among the task's algorithms.

Comparator values (One-shot, EoH, OpenEvolve, CORAL) are quoted from Table 3 of FrontierOR (Kong, M. et al., *FrontierOR: Benchmarking LLMs' Capacity for Efficient Algorithm Design in Large-Scale Optimization*, preprint v3, 2026, https://arxiv.org/html/2605.25246v3) and were not rerun.

### `data/frontieror_summary.csv`

| Column | Meaning |
|---|---|
| `method` | Method name |
| `execution`, `feasibility`, `solution_quality`, `qte` | Metric values as plotted in Fig. 3 |
| `source` | Where the values come from |
| `tasks`, `instances` | Coverage (instances given for the prototype only) |
| `execution_count`, `feasibility_count`, `quality_count`, `qte_count` | the prototype numerator/denominator computed from the per-instance data |

### `data/frontieror_tasks.csv` (20 rows)

| Column | Meaning |
|---|---|
| `task_id` | Directory name in `frontieror/` (e.g. `01_adulyasak2015`) |
| `task` | FrontierOR task identifier |
| `problem` | Short English problem name |
| `sense` | `min` or `max` |
| `instances` | Official large instances (always 5) |
| `feasible_instances` | Instances with at least one feasible algorithm run |
| `valid_references` | Instances whose Gurobi reference is accepted by the official checker |
| `quality_hits` | Instances counted towards solution quality |
| `better`, `ties`, `worse`, `reference_invalid` | Verdict counts (see below) |
| `algorithms` | Algorithms evaluated for the task |
| `feasible_runs`, `runs` | Feasible algorithm x instance runs, and all runs |

### `data/frontieror_instances.csv` (100 rows)

| Column | Meaning |
|---|---|
| `task_id`, `task`, `problem`, `sense` | As above |
| `instance` | Official large instance, `large_1` ... `large_5` |
| `prototype_objective` | Best feasible objective among the task's algorithms |
| `best_algorithm` | Algorithm that achieved `prototype_objective` (the first by name if several tie) |
| `best_algorithm_file` | Path of that algorithm's source file, relative to the repository root |
| `gurobi_reference` | Gurobi reference objective provided by FrontierOR |
| `reference_valid` | `True` if the official checker accepts the Gurobi reference |
| `signed_gap` | `g` as defined above (empty if the reference is invalid) |
| `verdict` | `better` (g < 0), `tie` (abs(g) < 5e-5, shown as 0.00%), `worse` (g > 0), `reference_invalid` |
| `quality_pass` | `True` if `g <= 0.01` (empty if the reference is invalid) |
| `feasible_algorithms`, `algorithms` | Algorithms with a feasible run on this instance, and algorithms run |

### `data/frontieror_runs.csv` (750 rows)

| Column | Meaning |
|---|---|
| `task_id`, `task`, `instance` | As above |
| `algorithm`, `algorithm_file` | Algorithm identifier and source file path |
| `status` | `feasible` (accepted by the official checker), `infeasible` (rejected by the official checker) or `no_solution` (no solution returned, e.g. timeout) |
| `objective` | Objective value of a feasible run (empty otherwise) |
| `runtime_s` | Wall-clock time of the run in seconds |

### All tasks (Fig. 3e)

Each algorithm of a task's stored portfolio was run for 60 s on a single core per official instance (the tiny instance and up to five large instances); the prototype's result on an instance is the best feasible objective among these runs. The Gurobi references are quoted from FrontierOR and were obtained with up to one hour per instance. The signed gap is defined as above; `match` means abs(g) <= 1e-6. Each task is placed in one band by its median gap: the median of its signed gaps over the instances on which the prototype returned a feasible solution, taking the lower middle value for an even number of instances. Ten tasks are not compared because errors or contradictions in their official materials invalidate the Gurobi reference.

#### `data/frontieror_full_tasks.csv` (179 rows)

| Column | Meaning |
|---|---|
| `task` | FrontierOR task identifier |
| `sense` | `min` or `max` |
| `compared` | `True` if the task is compared with the Gurobi reference |
| `instances`, `compared_instances` | Official instances, and those with a feasible the prototype solution and a valid reference |
| `better`, `match`, `behind` | Compared instances with g < -1e-6, abs(g) <= 1e-6 and g > 1e-6 |
| `median_gap` | Median signed gap as defined above |
| `band` | Band of Fig. 3e: `Better than Gurobi`, `Match`, `Behind by ≤1%`, `Behind by 1–5%`, `Behind by 5–10%` or `Behind by >10%` |
| `reason` | Why a task is not compared (empty otherwise) |

#### `data/frontieror_full_instances.csv` (984 rows)

| Column | Meaning |
|---|---|
| `task`, `sense` | As above |
| `instance` | Official instance, `tiny` or `large_1` ... `large_5` |
| `gurobi_reference` | Gurobi reference objective provided by FrontierOR |
| `prototype_objective` | Best feasible objective among the task's algorithms (empty if none) |
| `signed_gap` | `g` as defined above (empty if not compared) |
| `status` | `compared`, `no_feasible_solution` (no feasible solution within 60 s) or `not_comparable` (the reference cannot be compared) |

## CO-Bench

### Metrics (Fig. 4)

| Metric | Definition |
|---|---|
| Instance score | Compares the objective `u` with the reference value `b` (the optimal or best-known value): 1 if `u` equals or improves on `b`, otherwise `min(abs(u), abs(b)) / max(abs(u), abs(b))`; 0 for an error, timeout or infeasible solution. Instances whose reference value is 0 follow the official CO-Bench scoring rule. |
| Average score | Instance scores averaged within each problem, then across the 36 problems |
| Valid rate | Fraction of problems for which a feasible solution is returned on every instance |
| Survival rate | Share of instances with a score above 0.99; per-problem values (`prototype_survival`) averaged over the 36 problems |

For each instance, the prototype's result is the best feasible result of its algorithm portfolio (10 s per instance).

Comparator values are quoted and were not rerun: FunSearch, EoH, Greedy Refine, ReEvo, o3 Mini Direct and Classical solver from Figure 3 of CO-Bench (Sun, W., Feng, S., Li, S. & Yang, Y., *CO-Bench: Benchmarking Language Model Agents in Algorithm Search for Combinatorial Optimization*, Proceedings of AAAI 40, 33126-33134, 2026, https://doi.org/10.1609/aaai.v40i39.40596); LACE, EoH-S and HeurAgenix from the LACE paper (Gong, H., Wang, S., Song, D., Sheu, J.-B. & Yan, R., *Large language models discover complementary heuristics for combinatorial optimization*, Nature Machine Intelligence, 2026; supplementary information at https://github.com/PJ-NTU/LACE).

### `data/cobench_summary.csv` (10 rows)

| Column | Meaning |
|---|---|
| `method`, `group` | Method name and group (as in Fig. 4) |
| `average_score`, `valid_rate`, `survival_rate` | Values as plotted in Fig. 4 |
| `source` | Where the values come from |
| `problems`, `instances` | Coverage (instances given for the prototype only) |

### `data/cobench_problems.csv` (36 rows)

| Column | Meaning |
|---|---|
| `problem_id` | Directory name in `cobench/` (e.g. `01_aircraft_landing`) |
| `problem`, `category` | CO-Bench problem name and category |
| `instances` | Instances listed for the problem |
| `official_reference_share` | Share of instances whose reference value is taken from the official CO-Bench configuration |
| `prototype_avg_score`, `prototype_survival` | Mean instance score and share of instances with score > 0.99 |
| `prototype_above_reference` | Instances on which the prototype's objective improves on the reference value |
| `lace_avg_score`, `lace_survival` | The same two quantities computed from LACE's published per-instance objectives with the same reference values and scoring (empty for problem 36). The Fig. 4 value for LACE is the quoted one. |
| `classical_solver_score`, `funsearch_score` | Per-problem scores from the CO-Bench appendix tables |

### `data/cobench_instances.csv` (6,155 rows)

| Column | Meaning |
|---|---|
| `problem_id`, `problem` | As above |
| `instance`, `case` | Instance file and case index within the file |
| `split` | `train` or `test`, as labelled in the instance source |
| `objective_direction` | Whether larger or smaller objective values are better for the values listed |
| `reference_value` | Optimal or best-known value used for scoring |
| `reference_source` | `official` (CO-Bench configuration) or `supplementary` (best-known value listed in the LACE supplementary data) |
| `prototype_objective`, `prototype_score` | the prototype's objective and instance score |
| `best_algorithm`, `best_algorithm_file` | Algorithm that achieved `prototype_objective` (one of them if several tie) and its source file path (empty where not recorded) |
| `lace_objective`, `lace_score` | LACE's published objective and its score under the same reference and rule (empty if unavailable) |
