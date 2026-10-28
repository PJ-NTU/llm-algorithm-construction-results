# CO-Bench

prototype's algorithms, problem definitions and evaluation harness for the 36 problems of
[CO-Bench](https://arxiv.org/abs/2504.04310).

## Contents

```
cobench/
  README.md
  harness/                    evaluation harness (official CO-Bench instances, official eval_func)
    run.py                    command-line entry point
    fetch_data.py             downloads the official CO-Bench data (Hugging Face dataset CO-Bench/CO-Bench)
    adapters.py               official format <-> prototype problem format, one entry per problem
    adapters_corrected.py     corrected adapters of problems 10, 16, 19, 24 and 25 (replace their adapters.py entries)
    adapters_unconstrained_guillotine_cutting.py
    problems.json             problem ids, CO-Bench titles, direction of the official score
    instance_manifest.json    the 6,197 evaluated official instances: [file, case index, split] per problem
  <NN_problem>/               one directory per CO-Bench problem (01_aircraft_landing ... 36_unconstrained_guillotine_cutting)
    formulation.md            mathematical model of the prototype's problem type (variables, constraints, objective, I/O schema)
    input_format.json         instance fields, input schema and a small example instance
    output_format.json        solution fields (key, shape, example, completeness rule)
    checker.py                check_solution(instance, solution) -> list of violations (empty list = feasible)
    objective.py              objective(instance, solution) -> objective value
    config.py                 (problems 08, 09, 26, 27 only) the objective is defined here instead of objective.py
    instances/exam_00..09.json  prototype-generated evaluation instances (prototype problem format)
    algorithms/<algo_id>.py   the prototype's algorithms evaluated on CO-Bench; file name = algorithm id used in the results
```

Every algorithm file defines `solve(instance, time_limit_s)` (a few use `solve(instance)` or `solve(instance, tools,
time_limit_s)`), takes an instance in the prototype problem format described in `input_format.json` / `formulation.md`,
and returns a solution dict in the format of `output_format.json` within the time limit. Algorithms use only the
Python standard library, except some that import `numpy` and some that use `scipy` when it is available. The
portfolios of all problems except 04, 07, 18, 20 and 30 were updated by the continued evolution recorded here, together
with their evaluation instances (and, for problems 10, 11, 21, 22, 24, 25, 28, 29, 32, 34, 35 and 36, the
formulation, checker and objective of the revised problem type); the results give the recorded algorithms for these
problems. Problems 10, 24, 25, 28, 29, 32, 34 and 35 use the converters in `harness/adapters_continued/`,
written for the updated formats.

| # | Problem | Algorithms |
|---|---|---|
| 01 | aircraft_landing | 5 |
| 02 | assignment_problem | 5 |
| 03 | assortment_problem | 5 |
| 04 | bin_packing_one_dimensional | 10 |
| 05 | capacitated_warehouse_location | 5 |
| 06 | common_due_date_scheduling | 5 |
| 07 | constrained_guillotine_cutting | 10 |
| 08 | constrained_non_guillotine_cutting | 5 |
| 09 | container_loading | 5 |
| 10 | container_loading_with_weight_restrictions | 5 |
| 11 | corporate_structuring | 5 |
| 12 | crew_scheduling | 5 |
| 13 | equitable_partitioning_problem | 5 |
| 14 | euclidean_steiner_problem | 5 |
| 15 | flow_shop_scheduling | 5 |
| 16 | generalised_assignment_problem | 5 |
| 17 | graph_colouring | 5 |
| 18 | hybrid_reentrant_shop_scheduling | 10 |
| 19 | job_shop_scheduling | 5 |
| 20 | maximal_independent_set | 10 |
| 21 | multi_demand_multidimensional_knapsack_problem | 5 |
| 22 | multidimensional_knapsack_problem | 5 |
| 23 | open_shop_scheduling | 5 |
| 24 | p_median_capacitated | 5 |
| 25 | p_median_uncapacitated | 5 |
| 26 | packing_unequal_circles | 5 |
| 27 | packing_unequal_circles_area | 5 |
| 28 | packing_unequal_rectangles_and_squares | 5 |
| 29 | packing_unequal_rectangles_and_squares_area | 5 |
| 30 | resource_constrained_shortest_path | 3 |
| 31 | set_covering | 5 |
| 32 | set_partitioning | 5 |
| 33 | travelling_salesman_problem | 5 |
| 34 | uncapacitated_warehouse_location | 5 |
| 35 | vehicle_routing_period_routing | 5 |
| 36 | unconstrained_guillotine_cutting | 5 |

## Setup

Requirements: Python 3.10+ (tested with 3.13). From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python setup_runtime.py          # checks PuLP's CBC executable (links a native one on Apple Silicon)
```

`numpy` is used by some algorithms and official `config.py` files, `scipy` by some algorithms, and `networkx` by
the official data of maximal_independent_set.

## Official CO-Bench data

Download the official CO-Bench instances and evaluation code from the Hugging Face dataset [`CO-Bench/CO-Bench`](https://huggingface.co/datasets/CO-Bench/CO-Bench)
(about 720 MB, 856 files) with one command:

```bash
python cobench/harness/fetch_data.py /path/to/CO-Bench-data
```

The script downloads the dataset revision the harness was tested with (`--revision main` for the latest one).
The download has one directory per problem, named by its CO-Bench title (e.g. `Aircraft landing`), containing
the official `config.py` (with `load_data` and `eval_func`) and the instance files. The harness looks up a
problem directory under the data root by its CO-Bench title or its snake-case name (e.g. `aircraft_landing`, also
under `problems/`), and an instance file either directly in that directory or in its `instances/` subdirectory.
The instances evaluated for each problem (a file plus the case index inside multi-case files) are listed in
`harness/instance_manifest.json`.

## Running the harness

```bash
cd cobench/harness
python run.py --data-root /path/to/CO-Bench-data --problems 01 13 30 --out results
python run.py --data-root /path/to/CO-Bench-data --problems all --workers 8 --out results
```

Options: `--time-limit` (seconds per algorithm per instance, default 10), `--workers` (parallel processes,
default 1), `--max-instances N` (first N instances of each problem only). The data root can also be set with the
environment variable `COBENCH_DATA`.

For every official instance the harness runs each algorithm of the problem once, single-threaded, with
`time_limit_s = 10`; a run is aborted after 4x the time limit. Each worker process evaluates one instance at a time.
Per (instance, algorithm) cell it records in `results/<NN_problem>.jsonl`:

| field | meaning |
|---|---|
| `elapsed` | wall-clock seconds of `solve` |
| `lib_feasible`, `lib_viol` | verdict of the problem's `checker.py` on the prototype-format solution |
| `off_feasible`, `off_msg` | verdict of the official `eval_func` (feasible iff it returns a finite, non-negative value without raising) |
| `off_obj` | official objective, lower is better: `eval_func` value for minimisation problems, `1 / (value + 1e-12)` for maximisation problems (`eval_direction` in `problems.json`) |
| `err` | algorithm error, hard timeout or adapter error |

For some problems the official `eval_func` returns a normalised value rather than the raw objective: for
example job_shop_scheduling returns `lower_bound / makespan` and p_median_capacitated returns
`best_known / cost`, so `off_obj` is `makespan / lower_bound` and `cost / best_known` there.

## Adapters

The adapters only translate formats (field renaming, re-indexing, re-typing, restructuring) between the official
instances/solutions and the prototype problem format. All 36 problems have an adapter:

- `adapters.py`: 35 problems (the adapter module used for the evaluation);
- `adapters_unconstrained_guillotine_cutting.py`: problem 36;
- `adapters_corrected.py`: replaces the `adapters.py` entries of five problems.
  - 16 generalised_assignment_problem and 19 job_shop_scheduling: fixes of two format bugs that were corrected
    before the reported results were produced. The official job shop machine ids (1-based) are re-based to the
    0-based ids of the problem format, and the official generalised assignment data (floats) are converted to the
    integers of the problem format; the solution's 0-based machine index is mapped to the official 1-based agent.
  - 10 container_loading_with_weight_restrictions, 24 p_median_capacitated and 25 p_median_uncapacitated: the
    packaged algorithms solve corrected versions of these problem types, and the adapters map the official data
    to the formats given in each problem's `formulation.md`, `input_format.json` and `output_format.json`
    (per-orientation permissions and load-bearing limits for problem 10; `floor` of the Euclidean distance, the
    official cost, as `distance_matrix` for problem 24; the `chosen_medians` 0/1 vector for problem 25).
- `adapters.py` also contains three format fixes made on 2026-09-30, after the entries were found not to follow
  the problem formats when the notebook in `notebooks/` was tested on every problem. With them, the harness
  reproduces the recorded per-instance results for these problems:
  - 34 uncapacitated_warehouse_location: `alloc_cost` is passed as facility x customer, as in the problem format
    (it was customer x facility);
  - 28 packing_unequal_rectangles_and_squares: the problem format centres the container at (0, 0), so rectangle
    centres are shifted by the official container centre (cx, cy) on the way back;
  - 35 vehicle_routing_period_routing: `schedule_assignments` keys start at `customer_1`, and each value is read
    as a global schedule index, as the problem's checker does.
