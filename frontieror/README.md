# FrontierOR

Algorithms and evaluation code for the 20 FrontierOR tasks used for the recorded results
"Large language models construct and evolve reusable optimization algorithms".

The official FrontierOR benchmark data (instances, official feasibility checkers, Gurobi reference solutions)
are downloaded from Hugging Face (see [Official data](#official-data)).

## Layout

```
frontieror/
  harness/                 evaluation code (official instance -> adapter -> algorithm -> official checker -> objective)
  <NN_task>/
    formulation.md         mathematical model (decision variables, constraints, objective, data fields)
    input_format.json      input fields of the task format (where available)
    output_format.json     solution fields of the task format (where available)
    checker.py             check(instance, solution) / check_solution(instance, solution) -> list of violations
    objective.py           objective(instance, solution) -> float
    adapter.py             official format <-> task format
    instances/             prototype-generated evaluation instances in the task format
    algorithms/<algo_id>.py
```

Algorithm file names are the algorithm IDs used in the results ledgers. The portfolios of all tasks except 01 and 05
were updated by the continued evolution recorded here, together with their evaluation instances (and, for tasks 02, 04,
06 and 10, the formulation, checker and objective of the revised problem type); the ledgers give the recorded
algorithms for these tasks.

| Task | Official name | Sense | Algorithms | Instances | Adapter |
|---|---|---|---:|---:|---|
| 01_adulyasak2015 | adulyasak2015 | min | 5 | 10 | yes |
| 02_archetti2007 | archetti2007 | min | 5 | 10 | yes |
| 03_armbruster2012 | armbruster2012 | min | 5 | 10 | yes |
| 04_bard2002 | bard2002 | min | 5 | 10 | yes |
| 05_bertsimas2022 | bertsimas2022 | min | 5 | 10 | yes |
| 06_bront2009 | bront2009 | max | 5 | 10 | yes |
| 07_carvalho1999 | carvalho1999 | min | 5 | 10 | yes |
| 08_desaulniers2014 | desaulniers2014 | min | 5 | 10 | yes |
| 09_desaulniers2016 | desaulniers2016 | min | 5 | 10 | yes |
| 10_furini2021 | furini2021 | min | 5 | 10 | yes |
| 11_kobeaga2024 | kobeaga2024 | max | 5 | 10 | yes |
| 12_mehrotra1996 | mehrotra1996 | min | 5 | 10 | yes |
| 13_nagy2015 | nagy2015 | min | 5 | 10 | yes |
| 14_pedersen2024 | pedersen2024 | min | 5 | 10 | yes |
| 15_pinnoi1997 | pinnoi1997 | min | 4 | 10 | yes |
| 16_roberti2018 | roberti2018 | min | 5 | 10 | yes |
| 17_rostami2021 | rostami2021 | min | 5 | 10 | yes |
| 18_schwerdfeger2016 | schwerdfeger2016 | min | 5 | 8 | yes |
| 19_wangk2020 | wangk2020 | min | 5 | 10 | yes |
| 20_watermeyer2020 | watermeyer2020 | min | 5 | 7 | yes |

## Interfaces

- **Algorithm**: `solve(instance, time_limit_s) -> dict` or `solve(instance, tools, time_limit_s) -> dict`.
  `instance` is a dict in the task format (see `instances/`, `input_format.json`, `formulation.md`); the returned
  dict follows the task's output format. For the three-argument form, `tools["is_feasible"](solution)` returns
  `(bool, message)` and `tools["objective"](solution)` returns the objective (lower is better; maximisation
  objectives are negated).
- **Checker**: returns a list of violation messages; an empty list means feasible.
- **Adapter**: `to_ours(official_instance) -> dict`, `to_official(official_instance, solution) -> dict`,
  `official_objective(official_instance, official_solution) -> float`.

## Setup

From the repository root (Python 3.10 or later):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt   # numpy, pulp (with CBC), scipy, mip, huggingface_hub, ...
python setup_runtime.py                     # checks PuLP's CBC solver; links a native build on Apple Silicon
cd frontieror
```

## Official data

The data is the Hugging Face dataset [`frontieror/FrontierOR`](https://huggingface.co/datasets/frontieror/FrontierOR)
(formerly `SmartOR/FrontierOR`). Download the files the harness needs for the 20 tasks with one command:

```bash
python harness/fetch_official.py --out /path/to/FrontierOR                      # 20 tasks, about 555 MB
python harness/fetch_official.py --out /path/to/FrontierOR --tasks nagy2015     # a single task
```

By default this fetches dataset revision `37ccd8b6` (2026-06-14), the one the recorded results were produced with;
its layout is `/path/to/FrontierOR/<official_name>/{feasibility_check.py, instance/, gurobi_solution/, ...}`.
`--revision main` fetches the latest release (FrontierOR-v1, 2026-09-27), laid out as
`/path/to/FrontierOR/tasks/<official_name>/...`; the harness reads both layouts. FrontierOR-v1 differs from the
revision used here in ways that change results:

- every official checker first checks a solution contract (required, non-empty fields), and bard2002 also requires
  `num_vehicles` and `total_distance` to match the routes. The adapters write all required fields: re-judged with
  the v1 checkers, the solutions of bront2009, carvalho1999, furini2021, mehrotra1996, roberti2018, rostami2021 and
  schwerdfeger2016 (all algorithms, large_1..5) keep their verdicts;
- all Gurobi reference solutions were regenerated, so reference objectives and runtimes (and the relative gaps)
  differ; e.g. the invalid reference of roberti2018/large_1 was replaced;
- bard2002 instances now set `travel_time_includes_service_time` to false, so the official travel time includes
  the service time (10 or 90 per customer). The bard2002 task format assumes folded service times, as in the
  revision used here, so its algorithms plan with shorter travel times than the v1 checker uses and their solutions
  violate the v1 time windows (all 50 bard2002 cells are infeasible under v1).

Use the default revision to reproduce the recorded results.

## Running the harness

```bash
export FRONTIEROR_DATA=/path/to/FrontierOR
python harness/run_benchmark.py --task 13_nagy2015 --workers 4                   # all algorithms x large_1..5, 60 s each
python harness/run_benchmark.py --task 13_nagy2015 --splits large_1 --algorithms r2_O1_1.py --time-limit 10
python harness/score.py results/*.json                                   # Execution rate, Feasibility, Solution quality, QTE
```

For every (algorithm, instance) cell, `run_benchmark.py` converts the official instance with the task adapter, runs
the algorithm in a separate process with the given time limit (default 60 s) restricted to one CPU core (on Linux;
BLAS/OpenMP thread pools are limited to one thread on all platforms), converts the solution back to the official
format, computes the official objective, and judges the solution with the official `feasibility_check.py`. The
Gurobi reference solution of each instance is judged by the same checker. Per instance, the best officially feasible
objective over the task's algorithms is compared with the Gurobi reference,
`rel = (ours - ref) / |ref|` (sign flipped for maximisation; negative means better than the reference).
Results are written to `results/<task>.json` with one record per cell (`official_feasible`, `violations`,
`objective`, `elapsed`, `our_ok`, `detail`) and one summary row per instance.

`our_ok` records whether the task's own `checker.py` accepts the solution; it is informational and does not affect
the official verdict. The task checker of 10_furini2021 also proves that the reported clique number is globally
minimal, which does not finish on the official instances; the harness stops any task check after
max(60 s, time limit) (where SIGALRM is available) and leaves `our_ok` empty.

`--workers N` runs N cells in parallel. `--adapter path/to/adapter.py` overrides the task's `adapter.py`.
