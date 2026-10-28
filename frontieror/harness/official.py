"""Access to the official FrontierOR data of one task.

Expected layout (as downloaded by fetch_official.py from the Hugging Face dataset frontieror/FrontierOR, formerly
SmartOR/FrontierOR):

    <data_root>/<official_task_name>/            (revisions up to 2026-09-05, including the one used for the recorded results)
    <data_root>/tasks/<official_task_name>/      (revisions from 2026-09-06 on, e.g. FrontierOR-v1)
        feasibility_check.py
        instance/tiny_instance.json, instance/large_instance_<k>.json
        gurobi_solution/tiny_solution.json, gurobi_solution/large_solution_<k>.json

The data root is given with --official-dir or the environment variable FRONTIEROR_DATA.
"""
import io
import json
import os
import sys

SPLITS = ["tiny"] + ["large_%d" % k for k in range(1, 6)]
LARGE = ["large_%d" % k for k in range(1, 6)]
# field names used by the reference solution files for the Gurobi wall-clock time (not present for every task)
RUNTIME_KEYS = ("runtime", "runtime_s", "runtime_seconds", "solve_time", "elapsed_time_seconds")


def data_root(arg=None):
    root = arg or os.environ.get("FRONTIEROR_DATA")
    if not root:
        sys.exit("official data not found: pass --official-dir or set FRONTIEROR_DATA")
    return os.path.abspath(os.path.expanduser(root))


def task_dir(root, task):
    nested = os.path.join(root, "tasks", task)
    return nested if os.path.isdir(nested) else os.path.join(root, task)


def instance_path(root, task, split):
    name = ("tiny_instance.json" if split == "tiny"
            else "large_instance_%s.json" % split.split("_")[1])
    return os.path.join(task_dir(root, task), "instance", name)


def solution_path(root, task, split):
    name = ("tiny_solution.json" if split == "tiny"
            else "large_solution_%s.json" % split.split("_")[1])
    return os.path.join(task_dir(root, task), "gurobi_solution", name)


def reference_solution(root, task, split):
    return json.load(io.open(solution_path(root, task, split), encoding="utf-8"))


def reference(root, task, split):
    """Objective value, MIP gap and wall-clock runtime of the Gurobi reference solution."""
    d = reference_solution(root, task, split)
    runtime = next((d[k] for k in RUNTIME_KEYS if isinstance(d.get(k), (int, float))), None)
    return {"objective_value": d.get("objective_value"),
            "gap": d.get("mip_gap", d.get("gap")),
            "runtime": runtime,
            "status": d.get("solver_status") or d.get("status")}
