"""Run one algorithm on one official instance, in an isolated process.

    python run_cell.py --task-dir <frontieror/NN_task> --adapter <adapter.py> --algorithm <algo.py>
                       --instance <official_instance.json> --time-limit 60 --sense min --out <solution.json>

Pipeline: official instance --adapter.to_ours--> task-format instance --algorithm.solve--> task-format solution
          --task checker.py (recorded as our_ok)--> --adapter.to_official--> official-format solution (written to --out)
          --adapter.official_objective--> objective value.
The last stdout line is `RESULT {...}`. The official feasibility check is run afterwards by the caller.

Algorithms expose either `solve(instance, time_limit_s)` or `solve(instance, tools, time_limit_s)`; for the latter,
`tools` provides `is_feasible(solution) -> (bool, message)` and `objective(solution) -> float` (lower is better;
maximisation objectives are negated), bound to the instance and built from the task's checker.py / objective.py.
"""
import argparse
import importlib.util
import inspect
import io
import json
import math
import os
import signal
import sys
import time
import traceback


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m          # needed by modules that use dataclasses
    spec.loader.exec_module(m)
    return m


def pin_single_core():
    """Restrict this process to one CPU core where the OS supports it (Linux)."""
    if hasattr(os, "sched_setaffinity"):
        cores = sorted(os.sched_getaffinity(0))
        want = int(os.environ.get("FRONTIEROR_CPU", cores[0]))
        os.sched_setaffinity(0, {want if want in cores else cores[0]})


def task_checker(task_dir):
    m = load(os.path.join(task_dir, "checker.py"), "task_checker")
    fn = getattr(m, "check", None) or getattr(m, "check_solution", None)
    if fn is None:
        raise AttributeError("checker.py defines neither check nor check_solution")
    return fn


class CheckerTimeout(BaseException):     # not caught by checkers that catch Exception
    pass


def bounded(fn, seconds, *args):
    """Call fn(*args), giving up after `seconds` where SIGALRM is available (the task checker of furini2021 proves
    global optimality exactly, which can take far longer than the solve itself on the official instances)."""
    if not hasattr(signal, "SIGALRM"):
        return fn(*args)

    def expire(*_):
        raise CheckerTimeout()
    old = signal.signal(signal.SIGALRM, expire)
    signal.alarm(max(1, int(seconds)))
    try:
        return fn(*args)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def task_objective(task_dir):
    return load(os.path.join(task_dir, "objective.py"), "task_objective").objective


def make_tools(task_dir, instance, sense):
    check = task_checker(task_dir)
    objective = task_objective(task_dir)

    def is_feasible(solution):
        try:
            v = check(instance, solution)
        except Exception as e:
            return False, "is_feasible raised: %s: %s" % (type(e).__name__, e)
        return (True, None) if not v else (False, "; ".join(str(x) for x in v))

    def tool_objective(solution):
        try:
            val = float(objective(instance, solution))
        except Exception:
            return 1e10
        if not math.isfinite(val):
            return 1e10
        return -val if sense == "max" else val

    return {"is_feasible": is_feasible, "objective": tool_objective}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-dir", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--algorithm", required=True)
    ap.add_argument("--instance", required=True)
    ap.add_argument("--time-limit", type=float, default=60.0)
    ap.add_argument("--sense", choices=["min", "max"], required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    out = {"status": "ERR", "elapsed": 0.0, "objective": None, "our_ok": None, "detail": ""}
    try:
        pin_single_core()
        adapter = load(a.adapter, "adapter_mod")
        algo = load(a.algorithm, "algo_mod")
        official_instance = json.load(io.open(a.instance, encoding="utf-8"))
        instance = adapter.to_ours(official_instance)

        n_params = len(inspect.signature(algo.solve).parameters)
        t0 = time.time()
        if n_params >= 3:
            solution = algo.solve(instance, make_tools(a.task_dir, instance, a.sense), a.time_limit)
        else:
            solution = algo.solve(instance, a.time_limit)
        out["elapsed"] = round(time.time() - t0, 2)

        if not isinstance(solution, dict):
            out["detail"] = "solve returned %s, not dict" % type(solution).__name__
        else:
            try:
                v = bounded(task_checker(a.task_dir), max(60.0, a.time_limit), instance, solution)
                out["our_ok"] = not v
                if v:
                    out["detail"] = "task checker violations: " + "; ".join(map(str, v))[:200]
            except CheckerTimeout:
                out["detail"] = "task checker did not finish within %ds (our_ok left empty)" % max(60, int(a.time_limit))
            except Exception as e:
                out["detail"] = "task checker raised: %s" % e
            official_solution = adapter.to_official(official_instance, solution)
            json.dump(official_solution, io.open(a.out, "w", encoding="utf-8"))
            try:
                obj = float(adapter.official_objective(official_instance, official_solution))
                out["objective"] = obj if math.isfinite(obj) else None
            except Exception as e:
                out["detail"] = (out["detail"] + " | official objective failed: %s" % e)[:300]
            out["status"] = "OK"
    except Exception:
        out.update(status="ERR", detail=traceback.format_exc().strip().splitlines()[-1][:300])
    print("RESULT " + json.dumps(out))


if __name__ == "__main__":
    main()
