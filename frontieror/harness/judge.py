"""Run the official FrontierOR feasibility checker on one solution, in an isolated process.

    python judge.py <official_task_dir> <instance.json> <solution.json>

The last stdout line is `RESULT {"feasible": ..., "violations": [...], "how": ..., "error": ...}`.

Official checkers come in two forms: most expose `check_feasibility(instance, solution)`; some only provide a
command-line `main()` (called with --instance_path / --solution_path / --result_path). Their return values come in
three shapes, all normalised by `normalise()`.
"""
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import traceback


def _load(path):
    spec = importlib.util.spec_from_file_location("official_feasibility_check", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def normalise(r):
    """Map an official checker result to (feasible, violations).

    1) dict    {"feasible": bool, "violated_constraints": [...], ...}
    2) tuple   (feasible: bool, violated, violations, magnitudes)
    3) tuple   (violations_a, violations_b, ...) of containers only; feasible iff all are empty
    """
    if isinstance(r, dict):
        return bool(r.get("feasible")), list(r.get("violated_constraints") or r.get("violations") or [])
    if isinstance(r, (tuple, list)) and r:
        if isinstance(r[0], bool):
            return bool(r[0]), list(r[2] if len(r) > 2 else (r[1] if len(r) > 1 else []))
        buckets = [x for x in r if isinstance(x, (list, tuple, set, dict))]
        if buckets:
            first = next((list(x) for x in buckets if x), [])
            return all(len(x) == 0 for x in buckets), first
    return bool(r), []


def judge(official_task_dir, instance_path, solution, workdir, timeout=900):
    """Judge an official-format solution (dict) in a subprocess; returns the RESULT dict."""
    os.makedirs(workdir, exist_ok=True)
    sp = os.path.join(workdir, "solution_for_judge.json")
    json.dump(solution, io.open(sp, "w", encoding="utf-8"))
    try:
        r = subprocess.run([sys.executable, os.path.abspath(__file__), official_task_dir, instance_path, sp],
                           capture_output=True, text=True, timeout=timeout)
        for line in reversed((r.stdout or "").splitlines()):
            if line.startswith("RESULT "):
                return json.loads(line[7:])
        return {"feasible": False, "violations": [], "how": "",
                "error": ("checker produced no RESULT: " + (r.stderr or "")[-300:]).strip()}
    except subprocess.TimeoutExpired:
        return {"feasible": False, "violations": [], "how": "", "error": "official checker timed out"}


def main():
    task_dir, inst_p, sol_p = sys.argv[1], sys.argv[2], sys.argv[3]
    fc = os.path.join(task_dir, "feasibility_check.py")
    out = {"feasible": False, "violations": [], "how": "", "error": ""}
    try:
        m = _load(fc)
        if hasattr(m, "check_feasibility"):
            out["how"] = "check_feasibility"
            inst = json.load(io.open(inst_p, encoding="utf-8"))
            sol = json.load(io.open(sol_p, encoding="utf-8"))
            # checkers from FrontierOR-v1 on reject solutions that miss required fields before check_feasibility
            # is reached (in their command-line main()); apply the same contract check here
            contract = getattr(m, "_frontieror_validate_solution_contract", None)
            rejected = contract(sol) if contract else None
            ok, v = normalise(rejected if rejected is not None else m.check_feasibility(inst, sol))
            out.update(feasible=ok, violations=[str(x)[:200] for x in v][:20])
        else:
            out["how"] = "cli"
            rp = os.path.join(tempfile.mkdtemp(), "result.json")
            r = subprocess.run([sys.executable, fc, "--instance_path", inst_p,
                                "--solution_path", sol_p, "--result_path", rp],
                               capture_output=True, text=True, timeout=600)
            if not os.path.exists(rp):
                out["error"] = ("official CLI wrote no result: " + (r.stderr or "")[-300:]).strip()
            else:
                ok, v = normalise(json.load(io.open(rp, encoding="utf-8")))
                out.update(feasible=ok, violations=[str(x)[:200] for x in v][:20])
    except Exception:
        out["error"] = traceback.format_exc().strip().splitlines()[-1][:300]
    print("RESULT " + json.dumps(out))


if __name__ == "__main__":
    main()
