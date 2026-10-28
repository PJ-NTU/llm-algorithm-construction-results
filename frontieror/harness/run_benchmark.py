"""Evaluate a task's algorithms on the official FrontierOR instances.

    python harness/run_benchmark.py --task 13_nagy2015 --official-dir /path/to/FrontierOR
        [--splits large_1,large_2,large_3,large_4,large_5] [--algorithms r2_O1_1.py,...]
        [--time-limit 60] [--workers 1] [--adapter path/to/adapter.py] [--out results.json]

For every (algorithm, split) cell: run_cell.py translates the official instance with the task adapter, runs the
algorithm with the time limit on one core, translates the solution back, and computes the official objective; the
official feasibility checker then judges the official-format solution. The Gurobi reference solution of every split
is judged by the same checker (`baseline_ok`). Per split, the best officially feasible objective over the task's
algorithms is compared with the Gurobi reference:

    rel = (ours - ref) / |ref|, sign flipped for maximisation (negative = better than the reference);
    rel is left out when the reference itself is not officially feasible.
"""
import argparse
import concurrent.futures as cf
import io
import json
import os
import subprocess
import sys
import time

sys.dont_write_bytecode = True
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
H = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(H)
sys.path.insert(0, H)
import judge as J          # noqa: E402
import official as O       # noqa: E402

THREAD_ENV = {k: "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                               "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}


def run_cell(job):
    task_dir, official_task, root, adapter, algo_path, split, tl, sense, workdir, cpu = job
    algo = os.path.basename(algo_path)
    sol_p = os.path.join(workdir, "%s__%s.json" % (split, algo[:-3]))
    rec = {"split": split, "algo": algo, "official_feasible": False, "violations": [],
           "objective": None, "elapsed": 0.0, "our_ok": None, "detail": ""}
    env = dict(os.environ, **THREAD_ENV)
    if cpu is not None:
        env["FRONTIEROR_CPU"] = str(cpu)
    hard_wall = tl * 3 + 120
    try:
        r = subprocess.run([sys.executable, os.path.join(H, "run_cell.py"),
                            "--task-dir", task_dir, "--adapter", adapter, "--algorithm", algo_path,
                            "--instance", O.instance_path(root, official_task, split),
                            "--time-limit", str(tl), "--sense", sense, "--out", sol_p],
                           capture_output=True, text=True, timeout=hard_wall, env=env)
        got = None
        for line in reversed((r.stdout or "").splitlines()):
            if line.startswith("RESULT "):
                got = json.loads(line[7:])
                break
        if got is None:
            rec["detail"] = ("no RESULT from solver process: " + (r.stderr or "")[-250:]).strip()
            return rec
        rec.update(elapsed=got["elapsed"], objective=got["objective"], our_ok=got.get("our_ok"),
                   detail=got.get("detail", ""))
        if got["status"] != "OK" or not os.path.exists(sol_p):
            rec["detail"] = rec["detail"] or "solve or translation failed"
            return rec
        j = J.judge(O.task_dir(root, official_task), O.instance_path(root, official_task, split),
                    json.load(io.open(sol_p, encoding="utf-8")), os.path.join(workdir, "_judge", split, algo[:-3]))
        rec.update(official_feasible=bool(j["feasible"]), violations=j.get("violations", [])[:6])
        if j.get("error"):
            rec["detail"] = (rec["detail"] + " | official checker error: " + j["error"])[:300]
    except subprocess.TimeoutExpired:
        rec["detail"] = "no result within the hard wall of %ds" % int(hard_wall)
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True, help="task directory name, e.g. 13_nagy2015")
    ap.add_argument("--official-dir", help="root of the official FrontierOR data (or set FRONTIEROR_DATA)")
    ap.add_argument("--splits", default=",".join(O.LARGE))
    ap.add_argument("--algorithms", help="comma-separated file names in <task>/algorithms (default: all)")
    ap.add_argument("--time-limit", type=float, default=60.0, help="seconds per algorithm and instance")
    ap.add_argument("--workers", type=int, default=1, help="cells run in parallel (each on one core)")
    ap.add_argument("--adapter", help="adapter module (default: <task>/adapter.py)")
    ap.add_argument("--workdir", help="where solutions are written (default: ./work/<task>)")
    ap.add_argument("--out", help="results JSON (default: ./results/<task>.json)")
    a = ap.parse_args()

    tasks = json.load(io.open(os.path.join(H, "tasks.json"), encoding="utf-8"))
    if a.task not in tasks:
        sys.exit("unknown task %s; choose from: %s" % (a.task, ", ".join(sorted(tasks))))
    meta = tasks[a.task]
    official_task, sense = meta["official_name"], meta["sense"]
    task_dir = os.path.join(ROOT, a.task)
    adapter = os.path.abspath(a.adapter or os.path.join(task_dir, "adapter.py"))
    if not os.path.exists(adapter):
        sys.exit("no adapter for %s: provide a module with to_ours / to_official / official_objective "
                 "via --adapter" % a.task)
    root = O.data_root(a.official_dir)
    splits = [s.strip() for s in a.splits.split(",") if s.strip()]
    algo_dir = os.path.join(task_dir, "algorithms")
    names = (a.algorithms.split(",") if a.algorithms
             else sorted(f for f in os.listdir(algo_dir) if f.endswith(".py")))
    algos = [os.path.join(algo_dir, n) for n in names]
    workdir = os.path.abspath(a.workdir or os.path.join("work", a.task))
    os.makedirs(workdir, exist_ok=True)

    cores = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    jobs = []
    for s in splits:
        for p in algos:
            cpu = cores[len(jobs) % min(a.workers, len(cores))] if cores else None
            jobs.append((task_dir, official_task, root, adapter, p, s, a.time_limit, sense, workdir, cpu))
    print("%s: %d algorithms x %d splits = %d cells, %gs each, sense %s"
          % (a.task, len(algos), len(splits), len(jobs), a.time_limit, sense), flush=True)

    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=a.workers) as ex:
        cells = list(ex.map(run_cell, jobs))

    per_split = []
    for s in splits:
        got = [c for c in cells if c["split"] == s]
        ok = [c for c in got if c["official_feasible"] and c["objective"] is not None]
        ref = O.reference(root, official_task, s)
        bl = J.judge(O.task_dir(root, official_task), O.instance_path(root, official_task, s),
                     O.reference_solution(root, official_task, s), os.path.join(workdir, "_judge", s, "_reference"))
        best = None
        if ok:
            best = (min(ok, key=lambda c: c["objective"]) if sense == "min"
                    else max(ok, key=lambda c: c["objective"]))
        row = {"split": s, "n_feasible": len(ok), "n_algos": len(got),
               "ours": best["objective"] if best else None, "by": best["algo"] if best else None,
               "secs": best["elapsed"] if best else None,
               "ref": ref["objective_value"], "ref_gap": ref["gap"], "ref_runtime": ref["runtime"],
               "baseline_ok": bool(bl["feasible"])}
        rv = ref["objective_value"]
        if best and row["baseline_ok"] and isinstance(rv, (int, float)) and abs(rv) > 1e-12:
            d = (best["objective"] - rv) / abs(rv)
            row["rel"] = d if sense == "min" else -d
        per_split.append(row)

    res = {"task": official_task, "sense": sense, "tl": a.time_limit,
           "portfolio": [os.path.basename(p) for p in algos], "per_split": per_split, "cells": cells,
           "minutes": round((time.time() - t0) / 60, 1)}
    out = a.out or os.path.join("results", a.task + ".json")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    json.dump(res, io.open(out, "w", encoding="utf-8"), indent=1)

    for c in cells:
        print("  %-9s %-22s %-10s %-16s %6.1fs %s" % (
            c["split"], c["algo"], "feasible" if c["official_feasible"] else "INFEASIBLE",
            "-" if c["objective"] is None else "%.6g" % c["objective"], c["elapsed"],
            (c["detail"] or ";".join(map(str, c["violations"])))[:60]))
    for p in per_split:
        rel = ("reference infeasible" if not p["baseline_ok"] else
               ("%+.2f%%" % (100 * p["rel"]) if p.get("rel") is not None else "-"))
        print("%-9s best %-14s ref %-14s %s  (%d/%d feasible)" % (
            p["split"], "-" if p["ours"] is None else "%.6g" % p["ours"],
            "-" if p["ref"] is None else "%.6g" % p["ref"], rel, p["n_feasible"], p["n_algos"]))
    print("->", out)


if __name__ == "__main__":
    main()
