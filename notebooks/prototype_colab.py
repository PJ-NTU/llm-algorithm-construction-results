"""Helpers for notebooks/reproduce.ipynb: download the official data of one problem, run the prototype's
algorithms on it through the evaluation harnesses, and compare with the recorded per-instance results
(results/data/*.csv).
"""
import csv
import io
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FO_HARNESS = ROOT / "frontieror" / "harness"
CB_HARNESS = ROOT / "cobench" / "harness"
RESULTS = ROOT / "results" / "data"
PREVIEW_CHARS = 1500

# Problems whose portfolio was replaced by the continued evolution with a newer LLM recorded here
# (FrontierOR 18, CO-Bench 31). The per-instance tables still record the recorded algorithms, which are no longer in these
# portfolios (some names were reused).
EVOLVED = {"02_archetti2007", "03_armbruster2012", "04_bard2002", "06_bront2009", "07_carvalho1999",
           "08_desaulniers2014", "09_desaulniers2016", "10_furini2021", "11_kobeaga2024",
           "12_mehrotra1996", "13_nagy2015", "14_pedersen2024", "15_pinnoi1997", "16_roberti2018",
           "17_rostami2021", "18_schwerdfeger2016", "19_wangk2020", "20_watermeyer2020",
           "01_aircraft_landing", "02_assignment_problem", "03_assortment_problem",
           "05_capacitated_warehouse_location",
           "06_common_due_date_scheduling", "08_constrained_non_guillotine_cutting",
           "09_container_loading", "10_container_loading_with_weight_restrictions",
           "11_corporate_structuring", "12_crew_scheduling", "13_equitable_partitioning_problem",
           "14_euclidean_steiner_problem", "15_flow_shop_scheduling",
           "16_generalised_assignment_problem", "17_graph_colouring", "19_job_shop_scheduling",
           "21_multi_demand_multidimensional_knapsack_problem", "22_multidimensional_knapsack_problem",
           "23_open_shop_scheduling", "24_p_median_capacitated", "25_p_median_uncapacitated",
           "26_packing_unequal_circles", "27_packing_unequal_circles_area",
           "28_packing_unequal_rectangles_and_squares", "29_packing_unequal_rectangles_and_squares_area",
           "31_set_covering", "32_set_partitioning", "33_travelling_salesman_problem",
           "34_uncapacitated_warehouse_location", "35_vehicle_routing_period_routing",
           "36_unconstrained_guillotine_cutting"}
EVOLVED_NOTE = ("This portfolio was updated by continued evolution, so the recorded algorithm "
                "is not in it; running all %d algorithms of the current portfolio and keeping the best result.")


def _rows(name):
    with open(RESULTS / name, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _num(s):
    return None if s in ("", None) else float(s)


def _fmt(v):
    if v is None:
        return "-"
    return "%.6g" % v


def _score(v):
    return "-" if v is None else "%.3f" % v


def _preview(obj):
    s = json.dumps(obj, ensure_ascii=False)
    return s if len(s) <= PREVIEW_CHARS else s[:PREVIEW_CHARS] + " ... (%d characters in total)" % len(s)


def _run(cmd, cwd=None, show=False):
    """Run a harness script. Its output is captured and printed from Python, because notebook front ends such as
    Colab do not display what a child process writes directly."""
    r = subprocess.run([str(c) for c in cmd], cwd=cwd, capture_output=True, text=True)
    if r.returncode != 0:
        print((r.stdout or "")[-2000:])
        print((r.stderr or "")[-3000:])
        raise RuntimeError("%s failed (exit code %d); its output is shown above" % (Path(cmd[1]).name, r.returncode))
    if show:
        print("\n".join(l for l in r.stdout.splitlines() if l.strip()))
    return r


def _table(header, rows):
    cols = [header] + [[str(c) for c in r] for r in rows]
    w = [max(len(r[i]) for r in cols) for i in range(len(header))]
    line = lambda r: "  ".join(c.rjust(w[i]) if i else c.ljust(w[i]) for i, c in enumerate(r))
    print(line(header))
    print("  ".join("-" * x for x in w))
    for r in cols[1:]:
        print(line(r))


# ------------------------------------------------------------------ FrontierOR

def fo_tasks():
    return json.loads((FO_HARNESS / "tasks.json").read_text(encoding="utf-8"))


def fo_download(task, data_dir):
    official = fo_tasks()[task]["official_name"]
    _run([sys.executable, FO_HARNESS / "fetch_official.py", "--out", data_dir, "--tasks", official], show=True)


def fo_recorded(task):
    return {r["instance"]: r for r in _rows("frontieror_instances.csv") if r["task_id"] == task}


def _fo_run(task, data_dir, splits, algorithms, time_limit, workers, out_dir):
    out_dir = Path(out_dir).resolve()                      # the harness runs with cwd = frontieror/
    out = out_dir / (task + ".json")
    cmd = [sys.executable, str(FO_HARNESS / "run_benchmark.py"), "--task", task, "--official-dir", str(data_dir),
           "--splits", ",".join(splits), "--time-limit", str(time_limit), "--workers", str(workers),
           "--workdir", str(out_dir / task), "--out", str(out)]
    if algorithms:
        cmd += ["--algorithms", ",".join(algorithms)]
    _run(cmd, cwd=str(ROOT / "frontieror"))
    return json.loads(out.read_text(encoding="utf-8"))


def fo_run_one(task, split, data_dir, algorithm=None, time_limit=60, out_dir="runs"):
    """Run one algorithm (default: the one that gave the recorded result) on one official instance. For a task in
    EVOLVED, every algorithm of the current portfolio is run and the best feasible result is kept."""
    rec = fo_recorded(task)[split]
    available = sorted(f.name for f in (ROOT / "frontieror" / task / "algorithms").glob("*.py"))
    if algorithm is None and task in EVOLVED:
        print(EVOLVED_NOTE % len(available))
        algorithms = available
    else:
        algorithm = algorithm or rec["best_algorithm"]
        if not algorithm.endswith(".py"):                 # the results tables list some names without ".py"
            algorithm += ".py"
        if algorithm not in available:
            raise ValueError("unknown algorithm %s; choose from: %s" % (algorithm, ", ".join(available)))
        print("Task %s, instance %s, algorithm %s, time limit %gs ..." % (task, split, algorithm, time_limit))
        algorithms = [algorithm]
    res = _fo_run(task, data_dir, [split], algorithms, time_limit, min(len(algorithms), os.cpu_count() or 1), out_dir)
    row = res["per_split"][0]
    cell = next((c for c in res["cells"] if c["algo"] == row["by"]), res["cells"][0])
    algorithm = cell["algo"]
    sol_file = Path(out_dir).resolve() / task / ("%s__%s.json" % (split, algorithm[:-3]))
    print("\nSolution (official format, %s):" % sol_file)
    print(_preview(json.loads(sol_file.read_text(encoding="utf-8"))) if sol_file.exists() else "  none")
    print("\nOfficial checker: %s" % ("feasible" if cell["official_feasible"] else
                                     "INFEASIBLE " + "; ".join(map(str, cell["violations"]))[:300]))
    print("Objective (%s): %s    run time %.1fs\n" % (res["sense"], _fmt(cell["objective"]), cell["elapsed"]))
    _table(["", "objective"], [
        ["this run", _fmt(cell["objective"])],
        ["recorded (%s)" % rec["best_algorithm"], _fmt(_num(rec["prototype_objective"]))],
        ["Gurobi reference" + ("" if row["baseline_ok"] else " (rejected by checker)"), _fmt(row["ref"])],
    ])
    return res


def fo_run_portfolio(task, data_dir, time_limit=60, workers=None, out_dir="runs"):
    """Run every algorithm of the task on the five official large instances; best result per instance."""
    workers = workers or os.cpu_count() or 1
    rec = fo_recorded(task)
    splits = sorted(rec)
    n_alg = len(list((ROOT / "frontieror" / task / "algorithms").glob("*.py")))
    print("Task %s: %d algorithms x %d instances, %gs each, %d in parallel (about %.0f min) ..."
          % (task, n_alg, len(splits), time_limit, workers, n_alg * len(splits) * time_limit / workers / 60 + 1))
    t0 = time.time()
    res = _fo_run(task, data_dir, splits, None, time_limit, workers, out_dir)
    rows = []
    for r in res["per_split"]:
        p = rec[r["split"]]
        gap = ("ref. rejected" if not r["baseline_ok"] else
               "-" if r.get("rel") is None else "%+.2f%%" % (100 * r["rel"]))
        rows.append([r["split"], "%d/%d" % (r["n_feasible"], r["n_algos"]), _fmt(r["ours"]),
                     _fmt(_num(p["prototype_objective"])), _fmt(r["ref"]), gap])
    print()
    _table(["instance", "feasible algs", "this run (best)", "recorded", "Gurobi ref.", "gap vs Gurobi"], rows)
    print("\n%.1f min. Gap: negative = better than the Gurobi reference (sign follows the optimisation direction)."
          % ((time.time() - t0) / 60))
    return res


# ------------------------------------------------------------------ CO-Bench

REVISION = "6e561a36254d4315b8c2d155e81a465b634e87e1"   # as cobench/harness/fetch_data.py


def cb_problems():
    return {p["id"]: p for p in json.loads((CB_HARNESS / "problems.json").read_text(encoding="utf-8"))}


def cb_download(problem, data_dir):
    """Download only this problem's directory of the official CO-Bench dataset (pinned revision)."""
    import warnings
    from huggingface_hub import list_repo_files, snapshot_download
    from huggingface_hub.utils import logging as hf_logging
    hf_logging.set_verbosity_error()            # no "unauthenticated requests" notice: the dataset is public
    warnings.filterwarnings("ignore", message=".*unauthenticated requests.*")
    R = _cb_import()
    prob = cb_problems()[problem]
    # Dataset folders are named by CO-Bench titles, which may contain characters such as ':'; match them the way
    # the harness does (letters and digits only).
    wanted = {R._norm(prob["cobench_title"]), R._norm(prob["official_name"])}
    folders = {f.split("/")[0] for f in list_repo_files("CO-Bench/CO-Bench", repo_type="dataset", revision=REVISION)
               if "/" in f}
    folder = next(f for f in sorted(folders) if R._norm(f) in wanted)
    snapshot_download(repo_id="CO-Bench/CO-Bench", repo_type="dataset", revision=REVISION,
                      local_dir=str(data_dir), allow_patterns=[folder + "/*", folder + "/**"])
    print("CO-Bench data of %s is in %s" % (problem, Path(data_dir) / folder))


def cb_recorded(problem):
    return [r for r in _rows("cobench_instances.csv") if r["problem_id"] == problem]


def _cb_align(problem, rows):
    """Pair the harness's instance list (file path, case) with the rows of the per-instance table, matched by file
    name and case in order of appearance. Returns (path, case, row or None) in the harness's order."""
    prob = cb_problems()[problem]
    manifest = json.loads((CB_HARNESS / "instance_manifest.json").read_text(encoding="utf-8"))[prob["official_name"]]
    pool = {}
    for r in rows:
        pool.setdefault((r["instance"], int(r["case"])), []).append(r)
    return [(rel, int(idx), (pool.get((Path(rel).name, int(idx))) or [None]).pop(0)) for rel, idx, _ in manifest]


# For these problems the official evaluator returns a ratio (best-known or lower bound / objective) rather than the
# objective itself; the recorded per-instance table lists the objective, i.e. reference / ratio.
RATIO_EVAL = {"04_bin_packing_one_dimensional", "15_flow_shop_scheduling", "19_job_shop_scheduling",
              "23_open_shop_scheduling", "24_p_median_capacitated"}


def cb_objective(problem, value, ref):
    """Official evaluator value -> (objective in the units of the recorded table, direction)."""
    if problem in RATIO_EVAL:
        return (ref / value if value and ref else None), "min"
    return value, cb_problems()[problem]["eval_direction"]


def cb_score(u, b, direction):
    """Instance score: 1 if u equals or improves on the reference b, else min/max ratio."""
    if u is None or b is None:
        return None
    if b == 0:
        return 1.0 if abs(u) < 1e-9 else None     # zero references follow the official CO-Bench rule
    better = u >= b if direction == "max" else u <= b
    return 1.0 if better else min(abs(u), abs(b)) / max(abs(u), abs(b))


def _cb_import():
    if str(CB_HARNESS) not in sys.path:
        sys.path.insert(0, str(CB_HARNESS))
    import run as R
    return R


def cb_run_one(problem, index, data_dir, algorithm=None, time_limit=10):
    """Run one algorithm (default: the one that gave the recorded result) on instance `index` (0-based, in the
    order of the per-instance table). Where the table records no algorithm, every algorithm of the portfolio is run
    and the best feasible result is kept, as in the recorded results."""
    R = _cb_import()
    prob = cb_problems()[problem]
    d = prob["eval_direction"]
    rows = cb_recorded(problem)
    if not 0 <= index < len(rows):
        raise ValueError("INDEX must be between 0 and %d for %s" % (len(rows) - 1, problem))
    rec = rows[index]
    rel, case = next((f, c) for f, c, r in _cb_align(problem, rows) if r is rec)   # file path used by the harness
    off = R.OfficialProblem(R.find_problem_dir(data_dir, prob), d)
    algos, checker = R.load_package(prob)
    fns = dict(algos)
    recorded = None if problem in EVOLVED else rec["best_algorithm"]
    names = [algorithm or recorded] if (algorithm or recorded) else list(fns)
    names = [n[:-3] if n.endswith(".py") else n for n in names]
    unknown = [n for n in names if n not in fns]
    if unknown:
        raise ValueError("unknown algorithm %s; choose from: %s" % (unknown[0], ", ".join(fns)))
    to_lib, to_off = R.get_adapter(prob["official_name"])
    off_inst = off.load(rel, case)
    lib_inst = to_lib(off_inst)
    if len(names) > 1:
        print(EVOLVED_NOTE % len(names) if problem in EVOLVED else
              "No single algorithm is recorded for this instance; running all %d algorithms of the portfolio." % len(names))
    best = None
    for name in names:
        print("Problem %s, instance %s case %d, algorithm %s, time limit %gs ..." % (problem, rel, case, name, time_limit))
        signal.signal(signal.SIGALRM, R._alarm)
        signal.setitimer(signal.ITIMER_REAL, time_limit * R.HARD_MULT)
        t0 = time.time()
        try:
            lib_sol = R.call_solve(fns[name], lib_inst, time_limit)
        except BaseException as e:                     # timeout or error inside the algorithm
            print("  no solution: %s" % ("hard timeout" if isinstance(e, R._Timeout) else "%s: %s" % (type(e).__name__, e))[:300])
            continue
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
        run = {"algorithm": name, "solution": lib_sol, "elapsed": time.time() - t0}
        run["lib_violations"] = checker(lib_inst, lib_sol) if checker is not None else None
        try:
            run["official_solution"] = off_sol = to_off(lib_sol, off_inst, lib_inst)
        except Exception as e:                         # e.g. the algorithm reported that it found no solution
            print("  no official solution: %s" % ("%s: %s" % (type(e).__name__, e))[:300])
            continue
        ok, msg = off.is_feasible(off_inst, off_sol)
        run["official_msg"] = msg
        run["objective"] = float(off._eval({**off_inst, **off_sol})) if ok else None
        if len(names) > 1:
            print("  objective %s" % _fmt(run["objective"]))
        v = run["objective"]
        if best is None or (v is not None and (best["objective"] is None or
                                               (v > best["objective"] if d == "max" else v < best["objective"]))):
            best = run
    if best is None:
        print("\nNo algorithm returned a solution.")
        return None
    print("\nSolution of %s (prototype problem format):" % best["algorithm"])
    print(_preview(best["solution"]))
    if best["lib_violations"] is not None:
        v = best["lib_violations"]
        print("\nprototype checker: %s" % ("feasible" if not v else "INFEASIBLE " + str(v[:2])[:300]))
    print("Official evaluator: %s" % ("feasible, value %s" % _fmt(best["objective"]) if best["objective"] is not None
                                      else "INFEASIBLE " + str(best["official_msg"])))
    ref = _num(rec["reference_value"])
    value, dd = cb_objective(problem, best["objective"], ref)
    if problem in RATIO_EVAL and value is not None:
        print("  (the evaluator returns reference / objective for this problem; objective = %s)" % _fmt(value))
    print("Run time %.1fs\n" % best["elapsed"])
    _table(["", "objective", "score"], [
        ["this run (%s)" % best["algorithm"], _fmt(value), _score(cb_score(value, ref, dd))],
        ["recorded" + (" (%s)" % rec["best_algorithm"] if rec["best_algorithm"] else ""),
         _fmt(_num(rec["prototype_objective"])), _score(cb_score(_num(rec["prototype_objective"]), ref, dd))],
        ["best-known value", _fmt(ref), ""],
    ])
    print("Direction: %s is better." % ("larger" if dd == "max" else "smaller"))
    best["table_objective"] = value
    return best


def cb_run_portfolio(problem, data_dir, max_instances=None, time_limit=10, workers=None, out_dir="runs"):
    """Run every algorithm of the problem on its instances (optionally the first N); best result per instance."""
    workers = workers or os.cpu_count() or 1
    out_dir = Path(out_dir).resolve()
    prob = cb_problems()[problem]
    items = _cb_align(problem, cb_recorded(problem))[:max_instances]      # the harness runs its first N instances
    n_alg = len(list((ROOT / "cobench" / problem / "algorithms").glob("*.py")))
    print("Problem %s: %d algorithms x %d instances, up to %gs each, %d in parallel (at most about %.0f min) ..."
          % (problem, n_alg, len(items), time_limit, workers, n_alg * len(items) * time_limit / workers / 60 + 1))
    t0 = time.time()
    cmd = [sys.executable, str(CB_HARNESS / "run.py"), "--data-root", str(data_dir), "--problems", problem,
           "--time-limit", str(time_limit), "--workers", str(workers), "--out", str(out_dir)]
    if max_instances:
        cmd += ["--max-instances", str(max_instances)]
    _run(cmd)
    best = {}
    for line in open(out_dir / (problem + ".jsonl"), encoding="utf-8"):
        r = json.loads(line)
        if r.get("off_feasible") and r.get("off_obj", R_PENALTY) < R_PENALTY:
            k = (r["file"], r["case_idx"])
            best[k] = min(best.get(k, math.inf), r["off_obj"])
    d = prob["eval_direction"]
    raw = lambda o: (1.0 / o - 1e-12) if d == "max" else o      # harness stores max objectives as 1/value
    rows, ours, theirs = [], [], []
    for i, (rel, case, p) in enumerate(items):
        ref = _num(p["reference_value"]) if p else None
        u, dd = cb_objective(problem, raw(best[(rel, case)]) if (rel, case) in best else None, ref)
        pa = _num(p["prototype_objective"]) if p else None
        s_u, s_p = cb_score(u, ref, dd), cb_score(pa, ref, dd)
        ours.append(0.0 if s_u is None and u is None else s_u)
        theirs.append(s_p)
        rows.append([str(i), "%s #%d" % (Path(rel).name, case), _fmt(ref), _fmt(u), _fmt(pa), _score(s_u), _score(s_p)])
    print()
    _table(["#", "instance", "best-known", "this run", "recorded", "score (run)", "score (recorded)"], rows)
    mean = lambda xs: sum(x for x in xs if x is not None) / max(1, sum(x is not None for x in xs))
    print("\nMean score over these instances: this run %.4f, recorded %.4f   (%.1f min)"
          % (mean(ours), mean(theirs), (time.time() - t0) / 60))
    print("Direction: %s is better. Scores use the rule above; instances with a zero reference show '-'."
          % ("larger" if cb_objective(problem, 1.0, 1.0)[1] == "max" else "smaller"))
    return best


R_PENALTY = 1e10   # as cobench/harness/run.py
