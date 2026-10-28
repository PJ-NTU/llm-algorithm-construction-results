"""Evaluate the packaged prototype's algorithms on the official CO-Bench instances.

For every (instance, algorithm) cell the harness
  1. loads the official instance with the official `load_data` (from the problem's CO-Bench `config.py`),
  2. translates it into the prototype problem format with the adapter of that problem (`adapters.py`),
  3. runs the algorithm's `solve` with a time limit (default 10 s; a hard timeout of 4x the limit is enforced),
  4. checks the returned solution with the prototype checker of the problem (`checker.py`),
  5. translates the solution back to the official format and scores it with the official `eval_func`.

The official score follows the CO-Bench convention used in the evaluation: a solution is feasible iff `eval_func`
does not raise and returns a finite, non-negative number; the reported objective `off_obj` is lower-is-better
(`eval_func` value for minimisation problems, `1 / (value + 1e-12)` for maximisation problems).

Usage:
    python run.py --data-root /path/to/CO-Bench [--problems 01 13 30 | all] [--time-limit 10]
                  [--workers 1] [--max-instances N] [--out results]

The data root can also be given through the environment variable COBENCH_DATA.
"""
import argparse
import importlib.util
import inspect
import json
import math
import multiprocessing as mp
import os
import re
import signal
import sys
import time
import traceback
import warnings
from pathlib import Path

sys.dont_write_bytecode = True
warnings.filterwarnings("ignore", category=SyntaxWarning)   # invalid escapes in official DESCRIPTION strings

HARNESS = Path(__file__).resolve().parent
PACKAGE = HARNESS.parent
sys.path.insert(0, str(HARNESS))

HARD_MULT = 4.0          # hard timeout = time limit x HARD_MULT
PENALTY = 1e10           # objective returned for solutions the official evaluator cannot score

PROBLEMS = json.loads((HARNESS / "problems.json").read_text(encoding="utf-8"))
MANIFEST = json.loads((HARNESS / "instance_manifest.json").read_text(encoding="utf-8"))

# ---------------------------------------------------------------- official data

def _norm(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def find_problem_dir(root, prob):
    """Locate the official CO-Bench directory of a problem (by CO-Bench title or snake-case name)."""
    root = Path(root)
    for cand in (root / prob["cobench_title"], root / prob["official_name"],
                 root / "problems" / prob["official_name"]):
        if (cand / "config.py").exists():
            return cand
    wanted = {_norm(prob["cobench_title"]), _norm(prob["official_name"])}
    for base in (root, root / "problems"):
        if base.is_dir():
            for d in base.iterdir():
                if d.is_dir() and _norm(d.name) in wanted and (d / "config.py").exists():
                    return d
    raise FileNotFoundError(f"official data for {prob['official_name']!r} not found under {root}")


def instance_path(pdir, rel):
    for cand in (pdir / "instances" / rel, pdir / rel):
        if cand.exists():
            return cand
    raise FileNotFoundError(f"instance file {rel!r} not found in {pdir}")


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class OfficialProblem:
    """Official `load_data` / `eval_func` of one CO-Bench problem."""

    def __init__(self, pdir, direction):
        self.dir = Path(pdir)
        self.cfg = load_module(self.dir / "config.py", "_cobench_cfg_" + _norm(self.dir.name))
        self.direction = direction

    def load(self, rel, case_idx):
        data = self.cfg.load_data(str(instance_path(self.dir, rel)))
        if isinstance(data, list):
            if case_idx >= len(data):
                raise IndexError(f"case index {case_idx} >= {len(data)} cases in {rel}")
            return data[case_idx]
        return data

    def _eval(self, merged):
        merged = {(k if isinstance(k, str) else str(k)): v for k, v in merged.items()}
        sig = inspect.signature(self.cfg.eval_func)
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
            return self.cfg.eval_func(**merged)
        return self.cfg.eval_func(**{k: v for k, v in merged.items() if k in sig.parameters})

    def is_feasible(self, inst, sol):
        if not isinstance(sol, dict):
            return False, f"solution must be dict, got {type(sol).__name__}"
        try:
            raw = self._eval({**inst, **sol})
        except Exception as e:
            return False, f"eval_func raised: {type(e).__name__}: {e}"
        if raw is None:
            return False, "eval_func returned None"
        try:
            r = float(raw)
        except Exception as e:
            return False, f"eval_func returned non-numeric: {type(raw).__name__} ({e})"
        if not math.isfinite(r):
            return False, f"eval_func returned non-finite: {r}"
        if r < 0:
            return False, f"eval_func returned negative: {r}"
        return True, None

    def objective(self, inst, sol):
        try:
            r = float(self._eval({**inst, **sol}))
        except Exception:
            return PENALTY
        if not math.isfinite(r) or r < 0:
            return PENALTY
        return 1.0 / (r + 1e-12) if self.direction == "max" else r


# ---------------------------------------------------------------- prototype package

def load_package(prob):
    pdir = PACKAGE / prob["id"]
    algos = []
    for f in sorted((pdir / "algorithms").glob("*.py")):
        ns = {}
        try:
            exec(compile(f.read_text(encoding="utf-8"), f"<{f.stem}>", "exec"), ns)
            fn = ns.get("solve")
        except Exception:
            fn = None
        algos.append((f.stem, fn))
    ns = {}
    exec(compile((pdir / "checker.py").read_text(encoding="utf-8"), "<checker>", "exec"), ns)
    return algos, ns.get("check_solution") or ns.get("check")


def get_adapter(official):
    import adapters
    import adapters_corrected
    import adapters_unconstrained_guillotine_cutting as ugc
    table = dict(adapters.TABLE)
    table.update(adapters_corrected.TABLE)      # corrected entries for problems 10, 16, 19, 24, 25
    table["unconstrained_guillotine_cutting"] = ugc.ENTRY
    # Problems whose portfolio was updated by continued evolution with a different instance or solution format:
    # adapters_continued/<official>.py (to_lib, to_off) replaces the original entry.
    continued = HARNESS / "adapters_continued" / (official + ".py")
    if continued.exists():
        spec = importlib.util.spec_from_file_location("adapters_continued_" + official, continued)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        table[official] = {"to_lib": m.to_lib, "to_off": m.to_off}
    e = table.get(official, {})
    return e.get("to_lib", adapters._id_inst), e.get("to_off", adapters._id_sol)


def call_solve(fn, inst, t):
    """Algorithms use solve(instance, time_limit_s); a few use solve(instance) or solve(instance, tools, time_limit_s)."""
    try:
        ps = [p for p in inspect.signature(fn).parameters.values()
              if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    except Exception:
        ps = [None, None]
    if len(ps) >= 3:
        return fn(inst, {}, t)
    if len(ps) >= 2:
        return fn(inst, t)
    return fn(inst)


class _Timeout(Exception):
    pass


def _alarm(sig, frm):
    raise _Timeout()


# ---------------------------------------------------------------- one work unit

def work_unit(args):
    """One official instance x all algorithms of the problem."""
    prob, data_dir, rel, case_idx, t_max = args
    official = prob["official_name"]
    base = {"problem": prob["id"], "file": rel, "case_idx": case_idx}
    try:
        off = OfficialProblem(data_dir, prob["eval_direction"])
        algos, checker = load_package(prob)
        to_lib, to_off = get_adapter(official)
        off_inst = off.load(rel, case_idx)
        lib_inst = to_lib(off_inst)
    except Exception:
        return [{**base, "fatal": traceback.format_exc()[-700:]}]

    recs = []
    for algo_id, fn in algos:
        r = {**base, "algo": algo_id}
        if fn is None:
            r["err"] = "algo compile failed"
            recs.append(r)
            continue
        t0 = time.time()
        signal.signal(signal.SIGALRM, _alarm)
        signal.setitimer(signal.ITIMER_REAL, t_max * HARD_MULT)
        try:
            lib_sol = call_solve(fn, lib_inst, t_max)
            r["elapsed"] = round(time.time() - t0, 2)
        except _Timeout:
            r["err"] = "hard timeout"
            recs.append(r)
            continue
        except Exception as e:
            r["err"] = f"algo raised: {type(e).__name__}: {e}"[:200]
            recs.append(r)
            continue
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)

        if checker is not None:
            try:
                v = checker(lib_inst, lib_sol)
                r["lib_feasible"] = not v
                if v:
                    r["lib_viol"] = str(v[:2])[:200]
            except Exception as e:
                r["lib_feasible"] = None
                r["lib_err"] = f"{type(e).__name__}: {e}"[:150]

        try:
            off_sol = to_off(lib_sol, off_inst, lib_inst)
        except Exception as e:
            r["err"] = f"to_off raised: {type(e).__name__}: {e}"[:200]
            recs.append(r)
            continue
        try:
            ok, msg = off.is_feasible(off_inst, off_sol)
            r["off_feasible"] = bool(ok)
            if not ok:
                r["off_msg"] = str(msg)[:220]
            else:
                r["off_obj"] = float(off.objective(off_inst, off_sol))
        except Exception as e:
            r["off_feasible"] = False
            r["off_msg"] = f"judge raised: {type(e).__name__}: {e}"[:220]
        recs.append(r)
    return recs


# ---------------------------------------------------------------- driver

def select_problems(keys):
    if not keys or keys == ["all"]:
        return list(PROBLEMS)
    out = []
    for k in keys:
        hit = [p for p in PROBLEMS if k in (p["id"], p["id"][:2], p["official_name"])]
        if not hit:
            raise SystemExit(f"unknown problem {k!r}")
        out += hit
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default=os.environ.get("COBENCH_DATA"),
                    help="directory containing the official CO-Bench problems (env COBENCH_DATA)")
    ap.add_argument("--problems", nargs="*", default=["all"],
                    help="problem ids (e.g. 01 or 01_aircraft_landing or aircraft_landing); default all")
    ap.add_argument("--time-limit", type=float, default=float(os.environ.get("T_MAX", "10")),
                    help="seconds per algorithm run on one instance (default 10)")
    ap.add_argument("--workers", type=int, default=1, help="parallel processes, one instance each (default 1)")
    ap.add_argument("--max-instances", type=int, default=None, help="evaluate only the first N instances per problem")
    ap.add_argument("--out", default="results", help="output directory for <problem>.jsonl files")
    a = ap.parse_args()
    if not a.data_root:
        ap.error("--data-root (or COBENCH_DATA) is required")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    try:
        mp.set_start_method("fork")
    except (RuntimeError, ValueError):
        pass

    pool = mp.Pool(a.workers) if a.workers > 1 else None
    for prob in select_problems(a.problems):
        official = prob["official_name"]
        pdir = find_problem_dir(a.data_root, prob)
        items = MANIFEST[official][: a.max_instances]
        tasks = [(prob, str(pdir), rel, idx, a.time_limit) for rel, idx, _split in items]
        t0 = time.time()
        n_cell = n_ok = 0
        best = {}
        with open(out / f"{prob['id']}.jsonl", "w", encoding="utf-8") as fp:
            it = pool.imap_unordered(work_unit, tasks, chunksize=1) if pool else map(work_unit, tasks)
            for recs in it:
                for r in recs:
                    fp.write(json.dumps(r, ensure_ascii=False) + "\n")
                    if "fatal" in r:
                        print(f"  FATAL {r['file']}::{r['case_idx']}: {r['fatal'].strip().splitlines()[-1]}", flush=True)
                        continue
                    n_cell += 1
                    if r.get("off_feasible") and r.get("off_obj", PENALTY) < PENALTY:
                        n_ok += 1
                        k = f"{r['file']}::{r['case_idx']}"
                        best[k] = min(best.get(k, math.inf), r["off_obj"])
                fp.flush()
        print(f"DONE {prob['id']}: {len(tasks)} instances, {n_cell} cells, "
              f"{n_ok} officially feasible, {len(best)} instances solved, {time.time() - t0:.0f}s", flush=True)
    if pool is not None:
        pool.close()
        pool.join()


if __name__ == "__main__":
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(v, "1")
    main()
