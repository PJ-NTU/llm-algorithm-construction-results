"""Aggregate run_benchmark.py result files into the four FrontierOR metrics.

    python harness/score.py results/*.json [--official-dir /path/to/FrontierOR]

Definitions (FrontierOR paper, Section 4.2), per large instance:
  Execution rate    share of tasks whose programs ran without runtime errors
  Feasibility       share of instances with an officially feasible solution
  Solution quality  share of instances that are feasible and within 1% of (or better than) the Gurobi reference
  QTE               feasible, within 1% (or better), and not slower than the Gurobi reference wall-clock time
Instances whose Gurobi reference is not officially feasible (baseline_ok = false) are excluded from the quality and
QTE denominators. Two modes are reported: `portfolio` (best feasible solution over the task's algorithms per
instance) and `single` (the one algorithm per task with most feasible instances, then best mean objective).
"""
import argparse
import collections
import io
import json
import os
import sys

H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, H)
import official as O  # noqa: E402

TOL = 0.01


def score(results, root, mode):
    tot = collections.Counter()
    ok = collections.Counter()
    per_task = {}
    for v in results:
        sense, cells = v["sense"], v.get("cells") or []
        tot["exec"] += 1
        errs = sum(1 for c in cells if (c.get("detail") or "").strip() and not c.get("official_feasible")
                   and not c.get("violations") and c.get("our_ok") is not False)
        ok["exec"] += errs == 0
        if mode == "single":
            by_algo = collections.defaultdict(list)
            for c in cells:
                by_algo[c["algo"]].append(c)

            def key(kv):
                cs = kv[1]
                objs = [c["objective"] for c in cs if c["official_feasible"] and c["objective"] is not None]
                m = sum(objs) / len(objs) if objs else float("inf")
                return (-len(objs), m if sense == "min" else -m)
            chosen = sorted(by_algo.items(), key=key)[0][0] if by_algo else None
            pick = {c["split"]: c for c in cells if c["algo"] == chosen}
        else:
            pick = {}
            for c in cells:
                if not c["official_feasible"] or c["objective"] is None:
                    continue
                cur = pick.get(c["split"])
                if cur is None or ((c["objective"] < cur["objective"]) if sense == "min"
                                   else (c["objective"] > cur["objective"])):
                    pick[c["split"]] = c
        t = collections.Counter()
        for r in v["per_split"]:
            sp = r["split"]
            c = pick.get(sp)
            feas = bool(c and c.get("official_feasible") and c.get("objective") is not None)
            t["feas"] += feas
            t["n"] += 1
            if not r.get("baseline_ok") or r.get("ref") is None:
                continue
            t["valid"] += 1
            if not feas:
                continue
            ref, ours = float(r["ref"]), float(c["objective"])
            if abs(ref) < 1e-12:
                good = abs(ours) < 1e-12
            else:
                d = (ours - ref) / abs(ref)
                good = (d if sense == "min" else -d) <= TOL
            runtime = r.get("ref_runtime")
            if runtime is None and root:
                runtime = O.reference(root, v["task"], sp)["runtime"]
            fast = runtime is None or (c.get("elapsed") is not None and c["elapsed"] <= runtime)
            t["qual"] += good
            t["qte"] += good and fast
        for k in ("feas", "n", "valid", "qual", "qte"):
            tot[k] += t[k]
        per_task[v["task"]] = dict(t)
    return {"execution_rate": (ok["exec"], tot["exec"]), "feasibility": (tot["feas"], tot["n"]),
            "solution_quality": (tot["qual"], tot["valid"]), "qte": (tot["qte"], tot["valid"]),
            "per_task": per_task}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+")
    ap.add_argument("--official-dir", default=os.environ.get("FRONTIEROR_DATA"))
    a = ap.parse_args()
    results = [json.load(io.open(p, encoding="utf-8")) for p in a.results]
    for mode in ("portfolio", "single"):
        s = score(results, a.official_dir, mode)
        print("[%s]" % mode)
        for k in ("execution_rate", "feasibility", "solution_quality", "qte"):
            n, d = s[k]
            print("  %-17s %d/%d = %.2f" % (k, n, d, n / d if d else float("nan")))
    print(json.dumps(score(results, a.official_dir, "portfolio")["per_task"], indent=1))


if __name__ == "__main__":
    main()
