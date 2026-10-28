"""Download the official FrontierOR data (instances, feasibility checkers, Gurobi reference solutions) from the
Hugging Face dataset frontieror/FrontierOR (formerly SmartOR/FrontierOR).

    python harness/fetch_official.py --out /path/to/FrontierOR [--tasks nagy2015,kobeaga2024] [--revision REV]

By default the revision used for the recorded results is fetched (dataset commit 37ccd8b6, 2026-06-14). Its layout is
<out>/<task>/...; the harness also reads the layout of later revisions (<out>/tasks/<task>/...), e.g.
`--revision main` for FrontierOR-v1. Later revisions changed the official checkers, the Gurobi reference solutions
and some instances, so they do not reproduce the recorded numbers exactly (see ../README.md).

Only the files the harness needs are downloaded (instances, checker, reference solutions and their official
feasibility results, schemas and the formulation). The data is distributed under the dataset's own licence and is
not part of this repository. Requires `huggingface_hub` (see requirements.txt).
"""
import argparse
import io
import json
import os

H = os.path.dirname(os.path.abspath(__file__))
REPO = "frontieror/FrontierOR"
RECORDED_REVISION = "37ccd8b6dca3bf7f4e0c58941a6ed156832a6d9e"
WANT = ("instance/*", "gurobi_solution/*", "gurobi_feasi_result/*", "feasibility_check.py",
        "instance_schema.json", "solution_schema.json", "mathematical_formulation.md", "problem_description.txt")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.environ.get("FRONTIEROR_DATA"), help="target directory")
    ap.add_argument("--tasks", help="comma-separated official task names (default: all tasks in tasks.json)")
    ap.add_argument("--revision", default=RECORDED_REVISION,
                    help="dataset revision (default: the one used for the recorded results; 'main' for the latest)")
    a = ap.parse_args()
    if not a.out:
        ap.error("pass --out or set FRONTIEROR_DATA")
    from huggingface_hub import HfApi, snapshot_download

    tasks = (a.tasks.split(",") if a.tasks else
             [v["official_name"] for v in json.load(io.open(os.path.join(H, "tasks.json"))).values()])
    top = {x.path for x in HfApi().list_repo_tree(REPO, repo_type="dataset", revision=a.revision)}
    prefix = "tasks/" if "tasks" in top else ""
    patterns = ["%s%s/%s" % (prefix, t, w) for t in tasks for w in WANT]
    path = snapshot_download(REPO, repo_type="dataset", revision=a.revision, local_dir=a.out,
                             allow_patterns=patterns, max_workers=4)
    for t in tasks:
        d = os.path.join(path, prefix, t)
        files = [os.path.join(r, f) for r, _, fs in os.walk(d) for f in fs]
        print("%-18s %3d files  %6.1f MB" % (t, len(files), sum(map(os.path.getsize, files)) / 1e6), flush=True)
    print("data root: %s  (set FRONTIEROR_DATA to this path)" % os.path.abspath(a.out))


if __name__ == "__main__":
    main()
