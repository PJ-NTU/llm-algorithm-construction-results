"""Download the official CO-Bench data (instances and evaluation code of all 36 problems).

Source: the public Hugging Face dataset `CO-Bench/CO-Bench` (about 720 MB). The files are placed under the
target directory with the dataset's own layout (one directory per problem, named by its CO-Bench title,
each with `config.py` and the instance files), which is the layout `run.py --data-root` expects.

Usage:
    python fetch_data.py /path/to/CO-Bench-data [--revision REV]
"""
import argparse

REPO_ID = "CO-Bench/CO-Bench"
# Dataset revision the harness was tested with; pass --revision main for the latest one.
REVISION = "6e561a36254d4315b8c2d155e81a465b634e87e1"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target", help="directory to download into (use it as --data-root of run.py)")
    ap.add_argument("--revision", default=REVISION, help=f"dataset revision (default {REVISION[:12]})")
    a = ap.parse_args()

    from huggingface_hub import snapshot_download
    path = snapshot_download(repo_id=REPO_ID, repo_type="dataset", revision=a.revision, local_dir=a.target)
    print(f"CO-Bench data ({REPO_ID}@{a.revision}) is in {path}")


if __name__ == "__main__":
    main()
