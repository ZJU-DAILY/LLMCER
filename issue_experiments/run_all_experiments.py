"""
Master experiment runner — executes every validation/experiment script, captures
full stdout to timestamped log files under results/, and writes a JSON manifest
recording what ran, when, exit code, and where its log lives.

This makes the experimental record reproducible and commit-ready: after running
this once, `issue_experiments/results/` contains one .log per experiment plus a
`manifest.json` summarising the session.

Usage:
  .venv/Scripts/python.exe issue_experiments/run_all_experiments.py
  .venv/Scripts/python.exe issue_experiments/run_all_experiments.py --only blocking_recall

Note: only experiments that need NO API key are run by default (issue
validation, blocking recall, dataset audit). Real-LLM runs (test_real_dataset.py)
are listed but skipped unless --with-llm is passed and OPENAI_API_KEY is set.
"""

import os
import sys
import json
import time
import argparse
import subprocess
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(HERE, "results")
PYTHON = sys.executable  # the venv python running this script

# (name, args, needs_llm). args are passed to the script.
EXPERIMENTS = [
    ("check_datasets",   ["check_datasets.py"],                 False),
    ("test_issues",      ["test_issues.py"],                    False),
    ("test_end_to_end",  ["test_end_to_end.py"],                False),
    # blocking recall: report ONLY the best-b_t row per dataset (--no-sweep).
    ("blocking_recall",  ["blocking_recall.py", "--no-sweep"],  False),
    # real-LLM experiment, skipped unless --with-llm:
    ("real_cora",        ["test_real_dataset.py", "--dataset", "cora", "--records", "40"], True),
]


def run_one(name, script_args, log_dir, ts):
    log_path = os.path.join(log_dir, f"{name}_{ts}.log")
    cmd = [PYTHON, os.path.join(HERE, script_args[0])] + script_args[1:]
    print(f"  running {name} -> {os.path.relpath(log_path, ROOT)}")
    start = time.time()
    with open(log_path, "w", encoding="utf-8") as fh:
        fh.write(f"# experiment: {name}\n")
        fh.write(f"# command: {' '.join(cmd)}\n")
        fh.write(f"# started: {ts}\n\n")
        fh.flush()
        proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=ROOT)
    elapsed = time.time() - start
    return dict(name=name, command=" ".join(cmd), log=os.path.relpath(log_path, ROOT),
                exit_code=proc.returncode, seconds=round(elapsed, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="run only this experiment by name")
    ap.add_argument("--with-llm", action="store_true",
                    help="also run real-LLM experiments (needs OPENAI_API_KEY)")
    args = ap.parse_args()

    # NOTE: argless datetime.now() is fine in a plain script (only forbidden in
    # Workflow scripts); we need a real wall-clock stamp for the log filenames.
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Each session gets its own folder so logs are packaged together.
    session_dir = os.path.join(RESULTS, f"run_{ts}")
    os.makedirs(session_dir, exist_ok=True)

    has_key = bool(os.environ.get("OPENAI_API_KEY")) or os.path.exists(
        os.path.join(ROOT, ".env"))

    manifest = dict(session=ts, python=PYTHON, runs=[])
    print(f"=== Experiment session {ts}  ->  {os.path.relpath(session_dir, ROOT)} ===")
    for name, script_args, needs_llm in EXPERIMENTS:
        if args.only and name != args.only:
            continue
        if needs_llm and not args.with_llm:
            print(f"  skipping {name} (needs --with-llm)")
            manifest["runs"].append(dict(name=name, skipped="needs --with-llm"))
            continue
        if needs_llm and not has_key:
            print(f"  skipping {name} (no OPENAI_API_KEY / .env)")
            manifest["runs"].append(dict(name=name, skipped="no api key"))
            continue
        manifest["runs"].append(run_one(name, script_args, session_dir, ts))

    manifest_path = os.path.join(session_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"\nManifest written to {os.path.relpath(manifest_path, ROOT)}")
    print("Per-experiment logs:")
    for r in manifest["runs"]:
        if "skipped" in r:
            print(f"  [skipped] {r['name']}: {r['skipped']}")
        else:
            status = "OK" if r["exit_code"] == 0 else f"EXIT {r['exit_code']}"
            print(f"  [{status}] {r['name']}  ({r['seconds']}s)  {r['log']}")


if __name__ == "__main__":
    main()
