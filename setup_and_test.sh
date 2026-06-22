#!/bin/bash
# One-shot: create venv, install deps, run the no-API validation scripts.
# Usage (from project root, Git Bash):  bash setup_and_test.sh
set -e

# 1. Create the virtual environment if it doesn't exist.
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment .venv ..."
    python -m venv .venv
fi

# 2. Activate it (Windows Git Bash layout).
if [ -f ".venv/Scripts/activate" ]; then
    source .venv/Scripts/activate
else
    source .venv/bin/activate
fi

# 3. Install dependencies.
echo "Upgrading pip and installing requirements ..."
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# 4. Run the validation scripts (no API key needed) and capture output.
mkdir -p issue_experiments/results

echo ""
echo "===== Dataset diagnostic ====="
python issue_experiments/check_datasets.py | tee issue_experiments/results/check_datasets.txt

echo ""
echo "===== Per-issue tests ====="
python issue_experiments/test_issues.py | tee issue_experiments/results/test_issues.txt

echo ""
echo "===== End-to-end oracle test ====="
python issue_experiments/test_end_to_end.py | tee issue_experiments/results/test_end_to_end.txt

echo ""
echo "Done. Results saved under issue_experiments/results/."
