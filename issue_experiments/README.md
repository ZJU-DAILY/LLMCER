# Issue Experiments

Self-contained validation for the fixes addressing rcrdbnss's 7 issues. None of
these require an OpenAI API key — the LLM is replaced by a deterministic
**oracle** (`mock_llm.py`) so the *algorithms* can be tested in isolation.

## Files

| File | Purpose |
|------|---------|
| `ISSUE_RESPONSES.md` | Per-issue confirmation, fix description, and a ready-to-paste GitHub reply. |
| `mock_llm.py` | Deterministic oracle / noisy-oracle LLM replacements. |
| `test_issues.py` | Reproduces each issue (old behaviour) and proves the fix. |
| `test_end_to_end.py` | Full pipeline on synthetic data; a perfect LLM must recover ground truth (ACC = 1.0). |
| `check_datasets.py` | Diagnoses each dataset's loadability and data/GT id alignment; recommends KEEP/DELETE. |
| `results/` | Captured stdout from the scripts above. |

## Running

```bash
python issue_experiments/test_issues.py        # per-issue before/after checks
python issue_experiments/test_end_to_end.py     # oracle pipeline recovers truth
python issue_experiments/check_datasets.py      # which datasets are usable
```

## Real-LLM run

To run the actual pipeline against a dataset, set the API key in the
environment (never commit it) and use `run.sh`:

```bash
export OPENAI_API_KEY="sk-..."
export DATASET_PATH="datasets/cora/cora.csv"
export GROUND_TRUTH_PATH="datasets/cora/gt.csv"
./run.sh
```

Thresholds are fixed/validation-tuned in `llmcer/config.py` and overridable via
`BLOCK_THRESHOLD`, `MERGE_THRESHOLD`, `SEPARATION_THRESHOLD`, `SET_SIZE`,
`SET_DIVERSITY` environment variables.
