# Issue Experiments

Self-contained validation and experiments addressing the reviewer's GitHub
issues (#2–#9) plus the two follow-up experiment requests (blocking recall and
batch task-ordering). Most scripts need **no API key** — the LLM is replaced by a
deterministic oracle, and the blocking-recall study is pure geometry.

## Scripts

| File | Purpose | Needs LLM? |
|------|---------|-----------|
| `test_issues.py` | Per-issue before/after checks, in GitHub order #2–#9. | no |
| `test_end_to_end.py` | Full pipeline on synthetic data with an oracle LLM → ACC=1.0. | no |
| `check_datasets.py` | Dataset / ground-truth audit; KEEP/DELETE recommendation. | no |
| `blocking_recall.py` | Candidate-set recall (Pair Completeness) of blocking per dataset (issue #2). | no |
| `pick_threshold.py` | Recall-aware b_t selection (PC ≥ 0.85) from the sweep. | no |
| `batch_ordering.py` | Batch-construction experiment (issue #3): 3 strategies (similar / random / dissimilar) for grouping tasks into batched prompts, testing Algorithm 5's "similar tasks together" claim. Expect similar ≥ random ≥ dissimilar. `--mock` for smoke test. | optional |
| `check_api.py` | Minimal OpenAI connectivity check (standard SDK format). | yes |
| `test_real_dataset.py` | Real SBERT + real LLM run on a sampled real dataset. | yes |
| `mock_llm.py` | Deterministic oracle / noisy-oracle LLM used by the tests. | — |
| `run_all_experiments.py` | Master runner: runs everything, writes one timestamped folder `results/run_<ts>/` with per-experiment logs + `manifest.json`. | optional |

## Documents

| File | Contents |
|------|----------|
| `GITHUB_REPLIES.md` | Ready-to-paste English replies for issues #2–#9 (code fixes). |
| `ISSUE_2_3_DRAFTS.md` | Replies for the two follow-up experiment requests (blocking recall, batch ordering). |
| `ISSUE_3_PLAN.md` | Full plan for issue #3: experiment design, run instructions, and the (verified) Table 17/18 duplication + honest fix. |
| `ISSUE_RESPONSES.md` | Long-form per-issue confirmation + fix + result. |
| `results/run_<ts>/` | One folder per session: per-experiment logs + `manifest.json` (committed). |
| `results/batch_ordering/` | Logs from the issue-#3 batch task-ordering experiment. |

## Reproduce the whole record

```bash
.venv/Scripts/python.exe issue_experiments/run_all_experiments.py            # no-LLM experiments
.venv/Scripts/python.exe issue_experiments/run_all_experiments.py --with-llm  # also real-LLM runs
```

This creates `results/run_<timestamp>/` containing one `<name>_<timestamp>.log`
per experiment plus `manifest.json` (command, exit code, duration, log path).

## Key results

- `test_issues.py`: **16/16 checks pass** (issues #2–#9).
- `test_end_to_end.py`: oracle pipeline recovers ground truth exactly, **ACC = 1.0**.
- `blocking_recall.py`: candidate-set recall (Pair Completeness) **≥ 0.88 on all
  8 datasets** (Cora 0.91, CiteSeer 0.92, DG 0.88, Music 0.94, Alaska 0.95,
  Song 0.94, AS 0.90, Walmart-Amazon 0.95). [issue #2]
- `batch_ordering.py`: 3 batch-construction strategies (similar / random /
  dissimilar) testing Algorithm 5; mock smoke test passes (grouping-independent
  oracle → equal ACC, harness validated); real-LLM `--all` run pending the
  gateway `OPENAI_BASE_URL`. [issue #3]
- `check_datasets.py`: 8 datasets usable (Walmart-Amazon now has the single-table
  `walmart_amazon.csv` + `gt.csv`).
