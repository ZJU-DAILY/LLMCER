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
| `batch_ordering.py` | Batch TASK-ordering experiment (issue #3): permute task order within one batched prompt, measure ACC/FP variance. `--mock` for smoke test. | optional |
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
- `blocking_recall.py`: candidate-set recall (Pair Completeness) **≥ 0.86 on all
  7 datasets** (Cora 0.88, CiteSeer 0.86, DG 0.88, Music 0.94, Alaska 0.95,
  Song 0.94, AS 0.89). [issue #2]
- `batch_ordering.py`: mock smoke test shows **ACC std = 0** across task-order
  permutations (harness validated); real-LLM run pending a working API key. [issue #3]
- `check_datasets.py`: 7 datasets usable; `Walmart_Amazon` has no ground-truth
  file (two-table benchmark missing its second table).
