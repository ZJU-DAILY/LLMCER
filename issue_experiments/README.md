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
| `blocking_recall.py` | Pair-completeness (recall) of blocking per dataset + b_t sweep (issue #2). | no |
| `pick_threshold.py` | Recall-aware b_t selection (PC ≥ 0.85) from the sweep. | no |
| `check_api.py` | Minimal OpenAI connectivity check (standard SDK format). | yes |
| `test_real_dataset.py` | Real SBERT + real LLM run on a sampled real dataset. | yes |
| `mock_llm.py` | Deterministic oracle / noisy-oracle LLM used by the tests. | — |
| `run_all_experiments.py` | Master runner: runs everything, writes timestamped logs + `manifest.json` to `results/`. | optional |

## Documents

| File | Contents |
|------|----------|
| `GITHUB_REPLIES.md` | Ready-to-paste English replies for issues #2–#9 (code fixes). |
| `ISSUE_2_3_DRAFTS.md` | Replies for the two follow-up experiment requests (blocking recall, batch ordering). |
| `ISSUE_RESPONSES.md` | Long-form per-issue confirmation + fix + result. |
| `results/` | Timestamped experiment logs and `manifest.json` (committed). |

## Reproduce the whole record

```bash
.venv/Scripts/python.exe issue_experiments/run_all_experiments.py            # no-LLM experiments
.venv/Scripts/python.exe issue_experiments/run_all_experiments.py --with-llm  # also real-LLM runs
```

This writes one `<name>_<timestamp>.log` per experiment plus `results/manifest.json`
recording command, exit code, duration, and log path for each run.

## Key results

- `test_issues.py`: **16/16 checks pass** (issues #2–#9).
- `test_end_to_end.py`: oracle pipeline recovers ground truth exactly, **ACC = 1.0**.
- `blocking_recall.py`: candidate-set recall (Pair Completeness) **≥ 0.87 on all
  7 datasets** at recall-aware `b_t` (Cora 0.89, CiteSeer 0.88, DG 0.88,
  Music 0.93, Alaska 0.95, Song 0.95, AS 0.87).
- `check_datasets.py`: 7 datasets usable; `Walmart_Amazon` has no ground-truth
  file (two-table benchmark missing its second table).
