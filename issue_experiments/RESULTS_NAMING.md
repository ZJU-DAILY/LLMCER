# Result-file naming convention (review)

So every artifact is traceable to a reviewer issue and a run. Reviewed and
standardised as follows.

## Layout

```
issue_experiments/results/
├── run_<ts>/                         # one no-LLM validation+experiment session
│   ├── manifest.json                 #   what ran, exit codes, durations, log paths
│   ├── check_datasets_<ts>.log       #   dataset / GT audit
│   ├── test_issues_<ts>.log          #   per-issue unit checks (#2–#9), 16/16
│   ├── test_end_to_end_<ts>.log      #   oracle pipeline, ACC=1.0
│   ├── blocking_recall_<ts>.log      #   issue #2: recall per dataset (best b_t)
│   └── blocking_recall_final.txt     #   issue #2: clean recall-only table (report this)
│
└── batch_ordering/                   # issue #3 experiment (separate, may be re-run often)
    └── run_<mode>_<ts>/              #   mode = mock | real
        ├── <dataset>.log             #   per-dataset full trace (7 files)
        ├── summary.csv               #   machine-readable, one row per dataset
        └── summary.txt               #   ★ human-readable table to report ★
```

## Conventions

| Rule | Why |
|------|-----|
| `<ts>` = `YYYYMMDD_HHMMSS` | sortable, unique per session |
| each session in its own `run_*/` folder | logs packaged together; easy to hand off / delete |
| `batch_ordering/run_<mode>_<ts>/` carries `mock` vs `real` in the name | a smoke run is never mistaken for the real experiment |
| `summary.txt` is the one to read/paste | the `.log` files are the audit trail; the summary is the result |
| `*_final.txt` = the clean, report-ready table | distinguishes the reported number from verbose traces |

## Which file maps to which reviewer issue

| Reviewer issue | Report from |
|----------------|-------------|
| #2 blocking recall | `run_<ts>/blocking_recall_final.txt` |
| #3 batch task-ordering | `batch_ordering/run_real_<ts>/summary.txt` |
| #2–#9 code fixes (validation) | `run_<ts>/test_issues_<ts>.log` (16/16) + `test_end_to_end_<ts>.log` |

## Mapping to paper tables (what to replace/add)

| Paper table | Action | Source of new numbers |
|-------------|--------|-----------------------|
| Blocking table (Table 4) | add a **Recall (PC)** column | `blocking_recall_final.txt` |
| Table 18 (batch organization) | **remove duplicated "Similarity-Ordered" column**; rebuild as a real task-ordering table | `batch_ordering/run_real_*/summary.txt` |
| §7.8 text | scope "ordering" to *records within a set* | — |
