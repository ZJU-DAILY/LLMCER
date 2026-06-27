# Issue #3 — Batch task-ordering: full plan

## 1. What the reviewer claims (both points verified true)

1. **Wrong axis tested.** §7.8 ("Similarity-Ordered / Weak-Ordered / Random-Shuffle")
   studies the order of **records *within a single record set***. §7.7 varies only
   the **batch size**. Neither isolates the order of **tasks (record sets)
   *within one batched prompt***, which is what the reviewer asked about.

2. **Data reuse.** Table 18's "Similarity-Ordered" column is numerically identical
   to Table 17's "w/o batching" column — verified, all 9 datasets, to 4 decimals:

   | Dataset | T17 w/o batching | T18 Similarity-Ordered |
   |---|---|---|
   | Alaska | 0.8047 | 0.8047 |
   | AS | 0.6632 | 0.6632 |
   | Song | 0.7488 | 0.7488 |
   | Music | 0.6562 | 0.6562 |
   | DG | 0.7510 | 0.7510 |
   | Cora | 0.7938 | 0.7938 |
   | Citeseer | 0.9137 | 0.9137 |
   | Amazon-Google | 0.6732 | 0.6732 |
   | Walmart-Amazon | 0.5839 | 0.5839 |

   The two columns are the same numbers. This must be acknowledged and fixed, not
   defended: "w/o batching" has no batch, so it cannot be a batch-organization
   condition.

## 2. The experiment that actually answers the question

Implemented in `issue_experiments/batch_ordering.py`. It isolates **task order
within a batch**, holding everything else fixed:

- Sample whole ground-truth entities from a dataset → one block.
- NRS splits the block into K record sets (tasks). **Each task's content is
  fixed.**
- Pack the K tasks into ONE batched prompt; the only thing that changes between
  runs is the **order of the tasks** in that prompt:
  - `similarity` (tasks chained by descending mutual similarity),
  - `reverse` (of the similarity order),
  - `randomN` (deterministic shuffles).
- Send each ordering to the LLM, parse the per-task clustering, score ACC/FP
  against ground truth.
- Report **mean ± std across orderings**.

Interpretation:
- **std(ACC) ≈ 0** → task order has negligible influence; earlier tasks do not
  leak into later ones. We report this as a robustness result and replace the
  duplicated Table 18 column.
- **std(ACC) large** → a genuine ordering effect; we report it and discuss it as
  a limitation / design consideration.

This is orthogonal to §7.8 (records within a set) and §7.7 (batch size).

## 3. How to run

```bash
# Smoke test (no API key) — validates the harness. The oracle clusters each task
# independently, so std MUST be ~0; this checks the permutation/scoring plumbing.
.venv/Scripts/python.exe issue_experiments/batch_ordering.py \
    --mock --dataset cora --tasks 3 --records 60 --perms 4

# Real run (needs OPENAI_API_KEY in .env) — the actual experiment:
.venv/Scripts/python.exe issue_experiments/batch_ordering.py \
    --dataset cora --tasks 3 --records 60 --perms 5
# repeat per dataset: cora, citeseer, google-DBLP, music20K, sigmod, song, affiliation
```

Logs are written to `issue_experiments/results/batch_ordering/<dataset>_<mode>_<ts>.log`.

**Smoke-test result (committed):** `results/batch_ordering/cora_mock_*.log` —
6 permutations, ACC std = 0.0000, FP std = 0.0000 → harness is correct (order has
no effect when the backend is order-independent, as it must).

⚠️ The real-LLM run is currently blocked only by the API key (401 on the keys
tried so far). The code is ready; once a working key is in `.env`, the commands
above produce the real per-dataset task-order variance with no further changes.

## 4. Draft reply to the reviewer

### If the real run shows small variance (expected)

> You are right on both points. §7.8 varies record order *within* a record set
> and §7.7 varies only batch size; neither isolates the order of *tasks within a
> batched prompt*. We also confirm that Table 18's "Similarity-Ordered" column
> duplicates Table 17's "w/o batching" column — that is an error in table
> construction (a non-batched baseline mislabeled as a batch-organization
> condition), and we will remove it.
>
> We ran the missing experiment (`batch_ordering.py`): holding the K record sets
> in a batch fixed and permuting only their order (similarity / reverse / random),
> we measured ACC/FP across orderings on every dataset. The variance is [X]
> (std(ACC) = …), indicating that task order within a batch has [negligible /
> measurable] influence on the result. We will replace the Table 18 column with
> these correct numbers and add the experimental protocol.

### If a working key is not available before the deadline

> You are right on both points, and we will correct them. §7.8/§7.7 do not isolate
> the order of tasks within a batched prompt, and Table 18's "Similarity-Ordered"
> column is identical to Table 17's "w/o batching" column — an error we will fix.
> We have implemented the dedicated experiment (`batch_ordering.py`, with a
> deterministic smoke test verifying the harness) that permutes only the task
> order within a fixed batch and reports ACC/FP variance; we will include its
> results and scope our ordering claims to intra-record-set ordering, listing
> inter-task batch ordering explicitly as evaluated/limitation in the revision.

## 5. Honesty note

The duplicated column is not defensible — the reviewer has exact-match evidence.
The only credible responses are (a) run the real experiment and replace the
column, or (b) acknowledge the error and commit to the fix. Do not present the
duplicated numbers as an independent measurement.
