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

## 2. The experiment (current design — 3 batch-construction strategies)

The experiment now tests the actual claim of **Algorithm 5 (batch building)**:
that grouping *similar tasks* into the same batched prompt helps. It compares
three ways of constructing the batches from the SAME pool of tasks:

- **similar** — group the most mutually-similar tasks into each batch
  (this is what Algorithm 5 does);
- **random** — group tasks randomly;
- **dissimilar** — group the least-similar tasks into each batch.

Expected ordering, consistent with Algorithm 5:

> **similar ≥ random ≥ dissimilar**

Implemented in `issue_experiments/batch_ordering.py`.

### Execution chain (how one dataset is processed)

1. **Embed** all records with Sentence-BERT (`all-MiniLM-L6-v2`), same as the
   pipeline.
2. **Form the task pool.** Run the real LSH blocker
   (`llmcer.clustering.lsh_block`, per-dataset `best` threshold mirroring
   `run_pipeline.py`), take the largest usable blocks, and from each block carve
   ONE NRS record set (`llmcer.record_set.next_record_set`). This yields a pool
   of `--pool` independent tasks (records in different blocks are never compared
   in the main pipeline, so batching their record sets as independent tasks is
   faithful to the method).
3. **Task–task similarity.** For every pair of tasks, similarity = max pairwise
   record cosine similarity.
4. **Build batches three ways** (`build_batches`): for each strategy, partition
   the pool into batches of `--batch` tasks —
   - *similar*: greedy seed + add the most-similar remaining task;
   - *dissimilar*: greedy seed + add the least-similar remaining task;
   - *random*: deterministic pseudo-random chunking.
5. **Run the LLM** once per batch (`build_batched_prompt` → `call_real_batch`),
   asking it to cluster each task independently and return JSON per task; parse
   and pool the per-task clusterings (`pooled_pred`).
6. **Score** ACC and FP against ground truth restricted to the pool, averaged
   over `--reps` repeats (the LLM is non-deterministic, so repeats average out
   decoding noise).
7. **Report** the three strategies' ACC side by side, and whether the
   `similar ≥ random ≥ dissimilar` gradient holds.

This is orthogonal to §7.8 (records *within* a set) and §7.7 (batch *size*):
here the tasks and the batch size are fixed; only HOW tasks are grouped changes.

## 3. How to run

```bash
# Smoke test (no API key) — validates the plumbing. The oracle clusters each task
# perfectly and independently of grouping, so all three strategies score the SAME;
# this only checks batch construction / scoring, not the gradient.
.venv/Scripts/python.exe issue_experiments/batch_ordering.py --mock --all

# REAL experiment, all datasets (needs OPENAI_API_KEY, and the gateway base_url
# in .env if the key is not an api.openai.com key):
.venv/Scripts/python.exe issue_experiments/batch_ordering.py --all

# single dataset:
.venv/Scripts/python.exe issue_experiments/batch_ordering.py --dataset cora
```

Parameters: `--pool` (tasks in the pool, default 9), `--batch` (tasks per prompt,
default 3), `--reps` (repeats per strategy, default 3).

Outputs to `results/batch_ordering/run_<mode>_<ts>/`:
`<dataset>.log` (full trace), `summary.csv`, `summary.txt` (the ACC-per-strategy
table to report).

**Smoke-test result (committed):**
`results/batch_ordering/run_mock_20260628_003433/` — cora, pool of 9 tasks → 3
batches; all three strategies score ACC = 1.0 (oracle is grouping-independent),
confirming the harness measures grouping and nothing else.

⚠️ **Real run status:** the real-LLM run is blocked here by API connectivity —
the `.env` key, sent to the official `api.openai.com` endpoint, times out. It
needs the OpenAI-compatible **gateway URL** set as `OPENAI_BASE_URL` in `.env`
(the same gateway the experimenter used previously). The code is otherwise ready;
once `OPENAI_BASE_URL` is set, `--all` produces the three-strategy ACC table with
no further changes.

## 4. Draft reply to the reviewer

### If the gradient holds (similar ≥ random ≥ dissimilar — expected)

> You are right on both points. §7.8 varies record order *within* a record set
> and §7.7 varies only batch size; neither tests how *tasks* are grouped into a
> batch. We also confirm that Table 18's "Similarity-Ordered" column duplicated
> Table 17's "w/o batching" column — a table-construction error we will remove.
>
> We ran the experiment that tests Algorithm 5 directly: from the same pool of
> tasks we construct batches three ways — grouping the most-similar tasks together
> (Algorithm 5), grouping randomly, and grouping the least-similar tasks together —
> and measure end-to-end ACC/FP. Across datasets we observe
> similar ≥ random ≥ dissimilar (ACC …), confirming that batching similar tasks
> together improves clustering, consistent with Algorithm 5. We will replace the
> erroneous Table 18 column with this table.

### If the gradient is weak / within noise

> … (as above) … The three strategies are close (within decoding noise), so we
> report that batch composition has at most a minor effect and scope the
> Algorithm 5 claim accordingly; we also note the LLM's non-determinism as a
> caveat.

## 5. Honesty notes

- The duplicated Table 18 column is not defensible — the reviewer has exact-match
  evidence. Either replace it with the real three-strategy numbers, or
  acknowledge the error. Do not present the old numbers as an independent
  measurement.
- The LLM endpoint used is non-deterministic (`temperature=0` rejected by the
  reasoning-style model), so report ACC averaged over `--reps` repeats and state
  this caveat; the gradient should be read as a trend, not an exact ordering on
  every single dataset.
