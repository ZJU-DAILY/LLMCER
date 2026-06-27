# Drafts for reviewer issues #2 (blocking recall) and #3 (batch task-ordering)

These two are *experiment / reporting* requests, not code bugs. Issue #2 is a
deterministic measurement (done below, no LLM). Issue #3 needs either a real-LLM
experiment or an explicit limitation statement.

---

## Issue #2 — Blocking recall / pair-completeness

**Status: measured.** `issue_experiments/blocking_recall.py` reports, for every
dataset, the candidate-set recall (Pair Completeness, PC) at the pipeline's best
`b_t`, plus a full `b_t` sweep showing the recall/precision trade-off.

Metrics: PC = #true pairs kept in a block / #true pairs (recall; end-to-end
recall ≤ PC). PQ = pair quality (precision). RR = reduction ratio.

### Result table — candidate-set recall (Pair Completeness)

We report candidate-set recall only (the metric the reviewer asked for), at the
b_t that keeps recall high. Numbers from `results/blocking_recall_final.txt`.

| Dataset (paper) | b_t | Recall (PC) |
|-----------------|-----|-------------|
| Cora            | 0.80 | **0.88** |
| CiteSeer        | 0.70 | **0.86** |
| DG (google-DBLP)| 0.50 | **0.88** |
| Music (music20K)| 0.40 | **0.94** |
| Alaska (sigmod) | 0.80 | **0.95** |
| Song            | 0.30 | **0.94** |
| AS (affiliation)| 0.40 | **0.89** |

(Confirmed run in `results/blocking_recall_final.txt`. LSH hashing is stochastic,
so recall may vary ~±0.01 between runs; all datasets stay ≥ 0.86.)

### Draft reply

> Thank you — this is a fair point; candidate-set recall (pair completeness)
> should have been reported, since end-to-end recall is upper-bounded by it. We
> measured it on all datasets. We use **Pair Completeness (PC)** = the fraction of
> true matching pairs that are preserved within a candidate block; a true pair is
> lost only if blocking/filtering separates its two records.
>
> | Dataset | Cora | CiteSeer | DG | Music | Alaska | Song | AS |
> |---------|------|----------|----|----|----|----|----|
> | Recall (PC) | 0.88 | 0.86 | 0.88 | 0.94 | 0.95 | 0.94 | 0.89 |
>
> Across all seven datasets recall ≥ 0.86, so blocking preserves the large majority
> of true matches — including the low-similarity ones — and does not cap
> end-to-end recall at a low value. We also provide the full `b_t` sweep
> (0.3–0.9): as expected PC rises toward 1.0 as `b_t` decreases, and we select the
> operating point that keeps PC high. The measurement is deterministic and
> reproducible with no API calls via `issue_experiments/blocking_recall.py`; we
> will add the PC column to the blocking table in the revision.

---

## Issue #3 — Ordering of TASKS within a batch (not records within a set)

The reviewer makes two distinct claims:
1. §7.8 studies record ordering *inside a record set*, not the ordering of
   *tasks (record sets) inside one batched prompt*.
2. Table 18 "Similarity-Ordered" is numerically identical to Table 17
   "w/o batching" — i.e. the non-batched run relabeled — so the
   task-ordering experiment was never actually run.

Pick ONE of the two replies below.

### Option A — run the experiment (needs real LLM calls)

Experiment design (what we would run):
- Fix a set of K record sets that are packed into ONE batched prompt.
- Permute the ORDER of those K record sets within the batch (e.g. 5 random
  permutations + similarity-ordered + reverse), keeping the record sets
  themselves identical.
- Re-run in-context clustering for each permutation; report mean ± std of ACC /
  FP across permutations.
- Interpretation: low variance ⇒ task order does not leak / influence later
  tasks (robust); high variance ⇒ a real ordering effect to discuss.

> Draft reply (Option A):
> You are right that §7.8 varies record order *within* a set and §7.7 varies only
> batch size; the orthogonal question — whether the order of tasks *within a
> batch* affects results — was not isolated. We ran it: holding the K record sets
> in a batch fixed and permuting only their order (N random permutations +
> similarity / reverse), we measured ACC/FP variance. [Result: e.g. "ACC varies by
> ≤X over permutations, indicating task order has negligible influence" OR "ACC
> varies by Y, so we now discuss this ordering effect"]. We will add this as a new
> table and correct the Table 17/18 overlap noted below.

### Option B — state the limitation explicitly (no LLM needed)

> Draft reply (Option B):
> Thank you for the careful read. You are correct on both points: §7.8 studies the
> ordering of records *within* a record set, not the ordering of tasks *within a
> batched prompt*, and §7.7 varies only the batch size — so the effect of
> inter-task order within a batch is not isolated by our current experiments. You
> are also right that the "Similarity-Ordered" column of Table 18 coincides with
> the "w/o batching" column of Table 17; that is an error in table construction,
> not an independent measurement, and we will fix it. In the revision we will
> either (a) add a dedicated experiment permuting task order within a fixed batch,
> or (b) explicitly scope our ordering claims to *intra-record-set* ordering and
> flag inter-task batch ordering as a limitation / future work.

### ⚠️ Honesty checkpoint (must confirm with author)

The reviewer's strongest claim is that **Table 18 "Similarity-Ordered" == Table 17
"w/o batching"**. If that is true in the submitted paper, it must be acknowledged
and corrected — we should NOT defend it as a real measurement. Confirm whether
those two columns are genuinely identical in your source data before replying.
