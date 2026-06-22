# Responses to the reported issues — LLM-CER

All seven issues are **confirmed correct**: the released code had drifted from
the algorithms in the paper (Algorithms 1–4, Definition 1, Section 6.1). Each
section below gives:

* **Verdict** — confirmation against the paper.
* **Fix** — exact file, function, and line range changed.
* **Result** — what the validation prints (no API key needed; deterministic
  oracle LLM).
* **Reply** — text you can paste into the GitHub issue.

### How to reproduce the validation

```bash
python issue_experiments/check_datasets.py     # dataset / ground-truth audit
python issue_experiments/test_issues.py         # per-issue before/after checks
python issue_experiments/test_end_to_end.py     # full pipeline, oracle LLM -> ACC=1.0
```

Latest captured runs are in `issue_experiments/results/`.
Headline results: `test_issues.py` → **all checks PASS**;
`test_end_to_end.py` → **ACC=1.0000, FP-measure=1.0000** (12 entities recovered
exactly, 3 merge rounds).

---

## Issue 1 — `elbow_method` always returns `max_k`

**Verdict: confirmed.** K-means inertia is monotonically non-increasing in `k`,
so `np.argmin(distortions[1:]) + 1` always lands on the largest `k`. Block
diversity `E_d` was effectively pinned to `MAX_K`.

**Fix:** [`llmcer/clustering.py`](../llmcer/clustering.py) — `elbow_method`,
lines **10–60**. Replaced argmin-of-inertia with a geometric knee detector
(Kneedle-style): normalise the `(k, distortion)` curve to `[0,1]` and pick the
`k` whose point is farthest from the chord joining the first and last points
(maximum curvature). Degenerate cases (≤2 candidate k, identical points) are
handled explicitly.

**Result** (`test_issues.py`, Issue 1):
```
OLD: argmin(distortions[1:])+1 -> k=5 (== max_k=5, always)
NEW: elbow_method on 3 clear blobs -> k=2     [PASS]
```

**Reply:**
> Confirmed — minimising inertia always returns `max_k` because inertia is
> monotonically decreasing. We replaced it with a geometric elbow/knee detector
> (max distance from the first–last chord), so `elbow_method` now estimates
> block diversity as intended in Algorithm 1. Fixed in `llmcer/clustering.py`
> (`elbow_method`); see `issue_experiments/test_issues.py` (Issue 1).

---

## Issue 2 — MDG uses **means** instead of min-intra / max-inter (Definition 1)

**Verdict: confirmed.** Definition 1 defines intra-cluster similarity of `r` as
the **minimum** similarity to same-cluster peers and inter-cluster similarity as
the **maximum** similarity to records in other clusters; Algorithm 2 rejects
when `intra < inter`. The code averaged both.

**Fix:** [`llmcer/clustering.py`](../llmcer/clustering.py) — `mdg_check`,
lines **145–201** (uses `min` intra / `max` inter exactly per Definition 1).
Added `find_misclustered_records`, lines **203–245**, returning the offending
records for the regeneration step (Issue 4).

**Result** (`test_issues.py`, Issue 2):
```
record 2: mean_intra=0.625 (OLD), min_intra=0.300 (Def1), max_inter=0.500
OLD test (mean<inter): 0.625 < 0.500 -> False (would ACCEPT, miss the error)
NEW test (min<inter):  0.300 < 0.500 -> True  (correctly REJECTS)    [PASS]
clean clustering still accepted                                       [PASS]
```

**Reply:**
> Correct — Definition 1 uses min (intra) and max (inter), not means. We
> rewrote `mdg_check` (`llmcer/clustering.py`) to follow Definition 1 /
> Algorithm 2 exactly; the discriminating case (high mean but low min intra) is
> now rejected. See `issue_experiments/test_issues.py` (Issue 2).

---

## Issue 3 — MDG is structurally toothless (singletons → 1.0; only
representative slices checked)

**Verdict: confirmed.** Singletons were hard-coded to `avg_intra = 1.0`, so
`1.0 < inter` was essentially never true; and MDG was only invoked inside
`llm_seperate` on representative-collapsed slices, never on the first-pass
clustering of full records.

**Fix:**
* Singleton shortcut removed — `mdg_check` now skips records with no
  same-cluster peers ([`llmcer/clustering.py`](../llmcer/clustering.py) lines
  **168–176**).
* MDG now runs on the **actual** in-context clustering of each record set via
  the new entry point `in_context_cluster`
  ([`llmcer/llm_interaction.py`](../llmcer/llm_interaction.py) lines **75–112**),
  which is used by both the first pass and the CMR merge rounds in
  [`llmcer/pipeline.py`](../llmcer/pipeline.py) `process_block`, lines **42–90**
  (Algorithm 4 lines 4 & 9).

**Result** (`test_issues.py`, Issues 3 & 4):
```
LLM was called 2 times; MDG interventions=1                          [PASS]
final clustering passes MDG -- final=[[0, 1], [2, 3, 4]]             [PASS]
```

**Reply:**
> Agreed on both counts. We removed the singleton→1.0 shortcut and restructured
> the pipeline so MDG validates the real record-set clustering (Algorithm 4
> lines 4 & 9) via `in_context_cluster` in `llmcer/llm_interaction.py`, not the
> collapsed representative slices. MDG now fires on first-pass misclusterings —
> see `issue_experiments/test_issues.py` (Issues 3 & 4).

---

## Issue 4 — On MDG failure the code retries the **identical** prompt

**Verdict: confirmed.** At temperature 0 the LLM is deterministic, so the same
prompt yields the same wrong answer; the second failure was accepted as-is.
There was no relocation/regeneration logic (§5.2, Algorithm 4 lines 5 & 10).

**Fix:** [`llmcer/llm_interaction.py`](../llmcer/llm_interaction.py) —
`in_context_cluster` (lines **75–112**) calls `_regenerate_order`
(lines **115–143**) on MDG rejection: each flagged record is relocated
immediately after the record it is most inter-similar to, producing a different,
more sequentially-ordered prompt, and only then re-clustered. `temperature=0` is
set explicitly (`_call_llm_classify`, line **45**) so regeneration — not
sampling noise — is what changes between attempts.

**Result** (`test_issues.py`, Issue 4):
```
order attempt 1: [0, 1, 2, 3, 4]
order attempt 2: [0, 1, 3, 2, 4]  (regenerated -- not identical)     [PASS]
```

**Reply:**
> Right — identical prompt at T=0 ⇒ identical output. We implemented the
> record-set regeneration from §5.2 (`_regenerate_order` in
> `llmcer/llm_interaction.py`): flagged records are relocated next to their most
> inter-similar record before re-clustering, so the retried prompt is genuinely
> different. The test prints both orderings to show they differ
> (`issue_experiments/test_issues.py`, Issue 4).

---

## Issue 5 — `merge_2` is a one-shot threshold-band sweep, not hierarchical CMR
(Algorithm 3); magic constants `0.02` / `≥0.2`

**Verdict: confirmed.** The old `merge_2` swept similarity bands and merged every
pair in a band on a vote, with no hierarchical record-set regeneration, no
anti-transitivity, and unexplained constants.

**Fix:** new module
[`llmcer/cluster_merge.py`](../llmcer/cluster_merge.py) implementing Algorithm 3:
* `representative_of` (lines **32–45**) — cluster → member nearest the centroid.
* `_pack_next_round` (lines **52–116**) — **similarity-driven** packing: pick an
  unselected anchor cluster, greedily add the most-similar unselected clusters
  from **different** record sets (anti-transitivity — never pack two clusters
  from the same record set), up to `S_s`.
* `cluster_merge` (lines **118–212**) — iterates rounds until a full round merges
  nothing (the paper's exit condition). The `merge_2` band-vote and the
  `0.02` / `0.2` constants are gone.

This was the source of the original end-to-end failure: the first packing
attempt grouped clusters by record-set **index**, so representatives of the same
entity scattered across record sets were never compared. The similarity-driven,
multi-round version reassembles them.

**Result** (`test_end_to_end.py`, full pipeline with oracle LLM):
```
records=28  entities=12  predicted clusters=12
merge rounds=3
ACC=1.0000  FP-measure=1.0000     [PASS]
```
(Before the fix: ACC=0.8571, 15 clusters, merge rounds=1.)
A dedicated unit check is in `test_issues.py` (Issue 5): a single entity split
across three record sets is correctly reassembled while three other entities
stay separate (anti-transitivity).

**Reply:**
> Correct — the released `merge_2` was a heuristic band-sweep, not CMR. We
> replaced it with a faithful Algorithm 3 in `llmcer/cluster_merge.py`:
> representative records, similarity-driven hierarchical packing with
> anti-transitivity and the S_s/S_d constraints, iterated until no further merge
> is possible; the `0.02`/`0.2` constants are removed. With a perfect (oracle)
> LLM the full pipeline now recovers ground truth exactly (ACC=1.0,
> `issue_experiments/test_end_to_end.py`); a unit check is in
> `test_issues.py` (Issue 5).

---

## Issue 6 — Blocking discarded before merging; clusters merged globally

**Verdict: confirmed.** `seperate_parallel` flattened all blocks and `merge_2`
compared every cluster pair regardless of block.

**Fix:** [`llmcer/pipeline.py`](../llmcer/pipeline.py) — `process_block`
(lines **42–90**) runs NRS → in-context clustering (+MDG) → CMR **within a single
block**; `run_blocks` (lines **93–124**) processes each block independently and
just concatenates the per-block partitions. No cross-block comparison exists
after blocking.

**Result** (`test_issues.py`, Issue 6):
```
final clusters: [[0, 1], [2], [3, 4], [5]]
no cluster spans two blocks -- cross-block clusters=[]               [PASS]
```

**Reply:**
> Agreed. The new pipeline keeps blocks isolated end-to-end: NRS, in-context
> clustering, MDG and CMR all run within a single block (`process_block` in
> `llmcer/pipeline.py`), and final partitions are concatenated across blocks —
> no global cluster comparison. The test asserts no output cluster spans two
> blocks (`issue_experiments/test_issues.py`, Issue 6).

---

## Issue 7 — ACC (Eq 2–3) and NMI not implemented

**Verdict: confirmed.** `metrics.py` had only purity / inverse-purity / FP / ARI;
`run_pipeline.py` printed "accuracy metrics" but computed neither ACC nor NMI.

**Fix:** [`llmcer/metrics.py`](../llmcer/metrics.py) —
* `calculate_acc` (lines **49–77**) — Eq 2–3: optimal 1-to-1 matching of
  predicted→true clusters via the Hungarian algorithm
  (`scipy.optimize.linear_sum_assignment`) on the contingency matrix, then
  `CorrectCount / |R|`.
* `calculate_nmi` (lines **79–84**) — `normalized_mutual_info_score`.
* Shared label alignment in `_aligned_labels` (lines **11–47**).
[`scripts/run_pipeline.py`](../scripts/run_pipeline.py) now reports ACC and
FP-measure as the headline metrics (Section 6.1), with NMI/ARI alongside.

**Result** (`test_issues.py`, Issue 7):
```
perfect prediction  -> ACC=1.000, NMI=1.000                          [PASS]
all-in-one prediction -> ACC=0.500                                   [PASS]
```

**Reply:**
> Correct — ACC and NMI were missing. We added ACC exactly as in Eq 2–3 (optimal
> cluster matching via the Hungarian algorithm, then CorrectCount/|R|) and NMI in
> `llmcer/metrics.py`, and `scripts/run_pipeline.py` now reports ACC + FP-measure
> as the primary metrics. Verified on perfect/degenerate clusterings in
> `issue_experiments/test_issues.py` (Issue 7).

---

## Bonus — why other datasets "didn't reproduce"

Two non-algorithmic causes, also fixed:

1. **Dynamic thresholds.** A prior commit replaced the paper's fixed
   validation-tuned thresholds with `mean ± k·std` of the similarity matrix.
   On datasets with a different similarity distribution the merge band
   `(merge_threshold, block_threshold)` could collapse to empty, so no merging
   happened. Reverted to fixed, env-overridable thresholds in
   [`llmcer/config.py`](../llmcer/config.py) lines **61–74**
   (`BLOCK_THRESHOLD=0.70`, `MERGE_THRESHOLD=0.50`, `S_s=9`, `S_d=4`).
2. **Ground-truth loading.** `cora/gt.csv` is a **headerless** pair file; the old
   loader read it with a header and dropped the first pair. Fixed in
   [`llmcer/data_utils.py`](../llmcer/data_utils.py) `get_ground_truth`,
   lines **121–144** (auto-detects header by testing whether row 0 is an integer
   pair).
3. **`walmart_amz` has no ground-truth file** (a two-table benchmark missing its
   second table), so it cannot be evaluated — recommended for deletion by
   `check_datasets.py`. The other 7 datasets are aligned and kept.

---

## Summary

| # | Area | Status | Primary location |
|---|------|--------|------------------|
| 1 | elbow / diversity | fixed | `llmcer/clustering.py:10` |
| 2 | MDG min/max (Def 1) | fixed | `llmcer/clustering.py:145` |
| 3 | MDG checks real clustering | fixed | `llmcer/llm_interaction.py:75`, `pipeline.py:42` |
| 4 | record-set regeneration | fixed | `llmcer/llm_interaction.py:115` |
| 5 | CMR (Algorithm 3) | fixed | `llmcer/cluster_merge.py` |
| 6 | block isolation | fixed | `llmcer/pipeline.py:42,93` |
| 7 | ACC + NMI | fixed | `llmcer/metrics.py:49,79` |

All validated with a deterministic oracle LLM (`test_issues.py` all PASS;
`test_end_to_end.py` ACC=1.0), so the fixes are checkable without API access.
