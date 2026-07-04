# Ready-to-paste GitHub replies (issues #2–#9)

Each block below is a self-contained English reply you can paste directly into
the corresponding GitHub issue. All fixes are pushed to `main`; all are validated
without an API key by `issue_experiments/test_issues.py`
(**16/16 checks pass**) and `issue_experiments/test_end_to_end.py`
(full pipeline with an oracle LLM → **ACC = 1.0**).

---

## Issue #2 — `elbow_method` always returns `max_k`

> Thanks, this is correct. K-means inertia is monotonically non-increasing in
> `k`, so `np.argmin(distortions[1:]) + 1` always selects the largest `k` and the
> function never finds an elbow — block diversity was effectively pinned to
> `MAX_K`.
>
> **Fix** (`llmcer/clustering.py`, `elbow_method`, lines 10–60): we replaced the
> argmin-of-inertia rule with a geometric knee detector (Kneedle-style). We
> normalise the `(k, distortion)` curve to `[0, 1]` and pick the `k` whose point
> is farthest from the chord joining the first and last points, i.e. the point of
> maximum curvature / diminishing returns. Degenerate cases (≤ 2 candidate `k`,
> identical points) are handled explicitly.
>
> **Validation** (`issue_experiments/test_issues.py`, GH#2):
> ```
> OLD: argmin(distortions[1:])+1 -> k=5 (== max_k, always)
> NEW: elbow_method on 3 well-separated blobs -> k≈2–3   [PASS]
> ```

---

## Issue #3 — MDG uses mean intra/inter similarity instead of min/max (Definition 1)

> Correct. Definition 1 defines the intra-cluster similarity of a record `r` as
> the **minimum** similarity to its same-cluster peers, and the inter-cluster
> similarity as the **maximum** similarity to records in other clusters;
> Algorithm 2 rejects the clustering when `intra < inter`. The implementation
> averaged both, which changes the decision boundary and misses real
> misclusterings.
>
> **Fix** (`llmcer/clustering.py`, `mdg_check`, lines 145–201): `mdg_check` now
> computes `min` intra and `max` inter exactly per Definition 1 and rejects when
> any non-singleton record has `min_intra < max_inter`. We also added
> `find_misclustered_records` (lines 203–245) which returns the offending records
> for the regeneration step (see #5).
>
> **Validation** (`test_issues.py`, GH#3) — a record with high *mean* but low
> *min* intra-similarity:
> ```
> record 2: mean_intra=0.625 (OLD), min_intra=0.300 (Def 1), max_inter=0.500
> OLD (mean<inter): 0.625<0.500 -> False  (would wrongly ACCEPT)
> NEW (min <inter): 0.300<0.500 -> True   (correctly REJECTS)   [PASS]
> clean clustering still accepted                               [PASS]
> ```

---

## Issue #4 — MDG never flags a misclustering because it only sees representative-slice singletons

> Agreed. There were two compounding problems: (1) singleton clusters were
> hard-coded to `avg_intra = 1.0`, so `1.0 < inter` was essentially never true;
> and (2) MDG was only invoked deep inside `llm_seperate`, on the
> representative-collapsed slices, never on the first-pass in-context clustering
> of full records where the misclusterings actually occur.
>
> **Fix:**
> - The singleton shortcut is removed — `mdg_check` now skips records that have no
>   same-cluster peers (`llmcer/clustering.py`, lines ~168–176).
> - MDG now runs on the **actual** in-context clustering of each record set, via a
>   new single entry point `in_context_cluster`
>   (`llmcer/llm_interaction.py`, lines 81–119), which is used by both the
>   first-pass clustering and the CMR merge rounds in
>   `llmcer/pipeline.py::process_block` (lines 42–90), matching Algorithm 4
>   lines 4 & 9.
>
> **Validation** (`test_issues.py`, GH#4): on a record set whose first clustering
> is a genuine misclustering, MDG fires (`MDG interventions = 1`) and the final
> clustering passes MDG. `[PASS]`

---

## Issue #5 — Record-set regeneration on MDG failure is not implemented (slice retried unchanged)

> Correct — at temperature 0 the LLM is deterministic, so re-issuing the
> identical prompt returns the identical (still-misclustered) output, and there
> was no relocation logic.
>
> **Fix** (`llmcer/llm_interaction.py`): on MDG rejection, `in_context_cluster`
> (lines 81–119) calls `_regenerate_order` (lines 121–149), which implements the
> §5.2 *Record Set Regeneration*: each flagged record is relocated immediately
> after the record it is most inter-similar to, producing a different, more
> sequentially-ordered record set, and only then is it re-clustered. We also set
> `temperature=0` explicitly (`_call_llm_classify`, line 51) so that the
> regenerated ordering — not sampling noise — is what changes between attempts.
>
> **Validation** (`test_issues.py`, GH#5):
> ```
> order attempt 1: [0, 1, 2, 3, 4]
> order attempt 2: [0, 1, 3, 2, 4]   (regenerated -- not identical)   [PASS]
> ```

---

## Issue #6 — `merge_2` does not implement the CMR / hierarchical merging algorithm (Algorithm 3)

> Correct. The released `merge_2` was a one-shot threshold-band sweep with a vote
> and magic constants (`0.02`, `≥0.2`); it had no hierarchical record-set
> regeneration and no anti-transitivity.
>
> **Fix:** new module `llmcer/cluster_merge.py` implementing Algorithm 3:
> - `representative_of` (lines 32–45): each cluster → the member nearest its
>   centroid.
> - `_pack_next_round` (lines 52–116): **similarity-driven** packing — pick an
>   unselected anchor cluster, then greedily add the most-similar unselected
>   clusters drawn from **different** record sets (anti-transitivity: two clusters
>   from the same record set are never packed together), up to `S_s`.
> - `cluster_merge` (lines 118–212): iterates rounds, the LLM re-clusters each new
>   record set, grouped representatives union their underlying clusters, and the
>   process stops when a full round merges nothing (the paper's exit condition).
>   The band-vote and the `0.02`/`0.2` constants are gone.
>
> This was also the root cause of the end-to-end failure: the first packing
> attempt grouped clusters by record-set *index*, so representatives of the same
> entity scattered across record sets were never compared. The similarity-driven,
> multi-round version reassembles them.
>
> **Validation:** with a perfect (oracle) LLM the full pipeline now recovers the
> ground truth exactly — `ACC = 1.0`, `FP-measure = 1.0`, 12 entities → 12
> clusters, 3 merge rounds (`issue_experiments/test_end_to_end.py`; before the
> fix: ACC = 0.857, 15 clusters, 1 round). A dedicated unit check
> (`test_issues.py`, GH#6) splits one entity across three record sets and confirms
> CMR reassembles it while keeping the other entities separate. `[PASS]`

---

## Issue #7 — NRS record-set creation (Algorithm 1) is not faithfully implemented

> Agreed. The previous code did not construct record sets per Algorithm 1 — it
> relied on the broken `elbow_method` for diversity and did not enforce the
> set-size / diversity / variation constraints.
>
> **Fix:** new module `llmcer/record_set.py` implementing Algorithm 1:
> - `next_record_set` (lines 71–156): for a block larger than `S_s`, estimate
>   diversity `k` with the (now fixed) elbow method, k-means the remaining
>   records, seed the set with `⌊S_s/S_d⌋` representatives per cluster for
>   diversity, then fill up to `S_s` by adding the record that least increases the
>   set variation `S_v` (coefficient of variation of cluster sizes,
>   `coefficient_of_variation`, lines 21–35, = Eq. 1), and finally order similar
>   records together (sequential record order).
> - `create_record_sets` (lines 158–174): repeatedly applies NRS until the block
>   is exhausted.
> - Defaults follow the paper's optima: `S_s = 9`, `S_d = 4`
>   (`llmcer/config.py`).
>
> **Validation** (`test_issues.py`, GH#7): on a 20-record block over 4 entities,
> NRS partitions every record exactly once, every record set respects `S_s`, and
> the largest record set is diversity-aware (covers multiple entities). `[PASS]`

---

## Issue #8 — Merging operates across block boundaries, violating the blocking partition (Algorithm 4)

> Correct. `seperate_parallel` flattened all blocks and `merge_2` then compared
> every cluster pair regardless of block, breaking the hard-partition assumption
> and the `O(|R|·b)` complexity argument.
>
> **Fix** (`llmcer/pipeline.py`): the pipeline now processes each block
> independently. `process_block` (lines 42–90) runs NRS → in-context clustering
> (+ MDG) → CMR entirely **within a single block**, and `run_blocks`
> (lines 93–124) simply concatenates the per-block partitions. No cross-block
> cluster comparison exists anywhere after blocking.
>
> **Validation** (`test_issues.py`, GH#8): with two blocks and deliberately high
> cross-block similarity, no output cluster ever spans two blocks. `[PASS]`

---

## Issue #9 — ACC and NMI metrics are not implemented

> Correct — `metrics.py` only had purity / inverse-purity / FP-measure / ARI, and
> `run_pipeline.py` printed "accuracy metrics" without computing ACC or NMI.
>
> **Fix** (`llmcer/metrics.py`):
> - `calculate_acc` (lines 49–77) implements ACC exactly as in Eq. (2)–(3):
>   the predicted clusters are optimally matched 1-to-1 to ground-truth clusters
>   by maximising total intersection (Hungarian algorithm via
>   `scipy.optimize.linear_sum_assignment` on the contingency matrix), then
>   `CorrectCount / |R|`.
> - `calculate_nmi` (lines 79–84) uses `normalized_mutual_info_score`.
> - `scripts/run_pipeline.py` now reports ACC and FP-measure as the headline
>   metrics (Section 6.1), with NMI and ARI alongside.
>
> **Validation** (`test_issues.py`, GH#9):
> ```
> perfect prediction   -> ACC=1.000, NMI=1.000   [PASS]
> all-in-one prediction -> ACC=0.500              [PASS]
> ```

---

### One general note you may want to add to the repo / a pinned issue

Two non-algorithmic causes of "results don't reproduce" were also fixed:
(1) a prior commit had replaced the paper's fixed, validation-tuned similarity
thresholds with data-dependent `mean ± k·std` values, which could collapse the
merge band to empty on datasets with a different similarity distribution — now
reverted to fixed, env-overridable thresholds in `llmcer/config.py`; and
(2) `cora/gt.csv` is a headerless pair file that the loader read as having a
header (dropping the first pair) — fixed in `llmcer/data_utils.py`
(`get_ground_truth` now auto-detects the header). The `Walmart_Amazon` dataset
ships without a ground-truth file (a two-table benchmark missing its second
table) and cannot be evaluated.
