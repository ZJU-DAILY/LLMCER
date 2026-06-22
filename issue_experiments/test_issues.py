"""
Reproduces each reported GitHub issue (#2 .. #9) and demonstrates the fix, in
GitHub issue-number order:

  GH#2  elbow_method always returns max_k
  GH#3  MDG uses mean intra/inter instead of min/max (Definition 1)
  GH#4  MDG never flags -- only sees representative-slice singletons
  GH#5  record-set regeneration on MDG failure not implemented
  GH#6  merge_2 does not implement CMR / Algorithm 3
  GH#7  NRS record-set creation (Algorithm 1) not faithfully implemented
  GH#8  merging across block boundaries (Algorithm 4)
  GH#9  ACC and NMI not implemented

Run:  python issue_experiments/test_issues.py

For every issue we (a) reproduce the buggy behaviour with a minimal
self-contained snippet matching the ORIGINAL code, then (b) call the FIXED
function from llmcer and show it now behaves as the paper specifies. No API key
or network is required -- the LLM is replaced by a deterministic oracle.
"""

import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PASS = "PASS"
FAIL = "FAIL"
results = []


def check(name, condition, detail=""):
    status = PASS if condition else FAIL
    results.append((name, status, detail))
    print(f"[{status}] {name}" + (f" -- {detail}" if detail else ""))


# ---------------------------------------------------------------------------
# GitHub #2: elbow_method always returns max_k
# ---------------------------------------------------------------------------
def issue1():
    print("\n=== GitHub #2: elbow_method always returns max_k ===")

    # OLD behaviour, reproduced exactly: argmin of a monotonically-decreasing
    # distortion list always lands on the largest k.
    distortions = [100, 60, 40, 30, 25]
    K = list(range(1, 6))
    old_k = K[int(np.argmin(distortions[1:])) + 1]
    print(f"  OLD: argmin(distortions[1:])+1 -> k={old_k} (== max_k=5, always)")
    check("GH#2 reproduced (old returns max_k)", old_k == 5)

    # FIXED behaviour: a clear elbow at k=3 in well-separated data should be found.
    from llmcer.clustering import elbow_method
    rng = np.random.RandomState(0)
    # 3 well-separated gaussian blobs in 8-d
    centers = np.array([[0]*8, [10]*8, [20]*8], dtype=float)
    pts = np.vstack([c + rng.randn(15, 8) * 0.3 for c in centers])
    new_k = elbow_method(pts, max_k=5)
    print(f"  NEW: elbow_method on 3 clear blobs -> k={new_k}")
    check("GH#2 fixed (finds true elbow ~3, not max_k)", 2 <= new_k <= 3, f"k={new_k}")


# ---------------------------------------------------------------------------
# Issue 2: mdg_check used means instead of min-intra/max-inter (Definition 1)
# ---------------------------------------------------------------------------
def issue2():
    print("\n=== GitHub #3: MDG uses mean intra/inter instead of min/max (Definition 1) ===")
    from llmcer.clustering import mdg_check

    # Build a sim matrix where record 2 is in cluster {0,1,2} but is actually
    # much closer to cluster {3,4}. Definition 1 must REJECT this.
    n = 5
    S = np.full((n, n), 0.1)
    np.fill_diagonal(S, 1.0)
    # cluster A = {0,1,2}: 0-1 very similar; 2 weakly attached
    S[0][1] = S[1][0] = 0.9
    S[0][2] = S[2][0] = 0.2
    S[1][2] = S[2][1] = 0.2
    # record 2 is strongly similar to cluster B = {3,4}
    S[2][3] = S[3][2] = 0.85
    S[2][4] = S[4][2] = 0.8
    S[3][4] = S[4][3] = 0.9

    clusters = [[0, 1, 2], [3, 4]]

    # OLD (mean-based) would compute avg_intra for record 2 = mean(0.2,0.2)=0.2,
    # avg_inter to B = mean(0.85,0.8)=0.825 -> 0.2<0.825 it would also catch this
    # one; the real difference shows on a record whose MIN intra is low but MEAN
    # intra is high. Show the discriminating case:
    # cluster A={0,1,2}: 2 has one very-similar peer and one dissimilar peer.
    S2 = np.full((n, n), 0.1)
    np.fill_diagonal(S2, 1.0)
    S2[0][1] = S2[1][0] = 0.9
    S2[0][2] = S2[2][0] = 0.95   # 2 very close to 0
    S2[1][2] = S2[2][1] = 0.30   # 2 far from 1  -> MIN intra = 0.30
    S2[2][3] = S2[3][2] = 0.50   # inter to B = 0.50
    S2[2][4] = S2[4][2] = 0.45
    S2[3][4] = S2[4][3] = 0.9
    clusters2 = [[0, 1, 2], [3, 4]]

    mean_intra_2 = np.mean([0.95, 0.30])      # 0.625  (old)
    min_intra_2 = min(0.95, 0.30)             # 0.30   (new, Def 1)
    max_inter_2 = max(0.50, 0.45)             # 0.50
    print(f"  record 2: mean_intra={mean_intra_2:.3f} (OLD), "
          f"min_intra={min_intra_2:.3f} (Def1), max_inter={max_inter_2:.3f}")
    print(f"  OLD test (mean<inter): {mean_intra_2:.3f} < {max_inter_2:.3f} -> "
          f"{mean_intra_2 < max_inter_2} (would ACCEPT, miss the error)")
    print(f"  NEW test (min<inter):  {min_intra_2:.3f} < {max_inter_2:.3f} -> "
          f"{min_intra_2 < max_inter_2} (correctly REJECTS)")

    accepted = mdg_check(clusters2, S2)
    check("GH#3 fixed (Def-1 min-intra rejects weak attachment)",
          accepted is False, f"mdg_check returned {accepted}")

    # And a genuinely good clustering must still pass.
    good = mdg_check([[0, 1], [3, 4]], S2)
    check("GH#3 (clean clustering still accepted)", good is True)


# ---------------------------------------------------------------------------
# Issues 3 & 4: MDG toothless on singleton slices + identical-prompt retry
# ---------------------------------------------------------------------------
def issue3_4():
    print("\n=== GitHub #4 & #5: MDG checks real record sets + record-set regeneration ===")
    from llmcer import llm_interaction as li
    from llmcer.clustering import mdg_check

    # Oracle that first returns a misclustering, then (after regeneration, i.e.
    # a *different* order) returns the correct clustering -- proving the retry is
    # not an identical prompt and that MDG is checked on the real record set.
    n = 5
    S = np.full((n, n), 0.1)
    np.fill_diagonal(S, 1.0)
    S[0][1] = S[1][0] = 0.95
    S[3][4] = S[4][3] = 0.95
    S[0][2] = S[2][0] = 0.2
    S[1][2] = S[2][1] = 0.2
    S[2][3] = S[3][2] = 0.9   # 2 truly belongs with {3,4}
    S[2][4] = S[4][2] = 0.85

    state = {"n": 0, "orders": []}

    def fake_call(record_ids, df):
        state["n"] += 1
        state["orders"].append(list(record_ids))
        if state["n"] == 1:
            clusters = [[0, 1, 2], [3, 4]]      # misclustering: 2 wrongly with {0,1}
        else:
            clusters = [[0, 1], [2, 3, 4]]      # corrected after regeneration
        stats = dict(api_calls=1, time=0.0, tokens=50, in_tokens=40, out_tokens=10)
        return clusters, stats

    orig = li._call_llm_classify
    li._call_llm_classify = fake_call
    try:
        clusters, stats = li.in_context_cluster([0, 1, 2, 3, 4], df=None,
                                                similarity_matrix=S, max_regen=2)
    finally:
        li._call_llm_classify = orig

    print(f"  LLM was called {state['n']} times; MDG interventions={stats['mdg_fails']}")
    print(f"  order attempt 1: {state['orders'][0]}")
    if len(state['orders']) > 1:
        print(f"  order attempt 2: {state['orders'][1]} (regenerated -- not identical)")
    check("GH#4 fixed (MDG fired on the real record set, not slice singletons)",
          stats['mdg_fails'] >= 1)
    check("GH#5 fixed (regeneration produced a DIFFERENT order, not identical retry)",
          len(state['orders']) > 1 and state['orders'][0] != state['orders'][1])
    check("GH#4/#5 (final clustering passes MDG)", mdg_check(clusters, S),
          f"final={clusters}")


# ---------------------------------------------------------------------------
# Issue 5: CMR is hierarchical (Algorithm 3), with anti-transitivity, not a
# one-shot threshold-band sweep with magic constants.
# ---------------------------------------------------------------------------
def issue5():
    print("\n=== GitHub #6: merge_2 does not implement CMR / Algorithm 3 ===")
    from llmcer.cluster_merge import cluster_merge, representative_of
    from mock_llm import OracleLLM

    # Three record sets, each already in-context clustered. The SAME entity 'A'
    # is split across all three record sets (clusters [0,1], [4,5], [8,9]) -- the
    # one-shot band-sweep could not reassemble these; hierarchical CMR must.
    # Anti-transitivity: two clusters from the same record set are never packed
    # together (they're already distinct entities).
    rs0 = [[0, 1], [2, 3]]      # entity A part, entity B
    rs1 = [[4, 5], [6, 7]]      # entity A part, entity C
    rs2 = [[8, 9], [10, 11]]    # entity A part, entity D
    initial = [rs0, rs1, rs2]

    entity_of = {0: 'A', 1: 'A', 4: 'A', 5: 'A', 8: 'A', 9: 'A',
                 2: 'B', 3: 'B', 6: 'C', 7: 'C', 10: 'D', 11: 'D'}
    n = 12
    # Embeddings: same-entity records share a center so representatives are close.
    centers = {'A': np.array([0.0] * 8), 'B': np.array([10.0] * 8),
               'C': np.array([20.0] * 8), 'D': np.array([30.0] * 8)}
    vectors = [centers[entity_of[r]] + 0.01 * (r % 2) for r in range(n)]
    from sklearn.metrics.pairwise import cosine_similarity
    S = cosine_similarity(vectors)

    oracle = OracleLLM(entity_of)

    def llm_fn(ids, df):
        return oracle.cluster(ids, df)

    final, stats = cluster_merge(initial, vectors, S, df=None,
                                 llm_cluster_fn=llm_fn, S_s=9, S_d=4)

    # Find the cluster containing record 0 (entity A) -- it must now contain all
    # six A records gathered from the three different record sets.
    a_cluster = next((c for c in final if 0 in c), [])
    print(f"  final clusters: {[sorted(c) for c in final]}")
    print(f"  merge rounds={stats['rounds']}  (was 1 in the buggy band-sweep)")
    check("GH#6 fixed (hierarchical CMR reassembled split entity A across 3 sets)",
          set(a_cluster) == {0, 1, 4, 5, 8, 9}, f"A-cluster={sorted(a_cluster)}")
    check("GH#6 (distinct entities B/C/D kept separate -- anti-transitivity)",
          len(final) == 4, f"#clusters={len(final)} (expected 4)")


# ---------------------------------------------------------------------------
# GitHub #7: NRS record-set creation (Algorithm 1) not faithfully implemented
# ---------------------------------------------------------------------------
def issue7_nrs():
    print("\n=== GitHub #7: NRS record-set creation (Algorithm 1) ===")
    from llmcer.record_set import create_record_sets, coefficient_of_variation

    # A block of 20 records over 4 entities (5 each). NRS must split it into
    # record sets that (a) respect the set-size constraint S_s, (b) cover up to
    # S_d entities, and (c) keep set variation S_v low. Embeddings: 4 tight
    # entity clusters so the diversity/k-means logic has real structure.
    S_s, S_d = 9, 4
    rng = np.random.RandomState(3)
    centers = {e: rng.randn(8) * 5 for e in range(4)}
    vectors, entity_of = [], {}
    rid = 0
    for e in range(4):
        for _ in range(5):
            vectors.append(centers[e] + rng.randn(8) * 0.05)
            entity_of[rid] = e
            rid += 1
    n = rid
    from sklearn.metrics.pairwise import cosine_similarity
    S = cosine_similarity(vectors)
    block = list(range(n))

    record_sets = create_record_sets(block, vectors, S, S_s, S_d)

    sizes = [len(rs) for rs in record_sets]
    all_recs = [r for rs in record_sets for r in rs]
    print(f"  block of {n} records -> {len(record_sets)} record sets, sizes={sizes}")

    # (a) every record placed exactly once (no loss, no duplication)
    check("GH#7 (NRS partitions the block exactly once -- no loss/dup)",
          sorted(all_recs) == block, f"placed={len(all_recs)} of {n}")
    # (b) set-size constraint S_s respected
    check("GH#7 (every record set respects S_s)", all(s <= S_s for s in sizes),
          f"sizes={sizes}")
    # (c) diversity: a large record set should contain more than one entity
    #     (the OLD elbow bug pinned k=max_k; here we just require >1 entity)
    biggest = max(record_sets, key=len)
    n_entities_in_big = len({entity_of[r] for r in biggest})
    check("GH#7 (record set is diversity-aware -- covers multiple entities)",
          n_entities_in_big >= 2, f"entities in biggest set={n_entities_in_big}")
    print(f"  biggest set covers {n_entities_in_big} distinct entities "
          f"(S_d target={S_d})")


# ---------------------------------------------------------------------------
# GitHub #8: merging operates across block boundaries (Algorithm 4)
# ---------------------------------------------------------------------------
def issue6():
    print("\n=== GitHub #8: merging across block boundaries (blocks stay isolated) ===")
    from llmcer.pipeline import run_blocks
    from mock_llm import OracleLLM

    # Two blocks. Entity ids chosen so that IF cross-block merging happened,
    # block 0's records could be pulled together with block 1's. We assert the
    # final clusters never span both blocks.
    block0 = [0, 1, 2]
    block1 = [3, 4, 5]
    blocks = [block0, block1]
    entity_of = {0: 'A', 1: 'A', 2: 'B', 3: 'C', 4: 'C', 5: 'D'}

    n = 6
    rng = np.random.RandomState(1)
    vectors = [rng.randn(8) for _ in range(n)]
    # high cross-block similarity to tempt an (incorrect) merge
    S = np.full((n, n), 0.95)
    np.fill_diagonal(S, 1.0)

    oracle = OracleLLM(entity_of)
    # monkeypatch the LLM calls used inside the pipeline
    from llmcer import llm_interaction as li
    from llmcer import pipeline as pl
    orig_call = li._call_llm_classify
    li._call_llm_classify = lambda ids, df: oracle.cluster(ids, df)
    try:
        clusters, stats = run_blocks(vectors, S, blocks, df=None, parallel=False)
    finally:
        li._call_llm_classify = orig_call

    def block_of(r):
        return 0 if r in block0 else 1
    spans = [c for c in clusters if len({block_of(r) for r in c}) > 1]
    print(f"  final clusters: {clusters}")
    check("GH#8 fixed (no cluster spans two blocks)", len(spans) == 0,
          f"cross-block clusters={spans}")


# ---------------------------------------------------------------------------
# Issue 7: ACC and NMI now implemented (Eq 2-3 / Section 6.1)
# ---------------------------------------------------------------------------
def issue7():
    print("\n=== GitHub #9: ACC (Eq 2-3) and NMI implemented ===")
    from llmcer.metrics import calculate_acc, calculate_nmi

    gt = [[0, 1, 2], [3, 4], [5]]
    perfect = [[0, 1, 2], [3, 4], [5]]
    acc_perfect = calculate_acc(gt, perfect)
    nmi_perfect = calculate_nmi(gt, perfect)
    print(f"  perfect prediction -> ACC={acc_perfect:.3f}, NMI={nmi_perfect:.3f}")
    check("GH#9 (ACC=1.0 on perfect match)", abs(acc_perfect - 1.0) < 1e-9)
    check("GH#9 (NMI=1.0 on perfect match)", abs(nmi_perfect - 1.0) < 1e-9)

    # A wrong prediction (everything lumped together) should score below 1.
    lumped = [[0, 1, 2, 3, 4, 5]]
    acc_lumped = calculate_acc(gt, lumped)
    print(f"  all-in-one prediction -> ACC={acc_lumped:.3f}")
    check("GH#9 (ACC < 1 on bad clustering)", acc_lumped < 1.0, f"ACC={acc_lumped:.3f}")


def main():
    # Run in GitHub issue order #2 .. #9.
    issue1()        # GH#2  elbow_method
    issue2()        # GH#3  MDG mean vs min/max
    issue3_4()      # GH#4 + GH#5  MDG on real sets + regeneration
    issue5()        # GH#6  CMR / Algorithm 3
    issue7_nrs()    # GH#7  NRS / Algorithm 1
    issue6()        # GH#8  block boundaries
    issue7()        # GH#9  ACC + NMI

    print("\n" + "=" * 50)
    n_pass = sum(1 for _, s, _ in results if s == PASS)
    print(f"SUMMARY: {n_pass}/{len(results)} checks passed")
    for name, status, detail in results:
        if status == FAIL:
            print(f"  FAILED: {name} ({detail})")
    print("=" * 50)
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
