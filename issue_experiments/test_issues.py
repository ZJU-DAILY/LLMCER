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


def issue1():
    print("\n=== GitHub #2: elbow_method always returns max_k ===")

    distortions = [100, 60, 40, 30, 25]
    K = list(range(1, 6))
    old_k = K[int(np.argmin(distortions[1:])) + 1]
    print(f"  OLD: argmin(distortions[1:])+1 -> k={old_k} (== max_k=5, always)")
    check("GH#2 reproduced (old returns max_k)", old_k == 5)

    from llmcer.clustering import elbow_method
    rng = np.random.RandomState(0)
    centers = np.array([[0]*8, [10]*8, [20]*8], dtype=float)
    pts = np.vstack([c + rng.randn(15, 8) * 0.3 for c in centers])
    new_k = elbow_method(pts, max_k=5)
    print(f"  NEW: elbow_method on 3 clear blobs -> k={new_k}")
    check("GH#2 fixed (finds true elbow ~3, not max_k)", 2 <= new_k <= 3, f"k={new_k}")


def issue2():
    print("\n=== GitHub #3: MDG uses mean intra/inter instead of min/max (Definition 1) ===")
    from llmcer.clustering import mdg_check

    n = 5
    S = np.full((n, n), 0.1)
    np.fill_diagonal(S, 1.0)
    S[0][1] = S[1][0] = 0.9
    S[0][2] = S[2][0] = 0.2
    S[1][2] = S[2][1] = 0.2
    S[2][3] = S[3][2] = 0.85
    S[2][4] = S[4][2] = 0.8
    S[3][4] = S[4][3] = 0.9

    clusters = [[0, 1, 2], [3, 4]]

    S2 = np.full((n, n), 0.1)
    np.fill_diagonal(S2, 1.0)
    S2[0][1] = S2[1][0] = 0.9
    S2[0][2] = S2[2][0] = 0.95
    S2[1][2] = S2[2][1] = 0.30
    S2[2][3] = S2[3][2] = 0.50
    S2[2][4] = S2[4][2] = 0.45
    S2[3][4] = S2[4][3] = 0.9
    clusters2 = [[0, 1, 2], [3, 4]]

    mean_intra_2 = np.mean([0.95, 0.30])
    min_intra_2 = min(0.95, 0.30)
    max_inter_2 = max(0.50, 0.45)
    print(f"  record 2: mean_intra={mean_intra_2:.3f} (OLD), "
          f"min_intra={min_intra_2:.3f} (Def1), max_inter={max_inter_2:.3f}")
    print(f"  OLD test (mean<inter): {mean_intra_2:.3f} < {max_inter_2:.3f} -> "
          f"{mean_intra_2 < max_inter_2} (would ACCEPT, miss the error)")
    print(f"  NEW test (min<inter):  {min_intra_2:.3f} < {max_inter_2:.3f} -> "
          f"{min_intra_2 < max_inter_2} (correctly REJECTS)")

    accepted = mdg_check(clusters2, S2)
    check("GH#3 fixed (Def-1 min-intra rejects weak attachment)",
          accepted is False, f"mdg_check returned {accepted}")

    good = mdg_check([[0, 1], [3, 4]], S2)
    check("GH#3 (clean clustering still accepted)", good is True)


def issue3_4():
    print("\n=== GitHub #4 & #5: MDG checks real record sets + record-set regeneration ===")
    from llmcer import llm_interaction as li
    from llmcer.clustering import mdg_check

    n = 5
    S = np.full((n, n), 0.1)
    np.fill_diagonal(S, 1.0)
    S[0][1] = S[1][0] = 0.95
    S[3][4] = S[4][3] = 0.95
    S[0][2] = S[2][0] = 0.2
    S[1][2] = S[2][1] = 0.2
    S[2][3] = S[3][2] = 0.9
    S[2][4] = S[4][2] = 0.85

    state = {"n": 0, "orders": []}

    def fake_call(record_ids, df):
        state["n"] += 1
        state["orders"].append(list(record_ids))
        if state["n"] == 1:
            clusters = [[0, 1, 2], [3, 4]]
        else:
            clusters = [[0, 1], [2, 3, 4]]
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


def issue5():
    print("\n=== GitHub #6: merge_2 does not implement CMR / Algorithm 3 ===")
    from llmcer.cluster_merge import cluster_merge, representative_of
    from mock_llm import OracleLLM

    rs0 = [[0, 1], [2, 3]]
    rs1 = [[4, 5], [6, 7]]
    rs2 = [[8, 9], [10, 11]]
    initial = [rs0, rs1, rs2]

    entity_of = {0: 'A', 1: 'A', 4: 'A', 5: 'A', 8: 'A', 9: 'A',
                 2: 'B', 3: 'B', 6: 'C', 7: 'C', 10: 'D', 11: 'D'}
    n = 12
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

    a_cluster = next((c for c in final if 0 in c), [])
    print(f"  final clusters: {[sorted(c) for c in final]}")
    print(f"  merge rounds={stats['rounds']}  (was 1 in the buggy band-sweep)")
    check("GH#6 fixed (hierarchical CMR reassembled split entity A across 3 sets)",
          set(a_cluster) == {0, 1, 4, 5, 8, 9}, f"A-cluster={sorted(a_cluster)}")
    check("GH#6 (distinct entities B/C/D kept separate -- anti-transitivity)",
          len(final) == 4, f"#clusters={len(final)} (expected 4)")


def issue7_nrs():
    print("\n=== GitHub #7: NRS record-set creation (Algorithm 1) ===")
    from llmcer.record_set import create_record_sets, coefficient_of_variation

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

    check("GH#7 (NRS partitions the block exactly once -- no loss/dup)",
          sorted(all_recs) == block, f"placed={len(all_recs)} of {n}")
    check("GH#7 (every record set respects S_s)", all(s <= S_s for s in sizes),
          f"sizes={sizes}")
    biggest = max(record_sets, key=len)
    n_entities_in_big = len({entity_of[r] for r in biggest})
    check("GH#7 (record set is diversity-aware -- covers multiple entities)",
          n_entities_in_big >= 2, f"entities in biggest set={n_entities_in_big}")
    print(f"  biggest set covers {n_entities_in_big} distinct entities "
          f"(S_d target={S_d})")


def issue6():
    print("\n=== GitHub #8: merging across block boundaries (blocks stay isolated) ===")
    from llmcer.pipeline import run_blocks
    from mock_llm import OracleLLM

    block0 = [0, 1, 2]
    block1 = [3, 4, 5]
    blocks = [block0, block1]
    entity_of = {0: 'A', 1: 'A', 2: 'B', 3: 'C', 4: 'C', 5: 'D'}

    n = 6
    rng = np.random.RandomState(1)
    vectors = [rng.randn(8) for _ in range(n)]
    S = np.full((n, n), 0.95)
    np.fill_diagonal(S, 1.0)

    oracle = OracleLLM(entity_of)
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

    lumped = [[0, 1, 2, 3, 4, 5]]
    acc_lumped = calculate_acc(gt, lumped)
    print(f"  all-in-one prediction -> ACC={acc_lumped:.3f}")
    check("GH#9 (ACC < 1 on bad clustering)", acc_lumped < 1.0, f"ACC={acc_lumped:.3f}")


def main():
    issue1()
    issue2()
    issue3_4()
    issue5()
    issue7_nrs()
    issue6()
    issue7()

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
