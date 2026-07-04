"""
End-to-end pipeline test with a deterministic oracle LLM (no API key needed).

Builds a small synthetic dataset with a known ground truth, runs the full
NRS -> in-context clustering (+MDG) -> CMR pipeline per block with an oracle
that clusters by true entity, and checks that the recovered partition matches
ground truth (ACC == 1.0). This validates that the surrounding algorithms are
correct: if the LLM were perfect, the pipeline must recover the truth exactly.

Run:  python issue_experiments/test_end_to_end.py
"""

import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mock_llm import OracleLLM


def build_synthetic(n_entities=12, max_dup=4, dim=16, seed=7):
    """Create embeddings where each entity is a tight cluster in vector space."""
    rng = np.random.RandomState(seed)
    centers = rng.randn(n_entities, dim) * 5
    vectors, entity_of, rid = [], {}, 0
    for e in range(n_entities):
        k = rng.randint(1, max_dup + 1)
        for _ in range(k):
            vectors.append(centers[e] + rng.randn(dim) * 0.05)
            entity_of[rid] = e
            rid += 1
    return vectors, entity_of


def main():
    from sklearn.metrics.pairwise import cosine_similarity
    from llmcer.pipeline import run_blocks
    from llmcer.metrics import calculate_acc, calculate_fp_measure
    from llmcer import llm_interaction as li

    vectors, entity_of = build_synthetic()
    n = len(vectors)
    S = cosine_similarity(vectors)

    gt = {}
    for r, e in entity_of.items():
        gt.setdefault(e, []).append(r)
    ground_truth = [sorted(v) for v in gt.values()]

    blocks = [list(range(n))]

    oracle = OracleLLM(entity_of)
    orig = li._call_llm_classify
    li._call_llm_classify = lambda ids, df: oracle.cluster(ids, df)
    try:
        clusters, stats = run_blocks(vectors, S, blocks, df=None, parallel=False)
    finally:
        li._call_llm_classify = orig

    acc = calculate_acc(ground_truth, clusters)
    fp = calculate_fp_measure(ground_truth, clusters)
    print(f"records={n}  entities={len(ground_truth)}  predicted clusters={len(clusters)}")
    print(f"LLM calls={stats['api_calls']}  MDG interventions={stats['mdg_fails']}  "
          f"merge rounds={stats['rounds']}")
    print(f"ACC={acc:.4f}  FP-measure={fp:.4f}")

    ok = acc >= 0.999
    if not ok:
        pred_of = {}
        for ci, c in enumerate(clusters):
            for r in c:
                pred_of[r] = ci
        print("\n  Entities split across predicted clusters:")
        for e, recs in sorted(gt.items()):
            pcs = sorted({pred_of.get(r) for r in recs})
            if len(pcs) > 1:
                print(f"    entity {e}: records {sorted(recs)} -> predicted clusters {pcs}")
        print("\n  Predicted clusters with >1 entity (over-merge):")
        for ci, c in enumerate(clusters):
            ents = sorted({entity_of[r] for r in c})
            if len(ents) > 1:
                print(f"    cluster {ci}: records {sorted(c)} -> entities {ents}")

    print(f"\n[{'PASS' if ok else 'FAIL'}] oracle pipeline recovers ground truth "
          f"(ACC={acc:.4f})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
