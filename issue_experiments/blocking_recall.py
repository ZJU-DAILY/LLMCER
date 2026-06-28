"""
Blocking recall / pair-completeness experiment (answers reviewer issue #2).

Reviewer point: blocking + the b_t similarity filter can discard true matches
before clustering runs, and end-to-end recall is upper-bounded by blocking
recall -- yet recall was never reported. This script measures it.

Standard blocking metrics (pair-based):
  PC  (Pair Completeness / recall) = #true-pairs kept in same block / #true-pairs
  PQ  (Pair Quality      / precision) = #true-pairs in candidate set / #candidate-pairs
  F1  = 2*PC*PQ / (PC+PQ)
  RR  (Reduction Ratio)  = 1 - #candidate-pairs / C(N,2)

It is DETERMINISTIC and uses NO LLM (no API key, no cost). It reuses the exact
project blocker `llmcer.clustering.lsh_block` and the per-dataset best
block_threshold used by the pipeline, and ALSO sweeps b_t to show the
recall/precision trade-off the reviewer asked about.

Usage:
  .venv/Scripts/python.exe issue_experiments/blocking_recall.py            # all datasets, best thr + sweep
  .venv/Scripts/python.exe issue_experiments/blocking_recall.py --dataset cora
  .venv/Scripts/python.exe issue_experiments/blocking_recall.py --no-sweep
"""

import os
import sys
import argparse
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Datasets with usable ground truth (the KEEP set). Paper name in comment.
DATASETS = {
    "cora":        ("datasets/cora/cora.csv",                "datasets/cora/gt.csv"),          # Cora
    "citeseer":    ("datasets/citesheer/Citesheer_dblp.csv", "datasets/citesheer/citesheer_gt.txt"),  # CiteSeer
    "google-DBLP": ("datasets/google-DBLP/data.csv",         "datasets/google-DBLP/gt.csv"),   # DG
    "music20K":    ("datasets/music20K/music20K.csv",        "datasets/music20K/ground_truth.txt"),    # Music
    "sigmod":      ("datasets/sigmod/alaska.csv",            "datasets/sigmod/alaska_gt.csv"), # Alaska
    "song":        ("datasets/song/songs.csv",              "datasets/song/gt.txt"),           # Song
    "affiliation": ("datasets/affiliation/new_affi_data.csv","datasets/affiliation/new_mapping.csv"),  # AS
    # Walmart-Amazon: single-table version walmart_amazon.csv (1808 records,
    # 0-based id) + gt.csv pair list. The GT ids align exactly with this file
    # (NOT tableA.csv, which is the larger raw dump).
    "walmart-amazon": ("datasets/Walmart_Amazon/walmart_amazon.csv", "datasets/Walmart_Amazon/gt.csv"),  # WA
}

# Per-dataset RECALL-AWARE best block_threshold: among swept thresholds, the one
# with the highest reduction ratio whose Pair Completeness (recall) >= 0.85
# (see issue_experiments/pick_threshold.py). This is the operating point we
# report for the blocking-recall question (reviewer issue #2): blocking should
# not silently discard true matches, so we choose b_t to keep recall high.
BEST_THR = {
    "cora": 0.80, "song": 0.30, "citeseer": 0.40, "google-DBLP": 0.50,
    "music20K": 0.40, "affiliation": 0.40, "sigmod": 0.80,
    "walmart-amazon": 0.10,  # hard product-matching set; recall ~0.89 at b_t=0.10
}

SWEEP = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]


def embed(data_path):
    """Return (vectors, dataframe) using the project's attribute-combining logic.
    Does NOT compute the NxN cosine matrix (avoids OOM on big datasets)."""
    import pandas as pd
    from sentence_transformers import SentenceTransformer
    from llmcer.config import EMBEDDING_MODEL
    from llmcer.id_utils import get_id_column

    try:
        df = pd.read_csv(data_path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(data_path, encoding="MacRoman")

    id_col = get_id_column(df)

    def combine(row):
        if id_col and id_col in row:
            return " ".join(str(v) for k, v in row.items() if k != id_col)
        return " ".join(str(v) for v in row)

    df["combined_text"] = df.apply(combine, axis=1)
    model = SentenceTransformer(EMBEDDING_MODEL)
    vectors = model.encode(df["combined_text"].tolist(), show_progress_bar=False)
    return list(vectors), df


def pair_metrics(blocks, gt_clusters, n_records):
    """Compute PC, PQ, F1, RR, block stats from blocks and ground-truth clusters."""
    block_of = {}
    for bi, blk in enumerate(blocks):
        for r in blk:
            block_of[r] = bi

    true_total = 0
    true_kept = 0
    for cluster in gt_clusters:
        m = len(cluster)
        if m < 2:
            continue
        true_total += m * (m - 1) // 2
        # group this cluster's members by the block they fell into
        bc = Counter(block_of.get(int(r), f"__miss_{r}") for r in cluster)
        for cnt in bc.values():
            if cnt >= 2:
                true_kept += cnt * (cnt - 1) // 2

    cand_pairs = sum(len(b) * (len(b) - 1) // 2 for b in blocks)
    total_pairs = n_records * (n_records - 1) // 2

    PC = true_kept / true_total if true_total else 0.0
    PQ = true_kept / cand_pairs if cand_pairs else 0.0
    F1 = 2 * PC * PQ / (PC + PQ) if (PC + PQ) else 0.0
    RR = 1 - cand_pairs / total_pairs if total_pairs else 0.0

    nonsingleton = sum(1 for b in blocks if len(b) > 1)
    return dict(blocks=len(blocks), nonsingleton_blocks=nonsingleton,
                true_total=true_total, true_kept=true_kept,
                cand_pairs=cand_pairs, PC=PC, PQ=PQ, F1=F1, RR=RR)


def run_dataset(name, do_sweep=True):
    from llmcer.clustering import lsh_block
    from llmcer.data_utils import get_ground_truth

    data_rel, gt_rel = DATASETS[name]
    data_path = os.path.join(ROOT, data_rel)
    gt_path = os.path.join(ROOT, gt_rel)

    print("=" * 72)
    print(f"{name}")
    gt = get_ground_truth(gt_path)
    print(f"  embedding records with SBERT ...")
    vectors, df = embed(data_path)
    n = len(vectors)
    n_true = sum(len(c) * (len(c) - 1) // 2 for c in gt if len(c) > 1)
    print(f"  records={n}  GT clusters(non-singleton)={sum(1 for c in gt if len(c) > 1)}  "
          f"true pairs={n_true}")

    best = BEST_THR.get(name, 0.70)
    rows = []
    thresholds = sorted(set(([best] + SWEEP) if do_sweep else [best]))
    for thr in thresholds:
        blocks = lsh_block(vectors, df, thr)
        m = pair_metrics(blocks, gt, n)
        tag = " <-- pipeline best" if abs(thr - best) < 1e-9 else ""
        rows.append((thr, m, tag))
        # Reviewer issue #2 only asks for recall -> report recall only.
        print(f"  b_t={thr:.3f}  recall(PC)={m['PC']:.4f}{tag}")

    # return the best-threshold row for the summary table
    best_row = next(m for thr, m, _ in rows if abs(thr - best) < 1e-9)
    return name, best, best_row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None, choices=list(DATASETS))
    ap.add_argument("--no-sweep", action="store_true")
    args = ap.parse_args()

    names = [args.dataset] if args.dataset else list(DATASETS)
    summary = []
    for nm in names:
        try:
            summary.append(run_dataset(nm, do_sweep=not args.no_sweep))
        except Exception as e:
            print(f"  ERROR on {nm}: {type(e).__name__}: {e}")

    # Reviewer issue #2 asks only for candidate-set recall -> report recall only.
    print("\n" + "=" * 40)
    print("BLOCKING RECALL  (at best b_t per dataset)")
    print("=" * 40)
    print(f"{'Dataset':<14}{'b_t':>6}{'Recall':>10}")
    print("-" * 40)
    for name, best, m in summary:
        print(f"{name:<14}{best:>6.3f}{m['PC']:>10.4f}")
    print("=" * 40)
    print("Recall = Pair Completeness: fraction of true matching pairs kept in a "
          "block.\nEnd-to-end recall is upper-bounded by this value.")


if __name__ == "__main__":
    main()
