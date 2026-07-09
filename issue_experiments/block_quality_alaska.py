"""
Blocking quality for the Alaska (SIGMOD) dataset — to fill the missing Alaska
row in paper Table 4.

This measures the QUALITY OF THE BLOCKS THEMSELVES (no LLM, no matching): each
candidate block is treated as a predicted cluster and scored against ground
truth. We report:

  * Blocks_nums : number of blocks produced
  * FP          : FP-measure = harmonic mean of purity and inverse-purity
                  (the paper's "FP" column, llmcer.metrics.calculate_fp_measure)
  * F1          : pairwise F1 (pair precision / pair recall over record pairs),
                  for reference / cross-checking

Recall (Pair Completeness) is NOT recomputed here — it is already reported in the
blocking-recall experiment (Alaska/sigmod = 0.95).

Run:
  .venv/Scripts/python.exe issue_experiments/block_quality_alaska.py
  # or any dataset:
  .venv/Scripts/python.exe issue_experiments/block_quality_alaska.py --dataset sigmod --b_t 0.80
"""

import os
import sys
import argparse
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "block_quality")

# Alaska == the SIGMOD camera dataset in this repo.
DATASETS = {
    "sigmod": ("datasets/sigmod/alaska.csv", "datasets/sigmod/alaska_gt.csv"),
    "cora":   ("datasets/cora/cora.csv",     "datasets/cora/gt.csv"),
}
# best block_threshold per dataset (mirrors blocking-recall operating point)
BEST_BT = {"sigmod": 0.80, "cora": 0.80}


def embed(data_path):
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
    vecs = model.encode(df["combined_text"].tolist(), show_progress_bar=False)
    return list(vecs), df


def pairwise_f1(blocks, gt_clusters, n_records):
    """Pair precision / recall / F1 treating each block as a predicted cluster."""
    from collections import Counter
    block_of = {}
    for bi, blk in enumerate(blocks):
        for r in blk:
            block_of[int(r)] = bi

    # true pairs and how many are kept in a common block
    true_total = true_kept = 0
    for c in gt_clusters:
        m = len(c)
        if m < 2:
            continue
        true_total += m * (m - 1) // 2
        bc = Counter(block_of.get(int(r), f"__miss_{r}") for r in c)
        for cnt in bc.values():
            if cnt >= 2:
                true_kept += cnt * (cnt - 1) // 2

    cand_pairs = sum(len(b) * (len(b) - 1) // 2 for b in blocks)
    precision = true_kept / cand_pairs if cand_pairs else 0.0   # pair quality
    recall = true_kept / true_total if true_total else 0.0      # pair completeness
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return dict(pair_precision=precision, pair_recall=recall, pair_f1=f1,
                cand_pairs=cand_pairs, true_total=true_total, true_kept=true_kept)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="sigmod", choices=list(DATASETS))
    ap.add_argument("--b_t", type=float, default=None)
    args = ap.parse_args()

    os.makedirs(LOG_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(LOG_DIR, f"{args.dataset}_{ts}.log")
    log = open(log_path, "w", encoding="utf-8")

    def out(m=""):
        print(m); log.write(m + "\n"); log.flush()

    from llmcer.clustering import lsh_block
    from llmcer.data_utils import get_ground_truth
    from llmcer.metrics import calculate_fp_measure

    data_rel, gt_rel = DATASETS[args.dataset]
    b_t = args.b_t if args.b_t is not None else BEST_BT[args.dataset]

    out(f"# blocking quality (blocks-as-clusters, NO matching) | dataset={args.dataset} "
        f"b_t={b_t} ts={ts}")
    out(f"# Alaska == SIGMOD camera dataset" if args.dataset == "sigmod" else "")

    gt = get_ground_truth(gt_rel if os.path.isabs(gt_rel) else os.path.join(ROOT, gt_rel))
    vectors, df = embed(os.path.join(ROOT, data_rel))
    n = len(vectors)

    blocks = lsh_block(vectors, df, b_t)          # list of blocks (each = list of record ids)
    blocks = [b for b in blocks if b]
    n_blocks = len(blocks)
    n_nonsingleton = sum(1 for b in blocks if len(b) > 1)

    # blocks-as-clusters vs ground truth
    from llmcer.metrics import calculate_bcubed_metrics
    pred_clusters = [list(b) for b in blocks]
    fp = calculate_fp_measure(gt, pred_clusters)   # FP = purity-based FP-measure
    bc = calculate_bcubed_metrics(gt, pred_clusters)  # F1 = BCubed F1-score
    pw = pairwise_f1(blocks, gt, n)                # pairwise (reference only)

    out("")
    out(f"records            : {n}")
    out(f"Blocks_nums        : {n_blocks}   (non-singleton: {n_nonsingleton})")
    out(f"FP (FP-measure, purity) : {fp:.4f}")
    out(f"F1 (BCubed F1-score)    : {bc['f1']:.4f}   "
        f"(precision={bc['precision']:.4f}, recall={bc['recall']:.4f})")
    out(f"[ref] pairwise F1       : {pw['pair_f1']:.4f}")
    out("")
    out("Alaska blocking-quality row:")
    out(f"  Alaska | Blocks_nums={n_blocks} | FP={fp:.3f} | F1={bc['f1']:.3f}")
    out("")
    out("Note: 'FP' is the purity-based FP-measure; 'F1' is the BCubed F1-score. "
        "Blocks are scored as clusters (no LLM matching). Recall (pair "
        "completeness) is reported separately (Alaska/sigmod = 0.95).")
    log.close()
    print(f"\nLog: {os.path.relpath(log_path, ROOT)}")


if __name__ == "__main__":
    main()
