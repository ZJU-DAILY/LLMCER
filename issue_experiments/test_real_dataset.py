"""
REAL end-to-end test: real dataset records, real SBERT embeddings, real LLM
calls. No mock/oracle anywhere.

Because a full dataset means many paid LLM calls, this script samples a
controlled subset of *complete ground-truth entities* so the run is small,
cheap, and still exercises the whole NRS -> in-context clustering (+MDG) -> CMR
pipeline on genuine data. Increase --records to scale up toward a full run.

Usage:
  .venv/Scripts/python.exe issue_experiments/test_real_dataset.py \
      --dataset cora --records 60

  # full dataset (careful: many API calls):
  .venv/Scripts/python.exe issue_experiments/test_real_dataset.py \
      --dataset cora --records 0

Datasets (data csv, gt file) -- only the KEEP set from check_datasets.py:
  cora, citeseer, google-DBLP, music20K, sigmod, song, affiliation
"""

import os
import sys
import argparse
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATASETS = {
    "cora":        ("datasets/cora/cora.csv",                "datasets/cora/gt.csv"),
    "citeseer":    ("datasets/citesheer/Citesheer_dblp.csv", "datasets/citesheer/citesheer_gt.txt"),
    "google-DBLP": ("datasets/google-DBLP/data.csv",         "datasets/google-DBLP/gt.csv"),
    "music20K":    ("datasets/music20K/music20K.csv",        "datasets/music20K/ground_truth.txt"),
    "sigmod":      ("datasets/sigmod/alaska.csv",            "datasets/sigmod/alaska_gt.csv"),
    "song":        ("datasets/song/songs.csv",              "datasets/song/gt.txt"),
    "affiliation": ("datasets/affiliation/new_affi_data.csv","datasets/affiliation/new_mapping.csv"),
}


def sample_records(ground_truth, n_records, seed=0):
    """
    Take whole ground-truth entities (clusters) until we have ~n_records
    records. Keeping entities intact means the sampled subset has a meaningful
    ground truth. Returns the set of original record ids to keep.
    """
    rng = np.random.RandomState(seed)
    order = list(range(len(ground_truth)))
    rng.shuffle(order)
    keep = set()
    for i in order:
        if n_records and len(keep) >= n_records:
            break
        keep.update(ground_truth[i])
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cora", choices=list(DATASETS))
    ap.add_argument("--records", type=int, default=60,
                    help="approx #records to sample (0 = full dataset)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from llmcer.config import (BLOCK_THRESHOLD, SET_SIZE, SET_DIVERSITY,
                               OPENAI_API_KEY, OPENAI_MODEL, EMBEDDING_MODEL)
    if not OPENAI_API_KEY or OPENAI_API_KEY == "your_api_key_here":
        print("ERROR: OPENAI_API_KEY not set (put it in .env). Aborting.")
        return 2

    from llmcer.data_utils import get_ground_truth
    from llmcer.vectorization import cal_total_simi_vector
    from llmcer.clustering import lsh_block
    from llmcer.pipeline import run_blocks
    from llmcer.metrics import (calculate_acc, calculate_nmi, calculate_fp_measure,
                                calculate_ari)

    data_rel, gt_rel = DATASETS[args.dataset]
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(root, data_rel)
    gt_path = os.path.join(root, gt_rel)

    print(f"Dataset: {args.dataset}")
    print(f"Embedding model: {EMBEDDING_MODEL}")
    print(f"LLM: {OPENAI_MODEL}  | thresholds block={BLOCK_THRESHOLD} S_s={SET_SIZE} S_d={SET_DIVERSITY}")

    # 1. Ground truth (full), then sample whole entities down to ~--records.
    full_gt = get_ground_truth(gt_path)
    print(f"Full ground truth: {len(full_gt)} non-singleton clusters")

    keep = sample_records(full_gt, args.records, args.seed) if args.records else None

    # 2. Real embeddings + similarity for the FULL dataset, then subset.
    print("Embedding records with SBERT ...")
    t0 = time.time()
    vectors_all, simi_all, data = cal_total_simi_vector(data_path)
    print(f"  embedded {len(vectors_all)} records in {time.time()-t0:.1f}s")

    n_all = len(vectors_all)
    if keep is None:
        keep_idx = list(range(n_all))
    else:
        keep_idx = sorted(i for i in keep if 0 <= i < n_all)
    print(f"Using {len(keep_idx)} records "
          f"({'full dataset' if args.records == 0 else f'sampled ~{args.records}'}).")

    # Remap kept original indices -> contiguous 0..m-1 so the pipeline's
    # index space stays dense (vectors/simi/df are all re-indexed together).
    remap = {orig: new for new, orig in enumerate(keep_idx)}
    import numpy as _np
    vectors = [vectors_all[i] for i in keep_idx]
    simi = _np.array(simi_all)[_np.ix_(keep_idx, keep_idx)]
    sub_df = data.iloc[keep_idx].reset_index(drop=True)

    # Ground truth restricted & remapped to the subset.
    gt_sub = []
    for c in full_gt:
        members = [remap[r] for r in c if r in remap]
        if members:
            gt_sub.append(members)
    # add singletons for kept records not covered by GT pairs
    covered = {r for c in gt_sub for r in c}
    for new in range(len(keep_idx)):
        if new not in covered:
            gt_sub.append([new])

    # 3. Blocking on the subset.
    blocks = lsh_block(vectors, sub_df, BLOCK_THRESHOLD)
    print(f"LSH blocking -> {len(blocks)} blocks")

    # 4. Full pipeline with REAL LLM calls.
    print("Running pipeline with REAL LLM calls (this costs API tokens) ...")
    t0 = time.time()
    clusters, stats = run_blocks(vectors, simi, blocks, sub_df, parallel=False)
    wall = time.time() - t0

    # 5. Metrics.
    acc = calculate_acc(gt_sub, clusters)
    fp = calculate_fp_measure(gt_sub, clusters)
    nmi = calculate_nmi(gt_sub, clusters)
    ari = calculate_ari(gt_sub, clusters)

    print("\n" + "=" * 50)
    print(f"REAL-DATA RESULT  ({args.dataset}, {len(keep_idx)} records)")
    print("=" * 50)
    print(f"  predicted clusters : {len(clusters)}")
    print(f"  ground-truth clusters (incl. singletons): {len(gt_sub)}")
    print(f"  ACC        : {acc:.4f}")
    print(f"  FP-measure : {fp:.4f}")
    print(f"  NMI        : {nmi:.4f}")
    print(f"  ARI        : {ari:.4f}")
    print(f"  LLM calls  : {stats['api_calls']}   tokens: {stats['tokens']}")
    print(f"  MDG interventions: {stats['mdg_fails']}   merge rounds: {stats['rounds']}")
    print(f"  wall time  : {wall:.1f}s")
    print("=" * 50)
    return 0


if __name__ == "__main__":
    sys.exit(main())
