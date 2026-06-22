
import sys
import os
import time
import pandas as pd
import numpy as np

# Add parent directory to path to import llmcer
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llmcer.config import (DATASET_PATH, GROUND_TRUTH_PATH, OPENAI_MODEL,
                           BLOCK_THRESHOLD, SEPARATION_THRESHOLD, MERGE_THRESHOLD,
                           SET_SIZE, SET_DIVERSITY)
from llmcer.data_utils import get_ground_truth
from llmcer.vectorization import cal_total_simi_vector
from llmcer.clustering import lsh_block
from llmcer.pipeline import run_blocks
from llmcer.metrics import (calculate_purity, calculate_inverse_purity,
                            calculate_fp_measure, calculate_ari,
                            calculate_acc, calculate_nmi,
                            calculate_bcubed_metrics)
from llmcer.id_utils import get_id_column


def convert_xlsx_to_csv(xlsx_path):
    csv_path = xlsx_path.replace('.xlsx', '.csv')
    if not os.path.exists(csv_path):
        print(f"Converting {xlsx_path} to {csv_path}...")
        df = pd.read_excel(xlsx_path)
        df.to_csv(csv_path, index=False)
    return csv_path


def main():
    print("Starting LLMCER Pipeline...")

    # 0. Prepare Data
    if DATASET_PATH.endswith('.xlsx'):
        dataset_csv_path = convert_xlsx_to_csv(DATASET_PATH)
    else:
        dataset_csv_path = DATASET_PATH
    print(f"Using dataset: {dataset_csv_path}")

    # Load Ground Truth
    print(f"Loading ground truth from {GROUND_TRUTH_PATH}...")
    try:
        ground_truth = get_ground_truth(GROUND_TRUTH_PATH)
        print(f"Ground truth loaded, {len(ground_truth)} clusters.")
    except Exception as e:
        print(f"Warning: Could not load ground truth: {e}")
        ground_truth = []

    # 1. Vectorization & Similarity Matrix
    print("Calculating vectors and similarity matrix...")
    vectors, simi_matrix, data = cal_total_simi_vector(dataset_csv_path)

    # Fixed, validation-tuned thresholds (see llmcer/config.py).
    print(f"Thresholds: block={BLOCK_THRESHOLD}  separation={SEPARATION_THRESHOLD}  "
          f"merge={MERGE_THRESHOLD}  | S_s={SET_SIZE} S_d={SET_DIVERSITY}")

    # 2. Blocking (LSH) -- hard partition into blocks.
    print("Running LSH Blocking...")
    blocks = lsh_block(vectors, data, BLOCK_THRESHOLD)
    print(f"LSH Blocking done. Found {len(blocks)} blocks.")

    # 3-4. Per-block: NRS -> in-context clustering (+MDG) -> CMR (Algorithm 4).
    print("Running in-context clustering + hierarchical merge per block...")
    t0 = time.time()
    final_result, stats = run_blocks(vectors, simi_matrix, blocks, data,
                                     S_s=SET_SIZE, S_d=SET_DIVERSITY)
    wall = time.time() - t0
    print(f"Done. Final clusters: {len(final_result)}")
    print(f"Stats: API Calls={stats['api_calls']}, LLM time={stats['time']:.2f}s, "
          f"Tokens={stats['tokens']}, MDG interventions={stats['mdg_fails']}, "
          f"merge rounds={stats['rounds']}")

    # 5. Metrics
    print("=" * 40)
    print("FINAL METRICS REPORT")
    print("=" * 40)

    if ground_truth:
        # Augment Ground Truth with singletons for any record not in GT pairs,
        # so every record participates in the evaluation exactly once.
        if hasattr(data, 'iloc'):
            id_col = get_id_column(data)
            all_ids = data[id_col].tolist() if id_col else data.iloc[:, 0].tolist()
        else:
            all_ids = []

        gt_ids = set()
        for cluster in ground_truth:
            for item in cluster:
                gt_ids.add(str(item).strip())

        missing = 0
        for item in all_ids:
            if str(item).strip() not in gt_ids:
                ground_truth.append([item])
                missing += 1
        print(f"Augmented ground truth with {missing} singletons "
              f"(total records: {len(all_ids)}).")

        acc = calculate_acc(ground_truth, final_result)
        nmi = calculate_nmi(ground_truth, final_result)
        purity = calculate_purity(ground_truth, final_result)
        inv_purity = calculate_inverse_purity(ground_truth, final_result)
        f_measure = calculate_fp_measure(ground_truth, final_result)
        ari = calculate_ari(ground_truth, final_result)
        bcubed = calculate_bcubed_metrics(ground_truth, final_result)

        # ACC and FP-measure are the paper's primary metrics (Section 6.1).
        print(f"ACC:            {acc:.4f}")
        print(f"FP-measure:     {f_measure:.4f}")
        print(f"NMI:            {nmi:.4f}")
        print(f"ARI:            {ari:.4f}")
        print("-" * 20)
        print(f"Purity:         {purity:.4f}")
        print(f"Inverse Purity: {inv_purity:.4f}")
        print(f"BCubed F1:      {bcubed['f1']:.4f}  "
              f"(P={bcubed['precision']:.4f} R={bcubed['recall']:.4f})")
    else:
        print("No ground truth provided. Skipping accuracy metrics.")

    print("-" * 40)
    print(f"Total API Calls:      {stats['api_calls']}")
    print(f"Total LLM Time:       {stats['time']:.2f} s  (wall: {wall:.2f} s)")
    print(f"Total Tokens:         {stats['tokens']}")
    print(f"  - Input Tokens:     {stats['in_tokens']}")
    print(f"  - Output Tokens:    {stats['out_tokens']}")
    print(f"Total MDG Interventions: {stats['mdg_fails']}")
    print("=" * 40)

    # Save results
    output_path = "final_results.txt"
    with open(output_path, "w") as f:
        for cluster in final_result:
            f.write(" ".join(map(str, cluster)) + "\n")
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
