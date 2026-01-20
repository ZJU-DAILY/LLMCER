
import sys
import os
import time
import pandas as pd
import numpy as np

# Add parent directory to path to import llmcer
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llmcer.config import DATASET_PATH, GROUND_TRUTH_PATH, OPENAI_MODEL
from llmcer.data_utils import get_ground_truth
from llmcer.vectorization import cal_total_simi_vector
from llmcer.clustering import lsh_block
from llmcer.pipeline import seperate_parallel
from llmcer.llm_interaction import merge_2
from llmcer.metrics import calculate_purity, calculate_inverse_purity, calculate_fp_measure, calculate_ari
from llmcer.id_utils import get_id_column

def convert_xlsx_to_csv(xlsx_path):
    csv_path = xlsx_path.replace('.xlsx', '.csv')
    if not os.path.exists(csv_path):
        print(f"Converting {xlsx_path} to {csv_path}...")
        df = pd.read_excel(xlsx_path)
        
        # Check for ID column using case-insensitive check
        id_col = get_id_column(df)
        
        if id_col:
            # Check if IDs are integers (optional verification)
            try:
                first_id = int(df[id_col].iloc[0])
            except:
                pass
        else:
            print("Warning: No ID column found (case-insensitive search for 'id').")
            
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
    
    # 2. Blocking (LSH)
    print("Running LSH Blocking...")
    lsh_threshold = 0.5 # Default from intuition, check notebook for better value
    # Notebook cell 998 passes 'similarity_threshold'.
    # I'll stick with 0.5
    merge_clusters_pre = lsh_block(vectors, data, lsh_threshold)
    print(f"LSH Blocking done. Found {len(merge_clusters_pre)} blocks.")
    
    # 3. Separation
    print("Running Separation (Cluster Splitting)...")
    separation_threshold = 0.5 # Need to define this. 'the_max_nex' in notebook.
    # In notebook cell 765, `seperate` is called with `the_max_nex`.
    # In cell 915 `seperate_4` calls `llm_seperate` with `the_threhold`.
    # Value is not clear, but likely around 0.5.
    
    result_sep, api_calls, sep_time, sep_tokens, in_tokens, out_tokens, mdg_fails = seperate_parallel(
        vectors, simi_matrix, merge_clusters_pre, data, separation_threshold
    )
    print(f"Separation done. Resulting clusters: {len(result_sep)}")
    print(f"Stats: API Calls={api_calls}, Time={sep_time:.2f}s, Tokens={sep_tokens}")
    print(f"MDG Interventions: {mdg_fails}")
    
    # 4. Merging
    print("Running Merging...")
    block_threshold = 0.8 # From notebook usage of merge_2?
    merge_threshold = 0.6 # From notebook usage?
    # Cell 1107: merge_2(..., block_threshold , merge_threshold)
    # Cell 1168: np.arange(merge_threshold+0.02 , block_threshold, 0.02)
    # I'll use 0.8 and 0.6 as placeholders.
    
    final_result, merge_api_calls, merge_time, merge_tokens, m_in_tok, m_out_tok = merge_2(
        result_sep, simi_matrix, data, block_threshold, merge_threshold
    )
    print(f"Merging done. Final clusters: {len(final_result)}")
    print(f"Stats: API Calls={merge_api_calls}, Time={merge_time:.2f}s, Tokens={merge_tokens}")

    # 5. Metrics
    print("="*40)
    print("FINAL METRICS REPORT")
    print("="*40)
    
    if ground_truth:
        # Metrics functions expect list of lists.
        purity = calculate_purity(ground_truth, final_result)
        inv_purity = calculate_inverse_purity(ground_truth, final_result)
        f_measure = calculate_fp_measure(ground_truth, final_result)
        ari = calculate_ari(ground_truth, final_result)
        
        print(f"Purity:         {purity:.4f}")
        print(f"Inverse Purity: {inv_purity:.4f}")
        print(f"F-Measure:      {f_measure:.4f}")
        print(f"ARI:            {ari:.4f}")
    else:
        print("No ground truth provided. Skipping accuracy metrics.")

    # Total Stats
    total_api_calls = api_calls + merge_api_calls
    total_time = sep_time + merge_time
    total_tokens = sep_tokens + merge_tokens
    total_in_tokens = in_tokens + m_in_tok
    total_out_tokens = out_tokens + m_out_tok
    
    print("-" * 40)
    print(f"Total API Calls:     {total_api_calls}")
    print(f"Total Execution Time: {total_time:.2f} s")
    print(f"Total Tokens:        {total_tokens}")
    print(f"  - Input Tokens:    {total_in_tokens}")
    print(f"  - Output Tokens:   {total_out_tokens}")
    print(f"Total MDG Interventions: {mdg_fails}")
    print("="*40)
    
    # Save results
    output_path = "final_results.txt"
    with open(output_path, "w") as f:
        for cluster in final_result:
            f.write(" ".join(map(str, cluster)) + "\n")
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    main()
