
import numpy as np
import math
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor
from llmcer.clustering import elbow_method, kmeans_clustering
from llmcer.llm_interaction import process_sampled_ids, llm_seperate
from llmcer.config import MAX_K

def dynamic_sampling(original_array):
    result = []
    row_indices = {i: row.copy() for i, row in enumerate(original_array)}
    total_ids = sum(len(row) for row in original_array)
    
    while True:
        group = []
        while len(group) < 10:
            flag = False
            for i in range(len(original_array)):
                if row_indices[i]: 
                    group.append(row_indices[i].pop(0))
                    flag = True
                if len(group) == 10:  
                    break
            if not flag: 
                break
        if group:
            result.append(group)
        if not any(row_indices.values()):
            break
    
    output_ids = [id for group in result for id in group]
    if len(output_ids) != total_ids:
        raise ValueError("wrong number!")
    
    return result

def format_output(id_list, labels):
    clusters = {}
    for i, label in enumerate(labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(id_list[i])
    
    sorted_clusters = sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True)
    
    output = []
    for cluster in sorted_clusters:
        output.append([int(item) for item in cluster[1]])
    
    return output

def the_most_importent_one(vector_data, classified_results):
    result_for_find = []
    for classified_results_row in classified_results:
        list_select = []
        vectored_select = []
        for cluster_row in classified_results_row:
            vectored_select = np.array([vector_data[id_] for id_ in cluster_row])
            avg_vector = np.mean(vectored_select, axis=0) 
            distances = [np.linalg.norm(vectored_select[i] - avg_vector) for i in range(len(cluster_row))]
            representative_id = cluster_row[np.argmin(distances)]
            list_select.append(representative_id)
        result_for_find.append(list_select)
    return result_for_find

def the_most_importent_one_1(classified_results):
    result_for_find = []
    for classified_results_row in classified_results:
        list_select = []
        for cluster_row in classified_results_row:   
            representative_id = cluster_row[0]
            list_select.append(representative_id)
        result_for_find.append(list_select)
    return result_for_find

def find_most_similar(
    current_id: Optional[int], 
    candidate_ids: List[int], 
    similarity_matrix: List[List[float]]
) -> Optional[int]:

    if not candidate_ids:
        return None
    
    if current_id is None:
        return candidate_ids[0]

    max_similarity = -float('inf')
    most_similar_id = None
    
    for candidate_id in candidate_ids:
        similarity = similarity_matrix[current_id][candidate_id]
        if similarity > max_similarity:
            max_similarity = similarity
            most_similar_id = candidate_id
    
    return most_similar_id

def filter_available_ids(next_row_ids: List[int], id_assigned: set) -> List[int]:
    return [id_ for id_ in next_row_ids if id_ not in id_assigned]

def process_rounds(
    id_matrix: List[List[int]],
    similarity_matrix: List[List[float]],
    max_length: int
) -> List[List[int]]:
    all_rounds = []
    id_assigned = set()
    total_rows = len(id_matrix)

    while True:
        current_round = []
        current_id = None

        for row_index in range(total_rows):
            available_ids = filter_available_ids(id_matrix[row_index], id_assigned)
            if not available_ids:
                continue

            next_id = find_most_similar(current_id, available_ids, similarity_matrix)
            if next_id is not None:
                current_round.append(next_id)
                id_assigned.add(next_id)
                current_id = next_id
        if not current_round:
            break
        
        all_rounds.append(current_round)
    
    return all_rounds

def traverse_ids_to_2d(
    id_matrix: List[List[int]], 
    similarity_matrix: List[List[float]],
    max_length: int = 10, 
    batch_size: int = 10
) -> List[List[int]]:
    all_rounds = process_rounds(id_matrix, similarity_matrix, max_length)
    return all_rounds

def find_back(two_d_array, three_d_array):
    num_to_row = {}
    for matrix in three_d_array:
        for row in matrix:
            for number in row:
                if number not in num_to_row:
                    num_to_row[number] = row
    for i, row in enumerate(two_d_array):
        new_row = []
        for number in row:
            if number in num_to_row:
                new_row.extend(num_to_row[number])
        two_d_array[i] = list(dict.fromkeys(new_row))
    return two_d_array

def process_id_list(id_list, simi_matrix, data_file_path, the_threhold):
    llm_tmp = []
    api_call_time_all = 0
    sperate_time = 0
    sperate_token = 0
    seperate_token_input = 0
    seperate_token_output = 0
    
    # NOTE: The original code for process_id_list in seperate_4 was a bit different from seperate_2.
    # seperate_4 calls llm_seperate directly with id_list? 
    # Wait, in seperate_4:
    # futures = [executor.submit(process_id_list, id_list,  simi_matrix, data_file_path,the_threhold) ...]
    # And process_id_list calls llm_seperate.
    # But llm_seperate expects data_list (list of ids).
    # In seperate_4, it seems it skips the initial clustering step that was in seperate and seperate_2?
    # Let's check seperate function again.
    # seperate does: get_data -> vectorize -> kmeans -> dynamic_sampling -> process_sampled_ids -> the_most_importent_one -> traverse -> llm_seperate.
    # seperate_4 does: calls process_id_list which calls llm_seperate directly.
    # This implies seperate_4 expects id_list to be already prepared or something?
    # Or maybe I should implement the full logic inside process_id_list like in seperate_2 but return metrics like seperate_4.
    
    # I will implement the full logic as in 'seperate' function but modularized.
    # But wait, seperate_4 was:
    # def process_id_list(id_list, simi_matrix, data_file_path,the_threhold):
    #    array_new... = llm_seperate(id_list, data_file_path, simi_matrix, the_threhold)
    
    # It seems seperate_4 is just a parallel wrapper around llm_seperate?
    # But llm_seperate takes a list of IDs and splits them into chunks and calls LLM.
    
    # However, 'seperate' function has a lot more steps before calling llm_seperate.
    # It seems 'seperate' is the robust one. 'seperate_2' also has the full logic.
    # 'seperate_4' seems to be missing the vectorization/kmeans steps? Or maybe 'id_list' passed to it are already processed?
    
    # Let's look at 'seperate_2'.
    # def process_id_list(id_list, vectors, simi_matrix, data_file_path):
    #    text_data = get_data...
    #    vectorize...
    #    kmeans...
    #    process_sampled_ids...
    #    the_most_importent_one...
    #    traverse...
    #    loop over target_list -> llm_seperate...
    #    find_back...
    
    # This looks correct. I should implement this full logic.
    
    # But I need 'vectors' argument which is missing in seperate_4 signature.
    # I will stick to the logic of 'seperate_2' but with metrics from 'seperate_4'.
    
    # Wait, I need to know where 'vectors' comes from. It's passed from main.
    
    pass

def full_process_id_list(id_list, vectors, simi_matrix, df, the_threshold):
    # 1. Get vectors from global vectors (efficient slicing)
    # id_list contains indices into the global vectors/data
    vectorized_data = np.array([vectors[i] for i in id_list])
    
    # 2. Clustering
    n_clusters = elbow_method(vectorized_data) 
    labels = kmeans_clustering(vectorized_data, n_clusters)
    clusters_labels = format_output(id_list, labels)
    
    # 3. Sampling
    prompt_id = dynamic_sampling(clusters_labels)
    
    # 4. LLM Classification (First Pass)
    classified_results, execute_time, use_number, total_tokens = process_sampled_ids(df, prompt_id)
    
    # 5. Representative Selection
    # Note: 'the_most_importent_one' uses global vectors indexed by id_.
    # Since we pass global vectors, this works fine.
    result_for_found = the_most_importent_one(vectors, classified_results)
    
    # 6. Traversal
    target_list = traverse_ids_to_2d(result_for_found, simi_matrix, max_length=10, batch_size=10)
    
    # 7. LLM Separation (Second Pass)
    llm_tmp = []
    api_call_time_all = use_number
    sperate_time = execute_time
    sperate_token = total_tokens
    seperate_input_token = 0 # Need to track
    seperate_output_token = 0 # Need to track
    mdg_fail_total = 0  # Track total MDG interventions
    
    for row_slice in target_list:
        array_new, api_call, use_t, use_tok, in_tok, out_tok, mdg_count = llm_seperate(row_slice, df, simi_matrix, the_threshold)
        api_call_time_all += api_call
        sperate_time += use_t
        sperate_token += use_tok
        seperate_input_token += in_tok
        seperate_output_token += out_tok
        mdg_fail_total += mdg_count
        llm_tmp.extend(array_new)
        
    # 8. Find Back
    find_back_matrix = find_back(llm_tmp, classified_results)
    
    return find_back_matrix, api_call_time_all, sperate_time, sperate_token, seperate_input_token, seperate_output_token, mdg_fail_total

def seperate_parallel(vectors, simi_matrix, merge_clusters_pre, df, the_threshold):
    sperate_result = []
    api_call_time_all = 0
    sperate_time = 0
    sperate_token = 0
    seperate_input_token = 0
    seperate_output_token = 0
    total_mdg_fails = 0
    
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(full_process_id_list, id_list, vectors, simi_matrix, df, the_threshold) for id_list in merge_clusters_pre]

        for future in futures:
            result, api_call, time_each, token_each, in_tok, out_tok, mdg_cnt = future.result()
            sperate_result.extend(result)
            api_call_time_all += api_call
            sperate_time += time_each
            sperate_token += token_each
            seperate_input_token += in_tok
            seperate_output_token += out_tok
            total_mdg_fails += mdg_cnt

    print("separate done")
    return sperate_result, api_call_time_all, sperate_time, sperate_token, seperate_input_token, seperate_output_token, total_mdg_fails
