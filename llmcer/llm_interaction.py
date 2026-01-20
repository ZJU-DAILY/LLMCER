
import time
import re
import math
import numpy as np
import pandas as pd
import csv
from openai import OpenAI
from llmcer.config import OPENAI_API_KEY, OPENAI_MODEL, PRE_PROMPT_CLASSIFY, PRE_PROMPT_MERGE
from llmcer.data_utils import get_prompt_from_indices
from llmcer.utils import UnionFind, pick_elements
from llmcer.similarity import get_most_simi, is_act
from llmcer.clustering import mdg_check

client = OpenAI(api_key=OPENAI_API_KEY)

def process_sampled_ids(df, sample_ids_list):
    execution_time = 0
    use_number = 0
    total_tokens_call = 0
    
    all_classified_results = []
    for ids in sample_ids_list:
        content_prompt = get_prompt_from_indices(indices=ids, df=df)
        start_time = time.time()
        completion = client.chat.completions.create(
                            model=OPENAI_MODEL,
                            messages=[
                            {"role": "system", "content": "You are a worker with rich experience performing Entity Resolution tasks. You specialize in clustering and classification within ER."},
                            {"role": "user", "content": PRE_PROMPT_CLASSIFY + content_prompt},
                            ]
                        )
        execution_time += (time.time() - start_time)
        use_number += 1
        token_number = completion.usage.total_tokens
        total_tokens_call += token_number
        content = completion.choices[0].message.content
        content = content.replace('\n', '').replace(' ', '')
        content_cleaned = re.sub(r"[^\d\[\],]", "", content)
        content_cleaned = re.sub(r",\s*]", "]", content_cleaned)
        content_cleaned = re.sub(r",+", ",", content_cleaned)
        matches = re.findall(r'\[([^\[\]]*?)\]', content_cleaned)
        result_llm = []
        for match in matches:
            match_cleaned = match.strip()
            if ',' in match_cleaned:
                sublist = [int(num) for num in match_cleaned.split(',')]
                result_llm.append(sublist)
            else:
                if match_cleaned:
                    result_llm.append([int(num) for num in match_cleaned.split()])
        all_classified_results.append(result_llm) 
    return all_classified_results, execution_time, use_number, total_tokens_call

def check_and_handle_missing_ids(one_slice, result_tmp, attempt):
    detected_ids = set()
    for group in result_tmp:
        detected_ids.update(group)
    missing_ids = set(one_slice) - detected_ids
    
    if not missing_ids:
        return result_tmp, True
    else:
        if attempt == 0:
            print(f"Retrying...")
        else:
            for missing_id in missing_ids:
                result_tmp.append([missing_id])
        return result_tmp, False

def find_simi_nex(small_clusters, now_cluster, parent, ini_simi, the_max_nex):
    uf = UnionFind()
    pattern = [] 
    for i in range(len(now_cluster) - 1):
        for j in range(i + 1, len(now_cluster)):
            if ini_simi[now_cluster[i]][now_cluster[j]] >= the_max_nex:
                pattern.append([now_cluster[i], now_cluster[j]])
    
    # Initialize union find with indices of small_clusters
    for i in range(len(small_clusters)):
        uf.add(i)

    for x, y in pattern:
        i1 = -1
        i2 = -1
        for i in range(len(small_clusters)):
            if x in small_clusters[i]:
                i1 = i
                break
        for i in range(len(small_clusters)):
            if y in small_clusters[i]:
                i2 = i
                break
        if i1 != -1 and i2 != -1:
            uf.union(i1, i2)
            
    merged_groups = {}
    for i in range(len(small_clusters)):
        root = uf.find(i)
        if root not in merged_groups:
            merged_groups[root] = []
        merged_groups[root].extend(small_clusters[i])
    result = [sorted(set(values)) for values in merged_groups.values()]
    return result

def llm_seperate(data_list, df, ini_simi, the_max_nex):
    api_call_time = 0
    use_time = 0
    use_token = 0
    seperate_input_token = 0
    seperate_output_token = 0
    mdg_fail_count = 0  # Counter for MDG interventions
    
    result_sliced = []
    number = math.ceil(len(data_list) / 10)
    sliced_lists = [data_list[i * 10:(i + 1) * 10] for i in range(number)]

    for one_slice in sliced_lists:
        api_call_time += 1
        result_tmp = []
        for attempt in range(2):  
            start_time = time.time()
            prompt_sliced = get_prompt_from_indices(one_slice, df)
            completion = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system",
                     "content": "You are a worker specialize in clustering and classification within Entity Resolution."},
                    {"role": "user", "content": PRE_PROMPT_CLASSIFY + prompt_sliced},
                ]
            )
            use_time += time.time() - start_time
            prompt_tokens = completion.usage.prompt_tokens  
            seperate_input_token += prompt_tokens
            completion_tokens = completion.usage.completion_tokens 
            seperate_output_token += completion_tokens
            token_number = completion.usage.total_tokens
            use_token += token_number
            content = completion.choices[0].message.content
            content = content.replace('\n', '').replace(' ', '')
            content_cleaned = re.sub(r"[^\d\[\],]", "", content)
            content_cleaned = re.sub(r",\s*]", "]", content_cleaned)
            content_cleaned = re.sub(r",+", ",", content_cleaned)
            matches = re.findall(r'\[([^\[\]]*?)\]', content_cleaned)
            result_tmp = []
            for match in matches:
                match_cleaned = match.strip()
                if ',' in match_cleaned:
                    sublist = [int(num) for num in match_cleaned.split(',')]
                    result_tmp.append(sublist)
                else:
                    if match_cleaned:
                        result_tmp.append([int(num) for num in match_cleaned.split()])

            result_tmp, complete = check_and_handle_missing_ids(one_slice, result_tmp, attempt)
            
            # MDG Check: Misclustering Detection Guardrail
            # Only run if structure check (missing ids) passed
            if complete:
                is_mdg_acceptable = mdg_check(result_tmp, ini_simi)
                if not is_mdg_acceptable:
                    # If MDG fails, we reject this result and force a retry (if attempts left)
                    # print(f"MDG Check Failed for slice (Attempt {attempt}). Retrying...")
                    mdg_fail_count += 1
                    complete = False
            
            if complete:
                break

        for row_slice in result_tmp:
            result_sliced.append(row_slice)
            
    parent = list(range(len(result_sliced))) # This variable is actually unused in the new logic since we use local UnionFind
    array_new = find_simi_nex(result_sliced, data_list, parent, ini_simi, the_max_nex)
    return array_new, api_call_time, use_time, use_token, seperate_input_token, seperate_output_token, mdg_fail_count

def find_top_cells(similarity_matrix, xiaxian, shagnxian, shuliang):
    triu_indices = np.triu_indices_from(similarity_matrix, k=1)
    triu_values = similarity_matrix[triu_indices]
    valid_indices = np.where((triu_values >= xiaxian) & (triu_values <= shagnxian))[0]
    selected_indices = valid_indices[:shuliang]
    result = [[int(triu_indices[0][i]), int(triu_indices[1][i])] for i in selected_indices]
    return result

def find_merge_cells(similarity_matrix, xiaxian, shagnxian):
    triu_indices = np.triu_indices_from(similarity_matrix, k=1)
    triu_values = similarity_matrix[triu_indices]
    valid_indices = np.where((triu_values >= xiaxian) & (triu_values <= shagnxian))[0]
    result = [[int(triu_indices[0][i]), int(triu_indices[1][i])] for i in valid_indices]
    return result

def merge_2(clusters, simi_matrix, df, block_threshold, merge_threshold):
    def merge_coordinates(coordinates):
        uf = UnionFind()
        ids = set()

        for ltable_id, rtable_id in coordinates:
            uf.add(ltable_id)
            uf.add(rtable_id)
            uf.union(ltable_id, rtable_id)
            ids.add(ltable_id)
            ids.add(rtable_id)

        entity_groups = {}
        for _id in ids:
            root = uf.find(_id)
            if root not in entity_groups:
                entity_groups[root] = []
            entity_groups[root].append(_id)

        result_1 = []
        for root, records in entity_groups.items():
            result_1.append(records)

        return result_1
        
    api_use_time = 0
    merge_time = 0
    merge_token = 0
    merge_input_token = 0
    merge_output_token = 0
    map_merge = [0]*len(clusters)
    row_merge = len(clusters)
    batch_simi = [[0] * row_merge for _ in range(row_merge)]
    for i in range(row_merge):
        for j in range(i, row_merge):
            batch_simi[i][j] = get_most_simi(clusters[i], clusters[j], simi_matrix)
    similarity_matrix = np.array(batch_simi)
    need_merge = [] 
    for threshold in np.arange(merge_threshold+0.02, block_threshold, 0.02):
        print(threshold)
        selected_target = find_top_cells(similarity_matrix, threshold-0.02, threshold, 10)
        if len(selected_target) == 0:
            continue
        count_yes = 0
        for row_selected in selected_target:
            list_all = pick_elements(clusters[row_selected[0]], clusters[row_selected[1]])
            prompt = get_prompt_from_indices(list_all, df)
            start_time = time.time()
            completion = client.chat.completions.create(
                            model=OPENAI_MODEL,
                            messages=[
                            {"role": "system", "content": "You are a worker with rich experience performing Entity Resolution tasks. You specialize in clustering and classification within ER."},
                            {"role": "user", "content": PRE_PROMPT_MERGE + prompt},
                            ]
                        )
            merge_time += time.time() - start_time
            api_use_time = api_use_time + 1
            prompt_tokens = completion.usage.prompt_tokens  
            merge_input_token += prompt_tokens
            completion_tokens = completion.usage.completion_tokens  
            merge_output_token += completion_tokens
            token_number = completion.usage.total_tokens
            merge_token += token_number
            answer = completion.choices[0].message.content.lower().strip() 
            if 'yes' in answer:
                count_yes += 1
        if count_yes/len(selected_target) >= 0.2:
            need_merge += find_merge_cells(similarity_matrix, threshold-0.02, threshold)
    for row_in_need_merge in need_merge:
        map_merge[row_in_need_merge[0]] = 1
        map_merge[row_in_need_merge[1]] = 1 
    map_rest = []
    for i in range(len(clusters)):
        if map_merge[i] == 0:
            map_rest.append(clusters[i]) 
    new_result = []
    result_1 = merge_coordinates(need_merge)
    for row in result_1:
        tmp = []
        for ids in row:
            tmp.extend(clusters[ids])
        new_result.append(tmp)
    map_rest += new_result
    print("merge_done")
    return map_rest, api_use_time, merge_time, merge_token, merge_input_token, merge_output_token

def merge(clusters, simi_matrix, df):
    map_merge = [0]*len(clusters)
    row_merge = len(clusters)
    batch_simi = [[0] * row_merge for _ in range(row_merge)]
    for i in range(row_merge):
        for j in range(i, row_merge):
            batch_simi[i][j] = get_most_simi(clusters[i], clusters[j], simi_matrix)
    min_simi = 0.5891500074006119 
    new = []
    execution_time = 0
    use_number = 0
    total_tokens_call = 0

    while True:
        the_max_simi_batch = 0
        the_first_list = 0
        the_second_list = 0
        for i in range(row_merge):
            for j in range(i, row_merge):
 
                if i == j:
                    continue
                elif batch_simi[i][j] > the_max_simi_batch:
                    the_max_simi_batch = batch_simi[i][j]
                    the_first_list = i
                    the_second_list = j
        if the_max_simi_batch < min_simi:
            break
        if batch_simi[the_first_list][the_second_list] > 0.6:
            batch_simi[the_first_list][the_second_list] = min_simi - 0.01
            continue
        is_ok = is_act(the_first_list, the_second_list, batch_simi)
        print(is_ok)
        if is_ok == 1:
            batch_simi[the_first_list][the_second_list] = 0
        else:
            print(batch_simi[the_first_list][the_second_list])
            list1 = clusters[the_first_list]
            list2 = clusters[the_second_list]
            list_all = pick_elements(list1, list2)
            prompt = get_prompt_from_indices(list_all, df)
            start_time = time.time()
            completion = client.chat.completions.create(
                            model=OPENAI_MODEL,
                            messages=[
                            {"role": "system", "content": "You are a worker with rich experience performing Entity Resolution tasks. You specialize in clustering and classification within ER."},
                            {"role": "user", "content": PRE_PROMPT_MERGE + prompt},
                            ]
                        )
            execution_time += (time.time() - start_time)
            use_number = use_number + 1
            token_number = completion.usage.total_tokens
            total_tokens_call += token_number
            answer = completion.choices[0].message.content.lower().strip() 
            if 'yes' in answer:
                print("yes")
                new.append([the_first_list, the_second_list])
                map_merge[the_first_list] = 1
                map_merge[the_second_list] = 1
                batch_simi[the_first_list][the_second_list] = 0
            else:
                print("no")
                batch_simi[the_first_list][the_second_list] = min_simi - 0.01
                continue
    map_rest = []
    for i in range(len(map_merge)):
        if map_merge[i] == 0:
            map_rest.append(clusters[i])
    
    # NOTE: Writing to 'nex_step.csv' was in original code. 
    # We should probably avoid side effects if possible, but I will keep it for now or just use memory.
    # The original code writes to csv and then reads it back to use UnionFind. 
    # I can just use UnionFind directly.
    
    uf = UnionFind()
    ids = set()
    for row in new:
        uf.union(row[0], row[1])
        ids.add(row[0])
        ids.add(row[1])
        
    entity_groups = {}
    for _id in ids:
        root = uf.find(_id)
        if root not in entity_groups:
            entity_groups[root] = []
        entity_groups[root].append(_id)
    result_1 = []
    for root, records in entity_groups.items():
        result_1.append(records) 
    result_all_final = result_1 + map_rest 
    print("merge done")
    return result_all_final, use_number, execution_time, total_tokens_call
