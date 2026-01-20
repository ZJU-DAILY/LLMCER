
import re

def preprocess(text):
    text = re.sub(r'(?<=\w)([^\w\s]+)(?=\w)', ' ', text)  
    text = re.sub(r'(?<!\w)[^\w\s]+(?!\w)', '', text) 
    text = re.sub(r'(?<=\w)[^\w\s]+(?!\w)', '', text) 
    text = re.sub(r'(?<!\w)[^\w\s]+(?=\w)', '', text)  
    return text.strip().lower() 

def jaccard_similarity_token(str1, str2):
    if len(str1.strip().split()) == 1 and len(str2.strip().split()) == 1 and ('.' in str1 or '.' in str2):
        try:
            # num1 = float(next(iter(str1))) # Unused
            # num2 = float(next(iter(str2))) # Unused
            match_length = sum(1 for a, b in zip(str1, str2) if a == b)
            max_length = max(len(str1), len(str2))
            return match_length / max_length

        except ValueError:
            pass
    tokens1 = set(preprocess(str1).split())  
    tokens2 = set(preprocess(str2).split())
    intersection = tokens1.intersection(tokens2)
    union = tokens1.union(tokens2)

    if len(tokens1) == 0 or len(tokens2) == 0:
        return -1.0

    return len(intersection) / len(union)

def calsimi(row_1, row_2, rows):
    record1 = rows[row_1]
    record2 = rows[row_2]
    # attributes = set(record1.keys()) - {'id'} # Unused
    # total_weighted_similarity = 0.0 # Unused
    # This function seems incomplete in original code, stopping here as per original
    pass

def get_most_simi(list1, list2, init_simi):
    max_simi = 0
    # record_a = 0 # Unused
    # record_b = 0 # Unused
    for a in list1:
        for b in list2:
            if init_simi[a][b] > max_simi:
                max_simi = init_simi[a][b]
            else:
                continue

    return max_simi

def is_act(a, b, batch_simi):
    for i in range(a+1, b-1):
        if batch_simi[a][i] == 0 and batch_simi[i][b] == 0:
                return 1
    return 0
