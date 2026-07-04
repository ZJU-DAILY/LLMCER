
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from llmcer.config import EMBEDDING_MODEL
from llmcer.id_utils import get_id_column

def vectorize_data(text):
    model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = model.encode(text.split('\n'))  
    return embeddings

def cal_total_simi_vector(data_file_path, model_file=EMBEDDING_MODEL):
    model = SentenceTransformer(model_file)
    
    try:
        data = pd.read_csv(data_file_path, encoding="utf-8")
    except UnicodeDecodeError:
        print("Warning: UTF-8 decode failed, trying MacRoman...")
        data = pd.read_csv(data_file_path, encoding="MacRoman")

    id_col = get_id_column(data)

    def combine_attributes(row):
        if id_col and id_col in row:
             return ' '.join(str(value) for key, value in row.items() if key != id_col)
        return ' '.join(str(value) for value in row)
        
    data['combined_text'] = data.apply(combine_attributes, axis=1)
    vectors = data['combined_text'].apply(lambda text: model.encode(text)).tolist() 
    simi_matrix = cosine_similarity(vectors)
    print("calculate similarity matrix done")
    return vectors, simi_matrix, data
