
import csv
import pandas as pd
from typing import List, Dict
from llmcer.utils import UnionFind
from llmcer.id_utils import get_id_column

def get_prompt_from_indices(indices, df):
    """
    Efficiently generates prompt string from dataframe indices.
    """
    lines = []
    
    id_col = get_id_column(df)
    
    columns = [col for col in df.columns if col != id_col and col != 'combined_text']
    
    for idx in indices:
        try:
            row = df.iloc[idx]
            r_id = idx

            parts = [str(row[col]) for col in columns]
            rec_str = f"Record {r_id}: " + ",".join(parts)
            lines.append(rec_str)
        except IndexError:
            continue

    return '\n'.join(lines)

def get_data_from_indices(indices, df):
    """
    Get text data for vectorization or other uses from dataframe indices.
    Matches the comma-joined format of original get_data.
    """
    lines = []
    id_col = get_id_column(df)
    columns = [col for col in df.columns if col != id_col and col != 'combined_text']
    
    for idx in indices:
        try:
            row = df.iloc[idx]
            parts = [str(row[col]) for col in columns]
            lines.append(",".join(parts))
        except IndexError:
            continue
    return '\n'.join(lines)

def get_ground_truth(file_path):
    def merge_coordinates(coordinates):
        uf = UnionFind()
        ids = set()

        for ltable_id, rtable_id in coordinates:
            try:
                ltable_id = int(ltable_id)
                rtable_id = int(rtable_id)
            except ValueError:
                continue

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

    if file_path.endswith('.txt'):
        clusters = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    try:
                        cluster = [int(p) for p in parts]
                        clusters.append(cluster)
                    except ValueError:
                        continue
        return clusters

    elif file_path.endswith('.xlsx'):
        df = pd.read_excel(file_path)
        if df.shape[1] >= 2:
            pairs = df.iloc[:, :2].values.tolist()
            return merge_coordinates(pairs)
        else:
            return []

    elif file_path.endswith('.csv'):
        rows = []
        for enc in ('utf-8', 'MacRoman', 'latin-1'):
            try:
                with open(file_path, newline='', encoding=enc) as csvfile:
                    rows = list(csv.reader(csvfile, delimiter=','))
                break
            except UnicodeDecodeError:
                continue
        if not rows:
            return []

        def _is_int_pair(row):
            if len(row) < 2:
                return False
            try:
                int(row[0]); int(row[1])
                return True
            except (ValueError, TypeError):
                return False

        start = 0 if _is_int_pair(rows[0]) else 1
        data = [r[:2] for r in rows[start:] if _is_int_pair(r)]
        return merge_coordinates(data)

    else:
        print(f"Warning: Unsupported ground truth file format: {file_path}")
        return []

def get_data(id_list, file_path):
    lines = []
    with open(file_path, 'r', encoding='MacRoman') as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        for r_id in id_list:
            for row in rows:
                if row['ID'] == str(r_id): 
                    lines.append(','.join([str(row[key]) for key in reader.fieldnames if key != 'ID']))
                    break
    prompts = '\n'.join(lines)
    return prompts

def read_csv_to_2d_array(file_path):
    with open(file_path, 'r', encoding='MacRoman') as file:
        reader = csv.reader(file)
        data = list(reader)
    return data

def get_prompt_from_ids(id_list, file_path):
    lines = []
    with open(file_path, 'r', encoding='MacRoman') as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        for r_id in id_list:
            for row in rows:
                if row['ID'] == str(r_id): 
                    rec_str = f"Record {r_id}: "
                    rec_str += ','.join([str(row[key]) for key in reader.fieldnames if key != 'ID'])
                    lines.append(rec_str)
                    break
    return '\n'.join(lines)

def read_2d_array_from_file(file_path):
    array_list = []
    try:
        with open(file_path, 'r') as file:
            for line in file:
                row = list(map(int, line.strip().split()))
                array_list.append(row)
        return array_list
    except FileNotFoundError:
        print(f"{file_path} is not found!")
    except ValueError as e:
        print(f"Exist wrong number: {e}")
    return []

def read_clusters_from_csv(filename):
    clusters = []
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            clusters.append([int(item) for item in row if item])  
    return clusters
