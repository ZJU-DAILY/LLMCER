
from collections import Counter
import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from scipy.optimize import linear_sum_assignment

def normalize_id(item):
    """Normalize item ID to string for consistent comparison."""
    return str(item).strip()

def _aligned_labels(true_clusters, predicted_clusters):
    """
    Build integer label vectors (true_labels, predicted_labels) over the union
    of all items appearing in either clustering. Items missing from one side are
    given a unique singleton label there. Returns (true_labels, pred_labels) as
    numpy arrays aligned by a shared item ordering.
    """
    true_clusters = [[normalize_id(x) for x in c] for c in true_clusters]
    predicted_clusters = [[normalize_id(x) for x in c] for c in predicted_clusters]

    true_map = {}
    for i, c in enumerate(true_clusters):
        for item in c:
            true_map[item] = i
    pred_map = {}
    for i, c in enumerate(predicted_clusters):
        for item in c:
            pred_map[item] = i

    items = sorted(set(true_map) | set(pred_map))
    if not items:
        return np.array([], dtype=int), np.array([], dtype=int)

    # Fresh singleton labels for items missing on a side so they never spuriously match.
    next_t = len(true_clusters)
    next_p = len(predicted_clusters)
    t_labels, p_labels = [], []
    for it in items:
        if it in true_map:
            t_labels.append(true_map[it])
        else:
            t_labels.append(next_t); next_t += 1
        if it in pred_map:
            p_labels.append(pred_map[it])
        else:
            p_labels.append(next_p); next_p += 1
    return np.array(t_labels, dtype=int), np.array(p_labels, dtype=int)

def calculate_acc(true_clusters, predicted_clusters):
    """
    Clustering Accuracy (ACC) per paper Eq. (2)-(3).

    ACC = CorrectCount / |R|, where the predicted clusters are optimally
    matched (1-to-1) to ground-truth clusters by maximising the total
    intersection ("reordering Y based on their intersection sizes with the
    predicted clusters", Eq. 3). We solve the assignment with the Hungarian
    algorithm on the negated contingency matrix.
    """
    t_labels, p_labels = _aligned_labels(true_clusters, predicted_clusters)
    n = len(t_labels)
    if n == 0:
        return 0.0

    true_ids = sorted(set(t_labels.tolist()))
    pred_ids = sorted(set(p_labels.tolist()))
    t_index = {v: i for i, v in enumerate(true_ids)}
    p_index = {v: i for i, v in enumerate(pred_ids)}

    # contingency[pred, true] = #items in that predicted & true cluster
    contingency = np.zeros((len(pred_ids), len(true_ids)), dtype=np.int64)
    for tl, pl in zip(t_labels, p_labels):
        contingency[p_index[pl], t_index[tl]] += 1

    # Maximise overlap -> minimise negative overlap.
    row_ind, col_ind = linear_sum_assignment(-contingency)
    correct = contingency[row_ind, col_ind].sum()
    return float(correct) / n

def calculate_nmi(true_clusters, predicted_clusters):
    """Normalized Mutual Information between the two clusterings."""
    t_labels, p_labels = _aligned_labels(true_clusters, predicted_clusters)
    if len(t_labels) == 0:
        return 0.0
    return float(normalized_mutual_info_score(t_labels, p_labels))

def calculate_purity(true_clusters, predicted_clusters):
    # Normalize clusters
    true_clusters_norm = [[normalize_id(x) for x in c] for c in true_clusters]
    predicted_clusters_norm = [[normalize_id(x) for x in c] for c in predicted_clusters]
    
    total_samples = sum(len(cluster) for cluster in predicted_clusters_norm)
    total_correct = 0

    # Create map for faster lookup
    true_map = {}
    for i, cluster in enumerate(true_clusters_norm):
        for item in cluster:
            true_map[item] = i

    for pred_cluster in predicted_clusters_norm:
        label_count = Counter()
        for sample in pred_cluster:
            if sample in true_map:
                true_label = true_map[sample]
                label_count[true_label] += 1
            else:
                # Treat items not in GT as their own unique label (or ignore?)
                # If we assume GT is complete, items not in GT are errors or singletons.
                # User says "singletons are correct". 
                # If we treat unknown item as a unique label 'unknown_X':
                label_count[f"unknown_{sample}"] += 1
                
        if label_count:
            max_label_count = max(label_count.values())
            total_correct += max_label_count

    return total_correct / total_samples if total_samples > 0 else 0

def calculate_bcubed_metrics(true_clusters, predicted_clusters):
    """
    Calculates BCubed Precision, Recall, and F1.
    """
    # Normalize
    true_clusters = [[normalize_id(x) for x in c] for c in true_clusters]
    predicted_clusters = [[normalize_id(x) for x in c] for c in predicted_clusters]

    # Map item -> cluster index
    true_map = {}
    true_sets = [set(c) for c in true_clusters]
    for i, cluster in enumerate(true_clusters):
        for item in cluster:
            true_map[item] = i
            
    pred_map = {}
    pred_sets = [set(c) for c in predicted_clusters]
    for i, cluster in enumerate(predicted_clusters):
        for item in cluster:
            pred_map[item] = i
            
    all_items = set(true_map.keys()) | set(pred_map.keys())
    n = len(all_items)
    if n == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        
    p_sum = 0
    r_sum = 0
    
    for item in all_items:
        true_idx = true_map.get(item)
        pred_idx = pred_map.get(item)
        
        true_set = true_sets[true_idx] if true_idx is not None else {item} # Treat missing in GT as singleton
        pred_set = pred_sets[pred_idx] if pred_idx is not None else {item} # Treat missing in Pred as singleton
        
        # Intersection of the cluster containing the item in Truth and Prediction
        intersection = len(true_set & pred_set)
        
        # BCubed Precision: P(item) = |True(i) n Pred(i)| / |Pred(i)|
        if len(pred_set) > 0:
            p_sum += intersection / len(pred_set)
        
        # BCubed Recall: R(item) = |True(i) n Pred(i)| / |True(i)|
        if len(true_set) > 0:
            r_sum += intersection / len(true_set)
            
    precision = p_sum / n
    recall = r_sum / n
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {"precision": precision, "recall": recall, "f1": f1}

def calculate_inverse_purity(true_clusters, predicted_clusters):
    # Normalize
    true_clusters_norm = [[normalize_id(x) for x in c] for c in true_clusters]
    predicted_clusters_norm = [[normalize_id(x) for x in c] for c in predicted_clusters]
    
    total_samples = sum(len(cluster) for cluster in true_clusters_norm)
    total_correct = 0
    
    # Map pred items for speed
    pred_map = {}
    for i, cluster in enumerate(predicted_clusters_norm):
        for item in cluster:
            pred_map[item] = i

    for true_cluster in true_clusters_norm:
        if true_cluster:
            pred_labels = Counter()
            for sample in true_cluster:
                if sample in pred_map:
                    pred_labels[pred_map[sample]] += 1
                else:
                    pred_labels[f"unknown_{sample}"] += 1
                    
            if pred_labels:
                max_match = max(pred_labels.values())
                total_correct += max_match

    return total_correct / total_samples if total_samples > 0 else 0

def calculate_fp_measure(true_clusters, predicted_clusters, beta=1.0):
    purity = calculate_purity(true_clusters, predicted_clusters)
    inverse_purity = calculate_inverse_purity(true_clusters, predicted_clusters)

    if purity + inverse_purity == 0:
        return 0

    # F-Beta Measure
    # (1 + beta^2) * (P * IP) / (beta^2 * P + IP)
    beta_sq = beta ** 2
    return (1 + beta_sq) * (purity * inverse_purity) / ((beta_sq * purity) + inverse_purity)

def calculate_macro_purity(true_clusters, predicted_clusters):
    """
    Calculates Macro-Average Purity (Average Purity per Predicted Cluster).
    This gives equal weight to each cluster, regardless of size.
    Useful if user cares about 'percentage of correct clusters'.
    """
    # Normalize clusters
    true_clusters_norm = [[normalize_id(x) for x in c] for c in true_clusters]
    predicted_clusters_norm = [[normalize_id(x) for x in c] for c in predicted_clusters]
    
    # Create map for faster lookup
    true_map = {}
    for i, cluster in enumerate(true_clusters_norm):
        for item in cluster:
            true_map[item] = i
            
    total_purity = 0
    n_clusters = len(predicted_clusters_norm)
    
    if n_clusters == 0:
        return 0.0

    for pred_cluster in predicted_clusters_norm:
        if not pred_cluster:
            continue
            
        label_count = Counter()
        for sample in pred_cluster:
            if sample in true_map:
                true_label = true_map[sample]
                label_count[true_label] += 1
            else:
                label_count[f"unknown_{sample}"] += 1
                
        if label_count:
            max_label_count = max(label_count.values())
            cluster_purity = max_label_count / len(pred_cluster)
            total_purity += cluster_purity
            
    return total_purity / n_clusters

def calculate_pure_cluster_ratio(true_clusters, predicted_clusters):
    """
    Calculates the ratio of perfectly pure predicted clusters.
    """
    true_clusters_norm = [[normalize_id(x) for x in c] for c in true_clusters]
    predicted_clusters_norm = [[normalize_id(x) for x in c] for c in predicted_clusters]
    
    true_map = {}
    for i, cluster in enumerate(true_clusters_norm):
        for item in cluster:
            true_map[item] = i
            
    pure_clusters = 0
    n_clusters = len(predicted_clusters_norm)
    
    if n_clusters == 0:
        return 0.0

    for pred_cluster in predicted_clusters_norm:
        if not pred_cluster:
            continue
            
        label_count = Counter()
        for sample in pred_cluster:
            if sample in true_map:
                true_label = true_map[sample]
                label_count[true_label] += 1
            else:
                label_count[f"unknown_{sample}"] += 1
                
        if label_count:
            # Check if all items belong to the same label
            if len(label_count) == 1:
                pure_clusters += 1
                
    return pure_clusters / n_clusters


def convert_to_labels(clusters, n_samples):

    labels = [-1] * n_samples 
    for cluster_id, cluster in enumerate(clusters):
        for sample in cluster:
            labels[sample] = cluster_id
    return labels

def calculate_ari(true_clusters, predicted_clusters):
    # Normalize
    true_clusters_norm = [[normalize_id(x) for x in c] for c in true_clusters]
    predicted_clusters_norm = [[normalize_id(x) for x in c] for c in predicted_clusters]

    all_samples = set(sample for cluster in true_clusters_norm for sample in cluster) | \
                  set(sample for cluster in predicted_clusters_norm for sample in cluster)
    
    # Map to continuous integers
    sample_to_int = {sample: i for i, sample in enumerate(all_samples)}
    n_samples = len(all_samples)
    
    true_labels = [-1] * n_samples
    for cid, cluster in enumerate(true_clusters_norm):
        for sample in cluster:
            if sample in sample_to_int:
                true_labels[sample_to_int[sample]] = cid
                
    predicted_labels = [-1] * n_samples
    for cid, cluster in enumerate(predicted_clusters_norm):
        for sample in cluster:
            if sample in sample_to_int:
                predicted_labels[sample_to_int[sample]] = cid
                
    ari = adjusted_rand_score(true_labels, predicted_labels)
    return ari

def calculate_pairwise_metrics(true_clusters, predicted_clusters):
    """
    Calculates pairwise metrics: Accuracy, Precision, Recall, F1, FP, FN, TP, TN.
    """
    # Normalize
    true_clusters_norm = [[normalize_id(x) for x in c] for c in true_clusters]
    predicted_clusters_norm = [[normalize_id(x) for x in c] for c in predicted_clusters]

    all_samples = set(sample for cluster in true_clusters_norm for sample in cluster) | \
                  set(sample for cluster in predicted_clusters_norm for sample in cluster)
    
    if not all_samples:
        return {}

    sorted_samples = sorted(list(all_samples))
    n = len(sorted_samples)
    
    # Create cluster maps
    true_map = {}
    for cid, cluster in enumerate(true_clusters_norm):
        for sample in cluster:
            true_map[sample] = cid
            
    pred_map = {}
    for cid, cluster in enumerate(predicted_clusters_norm):
        for sample in cluster:
            pred_map[sample] = cid
            
    # Calculate pairs
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    
    # Using simple iteration is slow for large N (O(N^2)). 
    # For N=1000, N^2/2 = 500,000, feasible.
    for i in range(n):
        for j in range(i + 1, n):
            u = sorted_samples[i]
            v = sorted_samples[j]
            
            # If item not in GT, assume it's a singleton (unique label)
            # true_map.get(u, -1) == true_map.get(v, -2) ensures that if both missing, they are NOT same.
            # But wait, if u and v are both missing from GT, should they be considered "same"?
            # If we assume missing = singleton, then they are same ONLY if u == v (which is impossible in loop)
            # So missing items are never in the same cluster with anything else in GT.
            
            # However, if u is in GT and v is NOT, they are definitely not in same cluster.
            
            # Unique identifiers for missing items
            true_u = true_map.get(u, f"missing_{u}")
            true_v = true_map.get(v, f"missing_{v}")
            
            pred_u = pred_map.get(u, f"missing_pred_{u}")
            pred_v = pred_map.get(v, f"missing_pred_{v}")
            
            same_true = (true_u == true_v)
            same_pred = (pred_u == pred_v)
            
            if same_true and same_pred:
                tp += 1
            elif not same_true and same_pred:
                fp += 1
            elif same_true and not same_pred:
                fn += 1
            else:
                tn += 1
                
    total_pairs = tp + fp + fn + tn
    accuracy = (tp + tn) / total_pairs if total_pairs > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn
    }

def calculate_tolerant_purity(true_clusters, predicted_clusters, tolerance=1):
    """
    Calculates Purity but considers a cluster 'pure' if the number of errors 
    (minority elements) is <= tolerance.
    """
    total_samples = sum(len(cluster) for cluster in predicted_clusters)
    total_correct = 0

    for pred_cluster in predicted_clusters:
        label_count = Counter()
        for sample in pred_cluster:
            for true_cluster in true_clusters:
                if sample in true_cluster:
                    label_count[tuple(true_cluster)] += 1
        
        if label_count:
            # Find the dominant label
            dominant_label, dominant_count = label_count.most_common(1)[0]
            
            # Check if errors are within tolerance
            cluster_size = len(pred_cluster)
            errors = cluster_size - dominant_count
            
            if errors <= tolerance:
                # If errors are within tolerance, count ALL as correct
                total_correct += cluster_size
            else:
                # Otherwise, standard purity (count only dominant)
                total_correct += dominant_count

    return total_correct / total_samples if total_samples > 0 else 0
