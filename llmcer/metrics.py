
from collections import Counter
from sklearn.metrics import adjusted_rand_score

def calculate_purity(true_clusters, predicted_clusters):
    total_samples = sum(len(cluster) for cluster in predicted_clusters)
    total_correct = 0

    for pred_cluster in predicted_clusters:
        label_count = Counter()
        for sample in pred_cluster:
            for true_cluster in true_clusters:
                if sample in true_cluster:
                    label_count[tuple(true_cluster)] += 1
        if label_count:
            max_label_count = max(label_count.values())
            total_correct += max_label_count

    return total_correct / total_samples if total_samples > 0 else 0

def calculate_inverse_purity(true_clusters, predicted_clusters):
    total_samples = sum(len(cluster) for cluster in true_clusters)
    total_correct = 0

    for true_cluster in true_clusters:
        if true_cluster:
            pred_labels = Counter()
            for sample in true_cluster:
                for pred_cluster in predicted_clusters:
                    if sample in pred_cluster:
                        pred_labels[tuple(pred_cluster)] += 1
            if pred_labels:
                max_match = max(pred_labels.values())
                total_correct += max_match

    return total_correct / total_samples if total_samples > 0 else 0

def calculate_fp_measure(true_clusters, predicted_clusters):
    purity = calculate_purity(true_clusters, predicted_clusters)
    inverse_purity = calculate_inverse_purity(true_clusters, predicted_clusters)

    if purity + inverse_purity == 0:
        return 0

    return 2 * (purity * inverse_purity) / (purity + inverse_purity)

def convert_to_labels(clusters, n_samples):
    labels = [-1] * n_samples 
    for cluster_id, cluster in enumerate(clusters):
        for sample in cluster:
            labels[sample] = cluster_id
    return labels

def calculate_ari(true_clusters, predicted_clusters):
    all_samples = set(sample for cluster in true_clusters for sample in cluster) | \
    set(sample for cluster in predicted_clusters for sample in cluster)
    n_samples = max(all_samples) + 1 
    true_labels = convert_to_labels(true_clusters, n_samples)
    predicted_labels = convert_to_labels(predicted_clusters, n_samples)
    ari = adjusted_rand_score(true_labels, predicted_labels)
    return ari
