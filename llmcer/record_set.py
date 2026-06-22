"""
Next Record Set Creation (NRS) -- paper Algorithm 1 (Section 5.2).

Given the records of a single block, NRS repeatedly carves out record sets of
size up to S_s for in-context clustering. Each record set is built to:
  * respect the set-size constraint S_s,
  * cover up to S_d distinct entities (diversity), and
  * minimise the set variation S_v (coefficient of variation of cluster sizes,
    Eq. 1), i.e. keep the per-entity counts balanced,
  * and order similar records together so the LLM sees a coherent sequence.

These were the paper's empirically optimal knobs: S_s = 9, S_d = 4, sequential
record order, S_v minimised (Section 4.2 / 6.3).
"""

import numpy as np

from llmcer.clustering import elbow_method, kmeans_clustering


def coefficient_of_variation(label_counts):
    """
    Set variation S_v (Eq. 1): variation(S) = sigma(S) / mu(S), the coefficient
    of variation of the cluster (entity) sizes within a record set.

    `label_counts` is an iterable of per-entity counts currently in the set.
    A perfectly balanced set (all counts equal) yields 0.
    """
    counts = np.array([c for c in label_counts if c > 0], dtype=float)
    if counts.size == 0:
        return 0.0
    mu = counts.mean()
    if mu == 0:
        return 0.0
    return float(counts.std() / mu)


def _greedy_similarity_order(ids, similarity_matrix):
    """
    Order records so that similar ones sit next to each other (Algorithm 1
    lines 3-6 / 22): start from the first record and repeatedly append its most
    similar not-yet-placed neighbour. Sequential ordering improved the LLM's
    in-context clustering in the paper's experiments (Section 4.2).
    """
    remaining = list(ids)
    if not remaining:
        return []
    ordered = [remaining.pop(0)]
    while remaining:
        cur = ordered[-1]
        best_idx, best_sim = 0, -float('inf')
        for i, cand in enumerate(remaining):
            s = similarity_matrix[cur][cand]
            if s > best_sim:
                best_sim, best_idx = s, i
        ordered.append(remaining.pop(best_idx))
    return ordered


def _representatives_by_centroid(cluster_ids, vectors, k):
    """Pick the k records of a k-means cluster closest to its centroid."""
    if k >= len(cluster_ids):
        return list(cluster_ids)
    vecs = np.array([vectors[i] for i in cluster_ids])
    centroid = vecs.mean(axis=0)
    dists = np.linalg.norm(vecs - centroid, axis=1)
    order = np.argsort(dists)
    return [cluster_ids[i] for i in order[:k]]


def next_record_set(b_remain, vectors, similarity_matrix, S_s, S_d):
    """
    Build ONE record set from the remaining records of a block (Algorithm 1).

    Args:
        b_remain: list of record indices still to be placed (consumed here).
        vectors: global embedding list, indexable by record index.
        similarity_matrix: global cosine-similarity matrix.
        S_s: set size constraint (paper optimum 9).
        S_d: set diversity constraint (paper optimum 4).

    Returns:
        (record_set, b_remain_after) -- record_set is an ordered list of record
        indices; b_remain_after is the remaining block with the chosen records
        removed.
    """
    b_remain = list(b_remain)
    if not b_remain:
        return [], b_remain

    # --- Small block: the whole block is one ordered record set (lines 2-7). ---
    if len(b_remain) <= S_s:
        record_set = _greedy_similarity_order(b_remain, similarity_matrix)
        return record_set, []

    # --- Large block: diversity-aware construction (lines 8-22). ---
    idx = list(b_remain)
    vecs = np.array([vectors[i] for i in idx])

    # Line 9: estimate diversity k via the (fixed) elbow method.
    k = elbow_method(vecs, max_k=min(S_d, len(idx)))
    k = max(1, min(k, len(idx)))
    # Line 10: k-means on the remaining block.
    labels = kmeans_clustering(vecs, k)

    # group remaining records by k-means label
    clusters = {}
    label_by_record = {}
    for rec, lab in zip(idx, labels):
        clusters.setdefault(int(lab), []).append(rec)
        label_by_record[rec] = int(lab)

    target_size = max(1, S_s // S_d)  # line 11: floor(S_s / S_d)

    record_set = []
    set_label_counts = {}             # label -> count currently in record_set
    chosen = set()

    # Lines 12-17: seed the set with target_size records from each large-enough
    # cluster, giving the set its diversity.
    for lab, members in clusters.items():
        if len(record_set) >= S_s:
            break
        if len(members) >= target_size:
            picked = _representatives_by_centroid(members, vectors, target_size)
            for r in picked:
                if len(record_set) >= S_s:
                    break
                record_set.append(r)
                chosen.add(r)
                set_label_counts[lab] = set_label_counts.get(lab, 0) + 1

    # Lines 18-21: fill up to S_s by adding the remaining record that least
    # increases the set variation S_v (Eq. 1).
    pool = [r for r in idx if r not in chosen]
    while len(record_set) < S_s and pool:
        best_r, best_cv, best_pos = None, float('inf'), None
        for pos, r in enumerate(pool):
            lab = label_by_record[r]
            trial = dict(set_label_counts)
            trial[lab] = trial.get(lab, 0) + 1
            cv = coefficient_of_variation(trial.values())
            if cv < best_cv:
                best_cv, best_r, best_pos = cv, r, pos
        lab = label_by_record[best_r]
        record_set.append(best_r)
        chosen.add(best_r)
        set_label_counts[lab] = set_label_counts.get(lab, 0) + 1
        pool.pop(best_pos)

    # Line 22: order similar records together.
    record_set = _greedy_similarity_order(record_set, similarity_matrix)

    b_remain_after = [r for r in b_remain if r not in chosen]
    return record_set, b_remain_after


def create_record_sets(block, vectors, similarity_matrix, S_s, S_d):
    """
    Partition a whole block into a list of record sets by repeatedly applying
    NRS until the block is exhausted (paper Algorithm 4, line 2).
    """
    b_remain = list(block)
    record_sets = []
    # Guard against pathological non-progress.
    while b_remain:
        rset, b_remain = next_record_set(b_remain, vectors, similarity_matrix, S_s, S_d)
        if not rset:
            # Safety: avoid infinite loop -- flush whatever is left as one set.
            record_sets.append(list(b_remain))
            break
        record_sets.append(rset)
    return record_sets
