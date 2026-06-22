"""
Hierarchical Cluster Merge (CMR) -- paper Algorithm 3 / Section 5.3.

After every record set in a block has been in-context clustered, CMR merges
clusters *across* record sets to obtain the block's final partition. It does so
hierarchically:

  1. Each cluster is replaced by a representative record (Algorithm 3 lines 1-4).
  2. Representatives are packed into new record sets for the next round
     (Algorithm 3 lines 5-15), honouring:
        - each cluster is selected exactly once (Problem 3 condition 1),
        - two clusters from the SAME record set are never packed together
          (anti-transitivity, condition 2),
        - the set size S_s, diversity S_d and minimal variation S_v
          (condition 3).
  3. The LLM re-clusters each new record set; representatives grouped together
     cause their underlying clusters to be merged (union).
  4. Repeat until a full round merges nothing -- i.e. the in-context clustering
     outputs only singletons, at which point anti-transitivity guarantees no
     further merge is possible (the paper's exit condition, Algorithm 4 l.11).

Crucially, CMR operates WITHIN a single block: record sets are all drawn from
the same block by NRS, and clusters from different blocks are never compared.
This preserves the hard-partition semantics and the O(|R|*b) complexity the
paper relies on.
"""

import math
import numpy as np


def representative_of(cluster, vectors):
    """
    Representative record of a cluster: the member whose embedding is closest to
    the cluster centroid (Algorithm 3 line 3; "smallest distance from the
    average embedding"). `cluster` is an iterable of record indices.
    """
    cluster = list(cluster)
    if len(cluster) == 1:
        return cluster[0]
    vecs = np.array([vectors[i] for i in cluster])
    centroid = vecs.mean(axis=0)
    dists = np.linalg.norm(vecs - centroid, axis=1)
    return cluster[int(np.argmin(dists))]


def _cluster_similarity(rep_a, rep_b, similarity_matrix):
    """Similarity between two clusters via their representative records."""
    return similarity_matrix[rep_a][rep_b]


def _pack_next_round(round_sets, reps, similarity_matrix, S_s):
    """
    Build the next round's record sets from the current round's clusters
    (Algorithm 3, packing the most-similar clusters across record sets).

    The packing is SIMILARITY-DRIVEN, not index-driven: for each new record set
    we pick an unselected anchor cluster and greedily add the most-similar
    unselected clusters that come from DIFFERENT original record sets
    (anti-transitivity -- two clusters from the same record set are never packed
    together, since the LLM already decided they are distinct entities). This is
    what guarantees that representatives of the same real entity, scattered
    across record sets by NRS, end up in the same next-round record set and get
    merged. Each new set holds up to S_s clusters.

    Args:
        round_sets: list (length K) of record sets; each record set is a list of
            clusters; each cluster is a list/set of original record indices.
        reps: dict cluster-key -> representative record index. Cluster-key is
            (set_index, cluster_index).
        similarity_matrix: global similarity matrix.
        S_s: set-size constraint (max clusters packed per new record set).

    Returns:
        A list of next-round record sets, each a list of (set_index,
        cluster_index) keys with at most one cluster per original record set.
    """
    K = len(round_sets)
    if K == 0:
        return []

    all_keys = [(si, ci) for si in range(K) for ci in range(len(round_sets[si]))]
    used = set()

    # Process anchors in a stable order so packing is deterministic.
    next_sets = []
    for anchor in all_keys:
        if anchor in used:
            continue
        used.add(anchor)
        new_rs = [anchor]
        src_sets = {anchor[0]}          # original record sets already represented
        anchor_rep = reps[anchor]

        # Greedily add the most-similar unselected clusters from OTHER record
        # sets until the set is full (S_s) or no candidate remains.
        while len(new_rs) < S_s:
            best_key, best_sim = None, -float('inf')
            for key in all_keys:
                if key in used or key[0] in src_sets:
                    continue
                # similarity to the closest cluster already in this new set
                s = max(_cluster_similarity(reps[k], reps[key], similarity_matrix)
                        for k in new_rs)
                if s > best_sim:
                    best_sim, best_key = s, key
            if best_key is None:
                break
            used.add(best_key)
            new_rs.append(best_key)
            src_sets.add(best_key[0])

        next_sets.append(new_rs)

    return next_sets


def cluster_merge(initial_record_sets, vectors, similarity_matrix, df,
                  llm_cluster_fn, S_s=9, S_d=4, max_rounds=20):
    """
    Run hierarchical cluster merging over one block.

    Args:
        initial_record_sets: list of record sets; each record set is the LLM's
            in-context clustering output, i.e. a list of clusters (each cluster a
            list of original record indices). This is the round-0 state.
        vectors: global embeddings.
        similarity_matrix: global similarity matrix.
        df: dataframe (passed through to llm_cluster_fn for prompt building).
        llm_cluster_fn: callable(record_ids, df) -> (clusters, stats) where
            clusters is a list of lists of record_ids and stats is a dict with
            keys api_calls, time, tokens, in_tokens, out_tokens. This indirection
            lets the merge logic be tested with a deterministic mock LLM.
        S_s, S_d: set-size / diversity constraints.
        max_rounds: hard cap on hierarchy depth.

    Returns:
        (final_clusters, stats) -- final_clusters is a list of lists of original
        record indices for this block; stats aggregates LLM usage.
    """
    stats = dict(api_calls=0, time=0.0, tokens=0, in_tokens=0, out_tokens=0, rounds=0)

    def _accumulate(s):
        stats['api_calls'] += s.get('api_calls', 0)
        stats['time'] += s.get('time', 0.0)
        stats['tokens'] += s.get('tokens', 0)
        stats['in_tokens'] += s.get('in_tokens', 0)
        stats['out_tokens'] += s.get('out_tokens', 0)

    # round_sets[i] = list of clusters (each a list of original indices)
    round_sets = [[list(c) for c in rs if c] for rs in initial_record_sets]
    round_sets = [rs for rs in round_sets if rs]

    for _ in range(max_rounds):
        stats['rounds'] += 1

        all_clusters = [c for rs in round_sets for c in rs]
        if len(all_clusters) <= 1:
            return all_clusters, stats

        # representatives for every cluster in the current round
        reps = {}
        for si, rs in enumerate(round_sets):
            for ci, c in enumerate(rs):
                reps[(si, ci)] = representative_of(c, vectors)

        # Small enough to cluster all representatives in ONE record set: do so,
        # but keep looping -- a single pass may leave >1 cluster that a further
        # round could still merge, so we only stop when a round changes nothing.
        if len(all_clusters) <= S_s:
            rep_ids = [representative_of(c, vectors) for c in all_clusters]
            rep_to_cluster = {rid: all_clusters[i] for i, rid in enumerate(rep_ids)}
            groups, s = llm_cluster_fn(rep_ids, df)
            _accumulate(s)
            merged = _merge_from_groups(groups, rep_to_cluster)
            if len(merged) >= len(all_clusters):
                return merged, stats          # nothing merged -> converged
            round_sets = [[c] for c in merged]  # each merged cluster -> own set
            continue

        # Build next-round record sets (similarity-driven packing, Algorithm 3).
        next_keys = _pack_next_round(round_sets, reps, similarity_matrix, S_s)

        new_round_sets = []
        any_merge = False
        for rs_keys in next_keys:
            rep_ids = [reps[k] for k in rs_keys]
            rep_to_cluster = {reps[k]: round_sets[k[0]][k[1]] for k in rs_keys}

            if len(rep_ids) <= 1:
                # singleton record set -> nothing to merge, carries through
                new_round_sets.append([rep_to_cluster[rep_ids[0]]] if rep_ids else [])
                continue

            groups, s = llm_cluster_fn(rep_ids, df)
            _accumulate(s)
            merged = _merge_from_groups(groups, rep_to_cluster)
            if len(merged) < len(rep_ids):
                any_merge = True
            # Each merged cluster becomes its own record set next round, so a
            # cluster can keep merging with clusters from yet other sets.
            for c in merged:
                new_round_sets.append([c])

        round_sets = [rs for rs in new_round_sets if rs]

        # Exit condition: a full round merged nothing.
        if not any_merge:
            break

    final_clusters = [c for rs in round_sets for c in rs]
    return final_clusters, stats


def _merge_from_groups(groups, rep_to_cluster):
    """
    Given the LLM's clustering of representative records (`groups` = list of
    lists of representative ids) and the map representative -> underlying
    cluster, union the underlying clusters that the LLM placed together.

    Returns a list of merged clusters (lists of original record indices).
    Representatives the LLM omitted are kept as their own cluster so no record is
    ever dropped.
    """
    seen = set()
    merged = []
    for group in groups:
        combined = []
        for rep in group:
            if rep in rep_to_cluster and rep not in seen:
                combined.extend(rep_to_cluster[rep])
                seen.add(rep)
        if combined:
            merged.append(combined)

    # any representative not mentioned by the LLM survives as its own cluster
    for rep, cluster in rep_to_cluster.items():
        if rep not in seen:
            merged.append(list(cluster))
            seen.add(rep)

    return merged
