"""
End-to-end pipeline orchestration -- paper Algorithm 4 (Section 5.4).

For each block produced by blocking:
  1. NRS (Algorithm 1) carves the block into record sets of size <= S_s.
  2. Each record set is in-context clustered by the LLM, with the Misclustering
     Detection Guardrail (Algorithm 2) and record-set regeneration verifying /
     repairing the output (in_context_cluster).
  3. CMR (Algorithm 3) hierarchically merges the clusters of these record sets
     -- WITHIN the block only -- to obtain the block's final partition.
  4. Block partitions are concatenated to form the final ER result.

Because every step is confined to a single block, records in different blocks
are never compared or merged, preserving the paper's hard-partition semantics
and O(|R|*b) complexity (this fixes the previous global-merge behaviour).
"""

import warnings
from concurrent.futures import ThreadPoolExecutor

# Filter sklearn ConvergenceWarning
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

from llmcer.config import SET_SIZE, SET_DIVERSITY
from llmcer.record_set import create_record_sets
from llmcer.cluster_merge import cluster_merge
from llmcer.llm_interaction import in_context_cluster


def _empty_stats():
    return dict(api_calls=0, time=0.0, tokens=0, in_tokens=0,
                out_tokens=0, mdg_fails=0, rounds=0)


def _merge_stats(into, other):
    for k in ('api_calls', 'tokens', 'in_tokens', 'out_tokens', 'mdg_fails', 'rounds'):
        into[k] += other.get(k, 0)
    into['time'] += other.get('time', 0.0)


def process_block(block, vectors, simi_matrix, df, S_s=SET_SIZE, S_d=SET_DIVERSITY):
    """
    Resolve one block end-to-end (NRS -> in-context clustering + MDG -> CMR).

    Args:
        block: list of record indices belonging to this block.
        vectors: global embeddings (indexable by record index).
        simi_matrix: global similarity matrix.
        df: dataframe for prompt generation.

    Returns:
        (clusters, stats) for this block.
    """
    stats = _empty_stats()
    block = list(block)

    if len(block) == 0:
        return [], stats
    if len(block) == 1:
        return [block], stats

    # 1. NRS: partition the block into record sets of size <= S_s.
    record_sets = create_record_sets(block, vectors, simi_matrix, S_s, S_d)

    # 2. In-context cluster each record set (with MDG + regeneration).
    clustered_sets = []
    for rs in record_sets:
        if not rs:
            continue
        if len(rs) == 1:
            clustered_sets.append([list(rs)])
            continue
        clusters, s = in_context_cluster(rs, df, simi_matrix)
        _merge_stats(stats, s)
        clustered_sets.append(clusters)

    # 3. CMR: hierarchically merge clusters across record sets within this block.
    #    The merge rounds also in-context cluster representative records; per
    #    Algorithm 4 line 9 these outputs are MDG-checked / regenerated too.
    def _llm_cluster_fn(record_ids, frame):
        return in_context_cluster(record_ids, frame, simi_matrix)

    final_clusters, merge_stats = cluster_merge(
        clustered_sets, vectors, simi_matrix, df,
        llm_cluster_fn=_llm_cluster_fn, S_s=S_s, S_d=S_d,
    )
    _merge_stats(stats, merge_stats)

    return final_clusters, stats


def run_blocks(vectors, simi_matrix, blocks, df, S_s=SET_SIZE, S_d=SET_DIVERSITY,
               parallel=True):
    """
    Resolve every block independently and concatenate the partitions.

    Returns (final_clusters, stats).
    """
    total_stats = _empty_stats()
    all_clusters = []

    if parallel:
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_block, b, vectors, simi_matrix, df, S_s, S_d)
                       for b in blocks]
            for fut in futures:
                clusters, s = fut.result()
                all_clusters.extend(clusters)
                _merge_stats(total_stats, s)
    else:
        for b in blocks:
            clusters, s = process_block(b, vectors, simi_matrix, df, S_s, S_d)
            all_clusters.extend(clusters)
            _merge_stats(total_stats, s)

    return all_clusters, total_stats
