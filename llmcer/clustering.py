
import math
from sklearn.cluster import KMeans
import numpy as np
import networkx as nx
from lshashpy3 import LSHash
from collections import defaultdict
from llmcer.config import LSH_HASH_SIZE, LSH_INPUT_DIM, LSH_NUM_HASHTABLES

def elbow_method(embeddings, max_k=5):
    """
    Estimate the diversity (number of entities) of a block via the elbow
    heuristic (paper Algorithm 1, line 9: "compute diversity of B using the
    elbow method").

    K-means inertia is monotonically non-increasing in k, so picking the k with
    minimum distortion always returns max_k and never finds an elbow. Instead we
    locate the elbow as the k that lies furthest from the line connecting the
    first and last (k, distortion) points (the standard geometric "knee"
    detection used by Kneedle), which corresponds to the point of maximum
    curvature / diminishing returns.
    """
    if embeddings is None or embeddings.shape[0] < 2:
        return 1

    n = embeddings.shape[0]
    upper = min(max_k, n)
    K = list(range(1, upper + 1))
    if len(K) <= 2:
        return min(2, n)

    distortions = []
    for k in K:
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
        kmeans.fit(embeddings)
        distortions.append(kmeans.inertia_)

    distortions = np.asarray(distortions, dtype=float)

    if distortions[0] <= 1e-12:
        return 1

    x = np.asarray(K, dtype=float)
    x_norm = (x - x[0]) / (x[-1] - x[0])
    y = distortions
    y_norm = (y - y.min()) / (y.max() - y.min() + 1e-12)

    x0, y0 = x_norm[0], y_norm[0]
    x1, y1 = x_norm[-1], y_norm[-1]
    dx, dy = (x1 - x0), (y1 - y0)
    denom = math.hypot(dx, dy) + 1e-12
    distances = np.abs(dy * (x_norm - x0) - dx * (y_norm - y0)) / denom

    optimal_k = int(K[int(np.argmax(distances))])
    return max(1, optimal_k)

def kmeans_clustering(embeddings, n_clusters):
    if embeddings is None or len(embeddings) < n_clusters:
        return np.zeros(len(embeddings), dtype=int)
    
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42, n_init=10)
    kmeans.fit(embeddings)
    return kmeans.labels_

def bipartite_clustering(data, similarity_matrix):
    G = nx.Graph()
    for i in range(len(similarity_matrix)):
        G.add_node(i)
    for i in range(len(similarity_matrix)):
        for j in range(len(similarity_matrix)):
            if similarity_matrix[i][j] > 0:
                G.add_edge(i, j)
    node_partition = nx.bipartite.color(G)
    cluster1 = []
    cluster2 = []
    for i, color in node_partition.items():
        if color == 0:
            cluster1.append(data[i])
        else:
            cluster2.append(data[i])

    return [cluster1, cluster2]

def lsh_block(vectors, data, similarity_threshold):
    lsh = LSHash(hash_size=LSH_HASH_SIZE, input_dim=LSH_INPUT_DIM, num_hashtables=LSH_NUM_HASHTABLES)
    for ix, vec in enumerate(vectors):
        lsh.index(vec, extra_data=ix)
    graph = defaultdict(set)
    for ix, vec in enumerate(vectors):
        results = lsh.query(vec, num_results=None, distance_func='cosine')
        for res in results:
            if res[0][1] is not None:
                jx = res[0][1]
                if jx != ix:
                    similarity = 1 - res[1]
                    if similarity > similarity_threshold:
                        graph[ix].add(jx)
                        graph[jx].add(ix)
    
    def find_connected_components(graph):
        visited = set()
        components = []

        def dfs(node, component):
            stack = [node]
            while stack:
                current = stack.pop()
                if current not in visited:
                    visited.add(current)
                    component.append(current)
                    for neighbor in graph[current]:
                        if neighbor not in visited:
                            stack.append(neighbor)

        for node in range(len(data)):
            if node not in visited:
                component = []
                dfs(node, component)
                components.append(component)
        return components
    
    components = find_connected_components(graph)
    clusters_array = []
    clusters = []
    for component in components:
        valid_indices = [idx for idx in component if idx is not None and 0 <= idx < len(data)]
        if valid_indices:
            if hasattr(data, 'iloc'):
                cluster = data.iloc[valid_indices, 0].tolist()
            else:
                cluster = [data[i] for i in valid_indices]
            clusters.append(cluster)
            clusters_array.append(valid_indices)
    print("lsh done")
    return clusters_array

def mdg_check(clusters, similarity_matrix):
    """
    Misclustering Detection Guardrail (MDG) -- paper Algorithm 2 / Definition 1.

    Definition 1:
      * intra-cluster similarity of a record r = the MINIMUM similarity between
        r and the other records in the same cluster;
      * inter-cluster similarity of r = the MAXIMUM similarity between r and
        records in OTHER clusters.
    Algorithm 2 rejects the clustering (returns False) if ANY record has
    intra < inter.

    Args:
        clusters: List of lists of record indices (integers).
        similarity_matrix: Pre-computed similarity matrix.

    Returns:
        bool: True if acceptable, False if a misclustering is detected.

    Note on singletons: a singleton has no same-cluster peers, so its intra
    similarity is undefined. Per Definition 1 the guardrail compares pairs that
    exist; a lone record in its own cluster cannot be "misclustered with" peers
    it does not have. We therefore only test records that have at least one peer.
    A record set whose clustering is entirely singletons is trivially acceptable
    (there is nothing to split further), matching the exit-condition semantics
    in the paper.
    """
    valid_clusters = [c for c in clusters if c]
    if len(valid_clusters) < 2:
        return True

    for i, current_cluster in enumerate(valid_clusters):
        for r_j in current_cluster:
            intra_sims = [similarity_matrix[r_j][r_k]
                          for r_k in current_cluster if r_k != r_j]
            if not intra_sims:
                continue
            min_intra_sim = min(intra_sims)

            max_inter_sim = -float('inf')
            for k, other_cluster in enumerate(valid_clusters):
                if i == k:
                    continue
                for r_k in other_cluster:
                    s = similarity_matrix[r_j][r_k]
                    if s > max_inter_sim:
                        max_inter_sim = s

            if max_inter_sim > -float('inf') and min_intra_sim < max_inter_sim:
                return False

    return True


def find_misclustered_records(clusters, similarity_matrix):
    """
    Return the list of (cluster_index, record) pairs that violate Definition 1,
    i.e. records whose intra-cluster (min) similarity is below their
    inter-cluster (max) similarity. Used by record-set regeneration (paper
    Algorithm 4 lines 5 & 10) to relocate misclustered records rather than
    blindly retrying the same prompt.
    """
    valid_clusters = [c for c in clusters if c]
    offenders = []
    if len(valid_clusters) < 2:
        return offenders

    for i, current_cluster in enumerate(valid_clusters):
        for r_j in current_cluster:
            intra_sims = [similarity_matrix[r_j][r_k]
                          for r_k in current_cluster if r_k != r_j]
            if not intra_sims:
                continue
            min_intra_sim = min(intra_sims)

            max_inter_sim = -float('inf')
            best_other = None
            for k, other_cluster in enumerate(valid_clusters):
                if i == k:
                    continue
                for r_k in other_cluster:
                    s = similarity_matrix[r_j][r_k]
                    if s > max_inter_sim:
                        max_inter_sim = s
                        best_other = k

            if max_inter_sim > -float('inf') and min_intra_sim < max_inter_sim:
                offenders.append((i, r_j, best_other))
    return offenders
