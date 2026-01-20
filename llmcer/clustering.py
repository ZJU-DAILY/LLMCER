
from sklearn.cluster import KMeans
import numpy as np
import networkx as nx
from lshashpy3 import LSHash
from collections import defaultdict
from llmcer.config import LSH_HASH_SIZE, LSH_INPUT_DIM, LSH_NUM_HASHTABLES

def elbow_method(embeddings, max_k=5):
    if embeddings is None or embeddings.shape[0] < 2:
        return 1  
    
    distortions = []
    K = range(1, min(max_k, embeddings.shape[0]) + 1)
    
    for k in K:
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
        kmeans.fit(embeddings)
        distortions.append(kmeans.inertia_)
    optimal_k_index = np.argmin(distortions[1:]) + 1  
    optimal_k = K[optimal_k_index]

    # print(f"best is : {optimal_k}")
    return optimal_k

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
            # Assuming data is a DataFrame or list where we can get items by index
            # If data is DataFrame:
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
    Misclustering Detection Guardrail (MDG)
    Checks if any record is closer to another cluster than its own.
    
    Args:
        clusters: List of lists of indices (integers).
        similarity_matrix: Pre-computed similarity matrix.
        
    Returns:
        bool: True if acceptable, False if misclustering detected.
    """
    # Filter out empty clusters
    valid_clusters = [c for c in clusters if c]
    
    if len(valid_clusters) < 2:
        return True
        
    for i, current_cluster in enumerate(valid_clusters):
        for r_j in current_cluster:
            # 1. Calculate intra-cluster similarity (Average similarity to other members)
            intra_sims = []
            for r_k in current_cluster:
                if r_j != r_k:
                    intra_sims.append(similarity_matrix[r_j][r_k])
            
            # If singleton, intra-sim is effectively high (1.0) as it has no peers.
            # We treat it as 1.0 to avoid false positives unless it's very close to another cluster.
            avg_intra_sim = sum(intra_sims) / len(intra_sims) if intra_sims else 1.0
            
            # 2. Calculate inter-cluster similarity (Max of average similarity to other clusters)
            max_inter_sim = -1.0
            
            for k, other_cluster in enumerate(valid_clusters):
                if i == k:
                    continue
                    
                inter_sims = []
                for r_k in other_cluster:
                    inter_sims.append(similarity_matrix[r_j][r_k])
                
                if inter_sims:
                    avg_inter_sim = sum(inter_sims) / len(inter_sims)
                    if avg_inter_sim > max_inter_sim:
                        max_inter_sim = avg_inter_sim
            
            # 3. Compare
            # If max_inter_sim is -1 (no other valid clusters), pass.
            if max_inter_sim > -1.0:
                if avg_intra_sim < max_inter_sim:
                    # Misclustering detected
                    return False
                    
    return True
