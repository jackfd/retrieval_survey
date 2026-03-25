import logging
from typing import List

import numpy as np

try:
    from sklearn.cluster import KMeans

    SKLEARN_AVAILABLE = True
except ImportError:
    KMeans = None
    SKLEARN_AVAILABLE = False


class ChunkClusterer:
    def __init__(self, cluster_num: int):
        self.cluster_num = max(1, int(cluster_num))

    def cluster_chunks(self, embeddings: np.ndarray) -> List[int]:
        if embeddings.ndim != 2 or embeddings.shape[0] == 0:
            logging.warning("No valid chunks for clustering, embeddings shape: %s", embeddings.shape)
            return []

        cluster_num = min(self.cluster_num, embeddings.shape[0])
        if cluster_num <= 0:
            return []

        if SKLEARN_AVAILABLE:
            kmeans = KMeans(n_clusters=cluster_num, random_state=42, n_init=10, max_iter=300)
            kmeans.fit(embeddings)
            labels = kmeans.labels_
            centers = kmeans.cluster_centers_

            candidates = []
            for cluster_idx in range(cluster_num):
                idxs = np.where(labels == cluster_idx)[0]
                if len(idxs) == 0:
                    continue
                dists = np.linalg.norm(embeddings[idxs] - centers[cluster_idx], axis=1)
                candidates.append(int(idxs[np.argmin(dists)]))
            return candidates

        centroid = np.mean(embeddings, axis=0)
        dists = np.linalg.norm(embeddings - centroid, axis=1)
        ordered = np.argsort(dists)
        return [int(i) for i in ordered[:cluster_num]]
