import unittest
from unittest.mock import patch

import numpy as np

from embed_pipe.services.chunking.chunk_clusterer import ChunkClusterer


class TestChunkClusterer(unittest.TestCase):
    def setUp(self):
        self.clusterer = ChunkClusterer(cluster_num=3)

    @patch("embed_pipe.services.chunking.chunk_clusterer.SKLEARN_AVAILABLE", False)
    def test_cluster_chunks_fallback(self):
        embeddings = np.array(
            [
                [0.0, 0.0],
                [0.1, 0.1],
                [10.0, 10.0],
                [10.2, 10.1],
            ]
        )
        result = self.clusterer.cluster_chunks(embeddings)

        self.assertEqual(len(result), 3)
        self.assertTrue(all(isinstance(i, int) for i in result))

    def test_cluster_chunks_empty_embeddings(self):
        result = self.clusterer.cluster_chunks(np.empty((0, 0)))
        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()
