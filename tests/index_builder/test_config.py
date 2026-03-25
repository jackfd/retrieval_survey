import tempfile
import textwrap
import unittest
from pathlib import Path

from index_builder.config import load_builder_config


class TestConfig(unittest.TestCase):
    def test_blank_embedding_api_url_keeps_local_resolve_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = Path(tmp) / "model_config.yaml"
            cfg_path.write_text(
                textwrap.dedent(
                    """
                    embedding_dim: 768
                    normalize_embeddings: true
                    max_length: 512
                    query_prefix: ""
                    doc_prefix: ""
                    instruction_template: ""
                    batch_size: 64
                    device: "cpu"
                    embedding_api_url: "   "
                    http_timeout: 10
                    http_max_retries: 1
                    models:
                      m:
                        provider: sentence_transformers
                        model_id: x
                    """
                ),
                encoding="utf-8",
            )
            cfg = load_builder_config(cfg_path, "m")
            self.assertEqual(cfg.runtime.embedding_api_url, "")

    def test_non_empty_embedding_api_url(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = Path(tmp) / "model_config.yaml"
            cfg_path.write_text(
                textwrap.dedent(
                    """
                    embedding_dim: 768
                    normalize_embeddings: true
                    max_length: 512
                    query_prefix: ""
                    doc_prefix: ""
                    instruction_template: ""
                    batch_size: 64
                    device: "cpu"
                    embedding_api_url: "http://localhost:9000/embed"
                    http_timeout: 10
                    http_max_retries: 1
                    models:
                      m:
                        provider: sentence_transformers
                        model_id: x
                    """
                ),
                encoding="utf-8",
            )
            cfg = load_builder_config(cfg_path, "m")
            self.assertEqual(cfg.runtime.embedding_api_url, "http://localhost:9000/embed")


if __name__ == "__main__":
    unittest.main()

