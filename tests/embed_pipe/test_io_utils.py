import tempfile
import unittest
from pathlib import Path

from embed_pipe.infra.output_writer import OutputWriter


class TestOutputWriter(unittest.TestCase):
    def test_load_failures_mixed_record_types_only_returns_doc_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            failures_path = output_dir / "failures.jsonl"
            failures_path.write_text(
                "\n".join(
                    [
                        '{"record_type":"doc","record_id":"d1","error_type":"X","error_message":"m","timestamp_utc":"t","stage":"embedding"}',
                        '{"record_type":"query","record_id":"q1","error_type":"X","error_message":"m","timestamp_utc":"t","stage":"embedding"}',
                        '{"doc_id":"legacy_doc","error_type":"X","error_message":"m","timestamp_utc":"t"}',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            doc_ids = OutputWriter(output_dir=output_dir).load_failed_doc_ids()
            self.assertEqual(doc_ids, ["d1", "legacy_doc"])


if __name__ == "__main__":
    unittest.main()
