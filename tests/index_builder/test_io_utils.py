import tempfile
import unittest
from pathlib import Path

from index_builder.io_utils import load_failures


class TestIoUtils(unittest.TestCase):
    def test_load_failures_mixed_record_types_only_returns_doc_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "failures.jsonl"
            path.write_text(
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

            doc_ids = load_failures(path)
            self.assertEqual(doc_ids, ["d1", "legacy_doc"])


if __name__ == "__main__":
    unittest.main()
