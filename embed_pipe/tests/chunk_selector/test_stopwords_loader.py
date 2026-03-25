import os
import tempfile
import unittest

from embed_pipe.services.chunking.stopwords_loader import StopwordsLoader


class TestStopwordsLoader(unittest.TestCase):
    def tearDown(self):
        if "STOPWORDS_CONFIG_PATH" in os.environ:
            del os.environ["STOPWORDS_CONFIG_PATH"]
        StopwordsLoader._cached_stopwords = None

    def test_load_custom_stopwords_from_env_path(self):
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as file_handle:
            file_handle.write("english_stopwords:\n  - custom_word\nchinese_stopwords:\n  - 自定义词\n")
            temp_path = file_handle.name

        os.environ["STOPWORDS_CONFIG_PATH"] = temp_path
        StopwordsLoader._cached_stopwords = None

        stopwords = StopwordsLoader.load_stopwords()
        self.assertIn("custom_word", stopwords)
        self.assertIn("自定义词", stopwords)

        os.remove(temp_path)


if __name__ == "__main__":
    unittest.main()
