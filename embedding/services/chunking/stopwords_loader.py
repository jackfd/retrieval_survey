# -*- coding: utf-8 -*-
import logging
import os
from typing import Iterable, Set
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from spacy.lang.en.stop_words import STOP_WORDS as EN_STOP_WORDS
from spacy.lang.zh.stop_words import STOP_WORDS as ZH_STOP_WORDS
import yaml

logger = logging.getLogger(__name__)


class StopwordsLoader:
    _cached_stopwords: set[str] | None = None

    @staticmethod
    def load_stopwords() -> Set[str]:
        if StopwordsLoader._cached_stopwords is not None:
            return StopwordsLoader._cached_stopwords

        stop_words: set[str] = set()
        stop_words.update(StopwordsLoader._load_custom_stopwords())
        stop_words.update(StopwordsLoader._load_spacy_stopwords())

        if not stop_words:
            stop_words.update(StopwordsLoader._load_fallback_stopwords())

        StopwordsLoader._cached_stopwords = stop_words
        return stop_words

    @staticmethod
    def _candidate_paths() -> list[str]:
        env_path = os.getenv("STOPWORDS_CONFIG_PATH")
        base_dir = os.path.dirname(__file__)
        candidates = [
            env_path,
            os.path.join(base_dir, "stopwords.yaml"),
            os.path.join(base_dir, "configs", "stopwords.yaml"),
        ]
        return [path for path in candidates if path]

    @staticmethod
    def _extend_if_iterable(container: set[str], values: object) -> None:
        if isinstance(values, Iterable) and not isinstance(values, (str, bytes, dict)):
            container.update(str(v).strip().lower() for v in values if str(v).strip())

    @staticmethod
    def _load_custom_stopwords() -> Set[str]:
        custom_stopwords: set[str] = set()
        for config_path in StopwordsLoader._candidate_paths():
            if not os.path.exists(config_path):
                continue
            try:
                with open(config_path, "r", encoding="utf-8") as file_handle:
                    config = yaml.safe_load(file_handle) or {}
                if not isinstance(config, dict):
                    logger.error(
                        "stopwords_config_invalid config_path=%s: invalid stopwords config format",
                        config_path,
                    )
                    continue

                StopwordsLoader._extend_if_iterable(
                    custom_stopwords, config.get("english_stopwords")
                )
                StopwordsLoader._extend_if_iterable(
                    custom_stopwords, config.get("chinese_stopwords")
                )
                return custom_stopwords
            except Exception as exc:
                logger.exception(
                    "stopwords_load_failed config_path=%s error=%s",
                    config_path,
                    exc,
                )

        return custom_stopwords

    @staticmethod
    def _load_spacy_stopwords() -> Set[str]:
        spacy_stopwords: set[str] = set()

        spacy_stopwords.update({word.lower() for word in EN_STOP_WORDS})
        # spacy_stopwords.update({word.lower() for word in ZH_STOP_WORDS})
        return spacy_stopwords

    @staticmethod
    def _load_fallback_stopwords() -> Set[str]:
        fallback_stopwords = set()
        fallback_stopwords.update({word.lower() for word in ENGLISH_STOP_WORDS})
        return fallback_stopwords
