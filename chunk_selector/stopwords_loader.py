# -*- coding: utf-8 -*-
import logging
import os
from typing import Iterable, Set

import yaml


class StopwordsLoader:
    """
    工业级停用词加载器：支持自定义配置 + spaCy轻量停用词 + sklearn回退。
    """

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
                with open(config_path, "r", encoding="utf-8") as f:
                    config = yaml.safe_load(f) or {}
                if not isinstance(config, dict):
                    logging.warning("Invalid stopwords config format in %s", config_path)
                    continue

                StopwordsLoader._extend_if_iterable(
                    custom_stopwords, config.get("english_stopwords")
                )
                StopwordsLoader._extend_if_iterable(
                    custom_stopwords, config.get("chinese_stopwords")
                )
                return custom_stopwords
            except Exception as exc:
                logging.warning("Failed to load stopwords from %s: %s", config_path, exc)

        return custom_stopwords

    @staticmethod
    def _load_spacy_stopwords() -> Set[str]:
        spacy_stopwords: set[str] = set()
        try:
            from spacy.lang.en.stop_words import STOP_WORDS as EN_STOP_WORDS

            spacy_stopwords.update({word.lower() for word in EN_STOP_WORDS})
        except ImportError:
            logging.info("spaCy English stopwords not available")

        try:
            from spacy.lang.zh.stop_words import STOP_WORDS as ZH_STOP_WORDS

            spacy_stopwords.update({word.lower() for word in ZH_STOP_WORDS})
        except ImportError:
            logging.info("spaCy Chinese stopwords not available")

        return spacy_stopwords

    @staticmethod
    def _load_fallback_stopwords() -> Set[str]:
        fallback_stopwords = set()
        try:
            from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

            fallback_stopwords.update({word.lower() for word in ENGLISH_STOP_WORDS})
        except ImportError:
            logging.warning("sklearn is not installed, fallback stopwords unavailable")
        return fallback_stopwords
