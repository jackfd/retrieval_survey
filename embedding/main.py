import argparse
import logging
import sys
from pathlib import Path

from embedding.domain.exceptions import EmbedPipeError
from embedding.app.runner import BuilderRunner
from embedding.infra.config_loader import ConfigLoader, BuilderConfig
from embedding.infra.dataset_loader import DatasetLoader
from embedding.infra.embedding_strategies import EmbeddingStrategyFactory
from embedding.infra.logger import setup_logger
from embedding.infra.model_cache import initialize_model_cache
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
OUTPUT_ROOT = "output"
CONFIG_PATH = "model_config.yaml"
DATA_SETS = ["HotpotQA", "MSMARCO", "SciFact", "TREC-CAR"]


def run_once(
    dataset_root: str,
    dataset_name: str,
    builder_cfg: BuilderConfig,
    embedding_strategy,
):
    model_id = builder_cfg.model.model_id
    output_dir = Path(OUTPUT_ROOT) / dataset_name / model_id
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(output_dir / "app.log")

    dataset_loader = DatasetLoader()
    ds_context = dataset_loader.load_dataset_context(
        Path(dataset_root), dataset_name=dataset_name
    )

    runner = BuilderRunner(
        output_dir=output_dir,
        dataset_ctx=ds_context,
        builder_cfg=builder_cfg,
        embedding_strategy=embedding_strategy,
    )
    logger.info(
        "start: dataset_root=%s dataset_dir=%s model_id=%s",
        dataset_root,
        ds_context.resolved_dataset_dir,
        model_id,
    )
    runner.run()


def main(dataset_path) -> int:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("embedding")
    config_loader = ConfigLoader()
    try:
        cache_root = initialize_model_cache()
        logger.info("Using shared model cache root: %s", cache_root)
        all_cfg = config_loader.load_configs(Path(CONFIG_PATH))
        strategy_factory = EmbeddingStrategyFactory()
        strategy_cache = {}
        for builder_cfg in all_cfg.values():
            model_id = builder_cfg.model.model_id
            embedding_strategy = strategy_cache.get(model_id)
            if embedding_strategy is None:
                embedding_strategy = strategy_factory.build(
                    builder_cfg.experiment, builder_cfg.inference, builder_cfg.model
                )
                strategy_cache[model_id] = embedding_strategy
            for dname in DATA_SETS:
                run_once(dataset_path, dname, builder_cfg, embedding_strategy)
        return 0
    except EmbedPipeError as exc:
        logger.error("Pipeline failed with domain error: %s", exc)
        return 1
    except Exception:
        logger.error("Pipeline failed with unexpected error")
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build index input artifacts")
    parser.add_argument("--dataset-path", required=True)
    args = parser.parse_args()
    dataset_path = args.dataset_path
    sys.exit(main(dataset_path))
