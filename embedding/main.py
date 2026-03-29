import argparse
from pathlib import Path

from embedding.domain.exceptions import EmbedPipeError
from embedding.infra.config_loader import ConfigLoader, BuilderConfig
from embedding.infra.dataset_loader import DatasetLoader
from embedding.infra.embedding_strategies import EmbeddingStrategyFactory
from embedding.infra.logger import setup_logger
from embedding.infra.model_cache import initialize_model_cache
from embedding.infra.output_writer import OutputWriter
from embedding.services.document_service import process_doc
from embedding.services.query_service import process_query
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
OUTPUT_ROOT = "output"
DATA_SETS = ["scifact_v1", "hotpotqa_distractor_v1", "msmarco_v1", "trec_car_v1"]


def run_once(dataset_root: str, dataset_name: str, config: BuilderConfig, embedding):
    model_id = config.model.model_id
    output_dir = Path(OUTPUT_ROOT) / model_id / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_loader = DatasetLoader()
    ds_context = dataset_loader.load_dataset_context(Path(dataset_root), dataset_name)
    output_writer = OutputWriter(output_dir, config.experiment.embedding_dim)
    try:
        # doc
        process_doc(embedding, ds_context.docs_path, output_writer)

        # query
        process_query(embedding, ds_context.queries_path, output_writer)
    finally:
        output_writer.close()


def main(dataset_path: str, config_path: str) -> int:
    logger = setup_logger(Path("./logs/app.log"))
    config_loader = ConfigLoader()
    try:
        cache_root = initialize_model_cache()
        logger.info("Using shared model cache root: %s", cache_root)

        all_cfg = config_loader.load_configs(Path(config_path))
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
            logger.info("start model_id=%s", model_id)
            for dname in DATA_SETS:
                logger.info("   dataset path=%s name=%s", dataset_path, dname)
                run_once(dataset_path, dname, builder_cfg, embedding_strategy)
        return 0
    except EmbedPipeError as exc:
        logger.error("Pipeline failed with domain error: %s", exc)
        return 1
    except Exception:
        logger.error("Pipeline failed with unexpected error")
        return 1


def cli() -> int:
    parser = argparse.ArgumentParser(description="Build index input artifacts")
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--config-path", required=True)
    args = parser.parse_args()
    return main(args.dataset_path, args.config_path)
