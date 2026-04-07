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
    model_path = config.model.model_id.replace("/", "_")
    output_dir = Path(OUTPUT_ROOT) / model_path / dataset_name

    dataset_loader = DatasetLoader()
    ds_context = dataset_loader.load_dataset_context(Path(dataset_root), dataset_name)
    output_writer = OutputWriter(output_dir, config.experiment.embedding_dim)
    max_length = config.experiment.max_length
    try:
        process_doc(embedding, ds_context.docs_path, max_length, output_writer)
        process_query(embedding, ds_context.queries_path, output_writer)
    finally:
        output_writer.close()


def _validate_runtime_inputs(dataset_path: str, config_path: str) -> None:
    dataset_root = Path(dataset_path)
    if not dataset_root.exists():
        raise FileNotFoundError("Dataset path does not exist path=%s" % dataset_root)
    if not dataset_root.is_dir():
        raise NotADirectoryError(
            "Dataset path is not a directory path=%s" % dataset_root
        )

    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError("Config path does not exist path=%s" % config_file)
    if not config_file.is_file():
        raise IsADirectoryError("Config path is not a file path=%s" % config_file)


def main(dataset_path: str, config_path: str) -> int:
    logger = setup_logger(Path("./logs/app.log"))
    config_loader = ConfigLoader()
    current_model_id: str | None = None
    current_dataset_name: str | None = None
    try:
        _validate_runtime_inputs(dataset_path, config_path)
        cache_root = initialize_model_cache()
        logger.info("Using shared model cache root: %s", cache_root)

        all_cfg = config_loader.load_configs(Path(config_path))
        strategy_factory = EmbeddingStrategyFactory()
        strategy_cache = {}
        for builder_cfg in all_cfg.values():
            model_id = builder_cfg.model.model_id
            current_model_id = model_id
            embedding_strategy = strategy_cache.get(model_id)
            if embedding_strategy is None:
                embedding_strategy = strategy_factory.build(
                    builder_cfg.experiment, builder_cfg.inference, builder_cfg.model
                )
                strategy_cache[model_id] = embedding_strategy
            logger.info("start model_id=%s", model_id)
            for dname in DATA_SETS:
                current_dataset_name = dname
                logger.info("   dataset path=%s name=%s", dataset_path, dname)
                run_once(dataset_path, dname, builder_cfg, embedding_strategy)
        return 0
    except EmbedPipeError as exc:
        logger.error(
            "Pipeline failed with domain error: %s model_id=%s dataset_name=%s",
            exc,
            current_model_id,
            current_dataset_name,
        )
        return 1
    except Exception as exc:
        logger.exception(
            "Pipeline failed with unexpected error error_type=%s error=%s model_id=%s dataset_name=%s",
            type(exc).__name__,
            exc,
            current_model_id,
            current_dataset_name,
        )
        return 1


def cli() -> int:
    parser = argparse.ArgumentParser(description="Build index input artifacts")
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--config-path", required=True)
    args = parser.parse_args()
    return main(args.dataset_path, args.config_path)
