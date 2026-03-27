import argparse
import logging
import sys
from pathlib import Path

from embed_pipe.domain.exceptions import EmbedPipeError
from embed_pipe.app.runner import BuilderRunner
from embed_pipe.infra.config_loader import ConfigLoader
from embed_pipe.infra.dataset_loader import DatasetLoader
from embed_pipe.infra.embedding_strategies import EmbeddingStrategyFactory
from embed_pipe.infra.logger import setup_logger

output_root = "output"
config_path = "model_config.yaml"
dataset_path = "dataset"


def run_once(dataset_path, dataset_name, model_name, config_loader):
    output_dir = Path(output_root) / dataset_name / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(output_dir / "app.log")

    dataset_loader = DatasetLoader()
    strategy_factory = EmbeddingStrategyFactory()

    builder_cfg = config_loader.load_builder_config(Path(config_path), model_name=model_name)
    dataset_ctx = dataset_loader.load_dataset_context(
        Path(dataset_path), dataset_name=dataset_name
    )
    embedding_strategy = strategy_factory.build(
        builder_cfg.runtime, builder_cfg.model
    )

    runner = BuilderRunner(
        output_dir=output_dir,
        dataset_ctx=dataset_ctx,
        builder_cfg=builder_cfg,
        embedding_strategy=embedding_strategy,
    )
    logger.info(
        "dataset_root=%s dataset_dir=%s model_name=%s",
        dataset_ctx.dataset_root,
        dataset_ctx.resolved_dataset_dir,
        builder_cfg.model.model_name,
    )
    runner.run()
    return 0


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    bootstrap_logger = logging.getLogger("embed_pipe")
    parser = argparse.ArgumentParser(description="Build index input artifacts")
    parser.add_argument("--dataset-path", required=True)
    args = parser.parse_args()

    try:
        config_loader = ConfigLoader()
        models = config_loader.load_models(Path(config_path))
        data_sets = ["HotpotQA", "MSMARCO", "SciFact", "TREC-CAR"]

        for m in models:
            for dname in data_sets:
                ret = run_once(args.dataset_path, dname, m, config_loader)
                if ret != 0:
                    return ret
        return 0
    except EmbedPipeError as exc:
        bootstrap_logger.error("Pipeline failed with domain error: %s", exc)
        return 1
    except Exception:
        bootstrap_logger.exception("Pipeline failed with unexpected error")
        return 1


if __name__ == "__main__":
    sys.exit(main())
