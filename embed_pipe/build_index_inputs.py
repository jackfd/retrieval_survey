import argparse
import sys
from pathlib import Path

from embed_pipe.app.runner import BuilderRunner
from embed_pipe.infra.config_loader import ConfigLoader
from embed_pipe.infra.dataset_loader import DatasetLoader
from embed_pipe.infra.embedding_strategies import EmbeddingStrategyFactory
from embed_pipe.infra.logger import setup_logger


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build index input artifacts for one model."
    )
    parser.add_argument(
        "--dataset-path", required=True, help="Path to datasets root directory."
    )
    parser.add_argument(
        "--dataset-name", required=True, help="Sub-dataset directory name."
    )
    parser.add_argument(
        "--model-name", required=True, help="Model name key in model_config.yaml."
    )
    parser.add_argument(
        "--config-path", default="model_config.yaml", help="Path to model config YAML."
    )
    parser.add_argument(
        "--output-root", default="output", help="Output root directory."
    )
    args = parser.parse_args()

    output_dir = Path(args.output_root) / args.dataset_name / args.model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(output_dir / "app.log")

    config_loader = ConfigLoader()
    dataset_loader = DatasetLoader()
    strategy_factory = EmbeddingStrategyFactory()

    builder_cfg_result = config_loader.load_builder_config(
        Path(args.config_path), model_name=args.model_name
    )
    if not builder_cfg_result.ok:
        logger.error(
            "event=builder_start_failed reason=%s context=%s",
            "Failed to load builder config",
            "config_path=%s model_name=%s" % (args.config_path, args.model_name),
        )
        return 1

    dataset_ctx_result = dataset_loader.load_dataset_context(
        Path(args.dataset_path), dataset_name=args.dataset_name
    )
    if not dataset_ctx_result.ok:
        logger.error(
            "event=builder_start_failed reason=%s context=%s",
            "Failed to load dataset context",
            "dataset_path=%s dataset_name=%s" % (args.dataset_path, args.dataset_name),
        )
        return 1

    builder_cfg = builder_cfg_result.value
    dataset_ctx = dataset_ctx_result.value
    if builder_cfg is None or dataset_ctx is None:
        logger.error(
            "event=builder_start_failed reason=%s context=%s",
            "Missing required runtime objects",
            "",
        )
        return 1

    embedding_strategy_result = strategy_factory.build(
        builder_cfg.runtime, builder_cfg.model
    )
    if not embedding_strategy_result.ok:
        logger.error(
            "event=builder_start_failed reason=%s context=%s",
            "Failed to build embedding strategy",
            "model_name=%s" % args.model_name,
        )
        return 1
    embedding_strategy = embedding_strategy_result.value
    if embedding_strategy is None:
        logger.error(
            "event=builder_start_failed reason=%s context=%s",
            "Embedding strategy is empty",
            "",
        )
        return 1

    runner = BuilderRunner(
        output_dir=output_dir,
        dataset_ctx=dataset_ctx,
        builder_cfg=builder_cfg,
        embedding_strategy=embedding_strategy,
    )
    logger.info(
        "dataset_root=%s dataset_dir=%s dataset_name=%s model_name=%s",
        dataset_ctx.dataset_root,
        dataset_ctx.resolved_dataset_dir,
        args.dataset_name,
        builder_cfg.model,
    )
    run_result = runner.run()
    if not run_result.ok:
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
