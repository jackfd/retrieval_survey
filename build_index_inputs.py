import argparse
from pathlib import Path

from embed_pipe.app.runner import BuilderRunner
from embed_pipe.infra.config_loader import ConfigLoader
from embed_pipe.infra.dataset_loader import DatasetLoader
from embed_pipe.infra.embedding_gateway import EmbeddingStrategyFactory
from embed_pipe.infra.logger import setup_logger


def main() -> None:
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

    # Initialize logger early so source-layer validation/model-load errors are persisted.
    output_dir = Path(args.output_root) / args.dataset_name / args.model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(output_dir / "app.log")

    config_loader = ConfigLoader()
    dataset_loader = DatasetLoader()
    strategy_factory = EmbeddingStrategyFactory()

    builder_cfg = config_loader.load_builder_config(Path(args.config_path), model_name=args.model_name)
    dataset_ctx = dataset_loader.load_dataset_context(Path(args.dataset_path), dataset_name=args.dataset_name)
    embedding_strategy = strategy_factory.build(builder_cfg.runtime, builder_cfg.model)

    runner = BuilderRunner(
        dataset_name=args.dataset_name,
        output_dir=output_dir,
        dataset_ctx=dataset_ctx,
        builder_cfg=builder_cfg,
        embedding_strategy=embedding_strategy,
        logger=logger,
    )
    runner.run()


if __name__ == "__main__":
    main()
