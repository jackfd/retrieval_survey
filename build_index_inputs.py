import argparse
from pathlib import Path

from index_builder.config import load_builder_config
from index_builder.dataset import load_dataset_context
from index_builder.embedding import build_embedding_strategy
from index_builder.pipeline import run_builder


def main() -> None:
    parser = argparse.ArgumentParser(description="Build index input artifacts for one model.")
    parser.add_argument("--dataset-path", required=True, help="Path to datasets root directory.")
    parser.add_argument("--dataset-name", required=True, help="Sub-dataset directory name.")
    parser.add_argument("--model-name", required=True, help="Model name key in model_config.yaml.")
    parser.add_argument("--config-path", default="model_config.yaml", help="Path to model config YAML.")
    parser.add_argument("--output-root", default="output", help="Output root directory.")
    args = parser.parse_args()

    builder_cfg = load_builder_config(Path(args.config_path), model_name=args.model_name)
    dataset_ctx = load_dataset_context(Path(args.dataset_path), dataset_name=args.dataset_name)
    embedding_strategy = build_embedding_strategy(builder_cfg.runtime, builder_cfg.model)

    run_builder(
        dataset_name=args.dataset_name,
        output_root=Path(args.output_root),
        dataset_ctx=dataset_ctx,
        builder_cfg=builder_cfg,
        embedding_strategy=embedding_strategy,
    )


if __name__ == "__main__":
    main()

