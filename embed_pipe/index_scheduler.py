import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml


def load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing config file: {path}")
    with path.open("r", encoding="utf-8") as fin:
        cfg = yaml.safe_load(fin) or {}
    if not isinstance(cfg, dict):
        raise ValueError("Config must be a mapping")
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Schedule index input generation across models.")
    parser.add_argument("--dataset-path", required=True, help="Path to datasets root directory.")
    parser.add_argument("--dataset-name", required=True, help="Sub-dataset directory name.")
    parser.add_argument("--model-name", default=None, help="Optional single model to run.")
    parser.add_argument("--config-path", default="model_config.yaml", help="Path to model config YAML.")
    parser.add_argument("--output-root", default="output", help="Output root directory.")
    args = parser.parse_args()

    cfg = load_yaml(Path(args.config_path))
    models = cfg.get("models", {})
    if not isinstance(models, dict) or not models:
        raise ValueError("Config key 'models' must be a non-empty mapping")

    if args.model_name:
        if args.model_name not in models:
            raise ValueError(f"Unknown model_name={args.model_name!r}")
        selected_models: List[str] = [args.model_name]
    else:
        selected_models = list(models.keys())

    results: List[Dict[str, Any]] = []
    for model_name in selected_models:
        cmd = [
            sys.executable,
            "build_index_inputs.py",
            "--dataset-path",
            args.dataset_path,
            "--dataset-name",
            args.dataset_name,
            "--model-name",
            model_name,
            "--config-path",
            args.config_path,
            "--output-root",
            args.output_root,
        ]
        print(f"[scheduler] start model={model_name}")
        proc = subprocess.run(cmd, capture_output=True, text=True)
        results.append(
            {
                "model_name": model_name,
                "returncode": proc.returncode,
                "stdout_tail": proc.stdout[-3000:],
                "stderr_tail": proc.stderr[-3000:],
            }
        )
        if proc.returncode == 0:
            print(f"[scheduler] success model={model_name}")
        else:
            print(f"[scheduler] failed model={model_name} returncode={proc.returncode}")

    success = [r for r in results if r["returncode"] == 0]
    failed = [r for r in results if r["returncode"] != 0]
    summary = {
        "dataset_name": args.dataset_name,
        "total_models": len(results),
        "success_count": len(success),
        "failure_count": len(failed),
        "results": results,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
