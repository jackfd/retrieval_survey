#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  run_flag.sh --dataset-path <path> [--source-config <path>] [--generated-config <path>]

Options:
  --dataset-path      Required. Dataset root path.
  --source-config     Optional. Source model config yaml. Default: embedding/model_config.yaml
  --generated-config  Optional. Generated filtered config path. Default: ./model_config.flag.yaml
EOF
}

DATASET_PATH=""
SOURCE_CONFIG="${SOURCE_CONFIG:-embedding/model_config.yaml}"
GENERATED_CONFIG="${GENERATED_CONFIG:-./model_config.flag.yaml}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset-path)
      DATASET_PATH="${2:-}"
      shift 2
      ;;
    --source-config)
      SOURCE_CONFIG="${2:-}"
      shift 2
      ;;
    --generated-config)
      GENERATED_CONFIG="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$DATASET_PATH" ]]; then
  echo "Missing required argument: --dataset-path"
  usage
  exit 1
fi

PROVIDER="flag_embedding"

python3 - "$SOURCE_CONFIG" "$GENERATED_CONFIG" "$PROVIDER" <<'PY'
import sys
from pathlib import Path
import yaml

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
provider = sys.argv[3]

with src.open("r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}

models = cfg.get("models")
if not isinstance(models, list) or not models:
    raise SystemExit("Config key 'models' must be a non-empty list")

selected = [m for m in models if isinstance(m, dict) and m.get("provider") == provider]
if not selected:
    raise SystemExit(f"No models found for provider={provider}")

disabled = []
for m in models:
    if isinstance(m, dict):
        mid = str(m.get("model_id", "")).strip()
        if m.get("provider") != provider and mid:
            disabled.append(mid)

out = {
    "experiment": cfg.get("experiment", {}),
    "inference": cfg.get("inference", {}),
    "models": selected,
    "disabled_models": disabled,
}

dst.parent.mkdir(parents=True, exist_ok=True)
with dst.open("w", encoding="utf-8") as f:
    yaml.safe_dump(out, f, allow_unicode=True, sort_keys=False)
PY

python3 -m embedding.main --dataset-path "$DATASET_PATH" --config-path "$GENERATED_CONFIG"
