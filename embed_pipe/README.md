# Index Input Pipeline Governance Pack

## Purpose

This repository contains the governance baseline for building index-input artifacts used by retrieval benchmarks.

Current runtime entry:

`python embed_pipe/main.py --dataset-path <datasets_root>`

The entry script iterates all configured models and fixed datasets (`HotpotQA`, `MSMARCO`, `SciFact`, `TREC-CAR`) and generates embedding artifacts ready for indexing and evaluation.

## Scope

This governance pack defines:

- Required interfaces and runtime behavior
- Data contracts for input and output artifacts
- Agent-level implementation constraints
- Stage gates for design/contract quality

## Documentation-First Policy

- Contract: [contract.md](./contract.md)
- Agent constraints: [agent.md](./agent.md)
- Technical design: [docs/technical_design.md](./docs/technical_design.md)

## Runtime Workflow

1. Run `main.py` with `--dataset-path`.
2. Load builder configs from `model_config.yaml`.
3. Iterate fixed dataset list: `HotpotQA`, `MSMARCO`, `SciFact`, `TREC-CAR`.
4. Execute `run_once(dataset, model_id)` for each combination.
5. Exit with non-zero status when any combination fails.

## Prerequisites

- Python 3.10+
- `PyYAML`
- Other runtime dependencies from `requirements.txt`

## CLI

```bash
python embed_pipe/main.py --dataset-path datasets
```

Argument contract:

- `--dataset-path` (required): datasets root directory

Internal fixed sources:

- Config path: `model_config.yaml`
- Output root: `output`
- Dataset candidates: `HotpotQA`, `MSMARCO`, `SciFact`, `TREC-CAR`

## Expected Dataset Layout

```text
datasets/
├── HotpotQA/
│   ├── dataset.json
│   ├── docs.jsonl
│   └── train/
│       ├── queries.jsonl
│       └── qrels.jsonl
├── MSMARCO/
├── SciFact/
└── TREC-CAR/
```

Each resolved sub-dataset directory must contain `dataset.json`, docs file, and train queries file as defined in `dataset.json`.

## Output Layout Contract

```text
output/
└── <dataset_name>/
    └── <model_id>/
        ├── docs.parquet
        ├── queries.parquet
        ├── run_metadata.json
        └── app.log
```

The output path and filenames are fixed contract interfaces and must not be changed without contract revision approval.

## Test Command

```bash
python -m py_compile $(rg --files embed_pipe -g"*.py")
```
