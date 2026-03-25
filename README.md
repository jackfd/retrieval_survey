# Index Input Pipeline Governance Pack

## Purpose

This repository contains the governance baseline for building index-input artifacts used by retrieval benchmarks.
The governed pipeline is:

`index_scheduler.py` -> `build_index_inputs.py`

The pipeline transforms benchmark documents and queries into embedding artifacts that are ready for OpenSearch indexing and evaluation.

## Scope

This governance pack defines:

- Required interfaces and runtime behavior
- Data contracts for input and output artifacts
- Agent-level implementation constraints
- Stage gates that block implementation until design and contract quality is approved

This pack does **not** implement runtime code yet. It defines the mandatory rules that implementation must follow.

## Documentation-First Policy

Implementation is blocked until governance documents are reviewed and approved.

- Contract: [contract.md](./contract.md)
- Agent constraints: [agent.md](./agent.md)
- Technical design: [docs/technical_design.md](./docs/technical_design.md)

At minimum, **Design Gate** and **Contract Gate** must be marked `PASS` before coding begins.

## Planned Runtime Workflow

1. `index_scheduler.py` resolves model list and schedules runs.
2. `build_index_inputs.py` runs one model for one dataset:
   - loads model and config
   - selects Top1 chunk for each doc
   - generates query embeddings from `train` split
   - writes parquet artifacts, failures, metadata, and logs

## Prerequisites (for future implementation)

- Python 3.10+
- Core packages (planned):
  - `numpy`
  - `pandas`
  - `pyarrow`
  - `PyYAML`
  - `sentence-transformers`
  - `FlagEmbedding`
  - `torch`

## Expected Dataset Layout

```text
datasets/
├── HotpotQA/      # example only
│   ├── dataset.json
│   ├── docs.jsonl
│   └── train/
│       ├── queries.jsonl
│       └── qrels.jsonl
├── MSMARCO/       # example only
├── SciFact/       # example only
└── TREC-CAR/      # example only
```

`datasets/` is the fixed input root. Sub-dataset directories are extensible and not restricted to the four examples above.

`--dataset-name` is matched against sub-dataset directories under `datasets/` in a case-insensitive way (exact match first, then unique case-insensitive match).

Each resolved sub-dataset directory must contain `dataset.json`, `docs.jsonl`, and `train/queries.jsonl` per contract.

## Output Layout Contract

```text
output/
└── <dataset_name>/
    └── <model_name>/
        ├── docs.parquet
        ├── queries.parquet
        ├── failures.jsonl
        ├── run_metadata.json
        └── app.log
```

The output path and filenames are fixed contract interfaces and must not be changed without contract revision approval.
