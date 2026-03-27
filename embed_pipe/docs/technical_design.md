# Technical Design: Index Input Pipeline

## 1. Objective

Define an implementation-aligned architecture for generating index input artifacts from benchmark datasets.

Current scope:

- Single CLI entry: `build_index_inputs.py --dataset-path`
- Iterate all configured models from `model_config.yaml`
- Iterate fixed dataset candidates: `HotpotQA`, `MSMARCO`, `SciFact`, `TREC-CAR`
- Run one `(dataset, model)` build at a time
- Produce deterministic output layout under `output/<dataset_name>/<model_name>/`

## 2. Non-Negotiable Constraints

1. Public interfaces MUST follow [contract.md](../contract.md).
2. Vector dimension MUST be exactly 768 for docs and queries.
3. Output location is fixed: `output/<dataset_name>/<model_name>/`.
4. Any failed `(dataset, model)` combination MUST make process exit non-zero.

## 3. Modules and Responsibilities

## 3.1 `build_index_inputs.py` (entry + orchestrator)

Responsibilities:

1. Parse CLI (`--dataset-path` only).
2. Load model list via `ConfigLoader.load_models(model_config.yaml)`.
3. Use fixed dataset candidates list.
4. Execute nested loop:
   - outer: model list order from YAML
   - inner: fixed dataset order
5. Call `run_once(dataset_path, dataset_name, model_name, config_loader)`.
6. Stop and return non-zero immediately when one run fails.

CLI interface:

- `--dataset-path` (required)

Internal fixed sources:

- config path: `model_config.yaml`
- output root: `output`
- datasets: `HotpotQA`, `MSMARCO`, `SciFact`, `TREC-CAR`

## 3.2 Supporting Components

- `infra/config_loader.py`
  - load YAML config
  - build per-model runtime config
  - return model key list for orchestration
- `infra/dataset_loader.py`
  - resolve sub-dataset directory by exact then case-insensitive unique match
  - load dataset context and file paths from `dataset.json`
- `infra/embedding_strategies/*`
  - select local/http embedding strategy from YAML config
- `app/runner.py`
  - execute doc/query embedding pipeline
  - write artifacts, metadata, and logs

## 4. Runtime Sequence

1. Parse `--dataset-path`.
2. Load model registry from `model_config.yaml`.
3. For each model and each fixed dataset candidate:
   - create `output/<dataset_name>/<model_name>/`
   - init `app.log`
   - load builder config for current model
   - resolve dataset context under `--dataset-path`
   - build embedding strategy
   - run `BuilderRunner`
4. Return `0` if all runs succeed; otherwise return non-zero.

## 5. Data Model (Locked Interfaces)

## 5.1 Input Records

Input files are loaded from:

`<dataset_path>/<resolved_dataset_dir>/`

- `docs.jsonl`: `doc_id`, `doc_text`
- `train/queries.jsonl`: `query_id`, `query_text`
- `train/qrels.jsonl`: `query_id`, `doc_id`, `relevance` (validation compatibility)

## 5.2 Output Records

- docs parquet row:
  - `doc_id: string`
  - `chunk_text: string`
  - `chunk_embedding: list<float>[768]`
- queries parquet row:
  - `query_id: string`
  - `query_text: string`
  - `query_embedding: list<float>[768]`
- run metadata json:
  - run timestamps
  - model/provider identity
  - doc/query counters

## 6. Failure Modes and Observability

1. Business exceptions are logged at source layer before propagation.
2. Log entries include function name, line number, reason text, and context identifiers.
3. Silent exception swallowing is prohibited.
4. `app.log` path is fixed to `output/<dataset_name>/<model_name>/app.log`.
5. Orchestration return code is fail-fast by combination: any failure returns non-zero.

## 7. Verification Plan

1. Contract consistency:
   - CLI, iteration order, and exit semantics match `contract.md`.
2. Interface consistency:
   - README command and argparse signature match.
3. Regression checks:
   - unit test suite runs without interface regressions.

## 8. Requirement Traceability Matrix

| Requirement | Contract Reference | Validation Evidence |
|---|---|---|
| Single CLI arg `--dataset-path` | Contract Sec. 2.1 | argparse + docs check |
| Fixed model config source | Contract Sec. 2.1 | config loader path check |
| Fixed dataset candidates | Contract Sec. 2.1/2.3 | orchestrator loop check |
| 768 strict vector dim | Contract Sec. 2.1/4 | runtime dim assertions + failure-path tests |
| Local/HTTP strategy by YAML only | Contract Sec. 2.3 | strategy selection tests |
| Model output log path and exception detail | Contract Sec. 5 | log inspection tests |
