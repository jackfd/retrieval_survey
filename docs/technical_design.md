# Technical Design: Index Input Pipeline

## 1. Objective

Define an implementation-ready architecture for generating OpenSearch-ready index input artifacts from benchmark datasets, under strict governance and contract control.

Pipeline scope:

- One run per dataset and model
- Top1 chunk selection per document
- Train-split query embeddings
- Incremental failed-doc retry
- Deterministic output artifact layout

## 2. Non-Negotiable Constraints

1. All public interfaces MUST follow [contract.md](../contract.md).
2. Vector dimension MUST be exactly 768 for docs and queries.
3. No implementation may start before Design Gate and Contract Gate are `PASS`.
4. Output location is fixed: `output/<dataset_name>/<model_name>/`.

## 3. Planned Modules and Responsibilities

## 3.1 `index_scheduler.py`

Responsibilities:

1. Parse scheduler CLI.
2. Load model registry from config.
3. Resolve model execution list:
   - if `--model-name` is provided, run single model
   - otherwise run all configured models in deterministic order
4. Invoke `build_index_inputs.py` once per model.
5. Aggregate run status without early exit on single-model failure.

CLI interface (must match contract):

- `--dataset-path` (required)
- `--dataset-name` (required)
- `--model-name` (optional)
- `--config-path` (optional, default `model_config.yaml`)
- `--output-root` (optional, default `output`)

## 3.2 `build_index_inputs.py` (thin entry)

Responsibilities:

1. Parse builder CLI.
2. Delegate execution to `index_builder` package modules.

`index_builder` module split:

- `config.py`: load/validate YAML, build runtime/model config
- `dataset.py`: resolve sub-dataset directory and dataset paths
- `embedding.py`: strategy mode (`local` / `http`)
- `pipeline.py`: doc/query processing, retry merge, parquet/metadata/logging

CLI interface (must match contract):

- `--dataset-path` (required)
- `--dataset-name` (required)
- `--model-name` (required)
- `--config-path` (optional, default `model_config.yaml`)
- `--output-root` (optional, default `output`)

## 4. Runtime Architecture and Sequence

## 4.1 End-to-End Sequence

1. Scheduler resolves model list.
2. For each model, scheduler launches builder.
3. Builder initializes logging in model output directory.
4. Builder resolves target sub-dataset directory under `datasets/` root.
5. Builder loads dataset metadata and runtime config.
6. Builder selects embedding strategy from YAML:
   - `local`: direct local model encoding
   - `http`: request `embedding_api_url`
7. Builder runs doc pipeline:
   - read docs JSONL
   - run chunk selection
   - compute selected chunk embedding
   - enforce vector length check (768)
8. Builder runs query pipeline:
   - read train queries JSONL
   - embed in batches
   - enforce vector length check (768)
9. Builder writes parquet outputs and metadata.
10. Builder writes failure list and final run summary logs.

## 4.2 Incremental Retry Flow

1. If existing `failures.jsonl` is present:
   - load failed `doc_id` set
   - process only those docs
2. Merge success results into existing `docs.parquet`:
   - key: `doc_id`
   - strategy: overwrite existing row on key match
3. Rewrite `failures.jsonl` with only current unresolved failures.

## 4.3 Dataset Directory Resolution

Given `--dataset-path` as `datasets/` root and input `--dataset-name`:

1. Check exact subdirectory match first.
2. If no exact match, perform case-insensitive subdirectory matching.
3. If no matches, raise `InputValidationError` (dataset not found).
4. If multiple case-insensitive matches, raise `InputValidationError` (ambiguous dataset name).
5. Use the resolved unique subdirectory to locate `dataset.json`, `docs.jsonl`, and `train/queries.jsonl`.

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
- failures jsonl row:
  - `doc_id: string`
  - `error_type: string`
  - `error_message: string`
  - `timestamp_utc: string`
- run metadata json:
  - run timestamps
  - dataset metadata
  - model/provider identity
  - runtime config snapshot
  - doc/query/failure counters

## 6. Failure Modes and Observability

## 6.1 Canonical Error Taxonomy

- `InputValidationError`
- `ModelLoadError`
- `EmbeddingGenerationError`
- `ChunkSelectionError`
- `SerializationError`
- `UnexpectedRuntimeError`

## 6.2 Logging Plan

`app.log` MUST include:

1. Run start context: dataset/model/config summary
2. Per-failure entry with exception class and message
3. End summary: doc_count/query_count/failure_count and duration

## 6.3 Embedding Strategy Rules

1. Strategy source is `model_config.yaml` only.
2. `embedding_mode=local` is default.
3. `embedding_mode=http` requires non-empty `embedding_api_url`.
4. HTTP payload contract:
   - request `{"chunks":[...]}`
   - response `{"vectors":[...]}`

## 7. Stage Gates and Exit Criteria

## 7.1 Design Gate

Exit criteria:

1. This technical design is complete and internally consistent.
2. Sequence, retry flow, failure handling, and observability are explicit.
3. Public interfaces are listed and stable.

Artifacts:

- `docs/technical_design.md`

## 7.2 Contract Gate

Exit criteria:

1. CLI signatures are fully specified.
2. Data schemas and output paths are fully specified.
3. Behavioral constraints and error taxonomy are explicit and testable.

Artifacts:

- `contract.md`

## 7.3 Implementation Gate

Exit criteria:

1. Code follows contract exactly.
2. Retry logic and merge semantics implemented.
3. Logging contract implemented.

Artifacts:

- runtime scripts and config
- test evidence

## 7.4 Validation Gate

Exit criteria:

1. Consistency review passed.
2. Traceability review passed.
3. Gate readiness review passed.
4. Pre-implementation policy respected (no coding before Design+Contract PASS).

Artifacts:

- validation checklist/report

## 8. Verification Plan

## 8.1 Consistency Review

1. CLI args in this design MUST exactly match `contract.md`.
2. Output schemas in this design MUST exactly match `contract.md`.

## 8.2 Traceability Review

Every key requirement must map to:

1. at least one contract clause
2. at least one acceptance criterion or test scenario

## 8.3 Gate Readiness Review

Each gate must have:

1. explicit pass/fail criteria
2. named artifact(s)
3. no ambiguous acceptance language

## 9. Requirement Traceability Matrix

| Requirement | Contract Reference | Validation Evidence |
|---|---|---|
| Top1 doc chunk only | Contract Sec. 4.1 | schema checks + doc-level uniqueness tests |
| Train-only query embedding | Contract Sec. 4.2 | split-path assertion tests |
| 768 strict vector dim | Contract Sec. 2/4.3 | runtime dim assertions + failure-path tests |
| Local/HTTP strategy by YAML only | Contract Sec. 2.4/4.4 | config-driven strategy tests |
| Failed-doc incremental retry | Contract Sec. 4.5 | retry-flow integration test |
| Case-insensitive dataset resolution with ambiguity handling | Contract Sec. 2.1/2.3 | dataset resolution unit tests |
| Model output log path and exception detail | Contract Sec. 5 | log inspection tests |

## 10. Pre-Implementation Rule

Implementation work is blocked until:

1. Design Gate = `PASS`
2. Contract Gate = `PASS`

Only after both are passed may implementation proceed to Implementation Gate.
