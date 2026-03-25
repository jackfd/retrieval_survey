# Contract for Index Input Pipeline

## 1. Contract Intent

This document is normative. The implementation **MUST** comply with all requirements here unless this contract is revised and approved.

Normative keywords:

- **MUST** / **MUST NOT**: mandatory
- **SHALL**: required behavior
- **SHOULD**: recommended, may deviate with documented rationale

## 2. CLI Contracts

### 2.1 `build_index_inputs.py`

| Argument | Required | Default | Type | Rules |
|---|---|---|---|---|
| `--dataset-path` | Yes | None | path | MUST exist; MUST be the `datasets/` root directory |
| `--dataset-name` | Yes | None | string | MUST be non-empty; MUST resolve to one sub-dataset directory under `--dataset-path` |
| `--model-name` | Yes | None | string | MUST exist in `model_config.yaml` model registry |
| `--config-path` | No | `model_config.yaml` | path | MUST exist and be valid YAML |
| `--output-root` | No | `output` | path | MUST be writable |

Validation rules:

1. Resolve sub-dataset directory under `--dataset-path` by this order:
   - exact directory name match with `--dataset-name`
   - otherwise unique case-insensitive match
2. If case-insensitive matching returns multiple candidates, implementation MUST raise `InputValidationError` with ambiguous dataset-name reason.
3. Resolved sub-dataset directory MUST contain `dataset.json`.
4. Resolved `dataset.json` MUST provide usable `docs_file` and `splits.train.queries_file`.
5. Resolved docs and queries files MUST exist before embedding starts.
6. If `embedding_dim` is configured as `768`, all output vectors MUST have exactly 768 dimensions.
7. Embedding strategy selection MUST come from `model_config.yaml` only (no CLI strategy override).

### 2.2 `index_scheduler.py`

| Argument | Required | Default | Type | Rules |
|---|---|---|---|---|
| `--dataset-path` | Yes | None | path | Same validation as builder |
| `--dataset-name` | Yes | None | string | Same validation as builder |
| `--model-name` | No | None | string | If absent, scheduler MUST run all configured models |
| `--config-path` | No | `model_config.yaml` | path | MUST exist and be valid YAML |
| `--output-root` | No | `output` | path | MUST be writable |

Scheduling rules:

1. Default mode SHALL execute all configured models in deterministic order.
2. Single-model mode SHALL execute only the specified model.
3. One model failure MUST NOT terminate remaining model runs.

### 2.4 Embedding Strategy Conventions

`model_config.yaml` MUST define global strategy keys:

- `embedding_api_url`: strategy switch source

Behavioral rules:

1. If `embedding_api_url` is empty (or whitespace), implementation MUST use local model encoders directly.
2. If `embedding_api_url` is non-empty, implementation MUST use HTTP mode and payload contract:
   - request: `{"chunks":[...]}`
   - response: `{"vectors":[...]}`
3. Strategy resolution and URL source MUST come from YAML only.

### 2.3 Dataset Root Conventions

`--dataset-path` points to `datasets/` root. Sub-datasets are extensible and MUST NOT be constrained to a fixed whitelist.

Current examples include:

- `HotpotQA`
- `MSMARCO`
- `SciFact`
- `TREC-CAR`

## 3. Data Contracts

## 3.1 Input JSONL Schemas

Input files are read from the resolved sub-dataset directory:

`<dataset_path>/<resolved_dataset_dir>/`

### `docs.jsonl`

| Field | Type | Required | Constraints |
|---|---|---|---|
| `doc_id` | string | Yes | Non-empty; unique in file |
| `doc_text` | string | Yes | Non-empty after trim |

Example:

```json
{"doc_id":"d1","doc_text":"Document body text."}
```

### `queries.jsonl` (train split only)

| Field | Type | Required | Constraints |
|---|---|---|---|
| `query_id` | string | Yes | Non-empty; unique in file |
| `query_text` | string | Yes | Non-empty after trim |

Example:

```json
{"query_id":"q1","query_text":"What is ...?"}
```

### `qrels.jsonl` (read-only for validation compatibility)

| Field | Type | Required | Constraints |
|---|---|---|---|
| `query_id` | string | Yes | Non-empty |
| `doc_id` | string | Yes | Non-empty |
| `relevance` | integer | Yes | Integer >= 0 |

Example:

```json
{"query_id":"q1","doc_id":"d1","relevance":1}
```

## 3.2 Output Contracts

Output directory MUST be:

`output/<dataset_name>/<model_name>/`

### `docs.parquet`

| Field | Type | Required | Constraints |
|---|---|---|---|
| `doc_id` | string | Yes | Unique in output file |
| `chunk_text` | string | Yes | Top1 selected chunk text, non-empty |
| `chunk_embedding` | list<float> | Yes | Length exactly 768 |

### `queries.parquet`

| Field | Type | Required | Constraints |
|---|---|---|---|
| `query_id` | string | Yes | Unique in output file |
| `query_text` | string | Yes | Original query text |
| `query_embedding` | list<float> | Yes | Length exactly 768 |

### `failures.jsonl`

| Field | Type | Required | Constraints |
|---|---|---|---|
| `doc_id` | string | Yes | Failed doc identifier |
| `error_type` | string | Yes | Exception class name |
| `error_message` | string | Yes | Non-empty |
| `timestamp_utc` | string | Yes | ISO-8601 UTC timestamp |

Example:

```json
{"doc_id":"d2","error_type":"ChunkSelectionError","error_message":"No valid chunk selected","timestamp_utc":"2026-03-25T04:00:00Z"}
```

### `run_metadata.json`

Required top-level keys:

- `run`
- `dataset_metadata`
- `model`
- `runtime_config`
- `stats`

Minimum required fields:

| Path | Type | Required |
|---|---|---|
| `run.start_time_utc` | string | Yes |
| `run.end_time_utc` | string | Yes |
| `dataset_metadata.dataset_name` | string | Yes |
| `dataset_metadata.version` | string | Yes |
| `dataset_metadata.subset` | string | Yes |
| `dataset_metadata.task` | string | Yes |
| `dataset_metadata.domain` | string | Yes |
| `dataset_metadata.language` | string | Yes |
| `model.model_name` | string | Yes |
| `model.provider` | string | Yes |
| `runtime_config.embedding_dim` | integer | Yes |
| `runtime_config.batch_size` | integer | Yes |
| `stats.doc_count` | integer | Yes |
| `stats.query_count` | integer | Yes |
| `stats.failure_count` | integer | Yes |

## 4. Behavioral Contracts

1. **Top1 only**: each `doc_id` MUST map to exactly one selected chunk in final `docs.parquet`.
2. **Train-only queries**: query embeddings MUST be generated from `train/queries.jsonl` only.
3. **Strict 768 dimensions**: implementation MUST fail the affected item when vector dimension is not 768.
   - Padding or truncation fallback is prohibited.
4. **Embedding strategy**:
   - empty `embedding_api_url` MUST use direct local encoding.
   - non-empty `embedding_api_url` MUST use HTTP mode.
5. **Incremental retry**:
   - If `failures.jsonl` exists from a previous run, implementation MUST process only failed docs.
   - Successful retries MUST be merged into existing `docs.parquet` by `doc_id` overwrite semantics.
   - Query embeddings MAY be regenerated in full for consistency.

## 5. Logging Contract

1. Log file path MUST be `output/<dataset_name>/<model_name>/app.log`.
2. Every failure record in logs MUST include:
   - exception class name
   - message text
   - related `doc_id` when available
3. Logs SHALL include run start/end summary with counts.
4. Business exceptions MUST be logged at the source layer before re-raising or propagating. Each such log MUST include:
   - function name
   - source line number
   - reason text
   - key context identifier(s) when available (for example: `doc_id`, `query_id`, `batch_index`, `path`, `model_name`)
5. Silent exception swallowing is prohibited.
6. If code intentionally degrades to an empty result (for example `[]` or empty vectors), it MUST emit a complete log entry at that downgrade point, including function name, line number, reason, and context.
7. Orchestration layer (`pipeline` / `processors`) SHOULD avoid duplicate error logs when source-layer detailed logs already exist.
8. Function and line metadata SHOULD be emitted via logger formatter configuration (for example `%(filename)s:%(lineno)d`), not via custom logging wrapper functions.

## 6. Contract Change Protocol

Contract-breaking changes MUST follow this process:

1. Submit contract diff with rationale.
2. Update `docs/technical_design.md` impacted sections.
3. Obtain explicit approval.
4. Only then implement code changes.

Without approval, contract-breaking code changes are prohibited.

## 7. Acceptance Criteria Linkage

No implementation task is complete unless all are true:

1. CLI behavior matches Section 2.
2. Output files match Section 3 schema.
3. Runtime behavior matches Section 4.
4. Logging behavior matches Section 5.
