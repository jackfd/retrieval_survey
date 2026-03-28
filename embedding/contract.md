# Contract for Index Input Pipeline

## 1. Contract Intent

This document is normative. The implementation **MUST** comply with all requirements here unless this contract is revised and approved.

Normative keywords:

- **MUST** / **MUST NOT**: mandatory
- **SHALL**: required behavior
- **SHOULD**: recommended, may deviate with documented rationale

## 2. CLI Contracts

### 2.1 `main.py`

| Argument | Required | Default | Type | Rules |
|---|---|---|---|---|
| `--dataset-path` | Yes | None | path | MUST exist; MUST be the `datasets/` root directory |
| `--config-path` | Yes | None | file | MUST exist;  |

Validation rules:

1. Config source is fixed to `model_config.yaml`; builder configs MUST be loaded from YAML sections `experiment`, `inference`, and `models` where `models` is a non-empty list.
2. Output root is fixed to `output`.
3. Dataset candidates are fixed to "hotpotqa_distractor_v1", "msmarco_v1", "scifact_v1", "trec_car_v1".
4. For each dataset candidate, implementation MUST resolve sub-dataset directory under `--dataset-path` by this order:
   - exact directory name match
   - otherwise unique case-insensitive match
5. If case-insensitive matching returns multiple candidates, implementation MUST fail that dataset run with ambiguous dataset-name reason.
6. Resolved sub-dataset directory MUST contain `dataset.json`.
7. Resolved `dataset.json` MUST provide usable `docs_file` and `splits.train.queries_file`.
8. Resolved docs and queries files MUST exist before embedding starts.
9. Output vectors MUST have dimension less than or equal to the configured `embedding_dim`.
10. Embedding strategy selection MUST come from `model_config.yaml` only (no CLI strategy override).
11. Runtime execution order MUST be deterministic by nested iteration:
   - outer loop: models from YAML order
   - inner loop: dataset candidates in fixed list order
12. Any failed (dataset, model) combination MUST cause process exit code to be non-zero.

### 2.2 Dataset Root Conventions

`--dataset-path` points to `datasets/` root.

Current governed dataset candidates are: "hotpotqa_distractor_v1", "msmarco_v1", "scifact_v1", "trec_car_v1"

### 2.3 Embedding Strategy Conventions

`model_config.yaml` MUST define global strategy keys under `experiment` and `inference`:

- `experiment`: embedding behavior source
- `inference.embedding_api_url`: strategy switch source

Behavioral rules:

1. If `inference.embedding_api_url` is empty (or whitespace), implementation MUST use local model encoders directly.
2. If `inference.embedding_api_url` is non-empty, implementation MUST use HTTP mode and payload contract:
   - request: `{"chunks":[...]}`
   - response: `{"vectors":[...]}`
3. Strategy resolution and URL source MUST come from YAML only.

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

`output/<dataset_name>/<model_id>/`

### `docs_dim<embedding_dim>.parquet`

| Field | Type | Required | Constraints |
|---|---|---|---|
| `doc_id` | string | Yes | Unique in output file |
| `chunk_text` | string | Yes | Top1 selected chunk text, non-empty |
| `chunk_embedding` | list<float> | Yes | Length <= embedding_dim |

### `queries_dim<embedding_dim>.parquet`

| Field | Type | Required | Constraints |
|---|---|---|---|
| `query_id` | string | Yes | Unique in output file |
| `query_text` | string | Yes | Original query text |
| `query_embedding` | list<float> | Yes | Length <= embedding_dim |

### `run_metadata.json`

Required top-level keys:

- `run`
- `model`
- `stats`

Minimum required fields:

| Path | Type | Required |
|---|---|---|
| `run.start_time_utc` | string | Yes |
| `run.end_time_utc` | string | Yes |
| `model.model_id` | string | Yes |
| `model.provider` | string | Yes |
| `stats.doc_count` | integer | Yes |
| `stats.query_count` | integer | Yes |

## 4. Behavioral Contracts

1. **Top1 only**: each `doc_id` MUST map to exactly one selected chunk in final `docs_dim<embedding_dim>.parquet`.
2. **Train-only queries**: query embeddings MUST be generated from `train/queries.jsonl` only.
3. **Dimension upper bound**: implementation MUST fail the affected item when vector dimension exceeds `embedding_dim`.
   - Padding or truncation fallback is prohibited.
4. **Embedding strategy**:
   - empty `embedding_api_url` MUST use direct local encoding.
   - non-empty `embedding_api_url` MUST use HTTP mode.
5. **Fail-fast execution**:
   - Any document or query processing error MUST fail the current `(dataset, model)` run immediately.
   - Failed runs MUST return non-zero exit status and MUST NOT degrade into partial-success outputs.

## 5. Logging Contract

1. Every failure record in logs MUST include:
   - exception class name
   - message text
   - related `doc_id` when available
2. Logs SHALL include run start/end summary with counts.
3. Business exceptions MUST be logged at the source layer before re-raising or propagating. Each such log MUST include:
   - function name
   - source line number
   - reason text
   - key context identifier(s) when available (for example: `doc_id`, `query_id`, `batch_index`, `path`, `model_id`)
4. Silent exception swallowing is prohibited.
5. If code intentionally degrades to an empty result (for example `[]` or empty vectors), it MUST emit a complete log entry at that downgrade point, including function name, line number, reason, and context.
6. Orchestration layer (`pipeline` / `processors`) SHOULD avoid duplicate error logs when source-layer detailed logs already exist.
7. Function and line metadata SHOULD be emitted via logger formatter configuration (for example `%(filename)s:%(lineno)d`), not via custom logging wrapper functions.

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
