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
| `--config-path` | Yes | None | file | MUST exist |

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
10. Embedding strategy selection MUST come from `model_config.yaml` only.
11. Runtime execution order MUST be deterministic by nested iteration:
   - outer loop: models from YAML order
   - inner loop: dataset candidates in fixed list order
12. Any failed `(dataset, model)` combination MUST cause process exit code to be non-zero.

### 2.2 Dataset Root Conventions

`--dataset-path` points to `datasets/` root.

Current governed dataset candidates are: "hotpotqa_distractor_v1", "msmarco_v1", "scifact_v1", "trec_car_v1"

### 2.3 Embedding Strategy Conventions

`model_config.yaml` MUST define default strategy keys under `experiment` and `inference`:

- `experiment`: embedding behavior source
- `inference.embedding_api_url`: strategy switch source
- `models[]`: per-model override source for `query_prefix`, `doc_prefix`, `instruction_template`, and `batch_size`

Behavioral rules:

1. If `inference.embedding_api_url` is empty (or whitespace), implementation MUST use local model encoders directly.
2. If `inference.embedding_api_url` is non-empty, implementation MUST use HTTP mode and payload contract:
   - request: `{"chunks":[...]}`
   - response: `{"vectors":[...]}`
3. Strategy resolution and URL source MUST come from YAML only.
4. The effective `batch_size` for a builder (from `models[].batch_size` when present, otherwise `inference.batch_size`) MUST be applied by embedding strategy implementations (local/http); service-layer processors MUST NOT split batches by this field.
5. The effective text preparation fields for a builder (`query_prefix`, `doc_prefix`, `instruction_template`) MUST come from `models[]` overrides when present; otherwise they MUST fall back to top-level `experiment` defaults.

## 3. Data Contracts

### 3.1 Input JSONL Schemas

Input files are read from the resolved sub-dataset directory:

`<dataset_path>/<resolved_dataset_dir>/`

### `docs.jsonl`

| Field | Type | Required | Constraints |
|---|---|---|---|
| `doc_id` | string | Yes | Non-empty; unique in file |
| `doc_text` | string | Yes | Non-empty after trim; service layer also accepts `list[string]` as an ordered text-fragment sequence for compatibility |

Example:

```json
{"doc_id":"d1","doc_text":"Document body text."}
```

Compatibility note:

- When `doc_text` is provided as `list[string]`, each element is treated as an ordered text fragment after trim/filtering.
- The service layer does not interpret those elements as guaranteed sentence boundaries.

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

### 3.2 Output Contracts

Output directory MUST be:

`output/<dataset_name>/<model_id>/`

### `docs_dim<embedding_dim>/part-*.parquet`

| Field | Type | Required | Constraints |
|---|---|---|---|
| `doc_id` | string | Yes | Non-empty |
| `chunk_id` | string | Yes | Unique in docs output set; format `{doc_id}#cNNN...` |
| `chunk_text` | string | Yes | Non-empty |
| `chunk_vector` | list<float> | Yes | Length <= embedding_dim |
| `chunk_score` | float | Yes | Selection score used at admission time |
| `chunk_rank` | integer | Yes | Starts at 1 within a document |

### `queries_dim<embedding_dim>/queries.parquet`

| Field | Type | Required | Constraints |
|---|---|---|---|
| `query_id` | string | Yes | Unique in query output set |
| `query_text` | string | Yes | Original query text |
| `query_embedding` | list<float> | Yes | Length <= embedding_dim |

## 4. Behavioral Contracts

1. **MMR TopN**: each `doc_id` MUST emit up to `top_n` selected chunks in final `docs_dim<embedding_dim>/part-*.parquet`.
2. **Direct keep rule**: if a document produces `<= top_n` candidate chunks, implementation MUST keep all candidates.
3. **Selection determinism**: same input text, parameters, and embedding model MUST produce identical chunk order, chunk IDs, chunk ranks, and chunk scores.
4. **Chunk ID rule**: chunk IDs MUST use the original candidate order, not selection rank.
5. **Train-only queries**: query embeddings MUST be generated from `train/queries.jsonl` only.
6. **Dimension upper bound**: implementation MUST fail the affected item when vector dimension exceeds `embedding_dim`.
   - Padding or truncation fallback is prohibited.
7. **Embedding strategy**:
   - empty `embedding_api_url` MUST use direct local encoding.
   - non-empty `embedding_api_url` MUST use HTTP mode.
8. **Fail-fast execution**:
   - Any document or query processing error MUST fail the current `(dataset, model)` run immediately.
   - Failed runs MUST return non-zero exit status and MUST NOT degrade into partial-success outputs.
9. **Full regeneration**: each run MUST rewrite docs outputs from the current input set; merge-with-existing behavior is prohibited.

## 5. Logging Contract

1. Every failure record in logs MUST include:
   - exception class name
   - message text
   - related `doc_id` when available
2. Logs SHALL include run start/end summary with:
   - `model_id`
   - `resolved_dataset_dir`
   - start/end timestamps
   - `doc_count`
   - `chunk_count`
   - `query_count`
3. Business exceptions MUST be logged at the source layer before re-raising or propagating.
4. Silent exception swallowing is prohibited.
5. Function and line metadata SHOULD be emitted via logger formatter configuration.

## 6. Contract Change Protocol

Contract-breaking changes MUST follow this process:

1. Submit contract diff with rationale.
2. Update `docs/technical_design.md` impacted sections.
3. Obtain explicit approval.
4. Only then implement code changes.

## 7. Acceptance Criteria Linkage

No implementation task is complete unless all are true:

1. CLI behavior matches Section 2.
2. Output files match Section 3 schema.
3. Runtime behavior matches Section 4.
4. Logging behavior matches Section 5.
