DATASET_DIR  := dataset
EMBEDDING_DIR := embedding

.PHONY: install-dataset install-embedding install-all
.PHONY: test-dataset test-embedding test
.PHONY: lint fmt

# ── Installation ──────────────────────────────────────────────────────────────

install-dataset:
	cd $(DATASET_DIR) && uv venv && uv pip install -e ".[dev,trec-car]"

install-embedding:
	cd $(EMBEDDING_DIR) && uv pip install -e ".[st,dev]"

install-all: install-dataset install-embedding

# ── Tests ─────────────────────────────────────────────────────────────────────

test-dataset:
	cd $(DATASET_DIR) && uv run pytest tests/ -q

test-embedding:
	cd $(EMBEDDING_DIR) && uv run pytest tests/ -q

test: test-dataset test-embedding

# ── Linting & formatting ──────────────────────────────────────────────────────

lint:
	uv run --with ruff ruff check $(DATASET_DIR)/retrieval_dataset/ $(EMBEDDING_DIR)/embedding/ $(EMBEDDING_DIR)/tests/

fmt:
	uv run --with ruff ruff format $(DATASET_DIR)/retrieval_dataset/ $(EMBEDDING_DIR)/embedding/ $(EMBEDDING_DIR)/tests/
