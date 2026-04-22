.PHONY: install install-st install-flag test test-all lint fmt type-check run

install:
	uv pip install -e ".[dev]"

install-st:
	uv pip install -e ".[st,dev]"

install-flag:
	uv pip install -e ".[flag,dev]"

test:
	uv run pytest tests/unit -q

test-all:
	uv run pytest tests/ -q

lint:
	uv run ruff check embedding/ tests/

fmt:
	uv run ruff format embedding/ tests/

type-check:
	uv run mypy embedding/

run:
	uv run retrieval-embedding --config-path $(CONFIG) --dataset-path $(DATASET_PATH)
