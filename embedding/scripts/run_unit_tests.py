#!/usr/bin/env python3
"""Run the embedding unit and integration test suite.

This runner is intentionally lightweight:
- it does not touch production code
- it first tries the whole unit suite
- if the whole-suite run fails, it falls back to per-file runs so you can
  isolate the failing module quickly
- the first run may download model files into the shared local cache
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


UNIT_TEST_FILES = [
    "embedding/tests/unit/test_config_loader.py",
    "embedding/tests/unit/test_embedding_strategies.py",
    "embedding/tests/unit/test_http_local_strategies.py",
    "embedding/tests/unit/test_chunk_splitter.py",
    "embedding/tests/unit/test_chunk_selector.py",
    "embedding/tests/unit/test_services.py",
    "embedding/tests/unit/test_main_flow.py",
    "embedding/tests/unit/test_main_entry.py",
    "embedding/tests/unit/test_model_cache.py",
]

INTEGRATION_TEST_FILES = [
    "embedding/tests/integration/test_builder_runner.py",
]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _run(cmd: list[str], cwd: Path) -> int:
    print(f"\n>>> {' '.join(cmd)}")
    completed = subprocess.run(cmd, cwd=cwd)
    return completed.returncode


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the embedding unit test suite."
    )
    parser.add_argument(
        "--mode",
        choices=("full", "split", "both"),
        default="both",
        help=(
            "full: run unit + integration suites once; "
            "split: run tests file-by-file; "
            "both: run full first and fall back to split on failure"
        ),
    )
    parser.add_argument(
        "--files",
        nargs="*",
        default=UNIT_TEST_FILES + INTEGRATION_TEST_FILES,
        help="Test files to run in split mode.",
    )
    parser.add_argument(
        "--pytest-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Extra arguments passed through to pytest after `--`.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    root = _repo_root()
    python = sys.executable
    base_cmd = [python, "-m", "pytest"]
    extra = args.pytest_args
    if extra[:1] == ["--"]:
        extra = extra[1:]

    if args.mode in ("full", "both"):
        full_unit_cmd = base_cmd + ["embedding/tests/unit", "-q"] + extra
        rc = _run(full_unit_cmd, root)
        if rc != 0 and args.mode == "full":
            return rc
        if rc != 0:
            print("\nUnit suite failed; falling back to split mode.")
        else:
            full_integration_cmd = base_cmd + [
                "embedding/tests/integration",
                "-q",
            ] + extra
            rc = _run(full_integration_cmd, root)
            if rc == 0 or args.mode == "full":
                return rc
            print("\nIntegration suite failed; falling back to split mode.")

    if args.mode in ("split", "both"):
        worst_rc = 0
        for file_path in args.files:
            split_cmd = base_cmd + [file_path, "-q"] + extra
            rc = _run(split_cmd, root)
            worst_rc = worst_rc or rc
        return worst_rc

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
