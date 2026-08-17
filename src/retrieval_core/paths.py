"""Path resolution for the shared sample_data layout.

Supports two named datasets:
- "small": the committed default dataset under sample_data/small/.
- "generated_large": the git-ignored dataset produced by
  scripts/generate_large_corpus.py, under sample_data/generated_large/.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_DATA_DIR = PROJECT_ROOT / "sample_data"

DATASET_CHOICES: tuple[str, ...] = ("small", "generated_large")


@dataclass(frozen=True)
class DatasetPaths:
    name: str
    corpus_dir: Path
    eval_queries_path: Path


def resolve_dataset(name: str) -> DatasetPaths:
    """Resolve a dataset name to its corpus directory and eval_queries.json path."""
    if name not in DATASET_CHOICES:
        raise ValueError(
            f"Unknown dataset {name!r}. Choose one of: {', '.join(DATASET_CHOICES)}."
        )
    root = SAMPLE_DATA_DIR / name
    corpus_dir = root / "corpus"
    eval_queries_path = root / "eval_queries.json"
    return DatasetPaths(name=name, corpus_dir=corpus_dir, eval_queries_path=eval_queries_path)


def add_dataset_argument(parser: argparse.ArgumentParser, default: str = "small") -> None:
    """Attach the shared --dataset flag to a CLI parser."""
    parser.add_argument(
        "--dataset",
        choices=DATASET_CHOICES,
        default=default,
        help=f"Named dataset to load (default: {default}).",
    )
