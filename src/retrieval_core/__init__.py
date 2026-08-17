"""Shared corpus loading and evaluation utilities for Chapter 3 retrieval stages."""

from retrieval_core.corpus import index_by_id, load_corpus
from retrieval_core.evaluation import (
    build_result,
    hit_at_k,
    load_eval_queries,
    summarize,
    validate_against_corpus,
    write_results_csv,
)
from retrieval_core.paths import DatasetPaths, add_dataset_argument, resolve_dataset
from retrieval_core.types import CSV_FIELDNAMES, Document, EvalQuery, EvalResult

__all__ = [
    "Document",
    "EvalQuery",
    "EvalResult",
    "CSV_FIELDNAMES",
    "load_corpus",
    "index_by_id",
    "load_eval_queries",
    "validate_against_corpus",
    "hit_at_k",
    "build_result",
    "write_results_csv",
    "summarize",
    "DatasetPaths",
    "resolve_dataset",
    "add_dataset_argument",
]
