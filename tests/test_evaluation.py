"""Unit tests for evaluation queries, hit scoring, and CSV output in retrieval_core/evaluation.py."""

from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path

import pytest

from retrieval_core.evaluation import (
    build_result,
    hit_at_k,
    load_eval_queries,
    summarize,
    summarize_by_failure_mode,
    validate_against_corpus,
    write_results_csv,
)
from retrieval_core.types import CSV_FIELDNAMES, EvalQuery, EvalResult


def test_load_eval_queries_valid():
    """Verify loading valid eval queries JSON."""
    with tempfile.TemporaryDirectory() as tmpdir:
        qfile = Path(tmpdir) / "eval.json"
        data = [
            {
                "id": "q1",
                "query": "Where is the report?",
                "expected_doc_ids": ["doc_1"],
                "failure_mode": "keyword_exact_match",
                "notes": "Test query",
            }
        ]
        qfile.write_text(json.dumps(data), encoding="utf-8")

        queries = load_eval_queries(qfile)
        assert len(queries) == 1
        assert queries[0].id == "q1"
        assert queries[0].expected_doc_ids == ("doc_1",)
        assert queries[0].failure_mode == "keyword_exact_match"


def test_load_eval_queries_invalid_json():
    """Verify invalid JSON raises ValueError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        qfile = Path(tmpdir) / "eval.json"
        qfile.write_text("invalid json content", encoding="utf-8")
        with pytest.raises(ValueError, match="Invalid JSON"):
            load_eval_queries(qfile)


def test_load_eval_queries_empty_or_non_list():
    """Verify empty list or non-list raises ValueError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        qfile = Path(tmpdir) / "eval.json"
        qfile.write_text("[]", encoding="utf-8")
        with pytest.raises(ValueError, match="non-empty JSON list"):
            load_eval_queries(qfile)

        qfile.write_text('{"key": "value"}', encoding="utf-8")
        with pytest.raises(ValueError, match="non-empty JSON list"):
            load_eval_queries(qfile)


def test_load_eval_queries_missing_fields_and_duplicates():
    """Verify missing required fields and duplicate query IDs raise ValueError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        qfile = Path(tmpdir) / "eval.json"

        # Missing expected_doc_ids
        qfile.write_text(
            json.dumps([{"id": "q1", "query": "Test", "failure_mode": "keyword_exact_match"}]),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="missing fields"):
            load_eval_queries(qfile)

        # Duplicate ID
        qfile.write_text(
            json.dumps(
                [
                    {
                        "id": "q1",
                        "query": "Test 1",
                        "expected_doc_ids": ["doc_1"],
                        "failure_mode": "keyword_exact_match",
                    },
                    {
                        "id": "q1",
                        "query": "Test 2",
                        "expected_doc_ids": ["doc_2"],
                        "failure_mode": "keyword_exact_match",
                    },
                ]
            ),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="Duplicate query id"):
            load_eval_queries(qfile)


def test_validate_against_corpus():
    """Verify unknown expected doc IDs raise ValueError."""
    queries = [
        EvalQuery(
            id="q1",
            query="test",
            expected_doc_ids=("doc_known", "doc_unknown"),
            failure_mode="table_lookup",
        )
    ]
    with pytest.raises(ValueError, match="references unknown doc_id"):
        validate_against_corpus(queries, {"doc_known"})

    # Valid when all are known
    validate_against_corpus(queries, {"doc_known", "doc_unknown"})


def test_hit_at_k_and_build_result():
    """Verify hit@1 and hit@k calculations and result construction."""
    expected = ("doc_target",)
    retrieved = ["doc_other", "doc_target", "doc_third"]

    assert not hit_at_k(retrieved, expected, 1)
    assert hit_at_k(retrieved, expected, 2)
    assert hit_at_k(retrieved, expected, 3)

    q = EvalQuery(id="q1", query="Find target", expected_doc_ids=expected, failure_mode="long_context")
    res = build_result(q, retrieved, method="test_method", k=3)
    assert res.hit_at_1 is False
    assert res.hit_at_k is True
    assert res.method == "test_method"


def test_write_results_csv_schema_and_summarize():
    """Verify write_results_csv creates CSV with exact shared fieldnames and summarize aggregates accurately."""
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "results.csv"
        results = [
            EvalResult(
                query_id="q1",
                query="Query 1",
                expected_doc_ids=("doc_1",),
                retrieved_doc_ids=("doc_1", "doc_2"),
                hit_at_1=True,
                hit_at_k=True,
                failure_mode="keyword_exact_match",
                method="test_method",
            ),
            EvalResult(
                query_id="q2",
                query="Query 2",
                expected_doc_ids=("doc_2",),
                retrieved_doc_ids=("doc_1", "doc_2"),
                hit_at_1=False,
                hit_at_k=True,
                failure_mode="keyword_exact_match",
                method="test_method",
            ),
        ]
        write_results_csv(results, csv_path)

        with csv_path.open("r", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            header = next(reader)
            assert header == list(CSV_FIELDNAMES)
            rows = list(reader)
            assert len(rows) == 2

        stats = summarize(results)
        assert stats["count"] == 2
        assert stats["hit_at_1_rate"] == 0.5
        assert stats["hit_at_k_rate"] == 1.0

        by_mode = summarize_by_failure_mode(results)
        assert by_mode["keyword_exact_match"]["count"] == 2
        assert by_mode["keyword_exact_match"]["hit_at_1_rate"] == 0.5
