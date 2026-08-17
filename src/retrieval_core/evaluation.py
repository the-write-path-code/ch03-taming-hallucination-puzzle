"""Evaluation query loading, hit@k scoring, and CSV export."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from retrieval_core.types import CSV_FIELDNAMES, EvalQuery, EvalResult

REQUIRED_FIELDS: tuple[str, ...] = (
    "id",
    "query",
    "expected_doc_ids",
    "failure_mode",
)


def load_eval_queries(eval_queries_path: Path) -> list[EvalQuery]:
    """Load and validate eval_queries.json.

    Raises FileNotFoundError if the file is missing, ValueError if the
    JSON is malformed, not a list, empty, has duplicate ids, or any
    record is missing a required field.
    """
    if not eval_queries_path.exists():
        raise FileNotFoundError(f"Evaluation file not found: {eval_queries_path}")

    try:
        raw = json.loads(eval_queries_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {eval_queries_path}: {exc}") from exc

    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{eval_queries_path} must contain a non-empty JSON list.")

    queries: list[EvalQuery] = []
    seen_ids: set[str] = set()
    for index, record in enumerate(raw):
        if not isinstance(record, dict):
            raise ValueError(f"Record at index {index} in {eval_queries_path} is not an object.")
        missing = [field for field in REQUIRED_FIELDS if field not in record]
        if missing:
            raise ValueError(
                f"Record at index {index} in {eval_queries_path} is missing fields: {missing}"
            )
        expected = record["expected_doc_ids"]
        if not isinstance(expected, list) or not expected:
            raise ValueError(
                f"Record {record.get('id', index)!r} must have a non-empty expected_doc_ids list."
            )
        record_id = record["id"]
        if record_id in seen_ids:
            raise ValueError(f"Duplicate query id {record_id!r} in {eval_queries_path}.")
        seen_ids.add(record_id)
        queries.append(
            EvalQuery(
                id=record_id,
                query=record["query"],
                expected_doc_ids=tuple(expected),
                failure_mode=record["failure_mode"],
                notes=record.get("notes", ""),
            )
        )
    return queries


def validate_against_corpus(queries: list[EvalQuery], known_doc_ids: set[str]) -> None:
    """Raise ValueError if any expected_doc_ids reference a doc_id not in the corpus."""
    for eval_query in queries:
        missing = [doc_id for doc_id in eval_query.expected_doc_ids if doc_id not in known_doc_ids]
        if missing:
            raise ValueError(
                f"Query {eval_query.id!r} references unknown doc_id(s): {missing}"
            )


def hit_at_k(retrieved_doc_ids: list[str], expected_doc_ids: tuple[str, ...], k: int) -> bool:
    """True if any expected doc_id appears in the top-k retrieved doc_ids."""
    top_k = retrieved_doc_ids[:k]
    return any(doc_id in top_k for doc_id in expected_doc_ids)


def build_result(
    eval_query: EvalQuery,
    retrieved_doc_ids: list[str],
    method: str,
    k: int,
) -> EvalResult:
    return EvalResult(
        query_id=eval_query.id,
        query=eval_query.query,
        expected_doc_ids=eval_query.expected_doc_ids,
        retrieved_doc_ids=tuple(retrieved_doc_ids),
        hit_at_1=hit_at_k(retrieved_doc_ids, eval_query.expected_doc_ids, 1),
        hit_at_k=hit_at_k(retrieved_doc_ids, eval_query.expected_doc_ids, k),
        failure_mode=eval_query.failure_mode,
        method=method,
    )


def write_results_csv(results: list[EvalResult], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for result in results:
            writer.writerow(result.as_csv_row())


def summarize(results: list[EvalResult]) -> dict[str, float]:
    """Aggregate hit@1 and hit@k rates across all results."""
    if not results:
        return {"hit_at_1_rate": 0.0, "hit_at_k_rate": 0.0, "count": 0}
    count = len(results)
    return {
        "hit_at_1_rate": sum(r.hit_at_1 for r in results) / count,
        "hit_at_k_rate": sum(r.hit_at_k for r in results) / count,
        "count": count,
    }


def summarize_by_failure_mode(results: list[EvalResult]) -> dict[str, dict[str, Any]]:
    """Aggregate hit@1 and hit@k rates per failure mode."""
    by_mode: dict[str, list[EvalResult]] = {}
    for r in results:
        by_mode.setdefault(r.failure_mode, []).append(r)

    summary: dict[str, dict[str, Any]] = {}
    for mode, mode_results in sorted(by_mode.items()):
        count = len(mode_results)
        summary[mode] = {
            "count": count,
            "hit_at_1_rate": sum(r.hit_at_1 for r in mode_results) / count,
            "hit_at_k_rate": sum(r.hit_at_k for r in mode_results) / count,
        }
    return summary


def render_summary_markdown(
    results: list[EvalResult], dataset_name: str, method: str, k: int
) -> str:
    """Render a structured Markdown report of overall and failure-mode retrieval performance."""
    overall = summarize(results)
    by_mode = summarize_by_failure_mode(results)

    lines = [
        f"# Retrieval Benchmark Summary: `{method}`",
        "",
        f"- **Dataset**: `{dataset_name}`",
        f"- **Total Queries**: {overall['count']}",
        f"- **Top-K Parameter**: k={k}",
        f"- **Overall Hit@1**: {overall['hit_at_1_rate']:.1%}",
        f"- **Overall Hit@{k}**: {overall['hit_at_k_rate']:.1%}",
        "",
        "## Performance Breakdown by Failure Mode",
        "",
        f"| Failure Mode | Queries | Hit@1 Rate | Hit@{k} Rate |",
        "| :--- | :---: | :---: | :---: |",
    ]
    for mode, stats in by_mode.items():
        lines.append(
            f"| `{mode}` | {stats['count']} | {stats['hit_at_1_rate']:.1%} | {stats['hit_at_k_rate']:.1%} |"
        )
    lines.append(
        f"| **Overall** | **{overall['count']}** | **{overall['hit_at_1_rate']:.1%}** | **{overall['hit_at_k_rate']:.1%}** |"
    )
    lines.append("")
    return "\n".join(lines)

