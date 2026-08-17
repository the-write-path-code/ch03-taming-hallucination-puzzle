"""Shared dataclasses for corpus documents, evaluation queries, and results."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Document:
    """A single loaded corpus document."""

    doc_id: str
    path: str
    text: str


@dataclass(frozen=True)
class EvalQuery:
    """A single labeled evaluation record."""

    id: str
    query: str
    expected_doc_ids: tuple[str, ...]
    failure_mode: str
    notes: str = ""


@dataclass
class EvalResult:
    """One row of a retrieval evaluation run, ready for CSV export."""

    query_id: str
    query: str
    expected_doc_ids: tuple[str, ...]
    retrieved_doc_ids: tuple[str, ...]
    hit_at_1: bool
    hit_at_k: bool
    failure_mode: str
    method: str

    def as_csv_row(self) -> dict[str, str]:
        return {
            "query_id": self.query_id,
            "query": self.query,
            "expected_doc_ids": "|".join(self.expected_doc_ids),
            "retrieved_doc_ids": "|".join(self.retrieved_doc_ids),
            "hit_at_1": str(int(self.hit_at_1)),
            "hit_at_k": str(int(self.hit_at_k)),
            "failure_mode": self.failure_mode,
            "method": self.method,
        }


CSV_FIELDNAMES: tuple[str, ...] = (
    "query_id",
    "query",
    "expected_doc_ids",
    "retrieved_doc_ids",
    "hit_at_1",
    "hit_at_k",
    "failure_mode",
    "method",
)
