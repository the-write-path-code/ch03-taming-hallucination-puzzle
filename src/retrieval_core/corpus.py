"""Corpus loading utilities.

Documents are plain Markdown files. The document ID is derived from the
file stem (filename without extension), matching the doc_id convention
used by sample_data/small/eval_queries.json and the generated_large
manifest.
"""

from __future__ import annotations

from pathlib import Path

from retrieval_core.types import Document


def load_corpus(corpus_dir: Path) -> list[Document]:
    """Recursively load all Markdown documents under corpus_dir.

    Raises FileNotFoundError if the directory does not exist, and
    ValueError if two files under different subdirectories would
    resolve to the same doc_id.
    """
    if not corpus_dir.exists():
        raise FileNotFoundError(f"Corpus directory not found: {corpus_dir}")

    documents: list[Document] = []
    seen_ids: dict[str, Path] = {}
    for md_path in sorted(corpus_dir.rglob("*.md")):
        doc_id = md_path.stem
        if doc_id in seen_ids:
            raise ValueError(
                f"Duplicate doc_id {doc_id!r} from {md_path} and {seen_ids[doc_id]}"
            )
        seen_ids[doc_id] = md_path
        text = md_path.read_text(encoding="utf-8")
        documents.append(
            Document(doc_id=doc_id, path=str(md_path.relative_to(corpus_dir.parent)), text=text)
        )
    return documents


def index_by_id(documents: list[Document]) -> dict[str, Document]:
    """Build a doc_id -> Document lookup, asserting uniqueness."""
    index: dict[str, Document] = {}
    for doc in documents:
        if doc.doc_id in index:
            raise ValueError(f"Duplicate doc_id encountered during indexing: {doc.doc_id!r}")
        index[doc.doc_id] = doc
    return index
