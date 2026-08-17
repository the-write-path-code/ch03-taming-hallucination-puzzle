"""Unit tests for corpus loading in retrieval_core/corpus.py."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from retrieval_core.corpus import index_by_id, load_corpus


def test_load_corpus_recursive():
    """Verify load_corpus loads markdown files recursively and extracts doc_ids from stems."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "sub").mkdir()

        (root / "doc_1.md").write_text("# Doc 1 content", encoding="utf-8")
        (root / "sub" / "doc_2.md").write_text("# Doc 2 content", encoding="utf-8")
        (root / "ignored.txt").write_text("Ignored text", encoding="utf-8")

        docs = load_corpus(root)
        doc_map = index_by_id(docs)

        assert len(docs) == 2
        assert "doc_1" in doc_map
        assert "doc_2" in doc_map
        assert doc_map["doc_1"].text == "# Doc 1 content"
        assert doc_map["doc_2"].text == "# Doc 2 content"


def test_load_corpus_duplicate_stem_raises():
    """Verify duplicate stems in different directories raise ValueError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "sub").mkdir()

        (root / "doc_alpha.md").write_text("Root alpha", encoding="utf-8")
        (root / "sub" / "doc_alpha.md").write_text("Sub alpha", encoding="utf-8")

        with pytest.raises(ValueError, match="Duplicate doc_id 'doc_alpha'"):
            load_corpus(root)


def test_load_corpus_missing_directory_raises():
    """Verify missing corpus directory raises FileNotFoundError."""
    missing_dir = Path("/path/does/not/exist/for/sure/corpus")
    with pytest.raises(FileNotFoundError, match="Corpus directory not found"):
        load_corpus(missing_dir)
