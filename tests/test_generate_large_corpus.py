"""Unit tests for synthetic large corpus generation in scripts/generate_large_corpus.py."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "generate_large_corpus.py"


def test_generator_determinism_and_word_count():
    """Verify that same seed produces identical files and meets minimum word counts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out1 = Path(tmpdir) / "run1"
        out2 = Path(tmpdir) / "run2"

        cmd1 = [sys.executable, str(SCRIPT_PATH), "--documents", "10", "--seed", "42", "--output-dir", str(out1)]
        subprocess.run(cmd1, check=True, cwd=str(REPO_ROOT))

        cmd2 = [sys.executable, str(SCRIPT_PATH), "--documents", "10", "--seed", "42", "--output-dir", str(out2)]
        subprocess.run(cmd2, check=True, cwd=str(REPO_ROOT))

        # Check manifest
        manifest1 = json.loads((out1 / "manifest.json").read_text(encoding="utf-8"))
        manifest2 = json.loads((out2 / "manifest.json").read_text(encoding="utf-8"))
        assert manifest1 == manifest2
        assert manifest1["documents"] == 10

        # Check eval queries exist and resolve to documents
        queries1 = json.loads((out1 / "eval_queries.json").read_text(encoding="utf-8"))
        queries2 = json.loads((out2 / "eval_queries.json").read_text(encoding="utf-8"))
        assert queries1 == queries2
        assert len(queries1) > 0

        doc_stems = {p.stem for p in (out1 / "corpus").glob("*.md")}
        assert len(doc_stems) == 10

        for q in queries1:
            for exp in q["expected_doc_ids"]:
                assert exp in doc_stems, f"Query expected doc {exp} not in generated corpus {doc_stems}"

        # Check all files are byte-for-byte identical
        for f1 in (out1 / "corpus").glob("*.md"):
            f2 = out2 / "corpus" / f1.name
            assert f2.exists()
            assert f1.read_bytes() == f2.read_bytes()

            # Word count check (minimum 700 words)
            words = f1.read_text(encoding="utf-8").split()
            assert len(words) >= 700, f"{f1.name} has only {len(words)} words (expected >= 700)"


def test_generator_different_seed_produces_different_output():
    """Verify different seeds produce different corpus content."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out1 = Path(tmpdir) / "seed42"
        out2 = Path(tmpdir) / "seed99"

        subprocess.run(
            [sys.executable, str(SCRIPT_PATH), "--documents", "10", "--seed", "42", "--output-dir", str(out1)],
            check=True,
            cwd=str(REPO_ROOT),
        )
        subprocess.run(
            [sys.executable, str(SCRIPT_PATH), "--documents", "10", "--seed", "99", "--output-dir", str(out2)],
            check=True,
            cwd=str(REPO_ROOT),
        )

        manifest1 = (out1 / "manifest.json").read_text(encoding="utf-8")
        manifest2 = (out2 / "manifest.json").read_text(encoding="utf-8")
        assert manifest1 != manifest2


def test_generator_zero_documents_exits_with_error():
    """Verify --documents 0 exits with an error."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "run0"
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH), "--documents", "0", "--output-dir", str(out)],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
