"""Unit tests for Stage 3.4 High-Fidelity Retrieval in Qdrant."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from qdrant_client import models

from qdrant_high_fidelity.client import get_qdrant_client
from qdrant_high_fidelity.collection import (
    build_failure_tags_map,
    build_filter,
    extract_document_type,
    generate_point_id,
    get_collection_name,
)
from retrieval_core.config import QdrantConfig, load_qdrant_config
from retrieval_core.types import EvalQuery


def test_deterministic_point_id():
    """Verify generate_point_id produces stable UUID5 strings for identical doc_ids."""
    id1 = generate_point_id("doc_alpha")
    id2 = generate_point_id("doc_alpha")
    id3 = generate_point_id("doc_beta")

    assert id1 == id2
    assert id1 != id3
    assert len(id1) == 36  # Standard UUID string representation length


def test_extract_document_type_from_header():
    """Verify document_type is parsed accurately from Markdown header."""
    text_with_header = (
        "# Nimbus System Specification\n"
        "Document type: specification\n"
        "This document describes the optical sensors."
    )
    doc_type = extract_document_type(text_with_header, "spec_001.md")
    assert doc_type == "specification"


def test_extract_document_type_filename_fallback():
    """Verify document_type falls back to filename stem prefix when header is missing."""
    text_no_header = "# Overview\nNo explicit type header in this body."
    doc_type = extract_document_type(text_no_header, "policy_incident_severity.md")
    assert doc_type == "policy"

    doc_type_prod = extract_document_type(text_no_header, "product_atlas_gateway.md")
    assert doc_type_prod == "product"


def test_build_failure_tags_map():
    """Verify failure tags map groups failure modes by expected doc IDs."""
    queries = [
        EvalQuery(
            id="q1",
            query="Find code",
            expected_doc_ids=("doc_a", "doc_b"),
            failure_mode="keyword_exact_match",
        ),
        EvalQuery(
            id="q2",
            query="Lookup table",
            expected_doc_ids=("doc_a",),
            failure_mode="table_lookup",
        ),
    ]
    tags_map = build_failure_tags_map(queries)
    assert tags_map["doc_a"] == ["keyword_exact_match", "table_lookup"]
    assert tags_map["doc_b"] == ["keyword_exact_match"]


def test_build_filter_combinations():
    """Verify Qdrant Filter generation for document_type and failure_tags."""
    # None when no args
    assert build_filter() is None

    # Filter with document_type only
    flt_type = build_filter(document_type="policy")
    assert isinstance(flt_type, models.Filter)
    assert len(flt_type.must) == 1
    assert flt_type.must[0].key == "document_type"
    assert flt_type.must[0].match.value == "policy"

    # Filter with both
    flt_both = build_filter(document_type="product", failure_tag="table_lookup")
    assert isinstance(flt_both, models.Filter)
    assert len(flt_both.must) == 2


def test_get_collection_name():
    """Verify dataset-scoped collection name format."""
    name = get_collection_name("small", "ch03_retrieval")
    assert name == "ch03_retrieval_small"

    name_large = get_collection_name("generated_large", "ch03_retrieval")
    assert name_large == "ch03_retrieval_generated_large"


def test_get_qdrant_client_mode_selection():
    """Verify client constructor routes to embedded vs remote without starting a live server."""
    # Embedded local mode
    config_local = QdrantConfig(
        local_path="./.qdrant_local",
        url=None,
        api_key=None,
        collection_prefix="ch03_retrieval",
        dense_vector_name="dense",
        rerank_candidates=20,
        top_k=3,
    )
    with patch("qdrant_high_fidelity.client.QdrantClient") as mock_client:
        client, mode = get_qdrant_client(config_local)
        assert mode == "embedded"
        mock_client.assert_called_once()
        call_kwargs = mock_client.call_args[1]
        assert "path" in call_kwargs

    # Remote mode
    config_remote = QdrantConfig(
        local_path="./.qdrant_local",
        url="http://localhost:6333",
        api_key="secret-key",
        collection_prefix="ch03_retrieval",
        dense_vector_name="dense",
        rerank_candidates=20,
        top_k=3,
    )
    with patch("qdrant_high_fidelity.client.QdrantClient") as mock_client:
        client, mode = get_qdrant_client(config_remote)
        assert mode == "remote"
        mock_client.assert_called_once()
        call_kwargs = mock_client.call_args[1]
        assert call_kwargs.get("url") == "http://localhost:6333"
        assert call_kwargs.get("api_key") == "secret-key"


def test_qdrant_config_validation(monkeypatch):
    """Verify QdrantConfig validation rules."""
    monkeypatch.setenv("QDRANT_RERANK_CANDIDATES", "0")
    with pytest.raises(ValueError, match="QDRANT_RERANK_CANDIDATES"):
        load_qdrant_config()

    monkeypatch.setenv("QDRANT_RERANK_CANDIDATES", "20")
    monkeypatch.setenv("RETRIEVAL_TOP_K", "0")
    with pytest.raises(ValueError, match="RETRIEVAL_TOP_K"):
        load_qdrant_config()
