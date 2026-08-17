"""Unit tests for Stage 3.3 Hybrid Retrieval fusion algorithms and BM25 tokenization."""

from __future__ import annotations

import pytest

from hybrid_retrieval.fusion import reciprocal_rank_fusion, relative_score_fusion
from hybrid_retrieval.run import tokenize_bm25
from retrieval_core.config import HybridConfig, load_hybrid_config


def test_rrf_consensus_ranking():
    """Verify that a document ranked moderately well across multiple lists outranks a single-list leader."""
    list1 = ["doc_a", "doc_b", "doc_c"]
    list2 = ["doc_d", "doc_b", "doc_a"]

    # doc_b is rank 1 in list1 (score 1/62) and rank 1 in list2 (score 1/62) -> total 2/62 = 0.032258
    # doc_a is rank 0 in list1 (1/61) and rank 2 in list2 (1/63) -> 1/61 + 1/63 = 0.016393 + 0.015873 = 0.032266
    # doc_d is rank 0 in list2 (1/61) -> 0.016393
    scores = reciprocal_rank_fusion([list1, list2], rrf_k=60)
    assert scores["doc_a"] > scores["doc_d"]
    assert scores["doc_b"] > scores["doc_d"]
    assert "doc_c" in scores


def test_rrf_duplicate_handling():
    """Verify duplicate document IDs within the same list are counted only once at their first rank."""
    list_with_dups = ["doc_a", "doc_b", "doc_a", "doc_c", "doc_b"]
    scores = reciprocal_rank_fusion([list_with_dups], rrf_k=60)

    # doc_a should only get rank 0 score: 1 / 61
    assert pytest.approx(scores["doc_a"], rel=1e-6) == 1.0 / 61.0
    # doc_b should only get rank 1 score: 1 / 62
    assert pytest.approx(scores["doc_b"], rel=1e-6) == 1.0 / 62.0
    # doc_c should get rank 2 score: 1 / 63
    assert pytest.approx(scores["doc_c"], rel=1e-6) == 1.0 / 63.0


def test_rrf_invalid_k_raises():
    """Verify negative rrf_k raises ValueError."""
    with pytest.raises(ValueError, match="rrf_k must be non-negative"):
        reciprocal_rank_fusion([["doc_a"]], rrf_k=-1)


def test_relative_score_fusion_equal_weights():
    """Verify relative score fusion with default equal weights."""
    map1 = {"doc_a": 10.0, "doc_b": 20.0}  # normalized: doc_a=0.0, doc_b=1.0
    map2 = {"doc_a": 100.0, "doc_b": 50.0}  # normalized: doc_a=1.0, doc_b=0.0

    scores = relative_score_fusion([map1, map2])
    # Equal weights 0.5 each -> doc_a: 0.5*0 + 0.5*1 = 0.5, doc_b: 0.5*1 + 0.5*0 = 0.5
    assert pytest.approx(scores["doc_a"]) == 0.5
    assert pytest.approx(scores["doc_b"]) == 0.5


def test_relative_score_fusion_custom_weights():
    """Verify relative score fusion with custom weights."""
    map1 = {"doc_a": 0.0, "doc_b": 10.0}  # norm: doc_a=0, doc_b=1
    map2 = {"doc_a": 50.0, "doc_b": 100.0}  # norm: doc_a=0, doc_b=1

    # Weights [0.8, 0.2]
    scores = relative_score_fusion([map1, map2], weights=[0.8, 0.2])
    assert pytest.approx(scores["doc_a"]) == 0.0
    assert pytest.approx(scores["doc_b"]) == 1.0

    # Weights [3.0, 1.0] -> normalized to [0.75, 0.25]
    scores_unnormalized_weights = relative_score_fusion([map1, map2], weights=[3.0, 1.0])
    assert pytest.approx(scores_unnormalized_weights["doc_a"]) == 0.0
    assert pytest.approx(scores_unnormalized_weights["doc_b"]) == 1.0


def test_relative_score_fusion_constant_score_no_zero_division():
    """Verify constant score map contributes 1.0 without ZeroDivisionError."""
    map_const = {"doc_a": 5.0, "doc_b": 5.0}
    map_varying = {"doc_a": 1.0, "doc_b": 3.0}  # norm: doc_a=0, doc_b=1

    scores = relative_score_fusion([map_const, map_varying], weights=[0.5, 0.5])
    # doc_a: 0.5*1.0 + 0.5*0.0 = 0.5
    # doc_b: 0.5*1.0 + 0.5*1.0 = 1.0
    assert pytest.approx(scores["doc_a"]) == 0.5
    assert pytest.approx(scores["doc_b"]) == 1.0


def test_relative_score_fusion_missing_docs():
    """Verify missing document in a map contributes 0.0 for that source."""
    map1 = {"doc_a": 10.0, "doc_b": 20.0}  # norm: a=0, b=1
    map2 = {"doc_c": 5.0, "doc_b": 15.0}  # norm: c=0, b=1

    scores = relative_score_fusion([map1, map2], weights=[0.5, 0.5])
    assert pytest.approx(scores["doc_a"]) == 0.0  # map1: 0, map2: 0 (missing)
    assert pytest.approx(scores["doc_c"]) == 0.0  # map1: 0 (missing), map2: 0
    assert pytest.approx(scores["doc_b"]) == 1.0  # map1: 0.5, map2: 0.5


def test_relative_score_fusion_invalid_weights_raises():
    """Verify invalid weights raise ValueError."""
    # Empty score_maps
    with pytest.raises(ValueError, match="score_maps must not be empty"):
        relative_score_fusion([])

    # Weight length mismatch
    with pytest.raises(ValueError, match="Length of weights"):
        relative_score_fusion([{"doc_a": 1.0}], weights=[0.5, 0.5])

    # Negative weight
    with pytest.raises(ValueError, match="Weights must be non-negative"):
        relative_score_fusion([{"doc_a": 1.0}], weights=[-0.5])

    # All-zero total weight
    with pytest.raises(ValueError, match="Total weight sum must be positive"):
        relative_score_fusion([{"doc_a": 1.0}, {"doc_b": 1.0}], weights=[0.0, 0.0])


def test_bm25_tokenizer_preserves_identifiers():
    """Verify deterministic BM25 tokenizer preserves identifiers like AX4-E117, CC-17, NSN-42, HRD-9C03."""
    text = (
        "During incident INC-P1, check alert AX4-E117 on Atlas Gateway, "
        "policy CC-17, sensor NSN-42, and reader HRD-9C03."
    )
    tokens = tokenize_bm25(text)

    expected_identifiers = ["inc-p1", "ax4-e117", "cc-17", "nsn-42", "hrd-9c03"]
    for ident in expected_identifiers:
        assert ident in tokens, f"Expected identifier '{ident}' was fragmented in {tokens}"


def test_hybrid_config_validation(monkeypatch):
    """Verify HybridConfig validation in src/retrieval_core/config.py."""
    monkeypatch.setenv("HYBRID_SPARSE_METHOD", "invalid_sparse")
    with pytest.raises(ValueError, match="HYBRID_SPARSE_METHOD"):
        load_hybrid_config()

    monkeypatch.setenv("HYBRID_SPARSE_METHOD", "bm25")
    monkeypatch.setenv("HYBRID_FUSION_METHOD", "invalid_fusion")
    with pytest.raises(ValueError, match="HYBRID_FUSION_METHOD"):
        load_hybrid_config()

    monkeypatch.setenv("HYBRID_FUSION_METHOD", "rrf")
    monkeypatch.setenv("RRF_K", "-5")
    with pytest.raises(ValueError, match="RRF_K"):
        load_hybrid_config()

    monkeypatch.setenv("RRF_K", "60")
    monkeypatch.setenv("HYBRID_DENSE_WEIGHT", "-0.1")
    with pytest.raises(ValueError, match="HYBRID_DENSE_WEIGHT"):
        load_hybrid_config()

    monkeypatch.setenv("HYBRID_DENSE_WEIGHT", "0.0")
    monkeypatch.setenv("HYBRID_SPARSE_WEIGHT", "0.0")
    with pytest.raises(ValueError, match="cannot sum to zero"):
        load_hybrid_config()
