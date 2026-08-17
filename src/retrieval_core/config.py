"""Centralized environment configuration.

Every user-tunable setting for the retrieval stages lives here and is
sourced from a single `.env` file at the project root via python-dotenv.
Stages should import from this module instead of reading `os.environ`
directly, so there is exactly one place to look for configuration.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

from retrieval_core.paths import PROJECT_ROOT

load_dotenv(PROJECT_ROOT / ".env")

DENSE_PROVIDER_CHOICES: tuple[str, ...] = ("ollama", "openai", "local-bge-m3")
HYBRID_SPARSE_CHOICES: tuple[str, ...] = ("bm25", "bge_m3_sparse")
HYBRID_FUSION_CHOICES: tuple[str, ...] = ("rrf", "relative_score")


def _get(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    return value if value not in (None, "") else default


def _get_int(name: str, default: int) -> int:
    value = _get(name)
    return int(value) if value is not None else default


def _get_float(name: str, default: float) -> float:
    value = _get(name)
    return float(value) if value is not None else default


@dataclass(frozen=True)
class DenseProviderConfig:
    """All settings needed to embed text with the configured dense provider."""

    provider: str
    top_k: int

    ollama_base_url: str
    ollama_api_key: str | None
    ollama_embed_model: str

    openai_base_url: str | None
    openai_api_key: str | None
    openai_embed_model: str

    local_bge_m3_model_name: str


@dataclass(frozen=True)
class HybridConfig:
    """All settings needed for Stage 3.3 hybrid retrieval."""

    sparse_method: str
    fusion_method: str
    rrf_k: int
    dense_weight: float
    sparse_weight: float
    top_k: int


def load_dense_provider_config() -> DenseProviderConfig:
    """Load dense-embedding provider settings from the environment.

    DENSE_PROVIDER selects one of "ollama", "openai", or "local-bge-m3".
    All other settings are optional and fall back to sensible defaults.
    """
    provider = _get("DENSE_PROVIDER", "ollama")
    if provider not in DENSE_PROVIDER_CHOICES:
        raise ValueError(
            f"DENSE_PROVIDER={provider!r} is not supported. "
            f"Choose one of: {', '.join(DENSE_PROVIDER_CHOICES)}."
        )
    return DenseProviderConfig(
        provider=provider,
        top_k=_get_int("RETRIEVAL_TOP_K", 3),
        ollama_base_url=_get("OLLAMA_BASE_URL", "http://localhost:11434"),
        ollama_api_key=_get("OLLAMA_API_KEY"),
        ollama_embed_model=_get("OLLAMA_EMBED_MODEL", "bge-m3"),
        openai_base_url=_get("OPENAI_BASE_URL"),
        openai_api_key=_get("OPENAI_API_KEY"),
        openai_embed_model=_get("OPENAI_EMBED_MODEL", "text-embedding-3-small"),
        local_bge_m3_model_name=_get("LOCAL_BGE_M3_MODEL_NAME", "BAAI/bge-m3"),
    )


def load_hybrid_config() -> HybridConfig:
    """Load Section 3.3 hybrid retrieval settings from the environment.

    Validates sparse/fusion method choices, non-negative RRF_K, and valid
    non-negative fusion weights that do not sum to zero.
    """
    sparse_method = _get("HYBRID_SPARSE_METHOD", "bm25")
    if sparse_method not in HYBRID_SPARSE_CHOICES:
        raise ValueError(
            f"HYBRID_SPARSE_METHOD={sparse_method!r} is not supported. "
            f"Choose one of: {', '.join(HYBRID_SPARSE_CHOICES)}."
        )

    fusion_method = _get("HYBRID_FUSION_METHOD", "rrf")
    if fusion_method not in HYBRID_FUSION_CHOICES:
        raise ValueError(
            f"HYBRID_FUSION_METHOD={fusion_method!r} is not supported. "
            f"Choose one of: {', '.join(HYBRID_FUSION_CHOICES)}."
        )

    rrf_k = _get_int("RRF_K", 60)
    if rrf_k < 0:
        raise ValueError(f"RRF_K must be non-negative, got {rrf_k}.")

    dense_weight = _get_float("HYBRID_DENSE_WEIGHT", 0.5)
    if dense_weight < 0.0:
        raise ValueError(f"HYBRID_DENSE_WEIGHT must be non-negative, got {dense_weight}.")

    sparse_weight = _get_float("HYBRID_SPARSE_WEIGHT", 0.5)
    if sparse_weight < 0.0:
        raise ValueError(f"HYBRID_SPARSE_WEIGHT must be non-negative, got {sparse_weight}.")

    if dense_weight + sparse_weight <= 0.0:
        raise ValueError("HYBRID_DENSE_WEIGHT and HYBRID_SPARSE_WEIGHT cannot sum to zero or less.")

    top_k = _get_int("RETRIEVAL_TOP_K", 3)
    if top_k < 1:
        raise ValueError(f"RETRIEVAL_TOP_K must be at least 1, got {top_k}.")

    return HybridConfig(
        sparse_method=sparse_method,
        fusion_method=fusion_method,
        rrf_k=rrf_k,
        dense_weight=dense_weight,
        sparse_weight=sparse_weight,
        top_k=top_k,
    )

