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


def _get(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    return value if value not in (None, "") else default


def _get_int(name: str, default: int) -> int:
    value = _get(name)
    return int(value) if value is not None else default


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
