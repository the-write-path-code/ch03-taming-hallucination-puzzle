"""Unified Qdrant connection factory.

Supports persisted embedded Qdrant (default), Docker-hosted Qdrant, and
Qdrant Cloud through a single connection abstraction. Application retrieval
logic never branches on deployment mode.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from qdrant_client import QdrantClient

from retrieval_core.config import QdrantConfig, load_qdrant_config

if TYPE_CHECKING:
    pass


def get_qdrant_client(config: QdrantConfig | None = None) -> tuple[QdrantClient, str]:
    """Create and return a QdrantClient instance and its connection mode string.

    Args:
        config: Optional QdrantConfig. If omitted, loaded from environment via load_qdrant_config().

    Returns:
        A tuple of (client_instance, mode_str) where mode_str is 'embedded' or 'remote'.
    """
    if config is None:
        config = load_qdrant_config()

    if config.url:
        # Remote Qdrant (Docker or Qdrant Cloud)
        mode = "remote"
        print(f"Connecting to remote Qdrant at: {config.url}")
        client = QdrantClient(url=config.url, api_key=config.api_key or None)
        return client, mode

    # Default: persisted embedded local Qdrant
    mode = "embedded"
    storage_path = Path(config.local_path).resolve()
    print(f"Using persisted embedded Qdrant storage at: {storage_path}")
    client = QdrantClient(path=str(storage_path))
    return client, mode
