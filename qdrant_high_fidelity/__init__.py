"""Stage 3.4 High-Fidelity Retrieval in Qdrant."""

from qdrant_high_fidelity.client import get_qdrant_client
from qdrant_high_fidelity.collection import (
    build_failure_tags_map,
    build_filter,
    ensure_collection,
    extract_document_type,
    generate_point_id,
    get_collection_name,
    write_collection_config,
)

__all__ = [
    "get_qdrant_client",
    "get_collection_name",
    "extract_document_type",
    "build_failure_tags_map",
    "generate_point_id",
    "build_filter",
    "ensure_collection",
    "write_collection_config",
]
