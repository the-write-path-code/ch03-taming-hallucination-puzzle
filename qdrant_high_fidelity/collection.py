"""Collection schema management, metadata parsing, and inspection artifact export."""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient, models

from retrieval_core.types import EvalQuery

DOC_TYPE_PATTERN = re.compile(r"Document\s+type:\s*([a-zA-Z0-9_-]+)", re.IGNORECASE)


def get_collection_name(dataset_name: str, prefix: str = "ch03_retrieval") -> str:
    """Generate a dataset-scoped collection name (e.g. ch03_retrieval_small)."""
    clean_name = re.sub(r"[^a-zA-Z0-9_-]", "_", dataset_name)
    return f"{prefix}_{clean_name}"


def extract_document_type(text: str, filename: str) -> str:
    """Extract document type from Markdown header or fallback to filename prefix."""
    match = DOC_TYPE_PATTERN.search(text)
    if match:
        return match.group(1).lower().strip()

    # Fallback to first segment of filename (e.g. 'policy_0001' -> 'policy')
    stem = Path(filename).stem
    parts = stem.split("_")
    if parts and parts[0]:
        return parts[0].lower()
    return "unknown"


def build_failure_tags_map(queries: list[EvalQuery]) -> dict[str, list[str]]:
    """Build a mapping of doc_id to distinct failure modes from labeled queries."""
    doc_tags: dict[str, set[str]] = {}
    for q in queries:
        for doc_id in q.expected_doc_ids:
            doc_tags.setdefault(doc_id, set()).add(q.failure_mode)
    return {doc_id: sorted(tags) for doc_id, tags in doc_tags.items()}


def generate_point_id(doc_id: str) -> str:
    """Generate a stable, deterministic UUID string from doc_id."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, doc_id))


def build_filter(
    document_type: str | None = None,
    failure_tag: str | None = None,
) -> models.Filter | None:
    """Build a Qdrant Filter for document_type and/or failure_tags."""
    conditions: list[models.FieldCondition] = []
    if document_type:
        conditions.append(
            models.FieldCondition(
                key="document_type",
                match=models.MatchValue(value=document_type.lower()),
            )
        )
    if failure_tag:
        conditions.append(
            models.FieldCondition(
                key="failure_tags",
                match=models.MatchValue(value=failure_tag.lower()),
            )
        )
    if not conditions:
        return None
    return models.Filter(must=conditions)


def ensure_collection(
    client: QdrantClient,
    collection_name: str,
    vector_size: int,
    rebuild: bool = False,
    enable_sparse: bool = False,
    dense_vector_name: str = "dense",
    connection_mode: str = "embedded",
) -> tuple[bool, dict[str, Any]]:
    """Ensure the collection exists with the required schema and indexes.

    Args:
        client: QdrantClient instance.
        collection_name: Name of the collection.
        vector_size: Dimension size of the dense vector.
        rebuild: If True, deletes existing collection and recreates it.
        enable_sparse: If True and supported, configures named sparse vector 'bge_m3_sparse'.
        dense_vector_name: Named dense vector key (default 'dense').
        connection_mode: 'embedded' or 'remote'.

    Returns:
        A tuple of (was_newly_created_or_rebuilt, schema_dict).
    """
    exists = client.collection_exists(collection_name)
    was_rebuilt = False

    if exists and rebuild:
        print(f"Rebuilding collection '{collection_name}': deleting existing collection.")
        client.delete_collection(collection_name)
        exists = False
        was_rebuilt = True

    if not exists:
        print(
            f"Creating collection '{collection_name}' (dense size={vector_size}, "
            f"sparse={enable_sparse})."
        )
        vectors_config = {
            dense_vector_name: models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
            )
        }
        sparse_config = None
        if enable_sparse:
            sparse_config = {"bge_m3_sparse": models.SparseVectorParams()}

        client.create_collection(
            collection_name=collection_name,
            vectors_config=vectors_config,
            sparse_vectors_config=sparse_config,
        )

        # Create payload indexes for filtering
        try:
            client.create_payload_index(
                collection_name=collection_name,
                field_name="document_type",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
            client.create_payload_index(
                collection_name=collection_name,
                field_name="failure_tags",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
        except Exception as exc:
            # Local embedded Qdrant emits a warning; handle gracefully on all client versions
            pass

        was_created = True
    else:
        print(f"Collection '{collection_name}' already exists; reusing without re-creating schema.")
        was_created = False

    schema_info = {
        "collection_name": collection_name,
        "connection_mode": connection_mode,
        "dense_vector": {
            "name": dense_vector_name,
            "size": vector_size,
            "distance": "cosine",
        },
        "sparse_vectors": {"bge_m3_sparse": {"type": "sparse"}} if enable_sparse else None,
        "payload_indexes": ["document_type", "failure_tags"],
        "rebuilt": was_rebuilt or was_created,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    return was_created or was_rebuilt, schema_info


def write_collection_config(schema_info: dict[str, Any], output_path: Path) -> None:
    """Export live collection schema inspection artifact to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(schema_info, indent=2), encoding="utf-8")
    print(f"Wrote collection schema inspection artifact to {output_path}")
