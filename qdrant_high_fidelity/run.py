#!/usr/bin/env python3
"""3.4 High-Fidelity Retrieval in Qdrant.

Demonstrates persistent embedded vector storage, structured metadata filtering,
collection schema management, and optional second-stage ColBERT reranking.

Emits:
- 'qdrant_dense': Named dense vector retrieval in Qdrant (cosine similarity).
- 'qdrant_dense_colbert_rerank': Second-stage token-level ColBERT reranking over top candidates
  (when local-bge-m3 is available).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

# Ensure repository root is on sys.path
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from qdrant_client import models

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
from retrieval_core import (
    add_dataset_argument,
    build_result,
    load_corpus,
    load_eval_queries,
    render_summary_markdown,
    resolve_dataset,
    summarize,
    validate_against_corpus,
    write_results_csv,
)
from retrieval_core.config import (
    DenseProviderConfig,
    QdrantConfig,
    load_dense_provider_config,
    load_qdrant_config,
)
from retrieval_core.types import Document, EvalQuery, EvalResult

OUTPUT_PATH = Path(__file__).parent / "qdrant_results.csv"
CONFIG_OUTPUT_PATH = Path(__file__).parent / "collection_config.json"


def _is_local_url(url: str) -> bool:
    return "localhost" in url or "127.0.0.1" in url


def embed_dense_texts(texts: list[str], config: DenseProviderConfig) -> list[list[float]]:
    """Embed texts using the configured dense provider."""
    if config.provider == "local-bge-m3":
        try:
            from semantic_bridging_bge_m3.provider import BGEM3Provider
        except ImportError:
            try:
                from provider import BGEM3Provider
            except ImportError as exc:
                raise SystemExit(
                    "DENSE_PROVIDER=local-bge-m3 requires the local-bge-m3 extra. "
                    "Install it with: uv sync --extra local-bge-m3"
                ) from exc
        provider = BGEM3Provider(config.local_bge_m3_model_name)
        encoding = provider.encode(texts)
        return encoding.dense_vecs

    if config.provider == "openai":
        if not config.openai_api_key:
            raise SystemExit("DENSE_PROVIDER=openai requires OPENAI_API_KEY in .env.")
        from openai import OpenAI

        client = OpenAI(api_key=config.openai_api_key, base_url=config.openai_base_url or None)
        response = client.embeddings.create(model=config.openai_embed_model, input=texts)
        return [item.embedding for item in response.data]

    # Default to ollama
    import json
    import urllib.error
    import urllib.request

    url = config.ollama_base_url.rstrip("/") + "/api/embed"
    headers = {"Content-Type": "application/json"}
    if config.ollama_api_key:
        headers["Authorization"] = f"Bearer {config.ollama_api_key}"
    elif not _is_local_url(config.ollama_base_url):
        raise SystemExit(
            "DENSE_PROVIDER=ollama is pointed at a non-local OLLAMA_BASE_URL "
            f"({config.ollama_base_url!r}) but OLLAMA_API_KEY is not set in .env."
        )

    batch_size = 16
    all_embeddings: list[list[float]] = []
    for offset in range(0, len(texts), batch_size):
        batch = texts[offset : offset + batch_size]
        payload = json.dumps({"model": config.ollama_embed_model, "input": batch}).encode("utf-8")
        request = urllib.request.Request(url, data=payload, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise SystemExit(
                f"Could not reach Ollama at {config.ollama_base_url!r}: {exc}. "
                "Check that Ollama is running or configure a different provider in .env."
            ) from exc
        embeddings = body.get("embeddings")
        if not embeddings:
            raise SystemExit(f"Ollama response did not include embeddings: {body}")
        all_embeddings.extend(embeddings)
    return all_embeddings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Section 3.4 High-Fidelity Retrieval in Qdrant."
    )
    add_dataset_argument(parser)
    parser.add_argument("--top-k", type=int, default=None, help="Override RETRIEVAL_TOP_K.")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Delete and recreate the Qdrant collection for this dataset.",
    )
    parser.add_argument(
        "--filter-document-type",
        type=str,
        default=None,
        help="Demonstrate metadata filtering by document type (e.g. policy, product, operations).",
    )
    parser.add_argument(
        "--filter-failure-tag",
        type=str,
        default=None,
        help="Demonstrate metadata filtering by failure tag (e.g. table_lookup, keyword_exact_match).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help=f"Where to write results CSV (default: {OUTPUT_PATH}).",
    )
    parser.add_argument(
        "--collection-name",
        type=str,
        default=None,
        help="Override default dataset-scoped collection name.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dense_config = load_dense_provider_config()
    qdrant_config = load_qdrant_config()
    top_k = args.top_k if args.top_k is not None else qdrant_config.top_k

    dataset_paths = resolve_dataset(args.dataset)
    documents = load_corpus(dataset_paths.corpus_dir)
    queries = load_eval_queries(dataset_paths.eval_queries_path)
    validate_against_corpus(queries, {doc.doc_id for doc in documents})

    print(
        f"Loaded {len(documents)} documents and {len(queries)} queries from dataset={args.dataset!r}."
    )

    # 1. Connect to Qdrant
    client, connection_mode = get_qdrant_client(qdrant_config)
    collection_name = args.collection_name or get_collection_name(
        args.dataset, qdrant_config.collection_prefix
    )

    # 2. Check native local BGE-M3 availability for ColBERT reranking
    bge_provider = None
    try:
        from semantic_bridging_bge_m3.provider import BGEM3Provider

        bge_provider = BGEM3Provider(dense_config.local_bge_m3_model_name)
    except Exception:
        print(
            "Local ColBERT reranking unavailable (install with: uv sync --extra local-bge-m3). "
            "Running standard Qdrant dense retrieval."
        )

    # 3. Dense Embeddings for Documents
    print(f"Generating dense embeddings using DENSE_PROVIDER={dense_config.provider!r}...")
    embed_start = time.perf_counter()
    doc_vectors = embed_dense_texts([doc.text for doc in documents], dense_config)
    vector_size = len(doc_vectors[0])
    embed_duration = time.perf_counter() - embed_start
    print(f"Computed {len(doc_vectors)} document embeddings in {embed_duration:.3f}s (dim={vector_size}).")

    # 4. Ensure Collection & Payload Schema
    was_rebuilt, schema_info = ensure_collection(
        client=client,
        collection_name=collection_name,
        vector_size=vector_size,
        rebuild=args.rebuild,
        enable_sparse=False,
        dense_vector_name=qdrant_config.dense_vector_name,
        connection_mode=connection_mode,
    )
    schema_info["dataset"] = args.dataset
    write_collection_config(schema_info, CONFIG_OUTPUT_PATH)

    # 5. Index points if newly created or rebuilt
    if was_rebuilt or client.get_collection(collection_name).points_count == 0:
        failure_tags_map = build_failure_tags_map(queries)
        points: list[models.PointStruct] = []
        for doc, vec in zip(documents, doc_vectors):
            doc_type = extract_document_type(doc.text, doc.path)
            p_id = generate_point_id(doc.doc_id)
            payload = {
                "doc_id": doc.doc_id,
                "source_path": str(doc.path),
                "document_type": doc_type,
                "failure_tags": failure_tags_map.get(doc.doc_id, []),
            }
            points.append(
                models.PointStruct(
                    id=p_id,
                    vector={qdrant_config.dense_vector_name: vec},
                    payload=payload,
                )
            )

        print(f"Indexing {len(points)} documents into Qdrant collection '{collection_name}'...")
        index_start = time.perf_counter()
        client.upsert(collection_name=collection_name, points=points)
        index_duration = time.perf_counter() - index_start
        print(f"Indexed {len(points)} points in {index_duration:.3f}s.")
    else:
        print(f"Collection '{collection_name}' already populated ({client.get_collection(collection_name).points_count} points).")

    # 6. Build filter if demonstration flags supplied
    query_filter = build_filter(
        document_type=args.filter_document_type,
        failure_tag=args.filter_failure_tag,
    )
    if query_filter:
        print(f"Active demonstration filter: doc_type={args.filter_document_type}, tag={args.filter_failure_tag}")

    # 7. Query Embeddings and Retrieval Evaluation
    query_vectors = embed_dense_texts([q.query for q in queries], dense_config)
    candidate_limit = max(top_k, qdrant_config.rerank_candidates)

    all_results: list[EvalResult] = []
    doc_text_by_id = {doc.doc_id: doc.text for doc in documents}

    total_query_time = 0.0
    total_rerank_time = 0.0

    print("Running retrieval evaluation queries...")
    for eval_query, q_vec in zip(queries, query_vectors):
        q_start = time.perf_counter()
        search_res = client.query_points(
            collection_name=collection_name,
            query=q_vec,
            using=qdrant_config.dense_vector_name,
            query_filter=query_filter,
            limit=candidate_limit,
        )
        total_query_time += time.perf_counter() - q_start

        candidate_ids = [pt.payload["doc_id"] for pt in search_res.points]

        # 1. Baseline Qdrant dense result
        all_results.append(
            build_result(eval_query, candidate_ids[:top_k], method="qdrant_dense", k=top_k)
        )

        # 2. Second-stage ColBERT rerank if available
        if bge_provider is not None:
            r_start = time.perf_counter()
            query_colbert = bge_provider.model.encode(
                [eval_query.query],
                return_dense=False,
                return_sparse=False,
                return_colbert_vecs=True,
            )["colbert_vecs"][0]

            candidate_texts = [doc_text_by_id[cid] for cid in candidate_ids]
            candidate_colberts = bge_provider.model.encode(
                candidate_texts,
                return_dense=False,
                return_sparse=False,
                return_colbert_vecs=True,
            )["colbert_vecs"]

            candidate_scores = {
                cid: bge_provider.score_colbert(query_colbert, c_vec)
                for cid, c_vec in zip(candidate_ids, candidate_colberts)
            }
            reranked_ids = sorted(
                candidate_scores.keys(), key=lambda cid: (-candidate_scores[cid], cid)
            )
            total_rerank_time += time.perf_counter() - r_start

            all_results.append(
                build_result(
                    eval_query,
                    reranked_ids[:top_k],
                    method="qdrant_dense_colbert_rerank",
                    k=top_k,
                )
            )

    # 8. Write Results CSV
    write_results_csv(all_results, args.output)

    # Unique methods
    emitted_methods: list[str] = []
    for r in all_results:
        if r.method not in emitted_methods:
            emitted_methods.append(r.method)

    # Summary
    summary_lines = [
        f"# Qdrant High-Fidelity Retrieval Summary: `{collection_name}`",
        "",
        f"- **Dataset**: `{args.dataset}`",
        f"- **Collection Name**: `{collection_name}`",
        f"- **Connection Mode**: `{connection_mode}`",
        f"- **Total Queries**: {len(queries)}",
        f"- **Top-K Parameter**: k={top_k}",
        f"- **Rerank Candidate Pool**: {candidate_limit}",
        f"- **Total Search Time**: {total_query_time:.3f}s (avg {total_query_time / len(queries) * 1000:.1f}ms/query)",
    ]
    if bge_provider is not None:
        summary_lines.append(
            f"- **Total ColBERT Rerank Time**: {total_rerank_time:.3f}s (avg {total_rerank_time / len(queries) * 1000:.1f}ms/query)"
        )
    summary_lines.extend(
        [
            "",
            "## Overall Representation Comparison",
            "",
            f"| Method / Representation | Queries | Hit@1 Rate | Hit@{top_k} Rate |",
            "| :--- | :---: | :---: | :---: |",
        ]
    )

    for method in emitted_methods:
        subset = [r for r in all_results if r.method == method]
        stats = summarize(subset)
        summary_lines.append(
            f"| `{method}` | {stats['count']} | {stats['hit_at_1_rate']:.1%} | {stats['hit_at_k_rate']:.1%} |"
        )
        print(
            f"{method}: hit@1={stats['hit_at_1_rate']:.2f} "
            f"hit@{top_k}={stats['hit_at_k_rate']:.2f} (n={stats['count']})"
        )

    summary_lines.append("")
    summary_lines.append("## Performance Breakdown by Failure Mode")
    summary_lines.append("")

    from retrieval_core.evaluation import summarize_by_failure_mode

    for method in emitted_methods:
        subset = [r for r in all_results if r.method == method]
        by_mode = summarize_by_failure_mode(subset)
        summary_lines.append(f"### `{method}`")
        summary_lines.append("")
        summary_lines.append(f"| Failure Mode | Queries | Hit@1 Rate | Hit@{top_k} Rate |")
        summary_lines.append("| :--- | :---: | :---: | :---: |")
        for mode, stats in by_mode.items():
            summary_lines.append(
                f"| `{mode}` | {stats['count']} | {stats['hit_at_1_rate']:.1%} | {stats['hit_at_k_rate']:.1%} |"
            )
        summary_lines.append("")

    summary_path = args.output.with_suffix(".md")
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
    print(f"Wrote {len(all_results)} rows to {args.output} and summary to {summary_path}.")


if __name__ == "__main__":
    main()
