#!/usr/bin/env python3
"""Stage 3.1: Naive Dense-Only Baseline.

Embeds documents and eval queries, ranks documents by cosine
similarity, and writes results to results/<dataset>/naive_dense_only.csv.
"""

from __future__ import annotations

import argparse
import json
import math
import urllib.error
import urllib.request
from pathlib import Path

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
from retrieval_core.config import DenseProviderConfig, load_dense_provider_config

_repo_root = Path(__file__).parent.parent


def _is_local_url(url: str) -> bool:
    return "localhost" in url or "127.0.0.1" in url


def embed_with_ollama(texts: list[str], config: DenseProviderConfig) -> list[list[float]]:
    url = config.ollama_base_url.rstrip("/") + "/api/embed"
    headers = {"Content-Type": "application/json"}
    if config.ollama_api_key:
        headers["Authorization"] = f"Bearer {config.ollama_api_key}"
    elif not _is_local_url(config.ollama_base_url):
        raise SystemExit(
            "DENSE_PROVIDER=ollama is pointed at a non-local OLLAMA_BASE_URL "
            f"({config.ollama_base_url!r}) but OLLAMA_API_KEY is not set in .env. "
            "Set OLLAMA_API_KEY for Ollama Cloud, or point OLLAMA_BASE_URL at a "
            "local Ollama instance (default http://localhost:11434)."
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
                "Check OLLAMA_BASE_URL and that Ollama is running, or use a different "
                "DENSE_PROVIDER in .env."
            ) from exc
        embeddings = body.get("embeddings")
        if not embeddings:
            raise SystemExit(f"Ollama response did not include embeddings: {body}")
        all_embeddings.extend(embeddings)
    return all_embeddings


def embed_with_openai(texts: list[str], config: DenseProviderConfig) -> list[list[float]]:
    if not config.openai_api_key:
        raise SystemExit(
            "DENSE_PROVIDER=openai requires OPENAI_API_KEY to be set in .env."
        )
    from openai import OpenAI

    client = OpenAI(api_key=config.openai_api_key, base_url=config.openai_base_url or None)
    response = client.embeddings.create(model=config.openai_embed_model, input=texts)
    return [item.embedding for item in response.data]


def embed_with_local_bge_m3(texts: list[str], config: DenseProviderConfig) -> list[list[float]]:
    try:
        from FlagEmbedding import BGEM3FlagModel
    except ImportError as exc:
        raise SystemExit(
            "DENSE_PROVIDER=local-bge-m3 requires the local-bge-m3 extra. "
            "Install it with: uv sync --extra local-bge-m3"
        ) from exc
    model = BGEM3FlagModel(config.local_bge_m3_model_name, use_fp16=False)
    output = model.encode(
        texts, return_dense=True, return_sparse=False, return_colbert_vecs=False
    )
    return [vector.tolist() for vector in output["dense_vecs"]]


PROVIDERS = {
    "ollama": embed_with_ollama,
    "openai": embed_with_openai,
    "local-bge-m3": embed_with_local_bge_m3,
}


def embed_texts(texts: list[str], config: DenseProviderConfig) -> list[list[float]]:
    provider_fn = PROVIDERS[config.provider]
    return provider_fn(texts, config)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def rank_documents(
    query_vector: list[float], doc_vectors: dict[str, list[float]], top_k: int
) -> list[str]:
    scored = [
        (cosine_similarity(query_vector, vector), doc_id)
        for doc_id, vector in doc_vectors.items()
    ]
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [doc_id for _, doc_id in scored[:top_k]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the 3.1 naive dense baseline.")
    add_dataset_argument(parser)
    parser.add_argument("--top-k", type=int, default=None, help="Override RETRIEVAL_TOP_K.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Where to write results CSV (default: results/<dataset>/naive_dense_only.csv).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_dense_provider_config()
    top_k = args.top_k if args.top_k is not None else config.top_k

    dataset_paths = resolve_dataset(args.dataset)
    output_path = (
        args.output
        if args.output is not None
        else _repo_root / "results" / args.dataset / "naive_dense_only.csv"
    )

    documents = load_corpus(dataset_paths.corpus_dir)
    queries = load_eval_queries(dataset_paths.eval_queries_path)
    validate_against_corpus(queries, {doc.doc_id for doc in documents})

    print(
        f"Loaded {len(documents)} documents and {len(queries)} queries "
        f"from dataset={args.dataset!r} using DENSE_PROVIDER={config.provider!r}."
    )

    doc_ids = [doc.doc_id for doc in documents]
    doc_vectors_list = embed_texts([doc.text for doc in documents], config)
    doc_vectors = dict(zip(doc_ids, doc_vectors_list))

    query_vectors = embed_texts([q.query for q in queries], config)

    method = f"dense_only:{config.provider}"
    results = []
    for eval_query, query_vector in zip(queries, query_vectors):
        retrieved = rank_documents(query_vector, doc_vectors, top_k)
        results.append(build_result(eval_query, retrieved, method=method, k=top_k))

    write_results_csv(results, output_path)
    stats = summarize(results)

    print(
        f"{method}: hit@1={stats['hit_at_1_rate']:.2f} "
        f"hit@{top_k}={stats['hit_at_k_rate']:.2f} (n={stats['count']})"
    )
    print(f"Wrote {len(results)} rows to {output_path}.")


if __name__ == "__main__":
    main()
