#!/usr/bin/env python3
"""3.3 Hybrid retrieval with dense and sparse search.

Demonstrates independent dense and sparse retrieval legs, then compares
Reciprocal Rank Fusion (RRF) and Relative Score Fusion (RSF) against
standalone dense and sparse baselines.

Supports:
- Sparse leg: BM25 (via rank_bm25 with deterministic identifier-preserving tokenizer)
  or native BGE-M3 sparse lexical weights (via local-bge-m3).
- Dense leg: Configured dense provider (Ollama, OpenAI, or local BGE-M3).
- Fusion: RRF (rank-based) and Relative Score Fusion (normalized score weighted sum).

Writes all compared methods to fusion_comparison.csv and emits a companion summary.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path
from typing import Any

# Ensure repository root is on sys.path
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from rank_bm25 import BM25Okapi

from hybrid_retrieval.fusion import reciprocal_rank_fusion, relative_score_fusion
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
    HybridConfig,
    load_dense_provider_config,
    load_hybrid_config,
)
from retrieval_core.types import Document, EvalQuery, EvalResult

OUTPUT_PATH = Path(__file__).parent / "fusion_comparison.csv"

# Regex pattern for deterministic tokenization preserving identifiers like AX4-E117, CC-17, NSN-42, HRD-9C03
TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9]+(?:[-_][a-zA-Z0-9]+)*")


def tokenize_bm25(text: str) -> list[str]:
    """Tokenize text into lowercase lexical tokens while preserving identifiers.

    Preserves compound tokens with hyphens or underscores (e.g. AX4-E117, CC-17)
    as single cohesive tokens.
    """
    return TOKEN_PATTERN.findall(text.lower())


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


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def rank_from_scores(scores: dict[str, float], top_k: int) -> list[str]:
    """Sort documents by score descending, with deterministic doc_id tie-breaking."""
    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return [doc_id for doc_id, _ in ranked[:top_k]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Section 3.3 Hybrid Retrieval with dense and sparse search."
    )
    add_dataset_argument(parser)
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Override RETRIEVAL_TOP_K from .env.",
    )
    parser.add_argument(
        "--sparse-method",
        choices=["bm25", "bge_m3_sparse"],
        default=None,
        help="Override HYBRID_SPARSE_METHOD (bm25 or bge_m3_sparse).",
    )
    parser.add_argument(
        "--fusion-method",
        choices=["rrf", "relative_score"],
        default=None,
        help="Override HYBRID_FUSION_METHOD (rrf or relative_score).",
    )
    parser.add_argument(
        "--rrf-k",
        type=int,
        default=None,
        help="Override RRF_K parameter for Reciprocal Rank Fusion.",
    )
    parser.add_argument(
        "--dense-weight",
        type=float,
        default=None,
        help="Override HYBRID_DENSE_WEIGHT for Relative Score Fusion.",
    )
    parser.add_argument(
        "--sparse-weight",
        type=float,
        default=None,
        help="Override HYBRID_SPARSE_WEIGHT for Relative Score Fusion.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help=f"Where to write comparison CSV (default: {OUTPUT_PATH}).",
    )
    parser.add_argument(
        "--include-bm25",
        action="store_true",
        help="In bge_m3_sparse mode, also include BM25 comparison rows.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dense_config = load_dense_provider_config()
    hybrid_config = load_hybrid_config()

    top_k = args.top_k if args.top_k is not None else hybrid_config.top_k
    sparse_method = args.sparse_method or hybrid_config.sparse_method
    rrf_k = args.rrf_k if args.rrf_k is not None else hybrid_config.rrf_k
    dense_weight = args.dense_weight if args.dense_weight is not None else hybrid_config.dense_weight
    sparse_weight = args.sparse_weight if args.sparse_weight is not None else hybrid_config.sparse_weight

    dataset_paths = resolve_dataset(args.dataset)
    documents = load_corpus(dataset_paths.corpus_dir)
    queries = load_eval_queries(dataset_paths.eval_queries_path)
    validate_against_corpus(queries, {doc.doc_id for doc in documents})

    print(
        f"Loaded {len(documents)} documents and {len(queries)} queries "
        f"from dataset={args.dataset!r} (sparse_method={sparse_method!r})."
    )

    doc_ids = [doc.doc_id for doc in documents]
    all_results: list[EvalResult] = []

    if sparse_method == "bge_m3_sparse":
        # Native BGE-M3 local execution for both dense and sparse representations
        try:
            from semantic_bridging_bge_m3.provider import BGEM3Provider
        except ImportError:
            try:
                from provider import BGEM3Provider
            except ImportError as exc:
                raise SystemExit(
                    "HYBRID_SPARSE_METHOD=bge_m3_sparse requires the local-bge-m3 extra. "
                    "Install it with: uv sync --extra local-bge-m3"
                ) from exc

        provider = BGEM3Provider(dense_config.local_bge_m3_model_name)
        doc_encoding = provider.encode([doc.text for doc in documents])
        query_encoding = provider.encode([q.query for q in queries])

        # Optional BM25 indexing if --include-bm25 requested
        bm25_model = None
        if args.include_bm25:
            tokenized_corpus = [tokenize_bm25(doc.text) for doc in documents]
            bm25_model = BM25Okapi(tokenized_corpus)

        for q_index, eval_query in enumerate(queries):
            # 1. Dense leg scores
            dense_scores = {
                doc_id: provider.score_dense(query_encoding.dense_vecs[q_index], doc_vec)
                for doc_id, doc_vec in zip(doc_ids, doc_encoding.dense_vecs)
            }
            dense_ranked = rank_from_scores(dense_scores, len(doc_ids))

            # 2. Sparse leg scores (BGE-M3 lexical weights)
            sparse_scores = {
                doc_id: provider.score_sparse(
                    query_encoding.lexical_weights[q_index], doc_weights
                )
                for doc_id, doc_weights in zip(doc_ids, doc_encoding.lexical_weights)
            }
            sparse_ranked = rank_from_scores(sparse_scores, len(doc_ids))

            # 3. RRF Fusion
            rrf_scores = reciprocal_rank_fusion([dense_ranked, sparse_ranked], rrf_k=rrf_k)
            rrf_ranked = rank_from_scores(rrf_scores, top_k)

            # 4. Relative Score Fusion
            rsf_scores = relative_score_fusion(
                [dense_scores, sparse_scores], weights=[dense_weight, sparse_weight]
            )
            rsf_ranked = rank_from_scores(rsf_scores, top_k)

            all_results.append(
                build_result(eval_query, dense_ranked[:top_k], method="bge_m3_dense_only", k=top_k)
            )
            all_results.append(
                build_result(eval_query, sparse_ranked[:top_k], method="bge_m3_sparse_only", k=top_k)
            )
            all_results.append(
                build_result(eval_query, rrf_ranked, method="rrf_dense_bge_m3_sparse", k=top_k)
            )
            all_results.append(
                build_result(
                    eval_query, rsf_ranked, method="relative_score_dense_bge_m3_sparse", k=top_k
                )
            )

            # Optional BM25 rows
            if bm25_model is not None:
                tokenized_query = tokenize_bm25(eval_query.query)
                bm25_raw = bm25_model.get_scores(tokenized_query)
                bm25_scores = {doc_id: float(s) for doc_id, s in zip(doc_ids, bm25_raw)}
                bm25_ranked = rank_from_scores(bm25_scores, len(doc_ids))

                rrf_bm25 = reciprocal_rank_fusion([dense_ranked, bm25_ranked], rrf_k=rrf_k)
                rsf_bm25 = relative_score_fusion(
                    [dense_scores, bm25_scores], weights=[dense_weight, sparse_weight]
                )

                all_results.append(
                    build_result(eval_query, bm25_ranked[:top_k], method="bm25_only", k=top_k)
                )
                all_results.append(
                    build_result(
                        eval_query, rank_from_scores(rrf_bm25, top_k), method="rrf_dense_bm25", k=top_k
                    )
                )
                all_results.append(
                    build_result(
                        eval_query,
                        rank_from_scores(rsf_bm25, top_k),
                        method="relative_score_dense_bm25",
                        k=top_k,
                    )
                )

    else:
        # Standard BM25 sparse leg + configured dense provider
        doc_vectors_list = embed_dense_texts([doc.text for doc in documents], dense_config)
        doc_vectors = dict(zip(doc_ids, doc_vectors_list))

        query_vectors_list = embed_dense_texts([q.query for q in queries], dense_config)

        # Build BM25 index
        tokenized_corpus = [tokenize_bm25(doc.text) for doc in documents]
        bm25 = BM25Okapi(tokenized_corpus)

        for q_index, (eval_query, query_vec) in enumerate(zip(queries, query_vectors_list)):
            # 1. Dense leg
            dense_scores = {
                doc_id: cosine_similarity(query_vec, doc_vec)
                for doc_id, doc_vec in doc_vectors.items()
            }
            dense_ranked = rank_from_scores(dense_scores, len(doc_ids))

            # 2. Sparse leg (BM25)
            tokenized_query = tokenize_bm25(eval_query.query)
            bm25_raw = bm25.get_scores(tokenized_query)
            bm25_scores = {doc_id: float(s) for doc_id, s in zip(doc_ids, bm25_raw)}
            bm25_ranked = rank_from_scores(bm25_scores, len(doc_ids))

            # 3. RRF Fusion
            rrf_scores = reciprocal_rank_fusion([dense_ranked, bm25_ranked], rrf_k=rrf_k)
            rrf_ranked = rank_from_scores(rrf_scores, top_k)

            # 4. Relative Score Fusion
            rsf_scores = relative_score_fusion(
                [dense_scores, bm25_scores], weights=[dense_weight, sparse_weight]
            )
            rsf_ranked = rank_from_scores(rsf_scores, top_k)

            all_results.append(
                build_result(eval_query, dense_ranked[:top_k], method="dense_only", k=top_k)
            )
            all_results.append(
                build_result(eval_query, bm25_ranked[:top_k], method="bm25_only", k=top_k)
            )
            all_results.append(
                build_result(eval_query, rrf_ranked, method="rrf_dense_bm25", k=top_k)
            )
            all_results.append(
                build_result(eval_query, rsf_ranked, method="relative_score_dense_bm25", k=top_k)
            )

    # Write results CSV
    write_results_csv(all_results, args.output)

    # Unique methods in order of appearance
    emitted_methods: list[str] = []
    for r in all_results:
        if r.method not in emitted_methods:
            emitted_methods.append(r.method)

    summary_lines = [
        f"# Hybrid Retrieval Benchmark Summary: `{sparse_method}`",
        "",
        f"- **Dataset**: `{args.dataset}`",
        f"- **Total Queries**: {len(queries)}",
        f"- **Top-K Parameter**: k={top_k}",
        f"- **Sparse Method**: `{sparse_method}`",
        f"- **RRF Constant (rrf_k)**: {rrf_k}",
        f"- **RSF Weights**: dense={dense_weight}, sparse={sparse_weight}",
        "",
        "## Overall Representation Comparison",
        "",
        f"| Method / Representation | Queries | Hit@1 Rate | Hit@{top_k} Rate |",
        "| :--- | :---: | :---: | :---: |",
    ]

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
