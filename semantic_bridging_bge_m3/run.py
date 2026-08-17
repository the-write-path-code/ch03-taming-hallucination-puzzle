#!/usr/bin/env python3
"""Stage 3.2: Semantic bridging with native local BGE-M3.

Encodes the corpus and queries with FlagEmbedding.BGEM3FlagModel,
requesting dense vectors, sparse lexical weights, and ColBERT multi-vectors
in a single pass.

Writes results to results/<dataset>/bge_m3_representations.csv, with
method in {bge_m3_dense, bge_m3_sparse, bge_m3_colbert, bge_m3_hybrid}.
"""

import argparse
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

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
from retrieval_core.config import _get, load_dense_provider_config

try:
    from semantic_bridging_bge_m3.provider import BGEM3Encoding, BGEM3Provider
except ImportError:
    from provider import BGEM3Encoding, BGEM3Provider

_hyde_warned = False


def maybe_expand_with_hyde(query: str, use_hyde: bool, api_key: str | None, model: str) -> str:
    """Return the HyDE-expanded query, or the raw query if HyDE is off/unavailable."""
    global _hyde_warned
    if not use_hyde:
        return query
    if not api_key:
        if not _hyde_warned:
            print(
                "HyDE requested (--use-hyde) but OPENAI_API_KEY is not set in .env; "
                "falling back to the raw query for this run."
            )
            _hyde_warned = True
        return query

    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "Write a short, plausible passage (2-3 sentences) that would "
                    "directly answer the following question, as if it were an "
                    "excerpt from a technical document. Do not mention that you "
                    "are guessing."
                ),
            },
            {"role": "user", "content": query},
        ],
        temperature=0.0,
        max_tokens=120,
    )
    return response.choices[0].message.content or query


def rank(scores: dict[str, float], top_k: int) -> list[str]:
    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return [doc_id for doc_id, _ in ranked[:top_k]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the 3.2 native BGE-M3 semantic bridging stage."
    )
    add_dataset_argument(parser)
    parser.add_argument("--top-k", type=int, default=None, help="Override RETRIEVAL_TOP_K.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Where to write results CSV (default: results/<dataset>/bge_m3_representations.csv).",
    )
    parser.add_argument(
        "--use-hyde",
        action="store_true",
        help="Enable HyDE query expansion (requires OPENAI_API_KEY; off by default).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dense_config = load_dense_provider_config()
    top_k = args.top_k if args.top_k is not None else dense_config.top_k
    hyde_model = _get("HYDE_MODEL", "gpt-4o-mini")

    dataset_paths = resolve_dataset(args.dataset)
    output_path = (
        args.output
        if args.output is not None
        else _repo_root / "results" / args.dataset / "bge_m3_representations.csv"
    )

    documents = load_corpus(dataset_paths.corpus_dir)
    queries = load_eval_queries(dataset_paths.eval_queries_path)
    validate_against_corpus(queries, {doc.doc_id for doc in documents})

    print(
        f"Loaded {len(documents)} documents and {len(queries)} queries "
        f"from dataset={args.dataset!r}. HyDE enabled: {args.use_hyde}."
    )

    provider = BGEM3Provider(dense_config.local_bge_m3_model_name)

    doc_ids = [doc.doc_id for doc in documents]
    doc_encoding = provider.encode([doc.text for doc in documents])

    expanded_queries = [
        maybe_expand_with_hyde(
            q.query, args.use_hyde, dense_config.openai_api_key, hyde_model
        )
        for q in queries
    ]
    query_encoding = provider.encode(expanded_queries)

    all_results = []
    for q_index, eval_query in enumerate(queries):
        dense_scores = {
            doc_id: provider.score_dense(query_encoding.dense_vecs[q_index], doc_vec)
            for doc_id, doc_vec in zip(doc_ids, doc_encoding.dense_vecs)
        }
        sparse_scores = {
            doc_id: provider.score_sparse(
                query_encoding.lexical_weights[q_index], doc_weights
            )
            for doc_id, doc_weights in zip(doc_ids, doc_encoding.lexical_weights)
        }
        colbert_scores = {
            doc_id: provider.score_colbert(
                query_encoding.colbert_vecs[q_index], doc_vecs
            )
            for doc_id, doc_vecs in zip(doc_ids, doc_encoding.colbert_vecs)
        }
        hybrid_scores = {
            doc_id: provider.score_hybrid(query_encoding, q_index, doc_encoding, d_index)
            for d_index, doc_id in enumerate(doc_ids)
        }

        for method, scores in (
            ("bge_m3_dense", dense_scores),
            ("bge_m3_sparse", sparse_scores),
            ("bge_m3_colbert", colbert_scores),
            ("bge_m3_hybrid", hybrid_scores),
        ):
            retrieved = rank(scores, top_k)
            all_results.append(build_result(eval_query, retrieved, method=method, k=top_k))

    write_results_csv(all_results, output_path)

    methods = ("bge_m3_dense", "bge_m3_sparse", "bge_m3_colbert", "bge_m3_hybrid")
    for method in methods:
        subset = [r for r in all_results if r.method == method]
        stats = summarize(subset)
        print(
            f"{method}: hit@1={stats['hit_at_1_rate']:.2f} "
            f"hit@{top_k}={stats['hit_at_k_rate']:.2f} (n={stats['count']})"
        )

    print(f"Wrote {len(all_results)} rows to {output_path}.")


if __name__ == "__main__":
    main()

