# Chapter 3: Taming the Hallucination Puzzle

Source repository for Chapter 3 of *The Write Path* (working title). This chapter
covers how stronger retrieval design, semantic matching, and hybrid search reduce
weak grounding and context failure in enterprise RAG systems.

Companion article: "Fact-Checking the Future: How BGE-M3 is Taming the RAG
Hallucination Puzzle" by Mohit Aggarwal.

## Setup

```bash
uv sync
docker compose up -d      # starts a local Qdrant instance with persistent storage
cp .env.example .env      # fill in OPENAI_API_KEY; QDRANT_URL/API_KEY default to local
```

To use Qdrant Cloud instead of local Docker, replace `QDRANT_URL` and `QDRANT_API_KEY`
in `.env` with your cluster's values. No code changes are required.

## Structure

| Folder | Book section | What it demonstrates |
|---|---|---|
| `naive_baseline/` | 3.1 Why naive retrieval fails in enterprise RAG | Dense-only retrieval missing exact-match and table-lookup queries |
| `semantic_bridging_bge_m3/` | 3.2 Semantic bridging with BGE-M3 | Unified dense + sparse + multi-vector retrieval with HyDE query expansion |
| `hybrid_retrieval/` | 3.3 Hybrid retrieval with dense and sparse search | RRF and RSF fusion compared side by side |
| `qdrant_high_fidelity/` | 3.4 Building high-fidelity retrieval in Qdrant | Named-vector hybrid collection, persisted (not in-memory), with metadata filtering |

Section 3.5 (Summary) has no accompanying folder.

Further reading on newer embedding and sparse retrieval techniques (Qwen3-Embedding,
E5-Mistral, GTE, SPLADE-v3, uniCOIL) is in `docs/further_reading.md`.

## Data

`sample_data/corpus/` is a synthetic, self-authored multi-document corpus (no real
company or customer data), built to include exact-identifier jargon, paraphrase
pairs, long documents, and mixed text/table content. `sample_data/eval_queries.json`
is a labeled query set used to score each retrieval stage.

## Status

Scaffolding only. Stage folders are being built in order: sample data first, then
`naive_baseline`, `semantic_bridging_bge_m3`, `hybrid_retrieval`, `qdrant_high_fidelity`.
