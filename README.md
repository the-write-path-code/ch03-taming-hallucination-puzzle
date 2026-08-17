# Chapter 3: Taming the Hallucination Puzzle

Source repository for Chapter 3 of *The Write Path*. This chapter covers how stronger retrieval design, semantic matching, and hybrid search reduce weak grounding and context failure in enterprise RAG systems.

Companion article: *"Fact-Checking the Future: How BGE-M3 is Taming the RAG Hallucination Puzzle"* by Mohit Aggarwal.

---

## Setup

### 1. Base Environment (Ollama / OpenAI / BM25 / Embedded Qdrant)
```bash
uv sync
cp .env.example .env
```

### 2. Full Local BGE-M3 Environment (Native Sparse + ColBERT Reranking)
```bash
uv sync --extra local-bge-m3
```

---

## Qdrant Deployment Modes

Embedded local persistent storage is the default and requires no Docker daemon:
- **Embedded Local (Default)**: Persisted in `./.qdrant_local/`. Deliberately avoids ephemeral in-memory storage (`:memory:`).
- **Docker Server (Optional)**: Run `docker compose up -d` and set `QDRANT_URL=http://localhost:6333` in `.env` to use Docker with the Qdrant Web UI.
- **Qdrant Cloud**: Set `QDRANT_URL` and `QDRANT_API_KEY` in `.env`.

Application code connects through a single unified abstraction in `qdrant_high_fidelity/client.py`.

---

## Project Structure

```text
├── naive_baseline/             # 3.1 Dense-only retrieval baseline & failure analysis
├── semantic_bridging_bge_m3/   # 3.2 Native BGE-M3 dense, sparse, and ColBERT vectors
├── hybrid_retrieval/           # 3.3 Independent dense + sparse legs with RRF and RSF fusion
├── qdrant_high_fidelity/       # 3.4 Persistent Qdrant vector database + ColBERT reranking
├── src/retrieval_core/         # Shared corpus loader, query validation, and config modules
├── sample_data/
│   └── small/                  # Committed 16-document evaluation corpus
├── scripts/
│   └── generate_large_corpus.py# Deterministic 200-document synthetic stress-test generator
├── results/                    # Centralized benchmark results & canonical narrative report
│   ├── README.md
│   ├── benchmark_results.md    # Canonical consolidated benchmark report
│   ├── small/                  # Results on 16-document dataset
│   └── generated_large/        # Results on 200-document dataset
├── docs/
│   ├── workflow.md             # 7 GitHub-compatible Mermaid architecture flowcharts
│   └── further_reading.md      # Survey of modern dense & learned sparse architectures
└── tests/                      # Unit test suite covering all modules and algorithms
```

---

## Execution & Run Order

Run the four retrieval stages in sequence. All stages write row-level evidence directly to `results/<dataset>/`:

### 1. Small Dataset (16 Documents)
```bash
uv run python naive_baseline/run.py --dataset small
uv run python semantic_bridging_bge_m3/run.py --dataset small
uv run python hybrid_retrieval/run.py --dataset small
uv run python qdrant_high_fidelity/run.py --dataset small --rebuild
```

### 2. Large Dataset (200 Documents, 1,000+ words/doc)
```bash
# 1. Generate the deterministic 200-document corpus (seed 42)
uv run python scripts/generate_large_corpus.py --documents 200 --seed 42

# 2. Run retrieval stages
uv run python naive_baseline/run.py --dataset generated_large
uv run python hybrid_retrieval/run.py --dataset generated_large
uv run python qdrant_high_fidelity/run.py --dataset generated_large --rebuild
```

---

## Benchmark Highlights

Empirical results measured on the 200-document stress-testing corpus:

| Stage | Retrieval Method | Hit@1 Rate | Hit@3 Rate | Key Insight |
| :--- | :--- | :---: | :---: | :--- |
| **3.1** | `dense_only:ollama` | 12.5% | 21.9% | Dense vectors fail on exact identifiers and isolated tables |
| **3.3** | `bm25_only` | 84.4% | 87.5% | Strong lexical matching on rare alphanumeric codes |
| **3.3** | `rrf_dense_bm25` | 43.8% | 62.5% | Pure rank fusion ($k=60$) |
| **3.3** | `relative_score_dense_bm25` | **87.5%** | **90.6%** | Score-normalized fusion achieves highest retrieval fidelity |
| **3.4** | `qdrant_dense` | 12.5% | 21.9% | First-stage persistent vector search (1.4 ms/query) |
| **3.4** | `qdrant_dense_colbert_rerank` | **31.2%** | **56.2%** | Second-stage ColBERT reranking improves Hit@3 by 2.5x |

For full cross-stage benchmark tables and latency trade-offs, see [**`results/benchmark_results.md`**](results/benchmark_results.md).
For detailed architecture flowcharts, see [**`docs/workflow.md`**](docs/workflow.md).

