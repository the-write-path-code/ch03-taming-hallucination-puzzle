# Chapter 3: Taming the Hallucination Puzzle

Source repository for Chapter 3 of *The Write Path*. This chapter covers how stronger retrieval design, semantic matching, and hybrid search reduce weak grounding and context failure in enterprise RAG systems.

Companion article: *"Fact-Checking the Future: How BGE-M3 is Taming the RAG Hallucination Puzzle"* by Mohit Aggarwal.

---

## How to Navigate This Repository

This repository is organized progressively so you can explore the concepts at whichever level of detail you prefer:
- **Global Overview (This File)**: Getting started, setup paths, execution order, and chapter summary.
- **Stage Guides**: Each stage folder (`naive_baseline/`, `semantic_bridging_bge_m3/`, `hybrid_retrieval/`, `qdrant_high_fidelity/`) contains a dedicated README focusing on that specific retrieval mechanism.
- **Architecture Diagrams**: [**`docs/workflow.md`**](docs/workflow.md) contains 7 visual flowcharts illustrating the lifecycle and algorithms.
- **Consolidated Results**: [**`results/benchmark_results.md`**](results/benchmark_results.md) contains the authoritative evaluation metrics across all stages and failure modes.

---

## Setup Guide (Simple to Advanced)

Choose the setup path that matches your environment:

### Path 1: Quickstart (Recommended / Zero-Configuration)
Best for getting started immediately. Uses local Ollama or OpenAI with embedded persistent Qdrant (no Docker or heavy model downloads required).

```bash
# 1. Install lightweight base dependencies
uv sync

# 2. Configure environment (defaults to Ollama bge-m3 and embedded Qdrant)
cp .env.example .env

# 3. (If using Ollama locally) Pull the dense embedding model
ollama pull bge-m3
```

With this base environment, you can run Stage 3.1 (Naive Baseline), Stage 3.3 (BM25 + Dense Hybrid), and Stage 3.4 (Embedded Qdrant).

---

### Path 2: Full Local BGE-M3 (Sparse Lexical + ColBERT Reranking)
Required if you want to run Stage 3.2 (BGE-M3 multi-vector representation) or Stage 3.4 with second-stage ColBERT token reranking.

```bash
# Install local FlagEmbedding and PyTorch dependencies
uv sync --extra local-bge-m3
```
*Note: The first run will download the 2.27 GB BAAI/bge-m3 weights locally.*

---

### Path 3: Docker Dashboard or Qdrant Cloud (Optional)
By default, Qdrant runs embedded locally in `./.qdrant_local/` without background services. If you want the visual Qdrant Web UI dashboard or want to use a cloud cluster:

- **Local Docker with Web UI**:
  ```bash
  docker compose up -d
  ```
  Set `QDRANT_URL=http://localhost:6333` in `.env` and access the dashboard at `http://localhost:6333/dashboard`.
- **Qdrant Cloud**:
  Set `QDRANT_URL=https://<your-cluster-id>.qdrant.io` and `QDRANT_API_KEY=<key>` in `.env`.

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
│   ├── README.md               # CSV schema and reproducibility instructions
│   ├── benchmark_results.md    # Authoritative consolidated benchmark report
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

## Benchmark Highlights & Architectural Takeaways

Empirical results measured on the 200-document stress-testing corpus:

| Stage | Retrieval Method | Hit@1 Rate | Hit@3 Rate | Key Insight |
| :--- | :--- | :---: | :---: | :--- |
| **3.1** | `dense_only:ollama` | 12.5% | 21.9% | Dense vectors fail on exact identifiers and isolated tables |
| **3.3** | `bm25_only` | 84.4% | 87.5% | Strong lexical matching on rare alphanumeric codes |
| **3.3** | `rrf_dense_bm25` | 43.8% | 62.5% | Pure rank fusion (k=60) |
| **3.3** | `relative_score_dense_bm25` | **87.5%** | **90.6%** | Score-normalized fusion achieves highest retrieval fidelity |
| **3.4** | `qdrant_dense` | 12.5% | 21.9% | First-stage persistent vector search (1.4 ms/query) |
| **3.4** | `qdrant_dense_colbert_rerank` | **31.2%** | **56.2%** | Second-stage ColBERT reranking improves Hit@3 by 2.5x |

### Component Synergy in Retrieval

| Component | What it Solves | Where it Fails Alone | Practical Role |
| :--- | :--- | :--- | :--- |
| **Dense Vectors** | Conceptual meaning and paraphrasing | Misses exact identifiers and tables | First-stage semantic recall |
| **Sparse / BM25** | Exact alphanumeric codes and tables | Misses synonyms and conceptual intent | First-stage lexical recall |
| **Relative Score Fusion** | Normalizes and balances both legs | N/A (fusion algorithm) | Unifies first-stage search (12.5% to 87.5% Hit@1) |
| **ColBERT Multi-Vectors** | Token-level MaxSim alignment | High latency across full corpus | Precision reranker on top finalists |

**Two-Stage Retrieval Pattern**: In this measured workflow, using **Relative Score Fusion (Dense + BM25)** provides fast candidate retrieval over the full corpus, while applying **ColBERT token reranking** on the top 20 to 50 finalists establishes precision ranking before passing context to an LLM.

For full cross-stage benchmark tables and latency trade-offs, see [**`results/benchmark_results.md`**](results/benchmark_results.md).
For detailed architecture flowcharts, see [**`docs/workflow.md`**](docs/workflow.md).



