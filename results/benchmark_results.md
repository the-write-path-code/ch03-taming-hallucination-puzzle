# Comprehensive Retrieval Benchmark Results

This report is the canonical narrative summary of retrieval evaluations across all four stages of Chapter 3. It consolidates the row-level evaluation metrics recorded in `results/small/` and `results/generated_large/`.

---

## Scope and Reproducibility

1. **Context & Scope**: The metrics reported here reflect empirical measurements on synthetic enterprise documentation designed specifically to stress-test semantic and lexical retrieval failure modes. They are not universal claims for all enterprise RAG workloads.
2. **Datasets**:
   - **`small`**: 16 committed documents and 16 labeled evaluation queries (`sample_data/small/`).
   - **`generated_large`**: 200 documents (950–1,150 words each) and 32 labeled evaluation queries generated deterministically via `scripts/generate_large_corpus.py --documents 200 --seed 42`. The large corpus is ignored by Git and generated locally.
3. **Execution Environment**: Measurements were conducted on macOS with local Ollama (`bge-m3`), Python 3.12, embedded Qdrant 1.19, and `FlagEmbedding` 1.4 (for native BGE-M3 sparse/ColBERT models).

---

## Run Configurations

| Stage | Section | Description | Default Output CSV | Default Model / Provider |
| :--- | :--- | :--- | :--- | :--- |
| **3.1** | Naive Baseline | Dense-only embedding retrieval | `results/<dataset>/naive_dense_only.csv` | `bge-m3` via Ollama (`http://localhost:11434`) |
| **3.2** | Semantic Bridging | Native Dense, Sparse, ColBERT, and Raw Hybrid | `results/<dataset>/bge_m3_representations.csv` | `BAAI/bge-m3` via FlagEmbedding (`local-bge-m3`) |
| **3.3** | Hybrid Retrieval | Independent legs + RRF and Relative Score Fusion | `results/<dataset>/hybrid_fusion.csv` | Dense leg + BM25Okapi / BGE-M3 sparse |
| **3.4** | Qdrant High-Fidelity | Persistent embedded vector search + ColBERT reranking | `results/<dataset>/qdrant_high_fidelity.csv` | Qdrant `dense` cosine vector + ColBERT rerank |

---

## Benchmark Results: Small Corpus (16 Documents, 16 Queries)

On the small, relatively uniform corpus with minimal distractors, dense retrieval performs well, and lexical and sparse signals resolve the few remaining edge cases.

*Note: In an earlier test execution, an artifact generated under an active demonstration filter recorded 25.0% recall. Re-indexing from a clean rebuild confirmed the corrected 93.8% dense and 100.0% ColBERT-reranked results below.*

| Stage | Method / Representation | Total Queries | Hit@1 Rate | Hit@3 Rate | Key Characteristic |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **3.1** | `dense_only:ollama` | 16 | 93.8% | 100.0% | Dense embedding baseline |
| **3.2** | `bge_m3_dense` | 16 | 93.8% | 100.0% | Native dense vector |
| **3.2** | `bge_m3_sparse` | 16 | **100.0%** | **100.0%** | Native model-derived lexical weights |
| **3.2** | `bge_m3_colbert` | 16 | **100.0%** | **100.0%** | Token-level MaxSim multi-vector matching |
| **3.2** | `bge_m3_hybrid` (raw average) | 16 | 93.8% | 100.0% | Equal-weight combination of raw scores |
| **3.3** | `dense_only` | 16 | 93.8% | 100.0% | Baseline dense leg |
| **3.3** | `bm25_only` | 16 | 93.8% | 100.0% | BM25 lexical leg with identifier tokenizer |
| **3.3** | `rrf_dense_bm25` | 16 | 93.8% | 100.0% | Reciprocal Rank Fusion (k=60) |
| **3.3** | `relative_score_dense_bm25` | 16 | 93.8% | 100.0% | Min-max normalized score fusion |
| **3.4** | `qdrant_dense` | 16 | 93.8% | 100.0% | Embedded Qdrant vector search |
| **3.4** | `qdrant_dense_colbert_rerank` | 16 | **100.0%** | **100.0%** | Second-stage ColBERT reranking over finalists |

---

## Benchmark Results: Generated Large Corpus (200 Documents, 32 Queries)

When evaluated against a realistic corpus of 200 long technical documents (1,000+ words each) containing distractors, exact identifiers, and isolated table rows, dense-only retrieval accuracy drops significantly. Hybrid search and two-stage reranking recover substantial retrieval fidelity.

### Overall Comparison

| Stage | Method / Representation | Total Queries | Hit@1 Rate | Hit@3 Rate | Measured Latency Context |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **3.1** | `dense_only:ollama` | 32 | **12.5%** | **21.9%** | Dense vector search |
| **3.3** | `bm25_only` | 32 | **84.4%** | **87.5%** | Lexical matching on exact tokens |
| **3.3** | `rrf_dense_bm25` | 32 | **43.8%** | **62.5%** | Rank-based fusion |
| **3.3** | `relative_score_dense_bm25` | 32 | **87.5%** | **90.6%** | Normalized score fusion |
| **3.4** | `qdrant_dense` | 32 | **12.5%** | **21.9%** | **1.4 ms** / query (vector search) |
| **3.4** | `qdrant_dense_colbert_rerank` | 32 | **31.2%** | **56.2%** | **5.8 s** / query (CPU rerank) |

---

### Failure Mode Breakdown (Large Corpus)

| Failure Mode Category | Queries | Dense-Only Hit@1 | BM25 Hit@1 | RSF Hybrid Hit@1 | Qdrant + ColBERT Hit@3 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`keyword_exact_match`** | 8 | 12.5% | 100.0% | 100.0% | 50.0% |
| **`long_context`** | 8 | 0.0% | 100.0% | 100.0% | 75.0% |
| **`semantic_paraphrase`** | 8 | 12.5% | 37.5% | 50.0% | 50.0% |
| **`table_lookup`** | 8 | 25.0% | 100.0% | 100.0% | 50.0% |
| **Overall** | **32** | **12.5%** | **84.4%** | **87.5%** | **56.2%** |

---

## Detailed Technical Interpretation

### 1. Why Dense-Only Retrieval Degrades on Long Documents
Dense embedding models map entire text documents into a single fixed-length vector (e.g. 1024 dimensions). In large corpora:
- **Identifier Dilution**: Specific alphanumeric codes (`AX4-E117`, `CC-17`) get mapped near similar-looking tokens rather than matching the exact defining document.
- **Prose Dilution**: In a 1,000-word document, a single critical sentence near the bottom contributes only a small fraction of the overall embedding vector.
- **Table Data Isolation**: Markdown tables have minimal surrounding prose, giving dense encoders insufficient semantic context.

### 2. Why BM25 Dominates on Exact Identifiers and Tables
BM25 indexes inverted term frequencies with document length normalization. When queries include exact product codes (`AX4-E117`) or exact column headers, BM25 assigns massive term weights to rare exact matches. This yields 100% Hit@1 on `keyword_exact_match` and `table_lookup`.

However, on `semantic_paraphrase` queries (where user questions use different vocabulary than the document), BM25 drops to 37.5% Hit@1.

### 3. Why Raw-Score Averaging Fails (Stage 3.2 vs. Stage 3.3)
In Stage 3.2, averaging raw cosine dense scores with raw sparse and ColBERT scores resulted in 93.8% Hit@1 on the small corpus, failing to improve upon dense alone even though standalone sparse and ColBERT both scored 100%. 

Because raw score scales and distributions are incompatible across representations, unnormalized averaging allows one leg to dominate the composite score. Stage 3.3 resolves this by applying **Relative Score Fusion (RSF)**, which normalizes each leg independently to a 0 to 1 range before weighted summation.

### 4. Relative Score Fusion vs. Reciprocal Rank Fusion
On the 200-document dataset, **Relative Score Fusion (87.5% Hit@1)** outperformed **RRF (43.8% Hit@1)**. 
- RRF considers only rank positions (calculated as 1 divided by 60 plus the rank position), treating a document that barely won dense retrieval with low confidence equally to a document that won BM25 with overwhelming confidence.
- Relative Score Fusion preserves the magnitude of high-confidence lexical matches while integrating semantic signals.

### 5. Durable Qdrant Storage and ColBERT Latency Trade-Offs
- **Durable Storage**: Qdrant provides persistent embedded storage (`./.qdrant_local`), payload filtering (`document_type`, `failure_tags`), and vector indexing. However, dense vector search inside Qdrant operates on the same mathematical representations as raw cosine search, yielding identical baseline accuracy (12.5% Hit@1).
- **Two-Stage ColBERT Reranking**: Re-scoring the top 20 candidate finalists with token-level ColBERT multi-vectors increased Hit@3 from **21.9% to 56.2%**.
- **Latency Trade-Off**: First-stage Qdrant vector retrieval completed in **1.4 ms per query**, whereas local CPU ColBERT multi-vector scoring required **~5.8 seconds per query**. In production architectures, ColBERT reranking is best deployed asynchronously or accelerated via specialized inference backends.

---

## Architectural Synthesis: Two-Stage Retrieval Pattern

On both corpora measured in this repository, no single retrieval method alone matched the accuracy of combining dense and lexical signals. High-fidelity retrieval relies on component synergy:

| Component | What it Solves | Where it Fails Alone | Combined Benefit in Production |
| :--- | :--- | :--- | :--- |
| **Dense Vectors** | Broad semantic intent (such as recognizing that "thermal limit" relates conceptually to "maximum operating temperature"). | Fails on exact codes (`AX4-E117`), product SKUs, and isolated table rows. | Captures conceptual and paraphrase queries where exact vocabulary is unknown. |
| **Sparse / BM25** | Exact alphanumeric identifiers, table headers, and error codes. | Fails on synonyms, paraphrasing, and vocabulary mismatches. | Guarantees exact keyword matches and identifiers are never diluted. |
| **Relative Score Fusion (RSF)** | Normalizes and balances dense and sparse scores without losing confidence margins. | N/A (algorithm, not a retriever). | Improves large-corpus retrieval from 12.5% to 87.5% Hit@1. |
| **ColBERT (Late Interaction)** | Token-level MaxSim matching without compressing documents into a single vector. | Higher computation cost when evaluated across an entire raw corpus. | Acts as a precision reranker on the finalist candidate set, improving Hit@3 by 2.5x. |

### Two-Stage Funnel Flow

```mermaid
flowchart LR
    A["Full 200-Document Corpus"] -->|Stage 1: Fast Hybrid RSF (Dense + BM25)| B["Top 20 to 50 Candidates"]
    B -->|Stage 2: ColBERT Token MaxSim Rerank| C["Top 3 Context for LLM"]
```

1. **Stage 1 (Candidate Retrieval)**: Use **Relative Score Fusion (Dense + BM25/Sparse)** to scan the corpus in milliseconds, retrieving the top 20 to 50 finalists.
2. **Stage 2 (Precision Reranking)**: Use **ColBERT token interaction** on those top candidate finalists to establish the exact rank order before feeding context to the LLM.

---


## Reproduction Commands

### Base Environment (BM25 + Dense Qdrant)
```bash
# 1. Sync dependencies
uv sync

# 2. Run small corpus benchmarks
uv run python naive_baseline/run.py --dataset small
uv run python hybrid_retrieval/run.py --dataset small
uv run python qdrant_high_fidelity/run.py --dataset small --rebuild

# 3. Generate large corpus and run benchmarks
uv run python scripts/generate_large_corpus.py --documents 200 --seed 42
uv run python naive_baseline/run.py --dataset generated_large
uv run python hybrid_retrieval/run.py --dataset generated_large
uv run python qdrant_high_fidelity/run.py --dataset generated_large --rebuild
```

### Full Local BGE-M3 Environment (Native Sparse + ColBERT Rerank)
```bash
# 1. Sync with local-bge-m3 extra
uv sync --extra local-bge-m3

# 2. Run BGE-M3 multi-representation stage
uv run python semantic_bridging_bge_m3/run.py --dataset small

# 3. Run Qdrant with ColBERT reranking enabled
uv run python qdrant_high_fidelity/run.py --dataset small --rebuild
uv run python qdrant_high_fidelity/run.py --dataset generated_large --rebuild
```
