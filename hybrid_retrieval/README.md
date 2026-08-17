# 3.3 Hybrid Retrieval: Combining Dense and Sparse Search

This stage implements and compares **hybrid retrieval** by combining independent
dense and sparse search legs using two primary fusion techniques:
1. **Reciprocal Rank Fusion (RRF)**: A scale-agnostic, rank-based combination.
2. **Relative Score Fusion (RSF)**: Min-max score normalization followed by weighted summation.

---

## Why Raw-Score Averaging Fails

In Stage 3.2 (`semantic_bridging_bge_m3/`), we observed that native BGE-M3 sparse
and ColBERT multi-vector legs each achieved **100.0% Hit@1** on the small evaluation
set. However, a naive equal-weight average of raw dense, sparse, and ColBERT scores
dropped Hit@1 back down to **93.8%** (the same as dense-only).

Why did this happen?
- **Incompatible Score Scales**: Cosine similarities (dense), inner products / BM25 scores (sparse), and token MaxSim sums (ColBERT) have drastically different distributions, magnitudes, and variances.
- **Score Dominance**: A large raw score from one leg can completely drown out subtle but critical discriminating signals from another leg unless scores are properly calibrated or combined via ranks.

Hybrid retrieval solves this by either ignoring raw score magnitudes entirely (**RRF**) or normalizing each leg independently into $[0, 1]$ before weighting (**Relative Score Fusion**).

---

## Independent Retrieval Legs

In this stage, dense and sparse retrieval are executed as completely separate pipelines:

1. **Dense Leg**:
   - Encodes documents and queries into dense semantic embedding vectors using the configured provider (`ollama`, `openai`, or native `local-bge-m3`).
   - Ranks documents by cosine similarity.
2. **Sparse Leg**:
   - **BM25 (`HYBRID_SPARSE_METHOD=bm25`)**: Lightweight lexical baseline using `rank_bm25` with a deterministic regex tokenizer that preserves compound technical identifiers (`AX4-E117`, `CC-17`, `NSN-42`, `HRD-9C03`).
   - **BGE-M3 Lexical Weights (`HYBRID_SPARSE_METHOD=bge_m3_sparse`)**: Full model-based sparse representation (requires `uv sync --extra local-bge-m3`).

---

## Fusion Algorithms

### 1. Reciprocal Rank Fusion (RRF)
For each document at position `rank` (0-indexed) within each ranked list $L$:

$$\text{Score}_{\text{RRF}}(d) = \sum_{L} \frac{1}{\text{RRF\_K} + \text{rank}_L(d) + 1}$$

- Does not depend on score magnitudes or calibration.
- Controlled by the smoothing constant `RRF_K` (default `60`).

### 2. Relative Score Fusion (RSF)
For each score source $S$, raw scores are min-max normalized across all retrieved candidates:

$$\text{NormScore}_S(d) = \frac{\text{Score}_S(d) - \min(\text{Scores}_S)}{\max(\text{Scores}_S) - \min(\text{Scores}_S)}$$

If all scores in a source are identical, every candidate in that source receives $1.0$. The final score is the normalized weighted sum:

$$\text{Score}_{\text{RSF}}(d) = w_{\text{dense}} \cdot \text{NormScore}_{\text{dense}}(d) + w_{\text{sparse}} \cdot \text{NormScore}_{\text{sparse}}(d)$$

---

## Configuration

All settings live centrally in the root `.env` file:

| Variable | Default | Description |
|---|---|---|
| `HYBRID_SPARSE_METHOD` | `bm25` | Sparse leg: `bm25` (base environment) or `bge_m3_sparse` (requires `local-bge-m3`). |
| `HYBRID_FUSION_METHOD` | `rrf` | Default fusion method: `rrf` or `relative_score`. |
| `RRF_K` | `60` | Rank constant parameter for Reciprocal Rank Fusion. |
| `HYBRID_DENSE_WEIGHT` | `0.5` | Dense leg weight for relative score fusion (normalized by sum). |
| `HYBRID_SPARSE_WEIGHT` | `0.5` | Sparse leg weight for relative score fusion (normalized by sum). |
| `RETRIEVAL_TOP_K` | `3` | Default `k` for Hit@k evaluation. |

---

## Running

### 1. Lightweight BM25 Hybrid (Base Environment)

```bash
uv sync
uv run python hybrid_retrieval/run.py --dataset small
```

Output is written to `hybrid_retrieval/fusion_comparison.csv` and summarized in `hybrid_retrieval/fusion_comparison.md`, containing:
- `dense_only`
- `bm25_only`
- `rrf_dense_bm25`
- `relative_score_dense_bm25`

### 2. Full Native BGE-M3 Sparse Hybrid

```bash
uv sync --extra local-bge-m3
uv run python hybrid_retrieval/run.py --dataset small --sparse-method bge_m3_sparse
```

Generates:
- `bge_m3_dense_only`
- `bge_m3_sparse_only`
- `rrf_dense_bge_m3_sparse`
- `relative_score_dense_bge_m3_sparse`

### 3. Stress-Testing on the 200-Document Corpus

```bash
uv run python scripts/generate_large_corpus.py --documents 200 --seed 42
uv run python hybrid_retrieval/run.py --dataset generated_large --output hybrid_retrieval/fusion_comparison_large.csv
```

---

## Scope & Next Steps

This stage focuses strictly on algorithmic score fusion and lexical/dense retrieval legs without vector databases or generative LLMs.

In the next stage (**Section 3.4: `qdrant_high_fidelity/`**), we will deploy these hybrid vectors into a persistent Qdrant collection with payload filtering and vector index optimization.
