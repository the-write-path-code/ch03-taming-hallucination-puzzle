# Benchmark Results Directory

This directory contains the central, consolidated benchmark evaluation outputs across all Chapter 3 retrieval stages.

---

## Directory Structure

```text
results/
├── README.md                   # This documentation
├── benchmark_results.md        # Canonical narrative benchmark report
├── small/                      # Small committed corpus (16 docs, 16 queries)
│   ├── naive_dense_only.csv
│   ├── bge_m3_representations.csv
│   ├── hybrid_fusion.csv
│   ├── hybrid_fusion_bge_m3_sparse.csv
│   ├── qdrant_high_fidelity.csv
│   └── qdrant_collection_config.json
└── generated_large/            # Synthetic large corpus (200 docs, 32 queries)
    ├── naive_dense_only.csv
    ├── hybrid_fusion.csv
    ├── qdrant_high_fidelity.csv
    └── qdrant_collection_config.json
```

---

## CSV File Schema

Every result CSV uses the shared schema defined by `retrieval_core.types.CSV_FIELDNAMES`:

| Column | Description |
| :--- | :--- |
| `query_id` | Unique query identifier (e.g., `q-001`). |
| `query` | Evaluation query text. |
| `expected_doc_ids` | Pipe-delimited list of ground-truth document IDs (`doc1\|doc2`). |
| `retrieved_doc_ids` | Pipe-delimited list of top-k retrieved document IDs. |
| `hit_at_1` | `1` if an expected document appears at rank 1, else `0`. |
| `hit_at_k` | `1` if an expected document appears in top-k (default k=3), else `0`. |
| `failure_mode` | Labeled failure mode category (`keyword_exact_match`, `long_context`, `semantic_paraphrase`, `table_lookup`). |
| `method` | Exact retrieval pipeline or representation identifier (e.g., `dense_only:ollama`, `bge_m3_sparse`, `relative_score_dense_bm25`, `qdrant_dense_colbert_rerank`). |

---

## Reproducing Results

Stage runners write to this directory by default based on the `--dataset` argument:

### 1. Small Dataset Runs
```bash
uv run python naive_baseline/run.py --dataset small
uv run python semantic_bridging_bge_m3/run.py --dataset small
uv run python hybrid_retrieval/run.py --dataset small
uv run python qdrant_high_fidelity/run.py --dataset small --rebuild
```

### 2. Large Dataset Runs
```bash
# 1. Generate the deterministic 200-document corpus
uv run python scripts/generate_large_corpus.py --documents 200 --seed 42

# 2. Execute retrieval stages
uv run python naive_baseline/run.py --dataset generated_large
uv run python hybrid_retrieval/run.py --dataset generated_large
uv run python qdrant_high_fidelity/run.py --dataset generated_large --rebuild
```

---

## Canonical Interpretation

For comprehensive analysis, failure mode breakdowns, latency measurements, and retrieval trade-offs, refer to the canonical report in [**`results/benchmark_results.md`**](benchmark_results.md).
