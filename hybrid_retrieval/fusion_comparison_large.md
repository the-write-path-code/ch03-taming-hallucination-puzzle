# Hybrid Retrieval Benchmark Summary: `bm25`

- **Dataset**: `generated_large`
- **Total Queries**: 32
- **Top-K Parameter**: k=3
- **Sparse Method**: `bm25`
- **RRF Constant (rrf_k)**: 60
- **RSF Weights**: dense=0.5, sparse=0.5

## Overall Representation Comparison

| Method / Representation | Queries | Hit@1 Rate | Hit@3 Rate |
| :--- | :---: | :---: | :---: |
| `dense_only` | 32 | 12.5% | 21.9% |
| `bm25_only` | 32 | 84.4% | 87.5% |
| `rrf_dense_bm25` | 32 | 43.8% | 62.5% |
| `relative_score_dense_bm25` | 32 | 87.5% | 90.6% |

## Performance Breakdown by Failure Mode

### `dense_only`

| Failure Mode | Queries | Hit@1 Rate | Hit@3 Rate |
| :--- | :---: | :---: | :---: |
| `keyword_exact_match` | 8 | 12.5% | 12.5% |
| `long_context` | 8 | 0.0% | 25.0% |
| `semantic_paraphrase` | 8 | 12.5% | 12.5% |
| `table_lookup` | 8 | 25.0% | 37.5% |

### `bm25_only`

| Failure Mode | Queries | Hit@1 Rate | Hit@3 Rate |
| :--- | :---: | :---: | :---: |
| `keyword_exact_match` | 8 | 100.0% | 100.0% |
| `long_context` | 8 | 100.0% | 100.0% |
| `semantic_paraphrase` | 8 | 37.5% | 50.0% |
| `table_lookup` | 8 | 100.0% | 100.0% |

### `rrf_dense_bm25`

| Failure Mode | Queries | Hit@1 Rate | Hit@3 Rate |
| :--- | :---: | :---: | :---: |
| `keyword_exact_match` | 8 | 75.0% | 75.0% |
| `long_context` | 8 | 25.0% | 62.5% |
| `semantic_paraphrase` | 8 | 25.0% | 50.0% |
| `table_lookup` | 8 | 50.0% | 62.5% |

### `relative_score_dense_bm25`

| Failure Mode | Queries | Hit@1 Rate | Hit@3 Rate |
| :--- | :---: | :---: | :---: |
| `keyword_exact_match` | 8 | 100.0% | 100.0% |
| `long_context` | 8 | 100.0% | 100.0% |
| `semantic_paraphrase` | 8 | 50.0% | 62.5% |
| `table_lookup` | 8 | 100.0% | 100.0% |
