# Hybrid Retrieval Benchmark Summary: `bge_m3_sparse`

- **Dataset**: `small`
- **Total Queries**: 16
- **Top-K Parameter**: k=3
- **Sparse Method**: `bge_m3_sparse`
- **RRF Constant (rrf_k)**: 60
- **RSF Weights**: dense=0.5, sparse=0.5

## Overall Representation Comparison

| Method / Representation | Queries | Hit@1 Rate | Hit@3 Rate |
| :--- | :---: | :---: | :---: |
| `bge_m3_dense_only` | 16 | 93.8% | 100.0% |
| `bge_m3_sparse_only` | 16 | 100.0% | 100.0% |
| `rrf_dense_bge_m3_sparse` | 16 | 93.8% | 100.0% |
| `relative_score_dense_bge_m3_sparse` | 16 | 93.8% | 100.0% |

## Performance Breakdown by Failure Mode

### `bge_m3_dense_only`

| Failure Mode | Queries | Hit@1 Rate | Hit@3 Rate |
| :--- | :---: | :---: | :---: |
| `keyword_exact_match` | 6 | 100.0% | 100.0% |
| `long_context` | 3 | 100.0% | 100.0% |
| `semantic_paraphrase` | 3 | 100.0% | 100.0% |
| `table_lookup` | 4 | 75.0% | 100.0% |

### `bge_m3_sparse_only`

| Failure Mode | Queries | Hit@1 Rate | Hit@3 Rate |
| :--- | :---: | :---: | :---: |
| `keyword_exact_match` | 6 | 100.0% | 100.0% |
| `long_context` | 3 | 100.0% | 100.0% |
| `semantic_paraphrase` | 3 | 100.0% | 100.0% |
| `table_lookup` | 4 | 100.0% | 100.0% |

### `rrf_dense_bge_m3_sparse`

| Failure Mode | Queries | Hit@1 Rate | Hit@3 Rate |
| :--- | :---: | :---: | :---: |
| `keyword_exact_match` | 6 | 100.0% | 100.0% |
| `long_context` | 3 | 100.0% | 100.0% |
| `semantic_paraphrase` | 3 | 100.0% | 100.0% |
| `table_lookup` | 4 | 75.0% | 100.0% |

### `relative_score_dense_bge_m3_sparse`

| Failure Mode | Queries | Hit@1 Rate | Hit@3 Rate |
| :--- | :---: | :---: | :---: |
| `keyword_exact_match` | 6 | 100.0% | 100.0% |
| `long_context` | 3 | 100.0% | 100.0% |
| `semantic_paraphrase` | 3 | 100.0% | 100.0% |
| `table_lookup` | 4 | 75.0% | 100.0% |
