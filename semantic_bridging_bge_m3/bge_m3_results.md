# BGE-M3 Semantic Bridging Benchmark: `small`

- **Dataset**: `small`
- **Total Queries**: 16
- **Top-K Parameter**: k=3
- **HyDE Enabled**: False

## Overall Representation Comparison

| Method / Representation | Queries | Hit@1 Rate | Hit@3 Rate |
| :--- | :---: | :---: | :---: |
| `bge_m3_dense` | 16 | 93.8% | 100.0% |
| `bge_m3_sparse` | 16 | 100.0% | 100.0% |
| `bge_m3_colbert` | 16 | 100.0% | 100.0% |
| `bge_m3_hybrid` | 16 | 93.8% | 100.0% |

## Performance Breakdown by Failure Mode

### `bge_m3_dense`

| Failure Mode | Queries | Hit@1 Rate | Hit@3 Rate |
| :--- | :---: | :---: | :---: |
| `keyword_exact_match` | 6 | 100.0% | 100.0% |
| `long_context` | 3 | 100.0% | 100.0% |
| `semantic_paraphrase` | 3 | 100.0% | 100.0% |
| `table_lookup` | 4 | 75.0% | 100.0% |

### `bge_m3_sparse`

| Failure Mode | Queries | Hit@1 Rate | Hit@3 Rate |
| :--- | :---: | :---: | :---: |
| `keyword_exact_match` | 6 | 100.0% | 100.0% |
| `long_context` | 3 | 100.0% | 100.0% |
| `semantic_paraphrase` | 3 | 100.0% | 100.0% |
| `table_lookup` | 4 | 100.0% | 100.0% |

### `bge_m3_colbert`

| Failure Mode | Queries | Hit@1 Rate | Hit@3 Rate |
| :--- | :---: | :---: | :---: |
| `keyword_exact_match` | 6 | 100.0% | 100.0% |
| `long_context` | 3 | 100.0% | 100.0% |
| `semantic_paraphrase` | 3 | 100.0% | 100.0% |
| `table_lookup` | 4 | 100.0% | 100.0% |

### `bge_m3_hybrid`

| Failure Mode | Queries | Hit@1 Rate | Hit@3 Rate |
| :--- | :---: | :---: | :---: |
| `keyword_exact_match` | 6 | 100.0% | 100.0% |
| `long_context` | 3 | 100.0% | 100.0% |
| `semantic_paraphrase` | 3 | 100.0% | 100.0% |
| `table_lookup` | 4 | 75.0% | 100.0% |
