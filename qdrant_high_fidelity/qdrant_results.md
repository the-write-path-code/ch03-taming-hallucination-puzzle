# Qdrant High-Fidelity Retrieval Summary: `ch03_retrieval_small`

- **Dataset**: `small`
- **Collection Name**: `ch03_retrieval_small`
- **Connection Mode**: `embedded`
- **Total Queries**: 16
- **Top-K Parameter**: k=3
- **Rerank Candidate Pool**: 20
- **Total Search Time**: 0.026s (avg 1.7ms/query)
- **Total ColBERT Rerank Time**: 8.992s (avg 562.0ms/query)

## Overall Representation Comparison

| Method / Representation | Queries | Hit@1 Rate | Hit@3 Rate |
| :--- | :---: | :---: | :---: |
| `qdrant_dense` | 16 | 25.0% | 25.0% |
| `qdrant_dense_colbert_rerank` | 16 | 25.0% | 25.0% |

## Performance Breakdown by Failure Mode

### `qdrant_dense`

| Failure Mode | Queries | Hit@1 Rate | Hit@3 Rate |
| :--- | :---: | :---: | :---: |
| `keyword_exact_match` | 6 | 66.7% | 66.7% |
| `long_context` | 3 | 0.0% | 0.0% |
| `semantic_paraphrase` | 3 | 0.0% | 0.0% |
| `table_lookup` | 4 | 0.0% | 0.0% |

### `qdrant_dense_colbert_rerank`

| Failure Mode | Queries | Hit@1 Rate | Hit@3 Rate |
| :--- | :---: | :---: | :---: |
| `keyword_exact_match` | 6 | 66.7% | 66.7% |
| `long_context` | 3 | 0.0% | 0.0% |
| `semantic_paraphrase` | 3 | 0.0% | 0.0% |
| `table_lookup` | 4 | 0.0% | 0.0% |
