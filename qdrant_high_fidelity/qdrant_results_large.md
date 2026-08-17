# Qdrant High-Fidelity Retrieval Summary: `ch03_retrieval_generated_large`

- **Dataset**: `generated_large`
- **Collection Name**: `ch03_retrieval_generated_large`
- **Connection Mode**: `embedded`
- **Total Queries**: 32
- **Top-K Parameter**: k=3
- **Rerank Candidate Pool**: 20
- **Total Search Time**: 0.045s (avg 1.4ms/query)
- **Total ColBERT Rerank Time**: 187.449s (avg 5857.8ms/query)

## Overall Representation Comparison

| Method / Representation | Queries | Hit@1 Rate | Hit@3 Rate |
| :--- | :---: | :---: | :---: |
| `qdrant_dense` | 32 | 12.5% | 21.9% |
| `qdrant_dense_colbert_rerank` | 32 | 31.2% | 56.2% |

## Performance Breakdown by Failure Mode

### `qdrant_dense`

| Failure Mode | Queries | Hit@1 Rate | Hit@3 Rate |
| :--- | :---: | :---: | :---: |
| `keyword_exact_match` | 8 | 12.5% | 12.5% |
| `long_context` | 8 | 0.0% | 25.0% |
| `semantic_paraphrase` | 8 | 12.5% | 12.5% |
| `table_lookup` | 8 | 25.0% | 37.5% |

### `qdrant_dense_colbert_rerank`

| Failure Mode | Queries | Hit@1 Rate | Hit@3 Rate |
| :--- | :---: | :---: | :---: |
| `keyword_exact_match` | 8 | 25.0% | 50.0% |
| `long_context` | 8 | 50.0% | 75.0% |
| `semantic_paraphrase` | 8 | 12.5% | 50.0% |
| `table_lookup` | 8 | 37.5% | 50.0% |
