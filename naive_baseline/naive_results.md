# Retrieval Benchmark Summary: `dense_only:ollama`

- **Dataset**: `small`
- **Total Queries**: 16
- **Top-K Parameter**: k=3
- **Overall Hit@1**: 93.8%
- **Overall Hit@3**: 100.0%

## Performance Breakdown by Failure Mode

| Failure Mode | Queries | Hit@1 Rate | Hit@3 Rate |
| :--- | :---: | :---: | :---: |
| `keyword_exact_match` | 6 | 100.0% | 100.0% |
| `long_context` | 3 | 100.0% | 100.0% |
| `semantic_paraphrase` | 3 | 100.0% | 100.0% |
| `table_lookup` | 4 | 75.0% | 100.0% |
| **Overall** | **16** | **93.8%** | **100.0%** |
