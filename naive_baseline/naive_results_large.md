# Retrieval Benchmark Summary: `dense_only:ollama`

- **Dataset**: `generated_large`
- **Total Queries**: 32
- **Top-K Parameter**: k=3
- **Overall Hit@1**: 12.5%
- **Overall Hit@3**: 21.9%

## Performance Breakdown by Failure Mode

| Failure Mode | Queries | Hit@1 Rate | Hit@3 Rate |
| :--- | :---: | :---: | :---: |
| `keyword_exact_match` | 8 | 12.5% | 12.5% |
| `long_context` | 8 | 0.0% | 25.0% |
| `semantic_paraphrase` | 8 | 12.5% | 12.5% |
| `table_lookup` | 8 | 25.0% | 37.5% |
| **Overall** | **32** | **12.5%** | **21.9%** |
