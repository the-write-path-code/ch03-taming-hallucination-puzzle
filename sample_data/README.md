# Sample data

## Small corpus

`small/` is the committed default dataset. It contains 16 self-authored synthetic enterprise documents and a labeled evaluation set. It covers exact identifiers, semantic paraphrases, procedure-oriented long-context questions, and text/table lookups. Use it for the first runnable workflow and reproducible chapter tables.

## Generated large corpus

Generate a larger deterministic corpus only for stress testing or latency measurements:

```bash
uv run python scripts/generate_large_corpus.py --documents 200 --seed 42
```

The command writes to `sample_data/generated_large/`, which is ignored by Git. It does not call an LLM or remote API. The seed makes the generated corpus repeatable.
