# 3.2 Semantic bridging with native local BGE-M3

This stage uses `FlagEmbedding.BGEM3FlagModel` to encode the corpus and
queries once, requesting all three of BGE-M3's native representations in a
single pass: dense vectors, sparse lexical weights, and ColBERT
multi-vectors. It writes one row per (query, method) to
`results/<dataset>/bge_m3_representations.csv` so each representation's retrieval quality can be
compared directly against the 3.1 naive baseline.

## Why this stage requires the `local-bge-m3` extra

Every hosted embedding endpoint (Ollama, OpenAI-compatible providers used in
`naive_baseline/`) can only return BGE-M3 **dense** vectors. Sparse lexical
weights and ColBERT multi-vectors are not exposed by any serving layer we
verified (Ollama, Hugging Face Text Embeddings Inference); they exist only
through the native `FlagEmbedding.BGEM3FlagModel` Python interface. That is
why this stage is gated behind:

```bash
uv sync --extra local-bge-m3
```

Running `naive_baseline/run.py` with `DENSE_PROVIDER=ollama` or `openai`
does **not** give you sparse or ColBERT signals; only this stage does.

## What each method column means

`results/<dataset>/bge_m3_representations.csv` contains four rows per query:

| `method` | Signal | How it is scored |
|---|---|---|
| `bge_m3_dense` | Dense vector similarity | Inner product of pre-normalized dense vectors (equivalent to cosine similarity). |
| `bge_m3_sparse` | Lexical/token weight overlap | `model.compute_lexical_matching_score()` on the token-weight dictionaries. |
| `bge_m3_colbert` | Multi-vector late interaction | `model.colbert_score()`, BGE-M3's own max-similarity ColBERT scorer. |
| `bge_m3_hybrid` | Equal-weight combination | `1/3 * dense + 1/3 * sparse + 1/3 * colbert`, following BGE-M3's own documented hybrid-ranking formula. No additional normalization or fusion logic (RRF, weighted fusion across separate systems) is applied here, as that belongs to Stage 3.3. |

## HyDE query expansion (optional)

HyDE is off by default. Basic retrieval runs never require an LLM key.

```bash
uv run python semantic_bridging_bge_m3/run.py --dataset small
uv run python semantic_bridging_bge_m3/run.py --dataset small --use-hyde
```

If `--use-hyde` is passed without `OPENAI_API_KEY` set in `.env`, the script
prints one clear message and falls back to the raw query for the entire run,
never failing or silently skipping queries.

## Configuration

All settings come from the same `.env` file used across the repository (see
`.env.example`). This stage reads:

| Variable | Default | Meaning |
|---|---|---|
| `LOCAL_BGE_M3_MODEL_NAME` | `BAAI/bge-m3` | Model passed to `BGEM3FlagModel`. |
| `RETRIEVAL_TOP_K` | `3` | Default `k` for hit@k; override with `--top-k`. |
| `OPENAI_API_KEY` | unset | Only used when `--use-hyde` is passed. |
| `HYDE_MODEL` | `gpt-4o-mini` | Chat model used to generate the hypothetical passage. |

## Running

```bash
uv sync --extra local-bge-m3
cp .env.example .env   # if not already done
uv run python semantic_bridging_bge_m3/run.py --dataset small
```

The first run downloads the BGE-M3 model weights (roughly 2.27 GB); this is
why this stage is not part of the default Codespaces path.

## Output

Writes to `results/<dataset>/bge_m3_representations.csv` (e.g. `results/small/bge_m3_representations.csv`), recording one row per (query, method) pair for all four representations (`bge_m3_dense`, `bge_m3_sparse`, `bge_m3_colbert`, `bge_m3_hybrid`).

For full cross-stage benchmark comparisons and analysis, see [**`results/benchmark_results.md`**](../results/benchmark_results.md).

