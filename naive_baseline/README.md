# 3.1 Naive baseline: why dense-only retrieval fails

This stage implements a genuinely **dense-only** retrieval pipeline: no BM25,
no BGE-M3 sparse weights, no reranking, and no Qdrant hybrid fusion. Later
stages (3.2, 3.3, 3.4) build on top of this baseline to show what each added
technique fixes.

## Why exact identifiers and table values fail here

Dense embedding models are trained to capture semantic meaning, not exact
tokens. Two consequences show up directly in `naive_results.csv`:

- **Exact identifiers** such as `AX4-E117` or `CC-17` are short, low-frequency
  tokens. A dense encoder maps them close to other short alphanumeric strings
  in embedding space, not to the one document that defines them. A query like
  "What does AX4-E117 mean?" can retrieve documents about unrelated fault
  codes because the *shape* of the token, not its specific identity, drives
  the embedding.
- **Table values** are usually short numeric or categorical facts (a price, a
  retention period, a temperature limit) surrounded by very little
  descriptive prose. Dense retrieval relies on contextual language to place a
  passage correctly in embedding space; a bare table row does not give it
  enough signal, so table-lookup queries in the evaluation set are the
  failure mode most likely to score `hit_at_1 = 0` in `naive_results.csv`.

Both problems are addressed by later stages: sparse/lexical retrieval (3.2,
3.3) recovers exact-token matches, and hybrid fusion plus reranking (3.3, 3.4)
recovers table and long-context facts by combining signals rather than
relying on dense similarity alone.

## Configuration

All configuration lives in a single `.env` file at the project root (copy
`.env.example` to `.env`). Nothing else needs to be edited to change the
embedding provider or model.

| Variable | Default | Meaning |
|---|---|---|
| `DENSE_PROVIDER` | `ollama` | One of `ollama`, `openai`, `local-bge-m3`. |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Local Ollama, or `https://ollama.com` for Ollama Cloud. |
| `OLLAMA_API_KEY` | unset | Required only when `OLLAMA_BASE_URL` is not local (e.g. Ollama Cloud). |
| `OLLAMA_EMBED_MODEL` | `bge-m3` | Ollama serves BGE-M3 as dense-only; see note below. |
| `OPENAI_API_KEY` | unset | Required only when `DENSE_PROVIDER=openai`. |
| `OPENAI_BASE_URL` | unset (official OpenAI) | Point at any OpenAI-compatible embeddings endpoint. |
| `OPENAI_EMBED_MODEL` | `text-embedding-3-small` | Any embedding model your `OPENAI_BASE_URL` serves. |
| `LOCAL_BGE_M3_MODEL_NAME` | `BAAI/bge-m3` | Only used when `DENSE_PROVIDER=local-bge-m3`. |
| `RETRIEVAL_TOP_K` | `3` | Default `k` for hit@k; override per run with `--top-k`. |

### On BGE-M3 and hosted providers

Ollama, and any OpenAI-compatible hosted endpoint, can only return **dense**
embeddings for BGE-M3. Neither exposes the model's native sparse lexical
weights or ColBERT multi-vectors -- that capability exists only through the
`local-bge-m3` extra using `FlagEmbedding.BGEM3FlagModel` directly, which is
what Stage 3.2 (`semantic_bridging_bge_m3/`) uses. This stage never claims
otherwise.

### Provider notes

- **`ollama`** (default) works with a local Ollama install (`ollama pull
  bge-m3`, no key needed) or with Ollama Cloud (set `OLLAMA_BASE_URL` and
  `OLLAMA_API_KEY`).
- **`openai`** requires `OPENAI_API_KEY`. Point `OPENAI_BASE_URL` at any
  OpenAI-compatible provider if you want a different hosted embedding model.
- **`local-bge-m3`** requires `uv sync --extra local-bge-m3` and runs
  BGE-M3 locally in dense-only mode (no sparse or ColBERT vectors are
  requested at this stage).

## Running

```bash
uv sync
cp .env.example .env   # then edit DENSE_PROVIDER / keys as needed
uv run python naive_baseline/run.py --dataset small
```

Optional flags:

```bash
uv run python naive_baseline/run.py --dataset small --top-k 5
uv run python naive_baseline/run.py --dataset generated_large --output /tmp/naive_large.csv
```

## Output

Writes `naive_baseline/naive_results.csv` with columns: `query_id`, `query`,
`expected_doc_ids`, `retrieved_doc_ids`, `hit_at_1`, `hit_at_k`,
`failure_mode`, `method`. The `method` column records the exact provider used
(e.g. `dense_only:ollama`) so later stages' CSVs can be compared side by side.

Also generates a companion summary report (`naive_baseline/naive_results.md` or alongside the custom `--output` path) detailing overall and per-failure-mode hit rates.

## Benchmark Results

### Dense-Only Baseline Performance (BGE-M3 via Ollama)

| Dataset | Documents | Total Queries | Hit@1 Rate | Hit@3 Rate |
|---|---|---|---|---|
| **Small Corpus** | 16 | 16 | **93.8%** | **100.0%** |
| **Generated Large Corpus** | 200 | 32 | **12.5%** | **21.9%** |

### Breakdown by Failure Mode (Large Corpus)

| Failure Mode | Queries | Hit@1 Rate | Hit@3 Rate | Failure Mechanism |
|---|---|---|---|---|
| `keyword_exact_match` | 8 | 12.5% | 12.5% | Short identifiers map to similar-looking tokens rather than exact documents |
| `long_context` | 8 | 0.0% | 25.0% | Key facts in long docs get diluted by surrounding distractor prose |
| `semantic_paraphrase` | 8 | 12.5% | 12.5% | Distractor documents sharing general vocabulary outrank the paraphrased target |
| `table_lookup` | 8 | 25.0% | 37.5% | Markdown table rows lack surrounding narrative context for dense vectors |
| **Total / Overall** | **32** | **12.5%** | **21.9%** | **Dense-only baseline failure** |

