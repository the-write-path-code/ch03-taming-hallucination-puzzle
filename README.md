# Chapter 3: Taming the Hallucination Puzzle

Companion code for *Building Safe Agentic AI for Enterprise Systems* by Mohit Aggarwal.

This repository measures how retrieval design affects the evidence available to a retrieval-augmented generation (RAG) system. It begins with dense-only retrieval, then adds BGE-M3 multi-representation retrieval, explicit dense-and-sparse fusion, and persistent Qdrant retrieval with optional reranking.

The repository does not generate a final answer or decide whether an answer is safe to deliver. It ranks documents and measures whether the expected source appears in the returned results. Chapter 4 adds the evaluation and policy gates that turn retrieval evidence into an answer decision.

## What You Will Run

| Chapter section | Stage | What it tests |
| --- | --- | --- |
| 3.1 | Dense-only baseline | Whether one dense embedding representation can retrieve exact identifiers, paraphrases, long-context facts, and table values. |
| 3.2 | Semantic bridging with BGE-M3 | Dense vectors, sparse lexical weights, and ColBERT token-level vectors produced by one local model. |
| 3.3 | Hybrid dense and sparse retrieval | Independent dense and lexical legs combined by reciprocal rank fusion (RRF) or relative score fusion (RSF). |
| 3.4 | High-fidelity Qdrant retrieval | Persistent local or remote Qdrant storage, metadata filters, and optional ColBERT reranking over a bounded candidate set. |

The supplied corpus is synthetic and labeled. Every query names an expected source document, so the repository can measure retrieval rather than rely on an answer that merely sounds plausible.

## Production Warning

A retrieval result is evidence, not a final answer. A document can rank highly while being outdated, incomplete, or scoped to a different policy, product, or jurisdiction. This repository measures whether retrieval found the expected source. It does not establish claim grounding, evidence sufficiency, or whether a generated answer should reach a user.

Do not treat Hit@k, Recall@k, or a high similarity score as an approval decision. Use the evaluation and policy-gate patterns in Chapter 4 before connecting retrieval to a consequential workflow.

## Prerequisites

- Git
- [uv](https://docs.astral.sh/uv/)
- Python 3.11
- A local Ollama installation for the default dense-retrieval path, or an OpenAI-compatible embedding endpoint
- At least 3 GB of free disk space if you intend to run the optional local BGE-M3 stage

The repository uses embedded Qdrant storage by default. Docker, a Qdrant server, and a Qdrant Cloud account are optional.

## Quick Start

### 1. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Clone and synchronize the repository

```bash
git clone https://github.com/the-write-path-code/ch03-taming-hallucination-puzzle.git
cd ch03-taming-hallucination-puzzle
uv sync
```

The repository includes `uv.lock`. Run `uv sync` after pulling changes so the environment matches the committed dependency set.

### 3. Start the default local embedding provider

The default configuration uses Ollama and the BGE-M3 embedding model:

```bash
ollama serve
ollama pull bge-m3
```

Keep `ollama serve` running in a separate terminal.

### 4. Create local configuration

```bash
cp .env.example .env
```

The default `.env.example` uses embedded Qdrant and local Ollama. You can run the dense-only baseline without adding an API key.

## Configuration

All user-configurable retrieval settings live in `.env`. Application code loads configuration through the shared `retrieval_core.config` module.

### Default local path

```dotenv
DENSE_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_EMBED_MODEL=bge-m3
QDRANT_LOCAL_PATH=.qdrant_local
RETRIEVAL_TOP_K=3
HYBRID_SPARSE_METHOD=bm25
HYBRID_FUSION_METHOD=rrf
RRF_K=60
```

This path uses local Ollama embeddings, embedded persistent Qdrant, and BM25 for the sparse leg. It does not need an API key.

### OpenAI-compatible embeddings, optional

Set the provider and key in `.env`:

```dotenv
DENSE_PROVIDER=openai
OPENAI_API_KEY=your-key
OPENAI_EMBED_MODEL=text-embedding-3-small
```

Set `OPENAI_BASE_URL` only when you are using a compatible endpoint rather than the public OpenAI API.

### Local BGE-M3, optional

The native BGE-M3 stage exposes sparse weights and ColBERT multi-vectors that hosted embedding APIs do not expose in this repository.

```bash
uv sync --extra local-bge-m3
```

The first run downloads the local model weights, roughly 2.27 GB. This stage is intentionally optional because it increases disk use and install time.

### Qdrant server or cloud deployment, optional

Embedded Qdrant is the default. To run Qdrant in Docker:

```bash
docker compose up -d
```

Then set:

```dotenv
QDRANT_URL=http://localhost:6333
```

For Qdrant Cloud, set `QDRANT_URL` and `QDRANT_API_KEY`. The retrieval code stays the same.

> **Tip**
>
> Start with embedded Qdrant. It persists under `.qdrant_local` and removes a service dependency from the first run.

## Run the Chapter Demonstrations

### 1. Dense-Only Baseline, Section 3.1

Run the small labeled corpus first:

```bash
uv run python naive_baseline/run.py --dataset small
```

This stage uses dense similarity only. It has no lexical retrieval leg, no reranker, and no fusion method. It establishes the failure modes later stages are designed to address:

- Exact identifiers such as `AX4-E117` can be diluted in embedding space.
- Short table facts can be weakly represented by a whole-document embedding.
- Long documents can bury the relevant fact.

### 2. Semantic Bridging with BGE-M3, Section 3.2

Install the optional local dependency group, if you have not already:

```bash
uv sync --extra local-bge-m3
```

Run the stage:

```bash
uv run python semantic_bridging_bge_m3/run.py --dataset small
```

This stage writes one result row per query and representation:

- `bge_m3_dense`
- `bge_m3_sparse`
- `bge_m3_colbert`
- `bge_m3_hybrid`

To test HyDE query expansion, add `--use-hyde`. HyDE is off by default. If `OPENAI_API_KEY` is absent, the runner warns once and continues with the original query rather than failing or silently skipping cases.

### 3. Hybrid Dense and Sparse Retrieval, Section 3.3

Run the hybrid stage on the small corpus:

```bash
uv run python hybrid_retrieval/run.py --dataset small
```

The stage retrieves dense and lexical results independently, then combines them using the configured fusion method:

- `rrf`, reciprocal rank fusion, uses ranked positions only.
- `relative-score`, relative score fusion, normalizes each leg's scores before applying weights.

The base environment uses BM25 as the sparse leg. Set `HYBRID_SPARSE_METHOD=bge-m3-sparse` only after installing the `local-bge-m3` extra.

### 4. High-Fidelity Qdrant Retrieval, Section 3.4

Build or refresh the local collection and run the small corpus:

```bash
uv run python qdrant_high_fidelity/run.py --dataset small --rebuild
```

This stage stores documents as Qdrant points with a dense vector and metadata payloads, including document type and failure tags. It can apply metadata filters and optionally rerank only the bounded candidate set returned by the first-stage search.

Reranking can change the order of retrieved candidates. It cannot recover an expected document that the first-stage search never returned.

### 5. Run the Generated-Large Benchmark, Optional

Generate the deterministic large corpus:

```bash
uv run python scripts/generate_large_corpus.py --documents 200 --seed 42
```

Then run the relevant stages:

```bash
uv run python naive_baseline/run.py --dataset generated-large
uv run python hybrid_retrieval/run.py --dataset generated-large
uv run python qdrant_high_fidelity/run.py --dataset generated-large --rebuild
```

The generated-large corpus contains 200 technical documents and 32 labeled queries. It is intended for stress testing and is not committed to the repository.

## Expected Results

The small corpus contains 16 documents and 16 labeled queries. It is useful for understanding the staged progression, but it is too small to expose every retrieval weakness.

The generated-large corpus is the stronger stress condition. In the recorded benchmark:

| Method | Corpus | Hit@1 | Hit@3 |
| --- | ---: | ---: | ---: |
| Dense-only retrieval | Generated-large | 12.5% | 21.9% |
| BM25-only retrieval | Generated-large | 84.4% | Not reported in the chapter summary |
| RRF dense plus BM25 | Generated-large | 43.8% | 62.5% |
| Relative-score dense plus BM25 | Generated-large | 87.5% | 90.6% |

These results apply to the repository's synthetic corpus, labeled queries, configuration, and evaluation methods. They are not general claims about retrieval models, BM25, RRF, or RSF.

All stage runners write row-level evidence under `results/<dataset>/`. The canonical comparison report is:

```text
results/benchmark_results.md
```

## Run the Tests

Run the repository test suite:

```bash
uv run pytest
```

The tests cover shared corpus loading, result schemas, score fusion, boundary conditions, and retrieval-stage behavior. In particular, the fusion tests cover duplicate identifiers, missing documents, equal score maps, and invalid RRF constants.

## Repository Layout

```text
.
├── README.md
├── pyproject.toml
├── uv.lock
├── .env.example
├── docker-compose.yml                 # Optional local Qdrant server
├── src/retrieval_core/                # Shared corpus, configuration, scoring, and result types
├── naive_baseline/                    # Section 3.1 dense-only runner
├── semantic_bridging_bge_m3/          # Section 3.2 native BGE-M3 runner
├── hybrid_retrieval/                  # Section 3.3 dense and sparse fusion runner
├── qdrant_high_fidelity/              # Section 3.4 persistent Qdrant and reranking runner
├── sample_data/
│   ├── small/                         # Committed 16-document corpus and labels
│   └── README.md                      # Corpus and generated-large instructions
├── scripts/
│   ├── generate_large_corpus.py
│   └── run_all.py
├── results/
│   ├── benchmark_results.md           # Canonical benchmark narrative
│   └── README.md                      # Result schema and reproduction commands
├── docs/
│   ├── workflow.md                    # Retrieval architecture diagrams
│   └── further_reading.md             # Tool and retrieval references
└── tests/
```

## Architecture Diagrams and Supporting Documents

- `docs/workflow.md` contains the dense-only, BGE-M3, hybrid-fusion, Qdrant, and configuration-ownership diagrams used in Chapter 3.
- `naive_baseline/README.md` explains the dense-only failure modes.
- `semantic_bridging_bge_m3/README.md` explains the three BGE-M3 representations and why this stage needs the optional local extra.
- `hybrid_retrieval/README.md` explains RRF and relative score fusion.
- `qdrant_high_fidelity/README.md` explains persistent storage, payload filters, candidate limits, and reranking.
- `results/benchmark_results.md` is the authoritative narrative for the recorded measurements.

## Safety and Operational Limits

- Retrieval ranking does not establish factual correctness, source recency, scope match, or answer safety.
- The synthetic corpus is a controlled teaching artifact. It does not represent a production knowledge base.
- Embedded Qdrant persists locally. Delete `.qdrant_local` if you need a clean collection.
- The optional local BGE-M3 stage downloads a large model and can be slow on CPU-only machines.
- HyDE creates a hypothetical passage for retrieval. It does not verify the passage and does not replace evidence evaluation.
- Reranking only changes the order of documents already returned by first-stage retrieval. It cannot rescue a missing candidate.

## Troubleshooting

### Ollama is not running

Start the service in a separate terminal:

```bash
ollama serve
```

Then confirm that the embedding model is installed:

```bash
ollama pull bge-m3
```

### The local BGE-M3 stage cannot import FlagEmbedding

Install the optional dependency group:

```bash
uv sync --extra local-bge-m3
```

### Qdrant connection fails

If `QDRANT_URL` is unset, the repository uses embedded storage. Remove or comment out `QDRANT_URL` to return to that path. If you intend to use Docker, start the service with:

```bash
docker compose up -d
```

### A generated-large result differs from the recorded table

Confirm the corpus size, seed, provider, top-k value, fusion method, and model configuration before changing code. The benchmark values describe a particular synthetic corpus and run configuration.

### The Qdrant stage finds no expected document

Inspect the first-stage candidate set before changing the reranker. A reranker cannot promote a document that was never retrieved into the candidate pool.

## Related Chapters

- Chapter 2 establishes why deterministic ingestion and localized state boundaries matter before model reasoning begins.
- Chapter 4 measures retrieval, grounding, sufficiency, and policy decisions after retrieval.
- Chapter 5 adds corrective routing and recursive retrieval when initial evidence is incomplete or weak.
- Chapter 6 extends the same retrieval concerns to PDF, image, table, and audio inputs.

## License and Errata

See `LICENSE` for licensing terms. Report documentation or code issues through this repository's GitHub issue tracker.
