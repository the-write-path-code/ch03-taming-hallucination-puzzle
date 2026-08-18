# 3.4 Building High-Fidelity Retrieval in Qdrant

This stage implements durable, production-ready vector storage and search using
**Qdrant**, combining:
1. **Durable Vector Storage**: Persistent embedded local storage (default) or remote/Docker/Cloud deployment.
2. **Structured Payload Metadata & Filtering**: Structured payload indexes for `document_type` and `failure_tags` to enforce fast search constraints.
3. **Live Schema Inspection**: Exports actual live collection configuration to `collection_config.json`.
4. **Second-Stage ColBERT Reranking**: Re-scores top candidate finalists with native token-level MaxSim multi-vectors (when `local-bge-m3` is installed).

---

## Connection Architecture

A single unified client factory in [`qdrant_high_fidelity/client.py`](file:///Users/mohit/Documents/GitHub/ch03-taming-hallucination-puzzle/qdrant_high_fidelity/client.py) manages all connection modes:

- **Embedded Local Mode (Default)**:
  - When `QDRANT_URL` is empty, initializes `QdrantClient(path=QDRANT_LOCAL_PATH)` (persisted at `./.qdrant_local`).
  - Never uses in-memory mode (`:memory:`) so indexed documents persist across process restarts.
  - Zero external processes or Docker daemon required.
- **Docker / Remote / Qdrant Cloud Mode**:
  - When `QDRANT_URL` is set (e.g. `http://localhost:6333` or cloud URL), connects via HTTP/gRPC.
  - Passes `QDRANT_API_KEY` if configured.
  - Docker Compose is available via `docker compose up -d` for users wanting the Qdrant Web UI.

---

## Collection & Payload Schema

Collections are dataset-scoped (e.g., `ch03_retrieval_small`, `ch03_retrieval_generated_large`).

### Vector Schema
- **`dense`** (Mandatory): Cosine similarity vector matching the active dense embedding model dimensions (e.g., 1024 for BGE-M3, 1536 for OpenAI `text-embedding-3-small`).
- **`bge_m3_sparse`** (Optional): Named sparse lexical vector when native `local-bge-m3` is enabled.

### Payload Schema
Every point contains structured metadata:
- `doc_id`: Unique document identifier stem.
- `source_path`: Relative filesystem path to source Markdown document.
- `document_type`: Categorical document type (`policy`, `product`, `operations`, `pricing`, `specification`).
- `failure_tags`: Distinct list of labeled retrieval failure modes from the evaluation query suite.

Payload indexes (`KEYWORD`) are created for `document_type` and `failure_tags`.

---

## Second-Stage ColBERT Reranking

Dense retrieval narrows the corpus down to the top M candidates (default 20). 

When native `local-bge-m3` is available, `run.py` extracts token-level ColBERT multi-vectors for the query and candidates, computing fine-grained token-to-token MaxSim alignment:

$$\text{Score}_{\text{ColBERT}}(Q, D) = \sum_{i \in Q} \max_{j \in D} (E_i \cdot E_j)$$

This reranks the finalists, dramatically improving precision on specific technical clauses and long-context facts.

---

## Configuration

All settings live centrally in `.env`:

| Variable | Default | Description |
|---|---|---|
| `QDRANT_LOCAL_PATH` | `./.qdrant_local` | Directory for persistent embedded local storage. |
| `QDRANT_URL` | unset | Remote server URL (e.g. `http://localhost:6333` or cloud endpoint). |
| `QDRANT_API_KEY` | unset | Optional API key for remote / Qdrant Cloud clusters. |
| `QDRANT_COLLECTION_PREFIX` | `ch03_retrieval` | Prefix for dataset-scoped collection names. |
| `QDRANT_DENSE_VECTOR_NAME` | `dense` | Name of the dense vector field in Qdrant. |
| `QDRANT_RERANK_CANDIDATES` | `20` | Candidate pool size retrieved from Qdrant for second-stage reranking. |
| `RETRIEVAL_TOP_K` | `3` | Default `k` for Hit@k evaluation. |

---

## Running

### 1. Basic Dense Run (Persisted Embedded Qdrant)

```bash
uv sync
uv run python qdrant_high_fidelity/run.py --dataset small
```

To force recreation and re-indexing of the collection:
```bash
uv run python qdrant_high_fidelity/run.py --dataset small --rebuild
```

### 2. Metadata Filtering Demonstration

Apply structural metadata constraints during search:
```bash
# Filter only policy documents
uv run python qdrant_high_fidelity/run.py --dataset small --filter-document-type policy

# Filter by failure tag
uv run python qdrant_high_fidelity/run.py --dataset small --filter-failure-tag table_lookup
```

### 3. Full Local BGE-M3 with ColBERT Reranking

```bash
uv sync --extra local-bge-m3
uv run python qdrant_high_fidelity/run.py --dataset small --rebuild
```

Emits two evaluation methods into `results/<dataset>/qdrant_high_fidelity.csv`:
- `qdrant_dense`: First-stage Qdrant dense vector search.
- `qdrant_dense_colbert_rerank`: Second-stage ColBERT reranking over top candidates.

### 4. Stress-Testing on the 200-Document Corpus

```bash
uv run python scripts/generate_large_corpus.py --documents 200 --seed 42
uv run python qdrant_high_fidelity/run.py --dataset generated_large --rebuild
```

---

## Output Artifacts

- **Results CSV**: `results/<dataset>/qdrant_high_fidelity.csv` (e.g. `results/small/qdrant_high_fidelity.csv`)
- **Live Schema Config**: `results/<dataset>/qdrant_collection_config.json`
- **Canonical Benchmark Report**: [**`results/benchmark_results.md`**](../results/benchmark_results.md)

