# Chapter 3 Retrieval Architecture & Workflows

This document provides architectural flowcharts and execution diagrams for each stage of Chapter 3, detailing how synthetic data, embedding models, fusion algorithms, and vector databases connect together.

---

## 1. Chapter-Stage Progression

The chapter is structured in four progressive stages. Each stage isolates a specific retrieval concept or addresses a concrete failure mode identified in the preceding step before storing results in a single consolidated directory.

```mermaid
flowchart LR
    A["Synthetic Data (Small / Large)"] --> B["Stage 3.1: Dense-Only Baseline"]
    B --> C["Stage 3.2: BGE-M3 Multi-Representation"]
    C --> D["Stage 3.3: Independent Hybrid Fusion"]
    D --> E["Stage 3.4: High-Fidelity Qdrant"]
    E --> F["Centralized Results (results/)"]
```

- **Stage 3.1 (`naive_baseline/`)**: Demonstrates where single-vector dense retrieval fails on exact codes and tabular figures.
- **Stage 3.2 (`semantic_bridging_bge_m3/`)**: Demonstrates multi-vector representation (dense, sparse lexical weights, ColBERT) from a single model.
- **Stage 3.3 (`hybrid_retrieval/`)**: Demonstrates rank-based (RRF) and normalized-score (RSF) fusion of independent retrieval legs.
- **Stage 3.4 (`qdrant_high_fidelity/`)**: Demonstrates durable storage, payload metadata filtering, and two-stage ColBERT candidate reranking in Qdrant.

---

## 2. Dataset Lifecycle and Evaluation Workflow

The evaluation pipeline supports both a small committed test dataset and a large synthetic stress-testing corpus generated deterministically on local machines.

```mermaid
flowchart TD
    subgraph Datasets ["Dataset Sources"]
        S1["sample_data/small/ (Committed)"]
        S2["scripts/generate_large_corpus.py"]
        S3["sample_data/generated_large/ (Git Ignored)"]
        S2 -->|Deterministic Seed 42| S3
    end

    subgraph Core ["Shared Evaluation Core (retrieval_core)"]
        C1["load_corpus()"]
        C2["load_eval_queries()"]
        C3["validate_against_corpus()"]
        C4["build_result() & summarize()"]
    end

    subgraph Execution ["Stage Execution & Output"]
        R1["Stage Runner (run.py)"]
        O1["results/<dataset>/<stage_output>.csv"]
        O2["results/benchmark_results.md"]
    end

    S1 --> C1
    S1 --> C2
    S3 --> C1
    S3 --> C2
    C1 --> C3
    C2 --> C3
    C3 --> R1
    R1 --> C4
    C4 --> O1
    O1 --> O2
```

The large corpus contains 200 documents and 32 labeled queries. It is excluded from version control via `.gitignore` and rebuilt locally whenever needed.

---

## 3. Stage 3.1: Dense-Only Retrieval Flow

Dense-only retrieval embeds documents and queries into fixed-size semantic vectors. The diagram below illustrates how configuration flows into embedding generation and highlights the primary failure modes.

```mermaid
flowchart TD
    Env[".env (DENSE_PROVIDER: ollama / openai / local-bge-m3)"] --> Cfg["retrieval_core.config"]
    Cfg --> Embed["embed_texts()"]
    
    Docs["Corpus Documents"] --> Embed
    Query["User Query"] --> Embed
    
    Embed --> DocVecs["Document Vectors"]
    Embed --> QVec["Query Vector"]
    
    DocVecs --> CosSim["Cosine Similarity Scoring"]
    QVec --> CosSim
    
    CosSim --> Rank["Top-K Ranking"]
    Rank --> Eval["Hit@K Evaluation"]
    
    subgraph Failures ["Known Dense-Only Failure Modes"]
        F1["Exact Identifiers (e.g. AX4-E117) - Diluted in embedding space"]
        F2["Table Lookups - Lack surrounding prose context"]
        F3["Long Context - Target facts buried in 1,000+ words"]
    end
    
    Rank -.-> Failures
```

---

## 4. Stage 3.2: BGE-M3 Multi-Representation Flow

Native BGE-M3 generates three distinct representations in a single forward pass through `FlagEmbedding`.

```mermaid
flowchart TD
    Text["Input Text (Document or Query)"] --> Model["BGEM3FlagModel (FlagEmbedding)"]
    
    Model --> V1["Dense Vector (1024-dim)"]
    Model --> V2["Sparse Lexical Weights (Token Dictionary)"]
    Model --> V3["ColBERT Multi-Vectors (seq_len x 1024)"]
    
    V1 --> S1["Dense Cosine Score"]
    V2 --> S2["Lexical Matching Score (score_sparse)"]
    V3 --> S3["ColBERT MaxSim Score (score_colbert)"]
    
    S1 --> RawAvg["Raw Score Equal-Weight Average (bge_m3_hybrid)"]
    S2 --> RawAvg
    S3 --> RawAvg
    
    RawAvg --> Note["Note: Raw score scales are uncalibrated; motivated Stage 3.3"]
```

---

## 5. Stage 3.3: Independent Hybrid Retrieval Flow

Stage 3.3 decouples dense and sparse retrieval into independent legs and combines them using formal fusion algorithms.

```mermaid
flowchart TD
    Q["Evaluation Query"] --> DLeg["Dense Leg (Cosine Similarity)"]
    Q --> SLeg["Sparse Leg (BM25 or BGE-M3 Sparse)"]
    
    DLeg --> DScore["Dense Score Map / Ranks"]
    SLeg --> SScore["Sparse Score Map / Ranks"]
    
    DScore --> RRF["Reciprocal Rank Fusion (RRF)"]
    SScore --> RRF
    
    DScore --> RSF["Relative Score Fusion (RSF)"]
    SScore --> RSF
    
    subgraph Normalization ["RSF Min-Max Normalization"]
        N1["Dense: (score - min) / (max - min)"]
        N2["Sparse: (score - min) / (max - min)"]
    end
    
    RSF --- Normalization
    
    RRF --> Out1["rrf_dense_bm25 results"]
    RSF --> Out2["relative_score_dense_bm25 results"]
```

---

## 6. Stage 3.4: High-Fidelity Qdrant Deployment & Retrieval Flow

Stage 3.4 integrates persistent vector storage with payload metadata filtering and optional second-stage ColBERT reranking.

```mermaid
flowchart TD
    subgraph ClientFactory ["Single Connection Abstraction (client.py)"]
        EnvQ[".env Settings"] --> CF["get_qdrant_client()"]
        CF -->|QDRANT_URL unset| Emb["Embedded Local Qdrant (./.qdrant_local)"]
        CF -->|QDRANT_URL set| Rem["Remote Docker / Qdrant Cloud"]
    end

    subgraph Collection ["Collection & Schema (collection.py)"]
        Docs["Corpus Documents"] --> PGen["Generate Points (dense vector + metadata)"]
        PGen --> Col["Qdrant Collection (named 'dense' vector)"]
        Col --> Idx["Payload Indexes (document_type, failure_tags)"]
    end

    subgraph Search ["Two-Stage Retrieval (run.py)"]
        Q["Query"] --> QSearch["client.query_points()"]
        Flt["Optional Filter (document_type / failure_tags)"] --> QSearch
        QSearch --> Cand["Top 20 Finalist Candidates (qdrant_dense)"]
        Cand --> RerankCheck{"local-bge-m3 available?"}
        RerankCheck -->|Yes| ColBERT["ColBERT Token MaxSim Rerank (qdrant_dense_colbert_rerank)"]
        RerankCheck -->|No| Skip["Skip Reranking"]
    end
```

---

## 7. Configuration Ownership

All user-configurable options are centralized in the root `.env` file and loaded through `retrieval_core.config`. Application code never reads `os.environ` directly.

```mermaid
flowchart TD
    DotEnv[".env File (Root)"] --> Loader["src/retrieval_core/config.py"]
    
    Loader --> DCfg["DenseProviderConfig (DENSE_PROVIDER, OLLAMA_*, OPENAI_*)"]
    Loader --> HCfg["HybridConfig (HYBRID_SPARSE_METHOD, RRF_K, HYBRID_*_WEIGHT)"]
    Loader --> QCfg["QdrantConfig (QDRANT_LOCAL_PATH, QDRANT_URL, QDRANT_*)"]
    
    DCfg --> S1["Stage 3.1 (naive_baseline)"]
    DCfg --> S2["Stage 3.2 (semantic_bridging_bge_m3)"]
    HCfg --> S3["Stage 3.3 (hybrid_retrieval)"]
    QCfg --> S4["Stage 3.4 (qdrant_high_fidelity)"]
```
