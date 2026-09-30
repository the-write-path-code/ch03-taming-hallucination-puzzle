# Chapter 3 Retrieval Architecture & Workflows

This document provides architectural flowcharts and execution diagrams for each stage of Chapter 3, detailing how synthetic data, embedding models, fusion algorithms, and vector databases connect together.

---

## 1. Chapter-Stage Progression

The chapter is structured in four progressive stages. Each stage isolates a specific retrieval concept or addresses a concrete failure mode identified in the preceding step before storing results in a single consolidated directory.

```mermaid
%%{init: {"theme": "neutral", "themeVariables": {"fontFamily": "Arial, Helvetica, sans-serif", "fontSize": "11px", "actorFontSize": "11px", "noteFontSize": "10px", "messageFontSize": "10px"}}}%%
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
%%{init: {"theme": "neutral", "themeVariables": {"fontFamily": "Arial, Helvetica, sans-serif", "fontSize": "11px", "actorFontSize": "11px", "noteFontSize": "10px", "messageFontSize": "10px"}}}%%
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
%%{init: {"theme": "base", "themeVariables": {"fontFamily": "Arial, Helvetica, sans-serif", "fontSize": "22px", "primaryTextColor": "#000000", "lineColor": "#4B5563"}}}%%
flowchart TD
    Env[".env (DENSE_PROVIDER)"] --> Cfg["retrieval_core.config"]
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
        F1["Exact Identifiers (e.g. AX4-E117) &mdash; Diluted in embedding space"]
        F2["Table Lookups &mdash; Lack surrounding prose context"]
        F3["Long Context &mdash; Target facts buried in 1,000+ words"]
    end
    
    Rank -.-> Failures

    classDef cfgNode fill:#F3F4F6,stroke:#4B5563,color:#000000,stroke-width:1.5px
    classDef procNode fill:#EBF5FF,stroke:#2563EB,color:#000000,stroke-width:1.5px
    classDef scoreNode fill:#EDE9FE,stroke:#7C3AED,color:#000000,stroke-width:1.5px
    classDef evalNode fill:#DCFCE7,stroke:#15803D,color:#000000,stroke-width:1.5px
    classDef failNode fill:#FEE2E2,stroke:#DC2626,color:#000000,stroke-width:1.5px

    class Env,Cfg cfgNode
    class Docs,Query,Embed,DocVecs,QVec procNode
    class CosSim,Rank scoreNode
    class Eval evalNode
    class F1,F2,F3 failNode
```

---

## 4. Stage 3.2: BGE-M3 Multi-Representation Flow

Native BGE-M3 generates three distinct representations in a single forward pass through `FlagEmbedding`.

```mermaid
%%{init: {"theme": "base", "themeVariables": {"fontFamily": "Arial, Helvetica, sans-serif", "fontSize": "15px", "primaryTextColor": "#000000", "lineColor": "#4B5563"}}}%%
flowchart TD
    Text["Input Text (Document or Query)"] --> Model["BGEM3FlagModel (FlagEmbedding)"]
    
    Model --> V1["Dense Vector<br/>(1024-dim)"]
    Model --> V2["Sparse Lexical Weights<br/>(Token Dictionary)"]
    Model --> V3["ColBERT Multi-Vectors<br/>(seq_len &times; 1024)"]
    
    V1 --> S1["Dense Cosine<br/>Score"]
    V2 --> S2["Lexical Matching<br/>Score (score_sparse)"]
    V3 --> S3["ColBERT MaxSim<br/>Score (score_colbert)"]
    
    S1 --> RawAvg["Raw Score Equal-Weight Average (bge_m3_hybrid)"]
    S2 --> RawAvg
    S3 --> RawAvg
    
    RawAvg --> Note["Note: Raw score scales are uncalibrated; motivated Stage 3.3"]

    classDef inputNode fill:#F3F4F6,stroke:#4B5563,color:#000000,stroke-width:1.5px
    classDef modelNode fill:#EDE9FE,stroke:#7C3AED,color:#000000,stroke-width:1.5px
    classDef repNode fill:#EBF5FF,stroke:#2563EB,color:#000000,stroke-width:1.5px
    classDef scoreNode fill:#FEF9C3,stroke:#CA8A04,color:#000000,stroke-width:1.5px
    classDef avgNode fill:#DCFCE7,stroke:#15803D,color:#000000,stroke-width:1.5px
    classDef noteNode fill:#FFFBEB,stroke:#D97706,color:#000000,stroke-width:1.5px

    class Text inputNode
    class Model modelNode
    class V1,V2,V3 repNode
    class S1,S2,S3 scoreNode
    class RawAvg avgNode
    class Note noteNode
```

---

## 5. Stage 3.3: Independent Hybrid Retrieval Flow

Stage 3.3 decouples dense and sparse retrieval into independent legs and combines them using formal fusion algorithms.

```mermaid
%%{init: {"theme": "base", "themeVariables": {"fontFamily": "Arial, Helvetica, sans-serif", "fontSize": "28px", "primaryTextColor": "#000000", "lineColor": "#4B5563"}}}%%
flowchart TD
    Q["Evaluation Query"] --> DLeg["Dense Leg<br/>(Cosine Sim)"]
    Q --> SLeg["Sparse Leg<br/>(BM25 / Sparse)"]
    
    DLeg --> DScore["Dense Score<br/>Map & Ranks"]
    SLeg --> SScore["Sparse Score<br/>Map & Ranks"]
    
    DScore --> RRF["Reciprocal Rank<br/>Fusion (RRF)"]
    SScore --> RRF
    
    DScore --> RSF["Relative Score<br/>Fusion (RSF)"]
    SScore --> RSF
    
    subgraph Normalization ["RSF Score Normalization"]
        N1["Dense: (s &minus; min)<br/>/ (max &minus; min)"]
        N2["Sparse: (s &minus; min)<br/>/ (max &minus; min)"]
    end
    
    RSF --- Normalization
    
    RRF --> Out1["rrf_dense_bm25<br/>results"]
    RSF --> Out2["relative_score_<br/>dense_bm25 results"]

    classDef inputNode fill:#F3F4F6,stroke:#4B5563,color:#000000,stroke-width:1.5px
    classDef legNode fill:#EBF5FF,stroke:#2563EB,color:#000000,stroke-width:1.5px
    classDef scoreNode fill:#EDE9FE,stroke:#7C3AED,color:#000000,stroke-width:1.5px
    classDef fusionNode fill:#FEF3C7,stroke:#D97706,color:#000000,stroke-width:1.5px
    classDef normNode fill:#FEF9C3,stroke:#CA8A04,color:#000000,stroke-width:1.5px
    classDef outNode fill:#DCFCE7,stroke:#15803D,color:#000000,stroke-width:1.5px

    class Q inputNode
    class DLeg,SLeg legNode
    class DScore,SScore scoreNode
    class RRF,RSF fusionNode
    class N1,N2 normNode
    class Out1,Out2 outNode
```

---

## 6. Stage 3.4: High-Fidelity Qdrant Deployment & Retrieval Flow

Stage 3.4 integrates persistent vector storage with payload metadata filtering and optional second-stage ColBERT reranking.

```mermaid
%%{init: {"theme": "base", "themeVariables": {"fontFamily": "Arial, Helvetica, sans-serif", "fontSize": "48px", "primaryTextColor": "#000000", "lineColor": "#4B5563"}}}%%
flowchart TD
    subgraph ClientFactory ["Single Connection Abstraction (client.py)"]
        direction TB
        EnvQ[".env Settings"] --> CF["get_qdrant_client()"]
        CF -->|QDRANT_URL unset| Emb["Embedded Local Qdrant<br/>(./.qdrant_local)"]
        CF -->|QDRANT_URL set| Rem["Remote Docker /<br/>Qdrant Cloud"]
    end

    subgraph Collection ["Collection & Schema (collection.py)"]
        direction TB
        Docs["Corpus Documents"] --> PGen["Generate Points<br/>(dense vector + metadata)"]
        PGen --> Col["Qdrant Collection<br/>(named 'dense' vector)"]
        Col --> Idx["Payload Indexes<br/>(document_type, failure_tags)"]
    end

    subgraph Search ["Two-Stage Retrieval (run.py)"]
        direction LR
        subgraph Stage1 ["First-Stage Retrieval"]
            direction TB
            Q["Query"] --> QSearch["client.query_points()"]
            Flt["Optional Filter<br/>(document_type / failure_tags)"] --> QSearch
            QSearch --> Cand["Top 20 Finalist Candidates<br/>(qdrant_dense)"]
        end
        subgraph Stage2 ["Second-Stage Precision Rerank"]
            direction TB
            RerankCheck{"local-bge-m3<br/>available?"}
            RerankCheck -->|Yes| ColBERT["ColBERT Token MaxSim Rerank<br/>(qdrant_dense_colbert_rerank)"]
            RerankCheck -->|No| Skip["Skip Reranking"]
        end
        Stage1 --> Stage2
    end

    ClientFactory --> Search
    Collection --> Search

    classDef clientNode fill:#EBF5FF,stroke:#2563EB,color:#000000,stroke-width:1.5px
    classDef colNode fill:#EDE9FE,stroke:#7C3AED,color:#000000,stroke-width:1.5px
    classDef searchNode fill:#FEF3C7,stroke:#D97706,color:#000000,stroke-width:1.5px
    classDef decision fill:#FEF9C3,stroke:#CA8A04,color:#000000,stroke-width:1.5px
    classDef rerankNode fill:#DCFCE7,stroke:#15803D,color:#000000,stroke-width:1.5px

    class EnvQ,CF,Emb,Rem clientNode
    class Docs,PGen,Col,Idx colNode
    class Q,Flt,QSearch,Cand searchNode
    class RerankCheck decision
    class ColBERT,Skip rerankNode
```

---

## 7. Configuration Ownership

All user-configurable options are centralized in the root `.env` file and loaded through `retrieval_core.config`. Application code never reads `os.environ` directly.

```mermaid
%%{init: {"theme": "neutral", "themeVariables": {"fontFamily": "Arial, Helvetica, sans-serif", "fontSize": "11px", "actorFontSize": "11px", "noteFontSize": "10px", "messageFontSize": "10px"}}}%%
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
