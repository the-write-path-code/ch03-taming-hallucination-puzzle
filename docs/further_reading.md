# Further Reading: Beyond BGE-M3

While this repository uses BAAI's **BGE-M3** for dense, sparse lexical, and multi-vector retrieval, modern information retrieval encompasses a broad range of dense and learned sparse architectures. This document outlines prominent alternative architectures, their design trade-offs, and official references.

---

## 1. Dense Embedding Alternatives

Dense encoders map text passages into continuous vector spaces optimized for cosine similarity. Beyond BGE-M3, notable modern dense models include:

### Qwen3-Embedding
- **Category**: Dense Transformer Embedding Model
- **Developer / Organization**: Alibaba Cloud / Qwen Team
- **Why Consider It**: Built on top of the Qwen language model family, Qwen-Embedding models support long input contexts (up to 32k tokens) and exhibit strong multilingual retrieval capabilities across both short queries and dense long-form documentation.
- **Reference Links**:
  - [Qwen Official Model Card on HuggingFace](https://huggingface.co/Qwen)
  - [Qwen GitHub Repository](https://github.com/QwenLM/Qwen)

### E5-Mistral (e5-mistral-7b-instruct)
- **Category**: Decoder-Based Instruction-Tuned Dense Embeddings
- **Developer / Organization**: Microsoft Research
- **Why Consider It**: Converts a 7B parameter decoder LLM (Mistral) into an embedding encoder using bidirectional attention and instruction tuning. It excels at adhering to asymmetric task instructions (e.g., distinguishing summary matching from question answering).
- **Reference Links**:
  - [Paper: Improving Text Embeddings with Large Language Models (Wang et al., 2024)](https://arxiv.org/abs/2401.00368)
  - [E5-Mistral-7b-instruct on HuggingFace](https://huggingface.co/intfloat/e5-mistral-7b-instruct)

### GTE (General Text Embeddings) / GTE-Qwen
- **Category**: Dense Embedding Models
- **Developer / Organization**: Alibaba DAMO Academy
- **Why Consider It**: GTE models are trained on large-scale paired datasets with multi-stage contrastive pretraining. Recent variants (such as GTE-Qwen) leverage decoder backbones for high MTEB benchmark performance while offering flexible embedding dimensions.
- **Reference Links**:
  - [Paper: Towards General Text Embeddings with Multi-stage Contrastive Learning (Zhang et al., 2023)](https://arxiv.org/abs/2308.03281)
  - [GTE-Qwen on HuggingFace](https://huggingface.co/Alibaba-NLP/gte-Qwen2-7B-instruct)

---

## 2. Learned Sparse Retrieval Alternatives

Learned sparse models output sparse vector representations over a vocabulary space (like BM25), but use neural networks to assign dynamic term weights and perform term expansion.

### SPLADE-v3 (Sparse Lexical and Expansion Model)
- **Category**: Neural Sparse Lexical Representation
- **Developer / Organization**: NAVER LABS Europe
- **Why Consider It**: SPLADE predicts term importance weights directly for all vocabulary tokens, automatically adding related expansion terms (addressing the vocabulary mismatch problem) while remaining compatible with standard inverted index vector engines.
- **Reference Links**:
  - [Paper: SPLADE v2: Sparse Lexical and Expansion Model for Information Retrieval (Formal et al., 2021)](https://arxiv.org/abs/2109.10086)
  - [SPLADE Official GitHub Repository](https://github.com/naver/splade)

### uniCOIL
- **Category**: Learned Term Weighting (without expansion)
- **Developer / Organization**: University of Waterloo / Castorini Research Group
- **Why Consider It**: uniCOIL simplifies neural sparse retrieval by learning term weights solely for tokens already present in the input text (or pre-expanded via doc2query-T5). It achieves low indexing overhead and fast query processing on traditional Lucene engines.
- **Reference Links**:
  - [Paper: A Few Brief Notes on DeepImpact, COIL, and a Simple Effective Baseline (Lin & Ma, 2021)](https://arxiv.org/abs/2106.14807)
  - [Pyserini uniCOIL Guide](https://github.com/castorini/pyserini/blob/master/docs/experiments-unicoil.md)

---

## 3. Architecture Selection Matrix

| Model Architecture | Output Vector Type | Strengths | Deployment Considerations |
| :--- | :--- | :--- | :--- |
| **BGE-M3** | Dense + Sparse + ColBERT | 3-in-1 multi-vector representations; strong multilingual support | Larger inference memory footprint if all 3 heads are evaluated simultaneously |
| **Qwen3-Embedding** | Dense | Large context window; strong general multilingual retrieval | Standard dense vector storage |
| **E5-Mistral** | Dense | High zero-shot task instruction following | Requires large GPU memory for 7B parameter inference |
| **SPLADE-v3** | Learned Sparse | Resolves term mismatch with automated vocabulary expansion | Requires inverted index or sparse-vector enabled engine |
| **BM25 (Baseline)** | Lexical Inverted Index | Zero GPU required; exact token matching; instant indexing | Vocabulary mismatch on synonyms and semantic paraphrases |
