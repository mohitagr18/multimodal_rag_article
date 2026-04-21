# Multimodal RAG Prototype: Test Results

## Date
April 21, 2026

## Configuration

All models are sourced from `config.yaml` as the single source of truth:

```yaml
models:
  embedding: "qwen3-embedding:4b"
  llm: "qwen2.5vl:7b"
  vlm: "qwen2.5vl:7b"
  cross_encoder: "cross-encoder/ms-marco-MiniLM-L-12-v2"
```

## Pipeline Stages

1. **Dense Retrieval**: Vector search (top 20)
2. **Modality Boosting**: 35% boost for image chunks on visual queries
3. **Cross-Encoder Reranking**: ms-marco-MiniLM-L-12-v2 re-scoring
4. **LLM Synthesis**: qwen2.5vl:7b answer generation

---

## Test Queries Summary

| # | Type | Query |
|----|------|-------|
| 1 | Table | What are the different model complexities and their per-layer complexity scores? |
| 2 | Table | Which model has the lowest training cost and what is its BLEU score? |
| 3 | Image | What does the architecture diagram show? |
| 4 | Image | Describe the encoder and decoder blocks in the flowchart |
| 5 | Text | Explain the attention mechanism and how it differs from RNNs |

---

## Results

### Query 1: "What are the different model complexities and their per-layer complexity scores?"

**Type:** Table

| Rank | Chunk ID | Modality | Page | Score |
|------|---------|---------|------|-------|
| 1 | test.pdf_2_1 | TABLE | 2 | 2.465 |
| 2 | test.pdf_2_0 | TEXT | 2 | -0.145 |
| 3 | test.pdf_2_6 | TEXT | 2 | -5.75 |
| 4 | test.pdf_1_0 | IMAGE | 1 | -7.366 |
| 5 | test.pdf_3_2 | TEXT | 3 | -8.084 |

**Answer:**
Based on the provided context chunks, here's a summary of the different model complexities and their per-layer complexity scores for the three layer types: Self-Attention, Recurrent, and Convolutional:

1. **Self-Attention:**
   - **Per-layer complexity:** \( O(n^2 \cdot d) \)
   - **Sequential operations (per layer):** \( O(1) \)
   - **Maximum path length (per layer):** \( O(1) \)

2. **Recurrent:**
   - **Per-layer complexity:** \( O(n \cdot d^2) \)
   - **Sequential operations (per layer):**...

---

### Query 2: "Which model has the lowest training cost and what is its BLEU score?"

**Type:** Table

| Rank | Chunk ID | Modality | Page | Score |
|------|---------|---------|------|-------|
| 1 | test.pdf_3_1 | TABLE | 3 | 8.189 |
| 2 | test.pdf_3_0 | TEXT | 3 | 5.317 |
| 3 | test.pdf_3_4 | TEXT | 3 | 4.202 |
| 4 | test.pdf_3_2 | TEXT | 3 | -3.255 |
| 5 | test.pdf_3_3 | TEXT | 3 | -10.596 |

**Answer:**
The model with the lowest training cost is indicated as Model [18], which has a training cost of \(1.0 \cdot 10^{20}\) FLOPs. The corresponding BLEU score for this model is not explicitly mentioned in the provided context chunks....

---

### Query 3: "What does the architecture diagram show?"

**Type:** Image

| Rank | Chunk ID | Modality | Page | Score |
|------|---------|---------|------|-------|
| 1 | test.pdf_1_0 | IMAGE | 1 | 0.562 |
| 2 | test.pdf_1_1 | TEXT | 1 | 0.5 |
| 3 | test.pdf_3_3 | TEXT | 3 | 0.444 |
| 4 | test.pdf_2_0 | TEXT | 2 | 0.366 |
| 5 | test.pdf_2_1 | TABLE | 2 | 0.344 |

**Answer:**
The architecture diagram in the image illustrates the structure of a transformer model, a type of neural network commonly used in natural language processing (NLP). The diagram shows the general architecture of a transformer model divided into two main components: the encoder and the decoder, which communicate with each other through a series of layers. It highlights the process of input and output sequences being embedded into a vector space using embedding layers, where the input sequence is t...

---

### Query 4: "Describe the encoder and decoder blocks in the flowchart"

**Type:** Image

| Rank | Chunk ID | Modality | Page | Score |
|------|---------|---------|------|-------|
| 1 | test.pdf_1_0 | IMAGE | 1 | 0.747 |
| 2 | test.pdf_1_1 | TEXT | 1 | 0.56 |
| 3 | test.pdf_1_2 | TEXT | 1 | 0.519 |
| 4 | test.pdf_3_0 | TEXT | 3 | 0.457 |
| 5 | test.pdf_2_0 | TEXT | 2 | 0.414 |

**Answer:**
The flowchart depicts the architecture of a transformer model with an emphasis on its encoder and decoder components. In the context of this diagram, the encoder and decoder blocks are interconnected through a series of layers that facilitate the processing of input and output sequences in natural language processing tasks.

### Encoder Block:

The encoder block is composed of a stack of N=6 identical layers, where N represents the number of layers. Each layer within the encoder consists of two ...

---

### Query 5: "Explain the attention mechanism and how it differs from RNNs"

**Type:** Text

| Rank | Chunk ID | Modality | Page | Score |
|------|---------|---------|------|-------|
| 1 | test.pdf_1_3 | TEXT | 1 | -2.825 |
| 2 | test.pdf_1_2 | TEXT | 1 | -5.129 |
| 3 | test.pdf_2_6 | TEXT | 2 | -5.816 |
| 4 | test.pdf_1_0 | IMAGE | 1 | -6.06 |
| 5 | test.pdf_2_1 | TABLE | 2 | -10.166 |

**Answer:**
The attention mechanism in a transformer model is a method for allowing each computation step to focus on the most relevant data. Specifically, an attention function maps a query and a set of key-value pairs to an output, where: 

- The query is a vector,
- The keys and values are vectors pairs,
- The output is a weighted sum computed by each element of the key multiplying by the corresponding element from the value vector, then summed together based on the weights computed by the attention mech...

---

## Summary by Query Type

| Query Type | Top Rank | Top Modality | Status |
|-----------|---------|------------|--------|
| Table | #1 | TABLE | ✅ Working |
| Image | #1 | IMAGE | ✅ Working (with boost + reranking) |
| Text | #1 | TEXT | ✅ Working |

---

## Qdrant Database Stats

- **Embedding model:** qwen3-embedding:4b (dimension: 2560)
- **LLM model:** qwen2.5vl:7b
- **Cross-encoder:** cross-encoder/ms-marco-MiniLM-L-12-v2
- **Distance metric:** Cosine

---

## Conclusion

✅ **All modalities retrieve correctly with full pipeline (embedding + boosting + reranking)**
