# Multimodal RAG Test Results

## Configuration

```yaml
models:
  embedding: "qwen3-embedding:4b"
  llm: "qwen2.5vl:7b"
  cross_encoder: "cross-encoder/ms-marco-MiniLM-L-12-v2"
```

## Test Queries

| # | Type | Query |
|----|------|-------|
| 1 | Table | What are the different model complexities? |
| 2 | Table | Which model has the lowest training cost? |
| 3 | Image | What does the architecture diagram show? |
| 4 | Image | Describe the encoder and decoder blocks |
| 5 | Text | Explain the attention mechanism |

## Results

### Query 1: "What are the different model complexities and their per-layer complexity scores?"

**Type:** Table

| Rank | Chunk ID | Modality | Page | Score |
|------|---------|---------|------|-------|
| 1 | test.pdf_2_1 | TABLE | 2 | 2.465 |
| 2 | test.pdf_2_0 | TEXT | 2 | -0.145 |
| 3 | test.pdf_2_6 | TEXT | 2 | -5.75 |
| 4 | test.pdf_1_0 | IMAGE | 1 | -7.062 |
| 5 | test.pdf_3_2 | TEXT | 3 | -8.084 |

**Answer:**
The different model complexities and their per-layer complexity scores are as follows:

1. **Self-Attention**:
   - **Per Layer Complexity**: $O(n^2 \cdot d)$  
   - **Sequential Operations**: $O(1)$
   - **Maximum Path Length**: $O(1)$
   - Complexity per layer is restricted to $O(r \cdot n \cdot d)$ for restricted self-attention, and $O(k \cdot n \cdot d^2)$ for convolutional-based attention.

2. **Recurrent Neural Network (RNN)**:
   - **Per Layer Complexity**: $O(n \cdot d^2)$
   - **Sequent...

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
The model with the lowest training cost is Transformer (base model), and its BLEU score is not directly stated in the provided context. However, it does surpass all previously published models and ensembles in performance at a fraction of the training cost that competitive models require. Specifically, on the WMT 2014 English-to-German translation task, the big transformer model (Transformer (big)) achieves a BLEU score more than 2.0 points higher than previous state-of-the-art models, thus esta...

---
### Query 3: "What does the architecture diagram show?"

**Type:** Image

| Rank | Chunk ID | Modality | Page | Score |
|------|---------|---------|------|-------|
| 1 | test.pdf_1_0 | IMAGE | 1 | 0.586 |
| 2 | test.pdf_1_1 | TEXT | 1 | 0.5 |
| 3 | test.pdf_3_3 | TEXT | 3 | 0.444 |
| 4 | test.pdf_2_0 | TEXT | 2 | 0.366 |
| 5 | test.pdf_2_1 | TABLE | 2 | 0.344 |

**Answer:**
The architecture diagram shows the processing flow of input data through a transformer model, which is a type of neural network commonly used in natural language processing. The diagram illustrates the architecture's structure and the data processing steps involved in both the encoder and decoder components of the transformer.

- **Encoder**: Takes input sequences and processes them through multiple layers. Each layer has:
  - A multi-head attention mechanism that can focus on different parts of...

---
### Query 4: "Describe the encoder and decoder blocks in the flowchart"

**Type:** Image

| Rank | Chunk ID | Modality | Page | Score |
|------|---------|---------|------|-------|
| 1 | test.pdf_1_0 | IMAGE | 1 | 0.772 |
| 2 | test.pdf_1_1 | TEXT | 1 | 0.56 |
| 3 | test.pdf_1_2 | TEXT | 1 | 0.519 |
| 4 | test.pdf_3_0 | TEXT | 3 | 0.457 |
| 5 | test.pdf_2_0 | TEXT | 2 | 0.414 |

**Answer:**
The encoder and decoder blocks in the flowchart described in the context are part of the underlying architecture of the Transformer model, which is a type of neural network commonly used in natural language processing (NLP). Here is a breakdown:

### Encoder:

1. **Input Data**:
   - The encoder takes input sequences of data.

2. **Positional Encoding**:
   - The input data is passed through a "Positional Encoding" step (not explicitly labeled in the flowchart but implied by the text from the co...

---
### Query 5: "Explain the attention mechanism and how it differs from RNNs"

**Type:** Text

| Rank | Chunk ID | Modality | Page | Score |
|------|---------|---------|------|-------|
| 1 | test.pdf_1_0 | IMAGE | 1 | -1.004 |
| 2 | test.pdf_1_3 | TEXT | 1 | -2.825 |
| 3 | test.pdf_1_2 | TEXT | 1 | -5.129 |
| 4 | test.pdf_2_6 | TEXT | 2 | -5.816 |
| 5 | test.pdf_2_1 | TABLE | 2 | -10.166 |

**Answer:**
The attention mechanism in neural networks, such as those used in transformers, is a method for establishing "context" during the processing of a sequence of inputs. In contrast to recurrent neural networks (RNNs), which process the sequence sequentially and retain an internal state that captures information from previous inputs, the attention mechanism focuses on different parts of the input sequence simultaneously, allowing it to consider dependencies between elements in the context of the ent...

---
