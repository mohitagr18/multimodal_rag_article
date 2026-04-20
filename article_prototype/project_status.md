# Multimodal RAG Prototype: Implementation Plan & Progress

## Objective
Build a minimal, inspectable multimodal RAG prototype that proves document retrieval quality depends on structure-aware parsing and representation rather than prompt engineering.

---

## Phase 1: Structure-Aware Parsing (Completed)
**Goal:** Finalize the structure-aware parsing pipeline using `PPDocLayout` and `GLM` models to extract layout-accurate, retrieval-ready chunks. Validate parsing logic by comparing naive text extraction against structured output on custom PDF documents.

### Status: `[x] COMPLETED`
* Developed the `phase1_parse.py` pipeline.
* Integrated `PPDocLayoutV3` for identifying document regions (tables, figures, formulas, text).
* Integrated `GLM-OCR` for text recognition over bounding boxes.
* **Troubleshooting:** Resolved an issue where GLM OCR was returning empty chunks because it was configured to use Ollama's raw `/api/generate` endpoint (`api_mode: ollama_generate`). The pipeline was fixed by modifying `config.yaml` to leverage Ollama's OpenAI-compatible `/v1/chat/completions` endpoint (`api_mode: openai`), which fully supports robust multimodal requests.
* **Verification:** Successfully parsed `test.pdf`, generating a `structured_chunks.json` which demonstrated structural superiority over naive `fitz` extraction.

---

## Phase 2: Multimodal Enrichment & Vector Ingestion (Completed)
**Goal:** Prepare the codebase for multimodal enrichment and local vector retrieval.

### Status: `[x] COMPLETED`
* **Image Payload Injection:** Created `phase2_enrich.py` to crop specific elements using PyMuPDF (`fitz`), embedding the outputs as base64 image strings directly back into the component structures.
* **Caption Generation:** Prompted local `qwen2.5vl:7b` by sending the embedded payload to Ollama, dynamically generating rich, descriptive captions for extracted images and appending them to the chunk text to facilitate semantic search.
* **Vector Store Integration:** Scripted `phase3_ingest.py` to instantiate a persistent local instance of **Qdrant** targeting `./qdrant_db`.
* **Embedding Generation:** Transformed structure-aware chunks into 2304-dimensional vectors via Ollama's `/v1` endpoint mimicking OpenAI compatibility utilizing the `gemma2:2b` embeddings schema, finally upserting payloads and metadata vectors into Qdrant collections flawlessly.

---

## Phase 3: RAG Retrieval & Evaluation 
**Goal:** Implement the querying logic and demonstrate the effectiveness of the structured representation compared to naive ingestion.

### Status: `[ ] PENDING`
* **Query Implementation:** Develop a two-stage search interface. First, use dense vectors (`gemma2:2b`) to pull the top 20 candidate chunks from Qdrant.
* **Re-Ranking:** Implement a Cross-Encoder (e.g. using `sentence-transformers` cross-encoder models) to re-rank the top 20 candidates down to the true top 3-5 contexts most relevant to the query.
* **LLM Synthesis:** Pass the re-ranked structural contexts (including generated VLM captions and layout details) to an LLM to generate the final answer.
* **Evaluation:** Run comparative benchmarks measuring retrieval quality and evaluate the impact of the reranking stage.
