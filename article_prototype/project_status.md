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

## Phase 2: Multimodal Enrichment & Vector Ingestion (Up Next)
**Goal:** Prepare the codebase for multimodal enrichment and local vector retrieval.

### Status: `[ ] PENDING`
* **Image Payload Injection:** Embed base64 image strings within specific chunks (e.g., tables and figures) for storage.
* **Caption Generation:** Utilize local VLMs (like Qwen2.5-VL or GLM) to generate rich, descriptive captions for extracted images and charts to facilitate semantic search.
* **Vector Store Integration:** Spin up a local instance of **Qdrant**.
* **Embedding Generation:** Embed structure-aware chunks and load them into Qdrant collections.

---

## Phase 3: RAG Retrieval & Evaluation 
**Goal:** Implement the querying logic and demonstrate the effectiveness of the structured representation compared to naive ingestion.

### Status: `[ ] PENDING`
* **Query Implementation:** Develop a search interface that queries Qdrant with mixed-mode questions.
* **Evaluation:** Run comparative benchmarks measuring retrieval quality (F1 Score/Latency) between naive chunking text searches and the structure-aware embeddings.
* **System Traceability:** Provide output traces to easily inspect retrieved chunks and context windows verifying that layout, charts, and tables map cleanly back to original PDF pages.
