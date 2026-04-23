# Multimodal RAG Pipeline: Technical Workflows

This document provides detailed workflow diagrams for each phase of the Multimodal RAG pipeline.

---

## 1. Overall Pipeline

```mermaid
flowchart LR
    subgraph Input["Input"]
        direction TB
        PDF[PDF Document]
    end

    subgraph Phase1["Phase 1: Parse"]
        direction TB
        P1A[Naive Extraction]
        P1B[Structure-Aware Parsing]
        P1C[PP-DocLayout + GLM-OCR]
        P1B --> P1C
    end

    subgraph Phase2["Phase 2: Enrich"]
        direction TB
        P2A[Image Extraction]
        P2B[VLM Captioning]
        P2C[Base64 Encoding]
        P2A --> P2B --> P2C
    end

    subgraph Phase3["Phase 3: Ingest"]
        direction TB
        P3A[Embedding Generation]
        P3B[Qdrant Storage]
        P3A --> P3B
    end

    subgraph Phase4["Phase 4: Retrieve"]
        direction TB
        P4A[Query Embedding]
        P4B[Dense Search]
        P4C[Modality Boosting]
        P4D[Cross-Encoder Rerank]
        P4E[LLM Synthesis]
        P4A --> P4B --> P4C --> P4D --> P4E
    end

    Output[Final Answer]

    Input --> Phase1
    Phase1 --> Phase2
    Phase2 --> Phase3
    Phase3 --> Phase4
    Phase4 --> Output
```

---

## 2. Phase 1: Document Parsing

### 2.1 Naive vs Structure-Aware Comparison

```mermaid
flowchart LR
    subgraph Naive["Naive Extraction<br/>(Baseline)"]
        direction TB
        NB1[PDF] --> NB2[fitz.get_text blocks] --> NB3[Tokenize to 512] --> NB4[Flat Chunks]
    end

    subgraph StructureAware["Structure-Aware Parsing"]
        direction TB
        SA1[PDF] --> SA2[PP-DocLayout Detection] --> SA3[GLM-OCR Text Recognition] --> SA4[Element Classification] --> SA5[Structure-Aware Chunking] --> SA6[Multimodal Chunks]
    end

    NB4 -->|"5 chunks<br/>Text only"| Result1["Lost:<br/>Layout<br/>Tables<br/>Images<br/>Reading Order"]
    SA6 -->|"17 chunks<br/>Text + Image<br/>+ Table + Formula"| Result2["Preserved:<br/>Layout<br/>Bounding Boxes<br/>Element Types"]

    style Result1 fill:#ffebee,stroke:#c62828
    style Result2 fill:#e8f5e9,stroke:#2e7d32
```

### 2.2 Parsing Element Classification

```mermaid
flowchart TB
    Start[Raw PDF Elements] --> Classify{Element Type?}
    
    Classify -->|"label='image'"| Image[Image Element]
    Classify -->|"label='table'"| Table[Table Element]
    Classify -->|"label='formula'"| Formula[Formula Element]
    Classify -->|"label='paragraph'" | Paragraph[Text Element]
    Classify -->|"label='paragraph_title'"| Title[Title Element]
    
    Image --> Atomic1[Atomic Chunk<br/>modality: image]
    Table --> Atomic2[Atomic Chunk<br/>modality: table]
    Formula --> Atomic3[Atomic Chunk<br/>modality: formula]
    Paragraph --> Group1[Grouped with<br/>surrounding text]
    Title --> Group1
    
    Group1 --> TextChunk[Text Chunk<br/>modality: text]
```

---

## 3. Phase 2: Multimodal Enrichment

```mermaid
flowchart LR
    Input[structured_chunks.json]

    subgraph Enrich["For Each Chunk"]
        direction TB
        E1{Is image,<br/>table, or formula?}
        Skip[Skip]
        E2[Extract bbox<br/>coordinates]
        E3[PyMuPDF crop]
        E4[Render to PNG]
        E5[Base64 encode]
        E6[Store in chunk]

        E1 -->|No| Skip
        E1 -->|Yes| E2 --> E3 --> E4 --> E5 --> E6
    end

    subgraph VLM["VLM Captioning (Images Only)"]
        direction TB
        V1[Base64 image]
        V2[Ollama API<br/>qwen2.5vl:7b]
        V3[Prompt:<br/>Describe this image]
        V4[Generated caption]
        V5[Prepend to chunk text]

        V1 --> V2 --> V3 --> V4 --> V5
    end

    Output[enriched_chunks.json]

    Input --> E1
    Skip --> Output
    E6 --> V1
    V5 --> Output
```

### 3.1 Chunk Schema After Enrichment

```mermaid
classDiagram
    class Chunk {
        +string text
        +string chunk_id
        +int page
        +list element_types
        +list bbox
        +string source_file
        +bool is_atomic
        +string modality
        +string image_base64
        +string caption
    }

    note for Chunk "text: [IMAGE CAPTION] The transformer architecture shows encoder and decoder blocks...\n\n[ORIGINAL TEXT] The image is a flowchart...\n\nmodality: image | table | formula | text\n\nimage_base64: iVBORw0KGgoAAA..."
```

---

## 4. Phase 3: Vector Ingestion

```mermaid
flowchart LR
    Input["enriched_chunks.json"]

    subgraph Embed["Embedding Pipeline"]
        direction TB
        E1["Initialize Ollama client"]
        E2["Test embedding dimension"]
        E3["Create Qdrant collection<br/>Cosine distance"]

        E1 --> E2 --> E3
    end

    subgraph Process["For Each Chunk"]
        direction TB
        P1["Extract text field"]
        P2["Call Ollama<br/>embeddings API"]
        P3["Get vector<br/>2560-dim, qwen3"]
        P4["Build metadata<br/>chunk_id, page, modality"]

        P1 --> P2 --> P3 --> P4
    end

    subgraph Store["Qdrant Upsert"]
        direction TB
        S1["Create PointStruct"]
        S2["Add vector"]
        S3["Add payload<br/>text + metadata"]
        S4["Upsert to collection"]

        S1 --> S2 --> S3 --> S4
    end

    Output["Qdrant collection<br/>article_chunks"]

    Input --> Embed
    Embed --> Process
    Process --> Store
    Store --> Output
```

---

## 5. Phase 4: Retrieval & Synthesis

### 5.1 Complete Query Flow

```mermaid
flowchart LR
    Query["User Query"]

    subgraph EmbedQuery["Query Processing"]
        direction TB
        Q1["Embed query<br/>via Ollama"]
        Q2["Query vector<br/>2560-dim"]

        Q1 --> Q2
    end

    subgraph DenseSearch["Stage 1: Dense Retrieval"]
        direction TB
        D1["Qdrant vector search"]
        D2["Top 20 candidates"]
        D3["Cosine similarity<br/>scores"]

        D1 --> D2 --> D3
    end

    subgraph Boost["Stage 2: Modality Boosting"]
        direction TB
        B1{"Visual keywords<br/>detected?"}
        B2["Multiply image<br/>scores by 1.35"]
        B3["Skip boosting"]

        B1 -->|Yes| B2
        B1 -->|No| B3
    end

    subgraph Rerank["Stage 3: Cross-Encoder"]
        direction TB
        R1{"Visual query?"}
        R2["Skip reranking<br/>use boosted scores"]
        R3["Cross-encoder<br/>ms-marco-MiniLM-L-6-v2"]
        R4["Re-sort by<br/>cross-encoder score"]

        R1 -->|Yes| R2
        R1 -->|No| R3 --> R4
    end

    subgraph Synthesize["Stage 4: LLM Synthesis"]
        direction TB
        S1["Top 4 chunks<br/>as context"]
        S2["Build prompt with<br/>modality badges"]
        S3["Call LLM<br/>qwen2.5vl:7b"]
        S4["Generate answer"]

        S1 --> S2 --> S3 --> S4
    end

    Answer["Final Answer"]

    Query --> EmbedQuery
    EmbedQuery --> DenseSearch
    DenseSearch --> Boost
    Boost --> Rerank
    Rerank --> Synthesize
    Synthesize --> Answer
```

### 5.2 Modality Boosting Logic

```mermaid
flowchart LR
    Start["Incoming Query"] --> Extract{"Extract query words"}
    Extract --> Lower["Convert to lowercase"]
    Lower --> Split["Split by whitespace"]
    Split --> Keywords{"Contains visual<br/>keywords?"}

    Keywords -->|Yes| Visual["Visual query<br/>detected"]
    Keywords -->|No| NonVisual["Non-visual query"]

    Visual --> ForEach["For each result"]
    ForEach --> CheckMod{"modality == image?"}

    CheckMod -->|Yes| Boost["score = score × 1.35"]
    CheckMod -->|No| NoBoost["No change"]

    Boost --> AllDone["All results processed"]
    NoBoost --> AllDone
    NonVisual --> Skip["Skip boosting<br/>use original scores"]

    AllDone --> Sort["Sort by<br/>adjusted score"]
    Skip --> Sort
    Sort --> Output["Ranked results"]

    Note["Visual keywords:<br/>diagram, flowchart,<br/>figure, image, chart,<br/>visual, illustration,<br/>picture, encoder,<br/>decoder"]
    Keywords -.-> Note
```

### 5.3 Visual Query Detection & Boosting

```mermaid
flowchart TD
    Q[Query] --> Token[Tokenize + lowercase]
    Token --> KW{Any keyword matches<br/>visual_keywords?}

    KW -->|Yes| Visual[Visual Query Detected]
    Visual --> Boost[For each result<br/>modality == image?<br/>score = score × 1.35]
    Boost --> Skip[Skip cross-encoder<br/>use boosted scores]
    Skip --> Top4[Select top 4]

    KW -->|No| NonVisual[Non-Visual Query]
    NonVisual --> Rerank[Cross-encoder rerank<br/>ms-marco-MiniLM-L-6-v2]
    Rerank --> Sort[Sort by rerank score]
    Sort --> Top4

    KW -.-- Note[visual_keywords:<br/>diagram, flowchart,<br/>figure, image, chart,<br/>visual, illustration,<br/>picture, encoder,<br/>decoder]

    Note -.-> Visual
```

**Why It Works:**

| Scenario | Query Type | Behavior |
|----------|------------|----------|
| "How does the **encoder** work?" | Visual | Boost image scores ×1.35, skip cross-encoder |
| "What are the **results** in Table 2?" | Non-Visual | Use cross-encoder reranking |
| "Explain the **flowchart**" | Visual | Boost image scores ×1.35, skip cross-encoder |

**Score Example:**
```
Query: "diagram of transformer"
Initial: IMAGE #7, score 0.837 (ranked 7th)
After boost: 0.837 × 1.35 = 1.130 → moves to #1
```

---

## 6. Data Schemas

### 6.1 Chunk Metadata Fields

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `chunk_id` | string | Unique identifier | `test.pdf_1_0` |
| `page` | int | Page number | `1` |
| `modality` | string | Element type | `image`, `table`, `formula`, `text` |
| `source_file` | string | Original PDF | `test.pdf` |
| `element_types` | list | Element labels | `["figure_title", "image"]` |
| `bbox` | list | Bounding box | `[100, 380, 500, 580]` |
| `is_atomic` | bool | Single element? | `true` for tables/images |
| `image_base64` | string | PNG in base64 | `iVBORw0KGgo...` |
| `caption` | string | VLM-generated | `The transformer architecture...` |

### 6.2 Supported Modalities

```mermaid
classDiagram
    class Modality {
        <<enumeration>>
    }
    
    Modality : "text" - paragraphs, titles, abstracts
    Modality : "image" - figures, diagrams, charts
    Modality : "table" - tabular data
    Modality : "formula" - mathematical expressions
```

---

## 7. Configuration Reference

### 7.1 Model Configuration

```yaml
models:
  embedding: "qwen3-embedding:4b"  # Vector embeddings
  llm: "qwen2.5vl:7b"              # Answer generation
  vlm: "qwen2.5vl:7b"              # Image captioning
  cross_encoder: "cross-encoder/ms-marco-MiniLM-L-12-v2"  # Re-ranking
```

### 7.2 Pipeline Configuration

```yaml
pipeline:
  ocr_api:
    model: glm-ocr:latest    # Layout detection
    api_mode: openai         # Ollama OpenAI-compatible API
  layout:
    enable_layout: true      # PP-DocLayout-V3
```

---

## 8. Error Handling

### 8.1 Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| Empty embeddings | Ollama not running | Start `ollama serve` |
| Qdrant lock error | DB in use | Delete `.lock` file |
| VLM timeout | Large images | Reduce DPI or image size |
| Cross-encoder import | Python 3.9 numpy bug | Use Python 3.10+ or skip reranking |

---

## 9. Performance Characteristics

| Stage | Time | Notes |
|-------|------|-------|
| Phase 1 (Parse) | ~5s/page | Depends on PDF complexity |
| Phase 2 (Enrich) | ~10s/image | VLM inference time |
| Phase 3 (Ingest) | ~100ms/chunk | Embedding + upsert |
| Phase 4 (Query) | ~500ms | Embed + search + rerank + LLM |

---

*For more details on the overall project, see [README.md](README.md)*
