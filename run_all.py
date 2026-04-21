#!/usr/bin/env python3
"""
All-in-one script to run the full pipeline or test queries.

Usage:
    python run_all.py              # Run all phases + test queries
    python run_all.py --test-only   # Run only test queries
    python run_all.py --phase 2     # Run specific phase
    python run_all.py --pdf my.pdf   # Use custom PDF
"""
import argparse
import json
import sys
import yaml
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

with open(PROJECT_ROOT / "src" / "config.yaml") as f:
    config = yaml.safe_load(f)

RESULTS_DIR = Path(config.get("directories", {}).get("results", "results"))
EMBED_MODEL = config['models']['embedding']
LLM_MODEL = config['models']['llm']
CROSS_ENCODER_MODEL = config['models']['cross_encoder']
INPUT_DIR = PROJECT_ROOT / "input"
QDRANT_PATH = str(PROJECT_ROOT / config.get("qdrant", {}).get("path", ".qdrant"))

shared_qdrant = None
shared_openai = None


def run_phase1(pdf_path):
    from phase1_parse import main as p1_main
    sys.argv = ["phase1_parse.py", str(pdf_path), "--engine", "real"]
    p1_main()


def run_phase2():
    from phase2_enrich import main as p2_main
    p2_main()


def run_phase3():
    global shared_qdrant, shared_openai
    from openai import OpenAI
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct
    import uuid

    input_file = RESULTS_DIR / "enriched_chunks.json"
    if not input_file.exists():
        print(f"File {input_file} not found. Run phase2_enrich.py first."); return False

    with open(input_file) as f:
        chunks = json.load(f)

    shared_qdrant = QdrantClient(path=QDRANT_PATH)
    shared_openai = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

    def embed(c, t):
        return c.embeddings.create(model=EMBED_MODEL, input=t).data[0].embedding

    test_emb = embed(shared_openai, "test")
    emb_dim = len(test_emb)
    print(f"Embedding dimension: {emb_dim}")

    if shared_qdrant.collection_exists("article_chunks"):
        shared_qdrant.delete_collection("article_chunks")
    shared_qdrant.create_collection("article_chunks", vectors_config=VectorParams(size=emb_dim, distance=Distance.COSINE))

    points = []
    for i, chunk in enumerate(chunks):
        text = chunk.get("text", ""); 
        if not text: continue
        vec = embed(shared_openai, text)
        meta = {k: chunk[k] for k in ["chunk_id", "page", "element_types", "source_file"]}
        meta["modality"] = chunk.get("modality", "text")
        if chunk.get("image_base64"):
            meta["has_image"] = True
            meta["image_base64"] = chunk["image_base64"]
        points.append(PointStruct(id=str(uuid.uuid4()), vector=vec, payload={"text": text, **meta}))
        print(f"Embedded chunk {i+1}/{len(chunks)} (modality: {chunk.get('modality')})")

    shared_qdrant.upsert(collection_name="article_chunks", points=points)
    print(f"Ingested {len(points)} chunks.")
    return True


def run_test_queries():
    global shared_qdrant, shared_openai
    if shared_qdrant is None:
        shared_qdrant = QdrantClient(path=QDRANT_PATH)
    if shared_openai is None:
        shared_openai = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

    from sentence_transformers import CrossEncoder

    def embed_text(text):
        return shared_openai.embeddings.create(model=EMBED_MODEL, input=text).data[0].embedding

    def generate_answer(query, contexts):
        ctx_str = "".join(
            f"--- Chunk {i+1} (Page {c.get('page','?')}) [{c.get('modality','TEXT').upper()}] ---\n{c.get('text','')}\n\n"
            for i, c in enumerate(contexts))
        prompt = f"""You are an AI assistant using Multimodal Structural RAG. Use the context to answer.

Context:
{ctx_str}

User Question: {query}

Answer:"""
        resp = shared_openai.chat.completions.create(model=LLM_MODEL, messages=[{"role":"user","content":prompt}], stream=False)
        return resp.choices[0].message.content

    cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL, max_length=512)
    print(f"Models: {EMBED_MODEL}, {CROSS_ENCODER_MODEL}")

    queries = [
        ('table', 'What are the different model complexities and their per-layer complexity scores?'),
        ('table', 'Which model has the lowest training cost and what is its BLEU score?'),
        ('image', 'What does the architecture diagram show?'),
        ('image', 'Describe the encoder and decoder blocks in the flowchart'),
        ('text', 'Explain the attention mechanism and how it differs from RNNs'),
    ]
    visual_kw = {'diagram', 'flowchart', 'figure', 'image', 'chart', 'visual', 'illustration', 'picture', 'encoder', 'decoder'}
    results_data = []

    for modality, query in queries:
        print(f'\n[{modality.upper()}] {query}')
        q_emb = embed_text(query)
        results = shared_qdrant.query_points('article_chunks', query=q_emb, limit=20)
        print(f'   Dense retrieval: {len(results.points)} candidates')
        is_vis = bool(set(query.lower().split()) & visual_kw)
        if is_vis:
            print('   Visual query - boosting image 35%')
            for hit in results.points:
                if hit.payload.get('modality') == 'image':
                    hit.score *= 1.35
        if is_vis:
            print('   Skipping cross-encoder (boosting sufficient)')
        else:
            print('   Cross-encoder reranking...')
            pairs = [[query, hit.payload.get('text', '')] for hit in results.points]
            ce_scores = cross_encoder.predict(pairs)
            for hit, cs in zip(results.points, ce_scores):
                hit.score = float(cs)
        results.points.sort(key=lambda x: x.score, reverse=True)
        print(f'   Top: {results.points[0].payload.get("modality")} ({results.points[0].score:.3f})')
        answer = generate_answer(query, [h.payload for h in results.points[:4]])
        results_data.append({
            'query_modality': modality, 'query': query,
            'top_chunks': [{'rank': i+1, 'chunk_id': h.payload.get('chunk_id'),
                          'modality': h.payload.get('modality'),
                          'page': h.payload.get('page'), 'score': round(h.score, 3)}
                         for i, h in enumerate(results.points[:5])],
            'answer': answer
        })

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / 'test_run_results.json', 'w') as f:
        json.dump(results_data, f, indent=2)

    md = f"""# Multimodal RAG Test Results

## Configuration

```yaml
models:
  embedding: "{EMBED_MODEL}"
  llm: "{LLM_MODEL}"
  cross_encoder: "{CROSS_ENCODER_MODEL}"
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

"""
    for i, r in enumerate(results_data):
        md += f"### Query {i+1}: \"{r['query']}\"\n\n"
        md += f"**Type:** {r['query_modality'].capitalize()}\n\n"
        md += "| Rank | Chunk ID | Modality | Page | Score |\n|------|---------|---------|------|-------|\n"
        for c in r['top_chunks']:
            md += f"| {c['rank']} | {c['chunk_id']} | {c['modality'].upper()} | {c['page']} | {c['score']} |\n"
        md += f"\n**Answer:**\n{r['answer'][:500]}...\n\n---\n"

    with open(RESULTS_DIR / 'TEST_RESULTS.md', 'w') as f:
        f.write(md)
    print(f'Results saved to {RESULTS_DIR}/')

    shared_qdrant.close()
    shared_openai.close()
    return True


def main():
    parser = argparse.ArgumentParser(description="Run full pipeline or test queries")
    parser.add_argument("--phase", type=int, choices=[1, 2, 3, 4],
        help="Run specific phase (1=Parse, 2=Enrich, 3=Ingest, 4=Test)")
    parser.add_argument("--test-only", action="store_true",
        help="Only run test queries (skip phases 1-3)")
    parser.add_argument("--pdf", type=str, default="test.pdf",
        help="PDF file to process (default: test.pdf)")
    args = parser.parse_args()

    pdf_path = INPUT_DIR / args.pdf
    if not pdf_path.exists():
        pdf_path = PROJECT_ROOT / args.pdf
    if not pdf_path.exists() and not args.test_only:
        print(f"Error: PDF file '{pdf_path}' not found!"); return

    print("="*50 + "\nMULTIMODAL RAG PIPELINE\n" + "="*50)
    print(f"PDF: {pdf_path}")

    if args.test_only:
        run_test_queries(); return

    phases = [args.phase] if args.phase else [1, 2, 3, 4]
    print(f"Phases: {phases}")

    for phase in phases:
        print(f"\n{'='*50}\nPHASE {phase}\n{'='*50}")
        if phase == 1: run_phase1(pdf_path)
        elif phase == 2: run_phase2()
        elif phase == 3:
            if not run_phase3(): return
        elif phase == 4:
            if not run_test_queries(): return

    print(f"\n{'='*50}\nPIPELINE COMPLETE!\n{'='*50}")
    print(f"Results in: {PROJECT_ROOT}/results/")
    print(f"To query interactively: python src/phase4_retrieve.py")


if __name__ == "__main__":
    main()