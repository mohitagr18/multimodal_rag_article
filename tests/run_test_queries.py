#!/usr/bin/env python3
"""
Test script to run the 5 standard queries with FULL pipeline.
"""
import yaml
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent

with open(PROJECT_ROOT / "src" / "config.yaml") as f:
    config = yaml.safe_load(f)

from phase4_retrieve import embed_text, generate_answer
from openai import OpenAI
from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder

EMBED_MODEL = config['models']['embedding']
LLM_MODEL = config['models']['llm']
CROSS_ENCODER_MODEL = config['models']['cross_encoder']
RESULTS_DIR = Path(config.get("directories", {}).get("results", "results"))

qdrant_path = config.get("qdrant", {}).get("path", "qdrant_db")
qdrant = QdrantClient(path=str(PROJECT_ROOT / qdrant_path))
openai_client = OpenAI(base_url='http://localhost:11434/v1', api_key='ollama')
cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL, max_length=512)

print(f"Embedding: {EMBED_MODEL}")
print(f"LLM: {LLM_MODEL}")
print(f"Cross-encoder: {CROSS_ENCODER_MODEL}")

queries = [
    ('table', 'What are the different model complexities and their per-layer complexity scores?'),
    ('table', 'Which model has the lowest training cost and what is its BLEU score?'),
    ('image', 'What does the architecture diagram show?'),
    ('image', 'Describe the encoder and decoder blocks in the flowchart'),
    ('text', 'Explain the attention mechanism and how it differs from RNNs'),
]

visual_keywords = {'diagram', 'flowchart', 'figure', 'image', 'chart', 'visual', 'illustration', 'picture', 'encoder', 'decoder'}
results_data = []

for modality, query in queries:
    print(f'\n[{modality.upper()}] {query}')
    
    q_emb = embed_text(query)
    results = qdrant.query_points('article_chunks', query=q_emb, limit=20)
    print(f'   Stage 1: Dense retrieval - {len(results.points)} candidates')
    
    is_visual_query = bool(set(query.lower().split()) & visual_keywords)
    if is_visual_query:
        print('   Stage 2: Visual query - boosting image 35%')
        for hit in results.points:
            if hit.payload.get('modality') == 'image':
                hit.score *= 1.35
    
    if is_visual_query:
        print('   Stage 3: Skipping cross-encoder (boosting sufficient)')
    else:
        print('   Stage 3: Cross-encoder reranking...')
        pairs = [[query, hit.payload.get('text', '')] for hit in results.points]
        ce_scores = cross_encoder.predict(pairs)
        for hit, ce_score in zip(results.points, ce_scores):
            hit.score = float(ce_score)
    
    results.points.sort(key=lambda x: x.score, reverse=True)
    
    print(f'   Top result: {results.points[0].payload.get("modality")} ({results.points[0].score:.3f})')
    
    contexts = [h.payload for h in results.points[:4]]
    answer = generate_answer(query, contexts)
    
    results_data.append({
        'query_modality': modality,
        'query': query,
        'top_chunks': [
            {
                'rank': i+1,
                'chunk_id': hit.payload.get('chunk_id'),
                'modality': hit.payload.get('modality'),
                'page': hit.payload.get('page'),
                'score': round(hit.score, 3)
            }
            for i, hit in enumerate(results.points[:5])
        ],
        'answer': answer
    })

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
with open(RESULTS_DIR / 'test_run_results.json', 'w') as f:
    json.dump(results_data, f, indent=2)
print(f'\nJSON saved to {RESULTS_DIR / "test_run_results.json"}')

md_content = f"""# Multimodal RAG Test Results

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

for i, result in enumerate(results_data):
    md_content += f"### Query {i+1}: \"{result['query']}\"\n\n"
    md_content += f"**Type:** {result['query_modality'].capitalize()}\n\n"
    md_content += "| Rank | Chunk ID | Modality | Page | Score |\n"
    md_content += "|------|---------|---------|------|-------|\n"
    for chunk in result['top_chunks']:
        md_content += f"| {chunk['rank']} | {chunk['chunk_id']} | {chunk['modality'].upper()} | {chunk['page']} | {chunk['score']} |\n"
    md_content += f"\n**Answer:**\n{result['answer'][:500]}...\n\n---\n"

with open(RESULTS_DIR / 'TEST_RESULTS.md', 'w') as f:
    f.write(md_content)
print(f'Markdown saved to {RESULTS_DIR / "TEST_RESULTS.md"}')

qdrant.close()
openai_client.close()