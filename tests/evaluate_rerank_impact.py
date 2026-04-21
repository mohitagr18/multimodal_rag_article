#!/usr/bin/env python3
"""
Script to evaluate the impact of re-ranking stage on retrieval quality.
Compares dense retrieval only vs. dense retrieval + cross-encoder re-ranking.
"""
import json
import numpy as np
from pathlib import Path
from typing import List, Dict
import sys
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from phase4_retrieve import embed_text, generate_answer
from openai import OpenAI
from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder

PROJECT_ROOT = Path(__file__).parent.parent
config_path = PROJECT_ROOT / "src" / "config.yaml"
with open(config_path) as f:
    config = yaml.safe_load(f)

OLLAMA_BASE_URL = "http://localhost:11434/v1"
EMBED_MODEL = config["models"]["embedding"]
LLM_MODEL = config["models"]["llm"]
CROSS_ENCODER_MODEL = config["models"]["cross_encoder"]
COLLECTION_NAME = "article_chunks"


def initialize_clients():
    qdrant_path = config.get("qdrant", {}).get("path", "qdrant_db")
    qdrant = QdrantClient(path=str(PROJECT_ROOT / qdrant_path))
    openai_client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")
    cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL, max_length=512)
    return qdrant, openai_client, cross_encoder


def dense_retrieval_only(query: str, qdrant, openai_client, top_k: int = 20) -> List[Dict]:
    q_emb = embed_text(openai_client, query)
    results = qdrant.search(collection_name=COLLECTION_NAME, query_vector=q_emb, limit=top_k)
    hits = []
    for hit in results:
        hits.append({
            "chunk_id": hit.payload.get("chunk_id", "unknown"),
            "text": hit.payload.get("text", ""),
            "modality": hit.payload.get("modality", "text"),
            "page": hit.payload.get("page", "?"),
            "score": hit.score,
        })
    return hits


def dense_plus_rerank(query: str, qdrant, openai_client, cross_encoder, top_k: int = 20, rerank_top_n: int = 4) -> List[Dict]:
    q_emb = embed_text(openai_client, query)
    results = qdrant.search(collection_name=COLLECTION_NAME, query_vector=q_emb, limit=top_k)
    if not results:
        return []
    pairs = [[query, hit.payload.get("text", "")] for hit in results]
    scores = cross_encoder.predict(pairs)
    for hit, score in zip(results, scores):
        hit.score = float(score)
    results.sort(key=lambda x: x.score, reverse=True)
    best_hits = results[:rerank_top_n]
    hits = []
    for hit in best_hits:
        hits.append({
            "chunk_id": hit.payload.get("chunk_id", "unknown"),
            "text": hit.payload.get("text", ""),
            "modality": hit.payload.get("modality", "text"),
            "page": hit.payload.get("page", "?"),
            "score": hit.score,
        })
    return hits


def main():
    print("Evaluating impact of re-ranking stage...")
    print("=" * 50)
    try:
        qdrant, openai_client, cross_encoder = initialize_clients()
    except Exception as e:
        print(f"Failed to initialize clients: {e}")
        return

    from evaluate_retrieval import evaluate_retrieval
    queries = {
        "query_1": "What does the flowchart in the document illustrate?",
        "query_2": "How is positional encoding used in the architecture?",
        "query_3": "What is the difference between left and right sections?",
        "query_4": "Explain the multi-head attention mechanism steps",
        "query_5": "What comes after the softmax step in the processing?",
    }
    ground_truth = {
        "query_1": ["test.pdf_1_0"],
        "query_2": ["test.pdf_1_0"],
        "query_3": ["test.pdf_1_0"],
        "query_4": ["test.pdf_1_0"],
        "query_5": ["test.pdf_1_0"],
    }

    query_results_dense = {}
    query_results_rerank = {}
    for qid, qtext in queries.items():
        try:
            hits = dense_retrieval_only(qtext, qdrant, openai_client)
            query_results_dense[qid] = [h["chunk_id"] for h in hits]
            hits2 = dense_plus_rerank(qtext, qdrant, openai_client, cross_encoder)
            query_results_rerank[qid] = [h["chunk_id"] for h in hits2]
        except Exception as e:
            print(f"Error: {e}")

    dense_metrics = evaluate_retrieval(query_results_dense, ground_truth)
    rerank_metrics = evaluate_retrieval(query_results_rerank, ground_truth)

    print("\nDense Retrieval Only:")
    for k, v in dense_metrics.items():
        print(f"  {k}: {v:.4f}")
    print("\nDense Retrieval + Re-Ranking:")
    for k, v in rerank_metrics.items():
        print(f"  {k}: {v:.4f}")

    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / "rerank_impact.json"
    with open(results_path, "w") as f:
        json.dump({"dense_only": dense_metrics, "dense_rerank": rerank_metrics}, f, indent=2)
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()