#!/usr/bin/env python3
"""
Script to evaluate the impact of re-ranking stage on retrieval quality.
Compares dense retrieval only vs. dense retrieval + cross-encoder re-ranking.
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import sys
import os

# Add the article_prototype directory to path
sys.path.append(str(Path(__file__).parent))

# Import the retrieval functions from phase4_retrieve
from phase4_retrieve import embed_text, generate_answer
from openai import OpenAI
from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder
from pathlib import Path
import yaml

# Load configuration
config_path = Path(__file__).parent / "config.yaml"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

OLLAMA_BASE_URL = "http://localhost:11434/v1"
EMBED_MODEL = config["models"]["embedding"]
LLM_MODEL = config["models"]["llm"]
CROSS_ENCODER_MODEL = config["models"]["cross_encoder"]
COLLECTION_NAME = "article_chunks"


def initialize_clients():
    """Initialize clients (same as in phase4_retrieve.py)"""
    qdrant = QdrantClient(path="qdrant_db")
    openai_client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")
    cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL, max_length=512)
    return qdrant, openai_client, cross_encoder


def dense_retrieval_only(
    query: str, qdrant, openai_client, top_k: int = 20
) -> List[Dict]:
    """Perform dense retrieval only (no re-ranking)"""
    # Embed query
    q_emb = embed_text(openai_client, query)

    # Search Qdrant
    results = qdrant.search(
        collection_name=COLLECTION_NAME, query_vector=q_emb, limit=top_k
    )

    # Format results
    hits = []
    for hit in results:
        hits.append(
            {
                "chunk_id": hit.payload.get("chunk_id", "unknown"),
                "text": hit.payload.get("text", ""),
                "modality": hit.payload.get("modality", "text"),
                "page": hit.payload.get("page", "?"),
                "score": hit.score,  # Dense cosine similarity score
            }
        )

    return hits


def dense_plus_rerank(
    query: str,
    qdrant,
    openai_client,
    cross_encoder,
    top_k: int = 20,
    rerank_top_n: int = 4,
) -> List[Dict]:
    """Perform dense retrieval followed by cross-encoder re-ranking"""
    # Embed query
    q_emb = embed_text(openai_client, query)

    # Stage 1: Dense retrieval
    results = qdrant.search(
        collection_name=COLLECTION_NAME, query_vector=q_emb, limit=top_k
    )

    if not results:
        return []

    # Stage 2: Re-ranking
    pairs = [[query, hit.payload.get("text", "")] for hit in results]
    scores = cross_encoder.predict(pairs)

    # Reattach scores and sort
    for hit, score in zip(results, scores):
        hit.score = float(score)  # Overwrite with cross-encoder score

    results.sort(key=lambda x: x.score, reverse=True)

    # Take top_n after re-ranking
    best_hits = results[:rerank_top_n]

    # Format results
    hits = []
    for hit in best_hits:
        hits.append(
            {
                "chunk_id": hit.payload.get("chunk_id", "unknown"),
                "text": hit.payload.get("text", ""),
                "modality": hit.payload.get("modality", "text"),
                "page": hit.payload.get("page", "?"),
                "score": hit.score,  # Cross-encoder score
            }
        )

    return hits


def create_ground_truth() -> Dict[str, List[str]]:
    """Create ground truth relevance judgments"""
    # Based on manual inspection of what chunks should be relevant
    return {
        "query_1": ["test.pdf_1_0"],  # Flowchart question
        "query_2": ["test.pdf_1_0"],  # Positional encoding
        "query_3": ["test.pdf_1_0"],  # Left/right sections
        "query_4": ["test.pdf_1_0"],  # Multi-head attention
        "query_5": ["test.pdf_1_0"],  # After softmax
    }


def create_test_queries() -> Dict[str, str]:
    """Create test queries"""
    return {
        "query_1": "What does the flowchart in the document illustrate?",
        "query_2": "How is positional encoding used in the architecture?",
        "query_3": "What is the difference between the left and right sections of the flowchart?",
        "query_4": "Explain the multi-head attention mechanism steps",
        "query_5": "What comes after the softmax step in the processing?",
    }


def evaluate_retrieval_method(
    method_name: str, retrieval_func, *args
) -> Dict[str, float]:
    """Evaluate a specific retrieval method"""
    queries = create_test_queries()
    ground_truth = create_ground_truth()

    # Retrieve results for each query
    query_results = {}
    for query_id, query_text in queries.items():
        try:
            retrieved_hits = retrieval_func(query_text, *args)
            retrieved_ids = [hit["chunk_id"] for hit in retrieved_hits]
            query_results[query_id] = retrieved_ids
            print(
                f"  {method_name} - Query '{query_id}': Retrieved {len(retrieved_ids)} chunks"
            )
        except Exception as e:
            print(f"  {method_name} - Query '{query_id}': Error - {e}")
            query_results[query_id] = []

    # Calculate metrics
    from evaluate_retrieval import evaluate_retrieval

    metrics = evaluate_retrieval(query_results, ground_truth, ks=[1, 3, 5])
    return metrics


def main():
    """Main function to evaluate re-ranking impact"""
    print("Evaluating impact of re-ranking stage...")
    print("=" * 50)

    # Initialize clients
    try:
        qdrant, openai_client, cross_encoder = initialize_clients()
        print("Clients initialized successfully")
    except Exception as e:
        print(f"Failed to initialize clients: {e}")
        return

    # Evaluate dense retrieval only
    print("\nEvaluating dense retrieval only...")
    dense_metrics = evaluate_retrieval_method(
        "Dense-only", dense_retrieval_only, qdrant, openai_client
    )

    # Evaluate dense + re-ranking
    print("\nEvaluating dense retrieval + re-ranking...")
    rerank_metrics = evaluate_retrieval_method(
        "Dense+Rerank", dense_plus_rerank, qdrant, openai_client, cross_encoder
    )

    # Print results
    print("\n" + "=" * 60)
    print("RE-RANKING IMPACT EVALUATION")
    print("=" * 60)

    print("\nDense Retrieval Only:")
    for metric, value in dense_metrics.items():
        print(f"  {metric}: {value:.4f}")

    print("\nDense Retrieval + Re-Ranking:")
    for metric, value in rerank_metrics.items():
        print(f"  {metric}: {value:.4f}")

    print("\nImpact of Re-Ranking (Improvement %):")
    improvements = {}
    for metric in dense_metrics:
        if metric in rerank_metrics:
            dense_val = dense_metrics[metric]
            rerank_val = rerank_metrics[metric]
            if dense_val != 0:
                improvement = ((rerank_val - dense_val) / dense_val) * 100
            else:
                improvement = 0.0 if rerank_val == 0 else float("inf")
            improvements[f"{metric}_improvement_%"] = improvement

            if improvement != float("inf"):
                print(f"  {metric}: {improvement:+.2f}%")
            else:
                print(f"  {metric}: INF (from 0 to non-zero)")

    # Save results
    results = {
        "dense_only": dense_metrics,
        "dense_plus_rerank": rerank_metrics,
        "improvements": improvements,
        "metadata": {
            "eval_description": "Impact of cross-encoder re-ranking on retrieval quality",
            "dense_model": EMBED_MODEL,
            "reranker_model": CROSS_ENCODER_MODEL,
            "llm_model": LLM_MODEL,
        },
    }

    with open("rerank_impact_evaluation.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nDetailed results saved to: rerank_impact_evaluation.json")

    # Summary
    avg_dense = np.mean(list(dense_metrics.values()))
    avg_rerank = np.mean(list(rerank_metrics.values()))
    overall_improvement = (
        ((avg_rerank - avg_dense) / avg_dense * 100) if avg_dense != 0 else 0
    )

    print(f"\nOverall Performance Change: {overall_improvement:+.2f}%")
    if overall_improvement > 0:
        print("✓ Re-ranking improves retrieval quality")
    else:
        print("✗ Re-ranking does not improve retrieval quality (may need tuning)")


if __name__ == "__main__":
    main()
