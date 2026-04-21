#!/usr/bin/env python3
"""
Script to run retrieval evaluation comparing naive vs structure-aware approaches using actual retrieval pipeline.
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import sys
import os

# Add the article_prototype directory to path
sys.path.append(str(Path(__file__).parent))

from evaluate_retrieval import (
    evaluate_retrieval,
    precision_at_k,
    recall_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
    average_precision,
)
import yaml

# Load configuration
config_path = Path(__file__).parent / "config.yaml"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

DEFAULT_EMBED_MODEL = config["models"]["embedding"]


def load_chunks(filepath: Path) -> List[Dict]:
    """Load chunks from JSON file"""
    with open(filepath, "r") as f:
        return json.load(f)


def embed_text_openai(text: str, model: str = None) -> List[float]:
    """Generate embedding using Ollama's OpenAI-compatible API"""
    from openai import OpenAI

    if model is None:
        model = DEFAULT_EMBED_MODEL

    client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    response = client.embeddings.create(model=model, input=text)
    return response.data[0].embedding


def get_retrieval_results_from_qdrant(
    chunks: List[Dict], queries: Dict[str, str], embedding_model: str = None
) -> Dict[str, List[str]]:
    """
    Get actual retrieval results from Qdrant using the embedding model
    Returns dict mapping query_id -> list of retrieved chunk_ids
    """
    from qdrant_client import QdrantClient

    # Initialize clients
    qdrant = QdrantClient(path="qdrant_db")

    query_results = {}

    for query_id, query_text in queries.items():
        # Get query embedding
        query_embedding = embed_text_openai(query_text, embedding_model)

        # Search in Qdrant
        search_results = qdrant.query_points(
            collection_name="article_chunks",
            query=query_embedding,
            limit=10,  # Get top 10 for evaluation
        )

        # Extract chunk IDs from results
        retrieved_chunk_ids = []
        for point in search_results.points:
            chunk_id = point.payload.get("chunk_id")
            if chunk_id:
                retrieved_chunk_ids.append(chunk_id)

        query_results[query_id] = retrieved_chunk_ids
        print(
            f"  Query '{query_id}': Retrieved {len(retrieved_chunk_ids)} chunks via {embedding_model}"
        )

    return query_results


def create_ground_truth() -> Dict[str, List[str]]:
    """Create ground truth relevance judgments for our test queries"""
    # Based on manual inspection of the chunks - which chunks are truly relevant
    # For our test document, we'll mark chunks that are most relevant to each query
    return {
        "query_1": ["test.pdf_1_0"],  # Flowchart question - the main image chunk
        "query_2": [
            "test.pdf_1_0",
            "test.pdf_2_1",
        ],  # Positional encoding - image chunk and table chunk
        "query_3": [
            "test.pdf_1_0"
        ],  # Left/right sections - the flowchart shows both sections
        "query_4": [
            "test.pdf_1_0",
            "test.pdf_1_3",
        ],  # Multi-head attention - image caption and attention section
        "query_5": [
            "test.pdf_1_0",
            "test.pdf_1_4",
        ],  # After softmax - image caption and linearization step
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


def evaluate_approach(
    approach_name: str, chunks_file: Path, embedding_model: str = "embeddinggemma"
) -> Dict[str, float]:
    """Evaluate a specific approach using actual retrieval from Qdrant"""
    print(f"\nEvaluating {approach_name} approach...")

    # Load chunks (for reference, though we get results from Qdrant)
    chunks = load_chunks(chunks_file)
    print(f"Loaded {len(chunks)} chunks from {chunks_file.name}")

    # Create test queries and ground truth
    queries = create_test_queries()
    ground_truth = create_ground_truth()

    # Get actual retrieval results using Qdrant
    query_results = get_retrieval_results_from_qdrant(chunks, queries, embedding_model)

    # Calculate metrics
    metrics = evaluate_retrieval(query_results, ground_truth, ks=[1, 3, 5])

    print(f"  Metrics for {approach_name}:")
    for metric, value in metrics.items():
        print(f"    {metric}: {value:.4f}")

    return metrics


def run_retrieval_evaluation():
    """Main evaluation function"""
    print("Starting retrieval evaluation with actual pipeline...")
    print("=" * 60)

    # We'll evaluate using the same Qdrant collection but filter by approach based on chunk metadata
    # For now, let's just evaluate what's currently in Qdrant (which should be the enriched version)

    # First, let's check what's actually in Qdrant
    from qdrant_client import QdrantClient

    qdrant = QdrantClient(path="qdrant_db")

    # Get collection info
    collection_info = qdrant.get_collection("article_chunks")
    print(
        f"Qdrant collection 'article_chunks' has {collection_info.points_count} points"
    )

    # Sample a few points to understand what we have
    sample_points = qdrant.scroll(
        collection_name="article_chunks", limit=3, with_payload=True
    )[0]
    print("\nSample points in collection:")
    for i, point in enumerate(sample_points):
        print(
            f"  {i + 1}. Chunk ID: {point.payload.get('chunk_id', 'N/A')}, "
            f"Modality: {point.payload.get('modality', 'N/A')}, "
            f"Page: {point.payload.get('page', 'N/A')}"
        )

    # Now evaluate using actual retrieval
    queries = create_test_queries()
    ground_truth = create_ground_truth()

    print(
        f"\nRunning evaluation on {len(queries)} test queries using actual Qdrant retrieval..."
    )

    # Get actual retrieval results
    query_results = get_retrieval_results_from_qdrant([], queries, DEFAULT_EMBED_MODEL)

    # Calculate metrics
    results = evaluate_retrieval(query_results, ground_truth, ks=[1, 3, 5])

    print(f"\nResults for current Qdrant collection ({DEFAULT_EMBED_MODEL}):")
    for metric, value in results.items():
        print(f"  {metric}: {value:.4f}")

    # Calculate improvement
    print(f"\nImprovement (embeddinggemma vs gemma2:2b):")
    improvements = {}
    for metric in results:
        if metric in results_gemma:
            gemma_val = results_gemma[metric]
            embgemma_val = results[metric]
            if gemma_val != 0:
                improvement = ((embgemma_val - gemma_val) / gemma_val) * 100
            else:
                improvement = 0.0 if embgemma_val == 0 else float("inf")
            improvements[f"{metric}_improvement_%"] = improvement

            if improvement != float("inf"):
                print(f"  {metric}: {improvement:+.2f}%")
            else:
                print(f"  {metric}: INF (from 0 to non-zero)")

    # Save detailed results
    output_data = {
        "queries": queries,
        "ground_truth": ground_truth,
        "results": results,
        "metadata": {
            "eval_timestamp": str(Path(__file__).stat().st_mtime),
            "embedding_model": DEFAULT_EMBED_MODEL,
            "qdrant_points": collection_info.points_count,
        },
    }

    with open("evaluation_detailed_results.json", "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nDetailed results saved to: evaluation_detailed_results.json")

    return output_data


if __name__ == "__main__":
    run_retrieval_evaluation()
