#!/usr/bin/env python3
"""
Evaluation metrics for retrieval quality assessment.
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict


def precision_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    """Calculate Precision@k"""
    if k == 0:
        return 0.0
    retrieved_k = retrieved[:k]
    relevant_set = set(relevant)
    retrieved_set = set(retrieved_k)
    if not retrieved_k:
        return 0.0
    return len(relevant_set & retrieved_set) / len(retrieved_k)


def recall_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    """Calculate Recall@k"""
    if not relevant:
        return 0.0
    retrieved_k = retrieved[:k]
    relevant_set = set(relevant)
    retrieved_set = set(retrieved_k)
    if not relevant:
        return 0.0
    return len(relevant_set & retrieved_set) / len(relevant_set)


def mean_reciprocal_rank(retrieved: List[str], relevant: List[str]) -> float:
    """Calculate Mean Reciprocal Rank (MRR)"""
    relevant_set = set(relevant)
    for i, doc_id in enumerate(retrieved):
        if doc_id in relevant_set:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    """Calculate Normalized Discounted Cumulative Gain (NDCG)@k"""
    if k == 0:
        return 0.0

    # Create relevance scores (1 if relevant, 0 otherwise)
    relevant_set = set(relevant)
    relevance_scores = [
        1.0 if doc_id in relevant_set else 0.0 for doc_id in retrieved[:k]
    ]

    # Calculate DCG
    dcg = 0.0
    for i, rel in enumerate(relevance_scores):
        if i == 0:
            dcg += rel
        else:
            dcg += rel / np.log2(i + 1)

    # Calculate IDCG (ideal DCG)
    ideal_relevance = sorted(relevance_scores, reverse=True)
    idcg = 0.0
    for i, rel in enumerate(ideal_relevance[:k]):
        if i == 0:
            idcg += rel
        else:
            idcg += rel / np.log2(i + 1)

    if idcg == 0:
        return 0.0
    return dcg / idcg


def average_precision(retrieved: List[str], relevant: List[str]) -> float:
    """Calculate Average Precision (AP)"""
    if not relevant:
        return 0.0

    relevant_set = set(relevant)
    hits = 0
    sum_precisions = 0.0

    for i, doc_id in enumerate(retrieved):
        if doc_id in relevant_set:
            hits += 1
            precision_at_i = hits / (i + 1)
            sum_precisions += precision_at_i

    if hits == 0:
        return 0.0
    return sum_precisions / len(relevant_set)


def evaluate_retrieval(
    query_results: Dict[str, List[str]],
    ground_truth: Dict[str, List[str]],
    ks: List[int] = [1, 3, 5, 10],
) -> Dict[str, float]:
    """
    Evaluate retrieval quality across multiple queries.

    Args:
        query_results: Dictionary mapping query_id -> list of retrieved document IDs
        ground_truth: Dictionary mapping query_id -> list of relevant document IDs
        ks: List of k values for Precision@k and Recall@k

    Returns:
        Dictionary of evaluation metrics
    """
    metrics = defaultdict(list)

    # Ensure we have the same queries in both
    common_queries = set(query_results.keys()) & set(ground_truth.keys())

    if not common_queries:
        raise ValueError("No common queries found between results and ground truth")

    for query_id in common_queries:
        retrieved = query_results[query_id]
        relevant = ground_truth[query_id]

        # Calculate metrics for different k values
        for k in ks:
            metrics[f"precision@{k}"].append(precision_at_k(retrieved, relevant, k))
            metrics[f"recall@{k}"].append(recall_at_k(retrieved, relevant, k))

        metrics["mrr"].append(mean_reciprocal_rank(retrieved, relevant))
        metrics["map"].append(average_precision(retrieved, relevant))
        metrics["ndcg@5"].append(ndcg_at_k(retrieved, relevant, 5))
        metrics["ndcg@10"].append(ndcg_at_k(retrieved, relevant, 10))

    # Average metrics across all queries
    averaged_metrics = {}
    for metric_name, values in metrics.items():
        averaged_metrics[metric_name] = np.mean(values) if values else 0.0

    return dict(averaged_metrics)


def load_json_file(filepath: Path) -> Dict:
    """Load JSON file safely"""
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    with open(filepath, "r") as f:
        return json.load(f)


def save_results(results: Dict, output_path: Path):
    """Save evaluation results to JSON file"""
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)


def compare_approaches(
    naive_results_file: Path,
    structured_results_file: Path,
    ground_truth_file: Path,
    output_dir: Path,
):
    """Compare naive vs structure-aware retrieval approaches"""

    print(f"Loading results...")
    naive_results = load_json_file(naive_results_file)
    structured_results = load_json_file(structured_results_file)
    ground_truth = load_json_file(ground_truth_file)

    print(f"Evaluating naive approach...")
    naive_metrics = evaluate_retrieval(naive_results, ground_truth)

    print(f"Evaluating structure-aware approach...")
    structured_metrics = evaluate_retrieval(structured_results, ground_truth)

    # Calculate improvement
    improvements = {}
    for metric in naive_metrics:
        if metric in structured_metrics:
            if naive_metrics[metric] != 0:
                improvement = (
                    (structured_metrics[metric] - naive_metrics[metric])
                    / naive_metrics[metric]
                ) * 100
            else:
                improvement = 0.0 if structured_metrics[metric] == 0 else float("inf")
            improvements[f"{metric}_improvement_%"] = improvement

    # Prepare final results
    evaluation_results = {
        "naive_approach": naive_metrics,
        "structured_approach": structured_metrics,
        "improvements": improvements,
        "summary": {
            "best_approach": "structured"
            if sum(structured_metrics.values()) > sum(naive_metrics.values())
            else "naive",
            "metrics_compared": list(naive_metrics.keys()),
        },
    }

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    save_results(evaluation_results, output_dir / "evaluation_results.json")

    # Print summary
    print("\n" + "=" * 50)
    print("RETRIEVAL EVALUATION RESULTS")
    print("=" * 50)
    print(f"\nNaive Approach:")
    for metric, value in naive_metrics.items():
        print(f"  {metric}: {value:.4f}")

    print(f"\nStructure-Aware Approach:")
    for metric, value in structured_metrics.items():
        print(f"  {metric}: {value:.4f}")

    print(f"\nImprovements (%):")
    for metric, value in improvements.items():
        if value != float("inf"):
            print(f"  {metric}: {value:+.2f}%")
        else:
            print(f"  {metric}: INF (from 0 to non-zero)")

    print(f"\nResults saved to: {output_dir / 'evaluation_results.json'}")

    return evaluation_results


def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality")
    parser.add_argument(
        "--naive-results",
        type=str,
        required=True,
        help="Path to naive approach results JSON file",
    )
    parser.add_argument(
        "--structured-results",
        type=str,
        required=True,
        help="Path to structure-aware approach results JSON file",
    )
    parser.add_argument(
        "--ground-truth", type=str, required=True, help="Path to ground truth JSON file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./evaluation_results",
        help="Directory to save evaluation results",
    )

    args = parser.parse_args()

    compare_approaches(
        Path(args.naive_results),
        Path(args.structured_results),
        Path(args.ground_truth),
        Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
