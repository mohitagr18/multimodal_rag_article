#!/usr/bin/env python3
"""
Evaluation metrics for retrieval quality assessment.
"""

import argparse
import json
import numpy as np
from pathlib import Path
from typing import List, Dict
from collections import defaultdict


def precision_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    if k == 0:
        return 0.0
    retrieved_k = retrieved[:k]
    relevant_set = set(relevant)
    retrieved_set = set(retrieved_k)
    if not retrieved_k:
        return 0.0
    return len(relevant_set & retrieved_set) / len(retrieved_k)


def recall_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    if not relevant:
        return 0.0
    retrieved_k = retrieved[:k]
    relevant_set = set(relevant)
    retrieved_set = set(retrieved_k)
    if not relevant:
        return 0.0
    return len(relevant_set & retrieved_set) / len(relevant_set)


def mean_reciprocal_rank(retrieved: List[str], relevant: List[str]) -> float:
    relevant_set = set(relevant)
    for i, doc_id in enumerate(retrieved):
        if doc_id in relevant_set:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    if k == 0:
        return 0.0
    relevant_set = set(relevant)
    relevance_scores = [1.0 if doc_id in relevant_set else 0.0 for doc_id in retrieved[:k]]
    dcg = 0.0
    for i, rel in enumerate(relevance_scores):
        if i == 0:
            dcg += rel
        else:
            dcg += rel / np.log2(i + 1)
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
    metrics = defaultdict(list)
    common_queries = set(query_results.keys()) & set(ground_truth.keys())
    if not common_queries:
        raise ValueError("No common queries found between results and ground truth")
    for query_id in common_queries:
        retrieved = query_results[query_id]
        relevant = ground_truth[query_id]
        for k in ks:
            metrics[f"precision@{k}"].append(precision_at_k(retrieved, relevant, k))
            metrics[f"recall@{k}"].append(recall_at_k(retrieved, relevant, k))
        metrics["mrr"].append(mean_reciprocal_rank(retrieved, relevant))
        metrics["map"].append(average_precision(retrieved, relevant))
        metrics["ndcg@5"].append(ndcg_at_k(retrieved, relevant, 5))
        metrics["ndcg@10"].append(ndcg_at_k(retrieved, relevant, 10))
    averaged_metrics = {}
    for metric_name, values in metrics.items():
        averaged_metrics[metric_name] = np.mean(values) if values else 0.0
    return dict(averaged_metrics)


def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality")
    parser.add_argument("--naive-results", type=str, required=True)
    parser.add_argument("--structured-results", type=str, required=True)
    parser.add_argument("--ground-truth", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="results")
    args = parser.parse_args()

    naive_results = json.loads(Path(args.naive_results).read_text())
    structured_results = json.loads(Path(args.structured_results).read_text())
    ground_truth = json.loads(Path(args.ground_truth).read_text())

    naive_metrics = evaluate_retrieval(naive_results, ground_truth)
    structured_metrics = evaluate_retrieval(structured_results, ground_truth)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "naive_approach": naive_metrics,
        "structured_approach": structured_metrics,
    }
    (output_dir / "evaluation_results.json").write_text(json.dumps(results, indent=2))

    print("RETRIEVAL EVALUATION RESULTS")
    print("=" * 50)
    print(f"\nNaive Approach: {json.dumps(naive_metrics, indent=2)}")
    print(f"\nStructure-Aware Approach: {json.dumps(structured_metrics, indent=2)}")


if __name__ == "__main__":
    main()