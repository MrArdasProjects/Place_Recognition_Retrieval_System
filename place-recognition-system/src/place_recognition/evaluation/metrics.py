"""Retrieval evaluation metrics: Recall@K and mAP.

Implements standard IR metrics with multi-positive query support.
"""

from typing import Dict, List, Set, Tuple

import numpy as np

from place_recognition.logging_config import get_logger

logger = get_logger(__name__)


def recall_at_k(
    retrieved_place_ids: List[str],
    ground_truth_place_ids: Set[str],
    k: int,
) -> float:
    """Calculate Recall@K for a single query.
    
    Returns 1.0 if any correct place_id appears in top-K, else 0.0.
    
    Args:
        retrieved_place_ids: Retrieved place_ids ranked by similarity
        ground_truth_place_ids: Correct place_ids (supports multi-positive)
        k: Number of top results to consider
    
    Returns:
        1.0 if hit, 0.0 if miss
    """
    if k <= 0 or not ground_truth_place_ids:
        return 0.0
    
    top_k = retrieved_place_ids[:k]
    
    for place_id in top_k:
        if place_id in ground_truth_place_ids:
            return 1.0
    
    return 0.0


def average_precision(
    retrieved_place_ids: List[str],
    ground_truth_place_ids: Set[str],
) -> float:
    """Calculate Average Precision for a single query.
    
    AP = (1/R) * sum of precision at each relevant position.
    
    Args:
        retrieved_place_ids: Retrieved place_ids ranked by similarity
        ground_truth_place_ids: Correct place_ids (supports multi-positive)
    
    Returns:
        AP score in [0, 1]
    """
    if not ground_truth_place_ids:
        return 0.0
    
    num_correct = 0
    sum_precisions = 0.0
    
    for i, place_id in enumerate(retrieved_place_ids, start=1):
        if place_id in ground_truth_place_ids:
            num_correct += 1
            precision_at_i = num_correct / i
            sum_precisions += precision_at_i
    
    if num_correct == 0:
        return 0.0
    
    return sum_precisions / num_correct


def mean_average_precision(
    all_retrieved_place_ids: List[List[str]],
    all_ground_truth_place_ids: List[Set[str]],
) -> float:
    """Calculate mean Average Precision (mAP) across all queries.
    
    Args:
        all_retrieved_place_ids: Retrieved place_ids per query
        all_ground_truth_place_ids: Ground truth per query
    
    Returns:
        mAP score in [0, 1]
    """
    if len(all_retrieved_place_ids) != len(all_ground_truth_place_ids):
        raise ValueError(
            f"Mismatch: {len(all_retrieved_place_ids)} retrieved != "
            f"{len(all_ground_truth_place_ids)} ground truth"
        )
    
    if len(all_retrieved_place_ids) == 0:
        return 0.0
    
    ap_scores = []
    for retrieved, ground_truth in zip(all_retrieved_place_ids, all_ground_truth_place_ids):
        ap = average_precision(retrieved, ground_truth)
        ap_scores.append(ap)
    
    return float(np.mean(ap_scores))


def compute_metrics(
    query_results: List[List[Dict]],
    query_records: List[Dict],
    ks: List[int],
) -> Dict[str, float]:
    """Compute Recall@K and mAP for search results.
    
    Args:
        query_results: Search results from SearchIndex.search()
        query_records: Ground truth query records
        ks: K values for Recall@K
    
    Returns:
        Dict with recall@k values, mAP, and num_queries
    """
    if len(query_results) != len(query_records):
        raise ValueError(
            f"Mismatch: {len(query_results)} results != {len(query_records)} queries"
        )
    
    num_queries = len(query_records)
    
    if num_queries == 0:
        logger.warning("No queries to evaluate")
        return {f"recall@{k}": 0.0 for k in ks} | {"mAP": 0.0, "num_queries": 0}
    
    all_retrieved_place_ids = []
    all_ground_truth_place_ids = []
    
    for query_idx, query_results_list in enumerate(query_results):
        query_place_id = query_records[query_idx]["place_id"]
        ground_truth = {query_place_id}
        retrieved = [result["place_id"] for result in query_results_list]
        
        all_retrieved_place_ids.append(retrieved)
        all_ground_truth_place_ids.append(ground_truth)
    
    recall_scores = {}
    for k in ks:
        recalls = [
            recall_at_k(retrieved, ground_truth, k)
            for retrieved, ground_truth in zip(all_retrieved_place_ids, all_ground_truth_place_ids)
        ]
        recall_scores[f"recall@{k}"] = float(np.mean(recalls))
    
    map_score = mean_average_precision(all_retrieved_place_ids, all_ground_truth_place_ids)
    
    metrics = recall_scores | {
        "mAP": map_score,
        "num_queries": num_queries,
    }
    
    return metrics


def evaluate_results(
    query_results: List[List[Dict]],
    query_records: List[Dict],
    ks: List[int] = [1, 5, 10],
) -> Dict[str, float]:
    """Compute and log Recall@K and mAP metrics.
    
    Args:
        query_results: Search results from SearchIndex.search()
        query_records: Ground truth query records
        ks: K values for Recall@K
    
    Returns:
        Dictionary of metrics
    """
    logger.info("Evaluating retrieval results...")
    
    metrics = compute_metrics(query_results, query_records, ks)
    
    logger.info("Evaluation Results:")
    logger.info(f"  Queries: {metrics['num_queries']}")
    
    for k in sorted(ks):
        recall_key = f"recall@{k}"
        if recall_key in metrics:
            logger.info(f"  Recall@{k}: {metrics[recall_key]:.2%}")
    
    logger.info(f"  mAP: {metrics['mAP']:.2%}")
    
    return metrics


def analyze_failures(
    query_results: List[List[Dict]],
    query_records: List[Dict],
    top_k: int = 5,
) -> Dict[str, List[Dict]]:
    """Identify queries where correct place not in top-K results.
    
    Args:
        query_results: Search results from SearchIndex.search()
        query_records: Ground truth query records
        top_k: Number of top results to check
    
    Returns:
        Dict with failures list, num_failures, failure_rate, top_k
    """
    failures = []
    
    for query_idx, query_results_list in enumerate(query_results):
        query_record = query_records[query_idx]
        query_place_id = query_record["place_id"]
        retrieved_place_ids = [r["place_id"] for r in query_results_list[:top_k]]
        
        if query_place_id not in retrieved_place_ids:
            failure_info = {
                "query_idx": query_idx,
                "query_place_id": query_place_id,
                "query_image_path": query_record["image_path"],
                "top1_place_id": retrieved_place_ids[0] if retrieved_place_ids else None,
                "top1_similarity": query_results_list[0]["similarity"] if query_results_list else 0.0,
                "retrieved_place_ids": retrieved_place_ids,
            }
            failures.append(failure_info)
    
    num_failures = len(failures)
    num_queries = len(query_records)
    failure_rate = num_failures / num_queries if num_queries > 0 else 0.0
    
    logger.info(f"Failure Analysis (top-{top_k}):")
    logger.info(f"  Total failures: {num_failures}/{num_queries}")
    logger.info(f"  Failure rate: {failure_rate:.2%}")
    
    return {
        "failures": failures,
        "num_failures": num_failures,
        "failure_rate": failure_rate,
        "top_k": top_k,
    }
