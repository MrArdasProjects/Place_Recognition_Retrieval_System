"""Open-set handling with confidence strategies for UNKNOWN prediction."""

from typing import Dict, List, Literal

from place_recognition.logging_config import get_logger

logger = get_logger(__name__)

ConfidenceStrategy = Literal["none", "max_similarity", "margin"]


def apply_open_set_strategy(
    results: List[List[Dict]],
    strategy: ConfidenceStrategy = "none",
    threshold: float = 0.5,
) -> List[List[Dict]]:
    """Mark low-confidence predictions as UNKNOWN.
    
    Args:
        results: Search results from SearchIndex.search()
        strategy: "none", "max_similarity", or "margin"
        threshold: Threshold value
    
    Returns:
        Modified results with UNKNOWN labels for low-confidence predictions
    """
    if strategy == "none":
        # No filtering - return as-is
        return results
    
    modified_results = []
    unknown_count = 0
    
    for query_results in results:
        if not query_results:
            modified_results.append(query_results)
            continue
        
        # Make a copy to avoid modifying original
        query_results_copy = [r.copy() for r in query_results]
        
        top1 = query_results_copy[0]
        top1_sim = top1["similarity"]
        
        # Determine if prediction is confident
        is_confident = True
        
        if strategy == "max_similarity":
            # Strategy 1: Absolute similarity threshold
            if top1_sim < threshold:
                is_confident = False
        
        elif strategy == "margin":
            # Strategy 2: Margin between top-1 and top-2
            if len(query_results_copy) >= 2:
                top2_sim = query_results_copy[1]["similarity"]
                margin = top1_sim - top2_sim
                if margin < threshold:
                    is_confident = False
            else:
                # Only 1 result - use max_similarity fallback
                if top1_sim < threshold:
                    is_confident = False
        
        # Apply UNKNOWN label if not confident
        if not is_confident:
            query_results_copy[0]["place_id"] = "UNKNOWN"
            query_results_copy[0]["confidence_flag"] = "unknown"
            query_results_copy[0]["original_place_id"] = top1["place_id"]
            unknown_count += 1
        else:
            query_results_copy[0]["confidence_flag"] = "confident"
        
        modified_results.append(query_results_copy)
    
    total_queries = len(results)
    unknown_rate = unknown_count / total_queries if total_queries > 0 else 0.0
    
    logger.info(f"Open-set strategy applied: {strategy}")
    logger.info(f"  Threshold: {threshold}")
    logger.info(f"  UNKNOWN predictions: {unknown_count}/{total_queries} ({unknown_rate:.2%})")
    
    return modified_results


def evaluate_open_set_impact(
    original_results: List[List[Dict]],
    filtered_results: List[List[Dict]],
    query_records: List[Dict],
) -> Dict:
    """Evaluate the impact of open-set filtering on metrics.
    
    Compares metrics before and after applying confidence filtering.
    
    Args:
        original_results: Results before filtering
        filtered_results: Results after applying open-set strategy
        query_records: Ground truth query records
    
    Returns:
        Dictionary with impact analysis
    """
    # Count predictions
    total_queries = len(query_records)
    unknown_count = sum(
        1 for results in filtered_results
        if results and results[0]["place_id"] == "UNKNOWN"
    )
    
    # Count correct predictions in top-1 (before and after)
    correct_before = 0
    correct_after = 0
    false_positives_removed = 0
    true_positives_removed = 0
    
    for i, (orig, filt) in enumerate(zip(original_results, filtered_results)):
        query_place = query_records[i]["place_id"]
        
        # Before filtering
        if orig and orig[0]["place_id"] == query_place:
            correct_before += 1
        
        # After filtering
        if filt and filt[0]["place_id"] == query_place:
            correct_after += 1
        elif filt and filt[0]["place_id"] == "UNKNOWN":
            # Check if we removed a correct or incorrect prediction
            if orig[0]["place_id"] == query_place:
                true_positives_removed += 1
            else:
                false_positives_removed += 1
    
    return {
        "total_queries": total_queries,
        "unknown_predictions": unknown_count,
        "unknown_rate": unknown_count / total_queries,
        "recall@1_before": correct_before / total_queries,
        "recall@1_after": correct_after / total_queries,
        "false_positives_removed": false_positives_removed,
        "true_positives_removed": true_positives_removed,
    }
