"""Failure pattern analysis for retrieval system."""

from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import numpy as np

from place_recognition.logging_config import get_logger

logger = get_logger(__name__)


def analyze_failure_patterns(
    query_results: List[List[Dict]],
    query_records: List[Dict],
    top_k: int = 5,
) -> Dict:
    """Analyze failure patterns and identify confusion pairs.
    
    Args:
        query_results: Search results from SearchIndex.search()
        query_records: Ground truth query records
        top_k: Number of top results to consider
    
    Returns:
        Dictionary with confusion pairs and similarity margin statistics
    """
    confusion_pairs = []
    similarity_margins = []
    
    for query_idx, query_results_list in enumerate(query_results):
        query_place = query_records[query_idx]["place_id"]
        top_k_results = query_results_list[:top_k]
        
        top1_place = top_k_results[0]["place_id"]
        top1_sim = top_k_results[0]["similarity"]
        
        # Check if failed
        retrieved_places = [r["place_id"] for r in top_k_results]
        if query_place not in retrieved_places:
            # Complete failure
            confusion_pairs.append((query_place, top1_place))
            similarity_margins.append(None)  # No correct match in top-k
        elif top1_place != query_place:
            # Partial failure (correct in top-k but not top-1)
            confusion_pairs.append((query_place, top1_place))
            
            # Find correct match
            for r in top_k_results:
                if r["place_id"] == query_place:
                    correct_sim = r["similarity"]
                    margin = top1_sim - correct_sim
                    similarity_margins.append(margin)
                    break
    
    # Count confusion pairs
    confusion_counts = Counter(confusion_pairs)
    top_confusions = confusion_counts.most_common(10)
    
    # Margin statistics
    valid_margins = [m for m in similarity_margins if m is not None]
    if valid_margins:
        margin_stats = {
            "mean": float(np.mean(valid_margins)),
            "std": float(np.std(valid_margins)),
            "min": float(np.min(valid_margins)),
            "max": float(np.max(valid_margins)),
            "count": len(valid_margins),
        }
    else:
        margin_stats = {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "count": 0}
    
    return {
        "total_failures": len(confusion_pairs),
        "confusion_pairs": top_confusions,
        "margin_stats": margin_stats,
        "complete_failures": len([m for m in similarity_margins if m is None]),
    }


def print_failure_analysis(analysis: Dict) -> None:
    """Print structured failure analysis to console."""
    logger.info("="*70)
    logger.info("FAILURE PATTERN ANALYSIS")
    logger.info("="*70)
    
    logger.info(f"\nTotal Failures: {analysis['total_failures']}")
    logger.info(f"Complete Failures (not in top-k): {analysis['complete_failures']}")
    
    logger.info("\nTop Confusion Pairs:")
    for i, ((query, retrieved), count) in enumerate(analysis['confusion_pairs'][:5], 1):
        logger.info(f"  {i}. {query} → {retrieved} ({count} times)")
    
    logger.info("\nSimilarity Margin Statistics:")
    margin = analysis['margin_stats']
    logger.info(f"  Mean margin: {margin['mean']:.4f}")
    logger.info(f"  Std margin:  {margin['std']:.4f}")
    logger.info(f"  Min margin:  {margin['min']:.4f}")
    logger.info(f"  Max margin:  {margin['max']:.4f}")
    
    logger.info("\n" + "="*70)
