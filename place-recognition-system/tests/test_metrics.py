"""Unit tests for retrieval evaluation metrics.

Tests both single-positive and multi-positive cases to ensure metrics
are calculated correctly according to information retrieval standards.
"""

import pytest

from place_recognition.evaluation.metrics import (
    average_precision,
    compute_metrics,
    mean_average_precision,
    recall_at_k,
)


class TestRecallAtK:
    """Test Recall@K metric."""
    
    def test_recall_at_k_hit(self):
        """Test recall when correct place is in top-K."""
        retrieved = ["eiffel", "louvre", "notre_dame"]
        ground_truth = {"eiffel"}
        
        # Correct place is at position 1
        assert recall_at_k(retrieved, ground_truth, k=1) == 1.0
        assert recall_at_k(retrieved, ground_truth, k=3) == 1.0
    
    def test_recall_at_k_miss(self):
        """Test recall when correct place is NOT in top-K."""
        retrieved = ["louvre", "arc", "sacre"]
        ground_truth = {"eiffel"}
        
        # Correct place not in results
        assert recall_at_k(retrieved, ground_truth, k=1) == 0.0
        assert recall_at_k(retrieved, ground_truth, k=3) == 0.0
    
    def test_recall_at_k_multi_positive(self):
        """Test recall with multiple correct places (multi-positive)."""
        retrieved = ["louvre", "eiffel", "arc"]
        ground_truth = {"eiffel", "eiffel_tower"}  # Multiple valid place_ids
        
        # "eiffel" is at position 2, so not in top-1 but in top-3
        assert recall_at_k(retrieved, ground_truth, k=1) == 0.0
        assert recall_at_k(retrieved, ground_truth, k=2) == 1.0
        assert recall_at_k(retrieved, ground_truth, k=3) == 1.0
    
    def test_recall_at_k_edge_cases(self):
        """Test edge cases for Recall@K."""
        retrieved = ["eiffel"]
        ground_truth = {"eiffel"}
        
        # K=0 should return 0.0
        assert recall_at_k(retrieved, ground_truth, k=0) == 0.0
        
        # Empty ground truth
        assert recall_at_k(retrieved, set(), k=1) == 0.0
        
        # K larger than retrieved list
        assert recall_at_k(retrieved, ground_truth, k=10) == 1.0


class TestAveragePrecision:
    """Test Average Precision (AP) metric."""
    
    def test_perfect_ranking(self):
        """Test AP with perfect ranking (all correct items first)."""
        retrieved = ["eiffel", "eiffel", "louvre", "arc"]
        ground_truth = {"eiffel"}
        
        # Correct at positions 1,2
        # AP = (1/1 + 2/2) / 2 = (1.0 + 1.0) / 2 = 1.0
        ap = average_precision(retrieved, ground_truth)
        assert ap == pytest.approx(1.0)
    
    def test_imperfect_ranking(self):
        """Test AP with imperfect ranking."""
        retrieved = ["louvre", "eiffel", "arc", "eiffel"]
        ground_truth = {"eiffel"}
        
        # Correct at positions 2,4
        # AP = (1/2 + 2/4) / 2 = (0.5 + 0.5) / 2 = 0.5
        ap = average_precision(retrieved, ground_truth)
        assert ap == pytest.approx(0.5)
    
    def test_no_correct_results(self):
        """Test AP when no correct results are retrieved."""
        retrieved = ["louvre", "arc", "sacre"]
        ground_truth = {"eiffel"}
        
        # No correct results: AP = 0.0
        ap = average_precision(retrieved, ground_truth)
        assert ap == pytest.approx(0.0)
    
    def test_single_correct_at_end(self):
        """Test AP when single correct result is at the end."""
        retrieved = ["louvre", "arc", "sacre", "eiffel"]
        ground_truth = {"eiffel"}
        
        # Correct at position 4
        # AP = (1/4) / 1 = 0.25
        ap = average_precision(retrieved, ground_truth)
        assert ap == pytest.approx(0.25)
    
    def test_multi_positive_perfect(self):
        """Test AP with multi-positive case (perfect ranking)."""
        retrieved = ["eiffel_1", "eiffel_2", "louvre", "eiffel_3"]
        ground_truth = {"eiffel_1", "eiffel_2", "eiffel_3"}
        
        # Correct at positions 1,2,4
        # AP = (1/1 + 2/2 + 3/4) / 3 = (1.0 + 1.0 + 0.75) / 3 = 0.9166...
        ap = average_precision(retrieved, ground_truth)
        assert ap == pytest.approx(2.75 / 3, abs=1e-4)


class TestMeanAveragePrecision:
    """Test mean Average Precision (mAP) metric."""
    
    def test_map_perfect(self):
        """Test mAP with perfect results for all queries."""
        retrieved = [
            ["eiffel", "louvre"],  # Query 1
            ["arc", "sacre"],       # Query 2
        ]
        ground_truth = [
            {"eiffel"},  # Perfect: AP=1.0
            {"arc"},     # Perfect: AP=1.0
        ]
        
        map_score = mean_average_precision(retrieved, ground_truth)
        assert map_score == pytest.approx(1.0)
    
    def test_map_mixed(self):
        """Test mAP with mixed performance."""
        retrieved = [
            ["eiffel", "louvre"],      # Query 1: perfect (AP=1.0)
            ["louvre", "arc"],         # Query 2: AP=0.5 (correct at pos 2)
            ["sacre", "notre_dame"],   # Query 3: AP=0.0 (no correct)
        ]
        ground_truth = [
            {"eiffel"},
            {"arc"},
            {"pantheon"},
        ]
        
        # mAP = (1.0 + 0.5 + 0.0) / 3 = 0.5
        map_score = mean_average_precision(retrieved, ground_truth)
        assert map_score == pytest.approx(0.5)
    
    def test_map_mismatch_lengths(self):
        """Test mAP raises error on mismatched lengths."""
        retrieved = [["eiffel"], ["louvre"]]
        ground_truth = [{"eiffel"}]  # Mismatch!
        
        with pytest.raises(ValueError, match="Mismatch"):
            mean_average_precision(retrieved, ground_truth)


class TestComputeMetrics:
    """Test compute_metrics function (main evaluation function)."""
    
    def test_single_positive_case(self):
        """Test metrics with single-positive case (1 correct per query).
        
        This is the standard case where each query has exactly one correct
        gallery image.
        """
        # Setup: 3 queries with search results
        query_results = [
            # Query 1: eiffel (correct at rank 1)
            [
                {"place_id": "eiffel", "similarity": 0.95, "rank": 0},
                {"place_id": "louvre", "similarity": 0.85, "rank": 1},
                {"place_id": "arc", "similarity": 0.75, "rank": 2},
            ],
            # Query 2: louvre (correct at rank 2)
            [
                {"place_id": "eiffel", "similarity": 0.90, "rank": 0},
                {"place_id": "louvre", "similarity": 0.88, "rank": 1},
                {"place_id": "arc", "similarity": 0.70, "rank": 2},
            ],
            # Query 3: arc (correct at rank 3)
            [
                {"place_id": "eiffel", "similarity": 0.92, "rank": 0},
                {"place_id": "louvre", "similarity": 0.87, "rank": 1},
                {"place_id": "arc", "similarity": 0.80, "rank": 2},
            ],
        ]
        
        query_records = [
            {"place_id": "eiffel", "image_path": "eiffel/query/img1.jpg"},
            {"place_id": "louvre", "image_path": "louvre/query/img1.jpg"},
            {"place_id": "arc", "image_path": "arc/query/img1.jpg"},
        ]
        
        metrics = compute_metrics(query_results, query_records, ks=[1, 2, 3])
        
        # Query 1: correct at rank 1 ✓
        # Query 2: correct at rank 2 ✗ (not in top-1)
        # Query 3: correct at rank 3 ✗ (not in top-1)
        # Recall@1 = 1/3 = 0.333...
        assert metrics["recall@1"] == pytest.approx(1/3, abs=1e-6)
        
        # Query 1: correct at rank 1 ✓
        # Query 2: correct at rank 2 ✓
        # Query 3: correct at rank 3 ✗ (not in top-2)
        # Recall@2 = 2/3 = 0.666...
        assert metrics["recall@2"] == pytest.approx(2/3, abs=1e-6)
        
        # All queries have correct result in top-3
        # Recall@3 = 3/3 = 1.0
        assert metrics["recall@3"] == pytest.approx(1.0)
        
        # AP calculations:
        # Query 1: correct at pos 1 → AP = 1/1 = 1.0
        # Query 2: correct at pos 2 → AP = 1/2 = 0.5
        # Query 3: correct at pos 3 → AP = 1/3 = 0.333...
        # mAP = (1.0 + 0.5 + 0.333...) / 3 = 0.611...
        expected_map = (1.0 + 0.5 + 1/3) / 3
        assert metrics["mAP"] == pytest.approx(expected_map, abs=1e-6)
        
        assert metrics["num_queries"] == 3
    
    def test_multi_positive_case(self):
        """Test metrics with multi-positive case (multiple correct per query).
        
        This tests the case where a query can have multiple correct gallery
        images (e.g., multiple photos of the same place). All should be
        counted as positive.
        """
        # Setup: 2 queries where each query's place appears multiple times
        query_results = [
            # Query 1: eiffel (appears at ranks 1, 3, 5)
            [
                {"place_id": "eiffel", "similarity": 0.95, "rank": 0},    # ✓
                {"place_id": "louvre", "similarity": 0.92, "rank": 1},    # ✗
                {"place_id": "eiffel", "similarity": 0.90, "rank": 2},    # ✓
                {"place_id": "arc", "similarity": 0.85, "rank": 3},       # ✗
                {"place_id": "eiffel", "similarity": 0.80, "rank": 4},    # ✓
            ],
            # Query 2: louvre (appears at ranks 2, 4)
            [
                {"place_id": "arc", "similarity": 0.93, "rank": 0},       # ✗
                {"place_id": "louvre", "similarity": 0.91, "rank": 1},    # ✓
                {"place_id": "eiffel", "similarity": 0.88, "rank": 2},    # ✗
                {"place_id": "louvre", "similarity": 0.85, "rank": 3},    # ✓
                {"place_id": "sacre", "similarity": 0.80, "rank": 4},     # ✗
            ],
        ]
        
        query_records = [
            {"place_id": "eiffel", "image_path": "eiffel/query/img1.jpg"},
            {"place_id": "louvre", "image_path": "louvre/query/img1.jpg"},
        ]
        
        metrics = compute_metrics(query_results, query_records, ks=[1, 3, 5])
        
        # Query 1: eiffel at rank 1 ✓
        # Query 2: louvre at rank 2 ✗
        # Recall@1 = 1/2 = 0.5
        assert metrics["recall@1"] == pytest.approx(0.5)
        
        # Query 1: eiffel at ranks 1,3 ✓
        # Query 2: louvre at rank 2 ✓
        # Recall@3 = 2/2 = 1.0
        assert metrics["recall@3"] == pytest.approx(1.0)
        
        # Both queries have correct results in top-5
        assert metrics["recall@5"] == pytest.approx(1.0)
        
        # AP calculations with multi-positive:
        # Query 1: eiffel at positions 1,3,5
        #   - Pos 1: precision = 1/1 = 1.0
        #   - Pos 3: precision = 2/3 = 0.666...
        #   - Pos 5: precision = 3/5 = 0.6
        #   AP = (1.0 + 0.666... + 0.6) / 3 = 0.755...
        # Query 2: louvre at positions 2,4
        #   - Pos 2: precision = 1/2 = 0.5
        #   - Pos 4: precision = 2/4 = 0.5
        #   AP = (0.5 + 0.5) / 2 = 0.5
        # mAP = (0.755... + 0.5) / 2 = 0.627...
        expected_ap1 = (1.0 + 2/3 + 3/5) / 3
        expected_ap2 = (1/2 + 2/4) / 2
        expected_map = (expected_ap1 + expected_ap2) / 2
        assert metrics["mAP"] == pytest.approx(expected_map, abs=1e-6)
        
        assert metrics["num_queries"] == 2
    
    def test_empty_results(self):
        """Test metrics with empty results."""
        metrics = compute_metrics([], [], ks=[1, 5])
        
        assert metrics["recall@1"] == 0.0
        assert metrics["recall@5"] == 0.0
        assert metrics["mAP"] == 0.0
        assert metrics["num_queries"] == 0
    
    def test_all_failures(self):
        """Test metrics when all queries fail (no correct results)."""
        query_results = [
            [
                {"place_id": "louvre", "similarity": 0.95, "rank": 0},
                {"place_id": "arc", "similarity": 0.85, "rank": 1},
            ],
            [
                {"place_id": "eiffel", "similarity": 0.90, "rank": 0},
                {"place_id": "sacre", "similarity": 0.80, "rank": 1},
            ],
        ]
        
        query_records = [
            {"place_id": "eiffel", "image_path": "eiffel/query/img1.jpg"},
            {"place_id": "louvre", "image_path": "louvre/query/img1.jpg"},
        ]
        
        metrics = compute_metrics(query_results, query_records, ks=[1, 2])
        
        # No correct results for any query
        assert metrics["recall@1"] == 0.0
        assert metrics["recall@2"] == 0.0
        assert metrics["mAP"] == 0.0
        assert metrics["num_queries"] == 2
    
    def test_perfect_results(self):
        """Test metrics when all queries have perfect results."""
        query_results = [
            [
                {"place_id": "eiffel", "similarity": 0.95, "rank": 0},
                {"place_id": "louvre", "similarity": 0.85, "rank": 1},
            ],
            [
                {"place_id": "louvre", "similarity": 0.90, "rank": 0},
                {"place_id": "eiffel", "similarity": 0.80, "rank": 1},
            ],
        ]
        
        query_records = [
            {"place_id": "eiffel", "image_path": "eiffel/query/img1.jpg"},
            {"place_id": "louvre", "image_path": "louvre/query/img1.jpg"},
        ]
        
        metrics = compute_metrics(query_results, query_records, ks=[1, 2])
        
        # All queries have correct result at rank 1
        assert metrics["recall@1"] == 1.0
        assert metrics["recall@2"] == 1.0
        assert metrics["mAP"] == 1.0
        assert metrics["num_queries"] == 2


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_mismatched_lengths(self):
        """Test that mismatched lengths raise appropriate errors."""
        query_results = [
            [{"place_id": "eiffel", "similarity": 0.95, "rank": 0}],
        ]
        query_records = [
            {"place_id": "eiffel", "image_path": "eiffel/query/img1.jpg"},
            {"place_id": "louvre", "image_path": "louvre/query/img1.jpg"},
        ]
        
        with pytest.raises(ValueError, match="Mismatch"):
            compute_metrics(query_results, query_records, ks=[1])
    
    def test_empty_ground_truth(self):
        """Test behavior with empty ground truth set."""
        assert recall_at_k(["eiffel"], set(), k=1) == 0.0
        assert average_precision(["eiffel"], set()) == 0.0
    
    def test_zero_k(self):
        """Test Recall@0 (edge case)."""
        assert recall_at_k(["eiffel"], {"eiffel"}, k=0) == 0.0
