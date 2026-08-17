"""
Unit tests for RRF (Reciprocal Rank Fusion) search result fusion.

Tests cover:
- RRF score calculation
- Recency boost tier application
"""

import pytest

from searchkernel.search.fusion import (
    fuse_calibrated_scores,
    fuse_reciprocal_rank,
    rrf_score,
)


class TestRRFScore:
    """Tests for RRF score calculation."""

    def test_rrf_score_calculation(self):
        """
        Validates RRF score formula: 1 / (k + rank).
        Ensures correct score computation for various ranks and k values.
        """
        # k=60 is the default constant
        assert rrf_score(0, 60) == 1 / 60  # Rank 0 (first position)
        assert rrf_score(1, 60) == 1 / 61  # Rank 1 (second position)
        assert rrf_score(5, 60) == 1 / 65  # Rank 5
        assert rrf_score(10, 60) == 1 / 70  # Rank 10

        # Different k values
        assert rrf_score(0, 10) == 1 / 10
        assert rrf_score(0, 100) == 1 / 100

    def test_rrf_score_decreases_with_rank(self):
        """
        Verifies that RRF scores decrease monotonically with increasing rank.
        Higher ranks should always produce lower scores.
        """
        k = 60
        score_0 = rrf_score(0, k)
        score_1 = rrf_score(1, k)
        score_5 = rrf_score(5, k)
        score_10 = rrf_score(10, k)

        assert score_0 > score_1 > score_5 > score_10


class TestReciprocalRankFusion:
    def test_first_rank_uses_one_based_formula(self):
        assert fuse_reciprocal_rank([["doc1"]], k=60.0) == {"doc1": 1 / 61}

    def test_duplicate_ids_accumulate_across_rankings(self):
        scores = fuse_reciprocal_rank([["doc1", "doc2"], ["doc2", "doc1"]], k=10.0)

        assert scores == {
            "doc1": 1 / 11 + 1 / 12,
            "doc2": 1 / 12 + 1 / 11,
        }

    def test_empty_rankings_return_empty_scores(self):
        assert fuse_reciprocal_rank([[], []]) == {}

    @pytest.mark.parametrize("invalid_k", [0, -1.0])
    def test_non_positive_k_is_rejected(self, invalid_k):
        with pytest.raises(ValueError, match="k must be positive"):
            fuse_reciprocal_rank([["doc1"]], k=invalid_k)

    def test_named_unweighted_rankings_match_plain_rrf(self):
        rankings = [["doc1", "doc2"], ["doc2", "doc1"]]
        assert fuse_reciprocal_rank(rankings) == fuse_reciprocal_rank(
            {"keyword": rankings[0], "vector": rankings[1]}
        )

    def test_default_fusion_remains_rank_based(self):
        """
        Keeps the default fusion contract based on one-based rank positions.
        """
        assert fuse_reciprocal_rank({"keyword": ["doc-a", "doc-b"]}) == {
            "doc-a": 1 / 61,
            "doc-b": 1 / 62,
        }


class TestCalibratedScoreFusion:
    def test_normalizes_each_lane_relative_to_its_own_scores(self):
        """
        Confirms score scales are normalized independently before fusion.
        """
        scores = fuse_calibrated_scores(
            {
                "keyword": [("doc-a", 100.0), ("doc-b", 90.0)],
                "vector": [("doc-a", 0.6), ("doc-b", 0.5)],
            }
        )

        assert scores == {"doc-a": 2.0, "doc-b": 0.0}
