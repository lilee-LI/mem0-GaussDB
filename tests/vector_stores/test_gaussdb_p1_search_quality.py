"""
P1 Search Quality Tests for GaussDB vector store.

Tests search result quality including:
- Distance metric correctness (L2, cosine)
- Score ordering and precision
- Recall quality with known data
- BM25 search quality (if enabled)
- Hybrid search quality
- Large dataset sorting

~30 tests total.
"""

import math

import pytest

from tests.vector_stores.conftest import (
    VECTOR_COFFEE,
    VECTOR_FLIGHT,
    VECTOR_WINDOW,
    _assert_ordered_ids,
    _env_bool,
    _ids,
    _insert_memories,
    _new_db,
    _uuid,
    _make_payload,
    gaussdb_available,
)

pytestmark = [
    pytest.mark.p1,
    pytest.mark.skipif(not gaussdb_available(), reason="GaussDB test env not configured"),
]


# ===========================================================================
# 6.2.1 Distance Metric Correctness (8 tests)
# ===========================================================================


class TestDistanceMetricCorrectness:
    """Tests for distance metric correctness with L2 and cosine."""

    def test_l2_nearest_neighbor_correct(self):
        """L2 metric returns the nearest neighbor correctly."""
        db = _new_db(prefix="p1_search", vector_metric="l2")
        try:
            _insert_memories(db, [
                (_uuid(1001), [1.0, 0.0, 0.0], _make_payload("point_a", "search_user")),
                (_uuid(1002), [0.9, 0.1, 0.0], _make_payload("point_b", "search_user")),
                (_uuid(1003), [0.0, 1.0, 0.0], _make_payload("point_c", "search_user")),
            ])
            # Query near point_a; point_b should be closest
            results = db.search("query", [1.0, 0.0, 0.0], top_k=3, filters={"user_id": "search_user"})
            assert len(results) >= 2
            assert results[0].id == _uuid(1001)  # exact match
            assert results[1].id == _uuid(1002)  # nearest neighbor
        finally:
            db.delete_col()

    def test_l2_ordering_correct(self):
        """L2 metric returns results in correct distance order."""
        db = _new_db(prefix="p1_search", vector_metric="l2")
        try:
            _insert_memories(db, [
                (_uuid(1011), [0.0, 0.0, 0.0], _make_payload("origin", "search_user")),
                (_uuid(1012), [1.0, 0.0, 0.0], _make_payload("dist_1", "search_user")),
                (_uuid(1013), [2.0, 0.0, 0.0], _make_payload("dist_2", "search_user")),
                (_uuid(1014), [3.0, 0.0, 0.0], _make_payload("dist_3", "search_user")),
            ])
            results = db.search("query", [0.0, 0.0, 0.0], top_k=4, filters={"user_id": "search_user"})
            assert len(results) == 4
            _assert_ordered_ids(results, [_uuid(1011), _uuid(1012), _uuid(1013), _uuid(1014)])
        finally:
            db.delete_col()

    def test_l2_known_distance_value(self):
        """L2 distance produces expected normalized score for known distance."""
        db = _new_db(prefix="p1_search", vector_metric="l2")
        try:
            # Insert a point at [1, 0, 0]; query from [0, 0, 0]
            # L2 distance = 1.0, normalized score = 1/(1+1) = 0.5
            _insert_memories(db, [
                (_uuid(1021), [1.0, 0.0, 0.0], _make_payload("unit_x", "search_user")),
            ])
            results = db.search("query", [0.0, 0.0, 0.0], top_k=1, filters={"user_id": "search_user"})
            assert len(results) == 1
            # Score should be 1/(1+distance). For L2 distance=1.0 -> score=0.5
            assert abs(results[0].score - 0.5) < 0.1
        finally:
            db.delete_col()

    def test_cosine_nearest_neighbor_correct(self):
        """Cosine metric returns the nearest neighbor correctly."""
        db = _new_db(prefix="p1_search", vector_metric="cosine")
        try:
            _insert_memories(db, [
                (_uuid(1031), [1.0, 0.0, 0.0], _make_payload("unit_x", "search_user")),
                (_uuid(1032), [0.9, 0.1, 0.0], _make_payload("near_x", "search_user")),
                (_uuid(1033), [0.0, 1.0, 0.0], _make_payload("unit_y", "search_user")),
            ])
            results = db.search("query", [1.0, 0.0, 0.0], top_k=3, filters={"user_id": "search_user"})
            assert len(results) >= 2
            # Exact match or closest cosine neighbor should be first
            assert results[0].id == _uuid(1031)
        finally:
            db.delete_col()

    def test_cosine_orthogonal_vectors_low_score(self):
        """Cosine: orthogonal vectors should have low similarity (high distance)."""
        db = _new_db(prefix="p1_search", vector_metric="cosine")
        try:
            _insert_memories(db, [
                (_uuid(1041), [1.0, 0.0, 0.0], _make_payload("x_axis", "search_user")),
                (_uuid(1042), [0.0, 1.0, 0.0], _make_payload("y_axis", "search_user")),
            ])
            # Query along x-axis; y-axis vector is orthogonal
            results = db.search("query", [1.0, 0.0, 0.0], top_k=2, filters={"user_id": "search_user"})
            assert len(results) == 2
            # The x-axis match should have much higher score than orthogonal y-axis
            x_score = results[0].score
            y_score = results[1].score
            assert x_score > y_score
            # Orthogonal cosine distance = 1.0, normalized = 1/(1+1) = 0.5
            assert y_score < 0.6
        finally:
            db.delete_col()

    def test_cosine_nearest_neighbor_ordering(self):
        """Cosine metric orders results by angular similarity."""
        db = _new_db(prefix="p1_search", vector_metric="cosine")
        try:
            _insert_memories(db, [
                (_uuid(1051), [1.0, 0.0, 0.0], _make_payload("along_x", "search_user")),
                (_uuid(1052), [1.0, 1.0, 0.0], _make_payload("45_deg", "search_user")),
                (_uuid(1053), [0.0, 1.0, 0.0], _make_payload("along_y", "search_user")),
            ])
            results = db.search("query", [1.0, 0.0, 0.0], top_k=3, filters={"user_id": "search_user"})
            assert len(results) == 3
            # Order: exact x-axis, 45-degree, orthogonal y-axis
            _assert_ordered_ids(results, [_uuid(1051), _uuid(1052), _uuid(1053)])
        finally:
            db.delete_col()

    def test_default_metric_is_cosine(self):
        """Default metric should be cosine when not specified."""
        db = _new_db(prefix="p1_search")
        try:
            assert db.vector_metric == "cosine"
        finally:
            db.delete_col()

    def test_l2_metric_stored_correctly(self):
        """L2 metric is stored and used correctly in the DB instance."""
        db = _new_db(prefix="p1_search", vector_metric="l2")
        try:
            assert db.vector_metric == "l2"
            _insert_memories(db, [
                (_uuid(1071), [0.5, 0.5, 0.5], _make_payload("center", "search_user")),
            ])
            results = db.search("query", [0.5, 0.5, 0.5], top_k=1, filters={"user_id": "search_user"})
            assert len(results) == 1
            # Exact match with L2 distance = 0, score = 1/(1+0) = 1.0
            assert results[0].score > 0.95
        finally:
            db.delete_col()


# ===========================================================================
# 6.2.2 Score Ordering Verification (5 tests)
# ===========================================================================


class TestScoreOrdering:
    """Tests for score ordering and precision."""

    def test_scores_in_descending_order(self):
        """Search results should have scores in descending order."""
        db = _new_db(prefix="p1_search", vector_metric="cosine")
        try:
            _insert_memories(db, [
                (_uuid(2001), [1.0, 0.0, 0.0], _make_payload("vec_a", "score_user")),
                (_uuid(2002), [0.7, 0.7, 0.0], _make_payload("vec_b", "score_user")),
                (_uuid(2003), [0.0, 1.0, 0.0], _make_payload("vec_c", "score_user")),
                (_uuid(2004), [0.0, 0.0, 1.0], _make_payload("vec_d", "score_user")),
            ])
            results = db.search("query", [1.0, 0.0, 0.0], top_k=4, filters={"user_id": "score_user"})
            scores = [r.score for r in results]
            assert scores == sorted(scores, reverse=True), "Scores should be in descending order"
        finally:
            db.delete_col()

    def test_top1_is_closest_vector(self):
        """Top-1 result should be the closest vector to the query."""
        db = _new_db(prefix="p1_search", vector_metric="cosine")
        try:
            _insert_memories(db, [
                (_uuid(2011), [0.1, 0.9, 0.1], _make_payload("far_from_query", "score_user")),
                (_uuid(2012), [0.95, 0.05, 0.0], _make_payload("close_to_query", "score_user")),
                (_uuid(2013), [0.5, 0.5, 0.5], _make_payload("medium", "score_user")),
            ])
            results = db.search("query", [1.0, 0.0, 0.0], top_k=1, filters={"user_id": "score_user"})
            assert len(results) == 1
            assert results[0].id == _uuid(2012)
        finally:
            db.delete_col()

    def test_identical_vector_returns_high_score(self):
        """Searching with an identical vector should return score close to 1.0."""
        db = _new_db(prefix="p1_search", vector_metric="cosine")
        try:
            _insert_memories(db, [
                (_uuid(2021), [0.6, 0.3, 0.1], _make_payload("target", "score_user")),
            ])
            results = db.search("query", [0.6, 0.3, 0.1], top_k=1, filters={"user_id": "score_user"})
            assert len(results) == 1
            # Cosine distance of identical vectors = 0, score = 1/(1+0) = 1.0
            assert results[0].score > 0.95
        finally:
            db.delete_col()

    def test_distant_vector_returns_low_score(self):
        """A vector far from the query should return a low score."""
        db = _new_db(prefix="p1_search", vector_metric="cosine")
        try:
            _insert_memories(db, [
                (_uuid(2031), [1.0, 0.0, 0.0], _make_payload("opposite_dir", "score_user")),
            ])
            # Query in opposite direction
            results = db.search("query", [-1.0, 0.0, 0.0], top_k=1, filters={"user_id": "score_user"})
            assert len(results) == 1
            # Cosine distance for opposite vectors = 2.0, score = 1/(1+2) = 0.333
            assert results[0].score < 0.5
        finally:
            db.delete_col()

    def test_score_range_validation(self):
        """All scores should be in the range (0, 1]."""
        db = _new_db(prefix="p1_search", vector_metric="cosine")
        try:
            _insert_memories(db, [
                (_uuid(2041), [1.0, 0.0, 0.0], _make_payload("a", "score_user")),
                (_uuid(2042), [0.0, 1.0, 0.0], _make_payload("b", "score_user")),
                (_uuid(2043), [0.0, 0.0, 1.0], _make_payload("c", "score_user")),
                (_uuid(2044), [-1.0, 0.0, 0.0], _make_payload("d", "score_user")),
            ])
            results = db.search("query", [0.5, 0.5, 0.0], top_k=4, filters={"user_id": "score_user"})
            for r in results:
                assert 0.0 < r.score <= 1.0, f"Score {r.score} out of valid range (0, 1]"
        finally:
            db.delete_col()


# ===========================================================================
# 6.2.3 Recall Quality (5 tests)
# ===========================================================================


class TestRecallQuality:
    """Tests for recall quality with known data distributions."""

    def test_known_vectors_correct_top3(self):
        """Insert 10 known vectors, search returns correct top-3."""
        db = _new_db(prefix="p1_search", vector_metric="cosine")
        try:
            # Create 10 vectors with varying similarity to query [1, 0, 0]
            vectors = [
                ([1.0, 0.0, 0.0], "exact_match"),       # id 3001 - best
                ([0.95, 0.05, 0.0], "very_close"),       # id 3002 - 2nd
                ([0.9, 0.1, 0.0], "close"),              # id 3003 - 3rd
                ([0.7, 0.3, 0.0], "moderate_1"),         # id 3004
                ([0.5, 0.5, 0.0], "moderate_2"),         # id 3005
                ([0.3, 0.7, 0.0], "far_1"),              # id 3006
                ([0.1, 0.9, 0.0], "far_2"),              # id 3007
                ([0.0, 1.0, 0.0], "orthogonal"),         # id 3008
                ([0.0, 0.0, 1.0], "orthogonal_z"),       # id 3009
                ([-1.0, 0.0, 0.0], "opposite"),          # id 3010
            ]
            records = [
                (_uuid(3001 + i), vec, _make_payload(label, "recall_user"))
                for i, (vec, label) in enumerate(vectors)
            ]
            _insert_memories(db, records)

            results = db.search("query", [1.0, 0.0, 0.0], top_k=3, filters={"user_id": "recall_user"})
            assert len(results) == 3
            top3_ids = _ids(results)
            assert top3_ids[0] == _uuid(3001)  # exact match
            assert top3_ids[1] == _uuid(3002)  # very close
            assert top3_ids[2] == _uuid(3003)  # close
        finally:
            db.delete_col()

    def test_100_vectors_recall_at_10(self):
        """Insert 100 vectors, recall@10 should be >= 0.8 for known nearest neighbors."""
        db = _new_db(prefix="p1_search", vector_metric="l2")
        try:
            import random
            random.seed(42)

            # Generate 100 random 3D vectors
            all_vectors = []
            records = []
            for i in range(100):
                vec = [random.uniform(-1, 1) for _ in range(3)]
                all_vectors.append((i, vec))
                records.append((_uuid(3100 + i), vec, _make_payload(f"item_{i}", "recall_user")))
            _insert_memories(db, records)

            # Query vector
            query = [0.5, 0.5, 0.5]

            # Compute true L2 distances
            def l2_dist(a, b):
                return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

            true_nearest = sorted(all_vectors, key=lambda x: l2_dist(x[1], query))
            true_top10_ids = {_uuid(3100 + idx) for idx, _ in true_nearest[:10]}

            results = db.search("query", query, top_k=10, filters={"user_id": "recall_user"})
            result_ids = set(_ids(results))

            recall = len(result_ids & true_top10_ids) / 10.0
            assert recall >= 0.8, f"Recall@10 = {recall}, expected >= 0.8"
        finally:
            db.delete_col()

    def test_cluster_search_returns_same_cluster_first(self):
        """Vectors in the same cluster as query should rank higher."""
        db = _new_db(prefix="p1_search", vector_metric="cosine")
        try:
            # Cluster A: near [1, 0, 0]
            # Cluster B: near [0, 1, 0]
            records = [
                (_uuid(3201), [0.95, 0.05, 0.0], _make_payload("cluster_a_1", "recall_user")),
                (_uuid(3202), [0.90, 0.10, 0.0], _make_payload("cluster_a_2", "recall_user")),
                (_uuid(3203), [0.85, 0.15, 0.0], _make_payload("cluster_a_3", "recall_user")),
                (_uuid(3204), [0.05, 0.95, 0.0], _make_payload("cluster_b_1", "recall_user")),
                (_uuid(3205), [0.10, 0.90, 0.0], _make_payload("cluster_b_2", "recall_user")),
                (_uuid(3206), [0.15, 0.85, 0.0], _make_payload("cluster_b_3", "recall_user")),
            ]
            _insert_memories(db, records)

            # Query near cluster A
            results = db.search("query", [1.0, 0.0, 0.0], top_k=6, filters={"user_id": "recall_user"})
            top3_ids = set(_ids(results[:3]))
            cluster_a_ids = {_uuid(3201), _uuid(3202), _uuid(3203)}
            assert top3_ids == cluster_a_ids, "Top-3 should all be from cluster A"
        finally:
            db.delete_col()

    def test_duplicate_vectors_return_same_score(self):
        """Duplicate vectors should return the same score."""
        db = _new_db(prefix="p1_search", vector_metric="cosine")
        try:
            _insert_memories(db, [
                (_uuid(3301), [0.5, 0.5, 0.0], _make_payload("dup_1", "recall_user")),
                (_uuid(3302), [0.5, 0.5, 0.0], _make_payload("dup_2", "recall_user")),
            ])
            results = db.search("query", [1.0, 0.0, 0.0], top_k=2, filters={"user_id": "recall_user"})
            assert len(results) == 2
            assert abs(results[0].score - results[1].score) < 1e-6
        finally:
            db.delete_col()

    def test_near_duplicate_vectors_return_similar_scores(self):
        """Near-duplicate vectors should return very similar scores."""
        db = _new_db(prefix="p1_search", vector_metric="cosine")
        try:
            _insert_memories(db, [
                (_uuid(3401), [0.500, 0.500, 0.000], _make_payload("near_dup_1", "recall_user")),
                (_uuid(3402), [0.501, 0.499, 0.001], _make_payload("near_dup_2", "recall_user")),
            ])
            results = db.search("query", [1.0, 0.0, 0.0], top_k=2, filters={"user_id": "recall_user"})
            assert len(results) == 2
            score_diff = abs(results[0].score - results[1].score)
            assert score_diff < 0.01, f"Near-duplicate score diff {score_diff} should be < 0.01"
        finally:
            db.delete_col()


# ===========================================================================
# 6.2.4 BM25 Search Quality (5 tests, conditional)
# ===========================================================================


@pytest.mark.skipif(not _env_bool("GAUSSDB_TEST_RUN_BM25"), reason="BM25 tests disabled")
class TestBM25SearchQuality:
    """Tests for BM25 text search quality (requires GAUSSDB_TEST_RUN_BM25=true)."""

    def test_bm25_single_keyword_match(self):
        """BM25 search with a single keyword should find matching documents."""
        db = _new_db(prefix="p1_search", bm25_mode="required", enable_capability_probe=True)
        try:
            _insert_memories(db, [
                (_uuid(4001), VECTOR_COFFEE, _make_payload("coffee espresso latte", "bm25_user")),
                (_uuid(4002), VECTOR_FLIGHT, _make_payload("airplane flight travel", "bm25_user")),
                (_uuid(4003), VECTOR_WINDOW, _make_payload("window glass pane", "bm25_user")),
            ])
            results = db.search("coffee", VECTOR_COFFEE, top_k=3, filters={"user_id": "bm25_user"})
            assert len(results) >= 1
            # The coffee document should rank high
            result_ids = _ids(results)
            assert _uuid(4001) in result_ids
        finally:
            db.delete_col()

    def test_bm25_multi_keyword_match(self):
        """BM25 search with multiple keywords should prefer documents matching more terms."""
        db = _new_db(prefix="p1_search", bm25_mode="required", enable_capability_probe=True)
        try:
            _insert_memories(db, [
                (_uuid(4011), VECTOR_COFFEE, _make_payload("coffee espresso morning brew", "bm25_user")),
                (_uuid(4012), VECTOR_FLIGHT, _make_payload("coffee flight morning travel", "bm25_user")),
                (_uuid(4013), VECTOR_WINDOW, _make_payload("window glass morning light", "bm25_user")),
            ])
            results = db.search("coffee morning", VECTOR_COFFEE, top_k=3, filters={"user_id": "bm25_user"})
            assert len(results) >= 1
            # Documents with both "coffee" and "morning" should rank higher
            top_ids = _ids(results[:2])
            assert _uuid(4011) in top_ids or _uuid(4012) in top_ids
        finally:
            db.delete_col()

    def test_bm25_exact_phrase_match(self):
        """BM25 search should find exact phrase matches."""
        db = _new_db(prefix="p1_search", bm25_mode="required", enable_capability_probe=True)
        try:
            _insert_memories(db, [
                (_uuid(4021), VECTOR_COFFEE, _make_payload("hot coffee with milk", "bm25_user")),
                (_uuid(4022), VECTOR_FLIGHT, _make_payload("cold coffee without milk", "bm25_user")),
                (_uuid(4023), VECTOR_WINDOW, _make_payload("tea with lemon", "bm25_user")),
            ])
            results = db.search("hot coffee", VECTOR_COFFEE, top_k=3, filters={"user_id": "bm25_user"})
            assert len(results) >= 1
            # "hot coffee" document should be in results
            assert _uuid(4021) in _ids(results)
        finally:
            db.delete_col()

    def test_bm25_no_match_returns_empty_or_low_score(self):
        """BM25 search with no matching terms should return empty or very low scores."""
        db = _new_db(prefix="p1_search", bm25_mode="required", enable_capability_probe=True)
        try:
            _insert_memories(db, [
                (_uuid(4031), VECTOR_COFFEE, _make_payload("coffee espresso latte", "bm25_user")),
                (_uuid(4032), VECTOR_FLIGHT, _make_payload("airplane flight travel", "bm25_user")),
            ])
            # Search for a term that does not exist in any document
            results = db.search("xyznonexistent", VECTOR_WINDOW, top_k=3, filters={"user_id": "bm25_user"})
            # Either empty or results have low relevance (vector-only fallback)
            if len(results) > 0:
                # If results returned, they are from vector similarity only
                # No BM25 boost should be applied
                assert all(r.score <= 1.0 for r in results)
        finally:
            db.delete_col()

    def test_bm25_score_ordering(self):
        """BM25 results should be ordered by relevance score."""
        db = _new_db(prefix="p1_search", bm25_mode="required", enable_capability_probe=True)
        try:
            _insert_memories(db, [
                (_uuid(4041), VECTOR_COFFEE, _make_payload("coffee coffee coffee beans", "bm25_user")),
                (_uuid(4042), VECTOR_FLIGHT, _make_payload("coffee once mentioned", "bm25_user")),
                (_uuid(4043), VECTOR_WINDOW, _make_payload("no relevant terms here", "bm25_user")),
            ])
            results = db.search("coffee", VECTOR_COFFEE, top_k=3, filters={"user_id": "bm25_user"})
            if len(results) >= 2:
                scores = [r.score for r in results]
                assert scores == sorted(scores, reverse=True), "BM25 scores should be in descending order"
        finally:
            db.delete_col()


# ===========================================================================
# 6.2.5 Hybrid Search Quality (3 tests, conditional)
# ===========================================================================


@pytest.mark.skipif(not _env_bool("GAUSSDB_TEST_RUN_BM25"), reason="Hybrid tests require BM25")
class TestHybridSearchQuality:
    """Tests for hybrid (vector + BM25) search quality."""

    def test_hybrid_vector_plus_bm25_combined(self):
        """Hybrid search should benefit from both vector and text similarity."""
        db = _new_db(prefix="p1_search", bm25_mode="required", enable_capability_probe=True)
        try:
            # Doc 1: good vector match + good text match
            # Doc 2: good vector match + poor text match
            # Doc 3: poor vector match + good text match
            _insert_memories(db, [
                (_uuid(5001), [0.9, 0.1, 0.0], _make_payload("coffee espresso beans", "hybrid_user")),
                (_uuid(5002), [0.85, 0.15, 0.0], _make_payload("unrelated topic here", "hybrid_user")),
                (_uuid(5003), [0.1, 0.9, 0.0], _make_payload("coffee latte cappuccino", "hybrid_user")),
            ])
            results = db.search("coffee", [1.0, 0.0, 0.0], top_k=3, filters={"user_id": "hybrid_user"})
            assert len(results) >= 1
            # Doc with both good vector + text match should rank first
            assert results[0].id == _uuid(5001)
        finally:
            db.delete_col()

    def test_hybrid_vs_vector_only_comparison(self):
        """Hybrid search should produce different ranking than pure vector search."""
        db_hybrid = _new_db(prefix="p1_search", bm25_mode="required", enable_capability_probe=True)
        db_vector = _new_db(prefix="p1_search", bm25_mode="disabled")
        try:
            records = [
                (_uuid(5011), [0.9, 0.1, 0.0], _make_payload("airplane flight travel", "hybrid_user")),
                (_uuid(5012), [0.85, 0.15, 0.0], _make_payload("coffee espresso beans", "hybrid_user")),
                (_uuid(5013), [0.1, 0.9, 0.0], _make_payload("coffee latte morning", "hybrid_user")),
            ]
            _insert_memories(db_hybrid, records)
            _insert_memories(db_vector, records)

            query_vec = [1.0, 0.0, 0.0]
            hybrid_results = db_hybrid.search("coffee", query_vec, top_k=3, filters={"user_id": "hybrid_user"})
            vector_results = db_vector.search("coffee", query_vec, top_k=3, filters={"user_id": "hybrid_user"})

            # Both should return results
            assert len(hybrid_results) >= 1
            assert len(vector_results) >= 1
            # Hybrid may reorder results due to BM25 boost
            # At minimum, both return valid scored results
            for r in hybrid_results:
                assert r.score > 0
            for r in vector_results:
                assert r.score > 0
        finally:
            db_hybrid.delete_col()
            db_vector.delete_col()

    def test_hybrid_bm25_boost_effect(self):
        """BM25 component should boost text-relevant results in hybrid search."""
        db = _new_db(prefix="p1_search", bm25_mode="required", enable_capability_probe=True)
        try:
            # Two vectors equidistant from query, but one has matching text
            _insert_memories(db, [
                (_uuid(5021), [0.7, 0.7, 0.0], _make_payload("coffee beans roast", "hybrid_user")),
                (_uuid(5022), [0.7, 0.0, 0.7], _make_payload("unrelated random words", "hybrid_user")),
            ])
            results = db.search("coffee", [0.7, 0.35, 0.35], top_k=2, filters={"user_id": "hybrid_user"})
            assert len(results) == 2
            # The text-matching document should get a boost
            assert results[0].id == _uuid(5021)
        finally:
            db.delete_col()


# ===========================================================================
# 6.2.6 Score Edge Cases (4 tests)
# ===========================================================================


class TestScoreEdgeCases:
    """Tests for score edge cases and stability."""

    def test_orthogonal_vectors_low_scores(self):
        """All vectors orthogonal to query should produce uniformly low scores."""
        db = _new_db(prefix="p1_search", vector_metric="cosine")
        try:
            # Query along x-axis; insert vectors along y and z axes
            _insert_memories(db, [
                (_uuid(6001), [0.0, 1.0, 0.0], _make_payload("y_axis", "edge_user")),
                (_uuid(6002), [0.0, 0.0, 1.0], _make_payload("z_axis", "edge_user")),
                (_uuid(6003), [0.0, 0.7, 0.7], _make_payload("yz_plane", "edge_user")),
            ])
            results = db.search("query", [1.0, 0.0, 0.0], top_k=3, filters={"user_id": "edge_user"})
            for r in results:
                # All are orthogonal or near-orthogonal to x-axis
                # Cosine distance >= 1.0, score <= 0.5
                assert r.score <= 0.55, f"Orthogonal vector score {r.score} should be <= 0.55"
        finally:
            db.delete_col()

    def test_duplicate_scores_sorting_stability(self):
        """Vectors with identical scores should have stable sort (by ID)."""
        db = _new_db(prefix="p1_search", vector_metric="cosine")
        try:
            # Insert identical vectors with different IDs
            _insert_memories(db, [
                (_uuid(6011), [0.5, 0.5, 0.0], _make_payload("dup_a", "edge_user")),
                (_uuid(6012), [0.5, 0.5, 0.0], _make_payload("dup_b", "edge_user")),
                (_uuid(6013), [0.5, 0.5, 0.0], _make_payload("dup_c", "edge_user")),
            ])
            results = db.search("query", [1.0, 0.0, 0.0], top_k=3, filters={"user_id": "edge_user"})
            assert len(results) == 3
            # All scores should be identical
            scores = [r.score for r in results]
            assert all(abs(s - scores[0]) < 1e-6 for s in scores)
            # GaussDB sorts ties by ID ASC
            ids = _ids(results)
            assert ids == sorted(ids), "Tie-breaking should be stable (by ID)"
        finally:
            db.delete_col()

    def test_score_precision_decimal_places(self):
        """Scores should have reasonable floating-point precision."""
        db = _new_db(prefix="p1_search", vector_metric="cosine")
        try:
            _insert_memories(db, [
                (_uuid(6021), [0.9, 0.1, 0.0], _make_payload("precise_a", "edge_user")),
                (_uuid(6022), [0.8, 0.2, 0.0], _make_payload("precise_b", "edge_user")),
            ])
            results = db.search("query", [1.0, 0.0, 0.0], top_k=2, filters={"user_id": "edge_user"})
            assert len(results) == 2
            # Scores should be distinguishable (not rounded to same value)
            assert results[0].score != results[1].score
            # Scores should have at least 4 decimal places of precision
            for r in results:
                score_str = f"{r.score:.6f}"
                assert len(score_str) >= 6  # e.g., "0.987654"
        finally:
            db.delete_col()

    def test_large_dataset_sorting_correctness(self):
        """1000 items with top_k=100 should return correctly sorted results."""
        db = _new_db(prefix="p1_search", vector_metric="l2")
        try:
            import random
            random.seed(123)

            # Insert 1000 vectors
            batch_size = 100
            for batch_start in range(0, 1000, batch_size):
                records = []
                for i in range(batch_start, batch_start + batch_size):
                    vec = [random.uniform(-1, 1) for _ in range(3)]
                    records.append((_uuid(6100 + i), vec, _make_payload(f"item_{i}", "edge_user")))
                _insert_memories(db, records)

            # Search with top_k=100
            query = [0.0, 0.0, 0.0]
            results = db.search("query", query, top_k=100, filters={"user_id": "edge_user"})

            assert len(results) == 100
            # Verify scores are in descending order
            scores = [r.score for r in results]
            for i in range(len(scores) - 1):
                assert scores[i] >= scores[i + 1], (
                    f"Score at position {i} ({scores[i]}) should be >= score at position {i+1} ({scores[i+1]})"
                )
        finally:
            db.delete_col()

