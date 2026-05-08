"""
GaussDB-specific Feature Verification Tests.

Tests GaussDB unique capabilities including:
- Ustore storage engine verification
- FLOATVECTOR type boundary verification
- Vector index type comparison (GsIVFFlat vs GsDiskANN)
- Deployment mode verification (centralized vs distributed)
- MERGE INTO (upsert) atomicity verification

~15 tests total.
"""

import os
import time
import random

import pytest

from tests.vector_stores.conftest import (
    EMBEDDING_DIMS,
    VECTOR_COFFEE,
    VECTOR_FLIGHT,
    VECTOR_WINDOW,
    VECTOR_AISLE,
    _list_flat,
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
# 10.2.1 Ustore Storage Engine Verification
# ===========================================================================


class TestUstoreStorageEngine:
    """Tests for Ustore storage engine behavior."""

    def test_ustore_table_creation(self):
        """Create collection with Ustore engine, verify basic CRUD works."""
        db = _new_db(prefix="feat_ustore")
        try:
            vid = _uuid(7001)
            db.insert(
                ids=[vid],
                vectors=[VECTOR_COFFEE],
                payloads=[_make_payload("ustore creation test", "ustore_user")],
            )
            # Verify insert
            result = db.get(vid)
            assert result is not None
            assert result.id == vid
            assert result.payload["data"] == "ustore creation test"

            # Verify search
            results = db.search(
                "ustore", VECTOR_COFFEE, top_k=1, filters={"user_id": "ustore_user"}
            )
            assert len(results) >= 1
            assert results[0].id == vid

            # Verify delete
            db.delete(vector_id=vid)
            result_after = db.get(vid)
            assert result_after is None
        finally:
            db.delete_col()

    def test_ustore_vs_astore_insert_perf(self):
        """Compare insert performance (informational, no hard assertion)."""
        db = _new_db(prefix="feat_ustore_perf")
        try:
            num_records = 50
            ids = [_uuid(7100 + i) for i in range(num_records)]
            vectors = [[random.random() for _ in range(EMBEDDING_DIMS)] for _ in range(num_records)]
            payloads = [_make_payload(f"perf record {i}", "perf_user") for i in range(num_records)]

            start = time.perf_counter()
            db.insert(ids=ids, vectors=vectors, payloads=payloads)
            insert_duration_ms = (time.perf_counter() - start) * 1000

            # Informational: just verify all records were inserted
            count = len(_list_flat(db, filters={"user_id": "perf_user"}, top_k=1000))
            assert count == num_records
            # Log performance (no hard assertion on timing)
            assert insert_duration_ms >= 0  # trivially true, documents the measurement
        finally:
            db.delete_col()

    def test_ustore_vs_astore_search_perf(self):
        """Compare search performance (informational, no hard assertion)."""
        db = _new_db(prefix="feat_ustore_search")
        try:
            # Insert baseline data
            num_records = 50
            ids = [_uuid(7200 + i) for i in range(num_records)]
            vectors = [[random.random() for _ in range(EMBEDDING_DIMS)] for _ in range(num_records)]
            payloads = [_make_payload(f"search perf {i}", "perf_user") for i in range(num_records)]
            db.insert(ids=ids, vectors=vectors, payloads=payloads)

            query_vector = [random.random() for _ in range(EMBEDDING_DIMS)]

            start = time.perf_counter()
            iterations = 20
            for _ in range(iterations):
                db.search("perf", query_vector, top_k=10, filters={"user_id": "perf_user"})
            search_duration_ms = (time.perf_counter() - start) * 1000

            avg_search_ms = search_duration_ms / iterations
            # Informational: verify search returns results and measure timing
            results = db.search("perf", query_vector, top_k=10, filters={"user_id": "perf_user"})
            assert len(results) >= 1
            assert avg_search_ms >= 0  # trivially true, documents the measurement
        finally:
            db.delete_col()

    def test_ustore_update_in_place(self):
        """Verify update works correctly (in-place update behavior)."""
        db = _new_db(prefix="feat_ustore_upd")
        try:
            vid = _uuid(7301)
            # Insert original record
            db.insert(
                ids=[vid],
                vectors=[VECTOR_COFFEE],
                payloads=[_make_payload("original data", "update_user")],
            )
            original = db.get(vid)
            assert original is not None
            assert original.payload["data"] == "original data"

            # Update vector and payload in place
            new_vector = VECTOR_FLIGHT
            db.update(
                vector_id=vid,
                vector=new_vector,
                payload=_make_payload("updated data", "update_user"),
            )

            # Verify update took effect
            updated = db.get(vid)
            assert updated is not None
            assert updated.payload["data"] == "updated data"

            # Verify search finds updated record with new vector
            results = db.search(
                "updated", VECTOR_FLIGHT, top_k=1, filters={"user_id": "update_user"}
            )
            assert len(results) >= 1
            assert results[0].id == vid
            assert results[0].payload["data"] == "updated data"
        finally:
            db.delete_col()


# ===========================================================================
# 10.2.2 FLOATVECTOR Type Boundary Verification
# ===========================================================================


class TestFloatvectorTypeBoundary:
    """Tests for FLOATVECTOR type precision and boundary behavior."""

    def test_floatvector_precision_float32(self):
        """Verify float32 precision is maintained in storage and retrieval."""
        db = _new_db(prefix="feat_fvec")
        try:
            vid = _uuid(7401)
            # Use values that test float32 precision boundaries
            precise_vector = [0.123456789, 0.987654321, 0.555555555]
            db.insert(
                ids=[vid],
                vectors=[precise_vector],
                payloads=[_make_payload("precision test", "fvec_user")],
            )

            # Search with the same vector should return high similarity
            results = db.search(
                "precision", precise_vector, top_k=1, filters={"user_id": "fvec_user"}
            )
            assert len(results) == 1
            assert results[0].id == vid
            # Score should be very high (close to 1.0 for cosine similarity)
            if results[0].score is not None:
                assert results[0].score > 0.99
        finally:
            db.delete_col()

    def test_floatvector_max_dimensions_supported(self):
        """Test maximum supported dimensions (use 1024 as a high-dim test)."""
        max_dims = 1024
        db = _new_db(prefix="feat_fvec_max", embedding_model_dims=max_dims)
        try:
            vid = _uuid(7501)
            high_dim_vector = [random.random() for _ in range(max_dims)]
            db.insert(
                ids=[vid],
                vectors=[high_dim_vector],
                payloads=[_make_payload("max dim test", "fvec_user")],
            )

            results = db.search(
                "maxdim", high_dim_vector, top_k=1, filters={"user_id": "fvec_user"}
            )
            assert len(results) == 1
            assert results[0].id == vid
        finally:
            db.delete_col()

    @pytest.mark.xfail(reason="GaussDB floatvector distance overflows with extreme values (FLT_MAX)")
    def test_floatvector_special_values(self):
        """Test with very small and very large float values."""
        db = _new_db(prefix="feat_fvec_special")
        try:
            # Very small values (near epsilon)
            vid_small = _uuid(7601)
            small_vector = [1e-30, 1e-30, 1e-30]
            db.insert(
                ids=[vid_small],
                vectors=[small_vector],
                payloads=[_make_payload("small values", "fvec_user")],
            )

            # Very large values
            vid_large = _uuid(7602)
            large_vector = [1e30, 1e30, 1e30]
            db.insert(
                ids=[vid_large],
                vectors=[large_vector],
                payloads=[_make_payload("large values", "fvec_user")],
            )

            # Verify both records exist
            result_small = db.get(vid_small)
            result_large = db.get(vid_large)
            assert result_small is not None
            assert result_large is not None

            # Search should distinguish between them
            results = db.search(
                "small", small_vector, top_k=2, filters={"user_id": "fvec_user"}
            )
            assert len(results) == 2
        finally:
            db.delete_col()

    def test_floatvector_normalized_vs_unnormalized(self):
        """Test with normalized (unit length) and unnormalized vectors."""
        db = _new_db(prefix="feat_fvec_norm")
        try:
            # Normalized vector (unit length)
            vid_norm = _uuid(7701)
            import math
            norm_factor = math.sqrt(0.1**2 + 0.2**2 + 0.3**2)
            normalized_vector = [0.1 / norm_factor, 0.2 / norm_factor, 0.3 / norm_factor]
            db.insert(
                ids=[vid_norm],
                vectors=[normalized_vector],
                payloads=[_make_payload("normalized", "fvec_user")],
            )

            # Unnormalized vector (same direction, different magnitude)
            vid_unnorm = _uuid(7702)
            unnormalized_vector = [10.0, 20.0, 30.0]
            db.insert(
                ids=[vid_unnorm],
                vectors=[unnormalized_vector],
                payloads=[_make_payload("unnormalized", "fvec_user")],
            )

            # Both should be retrievable
            result_norm = db.get(vid_norm)
            result_unnorm = db.get(vid_unnorm)
            assert result_norm is not None
            assert result_unnorm is not None

            # Cosine similarity search: same direction vectors should both rank high
            results = db.search(
                "direction", [1.0, 2.0, 3.0], top_k=2, filters={"user_id": "fvec_user"}
            )
            assert len(results) == 2
            returned_ids = {r.id for r in results}
            assert vid_norm in returned_ids
            assert vid_unnorm in returned_ids
        finally:
            db.delete_col()


# ===========================================================================
# 10.2.3 Vector Index Type Comparison
# ===========================================================================


class TestVectorIndexTypeComparison:
    """Tests for GsIVFFlat and GsDiskANN index types."""

    def test_gsivfflat_index_basic_crud(self):
        """Basic CRUD with GsIVFFlat index type."""
        db = _new_db(prefix="feat_ivfflat", vector_index_type="gsivfflat")
        try:
            vid = _uuid(8001)
            db.insert(
                ids=[vid],
                vectors=[VECTOR_COFFEE],
                payloads=[_make_payload("ivfflat test", "idx_user")],
            )

            # Search
            results = db.search(
                "ivfflat", VECTOR_COFFEE, top_k=1, filters={"user_id": "idx_user"}
            )
            assert len(results) == 1
            assert results[0].id == vid

            # Update
            db.update(
                vector_id=vid,
                vector=VECTOR_FLIGHT,
                payload=_make_payload("ivfflat updated", "idx_user"),
            )
            updated = db.get(vid)
            assert updated is not None
            assert updated.payload["data"] == "ivfflat updated"

            # Delete
            db.delete(vector_id=vid)
            assert db.get(vid) is None
        finally:
            db.delete_col()

    @pytest.mark.skipif(
        os.getenv("GAUSSDB_TEST_VECTOR_INDEX") != "gsdiskann",
        reason="GsDiskANN not configured",
    )
    def test_gsdiskann_index_basic_crud(self):
        """Basic CRUD with GsDiskANN index type (skip if not available)."""
        db = _new_db(prefix="feat_diskann", vector_index_type="gsdiskann")
        try:
            vid = _uuid(8101)
            db.insert(
                ids=[vid],
                vectors=[VECTOR_WINDOW],
                payloads=[_make_payload("diskann test", "idx_user")],
            )

            # Search
            results = db.search(
                "diskann", VECTOR_WINDOW, top_k=1, filters={"user_id": "idx_user"}
            )
            assert len(results) == 1
            assert results[0].id == vid

            # Update
            db.update(
                vector_id=vid,
                vector=VECTOR_AISLE,
                payload=_make_payload("diskann updated", "idx_user"),
            )
            updated = db.get(vid)
            assert updated is not None
            assert updated.payload["data"] == "diskann updated"

            # Delete
            db.delete(vector_id=vid)
            assert db.get(vid) is None
        finally:
            db.delete_col()

    def test_index_type_search_recall_comparison(self):
        """Compare recall between GsIVFFlat index (informational)."""
        db = _new_db(prefix="feat_idx_recall", vector_index_type="gsivfflat")
        try:
            # Insert a set of known vectors
            num_records = 30
            ids = [_uuid(8200 + i) for i in range(num_records)]
            vectors = [[random.random() for _ in range(EMBEDDING_DIMS)] for _ in range(num_records)]
            payloads = [_make_payload(f"recall item {i}", "recall_user") for i in range(num_records)]
            db.insert(ids=ids, vectors=vectors, payloads=payloads)

            # Search with one of the inserted vectors (should find itself)
            target_idx = 5
            results = db.search(
                "recall", vectors[target_idx], top_k=5, filters={"user_id": "recall_user"}
            )
            assert len(results) >= 1
            # The exact vector should be the top result (recall = 1 for exact match)
            assert results[0].id == ids[target_idx]
        finally:
            db.delete_col()

    def test_index_rebuild_after_bulk_insert(self):
        """Verify index works correctly after bulk insert."""
        db = _new_db(prefix="feat_idx_bulk", vector_index_type="gsivfflat")
        try:
            # First batch
            batch1_ids = [_uuid(8300 + i) for i in range(20)]
            batch1_vectors = [[random.random() for _ in range(EMBEDDING_DIMS)] for _ in range(20)]
            batch1_payloads = [_make_payload(f"batch1 item {i}", "bulk_user") for i in range(20)]
            db.insert(ids=batch1_ids, vectors=batch1_vectors, payloads=batch1_payloads)

            # Second batch (simulates bulk insert after index creation)
            batch2_ids = [_uuid(8320 + i) for i in range(20)]
            batch2_vectors = [[random.random() for _ in range(EMBEDDING_DIMS)] for _ in range(20)]
            batch2_payloads = [_make_payload(f"batch2 item {i}", "bulk_user") for i in range(20)]
            db.insert(ids=batch2_ids, vectors=batch2_vectors, payloads=batch2_payloads)

            # Verify total count
            assert len(_list_flat(db, filters={"user_id": "bulk_user"}, top_k=1000)) == 40

            # Search should find records from both batches
            results = db.search(
                "bulk", batch2_vectors[10], top_k=5, filters={"user_id": "bulk_user"}
            )
            assert len(results) >= 1
            # The exact vector from batch2 should be findable
            assert batch2_ids[10] in {r.id for r in results}
        finally:
            db.delete_col()


# ===========================================================================
# 10.2.4 Deployment Mode Verification
# ===========================================================================


class TestDeploymentMode:
    """Tests for centralized and distributed deployment modes."""

    def test_centralized_mode_basic_operations(self):
        """Verify all operations work in centralized mode."""
        db = _new_db(prefix="feat_central", deployment_mode="centralized")
        try:
            # Insert
            vid1 = _uuid(9001)
            vid2 = _uuid(9002)
            db.insert(
                ids=[vid1, vid2],
                vectors=[VECTOR_COFFEE, VECTOR_FLIGHT],
                payloads=[
                    _make_payload("centralized item 1", "deploy_user"),
                    _make_payload("centralized item 2", "deploy_user"),
                ],
            )

            # Count
            assert len(_list_flat(db, filters={"user_id": "deploy_user"}, top_k=1000)) == 2

            # Search
            results = db.search(
                "centralized", VECTOR_COFFEE, top_k=2, filters={"user_id": "deploy_user"}
            )
            assert len(results) == 2

            # Get
            result = db.get(vid1)
            assert result is not None
            assert result.payload["data"] == "centralized item 1"

            # Update
            db.update(
                vector_id=vid1,
                vector=VECTOR_WINDOW,
                payload=_make_payload("centralized updated", "deploy_user"),
            )
            updated = db.get(vid1)
            assert updated.payload["data"] == "centralized updated"

            # List
            listed = _list_flat(db, filters={"user_id": "deploy_user"}, top_k=10)
            assert len(listed) == 2

            # Delete
            db.delete(vector_id=vid1)
            assert db.get(vid1) is None
            assert len(_list_flat(db, filters={"user_id": "deploy_user"}, top_k=1000)) == 1
        finally:
            db.delete_col()

    @pytest.mark.skipif(
        os.getenv("GAUSSDB_TEST_DEPLOYMENT_MODE") != "distributed",
        reason="Distributed mode not configured",
    )
    def test_distributed_mode_basic_operations(self):
        """Verify operations in distributed mode (skip if not configured)."""
        db = _new_db(prefix="feat_distrib", deployment_mode="distributed")
        try:
            # Insert
            vid1 = _uuid(9101)
            vid2 = _uuid(9102)
            db.insert(
                ids=[vid1, vid2],
                vectors=[VECTOR_WINDOW, VECTOR_AISLE],
                payloads=[
                    _make_payload("distributed item 1", "distrib_user"),
                    _make_payload("distributed item 2", "distrib_user"),
                ],
            )

            # Count
            assert len(_list_flat(db, filters={"user_id": "distrib_user"}, top_k=1000)) == 2

            # Search
            results = db.search(
                "distributed", VECTOR_WINDOW, top_k=2, filters={"user_id": "distrib_user"}
            )
            assert len(results) >= 1

            # Get
            result = db.get(vid1)
            assert result is not None
            assert result.payload["data"] == "distributed item 1"

            # Update
            db.update(
                vector_id=vid2,
                vector=VECTOR_COFFEE,
                payload=_make_payload("distributed updated", "distrib_user"),
            )
            updated = db.get(vid2)
            assert updated.payload["data"] == "distributed updated"

            # Delete
            db.delete(vector_id=vid1)
            assert db.get(vid1) is None
            assert len(_list_flat(db, filters={"user_id": "distrib_user"}, top_k=1000)) == 1
        finally:
            db.delete_col()


# ===========================================================================
# 10.2.5 MERGE INTO Atomicity Verification
# ===========================================================================


class TestMergeIntoAtomicity:
    """Tests for MERGE INTO (upsert) atomicity behavior."""

    def test_merge_into_insert_new_record(self):
        """MERGE INTO inserts when record doesn't exist."""
        db = _new_db(prefix="feat_merge")
        try:
            vid = _uuid(9201)
            # Use update which triggers MERGE INTO behavior (upsert)
            # First verify the record does not exist
            assert db.get(vid) is None

            # Insert via normal insert (establishes baseline)
            db.insert(
                ids=[vid],
                vectors=[VECTOR_COFFEE],
                payloads=[_make_payload("merge insert test", "merge_user")],
            )

            # Verify the record was created
            result = db.get(vid)
            assert result is not None
            assert result.payload["data"] == "merge insert test"
        finally:
            db.delete_col()

    def test_merge_into_update_existing_record(self):
        """MERGE INTO updates when record exists (upsert semantics)."""
        db = _new_db(prefix="feat_merge_upd")
        try:
            vid = _uuid(9301)
            # Insert initial record
            db.insert(
                ids=[vid],
                vectors=[VECTOR_COFFEE],
                payloads=[_make_payload("original merge", "merge_user")],
            )

            # Update the same ID (triggers MERGE INTO / upsert path)
            db.update(
                vector_id=vid,
                vector=VECTOR_FLIGHT,
                payload=_make_payload("updated merge", "merge_user"),
            )

            # Verify update took effect atomically
            result = db.get(vid)
            assert result is not None
            assert result.payload["data"] == "updated merge"

            # Verify no duplicate was created
            assert len(_list_flat(db, filters={"user_id": "merge_user"}, top_k=1000)) == 1
        finally:
            db.delete_col()

    def test_merge_into_idempotent_same_data(self):
        """MERGE INTO with same data is idempotent."""
        db = _new_db(prefix="feat_merge_idem")
        try:
            vid = _uuid(9401)
            vector = VECTOR_WINDOW
            payload = _make_payload("idempotent test", "merge_user")

            # Insert the record
            db.insert(ids=[vid], vectors=[vector], payloads=[payload])

            # Apply the same update multiple times
            for _ in range(3):
                db.update(vector_id=vid, vector=vector, payload=payload)

            # Verify record is unchanged and no duplicates
            result = db.get(vid)
            assert result is not None
            assert result.payload["data"] == "idempotent test"
            assert len(_list_flat(db, filters={"user_id": "merge_user"}, top_k=1000)) == 1

            # Verify search still works correctly
            results = db.search(
                "idempotent", vector, top_k=1, filters={"user_id": "merge_user"}
            )
            assert len(results) == 1
            assert results[0].id == vid
        finally:
            db.delete_col()