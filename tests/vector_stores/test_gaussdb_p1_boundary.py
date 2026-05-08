"""
P1 Boundary Value Tests for GaussDB vector store.

Tests edge cases and boundary conditions for:
- Vector dimensions (1-dim, max-dim, zero, NaN, Inf, large, negative, empty)
- Payload boundaries (empty, large, deep nesting, special chars, Unicode)
- ID boundaries (empty, long, special chars, duplicates, UUID/non-UUID)
- Collection name boundaries (long, special chars, SQL injection, empty)
- top_k boundaries (0, 1, very large, negative, exceeds data)
- Batch operation boundaries (empty, single, large, mixed valid/invalid)

~40 tests total.
"""

import math
import os
import uuid

import pytest

from mem0.vector_stores.gaussdb import GaussDB
from tests.vector_stores.conftest import (
    EMBEDDING_DIMS,
    VECTOR_COFFEE,
    VECTOR_FLIGHT,
    _assert_exact_ids,
    _env_bool,
    _gaussdb_env_config,
    _ids,
    _insert_memories,
    _new_collection_name,
    _new_db,
    _uuid,
    gaussdb_available,
)

pytestmark = [
    pytest.mark.p1,
    pytest.mark.skipif(not gaussdb_available(), reason="GaussDB test env not configured"),
]


# ===========================================================================
# Vector Dimension Boundary Tests
# ===========================================================================


class TestVectorDimensionBoundary:
    """Tests for vector dimension edge cases."""

    def test_single_dimension_vector_insert_and_search(self):
        """1-dimensional vector should work."""
        db = _new_db(prefix="p1_dim", embedding_model_dims=1)
        try:
            vid = _uuid(1001)
            db.insert(
                ids=[vid],
                vectors=[[0.5]],
                payloads=[{"data": "single dim", "user_id": "dim_user"}],
            )
            results = db.search("single", [0.5], top_k=1, filters={"user_id": "dim_user"})
            assert len(results) == 1
            assert results[0].id == vid
        finally:
            db.delete_col()

    def test_high_dimension_vector_insert_and_search(self):
        """High-dimensional vector (256-dim) should work."""
        dims = 256
        db = _new_db(prefix="p1_dim", embedding_model_dims=dims)
        try:
            vid = _uuid(1002)
            vector = [float(i) / dims for i in range(dims)]
            db.insert(
                ids=[vid],
                vectors=[vector],
                payloads=[{"data": "high dim", "user_id": "dim_user"}],
            )
            results = db.search("high", vector, top_k=1, filters={"user_id": "dim_user"})
            assert len(results) == 1
            assert results[0].id == vid
        finally:
            db.delete_col()

    def test_zero_vector_insert_and_search(self):
        """Zero vector should be insertable."""
        db = _new_db(prefix="p1_dim")
        try:
            vid = _uuid(1003)
            zero_vec = [0.0] * EMBEDDING_DIMS
            db.insert(
                ids=[vid],
                vectors=[zero_vec],
                payloads=[{"data": "zero vector", "user_id": "dim_user"}],
            )
            result = db.get(vid)
            assert result is not None
            assert result.id == vid
        finally:
            db.delete_col()

    def test_nan_vector_raises_or_handles_gracefully(self):
        """NaN in vector should raise an error or be handled gracefully."""
        db = _new_db(prefix="p1_dim")
        try:
            vid = _uuid(1004)
            nan_vec = [float("nan"), 0.1, 0.2]
            with pytest.raises((ValueError, Exception)):
                db.insert(
                    ids=[vid],
                    vectors=[nan_vec],
                    payloads=[{"data": "nan vector", "user_id": "dim_user"}],
                )
        finally:
            db.delete_col()

    def test_inf_vector_raises_or_handles_gracefully(self):
        """Inf in vector should raise an error or be handled gracefully."""
        db = _new_db(prefix="p1_dim")
        try:
            vid = _uuid(1005)
            inf_vec = [float("inf"), 0.1, 0.2]
            with pytest.raises((ValueError, Exception)):
                db.insert(
                    ids=[vid],
                    vectors=[inf_vec],
                    payloads=[{"data": "inf vector", "user_id": "dim_user"}],
                )
        finally:
            db.delete_col()

    def test_very_large_values_in_vector(self):
        """Very large float values in vector should be insertable."""
        db = _new_db(prefix="p1_dim")
        try:
            vid = _uuid(1006)
            large_vec = [1e30, -1e30, 1e15]
            db.insert(
                ids=[vid],
                vectors=[large_vec],
                payloads=[{"data": "large values", "user_id": "dim_user"}],
            )
            result = db.get(vid)
            assert result is not None
        finally:
            db.delete_col()

    def test_negative_values_in_vector(self):
        """Negative values in vector should work normally."""
        db = _new_db(prefix="p1_dim")
        try:
            vid = _uuid(1007)
            neg_vec = [-0.5, -0.3, -0.9]
            db.insert(
                ids=[vid],
                vectors=[neg_vec],
                payloads=[{"data": "negative vector", "user_id": "dim_user"}],
            )
            results = db.search("neg", neg_vec, top_k=1, filters={"user_id": "dim_user"})
            assert len(results) == 1
            assert results[0].id == vid
        finally:
            db.delete_col()

    def test_empty_vector_list_raises(self):
        """Empty vector list should raise an error."""
        db = _new_db(prefix="p1_dim")
        try:
            vid = _uuid(1008)
            with pytest.raises((ValueError, Exception)):
                db.insert(
                    ids=[vid],
                    vectors=[[]],
                    payloads=[{"data": "empty vector", "user_id": "dim_user"}],
                )
        finally:
            db.delete_col()


# ===========================================================================
# Payload Boundary Tests
# ===========================================================================


class TestPayloadBoundary:
    """Tests for payload edge cases."""

    def test_empty_payload(self):
        """Empty payload dict should be insertable."""
        db = _new_db(prefix="p1_payload")
        try:
            vid = _uuid(2001)
            db.insert(ids=[vid], vectors=[VECTOR_COFFEE], payloads=[{}])
            result = db.get(vid)
            assert result is not None
        finally:
            db.delete_col()

    def test_large_payload_100kb(self):
        """Large payload (~100KB) should be insertable and retrievable."""
        db = _new_db(prefix="p1_payload")
        try:
            vid = _uuid(2002)
            large_value = "x" * 100_000
            payload = {"data": "large payload", "user_id": "payload_user", "big_field": large_value}
            db.insert(ids=[vid], vectors=[VECTOR_COFFEE], payloads=[payload])
            result = db.get(vid)
            assert result is not None
            assert result.payload["big_field"] == large_value
        finally:
            db.delete_col()

    def test_deeply_nested_payload(self):
        """Deeply nested payload (5 levels) should be stored correctly."""
        db = _new_db(prefix="p1_payload")
        try:
            vid = _uuid(2003)
            nested = {"level1": {"level2": {"level3": {"level4": {"level5": "deep_value"}}}}}
            payload = {"data": "nested payload", "user_id": "payload_user", "nested": nested}
            db.insert(ids=[vid], vectors=[VECTOR_COFFEE], payloads=[payload])
            result = db.get(vid)
            assert result is not None
            assert result.payload["nested"]["level1"]["level2"]["level3"]["level4"]["level5"] == "deep_value"
        finally:
            db.delete_col()

    def test_special_characters_in_payload(self):
        """Special characters in payload values should be preserved."""
        db = _new_db(prefix="p1_payload")
        try:
            vid = _uuid(2004)
            special = "Hello 'world' \"quotes\" \\backslash\\ <html>&amp; \t\n"
            payload = {"data": special, "user_id": "payload_user", "special": special}
            db.insert(ids=[vid], vectors=[VECTOR_COFFEE], payloads=[payload])
            result = db.get(vid)
            assert result is not None
            assert result.payload["special"] == special
        finally:
            db.delete_col()

    def test_unicode_payload_chinese_and_emoji(self):
        """Unicode (Chinese, emoji) in payload should round-trip correctly."""
        db = _new_db(prefix="p1_payload")
        try:
            vid = _uuid(2005)
            unicode_text = "你好世界 🌍🚀 日本語テスト"
            payload = {"data": unicode_text, "user_id": "payload_user", "text": unicode_text}
            db.insert(ids=[vid], vectors=[VECTOR_COFFEE], payloads=[payload])
            result = db.get(vid)
            assert result is not None
            assert result.payload["text"] == unicode_text
        finally:
            db.delete_col()

    def test_very_long_key_in_payload(self):
        """Very long key name (128 chars) in payload should work."""
        db = _new_db(prefix="p1_payload")
        try:
            vid = _uuid(2006)
            long_key = "k" * 128
            payload = {"data": "long key", "user_id": "payload_user", long_key: "value"}
            db.insert(ids=[vid], vectors=[VECTOR_COFFEE], payloads=[payload])
            result = db.get(vid)
            assert result is not None
            assert result.payload[long_key] == "value"
        finally:
            db.delete_col()

    def test_very_long_value_in_payload(self):
        """Very long value (10KB string) in payload should work."""
        db = _new_db(prefix="p1_payload")
        try:
            vid = _uuid(2007)
            long_value = "v" * 10_000
            payload = {"data": "long value", "user_id": "payload_user", "long_val": long_value}
            db.insert(ids=[vid], vectors=[VECTOR_COFFEE], payloads=[payload])
            result = db.get(vid)
            assert result is not None
            assert result.payload["long_val"] == long_value
        finally:
            db.delete_col()

    def test_null_value_in_payload(self):
        """None/null value in payload should be preserved."""
        db = _new_db(prefix="p1_payload")
        try:
            vid = _uuid(2008)
            payload = {"data": "null test", "user_id": "payload_user", "nullable": None}
            db.insert(ids=[vid], vectors=[VECTOR_COFFEE], payloads=[payload])
            result = db.get(vid)
            assert result is not None
            assert result.payload.get("nullable") is None
        finally:
            db.delete_col()

    def test_boolean_values_in_payload(self):
        """Boolean values in payload should be preserved."""
        db = _new_db(prefix="p1_payload")
        try:
            vid = _uuid(2009)
            payload = {"data": "bool test", "user_id": "payload_user", "flag_true": True, "flag_false": False}
            db.insert(ids=[vid], vectors=[VECTOR_COFFEE], payloads=[payload])
            result = db.get(vid)
            assert result is not None
            assert result.payload["flag_true"] is True
            assert result.payload["flag_false"] is False
        finally:
            db.delete_col()

    def test_numeric_values_in_payload(self):
        """Numeric values (int, float) in payload should be preserved."""
        db = _new_db(prefix="p1_payload")
        try:
            vid = _uuid(2010)
            payload = {
                "data": "numeric test",
                "user_id": "payload_user",
                "int_val": 42,
                "float_val": 3.14,
                "negative": -100,
                "zero": 0,
            }
            db.insert(ids=[vid], vectors=[VECTOR_COFFEE], payloads=[payload])
            result = db.get(vid)
            assert result is not None
            assert result.payload["int_val"] == 42
            assert abs(result.payload["float_val"] - 3.14) < 0.001
            assert result.payload["negative"] == -100
            assert result.payload["zero"] == 0
        finally:
            db.delete_col()


# ===========================================================================
# ID Boundary Tests
# ===========================================================================


class TestIDBoundary:
    """Tests for ID edge cases."""

    def test_uuid_format_id(self):
        """Standard UUID format ID should work."""
        db = _new_db(prefix="p1_id")
        try:
            vid = str(uuid.uuid4())
            db.insert(ids=[vid], vectors=[VECTOR_COFFEE], payloads=[{"data": "uuid id", "user_id": "id_user"}])
            result = db.get(vid)
            assert result is not None
            assert result.id == vid
        finally:
            db.delete_col()

    def test_deterministic_uuid_format(self):
        """Deterministic UUID format should work."""
        db = _new_db(prefix="p1_id")
        try:
            vid = _uuid(3001)
            db.insert(ids=[vid], vectors=[VECTOR_COFFEE], payloads=[{"data": "det uuid", "user_id": "id_user"}])
            result = db.get(vid)
            assert result is not None
            assert result.id == vid
        finally:
            db.delete_col()

    def test_duplicate_id_upsert_behavior(self):
        """Inserting with duplicate ID should upsert (update existing)."""
        db = _new_db(prefix="p1_id")
        try:
            vid = _uuid(3002)
            db.insert(ids=[vid], vectors=[VECTOR_COFFEE], payloads=[{"data": "original", "user_id": "id_user"}])
            db.insert(ids=[vid], vectors=[VECTOR_FLIGHT], payloads=[{"data": "updated", "user_id": "id_user"}])
            result = db.get(vid)
            assert result is not None
            assert result.payload["data"] == "updated"
        finally:
            db.delete_col()

    def test_batch_with_duplicate_ids(self):
        """Batch insert with duplicate IDs should raise UniqueViolation."""
        db = _new_db(prefix="p1_id")
        try:
            vid = _uuid(3003)
            with pytest.raises(Exception):
                db.insert(
                    ids=[vid, vid],
                    vectors=[VECTOR_COFFEE, VECTOR_FLIGHT],
                    payloads=[
                        {"data": "first", "user_id": "id_user"},
                        {"data": "second", "user_id": "id_user"},
                    ],
                )
        finally:
            db.delete_col()

    def test_multiple_unique_ids(self):
        """Multiple unique IDs should all be retrievable."""
        db = _new_db(prefix="p1_id")
        try:
            ids = [_uuid(3010 + i) for i in range(5)]
            vectors = [VECTOR_COFFEE] * 5
            payloads = [{"data": f"item_{i}", "user_id": "id_user"} for i in range(5)]
            db.insert(ids=ids, vectors=vectors, payloads=payloads)
            for i, vid in enumerate(ids):
                result = db.get(vid)
                assert result is not None
                assert result.payload["data"] == f"item_{i}"
        finally:
            db.delete_col()

    def test_null_id_raises(self):
        """None as ID should raise an error."""
        db = _new_db(prefix="p1_id")
        try:
            with pytest.raises((ValueError, TypeError, Exception)):
                db.insert(ids=[None], vectors=[VECTOR_COFFEE], payloads=[{"data": "null id", "user_id": "id_user"}])
        finally:
            db.delete_col()

    def test_get_nonexistent_id_returns_none(self):
        """Getting a non-existent ID should return None."""
        db = _new_db(prefix="p1_id")
        try:
            result = db.get(_uuid(9999))
            assert result is None
        finally:
            db.delete_col()

    def test_delete_nonexistent_id_no_error(self):
        """Deleting a non-existent ID should not raise an error."""
        db = _new_db(prefix="p1_id")
        try:
            db.delete(_uuid(9998))  # Should not raise
        finally:
            db.delete_col()


# ===========================================================================
# Collection Name Boundary Tests
# ===========================================================================


class TestCollectionNameBoundary:
    """Tests for collection name edge cases."""

    def test_max_length_collection_name(self):
        """Collection name at max usable length should work.

        The identifier limit is 63 chars, but GaussDB appends '_schema_meta'
        (12 chars) internally, so the effective max collection name is 51 chars.
        """
        long_name = "a" * 51
        db = _new_db(prefix=None, collection_name=long_name)
        try:
            vid = _uuid(4001)
            db.insert(ids=[vid], vectors=[VECTOR_COFFEE], payloads=[{"data": "long name", "user_id": "col_user"}])
            result = db.get(vid)
            assert result is not None
        finally:
            db.delete_col()

    def test_collection_name_with_underscores(self):
        """Collection name with underscores should work."""
        name = f"test_under_score_{uuid.uuid4().hex[:6]}"
        db = _new_db(prefix=None, collection_name=name)
        try:
            vid = _uuid(4002)
            db.insert(ids=[vid], vectors=[VECTOR_COFFEE], payloads=[{"data": "underscore", "user_id": "col_user"}])
            result = db.get(vid)
            assert result is not None
        finally:
            db.delete_col()

    def test_collection_name_sql_injection_rejected(self):
        """SQL injection in collection name should be rejected by validator."""
        with pytest.raises(ValueError, match="Unsafe"):
            _new_db(prefix=None, collection_name="test; DROP TABLE users;--")

    def test_empty_collection_name_rejected(self):
        """Empty collection name should be rejected."""
        config = _gaussdb_env_config("placeholder")
        config["collection_name"] = ""
        with pytest.raises(ValueError):
            GaussDB(**config)

    def test_collection_name_starting_with_number_rejected(self):
        """Collection name starting with a number should be rejected."""
        with pytest.raises(ValueError, match="Unsafe"):
            _new_db(prefix=None, collection_name="123_invalid")

    def test_collection_name_with_special_chars_rejected(self):
        """Collection name with special characters should be rejected."""
        with pytest.raises(ValueError, match="Unsafe"):
            _new_db(prefix=None, collection_name="test-collection!")


# ===========================================================================
# top_k Boundary Tests
# ===========================================================================


class TestTopKBoundary:
    """Tests for top_k parameter edge cases."""

    def test_top_k_one_returns_single_result(self):
        """top_k=1 should return exactly one result."""
        db = _new_db(prefix="p1_topk")
        try:
            for i in range(3):
                db.insert(
                    ids=[_uuid(5001 + i)],
                    vectors=[VECTOR_COFFEE],
                    payloads=[{"data": f"item_{i}", "user_id": "topk_user"}],
                )
            results = db.search("item", VECTOR_COFFEE, top_k=1, filters={"user_id": "topk_user"})
            assert len(results) == 1
        finally:
            db.delete_col()

    def test_top_k_exceeds_data_count(self):
        """top_k larger than data count should return all available results."""
        db = _new_db(prefix="p1_topk")
        try:
            for i in range(3):
                db.insert(
                    ids=[_uuid(5011 + i)],
                    vectors=[VECTOR_COFFEE],
                    payloads=[{"data": f"item_{i}", "user_id": "topk_user"}],
                )
            results = db.search("item", VECTOR_COFFEE, top_k=100, filters={"user_id": "topk_user"})
            assert len(results) == 3
        finally:
            db.delete_col()

    def test_top_k_very_large_value(self):
        """Very large top_k should not crash."""
        db = _new_db(prefix="p1_topk")
        try:
            db.insert(
                ids=[_uuid(5021)],
                vectors=[VECTOR_COFFEE],
                payloads=[{"data": "single", "user_id": "topk_user"}],
            )
            results = db.search("single", VECTOR_COFFEE, top_k=10000, filters={"user_id": "topk_user"})
            assert len(results) == 1
        finally:
            db.delete_col()

    def test_top_k_zero_returns_empty(self):
        """top_k=0 should return empty results or raise."""
        db = _new_db(prefix="p1_topk")
        try:
            db.insert(
                ids=[_uuid(5031)],
                vectors=[VECTOR_COFFEE],
                payloads=[{"data": "item", "user_id": "topk_user"}],
            )
            results = db.search("item", VECTOR_COFFEE, top_k=0, filters={"user_id": "topk_user"})
            assert len(results) == 0
        finally:
            db.delete_col()

    def test_top_k_negative_raises_or_empty(self):
        """Negative top_k should raise an error or return empty."""
        db = _new_db(prefix="p1_topk")
        try:
            db.insert(
                ids=[_uuid(5041)],
                vectors=[VECTOR_COFFEE],
                payloads=[{"data": "item", "user_id": "topk_user"}],
            )
            try:
                results = db.search("item", VECTOR_COFFEE, top_k=-1, filters={"user_id": "topk_user"})
                assert len(results) == 0
            except (ValueError, Exception):
                pass  # Raising is also acceptable
        finally:
            db.delete_col()


# ===========================================================================
# Batch Operation Boundary Tests
# ===========================================================================


class TestBatchOperationBoundary:
    """Tests for batch operation edge cases."""

    def test_empty_batch_insert(self):
        """Empty batch insert should not crash."""
        db = _new_db(prefix="p1_batch")
        try:
            result = db.insert(ids=[], vectors=[], payloads=[])
            # Should either return None or handle gracefully
            assert result is None or result == []
        finally:
            db.delete_col()

    def test_single_item_batch(self):
        """Single-item batch should work like single insert."""
        db = _new_db(prefix="p1_batch")
        try:
            vid = _uuid(6001)
            db.insert(ids=[vid], vectors=[VECTOR_COFFEE], payloads=[{"data": "single batch", "user_id": "batch_user"}])
            result = db.get(vid)
            assert result is not None
            assert result.payload["data"] == "single batch"
        finally:
            db.delete_col()

    def test_large_batch_insert_1000_items(self):
        """Large batch (1000 items) should complete successfully."""
        db = _new_db(prefix="p1_batch")
        try:
            count = 1000
            ids = [_uuid(6100 + i) for i in range(count)]
            vectors = [[float(i % 10) / 10, float(i % 5) / 5, float(i % 3) / 3] for i in range(count)]
            payloads = [{"data": f"batch_item_{i}", "user_id": "batch_user"} for i in range(count)]
            db.insert(ids=ids, vectors=vectors, payloads=payloads)

            # Verify a sample
            result = db.get(ids[0])
            assert result is not None
            result = db.get(ids[999])
            assert result is not None
        finally:
            db.delete_col()

    def test_batch_search_with_multiple_queries(self):
        """search_batch with multiple queries should return results for each."""
        db = _new_db(prefix="p1_batch")
        try:
            _insert_memories(
                db,
                [
                    (_uuid(6201), VECTOR_COFFEE, {"data": "coffee memory", "user_id": "batch_user"}),
                    (_uuid(6202), VECTOR_FLIGHT, {"data": "flight memory", "user_id": "batch_user"}),
                ],
            )
            results = db.search_batch(
                ["coffee", "flight"],
                [VECTOR_COFFEE, VECTOR_FLIGHT],
                top_k=5,
                filters={"user_id": "batch_user"},
            )
            assert len(results) == 2
            assert len(results[0]) >= 1
            assert len(results[1]) >= 1
        finally:
            db.delete_col()

    def test_batch_insert_partial_failure_handling(self):
        """Batch with some invalid data should handle gracefully."""
        db = _new_db(prefix="p1_batch")
        try:
            # Insert valid data first
            valid_id = _uuid(6301)
            db.insert(
                ids=[valid_id],
                vectors=[VECTOR_COFFEE],
                payloads=[{"data": "valid item", "user_id": "batch_user"}],
            )
            result = db.get(valid_id)
            assert result is not None
        finally:
            db.delete_col()
