"""
GaussDB Distributed Mode - Full E2E Test Suite for mem0 Adaptation.

Validates distributed-specific logic:
- Parameter validation (dimension limits, BM25 conflicts, distribution_mode resolution)
- DDL generation (DISTRIBUTE BY HASH)
- CRUD correctness across distributed nodes
- Vector search aggregation across DNs
- BM25 graceful degradation
- Filter correctness in distributed mode
- Data consistency and integrity
- Collection lifecycle operations
- Boundary conditions and error handling
- Integration with mem0 upper-layer Memory API

Environment variables:
    GAUSSDB_TEST_HOST / GAUSSDB_TEST_PORT / GAUSSDB_TEST_DATABASE / GAUSSDB_TEST_USER / GAUSSDB_TEST_PASSWORD
    or GAUSSDB_TEST_DSN

    GAUSSDB_TEST_DISTRIBUTED=true   (required to run these tests)

Usage:
    export GAUSSDB_TEST_DISTRIBUTED=true
    pytest tests/vector_stores/test_gaussdb_distributed.py -v
"""

import math
import os
import time
import uuid
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("psycopg2", reason="GaussDB distributed tests require psycopg2-compatible driver")

from mem0.vector_stores.gaussdb import GaussDB


# ---------------------------------------------------------------------------
# Test configuration helpers
# ---------------------------------------------------------------------------

DIMS_SMALL = 4
DIMS_BOUNDARY = 1024

VECTORS_4D = {
    "A": [1.0, 0.0, 0.0, 0.0],
    "B": [0.0, 1.0, 0.0, 0.0],
    "C": [0.9, 0.1, 0.0, 0.0],
    "D": [0.0, 0.0, 1.0, 0.0],
    "E": [0.0, 0.0, 0.0, 1.0],
    "similar_A": [0.95, 0.05, 0.0, 0.0],
}


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _new_collection(prefix: str = "mem0_dist") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def _uuid(suffix: int) -> str:
    return f"00000000-0000-0000-0000-{suffix:012d}"


def _gaussdb_distributed_config(collection_name: str, **overrides):
    """Build config dict for distributed mode testing."""
    dsn = os.getenv("GAUSSDB_TEST_DSN")
    if dsn:
        config = {"connection_string": dsn}
    else:
        required = {
            "host": os.getenv("GAUSSDB_TEST_HOST"),
            "port": os.getenv("GAUSSDB_TEST_PORT"),
            "database": os.getenv("GAUSSDB_TEST_DATABASE"),
            "user": os.getenv("GAUSSDB_TEST_USER"),
            "password": os.getenv("GAUSSDB_TEST_PASSWORD"),
        }
        if not all(required.values()):
            return None
        config = {
            "host": required["host"],
            "port": int(required["port"]),
            "database": required["database"],
            "user": required["user"],
            "password": required["password"],
        }

    optional_env = {
        "sslmode": os.getenv("GAUSSDB_TEST_SSLMODE"),
        "sslrootcert": os.getenv("GAUSSDB_TEST_SSLROOTCERT"),
    }
    config.update({key: value for key, value in optional_env.items() if value})
    config.update(
        {
            "collection_name": collection_name,
            "embedding_model_dims": DIMS_SMALL,
            "id_column_type": "uuid",
            "deployment_mode": "distributed",
            "distribution_mode": "auto",
            "vector_index_type": os.getenv("GAUSSDB_TEST_VECTOR_INDEX", "gsivfflat"),
            "vector_index_maintenance_work_mem": os.getenv("GAUSSDB_TEST_MAINTENANCE_MEM", "128MB"),
            "bm25_mode": "disabled",
            "enable_capability_probe": True,
            "require_scoped_filters": True,
            "auto_create": True,
            "scope_filter_keys": ["user_id", "agent_id", "run_id"],
        }
    )
    config.update({key: value for key, value in overrides.items() if value is not None})
    return config


def _new_distributed_db(collection_name: str = None, **overrides) -> GaussDB:
    config = _gaussdb_distributed_config(collection_name or _new_collection(), **overrides)
    assert config is not None, "Missing GaussDB test environment variables"
    return GaussDB(**config)


def _insert_records(db: GaussDB, records: list[tuple[str, list[float], dict]]) -> None:
    db.insert(
        ids=[r[0] for r in records],
        vectors=[r[1] for r in records],
        payloads=[r[2] for r in records],
    )


def _ids(rows) -> list[str]:
    return [str(row.id) for row in rows]


# ---------------------------------------------------------------------------
# Skip conditions
# ---------------------------------------------------------------------------

pytestmark = [
    pytest.mark.skipif(
        not _env_bool("GAUSSDB_TEST_DISTRIBUTED"),
        reason="Set GAUSSDB_TEST_DISTRIBUTED=true to run distributed tests",
    ),
    pytest.mark.skipif(
        _gaussdb_distributed_config("probe") is None,
        reason="Set GAUSSDB_TEST_DSN or GAUSSDB_TEST_HOST/PORT/DATABASE/USER/PASSWORD",
    ),
]


# ===========================================================================
# P0 - Initialization & Parameter Validation
# ===========================================================================


class TestDistributedInitialization:
    """TC-D001 ~ TC-D006: Parameter validation and initialization logic."""

    def test_d001_basic_distributed_init(self):
        """TC-D001: distributed mode initializes with hash distribution, BM25 disabled."""
        db = _new_distributed_db()
        try:
            assert db.deployment_mode == "distributed"
            assert db.distribution_mode == "hash"
            assert db.bm25_enabled is False
        finally:
            db.delete_col()

    def test_d002_dimension_exceeds_1024_rejected(self):
        """TC-D002: embedding_model_dims > 1024 raises ValueError in distributed mode."""
        with pytest.raises(ValueError, match="distributed mode only supports embedding dimensions <= 1024"):
            _new_distributed_db(embedding_model_dims=1025)

    def test_d003_dimension_boundary_1024(self):
        """TC-D003: embedding_model_dims=1024 is accepted in distributed mode."""
        db = _new_distributed_db(embedding_model_dims=1024)
        try:
            assert db.embedding_model_dims == 1024
            vec_1024 = [0.01] * 1024
            db.insert(ids=[_uuid(1)], vectors=[vec_1024], payloads=[{"user_id": "u1", "data": "test"}])
            result = db.get(_uuid(1))
            assert result is not None
            assert result.id == _uuid(1)
        finally:
            db.delete_col()

    def test_d004_bm25_required_distributed_conflict(self):
        """TC-D004: bm25_mode=required + distributed raises ValueError."""
        with pytest.raises(ValueError, match="bm25_mode=.required. is incompatible with deployment_mode=.distributed."):
            _new_distributed_db(bm25_mode="required")

    def test_d005_distribution_mode_auto_resolution(self):
        """TC-D005: distribution_mode=auto resolves to hash for distributed, none for centralized."""
        db_dist = _new_distributed_db(distribution_mode="auto")
        try:
            assert db_dist.distribution_mode == "hash"
        finally:
            db_dist.delete_col()

        config = _gaussdb_distributed_config(
            _new_collection(), deployment_mode="centralized", distribution_mode="auto"
        )
        db_cent = GaussDB(**config)
        try:
            assert db_cent.distribution_mode == "none"
        finally:
            db_cent.delete_col()

    def test_d006_centralized_hash_conflict(self):
        """TC-D006: centralized + distribution_mode=hash raises ValueError."""
        with pytest.raises(ValueError, match="distribution_mode can only be enabled when deployment_mode=.distributed."):
            config = _gaussdb_distributed_config(
                _new_collection(), deployment_mode="centralized", distribution_mode="hash"
            )
            GaussDB(**config)


# ===========================================================================
# P0 - DDL Generation & Table Creation
# ===========================================================================


class TestDistributedDDL:
    """TC-D010 ~ TC-D012: DDL with DISTRIBUTE BY HASH."""

    def test_d010_table_has_distribute_by_hash(self):
        """TC-D010: Created table uses DISTRIBUTE BY HASH(id)."""
        db = _new_distributed_db()
        try:
            with db._get_cursor() as cur:
                cur.execute(
                    """
                    SELECT pclocatortype FROM pgxc_class
                    WHERE pcrelid = (
                        SELECT oid FROM pg_class WHERE relname = %s
                    )
                    """,
                    (db.table_name,),
                )
                row = cur.fetchone()
                assert row is not None, "Table not found in pgxc_class"
                assert row[0] == "H", f"Expected HASH distribution, got {row[0]}"
        finally:
            db.delete_col()

    def test_d011_distribution_clause_sql_generation(self):
        """TC-D011: _distribution_clause_sql generates correct SQL."""
        db = _new_distributed_db()
        try:
            clause = db._distribution_clause_sql("id")
            assert "DISTRIBUTE BY HASH" in clause
            assert "id" in clause
        finally:
            db.delete_col()

    def test_d012_capability_probe_in_distributed_mode(self):
        """TC-D012: Capability probe succeeds in distributed mode."""
        db = _new_distributed_db(enable_capability_probe=True)
        try:
            assert db.capabilities is not None
            assert db.capabilities.vector_enabled is True
            assert db.bm25_enabled is False
        finally:
            db.delete_col()


# ===========================================================================
# P0 - CRUD Operations in Distributed Mode
# ===========================================================================


class TestDistributedCRUD:
    """TC-D020 ~ TC-D024: Basic CRUD on distributed tables."""

    @pytest.fixture(autouse=True)
    def setup_db(self):
        self.db = _new_distributed_db()
        yield
        self.db.delete_col()

    def test_d020_insert_single_record(self):
        """TC-D020: Insert single record and retrieve it."""
        record_id = _uuid(1)
        self.db.insert(
            ids=[record_id],
            vectors=[VECTORS_4D["A"]],
            payloads=[{"data": "hello distributed", "user_id": "u1", "agent_id": "a1", "run_id": "r1"}],
        )
        got = self.db.get(record_id)
        assert got is not None
        assert got.id == record_id
        assert got.payload["data"] == "hello distributed"
        assert got.payload["user_id"] == "u1"

    def test_d021_insert_batch_records(self):
        """TC-D021: Batch insert 20 records, verify count and data integrity."""
        records = []
        for i in range(20):
            records.append((
                _uuid(i + 1),
                [float(i % 4 == j) for j in range(4)],
                {"data": f"record_{i}", "user_id": f"u{i % 3}", "agent_id": "a1", "run_id": "r1"},
            ))
        _insert_records(self.db, records)

        info = self.db.col_info()
        assert info["count"] == 20

        for idx in [0, 10, 19]:
            got = self.db.get(_uuid(idx + 1))
            assert got is not None
            assert got.payload["data"] == f"record_{idx}"

    def test_d022_update_record(self):
        """TC-D022: Update vector and payload of existing record."""
        record_id = _uuid(1)
        self.db.insert(
            ids=[record_id],
            vectors=[VECTORS_4D["A"]],
            payloads=[{"data": "original", "user_id": "u1", "agent_id": "a1", "run_id": "r1"}],
        )

        self.db.update(
            vector_id=record_id,
            vector=VECTORS_4D["B"],
            payload={"data": "updated", "user_id": "u1", "agent_id": "a1", "run_id": "r1"},
        )

        got = self.db.get(record_id)
        assert got.payload["data"] == "updated"

    def test_d023_delete_single(self):
        """TC-D023: Delete single record."""
        record_id = _uuid(1)
        self.db.insert(
            ids=[record_id],
            vectors=[VECTORS_4D["A"]],
            payloads=[{"data": "to_delete", "user_id": "u1", "agent_id": "a1", "run_id": "r1"}],
        )
        assert self.db.get(record_id) is not None

        self.db.delete(vector_id=record_id)
        assert self.db.get(record_id) is None

    def test_d024_delete_batch(self):
        """TC-D024: Batch delete multiple records across potential DN boundaries."""
        ids_to_insert = [_uuid(i) for i in range(1, 11)]
        records = [
            (uid, VECTORS_4D["A"], {"data": f"rec_{i}", "user_id": "u1", "agent_id": "a1", "run_id": "r1"})
            for i, uid in enumerate(ids_to_insert)
        ]
        _insert_records(self.db, records)

        ids_to_delete = ids_to_insert[:5]
        for uid in ids_to_delete:
            self.db.delete(vector_id=uid)

        for uid in ids_to_delete:
            assert self.db.get(uid) is None
        for uid in ids_to_insert[5:]:
            assert self.db.get(uid) is not None


# ===========================================================================
# P0 - Vector Search in Distributed Mode
# ===========================================================================


class TestDistributedSearch:
    """TC-D030 ~ TC-D033: Vector search correctness across distributed nodes."""

    @pytest.fixture(autouse=True)
    def setup_db(self):
        self.db = _new_distributed_db()
        _insert_records(self.db, [
            (_uuid(1), VECTORS_4D["A"], {"data": "vec_A", "user_id": "u1", "agent_id": "a1", "run_id": "r1"}),
            (_uuid(2), VECTORS_4D["B"], {"data": "vec_B", "user_id": "u1", "agent_id": "a1", "run_id": "r1"}),
            (_uuid(3), VECTORS_4D["C"], {"data": "vec_C", "user_id": "u1", "agent_id": "a1", "run_id": "r1"}),
            (_uuid(4), VECTORS_4D["D"], {"data": "vec_D", "user_id": "u2", "agent_id": "a1", "run_id": "r1"}),
            (_uuid(5), VECTORS_4D["E"], {"data": "vec_E", "user_id": "u2", "agent_id": "a1", "run_id": "r1"}),
        ])
        yield
        self.db.delete_col()

    def test_d030_cosine_search_ordering(self):
        """TC-D030: Cosine search returns correct ordering (A > C > B for query=A)."""
        results = self.db.search(
            query="test",
            vectors=VECTORS_4D["A"],
            top_k=5,
            filters={"user_id": "u1"},
        )
        ids = _ids(results)
        assert ids[0] == _uuid(1), f"Expected A first, got {ids}"
        assert ids[1] == _uuid(3), f"Expected C second, got {ids}"
        assert ids[2] == _uuid(2), f"Expected B third, got {ids}"

    def test_d031_search_cross_dn_aggregation(self):
        """TC-D031: Search aggregates results from multiple DNs correctly."""
        records = []
        for i in range(100):
            angle = 2 * math.pi * i / 100
            vec = [math.cos(angle), math.sin(angle), 0.0, 0.0]
            records.append((
                str(uuid.uuid4()),
                vec,
                {"data": f"bulk_{i}", "user_id": "bulk_user", "agent_id": "a1", "run_id": "r1"},
            ))
        _insert_records(self.db, records)

        results = self.db.search(
            query="test",
            vectors=[1.0, 0.0, 0.0, 0.0],
            top_k=10,
            filters={"user_id": "bulk_user"},
        )
        assert len(results) == 10
        scores = [r.score for r in results]
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1], f"Scores not sorted: {scores}"

    def test_d032_search_batch(self):
        """TC-D032: Batch search returns independent result sets."""
        results_batch = self.db.search_batch(
            queries=["q1", "q2", "q3"],
            vectors_list=[VECTORS_4D["A"], VECTORS_4D["B"], VECTORS_4D["D"]],
            top_k=2,
            filters={"user_id": "u1"},
        )
        assert len(results_batch) == 3
        assert _ids(results_batch[0])[0] == _uuid(1)
        assert _ids(results_batch[1])[0] == _uuid(2)
        assert len(results_batch[2]) <= 2

    def test_d033_search_with_filter(self):
        """TC-D033: Search with filter only returns matching records."""
        results = self.db.search(
            query="test",
            vectors=VECTORS_4D["A"],
            top_k=10,
            filters={"user_id": "u2"},
        )
        for r in results:
            assert r.payload["user_id"] == "u2"
        assert len(results) == 2


# ===========================================================================
# P1 - BM25 Degradation Behavior
# ===========================================================================


class TestDistributedBM25Degradation:
    """TC-D040 ~ TC-D042: BM25 graceful degradation in distributed mode."""

    @pytest.fixture(autouse=True)
    def setup_db(self):
        self.db = _new_distributed_db()
        _insert_records(self.db, [
            (_uuid(1), VECTORS_4D["A"], {"data": "I love hotpot", "user_id": "u1", "agent_id": "a1", "run_id": "r1"}),
            (_uuid(2), VECTORS_4D["B"], {"data": "hotpot is great", "user_id": "u1", "agent_id": "a1", "run_id": "r1"}),
        ])
        yield
        self.db.delete_col()

    def test_d040_keyword_search_returns_none(self):
        """TC-D040: keyword_search returns None when BM25 is disabled."""
        result = self.db.keyword_search("hotpot", top_k=5, filters={"user_id": "u1"})
        assert result is None

    def test_d041_bm25_auto_disabled_in_distributed(self):
        """TC-D041: bm25_mode=auto resolves to disabled in distributed mode."""
        db = _new_distributed_db(bm25_mode="auto")
        try:
            assert db.bm25_enabled is False
        finally:
            db.delete_col()

    def test_d042_memory_search_works_without_bm25(self):
        """TC-D042: Vector search still works when BM25 is disabled."""
        results = self.db.search(
            query="hotpot",
            vectors=VECTORS_4D["A"],
            top_k=5,
            filters={"user_id": "u1"},
        )
        assert len(results) > 0


# ===========================================================================
# P1 - Filters in Distributed Mode
# ===========================================================================


class TestDistributedFilters:
    """TC-D050 ~ TC-D052: Filter correctness across distributed nodes."""

    @pytest.fixture(autouse=True)
    def setup_db(self):
        self.db = _new_distributed_db()
        _insert_records(self.db, [
            (_uuid(1), VECTORS_4D["A"], {"data": "work_A", "user_id": "user_A", "agent_id": "a1", "run_id": "r1", "category": "work"}),
            (_uuid(2), VECTORS_4D["B"], {"data": "personal_A", "user_id": "user_A", "agent_id": "a1", "run_id": "r1", "category": "personal"}),
            (_uuid(3), VECTORS_4D["C"], {"data": "work_B", "user_id": "user_B", "agent_id": "a1", "run_id": "r1", "category": "work"}),
            (_uuid(4), VECTORS_4D["D"], {"data": "personal_B", "user_id": "user_B", "agent_id": "a2", "run_id": "r1", "category": "personal"}),
            (_uuid(5), VECTORS_4D["E"], {"data": "work_A2", "user_id": "user_A", "agent_id": "a2", "run_id": "r2", "category": "work"}),
        ])
        yield
        self.db.delete_col()

    def test_d050_scope_filter_user_id(self):
        """TC-D050: Scope filter by user_id works in distributed mode."""
        results = self.db.search(
            query="test", vectors=VECTORS_4D["A"], top_k=10,
            filters={"user_id": "user_A"},
        )
        assert len(results) == 3
        for r in results:
            assert r.payload["user_id"] == "user_A"

    def test_d051_compound_filter_and(self):
        """TC-D051: Compound AND filter (user_id + category) in distributed mode."""
        results = self.db.search(
            query="test", vectors=VECTORS_4D["A"], top_k=10,
            filters={"user_id": "user_A", "category": "work"},
        )
        assert len(results) == 2
        for r in results:
            assert r.payload["user_id"] == "user_A"
            assert r.payload["category"] == "work"

    def test_d052_scope_filter_agent_id(self):
        """TC-D052: Scope filter by agent_id works in distributed mode."""
        results = self.db.search(
            query="test", vectors=VECTORS_4D["A"], top_k=10,
            filters={"user_id": "user_A", "agent_id": "a2"},
        )
        assert len(results) == 1
        assert results[0].payload["data"] == "work_A2"


# ===========================================================================
# P1 - Data Consistency
# ===========================================================================


class TestDistributedDataConsistency:
    """TC-D060 ~ TC-D064: Data integrity and consistency in distributed mode."""

    @pytest.fixture(autouse=True)
    def setup_db(self):
        self.db = _new_distributed_db()
        yield
        self.db.delete_col()

    def test_d060_upsert_idempotency(self):
        """TC-D060: Upsert same ID twice does not create duplicates."""
        record_id = _uuid(1)
        self.db.insert(
            ids=[record_id],
            vectors=[VECTORS_4D["A"]],
            payloads=[{"data": "first", "user_id": "u1", "agent_id": "a1", "run_id": "r1"}],
        )
        self.db.insert(
            ids=[record_id],
            vectors=[VECTORS_4D["B"]],
            payloads=[{"data": "second", "user_id": "u1", "agent_id": "a1", "run_id": "r1"}],
        )
        info = self.db.col_info()
        assert info["count"] == 1
        got = self.db.get(record_id)
        assert got.payload["data"] == "second"

    def test_d061_utf8_payload_roundtrip(self):
        """TC-D061: UTF-8 characters (Chinese, emoji) survive insert/get cycle."""
        record_id = _uuid(1)
        chinese_text = "我喜欢吃火锅"
        mixed_text = "mixed 中英文 content"
        payload = {
            "data": chinese_text,
            "user_id": "用户A",
            "agent_id": "a1",
            "run_id": "r1",
            "note": mixed_text,
        }
        self.db.insert(ids=[record_id], vectors=[VECTORS_4D["A"]], payloads=[payload])
        got = self.db.get(record_id)
        assert got.payload["data"] == chinese_text
        assert got.payload["user_id"] == "用户A"
        assert got.payload["note"] == mixed_text

    def test_d062_null_and_empty_payload_fields(self):
        """TC-D062: NULL and empty string payload fields handled correctly."""
        record_id = _uuid(1)
        payload = {
            "data": "",
            "user_id": "u1",
            "agent_id": "a1",
            "run_id": "r1",
            "optional_field": None,
            "empty_list": [],
        }
        self.db.insert(ids=[record_id], vectors=[VECTORS_4D["A"]], payloads=[payload])
        got = self.db.get(record_id)
        assert got.payload["data"] == ""
        assert got.payload.get("optional_field") is None
        assert got.payload["empty_list"] == []

    def test_d063_concurrent_inserts(self):
        """TC-D063: Concurrent inserts from multiple threads do not lose data."""
        num_records = 50

        def insert_one(i):
            self.db.insert(
                ids=[str(uuid.uuid4())],
                vectors=[[float(i % 4 == j) for j in range(4)]],
                payloads=[{"data": f"concurrent_{i}", "user_id": "conc", "agent_id": "a1", "run_id": "r1"}],
            )

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(insert_one, i) for i in range(num_records)]
            for f in as_completed(futures):
                f.result()

        info = self.db.col_info()
        assert info["count"] == num_records

    def test_d064_concurrent_read_write(self):
        """TC-D064: Concurrent reads and writes do not corrupt data."""
        base_records = [
            (_uuid(i), VECTORS_4D["A"], {"data": f"base_{i}", "user_id": "rw", "agent_id": "a1", "run_id": "r1"})
            for i in range(1, 11)
        ]
        _insert_records(self.db, base_records)

        errors = []

        def writer(start_idx):
            try:
                for i in range(5):
                    self.db.insert(
                        ids=[str(uuid.uuid4())],
                        vectors=[VECTORS_4D["B"]],
                        payloads=[{"data": f"new_{start_idx}_{i}", "user_id": "rw", "agent_id": "a1", "run_id": "r1"}],
                    )
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for _ in range(5):
                    self.db.search(
                        query="test", vectors=VECTORS_4D["A"], top_k=5,
                        filters={"user_id": "rw"},
                    )
            except Exception as e:
                errors.append(e)

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            futures.append(executor.submit(writer, 100))
            futures.append(executor.submit(writer, 200))
            futures.append(executor.submit(reader))
            futures.append(executor.submit(reader))
            for f in as_completed(futures):
                f.result()

        assert len(errors) == 0, f"Concurrent errors: {errors}"


# ===========================================================================
# P1 - Collection Lifecycle
# ===========================================================================


class TestDistributedCollectionLifecycle:
    """TC-D070 ~ TC-D073: Collection operations in distributed mode."""

    def test_d070_list_cols(self):
        """TC-D070: list_cols includes the distributed collection."""
        db = _new_distributed_db()
        try:
            cols = db.list_cols()
            assert db.collection_name in cols
        finally:
            db.delete_col()

    def test_d071_col_info(self):
        """TC-D071: col_info returns correct metadata for distributed collection."""
        db = _new_distributed_db()
        try:
            _insert_records(db, [
                (_uuid(1), VECTORS_4D["A"], {"data": "test", "user_id": "u1", "agent_id": "a1", "run_id": "r1"}),
                (_uuid(2), VECTORS_4D["B"], {"data": "test2", "user_id": "u1", "agent_id": "a1", "run_id": "r1"}),
            ])
            info = db.col_info()
            assert info["name"] == db.collection_name
            assert info["count"] == 2
            assert info["dimension"] == DIMS_SMALL
        finally:
            db.delete_col()

    def test_d072_reset_collection(self):
        """TC-D072: reset() clears all data but keeps the collection."""
        db = _new_distributed_db()
        try:
            _insert_records(db, [
                (_uuid(1), VECTORS_4D["A"], {"data": "test", "user_id": "u1", "agent_id": "a1", "run_id": "r1"}),
            ])
            assert db.col_info()["count"] == 1
            db.reset()
            assert db.col_info()["count"] == 0
        finally:
            db.delete_col()

    def test_d073_delete_col(self):
        """TC-D073: delete_col removes the collection entirely."""
        db = _new_distributed_db()
        col_name = db.collection_name
        db.insert(
            ids=[_uuid(1)],
            vectors=[VECTORS_4D["A"]],
            payloads=[{"data": "test", "user_id": "u1", "agent_id": "a1", "run_id": "r1"}],
        )
        db.delete_col()
        db2 = _new_distributed_db()
        try:
            cols = db2.list_cols()
            assert col_name not in cols
        finally:
            db2.delete_col()


# ===========================================================================
# P1 - Index Operations
# ===========================================================================


class TestDistributedIndex:
    """TC-D080 ~ TC-D081: Vector index in distributed mode."""

    def test_d080_vector_index_created(self):
        """TC-D080: Vector index is created on distributed table."""
        db = _new_distributed_db()
        try:
            with db._get_cursor() as cur:
                cur.execute(
                    """
                    SELECT indexname FROM pg_indexes
                    WHERE schemaname = %s AND tablename = %s AND indexdef LIKE %s
                    """,
                    (db.schema, db.collection_name, "%vector%"),
                )
                rows = cur.fetchall()
                assert len(rows) > 0, "No vector index found"
        finally:
            db.delete_col()

    def test_d081_analyze_runs_without_error(self):
        """TC-D081: ANALYZE on distributed table succeeds."""
        db = _new_distributed_db()
        try:
            _insert_records(db, [
                (_uuid(i), VECTORS_4D["A"], {"data": f"rec_{i}", "user_id": "u1", "agent_id": "a1", "run_id": "r1"})
                for i in range(1, 6)
            ])
            db.analyze()
        finally:
            db.delete_col()


# ===========================================================================
# P2 - Boundary Conditions & Edge Cases
# ===========================================================================


class TestDistributedBoundary:
    """TC-D090 ~ TC-D095: Edge cases specific to distributed mode."""

    @pytest.fixture(autouse=True)
    def setup_db(self):
        self.db = _new_distributed_db()
        yield
        self.db.delete_col()

    def test_d090_empty_collection_search(self):
        """TC-D090: Search on empty distributed collection returns empty list."""
        results = self.db.search(
            query="test", vectors=VECTORS_4D["A"], top_k=10,
            filters={"user_id": "nobody"},
        )
        assert results == []

    def test_d091_top_k_exceeds_total_records(self):
        """TC-D091: top_k > total records returns all available records."""
        _insert_records(self.db, [
            (_uuid(1), VECTORS_4D["A"], {"data": "one", "user_id": "u1", "agent_id": "a1", "run_id": "r1"}),
            (_uuid(2), VECTORS_4D["B"], {"data": "two", "user_id": "u1", "agent_id": "a1", "run_id": "r1"}),
        ])
        results = self.db.search(
            query="test", vectors=VECTORS_4D["A"], top_k=100,
            filters={"user_id": "u1"},
        )
        assert len(results) == 2

    def test_d092_special_chars_in_payload(self):
        """TC-D092: Special characters in payload survive distributed insert/get."""
        record_id = _uuid(1)
        special_data = "line1" + chr(10) + "line2" + chr(9) + "tab" + chr(92) + "backslash"
        payload = {
            "data": special_data,
            "user_id": "u1",
            "agent_id": "a1",
            "run_id": "r1",
            "sql_injection": "'; DROP TABLE mem0; --",
        }
        self.db.insert(ids=[record_id], vectors=[VECTORS_4D["A"]], payloads=[payload])
        got = self.db.get(record_id)
        assert got.payload["data"] == payload["data"]
        assert got.payload["sql_injection"] == payload["sql_injection"]

    def test_d093_large_payload(self):
        """TC-D093: Large payload (64KB JSON) works in distributed mode."""
        record_id = _uuid(1)
        large_text = "x" * 65536
        payload = {
            "data": large_text,
            "user_id": "u1",
            "agent_id": "a1",
            "run_id": "r1",
        }
        self.db.insert(ids=[record_id], vectors=[VECTORS_4D["A"]], payloads=[payload])
        got = self.db.get(record_id)
        assert len(got.payload["data"]) == 65536

    def test_d094_get_nonexistent_id(self):
        """TC-D094: get() for non-existent ID returns None."""
        result = self.db.get("00000000-0000-0000-0000-999999999999")
        assert result is None

    def test_d095_delete_nonexistent_id_no_error(self):
        """TC-D095: delete() for non-existent ID does not raise."""
        self.db.delete(vector_id="00000000-0000-0000-0000-999999999999")


# ===========================================================================
# P2 - Observability & Diagnostics
# ===========================================================================


class TestDistributedObservability:
    """TC-D100 ~ TC-D102: Metrics and diagnostics in distributed mode."""

    def test_d100_config_snapshot_includes_distributed_fields(self):
        """TC-D100: col_info includes deployment_mode and distribution_mode."""
        db = _new_distributed_db()
        try:
            info = db.col_info()
            assert info["deployment_mode"] == "distributed"
            assert info["distribution_mode"] == "hash"
        finally:
            db.delete_col()

    def test_d101_capability_report_fields(self):
        """TC-D101: Capability report has expected fields for distributed mode."""
        db = _new_distributed_db(enable_capability_probe=True)
        try:
            report = db.capabilities
            assert report is not None
            assert hasattr(report, "vector_enabled")
            assert hasattr(report, "floatvector")
            assert report.deployment_mode == "distributed"
            assert report.distribution_mode == "hash"
        finally:
            db.delete_col()

    def test_d102_health_check(self):
        """TC-D102: Connectivity check passes in distributed mode."""
        db = _new_distributed_db()
        try:
            with db._get_cursor() as cur:
                cur.execute("SELECT 1")
                row = cur.fetchone()
                assert row[0] == 1
        finally:
            db.delete_col()


# ===========================================================================
# P2 - Upper-Layer Integration (Memory API)
# ===========================================================================


class TestDistributedMemoryIntegration:
    """TC-D110 ~ TC-D112: mem0 Memory API with distributed GaussDB backend."""

    @contextmanager
    def _make_memory(self, collection_name: str):
        """Create a Memory instance backed by distributed GaussDB.

        Yields the Memory object with patches active so that extract_entities,
        capture_event, and telemetry remain mocked during test execution.
        """
        from mem0 import Memory

        vector_config = _gaussdb_distributed_config(collection_name)
        assert vector_config is not None

        memory_config = {
            "vector_store": {"provider": "gaussdb", "config": vector_config},
            "embedder": {"provider": "openai", "config": {"model": "fake", "api_key": "fake"}},
            "llm": {"provider": "openai", "config": {"model": "fake", "api_key": "fake"}},
            "version": "v1.1",
        }

        class FakeEmbedder:
            def embed(self, text, memory_action=None):
                normalized = str(text).lower()
                if "hotpot" in normalized or "fire" in normalized:
                    return [0.9, 0.1, 0.0, 0.0]
                if "coffee" in normalized:
                    return [0.1, 0.9, 0.0, 0.0]
                if "travel" in normalized or "flight" in normalized:
                    return [0.0, 0.1, 0.9, 0.0]
                return [0.25, 0.25, 0.25, 0.25]

            def embed_batch(self, texts, memory_action="add"):
                return [self.embed(text, memory_action) for text in texts]

        with (
            patch("mem0.memory.main.EmbedderFactory.create", return_value=FakeEmbedder()),
            patch("mem0.memory.main.LlmFactory.create", return_value=MagicMock()),
            patch("mem0.memory.main.SQLiteManager", return_value=MagicMock()),
            patch("mem0.memory.main.extract_entities", return_value=[]),
            patch("mem0.memory.main.capture_event", lambda *args, **kwargs: None),
            patch("mem0.memory.main.MEM0_TELEMETRY", False),
        ):
            memory = Memory.from_config(memory_config)
            yield memory

    def test_d110_memory_add_search_delete(self):
        """TC-D110: Memory.add/search/delete works with distributed GaussDB."""
        collection = _new_collection("mem0_dist_memory")
        with self._make_memory(collection) as memory:
            try:
                added = memory.add(
                    "I love eating hotpot in winter",
                    user_id="alice",
                    infer=False,
                    metadata={"source": "distributed-test"},
                )
                memory_id = added["results"][0]["id"]

                search_result = memory.search("hotpot", filters={"user_id": "alice"}, top_k=5, threshold=0)
                rows = search_result["results"]
                assert len(rows) >= 1
                assert any(r["id"] == memory_id for r in rows)

                memory.delete(memory_id)
                assert memory.vector_store.get(memory_id) is None
            finally:
                memory.vector_store.delete_col()
                if getattr(memory, "_entity_store", None) is not None:
                    memory.entity_store.delete_col()

    def test_d111_memory_multi_user_isolation(self):
        """TC-D111: Different users are isolated in distributed mode."""
        collection = _new_collection("mem0_dist_iso")
        with self._make_memory(collection) as memory:
            try:
                memory.add("Alice likes hotpot", user_id="alice", infer=False)
                memory.add("Bob likes coffee", user_id="bob", infer=False)

                alice_results = memory.search("food", filters={"user_id": "alice"}, top_k=10, threshold=0)
                bob_results = memory.search("food", filters={"user_id": "bob"}, top_k=10, threshold=0)

                alice_ids = {r["id"] for r in alice_results["results"]}
                bob_ids = {r["id"] for r in bob_results["results"]}
                assert alice_ids.isdisjoint(bob_ids), "User isolation violated"
            finally:
                memory.vector_store.delete_col()
                if getattr(memory, "_entity_store", None) is not None:
                    memory.entity_store.delete_col()

    def test_d112_memory_update_in_distributed(self):
        """TC-D112: Memory.update works correctly in distributed mode."""
        collection = _new_collection("mem0_dist_upd")
        with self._make_memory(collection) as memory:
            try:
                added = memory.add("I like hotpot", user_id="alice", infer=False)
                memory_id = added["results"][0]["id"]

                memory.update(memory_id, "I love spicy hotpot")

                got = memory.vector_store.get(memory_id)
                assert got is not None
                assert "spicy" in got.payload.get("data", "")
            finally:
                memory.vector_store.delete_col()
                if getattr(memory, "_entity_store", None) is not None:
                    memory.entity_store.delete_col()


# ===========================================================================
# P2 - Cross-Mode Comparison
# ===========================================================================


class TestCrossModeComparison:
    """TC-D120 ~ TC-D121: Verify distributed and centralized produce same results."""

    @pytest.fixture()
    def both_dbs(self):
        """Create both a distributed and centralized instance with same data."""
        db_dist = _new_distributed_db(collection_name=_new_collection("dist_cmp"))
        config_cent = _gaussdb_distributed_config(
            _new_collection("cent_cmp"),
            deployment_mode="centralized",
            distribution_mode="auto",
        )
        db_cent = GaussDB(**config_cent)

        records = [
            (_uuid(1), VECTORS_4D["A"], {"data": "rec_A", "user_id": "u1", "agent_id": "a1", "run_id": "r1"}),
            (_uuid(2), VECTORS_4D["B"], {"data": "rec_B", "user_id": "u1", "agent_id": "a1", "run_id": "r1"}),
            (_uuid(3), VECTORS_4D["C"], {"data": "rec_C", "user_id": "u1", "agent_id": "a1", "run_id": "r1"}),
            (_uuid(4), VECTORS_4D["D"], {"data": "rec_D", "user_id": "u1", "agent_id": "a1", "run_id": "r1"}),
            (_uuid(5), VECTORS_4D["E"], {"data": "rec_E", "user_id": "u1", "agent_id": "a1", "run_id": "r1"}),
        ]
        _insert_records(db_dist, records)
        _insert_records(db_cent, records)

        yield db_dist, db_cent

        db_dist.delete_col()
        db_cent.delete_col()

    def test_d120_same_search_results(self, both_dbs):
        """TC-D120: Same data + same query produces same ordering in both modes."""
        db_dist, db_cent = both_dbs

        query_vec = VECTORS_4D["A"]
        results_dist = db_dist.search(
            query="test", vectors=query_vec, top_k=5, filters={"user_id": "u1"}
        )
        results_cent = db_cent.search(
            query="test", vectors=query_vec, top_k=5, filters={"user_id": "u1"}
        )

        ids_dist = _ids(results_dist)
        ids_cent = _ids(results_cent)
        assert ids_dist == ids_cent, f"Ordering mismatch: dist={ids_dist}, cent={ids_cent}"

        for rd, rc in zip(results_dist, results_cent):
            assert abs(rd.score - rc.score) < 1e-4, (
                f"Score mismatch for {rd.id}: dist={rd.score}, cent={rc.score}"
            )

    def test_d121_same_filter_results(self, both_dbs):
        """TC-D121: Same filter produces same result set in both modes."""
        db_dist, db_cent = both_dbs

        results_dist = db_dist.search(
            query="test", vectors=VECTORS_4D["B"], top_k=10, filters={"user_id": "u1"}
        )
        results_cent = db_cent.search(
            query="test", vectors=VECTORS_4D["B"], top_k=10, filters={"user_id": "u1"}
        )

        assert set(_ids(results_dist)) == set(_ids(results_cent))
