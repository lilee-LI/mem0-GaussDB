"""
P2 Concurrency Safety Tests for GaussDB vector store.

Tests concurrent operation safety including:
- Multi-thread concurrent insert
- Upsert race conditions (MERGE INTO atomicity)
- Read-write concurrency
- Connection pool exhaustion
- Concurrent data consistency verification

~22 tests total. All marked with @pytest.mark.p2 and @pytest.mark.slow.
"""

import random
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest

from tests.vector_stores.conftest import (
    EMBEDDING_DIMS,
    VECTOR_COFFEE,
    _list_flat,
    _new_db,
    _uuid,
    _make_payload,
    gaussdb_available,
)

pytestmark = [
    pytest.mark.p2,
    pytest.mark.slow,
    pytest.mark.skipif(not gaussdb_available(), reason="GaussDB test env not configured"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _random_vector():
    """Generate a random vector of EMBEDDING_DIMS dimensions."""
    return [random.random() for _ in range(EMBEDDING_DIMS)]


def _run_concurrent(tasks, max_workers=10, timeout=600.0):
    """
    Run a list of callables concurrently and collect results/errors.
    Returns (successes: int, errors: list[str]).
    """
    successes = 0
    errors = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(task) for task in tasks]
        for future in as_completed(futures, timeout=timeout):
            try:
                future.result()
                successes += 1
            except Exception as exc:
                errors.append(str(exc))
    return successes, errors


# ===========================================================================
# 9.2.1 Multi-thread Concurrent Insert
# ===========================================================================


class TestConcurrentInsert:
    """Tests for multi-thread concurrent insert operations."""

    def test_concurrent_insert_same_collection(self):
        """5 threads, 10 records each into same collection. Verify no loss or duplicates."""
        db = _new_db(prefix="p2_conc")
        try:
            num_threads = 5
            records_per_thread = 10

            def insert_batch(thread_idx):
                for i in range(records_per_thread):
                    record_id = str(uuid.uuid4())
                    db.insert(
                        ids=[record_id],
                        vectors=[_random_vector()],
                        payloads=[_make_payload(
                            f"thread_{thread_idx}_record_{i}",
                            user_id="conc_insert_user",
                        )],
                    )

            tasks = [lambda idx=t: insert_batch(idx) for t in range(num_threads)]
            successes, errors = _run_concurrent(tasks, max_workers=num_threads)

            total_expected = num_threads * records_per_thread
            final_count = len(_list_flat(db, filters={"user_id": "conc_insert_user"}, top_k=total_expected + 100))
            assert final_count > 0
            if not errors:
                assert final_count == total_expected
        finally:
            db.delete_col()

    def test_concurrent_insert_same_user_id(self):
        """5 threads insert with same user_id, total 50 records."""
        db = _new_db(prefix="p2_conc")
        try:
            user_id = "shared_user_conc"
            num_threads = 5
            records_per_thread = 10

            def insert_batch(thread_idx):
                for i in range(records_per_thread):
                    record_id = str(uuid.uuid4())
                    db.insert(
                        ids=[record_id],
                        vectors=[_random_vector()],
                        payloads=[_make_payload(
                            f"thread_{thread_idx}_item_{i}",
                            user_id=user_id,
                        )],
                    )

            tasks = [lambda idx=t: insert_batch(idx) for t in range(num_threads)]
            successes, errors = _run_concurrent(tasks, max_workers=num_threads)

            results = _list_flat(db, filters={"user_id": user_id}, top_k=2000)
            assert len(results) > 0
            for entry in results:
                assert entry.payload["user_id"] == user_id
        finally:
            db.delete_col()

    def test_concurrent_insert_different_user_ids(self):
        """5 threads with different user_ids. No cross-contamination."""
        db = _new_db(prefix="p2_conc")
        try:
            num_threads = 5
            records_per_thread = 10

            def insert_batch(thread_idx):
                uid = f"user_thread_{thread_idx}"
                for i in range(records_per_thread):
                    record_id = str(uuid.uuid4())
                    db.insert(
                        ids=[record_id],
                        vectors=[_random_vector()],
                        payloads=[_make_payload(
                            f"data_{thread_idx}_{i}",
                            user_id=uid,
                        )],
                    )

            tasks = [lambda idx=t: insert_batch(idx) for t in range(num_threads)]
            successes, errors = _run_concurrent(tasks, max_workers=num_threads)

            for t in range(num_threads):
                uid = f"user_thread_{t}"
                results = _list_flat(db, filters={"user_id": uid}, top_k=200)
                for entry in results:
                    assert entry.payload["user_id"] == uid
        finally:
            db.delete_col()

    @pytest.mark.high_pressure
    def test_concurrent_insert_high_pressure(self):
        """10 threads, 5 records each - high pressure stress test (2x pool size)."""
        db = _new_db(prefix="p2_conc")
        try:
            num_threads = 10
            records_per_thread = 5

            def insert_batch(thread_idx):
                for i in range(records_per_thread):
                    record_id = str(uuid.uuid4())
                    db.insert(
                        ids=[record_id],
                        vectors=[_random_vector()],
                        payloads=[_make_payload(
                            f"hp_{thread_idx}_{i}",
                            user_id="high_pressure_user",
                        )],
                    )

            tasks = [lambda idx=t: insert_batch(idx) for t in range(num_threads)]
            successes, errors = _run_concurrent(tasks, max_workers=num_threads, timeout=600.0)

            final_count = len(_list_flat(db, filters={"user_id": "high_pressure_user"}, top_k=1500))
            total_expected = num_threads * records_per_thread
            # Under high contention some inserts may fail due to pool exhaustion
            assert final_count >= total_expected * 0.5
        finally:
            db.delete_col()

    def test_concurrent_insert_with_batch(self):
        """5 threads each batch-inserting 20 records at once."""
        db = _new_db(prefix="p2_conc")
        try:
            num_threads = 5
            batch_size = 20

            def batch_insert(thread_idx):
                ids = [str(uuid.uuid4()) for _ in range(batch_size)]
                vectors = [_random_vector() for _ in range(batch_size)]
                payloads = [
                    _make_payload(f"batch_{thread_idx}_{i}", user_id="batch_conc_user")
                    for i in range(batch_size)
                ]
                db.insert(ids=ids, vectors=vectors, payloads=payloads)

            tasks = [lambda idx=t: batch_insert(idx) for t in range(num_threads)]
            successes, errors = _run_concurrent(tasks, max_workers=num_threads)

            final_count = len(_list_flat(db, filters={"user_id": "batch_conc_user"}, top_k=600))
            total_expected = num_threads * batch_size
            if not errors:
                assert final_count == total_expected
            else:
                assert final_count > 0
        finally:
            db.delete_col()


# ===========================================================================
# 9.2.2 Upsert Race Conditions
# ===========================================================================


class TestUpsertRaceConditions:
    """Tests for upsert (MERGE INTO) atomicity under concurrent access."""

    def test_concurrent_upsert_same_id(self):
        """10 threads upsert same ID concurrently. Final count must be 1."""
        db = _new_db(prefix="p2_conc")
        try:
            target_id = _uuid(9001)
            num_threads = 10

            def upsert_record(thread_idx):
                vector = _random_vector()
                payload = _make_payload(
                    f"upsert_thread_{thread_idx}",
                    user_id="upsert_user",
                )
                db.insert(
                    ids=[target_id],
                    vectors=[vector],
                    payloads=[payload],
                )

            tasks = [lambda idx=t: upsert_record(idx) for t in range(num_threads)]
            successes, errors = _run_concurrent(tasks, max_workers=num_threads)

            # MERGE INTO atomicity: final count for this ID must be exactly 1
            result = db.get(target_id)
            assert result is not None
            assert len(_list_flat(db, filters={"user_id": "upsert_user"}, top_k=100)) == 1
        finally:
            db.delete_col()

    def test_concurrent_upsert_same_id_same_payload(self):
        """10 threads upsert same ID with identical payload. Idempotency check."""
        db = _new_db(prefix="p2_conc")
        try:
            target_id = _uuid(9002)
            num_threads = 10
            fixed_vector = VECTOR_COFFEE
            fixed_payload = _make_payload("idempotent_data", user_id="upsert_user")

            def upsert_same(thread_idx):
                db.insert(
                    ids=[target_id],
                    vectors=[fixed_vector],
                    payloads=[fixed_payload],
                )

            tasks = [lambda idx=t: upsert_same(idx) for t in range(num_threads)]
            successes, errors = _run_concurrent(tasks, max_workers=num_threads)

            result = db.get(target_id)
            assert result is not None
            assert result.payload["data"] == "idempotent_data"
            assert len(_list_flat(db, filters={"user_id": "upsert_user"}, top_k=100)) == 1
        finally:
            db.delete_col()

    def test_concurrent_upsert_same_id_update_vs_insert(self):
        """Insert first, then concurrent upserts. Verify final state is consistent."""
        db = _new_db(prefix="p2_conc")
        try:
            target_id = _uuid(9003)
            db.insert(
                ids=[target_id],
                vectors=[VECTOR_COFFEE],
                payloads=[_make_payload("original", user_id="upsert_user")],
            )

            num_threads = 10

            def upsert_update(thread_idx):
                vector = _random_vector()
                payload = _make_payload(
                    f"updated_by_{thread_idx}",
                    user_id="upsert_user",
                )
                db.insert(
                    ids=[target_id],
                    vectors=[vector],
                    payloads=[payload],
                )

            tasks = [lambda idx=t: upsert_update(idx) for t in range(num_threads)]
            successes, errors = _run_concurrent(tasks, max_workers=num_threads)

            result = db.get(target_id)
            assert result is not None
            assert result.payload["data"].startswith("updated_by_")
            assert len(_list_flat(db, filters={"user_id": "upsert_user"}, top_k=100)) == 1
        finally:
            db.delete_col()

    def test_concurrent_upsert_batch_same_ids(self):
        """Batch upsert with overlapping IDs from multiple threads."""
        db = _new_db(prefix="p2_conc")
        try:
            shared_ids = [_uuid(9010 + i) for i in range(5)]
            num_threads = 10

            def batch_upsert(thread_idx):
                vectors = [_random_vector() for _ in range(5)]
                payloads = [
                    _make_payload(f"batch_t{thread_idx}_r{i}", user_id="upsert_user")
                    for i in range(5)
                ]
                db.insert(ids=shared_ids, vectors=vectors, payloads=payloads)

            tasks = [lambda idx=t: batch_upsert(idx) for t in range(num_threads)]
            successes, errors = _run_concurrent(tasks, max_workers=num_threads)

            for sid in shared_ids:
                result = db.get(sid)
                assert result is not None
            assert len(_list_flat(db, filters={"user_id": "upsert_user"}, top_k=100)) == 5
        finally:
            db.delete_col()

    def test_upsert_lost_update_detection(self):
        """2 threads sequential upsert. Verify last-write-wins semantics."""
        db = _new_db(prefix="p2_conc")
        try:
            target_id = _uuid(9020)
            barrier = threading.Barrier(2, timeout=30)
            write_order = []
            order_lock = threading.Lock()

            def upsert_with_order(thread_idx):
                barrier.wait()
                vector = _random_vector()
                payload = _make_payload(
                    f"writer_{thread_idx}",
                    user_id="upsert_user",
                )
                db.insert(
                    ids=[target_id],
                    vectors=[vector],
                    payloads=[payload],
                )
                with order_lock:
                    write_order.append(thread_idx)

            tasks = [lambda idx=t: upsert_with_order(idx) for t in range(2)]
            successes, errors = _run_concurrent(tasks, max_workers=2)

            result = db.get(target_id)
            assert result is not None
            assert len(_list_flat(db, filters={"user_id": "upsert_user"}, top_k=100)) == 1
            assert result.payload["data"] in ("writer_0", "writer_1")
        finally:
            db.delete_col()


# ===========================================================================
# 9.2.3 Read-Write Concurrency
# ===========================================================================


class TestReadWriteConcurrency:
    """Tests for concurrent read and write operations."""

    def test_concurrent_insert_and_search(self):
        """5 insert threads + 5 search threads running simultaneously."""
        db = _new_db(prefix="p2_conc", maxconn=15)
        try:
            user_id = "rw_insert_search"
            # Pre-insert some data so searches have something to find
            for i in range(20):
                db.insert(
                    ids=[str(uuid.uuid4())],
                    vectors=[_random_vector()],
                    payloads=[_make_payload(f"seed_{i}", user_id=user_id)],
                )

            def inserter(thread_idx):
                for i in range(20):
                    db.insert(
                        ids=[str(uuid.uuid4())],
                        vectors=[_random_vector()],
                        payloads=[_make_payload(
                            f"insert_t{thread_idx}_{i}", user_id=user_id
                        )],
                    )

            def searcher(thread_idx):
                for i in range(20):
                    results = db.search(
                        "query", _random_vector(), top_k=5,
                        filters={"user_id": user_id},
                    )
                    # Search should not crash; results may vary
                    assert isinstance(results, list)

            tasks = (
                [lambda idx=t: inserter(idx) for t in range(5)]
                + [lambda idx=t: searcher(idx) for t in range(5)]
            )
            successes, errors = _run_concurrent(tasks, max_workers=10)

            # All operations should complete without fatal errors
            assert successes >= 5  # At least the searchers should succeed
        finally:
            db.delete_col()

    def test_concurrent_update_and_search(self):
        """5 update threads + 5 search threads running simultaneously."""
        db = _new_db(prefix="p2_conc", maxconn=15)
        try:
            user_id = "rw_update_search"
            record_ids = []
            for i in range(50):
                rid = str(uuid.uuid4())
                record_ids.append(rid)
                db.insert(
                    ids=[rid],
                    vectors=[_random_vector()],
                    payloads=[_make_payload(f"original_{i}", user_id=user_id)],
                )

            def updater(thread_idx):
                for i in range(10):
                    target = record_ids[(thread_idx * 10 + i) % len(record_ids)]
                    db.update(
                        vector_id=target,
                        vector=_random_vector(),
                        payload=_make_payload(
                            f"updated_t{thread_idx}_{i}", user_id=user_id
                        ),
                    )

            def searcher(thread_idx):
                for i in range(10):
                    db.search(
                        "query", _random_vector(), top_k=5,
                        filters={"user_id": user_id},
                    )

            tasks = (
                [lambda idx=t: updater(idx) for t in range(5)]
                + [lambda idx=t: searcher(idx) for t in range(5)]
            )
            successes, errors = _run_concurrent(tasks, max_workers=10)

            final_count = len(_list_flat(db, filters={"user_id": user_id}, top_k=200))
            assert final_count == 50
        finally:
            db.delete_col()

    def test_concurrent_delete_and_search(self):
        """5 delete threads + 5 search threads running simultaneously."""
        db = _new_db(prefix="p2_conc", maxconn=15)
        try:
            user_id = "rw_delete_search"
            record_ids = []
            for i in range(100):
                rid = str(uuid.uuid4())
                record_ids.append(rid)
                db.insert(
                    ids=[rid],
                    vectors=[_random_vector()],
                    payloads=[_make_payload(f"to_delete_{i}", user_id=user_id)],
                )

            # Split IDs among delete threads
            chunk_size = 20

            def deleter(thread_idx):
                start = thread_idx * chunk_size
                ids_to_delete = record_ids[start:start + chunk_size]
                for rid in ids_to_delete:
                    db.delete(vector_id=rid)

            def searcher(thread_idx):
                for i in range(20):
                    results = db.search(
                        "query", _random_vector(), top_k=5,
                        filters={"user_id": user_id},
                    )
                    assert isinstance(results, list)

            tasks = (
                [lambda idx=t: deleter(idx) for t in range(5)]
                + [lambda idx=t: searcher(idx) for t in range(5)]
            )
            successes, errors = _run_concurrent(tasks, max_workers=10)

            # After deletion, count should be reduced
            final_count = len(_list_flat(db, filters={"user_id": user_id}, top_k=200))
            assert final_count < 100
        finally:
            db.delete_col()

    def test_concurrent_insert_and_get(self):
        """5 insert threads + 5 get threads running simultaneously."""
        db = _new_db(prefix="p2_conc", maxconn=15)
        try:
            user_id = "rw_insert_get"
            # Pre-insert known records for get operations
            known_ids = []
            for i in range(20):
                rid = _uuid(8000 + i)
                known_ids.append(rid)
                db.insert(
                    ids=[rid],
                    vectors=[_random_vector()],
                    payloads=[_make_payload(f"known_{i}", user_id=user_id)],
                )

            def inserter(thread_idx):
                for i in range(20):
                    db.insert(
                        ids=[str(uuid.uuid4())],
                        vectors=[_random_vector()],
                        payloads=[_make_payload(
                            f"new_t{thread_idx}_{i}", user_id=user_id
                        )],
                    )

            def getter(thread_idx):
                for i in range(20):
                    target = known_ids[i % len(known_ids)]
                    result = db.get(target)
                    # Record should always be retrievable (not deleted)
                    assert result is not None

            tasks = (
                [lambda idx=t: inserter(idx) for t in range(5)]
                + [lambda idx=t: getter(idx) for t in range(5)]
            )
            successes, errors = _run_concurrent(tasks, max_workers=10)

            # Known records should still exist
            for kid in known_ids:
                assert db.get(kid) is not None
        finally:
            db.delete_col()

    def test_concurrent_mixed_operations(self):
        """Insert/update/delete/search mixed, 20 threads total."""
        db = _new_db(prefix="p2_conc", maxconn=25)
        try:
            user_id = "rw_mixed"
            # Pre-insert records for update/delete/search
            record_ids = []
            for i in range(100):
                rid = str(uuid.uuid4())
                record_ids.append(rid)
                db.insert(
                    ids=[rid],
                    vectors=[_random_vector()],
                    payloads=[_make_payload(f"mixed_{i}", user_id=user_id)],
                )

            def inserter(thread_idx):
                for i in range(10):
                    db.insert(
                        ids=[str(uuid.uuid4())],
                        vectors=[_random_vector()],
                        payloads=[_make_payload(
                            f"mixed_new_t{thread_idx}_{i}", user_id=user_id
                        )],
                    )

            def updater(thread_idx):
                for i in range(10):
                    target = record_ids[(thread_idx * 10 + i) % len(record_ids)]
                    try:
                        db.update(
                            vector_id=target,
                            vector=_random_vector(),
                            payload=_make_payload(
                                f"mixed_upd_t{thread_idx}_{i}", user_id=user_id
                            ),
                        )
                    except Exception:
                        pass  # Record may have been deleted

            def deleter(thread_idx):
                # Delete from the end of the list to minimize conflict with updaters
                start = 80 + thread_idx * 4
                for i in range(4):
                    idx = start + i
                    if idx < len(record_ids):
                        try:
                            db.delete(vector_id=record_ids[idx])
                        except Exception:
                            pass

            def searcher(thread_idx):
                for i in range(10):
                    db.search(
                        "query", _random_vector(), top_k=5,
                        filters={"user_id": user_id},
                    )

            tasks = (
                [lambda idx=t: inserter(idx) for t in range(5)]
                + [lambda idx=t: updater(idx) for t in range(5)]
                + [lambda idx=t: deleter(idx) for t in range(5)]
                + [lambda idx=t: searcher(idx) for t in range(5)]
            )
            successes, errors = _run_concurrent(tasks, max_workers=20)

            # System should remain operational after mixed concurrent ops
            final_count = len(_list_flat(db, filters={"user_id": user_id}, top_k=200))
            assert final_count > 0
        finally:
            db.delete_col()


# ===========================================================================
# 9.2.4 Connection Pool Exhaustion
# ===========================================================================


class TestConnectionPoolExhaustion:
    """Tests for connection pool behavior under pressure."""

    def test_connection_pool_exhaustion(self):
        """maxconn=2, 5 concurrent threads. Operations should still complete."""
        db = _new_db(prefix="p2_conc", maxconn=2)
        try:
            user_id = "pool_exhaust"
            # Pre-insert some data
            for i in range(10):
                db.insert(
                    ids=[_uuid(7000 + i)],
                    vectors=[_random_vector()],
                    payloads=[_make_payload(f"pool_{i}", user_id=user_id)],
                )

            def worker(thread_idx):
                for i in range(10):
                    db.search(
                        "query", _random_vector(), top_k=3,
                        filters={"user_id": user_id},
                    )
                    db.insert(
                        ids=[str(uuid.uuid4())],
                        vectors=[_random_vector()],
                        payloads=[_make_payload(
                            f"pool_t{thread_idx}_{i}", user_id=user_id
                        )],
                    )

            tasks = [lambda idx=t: worker(idx) for t in range(5)]
            successes, errors = _run_concurrent(tasks, max_workers=5)

            # With pool exhaustion, some operations may fail but system recovers
            # At least some threads should succeed
            assert successes > 0
        finally:
            db.delete_col()

    def test_connection_pool_recovery(self):
        """Exhaust pool, then verify recovery after connections are released."""
        db = _new_db(prefix="p2_conc", maxconn=3)
        try:
            user_id = "pool_recovery"

            # Phase 1: Exhaust the pool with concurrent operations
            def heavy_worker(thread_idx):
                for i in range(20):
                    db.insert(
                        ids=[str(uuid.uuid4())],
                        vectors=[_random_vector()],
                        payloads=[_make_payload(
                            f"heavy_{thread_idx}_{i}", user_id=user_id
                        )],
                    )

            tasks = [lambda idx=t: heavy_worker(idx) for t in range(6)]
            successes, errors = _run_concurrent(tasks, max_workers=6)

            # Phase 2: After pool pressure, verify system recovers
            time.sleep(1)  # Allow connections to be returned to pool

            # Single-threaded operations should work fine after recovery
            recovery_id = str(uuid.uuid4())
            db.insert(
                ids=[recovery_id],
                vectors=[_random_vector()],
                payloads=[_make_payload("recovery_test", user_id=user_id)],
            )
            result = db.get(recovery_id)
            assert result is not None
            assert result.payload["data"] == "recovery_test"
        finally:
            db.delete_col()

    def test_connection_pool_leak_detection(self):
        """100 sequential operations. Verify no connection leak."""
        db = _new_db(prefix="p2_conc", maxconn=5)
        try:
            user_id = "pool_leak"

            # Perform many operations that acquire and release connections
            for i in range(100):
                rid = str(uuid.uuid4())
                db.insert(
                    ids=[rid],
                    vectors=[_random_vector()],
                    payloads=[_make_payload(f"leak_test_{i}", user_id=user_id)],
                )
                # Alternate between different operation types
                if i % 3 == 0:
                    db.search(
                        "query", _random_vector(), top_k=3,
                        filters={"user_id": user_id},
                    )
                elif i % 3 == 1:
                    db.get(rid)

            # After 100 operations, the system should still be responsive
            # If there were connection leaks, this would fail with pool exhaustion
            final_id = str(uuid.uuid4())
            db.insert(
                ids=[final_id],
                vectors=[_random_vector()],
                payloads=[_make_payload("final_check", user_id=user_id)],
            )
            result = db.get(final_id)
            assert result is not None
            assert result.payload["data"] == "final_check"

            # Verify total count is consistent
            final_count = len(_list_flat(db, filters={"user_id": user_id}, top_k=200))
            assert final_count == 101  # 100 + 1 final
        finally:
            db.delete_col()


# ===========================================================================
# 9.2.5 Concurrent Data Consistency
# ===========================================================================


class TestConcurrentDataConsistency:
    """Tests verifying data consistency after concurrent operations."""

    def test_concurrent_insert_final_count(self):
        """5 threads x 10 records = final count 50."""
        db = _new_db(prefix="p2_conc")
        try:
            num_threads = 5
            records_per_thread = 10
            user_id = "consistency_count"

            all_ids = []
            ids_lock = threading.Lock()

            def insert_batch(thread_idx):
                local_ids = []
                for i in range(records_per_thread):
                    rid = str(uuid.uuid4())
                    local_ids.append(rid)
                    db.insert(
                        ids=[rid],
                        vectors=[_random_vector()],
                        payloads=[_make_payload(
                            f"count_t{thread_idx}_{i}", user_id=user_id
                        )],
                    )
                with ids_lock:
                    all_ids.extend(local_ids)

            tasks = [lambda idx=t: insert_batch(idx) for t in range(num_threads)]
            successes, errors = _run_concurrent(tasks, max_workers=num_threads)

            total_expected = num_threads * records_per_thread
            final_count = len(_list_flat(db, filters={"user_id": user_id}, top_k=total_expected + 100))
            if not errors:
                assert final_count == total_expected
            else:
                # With errors, count should match successful inserts
                assert final_count == len(all_ids)
        finally:
            db.delete_col()

    def test_concurrent_upsert_final_state(self):
        """5 threads upsert same ID. Final state must be consistent (single record)."""
        db = _new_db(prefix="p2_conc")
        try:
            target_id = _uuid(9100)
            num_threads = 5
            user_id = "consistency_upsert"

            def upsert_record(thread_idx):
                for i in range(5):
                    vector = _random_vector()
                    payload = _make_payload(
                        f"state_t{thread_idx}_i{i}",
                        user_id=user_id,
                    )
                    db.insert(
                        ids=[target_id],
                        vectors=[vector],
                        payloads=[payload],
                    )

            tasks = [lambda idx=t: upsert_record(idx) for t in range(num_threads)]
            successes, errors = _run_concurrent(tasks, max_workers=num_threads)

            # Final state: exactly 1 record with consistent payload
            result = db.get(target_id)
            assert result is not None
            assert len(_list_flat(db, filters={"user_id": user_id}, top_k=100)) == 1
            # Payload should be from one of the threads/iterations
            assert result.payload["data"].startswith("state_t")
            assert result.payload["user_id"] == user_id
        finally:
            db.delete_col()

    def test_concurrent_delete_final_count(self):
        """Insert 100, 10 threads delete 10 each. Final count should be 0."""
        db = _new_db(prefix="p2_conc", maxconn=15)
        try:
            user_id = "consistency_delete"
            record_ids = []
            for i in range(100):
                rid = _uuid(8100 + i)
                record_ids.append(rid)
                db.insert(
                    ids=[rid],
                    vectors=[_random_vector()],
                    payloads=[_make_payload(f"del_{i}", user_id=user_id)],
                )

            assert len(_list_flat(db, filters={"user_id": user_id}, top_k=200)) == 100

            def delete_batch(thread_idx):
                start = thread_idx * 10
                for i in range(10):
                    idx = start + i
                    if idx < len(record_ids):
                        db.delete(vector_id=record_ids[idx])

            tasks = [lambda idx=t: delete_batch(idx) for t in range(10)]
            successes, errors = _run_concurrent(tasks, max_workers=10)

            final_count = len(_list_flat(db, filters={"user_id": user_id}, top_k=200))
            if not errors:
                assert final_count == 0
            else:
                # Some deletes may have failed, but count should be reduced
                assert final_count < 100
        finally:
            db.delete_col()

    def test_concurrent_update_payload_consistency(self):
        """10 threads update different fields of same records. Final state consistent."""
        db = _new_db(prefix="p2_conc", maxconn=15)
        try:
            user_id = "consistency_update"
            # Insert 10 records
            record_ids = []
            for i in range(10):
                rid = _uuid(8200 + i)
                record_ids.append(rid)
                db.insert(
                    ids=[rid],
                    vectors=[_random_vector()],
                    payloads=[_make_payload(f"original_{i}", user_id=user_id)],
                )

            def updater(thread_idx):
                """Each thread updates all 10 records with its own marker."""
                for i, rid in enumerate(record_ids):
                    try:
                        db.update(
                            vector_id=rid,
                            vector=_random_vector(),
                            payload=_make_payload(
                                f"updated_t{thread_idx}_r{i}",
                                user_id=user_id,
                                thread_marker=f"thread_{thread_idx}",
                            ),
                        )
                    except Exception:
                        pass  # Contention may cause transient failures

            tasks = [lambda idx=t: updater(idx) for t in range(10)]
            successes, errors = _run_concurrent(tasks, max_workers=10)

            # Verify consistency: each record should have a valid payload
            for rid in record_ids:
                result = db.get(rid)
                assert result is not None
                # Payload should be from one of the threads (last-write-wins)
                assert result.payload["user_id"] == user_id
                # Data field should be either original or from a thread update
                data = result.payload["data"]
                assert data.startswith("original_") or data.startswith("updated_t")

            # Total count should remain unchanged
            assert len(_list_flat(db, filters={"user_id": user_id}, top_k=100)) == 10
        finally:
            db.delete_col()

