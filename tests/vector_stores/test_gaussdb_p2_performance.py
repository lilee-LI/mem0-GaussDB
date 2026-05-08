"""
P2 Performance Baseline Tests for GaussDB vector store.

Tests performance characteristics including:
- Single operation latency (insert, search, update, get, delete)
- Batch operation throughput
- Data size vs search latency relationship
- Filter search performance comparison

~17 tests total. All marked with @pytest.mark.p2 and @pytest.mark.slow.
"""

import time
import random
import uuid

import pytest

from tests.vector_stores.conftest import (
    EMBEDDING_DIMS,
    _env_bool,
    _measure_latency,
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


def _random_vector(dims: int = EMBEDDING_DIMS) -> list:
    """Generate a random vector of given dimensions."""
    return [random.random() for _ in range(dims)]


def _print_latency_report(name: str, stats: dict):
    """Print a formatted latency report for pytest -s output."""
    print(f"\n  [PERF] {name}:")
    print(f"    iterations: {stats['iterations']}")
    print(f"    P50:  {stats['p50_ms']:.2f} ms")
    print(f"    P99:  {stats['p99_ms']:.2f} ms")
    print(f"    min:  {stats['min_ms']:.2f} ms")
    print(f"    max:  {stats['max_ms']:.2f} ms")
    print(f"    mean: {stats['mean_ms']:.2f} ms")


def _soft_assert(condition: bool, message: str):
    """Soft assertion: prints a warning instead of failing the test."""
    if not condition:
        print(f"  [WARN] Soft assertion failed: {message}")


# ===========================================================================
# 8.2.1 Single Operation Latency Tests
# ===========================================================================


class TestSingleOperationLatency:
    """Measure latency of individual CRUD operations."""

    def test_single_insert_latency(self):
        """Insert 100 times, report P50/P95/P99 (target P50<100ms)."""
        db = _new_db(prefix="p2_perf")
        try:
            counter = [0]

            def _do_insert():
                counter[0] += 1
                vid = str(uuid.uuid4())
                db.insert(
                    ids=[vid],
                    vectors=[_random_vector()],
                    payloads=[_make_payload(f"insert_latency_{counter[0]}")],
                )

            stats = _measure_latency(_do_insert, iterations=100)
            _print_latency_report("single_insert", stats)
            _soft_assert(stats["p50_ms"] < 100, f"P50 insert latency {stats['p50_ms']:.2f}ms > 100ms target")
        finally:
            db.delete_col()

    def test_single_search_latency(self):
        """Search 100 times, report P50/P95/P99 (target P50<50ms)."""
        db = _new_db(prefix="p2_perf")
        try:
            # Pre-populate with some data
            ids = [str(uuid.uuid4()) for _ in range(50)]
            vectors = [_random_vector() for _ in range(50)]
            payloads = [_make_payload(f"search_data_{i}") for i in range(50)]
            db.insert(ids=ids, vectors=vectors, payloads=payloads)

            def _do_search():
                db.search("query", _random_vector(), top_k=5, filters={"user_id": "test_user"})

            stats = _measure_latency(_do_search, iterations=100)
            _print_latency_report("single_search", stats)
            _soft_assert(stats["p50_ms"] < 50, f"P50 search latency {stats['p50_ms']:.2f}ms > 50ms target")
        finally:
            db.delete_col()

    def test_single_update_latency(self):
        """Update 100 times, report P50/P95/P99 (target P50<150ms)."""
        db = _new_db(prefix="p2_perf")
        try:
            # Insert a record to update repeatedly
            vid = _uuid(9001)
            db.insert(
                ids=[vid],
                vectors=[_random_vector()],
                payloads=[_make_payload("update_target")],
            )
            counter = [0]

            def _do_update():
                counter[0] += 1
                db.update(
                    vid,
                    vector=_random_vector(),
                    payload=_make_payload(f"updated_{counter[0]}"),
                )

            stats = _measure_latency(_do_update, iterations=100)
            _print_latency_report("single_update", stats)
            _soft_assert(stats["p50_ms"] < 150, f"P50 update latency {stats['p50_ms']:.2f}ms > 150ms target")
        finally:
            db.delete_col()

    def test_single_get_latency(self):
        """Get 100 times, report P50/P95/P99 (target P50<10ms)."""
        db = _new_db(prefix="p2_perf")
        try:
            vid = _uuid(9002)
            db.insert(
                ids=[vid],
                vectors=[_random_vector()],
                payloads=[_make_payload("get_target")],
            )

            def _do_get():
                db.get(vid)

            stats = _measure_latency(_do_get, iterations=100)
            _print_latency_report("single_get", stats)
            _soft_assert(stats["p50_ms"] < 10, f"P50 get latency {stats['p50_ms']:.2f}ms > 10ms target")
        finally:
            db.delete_col()

    def test_single_delete_latency(self):
        """Delete 100 times, report P50/P95/P99 (target P50<50ms)."""
        db = _new_db(prefix="p2_perf")
        try:
            # Pre-insert 100 records to delete one by one
            ids_to_delete = [str(uuid.uuid4()) for _ in range(100)]
            vectors = [_random_vector() for _ in range(100)]
            payloads = [_make_payload(f"delete_target_{i}") for i in range(100)]
            db.insert(ids=ids_to_delete, vectors=vectors, payloads=payloads)

            idx = [0]

            def _do_delete():
                db.delete(vector_id=ids_to_delete[idx[0]])
                idx[0] += 1

            stats = _measure_latency(_do_delete, iterations=100)
            _print_latency_report("single_delete", stats)
            _soft_assert(stats["p50_ms"] < 50, f"P50 delete latency {stats['p50_ms']:.2f}ms > 50ms target")
        finally:
            db.delete_col()


# ===========================================================================
# 8.2.2 Batch Operation Throughput Tests
# ===========================================================================


class TestBatchOperationThroughput:
    """Measure throughput of batch operations."""

    def test_batch_insert_100_throughput(self):
        """Insert 100 records in one batch, measure throughput (target >50 records/s)."""
        db = _new_db(prefix="p2_perf")
        try:
            count = 100
            ids = [str(uuid.uuid4()) for _ in range(count)]
            vectors = [_random_vector() for _ in range(count)]
            payloads = [_make_payload(f"batch100_{i}") for i in range(count)]

            start = time.perf_counter()
            db.insert(ids=ids, vectors=vectors, payloads=payloads)
            elapsed = time.perf_counter() - start

            throughput = count / elapsed
            print(f"\n  [PERF] batch_insert_100: {elapsed:.3f}s, throughput: {throughput:.1f} records/s")
            _soft_assert(throughput > 50, f"Batch insert 100 throughput {throughput:.1f} < 50 records/s target")
        finally:
            db.delete_col()

    def test_batch_insert_1000_throughput(self):
        """Insert 1000 records in one batch, measure throughput (target >100 records/s)."""
        db = _new_db(prefix="p2_perf")
        try:
            count = 1000
            ids = [str(uuid.uuid4()) for _ in range(count)]
            vectors = [_random_vector() for _ in range(count)]
            payloads = [_make_payload(f"batch1k_{i}") for i in range(count)]

            start = time.perf_counter()
            db.insert(ids=ids, vectors=vectors, payloads=payloads)
            elapsed = time.perf_counter() - start

            throughput = count / elapsed
            print(f"\n  [PERF] batch_insert_1000: {elapsed:.3f}s, throughput: {throughput:.1f} records/s")
            _soft_assert(throughput > 100, f"Batch insert 1000 throughput {throughput:.1f} < 100 records/s target")
        finally:
            db.delete_col()

    @pytest.mark.high_pressure
    def test_batch_insert_10000_throughput(self):
        """Insert 10000 records in one batch, measure throughput (target >200 records/s)."""
        db = _new_db(prefix="p2_perf")
        try:
            count = 10000
            ids = [str(uuid.uuid4()) for _ in range(count)]
            vectors = [_random_vector() for _ in range(count)]
            payloads = [_make_payload(f"batch10k_{i}") for i in range(count)]

            start = time.perf_counter()
            db.insert(ids=ids, vectors=vectors, payloads=payloads)
            elapsed = time.perf_counter() - start

            throughput = count / elapsed
            print(f"\n  [PERF] batch_insert_10000: {elapsed:.3f}s, throughput: {throughput:.1f} records/s")
            _soft_assert(throughput > 200, f"Batch insert 10000 throughput {throughput:.1f} < 200 records/s target")
        finally:
            db.delete_col()

    def test_batch_search_10_throughput(self):
        """10 sequential searches, measure throughput."""
        db = _new_db(prefix="p2_perf")
        try:
            # Pre-populate
            count = 200
            ids = [str(uuid.uuid4()) for _ in range(count)]
            vectors = [_random_vector() for _ in range(count)]
            payloads = [_make_payload(f"search_data_{i}") for i in range(count)]
            db.insert(ids=ids, vectors=vectors, payloads=payloads)

            num_searches = 10
            start = time.perf_counter()
            for _ in range(num_searches):
                db.search("query", _random_vector(), top_k=10, filters={"user_id": "test_user"})
            elapsed = time.perf_counter() - start

            throughput = num_searches / elapsed
            print(f"\n  [PERF] batch_search_10: {elapsed:.3f}s, throughput: {throughput:.1f} searches/s")
        finally:
            db.delete_col()

    def test_batch_search_100_throughput(self):
        """100 sequential searches, measure throughput."""
        db = _new_db(prefix="p2_perf")
        try:
            # Pre-populate
            count = 200
            ids = [str(uuid.uuid4()) for _ in range(count)]
            vectors = [_random_vector() for _ in range(count)]
            payloads = [_make_payload(f"search_data_{i}") for i in range(count)]
            db.insert(ids=ids, vectors=vectors, payloads=payloads)

            num_searches = 100
            start = time.perf_counter()
            for _ in range(num_searches):
                db.search("query", _random_vector(), top_k=10, filters={"user_id": "test_user"})
            elapsed = time.perf_counter() - start

            throughput = num_searches / elapsed
            print(f"\n  [PERF] batch_search_100: {elapsed:.3f}s, throughput: {throughput:.1f} searches/s")
        finally:
            db.delete_col()


# ===========================================================================
# 8.2.3 Data Size vs Search Latency Tests
# ===========================================================================


class TestDataSizeVsSearchLatency:
    """Measure how search latency scales with data size."""

    def _populate_and_measure(self, db, record_count: int, label: str):
        """Insert record_count records and measure search latency."""
        batch_size = 1000
        inserted = 0
        while inserted < record_count:
            batch = min(batch_size, record_count - inserted)
            ids = [str(uuid.uuid4()) for _ in range(batch)]
            vectors = [_random_vector() for _ in range(batch)]
            payloads = [_make_payload(f"data_{inserted + i}") for i in range(batch)]
            db.insert(ids=ids, vectors=vectors, payloads=payloads)
            inserted += batch

        def _do_search():
            db.search("query", _random_vector(), top_k=10, filters={"user_id": "test_user"})

        stats = _measure_latency(_do_search, iterations=50)
        _print_latency_report(f"search_latency_{label} (n={record_count})", stats)
        return stats

    def test_search_latency_vs_data_size_100(self):
        """Search latency with 100 records."""
        db = _new_db(prefix="p2_perf")
        try:
            stats = self._populate_and_measure(db, 100, "100")
            _soft_assert(stats["p50_ms"] < 100, f"P50 search@100 = {stats['p50_ms']:.2f}ms")
        finally:
            db.delete_col()

    def test_search_latency_vs_data_size_1000(self):
        """Search latency with 1000 records."""
        db = _new_db(prefix="p2_perf")
        try:
            stats = self._populate_and_measure(db, 1000, "1000")
            _soft_assert(stats["p50_ms"] < 200, f"P50 search@1000 = {stats['p50_ms']:.2f}ms")
        finally:
            db.delete_col()

    @pytest.mark.high_pressure
    def test_search_latency_vs_data_size_10000(self):
        """Search latency with 10000 records."""
        db = _new_db(prefix="p2_perf")
        try:
            stats = self._populate_and_measure(db, 10000, "10000")
            _soft_assert(stats["p50_ms"] < 500, f"P50 search@10000 = {stats['p50_ms']:.2f}ms")
        finally:
            db.delete_col()

    @pytest.mark.high_pressure
    def test_search_latency_vs_data_size_100000(self):
        """Search latency with 100000 records. Skipped if env not configured for high pressure."""
        if not _env_bool("GAUSSDB_TEST_RUN_HIGH_PRESSURE"):
            pytest.skip("Set GAUSSDB_TEST_RUN_HIGH_PRESSURE=true to run 100k record tests")
        db = _new_db(prefix="p2_perf")
        try:
            stats = self._populate_and_measure(db, 100000, "100000")
            _soft_assert(stats["p50_ms"] < 2000, f"P50 search@100000 = {stats['p50_ms']:.2f}ms")
        finally:
            db.delete_col()


# ===========================================================================
# 8.2.4 Filter Search Performance Tests
# ===========================================================================


class TestFilterSearchPerformance:
    """Compare search performance with and without filters."""

    def test_search_with_filter_vs_without_filter(self):
        """Compare latency of filtered vs unfiltered search."""
        db = _new_db(prefix="p2_perf")
        try:
            # Populate with mixed user_ids
            count = 500
            ids = [str(uuid.uuid4()) for _ in range(count)]
            vectors = [_random_vector() for _ in range(count)]
            payloads = [
                _make_payload(f"item_{i}", user_id=f"user_{i % 10}")
                for i in range(count)
            ]
            db.insert(ids=ids, vectors=vectors, payloads=payloads)

            def _search_no_filter():
                db.search("query", _random_vector(), top_k=10, filters={"user_id": "user_0"})

            def _search_with_filter():
                db.search(
                    "query", _random_vector(), top_k=10,
                    filters={"user_id": "user_0", "data": "item_0"},
                )

            stats_no_filter = _measure_latency(_search_no_filter, iterations=50)
            stats_with_filter = _measure_latency(_search_with_filter, iterations=50)

            _print_latency_report("search_single_filter", stats_no_filter)
            _print_latency_report("search_multi_filter", stats_with_filter)

            ratio = stats_with_filter["p50_ms"] / max(stats_no_filter["p50_ms"], 0.01)
            print(f"\n  [PERF] Filter overhead ratio (multi/single): {ratio:.2f}x")
        finally:
            db.delete_col()

    def test_search_filter_simple_vs_complex(self):
        """Compare simple filter vs complex nested filter performance."""
        db = _new_db(prefix="p2_perf")
        try:
            # Populate with varied metadata
            count = 500
            ids = [str(uuid.uuid4()) for _ in range(count)]
            vectors = [_random_vector() for _ in range(count)]
            payloads = [
                _make_payload(
                    f"item_{i}",
                    user_id=f"user_{i % 10}",
                    category=f"cat_{i % 5}",
                    priority=i % 3,
                )
                for i in range(count)
            ]
            db.insert(ids=ids, vectors=vectors, payloads=payloads)

            # Simple filter: single field
            def _search_simple():
                db.search("query", _random_vector(), top_k=10, filters={"user_id": "user_3"})

            # Complex filter: multiple fields combined
            def _search_complex():
                db.search(
                    "query", _random_vector(), top_k=10,
                    filters={"user_id": "user_3", "category": "cat_2", "priority": 1},
                )

            stats_simple = _measure_latency(_search_simple, iterations=50)
            stats_complex = _measure_latency(_search_complex, iterations=50)

            _print_latency_report("search_simple_filter", stats_simple)
            _print_latency_report("search_complex_filter", stats_complex)

            ratio = stats_complex["p50_ms"] / max(stats_simple["p50_ms"], 0.01)
            print(f"\n  [PERF] Complex/Simple filter ratio: {ratio:.2f}x")
        finally:
            db.delete_col()

    def test_search_filter_json_expression_vs_redundant_columns(self):
        """Compare JSON expression filter vs redundant column filter performance."""
        db = _new_db(prefix="p2_perf")
        try:
            # Populate data with metadata stored in payload
            count = 500
            ids = [str(uuid.uuid4()) for _ in range(count)]
            vectors = [_random_vector() for _ in range(count)]
            payloads = [
                _make_payload(
                    f"item_{i}",
                    user_id=f"user_{i % 10}",
                    region=f"region_{i % 4}",
                    score=float(i % 100),
                )
                for i in range(count)
            ]
            db.insert(ids=ids, vectors=vectors, payloads=payloads)

            # Filter by user_id (typically a redundant/indexed column)
            def _search_by_user_id():
                db.search("query", _random_vector(), top_k=10, filters={"user_id": "user_5"})

            # Filter by a JSON payload field (region) with required scope
            def _search_by_json_field():
                db.search("query", _random_vector(), top_k=10, filters={"user_id": "user_5", "region": "region_2"})

            stats_user_id = _measure_latency(_search_by_user_id, iterations=50)
            stats_json = _measure_latency(_search_by_json_field, iterations=50)

            _print_latency_report("search_filter_user_id_column", stats_user_id)
            _print_latency_report("search_filter_json_field", stats_json)

            ratio = stats_json["p50_ms"] / max(stats_user_id["p50_ms"], 0.01)
            print(f"\n  [PERF] JSON filter / column filter ratio: {ratio:.2f}x")
        finally:
            db.delete_col()
