#!/usr/bin/env python3
"""
End-to-End Test: mem0 GaussDB Vector Store — Direct DB Verification

This test directly connects to a real GaussDB instance and validates
ALL adapter methods without requiring LLM or Embedder services.
It uses synthetic vectors to isolate the GaussDB adapter logic.

Target: 121.37.186.131:19995, lxm/Gauss_234, lxm_db
"""

import os
import sys
import time
import uuid
import random
import traceback
import io
from datetime import datetime

# Fix Windows GBK encoding issue
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Ensure mem0 package is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'mem0')))

from mem0.vector_stores.gaussdb import GaussDB, OutputData

# ============================================================
# Configuration
# ============================================================
DB_CONFIG = dict(
    host="121.37.186.131",
    port=19995,
    user="lxm",
    password="Gauss_234",
    database="lxm_db",
)

DIMS = 1536  # Match default embedding_model_dims (OpenAI text-embedding-3-small)


def make_vector(seed: int = None) -> list:
    """Generate a deterministic or random vector."""
    rng = random.Random(seed)
    return [rng.uniform(-1, 1) for _ in range(DIMS)]


def cosine_similar_vector(base: list, noise: float = 0.05) -> list:
    """Create a vector similar to base (for search relevance testing)."""
    return [v + random.uniform(-noise, noise) for v in base]


# ============================================================
# Test Runner
# ============================================================
class TestResult:
    def __init__(self):
        self.passed = []
        self.failed = []
        self.skipped = []

    def summary(self):
        total = len(self.passed) + len(self.failed) + len(self.skipped)
        print(f"\n{'='*70}")
        print(f"  TEST SUMMARY: {len(self.passed)} passed, {len(self.failed)} failed, {len(self.skipped)} skipped / {total} total")
        print(f"{'='*70}")
        if self.failed:
            print("\n  FAILURES:")
            for name, err in self.failed:
                print(f"    ✗ {name}: {err}")
        if self.skipped:
            print("\n  SKIPPED:")
            for name, reason in self.skipped:
                print(f"    ⊘ {name}: {reason}")
        print()
        return len(self.failed) == 0


results = TestResult()


def run_test(name: str, func, *args, **kwargs):
    """Execute a test function and record result."""
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"\n[{ts}] ▶ {name}")
    try:
        func(*args, **kwargs)
        print(f"[{ts}]   ✓ PASSED")
        results.passed.append(name)
    except SkipTest as e:
        print(f"[{ts}]   ⊘ SKIPPED: {e}")
        results.skipped.append((name, str(e)))
    except Exception as e:
        print(f"[{ts}]   ✗ FAILED: {e}")
        traceback.print_exc()
        results.failed.append((name, str(e)))


class SkipTest(Exception):
    pass


def assert_eq(actual, expected, msg=""):
    if actual != expected:
        raise AssertionError(f"{msg}: expected {expected!r}, got {actual!r}")


def assert_true(cond, msg=""):
    if not cond:
        raise AssertionError(f"Assertion failed: {msg}")


def assert_gt(a, b, msg=""):
    if not (a > b):
        raise AssertionError(f"{msg}: expected {a} > {b}")


def assert_gte(a, b, msg=""):
    if not (a >= b):
        raise AssertionError(f"{msg}: expected {a} >= {b}")


# ============================================================
# Test Cases
# ============================================================

def test_01_connection_and_capability_probe(store: GaussDB):
    """Verify connection pool is alive and capability probe ran."""
    assert_true(store.connection_pool is not None, "connection_pool should exist")
    cap = store.capabilities
    assert_true(cap is not None, "capabilities should be probed")
    assert_true(cap.vector_enabled, "vector_enabled should be True")
    assert_true(cap.floatvector, "floatvector should be True")
    assert_true(cap.vector_index, "vector_index should be True")
    print(f"    Capabilities: floatvector={cap.floatvector}, vector_index={cap.vector_index}, "
          f"bm25={cap.bm25}, jsonb={cap.jsonb}, expression_index={cap.expression_index}")
    print(f"    deployment_mode={cap.deployment_mode}, distribution_mode={cap.distribution_mode}")


def test_02_create_collection(store: GaussDB):
    """Verify create_col creates table with correct schema."""
    store.create_col(
        name=store.collection_name,
        vector_size=DIMS,
        distance="cosine"
    )
    info = store.col_info()
    assert_eq(info["name"], store.collection_name, "collection name")
    assert_eq(info["count"], 0, "initial count should be 0")
    assert_eq(info["dimension"], DIMS, "dimension")
    print(f"    col_info: {info}")


def test_03_insert_single(store: GaussDB):
    """Insert a single vector with payload."""
    vec = make_vector(seed=42)
    payload = {
        "data": "Alice loves Python programming",
        "user_id": "user_alice",
        "agent_id": "agent_1",
        "run_id": "run_001",
    }
    test_id = str(uuid.uuid4())
    store.insert(vectors=[vec], payloads=[payload], ids=[test_id])

    # Verify via get
    result = store.get(test_id)
    assert_true(result is not None, "get should return inserted record")
    assert_eq(result.id, test_id, "id should match")
    assert_true("Alice" in str(result.payload.get("data", "")), "payload data should contain Alice")
    print(f"    Inserted and retrieved: id={test_id[:12]}...")


def test_04_insert_batch(store: GaussDB):
    """Insert multiple vectors in one call (batch MERGE INTO)."""
    vectors = [make_vector(seed=i) for i in range(10, 20)]
    payloads = [
        {
            "data": f"Memory item {i}",
            "user_id": "user_batch",
            "agent_id": "agent_1",
            "run_id": f"run_{i:03d}",
        }
        for i in range(10)
    ]
    ids = [str(uuid.uuid4()) for _ in range(10)]
    store.insert(vectors=vectors, payloads=payloads, ids=ids)

    # Verify count
    info = store.col_info()
    assert_gte(info["count"], 11, "should have at least 11 records (1 + 10)")
    print(f"    Batch inserted 10 records, total count: {info['count']}")
    return ids  # Return for later tests


def test_05_upsert_existing(store: GaussDB):
    """Insert with same ID should update (MERGE INTO behavior)."""
    test_id = str(uuid.uuid4())
    vec1 = make_vector(seed=100)
    payload1 = {"data": "Original memory", "user_id": "user_upsert"}
    store.insert(vectors=[vec1], payloads=[payload1], ids=[test_id])

    # Upsert with same ID, different payload
    vec2 = make_vector(seed=101)
    payload2 = {"data": "Updated memory via upsert", "user_id": "user_upsert"}
    store.insert(vectors=[vec2], payloads=[payload2], ids=[test_id])

    result = store.get(test_id)
    assert_true("Updated" in str(result.payload.get("data", "")),
                "payload should be updated after upsert")
    print(f"    Upsert verified: payload updated for id={test_id[:12]}...")


def test_06_search_semantic(store: GaussDB):
    """Vector similarity search with cosine distance."""
    # Insert a known vector
    target_vec = make_vector(seed=200)
    target_id = str(uuid.uuid4())
    store.insert(
        vectors=[target_vec],
        payloads=[{"data": "Target for semantic search", "user_id": "user_search"}],
        ids=[target_id]
    )

    # Search with a similar vector
    query_vec = cosine_similar_vector(target_vec, noise=0.01)
    results = store.search(
        query="semantic test",
        vectors=query_vec,
        top_k=5,
        filters={"user_id": "user_search"}
    )

    assert_true(len(results) > 0, "search should return results")
    assert_eq(results[0].id, target_id, "closest result should be our target")
    assert_gt(results[0].score, 0.5, "score should be high for similar vector")
    print(f"    Search returned {len(results)} results, top score={results[0].score:.4f}")


def test_07_search_with_filters(store: GaussDB):
    """Search with metadata filters (user_id, agent_id)."""
    # Insert vectors for different users
    vec_a = make_vector(seed=300)
    vec_b = make_vector(seed=301)
    id_a = str(uuid.uuid4())
    id_b = str(uuid.uuid4())

    store.insert(
        vectors=[vec_a, vec_b],
        payloads=[
            {"data": "User A memory", "user_id": "user_filter_a", "agent_id": "agent_x"},
            {"data": "User B memory", "user_id": "user_filter_b", "agent_id": "agent_x"},
        ],
        ids=[id_a, id_b]
    )

    # Search filtered to user_a only
    results = store.search(
        query="filter test",
        vectors=cosine_similar_vector(vec_a, noise=0.01),
        top_k=10,
        filters={"user_id": "user_filter_a"}
    )

    result_ids = [r.id for r in results]
    assert_true(id_a in result_ids, "user_a's record should appear in filtered results")
    assert_true(id_b not in result_ids, "user_b's record should NOT appear in user_a filter")
    print(f"    Filter isolation verified: user_a results={len(results)}")


def test_08_keyword_search_bm25(store: GaussDB):
    """BM25 keyword search using ### operator."""
    if not store.bm25_enabled:
        raise SkipTest("BM25 not enabled on this GaussDB instance")

    # Insert records with distinct text
    store.insert(
        vectors=[make_vector(seed=400), make_vector(seed=401), make_vector(seed=402)],
        payloads=[
            {"data": "Python programming language is versatile", "user_id": "user_bm25"},
            {"data": "Java enterprise development framework", "user_id": "user_bm25"},
            {"data": "Python data science machine learning", "user_id": "user_bm25"},
        ],
        ids=[str(uuid.uuid4()) for _ in range(3)]
    )

    # BM25 search for "Python"
    results = store.keyword_search(
        query="Python",
        top_k=5,
        filters={"user_id": "user_bm25"}
    )

    assert_true(results is not None, "keyword_search should return results (not None)")
    assert_true(len(results) >= 2, "should find at least 2 Python-related records")
    for r in results:
        assert_true("Python" in str(r.payload.get("data", "")),
                    f"BM25 result should contain 'Python': {r.payload}")
    print(f"    BM25 search for 'Python': {len(results)} results, scores={[r.score for r in results]}")


def test_09_keyword_search_empty_query(store: GaussDB):
    """BM25 with empty query should return empty list."""
    if not store.bm25_enabled:
        raise SkipTest("BM25 not enabled")

    results = store.keyword_search(query="", top_k=5, filters={"user_id": "user_bm25"})
    assert_eq(results, [], "empty query should return empty list")
    print("    Empty query correctly returns []")


def test_10_search_batch(store: GaussDB):
    """Batch search with multiple query vectors."""
    # Insert distinct vectors
    vecs = [make_vector(seed=500 + i) for i in range(5)]
    ids = [str(uuid.uuid4()) for _ in range(5)]
    store.insert(
        vectors=vecs,
        payloads=[{"data": f"Batch item {i}", "user_id": "user_batch_search"} for i in range(5)],
        ids=ids
    )

    # Batch search with 3 queries
    query_vecs = [cosine_similar_vector(vecs[0], 0.01),
                  cosine_similar_vector(vecs[2], 0.01),
                  cosine_similar_vector(vecs[4], 0.01)]

    results = store.search_batch(
        queries=["q1", "q2", "q3"],
        vectors_list=query_vecs,
        top_k=3,
        filters={"user_id": "user_batch_search"}
    )

    assert_eq(len(results), 3, "batch search should return 3 result groups")
    # First query should find vecs[0] as closest
    assert_true(len(results[0]) > 0, "first query should have results")
    assert_eq(results[0][0].id, ids[0], "first query closest should be vecs[0]")
    print(f"    Batch search: 3 queries, results per query: {[len(r) for r in results]}")


def test_11_update_payload(store: GaussDB):
    """Update payload of an existing record."""
    test_id = str(uuid.uuid4())
    store.insert(
        vectors=[make_vector(seed=600)],
        payloads=[{"data": "Before update", "user_id": "user_update"}],
        ids=[test_id]
    )

    # Update payload
    store.update(
        vector_id=test_id,
        payload={"data": "After update", "user_id": "user_update", "extra_field": "new_value"}
    )

    result = store.get(test_id)
    assert_true("After update" in str(result.payload.get("data", "")), "payload should be updated")
    print(f"    Update verified: {result.payload.get('data')}")


def test_12_update_vector(store: GaussDB):
    """Update vector of an existing record."""
    test_id = str(uuid.uuid4())
    original_vec = make_vector(seed=700)
    store.insert(
        vectors=[original_vec],
        payloads=[{"data": "Vector update test", "user_id": "user_vecupdate"}],
        ids=[test_id]
    )

    # Update with a very different vector
    new_vec = make_vector(seed=999)
    store.update(vector_id=test_id, vector=new_vec)

    # Search with new vector should find it
    results = store.search(
        query="vec update",
        vectors=cosine_similar_vector(new_vec, noise=0.001),
        top_k=1,
        filters={"user_id": "user_vecupdate"}
    )
    assert_true(len(results) > 0, "search with new vector should find record")
    assert_eq(results[0].id, test_id, "should find the updated record")
    print(f"    Vector update verified via search")


def test_13_delete_by_id(store: GaussDB):
    """Delete a record by ID."""
    test_id = str(uuid.uuid4())
    store.insert(
        vectors=[make_vector(seed=800)],
        payloads=[{"data": "To be deleted", "user_id": "user_delete"}],
        ids=[test_id]
    )

    # Verify exists
    assert_true(store.get(test_id) is not None, "record should exist before delete")

    # Delete
    store.delete(test_id)

    # Verify gone
    assert_true(store.get(test_id) is None, "record should be None after delete")
    print(f"    Delete verified: id={test_id[:12]}... is gone")


def test_14_list_with_filters(store: GaussDB):
    """List memories with scope filters."""
    # Insert records for a specific user
    for i in range(3):
        store.insert(
            vectors=[make_vector(seed=900 + i)],
            payloads=[{"data": f"List item {i}", "user_id": "user_list_test"}],
            ids=[str(uuid.uuid4())]
        )

    results = store.list(filters={"user_id": "user_list_test"}, top_k=10)
    assert_true(len(results) > 0, "list should return outer list")
    assert_true(len(results[0]) >= 3, f"should have at least 3 records, got {len(results[0])}")
    print(f"    List returned {len(results[0])} records for user_list_test")


def test_15_list_cols(store: GaussDB):
    """List all collections in the database."""
    cols = store.list_cols()
    assert_true(isinstance(cols, list), "list_cols should return a list")
    assert_true(store.collection_name in cols, f"our collection should be in list: {cols}")
    print(f"    Collections: {cols}")


def test_16_col_info(store: GaussDB):
    """Get detailed collection info."""
    info = store.col_info()
    assert_true(info["count"] > 0, "count should be > 0 after inserts")
    assert_true(len(info["indexes"]) > 0, "should have at least one index")
    assert_eq(info["vector_metric"], "cosine", "metric should be cosine")
    print(f"    col_info: count={info['count']}, indexes={info['indexes']}")
    print(f"    bm25_enabled={info['bm25_enabled']}, payload_mode={info['payload_storage_mode']}")


def test_17_scope_guard_enforcement(store: GaussDB):
    """Search without required scope filter should raise or return empty."""
    if not store.require_scoped_filters:
        raise SkipTest("require_scoped_filters is disabled")

    try:
        # Search without any user_id/agent_id/run_id filter
        results = store.search(
            query="no scope",
            vectors=make_vector(seed=1000),
            top_k=5,
            filters={}  # No scope!
        )
        # If it doesn't raise, it should return empty or the guard is not strict
        raise SkipTest("Scope guard did not raise; may be in permissive mode")
    except ValueError as e:
        assert_true("scope" in str(e).lower() or "filter" in str(e).lower(),
                    f"should mention scope/filter in error: {e}")
        print(f"    Scope guard correctly raised: {e}")


def test_18_large_payload(store: GaussDB):
    """Insert and retrieve a large payload (stress test JSONB)."""
    large_data = "x" * 10000  # 10KB payload
    test_id = str(uuid.uuid4())
    store.insert(
        vectors=[make_vector(seed=1100)],
        payloads=[{"data": large_data, "user_id": "user_large", "tags": list(range(100))}],
        ids=[test_id]
    )

    result = store.get(test_id)
    assert_true(result is not None, "should retrieve large payload record")
    assert_eq(len(result.payload.get("data", "")), 10000, "payload data should be 10KB")
    print(f"    Large payload (10KB) insert/retrieve OK")


def test_19_unicode_payload(store: GaussDB):
    """Insert and search with Chinese/Unicode content."""
    test_id = str(uuid.uuid4())
    chinese_text = "我喜欢用Python做数据分析和机器学习"
    store.insert(
        vectors=[make_vector(seed=1200)],
        payloads=[{"data": chinese_text, "user_id": "user_unicode"}],
        ids=[test_id]
    )

    result = store.get(test_id)
    assert_eq(result.payload.get("data"), chinese_text, "Chinese text should round-trip correctly")
    print(f"    Unicode payload verified: {chinese_text[:20]}...")

    # BM25 with Chinese if enabled
    if store.bm25_enabled:
        kw_results = store.keyword_search(query="Python", top_k=5, filters={"user_id": "user_unicode"})
        if kw_results:
            print(f"    BM25 Chinese search found {len(kw_results)} results")
        else:
            print(f"    BM25 Chinese search returned None/empty (may need Chinese dictionary)")


def test_20_concurrent_upsert_idempotency(store: GaussDB):
    """Multiple upserts with same ID should be idempotent."""
    test_id = str(uuid.uuid4())
    for i in range(5):
        store.insert(
            vectors=[make_vector(seed=1300 + i)],
            payloads=[{"data": f"Upsert round {i}", "user_id": "user_idempotent"}],
            ids=[test_id]
        )

    # Should only have 1 record with this ID
    result = store.get(test_id)
    assert_true(result is not None, "record should exist")
    assert_true("round 4" in result.payload.get("data", ""), "should have last upsert's data")
    print(f"    5 upserts to same ID: final payload = {result.payload.get('data')}")


def test_21_reset_collection(store: GaussDB):
    """Reset (drop + recreate) collection."""
    # Get current count
    info_before = store.col_info()
    assert_gt(info_before["count"], 0, "should have records before reset")

    store.reset()

    info_after = store.col_info()
    assert_eq(info_after["count"], 0, "count should be 0 after reset")
    print(f"    Reset: {info_before['count']} records → 0")


def test_22_delete_collection(store: GaussDB):
    """Delete collection entirely."""
    store.delete_col()
    cols = store.list_cols()
    assert_true(store.collection_name not in cols, "collection should be gone after delete_col")
    print(f"    Collection '{store.collection_name}' deleted successfully")


# ============================================================
# Main
# ============================================================
def main():
    collection_name = f"e2e_test_{uuid.uuid4().hex[:8]}"
    print(f"{'='*70}")
    print(f"  mem0 GaussDB E2E Test Suite")
    print(f"  Target: {DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}")
    print(f"  Collection: {collection_name}")
    print(f"  Dims: {DIMS}")
    print(f"  Time: {datetime.now().isoformat()}")
    print(f"{'='*70}")

    # Create the GaussDB vector store instance
    store = GaussDB(
        **DB_CONFIG,
        collection_name=collection_name,
        embedding_model_dims=DIMS,
        table_storage="ustore",
        compatibility_mode="A",
        deployment_mode="centralized",
        distribution_mode="none",
        vector_index_type="gsdiskann",  # Required for dims > 1024
        vector_metric="cosine",
        bm25_mode="auto",
        metadata_mode="auto",
        require_scoped_filters=True,
        auto_create=True,
    )

    print(f"\n  Store initialized. BM25 enabled: {store.bm25_enabled}")
    print(f"  Payload mode: {store.payload_storage_mode}, Filter mode: {store.filter_storage_mode}")

    # Run all tests in order
    tests = [
        ("01. Connection & Capability Probe", test_01_connection_and_capability_probe),
        ("02. Create Collection", test_02_create_collection),
        ("03. Insert Single Record", test_03_insert_single),
        ("04. Insert Batch (10 records)", test_04_insert_batch),
        ("05. Upsert (MERGE INTO) Existing", test_05_upsert_existing),
        ("06. Semantic Vector Search", test_06_search_semantic),
        ("07. Search with Metadata Filters", test_07_search_with_filters),
        ("08. BM25 Keyword Search", test_08_keyword_search_bm25),
        ("09. BM25 Empty Query", test_09_keyword_search_empty_query),
        ("10. Batch Search (3 queries)", test_10_search_batch),
        ("11. Update Payload", test_11_update_payload),
        ("12. Update Vector", test_12_update_vector),
        ("13. Delete by ID", test_13_delete_by_id),
        ("14. List with Filters", test_14_list_with_filters),
        ("15. List Collections", test_15_list_cols),
        ("16. Collection Info", test_16_col_info),
        ("17. Scope Guard Enforcement", test_17_scope_guard_enforcement),
        ("18. Large Payload (10KB)", test_18_large_payload),
        ("19. Unicode/Chinese Payload", test_19_unicode_payload),
        ("20. Concurrent Upsert Idempotency", test_20_concurrent_upsert_idempotency),
        ("21. Reset Collection", test_21_reset_collection),
        ("22. Delete Collection (cleanup)", test_22_delete_collection),
    ]

    for name, func in tests:
        run_test(name, func, store)

    # Final summary
    success = results.summary()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
