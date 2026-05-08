"""
P1 Multi-tenant Isolation Tests for GaussDB vector store.

Tests multi-tenant data isolation including:
- Memory API user_id isolation (add, search, get_all, delete, delete_all)
- agent_id multi-tenant isolation
- run_id isolation
- Combined scope isolation (user+agent, user+run, all three)
- Concurrent multi-user operations

~20 tests total.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest

from tests.vector_stores.conftest import (
    VECTOR_COFFEE,
    VECTOR_FLIGHT,
    VECTOR_WINDOW,
    VECTOR_AISLE,
    _assert_exact_ids,
    _ids,
    _insert_memories,
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
# 7.2.1 Memory API user_id Isolation (5 tests)
# ===========================================================================


class TestUserIdIsolation:
    """Tests for user_id based multi-tenant isolation."""

    def test_memory_add_user_id_isolation(self):
        """Data added for alice should not be visible to bob."""
        db = _new_db(prefix="p1_mt")
        try:
            alice_id = _uuid(7001)
            bob_id = _uuid(7002)
            _insert_memories(db, [
                (alice_id, VECTOR_COFFEE, _make_payload("alice coffee", user_id="alice")),
                (bob_id, VECTOR_FLIGHT, _make_payload("bob flight", user_id="bob")),
            ])
            # Search as bob should not return alice's data
            results = db.search("coffee", VECTOR_COFFEE, top_k=10, filters={"user_id": "bob"})
            result_ids = _ids(results)
            assert alice_id not in result_ids
            assert bob_id in result_ids
        finally:
            db.delete_col()

    def test_memory_search_user_id_isolation(self):
        """Search with user_id filter returns only that user's data."""
        db = _new_db(prefix="p1_mt")
        try:
            _insert_memories(db, [
                (_uuid(7011), VECTOR_COFFEE, _make_payload("alice likes coffee", user_id="alice")),
                (_uuid(7012), VECTOR_WINDOW, _make_payload("alice likes window", user_id="alice")),
                (_uuid(7013), VECTOR_COFFEE, _make_payload("bob likes coffee", user_id="bob")),
                (_uuid(7014), VECTOR_FLIGHT, _make_payload("charlie flight", user_id="charlie")),
            ])
            results = db.search("coffee", VECTOR_COFFEE, top_k=10, filters={"user_id": "alice"})
            result_ids = set(_ids(results))
            # Only alice's records should appear
            assert _uuid(7011) in result_ids
            assert _uuid(7012) in result_ids
            assert _uuid(7013) not in result_ids
            assert _uuid(7014) not in result_ids
        finally:
            db.delete_col()

    def test_memory_get_all_user_id_filter(self):
        """list with user_id filter returns only that user's data."""
        db = _new_db(prefix="p1_mt")
        try:
            _insert_memories(db, [
                (_uuid(7021), VECTOR_COFFEE, _make_payload("alice mem1", user_id="alice")),
                (_uuid(7022), VECTOR_FLIGHT, _make_payload("alice mem2", user_id="alice")),
                (_uuid(7023), VECTOR_WINDOW, _make_payload("bob mem1", user_id="bob")),
            ])
            results = _list_flat(db, filters={"user_id": "alice"}, top_k=100)
            _assert_exact_ids(results, {_uuid(7021), _uuid(7022)})
        finally:
            db.delete_col()

    def test_memory_delete_user_id_scope(self):
        """Deleting alice's record doesn't affect bob's data."""
        db = _new_db(prefix="p1_mt")
        try:
            alice_id = _uuid(7031)
            bob_id = _uuid(7032)
            _insert_memories(db, [
                (alice_id, VECTOR_COFFEE, _make_payload("alice data", user_id="alice")),
                (bob_id, VECTOR_FLIGHT, _make_payload("bob data", user_id="bob")),
            ])
            # Delete alice's record
            db.delete(vector_id=alice_id)
            # Bob's data should still exist
            bob_result = db.get(bob_id)
            assert bob_result is not None
            # Alice's data should be gone
            alice_result = db.get(alice_id)
            assert alice_result is None
        finally:
            db.delete_col()

    def test_memory_delete_all_user_id_scope(self):
        """delete_all for alice preserves bob's data."""
        db = _new_db(prefix="p1_mt")
        try:
            _insert_memories(db, [
                (_uuid(7041), VECTOR_COFFEE, _make_payload("alice mem1", user_id="alice")),
                (_uuid(7042), VECTOR_WINDOW, _make_payload("alice mem2", user_id="alice")),
                (_uuid(7043), VECTOR_FLIGHT, _make_payload("bob mem1", user_id="bob")),
                (_uuid(7044), VECTOR_AISLE, _make_payload("bob mem2", user_id="bob")),
            ])
            # Get alice's IDs and delete them
            alice_records = _list_flat(db, filters={"user_id": "alice"}, top_k=100)
            alice_ids = _ids(alice_records)
            for aid in alice_ids:
                db.delete(vector_id=aid)
            # Bob's data should be preserved
            bob_records = _list_flat(db, filters={"user_id": "bob"}, top_k=100)
            _assert_exact_ids(bob_records, {_uuid(7043), _uuid(7044)})
            # Alice should have nothing
            alice_after = _list_flat(db, filters={"user_id": "alice"}, top_k=100)
            assert len(alice_after) == 0
        finally:
            db.delete_col()


# ===========================================================================
# 7.2.2 agent_id Multi-tenant Isolation (5 tests)
# ===========================================================================


class TestAgentIdIsolation:
    """Tests for agent_id based multi-tenant isolation."""

    def test_agent_id_isolation_basic(self):
        """Same user, different agents should be isolated when filtered."""
        db = _new_db(prefix="p1_mt")
        try:
            _insert_memories(db, [
                (_uuid(7101), VECTOR_COFFEE, _make_payload("support chat", user_id="alice", agent_id="support_bot")),
                (_uuid(7102), VECTOR_FLIGHT, _make_payload("travel chat", user_id="alice", agent_id="travel_bot")),
                (_uuid(7103), VECTOR_WINDOW, _make_payload("coding chat", user_id="alice", agent_id="code_bot")),
            ])
            results = db.search("chat", VECTOR_COFFEE, top_k=10, filters={"agent_id": "support_bot"})
            result_ids = _ids(results)
            assert _uuid(7101) in result_ids
            assert _uuid(7102) not in result_ids
            assert _uuid(7103) not in result_ids
        finally:
            db.delete_col()

    def test_agent_id_cross_user_isolation(self):
        """Different users with same agent_id should be isolated by user_id."""
        db = _new_db(prefix="p1_mt")
        try:
            _insert_memories(db, [
                (_uuid(7111), VECTOR_COFFEE, _make_payload("alice support", user_id="alice", agent_id="support_bot")),
                (_uuid(7112), VECTOR_FLIGHT, _make_payload("bob support", user_id="bob", agent_id="support_bot")),
            ])
            results = db.search("support", VECTOR_COFFEE, top_k=10, filters={"user_id": "alice", "agent_id": "support_bot"})
            _assert_exact_ids(results, {_uuid(7111)})
        finally:
            db.delete_col()

    def test_agent_id_optional_filter(self):
        """Without agent_id filter, all records for user are visible."""
        db = _new_db(prefix="p1_mt")
        try:
            _insert_memories(db, [
                (_uuid(7121), VECTOR_COFFEE, _make_payload("agent1 data", user_id="alice", agent_id="agent1")),
                (_uuid(7122), VECTOR_FLIGHT, _make_payload("agent2 data", user_id="alice", agent_id="agent2")),
                (_uuid(7123), VECTOR_WINDOW, _make_payload("agent3 data", user_id="alice", agent_id="agent3")),
            ])
            # Filter only by user_id, no agent_id filter
            results = _list_flat(db, filters={"user_id": "alice"}, top_k=100)
            _assert_exact_ids(results, {_uuid(7121), _uuid(7122), _uuid(7123)})
        finally:
            db.delete_col()

    def test_agent_id_list_filter(self):
        """list filtered by specific agent_id returns only that agent's data."""
        db = _new_db(prefix="p1_mt")
        try:
            _insert_memories(db, [
                (_uuid(7131), VECTOR_COFFEE, _make_payload("travel mem1", user_id="alice", agent_id="travel_bot")),
                (_uuid(7132), VECTOR_FLIGHT, _make_payload("travel mem2", user_id="alice", agent_id="travel_bot")),
                (_uuid(7133), VECTOR_WINDOW, _make_payload("support mem1", user_id="alice", agent_id="support_bot")),
            ])
            results = _list_flat(db, filters={"agent_id": "travel_bot"}, top_k=100)
            _assert_exact_ids(results, {_uuid(7131), _uuid(7132)})
        finally:
            db.delete_col()

    def test_agent_id_combined_with_user_id(self):
        """Combined user_id + agent_id filter narrows results correctly."""
        db = _new_db(prefix="p1_mt")
        try:
            _insert_memories(db, [
                (_uuid(7141), VECTOR_COFFEE, _make_payload("alice travel", user_id="alice", agent_id="travel_bot")),
                (_uuid(7142), VECTOR_FLIGHT, _make_payload("bob travel", user_id="bob", agent_id="travel_bot")),
                (_uuid(7143), VECTOR_WINDOW, _make_payload("alice support", user_id="alice", agent_id="support_bot")),
            ])
            results = _list_flat(db, filters={"user_id": "alice", "agent_id": "travel_bot"}, top_k=100)
            _assert_exact_ids(results, {_uuid(7141)})
        finally:
            db.delete_col()


# ===========================================================================
# 7.2.3 run_id Isolation (3 tests)
# ===========================================================================


class TestRunIdIsolation:
    """Tests for run_id based isolation."""

    def test_run_id_isolation_basic(self):
        """Different run_ids should be isolated when filtered."""
        db = _new_db(prefix="p1_mt")
        try:
            _insert_memories(db, [
                (_uuid(7201), VECTOR_COFFEE, _make_payload("run1 data", user_id="alice", run_id="run_001")),
                (_uuid(7202), VECTOR_FLIGHT, _make_payload("run2 data", user_id="alice", run_id="run_002")),
                (_uuid(7203), VECTOR_WINDOW, _make_payload("run3 data", user_id="alice", run_id="run_003")),
            ])
            results = db.search("data", VECTOR_COFFEE, top_k=10, filters={"run_id": "run_001"})
            result_ids = _ids(results)
            assert _uuid(7201) in result_ids
            assert _uuid(7202) not in result_ids
            assert _uuid(7203) not in result_ids
        finally:
            db.delete_col()

    def test_run_id_combined_with_user_id(self):
        """user_id + run_id combined filter works correctly."""
        db = _new_db(prefix="p1_mt")
        try:
            _insert_memories(db, [
                (_uuid(7211), VECTOR_COFFEE, _make_payload("alice run1", user_id="alice", run_id="run_001")),
                (_uuid(7212), VECTOR_FLIGHT, _make_payload("alice run2", user_id="alice", run_id="run_002")),
                (_uuid(7213), VECTOR_WINDOW, _make_payload("bob run1", user_id="bob", run_id="run_001")),
            ])
            results = _list_flat(db, filters={"user_id": "alice", "run_id": "run_001"}, top_k=100)
            _assert_exact_ids(results, {_uuid(7211)})
        finally:
            db.delete_col()

    def test_run_id_combined_with_agent_id(self):
        """agent_id + run_id combined filter works correctly."""
        db = _new_db(prefix="p1_mt")
        try:
            _insert_memories(db, [
                (_uuid(7221), VECTOR_COFFEE, _make_payload("agent1 run1", user_id="alice", agent_id="bot_a", run_id="run_001")),
                (_uuid(7222), VECTOR_FLIGHT, _make_payload("agent1 run2", user_id="alice", agent_id="bot_a", run_id="run_002")),
                (_uuid(7223), VECTOR_WINDOW, _make_payload("agent2 run1", user_id="alice", agent_id="bot_b", run_id="run_001")),
            ])
            results = _list_flat(db, filters={"agent_id": "bot_a", "run_id": "run_001"}, top_k=100)
            _assert_exact_ids(results, {_uuid(7221)})
        finally:
            db.delete_col()


# ===========================================================================
# 7.2.4 Combined Scope Isolation (4 tests)
# ===========================================================================


class TestCombinedScopeIsolation:
    """Tests for combined scope (user_id + agent_id + run_id) isolation."""

    def test_scope_all_three_combined(self):
        """user_id + agent_id + run_id combined filter returns exact match."""
        db = _new_db(prefix="p1_mt")
        try:
            _insert_memories(db, [
                (_uuid(7301), VECTOR_COFFEE, _make_payload("target", user_id="alice", agent_id="bot_a", run_id="run_001")),
                (_uuid(7302), VECTOR_FLIGHT, _make_payload("diff run", user_id="alice", agent_id="bot_a", run_id="run_002")),
                (_uuid(7303), VECTOR_WINDOW, _make_payload("diff agent", user_id="alice", agent_id="bot_b", run_id="run_001")),
                (_uuid(7304), VECTOR_AISLE, _make_payload("diff user", user_id="bob", agent_id="bot_a", run_id="run_001")),
            ])
            results = _list_flat(db,
                filters={"user_id": "alice", "agent_id": "bot_a", "run_id": "run_001"},
                top_k=100,
            )
            _assert_exact_ids(results, {_uuid(7301)})
        finally:
            db.delete_col()

    def test_scope_partial_match_excluded(self):
        """Partial scope match (2 of 3 fields) should not return non-matching data."""
        db = _new_db(prefix="p1_mt")
        try:
            _insert_memories(db, [
                (_uuid(7311), VECTOR_COFFEE, _make_payload("full match", user_id="alice", agent_id="bot_a", run_id="run_001")),
                (_uuid(7312), VECTOR_FLIGHT, _make_payload("partial mismatch", user_id="alice", agent_id="bot_a", run_id="run_999")),
            ])
            results = _list_flat(db,
                filters={"user_id": "alice", "agent_id": "bot_a", "run_id": "run_001"},
                top_k=100,
            )
            _assert_exact_ids(results, {_uuid(7311)})
        finally:
            db.delete_col()

    def test_scope_empty_result_on_mismatch(self):
        """Completely wrong scope returns empty results."""
        db = _new_db(prefix="p1_mt")
        try:
            _insert_memories(db, [
                (_uuid(7321), VECTOR_COFFEE, _make_payload("some data", user_id="alice", agent_id="bot_a", run_id="run_001")),
                (_uuid(7322), VECTOR_FLIGHT, _make_payload("other data", user_id="bob", agent_id="bot_b", run_id="run_002")),
            ])
            results = _list_flat(db,
                filters={"user_id": "charlie", "agent_id": "bot_x", "run_id": "run_999"},
                top_k=100,
            )
            assert len(results) == 0
        finally:
            db.delete_col()

    def test_scope_wildcard_user_all_agents(self):
        """user_id filter only (no agent_id) sees all agents for that user."""
        db = _new_db(prefix="p1_mt")
        try:
            _insert_memories(db, [
                (_uuid(7331), VECTOR_COFFEE, _make_payload("alice bot_a", user_id="alice", agent_id="bot_a")),
                (_uuid(7332), VECTOR_FLIGHT, _make_payload("alice bot_b", user_id="alice", agent_id="bot_b")),
                (_uuid(7333), VECTOR_WINDOW, _make_payload("alice bot_c", user_id="alice", agent_id="bot_c")),
                (_uuid(7334), VECTOR_AISLE, _make_payload("bob bot_a", user_id="bob", agent_id="bot_a")),
            ])
            results = _list_flat(db, filters={"user_id": "alice"}, top_k=100)
            _assert_exact_ids(results, {_uuid(7331), _uuid(7332), _uuid(7333)})
        finally:
            db.delete_col()


# ===========================================================================
# 7.2.5 Concurrent Multi-user Operations (3 tests)
# ===========================================================================


class TestConcurrentMultiUser:
    """Tests for concurrent multi-user operations."""

    def test_concurrent_multi_user_add(self):
        """5 threads adding data for different users, verify isolation."""
        db = _new_db(prefix="p1_mt")
        try:
            users = [f"user_{i}" for i in range(5)]
            vectors = [VECTOR_COFFEE, VECTOR_FLIGHT, VECTOR_WINDOW, VECTOR_AISLE, VECTOR_COFFEE]

            def add_for_user(idx):
                uid = users[idx]
                record_id = _uuid(7400 + idx)
                db.insert(
                    ids=[record_id],
                    vectors=[vectors[idx]],
                    payloads=[_make_payload(f"{uid} memory", user_id=uid)],
                )
                return uid, record_id

            # Run concurrent inserts
            results_map = {}
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = {executor.submit(add_for_user, i): i for i in range(5)}
                for future in as_completed(futures):
                    uid, record_id = future.result()
                    results_map[uid] = record_id

            # Verify isolation: each user sees only their own data
            for uid, record_id in results_map.items():
                user_records = _list_flat(db, filters={"user_id": uid}, top_k=100)
                assert len(user_records) == 1
                assert _ids(user_records)[0] == record_id
        finally:
            db.delete_col()

    def test_concurrent_multi_user_search(self):
        """5 threads searching for different users, verify isolation."""
        db = _new_db(prefix="p1_mt")
        try:
            # Pre-populate data for 5 users
            records = []
            for i in range(5):
                uid = f"user_{i}"
                vectors = [VECTOR_COFFEE, VECTOR_FLIGHT, VECTOR_WINDOW, VECTOR_AISLE, VECTOR_COFFEE]
                records.append((_uuid(7410 + i), vectors[i], _make_payload(f"{uid} data", user_id=uid)))
            _insert_memories(db, records)

            def search_for_user(idx):
                uid = f"user_{idx}"
                vectors = [VECTOR_COFFEE, VECTOR_FLIGHT, VECTOR_WINDOW, VECTOR_AISLE, VECTOR_COFFEE]
                results = db.search("data", vectors[idx], top_k=10, filters={"user_id": uid})
                return uid, results

            # Run concurrent searches
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = {executor.submit(search_for_user, i): i for i in range(5)}
                for future in as_completed(futures):
                    uid, results = future.result()
                    result_ids = _ids(results)
                    idx = int(uid.split("_")[1])
                    expected_id = _uuid(7410 + idx)
                    assert expected_id in result_ids, f"{uid} should see their own data"
                    # Verify no other user's data leaked
                    for other_idx in range(5):
                        if other_idx != idx:
                            assert _uuid(7410 + other_idx) not in result_ids
        finally:
            db.delete_col()

    def test_concurrent_same_user_diff_agent(self):
        """Same user, different agents concurrent operations maintain isolation."""
        db = _new_db(prefix="p1_mt")
        try:
            agents = [f"agent_{i}" for i in range(5)]
            vectors = [VECTOR_COFFEE, VECTOR_FLIGHT, VECTOR_WINDOW, VECTOR_AISLE, VECTOR_COFFEE]

            def add_for_agent(idx):
                agent = agents[idx]
                record_id = _uuid(7420 + idx)
                db.insert(
                    ids=[record_id],
                    vectors=[vectors[idx]],
                    payloads=[_make_payload(f"alice {agent}", user_id="alice", agent_id=agent)],
                )
                return agent, record_id

            # Run concurrent inserts for same user, different agents
            results_map = {}
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = {executor.submit(add_for_agent, i): i for i in range(5)}
                for future in as_completed(futures):
                    agent, record_id = future.result()
                    results_map[agent] = record_id

            # Verify each agent's data is isolated when filtered
            for agent, record_id in results_map.items():
                agent_records = _list_flat(db, filters={"user_id": "alice", "agent_id": agent}, top_k=100)
                assert len(agent_records) == 1
                assert _ids(agent_records)[0] == record_id

            # Verify user sees all agents without agent_id filter
            all_records = _list_flat(db, filters={"user_id": "alice"}, top_k=100)
            assert len(all_records) == 5
        finally:
            db.delete_col()
