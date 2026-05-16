"""
Commercial-grade validation suite for the GaussDB mem0 provider.

This suite focuses on the behaviors we would want to hold for a production
export gate: collection lifecycle, scoped isolation, supported filter
semantics, unsupported range behavior, ranking stability, UTF-8 safety,
keyword search, and distributed-mode smoke coverage.
"""

import logging
import os
import uuid
from concurrent.futures import ThreadPoolExecutor

import pytest

pytest.importorskip("psycopg2", reason="GaussDB live tests require psycopg2-compatible driver")

from mem0.vector_stores.gaussdb import GaussDB
from tests.vector_stores.conftest import (
    VECTOR_AISLE,
    VECTOR_COFFEE,
    VECTOR_FLIGHT,
    VECTOR_UNIT_X,
    VECTOR_UNIT_Y,
    VECTOR_UNIT_Z,
    VECTOR_WINDOW,
    _assert_exact_ids,
    _assert_ordered_ids,
    _env_bool,
    _gaussdb_env_config,
    _ids,
    _insert_memories,
    _list_flat,
    _managed_db,
    _new_collection_name,
    _uuid,
    gaussdb_available,
)


pytestmark = pytest.mark.skipif(
    not gaussdb_available(),
    reason="Set GAUSSDB_TEST_DSN or GAUSSDB_TEST_HOST/PORT/DATABASE/USER/PASSWORD to run GaussDB live tests",
)


def _new_dist_db(prefix: str = "commercial_dist", **overrides) -> GaussDB:
    config = _gaussdb_env_config(
        _new_collection_name(prefix),
        deployment_mode="distributed",
        embedding_model_dims=overrides.pop("embedding_model_dims", 4),
        **overrides,
    )
    assert config is not None, "GaussDB distributed test environment not configured"
    return GaussDB(**config)


def test_commercial_centralized_collection_contract():
    with _managed_db(prefix="commercial_contract", embedding_model_dims=4, vector_index_type="gsdiskann") as db:
        info = db.col_info()

        assert db.collection_name in db.list_cols()
        assert info["name"] == db.collection_name
        assert info["dimension"] == 4
        assert info["deployment_mode"] == "centralized"
        assert info["payload_storage_mode"] == "jsonb"
        assert info["filter_storage_mode"] == "json_expression"
        assert any("vector" in index.lower() for index in info["indexes"])


def test_commercial_centralized_crud_scope_and_batch_paths():
    with _managed_db(prefix="commercial_crud") as db:
        alice_primary = _uuid(9101)
        alice_secondary = _uuid(9102)
        bob_only = _uuid(9103)

        _insert_memories(
            db,
            [
                (
                    alice_primary,
                    VECTOR_COFFEE,
                    {
                        "data": "Alice likes morning latte coffee",
                        "text_lemmatized": "alice likes morning latte coffee",
                        "user_id": "commercial_alice",
                        "category": "drink",
                    },
                ),
                (
                    alice_secondary,
                    VECTOR_WINDOW,
                    {
                        "data": "Alice prefers window seats on flights",
                        "text_lemmatized": "alice prefers window seats on flights",
                        "user_id": "commercial_alice",
                        "category": "travel",
                    },
                ),
                (
                    bob_only,
                    VECTOR_AISLE,
                    {
                        "data": "Bob prefers aisle seats",
                        "text_lemmatized": "bob prefers aisle seats",
                        "user_id": "commercial_bob",
                        "category": "travel",
                    },
                ),
            ],
        )

        assert db.get(alice_primary).payload["user_id"] == "commercial_alice"
        _assert_exact_ids(
            db.search("latte", VECTOR_COFFEE, top_k=10, filters={"user_id": "commercial_alice"}),
            {alice_primary, alice_secondary},
        )
        _assert_exact_ids(
            _list_flat(db, filters={"user_id": "commercial_alice"}, top_k=10),
            {alice_primary, alice_secondary},
        )

        batch_rows = db.search_batch(
            ["latte", "window"],
            [VECTOR_COFFEE, VECTOR_WINDOW],
            top_k=2,
            filters={"user_id": "commercial_alice"},
        )
        assert [set(_ids(rows)) for rows in batch_rows] == [{alice_primary, alice_secondary}, {alice_secondary, alice_primary}]

        db.update(
            vector_id=alice_secondary,
            vector=VECTOR_FLIGHT,
            payload={
                "data": "Alice now prefers flights with priority boarding",
                "text_lemmatized": "alice now prefers flights with priority boarding",
                "user_id": "commercial_alice",
                "category": "travel",
            },
        )
        assert db.get(alice_secondary).payload["data"] == "Alice now prefers flights with priority boarding"

        db.insert(
            ids=[alice_primary],
            vectors=[VECTOR_COFFEE],
            payloads=[
                {
                    "data": "Alice likes espresso now",
                    "text_lemmatized": "alice likes espresso now",
                    "user_id": "commercial_alice",
                    "category": "drink",
                }
            ],
        )
        assert db.get(alice_primary).payload["data"] == "Alice likes espresso now"
        assert len(_list_flat(db, filters={"user_id": "commercial_alice"}, top_k=10)) == 2

        db.delete(bob_only)
        assert db.get(bob_only) is None


def test_commercial_centralized_filter_matrix_and_unsupported_range(caplog):
    with _managed_db(prefix="commercial_filter") as db:
        travel_id = _uuid(9201)
        food_id = _uuid(9202)
        work_id = _uuid(9203)

        _insert_memories(
            db,
            [
                (
                    travel_id,
                    VECTOR_FLIGHT,
                    {
                        "data": "Plan flight with priority boarding",
                        "text_lemmatized": "plan flight with priority boarding",
                        "user_id": "commercial_filter",
                        "category": "travel",
                        "priority": 7,
                        "tag": "boarding-plan",
                        "status": "active",
                    },
                ),
                (
                    food_id,
                    VECTOR_COFFEE,
                    {
                        "data": "Find coffee near the office",
                        "text_lemmatized": "find coffee near the office",
                        "user_id": "commercial_filter",
                        "category": "food",
                        "priority": 2,
                        "tag": "coffee-shop",
                        "status": "active",
                    },
                ),
                (
                    work_id,
                    VECTOR_WINDOW,
                    {
                        "data": "Prepare quarterly plan",
                        "text_lemmatized": "prepare quarterly plan",
                        "user_id": "commercial_filter",
                        "category": "work",
                        "priority": 5,
                        "tag": "QuarterlyPlan",
                        "status": "archived",
                    },
                ),
            ],
        )

        cases = [
            ({"user_id": "commercial_filter", "category": {"eq": "travel"}}, {travel_id}),
            ({"user_id": "commercial_filter", "category": {"ne": "travel"}}, {food_id, work_id}),
            ({"user_id": "commercial_filter", "category": {"in": ["travel", "food"]}}, {travel_id, food_id}),
            ({"user_id": "commercial_filter", "category": {"nin": ["travel", "food"]}}, {work_id}),
            ({"user_id": "commercial_filter", "tag": {"contains": "coffee"}}, {food_id}),
            ({"user_id": "commercial_filter", "tag": {"icontains": "plan"}}, {travel_id, work_id}),
            (
                {
                    "$and": [
                        {"user_id": "commercial_filter"},
                        {"category": {"in": ["travel", "work"]}},
                        {"status": {"in": ["active", "archived"]}},
                    ]
                },
                {travel_id, work_id},
            ),
            (
                {
                    "$or": [
                        {"user_id": "commercial_filter", "category": "travel"},
                        {"user_id": "commercial_filter", "category": "food"},
                    ]
                },
                {travel_id, food_id},
            ),
            ({"user_id": "commercial_filter", "$not": [{"category": "food"}]}, {travel_id, work_id}),
        ]

        for filters, expected_ids in cases:
            _assert_exact_ids(db.search("filter", VECTOR_COFFEE, top_k=10, filters=filters), expected_ids)

        with caplog.at_level(logging.WARNING):
            rows = db.search(
                "filter",
                VECTOR_COFFEE,
                top_k=10,
                filters={"user_id": "commercial_filter", "priority": {"gte": 5}},
            )
        _assert_exact_ids(rows, set())
        assert "range filter operators" in caplog.text


def test_commercial_centralized_vector_order_and_topk():
    with _managed_db(prefix="commercial_rank", embedding_model_dims=3) as db:
        _insert_memories(
            db,
            [
                (_uuid(9301), VECTOR_UNIT_X, {"data": "x", "text_lemmatized": "x", "user_id": "rank_user"}),
                (_uuid(9302), [0.8, 0.2, 0.0], {"data": "x-near", "text_lemmatized": "x near", "user_id": "rank_user"}),
                (_uuid(9303), VECTOR_UNIT_Y, {"data": "y", "text_lemmatized": "y", "user_id": "rank_user"}),
                (_uuid(9304), VECTOR_UNIT_Z, {"data": "z", "text_lemmatized": "z", "user_id": "rank_user"}),
            ],
        )

        results = db.search("rank", VECTOR_UNIT_X, top_k=3, filters={"user_id": "rank_user"})
        assert len(results) == 3
        _assert_ordered_ids(results, [_uuid(9301), _uuid(9302), _uuid(9303)])


def test_commercial_centralized_utf8_roundtrip_and_filtering():
    with _managed_db(prefix="commercial_utf8", embedding_model_dims=3) as db:
        records = [
            (_uuid(9401), VECTOR_COFFEE, {"data": "我喜欢早晨喝拿铁咖啡", "text_lemmatized": "我喜欢早晨喝拿铁咖啡", "user_id": "utf8_user", "language": "zh-CN"}),
            (_uuid(9402), VECTOR_FLIGHT, {"data": "日本語の窓側席メモです", "text_lemmatized": "日本語の窓側席メモです", "user_id": "utf8_user", "language": "ja"}),
            (_uuid(9403), VECTOR_WINDOW, {"data": "مرحبا بالقهوة والسفر", "text_lemmatized": "مرحبا بالقهوة والسفر", "user_id": "utf8_user", "language": "ar"}),
            (_uuid(9404), VECTOR_AISLE, {"data": "emoji 😀🚀 mixed English 中文", "text_lemmatized": "emoji mixed english 中文", "user_id": "utf8_user", "language": "mixed"}),
        ]
        _insert_memories(db, records)

        listed = _list_flat(db, filters={"user_id": "utf8_user"}, top_k=10)
        assert {row.payload["data"] for row in listed} == {record[2]["data"] for record in records}

        chinese_rows = db.search("拿铁", VECTOR_COFFEE, top_k=10, filters={"user_id": "utf8_user", "language": "zh-CN"})
        _assert_exact_ids(chinese_rows, {_uuid(9401)})


def test_commercial_centralized_keyword_search_if_enabled():
    with _managed_db(prefix="commercial_bm25") as db:
        if not db.bm25_enabled:
            pytest.skip("BM25 was disabled by capability probe on this centralized environment")

        target_id = _uuid(9501)
        other_id = _uuid(9502)
        _insert_memories(
            db,
            [
                (
                    target_id,
                    VECTOR_COFFEE,
                    {
                        "data": "Alice likes latte coffee in the morning",
                        "text_lemmatized": "alice likes latte coffee in the morning",
                        "user_id": "bm25_user",
                    },
                ),
                (
                    other_id,
                    VECTOR_WINDOW,
                    {
                        "data": "Alice books quiet window seats",
                        "text_lemmatized": "alice books quiet window seats",
                        "user_id": "bm25_user",
                    },
                ),
            ],
        )

        rows = db.keyword_search("latte coffee", top_k=5, filters={"user_id": "bm25_user"})
        assert rows is not None
        assert len(rows) >= 1
        assert rows[0].id == target_id


@pytest.mark.skipif(not _env_bool("GAUSSDB_TEST_DISTRIBUTED"), reason="Set GAUSSDB_TEST_DISTRIBUTED=true to run distributed commercial validation")
def test_commercial_distributed_contract_and_crud_smoke():
    db = _new_dist_db()
    try:
        info = db.col_info()
        assert db.deployment_mode == "distributed"
        assert db.bm25_enabled is False
        assert info["deployment_mode"] == "distributed"

        memory_id = _uuid(9601)
        db.insert(
            ids=[memory_id],
            vectors=[[1.0, 0.0, 0.0, 0.0]],
            payloads=[{"data": "distributed memory", "text_lemmatized": "distributed memory", "user_id": "dist_user"}],
        )
        _assert_exact_ids(db.search("distributed", [1.0, 0.0, 0.0, 0.0], top_k=5, filters={"user_id": "dist_user"}), {memory_id})

        db.update(vector_id=memory_id, vector=[0.0, 1.0, 0.0, 0.0], payload={"data": "distributed updated", "user_id": "dist_user"})
        assert db.get(memory_id).payload["data"] == "distributed updated"
        _assert_exact_ids(_list_flat(db, filters={"user_id": "dist_user"}, top_k=10), {memory_id})

        batch_rows = db.search_batch(
            ["one", "two"],
            [[0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
            top_k=1,
            filters={"user_id": "dist_user"},
        )
        assert len(batch_rows) == 2
        assert all(rows and rows[0].id == memory_id for rows in batch_rows)
    finally:
        db.delete_col()


@pytest.mark.skipif(not _env_bool("GAUSSDB_TEST_DISTRIBUTED"), reason="Set GAUSSDB_TEST_DISTRIBUTED=true to run distributed commercial validation")
def test_commercial_distributed_scope_guard_and_tenant_isolation():
    db = _new_dist_db()
    try:
        alice_id = _uuid(9901)
        bob_id = _uuid(9902)
        public_like_id = _uuid(9903)
        _insert_memories(
            db,
            [
                (
                    alice_id,
                    [1.0, 0.0, 0.0, 0.0],
                    {"data": "alice coffee", "text_lemmatized": "alice coffee", "user_id": "dist_alice", "agent_id": "agent_a"},
                ),
                (
                    bob_id,
                    [0.0, 1.0, 0.0, 0.0],
                    {"data": "bob coffee", "text_lemmatized": "bob coffee", "user_id": "dist_bob", "agent_id": "agent_b"},
                ),
                (
                    public_like_id,
                    [0.0, 0.0, 1.0, 0.0],
                    {"data": "public note", "text_lemmatized": "public note", "category": "public"},
                ),
            ],
        )

        _assert_exact_ids(
            db.search("coffee", [1.0, 0.0, 0.0, 0.0], top_k=10, filters={"user_id": "dist_alice"}),
            {alice_id},
        )
        _assert_exact_ids(_list_flat(db, filters={"user_id": "dist_bob"}, top_k=10), {bob_id})

        with pytest.raises(ValueError, match="requires at least one scoped filter"):
            db.search("coffee", [1.0, 0.0, 0.0, 0.0], top_k=10, filters={"category": "public"})
        with pytest.raises(ValueError, match="requires at least one scoped filter"):
            db.search(
                "coffee",
                [1.0, 0.0, 0.0, 0.0],
                top_k=10,
                filters={"$or": [{"user_id": "dist_alice"}, {"category": "public"}]},
            )
    finally:
        db.delete_col()


@pytest.mark.skipif(not _env_bool("GAUSSDB_TEST_DISTRIBUTED"), reason="Set GAUSSDB_TEST_DISTRIBUTED=true to run distributed commercial validation")
def test_commercial_distributed_vector_order_and_topk():
    db = _new_dist_db(embedding_model_dims=4)
    try:
        _insert_memories(
            db,
            [
                (_uuid(9911), [1.0, 0.0, 0.0, 0.0], {"data": "x", "user_id": "dist_rank"}),
                (_uuid(9912), [0.8, 0.2, 0.0, 0.0], {"data": "x-near", "user_id": "dist_rank"}),
                (_uuid(9913), [0.0, 1.0, 0.0, 0.0], {"data": "y", "user_id": "dist_rank"}),
                (_uuid(9914), [0.0, 0.0, 1.0, 0.0], {"data": "z", "user_id": "dist_rank"}),
            ],
        )

        results = db.search("rank", [1.0, 0.0, 0.0, 0.0], top_k=3, filters={"user_id": "dist_rank"})
        assert len(results) == 3
        _assert_ordered_ids(results, [_uuid(9911), _uuid(9912), _uuid(9913)])
        assert len(db.search("rank", [1.0, 0.0, 0.0, 0.0], top_k=1, filters={"user_id": "dist_rank"})) == 1
    finally:
        db.delete_col()


@pytest.mark.skipif(not _env_bool("GAUSSDB_TEST_DISTRIBUTED"), reason="Set GAUSSDB_TEST_DISTRIBUTED=true to run distributed commercial validation")
def test_commercial_distributed_concurrent_upsert_idempotency():
    db = _new_dist_db(embedding_model_dims=4, maxconn=5)
    try:
        record_id = _uuid(9921)
        db.insert(
            ids=[record_id],
            vectors=[[1.0, 0.0, 0.0, 0.0]],
            payloads=[{"data": "initial", "text_lemmatized": "initial", "user_id": "dist_concurrent"}],
        )

        def do_update(index: int):
            db.update(
                vector_id=record_id,
                vector=[0.0, 1.0, 0.0, 0.0],
                payload={
                    "data": f"update_{index}",
                    "text_lemmatized": f"update_{index}",
                    "user_id": "dist_concurrent",
                },
            )

        with ThreadPoolExecutor(max_workers=5) as executor:
            list(executor.map(do_update, range(10)))

        listed = _list_flat(db, filters={"user_id": "dist_concurrent"}, top_k=10)
        assert len(listed) == 1
        assert db.get(record_id) is not None
    finally:
        db.delete_col()


@pytest.mark.skipif(not _env_bool("GAUSSDB_TEST_DISTRIBUTED"), reason="Set GAUSSDB_TEST_DISTRIBUTED=true to run distributed commercial validation")
def test_commercial_distributed_collection_lifecycle():
    db = _new_dist_db(embedding_model_dims=4)
    try:
        assert db.collection_name in db.list_cols()
        record_id = _uuid(9931)
        db.insert(
            ids=[record_id],
            vectors=[[1.0, 0.0, 0.0, 0.0]],
            payloads=[{"data": "reset me", "user_id": "dist_lifecycle"}],
        )
        assert len(_list_flat(db, filters={"user_id": "dist_lifecycle"}, top_k=10)) == 1

        db.reset()
        assert len(_list_flat(db, filters={"user_id": "dist_lifecycle"}, top_k=10)) == 0
        assert db.collection_name in db.list_cols()
        assert db.col_info()["count"] == 0
    finally:
        db.delete_col()


@pytest.mark.skipif(not _env_bool("GAUSSDB_TEST_DISTRIBUTED"), reason="Set GAUSSDB_TEST_DISTRIBUTED=true to run distributed commercial validation")
def test_commercial_distributed_filter_and_range_semantics(caplog):
    db = _new_dist_db()
    try:
        low_id = _uuid(9701)
        high_id = _uuid(9702)
        _insert_memories(
            db,
            [
                (low_id, [1.0, 0.0, 0.0, 0.0], {"data": "food", "user_id": "dist_filter", "category": "food", "priority": "3"}),
                (high_id, [0.0, 1.0, 0.0, 0.0], {"data": "travel", "user_id": "dist_filter", "category": "travel", "priority": "8"}),
            ],
        )

        _assert_exact_ids(
            db.search("distributed", [1.0, 0.0, 0.0, 0.0], top_k=10, filters={"user_id": "dist_filter", "category": {"in": ["food", "travel"]}}),
            {low_id, high_id},
        )

        with caplog.at_level(logging.WARNING):
            rows = db.search(
                "distributed",
                [1.0, 0.0, 0.0, 0.0],
                top_k=10,
                filters={"user_id": "dist_filter", "priority": {"gt": "5"}},
            )
        _assert_exact_ids(rows, set())
        assert "range filter operators" in caplog.text
    finally:
        db.delete_col()


@pytest.mark.skipif(not _env_bool("GAUSSDB_TEST_DISTRIBUTED"), reason="Set GAUSSDB_TEST_DISTRIBUTED=true to run distributed commercial validation")
def test_commercial_distributed_utf8_roundtrip():
    db = _new_dist_db(embedding_model_dims=4)
    try:
        record_id = _uuid(9801)
        payload = {"data": "分布式 UTF8 校验 with 日本語 and 😀", "user_id": "dist_utf8", "language": "mixed"}
        db.insert(ids=[record_id], vectors=[[0.0, 0.0, 1.0, 0.0]], payloads=[payload])

        result = db.get(record_id)
        assert result.payload["data"] == payload["data"]
        _assert_exact_ids(
            _list_flat(db, filters={"user_id": "dist_utf8", "language": "mixed"}, top_k=10),
            {record_id},
        )
    finally:
        db.delete_col()
