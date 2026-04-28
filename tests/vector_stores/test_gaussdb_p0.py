"""
GaussDB live capability tests for the mem0 integration.

These tests intentionally require explicit test-only environment variables. They
must not contain real database addresses, usernames, or passwords as defaults.

Covered test points:
- Memory.from_config upper path: add/search/delete using a real GaussDB provider.
- Provider CRUD: insert, batch upsert, get, update, delete, search.
- Tenant isolation: search, search_batch, list, OR/NOT guard behavior.
- JSON payload filters: eq, ne, in, nin, range, contains, icontains, AND, OR, NOT.
- Compatibility profile: TEXT payload plus redundant scoped columns.
- UTF-8 payload round trip: Chinese and mixed Chinese/English memories.
- Vector metrics: cosine and l2 exact-match behavior.
- Optional BM25: keyword_search with scoped filters and empty-query contract.
- Collection operations: list_cols, col_info, analyze, reset, delete_col.
- Migration helpers: migration_dry_run and backfill_derived_fields.
- Optional vector index matrix: gsivfflat/gsdiskann with cosine/l2.

Example:
    $env:GAUSSDB_TEST_HOST = "<host>"
    $env:GAUSSDB_TEST_PORT = "19995"
    $env:GAUSSDB_TEST_DATABASE = "<database>"
    $env:GAUSSDB_TEST_USER = "<user>"
    $env:GAUSSDB_TEST_PASSWORD = "<password>"
    pytest tests/vector_stores/test_gaussdb_p0.py -v

Optional switches:
    GAUSSDB_TEST_DSN                 Full test DSN, overrides host/port/database/user/password.
    GAUSSDB_TEST_SSLMODE             Optional SSL mode.
    GAUSSDB_TEST_SSLROOTCERT         Optional SSL root certificate path.
    GAUSSDB_TEST_VECTOR_INDEX        Defaults to gsivfflat for test speed.
    GAUSSDB_TEST_MAINTENANCE_MEM     Defaults to 128MB for index creation.
    GAUSSDB_TEST_ENABLE_PROBE        Defaults to false for faster collection setup.
    GAUSSDB_TEST_RUN_BM25            Set to true to require and verify BM25.
    GAUSSDB_TEST_RUN_INDEX_MATRIX    Set to true to run the full index/metric matrix.
"""

import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("psycopg2", reason="GaussDB live tests require psycopg2-compatible driver")

from mem0 import Memory
from mem0.vector_stores.gaussdb import GaussDB


EMBEDDING_DIMS = 3
VECTOR_COFFEE = [0.10, 0.20, 0.30]
VECTOR_FLIGHT = [0.90, 0.10, 0.10]
VECTOR_WINDOW = [0.20, 0.80, 0.20]
VECTOR_AISLE = [0.20, 0.20, 0.80]


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _new_collection(prefix: str = "mem0_p0") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def _gaussdb_env_config(collection_name: str, **overrides):
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
            "embedding_model_dims": EMBEDDING_DIMS,
            "id_column_type": "uuid",
            "vector_index_type": os.getenv("GAUSSDB_TEST_VECTOR_INDEX", "gsivfflat"),
            "vector_index_maintenance_work_mem": os.getenv("GAUSSDB_TEST_MAINTENANCE_MEM", "128MB"),
            "bm25_mode": "disabled",
            "enable_capability_probe": _env_bool("GAUSSDB_TEST_ENABLE_PROBE", default=False),
            "require_scoped_filters": True,
            "auto_create": True,
        }
    )
    config.update({key: value for key, value in overrides.items() if value is not None})
    return config


pytestmark = pytest.mark.skipif(
    _gaussdb_env_config("mem0_p0_probe") is None,
    reason=("Set GAUSSDB_TEST_DSN or all of GAUSSDB_TEST_HOST/PORT/DATABASE/USER/PASSWORD to run GaussDB live tests"),
)


class FakeEmbedder:
    def embed(self, text, memory_action=None):
        normalized = str(text).lower()
        if "aisle" in normalized:
            return VECTOR_AISLE
        if "window" in normalized:
            return VECTOR_WINDOW
        if "flight" in normalized:
            return VECTOR_FLIGHT
        return VECTOR_COFFEE

    def embed_batch(self, texts, memory_action="add"):
        return [self.embed(text, memory_action) for text in texts]


def _new_db(collection_name: str | None = None, **overrides) -> GaussDB:
    config = _gaussdb_env_config(collection_name or _new_collection(), **overrides)
    assert config is not None
    return GaussDB(**config)


def _uuid(suffix: int) -> str:
    return f"00000000-0000-0000-0000-{suffix:012d}"


def _ids(rows) -> list[str]:
    return [str(row.id) for row in rows]


def _assert_exact_ids(rows, expected_ids: set[str]) -> None:
    assert set(_ids(rows)) == expected_ids


def _list_rows(db: GaussDB, filters: dict, top_k: int = 100):
    listed = db.list(filters=filters, top_k=top_k)
    assert len(listed) == 1
    return listed[0]


def _server_encoding(db: GaussDB) -> str:
    with db._get_cursor() as cur:
        cur.execute("SHOW server_encoding")
        return str(cur.fetchone()[0]).upper()


def _insert_memories(db: GaussDB, records: list[tuple[str, list[float], dict]]) -> None:
    db.insert(
        ids=[record_id for record_id, _, _ in records],
        vectors=[vector for _, vector, _ in records],
        payloads=[payload for _, _, payload in records],
    )


def test_memory_from_config_add_search_delete_uses_real_gaussdb_provider():
    collection = _new_collection("mem0_p0_memory")
    vector_config = _gaussdb_env_config(collection)
    assert vector_config is not None

    memory_config = {
        "vector_store": {"provider": "gaussdb", "config": vector_config},
        "embedder": {"provider": "openai", "config": {"model": "fake", "api_key": "fake"}},
        "llm": {"provider": "openai", "config": {"model": "fake", "api_key": "fake"}},
        "version": "v1.1",
    }

    with (
        patch("mem0.memory.main.EmbedderFactory.create", return_value=FakeEmbedder()),
        patch("mem0.memory.main.LlmFactory.create", return_value=MagicMock()),
        patch("mem0.memory.main.SQLiteManager", return_value=MagicMock()),
        patch("mem0.memory.main.extract_entities", return_value=[]),
        patch("mem0.memory.main.capture_event", lambda *args, **kwargs: None),
        patch("mem0.memory.main.MEM0_TELEMETRY", False),
    ):
        memory = Memory.from_config(memory_config)

    try:
        added = memory.add(
            "Alice prefers window seats on morning flights",
            user_id="alice",
            infer=False,
            metadata={"source": "p0-memory"},
        )
        memory_id = added["results"][0]["id"]

        search_result = memory.search("window seat", filters={"user_id": "alice"}, top_k=5, threshold=0)
        rows = search_result["results"]
        assert [row["id"] for row in rows] == [memory_id]
        assert rows[0]["memory"] == "Alice prefers window seats on morning flights"
        assert rows[0]["user_id"] == "alice"
        assert rows[0]["metadata"]["source"] == "p0-memory"

        memory.delete(memory_id)
        assert memory.vector_store.get(memory_id) is None
    finally:
        memory.vector_store.delete_col()
        if getattr(memory, "_entity_store", None) is not None:
            memory.entity_store.delete_col()


def test_provider_crud_batch_upsert_update_and_delete():
    db = _new_db()
    try:
        first_id = _uuid(1)
        second_id = _uuid(2)
        _insert_memories(
            db,
            [
                (
                    first_id,
                    VECTOR_COFFEE,
                    {
                        "data": "Alice drinks latte coffee",
                        "text_lemmatized": "alice drinks latte coffee",
                        "user_id": "alice",
                        "category": "drink",
                    },
                ),
                (
                    second_id,
                    VECTOR_FLIGHT,
                    {
                        "data": "Alice books morning flights",
                        "text_lemmatized": "alice books morning flights",
                        "user_id": "alice",
                        "category": "travel",
                    },
                ),
            ],
        )

        assert db.get(first_id).payload["data"] == "Alice drinks latte coffee"
        search_rows = db.search("coffee", VECTOR_COFFEE, top_k=2, filters={"user_id": "alice"})
        assert _ids(search_rows)[0] == first_id

        db.update(
            first_id,
            vector=VECTOR_WINDOW,
            payload={
                "data": "Alice now prefers a window seat",
                "text_lemmatized": "alice now prefers a window seat",
                "user_id": "alice",
                "category": "travel",
            },
        )
        updated = db.get(first_id)
        assert updated.payload["data"] == "Alice now prefers a window seat"
        assert _ids(db.search("window", VECTOR_WINDOW, top_k=2, filters={"user_id": "alice"}))[0] == first_id

        db.insert(
            ids=[second_id],
            vectors=[VECTOR_AISLE],
            payloads=[
                {
                    "data": "Alice changed to an aisle seat",
                    "text_lemmatized": "alice changed to an aisle seat",
                    "user_id": "alice",
                    "category": "travel",
                }
            ],
        )
        assert db.get(second_id).payload["data"] == "Alice changed to an aisle seat"

        db.delete(first_id)
        assert db.get(first_id) is None
        _assert_exact_ids(_list_rows(db, {"user_id": "alice"}), {second_id})
    finally:
        db.delete_col()


def test_scoped_search_list_and_batch_do_not_cross_tenants():
    db = _new_db()
    try:
        alice_id = _uuid(11)
        bob_id = _uuid(12)
        public_id = _uuid(13)
        _insert_memories(
            db,
            [
                (
                    alice_id,
                    VECTOR_COFFEE,
                    {
                        "data": "Alice likes latte coffee",
                        "text_lemmatized": "alice likes latte coffee",
                        "user_id": "alice",
                        "category": "private",
                    },
                ),
                (
                    bob_id,
                    VECTOR_COFFEE,
                    {
                        "data": "Bob likes latte coffee",
                        "text_lemmatized": "bob likes latte coffee",
                        "user_id": "bob",
                        "category": "private",
                    },
                ),
                (
                    public_id,
                    VECTOR_COFFEE,
                    {
                        "data": "Public coffee note",
                        "text_lemmatized": "public coffee note",
                        "category": "public",
                    },
                ),
            ],
        )

        _assert_exact_ids(db.search("latte", VECTOR_COFFEE, top_k=10, filters={"user_id": "alice"}), {alice_id})
        _assert_exact_ids(_list_rows(db, {"user_id": "alice"}), {alice_id})

        batch_rows = db.search_batch(
            ["latte", "coffee"],
            [VECTOR_COFFEE, VECTOR_COFFEE],
            top_k=10,
            filters={"user_id": "alice"},
        )
        assert [set(_ids(rows)) for rows in batch_rows] == [{alice_id}, {alice_id}]

        valid_or_rows = db.search(
            "latte",
            VECTOR_COFFEE,
            top_k=10,
            filters={"$or": [{"user_id": "alice"}, {"user_id": "bob"}]},
        )
        _assert_exact_ids(valid_or_rows, {alice_id, bob_id})

        with pytest.raises(ValueError, match="requires at least one scoped filter"):
            db.search("latte", VECTOR_COFFEE, top_k=10, filters={"$or": [{"user_id": "alice"}, {"category": "public"}]})

        with pytest.raises(ValueError, match="requires at least one scoped filter"):
            db.search("latte", VECTOR_COFFEE, top_k=10, filters={"user_id": {"ne": "bob"}})

        with pytest.raises(ValueError, match="requires at least one scoped filter"):
            db.list(filters={"category": "public"})
    finally:
        db.delete_col()


def test_json_payload_filter_operator_matrix():
    db = _new_db()
    try:
        travel_id = _uuid(21)
        food_id = _uuid(22)
        work_id = _uuid(23)
        _insert_memories(
            db,
            [
                (
                    travel_id,
                    VECTOR_FLIGHT,
                    {
                        "data": "Plan flight with priority boarding",
                        "text_lemmatized": "plan flight with priority boarding",
                        "user_id": "filter_user",
                        "category": "travel",
                        "priority": 7,
                        "tag": "boarding-plan",
                    },
                ),
                (
                    food_id,
                    VECTOR_COFFEE,
                    {
                        "data": "Find coffee near the office",
                        "text_lemmatized": "find coffee near the office",
                        "user_id": "filter_user",
                        "category": "food",
                        "priority": 2,
                        "tag": "coffee-shop",
                    },
                ),
                (
                    work_id,
                    VECTOR_WINDOW,
                    {
                        "data": "Prepare quarterly plan",
                        "text_lemmatized": "prepare quarterly plan",
                        "user_id": "filter_user",
                        "category": "work",
                        "priority": 5,
                        "tag": "QuarterlyPlan",
                    },
                ),
            ],
        )

        cases = [
            ({"user_id": "filter_user", "category": {"eq": "travel"}}, {travel_id}),
            ({"user_id": "filter_user", "category": {"ne": "travel"}}, {food_id, work_id}),
            ({"user_id": "filter_user", "category": {"in": ["travel", "food"]}}, {travel_id, food_id}),
            ({"user_id": "filter_user", "category": {"nin": ["travel", "food"]}}, {work_id}),
            ({"user_id": "filter_user", "priority": {"gt": 3}}, {travel_id, work_id}),
            ({"user_id": "filter_user", "priority": {"gte": 5, "lte": 7}}, {travel_id, work_id}),
            ({"user_id": "filter_user", "tag": {"contains": "coffee"}}, {food_id}),
            ({"user_id": "filter_user", "tag": {"icontains": "plan"}}, {travel_id, work_id}),
            ({"$and": [{"user_id": "filter_user"}, {"category": "travel"}, {"priority": {"gt": 4}}]}, {travel_id}),
            (
                {
                    "$or": [
                        {"user_id": "filter_user", "category": "travel"},
                        {"user_id": "filter_user", "category": "food"},
                    ]
                },
                {travel_id, food_id},
            ),
            ({"user_id": "filter_user", "$not": [{"category": "food"}]}, {travel_id, work_id}),
        ]

        for filters, expected_ids in cases:
            rows = db.search("filter", VECTOR_COFFEE, top_k=10, filters=filters)
            _assert_exact_ids(rows, expected_ids)
    finally:
        db.delete_col()


def test_compatibility_profile_uses_text_payload_and_preserves_redundant_scope_on_partial_update():
    db = _new_db(profile="compatibility", vector_index_type="gsivfflat")
    try:
        compat_id = _uuid(31)
        _insert_memories(
            db,
            [
                (
                    compat_id,
                    VECTOR_COFFEE,
                    {
                        "data": "Compatibility profile memory",
                        "text_lemmatized": "compatibility profile memory",
                        "user_id": "compat_user",
                        "category": "compat",
                    },
                )
            ],
        )

        info = db.col_info()
        assert info["profile"] == "compatibility"
        assert info["payload_storage_mode"] == "text"
        assert info["filter_storage_mode"] == "redundant_columns"

        _assert_exact_ids(db.search("compat", VECTOR_COFFEE, filters={"user_id": "compat_user"}), {compat_id})
        db.update(
            compat_id,
            payload={"data": "Compatibility payload changed", "text_lemmatized": "compatibility payload changed"},
        )
        rows = db.search("compat", VECTOR_COFFEE, filters={"user_id": "compat_user"})
        _assert_exact_ids(rows, {compat_id})
        assert rows[0].payload["data"] == "Compatibility payload changed"

        with pytest.raises(ValueError, match="not available in filter_storage_mode"):
            db.search("compat", VECTOR_COFFEE, filters={"user_id": "compat_user", "category": "compat"})
    finally:
        db.delete_col()


def test_utf8_chinese_and_mixed_payload_round_trip():
    db = _new_db()
    try:
        if _server_encoding(db) != "UTF8":
            pytest.skip("UTF-8 payload round trip requires a UTF8 GaussDB database")

        chinese_id = _uuid(41)
        mixed_id = _uuid(42)
        _insert_memories(
            db,
            [
                (
                    chinese_id,
                    VECTOR_COFFEE,
                    {
                        "data": "我喜欢早晨喝拿铁咖啡",
                        "text_lemmatized": "我 喜欢 早晨 喝 拿铁 咖啡",
                        "user_id": "zh_user",
                        "language": "zh",
                    },
                ),
                (
                    mixed_id,
                    VECTOR_FLIGHT,
                    {
                        "data": "小李 books flights with priority boarding",
                        "text_lemmatized": "小李 books flights with priority boarding",
                        "user_id": "zh_user",
                        "language": "mixed",
                    },
                ),
            ],
        )

        assert db.get(chinese_id).payload["data"] == "我喜欢早晨喝拿铁咖啡"
        assert db.get(mixed_id).payload["data"] == "小李 books flights with priority boarding"
        _assert_exact_ids(
            db.search("拿铁", VECTOR_COFFEE, top_k=2, filters={"user_id": "zh_user"}), {chinese_id, mixed_id}
        )
        _assert_exact_ids(_list_rows(db, {"user_id": "zh_user"}), {chinese_id, mixed_id})
    finally:
        db.delete_col()


@pytest.mark.parametrize("metric", ["cosine", "l2"])
def test_vector_metric_exact_match_returns_first(metric):
    db = _new_db(vector_metric=metric)
    try:
        exact_id = _uuid(51)
        far_id = _uuid(52)
        _insert_memories(
            db,
            [
                (
                    exact_id,
                    [1.0, 0.0, 0.0],
                    {
                        "data": f"Exact vector for {metric}",
                        "text_lemmatized": f"exact vector for {metric}",
                        "user_id": "metric_user",
                    },
                ),
                (
                    far_id,
                    [0.0, 1.0, 0.0],
                    {
                        "data": f"Far vector for {metric}",
                        "text_lemmatized": f"far vector for {metric}",
                        "user_id": "metric_user",
                    },
                ),
            ],
        )

        rows = db.search("exact", [1.0, 0.0, 0.0], top_k=2, filters={"user_id": "metric_user"})
        assert _ids(rows) == [exact_id, far_id]
        assert rows[0].score >= rows[1].score
    finally:
        db.delete_col()


def test_search_batch_native_results_match_sequential_results():
    db = _new_db()
    try:
        coffee_id = _uuid(61)
        flight_id = _uuid(62)
        _insert_memories(
            db,
            [
                (
                    coffee_id,
                    VECTOR_COFFEE,
                    {
                        "data": "Coffee note",
                        "text_lemmatized": "coffee note",
                        "user_id": "batch_user",
                    },
                ),
                (
                    flight_id,
                    VECTOR_FLIGHT,
                    {
                        "data": "Flight note",
                        "text_lemmatized": "flight note",
                        "user_id": "batch_user",
                    },
                ),
            ],
        )

        batch_rows = db.search_batch(
            ["coffee", "flight"],
            [VECTOR_COFFEE, VECTOR_FLIGHT],
            top_k=1,
            filters={"user_id": "batch_user"},
        )
        sequential_rows = [
            db.search("coffee", VECTOR_COFFEE, top_k=1, filters={"user_id": "batch_user"}),
            db.search("flight", VECTOR_FLIGHT, top_k=1, filters={"user_id": "batch_user"}),
        ]
        assert [[row.id for row in rows] for rows in batch_rows] == [
            [row.id for row in rows] for rows in sequential_rows
        ]
        assert [[row.id for row in rows] for rows in batch_rows] == [[coffee_id], [flight_id]]
    finally:
        db.delete_col()


def test_collection_operations_schema_info_analyze_reset_and_list_cols():
    collection = _new_collection("mem0_p0_ops")
    db = _new_db(collection)
    try:
        _insert_memories(
            db,
            [
                (
                    _uuid(71),
                    VECTOR_COFFEE,
                    {
                        "data": "Operational memory one",
                        "text_lemmatized": "operational memory one",
                        "user_id": "ops_user",
                    },
                ),
                (
                    _uuid(72),
                    VECTOR_FLIGHT,
                    {
                        "data": "Operational memory two",
                        "text_lemmatized": "operational memory two",
                        "user_id": "ops_user",
                    },
                ),
            ],
        )

        info = db.col_info()
        assert info["name"] == collection
        assert info["count"] == 2
        assert info["schema_version"] >= 1
        assert info["payload_storage_mode"] == "jsonb"
        assert info["filter_storage_mode"] == "json_expression"
        assert any("vector_idx" in index for index in info["indexes"])

        db.analyze()
        listed_collections = db.list_cols()
        assert collection in listed_collections
        assert f"{collection}_schema_meta" not in listed_collections

        db.reset()
        assert db.col_info()["count"] == 0
    finally:
        db.delete_col()


def test_migration_dry_run_and_json_backfill_helpers():
    db = _new_db()
    try:
        backfill_id = _uuid(81)
        _insert_memories(
            db,
            [
                (
                    backfill_id,
                    VECTOR_COFFEE,
                    {
                        "data": "Backfill should restore derived text",
                        "text_lemmatized": "backfill should restore derived text",
                        "user_id": "migration_user",
                    },
                )
            ],
        )

        with db._get_cursor(commit=True) as cur:
            cur.execute(f"UPDATE {db.table_name} SET memory = NULL, text_lemmatized = '' WHERE id = %s", (backfill_id,))

        plan = db.migration_dry_run()
        assert plan["collection_name"] == db.collection_name
        assert plan["mutates_data"] is False
        assert "ensure_text_lemmatized_column" in plan["planned_actions"]

        dry_run = db.backfill_derived_fields(dry_run=True)
        assert dry_run["estimated_rows"] >= 1

        result = db.backfill_derived_fields(dry_run=False)
        assert result["affected_rows"] >= 1
        assert db.get(backfill_id).payload["data"] == "Backfill should restore derived text"
    finally:
        db.delete_col()


def test_allowed_filter_keys_and_admin_unscoped_mode_are_explicit():
    scoped_db = _new_db(allowed_filter_keys=["user_id", "agent_id", "run_id", "category"])
    admin_db = _new_db(require_scoped_filters=False)
    try:
        _insert_memories(
            scoped_db,
            [
                (
                    _uuid(91),
                    VECTOR_COFFEE,
                    {
                        "data": "Allowed filter memory",
                        "text_lemmatized": "allowed filter memory",
                        "user_id": "allowed_user",
                        "category": "allowed",
                    },
                )
            ],
        )
        _assert_exact_ids(
            scoped_db.search("allowed", VECTOR_COFFEE, filters={"user_id": "allowed_user", "category": "allowed"}),
            {_uuid(91)},
        )
        with pytest.raises(ValueError, match="Unsupported filter key"):
            scoped_db.search("allowed", VECTOR_COFFEE, filters={"user_id": "allowed_user", "tenant": "forbidden"})

        _insert_memories(
            admin_db,
            [
                (
                    _uuid(92),
                    VECTOR_COFFEE,
                    {
                        "data": "Admin-visible memory",
                        "text_lemmatized": "admin visible memory",
                        "user_id": "admin_user",
                    },
                )
            ],
        )
        assert _ids(admin_db.search("admin", VECTOR_COFFEE, top_k=10, filters=None)) == [_uuid(92)]
        assert _ids(_list_rows(admin_db, filters=None)) == [_uuid(92)]
    finally:
        scoped_db.delete_col()
        admin_db.delete_col()


def test_concurrent_scoped_searches_return_stable_ids():
    db = _new_db(maxconn=5)
    try:
        expected_id = _uuid(101)
        _insert_memories(
            db,
            [
                (
                    expected_id,
                    VECTOR_COFFEE,
                    {
                        "data": "Concurrent search memory",
                        "text_lemmatized": "concurrent search memory",
                        "user_id": "concurrent_user",
                    },
                )
            ],
        )

        def search_once():
            rows = db.search("concurrent", VECTOR_COFFEE, top_k=1, filters={"user_id": "concurrent_user"})
            return _ids(rows)

        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(lambda _: search_once(), range(8)))

        assert results == [[expected_id]] * 8
    finally:
        db.delete_col()


@pytest.mark.skipif(
    not _env_bool("GAUSSDB_TEST_RUN_BM25"),
    reason="Set GAUSSDB_TEST_RUN_BM25=true to run native BM25 live tests",
)
def test_bm25_keyword_search_is_scoped_and_empty_query_returns_empty_list():
    db = _new_db(bm25_mode="required", enable_capability_probe=True)
    try:
        alice_id = _uuid(111)
        bob_id = _uuid(112)
        _insert_memories(
            db,
            [
                (
                    alice_id,
                    VECTOR_COFFEE,
                    {
                        "data": "Alice likes latte coffee",
                        "text_lemmatized": "alice likes latte coffee",
                        "user_id": "bm25_alice",
                    },
                ),
                (
                    bob_id,
                    VECTOR_COFFEE,
                    {
                        "data": "Bob likes latte coffee",
                        "text_lemmatized": "bob likes latte coffee",
                        "user_id": "bm25_bob",
                    },
                ),
            ],
        )

        assert db.keyword_search("", filters={"user_id": "bm25_alice"}) == []
        rows = db.keyword_search("latte", top_k=10, filters={"user_id": "bm25_alice"})
        assert rows is not None
        _assert_exact_ids(rows, {alice_id})
    finally:
        db.delete_col()


@pytest.mark.skipif(
    not _env_bool("GAUSSDB_TEST_RUN_INDEX_MATRIX"),
    reason="Set GAUSSDB_TEST_RUN_INDEX_MATRIX=true to run all vector index and metric combinations",
)
@pytest.mark.parametrize("vector_index_type", ["gsivfflat", "gsdiskann"])
@pytest.mark.parametrize("vector_metric", ["cosine", "l2"])
def test_vector_index_metric_matrix_builds_and_searches(vector_index_type, vector_metric):
    db = _new_db(vector_index_type=vector_index_type, vector_metric=vector_metric)
    try:
        matrix_id = _uuid(121)
        _insert_memories(
            db,
            [
                (
                    matrix_id,
                    VECTOR_COFFEE,
                    {
                        "data": f"{vector_index_type} {vector_metric} matrix memory",
                        "text_lemmatized": f"{vector_index_type} {vector_metric} matrix memory",
                        "user_id": "matrix_user",
                    },
                )
            ],
        )
        assert _ids(db.search("matrix", VECTOR_COFFEE, top_k=1, filters={"user_id": "matrix_user"})) == [matrix_id]
        assert any("vector_idx" in index for index in db.col_info()["indexes"])
    finally:
        db.delete_col()
