import pytest
from pydantic import ValidationError
from unittest.mock import MagicMock

from mem0.configs.vector_stores.gaussdb import GaussDBConfig
from mem0.utils.factory import VectorStoreFactory
from mem0.vector_stores.configs import VectorStoreConfig
from mem0.vector_stores.gaussdb import GaussDB


def make_gaussdb(**kwargs):
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    mock_pool = MagicMock()
    mock_pool.getconn.return_value = mock_conn

    config = {
        "connection_pool": mock_pool,
        "collection_name": "test_collection",
        "embedding_model_dims": 3,
        "enable_capability_probe": False,
        "auto_create": False,
    }
    config.update(kwargs)
    db = GaussDB(**config)
    return db, mock_pool, mock_conn, mock_cursor


def executed_sql(mock_cursor):
    return "\n".join(str(call.args[0]) for call in mock_cursor.execute.call_args_list)


def test_gaussdb_config_defaults_and_alias():
    cfg = GaussDBConfig(
        dbname="mem0db",
        connection_pool=object(),
        auto_create=False,
        enable_capability_probe=False,
    )

    assert cfg.database == "mem0db"
    assert cfg.table_storage == "ustore"
    assert cfg.compatibility_mode == "A"
    assert cfg.gaussdb_version_baseline == "506"
    assert cfg.vector_index_type == "gsdiskann"
    assert cfg.vector_metric == "cosine"
    assert cfg.vector_index_maintenance_work_mem == "128MB"
    assert cfg.bm25_ranking_metric == 0
    assert cfg.bm25_ncandidates == 128
    assert cfg.require_scoped_filters is True


def test_gaussdb_config_accepts_dsn_alias():
    cfg = GaussDBConfig(dsn="postgresql://user:pass@localhost:19995/mem0db")

    assert cfg.connection_string == "postgresql://user:pass@localhost:19995/mem0db"


def test_gaussdb_config_reads_connection_from_env(monkeypatch):
    monkeypatch.setenv("GAUSSDB_HOST", "db.example.com")
    monkeypatch.setenv("GAUSSDB_PORT", "19995")
    monkeypatch.setenv("GAUSSDB_DATABASE", "mem0db")
    monkeypatch.setenv("GAUSSDB_USER", "mem0_user")
    monkeypatch.setenv("GAUSSDB_PASSWORD", "secret")

    cfg = GaussDBConfig()

    assert cfg.host == "db.example.com"
    assert cfg.port == 19995
    assert cfg.database == "mem0db"
    assert cfg.user == "mem0_user"
    assert cfg.password == "secret"


def test_gaussdb_config_rejects_extra_fields():
    with pytest.raises(ValidationError):
        GaussDBConfig(connection_pool=object(), unexpected=True)


def test_vector_store_config_and_factory_register_gaussdb():
    cfg = VectorStoreConfig(
        provider="gaussdb",
        config={
            "connection_pool": object(),
            "auto_create": False,
            "enable_capability_probe": False,
        },
    )

    assert isinstance(cfg.config, GaussDBConfig)
    assert VectorStoreFactory.provider_to_class["gaussdb"] == "mem0.vector_stores.gaussdb.GaussDB"


def test_factory_creates_gaussdb_instance():
    db, mock_pool, _, _ = make_gaussdb()
    created = VectorStoreFactory.create(
        "gaussdb",
        {
            "connection_pool": mock_pool,
            "collection_name": "test_collection",
            "embedding_model_dims": 3,
            "enable_capability_probe": False,
            "auto_create": False,
        },
    )

    assert isinstance(created, GaussDB)
    assert created.collection_name == db.collection_name


def test_rejects_unsafe_identifier():
    with pytest.raises(ValueError, match="Unsafe collection_name"):
        make_gaussdb(collection_name='bad";drop')


def test_create_col_generates_ustore_vector_bm25_and_filter_indexes():
    db, _, mock_conn, mock_cursor = make_gaussdb(require_scoped_filters=False)

    db.create_col()

    sql = executed_sql(mock_cursor)
    assert "WITH (storage_type=ustore)" in sql
    assert "FLOATVECTOR(3)" in sql
    assert "SET LOCAL maintenance_work_mem" in sql
    assert "USING gsdiskann (vector COSINE)" in sql
    assert "USING bm25 (text_lemmatized)" in sql
    assert "storage_type='USTORE'" in sql
    assert "payload->>'user_id'" in sql
    mock_cursor.execute.assert_any_call("SET LOCAL maintenance_work_mem = %s", ("128MB",))
    mock_conn.commit.assert_called()


def test_capability_probe_sets_vector_index_maintenance_work_mem():
    db, _, _, mock_cursor = make_gaussdb(require_scoped_filters=False)
    mock_cursor.fetchone.return_value = ("on",)

    db._probe_capabilities()

    sql = executed_sql(mock_cursor)
    assert "SHOW enable_vectordb" in sql
    assert "CREATE INDEX" in sql
    mock_cursor.execute.assert_any_call("SET LOCAL maintenance_work_mem = %s", ("128MB",))


def test_insert_uses_upsert_and_vector_cast():
    db, _, _, mock_cursor = make_gaussdb(require_scoped_filters=False)

    db.insert(
        vectors=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        payloads=[
            {"data": "hello", "text_lemmatized": "hello", "user_id": "u1"},
            {"data": "world", "text_lemmatized": "world", "user_id": "u1"},
        ],
        ids=["11111111-1111-1111-1111-111111111111", "22222222-2222-2222-2222-222222222222"],
    )

    sql = executed_sql(mock_cursor)
    insert_args = mock_cursor.execute.call_args_list[-1].args[1]
    assert "UPDATE" in sql
    assert "INSERT INTO" in sql
    assert "WITH incoming" in sql
    assert "FROM incoming" in sql
    assert mock_cursor.execute.call_count == 2
    assert "%s::FLOATVECTOR" in sql
    assert insert_args[1] == "[0.1,0.2,0.3]"
    assert insert_args[3] == "hello"


def test_search_uses_cosine_operator_filters_and_normalized_score():
    db, _, _, mock_cursor = make_gaussdb()
    mock_cursor.fetchall.return_value = [("id1", 0.25, {"data": "hello", "user_id": "u1"})]

    results = db.search("hello", [0.1, 0.2, 0.3], top_k=5, filters={"user_id": "u1"})

    sql = executed_sql(mock_cursor)
    assert "vector <+> %s::FLOATVECTOR AS distance" in sql
    assert "payload->>'user_id' = %s" in sql
    assert results[0].id == "id1"
    assert results[0].score == pytest.approx(0.8)
    assert results[0].payload["data"] == "hello"


def test_search_requires_scoped_filters_by_default():
    db, _, _, _ = make_gaussdb()

    with pytest.raises(ValueError, match="requires at least one scoped filter"):
        db.search("hello", [0.1, 0.2, 0.3], filters={"category": "test"})


@pytest.mark.parametrize(
    "filters",
    [
        {"OR": [{"user_id": "alice"}, {"category": "public"}]},
        {"$or": [{"user_id": "alice"}, {"category": "public"}]},
        {"NOT": [{"user_id": "alice"}]},
        {"user_id": {"ne": "alice"}},
        {"user_id": {"nin": ["alice"]}},
        {"user_id": "*"},
        {"AND": [{"category": "travel"}, {"OR": [{"user_id": "alice"}, {"category": "public"}]}]},
    ],
)
def test_search_rejects_non_constraining_scope_filters(filters):
    db, _, _, _ = make_gaussdb()

    with pytest.raises(ValueError, match="requires at least one scoped filter"):
        db.search("hello", [0.1, 0.2, 0.3], filters=filters)


@pytest.mark.parametrize(
    "filters",
    [
        {"user_id": "alice"},
        {"user_id": {"eq": "alice"}},
        {"user_id": {"in": ["alice", "bob"]}},
        {"AND": [{"category": "travel"}, {"user_id": "alice"}]},
        {"$and": [{"category": "travel"}, {"user_id": {"eq": "alice"}}]},
    ],
)
def test_search_accepts_positive_constraining_scope_filters(filters):
    db, _, _, mock_cursor = make_gaussdb()
    mock_cursor.fetchall.return_value = []

    assert db.search("hello", [0.1, 0.2, 0.3], filters=filters) == []
    assert mock_cursor.execute.called


def test_filter_builder_rejects_unsafe_keys():
    db, _, _, _ = make_gaussdb(require_scoped_filters=False)

    with pytest.raises(ValueError, match="Unsafe filter key"):
        db.list(filters={"bad-key": "x"})


def test_keyword_search_uses_bm25_defaults_and_filters():
    db, _, _, mock_cursor = make_gaussdb()
    mock_cursor.fetchall.return_value = [("id1", 2.5, {"data": "hello", "user_id": "u1"})]

    results = db.keyword_search("hello", top_k=3, filters={"user_id": "u1"})

    sql = executed_sql(mock_cursor)
    assert "SET LOCAL bm25_ranking_metric = 0" in sql
    assert "SET LOCAL bm25_ncandidates = 128" in sql
    assert "SET LOCAL enable_seqscan = off" in sql
    assert "text_lemmatized ### %s AS score" in sql
    assert "ORDER BY score DESC" in sql
    assert results[0].score == 2.5


def test_keyword_search_empty_query_returns_empty_list():
    db, _, _, mock_cursor = make_gaussdb()

    assert db.keyword_search(" ", filters={"user_id": "u1"}) == []
    mock_cursor.execute.assert_not_called()


def test_keyword_search_returns_none_when_bm25_disabled():
    db, _, _, _ = make_gaussdb(bm25_enabled=False)

    assert db.keyword_search("hello", filters={"user_id": "u1"}) is None


def test_search_batch_returns_one_result_list_per_query():
    db, _, _, mock_cursor = make_gaussdb()
    mock_cursor.fetchall.return_value = [
        (0, "id1", 0.1, {"data": "a", "user_id": "u1"}),
        (1, "id2", 0.2, {"data": "b", "user_id": "u1"}),
    ]

    results = db.search_batch(
        queries=["a", "b"],
        vectors_list=[[0.1, 0.2, 0.3], [0.3, 0.2, 0.1]],
        filters={"user_id": "u1"},
    )

    assert len(results) == 2
    assert results[0][0].id == "id1"
    assert results[1][0].id == "id2"
    assert "CROSS JOIN LATERAL" in executed_sql(mock_cursor)


def test_search_batch_falls_back_to_sequential_when_native_batch_fails():
    db, _, _, mock_cursor = make_gaussdb()
    mock_cursor.execute.side_effect = [Exception("lateral unsupported"), None, None]
    mock_cursor.fetchall.side_effect = [
        [("id1", 0.1, {"data": "a", "user_id": "u1"})],
        [("id2", 0.2, {"data": "b", "user_id": "u1"})],
    ]

    results = db.search_batch(
        queries=["a", "b"],
        vectors_list=[[0.1, 0.2, 0.3], [0.3, 0.2, 0.1]],
        filters={"user_id": "u1"},
    )

    assert len(results) == 2
    assert results[0][0].id == "id1"
    assert results[1][0].id == "id2"
    assert db.metrics["gaussdb_fallback_count"] == 1


def test_update_vector_and_payload_updates_timestamp():
    db, _, _, mock_cursor = make_gaussdb(require_scoped_filters=False)

    db.update("id1", vector=[0.1, 0.2, 0.3], payload={"data": "new", "text_lemmatized": "new"})

    sql = executed_sql(mock_cursor)
    assert 'UPDATE "test_collection"' in sql
    assert "vector = %s::FLOATVECTOR" in sql
    assert "payload = %s" in sql
    assert "updated_at = CURRENT_TIMESTAMP" in sql


def test_update_payload_preserves_missing_redundant_scope_columns():
    db, _, _, mock_cursor = make_gaussdb(
        metadata_column_mode="redundant_columns",
        require_scoped_filters=False,
    )

    db.update("id1", payload={"data": "new", "text_lemmatized": "new", "user_id": "u2"})

    sql = executed_sql(mock_cursor)
    params = mock_cursor.execute.call_args.args[1]
    assert '"user_id" = %s' in sql
    assert '"agent_id" = %s' not in sql
    assert '"run_id" = %s' not in sql
    assert params[-2:] == ("u2", "id1")


def test_delete_is_idempotent_sql_path():
    db, _, mock_conn, mock_cursor = make_gaussdb(require_scoped_filters=False)

    db.delete("id1")

    sql = executed_sql(mock_cursor)
    assert 'DELETE FROM "test_collection" WHERE id = %s' in sql
    mock_conn.commit.assert_called()


def test_list_returns_wrapped_results():
    db, _, _, mock_cursor = make_gaussdb()
    mock_cursor.fetchall.return_value = [("id1", {"data": "hello", "user_id": "u1"})]

    results = db.list(filters={"user_id": "u1"})

    assert isinstance(results, list)
    assert isinstance(results[0], list)
    assert results[0][0].id == "id1"


def test_col_info_reads_schema_version_from_metadata_table():
    db, _, _, mock_cursor = make_gaussdb(require_scoped_filters=False)
    mock_cursor.fetchone.side_effect = [(3,), (True,), (7,)]
    mock_cursor.fetchall.return_value = [("test_collection_vector_idx",), ("test_collection_bm25_idx",)]

    info = db.col_info()

    sql = executed_sql(mock_cursor)
    assert "information_schema.tables" in sql
    assert 'FROM "test_collection_schema_meta"' in sql
    assert info["count"] == 3
    assert info["schema_version"] == 7
    assert info["indexes"] == ["test_collection_vector_idx", "test_collection_bm25_idx"]


def test_col_info_defaults_schema_version_when_metadata_table_is_missing():
    db, _, _, mock_cursor = make_gaussdb(require_scoped_filters=False)
    mock_cursor.fetchone.side_effect = [(3,), (False,)]
    mock_cursor.fetchall.return_value = []

    info = db.col_info()

    assert info["schema_version"] == 1


def test_transaction_rollback_on_error():
    db, _, mock_conn, mock_cursor = make_gaussdb(require_scoped_filters=False)
    mock_cursor.execute.side_effect = Exception("Database error")

    with pytest.raises(Exception, match="Database error"):
        db.delete("id1")

    mock_conn.rollback.assert_called()


def test_migration_dry_run_and_backfill_report():
    db, _, _, mock_cursor = make_gaussdb(require_scoped_filters=False)
    mock_cursor.fetchone.return_value = (7,)

    plan = db.migration_dry_run()
    report = db.backfill_derived_fields(dry_run=True)

    assert plan["mutates_data"] is False
    assert report == {"dry_run": True, "estimated_rows": 7}
