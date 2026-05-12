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
    assert cfg.deployment_mode == "centralized"
    assert cfg.distribution_mode == "auto"
    assert cfg.gaussdb_version_baseline == "506"
    assert cfg.vector_index_type == "gsdiskann"
    assert cfg.vector_metric == "cosine"
    assert cfg.vector_index_maintenance_work_mem == "128MB"
    assert cfg.bm25_ranking_metric == 0
    assert cfg.bm25_ncandidates == 128
    assert cfg.payload_storage_mode is None
    assert cfg.filter_storage_mode is None
    assert cfg.profile == "commercial"
    assert cfg.metadata_mode == "auto"
    assert cfg.bm25_mode == "auto"
    assert cfg.require_scoped_filters is True


def test_gaussdb_config_maps_high_level_modes():
    cfg = GaussDBConfig(
        connection_pool=object(),
        profile="compatibility",
        bm25_mode="required",
        embedding_model_dims=512,
        auto_create=False,
        enable_capability_probe=False,
    )

    assert cfg.profile == "compatibility"
    assert cfg.metadata_mode == "compatible"
    assert cfg.payload_storage_mode == "text"
    assert cfg.filter_storage_mode == "redundant_columns"
    assert cfg.vector_index_type == "gsivfflat"
    assert cfg.bm25_enabled is True
    assert cfg.bm25_fail_fast is True


def test_gaussdb_config_low_level_overrides_high_level_defaults():
    cfg = GaussDBConfig(
        connection_pool=object(),
        bm25_enabled=False,
        payload_storage_mode="text",
        filter_storage_mode="redundant_columns",
        auto_create=False,
        enable_capability_probe=False,
    )

    assert cfg.bm25_mode is None
    assert cfg.bm25_enabled is False
    assert cfg.metadata_mode is None
    assert cfg.payload_storage_mode == "text"
    assert cfg.filter_storage_mode == "redundant_columns"


def test_gaussdb_config_rejects_mixed_high_and_low_level_modes():
    with pytest.raises(ValidationError, match="metadata_mode cannot be combined"):
        GaussDBConfig(
            connection_pool=object(),
            metadata_mode="compatible",
            payload_storage_mode="text",
            auto_create=False,
            enable_capability_probe=False,
        )

    with pytest.raises(ValidationError, match="bm25_mode cannot be combined"):
        GaussDBConfig(
            connection_pool=object(),
            bm25_mode="disabled",
            bm25_enabled=True,
            auto_create=False,
            enable_capability_probe=False,
        )


def test_gaussdb_config_accepts_split_metadata_storage_modes():
    cfg = GaussDBConfig(
        connection_pool=object(),
        payload_storage_mode="text",
        filter_storage_mode="redundant_columns",
        auto_create=False,
        enable_capability_probe=False,
    )

    assert cfg.payload_storage_mode == "text"
    assert cfg.filter_storage_mode == "redundant_columns"


def test_gaussdb_config_accepts_distributed_deployment_mode():
    cfg = GaussDBConfig(
        connection_pool=object(),
        deployment_mode="distributed",
        embedding_model_dims=512,
        auto_create=False,
        enable_capability_probe=False,
    )

    assert cfg.deployment_mode == "distributed"
    assert cfg.distribution_mode == "auto"


def test_gaussdb_config_rejects_hash_distribution_for_centralized():
    with pytest.raises(ValidationError, match="deployment_mode='distributed'"):
        GaussDBConfig(
            connection_pool=object(),
            deployment_mode="centralized",
            distribution_mode="hash",
            auto_create=False,
            enable_capability_probe=False,
        )


def test_allowed_filter_keys_must_include_all_scope_keys():
    with pytest.raises(ValueError, match="Unsupported filter key: 'agent_id'"):
        make_gaussdb(allowed_filter_keys=["user_id"])

    db, _, _, _ = make_gaussdb(allowed_filter_keys=["user_id", "agent_id", "run_id", "category"])
    assert db.allowed_filter_keys == {"user_id", "agent_id", "run_id", "category"}


def test_gaussdb_provider_accepts_high_level_modes():
    db, _, _, _ = make_gaussdb(metadata_mode="compatible", bm25_mode="disabled")

    assert db.metadata_mode == "compatible"
    assert db.payload_storage_mode == "text"
    assert db.filter_storage_mode == "redundant_columns"
    assert db.metadata_column_mode == "text"
    assert db.bm25_mode == "disabled"
    assert db.bm25_enabled is False
    assert db.bm25_fail_fast is False


def test_gaussdb_config_rejects_text_payload_with_json_expression_filters():
    with pytest.raises(ValidationError, match="json_expression"):
        GaussDBConfig(
            connection_pool=object(),
            payload_storage_mode="text",
            filter_storage_mode="json_expression",
            auto_create=False,
            enable_capability_probe=False,
        )


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


def test_distributed_create_col_generates_hash_distribution_clauses():
    db, _, _, mock_cursor = make_gaussdb(deployment_mode="distributed", require_scoped_filters=False)

    db.create_col()

    sql = executed_sql(mock_cursor)
    assert db.deployment_mode == "distributed"
    assert db.distribution_mode == "hash"
    assert 'DISTRIBUTE BY HASH ("id")' in sql
    assert 'DISTRIBUTE BY HASH ("collection_name")' in sql


def test_text_payload_mode_uses_redundant_scope_filters():
    db, _, _, mock_cursor = make_gaussdb(metadata_column_mode="text")
    mock_cursor.fetchall.return_value = []

    db.create_col()
    db.search("hello", [0.1, 0.2, 0.3], filters={"user_id": "u1"})

    sql = executed_sql(mock_cursor)
    assert "payload TEXT NOT NULL" in sql
    assert "user_id VARCHAR(128)" in sql
    assert '"user_id" = %s' in sql
    assert "payload->>'user_id'" not in sql
    assert db.payload_storage_mode == "text"
    assert db.filter_storage_mode == "redundant_columns"


def test_capability_probe_sets_vector_index_maintenance_work_mem():
    db, _, _, mock_cursor = make_gaussdb(require_scoped_filters=False)
    mock_cursor.fetchone.return_value = ("on",)

    db._probe_capabilities()

    sql = executed_sql(mock_cursor)
    assert "SHOW enable_vectordb" in sql
    assert "CREATE INDEX" in sql
    assert "INSERT INTO" in sql
    assert "text_lemmatized ### %s AS score" in sql
    mock_cursor.execute.assert_any_call("SET LOCAL maintenance_work_mem = %s", ("128MB",))


def test_capability_probe_uses_distributed_probe_table_suffix():
    db, _, _, mock_cursor = make_gaussdb(deployment_mode="distributed", require_scoped_filters=False)
    mock_cursor.fetchone.return_value = ("on",)

    db._probe_capabilities()

    sql = executed_sql(mock_cursor)
    assert 'DISTRIBUTE BY HASH ("id")' in sql


def test_capability_probe_falls_back_to_text_payload_and_redundant_filters_on_jsonb_failure():
    db, _, _, mock_cursor = make_gaussdb(require_scoped_filters=False)
    mock_cursor.fetchone.return_value = ("on",)

    def execute_side_effect(sql, *args):
        if "CREATE TABLE" in str(sql) and "payload JSONB" in str(sql):
            raise Exception("jsonb type unsupported")

    mock_cursor.execute.side_effect = execute_side_effect

    db._probe_capabilities()

    assert db.payload_storage_mode == "text"
    assert db.filter_storage_mode == "redundant_columns"
    assert db.metadata_column_mode == "text"


def test_capability_probe_falls_back_to_redundant_filters_on_expression_index_failure():
    db, _, _, mock_cursor = make_gaussdb(require_scoped_filters=False)
    mock_cursor.fetchone.return_value = ("on",)

    def execute_side_effect(sql, *args):
        if "payload->>'user_id'" in str(sql):
            raise Exception("expression index unsupported")

    mock_cursor.execute.side_effect = execute_side_effect

    db._probe_capabilities()

    assert db.payload_storage_mode == "jsonb"
    assert db.filter_storage_mode == "redundant_columns"
    assert db.metadata_column_mode == "redundant_columns"


def test_bm25_index_failure_rolls_back_savepoint_and_disables_bm25():
    db, _, _, mock_cursor = make_gaussdb(require_scoped_filters=False)
    mock_cursor.execute.side_effect = [None, Exception("bm25 unsupported"), None, None]

    db._create_bm25_index(mock_cursor, '"test_collection"')

    sql = executed_sql(mock_cursor)
    assert "SAVEPOINT" in sql
    assert "ROLLBACK TO SAVEPOINT" in sql
    assert "RELEASE SAVEPOINT" in sql
    assert db.bm25_enabled is False
    assert db.metrics["gaussdb_fallback_count"] == 1


def test_create_col_keeps_collection_when_optional_bm25_index_fails():
    db, _, mock_conn, mock_cursor = make_gaussdb(require_scoped_filters=False)

    def execute_side_effect(sql, *args):
        if "USING bm25" in str(sql):
            raise Exception("bm25 unsupported")

    mock_cursor.execute.side_effect = execute_side_effect

    db.create_col()

    sql = executed_sql(mock_cursor)
    assert "CREATE TABLE IF NOT EXISTS" in sql
    assert "USING gsdiskann (vector COSINE)" in sql
    assert "USING bm25 (text_lemmatized)" in sql
    assert "ROLLBACK TO SAVEPOINT" in sql
    assert "payload->>'user_id'" in sql
    assert db.bm25_enabled is False
    assert db.metrics["gaussdb_fallback_count"] == 1
    mock_conn.commit.assert_called()


def test_capability_probe_bm25_score_failure_uses_savepoint_fallback():
    db, _, _, mock_cursor = make_gaussdb(require_scoped_filters=False)
    mock_cursor.fetchone.return_value = ("on",)

    def execute_side_effect(sql, *args):
        if "SELECT text_lemmatized ###" in str(sql):
            raise Exception("operator ### unsupported")

    mock_cursor.execute.side_effect = execute_side_effect

    db._probe_capabilities()

    sql = executed_sql(mock_cursor)
    assert "ROLLBACK TO SAVEPOINT" in sql
    assert "payload->>'user_id'" in sql
    assert db.bm25_enabled is False
    assert db.metrics["gaussdb_fallback_count"] == 1


def test_insert_uses_merge_into_and_vector_cast():
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
    merge_args = mock_cursor.execute.call_args_list[-1].args[1]
    assert "MERGE INTO" in sql
    assert "WHEN MATCHED THEN" in sql
    assert "WHEN NOT MATCHED THEN" in sql
    assert mock_cursor.execute.call_count == 1
    assert "%s::FLOATVECTOR" in sql
    assert merge_args[1] == "[0.1,0.2,0.3]"
    assert merge_args[3] == "hello"


def test_insert_many_rows_uses_single_merge_statement():
    db, _, _, mock_cursor = make_gaussdb(require_scoped_filters=False)

    db.insert(
        vectors=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
        payloads=[
            {"data": "a", "text_lemmatized": "a", "user_id": "u1"},
            {"data": "b", "text_lemmatized": "b", "user_id": "u1"},
            {"data": "c", "text_lemmatized": "c", "user_id": "u1"},
        ],
        ids=[
            "11111111-1111-1111-1111-111111111111",
            "22222222-2222-2222-2222-222222222222",
            "33333333-3333-3333-3333-333333333333",
        ],
    )

    calls = mock_cursor.execute.call_args_list
    assert len(calls) == 1
    assert "MERGE INTO" in str(calls[0].args[0])
    assert "WHEN MATCHED THEN" in str(calls[0].args[0])
    assert "WHEN NOT MATCHED THEN" in str(calls[0].args[0])
    assert len(calls[0].args[1]) == 18


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
    ],
)
def test_keyword_search_rejects_non_constraining_scope_filters(filters):
    db, _, _, mock_cursor = make_gaussdb()

    with pytest.raises(ValueError, match="requires at least one scoped filter"):
        db.keyword_search("hello", top_k=3, filters=filters)

    mock_cursor.execute.assert_not_called()


@pytest.mark.parametrize(
    "filters",
    [
        {"OR": [{"user_id": "alice"}, {"category": "public"}]},
        {"$or": [{"user_id": "alice"}, {"category": "public"}]},
        {"NOT": [{"user_id": "alice"}]},
        {"user_id": {"ne": "alice"}},
        {"user_id": {"nin": ["alice"]}},
    ],
)
def test_search_batch_rejects_non_constraining_scope_filters(filters):
    db, _, _, mock_cursor = make_gaussdb()

    with pytest.raises(ValueError, match="requires at least one scoped filter"):
        db.search_batch(["hello"], [[0.1, 0.2, 0.3]], filters=filters)

    mock_cursor.execute.assert_not_called()


@pytest.mark.parametrize(
    "filters",
    [
        {"OR": [{"user_id": "alice"}, {"category": "public"}]},
        {"$or": [{"user_id": "alice"}, {"category": "public"}]},
        {"$or": []},
        {"NOT": [{"user_id": "alice"}]},
        {"user_id": {"ne": "alice"}},
        {"user_id": {"nin": ["alice"]}},
        {"user_id": "*"},
        {"AND": [{"category": "travel"}, {"OR": [{"user_id": "alice"}, {"category": "public"}]}]},
        {"AND": [{"category": "travel"}, {"OR": [{"user_id": "alice"}, {"user_id": {"ne": "bob"}}]}]},
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
        {"OR": [{"user_id": "alice"}, {"user_id": "bob"}]},
        {"$or": [{"user_id": {"eq": "alice"}}, {"agent_id": {"in": ["agent-1", "agent-2"]}}]},
        {"AND": [{"category": "travel"}, {"OR": [{"user_id": "alice"}, {"run_id": "run-1"}]}]},
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
    sql = executed_sql(mock_cursor)
    assert "ROW_NUMBER() OVER" in sql
    assert "PARTITION BY q.query_index" in sql


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


def test_update_vector_only_does_not_touch_payload_or_text_fields():
    db, _, _, mock_cursor = make_gaussdb(require_scoped_filters=False)

    db.update("id1", vector=[0.1, 0.2, 0.3])

    sql = executed_sql(mock_cursor)
    params = mock_cursor.execute.call_args.args[1]
    assert "vector = %s::FLOATVECTOR" in sql
    assert "payload = %s" not in sql
    assert "memory = %s" not in sql
    assert "text_lemmatized = %s" not in sql
    assert params == ("[0.1,0.2,0.3]", "id1")


def test_update_payload_only_refreshes_payload_memory_and_text_fields():
    db, _, _, mock_cursor = make_gaussdb(require_scoped_filters=False)

    db.update("id1", payload={"data": "new", "text_lemmatized": "new lemma"})

    sql = executed_sql(mock_cursor)
    params = mock_cursor.execute.call_args.args[1]
    assert "vector = %s::FLOATVECTOR" not in sql
    assert "payload = %s" in sql
    assert "memory = %s" in sql
    assert "text_lemmatized = %s" in sql
    assert params[-3:] == ("new", "new lemma", "id1")


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


def test_update_payload_without_scope_keys_does_not_touch_redundant_scope_columns():
    db, _, _, mock_cursor = make_gaussdb(
        metadata_column_mode="redundant_columns",
        require_scoped_filters=False,
    )

    db.update("id1", payload={"data": "new", "text_lemmatized": "new"})

    sql = executed_sql(mock_cursor)
    params = mock_cursor.execute.call_args.args[1]
    assert '"user_id" = %s' not in sql
    assert '"agent_id" = %s' not in sql
    assert '"run_id" = %s' not in sql
    assert params[-1] == "id1"


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
    assert info["deployment_mode"] == "centralized"
    assert info["distribution_mode"] == "none"
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
    assert plan["deployment_mode"] == "centralized"
    assert plan["distribution_mode"] == "none"
    assert "ensure_hash_distribution_when_deployment_mode_is_distributed" in plan["planned_actions"]
    assert report == {"dry_run": True, "estimated_rows": 7}


# ============================================================
# Group 1: close() and context manager
# ============================================================


def test_close_calls_closeall_and_nullifies_pool():
    """close() should call pool.closeall() and set pool to None."""
    db, mock_pool, _, _ = make_gaussdb()

    db.close()

    mock_pool.closeall.assert_called_once()
    assert db.connection_pool is None


def test_close_swallows_exception_from_closeall():
    """close() should not raise even if closeall() throws."""
    db, mock_pool, _, _ = make_gaussdb()
    mock_pool.closeall.side_effect = RuntimeError("pool error")

    db.close()  # should not raise

    assert db.connection_pool is None


def test_close_is_idempotent_when_pool_already_none():
    """Calling close() twice should not raise."""
    db, _, _, _ = make_gaussdb()

    db.close()
    db.close()  # should not raise


def test_context_manager_calls_close_on_normal_exit():
    """__exit__ should call close() on normal exit."""
    db, mock_pool, _, _ = make_gaussdb()

    with db:
        pass

    mock_pool.closeall.assert_called_once()
    assert db.connection_pool is None


def test_context_manager_calls_close_on_exception():
    """__exit__ should call close() even when exception occurs."""
    db, mock_pool, _, _ = make_gaussdb()

    with pytest.raises(ValueError):
        with db:
            raise ValueError("boom")

    mock_pool.closeall.assert_called_once()
    assert db.connection_pool is None


# ============================================================
# Group 2: Retry logic
# ============================================================


def test_retryable_error_triggers_retry_and_succeeds():
    """A transient 'connection' error should retry and succeed on 2nd attempt."""
    db, _, _, mock_cursor = make_gaussdb(
        require_scoped_filters=False, retry_attempts=2, retry_backoff_seconds=0.0
    )
    call_count = {"n": 0}

    def side_effect(sql, *args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1 and "DELETE" in str(sql):
            raise Exception("connection reset by peer")
        return None

    mock_cursor.execute.side_effect = side_effect

    db.delete("id1")

    assert call_count["n"] >= 2


def test_non_retryable_error_raises_immediately():
    """A non-retryable error (e.g., 'syntax error') should not retry."""
    db, _, _, mock_cursor = make_gaussdb(
        require_scoped_filters=False, retry_attempts=2, retry_backoff_seconds=0.0
    )
    mock_cursor.execute.side_effect = Exception("syntax error at position 42")

    with pytest.raises(Exception, match="syntax error"):
        db.delete("id1")

    # Should only have been called once (no retry)
    assert mock_cursor.execute.call_count == 1


def test_retry_exhaustion_raises_last_error():
    """After all retry attempts fail, the last exception is raised."""
    db, _, _, mock_cursor = make_gaussdb(
        require_scoped_filters=False, retry_attempts=2, retry_backoff_seconds=0.0
    )
    mock_cursor.execute.side_effect = Exception("connection timeout")

    with pytest.raises(Exception, match="connection timeout"):
        db.delete("id1")


def test_is_retryable_classifies_known_fragments():
    """_is_retryable correctly identifies retryable error messages."""
    assert GaussDB._is_retryable(Exception("connection reset")) is True
    assert GaussDB._is_retryable(Exception("timeout expired")) is True
    assert GaussDB._is_retryable(Exception("deadlock detected")) is True
    assert GaussDB._is_retryable(Exception("lock wait timeout")) is True
    assert GaussDB._is_retryable(Exception("could not serialize access")) is True
    assert GaussDB._is_retryable(Exception("server closed the connection")) is True
    assert GaussDB._is_retryable(Exception("terminating connection")) is True
    assert GaussDB._is_retryable(Exception("syntax error")) is False
    assert GaussDB._is_retryable(Exception("unique violation")) is False


# ============================================================
# Group 3: LIKE escape
# ============================================================


def test_contains_filter_escapes_percent_wildcard():
    """Value containing '%' should be escaped in LIKE pattern."""
    db, _, _, mock_cursor = make_gaussdb(require_scoped_filters=False)
    mock_cursor.fetchall.return_value = []

    db.list(filters={"user_id": {"contains": "100%"}})

    sql = executed_sql(mock_cursor)
    # The percent should be escaped
    assert "LIKE %s ESCAPE" in sql
    # Check the parameter passed
    call_args = mock_cursor.execute.call_args
    params = call_args[0][1] if len(call_args[0]) > 1 else call_args[1].get("params", [])
    assert any("%100\\%%" in str(p) for p in params)


def test_contains_filter_escapes_underscore_wildcard():
    """Value containing '_' should be escaped in LIKE pattern."""
    db, _, _, mock_cursor = make_gaussdb(require_scoped_filters=False)
    mock_cursor.fetchall.return_value = []

    db.list(filters={"user_id": {"contains": "a_b"}})

    call_args = mock_cursor.execute.call_args
    params = call_args[0][1] if len(call_args[0]) > 1 else call_args[1].get("params", [])
    assert any("%a\\_b%" in str(p) for p in params)


def test_icontains_filter_escapes_backslash():
    """Value containing '\\' should be double-escaped."""
    db, _, _, mock_cursor = make_gaussdb(require_scoped_filters=False)
    mock_cursor.fetchall.return_value = []

    db.list(filters={"user_id": {"icontains": "a\\b"}})

    sql = executed_sql(mock_cursor)
    assert "LOWER" in sql
    assert "ESCAPE" in sql
    call_args = mock_cursor.execute.call_args
    params = call_args[0][1] if len(call_args[0]) > 1 else call_args[1].get("params", [])
    assert any("%a\\\\b%" in str(p) for p in params)


# ============================================================
# Group 4: BM25 graceful degradation
# ============================================================


def test_bm25_index_failure_disables_bm25_when_not_fail_fast():
    """BM25 index creation failure should set bm25_enabled=False."""
    db, _, _, mock_cursor = make_gaussdb(require_scoped_filters=False, bm25_fail_fast=False)
    mock_cursor.execute.side_effect = [None, Exception("bm25 unsupported"), None, None]

    db._create_bm25_index(mock_cursor, '"test_collection"')

    assert db.bm25_enabled is False
    assert db.metrics.get("gaussdb_fallback_count", 0) == 1


def test_bm25_index_failure_raises_when_fail_fast():
    """BM25 index creation failure should raise when bm25_fail_fast=True."""
    db, _, _, mock_cursor = make_gaussdb(require_scoped_filters=False, bm25_fail_fast=True)
    mock_cursor.execute.side_effect = [None, Exception("bm25 unsupported"), None, None]

    with pytest.raises(Exception, match="bm25 unsupported"):
        db._create_bm25_index(mock_cursor, '"test_collection"')


# ============================================================
# Group 5: Connection pool safety
# ============================================================


def test_get_cursor_returns_conn_to_pool_on_encoding_failure():
    """Connection must be returned to pool even if set_client_encoding fails."""
    db, mock_pool, mock_conn, _ = make_gaussdb(client_encoding="UTF8")
    mock_conn.set_client_encoding.side_effect = RuntimeError("encoding error")

    with pytest.raises(RuntimeError, match="encoding error"):
        with db._get_cursor() as cur:
            pass

    mock_pool.putconn.assert_called_once_with(mock_conn)


def test_get_cursor_returns_conn_to_pool_on_cursor_exception():
    """Connection must be returned to pool when cursor operation raises."""
    db, mock_pool, mock_conn, mock_cursor = make_gaussdb(require_scoped_filters=False)
    mock_cursor.execute.side_effect = RuntimeError("query failed")

    with pytest.raises(RuntimeError, match="query failed"):
        with db._get_cursor() as cur:
            cur.execute("SELECT 1")

    mock_pool.putconn.assert_called_once_with(mock_conn)


# ============================================================
# Group 6: insert parameter validation
# ============================================================


def test_insert_raises_on_mismatched_vectors_payloads_ids_length():
    """insert() should raise ValueError when input lengths don't match."""
    db, _, _, _ = make_gaussdb(require_scoped_filters=False)

    with pytest.raises(ValueError, match="same length"):
        db.insert(vectors=[[0.1, 0.2, 0.3]], payloads=[{"a": 1}, {"b": 2}])

    with pytest.raises(ValueError, match="same length"):
        db.insert(vectors=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], ids=["id1"])


# ============================================================
# Group 7: _upsert_schema_meta uses MERGE INTO
# ============================================================


def test_upsert_schema_meta_uses_merge_into():
    """_upsert_schema_meta should use MERGE INTO for atomic upsert."""
    db, _, _, mock_cursor = make_gaussdb(require_scoped_filters=False)

    db._upsert_schema_meta(mock_cursor, "test_collection", 3)

    sql = executed_sql(mock_cursor)
    assert "MERGE INTO" in sql
    assert "WHEN MATCHED THEN" in sql
    assert "WHEN NOT MATCHED THEN" in sql


def test_upsert_schema_meta_passes_correct_params():
    """MERGE INTO should receive (collection_name, schema_version) params."""
    db, _, _, mock_cursor = make_gaussdb(require_scoped_filters=False)

    db._upsert_schema_meta(mock_cursor, "my_collection", 5)

    call_args = mock_cursor.execute.call_args
    params = call_args[0][1]
    assert params == ("my_collection", 5)