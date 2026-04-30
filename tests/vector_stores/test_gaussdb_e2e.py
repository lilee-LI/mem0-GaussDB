import os
import uuid

import pytest

from mem0.vector_stores.gaussdb import GaussDB


def _gaussdb_env_config(collection_name: str):
    deployment_config = {
        "deployment_mode": os.getenv("GAUSSDB_TEST_DEPLOYMENT_MODE", "centralized"),
        "distribution_mode": os.getenv("GAUSSDB_TEST_DISTRIBUTION_MODE", "auto"),
    }
    dsn = os.getenv("GAUSSDB_TEST_DSN")
    if dsn:
        return {"connection_string": dsn, "collection_name": collection_name, **deployment_config}

    required = {
        "host": os.getenv("GAUSSDB_TEST_HOST"),
        "port": os.getenv("GAUSSDB_TEST_PORT"),
        "database": os.getenv("GAUSSDB_TEST_DATABASE"),
        "user": os.getenv("GAUSSDB_TEST_USER"),
        "password": os.getenv("GAUSSDB_TEST_PASSWORD"),
    }
    if all(required.values()):
        return {
            "host": required["host"],
            "port": int(required["port"]),
            "database": required["database"],
            "user": required["user"],
            "password": required["password"],
            "collection_name": collection_name,
            "sslmode": os.getenv("GAUSSDB_TEST_SSLMODE"),
            "sslrootcert": os.getenv("GAUSSDB_TEST_SSLROOTCERT"),
            **deployment_config,
        }
    return None


pytestmark = pytest.mark.skipif(
    _gaussdb_env_config("probe") is None,
    reason=(
        "Set GAUSSDB_TEST_DSN or GAUSSDB_TEST_HOST/PORT/DATABASE/USER/PASSWORD to run live GaussDB vector store tests."
    ),
)


def test_gaussdb_live_ustore_vector_bm25_crud_and_batch_search():
    collection_name = f"mem0_gdb_e2e_{uuid.uuid4().hex[:8]}"
    config = _gaussdb_env_config(collection_name)
    db = GaussDB(
        **config,
        embedding_model_dims=3,
        vector_index_type=os.getenv("GAUSSDB_TEST_VECTOR_INDEX", "gsdiskann"),
        vector_metric=os.getenv("GAUSSDB_TEST_VECTOR_METRIC", "cosine"),
        bm25_enabled=True,
        bm25_fail_fast=True,
        metadata_column_mode=os.getenv("GAUSSDB_TEST_METADATA_MODE", "jsonb"),
        require_scoped_filters=True,
        enable_capability_probe=os.getenv("GAUSSDB_TEST_ENABLE_PROBE", "true").lower() == "true",
        retry_attempts=1,
    )

    try:
        db.insert(
            vectors=[[0.1, 0.2, 0.3], [0.9, 0.1, 0.1]],
            ids=[
                "11111111-1111-1111-1111-111111111111",
                "22222222-2222-2222-2222-222222222222",
            ],
            payloads=[
                {
                    "data": "Alice prefers window seats on morning flights",
                    "text_lemmatized": "alice prefer window seat morning flight",
                    "user_id": "alice",
                    "agent_id": "agent1",
                    "run_id": "run1",
                },
                {
                    "data": "Bob likes aisle seats on evening trains",
                    "text_lemmatized": "bob like aisle seat evening train",
                    "user_id": "bob",
                    "agent_id": "agent1",
                    "run_id": "run1",
                },
            ],
        )

        semantic = db.search(
            "window seat",
            [0.1, 0.2, 0.3],
            top_k=5,
            filters={"user_id": "alice"},
        )
        assert semantic
        assert all(item.payload["user_id"] == "alice" for item in semantic)
        assert semantic[0].score > 0

        keyword = db.keyword_search("window seat", top_k=5, filters={"user_id": "alice"})
        assert keyword is not None
        assert keyword
        assert all(item.payload["user_id"] == "alice" for item in keyword)

        batch = db.search_batch(
            ["window seat", "aisle seat"],
            [[0.1, 0.2, 0.3], [0.9, 0.1, 0.1]],
            top_k=1,
            filters={"agent_id": "agent1"},
        )
        assert len(batch) == 2
        assert all(len(group) <= 1 for group in batch)

        listed = db.list(filters={"run_id": "run1"}, top_k=10)
        assert len(listed[0]) == 2

        db.update(
            "11111111-1111-1111-1111-111111111111",
            payload={
                "data": "Alice prefers quiet window seats",
                "text_lemmatized": "alice prefer quiet window seat",
                "user_id": "alice",
                "agent_id": "agent1",
                "run_id": "run1",
            },
        )
        updated = db.get("11111111-1111-1111-1111-111111111111")
        assert updated.payload["data"] == "Alice prefers quiet window seats"

        db.delete("22222222-2222-2222-2222-222222222222")
        assert db.get("22222222-2222-2222-2222-222222222222") is None

        info = db.col_info()
        assert info["name"] == collection_name
        assert info["bm25_enabled"] is True

        dry_run = db.migration_dry_run()
        assert dry_run["mutates_data"] is False
    finally:
        db.delete_col()


def test_gaussdb_live_redundant_scope_survives_partial_payload_update():
    collection_name = f"mem0_gdb_scope_{uuid.uuid4().hex[:8]}"
    config = _gaussdb_env_config(collection_name)
    db = GaussDB(
        **config,
        embedding_model_dims=3,
        vector_index_type=os.getenv("GAUSSDB_TEST_VECTOR_INDEX", "gsdiskann"),
        vector_metric=os.getenv("GAUSSDB_TEST_VECTOR_METRIC", "cosine"),
        bm25_enabled=False,
        metadata_column_mode="redundant_columns",
        require_scoped_filters=True,
        enable_capability_probe=os.getenv("GAUSSDB_TEST_ENABLE_PROBE", "true").lower() == "true",
        retry_attempts=1,
    )

    vector_id = "33333333-3333-3333-3333-333333333333"
    try:
        db.insert(
            vectors=[[0.2, 0.3, 0.4]],
            ids=[vector_id],
            payloads=[
                {
                    "data": "Alice likes compact window seats",
                    "text_lemmatized": "alice like compact window seat",
                    "user_id": "alice",
                    "agent_id": "agent1",
                    "run_id": "run1",
                }
            ],
        )

        db.update(
            vector_id,
            payload={
                "data": "Alice likes quiet window seats",
                "text_lemmatized": "alice like quiet window seat",
            },
        )

        semantic = db.search("window seat", [0.2, 0.3, 0.4], top_k=5, filters={"user_id": "alice"})
        assert [item.id for item in semantic] == [vector_id]

        listed = db.list(filters={"agent_id": "agent1"}, top_k=5)
        assert [item.id for item in listed[0]] == [vector_id]

        other_tenant = db.search("window seat", [0.2, 0.3, 0.4], top_k=5, filters={"user_id": "bob"})
        assert other_tenant == []
    finally:
        db.delete_col()
