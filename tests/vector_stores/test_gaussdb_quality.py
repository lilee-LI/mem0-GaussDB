import json
import os
import statistics
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from mem0.vector_stores.gaussdb import GaussDB


def _gaussdb_env_config(collection_name: str):
    dsn = os.getenv("GAUSSDB_TEST_DSN")
    if dsn:
        return {"connection_string": dsn, "collection_name": collection_name}

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
        }
    return None


pytestmark = pytest.mark.skipif(
    _gaussdb_env_config("probe") is None,
    reason=("Set GAUSSDB_TEST_DSN or GAUSSDB_TEST_HOST/PORT/DATABASE/USER/PASSWORD to run live GaussDB quality tests."),
)


def _load_cases():
    path = Path(__file__).with_name("gaussdb_quality_cases.json")
    return json.loads(path.read_text(encoding="utf-8"))


def _new_db(collection_name: str):
    config = _gaussdb_env_config(collection_name)
    return GaussDB(
        **config,
        embedding_model_dims=3,
        bm25_enabled=True,
        bm25_fail_fast=True,
        require_scoped_filters=True,
        enable_capability_probe=os.getenv("GAUSSDB_TEST_ENABLE_PROBE", "true").lower() == "true",
        retry_attempts=1,
    )


def _seed_cases(db: GaussDB, cases):
    db.insert(
        vectors=[case["vector"] for case in cases],
        ids=[case["id"] for case in cases],
        payloads=[
            {
                "data": case["data"],
                "text_lemmatized": case["text_lemmatized"],
                "user_id": case["user_id"],
                "agent_id": case["agent_id"],
                "run_id": case["run_id"],
            }
            for case in cases
        ],
    )


def test_gaussdb_live_quality_replay_gates():
    cases = _load_cases()
    collection_name = f"mem0_gdb_quality_{uuid.uuid4().hex[:8]}"
    db = _new_db(collection_name)

    try:
        _seed_cases(db, cases)

        semantic_hits = 0
        keyword_hits = 0
        filter_leaks = 0
        for case in cases:
            filters = {"user_id": case["user_id"]}
            semantic = db.search(case["semantic_query"], case["semantic_vector"], top_k=5, filters=filters)
            keyword = db.keyword_search(case["keyword_query"], top_k=5, filters=filters)

            semantic_ids = {item.id for item in semantic}
            keyword_ids = {item.id for item in keyword or []}
            semantic_hits += int(case["id"] in semantic_ids)
            keyword_hits += int(case["id"] in keyword_ids)
            filter_leaks += sum(1 for item in semantic if item.payload["user_id"] != case["user_id"])
            filter_leaks += sum(1 for item in keyword or [] if item.payload["user_id"] != case["user_id"])

        assert semantic_hits / len(cases) >= 0.95
        assert keyword_hits / len(cases) >= 0.90
        assert filter_leaks == 0

        update_case = cases[0]
        db.update(
            update_case["id"],
            payload={
                "data": "Alice now prefers aisle seats",
                "text_lemmatized": "alice now prefer aisle seat",
                "user_id": update_case["user_id"],
                "agent_id": update_case["agent_id"],
                "run_id": update_case["run_id"],
            },
        )
        assert db.get(update_case["id"]).payload["data"] == "Alice now prefers aisle seats"

        db.delete(cases[1]["id"])
        assert db.get(cases[1]["id"]) is None
    finally:
        db.delete_col()


def test_gaussdb_live_concurrent_operations():
    cases = _load_cases()
    collection_name = f"mem0_gdb_concurrent_{uuid.uuid4().hex[:8]}"
    db = _new_db(collection_name)

    try:
        _seed_cases(db, cases)

        def search_case(case):
            return db.search(
                case["semantic_query"], case["semantic_vector"], top_k=3, filters={"user_id": case["user_id"]}
            )

        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(search_case, cases))

        assert len(results) == len(cases)
        assert all(result for result in results)
        assert all(group[0].score > 0 for group in results)
    finally:
        db.delete_col()


@pytest.mark.skipif(
    os.getenv("GAUSSDB_TEST_BENCHMARK", "false").lower() != "true",
    reason="Set GAUSSDB_TEST_BENCHMARK=true to run GaussDB benchmark reporting.",
)
def test_gaussdb_live_benchmark_report():
    cases = _load_cases()
    collection_name = f"mem0_gdb_bench_{uuid.uuid4().hex[:8]}"
    db = _new_db(collection_name)

    try:
        latencies = {"insert": [], "semantic_search": [], "keyword_search": [], "update": [], "delete": []}

        for _ in range(5):
            started = time.perf_counter()
            _seed_cases(db, cases)
            latencies["insert"].append((time.perf_counter() - started) * 1000)

            for case in cases:
                started = time.perf_counter()
                db.search(
                    case["semantic_query"], case["semantic_vector"], top_k=5, filters={"user_id": case["user_id"]}
                )
                latencies["semantic_search"].append((time.perf_counter() - started) * 1000)

                started = time.perf_counter()
                db.keyword_search(case["keyword_query"], top_k=5, filters={"user_id": case["user_id"]})
                latencies["keyword_search"].append((time.perf_counter() - started) * 1000)

                started = time.perf_counter()
                db.update(
                    case["id"],
                    payload={
                        "data": case["data"],
                        "text_lemmatized": case["text_lemmatized"],
                        "user_id": case["user_id"],
                        "agent_id": case["agent_id"],
                        "run_id": case["run_id"],
                    },
                )
                latencies["update"].append((time.perf_counter() - started) * 1000)

            started = time.perf_counter()
            db.delete(cases[-1]["id"])
            latencies["delete"].append((time.perf_counter() - started) * 1000)

        report = {
            key: statistics.quantiles(values, n=20)[-1] if len(values) >= 2 else values[0]
            for key, values in latencies.items()
        }
        assert report["semantic_search"] < float(os.getenv("GAUSSDB_TEST_SEMANTIC_P95_MS", "300"))
        assert report["keyword_search"] < float(os.getenv("GAUSSDB_TEST_KEYWORD_P95_MS", "300"))
    finally:
        db.delete_col()
