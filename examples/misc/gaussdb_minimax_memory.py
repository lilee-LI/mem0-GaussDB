"""Run a real mem0 + GaussDB + MiniMax memory smoke test.

The script intentionally reads all credentials from environment variables.
Do not hard-code database passwords or LLM API keys in this file.

Required environment variables:
  GaussDB:
    GAUSSDB_HOST, GAUSSDB_PORT, GAUSSDB_DATABASE, GAUSSDB_USER, GAUSSDB_PASSWORD

  MiniMax LLM:
    MINIMAX_API_KEY

  Embeddings:
    For --embedder openai:
      EMBEDDING_API_KEY or OPENAI_API_KEY
      EMBEDDING_MODEL, EMBEDDING_DIMS, and optionally EMBEDDING_BASE_URL

Example:
  python examples/misc/gaussdb_minimax_memory.py --reset --details
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from mem0 import Memory


def env(*names: str, default: str | None = None) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return default


def require_env(*names: str) -> str:
    value = env(*names)
    if value is None:
        raise RuntimeError(f"Missing environment variable. Provide one of: {', '.join(names)}")
    return value


def default_embedding_dims(embedder: str) -> int:
    if embedder == "fastembed":
        return int(env("EMBEDDING_DIMS", default="1024"))
    return int(env("EMBEDDING_DIMS", default="1536"))


def build_embedder_config(args: argparse.Namespace) -> dict[str, Any]:
    if args.embedder == "fastembed":
        return {
            "provider": "fastembed",
            "config": {
                "model": env("EMBEDDING_MODEL", default="thenlper/gte-large"),
                "embedding_dims": args.embedding_dims,
            },
        }

    return {
        "provider": "openai",
        "config": {
            "model": env("EMBEDDING_MODEL", default="text-embedding-3-small"),
            "api_key": require_env("EMBEDDING_API_KEY", "OPENAI_API_KEY"),
            "openai_base_url": env("EMBEDDING_BASE_URL", "OPENAI_BASE_URL"),
            "embedding_dims": args.embedding_dims,
        },
    }


def build_memory(args: argparse.Namespace) -> Memory:
    history_path = Path(env("MEM0_HISTORY_DB", default=".mem0-gaussdb-minimax-history.db")).resolve()
    config = {
        "version": "v1.1",
        "history_db_path": str(history_path),
        "llm": {
            "provider": "minimax",
            "config": {
                "model": env("MINIMAX_MODEL", default="MiniMax-M2.7"),
                "api_key": require_env("MINIMAX_API_KEY"),
                "minimax_base_url": env(
                    "MINIMAX_BASE_URL",
                    "MINIMAX_API_BASE",
                    default="https://api.minimax.io/v1",
                ),
                "temperature": float(env("MINIMAX_TEMPERATURE", default="0.1")),
                "max_tokens": int(env("MINIMAX_MAX_TOKENS", default="2000")),
                "top_p": float(env("MINIMAX_TOP_P", default="0.95")),
                "reasoning_split": env("MINIMAX_REASONING_SPLIT", default="true").lower() != "false",
            },
        },
        "embedder": build_embedder_config(args),
        "vector_store": {
            "provider": "gaussdb",
            "config": {
                "host": require_env("GAUSSDB_HOST"),
                "port": int(require_env("GAUSSDB_PORT")),
                "database": require_env("GAUSSDB_DATABASE", "GAUSSDB_DBNAME"),
                "user": require_env("GAUSSDB_USER"),
                "password": require_env("GAUSSDB_PASSWORD"),
                "collection_name": args.collection,
                "embedding_model_dims": args.embedding_dims,
                "profile": env("GAUSSDB_PROFILE", default="commercial"),
                "metadata_mode": env("GAUSSDB_METADATA_MODE", default="auto"),
                "bm25_mode": env("GAUSSDB_BM25_MODE", default="auto"),
                "vector_index_type": env("GAUSSDB_VECTOR_INDEX", default="gsdiskann"),
                "vector_metric": env("GAUSSDB_VECTOR_METRIC", default="cosine"),
                "require_scoped_filters": not args.allow_unscoped,
                "auto_create": True,
            },
        },
    }
    memory = Memory.from_config(config)
    if args.reset:
        memory.vector_store.reset()
    return memory


def print_json(title: str, value: Any) -> None:
    print(f"\n# {title}")
    print(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def llm_smoke(memory: Memory) -> None:
    response = memory.llm.generate_response(
        messages=[
            {"role": "system", "content": "你是一个只返回 JSON 的助手。"},
            {"role": "user", "content": '请返回 {"status":"ok","provider":"minimax"}。'},
        ],
        response_format={"type": "json_object"},
    )
    print_json("MiniMax LLM smoke response", response)


def run_demo(args: argparse.Namespace) -> None:
    memory = build_memory(args)
    if not args.skip_llm_smoke:
        llm_smoke(memory)

    messages = [
        {
            "role": "user",
            "content": "我喜欢早晨喝拿铁咖啡，出差坐飞机时尽量选择安静的靠窗座位。",
        },
        {
            "role": "assistant",
            "content": "好的，我会记住你偏好早晨拿铁，以及飞行时更喜欢安静靠窗座位。",
        },
        {
            "role": "user",
            "content": "我的报销单一般要在季度汇报前完成审批。",
        },
    ]

    add_result = memory.add(
        messages,
        user_id=args.user_id,
        agent_id=args.agent_id,
        run_id=args.run_id,
        metadata={"source": "gaussdb_minimax_demo"},
        infer=not args.no_infer,
    )
    print_json("memory.add result", add_result)

    queries = [
        "我下周出差，座位和咖啡有什么偏好需要记住？",
        "季度汇报前财务流程有什么要注意？",
    ]
    filters = {"user_id": args.user_id}
    if args.agent_id:
        filters["agent_id"] = args.agent_id
    if args.run_id:
        filters["run_id"] = args.run_id

    for query in queries:
        search_result = memory.search(query=query, filters=filters, top_k=args.top_k)
        print_json(f'memory.search result: "{query}"', search_result)

    if args.details:
        print_json("GaussDB collection info", memory.vector_store.col_info())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", default=env("GAUSSDB_COLLECTION", default="mem0_gaussdb_minimax_demo"))
    parser.add_argument("--embedder", choices=["openai", "fastembed"], default=env("MEM0_EMBEDDER", default="openai"))
    parser.add_argument("--embedding-dims", type=int, default=None)
    parser.add_argument("--user-id", default=env("MEM0_USER_ID", default="demo_user"))
    parser.add_argument("--agent-id", default=env("MEM0_AGENT_ID", default="gaussdb_minimax_agent"))
    parser.add_argument("--run-id", default=env("MEM0_RUN_ID", default="manual_smoke"))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--reset", action="store_true", help="Drop and recreate the demo collection before running")
    parser.add_argument("--no-infer", action="store_true", help="Store raw messages without MiniMax extraction")
    parser.add_argument("--skip-llm-smoke", action="store_true", help="Skip the direct MiniMax JSON smoke test")
    parser.add_argument(
        "--allow-unscoped", action="store_true", help="Disable GaussDB scoped-filter guard for local experiments"
    )
    parser.add_argument("--details", action="store_true", help="Print GaussDB collection metadata after the demo")
    args = parser.parse_args()
    args.embedding_dims = args.embedding_dims or default_embedding_dims(args.embedder)
    return args


if __name__ == "__main__":
    run_demo(parse_args())
