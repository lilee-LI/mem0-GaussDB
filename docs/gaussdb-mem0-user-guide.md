# mem0 + GaussDB + MiniMax 使用手册

本文面向想手工验证 mem0 使用 GaussDB 作为记忆数据库的开发者。示例使用 GaussDB 集中式 A 模式 Ustore 作为 vector store，使用 MiniMax 作为真实 LLM，embedding 可以选择 OpenAI-compatible embedding 服务或本地 `fastembed`。

## 1. mem0 是什么

mem0 是面向 AI Agent/助手的长期记忆层。它不是普通聊天历史表，而是把用户对话抽取成可检索的“记忆”，并在后续 query 时做多信号召回。

在当前代码里，一次典型 `Memory.add()` 会走这条链路：

1. 接收用户/助手 messages。
2. 用 LLM 从对话里抽取结构化 memory。
3. 用 embedding 模型把 memory 文本转成向量。
4. 写入 vector store，这里就是 GaussDB。
5. 同时写入本地 SQLite history，用于 add 阶段上下文。

一次典型 `Memory.search()` 会走这条链路：

1. 对 query 做 embedding。
2. 在 GaussDB 中做向量召回。
3. 如果 GaussDB BM25 可用，同时做关键词召回。
4. 做 entity boost、BM25 score、semantic score 融合排序。
5. 返回和当前 `user_id`、`agent_id`、`run_id` scope 匹配的记忆。

## 2. 当前 GaussDB 适配能力

GaussDB provider 已接入 mem0 的标准 vector store 工厂：

```python
from mem0 import Memory

memory = Memory.from_config({
    "vector_store": {
        "provider": "gaussdb",
        "config": {...}
    }
})
```

已适配能力：

- 建表：Ustore 表，`FLOATVECTOR` 向量列，payload，`memory`，`text_lemmatized`，schema meta。
- 向量检索：支持 `cosine` 和 `l2`，默认 `cosine`。
- 向量索引：支持 `gsdiskann` 和 `gsivfflat`，默认商业 profile 使用 `gsdiskann`。
- 关键词检索：支持 GaussDB BM25，`keyword_search()` 使用 `text_lemmatized ### query`。
- 混合召回：mem0 上层会融合 semantic + BM25 + entity boost。
- 多租户隔离：默认要求 read/search 路径必须带 `user_id`、`agent_id`、`run_id` 至少一个有效 scope。
- metadata 兼容：支持 JSONB payload，也支持 TEXT payload + 冗余 scope 列 fallback。
- 批量写入：使用 CTE set-based update/insert，不依赖 PostgreSQL `ON CONFLICT`。
- 批量检索：使用 A 模式兼容的 window function 实现，不使用 `LATERAL`。
- 可观测：慢查询日志、fallback 计数、`col_info()`。

## 3. 准备环境

在仓库根目录：

```powershell
cd D:\lxm\code\mem0-GaussDB\mem0_codex\mem0
python -m pip install -e ".[vector_stores,llms,nlp]"
python -m spacy download en_core_web_sm
```

如果想完全不用外部 embedding 服务，可以装本地 embedding：

```powershell
python -m pip install fastembed
```

说明：

- MiniMax 只负责 LLM 抽取记忆。
- mem0 还需要 embedding 模型。你可以用 OpenAI-compatible embedding 服务，也可以用本地 `fastembed`。
- 如果你的 MiniMax 账号也提供 OpenAI-compatible embedding endpoint，可以把它配置到 `EMBEDDING_BASE_URL` 和 `EMBEDDING_MODEL`。

## 4. 配置环境变量

不要把密码或 API key 写进代码。PowerShell 示例：

```powershell
$env:GAUSSDB_HOST='<your-gaussdb-host>'
$env:GAUSSDB_PORT='19995'
$env:GAUSSDB_DATABASE='<your-database>'
$env:GAUSSDB_USER='<your-user>'
$env:GAUSSDB_PASSWORD='<your-password>'
```

MiniMax：

```powershell
$env:MINIMAX_API_KEY='<your-minimax-api-key>'
$env:MINIMAX_MODEL='MiniMax-M2.7'
$env:MINIMAX_MAX_TOKENS='2000'

# 国际站常用：
$env:MINIMAX_BASE_URL='https://api.minimax.io/v1'

# 如果你的账号/网络使用国内 endpoint，可改成：
# $env:MINIMAX_BASE_URL='https://api.minimaxi.com/v1'
```

OpenAI-compatible embedding 示例：

```powershell
$env:EMBEDDING_API_KEY='<your-embedding-api-key>'
$env:EMBEDDING_BASE_URL='<your-openai-compatible-embedding-base-url>'
$env:EMBEDDING_MODEL='<your-embedding-model>'
$env:EMBEDDING_DIMS='1536'
```

如果用本地 `fastembed`：

```powershell
$env:MEM0_EMBEDDER='fastembed'
$env:EMBEDDING_MODEL='thenlper/gte-large'
$env:EMBEDDING_DIMS='1024'
```

## 5. 一键手工验证

仓库里提供了脚本：

[examples/misc/gaussdb_minimax_memory.py](../examples/misc/gaussdb_minimax_memory.py)

第一次建议重置 demo collection：

```powershell
python examples\misc\gaussdb_minimax_memory.py --reset --details
```

使用本地 fastembed：

```powershell
python examples\misc\gaussdb_minimax_memory.py --embedder fastembed --reset --details
```

脚本会做四件事：

1. 初始化 `Memory.from_config()`。
2. 直接调用 MiniMax 做 JSON smoke test，确认 LLM 真的连通。
3. 调用 `memory.add()` 写入中文/英文混合对话记忆。
4. 调用 `memory.search()` 检索“咖啡/座位偏好”和“季度报销审批”。

成功时你应该看到：

- `MiniMax LLM smoke response` 返回 JSON 内容。
- `memory.add result` 中出现新增 memory。
- `memory.search result` 中召回包含“拿铁/靠窗座位/报销审批”等记忆。
- `GaussDB collection info` 显示 `bm25_enabled`、`vector_metric`、`indexes` 等信息。

## 6. 最小 Python 示例

```python
from mem0 import Memory

config = {
    "version": "v1.1",
    "llm": {
        "provider": "minimax",
        "config": {
            "model": "MiniMax-M2.7",
            "api_key": "<read-from-env-in-real-code>",
            "minimax_base_url": "https://api.minimax.io/v1",
            "reasoning_split": True,
        },
    },
    "embedder": {
        "provider": "openai",
        "config": {
            "model": "<embedding-model>",
            "api_key": "<read-from-env-in-real-code>",
            "openai_base_url": "<embedding-base-url>",
            "embedding_dims": 1536,
        },
    },
    "vector_store": {
        "provider": "gaussdb",
        "config": {
            "host": "<gaussdb-host>",
            "port": 19995,
            "database": "<database>",
            "user": "<user>",
            "password": "<password>",
            "collection_name": "mem0_manual_demo",
            "embedding_model_dims": 1536,
            "profile": "commercial",
            "metadata_mode": "auto",
            "bm25_mode": "auto",
            "require_scoped_filters": True,
        },
    },
}

memory = Memory.from_config(config)

memory.add(
    [
        {"role": "user", "content": "我喜欢早晨喝拿铁，出差时更喜欢安静靠窗座位。"},
        {"role": "assistant", "content": "好的，我会记住这些偏好。"},
    ],
    user_id="alice",
    agent_id="travel_agent",
    run_id="manual_test",
)

result = memory.search(
    "我下次出差时有什么座位和饮品偏好？",
    filters={"user_id": "alice", "agent_id": "travel_agent", "run_id": "manual_test"},
    top_k=5,
)
print(result)
```

## 7. Scope 隔离怎么用

GaussDB provider 默认 `require_scoped_filters=True`。这意味着 search/list/keyword_search 必须带至少一个有效 scope：

```python
memory.search("咖啡偏好", filters={"user_id": "alice"})
```

不建议这么做：

```python
memory.search("咖啡偏好", filters=None)
```

商用场景建议固定带：

```python
filters = {
    "user_id": "<tenant-or-user-id>",
    "agent_id": "<agent-id>",
    "run_id": "<session-or-workflow-id>",
}
```

## 8. 常见问题

### 8.1 MiniMax smoke test 成功，但 `memory.add result` 为空

这通常是 LLM 没抽取出 memory。先看 MiniMax 返回是否是合法 JSON。也可以用：

```powershell
python examples\misc\gaussdb_minimax_memory.py --no-infer --reset
```

`--no-infer` 会跳过 LLM 抽取，把原始 message 直接写入 GaussDB，用于确认 DB 和 embedding 链路。

### 8.2 embedding 维度不匹配

GaussDB collection 的 `embedding_model_dims` 必须和 embedding 模型输出维度一致。换 embedding 模型后，建议换 collection 或执行：

```powershell
python examples\misc\gaussdb_minimax_memory.py --reset --embedding-dims <dims>
```

### 8.3 search 报 scope 相关错误

确认 search 时传了 `filters={"user_id": "..."}`，或者脚本没有误加 `--allow-unscoped`。

### 8.4 BM25 不可用

如果 `col_info()` 里 `bm25_enabled=false`，说明当前库 BM25 建索引或 score probe 没通过。你仍然可以用向量召回；若要强制 BM25 必须可用：

```powershell
$env:GAUSSDB_BM25_MODE='required'
python examples\misc\gaussdb_minimax_memory.py --reset
```

### 8.5 MiniMax endpoint

当前 provider 支持：

- `MINIMAX_BASE_URL`
- `MINIMAX_API_BASE`
- config 中的 `minimax_base_url`

默认是 `https://api.minimax.io/v1`。

### 8.6 MiniMax 返回空内容或只有解释文本

优先确认：

- `MINIMAX_MAX_TOKENS` 不要太小，建议先用 `2000`。
- `MINIMAX_REASONING_SPLIT` 默认开启；如果你的网关不支持，可设为 `false`。
- `MINIMAX_MODEL` 建议先用 `MiniMax-M2.7` 做 smoke test，再替换成你的目标模型。

## 9. 推荐验证顺序

1. `--no-infer --reset --details`：先验证 GaussDB + embedding 写入/检索。
2. 去掉 `--no-infer`：验证 MiniMax 抽取记忆。
3. 多跑几条中文、中英混合、业务相近样例：看 semantic/BM25 融合效果。
4. 换不同 `user_id`：验证 tenant 隔离。
5. 用 `scripts/compare_gaussdb_pgvector.py --scenario complex --details`：对比 GaussDB 与 pgvector provider-level 召回效果。
