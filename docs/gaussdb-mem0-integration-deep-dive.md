# mem0 GaussDB 适配实现深度说明与同类 Provider 对比

> 生成日期：2026-04-28  
> 代码基线：`codex/add-gaussdb-ustore-provider`，当前提交 `dd33a16f`  
> 范围：基于当前仓库代码分析 `mem0` 执行链路、GaussDB provider 设计逻辑、已适配功能点、测试覆盖，以及与 mem0 已有 vector store provider 的能力对比。

## 1. 总体结论

当前 GaussDB 适配已经不是“只实现基础向量 CRUD”的 provider，而是按 mem0 v1.1/v3 检索链路做了完整商用化补齐：

- 接入 mem0 标准 `VectorStoreBase` 必选接口：`create_col`、`insert`、`search`、`delete`、`update`、`get`、`list_cols`、`delete_col`、`col_info`、`list`、`reset`。
- 接入 mem0 可选增强接口：`keyword_search` 和 `search_batch`。在当前 mem0 25 个 vector store provider 中，只有 `qdrant` 和 `gaussdb` 同时 override 了 `keyword_search` 与 `search_batch`。
- 针对 GaussDB 商用形态额外补齐：集中式、分布式兼容模式、A 兼容模式、Ustore、`FLOATVECTOR`、`gsdiskann`/`gsivfflat`、原生 BM25、租户 scope 强制隔离、能力探测、事务 savepoint 保护、schema meta、backfill、连接池、重试、观测指标和 live P0/P1/P2 验证。
- 适配深度对标 `pgvector`，但在以下方面已经超过当前 mem0 `pgvector.py`：原生 batch search、scope guard 防绕过、JSONB 失败 fallback、TEXT payload + redundant scope columns 兼容模式、BM25 建索引事务保护、schema version 元数据、live 能力矩阵测试。

如果用“mem0 现有 provider 适配成熟度”来衡量，GaussDB 当前属于高适配度，功能面已经接近或超过 `pgvector`，并在商用可交付性上更重。

### 1.1 mem0 的具体作用

mem0 是面向 AI 应用的 memory layer。它本身不是一个向量数据库，也不是单纯的 embedding wrapper，而是把“用户长期记忆、会话上下文、事实抽取、向量检索、关键词检索、更新/删除历史”组织成一套可被应用直接调用的记忆系统。

在典型 AI 应用里，大模型本身没有可靠的长期状态。mem0 的作用是把对话或业务事件中值得保留的信息抽取出来，存成可检索的 memory，并在后续请求中根据用户、agent、run 等 scope 找回相关 memory，作为个性化上下文或业务上下文提供给 LLM。

mem0 在系统中通常承担这些职责：

| 职责 | 说明 |
|---|---|
| 记忆写入 | 接收用户消息、助手消息或业务文本，抽取或直接写入 memory。 |
| 事实抽取 | `infer=True` 时使用 LLM 从原始消息中抽取更稳定的事实型 memory。 |
| 向量化 | 使用 embedder 把 memory 文本转成 embedding。 |
| 持久化 | 通过 vector store provider 保存 id、vector、payload、scope metadata。 |
| 语义检索 | 根据 query embedding 找回语义相似 memory。 |
| 关键词检索 | 如果 provider 支持 `keyword_search`，mem0 会把 BM25/全文检索结果纳入候选。 |
| 融合排序 | 合并 semantic score、keyword score、entity boost，输出最终 memory 列表。 |
| 隔离与管理 | 通过 `user_id`、`agent_id`、`run_id` 区分不同用户、智能体或运行会话。 |
| 更新/删除 | 支持 memory 更新、删除、删除某个 scope 下全部 memory。 |
| 历史追踪 | 通过 history DB 记录 memory add/update/delete 历史。 |

一句话概括：mem0 负责“什么应该被记住、怎么存、怎么找、怎么更新”；GaussDB provider 负责“把这些 memory 用 GaussDB 的表、向量索引、BM25、filter 和事务能力可靠落地”。

### 1.2 mem0 怎么使用

最常见的使用方式是通过 `Memory.from_config()` 初始化一个 memory 实例，然后调用 `add/search/update/delete`。

示例配置，使用 GaussDB 作为 vector store：

```python
from mem0 import Memory

config = {
    "vector_store": {
        "provider": "gaussdb",
        "config": {
            "host": "<gaussdb-host>",
            "port": 19995,
            "database": "<database>",
            "user": "<user>",
            "password": "<password>",
            "collection_name": "mem0_memories",
            "embedding_model_dims": 1536,
            "profile": "commercial",
            "metadata_mode": "auto",
            "bm25_mode": "auto",
        },
    },
    "embedder": {
        "provider": "openai",
        "config": {
            "model": "text-embedding-3-small",
            "api_key": "<api-key>",
        },
    },
    "llm": {
        "provider": "openai",
        "config": {
            "model": "gpt-4o-mini",
            "api_key": "<api-key>",
        },
    },
}

memory = Memory.from_config(config)
```

写入 memory：

```python
memory.add(
    "用户喜欢早晨喝拿铁咖啡，出差时优先选择靠窗座位。",
    user_id="user-001",
    agent_id="travel-agent",
    run_id="run-20260428",
)
```

检索 memory：

```python
result = memory.search(
    "帮用户安排明早的航班和早餐",
    filters={"user_id": "user-001"},
    top_k=5,
)

for item in result["results"]:
    print(item["memory"], item["score"])
```

更新 memory：

```python
memory.update(
    memory_id="<memory-id>",
    data="用户现在更喜欢走廊座位，但早餐仍然喜欢拿铁咖啡。",
)
```

删除 memory：

```python
memory.delete(memory_id="<memory-id>")
```

删除某个用户下的全部 memory：

```python
memory.delete_all(user_id="user-001")
```

对 GaussDB 商用模式，读路径建议始终携带 scope filter：

```python
filters={"user_id": "user-001"}
```

或：

```python
filters={"agent_id": "travel-agent", "run_id": "run-20260428"}
```

这样可以避免跨用户、跨 agent、跨 run 的 memory 泄漏。

### 1.3 mem0 能应用在哪些地方

mem0 适合所有需要“长期记忆 + 个性化上下文 + 可更新知识”的 AI 应用。典型场景如下：

| 场景 | mem0 的作用 | GaussDB 适配价值 |
|---|---|---|
| AI 助手 / 个人助手 | 记住用户偏好、习惯、历史任务、常用表达。 | 用 scope filter 隔离不同用户，用向量 + BM25 找回相关偏好。 |
| 智能客服 | 记住客户历史问题、工单状态、产品偏好、服务记录。 | 用事务型数据库承载客户 memory，支持审计、更新和删除。 |
| 企业知识助手 | 记住员工查询习惯、项目上下文、团队术语。 | 数据留在企业数据库内，便于统一安全和运维。 |
| Agent 平台 | 每个 agent 拥有独立记忆，按 run 记录任务过程。 | `agent_id/run_id` 可以直接映射到 GaussDB filter/index。 |
| 销售/CRM Copilot | 记住客户画像、沟通历史、购买意向、下一步动作。 | 关系型数据库更适合和业务系统集成。 |
| 医疗/教育/金融助理 | 需要强隔离、可审计、可删除、可追踪 memory。 | GaussDB 的事务、权限、备份、审计和隔离策略更适合商用合规。 |
| 多轮任务规划 | 记住任务约束、用户反馈、阶段性决策。 | semantic search 找相似历史任务，BM25 找关键词约束。 |
| RAG 个性化增强 | 在检索外部知识前，先召回用户长期偏好和会话事实。 | GaussDB 同时承载 memory 向量检索和关键词检索。 |

mem0 和传统 RAG 的区别在于：

| 维度 | 传统 RAG | mem0 |
|---|---|---|
| 数据来源 | 文档、知识库、网页、结构化资料 | 用户对话、业务事件、agent 运行过程、长期偏好 |
| 数据粒度 | 文档 chunk | memory/fact |
| 生命周期 | 通常较稳定，批量构建 | 持续 add/update/delete |
| 检索目标 | 找外部知识 | 找用户/agent/run 相关历史记忆 |
| 隔离重点 | 多租户文档权限 | user_id/agent_id/run_id 记忆隔离 |
| 更新方式 | 离线重建或增量同步 | 在线实时更新 memory |

因此，mem0 更适合做 AI 应用的“长期状态层”。GaussDB 适配的意义，是让这个长期状态层可以落在企业级数据库中，而不是只能依赖专用向量库或外部搜索服务。

## 2. mem0 是怎么执行 Vector Store 的

### 2.1 初始化链路

mem0 的入口通常是 `Memory.from_config(config)`：

```text
Memory.from_config
  -> MemoryConfig / VectorStoreConfig 校验配置
  -> EmbedderFactory.create(...)
  -> VectorStoreFactory.create(provider, config)
  -> LlmFactory.create(...)
  -> SQLiteManager(history_db_path)
```

GaussDB 的接入点在 `VectorStoreFactory.provider_to_class`：

```python
"gaussdb": "mem0.vector_stores.gaussdb.GaussDB"
```

因此用户配置：

```python
{
    "vector_store": {
        "provider": "gaussdb",
        "config": {...}
    }
}
```

最终会实例化 `mem0.vector_stores.gaussdb.GaussDB`。后续所有 memory 写入、检索、更新、删除都通过这个 provider 完成。

### 2.2 add 写入链路

`Memory.add()` 有两种路径。

当 `infer=False` 时：

```text
Memory.add
  -> _build_filters_and_metadata(user_id/agent_id/run_id, metadata)
  -> _add_to_vector_store(..., infer=False)
  -> embedding_model.embed(message, "add")
  -> _create_memory(...)
  -> vector_store.insert(vectors=[...], ids=[...], payloads=[...])
  -> db.add_history(...)
```

当 `infer=True` 时：

```text
Memory.add
  -> 读取历史消息
  -> 语义检索已有 memory
  -> LLM 抽取新增/更新/删除候选 memory
  -> embed_batch / embed
  -> vector_store.insert(...)
  -> 记录 history
```

对 GaussDB 来说，最终落点都是 `GaussDB.insert()`。payload 中关键字段包括：

- `data`：原始 memory 文本。
- `text_lemmatized`：BM25 keyword search 使用的归一化文本。
- `user_id`、`agent_id`、`run_id`：mem0 的主要 session/scope 字段。
- `hash`、`created_at`、`updated_at`、`role`、`actor_id` 等上层元数据。

### 2.3 search 检索链路

`Memory.search()` 不是简单调用一次向量检索，而是一个融合检索流程：

```text
Memory.search
  -> 校验 filters 必须包含 user_id / agent_id / run_id 至少一个
  -> 处理高级 metadata filter
  -> _search_vector_store(...)
      -> lemmatize_for_bm25(query)
      -> extract_entities(query)
      -> embedding_model.embed(query, "search")
      -> vector_store.search(query, embedding, top_k=max(top_k*4, 60), filters)
      -> vector_store.keyword_search(query_lemmatized, top_k=max(top_k*4, 60), filters)
      -> entity_store search 做 entity boost
      -> score_and_rank(semantic_results, bm25_scores, entity_boosts)
  -> 返回 {"results": [...]}
```

这意味着 provider 只负责提供单路能力：

- `search()` 返回语义向量结果，分数方向需要“越大越好”。
- `keyword_search()` 如果支持，返回 keyword/BM25 结果；如果不支持返回 `None`。
- `search_batch()` 是 provider 可选优化，mem0 基类默认逐条调用 `search()`。

mem0 最终排序在 Memory 层完成，不要求底层 provider 直接做 hybrid rank。

### 2.4 update / delete / list 链路

`Memory.update(memory_id, data, metadata)`：

```text
Memory.update
  -> embedding_model.embed(data, "update")
  -> _update_memory(...)
  -> vector_store.update(memory_id, vector=new_embedding, payload=new_payload)
  -> db.add_history(...)
```

`Memory.delete(memory_id)`：

```text
Memory.delete
  -> vector_store.get(memory_id)
  -> vector_store.delete(memory_id)
  -> db.add_history(...)
```

`Memory.delete_all(user_id/agent_id/run_id)`：

```text
Memory.delete_all
  -> vector_store.list(filters)
  -> 对每条 memory 调用 _delete_memory(...)
```

因此 provider 的 `get/list/delete/update` 行为会直接影响上层 mem0 API 的正确性。

## 3. mem0 VectorStoreBase 的接口口径

当前 `VectorStoreBase` 必选接口如下：

| 接口 | mem0 语义 |
|---|---|
| `create_col` | 创建 collection，SQL 类 provider 通常映射为建表/建索引。 |
| `insert` | 写入向量、id、payload。多数 provider 支持批量输入。 |
| `search` | 基于 query vector 做 dense semantic search。 |
| `delete` | 按 id 删除一条 memory。 |
| `update` | 更新 vector 和/或 payload。 |
| `get` | 按 id 读取一条 memory。 |
| `list_cols` | 列出 collection。 |
| `delete_col` | 删除 collection。 |
| `col_info` | 返回 collection 信息。不同 provider 字段不完全统一。 |
| `list` | 按 filter 列出 memory。 |
| `reset` | 删除并重建 collection。 |

可选增强接口：

| 接口 | 默认行为 | 当前支持情况 |
|---|---|---|
| `keyword_search` | 基类返回 `None` | 当前 25 个 provider 中 16 个 override。 |
| `search_batch` | 基类逐条调用 `search()` | 当前只有 `qdrant` 和 `gaussdb` override。 |

## 4. GaussDB Provider 的总体设计逻辑

### 4.1 设计目标

GaussDB provider 的目标不是“最低可用”，而是面向商用交付：

- 数据库形态：GaussDB 集中式。
- 兼容模式：A 模式。
- 存储模式：默认 Ustore。
- 向量能力：`FLOATVECTOR(dim)` + `gsdiskann` / `gsivfflat`。
- 关键词能力：GaussDB 原生 BM25 index + `###` score 查询。
- 隔离策略：默认强制 read path 必须带正向 scope filter。
- 配置体验：用 `profile`、`metadata_mode`、`bm25_mode` 收敛常用参数。
- 生产能力：连接池、事务 rollback、retry、慢查询日志、metrics、能力探测、fallback、schema meta 和迁移辅助。

### 4.2 数据模型

GaussDB 将一个 mem0 collection 映射为一张 Ustore 主表：

```sql
CREATE TABLE IF NOT EXISTS "<collection>" (
    id UUID PRIMARY KEY,                 -- 或 VARCHAR(36)
    vector FLOATVECTOR(<dims>) NOT NULL,
    payload JSONB NOT NULL,              -- 或 TEXT
    memory TEXT,
    text_lemmatized TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    schema_version INTEGER DEFAULT 1,
    user_id VARCHAR(128),                -- redundant_columns 模式下存在
    agent_id VARCHAR(128),               -- redundant_columns 模式下存在
    run_id VARCHAR(128)                  -- redundant_columns 模式下存在
) WITH (storage_type=ustore);
```

同时创建 schema meta 表：

```sql
CREATE TABLE IF NOT EXISTS "<collection>_schema_meta" (
    collection_name VARCHAR(128) PRIMARY KEY,
    schema_version INTEGER NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) WITH (storage_type=ustore);
```

### 4.3 索引模型

| 索引 | 用途 | 当前实现 |
|---|---|---|
| Vector index | 语义向量检索 | `CREATE INDEX ... USING gsdiskann/gsivfflat (vector COSINE/L2)` |
| BM25 index | keyword/BM25 检索 | `CREATE INDEX ... USING bm25 (text_lemmatized) WITH (storage_type='USTORE')` |
| Filter index | scope filter 加速 | JSON expression 模式下建 `payload->>'user_id'` 等表达式索引；redundant 模式下建普通列索引。 |

BM25 是 optional/required/disabled 三态：

| `bm25_mode` | 行为 |
|---|---|
| `auto` | 默认启用，建索引/查询失败时降级并返回 `None`。 |
| `required` | BM25 失败直接抛错，适合验收或强依赖场景。 |
| `disabled` | 不建 BM25，不执行 keyword search。 |

### 4.4 配置模式

为降低用户配置复杂度，GaussDB 适配增加了高层配置：

| 配置 | 值 | 语义 |
|---|---|---|
| `profile` | `commercial` | 默认商用模式，偏高性能和完整能力。 |
| `profile` | `compatibility` | 兼容模式，倾向 TEXT payload、redundant scope columns、`gsivfflat`。 |
| `metadata_mode` | `auto` | 默认按 profile 和 capability probe 决定。 |
| `metadata_mode` | `jsonb` | payload 用 JSONB，filter 用 JSON expression。 |
| `metadata_mode` | `redundant_columns` | payload 用 JSONB，scope filter 用冗余列。 |
| `metadata_mode` | `compatible` / `text` | payload 用 TEXT，scope filter 用冗余列。 |
| `payload_storage_mode` | `jsonb` / `text` | 低层显式覆盖 payload 存储。 |
| `filter_storage_mode` | `json_expression` / `redundant_columns` | 低层显式覆盖 filter 实现。 |
| `bm25_mode` | `auto` / `required` / `disabled` | 控制 BM25 启用与失败策略。 |

关键设计点：payload 存储和 filter 实现被拆成两个维度。这样即使 JSONB 不可用，payload 降为 TEXT 后，`user_id/agent_id/run_id` 仍可通过 redundant columns 保持商用隔离能力。

### 4.5 写入与 upsert

GaussDB 没有直接照搬 PostgreSQL `ON CONFLICT`，而是使用 set-based CTE upsert：

```text
WITH incoming (...) AS (VALUES ...)
UPDATE target
SET ...
FROM incoming
WHERE target.id = incoming.id;

WITH incoming (...) AS (VALUES ...)
INSERT INTO target (...)
SELECT ...
FROM incoming
WHERE NOT EXISTS (...);
```

这样有几个好处：

- 适配 A 模式下 `ON CONFLICT` 兼容性风险。
- 避免逐行 DML，batch 写入仍是集合化 SQL。
- 可同时维护 `memory`、`text_lemmatized`、`schema_version` 和 redundant scope columns。

### 4.6 语义检索

GaussDB `search()`：

| `vector_metric` | 距离算子 | Index metric | 排序 |
|---|---|---|---|
| `cosine` | `<+>` | `COSINE` | distance ASC |
| `l2` | `<->` | `L2` | distance ASC |

返回给 mem0 的 score 统一归一化为：

```python
score = 1.0 / (1.0 + max(distance, 0.0))
```

这保证 Memory 层拿到的是“越大越好”的分数。注意 cosine 与 L2 的原始距离范围不同，但经过该函数后都映射到 `(0, 1]`。

### 4.7 BM25 keyword_search

GaussDB `keyword_search()`：

- 使用 `text_lemmatized ### %s AS score` 获取 BM25 分数。
- 设置 session-local BM25 参数：
  - `bm25_ranking_metric`
  - `bm25_ncandidates`
  - 可选 `bm25_dictionary`
  - `enable_seqscan=off`
- 按 score DESC 排序。
- 如果 BM25 disabled，返回 `None`，与 mem0 provider 约定一致。
- 空 query 返回 `[]`。
- BM25 runtime 失败且 `bm25_fail_fast=False` 时返回 `None` 并增加 fallback metric。

### 4.8 search_batch

GaussDB override 了 `search_batch()`，不是走基类的逐条循环。实现方式：

```sql
WITH query_vectors(query_index, query_vector) AS (
    VALUES (0, %s::FLOATVECTOR), (1, %s::FLOATVECTOR), ...
)
SELECT q.query_index, r.id, r.distance, r.payload
FROM query_vectors q
CROSS JOIN LATERAL (
    SELECT id, vector <op> q.query_vector AS distance, payload
    FROM <table>
    WHERE <filters>
    ORDER BY distance ASC, id ASC
    LIMIT %s
) r
ORDER BY q.query_index ASC, r.distance ASC, r.id ASC;
```

如果 native batch SQL 失败，会 fallback 到逐条 `search()`，并记录 `gaussdb_fallback_count`。

### 4.9 Filter 与租户隔离

GaussDB 默认 `require_scoped_filters=True`。所有 read path：

- `search`
- `keyword_search`
- `search_batch`
- `list`

都会要求 filters 中存在至少一个正向 scope predicate：

- `user_id`
- `agent_id`
- `run_id`

支持的正向 scope 示例：

```python
{"user_id": "alice"}
{"user_id": {"eq": "alice"}}
{"user_id": {"in": ["alice", "bob"]}}
{"$and": [{"category": "travel"}, {"user_id": "alice"}]}
{"$or": [{"user_id": "alice"}, {"agent_id": "agent-1"}]}
```

拒绝的非约束或可绕过 scope 示例：

```python
{"category": "public"}
{"$or": [{"user_id": "alice"}, {"category": "public"}]}
{"user_id": {"ne": "bob"}}
{"user_id": {"nin": ["bob"]}}
{"$not": [{"user_id": "alice"}]}
```

普通 metadata filter 支持：

| 操作 | 示例 |
|---|---|
| eq | `{"category": "travel"}` / `{"category": {"eq": "travel"}}` |
| ne | `{"category": {"ne": "travel"}}` |
| in | `{"category": {"in": ["travel", "food"]}}` |
| nin | `{"category": {"nin": ["travel"]}}` |
| gt/gte/lt/lte | `{"priority": {"gte": 5, "lte": 7}}` |
| contains | `{"tag": {"contains": "coffee"}}` |
| icontains | `{"tag": {"icontains": "plan"}}` |
| AND/OR/NOT | `{"$and": [...]}`、`{"$or": [...]}`、`{"$not": [...]}` |

### 4.10 能力探测与 fallback

`enable_capability_probe=True` 时，GaussDB 初始化会探测：

- `enable_vectordb`
- `FLOATVECTOR` 类型
- UUID 类型
- JSONB payload
- JSON expression index
- vector index 创建
- BM25 index 创建
- BM25 score 查询是否可执行

fallback 规则：

| 探测失败项 | fallback |
|---|---|
| JSONB 不可用 | payload 降为 TEXT，filter 改为 redundant columns。 |
| JSON expression index 不可用 | payload 保持 JSONB，filter 改为 redundant columns。 |
| BM25 index/score 不可用且非 required | `bm25_enabled=False`，keyword_search 返回 `None`。 |
| vector 能力不可用 | 抛错，不继续。 |

### 4.11 运维与迁移辅助

GaussDB 适配增加了现有多数 provider 没有的商用辅助能力：

| 能力 | 说明 |
|---|---|
| `col_info()` | 返回 count、schema_version、profile、metadata mode、vector index、BM25 状态、索引列表。 |
| `migration_dry_run()` | 返回 v1 provider-managed 迁移计划。 |
| `backfill_derived_fields()` | JSONB 模式下回填 `memory/text_lemmatized`；TEXT 模式下明确要求应用侧 recompute。 |
| `analyze()` | 执行 `ANALYZE <table>`。 |
| `_ensure_indexes()` | 内部建表/修复索引，不暴露为 mem0 标准接口。 |
| metrics | 记录 fallback、retry 等计数。 |
| retry | 对连接、timeout、deadlock、serialization 等 transient error 做重试。 |
| transaction rollback | cursor context 失败自动 rollback。 |
| BM25 savepoint | BM25 optional DDL 失败不会拖垮 create_col 事务。 |

## 5. 当前 GaussDB 已适配功能点清单

| 功能域 | 当前状态 | 说明 |
|---|---|---|
| mem0 provider 注册 | 已完成 | `VectorStoreFactory` 支持 `gaussdb`。 |
| 配置模型 | 已完成 | `GaussDBConfig` 支持连接、profile、metadata、BM25、索引、probe、retry。 |
| 连接方式 | 已完成 | 支持 `connection_string/dsn/url`、host/port/database/user/password、环境变量、外部 connection_pool。 |
| 连接池 | 已完成 | 使用 psycopg2-compatible `ThreadedConnectionPool`。 |
| Ustore 表 | 已完成 | `WITH (storage_type=ustore)`。 |
| A 模式约束 | 已完成 | `compatibility_mode="A"` 校验。 |
| 向量列 | 已完成 | `FLOATVECTOR(dim)`。 |
| 向量索引 | 已完成 | `gsdiskann` / `gsivfflat`。 |
| 向量 metric | 已完成 | cosine/L2。 |
| 批量 insert/upsert | 已完成 | CTE UPDATE + INSERT-not-exists。 |
| dense search | 已完成 | distance ASC + score normalization。 |
| BM25 keyword search | 已完成 | 原生 BM25 index + `###` score。 |
| batch search | 已完成 | native lateral top-k，失败 fallback。 |
| JSONB payload | 已完成 | 默认商用模式。 |
| TEXT payload | 已完成 | compatibility/text 模式。 |
| redundant scope columns | 已完成 | 支持 TEXT payload 或 expression index fallback 下的 scope filter。 |
| filter operators | 已完成 | eq/ne/in/nin/range/contains/icontains/AND/OR/NOT。 |
| scope 强制隔离 | 已完成 | 默认 read path 必须正向约束 user/agent/run。 |
| list/get/delete/update | 已完成 | 对齐 mem0 provider 口径。 |
| partial update | 已完成 | payload partial 不清空 redundant scope；vector-only 不改 payload。 |
| schema meta | 已完成 | `col_info()` 读取 schema version。 |
| migration/backfill | 已完成 v1 helper | 不是完整 migration framework，但有 dry-run/backfill helper。 |
| observability | 已完成 | metrics、slow query、fallback count。 |
| live P0/P1/P2 测试 | 已完成 | `mem0_e2e_db` UTF-8 库上 P0 全量 `22 passed`。 |

## 6. 与 mem0 已有 Provider 的横向对比

当前仓库 `mem0/vector_stores` 下共有 25 个 provider 文件，其中 24 个是 mem0 原有适配，`gaussdb` 是当前新增适配。

### 6.1 总体接口覆盖

所有 provider 都实现了 `VectorStoreBase` 的必选接口，但增强能力差异明显：

| 能力 | 当前覆盖 |
|---|---|
| 基础 CRUD/list/reset | 25/25 |
| dense vector search | 25/25 |
| `keyword_search` override | 16/25 |
| `search_batch` override | 2/25：`qdrant`、`gaussdb` |
| provider 内部强制 scope guard | 少数。GaussDB 是明确默认强制的实现之一。 |
| 商用能力探测/fallback/schema meta/backfill | 多数 provider 没有统一实现，GaussDB 覆盖较多。 |

### 6.2 重点 Provider 对比

| Provider | 类型 | keyword/BM25 | batch search | filter 能力 | 与 GaussDB 对比 |
|---|---|---|---|---|---|
| `pgvector` | PostgreSQL + pgvector | PostgreSQL full-text，`ts_rank_cd`，不是原生 BM25 | 无 native override | 简单 JSONB equality filter | GaussDB 对标它的 SQL 表模型，但增强了 native BM25、batch search、scope guard、fallback、schema meta 和商用测试。 |
| `qdrant` | 专用向量库 | 支持 sparse/BM25 方向 | 支持 native batch | payload filter 较成熟 | Qdrant 是唯一与 GaussDB 同时具备 keyword + batch 的 provider。GaussDB 的优势是 SQL/Ustore/BM25/事务/商用数据库能力；Qdrant 优势是专用向量引擎和 payload index。 |
| `mongodb` | MongoDB Atlas Vector Search | Atlas Search text index | 无 native override | pipeline/filter | GaussDB 的 JSONB/redundant filter 与 MongoDB payload/filter 类似，但 GaussDB 更强调事务和 scope guard。 |
| `elasticsearch` / `opensearch` | 搜索引擎 | 原生 BM25/match | 无 native override | term filter | ES/OpenSearch keyword 能力成熟；GaussDB 的优势是同一 Ustore 表同时承载向量、payload、BM25 与事务语义。 |
| `azure_ai_search` | Azure AI Search | search text/BM25 服务能力 | 无 native override | OData filter | GaussDB 不依赖外部搜索服务，能力在数据库内闭环。 |
| `azure_mysql` | MySQL 类 SQL | FULLTEXT `MATCH ... AGAINST` | 无 native override | JSON_EXTRACT equality | GaussDB 的向量、BM25、filter 和 batch search 更贴近 mem0 检索链路。 |
| `redis` / `valkey` | Search + vector | Redis 支持 keyword，Valkey 当前未 override | 无 native override | schema/filter 表达能力依赖 Search | GaussDB 更偏关系型强事务与 schema 演进。 |
| `pinecone` / `weaviate` / `milvus` / `upstash_vector` | 专用/云向量服务 | 多数有 keyword 或 hybrid 能力 | 无 native override | metadata filter | 这些 provider 依赖各自云服务能力；GaussDB 适合数据库内统一承载 memory 数据。 |
| `faiss` | 本地索引 | 无 server-side keyword | 无 native override | 本地 metadata filter | FAISS 适合本地轻量场景；GaussDB 面向生产数据库。 |
| `langchain` | wrapper | 无独立实现 | 无 native override | 取决于底层 LangChain store | GaussDB 是一等 provider，不是 wrapper。 |

### 6.3 全量 Provider 能力矩阵

| Provider | 文件行数 | keyword_search | search_batch | 适配度判断 | 备注 |
|---|---:|---|---|---|---|
| `azure_ai_search` | 424 | 是 | 否 | 高 | Azure Search index、vector search、keyword search。 |
| `azure_mysql` | 533 | 是 | 否 | 中高 | SQL + JSON + FULLTEXT，向量距离更多在 SQL/客户端侧计算。 |
| `baidu` | 411 | 是 | 否 | 高 | 云向量服务适配。 |
| `cassandra` | 496 | 否 | 否 | 中高 | 基础向量检索和 CRUD 完整。 |
| `chroma` | 332 | 否 | 否 | 中高 | 本地/服务型向量库，filter 基础可用。 |
| `databricks` | 875 | 是 | 否 | 高 | Databricks Vector Search，工程配置较重。 |
| `elasticsearch` | 289 | 是 | 否 | 高 | dense vector + BM25/match。 |
| `faiss` | 631 | 否 | 否 | 中 | 本地索引，非数据库型。 |
| `gaussdb` | 1343 | 是 | 是 | 高 | SQL/Ustore/FLOATVECTOR/BM25/batch/scope/商用能力最完整。 |
| `langchain` | 180 | 否 | 否 | 中低 | wrapper，能力取决于底层 store。 |
| `milvus` | 346 | 是 | 否 | 高 | 专用向量库，keyword 能力依赖字段/版本。 |
| `mongodb` | 400 | 是 | 否 | 高 | Atlas Vector Search + Atlas Search。 |
| `neptune_analytics` | 467 | 否 | 否 | 中 | 图数据库/Analytics 模型。 |
| `opensearch` | 380 | 是 | 否 | 高 | KNN + BM25/match。 |
| `pgvector` | 455 | 是 | 否 | 高 | SQL 基线 provider，但缺 batch 和商用 guard。 |
| `pinecone` | 418 | 是 | 否 | 高 | 云向量服务，支持 hybrid/keyword 方向。 |
| `qdrant` | 556 | 是 | 是 | 高 | 专用向量库中最接近 GaussDB 的接口增强程度。 |
| `redis` | 351 | 是 | 否 | 高 | RediSearch vector + keyword。 |
| `s3_vectors` | 206 | 否 | 否 | 中 | AWS S3 Vectors 基础适配。 |
| `supabase` | 237 | 否 | 否 | 中高 | PostgreSQL/Supabase RPC 风格。 |
| `turbopuffer` | 337 | 否 | 否 | 中高 | 云向量 namespace。 |
| `upstash_vector` | 332 | 是 | 否 | 高 | 云向量服务，支持内置 embedding 场景。 |
| `valkey` | 837 | 否 | 否 | 中高 | 实现较重，但当前没有 keyword override。 |
| `vertex_ai_vector_search` | 644 | 是 | 否 | 高 | Vertex AI Vector Search。 |
| `weaviate` | 392 | 是 | 否 | 高 | 专用向量库，支持 keyword 方向。 |

### 6.4 GaussDB 适配点逐项走读：为什么这么做，友商怎么做

本节按当前 `mem0/vector_stores/gaussdb.py` 的实现顺序走读。这里的“友商”指 mem0 已有 provider 背后的同类后端，包括 `pgvector`、`qdrant`、`mongodb`、`elasticsearch`、`opensearch`、`azure_mysql`、`azure_ai_search`、`pinecone`、`weaviate` 等。

#### 6.4.1 Provider 注册与配置入口

| 维度 | GaussDB 当前适配 | 为什么这么适配 | 友商/已有 provider 做法 | 判断 |
|---|---|---|---|---|
| provider 注册 | 在 `VectorStoreFactory.provider_to_class` 中注册 `"gaussdb": "mem0.vector_stores.gaussdb.GaussDB"`。 | mem0 统一通过 `Memory.from_config()` 和 `VectorStoreFactory.create()` 创建 vector store，必须注册 provider 名称。 | 所有 provider 都走这个工厂注册，例如 `pgvector`、`qdrant`、`mongodb`、`elasticsearch`。 | 与 mem0 标准一致。 |
| 配置模型 | 新增 `GaussDBConfig`，支持连接参数、索引参数、metadata mode、BM25 mode、profile、probe、retry、observability。 | GaussDB 商用部署参数较多，如果只暴露低层字段，用户使用成本高；因此用高层 profile/mode 收敛默认行为。 | `qdrant`、`pinecone`、`weaviate` 更偏服务连接配置；`pgvector` 主要是 DB 连接和 index 开关；`databricks` 配置也很重。 | GaussDB 配置复杂度高于 pgvector，但通过 profile 降低了用户感知。 |
| 环境变量 | 支持 `GAUSSDB_CONNECTION_STRING`、`GAUSSDB_DSN`、`GAUSSDB_URL`、`GAUSSDB_HOST` 等。 | 方便容器、CI、生产环境注入，不把密码写进代码。 | PostgreSQL/MySQL/云服务 provider 通常都支持显式参数，环境变量支持程度不一。 | 对商用部署更友好。 |

设计取舍：

- 不要求用户必须提供所有底层参数；推荐 `profile="commercial"`、`metadata_mode="auto"`、`bm25_mode="auto"`。
- 允许低层参数覆盖高层模式，但通过配置校验避免混用造成语义不清。

#### 6.4.2 驱动、连接池与事务

| 维度 | GaussDB 当前适配 | 为什么这么适配 | 友商/已有 provider 做法 | 判断 |
|---|---|---|---|---|
| Python driver | 使用 psycopg2-compatible API，`ThreadedConnectionPool`。 | GaussDB 官方 Python 示例和生态更接近 psycopg2；避免引入 psycopg3 特性导致兼容风险。 | `pgvector` 同时兼容 psycopg2/psycopg3；`azure_mysql` 使用 PyMySQL/DBUtils；云向量库使用各自 SDK。 | 对 GaussDB 兼容性优先。 |
| 连接池 | 默认 `minconn=1`、`maxconn=5`，也可传入外部 connection_pool。 | mem0 search/add 会频繁调用 provider；连接池避免每次创建连接。 | SQL 类 provider 通常有连接池或客户端复用；Qdrant/ES/MongoDB 复用 SDK client。 | 商用必要能力。 |
| 事务处理 | `_get_cursor(commit=True)` 成功 commit，异常 rollback，并归还连接。 | DDL/DML 任一失败必须回滚，避免连接留在 aborted transaction 状态。 | `pgvector` 有 cursor context；部分云 SDK provider 由 SDK 管理事务或无显式事务。 | GaussDB 做法符合数据库型 provider 要求。 |
| retry | 对连接、timeout、deadlock、serialization 等 transient error 做重试。 | 商用数据库可能遇到瞬态锁冲突或网络抖动，provider 层要有基本恢复能力。 | 多数 provider 依赖 SDK retry；`pgvector` 当前没有同等显式重试策略。 | GaussDB 工程化更强。 |

#### 6.4.3 Collection 映射与表结构

| 维度 | GaussDB 当前适配 | 为什么这么适配 | 友商/已有 provider 做法 | 判断 |
|---|---|---|---|---|
| collection 形态 | 一个 mem0 collection 对应一张 GaussDB Ustore 表。 | mem0 collection 与 SQL table 天然对应；Ustore 是目标商用基线。 | `pgvector` 也是一张 PostgreSQL 表；`azure_mysql` 是 MySQL 表；`qdrant` 是 collection；ES/OpenSearch 是 index；MongoDB 是 collection。 | 与 SQL provider 对齐。 |
| 主表字段 | `id`、`vector`、`payload`、`memory`、`text_lemmatized`、时间戳、`schema_version`、可选 scope 冗余列。 | mem0 不只需要向量，还需要 payload、关键词字段、scope 隔离、迁移标识。 | `pgvector` 只有 `id/vector/payload`，关键词直接从 payload 取；`azure_mysql` 用 JSON payload + generated `text_lemmatized`；ES/OpenSearch mapping 中拆 `vector/metadata`。 | GaussDB 表结构更面向 mem0 检索链路。 |
| schema meta | 额外创建 `{collection}_schema_meta`。 | `col_info()` 和后续 migration 需要知道真实 schema version，不能硬编码。 | 多数 provider 没有 schema meta；云服务通常由 index metadata 承担。 | GaussDB 为后续升级预留空间。 |
| Ustore | 主表和 meta 表都 `WITH (storage_type=ustore)`。 | 用户目标是集中式、A 模式、Ustore；BM25 已验证可在 Ustore 返回 score。 | PostgreSQL/MySQL/ES/Qdrant 没有 Ustore 概念。 | GaussDB 特有适配点。 |
| 分布式兼容 | `deployment_mode="distributed"` 时追加 `DISTRIBUTE BY HASH ("id")`，schema meta 表追加 `DISTRIBUTE BY HASH ("collection_name")`。 | mem0 标准 provider 接口以 `id` 为主键和 DML 锚点，先保证分布式库能建表、写入和查询。 | 大多数 mem0 provider 不暴露集中式/分布式形态；服务型 provider 由后端隐藏分片。 | GaussDB 需要显式适配 SQL 分布式 DDL，但当前定位是兼容模式，不是 scope-hash 性能最优模式。 |

为什么不单独拆 BM25 影子表：

- 当前已验证 Ustore 主表可建 BM25 并返回 score，因此不需要默认引入 Astore 影子表。
- 单表设计减少同步复杂度，`insert/update/delete/reset` 更容易保持一致。
- 如果未来某些商用版本限制 Ustore BM25，再考虑影子表或降级策略。

#### 6.4.4 向量类型、metric 与索引

| 维度 | GaussDB 当前适配 | 为什么这么适配 | 友商/已有 provider 做法 | 判断 |
|---|---|---|---|---|
| 向量类型 | `FLOATVECTOR(dim)`。 | 对接 GaussDB 向量数据库能力，避免用 JSON/list 模拟向量。 | `pgvector` 用 `vector(dim)`；ES 用 `dense_vector`；Qdrant 用 dense vector slot；MongoDB 用 Atlas vector index。 | 对齐 GaussDB 原生能力。 |
| metric | 支持 `cosine` 和 `l2`。 | mem0 用户可能使用不同 embedding 模型和距离口径；GaussDB 算子需要显式映射。 | `pgvector` 当前主要使用 cosine 距离；Qdrant/Pinecone/Weaviate 等通常支持多 metric。 | 覆盖主流需求。 |
| 算子映射 | cosine -> `<+>`，L2 -> `<->`。 | GaussDB 手册定义的向量距离算子；查询按 distance ASC。 | `pgvector` 使用 `<=>`；MySQL 适配中有 Python 侧 cosine 计算；ES/Qdrant 由服务端 query API 封装。 | GaussDB 使用数据库内原生算子。 |
| index 类型 | `gsdiskann` / `gsivfflat`。 | `gsdiskann` 面向较大规模和性能，`gsivfflat` 适合兼容/轻量验收。 | `pgvector` 有 HNSW/DiskANN 逻辑；Qdrant 内部 HNSW；ES/OpenSearch KNN；MongoDB Atlas vectorSearch index。 | 与主流向量索引能力对齐。 |
| maintenance_work_mem | 建索引前 `SET LOCAL maintenance_work_mem`。 | GaussDB 向量索引构建可能需要较高内存；session-local 设置比要求用户全局改参数更安全。 | `pgvector` 当前没有同等封装；部分服务型 provider 由服务端管理资源。 | 体现数据库运维适配。 |

#### 6.4.5 Payload、metadata 与 filter 存储模式

| 维度 | GaussDB 当前适配 | 为什么这么适配 | 友商/已有 provider 做法 | 判断 |
|---|---|---|---|---|
| 默认 payload | JSONB。 | mem0 payload 是半结构化 metadata，JSONB 适合保存并做表达式过滤。 | `pgvector` 用 JSONB；`azure_mysql` 用 JSON；ES/OpenSearch 用 object/metadata；Qdrant/MongoDB 用 payload/document。 | 与主流设计一致。 |
| TEXT payload | compatibility 模式支持 TEXT。 | 某些 GaussDB 商用版本/环境可能 JSONB 或 JSON expression index 受限，需要兼容路径。 | `pgvector` 当前没有 TEXT fallback；ES/Qdrant/MongoDB 没有这个问题。 | GaussDB 特有兼容设计。 |
| filter_storage_mode | `json_expression` 或 `redundant_columns`。 | payload 存储和 filter 实现分离；即使 payload 降 TEXT，也要保证 scope filter 可用。 | Qdrant 建 payload index；ES/OpenSearch 建 keyword field；Azure AI Search 将 user/run/agent 独立为 filterable field；pgvector 直接 `payload->>`。 | GaussDB redundant columns 对标搜索/向量服务中的独立 filter 字段。 |
| allowed_filter_keys | 可选 allowlist。 | 防止任意 payload key 被拼进 SQL 表达式，降低误用和注入风险。 | ES/Qdrant 通常通过 schema/index field 限制；pgvector 当前较宽松。 | 商用安全性更强。 |

核心原因：GaussDB 面向企业数据库，不应假设所有版本/模式都能稳定支持 JSONB expression index。拆分 storage 与 filter 后，provider 可以按能力自动选择更稳路径。

#### 6.4.6 写入与 Upsert

| 维度 | GaussDB 当前适配 | 为什么这么适配 | 友商/已有 provider 做法 | 判断 |
|---|---|---|---|---|
| insert 输入 | 接收批量 `vectors/payloads/ids`。 | mem0 `_create_memory`、批量抽取和测试都可能一次写多条。 | 所有 provider 基本都支持批量输入。 | 标准能力。 |
| upsert 方式 | 两段 CTE：先 set-based UPDATE，再 INSERT-not-exists。 | A 模式下不直接依赖 PostgreSQL `ON CONFLICT`；同时避免逐行 UPDATE/INSERT。 | `qdrant` 是 upsert points；ES bulk index；Azure MySQL 用 `ON DUPLICATE KEY UPDATE`；`pgvector` 当前是 insert，不是显式 upsert。 | GaussDB 更兼容 A 模式且有批量性能。 |
| 派生字段 | 写入 `memory`、`text_lemmatized`、`schema_version`，redundant 模式写 scope columns。 | Memory 层 keyword_search 需要 `text_lemmatized`；list/search 需要 payload；scope filter 需要冗余列。 | `pgvector` keyword 从 payload 读取；Azure MySQL 用 generated column；Qdrant 写 dense vector 时也可写 BM25 sparse vector。 | GaussDB 选择显式派生列，便于建索引和 backfill。 |

为什么不用逐行 DML：

- mem0 的 add 可能由 LLM 一次抽取多条 memory，逐行 DML 会放大网络往返和事务开销。
- 当前测试已锁定多行 insert 只发两条 set-based SQL。

#### 6.4.7 语义向量检索 search

| 维度 | GaussDB 当前适配 | 为什么这么适配 | 友商/已有 provider 做法 | 判断 |
|---|---|---|---|---|
| 查询方式 | SQL 中 `vector <op> %s::FLOATVECTOR AS distance`，按 distance ASC。 | 使用 GaussDB 原生向量算子，保证索引可用。 | `pgvector` 类似 SQL distance；Qdrant/MongoDB/ES 使用各自 query API；Azure MySQL 当前更多在 Python 侧算 cosine。 | GaussDB 与 pgvector 同类，但更贴近数据库内执行。 |
| score 输出 | `1 / (1 + distance)`。 | mem0 Memory 层融合排序要求 provider score 越大越好；距离越小越好，需要转换。 | 不同 provider 分数语义不完全一致；ES/Qdrant/Pinecone 通常服务端直接返回 similarity score。 | GaussDB 做了方向统一。 |
| 排序稳定性 | `ORDER BY distance ASC, id ASC`。 | 距离相同或接近时保证结果稳定，测试和线上分页更可预测。 | 不同 provider 由服务端排序决定；pgvector 当前主要按 distance。 | GaussDB 结果更稳定。 |

#### 6.4.8 BM25 / keyword_search

| 维度 | GaussDB 当前适配 | 为什么这么适配 | 友商/已有 provider 做法 | 判断 |
|---|---|---|---|---|
| keyword 字段 | `text_lemmatized` 独立列。 | BM25 index 直接建在文本列上，避免每次从 JSON 解析。 | `pgvector` 从 `payload->>'text_lemmatized'` 做 `to_tsvector`；Azure MySQL 用 generated column；ES/OpenSearch match metadata 字段；Qdrant 用 sparse vector slot。 | GaussDB 更利于数据库原生 BM25。 |
| keyword 算法 | GaussDB native BM25，`text_lemmatized ### query`。 | 用户目标明确包含 BM25；普通 FTS 不等价于 BM25。 | ES/OpenSearch 原生 BM25；MongoDB Atlas Search text；Qdrant 用 fastembed sparse BM25；pgvector 用 PostgreSQL FTS rank，不是 BM25。 | GaussDB 对标搜索引擎和 Qdrant 的 keyword 能力。 |
| 失败策略 | `bm25_mode=auto/required/disabled`，auto 失败返回 `None`。 | mem0 约定不支持 keyword 时返回 `None`；验收场景可用 required 强制失败。 | pgvector keyword 异常返回 `None`；Azure MySQL FULLTEXT 异常也返回 `None`；Qdrant 缺 fastembed 或 sparse slot 时禁用。 | GaussDB 策略更明确。 |
| 事务保护 | BM25 optional DDL 用 SAVEPOINT。 | DDL 失败后事务可能 aborted；必须 rollback 到 savepoint，不能拖垮主表/vector/filter index 创建。 | 部分 provider 没有 DDL 事务问题；pgvector 的 FTS index 失败没有同级 optional savepoint 设计。 | GaussDB 对数据库 DDL 风险处理更完整。 |

#### 6.4.9 原生 search_batch

| 维度 | GaussDB 当前适配 | 为什么这么适配 | 友商/已有 provider 做法 | 判断 |
|---|---|---|---|---|
| batch search | 使用 `WITH query_vectors ... CROSS JOIN LATERAL` 一次查询多个 query vector。 | mem0 默认基类逐条 search，网络往返多；SQL 能天然表达 per-query top-k。 | 当前只有 `qdrant` 和 `gaussdb` override `search_batch`；Qdrant 调用原生 batch points。 | GaussDB 在 batch 能力上处于第一梯队。 |
| fallback | native batch 失败后逐条调用 `search()`。 | 兼容不支持 LATERAL 或某些版本 SQL 差异的场景。 | 基类默认就是逐条调用；GaussDB 在失败时回退到基类等价行为。 | 可用性优先。 |

为什么要做 batch：

- Memory 层或上层应用可能需要同时检索多个 query，例如多实体、多候选问题、多轮任务 planning。
- 数据库内 batch 能显著减少连接池压力和网络往返。

#### 6.4.10 Scope filter 与商用隔离

| 维度 | GaussDB 当前适配 | 为什么这么适配 | 友商/已有 provider 做法 | 判断 |
|---|---|---|---|---|
| 默认策略 | read path 默认要求 `user_id/agent_id/run_id` 至少一个正向约束。 | mem0 memory 是用户/agent 长期状态，跨租户泄漏属于高风险问题。 | Memory 层也要求 search filters 包含 scope；但多数 provider 本身不强制，直接 provider 调用可能绕过。 | GaussDB provider-level guard 更安全。 |
| OR/NOT 处理 | OR 必须每个分支都有正向 scope；NOT/NE/NIN 不算有效 scope。 | 防止 `OR(user_id=alice, category=public)` 或 `user_id != bob` 这类条件扩大结果集。 | 很多 provider 只是把 filters 转成后端 filter，不做“是否约束全查询”的逻辑判断。 | GaussDB 更符合商用隔离。 |
| filter 能力 | 支持 eq/ne/in/nin/range/contains/icontains/AND/OR/NOT。 | mem0 上层已经支持增强 filter，provider 需要能落 SQL。 | Qdrant filter 表达能力强；ES/OpenSearch bool/term/match 强；pgvector 目前 filter 较简单；Azure AI Search 是 OData filter。 | GaussDB filter 表达能力高于 pgvector。 |

注意：`get(id)` 是 mem0 标准接口，没有 filters 参数，因此当前不承诺 provider-level scoped get。若商用强隔离要求覆盖 ID 查询，需要修改 mem0 API 或在 Memory 层引入 scope 校验。

#### 6.4.11 update 行为

| 维度 | GaussDB 当前适配 | 为什么这么适配 | 友商/已有 provider 做法 | 判断 |
|---|---|---|---|---|
| vector-only update | 只更新 vector 和 `updated_at`，不改 payload/memory/text。 | 调用方可能只想刷新 embedding；不能隐式篡改文本和 BM25 字段。 | pgvector、ES、Azure MySQL 也大多按 vector/payload 分开更新。 | 行为清晰。 |
| payload-only update | 更新 payload、`memory`、`text_lemmatized`、`updated_at`。 | payload 文本变化后，BM25 字段必须同步，否则 keyword_search 会 stale。 | Azure MySQL generated column 从 payload 派生；pgvector keyword 从 payload 派生；Qdrant payload update 也会影响 keyword 取决于 sparse vector 是否重建。 | GaussDB 显式同步派生列。 |
| redundant scope partial update | payload 缺少某个 scope key 时，不把对应冗余列置空。 | 部分 payload 更新不应破坏隔离索引，否则后续 scoped 查询漏数据。 | 搜索服务中独立 filter 字段也需要显式维护；pgvector 没有 redundant column 问题。 | 已用 P1/P2 测试锁定。 |

#### 6.4.12 能力探测与兼容 fallback

| 维度 | GaussDB 当前适配 | 为什么这么适配 | 友商/已有 provider 做法 | 判断 |
|---|---|---|---|---|
| capability probe | 初始化可探测 vector、FLOATVECTOR、UUID、JSONB、expression index、vector index、BM25 index、BM25 score query。 | GaussDB 商用版本/构建号、兼容模式和参数可能不同，不能只靠静态配置判断能力。 | Qdrant 检查 collection 是否有 bm25 sparse slot；云服务 provider 通常依赖 API 报错；pgvector 检查 extension/index 较有限。 | GaussDB probe 更系统。 |
| JSONB fallback | JSONB 不可用时降为 TEXT + redundant scope columns。 | 保证最核心的 memory 存储和租户隔离仍可工作。 | 多数 provider 没有类似 DB 类型 fallback；SQL provider 通常要求目标类型可用。 | GaussDB 兼容性更强。 |
| expression index fallback | expression index 不可用时 JSONB payload 保留，但 scope filter 改 redundant columns。 | payload 能力和隔离能力分离，避免因为索引限制牺牲隔离。 | Azure AI Search/ES/Qdrant 通常把 scope 做成独立可过滤字段；GaussDB redundant columns 是 SQL 等价设计。 | 设计合理。 |
| BM25 fallback | BM25 不可用时禁用 keyword_search，semantic search 保持可用。 | mem0 允许 keyword_search 返回 `None`；不能让可选 keyword 能力拖垮主链路。 | pgvector/Azure MySQL keyword 异常返回 `None`；Qdrant 缺 sparse slot 时禁用 keyword。 | 与 mem0 生态一致。 |

#### 6.4.13 Collection 运维、schema 与 backfill

| 维度 | GaussDB 当前适配 | 为什么这么适配 | 友商/已有 provider 做法 | 判断 |
|---|---|---|---|---|
| `col_info()` | 返回 name/count/dimension/schema_version/profile/modes/indexes/BM25 状态。 | 商用排障需要知道真实 schema 和索引状态。 | `pgvector` 返回 name/count/size；多数 provider 返回较基础信息。 | GaussDB 信息更完整。 |
| `migration_dry_run()` | 返回 v1 provider-managed 计划。 | 当前不是完整 migration framework，但先提供安全可见性。 | 多数 provider 没有 provider 内 migration 计划。 | 后续可扩展。 |
| `backfill_derived_fields()` | JSONB 模式可回填 `memory/text_lemmatized`；TEXT 模式明确要求应用侧 recompute。 | 旧数据或异常数据可能缺派生字段，影响 BM25 和展示。 | 其他 provider 通常不维护派生列，因此很少有 backfill helper。 | GaussDB 更偏商用运维。 |
| `analyze()` | 执行数据库统计信息刷新。 | 向量/过滤/排序性能依赖统计信息，尤其批量导入后。 | SQL provider 可能需要手工运维；云服务 provider 由服务端处理。 | 数据库型 provider 的必要补充。 |

#### 6.4.14 测试适配策略

| 维度 | GaussDB 当前测试 | 为什么这么测 | 友商/已有 provider 测试常见情况 | 判断 |
|---|---|---|---|---|
| mock/unit | 覆盖配置、SQL 构造、scope guard、BM25 savepoint、insert batch、update、col_info、backfill。 | 不依赖真实库，快速锁住 P1/P2 行为。 | 其他 provider 多数也有 mock 单测，但商用隔离/DDL fallback 覆盖不一定完整。 | P1/P2 已闭合。 |
| live P0 | 覆盖 Memory.from_config、CRUD、filter、compatibility、UTF-8、metric、search_batch、BM25、index matrix。 | provider 适配最终要证明真实 mem0 + 真实 GaussDB 可用。 | 云服务 provider 通常依赖环境变量跑 live tests；本地 provider 有本地测试。 | 已在 UTF-8 库 `mem0_e2e_db` 上 `22 passed`。 |
| 敏感信息 | 测试文件不保留真实地址/密码默认值。 | 避免误连真实环境和泄密。 | 所有 provider 测试都应遵循此原则。 | 已修复并扫描。 |

#### 6.4.15 总体走读结论

GaussDB 的适配路线可以概括为：

1. 先按 mem0 标准接口实现完整 CRUD 和 dense vector search，保证最小可用。
2. 再对齐 mem0 Memory 层真实检索链路，补 `keyword_search`、`search_batch`、score normalization 和 payload 字段。
3. 针对企业 memory 场景补强 scope filter guard，避免 provider 被直接调用时绕过 Memory 层隔离。
4. 针对 GaussDB 商用形态补 Ustore、A 模式、能力探测、JSONB/TEXT fallback、redundant scope columns、BM25 savepoint。
5. 针对生产运维补连接池、事务、retry、metrics、schema meta、backfill、analyze。
6. 最后用 unit + live P0/P1/P2 验证，确保不是“代码看起来适配”，而是真实 mem0 + GaussDB 跑通。

和友商/已有 provider 的核心差异是：

- 相比 `pgvector`：GaussDB 更重视商用隔离、BM25 原生能力、batch search 和兼容 fallback。
- 相比 `qdrant`：GaussDB 同样具备 keyword + batch，但选择数据库内 SQL/Ustore/事务闭环，而不是专用向量服务。
- 相比 `elasticsearch` / `opensearch`：GaussDB 的 BM25 没有搜索引擎生态那么宽，但可以和事务型 memory 表、向量索引、payload/filter 在同一数据库内闭环。
- 相比 `mongodb`：GaussDB 不是文档数据库 pipeline 模式，而是 SQL 表 + JSONB/TEXT payload + 索引组合，适合企业关系型数据库运维体系。
- 相比 `azure_mysql`：GaussDB 使用数据库原生向量类型和 BM25，而不是 JSON 向量或 Python 侧距离计算，向量检索链路更直接。

## 7. GaussDB 与 pgvector 的细粒度对标

| 维度 | pgvector 当前实现 | GaussDB 当前实现 | 结论 |
|---|---|---|---|
| provider 注册 | 已有 | 已有 | 对齐。 |
| 建表 | `id UUID`、`vector vector(dim)`、`payload JSONB` | `id UUID/VARCHAR`、`FLOATVECTOR(dim)`、`payload JSONB/TEXT`、派生列、schema_version、redundant scope columns | GaussDB 更完整。 |
| 向量索引 | DiskANN/HNSW 逻辑 | `gsdiskann`/`gsivfflat` + maintenance_work_mem | 对齐并适配 GaussDB。 |
| 向量 metric | 主要 cosine `<=>` | cosine `<+>`、L2 `<->` | GaussDB 支持明确双 metric。 |
| insert | psycopg3 executemany 或 psycopg2 execute_values，不 upsert | CTE set-based upsert | GaussDB 更适合 mem0 重复 id/update 场景。 |
| search filter | 简单 `payload->>%s = %s` | scope guard + 多操作符 filter + JSON/redundant 双模式 | GaussDB 更强。 |
| keyword search | PostgreSQL FTS `to_tsvector`/`ts_rank_cd` | GaussDB native BM25 + `###` score | GaussDB 更贴近 BM25 要求。 |
| keyword 失败策略 | 异常返回 `None` | auto/required/disabled，savepoint + fallback metrics | GaussDB 更可控。 |
| search_batch | 无 override | native SQL batch + fallback | GaussDB 更强。 |
| col_info | name/count/size | name/count/schema_version/profile/modes/indexes/BM25 状态 | GaussDB 更完整。 |
| migration/backfill | 无 | v1 dry-run + derived field backfill | GaussDB 更强。 |
| 商用隔离 | 无 provider-level 强制 | 默认强制 scoped filters | GaussDB 更强。 |

结论：GaussDB 当前已经可以对标 `pgvector`，并在 mem0 商用场景所需的隔离、BM25、batch、fallback、schema 演进和测试方面明显增强。

## 8. 当前测试与验证情况

已提交的主要测试：

| 测试文件 | 内容 | 最近结果 |
|---|---|---|
| `tests/vector_stores/test_gaussdb.py` | mock/unit 回归，覆盖配置、SQL 构造、scope guard、BM25 savepoint、batch SQL、update 行为、schema meta、backfill。 | `70 passed` |
| `tests/vector_stores/test_gaussdb_p0.py` | live P0/P1/P2 能力测试，覆盖 Memory.from_config、CRUD、scope、filter matrix、compatibility、UTF-8、metric、batch、collection ops、migration、BM25、index matrix。 | 在 `mem0_e2e_db` 上 `22 passed` |
| `tests/vector_stores/test_gaussdb_e2e.py` | 早期 live e2e 样例。 | 可作为补充。 |
| `tests/vector_stores/test_gaussdb_quality.py` | 质量/性能门槛，keyword hit-rate、filter leakage、latency。 | 可按环境变量执行。 |

真实库验证结果：

- `mem0_e2e_db`：UTF-8 数据库，完整 P0 + BM25 + index matrix：`22 passed`。
- `lxm_db`：非 UTF-8 数据库，中文 round-trip 用例按设计 skip；其他能力通过。

## 9. 当前实现的边界和后续增强项

当前 P0/P1/P2 范围已经闭合，但仍有一些 P3/后续增强项：

| 项目 | 当前状态 | 建议 |
|---|---|---|
| 完整 migration framework | 目前是 v1 helper，不是完整迁移框架 | 后续增加 schema inspector、逐项 dry-run、可执行 plan、rollback notes。 |
| benchmark 常态化 | 已有 quality benchmark，但未做 CI 矩阵 | 后续按数据库规格配置 p95 门槛。 |
| CI live matrix | 本地真实库已跑通 | 后续可接入 UTF-8/BM25/index matrix 数据库环境。 |
| get 的 provider-level scope | mem0 `get(id)` 没有 filters 参数 | 短期不承诺 provider-level scoped get；长期需改 mem0 API 或 Memory 层补校验。 |
| TEXT payload 下复杂 metadata filter | 当前 redundant 模式只支持 scope columns | 如商用要求 TEXT payload 也支持任意 metadata filter，需要扩展冗余列策略或外部二级索引。 |
| BM25 语言字典策略 | 参数已支持 `bm25_dictionary` | 中文/英文/混合的默认字典策略可结合商用手册和真实语料继续固化。 |

## 10. 使用建议

### 10.1 默认商用配置

推荐用户尽量使用高层配置，避免暴露过多低层参数：

```python
config = {
    "vector_store": {
        "provider": "gaussdb",
        "config": {
            "host": "...",
            "port": 19995,
            "database": "...",
            "user": "...",
            "password": "...",
            "collection_name": "mem0",
            "embedding_model_dims": 1536,
            "profile": "commercial",
            "metadata_mode": "auto",
            "bm25_mode": "auto",
        },
    },
}
```

### 10.2 验收配置

如果要强制验收 BM25：

```python
{
    "bm25_mode": "required",
    "enable_capability_probe": True,
}
```

如果要兼容 JSONB/expression index 受限环境：

```python
{
    "profile": "compatibility",
    "metadata_mode": "compatible",
}
```

### 10.3 生产隔离建议

保持默认：

```python
{
    "require_scoped_filters": True,
    "scope_filter_keys": ["user_id", "agent_id", "run_id"],
}
```

read path 必须传：

```python
filters={"user_id": "..."}
```

或：

```python
filters={"agent_id": "...", "run_id": "..."}
```

不要把 `require_scoped_filters=False` 用在普通业务请求中，它只适合管理员任务、离线迁移或全局审计。

## 11. 最终判断

当前 GaussDB provider 的实现情况可以概括为：

1. mem0 标准接口已经完整适配。
2. mem0 检索链路需要的 dense search、keyword_search、score normalization、filter、list/delete/update 均已打通。
3. GaussDB 特有能力已落地到 Ustore、FLOATVECTOR、BM25、vector index、A 模式约束、能力探测和 fallback。
4. 相比 `pgvector`，GaussDB 当前在 batch search、scope guard、BM25 商用策略、schema meta、backfill 和测试矩阵上更完整。
5. 相比 Qdrant/ES/MongoDB 等 provider，GaussDB 的特色是数据库内闭环、事务能力、SQL 运维能力和商用隔离策略；专用向量库的优势则在原生向量服务生态和托管便利性。
6. P0/P1/P2 测试已经完成，并在 UTF-8 真实库 `mem0_e2e_db` 上全量通过。
