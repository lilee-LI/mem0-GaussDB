# mem0 GaussDB 适配实现深度说明与同类 Provider 对比

> 生成日期：2026-04-28  
> 代码基线：`codex/add-gaussdb-ustore-provider`，当前提交 `dd33a16f`  
> 范围：基于当前仓库代码分析 `mem0` 执行链路、GaussDB provider 设计逻辑、已适配功能点、测试覆盖，以及与 mem0 已有 vector store provider 的能力对比。

## 1. 总体结论

当前 GaussDB 适配已经不是“只实现基础向量 CRUD”的 provider，而是按 mem0 v1.1/v3 检索链路做了完整商用化补齐：

- 接入 mem0 标准 `VectorStoreBase` 必选接口：`create_col`、`insert`、`search`、`delete`、`update`、`get`、`list_cols`、`delete_col`、`col_info`、`list`、`reset`。
- 接入 mem0 可选增强接口：`keyword_search` 和 `search_batch`。在当前 mem0 25 个 vector store provider 中，只有 `qdrant` 和 `gaussdb` 同时 override 了 `keyword_search` 与 `search_batch`。
- 针对 GaussDB 商用形态额外补齐：集中式、A 兼容模式、Ustore、`FLOATVECTOR`、`gsdiskann`/`gsivfflat`、原生 BM25、租户 scope 强制隔离、能力探测、事务 savepoint 保护、schema meta、backfill、连接池、重试、观测指标和 live P0/P1/P2 验证。
- 适配深度对标 `pgvector`，但在以下方面已经超过当前 mem0 `pgvector.py`：原生 batch search、scope guard 防绕过、JSONB 失败 fallback、TEXT payload + redundant scope columns 兼容模式、BM25 建索引事务保护、schema version 元数据、live 能力矩阵测试。

如果用“mem0 现有 provider 适配成熟度”来衡量，GaussDB 当前属于高适配度，功能面已经接近或超过 `pgvector`，并在商用可交付性上更重。

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

