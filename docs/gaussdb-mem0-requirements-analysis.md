# mem0 GaussDB 适配需求分析文档

> 生成日期：2026-04-28  
> 代码基线：`codex/add-gaussdb-ustore-provider`，当前提交 `463718fc`  
> 范围：mem0 OSS Python 版 GaussDB vector store provider，目标数据库形态为 GaussDB 集中式、A 兼容模式、Ustore、`FLOATVECTOR`、向量索引与 BM25。

## 1. 背景

mem0 是面向 AI 应用的长期记忆层。它负责把用户对话、业务事件或 agent 运行过程中的信息抽取为 memory，再通过 embedding、向量检索、关键词检索、过滤隔离和更新删除能力，为后续 LLM 请求提供长期上下文。

当前 mem0 已经支持多种 vector store provider，包括 `pgvector`、`qdrant`、`mongodb`、`elasticsearch`、`opensearch`、`milvus`、`pinecone`、`weaviate`、`faiss` 等。GaussDB 团队希望把 GaussDB 作为 mem0 的一等 provider，使企业用户可以把 AI memory 数据落在 GaussDB 商用数据库中，而不是强依赖外部专用向量库或搜索服务。

本适配的核心价值是：

- 复用 GaussDB 已有的企业级数据库能力：事务、连接、权限、备份、审计、运维体系。
- 用 GaussDB 原生 `FLOATVECTOR` 和向量索引承载 mem0 semantic search。
- 用 GaussDB 原生 BM25 承载 mem0 `keyword_search`，补齐混合检索链路。
- 通过 `user_id`、`agent_id`、`run_id` 的 scope filters 满足多租户 memory 隔离。
- 面向集中式 A 模式 Ustore 商用形态提供能力探测、fallback、测试和验收口径。

## 2. 目标用户与使用场景

### 2.1 目标用户

| 用户角色 | 诉求 |
|---|---|
| GaussDB 产品与解决方案团队 | 需要展示 GaussDB 可承载 AI memory 和向量检索业务。 |
| AI 应用开发者 | 希望用 mem0 快速接入长期记忆，不想单独维护多个存储系统。 |
| 企业客户 DBA/运维 | 希望 memory 数据留在企业数据库内，便于权限、备份、审计和监控。 |
| 平台架构师 | 需要评估 GaussDB 与 pgvector、Qdrant、MongoDB 等 provider 的能力差异。 |
| 测试与交付团队 | 需要明确 P0/P1/P2 验收项、真实库验证方式和边界条件。 |

### 2.2 典型应用场景

| 场景 | mem0 的作用 | GaussDB 适配价值 |
|---|---|---|
| AI 个人助理 | 记住用户偏好、习惯、历史任务。 | 使用 scope filters 隔离不同用户，向量 + BM25 找回相关偏好。 |
| 智能客服 | 记住客户历史问题、工单状态、产品偏好。 | memory 与企业业务数据同库治理，便于审计和删除。 |
| 企业知识助手 | 记住员工查询习惯、项目上下文和团队术语。 | 数据留在企业数据库，降低外部服务依赖。 |
| Agent 平台 | 按 agent/run 记住任务过程、约束和阶段性结果。 | `agent_id`、`run_id` 可直接映射为 GaussDB filter/index。 |
| CRM Copilot | 记住客户画像、沟通记录和下一步动作。 | 关系型数据库更容易和既有 CRM 数据链路集成。 |
| 合规行业助手 | 需要隔离、可审计、可删除、可回溯 memory。 | GaussDB 的事务、权限和运维能力更适合商用合规。 |

## 3. 范围

### 3.1 本期范围

本期适配覆盖 mem0 Python 版 vector store provider：

- 在 mem0 factory/config 路径中注册 `gaussdb` provider。
- 实现 `VectorStoreBase` 标准接口：
  - `create_col`
  - `insert`
  - `search`
  - `delete`
  - `update`
  - `get`
  - `list_cols`
  - `delete_col`
  - `col_info`
  - `list`
  - `reset`
- 实现 mem0 可选增强接口：
  - `keyword_search`
  - `search_batch`
- 支持 GaussDB 集中式 A 模式 Ustore 表。
- 支持 GaussDB 分布式兼容建表模式，用于在分布式库上验证 mem0 provider 基础链路。
- 支持 `FLOATVECTOR`、`gsdiskann`、`gsivfflat`。
- 支持 BM25 索引和 BM25 score 查询。
- 支持 metadata filters 和商用默认 scope 隔离。
- 支持能力探测、fallback、schema metadata、回填、观测指标和真实库测试。

### 3.2 非目标范围

- 不修改 GaussDB 内核。
- 不实现 GaussDB 控制台、云服务编排或自动建库。
- 不改变 mem0 上层 `Memory` API 的公共接口。
- 不实现 graph store。当前 mem0 该路径下主要通过 vector store 和 entity collection 支撑 entity memory。
- 不默认引入 Astore 影子表。Ustore + vector + BM25 是本期主路径。
- 不在本期承诺分布式性能最优模型。当前分布式适配以 `id` hash 分布保证兼容；scope-hash、跨 DN 全局 top-k 优化和数据倾斜治理属于后续优化范围。
- 不承诺 provider-level `get(vector_id)` 的租户过滤，因为 mem0 标准接口没有 `filters` 参数；强隔离 ID 查询需要上层 API 或 Memory 层补充。

## 4. 现状分析

### 4.1 mem0 provider 接口现状

mem0 的 `VectorStoreBase` 定义了标准生命周期接口，并提供默认的 `keyword_search` 和 `search_batch`：

- 标准接口用于 collection 与 memory record 的增删改查。
- `keyword_search` 默认返回 `None`，表示 provider 不支持关键词检索。
- `search_batch` 默认逐条调用 `search`，provider 可以覆盖为原生批量查询。

从现有 provider 看：

| Provider 类型 | 代表 | 适配重点 |
|---|---|---|
| PostgreSQL/SQL 型 | `pgvector`、`supabase`、`azure_mysql` | 表结构、连接池、事务、JSON payload、向量 SQL。 |
| 专用向量库 | `qdrant`、`milvus`、`pinecone`、`weaviate` | collection/index 创建、payload filter、向量检索。 |
| 搜索引擎型 | `elasticsearch`、`opensearch` | 向量检索 + keyword/BM25 检索。 |
| 文档数据库型 | `mongodb` | 文档 payload、Atlas vector search、text search。 |
| 本地索引型 | `faiss` | 本地索引文件、序列化安全、简单 filter。 |

GaussDB 与 `pgvector` 最接近，但相比普通 PostgreSQL 适配，需要额外处理：

- A 兼容模式类型/索引可用性差异。
- Ustore 表存储要求。
- GaussDB `FLOATVECTOR` 与向量索引语法。
- GaussDB BM25 索引语法和 score 查询。
- 商用多租户隔离策略。
- 能力探测和兼容 fallback。

### 4.2 已验证的数据库能力

本期实现以真实 GaussDB 环境验证为依据：

- `FLOATVECTOR` 列可用于向量存储。
- `gsdiskann` 与 `gsivfflat` 可用于向量索引。
- `cosine` 默认 metric 可通过 `<+>` 执行距离计算。
- `l2` metric 可通过 `<->` 执行距离计算。
- Ustore 表上可以创建 BM25 索引并执行 score 查询。
- `maintenance_work_mem` 可通过 session-local `SET LOCAL` 辅助向量索引构建。
- UTF-8 数据库可正确承载中文和中英混合 memory。

### 4.3 主要差距

在适配前，mem0 没有 GaussDB provider，用户无法通过：

```python
Memory.from_config({
    "vector_store": {
        "provider": "gaussdb",
        "config": {...}
    }
})
```

直接把 memory 落到 GaussDB。

同时，简单照搬 `pgvector` 也不够：

- GaussDB A 模式可能不完全支持 PostgreSQL JSONB/UUID/表达式索引路径。
- GaussDB 向量类型和索引语法不是 pgvector extension 的 `vector`/HNSW/IVFFlat 语法。
- 商用默认需要 scope 强隔离，而 `pgvector` 适配没有 provider-level scope guard。
- BM25 建索引失败不能破坏主表和向量索引创建事务。
- 需要真实库 P0/P1/P2 验收，不只是 mock SQL 单测。

## 5. 业务需求

### BR-01 支持 GaussDB 作为 mem0 长期记忆存储

用户应能通过 mem0 标准配置选择 `provider: gaussdb`，并使用 `Memory.add/search/update/delete` 完成 memory 生命周期管理。

验收标准：

- `Memory.from_config` 能实例化 GaussDB provider。
- `add` 写入后可以通过 `search` 找回。
- `update` 后检索结果可见。
- `delete` 后记录不再出现在 `get/list/search/keyword_search` 中。

### BR-02 支持企业级数据隔离

商用默认配置下，读路径应要求携带 `user_id`、`agent_id` 或 `run_id` 至少一个正向 scope 条件。

验收标准：

- 无 scope 的 `search/list/keyword_search/search_batch` 被拒绝。
- 只在 `$or`、`not` 或负向条件中出现 scope 时不得绕过隔离。
- 多租户数据同 collection 下存储时，带 scope 查询只返回匹配租户记录。

### BR-03 支持语义检索与关键词检索

GaussDB provider 应同时支持 semantic vector search 和 BM25 keyword search，以匹配 mem0 hybrid memory search 的调用链。

验收标准：

- `search` 返回 score 越大越相关的结果。
- `keyword_search` 在 BM25 可用时返回 BM25 score 排序结果。
- BM25 不可用且配置允许 fallback 时，不影响 semantic search。

### BR-04 支持 GaussDB 商用基线差异

不同 GaussDB 506/507 小版本和 A 模式能力可能存在差异，provider 不应硬编码所有能力假设，而应通过能力探测和配置 fallback 决定实际执行路径。

验收标准：

- JSONB 不可用时可 fallback 到 text payload。
- JSON 表达式索引不可用时可 fallback 到 redundant scope columns。
- BM25 建索引失败时可按配置禁用 keyword search 或 fail fast。
- vector 必需能力不可用时应 fail fast。

### BR-05 提供可交付的测试与文档

适配不仅需要代码，还需要单测、真实库测试、质量回放、性能报告和运维说明。

验收标准：

- mock 单测覆盖配置、SQL 生成、事务、fallback 和 filter。
- live P0 覆盖真实 GaussDB CRUD、检索、隔离、BM25、中文、batch 和 index matrix。
- 文档说明配置、能力、限制、验收命令和故障排查。

## 6. 功能需求

| 编号 | 需求 | 优先级 | 说明 |
|---|---|---|---|
| FR-01 | Provider 注册 | P0 | `gaussdb` 可通过 mem0 config/factory 创建。 |
| FR-02 | 配置模型 | P0 | 支持连接、collection、维度、profile、metadata、BM25、索引、重试、观测配置。 |
| FR-03 | Ustore schema | P0 | 每个 collection 映射为一张 Ustore 表和一张 schema meta 表。 |
| FR-04 | 向量写入 | P0 | 支持单条/批量 insert，并按 id upsert。 |
| FR-05 | 向量检索 | P0 | 支持 cosine 默认检索和 l2 可选检索。 |
| FR-06 | Metadata filters | P0 | 支持安全 filter key、参数化值和范围/集合/逻辑操作。 |
| FR-07 | Scope 强隔离 | P0 | 商用默认要求正向 scope predicate。 |
| FR-08 | BM25 keyword search | P0 | 支持空 query、BM25 默认参数、失败 fallback。 |
| FR-09 | Update/delete/get/list | P0 | 对齐 mem0 标准生命周期语义。 |
| FR-10 | Batch search | P1 | 支持 entity store 和多 query 批量检索。 |
| FR-11 | 能力探测 | P1 | 检测 vector、index、BM25、JSONB、UUID、表达式索引。 |
| FR-12 | 兼容模式 | P1 | `profile=compatibility` 使用更保守的 metadata/index 路径。 |
| FR-13 | Schema metadata | P1 | `col_info` 返回真实 schema version 和能力信息。 |
| FR-14 | Migration/backfill helper | P2 | 提供 dry-run 和派生字段回填能力。 |
| FR-15 | Observability | P2 | 提供 latency、fallback、retry、error metrics 与日志。 |
| FR-16 | Analyze/index ensure | P2 | 提供内部索引确保与表统计维护能力。 |

## 7. 非功能需求

### 7.1 安全

- 所有用户值必须通过 driver 参数绑定，不允许字符串拼接。
- collection/table/index/filter key 这类 SQL identifier 必须经过 allowlist 校验。
- 日志和异常不得泄露 password、token、完整 DSN。
- 测试代码不得包含真实公网地址、用户名或密码默认值。

### 7.2 可靠性

- DDL/DML 必须有明确事务边界。
- BM25 等可选能力失败应通过 savepoint 或独立事务隔离，不能拖垮主表和向量索引。
- 可重试错误仅限连接抖动、锁等待、死锁、序列化冲突等瞬态错误。
- 输入校验和 schema mismatch 不应重试。

### 7.3 性能

- 批量 insert 不应逐行 DML，应使用 set-based SQL 路径。
- 查询必须在数据库侧应用 filters 和 top-k，避免全量拉回 Python 排序。
- Scope fields 在 redundant columns 模式下必须有索引。
- `gsdiskann` 建索引需要允许配置 `maintenance_work_mem`。

### 7.4 兼容性

- 运行时使用 GaussDB 官方兼容 `psycopg2` API。
- 支持 `connection_string`、`dsn`、`url` 别名和 `GAUSSDB_*` 环境变量。
- 支持 `uuid` 和 `varchar` id fallback。
- 支持 `jsonb` payload 和 `text` payload fallback。

### 7.5 可维护性

- 高层配置优先简单：`profile`、`metadata_mode`、`bm25_mode`。
- 低层配置保留专家调优能力：`payload_storage_mode`、`filter_storage_mode`、`bm25_enabled`、`bm25_fail_fast` 等。
- 不新增非标准公共接口污染 mem0 provider 契约；索引修复能力保持私有 `_ensure_indexes()`。

## 8. 配置需求

### 8.1 推荐最小配置

```python
config = {
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
}
```

### 8.2 高层配置

| 参数 | 默认值 | 说明 |
|---|---|---|
| `profile` | `commercial` | 商用默认。启用 Ustore、能力探测、scope guard、BM25 auto。 |
| `profile=compatibility` | 无 | 更保守模式，倾向 text payload、redundant filters、`gsivfflat`。 |
| `metadata_mode` | `auto` | 自动选择 `jsonb/json_expression`，失败时 fallback。 |
| `metadata_mode=compatible/text` | 无 | 使用 text payload + redundant scope columns。 |
| `bm25_mode` | `auto` | 尝试 BM25，失败时允许禁用。 |
| `bm25_mode=required` | 无 | BM25 不可用则 fail fast。 |
| `bm25_mode=disabled` | 无 | 禁用 keyword search。 |

### 8.3 专家配置

| 参数 | 默认值 | 说明 |
|---|---|---|
| `vector_index_type` | `gsdiskann` | 可选 `gsdiskann`、`gsivfflat`。 |
| `vector_metric` | `cosine` | 可选 `cosine`、`l2`。 |
| `payload_storage_mode` | 由高层模式派生 | `jsonb` 或 `text`。 |
| `filter_storage_mode` | 由高层模式派生 | `json_expression` 或 `redundant_columns`。 |
| `require_scoped_filters` | `True` | 商用读路径强制 scope。 |
| `scope_filter_keys` | `user_id/agent_id/run_id` | 可作为租户隔离字段的 key。 |
| `allowed_filter_keys` | `None` | 额外限制可查询 metadata key。 |
| `bm25_ranking_metric` | `0` | GaussDB BM25 OKAPI 默认。 |
| `bm25_ncandidates` | `128` | BM25 candidate count。 |
| `bm25_dictionary` | `None` | 使用数据库默认 dictionary。 |

## 9. 验收需求

### 9.1 P0 验收

P0 证明“能真实使用”：

- `Memory.from_config` 上层链路接入 GaussDB。
- provider CRUD、batch upsert、update、delete。
- scope search/list/batch 不串租户。
- 非约束性 scope filters 被拒绝。
- JSON payload filter operator matrix。
- compatibility profile 与 redundant scope partial update。
- UTF-8 中文和中英混合 payload round trip。
- payload-only、vector-only update。
- cosine/l2 metric exact-match 检索。
- native batch search 与 sequential search 一致。
- schema info、analyze、reset、list_cols。
- migration dry-run 与 backfill helper。
- BM25 keyword search scoped 行为和空 query 行为。
- `gsdiskann`/`gsivfflat` 与 `cosine`/`l2` 组合建索引和检索。

### 9.2 P1 验收

P1 证明“商用风险可控”：

- 能力探测 fallback 正确。
- BM25 建索引失败通过 savepoint 保护事务。
- JSONB 失败 fallback 到 text payload + redundant filters。
- scope guard 无法通过 `$or`、`not`、负向条件绕过。
- `col_info` 读取 schema meta 中的真实 schema version。
- batch search fallback 语义清晰。
- 部分 payload update 不清空 redundant scope columns。

### 9.3 P2 验收

P2 证明“可运维、可演进”：

- migration dry-run 输出 action、影响行数和限制说明。
- `backfill_derived_fields` 可回填 `text_lemmatized` 和 redundant scope columns。
- `analyze` 可维护统计信息。
- quality replay 包含语义召回、关键词命中、隔离、更新可见性、删除残留。
- benchmark 可输出 insert/search/update/delete 延迟报告。

## 10. 测试现状

当前仓库已具备以下测试文件：

| 文件 | 作用 |
|---|---|
| `tests/vector_stores/test_gaussdb.py` | Mock 单测，覆盖配置、SQL、filters、fallback、事务、update、schema。 |
| `tests/vector_stores/test_gaussdb_p0.py` | 真实库 P0 验收，覆盖 Memory 链路和核心商用能力。 |
| `tests/vector_stores/test_gaussdb_e2e.py` | 真实库 e2e，用较少用例串起 Ustore/vector/BM25/CRUD/batch。 |
| `tests/vector_stores/test_gaussdb_quality.py` | 质量回放、并发和 benchmark 报告。 |

已执行结果：

```text
pytest tests/vector_stores/test_gaussdb.py \
       tests/vector_stores/test_gaussdb_p0.py \
       tests/vector_stores/test_gaussdb_e2e.py \
       tests/vector_stores/test_gaussdb_quality.py -q

96 passed, 1 skipped
```

单独真实库 P0：

```text
pytest tests/vector_stores/test_gaussdb_p0.py -q

22 passed
```

mem0 全仓测试说明：

- 全仓存在 900+ 测试，但许多 provider 测试需要可选 SDK 或真实云服务环境。
- 当前本地 `pytest tests --collect-only -q` 可收集 916 项，因缺少 Azure、MongoDB、Milvus、Chroma、Pinecone、Weaviate 等可选依赖产生 collection errors。
- 这不是 GaussDB 适配失败，而是 mem0 多 provider 测试体系的依赖隔离现状。

## 11. 风险与约束

| 风险 | 影响 | 缓解 |
|---|---|---|
| GaussDB 小版本能力差异 | JSONB、UUID、表达式索引、BM25 语法可能不同。 | 能力探测 + fallback + fail-fast。 |
| BM25 语言质量差异 | 中文/中英混合排序可能需要调参。 | 固定质量回放集 + 暴露 BM25 参数。 |
| Scope 配置不当 | 可能导致跨租户读取。 | 商用默认 `require_scoped_filters=True`。 |
| 非 UTF-8 数据库 | 中文 payload 可能乱码或失败。 | 中文验收要求 UTF-8 数据库。 |
| 大规模建索引资源消耗 | `gsdiskann` 建索引可能受内存限制。 | `SET LOCAL maintenance_work_mem` + 文档化运维前置条件。 |
| `get(id)` 无 filters | provider 层无法过滤 ID 查询。 | 文档明确限制，必要时在 Memory/API 层补 scope 校验。 |

## 12. 交付物

- GaussDB provider 代码：`mem0/vector_stores/gaussdb.py`
- GaussDB config 模型：`mem0/configs/vector_stores/gaussdb.py`
- factory/config 注册路径：`mem0/vector_stores/configs.py`、`mem0/utils/factory.py`
- 单测与真实库测试：`tests/vector_stores/test_gaussdb*.py`
- 深度说明与对比文档：`docs/gaussdb-mem0-integration-deep-dive.md`
- 本需求分析文档：`docs/gaussdb-mem0-requirements-analysis.md`
- 技术设计文档：`docs/gaussdb-mem0-technical-design.md`
