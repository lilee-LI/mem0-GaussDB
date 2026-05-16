# GaussDB 适配 mem0 详细技术说明

## 一、GaussDB 适配点全景

本文档完整梳理 GaussDB vector store provider 对 mem0 框架的每一个适配点，说明"做了什么"和"为什么这么做"。

---

### 1. 向量类型：FLOATVECTOR 替代 pgvector 的 vector

**做了什么**：所有向量列使用 GaussDB 原生 `FLOATVECTOR(N)` 类型，而非 PostgreSQL 的 `vector(N)` 扩展类型。

**为什么**：
- GaussDB 不支持 pgvector 扩展（`CREATE EXTENSION vector` 会失败）
- GaussDB 内核原生提供 `FLOATVECTOR` 类型，无需安装任何扩展
- `FLOATVECTOR` 的距离运算符也不同：cosine 用 `<+>`，L2 用 `<->`（pgvector 用 `<=>`/`<->`）

**代码位置**：`gaussdb.py` 第 594 行建表、第 810 行搜索、第 949 行更新

---

### 2. 存储引擎：Ustore 替代默认 heap

**做了什么**：建表时强制使用 `WITH (storage_type=ustore)`。

**为什么**：
- Ustore 是 GaussDB 的多版本存储引擎，支持原地更新（in-place update），减少 MVCC 膨胀
- mem0 的 memory 会频繁 update（payload 变更、text_lemmatized 回填），Ustore 比 Astore（append-only）更适合这种写入模式
- Ustore 对 MERGE INTO 的并发性能更好

**代码位置**：`gaussdb.py` 第 149 行 `self.table_storage = "ustore"`，第 385 行 `_create_table_suffix_sql`

---

### 3. Upsert 机制：MERGE INTO 替代 INSERT ON CONFLICT

**做了什么**：使用 GaussDB A 模式（Oracle 兼容模式）的 `MERGE INTO ... USING (VALUES ...) AS src ON (...) WHEN MATCHED THEN UPDATE WHEN NOT MATCHED THEN INSERT` 语法。

**为什么**：
- GaussDB 集中式 A 模式不支持 PostgreSQL 的 `INSERT ... ON CONFLICT DO UPDATE`
- `MERGE INTO` 是 SQL:2003 标准语法，GaussDB A 模式原生支持
- `MERGE INTO` 是原子操作，避免了 `UPDATE + INSERT WHERE NOT EXISTS` 的竞态条件
- 支持批量 upsert：多行 VALUES 一次性传入，减少网络往返

**代码位置**：`gaussdb.py` 第 755-767 行 `insert` 方法

---

### 4. 向量索引：gsdiskann / gsivfflat 替代 pgvector 的 ivfflat/hnsw

**做了什么**：
- 默认使用 `gsdiskann` 索引类型
- 可选 `gsivfflat`
- 高维向量（>1024 维）自动配置 `enable_vector_copy=false` + `subgraph_count`

**为什么**：
- GaussDB 不支持 pgvector 的 `ivfflat` 或 `hnsw` 索引
- `gsdiskann` 是 GaussDB 原生的 DiskANN 实现，支持最高 4096 维，适合生产环境大规模向量检索
- `gsivfflat` 是 GaussDB 原生的 IVF-Flat 实现，建索引更快但召回率略低
- 高维（>1024）时 GsDiskANN 需要特殊参数避免内存溢出

**代码位置**：`gaussdb.py` 第 643-668 行 `_create_vector_index`

---

### 5. BM25 关键词搜索：GaussDB 原生 `###` 运算符

**做了什么**：
- 建表时创建 BM25 索引：`CREATE INDEX ... USING bm25 (text_lemmatized)`
- 搜索时使用 `text_lemmatized ### query AS score` 语法
- 搜索前设置 `SET LOCAL bm25_ranking_metric`、`bm25_ncandidates`、`enable_seqscan = off`

**为什么**：
- mem0 的搜索链路需要 semantic + keyword 双路召回做融合排序
- GaussDB 原生支持 BM25 索引和 `###` 评分运算符，无需外部全文搜索引擎
- pgvector 版本没有 BM25 能力，Qdrant 版本需要 fastembed 做客户端 BM25 编码
- GaussDB 的 BM25 是服务端计算，性能更好且不需要客户端维护倒排索引

**代码位置**：`gaussdb.py` 第 831-865 行 `keyword_search`，第 867-873 行 `_apply_bm25_settings`

---

### 6. 批量搜索：CTE + ROW_NUMBER 替代 LATERAL JOIN

**做了什么**：使用 CTE（WITH 子句）+ `ROW_NUMBER() OVER (PARTITION BY query_index ORDER BY distance)` 实现单次往返的多向量搜索。

**为什么**：
- GaussDB A 模式不支持 PostgreSQL 的 `LATERAL JOIN`
- 多次单独 `search()` 调用会产生 N 次网络往返，延迟线性增长
- CTE + window function 是 SQL 标准语法，GaussDB 完全支持
- 如果 CTE 方案失败（极端情况），自动 fallback 到逐条搜索

**代码位置**：`gaussdb.py` 第 875-933 行 `search_batch`

---

### 7. 分布式模式：DISTRIBUTE BY HASH

**做了什么**：
- 当 `deployment_mode="distributed"` 时，建表加 `DISTRIBUTE BY HASH("id")`
- 分布式模式下自动禁用 BM25（GaussDB 分布式不支持 BM25 索引）
- 分布式模式下向量维度上限为 1024

**为什么**：
- GaussDB 分布式部署要求每张表指定分布键
- 用 `id` 做 hash 分布可以保证单行操作（get/update/delete）只命中一个 DN 节点
- 分布式内核对 BM25 和高维向量有限制，需要在 provider 层提前拦截

**代码位置**：`gaussdb.py` 第 133-144 行维度限制，第 153 行 BM25 禁用，第 391-396 行分布子句

---

### 8. 能力探测：_probe_capabilities

**做了什么**：初始化时创建一个临时探测表，逐项测试服务器实际支持的能力：
1. `SHOW enable_vectordb` — 向量功能是否开启
2. 创建 `FLOATVECTOR` 列 — 向量类型是否可用
3. 创建向量索引 — 索引类型是否支持
4. 创建 BM25 索引 — BM25 是否可用
5. 插入 JSONB 数据 — JSONB 类型是否支持
6. 创建表达式索引 `(payload->>'user_id')` — 表达式索引是否支持

每项测试失败时优雅降级而非崩溃。

**为什么**：
- GaussDB 不同版本/部署模式的能力差异很大（有的版本没有 BM25，有的没有 JSONB）
- 不能假设所有 GaussDB 实例都有相同能力
- 运行时探测比硬编码版本号更可靠（同一版本号的不同编译选项可能有不同能力）
- 探测失败时降级（如 JSONB→TEXT、json_expression→redundant_columns）保证 provider 在低版本上也能工作

**代码位置**：`gaussdb.py` 第 421-571 行 `_probe_capabilities`

---

### 9. Metadata 存储双模式：JSONB vs TEXT + 冗余列

**做了什么**：
- 默认模式 `payload_storage_mode="jsonb"` + `filter_storage_mode="json_expression"`：payload 存为 JSONB，过滤用 `payload->>'key'` 表达式索引
- 降级模式 `payload_storage_mode="text"` + `filter_storage_mode="redundant_columns"`：payload 存为 TEXT（JSON 字符串），过滤用独立的 `user_id`/`agent_id`/`run_id` VARCHAR 列

**为什么**：
- 某些 GaussDB 版本或配置不支持 JSONB 类型
- 某些版本支持 JSONB 但不支持表达式索引
- 冗余列模式虽然不够灵活（只能过滤预定义的 scope 字段），但兼容性最好
- 两种模式对上层 API 完全透明，调用方无需感知底层存储差异

**代码位置**：`gaussdb.py` 第 157-158 行默认值，第 549-558 行降级逻辑，第 1298-1307 行 `_field_sql`

---

### 10. 多租户 Scope 强隔离

**做了什么**：
- 默认 `require_scoped_filters=True`
- `search`、`keyword_search`、`list` 操作必须携带至少一个有效的 scope filter（`user_id`/`agent_id`/`run_id`）
- 无效值（None、空字符串、`"*"`）不算有效 scope
- `$or` 中所有分支都必须有 scope 才算有效

**为什么**：
- mem0 是多租户记忆系统，一个 GaussDB 表存储所有用户的记忆
- 如果不强制 scope，一个用户的搜索可能返回其他用户的记忆（数据泄露）
- 这是企业级安全要求，pgvector 版本没有这个保护
- `get(id)` 和 `delete(id)` 不强制 scope，因为 UUID 本身就是全局唯一的

**代码位置**：`gaussdb.py` 第 160-161 行配置，第 1172-1216 行 `_build_where_clause` + `_has_scope_filter`

---

### 11. 连接池 + 重试机制

**做了什么**：
- 使用 `psycopg2.pool.ThreadedConnectionPool`（线程安全连接池）
- 所有数据库操作包裹在 `_run_with_retry` 中，默认重试 2 次，指数退避
- 可重试错误：connection、timeout、deadlock、lock wait、serialization failure、server closed

**为什么**：
- GaussDB 企业部署可能有网络抖动、主备切换、连接超时
- pgvector 版本没有重试，一次失败就抛异常，不适合生产环境
- 连接池避免每次操作都建立新连接（GaussDB 连接建立开销较大）
- 指数退避避免重试风暴

**代码位置**：`gaussdb.py` 第 245-254 行连接池创建，第 330-349 行 `_run_with_retry`

---

### 12. 事务管理：显式 commit/rollback + SAVEPOINT

**做了什么**：
- `_get_cursor(commit=True)` 用于写操作，成功时 commit，异常时 rollback
- `_get_cursor(commit=False)` 用于只读操作，结束时 rollback（释放事务）
- 索引创建使用 SAVEPOINT 隔离，单个索引失败不影响整个事务

**为什么**：
- GaussDB 默认不自动提交（autocommit=False）
- 如果不显式 commit，写入不会持久化
- 如果不显式 rollback 只读事务，连接会保持在 idle-in-transaction 状态，浪费资源
- SAVEPOINT 保证可选索引（如表达式索引）创建失败时不会 abort 整个建表事务

**代码位置**：`gaussdb.py` 第 307-328 行 `_get_cursor`，第 694-720 行 `_create_filter_indexes`

---

### 13. 编码处理：强制 UTF-8 + JSON 序列化绕过 latin-1

**做了什么**：
- 每次获取连接时设置 `conn.set_client_encoding('UTF8')`
- payload 序列化使用 `json.dumps(payload, ensure_ascii=False)` 而非 psycopg2 的 `Json` 适配器

**为什么**：
- GaussDB 的 `server_encoding` 可能是 `SQL_ASCII`（不强制 UTF-8）
- psycopg2 的 `Json` 适配器内部使用 `latin-1` 编码调用 `getquoted()`，遇到中文等非 ASCII 字符会报错
- 直接用 `json.dumps` 生成 UTF-8 字符串，配合 `client_encoding=UTF8`，确保中文/多语言数据正确传输
- 这是一个实际踩过的坑：不绕过 `Json` 适配器，中文 payload 会报 `UnicodeEncodeError`

**代码位置**：`gaussdb.py` 第 311-312 行设置编码，第 398-405 行 `_payload_value`

---

### 14. Schema 版本管理

**做了什么**：
- 每个 collection 有一个对应的 `{collection_name}_schema_meta` 表
- 记录 `collection_name`、`schema_version`、`updated_at`
- 每行数据也有 `schema_version` 列

**为什么**：
- 未来 schema 升级（如加列、改索引）需要知道当前版本
- pgvector 版本没有 schema 管理，升级时只能手动 ALTER TABLE
- 行级 `schema_version` 支持渐进式迁移（新旧 schema 数据共存）

**代码位置**：`gaussdb.py` 第 617-641 行 `_create_schema_meta` + `_upsert_schema_meta`

---

### 15. 可观测性：延迟指标 + 慢查询日志

**做了什么**：
- 每个操作记录延迟和成功/失败状态
- 超过 `slow_query_ms`（默认 1000ms）的操作输出 WARNING 日志
- 内部 `metrics` 字典记录各类计数（error_count、retry_count、fallback_count）

**为什么**：
- 生产环境需要监控 GaussDB 操作的延迟分布
- 慢查询日志帮助 DBA 定位性能瓶颈
- pgvector 版本没有任何可观测性，出问题时无法定位

**代码位置**：`gaussdb.py` 第 356-368 行 `_record_latency` + `_increment_metric`

---

### 16. SQL 注入防护

**做了什么**：
- 所有标识符（表名、列名、索引名）通过 `_validate_identifier` 正则校验 + `_quote_identifier` 双引号包裹
- 所有 filter key 通过 `_validate_filter_key` 正则校验（`^[A-Za-z_][A-Za-z0-9_]{0,127}$`）
- 所有 filter value 使用 `%s` 参数化传入
- DSN 中的密码在日志中被 `_sanitize_dsn` 脱敏

**为什么**：
- filter key 需要拼入 SQL（`payload->>'key'`），不能参数化，必须白名单校验
- 标识符双引号包裹防止保留字冲突
- 这是 OWASP Top 10 的基本要求

**代码位置**：`gaussdb.py` 第 213-238 行校验方法，第 294-305 行 DSN 脱敏

---

### 17. 分数归一化

**做了什么**：`score = 1.0 / (1.0 + max(distance, 0.0))`

**为什么**：
- GaussDB 返回的是距离（越小越相似），mem0 上层期望的是相似度分数（越大越相似）
- cosine 距离范围 [0, 2]，归一化后映射到 (0, 1]
- 与 Qdrant 等返回相似度分数的 provider 对齐，方便上层做统一的阈值判断

**代码位置**：`gaussdb.py` 第 822-826 行

---

### 18. 索引命名：hash 截断防超长

**做了什么**：索引名 = `{collection_name}_{suffix}`，超过 63 字符时用 SHA1 hash 截断。

**为什么**：
- GaussDB（和 PostgreSQL）的标识符长度限制为 63 字节
- collection_name 可能很长，直接拼接会超限导致建索引失败
- hash 截断保证唯一性且不超限

**代码位置**：`gaussdb.py` 第 224-231 行 `_index_name`

---

### 19. 派生字段回填：backfill_derived_fields

**做了什么**：
- 对已有数据补填 `memory` 和 `text_lemmatized` 列
- 支持 dry_run 模式（只统计需要回填的行数）
- 从 `payload->>'data'` 或 `payload->>'memory'` 提取原始文本

**为什么**：
- BM25 搜索依赖 `text_lemmatized` 列，旧数据可能没有这个字段
- 提供 dry_run 让用户评估影响范围后再执行
- pgvector 版本没有这个能力，BM25 字段需要手动维护

**代码位置**：`gaussdb.py` 第 1124-1163 行 `backfill_derived_fields`

---

### 20. Range Filter 降级处理

**做了什么**：`gt/gte/lt/lte` 操作符不生成 SQL 比较，而是 log warning 并按字面值等值匹配。

**为什么**：
- GaussDB 的 `payload->>'key'` 返回的是字符串，字符串比较 `'10' < '2'` 语义错误
- 要正确支持 range 需要类型映射（知道哪个字段是数字/日期），当前版本没有 metadata schema
- 与其静默产生错误结果，不如明确告知用户不支持
- 未来引入 typed metadata 后再开放

**代码位置**：`gaussdb.py` 第 1259-1265 行

---

## 二、与其他 Vector Store 适配方式对比

### PGVector 适配方式

| 方面 | PGVector 做法 | 备注 |
|------|--------------|------|
| 向量类型 | `vector(N)` 扩展类型 | 需要 `CREATE EXTENSION vector` |
| 存储引擎 | 默认 heap | 无特殊配置 |
| Upsert | psycopg2 `execute_values` 或 psycopg3 `executemany` | 不是原子 upsert，是批量 INSERT |
| 向量索引 | DiskANN（需 vectorscale 扩展）或 HNSW | 都是第三方扩展 |
| BM25 | 无 | 不支持关键词搜索 |
| 批量搜索 | 无原生实现 | 使用基类默认的逐条搜索 |
| 分布式 | 不支持 | 仅单机 |
| 能力探测 | 无 | 假设所有能力都可用 |
| 重试 | 无 | 一次失败即抛异常 |
| 可观测性 | 无 | 无延迟/错误指标 |
| Scope 隔离 | 无强制 | 不校验是否带 scope filter |
| Schema 管理 | 无 | 无版本追踪 |
| 编码处理 | 无特殊处理 | 依赖 psycopg2 默认行为 |
| 连接池 | psycopg2 ThreadedConnectionPool 或 psycopg3 ConnectionPool | 有池但无重试 |
| Filter | 仅 `payload->>%s = %s` 等值 | 无复杂 filter 表达式 |

### Qdrant 适配方式

| 方面 | Qdrant 做法 | 备注 |
|------|------------|------|
| 向量类型 | Qdrant 原生 vector | 通过 SDK 操作 |
| 存储引擎 | Qdrant 内部管理 | 用户不可控 |
| Upsert | `client.upsert(points=[...])` | SDK 原生支持 |
| 向量索引 | HNSW（Qdrant 内置） | 自动管理 |
| BM25 | fastembed 客户端编码 + sparse vector | 需要额外依赖 |
| 批量搜索 | `client.query_batch_points` | SDK 原生支持 |
| 分布式 | Qdrant Cloud 自动分片 | 用户透明 |
| 能力探测 | 检查 collection 是否有 bm25 sparse vector slot | 简单检查 |
| 重试 | Qdrant SDK 内置 | 用户不需要自己实现 |
| 可观测性 | 无（依赖 Qdrant 服务端监控） | Provider 层无指标 |
| Scope 隔离 | 无强制 | 不校验 |
| Schema 管理 | 无 | Qdrant 自动管理 |
| Filter | Qdrant 原生 filter 模型（Range、Match、MatchAny 等） | 功能丰富 |

### 对比总结

GaussDB provider 是 mem0 所有 vector store 适配中**功能最完整**的一个：

| 能力 | GaussDB | PGVector | Qdrant | Milvus | Elasticsearch |
|------|---------|----------|--------|--------|---------------|
| 向量搜索 | ✅ | ✅ | ✅ | ✅ | ✅ |
| BM25 关键词搜索 | ✅ 服务端原生 | ❌ | ✅ 客户端编码 | ❌ | ✅ 服务端原生 |
| 原生批量搜索 | ✅ CTE | ❌ 逐条 | ✅ SDK | ❌ 逐条 | ❌ 逐条 |
| 分布式支持 | ✅ | ❌ | ✅ 云端 | ✅ | ✅ |
| 能力探测 | ✅ 运行时 | ❌ | 部分 | ❌ | ❌ |
| 重试机制 | ✅ 指数退避 | ❌ | SDK 内置 | ❌ | ❌ |
| 可观测性 | ✅ | ❌ | ❌ | ❌ | ❌ |
| Scope 强隔离 | ✅ | ❌ | ❌ | ❌ | ❌ |
| Schema 版本管理 | ✅ | ❌ | ❌ | ❌ | ❌ |
| 事务 SAVEPOINT | ✅ | ❌ | N/A | N/A | N/A |

---

## 三、GaussDB Provider 内部架构

```
┌─────────────────────────────────────────────────────────────┐
│                    mem0 Memory Layer                         │
│  Memory.add() / Memory.search() / Memory.get_all()          │
└─────────────────────────┬───────────────────────────────────┘
                          │ VectorStoreBase interface
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                   GaussDB Provider                           │
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ Connection   │  │ Capability   │  │ Observability    │  │
│  │ Pool + Retry │  │ Probing      │  │ Metrics + Logs   │  │
│  └──────┬───────┘  └──────┬───────┘  └──────────────────┘  │
│         │                  │                                 │
│  ┌──────┴──────────────────┴────────────────────────────┐   │
│  │              Core Operations                          │   │
│  │  insert (MERGE INTO)                                  │   │
│  │  search (FLOATVECTOR <+> cosine)                      │   │
│  │  keyword_search (BM25 ###)                            │   │
│  │  search_batch (CTE + ROW_NUMBER)                      │   │
│  │  update / delete / get / list                         │   │
│  └──────┬───────────────────────────────────────────────┘   │
│         │                                                    │
│  ┌──────┴───────────────────────────────────────────────┐   │
│  │              Filter Engine                             │   │
│  │  $and / $or / $not / eq / ne / in / nin               │   │
│  │  contains / icontains / startswith                     │   │
│  │  Scope guard (require_scoped_filters)                  │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────┬───────────────────────────────────┘
                          │ psycopg2 (GaussDB official driver)
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    GaussDB Server                            │
│  Ustore Table + FLOATVECTOR + GsDiskANN + BM25 Index        │
└─────────────────────────────────────────────────────────────┘
```
