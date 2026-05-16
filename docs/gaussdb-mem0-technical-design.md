# mem0 GaussDB 适配技术设计

> 更新时间: 2026-05-16  
> 设计对象: `mem0.vector_stores.gaussdb.GaussDB`  
> 对应代码: `mem0/mem0/vector_stores/gaussdb.py`

## 1. 设计原则

GaussDB provider 的设计遵循以下原则:

1. 以 mem0 现有 provider 契约为边界，不发明新的公共接口。
2. 以商用正确性优先，不为了表面兼容而返回错误结果。
3. 能力探测和降级必须可解释。
4. 集中式优先保证完整体验，分布式优先保证兼容与可验证。
5. 过滤语义和索引能力分离。
6. 单实例绑定单 collection，不做隐式切表。

## 2. 整体架构

```text
Memory.from_config
  -> VectorStoreFactory.create("gaussdb", config)
  -> GaussDB.__init__
     -> _create_connection_pool()
     -> _probe_capabilities()
     -> auto_create -> list_cols() / create_col()

运行期主链路:
  add    -> insert()
  search -> search() + keyword_search()
  batch  -> search_batch()
  admin  -> col_info() / list_cols() / reset() / analyze()
```

GaussDB provider 只负责向量存储与检索层，不负责:

- LLM memory 提取
- embedding 生成
- semantic/BM25/entity 融合排序
- 上层鉴权

这些逻辑仍在 mem0 的 `Memory` 层完成。

## 3. 配置设计

### 3.1 当前真实配置项

当前 `GaussDBConfig` 只支持以下字段:

| 分类 | 字段 |
|---|---|
| 连接 | `database`, `user`, `password`, `host`, `port`, `connection_string`, `sslmode`, `sslrootcert` |
| collection | `collection_name`, `embedding_model_dims` |
| 连接池 | `minconn`, `maxconn` |
| 部署 | `deployment_mode` |
| 向量 | `vector_index_type`, `vector_metric` |
| 运维 | `auto_create`, `require_scoped_filters` |

说明:

- 默认 `minconn=1`, `maxconn=5`
- 默认 `deployment_mode="centralized"`
- 默认 `vector_index_type="gsdiskann"`
- 默认 `vector_metric="cosine"`
- 默认 `require_scoped_filters=True`

建议对外按两层理解这些配置:

- 基础项: 连接信息、`collection_name`、`embedding_model_dims`、`deployment_mode`
- 高阶项: `sslmode`、`sslrootcert`、`minconn`、`maxconn`、`vector_index_type`、`vector_metric`、`auto_create`、`require_scoped_filters`

这里的“基础项”当前是文档层约定，不是运行时强制必填。也就是说，代码仍然保留默认值，但商用接入建议显式填写这些关键字段，避免隐式落到默认 collection、默认维度或默认部署模式。

### 3.2 为什么不暴露更多高阶配置

当前实现里，`payload_storage_mode`、`filter_storage_mode`、`bm25_enabled` 等行为参数仍然是 provider 内部默认值；`require_scoped_filters` 已作为高级安全配置正式暴露，默认保持开启。

原因:

- P0 阶段优先保证接口简洁和默认行为稳定。
- capability probe 已经承担了自动选择和自动降级责任。
- 过早暴露底层旋钮，会让用户文档复杂度远大于收益。
- `require_scoped_filters` 之所以例外开放，是因为它直接决定检索类接口是否强制作用域约束，属于商用安全策略，而不只是调优项。

因此当前推荐策略是:

- 基础项在接入文档和样例中显式给出
- 高阶项保留默认值，按需覆盖
- 暂不继续开放 `payload_storage_mode`、`filter_storage_mode`、`bm25_enabled`、`retry_attempts` 等更底层旋钮

### 3.3 环境变量解析

当前支持以下环境变量:

- `GAUSSDB_CONNECTION_STRING`
- `GAUSSDB_DSN`
- `GAUSSDB_URL`
- `GAUSSDB_HOST`
- `GAUSSDB_PORT`
- `GAUSSDB_DATABASE`
- `GAUSSDB_DBNAME`
- `GAUSSDB_USER`
- `GAUSSDB_PASSWORD`
- `GAUSSDB_SSLMODE`
- `GAUSSDB_SSLROOTCERT`

设计目的:

- 对齐企业环境中“配置来自环境变量”的常见部署方式。
- 允许连接信息与业务代码解耦。

## 4. 连接池设计

### 4.1 当前实现

GaussDB 使用 `psycopg2.pool.ThreadedConnectionPool`:

- `minconn=1`
- `maxconn=5`

设计考虑:

- 与 mem0 中 `pgvector` 和 `azure_mysql` 的 SQL provider 默认值保持一致。
- 保守默认值更适合 SDK/provider 层。
- 具体并发容量应由部署方根据 worker 数和数据库容量调优。

### 4.2 与其他 provider 对比

| Provider | 连接池设计 |
|---|---|
| GaussDB | 显式连接池，默认 1/5 |
| pgvector | 显式连接池，默认 1/5 |
| Azure MySQL | 显式连接池，默认 1/5 |
| MongoDB | 依赖 MongoClient 默认池 |
| Qdrant | 依赖 QdrantClient 默认 HTTP 池 |
| Redis | 依赖 redis-py 默认连接池 |
| OpenSearch | 显式 `pool_maxsize=20` |

结论:

- GaussDB 的连接池方案属于 SQL provider 常规做法，不是特例。

## 5. collection 与 schema 设计

### 5.1 单实例单 collection

一个 `GaussDB` 实例只绑定一个 `self.collection_name`。  
它对应:

- 一张主表
- 一张 schema meta 表
- 若干索引

当前不支持:

```python
db.create_col(name="other_collection")
```

设计原因:

- 避免“实例状态绑定 A，建表却建到 B”的半支持语义。
- 与 `pgvector`/`MongoDB`/`Qdrant` 的单 collection 实例模型保持一致。

### 5.2 主表结构

当前主表核心字段如下:

| 字段 | 说明 |
|---|---|
| `id` | 主键，当前默认 `UUID` |
| `vector` | `FLOATVECTOR(dims)` |
| `payload` | `JSONB` 或 `TEXT` |
| `memory` | 从 payload 派生出的主文本 |
| `text_lemmatized` | 用于 BM25 |
| `created_at` | 创建时间 |
| `updated_at` | 更新时间 |
| `schema_version` | 行级 schema 版本 |
| `user_id/agent_id/run_id` | 仅在 `redundant_columns` 模式下出现 |

### 5.3 schema meta 表

每个 collection 还会有一张:

```text
<collection_name>_schema_meta
```

用途:

- 记录 collection 的 schema version
- 供 `col_info()` 与运维工具读取

设计原因:

- 把 provider 管理状态从业务表中解耦出来。
- 为后续迁移与回填提供最小元数据基座。

## 6. capability probe 设计

### 6.1 探测目标

构造阶段 `_probe_capabilities()` 会探测:

- `enable_vectordb`
- `FLOATVECTOR`
- 向量索引可用性
- `JSONB`
- BM25 index
- BM25 score query
- JSON expression index

### 6.2 为什么要探测而不是写死

GaussDB 的实际运行环境可能在以下方面存在差异:

- 集中式 / 分布式
- 小版本差异
- 某些特性是否启用
- 某些索引/操作符是否可用

如果写死假设，用户会在初始化后第一次真实写入或查询时才踩雷。  
探测的价值是尽量把失败前移到 provider 初始化阶段，或者提前决定降级路径。

### 6.3 当前降级策略

#### 向量能力不可用

- 直接失败

原因:

- 没有向量能力，provider 不成立

#### JSONB 不可用

- `payload_storage_mode = "text"`
- `filter_storage_mode = "redundant_columns"`

原因:

- JSONB 都不可用时，不可能再维持任意 metadata SQL 过滤
- 退化到最小可商用路径，只保障 scope filter

#### BM25 不可用

- `bm25_enabled = False`
- 主链路继续可用

原因:

- BM25 是增强能力，不应拖垮语义检索

#### JSON expression index 不可用

- 只 warning
- 保持 `filter_storage_mode = "json_expression"`

原因:

- 表达式索引失败不等于 payload 查询失败
- 应该损失性能，不应该损失 metadata filter 语义

## 7. 建表与索引设计

### 7.1 storage type

当前主表统一使用:

```sql
WITH (storage_type=ustore)
```

原因:

- 这是当前商用目标形态。
- 对更新型 workload 更友好。

### 7.2 集中式与分布式

当前 `distribution_mode` 规则:

- 集中式 -> `none`
- 分布式 -> `hash`

分布式 DDL 会追加:

```sql
DISTRIBUTE BY HASH ("id")
```

设计原因:

- 当前 provider 的 CRUD、检索和管理接口都以 `id` 为主键基础。
- 用 `id` 做 hash 分布最直接，兼容性最好。

### 7.3 向量索引

支持:

- `gsdiskann`
- `gsivfflat`

支持 metric:

- `cosine`
- `l2`

映射关系:

| 业务 metric | 查询算子 | 索引 metric |
|---|---|---|
| cosine | `<+>` | `COSINE` |
| l2 | `<->` | `L2` |

设计原因:

- 对齐 GaussDB 当前向量能力。
- 对 mem0 上层隐藏底层算子差异。

### 7.4 maintenance_work_mem

建向量索引前会设置:

```sql
SET LOCAL maintenance_work_mem = ...
```

高维 `gsdiskann` 会自动把默认值从 `128MB` 提高到 `2GB`。

设计原因:

- 避免修改全局 DB 参数。
- 把优化限制在当前建索引事务内。

### 7.5 BM25 索引

BM25 索引创建走 savepoint 包裹。

设计原因:

- BM25 是可选能力。
- 即便 BM25 索引失败，也不能影响主表和向量索引创建。

### 7.6 scope filter 索引

默认会为:

- `user_id`
- `agent_id`
- `run_id`

创建过滤索引。

当前有两种实现路径:

1. `redundant_columns` 模式: 普通列索引
2. `json_expression` 模式: `payload->>'key'` 表达式索引

当前每个索引都用 savepoint 包裹，失败仅 warning。

## 8. 写入设计

### 8.1 insert

`insert()` 使用单条 `MERGE INTO` 完成批量 upsert。

设计原因:

- 比“先查再更再插”更原子。
- 比逐条 DML 更高效。
- 避免某些数据库模式下 `ON CONFLICT` 路线的不确定性。

### 8.2 派生字段

每条记录在 provider 层自动派生:

- `memory = payload["data"] or payload["memory"]`
- `text_lemmatized = payload["text_lemmatized"] or memory`

设计原因:

- 让 BM25 和展示主文本都有稳定来源。
- 降低对上层传参完整性的依赖。

### 8.3 UTF-8 处理

provider 统一:

- 设置 client encoding 为 `UTF8`
- payload 使用 `json.dumps(..., ensure_ascii=False)`

设计原因:

- 避免 psycopg2 JSON adapter 在某些环境下的编码问题
- 保障中文和多语言 payload 在集中式真实库中稳定 round-trip

## 9. 查询设计

### 9.1 语义检索

`search()` 的 SQL 语义是:

- 在 DB 内完成距离计算
- 在 DB 内完成 filter
- 在 DB 内完成 top_k 与排序

排序规则:

```text
distance ASC, id ASC
```

返回分数:

```text
score = 1 / (1 + max(distance, 0))
```

设计原因:

- 对上层暴露单调可比较的正向分数
- 保持结果稳定性

注意:

- 这个 `score` 是 **GaussDB provider 自己归一化后的 semantic score**
- 它不是原始距离，也不是严格数学意义上的 cosine similarity
- 它的设计目标是适配 mem0 上层 `threshold` 与混合排序逻辑

因此:

- **不能把 GaussDB 的 `score` 与 pgvector 返回的 raw distance 直接横向比较**
- 也**不能把不同 provider 返回的 `score` 数值混在一起做统一排序**

原因是 mem0 当前各 provider 的 `score` 语义并不统一:

| Provider | score 语义 |
|---|---|
| GaussDB | provider-normalized semantic score，越大越好 |
| pgvector | raw distance，越小越好 |
| Azure MySQL | raw distance，越小越好 |
| MongoDB / OpenSearch / Elasticsearch / Qdrant | 后端原生 score，通常越大越好，但量纲不统一 |

所以当前正确使用方式是:

- 同一次检索只在单个 provider 内解释 `score`
- 不跨 provider 直接比较原始 `score`

### 9.2 关键词检索

`keyword_search()` 只有在 `bm25_enabled=True` 时才工作。  
否则返回 `None`。

设计原因:

- `None` 能明确表达“该 provider 当前不支持关键词检索”
- 上层 `Memory.search()` 会自行做兼容处理

对于当前 GaussDB provider，还需要明确一点:

- **集中式模式**: 默认尝试启用 BM25，因此可提供 `keyword_search()`
- **分布式模式**: 当前直接禁用 BM25，因此不提供 `keyword_search()` 能力

这不是异常降级，而是当前产品边界。  
分布式场景下，mem0 仍然使用 semantic search 主链路完成召回。

### 9.3 batch search

优先走原生 SQL:

- 用 `VALUES` 构造 query vector 集
- 用窗口函数 `ROW_NUMBER() OVER (PARTITION BY query_index ...)`

失败时退化为逐条 `search()`

设计原因:

- 原生 batch 在可用时更高效
- fallback 保证功能完整性

## 10. filter 设计

### 10.1 scope guard

默认开启 `require_scoped_filters = True`。

要求:

- `search`
- `keyword_search`
- `search_batch`
- `list`

都必须至少带一个正向 scope 条件。

正向 scope 的定义包括:

- `{"user_id": "u1"}`
- `{"user_id": {"eq": "u1"}}`
- `{"user_id": {"in": ["u1", "u2"]}}`
- 合法 `AND`
- 所有分支都收敛到正向 scope 的 `OR`

不允许:

- `{"user_id": {"ne": "u1"}}`
- `{"$or": [{"user_id": "u1"}, {"category": "public"}]}`
- 仅在 `NOT` 中出现 scope

设计原因:

- 防止用户用布尔逻辑绕过隔离约束。

### 10.2 当前支持的 operator

当前支持:

- 直接值等值
- `eq`
- `ne`
- `in`
- `nin`
- `contains`
- `icontains`
- `AND/$and`
- `OR/$or`
- `NOT/$not`

### 10.3 当前不支持的 operator

`gt/gte/lt/lte` 当前不支持 typed range。

当前行为:

- 记录 warning
- 按 dict 字面值做等值比较

为什么这么设计:

- 当前 SQL 走的是 `payload->>'key'`，得到的是文本
- 如果直接用字符串比较，`10 > 2` 会出现错误结果
- 错误结果比显式不支持更危险

### 10.4 为什么不直接删掉这些 operator

因为 mem0 上层 filter 语义已经公开存在。  
provider 需要有明确响应，而不是完全不认识该结构。

当前策略是:

- 明确 warning
- 保证不会误做错误的 range 语义

### 10.5 与其他 provider 对比

| Provider | range 支持情况 |
|---|---|
| Qdrant | 原生支持 typed range |
| GaussDB 当前 | 不支持 typed range |
| pgvector | 通常不做真正 typed payload range |
| Azure MySQL | 通常也是 JSON 文本抽取思路 |

结论:

- GaussDB 当前不是最强 filter provider
- 但当前设计比“假支持字符串 range”更正确

## 11. 管理接口设计

### 11.1 get / update / delete

这三个接口当前都是按 `id` 直接操作，不附带 scope filter。

设计原因:

- 对齐 `VectorStoreBase` 当前契约

代价:

- provider 自身无法独立承诺按 id 的强租户隔离

这需要在上层 API、业务鉴权或更改公共契约时解决。

### 11.2 list

`list()` 返回值保持与 mem0 其他 provider 一致，是:

```python
List[List[OutputData]]
```

而不是直接扁平 list。

设计原因:

- 与上层现有使用方式兼容

### 11.3 reset

`reset()` 的语义是:

```text
drop collection + schema_meta
recreate collection
```

设计原因:

- 用于测试和全量重置场景

### 11.4 analyze

`analyze()` 当前单独走 autocommit。

设计原因:

- 分布式环境已验证 `ANALYZE` 不能在 transaction block 中执行
- 集中式与分布式统一走 autocommit 更稳

## 12. retry 与观测设计

### 12.1 retry

当前只对明显瞬态错误重试:

- connection
- timeout
- deadlock
- lock wait
- serialization failure

设计原因:

- 对永久性 SQL 错误重试没有意义
- 只对短暂性数据库抖动兜底

### 12.2 metrics

当前 provider 内部会维护简单计数:

- fallback count
- retry count
- error count
- latency count

设计目的:

- 作为最小可观测性基础
- 为后续导出到日志/监控系统预留接口

## 13. 测试设计

### 13.1 unit tests

覆盖:

- 配置校验
- capability probe
- DDL SQL
- fallback
- filter 语义
- retry / rollback
- analyze autocommit

### 13.2 centralized live tests

覆盖:

- CRUD
- filter matrix
- scope guard
- UTF-8 / multilang
- BM25
- search_batch
- e2e memory flow

### 13.3 commercial validation suite

这是最接近交付门禁的一层:

- collection contract
- CRUD and batch
- filter matrix
- unsupported range behavior
- vector order
- UTF-8
- keyword search
- distributed smoke

## 14. 已知限制

1. 未支持 typed metadata range。
2. `get/update/delete` 无 scope guard。
3. 公共配置还没有暴露更细粒度 BM25 / metadata 策略开关。
4. 分布式场景已具备代码路径和样例，但商用结论仍依赖客户环境复验。

## 15. 设计结论

GaussDB 当前设计不是“能力最多”的 provider，但它已经形成了一条很清晰的商用设计路线:

- 用单 collection 实例模型保持接口干净
- 用 capability probe 吞掉环境差异
- 用 scope guard 强化商用默认安全性
- 用 savepoint 和 fallback 保护可选能力
- 用明确 warning 代替错误的伪兼容

从工程判断上看，这套设计是稳的，也适合继续向交付文档和生产门禁推进。
