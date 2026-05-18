# mem0 GaussDB 适配技术设计文档

> 更新时间: 2026-05-17  
> 设计对象: `mem0.vector_stores.gaussdb.GaussDB`、`mem0.configs.vector_stores.gaussdb.GaussDBConfig`  
> 对应代码:
>
> - `mem0/mem0/vector_stores/gaussdb.py`
> - `mem0/mem0/configs/vector_stores/gaussdb.py`

---

## 1. 设计目标

本设计文档回答的是“GaussDB provider 现在到底是如何工作的”。

它覆盖以下内容:

1. provider 在 mem0 中的角色边界
2. 配置、连接、能力探测、建表、索引、CRUD、检索、过滤、降级、观测等完整逻辑
3. 每一个关键适配点为什么这样设计
4. 与其他 provider 相比，GaussDB 当前的主要差异和取舍是什么
5. 哪些能力已经稳定，哪些是明确边界

本文档不按提交逐条记录，而按当前真实代码结构做整体归纳。

---

## 2. 设计原则

GaussDB provider 当前遵循以下原则:

1. **公共契约优先**  
   不改 mem0 现有 `VectorStoreBase` 契约，所有适配都在 provider 内完成。

2. **正确性优先于表面兼容**  
   对数据库不稳定支持或语义不明确的能力，宁可显式降级，也不返回看似兼容但实际错误的结果。

3. **集中式优先做完整体验，分布式优先做边界清晰**  
   集中式承载更完整的能力集合；分布式优先确保主链路成立和边界可验证。

4. **过滤语义与索引能力分离**  
   索引创建失败不应直接导致过滤语义消失，最多只影响性能。

5. **单实例绑定单 collection**  
   provider 实例一旦初始化，就绑定唯一 `collection_name`，不做隐式切表。

6. **商用默认值优先**  
   配置面尽量收敛，默认行为尽量安全、可解释、可交付。

---

## 3. 在 mem0 架构中的位置

### 3.1 整体架构

```text
+-------------------------------------------------------------------+
|                         mem0 Memory Layer                         |
|   Memory.add / search / update / delete / get_all / reset        |
+------------------------------+------------------------------------+
                               |
                               v
+-------------------------------------------------------------------+
|                      VectorStoreFactory Layer                     |
|       provider="gaussdb" -> GaussDBConfig -> GaussDB              |
+------------------------------+------------------------------------+
                               |
                               v
+-------------------------------------------------------------------+
|                       GaussDB Provider Layer                      |
| config | pool | capability probe | DDL | CRUD | search | filter  |
+------------------------------+------------------------------------+
                               |
                               v
+-------------------------------------------------------------------+
|                            GaussDB DB                             |
| FLOATVECTOR | GsDiskANN/GSIVFFlat | JSONB/TEXT | BM25 | Ustore    |
+-------------------------------------------------------------------+
```

### 3.2 provider 的职责边界

GaussDB provider 负责:

- 向量记录存取
- metadata 过滤
- 语义向量检索
- 可选关键词检索
- collection 生命周期管理
- 模式探测与能力降级
- 连接与执行可靠性

GaussDB provider 不负责:

- 记忆提取
- embedding 生成
- 上层鉴权
- 跨 provider score 对齐
- semantic/BM25/entity 的最终融合排序

这些职责仍在 `Memory` 层或业务上层。

---

## 4. mem0 主链路走读

### 4.1 初始化链路

```text
Memory.from_config(...)
  -> VectorStoreFactory.create("gaussdb", config)
  -> GaussDBConfig 校验
  -> GaussDB(...)
     -> 读取环境变量 / 参数
     -> 校验 deployment / dims / index type / pool size
     -> 创建连接池
     -> probe 数据库能力
     -> auto_create 时检测并创建 collection
```

### 4.2 写入链路

```text
Memory.add(...)
  -> embedding 已由上层生成
  -> vector_store.insert(vectors, payloads, ids)
  -> GaussDB MERGE INTO 原子 upsert
```

### 4.3 检索链路

```text
Memory.search(...)
  -> vector_store.search(...)
  -> 如 provider 支持 keyword_search，则额外调用 keyword_search(...)
  -> 上层 score_and_rank 进行融合排序
```

注意:

- semantic search 永远是主链路
- `keyword_search()` 是可选增强能力
- provider 返回的 `score` 会继续被 mem0 上层消费

### 4.4 更新与删除链路

```text
Memory.update(...) -> vector_store.update(id, ...)
Memory.delete(...) -> vector_store.delete(id)
Memory.get(...)    -> vector_store.get(id)
Memory.get_all()   -> vector_store.list(...)
```

当前 `get/update/delete` 均按 `id` 直操作，不带 scope filter，这一点与多数现有 provider 保持一致。

---

## 5. 模块划分

| 模块 | 位置 | 职责 |
|---|---|---|
| 配置模型 | `configs/vector_stores/gaussdb.py` | 参数校验、默认值、环境变量兼容 |
| provider 主体 | `vector_stores/gaussdb.py` | 运行时逻辑总入口 |
| 连接池管理 | `gaussdb.py` 内部 | 建立连接池、借还连接、事务/回滚 |
| 能力探测 | `_probe_capabilities()` | 判断 JSONB / BM25 / 向量索引等能力 |
| collection DDL | `create_col()` 等 | 建表、建索引、schema meta |
| CRUD 模块 | `insert/get/update/delete/list` | 数据读写与管理 |
| 搜索模块 | `search/keyword_search/search_batch` | 检索执行 |
| 过滤模块 | `_build_where_clause()` 等 | filter 解析与 scope guard |
| 可靠性模块 | `_run_with_retry()` 等 | 重试、延迟记录、慢查询告警 |

---

## 6. 配置系统设计

### 6.1 正式公开的配置项

当前 `GaussDBConfig` 暴露以下字段:

| 分类 | 字段 |
|---|---|
| 连接 | `database`, `host`, `port`, `user`, `password`, `connection_string`, `sslmode`, `sslrootcert` |
| collection | `collection_name`, `embedding_model_dims` |
| 连接池 | `minconn`, `maxconn` |
| 部署 | `deployment_mode` |
| 向量索引 | `vector_index_type`, `vector_metric` |
| 运行策略 | `auto_create`, `require_scoped_filters` |

### 6.2 基础项与高阶项

#### 基础项

建议接入时显式填写:

- `connection_string` 或 `host/port/database/user/password`
- `collection_name`
- `embedding_model_dims`
- `deployment_mode`

这些字段决定了连接目标、业务表、向量维度和部署模式。

#### 高阶项

可以先吃默认值:

- `sslmode`
- `sslrootcert`
- `minconn`
- `maxconn`
- `vector_index_type`
- `vector_metric`
- `auto_create`
- `require_scoped_filters`

### 6.3 为什么目前不继续开放更多参数

当前 provider 内部仍有很多行为参数，例如:

- `payload_storage_mode`
- `filter_storage_mode`
- `bm25_enabled`
- `retry_attempts`
- `vector_index_maintenance_work_mem`

这些参数暂不进一步开放，原因是:

1. 过多底层旋钮会明显增加用户理解成本。
2. 当前 capability probe 已承担自动选择和自动降级责任。
3. 大多数参数属于 provider 内部实现细节，不是接入方第一优先级。
4. 当前只有 `require_scoped_filters` 直接影响检索安全语义，因此被正式公开。

### 6.4 配置校验逻辑

当前配置层与运行时共同保证以下约束:

1. 如未提供 `connection_string`，则:
   - `user/password` 必须同时提供
   - `host/port` 必须同时提供
2. `deployment_mode` 仅允许:
   - `centralized`
   - `distributed`
3. `vector_index_type` 仅允许:
   - `gsdiskann`
   - `gsivfflat`
4. `vector_metric` 仅允许:
   - `cosine`
   - `l2`
5. 维度限制:
   - centralized: `<= 4096`
   - distributed: `<= 1024`
6. `embedding_model_dims > 1024` 时只允许 `gsdiskann`
7. 连接池大小:
   - `minconn >= 1`
   - `maxconn >= 1`
   - `maxconn >= minconn`

### 6.5 环境变量兼容

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

- 对齐企业部署常见的环境注入方式
- 允许连接信息与业务代码解耦

---

## 7. 连接与事务设计

### 7.1 连接池设计

当前使用 `psycopg2.pool.ThreadedConnectionPool`，默认:

- `minconn=1`
- `maxconn=5`

设计原因:

1. 与 mem0 现有 SQL 类 provider 的默认规模接近。
2. 对 SDK 默认值而言足够保守。
3. 支持多线程应用场景。

### 7.2 为什么选择 psycopg2 兼容方式

GaussDB 官方 Python 驱动示例以 psycopg2 兼容接口为基础，当前 provider 也沿用该方向。  
这意味着实现优先考虑:

- 官方驱动兼容性
- 事务控制的可预期性
- 与 PostgreSQL 风格 SQL/连接 API 的相似性

### 7.3 `_get_cursor()` 的职责

`_get_cursor(commit=False)` 负责:

- 从连接池取连接
- 设置客户端编码为 `UTF8`
- 提供游标
- 正常结束时根据 `commit` 控制提交
- 失败时回滚
- 关闭游标
- 归还连接

这里的 UTF8 强制非常关键。  
GaussDB 适配里显式设置 `client_encoding="UTF8"`，是为了避免某些非 UTF8 会话或环境下，payload / 文本字段行为不稳定。

当前实现还会在初始化时探测一次 `server_encoding`。如果检测到数据库本身不是 `UTF8`，provider 不会强制阻断连接，但会记录 warning，明确提示:

- GaussDB mem0 部署的推荐与验证前提是 UTF8 数据库
- `client_encoding=UTF8` 只能保证客户端会话按 UTF8 传输
- 非 UTF8 数据库下，文本、JSON、metadata filter 与关键词检索行为可能不稳定

因此从设计口径上，非 UTF8 数据库不作为正式商用支持目标。

### 7.4 为什么 `analyze()` 单独走 autocommit

在分布式场景中，`ANALYZE` 不能在 transaction block 内执行。  
因此当前实现没有复用 `_get_cursor(commit=True)`，而是:

1. 单独从连接池取连接
2. 暂时设置 `autocommit=True`
3. 执行 `ANALYZE`
4. 恢复原 autocommit 状态
5. 归还连接

这个设计同时适用于 centralized 和 distributed，避免再维护双路径。

---

## 8. 能力探测与降级设计

### 8.1 为什么需要 capability probe

GaussDB 在不同版本、不同模式、不同部署环境下，向量、JSONB、BM25、表达式索引等能力组合不一定完全一致。  
如果只根据配置或版本字符串假设能力存在，风险很高。

因此当前 provider 采用“真实建表 / 建索引 / 执行 SQL”的 probe 模式。

### 8.2 探测内容

`_probe_capabilities()` 主要探测:

- `FLOATVECTOR`
- 向量索引
- BM25
- JSONB
- expression index
- UUID

并将结果收敛到 `CapabilityReport`。

### 8.3 降级策略

#### JSONB 失败

若 JSONB 本身不可用，则:

- `payload_storage_mode` 从 `jsonb` 降级为 `text`
- `filter_storage_mode` 从 `json_expression` 降级为 `redundant_columns`
- `metadata_column_mode` 同步调整

这时普通 metadata filter 能力会明显收缩，只保证 scope 相关冗余列过滤。

#### BM25 失败

若 BM25 probe 失败，则:

- 仅关闭 `bm25_enabled`
- semantic search 继续可用

#### expression index 失败

若 JSONB 可用，但 expression index 建不起来，则:

- 保持 `filter_storage_mode="json_expression"`
- 普通 metadata filter 仍继续使用 `payload->>'key'`
- 仅记录 warning，表示缺少索引加速

这是当前设计的关键改动之一。  
它把“索引建不起来”与“不能按 metadata 过滤”这两件事彻底拆开了。

### 8.4 与其他 provider 的对齐逻辑

这一策略更接近:

- pgvector
- Azure MySQL
- MongoDB
- Qdrant

这些 provider 通常不会因为某个优化索引不可用，就直接把过滤语义缩成只剩 scope 字段。

---

## 9. 数据模型设计

### 9.1 主表模型

当前 collection 主表至少包含:

| 字段 | 含义 |
|---|---|
| `id` | 主键，当前为 UUID |
| `vector` | 向量列 |
| `payload` | metadata，默认 JSONB |
| `memory` | 原始记忆文本 |
| `text_lemmatized` | 关键词检索使用的衍生文本 |
| `created_at` | 创建时间 |
| `updated_at` | 更新时间 |
| `schema_version` | schema 版本 |

当 `filter_storage_mode="redundant_columns"` 时，还会显式落下:

- `user_id`
- `agent_id`
- `run_id`

### 9.2 schema meta 表

除主表外，还维护单独的 `<collection>_schema_meta` 表，用于记录:

- collection 名称
- schema 版本
- 更新时间
- 当前模式元信息

设计目的:

1. 为后续 schema 演进保留锚点
2. 让 provider 能在 collection 生命周期中做版本识别
3. 避免把模式状态散落在代码假设里

### 9.3 schema 默认 `public`，可作为高级配置覆盖

当前 provider 将 `schema` 作为高级配置项暴露，默认值为 `public`，并统一做显式 schema qualification。

这样做的原因:

- 避免不同 session search_path 带来的漂移
- 保证表名、索引名、meta 表名可预测
- 降低 DDL 兼容风险
- 在客户有命名空间隔离要求时，提供不破坏主链路的可配置出口

如果用户不配置 `schema`，行为与旧版保持一致；如果客户有 DBA 治理或命名空间隔离要求，可以显式指定其他安全 schema 名称。

### 9.4 Ustore 与分布式策略

当前主表采用:

- `WITH (storage_type=ustore)`

分布式模式下采用:

- `DISTRIBUTE BY HASH(id)`

这两者一起构成了当前 GaussDB provider 的基础物理模型。

---

## 10. collection 生命周期设计

### 10.1 单实例单 collection

当前设计已明确收敛为:

- 一个 provider 实例
- 绑定一个 `collection_name`
- 对应一张主表和一张 schema meta 表

`create_col()` 已不再接受 `name` 参数。  
这避免了此前“临时建了一张别的表，但实例状态仍指向旧 collection”的半支持问题。

### 10.2 `create_col()` 行为

`create_col(*, vector_size=None, distance=None)` 负责:

1. 创建主表
2. 创建 schema meta 表
3. 回写 schema meta
4. 创建向量索引
5. 如 BM25 可用则创建 BM25 索引
6. 创建 scope 相关过滤索引

### 10.3 索引创建的 savepoint 设计

向量索引是主能力的一部分，失败通常应被视为严重错误。  
但 BM25 和 scope 表达式索引属于增强或加速能力，因此当前设计对这类索引采用 savepoint 包装:

1. 创建 savepoint
2. 尝试创建索引
3. 失败则 rollback to savepoint
4. 打 warning
5. collection 创建继续完成

这样可以避免“可选索引失败导致 collection 整体创建失败”。

### 10.4 其他生命周期接口

provider 当前还实现:

- `list_cols()`
- `col_info()`
- `delete_col()`
- `reset()`

其中 `reset()` 会删除并重建当前 collection，适合测试与重置场景。

---

## 11. 向量索引与高维策略

### 11.1 支持的索引类型

当前公开支持:

- `gsdiskann`
- `gsivfflat`

### 11.2 为什么 >1024 维只允许 `gsdiskann`

这是当前配置和运行时共同 enforced 的约束。  
原因很直接:

- distributed 模式上限本身是 1024
- centralized 模式要支持更高维时，当前只保留 `gsdiskann` 这条能力路径

这不是随意限制，而是当前数据库能力和适配安全边界共同决定的。

### 11.3 `maintenance_work_mem` 设计

当前 `_set_vector_index_maintenance_work_mem()` 的策略是:

1. 默认目标值为 `128MB`
2. 若为高维 `gsdiskann` 且仍是默认值，则目标值提升为 `2GB`
3. 先 `SHOW maintenance_work_mem`
4. 仅当当前值小于目标值时，才 `SET LOCAL maintenance_work_mem`
5. 若数据库或 DBA 已配置更高值，则不覆盖

这样做的目的是:

- 保底抬升高维索引构建资源
- 又不压低客户已调优过的更大内存配置

---

## 12. 写入设计

### 12.1 为什么使用 `MERGE INTO`

当前 `insert()` 使用批量 `MERGE INTO` 做原子 upsert，而不是:

- 先 `UPDATE`
- 再 `INSERT`

原因:

1. 避免并发条件下出现更新/插入竞态
2. 保证同一 `id` 的幂等写入语义
3. 更适合商用批量写入路径

### 12.2 payload 与文本处理

当前写入时会统一处理:

- `payload`
- `memory`
- `text_lemmatized`

设计考虑:

- payload 以 JSON 语义承载 metadata
- `memory` 保留原始业务文本
- `text_lemmatized` 供关键词检索使用

### 12.3 默认值处理

当前只有在以下情况下才自动补默认值:

- `ids is None`
- `payloads is None`

如果传的是空列表，则不会被当成“没传”，而是走长度校验。  
这样避免了调用方误把空输入当成可静默接受的情况。

---

## 13. 检索设计

### 13.1 semantic search

当前 `search()` 的核心逻辑:

1. 根据 `vector_metric` 选择距离操作符
2. 拼接过滤条件
3. 执行 top-k SQL
4. 按距离升序返回
5. 直接返回数据库距离值作为 `score`

当前支持:

- `cosine`
- `l2`

### 13.2 为什么现在直接返回 raw distance

GaussDB 当前不再对距离做 provider 专属归一化，而是直接返回数据库计算出的距离值。

这样做的原因是:

1. 与 `pgvector`、`Azure MySQL` 等 SQL provider 的 score 风格更一致
2. 避免 GaussDB 成为当前 mem0 provider 中唯一一个自定义归一化 score 语义的特例
3. 不再让用户误以为 GaussDB 的 score 可以天然跨 provider 对齐

### 13.3 与其他 provider 的差异

其他 provider 中常见三种做法:

1. 直接返回 raw distance
2. 直接返回后端 `_score`
3. 返回后端 SDK 原生相似度

GaussDB 当前属于第一种:

- 直接返回 raw distance

因此文档必须明确:

> GaussDB 的 score 不能与其他 provider 的 score 直接横向比较，也不应把多个 provider 的 score 混在一起统一排序。

### 13.4 batch search

`search_batch()` 优先尝试 native SQL 批量查询。  
若失败，则:

- 打 warning
- 回退到顺序调用 `search()`

这样做的原因是:

1. 给支持 native batch 的场景更好的性能
2. 让单条 `search()` 逻辑继续作为兜底主链路

---

## 14. 关键词检索设计

### 14.1 当前定义

GaussDB provider 当前将 `keyword_search()` 明确定义为:

- 基于 BM25 的可选能力

它不是一个“任意文本匹配接口”，而是一条 BM25 检索路径。

### 14.2 centralized 与 distributed 的差异

#### centralized

- probe 成功则开启 BM25
- `keyword_search()` 可用

#### distributed

- 当前明确关闭 BM25
- `keyword_search()` 返回 `None`

这是正式能力边界，而不是异常情况。

### 14.3 为什么不强行补其他 fallback

理论上可以给 `keyword_search()` 增加:

- `LIKE/ILIKE`
- 其他全文检索
- 自定义文本匹配 SQL

但当前设计没有这样做，原因是:

1. 会显著增加能力分支和解释成本
2. 相关性不一定能与 BM25 对齐
3. 现有 mem0 生态本来也允许 provider 完全不支持 `keyword_search()`

因此当前策略是:

- 有 BM25 就支持关键词检索
- 没有 BM25 就显式返回 `None`

---

## 15. 过滤系统设计

### 15.1 当前正式支持的操作符

字段比较:

- 隐式等值
- `eq`
- `ne`
- `in`
- `nin`
- `contains`
- `icontains`

逻辑操作:

- `$and` / `AND`
- `$or` / `OR`
- `$not` / `NOT`

### 15.2 scope guard

当前 provider 默认 `require_scoped_filters=True`。  
这意味着以下读路径都要求 filters 中至少有一个正向 scope 条件:

- `search`
- `keyword_search`
- `search_batch`
- `list`

scope key 当前固定为:

- `user_id`
- `agent_id`
- `run_id`

### 15.3 为什么这样设计

大多数其他 provider 只是“支持这些字段过滤”，但不强制调用方带它们。  
GaussDB 当前更进一步，是因为它的目标不是只做通用向量库，而是希望在商用多租户场景下提供更安全的默认行为。

### 15.4 防绕过逻辑

当前 `_has_scope_filter()` 不会把以下情况认定为有效 scope:

- 只有 `ne`
- 只有 `nin`
- 只在 `NOT` 分支出现
- 通过宽松 `OR` 试图绕过约束

这样可避免“形式上出现了 scope 字段，实际上却没有真正收敛查询范围”的情况。

### 15.6 与其他 provider 的对比

- SQL/JSON 类 provider 往往也没有强类型 range schema
- Qdrant 这类原生 typed payload provider 可以支持 range
- 其他多数 provider 即使不支持，也未必显式报错

GaussDB 当前选择的是更可解释的路径:

- 明确边界
- 不误导用户

---

## 16. 读取、更新、删除设计

### 16.1 当前语义

`get/update/delete` 当前都按 `id` 直操作，不附带 scope filter。

这并不是 GaussDB 的特例，而是与当前多数 provider 一致，也与 `VectorStoreBase` 的公共签名一致。

### 16.2 为什么当前不内建 scope 版 `get/update/delete`

因为公共契约只提供:

- `id`

没有提供:

- `id + filters`
- `id + scope`

如果 provider 私自追加 scope 参数，会直接破坏现有工厂和上层调用方式。

### 16.3 这意味着什么

这意味着:

- 检索类接口可以由 provider 做默认作用域兜底
- 管理类 `id` 直达接口仍需依赖上层鉴权

因此要清楚区分:

1. provider 能做的默认安全防护
2. 业务 API 仍需承担的鉴权职责

---

## 17. 运维、可靠性与观测设计

### 17.1 重试机制

当前 `_run_with_retry()` 会对部分瞬态错误做有限重试。  
识别的错误片段包括:

- `connection`
- `timeout`
- `deadlock`
- `lock wait`
- `could not serialize`
- `serialization failure`
- `server closed`
- `terminating connection`

默认:

- `retry_attempts = 2`
- `retry_backoff_seconds = 0.1`

### 17.2 为什么不把重试暴露成更大配置面

目前它更偏 provider 内部容错策略，不是接入层一开始就必须调的参数。  
因此先保留内部默认值，避免配置面进一步膨胀。

### 17.3 观测与慢查询

当前 provider 内部维护:

- 简单 metrics 计数
- latency 记录
- 慢查询 warning

默认慢查询阈值为:

- `slow_query_ms = 1000`

这套机制的目标不是替代完整 APM，而是帮助定位:

- 哪条 SQL 在拖慢响应
- 哪类能力在频繁降级或重试

---

## 18. centralized 与 distributed 的差异设计

### 18.1 centralized

当前集中式设计目标是“完整体验优先”，因此:

- 维度上限更高
- 支持 BM25
- 支持更完整的向量索引路径
- 商用主体验证优先围绕 centralized 展开

### 18.2 distributed

当前分布式设计目标是“主链路成立 + 边界清晰”，因此:

- 维度上限收缩到 1024
- 关闭 BM25
- `keyword_search()` 不提供
- `analyze()` 使用 autocommit 以兼容分布式限制

### 18.3 为什么分布式不强补 BM25

因为当前没有稳定、可验证、可解释的实现基础。  
在这种情况下，明确“不支持”比做一个弱替代实现更稳。

---

## 19. 测试与验证设计

### 19.1 测试分层

当前测试体系分为四层:

| 测试文件 | 角色 |
|---|---|
| `test_gaussdb.py` | 轻量单测/契约测试 |
| `test_gaussdb_commercial_validation.py` | 商用出口门禁 |
| `test_gaussdb_centralized.py` | centralized live 深度回归 |
| `test_gaussdb_distributed.py` | distributed live 回归，已分 smoke/full |

### 19.2 为什么要拆 smoke/full

尤其是 distributed 场景，DDL 和索引创建成本很高。  
若所有 live 用例都作为默认门禁执行，成本过大、抖动也更明显。

因此当前设计是:

- 默认分布式跑 smoke/主链路
- 更重的 full regression 通过环境变量显式开启

### 19.3 测试命名与边界一致性

本轮测试重构中特别做了两件事:

1. live 默认索引路径改回 `gsdiskann`，对齐 provider 真默认值
2. `gt/gte/lt/lte` 用例按“声明字段支持 typed range、未声明字段 warning + 兼容匹配”的现口径重构，避免旧测试名继续暗示过时语义

---

## 20. 当前设计的主要差异化与 trade-off

### 20.1 相比其他 provider 更严格的地方

1. 默认 scope guard
2. 显式 UTF8 客户端编码
3. expression index 与 metadata filter 语义解耦
4. 范围过滤宁可不支持也不假支持

### 20.2 相比其他 provider 不完全一致的地方

1. centralized 支持 BM25，distributed 显式不支持
2. 默认行为比很多 provider 更保守

### 20.3 为什么这些差异是合理的

因为当前目标不是把 GaussDB 做成“看起来和别的 provider 一样”，而是:

- 在 mem0 公共契约内稳定工作
- 商用默认行为更安全
- 明确告诉用户什么支持、什么不支持

---

## 21. 当前已知边界

当前设计文档必须明确以下边界:

1. `gt/gte/lt/lte` 仅对声明为 `number/datetime` 的字段执行 typed range；未声明字段走 warning + 兼容匹配
2. distributed 不支持 BM25 / `keyword_search()`
3. `get/update/delete` 不带 scope guard
4. score 不可跨 provider 横向比较
5. scope guard 关闭后会更像多数其他 provider，但安全性下降

这些边界并不意味着 provider 不可用，而是意味着当前交付时必须把承诺说清楚。

---

## 22. 未来扩展点

如果后续继续增强，最合理的扩展点依次是:

1. typed metadata schema 与 typed range filter
2. 更细粒度的企业级 scope key 扩展
3. 更强的关键词检索策略选择
4. 更统一的 provider score 语义抽象
5. 更丰富的运行时观测和性能指标输出

这些都应建立在当前主链路稳定的基础上推进，而不是推倒重来。

---

## 23. 总结

当前 GaussDB provider 的设计已经形成一个比较完整的闭环:

- 配置层有明确边界
- 连接与事务层可控
- capability probe 可解释
- collection/索引模型稳定
- CRUD 与检索主链路齐全
- 作用域过滤具备商用安全默认值
- 集中式/分布式差异清晰
- 测试与文档已开始按商用出口组织

一句话概括当前设计取向:

> GaussDB provider 不是为了做“功能看起来很多”的适配，而是为了在 mem0 当前契约下，提供一套默认更安全、边界更清楚、出错更可解释的商用级实现。

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
score = distance
```

设计原因:

- 与现有 SQL provider 的 score 风格保持一致
- 避免引入仅 GaussDB 独有的 provider-normalized 分数语义

注意:

- 这个 `score` 是 **GaussDB 返回的原始距离值**
- 它不是严格数学意义上的 cosine similarity
- 不同 provider 的 `score` 语义仍然不统一

因此:

- **不能把 GaussDB 的 `score` 与 pgvector 返回的 raw distance 直接横向比较**
- 也**不能把不同 provider 返回的 `score` 数值混在一起做统一排序**

原因是 mem0 当前各 provider 的 `score` 语义并不统一:

| Provider | score 语义 |
|---|---|
| GaussDB | raw distance，越小越好 |
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

### 10.3 当前 range operator 语义

`gt/gte/lt/lte` 当前采用受控 typed range 策略：

- 已声明为 `number` / `datetime` 的字段：执行真正的 typed range
- 未声明字段或非 `number/datetime` 声明字段：记录 warning，并回落到兼容匹配
- 已声明字段中的脏历史数据行：忽略坏行，不让整次查询报错

### 10.4 为什么不直接删掉这些 operator

因为 mem0 上层 filter 语义已经公开存在。  
provider 需要有明确响应，而不是完全不认识该结构。

当前策略是:

- 声明字段真支持 typed range
- 未声明字段明确 warning，并保证不会误做错误的 range 语义

### 10.5 与其他 provider 对比

| Provider | range 支持情况 |
|---|---|
| Qdrant | 原生支持 typed range |
| GaussDB 当前 | 已声明 `number/datetime` 字段支持 typed range；未声明字段 warning + 兼容匹配 |
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

当前对它的定位应当是:

- **辅助维护接口**
- 不是 mem0 主链路必需能力
- 非必要场景下不一定要调用

更适合的使用时机:

- 建表后
- 大批量导入后
- 离峰维护窗口
- 测试或排障时手工触发

不建议把 `analyze()` 放进高频业务请求路径。

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

---

## 16. 2026-05-18 Typed Filter 重构现行口径

> 本节用于覆盖本文档中较早阶段的旧实现描述。若本文档其他章节与本节冲突，以本节为准。

### 16.1 scope 字段已从“冗余列”升级为正式列

当前 `user_id`、`agent_id`、`run_id` 不再只是 fallback 或降级模式下的冗余列，而是主表中的正式列。

这意味着：

- scope 过滤默认优先走实体列
- scope 索引走真实列索引
- `insert / update / upsert` 会同步写入 scope 列
- scope guard 仍然默认开启

### 16.2 typed exact 已替代 `str(value)` 文本比较

当前普通 metadata 的 `eq / ne / in / nin` 不再依赖 Python `str(value)` 再去比较 `payload->>'key'`。

现行实现改为基于 JSONB 语义的 typed exact：

- `eq` / `in` 走 `payload @> ...::JSONB`
- `ne` / `nin` 走 `(payload @> ...::JSONB) IS NOT TRUE`
- `bool`、`number`、`null`、`string` 都按 JSON 标量语义匹配

这一步的目的，是把布尔值、空值和普通标量从“文本偶然匹配”切换到“JSON 语义正确匹配”。

### 16.3 range 已改为“声明类型后支持”

旧口径里曾把 `gt/gte/lt/lte` 统一视为“不支持 typed range”。当前实现已经更新为：

- 未声明类型字段或非 `number/datetime` 声明字段：记录 warning，并回落到兼容匹配
- 已声明 `number` / `datetime` 字段：允许 typed range
- 已声明字段中的脏历史数据行：忽略坏行，不让整次查询报错

对应高级配置项是：

- `metadata_schema`

当前推荐的声明方式例如：

```python
metadata_schema = {
    "priority": "number",
    "score": "number",
    "created_at": "datetime",
}
```

因此，本文档中所有“当前完全不支持 range”的旧描述，应理解为：

> 当前对**未声明类型字段**的 range 不执行真正的 typed range，而是记录 warning 后回落到兼容匹配；对**声明类型字段**已支持 typed range，并会忽略坏历史数据行。

### 16.4 wildcard / exists / missing / null 语义

当前 typed-filter 第二阶段口径如下：

- 普通 metadata 字段上的 `*`：跳过该字段约束，不做字面量匹配
- scope 字段上的 `*`：不算有效正向 scope
- `{"field": None}`：表示 JSON `null`
- `{"field": {"exists": True}}`：字段存在
- `{"field": {"missing": True}}`：字段不存在

这意味着：

- `null` 与 `missing` 已经区分
- wildcard 不再被当作普通字符串 `'*'`
- `exists / missing` 已经进入 provider 级正式能力

### 16.5 跨路径对齐

当前以下读路径共享同一套 typed-filter 规则：

- `search`
- `list`
- `search_batch`
- `keyword_search`

也就是说，typed exact、declared range、wildcard、exists/missing 的主要行为不再只在单一路径成立，而是按 provider 级统一语义执行。

### 16.6 当前阶段验证状态

截至本轮重构，以下验证已通过：

- 本地单测：`tests/vector_stores/test_gaussdb.py`
- 商用门禁：`tests/vector_stores/test_gaussdb_commercial_validation.py`
- 集中式真库验证：新增 typed exact、declared range、wildcard、exists/missing 场景均已通过
