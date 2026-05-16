# mem0 GaussDB 使用指南与用户手册

> 更新时间: 2026-05-16  
> 面向对象: 应用研发、DBA、测试、交付、运维  
> 对应实现: `mem0.vector_stores.gaussdb.GaussDB`

## 1. 这份手册解决什么问题

这份手册面向实际使用者，重点回答:

- 如何把 mem0 接到 GaussDB。
- 集中式和分布式分别应该怎么用。
- 现在支持哪些能力，不支持哪些能力。
- 如何验证环境是否可用。
- 常见报错该怎么理解。

## 2. 快速结论

如果你只想先用起来，记住下面几件事:

1. 推荐优先使用 GaussDB 集中式模式。
2. 数据库建议使用 UTF-8。
3. `collection_name` 对应一张主表，不支持在同一个 provider 实例里动态切别的表。
4. 查询时必须带 `user_id`、`agent_id`、`run_id` 中至少一个正向 filter。
5. 当前不支持真正的 `gt/gte/lt/lte` range 过滤。
6. 集中式支持 BM25，分布式默认不支持。

## 3. 环境准备

### 3.1 Python 依赖

至少需要:

```bash
pip install mem0ai
```

如果你在本仓库开发或验证:

```bash
pip install -e ".[vector_stores,llms,nlp]"
```

同时需要可用的 GaussDB psycopg2 兼容驱动。  
当前实现依赖 `psycopg2` 风格 API。

### 3.2 GaussDB 基础要求

推荐:

- 集中式模式
- UTF-8 数据库
- 已启用向量能力
- 允许创建表、索引、BM25 索引

### 3.3 为什么建议 UTF-8

当前 provider 已经显式把 client encoding 设为 `UTF8`，并且 payload 序列化会保留原始 Unicode。

这能显著降低以下风险:

- 中文 payload 乱码
- 多语言 payload round-trip 异常
- psycopg2 JSON 适配器在某些编码下的兼容问题

所以商用建议非常明确:

> 优先建议客户使用 UTF-8 数据库。

## 4. 支持矩阵

### 4.1 集中式

| 能力 | 状态 |
|---|---|
| 向量 CRUD | 支持 |
| semantic search | 支持 |
| search_batch | 支持 |
| metadata filter | 支持 |
| `eq/ne/in/nin/contains/icontains` | 支持 |
| `AND/OR/NOT` | 支持 |
| typed range | 不支持 |
| BM25 keyword_search | 支持，可降级 |
| UTF-8 / 多语言 | 支持 |
| scope guard | 默认开启 |

### 4.2 分布式

| 能力 | 状态 |
|---|---|
| 向量 CRUD | 支持 |
| semantic search | 支持 |
| search_batch | 支持 |
| metadata filter | 支持 |
| BM25 keyword_search | 默认不支持 |
| 最大维度 | 1024 |
| analyze | 已走 autocommit |

## 5. 基本配置

### 5.1 最小可用示例

```python
from mem0 import Memory

config = {
    "vector_store": {
        "provider": "gaussdb",
        "config": {
            "host": "your-host",
            "port": 19995,
            "database": "your_db",
            "user": "your_user",
            "password": "your_password",
            "collection_name": "mem0_memories",
            "embedding_model_dims": 1536,
            "deployment_mode": "centralized",
            "vector_index_type": "gsdiskann",
            "vector_metric": "cosine",
        },
    },
    "embedder": {
        "provider": "openai",
        "config": {
            "model": "text-embedding-3-small",
        },
    },
    "llm": {
        "provider": "openai",
        "config": {
            "model": "gpt-4o-mini",
        },
    },
}

m = Memory.from_config(config)
```

### 5.2 环境变量方式

```bash
set GAUSSDB_HOST=your-host
set GAUSSDB_PORT=19995
set GAUSSDB_DATABASE=your_db
set GAUSSDB_USER=your_user
set GAUSSDB_PASSWORD=your_password
```

然后代码里可以只写:

```python
config = {
    "vector_store": {
        "provider": "gaussdb",
        "config": {
            "collection_name": "mem0_memories",
            "embedding_model_dims": 1536,
            "deployment_mode": "centralized",
        },
    },
    ...
}
```

### 5.3 当前真实可用配置字段

```python
{
    "database": "postgres",
    "collection_name": "mem0",
    "embedding_model_dims": 1536,
    "user": "...",
    "password": "...",
    "host": "...",
    "port": 19995,
    "connection_string": None,
    "sslmode": None,
    "sslrootcert": None,
    "minconn": 1,
    "maxconn": 5,
    "deployment_mode": "centralized",
    "vector_index_type": "gsdiskann",
    "vector_metric": "cosine",
    "auto_create": True,
    "require_scoped_filters": True,
}
```

注意:

- 当前配置模型不接受多余字段。
- 文档和脚本里如果写了不存在的高阶字段，初始化会报错。
- `require_scoped_filters` 是高级安全配置。默认 `True`，建议在多租户或生产环境保持开启；只有在单租户调试、离线排查等场景下，才考虑显式关闭。

### 5.4 基础项与高阶项

建议将对外配置理解为两层:

- 基础项: 建议显式填写，决定连接对象、collection、向量维度和部署模式
- 高阶项: 可以不填，provider 会使用默认值

推荐作为基础项显式配置:

- `connection_string`，或 `host` + `port` + `database` + `user` + `password`
- `collection_name`
- `embedding_model_dims`
- `deployment_mode`

推荐作为高阶项按需覆盖:

- `sslmode`
- `sslrootcert`
- `minconn`
- `maxconn`
- `vector_index_type`
- `vector_metric`
- `auto_create`
- `require_scoped_filters`

最小可用示例:

```python
{
    "connection_string": "...",
    "collection_name": "mem0_prod",
    "embedding_model_dims": 1536,
    "deployment_mode": "centralized",
}
```

商用推荐示例:

```python
{
    "connection_string": "...",
    "collection_name": "mem0_prod",
    "embedding_model_dims": 1536,
    "deployment_mode": "centralized",
    "sslmode": "require",
    "minconn": 1,
    "maxconn": 10,
    "vector_index_type": "gsdiskann",
    "vector_metric": "cosine",
    "auto_create": True,
    "require_scoped_filters": True,
}
```

## 6. 连接与表行为

### 6.1 auto_create

默认 `auto_create=True`。  
这意味着初始化 `GaussDB(...)` 时，如果 collection 不存在，会自动创建:

- 主表
- schema meta 表
- 向量索引
- 可选 BM25 索引
- scope filter 索引

### 6.2 collection_name 是什么

`collection_name` 对应的就是主表名，不是 schema 名。

例如:

```python
GaussDB(collection_name="mem0_user_memory")
```

对应主表大致是:

```sql
"public"."mem0_user_memory"
```

以及:

```sql
"public"."mem0_user_memory_schema_meta"
```

### 6.3 create_col 的语义

当前 `create_col()` 只支持创建当前实例绑定的 collection。

支持:

```python
db.create_col()
```

不支持:

```python
db.create_col(name="other")
```

这是有意设计，不是缺陷。

## 7. 常用操作示例

### 7.1 添加记忆

```python
m.add("用户喜欢早上喝拿铁", user_id="u1")
```

### 7.2 带 metadata 添加

```python
m.add(
    "用户偏好靠窗座位",
    user_id="u1",
    metadata={
        "category": "travel",
        "status": "active",
        "language": "zh-CN",
    },
)
```

### 7.3 搜索记忆

```python
results = m.search(
    "早上的饮品习惯",
    filters={"user_id": "u1"},
    top_k=5,
)
```

### 7.4 带 metadata filter 搜索

```python
results = m.search(
    "座位偏好",
    filters={
        "user_id": "u1",
        "category": {"eq": "travel"},
        "status": {"in": ["active", "draft"]},
    },
    top_k=10,
)
```

### 7.5 更新记忆

```python
m.update(memory_id, "用户现在更喜欢美式咖啡")
```

### 7.6 删除记忆

```python
m.delete(memory_id)
```

### 7.7 获取全部记忆

```python
all_memories = m.get_all(filters={"user_id": "u1"})
```

## 8. filter 语义说明

### 8.1 必须带 scope

以下接口默认都要求至少带一个正向 scope:

- `search`
- `keyword_search`
- `search_batch`
- `list`

例如:

```python
filters={"user_id": "u1"}
```

如果只写:

```python
filters={"category": "travel"}
```

会报错。

### 8.2 当前支持的 filter

| 语义 | 示例 |
|---|---|
| 直接等值 | `{"category": "travel"}` |
| `eq` | `{"category": {"eq": "travel"}}` |
| `ne` | `{"category": {"ne": "travel"}}` |
| `in` | `{"category": {"in": ["travel", "food"]}}` |
| `nin` | `{"category": {"nin": ["travel"]}}` |
| `contains` | `{"tag": {"contains": "coffee"}}` |
| `icontains` | `{"tag": {"icontains": "plan"}}` |
| `AND` / `$and` | `{"$and": [{"user_id": "u1"}, {"category": "travel"}]}` |
| `OR` / `$or` | `{"$or": [{"user_id": "u1"}, {"user_id": "u2"}]}` |
| `NOT` / `$not` | `{"$not": [{"category": "travel"}]}` |

### 8.3 当前不支持的 range

下面这些当前不是 typed range:

- `gt`
- `gte`
- `lt`
- `lte`

例如:

```python
{"priority": {"gt": 2}}
```

当前行为是:

- 记录 warning
- 不会按数值 range 过滤

因此在商用代码里，当前不要依赖这四个 operator 做正确业务筛选。

### 8.4 为什么不支持

因为当前实现走的是:

```sql
payload->>'priority'
```

这会把 JSONB 中的数值也取成文本。  
如果直接用 SQL 字符串比较，会产生错误结果。

## 9. 关于 score 的理解

### 9.1 GaussDB 返回的 score 是什么

GaussDB 当前返回的 semantic `score` 不是原始距离，而是 provider 内部归一化后的正向分数。

它的目标是:

- 更适配 mem0 上层 `threshold`
- 更适配 semantic + BM25 + entity boost 的融合排序
- 让分数语义更接近“越大越相关”

### 9.2 能不能和其他数据库直接比 score

**不能。**

原因是 mem0 当前不同 provider 的 `score` 语义并不统一:

- GaussDB: 归一化后的正向分数，越大越好
- pgvector: raw distance，越小越好
- Azure MySQL: raw distance，越小越好
- MongoDB / OpenSearch / Elasticsearch / Qdrant: 后端原生 score，通常越大越好，但范围不统一

所以不要做下面这种事情:

- 把 GaussDB 和 pgvector 的 `score` 放在一张表里直接排序
- 把多个 provider 的结果混在一起后直接按 `score` 大小统一排名

正确理解是:

- 同一次查询里，只在当前 provider 内解释它自己的 `score`
- `score` 主要用于当前 provider 的结果排序和 mem0 内部阈值过滤

### 9.3 那这样会不会影响正常使用

正常使用 mem0 时，一次检索只会走一个 provider，所以通常不会有问题。

真正需要注意的是:

- 做跨 provider benchmark 时，不要直接横比原始 `score`
- 做跨 provider 结果聚合时，要先统一打分语义，不能直接拿现成 `score` 混排

## 10. BM25 行为说明

### 10.1 集中式

集中式默认尝试启用 BM25。  
如果建索引或 score probe 失败，会自动关闭 `keyword_search`，但不会影响 semantic search。

### 10.2 分布式

分布式默认 `bm25_enabled=False`。

这也意味着:

- 分布式当前不支持 BM25
- 分布式当前不提供 `keyword_search()`
- 上层 mem0 会继续只使用 semantic search

### 10.3 使用建议

如果你依赖关键词召回:

- 优先用集中式
- 上线前在目标环境跑一遍 live validation

## 11. UTF-8 与多语言

当前 provider 明确支持:

- 中文
- 英文
- 中英混合
- 多语言 payload round-trip

推荐实践:

- 数据库使用 UTF-8
- 应用侧文本统一使用 Unicode 字符串
- 不要手工对 payload 做二次 JSON 转义

## 12. 连接池与并发建议

当前默认:

- `minconn=1`
- `maxconn=5`

建议:

- 本地开发和小规模服务: 保持默认
- 单进程中等并发: 可调到 `10`
- 更高并发: 结合 worker 数与数据库 `max_connections` 统一评估

注意:

- 连接池大小不是越大越好
- 如果你有多个 worker，每个 worker 都会各自建池

## 13. 集中式与分布式选型建议

### 13.1 什么时候选集中式

适合:

- 需要完整功能
- 需要 BM25
- 需要最稳妥的商用落地

### 13.2 什么时候选分布式

适合:

- 更关注容量与分布式部署
- 能接受当前 BM25 不支持
- 愿意配合做额外实库验证

### 13.3 推荐结论

如果你是第一次上线 mem0 + GaussDB:

> 优先集中式。

## 14. 验证与测试

### 14.1 单元测试

```bash
pytest mem0/tests/vector_stores/test_gaussdb.py -q
```

### 14.2 商用门禁

```bash
pytest mem0/tests/vector_stores/test_gaussdb_commercial_validation.py -q
```

### 14.3 集中式 live tests

先设置环境变量:

```bash
set GAUSSDB_TEST_HOST=...
set GAUSSDB_TEST_PORT=...
set GAUSSDB_TEST_DATABASE=...
set GAUSSDB_TEST_USER=...
set GAUSSDB_TEST_PASSWORD=...
```

再跑:

```bash
pytest mem0/tests/vector_stores/test_gaussdb_centralized.py -q
```

### 14.4 分布式 live tests

```bash
set GAUSSDB_TEST_DISTRIBUTED=true
pytest mem0/tests/vector_stores/test_gaussdb_commercial_validation.py -q
```

## 15. 常见问题

### 15.1 为什么查询必须带 `user_id` / `agent_id` / `run_id`

因为 provider 默认开启 scope guard，防止全表无约束搜索带来租户泄漏风险。

### 15.2 为什么 `gt/gte/lt/lte` 没生效

因为当前没有实现 typed range。  
这是已知边界，不是偶发 bug。

### 15.3 为什么 `keyword_search()` 返回 `None`

通常说明:

- 当前是分布式模式
- 或者集中式环境里 BM25 probe / BM25 index 创建失败，已自动关闭

### 15.4 为什么 `analyze` 以前报 transaction block

这是因为某些 GaussDB 模式下 `ANALYZE` 不能在事务块里执行。  
当前实现已经改成 autocommit 路径。

### 15.5 为什么推荐 UTF-8

因为这能显著减少中文和多语言 payload 的编码风险，也是当前商用最稳妥配置。

## 16. 故障排查建议

按这个顺序排查最省事:

1. 先看初始化是否成功。
2. 看 capability probe 是否有 warning。
3. 看 `col_info()` 输出。
4. 用最简单的 `user_id` scope 搜索先跑通主链路。
5. 再逐步加 metadata filter、BM25、batch search。

建议重点关注:

- 向量能力是否启用
- JSONB 是否可用
- BM25 是否被自动关闭
- 是否误用了 range operator
- 数据库编码是否为 UTF-8

## 17. 上线建议

上线前至少做这几件事:

1. 跑 `test_gaussdb_commercial_validation.py`
2. 跑一次真实集中式环境的 live tests
3. 验证中文与多语言样例
4. 验证 scope 隔离
5. 验证你的业务是否依赖 range filter；如果依赖，当前不要直接上线

## 18. 当前已知限制汇总

- 当前不支持 typed range
- `get/update/delete` 不带 scope guard
- 分布式默认不支持 BM25
- `create_col` 不支持动态创建其他 collection

## 19. 最后建议

如果你的目标是“让 mem0 在企业里稳定可用”，当前最推荐的实践是:

- 选集中式
- 用 UTF-8 数据库
- 所有搜索都显式带 scope
- 不使用 range operator 做业务判断
- 用 commercial validation suite 作为交付门禁

这套用法和当前实现是最贴合、也最稳的。
