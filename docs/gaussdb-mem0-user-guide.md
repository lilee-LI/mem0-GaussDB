# mem0 + GaussDB 用户手册

## 1. 概述

### 1.1 mem0 是什么

mem0 是面向 AI Agent/助手的长期记忆层。它不是普通聊天历史表，而是把用户对话抽取成可检索的”记忆”，并在后续 query 时做多信号召回。

一次典型 `Memory.add()` 的处理链路：

1. 接收用户/助手 messages。
2. 用 LLM 从对话里抽取结构化 memory。
3. 用 embedding 模型把 memory 文本转成向量。
4. 写入 vector store（GaussDB）。
5. 同时写入本地 SQLite history，用于 add 阶段上下文。

一次典型 `Memory.search()` 的处理链路：

1. 对 query 做 embedding。
2. 在 GaussDB 中做向量召回。
3. 如果 GaussDB BM25 可用，同时做关键词召回。
4. 做 entity boost、BM25 score、semantic score 融合排序。
5. 返回和当前 `user_id`、`agent_id`、`run_id` scope 匹配的记忆。

### 1.2 GaussDB provider 的定位

GaussDB provider 是 mem0 的企业级向量存储后端，专为华为 GaussDB 数据库设计。相比社区版 pgvector provider，它提供：

- 原生 `FLOATVECTOR` 类型支持，无需安装扩展
- 服务端 BM25 关键词索引，无需客户端分词
- 强制多租户 scope 隔离，防止跨租户数据泄露
- 运行时能力探测与自动降级，适应不同 GaussDB 版本和部署模式
- 集中式（A 模式）和分布式双模式支持

### 1.3 核心能力一览

| 能力 | 说明 |
|------|------|
| 语义搜索 | FLOATVECTOR 向量检索，支持 cosine/l2 距离 |
| BM25 关键词搜索 | GaussDB 原生 BM25 索引，服务端计算 |
| 混合排序 | semantic score + BM25 score + entity boost 融合 |
| 多租户隔离 | user_id / agent_id / run_id 三级 scope 强制过滤 |
| 能力探测降级 | 初始化时自动探测，不可用能力自动降级 |
| 批量操作 | CTE 单次往返 upsert，window function 批量搜索 |
| 可观测性 | 慢查询日志、操作延迟指标、fallback 计数 |

## 2. 环境准备

### 2.1 GaussDB 服务端要求

- **版本**：GaussDB 506 及以上（集中式 A 模式）
- **向量功能**：需要开启 `enable_vectordb`

```bash
# 开启向量功能（需要 DBA 权限）
gs_guc reload -D <datadir> -c "enable_vectordb=on"
# 重启数据库实例使配置生效
```

- **字符集**：推荐 UTF-8 编码数据库，避免中文乱码

```sql
-- 创建 UTF-8 数据库
CREATE DATABASE mem0_db ENCODING 'UTF8' LC_COLLATE 'en_US.UTF-8' LC_CTYPE 'en_US.UTF-8';
```

- **集中式 vs 分布式**：

| 特性 | 集中式（A 模式） | 分布式 |
|------|----------------|--------|
| 最大向量维度 | 4096（gsdiskann） | 1024 |
| BM25 搜索 | 支持 | 不支持 |
| 表存储引擎 | Ustore（原地更新） | 分布式 hash |
| 推荐场景 | 生产环境，功能完整 | 大规模水平扩展 |

### 2.2 Python 客户端安装

```bash
# 安装 mem0 及 GaussDB 相关依赖
pip install mem0ai

# 或从源码安装（开发模式）
cd mem0
pip install -e ".[vector_stores,llms,nlp]"

# GaussDB psycopg2 驱动（使用官方 wheel，不要用社区版 psycopg2）
# 从华为 GaussDB 官方渠道获取对应版本的 wheel 文件
pip install gaussdb_psycopg2-<version>-<platform>.whl

# spaCy 英文模型（用于 text_lemmatized 分词）
python -m spacy download en_core_web_sm

# fastembed（可选，用于本地 embedding，无需外部 API）
pip install fastembed
```

### 2.3 环境变量配置

推荐通过环境变量传递敏感信息，不要硬编码在代码中。

**GaussDB 连接参数：**

```bash
export GAUSSDB_HOST="your-gaussdb-host"
export GAUSSDB_PORT="5432"
export GAUSSDB_DATABASE="mem0_db"
export GAUSSDB_USER="your_user"
export GAUSSDB_PASSWORD="your_password"

# 或使用 DSN 格式连接字符串（优先级高于独立参数）
export GAUSSDB_CONNECTION_STRING="host=your-host port=5432 dbname=mem0_db user=your_user password=your_password"

# SSL 配置（生产环境推荐）
export GAUSSDB_SSLMODE="verify-full"
export GAUSSDB_SSLROOTCERT="/path/to/ca.crt"
```

**LLM 配置：**

```bash
# OpenAI
export OPENAI_API_KEY="sk-..."

# MiniMax
export MINIMAX_API_KEY="your-minimax-key"
export MINIMAX_BASE_URL="https://api.minimax.io/v1"
export MINIMAX_MODEL="MiniMax-M2.7"
```

**Embedding 配置：**

```bash
# OpenAI embedding
export OPENAI_API_KEY="sk-..."

# 自定义 OpenAI-compatible embedding 服务
export EMBEDDING_API_KEY="your-key"
export EMBEDDING_BASE_URL="https://your-embedding-service/v1"
export EMBEDDING_MODEL="text-embedding-3-small"
export EMBEDDING_DIMS="1536"
```

## 3. 快速开始

### 3.1 最小配置示例

```python
from mem0 import Memory

config = {
    "vector_store": {
        "provider": "gaussdb",
        "config": {
            "host": "your-gaussdb-host",
            "port": 5432,
            "database": "mem0_db",
            "user": "your_user",
            "password": "your_password",
            "collection_name": "memories",
            "embedding_model_dims": 1536,
        }
    },
    "llm": {
        "provider": "openai",
        "config": {"model": "gpt-4o-mini"}
    },
    "embedder": {
        "provider": "openai",
        "config": {"model": "text-embedding-3-small"}
    }
}

m = Memory.from_config(config)
```

### 3.2 添加记忆

```python
# 添加单条消息
m.add("我喜欢喝咖啡，每天早上必须来一杯", user_id="alice")

# 添加对话历史
m.add(
    [
        {"role": "user", "content": "下周三有个重要会议，需要准备季度报告"},
        {"role": "assistant", "content": "好的，我会帮你记住这个安排"},
    ],
    user_id="alice",
    agent_id="assistant_bot"
)
```

### 3.3 搜索记忆

```python
# 语义搜索
results = m.search("早上的习惯", user_id="alice")
for r in results:
    print(r["memory"], r["score"])

# 带 agent 范围的搜索
results = m.search(
    "会议安排",
    user_id="alice",
    agent_id="assistant_bot"
)
```

### 3.4 更新和删除

```python
# 获取所有记忆
all_memories = m.get_all(user_id="alice")
memory_id = all_memories[0]["id"]

# 更新记忆
m.update(memory_id, "我现在改喝茶了")

# 删除单条记忆
m.delete(memory_id)

# 删除用户所有记忆
m.delete_all(user_id="alice")
```

## 4. 配置详解

### 4.1 完整配置参数表

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `host` | str | `"localhost"` | GaussDB 服务器地址 |
| `port` | int | `5432` | GaussDB 端口 |
| `database` | str | `"postgres"` | 数据库名 |
| `user` | str | `"postgres"` | 用户名 |
| `password` | str | `""` | 密码 |
| `connection_string` | str | `None` | DSN 格式连接字符串，优先级高于独立参数 |
| `collection_name` | str | `"mem0"` | 向量表名（collection） |
| `embedding_model_dims` | int | `1536` | embedding 向量维度，必须与 embedding 模型一致 |
| `index_type` | str | `"gsdiskann"` | 向量索引类型：`gsdiskann` 或 `gsivfflat` |
| `distance_metric` | str | `"cosine"` | 距离度量：`cosine` 或 `l2` |
| `deployment_mode` | str | `"centralized"` | 部署模式：`centralized` 或 `distributed` |
| `distribution_mode` | str | `"auto"` | 分布式建表策略：`auto`、`hash`、`none` |
| `profile` | str | `"commercial"` | 能力 profile：`commercial`（默认）或 `community` |
| `metadata_mode` | str | `"auto"` | payload 存储模式：`auto`、`jsonb`、`text` |
| `bm25_mode` | str | `"auto"` | BM25 模式：`auto`（自动探测）、`required`（强制）、`disabled`（禁用） |
| `require_scoped_filters` | bool | `True` | 是否强制要求 scope 过滤（user_id/agent_id/run_id） |
| `sslmode` | str | `None` | SSL 模式：`disable`、`require`、`verify-ca`、`verify-full` |
| `sslrootcert` | str | `None` | SSL CA 证书路径 |
| `minconn` | int | `1` | 连接池最小连接数 |
| `maxconn` | int | `10` | 连接池最大连接数 |
| `slow_query_threshold_ms` | int | `1000` | 慢查询日志阈值（毫秒） |

### 4.2 连接方式

**方式一：独立参数**

```python
config = {
    "vector_store": {
        "provider": "gaussdb",
        "config": {
            "host": "192.168.1.100",
            "port": 5432,
            "database": "mem0_db",
            "user": "mem0_user",
            "password": "your_password",
        }
    }
}
```

**方式二：连接字符串（DSN 格式）**

```python
config = {
    "vector_store": {
        "provider": "gaussdb",
        "config": {
            "connection_string": "host=192.168.1.100 port=5432 dbname=mem0_db user=mem0_user password=your_password",
        }
    }
}
```

**SSL 配置：**

```python
config = {
    "vector_store": {
        "provider": "gaussdb",
        "config": {
            "host": "your-host",
            "sslmode": "verify-full",
            "sslrootcert": "/etc/ssl/certs/gaussdb-ca.crt",
        }
    }
}
```

### 4.3 向量索引选择

**gsdiskann（默认，推荐生产环境）：**

- 基于 DiskANN 算法，支持高维向量（最高 4096 维）
- 召回率高，适合大规模数据集
- 建索引时间较长，但查询性能优秀
- 仅集中式模式支持

**gsivfflat（适合开发测试）：**

- 基于 IVFFlat 算法，建索引速度快
- 维度上限较低（通常 2000 维以内）
- 适合快速验证和小规模数据

```python
# 使用 gsivfflat（开发测试）
config["vector_store"]["config"]["index_type"] = "gsivfflat"

# 使用 gsdiskann（生产环境，默认）
config["vector_store"]["config"]["index_type"] = "gsdiskann"
```

### 4.4 距离度量

```python
# cosine（默认）：适合文本 embedding，归一化向量
config["vector_store"]["config"]["distance_metric"] = "cosine"

# l2：欧氏距离，适合图像特征等欧氏空间场景
config["vector_store"]["config"]["distance_metric"] = "l2"
```

### 4.5 部署模式

```python
# 集中式（默认）：功能完整，支持 BM25，最高 4096 维
config["vector_store"]["config"]["deployment_mode"] = "centralized"

# 分布式：水平扩展，维度上限 1024，BM25 不可用
config["vector_store"]["config"]["deployment_mode"] = "distributed"
config["vector_store"]["config"]["distribution_mode"] = "auto"
```

## 5. 多租户隔离

### 5.1 Scope 机制

GaussDB provider 通过 `user_id`、`agent_id`、`run_id` 三级 scope 实现多租户隔离。默认配置下（`require_scoped_filters=True`），所有读取操作（search、list、keyword_search）必须携带至少一个有效 scope，否则抛出异常。

这个设计防止了跨租户数据泄露：一个用户的记忆不会出现在另一个用户的搜索结果中。

### 5.2 使用示例

```python
# 添加时指定 scope
m.add(“我喜欢喝咖啡”, user_id=”alice”, agent_id=”support_bot”)
m.add(“我喜欢喝茶”, user_id=”bob”, agent_id=”support_bot”)

# 搜索时必须指定 scope（只返回 alice 的记忆）
results = m.search(“饮品偏好”, user_id=”alice”)

# 跨 agent 搜索（alice 在 support_bot 下的记忆）
results = m.search(“饮品偏好”, user_id=”alice”, agent_id=”support_bot”)

# 按 run_id 隔离（同一用户不同会话）
m.add(“今天讨论了项目进度”, user_id=”alice”, run_id=”session_001”)
results = m.search(“项目”, user_id=”alice”, run_id=”session_001”)
```

### 5.3 安全说明

**为什么强制 scope：**

- 防止跨租户数据泄露，符合企业安全合规要求
- 避免误操作导致全表扫描，保护性能
- 明确数据归属，便于审计和数据治理

**如何关闭（不推荐）：**

```python
# 仅在开发测试或单租户场景下使用
config[“vector_store”][“config”][“require_scoped_filters”] = False
```

关闭后，search/list 操作不再强制要求 scope，可能返回所有租户的数据。生产环境不建议关闭。

## 6. BM25 关键词搜索

### 6.1 工作原理

GaussDB provider 利用 GaussDB 原生 BM25 索引实现关键词搜索：

1. 写入记忆时，`text_lemmatized` 列存储经过词形还原（lemmatization）处理的文本
2. BM25 索引建立在 `text_lemmatized` 列上
3. 搜索时使用 `###` 运算符计算 BM25 相关性分数

```sql
-- GaussDB BM25 搜索示例（内部实现）
SELECT id, text_lemmatized ### 'coffee morning' AS bm25_score
FROM memories
WHERE user_id = 'alice'
ORDER BY bm25_score DESC
LIMIT 10;
```

### 6.2 mem0 融合排序

mem0 上层会自动融合多路召回结果：

- **semantic score**：向量余弦相似度（0~1）
- **BM25 score**：关键词相关性分数（归一化后）
- **entity boost**：实体匹配加权

融合是自动的，无需手动配置。最终排序综合考虑语义相似度和关键词匹配度，对于精确关键词查询效果更好。

### 6.3 BM25 不可用时的行为

以下情况会自动禁用 BM25：

- 分布式模式（`deployment_mode="distributed"`）
- GaussDB 版本不支持 BM25 索引
- 能力探测阶段 BM25 测试失败

禁用后：

- `keyword_search()` 返回 `None`
- mem0 上层自动降级为纯语义搜索
- 不影响 `search()` 的正常使用，只是少了关键词召回路径

查看 BM25 状态：

```python
from mem0.vector_stores.gaussdb import GaussDB

db = GaussDB(host="...", port=5432, database="...", user="...", password="...",
             collection_name="memories", embedding_model_dims=1536)
print(f"BM25 enabled: {db.bm25_enabled}")
```

## 7. 能力探测与降级

### 7.1 探测流程

GaussDB provider 在初始化时自动执行能力探测：

1. 创建临时探测表（`_probe_<uuid>`）
2. 测试 JSONB 类型是否可用
3. 测试表达式索引是否可用
4. 测试 BM25 索引是否可用
5. 测试完成后自动清理临时表

探测结果缓存在实例中，不会重复执行。

### 7.2 降级路径

| 能力 | 正常模式 | 降级模式 | 触发条件 |
|------|----------|----------|----------|
| Payload 存储 | JSONB | TEXT + 冗余列 | JSONB 类型不可用 |
| Filter 索引 | 表达式索引 | 冗余 scope 列 | 表达式索引不可用 |
| BM25 | 启用 | 禁用 | BM25 索引不可用 |
| 向量索引 | gsdiskann | gsivfflat | gsdiskann 不可用 |

### 7.3 查看当前能力

```python
from mem0.vector_stores.gaussdb import GaussDB

db = GaussDB(
    host=”your-host”,
    port=5432,
    database=”mem0_db”,
    user=”your_user”,
    password=”your_password”,
    collection_name=”memories”,
    embedding_model_dims=1536,
)

# 查看所有能力
print(db.capabilities)

# 查看 payload 存储模式（jsonb 或 text）
print(db.payload_storage_mode)

# 查看 filter 存储模式（expression_index 或 redundant_columns）
print(db.filter_storage_mode)

# 查看 BM25 是否启用
print(db.bm25_enabled)

# 查看 collection 详细信息
info = db.col_info()
print(info)
```

## 8. 可观测性

### 8.1 慢查询日志

GaussDB provider 内置慢查询日志，超过阈值的查询会自动记录：

```python
import logging
logging.basicConfig(level=logging.WARNING)

# 配置慢查询阈值（毫秒）
config["vector_store"]["config"]["slow_query_threshold_ms"] = 500

# 超过 500ms 的查询会输出类似：
# WARNING:mem0.vector_stores.gaussdb:Slow query (1234ms): SELECT ...
```

### 8.2 操作延迟指标

```python
# 获取操作统计
stats = db.get_stats()
print(stats)
# {
#   "insert_count": 42,
#   "search_count": 156,
#   "avg_insert_ms": 12.3,
#   "avg_search_ms": 8.7,
#   "fallback_count": 0,
# }
```

### 8.3 Fallback 计数

当 BM25 或其他能力降级时，fallback 计数会增加：

```python
print(f"Fallback count: {db.fallback_count}")
```

### 8.4 Collection 信息

```python
info = db.col_info()
# {
#   "collection_name": "memories",
#   "vector_dims": 1536,
#   "vector_metric": "cosine",
#   "bm25_enabled": True,
#   "payload_mode": "jsonb",
#   "filter_mode": "expression_index",
#   "indexes": ["gsdiskann_idx", "bm25_idx"],
#   "row_count": 1024,
# }
```

## 9. 常见问题排查

### 9.1 连接失败

**症状：** `psycopg2.OperationalError: could not connect to server`

**排查步骤：**

1. 确认 GaussDB 服务正在运行
2. 确认 host/port 可达（`telnet your-host 5432`）
3. 确认用户名密码正确
4. 确认数据库存在（`psql -h your-host -U your_user -l`）
5. 检查防火墙规则

```python
# 测试连接
import psycopg2
conn = psycopg2.connect(
    host="your-host", port=5432, database="mem0_db",
    user="your_user", password="your_password"
)
print("Connection OK")
conn.close()
```

### 9.2 向量维度不匹配

**症状：** `ERROR: vector dimension mismatch`

**原因：** `embedding_model_dims` 配置与实际 embedding 模型输出维度不一致，或者 collection 已用不同维度创建。

**解决：**

```python
# 方法一：修改配置与模型一致
config["vector_store"]["config"]["embedding_model_dims"] = 1536  # 与模型一致

# 方法二：删除旧 collection 重建
db.delete_col()  # 删除旧表
# 重新初始化 Memory 会自动创建新表
```

### 9.3 BM25 索引创建失败

**症状：** 初始化时 warning `BM25 probe failed, disabling BM25`

**原因：** GaussDB 版本不支持 BM25，或 `enable_vectordb` 未开启。

**解决：**

```bash
# 检查 GaussDB 版本
SELECT version();

# 确认向量功能已开启
SHOW enable_vectordb;

# 如果未开启，联系 DBA 开启
gs_guc reload -D <datadir> -c "enable_vectordb=on"
```

如果确认不需要 BM25，可以显式禁用避免 warning：

```python
config["vector_store"]["config"]["bm25_mode"] = "disabled"
```

### 9.4 Scope 过滤错误

**症状：** `ValueError: At least one scope filter (user_id/agent_id/run_id) is required`

**原因：** `require_scoped_filters=True`（默认），但 search/list 调用没有传 scope。

**解决：**

```python
# 正确：传入 scope
results = m.search("query", user_id="alice")

# 或者关闭强制 scope（不推荐生产环境）
config["vector_store"]["config"]["require_scoped_filters"] = False
```

### 9.5 分布式模式下 BM25 不可用

分布式模式（`deployment_mode="distributed"`）不支持 BM25，这是已知限制。

```python
# 分布式模式下自动禁用 BM25
config["vector_store"]["config"]["deployment_mode"] = "distributed"
# bm25_mode 会自动设为 disabled，无需手动配置
```

### 9.6 内存不足（大规模数据）

**症状：** 批量写入时 OOM 或超时

**解决：**

```python
# 减小批量写入大小
config["vector_store"]["config"]["batch_size"] = 50  # 默认 100

# 增加连接池大小
config["vector_store"]["config"]["maxconn"] = 20
```

### 9.7 中文搜索效果差

**原因：** BM25 的 `text_lemmatized` 列使用 spaCy 英文模型处理，对中文效果有限。

**建议：**

- 对于中文场景，主要依赖语义搜索（向量召回）
- 确保 embedding 模型支持中文（如 `text-embedding-3-small` 支持多语言）
- 中英混合内容会同时走语义和 BM25 两路召回，融合后效果通常较好
