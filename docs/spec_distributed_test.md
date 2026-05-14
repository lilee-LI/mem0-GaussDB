# GaussDB 分布式模式 mem0 适配测试用例设计

## 1. 测试背景

GaussDB 分布式部署模式（`deployment_mode="distributed"`）与集中式模式存在以下关键差异：

| 维度 | 集中式 | 分布式 |
|------|--------|--------|
| 表分布 | 无 | DISTRIBUTE BY HASH (id) |
| 向量维度上限 | 4096 | 1024 |
| BM25 全文搜索 | 支持 | 不支持（自动禁用） |
| 数据分片 | 单节点 | 多 DN 节点 |
| 事务一致性 | 本地事务 | 分布式事务（GTM 协调） |

测试目标：验证 GaussDB 适配层在分布式模式下的功能正确性、降级行为、数据一致性。

---

## 2. 测试环境要求

- GaussDB 分布式集群（至少 2 DN 节点）
- Python 3.9+
- psycopg2 驱动
- 测试用 embedding 维度：4 维（快速验证）和 1024 维（边界验证）

---

## 3. 测试用例

### P0 - 初始化与参数校验

#### TC-D001: 分布式模式基本初始化
- **前置条件**: 分布式 GaussDB 集群可用
- **测试步骤**:
  1. 创建 `GaussDB(deployment_mode="distributed", embedding_model_dims=4, ...)`
  2. 检查 `db.distribution_mode == "hash"`
  3. 检查 `db.bm25_enabled == False`
- **预期结果**: 初始化成功，distribution_mode 自动解析为 hash，BM25 自动禁用
- **优先级**: P0

#### TC-D002: 分布式模式维度超限拒绝
- **前置条件**: 无
- **测试步骤**:
  1. 尝试 `GaussDB(deployment_mode="distributed", embedding_model_dims=1025)`
- **预期结果**: 抛出 `ValueError`，错误信息包含 "distributed mode only supports embedding dimensions <= 1024"
- **优先级**: P0

#### TC-D003: 分布式模式维度边界 1024
- **前置条件**: 分布式集群可用
- **测试步骤**:
  1. `GaussDB(deployment_mode="distributed", embedding_model_dims=1024, ...)`
  2. 插入一条 1024 维向量
  3. 搜索验证
- **预期结果**: 初始化成功，插入和搜索正常
- **优先级**: P0

#### TC-D004: bm25_mode="required" + distributed 冲突
- **前置条件**: 无
- **测试步骤**:
  1. 尝试 `GaussDB(deployment_mode="distributed", bm25_mode="required")`
- **预期结果**: 抛出 `ValueError`，错误信息包含 "bm25_mode='required' is incompatible with deployment_mode='distributed'"
- **优先级**: P0

#### TC-D005: distribution_mode="auto" 自动解析
- **前置条件**: 无
- **测试步骤**:
  1. `GaussDB(deployment_mode="distributed", distribution_mode="auto")` → 验证 distribution_mode == "hash"
  2. `GaussDB(deployment_mode="centralized", distribution_mode="auto")` → 验证 distribution_mode == "none"
- **预期结果**: auto 根据 deployment_mode 正确解析
- **优先级**: P0

#### TC-D006: centralized + distribution_mode="hash" 冲突
- **前置条件**: 无
- **测试步骤**:
  1. 尝试 `GaussDB(deployment_mode="centralized", distribution_mode="hash")`
- **预期结果**: 抛出 `ValueError`，错误信息包含 "distribution_mode can only be enabled when deployment_mode='distributed'"
- **优先级**: P0

---

### P0 - DDL 生成与表创建

#### TC-D010: 分布式表创建 DDL 包含 DISTRIBUTE BY HASH
- **前置条件**: 分布式集群可用
- **测试步骤**:
  1. 创建分布式模式 GaussDB 实例（auto_create=True）
  2. 查询 `pg_class` 或 `pgxc_class` 验证表的分布方式
  3. 验证分布键为 id 列
- **预期结果**: 表成功创建，分布方式为 HASH，分布键为 id
- **优先级**: P0

#### TC-D011: schema_meta 表也带分布子句
- **前置条件**: 分布式集群可用
- **测试步骤**:
  1. 创建分布式模式实例
  2. 查询 schema_meta 表的分布信息
- **预期结果**: mem0_schema_meta 表也使用 DISTRIBUTE BY HASH
- **优先级**: P0

#### TC-D012: 能力探测表（probe table）在分布式模式下正确创建和清理
- **前置条件**: 分布式集群可用，enable_capability_probe=True
- **测试步骤**:
  1. 创建实例，观察 probe 过程
  2. 验证 probe 表被正确清理（DROP TABLE）
  3. 验证 probe 结果：vector_enabled=True, bm25=False
- **预期结果**: probe 表创建带 DISTRIBUTE BY HASH，探测完成后被清理，BM25 探测跳过或失败后正确标记
- **优先级**: P0

---

### P0 - CRUD 基本操作

#### TC-D020: 分布式模式下 insert 单条记录
- **前置条件**: 分布式表已创建
- **测试步骤**:
  1. 插入一条记录 (id, vector, payload)
  2. 通过 get() 读回
- **预期结果**: 数据完整，payload 无损
- **优先级**: P0

#### TC-D021: 分布式模式下 insert 批量记录
- **前置条件**: 分布式表已创建
- **测试步骤**:
  1. 批量插入 20 条记录
  2. list() 验证总数
  3. 随机 get 3 条验证数据完整性
- **预期结果**: 所有记录正确写入，数据分布在多个 DN 上
- **优先级**: P0

#### TC-D022: 分布式模式下 update
- **前置条件**: 已插入记录
- **测试步骤**:
  1. 更新某条记录的 vector 和 payload
  2. get() 验证更新后的值
- **预期结果**: 更新成功，新值正确
- **优先级**: P0

#### TC-D023: 分布式模式下 delete 单条
- **前置条件**: 已插入记录
- **测试步骤**:
  1. 删除一条记录
  2. get() 验证返回 None
  3. list() 验证总数减 1
- **预期结果**: 删除成功
- **优先级**: P0

#### TC-D024: 分布式模式下 delete 批量（ids 参数）
- **前置条件**: 已插入多条记录
- **测试步骤**:
  1. 一次删除 5 条记录（传 ids 列表）
  2. 验证全部删除成功
- **预期结果**: 批量删除正确，即使 id 分布在不同 DN 上
- **优先级**: P0

---

### P0 - 向量搜索

#### TC-D030: 分布式模式下 cosine search 正确性
- **前置条件**: 已插入多条已知向量的记录
- **测试步骤**:
  1. 插入向量 A=[1,0,0,0], B=[0,1,0,0], C=[0.9,0.1,0,0]
  2. 用 query=[1,0,0,0] 搜索 top_k=3
  3. 验证排序：A > C > B
- **预期结果**: 排序正确，score 符合 cosine 语义
- **优先级**: P0

#### TC-D031: 分布式模式下 search 跨 DN 聚合
- **前置条件**: 插入 100 条随机向量（确保分布在多个 DN）
- **测试步骤**:
  1. 搜索 top_k=10
  2. 验证返回 10 条结果
  3. 验证 score 单调递减
- **预期结果**: 跨 DN 的结果正确聚合排序
- **优先级**: P0

#### TC-D032: 分布式模式下 search_batch
- **前置条件**: 已插入多条记录
- **测试步骤**:
  1. 同时搜索 3 个不同的 query vector
  2. 验证每个 query 返回独立的结果集
- **预期结果**: batch search 在分布式模式下正常工作
- **优先级**: P0

#### TC-D033: 分布式模式下 search + filter
- **前置条件**: 插入带不同 user_id 的记录
- **测试步骤**:
  1. 搜索时带 filter `{"user_id": "user_A"}`
  2. 验证结果只包含 user_A 的记录
- **预期结果**: 过滤在分布式模式下正确工作
- **优先级**: P0

---

### P1 - BM25 降级行为

#### TC-D040: keyword_search 在分布式模式下返回 None
- **前置条件**: 分布式模式实例
- **测试步骤**:
  1. 调用 `db.keyword_search("some query")`
- **预期结果**: 返回 None（不是报错），因为 BM25 已禁用
- **优先级**: P1

#### TC-D041: bm25_enabled=True 但 distributed 时自动降级
- **前置条件**: 无
- **测试步骤**:
  1. `GaussDB(deployment_mode="distributed", bm25_enabled=True, bm25_fail_fast=False)`
  2. 检查 `db.bm25_enabled`
- **预期结果**: bm25_enabled 被自动设为 False，日志输出 "BM25 disabled: not supported in distributed deployment mode"
- **优先级**: P1

#### TC-D042: 上层 Memory.search 在分布式模式下的降级
- **前置条件**: 通过 mem0 Memory 接口使用分布式 GaussDB
- **测试步骤**:
  1. `m.add("I love hotpot", user_id="u1")`
  2. `m.search("hotpot", user_id="u1")`
- **预期结果**: 搜索正常返回结果（仅用向量搜索，BM25 分支被跳过），不报错
- **优先级**: P1

---

### P1 - 过滤器

#### TC-D050: 分布式模式下 scope filter（user_id, agent_id, run_id）
- **前置条件**: 插入多租户数据
- **测试步骤**:
  1. 插入 user_A 的 5 条记录和 user_B 的 5 条记录
  2. search(filter={"user_id": "user_A"}) 
  3. 验证只返回 user_A 的记录
- **预期结果**: scope 过滤在分布式模式下正确
- **优先级**: P1

#### TC-D051: 分布式模式下复合过滤（AND/OR）
- **前置条件**: 插入带多种 metadata 的记录
- **测试步骤**:
  1. 插入记录：{user_id: "A", category: "work"}, {user_id: "A", category: "personal"}, {user_id: "B", category: "work"}
  2. search(filter={"user_id": "A", "category": "work"})
- **预期结果**: 只返回同时满足两个条件的记录
- **优先级**: P1

#### TC-D052: 分布式模式下范围过滤（gte/lte）
- **前置条件**: 插入带时间戳 metadata 的记录
- **测试步骤**:
  1. 插入 10 条记录，created_at 从 1 到 10
  2. search(filter={"created_at": {"gte": 5, "lte": 8}})
- **预期结果**: 只返回 created_at 在 [5,8] 范围内的记录
- **优先级**: P1

#### TC-D053: 分布式模式下 IN 过滤
- **前置条件**: 插入多条记录
- **测试步骤**:
  1. search(filter={"user_id": {"in": ["A", "C"]}})
- **预期结果**: 返回 user_id 为 A 或 C 的记录
- **优先级**: P1

---

### P1 - 数据一致性

#### TC-D060: 插入后立即可读（read-your-writes）
- **前置条件**: 分布式表
- **测试步骤**:
  1. 插入一条记录
  2. 立即 get() 读取
- **预期结果**: 能立即读到刚写入的数据（GaussDB 分布式事务保证强一致）
- **优先级**: P1

#### TC-D061: 更新后搜索结果反映新值
- **前置条件**: 已插入记录 A，vector=[1,0,0,0]
- **测试步骤**:
  1. 更新 A 的 vector 为 [0,1,0,0]
  2. 用 query=[0,1,0,0] 搜索
  3. 验证 A 排在第一位
- **预期结果**: 索引在更新后正确反映新向量
- **优先级**: P1

#### TC-D062: 删除后搜索不返回已删除记录
- **前置条件**: 已插入记录
- **测试步骤**:
  1. 删除记录 A
  2. 搜索，验证结果中不包含 A
- **预期结果**: 已删除记录不出现在搜索结果中
- **优先级**: P1

#### TC-D063: 并发写入不丢数据
- **前置条件**: 分布式表
- **测试步骤**:
  1. 10 个线程并发插入，每个线程插入 10 条记录（共 100 条）
  2. 等待全部完成
  3. list() 验证总数 == 100
- **预期结果**: 无数据丢失
- **优先级**: P1

#### TC-D064: 并发更新同一条记录
- **前置条件**: 已插入记录 A
- **测试步骤**:
  1. 5 个线程并发更新 A 的 payload（每个线程写不同值）
  2. 等待全部完成
  3. get(A) 验证 payload 是某个线程写入的完整值（不是混合值）
- **预期结果**: 最终值是某次完整更新的结果，不会出现数据撕裂
- **优先级**: P1

---

### P1 - 集合生命周期

#### TC-D070: 分布式模式下 create_col
- **前置条件**: 分布式集群
- **测试步骤**:
  1. 创建集合
  2. 验证表存在且分布方式正确
  3. 验证向量索引存在
  4. 验证 BM25 索引不存在
- **预期结果**: 集合正确创建，无 BM25 索引
- **优先级**: P1

#### TC-D071: 分布式模式下 delete_col
- **前置条件**: 已创建分布式集合
- **测试步骤**:
  1. 删除集合
  2. 验证表不存在
  3. 验证 schema_meta 中对应记录被清理
- **预期结果**: 集合完全清理
- **优先级**: P1

#### TC-D072: 分布式模式下 list_cols
- **前置条件**: 创建多个分布式集合
- **测试步骤**:
  1. 创建 col_a, col_b, col_c
  2. list_cols() 验证包含这三个
  3. 删除 col_b
  4. list_cols() 验证不再包含 col_b
- **预期结果**: list_cols 正确反映当前状态
- **优先级**: P1

#### TC-D073: 分布式模式下 col_info
- **前置条件**: 已创建分布式集合并插入数据
- **测试步骤**:
  1. 调用 col_info()
  2. 验证返回的 deployment_mode, distribution_mode, vector_count 等字段
- **预期结果**: 信息准确，deployment_mode="distributed", distribution_mode="hash"
- **优先级**: P1

---

### P1 - 向量索引

#### TC-D080: 分布式模式下 GsDiskANN 索引创建
- **前置条件**: 分布式集群
- **测试步骤**:
  1. `GaussDB(deployment_mode="distributed", vector_index_type="gsdiskann", embedding_model_dims=128)`
  2. 验证索引创建成功
  3. 插入数据并搜索验证索引生效
- **预期结果**: GsDiskANN 索引在分布式模式下正常工作
- **优先级**: P1

#### TC-D081: 分布式模式下 HNSW 索引（如果支持）
- **前置条件**: 分布式集群
- **测试步骤**:
  1. `GaussDB(deployment_mode="distributed", vector_index_type="hnsw", embedding_model_dims=128)`
  2. 观察是否成功或报错
- **预期结果**: 如果 GaussDB 分布式支持 HNSW 则成功；否则应有明确错误信息
- **优先级**: P1

---

### P2 - 边界条件与异常

#### TC-D090: 分布式模式下空表搜索
- **前置条件**: 创建分布式集合但不插入数据
- **测试步骤**:
  1. search(query_vector) top_k=5
- **预期结果**: 返回空列表，不报错
- **优先级**: P2

#### TC-D091: 分布式模式下 top_k 大于总记录数
- **前置条件**: 插入 3 条记录
- **测试步骤**:
  1. search top_k=100
- **预期结果**: 返回 3 条结果，不报错
- **优先级**: P2

#### TC-D092: 分布式模式下大 payload
- **前置条件**: 分布式表
- **测试步骤**:
  1. 插入 payload 包含 10KB 文本的记录
  2. get() 验证完整性
- **预期结果**: 大 payload 在分布式模式下正确存储和读取
- **优先级**: P2

#### TC-D093: 分布式模式下 Unicode/中文 payload
- **前置条件**: 分布式表
- **测试步骤**:
  1. 插入 payload 包含中文、日文、emoji 的记录
  2. get() 验证无乱码
  3. search + filter 包含中文值
- **预期结果**: 多语言字符正确处理
- **优先级**: P2

#### TC-D094: 分布式模式下重复 ID 插入（upsert 语义）
- **前置条件**: 已插入 id="abc" 的记录
- **测试步骤**:
  1. 再次插入 id="abc" 但不同 vector 和 payload
  2. get("abc") 验证值
- **预期结果**: 行为与 upsert 一致（更新已有记录），不产生重复行
- **优先级**: P2

#### TC-D095: 分布式模式下 NULL/空值 payload 字段
- **前置条件**: 分布式表
- **测试步骤**:
  1. 插入 payload={"user_id": "A", "note": None, "tags": []}
  2. get() 验证
  3. filter({"user_id": "A"}) 验证
- **预期结果**: NULL 和空值正确存储，过滤不受影响
- **优先级**: P2

---

### P2 - 可观测性与重试

#### TC-D100: 分布式模式下 metrics 正确记录
- **前置条件**: enable_observability=True
- **测试步骤**:
  1. 执行 insert, search, update, delete 各一次
  2. 检查 db.metrics
- **预期结果**: 各操作计数正确，无 BM25 相关计数
- **优先级**: P2

#### TC-D101: 分布式模式下连接重试
- **前置条件**: 分布式集群
- **测试步骤**:
  1. 设置 retry_attempts=3
  2. 模拟一次瞬时连接失败（如果可能）
  3. 验证重试后成功
- **预期结果**: 重试机制在分布式模式下正常工作
- **优先级**: P2

#### TC-D102: 分布式模式下慢查询日志
- **前置条件**: slow_query_ms=1
- **测试步骤**:
  1. 执行一次搜索（大概率超过 1ms）
  2. 检查日志输出包含慢查询警告
- **预期结果**: 慢查询检测在分布式模式下正常
- **优先级**: P2

---

### P2 - mem0 上层集成

#### TC-D110: Memory.add 在分布式 GaussDB 上
- **前置条件**: mem0 Memory 配置使用分布式 GaussDB
- **测试步骤**:
  1. `m.add("I had a great meeting today", user_id="alice")`
  2. 验证返回的 memory id 非空
- **预期结果**: 记忆成功添加
- **优先级**: P2

#### TC-D111: Memory.search 在分布式 GaussDB 上
- **前置条件**: 已添加记忆
- **测试步骤**:
  1. `m.search("meeting", user_id="alice")`
  2. 验证返回相关记忆
- **预期结果**: 搜索正常（仅向量搜索，无 BM25）
- **优先级**: P2

#### TC-D112: Memory.update 在分布式 GaussDB 上
- **前置条件**: 已添加记忆
- **测试步骤**:
  1. `m.update(memory_id, "Updated: great meeting with the team")`
  2. `m.get(memory_id)` 验证内容更新
- **预期结果**: 更新成功
- **优先级**: P2

#### TC-D113: Memory.delete 在分布式 GaussDB 上
- **前置条件**: 已添加记忆
- **测试步骤**:
  1. `m.delete(memory_id)`
  2. `m.get(memory_id)` 验证返回 None
- **预期结果**: 删除成功
- **优先级**: P2

#### TC-D114: Memory.get_all 在分布式 GaussDB 上
- **前置条件**: 已添加多条记忆
- **测试步骤**:
  1. `m.get_all(user_id="alice")`
  2. 验证返回所有 alice 的记忆
- **预期结果**: 列表完整
- **优先级**: P2

---

### P2 - 跨模式对比验证

#### TC-D120: 相同数据在集中式和分布式下搜索结果一致
- **前置条件**: 同时有集中式和分布式实例
- **测试步骤**:
  1. 向两个实例插入完全相同的 10 条记录（相同 id, vector, payload）
  2. 用相同 query vector 搜索 top_k=5
  3. 对比两个结果的 id 排序和 score
- **预期结果**: 排序完全一致，score 数值一致（因为都用 cosine + 相同归一化）
- **优先级**: P2

#### TC-D121: 相同过滤条件在两种模式下结果一致
- **前置条件**: 同上
- **测试步骤**:
  1. 用相同 filter 搜索
  2. 对比结果集
- **预期结果**: 结果集完全一致
- **优先级**: P2

---

## 4. 测试用例统计

| 优先级 | 数量 | 覆盖范围 |
|--------|------|---------|
| P0 | 12 | 初始化校验、DDL、基本 CRUD、向量搜索 |
| P1 | 14 | BM25 降级、过滤器、数据一致性、集合生命周期、索引 |
| P2 | 14 | 边界条件、可观测性、上层集成、跨模式对比 |
| **合计** | **40** | |

---

## 5. 测试执行建议

1. **P0 用例必须全部通过**才能认为分布式适配基本可用
2. P1 用例覆盖生产环境常见场景，建议在合入前全部通过
3. P2 用例为增强验证，可在后续迭代中补充
4. TC-D120/D121（跨模式对比）是最有说服力的验证——证明分布式模式不引入行为差异
5. 并发测试（TC-D063/D064）建议在真实多 DN 环境下执行，单 DN 模拟分布式无法暴露真实问题
