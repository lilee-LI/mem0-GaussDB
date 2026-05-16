# GaussDB 适配 mem0 最终验证清单

## 1. 目标

本文档用于给 GaussDB 适配 mem0 提供一套统一、可复用、可交付的最终验证清单。

目标不是覆盖所有理论路径，而是验证:

- 当前实现是否能稳定支撑 mem0 核心商用场景
- 集中式与分布式的已承诺能力是否符合预期
- 已知边界是否表现为明确、稳定、可解释的行为

---

## 2. 验收口径

当前推荐的对外口径是:

> GaussDB 已经可以支撑 mem0 的核心商用场景，尤其是集中式部署；当前已知边界主要包括未支持 typed metadata range、`get/update/delete` 仍为按 id 直操作、以及分布式模式当前不支持 BM25 / `keyword_search`。

因此，本清单的重点是:

- 验证“主链路可用”
- 验证“边界清晰且行为稳定”
- 不将当前未承诺能力误判为缺陷

---

## 3. 当前已承诺能力

### 3.1 集中式

集中式当前应能稳定支持:

- collection lifecycle: `create_col` / `list_cols` / `col_info` / `reset` / `delete_col`
- memory 数据面: `insert` / `search` / `search_batch` / `get` / `update` / `delete` / `list`
- metadata filter
- `user_id` / `agent_id` / `run_id` scope guard
- UTF-8 / 多语言 payload roundtrip
- BM25 `keyword_search`
- `analyze()`
- JSON expression filter index 失败后的语义保留

### 3.2 分布式

分布式当前应能稳定支持:

- collection lifecycle: `create_col` / `list_cols` / `col_info` / `reset` / `delete_col`
- memory 数据面: `insert` / `search` / `search_batch` / `get` / `update` / `delete` / `list`
- metadata filter
- `user_id` / `agent_id` / `run_id` scope guard
- UTF-8 / 多语言 payload roundtrip
- `analyze()`

分布式当前明确不承诺:

- BM25
- `keyword_search`
- 1024 维以上 embedding

---

## 4. 当前已知边界

以下行为应被视为“明确边界”，而不是 bug:

### 4.1 range filter

当前 `gt` / `gte` / `lt` / `lte` 不支持 typed range 语义。

预期行为:

- provider 打 warning
- 不执行 typed range SQL 过滤
- 不把字符串比较误当数值比较

### 4.2 按 id 管理接口

`get` / `update` / `delete` 当前都是按 `id` 直操作，不附带 scope filter。

预期行为:

- 与 `VectorStoreBase` 当前契约一致
- 不由 provider 单独承诺按 id 的强租户隔离
- 上层 API / 鉴权层需要兜底

### 4.3 score 语义

GaussDB 返回的是 provider-normalized semantic score，不是原始距离。

预期行为:

- 可在 GaussDB provider 内部稳定使用
- 不应与 pgvector raw distance 或其他 provider 原生 score 横向比较

### 4.4 distributed keyword search

分布式模式下 `keyword_search()` 当前不提供能力。

预期行为:

- 不报错
- 返回 `None` 或按上层约定跳过

---

## 5. 推荐执行顺序

建议按以下顺序执行验证。

### 5.1 第一步: 单测基线

执行:

```powershell
pytest D:\lxm\code\mem0-GaussDB\mem0_codex\mem0\tests\vector_stores\test_gaussdb.py -q
```

目的:

- 验证 provider 核心逻辑
- 验证配置校验
- 验证 fallback、warning、边界语义

通过标准:

- 全量通过

### 5.2 第二步: 商用门禁测试

执行:

```powershell
pytest D:\lxm\code\mem0-GaussDB\mem0_codex\mem0\tests\vector_stores\test_gaussdb_commercial_validation.py -q
```

目的:

- 验证交付口径下最核心的商用路径
- 验证集中式与分布式 smoke case

通过标准:

- 当前环境对应的场景全部通过
- 未提供环境的场景允许 skip

### 5.3 第三步: 集中式 live 回归

执行:

```powershell
$env:GAUSSDB_TEST_CENTRALIZED='true'
pytest D:\lxm\code\mem0-GaussDB\mem0_codex\mem0\tests\vector_stores\test_gaussdb_centralized.py -q
```

目的:

- 在真实 GaussDB 集中式环境中验证 SQL 路径、索引创建、编码、过滤语义

通过标准:

- 主用例通过
- 若有历史遗留非主路径 case，需要结合当前设计口径解释

### 5.4 第四步: 分布式 live 回归

执行:

```powershell
$env:GAUSSDB_TEST_DISTRIBUTED='true'
pytest D:\lxm\code\mem0-GaussDB\mem0_codex\mem0\tests\vector_stores\test_gaussdb_distributed.py -q
```

目的:

- 在真实 GaussDB 分布式环境中验证主链路、能力边界和运维行为

通过标准:

- distributed 主链路通过
- 明确接受“无 BM25 / 无 keyword_search / 最大 1024 维”的边界

---

## 6. 核心验证矩阵

### 6.1 collection lifecycle

需要验证:

- 首次初始化自动建表
- `list_cols()` 可见 collection
- `col_info()` 返回 schema 信息
- `reset()` 后表与索引可重建
- `delete_col()` 后 collection 消失

重点结论:

- provider 实例只绑定一个 `collection_name`
- `create_col()` 只创建当前 collection，不支持传其他 name

### 6.2 CRUD

需要验证:

- `insert()` 可写入多条 memory
- 同 id upsert 不产生重复记录
- `get()` 能按 id 取回 payload
- `update()` 能更新 vector / payload
- `delete()` 能删除记录

重点结论:

- upsert 主路径成立
- `update(payload=...)` 为覆盖式更新，不是 JSON merge

### 6.3 semantic search

需要验证:

- `search()` 返回相关结果
- 排序方向稳定
- `top_k` 正常工作
- `search_batch()` 在集中式和分布式下都可用

重点结论:

- score 可用于本 provider 内部排序与阈值判断
- 不做跨 provider 分数直接比较

### 6.4 metadata filter

需要验证:

- 普通 key 等值过滤
- `ne`
- `in` / `nin`
- `contains` / `icontains`
- `AND` / `OR` / `NOT`
- scope filter 与普通 metadata 组合

重点结论:

- JSON expression index 失败不应导致 metadata filter 能力丢失
- 只允许当前实现支持的 operator

### 6.5 unsupported range behavior

需要验证:

- 输入 `gt` / `gte` / `lt` / `lte`
- provider 打 warning
- 不产生错误字符串大小比较语义

重点结论:

- 当前版本对 range 的处理是“显式不承诺 typed range”

### 6.6 scope guard

需要验证:

- 默认开启时，无 scope 条件的 `search` / `list` / `search_batch` / `keyword_search` 被拒绝
- 只有负向 scope 条件时被拒绝
- 只有 `OR` 中部分分支带 scope 时按当前规则校验
- `require_scoped_filters=False` 时，普通 metadata filter 可执行

重点结论:

- GaussDB 的默认隔离策略比多数 mem0 provider 更严格

### 6.7 UTF-8 / 多语言

需要验证:

- 中文
- 英文
- 中英混合
- emoji 或特殊符号可按需要补充

重点结论:

- 数据可写入、可读回、可检索
- client encoding 相关逻辑稳定

### 6.8 BM25 / keyword_search

集中式验证:

- `keyword_search()` 可执行
- 结果与关键词相关
- BM25 不可用时降级行为明确

分布式验证:

- 不承诺 BM25
- `keyword_search()` 不作为通过条件

### 6.9 analyze

需要验证:

- `analyze()` 可执行
- 集中式可执行
- 分布式可执行

重点结论:

- 当前实现走 autocommit 路径
- 不应再出现 `analyze cannot be executed in a transaction block`

---

## 7. 商用推荐验收结论模板

建议最终汇报使用以下模板:

### 7.1 通过项

- 集中式主链路通过
- 分布式主链路通过
- UTF-8 与多语言验证通过
- scope guard 验证通过
- metadata filter 验证通过
- collection lifecycle 验证通过
- analyze 验证通过

### 7.2 明确边界

- 当前不支持 typed metadata range
- `get/update/delete` 不自带 scope guard
- 分布式不支持 BM25 / `keyword_search`
- score 不可跨 provider 直接比较

### 7.3 结论话术

可使用如下结论:

> GaussDB 当前已经可以支撑 mem0 的核心商用场景。集中式能力较完整，分布式主链路可用且边界清晰。当前剩余限制主要集中在 typed range、按 id 管理接口的上层鉴权协同，以及分布式不提供 BM25 / `keyword_search`。

---

## 8. 建议的最终交付门禁

建议将以下三项共同作为最终交付门禁:

1. `test_gaussdb.py`
2. `test_gaussdb_commercial_validation.py`
3. 目标环境对应的 centralized / distributed live tests

只有三者同时通过，才建议给出“该环境已完成最终验证”的结论。
