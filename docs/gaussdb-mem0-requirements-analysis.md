# mem0 GaussDB 适配需求分析

> 更新时间: 2026-05-16  
> 适用范围: `mem0.vector_stores.gaussdb.GaussDB` 当前实现  
> 目标读者: 产品经理、架构师、数据库研发、测试、交付、售前

## 1. 文档目标

本文档回答四个问题:

1. mem0 到底如何使用一个 vector store provider。
2. GaussDB 适配要解决哪些真实业务问题。
3. 其他主流 provider 是怎么设计的，GaussDB 应该对齐什么，不应该盲目对齐什么。
4. 以当前代码实现为准，集中式和分布式分别已经做到什么程度，还存在哪些边界。

本文档是需求分析，不展开具体 SQL 细节和类图；具体实现见技术设计文档。

## 2. 背景

mem0 的定位不是普通聊天记录存档，而是 AI 应用的长期记忆层。它把对话和业务事实转成可检索的 memory，再通过语义检索、关键词检索、实体增强和 metadata filter，把相关记忆在后续请求时召回给模型。

对企业客户来说，单独再引入一个外部向量库往往会带来以下问题:

- 数据分散，审计和备份链路变复杂。
- 用户、Agent、Run 等隔离语义无法直接复用现有数据库治理。
- 运维团队不愿意为一个 memory 子系统额外维护一套新基础设施。
- 需要一个可直接纳入企业数据库标准体系的商用落地方案。

GaussDB 适配的目标，就是让 mem0 的长期记忆能力可以直接落在 GaussDB 上，并尽可能保持与 mem0 现有 provider 生态一致的使用体验。

## 3. 问题定义

GaussDB 适配不是“写一个表，塞 embedding”这么简单。它至少要同时满足以下几件事:

- 接上 mem0 的标准 provider 契约。
- 让 `Memory.add/search/update/delete/get_all` 能真实跑通。
- 支持向量检索和可选关键词检索。
- 支持商用场景必须要有的 scope 隔离。
- 在 GaussDB 能力存在版本差异、DDL 差异、索引差异时，做出明确而可解释的降级。
- 提供足够完备的测试样例，能作为交付门禁。

## 4. mem0 使用路径走读

### 4.1 初始化路径

以 `Memory.from_config(...)` 为入口，mem0 的关键初始化链路如下:

```text
Memory.from_config
  -> MemoryConfig 校验
  -> EmbedderFactory.create(...)
  -> VectorStoreFactory.create("gaussdb", ...)
  -> GaussDB(...)
     -> 创建连接池
     -> capability probe
     -> auto_create 时检查 collection 是否存在
     -> create_col()
```

对 GaussDB 适配而言，这意味着:

- provider 必须能被 `VectorStoreFactory` 正确注册和构造。
- 构造阶段就要完成运行时能力确认。
- collection 生命周期必须是 provider 自管理，而不能依赖用户手工先建完整 schema。

### 4.2 写入路径

`Memory.add(...)` 的主路径如下:

```text
Memory.add
  -> 校验 user_id / agent_id / run_id
  -> 组装 metadata
  -> infer=True 时做 memory 提取
  -> embedder 生成向量
  -> vector_store.insert(vectors, payloads, ids)
```

GaussDB provider 需要承接的真实需求:

- 支持单条和批量写入。
- 支持基于 id 的 upsert。
- 自动从 payload 派生 `memory` 与 `text_lemmatized`。
- 在商用默认下，把 `user_id / agent_id / run_id` 写入 payload，并在需要时写入冗余列。

### 4.3 查询路径

`Memory.search(...)` 的主路径如下:

```text
Memory.search
  -> 校验 filters 至少包含 user_id / agent_id / run_id 之一
  -> 识别高级 metadata operator
  -> embed query
  -> vector_store.search(...)
  -> vector_store.keyword_search(...)
  -> entity boost
  -> score_and_rank
```

这里有两个关键点:

1. mem0 上层会暴露 `eq/ne/in/nin/gt/gte/lt/lte/contains/icontains/AND/OR/NOT` 这套 filter 接口。
2. 但每个 provider 并不一定完整支持所有 operator。

因此，GaussDB 的需求不是“机械支持所有接口名”，而是:

- 对支持的能力，要行为正确。
- 对不支持的能力，要语义明确、结果可解释、不要静默产生错误结果。

### 4.4 更新与删除路径

`Memory.update(...)` 和 `Memory.delete(...)` 在 mem0 当前架构里，最终都会走 provider 的 `update(id, ...)` 与 `delete(id)`。

这意味着一个天然事实:

- provider 层的 `get/update/delete` 是按 id 操作。
- mem0 标准接口本身并没有把 scope filter 一并传给这些方法。

所以如果业务要求“按 id 的读写也必须强租户隔离”，那是上层 API 或鉴权层的责任，不是单个 vector store provider 在当前契约下能独立彻底解决的问题。

## 5. 目标用户与典型场景

### 5.1 目标用户

- 需要把 AI memory 纳入企业数据库体系的客户。
- 希望直接复用 GaussDB 运维体系的数据库团队。
- 需要构建 Agent 平台、知识助理、智能客服、CRM Copilot 的应用研发团队。
- 需要有一套可测试、可验收、可交付口径的售前和交付团队。

### 5.2 典型场景

- 单用户长期偏好记忆
- 多租户 SaaS 助手
- Agent / Run 过程记忆
- 企业知识问答辅助记忆
- 审计敏感场景下的可删除、可追踪 memory 存储

## 6. 需求规划

### 6.1 P0 需求

P0 的定义是“必须能作为一个可用 provider 跑通 mem0 主链路”。

- `provider=gaussdb` 可通过 `Memory.from_config` 初始化。
- 支持集中式 GaussDB collection 自动创建。
- 支持 `insert/search/update/delete/get/list/reset/col_info/list_cols`。
- 支持 `search_batch`。
- 集中式支持 BM25，可降级。
- 支持 `eq/ne/in/nin/contains/icontains` 及 `AND/OR/NOT`。
- 默认要求 `user_id / agent_id / run_id` 中至少一个正向 scope filter。
- UTF-8 client encoding 明确设置为 `UTF8`。
- capability probe 能识别向量、JSONB、BM25、表达式索引等能力。
- 当表达式索引不可用时，metadata filter 语义不丢。

### 6.2 P1 需求

P1 的定义是“商用风险要可解释、可验证”。

- JSONB 不可用时退化到 text payload + redundant scope columns。
- BM25 不可用时只关闭关键词检索，不拖垮主链路。
- `search_batch` 原生 SQL 失败时可退化为逐条搜索。
- 有 live tests 和 commercial validation tests。
- 对集中式与分布式边界有明确说明。

### 6.3 P2 需求

P2 的定义是“便于长期交付与维护”。

- `migration_dry_run`
- `backfill_derived_fields`
- `analyze`
- retry / slow query / fallback metrics
- 更细的迁移与运维文档

## 7. 友商与现有 provider 设计分析

这里的“友商”主要指 mem0 现有 provider 设计，而不是数据库厂商对外营销口径。

### 7.1 pgvector

特点:

- 典型 SQL provider。
- `create_col()` 只操作当前实例绑定的 collection。
- 连接池默认 `minconn=1, maxconn=5`。
- 过滤通常依赖 `payload->>'key'` 这种 JSON 文本抽取。
- 没有 provider-level scope guard。

对 GaussDB 的启发:

- 单 collection 实例模型是合理的。
- 显式连接池参数是合理的。
- SQL/JSON 类 provider 天生更难做 typed range。

### 7.2 Azure MySQL

特点:

- 也是 SQL provider。
- 默认连接池也是 `1/5`。
- `create_col(name=...)` 能创建别的表，但后续 CRUD 仍主要绑定当前实例的 collection。
- `JSON_EXTRACT` 为主，filter 语义偏文本。

对 GaussDB 的启发:

- 半支持多表名很容易造成语义混乱。
- 如果不打算做完整多 collection 管理，就应该明确采用单 collection 实例模型。

### 7.3 MongoDB

特点:

- 文档模型，filter 天然按 `payload.key` 走。
- 不需要额外设计 scope 冗余列。
- Atlas Search 与向量检索是两套索引体系。

对 GaussDB 的启发:

- metadata filter 能力与索引能力应该解耦。
- 即使索引创建失败，也不应该把 filter 语义砍掉。

### 7.4 Qdrant

特点:

- 原生 typed payload filter。
- 支持 numeric range 与 datetime range。
- 不需要用户额外声明 SQL cast，因为底层 payload 本身就是 typed。

对 GaussDB 的启发:

- `gt/gte/lt/lte` 在 Qdrant 这类引擎里天然成立。
- 但 GaussDB 当前实现基于 `payload->>'key'`，得到的是文本，不具备自动 typed range 基础。

因此，GaussDB 当前阶段不支持 typed range 是可以接受的，但必须在文档和测试里明确。

### 7.5 OpenSearch / Elasticsearch / Redis

特点:

- 更多依赖搜索引擎或 schema 化字段。
- 通常对部分 filter 字段有较强支持，但不一定等价于任意 payload JSON 范式。

对 GaussDB 的启发:

- 不是所有 provider 都支持任意复杂 metadata operator。
- “和友商一致”更应该理解为语义清晰、边界明确，而不是表面上暴露同名 operator。

## 8. 当前 GaussDB 适配的需求落地状态

### 8.1 已完成

- provider 注册完成。
- `GaussDBConfig` 可校验连接、维度、部署模式、索引类型、连接池大小。
- 集中式主链路实现完成。
- 分布式兼容路径实现完成。
- expression index fallback 已修正为“降性能、不降语义”。
- `create_col` 已收敛为单 collection 实例模型。
- range operator 已明确设为“不支持 typed range，warning + literal equality”。
- `analyze()` 已改成 autocommit 路径。
- 商用门禁测试文件已建立。

### 8.2 明确边界

- 当前不支持 typed range。
- 当前 `get/update/delete` 不带 scope guard。
- BM25 是 auto-enable / auto-disable 语义，不是 required/fail-fast 语义。
- 分布式模式当前不支持 BM25，因此也不提供 `keyword_search` 能力。
- 集中式是主推荐模式；分布式已做兼容实现和样例，但商用结论仍依赖客户内网实库复验。

## 9. 关键设计要求

### 9.1 强 scope 隔离

这是 GaussDB 方案区别于多数现有 SQL provider 的关键点之一。

要求:

- `search`
- `keyword_search`
- `search_batch`
- `list`

默认都必须要求至少一个正向 scope filter。

原因:

- 商用 memory 场景天然多租户。
- 相比“默认全表可搜”，强约束更适合企业交付。

### 9.2 结果正确性优先于表面兼容

对 `gt/gte/lt/lte` 的策略就是这个原则的体现。

不应因为 mem0 上层暴露了某个 operator，就在底层用错误的字符串比较强行“假支持”。  
错误结果比明确不支持更危险。

### 9.3 索引能力与过滤语义解耦

JSON expression index 失败，不应导致 metadata filter 只剩 `user_id / agent_id / run_id`。

正确做法是:

- JSONB 可用 -> 任意 metadata filter 继续可用
- 表达式索引失败 -> 只是少了加速
- JSONB 本身不可用 -> 才退化到 text + redundant columns

### 9.4 单实例单 collection 模型

`self.collection_name` 只有一个。  
因此 GaussDB provider 不应该假装支持 `create_col(name="other")` 这种动态切表语义。

## 10. 验收口径

### 10.1 单元测试

覆盖以下内容:

- 配置校验
- 建表 SQL
- 索引创建
- capability probe
- fallback 行为
- filter 语义
- retry / rollback / analyze

### 10.2 集中式 live tests

重点验证:

- CRUD
- filter matrix
- scope isolation
- UTF-8
- BM25
- batch search
- e2e memory 链路

### 10.3 商用门禁样例

`test_gaussdb_commercial_validation.py` 用于承接更明确的交付验收，包括:

- collection contract
- CRUD smoke
- filter matrix
- unsupported range behavior
- vector order and top_k
- UTF-8 roundtrip
- keyword search
- distributed smoke

## 11. 需求结论

基于当前实现，GaussDB 集中式已经满足以下结论:

- 能作为 mem0 的一个可用 provider 落地。
- 能支撑主流商用 memory 场景。
- 在 SQL/JSON 类 provider 中，设计已经比较扎实。

但同时也必须明确:

- 它还不是“所有 mem0 provider 中能力最强”的实现。
- 它没有对齐 Qdrant 这类原生 typed payload range 能力。
- 它也不能脱离上层鉴权，单独承诺按 id 的强租户隔离。

因此，最准确的产品口径应是:

> GaussDB 集中式已经可以完整支撑 mem0 的核心商用场景；当前已知边界主要是 typed range 未支持，以及按 id 的管理接口仍需上层鉴权配合。

## 12. 后续建议

建议将后续工作分两条线推进:

1. 商用交付线
   - 保持集中式主路径稳定。
   - 完善用户手册、部署说明、故障排查。
   - 用 commercial validation suite 作为交付门禁。

2. 能力增强线
   - 如果后续确实有强需求，再设计 typed metadata range。
   - 如果要承诺更强隔离，再推动上层 API 契约扩展。
   - 分布式场景继续做公司内网实库复验，沉淀单独结论。

## 13. 配置分层建议

为了降低对外接入复杂度，同时保留商用部署所需的调优空间，建议将 GaussDB 配置按两层表达:

- 基础项: 建议显式填写，包括连接信息、`collection_name`、`embedding_model_dims`、`deployment_mode`
- 高阶项: 可以不填，由 provider 提供默认值，包括 `sslmode`、`sslrootcert`、`minconn`、`maxconn`、`vector_index_type`、`vector_metric`、`auto_create`、`require_scoped_filters`

这里的“基础项”当前是交付和文档层面的约定，不是运行时硬性必填。当前代码仍然保留默认值，但商用接入时建议显式写出这些关键字段，避免落入默认 collection、默认维度或默认部署模式。
