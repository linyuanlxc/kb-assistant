# LabKB 升级设计文档

## 1. 文档目标

本文档用于指导当前项目从"可用 Demo"升级为"可部署上线、功能齐全、体验良好"的实验室知识库系统。  
文档重点覆盖：

- 数据准备链路
- 索引构建与增量更新
- 生成链路（LLM + Prompt + 流式输出）
- 检索优化（向量 + BM25 + 混合 + 重排）
- 工程化能力（日志、调试、部署、性能）

## 2. 当前项目基线（现状）

当前项目已具备：

- FastAPI 前端对话界面（SSE 流式）
- 基于 Qdrant 的向量检索
- Qwen Embedding + Qwen Chat API 接入
- 基础的问答链路和流式输出
- 基础建库脚本（文档加载 + 切分 + 入库）

当前主要不足：

- 数据准备流程缺少统一元数据模型、父子文档管理与去重策略
- 索引层不支持标准化增量更新、版本管理与回滚
- 检索层只有向量检索，无 BM25 与混合检索
- 生成层缺少模型路由、查询重写策略分级、上下文治理
- 观测与调试能力不足（日志结构化、检索可解释性、性能剖析）
- 缺少可上线部署规范和验收指标

## 3. 目标架构（升级后）

建议采用分层架构，避免 UI、检索、生成、数据处理耦合：

1. `ingestion`（数据准备）
2. `indexing`（索引构建与管理）
3. `retrieval`（检索与重排）
4. `generation`（Prompt、LLM、流式输出）
5. `orchestration`（RAG 主流程编排）
6. `interfaces`（FastAPI Web 服务）
7. `observability`（日志、指标、调试追踪）

建议目录（目标）：

```txt
app/
core/
  config/
  ingestion/
  indexing/
  retrieval/
  generation/
  orchestration/
  observability/
services/
  api/
  worker/
scripts/
docs/
tests/
```

## 4. 模块详细设计

## 4.1 数据准备（Data Preparation）

### 4.1.1 文档加载

支持文件类型：

- `.md` / `.txt` / `.pdf` / `.docx` / `.html`

加载输出统一为 `RawDocument`：

```text
doc_id: str
source_path: str
source_type: str
title: str | None
content: str
checksum: str (sha256)
created_at: datetime
updated_at: datetime
extra_meta: dict
```

### 4.1.2 文档分块

目标：兼顾召回率和答案可读性。  
策略：

- 父块（Parent Chunk）：1000~1500 tokens，保留段落完整语义
- 子块（Child Chunk）：250~400 tokens，`overlap` 40~80
- 中英文分隔符优化（句号、换行、标题符号）

输出：

```text
parent_id, child_id, doc_id, chunk_text, chunk_index, token_count
```

### 4.1.3 元数据生成

子块元数据建议强制包含：

- `doc_id`
- `parent_id`
- `child_id`
- `source_path`
- `file_name`
- `file_type`
- `tags`（可为空）
- `lang`
- `created_at`
- `checksum`
- `version`

### 4.1.4 父子文档关联

核心原则：检索命中子块，拼装上下文时回溯父块。  
作用：

- 解决子块过碎导致答案缺上下文的问题
- 保证引用来源可追踪

### 4.1.5 父文档去重

去重分两层：

1. 文件级去重：`checksum` 相同则跳过
2. 语义级去重：父块 MinHash/SimHash 近似相似度 > 阈值（如 0.95）则标记重复

### 4.1.6 相关性排序（预处理侧）

在构建索引时为文档生成轻量特征，用于后续排序：

- 标题命中权重
- 最近更新时间权重
- 标签权重
- 来源可信度权重（手工配置）

## 4.2 索引构建（Indexing）

### 4.2.1 向量模型加载

统一 Embedding 接口 `EmbeddingProvider`：

- `embed_documents(texts) -> vectors`
- `embed_query(text) -> vector`
- `model_name`
- `dimension`

支持多 Provider（Qwen / Zhipu / OpenAI-compatible）。

### 4.2.2 向量库构建与保存

默认：`Chroma`（本地部署友好）。  
要求：

- `collection_name` 与业务隔离
- 持久化路径可配置
- 支持索引快照版本（`index_version`）

### 4.2.3 索引加载

加载时校验：

- 向量维度一致性
- 索引版本可用性
- 元数据字段完整性

### 4.2.4 新增文档与增量更新

增量策略：

1. 扫描源目录并计算 `checksum`
2. 对比 manifest（已入库文档清单）
3. 对新增/变更文档执行：重切分 -> 重嵌入 -> upsert
4. 对已删除文档执行：按 `doc_id` 删除对应向量
5. 更新 manifest 和索引版本

建议新增文件：

- `data_base/index_manifest.json`
- `data_base/index_versions/<version>.json`

## 4.3 生成集合（Generation Pipeline）

### 4.3.1 LLM 初始化与多模型支持

统一 `LLMProvider` 抽象：

- `generate(messages, **kwargs)`
- `stream(messages, **kwargs)`

配置可切换：

- `fast_model`：低延迟模型，用于重写、路由
- `quality_model`：高质量模型，用于最终回答

### 4.3.2 查询优化重写

分级重写：

- Level 0：无历史对话，直接检索
- Level 1：有历史对话，进行指代消解
- Level 2：复杂问题拆分（可选）

### 4.3.3 基础路由

路由决策：

- `chat`（闲聊）
- `rag`（知识库检索问答）
- `fallback`（无上下文时答复策略）

可先使用规则路由，后续替换为轻量 LLM 路由器。

### 4.3.4 基础回答与流式回答

支持两种输出模式：

- 同步回答：一次性返回
- 流式回答：token 级流式返回

流式回答同时记录：

- 首 token 延迟（TTFT）
- 完整响应时延

### 4.3.5 上下文构建

拼装策略：

1. 子块召回
2. 回溯父块
3. 去重与预算裁剪（token budget）
4. 注入引用标识（`[source:xxx]`）

### 4.3.6 Prompt 构建

Prompt 模板至少分 3 段：

- 系统约束（不编造、引用来源、回答风格）
- 上下文片段（带来源）
- 用户问题

建议 Prompt 参数化管理：

- `prompts/system_*.md`
- `prompts/rag_*.md`

## 4.4 检索优化（Retrieval）

### 4.4.1 向量检索器

默认 `TopK=20`，支持按元数据过滤：

- `file_type`
- `source_path`
- `tags`
- `time_range`

### 4.4.2 BM25 检索器

建立稀疏倒排索引（本地可用 Whoosh/BM25 实现）。  
适合关键词精确匹配（命名实体、术语、代码名）。

### 4.4.3 混合检索

并行执行：

- Dense Retriever（向量）
- Sparse Retriever（BM25）

合并候选后交由 RRF 重排。

### 4.4.4 RRF 重排

公式：

```text
RRF(d) = Σ 1 / (k + rank_i(d))
```

建议：

- `k = 60`
- Dense 与 Sparse 候选各取 `TopN=30`
- 合并后取最终 `TopM=10`

### 4.4.5 元数据过滤

过滤在检索前执行（缩小候选空间），在重排后再次校验（防止污染）：

- include filters
- exclude filters
- 权限过滤（后续若引入多用户）

## 5. 工程化与体验优化

## 5.1 性能目标（可观测）

建议线上目标（单机）：

- P50 首 token < 1.2s
- P95 首 token < 3.0s
- P95 全响应 < 8.0s
- 检索耗时 < 400ms（无冷启动）

## 5.2 日志与调试

采用结构化日志（JSON）并提供 `trace_id`：

- 请求级：query、route、latency、token usage
- 检索级：召回条数、重排前后排名、过滤命中
- 生成级：模型、prompt 长度、stream 统计
- 错误级：异常栈 + 上下文快照

新增调试模式（`DEBUG_RAG=true`）：

- 展示 query rewrite 结果
- 展示召回片段与得分
- 展示最终 prompt（脱敏）

## 5.3 配置中心

统一配置来源优先级：

1. 环境变量
2. 默认配置文件 `core/config/default.yaml`

关键配置：

- LLM 模型与路由策略
- Embedding 模型
- 检索参数（TopK、RRF k）
- Chunk 参数
- 日志级别

## 5.4 安全与合规

- 禁止硬编码 API Key（移除源码明文密钥）
- 日志脱敏（密钥、用户隐私片段）
- 错误提示对用户友好，对开发者保留完整日志

## 6. 部署方案（建议）

阶段一（快速上线）：

- FastAPI + 本地 Qdrant + 单进程服务

阶段二（可扩展）：

- FastAPI Web 前端 + FastAPI RAG 服务
- 异步索引 Worker（增量构建任务）
- 向量库独立目录挂载

建议补充：

- `Dockerfile`
- `docker-compose.yml`
- `Makefile`（`build-index` / `run-app` / `run-api`）

## 7. 测试与评估

## 7.1 自动化测试

- 单元测试：文档切分、元数据、路由、过滤
- 集成测试：端到端检索 + 生成
- 回归测试：固定 query 集合，比较命中率与答案质量

## 7.2 评估指标

- 检索：Recall@K、MRR、nDCG
- 生成：答案正确性、引用覆盖率、幻觉率
- 体验：TTFT、完整时延、失败率

建议维护评测集：

- `tests/eval/eval_queries.jsonl`
- `tests/eval/golden_answers.jsonl`

## 8. 分阶段实施计划

### Phase 1：数据与索引基础重构（优先级 P0）

- 建立统一文档模型与 manifest
- 实现父子分块与元数据标准
- 实现增量更新（新增/修改/删除）

交付标准：

- 索引可重复构建
- 增量更新可用
- 无重复入库

### Phase 2：检索优化（优先级 P0）

- 接入 BM25
- 完成混合检索 + RRF
- 支持元数据过滤

交付标准：

- 相比仅向量检索，Recall@10 提升可量化

### Phase 3：生成链路升级（优先级 P1）

- LLM Provider 抽象
- 查询重写分级
- 路由与上下文构建标准化
- Prompt 模板化

交付标准：

- 可切换模型
- 多轮问答稳定
- 引用来源可追踪

### Phase 4：工程化与上线（优先级 P1）

- 结构化日志 + 调试面板
- Docker 化部署
- 基础监控指标

交付标准：

- 可部署、可观测、可排障

## 9. 验收清单（上线前）

- 支持全量构建 + 增量更新
- 支持向量检索 + BM25 + 混合检索 + RRF
- 支持多 LLM 与流式输出
- 支持元数据过滤与来源引用
- 日志完整、可定位请求链路
- 无明文密钥
- 提供部署脚本与运行文档

## 10. 下一步落地建议

按以下顺序推进代码改造：

1. 先重构 `scripts/build_kb.py` 为可复用的 `core/ingestion` + `core/indexing`
2. 新建 `core/retrieval` 实现 dense/sparse/hybrid/rrf
3. 新建 `core/generation` 实现模型抽象、重写、Prompt Builder
4. 最后接入 `webapp/main.py`，只保留 UI 与调用编排

以上顺序可以最小化回归风险，并确保每个阶段都能独立验证收益。
