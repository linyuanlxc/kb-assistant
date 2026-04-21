<p align="center">
  <h1 align="center">LabKB</h1>
  <p align="center">
    <strong>实验室多模态 RAG 知识库系统</strong>
  </p>
  <p align="center">
    <a href="#核心特性">特性</a> · <a href="#快速开始">快速开始</a> · <a href="#系统架构">架构</a> · <a href="#api-接口">接口</a> · <a href="#评估系统">评估</a> · <a href="#配置说明">配置</a>
  </p>
</p>

---

## 项目背景

实验室内部长期积累了大量学术论文、课题申报书、竞赛材料、项目文档及往届毕业生遗留资料。这些资源分散在不同成员的电脑、网盘和群聊记录中，格式杂乱（PDF、Word、Markdown、图片），导致手工检索和查找效率极低，新成员难以快速了解已有工作，重复劳动时有发生。

为此，我们构建了 **LabKB** —— 一个面向实验室约十位成员的知识库系统，实现资源的快速查找、精准检索以及内容自动总结。

---

**LabKB** 基于 RAG（检索增强生成）架构构建，整合了 Dense 向量检索、BM25 关键词检索、Neo4j 知识图谱检索和 CLIP 多模态图像检索四路召回，支持文本与图片统一入库、SSE 流式回答、多轮对话查询改写、调试信息面板，并集成了基于 RAGAS 框架的完整性能评估系统。

技术栈：`FastAPI` + `Qdrant` + `Neo4j` + `LangChain` + `智谱AI GLM-4`

## 核心特性

### 检索引擎
- **Dense 向量检索** — 基于 Qdrant 的语义向量搜索，支持 CLIP 本地模型 / 智谱 API / 千问 API 等多种 Embedding 方案
- **BM25 稀疏检索** — 基于 `rank_bm25` 的精确关键词匹配，持久化 JSON 索引
- **知识图谱检索** — 基于 Neo4j 自动构建实体-关系图谱（jieba 实体提取 + 共现加权）
- **多模态图像检索** — CLIP ViT-B-32 共享向量空间，支持以文搜图和以图搜图
- **两阶段融合排序** — 加权合并（业务优先） + RRF 名次融合（公平补偿）

### 生成引擎
- **SSE 流式输出** — 基于 Server-Sent Events 协议的逐 token 实时推送
- **查询改写** — 多轮对话指代消解，使用快速模型低温度改写
- **父子分块** — 子块（380 字符）精准检索，父块（1500 字符）完整上下文注入
- **来源溯源** — 每段上下文附带 `[source:path]` 标记，答案可验证

### 工程能力
- **增量索引** — 基于 SHA-256 校验和的变更检测，仅处理新增/修改文档
- **优雅降级** — Neo4j 或 CLIP 不可用时自动退化，不影响其他通道
- **结构化日志** — JSON 格式日志，支持 trace ID 追踪，按天滚动保留 14 天
- **健康检查** — `/api/health` 实时监控 Qdrant / Neo4j 连通性
- **评估框架** — 内置 RAGAS 评估，覆盖 10+ 指标，支持 5 个公开基准数据集

## 快速开始

### 前置条件

- Python 3.10+
- Docker（用于启动 Qdrant 和 Neo4j）
- 智谱 AI API Key（或其他 OpenAI 兼容的 LLM 服务）

### 1. 启动依赖服务

```bash
# Qdrant 向量数据库
docker run -d --name qdrant -p 6333:6333 -v qdrant_storage:/qdrant/storage qdrant/qdrant

# Neo4j 图数据库
docker run -d --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/neo4j \
  -v neo4j_data:/data \
  neo4j:latest
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置环境变量

```bash
# 必填
export ZHIPUAI_API_KEY="your_api_key_here"

# 可选（以下为默认值）
export QDRANT_URL="http://localhost:6333"
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="neo4j"
export DEBUG_RAG="false"
```

> 国内用户可设置 `HF_ENDPOINT=https://hf-mirror.com` 加速 HuggingFace 模型下载。

### 4. 构建索引

将知识文档（Markdown、PDF、Word、图片等）放入 `kb_source/` 目录后执行：

```bash
# 全量构建
python scripts/build_kb.py --full

# 增量更新（仅处理新增/变更文件）
python scripts/build_kb.py
```

### 5. 启动服务

```bash
uvicorn webapp.main:app --host 0.0.0.0 --port 8000 --reload
```

浏览器打开 `http://localhost:8000` 即可使用。

## 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                      Web 层 (webapp/)                        │
│   FastAPI · SSE 流式接口 · 文件上传 · 静态资源               │
├─────────────────────────────────────────────────────────────┤
│                 编排层 (core/orchestration/)                   │
│   RAGPipeline 端到端流水线 · RetrievalOrchestrator 四路召回   │
├──────────┬──────────┬──────────┬──────────┬─────────────────┤
│ 检索层   │ 索引层   │ 生成层   │ 提供层   │ 评估层           │
│ 融合排序  │ 向量存储  │ 提示词   │ 模型注册  │ RAGAS 评估器     │
│ 图谱检索  │ BM25索引 │ 查询改写 │ Chat封装  │ 数据集加载       │
│ 重排序   │ Manifest │ 上下文   │ Embed封装 │ 报告生成        │
├──────────┴──────────┴──────────┴──────────┴─────────────────┤
│                  数据摄入层 (core/ingestion/)                   │
│   多格式加载器 · 父子分块策略 · Checksum 变更检测             │
├─────────────────────────────────────────────────────────────┤
│              基础设施层 (core/config + observability)          │
│   YAML 配置管理 · JSON 结构化日志 · Trace 追踪链路            │
└─────────────────────────────────────────────────────────────┘
```

### 索引构建流程

```
kb_source/*.md,*.pdf,*.docx,*.jpg
    │
    ├─► 文本文件 ─► 加载器 ─► RawDocument(text)
    │                              │
    │                        SHA-256 校验和
    │                              │
    │                   增量过滤（与 manifest 比对）
    │                              │
    │                  分块器: 父块(1500) + 子块(380)
    │                      │                    │
    │              Qdrant kb_parent_docs   Qdrant kb_text_chunks
    │                      │
    │                  BM25 索引构建
    │                      │
    │                  Neo4j 图谱更新
    │
    └─► 图片文件 ─► RawDocument(image)
                          │
                  CLIP 编码 ─► Qdrant kb_image_assets
```

### 在线问答流程

```
用户查询 + 对话历史 + [图片]
    │
    ├─① 查询改写（fast_model，有历史时执行）
    ├─② 四路并行召回（各召回 3x 超额候选）
    │     ├─ Dense 向量检索   (Qdrant kb_text_chunks)
    │     ├─ BM25 关键词检索   (rank_bm25)
    │     ├─ Graph 图谱检索   (Neo4j)
    │     └─ CLIP 图像检索     (上传图片 / 以文搜图)
    ├─③ 两阶段融合: 加权合并 + RRF 补偿
    ├─④ 上下文组装（top-k 结果, 上限 9000 字符, 带 source 标签）
    └─⑤ 流式答案生成（quality_model, temperature=0.3）
```

## 项目结构

```
lab-kb/
├── webapp/                       # FastAPI Web 服务层
│   ├── main.py                   #   路由定义、SSE 流式接口、文件上传
│   ├── bootstrap.py              #   依赖注入、Pipeline 进程级单例
│   ├── templates/                #   Jinja2 HTML 模板
│   └── static/                   #   CSS / JS 静态资源
├── core/                         # 核心业务逻辑
│   ├── types.py                  #   全局数据契约（dataclass 定义）
│   ├── config/
│   │   ├── settings.py           #   AppSettings 配置加载器（环境变量 + YAML）
│   │   ├── default.yaml          #   默认配置覆盖
│   │   └── model_registry.yaml   #   模型注册表（LLM / Embedding 配置）
│   ├── ingestion/
│   │   ├── loaders.py            #   多格式文档加载器
│   │   └── splitter.py           #   父子文本分块器
│   ├── indexing/
│   │   ├── builder.py            #   IndexBuilder 索引构建流水线
│   │   ├── vector_store.py       #   Qdrant 向量存储适配器
│   │   ├── bm25_index.py         #   BM25 稀疏索引
│   │   └── manifest.py           #   索引清单管理（增量更新）
│   ├── retrieval/
│   │   ├── fusion.py             #   加权合并 + RRF 融合排序
│   │   ├── graph_retriever.py    #   Neo4j 知识图谱引擎
│   │   └── reranker.py           #   Cross-encoder 重排序
│   ├── orchestration/
│   │   └── pipeline.py           #   RAGPipeline + RetrievalOrchestrator
│   ├── generation/
│   │   └── prompting.py          #   提示词模板（改写 / 回答 / 上下文）
│   ├── providers/
│   │   └── model_provider.py     #   ChatProvider + EmbeddingProvider + ModelRegistry
│   ├── evaluation/               #   性能评估子系统
│   │   ├── evaluator.py          #   RAGAS 评估器封装
│   │   ├── dataset_loader.py     #   公开基准数据集加载器
│   │   ├── eval_ingestor.py      #   评估语料库导入器
│   │   ├── recorder.py           #   运行时 QA 数据采集器
│   │   └── report_generator.py   #   评估报告生成器
│   └── observability/
│       └── logging_utils.py      #   JSON 结构化日志 + Trace 追踪
├── scripts/
│   ├── build_kb.py               #   索引构建命令行入口
│   └── eval/                     #   评估脚本集
│       ├── run_benchmark.py      #   标准化 6 步评估流程
│       ├── quick_eval.py         #   预设快捷评估
│       ├── run_batch_eval.py     #   离线批量评估
│       ├── download_dataset.py   #   下载公开数据集
│       ├── generate_csv_report.py
│       └── test_installation.py
├── configs/
│   └── eval_config.yaml          #   评估系统配置
├── kb_source/                    #   知识源文档目录
├── data_base/                    #   索引持久化数据
├── runtime/                      #   运行时目录
│   ├── uploads/                  #   会话上传图片缓存
│   ├── model_cache/              #   HuggingFace 本地模型缓存
│   ├── logs/                     #   应用日志
│   └── eval/                     #   评估输出
├── docs/                         #   开发文档
├── requirements.txt
└── README.md
```

## API 接口

### `GET /`
渲染 Web UI 页面。

### `GET /api/health`
返回各依赖服务的健康状态。

```json
{
  "ready": true,
  "qdrant": true,
  "neo4j": true,
  "errors": {}
}
```

### `POST /api/chat/stream`
SSE 流式对话接口。

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `query` | Form(str) | 是 | 用户问题 |
| `mode` | Form(str) | 否 | 检索模式：`hybrid` / `text_only` / `multimodal` / `graph_first` |
| `top_k` | Form(int) | 否 | 检索结果数量（默认 10） |
| `debug` | Form(str) | 否 | 开启调试面板（`true` / `false`） |
| `chat_history` | Form(str) | 否 | JSON 数组，格式 `[[role, content], ...]` |
| `image` | File | 否 | 图片文件，用于多模态检索 |

**SSE 事件格式：**

```
event: meta
data: {"rewritten_query": "改写后的查询", "uploaded_images": [...]}

event: token
data: {"delta": "你"}

event: token
data: {"delta": "好"}

event: done
data: {"answer": "完整答案...", "retrieval": {...}}
```

### `GET /files?path=...`
安全的文件透传接口，仅允许访问项目目录内的文件。

## 评估系统

LabKB 集成了基于 [RAGAS](https://github.com/explodinggradients/ragas) 0.2.10 的完整评估系统，支持使用公开基准数据集进行标准化评估。

### 支持的数据集

| 数据集 | 语言 | 来源 | 样本量 |
|--------|------|------|--------|
| CRUD-RAG | 中文 | AndrewTsai0406/CRUD_RAG_3QA | 3,199 |
| SuperCLUE-C3 | 中文 | TigerResearch/tigerbot-superclue-c3-zh-5k | 4,792 |
| HotpotQA | 英文 | hotpot_qa (fullwiki) | 大规模 |
| SQuAD v2 | 英文 | rajpurkar/squad_v2 | 大规模 |
| Amnesty QA | 英文 | explodinggradients/amnesty_qa | 20 |

### 评估指标

| 阶段 | 指标 | 说明 | 优秀阈值 |
|------|------|------|---------|
| 检索 | `recall@k` | 上下文召回率 | > 0.7 |
| 检索 | `precision@k` | 上下文精度 | > 0.7 |
| 检索 | `mrr@k` | 平均倒数排名 | > 0.7 |
| 检索 | `entity_recall@k` | 实体召回率 | — |
| 生成 | `faithfulness` | 忠实度（幻觉检测） | > 0.8 |
| 生成 | `answer_relevancy` | 答案相关性 | > 0.7 |
| 端到端 | `answer_correctness` | 答案正确性 | > 0.7 |
| 端到端 | `answer_similarity` | 答案语义相似度 | > 0.7 |

### 使用方法

```bash
# 中文 CRUD-RAG 评估（100 条样本）
python scripts/eval/run_benchmark.py \
  --dataset crud_rag \
  --subset-size 100 \
  --output runtime/eval/crud_rag_eval.json

# 英文 HotpotQA 评估（50 条样本）
python scripts/eval/run_benchmark.py \
  --dataset hotpotqa \
  --subset-size 50 \
  --language en

# 使用配置文件
python scripts/eval/run_benchmark.py \
  --config configs/eval_config.yaml

# 快捷预设
python scripts/eval/quick_eval.py --zh          # CRUD-RAG, 100 条
python scripts/eval/quick_eval.py --en          # HotpotQA, 50 条
python scripts/eval/quick_eval.py --zh-small    # CRUD-RAG, 30 条（快速验证）
```

### 常用参数

| 参数 | 说明 |
|------|------|
| `--skip-ingest` | 跳过语料库导入，直接使用 golden_docs 评估 |
| `--ingest-only` | 仅导入语料库，不执行评估 |
| `--cleanup` | 清理评估用 Qdrant 集合 |
| `--batch-size N` | RAGAS 批处理大小（默认 5） |
| `--output PATH` | 报告输出路径 |

### 结果解读

评估完成后，报告包含：

- **整体摘要** — 样本数量、成功率、平均延迟
- **指标统计** — 各指标的均值、标准差、最小值、最大值
- **样本详情** — 每个样本的问题、答案、标准答案、各项分数
- **异常样本** — 低于阈值（默认 0.5）的样本自动标注并分析原因
- **性能统计** — 检索延迟、生成延迟分布

重点关注的三个指标：`faithfulness`（> 0.8，检测幻觉）、`recall@k`（> 0.7，检索覆盖）、`answer_correctness`（> 0.7，答案质量）。

## 配置说明

### 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `ZHIPUAI_API_KEY` | — | **必填。** 智谱 AI API Key |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant 服务地址 |
| `QDRANT_API_KEY` | `""` | Qdrant API Key（可选） |
| `NEO4J_URI` | `bolt://localhost:7687` | Neo4j 连接地址 |
| `NEO4J_USER` | `neo4j` | Neo4j 用户名 |
| `NEO4J_PASSWORD` | `neo4j` | Neo4j 密码 |
| `DEBUG_RAG` | `false` | 开启检索调试信息 |
| `RERANK_ENABLED` | `true` | 启用 Cross-encoder 重排序 |
| `HF_ENDPOINT` | `https://hf-mirror.com` | HuggingFace 镜像（国内加速） |
| `HF_HUB_OFFLINE` | `false` | 离线模式（使用本地缓存） |
| `KB_MODEL_CACHE` | `runtime/model_cache` | 本地模型缓存目录 |
| `KB_LOG_FILE` | `runtime/logs/app.log` | 日志文件路径 |

### 检索权重

默认融合权重（可在 `core/config/default.yaml` 中修改）：

```yaml
retrieval_weights:
  text_dense: 0.35    # Dense 向量检索
  bm25: 0.20          # BM25 关键词检索
  graph: 0.25         # 知识图谱检索
  image_clip: 0.20    # 多模态图像检索
```

无图片输入时，`image_clip` 权重自动归零，其余三路权重按比例重新归一化。

### 检索模式

| 模式 | 说明 |
|------|------|
| `hybrid`（默认） | 四路检索全部启用 |
| `text_only` | 仅向量 + BM25 |
| `multimodal` | 启用图片引导的跨模态检索 |
| `graph_first` | 优先使用图谱证据 |

### 模型注册表

LLM 和 Embedding 模型在 `core/config/model_registry.yaml` 中配置，支持热切换（修改后重启即可）。

| 配置项 | 默认模型 | 用途 |
|--------|---------|------|
| `fast_model` | glm-4-flash | 查询改写、评估 |
| `quality_model` | glm-4-flash | 答案生成 |
| `vision_model` | glm-4v-flash | 多模态理解 |
| `text`（Embedding） | CLIP ViT-B-32-multilingual-v1 | 文本向量化（本地，512 维） |
| `image`（Embedding） | CLIP ViT-B-32-multilingual-v1 | 图片向量化（本地，512 维） |

> **注意：** 更换 Embedding 模型（尤其是维度变更）后必须全量重建索引：`python scripts/build_kb.py --full`

## 支持的文件格式

| 扩展名 | 类型 | 加载方式 |
|--------|------|---------|
| `.md`, `.txt`, `.json`, `.srt`, `.vtt`, `.tsv`, `.html` | 文本 | `TextLoader`（UTF-8 编码） |
| `.pdf` | 文本 | `PyPDFLoader`（逐页加载） |
| `.doc`, `.docx` | 文本 | `Docx2txtLoader` |
| `.jpg`, `.jpeg`, `.png`, `.webp` | 图片 | 仅记录元信息（索引阶段通过 CLIP 编码） |

## 优雅降级

系统设计了多层降级策略，任何单一组件故障不影响整体运行：

| 场景 | 行为 |
|------|------|
| Neo4j 不可用 | 自动跳过图谱检索，其余通道正常工作 |
| CLIP 不可用 | 返回零向量，图像检索通道自动禁用 |
| 无图片输入 | `image_clip` 权重归零，剩余权重重新归一化 |
| Qdrant 批量写入失败 | 自动将 batch_size 减半重试，直到最小为 1 条 |

## 技术栈

| 层级 | 技术选型 |
|------|---------|
| Web 框架 | FastAPI + Jinja2 模板 |
| 向量数据库 | Qdrant（HNSW 索引，余弦距离） |
| 图数据库 | Neo4j |
| 大语言模型 | 智谱 AI GLM-4（OpenAI 兼容协议） |
| 文本向量化 | CLIP ViT-B-32-multilingual-v1（本地）/ 智谱 embedding-3（API） |
| 图片向量化 | CLIP ViT-B-32-multilingual-v1（本地） |
| 稀疏检索 | rank-bm25 (BM25Okapi) |
| 中文分词 | jieba |
| 评估框架 | RAGAS 0.2.10 + datasets |
| 文档处理 | LangChain |

## 文档

- [项目说明书](docs/PROJECT_SPECIFICATION.md) — 模块级详细设计文档
- [实施与运行指南](docs/V2_IMPLEMENTATION_GUIDE.md) — 环境搭建与操作说明
- [升级设计文档](docs/PROJECT_UPGRADE_SPEC.md) — 架构升级方案
- [面试八股问答](docs/INTERVIEW_QA.md) — 系统设计面试准备

## 许可证

本项目仅供实验室内部使用与学术研究参考。
