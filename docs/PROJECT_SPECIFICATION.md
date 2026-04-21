# LabKB 项目说明书

## 一、项目概述

### 1.1 项目背景

LabKB 是一个可部署的**实验室知识库智能问答系统**，基于 RAG（Retrieval-Augmented Generation，检索增强生成）架构构建。该系统通过整合多种检索手段（向量检索、BM25 关键词检索、知识图谱检索、多模态图像检索），为实验室成员提供精准、可靠的知识问答服务。系统支持文本与图片的统一入库、SSE 流式回答、图片随问随传、调试信息面板等高级功能，并集成了基于 RAGAS 框架的完整性能评估系统。

### 1.2 核心目标

- **统一入库**：文本与图片统一管理，支持多种文件格式（Markdown、PDF、Word、纯文本、图片）
- **混合检索**：Dense 向量 + BM25 关键词 + 知识图谱 + 多模态图像四路召回
- **流式交互**：基于 SSE（Server-Sent Events）的实时流式回答
- **可观测性**：结构化 JSON 日志、检索调试面板、健康检查接口
- **性能评估**：基于 RAGAS 框架的标准化评估，支持公开测试集与自动标注

### 1.3 技术栈概览

| 层级 | 技术选型 |
|------|---------|
| Web 框架 | FastAPI + Jinja2 模板 |
| 向量数据库 | Qdrant |
| 图数据库 | Neo4j |
| LLM 服务 | 智谱 AI GLM-4（OpenAI 兼容协议）/ 阿里云千问（备选） |
| Embedding | CLIP ViT-B-32 多语言模型（本地）/ 智谱 embedding-3（API 备选） |
| 稀疏检索 | rank-bm25 (BM25Okapi) |
| 评估框架 | RAGAS 0.2.10 + datasets |
| 文档处理 | LangChain + unstructured + PyMuPDF |
| 文本分词 | jieba |


## 二、整体架构设计

### 2.1 系统架构总览

系统采用**分层模块化架构**，各层职责清晰、解耦良好：

```
┌─────────────────────────────────────────────────────────────┐
│                      Web 层 (webapp/)                        │
│   FastAPI 应用入口 · SSE 流式接口 · 文件上传 · 静态资源      │
├─────────────────────────────────────────────────────────────┤
│                   编排层 (core/orchestration/)                │
│   RAGPipeline 端到端流水线 · RetrievalOrchestrator 多路召回   │
├──────────┬──────────┬──────────┬──────────┬─────────────────┤
│ 检索层   │ 索引层   │ 生成层   │ 提供层   │ 评估层           │
│(retrieval)│(indexing)│(generation)│(providers)│(evaluation) │
│ 融合排序  │ 向量存储  │ 提示词   │ 模型注册  │ RAGAS评估器     │
│ 图谱检索  │ BM25索引 │ 查询改写 │ Chat封装  │ 数据集加载      │
│          │ Manifest │ 上下文   │ Embed封装 │ 报告生成        │
├──────────┴──────────┴──────────┴──────────┴─────────────────┤
│                   数据摄入层 (core/ingestion/)                │
│   多格式加载器 · 父子分块策略 · Checksum 变更检测            │
├─────────────────────────────────────────────────────────────┤
│              基础设施层 (core/config + observability)         │
│   配置管理 · 结构化日志 · 追踪链路                            │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 目录结构

```
kb-assistant/
├── webapp/                    # FastAPI Web 服务层
│   ├── main.py               # 应用入口：路由定义、SSE 流式接口
│   ├── bootstrap.py          # 启动引导：依赖注入、Pipeline 初始化
│   ├── templates/            # Jinja2 HTML 模板
│   └── static/               # CSS / JS 静态资源
├── core/                     # 核心业务逻辑
│   ├── types.py              # 全局数据契约（dataclass 定义）
│   ├── config/               # 配置管理
│   │   ├── settings.py       # AppSettings 配置加载器
│   │   ├── default.yaml      # 默认配置覆盖
│   │   └── model_registry.yaml  # 模型注册表（LLM/Embedding 配置）
│   ├── ingestion/            # 数据摄入
│   │   ├── loaders.py        # 多格式文档加载器
│   │   └── splitter.py       # 父子分块策略
│   ├── indexing/             # 索引构建
│   │   ├── builder.py        # IndexBuilder 索引构建流水线
│   │   ├── vector_store.py   # Qdrant 向量存储适配器
│   │   ├── bm25_index.py     # BM25 稀疏索引封装
│   │   └── manifest.py       # 增量索引清单管理
│   ├── retrieval/            # 检索引擎
│   │   ├── fusion.py         # 混合检索融合工具（RRF + 加权合并）
│   │   └── graph_retriever.py # Neo4j 图谱检索引擎
│   ├── orchestration/        # RAG 编排
│   │   └── pipeline.py       # RAGPipeline + RetrievalOrchestrator
│   ├── generation/           # 生成模块
│   │   └── prompting.py      # 提示词构造器（改写/回答/上下文）
│   ├── providers/            # 模型提供层
│   │   └── model_provider.py # ChatProvider + EmbeddingProvider + ModelRegistry
│   ├── evaluation/           # 性能评估系统
│   │   ├── evaluator.py      # RagasEvaluator（RAGAS 框架封装）
│   │   ├── dataset_loader.py # 公开测试集加载器
│   │   ├── eval_ingestor.py  # 评估语料库导入器
│   │   ├── recorder.py       # 运行时 QA 数据采集器
│   │   └── report_generator.py # 评估报告生成器
│   └── observability/        # 可观测性
│       └── logging_utils.py  # JSON 结构化日志 + trace 追踪
├── scripts/                  # 脚本工具
│   ├── build_kb.py           # 索引构建命令行入口
│   └── eval/                 # 评估脚本集
│       ├── run_benchmark.py  # 标准化 6 步评估流程
│       ├── test_installation.py
│       ├── quick_eval.py
│       ├── run_batch_eval.py
│       ├── download_dataset.py
│       └── generate_csv_report.py
├── configs/                  # 配置文件
│   └── eval_config.yaml      # 评估系统完整配置
├── kb_source/                # 知识源目录（Markdown/PDF/Word/图片）
├── data_base/                # 索引持久化数据
├── runtime/                  # 运行时目录
│   ├── uploads/              # 会话上传图片缓存
│   ├── model_cache/          # 本地模型缓存
│   ├── logs/                 # 日志文件
│   └── eval/                 # 评估输出
└── docs/                     # 开发文档
```

---

## 三、核心模块详细说明

### 3.1 数据契约层 — `core/types.py`

该模块是整个系统的**数据骨架**，定义了所有跨模块传递的核心数据结构。所有数据类基于 Python `@dataclass` 实现，避免跨模块的隐式字段依赖，提升可维护性和可测试性。

#### 3.1.1 检索模式枚举 — `SearchMode`

```python
class SearchMode(str, Enum):
    TEXT_ONLY = "text_only"      # 仅文本：向量 + BM25
    MULTIMODAL = "multimodal"    # 多模态：启用图片引导检索
    GRAPH_FIRST = "graph_first"  # 图优先：优先使用图谱证据
    HYBRID = "hybrid"            # 混合：四路全开（默认）
```

继承 `str` 和 `Enum`，可直接与字符串比较，也支持 FastAPI 的表单参数自动解析。当用户传入无效值时，`pipeline.py` 中会静默降级为 `HYBRID`。

#### 3.1.2 统一原始文档 — `RawDocument`

```python
@dataclass
class RawDocument:
    doc_id: str              # 确定性 ID（uuid5 基于路径生成）
    source_path: str         # 文件绝对路径
    source_type: str         # 文件扩展名（.md / .pdf / .jpg 等）
    modality: str            # 模态类型："text" 或 "image"
    content: str             # 文本内容 / 图片文件名
    title: str | None        # 文档标题（取自文件名 stem）
    checksum: str            # SHA-256 校验和（用于增量比对）
    created_at: datetime | None
    updated_at: datetime | None
    extra_meta: dict[str, Any]  # 扩展元信息（如 file_name）
```

**设计意图**：文本与图片共用同一数据结构。文本的 `content` 是实际文本内容，图片的 `content` 是文件名字符串（图片本身在向量阶段以路径方式传入 CLIP 编码器）。

#### 3.1.3 文本子块记录 — `ChunkRecord`

```python
@dataclass
class ChunkRecord:
    doc_id: str              # 所属文档 ID
    parent_id: str           # 所属父块 ID
    chunk_id: str            # 子块唯一 ID
    chunk_text: str          # 子块文本内容
    chunk_index: int         # 在父块内的序号
    token_count: int         # Token 估算（字符数 / 4）
    metadata: dict[str, Any] # 附带 source_path、file_type 等
```

`chunk_id` 格式为 `{doc_id}:c:{parent_index}:{child_index}:{uuid8}`，`parent_id` 格式为 `{doc_id}:p:{parent_index}`，二者之间通过前缀建立父子关联。

#### 3.1.4 检索请求 / 结果契约 — `RetrieverRequest` / `RetrieverResult`

`RetrieverRequest` 是检索阶段的统一入口参数：

| 字段 | 类型 | 说明 |
|------|------|------|
| `query` | `str` | 用户查询文本（或改写后的查询） |
| `chat_history` | `list[tuple[str, str]]` | 对话历史 `(role, content)` |
| `filters` | `dict[str, Any]` | 元数据精确匹配过滤条件 |
| `modality` | `SearchMode` | 检索模式 |
| `image_inputs` | `list[str]` | 用户上传图片的本地路径列表 |
| `top_k` | `int` | 最终返回的候选数量（默认 10） |
| `debug` | `bool` | 是否输出调试信息 |

`RetrieverResult` 是检索阶段的统一返回值：

| 字段 | 类型 | 说明 |
|------|------|------|
| `items` | `list[RetrieverItem]` | 融合排序后的 top-k 候选 |
| `scores` | `dict[str, float]` | 每个 item_id 对应的最终融合分数 |
| `sources` | `list[str]` | 去重后的来源文件路径列表 |
| `graph_evidence` | `list[dict]` | 图谱检索的证据链接（entity → doc_id） |
| `latency_ms` | `float` | 检索阶段总耗时（毫秒） |
| `debug_info` | `dict` | 调试信息（各路 top-5、权重、RRF 分数） |

#### 3.1.5 索引清单 — `IndexManifest`

```python
@dataclass
class IndexManifest:
    docs: dict[str, dict[str, Any]]     # 文本文档索引 {doc_id: {checksum, source_path, updated_at}}
    assets: dict[str, dict[str, Any]]   # 图片资源索引 {doc_id: {checksum, source_path, updated_at}}
    versions: list[dict[str, Any]]      # 版本记录 [{time, note}]
    last_sync_at: str                   # 最后同步时间（ISO 格式）
```

用于增量索引构建：比对每个文档的 checksum，仅处理新增或变更文件。

---

### 3.2 配置管理层 — `core/config/`

配置管理采用**三级优先级**设计：`Python 默认值` → `default.yaml 覆盖` → `环境变量覆盖`。

#### 3.2.1 应用配置 — `settings.py`

**`AppSettings` 数据类**完整字段说明：

```python
@dataclass
class AppSettings:
    # ─── 路径配置 ───
    source_dir: Path          # 知识源目录，默认 kb_source/
    index_dir: Path           # 索引持久化目录，默认 data_base/
    runtime_dir: Path         # 运行时目录，默认 runtime/
    model_cache_dir: Path     # 本地模型缓存，默认 runtime/model_cache/
    log_dir: Path             # 日志目录，默认 runtime/logs/
    log_file: Path            # 日志文件，默认 runtime/logs/app.log

    # ─── Qdrant 向量数据库 ───
    qdrant_url: str           # 环境变量 QDRANT_URL，默认 http://localhost:6333
    qdrant_api_key: str       # 环境变量 QDRANT_API_KEY，默认空

    # ─── Neo4j 图数据库 ───
    neo4j_uri: str            # 环境变量 NEO4J_URI，默认 bolt://localhost:7687
    neo4j_user: str           # 环境变量 NEO4J_USER，默认 neo4j
    neo4j_password: str       # 环境变量 NEO4J_PASSWORD，默认 neo4j

    # ─── 模型注册 ───
    debug_rag: bool           # 环境变量 DEBUG_RAG，默认 false
    model_registry_path: Path # 模型注册表路径，默认 core/config/model_registry.yaml

    # ─── Qdrant 集合名称 ───
    text_collection: str      # 文本子块集合，默认 kb_text_chunks
    image_collection: str     # 图片资产集合，默认 kb_image_assets
    parent_collection: str    # 父文档集合，默认 kb_parent_docs

    # ─── 文件路径 ───
    bm25_index_file: Path     # BM25 索引文件，默认 data_base/bm25_index.json
    manifest_path: Path       # 索引清单文件，默认 data_base/index_manifest.json

    # ─── 检索融合权重 ───
    retrieval_weights: dict   # 默认 text_dense=0.35, bm25=0.20, graph=0.25, image_clip=0.20
```

**`load_settings()` 加载流程**：
1. 创建 `AppSettings()` 实例（所有默认值）
2. 读取 `core/config/default.yaml`，按字段名逐个覆盖
3. 路径类字段（在 `PATH_KEYS` 集合中定义）通过 `_to_path()` 转换为 `Path` 对象
4. 确保运行时目录存在（`mkdir(parents=True, exist_ok=True)`）

**`default.yaml`** 仅用于覆盖路径和集合名称，不包含敏感信息：

```yaml
source_dir: kb_source
index_dir: data_base
runtime_dir: runtime
text_collection: kb_text_chunks
image_collection: kb_image_assets
parent_collection: kb_parent_docs
```

#### 3.2.2 模型注册表 — `model_registry.yaml`

该文件定义了系统使用的所有 LLM 和 Embedding 模型，支持**热切换**（修改后重启即可）。

**LLM 配置**（`providers` 节）：

| 配置键 | 用途 | 默认模型 | 特性 |
|--------|------|---------|------|
| `fast_model` | 查询改写、评估 | `glm-4-flash` | 流式、JSON 模式、32K 上下文 |
| `quality_model` | 答案生成 | `glm-4-flash` | 流式、JSON 模式、32K 上下文 |
| `vision_model` | 多模态理解 | `glm-4v-flash` | 流式、**支持图片输入**、32K 上下文 |

每个模型配置项包括：`model`（模型名）、`base_url`（API 地址）、`api_key_env`（API Key 环境变量名）、`api_protocol`（协议类型）、`supports_stream/vision/json_mode`（能力标志）、`max_context_tokens`（上下文窗口大小）。

**Embedding 配置**（`embeddings` 节）：

| 配置键 | 默认方案 | 维度 | 跨模态 |
|--------|---------|------|--------|
| `text` | CLIP ViT-B-32-multilingual-v1（本地） | 512 | 是 |
| `image` | CLIP ViT-B-32-multilingual-v1（本地） | 512 | — |

备选方案包括智谱 embedding-3（1024 维）和阿里 text-embedding-v3（1024 维），均为 API 调用方式，**不支持跨模态检索**。

> **切换注意事项**：如果更换了 Embedding 模型（特别是维度变更），必须全量重建索引：`python scripts/build_kb.py --full`。

---

### 3.3 数据摄入层 — `core/ingestion/`

数据摄入层负责将原始文件转换为系统可处理的结构化数据，是整个流水线的**入口**。

#### 3.3.1 文档加载器 — `loaders.py`

**核心函数**：`load_raw_documents(source_dir: Path) -> tuple[list[RawDocument], list[RawDocument]]`

递归扫描指定目录，按文件后缀分流处理，返回 `(text_docs, image_docs)`。

**文件格式与加载方式**：

| 后缀 | 类型 | 加载方式 | 说明 |
|------|------|---------|------|
| `.md`, `.markdown`, `.txt`, `.json`, `.srt`, `.vtt`, `.tsv`, `.html` | 文本 | `TextLoader` | LangChain 通用文本加载器，UTF-8 编码 |
| `.pdf` | 文本 | `PyPDFLoader` | 逐页加载，页间用 `\n\n` 连接 |
| `.doc`, `.docx` | 文本 | `Docx2txtLoader` | 逐页加载，页间用 `\n\n` 连接 |
| `.jpg`, `.jpeg`, `.png`, `.webp` | 图片 | — | 仅记录元信息，不读取内容 |

**关键实现细节**：

1. **确定性 ID 生成**：`_id_from_path(path)` 使用 `uuid5(NAMESPACE_URL, str(path.resolve()))`，同一文件路径始终产生相同 ID，与操作系统无关。

2. **校验和计算**：`sha256_file(path)` 以 8KB 为单位流式读取文件计算 SHA-256，适合大文件处理。校验和用于增量索引的变更检测。

3. **空内容过滤**：文本加载后检查 `content.strip()`，空内容文档被静默跳过。

4. **图片特殊处理**：图片的 `content` 字段存储文件名（如 `photo.jpg`），`modality` 设为 `"image"`，实际向量编码在索引阶段通过文件路径完成。

5. **遍历顺序**：使用 `sorted(source_dir.rglob("*"))` 保证文件处理顺序的确定性。

#### 3.3.2 父子分块器 — `splitter.py`

**核心函数**：`split_parent_child(docs, parent_chunk_size=1500, parent_overlap=120, child_chunk_size=380, child_overlap=70) -> tuple[list[ChunkRecord], dict[str, str]]`

采用**两级分块**策略，平衡检索精度和语义完整性：

```
原始文档 content
    │
    │  RecursiveCharacterTextSplitter(chunk_size=1500, overlap=120)
    │  分隔符优先级: "\n\n" > "\n" > "。" > "." > " " > ""
    │
    ├── 父块 0  →  parent_id = "{doc_id}:p:0"
    │   ├── 子块 0-0  →  chunk_id = "{doc_id}:c:0:0:{uuid8}"
    │   ├── 子块 0-1  →  chunk_id = "{doc_id}:c:0:1:{uuid8}"
    │   └── ...
    ├── 父块 1  →  parent_id = "{doc_id}:p:1"
    │   ├── 子块 1-0  →  chunk_id = "{doc_id}:c:1:0:{uuid8}"
    │   └── ...
    └── ...
```

**参数设计**：

| 参数 | 默认值 | 设计依据 |
|------|--------|---------|
| `parent_chunk_size` | 1500 | 约 375 个 token（按中文 4 字符/token），保证语义完整 |
| `parent_overlap` | 120 | 父块间重叠，防止语义截断 |
| `child_chunk_size` | 380 | 约 95 个 token，适合向量检索粒度 |
| `child_overlap` | 70 | 子块间重叠，保证关键词不丢失 |

**元数据传递**：每个 `ChunkRecord.metadata` 包含 `source_path`、`file_type`、`file_name`、`checksum`、`title`、`modality`，这些信息会随向量一起存入 Qdrant payload，检索后用于来源溯源。

**返回值**：`parent_store: dict[str, str]` 是 `{parent_id: parent_text}` 映射，供向量存储和图谱索引使用。

---

### 3.4 索引构建层 — `core/indexing/`

索引构建层负责将摄入的数据持久化为可检索的索引结构，包括向量索引、BM25 索引和图谱索引。

#### 3.4.1 索引构建器 — `builder.py` — `IndexBuilder`

**类初始化**（`__init__`）创建完整的依赖链：

```
IndexBuilder(settings)
    ├── ModelRegistry(settings.model_registry_path)  → 加载模型配置
    ├── EmbeddingProvider(text_cfg, image_cfg)        → 向量化能力
    ├── QdrantVectorStoreAdapter(...)                 → 向量存储
    ├── BM25Indexer(settings.bm25_index_file)         → BM25 索引
    └── LightRAGGraphEngine(...)                      → 图谱引擎
```

**`build(incremental=True)` 方法**完整流程：

```
Step 1: load_manifest  → 加载已有索引清单（增量时需要比对）
Step 2: load_raw_documents  → 扫描知识源目录，返回 (text_docs, image_docs)
Step 3: _incremental_filter  → [增量模式] 与 manifest 比对 checksum，过滤未变更文档
Step 4: split_parent_child  → 父子分块，返回 (chunks, parent_map)
Step 5: embed_text + upsert_text_chunks  → 子块向量化并写入 Qdrant kb_text_chunks
Step 6: embed_text + upsert_parent_docs  → 父块向量化并写入 Qdrant kb_parent_docs
Step 7: bm25.build(chunks) + bm25.save()  → BM25 索引构建并持久化
Step 8: embed_image_paths + upsert_image_assets  → 图片向量化并写入 Qdrant kb_image_assets
Step 9: graph_engine.upsert_from_parent_docs(parent_map)  → [Neo4j 可用时] 更新图谱
Step 10: _update_manifest + register_version + save_manifest  → 更新清单和版本记录
```

**增量过滤逻辑**（`_incremental_filter`）：

```python
for doc in text_docs:
    prev = manifest.docs.get(doc.doc_id)
    if not prev or prev["checksum"] != doc.checksum:
        changed_text.append(doc)
```

只有新增文件（`not prev`）或内容变更文件（`checksum` 不一致）才会进入后续流程。图片资源同理，通过 `manifest.assets` 字典比对。

**步骤容错**：Neo4j 不可用时（`graph_engine.health()` 返回 `False`），Step 9 自动跳过，不影响其他索引的构建。

#### 3.4.2 向量存储适配器 — `vector_store.py` — `QdrantVectorStoreAdapter`

该类继承自抽象接口 `VectorStoreAdapter`，是向量存储的可插拔后端实现。

**集合管理**：

| 集合名 | 用途 | 向量维度 | payload 关键字段 |
|--------|------|---------|-----------------|
| `kb_text_chunks` | 文本子块 | text_dim（默认 512） | content, source, chunk_id, parent_id |
| `kb_parent_docs` | 父文档补充层 | text_dim | content, parent_id, source |
| `kb_image_assets` | 图片资产 | image_dim（默认 512） | content, source_path, caption, modality |

**ID 转换策略**：业务层使用字符串 ID（如 `{doc_id}:c:0:0:abc123`），Qdrant 要求 UUID 格式。`_to_qdrant_id(raw_id)` 使用 `uuid5(NAMESPACE_URL, raw_id)` 生成稳定 UUID，保证同一业务 ID 始终映射到同一 Qdrant 点。

**分批写入与自动降级**：

```python
def _upsert_in_batches(self, collection_name, points, batch_size=None):
    size = batch_size or self.upsert_batch_size  # 默认 64
    while idx < len(points):
        try:
            self.client.upsert(collection_name=collection_name, points=batch)
            idx += size
        except Exception:
            if size <= 1:
                raise         # 单条也失败则抛出
            size = max(1, size // 2)  # 批量大小减半重试
```

Qdrant 单请求有 32MB payload 限制。当写入失败时，自动将批量大小减半重试，直到单条写入。这保证了大批量数据写入的稳定性。

**搜索接口**（`search`）：

```python
def search(self, collection, vector, top_k, filters=None) -> list[RetrieverItem]:
    # filters 转换为 Qdrant Filter（精确匹配）
    if filters:
        conditions = [FieldCondition(key=k, match=MatchValue(value=v)) for k, v in filters.items()]
        query_filter = Filter(must=conditions)

    hits = self.client.search(collection_name=collection, query_vector=vector,
                              limit=top_k, query_filter=query_filter)

    # 从 payload 中提取逻辑 ID（优先级：chunk_id > parent_id > doc_id > Qdrant UUID）
    for hit in hits:
        logical_id = payload.get("chunk_id") or payload.get("parent_id") \
                     or payload.get("doc_id") or str(hit.id)
```

返回的 `RetrieverItem` 中 `content` 字段携带实际文本，`metadata` 携带完整 payload，便于后续上下文构建和来源溯源。

#### 3.4.3 BM25 索引 — `bm25_index.py` — `BM25Indexer`

基于 `rank_bm25.BM25Okapi` 的轻量级稀疏检索实现。

**数据结构**：

```python
class BM25Indexer:
    index_file: Path          # 持久化文件路径
    corpus: list[list[str]]   # 分词后的语料库（每个元素是一个文档的词列表）
    records: list[dict]       # 结构化记录列表 [{chunk_id, text, source, metadata}]
    bm25: BM25Okapi | None    # 运行时 BM25 模型
```

**构建流程**（`build`）：

```python
def build(self, chunks: list[ChunkRecord]):
    self.records = [{"chunk_id": c.chunk_id, "text": c.chunk_text,
                     "source": c.metadata.get("source_path", ""), ...} for c in chunks]
    self.corpus = [r["text"].lower().split() for r in self.records]  # 简单词级分词
    self.bm25 = BM25Okapi(self.corpus)
```

**搜索流程**（`search`）：

```python
def search(self, query: str, top_k: int = 10) -> list[RetrieverItem]:
    q = query.lower().split()        # 查询词分词
    scores = self.bm25.get_scores(q) # 计算 BM25 分数
    indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
    # 转换为 RetrieverItem 列表
```

**持久化策略**：`save()` 将 `records` 列表序列化为 JSON 文件（不保存 BM25 模型本身）；`load()` 从 JSON 反序列化后重建 BM25 模型。这种设计使得索引文件较小且可人工检查。

#### 3.4.4 索引清单 — `manifest.py`

提供三个工具函数：

| 函数 | 功能 |
|------|------|
| `load_manifest(path)` | 从 JSON 加载，不存在时返回空 `IndexManifest()` |
| `save_manifest(path, manifest)` | 序列化为 JSON，自动更新 `last_sync_at` 时间戳 |
| `register_version(manifest, note)` | 在 `versions` 列表追加 `{time, note}` 记录 |

Manifest 文件结构示例：
```json
{
  "docs": {
    "uuid-of-doc-1": {"checksum": "abc123...", "source_path": "kb_source/guide.md", "updated_at": "..."},
    "uuid-of-doc-2": {"checksum": "def456...", "source_path": "kb_source/api.pdf", "updated_at": "..."}
  },
  "assets": {
    "uuid-of-img-1": {"checksum": "789...", "source_path": "kb_source/photo.png", "updated_at": "..."}
  },
  "versions": [
    {"time": "2026-04-05T15:00:00", "note": "incremental=True, chunks=42, image_docs=5"}
  ],
  "last_sync_at": "2026-04-05T15:00:00"
}
```

---

### 3.5 模型提供层 — `core/providers/model_provider.py`

该层是对外模型服务的统一抽象层，封装了对话生成和向量化能力。

#### 3.5.1 模型缓存初始化 — `_prepare_local_model_cache()`

```python
def _prepare_local_model_cache() -> Path:
    cache_dir = Path(os.getenv("KB_MODEL_CACHE", str(DEFAULT_MODEL_CACHE)))
    cache_dir.mkdir(parents=True, exist_ok=True)
    # 统一三个 HuggingFace 生态的缓存目录
    os.environ.setdefault("HF_HOME", str(cache_dir / "hf_home"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_dir / "transformers"))
    os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(cache_dir / "sentence_transformers"))
```

将 HuggingFace、Transformers、SentenceTransformers 三者的缓存目录统一到 `runtime/model_cache/` 下，便于环境复现、迁移和清理。通过环境变量 `KB_MODEL_CACHE` 可自定义缓存位置。

#### 3.5.2 对话模型封装 — `ChatProvider`

```python
class ChatProvider:
    def __init__(self, model_cfg: dict):
        api_key = os.getenv(model_cfg["api_key_env"], "")
        self.client = OpenAI(api_key=api_key, base_url=model_cfg["base_url"], timeout=30)
        self.model = model_cfg["model"]

    def generate(self, messages, **kwargs) -> str:
        """同步生成，用于查询改写等非流式场景"""
        resp = self.client.chat.completions.create(model=self.model, messages=messages,
                                                   stream=False, **kwargs)
        return resp.choices[0].message.content or ""

    def stream(self, messages, **kwargs):
        """流式生成，用于答案输出"""
        resp = self.client.chat.completions.create(model=self.model, messages=messages,
                                                   stream=True, **kwargs)
        for chunk in resp:
            delta = chunk.choices[0].delta.content or ""
            if delta:
                yield delta
```

所有 LLM 均通过 OpenAI 兼容协议调用，`ChatProvider` 不关心底层是智谱、千问还是其他兼容服务。超时设置为 30 秒。

#### 3.5.3 向量化封装 — `EmbeddingProvider`

```python
class EmbeddingProvider:
    def __init__(self, text_cfg, image_cfg):
        self.text_provider = text_cfg.get("provider", "api")  # "clip" 或 "api"
        self.text_dimensions = text_cfg.get("dimensions", 1024)

        if self.text_provider == "clip":
            # CLIP 本地模式：SentenceTransformer，文本和图片共享向量空间
            self._text_model = SentenceTransformer(text_cfg.get("model", ""))
        else:
            # API 模式：OpenAI 兼容 Embedding API
            self.text_client = OpenAI(api_key=..., base_url=..., timeout=20)
            self.text_model = text_cfg.get("model", "")

        self._image_model = SentenceTransformer(image_cfg.get("model", ""))
```

**三种向量化方法**：

| 方法 | 用途 | CLIP 模式 | API 模式 |
|------|------|-----------|----------|
| `embed_text(texts: list[str])` | 批量文本向量化 | `SentenceTransformer.encode()` | `OpenAI.embeddings.create(batch_size=10)` |
| `embed_query(text: str)` | 单条查询向量化 | 调用 `embed_text([text])[0]` | 同左 |
| `embed_image_paths(paths: list[str])` | 图片路径向量化 | `SentenceTransformer.encode(paths)` | 返回零向量（降级） |

**降级策略**：
- CLIP 模型未安装（`SentenceTransformer is None`）→ `embed_text` 返回全零向量列表
- 图片模型未安装 → `embed_image_paths` 返回全零向量列表
- 图片搜索时全零向量不会匹配到有意义的图片，相当于自动禁用图像检索通道

**`text_provider` 属性**：外部可通过检查 `embedder.text_provider == "clip"` 判断是否启用跨模态检索能力。`RetrievalOrchestrator` 正是通过此属性决定是否发起"以文搜图"请求。

#### 3.5.4 模型注册中心 — `ModelRegistry`

```python
class ModelRegistry:
    def __init__(self, registry_path: Path):
        with registry_path.open("r", encoding="utf-8") as file:
            self._cfg = yaml.safe_load(file) or {}

    def build_chat(self, key: str) -> ChatProvider:
        """key: "fast_model" / "quality_model" / "vision_model" """
        return ChatProvider(self.providers[key])

    def build_embedding(self) -> EmbeddingProvider:
        """同时传入 text 和 image 配置"""
        return EmbeddingProvider(self.embeddings["text"], self.embeddings["image"])

    @staticmethod
    def image_bytes_to_data_url(image_bytes: bytes, mime: str = "image/jpeg") -> str:
        """将图片字节转换为 base64 data URL，供多模态 LLM 使用"""
```

`ModelRegistry` 在系统启动时创建，后续所有模块通过它获取模型实例，实现了模型配置的集中管理。

---

### 3.6 检索引擎层 — `core/retrieval/`

检索引擎层是系统的核心能力层，负责从多个数据源召回相关文档并进行融合排序。

#### 3.6.1 混合融合工具 — `fusion.py`

提供三个核心函数，实现**两阶段融合排序**策略：

**函数 1：权重归一化 — `normalize_weights`**

```python
def normalize_weights(weights: dict, use_image: bool) -> dict:
    w = dict(weights)
    if not use_image:
        w["image_clip"] = 0.0   # 无图片时禁用 image 分支
    total = sum(w.values())
    return {k: v / total for k, v in w.items()}  # 归一化到总和为 1
```

无图片输入时，`image_clip` 权重归零，其余三路权重按比例放大。例如默认权重 `{0.35, 0.20, 0.25, 0.20}` 在无图片时变为 `{0.4375, 0.25, 0.3125, 0}`。

**函数 2：加权合并 — `weighted_merge`**

```python
def weighted_merge(dense, bm25, graph, image, weights, top_k):
    id2item: dict[str, RetrieverItem] = {}    # item_id → RetrieverItem 映射
    merged_scores: dict[str, float] = {}       # item_id → 加权累计分数

    for i in dense:    merged_scores[i.item_id] += i.score * weights["text_dense"]
    for i in bm25:     merged_scores[i.item_id] += i.score * weights["bm25"]
    for i in graph:    merged_scores[i.item_id] += i.score * weights["graph"]
    for i in image:    merged_scores[i.item_id] += i.score * weights["image_clip"]

    ranked = sorted(merged_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return results, dict(ranked)
```

同一文档可能被多路检索命中，分数会**累加**。例如一个文档同时在 Dense 排名第 1（score=0.8）和 BM25 排名第 3（score=0.5），其加权分数为 `0.8×0.35 + 0.5×0.20 = 0.38`。

**函数 3：RRF 名次补偿 — `reciprocal_rank_fusion`**

```python
def reciprocal_rank_fusion(rank_lists, k=60) -> dict[str, float]:
    result = defaultdict(float)
    for items in rank_lists:
        for rank, item in enumerate(items, start=1):
            result[item.item_id] += 1.0 / (k + rank)
    return dict(result)
```

RRF 公式：`RRF_score(d) = Σ 1/(k + rank_i(d))`，其中 `k=60` 是平滑常数。

RRF 的核心价值：**名次补偿**。一个在 Dense 排名第 50、但在 BM25 排名第 1 的文档，即使加权分数不高，也能通过 RRF 获得额外补偿分数。这有效缓解了单一检索通道偏废的问题。

**融合流程的顺序**：先加权合并取 top-k，再叠加 RRF 分数，最后重新排序。两阶段互补——加权合并体现了业务优先级，RRF 保证了名次公平性。

#### 3.6.2 图谱检索引擎 — `graph_retriever.py` — `LightRAGGraphEngine`

基于 Neo4j 的知识图谱检索引擎，采用 LightRAG 风格的实体-文档关联图谱。

**图谱 Schema 设计**：

```
(:Doc {id, content}) -[:MENTIONS]-> (:Entity {name})
(:Entity {name}) -[:RELATED {weight}]-> (:Entity {name})
```

- `Doc` 节点：每个父文档一个节点，`content` 截取前 4000 字符
- `Entity` 节点：从文档中提取的关键词实体
- `MENTIONS` 关系：文档提到某实体
- `RELATED` 关系：实体间共现关系，`weight` 表示共现次数（每次 `+1`）

**实体提取**（`_extract_entities`）：

```python
@staticmethod
def _extract_entities(text: str, topk: int = 8) -> list[str]:
    return [w for w in jieba.analyse.extract_tags(text, topK=topk) if len(w) >= 2]
```

使用 jieba 的 TF-IDF 算法提取 top-k 关键词，过滤掉单字实体（长度 >= 2）。`topk=8` 在信息量和噪音之间取得平衡。

**图谱写入**（`upsert_from_parent_docs`）：

对每个父文档执行三步 Cypher 操作：

1. **创建/更新文档节点**：`MERGE (d:Doc {id:$pid}) SET d.content = $content`
2. **创建实体并建立 MENTIONS 关系**：
   ```cypher
   MERGE (n:Entity {name:$name})
   WITH n MATCH (d:Doc {id:$pid})
   MERGE (d)-[:MENTIONS]->(n)
   ```
3. **构建实体间共现关系**（对同一文档内的实体两两配对）：
   ```cypher
   MATCH (a:Entity {name:$a}), (b:Entity {name:$b})
   MERGE (a)-[r:RELATED]->(b)
   ON CREATE SET r.weight=1
   ON MATCH SET r.weight=r.weight+1
   ```

共现权重使用 `ON CREATE SET / ON MATCH SET` 幂等操作，支持增量更新。

**图谱检索**（`search`）：

```python
def search(self, query: str, top_k: int = 8) -> tuple[list[RetrieverItem], list[dict]]:
    entities = self._extract_entities(query, topk=5)  # 从查询提取 5 个实体
    doc_scores = defaultdict(float)                     # doc_id → 命中次数

    for entity in entities:
        rows = session.run(
            "MATCH (n:Entity {name:$name})<-[:MENTIONS]-(d:Doc) "
            "RETURN d.id AS doc_id, d.content AS content LIMIT 20", name=entity)
        for row in rows:
            doc_scores[row["doc_id"]] += 1.0          # 每命中一个实体 +1
            evidence.append({"entity": entity, "doc_id": row["doc_id"]})

    ranked = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
```

**计分逻辑**：如果一个文档被查询中提取的多个实体命中，分数会叠加。例如查询"知识库系统的检索优化"，提取实体为["知识库", "检索", "优化"]，一个同时提到这三个实体的文档得分为 3.0，远高于只提到一个实体的文档。

**返回值**：`RetrieverItem` 的 `content` 字段为空字符串（图谱检索不直接返回文本内容，仅提供文档关联），`source` 为 `"neo4j_graph"`，`metadata` 中包含 `doc_id`。`evidence` 列表记录每个实体命中的文档 ID 对，供调试分析使用。

**健康检查**（`health`）：执行 `RETURN 1` 测试 Cypher，捕获所有异常返回 `False`。上层代码据此决定是否跳过图谱检索。

---

### 3.7 RAG 编排层 — `core/orchestration/pipeline.py`

RAG 编排层是系统的**指挥中心**，协调检索和生成两个阶段，实现端到端的问答流水线。

#### 3.7.1 检索编排器 — `RetrievalOrchestrator`

**初始化依赖**：

```python
class RetrievalOrchestrator:
    def __init__(self, settings, registry, vector_store, bm25, graph_engine):
        self.embedder = registry.build_embedding()  # EmbeddingProvider 实例
```

**`retrieve(req: RetrieverRequest)` 方法**详细流程：

```
① 查询向量化
   query_vec = self.embedder.embed_query(req.query)
   # 一次 embed_query 调用，同时用于 Dense 检索和（CLIP 模式下的）以文搜图

② 四路并行召回（各召回 3x 超额候选）

   ②-a Dense 向量检索
       dense_items = vector_store.search(
           collection="kb_text_chunks",
           vector=query_vec,
           top_k=max(req.top_k * 3, 20))    # 最少 20 条，保证融合质量

   ②-b BM25 关键词检索
       bm25_items = bm25.search(req.query, top_k=max(req.top_k * 3, 20))

   ②-c Neo4j 图谱检索（健康检查通过时）
       if graph_engine.health():
           graph_items, graph_evidence = graph_engine.search(req.query, top_k=max(req.top_k * 2, 12))

   ②-d 图像检索
       if req.image_inputs:
           # 有上传图片 → CLIP 编码图片 → 在 kb_image_assets 中检索
           image_vectors = self.embedder.embed_image_paths(req.image_inputs)
           image_items = vector_store.search(collection="kb_image_assets", vector=image_vectors[0], ...)
       elif self.embedder.text_provider == "clip":
           # CLIP 模式 → 以文搜图（查询向量和图片在同一向量空间）
           image_items = vector_store.search(collection="kb_image_assets", vector=query_vec, ...)

③ 两阶段融合排序

   weights = normalize_weights(retrieval_weights, use_image=是否有图片)
   merged_items, merged_scores = weighted_merge(dense_items, bm25_items, graph_items,
                                                image_items, weights, req.top_k)

   rrf_scores = reciprocal_rank_fusion([dense_items, bm25_items, graph_items, image_items])
   for item in merged_items:
       item.score += rrf_scores.get(item.item_id, 0.0)    # RRF 补偿叠加
   merged_items = sorted(merged_items, key=lambda i: i.score, reverse=True)

④ 构建返回结果
   return RetrieverResult(
       items=merged_items,
       scores=merged_scores,
       sources=sorted({item.source for item in merged_items if item.source}),
       graph_evidence=graph_evidence,
       latency_ms=round((time.perf_counter() - start) * 1000, 2),
       debug_info={...} if req.debug else {}
   )
```

**超额召回设计**：四路检索各召回 `top_k × 3` 条（最少 20 条），融合时取 top-k。超额召回保证了融合后不会因为某路检索质量差而导致最终结果不足。例如用户请求 `top_k=10`，Dense 召回 30 条、BM25 召回 30 条，融合后取前 10 名。

**Debug 模式输出**：当 `req.debug=True` 时，`debug_info` 包含各路检索的 top-5 结果、归一化后的权重、RRF 分数字典，可在前端调试面板中查看原始召回数据。

#### 3.7.2 端到端问答流水线 — `RAGPipeline`

**初始化**：

```python
class RAGPipeline:
    def __init__(self, settings, orchestrator, registry):
        self.fast_model = registry.build_chat("fast_model")      # 用于查询改写
        self.quality_model = registry.build_chat("quality_model")  # 用于答案生成
```

使用两个不同的模型实例：`fast_model` 负责轻量级的查询改写（追求速度），`quality_model` 负责答案生成（追求质量）。当前默认配置中两者都是 `glm-4-flash`，但架构上支持使用不同能力的模型。

**`answer_stream(req: RetrieverRequest)` 方法**详细流程：

```
① 查询改写（rewrite_query）
   if not chat_history:
       return query                    # 无对话历史，直接使用原始查询
   messages = build_rewrite_messages(query, chat_history[-8:])  # 最近 8 轮
   rewritten = self.fast_model.generate(messages, temperature=0.1)
   return rewritten.strip() or query   # 改写失败时回退到原始查询

② 构造检索请求（使用改写后的查询）
   retrieval_req = RetrieverRequest(
       query=rewritten_query,          # 替换为改写后的查询
       chat_history=req.chat_history,
       modality=req.modality,
       image_inputs=req.image_inputs,
       top_k=req.top_k,
       debug=req.debug)

③ 多路检索
   retrieval_result = self.orchestrator.retrieve(retrieval_req)

④ 上下文构建
   context = build_context(
       items=[asdict(item) for item in retrieval_result.items],
       max_chars=9000)                 # 上下文窗口上限 9000 字符

⑤ 答案生成（流式）
   messages = build_answer_messages(req.query, context, chat_history[-8:])
   stream = self.quality_model.stream(messages, temperature=0.3)

⑥ 返回三元组
   return (stream, rewritten_query, retrieval_result)
```

**查询改写的价值**：在多轮对话场景下，用户可能说"它的原理是什么"，改写器会结合历史将"它"替换为具体实体，生成如"知识库系统的检索原理是什么"这样的独立查询，大幅提高检索准确性。改写使用 `temperature=0.1` 保证输出确定性。

**答案生成的 temperature=0.3**：比改写稍高，允许一定的创造性，但仍然以事实性回答为主。

---

### 3.8 生成层 — `core/generation/prompting.py`

生成层负责构造 LLM 的输入消息，是连接检索结果和最终答案的桥梁。

#### 3.8.1 查询改写提示词 — `build_rewrite_messages`

```python
def build_rewrite_messages(query, chat_history):
    history = "\n".join([f"{role}: {text}" for role, text in chat_history[-8:]])
    return [
        {"role": "system", "content": "你是查询优化器。请将用户问题改写为适合检索的单句，不要回答问题本身。"},
        {"role": "user", "content": f"历史对话:\n{history}\n\n当前问题:\n{query}\n\n输出改写后的查询："},
    ]
```

**设计要点**：
- 系统提示词明确要求"改写为适合检索的单句"，约束 LLM 不要回答问题本身
- 仅保留最近 8 轮对话，避免提示词过长导致改写目标偏移
- 对话格式为 `role: content`，简洁明了

#### 3.8.2 答案生成提示词 — `build_answer_messages`

```python
def build_answer_messages(query, context, chat_history):
    history = "\n".join([f"{role}: {text}" for role, text in chat_history[-8:]])
    return [
        {"role": "system", "content": "你是实验室知识库助手 LabKB。必须优先依据给定上下文回答；若证据不足明确说不知道；给出简洁回答并附来源。"},
        {"role": "user", "content": f"历史对话:\n{history}\n\n上下文:\n{context}\n\n问题:\n{query}\n\n请作答。"},
    ]
```

**设计要点**：
- "必须优先依据给定上下文回答" → 约束模型不要编造信息
- "若证据不足明确说不知道" → 避免幻觉
- "给出简洁回答并附来源" → 要求回答风格简洁、可溯源
- 上下文注入到 user message 中（非 system message），这是多数 RAG 系统的最佳实践

#### 3.8.3 上下文构造 — `build_context`

```python
def build_context(items: list[dict], max_chars: int = 9000) -> str:
    chunks: list[str] = []
    total = 0
    for it in items:
        piece = f"[source:{it.get('source','')}] {it.get('content','')}"
        if total + len(piece) > max_chars:
            break                              # 硬截断，超出即停
        chunks.append(piece)
        total += len(piece)
    return "\n\n".join(chunks)
```

**设计要点**：
- 每个检索片段前附带 `[source:path]` 标记，模型回答时可引用来源
- `max_chars=9000` 硬上限，避免超出模型上下文窗口
- 按融合排序后的顺序拼接，最相关的排在前面
- 简单的字符数截断策略（不按 token 精确计算），实现简洁

---

### 3.9 Web 服务层 — `webapp/`

#### 3.9.1 应用入口 — `main.py`

**应用初始化**：

```python
app = FastAPI(title="LabKB", version="3.0.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
recorder = EvaluationRecorder(log_dir=ROOT_DIR / "runtime" / "eval")  # 评估数据采集器
```

`EvaluationRecorder` 在应用启动时创建，后续每次 SSE 对话结束后自动记录评估数据。

**路由详解**：

| 路由 | 方法 | 参数 | 功能 | 返回类型 |
|------|------|------|------|---------|
| `/` | GET | — | 渲染 Web UI，传入 `mode_options` 和 `default_mode` | `HTMLResponse` |
| `/api/health` | GET | — | 检查 Qdrant 和 Neo4j 连通性 | `dict` |
| `/api/chat/stream` | POST | `query`(Form), `mode`(Form), `top_k`(Form), `debug`(Form), `chat_history`(Form), `image`(File) | SSE 流式对话 | `StreamingResponse` |
| `/files` | GET | `path`(Query) | 安全的文件透传 | `FileResponse` |

**SSE 流式对话接口**（`/api/chat/stream`）的完整处理流程：

```
① 参数解析与校验
   - 解析 Form 字段：query, mode, top_k, debug, chat_history
   - 解析 File 字段：image（可选）
   - 校验 query 非空，无效 mode 降级为 HYBRID
   - 解析 chat_history JSON 为 list[tuple[str, str]]

② 图片上传处理
   if image and image.filename:
       saved_path = _save_upload(image)
       # _save_upload: UUID 重命名 + 写入 runtime/uploads/

③ Pipeline 调用
   req = RetrieverRequest(query=..., chat_history=..., modality=..., ...)
   stream, rewritten_query, retrieval_result = pipeline.answer_stream(req)

④ SSE 事件流生成
   async def event_stream():
       # 事件 1: meta — 元信息
       yield _sse("meta", {"rewritten_query": ..., "uploaded_images": [...]})

       # 事件 2-N: token — 逐 token 增量
       for delta in stream:
           answer_parts.append(delta)
           yield _sse("token", {"delta": delta})

       # 事件 N+1: done — 完成（含完整答案和检索结果）
       yield _sse("done", {"answer": answer, "retrieval": serialized_result, ...})

       # 异步记录评估数据（不阻塞响应）
       recorder.record(query=..., rewritten_query=..., retrieval_result=..., answer=...)
```

**SSE 协议格式**：

```
event: meta
data: {"rewritten_query": "知识库系统架构", "uploaded_images": [...]}

event: token
data: {"delta": "KB"}

event: token
data: {"delta": " Assistant"}

event: done
data: {"answer": "LabKB 是一个...", "retrieval": {...}}
```

**响应头设置**：

```python
StreamingResponse(event_stream(), media_type="text/event-stream", headers={
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",   # 禁用 Nginx 缓冲，确保流式传输
})
```

**安全机制**：

1. **文件路径限制**（`_ensure_workspace_file`）：使用 `Path.relative_to(root)` 检查，任何尝试访问项目目录之外的文件请求返回 403。
2. **上传文件重命名**（`_save_upload`）：原始文件名经过过滤（仅保留字母数字、`-`、`_`），然后拼接 8 位随机 UUID 后缀，彻底消除路径注入风险。
3. **HTML 转义**：前端 `app.js` 中所有用户输入通过 `escapeHtml()` 转义后再渲染，防止 XSS。

**检索结果序列化**（`_serialize_result`）：

```python
def _serialize_result(result: RetrieverResult) -> dict:
    return {
        "items": [_serialize_item(item) for item in result.items],
        "scores": result.scores,
        "sources": [{"path": ..., "name": ..., "url": ..., "is_image": ...}],
        "graph_evidence": result.graph_evidence,
        "latency_ms": result.latency_ms,
        "debug_info": result.debug_info,
    }
```

每个 item 会额外计算 `is_image`（通过后缀判断）和 `source_name`（文件名）和 `source_url`（可点击的文件链接），供前端展示。

#### 3.9.2 启动引导 — `bootstrap.py`

使用 Python `@lru_cache(maxsize=1)` 装饰器实现**进程级单例**：

```python
@lru_cache(maxsize=1)
def get_pipeline() -> "RAGPipeline":
    settings = get_settings()
    registry = ModelRegistry(settings.model_registry_path)
    vector_store = QdrantVectorStoreAdapter(
        url=settings.qdrant_url, api_key=settings.qdrant_api_key,
        text_collection=settings.text_collection, image_collection=settings.image_collection,
        parent_collection=settings.parent_collection,
        text_dim=registry.embeddings["text"].get("dimensions", 1024),
        image_dim=registry.embeddings["image"].get("dimensions", 512))
    bm25 = BM25Indexer(settings.bm25_index_file)
    bm25.load()
    graph_engine = LightRAGGraphEngine(settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password)
    orchestrator = RetrievalOrchestrator(settings=settings, registry=registry,
                                         vector_store=vector_store, bm25=bm25,
                                         graph_engine=graph_engine)
    return RAGPipeline(settings=settings, orchestrator=orchestrator, registry=registry)
```

初始化顺序：Settings → ModelRegistry → VectorStore + BM25 + GraphEngine → Orchestrator → Pipeline。每个组件都有明确的依赖关系，通过构造函数注入。

`get_pipeline()` 仅在首次 API 请求时执行（惰性初始化），后续请求复用缓存实例，避免了每次请求重建模型的开销。注意：`QdrantVectorStoreAdapter` 初始化时会自动创建不存在的集合，`BM25Indexer.load()` 会加载已有索引或静默返回空索引。

#### 3.9.3 前端界面 — `templates/index.html` + `static/app.js`

**三栏布局**：

```
┌──────────────┬──────────────────────┬──────────────┐
│  左侧边栏     │     主内容区          │  右侧边栏     │
│  (sidebar)   │    (main-shell)       │  (inspector)  │
│              │                       │              │
│  品牌标识     │  顶部工具栏            │  检索结果      │
│  新建对话按钮  │  ┌─────────────────┐ │  文本来源      │
│  系统状态     │  │  对话消息列表     │ │  图片来源      │
│  (Qdrant/    │  │  (message-list)  │ │  调试信息      │
│   Neo4j/     │  └─────────────────┘ │  (折叠面板)    │
│   Ready)     │  ┌─────────────────┐ │              │
│  检索模式     │  │  输入框 + 上传   │ │              │
│  (radio)     │  │  (composer)     │ │              │
│  Top K 滑块   │  └─────────────────┘ │              │
│  调试开关     │                       │              │
└──────────────┴──────────────────────┴──────────────┘
```

**前端核心功能**：

1. **SSE 流式渲染**：`consumeStream()` 使用 `ReadableStream` API 读取 SSE 事件流，解析 `event:` + `data:` 行，实时更新 assistant 消息内容。
2. **健康检查轮询**：`refreshHealth()` 每 30 秒调用 `/api/health`，更新侧边栏状态指示灯（绿色/红色圆点）。
3. **图片上传预览**：选择图片后通过 `URL.createObjectURL()` 生成本地预览，发送后清除。
4. **对话历史管理**：`state.messages` 数组维护完整对话历史，每次请求携带 `chat_history` JSON 字符串。"新建对话"按钮清空数组和检索面板。
5. **检索结果面板**：`renderInspector()` 在收到 `done` 事件后，将检索结果分为文本来源和图片来源两组展示，并显示调试信息（JSON 格式）。
6. **自动滚动**：每次新消息追加后调用 `scrollIntoView({ behavior: "smooth" })`。

---

### 3.10 可观测性层 — `core/observability/logging_utils.py`

#### 3.10.1 JSON 格式化器 — `JsonFormatter`

所有日志以 JSON 格式输出，每条记录包含：

```json
{
  "ts": "2026-04-05T15:00:00",
  "level": "INFO",
  "name": "kb_assistant_v2",
  "msg": "index build started",
  "trace_id": "abc123def456",
  "extra_data": {"text_docs": 42, "image_docs": 5}
}
```

`trace_id` 和 `extra_data` 是可选字段，仅在使用 `trace_span` 或手动传入 `extra` 时包含。

#### 3.10.2 日志初始化 — `setup_logging`

```python
def setup_logging(level=logging.INFO) -> logging.Logger:
    logger = logging.getLogger("kb_assistant_v2")
    if logger.handlers:     # 防止重复添加 handler
        return logger

    # 通道 1: 控制台（StreamHandler）
    console_handler = logging.StreamHandler()

    # 通道 2: 文件（TimedRotatingFileHandler，每天滚动，保留 14 天）
    file_handler = TimedRotatingFileHandler(
        filename=log_file, when="midnight", interval=1,
        backupCount=14, encoding="utf-8")

    logger.propagate = False  # 阻止传播到 root logger
```

日志文件命名规则：当天为 `app.log`，历史文件自动重命名为 `app.log.2026-04-04`、`app.log.2026-04-03` 等，超过 14 天的自动删除。

#### 3.10.3 阶段追踪 — `trace_span`

```python
@contextmanager
def trace_span(logger, stage: str) -> Iterator[str]:
    trace_id = uuid.uuid4().hex[:16]    # 生成 16 字符的追踪 ID
    start = time.perf_counter()
    logger.info(f"{stage} started", extra={"trace_id": trace_id})
    try:
        yield trace_id                  # 追踪 ID 传递给调用方
    finally:
        cost_ms = (time.perf_counter() - start) * 1000
        logger.info(f"{stage} finished",
                    extra={"trace_id": trace_id, "extra_data": {"latency_ms": round(cost_ms, 2)}})
```

使用示例（当前代码中尚未广泛使用，但已预留）：

```python
with trace_span(logger, "retrieval") as trace_id:
    result = orchestrator.retrieve(req)
    # 日志自动记录: "retrieval started" + "retrieval finished (latency_ms=150.23)"
```

---

### 3.11 性能评估系统 — `core/evaluation/`

评估系统是 LabKB 的重要能力模块，提供从数据采集到自动化评估再到报告生成的完整链路。

#### 3.11.1 架构概览

评估系统分为**实时采集**和**离线评估**两条链路：

```
┌─────────────────────────────────────────────────────────────────┐
│  实时采集链路（每次问答自动执行）                                    │
│  API 请求 → EvaluationRecorder.record() → runtime/eval/qa_logs.jsonl │
├─────────────────────────────────────────────────────────────────┤
│  离线评估链路（手动或定时触发）                                     │
│                                                                  │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────┐            │
│  │ Dataset   │    │ Eval         │    │ Ragas        │            │
│  │ Loader    │───→│ Ingestor     │───→│ Evaluator    │            │
│  │ (加载测试集)│    │ (语料库导入)  │    │ (指标计算)    │            │
│  └──────────┘    └──────────────┘    └──────┬───────┘            │
│                                              │                    │
│                                       ┌──────▼───────┐            │
│                                       │ Report       │            │
│                                       │ Generator    │            │
│                                       │ (报告导出)    │            │
│                                       └──────────────┘            │
├─────────────────────────────────────────────────────────────────┤
│  评估配置: configs/eval_config.yaml                                │
│  评估脚本: scripts/eval/run_benchmark.py / quick_eval.py           │
└─────────────────────────────────────────────────────────────────┘
```

#### 3.11.2 数据集加载器 — `dataset_loader.py` — `DatasetLoader`

支持从 Hugging Face Hub 或本地 JSONL 文件加载标准化测试集。

**支持的数据集**：

| 数据集键 | 语言 | HF 路径 | Split | 样本量 | 数据结构 |
|---------|------|---------|-------|--------|---------|
| `superclue_c3` | 中文 | TigerResearch/tigerbot-superclue-c3-zh-5k | train | 4792 | instruction(问题), input(短文), output(答案) |
| `crud_rag` | 中文 | AndrewTsai0406/CRUD_RAG_3QA | train | 3199 | questions, answers, event, news1/2/3 |
| `hotpotqa` | 英文 | hotpot_qa (fullwiki) | validation | 大规模 | question, answer, context=[[title,text],...] |
| `squad_v2` | 英文 | rajpurkar/squad_v2 | validation | 大规模 | question, answers.text, context |
| `amnesty_qa` | 英文 | explodinggradients/amnesty_qa (english_v3) | eval | 20 | user_input, reference, retrieved_contexts |

**加载优先级**：优先从本地 `data/eval/{dataset_name}.jsonl` 加载；若本地文件不存在则从 Hugging Face 下载。

**数据格式化流程**：

```
原始数据集 → _format_dataset() → 统一格式
    每条样本包含:
    - question: str           用户问题
    - ground_truth: str|list  标准答案
    - golden_docs: list[str]  黄金参考文档（该问题对应的相关文档）
    - metadata: dict          数据集来源、样本 ID 等
```

各数据集的格式差异通过 `_config_mapping` 中的 `field_mapping` 字段映射解决，特殊格式通过后处理函数处理。例如 `crud_rag` 的三篇新闻需要合并为文档列表。

**`extract_corpus()`** — 从测试集提取全量去重语料库：

```python
def extract_corpus(self, dataset) -> list[str]:
    seen = set()
    corpus = []
    for sample in dataset:
        for doc in sample["golden_docs"]:
            doc_key = doc.strip()[:200]   # 用前 200 字符做去重 key
            if doc_key not in seen:
                seen.add(doc_key)
                corpus.append(doc.strip())
    return corpus
```

语料库作为评估检索的源数据，与生产知识库完全隔离。

#### 3.11.3 评估语料库导入器 — `eval_ingestor.py` — `EvalDataIngestor`

将测试集的全量语料库切块、向量化后导入独立的 Qdrant 评估集合。

**集合命名规则**：`{dataset_name}_eval`，例如 `crud_rag_eval`。

**导入流程**：

```
语料库文本 → MD5 去重 → 固定长度切块（500 字符，50 重叠）
    → 批量 Embedding（32 条/批）
    → 创建/重建 Qdrant 集合
    → 分批写入（64 条/批）
```

**与生产索引的区别**：
- 使用独立的集合名（`_eval` 后缀），不影响生产知识库
- 使用简单的固定长度切块（非父子分块），评估场景下无需复杂的分块策略
- 集合存在时自动删除重建（`delete_collection` + `create_collection`），确保评估数据干净

**支持的 Embedding 接口适配**：自动检测 EmbeddingProvider 的方法名（`embed_documents` / `embed_text` / `embed_texts` / `embed`），兼容不同接口的 Embedding 实现。

#### 3.11.4 RAGAS 评估器 — `evaluator.py` — `RagasEvaluator`

**初始化与 RAGAS 配置**：

```python
class RagasEvaluator:
    def __init__(self, llm, embeddings):
        self._configure_ragas()

    def _configure_ragas(self):
        # 使用 ChatZhipuAI（智谱官方 LangChain 适配器）而非 ChatOpenAI
        from langchain_community.chat_models import ChatZhipuAI
        self._ragas_llm = ChatZhipuAI(model=ragas_model, api_key=api_key,
                                       temperature=0.1, max_tokens=2048)

        # 将项目的 EmbeddingProvider 包装为 Langchain Embeddings 接口
        class _ProviderEmbeddings(Embeddings):
            def embed_documents(self, texts): return self._provider.embed_text(texts)
            def embed_query(self, text): return self._provider.embed_text([text])[0]
        self._ragas_embeddings = _ProviderEmbeddings(self.embeddings)
```

使用 `ChatZhipuAI` 而非 `ChatOpenAI` 的原因：智谱 API 与 OpenAI 协议存在参数不兼容问题（如 `max_tokens` 参数位置不同），使用官方适配器可避免此类问题。Embedding 通过适配器类包装，使项目的 `EmbeddingProvider` 符合 Langchain 的 `Embeddings` 接口规范。

**评估模式一：无监督批量评估**（`evaluate_batch`）

```python
def evaluate_batch(self, qa_logs: list[dict]) -> dict:
    dataset = self._convert_to_ragas_dataset(qa_logs)
    # 转换格式: {"question": [...], "answer": [...], "contexts": [[...], [...]]}
    metrics = [faithfulness, answer_relevancy, context_precision, context_recall, context_entity_recall]
    results = evaluate(dataset=dataset, metrics=metrics, batch_size=5, llm=self._ragas_llm, embeddings=self._ragas_embeddings)
    return self._parse_results(results, qa_logs)
```

输入来自 `EvaluationRecorder` 采集的 QA 日志。由于没有标准答案，只能计算不需要 ground_truth 的指标。`batch_size=5` 控制每次发送给 LLM 的评估请求数量。

**评估模式二：基于标准答案的评估**（`evaluate_with_ground_truth`）

```python
def evaluate_with_ground_truth(self, questions, answers, contexts, ground_truths, documents=None, batch_size=5):
    metrics = [faithfulness, answer_relevancy, context_precision, context_recall,
               context_entity_recall, answer_correctness, answer_similarity]
    results = evaluate(dataset=..., metrics=metrics, ...)

    # 额外计算自定义指标
    custom_metrics = {}
    if documents:
        custom_metrics["mrr@k"] = self._calculate_mrr(contexts, documents)
        custom_metrics["coverage@k"] = self._calculate_context_utilization(contexts, documents)
    custom_metrics["answer_coverage"] = self._calculate_answer_completeness(answers, ground_truths)
```

**自定义指标算法**：

| 指标 | 算法 | 说明 |
|------|------|------|
| `mrr@k` | 遍历每个检索结果，找到第一个与参考文档匹配的位置，取 `1/rank` | 衡量首个相关文档的排名 |
| `answer_coverage` | 答案词集与标准答案词集的交集大小 / 标准答案词集大小 | 基于关键词覆盖率的简单度量 |
| `coverage@k` | 检索内容词集与参考文档词集的交集大小 / 参考文档词集大小 | 衡量检索的有效信息覆盖率 |

**结果解析兼容性**：由于 RAGAS 0.2.x 的 API 可能变化，结果解析兼容三种格式：
```python
if hasattr(results, '_scores_dict'):
    results_dict = results._scores_dict
elif hasattr(results, 'scores') and isinstance(results.scores, list):
    results_dict = {k: [d[k] for d in results.scores] for k in results.scores[0].keys()}
else:
    results_dict = dict(results)
```

#### 3.11.5 数据采集器 — `recorder.py` — `EvaluationRecorder`

在 API 层（`main.py` 的 `event_stream()` 函数中）注入，每次问答完成后自动记录：

```python
recorder.record(
    query=query,
    rewritten_query=rewritten_query,
    retrieval_result=retrieval_result,  # RetrieverResult 对象
    answer=answer,                      # 完整答案字符串
    search_mode=search_mode.value,
    image_inputs=image_inputs)
```

**记录格式**（每行一个 JSON 对象，JSONL 格式）：

```json
{
  "trace_id": "uuid",
  "timestamp": "2026-04-05T15:00:00",
  "query": "什么是 RAG？",
  "rewritten_query": "RAG 检索增强生成技术的原理",
  "retrieval_result": {
    "items": [{"item_id": "...", "content": "...", "source": "...", "score": 0.85, "metadata": {...}}],
    "sources": ["kb_source/rag_guide.md"],
    "graph_evidence": [{"entity": "RAG", "doc_id": "uuid-of-doc"}],
    "latency_ms": 150.23,
    "debug_info": {}
  },
  "answer": "RAG（Retrieval-Augmented Generation）是一种...",
  "latency_ms": {"retrieval": 150.23, "generation": 0, "total": 150.23},
  "search_mode": "hybrid",
  "image_inputs": []
}
```

**容错设计**：记录失败时 `try-except` 捕获异常并 `print` 到控制台，不影响 SSE 响应。日志文件路径为 `runtime/eval/qa_logs.jsonl`，自动创建目录。

#### 3.11.6 报告生成器 — `report_generator.py` — `ReportGenerator`

**核心方法**：

| 方法 | 功能 | 输出 |
|------|------|------|
| `save_json(output_path)` | JSON 格式报告（可格式化或压缩） | `.json` |
| `save_csv(output_path, include_contexts)` | 样本级 CSV（每行一个样本 + 各指标分数） | `.csv` |
| `generate_summary_csv(output_path)` | 指标摘要 CSV（每行一个指标 + 均值/标准差/最值） | `_summary.csv` |
| `annotate_anomalies(threshold)` | 标注异常样本（任一关键指标低于阈值） | `list[dict]` |
| `save_anomaly_report(anomalies, output_path)` | 异常样本 CSV 报告 | `_anomalies.csv` |
| `export_all(base_path)` | 一键导出所有格式 | 多个文件 |
| `print_summary()` | 在控制台打印报告摘要 | stdout |

**异常标注逻辑**：

```python
def annotate_anomalies(self, threshold=0.5, metrics=None):
    # 默认检查: faithfulness, answer_correctness, answer_similarity, recall@k
    for sample in samples:
        for metric in metrics:
            if sample["metrics"][metric] < threshold:
                anomaly_metrics[metric] = score
```

每个异常样本记录其 `sample_id`、`question`（截断 100 字符）、`anomaly_metrics`（低于阈值的指标及分数）和 `all_metrics`（全量指标），便于后续分析。

#### 3.11.7 标准化评估脚本 — `scripts/eval/run_benchmark.py`

**6 步评估流程详解**：

```
Step 1/6: 准备测试数据
  ├── DatasetLoader.load() 加载并格式化测试集
  └── DatasetLoader.extract_corpus() 提取全量去重语料库

Step 2/6: 语料库向量化，导入专用 Qdrant 集合
  ├── EvalDataIngestor.ingest_corpus() 切块 + 向量化 + 写入 {dataset}_eval
  └── [--skip-ingest 时跳过] 直接用 golden_docs 作为上下文

Step 3/6: 检索 + 答案生成（逐样本）
  for sample in dataset:
      ├── [skip-ingest] 直接用 golden_docs 作为上下文
      │   或 [正常模式] embed_query → search_eval_collection → 检索 top-k
      ├── query_llm() → 调用 fast_model 生成答案
      └── 记录 questions, answers, contexts, ground_truths, golden_docs

Step 4/6 + Step 5/6: RAGAS 指标计算
  ├── RagasEvaluator.evaluate_with_ground_truth()
  │   ├── RAGAS 指标: faithfulness, answer_relevancy, precision@k/recall@k/entity_recall@k,
  │   │               answer_correctness, answer_similarity
  │   └── 自定义指标: mrr@k, answer_coverage, coverage@k
  └── 返回结构化报告

Step 6/6: 报告生成 + Bad Case 分析
  ├── build_report() 组装完整报告
  │   ├── summary: 样本数、成功率、平均延迟
  │   ├── metrics: 各指标统计（均值/标准差/最值）
  │   ├── samples: 每个样本的详细结果
  │   ├── bad_cases: 异常样本列表（含原因分析）
  │   └── failed_samples: 失败样本列表
  └── save_report() 写入 JSON 文件
```

**Bad Case 自动分析**（`_analyze_bad_case`）：

```python
def _analyze_bad_case(weak_metrics):
    reasons = []
    if "faithfulness" in weak_metrics:
        reasons.append("答案可能包含幻觉（faithfulness过低），未忠实于检索上下文")
    if "recall@k" in weak_metrics:
        reasons.append("检索召回率低，黄金参考文档中关键信息未被检索到")
    # ... 其他指标分析
    return "; ".join(reasons)
```

每个 Bad Case 不仅标注分数，还给出**自然语言原因分析**，帮助开发者快速定位问题方向。

**命令行参数**：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--dataset` | 测试集名称 | `crud_rag` |
| `--subset-size` | 样本数量 | None（全部） |
| `--language` | 语言 | `zh` |
| `--top-k` | 检索 top_k | 10 |
| `--skip-ingest` | 跳过语料库导入，直接用 golden_docs 评估生成 | False |
| `--ingest-only` | 仅导入语料库 | False |
| `--cleanup` | 清理评估集合 | False |
| `--output` | 输出文件路径 | 自动生成带时间戳 |
| `--config` | 配置文件路径 | None |
| `--batch-size` | RAGAS 批处理大小 | 5 |

#### 3.11.8 快速评估脚本 — `scripts/eval/quick_eval.py`

封装了预定义配置的快捷评估工具，通过 `--zh`、`--en` 等标志一键启动：

| 快捷标志 | 数据集 | 样本数 | 说明 |
|---------|--------|--------|------|
| `--zh` | CRUD-RAG | 100 | 中文标准评估 |
| `--zh-small` | CRUD-RAG | 30 | 快速验证 |
| `--zh-superclue` | SuperCLUE-C3 | 50 | 中文阅读理解 |
| `--en` | HotpotQA | 50 | 英文多跳问答 |
| `--en-small` | HotpotQA | 20 | 快速验证 |
| `--en-nq` | SQuAD v2 | 100 | 英文抽取式问答 |
| `--all-zh` | — | — | 运行所有中文测试集 |
| `--all-en` | — | — | 运行所有英文测试集 |

内部通过 `subprocess.run()` 调用 `run_benchmark.py`，实际执行逻辑不变。

#### 3.11.9 离线批量评估脚本 — `scripts/eval/run_batch_eval.py`

针对已采集的 QA 日志（`runtime/eval/qa_logs.jsonl`）进行离线评估：

- 支持按 `search_mode` 筛选（如只评估 `hybrid` 模式的日志）
- 支持只评估最近 N 条记录（`--last-n`）
- 支持单条评估模式（`--single`），用于调试评估流程
- 评估完成后自动保存 JSON 报告并打印摘要

---

## 四、核心数据流

### 4.1 索引构建流程

```
kb_source/ 中的文件
    │
    ├── 文本文件 → load_raw_documents → RawDocument(modality=text)
    │                                          │
    │                                   sha256_file → checksum
    │                                          │
    │                              增量过滤（与 manifest 比对）
    │                                          │
    │                              split_parent_child
    │                                 │              │
    │                            父块(1500)      子块(380)
    │                                 │              │
    │                         embed_text      embed_text
    │                                 │              │
    │                    Qdrant kb_parent_docs  Qdrant kb_text_chunks
    │                                 │
    │                         BM25 索引构建
    │                                 │
    │                         Neo4j 图谱更新
    │
    └── 图片文件 → RawDocument(modality=image)
                          │
                    embed_image_paths (CLIP)
                          │
                Qdrant kb_image_assets
```

### 4.2 在线问答流程

```
用户查询 + 对话历史 + [图片]
        │
  ┌─────┴──────┐
  │ RAGPipeline │
  │             │
  │ ① rewrite_query  (fast_model, 有历史时)
  │      │
  │ ② RetrievalOrchestrator.retrieve
  │      │
  │      ├── embed_query → Dense 向量检索 (Qdrant)
  │      ├── BM25 关键词检索
  │      ├── Neo4j 图谱检索
  │      └── CLIP 图像检索 (图片输入 / 以文搜图)
  │      │
  │      ├── 加权合并 (text_dense=0.35, bm25=0.20, graph=0.25, image=0.20)
  │      └── RRF 补偿 (k=60)
  │      │
  │ ③ build_context (top-k 结果, 最大 9000 字符)
  │      │
  │ ④ quality_model.stream (temperature=0.3)
  │      │
  │ ⑤ SSE 流式返回
  └─────┴──────┘
        │
  [meta] → 改写查询 + 图片信息
  [token] → 逐 token 增量
  [done]  → 完整答案 + 检索来源 + 调试信息
```

---

## 五、配置说明

### 5.1 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `ZHIPUAI_API_KEY` | — | 智谱 AI API Key（必须） |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant 服务地址 |
| `QDRANT_API_KEY` | 空 | Qdrant API Key（可选） |
| `NEO4J_URI` | `bolt://localhost:7687` | Neo4j 连接地址 |
| `NEO4J_USER` | `neo4j` | Neo4j 用户名 |
| `NEO4J_PASSWORD` | `neo4j` | Neo4j 密码 |
| `DEBUG_RAG` | `false` | 调试模式开关 |
| `KB_MODEL_CACHE` | `runtime/model_cache` | 本地模型缓存目录 |
| `KB_LOG_FILE` | `runtime/logs/app.log` | 日志文件路径 |

### 5.2 检索权重配置

在 `AppSettings.retrieval_weights` 中配置（默认值）：

```python
{
    "text_dense": 0.35,   # 向量检索权重
    "bm25": 0.20,         # BM25 关键词检索权重
    "graph": 0.25,        # 知识图谱检索权重
    "image_clip": 0.20,   # 多模态图像检索权重
}
```

### 5.3 检索模式

| 模式 | 说明 |
|------|------|
| `hybrid`（默认） | 四路检索全部启用 |
| `text_only` | 仅向量 + BM25 |
| `multimodal` | 启用图片引导的跨模态检索 |
| `graph_first` | 优先使用图谱证据 |

---

## 六、关键设计决策

### 6.1 父子分块策略

采用两级分块而非单一分块，解决了 RAG 系统中"检索粒度 vs 语义完整性"的矛盾：
- **子块**（380 字符）用于精确匹配检索
- **父块**（1500 字符）作为召回后的上下文补充，存入独立的 Qdrant 集合

### 6.2 两阶段融合排序

- **加权合并**：根据业务经验配置权重，适合稳定场景
- **RRF 补偿**：名次融合算法，对单路检索偏废进行补偿，增强鲁棒性

### 6.3 CLIP 共享向量空间

使用 CLIP ViT-B-32 多语言模型作为默认 Embedding：
- 文本和图片处于同一向量空间
- 支持"以文搜图"和"以图搜文"
- 本地推理，无 API 调用开销

### 6.4 优雅降级

系统各组件设计了降级策略：
- Neo4j 不可用 → 跳过图谱检索，其余通道正常工作
- CLIP 不可用 → 返回零向量，图像检索通道自动禁用
- 评估记录失败 → 不影响主流程（try-except 静默处理）

### 6.5 增量索引

通过 `IndexManifest` 中记录的文件 checksum 实现增量更新：
- 仅处理新增或变更的文件
- 避免重复计算和写入
- 版本记录支持回溯

---

## 七、部署与运维

### 7.1 启动依赖服务

```bash
# Qdrant 向量数据库
docker run -d --name qdrant -p 6333:6333 -v qdrant_storage:/qdrant/storage qdrant/qdrant

# Neo4j 图数据库
docker run -d --name neo4j -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/neo4j -v neo4j_data:/data neo4j:latest
```

### 7.2 构建与运行

```bash
# 安装依赖
pip install -r requirements.txt

# 全量构建索引
python scripts/build_kb.py --full

# 增量更新索引
python scripts/build_kb.py

# 启动 Web 服务
uvicorn webapp.main:app --host 0.0.0.0 --port 8000 --reload
```

### 7.3 健康检查

`GET /api/health` 返回各依赖服务状态：
```json
{
  "ready": true,
  "qdrant": true,
  "neo4j": true,
  "errors": {}
}
```

---

## 八、评估系统使用指南

### 8.1 快速评估

```bash
# 中文 CRUD-RAG 测试集（100 条样本）
python scripts/eval/run_benchmark.py --dataset crud_rag --subset-size 100

# 英文 HotpotQA（50 条样本）
python scripts/eval/run_benchmark.py --dataset hotpotqa --subset-size 50 --language en

# 使用配置文件
python scripts/eval/run_benchmark.py --config configs/eval_config.yaml
```

### 8.2 评估报告解读

重点关注三个核心指标：
- **faithfulness > 0.8**：答案忠实度，低于此值说明存在幻觉
- **recall@k > 0.7**：检索召回率，低于此值说明检索策略需优化
- **answer_correctness > 0.7**：答案正确性，低于此值说明生成质量不佳

Bad Case 报告会自动标注低于阈值的样本，并给出原因分析。

### 8.3 评估集合管理

```bash
# 只导入语料库（不执行评估）
python scripts/eval/run_benchmark.py --dataset crud_rag --ingest-only

# 清理评估集合
python scripts/eval/run_benchmark.py --dataset crud_rag --cleanup
```
