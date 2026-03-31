# KB Assistant V2

一个可部署的个人知识库助手，基于 `LightRAG + Qdrant + Neo4j + 多模态检索`，支持文本/图片入库、混合检索、流式回答、调试面板和增量索引。

## Quick Start

### 1. 启动依赖服务（Qdrant & Neo4j）

使用 Docker 一键启动：

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

> 启动后可通过 `http://localhost:7474` 访问 Neo4j 浏览器（账号 `neo4j` / 密码 `neo4j`）。
>
> 若不启动 Neo4j，系统会自动降级为无图检索模式，不影响基本使用。

### 2. 克隆项目 & 安装依赖

```bash
git clone <repo-url> && cd kb-assistant
pip install -r requirements.txt
```

### 3. 配置 API Key

```bash
# 默认使用智谱 AI（免费），去 https://open.bigmodel.cn 注册获取
export ZHIPUAI_API_KEY="your_api_key_here"

# 如需使用其他服务，可通过环境变量覆盖
# export QDRANT_URL="http://localhost:6333"
# export NEO4J_URI="bolt://localhost:7687"
# export NEO4J_USER="neo4j"
# export NEO4J_PASSWORD="neo4j"
```

> 如果想用阿里云千问模型，修改 `core/config/model_registry.yaml` 中的配置即可，详见文件内注释。

### 4. 构建索引

将 `kb_source/` 目录下的知识文件入库：

```bash
# 全量构建（首次运行 / 切换 Embedding 模型后）
python scripts/build_kb.py --full

# 后续增量更新
python scripts/build_kb.py
```

### 5. 启动应用

```bash
streamlit run app/streamlit_app.py
```

浏览器打开 `http://localhost:8501`，即可开始使用。支持切换检索模式（hybrid/text_only/multimodal/graph_first）和图片上传多模态检索。

## 1. 核心能力

- 文本与图片统一入库（增量更新）
- 混合检索：Dense（Qdrant）+ BM25 + Graph（Neo4j）+ Image（CLIP）
- 检索融合：RRF + 权重融合
- 生成链路：查询改写、上下文构建、流式回答
- 模型兼容：OpenAI-compatible 协议，配置化模型注册
- 可观测：结构化日志、调试模式（DEBUG_RAG）

## 2. 项目结构

```text
app/                    # Streamlit 前端
core/
  config/               # 配置与模型注册
  ingestion/            # 文档加载、分块
  indexing/             # 索引构建（Qdrant/BM25/manifest）
  retrieval/            # 图检索、融合检索
  generation/           # Prompt 构建
  orchestration/        # 端到端 RAG 编排
  observability/        # 日志与追踪
scripts/                # 构建脚本
docs/                   # 设计与开发文档
kb_source/              # 知识源文件目录
data_base/              # 索引与运行数据目录
```

## 3. 环境要求

- Python 3.10+
- Qdrant（默认：`http://localhost:6333`）
- Neo4j（默认：`bolt://localhost:7687`）

## 4. 安装

```bash
pip install -r requirements.txt
```

## 5. 配置

通过环境变量配置（推荐）：

- `QWEN_API_KEY`：模型 API Key（OpenAI-compatible）
- `QDRANT_URL`：Qdrant 地址
- `QDRANT_API_KEY`：Qdrant 鉴权（可选）
- `NEO4J_URI`：Neo4j 地址
- `NEO4J_USER`：Neo4j 用户名
- `NEO4J_PASSWORD`：Neo4j 密码
- `DEBUG_RAG`：调试开关（`true/false`）

模型与能力矩阵见：
- `core/config/model_registry.yaml`

## 6. 构建索引

增量构建：

```bash
python scripts/build_kb.py
```

全量重建：

```bash
python scripts/build_kb.py --full
```

## 7. 启动应用

```bash
streamlit run app/streamlit_app.py
```

页面支持：
- 检索模式切换（hybrid/text_only/multimodal/graph_first）
- 图片上传触发多模态检索
- 调试信息展示（召回结果、融合信息、延迟）

## 8. 关键文档

- `docs/V2_IMPLEMENTATION_GUIDE.md`：实施与运行说明
- `docs/DEVELOPER_API.md`：接口契约
- `docs/PROJECT_UPGRADE_SPEC.md`：升级方案细节

## 9. 注意事项

- 首次运行前请确保 Qdrant 和 Neo4j 已可访问。
- 若 Neo4j 不可用，系统会降级为无图检索模式。
- 若图像向量模型不可用，系统会降级为零向量占位（功能可运行，效果会下降）。
