# KB Assistant V2

一个可部署的个人知识库助手，基于 `LightRAG + Qdrant + Neo4j + 多模态检索`，支持文本/图片入库、混合检索、流式回答、调试面板和增量索引。

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
