# LabKB V2 实施与运行说明

## 1. 项目结构

- `webapp/main.py`：FastAPI 前端入口与 SSE 流式接口。
- `core/ingestion`：文档 / 图片加载与父子分块。
- `core/indexing`：Qdrant、BM25、manifest 和索引构建器。
- `core/retrieval`：GraphRAG（Neo4j）与融合检索。
- `core/orchestration`：查询改写、检索编排、回答生成。
- `core/providers`：OpenAI-compatible 模型与 embedding 抽象。
- `core/observability`：结构化日志与 trace。

## 2. 环境变量

- `QWEN_API_KEY`：Qwen / OpenAI-compatible API key。
- `QDRANT_URL`：Qdrant 地址，默认 `http://localhost:6333`。
- `QDRANT_API_KEY`：Qdrant key，可选。
- `NEO4J_URI`：Neo4j Bolt URI，默认 `bolt://localhost:7687`。
- `NEO4J_USER`：Neo4j 用户名，默认 `neo4j`。
- `NEO4J_PASSWORD`：Neo4j 密码，默认 `neo4j`。
- `DEBUG_RAG`：调试模式开关，`true/false`。

## 3. 安装与启动

```bash
pip install -r requirements.txt
python scripts/build_kb.py --full
uvicorn webapp.main:app --host 0.0.0.0 --port 8000 --reload
```

## 4. 索引说明

- 文本索引：Qdrant `kb_text_chunks`。
- 父文档索引：Qdrant `kb_parent_docs`。
- 图片索引：Qdrant `kb_image_assets`。
- 稀疏索引：`data_base/bm25_index.json`。
- manifest：`data_base/index_manifest.json`。

## 5. 调试说明

启用 `DEBUG_RAG` 后可在页面查看：

- 查询重写结果
- Dense / BM25 / Graph / Image 各路召回的 Top 片段
- 融合权重与 RRF 信息
- 检索耗时

## 6. 兼容与降级

- Neo4j 不可用：自动降级到文本 Hybrid 检索。
- 图片向量模型不可用：自动降级为零向量（可运行但效果弱）。
- 视觉模型不可用：后续可接入 OCR + 文本检索降级路径。
