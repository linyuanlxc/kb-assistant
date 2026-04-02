# KB Assistant V3

一个可部署的个人知识库助手，基于 `FastAPI + Qdrant + Neo4j + 多模态检索`，支持文本/图片入库、混合检索、流式回答、调试面板和增量索引。

## Quick Start

### 1. 启动依赖服务

```bash
# Qdrant
Docker run -d --name qdrant -p 6333:6333 -v qdrant_storage:/qdrant/storage qdrant/qdrant

# Neo4j
Docker run -d --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/neo4j \
  -v neo4j_data:/data \
  neo4j:latest
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置模型与服务

推荐通过环境变量配置：

```bash
export ZHIPUAI_API_KEY="your_api_key_here"
export QDRANT_URL="http://localhost:6333"
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="neo4j"
```

### 4. 构建索引

```bash
python scripts/build_kb.py --full
# 或后续增量更新
python scripts/build_kb.py
```

### 5. 启动 Web 应用

```bash
uvicorn webapp.main:app --host 0.0.0.0 --port 8000 --reload
```

浏览器打开 `http://localhost:8000`。

## 当前架构

- `webapp/`：FastAPI 服务入口、模板和静态资源
- `core/`：检索、索引、生成和编排逻辑
- `scripts/build_kb.py`：索引构建入口
- `kb_source/`：知识源目录
- `runtime/uploads/`：会话上传图片缓存

## 核心能力

- 文本与图片统一入库
- Dense + BM25 + Graph + Image 混合检索
- SSE 流式回答
- 图片随问随传
- 最新检索来源和调试信息面板
- Qdrant / Neo4j 健康检查

## API 概览

- `GET /`：Web UI
- `GET /api/health`：依赖健康检查
- `POST /api/chat/stream`：流式对话接口
- `GET /files?path=...`：工作区文件透传，仅允许访问仓库内文件

## 说明

- `app/streamlit_app.py` 保留在仓库中作为旧实现参考，但新的启动入口是 `webapp.main:app`。
- 若 Neo4j 不可用，检索会退化；若图像模型不可用，多模态效果会下降。
