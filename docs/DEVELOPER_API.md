# KB Assistant V3 接口契约

## 数据契约

### RetrieverRequest

```python
RetrieverRequest(
  query: str,
  chat_history: list[tuple[str, str]],
  filters: dict[str, Any] = {},
  modality: SearchMode = SearchMode.HYBRID,
  image_inputs: list[str] = [],
  top_k: int = 10,
  debug: bool = False,
)
```

### RetrieverResult

```python
RetrieverResult(
  items: list[RetrieverItem],
  scores: dict[str, float],
  sources: list[str],
  graph_evidence: list[dict[str, Any]],
  latency_ms: float,
  debug_info: dict[str, Any],
)
```

## 运行接口

### 索引构建

```bash
python scripts/build_kb.py
python scripts/build_kb.py --full
```

### Web 服务

```bash
uvicorn webapp.main:app --host 0.0.0.0 --port 8000 --reload
```

## HTTP 接口

### `GET /api/health`

返回：

```json
{
  "ready": true,
  "qdrant": true,
  "neo4j": true,
  "modes": []
}
```

### `POST /api/chat/stream`

请求类型：`multipart/form-data`

字段：

- `query`: 当前问题
- `mode`: `hybrid | text_only | multimodal | graph_first`
- `top_k`: 召回条数
- `debug`: 是否返回调试信息
- `chat_history`: JSON 数组
- `image`: 可选图片文件

响应类型：`text/event-stream`

事件：

- `meta`
- `token`
- `done`
- `error`

### `GET /files?path=...`

仅允许访问仓库根目录内的文件，用于图片和来源文件展示。

## 设计约束

- 没有图片输入时，自动禁用 `image` 分支并重归一化权重
- Neo4j 不可用时，系统会降级为无图谱优先的检索路径
- 所有模型调用均使用 OpenAI-compatible 协议
- Web UI 通过 SSE 接收流式回答
