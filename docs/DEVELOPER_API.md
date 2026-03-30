# KB Assistant V2 接口契约

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
python scripts/build_kb.py        # 默认增量构建
python scripts/build_kb.py --full # 执行全量重建
```

### Web 应用

```bash
streamlit run app/streamlit_app.py
```

## 设计约束

- 检索融合默认权重：`text_dense=0.35`、`bm25=0.20`、`graph=0.25`、`image_clip=0.20`
- 没有图片输入时，自动禁用 `image` 分支并重新归一化权重
- Neo4j 不可用时，自动降级为不包含 `graph` 的 `hybrid` 模式
- 所有模型调用均使用 OpenAI-compatible 协议
