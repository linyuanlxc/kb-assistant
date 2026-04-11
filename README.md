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

## 性能评估系统

KB Assistant V3 集成了基于 RAGAS 框架的完整性能评估系统，支持使用公开测试集对系统进行标准化评估。

### 核心特性

- **公开测试集支持**：中文（CRUD-RAG、SuperCLUE-RAG）、英文（HotpotQA、NQ）
- **完整评估指标**：覆盖检索、生成、端到端三阶段
- **批量评估**：支持大规模自动化评估
- **多格式报告**：JSON、CSV导出，异常样本标注
- **配置驱动**：通过配置文件灵活调整参数

### 评估指标

#### 检索阶段
- `recall@k`：上下文召回率
- `precision@k`：上下文精度
- `mrr@k`：平均倒数排名

#### 生成阶段
- `faithfulness`：忠实度（检测幻觉）
- `answer_relevancy`：答案相关性
- `answer_coverage`：答案覆盖率

#### 端到端阶段
- `answer_correctness`：答案正确性
- `answer_similarity`：答案相似度
- `coverage@k`：上下文覆盖率

### 快速开始

#### 1. 测试安装

```bash
python scripts/eval/test_installation.py
```

#### 2. 评估中文CRUD-RAG测试集（100条样本）

```bash
python scripts/eval/run_benchmark.py \
  --dataset crud_rag \
  --subset-size 100 \
  --output runtime/eval/crud_rag_eval.json
```

#### 3. 评估英文HotpotQA测试集（50条样本）

```bash
python scripts/eval/run_benchmark.py \
  --dataset hotpotqa \
  --subset-size 50 \
  --language en \
  --output runtime/eval/hotpotqa_eval.json
```

#### 4. 使用配置文件

```bash
python scripts/eval/run_benchmark.py \
  --config configs/eval_config.yaml
```

### 生成CSV报告

```bash
# 从JSON报告生成CSV
python -c "
from core.evaluation.report_generator import convert_json_to_csv
from pathlib import Path
convert_json_to_csv(Path('runtime/eval/crud_rag_eval.json'))
"
```

### 评估结果分析

评估完成后，报告包含以下内容：

- **整体摘要**：样本数量、成功率、平均延迟
- **指标统计**：各指标的均值、标准差、最小值、最大值
- **样本详情**：每个样本的问题、答案、ground_truth、各项指标分数
- **异常样本**：低于阈值（默认0.5）的样本自动标注
- **性能统计**：检索延迟、生成延迟分布

### 配置文件说明

评估系统支持通过 `configs/eval_config.yaml` 进行配置：

```yaml
# 数据集配置
dataset:
  name: "crud_rag"          # 测试集名称
  subset_size: 100          # 样本数量
  language: "zh"            # 语言

# 检索配置
retrieval:
  search_mode: "hybrid"     # 检索模式
  top_k: 10                 # 检索数量

# 评估配置
evaluation:
  batch_size: 5             # 批处理大小
  anomaly_threshold: 0.5    # 异常阈值
```

完整配置示例见 `configs/eval_config.yaml`。

### 常见问题

**Q: 如何选择测试集？**
A: 中文知识库推荐CRUD-RAG（100条），英文推荐HotpotQA（50条）。初次评估建议从50-100条开始。

**Q: 评估需要多长时间？**
A: 100条样本，hybrid模式，约需10-20分钟。只评估检索阶段会快很多（3-5分钟）。

**Q: 如何解读评估结果？**
A: 重点关注：faithfulness（>0.8）、recall@k（>0.7）、answer_correctness（>0.7）。查看异常样本报告，分析低分原因。

## 说明

- `app/streamlit_app.py` 保留在仓库中作为旧实现参考，但新的启动入口是 `webapp.main:app`。
- 若 Neo4j 不可用，检索会退化；若图像模型不可用，多模态效果会下降。
- 评估系统使用 RAGAS 0.2.10 和 datasets 库，确保 `requirements.txt` 已安装。
