"""RAG 评估模块。

本模块提供基于 Ragas 的自动化评估能力，支持：
- 实时数据采集（每次问答自动记录）
- 离线批量评估（检索效果、生成质量、性能指标）
- LLM-as-a-Judge 自动标注（无需人工测试集）

主要组件：
- recorder: 数据采集与存储
- evaluator: Ragas 指标计算（faithfulness, answer_relevancy, precision@k, recall@k）
- performance_analyzer: 性能指标分析（Latency, TTFT）
"""

__version__ = "0.1.0"
