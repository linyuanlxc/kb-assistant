from langchain_core.prompts import ChatPromptTemplate

# 步骤1：定义模板（角色+内容模板，支持占位符）
prompt = ChatPromptTemplate.from_messages(
    [
        # 消息元组：(角色类型, 内容模板字符串)
        ("system", "你是一个严谨的知识库问答助手，仅根据{context}回答用户问题，禁止编造信息。"),
        ("user", "我的问题是：{question}"),
        # 可选：添加历史助手回答（多轮对话用）
        ("assistant", "{history_answer}"),
    ]
)

# 步骤2：填充占位符（3种常用方式）
## 方式A：invoke()（推荐，返回ChatPromptValue对象，可直接传给Chat Model）
prompt_value = prompt.invoke({
    "context": "南瓜书是《机器学习公式详解》的俗称",
    "question": "南瓜书是什么？",
    "history_answer": ""  # 无历史时传空字符串
})

## 方式B：format()（返回纯文本字符串，调试用）
prompt_text = prompt.format(
    context="南瓜书是《机器学习公式详解》的俗称",
    question="南瓜书是什么？",
    history_answer=""
)

## 方式C：format_messages()（返回消息列表，底层调用）
messages = prompt.format_messages(
    context="南瓜书是《机器学习公式详解》的俗称",
    question="南瓜书是什么？",
    history_answer=""
)

# 步骤3：输出结果查看
print("=== ChatPromptValue对象 ===")
print(prompt_value)
print("\n=== 纯文本格式 ===")
print(prompt_text)
print("\n=== 消息列表格式 ===")
print(messages)