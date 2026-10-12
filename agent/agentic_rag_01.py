import os                     #仅测试是否具有调用检索器的能力

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool


# ============================================================
# 1. LLM
# ============================================================

llm = ChatOpenAI(
    model="deepseek-v4-flash",
    api_key=os.environ["ANTHROPIC_AUTH_TOKEN"],
    base_url="https://api.deepseek.com"
)


# ============================================================
# 2. Retrieval Tool
# ============================================================

@tool
def retrieve_knowledge(query: str) -> str:
    """
    从知识库中检索与问题相关的信息。
    """

    print(f"\n===== Retrieval =====")
    print(f"检索问题：{query}")

    # 暂时用假的知识库模拟真实 Retrieval
    knowledge = {
        "人工智能教育": (
            "人工智能可以应用于个性化学习、智能教学、"
            "学习分析和教育资源管理。"
        ),
        "人工智能风险": (
            "人工智能教育应用需要关注数据隐私、"
            "算法可靠性以及教师监督。"
        )
    }

    for key, value in knowledge.items():#这里的items()是把字典键值对变成元组放进列表的工具，无他单纯遍历字典，仅仅可以遍历KEY

        if key in query:
            return value

    return "知识库中没有找到相关信息。"


# ============================================================
# 3. Agent
# ============================================================

tools = [retrieve_knowledge]

agent = llm.bind_tools(tools)


# ============================================================
# 4. 测试
# ============================================================

question = "人工智能在教育领域有哪些应用？"

messages = [
    {
        "role": "system",
        "content": """
你是一个 Agentic RAG Agent。

你需要先判断自己是否拥有回答用户问题所需的信息。

如果你缺少可靠信息：
必须调用 retrieve_knowledge。

如果你已经拥有足够信息：
可以直接回答，不需要检索。

不要为了形式而检索。
"""
    },
    {
        "role": "user",
        "content": question
    }
]


response = agent.invoke(messages)

print("\n===== Agent 第一次决策 =====")
print(response)

print("\n===== Tool Calls =====")
print(response.tool_calls)