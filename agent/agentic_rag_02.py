import os                           #测试他是否可以调用检索器返回结构重新思考loop

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage


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

    print("\n===== Retrieval Tool =====")
    print(f"检索问题：{query}")

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

    for key, value in knowledge.items():
        if key in query:
            return value

    return "知识库中没有找到相关信息。"


# ============================================================
# 3. Agent
# ============================================================

tools = [retrieve_knowledge]

tool_map = {
    "retrieve_knowledge": retrieve_knowledge
}

agent = llm.bind_tools(tools)


# ============================================================
# 4. Agent Loop
# ============================================================

question = "人工智能教育领域有哪些应用？"

messages = [
    {
        "role": "system",
        "content": """
你是一个 Agentic RAG Agent。

如果你缺少回答问题所需的可靠信息，
调用 retrieve_knowledge。

获得检索结果后，根据检索结果回答用户。

如果已经拥有足够信息，则直接回答。

不要为了形式而重复检索。
"""
    },
    {
        "role": "user",
        "content": question
    }
]


while True:

    response = agent.invoke(messages)

    print("\n===== Agent =====")
    print(response)

    # --------------------------------------------------------
    # 没有 Tool Call
    # → Agent 已经决定可以直接回答
    # --------------------------------------------------------

    if not response.tool_calls:

        print("\n===== 最终回答 =====")
        print(response.content)

        break

    # --------------------------------------------------------
    # 有 Tool Call
    # → 执行 Tool
    # --------------------------------------------------------

    messages.append(response)

    for tool_call in response.tool_calls:

        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_call_id = tool_call["id"]

        print(
            f"\nAgent → Tool: "
            f"{tool_name}({tool_args})"
        )

        result = tool_map[tool_name].invoke(tool_args)

        messages.append(
            ToolMessage(
                content=result,
                tool_call_id=tool_call_id
            )
        )