import os           #在循环跑通的基础上引入了问题重写机制（Agent层面与RAG工具层面是两个层级）

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

@tool                                      #这一步就是结构化工具，用于llm使用工具时触发原生工具调用
def retrieve_knowledge(query: str) -> str:
    """
    根据查询从知识库中检索相关信息。
    """

    print("\n===== Retrieval Tool =====")
    print(f"Query：{query}")

    # --------------------------------------------------------
    # 第一次查询：故意返回不完整信息
    # --------------------------------------------------------

    if query == "人工智能在教育领域的应用":

        return (
            "人工智能可以应用于个性化学习和智能教学。"
        )

    # --------------------------------------------------------
    # 查询更加具体时，返回更多信息
    # --------------------------------------------------------

    if "学习分析" in query or "教育资源管理" in query:

        return (
            "人工智能在教育领域可以用于个性化学习、"
            "智能教学、学习分析以及教育资源管理。"
        )

    return (
        "没有找到足够相关的信息。"
    )


# ============================================================
# 3. Agent
# ============================================================

agent = llm.bind_tools(              ##创建带有工具的llm为实际实现agent loop的决策层和执行层做准备
    [retrieve_knowledge]
)


tool_map = {                         #方便后续调用，注册表
    "retrieve_knowledge": retrieve_knowledge
}


# ============================================================
# 4. Agent Loop
# ============================================================

messages = [                  #发给llm的请求体内部的一部分数据此外还有挂载的工具...其实这部分就相当于prompt
    {
        "role": "system",
        "content": """
你是一个 Agentic RAG Agent。

你的任务是根据知识库信息回答用户问题。

工作原则：

1. 如果没有足够信息，调用 retrieve_knowledge。
2. 获取检索结果后，检查结果是否足够回答用户。
3. 如果结果不完整：
   - 不要立即回答；
   - 思考缺少什么信息；
   - 修改查询 Query；
   - 再次调用 retrieve_knowledge。
4. 如果信息已经足够，再生成最终答案。
5. 不要重复使用完全相同的 Query。
"""
    },
    {
        "role": "user",
        "content": "人工智能在教育领域有哪些应用？"
    }
]


while True:

    # ========================================================
    # Agent 思考
    # ========================================================

    response = agent.invoke(messages)            ##调用llm也就是实际实现agent loop的决策层和执行层

    print("\n===== Agent =====")

    if response.tool_calls:                   #根据是否有工具判断下一步是结束还是调用工具

        print("Agent 决定继续检索：")
        print(response.tool_calls)

    else:

        print("Agent 决定结束。")
        print(response.content)

    # ========================================================
    # 情况 1：没有 Tool Call
    # ========================================================

    if not response.tool_calls:

        print("\n==============================")
        print("最终答案")
        print("==============================")

        print(response.content)

        break

    # ========================================================
    # 情况 2：存在 Tool Call
    # ========================================================

    messages.append(response)

    for tool_call in response.tool_calls:               #实际工具解析和工具调用

        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_call_id = tool_call["id"]

        result = tool_map[tool_name].invoke(
            tool_args
        )

        messages.append(
            ToolMessage(
                content=result,
                tool_call_id=tool_call_id
            )
        )