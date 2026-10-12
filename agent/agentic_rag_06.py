import os                                   #真正将rag封装成工具被阿根廷调用           这里的重写和评估调用了两次llm     其中重写设备融合到了决策层负责重写加返回原生工具调用

from pydantic import BaseModel, Field              #决策层重写加决策---执行层执行执行工具---评估层按契约评估放入message列表里面---重写加决策---

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
    从知识库中检索与查询相关的信息。
    """

    print("\n===== Retrieval Tool =====")
    print("Query:", query)

    # 第一次查询：故意返回不完整信息
    if query == "人工智能在教育领域有哪些应用？":

        return (
            "人工智能可以应用于个性化学习和智能教学。"
        )

    # 第二次查询：返回更加完整的信息
    return (
        "人工智能在教育领域可以应用于："
        "个性化学习、智能教学、学习分析、"
        "教育资源管理。"
    )


# ============================================================
# 3. Retrieval Evaluation
# ============================================================

class RetrievalDecision(BaseModel):

    sufficient: bool = Field(
        description="当前检索结果是否足以回答用户问题"
    )

    reason: str = Field(
        description="判断当前检索结果是否足够的原因"
    )


evaluator = llm.with_structured_output(
    RetrievalDecision,
    method="json_mode"
)


# ============================================================
# 4. Agent
# ============================================================

agent = llm.bind_tools(
    [retrieve_knowledge]
)


tool_map = {
    "retrieve_knowledge": retrieve_knowledge
}


# ============================================================
# 5. User Question
# ============================================================

question = "人工智能在教育领域有哪些应用？"


messages = [
    {
        "role": "system",
        "content": """
你是一个 Agentic RAG Agent。

你的任务是使用知识库回答用户问题。

工作流程：

1. 如果缺少信息，调用 retrieve_knowledge。
2. 获取检索结果后，判断结果是否足够。
3. 如果结果不足：
   - 分析缺少什么信息；
   - 修改 Query；
   - 再次调用 retrieve_knowledge。
4. 如果结果足够，直接生成最终回答。
5. 不要重复使用完全相同的 Query。
6. 不要因为已经获得了一部分信息就立即结束。
"""
    },
    {
        "role": "user",
        "content": question
    }
]


# ============================================================
# 6. Agentic RAG Loop
# ============================================================

while True:

    # --------------------------------------------------------
    # Agent
    # --------------------------------------------------------

    response = agent.invoke(messages)

    print("\n===== Agent =====")

    if response.tool_calls:

        print("Agent 决定调用 Retrieval：")
        print(response.tool_calls)

    else:

        print("Agent 决定结束。")

    # --------------------------------------------------------
    # 没有 Tool Call
    # --------------------------------------------------------

    if not response.tool_calls:

        print("\n==============================")
        print("最终答案")
        print("==============================")

        print(response.content)

        break

    # --------------------------------------------------------
    # 保存 Agent 的 Tool Call
    # --------------------------------------------------------

    messages.append(response)

    # --------------------------------------------------------
    # 执行 Tool
    # --------------------------------------------------------

    for tool_call in response.tool_calls:

        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_call_id = tool_call["id"]

        result = tool_map[tool_name].invoke(
            tool_args
        )

        print("\nTool 返回：")
        print(result)

        messages.append(
            ToolMessage(
                content=result,
                tool_call_id=tool_call_id
            )
        )

    # --------------------------------------------------------
    # Evaluator
    # --------------------------------------------------------

    evaluation_prompt = f"""
你是 Retrieval Evaluator。

用户问题：
{question}

最新检索结果：
{result}

判断这个检索结果是否已经足以回答用户问题。

如果足够：
sufficient = true

如果不够：
sufficient = false

reason 必须说明原因。

请严格输出 JSON。
"""

    decision = evaluator.invoke(
        evaluation_prompt
    )

    print("\n===== Evaluator =====")
    print(decision)

    # --------------------------------------------------------
    # 把 Evaluator 结果告诉 Agent
    # --------------------------------------------------------

    messages.append(
        {
            "role": "system",
            "content": (
                "最近一次 Retrieval 结果评估如下：\n"
                f"sufficient = {decision.sufficient}\n"
                f"reason = {decision.reason}\n\n"
                "如果 sufficient = false，"
                "下一步必须修改 Query 后重新调用 Retrieval。"
                if not decision.sufficient
                else
                "最近一次 Retrieval 结果已经足够，"
                "下一步可以直接生成最终答案。"
            )
        }
    )