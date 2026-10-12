import os                                      ##########此段代码可能经常报错，因为计算机显存太小调用RAG容易爆显存
# 必须在导入 transformers / sentence_transformers 之前设置！
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


import sys       
import os

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)
      #这是针对group(类)的检索处理上下文管理



from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage

from retrieval_chain_group import retrieve_context


# ============================================================
# 1. LLM
# ============================================================

llm = ChatOpenAI(
    model="deepseek-v4-flash",
    api_key=os.environ["ANTHROPIC_AUTH_TOKEN"],
    base_url="https://api.deepseek.com"
)


# ============================================================
# 2. Real MyRAG Retrieval Tool
# ============================================================

@tool
def retrieve_knowledge(query: str) -> str:
    """
    使用 MyRAG 完整检索链检索知识库。
    """

    print("\n===== MyRAG Retrieval =====")
    print("Query:")
    print(query)

    result = retrieve_context(query)

    context = result["context"]

    print("\n===== Retrieved Context =====")
    print(context)

    return context


# ============================================================
# 3. Retrieval Evaluator
# ============================================================

class RetrievalDecision(BaseModel):

    sufficient: bool = Field(
        description="当前检索结果是否足以回答用户问题"
    )

    reason: str = Field(
        description="判断检索结果是否足够的原因"
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

question = "赵雪丽是谁"


messages = [
    {
        "role": "system",
        "content": """
你是一个 Agentic RAG Agent。

你的任务是使用知识库回答用户问题。

工作流程：

1. 如果缺少可靠信息，调用 retrieve_knowledge。
2. 获取检索结果后，判断结果是否足够。
3. 如果结果不足：
   - 分析缺少什么；
   - 修改 Query；
   - 再次调用 retrieve_knowledge。
4. 如果结果足够，生成最终回答。
5. 不要重复使用完全相同的 Query。
6. 不要因为检索到了部分信息就立即结束。

注意：

retrieve_knowledge 已经封装了完整的 MyRAG 检索流程。
不要自己模拟知识库。
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
    # Agent 决策
    # --------------------------------------------------------

    response = agent.invoke(messages)

    print("\n===== Agent =====")

    if response.tool_calls:

        print("Agent → Retrieval")

        for tool_call in response.tool_calls:
            print(
                "Query:",
                tool_call["args"]["query"]
            )

    else:

        print("Agent → Final Answer")

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
    # 执行 MyRAG Retrieval
    # --------------------------------------------------------

    latest_result = ""

    for tool_call in response.tool_calls:

        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_call_id = tool_call["id"]

        result = tool_map[tool_name].invoke(
            tool_args
        )

        latest_result = result

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

当前检索结果：
{latest_result}

判断当前检索结果是否已经足以回答用户问题。

如果已经包含回答问题所需要的主要信息：

sufficient = true

如果缺少关键事实：

sufficient = false

reason 必须说明判断依据。

请严格输出 JSON。
"""

    decision = evaluator.invoke(
        evaluation_prompt
    )

    print("\n===== Evaluator =====")
    print("sufficient =", decision.sufficient)
    print("reason =", decision.reason)

    # --------------------------------------------------------
    # 把评估结果交给 Agent
    # --------------------------------------------------------

    if decision.sufficient:

        messages.append(
            {
                "role": "system",
                "content": (
                    "Evaluator 判断：当前检索结果已经足够。\n"
                    f"原因：{decision.reason}\n"
                    "现在可以直接生成最终答案。"
                )
            }
        )

    else:

        messages.append(
            {
                "role": "system",
                "content": (
                    "Evaluator 判断：当前检索结果不足。\n"
                    f"原因：{decision.reason}\n\n"
                    "下一步必须重新设计 Query，"
                    "然后再次调用 retrieve_knowledge。"
                )
            }
        )