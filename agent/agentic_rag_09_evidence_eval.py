import os                  #这次实现了context+evident（这部分存在1content的重复起始可以直接返回evident即可）和queation双保险比对的方式评估是否信息足够，要不要继续检索。比之前仅靠context的方式国家可靠

# 必须在导入 transformers / sentence_transformers 之前设置！
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import sys

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from typing import TypedDict

from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

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
# 2. Retrieval Result Contract
# ============================================================

class Evidence(TypedDict):
    content: str
    source: str
    sheet: str
    chunk_id: int


class RetrievalResult(TypedDict):
    context: str
    evidence: list[Evidence]


# ============================================================
# 3. Evidence-aware Retrieval Tool
# ============================================================

@tool
def retrieve_knowledge(query: str) -> RetrievalResult:
    """
    使用完整 MyRAG 检索知识，
    同时返回 Context 和 Evidence。
    """

    print("\n===== MyRAG Retrieval =====")
    print("Query:")
    print(query)

    result = retrieve_context(query)

    context = result["context"]
    final_docs = result["final_docs"]

    evidence = []

    for doc in final_docs:

        metadata = doc.metadata

        evidence.append(
            {
                "content": doc.page_content,
                "source": metadata.get("source", ""),
                "sheet": metadata.get("sheet", ""),
                "chunk_id": metadata.get("chunk_id", -1),
            }
        )

    return {
        "context": context,
        "evidence": evidence,
    }


# ============================================================
# 4. Evidence Evaluator Decision
# ============================================================

class EvidenceDecision(BaseModel):

    sufficient: bool = Field(
        description="当前证据是否足以支撑回答用户问题"
    )

    reason: str = Field(
        description="说明证据为什么足够或为什么不足"
    )


evaluator = llm.with_structured_output(
    EvidenceDecision,
    method="json_mode"
)


# ============================================================
# 5. Agent
# ============================================================

agent = llm.bind_tools(
    [retrieve_knowledge]
)


# ============================================================
# 6. User Question
# ============================================================

question = "赵雪丽是谁"


messages = [
    {
        "role": "system",
        "content": """
你是一个 Evidence-aware Agentic RAG Agent。

你的任务是使用知识库回答用户问题。

如果缺少可靠知识库信息，就调用 retrieve_knowledge。

retrieve_knowledge 返回：

1. context
   - 检索得到的知识内容。

2. evidence
   - 支撑这些内容的原始证据信息。
   - 包含 source、sheet、chunk_id。

注意：

不能因为 context 中出现了相关文字，
就直接认为证据足够。

需要根据当前证据判断：
这些证据是否真正包含回答用户问题所需要的事实。

不要自己模拟知识库。
"""
    },
    {
        "role": "user",
        "content": question
    }
]


# ============================================================
# 7. Agentic RAG Loop
# ============================================================

while True:

    # --------------------------------------------------------
    # Agent
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
    # Final Answer
    # --------------------------------------------------------

    if not response.tool_calls:

        print("\n==============================")
        print("最终答案")
        print("==============================")

        print(response.content)

        break

    # --------------------------------------------------------
    # 保存 Tool Call
    # --------------------------------------------------------

    messages.append(response)

    # --------------------------------------------------------
    # Retrieval
    # --------------------------------------------------------

    latest_result = None

    for tool_call in response.tool_calls:

        result = retrieve_knowledge.invoke(
            tool_call["args"]
        )

        latest_result = result

        messages.append(
            {
                "role": "tool",
                "content": str(result),
                "tool_call_id": tool_call["id"]
            }
        )

    # ========================================================
    # Evidence Evaluation
    # ========================================================

    evidence_text = ""

    for item in latest_result["evidence"]:

        evidence_text += f"""
Source: {item["source"]}
Sheet: {item["sheet"]}
Chunk ID: {item["chunk_id"]}

Content:
{item["content"]}

------------------------------
"""

    evaluation_prompt = f"""
你是 Evidence Evaluator。

你的任务不是判断“有没有相关文字”。

你的任务是判断：

当前 Evidence 是否真正足以支撑回答用户问题。

用户问题：
{question}

当前 Context：
{latest_result["context"]}

当前 Evidence：
{evidence_text}

请检查：

1. Evidence 是否包含回答问题所需要的关键事实？
2. Evidence 中的事实是否与用户问题直接相关？
3. 是否存在明显的信息缺口？
4. 如果最终回答使用这些证据，是否能够得到有依据的答案？

如果证据足够：

sufficient = true

如果证据不足：

sufficient = false

reason 必须明确说明判断依据。

请严格输出 JSON。
"""

    decision = evaluator.invoke(
        evaluation_prompt
    )

    print("\n===== Evidence Evaluator =====")

    print(
        "sufficient =",
        decision.sufficient
    )

    print(
        "reason =",
        decision.reason
    )

    # ========================================================
    # 将 Evidence Evaluation 结果交给 Agent
    # ========================================================

    if decision.sufficient:

        messages.append(
            {
                "role": "system",
                "content": (
                    "Evidence Evaluator 判断："
                    "当前证据足以支撑回答。\n"
                    f"原因：{decision.reason}\n"
                    "现在可以生成最终答案。"
                )
            }
        )

    else:

        messages.append(
            {
                "role": "system",
                "content": (
                    "Evidence Evaluator 判断："
                    "当前证据不足以支撑回答。\n"
                    f"原因：{decision.reason}\n\n"
                    "必须重新设计 Query，"
                    "然后再次调用 retrieve_knowledge。"
                )
            }
        )