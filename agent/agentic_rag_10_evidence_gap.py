import os               #这部分是弥补之前只靠是否足够这点消息进行问题重写的短板，采用证据导向的问题重写机制，加入missing evidence直接指出下一步据点检索哪几项缺少的证据

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
from langchain_core.messages import ToolMessage

from retrieval_chain import retrieve_context


# ============================================================
# 1. LLM
# ============================================================

llm = ChatOpenAI(
    model="deepseek-v4-flash",
    api_key=os.environ["ANTHROPIC_AUTH_TOKEN"],
    base_url="https://api.deepseek.com"
)


# ============================================================
# 2. Evidence Contract
# ============================================================

class Evidence(TypedDict):

    source: str
    sheet: str
    chunk_id: int


class RetrievalResult(TypedDict):

    context: str
    evidence: list[Evidence]


# ============================================================
# 3. Retrieval Tool
# ============================================================

@tool
def retrieve_knowledge(query: str) -> RetrievalResult:
    """
    使用完整 MyRAG 检索知识，
    返回 Context + Evidence。
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
                "source": metadata.get("source", ""),
                "sheet": metadata.get("sheet", ""),
                "chunk_id": metadata.get("chunk_id", -1),
            }
        )

    print("\n===== Retrieved Context =====")
    print(context)

    print("\n===== Evidence =====")

    for item in evidence:

        print(
            f"source={item['source']}, "
            f"sheet={item['sheet']}, "
            f"chunk_id={item['chunk_id']}"
        )

    return {
        "context": context,
        "evidence": evidence,
    }


# ============================================================
# 4. Evidence Evaluation Contract
# ============================================================

class EvidenceDecision(BaseModel):

    sufficient: bool = Field(
        description="当前证据是否足以回答用户问题"
    )

    reason: str = Field(
        description="判断当前证据是否足够的原因"
    )

    missing_evidence: str = Field(
        description="如果证据不足，明确说明缺少什么事实；如果足够则填写空字符串"
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

context：
检索得到的知识内容。

evidence：
这些内容对应的来源定位信息，
包括 source、sheet、chunk_id。

工作原则：

1. 首先判断是否需要检索。
2. 检索后不能只看文字是否相关。
3. 必须关注当前证据是否能够支撑回答。
4. 如果证据不足，Evaluator 会告诉你缺少什么。
5. 如果需要再次检索，下一次 Query 必须针对缺失证据设计。
6. 不要重复使用完全相同的 Query。
7. 如果证据已经足够，直接回答用户。
8. 不要自己模拟知识库。

现在执行任务。
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
    # Agent Decision
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
    # 保存 Agent Tool Call
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
            ToolMessage(
                content=str(result),
                tool_call_id=tool_call["id"]
            )
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

"""

    evaluation_prompt = f"""
你是一个 Evidence Evaluator。

用户问题：
{question}

当前检索内容：
{latest_result["context"]}

当前证据：
{latest_result["evidence"]}

请判断当前证据是否足以回答用户问题。

必须返回 JSON，且必须包含以下三个字段：

{{
    "sufficient": true 或 false,
    "reason": "判断理由",
    "missing_evidence": "如果证据不足，明确说明缺少什么事实；如果证据足够，则填写空字符串"
}}

字段要求：

1. sufficient
   - true：当前证据足以回答问题
   - false：当前证据不足

2. reason
   - 必须说明为什么当前证据足够或不足
   - 不能为空

3. missing_evidence
   - sufficient=false 时，必须明确指出缺少的事实
   - sufficient=true 时，必须填写 ""

只返回 JSON，不要输出其他内容。
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

    print(
        "missing_evidence =",
        decision.missing_evidence
    )

    # ========================================================
    # Feed Evaluation Back To Agent
    # ========================================================

    if decision.sufficient:

        messages.append(
            {
                "role": "system",
                "content": (
                    "Evidence Evaluator 判断："
                    "当前证据已经足够。\n"
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
                    "Evidence Evaluator 判断："
                    "当前证据不足。\n"
                    f"原因：{decision.reason}\n"
                    f"缺少证据：{decision.missing_evidence}\n\n"
                    "下一次检索必须针对上述缺失证据重新设计 Query。"
                    "不要重复之前的 Query。"
                )
            }
        )