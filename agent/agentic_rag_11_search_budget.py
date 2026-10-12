import os             #这里增加的agent系统安全层面的限制（最大检索次数防止无限检索），采用双校验的方式（在每次使用完llm后）确保agent稳定输出

# 必须在导入 transformers / sentence_transformers 之前设置
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
# 2. Evidence
#    只保存证据来源，不重复保存 content
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
    根据查询从学校知识库中检索相关知识。
    返回：
    - context：实际检索到的内容
    - evidence：这些内容对应的来源位置
    """

    print("\n" + "=" * 70)
    print(f"执行 Retrieval Tool")
    print(f"Query: {query}")
    print("=" * 70)

    result = retrieve_context(query)

    context = result["context"]
    final_docs = result["final_docs"]

    evidence = []

    for doc in final_docs:
        metadata = doc.metadata

        evidence.append({
            "source": metadata.get("source", ""),
            "sheet": metadata.get("sheet", ""),
            "chunk_id": metadata.get("chunk_id", -1)
        })

    print("\n----- Context -----")
    print(context)

    print("\n----- Evidence -----")
    for item in evidence:
        print(item)

    return {
        "context": context,
        "evidence": evidence
    }


# ============================================================
# 4. Evidence Evaluator
# ============================================================

class EvidenceDecision(BaseModel):
    sufficient: bool = Field(
        description="当前证据是否足以可靠回答用户问题"
    )

    reason: str = Field(
        description="判断当前证据是否足够的原因"
    )

    missing_evidence: str = Field(
        description=(
            "如果证据不足，明确说明还缺少什么事实；"
            "如果证据足够，则填写空字符串"
        )
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

tool_map = {
    "retrieve_knowledge": retrieve_knowledge
}


# ============================================================
# 6. 用户问题
# ============================================================

question = "邓开平是谁"


# ============================================================
# 7. Search Budget
# ============================================================

MAX_RETRIEVALS = 3

retrieval_count = 0


# ============================================================
# 8. Agent Messages
# ============================================================

messages = [
    {
        "role": "system",
        "content": """
你是一个学校知识库 Agent。

你的任务是回答用户问题。

规则：

1. 如果当前没有可靠信息，调用 retrieve_knowledge。
2. 检索后需要根据检索结果判断是否已经获得足够信息。
3. 如果证据不足，不要盲目重复相同 Query。
4. 必须根据缺失的证据重新设计下一次 Query。
5. 如果系统明确告诉你检索预算已经耗尽，禁止再次调用 retrieve_knowledge。
6. 如果证据不足但无法继续检索，则必须基于现有证据回答，并明确说明哪些信息无法确认。
7. 不要编造知识库中不存在的信息。

你的目标不是尽可能多检索，而是用尽可能少的检索获得足够证据。
"""
    },
    {
        "role": "user",
        "content": question
    }
]


# ============================================================
# 9. Agentic RAG Loop
# ============================================================

while True:

    print("\n" + "#" * 70)
    print(f"Agent Loop")
    print(f"当前检索次数：{retrieval_count}/{MAX_RETRIEVALS}")
    print("#" * 70)

    response = agent.invoke(messages)

    # --------------------------------------------------------
    # Agent 不再调用工具
    # --------------------------------------------------------

    if not response.tool_calls:

        print("\n" + "=" * 70)
        print("最终答案")
        print("=" * 70)

        print(response.content)

        break

    # 保存 Agent 的 tool call
    messages.append(response)

    # --------------------------------------------------------
    # Search Budget 检查
    # --------------------------------------------------------

    if retrieval_count >= MAX_RETRIEVALS:                #这是规则层面的硬约束，如果llm在预算耗尽后还执行工具调用那么会直接失败

        print("\n" + "!" * 70)
        print("Search Budget 已耗尽")
        print("禁止继续 Retrieval")
        print("!" * 70)

        messages.append({
            "role": "system",
            "content": """
检索次数已经达到最大预算。

禁止再次调用 retrieve_knowledge。

请立即基于当前已经获得的信息回答用户问题。

如果现有证据不足以可靠回答，请明确说明：
“知识库现有证据不足，无法确认该信息。”

禁止编造不存在的事实。
"""
        })

        continue

    # --------------------------------------------------------
    # 执行 Tool Calls
    # --------------------------------------------------------

    latest_result = None

    for tool_call in response.tool_calls:

        tool_name = tool_call["name"]

        if tool_name not in tool_map:
            raise ValueError(
                f"未知 Tool: {tool_name}"
            )

        result = tool_map[tool_name].invoke(
            tool_call["args"]
        )

        latest_result = result

        messages.append(
            ToolMessage(
                content=str(result),
                tool_call_id=tool_call["id"]
            )
        )

        retrieval_count += 1

    # --------------------------------------------------------
    # 如果没有成功得到检索结果
    # --------------------------------------------------------

    if latest_result is None:
        continue

    # ========================================================
    # Evidence Evaluator
    # ========================================================

    print("\n" + "-" * 70)
    print("Evidence Evaluator")
    print("-" * 70)

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

    print(f"\nsufficient: {decision.sufficient}")
    print(f"reason: {decision.reason}")
    print(f"missing_evidence: {decision.missing_evidence}")

    # ========================================================
    # Evidence 足够
    # ========================================================

    if decision.sufficient:

        messages.append({
            "role": "system",
            "content": f"""
Evidence Evaluator 判断当前证据已经足够。

原因：
{decision.reason}

请直接基于当前检索结果回答用户问题。

不要再次调用 retrieve_knowledge。
不要编造检索结果中不存在的信息。
"""
        })

    # ========================================================
    # Evidence 不足
    # ========================================================

    else:

        if retrieval_count >= MAX_RETRIEVALS:                #这是告诉llm停止检索是语义层面的软约束

            print("\n" + "!" * 70)
            print("证据仍不足，但 Search Budget 已耗尽")
            print("强制终止检索")
            print("!" * 70)

            messages.append({
                "role": "system",
                "content": f"""
当前证据仍然不足。

Evidence Evaluator 判断：
{decision.reason}

缺少的证据：
{decision.missing_evidence}

但是检索预算已经耗尽。

禁止继续调用 retrieve_knowledge。

请基于目前已有证据回答。
如果无法可靠回答，请明确说明信息不足。
禁止编造事实。
"""
            })

        else:

            print("\n" + ">" * 70)
            print("Evidence 不足")
            print(f"缺少证据：{decision.missing_evidence}")
            print("允许继续检索")
            print(">" * 70)

            messages.append({
                "role": "system",
                "content": f"""
Evidence Evaluator 判断当前证据不足。

原因：
{decision.reason}

缺少的关键证据：
{decision.missing_evidence}

请根据“缺少的证据”重新设计下一次 Query。

要求：

1. 下一次 Query 必须针对 missing_evidence。
2. 不要重复完全相同的 Query。
3. 不要为了增加结果数量而盲目检索。
4. 必须通过下一次检索尝试补齐缺失事实。
"""
            })