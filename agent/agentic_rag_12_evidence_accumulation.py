import os                ##这一步解决了之前评估llm只能看到最后一次检索的上下文和证据的去电，把历次检索的上下文和证据都给评估llm了，注意主管llm负责决策本来就有产看安居多次去状态的能力

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
# 2. Evidence
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
    从学校知识库检索信息，并返回：
    context：检索到的文本
    evidence：对应的来源位置
    """

    print("\n" + "=" * 70)
    print("Retrieval Tool")
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

    print("\n----- 当前 Retrieval Context -----")
    print(context)

    print("\n----- 当前 Retrieval Evidence -----")

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
        description="当前累计证据是否足以回答用户问题"
    )

    reason: str = Field(
        description="判断当前累计证据是否足够的原因"
    )

    missing_evidence: str = Field(
        description=(
            "如果证据不足，说明还缺少什么事实；"
            "如果证据足够，填写空字符串"
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
# 6. User Question
# ============================================================

question = "赵雪丽是谁"


# ============================================================
# 7. Search Budget
# ============================================================

MAX_RETRIEVALS = 3

retrieval_count = 0


# ============================================================
# 8. Evidence Accumulation
# ============================================================

all_context = []

all_evidence = []


# ============================================================
# 9. Message
# ============================================================

messages = [
    {
        "role": "system",
        "content": """
你是一个学校知识库 Agent。

你的任务是回答用户问题。

规则：

1. 如果没有可靠信息，调用 retrieve_knowledge。
2. 检索后必须判断当前累计证据是否足以回答问题。
3. 每次新的检索结果都会加入历史证据。
4. 不允许因为进行了新的检索，就忘记之前已经获得的证据。
5. 如果证据不足，必须根据 missing_evidence 设计下一次 Query。
6. 不要重复完全相同的 Query。
7. 如果系统告诉你检索预算已经耗尽，禁止继续调用工具。
8. 如果证据不足但已经无法继续检索，必须明确说明信息不足。
9. 禁止编造知识库中不存在的信息。

目标：
使用尽可能少的检索获得足够证据。
"""
    },
    {
        "role": "user",
        "content": question
    }
]


# ============================================================
# 10. Agentic RAG Loop
# ============================================================

while True:

    print("\n")
    print("#" * 70)
    print("Agent Loop")
    print(f"Retrieval Count: {retrieval_count}/{MAX_RETRIEVALS}")
    print(f"累计 Context 数量: {len(all_context)}")
    print(f"累计 Evidence 数量: {len(all_evidence)}")
    print("#" * 70)

    response = agent.invoke(messages)

    # --------------------------------------------------------
    # Agent 直接回答
    # --------------------------------------------------------

    if not response.tool_calls:

        print("\n" + "=" * 70)
        print("最终答案")
        print("=" * 70)

        print(response.content)

        break

    messages.append(response)

    # --------------------------------------------------------
    # Search Budget
    # --------------------------------------------------------

    if retrieval_count >= MAX_RETRIEVALS:

        print("\n" + "!" * 70)
        print("Search Budget 已耗尽")
        print("禁止继续 Retrieval")
        print("!" * 70)

        messages.append({
            "role": "system",
            "content": """
检索预算已经耗尽。

禁止再次调用 retrieve_knowledge。

请使用目前已经累计获得的全部 Context 和 Evidence 回答问题。

如果现有证据不足以可靠回答，请明确说明信息不足。

禁止编造事实。
"""
        })

        continue

    # --------------------------------------------------------
    # Execute Tool
    # --------------------------------------------------------

    for tool_call in response.tool_calls:

        tool_name = tool_call["name"]

        if tool_name not in tool_map:
            raise ValueError(
                f"未知 Tool：{tool_name}"
            )

        result = tool_map[tool_name].invoke(
            tool_call["args"]
        )

        # ====================================================
        # 累积 Context
        # ====================================================

        current_context = result["context"]

        if current_context:
            all_context.append(current_context)

        # ====================================================
        # 累积 Evidence
        # ====================================================

        for evidence in result["evidence"]:

            if evidence not in all_evidence:
                all_evidence.append(evidence)

        # ====================================================
        # ToolMessage
        # ====================================================

        messages.append(
            ToolMessage(
                content=str(result),
                tool_call_id=tool_call["id"]
            )
        )

        retrieval_count += 1

    # ========================================================
    # 构建累计 Context
    # ========================================================

    accumulated_context = "\n\n".join(
        all_context
    )

    # ========================================================
    # 构建累计 Evidence
    # ========================================================

    accumulated_evidence = "\n".join(
        str(item)
        for item in all_evidence
    )

    # ========================================================
    # 打印累计状态
    # ========================================================

    print("\n" + "-" * 70)
    print("累计 Evidence")
    print("-" * 70)

    for item in all_evidence:
        print(item)

    print("\n" + "-" * 70)
    print(f"累计 Evidence 总数：{len(all_evidence)}")
    print("-" * 70)

    # ========================================================
    # Evidence Evaluation
    # ========================================================

    evaluation_prompt = f"""
请判断当前累计证据是否足以回答用户问题。

用户问题：

{question}


========================
累计 Context
========================

{accumulated_context}


========================
累计 Evidence
========================

{accumulated_evidence}


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
    print("\n" + "-" * 70)
    print("Evidence Evaluator")
    print("-" * 70)

    decision = evaluator.invoke(
        evaluation_prompt
    )

    print(f"\nsufficient: {decision.sufficient}")
    print(f"reason: {decision.reason}")
    print(
        f"missing_evidence: "
        f"{decision.missing_evidence}"
    )

    # ========================================================
    # Evidence 足够
    # ========================================================

    if decision.sufficient:

        print("\n" + "=" * 70)
        print("Evidence 足够")
        print("=" * 70)

        messages.append({
            "role": "system",
            "content": f"""
Evidence Evaluator 判断当前累计证据已经足够。

原因：

{decision.reason}

现在请直接回答用户问题。

必须使用目前累计获得的全部 Context。

不要再次调用 retrieve_knowledge。

不要编造知识库中不存在的信息。
"""
        })

    # ========================================================
    # Evidence 不足
    # ========================================================

    else:

        # ----------------------------------------------------
        # Budget 已耗尽
        # ----------------------------------------------------

        if retrieval_count >= MAX_RETRIEVALS:

            print("\n" + "!" * 70)
            print("累计证据仍不足")
            print("Search Budget 已耗尽")
            print("强制结束 Retrieval")
            print("!" * 70)

            messages.append({
                "role": "system",
                "content": f"""
当前累计证据仍然不足。

原因：

{decision.reason}

缺少：

{decision.missing_evidence}

但是检索预算已经耗尽。

禁止再次调用 retrieve_knowledge。

请基于目前累计获得的全部证据回答。

如果无法可靠回答，请明确说明：
“知识库现有证据不足，无法确认该信息。”

禁止编造事实。
"""
            })

        # ----------------------------------------------------
        # Budget 尚未耗尽
        # ----------------------------------------------------

        else:

            print("\n" + ">" * 70)
            print("累计证据不足")
            print("准备继续检索")
            print(f"缺少证据：{decision.missing_evidence}")
            print(">" * 70)

            messages.append({
                "role": "system",
                "content": f"""
当前累计证据不足。

原因：

{decision.reason}

缺少的关键证据：

{decision.missing_evidence}

请根据缺失证据重新设计下一次 Query。

要求：

1. Query 必须针对缺失证据。
2. 不要重复完全相同的 Query。
3. 保留并利用之前已经获得的证据。
4. 下一次 Retrieval 的目标是补齐当前 Evidence Gap。
"""
            })