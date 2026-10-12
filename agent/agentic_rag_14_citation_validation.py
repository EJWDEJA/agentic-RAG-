import os                    #这部分只要是对应引用证据以后的结果进行引用来源的校验看引用的证据是不是能够支持答案

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
    id: str
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
    从学校知识库检索 Context 和 Evidence。
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
            "id": "",
            "source": metadata.get("source", ""),
            "sheet": metadata.get("sheet", ""),
            "chunk_id": metadata.get(
                "chunk_id",
                -1
            )
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
# 4. Grounded Answer
# ============================================================

class GroundedAnswer(BaseModel):

    answer: str = Field(
        description="最终答案"
    )

    evidence_ids: list[str] = Field(
        description=(
            "真正支持最终答案的 Evidence ID"
        )
    )


# ============================================================
# 5. Citation Validation Result
# ============================================================

class CitationValidation(BaseModel):

    valid: bool = Field(
        description=(
            "答案中的引用是否全部有效"
        )
    )

    reason: str = Field(
        description=(
            "判断引用是否有效的原因"
        )
    )

    invalid_evidence_ids: list[str] = Field(
        description=(
            "不存在或无法支持答案的 Evidence ID"
        )
    )


# ============================================================
# 6. Structured LLM
# ============================================================

grounding_llm = llm.with_structured_output(
    GroundedAnswer,
    method="json_mode"
)

validator_llm = llm.with_structured_output(
    CitationValidation,
    method="json_mode"
)


# ============================================================
# 7. Agent
# ============================================================

agent = llm.bind_tools(
    [retrieve_knowledge]
)

tool_map = {
    "retrieve_knowledge": retrieve_knowledge
}


# ============================================================
# 8. Question
# ============================================================

question = "赵雪丽是谁"


# ============================================================
# 9. Search Budget
# ============================================================

MAX_RETRIEVALS = 3

retrieval_count = 0


# ============================================================
# 10. Accumulated Data
# ============================================================

all_context = []

all_evidence = []


# ============================================================
# 11. Messages
# ============================================================

messages = [
    {
        "role": "system",
        "content": """
你是一个学校知识库 Agent。

你的任务是回答用户问题。

规则：

1. 信息不足时调用 retrieve_knowledge。
2. 使用知识库检索结果作为事实依据。
3. 不允许编造知识库中不存在的信息。
4. 不要重复完全相同的 Query。
5. 如果检索预算耗尽，禁止继续调用工具。
6. 最终答案必须建立在检索结果之上。
"""
    },
    {
        "role": "user",
        "content": question
    }
]


# ============================================================
# 12. Agent Retrieval Loop
# ============================================================

while True:

    print("\n" + "#" * 70)
    print("Agent Loop")
    print(
        f"Retrieval Count: "
        f"{retrieval_count}/{MAX_RETRIEVALS}"
    )
    print("#" * 70)

    response = agent.invoke(messages)

    # --------------------------------------------------------
    # 没有 Tool Call
    # --------------------------------------------------------

    if not response.tool_calls:
        break

    messages.append(response)

    # --------------------------------------------------------
    # Search Budget
    # --------------------------------------------------------

    if retrieval_count >= MAX_RETRIEVALS:

        print("\n" + "!" * 70)
        print("Search Budget 已耗尽")
        print("!" * 70)

        messages.append({
            "role": "system",
            "content": """
检索预算已经耗尽。

禁止继续调用 retrieve_knowledge。

请使用当前已经获得的证据生成最终答案。
如果证据不足，请明确说明。
"""
        })

        continue

    # --------------------------------------------------------
    # Tool Execution
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
        # Context Accumulation
        # ====================================================

        if result["context"]:
            all_context.append(
                result["context"]
            )

        # ====================================================
        # Evidence Accumulation
        # ====================================================

        for evidence in result["evidence"]:

            exists = any(
                item["source"] == evidence["source"]
                and item["sheet"] == evidence["sheet"]
                and item["chunk_id"] == evidence["chunk_id"]
                for item in all_evidence
            )

            if not exists:

                evidence["id"] = (
                    f"E{len(all_evidence) + 1}"
                )

                all_evidence.append(
                    evidence
                )

        # ====================================================
        # Tool Message
        # ====================================================

        messages.append(
            ToolMessage(
                content=str(result),
                tool_call_id=tool_call["id"]
            )
        )

        retrieval_count += 1


# ============================================================
# 13. Prepare Context
# ============================================================

accumulated_context = "\n\n".join(
    all_context
)


# ============================================================
# 14. Prepare Evidence
# ============================================================

evidence_text = "\n".join(
    f"""
Evidence ID: {item["id"]}
source: {item["source"]}
sheet: {item["sheet"]}
chunk_id: {item["chunk_id"]}
"""
    for item in all_evidence
)


# ============================================================
# 15. Generate Grounded Answer
# ============================================================

print("\n" + "=" * 70)
print("Generating Grounded Answer")
print("=" * 70)


ground_prompt = f"""
你是一个严格的知识库问答助手。

请根据下面提供的【问题】、【检索上下文】和【证据列表】生成最终答案。

【问题】
{question}

【检索上下文】
{all_context}

【证据列表】
{all_evidence}

要求：

1. 只能使用【检索上下文】中明确存在的信息回答。
2. 不允许根据常识、猜测或上下文之外的信息补充事实。
3. 每个重要事实都必须能够被【证据列表】中的一个或多个 evidence_id 支持。
4. evidence_ids 只能从【证据列表】中已经存在的 evidence_id 中选择。
5. 如果检索到的证据不足以完整回答问题，可以生成一个部分答案，但不能编造缺失信息。
6. 输出必须严格符合要求的结构：
   - answer：最终回答文本
   - evidence_ids：支持该回答的 evidence_id 列表
7. evidence_ids 必须是真实存在的 ID，不能自己创建新的 ID。

注意：
你的回答随后还会经过独立的 Citation Validation 校验。
即使你认为自己的回答正确，也必须确保每个引用的 evidence_id
确实能够支持对应的事实。json
"""

grounded_answer = grounding_llm.invoke(
    ground_prompt
)


# ============================================================
# 16. Print Grounded Answer
# ============================================================

print("\n" + "=" * 70)
print("Grounded Answer")
print("=" * 70)

print(
    f"\nAnswer:\n"
    f"{grounded_answer.answer}"
)

print(
    f"\nEvidence IDs:\n"
    f"{grounded_answer.evidence_ids}"
)


# ============================================================
# 17. Citation Validation
# ============================================================

print("\n" + "=" * 70)
print("Citation Validation")
print("=" * 70)


# ------------------------------------------------------------
# 先做程序级 ID Validation
# ------------------------------------------------------------

valid_ids = {
    item["id"]
    for item in all_evidence
}

invalid_ids = [
    evidence_id
    for evidence_id in grounded_answer.evidence_ids
    if evidence_id not in valid_ids
]


# ============================================================
# 18. 如果出现不存在的 Evidence ID
# ============================================================

if invalid_ids:                 #规则级别来源校验，排除所选evidence不存在于all evidence内部的这种情况

    print("\n程序级校验失败")

    print(
        "不存在的 Evidence ID：",
        invalid_ids
    )

    validation = CitationValidation(
        valid=False,
        reason=(
            "Grounded Answer 引用了不存在的 Evidence ID。"
        ),
        invalid_evidence_ids=invalid_ids
    )


# ============================================================
# 19. Evidence ID 存在
# ============================================================

else:

    # --------------------------------------------------------
    # 找出实际引用的 Evidence
    # --------------------------------------------------------

    cited_evidence = [
        item
        for item in all_evidence
        if item["id"]
        in grounded_answer.evidence_ids
    ]


    cited_evidence_text = "\n".join(
        f"""
Evidence ID: {item["id"]}
source: {item["source"]}
sheet: {item["sheet"]}
chunk_id: {item["chunk_id"]}
"""
        for item in cited_evidence
    )


    # --------------------------------------------------------
    # LLM Citation Validation          #这里是应用llm实现语义级别的来源校验
    # --------------------------------------------------------

    validation_prompt = f"""
你是一个严格的答案引用校验器。

你的任务不是重新回答问题，而是检查下面的【答案】是否真的被
【证据列表】中的引用证据支持。

【问题】
{question}

【答案】
{grounded_answer.answer}

【引用的 evidence_ids】
{grounded_answer.evidence_ids}

【全部证据】
{all_evidence}

请逐项检查：

1. 答案中的事实是否能够从引用的 evidence_ids 对应证据中直接得到支持。
2. 是否存在答案声称了某个事实，但引用证据没有支持该事实。
3. 是否存在引用了错误 evidence_id 的情况。
4. 是否存在答案中的信息超出了检索证据能够证明的范围。
5. 不要因为答案看起来合理，就默认 valid=True。
6. 如果任何重要事实没有得到证据支持，必须判定 valid=False。
7. 如果所有重要事实都能够被对应证据支持，才判定 valid=True。

校验失败是允许且预期的结果。

如果校验失败：
- valid 必须为 false
- reason 必须说明失败原因
- invalid_evidence_ids 必须列出存在问题的 evidence_id
- unsupported_claims 必须列出没有得到证据支持的事实

如果校验通过：
- valid 必须为 true
- reason 说明为什么证据足以支持答案
- invalid_evidence_ids 必须为空列表
- unsupported_claims 必须为空列表

输出必须严格符合要求的结构输出json，不要输出额外解释。
"""


    validation = validator_llm.invoke(
        validation_prompt
    )


# ============================================================
# 20. Validation Result
# ============================================================

print("\n" + "=" * 70)
print("Citation Validation Result")
print("=" * 70)

print(
    f"\nvalid: "
    f"{validation.valid}"
)

print(
    f"\nreason:\n"
    f"{validation.reason}"
)

print(
    f"\ninvalid_evidence_ids:\n"
    f"{validation.invalid_evidence_ids}"
)


# ============================================================
# 21. Final Decision
# ============================================================

print("\n" + "=" * 70)
print("Final Grounding Status")
print("=" * 70)


if validation.valid:

    print(
        "\n✓ Answer Grounding 通过"
    )

    print(
        "最终答案具有有效 Evidence 支撑。"
    )

else:

    print(
        "\n✗ Answer Grounding 未通过"
    )

    print(
        "最终答案存在引用问题，需要重新生成或修正。"
    )


# ============================================================
# 22. Evidence Mapping
# ============================================================

print("\n" + "=" * 70)
print("Evidence Mapping")
print("=" * 70)


for evidence_id in grounded_answer.evidence_ids:

    matched = [
        item
        for item in all_evidence
        if item["id"] == evidence_id
    ]

    if not matched:

        print(
            f"{evidence_id} -> NOT FOUND"
        )

        continue

    item = matched[0]

    print(
        f"{evidence_id} -> "
        f"source={item['source']}, "
        f"sheet={item['sheet']}, "
        f"chunk_id={item['chunk_id']}"
    )