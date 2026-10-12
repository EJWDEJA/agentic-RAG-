import os                        #用于测试citation validation失败以后进行repair修复的三个方向

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from typing import Literal

from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI


# ============================================================
# 1. LLM
# ============================================================

llm = ChatOpenAI(
    model="deepseek-v4-flash",
    api_key=os.environ["ANTHROPIC_AUTH_TOKEN"],
    base_url="https://api.deepseek.com"
)


# ============================================================
# 2. Citation Validation Result
# ============================================================

class CitationValidation(BaseModel):
    valid: bool = Field(
        description="答案是否完全被提供的 Evidence 支持"
    )

    unsupported_claims: list[str] = Field(
        description="答案中没有被 Evidence 支持的具体陈述"
    )

    reason: str = Field(
        description="判断原因"
    )


# ============================================================
# 3. Repair Decision
# ============================================================

class RepairDecision(BaseModel):
    repair_type: Literal[
        "retrieval",
        "answer",
        "none"
    ] = Field(
        description=(
            "retrieval = 证据不足，需要重新检索；"
            "answer = 已有证据，但答案存在过度推断，需要修改答案；"
            "none = 不需要 Repair"
        )
    )

    reason: str = Field(
        description="为什么选择这种 Repair"
    )


# ============================================================
# 4. Structured LLM
# ============================================================

repair_llm = llm.with_structured_output(
    RepairDecision,
    method="json_mode"
)


# ============================================================
# 5. Repair Diagnosis
# ============================================================

def diagnose_repair(
    question: str,
    evidence: list[dict],
    answer: str,
    validation: CitationValidation
) -> RepairDecision:

    evidence_text = "\n".join(
        f"""
Evidence ID: {item["id"]}
source: {item["source"]}
sheet: {item.get("sheet", "")}
chunk_id: {item["chunk_id"]}
content: {item.get("content", "")}
"""
        for item in evidence
    )

    prompt = f"""
你是一个 Agentic RAG Repair Diagnosis 模块。

你的任务不是修改答案，也不是重新检索。
你的任务只是判断 Citation Validation 失败后应该采取哪一种 Repair。

你必须严格输出一个 JSON 对象。

JSON 必须严格包含以下两个字段：

"repair_type": 字符串，只允许以下三个值之一：
- "retrieval"
- "answer"
- "none"

"reason": 字符串，说明为什么选择这个 repair_type。

JSON 格式必须严格如下：

{{
    "repair_type": "retrieval",
    "reason": "当前 Evidence 缺少回答问题所需要的信息，因此需要重新检索。"
}}

重要要求：

1. 输出必须是合法 JSON。
2. 只能输出 JSON 对象。
3. 不要输出 Markdown。
4. 不要使用 ```json 或 ``` 包裹。
5. 不要输出 JSON 以外的任何解释。
6. "repair_type" 必须是 "retrieval"、"answer"、"none" 三个值之一。
7. "reason" 必须是字符串。
8. 不要增加其他字段。

Repair 类型定义：

"retrieval"：
表示 Evidence 不足，当前 Evidence 无法支持用户问题中的必要信息，
需要重新检索。

"answer"：
表示 Evidence 已经包含必要信息，但是当前 Answer 超出了 Evidence
能够支持的范围，因此需要修改 Answer，不需要重新检索。

"none"：
表示当前不需要 Repair。

判断原则：

Evidence 不足 → "retrieval"

Evidence 已经存在，但 Answer 说多了 → "answer"

Citation 已经有效 → "none"


用户问题：
{question}

当前 Evidence：
{evidence_text}

当前 Answer：
{answer}

Citation Validation：
valid = {validation.valid}

unsupported_claims =
{validation.unsupported_claims}

reason =
{validation.reason}

现在只输出符合上述要求的 JSON。
"""

    return repair_llm.invoke(prompt)


# ============================================================
# 6. Test
# ============================================================

def run_test(
    title: str,
    question: str,
    evidence: list[dict],
    answer: str,
    validation: CitationValidation
):

    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

    print("\n[Question]")
    print(question)

    print("\n[Answer]")
    print(answer)

    print("\n[Validation]")
    print(validation.model_dump())

    result = diagnose_repair(
        question=question,
        evidence=evidence,
        answer=answer,
        validation=validation
    )

    print("\n[Repair Decision]")
    print(result.model_dump())


# ============================================================
# Case 1
# Evidence 不足
# ============================================================

evidence_1 = [
    {
        "id": "E1",
        "source": "教职工信息.xlsx",
        "sheet": "教职工与教辅人员信息",
        "chunk_id": 38,
        "content": "王建国，教授。"
    }
]

validation_1 = CitationValidation(
    valid=False,
    unsupported_claims=[
        "王建国的研究方向是人工智能教育"
    ],
    reason="当前 Evidence 只包含职称，没有研究方向信息。"
)

run_test(
    title="CASE 1：Evidence 不足",
    question="王建国的职称和研究方向是什么？",
    evidence=evidence_1,
    answer="王建国是教授，研究方向是人工智能教育。",
    validation=validation_1
)


# ============================================================
# Case 2
# Answer 过度推断
# ============================================================

evidence_2 = [
    {
        "id": "E1",
        "source": "教职工信息.xlsx",
        "sheet": "教职工与教辅人员信息",
        "chunk_id": 38,
        "content": "王建国，教授，研究方向为人工智能教育。"
    }
]

validation_2 = CitationValidation(
    valid=False,
    unsupported_claims=[
        "王建国长期负责学校人工智能课程建设"
    ],
    reason="Evidence 支持职称和研究方向，但没有支持课程建设经历。"
)

run_test(
    title="CASE 2：Answer 过度推断",
    question="王建国的职称和研究方向是什么？",
    evidence=evidence_2,
    answer=(
        "王建国是教授，研究方向为人工智能教育，"
        "并且长期负责学校人工智能课程建设。"
    ),
    validation=validation_2
)


# ============================================================
# Case 3
# Citation 有效
# ============================================================

evidence_3 = [
    {
        "id": "E1",
        "source": "教职工信息.xlsx",
        "sheet": "教职工与教辅人员信息",
        "chunk_id": 38,
        "content": "王建国，教授，研究方向为人工智能教育。"
    }
]

validation_3 = CitationValidation(
    valid=True,
    unsupported_claims=[],
    reason="答案中的所有信息均可由 Evidence 支持。"
)

run_test(
    title="CASE 3：Citation 有效",
    question="王建国的职称和研究方向是什么？",
    evidence=evidence_3,
    answer="王建国是教授，研究方向为人工智能教育。",
    validation=validation_3
)