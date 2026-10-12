import os                   #垃圾架构无参考价值将来源校验和证据校验混为一谈

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
# 2. Grounded Answer
# ============================================================

class GroundedAnswer(BaseModel):

    answer: str = Field(
        description="基于 Evidence 生成的最终答案"
    )

    evidence_ids: list[str] = Field(
        description="答案实际依赖的 Evidence ID"
    )


# ============================================================
# 3. Citation Validation
# ============================================================

class CitationValidation(BaseModel):

    valid: bool = Field(
        description="答案是否完全被 Evidence 支持"
    )

    unsupported_claims: list[str] = Field(
        description="没有被 Evidence 支持的答案内容"
    )

    reason: str = Field(
        description="验证原因"
    )


# ============================================================
# 4. Repair Decision
# ============================================================

class RepairDecision(BaseModel):

    repair_type: Literal[
        "retrieval",
        "answer",
        "none"
    ]

    reason: str


# ============================================================
# 5. Repair Query
# ============================================================

class RepairQuery(BaseModel):

    query: str

    target: str


# ============================================================
# 6. Grounding LLM
# ============================================================

grounding_llm = llm.with_structured_output(
    GroundedAnswer,
    method="json_mode"
)


# ============================================================
# 7. Validation LLM
# ============================================================

validation_llm = llm.with_structured_output(
    CitationValidation,
    method="json_mode"
)


# ============================================================
# 8. Repair Diagnosis LLM
# ============================================================

repair_llm = llm.with_structured_output(
    RepairDecision,
    method="json_mode"
)


# ============================================================
# 9. Repair Query LLM
# ============================================================

repair_query_llm = llm.with_structured_output(
    RepairQuery,
    method="json_mode"
)


# ============================================================
# 10. Grounding
# ============================================================

def ground_answer(
    question: str,
    evidence: list[dict]
) -> GroundedAnswer:

    evidence_text = "\n".join(
        f"""
Evidence ID: {item["id"]}
source: {item["source"]}
sheet: {item["sheet"]}
chunk_id: {item["chunk_id"]}
content: {item["content"]}
"""
        for item in evidence
    )

    prompt = f"""
你是 Agentic RAG 的 Grounding 模块。

你的任务是严格依据 Evidence 回答用户问题。

重要要求：

1. 只能使用 Evidence 中明确存在的信息。
2. 不允许补充 Evidence 中不存在的事实。
3. 不允许猜测。
4. answer 必须是最终给用户看的回答。
5. evidence_ids 必须只包含实际支持 answer 的 Evidence ID。

你必须输出合法 JSON。

JSON 必须严格包含：

"answer": 字符串

"evidence_ids": 字符串数组

格式：

{{
    "answer": "王建国是教授，研究方向为人工智能教育。",
    "evidence_ids": ["E38", "E40"]
}}

只能输出 JSON。
不要输出 Markdown。
不要输出 ```json。
不要增加其他字段。

用户问题：

{question}

Evidence：

{evidence_text}
"""

    return grounding_llm.invoke(prompt)


# ============================================================
# 11. Citation Validation
# ============================================================

def validate_citation(
    question: str,
    answer: GroundedAnswer,
    evidence: list[dict]
) -> CitationValidation:

    evidence_text = "\n".join(
        f"""
Evidence ID: {item["id"]}
content: {item["content"]}
"""
        for item in evidence
    )

    prompt = f"""
你是 Agentic RAG 的 Citation Validation 模块。

判断 Answer 中的每一个事实是否都能够被 Evidence 明确支持。

必须输出合法 JSON。

JSON 必须严格包含：

"valid": 布尔值     表示回答中的答案是否被证据全部支持

"unsupported_claims": 字符串数组

"reason": 字符串

格式：

{{
    "valid": false,  
    "unsupported_claims": [
        "没有被 Evidence 支持的具体内容"
    ],
    "reason": "说明为什么不支持"
}}

严格要求：

1. 只能输出 JSON。
2. 不要输出 Markdown。
3. 不要使用 ```json。
4. 不要增加其他字段。

用户问题：

{question}

Answer：

{answer.answer}

Answer 使用的 Evidence IDs：

{answer.evidence_ids}

Evidence：

{evidence_text}
"""

    return validation_llm.invoke(prompt)


# ============================================================
# 12. Repair Diagnosis
# ============================================================

def diagnose_repair(
    validation: CitationValidation
) -> RepairDecision:

    prompt = f"""
你是 Agentic RAG Repair Diagnosis 模块。

根据 Citation Validation 的结果，
判断是否需要 Repair。

只能选择：

"retrieval"
"answer"
"none"

含义：

retrieval：
Evidence 不足，需要重新检索。

answer：
Evidence 已经足够，但是 Answer 说出了 Evidence
没有支持的信息，需要修改 Answer。

none：
不需要 Repair。

必须输出合法 JSON。

JSON 必须严格包含：

"repair_type": 字符串

"reason": 字符串

其中 repair_type 只能是：

"retrieval"
"answer"
"none"

格式：

{{
    "repair_type": "retrieval",
    "reason": "当前 Evidence 缺少必要信息，需要重新检索。"
}}

只能输出 JSON。
不要输出 Markdown。
不要增加其他字段。

Citation Validation：

valid = {validation.valid}

unsupported_claims =
{validation.unsupported_claims}

reason =
{validation.reason}
"""

    return repair_llm.invoke(prompt)


# ============================================================
# 13. Generate Repair Query
# ============================================================

def generate_repair_query(
    question: str,
    evidence: list[dict],
    unsupported_claims: list[str]
) -> RepairQuery:

    evidence_text = "\n".join(
        f"""
Evidence ID: {item["id"]}
content: {item["content"]}
"""
        for item in evidence
    )

    unsupported_text = "\n".join(
        f"- {claim}"
        for claim in unsupported_claims
    )

    prompt = f"""
你是 Agentic RAG 的 Retrieval Repair Query Generator。

根据用户问题、已有 Evidence 和 unsupported claims，
生成下一次 Targeted Retrieval 使用的精准 Query。

你的任务不是回答问题。

必须输出合法 JSON。

JSON 必须严格包含：

"query": 字符串

"target": 字符串

格式：

{{
    "query": "王建国 研究方向",
    "target": "王建国的研究方向"
}}

严格要求：

1. 只能输出 JSON。
2. 不要输出 Markdown。
3. 不要使用 ```json。
4. 不要增加其他字段。
5. query 必须针对缺失信息。
6. 不要重复已经存在的 Evidence。

用户问题：

{question}

已有 Evidence：

{evidence_text}

Unsupported Claims：

{unsupported_text}
"""

    return repair_query_llm.invoke(prompt)


# ============================================================
# 14. Fake Retrieval
# ============================================================

def fake_retrieve(
    query: str
) -> list[dict]:

    knowledge_base = [

        {
            "id": "E38",
            "source": "教职工信息.xlsx",
            "sheet": "教职工与教辅人员信息",
            "chunk_id": 38,
            "content": "王建国，教授。"
        },

        {
            "id": "E40",
            "source": "教职工信息.xlsx",
            "sheet": "教职工与教辅人员信息",
            "chunk_id": 40,
            "content": "王建国的研究方向为人工智能教育。"
        }
    ]

    results = []

    if "研究方向" in query:

        for item in knowledge_base:

            if "研究方向" in item["content"]:
                results.append(item)

    return results


# ============================================================
# 15. Evidence Accumulation
# ============================================================

def accumulate_evidence(
    all_evidence: list[dict],
    new_evidence: list[dict]
):

    for evidence in new_evidence:

        exists = any(
            item["source"] == evidence["source"]
            and item["sheet"] == evidence["sheet"]
            and item["chunk_id"] == evidence["chunk_id"]
            for item in all_evidence
        )

        if not exists:

            all_evidence.append(evidence)


# ============================================================
# 16. Main Repair Loop
# ============================================================

def main():

    question = "王建国的职称和研究方向是什么？"

    all_evidence = [

        {
            "id": "E38",
            "source": "教职工信息.xlsx",
            "sheet": "教职工与教辅人员信息",
            "chunk_id": 38,
            "content": "王建国，教授。"
        }
    ]

    MAX_REPAIRS = 2

    repair_count = 0

    # ========================================================
    # Initial Grounding
    # ========================================================

    print("\n" + "=" * 70)
    print("INITIAL GROUNDING")
    print("=" * 70)

    answer = ground_answer(
        question,
        all_evidence
    )

    print("\nAnswer:")
    print(answer.model_dump())

    # ========================================================
    # Repair Loop
    # ========================================================

    while True:

        print("\n" + "=" * 70)
        print(
            f"CITATION VALIDATION | repair_count={repair_count}"
        )
        print("=" * 70)

        validation = validate_citation(
            question,
            answer,
            all_evidence
        )

        print("\nValidation:")
        print(validation.model_dump())

        # ----------------------------------------------------
        # Success
        # ----------------------------------------------------

        if validation.valid:

            print("\nCitation Validation PASS.")

            break

        # ----------------------------------------------------
        # Repair Budget
        # ----------------------------------------------------

        if repair_count >= MAX_REPAIRS:

            print("\nRepair budget exhausted.")

            break

        # ----------------------------------------------------
        # Repair Diagnosis
        # ----------------------------------------------------

        repair = diagnose_repair(
            validation
        )

        print("\nRepair Decision:")
        print(repair.model_dump())

        # ----------------------------------------------------
        # Retrieval Repair
        # ----------------------------------------------------

        if repair.repair_type == "retrieval":

            repair_query = generate_repair_query(
                question=question,
                evidence=all_evidence,
                unsupported_claims=validation.unsupported_claims
            )

            print("\nRepair Query:")
            print(repair_query.model_dump())

            new_evidence = fake_retrieve(
                repair_query.query
            )

            print("\nNew Evidence:")

            for item in new_evidence:
                print(item)

            accumulate_evidence(
                all_evidence,
                new_evidence
            )

            print("\nAccumulated Evidence:")

            for item in all_evidence:
                print(item)

            # ------------------------------------------------
            # Evidence changed
            # ------------------------------------------------

            print("\nRe-grounding...")

            answer = ground_answer(
                question,
                all_evidence
            )

            print("\nNew Answer:")
            print(answer.model_dump())
            print(answer)

            repair_count += 1

            continue

        # ----------------------------------------------------
        # Answer Repair
        # ----------------------------------------------------

        elif repair.repair_type == "answer":

            print(
                "\nAnswer Repair is not implemented yet."
            )

            break

        # ----------------------------------------------------
        # None
        # ----------------------------------------------------

        else:

            print(
                "\nRepair stopped."
            )

            break

    # ========================================================
    # Final
    # ========================================================

    print("\n" + "=" * 70)
    print("FINAL RESULT")
    print("=" * 70)

    print("\nAnswer:")
    print(answer.model_dump())

    print("\nEvidence:")

    for item in all_evidence:
        print(item)


if __name__ == "__main__":
    main()