import os                              #用于测试repair当证据不足是的情况，   垃圾架构这一步在证据校验的while true循环就应该被解决，来源校验只要针对answer有没有多说

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
# 2. Repair Query
# ============================================================

class RepairQuery(BaseModel):

    query: str = Field(
        description="针对缺失 Evidence 生成的精准检索 Query"
    )

    target: str = Field(
        description="本次检索需要寻找的具体信息"
    )


# ============================================================
# 3. JSON Mode
# ============================================================

repair_query_llm = llm.with_structured_output(
    RepairQuery,
    method="json_mode"
)


# ============================================================
# 4. Generate Repair Query
# ============================================================

def generate_repair_query(
    question: str,
    evidence: list[dict],
    unsupported_claims: list[str]
) -> RepairQuery:

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

    unsupported_text = "\n".join(
        f"- {item}"
        for item in unsupported_claims
    )

    prompt = f"""
你是 Agentic RAG 的 Retrieval Repair Query Generator。

你的任务是：

根据当前已有 Evidence 和 Citation Validation 发现的
unsupported claims，生成一个精准的 Repair Retrieval Query。

你的任务不是回答用户问题。

你的任务不是修改 Answer。

你的任务只是：
找出当前缺失的信息，并生成下一次检索使用的 Query。

必须输出合法 JSON。

JSON 必须严格包含以下两个字段：

"query": 字符串
表示下一次实际用于知识库检索的精准 Query。

"target": 字符串
表示本次检索具体要寻找什么信息。

JSON 格式必须严格如下：

{{
    "query": "王建国 研究方向",
    "target": "王建国的研究方向"
}}

严格要求：

1. 只能输出 JSON。
2. 不要输出 Markdown。
3. 不要使用 ```json。
4. 不要输出 JSON 以外的任何文字。
5. "query" 必须是字符串。
6. "target" 必须是字符串。
7. 不要增加其他字段。
8. Query 应该尽可能针对缺失信息，而不是重复已经存在的 Evidence。
9. 不要生成答案。

用户原问题：

{question}

当前 Evidence：

{evidence_text}

当前未被 Evidence 支持的内容：

{unsupported_text}

请根据缺失信息生成精准的 Repair Query。
"""


    return repair_query_llm.invoke(prompt)


# ============================================================
# 5. Fake Retrieval
# ============================================================

def fake_retrieve(query: str) -> list[dict]:

    print("\n[Retrieval Repair]")
    print("Query:", query)

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
        },
        {
            "id": "E41",
            "source": "教职工信息.xlsx",
            "sheet": "教职工与教辅人员信息",
            "chunk_id": 41,
            "content": "王建国承担人工智能相关课程教学工作。"
        }
    ]

    # 模拟 Targeted Retrieval
    results = []

    for item in knowledge_base:

        if "研究方向" in query:

            if "研究方向" in item["content"]:
                results.append(item)

    return results


# ============================================================
# 6. Evidence Deduplication
# ============================================================

def accumulate_evidence(
    all_evidence: list[dict],
    new_evidence: list[dict]
) -> list[dict]:

    for evidence in new_evidence:

        exists = any(
            item["source"] == evidence["source"]
            and item["sheet"] == evidence["sheet"]
            and item["chunk_id"] == evidence["chunk_id"]
            for item in all_evidence
        )

        if not exists:
            all_evidence.append(evidence)

    return all_evidence


# ============================================================
# 7. Main Test
# ============================================================

def main():

    question = "王建国的职称和研究方向是什么？"

    # ----------------------------------------
    # 原始 Evidence
    # ----------------------------------------

    all_evidence = [
        {
            "id": "E38",
            "source": "教职工信息.xlsx",
            "sheet": "教职工与教辅人员信息",
            "chunk_id": 38,
            "content": "王建国，教授。"
        }
    ]

    # ----------------------------------------
    # Citation Validation
    # ----------------------------------------

    unsupported_claims = [
        "王建国的研究方向为人工智能教育"
    ]

    print("=" * 70)
    print("原始 Evidence")
    print("=" * 70)

    for item in all_evidence:
        print(item)

    print("\n" + "=" * 70)
    print("Unsupported Claims")
    print("=" * 70)

    for item in unsupported_claims:
        print(item)

    # ----------------------------------------
    # 生成 Repair Query
    # ----------------------------------------

    repair_query = generate_repair_query(
        question=question,
        evidence=all_evidence,
        unsupported_claims=unsupported_claims
    )

    print("\n" + "=" * 70)
    print("Repair Query")
    print("=" * 70)

    print(
        repair_query.model_dump()
    )

    # ----------------------------------------
    # Targeted Retrieval
    # ----------------------------------------

    new_evidence = fake_retrieve(
        repair_query.query
    )

    print("\n" + "=" * 70)
    print("New Evidence")
    print("=" * 70)

    for item in new_evidence:
        print(item)

    # ----------------------------------------
    # Evidence Accumulation
    # ----------------------------------------

    all_evidence = accumulate_evidence(
        all_evidence,
        new_evidence
    )

    print("\n" + "=" * 70)
    print("Accumulated Evidence")
    print("=" * 70)

    for item in all_evidence:
        print(item)


if __name__ == "__main__":
    main()