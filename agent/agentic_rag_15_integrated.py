import os                 #这里是集成整合了之前学过的使用模块包括评估证据来源，来源校验重写...但是没有repair下一阶段将讲述

# ============================================================
# 0. 环境
# ============================================================

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import sys
import json

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from typing import TypedDict, Literal

from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage

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
# 2. 全局参数
# ============================================================

MAX_RETRIEVALS = 3


# ============================================================
# 3. Evidence 数据结构
#
# Evidence 只保存“证据定位信息”
# 不重复保存 content，避免和 context 重复占 token
# ============================================================

class Evidence(TypedDict):
    id: str
    source: str
    sheet: str
    chunk_id: int


# ============================================================
# 4. Retrieval Evaluation
#
# 判断：
# 当前证据是否足够回答问题
# ============================================================

class RetrievalDecision(BaseModel):
    sufficient: bool
    reason: str


retrieval_evaluator = llm.with_structured_output(
    RetrievalDecision,
    method="json_mode"
)


# ============================================================
# 5. Evidence Gap
#
# 如果证据不足：
# 到底缺什么？
# ============================================================

class EvidenceGap(BaseModel):
    has_gap: bool
    missing_information: list[str] = Field(
        default_factory=list
    )


evidence_gap_evaluator = llm.with_structured_output(
    EvidenceGap,
    method="json_mode"
)


# ============================================================
# 6. Query Rewrite
#
# 根据原问题 + 当前 Evidence Gap
# 生成下一轮针对性查询
# ============================================================

class SearchQuery(BaseModel):
    query: str


query_rewriter = llm.with_structured_output(
    SearchQuery,
    method="json_mode"
)


# ============================================================
# 7. Grounded Answer
#
# Answer + 使用了哪些 Evidence
# ============================================================

class GroundedAnswer(BaseModel):
    answer: str
    evidence_ids: list[str]


grounding_llm = llm.with_structured_output(
    GroundedAnswer,
    method="json_mode"
)


# ============================================================
# 8. Citation Validation
#
# 第一层：
# evidence_id 是否真的存在
#
# 第二层：
# Answer 是否被当前检索内容支持
# ============================================================

class CitationValidation(BaseModel):
    valid: bool
    reason: str
    invalid_evidence_ids: list[str] = Field(
        default_factory=list
    )


citation_validator = llm.with_structured_output(
    CitationValidation,
    method="json_mode"
)


# ============================================================
# 9. Retrieval Tool
# ============================================================

@tool
def retrieve_knowledge(query: str) -> dict:
    """
    从学校知识库中检索相关知识。
    返回 context 和 evidence。
    """

    print("\n" + "=" * 70)
    print("执行 Retrieval")
    print("Query:", query)
    print("=" * 70)

    result = retrieve_context(query)

    context = result["context"]

    # --------------------------------------------------------
    # 将 sources 转换成统一 Evidence
    # --------------------------------------------------------

    evidence_list = []

    for source in result["sources"]:

        evidence = {
            "id": "",
            "source": source.get("source", ""),
            "sheet": source.get("sheet", ""),
            "chunk_id": source.get("chunk_id", -1)
        }

        evidence_list.append(evidence)

    return {
        "context": context,
        "evidence": evidence_list
    }


# ============================================================
# 10. Evidence ID
#
# 使用 source + sheet + chunk_id 形成稳定 ID
# ============================================================

def build_evidence_id(evidence: Evidence) -> str:

    return (
        f'{evidence["source"]}'
        f'|{evidence["sheet"]}'
        f'|{evidence["chunk_id"]}'
    )


# ============================================================
# 11. Evidence Dedup
#
# source / sheet / chunk_id 相同
# 认为是同一个 Evidence
# ============================================================

def add_evidence(
    all_evidence: list[Evidence],
    new_evidence: list[Evidence]
):

    for evidence in new_evidence:

        evidence_id = build_evidence_id(evidence)

        evidence["id"] = evidence_id

        exists = any(
            item["source"] == evidence["source"]
            and item["sheet"] == evidence["sheet"]
            and item["chunk_id"] == evidence["chunk_id"]
            for item in all_evidence
        )

        if not exists:
            all_evidence.append(evidence)


# ============================================================
# 12. Query Rewrite
# ============================================================

def rewrite_query(
    original_question: str,
    missing_information: list[str]
) -> str:

    prompt = f"""
你是 Query Rewrite Agent。

你的任务：
根据用户原始问题和当前 Evidence Gap，生成下一轮知识库检索 Query。

【原始问题】
{original_question}

【缺失信息】
{missing_information}

【生成规则】
1. Query 必须围绕原始问题。
2. Query 必须针对当前缺失信息。
3. 不要回答问题。
4. 不要解释 Query。
5. 只生成一个最有针对性的检索 Query。
6. Query 应包含回答缺失信息所需要的关键实体或关键词。

【JSON 输出格式】
必须严格输出：

{
  "query": "王建国研究方向"
}

【字段定义】
- query：字符串。
- query 是下一轮实际用于知识库检索的查询语句。

【重要要求】
- 只能输出 JSON。
- 不要输出 Markdown。
- 不要输出 ```json。
- 不要输出 JSON 之外的任何文字。
"""

    result = query_rewriter.invoke(
        [
            SystemMessage(
                content="你负责生成精准的知识库检索 Query。"
            ),
            HumanMessage(content=prompt)
        ]
    )

    return result.query


# ============================================================
# 13. Evidence Evaluation
#
# 注意：
#
# Retrieval Evaluation：
# “检索结果够不够？”
#
# Evidence Evaluation：
# “当前 Evidence 能不能支撑这个问题？”
#
# ============================================================

def evaluate_evidence(
    question: str,
    context: str,
    evidence: list[Evidence]
) -> RetrievalDecision:

    evidence_info = "\n".join(
        [
            f'- {item["id"]} | '
            f'{item["source"]} | '
            f'{item["sheet"]} | '
            f'chunk={item["chunk_id"]}'
            for item in evidence
        ]
    )

    prompt = f"""
你是 Evidence Evaluation Agent。

你的任务是：
判断当前已经累计获得的知识内容，是否已经包含回答用户问题所需要的关键信息。

【用户问题】
{question}

【当前累计知识内容】
{context}

【当前 Evidence】
{evidence_info}

【判断标准】
请判断当前知识内容是否足以回答用户问题。

sufficient = true：
表示当前知识内容已经包含回答问题所需要的关键事实，
可以直接进入答案生成阶段，不需要继续检索。

sufficient = false：
表示当前知识内容缺少回答问题所需要的一个或多个关键事实，
需要继续进行 Evidence Gap 分析和后续检索。

【重要区分】
不要仅仅因为存在 Evidence 就判断 sufficient = true。

必须判断：
“这些 Evidence 中包含的实际知识内容，
是否能够回答用户的问题。”

Evidence 只是证据定位信息。
真正用于判断知识是否充分的内容来自当前累计知识内容。

【reason 要求】
reason 必须简洁说明：
1. 为什么当前证据足够；或者
2. 为什么当前证据不足。

如果不足，应指出主要缺失方向，但不要代替 Evidence Gap 模块详细分析。

【输出格式】
必须输出一个 JSON 对象：

{{
    "sufficient": true,
    "reason": "字符串"
}}

【字段说明】
- sufficient：布尔类型，只能是 true 或 false。
- reason：字符串类型，说明当前 Evidence 是否足够以及原因。

【重要要求】
1. 必须输出合法 JSON。
2. 不要输出 Markdown。
3. 不要使用 ```json。
4. 不要输出 JSON 以外的任何文字。
"""

    return retrieval_evaluator.invoke(
        [
            SystemMessage(
                content="你负责判断 Evidence 是否足以支撑当前任务。"
            ),
            HumanMessage(content=prompt)
        ]
    )


# ============================================================
# 14. Evidence Gap Detection
#
# 负责回答：
# “不够的话，到底缺什么？”
# ============================================================

def detect_evidence_gap(
    question: str,
    context: str
) -> EvidenceGap:

    prompt = f"""
你是 Evidence Gap Detection Agent。

你的任务是：
比较用户问题和当前已有知识，
找出回答用户问题所缺少的具体信息。

【用户问题】
{question}

【当前已有知识】
{context}

【判断规则】
如果当前已有知识已经包含回答用户问题所需要的全部关键事实：

- has_gap 必须为 false
- missing_information 必须为空列表 []

如果当前已有知识缺少关键事实：

- has_gap 必须为 true
- missing_information 必须列出具体缺少的信息。

【missing_information 要求】
1. 必须描述具体缺失的信息。
2. 不要写“更多信息”“相关信息”“详细信息”等笼统描述。
3. 每一项都应该能够直接用于指导下一次检索。
4. 不要回答用户问题。
5. 不要重复已经存在于当前知识中的信息。
6. 如果存在多个独立缺口，可以列出多个缺失信息。

【示例】

用户问题：
“王建国的职称和研究方向是什么？”

当前已有知识：
“王建国，职称：教授。”

正确判断应该类似：

{{
    "has_gap": true,
    "missing_information": [
        "王建国的研究方向"
    ]
}}

【输出格式】
必须输出一个 JSON 对象：

{{
    "has_gap": true,
    "missing_information": [
        "字符串"
    ]
}}

【字段说明】
- has_gap：布尔类型。
  true 表示存在信息缺口。
  false 表示不存在信息缺口。

- missing_information：字符串列表。
  每个字符串表示一个具体缺失的信息。

【重要要求】
1. 必须输出合法 JSON。
2. 不要输出 Markdown。
3. 不要使用 ```json。
4. 不要输出 JSON 以外的任何文字。
"""

    return evidence_gap_evaluator.invoke(
        [
            SystemMessage(
                content="你负责发现当前 Evidence 中缺失的信息。"
            ),
            HumanMessage(content=prompt)
        ]
    )


# ============================================================
# 15. Grounded Answer
# ============================================================

def generate_grounded_answer(
    question: str,
    context: str,
    evidence: list[Evidence]
) -> GroundedAnswer:

    evidence_info = "\n".join(
        [
            f'- {item["id"]}'
            for item in evidence
        ]
    )

    prompt = f"""
你是一个严格基于知识库证据回答问题的 Grounded Answer Agent。

你的任务是：
仅使用当前提供的知识库内容回答用户问题，
并列出真正用于支持答案的 Evidence ID。

【用户问题】
{question}

【知识库内容】
{context}

【可用 Evidence ID】
{evidence_info}

【回答规则】
1. 只能使用知识库内容中的事实。
2. 不允许使用知识库之外的常识、猜测或推断补充事实。
3. 如果知识库没有提供某项信息，不要编造。
4. 可以对知识库中的信息进行正常的语言组织。
5. answer 必须直接回答用户问题。
6. evidence_ids 只能使用当前提供的 Evidence ID。
7. 只引用真正用于支持答案的 Evidence。
8. 不要为了增加引用数量而引用无关 Evidence。

【Evidence ID 规则】
Evidence ID 表示一个具体证据的位置。

只有当某个 Evidence 实际支持答案中的信息时，
才能将它加入 evidence_ids。

【输出格式】
必须输出一个 JSON 对象：

{{
    "answer": "字符串",
    "evidence_ids": [
        "Evidence ID"
    ]
}}

【字段说明】
- answer：字符串类型。
  表示最终回答用户的问题。

- evidence_ids：字符串列表。
  表示真正用于支持 answer 的 Evidence ID。

【重要要求】
1. 必须输出合法 JSON。
2. 不要输出 Markdown。
3. 不要使用 ```json。
4. 不要输出 JSON 以外的任何文字。
"""

    return grounding_llm.invoke(
        [
            SystemMessage(
                content="你负责生成有证据依据的答案。"
            ),
            HumanMessage(content=prompt)
        ]
    )


# ============================================================
# 16. Citation Validation
# ============================================================

def validate_citations(
    question: str,
    answer: GroundedAnswer,
    context: str,
    evidence: list[Evidence]
) -> CitationValidation:

    valid_ids = {
        item["id"]
        for item in evidence
    }

    # --------------------------------------------------------
    # 第一层：
    # 程序直接检查 Evidence ID 是否存在
    # --------------------------------------------------------

    invalid_ids = [
        evidence_id
        for evidence_id in answer.evidence_ids
        if evidence_id not in valid_ids
    ]

    if invalid_ids:

        return CitationValidation(
            valid=False,
            reason=(
                "Answer 引用了不存在于当前 Evidence 集合中的 Evidence ID。"
            ),
            invalid_evidence_ids=invalid_ids
        )

    # --------------------------------------------------------
    # 第二层：
    # LLM 判断答案是否被当前知识内容支持
    # --------------------------------------------------------

    prompt = f"""
你是 Citation Validation Agent。

你的任务是：
验证最终答案中的事实是否能够被当前知识库内容支持，
并验证答案引用的 Evidence 是否真正与答案相关。

【用户问题】
{question}

【知识库内容】
{context}

【最终答案】
{answer.answer}

【答案引用的 Evidence ID】
{answer.evidence_ids}

【验证规则】

请逐项检查最终答案：

1. 答案中的关键事实是否出现在知识库内容中？
2. 答案中的关键事实是否可以由知识库内容直接支持？
3. 答案是否加入了知识库中不存在的事实？
4. 引用的 Evidence 是否真正支持答案中的相关事实？
5. 是否存在引用了 Evidence，但该 Evidence 与答案内容无关的情况？

【valid 判断规则】

valid = true：

只有当最终答案中的关键事实都能够被当前知识库内容支持，
并且引用的 Evidence 与答案相关时，才能判断为 true。

valid = false：

只要存在以下任意一种情况，就必须判断为 false：

- 答案包含知识库没有提供的事实。
- 答案包含无法从知识库内容得到支持的推断。
- 引用的 Evidence 不支持对应答案。
- 引用的 Evidence 与答案无关。
- 答案存在重要事实缺少证据支持。

【重要原则】

“答案看起来合理”不能作为 valid=true 的依据。

必须判断：
“答案是否能够被当前知识库内容和引用的 Evidence 支持。”

【invalid_evidence_ids】

如果存在与答案不匹配的 Evidence ID，
将这些 Evidence ID 放入 invalid_evidence_ids。

如果没有发现无效 Evidence：

[
]

【reason】

简洁说明验证结果。

如果 valid=false，
说明具体是哪一类事实没有得到支持，
或者哪个 Evidence 与答案不匹配。

【输出格式】

必须输出一个 JSON 对象：

{{
    "valid": true,
    "reason": "字符串",
    "invalid_evidence_ids": []
}}

【字段说明】

- valid：布尔类型。
  true 表示答案和引用通过验证。
  false 表示答案或引用存在证据问题。

- reason：字符串类型。
  表示验证结果及原因。

- invalid_evidence_ids：字符串列表。
  表示无法支持答案的 Evidence ID。
  如果没有无效 Evidence，必须返回空列表 []。

【重要要求】

1. 必须输出合法 JSON。
2. 不要输出 Markdown。
3. 不要使用 ```json。
4. 不要输出 JSON 以外的任何文字。
"""

    return citation_validator.invoke(
        [
            SystemMessage(
                content="你负责验证答案和引用之间的证据关系。"
            ),
            HumanMessage(content=prompt)
        ]
    )


# ============================================================
# 17. 主 Agentic RAG
# ============================================================

def agentic_rag(question: str):

    print("\n")
    print("#" * 70)
    print("Agentic RAG")
    print("#" * 70)

    print("\n用户问题：")
    print(question)

    # --------------------------------------------------------
    # Evidence Accumulation
    # --------------------------------------------------------

    all_context = []
    all_evidence = []

    retrieval_count = 0

    current_query = question

    # ========================================================
    # Retrieval Loop
    # ========================================================

    while retrieval_count < MAX_RETRIEVALS:

        retrieval_count += 1

        print("\n")
        print("─" * 70)
        print(
            f"第 {retrieval_count} 次 Retrieval"
        )
        print("─" * 70)

        # ----------------------------------------------------
        # Retrieval
        # ----------------------------------------------------

        result = retrieve_knowledge.invoke(
            {
                "query": current_query
            }
        )

        context = result["context"]
        evidence = result["evidence"]

        # ----------------------------------------------------
        # Evidence Accumulation
        # ----------------------------------------------------

        all_context.append(context)

        add_evidence(
            all_evidence,
            evidence
        )

        print("\n当前累计 Evidence：")

        for item in all_evidence:
            print(
                f'  {item["id"]}'
            )

        # ----------------------------------------------------
        # 合并 Context
        # ----------------------------------------------------

        accumulated_context = "\n\n".join(
            all_context
        )

        # ====================================================
        # Evidence Evaluation
        # ====================================================

        evaluation = evaluate_evidence(
            question,
            accumulated_context,
            all_evidence
        )

        print("\nEvidence Evaluation：")
        print(
            "sufficient =",
            evaluation.sufficient
        )
        print(
            "reason =",
            evaluation.reason
        )

        # ----------------------------------------------------
        # 如果 Evidence 已经足够
        # ----------------------------------------------------

        if evaluation.sufficient:

            print(
                "\nEvidence 已经足够，停止 Retrieval。"
            )

            break

        # ====================================================
        # Evidence Gap
        # ====================================================

        gap = detect_evidence_gap(
            question,
            accumulated_context
        )

        print("\nEvidence Gap：")
        print(
            "has_gap =",
            gap.has_gap
        )
        print(
            "missing_information =",
            gap.missing_information
        )

        # ----------------------------------------------------
        # 没有明确 Gap
        # ----------------------------------------------------

        if not gap.has_gap:

            print(
                "\n没有发现明确 Evidence Gap，"
                "停止 Retrieval。"
            )

            break

        # ----------------------------------------------------
        # Search Budget
        # ----------------------------------------------------

        if retrieval_count >= MAX_RETRIEVALS:

            print(
                "\n达到 Retrieval Search Budget。"
            )

            break

        # ====================================================
        # Targeted Retrieval
        # ====================================================

        current_query = rewrite_query(
            question,
            gap.missing_information
        )

        print("\nTargeted Query：")
        print(current_query)

    # ========================================================
    # 最终 Evidence
    # ========================================================

    accumulated_context = "\n\n".join(
        all_context
    )

    print("\n")
    print("=" * 70)
    print("最终 Evidence")
    print("=" * 70)

    for item in all_evidence:

        print(
            f'ID: {item["id"]}'
        )

        print(
            f'  source: {item["source"]}'
        )

        print(
            f'  sheet: {item["sheet"]}'
        )

        print(
            f'  chunk_id: {item["chunk_id"]}'
        )

    # ========================================================
    # Grounded Answer
    # ========================================================

    print("\n")
    print("=" * 70)
    print("Grounded Answer")
    print("=" * 70)

    answer = generate_grounded_answer(
        question,
        accumulated_context,
        all_evidence
    )

    print("\n答案：")
    print(answer.answer)

    print("\n引用 Evidence：")

    for evidence_id in answer.evidence_ids:
        print(
            " ",
            evidence_id
        )

    # ========================================================
    # Citation Validation
    # ========================================================

    print("\n")
    print("=" * 70)
    print("Citation Validation")
    print("=" * 70)

    validation = validate_citations(
        question,
        answer,
        accumulated_context,
        all_evidence
    )

    print("\nvalid：")
    print(validation.valid)

    print("\nreason：")
    print(validation.reason)

    print("\ninvalid_evidence_ids：")
    print(validation.invalid_evidence_ids)

    # ========================================================
    # 最终输出
    # ========================================================

    print("\n")
    print("#" * 70)
    print("Agentic RAG 完成")
    print("#" * 70)

    return {
        "question": question,
        "context": accumulated_context,
        "evidence": all_evidence,
        "answer": answer.answer,
        "evidence_ids": answer.evidence_ids,
        "citation_valid": validation.valid,
        "citation_reason": validation.reason,
        "retrieval_count": retrieval_count
    }


# ============================================================
# 18. 测试
# ============================================================

if __name__ == "__main__":

    question = input(
        "\n请输入问题："
    )

    result = agentic_rag(
        question
    )

    print("\n")
    print("=" * 70)
    print("最终结构化结果")
    print("=" * 70)

    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2
        )
    )
