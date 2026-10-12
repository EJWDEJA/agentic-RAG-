import os                               #这一步是全新架构完整真正将证据是否足够（evoluation evidence）和ground的answer的每一个claim是否有足够的证据支撑（citation evoluation），没有的话就进行answer修复   分开         剔除就得context和evidence分离思想现在的evidence内含来源和玩崩的字典

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
# 2. 全局参数
# ============================================================

MAX_RETRIEVALS = 3

MAX_ANSWER_REPAIRS = 2

FINAL_FALLBACK_MESSAGE = (
    "当前检索到的证据无法可靠支持一个经过引用校验的完整答案。"
    "因此没有输出未经验证的结论。"
)
# ============================================================
# 3. Evidence
#
# Evidence  保存定位信息
#           保存 content

# Evidence 是 Agentic RAG 后续阶段的统一事实载体
#
# ============================================================

class Evidence(TypedDict):
    id: str
    source: str
    sheet: str
    chunk_id: int
    content: str


# ============================================================
# 4. Evidence Evaluation
#
# 唯一职责：
# 判断当前 Evidence 是否足以回答用户问题
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
# 唯一职责：
# Evidence 不足时，找出具体缺失信息
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
# 根据 Evidence Gap 生成 Targeted Query
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
# 只有 Evidence Evaluation = True
# 才允许生成
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
# 核心职责：
#
# Answer 中的事实性主张
# 是否被其引用的 Evidence 支持
#
# 注意：
# 不再负责判断 Evidence 是否足够。
# ============================================================
class CitationValidation(BaseModel):

    valid: bool

    unsupported_claims: list[str] = Field(
        default_factory=list
    )

    invalid_evidence_ids: list[str] = Field(
        default_factory=list
    )

    reason: str


validation_llm = llm.with_structured_output(
    CitationValidation,
    method="json_mode"
)


# ============================================================
# 16.5 Failure Classification
#
# Citation Validation 失败以后，
# 判断失败属于哪一种类型。
# ============================================================

class FailureClassification(BaseModel):

    failure_type: Literal[
        "unsupported_claim",
        "wrong_citation",
        "missing_citation",
        "answer_structure"
    ]

    reason: str


failure_classifier_llm = llm.with_structured_output(
    FailureClassification,
    method="json_mode"
)


class RepairedAnswer(BaseModel):

    answer: str

    evidence_ids: list[str]

answer_repair_llm = llm.with_structured_output(
    RepairedAnswer,
    method="json_mode"
)
# ============================================================
# 9. Retrieval Tool
# ============================================================

@tool
def retrieve_knowledge(query: str) -> dict:             ##创建检索器，检索的context 和evidence
    """
    从真实 MyRAG 中检索知识，
    返回 Context + Evidence。

    Context：
        用于 LLM 阅读正文。

    Evidence：
        只保存定位信息，不保存 content。
    """

    print("\n" + "=" * 70)
    print("执行 Retrieval")
    print("Query:", query)
    print("=" * 70)

    # ========================================================
    # 调用真实 MyRAG
    # ========================================================

    result = retrieve_context(query)

    

    final_docs = result["final_docs"]

    # ========================================================
    # 从最终 Document 的 metadata 构造 Evidence
    #
    # 不使用 result["sources"]
    #
    # 因为 sources 是 get_sources(docs) 的结果，
    # 它不是完整 metadata。
    # ========================================================

    evidence_list = []

    for doc in final_docs:

        metadata = doc.metadata

        evidence = {
            "id": "",
            "source": metadata.get("source", ""),
            "sheet": metadata.get("sheet", ""),
            "chunk_id": metadata.get("chunk_id", -1),
            "content": doc.page_content
        }

        evidence_list.append(
            evidence
        )

    return {
        
        "evidence": evidence_list
    }

# ============================================================
# 10. Evidence ID
# ============================================================

def build_evidence_id(
    evidence: Evidence
) -> str:

    return (
        f'{evidence["source"]}'
        f'|{evidence["sheet"]}'
        f'|{evidence["chunk_id"]}'
        
    )


# ============================================================
# 11. Evidence Dedup
# ============================================================

def add_evidence(                              ###新证据一个个和就证据比对全部为flash则表示该新证据为全新证据加入all evidence       
    all_evidence: list[Evidence],
    new_evidence: list[Evidence]
):

    for evidence in new_evidence:

        evidence_id = build_evidence_id(
            evidence
        )

        evidence["id"] = evidence_id

        exists = any(
            item["source"] == evidence["source"]
            and item["sheet"] == evidence["sheet"]
            and item["chunk_id"] == evidence["chunk_id"]
            for item in all_evidence
        )

        if not exists:

            all_evidence.append(
                evidence
            )


# ============================================================
# 12. Query Rewrite
# ============================================================

def rewrite_query(                           #当第一个while 循环的最后一个模块evidence evoluation发现证据不足时调用它进行问题重写二次检索
    original_question: str,
    missing_information: list[str]
) -> str:

    prompt = f"""
你是 Query Rewrite Agent。

你的任务：

根据用户原始问题和当前 Evidence Gap，
生成下一轮知识库检索 Query。

你的任务不是回答问题。

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

{{
    "query": "王建国研究方向"
}}

【字段定义】

- query：字符串。
- query 是下一轮实际用于知识库检索的查询语句。

【重要要求】

1. 必须输出合法 JSON。
2. 不要输出 Markdown。
3. 不要使用 ```json。
4. 不要输出 JSON 之外的任何文字。
5. 不要增加其他字段。
"""

    result = query_rewriter.invoke(
        [
            SystemMessage(
                content=(
                    "你负责生成精准的知识库检索 Query。"
                )
            ),
            HumanMessage(
                content=prompt
            )
        ]
    )

    return result.query


# ============================================================
# 13. Evidence Evaluation
#
# 注意：
#
# 这里是 Grounding 的“门”
#
# sufficient = False
# → 不允许进入 Grounding
#
# ============================================================

def evaluate_evidence(                             #证据evidence和question判断证据是否足够回答question
    question: str,
   
    evidence: list[Evidence]
) -> RetrievalDecision:

    evidence_info = "\n".join(
        [
            f'- {item["id"]} | '
            f'{item["source"]} | '
            f'{item["sheet"]} | '
            f'{item["content"]} | '
            f'chunk={item["chunk_id"]}'
            for item in evidence
        ]
    )

    prompt = f"""
你是 Evidence Evaluation Agent。

你的唯一任务是：

判断当前已经累计获得的知识内容，
是否已经包含回答用户问题所需要的全部关键事实。

你不是 Grounding Agent。

你不能生成答案。

你不能判断 Citation。

【用户问题】

{question}



【当前 Evidence】

{evidence_info}

【判断标准】

sufficient = true：

表示当前知识内容已经包含回答用户问题所需要的关键事实，
可以进入 Grounding 阶段。

sufficient = false：

表示当前知识内容缺少回答用户问题所需要的重要事实，
必须继续进行 Evidence Gap 和 Retrieval。

【重要规则】

必须判断：

“当前知识内容是否足以回答用户的问题？”

不要因为存在 Evidence 就自动判断 true。

Evidence 只是定位信息。

真正判断依据是当前累计知识内容。

如果问题包含多个信息要求，
必须检查这些要求是否都已经得到支持。

【JSON 输出格式】

必须严格输出：

{{
    "sufficient": true,
    "reason": "当前知识内容已经包含回答问题所需的全部关键事实。"
}}

【字段】

- sufficient：布尔值，只能是 true 或 false。
- reason：字符串。

【重要要求】

1. 必须输出合法 JSON。
2. 不要输出 Markdown。
3. 不要使用 ```json。
4. 不要输出 JSON 之外的任何文字。
5. 不要增加其他字段。
"""

    return retrieval_evaluator.invoke(
        [
            SystemMessage(
                content=(
                    "你负责判断 Evidence 是否足以进入 Grounding。"
                )
            ),
            HumanMessage(
                content=prompt
            )
        ]
    )



# ============================================================
# 14. Evidence Gap
# ============================================================

def detect_evidence_gap(             #输出针对性月目标的却是信息方便重写机制重写
    question: str,
    evidence: list[Evidence]
) -> EvidenceGap:



    evidence_info = "\n".join(
        [
            f"""
        Evidence ID: {item["id"]}
        source: {item["source"]}
        sheet: {item["sheet"]}
        chunk_id: {item["chunk_id"]}
        content:
        {item["content"]}
        """
            for item in evidence
        ]
    )


    prompt = f"""
你是 Evidence Gap Detection Agent。

你的任务是：

比较用户问题和当前已有知识，
找出回答用户问题所缺少的具体信息。

【用户问题】

{question}

【当前 Evidence】

{evidence_info}
【判断规则】

如果当前已有知识已经包含回答用户问题
所需要的全部关键事实：

has_gap = false

missing_information = []

如果当前已有知识缺少关键事实：

has_gap = true

missing_information 必须列出具体缺失的信息。

【missing_information 要求】

1. 必须描述具体缺失的信息。
2. 不要写“更多信息”。
3. 不要写“相关信息”。
4. 每一项都应该能够指导下一次检索。
5. 不要回答用户问题。
6. 不要重复已经存在的信息。

【示例】

用户问题：

王建国的职称和研究方向是什么？

当前知识：

王建国，教授。

正确结果：

{{
    "has_gap": true,
    "missing_information": [
        "王建国的研究方向"
    ]
}}

【JSON 输出格式】

必须严格输出：

{{
    "has_gap": true,
    "missing_information": [
        "字符串"
    ]
}}

【重要要求】

1. 必须输出合法 JSON。
2. 不要输出 Markdown。
3. 不要使用 ```json。
4. 不要输出 JSON 之外的任何文字。
5. 不要增加其他字段。
"""

    return evidence_gap_evaluator.invoke(
        [
            SystemMessage(
                content=(
                    "你负责发现当前 Evidence 中缺失的信息。"
                )
            ),
            HumanMessage(
                content=prompt
            )
        ]
    )


# ============================================================
# 15. Grounding
#
# 注意：
#
# 只有 Evidence Evaluation = True
# 才会执行这个函数
# ============================================================

def generate_grounded_answer(##这个模块才刚开始脱离第一个while循环，开始工具第一个while提供的问题上下文和证据，回答问题和与该问题有关的证据（当然这个证据筛选的过程会筛选掉一批过一次while阶段检索器检索出来但是无关的证据，也有可能筛选掉一批有关的证据导致远不证据足够现在不够了这部分就要靠后续的来源校验判别有repair处理了）
    question: str,
   
    evidence: list[Evidence]
) -> GroundedAnswer:

    evidence_info = "\n".join(
    [
            f"""
    Evidence ID: {item["id"]}
    source: {item["source"]}
    sheet: {item["sheet"]}
    chunk_id: {item["chunk_id"]}
    content:
    {item["content"]}
    """
            for item in evidence
        ]
    )

    prompt = f"""
你是 Grounded Answer Agent。

当前 Evidence Evaluation 已经确认：

Evidence 足以回答用户问题。

你的任务是：

严格依据当前知识库内容回答用户问题，
并列出真正支持答案的 Evidence ID。

【用户问题】

{question}



【可用 Evidence ID】

{evidence_info}

【回答规则】

1. 只能使用知识库中的事实。
2. 不允许使用知识库之外的事实。
3. 不允许猜测。
4. 不允许进行没有证据支持的推断。
5. answer 必须直接回答用户问题。
6. evidence_ids 只能使用当前存在的 Evidence ID。
7. 只引用真正支持答案的 Evidence。
8. 不要引用无关 Evidence。

【JSON 输出格式】

必须严格输出：

{{
    "answer": "字符串",
    "evidence_ids": [
        "Evidence ID"
    ]
}}

【重要要求】

1. 必须输出合法 JSON。
2. 不要输出 Markdown。
3. 不要使用 ```json。
4. 不要输出 JSON 之外的任何文字。
5. 不要增加其他字段。
"""

    return grounding_llm.invoke(
        [
            SystemMessage(
                content=(
                    "你负责生成严格基于 Evidence 的答案。"
                )
            ),
            HumanMessage(
                content=prompt
            )
        ]
    )

# ============================================================
# 16. Citation Validation
#
# 核心职责只有一个：
#
# Answer ↔ Evidence
#
# 是否成立？
#
# ============================================================

def validate_citations(                            #这个模块负责检查上一个模块基于问题回答的答案是否有过多瞎编即是否answer的每一个claim都有证据支持
    question: str,
    answer: GroundedAnswer,
    
    evidence: list[Evidence]
) -> CitationValidation:

    # --------------------------------------------------------
    # 第一层：
    # 程序检查 ID 是否存在
    # --------------------------------------------------------

    valid_ids = {
        item["id"]
        for item in evidence
    }

    invalid_ids = [
        evidence_id
        for evidence_id in answer.evidence_ids
        if evidence_id not in valid_ids
    ]

    if invalid_ids:

        return CitationValidation(
            valid=False,
            reason=(
                "Answer 引用了当前 Evidence 集合中不存在的 "
                "Evidence ID。"
            ),
            invalid_evidence_ids=invalid_ids
        )

    # --------------------------------------------------------
    # 第二层：
    # LLM 只检查：
    #
    # Answer 中的事实性主张
    # 是否被引用 Evidence 支持
    # --------------------------------------------------------

    cited_evidence = []

    for item in evidence:

        if item["id"] in answer.evidence_ids:

            cited_evidence.append(
                item
            )

    cited_evidence_info = "\n".join(
        [
            f"""
    Evidence ID: {item["id"]}
    source: {item["source"]}
    sheet: {item["sheet"]}
    chunk_id: {item["chunk_id"]}
    content:
    {item["content"]}
    """
            for item in cited_evidence
        ]
    )

    prompt = f"""
你是 Citation Validation Agent。

你的唯一职责是：

验证最终 Answer 中已经明确表达的事实性主张，
是否被 Answer 所引用的 Evidence 支持。

你不是 Evidence Evaluation Agent。

你不负责判断：

“Evidence 是否足以完整回答用户问题”。

你只负责判断：

“Answer 中已经说出来的事实，
是否能够由它引用的 Evidence 支持。”

【用户问题】

{question}

【最终 Answer】

{answer.answer}

【Answer 引用的 Evidence ID】

{answer.evidence_ids}

【被引用的 Evidence】

{cited_evidence_info}

【验证规则】

逐项检查 Answer 中的事实性主张。

如果：

Answer 中的事实能够被引用 Evidence 明确支持，

则：

valid = true

如果存在任何一个事实性主张无法被引用 Evidence 支持，

则：

valid = false

并在 invalid_evidence_ids 中列出
与错误引用直接相关的 Evidence ID。

【特别注意】

不要因为用户问题没有被完整回答，
就把 valid 设置为 false。

“问题没有完全回答”
属于 Evidence Evaluation 的职责。

这里唯一判断：

Answer ↔ Evidence

是否成立。

例如：

Question：

王建国的职称和研究方向是什么？

Evidence：

王建国，教授。

Answer：

王建国是教授。

那么：

valid = true

因为 Answer 中的事实得到了 Evidence 支持。

如果 Answer 是：

王建国是教授，研究方向为人工智能教育。

而 Evidence 中没有研究方向，

则：

valid = false
【JSON 输出格式】

必须严格输出：

{{
    "valid": true,
    "unsupported_claims": [],
     "invalid_evidence_ids": [],
    "reason": "Answer 中的事实性主张均能够被引用的 Evidence 支持。"
}}

如果存在无法被 Evidence 支持的事实性主张：

{{
    "valid": false,
    "unsupported_claims": [
        "王建国获得过国家级教学名师称号"
    ],
     "invalid_evidence_ids": [],
    "reason": "该主张没有出现在引用的 Evidence 中。"
}}

【一致性要求】

如果 valid = true：

unsupported_claims 必须是 []

如果 valid = false：

unsupported_claims 必须至少包含一个元素。

【特别注意】

你只检查：

Answer ↔ 引用的 Evidence

不要检查：

Question ↔ Evidence

不要因为 Answer 没有完整回答 Question
而判定 valid=false。

如果 Answer 中明确说：

“现有证据没有提供研究方向，因此无法确定。”

这不是 unsupported claim。

你只判断 Answer 已经明确表达的事实性主张
是否能够被引用 Evidence 支持。
"""

    return validation_llm.invoke(
        [
            SystemMessage(
                content=(
                    "你只负责验证 Answer 与 Evidence 的对应关系。"
                )
            ),
            HumanMessage(
                content=prompt
            )
        ]
    )



# ============================================================
# Failure Classification
# ============================================================

def classify_failure(
    question: str,
    answer: GroundedAnswer,
    validation: CitationValidation,
    evidence: list[Evidence]
) -> FailureClassification:

    evidence_info = "\n".join(
        [
            f"""
        Evidence ID: {item["id"]}
        source: {item["source"]}
        sheet: {item["sheet"]}
        chunk_id: {item["chunk_id"]}
        content:
        {item["content"]}
        """
            for item in evidence
        ]
    )

    prompt = f"""
你是 Failure Classification Agent。

你的唯一职责是：

根据 Citation Validation 的失败结果，
判断当前 Answer 最主要属于哪一种失败类型。

你不能重新检索。

你不能修改 Evidence。

你不能直接修复 Answer。

你只负责诊断失败类型。

【用户问题】

{question}

【当前 Answer】

{answer.answer}

【当前 Evidence IDs】

{answer.evidence_ids}

【当前全部 Evidence】

{evidence_info}

【Citation Validation】

valid = {validation.valid}

unsupported_claims =
{validation.unsupported_claims}

invalid_evidence_ids =
{validation.invalid_evidence_ids}

reason =
{validation.reason}

【失败类型】

1. unsupported_claim

含义：

Answer 中出现了 Evidence 无法支持的事实性主张。

例如：

Evidence：
王建国是教授。

Answer：
王建国是教授，研究方向是人工智能。

如果 Evidence 没有研究方向信息，
则属于 unsupported_claim。

--------------------------------------------------

2. wrong_citation

含义：

Answer 中的事实可能有 Evidence 支持，
但是 Answer 当前引用了错误的 Evidence ID。

例如：

Answer：
王建国是教授。

真正支持该事实的是 E2，
但 Answer 引用了 E5。

则属于 wrong_citation。

--------------------------------------------------

3. missing_citation

含义：

Answer 中的事实有 Evidence 支持，
但是 Answer 没有引用对应的 Evidence。

例如：

Evidence：
E2：王建国是教授。

Answer：
王建国是教授。

但是：

evidence_ids = []

则属于 missing_citation。

--------------------------------------------------

4. answer_structure

含义：

Answer 的事实基本能够得到 Evidence 支持，
但是 Answer 的组织结构存在明显问题。

例如：

回答重复、
顺序混乱、
没有直接回答问题、
表达结构明显不合理。

【判断优先级】

如果存在明确的 unsupported claim：

优先选择：

unsupported_claim

如果事实本身可以由 Evidence 支持，
但引用的 Evidence 不正确：

选择：

wrong_citation

如果事实有 Evidence 支持，
但是没有正确引用：

选择：

missing_citation

只有前三种都无法解释问题，
并且主要问题是答案组织方式时：

选择：

answer_structure

【重要】

不要因为 Evidence 不足而选择任何失败类型。

Evidence 是否足够属于：

Evidence Evaluation

不是你的职责。

【JSON 输出格式】

必须严格输出：

{{
    "failure_type": "unsupported_claim",
    "reason": "当前 Answer 包含没有被引用 Evidence 支持的事实性主张。"
}}

failure_type 只能是：

"unsupported_claim"

"wrong_citation"

"missing_citation"

"answer_structure"

【重要要求】

1. 必须输出合法 JSON。
2. 不要输出 Markdown。
3. 不要使用 ```json。
4. 不要输出 JSON 之外的任何文字。
5. 不要增加其他字段。
"""

    return failure_classifier_llm.invoke(
        [
            SystemMessage(
                content=(
                    "你负责诊断 Citation Validation 失败原因。"
                )
            ),
            HumanMessage(
                content=prompt
            )
        ]
    )



def repair_answer(
    question: str,
    answer: GroundedAnswer,
    validation: CitationValidation,
    failure: FailureClassification,
   
    evidence: list[Evidence]
) -> RepairedAnswer:

    evidence_info = "\n".join(
        [
            f'''
    Evidence ID: {item["id"]}
    source: {item["source"]}
    sheet: {item["sheet"]}
    chunk_id: {item["chunk_id"]}
    content:
    {item["content"]}
    '''
            for item in evidence
        ]
    )
    prompt = f"""
你是 Agentic RAG 的 Adaptive Answer Repair Agent。

当前 Evidence 已经通过 Evidence Evaluation。

因此：

Evidence 不需要重新检索。

你的任务是：

根据 Failure Classification 的结果，
使用对应的 Repair Strategy 修复当前 Answer。

【用户问题】

{question}

【当前 Answer】

{answer.answer}

【当前 Evidence IDs】

{answer.evidence_ids}

【当前 Evidence】

{evidence_info}

【Citation Validation】

valid = {validation.valid}

unsupported_claims =
{validation.unsupported_claims}

invalid_evidence_ids =
{validation.invalid_evidence_ids}

reason =
{validation.reason}

【Failure Classification】

failure_type =
{failure.failure_type}

reason =
{failure.reason}

============================================================
Repair Strategy
============================================================

如果：

failure_type = "unsupported_claim"

执行：

Unsupported Claim Repair

规则：

1. 找出没有 Evidence 支持的事实。
2. 删除这些事实，或者将其改写为“现有证据无法确定”。
3. 保留 Evidence 能够支持的事实。
4. 不允许自行补充新事实。
5. 不允许重新检索。
6. evidence_ids 只能保留真正支持修复后 Answer 的 Evidence。

------------------------------------------------------------

如果：

failure_type = "wrong_citation"

执行：

Wrong Citation Repair

规则：

1. 保留 Evidence 能够支持的事实。
2. 找到真正支持这些事实的 Evidence ID。
3. 删除错误的 Evidence ID。
4. 加入真正支持 Answer 的 Evidence ID。
5. 不要因为 Citation 错误而删除本身正确的事实。
6. 不允许重新检索。
7. evidence_ids 只能使用当前 Evidence 中存在的 ID。

------------------------------------------------------------

如果：

failure_type = "missing_citation"

执行：

Missing Citation Repair

规则：

1. 保留已经被 Evidence 支持的事实。
2. 为这些事实补充正确的 Evidence ID。
3. 不要删除已经被 Evidence 支持的事实。
4. 不允许添加没有 Evidence 支持的新事实。
5. evidence_ids 只能使用当前 Evidence 中存在的 ID。

------------------------------------------------------------

如果：

failure_type = "answer_structure"

执行：

Answer Structure Repair

规则：

1. 保留 Evidence 支持的事实。
2. 重新组织答案结构。
3. 让答案直接回答用户问题。
4. 删除重复表达。
5. 删除无关内容。
6. 不允许添加新事实。
7. evidence_ids 必须继续对应真正支持答案的 Evidence。

============================================================
通用规则
============================================================

1. 不允许重新检索。
2. 不允许修改 Evidence。
3. 不允许编造事实。
4. 不允许使用 Evidence 之外的信息。
5. evidence_ids 只能使用当前存在的 Evidence ID。
6. evidence_ids 只能引用真正支持修复后 Answer 的 Evidence。
7. 修复后的 Answer 必须直接回答用户问题。
8. 如果 Evidence 无法支持某项事实，可以明确说明无法确定。
9. 不要为了让答案看起来完整而编造信息。

【JSON 输出格式】

必须严格输出：

{{
    "answer": "修复后的答案",
    "evidence_ids": [
        "Evidence ID"
    ]
}}

【重要要求】

1. 必须输出合法 JSON。
2. 不要输出 Markdown。
3. 不要使用 ```json。
4. 不要输出 JSON 之外的任何文字。
5. 不要增加其他字段。
"""

    return answer_repair_llm.invoke(
        [
            SystemMessage(
                content=(
                    "你负责根据失败类型执行 Adaptive Answer Repair。"
                )
            ),
            HumanMessage(
                content=prompt
            )
        ]
    )
# ============================================================
# 17. 主 Agentic RAG
#
# 核心架构：
#
# Retrieval
# ↓
# Evidence Evaluation
# ↓
# 不够 → Gap → Query → Retrieval
# ↓
# 足够
# ↓
# Grounding
# ↓
# Citation Validation
#
# ============================================================

def agentic_rag(                ##这是全局编排
    question: str
):

    print("\n")
    print("#" * 70)
    print("Agentic RAG")
    print("#" * 70)

    print("\n用户问题：")
    print(question)

    # ========================================================
    # Evidence Accumulation
    # ========================================================


    all_evidence = []

    trace = []

    retrieval_count = 0
    
    current_query = question

    evidence_sufficient = False

    status = "retrieval_failure"

    # ========================================================
    # Evidence Acquisition Loop
    # ========================================================

    while retrieval_count < MAX_RETRIEVALS:

        retrieval_count += 1

        trace.append({
            "step": len(trace) + 1,
            "event": "retrieval",
            "detail": f"第 {retrieval_count} 次 Retrieval，Query={current_query}"
        })

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

   

        evidence = result["evidence"]

        # ----------------------------------------------------
        # Evidence Accumulation
        # ----------------------------------------------------

       
        add_evidence(
            all_evidence,
            evidence
        )

      
        print("\n当前累计 Evidence：")

        for item in all_evidence:

            print(
                f'  {item["id"]}'
            )

        # ====================================================
        # Evidence Evaluation
        # ====================================================

        evaluation = evaluate_evidence(
            question=question,
            
            evidence=all_evidence
        )


        trace.append({
            "step": len(trace) + 1,
            "event": "evidence_evaluation",
            "detail": (
                "Evidence sufficient"
                if evaluation.sufficient
                else "Evidence insufficient"
            )
        })

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
        # 关键门槛
        #
        # sufficient = True
        # 才允许离开 Retrieval Loop
        # ----------------------------------------------------

        if evaluation.sufficient:

            evidence_sufficient = True

            print(
                "\nEvidence 已经足够。"
            )

            print(
                "停止 Evidence Acquisition Loop。"
            )

            break

        # ====================================================
        # Evidence 不足
        # ====================================================

        print(
            "\nEvidence 不足。"
        )

        # ----------------------------------------------------
        # Search Budget
        # ----------------------------------------------------

        if retrieval_count >= MAX_RETRIEVALS:

            print(
                "\n达到 MAX_RETRIEVALS。"
            )

            print(
                "Evidence 仍然不足。"
            )

            break

        # ====================================================
        # Evidence Gap
        # ====================================================

        gap = detect_evidence_gap(
            question=question,
            evidence=all_evidence
        )

        trace.append({
            "step": len(trace) + 1,
            "event": "evidence_gap",
            "detail": str(gap.missing_information)
        })

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
                "\n没有发现明确 Evidence Gap。"
            )

            print(
                "停止 Retrieval。"
            )

            break

        # ====================================================
        # Targeted Retrieval
        # ====================================================

        current_query = rewrite_query(
            original_question=question,
            missing_information=(
                gap.missing_information
            )
        )

        trace.append({
            "step": len(trace) + 1,
            "event": "targeted_retrieval",
            "detail": current_query
        })

        print(
            "\nTargeted Query："
        )

        print(
            current_query
        )

    # ========================================================
    # 关键安全门
    #
    # 如果 Evidence 不够：
    #
    # 不允许 Grounding
    # 不允许 Citation Validation
    #
    # ========================================================

    if not evidence_sufficient:

        print("\n")
        print("=" * 70)
        print("Evidence Acquisition 未成功")
        print("=" * 70)

        print(
            "\n当前 Evidence 不足以可靠回答问题。"
        )

        print(
            "因此不会进入 Grounding。"
        )

        print(
            "Citation Validation 也不会执行。"
        )



        trace.append({
            "step": len(trace) + 1,
            "event": "retrieval_failure",
            "detail": "达到 Retrieval Budget 后 Evidence 仍然不足"
        })

        return {                  ##必须放在if内部否则无论if是否处理rag agemt都会立刻结束
                "question": question,
                "evidence": all_evidence,
                "retrieval_count": retrieval_count,
                "evidence_sufficient": False,
                "answer": None,
                "evidence_ids": [],
                "citation_valid": None,
                "citation_reason": None,
                "trace": trace
        }

    # ========================================================
    # 到这里意味着：
    #
    # Evidence Evaluation = True
    #
    # 才允许 Grounding
    # ========================================================

   

    # ========================================================
    # Grounding  + Citation Validation
    # ========================================================

    print("\n")
    print("=" * 70)
    print("Grounded Answer")
    print("=" * 70)

    # ========================================================
# Grounding + Citation Repair Loop
# ========================================================

    answer = generate_grounded_answer(
        question=question,
      
        evidence=all_evidence
    )

    trace.append({

    "step": len(trace) + 1,
    "event": "grounding",
    "detail": "Evidence 足够，生成 Grounded Answer"
    })



    repair_count = 0

    status = "retrieval_failure"

    while True:

        print("\n")
        print("=" * 70)
        print(
            f"CITATION VALIDATION | repair_count={repair_count}"
        )
        print("=" * 70)

        validation = validate_citations(
            question=question,
            answer=answer,
         
            evidence=all_evidence
        )
         
        trace.append({
            "step": len(trace) + 1,
            "event": "citation_validation",
            "detail": (
                "PASS"
                if validation.valid
                else "FAIL"
            )
        })




        print("\nvalid：")
        print(validation.valid)

        print("\nunsupported_claims：")
        print(validation.unsupported_claims)

        print("\nreason：")
        print(validation.reason)

        # ====================================================
        # Citation Validation PASS
        # ====================================================

        if validation.valid:

            print("\nCitation Validation PASS")

            status = "success"

            break

        # ====================================================
        # Citation Validation FAIL
        # ====================================================

        print("\nCitation Validation FAIL")
        

        # ====================================================
        # Failure Classification
        # ====================================================

        failure = classify_failure(
            question=question,
            answer=answer,
            validation=validation,
            evidence=all_evidence
        )


        trace.append({
            "step": len(trace) + 1,
            "event": "failure_classification",
            "detail": (
                f"type={failure.failure_type}; "
                f"reason={failure.reason}"
            )
        })

        print("\nFailure Classification：")

        print(
            "failure_type =",
            failure.failure_type
        )

        print(
            "reason =",
            failure.reason
        )



        # ----------------------------------------------------
        # Repair Budget
        # ----------------------------------------------------

        if repair_count >= MAX_ANSWER_REPAIRS:

            print(
                "\n达到 MAX_ANSWER_REPAIRS。"
            )

            print(
                "停止 Answer Repair。"
            )

            print(
                "进入 Safe Fallback。"
            )

            status = "safe_fallback"

            answer = GroundedAnswer(
                answer=(
                    "当前检索到的证据无法可靠支持一个"
                    "经过引用校验的完整答案。"
                    "因此没有输出未经验证的结论。"
                ),
                evidence_ids=[]
            )


            trace.append({
                "step": len(trace) + 1,
                "event": "safe_fallback",
                "detail": "达到 MAX_ANSWER_REPAIRS，进入 Safe Fallback"
            })

            break

        # ----------------------------------------------------
        # Answer Repair
        # ----------------------------------------------------

        repair_count += 1

        print("\n")
        print("=" * 70)
        print(
            f"ANSWER REPAIR | repair_count={repair_count}"
        )
        print("=" * 70)

        answer = repair_answer(
        question=question,
        answer=answer,
        validation=validation,
        failure=failure,
        
        evidence=all_evidence
        )


        trace.append({
            "step": len(trace) + 1,
            "event": "answer_repair",
            "detail": f"第 {repair_count} 次 Answer Repair"
        })

        print("\n修复后的 Answer：")

        print(
            answer.answer
        )

        print("\n修复后的 Evidence IDs：")

        print(
            answer.evidence_ids
        )
    return {               ##此处的return不能放在while true内部负责无论循环有没有结束出现第一轮while以后都会自动结束整个rag agent
    "question": question,
    
    "evidence": all_evidence,

    "answer": answer.answer,
    "evidence_ids": answer.evidence_ids,

    "citation_valid": (
    validation.valid
    if status == "success"
    else None
    ),

    "citation_reason": validation.reason,

    "unsupported_claims": (
    validation.unsupported_claims
    if status == "success"
    else []
    ),

    "retrieval_count": retrieval_count,
    "repair_count": repair_count,

    "evidence_sufficient": True,

    "status": status,
    "trace": trace
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