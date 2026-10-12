import os                   #把llm作为answer正确错误评价的工具代替之前的语义相似度判断
import json


# ============================================================
# 1. Environment
# ============================================================

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


# ============================================================
# 2. Imports
# ============================================================

from typing import Literal

from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI


from agentic_rag_19_architecture import (
    agentic_rag
)

from agentic_rag_20_3_dataset import (
    TEST_CASES
)


# ============================================================
# 3. LLM
# ============================================================

llm = ChatOpenAI(
    model="deepseek-v4-flash",
    api_key=os.environ[
        "ANTHROPIC_AUTH_TOKEN"
    ],
    base_url="https://api.deepseek.com"
)


# ============================================================
# 4. LLM Judge Result
# ============================================================

class LLMJudgeResult(BaseModel):

    correct: bool = Field(
        description="Whether the actual answer correctly answers the question."
    )

    reason: str = Field(
        description="Reason explaining why the answer is correct or incorrect."
    )


# ============================================================
# 5. LLM Judge
# ============================================================

judge_llm = llm.with_structured_output(
    LLMJudgeResult,
    method="json_mode"
)


def judge_answer(
    question,
    expected_answer,
    actual_answer
):
    """
    使用 LLM 判断实际答案是否正确。

    注意：

    DeepSeek JSON Mode 不会自动获得
    Pydantic Schema。

    因此 Prompt 中必须明确规定：
    - JSON
    - 字段
    - 字段类型
    - 不允许 Markdown
    """

    prompt = f"""
你是一个严格的答案评估器。

你的任务是判断“实际答案”是否正确回答了“问题”。

你会获得：

1. 问题
2. 参考答案
3. 实际答案

请根据参考答案判断实际答案是否正确。

判断规则：

- 如果实际答案表达的核心事实与参考答案一致，
  即使措辞不同，也可以判定为 correct=true。

- 不要求实际答案与参考答案文字完全一致。

- 如果实际答案只是换了一种表达方式，
  但核心事实一致，应判定为 correct=true。

- 如果实际答案遗漏了问题要求的关键事实，
  应判定为 correct=false。

- 如果实际答案包含与参考答案冲突的关键事实，
  应判定为 correct=false。

- 如果实际答案无法回答问题，
  应判定为 correct=false。

- 不要因为文字相似就直接判定正确。
  必须关注实际答案表达的事实是否正确。

输出必须是合法 JSON。

JSON 格式必须严格为：

{{
    "correct": true,
    "reason": "这里解释判断原因"
}}

字段要求：

correct:
boolean，只能是 true 或 false。

reason:
string，用简短文字解释判断理由。

禁止：

- Markdown
- ```json
- 额外字段
- JSON 之外的任何文字


问题：
{question}

参考答案：
{expected_answer}

实际答案：
{actual_answer}
"""


    result = judge_llm.invoke(
        prompt
    )


    # ========================================================
    # 6. Structured Output
    # ========================================================

    if isinstance(          #判断第一个参数对象是不是第二个参数类型的实例
        result,
        LLMJudgeResult
    ):

        return result


    # ========================================================
    # 7. Fallback Parsing
    # ========================================================

    if isinstance(
        result,
        dict
    ):

        return LLMJudgeResult(    #将返回的字典转化成LLM JudgeR...对象
            **result
        )


    # ========================================================
    # 8. Unexpected Output
    # ========================================================

    raise ValueError(    #最后的兜底
        f"Unexpected LLM Judge output: {result}"
    )


# ============================================================
# 9. Workflow Evaluation
# ============================================================

def evaluate_workflow(trace):

    events = [
        item["event"]
        for item in trace
    ]

    violations = []


    # --------------------------------------------------------
    # Rule 1
    # retrieval_failure must terminate workflow
    # --------------------------------------------------------

    for index, event in enumerate(events):

        if event == "retrieval_failure":

            if index != len(events) - 1:

                violations.append(
                    "retrieval_failure "
                    "must be the final event."
                )


    # --------------------------------------------------------
    # Rule 2
    # safe_fallback must terminate workflow
    # --------------------------------------------------------

    for index, event in enumerate(events):

        if event == "safe_fallback":

            if index != len(events) - 1:

                violations.append(
                    "safe_fallback "
                    "must be the final event."
                )


    # --------------------------------------------------------
    # Rule 3
    # Evidence Evaluation
    # --------------------------------------------------------

    for index, event in enumerate(events[:-1]):

        if event != "evidence_evaluation":

            continue


        detail = trace[index].get(
            "detail",
            ""
        ).lower()


        next_event = events[
            index + 1
        ]


        if "insufficient" in detail:

            if next_event != "evidence_gap":

                violations.append(
                    "Insufficient evidence "
                    "must be followed by "
                    "evidence_gap."
                )


        elif "sufficient" in detail:

            if next_event != "grounding":

                violations.append(
                    "Sufficient evidence "
                    "must be followed by "
                    "grounding."
                )


    # --------------------------------------------------------
    # Rule 4
    # Citation Validation Failure
    # --------------------------------------------------------

    for index, event in enumerate(events[:-1]):

        if event != "citation_validation":

            continue


        detail = trace[index].get(
            "detail",
            ""
        ).lower()


        if "fail" in detail:

            next_event = events[
                index + 1
            ]


            if next_event != "failure_classification":

                violations.append(
                    "Citation validation failure "
                    "must be followed by "
                    "failure_classification."
                )


    # --------------------------------------------------------
    # Rule 5
    # Answer Repair
    # --------------------------------------------------------

    for index, event in enumerate(events[:-1]):

        if event != "answer_repair":

            continue


        next_event = events[
            index + 1
        ]


        if next_event not in {
            "citation_validation",
            "safe_fallback"
        }:

            violations.append(
                "answer_repair must be followed "
                "by citation_validation or "
                "safe_fallback."
            )


    return {
        "valid":
            len(violations) == 0,

        "violations":
            violations
    }


# ============================================================
# 10. Evaluate One Case
# ============================================================

def evaluate_case(test_case):

    question = test_case[
        "question"
    ]

    expected_answer = test_case[
        "expected_answer"
    ]


    print("\n")
    print("=" * 70)

    print(
        f"Case: {test_case['id']}"
    )

    print(
        f"Question: {question}"
    )


    # ========================================================
    # Run Agentic RAG
    # ========================================================

    result = agentic_rag(
        question
    )


    # ========================================================
    # Actual Answer
    # ========================================================

    actual_answer = result.get(
        "answer",
        ""
    )


    # ========================================================
    # LLM Judge
    # ========================================================

    judge_result = judge_answer(
        question=question,
        expected_answer=expected_answer,
        actual_answer=actual_answer
    )


    # ========================================================
    # Evidence
    # ========================================================

    evidence_sufficient = result.get(
        "evidence_sufficient",
        False
    )


    # ========================================================
    # Citation
    # ========================================================

    citation_valid = result.get(
        "citation_valid",
        False
    )


    # ========================================================
    # Trace
    # ========================================================

    trace = result.get(
        "trace",
        []
    )


    workflow_result = evaluate_workflow(
        trace
    )


    # ========================================================
    # Counters
    # ========================================================

    retrieval_count = result.get(
        "retrieval_count",
        0
    )


    repair_count = result.get(
        "repair_count",
        0
    )


    status = result.get(
        "status",
        "unknown"
    )


    # ========================================================
    # Print Judge Result
    # ========================================================

    print("\nLLM Judge:")

    print(
        f"correct = "
        f"{judge_result.correct}"
    )

    print(
        f"reason = "
        f"{judge_result.reason}"
    )


    return {

        "id":
            test_case["id"],

        "question":
            question,

        "expected_answer":
            expected_answer,

        "actual_answer":
            actual_answer,

        "llm_correct":
            judge_result.correct,

        "llm_reason":
            judge_result.reason,

        "evidence_sufficient":
            evidence_sufficient,

        "citation_valid":
            citation_valid,

        "workflow_valid":
            workflow_result[
                "valid"
            ],

        "workflow_violations":
            workflow_result[
                "violations"
            ],

        "retrieval_count":
            retrieval_count,

        "repair_count":
            repair_count,

        "status":
            status,

        "trace":
            trace
    }


# ============================================================
# 11. Batch Evaluation
# ============================================================

def run_evaluation():

    results = []


    for test_case in TEST_CASES:

        result = evaluate_case(
            test_case
        )


        results.append(
            result
        )


    return results


# ============================================================
# 12. Report
# ============================================================

def print_report(results):

    total = len(
        results
    )


    if total == 0:

        print(
            "No test cases."
        )

        return


    # ========================================================
    # Counts
    # ========================================================

    llm_correct = sum(
        result["llm_correct"]
        for result in results
    )


    evidence_sufficient = sum(
        result["evidence_sufficient"]
        for result in results
    )


    citation_valid = sum(
        result["citation_valid"]
        for result in results
    )


    workflow_valid = sum(
        result["workflow_valid"]
        for result in results
    )


    # ========================================================
    # Report
    # ========================================================

    print("\n\n")

    print("=" * 70)
    print("AGENT LLM-AS-A-JUDGE EVALUATION")
    print("=" * 70)


    print(
        f"\nTotal Cases: "
        f"{total}"
    )


    print(
        f"\nLLM Judge Correct: "
        f"{llm_correct}/{total}"
    )


    print(
        f"LLM Judge Accuracy: "
        f"{llm_correct / total:.2%}"
    )


    print(
        f"\nEvidence Sufficient: "
        f"{evidence_sufficient}/{total}"
    )


    print(
        f"Evidence Sufficiency: "
        f"{evidence_sufficient / total:.2%}"
    )


    print(
        f"\nCitation Valid: "
        f"{citation_valid}/{total}"
    )


    print(
        f"Citation Validity: "
        f"{citation_valid / total:.2%}"
    )


    print(
        f"\nWorkflow Valid: "
        f"{workflow_valid}/{total}"
    )


    print(
        f"Workflow Validity: "
        f"{workflow_valid / total:.2%}"
    )


    # ========================================================
    # Detailed Results
    # ========================================================

    print("\n")
    print("=" * 70)
    print("DETAILED LLM JUDGE RESULTS")
    print("=" * 70)


    for result in results:

        print("\n")


        print(
            f"[{result['id']}]"
        )


        print(
            f"Question:\n"
            f"{result['question']}"
        )


        print(
            f"\nExpected Answer:\n"
            f"{result['expected_answer']}"
        )


        print(
            f"\nActual Answer:\n"
            f"{result['actual_answer']}"
        )


        print(
            f"\nLLM Judge Correct: "
            f"{result['llm_correct']}"
        )


        print(
            f"LLM Judge Reason:\n"
            f"{result['llm_reason']}"
        )


        print(
            f"\nEvidence Sufficient: "
            f"{result['evidence_sufficient']}"
        )


        print(
            f"Citation Valid: "
            f"{result['citation_valid']}"
        )


        print(
            f"Workflow Valid: "
            f"{result['workflow_valid']}"
        )


        print(
            f"Retrieval Count: "
            f"{result['retrieval_count']}"
        )


        print(
            f"Repair Count: "
            f"{result['repair_count']}"
        )


        print(
            f"Status: "
            f"{result['status']}"
        )


        if result[
            "workflow_violations"
        ]:

            print(
                "\nWorkflow Violations:"
            )


            for violation in result[
                "workflow_violations"
            ]:

                print(
                    f"  - {violation}"
                )


# ============================================================
# 13. Main
# ============================================================

if __name__ == "__main__":

    results = run_evaluation()

    print_report(
        results
    )