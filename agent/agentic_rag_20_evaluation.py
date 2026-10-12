import os                              ##测试评估rag agent reault（暂时不涉及llm）

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from typing import Literal

from pydantic import BaseModel, Field

from agentic_rag_19_architecture import agentic_rag


# ============================================================
# 1. Agent Evaluation Result
#
# 这是 Evaluation 的 Result Contract
#
# 不是 Agent State
# 不是 Agent Internal State
#
# 它描述：
# “这一次 Agent 执行得怎么样”
# ============================================================

class AgentEvaluation(BaseModel):          #创建类对象

    answer_correct: bool

    evidence_sufficient: bool

    citation_valid: bool

    retrieval_count: int

    repair_count: int

    status: Literal[
        "success",
        "safe_fallback",
        "retrieval_failure"
    ]

    reason: str


# ============================================================
# 2. 单次 Agent Result Evaluation
# ============================================================

def evaluate_agent_result(
    result: dict,
    expected_answer: str
) -> AgentEvaluation:

    # --------------------------------------------------------
    # Answer Correctness
    #
    # 第一版故意使用最简单的精确匹配。
    #
    # 现在的重点不是研究“语义正确性”
    # 而是先学习：
    #
    # Agent Result
    #       ↓
    # Evaluation
    # --------------------------------------------------------

    answer = result.get(                       ##提取rag agentic返回的信息并赋值变量
        "answer"
    )

    answer_correct = (
        answer == expected_answer
    )

    # --------------------------------------------------------
    # Evidence Sufficiency
    # --------------------------------------------------------

    evidence_sufficient = result.get(
        "evidence_sufficient",
        False
    )

    # --------------------------------------------------------
    # Citation Validity
    #
    # Retrieval Failure 时可能没有执行 Citation Validation
    # 所以不能简单地：
    #
    # result["citation_valid"]
    #
    # --------------------------------------------------------

    citation_valid = result.get(
        "citation_valid",
        False
    )

    if citation_valid is None:
        citation_valid = False

    # --------------------------------------------------------
    # Retrieval Count
    # --------------------------------------------------------

    retrieval_count = result.get(
        "retrieval_count",
        0
    )

    # --------------------------------------------------------
    # Repair Count
    # --------------------------------------------------------

    repair_count = result.get(
        "repair_count",
        0
    )

    # --------------------------------------------------------
    # Status
    # --------------------------------------------------------

    status = result.get(
        "status"
    )

    # --------------------------------------------------------
    # 第一版兼容：
    #
    # 如果你的 19 某个失败分支暂时没有显式 status，
    # 根据已有结果推断。
    #
    # 后续正式版本可以让 19 的所有出口
    # 都显式返回 status。
    # --------------------------------------------------------

    if status is None:

        if not evidence_sufficient:

            status = "retrieval_failure"

        elif answer_correct:

            status = "success"

        else:

            status = "safe_fallback"

    # --------------------------------------------------------
    # Evaluation Reason
    # --------------------------------------------------------

    if status == "retrieval_failure":

        reason = (
            "Agent 未获得足够 Evidence，"
            "因此没有进入可靠的 Grounding 流程。"
        )

    elif status == "safe_fallback":

        reason = (
            "Agent 在 Repair Budget 耗尽后进入 "
            "Safe Fallback，没有输出未经验证的结论。"
        )

    elif answer_correct:

        reason = (
            "最终 Answer 与测试集中的 Expected Answer 一致。"
        )

    else:

        reason = (
            "Agent 完成了执行，但最终 Answer "
            "与 Expected Answer 不一致。"
        )

    return AgentEvaluation(             #创建对象实例形成契约

        answer_correct=answer_correct,

        evidence_sufficient=evidence_sufficient,

        citation_valid=citation_valid,

        retrieval_count=retrieval_count,

        repair_count=repair_count,

        status=status,

        reason=reason
    )


# ============================================================
# 3. Test Cases
#
# 现在先使用非常小的测试集。
#
# 注意：
# 这里的 Expected Answer 是“参考答案”。
#
# 后面学习 LLM Judge 时，
# 我们会解决“答案不可能完全逐字一致”的问题。
# ============================================================

test_cases = [               #测试集

    {
        "question": "赵雪丽是谁",

        "expected_answer": (
            "这里填写你根据 MyRAG 实际数据确认的标准答案。"
        )
    },

    {
        "question": "王建国的职称和研究方向是什么",

        "expected_answer": (
            "这里填写你根据 MyRAG 实际数据确认的标准答案。"
        )
    }

]


# ============================================================
# 4. Batch Evaluation
# ============================================================

if __name__ == "__main__":

    print("\n")
    print("=" * 70)
    print("Agentic RAG Evaluation")
    print("=" * 70)

    evaluations = []

    for index, case in enumerate(
        test_cases,
        start=1
    ):

        print("\n")
        print("-" * 70)
        print(
            f"Test Case {index}"
        )
        print("-" * 70)

        question = case[
            "question"
        ]

        expected_answer = case[
            "expected_answer"
        ]

        print("\nQuestion:")
        print(question)

        # ----------------------------------------------------
        # Agent 执行
        # ----------------------------------------------------

        result = agentic_rag(
            question
        )

        # ----------------------------------------------------
        # Evaluation
        # ----------------------------------------------------

        evaluation = evaluate_agent_result(
            result=result,
            expected_answer=expected_answer
        )

        evaluations.append(
            evaluation
        )

        print("\nEvaluation:")

        print(
            evaluation.model_dump()          #对象实例转字典（pydantic内置）
        )

    # ========================================================
    # 5. Batch Summary
    # ========================================================

    total = len(                ##开始用从rag agentic返回值中提取出来并赋值组成的对象实例，做二次加工sum计算得到各种指标的分子分母，如果直接为数字直接除法
        evaluations
    )

    correct = sum(                             ###在python里面true和flash都可以参数计算记为       1/0
        evaluation.answer_correct
        for evaluation in evaluations
    )

    citation_valid = sum(
        evaluation.citation_valid
        for evaluation in evaluations
    )

    evidence_sufficient = sum(
        evaluation.evidence_sufficient
        for evaluation in evaluations
    )

    average_retrieval = (                     ##除法开始
        sum(
            evaluation.retrieval_count
            for evaluation in evaluations
        )
        / total
    )

    average_repair = (
        sum(
            evaluation.repair_count
            for evaluation in evaluations
        )
        / total
    )

    print("\n")                                ####利用这些指标的分子分母做除法计算指标（即分数或者百分数）
    print("=" * 70)
    print("Evaluation Summary")
    print("=" * 70)

    print(
        f"\nTotal Cases: {total}"
    )

    print(
        f"Answer Correct: "
        f"{correct}/{total}"          #分数格式显示
    )

    print(
        f"Answer Correctness: "
        f"{correct / total:.2%}"           #保留两位小数的百分数形式显示
    )

    print(
        f"Evidence Sufficient: "
        f"{evidence_sufficient}/{total}"
    )

    print(
        f"Evidence Sufficiency: "
        f"{evidence_sufficient / total:.2%}"
    )

    print(
        f"Citation Valid: "
        f"{citation_valid}/{total}"
    )

    print(
        f"Citation Validity: "
        f"{citation_valid / total:.2%}"
    )

    print(
        f"Average Retrieval Count: "
        f"{average_retrieval:.2f}"            #把之前结算的学术进行浮点数保留两位小数化展示
    )

    print(
        f"Average Repair Count: "
        f"{average_repair:.2f}"
    )