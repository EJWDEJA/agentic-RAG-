import json                      ##这一步的作用是把所有可用的评价参数打包成一个契约整体，方便输出和后续提取使用，输出评估系统性能指标
from pathlib import Path               #也就是整理并汇总处理衍生更多20.7agent评估系统评估值
                                      ##也就是进行多维度评估

                                      ##############################################核心核心          这一步进行了指标多维化处理，后续也要进行指标使用，使用要进行参数合法性校验      
from typing import Literal            ##############################################之前的20.0-20.5他们输入的值都是确定正确，生成的参数自己生产的，各自独立输出值不需要给其他测试脚本使用，20.6生产的需要给20.7使用但是仅仅添加两个字段，操作简单，排查bug也简单，没必要专门设置一个参数值合法性校验，20.7-20.8不需要结果参数处理，一般不会错也不需要额外的参数合法性校验，20.8要进行参数字段大量处理指标生产，使用参数有可能记录错误，合法性风险较高，使用需要进行工作流校验之外的合法性校验

from pydantic import BaseModel


# ============================================================
# 1. 文件配置
# ============================================================

INPUT_FILE = Path(                        ###这一步的作用是把字符串转化成路径，可后续使用path相关方法，比如对象.exist()...对象.name()...            变量可以随意自定义
    "agentic_rag_20_7_llm_judge_comparison.json"
)

OUTPUT_FILE = Path(
    "agentic_rag_20_8_evaluation_report.json"
)


# ============================================================
# 2. Evaluation Result Contract
# ============================================================

class AgentEvaluation(BaseModel):               ##创建一个契约
    """
    单个测试样本的最终评估结果。

    注意：
    这里不是 Agent 的内部 State。

    它属于 Evaluation 层的 Result Contract。
    """

    case_id: str

    # --------------------------------------------------------
    # Answer Correctness
    # --------------------------------------------------------

    answer_correct: bool

    # --------------------------------------------------------
    # Evidence
    # --------------------------------------------------------

    evidence_sufficient: bool

    # --------------------------------------------------------
    # Citation
    # --------------------------------------------------------

    citation_valid: bool

    # --------------------------------------------------------
    # Agent 行为指标
    # --------------------------------------------------------

    retrieval_count: int

    repair_count: int

    # --------------------------------------------------------
    # 最终状态
    # --------------------------------------------------------

    status: Literal[
        "success",
        "safe_fallback",
        "retrieval_failure",
    ]

    # --------------------------------------------------------
    # LLM Judge 原因
    # --------------------------------------------------------

    reason: str


# ============================================================
# 3. 安全读取工具
# ============================================================

def get_bool(              ##提取20.7json里面的某些值(key)的true  false(value)
    data: dict,
    key: str,
    default: bool = False
) -> bool:

    value = data.get(key, default)

    if isinstance(value, bool):
        return value

    if isinstance(value, str):

        value = value.lower().strip()

        if value == "true":
            return True

        if value == "false":
            return False

    return default


def get_int(               ####提取20.7json里面的某些值(key)的int(value)
    data: dict,
    key: str,
    default: int = 0
) -> int:

    value = data.get(key, default)

    try:
        return int(value)

    except (TypeError, ValueError):
        return default


def normalize_status(       ##同理提取status
    status: str
) -> str:

    status = str(status).lower().strip()

    if status in {
        "success",
        "safe_fallback",
        "retrieval_failure",
    }:
        return status

    # --------------------------------------------------------
    # 兼容一些可能存在的旧写法
    # --------------------------------------------------------

    if status in {
        "fallback",
        "safe-fallback",
        "safe fallback",
    }:
        return "safe_fallback"

    if status in {
        "retrieval-fail",
        "retrieval_failed",
        "retrieval failure",
    }:
        return "retrieval_failure"

    # 如果未知，按照成功处理，
    # 但后面会在控制台给出提示。
    return "success"


# ============================================================
# 4. 从 20.7 单条结果生成 AgentEvaluation
# ============================================================

def evaluate_case(              ##利用前面的工具制作契约实例
    result: dict
) -> AgentEvaluation:

    # --------------------------------------------------------
    # Answer Correctness
    #
    # 20.7 已经同时存在：
    #
    #   similarity
    #   gold_label
    #   llm_judge
    #
    # 20.8 不再重新判断答案。
    #
    # 直接使用 20.7 已经确定的 gold_label。
    # --------------------------------------------------------

    answer_correct = get_bool(
        result,
        "gold_label",
        False
    )

    # --------------------------------------------------------
    # Evidence
    # --------------------------------------------------------

    evidence_sufficient = get_bool(
        result,
        "evidence_sufficient",
        False
    )

    # --------------------------------------------------------
    # Citation
    # --------------------------------------------------------

    citation_valid = get_bool(
        result,
        "citation_valid",
        False
    )

    # --------------------------------------------------------
    # Retrieval Count
    # --------------------------------------------------------

    retrieval_count = get_int(
        result,
        "retrieval_count",
        0
    )

    # --------------------------------------------------------
    # Repair Count
    # --------------------------------------------------------

    repair_count = get_int(
        result,
        "repair_count",
        0
    )

    # --------------------------------------------------------
    # Status
    # --------------------------------------------------------

    raw_status = result.get(
        "status",
        "success"
    )

    status = normalize_status(
        raw_status
    )

    # --------------------------------------------------------
    # Reason
    # --------------------------------------------------------

    reason = str(
        result.get(
            "llm_reason",
            ""
        )
    )

    return AgentEvaluation(

        case_id=str(
            result.get(
                "id",
                ""
            )
        ),

        answer_correct=answer_correct,

        evidence_sufficient=evidence_sufficient,

        citation_valid=citation_valid,

        retrieval_count=retrieval_count,

        repair_count=repair_count,

        status=status,

        reason=reason,
    )


# ============================================================
# 5. Dataset Summary
# ============================================================

def summarize(                           ###汇总实例计算各种率
    evaluations: list[AgentEvaluation]
) -> dict:

    total = len(evaluations)

    if total == 0:

        return {
            "total": 0,
            "answer_correctness": 0.0,
            "evidence_sufficiency": 0.0,
            "citation_validity": 0.0,
            "safe_fallback_rate": 0.0,
            "retrieval_failure_rate": 0.0,
            "average_retrieval_count": 0.0,
            "repair_rate": 0.0,
            "average_repair_count": 0.0,
        }

    # ========================================================
    # Answer Correctness
    # ========================================================

    answer_correct = sum(
        evaluation.answer_correct
        for evaluation in evaluations
    )

    # ========================================================
    # Evidence Sufficiency
    # ========================================================

    evidence_sufficient = sum(
        evaluation.evidence_sufficient
        for evaluation in evaluations
    )

    # ========================================================
    # Citation Validity
    # ========================================================

    citation_valid = sum(
        evaluation.citation_valid
        for evaluation in evaluations
    )

    # ========================================================
    # Safe Fallback
    # ========================================================

    safe_fallback = sum(
        evaluation.status == "safe_fallback"
        for evaluation in evaluations
    )

    # ========================================================
    # Retrieval Failure
    # ========================================================

    retrieval_failure = sum(
        evaluation.status == "retrieval_failure"
        for evaluation in evaluations
    )

    # ========================================================
    # Retrieval Count
    # ========================================================

    retrieval_total = sum(
        evaluation.retrieval_count
        for evaluation in evaluations
    )

    # ========================================================
    # Repair Cases
    #
    # repair_count > 0
    # 表示这个 Case 至少发生过一次 Answer Repair。
    # ========================================================

    repair_cases = sum(
        evaluation.repair_count > 0
        for evaluation in evaluations
    )

    # ========================================================
    # Repair Count
    # ========================================================

    repair_total = sum(
        evaluation.repair_count
        for evaluation in evaluations
    )

    # ========================================================
    # 最终结果
    # ========================================================

    return {

        "total": total,

        "answer_correctness":
            answer_correct / total,

        "evidence_sufficiency":
            evidence_sufficient / total,

        "citation_validity":
            citation_valid / total,

        "safe_fallback_rate":
            safe_fallback / total,

        "retrieval_failure_rate":
            retrieval_failure / total,

        "average_retrieval_count":
            retrieval_total / total,

        "repair_rate":
            repair_cases / total,

        "average_repair_count":
            repair_total / total,
    }


# ============================================================
# 6. 打印 Summary
# ============================================================

def print_summary(                  ###输出各种率
    summary: dict
):

    print()
    print("=" * 60)
    print("Agentic RAG Evaluation Report")
    print("=" * 60)

    print()

    print(
        f"Total Cases: "
        f"{summary['total']}"
    )

    print()

    print(
        f"Answer Correctness: "
        f"{summary['answer_correctness']:.2%}"
    )

    print(
        f"Evidence Sufficiency: "
        f"{summary['evidence_sufficiency']:.2%}"
    )

    print(
        f"Citation Validity: "
        f"{summary['citation_validity']:.2%}"
    )

    print()

    print(
        f"Safe Fallback Rate: "
        f"{summary['safe_fallback_rate']:.2%}"
    )

    print(
        f"Retrieval Failure Rate: "
        f"{summary['retrieval_failure_rate']:.2%}"
    )

    print()

    print(
        f"Average Retrieval Count: "
        f"{summary['average_retrieval_count']:.2f}"
    )

    print(
        f"Repair Rate: "
        f"{summary['repair_rate']:.2%}"
    )

    print(
        f"Average Repair Count: "
        f"{summary['average_repair_count']:.2f}"
    )

    print()
    print("=" * 60)


# ============================================================
# 7. 打印每一个 Case
# ============================================================

def print_case_details(                          ##输出契约实例里面的关注细节
    evaluations: list[AgentEvaluation]
):

    print()
    print("=" * 60)
    print("Case Details")
    print("=" * 60)

    for evaluation in evaluations:

        print()

        print(
            f"Case: {evaluation.case_id}"
        )

        print(
            f"Answer Correct: "
            f"{evaluation.answer_correct}"
        )

        print(
            f"Evidence Sufficient: "
            f"{evaluation.evidence_sufficient}"
        )

        print(
            f"Citation Valid: "
            f"{evaluation.citation_valid}"
        )

        print(
            f"Retrieval Count: "
            f"{evaluation.retrieval_count}"
        )

        print(
            f"Repair Count: "
            f"{evaluation.repair_count}"
        )

        print(
            f"Status: "
            f"{evaluation.status}"
        )

        if evaluation.reason:

            print(
                f"Reason: "
                f"{evaluation.reason}"
            )


# ============================================================
# 8. 保存最终报告
# ============================================================

def save_report(                     ##把实例做成报告dump转储实例
    evaluations: list[AgentEvaluation],
    summary: dict
):

    report = {

        "evaluation_stage":
            "20.8",

        "evaluation_name":
            "Multi-dimensional Evaluation",

        "summary":
            summary,

        "cases":
            [
                evaluation.model_dump()
                for evaluation in evaluations
            ],
    }

    with open(
        OUTPUT_FILE,         
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(            ##对象数据转化成json存储到file
            report,
            file,
            ensure_ascii=False,
            indent=2
        )


# ============================================================
# 9. Main
# ============================================================

def main():

    # --------------------------------------------------------
    # 检查输入文件
    # --------------------------------------------------------

    if not INPUT_FILE.exists():    #path内置方法

        print(
            f"ERROR: 找不到输入文件："
            f"{INPUT_FILE}"
        )

        print()

        print(
            "请确认 agentic_rag_20_7_llm_judge_comparison.py "
            "已经运行，并且 JSON 文件与本脚本位于同一目录。"
        )

        return

    # --------------------------------------------------------
    # 读取 20.7 结果
    # --------------------------------------------------------

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        data = json.load(file)          ##json内置方法

    # --------------------------------------------------------
    # 兼容两种常见 JSON 结构
    #
    # 结构 A：
    #
    # [
    #   {...},
    #   {...}
    # ]
    #
    # 结构 B：
    #
    # {
    #   "results": [
    #       {...},
    #       {...}
    #   ]
    # }
    # --------------------------------------------------------

    if isinstance(data, list):

        results = data

    elif isinstance(data, dict):

        if isinstance(
            data.get("results"),
            list
        ):

            results = data["results"]

        elif isinstance(
            data.get("cases"),
            list
        ):

            results = data["cases"]

        else:

            print(
                "ERROR: JSON 中没有找到 "
                "'results' 或 'cases'。"
            )

            return

    else:

        print(
            "ERROR: JSON 数据结构无法识别。"
        )

        return

    # --------------------------------------------------------
    # 空数据
    # --------------------------------------------------------

    if not results:

        print(
            "ERROR: 输入 JSON 没有测试结果。"
        )

        return

    # --------------------------------------------------------
    # 构建 AgentEvaluation
    # --------------------------------------------------------

    evaluations = []      ####生成一个个report对象组成的列表

    for result in results:

        if not isinstance(
            result,
            dict
        ):
            continue

        evaluation = evaluate_case(
            result
        )

        evaluations.append(
            evaluation
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = summarize(            ##汇总数据输出各种率
        evaluations
    )

    # --------------------------------------------------------
    # 控制台输出
    # --------------------------------------------------------

    print_summary(
        summary
    )

    print_case_details(               ##输出细节
        evaluations
    )

    # --------------------------------------------------------
    # 保存
    # --------------------------------------------------------

    save_report(
        evaluations,
        summary
    )

    print()

    print(
        f"Evaluation report saved to:"
    )

    print(
        OUTPUT_FILE.resolve()
    )

    print()


# ============================================================
# 10. Entry Point
# ============================================================

if __name__ == "__main__":

    main()