import json              #这一步的作用是利用20.8整理好的变成多维评估指标（各种率）以及cases来自20.7的评估指标（增加评估评估器），判断具体指标不完美问题出在哪里，具体到那个case错了，错误的原因是什么
from pathlib import Path
from collections import Counter                             ##也就是为错误案例添加错误类型这个字段（核心），并顺便计算一下错误率...区别是案例集里面只有错误案例，正确阿里从20.8里面优化掉了



# ============================================================
# 1. 文件配置
# ============================================================

INPUT_FILE = Path(
    "agentic_rag_20_8_evaluation_report.json"
)

OUTPUT_FILE = Path(
    "agentic_rag_20_9_failure_analysis.json"
)


# ============================================================
# 2. Failure Type
# ============================================================

FAILURE_TYPES = [
    "evidence_failure",
    "repair_failure",
    "grounding_failure",
    "unknown_failure",
]


# ============================================================
# 3. 读取 20.8 Evaluation Report
# ============================================================

with INPUT_FILE.open(
    "r",
    encoding="utf-8"
) as f:

    report = json.load(f)


cases = report["cases"]


# ============================================================
# 4. Failure Classification
# ============================================================

def classify_failure(case: dict):
    """
    根据 20.8 中已经存在的评估结果，
    判断最终失败属于哪一种类型。

    当前只使用三个核心判断条件：

    1.
    evidence_sufficient == False
        -> evidence_failure

    2.
    evidence_sufficient == True
    citation_valid == False
        -> repair_failure

    3.
    evidence_sufficient == True
    citation_valid == True
    answer_correct == False
        -> grounding_failure

    其他无法由 20.8 明确判断的情况：
        -> unknown_failure

    注意：
    repair_count 不参与 failure_type 判断。
    """

    answer_correct = case.get(
        "answer_correct"
    )

    evidence_sufficient = case.get(
        "evidence_sufficient"
    )

    citation_valid = case.get(
        "citation_valid"
    )


    # --------------------------------------------------------
    # 如果答案本身是正确的
    #
    # 当前 Failure Analysis 只分析最终答案错误的 Case。
    # --------------------------------------------------------

    if answer_correct is True:
        return None


    # --------------------------------------------------------
    # 情况 1：
    #
    # evidence 不足
    # --------------------------------------------------------

    if evidence_sufficient is False:

        return "evidence_failure"


    # --------------------------------------------------------
    # 情况 2：
    #
    # evidence 足够
    # citation 无效
    # --------------------------------------------------------

    if (
        evidence_sufficient is True
        and citation_valid is False
    ):

        return "repair_failure"


    # --------------------------------------------------------
    # 情况 3：
    #
    # evidence 足够
    # citation 有效
    # answer 错误
    # --------------------------------------------------------

    if (
        evidence_sufficient is True
        and citation_valid is True
        and answer_correct is False
    ):

        return "grounding_failure"


    # --------------------------------------------------------
    # 其他情况
    #
    # 当前 20.8 数据不足以判断
    # --------------------------------------------------------

    return "unknown_failure"


# ============================================================
# 5. 开始 Failure Analysis
# ============================================================

failure_cases = []

failure_counts = {              ##这一步的作用是根据制作一个字典内部各对象是   错误类型：0    这种形式方便后续统计各类型失败次数
    failure_type: 0
    for failure_type in FAILURE_TYPES
}


# ============================================================
# 6. 遍历所有 Case
# ============================================================

for case in cases:

    failure_type = classify_failure(case)


    # --------------------------------------------------------
    # 正确答案不属于 Failure Analysis
    # --------------------------------------------------------

    if failure_type is None:
        continue


    failure_counts[failure_type] += 1    #改类型次数加1


    failure_cases.append(
        {
            "case_id": case.get("case_id"),
          
            "failure_type": failure_type,

            "answer_correct":
                case.get("answer_correct"),

            "evidence_sufficient":
                case.get("evidence_sufficient"),

            "citation_valid":
                case.get("citation_valid"),

            "retrieval_count":
                case.get("retrieval_count"),

            "repair_count":
                case.get("repair_count"),

            "status":
                case.get("status"),

        

           
            "reason":
                case.get("reason"),
        }
    )


# ============================================================
# 7. 基本统计
# ============================================================

total_cases = len(cases)

failed_cases = len(failure_cases)


if failed_cases > 0:

    failure_rates = {
        failure_type:
            count / failed_cases

        for failure_type, count
        in failure_counts.items()          ##把字典里面的键值对转化成一个个元组放入列表方便for in与encurme...执行不同他是编号此处的编号变成了键值
    }

else:

    failure_rates = {
        failure_type: 0.0
        for failure_type in FAILURE_TYPES
    }


# ============================================================
# 8. 输出报告
# ============================================================

result = {

    "total_cases":
        total_cases,

    "failed_cases":
        failed_cases,

    "failure_counts":
        failure_counts,

    "failure_rates":
        failure_rates,

    "failure_cases":
        failure_cases,
}


# ============================================================
# 9. 保存 JSON
# ============================================================

with OUTPUT_FILE.open(
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        result,
        f,
        ensure_ascii=False,
        indent=4
    )


# ============================================================
# 10. 控制台输出
# ============================================================

print("=" * 60)

print(
    "Failure Analysis"
)

print("=" * 60)


print(
    f"Total Cases: "
    f"{total_cases}"
)


print(
    f"Failed Cases: "
    f"{failed_cases}"
)


print()


print(
    "Failure Breakdown:"
)


for failure_type in FAILURE_TYPES:

    count = failure_counts[
        failure_type
    ]

    rate = failure_rates[
        failure_type
    ]

    print(
        f"  {failure_type}: "
        f"{count} "
        f"({rate:.2%})"
    )


print()


print(
    "Failure Cases:"
)


if not failure_cases:

    print(
        "  No failure cases."
    )

else:

    for case in failure_cases:

        print(
            f"\n[{case['case_id']}]"
        )

    

        print(
            f"Failure Type: "
            f"{case['failure_type']}"
        )

        print(
            f"Answer Correct: "
            f"{case['answer_correct']}"
        )

        print(
            f"Evidence Sufficient: "
            f"{case['evidence_sufficient']}"
        )

        print(
            f"Citation Valid: "
            f"{case['citation_valid']}"
        )

        print(
            f"Retrieval Count: "
            f"{case['retrieval_count']}"
        )

        print(
            f"Repair Count: "
            f"{case['repair_count']}"
        )


print()


print(
    f"Saved to: "
    f"{OUTPUT_FILE}"
)