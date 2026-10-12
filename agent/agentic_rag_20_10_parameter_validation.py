import json                              #####################20.10存在的目的是完成工作流校验未完成的蚕食值合法性校验，这个校验应该放在20.8紧接着之后20.91之前的，这算是后补的。本质上也和轨迹校验一样找茬的
from pathlib import Path
from typing import Any                   ##表示任意类型


# ============================================================
# 1. 文件配置
# ============================================================

INPUT_FILE = Path(
    "agentic_rag_20_8_evaluation_report.json"
)

OUTPUT_FILE = Path(
    "agentic_rag_20_10_parameter_validation.json"
)


# ============================================================
# 2. Runtime Contract 参数
#
# 注意：
# 这里不是重新做轨迹校验。
#
# 只是把 19 Runtime 已经确定的边界，
# 用来判断 20.8 中的参数值是否合法。
#
# 当前 19 架构：
#
# MAX_RETRIEVALS = 3
# MAX_ANSWER_REPAIRS = 2
# ============================================================

MAX_RETRIEVALS = 3

MAX_ANSWER_REPAIRS = 2


# ============================================================
# 3. 合法状态空间
# ============================================================

VALID_STATUSES = {
    "success",
    "retrieval_failure",
    "safe_fallback",
}


# ============================================================
# 4. 必需字段
# ============================================================

REQUIRED_CASE_FIELDS = {
    "case_id",
    "answer_correct",
    "evidence_sufficient",
    "citation_valid",
    "retrieval_count",
    "repair_count",
    "status",
    "reason",
}


# ============================================================
# 5. 工具函数
# ============================================================

def add_violation(           ##添加校验错误案例的具体信息
    violations: list[dict[str, Any]],    #字典里面的键是str，值是任意类型
    case_id: str | None,
    field: str,
    rule: str,
    actual: Any,
    expected: str,
):
    """
    添加一条参数值违规记录。
    """

    violations.append(
        {
            "case_id": case_id,
            "field": field,
            "rule": rule,
            "actual": actual,
            "expected": expected,
        }
    )


# ============================================================
# 6. 校验 summary 中的比例参数               
# ============================================================

def validate_rate(                ##
    summary: dict[str, Any],
    field: str,
    violations: list[dict[str, Any]],
):
    """
    检查比例是否处于 [0, 1]。
    """

    if field not in summary:              ##验证字段是否存在于summary，字段名校验

        add_violation(
            violations=violations,
            case_id=None,
            field=field,
            rule="summary_field_exists",
            actual=None,
            expected="字段必须存在",
        )

        return

    value = summary[field]

    # bool 是 int 的子类，所以必须排除 bool
    if (                                            ##校验summary 的字段值的类型是不是合法，即都为整数或者浮点数
        not isinstance(value, (int, float))
        or isinstance(value, bool)
    ):

        add_violation(
            violations=violations,
            case_id=None,
            field=field,
            rule="summary_rate_type",
            actual=(
                f"{value!r}, "            
                f"type={type(value).__name__}"
            ),
            expected="int 或 float",
        )

        return

    if not 0 <= value <= 1:

        add_violation(
            violations=violations,
            case_id=None,
            field=field,
            rule="summary_rate_range",
            actual=value,
            expected="0 <= value <= 1",
        )


# ============================================================
# 7. 校验单个 Case                    本质上还是找茬
# ============================================================

def validate_case(
    case: dict[str, Any]
) -> list[dict[str, Any]]:

    violations = []

    case_id = case.get(
        "case_id"
    )

    # ========================================================
    # Rule 1
    #
    # Case 必须包含所有字段
    # ========================================================

    missing_fields = (                   ##集合的减法运算
        REQUIRED_CASE_FIELDS
        - set(case.keys())
    )

    if missing_fields:

        for field in sorted(missing_fields):

            add_violation(
                violations=violations,
                case_id=case_id,
                field=field,
                rule="required_field",
                actual=None,
                expected="字段必须存在",
            )

        # 如果字段都缺失，继续往下检查没有意义
        return violations

    # ========================================================
    # Rule 2
    #
    # answer_correct 必须是 bool
    # ========================================================

    answer_correct = case[
        "answer_correct"
    ]

    if not isinstance(
        answer_correct,
        bool
    ):

        add_violation(
            violations=violations,
            case_id=case_id,
            field="answer_correct",
            rule="boolean_type",
            actual=(
                f"{answer_correct!r}, "
                f"type={type(answer_correct).__name__}"
            ),
            expected="True 或 False",
        )

    # ========================================================
    # Rule 3
    #
    # evidence_sufficient 必须是 bool
    # ========================================================

    evidence_sufficient = case[
        "evidence_sufficient"
    ]

    if not isinstance(
        evidence_sufficient,
        bool
    ):

        add_violation(
            violations=violations,
            case_id=case_id,
            field="evidence_sufficient",
            rule="boolean_type",
            actual=(
                f"{evidence_sufficient!r}, "
                f"type={type(evidence_sufficient).__name__}"
            ),
            expected="True 或 False",
        )

    # ========================================================
    # Rule 4
    #
    # citation_valid：
    #
    # True
    # False
    # None
    #
    # 都属于合法值。
    #
    # None 的意义：
    # Citation Validation 没有执行 / 最终状态不适用。
    # ========================================================

    citation_valid = case[
        "citation_valid"
    ]

    if (
        citation_valid is not True
        and citation_valid is not False
        and citation_valid is not None
    ):

        add_violation(
            violations=violations,
            case_id=case_id,
            field="citation_valid",
            rule="citation_valid_value",
            actual=(
                f"{citation_valid!r}, "
                f"type={type(citation_valid).__name__}"
            ),
            expected="True / False / None",
        )

    # ========================================================
    # Rule 5                                    ###先校验值的类型是不是合法在校验值的取值范围是不是合法
    #
    # retrieval_count
    #
    # 合法范围：
    #
    # 1 <= retrieval_count <= MAX_RETRIEVALS
    # ========================================================

    retrieval_count = case[
        "retrieval_count"
    ]

    if (
        not isinstance(
            retrieval_count,
            int
        )
        or isinstance(
            retrieval_count,
            bool
        )
    ):

        add_violation(
            violations=violations,
            case_id=case_id,
            field="retrieval_count",
            rule="integer_type",
            actual=(
                f"{retrieval_count!r}, "
                f"type={type(retrieval_count).__name__}"
            ),
            expected="整数",
        )

    else:

        if retrieval_count < 1:

            add_violation(
                violations=violations,
                case_id=case_id,
                field="retrieval_count",
                rule="retrieval_count_lower_bound",
                actual=retrieval_count,
                expected=">= 1",
            )

        if retrieval_count > MAX_RETRIEVALS:

            add_violation(
                violations=violations,
                case_id=case_id,
                field="retrieval_count",
                rule="retrieval_count_upper_bound",
                actual=retrieval_count,
                expected=(
                    f"<= {MAX_RETRIEVALS}"
                ),
            )

    # ========================================================
    # Rule 6
    #
    # repair_count
    #
    # 合法范围：
    #
    # 0 <= repair_count <= MAX_ANSWER_REPAIRS
    # ========================================================

    repair_count = case[
        "repair_count"
    ]

    if (
        not isinstance(
            repair_count,
            int
        )
        or isinstance(
            repair_count,
            bool
        )
    ):

        add_violation(
            violations=violations,
            case_id=case_id,
            field="repair_count",
            rule="integer_type",
            actual=(
                f"{repair_count!r}, "
                f"type={type(repair_count).__name__}"
            ),
            expected="整数",
        )

    else:

        if repair_count < 0:

            add_violation(
                violations=violations,
                case_id=case_id,
                field="repair_count",
                rule="repair_count_lower_bound",
                actual=repair_count,
                expected=">= 0",
            )

        if repair_count > MAX_ANSWER_REPAIRS:

            add_violation(
                violations=violations,
                case_id=case_id,
                field="repair_count",
                rule="repair_count_upper_bound",
                actual=repair_count,
                expected=(
                    f"<= {MAX_ANSWER_REPAIRS}"
                ),
            )

    # ========================================================
    # Rule 7
    #
    # status 必须属于合法状态空间
    # ========================================================

    status = case[
        "status"
    ]

    if status not in VALID_STATUSES:

        add_violation(
            violations=violations,
            case_id=case_id,
            field="status",
            rule="status_value",
            actual=status,
            expected=(
                "success / "
                "retrieval_failure / "
                "safe_fallback"
            ),
        )

    # ========================================================
    # Rule 8
    #
    # reason 必须是字符串
    # ========================================================

    reason = case[
        "reason"
    ]

    if not isinstance(
        reason,
        str
    ):

        add_violation(
            violations=violations,
            case_id=case_id,
            field="reason",
            rule="string_type",
            actual=(
                f"{reason!r}, "
                f"type={type(reason).__name__}"
            ),
            expected="字符串",
        )

    # ========================================================
    # ========================================================
    #
    # 以下开始检查“参数之间的组合是否合法”
    #
    # 注意：
    # 这不是重新检查 Trace。
    #
    # 而是检查：
    #
    # 20.8 这一条记录有没有出现
    # 自相矛盾的参数组合。
    #
    # ========================================================
    # ========================================================

    # ========================================================
    # Rule 9
    #
    # evidence_sufficient = False
    #
    # 根据 19 Runtime：
    #
    # Retrieval Budget 耗尽
    #        ↓
    # retrieval_failure
    #
    # 因此：
    #
    # retrieval_count = MAX_RETRIEVALS
    # repair_count = 0
    # citation_valid = None
    # status = retrieval_failure
    #
    # ========================================================

    if evidence_sufficient is False:

        if (
            isinstance(
                retrieval_count,
                int
            )
            and retrieval_count
            != MAX_RETRIEVALS
        ):

            add_violation(
                violations=violations,
                case_id=case_id,
                field="retrieval_count",
                rule=(
                    "insufficient_evidence_requires_max_retrieval"
                ),
                actual=retrieval_count,
                expected=(
                    f"{MAX_RETRIEVALS}"
                ),
            )

        if (
            isinstance(
                repair_count,
                int
            )
            and repair_count != 0
        ):

            add_violation(
                violations=violations,
                case_id=case_id,
                field="repair_count",
                rule=(
                    "insufficient_evidence_cannot_repair"
                ),
                actual=repair_count,
                expected="0",
            )

        if citation_valid is not None:

            add_violation(
                violations=violations,
                case_id=case_id,
                field="citation_valid",
                rule=(
                    "insufficient_evidence_cannot_validate_citation"
                ),
                actual=citation_valid,
                expected="None",
            )

        if status != "retrieval_failure":

            add_violation(
                violations=violations,
                case_id=case_id,
                field="status",
                rule=(
                    "insufficient_evidence_requires_retrieval_failure"
                ),
                actual=status,
                expected="retrieval_failure",
            )

    # ========================================================
    # Rule 10
    #
    # evidence_sufficient = True
    #
    # 不可能最终成为 retrieval_failure。
    # ========================================================

    if evidence_sufficient is True:

        if status == "retrieval_failure":

            add_violation(
                violations=violations,
                case_id=case_id,
                field="status",
                rule=(
                    "sufficient_evidence_cannot_be_retrieval_failure"
                ),
                actual=status,
                expected=(
                    "success 或 safe_fallback"
                ),
            )

    # ========================================================
    # Rule 11
    #
    # status = success
    #
    # 根据 Runtime：
    #
    # Evidence 足够
    #        ↓
    # Grounding
    #        ↓
    # Citation Validation PASS
    #        ↓
    # success
    #
    # 因此：
    #
    # evidence_sufficient = True
    # citation_valid = True
    #
    # ========================================================

    if status == "success":

        if evidence_sufficient is not True:

            add_violation(
                violations=violations,
                case_id=case_id,
                field="evidence_sufficient",
                rule=(
                    "success_requires_sufficient_evidence"
                ),
                actual=evidence_sufficient,
                expected="True",
            )

        if citation_valid is not True:

            add_violation(
                violations=violations,
                case_id=case_id,
                field="citation_valid",
                rule=(
                    "success_requires_valid_citation"
                ),
                actual=citation_valid,
                expected="True",
            )

    # ========================================================
    # Rule 12
    #
    # status = retrieval_failure
    #
    # 必须是 Evidence 不足的终态。
    # ========================================================

    if status == "retrieval_failure":

        if evidence_sufficient is not False:

            add_violation(
                violations=violations,
                case_id=case_id,
                field="evidence_sufficient",
                rule=(
                    "retrieval_failure_requires_insufficient_evidence"
                ),
                actual=evidence_sufficient,
                expected="False",
            )

        if citation_valid is not None:

            add_violation(
                violations=violations,
                case_id=case_id,
                field="citation_valid",
                rule=(
                    "retrieval_failure_has_no_citation_validation"
                ),
                actual=citation_valid,
                expected="None",
            )

        if repair_count != 0:

            add_violation(
                violations=violations,
                case_id=case_id,
                field="repair_count",
                rule=(
                    "retrieval_failure_has_no_repair"
                ),
                actual=repair_count,
                expected="0",
            )

        if retrieval_count != MAX_RETRIEVALS:

            add_violation(
                violations=violations,
                case_id=case_id,
                field="retrieval_count",
                rule=(
                    "retrieval_failure_requires_max_retrieval"
                ),
                actual=retrieval_count,
                expected=(
                    str(MAX_RETRIEVALS)
                ),
            )

    # ========================================================
    # Rule 13
    #
    # status = safe_fallback
    #
    # 根据 19 Runtime：
    #
    # Evidence 足够
    # ↓
    # Citation Validation FAIL
    # ↓
    # Repair
    # ↓
    # 达到 MAX_ANSWER_REPAIRS
    # ↓
    # Safe Fallback
    #
    # 当前 Runtime 最终返回：
    #
    # citation_valid = None
    #
    # ========================================================

    if status == "safe_fallback":

        if evidence_sufficient is not True:

            add_violation(
                violations=violations,
                case_id=case_id,
                field="evidence_sufficient",
                rule=(
                    "safe_fallback_requires_sufficient_evidence"
                ),
                actual=evidence_sufficient,
                expected="True",
            )

        if repair_count != MAX_ANSWER_REPAIRS:

            add_violation(
                violations=violations,
                case_id=case_id,
                field="repair_count",
                rule=(
                    "safe_fallback_requires_max_repair_count"
                ),
                actual=repair_count,
                expected=(
                    str(MAX_ANSWER_REPAIRS)
                ),
            )

        if citation_valid is not None:

            add_violation(
                violations=violations,
                case_id=case_id,
                field="citation_valid",
                rule=(
                    "safe_fallback_final_citation_is_none"
                ),
                actual=citation_valid,
                expected="None",
            )

    # ========================================================
    # Rule 14
    #
    # repair_count > 0
    #
    # 说明发生过 Answer Repair。
    #
    # Answer Repair 只能发生在：
    #
    # evidence_sufficient = True
    #
    # ========================================================

    if (
        isinstance(
            repair_count,
            int
        )
        and repair_count > 0
    ):

        if evidence_sufficient is not True:

            add_violation(
                violations=violations,
                case_id=case_id,
                field="evidence_sufficient",
                rule=(
                    "repair_requires_sufficient_evidence"
                ),
                actual=evidence_sufficient,
                expected="True",
            )

    # ========================================================
    # Rule 15
    #
    # 非常重要：
    #
    # answer_correct = False
    #
    # 本身绝对不是 Runtime Contract Violation。
    #
    # 因为：
    #
    # Agent 可以正常运行，
    # Evidence 足够，
    # Citation 正确，
    # 但最终答案仍然错误。
    #
    # 这正是 20.9 要识别的：
    #
    # grounding_failure
    #
    # 所以这里故意不检查 answer_correct
    # 与其他 Runtime 字段的“正确性”关系。
    # ========================================================

    return violations


# ============================================================
# 8. 校验整个 20.8 Report
# ============================================================

def validate_report(
    report: dict[str, Any]
) -> dict[str, Any]:

    violations = []

    case_results = []

    # ========================================================
    # Top-level
    # ========================================================

    if not isinstance(
        report,
        dict
    ):

        return {
            "parameter_valid": False,
            "total_cases": 0,
            "valid_cases": 0,
            "invalid_cases": 0,
            "violations": [
                {
                    "case_id": None,
                    "field": "root",
                    "rule": "object_type",
                    "actual": type(report).__name__,
                    "expected": "object",
                }
            ],
        }

    # ========================================================
    # evaluation_stage
    # ========================================================

    if report.get(
        "evaluation_stage"
    ) != "20.8":

        add_violation(
            violations=violations,
            case_id=None,
            field="evaluation_stage",
            rule="evaluation_stage_value",
            actual=report.get(
                "evaluation_stage"
            ),
            expected="20.8",
        )

    # ========================================================
    # evaluation_name
    # ========================================================

    if report.get(
        "evaluation_name"
    ) != "Multi-dimensional Evaluation":

        add_violation(
            violations=violations,
            case_id=None,
            field="evaluation_name",
            rule="evaluation_name_value",
            actual=report.get(
                "evaluation_name"
            ),
            expected=(
                "Multi-dimensional Evaluation"
            ),
        )

    # ========================================================
    # summary
    # ========================================================

    summary = report.get(
        "summary"
    )

    if not isinstance(
        summary,
        dict
    ):

        add_violation(
            violations=violations,
            case_id=None,
            field="summary",
            rule="summary_type",
            actual=(
                type(summary).__name__
            ),
            expected="object",
        )

        summary = {}

    # ========================================================
    # cases
    # ========================================================

    cases = report.get(
        "cases"
    )

    if not isinstance(
        cases,
        list
    ):

        add_violation(
            violations=violations,
            case_id=None,
            field="cases",
            rule="cases_type",
            actual=(
                type(cases).__name__
            ),
            expected="list",
        )

        cases = []

    # ========================================================
    # summary.total
    # ========================================================

    total = summary.get(
        "total"
    )

    if (
        not isinstance(
            total,
            int
        )
        or isinstance(
            total,
            bool
        )
    ):

        add_violation(
            violations=violations,
            case_id=None,
            field="summary.total",
            rule="integer_type",
            actual=(
                f"{total!r}, "
                f"type={type(total).__name__}"
            ),
            expected="非负整数",
        )

    else:

        if total < 0:

            add_violation(
                violations=violations,
                case_id=None,
                field="summary.total",
                rule="non_negative",
                actual=total,
                expected=">= 0",
            )

        if total != len(cases):

            add_violation(
                violations=violations,
                case_id=None,
                field="summary.total",
                rule="total_matches_cases",
                actual=total,
                expected=(
                    f"{len(cases)}"
                ),
            )

    # ========================================================
    # summary rates
    # ========================================================

    rate_fields = [
        "answer_correctness",
        "evidence_sufficiency",
        "citation_validity",
        "safe_fallback_rate",
        "retrieval_failure_rate",
        "repair_rate",
    ]

    for field in rate_fields:

        validate_rate(
            summary=summary,
            field=field,
            violations=violations,
        )

    # ========================================================
    # summary average_retrieval_count
    # ========================================================

    average_retrieval_count = summary.get(
        "average_retrieval_count"
    )

    if (
        not isinstance(
            average_retrieval_count,
            (int, float)
        )
        or isinstance(
            average_retrieval_count,
            bool
        )
    ):

        add_violation(
            violations=violations,
            case_id=None,
            field="summary.average_retrieval_count",
            rule="numeric_type",
            actual=(
                f"{average_retrieval_count!r}, "
                f"type={type(average_retrieval_count).__name__}"
            ),
            expected="int 或 float",
        )

    else:

        if not (
            0
            <= average_retrieval_count
            <= MAX_RETRIEVALS
        ):

            add_violation(
                violations=violations,
                case_id=None,
                field="summary.average_retrieval_count",
                rule="average_retrieval_range",
                actual=average_retrieval_count,
                expected=(
                    f"0 <= value <= "
                    f"{MAX_RETRIEVALS}"
                ),
            )

    # ========================================================
    # summary average_repair_count
    # ========================================================

    average_repair_count = summary.get(
        "average_repair_count"
    )

    if (
        not isinstance(
            average_repair_count,
            (int, float)
        )
        or isinstance(
            average_repair_count,
            bool
        )
    ):

        add_violation(
            violations=violations,
            case_id=None,
            field="summary.average_repair_count",
            rule="numeric_type",
            actual=(
                f"{average_repair_count!r}, "
                f"type={type(average_repair_count).__name__}"
            ),
            expected="int 或 float",
        )

    else:

        if not (
            0
            <= average_repair_count
            <= MAX_ANSWER_REPAIRS
        ):

            add_violation(
                violations=violations,
                case_id=None,
                field="summary.average_repair_count",
                rule="average_repair_range",
                actual=average_repair_count,
                expected=(
                    f"0 <= value <= "
                    f"{MAX_ANSWER_REPAIRS}"
                ),
            )

    # ========================================================
    # Case validation
    # ========================================================

    for case in cases:

        if not isinstance(
            case,
            dict
        ):

            violation = {
                "case_id": None,
                "field": "case",
                "rule": "case_type",
                "actual": (
                    type(case).__name__
                ),
                "expected": "object",
            }

            violations.append(
                violation
            )

            case_results.append(
                {
                    "case_id": None,
                    "valid": False,
                    "violations": [
                        violation
                    ],
                }
            )

            continue

        case_id = case.get(
            "case_id"
        )

        case_violations = validate_case(
            case
        )

        case_valid = (
            len(case_violations) == 0
        )

        case_results.append(
            {
                "case_id": case_id,
                "valid": case_valid,
                "violations": case_violations,
            }
        )

        violations.extend(
            case_violations
        )

    # ========================================================
    # 统计
    # ========================================================

    valid_case_count = sum(
        1
        for item in case_results
        if item["valid"]
    )

    invalid_case_count = (
        len(case_results)
        - valid_case_count
    )

    # ========================================================
    # 最终结果
    # ========================================================

    return {
        "parameter_valid": (
            len(violations) == 0
        ),
        "total_cases": len(cases),
        "valid_cases": valid_case_count,
        "invalid_cases": invalid_case_count,
        "runtime_limits": {
            "MAX_RETRIEVALS":
                MAX_RETRIEVALS,
            "MAX_ANSWER_REPAIRS":
                MAX_ANSWER_REPAIRS,
        },
        "case_results": case_results,
        "violations": violations,
    }


# ============================================================
# 9. 打印结果
# ============================================================

def print_validation(
    result: dict[str, Any]
):

    print("\n")

    print(
        "=" * 70
    )

    print(
        "AGENTIC RAG 20.10"
    )

    print(
        "PARAMETER VALUE VALIDATION"
    )

    print(
        "=" * 70
    )

    print()

    print(
        "Total Cases:",
        result["total_cases"]
    )

    print(
        "Valid Cases:",
        result["valid_cases"]
    )

    print(
        "Invalid Cases:",
        result["invalid_cases"]
    )

    print()

    if result[
        "parameter_valid"
    ]:

        print(
            "✅ PARAMETER VALIDATION: PASS"
        )

        print()

        print(
            "20.8 Evaluation Report 中的参数值"
        )

        print(
            "及其组合均符合当前 Evaluation / Runtime Contract。"
        )

    else:

        print(
            "❌ PARAMETER VALIDATION: FAIL"
        )

        print()

        print(
            "发现以下问题："
        )

        print()

        for violation in result[
            "violations"
        ]:

            print(
                f"Case: "
                f"{violation['case_id']}"
            )

            print(
                f"Field: "
                f"{violation['field']}"
            )

            print(
                f"Rule: "
                f"{violation['rule']}"
            )

            print(
                f"Actual: "
                f"{violation['actual']}"
            )

            print(
                f"Expected: "
                f"{violation['expected']}"
            )

            print(
                "-" * 70
            )


# ============================================================
# 10. Main
# ============================================================

def main():

    print(
        "=" * 70
    )

    print(
        "Agentic RAG 20.10"
    )

    print(
        "Parameter Value Validation"
    )

    print(
        "=" * 70
    )

    print()

    # --------------------------------------------------------
    # 读取文件
    # --------------------------------------------------------

    print(
        "Input:"
    )

    print(
        INPUT_FILE
    )

    if not INPUT_FILE.exists():

        print()

        print(
            "❌ 输入文件不存在。"
        )

        return

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        report = json.load(f)

    print()

    print(
        "✓ 20.8 Evaluation Report 读取成功"
    )

    print()

    # --------------------------------------------------------
    # 参数校验
    # --------------------------------------------------------

    result = validate_report(
        report
    )

    # --------------------------------------------------------
    # 打印
    # --------------------------------------------------------

    print_validation(
        result
    )

    # --------------------------------------------------------
    # 保存
    # --------------------------------------------------------

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            result,
            f,
            ensure_ascii=False,
            indent=4
        )

    print()

    print(
        "Output:"
    )

    print(
        OUTPUT_FILE
    )

    print()

    print(
        "✓ 20.10 校验结果已保存。"
    )


# ============================================================
# 11. 程序入口
# ============================================================

if __name__ == "__main__":

    main()