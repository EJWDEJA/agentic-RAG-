import json                       ###这个测试文件的核心目的是，综合20.8json里面的所有案例和20.9里面的失败案例，分析规律，并输出一个规律契约文档，方便后续提出假设                            
from pathlib import Path
from collections import Counter, defaultdict


# ============================================================
# 1. 文件配置
# ============================================================

INPUT_20_8 = Path(
    "agentic_rag_20_8_evaluation_report.json"
)

INPUT_20_9 = Path(
    "agentic_rag_20_9_failure_analysis.json"
)

OUTPUT_FILE = Path(
    "agentic_rag_20_11_failure_analysis.json"
)


# ============================================================
# 2. Pattern Analysis 参数
# ============================================================

# 一个 Failure Type 至少有多少样本，
# 才允许把统计结果视为候选 Pattern。
MIN_PATTERN_SAMPLES = 5

# 某个行为值在某个 Failure Type 中
# 至少占多少比例，才认为存在明显集中现象。
PATTERN_THRESHOLD = 0.70


# ============================================================
# 3. Failure Type
# ============================================================

FAILURE_TYPES = {
    "evidence_failure",
    "repair_failure",
    "grounding_failure",
    "unknown_failure",
}


# ============================================================
# 4. 读取 JSON
# ============================================================

def load_json(path: Path):
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(file)


# ============================================================
# 5. 构造 Failure Type 映射
# ============================================================

def build_failure_map(failure_report):
    """
    将 20.9 的 failure_cases 转换成：

    {
        "case_02": "grounding_failure",
        "case_03": "grounding_failure"
    }

    key:
        case_id

    value:
        failure_type
    """

    failure_map = {}

    for case in failure_report["failure_cases"]:

        case_id = case["case_id"]

        failure_type = case["failure_type"]

        if failure_type not in FAILURE_TYPES:
            raise ValueError(
                f"Unknown failure_type: {failure_type}"
            )

        if case_id in failure_map:
            raise ValueError(
                f"Duplicate case_id in 20.9: {case_id}"
            )

        failure_map[case_id] = failure_type

    return failure_map


# ============================================================
# 6. 构造 20.11 Analysis Records                           把20.9融合到20.8
# ============================================================

def build_analysis_records(
    evaluation_report,
    failure_map,
):
    """
    将 20.8 cases 和 20.9 failure_cases
    根据 case_id 合并。

    注意：

    所有 20.8 Case 都会保留。

    成功 Case：
        failure_type = None

    失败 Case：
        failure_type = 20.9 提供的 failure_type
    """

    records = []

    for case in evaluation_report["cases"]:

        case_id = case["case_id"]

        failure_type = failure_map.get(
            case_id
        )

        record = {
            "case_id": case_id,
            "failure_type": failure_type,
            "answer_correct": case[
                "answer_correct"
            ],
            "evidence_sufficient": case[
                "evidence_sufficient"
            ],
            "citation_valid": case[
                "citation_valid"
            ],
            "retrieval_count": case[
                "retrieval_count"
            ],
            "repair_count": case[
                "repair_count"
            ],
            "status": case[
                "status"
            ],
        }

        records.append(record)

    return records


# ============================================================
# 7. 检查 20.9 是否存在孤立 Case                 
# ============================================================

def validate_failure_case_ids(                     #防止出现案例集中不包含的错误案例
    evaluation_report,
    failure_map,
):
    """
    检查：

    20.9 中的 case_id
    是否都存在于
    20.8 cases。

    防止：

    20.9 有 case_999
    但 20.8 根本不存在。
    """

    evaluation_case_ids = {
        case["case_id"]
        for case in evaluation_report["cases"]
    }

    unknown_case_ids = (
        set(failure_map.keys())
        - evaluation_case_ids
    )

    if unknown_case_ids:

        raise ValueError(
            "20.9 contains case_id not found "
            f"in 20.8: {sorted(unknown_case_ids)}"
        )


# ============================================================
# 8. Summary
# ============================================================

def build_summary(records):                   ##计算总体错误率

    total_cases = len(records)

    failed_cases = sum(
        record["failure_type"] is not None
        for record in records
    )

    failure_rate = (
        failed_cases / total_cases
        if total_cases > 0
        else 0.0
    )

    return {
        "total_cases": total_cases,
        "failed_cases": failed_cases,
        "failure_rate": failure_rate,
    }


# ============================================================
# 9. Failure Distribution
# ============================================================

def build_failure_distribution(records):            ###每种错误在综合错误中的占比

    failure_counts = Counter(                 ##一个出现粗疏统计器，数据变成一个字典的子集，键是错误类型，值是该类型出现次数
        record["failure_type"]
        for record in records
        if record["failure_type"] is not None
    )

    failed_cases = sum(
        failure_counts.values()
    )

    distribution = {}

    for failure_type in sorted(          ##把这个类字典按照key排序组成一个列表
        failure_counts
    ):

        count = failure_counts[
            failure_type
        ]

        rate = (
            count / failed_cases
            if failed_cases > 0
            else 0.0
        )

        distribution[
            failure_type
        ] = {
            "count": count,
            "rate": rate,
        }

    return distribution


# ============================================================
# 10. Failure-centric Analysis
# ============================================================

def build_by_failure_type(records):                         ##以错误视角分析运行时三种行为

    """
    从：

        Failure Type
            ↓
        Runtime Behavior

    的方向进行分析。
    """

    grouped = defaultdict(list)           ##是一个特殊字典当请求key不存在的时候他会自动创建一个该·键值：[],的一个新键值对retrieval_count，repair_counter，status_counter

    for record in records:

        failure_type = record[
            "failure_type"
        ]

        if failure_type is None:
            continue

        grouped[
            failure_type
        ].append(record)

    result = {}

    for failure_type in sorted(grouped):

        failure_records = grouped[
            failure_type
        ]

        retrieval_counter = Counter(
            record["retrieval_count"]
            for record in failure_records
        )

        repair_counter = Counter(
            record["repair_count"]
            for record in failure_records
        )

        status_counter = Counter(
            record["status"]
            for record in failure_records
        )

        result[
            failure_type
        ] = {
            "sample_size": len(
                failure_records
            ),
            "retrieval_count": {
                str(key): value
                for key, value
                in sorted(
                    retrieval_counter.items()
                )
            },
            "repair_count": {
                str(key): value
                for key, value
                in sorted(
                    repair_counter.items()
                )
            },
            "status": {
                str(key): value
                for key, value
                in sorted(
                    status_counter.items()
                )
            },
        }

    return result


# ============================================================
# 11. Behavior-centric Analysis
# ============================================================

def build_by_behavior(records):        ##以运行时三类行为为视角分析结果

    """
    从：

        Runtime Behavior
              ↓
        Outcome

    的方向进行分析。
    """

    result = {                     ###这是一个三层字典
        "retrieval_count": {},
        "repair_count": {},
        "status": {},
    }

    # --------------------------------------------------------
    # Retrieval Count
    # --------------------------------------------------------

    retrieval_groups = defaultdict(list)                 ###这里的数据结构应该是｛键值（当前以什么为核心分析谁就是键值此处为一种错误类型）：[内部类该种类型的键值所对应的多个案例]，键值（当前以什么为核心分析谁就是键值此处为另外一种错误类型）：[内部类该种类型的键值所对应的多个案例]｝

    for record in records:

        retrieval_groups[
            record["retrieval_count"]
        ].append(record)

    for value in sorted(
        retrieval_groups
    ):

        group = retrieval_groups[value]

        outcomes = Counter()

        for record in group:

            failure_type = record[
                "failure_type"
            ]

            if failure_type is None:

                outcome = "success"

            else:

                outcome = failure_type

            outcomes[outcome] += 1

        result[
            "retrieval_count"
        ][str(value)] = {
            "total": len(group),
            "outcomes": dict(
                sorted(
                    outcomes.items()
                )
            ),
        }

    # --------------------------------------------------------
    # Repair Count
    # --------------------------------------------------------

    repair_groups = defaultdict(list)

    for record in records:

        repair_groups[
            record["repair_count"]
        ].append(record)

    for value in sorted(
        repair_groups
    ):

        group = repair_groups[value]

        outcomes = Counter()

        for record in group:

            failure_type = record[
                "failure_type"
            ]

            if failure_type is None:

                outcome = "success"

            else:

                outcome = failure_type

            outcomes[outcome] += 1

        result[
            "repair_count"            ##这一行是进入result的"repair_count"字段
        ][str(value)] = {             ##这一行是在"repair_count"内部创建一个字段名为的value代指内容且已经被字符串话，字段值为一个字典内部包含如下信息          
            "total": len(group),
            "outcomes": dict(
                sorted(                 
                    outcomes.items()##把counter（）子字典对象变成一个列表（内部元素是一个个元组）在排序，最后又转化成字典
                )
            ),
        }

    # --------------------------------------------------------
    # Status
    # --------------------------------------------------------

    status_groups = defaultdict(list)

    for record in records:

        status_groups[
            record["status"]
        ].append(record)

    for value in sorted(
        status_groups
    ):

        group = status_groups[value]

        outcomes = Counter()

        for record in group:

            failure_type = record[
                "failure_type"
            ]

            if failure_type is None:

                outcome = "success"

            else:

                outcome = failure_type

            outcomes[outcome] += 1

        result[
            "status"
        ][str(value)] = {
            "total": len(group),
            "outcomes": dict(
                sorted(
                    outcomes.items()
                )
            ),
        }

    return result


# ============================================================
# 12. Pattern Detection      这一步真正开始总结规律，注意由架构决定的固定契约不能作为规律总结，规律也不能看做根本原因，还需要结果后续的假设和验证
# ============================================================

def detect_patterns(records):

    """
    只针对 Failure Type：

        failure_type
            ×
        runtime behavior

    寻找明显集中现象。

    Pattern 条件：

        sample_size >= MIN_PATTERN_SAMPLES

        AND

        count / sample_size
        >= PATTERN_THRESHOLD

    注意：

        Pattern != Cause
    """

    patterns = []

    grouped = defaultdict(list)

    for record in records:

        failure_type = record[
            "failure_type"
        ]

        if failure_type is None:
            continue

        grouped[
            failure_type
        ].append(record)

    dimensions = [
        "retrieval_count",
        "repair_count",
        "status",
    ]

    for failure_type in sorted(
        grouped
    ):

        failure_records = grouped[
            failure_type
        ]

        sample_size = len(
            failure_records
        )

        for dimension in dimensions:

            counter = Counter(
                record[dimension]
                for record
                in failure_records
            )

            for value, count in sorted(
                counter.items(),
                key=lambda item: str(
                    item[0]
                ),
            ):

                rate = (
                    count / sample_size
                    if sample_size > 0
                    else 0.0
                )

                sample_size_warning = (
                    sample_size
                    < MIN_PATTERN_SAMPLES
                )

                pattern_detected = (
                    not sample_size_warning
                    and
                    rate
                    >= PATTERN_THRESHOLD
                )

                patterns.append(
                    {
                        "failure_type":
                            failure_type,

                        "dimension":
                            dimension,

                        "value":
                            value,

                        "count":
                            count,

                        "rate":
                            rate,

                        "sample_size":
                            sample_size,

                        "pattern_detected":
                            pattern_detected,

                        "sample_size_warning":
                            sample_size_warning,
                    }
                )

    return patterns


# ============================================================
# 13. Warnings
# ============================================================

def build_warnings(
    total_cases,
    patterns,
):

    warnings = []

    if total_cases < MIN_PATTERN_SAMPLES:

        warnings.append(
            "Total sample size is too small "
            "for strong empirical conclusions."
        )

    warnings.append(
        "Observed patterns should not be "
        "interpreted as causal relationships."
    )

    if any(
        pattern["sample_size_warning"]
        for pattern in patterns
    ):

        warnings.append(
            "Some failure patterns have "
            "insufficient sample size."
        )

    return warnings


# ============================================================
# 14. 主分析函数
# ============================================================

def analyze():

    # --------------------------------------------------------
    # 读取输入
    # --------------------------------------------------------

    evaluation_report = load_json(
        INPUT_20_8
    )

    failure_report = load_json(
        INPUT_20_9
    )

    # --------------------------------------------------------
    # 构造 Failure Map
    # --------------------------------------------------------

    failure_map = build_failure_map(
        failure_report
    )

    # --------------------------------------------------------
    # 检查 Case ID
    # --------------------------------------------------------

    validate_failure_case_ids(
        evaluation_report,
        failure_map,
    )

    # --------------------------------------------------------
    # 构造 Analysis Records
    # --------------------------------------------------------

    records = build_analysis_records(
        evaluation_report,
        failure_map,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = build_summary(
        records
    )

    # --------------------------------------------------------
    # Failure Distribution
    # --------------------------------------------------------

    failure_distribution = (
        build_failure_distribution(
            records
        )
    )

    # --------------------------------------------------------
    # Behavioral Analysis
    # --------------------------------------------------------

    by_failure_type = (
        build_by_failure_type(
            records
        )
    )

    by_behavior = (
        build_by_behavior(
            records
        )
    )

    behavioral_analysis = {
        "by_failure_type":
            by_failure_type,

        "by_behavior":
            by_behavior,
    }

    # --------------------------------------------------------
    # Pattern Detection
    # --------------------------------------------------------

    patterns = detect_patterns(
        records
    )

    # --------------------------------------------------------
    # Warnings
    # --------------------------------------------------------

    warnings = build_warnings(
        summary["total_cases"],
        patterns,
    )

    # --------------------------------------------------------
    # 最终 Report
    # --------------------------------------------------------

    report = {

        "evaluation_stage":
            "20.11",

        "evaluation_name":
            "Failure Pattern Analysis",

        "summary":
            summary,

        "failure_distribution":
            failure_distribution,

        "behavioral_analysis":
            behavioral_analysis,

        "patterns":
            patterns,

        "warnings":
            warnings,
    }

    # --------------------------------------------------------
    # 输出
    # --------------------------------------------------------

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            ensure_ascii=False,
            indent=2,
        )

    return report


# ============================================================
# 15. 程序入口
# ============================================================

if __name__ == "__main__":

    report = analyze()

    print(
        "\n=== 20.11 Failure Pattern Analysis ==="
    )

    print(
        f"Total Cases: "
        f"{report['summary']['total_cases']}"
    )

    print(
        f"Failed Cases: "
        f"{report['summary']['failed_cases']}"
    )

    print(
        f"Failure Rate: "
        f"{report['summary']['failure_rate']:.2%}"
    )

    print(
        "\nFailure Distribution:"
    )

    for failure_type, data in (
        report[
            "failure_distribution"
        ].items()
    ):

        print(
            f"  {failure_type}: "
            f"{data['count']} "
            f"({data['rate']:.2%})"
        )

    print(
        "\nPatterns:"
    )

    for pattern in report["patterns"]:

        print(
            f"  {pattern['failure_type']} "
            f"| "
            f"{pattern['dimension']}="
            f"{pattern['value']} "
            f"| "
            f"count="
            f"{pattern['count']} "
            f"| "
            f"rate="
            f"{pattern['rate']:.2%} "
            f"| "
            f"detected="
            f"{pattern['pattern_detected']}"
        )

    print(
        "\nOutput:"
        f" {OUTPUT_FILE}"
    )