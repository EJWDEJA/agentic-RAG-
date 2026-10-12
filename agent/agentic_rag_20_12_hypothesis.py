import json                   ###本次测试旨在针对20.11提出的规律进行文字化现象总结和文字化假设推演（假设的可信度完全有规律是否通过强模式规律筛选，规律强模式规律可信度高，弱模式，规律可信度低）
from pathlib import Path


# ============================================================
# 1. 文件配置
# ============================================================

INPUT_FILE = Path(
    "agentic_rag_20_11_failure_analysis.json"
)

OUTPUT_FILE = Path(
    "agentic_rag_20_12_hypothesis.json"
)


# ============================================================
# 2. 固定值
# ============================================================

VALID_STATUSES = {
    "unverified",
    "supported",
    "rejected",
}

VALID_EVIDENCE_STRENGTHS = {
    "strong",
    "weak",
}


# ============================================================
# 3. 读取 JSON
# ============================================================

def load_json(path):
    """
    读取 JSON 文件。
    """

    with path.open(
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


# ============================================================
# 4. 根据 Pattern 判断证据强度
# ============================================================

def determine_evidence_strength(pattern):
    """
    根据 20.11 的 Pattern 筛选结果，
    判断当前 Hypothesis 的证据强度。

    pattern_detected=True
        → strong

    pattern_detected=False
        → weak
    """

    if pattern["pattern_detected"]:
        return "strong"

    return "weak"


# ============================================================
# 5. 构造 Observation
# ============================================================

def build_observation(pattern):
    """
    将 20.11 的统计结果转换成自然语言 Observation。

    注意：
    Observation 只描述数据现象，
    不解释原因。
    """

    failure_type = pattern["failure_type"]
    dimension = pattern["dimension"]
    value = pattern["value"]
    count = pattern["count"]
    sample_size = pattern["sample_size"]
    rate = pattern["rate"]

    percentage = f"{rate:.0%}"

    return (
        f"{sample_size} 个 {failure_type} 案例中，"
        f"{count} 个案例的 "
        f"{dimension} = {value}，"
        f"占该失败类型样本的 {percentage}。"
    )


# ============================================================
# 6. 构造 Hypothesis
# ============================================================

def build_hypothesis_text(pattern):
    """
    根据 Pattern 构造待验证假设。

    注意：
    这里只能使用“可能”“可能与……有关”
    这样的假设表达。

    不能直接写成“导致”。
    """

    failure_type = pattern["failure_type"]
    dimension = pattern["dimension"]
    value = pattern["value"]

    return (
        f"{dimension} = {value} "
        f"可能与 {failure_type} 的发生存在关联。"
    )


# ============================================================
# 7. 构造 Condition
# ============================================================

def build_condition(pattern):
    """
    把 Pattern 的 dimension + value
    转换成假设条件。
    """

    dimension = pattern["dimension"]
    value = pattern["value"]

    if isinstance(value, str):
        return f'{dimension} == "{value}"'

    return f"{dimension} == {value}"


# ============================================================
# 8. 构造单个 Hypothesis
# ============================================================

def build_hypothesis(pattern, index):
    """
    将一个 20.11 Pattern
    转换成一个 20.12 Hypothesis。
    """

    pattern_detected = pattern["pattern_detected"]

    evidence_strength = (
        determine_evidence_strength(pattern)
    )

    return {
        "hypothesis_id": f"H{index:03d}",

        "failure_type":
            pattern["failure_type"],

        "dimension":
            pattern["dimension"],

        "condition":
            build_condition(pattern),

        "observation":
            build_observation(pattern),

        "hypothesis":
            build_hypothesis_text(pattern),

        "pattern_detected":
            pattern_detected,

        "sample_size":
            pattern["sample_size"],

        "pattern_rate":
            pattern["rate"],

        "evidence_strength":
            evidence_strength,

        "status":
            "unverified",
    }


# ============================================================
# 9. 构造 Summary
# ============================================================

def build_summary(hypotheses):
    """
    统计正式候选和探索性候选。
    """

    total = len(hypotheses)

    formal_candidates = sum(
        1
        for hypothesis in hypotheses
        if hypothesis["evidence_strength"]
        == "strong"
    )

    exploratory_candidates = sum(
        1
        for hypothesis in hypotheses
        if hypothesis["evidence_strength"]
        == "weak"
    )

    return {
        "total_hypotheses": total,

        "formal_candidates":
            formal_candidates,

        "exploratory_candidates":
            exploratory_candidates,
    }


# ============================================================
# 10. 构造 Warnings
# ============================================================

def build_warnings(hypotheses):
    """
    生成 Hypothesis 层面的警告。

    这里不否定 Hypothesis，
    只是提醒后续实验阶段注意证据强度。
    """

    warnings = []

    weak_count = sum(
        1
        for hypothesis in hypotheses
        if hypothesis["evidence_strength"]
        == "weak"
    )

    if weak_count > 0:

        warnings.append(
            "部分 Hypothesis 基于未通过 Pattern "
            "样本量筛选的观察结果，仅属于探索性假设。"
        )

    warnings.append(
        "Hypothesis 仅表示待验证命题，"
        "不能直接视为因果关系或 Agent 优化依据。"
    )

    warnings.append(
        "真实 Evaluation 数据未被修改；"
        "后续验证应通过新的 Experiment 数据完成。"
    )

    return warnings


# ============================================================
# 11. 主分析函数
# ============================================================

def analyze():

    # --------------------------------------------------------
    # 读取 20.11
    # --------------------------------------------------------

    report = load_json(INPUT_FILE)

    patterns = report.get(
        "patterns",
        []
    )

    hypotheses = []

    hypothesis_index = 1

    # --------------------------------------------------------
    # Pattern → Hypothesis
    # --------------------------------------------------------

    for pattern in patterns:

        # --------------------------------------------
        # 只处理真正存在观察样本的 Pattern
        # --------------------------------------------

        if pattern["count"] <= 0:
            continue

        if pattern["sample_size"] <= 0:
            continue

        hypothesis = build_hypothesis(
            pattern,
            hypothesis_index
        )

        hypotheses.append(
            hypothesis
        )

        hypothesis_index += 1

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = build_summary(
        hypotheses
    )

    # --------------------------------------------------------
    # Warnings
    # --------------------------------------------------------

    warnings = build_warnings(
        hypotheses
    )

    # --------------------------------------------------------
    # 最终输出
    # --------------------------------------------------------

    result = {

        "evaluation_stage":
            "20.12",

        "evaluation_name":
            "Hypothesis Management",

        "summary":
            summary,

        "hypotheses":
            hypotheses,

        "warnings":
            warnings,
    }

    # --------------------------------------------------------
    # 写文件
    # --------------------------------------------------------

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            result,
            f,
            ensure_ascii=False,
            indent=2
        )

    # --------------------------------------------------------
    # 控制台输出
    # --------------------------------------------------------

    print("=" * 60)
    print("20.12 Hypothesis Management")
    print("=" * 60)

    print()

    print(
        f"Total Hypotheses: "
        f"{summary['total_hypotheses']}"
    )

    print(
        f"Formal Candidates: "
        f"{summary['formal_candidates']}"
    )

    print(
        f"Exploratory Candidates: "
        f"{summary['exploratory_candidates']}"
    )

    print()

    for hypothesis in hypotheses:

        print(
            f"{hypothesis['hypothesis_id']} | "
            f"{hypothesis['failure_type']} | "
            f"{hypothesis['dimension']} | "
            f"{hypothesis['condition']} | "
            f"{hypothesis['evidence_strength']} | "
            f"{hypothesis['status']}"
        )

    print()

    print(
        f"Output: {OUTPUT_FILE}"
    )


# ============================================================
# 12. 程序入口
# ============================================================

if __name__ == "__main__":
    analyze()