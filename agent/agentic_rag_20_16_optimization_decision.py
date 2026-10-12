import json ##此文件的作用是根据20.15的实验分析，判断是否有优化运行时agent的必要(只是业务会考虑成本--延迟--错误率以及其他特殊需求中间的关系综合权衡优化的必要性)，如果有必要后续：修改agent变量----验证新agent实际效果---判断是否真的有优化的必要----实际优化升级agent
from pathlib import Path
from collections import Counter
from typing import Any


# ============================================================
# 1. 路径配置
# ============================================================

BASE_DIR = Path(r"D:\MyRAG")

INPUT_PATH = (
    BASE_DIR
    / "agentic_rag_20_15_experiment_analysis.json"
)

OUTPUT_PATH = (
    BASE_DIR
    / "agentic_rag_20_16_optimization_decision.json"
)


# ============================================================
# 2. Contract 配置
# ============================================================

SUPPORTED_RESULTS = {
    "supported",
    "rejected",
    "inconclusive",
    "not_experimentable",
}

VALID_DECISIONS = {
    "optimize",
    "keep",
    "investigate",
}

REQUIRED_TOP_LEVEL_FIELDS = {
    "evaluation_stage",
    "summary",
    "experiments",
}

REQUIRED_EXPERIMENT_FIELDS = {
    "experiment_id",
    "hypothesis_id",
    "experimentability",
    "metric",
    "control_value",
    "treatment_value",
    "difference",
    "expected_direction",
    "observed_direction",
    "result",
}


# ============================================================
# 3. 输入加载
# ============================================================

def load_json(file_path: Path) -> dict[str, Any]:
    """
    读取 JSON 文件，并检查顶层数据类型。
    """

    if not file_path.exists():
        raise FileNotFoundError(
            f"找不到 20.15 输入文件：{file_path}\n"
            "请将 20.15 的真实输出保存到该路径，"
            "或者修改 INPUT_PATH。"
        )

    with file_path.open(
        "r",
        encoding="utf-8"
    ) as file:

        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError(
            "20.15 JSON 的顶层必须是对象。"
        )

    return data


# ============================================================
# 4. 输入 Contract 验证
# ============================================================

def validate_input(
    data: dict[str, Any]
) -> None:
    """
    验证 20.15 输出是否满足 20.16 的输入 Contract。

    这里只验证数据结构和关键字段。
    不重新计算实验结果。
    """

    missing_fields = (
        REQUIRED_TOP_LEVEL_FIELDS
        - set(data.keys())
    )

    if missing_fields:
        raise ValueError(
            "20.15 缺少顶层字段："
            f"{sorted(missing_fields)}"
        )

    if data["evaluation_stage"] != "20.15":
        raise ValueError(
            "输入 evaluation_stage 必须为 20.15。"
        )

    summary = data["summary"]

    if not isinstance(summary, dict):
        raise ValueError(
            "summary 必须是字典。"
        )

    experiments = data["experiments"]

    if not isinstance(experiments, list):
        raise ValueError(
            "experiments 必须是列表。"
        )

    seen_experiment_ids = set()

    for index, experiment in enumerate(experiments):

        if not isinstance(experiment, dict):
            raise ValueError(
                f"experiments[{index}] 必须是字典。"
            )

        missing_experiment_fields = (
            REQUIRED_EXPERIMENT_FIELDS
            - set(experiment.keys())
        )

        if missing_experiment_fields:
            raise ValueError(
                f"实验索引 {index} 缺少字段："
                f"{sorted(missing_experiment_fields)}"
            )

        experiment_id = experiment["experiment_id"]

        if (
            not isinstance(experiment_id, str)
            or not experiment_id.strip()
        ):
            raise ValueError(
                f"实验索引 {index} 的 experiment_id 无效。"
            )

        if experiment_id in seen_experiment_ids:
            raise ValueError(
                f"发现重复实验编号：{experiment_id}"
            )

        seen_experiment_ids.add(experiment_id)

        result = experiment["result"]

        if result not in SUPPORTED_RESULTS:
            raise ValueError(
                f"{experiment_id} 的 result 无效："
                f"{result}"
            )

        experimentability = experiment[
            "experimentability"
        ]

        if experimentability not in {
            "controllable",
            "not_directly_controllable",
        }:
            raise ValueError(
                f"{experiment_id} 的 experimentability 无效："
                f"{experimentability}"
            )

        if (
            result == "not_experimentable"
            and experimentability
            != "not_directly_controllable"
        ):
            raise ValueError(
                f"{experiment_id} 的结果与可实验性不一致："
                "not_experimentable 应对应 "
                "not_directly_controllable。"
            )

        if (
            result != "not_experimentable"
            and experimentability
            != "controllable"
        ):
            raise ValueError(
                f"{experiment_id} 的结果与可实验性不一致。"
            )

    if summary.get("total_experiments") != len(experiments):
        raise ValueError(
            "summary.total_experiments 与 experiments 列表长度不一致。"
        )


# ============================================================
# 5. 单个实验的决策
# ============================================================

def decide_experiment(
    experiment: dict[str, Any]
) -> dict[str, Any]:
    """
    根据 20.15 已有结论，生成 20.16 工程决策。

    本版本采取保守规则：
    当前字段不足以支持直接优化或确定保留，
    因此需要进一步调查。
    """

    experiment_id = experiment["experiment_id"]

    hypothesis_id = experiment["hypothesis_id"]

    result = experiment["result"]

    experimentability = experiment[
        "experimentability"
    ]

    if result == "not_experimentable":

        decision = "investigate"

        reason = (
            "当前实验无法直接控制对应变量，"
            "需要重新设计可验证的实验方案。"
        )

    elif result == "supported":

        decision = "investigate"

        reason = (
            "实验结果支持原假设，但当前输入未提供"
            "足够的优化收益、成本和风险信息，"
            "暂不直接修改运行时配置。"
        )

    elif result == "rejected":

        decision = "investigate"

        reason = (
            "实验结果未支持原假设。"
            "需要进一步检查样本量、干预是否生效"
            "以及实验是否具备区分不同配置的能力。"
        )

    elif result == "inconclusive":

        decision = "investigate"

        reason = (
            "实验结果不明确，当前证据不足以"
            "支持直接优化或确定保留。"
        )

    else:

        # 正常情况下，validate_input 已经阻止该分支。
        raise ValueError(
            f"{experiment_id} 存在无法处理的实验结果："
            f"{result}"
        )

    return {
        "experiment_id": experiment_id,
        "hypothesis_id": hypothesis_id,
        "experiment_result": result,
        "experimentability": experimentability,
        "decision": decision,
        "reason": reason,
        "runtime_modified": False,
    }


# ============================================================
# 6. 全部实验的决策
# ============================================================

def make_optimization_decision(
    data: dict[str, Any]
) -> dict[str, Any]:
    """
    将 20.15 实验分析结果转换为 20.16 决策结果。
    """

    validate_input(data)

    experiments = data["experiments"]

    decisions = [
        decide_experiment(experiment)
        for experiment in experiments
    ]

    decision_counts = Counter(
        item["decision"]
        for item in decisions
    )

    result_counts = Counter(
        item["experiment_result"]
        for item in decisions
    )

    return {
        "evaluation_stage": "20.16",
        "evaluation_name": "Optimization Decision",
        "input_stage": data["evaluation_stage"],
        "summary": {
            "total_experiments": len(decisions),
            "optimize": decision_counts["optimize"],
            "keep": decision_counts["keep"],
            "investigate": decision_counts["investigate"],
            "source_supported": result_counts["supported"],
            "source_rejected": result_counts["rejected"],
            "source_inconclusive": result_counts["inconclusive"],
            "source_not_experimentable": (
                result_counts["not_experimentable"]
            ),
            "runtime_modified": False,
        },
        "decisions": decisions,
    }


# ============================================================
# 7. 输出保存
# ============================================================

def save_json(
    data: dict[str, Any],
    file_path: Path
) -> None:
    """
    将决策结果保存为 JSON。
    """

    file_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with file_path.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=4
        )


# ============================================================
# 8. 测试断言                ###成、assert 后面条件为true正常运行，如果为false直接报错  Raise AssertErroe
# ============================================================

def run_assertions(
    source_data: dict[str, Any],
    result: dict[str, Any]
) -> None:
    """
    验证决策输出的关键不变量。
    """

    decisions = result["decisions"]

    assert len(decisions) == len(
        source_data["experiments"]
    )

    source_by_id = {
        item["experiment_id"]: item
        for item in source_data["experiments"]
    }

    for item in decisions:

        experiment_id = item["experiment_id"]

        # 20.16 必须保留 20.15 的实验结论。
        assert item["experiment_result"] == (
            source_by_id[experiment_id]["result"]
        )

        # 当前版本的保守规则不直接执行优化。
        assert item["decision"] == "investigate"

        # 20.16 不得修改 Runtime。
        assert item["runtime_modified"] is False

        # 决策值必须属于约定的枚举。
        assert item["decision"] in VALID_DECISIONS

    assert result["summary"]["total_experiments"] == 3

    assert result["summary"]["investigate"] == 3

    assert result["summary"]["runtime_modified"] is False


# ============================================================
# 9. 主程序
# ============================================================

def main() -> None:

    print("=" * 60)
    print("20.16 Optimization Decision")
    print("=" * 60)

    # 读取真实的 20.15 输出。
    source_data = load_json(INPUT_PATH)

    print("\n[1] 20.15 输入读取成功")

    print(
        "实验总数：",
        source_data["summary"]["total_experiments"]
    )

    # 生成 20.16 决策。
    result = make_optimization_decision(
        source_data
    )

    print("\n[2] 20.16 决策生成完成")

    # 打印逐实验结果。
    for item in result["decisions"]:

        print("\n实验编号：", item["experiment_id"])

        print(
            "20.15 实验结果：",
            item["experiment_result"]
        )

        print(
            "20.16 工程决策：",
            item["decision"]
        )

        print(
            "决策原因：",
            item["reason"]
        )

        print(
            "是否修改 Runtime：",
            item["runtime_modified"]
        )

    # 执行断言测试。
    run_assertions(
        source_data,
        result
    )

    print("\n[3] 所有断言测试通过")

    # 保存 JSON。
    save_json(
        result,
        OUTPUT_PATH
    )

    print("\n[4] 输出文件保存成功")

    print(OUTPUT_PATH)

    print("\n汇总：")

    print(
        json.dumps(
            result["summary"],
            ensure_ascii=False,
            indent=4
        )
    )


if __name__ == "__main__":
    main()