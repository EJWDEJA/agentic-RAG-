import json                    ##本文件的作用是根据20.13的期望答案指标评估规则，分析20.14实际实验产生结果，输出支持还是1拒绝选项
import re
from pathlib import Path


# ============================================================
# 20.15 Experiment Result Analysis
#
# 输入：
#     20.13 Experiment Design
#     20.14 Experiment Execution
#
# 输出：
#     20.15 Experiment Result Analysis
#
# 核心职责：
#
#     20.13
#         ↓
#     Expected Outcome
#         ↓
#     20.14
#         ↓
#     Actual Control / Treatment Result
#         ↓
#     20.15
#         ↓
#     Supported / Rejected / Inconclusive
#
# 特殊情况：
#
#     20.13 not_experimentable
#         ↓
#     不进入实验结果分析
#         ↓
#     result = not_experimentable
#
# 注意：
#     20.15 不重新运行 Agent
#     20.15 不重新调用 LLM Judge
#     20.15 不修改 Runtime
# ============================================================


# ============================================================
# 1. 路径配置
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

ROOT_DIR = BASE_DIR.parent


DESIGN_FILE = (
    ROOT_DIR /
    "agentic_rag_20_13_experiment_design.json"
)


EXECUTION_FILE = (
    ROOT_DIR /
    "agentic_rag_20_14_experiment_execution.json"
)


OUTPUT_FILE = (
    ROOT_DIR /
    "agentic_rag_20_15_experiment_analysis.json"
)


# ============================================================
# 2. JSON 加载
# ============================================================

def load_json(
    file_path: Path
):
    """
    读取 JSON 文件。
    """

    if not file_path.exists():

        raise FileNotFoundError(
            f"找不到文件：{file_path}"
        )

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


# ============================================================
# 3. 验证 20.13
# ============================================================

def validate_design(
    design
):
    """
    验证输入是否真的是 20.13 Experiment Design。
    """

    if not isinstance(
        design,
        dict
    ):

        raise TypeError(
            "20.13 数据必须是 dict。"
        )

    if design.get(
        "evaluation_stage"
    ) != "20.13":

        raise ValueError(
            "输入文件不是 evaluation_stage=20.13。"
        )

    if design.get(
        "evaluation_name"
    ) != "Experiment Design":

        raise ValueError(
            "输入文件不是 Experiment Design。"
        )

    if "experiments" not in design:

        raise ValueError(
            "20.13 缺少 experiments。"
        )

    if not isinstance(
        design["experiments"],
        list
    ):

        raise TypeError(
            "20.13 experiments 必须是 list。"
        )


# ============================================================
# 4. 验证 20.14
# ============================================================

def validate_execution(
    execution
):
    """
    验证输入是否真的是 20.14 Experiment Execution。
    """

    if not isinstance(
        execution,
        dict
    ):

        raise TypeError(
            "20.14 数据必须是 dict。"
        )

    if execution.get(
        "evaluation_stage"
    ) != "20.14":

        raise ValueError(
            "输入文件不是 evaluation_stage=20.14。"
        )

    if execution.get(
        "evaluation_name"
    ) != "Experiment Execution":

        raise ValueError(
            "输入文件不是 Experiment Execution。"
        )

    if "experiments" not in execution:

        raise ValueError(
            "20.14 缺少 experiments。"
        )

    if not isinstance(
        execution["experiments"],
        list
    ):

        raise TypeError(
            "20.14 experiments 必须是 list。"
        )


# ============================================================
# 5. 建立 Experiment Index
# ============================================================

def build_experiment_index(
    experiments
):
    """
    把：

        [
            {
                "experiment_id": "E001",
                ...
            },
            {
                "experiment_id": "E002",
                ...
            }
        ]

    转换为：

        {
            "E001": {...},
            "E002": {...}
        }

    方便后面根据 experiment_id
    对齐 20.13 和 20.14。
    """

    index = {}

    for experiment in experiments:

        experiment_id = experiment.get(
            "experiment_id"
        )

        if experiment_id is None:

            raise ValueError(
                "Experiment 缺少 experiment_id。"
            )

        if experiment_id in index:

            raise ValueError(
                f"发现重复 experiment_id："
                f"{experiment_id}"
            )

        index[
            experiment_id
        ] = experiment

    return index


# ============================================================
# 6. 判断 Experiment 是否属于
#    not_experimentable
# ============================================================

def is_not_experimentable(
    design_experiment
):
    """
    判断 20.13 中的 Experiment 是否明确不可实验。

    当前 20.13 的 E003：

        "experimentability":
            "not_directly_controllable"

        "status":
            "not_experimentable"

    如果明确属于这种情况，
    20.15 不应该要求：

        metric
        expected_outcome
        control
        treatment

    因为它根本没有进入 20.14 实验执行阶段。
    """

    experimentability = (
        design_experiment.get(
            "experimentability"
        )
    )

    status = (
        design_experiment.get(
            "status"
        )
    )

    if status == "not_experimentable":

        return True

    if (
        experimentability
        == "not_directly_controllable"
    ):

        return True

    return False


# ============================================================
# 7. 提取 Metric 名称
# ============================================================

def get_metric_name(
    design_experiment
):
    """
    从 20.13 中读取：

        "metric": {
            "name": "grounding_failure_rate",
            ...
        }

    注意：

    该函数只允许用于真正可执行的 Experiment。

    not_experimentable Experiment
    不应该调用此函数。
    """

    metric = design_experiment.get(
        "metric"
    )

    if not isinstance(
        metric,
        dict
    ):

        raise ValueError(
            f"{design_experiment['experiment_id']} "
            "的 metric 不是 dict。"
        )

    metric_name = metric.get(
        "name"
    )

    if not metric_name:

        raise ValueError(
            f"{design_experiment['experiment_id']} "
            "缺少 metric.name。"
        )

    return metric_name


# ============================================================
# 8. 从 20.14 提取 Metric
# ============================================================

def extract_metric_value(
    execution_group,
    metric_name
):
    """
    当前 20.14 的结构：

        control:
            results: [...]

            metrics:
                {
                    "total_cases": 3,
                    "grounding_failure_cases": 2,
                    "grounding_failure_rate": 0.666...
                }

        treatment:
            results: [...]

            metrics:
                {
                    ...
                }

    从 metrics 中读取指定指标。
    """

    metrics = execution_group.get(
        "metrics"
    )

    if not isinstance(
        metrics,
        dict
    ):

        raise ValueError(
            "20.14 缺少合法的 metrics。"
        )

    if metric_name not in metrics:

        raise ValueError(
            f"20.14 metrics 中不存在："
            f"{metric_name}"
        )

    value = metrics[
        metric_name
    ]

    if not isinstance(
        value,
        (int, float)
    ):

        raise TypeError(
            f"Metric {metric_name} "
            "必须是数字。"
        )

    return float(value)


# ============================================================
# 9. 解析 Expected Outcome
# ============================================================

def parse_expected_outcome(
    design_experiment
):
    """
    当前 20.13：

        "expected_outcome": {
            "if_hypothesis_supported":
                "treatment grounding_failure_rate > control grounding_failure_rate",

            "if_hypothesis_rejected":
                "treatment grounding_failure_rate <= control grounding_failure_rate"
        }

    20.15 不自己创造新的实验规则。

    直接从 20.13 Expected Outcome
    中读取比较方向。
    """

    expected_outcome = design_experiment.get(
        "expected_outcome"
    )

    if not isinstance(
        expected_outcome,
        dict
    ):

        raise ValueError(
            f"{design_experiment['experiment_id']} "
            "缺少 expected_outcome。"
        )

    supported_rule = expected_outcome.get(
        "if_hypothesis_supported"
    )

    rejected_rule = expected_outcome.get(
        "if_hypothesis_rejected"
    )

    if not supported_rule:

        raise ValueError(
            f"{design_experiment['experiment_id']} "
            "缺少 if_hypothesis_supported。"
        )

    if not rejected_rule:

        raise ValueError(
            f"{design_experiment['experiment_id']} "
            "缺少 if_hypothesis_rejected。"
        )

    return {
        "supported_rule":
            supported_rule,

        "rejected_rule":
            rejected_rule
    }


# ============================================================
# 10. 判断比较方向
# ============================================================

def determine_observed_direction(
    control_value,
    treatment_value
):
    """
    只描述实际观察到的方向。

    treatment > control
        -> treatment > control

    treatment < control
        -> treatment < control

    treatment == control
        -> treatment == control
    """

    if treatment_value > control_value:

        return "treatment > control"

    if treatment_value < control_value:

        return "treatment < control"

    return "treatment == control"


# ============================================================
# 11. 根据 Expected Outcome 判断结果
# ============================================================

def determine_result(
    control_value,
    treatment_value,
    expected_outcome
):
    """
    根据 20.13 Expected Outcome
    判断：

        supported
        rejected
        inconclusive
    """

    supported_rule = (
        expected_outcome[
            "supported_rule"
        ]
    )

    rejected_rule = (
        expected_outcome[
            "rejected_rule"
        ]
    )

    observed_direction = (
        determine_observed_direction(
            control_value,
            treatment_value
        )
    )

    # --------------------------------------------------------
    # 从 supported rule 中提取比较符号
    # --------------------------------------------------------

    supported_match = re.search(              ###返回一个1re.match对象，，supported_match.group(1)这个的意思是提取第一个（）捕获到的文本信息此处只有一个（）
        r"(>=|<=|>|<|==)",
        supported_rule
    )

    rejected_match = re.search(
        r"(>=|<=|>|<|==)",
        rejected_rule
    )

    if (
        supported_match is None
        or
        rejected_match is None
    ):

        return (
            "inconclusive",
            observed_direction
        )

    supported_operator = (
        supported_match.group(1)
    )

    rejected_operator = (
        rejected_match.group(1)
    )

    # --------------------------------------------------------
    # 根据 supported rule 判断
    # --------------------------------------------------------

    supported = False

    if supported_operator == ">":

        supported = (
            treatment_value
            >
            control_value
        )

    elif supported_operator == ">=":

        supported = (
            treatment_value
            >=
            control_value
        )

    elif supported_operator == "<":

        supported = (
            treatment_value
            <
            control_value
        )

    elif supported_operator == "<=":

        supported = (
            treatment_value
            <=
            control_value
        )

    elif supported_operator == "==":

        supported = (
            treatment_value
            ==
            control_value
        )

    # --------------------------------------------------------
    # 根据 rejected rule 判断
    # --------------------------------------------------------

    rejected = False

    if rejected_operator == ">":

        rejected = (
            treatment_value
            >
            control_value
        )

    elif rejected_operator == ">=":

        rejected = (
            treatment_value
            >=
            control_value
        )

    elif rejected_operator == "<":

        rejected = (
            treatment_value
            <
            control_value
        )

    elif rejected_operator == "<=":

        rejected = (
            treatment_value
            <=
            control_value
        )

    elif rejected_operator == "==":

        rejected = (
            treatment_value
            ==
            control_value
        )

    # --------------------------------------------------------
    # supported / rejected
    # 应该覆盖所有情况
    #
    # 如果 20.13 的规则发生重叠，
    # 或出现无法判断的情况，
    # 不擅自选择。
    # --------------------------------------------------------

    if supported and not rejected:

        return (
            "supported",
            observed_direction
        )

    if rejected and not supported:

        return (
            "rejected",
            observed_direction
        )

    return (
        "inconclusive",
        observed_direction
    )


# ============================================================
# 12. 分析单个 Experiment
# ============================================================

def analyze_experiment(
    design_experiment,
    execution_experiment
):
    """
    对一个真正执行过的 Experiment 完成：

        20.13 Expected Outcome
                +
        20.14 Actual Result
                ↓
        20.15 Analysis
    """

    design_id = design_experiment[
        "experiment_id"
    ]

    execution_id = execution_experiment[
        "experiment_id"
    ]

    # --------------------------------------------------------
    # Experiment ID 必须一致
    # --------------------------------------------------------

    if design_id != execution_id:

        raise ValueError(
            "20.13 / 20.14 Experiment ID 不一致："
            f"{design_id} != {execution_id}"
        )

    # --------------------------------------------------------
    # Hypothesis ID
    # --------------------------------------------------------

    hypothesis_id = design_experiment.get(
        "hypothesis_id"
    )

    if hypothesis_id is None:

        raise ValueError(
            f"{design_id} 缺少 hypothesis_id。"
        )

    # --------------------------------------------------------
    # Metric
    # --------------------------------------------------------

    metric_name = get_metric_name(
        design_experiment
    )

    # --------------------------------------------------------
    # Expected Outcome
    # --------------------------------------------------------

    expected_outcome = parse_expected_outcome(
        design_experiment
    )

    # --------------------------------------------------------
    # Control / Treatment
    # --------------------------------------------------------

    control = execution_experiment.get(
        "control"
    )

    treatment = execution_experiment.get(
        "treatment"
    )

    if not isinstance(
        control,
        dict
    ):

        raise ValueError(
            f"{design_id} 缺少 control。"
        )

    if not isinstance(
        treatment,
        dict
    ):

        raise ValueError(
            f"{design_id} 缺少 treatment。"
        )

    # --------------------------------------------------------
    # 提取 Metric
    # --------------------------------------------------------

    control_value = extract_metric_value(
        control,
        metric_name
    )

    treatment_value = extract_metric_value(
        treatment,
        metric_name
    )

    # --------------------------------------------------------
    # Difference
    #
    # treatment - control
    # --------------------------------------------------------

    difference = (
        treatment_value
        -
        control_value
    )

    # --------------------------------------------------------
    # Observed Direction
    # --------------------------------------------------------

    observed_direction = (
        determine_observed_direction(
            control_value,
            treatment_value
        )
    )

    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    result, _ = determine_result(
        control_value=control_value,
        treatment_value=treatment_value,
        expected_outcome=expected_outcome
    )

    # --------------------------------------------------------
    # 返回 20.15 Analysis Result
    # --------------------------------------------------------

    return {
        "experiment_id":
            design_id,

        "hypothesis_id":
            hypothesis_id,

        "experimentability":
            design_experiment.get(
                "experimentability"
            ),

        "metric":
            metric_name,

        "control_value":
            control_value,

        "treatment_value":
            treatment_value,

        "difference":
            difference,

        "expected_direction":
            expected_outcome[
                "supported_rule"
            ],

        "rejected_direction":
            expected_outcome[
                "rejected_rule"
            ],

        "observed_direction":
            observed_direction,

        "result":
            result
    }


# ============================================================
# 13. 构造 Not Experimentable Result
# ============================================================

def build_not_experimentable_result(
    design_experiment
):
    """
    处理 20.13 明确标记为：

        not_experimentable

    的 Experiment。

    这种 Experiment：

        不需要 metric
        不需要 expected_outcome
        不需要 control
        不需要 treatment

    因此不能调用：

        get_metric_name()
        parse_expected_outcome()
        analyze_experiment()

    它只是被保留在 20.15 中，
    说明该假设没有进入实验执行阶段。
    """

    experiment_id = design_experiment.get(
        "experiment_id"
    )

    hypothesis_id = design_experiment.get(
        "hypothesis_id"
    )

    return {
        "experiment_id":
            experiment_id,

        "hypothesis_id":
            hypothesis_id,

        "experimentability":
            design_experiment.get(
                "experimentability"
            ),

        "metric":
            None,

        "control_value":
            None,

        "treatment_value":
            None,

        "difference":
            None,

        "expected_direction":
            None,

        "rejected_direction":
            None,

        "observed_direction":
            None,

        "result":
            "not_experimentable"
    }


# ============================================================
# 14. 构造未执行 Experiment Result
# ============================================================

def build_inconclusive_result(
    design_experiment
):
    """
    处理：

        20.13 是可实验的
        但是 20.14 没有对应 Experiment

    这种情况下：

        不能判定 supported
        不能判定 rejected

    所以：

        inconclusive
    """

    experiment_id = (
        design_experiment[
            "experiment_id"
        ]
    )

    metric_name = get_metric_name(
        design_experiment
    )

    expected_outcome = parse_expected_outcome(
        design_experiment
    )

    return {
        "experiment_id":
            experiment_id,

        "hypothesis_id":
            design_experiment.get(
                "hypothesis_id"
            ),

        "experimentability":
            design_experiment.get(
                "experimentability"
            ),

        "metric":
            metric_name,

        "control_value":
            None,

        "treatment_value":
            None,

        "difference":
            None,

        "expected_direction":
            expected_outcome[
                "supported_rule"
            ],

        "rejected_direction":
            expected_outcome[
                "rejected_rule"
            ],

        "observed_direction":
            None,

        "result":
            "inconclusive"
    }


# ============================================================
# 15. 分析所有 Experiments
# ============================================================

def analyze_all_experiments(
    design,
    execution
):
    """
    对齐：

        20.13 experiments
            ↕
        20.14 experiments

    然后逐个处理。

    三种情况：

    1.
        20.13 controllable
        +
        20.14 存在
        ↓
        正常实验分析

    2.
        20.13 controllable
        +
        20.14 不存在
        ↓
        inconclusive

    3.
        20.13 not_experimentable
        ↓
        not_experimentable

    注意：

    第 3 种情况不能调用
    get_metric_name()。
    """

    design_index = build_experiment_index(
        design["experiments"]
    )

    execution_index = build_experiment_index(
        execution["experiments"]
    )

    results = []

    # --------------------------------------------------------
    # 以 20.13 为基准
    # --------------------------------------------------------

    for experiment_id in design_index:

        design_experiment = design_index[
            experiment_id
        ]

        # ----------------------------------------------------
        # 情况 A：
        #
        # 20.13 明确不可实验
        #
        # 必须优先判断。
        #
        # 因为它本身就不需要：
        #
        # metric
        # expected_outcome
        # control
        # treatment
        # ----------------------------------------------------

        if is_not_experimentable(
            design_experiment
        ):

            results.append(
                build_not_experimentable_result(
                    design_experiment
                )
            )

            continue

        # ----------------------------------------------------
        # 情况 B：
        #
        # 可实验，但 20.14 没有执行结果
        # ----------------------------------------------------

        if experiment_id not in execution_index:

            results.append(
                build_inconclusive_result(
                    design_experiment
                )
            )

            continue

        # ----------------------------------------------------
        # 情况 C：
        #
        # 正常分析
        # ----------------------------------------------------

        analysis = analyze_experiment(
            design_experiment=
                design_experiment,

            execution_experiment=
                execution_index[
                    experiment_id
                ]
        )

        results.append(
            analysis
        )

    # --------------------------------------------------------
    # 20.14 多出来的 Experiment
    #
    # 不能静默忽略。
    # --------------------------------------------------------

    extra_experiments = (
        set(execution_index)
        -
        set(design_index)
    )

    if extra_experiments:

        raise ValueError(
            "20.14 存在 20.13 未定义的 Experiment："
            +
            ", ".join(
                sorted(
                    extra_experiments
                )
            )
        )

    return results


# ============================================================
# 16. 生成 Summary
# ============================================================

def build_summary(
    results
):
    """
    汇总：

        total_experiments
        supported
        rejected
        inconclusive
        not_experimentable
    """

    summary = {
        "total_experiments":
            len(results),

        "supported":
            0,

        "rejected":
            0,

        "inconclusive":
            0,

        "not_experimentable":
            0
    }

    for result in results:

        status = result[
            "result"
        ]

        if status == "supported":

            summary[
                "supported"
            ] += 1

        elif status == "rejected":

            summary[
                "rejected"
            ] += 1

        elif status == "inconclusive":

            summary[
                "inconclusive"
            ] += 1

        elif status == "not_experimentable":

            summary[
                "not_experimentable"
            ] += 1

        else:

            raise ValueError(
                f"未知 Experiment Result："
                f"{status}"
            )

    return summary


# ============================================================
# 17. Main
# ============================================================

def main():

    print(
        "=" * 70
    )

    print(
        "20.15 EXPERIMENT RESULT ANALYSIS"
    )

    print(
        "=" * 70
    )

    print()

    print(
        f"Project Root: {ROOT_DIR}"
    )

    # ========================================================
    # Step 1
    # 读取 20.13
    # ========================================================

    design = load_json(
        DESIGN_FILE
    )

    validate_design(
        design
    )

    print()

    print(
        f"Loaded 20.13: {DESIGN_FILE}"
    )

    print(
        f"20.13 experiments: "
        f"{len(design['experiments'])}"
    )

    # ========================================================
    # Step 2
    # 读取 20.14
    # ========================================================

    execution = load_json(
        EXECUTION_FILE
    )

    validate_execution(
        execution
    )

    print()

    print(
        f"Loaded 20.14: {EXECUTION_FILE}"
    )

    print(
        f"20.14 experiments: "
        f"{len(execution['experiments'])}"
    )

    # ========================================================
    # Step 3
    # 分析实验
    # ========================================================

    print()

    print(
        "=" * 70
    )

    print(
        "ANALYZING EXPERIMENTS"
    )

    print(
        "=" * 70
    )

    results = analyze_all_experiments(
        design=design,
        execution=execution
    )

    # ========================================================
    # Step 4
    # 打印每个实验结果
    # ========================================================

    for result in results:

        print()

        print(
            "-" * 70
        )

        print(
            f"Experiment: "
            f"{result['experiment_id']}"
        )

        print(
            f"Hypothesis: "
            f"{result['hypothesis_id']}"
        )

        print(
            f"Experimentability: "
            f"{result['experimentability']}"
        )

        print(
            f"Metric: "
            f"{result['metric']}"
        )

        print(
            f"Control: "
            f"{result['control_value']}"
        )

        print(
            f"Treatment: "
            f"{result['treatment_value']}"
        )

        print(
            f"Difference: "
            f"{result['difference']}"
        )

        print(
            f"Expected Direction: "
            f"{result['expected_direction']}"
        )

        print(
            f"Observed Direction: "
            f"{result['observed_direction']}"
        )

        print(
            f"Result: "
            f"{result['result']}"
        )

    # ========================================================
    # Step 5
    # Summary
    # ========================================================

    summary = build_summary(
        results
    )

    # ========================================================
    # Step 6
    # 构造 20.15 Output Contract
    # ========================================================

    output = {
        "evaluation_stage":
            "20.15",

        "evaluation_name":
            "Experiment Result Analysis",

        "summary":
            summary,

        "experiments":
            results
    }

    # ========================================================
    # Step 7
    # 保存 JSON
    # ========================================================

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            output,
            file,
            ensure_ascii=False,
            indent=4
        )

    # ========================================================
    # Step 8
    # 最终输出
    # ========================================================

    print()

    print(
        "=" * 70
    )

    print(
        "20.15 ANALYSIS FINISHED"
    )

    print(
        "=" * 70
    )

    print()

    print(
        f"Total Experiments: "
        f"{summary['total_experiments']}"
    )

    print(
        f"Supported: "
        f"{summary['supported']}"
    )

    print(
        f"Rejected: "
        f"{summary['rejected']}"
    )

    print(
        f"Inconclusive: "
        f"{summary['inconclusive']}"
    )

    print(
        f"Not Experimentable: "
        f"{summary['not_experimentable']}"
    )

    print()

    print(
        "Output saved to:"
    )

    print(
        OUTPUT_FILE
    )


# ============================================================
# 18. 程序入口
# ============================================================

if __name__ == "__main__":
    main()