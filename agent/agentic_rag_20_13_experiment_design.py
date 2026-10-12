import json                        ###这个测试文件的目的是验证，一种失败原因下的三种变量（最大检索，最大修复数，状态）是否可以被实验设计即能不能被人为控制（最大检索和最大修复数可以）以及具体的实验组对照组变量怎么设计
from pathlib import Path
###“改什么、对照组是什么、实验组是什么、用什么指标判断”。拿同一批数据，Control 按什么配置跑，Treatment 改什么配置跑，最后比较什么指标，以及什么结果算支持/否定假设。”

# ============================================================
# 1. 文件配置
# ============================================================

INPUT_FILE = Path(
    "agentic_rag_20_12_hypothesis.json"
)

OUTPUT_FILE = Path(
    "agentic_rag_20_13_experiment_design.json"
)


# ============================================================
# 2. 当前 Runtime 的真实控制边界              当前系统允许控制谁，控制到什么程度
# ============================================================

# 当前 19.x Runtime：
#
# MAX_RETRIEVALS = 3
# MAX_ANSWER_REPAIRS = 2
#
# 这里不是重新定义 Runtime。
# 只是 Experiment Design 需要知道：
# “当前系统允许控制到什么范围”。

INTERVENTION_CONFIG = {

    "retrieval_count": {

        "variable_name":
            "retrieval_limit",

        "control_value":
            3,

        "min_value":
            1,

        "max_value":
            3,
    },

    "repair_count": {

        "variable_name":
            "repair_limit",

        "control_value":
            2,

        "min_value":
            0,

        "max_value":
            2,
    },
}


# ============================================================
# 3. 指标配置
# ============================================================

FAILURE_METRICS = {

    "evidence_failure":
        "evidence_failure_rate",

    "repair_failure":
        "repair_failure_rate",

    "grounding_failure":
        "grounding_failure_rate",

    "unknown_failure":
        "unknown_failure_rate",
}


# ============================================================
# 4. 读取 JSON
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
# 5. 验证 Hypothesis            字段名和状态值校验
# ============================================================

def validate_hypothesis(hypothesis):
    """
    验证 Hypothesis 是否具备基本结构。

    注意：
    这里不要求 Hypothesis 必须是 strong。

    weak / exploratory Hypothesis
    也可以进入 Experiment Design。
    """

    required_fields = [
        "hypothesis_id",
        "failure_type",
        "dimension",
        "condition",
        "status",
    ]

    missing_fields = [
        field
        for field in required_fields
        if field not in hypothesis
    ]

    if missing_fields:

        raise ValueError(
            f"Hypothesis "
            f"{hypothesis.get('hypothesis_id')} "
            f"缺少字段："
            f"{missing_fields}"
        )

    if hypothesis["status"] != "unverified":

        raise ValueError(
            f"Hypothesis "
            f"{hypothesis['hypothesis_id']} "
            f"当前状态为 "
            f"{hypothesis['status']}，"
            f"只能对 unverified Hypothesis "
            f"设计新实验。"
        )


# ============================================================
# 6. 判断是否可以进行干预实验         ##校验三个变量那些可控
# ============================================================

def determine_experimentability(
    hypothesis
):
    """
    判断 Hypothesis 对应的 Dimension
    是否存在可控 Intervention Variable。

    例如：

    retrieval_count
        → retrieval_limit
        → 可实验

    repair_count
        → repair_limit
        → 可实验

    status
        → 没有直接对应的控制变量
        → 不可直接实验
    """

    dimension = hypothesis[
        "dimension"
    ]

    if dimension not in INTERVENTION_CONFIG:

        return {
            "experimentable":
                False,

            "reason":
                (
                    f"{dimension} 是 Runtime 的"
                    f"观察结果，当前没有对应的"
                    f"直接 Intervention Variable。"
                ),
        }

    return {
        "experimentable":
            True,

        "reason":
            "存在对应的可控 Runtime 参数。",
    }


# ============================================================
# 7. 获取实验变量             
# ============================================================

def build_variable(              ##此处用于生成treatment variable,提取INTERVENTION_CONFIG 中对应元素的config中的control variable           z注意这两个和20.12内部的观测变量不一样属于重新定义的只是实验值会取"condition": "retrieval_count == 1"作为新的最大检索次数
    hypothesis
):
    """
    将：

        Observed Variable
            ↓
        Intervention Variable

    进行映射。
    """

    dimension = hypothesis[
        "dimension"
    ]

    config = INTERVENTION_CONFIG[
        dimension
    ]

    observed_value = hypothesis.get(
        "condition"
    )

    # --------------------------------------------------------
    # 当前第一版要求 Pattern 的 value
    # 可以从 condition 中得到。
    #
    # 例如：
    #
    # retrieval_count == 1
    #
    # repair_count == 0
    # --------------------------------------------------------

    if "==" not in observed_value:

        raise ValueError(
            f"Hypothesis "
            f"{hypothesis['hypothesis_id']} "
            f"的 condition 无法转换成实验值："
            f"{observed_value}"
        )

    raw_value = (
        observed_value
        .split("==", 1)[1]
        .strip()
        .strip('"')
        .strip("'")
    )

    try:

        treatment_value = int(
            raw_value
        )

    except ValueError:

        raise ValueError(
            f"Hypothesis "
            f"{hypothesis['hypothesis_id']} "
            f"的实验值必须是整数，"
            f"当前值：{raw_value}"
        )

    # --------------------------------------------------------
    # 检查实验值是否处于 Runtime 合法范围
    # --------------------------------------------------------

    if not (
        config["min_value"]
        <= treatment_value
        <= config["max_value"]
    ):

        raise ValueError(
            f"Hypothesis "
            f"{hypothesis['hypothesis_id']} "
            f"的 treatment_value="
            f"{treatment_value} "
            f"超出 Runtime 合法范围："
            f"{config['min_value']}"
            f"~"
            f"{config['max_value']}"
        )

    # --------------------------------------------------------
    # 如果 treatment 和 control 完全一样，
    # 那么这个实验没有干预差异。
    # --------------------------------------------------------

    informative = (
        treatment_value
        != config["control_value"]
    )

    return {

        "name":
            config["variable_name"],

        "observed_dimension":
            dimension,

        "observed_value":
            treatment_value,

        "control_value":
            config["control_value"],

        "treatment_value":
            treatment_value,

        "informative":
            informative,
    }


# ============================================================
# 8. 构造 Objective
# ============================================================

def build_objective(
    hypothesis
):
    """
    生成实验目标。
    """

    failure_type = (
        hypothesis["failure_type"]
    )

    dimension = (
        hypothesis["dimension"]
    )

    condition = (
        hypothesis["condition"]
    )

    return (
        f"验证 {dimension}={condition.split('==')[-1].strip()} "
        f"这一运行行为是否与 "
        f"{failure_type} 的发生率存在关联。"
    )


# ============================================================
# 9. 构造 Control
# ============================================================

def build_control(
    variable
):
    """
    Control：
    使用当前正常 Runtime 配置。
    """

    return {

        "description":
            "使用当前正常 Agent Runtime 配置。",

        "intervention_variable":
            variable["name"],

        "value":
            variable["control_value"],
    }


# ============================================================
# 10. 构造 Treatment
# ============================================================

def build_treatment(
    variable
):
    """
    Treatment：
    只修改当前实验变量。
    """

    return {

        "description":
            (
                "仅修改实验变量，"
                "其余 Agent 配置保持不变。"
            ),

        "intervention_variable":
            variable["name"],

        "value":
            variable["treatment_value"],
    }


# ============================================================
# 11. 构造 Metric
# ============================================================

def build_metric(                     ###告知失败类型，要研究情况的失败率计算公式，描述
    hypothesis
):
    """
    根据 Failure Type
    确定实验指标。
    """

    failure_type = (
        hypothesis["failure_type"]
    )

    if failure_type not in FAILURE_METRICS:

        raise ValueError(
            f"未知 failure_type："
            f"{failure_type}"
        )

    return {

        "name":
            FAILURE_METRICS[
                failure_type
            ],

        "formula":
            (
                f"{failure_type}_cases "
                f"/ total_cases"
            ),

        "direction":
            "lower_is_better",
    }


# ============================================================
# 12. Dataset Contract
# ============================================================

def build_dataset():
    """
    Control / Treatment
    必须使用同一份 Dataset。
    """

    return {

        "type":
            "fixed",

        "same_for_control_and_treatment":
            True,

        "description":
            (
                "Control 与 Treatment "
                "必须使用完全相同的问题集合。"
            ),
    }


# ============================================================
# 13. Experiment Constraints
# ============================================================

def build_constraints(
    variable
):
    """
    明确实验中只有一个主要变量可以改变。
    """

    return [

        "question dataset must remain unchanged",

        "LLM configuration must remain unchanged",

        "embedding configuration must remain unchanged",

        "retrieval pipeline must remain unchanged",

        "reranker configuration must remain unchanged",

        "answer generation configuration must remain unchanged",

        "evaluation criteria must remain unchanged",

        (
            "only "
            + variable["name"]
            + " may change"
        ),
    ]


# ============================================================
# 14. Expected Outcome
# ============================================================

def build_expected_outcome(
    hypothesis
):
    """
    在实验执行前定义：
    什么结果支持 / 否定 Hypothesis。
    """

    failure_type = (
        hypothesis["failure_type"]
    )

    metric = FAILURE_METRICS[
        failure_type
    ]

    return {

        "if_hypothesis_supported":
            (
                "treatment "
                + metric
                + " > control "
                + metric
            ),

        "if_hypothesis_rejected":
            (
                "treatment "
                + metric
                + " <= control "
                + metric
            ),
    }


# ============================================================
# 15. 构造 Experiment
# ============================================================

def build_experiment(            ###针对一个假设，假设要一个个处理
    hypothesis,
    experiment_index
):
    """
    Hypothesis
        ↓
    Experimentability
        ↓
    Experiment Contract
    """

    validate_hypothesis(           ##验证字段名和状态值是否合法
        hypothesis
    )

    experimentability = (                 ##一个个看那些变量可控，返回两个值
        determine_experimentability(
            hypothesis
        )
    )

    # --------------------------------------------------------
    # 不可直接干预
    # --------------------------------------------------------

    if not experimentability[
        "experimentable"
    ]:

        return {

            "experiment_id":
                f"E{experiment_index:03d}",    #数字占三位不足补0

            "hypothesis_id":
                hypothesis[
                    "hypothesis_id"
                ],

            "experimentability":
                "not_directly_controllable",

            "status":
                "not_experimentable",

            "reason":
                experimentability[
                    "reason"
                ],
        }

    # --------------------------------------------------------
    # 可以干预
    # --------------------------------------------------------

    variable = build_variable(                                  
        hypothesis
    )

    # --------------------------------------------------------
    # 没有真正的 Control / Treatment 差异
    # --------------------------------------------------------

    if not variable[
        "informative"
    ]:

        return {

            "experiment_id":
                f"E{experiment_index:03d}",

            "hypothesis_id":
                hypothesis[
                    "hypothesis_id"
                ],

            "experimentability":
                "controllable",

            "status":
                "not_informative",

            "reason":
                (
                    "Treatment value 与 "
                    "Control value 相同，"
                    "无法形成有效实验对照。"
                ),

            "variable":
                variable,
        }

    # --------------------------------------------------------
    # 正常 Experiment
    # --------------------------------------------------------

    return {

        "experiment_id":
            f"E{experiment_index:03d}",

        "hypothesis_id":
            hypothesis[
                "hypothesis_id"
            ],

        "experimentability":           ####校验三个变量那些可控
            "controllable",

        "objective":                    ##实验谜底要验证什么假设                 
            build_objective(
                hypothesis
            ),

        "variable":                 ##综合一下下面两个，只是不细
            variable,

        "control":                ##变量描述，变量名，变量值
            build_control(
                variable
            ),

        "treatment":                  ##变量描述，变量名，变量值
            build_treatment(
                variable
            ),

        "metric":                    ##描述越...越...，错误率计算公式，错误率名
            build_metric(
                hypothesis
            ),

        "dataset":                          ###数据集类型（fixed），是否实验对照组公用，描述
            build_dataset(),

        "constraints":                     ##文字化约束条件
            build_constraints(
                variable
            ),

        "expected_outcome":               ##文字化描述，期待的结果即假设支持，假设拒绝
            build_expected_outcome(
                hypothesis
            ),

        "status":
            "designed",
    }


# ============================================================
# 16. Summary
# ============================================================

def build_summary(
    experiments
):
    """
    统计：

    designed
    not_experimentable
    not_informative
    """

    return {

        "total_hypotheses":
            len(experiments),

        "designed_experiments":
            sum(
                1
                for experiment
                in experiments
                if experiment["status"]
                == "designed"
            ),

        "not_experimentable":
            sum(
                1
                for experiment
                in experiments
                if experiment["status"]
                == "not_experimentable"
            ),

        "not_informative":
            sum(
                1
                for experiment
                in experiments
                if experiment["status"]
                == "not_informative"
            ),
    }


# ============================================================
# 17. 主分析
# ============================================================

def analyze():

    report = load_json(
        INPUT_FILE
    )

    hypotheses = report.get(
        "hypotheses",
        []
    )

    experiments = []

    experiment_index = 1

    # --------------------------------------------------------
    # Hypothesis → Experiment
    # --------------------------------------------------------

    for hypothesis in hypotheses:

        experiment = build_experiment(
            hypothesis,
            experiment_index
        )

        experiments.append(
            experiment
        )

        experiment_index += 1

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = build_summary(
        experiments
    )

    # --------------------------------------------------------
    # Final Result
    # --------------------------------------------------------

    result = {

        "evaluation_stage":
            "20.13",

        "evaluation_name":
            "Experiment Design",

        "summary":
            summary,

        "experiments":
            experiments,
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
    print(
        "20.13 Experiment Design"
    )
    print("=" * 60)

    print()

    print(
        f"Total Hypotheses: "
        f"{summary['total_hypotheses']}"
    )

    print(
        f"Designed Experiments: "
        f"{summary['designed_experiments']}"
    )

    print(
        f"Not Experimentable: "
        f"{summary['not_experimentable']}"
    )

    print(
        f"Not Informative: "
        f"{summary['not_informative']}"
    )

    print()

    for experiment in experiments:

        print(
            f"{experiment['experiment_id']} | "
            f"Hypothesis: "
            f"{experiment['hypothesis_id']} | "
            f"Status: "
            f"{experiment['status']}"
        )

        if (
            experiment["status"]
            == "designed"
        ):

            print(
                f"    Variable: "
                f"{experiment['variable']['name']}"
            )

            print(
                f"    Control: "
                f"{experiment['control']['value']}"
            )

            print(
                f"    Treatment: "
                f"{experiment['treatment']['value']}"
            )

        else:

            print(
                f"    Reason: "
                f"{experiment['reason']}"
            )

    print()

    print(
        f"Output: {OUTPUT_FILE}"
    )


# ============================================================
# 18. 程序入口
# ============================================================

if __name__ == "__main__":
    analyze()