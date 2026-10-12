import json                        ##这段代码的作用具体执行实验，会用到20.13json里面的实验配置，测试集，agentIC.rag（19），judge llm
import importlib.util
from pathlib import Path


# ============================================================
# 1. 文件路径配置
#
# 当前真实目录结构：
#
# D:\MyRAG\
# ├─ agent\
# │   ├─ agentic_rag_20_14_experiment_execution.py
# │   ├─ agentic_rag_20_13_experiment_design.py
# │   ├─ agentic_rag_20_3_dataset.py
# │   ├─ agentic_rag_20_7_llm_judge_comparison.py
# │   └─ agentic_rag_19_architecture.py
# │
# ├─ agentic_rag_20_13_experiment_design.json
# └─ agentic_rag_20_14_experiment_execution.json
#
# 因此：
# BASE_DIR = D:\MyRAG\agent
# ROOT_DIR = D:\MyRAG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent          ##Path(__file__)路径化为相对路径；resolve()变为绝对路径；parent变为上级目录路径
ROOT_DIR = BASE_DIR.parent


DESIGN_FILE = (
    ROOT_DIR /
    "agentic_rag_20_13_experiment_design.json"
)

DATASET_FILE = (
    BASE_DIR /                ##路径中间可以通过/链接
    "agentic_rag_20_3_dataset.py"
)

RUNTIME_FILE = (
    BASE_DIR /
    "agentic_rag_19_architecture.py"
)

JUDGE_FILE = (
    BASE_DIR /
    "agentic_rag_20_7_llm_judge_comparison.py"
)

OUTPUT_FILE = (
    ROOT_DIR /
    "agentic_rag_20_14_experiment_execution.json"
)


# ============================================================
# 2. Experiment Design Variable
#
# 20.13 使用的是实验层名称：
#
#     retrieval_limit
#     repair_limit
#
# 19.x Runtime 实际使用的是：
#
#     MAX_RETRIEVALS
#     MAX_ANSWER_REPAIRS
#
# 20.14 是 Experiment Design -> Runtime 的执行边界，
# 因此这里显式建立映射。
# ============================================================

INTERVENTION_TO_RUNTIME_VARIABLE = {
    "retrieval_limit": "MAX_RETRIEVALS",
    "repair_limit": "MAX_ANSWER_REPAIRS",
}


# ============================================================
# 3. 动态加载 Python 模块
# ============================================================

def load_module(
    file_path: Path,
    module_name: str
):
    if not file_path.exists():
        raise FileNotFoundError(
            f"找不到文件：{file_path}"
        )

    spec = importlib.util.spec_from_file_location(               ###导入库.工具模块.规格来自文件路径          作用创建规格加载计划书
        module_name,
        file_path
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"无法加载模块：{file_path}"
        )

    module = importlib.util.module_from_spec(                  ##加载模块对象
        spec
    )

    spec.loader.exec_module(                              ##把路径内的变量函数类...真正放进模块中实现可调用
        module
    )

    return module                             ##返回加载完成可实际调用的模块对象


# ============================================================
# 4. 读取 20.13 Experiment Design
# ============================================================

def load_experiment_design():

    if not DESIGN_FILE.exists():
        raise FileNotFoundError(
            f"找不到 20.13 实验设计文件：{DESIGN_FILE}"
        )

    with open(
        DESIGN_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        design = json.load(file)

    if design.get(
        "evaluation_stage"
    ) != "20.13":

        raise ValueError(
            "输入文件不是 20.13 Experiment Design。"
        )

    if "experiments" not in design:
        raise ValueError(
            "20.13 文件缺少 experiments 字段。"
        )

    return design


# ============================================================
# 5. 读取固定测试集
# ============================================================

def load_test_cases():

    dataset_module = load_module(
        DATASET_FILE,
        "agentic_rag_20_3_dataset_runtime"
    )

    if not hasattr(             ##模块中是否存在，参2属性
        dataset_module,
        "TEST_CASES"
    ):

        raise ValueError(
            "20.3 数据集文件缺少 TEST_CASES。"
        )

    test_cases = dataset_module.TEST_CASES           ##调用导入模块中的变量

    if not isinstance(
        test_cases,
        list
    ):

        raise TypeError(
            "TEST_CASES 必须是 list。"
        )

    normalized = []

    for case in test_cases:

        case_id = case.get(
            "id",
            case.get("case_id")
        )

        if case_id is None:
            raise ValueError(
                "测试集存在缺少 id/case_id 的 Case。"
            )

        if "question" not in case:
            raise ValueError(
                f"{case_id} 缺少 question。"
            )

        if "expected_answer" not in case:
            raise ValueError(
                f"{case_id} 缺少 expected_answer。"
            )

        normalized.append(
            {
                "case_id": str(case_id),
                "question": case["question"],
                "expected_answer": case["expected_answer"]
            }
        )

    if not normalized:           ##正常化的
        raise ValueError(
            "TEST_CASES 为空。"
        )

    return normalized


# ============================================================
# 6. 获取可执行实验
#
# 只执行：
#
#     status == designed
#     experimentability == controllable
#
# 所以当前：
#
#     E001 -> 执行
#     E002 -> 执行
#     E003 -> 跳过
# ============================================================

def get_designed_experiments(
    design
):

    experiments = []

    for experiment in design["experiments"]:

        if experiment.get(
            "status"
        ) != "designed":

            continue

        if experiment.get(
            "experimentability"
        ) != "controllable":

            continue

        required_fields = [
            "experiment_id",
            "hypothesis_id",
            "variable",
            "control",
            "treatment",
            "metric",
            "dataset"
        ]

        for field in required_fields:

            if field not in experiment:

                raise ValueError(
                    f"{experiment.get('experiment_id', 'UNKNOWN')} "
                    f"缺少字段：{field}"
                )

        variable = experiment["variable"]

        required_variable_fields = [
            "name",
            "control_value",
            "treatment_value"
        ]

        for field in required_variable_fields:

            if field not in variable:

                raise ValueError(
                    f"{experiment['experiment_id']} "
                    f"的 variable 缺少 {field}。"
                )

        if experiment["dataset"].get(
            "same_for_control_and_treatment"
        ) is not True:

            raise ValueError(
                f"{experiment['experiment_id']} "
                "没有要求 Control / Treatment 使用相同数据集。"
            )

        experiments.append(
            experiment
        )

    return experiments


# ============================================================
# 7. 获取 Runtime 中真正需要修改的变量
# ============================================================

def resolve_runtime_variable(
    experiment_variable_name
):

    if experiment_variable_name not in (
        INTERVENTION_TO_RUNTIME_VARIABLE
    ):

        raise ValueError(
            "20.13 中存在未定义 Runtime 映射的 "
            f"Intervention Variable："
            f"{experiment_variable_name}"
        )

    return INTERVENTION_TO_RUNTIME_VARIABLE[
        experiment_variable_name
    ]


# ============================================================
# 8. LLM Judge
#
# 每一次 Control / Treatment 的 Runtime 运行，
# 都重新调用 20.7 的 llm_judge。
#
# 不复用历史 gold_label。
# ============================================================

def judge_answer(
    judge_module,
    question,
    expected_answer,
    actual_answer
):

    result = judge_module.llm_judge(
        question=question,
        expected_answer=expected_answer,
        actual_answer=actual_answer
    )

    if not isinstance(
        result,
        dict
    ):

        raise TypeError(
            "20.7 LLM Judge 返回值必须是 dict。"
        )

    if "correct" not in result:
        raise ValueError(
            "LLM Judge 返回结果缺少 correct。"
        )

    if not isinstance(
        result["correct"],
        bool
    ):

        raise ValueError(
            "LLM Judge 的 correct 必须是 bool。"
        )

    if "reason" not in result:
        raise ValueError(
            "LLM Judge 返回结果缺少 reason。"
        )

    return result


# ============================================================
# 9. 运行单个 Case
# ============================================================

def run_case(
    runtime,
    judge_module,
    test_case
):

    case_id = test_case["case_id"]
    question = test_case["question"]
    expected_answer = test_case["expected_answer"]

    print()
    print("-" * 70)
    print(f"Case: {case_id}")
    print(f"Question: {question}")

    # --------------------------------------------------------
    # 运行 19.x Agent Runtime
    # --------------------------------------------------------

    runtime_result = runtime.agentic_rag(           ##模块变量人类名.具体函数名
        question
    )

    if not isinstance(
        runtime_result,
        dict
    ):

        raise TypeError(
            f"{case_id} 的 Runtime 返回值必须是 dict。"
        )

    actual_answer = runtime_result.get(
        "answer"
    )

    if actual_answer is None:
        actual_answer = ""

    # --------------------------------------------------------
    # 重新调用 20.7 LLM Judge
    # --------------------------------------------------------

    judge_result = judge_answer(
        judge_module=judge_module,
        question=question,
        expected_answer=expected_answer,
        actual_answer=actual_answer
    )

    answer_correct = judge_result["correct"]

    print(
        f"Actual Answer: {actual_answer}"
    )

    print(
        f"LLM Judge: {answer_correct}"
    )

    print(
        f"LLM Reason: {judge_result['reason']}"
    )

    return {
        "case_id": case_id,

        "question": question,

        "expected_answer": expected_answer,

        "actual_answer": actual_answer,

        "answer_correct": answer_correct,

        "llm_reason": judge_result["reason"],

        "evidence_sufficient":
            runtime_result.get(
                "evidence_sufficient",
                False
            ),

        "citation_valid":
            runtime_result.get(
                "citation_valid"
            ),

        "retrieval_count":
            runtime_result.get(
                "retrieval_count",
                0
            ),

        "repair_count":
            runtime_result.get(
                "repair_count",
                0
            ),

        "status":
            runtime_result.get(
                "status",
                "unknown"
            ),

        "evidence_ids":
            runtime_result.get(
                "evidence_ids",
                []
            ),

        "trace":
            runtime_result.get(
                "trace",
                []
            )
    }


# ============================================================
# 10. 20.9 Failure Attribution
#
# 固定规则：
#
# answer_correct == True
#     -> 无 failure
#
# evidence_sufficient == False
#     -> evidence_failure
#
# evidence_sufficient == True
# citation_valid == False
#     -> repair_failure
#
# evidence_sufficient == True
# citation_valid == True
# answer_correct == False
#     -> grounding_failure
#
# 其他
#     -> unknown_failure
#
# retrieval_count / repair_count
# 不参与 failure_type 判断。
# ============================================================

def classify_failure(
    result
):

    if result["answer_correct"]:
        return None

    if not result["evidence_sufficient"]:
        return "evidence_failure"

    if (
        result["evidence_sufficient"]
        and result["citation_valid"] is False
    ):

        return "repair_failure"

    if (
        result["evidence_sufficient"]
        and result["citation_valid"] is True
        and not result["answer_correct"]
    ):

        return "grounding_failure"

    return "unknown_failure"


# ============================================================
# 11. 计算实验指标
#
# 20.13：
#
# grounding_failure_rate
# =
# grounding_failure_cases / total_cases
# ============================================================

def calculate_metrics(
    results
):

    total_cases = len(
        results
    )

    grounding_failure_cases = 0

    for result in results:

        failure_type = classify_failure(
            result
        )

        if failure_type == "grounding_failure":

            grounding_failure_cases += 1

    if total_cases == 0:

        grounding_failure_rate = 0.0

    else:

        grounding_failure_rate = (
            grounding_failure_cases
            /
            total_cases
        )

    return {
        "total_cases":
            total_cases,

        "grounding_failure_cases":
            grounding_failure_cases,

        "grounding_failure_rate":
            grounding_failure_rate
    }


# ============================================================
# 12. 执行一个 Experiment
# ============================================================

def execute_experiment(
    experiment,
    runtime,
    judge_module,
    test_cases
):

    experiment_id = experiment[
        "experiment_id"
    ]

    intervention_variable = experiment[
        "variable"
    ]["name"]

    runtime_variable = resolve_runtime_variable(
        intervention_variable
    )

    control_value = experiment[
        "variable"
    ]["control_value"]

    treatment_value = experiment[
        "variable"
    ]["treatment_value"]

    print()
    print("=" * 70)
    print(
        f"EXPERIMENT {experiment_id}"
    )
    print("=" * 70)

    print(
        f"Intervention Variable: "
        f"{intervention_variable}"
    )

    print(
        f"Runtime Variable: "
        f"{runtime_variable}"
    )

    print(
        f"Control Value: "
        f"{control_value}"
    )

    print(
        f"Treatment Value: "
        f"{treatment_value}"
    )

    # --------------------------------------------------------
    # 保存 Runtime 原始值
    # --------------------------------------------------------

    if not hasattr(
        runtime,
        runtime_variable
    ):

        raise AttributeError(
            f"Runtime 中不存在变量："
            f"{runtime_variable}"
        )

    original_value = getattr(
        runtime,
        runtime_variable
    )

    control_results = []
    treatment_results = []

    try:

        # ====================================================
        # Control
        # ====================================================

        print()
        print("=" * 70)
        print("CONTROL")
        print("=" * 70)

        setattr(
            runtime,
            runtime_variable,
            control_value
        )

        for test_case in test_cases:

            result = run_case(
                runtime=runtime,
                judge_module=judge_module,
                test_case=test_case
            )

            control_results.append(
                result
            )

        control_metrics = calculate_metrics(
            control_results
        )

        print()
        print(
            "Control grounding_failure_rate = "
            f"{control_metrics['grounding_failure_rate']:.4f}"
        )

        # ====================================================
        # Treatment
        # ====================================================

        print()
        print("=" * 70)
        print("TREATMENT")
        print("=" * 70)

        setattr(                 ##修改对象属性值
            runtime,            ##对象名
            runtime_variable,    ##属性名
            treatment_value       ##新属性值
        )

        for test_case in test_cases:

            result = run_case(
                runtime=runtime,
                judge_module=judge_module,
                test_case=test_case
            )

            treatment_results.append(
                result
            )

        treatment_metrics = calculate_metrics(
            treatment_results
        )

        print()
        print(
            "Treatment grounding_failure_rate = "
            f"{treatment_metrics['grounding_failure_rate']:.4f}"
        )

    finally:             ###try运行失败不影响finally的正常执行，用来恢复对象属性，防止try运行失败属性被永久修改，兜底

        # ----------------------------------------------------
        # 无论实验是否成功，都恢复 Runtime 原始配置
        # ----------------------------------------------------

        setattr(
            runtime,
            runtime_variable,
            original_value
        )

        print()
        print(
            f"Runtime {runtime_variable} "
            f"已恢复为：{original_value}"
        )

    # ========================================================
    # Control / Treatment 比较
    # ========================================================

    control_rate = control_metrics[
        "grounding_failure_rate"
    ]

    treatment_rate = treatment_metrics[
        "grounding_failure_rate"
    ]

    difference = (
        treatment_rate
        -
        control_rate
    )

    # --------------------------------------------------------
    # 根据 20.13 Expected Outcome 判断
    #
    # hypothesis supported：
    # treatment > control
    #
    # hypothesis rejected：
    # treatment <= control
    # --------------------------------------------------------

    if treatment_rate > control_rate:

        hypothesis_result = "supported"

    else:

        hypothesis_result = "rejected"

    return {
        "experiment_id":
            experiment_id,

        "hypothesis_id":
            experiment["hypothesis_id"],

        "intervention_variable":
            intervention_variable,

        "control_value":
            control_value,

        "treatment_value":
            treatment_value,

        "metric":
            experiment["metric"],

        "control": {
            "results":
                control_results,

            "metrics":
                control_metrics
        },

        "treatment": {
            "results":
                treatment_results,

            "metrics":
                treatment_metrics
        },

        "comparison": {
            "control_grounding_failure_rate":
                control_rate,

            "treatment_grounding_failure_rate":
                treatment_rate,

            "difference":
                difference
        },

        "hypothesis_result":
            hypothesis_result,

        "status":
            "executed"
    }


# ============================================================
# 13. Main
# ============================================================

def main():

    print("=" * 70)
    print("20.14 EXPERIMENT EXECUTION")
    print("=" * 70)

    print()
    print(
        f"Project Root: {ROOT_DIR}"
    )

    print(
        f"Agent Directory: {BASE_DIR}"
    )

    # --------------------------------------------------------
    # Step 1：读取 20.13
    # --------------------------------------------------------

    design = load_experiment_design()

    print()
    print(
        f"Loaded Design: {DESIGN_FILE}"
    )

    # --------------------------------------------------------
    # Step 2：读取固定测试集
    # --------------------------------------------------------

    test_cases = load_test_cases()

    print(
        f"Dataset cases: {len(test_cases)}"
    )

    # --------------------------------------------------------
    # Step 3：获取可执行实验
    # --------------------------------------------------------

    experiments = get_designed_experiments(
        design
    )

    if not experiments:

        raise ValueError(
            "20.13 中没有可执行的 "
            "designed + controllable 实验。"
        )

    print(
        f"Executable experiments: "
        f"{len(experiments)}"
    )

    # --------------------------------------------------------
    # Step 4：加载 Runtime
    # --------------------------------------------------------

    runtime = load_module(
        RUNTIME_FILE,
        "agentic_rag_19_runtime_for_20_14"
    )

    if not hasattr(
        runtime,
        "agentic_rag"
    ):

        raise AttributeError(
            "Runtime 中不存在 agentic_rag。"
        )

    print(
        f"Loaded Runtime: {RUNTIME_FILE}"
    )

    # --------------------------------------------------------
    # Step 5：检查所有实验变量映射
    # --------------------------------------------------------

    for experiment in experiments:

        intervention_variable = experiment[
            "variable"
        ]["name"]

        runtime_variable = resolve_runtime_variable(
            intervention_variable
        )

        if not hasattr(
            runtime,
            runtime_variable
        ):

            raise AttributeError(
                f"{experiment['experiment_id']} "
                f"需要 Runtime 变量 "
                f"{runtime_variable}，"
                "但该变量不存在。"
            )

        print(
            f"{experiment['experiment_id']}: "
            f"{intervention_variable} "
            f"-> "
            f"{runtime_variable}"
        )

    # --------------------------------------------------------
    # Step 6：加载 20.7 LLM Judge
    # --------------------------------------------------------

    judge_module = load_module(
        JUDGE_FILE,
        "agentic_rag_20_7_judge_for_20_14"
    )

    if not hasattr(
        judge_module,
        "llm_judge"
    ):

        raise AttributeError(
            "20.7 文件中不存在 llm_judge。"
        )

    print(
        f"Loaded LLM Judge: {JUDGE_FILE}"
    )

    # --------------------------------------------------------
    # Step 7：执行实验
    # --------------------------------------------------------

    experiment_results = []

    for experiment in experiments:

        result = execute_experiment(
            experiment=experiment,
            runtime=runtime,
            judge_module=judge_module,
            test_cases=test_cases
        )

        experiment_results.append(
            result
        )

    # --------------------------------------------------------
    # Step 8：生成 20.14 输出
    # --------------------------------------------------------

    output = {
        "evaluation_stage":
            "20.14",

        "evaluation_name":
            "Experiment Execution",

        "summary": {
            "designed_experiments":
                len(experiments),

            "executed_experiments":
                len(experiment_results),

            "dataset_cases":
                len(test_cases)
        },

        "experiments":
            experiment_results
    }

    # --------------------------------------------------------
    # Step 9：保存 JSON
    # --------------------------------------------------------

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

    print()
    print("=" * 70)
    print("20.14 EXECUTION FINISHED")
    print("=" * 70)

    print(
        f"Results saved to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()
