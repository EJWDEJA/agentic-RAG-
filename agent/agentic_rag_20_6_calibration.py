import os                    ###这个文件是用来设计评估评估器的评估器，目的是判断评估气哭判断answer正确性的能力怎么样，顺便看看不同阈值下评估器的效果选择阈值（此处以语义相似度评估期为例子）
import json                   ###也就是当前阿根廷运行状态的各种指标提取
import math
import numpy as np


# ============================================================
# 1. Environment
# ============================================================

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


# ============================================================
# 2. Imports
# ============================================================

from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaEmbeddings

from agentic_rag_19_architecture import (
    agentic_rag
)

from agentic_rag_20_3_dataset import (
    TEST_CASES
)


# ============================================================
# 3. LLM Judge
# ============================================================

llm = ChatOpenAI(
    model="deepseek-v4-flash",
    api_key=os.environ[
        "ANTHROPIC_AUTH_TOKEN"
    ],
    base_url="https://api.deepseek.com"
)


# ============================================================
# 4. Embedding Model
# ============================================================

embeddings = OllamaEmbeddings(
    model="qwen3-embedding:latest"
)


# ============================================================
# 5. IMPORTANT
#
# 人工 Gold Label
#
# True  = 实际答案正确
# False = 实际答案错误
#
# 必须根据实际运行结果人工填写。
# ============================================================

GOLD_LABELS = {

    
    "case_01": True,
    "case_02": False,
    "case_03": False,

}


# ============================================================
# 6. Thresholds
# ============================================================

THRESHOLDS = [
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95,
]


# ============================================================
# 7. Cosine Similarity
# ============================================================

def cosine_similarity(
    vector_a,
    vector_b
):

    vector_a = np.array(
        vector_a,
        dtype=float
    )

    vector_b = np.array(
        vector_b,
        dtype=float
    )


    norm_a = np.linalg.norm(
        vector_a
    )

    norm_b = np.linalg.norm(
        vector_b
    )


    if norm_a == 0 or norm_b == 0:

        return 0.0


    return float(
        np.dot(
            vector_a,
            vector_b
        )
        /
        (
            norm_a * norm_b
        )
    )


# ============================================================
# 8. Embedding Evaluation
# ============================================================

def calculate_similarity(
    expected_answer,
    actual_answer
):

    expected_vector = embeddings.embed_query(
        expected_answer
    )

    actual_vector = embeddings.embed_query(
        actual_answer
    )


    similarity = cosine_similarity(
        expected_vector,
        actual_vector
    )


    return similarity


# ============================================================
# 9. Run Agent Once
#
# 非常重要：
#
# Calibration 不应该为了测试不同 threshold
# 反复运行 Agent。
#
# Agent 只运行一次。
#
# 后面不同 threshold 都使用同一批结果。
# ============================================================

def collect_agent_results():

    results = []


    print(
        "\n"
        + "=" * 70
    )

    print(
        "STEP 1: RUN AGENT"
    )

    print(
        "=" * 70
    )


    for test_case in TEST_CASES:

        case_id = test_case[
            "id"
        ]

        question = test_case[
            "question"
        ]

        expected_answer = test_case[
            "expected_answer"
        ]


        print(
            f"\nRunning {case_id}"
        )

        print(
            f"Question: {question}"
        )


        # ----------------------------------------------------
        # Run Agent
        # ----------------------------------------------------

        result = agentic_rag(
            question
        )


        actual_answer = result.get(
            "answer",
            ""
        )


        # ----------------------------------------------------
        # Gold Label Check
        # ----------------------------------------------------

        if case_id not in GOLD_LABELS:

            raise ValueError(
                f"\nMissing GOLD_LABEL "
                f"for {case_id}\n\n"
                f"Please manually judge the actual "
                f"answer and add:\n\n"
                f'"{case_id}": True\n'
                f'or\n'
                f'"{case_id}": False'
            )


        gold_label = GOLD_LABELS[
            case_id
        ]


        # ----------------------------------------------------
        # Embedding Similarity
        # ----------------------------------------------------

        similarity = calculate_similarity(
            expected_answer,
            actual_answer
        )


        print(
            f"Actual Answer: "
            f"{actual_answer}"
        )


        print(
            f"Similarity: "
            f"{similarity:.4f}"
        )


        print(
            f"Gold Label: "
            f"{gold_label}"
        )


        results.append({

            "id":
                case_id,

            "question":
                question,

            "expected_answer":
                expected_answer,

            "actual_answer":
                actual_answer,

            "similarity":
                similarity,

            "gold_label":
                gold_label,

            "status":
                result.get(
                    "status",
                    "unknown"
                ),

            "evidence_sufficient":
                result.get(
                    "evidence_sufficient",
                    False
                ),

            "citation_valid":
                result.get(
                    "citation_valid",
                    False
                ),

            "retrieval_count":
                result.get(
                    "retrieval_count",
                    0
                ),

            "repair_count":
                result.get(
                    "repair_count",
                    0
                ),

            "trace":
                result.get(
                    "trace",
                    []
                )

        })


    return results


# ============================================================
# 10. Confusion Matrix
# ============================================================

def calculate_confusion_matrix(
    results,
    threshold
):

    tp = 0
    tn = 0
    fp = 0
    fn = 0


    for result in results:

        gold = result[
            "gold_label"
        ]

        predicted = (
            result["similarity"]
            >= threshold                  ###结果直接是true或者flash
        )


        if gold and predicted:

            tp += 1


        elif not gold and not predicted:

            tn += 1


        elif not gold and predicted:

            fp += 1


        elif gold and not predicted:

            fn += 1


    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn
    }


# ============================================================
# 11. Metrics
# ============================================================

def calculate_metrics(
    confusion
):

    tp = confusion[
        "tp"
    ]

    tn = confusion[
        "tn"
    ]

    fp = confusion[
        "fp"
    ]

    fn = confusion[
        "fn"
    ]


    total = (
        tp
        + tn
        + fp
        + fn
    )


    # --------------------------------------------------------
    # Accuracy     ###准确率，评估器判的对不对（不论正反）
    # --------------------------------------------------------

    if total == 0:

        accuracy = 0.0

    else:

        accuracy = (
            tp + tn
        ) / total


    # --------------------------------------------------------
    # Precision     #精确率，评估器判段正确的里面到底有多少真的正确，判对的可信度
    # --------------------------------------------------------

    if (
        tp + fp
        == 0
    ):

        precision = 0.0

    else:

        precision = (
            tp
            /
            (tp + fp)
        )


    # --------------------------------------------------------
    # Recall             ##真真正确的答案有多少被评估器判断出来了
    # --------------------------------------------------------

    if (
        tp + fn
        == 0
    ):

        recall = 0.0

    else:

        recall = (
            tp
            /
            (tp + fn)
        )


    # --------------------------------------------------------
    # F1      综合precision + recall
    # --------------------------------------------------------

    if (
        precision + recall
        == 0
    ):

        f1 = 0.0

    else:

        f1 = (
            2
            * precision
            * recall
            /
            (
                precision
                + recall
            )
        )


    return {

        "accuracy":
            accuracy,

        "precision":
            precision,

        "recall":
            recall,

        "f1":
            f1
    }


# ============================================================
# 12. Threshold Calibration
# ============================================================

def calibrate_thresholds(
    results
):

    calibration_results = []


    print(
        "\n\n"
        + "=" * 70
    )

    print(
        "STEP 2: THRESHOLD CALIBRATION"
    )

    print(
        "=" * 70
    )


    for threshold in THRESHOLDS:

        confusion = calculate_confusion_matrix(
            results,
            threshold
        )


        metrics = calculate_metrics(
            confusion
        )


        row = {

            "threshold":
                threshold,

            **confusion,

            **metrics

        }


        calibration_results.append(
            row
        )


    return calibration_results


# ============================================================
# 13. Print Calibration Report
# ============================================================

def print_calibration_report(
    calibration_results
):

    print(
        "\n"
        + "=" * 70
    )

    print(
        "THRESHOLD CALIBRATION REPORT"
    )

    print(
        "=" * 70
    )


    print(
        "\n"
        "Threshold | Accuracy | Precision | Recall | F1"
    )

    print(
        "-" * 70
    )


    for result in calibration_results:

        print(
            f"{result['threshold']:<9.2f} | "
            f"{result['accuracy']:<8.2%} | "
            f"{result['precision']:<9.2%} | "
            f"{result['recall']:<6.2%} | "
            f"{result['f1']:<6.2%}"
        )


# ============================================================
# 14. Detailed Case Report
# ============================================================

def print_case_report(
    results
):

    print(
        "\n\n"
        + "=" * 70
    )

    print(
        "CASE-LEVEL CALIBRATION DATA"
    )

    print(
        "=" * 70
    )


    for result in results:

        print(
            f"\n[{result['id']}]"
        )


        print(
            f"Question:\n"
            f"{result['question']}"
        )


        print(
            f"\nExpected:\n"
            f"{result['expected_answer']}"
        )


        print(
            f"\nActual:\n"
            f"{result['actual_answer']}"
        )


        print(
            f"\nSimilarity: "
            f"{result['similarity']:.4f}"
        )


        print(
            f"Gold Label: "
            f"{result['gold_label']}"
        )


        print(
            f"Status: "
            f"{result['status']}"
        )


        print(
            f"Evidence Sufficient: "
            f"{result['evidence_sufficient']}"
        )


        print(
            f"Citation Valid: "
            f"{result['citation_valid']}"
        )


        print(
            f"Retrieval Count: "
            f"{result['retrieval_count']}"
        )


        print(
            f"Repair Count: "
            f"{result['repair_count']}"
        )


# ============================================================
# 15. Save Calibration Results
# ============================================================

def save_results(            ##这一步是创立文件保存数据，没有文件创立文件，已有文件，替换文件内部内容
    results,
    calibration_results
):

    output = {

        "case_results":
            results,

        "calibration":
            calibration_results

    }


    with open(
        "agentic_rag_20_6_calibration_results.json",
        "w",
        encoding="utf-8"      ##采用utf 8编码，编号变字节
    ) as f:

        json.dump(
            output,
            f,
            ensure_ascii=False,        #json中非ASCII编码的字符直接表示成原字符不用unicode转义（汉字）
            indent=4            ##缩进四格，让保存的数据更加规整
        )


    print(
        "\n"
        "Calibration results saved to:"
    )

    print(
        "agentic_rag_20_6_calibration_results.json"
    )


# ============================================================
# 16. Main
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Step 1
    # Run Agent
    # --------------------------------------------------------

    results = collect_agent_results()


    # --------------------------------------------------------
    # Step 2
    # Threshold Calibration
    # --------------------------------------------------------

    calibration_results = calibrate_thresholds(
        results
    )


    # --------------------------------------------------------
    # Step 3
    # Reports
    # --------------------------------------------------------

    print_case_report(
        results
    )


    print_calibration_report(
        calibration_results
    )


    # --------------------------------------------------------
    # Step 4
    # Save
    # --------------------------------------------------------

    save_results(
        results,
        calibration_results
    )