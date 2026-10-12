from agentic_rag_19_architecture import agentic_rag                 ###相似度评估器

from agentic_rag_20_3_dataset import TEST_CASES

from langchain_ollama import OllamaEmbeddings

import numpy as np


# ============================================================
# 1. Semantic Similarity Configuration
# ============================================================

SIMILARITY_THRESHOLD = 0.80


# ============================================================
# 2. Embedding Model
# ============================================================

embeddings = OllamaEmbeddings(
    model="qwen3-embedding:latest"
)


# ============================================================
# 3. Cosine Similarity
# ============================================================

def cosine_similarity(
    vector_a,
    vector_b
):
    """
    计算两个向量之间的余弦相似度。

    similarity 越接近 1，
    两个文本在 embedding 空间中越相似。
    """

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


    # 防止零向量导致除零
    if norm_a == 0 or norm_b == 0:

        return 0.0


    similarity = (
        np.dot(
            vector_a,
            vector_b
        )
        /
        (
            norm_a
            *
            norm_b
        )
    )


    return float(
        similarity
    )


# ============================================================
# 4. Semantic Answer Evaluation
# ============================================================

def evaluate_answer(
    actual_answer,
    expected_answer
):
    """
    使用 Embedding + Cosine Similarity
    判断 Agent 最终答案与标准答案
    在语义空间中的相似程度。
    """

    # --------------------------------------------------------
    # 空答案直接判定失败
    # --------------------------------------------------------

    if not actual_answer:

        return {
            "correct": False,
            "similarity": 0.0
        }


    # --------------------------------------------------------
    # Expected Answer → Embedding
    # --------------------------------------------------------

    expected_vector = embeddings.embed_query(
        expected_answer
    )


    # --------------------------------------------------------
    # Actual Answer → Embedding
    # --------------------------------------------------------

    actual_vector = embeddings.embed_query(
        actual_answer
    )


    # --------------------------------------------------------
    # Cosine Similarity
    # --------------------------------------------------------

    similarity = cosine_similarity(
        expected_vector,
        actual_vector
    )


    # --------------------------------------------------------
    # Threshold
    # --------------------------------------------------------

    correct = (
        similarity
        >=
        SIMILARITY_THRESHOLD
    )


    return {

        "correct":
            correct,

        "similarity":
            similarity
    }


# ============================================================
# 5. Workflow Evaluation
# ============================================================

def evaluate_workflow(trace):

    events = [
        item["event"]
        for item in trace
    ]

    violations = []


    # --------------------------------------------------------
    # Rule 1
    # retrieval_failure 必须是最后事件
    # --------------------------------------------------------

    for index, event in enumerate(events):

        if event == "retrieval_failure":

            if index != len(events) - 1:

                violations.append(
                    "retrieval_failure "
                    "must be the final event."
                )


    # --------------------------------------------------------
    # Rule 2
    # safe_fallback 必须是最后事件
    # --------------------------------------------------------

    for index, event in enumerate(events):

        if event == "safe_fallback":

            if index != len(events) - 1:

                violations.append(
                    "safe_fallback "
                    "must be the final event."
                )


    # --------------------------------------------------------
    # Rule 3
    # Evidence Evaluation
    # --------------------------------------------------------

    for index, event in enumerate(events[:-1]):

        if event != "evidence_evaluation":

            continue


        next_event = events[
            index + 1
        ]


        detail = trace[index].get(
            "detail",
            ""
        ).lower()


        # ----------------------------------------------------
        # insufficient
        # ----------------------------------------------------

        if "insufficient" in detail:

            if next_event != "evidence_gap":

                violations.append(
                    "Insufficient evidence "
                    "must be followed by "
                    "evidence_gap."
                )


        # ----------------------------------------------------
        # sufficient
        # ----------------------------------------------------

        elif "sufficient" in detail:

            if next_event != "grounding":

                violations.append(
                    "Sufficient evidence "
                    "must be followed by "
                    "grounding."
                )


    # --------------------------------------------------------
    # Rule 4
    # Citation Validation FAIL
    # --------------------------------------------------------

    for index, event in enumerate(events[:-1]):

        if event != "citation_validation":

            continue


        detail = trace[index].get(
            "detail",
            ""
        ).lower()


        if "fail" in detail:

            next_event = events[
                index + 1
            ]


            if next_event != "failure_classification":

                violations.append(
                    "Citation validation failure "
                    "must be followed by "
                    "failure_classification."
                )


    # --------------------------------------------------------
    # Rule 5
    # Answer Repair
    # --------------------------------------------------------

    for index, event in enumerate(events[:-1]):

        if event != "answer_repair":

            continue


        next_event = events[
            index + 1
        ]


        if next_event not in {
            "citation_validation",
            "safe_fallback"
        }:

            violations.append(
                "answer_repair must be followed "
                "by citation_validation or "
                "safe_fallback."
            )


    return {

        "valid":
            len(violations) == 0,

        "violations":
            violations
    }


# ============================================================
# 6. Evaluate One Test Case
# ============================================================

def evaluate_case(test_case):

    question = test_case[
        "question"
    ]

    expected_answer = test_case[
        "expected_answer"
    ]


    print("\n")
    print("=" * 70)

    print(
        f"Case: {test_case['id']}"
    )

    print(
        f"Question: {question}"
    )


    # ========================================================
    # Run Agent
    # ========================================================

    result = agentic_rag(
        question
    )


    # ========================================================
    # Actual Answer
    # ========================================================

    actual_answer = result.get(
        "answer",
        ""
    )


    # ========================================================
    # Semantic Answer Evaluation
    # ========================================================

    answer_result = evaluate_answer(
        actual_answer,
        expected_answer
    )


    answer_correct = answer_result[
        "correct"
    ]

    similarity = answer_result[
        "similarity"
    ]


    # ========================================================
    # Evidence Evaluation
    # ========================================================

    evidence_sufficient = result.get(
        "evidence_sufficient",
        False
    )


    # ========================================================
    # Citation Evaluation
    # ========================================================

    citation_valid = result.get(
        "citation_valid",
        False
    )


    # ========================================================
    # Trace
    # ========================================================

    trace = result.get(
        "trace",
        []
    )


    workflow_result = evaluate_workflow(
        trace
    )


    # ========================================================
    # Retrieval / Repair
    # ========================================================

    retrieval_count = result.get(
        "retrieval_count",
        0
    )


    repair_count = result.get(
        "repair_count",
        0
    )


    status = result.get(
        "status",
        "unknown"
    )


    return {

        "id":
            test_case["id"],

        "question":
            question,

        "expected_answer":
            expected_answer,

        "actual_answer":
            actual_answer,

        "answer_correct":
            answer_correct,

        "similarity":
            similarity,

        "threshold":
            SIMILARITY_THRESHOLD,

        "evidence_sufficient":
            evidence_sufficient,

        "citation_valid":
            citation_valid,

        "workflow_valid":
            workflow_result[
                "valid"
            ],

        "workflow_violations":
            workflow_result[
                "violations"
            ],

        "retrieval_count":
            retrieval_count,

        "repair_count":
            repair_count,

        "status":
            status,

        "trace":
            trace
    }


# ============================================================
# 7. Batch Evaluation
# ============================================================

def run_evaluation():

    results = []


    for test_case in TEST_CASES:

        result = evaluate_case(
            test_case
        )


        results.append(
            result
        )


    return results


# ============================================================
# 8. Evaluation Report
# ============================================================

def print_report(results):

    total = len(
        results
    )


    if total == 0:

        print(
            "No test cases."
        )

        return


    # ========================================================
    # Count
    # ========================================================

    answer_correct = sum(
        result["answer_correct"]
        for result in results
    )


    evidence_sufficient = sum(
        result["evidence_sufficient"]
        for result in results
    )


    citation_valid = sum(
        result["citation_valid"]
        for result in results
    )


    workflow_valid = sum(
        result["workflow_valid"]
        for result in results
    )


    # ========================================================
    # Average Similarity
    # ========================================================

    average_similarity = (
        sum(
            result["similarity"]
            for result in results
        )
        /
        total
    )


    # ========================================================
    # Report
    # ========================================================

    print("\n\n")

    print("=" * 70)
    print("AGENT SEMANTIC EVALUATION REPORT")
    print("=" * 70)


    print(
        f"Total Cases: "
        f"{total}"
    )


    print(
        f"\nSimilarity Threshold: "
        f"{SIMILARITY_THRESHOLD:.2f}"
    )


    print(
        f"\nAnswer Correct: "
        f"{answer_correct}/{total}"
    )


    print(
        f"Answer Semantic Accuracy: "
        f"{answer_correct / total:.2%}"
    )


    print(
        f"\nAverage Answer Similarity: "
        f"{average_similarity:.4f}"
    )


    print(
        f"\nEvidence Sufficient: "
        f"{evidence_sufficient}/{total}"
    )


    print(
        f"Evidence Sufficiency: "
        f"{evidence_sufficient / total:.2%}"
    )


    print(
        f"\nCitation Valid: "
        f"{citation_valid}/{total}"
    )


    print(
        f"Citation Validity: "
        f"{citation_valid / total:.2%}"
    )


    print(
        f"\nWorkflow Valid: "
        f"{workflow_valid}/{total}"
    )


    print(
        f"Workflow Validity: "
        f"{workflow_valid / total:.2%}"
    )


    # ========================================================
    # Detailed Results
    # ========================================================

    print("\n")
    print("=" * 70)
    print("DETAILED RESULTS")
    print("=" * 70)


    for result in results:

        print("\n")
        print(
            f"[{result['id']}]"
        )


        print(
            f"Question: "
            f"{result['question']}"
        )


        print(
            f"Expected Answer: "
            f"{result['expected_answer']}"
        )


        print(
            f"Actual Answer: "
            f"{result['actual_answer']}"
        )


        print(
            f"Similarity: "
            f"{result['similarity']:.4f}"
        )


        print(
            f"Threshold: "
            f"{result['threshold']:.4f}"
        )


        print(
            f"Answer Correct: "
            f"{result['answer_correct']}"
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
            f"Workflow Valid: "
            f"{result['workflow_valid']}"
        )


        print(
            f"Retrieval Count: "
            f"{result['retrieval_count']}"
        )


        print(
            f"Repair Count: "
            f"{result['repair_count']}"
        )


        print(
            f"Status: "
            f"{result['status']}"
        )


        if result[
            "workflow_violations"
        ]:

            print(
                "Workflow Violations:"
            )


            for violation in result[
                "workflow_violations"
            ]:

                print(
                    f"  - {violation}"
                )


# ============================================================
# 9. Main
# ============================================================

if __name__ == "__main__":

    results = run_evaluation()

    print_report(
        results
    )