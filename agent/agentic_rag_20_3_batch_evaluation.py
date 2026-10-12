from agentic_rag_19_architecture import agentic_rag                ##此处将20.0结果评估和20.2工作流评估进行合并使用，通过大量case测试集得出关注评判指标（百分数或者浮点数）

from agentic_rag_20_3_dataset import TEST_CASES


# ============================================================
# 1. Answer Evaluation
# ============================================================

def evaluate_answer(          ##################################################################规则校验，由于llm生成答案的不确定性，使用一般情况下结果校验都是失败的，不过后续会升级影响不大
    actual_answer,
    expected_answer
):
    """
    第一版使用最简单的 exact match。

    后面再升级：
        semantic evaluation
        LLM Judge
        structured evaluation
    """

    if not actual_answer:
        return False

    actual = actual_answer.strip().lower()
    expected = expected_answer.strip().lower()

    return actual == expected


# ============================================================
# 2. Workflow Evaluation
# ============================================================

def evaluate_workflow(trace):

    events = [
        item["event"]
        for item in trace
    ]

    violations = []


    # --------------------------------------------------------
    # Rule 1
    # retrieval_failure 必须终止
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
    # safe_fallback 必须终止
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
    # evidence evaluation
    # --------------------------------------------------------

    for index, event in enumerate(events[:-1]):

        if event != "evidence_evaluation":
            continue


        next_event = events[index + 1]


        detail = trace[index].get(
            "detail",
            ""
        ).lower()


        if "insufficient" in detail:

            if next_event != "evidence_gap":

                violations.append(
                    "Insufficient evidence "
                    "must be followed by "
                    "evidence_gap."
                )


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

            next_event = events[index + 1]


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


        next_event = events[index + 1]


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
        "valid": len(violations) == 0,
        "violations": violations
    }


# ============================================================
# 3. Evaluate One Case
# ============================================================

def evaluate_case(test_case):

    question = test_case[
        "question"
    ]

    expected_answer = test_case[
        "expected_answer"
    ]


    print("\n" + "=" * 70)

    print(
        f"Case: {test_case['id']}"
    )

    print(
        f"Question: {question}"
    )


    # --------------------------------------------------------
    # Run Agent
    # --------------------------------------------------------

    result = agentic_rag(
        question
    )


    # --------------------------------------------------------
    # Answer
    # --------------------------------------------------------

    actual_answer = result.get(
        "answer",
        ""
    )


    answer_correct = evaluate_answer(
        actual_answer,
        expected_answer
    )


    # --------------------------------------------------------
    # Evidence
    # --------------------------------------------------------

    evidence_sufficient = result.get(
        "evidence_sufficient",
        False
    )


    # --------------------------------------------------------
    # Citation
    # --------------------------------------------------------

    citation_valid = result.get(
        "citation_valid",
        False
    )


    # --------------------------------------------------------
    # Trace
    # --------------------------------------------------------

    trace = result.get(
        "trace",
        []
    )


    workflow_result = evaluate_workflow(
        trace
    )


    # --------------------------------------------------------
    # Counters
    # --------------------------------------------------------

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

        "id": test_case["id"],

        "question": question,

        "expected_answer":
            expected_answer,

        "actual_answer":
            actual_answer,

        "answer_correct":
            answer_correct,

        "evidence_sufficient":
            evidence_sufficient,

        "citation_valid":
            citation_valid,

        "workflow_valid":
            workflow_result["valid"],

        "workflow_violations":
            workflow_result["violations"],

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
# 4. Batch Evaluation
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
# 5. Evaluation Report
# ============================================================

def print_report(results):

    total = len(results)


    if total == 0:

        print(
            "No test cases."
        )

        return


    answer_correct = sum(
        r["answer_correct"]
        for r in results
    )


    evidence_sufficient = sum(
        r["evidence_sufficient"]
        for r in results
    )


    citation_valid = sum(
        r["citation_valid"]
        for r in results
    )


    workflow_valid = sum(
        r["workflow_valid"]
        for r in results
    )


    print("\n\n")
    print("=" * 70)
    print("AGENT EVALUATION REPORT")
    print("=" * 70)


    print(
        f"Total Cases: "
        f"{total}"
    )


    print(
        f"\nAnswer Correct: "
        f"{answer_correct}/{total}"
    )

    print(
        f"Answer Correctness: "
        f"{answer_correct / total:.2%}"
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


    # --------------------------------------------------------
    # Detailed Results
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("CASE RESULTS")
    print("=" * 70)


    for result in results:

        print(
            f"\n[{result['id']}]"
        )

        print(
            f"Question: "
            f"{result['question']}"
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

        if result["workflow_violations"]:

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
# 6. Main
# ============================================================

if __name__ == "__main__":

    results = run_evaluation()

    print_report(
        results
    )