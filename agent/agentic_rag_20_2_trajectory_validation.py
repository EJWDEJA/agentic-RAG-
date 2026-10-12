from agentic_rag_19_architecture import agentic_rag        ##此处的路径轨迹校验用于校验工作流的合法性，但是不保证内部参数一定会被正确记录，尤其是repair count=2改成=3工作流校验根本发现不了，后面真的要用到数量来总结规律的时候就要进行参数合法性校验


# ============================================================
# 1. Workflow Contract
# ============================================================

def validate_trajectory(           #工作流校验本质上就是找茬
    trace: list[dict]
) -> dict:

    events = [
        item["event"]
        for item in trace
    ]

    violations = []

    # --------------------------------------------------------
    # Rule 1
    #
    # Retrieval Failure 后不能继续执行 Grounding
    # --------------------------------------------------------

    if "retrieval_failure" in events:

        failure_index = events.index(  ##输出下标索引的
            "retrieval_failure"
        )

        later_events = events[
            failure_index + 1:
        ]

        if later_events:

            violations.append(
                "Retrieval Failure 后仍然继续执行 Agent Workflow"
            )

    # --------------------------------------------------------
    # Rule 2
    #
    # Evidence Gap 后必须进入 Targeted Retrieval
    # --------------------------------------------------------

    for index, event in enumerate(events):

        if event == "evidence_gap":

            later_events = events[
                index + 1:
            ]

            if (
                "targeted_retrieval"
                not in later_events
            ):

                violations.append(
                    "Evidence Gap 后没有执行 Targeted Retrieval"
                )

    # --------------------------------------------------------
    # Rule 3
    #
    # Evidence Evaluation insufficient
    # 后不能直接 Grounding
    #
    # 注意：
    # 我们通过相邻事件判断。
    # --------------------------------------------------------

    for index, event in enumerate(events[:-1]):##剔除最后一个元素

        if event == "evidence_evaluation":

            next_event = events[
                index + 1
            ]

            # 当前 Trace detail 中记录了
            # insufficient / sufficient。
            #
            # 所以这里进一步读取原始 event。
            current_trace_event = trace[
                index
            ]

            detail = current_trace_event[
                "detail"
            ].lower()

            if (
                "insufficient"
                in detail
                and next_event == "grounding"
            ):

                violations.append(
                    "Evidence insufficient 后直接进入 Grounding"
                )

    # --------------------------------------------------------
    # Rule 4
    #
    # Citation Validation FAIL
    # 后必须 Failure Classification
    #
    # 这里根据 detail 判断 PASS / FAIL
    # --------------------------------------------------------

    for index, event in enumerate(trace):

        if event["event"] == "citation_validation":

            detail = event[
                "detail"
            ].upper()

            if "FAIL" in detail:

                later_events = events[
                    index + 1:
                ]

                if (
                    "failure_classification"
                    not in later_events
                ):

                    violations.append(
                        "Citation Validation FAIL 后没有 Failure Classification"
                    )

    # --------------------------------------------------------
    # Rule 5
    #
    # Answer Repair 后必须再次 Citation Validation
    # --------------------------------------------------------

    for index, event in enumerate(events):

        if event == "answer_repair":

            later_events = events[
                index + 1:
            ]

            if (
                "citation_validation"
                not in later_events
            ):

                violations.append(
                    "Answer Repair 后没有重新进行 Citation Validation"
                )

    # --------------------------------------------------------
    # Rule 6
    #
    # Safe Fallback 必须是最后一个事件
    # --------------------------------------------------------

    if "safe_fallback" in events:

        fallback_index = events.index(
            "safe_fallback"
        )

        if fallback_index != len(events) - 1:

            violations.append(
                "Safe Fallback 后仍然存在其他 Agent 行为"
            )

    # --------------------------------------------------------
    # 最终结果
    # --------------------------------------------------------

    valid = len(
        violations
    ) == 0

    return {
        "valid": valid,
        "violations": violations,
        "event_count": len(events),
        "events": events
    }


# ============================================================
# 2. 打印 Validation Result
# ============================================================

def print_validation(             #打印找完茬以后的结果
    validation: dict
):

    print("\n")
    print("=" * 70)
    print("TRAJECTORY VALIDATION")
    print("=" * 70)

    print(
        "\nValid:"
    )

    print(
        validation["valid"]
    )

    print(
        "\nEvent Count:"
    )

    print(
        validation["event_count"]
    )

    if validation["valid"]:

        print(
            "\nWorkflow Contract: PASS"
        )

        print(
            "Agent 执行轨迹符合当前 Workflow Contract。"
        )

    else:

        print(
            "\nWorkflow Contract: FAIL"
        )

        print(
            "\nViolations:"
        )

        for violation in validation[
            "violations"
        ]:

            print(
                f"  - {violation}"
            )


# ============================================================
# 3. 完整测试
# ============================================================

def run_test(   #实际运行
    question: str
):

    print("\n")
    print("#" * 70)
    print("Agentic RAG Trajectory Validation Test")
    print("#" * 70)

    print(
        "\nQuestion:"
    )

    print(
        question
    )

    # --------------------------------------------------------
    # Agent Execution
    # --------------------------------------------------------

    result = agentic_rag(
        question
    )

    # --------------------------------------------------------
    # 获取 Trace
    # --------------------------------------------------------

    trace = result.get(
        "trace",
        []
    )

    # --------------------------------------------------------
    # 打印 Trace
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("TRACE")
    print("=" * 70)

    for event in trace:

        print(
            f"\nStep {event['step']}"
        )

        print(
            f"Event: {event['event']}"
        )

        print(
            f"Detail: {event['detail']}"
        )

    # --------------------------------------------------------
    # Trajectory Validation
    # --------------------------------------------------------

    validation = validate_trajectory(
        trace
    )

    # --------------------------------------------------------
    # 打印 Validation
    # --------------------------------------------------------

    print_validation(
        validation
    )

    # --------------------------------------------------------
    # 最终 Agent Result
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("AGENT RESULT")
    print("=" * 70)

    print(
        "\nStatus:"
    )

    print(
        result.get("status")
    )

    print(
        "\nAnswer:"
    )

    print(
        result.get("answer")
    )

    print(
        "\nRetrieval Count:"
    )

    print(
        result.get("retrieval_count")
    )

    print(
        "\nRepair Count:"
    )

    print(
        result.get("repair_count")
    )

    return result, validation


# ============================================================
# 4. Main
# ============================================================

if __name__ == "__main__":

    question = input(
        "\n请输入问题："
    )

    run_test(
        question
    )