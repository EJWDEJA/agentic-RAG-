from agentic_rag_19_architecture import agentic_rag               #进行agentic rag轨迹trace 评估，这一步主要是一步步输出轨迹方便后续进行轨迹评估进而判断agent工作流是否和逻辑


# ============================================================
# 1. Trajectory Evaluation
#
# 第一版只做：
#
# Agent Result
#      ↓
# Trace
#      ↓
# 基础轨迹分析
#
# 暂时不做 LLM Judge。
# ============================================================

def evaluate_trajectory(            ##有多少事件以及高关注事件是否发生
    trace: list[dict]
) -> dict:

    events = [
        item["event"]
        for item in trace
    ]

    return {
        "event_count": len(events),

        "events": events,

        "used_targeted_retrieval": (
            "targeted_retrieval"
            in events
        ),

        "used_repair": (
            "answer_repair"
            in events
        ),

        "used_safe_fallback": (
            "safe_fallback"
            in events
        ),

        "retrieval_failed": (
            "retrieval_failure"
            in events
        )
    }


# ============================================================
# 2. 打印完整 Trace
# ============================================================

def print_trajectory(            ####输出每一步的详细信息Step   event    detail
    trace: list[dict]
):

    print("\n")
    print("=" * 70)
    print("AGENT TRAJECTORY")
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


# ============================================================
# 3. 测试一个问题
# ============================================================

def run_test(
    question: str
):

    print("\n")
    print("#" * 70)
    print("Agentic RAG Trajectory Test")
    print("#" * 70)

    print("\nQuestion:")
    print(question)

    # --------------------------------------------------------
    # Agent 执行
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
    # 打印最终结果
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("FINAL RESULT")
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
        "\nEvidence IDs:"
    )

    print(
        result.get("evidence_ids")
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

    # --------------------------------------------------------
    # 打印 Trace
    # --------------------------------------------------------

    print_trajectory(     #触发步骤打印
        trace
    )

    # --------------------------------------------------------
    # Trajectory Evaluation
    # --------------------------------------------------------

    trajectory_evaluation = evaluate_trajectory(           #触发关键状态是否执行判别和有多少事件
        trace
    )

    print("\n")
    print("=" * 70)
    print("TRAJECTORY EVALUATION")
    print("=" * 70)

    print(
        "\nEvent Count:"
    )

    print(
        trajectory_evaluation[
            "event_count"
        ]
    )

    print(
        "\nEvents:"
    )

    for event in trajectory_evaluation[
        "events"
    ]:

        print(
            f"  - {event}"
        )

    print(
        "\nUsed Targeted Retrieval:"
    )

    print(
        trajectory_evaluation[
            "used_targeted_retrieval"
        ]
    )

    print(
        "\nUsed Repair:"
    )

    print(
        trajectory_evaluation[
            "used_repair"
        ]
    )

    print(
        "\nUsed Safe Fallback:"
    )

    print(
        trajectory_evaluation[
            "used_safe_fallback"
        ]
    )

    print(
        "\nRetrieval Failed:"
    )

    print(
        trajectory_evaluation[
            "retrieval_failed"
        ]
    )

    return result


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