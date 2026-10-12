import sys
import os

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from test_dataset import test_cases
from retrieval_chain_group import retrieve_context


def evaluate_context(test_cases):

    total_count = len(test_cases)

    context_hit_count = 0

    for i, case in enumerate(test_cases, start=1):

        question = case["question"]

        expected_chunks = set(
            case["expected_chunks"]
        )

        # =========================
        # 运行完整 Context Engineering
        # =========================

        result = retrieve_context(
            question
        )

        context = result["context"]

        final_docs = result["final_docs"]

        # =========================
        # 获取最终 Chunk ID
        # =========================

        final_chunks = {
            doc.metadata.get("chunk_id")
            for doc in final_docs
        }

        # =========================
        # 判断正确 Chunk 是否保留
        # =========================

        hit = bool(
            expected_chunks & final_chunks
        )

        if hit:
            context_hit_count += 1

        print("\n==============================")
        print(f"测试 {i}")

        print(
            "问题:",
            question
        )

        print(
            "正确 Chunk:",
            expected_chunks
        )

        print(
            "最终 Chunk:",
            final_chunks
        )

        print(
            "结果:",
            "✓ 保留" if hit else "✗ 丢失"
        )

        print(
            "\n最终 Context:"
        )

        print(context)

    # =========================
    # Context Recall
    # =========================

    context_recall = (
        context_hit_count / total_count
    )

    print("\n==============================")
    print("Context Evaluation")
    print("==============================")

    print(
        "总问题数:",
        total_count
    )

    print(
        "保留正确 Chunk:",
        context_hit_count
    )

    print(
        "Context Recall:",
        context_recall
    )


if __name__ == "__main__":

    evaluate_context(test_cases)