import sys    #测试混合检索的召回率
import os

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from test_dataset import test_cases
from hybrid_retriever import hybrid_search


def evaluate_retrieval(test_cases):

    hit_count = 0

    recall_1_count = 0
    recall_3_count = 0
    recall_5_count = 0

    total_count = len(test_cases)

    for i, case in enumerate(test_cases, start=1):

        question = case["question"]

        expected_chunks = set(
            case["expected_chunks"]
        )

        docs = hybrid_search(question)

        retrieved_chunks = [
            doc.metadata.get("chunk_id")
            for doc in docs
        ]

        retrieved_set = set(
            retrieved_chunks
        )

        # -------------------------
        # Hit Rate
        # -------------------------

        hit = bool(
            expected_chunks & retrieved_set
        )

        if hit:
            hit_count += 1

        # -------------------------
        # Recall@1
        # -------------------------

        top_1 = set(
            retrieved_chunks[:1]
        )

        if expected_chunks & top_1:
            recall_1_count += 1

        # -------------------------
        # Recall@3
        # -------------------------

        top_3 = set(
            retrieved_chunks[:3]
        )

        if expected_chunks & top_3:
            recall_3_count += 1

        # -------------------------
        # Recall@5
        # -------------------------

        top_5 = set(
            retrieved_chunks[:5]
        )

        if expected_chunks & top_5:
            recall_5_count += 1

        print("\n==============================")
        print(f"测试 {i}")
        print("问题:", question)
        print("正确 Chunk:", expected_chunks)
        print("检索顺序:", retrieved_chunks)
        print(
            "结果:",
            "✓ 命中" if hit else "✗ 未命中"
        )

    # -------------------------
    # 计算指标
    # -------------------------

    hit_rate = hit_count / total_count

    recall_1 = recall_1_count / total_count
    recall_3 = recall_3_count / total_count
    recall_5 = recall_5_count / total_count

    print("\n==============================")
    print("Retrieval Evaluation")
    print("==============================")

    print("总问题数:", total_count)

    print("Hit Rate:", hit_rate)

    print("Recall@1:", recall_1)

    print("Recall@3:", recall_3)

    print("Recall@5:", recall_5)


if __name__ == "__main__":

    evaluate_retrieval(test_cases)