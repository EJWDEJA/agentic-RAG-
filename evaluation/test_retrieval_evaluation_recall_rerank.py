import sys       #测试混合检索和重排序的召回率
import os

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)
import os       #这是针对group(类)的检索处理上下文管理

# 必须在导入 transformers / sentence_transformers 之前设置！
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"



from test_dataset import test_cases
from hybrid_retriever import hybrid_search
from reranker import rerank_documents_with_scores


def evaluate_reranker(test_cases):

    total_count = len(test_cases)

    hybrid_recall_1 = 0
    hybrid_recall_3 = 0
    hybrid_recall_5 = 0

    rerank_recall_1 = 0
    rerank_recall_3 = 0
    rerank_recall_5 = 0

    for i, case in enumerate(test_cases, start=1):

        question = case["question"]

        expected_chunks = set(
            case["expected_chunks"]
        )

        # =========================
        # ① Hybrid Search
        # =========================

        hybrid_docs = hybrid_search(question)

        hybrid_chunks = [
            doc.metadata.get("chunk_id")
            for doc in hybrid_docs
        ]

        # =========================
        # ② BGE Reranker
        # =========================

        reranked = rerank_documents_with_scores(
            question,
            hybrid_docs
        )

        rerank_chunks = [
            doc.metadata.get("chunk_id")
            for score, doc in reranked
        ]

        # =========================
        # Hybrid Recall@K
        # =========================

        if expected_chunks & set(hybrid_chunks[:1]):
            hybrid_recall_1 += 1

        if expected_chunks & set(hybrid_chunks[:3]):
            hybrid_recall_3 += 1

        if expected_chunks & set(hybrid_chunks[:5]):
            hybrid_recall_5 += 1

        # =========================
        # Reranker Recall@K
        # =========================

        if expected_chunks & set(rerank_chunks[:1]):
            rerank_recall_1 += 1

        if expected_chunks & set(rerank_chunks[:3]):
            rerank_recall_3 += 1

        if expected_chunks & set(rerank_chunks[:5]):
            rerank_recall_5 += 1

        # =========================
        # 打印单题结果
        # =========================

        print("\n==============================")
        print(f"测试 {i}")
        print("问题:", question)

        print(
            "正确 Chunk:",
            expected_chunks
        )

        print(
            "Hybrid:",
            hybrid_chunks
        )

        print(
            "Reranker:",
            rerank_chunks
        )

    # =========================
    # 计算结果
    # =========================

    print("\n==============================")
    print("Hybrid Search")
    print("==============================")

    print(
        "Recall@1:",
        hybrid_recall_1 / total_count
    )

    print(
        "Recall@3:",
        hybrid_recall_3 / total_count
    )

    print(
        "Recall@5:",
        hybrid_recall_5 / total_count
    )

    print("\n==============================")
    print("Hybrid + BGE Reranker")
    print("==============================")

    print(
        "Recall@1:",
        rerank_recall_1 / total_count
    )

    print(
        "Recall@3:",
        rerank_recall_3 / total_count
    )

    print(
        "Recall@5:",
        rerank_recall_5 / total_count
    )


if __name__ == "__main__":

    evaluate_reranker(test_cases)