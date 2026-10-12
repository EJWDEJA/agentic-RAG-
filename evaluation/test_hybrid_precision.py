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




from retrieval_chain_group import hybrid_search


# =========================
# Golden Dataset
# =========================

golden_dataset = [
    {
        "question": "T2019033是谁？",
        "correct_chunk_id": 38
    },
    {
        "question": "王建国的职称是什么？",
        "correct_chunk_id": 38
    },
    {
        "question": "王建国的研究方向是什么？",
        "correct_chunk_id": 38
    },
    {
        "question": "赵雪丽的研究方向是什么？",
        "correct_chunk_id": 39
    },
    {
        "question": "谁是计算机科学与技术2024级2班的班主任？",
        "correct_chunk_id": 40
    },
    {
        "question": "孙艺萌负责哪些工作？",
        "correct_chunk_id": 41
    }
]


# =========================
# Precision@K
# =========================

def precision_at_k(docs, correct_chunk_id, k):

    top_k_docs = docs[:k]

    for doc in top_k_docs:

        chunk_id = doc.metadata.get("chunk_id")

        if chunk_id == correct_chunk_id:
            return 1 / k

    return 0


# =========================
# 开始测试
# =========================

total_p1 = 0
total_p3 = 0
total_p5 = 0


for item in golden_dataset:

    question = item["question"]
    correct_chunk_id = item["correct_chunk_id"]

    print("\n" + "=" * 80)
    print("问题：", question)
    print("正确 Chunk：", correct_chunk_id)
    print("=" * 80)

    # Hybrid Search
    docs = hybrid_search(question)

    # 打印实际返回结果
    print("\nHybrid Search Top-K：")

    for i, doc in enumerate(docs, start=1):

        print(
            f"Top {i}："
            f"chunk_id={doc.metadata.get('chunk_id')}"
        )

    # Precision
    p1 = precision_at_k(
        docs,
        correct_chunk_id,
        1
    )

    p3 = precision_at_k(
        docs,
        correct_chunk_id,
        3
    )

    p5 = precision_at_k(
        docs,
        correct_chunk_id,
        5
    )

    total_p1 += p1
    total_p3 += p3
    total_p5 += p5

    print("\nPrecision：")
    print(f"Precision@1：{p1:.3f}")
    print(f"Precision@3：{p3:.3f}")
    print(f"Precision@5：{p5:.3f}")


# =========================
# 平均 Precision
# =========================

count = len(golden_dataset)

avg_p1 = total_p1 / count
avg_p3 = total_p3 / count
avg_p5 = total_p5 / count


print("\n")
print("=" * 80)
print("最终平均 Precision")
print("=" * 80)

print(f"Precision@1：{avg_p1:.3f}")
print(f"Precision@3：{avg_p3:.3f}")
print(f"Precision@5：{avg_p5:.3f}")

print("=" * 80)
