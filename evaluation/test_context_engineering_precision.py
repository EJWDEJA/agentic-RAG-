
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





from context_engineering_test_data import (
    test_docs,
    golden_context_dataset
)

from context_engineering import build_context


def context_precision(final_docs, relevant_chunk_ids):

    if not final_docs:
        return 0

    relevant_count = 0

    for doc in final_docs:

        chunk_id = doc.metadata.get("chunk_id")

        if chunk_id in relevant_chunk_ids:
            relevant_count += 1

    return relevant_count / len(final_docs)


total_precision = 0


for item in golden_context_dataset:

    question = item["question"]

    relevant_chunk_ids = item["relevant_chunk_ids"]

    print("\n")
    print("=" * 100)

    print("问题：", question)

    print("标准相关 Chunk：", relevant_chunk_ids)

    print("=" * 100)


    selected_groups = build_context(
        question,
        test_docs,
        test_docs,
        max_tokens=3000
    )


    final_docs = []

    for group in selected_groups:

        final_docs.extend(group.docs)


    precision = context_precision(
        final_docs,
        relevant_chunk_ids
    )


    print("\n========== Context Engineering 最终结果 ==========\n")


    relevant_count = 0

    for i, doc in enumerate(final_docs):

        chunk_id = doc.metadata.get("chunk_id")

        if chunk_id in relevant_chunk_ids:

            label = " 相关"

            relevant_count += 1

        else:

            label = " 无关"


        print("\n---", i + 1, "---")

        print("chunk_id:", chunk_id)

        print("判断:", label)

        print("content:", doc.page_content)


    print("\n----------------------------------------")

    print("相关 Chunk 数量：", relevant_count)

    print("最终 Context Chunk 总数：", len(final_docs))

    print("Context Precision：", round(precision, 4))


    total_precision += precision


avg_precision = total_precision / len(
    golden_context_dataset
)


print("\n")
print("=" * 100)

print("Synthetic Context Precision 总体结果")

print("=" * 100)

print(
    "平均 Context Precision：",
    round(avg_precision, 4)
)
