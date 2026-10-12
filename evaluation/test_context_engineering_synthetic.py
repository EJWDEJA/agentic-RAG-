
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

    print("\n========== Context Engineering 最终结果 ==========\n")

    final_docs = []

    for group in selected_groups:
        final_docs.extend(group.docs)

    for i, doc in enumerate(final_docs):

        chunk_id = doc.metadata.get("chunk_id")

        if chunk_id in relevant_chunk_ids:
            label = " 相关"
        else:
            label = " 无关"

        print("\n---", i + 1, "---")
        print("chunk_id:", chunk_id)
        print("判断:", label)
        print("content:", doc.page_content)
