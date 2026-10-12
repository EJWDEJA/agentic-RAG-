import sys     
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




from answer_test_dataset import golden_answer_dataset

from rag_chain_group import rag_chain


for item in golden_answer_dataset:

    question = item["question"]

    reference_answer = item["reference_answer"]


    print("\n")
    print("=" * 100)

    print("问题：", question)

    print("标准答案：", reference_answer)

    print("=" * 100)


    result = rag_chain.invoke(
    {
        "question": question
    },
    config={
        "configurable": {
            "session_id": "evaluation_test"
        }
    }
    )



    answer = result["answer"].content



    print("\n========== MyRAG 最终回答 ==========\n")

    print(answer)


    print("\n========== 标准答案 ==========\n")

    print(reference_answer)


    print("\n")
    print("-" * 100)
