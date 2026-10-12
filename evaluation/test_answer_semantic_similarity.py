
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

from langchain_community.embeddings import OllamaEmbeddings

import numpy as np


embeddings = OllamaEmbeddings(
    model="qwen3-embedding"
)


def cosine_similarity(vec1, vec2):

    vec1 = np.array(vec1)

    vec2 = np.array(vec2)

    return np.dot(vec1, vec2) / (
        np.linalg.norm(vec1) *
        np.linalg.norm(vec2)
    )


total_score = 0


for i, item in enumerate(golden_answer_dataset):

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
                "session_id": f"answer_eval_{i}"
            }
        }
    )


    answer = result["answer"].content


    print("\n========== MyRAG 最终回答 ==========\n")

    print(answer)


    reference_vector = embeddings.embed_query(
        reference_answer
    )

    answer_vector = embeddings.embed_query(
        answer
    )


    score = cosine_similarity(
        reference_vector,
        answer_vector
    )


    print("\n========== Semantic Similarity ==========\n")

    print("语义相似度：", round(score, 4))


    total_score += score


average_score = total_score / len(
    golden_answer_dataset
)


print("\n")
print("=" * 100)

print("Answer Correctness —— Semantic Similarity")

print("=" * 100)

print(
    "平均语义相似度：",
    round(average_score, 4)
)
