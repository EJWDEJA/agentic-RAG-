import os

# 必须在导入 transformers / sentence_transformers 之前设置！
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"



from hybrid_retriever import hybrid_search
from reranker import rerank_documents_with_scores


question = "赵雪丽的工作职责是什么？"


docs = hybrid_search(question)


scored_docs = rerank_documents_with_scores(
    question,
    docs,
    top_k=3
)


print("\n===== Reranker Scores =====")

for score, doc in scored_docs:

    print(
        "score:",
        score
    )

    print(
        "chunk_id:",
        doc.metadata.get("chunk_id")
    )

    print(
        "content:",
        doc.page_content[:100]
    )

    print("-" * 50)

