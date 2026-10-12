import os

# 必须在导入 transformers / sentence_transformers 之前设置！
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"





import pickle

from hybrid_retriever import hybrid_search
from reranker import rerank_documents_with_scores
from context_group import create_context_groups


question = "赵雪丽的工作职责是什么？"


# Hybrid Search
docs = hybrid_search(question)


# Reranker + Score
scored_docs = rerank_documents_with_scores(
    question,
    docs,
    top_k=3
)


# 加载完整知识库
with open("documents.pkl", "rb") as f:
    all_docs = pickle.load(f)


# 创建 Context Groups
groups = create_context_groups(
    scored_docs,
    all_docs,
    neighbor_size=1
)


print("\n===== Context Groups =====")


for i, group in enumerate(groups, start=1):

    print("\nGroup:", i)

    print("Score:", group.score)

    print("Chunks:")

    for doc in group.docs:

        print(
            doc.metadata.get("chunk_id")
        )
