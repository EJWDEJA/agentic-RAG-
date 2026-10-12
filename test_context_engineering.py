import os           #未封装版上下文工程链

# 必须在导入 transformers / sentence_transformers 之前设置！
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"





import pickle

from hybrid_retriever import hybrid_search

from reranker import rerank_documents_with_scores

from context_group import (
    create_context_groups,
    merge_overlapping_groups_transitively
)

from context_selection import (
    select_groups,
    count_group_tokens
)


# ==============================
# 1. 加载全部 Document
# ==============================

with open(
    "documents.pkl",
    "rb"
) as f:

    all_docs = pickle.load(f)


# ==============================
# 2. 测试问题
# ==============================

question = "学校什么时候成立？"


# ==============================
# 3. Hybrid Search
# ==============================

hybrid_docs = hybrid_search(
    question
)


print("\n===== Hybrid Search =====")

print(
    "候选 Document 数量:",
    len(hybrid_docs)
)

for doc in hybrid_docs:

    print(
        "chunk_id:",
        doc.metadata.get("chunk_id")
    )


# ==============================
# 4. BGE Reranker
# ==============================

scored_docs = rerank_documents_with_scores(
    question,
    hybrid_docs,
    top_k=3
)


print("\n===== Reranker Top 3 =====")

for score, doc in scored_docs:

    print(
        "score:",
        score,
        "chunk_id:",
        doc.metadata.get("chunk_id")
    )


# ==============================
# 5. 创建 Context Groups
# ==============================

groups = create_context_groups(
    scored_docs,
    all_docs,
    neighbor_size=1
)


print("\n===== Context Groups =====")

for i, group in enumerate(
    groups,
    start=1
):

    print(
        f"Group {i}:"
    )

    print(
        "score:",
        group.score
    )

    print(
        "chunks:",
        [
            doc.metadata.get("chunk_id")
            for doc in group.docs
        ]
    )


# ==============================
# 6. Transitive Overlap Merge
# ==============================

groups = merge_overlapping_groups_transitively(
    groups
)


print("\n===== After Transitive Merge =====")

for i, group in enumerate(
    groups,
    start=1
):

    print(
        f"Group {i}:"
    )

    print(
        "score:",
        group.score
    )

    print(
        "chunks:",
        [
            doc.metadata.get("chunk_id")
            for doc in group.docs
        ]
    )


# ==============================
# 7. Group Ranking
#    +
#    Token Budget
# ==============================

selected_groups = select_groups(
    groups,
    max_tokens=3000
)


# ==============================
# 8. 最终 Context Groups
# ==============================

print("\n===== Final Selected Groups =====")

total_tokens = 0

for i, group in enumerate(
    selected_groups,
    start=1
):

    group_tokens = count_group_tokens(
        group
    )

    total_tokens += group_tokens

    print(
        f"Group {i}:"
    )

    print(
        "score:",
        group.score
    )

    print(
        "chunks:",
        [
            doc.metadata.get("chunk_id")
            for doc in group.docs
        ]
    )

    print(
        "tokens:",
        group_tokens
    )


# ==============================
# 9. Token 总量
# ==============================

print(
    "\nTotal tokens:",
    total_tokens
)

print(
    "Token Budget:",
    3000
)

print(
    "Budget 是否通过:",
    total_tokens <= 3000
)
