import os           #封装版上下文工程链

# 必须在导入 transformers / sentence_transformers 之前设置！
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import pickle

from hybrid_retriever import hybrid_search

from context_engineering import build_context

from context_selection import count_group_tokens


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
# 4. Context Engineering
# ==============================

selected_groups = build_context(
    question,
    hybrid_docs,
    all_docs,
    max_tokens=3000
)




print("\n===== After Group Document Sorting =====")

for i, group in enumerate(
    selected_groups,
    start=1
):

    print(
        f"Group {i}:"
    )

    print(
        "chunks:",
        [
            doc.metadata.get("chunk_id")
            for doc in group.docs
        ]
    )



# ==============================
# 5. 输出最终 Groups
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
        f"\nGroup {i}"
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
# 6. Token Budget 检查
# ==============================

print(
    "\n===== Token Budget ====="
)

print(
    "Total tokens:",
    total_tokens
)

print(
    "Max tokens:",
    3000
)

print(
    "Budget 是否通过:",
    total_tokens <= 3000
)