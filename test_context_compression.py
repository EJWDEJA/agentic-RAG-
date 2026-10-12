import pickle

from context_group import ContextGroup

from context_compression import (
    compress_group
)

from context_selection import (
    select_groups_by_token_budget,
    count_group_tokens
)


# =====================================
# 1. 加载 Document
# =====================================

with open(
    "documents.pkl",
    "rb"
) as f:

    all_docs = pickle.load(f)


# =====================================
# 2. 创建两个测试 Group
# =====================================

group1_docs = []

group2_docs = []


for doc in all_docs:

    chunk_id = doc.metadata.get("chunk_id")

    if chunk_id in [38, 39, 40]:

        group1_docs.append(doc)

    elif chunk_id in [41, 42, 43]:

        group2_docs.append(doc)


group1 = ContextGroup(
    docs=group1_docs,
    score=0.9
)

group2 = ContextGroup(
    docs=group2_docs,
    score=0.8
)


groups = [
    group1,
    group2
]


# =====================================
# 3. Compression 前 Token
# =====================================

print("\n===== BEFORE COMPRESSION =====")


for i, group in enumerate(
    groups,
    start=1
):

    tokens = count_group_tokens(
        group
    )

    print(
        f"Group {i} Token:",
        tokens
    )


# =====================================
# 4. Compression
# =====================================

question = "计算机科学与技术学院有哪些教师？"


compressed_groups = []


for group in groups:

    compressed_group = compress_group(
        question,
        group,
        threshold=0.5,
        min_sentences=1
    )

    compressed_groups.append(
        compressed_group
    )


# =====================================
# 5. Compression 后 Token
# =====================================

print("\n===== AFTER COMPRESSION =====")


for i, group in enumerate(
    compressed_groups,
    start=1
):

    tokens = count_group_tokens(
        group
    )

    print(
        f"Group {i} Token:",
        tokens
    )


# =====================================
# 6. Token Budget
# =====================================

print("\n===== TOKEN BUDGET =====")


selected_groups = select_groups_by_token_budget(
    compressed_groups,
    max_tokens=3000
)


# =====================================
# 7. 最终选择结果
# =====================================

print(
    "\nSelected Groups:",
    len(selected_groups)
)


for i, group in enumerate(
    selected_groups,
    start=1
):

    print(
        f"\nGroup {i}"
    )

    print(
        "Score:",
        group.score
    )

    print(
        "Tokens:",
        count_group_tokens(group)
    )

    print(
        "Chunks:"
    )

    for doc in group.docs:

        print(
            doc.metadata.get("chunk_id")
        )
