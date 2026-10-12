
import os

# 必须在导入 transformers / sentence_transformers 之前设置
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import pickle

from hybrid_retriever import hybrid_search
from reranker import rerank_documents_with_scores

from context_group import (
    create_context_groups,
    merge_overlapping_groups_transitively,
    sort_all_group_docs
)

from context_compression import compress_group
from context_selection import select_groups


question = "计算机科学与技术学院有哪些教师？"


# ============================================================
# 加载全部原始 Documents
# ============================================================

with open("documents.pkl", "rb") as f:
    all_docs = pickle.load(f)


# ============================================================
# 通用：打印 Documents
# ============================================================

def print_docs(title, docs):

    print("\n")
    print("=" * 80)
    print(title)
    print("=" * 80)

    for i, doc in enumerate(docs):

        print(f"\n--- {i + 1} ---")

        print(
            "source:",
            doc.metadata.get("source")
        )

        print(
            "sheet:",
            doc.metadata.get("sheet")
        )

        print(
            "chunk_id:",
            doc.metadata.get("chunk_id")
        )

        print(
            "content:",
            doc.page_content
        )


# ============================================================
# 通用：打印 Groups
# ============================================================

def print_groups(title, groups, show_content=True):

    print("\n")
    print("=" * 80)
    print(title)
    print("=" * 80)

    for i, group in enumerate(groups):

        print(f"\n### GROUP {i + 1}")

        print(
            "group score:",
            group.score
        )

        print(
            "chunk_ids:",
            [
                doc.metadata.get("chunk_id")
                for doc in group.docs
            ]
        )

        if show_content:

            for doc in group.docs:

                print(
                    "content:",
                    doc.page_content
                )


# ============================================================
# ① Hybrid Search
# ============================================================

docs = hybrid_search(question)

print_docs(
    "① HYBRID SEARCH",
    docs
)


# ============================================================
# ② BGE Reranker
# ============================================================

reranked = rerank_documents_with_scores(
    question,
    docs,
    top_k=3
)

print("\n")
print("=" * 80)
print("② BGE RERANKER")
print("=" * 80)

for i, (score, doc) in enumerate(reranked):

    print(f"\n--- {i + 1} ---")

    print(
        "score:",
        score
    )

    print(
        "source:",
        doc.metadata.get("source")
    )

    print(
        "sheet:",
        doc.metadata.get("sheet")
    )

    print(
        "chunk_id:",
        doc.metadata.get("chunk_id")
    )

    print(
        "content:",
        doc.page_content
    )


# ============================================================
# ③ Create Context Groups
#
# create_context_groups 内部已经完成：
#
# Core Doc
#     ↓
# Neighbor Expansion
#     ↓
# ContextGroup 创建
# ============================================================

groups = create_context_groups(
    reranked,
    all_docs,
    neighbor_size=1
)

print_groups(
    "③ CREATE CONTEXT GROUPS",
    groups
)


# ============================================================
# ④ Transitive Merge
# ============================================================

groups = merge_overlapping_groups_transitively(
    groups
)

print_groups(
    "④ AFTER TRANSITIVE MERGE",
    groups,
    show_content=False
)


# ============================================================
# ⑤ Group 内部 Chunk 排序
# ============================================================

groups = sort_all_group_docs(
    groups
)

print_groups(
    "⑤ AFTER SORTING GROUP DOCS",
    groups,
    show_content=False
)


# ============================================================
# ⑥ Compression
# ============================================================

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


print_groups(
    "⑥ AFTER COMPRESSION",
    compressed_groups
)


# ===============================

# ⑦ Token Budget Selection
# ============================================================

selected_groups = select_groups(
    compressed_groups,
    max_tokens=3000
)

print_groups(
    "⑦ AFTER TOKEN BUDGET",
    selected_groups
)


# ============================================================
# ⑧ 最终结果总结
# ============================================================

print("\n")
print("=" * 80)
print("⑧ FINAL SUMMARY")
print("=" * 80)

print(
    "Hybrid Search:",
    len(docs),
    "docs"
)

print(
    "Reranker:",
    len(reranked),
    "core docs"
)

print(
    "Context Groups:",
    len(groups),
    "groups"
)

print(
    "Compressed Groups:",
    len(compressed_groups),
    "groups"
)

print(
    "Selected Groups:",
    len(selected_groups),
    "groups"
)

print("\n最终选中的 Chunk：")

for i, group in enumerate(selected_groups):

    print(
        f"Group {i + 1}:",
        [
            doc.metadata.get("chunk_id")
            for doc in group.docs
        ]
    )

