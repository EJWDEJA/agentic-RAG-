import pickle
import copy

from context_compression import compress_group
from context_group import ContextGroup


# =========================
# 加载全部 Documents
# =========================

with open("documents.pkl", "rb") as f:
    all_docs = pickle.load(f)


# =========================
# 找到 38、39、40、41
# =========================

target_docs = []

for doc in all_docs:

    chunk_id = doc.metadata.get("chunk_id")

    if chunk_id in [38, 39, 40, 41]:

        target_docs.append(doc)


# =========================
# 按 chunk_id 排序
# =========================

target_docs = sorted(
    target_docs,
    key=lambda doc: doc.metadata.get("chunk_id")
)


# =========================
# 打印压缩前
# =========================

print("\n")
print("=" * 80)
print("===== BEFORE COMPRESSION =====")
print("=" * 80)


for doc in target_docs:

    print("\n")
    print("--- chunk_id:", doc.metadata.get("chunk_id"), "---")

    print(
        "source:",
        doc.metadata.get("source")
    )

    print(
        "sheet:",
        doc.metadata.get("sheet")
    )

    print(
        "content:"
    )

    print(doc.page_content)


# =========================
# 深拷贝
# 防止修改原始 Documents
# =========================

test_docs = copy.deepcopy(
    target_docs
)


# =========================
# 创建 ContextGroup
# =========================

group = ContextGroup(
    docs=test_docs,
    score=1.0
)


# =========================
# 执行 Compression
# =========================

compressed_group = compress_group(
    "计算机科学与技术学院有哪些教师？",
    group,
    threshold=0.5,
    min_sentences=1
)


# =========================
# 打印压缩后
# =========================

print("\n")
print("=" * 80)
print("===== AFTER COMPRESSION =====")
print("=" * 80)


for doc in compressed_group.docs:

    print("\n")
    print("--- chunk_id:", doc.metadata.get("chunk_id"), "---")

    print(
        "source:",
        doc.metadata.get("source")
    )

    print(
        "sheet:",
        doc.metadata.get("sheet")
    )

    print(
        "content:"
    )

    print(doc.page_content)
