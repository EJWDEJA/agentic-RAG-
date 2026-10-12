import os       #这是针对group(类)的检索处理上下文管理中间各环节数据print方便查找数据流中数据的丢失

# 必须在导入 transformers / sentence_transformers 之前设置！
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import sys
import os

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from test_dataset import test_cases

import context_engineering
import context_selection

from retrieval_chain_group import retrieve_context


# =========================================================
# 保存原始函数
# =========================================================

_original_rerank = (
    context_engineering.rerank_documents_with_scores
)

_original_create_groups = (
    context_engineering.create_context_groups
)

_original_merge_groups = (
    context_engineering.merge_overlapping_groups_transitively
)

_original_sort_groups = (
    context_engineering.sort_all_group_docs
)

_original_compress_group = (
    context_engineering.compress_group
)

_original_select_groups = (
    context_engineering.select_groups
)

_original_select_groups_by_token_budget = (
    context_selection.select_groups_by_token_budget
)


# =========================================================
# ① 监控 BGE Reranker
# =========================================================

def debug_rerank(
    question,
    docs,
    top_k=3
):

    result = _original_rerank(
        question,
        docs,
        top_k=top_k
    )

    print("\n")
    print("=" * 80)
    print("① BGE RERANKER")
    print("=" * 80)

    for rank, (score, doc) in enumerate(
        result,
        start=1
    ):

        print(
            f"Rank {rank} | "
            f"chunk_id={doc.metadata.get('chunk_id')} | "
            f"score={score}"
        )

    return result


# =========================================================
# ② 监控 Group 创建
# =========================================================

def debug_create_groups(
    scored_docs,
    all_docs,
    neighbor_size=1
):

    groups = _original_create_groups(
        scored_docs,
        all_docs,
        neighbor_size=neighbor_size
    )

    print("\n")
    print("=" * 80)
    print("② GROUPS BEFORE MERGE")
    print("=" * 80)

    for index, group in enumerate(
        groups,
        start=1
    ):

        chunk_ids = [
            doc.metadata.get("chunk_id")
            for doc in group.docs
        ]

        print(
            f"Group {index} | "
            f"score={group.score} | "
            f"chunks={chunk_ids}"
        )

    return groups


# =========================================================
# ③ 监控 Transitive Merge
# =========================================================

def debug_merge_groups(groups):

    result = _original_merge_groups(
        groups
    )

    print("\n")
    print("=" * 80)
    print("③ GROUPS AFTER TRANSITIVE MERGE")
    print("=" * 80)

    for index, group in enumerate(
        result,
        start=1
    ):

        chunk_ids = [
            doc.metadata.get("chunk_id")
            for doc in group.docs
        ]

        print(
            f"Group {index} | "
            f"score={group.score} | "
            f"chunks={chunk_ids}"
        )

    return result

# =========================================================
# ④ 监控 Group 内部排序
# =========================================================

def debug_sort_groups(groups):

    result = _original_sort_groups(
        groups
    )

    print("\n")
    print("=" * 80)
    print("④ AFTER SORT INSIDE GROUP")
    print("=" * 80)

    for index, group in enumerate(
        result,
        start=1
    ):

        chunk_ids = [
            doc.metadata.get("chunk_id")
            for doc in group.docs
        ]

        print(
            f"Group {index} | "
            f"chunks={chunk_ids}"
        )

    return result


# =========================================================
# ⑤ 监控 Compression
# =========================================================

def debug_compress_group(
    question,
    group,
    threshold=0.5,
    min_sentences=1
):

    result = _original_compress_group(
        question,
        group,
        threshold=threshold,
        min_sentences=min_sentences
    )

    chunk_ids = [
        doc.metadata.get("chunk_id")
        for doc in result.docs
    ]

    print(
        "\nCompression:"
    )

    print(
        "  before chunks:",
        [
            doc.metadata.get("chunk_id")
            for doc in group.docs
        ]
    )

    print(
        "  after chunks:",
        chunk_ids
    )

    return result


# =========================================================
# ⑥ 监控 Token Budget
# =========================================================

def debug_select_groups_by_token_budget(
    groups,
    max_tokens=3000
):

    print("\n")
    print("=" * 80)
    print("⑥ TOKEN BUDGET")
    print("=" * 80)

    print(
        "max_tokens:",
        max_tokens
    )

    total_tokens = 0

    for index, group in enumerate(
        groups,
        start=1
    ):

        group_tokens = (
            context_selection.count_group_tokens(
                group
            )
        )

        chunk_ids = [
            doc.metadata.get("chunk_id")
            for doc in group.docs
        ]

        print(
            f"\nRank {index}"
        )

        print(
            "  chunks:",
            chunk_ids
        )

        print(
            "  score:",
            group.score
        )

        print(
            "  group tokens:",
            group_tokens
        )

        print(
            "  total before:",
            total_tokens
        )

        if (
            total_tokens + group_tokens
            > max_tokens
        ):

            print(
                "  → SKIP"
            )

        else:

            print(
                "  → SELECT"
            )

            total_tokens += group_tokens

        print(
            "  total after:",
            total_tokens
        )

    # =====================================================
    # 最关键：
    # 真正调用正式的 token budget 函数
    # =====================================================

    result = _original_select_groups_by_token_budget(
        groups,
        max_tokens=max_tokens
    )

    print("\n")
    print(
        "真正 select_groups_by_token_budget "
        "最终选择:"
    )

    for index, group in enumerate(
        result,
        start=1
    ):

        print(
            f"Selected Group {index}:",
            [
                doc.metadata.get("chunk_id")
                for doc in group.docs
            ]
        )

    return result


# =========================================================
# 把 Context Engineering 里面的函数替换成 Debug 包装器
# =========================================================

context_engineering.rerank_documents_with_scores = (
    debug_rerank
)

context_engineering.create_context_groups = (
    debug_create_groups
)

context_engineering.merge_overlapping_groups_transitively = (
    debug_merge_groups
)

context_engineering.sort_all_group_docs = (
    debug_sort_groups
)

context_engineering.compress_group = (
    debug_compress_group
)

context_engineering.select_groups = (
    _original_select_groups
)

context_selection.select_groups_by_token_budget = (
    debug_select_groups_by_token_budget
)

# =========================================================
# 开始 Debug
# =========================================================

def main():

    for i, case in enumerate(
        test_cases,
        start=1
    ):

        # 只 Debug 原来失败的 3 个测试
        if i not in [2, 3, 6]:
            continue

        question = case["question"]

        expected_chunks = set(
            case["expected_chunks"]
        )

        print("\n")
        print("#" * 80)
        print(
            f"DEBUG TEST {i}"
        )
        print("#" * 80)

        print(
            "问题:",
            question
        )

        print(
            "正确 Chunk:",
            expected_chunks
        )

        # =====================================================
        # 这里非常重要：
        #
        # 不是我们自己重新写 Context Engineering。
        #
        # 而是直接调用你正式代码：
        #
        # retrieve_context()
        #
        # 所以这次 Debug 和正式测试走的是同一条链。
        # =====================================================

        result = retrieve_context(
            question
        )

        final_docs = result[
            "final_docs"
        ]

        final_chunks = {
            doc.metadata.get("chunk_id")
            for doc in final_docs
        }

        print("\n")
        print("=" * 80)
        print("⑦ FINAL RESULT")
        print("=" * 80)

        print(
            "Expected:",
            expected_chunks
        )

        print(
            "Final:",
            final_chunks
        )

        if expected_chunks & final_chunks:

            print(
                "结果: ✓ 正确 Chunk 保留"
            )

        else:

            print(
                "结果: ✗ 正确 Chunk 丢失"
            )


if __name__ == "__main__":

    main()