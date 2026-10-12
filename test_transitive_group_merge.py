from langchain_core.documents import Document

from context_group import (
    ContextGroup,
    merge_overlapping_groups_transitively
)


def create_doc(chunk_id):

    return Document(
        page_content=f"Chunk {chunk_id}",
        metadata={
            "source": "test.txt",
            "sheet": None,
            "chunk_id": chunk_id
        }
    )


group_a = ContextGroup(
    docs=[
        create_doc(10),
        create_doc(11)
    ],
    score=0.90
)


group_b = ContextGroup(
    docs=[
        create_doc(11),
        create_doc(12)
    ],
    score=0.80
)


group_c = ContextGroup(
    docs=[
        create_doc(12),
        create_doc(13)
    ],
    score=0.70
)


groups = [
    group_a,
    group_b,
    group_c
]


merged_groups = merge_overlapping_groups_transitively(
    groups
)


print("\n===== Merge 后 =====")


for i, group in enumerate(
    merged_groups,
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
        "Chunks:",
        [
            doc.metadata.get("chunk_id")
            for doc in group.docs
        ]
    )
