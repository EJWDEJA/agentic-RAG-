from langchain_core.documents import Document

from context_group import ContextGroup

from context_selection import (
    count_group_tokens,
    select_groups_by_token_budget
)


def create_doc(text):

    return Document(
        page_content=text,
        metadata={}
    )


group_a = ContextGroup(
    docs=[
        create_doc("学校历史信息。" * 100),
        create_doc("学校发展信息。" * 100)
    ],
    score=0.95
)


group_b = ContextGroup(
    docs=[
        create_doc("计算机学院信息。" * 50)
    ],
    score=0.88
)


group_c = ContextGroup(
    docs=[
        create_doc("学生培养信息。" * 300)
    ],
    score=0.80
)


groups = [
    group_a,
    group_b,
    group_c
]


print("\n===== Group Token Count =====")

for i, group in enumerate(
    groups,
    start=1
):

    print(
        "Group",
        i,
        "Score:",
        group.score,
        "Tokens:",
        count_group_tokens(group)
    )


selected_groups = select_groups_by_token_budget(
    groups,
    max_tokens=2000
)


print("\n===== Selected Groups =====")

for i, group in enumerate(
    selected_groups,
    start=1
):

    print(
        "Group",
        i,
        "Score:",
        group.score
    )

    print(
        "Tokens:",
        count_group_tokens(group)
    )
