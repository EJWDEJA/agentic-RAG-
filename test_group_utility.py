from langchain_core.documents import Document

from context_group import ContextGroup

from context_selection import (
    count_group_tokens,
    calculate_group_utility,
    sort_groups_by_utility,
    select_groups_by_utility_budget
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
    score=0.90
)


group_c = ContextGroup(
    docs=[
        create_doc("学生培养信息。" * 30)
    ],
    score=0.85
)


groups = [
    group_a,
    group_b,
    group_c
]


print("\n===== Original Groups =====")

for i, group in enumerate(
    groups,
    start=1
):

    tokens = count_group_tokens(group)

    utility = calculate_group_utility(
        group
    )

    print(
        f"Group {i}"
    )

    print(
        "Score:",
        group.score
    )

    print(
        "Tokens:",
        tokens
    )

    print(
        "Utility:",
        utility
    )


sorted_groups = sort_groups_by_utility(
    groups
)


print("\n===== Utility Sorted =====")

for i, group in enumerate(
    sorted_groups,
    start=1
):

    print(
        f"Group {i}"
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
        "Utility:",
        calculate_group_utility(group)
    )


selected_groups = select_groups_by_utility_budget(
    groups,
    max_tokens=2000
)


print("\n===== Selected Groups =====")

total_tokens = 0

for i, group in enumerate(
    selected_groups,
    start=1
):

    tokens = count_group_tokens(group)

    total_tokens += tokens

    print(
        f"Group {i}"
    )

    print(
        "Score:",
        group.score
    )

    print(
        "Tokens:",
        tokens
    )

print(
    "\nTotal Tokens:",
    total_tokens
)
