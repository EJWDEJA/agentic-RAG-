from context_group import ContextGroup
from context_selection import rank_groups


group_a = ContextGroup(
    docs=[],
    score=0.72
)

group_b = ContextGroup(
    docs=[],
    score=0.91
)

group_c = ContextGroup(    
    docs=[],
    score=0.85
)


groups = [
    group_a,
    group_b,
    group_c
]


ranked_groups = rank_groups(groups)


print("===== Group Ranking =====")

for group in ranked_groups:

    print(
        "score:",
        group.score
    )
