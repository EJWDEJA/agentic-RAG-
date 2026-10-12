from token_counter import count_tokens         #按token数多少控制上下文依次加和doc超长跳过当前doc可进行加和识别后续doc


def select_by_token_budget(
    scored_docs,
    max_tokens=3000
):

    selected_docs = []

    total_tokens = 0

    for score, doc in scored_docs:

        doc_tokens = count_tokens(
            doc.page_content
        )

        if total_tokens + doc_tokens > max_tokens:
            continue

        selected_docs.append(doc)

        total_tokens += doc_tokens

    return selected_docs









def count_group_tokens(group):   #用于给一个组内的text计数

    text = "\n".join(
        doc.page_content
        for doc in group.docs
    )

    return count_tokens(text)






def select_groups_by_token_budget(      #组预算判断限制
    groups,
    max_tokens=3000
):

    selected_groups = []

    total_tokens = 0

    for group in groups:

        group_tokens = count_group_tokens(
            group
        )

        if total_tokens + group_tokens > max_tokens:
            continue             #此处的continue表示结束本次循环后续不用执行break是结束所有循环直接退出他们在while 和for循环都适用

        selected_groups.append(group)

        total_tokens += group_tokens

    return selected_groups











def calculate_group_utility(group):      #计算单个token价值即utility

    tokens = count_group_tokens(group)

    if tokens == 0:
        return 0

    return group.score / tokens




def sort_groups_by_utility(groups):      #按照utility排序

    return sorted(
        groups,
        key=calculate_group_utility,
        reverse=True
    )





def select_groups_by_utility_budget(          #按照utility排序结果选择group
    groups,
    max_tokens=3000
):

    sorted_groups = sort_groups_by_utility(
        groups
    )

    selected_groups = []

    total_tokens = 0

    for group in sorted_groups:

        group_tokens = count_group_tokens(
            group
        )

        if total_tokens + group_tokens > max_tokens:
            continue

        selected_groups.append(group)

        total_tokens += group_tokens

    return selected_groups





def rank_groups(groups):    #帮助merged transistive group以后的groups进行socre排序     #因为他是取两个组最大socre作为新的socre使用rerank出来以后的排序会乱

    return sorted(
        groups,
        key=lambda group: group.score,
        reverse=True
    )







def select_groups(
    groups,
    max_tokens=3000
):

    ranked_groups = rank_groups(groups)

    selected_groups = select_groups_by_token_budget(
        ranked_groups,
        max_tokens
    )

    return selected_groups

