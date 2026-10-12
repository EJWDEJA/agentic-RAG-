from reranker import rerank_documents_with_scores

from context_group import (
    create_context_groups,
    merge_overlapping_groups_transitively
)

from context_selection import select_groups
from context_group import sort_all_group_docs

from context_compression import compress_group








def check_duplicate_chunks(docs):        #测试document.pkl里面是否存在重复doc

    seen = set()

    print("\n===== all_docs 重复 Chunk =====")

    for doc in docs:

        key = (
            doc.metadata.get("source"),
            doc.metadata.get("sheet"),
            doc.metadata.get("chunk_id")
        )

        if key in seen:
            print("发现重复:", key)

        else:
            seen.add(key)







def expand_neighbor_chunks(            #用于扩展rerank里面输出的core doc
    selected_docs,
    all_docs,
    neighbor_size=1
):

    expanded_docs = []

    for doc in selected_docs:

        source = doc.metadata.get("source")
        sheet = doc.metadata.get("sheet")
        chunk_id = doc.metadata.get("chunk_id")

        for candidate in all_docs:

            candidate_source = candidate.metadata.get("source")
            candidate_sheet = candidate.metadata.get("sheet")
            candidate_chunk_id = candidate.metadata.get("chunk_id")

            if (
                source == candidate_source
                and sheet == candidate_sheet
                and abs(candidate_chunk_id - chunk_id) <= neighbor_size
            ):

                expanded_docs.append(candidate)

    return expanded_docs





def deduplicate_documents(docs):      #删除重复项

    seen = set()    #集合去重
    unique_docs = []    

    for doc in docs:

        content = doc.page_content

        if content not in seen:
            seen.add(content)
            unique_docs.append(doc)

    return unique_docs         #前面的RRF其实已经通过判断原始文本去重了这一步重复没什么用留着学写法吧







def restore_chunk_order(docs):              #还原块顺序 
    return sorted(
        docs,    #待排序对象
        key=lambda doc: (                    #排序依据
            doc.metadata.get("source", ""),  #""默认值防止没有source时报错，可输出""
            doc.metadata.get("sheet", ""),     
            int(doc.metadata.get("chunk_id", 0))
        )

    )



def merge_adjacent_chunks(docs):            #合并相邻块

    if not docs:
        return []                           #防报错

    merged_docs = []

    current_doc = docs[0]              #给变量currennt dooc赋值docs列表的第一个元素，他不是创建了列表创建列表应该是current doc = [docs[0]]

    for next_doc in docs[1:]:          #docs[1:]从1开始提取docs里面除了第一个元素以外的all元素组成一个没有命名变量而是直接交给for的新列表

        current_source = current_doc.metadata.get("source")
        next_source = next_doc.metadata.get("source")

        current_sheet = current_doc.metadata.get("sheet")
        next_sheet = next_doc.metadata.get("sheet")

        current_chunk_id = current_doc.metadata.get("chunk_id")
        next_chunk_id = next_doc.metadata.get("chunk_id")

        if (
            current_source == next_source
            and current_sheet == next_sheet
            and current_chunk_id + 1 == next_chunk_id    
        ):

            current_doc.page_content += "\n" + next_doc.page_content        #这只是对原始文本的加和且范围只是两个chunk id，取前一个chunk ID作为整体的chunk ID，35 36 37合并完以后是35 37 

        else:

            merged_docs.append(current_doc)
            current_doc = next_doc

    merged_docs.append(current_doc)         #给最后一个无next source可比较的current docs兜底

    return merged_docs









def build_context(          #串联完整检索链

    question,
    hybrid_docs,
    all_docs,
    max_tokens=3000
):

    scored_docs = rerank_documents_with_scores(
        question,
        hybrid_docs,
        top_k=3
    )

    groups = create_context_groups(
        scored_docs,
        all_docs,
        neighbor_size=1
    )

    groups = merge_overlapping_groups_transitively(
        groups
    )

    groups = sort_all_group_docs(
        groups
    )

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

    

    print("\n===== BEFORE SELECT_GROUPS =====")

    for i, group in enumerate(
        compressed_groups,
        start=1
    ):
        print(
            "Group",
            i,
            "score=",
            group.score,
            "chunks=",
            [
            doc.metadata.get("chunk_id")
            for doc in group.docs
            ]
        )





    selected_groups = select_groups(
        compressed_groups,
        max_tokens=max_tokens
    )

    return selected_groups
