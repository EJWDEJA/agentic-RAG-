
from sentence_transformers import CrossEncoder       #仅仅返回doc列表的rerank

# 加载重排序模型
reranker = CrossEncoder(
    "BAAI/bge-reranker-v2-m3"
)


def rerank_documents(question, docs, top_k=3):    #这里的top k=3是默认值，实际以实际传参为标准，无实际传参则使用默认传参

    
    pairs = []

    for doc in docs:
        pairs.append([question, doc.page_content])

    # 计算相关性分数
    scores = reranker.predict(pairs)

    print("\n========== Reranker得分 ==========\n")

    result = list(zip(scores, docs))

    for score, doc in result:      #这部分单纯打印检索结构和分数没有固定条数限制完全删除不影响程序效果
        print(f"分数：{score:.4f}")
        print(doc.page_content[:100])
        print("-" * 60)

    # 按分数排序
    result.sort(
        key=lambda x: x[0],
        reverse=True
    )
    

   
    # 返回排序后的文档
    return [
    doc
    for score, doc in result[:top_k]       #这部分负责规定返回最可能的3个结果，应用于后续prompt
    ]









def rerank_documents_with_scores(                    #返回一个个scores和 docs组成元组的列表
    question,
    docs,
    top_k=3
):

    pairs = []

    for doc in docs:

        pairs.append(
            (
                question,
                doc.page_content
            )
        )

    scores = reranker.predict(pairs)

    result = list(zip(scores, docs))

    result.sort(
        key=lambda x: x[0],
        reverse=True
    )

    return result[:top_k]

