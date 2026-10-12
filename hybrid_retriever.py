from retriever import get_retriever
from bm25_retriever import bm25_retriever




retriever = get_retriever()




def hybrid_search(question):

    # =====================
    # 1. 向量检索
    # =====================

    vector_docs = retriever.invoke(
        question
    )             #一般默认返回的是一个列表，列表元素顺序就是排列顺序


    # =====================
    # 2. BM25检索
    # =====================

    bm25_docs = bm25_retriever.invoke(
        question
    )              #一般默认返回的是一个列表，列表元素顺序就是排列顺序


    # =====================
    # 3. RRF
    # =====================

    rrf_scores = {}

    documents = {}


    k = 60


    # Vector排名

    for rank, doc in enumerate(    #枚举负责编号    rank随意熟悉第一个位置是编号存放位置，doc随意第二个位置是数据Doc...存放位置 
        vector_docs,     #要编号对象     
        start=1       #从1开始编号
    ):               #这不是是对列表元素的编号不具备排序能力

        doc_id = doc.page_content       

        score = 1 / (
            k + rank
        )

        rrf_scores[doc_id] = (
            rrf_scores.get(
                doc_id,
                0
            )
            +
            score
        )                     # rrf_scores[doc_id] = (xxxxx)   这种事保存键值对进入字典的rrf_scores这个是字典【】这个里面放key，（）这个里面放values    保存id:score这种键值对

        documents[doc_id] = doc         #这个是同理保存id:doc症状键值对


    # BM25排名

    for rank, doc in enumerate(
        bm25_docs,
        start=1
    ):

        doc_id = doc.page_content

        score = 1 / (
            k + rank
        )

        rrf_scores[doc_id] = (
            rrf_scores.get(
                doc_id,
                0
            )
            +
            score
        )          #对已经被MMR修过的字典二次修改

        documents[doc_id] = doc


    # =====================
    # 4. 按RRF分数排序
    # =====================

    ranked_ids = sorted(             #sorted()对字典排序时候默认遍历key输出key的列表
        rrf_scores,
        key=rrf_scores.get,           #规定按照什么排序即values排序
        reverse=True           #从大到小
    )


    # =====================
    # 5. 返回排序后的Document
    # =====================

    return [
        documents[doc_id]     #待输出至列表的对象
        for doc_id in ranked_ids
    ]                #这是一种列表简写不开doc包含在for里面的

