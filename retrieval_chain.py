import os           #这是针对chunk块（doc）的检索处理上下文管理

# 必须在导入 transformers / sentence_transformers 之前设置！
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from langchain_core.runnables import RunnableLambda
from reranker import rerank_documents
from citation_chain import get_sources
from hybrid_retriever import hybrid_search
from context_engineering import deduplicate_documents
from context_engineering import restore_chunk_order
from context_engineering import merge_adjacent_chunks
from context_engineering import expand_neighbor_chunks
import pickle
from context_length import limit_context_tokens


with open("documents.pkl", "rb") as f:
    all_docs = pickle.load(f)





def retrieve_context(question):

       # 1. Hybrid Retrieval
    # =====================

    docs = hybrid_search(
        question
    )

    # Reranker排序
    docs = rerank_documents(
    question,
    docs,
    top_k=3           
    )

    ##
    print("\n===== Reranker 后 =====")

    for doc in docs:
        print(doc.metadata.get("chunk_id"))
    ##
    
    docs = expand_neighbor_chunks(
    docs,
    all_docs,
    neighbor_size=1
    )

    ##
    print("\n===== Expansion 后 =====")
    for doc in docs:
        print(doc.metadata.get("chunk_id"))
    ##


    docs = deduplicate_documents(docs)


    ##
    print("\n===== Dedup 后 =====")
    for doc in docs:
        print(doc.metadata.get("chunk_id"))
     ##


    docs = restore_chunk_order(docs)


    # ===== 临时测试：检查 Restore 后的完整 metadata =====

    print("\n===== Restore 后详细信息 =====")

    for doc in docs:

        print(
        "source =", repr(doc.metadata.get("source")),
        "| sheet =", repr(doc.metadata.get("sheet")),
        "| chunk_id =", repr(doc.metadata.get("chunk_id"))
        )


    ##
    print("\n===== Restore 后 =====")
    for doc in docs:
        print(doc.metadata.get("chunk_id"))
    ##



    docs = merge_adjacent_chunks(docs)


    docs = limit_context_tokens(
    docs,
    max_tokens=3000
    )



    # 拼接文本
    

    context = ""

    for doc in docs:
        context += doc.page_content + "\n"



    print("\n===== Context Engineering 最终顺序 =====")

    for doc in docs:
        print(
        "chunk_id:",
        doc.metadata.get("chunk_id")
        )

    sources = get_sources(docs)


    return {
        "context": context,
        "sources": sources,
        "final_docs": docs      #这部分是为了测试agentic_rag_10_evidence_gap.py后加的9.25前要使用可以删掉
    }


retrieval_chain = RunnableLambda(
    retrieve_context
)
