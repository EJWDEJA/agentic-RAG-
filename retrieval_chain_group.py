import os       #这是针对group(类)的检索处理上下文管理

# 必须在导入 transformers / sentence_transformers 之前设置！
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


import pickle

from langchain_core.runnables import RunnableLambda

from hybrid_retriever import hybrid_search

from context_engineering import build_context

from citation_chain import get_sources


# 加载全部 Document
with open(
    "documents.pkl",
    "rb"
) as f:

    all_docs = pickle.load(f)


def retrieve_context(question):

    # ==============================
    # 1. Hybrid Search
    # ==============================

    hybrid_docs = hybrid_search(
        question
    )


    # ==============================
    # 2. Context Engineering
    # ==============================

    selected_groups = build_context(
        question,
        hybrid_docs,
        all_docs,
        max_tokens=3000
    )


    # ==============================
    # 3. 提取最终 Document
    # ==============================

    final_docs = []

    for group in selected_groups:

        for doc in group.docs:

            final_docs.append(doc)


    # ==============================
    # 4. 拼接 Context
    # ==============================

    context = ""

    for doc in final_docs:

        context += doc.page_content + "\n"


    # ==============================
    # 5. Sources
    # ==============================

    sources = get_sources(
        final_docs
    )


    # ==============================
    # 6. 返回
    # ==============================

    return {
        "context": context,
        "sources": sources,
    "final_docs": final_docs     #用于context engening评估
    }


retrieval_chain = RunnableLambda(
    retrieve_context
)
