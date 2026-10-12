from bm25_retriever import bm25_retriever


question="学校什么时候成立"


docs = bm25_retriever.invoke(
    question
)


for doc in docs:

    print("----------------")

    print(
        doc.page_content[:100]
    )

    print(
        doc.metadata
    )
