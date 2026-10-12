from hybrid_retriever import hybrid_search


question="学校什么时候成立"


docs = hybrid_search(
    question
)


for i,doc in enumerate(docs):

    print(
        "\n======",
        i
    )


    print(
        doc.page_content[:100]
    )


    print(
        doc.metadata
    )
