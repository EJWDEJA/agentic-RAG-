from langchain_core.documents import Document

from context_selection import select_by_token_budget


doc_a = Document(
    page_content="学校成立于1950年。" * 130
)

doc_b = Document(
    page_content="计算机科学与技术学院。" * 300
)

doc_c = Document(
    page_content="学校位于某某地区。" * 20
)


scored_docs = [
    (0.95, doc_a),
    (0.90, doc_b),
    (0.80, doc_c)
]


docs = select_by_token_budget(
    scored_docs,
    max_tokens=2000
)


print("\n===== Selected Docs =====")

for doc in docs:

    print(
        doc.page_content[:30]
    )
