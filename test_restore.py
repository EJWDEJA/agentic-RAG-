import pickle

from context_engineering import restore_chunk_order


with open("documents.pkl", "rb") as f:
    all_docs = pickle.load(f)


docs = all_docs[30:45][::-1]


print("\n===== 测试输入 =====")

for doc in docs:

    print(
        "source =", repr(doc.metadata.get("source")),
        "| sheet =", repr(doc.metadata.get("sheet")),
        "| chunk_id =", repr(doc.metadata.get("chunk_id"))
    )


docs = restore_chunk_order(docs)


print("\n===== Restore 后 =====")

for doc in docs:

    print(
        "source =", repr(doc.metadata.get("source")),
        "| sheet =", repr(doc.metadata.get("sheet")),
        "| chunk_id =", repr(doc.metadata.get("chunk_id"))
    )
