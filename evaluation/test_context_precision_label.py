import pickle      #用于读取并输出核心chunk的相邻chunk块，方便后续的人工选择最相关chunk用于精确度计算


# =========================
# 加载全部 Chunk
# =========================

with open("documents.pkl", "rb") as f:
    all_docs = pickle.load(f)


# =========================
# Golden Dataset
# =========================

golden_dataset = [
    {
        "question": "T2019033是谁？",
        "correct_chunk_id": 38
    },
    {
        "question": "王建国的职称是什么？",
        "correct_chunk_id": 38
    },
    {
        "question": "王建国的研究方向是什么？",
        "correct_chunk_id": 38
    },
    {
        "question": "赵雪丽的研究方向是什么？",
        "correct_chunk_id": 39
    },
    {
        "question": "谁是计算机科学与技术2024级2班的班主任？",
        "correct_chunk_id": 40
    },
    {
        "question": "孙艺萌负责哪些工作？",
        "correct_chunk_id": 41
    }
]


# =========================
# 打印指定 Chunk 及邻居
# =========================

def print_chunk_context(
    docs,
    center_chunk_id,
    neighbor_size=2
):

    start = max(
        0,
        center_chunk_id - neighbor_size
    )

    end = center_chunk_id + neighbor_size + 1

    print("\n" + "=" * 100)

    for doc in docs:

        chunk_id = doc.metadata.get("chunk_id")

        if start <= chunk_id < end:

            if chunk_id == center_chunk_id:
                label = "【核心 Chunk】"
            else:
                label = "【邻居 Chunk】"

            print("\n" + "-" * 100)
            print(label)
            print("chunk_id:", chunk_id)
            print("source:", doc.metadata.get("source"))
            print("sheet:", doc.metadata.get("sheet"))
            print("content:")
            print(doc.page_content)


# =========================
# 开始标注辅助测试
# =========================

for item in golden_dataset:

    question = item["question"]
    correct_chunk_id = item["correct_chunk_id"]

    print("\n\n")
    print("#" * 100)
    print("问题：", question)
    print("核心 Chunk：", correct_chunk_id)
    print("#" * 100)

    print_chunk_context(
        all_docs,
        correct_chunk_id,
        neighbor_size=2
    )
