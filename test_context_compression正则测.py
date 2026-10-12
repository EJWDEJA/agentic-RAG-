from context_compression import split_sentences


# ==============================
# 测试1：中文普通句子
# ==============================

text1 = """
学校成立于1950年。
学校位于东京。
学校共有20个学院！
学校设有图书馆？
"""


print("\n===== Test 1 =====")

sentences = split_sentences(text1)

for i, sentence in enumerate(
    sentences,
    start=1
):

    print(i, sentence)


# ==============================
# 测试2：邮箱
# ==============================

text2 = """
姓名：周浩
邮箱：zhouhao@university.edu.cn
电话：010-88882018
"""


print("\n===== Test 2 =====")

sentences = split_sentences(text2)

for i, sentence in enumerate(
    sentences,
    start=1
):

    print(i, sentence)


# ==============================
# 测试3：英文句号 + 邮箱
# ==============================

text3 = """
The university was founded in 1950.
Contact us at zhouhao@university.edu.cn
The campus is located in Tokyo.
"""


print("\n===== Test 3 =====")

sentences = split_sentences(text3)

for i, sentence in enumerate(
    sentences,
    start=1
):

    print(i, sentence)


# ==============================
# 测试4：真实知识库 Chunk
# ==============================

import pickle


with open(
    "documents.pkl",
    "rb"
) as f:

    all_docs = pickle.load(f)


print("\n===== Test 4: Real Chunks =====")


for chunk_id in [38, 39, 40]:

    for doc in all_docs:

        if doc.metadata.get("chunk_id") == chunk_id:

            print(
                f"\n----- Chunk {chunk_id} -----"
            )

            sentences = split_sentences(
                doc.page_content
            )

            for i, sentence in enumerate(
                sentences,
                start=1
            ):

                print(
                    i,
                    sentence
                )

            break
