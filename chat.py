from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
from config import *

# 加载Embedding模型
embeddings = OllamaEmbeddings(
    model=EMBEDDING_MODEL
)

# 加载向量数据库
db = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embeddings
)

# 加载聊天模型
llm = ChatOllama(
    model="qwen2.5:1.5b"
)

# 相似度阈值（距离越小越相似）
THRESHOLD = 0.8

print("===== 本地RAG聊天 =====")
print("输入 exit 退出\n")

while True:
    question = input("你：")

    if question.lower() == "exit":
        break

    # 检索最相关的3段内容
    docs = db.similarity_search_with_score(question, k=3)

    print("\n========== 检索结果 ==========\n")

    context = ""
    valid_docs = []

    for i, (doc, score) in enumerate(docs, 1):

        print(f"第{i}段")
        print(f"距离：{score:.4f}")
        print(doc.page_content)
        print("-" * 60)

        # 只有距离小于阈值才认为有效
        if score < THRESHOLD:
            valid_docs.append(doc)
            context += doc.page_content + "\n\n"

    # 如果没有任何有效知识，直接结束本轮        
    if len(valid_docs) == 0:
        print("\nAI：")
        print("知识库中没有找到内容。")
        print("-" * 60)
        continue


    prompt = f"""
你是一名知识库助手。

请严格依据下面提供的知识回答。

如果知识库没有答案，请回答：

"知识库中没有找到相关内容。"

知识：

{context}

问题：

{question}
"""

    answer = llm.invoke(prompt)

    print("\nAI：")
    print(answer.content)
    print("-" * 60)