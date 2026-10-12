import os

# 必须在导入 transformers / sentence_transformers 之前设置！
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
from config import *
from langchain_classic.memory import ConversationBufferMemory
from rewrite import rewrite_question
from reranker import rerank_documents
from citation import build_reference


# Embedding
embeddings = OllamaEmbeddings(
    model=EMBEDDING_MODEL
)

# Vector DB
db = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embeddings
)

# 创建MMR检索器
retriever = db.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 10,
        "fetch_k": 20
    }
)

# 大模型
llm = ChatOllama(
    model="qwen2.5:1.5b"
)


print("===== MMR版RAG =====")
print("输入 exit 退出\n")
memory = ConversationBufferMemory(
    return_messages=True
)


while True:

    question = input("你：")
    

    if question.lower() == "exit":
        break


    history = memory.load_memory_variables({})

    chat_history = history.get("history", [])
    
    history_text = ""

    for msg in chat_history:
        if msg.type == "human":
            history_text += f"用户：{msg.content}\n"
        elif msg.type == "ai":
            history_text += f"AI：{msg.content}\n"

    # MMR检索
    new_question = rewrite_question(
    question,
    history_text
    )
    docs = retriever.invoke(new_question)


    #重排序
    docs = rerank_documents(
    new_question,
    docs
    )



    print("\n========== MMR检索结果 ==========\n")

    context = ""

    for i, doc in enumerate(docs, 1):

        print(f"第{i}段")
        print(doc.page_content)
        print(doc.metadata)
        print("-" * 60)

        context += doc.page_content + "\n\n"



    prompt = f"""
# 角色

你是一名高校知识库智能助手。

# 工作要求

请严格依据知识库回答。

不得使用自己的知识。

不得猜测。

不得编造。

# 回答规则

1. 只能依据知识库回答。

2. 不允许使用模型自己的知识。

3. 如果知识库没有明确说明，不允许推测。

4. 不允许补充常识。

5. 不允许编造人物、时间、地点、数字。

6. 如果无法回答，请直接回答：

"知识库中没有找到相关内容。"
回答前，请确认：

你的每一句话都能够在知识库中找到依据。

如果不能，请删除这句话。



示例：

用户：
学院院长是谁？

回答：

姓名：高振华
职务：院长

# 回答要求

回答必须：

1. 条理清晰

2. 使用中文

3. 尽量使用知识库原文

4. 不要输出与问题无关内容

------------------

历史聊天：

{history_text}

------------------

知识库内容：

{context}

------------------

用户问题：

{question}
"""



    reference = build_reference(docs)

    memory.chat_memory.add_user_message(question)

    print("\nAI：")

    # 用于保存完整回答
    answer_text = ""

    # 流式输出
    for chunk in llm.stream(prompt):

        print(chunk.content, end="", flush=True)

        answer_text += chunk.content

    print()

    # 保存AI回答到Memory
    memory.chat_memory.add_ai_message(answer_text)

    print(reference)
    print("-" * 60)
  
