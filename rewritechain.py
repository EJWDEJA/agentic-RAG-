from langchain_ollama import ChatOllama


llm = ChatOllama(
    model="qwen2.5:1.5b"
)



def rewrite_question(question, chat_history):

    prompt = f"""
你是一个问题重写助手。

根据聊天历史，把用户当前问题改写成一个完整的问题。

要求：
1. 保留原问题含义
2. 补充省略的信息
3. 只输出改写后的问题
4. 不要回答问题


聊天历史：

{chat_history}


当前问题：

{question}


改写后的完整问题：
"""


    result = llm.invoke(prompt)


    return result.content
