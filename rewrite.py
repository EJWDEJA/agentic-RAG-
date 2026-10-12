from langchain_ollama import ChatOllama


llm = ChatOllama(
    model="qwen2.5:1.5b"
)



def rewrite_question(question, history):

    prompt = f"""

你是一个问题改写助手。

你的任务：
根据历史聊天，把用户当前问题改写成完整问题。


历史聊天：

{history}


当前问题：

{question}


只输出改写后的问题。

"""


    response = llm.invoke(prompt)


    return response.content
