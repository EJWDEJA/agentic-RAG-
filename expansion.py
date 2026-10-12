from langchain_ollama import ChatOllama
from config import *

llm = ChatOllama(
    model=LLM_MODEL
)


def expand_query(question):

    prompt = f"""
请针对下面的问题，生成3个不同表达方式。

要求：

1. 保持原意不变。
2. 更适合知识库检索。
3. 每行一句。
4. 不要解释。

用户问题：

{question}
"""

    answer = llm.invoke(prompt)

    return answer.content.split("\n")
