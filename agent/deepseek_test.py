import os

from langchain_openai import ChatOpenAI


llm = ChatOpenAI(
    model="deepseek-v4-flash",
    api_key=os.environ["ANTHROPIC_AUTH_TOKEN"],
    base_url="https://api.deepseek.com"
)


response = llm.invoke(
    "你好，请简单介绍一下你自己。"
)


print(response.content)
