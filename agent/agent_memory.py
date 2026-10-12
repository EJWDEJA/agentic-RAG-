import os

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage


# ============================================================
# 1. 创建 LLM
# ============================================================

llm = ChatOpenAI(
    model="deepseek-v4-flash",
    api_key=os.environ["ANTHROPIC_AUTH_TOKEN"],
    base_url="https://api.deepseek.com"
)


# ============================================================
# 2. Memory
# ============================================================

memory = []


# ============================================================
# 3. 连续对话
# ============================================================

while True:

    question = input("\n用户：")

    if question == "exit":
        break


    # ========================================================
    # 把当前用户消息加入 Memory
    # ========================================================

    memory.append(
        HumanMessage(content=question)
    )


    # ========================================================
    # 把整个 Memory 交给 LLM
    # ========================================================

    response = llm.invoke(memory)


    # ========================================================
    # 把 LLM 回复保存进 Memory
    # ========================================================

    memory.append(
        AIMessage(content=response.content)
    )


    print("Agent：", response.content)


# ============================================================
# 4. 查看 Memory
# ============================================================

print("\n========== Memory ==========")

for message in memory:

    print(
        type(message).__name__,    #把类型变成字符串    
        "：",
        message.content
    )