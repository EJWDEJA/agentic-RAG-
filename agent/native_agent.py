import os

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, ToolMessage


# =========================
# 1. 定义多个工具
# =========================

def get_weather(city):
    return f"{city}今天晴天，25度"


def calculate(expression):
    return f"计算结果是：{eval(expression)}"


def get_school_info(question):
    return f"学校知识库查询结果：{question}，目前没有找到相关信息。"


# =========================
# 2. 工具注册表
# =========================

tools = {
    "get_weather": get_weather,
    "calculate": calculate,
    "get_school_info": get_school_info
}


# =========================
# 3. 创建 DeepSeek
# =========================

llm = ChatOpenAI(
    model="deepseek-v4-flash",
    api_key=os.environ["ANTHROPIC_AUTH_TOKEN"],
    base_url="https://api.deepseek.com"
)


# =========================
# 4. 告诉 DeepSeek有哪些工具
# =========================

llm_with_tools = llm.bind_tools(
    [
        get_weather,
        calculate,
        get_school_info
    ]
)


# =========================
# 5. 用户问题
# =========================

question = input("请输入问题：")


messages = [
    HumanMessage(content=question)
]


# =========================
# 6. Agent Loop
# =========================

while True:

    print()
    print("========== LLM 决策 ==========")

    response = llm_with_tools.invoke(messages)

    # 保存 LLM 的消息
    messages.append(response)


    # =========================
    # 没有 Tool Call
    # =========================

    if not response.tool_calls:

        print("最终答案：")
        print(response.content)

        break


    # =========================
    # 有 Tool Call
    # =========================

    for tool_call in response.tool_calls:

        tool_name = tool_call["name"]

        arguments = tool_call["args"]

        tool_call_id = tool_call["id"]


        print("调用工具：", tool_name)

        print("工具参数：", arguments)


        # =========================
        # Dispatcher
        # =========================

        tool = tools.get(tool_name)


        if tool is None:

            tool_result = "错误：不存在这个工具。"

        else:

            try:

                tool_result = tool(**arguments)

            except Exception as e:

                tool_result = f"工具执行失败：{e}"


        print("工具结果：", tool_result)


        # =========================
        # 返回 Tool Result
        # =========================

        tool_message = ToolMessage(
            content=tool_result,
            tool_call_id=tool_call_id
        )


        messages.append(tool_message)
