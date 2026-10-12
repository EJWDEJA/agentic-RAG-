import os               #体现agent loop的错误重试自我矫正

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool


# ============================================================
# 1. 定义 Tool
# ============================================================

@tool
def get_weather(city: str) -> str:
    """查询指定城市的天气。"""

    # 模拟一个工具执行错误
    if city == "北京":
        raise ValueError("天气服务暂时不可用，请重新查询")

    return f"{city}今天晴天，25度"


# ============================================================
# 2. 建立工具字典
# ============================================================

tools = {
    "get_weather": get_weather
}


# ============================================================
# 3. 创建 LLM
# ============================================================

llm = ChatOpenAI(
    model="deepseek-v4-flash",
    api_key=os.environ["ANTHROPIC_AUTH_TOKEN"],
    base_url="https://api.deepseek.com"
)


# ============================================================
# 4. 绑定 Tool
# ============================================================

llm_with_tools = llm.bind_tools([
    get_weather
])


# ============================================================
# 5. 获取用户问题
# ============================================================

question = input("请输入问题：")

messages = [
    HumanMessage(content=question)
]


# ============================================================
# 6. Agent Loop
# ============================================================

while True:

    print("\n========== LLM 思考 ==========")

    response = llm_with_tools.invoke(messages)

    messages.append(response)


    # ========================================================
    # 没有 Tool Call
    # ========================================================

    if not response.tool_calls:

        print("\n========== 最终答案 ==========")
        print(response.content)

        break


    # ========================================================
    # 处理 Tool Call
    # ========================================================

    for tool_call in response.tool_calls:

        tool_name = tool_call["name"]
        arguments = tool_call["args"]
        tool_call_id = tool_call["id"]

        print("\n========== 调用工具 ==========")
        print("工具名称：", tool_name)
        print("工具参数：", arguments)


        tool = tools.get(tool_name)


        if tool is None:

            tool_result = "错误：不存在这个工具。"

        else:

            try:

                tool_result = tool.invoke(arguments)

            except Exception as e:

                tool_result = f"工具执行失败：{e}"


        print("\n========== 工具结果 ==========")
        print(tool_result)


        # ====================================================
        # 错误/成功结果统一交给 LLM
        # ====================================================

        tool_message = ToolMessage(
            content=tool_result,
            tool_call_id=tool_call_id
        )

        messages.append(tool_message)
