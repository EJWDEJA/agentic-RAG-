import os

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool


# ============================================================
# 1. Tool A：查询天气
# ============================================================

@tool
def get_weather(city: str) -> str:
    """查询指定城市的天气和当前温度。"""

    return f"{city}今天晴天，当前温度25°C"


# ============================================================
# 2. Tool B：数学计算
# ============================================================

@tool
def calculate(expression: str) -> str:
    """执行数学计算，例如：25 + 5。"""

    try:
        result = eval(expression)
        return str(result)

    except Exception as e:
        return f"计算失败：{e}"


# ============================================================
# 3. 工具字典
# ============================================================

tools = {
    "get_weather": get_weather,
    "calculate": calculate
}


# ============================================================
# 4. 创建 LLM
# ============================================================

llm = ChatOpenAI(
    model="deepseek-v4-flash",
    api_key=os.environ["ANTHROPIC_AUTH_TOKEN"],
    base_url="https://api.deepseek.com"
)


# ============================================================
# 5. 绑定多个 Tool
# ============================================================

llm_with_tools = llm.bind_tools([
    get_weather,
    calculate
])


# ============================================================
# 6. 用户问题
# ============================================================

question = input("请输入问题：")

messages = [
    HumanMessage(content=question)
]


# ============================================================
# 7. Agent Loop
# ============================================================

while True:

    print("\n========== LLM 思考 ==========")

    response = llm_with_tools.invoke(messages)

    messages.append(response)

    print("\nAIMessage：")
    print(response)

    print("\nTool Calls：")
    print(response.tool_calls)


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


        # ====================================================
        # Dispatcher
        # ====================================================

        selected_tool = tools.get(tool_name)


        if selected_tool is None:

            tool_result = f"错误：不存在工具 {tool_name}"

        else:

            try:

                tool_result = selected_tool.invoke(arguments)

            except Exception as e:

                tool_result = f"工具执行失败：{e}"


        print("\n========== 工具结果 ==========")
        print(tool_result)


        # ====================================================
        # Tool Result → ToolMessage
        # ====================================================

        tool_message = ToolMessage(
            content=tool_result,
            tool_call_id=tool_call_id
        )

        messages.append(tool_message)