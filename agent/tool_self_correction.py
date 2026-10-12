import os

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool


# ============================================================
# 1. 定义 Tool
# ============================================================

@tool
def get_weather(city: str) -> str:
    """查询指定城市的天气。城市参数应使用标准城市名称，例如：北京、上海。"""

    # 模拟参数格式错误
    if city == "北京市":
        raise ValueError(
            "城市参数格式错误：请将“北京市”修正为“北京”后重新调用。"
        )

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
# 6. 重试控制
# ============================================================

retry_count = 0
max_retries = 3


# ============================================================
# 7. Agent Loop
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

                # Structured Tool
                tool_result = tool.invoke(arguments)

            except Exception as e:

                retry_count += 1

                print("\n========== Tool 执行异常 ==========")
                print("错误：", e)
                print("当前失败次数：", retry_count)
                print("最大允许失败次数：", max_retries)

                # --------------------------------------------
                # 达到最大重试次数
                # --------------------------------------------

                if retry_count >= max_retries:

                    tool_result = (
                        f"工具连续执行失败 {max_retries} 次，"
                        "Agent 停止继续调用工具。"
                    )

                else:

                    # ----------------------------------------
                    # 把错误信息交给 LLM
                    # ----------------------------------------

                    tool_result = f"工具执行失败：{e}"


        print("\n========== 工具结果 ==========")
        print(tool_result)


        # ====================================================
        # 无论 Tool 成功还是失败
        # 都必须先创建并加入 ToolMessage
        # ====================================================

        tool_message = ToolMessage(
            content=tool_result,
            tool_call_id=tool_call_id
        )

        messages.append(tool_message)


    # ========================================================
    # 达到最大重试次数后，正常结束
    # ========================================================

    if retry_count >= max_retries:

        print("\n========== 达到最大重试次数 ==========")
        print("Agent 停止运行。")

        break
