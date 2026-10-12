import os

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool


# ============================================================
# 1. Tool
# ============================================================

@tool
def get_weather(city: str) -> str:
    """查询指定城市的天气。"""

    return f"{city}今天晴天，25°C"


# ============================================================
# 2. Tool 集合
# ============================================================

tools = {
    "get_weather": get_weather
}


# ============================================================
# 3. LLM
# ============================================================

llm = ChatOpenAI(
    model="deepseek-v4-flash",
    api_key=os.environ["ANTHROPIC_AUTH_TOKEN"],
    base_url="https://api.deepseek.com"
)

llm_with_tools = llm.bind_tools([
    get_weather
])


# ============================================================
# 4. 创建 Agent State
# ============================================================

state = {
    "messages": [],
    "step": 0,
    "last_tool_result": None,
    "finished": False
}


# ============================================================
# 5. 用户输入
# ============================================================

question = input("请输入问题：")


state["messages"].append(
    HumanMessage(content=question)
)


# ============================================================
# 6. Agent Loop
# ============================================================

while not state["finished"]:

    state["step"] += 1

    print(f"\n========== Agent Step {state['step']} ==========")

    response = llm_with_tools.invoke(
        state["messages"]
    )

    state["messages"].append(response)


    # ========================================================
    # LLM 没有调用 Tool
    # ========================================================

    if not response.tool_calls:

        print("\n========== 最终答案 ==========")
        print(response.content)

        state["finished"] = True

        break


    # ========================================================
    # Tool Calling
    # ========================================================

    for tool_call in response.tool_calls:

        tool_name = tool_call["name"]
        arguments = tool_call["args"]
        tool_call_id = tool_call["id"]

        print("\n调用工具：", tool_name)
        print("工具参数：", arguments)


        selected_tool = tools.get(tool_name)


        if selected_tool is None:

            tool_result = f"错误：不存在工具 {tool_name}"

        else:

            try:

                tool_result = selected_tool.invoke(arguments)

            except Exception as e:

                tool_result = f"工具执行失败：{e}"


        print("工具结果：", tool_result)


        # ====================================================
        # 更新 State
        # ====================================================

        state["last_tool_result"] = tool_result


        tool_message = ToolMessage(
            content=tool_result,
            tool_call_id=tool_call_id
        )

        state["messages"].append(tool_message)


# ============================================================
# 7. 查看最终 State
# ============================================================

print("\n========== 最终 State ==========")

print("Step：", state["step"])
print("最后一次 Tool Result：", state["last_tool_result"])
print("是否完成：", state["finished"])
print("消息数量：", len(state["messages"]))
