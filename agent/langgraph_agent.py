
import os

from typing import TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool

from langgraph.graph import StateGraph, START, END


# ============================================================
# 1. 定义 Tool
# ============================================================

@tool
def get_weather(city: str) -> str:
    """查询指定城市的天气。"""

    weather_data = {
        "北京": "晴天，25°C",
        "上海": "多云，28°C",
        "广州": "小雨，30°C"
    }

    return weather_data.get(city, f"{city}：暂无天气数据")    #存在city抓取不存在的时候返回 f"{city}：暂无天气数据"


# ============================================================
# 2. 定义 LLM
# ============================================================

llm = ChatOpenAI(
    model="deepseek-v4-flash",
    api_key=os.environ["ANTHROPIC_AUTH_TOKEN"],
    base_url="https://api.deepseek.com"
)


tools = [get_weather]

llm_with_tools = llm.bind_tools(tools)


# ============================================================
# 3. 定义 State
# ============================================================

class State(TypedDict):
    messages: list


# ============================================================
# 4. LLM Node
# ============================================================

def llm_node(state: State):

    print("\n========== LLM Node ==========")

    response = llm_with_tools.invoke(state["messages"])    #调用llm返回的reaponse本生就是一个单条的AImessage类型数据不需要像Toolmessage一样手动的封装tool reault和id为Toolreault格式的数据

    print("LLM 输出：")
    print(response)

    return {
        "messages": state["messages"] + [response]   #解包双边列表生成性列表不破坏1原有列表，作为新字典取返回并修改全局状态
    }


# ============================================================
# 5. Tool Node
# ============================================================

def tool_node(state: State):

    print("\n========== Tool Node ==========")

    last_message = state["messages"][-1]     #从全局状态里面取出messages列表的最后一个元素

    tool_messages = []

    for tool_call in last_message.tool_calls:

        tool_name = tool_call["name"]
        arguments = tool_call["args"]
        tool_call_id = tool_call["id"]

        print("调用 Tool：", tool_name)
        print("参数：", arguments)

        tool = {
            "get_weather": get_weather
        }.get(tool_name)     #获取工具函数并赋值给tool

        if tool is None:

            tool_result = "错误：不存在这个工具。"

        else:

            try:
                tool_result = tool.invoke(arguments)

            except Exception as e:

                tool_result = f"工具执行失败：{e}"

        tool_messages.append(
            ToolMessage(
                content=tool_result,
                tool_call_id=tool_call_id
            )
        )

    return {
        "messages": state["messages"] + tool_messages
    }


# ============================================================
# 6. 条件路由
# ============================================================

def route_after_llm(state: State):

    last_message = state["messages"][-1]

    if last_message.tool_calls:
        return "tool"

    return "end"


# ============================================================
# 7. 创建 Graph
# ============================================================

graph_builder = StateGraph(State)


# ============================================================
# 8. 添加 Node
# ============================================================

graph_builder.add_node("llm", llm_node)
graph_builder.add_node("tool", tool_node)


# ============================================================
# 9. START → LLM
# ============================================================

graph_builder.add_edge(START, "llm")


# ============================================================
# 10. LLM 后进行条件路由
# ============================================================

graph_builder.add_conditional_edges(
    "llm",
    route_after_llm,
    {
        "tool": "tool",
        "end": END
    }
)


# ============================================================
# 11. Tool → LLM
# ============================================================

graph_builder.add_edge("tool", "llm")


# ============================================================
# 12. 编译
# ============================================================

graph = graph_builder.compile()


# ============================================================
# 13. 执行 Agent
# ============================================================

result = graph.invoke(
    {
        "messages": [
            HumanMessage(
                content="北京今天天气怎么样？"
            )
        ]
    }
)


# ============================================================
# 14. 输出最终消息
# ============================================


print("\n========== 最终结果 ==========")

for message in result["messages"]:
    print(type(message).__name__, "：", message.content)

