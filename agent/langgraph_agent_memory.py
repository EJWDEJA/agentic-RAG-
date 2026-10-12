import os

from typing import Annotated, TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from langchain_core.tools import tool

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from langgraph.checkpoint.memory import InMemorySaver


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

    return weather_data.get(
        city,
        f"{city}：暂无天气数据"
    )


tools = [get_weather]


# ============================================================
# 2. 定义 LLM
# ============================================================

llm = ChatOpenAI(
    model="deepseek-v4-flash",
    api_key=os.environ["ANTHROPIC_AUTH_TOKEN"],
    base_url="https://api.deepseek.com"
)

llm_with_tools = llm.bind_tools(tools)


# ============================================================
# 3. 定义 State
# ============================================================

class State(TypedDict):
    messages: Annotated[list, add_messages]


# ============================================================
# 4. LLM Node
# ============================================================

def llm_node(state: State):

    print("\n========== LLM Node ==========")

    response = llm_with_tools.invoke(
        state["messages"]
    )

    if response.tool_calls:

        print("LLM 决定调用 Tool：")

        for tool_call in response.tool_calls:
            print("工具名称：", tool_call["name"])
            print("工具参数：", tool_call["args"])

    else:

        print("LLM 最终回答：")
        print(response.content)

    return {
        "messages": [response]
    }


# ============================================================
# 5. ToolNode
# ============================================================

tool_node = ToolNode(tools)


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

graph_builder.add_node(
    "llm",
    llm_node
)

graph_builder.add_node(
    "tool",
    tool_node
)


# ============================================================
# 9. START → LLM
# ============================================================

graph_builder.add_edge(
    START,
    "llm"
)


# ============================================================
# 10. LLM → Tool / END
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

graph_builder.add_edge(
    "tool",
    "llm"
)


# ============================================================
# 12. 创建 Checkpointer
# ============================================================

checkpointer = InMemorySaver()


# ============================================================
# 13. 编译 Graph
# ============================================================

graph = graph_builder.compile(
    checkpointer=checkpointer
)


# ============================================================
# 14. 创建会话
# ============================================================

config = {
    "configurable": {
        "thread_id": "user001"
    }
}


# ============================================================
# 15. 第一次对话
# ============================================================

print("\n========================================")
print("第一次对话")
print("========================================")

result1 = graph.invoke(
    {
        "messages": [
            HumanMessage(
                content="北京今天天气怎么样？"
            )
        ]
    },
    config
)

print("\n第一次对话完成。")


# ============================================================
# 16. 第二次对话
# ============================================================

print("\n========================================")
print("第二次对话")
print("========================================")

result2 = graph.invoke(
    {
        "messages": [
            HumanMessage(
                content="刚才查询的城市叫什么？"
            )
        ]
    },
    config
)

print("\n第二次对话完成。")
