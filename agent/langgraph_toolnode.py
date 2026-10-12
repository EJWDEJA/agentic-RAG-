import os    #负责使用Tool Node 和Redure(接收新状态更新全局状态，方式是直接追加代替覆盖)

from typing import Annotated, TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode


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
    messages: Annotated[list, add_messages]   #告诉langgraph新状态以追加代替替代的方式更新全局状态

# ============================================================
# 4. LLM Node
# ============================================================

def llm_node(state: State):

    print("\n========== LLM Node ==========")

    response = llm_with_tools.invoke(
        state["messages"]
    )

    # 判断 LLM 是否产生了 Tool Call
    if response.tool_calls:

        print("LLM 决定调用 Tool")

        print("Tool Calls：")
        print(response.tool_calls)

    else:

        print("LLM 直接给出最终答案：")
        print(response.content)

    # 注意：
    # 这里只返回“本次新增的消息”
    #
    # 不再写：
    # state["messages"] + [response]
    #
    # 因为 messages 已经配置了 add_messages Reducer，
    # LangGraph 会自动把 response 追加到历史消息中。

    return {
        "messages": [response]
    }


# ============================================================
# 5. ToolNode
# ============================================================

tool_node = ToolNode(tools)


# ============================================================
# 6. LLM 后的条件路由
# ============================================================

def route_after_llm(state: State):

    # 获取当前最后一条消息
    last_message = state["messages"][-1]

    # 如果 LLM 要调用 Tool
    if last_message.tool_calls:

        return "tool"

    # 如果没有 Tool Call
    # 说明 LLM 已经可以直接给出最终答案
    return "end"


# ============================================================
# 7. 创建 StateGraph
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
# 10. LLM → 条件路由
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
# 12. 编译 Graph
# ============================================================

graph = graph_builder.compile()


# ============================================================
# 13. 执行 Graph
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
# 14. 查看最终完整消息历史
# ============================================================

print("\n========== 最终消息历史 ==========")


for i, message in enumerate(
    result["messages"],
    start=1
):

    print(f"\n--- Message {i} ---")

    print(
        "类型：",
        type(message).__name__
    )

    print(
        "内容：",
        message.content
    )

    # 如果是 AIMessage，并且存在 Tool Call
    if getattr(message, "tool_calls", None):

        print(
            "Tool Calls：",
            message.tool_calls
        )
