import os          #各知识点模块综合汇总应用
from typing import TypedDict, Annotated

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import Command
from langgraph.prebuilt import ToolNode


# ============================================================
# 1. LLM
# ============================================================

llm = ChatOpenAI(
    model="deepseek-v4-flash",
    api_key=os.environ["ANTHROPIC_AUTH_TOKEN"],
    base_url="https://api.deepseek.com"
)


# ============================================================
# 2. 定义 Tool
# ============================================================

@tool
def get_weather(city: str) -> str:
    """查询指定城市的天气。"""

    print(f"\n[Tool] 查询天气：{city}")

    return f"{city}今天晴天，25℃"


tools = [get_weather]


# ============================================================
# 3. 给 LLM 绑定 Tool
# ============================================================

llm_with_tools = llm.bind_tools(tools)


# ============================================================
# 4. 定义 State
# ============================================================

class State(TypedDict):
    messages: Annotated[list, add_messages]


# ============================================================
# 5. LLM Node
# ============================================================

def llm_node(state: State):

    print("\n========== LLM Node ==========")

    response = llm_with_tools.invoke(state["messages"])

    print("LLM 返回：")
    print(response)

    # --------------------------------------------------------
    # 如果 LLM 要调用 Tool
    # --------------------------------------------------------

    if response.tool_calls:

        print("\n检测到 Tool Call")

        return Command(
            update={
                "messages": [response]
            },
            goto="tools"
        )

    # --------------------------------------------------------
    # 如果 LLM 不需要 Tool
    # --------------------------------------------------------

    print("\n没有 Tool Call，结束")

    return Command(
        update={
            "messages": [response]
        },
        goto=END
    )


# ============================================================
# 6. ToolNode
# ============================================================

tool_node = ToolNode(tools)


# ============================================================
# 7. 构建 Graph
# ============================================================

graph_builder = StateGraph(State)

graph_builder.add_node("llm", llm_node)
graph_builder.add_node("tools", tool_node)

graph_builder.add_edge(START, "llm")

# Tool 执行完成后回到 LLM
graph_builder.add_edge("tools", "llm")

graph = graph_builder.compile()


# ============================================================
# 8. 测试
# ============================================================

result = graph.invoke({
    "messages": [
        HumanMessage(
            content="北京今天天气怎么样？"
        )
    ]
})


# ============================================================
# 9. 输出最终结果
# ============================================================

print("\n\n========================================")
print("最终 State")
print("========================================")

for message in result["messages"]:

    print("\n----------------------------------------")
    print("类型：", type(message).__name__)
    print("内容：", message.content)

    if hasattr(message, "tool_calls") and message.tool_calls:
        print("Tool Calls：", message.tool_calls)
