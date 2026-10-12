import os

from typing import Annotated, TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from langgraph.checkpoint.memory import InMemorySaver


# ============================================================
# 1. 定义 State
# ============================================================

class State(TypedDict):
    messages: Annotated[list, add_messages]


# ============================================================
# 2. 定义 LLM
# ============================================================

llm = ChatOpenAI(
    model="deepseek-v4-flash",
    api_key=os.environ["ANTHROPIC_AUTH_TOKEN"],
    base_url="https://api.deepseek.com"
)


# ============================================================
# 3. 定义 LLM Node
# ============================================================

def llm_node(state: State):

    print("\n========== LLM Node ==========")

    response = llm.invoke(
        state["messages"]
    )

    print("LLM：", response.content)

    return {
        "messages": [response]
    }


# ============================================================
# 4. 创建 Graph
# ============================================================

graph_builder = StateGraph(State)

graph_builder.add_node(
    "llm",
    llm_node
)

graph_builder.add_edge(
    START,
    "llm"
)

graph_builder.add_edge(
    "llm",
    END
)


# ============================================================
# 5. 创建 Checkpointer
# ============================================================

checkpointer = InMemorySaver()     #仅inmemorysaver是langgraph固定内置字段，这行用于创建一个在内存存贮一个会话多轮对话的状态记忆，要想长期或者永久储存可以用Redis,Postgressaver,SqliteSaver...


# ============================================================
# 6. 编译 Graph
# ============================================================

graph = graph_builder.compile(
    checkpointer=checkpointer      #这两行是编译graph并给graph挂载状态管理器几检查点保存器的的工具checkpointer（langgraph内置固定名称）=checkpointer（前面自主定义的保存器）
)


# ============================================================
# 7. 定义同一个会话 ID
# ============================================================

config = {    #自主定义
    "configurable": {
        "thread_id": "user001"      #"configurable"langgraph内置固定名称，"thread_id""内置固定名称
    }
}


# ============================================================
# 8. 第一次对话
# ============================================================

print("\n==============================")
print("第一次对话")
print("==============================")

result1 = graph.invoke(
    {
        "messages": [
            HumanMessage(
                content="我叫小明，请记住我的名字。"
            )
        ]
    },
    config
)

print("\n第一次对话结束。")


# ============================================================
# 9. 第二次对话
# ============================================================

print("\n==============================")
print("第二次对话")
print("==============================")

result2 = graph.invoke(
    {
        "messages": [
            HumanMessage(
                content="我刚才告诉你我叫什么？"
            )
        ]
    },
    config
)

print("\n第二次对话结束。")