from typing import TypedDict, Annotated    #在真实场景下即带有tool的情况下的测试human in loop效果

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import InMemorySaver


# ============================================================
# 1. State
# ============================================================

class State(TypedDict):
    messages: Annotated[list, add_messages]
    result: str


# ============================================================
# 2. Agent Node
# ============================================================

def agent_node(state: State):

    print("Agent：产生 Tool Call")

    return {
        "messages": [
            {
                "role": "assistant",
                "content": "请求执行 delete_file"
            }
        ]
    }


# ============================================================
# 3. 人工审批 Node
# ============================================================

def approval_node(state: State):

    print("进入人工审批")

    answer = interrupt(
        "Agent 请求执行 delete_file，是否允许？"
    )

    print("人工决定：", answer)

    if answer != "yes":
        return {
            "result": "Tool 执行被拒绝"
        }

    return {
        "result": "允许执行 Tool"
    }


# ============================================================
# 4. Tool Node（模拟真正执行）
# ============================================================

def tool_node(state: State):

    print("执行 Tool")

    return {
        "result": state["result"] + " → delete_file 执行成功"
    }


# ============================================================
# 5. Graph
# ============================================================

graph_builder = StateGraph(State)

graph_builder.add_node("agent", agent_node)
graph_builder.add_node("approval", approval_node)
graph_builder.add_node("tool", tool_node)

graph_builder.add_edge(START, "agent")
graph_builder.add_edge("agent", "approval")
graph_builder.add_edge("approval", "tool")
graph_builder.add_edge("tool", END)

checkpointer = InMemorySaver()

graph = graph_builder.compile(
    checkpointer=checkpointer
)


# ============================================================
# 6. 第一次执行
# ============================================================

config = {
    "configurable": {
        "thread_id": "user001"
    }
}

result = graph.invoke(
    {
        "messages": [],
        "result": ""
    },
    config
)

print("\n第一次执行：")
print(result)


# ============================================================
# 7. 恢复执行
# ============================================================

result = graph.invoke(
    Command(resume="s"),
    config
)

print("\n恢复后的结果：")
print(result)
