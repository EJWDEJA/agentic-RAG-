from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.types import Command


# ============================================================
# 1. 定义 State
# ============================================================

class State(TypedDict):
    message: str
    route: str


# ============================================================
# 2. Node A
# ============================================================

def node_a(state: State):

    print("执行 Node A")

    return Command(
        update={
            "message": state["message"] + " → A",
            "route": "node_b"
        },
        goto="node_b"
    )


# ============================================================
# 3. Node B
# ============================================================

def node_b(state: State):

    print("执行 Node B")

    return {
        "message": state["message"] + " → B"
    }


# ============================================================
# 4. 构建 Graph
# ============================================================

graph_builder = StateGraph(State)

graph_builder.add_node("node_a", node_a)
graph_builder.add_node("node_b", node_b)

graph_builder.add_edge(START, "node_a")
graph_builder.add_edge("node_b", END)

graph = graph_builder.compile()


# ============================================================
# 5. 执行
# ============================================================

result = graph.invoke({
    "message": "开始",
    "route": ""
})

print("\n最终 State：")
print(result)
