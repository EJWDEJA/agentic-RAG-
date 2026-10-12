from typing import TypedDict      #单agent仅仅作为一个个节点，共享全局状态

from langgraph.graph import StateGraph, START, END
from langgraph.types import Command


# ============================================================
# 1. 全局 State
# ============================================================

class State(TypedDict):
    task: str
    result: str
    current_agent: str


# ============================================================
# 2. Supervisor Agent
# ============================================================

def supervisor(state: State):

    print("执行 Supervisor")

    if "研究" in state["task"]:
        return Command(
            update={
                "current_agent": "researcher"
            },
            goto="researcher"
        )

    return Command(
        update={
            "current_agent": "writer"
        },
        goto="writer"
    )


# ============================================================
# 3. Research Agent
# ============================================================

def researcher(state: State):

    print("执行 Research Agent")

    return {
        "result": "Research Agent 完成研究"
    }


# ============================================================
# 4. Writer Agent
# ============================================================

def writer(state: State):

    print("执行 Writer Agent")

    return {
        "result": "Writer Agent 完成写作"
    }


# ============================================================
# 5. 构建 Graph
# ============================================================

graph_builder = StateGraph(State)

graph_builder.add_node("supervisor", supervisor)
graph_builder.add_node("researcher", researcher)
graph_builder.add_node("writer", writer)

graph_builder.add_edge(START, "supervisor")

graph_builder.add_edge("researcher", END)
graph_builder.add_edge("writer", END)

graph = graph_builder.compile()


# ============================================================
# 6. 测试
# ============================================================

result = graph.invoke({
    "task": "研究学校人工智能发展情况",
    "result": "",
    "current_agent": ""
})

print("\n最终 State：")
print(result)