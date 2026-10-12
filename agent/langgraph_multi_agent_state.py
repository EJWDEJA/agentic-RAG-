from typing import TypedDict    #测试共享全局状态效果

from langgraph.graph import StateGraph, START, END


# ============================================================
# 1. Global State
# ============================================================

class State(TypedDict):
    task: str
    research: str
    draft: str


# ============================================================
# 2. Research Agent
# ============================================================

def researcher(state: State):

    print("Research Agent 看到的 State：")
    print(state)

    return {
        "research": "Research Agent 的研究结果"
    }


# ============================================================
# 3. Writer Agent
# ============================================================

def writer(state: State):

    print("\nWriter Agent 看到的 State：")
    print(state)

    return {
        "draft": "Writer Agent 根据研究结果生成草稿"
    }


# ============================================================
# 4. Graph
# ============================================================

graph_builder = StateGraph(State)

graph_builder.add_node("researcher", researcher)
graph_builder.add_node("writer", writer)

graph_builder.add_edge(START, "researcher")
graph_builder.add_edge("researcher", "writer")
graph_builder.add_edge("writer", END)

graph = graph_builder.compile()


# ============================================================
# 5. 执行
# ============================================================

result = graph.invoke({
    "task": "研究人工智能",
    "research": "",
    "draft": ""
})

print("\n最终 State：")
print(result)