from typing import TypedDict#测试状态隔离，此时的agent是一个普通函数，子状态没更新（python普通函数无注册状态无节点没权利）也基本上没什么用

from langgraph.graph import StateGraph, START, END


# ============================================================
# 1. Global State
# ============================================================

class GlobalState(TypedDict):
    task: str
    research_result: str
    draft: str


# ============================================================
# 2. Research Agent 自己的局部 State
# ============================================================

class ResearchState(TypedDict):
    task: str


# ============================================================
# 3. Writer Agent 自己的局部 State
# ============================================================

class WriterState(TypedDict):
    task: str
    research_result: str


# ============================================================
# 4. Research Agent
# ============================================================

def research_agent(state: ResearchState):

    print("Research Agent 收到：")
    print(state)

    return {
        "research_result": "Research Agent 完成研究"
    }


# ============================================================
# 5. Writer Agent
# ============================================================

def writer_agent(state: WriterState):

    print("\nWriter Agent 收到：")
    print(state)

    return {
        "draft": "Writer Agent 根据研究结果完成草稿"
    }


# ============================================================
# 6. Global State → Research Agent
# ============================================================

def research_node(state: GlobalState):

    # 只提取 Research Agent 需要的数据
    local_state: ResearchState = {
        "task": state["task"]
    }

    result = research_agent(local_state)

    # 只把 Agent 的结果写回 Global State
    return {
        "research_result": result["research_result"]
    }


# ============================================================
# 7. Global State → Writer Agent
# ============================================================

def writer_node(state: GlobalState):

    # 只提取 Writer Agent 需要的数据
    local_state: WriterState = {
        "task": state["task"],
        "research_result": state["research_result"]
    }

    result = writer_agent(local_state)

    # 只把 Agent 的结果写回 Global State
    return {
        "draft": result["draft"]
    }


# ============================================================
# 8. Graph
# ============================================================

graph_builder = StateGraph(GlobalState)

graph_builder.add_node("research", research_node)
graph_builder.add_node("writer", writer_node)

graph_builder.add_edge(START, "research")
graph_builder.add_edge("research", "writer")
graph_builder.add_edge("writer", END)

graph = graph_builder.compile()


# ============================================================
# 9. 执行
# ============================================================

result = graph.invoke({
    "task": "研究人工智能",
    "research_result": "",
    "draft": ""
})

print("\n最终 Global State：")
print(result)