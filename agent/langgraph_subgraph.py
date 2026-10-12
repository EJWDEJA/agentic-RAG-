from typing import TypedDict         #测试状态隔离，此时的agent是一个子图，子状态更新也用上了，真正实现状态隔离

from langgraph.graph import StateGraph, START, END


# ============================================================
# 1. 主 Graph State
# ============================================================

class MainState(TypedDict):
    task: str
    research_result: str


# ============================================================
# 2. Research Subgraph 自己的 State
# ============================================================

class ResearchState(TypedDict):
    task: str
    result: str


# ============================================================
# 3. Research Subgraph Node
# ============================================================

def research_node(state: ResearchState):

    print("执行 Research Node")

    return {
        "result": f"针对「{state['task']}」完成研究"
    }


# ============================================================
# 4. 构建 Research Subgraph
# ============================================================

research_builder = StateGraph(ResearchState)

research_builder.add_node("research", research_node)

research_builder.add_edge(START, "research")
research_builder.add_edge("research", END)

research_graph = research_builder.compile()


# ============================================================
# 5. 主 Graph 调用 Subgraph
# ============================================================

def run_research_subgraph(state: MainState):

    print("进入 Research Subgraph")

    # MainState → ResearchState
    subgraph_input = {
        "task": state["task"],
        "result": ""
    }

    subgraph_result = research_graph.invoke(subgraph_input)

    # ResearchState → MainState
    return {
        "research_result": subgraph_result["result"]
    }


# ============================================================
# 6. 构建主 Graph
# ============================================================

main_builder = StateGraph(MainState)

main_builder.add_node(
    "research_agent",
    run_research_subgraph
)

main_builder.add_edge(
    START,
    "research_agent"
)

main_builder.add_edge(
    "research_agent",
    END
)

main_graph = main_builder.compile()


# ============================================================
# 7. 执行
# ============================================================

result = main_graph.invoke({
    "task": "研究人工智能",
    "research_result": ""
})

print("\n最终 Main State：")
print(result)