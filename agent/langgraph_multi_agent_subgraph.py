from typing import TypedDict     #这一步测试固定流程链式父子图是否能跑通，    这种只适合简单任务agent间数据传输复杂任务需要通信协议

from langgraph.graph import StateGraph, START, END


# ============================================================
# 1. Main Graph State
# ============================================================

class MainState(TypedDict):
    task: str
    research_result: str
    draft: str


# ============================================================
# 2. Research Subgraph State
# ============================================================

class ResearchState(TypedDict):
    task: str
    result: str


# ============================================================
# 3. Research Agent
# ============================================================

def research_agent(state: ResearchState):

    print("执行 Research Agent")

    return {
        "result": f"针对「{state['task']}」完成研究"
    }


# ============================================================
# 4. Research Subgraph
# ============================================================

research_builder = StateGraph(ResearchState)

research_builder.add_node("research", research_agent)

research_builder.add_edge(START, "research")
research_builder.add_edge("research", END)

research_graph = research_builder.compile()


# ============================================================
# 5. Writer Subgraph State
# ============================================================

class WriterState(TypedDict):
    task: str
    research_result: str
    draft: str


# ============================================================
# 6. Writer Agent
# ============================================================

def writer_agent(state: WriterState):

    print("执行 Writer Agent")

    return {
        "draft": (
            f"根据研究结果「{state['research_result']}」"
            f"完成「{state['task']}」的草稿"
        )
    }


# ============================================================
# 7. Writer Subgraph
# ============================================================

writer_builder = StateGraph(WriterState)

writer_builder.add_node("writer", writer_agent)

writer_builder.add_edge(START, "writer")
writer_builder.add_edge("writer", END)

writer_graph = writer_builder.compile()


# ============================================================
# 8. Supervisor
# ============================================================

def supervisor(state: MainState):

    print("执行 Supervisor")

    # Supervisor 负责调度，
    # 这里先固定：Research → Writer
    return {}


# ============================================================
# 9. Research Subgraph Node     这是为了提取子图状态内部父图需要的字段
# ============================================================

def run_research(state: MainState):

    result = research_graph.invoke({
        "task": state["task"],
        "result": ""
    })

    return {
        "research_result": result["result"]
    }      #这里是核心真正实现了父子图状态隔离


# ============================================================
# 10. Writer Subgraph Node          这是为了提取子图状态内部父图需要的字段
# ============================================================

def run_writer(state: MainState):

    result = writer_graph.invoke({
        "task": state["task"],
        "research_result": state["research_result"],
        "draft": ""
    })

    return {
        "draft": result["draft"]
    }


# ============================================================
# 11. Main Graph
# ============================================================

main_builder = StateGraph(MainState)

main_builder.add_node("supervisor", supervisor)
main_builder.add_node("research", run_research)
main_builder.add_node("writer", run_writer)

main_builder.add_edge(START, "supervisor")
main_builder.add_edge("supervisor", "research")
main_builder.add_edge("research", "writer")
main_builder.add_edge("writer", END)

main_graph = main_builder.compile()


# ============================================================
# 12. Execute
# ============================================================

result = main_graph.invoke({
    "task": "研究人工智能并写一篇介绍",
    "research_result": "",
    "draft": ""
})

print("\n最终 Main State：")
print(result)