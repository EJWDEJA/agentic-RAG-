from typing import TypedDict     #这一步测试动态路由的父子图是否能跑通

from langgraph.graph import StateGraph, START, END
from langgraph.types import Command


# ============================================================
# 1. Main Global State
# ============================================================

class MainState(TypedDict):
    task: str
    research_result: str
    draft: str
    next_agent: str


# ============================================================
# 2. Research Agent
# ============================================================

def research_agent(state):
    print("执行 Research Agent")

    return {
        "result": f"针对「{state['task']}」完成研究"
    }


# ============================================================
# 3. Research Subgraph State
# ============================================================

class ResearchState(TypedDict):
    task: str
    result: str


# ============================================================
# 4. Research Subgraph
# ============================================================

research_builder = StateGraph(ResearchState)

research_builder.add_node("research_agent", research_agent)

research_builder.add_edge(START, "research_agent")
research_builder.add_edge("research_agent", END)

research_graph = research_builder.compile()


# ============================================================
# 5. Writer Agent
# ============================================================

def writer_agent(state):
    print("执行 Writer Agent")

    return {
        "draft": (
            f"根据研究结果「{state['research_result']}」"
            f"完成「{state['task']}」的草稿"
        )
    }


# ============================================================
# 6. Writer Subgraph State
# ============================================================

class WriterState(TypedDict):
    task: str
    research_result: str
    draft: str


# ============================================================
# 7. Writer Subgraph
# ============================================================

writer_builder = StateGraph(WriterState)

writer_builder.add_node("writer_agent", writer_agent)

writer_builder.add_edge(START, "writer_agent")
writer_builder.add_edge("writer_agent", END)

writer_graph = writer_builder.compile()


# ============================================================
# 8. 调用 Research Subgraph
# ============================================================

def run_research(state: MainState):

    result = research_graph.invoke({
        "task": state["task"],
        "result": ""
    })

    return {
        "research_result": result["result"]
    }


# ============================================================
# 9. 调用 Writer Subgraph
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
# 10. Dynamic Supervisor
# ============================================================

def supervisor(state: MainState):

    print("执行 Supervisor")

    # 还没有研究结果
    if not state["research_result"]:

        return Command(
            update={
                "next_agent": "research"
            },
            goto="research"
        )

    # 已经研究，但还没有草稿
    if not state["draft"]:

        return Command(
            update={
                "next_agent": "writer"
            },
            goto="writer"
        )

    # 研究和写作都完成
    return Command(
        update={
            "next_agent": "end"
        },
        goto=END
    )


# ============================================================
# 11. Main Graph
# ============================================================

builder = StateGraph(MainState)

builder.add_node("supervisor", supervisor)
builder.add_node("research", run_research)
builder.add_node("writer", run_writer)

builder.add_edge(START, "supervisor")    #用command选择下一个节点

builder.add_edge("research", "supervisor")
builder.add_edge("writer", "supervisor")

graph = builder.compile()


# ============================================================
# 12. 测试
# ============================================================

result = graph.invoke({
    "task": "研究人工智能并写一篇介绍",
    "research_result": "",
    "draft": "",
    "next_agent": ""
})

print("\n最终 State：")
print(result)