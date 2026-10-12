from typing import TypedDict

from langgraph.graph import StateGraph, START, END


# ============================================================
# 1. Task Contract
#    Supervisor → Research Agent
# ============================================================

class AgentTask(TypedDict):
    task: str    #输入子agent的协议


# ============================================================
# 2. Research Communication Contract
#    Research Agent → Supervisor
# ============================================================

class ResearchReport(TypedDict):    #阿根廷间通讯协议即返回给全局状态的一个类
    summary: str
    key_points: list[str]


# ============================================================
# 3. Research Agent Internal State
# ============================================================

class ResearchState(TypedDict):#子agent状态
    task: str
    summary: str
    key_points: list[str]


# ============================================================
# 4. Main State
# ============================================================

class MainState(TypedDict):
    user_task: str
    research_task: AgentTask
    research_report: ResearchReport
    draft: str


# ============================================================
# 5. Research Agent
# ============================================================

def research_agent(state: ResearchState):

    print("执行 Research Agent")
    print("收到任务：", state["task"])

    return {
        "summary": f"完成了对「{state['task']}」的研究",
        "key_points": [
            "教育领域应用",
            "智能教学",
            "个性化学习"
        ]
    }


# ============================================================
# 6. Research Subgraph
# ============================================================

research_builder = StateGraph(ResearchState)

research_builder.add_node("research", research_agent)

research_builder.add_edge(START, "research")
research_builder.add_edge("research", END)

research_graph = research_builder.compile()


# ============================================================
# 7. Supervisor
# ============================================================

def supervisor(state: MainState):

    print("\n执行 Supervisor")

    delegated_task: AgentTask = {
        "task": f"研究以下问题：{state['user_task']}"
    }

    return {
        "research_task": delegated_task
    }


# ============================================================
# 8. Run Research
# ============================================================

def run_research(state: MainState):

    print("\nSupervisor 委派 Research 任务")

    # --------------------------------------------------------
    # AgentTask → ResearchState
    # --------------------------------------------------------

    subgraph_input: ResearchState = {
        "task": state["research_task"]["task"],
        "summary": "",
        "key_points": []
    }

    result = research_graph.invoke(subgraph_input)

    # --------------------------------------------------------
    # ResearchState → ResearchReport
    # --------------------------------------------------------

    research_report: ResearchReport = {
        "summary": result["summary"],
        "key_points": result["key_points"]
    }

    return {
        "research_report": research_report
    }


# ============================================================
# 9. Main Graph
# ============================================================

builder = StateGraph(MainState)

builder.add_node("supervisor", supervisor)
builder.add_node("research", run_research)

builder.add_edge(START, "supervisor")
builder.add_edge("supervisor", "research")
builder.add_edge("research", END)

graph = builder.compile()


# ============================================================
# 10. Test
# ============================================================

result = graph.invoke({
    "user_task": "人工智能在教育领域的应用",

    "research_task": {
        "task": ""
    },

    "research_report": {
        "summary": "",
        "key_points": []
    },

    "draft": ""
})


# ============================================================
# 11. Output
# ============================================================

print("\n最终结果：")
print(result)