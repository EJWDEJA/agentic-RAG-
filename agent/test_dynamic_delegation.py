from typing import TypedDict, Literal      #这个是用来测试委托契约的，理解方式和沟通1契约一样，最终都是以类的形式返回给全局状态的

from langgraph.graph import StateGraph, START, END         #这一版的surpervision是规则型的后续引入llm会优化掉这部分规则


# ============================================================
# 1. Delegation Contract
#    Supervisor → Agent
# ============================================================

class AgentTask(TypedDict):
    target: Literal["research", "writer"]
    task: str


# ============================================================
# 2. Agent Result Contract
#    Agent → Supervisor
# ============================================================

class ResearchReport(TypedDict):
    summary: str
    key_points: list[str]


class WritingReport(TypedDict):
    draft: str


# ============================================================
# 3. Research Agent Internal State
# ============================================================

class ResearchState(TypedDict):
    task: str
    summary: str
    key_points: list[str]


# ============================================================
# 4. Writer Agent Internal State
# ============================================================

class WriterState(TypedDict):
    task: str
    draft: str


# ============================================================
# 5. Main State
# ============================================================

class MainState(TypedDict):
    user_task: str
    agent_task: AgentTask
    research_report: ResearchReport
    writing_report: WritingReport


# ============================================================
# 6. Research Agent
# ============================================================

def research_agent(state: ResearchState):

    print("\n[Research Agent]")
    print("收到任务：", state["task"])

    return {
        "summary": f"完成研究：{state['task']}",
        "key_points": [
            "教育领域应用",
            "智能教学",
            "个性化学习"
        ]
    }


# ============================================================
# 7. Research Subgraph
# ============================================================

research_builder = StateGraph(ResearchState)

research_builder.add_node("research_agent", research_agent)

research_builder.add_edge(START, "research_agent")
research_builder.add_edge("research_agent", END)

research_graph = research_builder.compile()


# ============================================================
# 8. Writer Agent
# ============================================================

def writer_agent(state: WriterState):

    print("\n[Writer Agent]")
    print("收到任务：", state["task"])

    return {
        "draft": f"根据任务「{state['task']}」生成了一份初稿。"
    }


# ============================================================
# 9. Writer Subgraph
# ============================================================

writer_builder = StateGraph(WriterState)

writer_builder.add_node("writer_agent", writer_agent)

writer_builder.add_edge(START, "writer_agent")
writer_builder.add_edge("writer_agent", END)

writer_graph = writer_builder.compile()


# ============================================================
# 10. Supervisor
#     暂时使用规则，不使用 LLM
# ============================================================

def supervisor(state: MainState):

    user_task = state["user_task"]

    print("\n[Supervisor]")
    print("收到用户任务：", user_task)

    if "研究" in user_task:                                ##########这部分的if  ...in待优化，后续引入llm，llm会自主输出有效的task，target，并以一个类整体的形式汇总到全局变量里面
        delegated_task: AgentTask = {
            "target": "research",
            "task": user_task
        }

    elif "写" in user_task:
        delegated_task: AgentTask = {
            "target": "writer",
            "task": user_task
        }

    else:
        delegated_task: AgentTask = {
            "target": "research",
            "task": user_task
        }

    print("决定委派给：", delegated_task["target"])
    print("委派任务：", delegated_task["task"])

    return {
        "agent_task": delegated_task
    }


# ============================================================
# 11. Dispatcher
# ============================================================

def dispatcher(state: MainState):

    target = state["agent_task"]["target"]

    print("\n[Dispatcher]")
    print("目标 Agent：", target)

    if target == "research":
        return "run_research"

    elif target == "writer":
        return "run_writer"

    raise ValueError(f"未知 Agent: {target}")


# ============================================================
# 12. Run Research
# ============================================================

def run_research(state: MainState):

    print("\n[Supervisor → Research]")

    subgraph_input: ResearchState = {
        "task": state["agent_task"]["task"],
        "summary": "",
        "key_points": []
    }

    result = research_graph.invoke(subgraph_input)

    research_report: ResearchReport = {
        "summary": result["summary"],
        "key_points": result["key_points"]
    }

    return {
        "research_report": research_report
    }


# ============================================================
# 13. Run Writer
# ============================================================

def run_writer(state: MainState):

    print("\n[Supervisor → Writer]")

    subgraph_input: WriterState = {
        "task": state["agent_task"]["task"],
        "draft": ""
    }

    result = writer_graph.invoke(subgraph_input)

    writing_report: WritingReport = {
        "draft": result["draft"]
    }

    return {
        "writing_report": writing_report
    }


# ============================================================
# 14. Main Graph
# ============================================================

builder = StateGraph(MainState)

builder.add_node("supervisor", supervisor)
builder.add_node("dispatcher", lambda state: {})
builder.add_node("run_research", run_research)
builder.add_node("run_writer", run_writer)

builder.add_edge(START, "supervisor")
builder.add_edge("supervisor", "dispatcher")

builder.add_conditional_edges(
    "dispatcher",
    dispatcher,
    {
        "run_research": "run_research",
        "run_writer": "run_writer",
    }
)

builder.add_edge("run_research", END)
builder.add_edge("run_writer", END)

graph = builder.compile()


# ============================================================
# 15. Test
# ============================================================

print("=" * 60)
print("测试 1：Research Delegation")
print("=" * 60)

result = graph.invoke({
    "user_task": "研究人工智能在教育领域的应用",
    "agent_task": {
        "target": "research",
        "task": ""
    },
    "research_report": {
        "summary": "",
        "key_points": []
    },
    "writing_report": {
        "draft": ""
    }
})

print("\n最终结果：")
print(result)


print("\n" + "=" * 60)
print("测试 2：Writer Delegation")
print("=" * 60)

result = graph.invoke({
    "user_task": "写一篇关于人工智能教育应用的文章",
    "agent_task": {
        "target": "research",
        "task": ""
    },
    "research_report": {
        "summary": "",
        "key_points": []
    },
    "writing_report": {
        "draft": ""
    }
})

print("\n最终结果：")
print(result)