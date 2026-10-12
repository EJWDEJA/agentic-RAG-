from typing import Literal, TypedDict    #第一版戴委托协议和沟通协议的多agent协调，此处的主管为规则不够通用输出带有target和task的类

from langgraph.graph import StateGraph, START, END


# ============================================================
# 1. Agent Task
# ============================================================

class AgentTask(TypedDict):
    target: Literal["research", "writer"]
    task: str


# ============================================================
# 2. Research Report
# ============================================================

class ResearchReport(TypedDict):
    summary: str
    key_points: list[str]


# ============================================================
# 3. Main State
# ============================================================

class MainState(TypedDict):
    user_task: str
    agent_task: AgentTask
    research_report: ResearchReport
    draft: str


# ============================================================
# 4. Supervisor
# ============================================================

def supervisor(state: MainState):

    print("\n===== Supervisor =====")

    # 第一次进入：还没有研究结果
    if not state["research_report"]["summary"]:

        task: AgentTask = {
            "target": "research",
            "task": f"研究以下问题：{state['user_task']}"
        }

        print("Supervisor → Research")

        return {
            "agent_task": task
        }

    # 第二次进入：已经拿到了 ResearchReport
    else:           #这部分属于硬编码死板固定只能通向writer，后续引入llm以后这部分就可以被llm替代由提示词驱动llm动态决定下一步进行深度研究还是，开始写作
        task: AgentTask = {
            "target": "writer",
            "task": f"根据研究结果撰写答案：{state['user_task']}"
        }

        print("Supervisor → Writer")

        return {
            "agent_task": task
        }


# ============================================================
# 5. Dispatcher
# ============================================================

def dispatcher(state: MainState):

    target = state["agent_task"]["target"]

    print(f"Dispatcher → {target}")

    return target


# ============================================================
# 6. Research Agent
# ============================================================

def research(state: MainState):

    print("\n===== Research Agent =====")

    report: ResearchReport = {
        "summary": "人工智能正在改变教育方式。",
        "key_points": [
            "个性化学习",
            "智能教学辅助",
            "自动化教育管理"
        ]
    }

    return {
        "research_report": report
    }


# ============================================================
# 7. Writer Agent
# ============================================================

def writer(state: MainState):

    print("\n===== Writer Agent =====")

    report = state["research_report"]

    draft = (
        f"主题：{state['user_task']}\n\n"
        f"{report['summary']}\n\n"
        f"主要应用包括：\n"
        + "\n".join(f"- {item}" for item in report["key_points"])
    )

    return {
        "draft": draft
    }


# ============================================================
# 8. Build Graph
# ============================================================

builder = StateGraph(MainState)

builder.add_node("supervisor", supervisor)
builder.add_node("research", research)
builder.add_node("writer", writer)

builder.add_edge(START, "supervisor")


# Supervisor → Dispatcher
builder.add_conditional_edges(
    "supervisor",
    dispatcher,
    {
        "research": "research",
        "writer": "writer",
    }
)


# Research 完成后重新回 Supervisor
builder.add_edge("research", "supervisor")


# Writer 完成后结束
builder.add_edge("writer", END)


# ============================================================
# 9. Compile
# ============================================================

graph = builder.compile()


# ============================================================
# 10. Test
# ============================================================

result = graph.invoke({
    "user_task": "人工智能在教育领域的应用",
    "agent_task": {
        "target": "research",
        "task": ""
    },
    "research_report": {
        "summary": "",
        "key_points": []
    },
    "draft": ""
})


print("\n==============================")
print("最终结果")
print("==============================")

print(result["draft"])