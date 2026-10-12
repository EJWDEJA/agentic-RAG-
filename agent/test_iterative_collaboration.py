from typing import Literal, TypedDict                   #硬编码可迭代对象流程严格固定，未引入llm实现动态判断，这里的迭代次数是严格限制的属于工作流

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
    research_round: int
    draft: str


# ============================================================
# 4. Supervisor
# ============================================================

def supervisor(state: MainState):

    print("\n===== Supervisor =====")

    report = state["research_report"]
    round_number = state["research_round"]

    # 第一次：开始研究
    if not report["summary"]:

        print("第一次委托 Research")

        return {
            "agent_task": {
                "target": "research",
                "task": f"研究：{state['user_task']}"
            }
        }

    # 研究已经进行过一次，但信息不足
    if round_number == 1:

        print("研究结果不足，要求 Research 补充")

        return {
            "agent_task": {
                "target": "research",
                "task": (
                    "根据已有研究结果继续补充，"
                    "重点增加具体应用场景。"
                )
            }
        }

    # 第二轮研究完成
    print("研究结果已经足够，交给 Writer")

    return {
        "agent_task": {
            "target": "writer",
            "task": "根据完整研究结果撰写最终答案"
        }
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

    round_number = state["research_round"] + 1

    print(f"\n===== Research Agent 第 {round_number} 轮 =====")

    if round_number == 1:

        report: ResearchReport = {
            "summary": "人工智能正在改变教育方式。",
            "key_points": [
                "个性化学习"
            ]
        }

    else:

        report: ResearchReport = {
            "summary": "人工智能正在改变教育方式。",
            "key_points": [
                "个性化学习",
                "智能教学辅助",
                "自动化教育管理"
            ]
        }

    return {
        "research_report": report,
        "research_round": round_number
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
        + "\n".join(
            f"- {item}"
            for item in report["key_points"]
        )
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
    "research_round": 0,
    "draft": ""
})


# ============================================================
# 11. Result
# ============================================================

print("\n==============================")
print("最终结果")
print("==============================")

print(result["draft"])