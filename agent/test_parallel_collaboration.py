from typing import TypedDict                     #这个版本的supervision被加入节点，使用必须使用command返回结果，后续用普通边加send触发

from langgraph.graph import StateGraph, START, END            #测试并发协作
from langgraph.types import Command, Send


# ============================================================
# 1. Main State
# ============================================================

class MainState(TypedDict):
    user_task: str

    research_result: str
    analysis_result: str
    review_result: str

    final_answer: str


# ============================================================
# 2. Agent Local State
# ============================================================

class AgentState(TypedDict):
    user_task: str


# ============================================================
# 3. Supervisor
# ============================================================

def supervisor(state: MainState):

    print("\n===== Supervisor =====")

    # 第一次进入：并行派发任务
    if not state["research_result"]:

        print("Supervisor → 并行派发 Research / Analyst / Reviewer")

        return Command(
        goto=[                        #此处必须是command不是send列表，因为这里的主管是一个节点，不用command他会作为一个普通节点，要求返回字典用于更新全局状态，当为command这种langgraph内置控制流指令才能符合条件
            Send(
                "research",
                {
                    "user_task": state["user_task"]
                }
            ),
            Send(
                "analyst",
                {
                    "user_task": state["user_task"]
                }
            ),
            Send(
                "reviewer",
                {
                    "user_task": state["user_task"]
                }
            )
        ]
        )   

    # 三个 Agent 都完成后
    print("三个 Agent 均已完成")
    print("Supervisor → 汇总结果")

    return {}


# ============================================================
# 4. Research Agent
# ============================================================

def research(state: AgentState):

    print("\n===== Research Agent =====")

    return {
        "research_result": (
            "Research：人工智能可以应用于个性化学习、"
            "智能教学和教育资源管理。"
        )
    }


# ============================================================
# 5. Analyst Agent
# ============================================================

def analyst(state: AgentState):

    print("\n===== Analyst Agent =====")

    return {
        "analysis_result": (
            "Analyst：人工智能的主要价值包括提高教学效率、"
            "降低重复性工作，并辅助教师进行学习分析。"
        )
    }


# ============================================================
# 6. Reviewer Agent
# ============================================================

def reviewer(state: AgentState):

    print("\n===== Reviewer Agent =====")

    return {
        "review_result": (
            "Reviewer：使用人工智能时需要关注数据隐私、"
            "结果可靠性以及教师监督。"
        )
    }


# ============================================================
# 7. Aggregator
# ============================================================

def aggregator(state: MainState):

    print("\n===== Aggregator =====")

    final_answer = (
        f"用户问题：{state['user_task']}\n\n"

        f"【研究结果】\n"
        f"{state['research_result']}\n\n"

        f"【分析结果】\n"
        f"{state['analysis_result']}\n\n"

        f"【审查结果】\n"
        f"{state['review_result']}"
    )

    return {
        "final_answer": final_answer
    }


# ============================================================
# 8. Build Graph
# ============================================================

builder = StateGraph(MainState)

builder.add_node("supervisor", supervisor)
builder.add_node("research", research)
builder.add_node("analyst", analyst)
builder.add_node("reviewer", reviewer)
builder.add_node("aggregator", aggregator)

builder.add_edge(START, "supervisor")      #主管是节点直接普通边触发






# ------------------------------------------------------------
# 注意：
# LangGraph 中更直接的方式是让 Supervisor 返回 Send，
# 所以这里使用一个普通入口节点进行并行分发。
# ------------------------------------------------------------



# Research / Analyst / Reviewer → Aggregator
builder.add_edge("research", "aggregator")
builder.add_edge("analyst", "aggregator")
builder.add_edge("reviewer", "aggregator")


# Aggregator → END
builder.add_edge("aggregator", END)


# ============================================================
# 9. Compile
# ============================================================

graph = builder.compile()


# ============================================================
# 10. Test
# ============================================================

result = graph.invoke({
    "user_task": "人工智能在教育领域的应用",

    "research_result": "",
    "analysis_result": "",
    "review_result": "",

    "final_answer": ""
})


print("\n==============================")
print("最终结果")
print("==============================")

print(result["final_answer"])