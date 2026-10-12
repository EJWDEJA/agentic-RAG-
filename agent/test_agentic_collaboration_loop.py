from typing import Literal, TypedDict             #这里llm动态研究，并应用到了迭代技术，设置了最大迭代上限，防止llm无限迭代，属于智能体
                                                  #但是这里有一个问题仅仅通过自然语言描述让LLM自主判断是否完成研究任务就很容易陷入死循环，没有明确的终止条件（这也是后续设置很多评估机制的作用，就是为了判断LLM回答是否已经足够好）

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END

import os


# ============================================================
# 1. LLM
# ============================================================

llm = ChatOpenAI(
    model="deepseek-v4-flash",
    api_key=os.environ["ANTHROPIC_AUTH_TOKEN"],
    base_url="https://api.deepseek.com"
)


# ============================================================
# 2. Agent Task
# ============================================================

class AgentTask(BaseModel):
    target: Literal["research", "writer", "finish"] = Field(
        description="下一步动作，只能是 research、writer 或 finish"
    )
    task: str = Field(
        description="下一步 Agent 要执行的任务"
    )


# ============================================================
# 3. Research Report
# ============================================================

class ResearchReport(TypedDict):
    summary: str
    key_points: list[str]


# ============================================================
# 4. Main State
# ============================================================

class MainState(TypedDict):
    user_task: str
    agent_task: AgentTask
    research_report: ResearchReport
    research_round: int
    draft: str


# ============================================================
# 5. Structured LLM
# ============================================================

structured_llm = llm.with_structured_output(
    AgentTask,
    method="json_mode"
)


# ============================================================
# 6. Supervisor
# ============================================================

MAX_RESEARCH_ROUNDS = 3


def supervisor(state: MainState):

    print("\n===== Supervisor =====")

    report = state["research_report"]
    round_number = state["research_round"]

    # --------------------------------------------------------
    # 第一次进入：开始研究
    # --------------------------------------------------------

    if not report["summary"]:

        print("首次委托 Research")

        return {
            "agent_task": AgentTask(
                target="research",
                task=f"研究：{state['user_task']}"
            )
        }

    # --------------------------------------------------------
    # 最大轮次保护
    # --------------------------------------------------------

    if round_number >= MAX_RESEARCH_ROUNDS:

        print("达到最大研究轮次，强制进入 Writer")

        return {
            "agent_task": AgentTask(
                target="writer",
                task="根据当前研究结果撰写最终答案"
            )
        }

    # --------------------------------------------------------
    # LLM 判断下一步
    # --------------------------------------------------------

    prompt = f"""
你是一个多 Agent 系统中的 Supervisor。

用户任务：
{state["user_task"]}

当前已经进行了 {round_number} 轮研究。

研究结果：

摘要：
{report["summary"]}

关键点：
{report["key_points"]}

请决定下一步执行的 Agent。

你必须输出一个 AgentTask。

AgentTask 有两个字段：

target：
只能填写以下三个值之一：
- "research"
- "writer"
- "finish"

task：
填写下一步具体要执行的任务。

决策规则：

如果当前研究还不充分：
target 必须填写 "research"
task 必须说明还需要继续研究什么。

如果当前研究已经足够：
target 必须填写 "writer"
task 必须说明 Writer 应该如何利用当前研究结果。

如果任务已经可以结束：
target 必须填写 "finish"
task 必须说明结束原因。

非常重要：
字段名称必须严格使用 "target" 和 "task"。
不要使用 "action"。
不要输出其他字段。

请严格输出 JSON。
"""

    decision = structured_llm.invoke(prompt)

    print(f"LLM 决策：{decision}")

    return {
        "agent_task": decision
    }


# ============================================================
# 7. Dispatcher
# ============================================================

def dispatcher(state: MainState):

    target = state["agent_task"].target

    print(f"Dispatcher → {target}")

    return target


# ============================================================
# 8. Research Agent
# ============================================================

def research(state: MainState):

    round_number = state["research_round"] + 1

    print(f"\n===== Research Agent 第 {round_number} 轮 =====")

    # 模拟每一轮研究不断增加信息
    if round_number == 1:

        report = {
            "summary": "人工智能正在改变教育方式。",
            "key_points": [
                "个性化学习"
            ]
        }

    elif round_number == 2:

        report = {
            "summary": "人工智能正在改变教育方式，并提高教学效率。",
            "key_points": [
                "个性化学习",
                "智能教学辅助"
            ]
        }

    else:

        report = {
            "summary": (
                "人工智能正在改变教育方式，"
                "可以提高教学效率并辅助教育管理。"
            ),
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
# 9. Writer Agent
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
# 10. Finish Agent
# ============================================================

def finish(state: MainState):

    print("\n===== Finish =====")

    return {}


# ============================================================
# 11. Build Graph
# ============================================================

builder = StateGraph(MainState)

builder.add_node("supervisor", supervisor)
builder.add_node("research", research)
builder.add_node("writer", writer)
builder.add_node("finish", finish)

builder.add_edge(START, "supervisor")


builder.add_conditional_edges(
    "supervisor",
    dispatcher,
    {
        "research": "research",
        "writer": "writer",
        "finish": "finish",
    }
)


# Research 完成后重新进入 Supervisor
builder.add_edge("research", "supervisor")


# Writer / Finish 结束
builder.add_edge("writer", END)
builder.add_edge("finish", END)


# ============================================================
# 12. Compile
# ============================================================

graph = builder.compile()


# ============================================================
# 13. Test
# ============================================================

result = graph.invoke({
    "user_task": "人工智能在教育领域的应用",
    "agent_task": AgentTask(
        target="research",
        task=""
    ),
    "research_report": {
        "summary": "",
        "key_points": []
    },
    "research_round": 0,
    "draft": ""
})


# ============================================================
# 14. Result
# ============================================================

print("\n==============================")
print("最终结果")
print("==============================")

print(result)