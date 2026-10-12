from typing import Literal, TypedDict                #此处将上一版的主管规则变成了llm   ，这里并没有明确说或者教学进行迭代

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END


# ============================================================
# 1. LLM
# ============================================================

import os

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
        description="下一步要执行的动作，只能是 research、writer 或 finish"
    )
    task: str = Field(
        description="交给下一步 Agent 的具体任务"
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

def supervisor(state: MainState):

    print("\n===== Supervisor =====")

    research_report = state["research_report"]

    # --------------------------------------------------------
    # 第一次进入 Supervisor
    # --------------------------------------------------------

    if not research_report["summary"]:

        task = AgentTask(
            target="research",
            task=f"研究以下问题：{state['user_task']}"
        )

        print("Supervisor → Research")

        return {
            "agent_task": task
        }

    # --------------------------------------------------------
    # Research 完成后再次进入 Supervisor
    # --------------------------------------------------------
                                                                        #此处替代了上一版硬编码的else部分，由提示词驱动llm动态决定下一步进行深度研究还是，开始写作又或者结束（此处由于给writer配置了end使用发挥不出来finish逻辑里面有没有配置写作在此处是为后续完善做铺垫）
    prompt = f"""                                                           

你现在是一个多 Agent 系统中的 Supervisor。

用户任务：
{state["user_task"]}

Research Agent 已经完成研究。

研究结果：
摘要：
{research_report["summary"]}

关键点：
{research_report["key_points"]}

现在请决定下一步应该做什么。

允许的 target 只有：
- research：研究结果还不充分，需要继续研究
- writer：研究结果已经足够，可以交给 Writer
- finish：当前任务已经可以直接结束

你的输出必须严格包含以下两个 JSON 字段：

"target"
"task"

注意：
- 字段名必须使用 target，绝对不要使用 action
- 字段名必须使用 task
- 不允许输出 action 字段
- target 必须是 research、writer、finish 三者之一

例如，如果决定交给 Writer，必须输出：

{{
    "target": "writer",
    "task": "根据 Research Agent 提供的研究结果撰写最终回答"
}}

如果决定继续研究，必须输出：

{{
    "target": "research",
    "task": "进一步研究当前研究结果中缺失的内容"
}}

如果决定结束，必须输出：

{{
    "target": "finish",
    "task": "研究结果已经足够完成当前任务"
}}

请严格输出 JSON。
"""

    decision = structured_llm.invoke(prompt)

    print(f"Supervisor 决策：{decision}")



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


# Supervisor → Dispatcher
builder.add_conditional_edges(
    "supervisor",
    dispatcher,
    {
        "research": "research",
        "writer": "writer",
        "finish": "finish",
    }
)


# Research 完成后重新回 Supervisor
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
    "draft": ""
})


# ============================================================
# 14. Result
# ============================================================

print("\n==============================")
print("最终结果")
print("==============================")

print(result)