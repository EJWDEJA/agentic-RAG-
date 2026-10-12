import os      #因为一个提示词bug导致pydantic校验失败nxtr agentLLM不知道返回什么返回了None导致校验失败

from typing import TypedDict, Literal

from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI

from langgraph.graph import StateGraph, START, END
from langgraph.types import Command


# ============================================================
# 1. Supervisor Decision Schema
# ============================================================

class SupervisorDecision(BaseModel):

    next_agent: Literal["research", "writing"] = Field(             #这个llm真的不一定会填在最后一次到主管后他的值就没有意义的，agent不会执行他会直接执行true     end，他存在的意义就是让输出值通过校验所以后面优化了让finish在最后一次给他顶上了
        description="下一步执行的 Agent，只能是 research 或 writing"
    )

    reason: str = Field(
        description="说明为什么做出这个决定"     #这个llm一定会生成填写
    )

    finished: bool = Field(
        description="任务是否已经完成"
    )


# ============================================================
# 2. Global State
# ============================================================

class State(TypedDict):

    task: str

    supervisor_reason: str

    research_result: str

    draft: str


# ============================================================
# 3. LLM
# ============================================================

llm = ChatOpenAI(
    model="deepseek-v4-flash",
    api_key=os.environ["ANTHROPIC_AUTH_TOKEN"],
    base_url="https://api.deepseek.com"
)


# ============================================================
# 4. Structured LLM
# ============================================================

structured_llm = llm.with_structured_output(
    SupervisorDecision,
    method="json_mode"
)


# ============================================================
# 5. 根据当前 State 计算合法动作空间
# ============================================================

def get_allowed_actions(
    state: State
):

    # --------------------------------------------------------
    # 还没有研究结果
    # --------------------------------------------------------

    if not state["research_result"]:

        return ["research"]

    # --------------------------------------------------------
    # 有研究结果，但没有文章
    # --------------------------------------------------------

    if not state["draft"]:

        return ["writing"]

    # --------------------------------------------------------
    # 研究和文章都已经完成
    # --------------------------------------------------------

    return ["finish"]


# ============================================================
# 6. Supervisor
# ============================================================

def supervisor(
    state: State
) -> Command:

    print("\n")
    print("=" * 60)
    print("执行 Supervisor")
    print("=" * 60)

    # --------------------------------------------------------
    # 由程序根据 State 决定：
    # 当前允许 LLM 做什么
    # --------------------------------------------------------

    allowed_actions = get_allowed_actions(state)

    print("程序计算出的合法动作：")
    print(allowed_actions)

    # --------------------------------------------------------
    # 根据当前 State 给 LLM 提供合法动作空间
    # --------------------------------------------------------

    prompt = f"""
你是一个 Supervisor。

任务：
{state["task"]}

当前 research_result：
{state["research_result"]}

当前 draft：
{state["draft"]}

当前程序允许的动作只有：

{allowed_actions}

请从程序允许的动作中选择下一步。

规则：

如果允许动作中包含 research：
next_agent 必须选择 research
finished 必须为 false

如果允许动作中包含 writing：
next_agent 必须选择 writing
finished 必须为 false

如果允许动作是 finish：
finished 必须为 true
next_agent 必须选择writing    ####################################################

必须返回 JSON。

JSON 必须包含：

next_agent
reason
finished

next_agent 只能是：
research
writing
"""

    # --------------------------------------------------------
    # LLM 决策
    # --------------------------------------------------------

    decision = structured_llm.invoke(prompt)

    print("\nLLM 返回：")
    print("next_agent =", decision.next_agent)
    print("reason     =", decision.reason)
    print("finished   =", decision.finished)

    # ========================================================
    # 7. 程序最终安全检查
    # ========================================================

    # --------------------------------------------------------
    # 情况 A：LLM 声称完成
    # --------------------------------------------------------

    if decision.finished:

        if "finish" not in allowed_actions:

            raise ValueError(
                "安全检查失败：LLM 判断任务完成，"
                "但当前 State 不允许 finish。"
            )

        print("\n程序最终检查：允许完成 → END")

        return Command(
            goto=END,
            update={
                "supervisor_reason": decision.reason
            }
        )

    # --------------------------------------------------------
    # 情况 B：LLM 选择 Research
    # --------------------------------------------------------

    if decision.next_agent == "research":

        if "research" not in allowed_actions:

            raise ValueError(
                "安全检查失败：LLM 选择 research，"
                "但当前 State 不允许 research。"
            )

        print("\n程序最终检查：允许 research")

        return Command(
            goto="research",
            update={
                "supervisor_reason": decision.reason
            }
        )

    # --------------------------------------------------------
    # 情况 C：LLM 选择 Writing
    # --------------------------------------------------------

    if decision.next_agent == "writing":

        if "writing" not in allowed_actions:

            raise ValueError(
                "安全检查失败：LLM 选择 writing，"
                "但当前 State 不允许 writing。"
            )

        print("\n程序最终检查：允许 writing")

        return Command(
            goto="writing",
            update={
                "supervisor_reason": decision.reason
            }
        )

    raise ValueError(
        f"非法 Supervisor Decision：{decision}"
    )


# ============================================================
# 8. Research Agent
# ============================================================

def research_agent(
    state: State
):

    print("\n")
    print("-" * 60)
    print("执行 Research Agent")
    print("-" * 60)

    result = (
        "人工智能可以通过智能教学、个性化学习、"
        "自动批改和学习分析等方式应用于教育领域。"
    )

    print("Research 完成")

    return {
        "research_result": result
    }


# ============================================================
# 9. Writing Agent
# ============================================================

def writing_agent(
    state: State
):

    print("\n")
    print("-" * 60)
    print("执行 Writing Agent")
    print("-" * 60)

    draft = (
        "人工智能正在改变教育方式。\n"
        f"研究结果：{state['research_result']}"
    )

    print("Writing 完成")

    return {
        "draft": draft
    }


# ============================================================
# 10. Build Graph
# ============================================================

builder = StateGraph(State)

builder.add_node(
    "supervisor",
    supervisor
)

builder.add_node(
    "research",
    research_agent
)

builder.add_node(
    "writing",
    writing_agent
)


# ============================================================
# 11. Graph
# ============================================================

builder.add_edge(
    START,
    "supervisor"
)

builder.add_edge(
    "research",
    "supervisor"
)

builder.add_edge(
    "writing",
    "supervisor"
)


# ============================================================
# 12. Compile
# ============================================================

graph = builder.compile()


# ============================================================
# 13. Test
# ============================================================

result = graph.invoke({

    "task": "研究人工智能在教育领域的应用，并整理成一篇文章",

    "supervisor_reason": "",

    "research_result": "",

    "draft": ""
})


# ============================================================
# 14. Final Result
# ============================================================

print("\n")
print("=" * 60)
print("最终结果")
print("=" * 60)

print("\nResearch Result:")
print(result["research_result"])

print("\nDraft:")
print(result["draft"])

print("\nSupervisor Reason:")
print(result["supervisor_reason"])