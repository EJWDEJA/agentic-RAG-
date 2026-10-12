import os        #外部程序限定可执行操作范围，把范围交给llm，让llm不至于自由发挥提高输出准确性。总的来说就是进入llm前约束可选范围让llm知道，llm返回结果以后进行pydantic校验。这种设计比第一版优雅第一版会产生一个next_agent =writing多余项目的仅仅是通过pydantic参数校验

from typing import TypedDict, Literal

from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI

from langgraph.graph import StateGraph, START, END
from langgraph.types import Command


# ============================================================
# 1. Supervisor Decision
# ============================================================

class SupervisorDecision(BaseModel):

    action: Literal[
        "research",
        "writing",
        "finish"
    ] = Field(
        description=(
            "下一步动作，只能是 research、writing 或 finish"
        )
    )

    reason: str = Field(
        description="说明为什么选择这个动作"
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
# 5. 计算当前允许的动作
# ============================================================

def get_allowed_actions(
    state: State
):

    if not state["research_result"]:

        return ["research"]

    if not state["draft"]:

        return ["writing"]

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
    # 程序根据 State 决定合法动作
    # --------------------------------------------------------

    allowed_actions = get_allowed_actions(state)

    print("程序允许的动作：")
    print(allowed_actions)

    # --------------------------------------------------------
    # 告诉 LLM 当前允许哪些动作
    # --------------------------------------------------------

    prompt = f"""
你是一个 Supervisor。

任务：
{state["task"]}

当前 research_result：
{state["research_result"]}

当前 draft：
{state["draft"]}

程序当前允许的动作：

{allowed_actions}

你只能从这些允许的动作中选择一个。

规则：

- 如果允许 research，选择 research
- 如果允许 writing，选择 writing
- 如果允许 finish，选择 finish

必须返回 JSON。

JSON 必须包含：

action
reason

action 必须是：

research
writing
finish
"""

    # --------------------------------------------------------
    # LLM 决策
    # --------------------------------------------------------

    decision = structured_llm.invoke(prompt)

    print("\nLLM Decision：")
    print("action =", decision.action)
    print("reason =", decision.reason)

    # ========================================================
    # 7. 程序最终安全检查
    # ========================================================

    if decision.action not in allowed_actions:

        raise ValueError(
            f"LLM 返回了当前不允许的动作："
            f"{decision.action}，"
            f"当前允许：{allowed_actions}"
        )

    print("\n程序最终检查：通过")

    # ========================================================
    # 8. 根据合法 Action 路由
    # ========================================================

    if decision.action == "research":

        print("→ Command(goto='research')")

        return Command(
            goto="research",
            update={
                "supervisor_reason": decision.reason
            }
        )

    if decision.action == "writing":

        print("→ Command(goto='writing')")

        return Command(
            goto="writing",
            update={
                "supervisor_reason": decision.reason
            }
        )

    if decision.action == "finish":

        print("→ Command(goto=END)")

        return Command(
            goto=END,
            update={
                "supervisor_reason": decision.reason
            }
        )

    raise ValueError(
        f"未知 Action：{decision.action}"
    )


# ============================================================
# 9. Research Agent
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
# 10. Writing Agent
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
# 11. Build Graph
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
# 12. Graph Edges
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
# 13. Compile
# ============================================================

graph = builder.compile()


# ============================================================
# 14. Test
# ============================================================

result = graph.invoke({

    "task": "研究人工智能在教育领域的应用，并整理成一篇文章",

    "supervisor_reason": "",

    "research_result": "",

    "draft": ""
})


# ============================================================
# 15. Final Result
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