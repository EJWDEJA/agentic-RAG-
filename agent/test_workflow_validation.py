import os      #此代码因为llm无法遵从提示词retry导致尝试次数过多，只是把主动权交给llm没有约束llm的行为导致的报错

from typing import TypedDict, Literal

from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI

from langgraph.graph import StateGraph, START, END
from langgraph.types import Command


# ============================================================
# 1. Supervisor Decision Schema
# ============================================================

class SupervisorDecision(BaseModel):

    next_agent: Literal["research", "writing"] = Field(
        description="下一步执行的 Agent，只能是 research 或 writing"
    )

    reason: str = Field(
        description="说明为什么做出这个决定"
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
# 5. Workflow Validation
#
# 注意：
# 这里不是检查 Schema。
#
# Schema 已经由 Pydantic 检查。
#
# 这里检查：
# “这个决策是否符合当前 State？”
# ============================================================

def validate_workflow_decision(
    decision: SupervisorDecision,
    state: State
):

    # --------------------------------------------------------
    # 情况 1：
    # 没有研究结果，却准备直接 Writing
    # --------------------------------------------------------

    if (
        not state["research_result"]
        and decision.next_agent == "writing"
    ):

        raise ValueError(
            "Workflow 决策非法：当前还没有 research_result，"
            "不能直接执行 writing。"
        )

    # --------------------------------------------------------
    # 情况 2：
    # 没有研究结果，却认为任务已经完成
    # --------------------------------------------------------

    if (
        not state["research_result"]
        and decision.finished
    ):

        raise ValueError(
            "Workflow 决策非法：当前还没有 research_result，"
            "任务不能完成。"
        )

    # --------------------------------------------------------
    # 情况 3：
    # 已经有研究结果，但没有文章，却认为任务完成
    # --------------------------------------------------------

    if (
        state["research_result"]
        and not state["draft"]
        and decision.finished
    ):

        raise ValueError(
            "Workflow 决策非法：已经完成研究，但还没有 draft，"
            "任务不能完成。"
        )

    # --------------------------------------------------------
    # 情况 4：
    # 已经有研究结果，但没有文章，却又选择 research
    # --------------------------------------------------------

    if (
        state["research_result"]
        and not state["draft"]
        and decision.next_agent == "research"
    ):

        raise ValueError(
            "Workflow 决策非法：research_result 已经存在，"
            "当前应该执行 writing，而不是 research。"
        )

    print("Workflow Validation：通过")


# ============================================================
# 6. 获取合法 Supervisor Decision
# ============================================================

def get_supervisor_decision(
    state: State
) -> SupervisorDecision:

    max_retries = 3

    # --------------------------------------------------------
    # 第一次故意制造一个：
    #
    # Schema 合法
    # 但是 Workflow 非法
    #
    # 当前 research_result 是空的，
    # 但要求 LLM 选择 writing。
    # --------------------------------------------------------

    prompt = f"""
你是一个 Supervisor。

任务：
{state["task"]}

当前 research_result：
{state["research_result"]}

当前 draft：
{state["draft"]}

请决定下一步。

可用 Agent：

research
writing

必须返回 JSON。

必须包含：

next_agent
reason
finished

next_agent 只能是 research 或 writing。
finished 必须是 true 或 false。

为了测试 Workflow Validation，
请第一次选择：

next_agent = writing
finished = false

注意：
这个选择在 Schema 层面是合法的，
但程序会进一步检查它是否符合当前 State。
"""

    for attempt in range(1, max_retries + 1):

        print("\n")
        print("=" * 60)
        print(f"Supervisor 第 {attempt} 次决策")
        print("=" * 60)

        try:

            # ------------------------------------------------
            # 第一步：LLM 输出
            # ------------------------------------------------

            decision = structured_llm.invoke(prompt)

            print("\nSchema Validation：通过")

            print("next_agent =", decision.next_agent)
            print("reason     =", decision.reason)
            print("finished   =", decision.finished)

            # ------------------------------------------------
            # 第二步：Workflow Validation
            # ------------------------------------------------

            validate_workflow_decision(
                decision,
                state
            )

            # ------------------------------------------------
            # 两层验证全部通过
            # ------------------------------------------------

            return decision

        except Exception as e:

            print("\n验证失败！")
            print("错误类型：", type(e).__name__)
            print("错误信息：")
            print(e)

            if attempt == max_retries:

                print("\n达到最大重试次数。")
                raise

            # ------------------------------------------------
            # Repair
            # ------------------------------------------------

            prompt = f"""
你是一个 Supervisor。

任务：
{state["task"]}

当前 State：

research_result：
{state["research_result"]}

draft：
{state["draft"]}

你刚才的 Supervisor Decision 没有通过
Workflow Validation。

Supervisor Decision信息：

｛decision｝


错误信息：

{e}

请重新做出决策。

必须遵守：

1. 如果 research_result 为空：
   next_agent 必须是 research
   finished 必须是 false

2. 如果 research_result 不为空，
   但 draft 为空：
   next_agent 必须是 writing
   finished 必须是 false

3. 如果 research_result 和 draft 都不为空：
   finished 必须是 true

4. next_agent 只能是：
   research
   writing

5. 必须返回合法 JSON。

只输出 JSON。
"""

            print("\n已经生成 Repair Prompt...")


# ============================================================
# 7. Supervisor
# ============================================================

def supervisor(
    state: State
) -> Command:

    decision = get_supervisor_decision(state)

    print("\n最终合法 Decision：")
    print(decision)

    # --------------------------------------------------------
    # 完成
    # --------------------------------------------------------

    if decision.finished:

        print("\n→ Workflow 完成")

        return Command(
            goto=END,
            update={
                "supervisor_reason": decision.reason
            }
        )

    # --------------------------------------------------------
    # Research
    # --------------------------------------------------------

    if decision.next_agent == "research":

        print("\n→ 执行 Research")

        return Command(
            goto="research",
            update={
                "supervisor_reason": decision.reason
            }
        )

    # --------------------------------------------------------
    # Writing
    # --------------------------------------------------------

    if decision.next_agent == "writing":

        print("\n→ 执行 Writing")

        return Command(
            goto="writing",
            update={
                "supervisor_reason": decision.reason
            }
        )


# ============================================================
# 8. Research Agent
# ============================================================

def research_agent(
    state: State
):

    print("\n")
    print("-" * 60)
    print("Research Agent")
    print("-" * 60)

    return {
        "research_result": (
            "人工智能可以通过智能教学、个性化学习、"
            "自动批改和学习分析等方式应用于教育领域。"
        )
    }


# ============================================================
# 9. Writing Agent
# ============================================================

def writing_agent(
    state: State
):

    print("\n")
    print("-" * 60)
    print("Writing Agent")
    print("-" * 60)

    return {
        "draft": (
            "人工智能正在改变教育方式。\n"
            f"研究结果：{state['research_result']}"
        )
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