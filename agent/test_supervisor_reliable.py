import os                           #这个文件把，程序约束（活动空间）----pydantic校验（参数类型和参数值校验）----工作流校验（判断是否符合当前状态）----（校验失败出现）结合到一起去了

from typing import TypedDict, Literal

from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI

from langgraph.graph import StateGraph, START, END
from langgraph.types import Command


# ============================================================
# 1. Supervisor Decision Schema
# ============================================================

class SupervisorDecision(BaseModel):

    action: Literal[
        "research",
        "writing",
        "finish"
    ] = Field(
        description="下一步动作，只能是 research、writing 或 finish"
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
# 6. Workflow Validation
# ============================================================

def validate_workflow_decision(
    decision: SupervisorDecision,     #13负责测试决策合法性，12负责测试决策是否符合当前状态
    state: State,
    allowed_actions: list[str]
):

    # --------------------------------------------------------
    # 第一层：
    # Action 是否属于程序当前允许的动作？
    # --------------------------------------------------------

    if decision.action not in allowed_actions:

        raise ValueError(
            f"Action 不合法：{decision.action}；"
            f"当前允许的动作：{allowed_actions}"
        )

    # --------------------------------------------------------
    # 第二层：
    # 根据当前 State 检查 Workflow 逻辑
    # --------------------------------------------------------

    if (
        not state["research_result"]
        and decision.action != "research"
    ):

        raise ValueError(
            "Workflow 决策非法：当前没有 research_result，"
            "只能执行 research。"
        )

    if (
        state["research_result"]
        and not state["draft"]
        and decision.action != "writing"
    ):

        raise ValueError(
            "Workflow 决策非法：已经完成 research，"
            "但还没有 draft，只能执行 writing。"
        )

    if (
        state["research_result"]
        and state["draft"]
        and decision.action != "finish"
    ):

        raise ValueError(
            "Workflow 决策非法：research 和 writing 都已完成，"
            "此时只能 finish。"
        )

    print("Workflow Validation：通过")


# ============================================================
# 7. Supervisor Decision + Retry / Repair
# ============================================================

def get_supervisor_decision(
    state: State
) -> SupervisorDecision:

    max_retries = 3

    allowed_actions = get_allowed_actions(state)

    print("\n程序计算出的 Allowed Actions：")
    print(allowed_actions)

    # --------------------------------------------------------
    # 第一次 Prompt
    #
    # 故意让第一次决策容易产生 Workflow 错误，
    # 用来测试 Repair。
    # --------------------------------------------------------

    prompt = f"""
你是一个 Supervisor。

任务：
{state["task"]}

当前 State：

research_result：
{state["research_result"]}

draft：
{state["draft"]}

程序允许的动作：

{allowed_actions}

必须返回 JSON。

JSON 必须包含：

action
reason

action 只能是：

research
writing
finish

为了测试 Workflow Validation，
第一次请选择 writing。

注意：
程序之后会检查这个决定是否符合当前 State。
"""

    # ========================================================
    # Retry Loop
    # ========================================================

    for attempt in range(1, max_retries + 1):

        print("\n")
        print("=" * 60)
        print(f"Supervisor 第 {attempt} 次决策")
        print("=" * 60)

        try:

            # ------------------------------------------------
            # 1. LLM + Structured Output
            # ------------------------------------------------

            decision = structured_llm.invoke(prompt)

            print("\nSchema Validation：通过")

            print("action =", decision.action)
            print("reason =", decision.reason)

            # ------------------------------------------------
            # 2. Action + Workflow Validation
            # ------------------------------------------------

            validate_workflow_decision(
                decision,
                state,
                allowed_actions
            )

            # ------------------------------------------------
            # 3. 全部通过
            # ------------------------------------------------

            print("\nSupervisor Decision 最终通过")

            return decision

        except Exception as e:            #前面抛出了大量校验错误这一步就是捕获他们防止程序崩溃的

            print("\n验证失败！")

            print("错误类型：")
            print(type(e).__name__)

            print("\n错误信息：")
            print(e)

            # ------------------------------------------------
            # 达到最大重试次数
            # ------------------------------------------------

            if attempt == max_retries:

                print("\n达到最大 Retry 次数。")
                raise

            # ------------------------------------------------
            # Repair Prompt
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

程序当前允许的动作：

{allowed_actions}

你刚才的决策没有通过程序验证。

错误信息：

{e}

请重新生成正确的 Supervisor Decision。

必须严格遵守：

1. action 必须从当前允许动作中选择：

{allowed_actions}

2. 如果允许动作是 research：
   action = research

3. 如果允许动作是 writing：
   action = writing

4. 如果允许动作是 finish：
   action = finish

5. 必须返回合法 JSON。

6. JSON 必须包含：

action
reason

只输出 JSON。
"""

            print("\n已生成 Repair Prompt，准备重新请求 LLM...")


# ============================================================
# 8. Supervisor
# ============================================================

def supervisor(
    state: State
) -> Command:

    decision = get_supervisor_decision(state)

    print("\n最终 Supervisor Decision：")
    print(decision)

    # --------------------------------------------------------
    # Command Routing
    # --------------------------------------------------------

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
    print("Research Agent")
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
    print("Writing Agent")
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