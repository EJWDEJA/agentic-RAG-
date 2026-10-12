import os          #之前的结构化输出在pydantic校验阶段只能校验字段类型，这次pydantic加上了字段具体字符串值（业务值）的校验，让校验更加完善


from typing import TypedDict, Literal

from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI

from langgraph.graph import StateGraph, START, END
from langgraph.types import Command


# ============================================================
# 1. Supervisor Decision Schema
# ============================================================

class SupervisorDecision(BaseModel):

    # 不再只是 str
    # 而是限制只能出现两个合法值
    next_agent: Literal["research", "writing"] = Field(      #核心就是这一行字面量类型强制规定了变量必须出自Literal里面的几个之一（此处情况是之一不固定）
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
# 5. Supervisor Decision + Retry
# ============================================================

def get_supervisor_decision(
    state: State
) -> SupervisorDecision:

    max_retries = 3

    prompt = f"""
你是一个 Supervisor。

任务：
{state["task"]}

当前研究结果：
{state["research_result"]}

当前文章：
{state["draft"]}

请决定下一步。

可用 Agent：

research
writing

规则：

1. 如果还没有研究结果：
   next_agent = research
   finished = false

2. 如果已经有研究结果但没有文章：
   next_agent = writing
   finished = false

3. 如果研究结果和文章都已经完成：
   finished = true

必须返回 JSON。

JSON 必须包含：

next_agent
reason
finished

next_agent 必须是 research 或 writing。

为了测试业务值校验，
第一次请尝试返回：

next_agent = database

这不是合法 Agent。
程序应该检测到这个错误并要求你修复。
"""

    for attempt in range(1, max_retries + 1):

        print("\n")
        print("=" * 60)
        print(f"Supervisor 第 {attempt} 次决策")
        print("=" * 60)

        try:

            decision = structured_llm.invoke(prompt)

            print("Schema 校验成功")
            print("next_agent =", decision.next_agent)
            print("reason     =", decision.reason)
            print("finished   =", decision.finished)

            return decision

        except Exception as e:

            print("\n决策校验失败！")
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

你的上一次输出没有通过 SupervisorDecision 的校验。

错误信息：

{e}

请修复你的输出。

严格要求：

1. 必须返回合法 JSON
2. 必须包含：
   - next_agent
   - reason
   - finished

3. next_agent 只能是：
   - research
   - writing

4. 不允许：
   - database
   - search
   - planner
   - 其他任何值

5. finished 必须是 true 或 false。

只输出 JSON。
"""

            print("\n已经生成 Repair Prompt，准备重新请求 LLM...")


# ============================================================

# 6. Supervisor
# ============================================================

def supervisor(
    state: State
) -> Command:

    decision = get_supervisor_decision(state)

    print("\n最终 Supervisor Decision：")
    print(decision)

    # --------------------------------------------------------
    # 任务完成
    # --------------------------------------------------------

    if decision.finished:

        print("\n→ Supervisor 判断任务完成")

        return Command(
            goto=END,
            update={
                "supervisor_reason": decision.reason
            }
        )

    # --------------------------------------------------------
    # 未完成：执行 Research
    # --------------------------------------------------------

    if decision.next_agent == "research":

        print("\n→ Supervisor 决定执行 Research")

        return Command(
            goto="research",
            update={
                "supervisor_reason": decision.reason
            }
        )

    # --------------------------------------------------------
    # 未完成：执行 Writing
    # --------------------------------------------------------

    if decision.next_agent == "writing":

        print("\n→ Supervisor 决定执行 Writing")

        return Command(
            goto="writing",
            update={
                "supervisor_reason": decision.reason
            }
        )


# ============================================================
# 7. Research Agent
# ============================================================

def research_agent(
    state: State
):

    print("\n")
    print("-" * 60)
    print("执行 Research Agent")
    print("-" * 60)

    return {
        "research_result": (
            "人工智能可以通过智能教学、"
            "个性化学习、自动批改和学习分析应用于教育领域。"
        )
    }


# ============================================================
# 8. Writing Agent
# ============================================================

def writing_agent(
    state: State
):

    print("\n")
    print("-" * 60)
    print("执行 Writing Agent")
    print("-" * 60)

    return {
        "draft": (
            "人工智能正在改变教育方式。"
            f"\n研究结果：{state['research_result']}"
        )
    }


# ============================================================
# 9. Build Graph
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
# 10. Graph Edges
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
# 11. Compile
# ============================================================

graph = builder.compile()


# ============================================================
# 12. Test
# ============================================================

result = graph.invoke({

    "task": "研究人工智能在教育领域的应用，并整理成一篇文章",

    "supervisor_reason": "",

    "research_result": "",

    "draft": ""
})


# ============================================================
# 13. Final Result
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
