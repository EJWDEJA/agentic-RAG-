import os       #本质上就是对规范，校验llm发给surpervsion的数据，减少aurpervsion的解析器压力让他指路更加规范

from typing import TypedDict

from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI

from langgraph.graph import StateGraph, START, END
from langgraph.types import Command


# ============================================================
# 1. Supervisor Decision Schema
# ============================================================

class SupervisorDecision(BaseModel):      #只是校验不被更新，不作为状态

    next_agent: str = Field(
        description="下一步执行的 Agent，只能是 research 或 writing"
    )

    reason: str = Field(
        description="说明为什么选择这个 Agent"
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
# 5. Supervisor
# ============================================================

def supervisor(
    state: State
) -> Command:

    print("\n")
    print("=" * 60)
    print("执行 Supervisor")
    print("=" * 60)

    prompt = f"""
你是一个 Supervisor，负责决定下一步执行哪个 Agent。

当前任务：
{state["task"]}

已经完成的研究结果：
{state["research_result"]}

已经生成的文章：
{state["draft"]}

请决定下一步应该执行哪个 Agent。

可选择：

research
writing

决策规则：

如果还没有研究结果，选择 research。

如果已经有研究结果但还没有文章，选择 writing。

如果文章已经生成，则选择 writing。

必须返回 JSON。

JSON 必须包含：

next_agent
reason

next_agent 只能是 research 或 writing。
"""

    # --------------------------------------------------------
    # LLM 做决策
    # --------------------------------------------------------

    decision = structured_llm.invoke(prompt)

    print("Supervisor Decision:")
    print("next_agent =", decision.next_agent)
    print("reason     =", decision.reason)

    # --------------------------------------------------------
    # 关键：
    # LLM 不直接控制 Graph
    #
    # LLM → SupervisorDecision
    #       ↓
    # Python → Command
    # --------------------------------------------------------

    if decision.next_agent == "research":

        return Command(
            goto="research",
            update={
                "supervisor_reason": decision.reason
            }
        )

    elif decision.next_agent == "writing":

        return Command(
            goto="writing",
            update={
                "supervisor_reason": decision.reason
            }
        )

    else:

        raise ValueError(
            f"非法 Agent：{decision.next_agent}"
        )


# ============================================================
# 6. Research Agent
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
# 7. Writing Agent
# ============================================================

def writing_agent(
    state: State
):

    print("\n")
    print("-" * 60)
    print("执行 Writing Agent")
    print("-" * 60)

    draft = f"""
人工智能在教育领域具有广泛的应用价值。

根据研究结果：
{state["research_result"]}

人工智能可以帮助教师提高教学效率，
同时为学生提供更加个性化的学习体验。
"""

    print("Writing 完成")

    return {
        "draft": draft
    }


# ============================================================
# 8. Build Graph
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
# 9. Entry
# ============================================================

builder.add_edge(
    START,
    "supervisor"
)


# ============================================================

# 10. Research 完成后回 Supervisor
# ============================================================

builder.add_edge(
    "research",
    "supervisor"
)


# ============================================================
# 11. Writing 完成后结束
# ============================================================

builder.add_edge(
    "writing",
    END
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

print("Supervisor Reason:")
print(result["supervisor_reason"])

print("\nResearch Result:")
print(result["research_result"])

print("\nDraft:")
print(result["draft"])
