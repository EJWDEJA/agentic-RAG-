import os      #之前的surpervision只能进行任务调度路由，现在加入的控制任务是否终止的机制，通过finish

from typing import TypedDict

from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI

from langgraph.graph import StateGraph, START, END
from langgraph.types import Command


# ============================================================
# 1. Supervisor Decision Schema
# ============================================================

class SupervisorDecision(BaseModel):

    next_agent: str = Field(
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
你是一个 Supervisor，负责管理整个任务。

任务：
{state["task"]}

当前研究结果：
{state["research_result"]}

当前文章：
{state["draft"]}

请决定下一步应该怎么做。

可选择的 Agent：

research
writing

判断规则：

1. 如果还没有研究结果：
   - finished = false
   - next_agent = research

2. 如果已经有研究结果，但还没有文章：
   - finished = false
   - next_agent = writing

3. 如果研究结果和文章都已经完成：
   - finished = true

如果 finished = true，
next_agent 请填写 writing。

必须返回 JSON。

JSON 必须包含：

next_agent
reason
finished

next_agent 只能是 research 或 writing。
finished 必须是 true 或 false。
"""

    # --------------------------------------------------------
    # LLM 进行结构化决策
    # --------------------------------------------------------

    decision = structured_llm.invoke(prompt)

    print("Supervisor Decision:")
    print("next_agent =", decision.next_agent)
    print("reason     =", decision.reason)
    print("finished   =", decision.finished)

    # --------------------------------------------------------
    # 判断任务是否完成
    # --------------------------------------------------------

    if decision.finished:

        print("Supervisor 判断：任务已经完成 → END")

        return Command(
            goto=END,
            update={
                "supervisor_reason": decision.reason
            }
        )

    # --------------------------------------------------------
    # 任务没有完成，继续调度 Agent
    # --------------------------------------------------------

    if decision.next_agent == "research":

        print("Supervisor 决定：→ Research")

        return Command(
            goto="research",
            update={
                "supervisor_reason": decision.reason
            }
        )

    elif decision.next_agent == "writing":

        print("Supervisor 决定：→ Writing")

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

研究表明：
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
# 9. START → Supervisor
# ============================================================

builder.add_edge(
    START,
    "supervisor"
)


# ============================================================
# 10. Research → Supervisor
# ============================================================

builder.add_edge(
    "research",
    "supervisor"
)


# ============================================================
# 11. Writing → Supervisor
#
# 注意：
# Writing 不再直接 END。
#
# 必须重新交给 Supervisor 判断是否完成。
# ============================================================

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

print("\nSupervisor Reason:")
print(result["supervisor_reason"])

print("\nResearch Result:")
print(result["research_result"])

print("\nDraft:")
print(result["draft"])
