import os

from typing import Annotated, TypedDict

from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI

from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, Send


# ============================================================
# 1. LLM
# ============================================================

llm = ChatOpenAI(
    model="deepseek-v4-flash",
    api_key=os.environ["ANTHROPIC_AUTH_TOKEN"],
    base_url="https://api.deepseek.com"
)


# ============================================================
# 2. Supervisor Decision Schema
# ============================================================

class SupervisorDecision(BaseModel):

    next_agent: str = Field(
        description="下一步工作流，只能是 research 或 writing"
    )

    reason: str = Field(
        description="选择该工作流的原因"
    )


# ============================================================
# 3. Structured LLM
# ============================================================

structured_llm = llm.with_structured_output(
    SupervisorDecision,
    method="json_mode"

)


# ============================================================
# 4. Global State
# ============================================================

class MainState(TypedDict):
    task: str
    results: Annotated[list[str], lambda old, new: old + new]
    research_result: str
    draft: str
    final_result: str


# ============================================================
# 5. Supervisor
# ============================================================

def supervisor(state: MainState):

    print("执行 Supervisor")

    prompt = f"""
你是 Multi-Agent Supervisor。

根据用户任务，选择最合适的工作流。

你必须以 JSON 格式输出决策。

JSON 格式必须包含：

{{
    "next_agent": "research 或 writing",
    "reason": "选择该工作流的原因"
}}

可选工作流：

research
适合研究、调查、资料分析。

writing
适合文章写作、内容生成。

用户任务：
{state["task"]}
    """


    decision = structured_llm.invoke(prompt)

    print("Supervisor 决策：")
    print("next_agent =", decision.next_agent)
    print("reason =", decision.reason)

    if decision.next_agent == "research":

        return Command(
            goto="research_dispatcher"
        )

    if decision.next_agent == "writing":

        return Command(
            goto="research_for_writer"
        )

    raise ValueError(
        f"未知的 next_agent：{decision.next_agent}"
    )


# ============================================================
# 6. Research Agents
# ============================================================

def research_agent(state):

    print("执行 Research Agent")

    return {
        "results": [
            f"Research：完成「{state['task']}」的资料研究"
        ]
    }


def analyst_agent(state):

    print("执行 Analyst Agent")

    return {
        "results": [
            f"Analyst：完成「{state['task']}」的分析"
        ]
    }


def reviewer_agent(state):

    print("执行 Reviewer Agent")

    return {
        "results": [
            f"Reviewer：完成「{state['task']}」的审查"
        ]
    }


# ============================================================
# 7. Parallel Dispatcher
# ============================================================

def research_dispatcher(state):

    print("执行 Parallel Dispatcher")

    return Command(
        goto=[
        Send("research_agent", {
            "task": state["task"]
        }),
        Send("analyst_agent", {
            "task": state["task"]
        }),
        Send("reviewer_agent", {
            "task": state["task"]
        })
    ]
    )


# ============================================================
# 8. Aggregator
# ============================================================

def aggregator(state):

    print("执行 Aggregator Agent")

    return {
        "final_result": (
            "研究工作汇总：\n"
            + "\n".join(state["results"])
        )
    }


# ============================================================
# 9. Writing Workflow
# ============================================================

def research_for_writer(state):

    print("执行 Writing Workflow 的 Research")

    return {
        "research_result": (
            f"针对「{state['task']}」完成基础研究"
        )
    }


def writer_agent(state):

    print("执行 Writer Agent")

    return {
        "draft": (
            f"根据研究结果「{state['research_result']}」"
            f"完成「{state['task']}」的草稿"
        )
    }


def writing_finalizer(state):

    print("执行 Writing Finalizer")

    return {
        "final_result": state["draft"]
    }


# ============================================================
# 10. Build Graph
# ============================================================

builder = StateGraph(MainState)

builder.add_node("supervisor", supervisor)

builder.add_node(
    "research_dispatcher",
    research_dispatcher
)

builder.add_node(
    "research_agent",
    research_agent
)

builder.add_node(
    "analyst_agent",
    analyst_agent
)

builder.add_node(
    "reviewer_agent",
    reviewer_agent
)

builder.add_node(
    "aggregator",
    aggregator
)

builder.add_node(
    "research_for_writer",
    research_for_writer
)

builder.add_node(
    "writer",
    writer_agent
)

builder.add_node(
    "writing_finalizer",
    writing_finalizer
)


# ============================================================
# 11. Edges
# ============================================================

builder.add_edge(
    START,
    "supervisor"
)

# Research Workflow
builder.add_edge(
    "research_agent",
    "aggregator"
)

builder.add_edge(
    "analyst_agent",
    "aggregator"
)

builder.add_edge(
    "reviewer_agent",
    "aggregator"
)

builder.add_edge(
    "aggregator",
    END
)

# Writing Workflow
builder.add_edge(
    "research_for_writer",
    "writer"
)

builder.add_edge(
    "writer",
    "writing_finalizer"
)

builder.add_edge(
    "writing_finalizer",
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
    "task": "研究人工智能在教育领域的应用",
    "results": [],
    "research_result": "",
    "draft": "",
    "final_result": ""
})


print("\n==============================")
print("最终结果")
print("==============================")

print(result["final_result"])
