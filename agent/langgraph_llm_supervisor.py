import os                              #测试chain式工作流和并发工作流的混合编排，由supervisor（LLM选择，通过语言提示词限制llm行为）选择工作流

from typing import Annotated, TypedDict

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
# 2. Global State
# ============================================================

class MainState(TypedDict):
    task: str
    results: Annotated[list[str], lambda old, new: old + new]
    research_result: str
    draft: str
    final_result: str


# ============================================================
# 3. Supervisor
# ============================================================

def supervisor(state: MainState):

    print("执行 Supervisor")

    prompt = f"""
你是一个 Multi-Agent Supervisor。

现在有两个工作流：

1. research
   - 适合研究、调查、分析资料类任务

2. writing
   - 适合写作、生成文章、撰写内容类任务

用户任务：
{state["task"]}

请只回答：
research
或
writing
"""

    response = llm.invoke(prompt)

    decision = response.content.strip().lower()

    print(f"Supervisor 决策：{decision}")

    if "research" in decision:

        return Command(
            goto="research_dispatcher"
        )

    return Command(
        goto="research_for_writer"
    )


# ============================================================
# 4. Research Agents
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
# 5. Research Dispatcher
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
# 6. Research Aggregator
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
# 7. Writing Workflow
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
# 8. Build Graph
# ============================================================

builder = StateGraph(MainState)

builder.add_node("supervisor", supervisor)

builder.add_node("research_dispatcher", research_dispatcher)
builder.add_node("research_agent", research_agent)
builder.add_node("analyst_agent", analyst_agent)
builder.add_node("reviewer_agent", reviewer_agent)
builder.add_node("aggregator", aggregator)

builder.add_node("research_for_writer", research_for_writer)
builder.add_node("writer", writer_agent)
builder.add_node("writing_finalizer", writing_finalizer)


# ============================================================
# 9. Edges
# ============================================================

builder.add_edge(START, "supervisor")

# Research Workflow

builder.add_edge("research_agent", "aggregator")
builder.add_edge("analyst_agent", "aggregator")
builder.add_edge("reviewer_agent", "aggregator")

builder.add_edge("aggregator", END)

# Writing Workflow
builder.add_edge("research_for_writer", "writer")
builder.add_edge("writer", "writing_finalizer")
builder.add_edge("writing_finalizer", END)


# ============================================================
# 10. Compile
# ============================================================

graph = builder.compile()


# ============================================================
# 11. Test
# ============================================================

result = graph.invoke({
    "task": "写一篇人工智能在教育领域应用的介绍文章",
    "results": [],
    "research_result": "",
    "draft": "",
    "final_result": ""
})


print("\n==============================")
print("最终结果")
print("==============================")

print(result["final_result"])
