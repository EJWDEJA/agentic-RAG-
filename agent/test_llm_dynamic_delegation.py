from typing import TypedDict, Literal               #这里就是为了优化上一版规则型surpervision的不通用不智能性，引入了llm
import os

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END


# ============================================================
# 1. LLM
# ============================================================

llm = ChatOpenAI(
    model="deepseek-v4-flash",
    api_key=os.environ["ANTHROPIC_AUTH_TOKEN"],
    base_url="https://api.deepseek.com"
)


# ============================================================
# 2. AgentTask
#    Supervisor → Agent
# ============================================================

class AgentTask(TypedDict):
    target: Literal["research", "writer"]
    task: str


# ============================================================
# 3. Agent Result Contracts
# ============================================================

class ResearchReport(TypedDict):
    summary: str
    key_points: list[str]


class WritingReport(TypedDict):
    draft: str


# ============================================================
# 4. Research Internal State
# ============================================================

class ResearchState(TypedDict):
    task: str
    summary: str
    key_points: list[str]


# ============================================================
# 5. Writer Internal State
# ============================================================

class WriterState(TypedDict):
    task: str
    draft: str


# ============================================================
# 6. Main State
# ============================================================

class MainState(TypedDict):
    user_task: str
    agent_task: AgentTask

    research_report: ResearchReport
    writing_report: WritingReport


# ============================================================
# 7. Research Agent
# ============================================================

def research_agent(state: ResearchState):

    print("\n[Research Agent]")
    print("收到任务：", state["task"])

    return {
        "summary": f"完成研究：{state['task']}",
        "key_points": [
            "教育领域应用",
            "智能教学",
            "个性化学习"
        ]
    }


# ============================================================
# 8. Research Graph
# ============================================================

research_builder = StateGraph(ResearchState)

research_builder.add_node(
    "research_agent",
    research_agent
)

research_builder.add_edge(
    START,
    "research_agent"
)

research_builder.add_edge(
    "research_agent",
    END
)

research_graph = research_builder.compile()


# ============================================================
# 9. Writer Agent
# ============================================================

def writer_agent(state: WriterState):

    print("\n[Writer Agent]")
    print("收到任务：", state["task"])

    return {
        "draft": f"根据任务「{state['task']}」生成了一份初稿。"
    }


# ============================================================
# 10. Writer Graph
# ============================================================

writer_builder = StateGraph(WriterState)

writer_builder.add_node(
    "writer_agent",
    writer_agent
)

writer_builder.add_edge(
    START,
    "writer_agent"
)

writer_builder.add_edge(
    "writer_agent",
    END
)

writer_graph = writer_builder.compile()


# ============================================================
# 11. Structured LLM
# ============================================================

structured_llm = llm.with_structured_output(
    AgentTask,
    method="json_mode"
)


# ============================================================
# 12. Supervisor
# ============================================================

def supervisor(state: MainState):

    print("\n[Supervisor]")
    print("用户任务：", state["user_task"])

    prompt = f"""
你是一个负责任务委派的 Supervisor。

根据用户任务，决定应该把任务交给哪个 Agent。

可选 Agent：

research
- 负责研究、调查、分析、收集信息

writer
- 负责写作、起草文章、生成文本

请返回 JSON。

JSON 必须包含：

target:
只能是 "research" 或 "writer"

task:
给目标 Agent 的具体任务

用户任务：
{state["user_task"]}
"""

    result = structured_llm.invoke(
        [HumanMessage(content=prompt)]
    )

    delegated_task: AgentTask = {
        "target": result["target"],
        "task": result["task"]
    }

    print("Supervisor 决定：")
    print("target =", delegated_task["target"])
    print("task   =", delegated_task["task"])

    return {
        "agent_task": delegated_task
    }


# ============================================================
# 13. Dispatcher
# ============================================================

def dispatcher(state: MainState):

    target = state["agent_task"]["target"]

    print("\n[Dispatcher]")
    print("目标 Agent：", target)

    if target == "research":
        return "run_research"

    if target == "writer":
        return "run_writer"

    raise ValueError(
        f"未知 Agent: {target}"
    )


# ============================================================
# 14. Run Research
# ============================================================

def run_research(state: MainState):

    print("\n[Supervisor → Research]")

    subgraph_input: ResearchState = {
        "task": state["agent_task"]["task"],
        "summary": "",
        "key_points": []
    }

    result = research_graph.invoke(
        subgraph_input
    )

    research_report: ResearchReport = {
        "summary": result["summary"],
        "key_points": result["key_points"]
    }

    return {
        "research_report": research_report
    }


# ============================================================
# 15. Run Writer
# ============================================================

def run_writer(state: MainState):

    print("\n[Supervisor → Writer]")

    subgraph_input: WriterState = {
        "task": state["agent_task"]["task"],
        "draft": ""
    }

    result = writer_graph.invoke(
        subgraph_input
    )

    writing_report: WritingReport = {
        "draft": result["draft"]
    }

    return {
        "writing_report": writing_report
    }


# ============================================================
# 16. Main Graph
# ============================================================

builder = StateGraph(MainState)

builder.add_node(
    "supervisor",
    supervisor
)

builder.add_node(
    "dispatcher",
    lambda state: {}
)

builder.add_node(
    "run_research",
    run_research
)

builder.add_node(
    "run_writer",
    run_writer
)

builder.add_edge(
    START,
    "supervisor"
)

builder.add_edge(
    "supervisor",
    "dispatcher"
)

builder.add_conditional_edges(
    "dispatcher",
    dispatcher,
    {
        "run_research": "run_research",
        "run_writer": "run_writer"
    }
)

builder.add_edge(
    "run_research",
    END
)

builder.add_edge(
    "run_writer",
    END
)

graph = builder.compile()


# ============================================================
# 17. Test
# ============================================================

print("=" * 60)
print("测试 1：Research")
print("=" * 60)

result = graph.invoke({
    "user_task": "研究人工智能在高校教育中的应用",
    "agent_task": {
        "target": "research",
        "task": ""
    },
    "research_report": {
        "summary": "",
        "key_points": []
    },
    "writing_report": {
        "draft": ""
    }
})

print("\n最终结果：")
print(result)


print("\n" + "=" * 60)
print("测试 2：Writer")
print("=" * 60)

result = graph.invoke({
    "user_task": "写一篇关于人工智能教育应用的文章",
    "agent_task": {
        "target": "writer",
        "task": ""
    },
    "research_report": {
        "summary": "",
        "key_points": []
    },
    "writing_report": {
        "draft": ""
    }
})

print("\n最终结果：")
print(result)