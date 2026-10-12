from typing import TypedDict, Literal        #这次是为了测试参数校验和工作流校验，为了不搞乱暂时1没有用到basemodel的自带的pydantic校验以及发给llm前进行的action space（这部分放在提示词里面）
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
# 4. Internal States
# ============================================================

class ResearchState(TypedDict):
    task: str
    summary: str
    key_points: list[str]


class WriterState(TypedDict):
    task: str
    draft: str


# ============================================================
# 5. Main State
# ============================================================

class MainState(TypedDict):
    user_task: str
    agent_task: AgentTask

    research_report: ResearchReport
    writing_report: WritingReport


# ============================================================
# 6. Research Agent
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
# 7. Research Graph
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
# 8. Writer Agent
# ============================================================

def writer_agent(state: WriterState):

    print("\n[Writer Agent]")
    print("收到任务：", state["task"])

    return {
        "draft": f"根据任务「{state['task']}」生成了一份初稿。"
    }


# ============================================================
# 9. Writer Graph
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
# 10. Structured LLM
# ============================================================

structured_llm = llm.with_structured_output(       #with_structured_output这个的作用是把llm输出的数据结构化成参数1自定义的格式。    
    AgentTask,                                     #当参数1为basemodel的时候因为basemodel来自pydantic所以他运行时自带参数类型参数值校验，此处为typemodel他就无参数校验
    method="json_mode"                             #第二个参数的是告诉llm输出json格式数据
)       


# ============================================================
# 11. Supervisor
# ============================================================

def supervisor(state: MainState):

    print("\n[Supervisor]")
    print("用户任务：", state["user_task"])

    prompt = f"""
你是一个负责任务委派的 Supervisor。

根据用户任务，选择最合适的 Agent。

可选 Agent：

research
负责研究、调查、分析、信息整理。

writer
负责写作、起草文章、生成文本。

请返回 JSON。

必须包含：

target:
只能是 "research" 或 "writer"

task:
必须是一个明确、具体、可执行的任务。
不能为空。

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

    print("\nSupervisor 生成委派任务：")
    print(delegated_task)

    return {
        "agent_task": delegated_task
    }


# ============================================================
# 12. Delegation Validation
# ============================================================

def validate_delegation(
    agent_task: AgentTask
) -> bool:

    # --------------------------------------------------------
    # 1. target 必须存在
    # --------------------------------------------------------

    if agent_task["target"] not in {
        "research",
        "writer"
    }:
        return False

    # --------------------------------------------------------
    # 2. task 不能为空
    # --------------------------------------------------------

    if not agent_task["task"].strip():
        return False

    # --------------------------------------------------------
    # 3. task 至少需要具有一定信息量
    # --------------------------------------------------------

    if len(agent_task["task"].strip()) < 5:
        return False

    return True


# ============================================================
# 13. Delegation Repair
# ============================================================

def repair_delegation(
    state: MainState
):

    print("\n[Delegation Repair]")

    print("原始委派：")
    print(state["agent_task"])

    prompt = f"""
你是一个任务委派修复器。

下面的 AgentTask 无法通过任务有效性检查。

用户原始任务：
{state["user_task"]}

原始 AgentTask：
target = {state["agent_task"]["target"]}
task = {state["agent_task"]["task"]}

请修复这个 AgentTask。

要求：

1. target 只能是 research 或 writer
2. task 必须具体
3. task 不能为空
4. task 必须能够直接交给目标 Agent 执行

返回 JSON：

{{
    "target": "research 或 writer",
    "task": "具体任务"
}}
"""

    result = structured_llm.invoke(
        [HumanMessage(content=prompt)]
    )

    repaired_task: AgentTask = {
        "target": result["target"],
        "task": result["task"]
    }

    print("\n修复后的委派：")
    print(repaired_task)

    return {
        "agent_task": repaired_task
    }


# ============================================================
# 14. Dispatcher
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
# 15. Run Research
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
# 16. Run Writer
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
# 17. Validation Router
# ============================================================

def validation_router(state: MainState):

    if validate_delegation(
        state["agent_task"]
    ):
        print("\n[Validation] 委派任务合法")
        return "dispatcher"

    print("\n[Validation] 委派任务不合法")
    return "repair"


# ============================================================
# 18. Main Graph
# ============================================================

builder = StateGraph(MainState)

builder.add_node(
    "supervisor",
    supervisor
)

builder.add_node(
    "validation",
    lambda state: {}
)

builder.add_node(
    "repair",
    repair_delegation
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
    "validation"
)

builder.add_conditional_edges(
    "validation",
    validation_router,
    {
        "dispatcher": "dispatcher",
        "repair": "repair"
    }
)

builder.add_edge(
    "repair",
    "validation"
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
# 19. Test
# ============================================================

print("=" * 60)
print("测试：Dynamic Delegation + Validation + Repair")
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