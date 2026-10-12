from typing import Annotated, TypedDict              #测试并发协作汇总阶段使用协调器综合汇总代替aggregator硬拼接

from operator import add
import os

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
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
# 2. Main State
# ============================================================

class MainState(TypedDict):
    user_task: str

    research_result: Annotated[list[str], add]
    analysis_result: Annotated[list[str], add]
    review_result: Annotated[list[str], add]

    final_answer: str


# ============================================================
# 3. Agent State
# ============================================================

class AgentState(TypedDict):
    user_task: str


# ============================================================
# 4. Coordinator Output
# ============================================================

class CoordinatorResult(BaseModel):
    final_answer: str = Field(
        description="综合多个 Agent 结果后形成的最终答案"
    )


structured_llm = llm.with_structured_output(
    CoordinatorResult,
    method="json_mode"
)


# ============================================================
# 5. Supervisor
# ============================================================

def supervisor(state: MainState):

    print("\n===== Supervisor =====")

    return Command(
        goto=[
            Send(
                "research",
                {
                    "user_task": state["user_task"]
                }
            ),
            Send(
                "analyst",
                {
                    "user_task": state["user_task"]
                }
            ),
            Send(
                "reviewer",
                {
                    "user_task": state["user_task"]
                }
            )
        ]
    )


# ============================================================
# 6. Research Agent
# ============================================================

def research(state: AgentState):

    print("\n===== Research Agent =====")

    return {
        "research_result": [
            "人工智能可以用于个性化学习、智能教学和教育资源管理。"
        ]
    }


# ============================================================
# 7. Analyst Agent
# ============================================================

def analyst(state: AgentState):

    print("\n===== Analyst Agent =====")

    return {
        "analysis_result": [
            "人工智能可以提高教学效率，并辅助教师进行学习分析。"
        ]
    }


# ============================================================
# 8. Reviewer Agent
# ============================================================

def reviewer(state: AgentState):

    print("\n===== Reviewer Agent =====")

    return {
        "review_result": [
            "人工智能应用需要关注数据隐私、结果可靠性以及教师监督。"
        ]
    }


# ============================================================
# 9. Coordinator
# ============================================================

def coordinator(state: MainState):

    print("\n===== Coordinator =====")

    prompt = f"""
你是一个 Multi-Agent 系统中的 Coordinator。

用户任务：
{state["user_task"]}

现在有三个 Agent 提供了结果。

Research Agent：
{state["research_result"]}

Analyst Agent：
{state["analysis_result"]}

Reviewer Agent：
{state["review_result"]}

请综合这些结果，形成最终答案。

要求：

1. 不要简单机械拼接三个结果。
2. 合并重复信息。
3. 保留互补信息。
4. 如果不同 Agent 的观点存在冲突，需要根据上下文进行合理处理。
5. Reviewer 提出的限制条件不能被忽略。
6. 最终答案应该直接回答用户任务。

请严格输出 JSON。
"""

    result = structured_llm.invoke(prompt)

    print("Coordinator 已完成综合")

    return {
        "final_answer": result.final_answer
    }


# ============================================================
# 10. Build Graph
# ============================================================

builder = StateGraph(MainState)

builder.add_node("supervisor", supervisor)

builder.add_node("research", research)
builder.add_node("analyst", analyst)
builder.add_node("reviewer", reviewer)

builder.add_node("coordinator", coordinator)


builder.add_edge(START, "supervisor")


# 三个并行 Agent 完成后进入 Coordinator
builder.add_edge("research", "coordinator")
builder.add_edge("analyst", "coordinator")
builder.add_edge("reviewer", "coordinator")


builder.add_edge("coordinator", END)


# ============================================================
# 11. Compile
# ============================================================

graph = builder.compile()


# ============================================================
# 12. Test
# ============================================================

result = graph.invoke({
    "user_task": "人工智能在教育领域的应用",

    "research_result": [],
    "analysis_result": [],
    "review_result": [],

    "final_answer": ""
})


# ============================================================
# 13. Result
# ============================================================

print("\n==============================")
print("最终结果")
print("==============================")

print(result["final_answer"])