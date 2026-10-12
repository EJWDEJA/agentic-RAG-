from typing import Annotated, Literal, TypedDict          #这次将并行----协作-----协调（智能汇总）---迭代融合到了一起，这次supervision不是主脑

                                                           #并行架构下主管supervision是调度器/执行层，coordinator负责决策    
from operator import add
import os

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, Send

MAX_COLLABORATION_ROUNDS =3         #此处是系统安全边界上限
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

    collaboration_round: int

    replan_task: "AgentTask | None"


# ============================================================
# 3. Agent Task
# ============================================================

class AgentTask(BaseModel):                  ###############这是协作里面的契约
    target: Literal[
        "research",
        "analyst",
        "reviewer"
    ]

    task: str


# ============================================================
# 4. Coordinator Decision
# ============================================================
class CoordinatorDecision(BaseModel):              #################这也是协作里面的契约
    action: Literal["continue", "finish"]

    target: Literal[
        "research",
        "analyst",
        "reviewer"
    ] | None = None

    task: str

    final_answer: str | None = None


structured_llm = llm.with_structured_output(
    CoordinatorDecision,
    method="json_mode"
)


# ============================================================
# 5. Supervisor
# ============================================================

def supervisor(state: MainState):

    print("\n===== Supervisor =====")

    # 第一轮：并行启动三个 Agent
    if state["collaboration_round"] == 0:

        print("Supervisor → Research / Analyst / Reviewer")

        return Command(                         #################这部分是第一轮负责触发并行
            goto=[
                Send(
                    "research",
                    {
                        "user_task": state["user_task"],
                        "task": "研究人工智能在教育中的主要应用"
                    }
                ),
                Send(
                    "analyst",
                    {
                        "user_task": state["user_task"],
                        "task": "分析人工智能在教育中的主要价值"
                    }
                ),
                Send(
                    "reviewer",
                    {
                        "user_task": state["user_task"],
                        "task": "分析人工智能教育应用中的风险和限制"
                    }
                )
            ],
            update={
                "collaboration_round": 1
            }
        )

    # 后续轮次：必须存在重新规划任务
    task = state["replan_task"]

    if task is None:
        raise ValueError(
            "Supervisor 收到后续轮次，但 replan_task 为 None"
        )

    print(
        f"Supervisor → {task.target}：{task.task}"
    )

    return Command(
        goto=[
            Send(
                task.target,
                {
                    "user_task": state["user_task"],
                    "task": task.task
                }
            )
        ],
        update={
            "collaboration_round": (
                state["collaboration_round"] + 1
            ),
            "replan_task": None
        }
    )


# ============================================================
# 6. Research        ##接下来的几个并行agent的返回值进行了矛盾化处理测试，冲突协作
# ============================================================

def research(state):

    print("\n===== Research Agent =====")

    return {
        "research_result": [
            f"Research：{state['task']} → "
            "研究表明，人工智能能够显著提高教育教学效率。"
        ]
    }


# ============================================================
# 7. Analyst
# ============================================================

def analyst(state):

    print("\n===== Analyst Agent =====")

    return {
        "analysis_result": [
            f"Analyst：{state['task']} → "
            "人工智能可能提高教学效率，但效果取决于具体应用场景。"
        ]
    }


# ============================================================
# 8. Reviewer
# ============================================================

def reviewer(state):

    print("\n===== Reviewer Agent =====")

    return {
        "review_result": [
            f"Reviewer：{state['task']} → "
            "不能简单认为人工智能一定提高教学效率，需要关注数据质量、教师参与和实际应用条件。"
        ]
    }


# ============================================================
# 9. Coordinator
# ============================================================

def coordinator(state: MainState): ##############这部分负责以前的规则汇总，这部分采用llm动态汇总综合判断，信息充足则action：finish直接command（goto:end）目标为空，如果不充足则action:continue,target:（动态选择一个不充足的点进行补全全局状态）

    

    print("\n===== Coordinator =====")

    if state["collaboration_round"] >= MAX_COLLABORATION_ROUNDS:            #此处是系统安全边界验证
        print("达到最大协作轮次，强制结束")

        return Command(
            goto=END,
            update={
                "final_answer": (
                    "已达到最大协作轮次。"
                    "Coordinator 基于当前已有信息生成最终结果。"
                )
            }
        )
                                        #########这里的提示词承担两个作用，先判断消息是否充足，不充足则以不足点为task进行二次迭代研究，在判断各agent消息是否是否存在冲突，冲突的的话，以冲突点为二次task开始迭代
    prompt = f"""
你是一个 Multi-Agent Coordinator。

用户任务：
{state["user_task"]}

Research：
{state["research_result"]}

Analyst：
{state["analysis_result"]}

Reviewer：
{state["review_result"]}

当前协作轮次：
{state["collaboration_round"]}

判断当前信息是否足够。

如果信息足够：

action = "finish"
target = null
task = 说明为什么可以结束
final_answer = 综合所有 Agent 的信息，生成最终回答。

如果信息不足：

action = "continue"
target 必须选择：
- research
- analyst
- reviewer

task 必须说明该 Agent 还需要完成什么。

如果 action = "finish"，必须提供 final_answer。


如果不同 Agent 对同一个问题存在明显不同的结论、确定程度或证据，
不要直接结束。

应该：
action = "continue"

选择最适合进行验证的 Agent，
并在 task 中明确要求它验证冲突点。

只有在冲突得到合理解释或验证后，才能 finish。



请严格输出 JSON。
"""

    decision = structured_llm.invoke(prompt)

    print(f"Coordinator → {decision}")

    if decision.action == "finish":

        if not decision.final_answer:
            raise ValueError(
                "Coordinator 选择 finish 时必须提供 final_answer"
            )

        return Command(
            goto=END,
            update={
                "final_answer": decision.final_answer
            }
        )

    if decision.target is None:
        raise ValueError(
            "Coordinator 选择 continue 时必须提供 target"
        )

    return Command(
        goto="supervisor",                           ############这部分就是负责迭代的
        update={
            "replan_task": AgentTask(
                target=decision.target,
                task=decision.task
            )
        }
    )













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


# Parallel Agents → Coordinator
builder.add_edge("research", "coordinator")
builder.add_edge("analyst", "coordinator")
builder.add_edge("reviewer", "coordinator")


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

    "final_answer": "",

    "collaboration_round": 0,

    "replan_task": None
})


print("\n==============================")
print("最终结果")
print("==============================")

print(result)