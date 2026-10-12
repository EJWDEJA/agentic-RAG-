from typing import Annotated, TypedDict        #一个任务多个不用职责agent解决

from langgraph.graph import StateGraph, START, END               #这个版本的supervision不不被加入节点，，后续用条件边作为第二个参数路由函数（内含send）
from langgraph.types import Send


# ============================================================
# 1. Global State
# ============================================================

class MainState(TypedDict):
    task: str
    results: Annotated[list[str], lambda old, new: old + new]
    final_result: str


# ============================================================
# 2. Research Agent
# ============================================================

def research_agent(state):
    print("执行 Research Agent")

    return {
        "results": [
            f"Research：完成「{state['task']}」的资料研究"
        ]
    }


# ============================================================
# 3. Analyst Agent
# ============================================================

def analyst_agent(state):
    print("执行 Analyst Agent")

    return {
        "results": [
            f"Analyst：完成「{state['task']}」的分析"
        ]
    }


# ============================================================
# 4. Reviewer Agent
# ============================================================

def reviewer_agent(state):
    print("执行 Reviewer Agent")

    return {
        "results": [
            f"Reviewer：完成「{state['task']}」的审查"
        ]
    }


# ============================================================
# 5. Supervisor
# ============================================================

def supervisor(state):

    print("执行 Supervisor")

    return [
        Send("research", {
            "task": state["task"]
        }),
        Send("analyst", {
            "task": state["task"]
        }),
        Send("reviewer", {
            "task": state["task"]
        })
    ]


# ============================================================
# 6. Aggregator Agent
# ============================================================

def aggregator(state):

    print("执行 Aggregator Agent")

    return {
        "final_result": (
            "汇总结果：\n"
            + "\n".join(state["results"])
        )
    }


# ============================================================
# 7. Build Graph
# ============================================================

builder = StateGraph(MainState)



builder.add_node("research", research_agent)
builder.add_node("analyst", analyst_agent)
builder.add_node("reviewer", reviewer_agent)

builder.add_node("aggregator", aggregator)

# START → Supervisor
builder.add_conditional_edges(
    START,
    supervisor      #用send的第一个参数选择节点第二个参数用于状态隔离，分发任务
)


# 三个并行 Agent → Aggregator
builder.add_edge("research", "aggregator")
builder.add_edge("analyst", "aggregator")
builder.add_edge("reviewer", "aggregator")

# Aggregator → END
builder.add_edge("aggregator", END)

graph = builder.compile()


# ============================================================
# 8. Test
# ============================================================

result = graph.invoke({
    "task": "写一篇人工智能在教育领域应用的介绍文章",
    "results": [],
    "final_result": ""
})


print("\n==============================")
print("最终结果")
print("==============================")

print(result["final_result"])
