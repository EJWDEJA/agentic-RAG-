from typing import Annotated, TypedDict        #测试chain式工作流和并发工作流的混合编排，由supervisor（规则式路由if   else）选择工作流


from langgraph.graph import StateGraph, START, END
from langgraph.types import Send, Command


# ============================================================
# 1. Global State
# ============================================================

class MainState(TypedDict):
    task: str
    task_type: str
    results: Annotated[list[str], lambda old, new: old + new]
    research_result: str
    draft: str
    final_result: str


# ============================================================
# 2. Supervisor
# ============================================================

def supervisor(state: MainState):

    print("执行 Supervisor")

    # 如果任务包含“研究”，进入 Parallel Research Workflow
    if "研究" in state["task"]:
        return Command(
            update={
                "task_type": "research"
            },
            goto="research_dispatcher"
        )

    # 否则进入 Writing Workflow
    return Command(
        update={
            "task_type": "writing"
        },
        goto="research_for_writer"
    )


# ============================================================
# 3. Parallel Research Agents
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
# 4. Parallel Dispatcher
# ============================================================

def research_dispatcher(state):

    print("执行 Parallel Dispatcher")

    return [
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
  



# ============================================================
# 5. Aggregator
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
# 6. Writing Workflow
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


# ============================================================
# 7. Writing Finalizer
# ============================================================

def writing_finalizer(state):

    print("执行 Writing Finalizer")

    return {
        "final_result": state["draft"]
    }


# ============================================================
# 8. Build Graph
# ============================================================

builder = StateGraph(MainState)

# Supervisor
builder.add_node("supervisor", supervisor)

# Parallel workflow
builder.add_node("research_dispatcher", research_dispatcher)
builder.add_node("research_agent", research_agent)
builder.add_node("analyst_agent", analyst_agent)
builder.add_node("reviewer_agent", reviewer_agent)
builder.add_node("aggregator", aggregator)

# Writing workflow
builder.add_node("research_for_writer", research_for_writer)
builder.add_node("writer", writer_agent)
builder.add_node("writing_finalizer", writing_finalizer)


# ============================================================
# 9. Main Routing
# ============================================================

builder.add_edge(START, "supervisor")


# ============================================================
# 10. Parallel Workflow
# ============================================================

builder.add_edge("research_agent", "aggregator")
builder.add_edge("analyst_agent", "aggregator")
builder.add_edge("reviewer_agent", "aggregator")

builder.add_edge("aggregator", END)

# ============================================================
# 11. Writing Workflow
# ============================================================

builder.add_edge("research_for_writer", "writer")
builder.add_edge("writer", "writing_finalizer")
builder.add_edge("writing_finalizer", END)


# ============================================================
# 12. Compile
# ============================================================

graph = builder.compile()


# ============================================================
# 13. Test
# ============================================================

result = graph.invoke({
    "task": "人工智能在教育领域的应用",
    "task_type": "",
    "results": [],
    "research_result": "",
    "draft": "",
    "final_result": ""
})


print("\n==============================")
print("最终结果")
print("==============================")

print(result["final_result"])
