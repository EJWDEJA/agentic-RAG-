from typing import Annotated, TypedDict      #测试send类组成的列表通过condition edge触发的多任务并行机制，多个任务一个agent解决

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from langgraph.graph.message import add_messages


# ============================================================
# 1. Global State
# ============================================================

class State(TypedDict):
    tasks: list[str]
    results: Annotated[list[str], lambda old, new: old + new]    #lambda不是通过参数名赋值的是通过位置赋值的第一个位置的全局状态的当前字段第二个是当前字段的新状态


# ============================================================
# 2. Dispatcher
# ============================================================

def dispatcher(state: State):

    print("执行 Dispatcher")

    return [
        Send("worker", {"task": task})
        for task in state["tasks"]    #统一返回
    ]


# ============================================================
# 3. Worker
# ============================================================

def worker(state):

    print(f"执行 Worker：{state['task']}")

    return {
        "results": [
            f"{state['task']} 完成"
        ]     #一个个更新全局状态
    }


# ============================================================
# 4. Graph
# ============================================================

builder = StateGraph(State)

builder.add_node("worker", worker)

builder.add_conditional_edges(       #条件边加返回的send列表触发并行任务
    START,
    dispatcher
)

builder.add_edge("worker", END)

graph = builder.compile()


# ============================================================
# 5. Test
# ============================================================

result = graph.invoke({
    "tasks": [
        "研究人工智能",
        "研究机器人",
        "研究自动驾驶"
    ],
    "results": []
})

print("\n最终 State：")
print(result)