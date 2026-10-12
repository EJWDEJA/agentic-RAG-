from typing import TypedDict     #单测试agent的中断---人工确认---恢复机制

from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import InMemorySaver


# ============================================================
# 1. State
# ============================================================

class State(TypedDict):
    message: str
    result: str


# ============================================================
# 2. Node
# ============================================================

def approval_node(state: State):

    print("进入 Approval Node")

    # 暂停 Graph，等待人工输入
    answer = interrupt("是否允许继续？")

    print("收到人工决定：", answer)

    if answer == "fff":
        return {
            "result": "已批准"
        }

    return {
        "result": "已拒绝"
    }


# ============================================================
# 3. Graph
# ============================================================

graph_builder = StateGraph(State)

graph_builder.add_node("approval_node", approval_node)

graph_builder.add_edge(START, "approval_node")
graph_builder.add_edge("approval_node", END)

checkpointer = InMemorySaver()

graph = graph_builder.compile(
    checkpointer=checkpointer
)


# ============================================================
# 4. 第一次执行
# ============================================================

config = {
    "configurable": {
        "thread_id": "user001"
    }
}

result = graph.invoke(
    {
        "message": "执行操作",
        "result": ""
    },
    config
)


print("\n第一次执行结果：")
print(result)


result = graph.invoke(
    Command(resume="ggg"),
    config
)

print("\n恢复后的结果：")
print(result)

