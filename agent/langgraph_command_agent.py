from typing import TypedDict      #测试command效果

from langgraph.graph import StateGraph, START, END
from langgraph.types import Command


# ============================================================
# 1. State
# ============================================================

class State(TypedDict):
    question: str
    decision: str
    result: str


# ============================================================
# 2. Agent Node
# ============================================================

def agent_node(state: State):

    print("执行 Agent Node")

    # 模拟 Agent 的判断
    if "天气" in state["question"]:
        return Command(
            update={
                "decision": "weather",
                "result": "需要查询天气"
            },
            goto="weather_node"
        )

    return Command(
        update={
            "decision": "normal",
            "result": "直接回答"
        },
        goto="answer_node"
    )


# ============================================================
# 3. Weather Node
# ============================================================

def weather_node(state: State):

    print("执行 Weather Node")

    return {
        "result": state["result"] + " → Weather Tool"
    }


# ============================================================
# 4. Answer Node
# ============================================================

def answer_node(state: State):

    print("执行 Answer Node")

    return {
        "result": state["result"] + " → Answer"
    }


# ============================================================
# 5. 构建 Graph
# ============================================================

graph_builder = StateGraph(State)

graph_builder.add_node("agent_node", agent_node)
graph_builder.add_node("weather_node", weather_node)
graph_builder.add_node("answer_node", answer_node)

graph_builder.add_edge(START, "agent_node")

graph_builder.add_edge("weather_node", END)
graph_builder.add_edge("answer_node", END)

graph = graph_builder.compile()


# ============================================================
# 6. 测试
# ============================================================

result = graph.invoke({
    "question": "学校有几个学院？",
    "decision": "",
    "result": ""
})

print("\n最终 State：")
print(result)
