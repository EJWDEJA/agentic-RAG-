from typing import TypedDict

from langgraph.graph import StateGraph, START, END


# ============================================================
# 1. 定义 State
# ============================================================

class State(TypedDict):    #定义一个叫State(的类可自定义)   类似于Document
    message: str       #类里面字段名和字段内数据类型


# ============================================================
# 2. 定义节点
# ============================================================

def node_a(state: State):     #括号里面是输入参数以及他的类型
    print("执行 Node A")

    return {
        "message": state["message"] + " → A"     
    }


def node_b(state: State):
    print("执行 Node B")

    return {
        "message": state["message"] + " → B"
    }


# ============================================================
# 3. 创建 Graph
# ============================================================

graph_builder = StateGraph(State)     #创建一个图创造器


# ============================================================
# 4. 添加节点
# ============================================================

graph_builder.add_node("node_a", node_a)
graph_builder.add_node("node_b", node_b)


# ============================================================
# 5. 定义执行边
# ============================================================

graph_builder.add_edge(START, "node_a")
graph_builder.add_edge("node_a", "node_b")
graph_builder.add_edge("node_b", END)


# ============================================================
# 6. 编译 Graph
# ============================================================

graph = graph_builder.compile()


# ============================================================
# 7. 执行
# ============================================================

result = graph.invoke({
    "message": "开始"
})


print("\n最终 State：")
print(result)