from typing import TypedDict     #测试全局状态除了保存messages还需要考虑保存的其他消息

from langgraph.graph import StateGraph, START, END


# ============================================================
# 1. 定义 State
# ============================================================

class State(TypedDict):
    user_name: str
    city: str
    weather: str


# ============================================================
# 2. Node A：获取用户信息
# ============================================================

def node_a(state: State):      #state是自定义参数langgraph是根据位置传参的不是根据名字这里重要的只有位置和类型

    print("\n========== Node A ==========")

    print("当前 user_name：", state["user_name"])
    print("当前 city：", state["city"])
    print("当前 weather：", state["weather"])

    return {
        "user_name": "小明",
        "city": "北京"
    }


# ============================================================
# 3. Node B：根据 State 获取天气
# ============================================================

def node_b(state: State):         #各节点默认读取全局信息，但只是返回局部消息进行局部修改

    print("\n========== Node B ==========")

    print("读取到 user_name：", state["user_name"])
    print("读取到 city：", state["city"])

    weather = "晴天，25°C"

    print("生成 weather：", weather)

    return {
        "weather": weather
    }


# ============================================================
# 4. Node C：读取完整 State
# ============================================================

def node_c(state: State):

    print("\n========== Node C ==========")

    print("user_name：", state["user_name"])
    print("city：", state["city"])
    print("weather：", state["weather"])

    return {}   #返回空字典就是不修改


# ============================================================
# 5. 创建 Graph
# ============================================================

graph_builder = StateGraph(State)


# ============================================================
# 6. 添加 Node
# ============================================================

graph_builder.add_node(
    "node_a",
    node_a
)

graph_builder.add_node(
    "node_b",
    node_b
)

graph_builder.add_node(
    "node_c",
    node_c
)


# ============================================================
# 7. 定义 Edge
# ============================================================

graph_builder.add_edge(
    START,
    "node_a"
)

graph_builder.add_edge(
    "node_a",
    "node_b"
)

graph_builder.add_edge(
    "node_b",
    "node_c"
)

graph_builder.add_edge(
    "node_c",
    END
)


# ============================================================
# 8. 编译
# ============================================================

graph = graph_builder.compile()


# ============================================================
# 9. 初始 State
# ============================================================

initial_state = {     #仅仅负责提供一个输入字典，并不是全局变量名
    "user_name": "",
    "city": "",
    "weather": ""
}


# ============================================================
# 10. 执行 Graph
# ============================================================

result = graph.invoke(initial_state)


# ============================================================
# 11. 查看最终 State
# ============================================================

print("\n========================================")
print("最终 State")
print("========================================")

print(result)
