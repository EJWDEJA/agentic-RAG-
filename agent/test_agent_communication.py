from typing import TypedDict      #这个代码测试了agent的子图内部代码与输出协议

from langgraph.graph import StateGraph, START, END


# ============================================================
# 1. Main State
# ============================================================

class MainState(TypedDict):
    task: str
    research_report: dict
    draft: str




# ============================================================
# 3. Agent Communication Protocol
# ============================================================

class ResearchReport(TypedDict):     #定义输出协议方便各agent之间的通信，方便后续子图添加新字段时对全局的修改（只需要改生产端输出加逻辑，消费端解析加逻辑即可，不用想赚钱无协议一样修改系统的骨架）
    summary: str
    key_points: list[str]
    sources: list[str]




# ============================================================
#4 Research Subgraph State
# ============================================================

class ResearchSubgraphState(TypedDict):
    task: str
    raw_notes: str
    internal_status: str
    report: ResearchReport




# ============================================================
#5.Research Agent
# ============================================================

def research_agent(state: ResearchSubgraphState):

    print("执行 Research Agent")

    # 这里模拟 Research Agent 内部工作
    raw_notes = state["raw_notes"]

    report: ResearchReport = {
        "summary": f"针对「{state['task']}」的研究结论",
        "key_points": [
            "人工智能可以辅助教学",
            "人工智能可以辅助学习分析",
            "人工智能可以提高教学效率"
        ],
        "sources": [
            "source_A",
            "source_B"
        ]
    }

    print("\nResearch 内部状态：")
    print(state)

    print("\nResearch 对外通信协议：")
    print(report)

    return {
        "report": report
    }





research_builder = StateGraph(ResearchSubgraphState)

research_builder.add_node("research", research_agent)

research_builder.add_edge(START, "research")
research_builder.add_edge("research", END)

research_graph = research_builder.compile()


# ============================================================
# 6. Research Node
# ============================================================

def run_research(state: MainState):

    result = research_graph.invoke({
        "task": state["task"],
        "raw_notes": "Research 内部原始资料",
        "internal_status": "completed",
        "report": {
            "summary": "",
            "key_points": [],
            "sources": []
        }
    })

    # 父图只接收通信协议规定的数据
    report: ResearchReport = result["report"]

    return {
        "research_report": report
    }


# ============================================================
# 7. Writer Agent
# ============================================================

class WriterInput(TypedDict):
    task: str
    research_report: ResearchReport


def writer_agent(state: WriterInput):

    print("\n执行 Writer Agent")

    report = state["research_report"]

    print("\nWriter 收到的通信数据：")
    print(report)

    draft = (
        f"主题：{state['task']}\n"
        f"研究结论：{report['summary']}\n"
        f"关键观点：{report['key_points']}\n"
        f"参考来源：{report['sources']}"
    )

    return {
        "draft": draft
    }


# ============================================================
# 8. Writer Node
# ============================================================

def run_writer(state: MainState):

    writer_input: WriterInput = {
        "task": state["task"],
        "research_report": state["research_report"]
    }

    result = writer_agent(writer_input)

    return {
        "draft": result["draft"]
    }


# ============================================================
# 9. Main Graph
# ============================================================

main_builder = StateGraph(MainState)

main_builder.add_node("research", run_research)
main_builder.add_node("writer", run_writer)

main_builder.add_edge(START, "research")
main_builder.add_edge("research", "writer")
main_builder.add_edge("writer", END)

main_graph = main_builder.compile()


# ============================================================
# 10. Execute
# ============================================================

result = main_graph.invoke({
    "task": "研究人工智能在教育领域的应用",
    "research_report": {
        "summary": "",
        "key_points": [],
        "sources": []
    },
    "draft": ""
})


print("\n==============================")
print("最终 Main State")
print("==============================")

print(result)