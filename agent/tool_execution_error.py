import os       #和tool_validation.py基本上一样    体现agent loop 的错误反馈

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool


# ============================================================
# 1. 定义 Tool
# ============================================================

@tool
def divide(a: float, b: float) -> str:
    """计算两个数字相除的结果，b 不能为 0。"""

    if b == 0:
        raise ValueError("除数不能为 0")

    return f"{a} ÷ {b} = {a / b}"


# ============================================================
# 2. 建立工具字典
# ============================================================

tools = {
    "divide": divide
}


# ============================================================
# 3. 创建 LLM
# ============================================================

llm = ChatOpenAI(
    model="deepseek-v4-flash",
    api_key=os.environ["ANTHROPIC_AUTH_TOKEN"],
    base_url="https://api.deepseek.com"
)


# ============================================================
# 4. 把 Tool 绑定给 LLM
# ============================================================

llm_with_tools = llm.bind_tools([
    divide
])


# ============================================================
# 5. 获取用户问题
# ============================================================

question = input("请输入问题：")

messages = [
    HumanMessage(content=question)
]


# ============================================================
# 6. Agent Loop
# ============================================================

while True:

    print("\n========== LLM 思考 ==========")

    response = llm_with_tools.invoke(messages)

    messages.append(response)


    # ========================================================
    # 没有 Tool Call
    # ========================================================

    if not response.tool_calls:

        print("\n========== 最终答案 ==========")
        print(response.content)

        break


    # ========================================================
    # 处理 Tool Call
    # ========================================================

    for tool_call in response.tool_calls:

        tool_name = tool_call["name"]
        arguments = tool_call["args"]
        tool_call_id = tool_call["id"]

        print("\n========== 调用工具 ==========")
        print("工具名称：", tool_name)
        print("工具参数：", arguments)


        # ----------------------------------------------------
        # 根据 Tool 名称找到真正的 Python Tool
        # ----------------------------------------------------

        tool = tools.get(tool_name)


        if tool is None:

            tool_result = "错误：不存在这个工具。"

        else:

            try:

                # Structured Tool 调用
                tool_result = tool.invoke(arguments)

            except Exception as e:

                # 捕获 Tool 执行异常
                tool_result = f"工具执行失败：{e}"


        print("\n========== 工具结果 ==========")
        print(tool_result)


        # ====================================================
        # 把 Tool Result 包装成 ToolMessage
        # ====================================================

        tool_message = ToolMessage(
            content=tool_result,
            tool_call_id=tool_call_id
        )

        messages.append(tool_message)
