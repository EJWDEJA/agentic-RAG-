import os

from langchain_openai import ChatOpenAI

from langchain_core.messages import HumanMessage, ToolMessage    #他们是类和Document本质上一样方便数据调用

# =========================
# 1. 定义工具
# =========================

def get_weather(city):
    return f"{city}今天晴天，25度"


# =========================
# 2. 注册工具
# =========================

tools = {
    "get_weather": get_weather
}


# =========================
# 3. 创建 DeepSeek
# =========================

llm = ChatOpenAI(
    model="deepseek-v4-flash",
    api_key=os.environ["ANTHROPIC_AUTH_TOKEN"],
    base_url="https://api.deepseek.com"
)

llm_with_tools = llm.bind_tools(
    [get_weather]    #告诉lllm他有那些工具
)
# =========================
# 4. 用户问题
# =========================

question = "北京今天天气怎么样？"


messages = [
    HumanMessage(content=question)
]


# =========================
# 5. 第一次请求 DeepSeek
# =========================

response = llm_with_tools.invoke(messages)


print("===== DeepSeek 第一次返回 =====")
print(response.tool_calls)


# =========================
# 6. 获取 Tool Call
# =========================

tool_call = response.tool_calls[0]

tool_name = tool_call["name"]

arguments = tool_call["args"]

tool_call_id = tool_call["id"]


# =========================
# 7. 找到 Python Tool
# =========================

tool = tools.get(tool_name)


if tool is None:

    tool_result = "错误：不存在这个工具。"

else:

    tool_result = tool(**arguments)


print()
print("===== Tool 执行结果 =====")
print(tool_result)


# =========================
# 8. 把 Tool Result 包装成 ToolMessage
# =========================

tool_message = ToolMessage(
    content=tool_result,
    tool_call_id=tool_call_id
)


# =========================
# 9. 把之前的消息 + Tool Result
#    一起交给 DeepSeek
# =========================

messages.append(response)

messages.append(tool_message)


# =========================
# 10. 第二次请求 DeepSeek
# =========================

final_response = llm_with_tools.invoke(messages)


print()
print("===== DeepSeek 最终回答 =====")
print(final_response.content)