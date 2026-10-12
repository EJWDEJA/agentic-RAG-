import os

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool


# =========================
# 1. 定义结构化工具
# =========================

@tool    #读取，提取，转化格式（json），封装成类
def get_weather(city: str) -> str:    #-> str是python内置注解注释函数返回值的数据类型。city: str是标准的参数类型注解
    """查询指定城市的天气。"""   #他是docstring注解负责介绍函数是干什么的
    return f"{city}今天晴天，25度"


@tool
def calculate(a: int, b: int) -> str:
    """计算两个整数的和。"""
    return f"计算结果是：{a + b}"


# =========================
# 2. 创建 LLM
# =========================

llm = ChatOpenAI(
    model="deepseek-v4-flash",
    api_key=os.environ["ANTHROPIC_AUTH_TOKEN"],
    base_url="https://api.deepseek.com"
)


# =========================
# 3. 把结构化工具绑定给 LLM
# =========================

llm_with_tools = llm.bind_tools([    #挂载封装好的对象
    get_weather,
    calculate
])


# =========================
# 4. 测试用户问题
# =========================

question = input("请输入问题：")


# =========================
# 5. 调用 LLM
# =========================

response = llm_with_tools.invoke(question)


# =========================
# 6. 查看 LLM 返回结果
# =========================

print("\n========== LLM 原始返回 ==========")
print(response)


# =========================
# 7. 查看文本答案
# =========================

print("\n========== content ==========")
print(response.content)


# =========================
# 8. 查看 Tool Calls
# =========================

print("\n========== tool_calls ==========")
print(response.tool_calls)
