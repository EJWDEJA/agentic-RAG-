import os

from pydantic import BaseModel
from langchain_openai import ChatOpenAI


# ============================================================
# 1. 定义 Supervisor 的结构化决策
# ============================================================

class SupervisorDecision(BaseModel):
    next_agent: str
    reason: str


# ============================================================
# 2. 创建 LLM
# ============================================================

llm = ChatOpenAI(
    model="deepseek-v4-flash",
    api_key=os.environ["ANTHROPIC_AUTH_TOKEN"],
    base_url="https://api.deepseek.com"
)


# ============================================================
# 3. 给 LLM 加上 Structured Output
# ============================================================

structured_llm = llm.with_structured_output(
    SupervisorDecision,
    method="json_mode"
)


# ============================================================
# 4. 单独调用 Structured LLM
# ============================================================

result = structured_llm.invoke(
    "请以 JSON 格式进行结构化输出。"
    "请判断这个任务应该交给哪个 Agent："
    "研究人工智能在教育领域的应用。"
    "如果需要研究，请选择 research；"
    "如果需要写作，请选择 writing。"
)



# ============================================================
# 5. 观察真实返回结果
# ============================================================

print("返回结果：")
print(result)

print("\n返回类型：")
print(type(result))

print("\nnext_agent：")
print(result.next_agent)

print("\nreason：")
print(result.reason)

print("\n是否为 SupervisorDecision：")
print(isinstance(result, SupervisorDecision))
