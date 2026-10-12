import os      #学习结构化输出个部分函数的作用

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI


# ============================================================
# 1. 定义结构化输出 Schema
# ============================================================

class SupervisorDecision(BaseModel):     #与普通类结构相似只是引入了pydanic搭配with struction output实现参数校验与格式转化 

    next_agent: str = Field(
        description="下一步应该执行的 Agent，只能选择 research 或 writing"     #这部分的field是给字段加约束，注释的，在本项目没什么用因为deepseekAPI不支持直接定义method=json scheme这种输出方式，会被API直接拦截无法进入LLM
    )

    reason: str = Field(
        description="说明为什么选择这个 Agent"
    )


# ============================================================
# 2. 创建 LLM
# ============================================================

llm = ChatOpenAI(
    model="deepseek-v4-flash",
    api_key=os.environ["ANTHROPIC_AUTH_TOKEN"],
    base_url="https://api.deepseek.com"
)


# ============================================================
# 3. 使用 Structured Output
# ============================================================

structured_llm = llm.with_structured_output(    #这一步用于告诉LLM用json mode输出（即json格式输出）并对输出结果进行解析校验格式转化
    SupervisorDecision,
    method="json_mode"
)


# ============================================================
# 4. 单独调用
# ============================================================

result = structured_llm.invoke(       #这一步返回结果不是AIMessage变成了我们自主定义的SurpervisionDecision
    """
    请判断下面这个任务应该交给哪个 Agent。

    任务：
    研究人工智能在教育领域的应用。

    可选择：
    research
    writing

    请按照指定的 JSON Schema 返回结果。     #提示词里面必须出现json才能和json mode搭配通过llmAPI的校验成功进入LLM
    """
)


# ============================================================
# 5. 查看结果
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