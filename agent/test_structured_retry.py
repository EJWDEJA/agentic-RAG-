import os      #测试结构化输出的agent自我纠错机制

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI


# ============================================================
# 1. 定义 Supervisor Decision Schema
# ============================================================

class SupervisorDecision(BaseModel):
    next_agent: str = Field(
        description="下一步执行的 Agent，只能是 research 或 writing"
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
# 3. 创建 Structured LLM
# ============================================================

structured_llm = llm.with_structured_output(
    SupervisorDecision,
    method="json_mode"
)


# ============================================================
# 4. Retry / Repair
# ============================================================

def get_supervisor_decision(task: str):

    max_retries = 3

    prompt = f"""
你是一个 Supervisor。

任务：
{task}

请选择下一步 Agent。

只能选择：
- research
- writing

必须返回 JSON。

JSON 必须包含：
- next_agent
- reason

注意：
第一次请故意返回错误字段名 agent，而不是 next_agent。
这是为了测试程序的错误检测和自动修复机制。
"""

    for attempt in range(1, max_retries + 1):

        print(f"\n========== 第 {attempt} 次尝试 ==========")

        try:

            result = structured_llm.invoke(prompt)

            print("校验成功！")
            print("result =", result)
            print("类型 =", type(result))

            return result

        except Exception as e:

            print("Schema 校验失败！")
            print("错误信息：")
            print(e)

            if attempt == max_retries:    #这里是max_retries的原因是第三次都失败了不可能给你第四次的机会了
                print("\n已经达到最大重试次数，任务失败。")
                raise    #出现把捕获的异常抛出，打印堆栈traceback，如果每一个程序会继续执行返回NONE爆错nonetype但这不是此次报错的根本一样会导致debug不方便

            # ------------------------------------------------
            # 把错误反馈给 LLM，让 LLM 修复
            # ------------------------------------------------

            prompt = f"""
你是一个 Supervisor。

任务：
{task}

你上一次生成的结果没有通过 SupervisorDecision Schema 校验。

错误信息：
{e}

请修复这个错误。

要求：

1. 必须返回合法 JSON
2. 必须使用字段：
   - next_agent
   - reason
3. next_agent 只能是：
   - research
   - writing
4. 不允许使用 agent 字段
5. 不要输出任何 JSON 之外的内容
"""

            print("\n已经生成修复 Prompt，准备重新请求 LLM...")


# ============================================================
# 5. Test
# ============================================================

result = get_supervisor_decision(
    "研究人工智能在教育领域的应用，然后整理成文章"
)


# ============================================================
# 6. 查看最终结果
# ============================================================

print("\n================================")
print("最终结果")
print("================================")

print("result =", result)
print("next_agent =", result.next_agent)
print("reason =", result.reason)
print("type =", type(result))
