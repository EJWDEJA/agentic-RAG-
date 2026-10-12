import os                      #这一步单独测试用契约（方便程序判断）的方式由llm评估是否继续，评估的布置就是用契约约束llm输出结构化对象方便后续出现的判断

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI


# ============================================================
# 1. LLM
# ============================================================

llm = ChatOpenAI(
    model="deepseek-v4-flash",
    api_key=os.environ["ANTHROPIC_AUTH_TOKEN"],
    base_url="https://api.deepseek.com"
)


# ============================================================
# 2. Retrieval Result
# ============================================================

retrieval_result = """
人工智能可以应用于个性化学习和智能教学。
"""


# ============================================================
# 3. Retrieval Evaluation Schema
# ============================================================

class RetrievalDecision(BaseModel):

    sufficient: bool = Field(
        description="当前检索结果是否足以回答用户问题"
    )

    reason: str = Field(
        description="判断检索结果是否足够的原因"
    )


# ============================================================
# 4. Structured Evaluator
# ============================================================

evaluator = llm.with_structured_output(
    RetrievalDecision,
    method="json_mode"
)


# ============================================================
# 5. Evaluation Prompt
# ============================================================

question = "人工智能在教育领域有哪些应用？"

prompt = f"""
你是一个 Retrieval Evaluator。

你的任务不是回答用户问题，
而是判断当前检索结果是否足以回答用户问题。

用户问题：
{question}

当前检索结果：
{retrieval_result}

判断标准：

1. 如果检索结果已经覆盖回答问题所需要的主要信息：
   sufficient = true

2. 如果检索结果缺少重要信息：
   sufficient = false

reason 必须说明为什么。

请严格输出 JSON。
"""


# ============================================================
# 6. Evaluate
# ============================================================

decision = evaluator.invoke(prompt)


# ============================================================
# 7. Result
# ============================================================

print("\n==============================")
print("Retrieval Evaluation")
print("==============================")

print("用户问题：")
print(question)

print("\n检索结果：")
print(retrieval_result)

print("\nEvaluator 决策：")
print(decision)

print("\nsufficient =", decision.sufficient)
print("reason =", decision.reason)