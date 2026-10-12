import os                         ##这个文件是用来测试检索---评估（flash）---重写---再检索---评估（true）---llm汇总---输出。合在一起的整体测试（依旧不完整）

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
# 2. Retrieval
# ============================================================

def retrieve(query: str) -> str:

    print("\n===== Retrieval =====")
    print("Query:", query)

    # 第一次检索：故意不完整
    if query == "人工智能在教育领域的应用":
        return (
            "人工智能可以应用于个性化学习和智能教学。"
        )

    # 第二次检索：提供完整信息
    return (
        "人工智能在教育领域可以应用于："
        "个性化学习、智能教学、学习分析、"
        "教育资源管理。"
    )


# ============================================================
# 3. Retrieval Evaluation
# ============================================================

class RetrievalDecision(BaseModel):

    sufficient: bool = Field(
        description="当前检索结果是否足以回答用户问题"
    )

    reason: str = Field(
        description="判断原因"
    )


evaluator = llm.with_structured_output(
    RetrievalDecision,
    method="json_mode"
)


# ============================================================
# 4. Query Rewriter
# ============================================================

class QueryRewrite(BaseModel):

    query: str = Field(
        description="下一次检索应该使用的查询"
    )

    reason: str = Field(
        description="为什么需要这样修改查询"
    )


rewriter = llm.with_structured_output(
    QueryRewrite,
    method="json_mode"
)


# ============================================================
# 5. User Question
# ============================================================

question = "人工智能在教育领域有哪些应用？"


# ============================================================
# 6. Agentic RAG Loop
# ============================================================

current_query = question
retrieval_result = ""

while True:

    # --------------------------------------------------------
    # Step 1：Retrieval
    # --------------------------------------------------------

    retrieval_result = retrieve(current_query)

    print("\n检索结果：")
    print(retrieval_result)

    # --------------------------------------------------------
    # Step 2：Evaluate
    # --------------------------------------------------------

    evaluation_prompt = f"""
你是 Retrieval Evaluator。

用户问题：
{question}

当前查询：
{current_query}

当前检索结果：
{retrieval_result}

判断当前检索结果是否已经足以回答用户问题。

如果足够：
sufficient = true

如果不够：
sufficient = false

reason 必须说明原因。

请严格输出 JSON。
"""

    decision = evaluator.invoke(
        evaluation_prompt
    )

    print("\n===== Evaluator =====")
    print(decision)

    # --------------------------------------------------------
    # Step 3：结果足够
    # --------------------------------------------------------

    if decision.sufficient:

        print("\n===== 最终回答阶段 =====")

        answer_prompt = f"""
根据下面的检索结果回答用户问题。

用户问题：
{question}

检索结果：
{retrieval_result}

请直接回答用户问题。
"""

        answer = llm.invoke(answer_prompt)

        print("\n==============================")
        print("最终答案")
        print("==============================")
        print(answer.content)

        break

    # --------------------------------------------------------
    # Step 4：结果不足 → Query Rewrite
    # --------------------------------------------------------

    rewrite_prompt = f"""
你是 Query Rewriter。

用户问题：
{question}

当前查询：
{current_query}

当前检索结果：
{retrieval_result}

Evaluator 判断：
{decision.reason}

当前检索结果不足。

请设计一个新的、更具体的查询，
用于下一次 Retrieval。

不要重复当前查询。

请严格输出 JSON。
"""

    rewritten = rewriter.invoke(
        rewrite_prompt
    )

    print("\n===== Query Rewrite =====")
    print(rewritten)

    current_query = rewritten.query