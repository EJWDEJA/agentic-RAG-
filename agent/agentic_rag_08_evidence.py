import os                    #创建一个evident的类方便后续证据感知评估

# 必须在导入 transformers / sentence_transformers 之前设置！
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import sys

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from typing import TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

from retrieval_chain_group import retrieve_context


# ============================================================
# 1. LLM
# ============================================================

llm = ChatOpenAI(
    model="deepseek-v4-flash",
    api_key=os.environ["ANTHROPIC_AUTH_TOKEN"],
    base_url="https://api.deepseek.com"
)


# ============================================================
# 2. Evidence Contract
# ============================================================

class Evidence(TypedDict):
    content: str
    source: str
    sheet: str
    chunk_id: int


class RetrievalResult(TypedDict):
    context: str
    evidence: list[Evidence]


# ============================================================
# 3. Real MyRAG Retrieval Tool
# ============================================================

@tool
def retrieve_knowledge(query: str) -> RetrievalResult:
    """
    使用 MyRAG 完整检索链检索知识库，
    同时返回 Context 和结构化 Evidence。
    """

    print("\n===== MyRAG Retrieval =====")
    print("Query:")
    print(query)

    result = retrieve_context(query)

    context = result["context"]
    final_docs = result["final_docs"]

    # --------------------------------------------------------
    # Evidence Adapter
    # --------------------------------------------------------

    evidence = []

    for doc in final_docs:

        metadata = doc.metadata

        evidence.append(
            {
                "content": doc.page_content,
                "source": metadata.get("source", ""),
                "sheet": metadata.get("sheet", ""),
                "chunk_id": metadata.get("chunk_id", -1),
            }
        )

    retrieval_result: RetrievalResult = {
        "context": context,
        "evidence": evidence,
    }

    # --------------------------------------------------------
    # Debug
    # --------------------------------------------------------

    print("\n===== Retrieved Context =====")
    print(context)

    print("\n===== Evidence =====")

    for item in evidence:

        print(
            f"source={item['source']}, "
            f"sheet={item['sheet']}, "
            f"chunk_id={item['chunk_id']}"
        )

    return retrieval_result


# ============================================================
# 4. Agent
# ============================================================

agent = llm.bind_tools(
    [retrieve_knowledge]
)


# ============================================================
# 5. User Question
# ============================================================

question = "赵雪丽是谁"


messages = [
    {
        "role": "system",
        "content": """
你是一个 Agentic RAG Agent。

你的任务是使用知识库回答用户问题。

如果缺少知识库信息，就调用 retrieve_knowledge。

retrieve_knowledge 返回两个部分：

1. context
   - 用于阅读和理解知识内容。

2. evidence
   - 用于了解这些内容来自哪个 source、
     sheet 和 chunk。

现在先正常使用 Retrieval Tool。
不要自己模拟知识库。
"""
    },
    {
        "role": "user",
        "content": question
    }
]


# ============================================================
# 6. Agent Loop
# ============================================================

while True:

    response = agent.invoke(messages)

    print("\n===== Agent =====")

    if response.tool_calls:

        print("Agent → Retrieval")

        for tool_call in response.tool_calls:

            print(
                "Query:",
                tool_call["args"]["query"]
            )

    else:

        print("Agent → Final Answer")

    # --------------------------------------------------------
    # Final Answer
    # --------------------------------------------------------

    if not response.tool_calls:

        print("\n==============================")
        print("最终答案")
        print("==============================")

        print(response.content)

        break

    # --------------------------------------------------------
    # 保存 Tool Call
    # --------------------------------------------------------

    messages.append(response)

    # --------------------------------------------------------
    # 执行 Retrieval
    # --------------------------------------------------------

    for tool_call in response.tool_calls:

        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_call_id = tool_call["id"]

        result = retrieve_knowledge.invoke(
            tool_args
        )

        print("\n===== Tool Result Type =====")
        print(type(result))

        print("\n===== Tool Result =====")
        print(result)

        messages.append(
            {
                "role": "tool",
                "content": str(result),
                "tool_call_id": tool_call_id
            }
        )