import os                         #这部分是给evidence进行编号，方便直接输出支撑答案的证据来自哪里              证据来源引用

# 必须在导入 transformers / sentence_transformers 之前设置
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

from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage

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
# 2. Evidence
# ============================================================

class Evidence(TypedDict):
    id: str
    source: str
    sheet: str
    chunk_id: int


class RetrievalResult(TypedDict):
    context: str
    evidence: list[Evidence]


# ============================================================
# 3. Retrieval Tool
# ============================================================

@tool
def retrieve_knowledge(query: str) -> RetrievalResult:
    """
    从学校知识库检索信息。
    返回 Context 和带有 Evidence ID 的证据。
    """

    print("\n" + "=" * 70)
    print("Retrieval Tool")
    print(f"Query: {query}")
    print("=" * 70)

    result = retrieve_context(query)

    context = result["context"]
    final_docs = result["final_docs"]

    evidence = []

    for index, doc in enumerate(final_docs):

        metadata = doc.metadata

        evidence.append({                  ###这部分是单次检索的使用证据进行编号，后面虽然又会有去重编号但是这里的编号并不多余主要是给evidence列表里面的对象加一个字段“id”方便后续去重编号时的id修改
            "id": f"E{index + 1}",
            "source": metadata.get("source", ""),
            "sheet": metadata.get("sheet", ""),
            "chunk_id": metadata.get("chunk_id", -1)
        })

    print("\n----- Context -----")
    print(context)

    print("\n----- Evidence -----")

    for item in evidence:
        print(item)

    return {
        "context": context,
        "evidence": evidence
    }


# ============================================================
# 4. Grounded Answer
# ============================================================

class GroundedAnswer(BaseModel):

    answer: str = Field(
        description="最终回答"
    )

    evidence_ids: list[str] = Field(
        description=(
            "支持最终答案的 Evidence ID。"
            "例如 ['E1', 'E3']"
        )
    )


# ============================================================
# 5. Grounding LLM
# ============================================================

grounding_llm = llm.with_structured_output(
    GroundedAnswer,
    method="json_mode"
)


# ============================================================
# 6. Agent
# ============================================================

agent = llm.bind_tools(
    [retrieve_knowledge]
)

tool_map = {
    "retrieve_knowledge": retrieve_knowledge
}


# ============================================================
# 7. Question
# ============================================================

question = "赵雪丽是谁"


# ============================================================
# 8. Search Budget
# ============================================================

MAX_RETRIEVALS = 3

retrieval_count = 0


# ============================================================
# 9. Accumulated Evidence
# ============================================================

all_context = []

all_evidence = []


# ============================================================
# 10. Messages
# ============================================================

messages = [
    {
        "role": "system",
        "content": """
你是一个学校知识库 Agent。

你的任务是回答用户问题。

规则：

1. 信息不足时调用 retrieve_knowledge。
2. 检索结果必须作为事实依据。
3. 不允许编造知识库中不存在的信息。
4. 如果证据不足，可以继续检索。
5. 不要重复完全相同的 Query。
6. 如果系统告诉你检索预算耗尽，禁止继续调用工具。
7. 最终回答必须建立在检索得到的证据之上。
"""
    },
    {
        "role": "user",
        "content": question
    }
]


# ============================================================
# 11. Retrieval Loop
# ============================================================

while True:

    print("\n" + "#" * 70)
    print("Agent Loop")
    print(
        f"Retrieval Count: "
        f"{retrieval_count}/{MAX_RETRIEVALS}"
    )
    print("#" * 70)

    response = agent.invoke(messages)

    # --------------------------------------------------------
    # Agent 不再调用工具
    # --------------------------------------------------------

    if not response.tool_calls:
        break

    messages.append(response)

    # --------------------------------------------------------
    # Search Budget
    # --------------------------------------------------------

    if retrieval_count >= MAX_RETRIEVALS:

        print("\n" + "!" * 70)
        print("Search Budget 已耗尽")
        print("!" * 70)

        messages.append({
            "role": "system",
            "content": """
检索预算已经耗尽。

禁止继续调用 retrieve_knowledge。

请停止检索。
后续必须使用目前已经获得的证据生成答案。
如果证据不足，请明确说明。
"""
        })

        continue

    # --------------------------------------------------------
    # Execute Tool
    # --------------------------------------------------------

    for tool_call in response.tool_calls:

        tool_name = tool_call["name"]

        result = tool_map[tool_name].invoke(
            tool_call["args"]
        )

        # ====================================================
        # 累积 Context
        # ====================================================

        if result["context"]:
            all_context.append(
                result["context"]
            )

        # ====================================================
        # 累积 Evidence
        # ====================================================

        for evidence in result["evidence"]:           #这部分是一个个从列表中取出单次检索并编号的的证据evident把他们和多次证据all evidence进行比对去重并重新编号

            # 通过 source / sheet / chunk_id 去重
            exists = any(                                     #此处可迭代对象有一个时trueang（）为true，全为flash，ang（）才为flash
                item["source"] == evidence["source"]
                and item["sheet"] == evidence["sheet"]
                and item["chunk_id"] == evidence["chunk_id"]
                for item in all_evidence
            )

            if not exists:

                # 重新生成全局 Evidence ID
                evidence["id"] = (
                    f"E{len(all_evidence) + 1}"
                )

                all_evidence.append(
                    evidence
                )

        # ====================================================
        # ToolMessage
        # ====================================================

        messages.append(
            ToolMessage(
                content=str(result),
                tool_call_id=tool_call["id"]
            )
        )

        retrieval_count += 1


# ============================================================
# 12. Grounding Preparation
# ============================================================

accumulated_context = "\n\n".join(
    all_context
)

evidence_text = "\n".join(
    f"""
Evidence ID: {item["id"]}
source: {item["source"]}
sheet: {item["sheet"]}
chunk_id: {item["chunk_id"]}
"""
    for item in all_evidence
)


# ============================================================
# 13. Answer Grounding
# ============================================================

print("\n" + "=" * 70)
print("Answer Grounding")
print("=" * 70)

grounding_prompt = f"""
你是一个 Answer Grounding Agent。

你的任务是根据知识库 Context 和 Evidence，
生成一个有证据依据的最终答案。


========================
用户问题
========================

{question}


========================
知识库 Context
========================

{accumulated_context}


========================
可用 Evidence
========================

{evidence_text}


========================
任务要求
========================

1. 根据 Context 回答用户问题。

2. answer：
   - 只能使用 Context 中能够支持的事实。
   - 不允许使用知识库之外的信息进行事实补充。
   - 如果现有证据不足以回答问题，
     必须明确说明信息不足。

3. evidence_ids：
   - 必须填写真正支持 answer 的 Evidence ID。
   - 可以填写一个或多个 Evidence ID。
   - 如果一个事实由多个 Evidence 共同支持，
     可以填写多个 ID。
   - 与 answer 无关的 Evidence 不得填写。
   - Evidence ID 必须来自下面提供的 Evidence。

4. 如果无法从现有 Evidence 得到可靠答案：
   - answer 中明确说明无法确认；
   - evidence_ids 填写实际参考过的相关 Evidence，
     如果没有相关 Evidence，则填写 []。


========================
输出格式
========================

必须输出合法 JSON。

JSON 必须严格符合以下结构：

{{
    "answer": "最终答案字符串",
    "evidence_ids": ["E1", "E2"]
}}

字段要求：

answer：
字符串类型。

evidence_ids：
字符串数组，每个元素必须是实际存在的 Evidence ID。

不要输出 Markdown。
不要输出 ```json。
不要输出 JSON 以外的任何文字。
"""

grounded_answer = grounding_llm.invoke(
    grounding_prompt
)


# ============================================================
# 14. Output
# ============================================================

print("\n" + "=" * 70)
print("最终 Grounded Answer")
print("=" * 70)

print(
    f"\nAnswer:\n"
    f"{grounded_answer.answer}"
)

print(
    f"\nEvidence IDs:\n"
    f"{grounded_answer.evidence_ids}"
)


# ============================================================
# 15. Evidence Mapping
# ============================================================

print("\n" + "=" * 70)
print("Evidence Mapping")
print("=" * 70)

for evidence_id in grounded_answer.evidence_ids:           #这部分是通过ground evidence从all evidence中提取出来的部分evidence和all evidence进行比对校验防止llm自己提取evidence的时候改变evidence信息

    matched = [                       #列表内置可迭代对象当if为true时此时的item放入列表否则不放
        item
        for item in all_evidence
        if item["id"] == evidence_id
    ]

    if not matched:
        print(
            f"WARNING: {evidence_id} "
            f"不存在！"
        )
        continue

    item = matched[0]

    print(
        f"{evidence_id} -> "
        f"source={item['source']}, "
        f"sheet={item['sheet']}, "
        f"chunk_id={item['chunk_id']}"
    )