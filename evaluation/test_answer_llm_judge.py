import sys       
import os

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)
import os       #这是针对group(类)的检索处理上下文管理

# 必须在导入 transformers / sentence_transformers 之前设置！
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"









from answer_test_dataset import golden_answer_dataset

from rag_chain_group import rag_chain

from langchain_ollama import ChatOllama

import re


judge_llm = ChatOllama(
    model="qwen2.5:1.5b"
)


def judge_answer(
    question,
    reference_answer,
    answer
):

    prompt = f"""
你是一名 RAG 系统答案评审员。

请根据“问题”和“标准答案”，判断“MyRAG回答”是否正确。

评分标准：

0 = 错误
答案的核心事实错误，或者与标准答案不符。

1 = 部分正确
答案包含部分正确的信息，但存在遗漏，或者只回答了部分问题。

2 = 完全正确
答案的核心事实正确，并且完整回答了问题。
允许使用与标准答案不同的表述方式。

问题：
{question}

标准答案：
{reference_answer}

MyRAG回答：
{answer}

请只输出一个数字：
0、1 或 2

不要输出其他内容。
"""

    result = judge_llm.invoke(prompt)

    judge_result = result.content.strip()

    match = re.search(r"[012]", judge_result)

    if match:

        return int(match.group())

    return None


total_score = 0

valid_count = 0


for i, item in enumerate(golden_answer_dataset):

    question = item["question"]

    reference_answer = item["reference_answer"]


    print("\n")
    print("=" * 100)

    print("问题：", question)

    print("标准答案：", reference_answer)

    print("=" * 100)


    result = rag_chain.invoke(
        {
            "question": question
        },
        config={
            "configurable": {
                "session_id": f"judge_eval_{i}"
            }
        }
    )


    answer = result["answer"].content


    print("\n========== MyRAG回答 ==========\n")

    print(answer)


    score = judge_answer(
        question,
        reference_answer,
        answer
    )


    print("\n========== LLM Judge ==========\n")

    print("Judge评分：", score)


    if score is not None:

        total_score += score

        valid_count += 1


if valid_count > 0:

    average_score = total_score / valid_count

else:

    average_score = 0


print("\n")
print("=" * 100)

print("Answer Correctness —— LLM Judge")

print("=" * 100)

print("总题数：", len(golden_answer_dataset))

print("有效评分数：", valid_count)

print("总得分：", total_score)

print("平均得分：", round(average_score, 4))

print("满分：2")

print(
    "归一化得分：",
    round(average_score / 2, 4)
)