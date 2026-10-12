
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


def judge_faithfulness(
    question,
    context,
    answer
):

    prompt = f"""
你是一名 RAG 系统 Faithfulness 评审员。

请判断 MyRAG 的回答是否完全基于提供的 Context。

判断标准：

2 = 完全忠实
回答中的事实都能够被 Context 支持，没有明显的额外事实。

1 = 部分忠实
回答中的部分事实能够被 Context 支持，但存在部分无法从 Context 得到的信息。

0 = 不忠实
回答包含明显与 Context 不一致的事实，或者主要内容无法从 Context 得到支持。

注意：
不要根据你自己的知识判断。
只能根据提供的 Context 判断。
允许答案与 Context 使用不同的表述方式。

问题：
{question}

Context：
{context}

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

    print("\n")
    print("=" * 100)

    print("问题：", question)

    print("=" * 100)


    result = rag_chain.invoke(
        {
            "question": question
        },
        config={
            "configurable": {
                "session_id": f"faithfulness_eval_{i}"
            }
        }
    )


    context = result["context"]

    answer = result["answer"].content


    print("\n========== Context ==========\n")

    print(context)


    print("\n========== MyRAG回答 ==========\n")

    print(answer)


    score = judge_faithfulness(
        question,
        context,
        answer
    )


    print("\n========== Faithfulness ==========\n")

    print("Faithfulness评分：", score)


    if score is not None:

        total_score += score

        valid_count += 1


if valid_count > 0:

    average_score = total_score / valid_count

else:

    average_score = 0


print("\n")
print("=" * 100)

print("Faithfulness 总体结果")

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
