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


def extract_claims(answer):

    prompt = f"""
请把下面的回答拆分成若干个独立的事实。

要求：

1. 每一条只表达一个事实。
2. 不要添加原回答没有的信息。
3. 不要解释。
4. 每行输出一个事实。
5. 如果回答只有一个事实，就只输出一条。

回答：
{answer}

请直接输出事实列表。     #这里是列状数据不是，python列表
"""

    result = judge_llm.invoke(prompt)

    text = result.content.strip()  #去除字符串首尾空白字符

    claims = []

    for line in text.splitlines():   #splitline是处理字符串按换行符切割如回车\n并返回列表

        line = line.strip()  #去除各取出后元素的首尾空白字符

        line = re.sub(    #re内置函数替换
            r"^[\d\-\*\.\、\s]+",   #去除字符串内部的标号，点，空白字符
            "",
            line
        )

        if line:
            claims.append(line)

    return claims


def judge_claim(context, claim):

    prompt = f"""
你是一名 RAG Faithfulness 评审员。

请判断下面这个事实是否能够被 Context 明确支持。

只能根据 Context 判断。

如果 Context 明确支持这个事实：
输出 1

如果 Context 不支持这个事实：
输出 0

不要根据自己的知识进行补充。

Context：
{context}

Claim：
{claim}

请只输出：
0 或 1
"""

    result = judge_llm.invoke(prompt)

    text = result.content.strip()

    match = re.search(r"[01]", text)     #re内置函数，搜索

    if match:
        return int(match.group())

    return None


total_supported = 0
total_claims = 0


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
                "session_id": f"claim_faithfulness_{i}"
            }
        }
    )


    context = result["context"]

    answer = result["answer"].content


    print("\n========== MyRAG回答 ==========\n")

    print(answer)


    print("\n========== Claims ==========\n")


    claims = extract_claims(answer)


    supported_count = 0


    for index, claim in enumerate(claims, start=1):

        score = judge_claim(
            context,
            claim
        )


        if score == 1:

            label = " Context支持"

            supported_count += 1

        elif score == 0:

            label = " Context不支持"

        else:

            label = " 无法判断"


        print(
            f"Claim {index}：{claim}"
        )

        print(
            "判断：",
            label
        )

        print("-" * 80)


    claim_count = len(claims)


    if claim_count > 0:

        faithfulness = (
            supported_count / claim_count
        )

    else:

        faithfulness = 0


    print("\n========== 本题结果 ==========\n")

    print(
        "支持 Claim：",
        supported_count
    )

    print(
        "Claim 总数：",
        claim_count
    )

    print(
        "Faithfulness：",
        round(faithfulness, 4)
    )


    total_supported += supported_count

    total_claims += claim_count


print("\n")
print("=" * 100)

print("Claim-level Faithfulness 总体结果")

print("=" * 100)

print(
    "支持 Claim 总数：",
    total_supported
)

print(
    "Claim 总数：",
    total_claims
)


if total_claims > 0:

    overall_faithfulness = (
        total_supported / total_claims
    )

else:

    overall_faithfulness = 0


print(
    "总体 Faithfulness：",
    round(overall_faithfulness, 4)
)
