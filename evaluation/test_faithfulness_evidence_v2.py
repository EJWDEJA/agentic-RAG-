import sys             #这里忠实的评估用到的方法是把答案拆成多个不用意思的说法claim，组成列表claims。把凭借好的context（doc.page拼接）按照规则切分成块组成列表，然后用for循环加enumerate依次编号同时拼接，最后用foe循环一个claim对应我们拼接好的编完号的context输出是否有上下文支撑有的话编号是多少
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


# ============================================================
# LLM
# ============================================================

llm = ChatOllama(
    model="qwen2.5:1.5b"
)


# ============================================================
# 1. 从 Answer 中提取 Claims
# ============================================================

def extract_claims(answer):

    prompt = f"""
你是一名 RAG Evaluation 评审员。

请把下面的 Answer 拆分成若干个独立的事实 Claim。

要求：

1. 每个 Claim 必须是一个独立事实。
2. 不要添加 Answer 中不存在的信息。
3. 不要进行知识补充。
4. 如果 Answer 只有一个事实，就只输出一个 Claim。
5. 每行输出一个 Claim。
6. 不要输出编号。
7. 不要输出任何解释。

Answer：
{answer}

请只输出 Claims：
"""

    result = llm.invoke(prompt)

    claims = []

    for line in result.content.splitlines():

        line = line.strip()

        if not line:
            continue

        if line[0].isdigit() and "." in line:
            line = line.split(".", 1)[1].strip()

        claims.append(line)

    return claims


# ============================================================
# 2. 把 Context 切成句子并编号
# ============================================================

def split_context_into_sentences(context):

    sentences = re.split(
        r"(?<=[。！？])|(?<=[.!?])(?=\s|$)|\n+",
        context
    )

    result = []

    for sentence in sentences:

        sentence = sentence.strip()

        if sentence:
            result.append(sentence)

    return result


# ============================================================
# 3. 让 LLM 判断 Claim，并选择 Evidence 编号
# ============================================================

def judge_claim_and_select_evidence(
    claim,
    context_sentences
):

    numbered_context = ""

    for i, sentence in enumerate(
        context_sentences,
        start=1
    ):

        numbered_context += (
            f"[{i}] {sentence}\n"
        )


    prompt = f"""
你是一名 RAG Faithfulness 评审员。

请判断 Claim 是否能够被下面的 Context 明确支持。

只能根据 Context 判断。

如果 Context 明确支持 Claim：

输出：

1|||证据编号

如果 Context 不支持 Claim：

输出：

0|||0

重要要求：

1. Evidence 必须来自 Context。
2. 不要自己改写 Evidence。
3. 只需要返回能够支持 Claim 的 Context 编号。
4. 如果有多个证据，只选择最直接的一条。
5. 如果没有明确证据，输出 0|||0。
6. 只能输出一行。
7. 必须严格使用 ||| 作为分隔符。

Context：

{numbered_context}

Claim：

{claim}

请严格按照格式输出：

1|||数字

或者：

0|||0
"""

    result = llm.invoke(prompt)

    return result.content.strip()


# ============================================================
# 4. 解析 Judge 结果
# ============================================================

def parse_judge_result(result):

    parts = result.split(
        "|||",
        1
    )

    if len(parts) != 2:

        return 0, 0


    supported_text = parts[0].strip()

    evidence_text = parts[1].strip()


    if supported_text == "1":

        supported = 1

    else:

        supported = 0


    # 从 Evidence 部分提取数字
    match = re.search(
        r"\d+",
        evidence_text
    )


    if match:

        evidence_index = int(
            match.group()
        )

    else:

        evidence_index = 0


    return supported, evidence_index


# ============================================================
# 5. 验证 Evidence 编号
# ============================================================

def validate_evidence_index(
    evidence_index,
    context_sentences
):

    if evidence_index == 0:

        return True

    return (
        1
        <= evidence_index
        <= len(context_sentences)
    )


# ============================================================
# 6. 主测试
# ============================================================

total_supported_claims = 0

total_claims = 0

total_valid_evidence = 0

total_evidence_count = 0


for index, item in enumerate(
    golden_answer_dataset,
    start=1
):

    question = item["question"]

    reference_answer = item["reference_answer"]


    print("\n")

    print("=" * 100)

    print(
        f"测试问题 {index}"
    )

    print("=" * 100)


    print("Question：")

    print(question)


    print("\nReference Answer：")

    print(reference_answer)


    # ========================================================
    # 1. 调用 RAG
    # ========================================================

    result = rag_chain.invoke(
        {
            "question": question
        },
        config={
            "configurable": {
                "session_id":
                f"faithfulness_evidence_v2_{index}"
            }
        }
    )


    # ========================================================
    # 2. 获取 Answer
    # ========================================================

    answer = result["answer"].content


    # ========================================================
    # 3. 获取 Context
    # ========================================================

    context = result["context"]


    print("\nActual Answer：")

    print(answer)


    # ========================================================
    # 4. Context 切句
    # ========================================================

    context_sentences = (
        split_context_into_sentences(
            context
        )
    )


    print(
        "\n========== Context Evidence Candidates ==========\n"
    )


    for i, sentence in enumerate(
        context_sentences,
        start=1
    ):

        print(
            f"[{i}] {sentence}"
        )


    # ========================================================
    # 5. Claim Extraction
    # ========================================================

    claims = extract_claims(
        answer
    )


    print(
        "\n========== Claims ==========\n"
    )


    # 当前问题统计

    question_supported_claims = 0

    question_valid_evidence = 0

    question_evidence_count = 0


    # ========================================================
    # 6. 逐个 Claim 判断
    # ========================================================

    for claim_index, claim in enumerate(
        claims,
        start=1
    ):

        print(
            f"Claim {claim_index}："
        )

        print(claim)


        # ====================================================
        # Judge
        # ====================================================

        judge_result = (
            judge_claim_and_select_evidence(
                claim,
                context_sentences
            )
        )


        # ====================================================
        # 解析结果
        # ====================================================

        supported, evidence_index = (
            parse_judge_result(
                judge_result
            )
        )


        # ====================================================
        # Evidence 编号验证
        # ====================================================

        evidence_valid = (
            validate_evidence_index(
                evidence_index,
                context_sentences
            )
        )


        # ====================================================
        # 获取真正 Evidence
        # ====================================================

        if (
            evidence_index > 0
            and evidence_valid
        ):

            evidence = (
                context_sentences[
                    evidence_index - 1
                ]
            )

        else:

            evidence = "无"


        # ====================================================
        # 输出
        # ====================================================

        print(
            "\nJudge Result："
        )

        print(judge_result)


        print(
            "\nSupported："
        )

        print(supported)


        print(
            "\nEvidence Index："
        )

        print(evidence_index)


        print(
            "\nEvidence："
        )

        print(evidence)


        print(
            "\nEvidence Valid："
        )

        print(evidence_valid)


        print(
            "\n"
            + "-" * 80
        )


        # ====================================================
        # 当前问题统计
        # ====================================================

        if supported == 1:

            question_supported_claims += 1


        if evidence_index > 0:

            question_evidence_count += 1


            if evidence_valid:

                question_valid_evidence += 1


        # ====================================================
        # 总体统计
        # ====================================================

        total_supported_claims += (
            supported
        )

        total_claims += 1


        if evidence_index > 0:

            total_evidence_count += 1


            if evidence_valid:

                total_valid_evidence += 1


    # ========================================================
    # 7. 当前问题 Faithfulness
    # ========================================================

    if len(claims) > 0:

        question_faithfulness = (
            question_supported_claims
            / len(claims)
        )

    else:

        question_faithfulness = 0


    # ========================================================
    # 8. 当前问题 Evidence Validity
    # ========================================================

    if question_evidence_count > 0:

        question_evidence_validity = (
            question_valid_evidence
            / question_evidence_count
        )

    else:

        question_evidence_validity = 0


    # ========================================================
    # 9. 当前问题结果
    # ========================================================

    print("\n")

    print("=" * 100)

    print("当前问题结果")

    print("=" * 100)


    print(
        "Claim 数量：",
        len(claims)
    )


    print(
        "被 Context 支持的 Claim 数量：",
        question_supported_claims
    )


    print(
        "Faithfulness：",
        round(
            question_faithfulness,
            4
        )
    )


    print(
        "Evidence 数量：",
        question_evidence_count
    )


    print(
        "有效 Evidence 数量：",
        question_valid_evidence
    )


    print(
        "Evidence Validity：",
        round(
            question_evidence_validity,
            4
        )
    )


# ============================================================
# 10. 总体 Faithfulness
# ============================================================

if total_claims > 0:

    overall_faithfulness = (
        total_supported_claims
        / total_claims
    )

else:

    overall_faithfulness = 0


# ============================================================
# 11. 总体 Evidence Validity
# ============================================================

if total_evidence_count > 0:

    evidence_validity = (
        total_valid_evidence
        / total_evidence_count
    )

else:

    evidence_validity = 0


# ============================================================
# 12. 最终结果
# ============================================================

print("\n")

print("=" * 100)

print(
    "Evidence Verification 总体结果"
)

print("=" * 100)


print(
    "总 Claim 数量：",
    total_claims
)


print(
    "被 Context 支持的 Claim 数量：",
    total_supported_claims
)


print(
    "Overall Faithfulness：",
    round(
        overall_faithfulness,
        4
    )
)


print(
    "Evidence 数量：",
    total_evidence_count
)


print(
    "有效 Evidence 数量：",
    total_valid_evidence
)


print(
    "Evidence Validity：",
    round(
        evidence_validity,
        4
    )
)
