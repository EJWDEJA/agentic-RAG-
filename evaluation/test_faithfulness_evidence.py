
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


# ============================================================
# LLM：用于 Claim 提取 + Evidence Faithfulness 判断
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

        # 去除可能出现的编号
        if line[0].isdigit() and "." in line:      #看line字符串第1个字符看是否是数字并且字符串有  .  
            line = line.split(".", 1)[1].strip()   #有  .  的话以   .   切只切一次  切完以后去除字符串前后空白

        claims.append(line)

    return claims


# ============================================================
# 2. 判断 Claim 是否被 Context 支持，并提取 Evidence
# ============================================================

def judge_claim_with_evidence(context, claim):

    prompt = f"""
你是一名 RAG Faithfulness 评审员。

请判断下面这个事实是否能够被 Context 明确支持。

只能根据 Context 判断。

如果 Context 明确支持这个事实：

输出：

1|||Context 中支持这个事实的原文

如果 Context 不支持这个事实：

输出：

0|||无

重要要求：

1. Evidence 必须直接来自 Context。
2. 不允许自己改写 Evidence。
3. 不允许自己补充 Context 中不存在的信息。
4. 如果找不到明确证据，必须输出 0|||无。
5. 只能输出一行。
6. 必须严格使用 ||| 作为分隔符。

Context：
{context}

Claim：
{claim}

请严格按照下面格式输出：

1|||Evidence

或者：

0|||无
"""

    result = llm.invoke(prompt)

    return result.content.strip()


# ============================================================
# 3. 解析 Judge 返回结果
# ============================================================

def parse_judge_result(result):

    parts = result.split("|||", 1)   #按三蜀切只切一次

    if len(parts) != 2:
        return 0, "无"

    supported_text = parts[0].strip()    #取列表第一个元素并去除元素首尾空白符号
    evidence = parts[1].strip()    #取列表第二个元素并去除首尾空白符号

    if supported_text == "1":
        supported = 1
    else:
        supported = 0

    return supported, evidence


# ============================================================
# 4. 验证 Evidence 是否真的存在于 Context
# ============================================================

def validate_evidence(context, evidence):

    if evidence == "无":
        return True

    return evidence in context


# ============================================================
# 5. 主测试
# ============================================================

# 总体统计
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

    print(f"测试问题 {index}")

    print("=" * 100)

    print("Question：")
    print(question)

    print("\nReference Answer：")
    print(reference_answer)


    # ========================================================
    # 1. 调用当前 RAG
    # ========================================================

    result = rag_chain.invoke(
        {
            "question": question
        },
        config={
            "configurable": {
                "session_id": f"faithfulness_evidence_{index}"
            }
        }
    )


    # ========================================================
    # 2. 获取 Answer
    # ========================================================

    answer = result["answer"].content


    # ========================================================
    # 3. 获取真正的 Context
    # ========================================================

    context = result["context"]


    print("\nActual Answer：")
    print(answer)


    print("\n========== Context ==========\n")
    print(context)


    # ========================================================
    # 4. Claim Extraction
    # ========================================================

    claims = extract_claims(answer)


    print("\n========== Claims ==========\n")


    # 当前问题统计
    question_supported_claims = 0

    question_valid_evidence = 0

    question_evidence_count = 0


    # ========================================================
    # 5. 逐个 Claim 进行 Faithfulness + Evidence 判断
    # ========================================================

    for claim_index, claim in enumerate(
        claims,
        start=1
    ):

        print(f"Claim {claim_index}：")
        print(claim)


        # ====================================================
        # Judge   #有for非列表数据
        # ====================================================

        judge_result = judge_claim_with_evidence(
            context,
            claim
        )


        # ====================================================
        # 解析 Judge 结果
        # ====================================================

        supported, evidence = parse_judge_result(
            judge_result
        )


        # ====================================================
        # 验证 Evidence
        # ====================================================

        evidence_valid = validate_evidence(
            context,
            evidence
        )


        print("\nJudge Result：")
        print(judge_result)


        print("\nSupported：")
        print(supported)


        print("\nEvidence：")
        print(evidence)


        print("\nEvidence Valid：")
        print(evidence_valid)


        print("\n" + "-" * 80)
# ====================================================
        # 当前问题统计
        # ====================================================

        if supported == 1:

            question_supported_claims += 1


        if evidence != "无":

            question_evidence_count += 1

            if evidence_valid:

                question_valid_evidence += 1


        # ====================================================
        # 总体统计
        # ====================================================

        total_supported_claims += supported

        total_claims += 1


        if evidence != "无":

            total_evidence_count += 1

            if evidence_valid:

                total_valid_evidence += 1


    # ========================================================
    # 6. 当前问题 Faithfulness
    # ========================================================

    if len(claims) > 0:

        question_faithfulness = (
            question_supported_claims
            / len(claims)
        )

    else:

        question_faithfulness = 0


    # ========================================================
    # 7. 当前问题 Evidence Validity
    # ========================================================

    if question_evidence_count > 0:

        question_evidence_validity = (
            question_valid_evidence
            / question_evidence_count
        )

    else:

        question_evidence_validity = 0


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
        round(question_faithfulness, 4)
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
        round(question_evidence_validity, 4)
    )

# ============================================================
# 8. 总体 Faithfulness
# ============================================================

if total_claims > 0:

    overall_faithfulness = (
        total_supported_claims
        / total_claims
    )

else:

    overall_faithfulness = 0


# ============================================================
# 9. 总体 Evidence Validity
# ============================================================

if total_evidence_count > 0:

    evidence_validity = (
        total_valid_evidence
        / total_evidence_count
    )

else:

    evidence_validity = 0


# ============================================================
# 10. 输出最终结果
# ============================================================

print("\n")
print("=" * 100)

print("Evidence-based Faithfulness 总体结果")

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
    round(overall_faithfulness, 4)
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
    round(evidence_validity, 4)
)