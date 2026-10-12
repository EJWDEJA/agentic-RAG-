import os                                ####这个文件是用来计算embedding_accuracy":"llm_accuracy":"agreement / total的他们的分母都是case数，还用来查找语义相似度评判，大模型评判意见分歧（一个个item找）
import json                             ####也就是给20.6的results加几个评估agent答案评估系统的字段
                                        #也就是加入评估答案评估器的评估器


# ============================================================
# 1. Environment
# ============================================================

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


# ============================================================
# 2. Imports
# ============================================================

from langchain_openai import ChatOpenAI

from agentic_rag_20_3_dataset import (
    TEST_CASES
)

from agentic_rag_20_6_calibration import (
    collect_agent_results,
    GOLD_LABELS
)


# ============================================================
# 3. LLM
# ============================================================

llm = ChatOpenAI(
    model="deepseek-v4-flash",
    api_key=os.environ[
        "ANTHROPIC_AUTH_TOKEN"
    ],
    base_url="https://api.deepseek.com"
)


# ============================================================
# 4. LLM Judge Prompt
# ============================================================

JUDGE_PROMPT = """
你是一个严格的答案正确性评估器。

你的任务是判断“实际答案”是否正确回答了用户问题。

判断标准：

1. 实际答案必须回答问题。
2. 核心事实必须正确。
3. 如果实际答案只是表达方式不同，但事实与期望答案一致，
   应判断为正确。
4. 如果实际答案只包含部分正确内容，但遗漏了问题要求的
   关键内容，应判断为错误。
5. 如果实际答案与期望答案事实冲突，应判断为错误。
6. 不要求实际答案与期望答案文字完全一致。
7. 不要因为措辞不同而判错。
8. 只能根据 Question、Expected Answer 和 Actual Answer 判断。

必须只输出合法 JSON。

输出格式：

{{
    "correct": true,
    "reason": "简短说明判断理由"
}}

其中：

correct 必须是 true 或 false。

不要输出 Markdown。
不要输出 ```json。
不要增加其他字段。

Question:
{question}

Expected Answer:
{expected_answer}

Actual Answer:
{actual_answer}
"""


# ============================================================
# 5. LLM Judge
# ============================================================

def llm_judge(
    question,
    expected_answer,
    actual_answer
):

    prompt = JUDGE_PROMPT.format(       #把参数塞入JUDGE_PROMPT对应的占位符组成一个完整提示词
        question=question,
        expected_answer=expected_answer,
        actual_answer=actual_answer
    )


    response = llm.invoke(
        prompt
    )


    content = response.content.strip()  #####本质字符串


    try:

        result = json.loads(    ##把json字符串转化成python内置字典
            content
        )

    except json.JSONDecodeError:                    ##这里的捕获异常又抛出异常是为了先捕获异常，然后重新定义一个更加符合业务逻辑的异常返回值，方便排查具体什么地方错了

        raise ValueError(
            "\nLLM Judge 返回的不是合法 JSON：\n"
            f"{content}"                               #################核心
        )


    if "correct" not in result:

        raise ValueError(
            "LLM Judge 缺少 correct 字段"
        )


    if not isinstance(
        result["correct"],
        bool
    ):

        raise ValueError(
            "correct 必须是 bool"
        )


    return result


# ============================================================
# 6. Run LLM Judge
# ============================================================

def run_llm_judge(
    agent_results
):

    results = []


    print(
        "\n"
        + "=" * 70
    )

    print(
        "STEP 1: LLM JUDGE"
    )

    print(
        "=" * 70
    )


    for item in agent_results:

        print(
            f"\nEvaluating "
            f"{item['id']}"
        )


        judge_result = llm_judge(

            question=item[
                "question"
            ],

            expected_answer=item[
                "expected_answer"
            ],

            actual_answer=item[
                "actual_answer"
            ]

        )


        item = {
            **item,

            "llm_judge":
                judge_result[
                    "correct"
                ],

            "llm_reason":
                judge_result[
                    "reason"
                ]
        }


        results.append(
            item
        )


        print(
            "LLM Judge:",
            judge_result[
                "correct"
            ]
        )


        print(
            "Reason:",
            judge_result[
                "reason"
            ]
        )


    return results


# ============================================================
# 7. Embedding Judge
# ============================================================

def embedding_prediction(
    similarity,
    threshold
):

    return (
        similarity >= threshold
    )


# ============================================================
# 8. Compare Evaluators
# ============================================================

def compare_evaluators(
    results,
    threshold
):

    embedding_correct = 0
    llm_correct = 0
    agreement = 0

    total = len(
        results
    )


    print(
        "\n"
        + "=" * 70
    )

    print(
        f"STEP 2: EVALUATOR COMPARISON"
    )

    print(
        f"Threshold = {threshold}"
    )

    print(
        "=" * 70
    )


    for item in results:

        gold = item[
            "gold_label"
        ]


        embedding_result = (
            item[
                "similarity"
            ]
            >= threshold
        )


        llm_result = item[
            "llm_judge"
        ]


        if (
            embedding_result
            == gold
        ):

            embedding_correct += 1


        if (
            llm_result
            == gold
        ):

            llm_correct += 1


        if (
            embedding_result
            == llm_result
        ):

            agreement += 1


        print(
            f"\n{item['id']}"
        )


        print(
            f"Gold: "
            f"{gold}"
        )


        print(
            f"Embedding: "
            f"{embedding_result}"
        )


        print(
            f"Similarity: "
            f"{item['similarity']:.4f}"
        )


        print(
            f"LLM Judge: "
            f"{llm_result}"
        )


        print(
            f"LLM Reason: "
            f"{item['llm_reason']}"
        )


    print(
        "\n"
        + "-" * 70
    )


    print(
        f"Embedding Accuracy: "
        f"{embedding_correct / total:.2%}"      #这个保留多少位是给比尔看的
    )


    print(
        f"LLM Judge Accuracy: "
        f"{llm_correct / total:.2%}"
    )


    print(
        f"Embedding / LLM Agreement: "
        f"{agreement / total:.2%}"
    )


    return {

        "threshold":
            threshold,

        "embedding_accuracy":
            embedding_correct / total,             #这个没有默认活着要求保留大多数位一说

        "llm_accuracy":
            llm_correct / total,

        "agreement":
            agreement / total

    }


# ============================================================
# 9. Disagreement Analysis
# ============================================================

def analyze_disagreement(
    results,
    threshold
):

    print(
        "\n"
        + "=" * 70
    )

    print(
        "STEP 3: DISAGREEMENT ANALYSIS"
    )

    print(
        "=" * 70
    )


    found = False


    for item in results:

        embedding_result = (
            item[
                "similarity"
            ]
            >= threshold
        )


        llm_result = item[
            "llm_judge"
        ]


        if (
            embedding_result
            != llm_result
        ):

            found = True


            print(
                f"\n{'=' * 50}"
            )


            print(
                f"ID: "
                f"{item['id']}"
            )


            print(
                f"\nQuestion:\n"
                f"{item['question']}"
            )


            print(
                f"\nExpected:\n"
                f"{item['expected_answer']}"
            )


            print(
                f"\nActual:\n"
                f"{item['actual_answer']}"
            )


            print(
                f"\nSimilarity: "
                f"{item['similarity']:.4f}"
            )


            print(
                f"\nEmbedding Judge: "
                f"{embedding_result}"
            )


            print(
                f"LLM Judge: "
                f"{llm_result}"
            )


            print(
                f"\nLLM Reason:\n"
                f"{item['llm_reason']}"
            )


            print(
                f"\nGold Label: "
                f"{item['gold_label']}"
            )


    if not found:

        print(
            "\n当前没有发现 "
            "Embedding / LLM Judge 分歧。"
        )


# ============================================================
# 10. Save Results
# ============================================================

def save_results(
    results,
    comparison
):

    output = {

        "results":
            results,

        "comparison":
            comparison

    }


    with open(
        "agentic_rag_20_7_llm_judge_comparison.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            output,      ###待存储信息
            f,           #已打开文件（容器）
            ensure_ascii=False,         ###（文件内汉字显示汉字而不是unicode转义的编号）
            indent=4
        )


    print(
        "\n"
        "Results saved to:"
    )

    print(
        "agentic_rag_20_7_llm_judge_comparison.json"
    )


# ============================================================
# 11. Main
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Threshold
    # --------------------------------------------------------

    THRESHOLD = 0.80


    # --------------------------------------------------------
    # Agent Results
    # --------------------------------------------------------

    agent_results = (
        collect_agent_results()                  ####主要用来获取并输出实际值和相似度
    )


    # --------------------------------------------------------
    # LLM Judge
    # --------------------------------------------------------

    results = run_llm_judge(                  ##主要用来获取并输出llm判断结果和这个结果的原因
        agent_results
    )


    # --------------------------------------------------------
    # Compare
    # --------------------------------------------------------

    comparison = compare_evaluators(             #主要用来获取并输出一定阈值下"embedding_accuracy":"llm_accuracy":"llm_accuracy":

        results,

        threshold=THRESHOLD

    )


    # --------------------------------------------------------
    # Disagreement
    # --------------------------------------------------------

    analyze_disagreement(               #主要用来获取并输出一定阈值下         语义相似度评判，大模型评判有无意见分歧   

        results,

        threshold=THRESHOLD

    )


    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_results(

        results,

        comparison

    )