from langchain_ollama import ChatOllama


# =========================
# 1. Tool
# =========================

def get_weather(city):

    return f"{city}今天晴天，25度"


# =========================
# 2. Tool Registry
# =========================

tools = {
    "get_weather": get_weather
}


# =========================
# 3. LLM
# =========================

llm = ChatOllama(
    model="qwen2.5:1.5b"
)


# =========================
# 4. 用户问题
# =========================

question = input("请输入问题：")


# =========================
# 5. 初始 Prompt
# =========================

prompt = f"""
你是一个 ReAct Agent。

用户问题：

{question}

你可以使用：

get_weather(city)

如果需要查询天气，请严格按照下面格式：

THOUGHT:你的判断
ACTION:get_weather
INPUT:城市

如果已经可以回答，请严格输出：

FINAL:最终答案

不要输出其他格式。
"""


# =========================
# 6. ReAct Loop
# =========================

while True:

    response = llm.invoke(prompt)

    output = response.content.strip()

    print("\n===== LLM 输出 =====")
    print(output)


    # =========================
    # 7. Final
    # =========================

    if output.startswith("FINAL:"):

        answer = output.split(":", 1)[1].strip()

        print("\n===== 最终答案 =====")
        print(answer)

        break


    # =========================
    # 8. Action
    # =========================

    if "ACTION:" in output:

        lines = output.splitlines()

        action = None
        tool_input = None

        for line in lines:

            if line.startswith("ACTION:"):

                action = line.split(":", 1)[1].strip()


            elif line.startswith("INPUT:"):

                tool_input = line.split(":", 1)[1].strip()


        print("\n===== Action =====")
        print("Tool：", action)
        print("Input：", tool_input)


        # =========================
        # 9. 找到 Tool
        # =========================

        tool = tools.get(action)


        if tool is None:

            observation = "错误：不存在这个 Tool。"

        else:

            observation = tool(tool_input)


        # =========================
        # 10. Observation
        # =========================

        print("\n===== Observation =====")
        print(observation)


        # =========================
        # 11. 把 Observation 返回给 LLM
        # =========================

        prompt = f"""
你是一个 ReAct Agent。

用户问题：

{question}

你之前采取了行动：

ACTION:{action}

INPUT:{tool_input}

工具返回：

OBSERVATION:{observation}

现在根据 Observation 继续处理问题。

如果已经可以回答：

FINAL:最终答案

如果还需要工具：

THOUGHT:你的判断
ACTION:get_weather
INPUT:城市

不要输出其他格式。"""
