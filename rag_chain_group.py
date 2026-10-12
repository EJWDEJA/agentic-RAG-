from retrieval_chain_group import retrieval_chain     #检索结果基于group上下文管理
from prompt import prompt

from langchain_ollama import ChatOllama
from langchain_core.runnables.history import RunnableWithMessageHistory

from memory import get_session_history
from rewritechain import rewrite_question
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables import RunnableParallel




# 大模型
llm = ChatOllama(
    model="qwen2.5:1.5b"
)



def get_retrieval_result(x):     #返回retrival返回的字典信息包括文本和metadata

    rewritten_question = rewrite_question(
        x["question"],
        x.get("chat_history", [])
    )

    return retrieval_chain.invoke(
        rewritten_question
    )




def get_prompt_data(x):       # 提取context,question,chat_history

    result = get_retrieval_result(x)


    return {

        "context":      #定义prompt里面的｛chontext｝
        result["context"],


        "question":         #定义prompt里面的｛queation｝
        x["question"],


        "chat_history":         #定义prompt里面的{chat history}
        x.get(
            "chat_history",
            []                #通过git（ "chat_history",[] ）提取的历史不存在时会返回【】而直接【"chat_history"】提取的历史不存在时会报错不安全
        )


    }



# =====================
# 回答链
# =====================

answer_chain = (          #创建一个专门llm回答的管道

    RunnableLambda(
        get_prompt_data       #把Runnablelambda模块化方便用管道符号链接
    )

    |

    prompt

    |

    llm

)


# 基础RAG Chain
# 注意：这里还没有Memory

base_rag_chain = RunnableParallel(

    answer=
    answer_chain,


    context=       #用于进行LLM回答faithfulness临时
    RunnableLambda(
        lambda x:
        get_retrieval_result(x)["context"]
    ),




    sources=
    RunnableLambda(

        lambda x:

        get_retrieval_result(x)["sources"]     #元数据

    )

)





# 加入Memory
# 这里才是真正使用memory.py

rag_chain = RunnableWithMessageHistory(

    base_rag_chain,

    get_session_history,

    input_messages_key="question",

    history_messages_key="chat_history",

    output_messages_key="answer"
)
