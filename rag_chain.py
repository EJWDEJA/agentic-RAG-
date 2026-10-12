from retrieval_chain import retrieval_chain     #检索结果基于doc  chunk块  上下文管理
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


    sources=
    RunnableLambda(

        lambda x:

        get_retrieval_result(x)["sources"]     #文本加元数据

    )

)





# 加入Memory
# 这里才是真正使用memory.py

rag_chain = RunnableWithMessageHistory(      #这里runnal...负责快速给base rag history 加上记忆模块。而且他会把负责存储question，和ai回复的函数隐藏内置到langchain的源码里面实现填参立刻使用的低代码工作流开发

    base_rag_chain,      #负责指定给谁加记忆模块

    get_session_history,      #负责从inmeretory...的message=[]里面拿数据

    input_messages_key="question",

    history_messages_key="chat_history",

    output_messages_key="answer"    #黄色字体是是两个固定的参数名字h由于定义把督导的信息放在那个变量里面，input...负责定义当前问题变量

)

