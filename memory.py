from langchain_core.chat_history import InMemoryChatMessageHistory    #这个函数是用来以列表的形式存储以后聊天记录的


store = {}    #建立一个键值对字典，键是以后ID，值是用户AI输入


def get_session_history(session_id):

    if session_id not in store:    #如果ID不存在建一个键值对存这个用户的记忆

        store[session_id] = InMemoryChatMessageHistory() #字典里面A键赋值inmermory 确定记忆在store里的储存地址的，InMemoryChatMessageHistory里面是一个列表，这个里面的信息靠runnablewithmemoryhistory存

    return store[session_id]   #返回inmemory



    
def show_memory(session_id):

    history = store.get(session_id)

    if history:

        print("\n===== 当前历史 =====")

        for message in history.messages:

            print(
                type(message).__name__,
                ":",
                message.content
            )

        print("====================")
