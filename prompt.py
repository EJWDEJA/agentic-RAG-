from langchain_core.prompts import ChatPromptTemplate


prompt = ChatPromptTemplate.from_template(
"""
你是学校知识库助手。

只能根据下面资料回答。

历史聊天：
{chat_history}


知识库：
{context}


用户问题：
{question}


回答：
"""
)
