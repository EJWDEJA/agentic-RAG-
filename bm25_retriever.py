import pickle

from langchain_community.retrievers import BM25Retriever



# 加载Document

with open(
    "documents.pkl",    #打开的文件名
    "rb"       #这里的rb全称read binary二级制读取模式
) as f:        #这里的f是对打开了的文件的命名名字随意，核心是方便后续调

    docs = pickle.load(f)     #反序列化加载data



# 创建BM25

bm25_retriever = BM25Retriever.from_documents(
    docs
)


# 返回数量

bm25_retriever.k = 5       
